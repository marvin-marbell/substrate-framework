"""Fresh native sources/eigenstates/propagation; no producer CLI or evolved input."""
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_science import load
from rich_native_body_review_20261016_math import FieldHamiltonian,metric_roots,data_from_labels,peierls,evolve,readout,count_distribution,total_variation
from rich_native_body_review_20261016_primitive import audit
ROOT=Path(__file__).resolve().parent
P='rich_native_body_review_20261016_'
RUN='rich_composite_geometry_20261015_correlated_dressing3'
NAMES=('Gram','rotation','electric','plaquette','quark_vector','white_vector','quark_scalar','white_scalar','loopB','loopC')


def write_result(result):
    (ROOT/(P+'result.json')).write_text(json.dumps(result,indent=2)+'\n')


def sector(meta,allops,j):
    started=time.time();labels=meta['labels'];keep=np.array([i for i,row in enumerate(labels) if row['twice_J']==j]);local_labels=[labels[i] for i in keep]
    data=data_from_labels(local_labels);ops={name:matrix[keep][:,keep].tocsr() for name,matrix in allops.items()};g=ops['Gram'];s,t,metric=metric_roots(g,local_labels);scalar,_=model.field_and_rotor(meta['p']);nf=len(scalar['grid'])
    physical={name:(t@matrix@t).tocsr() for name,matrix in ops.items() if name!='Gram'}
    c0=physical['rotation']+physical['electric']+physical['plaquette']+physical['quark_vector']+physical['white_vector'];cp=physical['quark_scalar']+physical['white_scalar'];hop0=ops['quark_vector']+ops['white_vector'];hopphi=ops['quark_scalar']+ops['white_scalar']
    h=FieldHamiltonian(c0,cp,scalar['H'],scalar['grid']);off=FieldHamiltonian(c0-physical['electric'],cp,scalar['H'],scalar['grid']);nohop=FieldHamiltonian(physical['rotation']+physical['electric']+physical['plaquette'],sparse.csr_matrix(g.shape),scalar['H'],scalar['grid'])
    probe=next(probe for probe in meta['probes'] if probe['descriptor']==dict(twice_J=j,spec=([0,0,0],'half_triplet' if j==1 else 'threehalf',j),family='bare'))
    x=probe['coordinates'][keep];source_raw=float(np.vdot(x,g@x).real);conditioned=x*data['Csite'][:,0,None];strength=float(np.vdot(conditioned,g@conditioned).real)
    if strength<=0:raise AssertionError('Original pinned compact0 source has zero strength')
    prepared=(s@conditioned/np.sqrt(strength)).reshape(-1);initial_count=count_distribution(prepared,data);times=list(meta['p']['times'])
    # Independent RNG and full-space solve; no imported eigen/prepared/evolved states.
    rng=np.random.default_rng(20261016+j);v0=rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0])
    energies,vectors=eigsh(h,k=2,which='SA',tol=8e-11,maxiter=6000,v0=v0);order=np.argsort(energies);energies=energies[order];vectors=vectors[:,order]
    residuals=[float(np.linalg.norm(h@vectors[:,i]-energies[i]*vectors[:,i])) for i in range(len(energies))]
    ground=readout(vectors[:,0],data,physical,scalar,h);dual=(s@x).reshape(-1);residue=float(abs(np.vdot(vectors[:,0],dual))**2)
    print(json.dumps(dict(stage='fresh_full_native_spectrum',twice_J=j,energies=energies.tolist(),residuals=residuals)),flush=True)
    trajectories={'full':evolve(h,prepared,times),'electric_OFF':evolve(off,prepared,times),'ALLhopOFF':evolve(nohop,prepared,times)}
    result=dict(twice_J=j,body_dimension=len(keep),full_Phi_dimension=h.shape[0],metric=metric,source_raw_retained_strength=source_raw,source_declared_raw_strength=probe['raw_strength'],compact0_conditioning_strength=strength,fresh_energies=energies.tolist(),fresh_eigen_residuals=residuals,raw_offset_to_hot_selected_reference=float(energies[0]-meta['reference_energy']),raw_observed_lowest_single_vector_source_residue=residue,ground=ground,trajectories={})
    for name,states in trajectories.items():
        op=h if name=='full' else off if name=='electric_OFF' else nohop
        result['trajectories'][name]=[dict(t=float(timepoint),material_count_TV=total_variation(count_distribution(state,data),initial_count),**readout(state,data,physical,scalar,op)) for timepoint,state in zip(times,states)]
    nohopTV=max(row['material_count_TV'] for row in result['trajectories']['ALLhopOFF'])
    if nohopTV>2e-10:raise AssertionError(('ALLhopOFF changed actual material count probabilities',nohopTV))
    current0=[];currentphi=[];ward=[];finite_checks=[]
    div0=[sparse.csr_matrix(g.shape,dtype=complex) for _ in range(4)];divphi=[m.copy() for m in div0]
    for edge,(start,end) in enumerate(model.EDGES):
        jw0=peierls(hop0,data,edge,1);jwphi=peierls(hopphi,data,edge,1)
        div0[start]+=jw0;div0[end]-=jw0;divphi[start]+=jwphi;divphi[end]-=jwphi
        current0.append(t@jw0@t);currentphi.append(t@jwphi@t)
    for site in range(4):
        q=sparse.diags(data['Q'][:,site]);w0=1j*((ops['rotation']+ops['electric']+ops['plaquette']+hop0)@q-q@(ops['rotation']+ops['electric']+ops['plaquette']+hop0))+div0[site];wp=1j*(hopphi@q-q@hopphi)+divphi[site]
        ward.append(float(np.sqrt(nf*sparse.linalg.norm(w0)**2+np.dot(scalar['grid'],scalar['grid'])*sparse.linalg.norm(wp)**2)))
    current_operators=[FieldHamiltonian(a,b,np.zeros_like(scalar['H']),scalar['grid']) for a,b in zip(current0,currentphi)]
    result['current_EM_by_edge']=[[float(np.vdot(state,op@state).real) for state in trajectories['full']] for op in current_operators]
    result['material_Q_Ward_Frobenius']=ward
    selected=trajectories['full'][3];dy=(-1j*(h@selected)).reshape(len(keep),nf);z=selected.reshape(len(keep),nf);qdot=(2*np.real(np.sum(z.conj()*dy,axis=1)))@data['Q'];flux=np.zeros(4)
    for edge,(start,end) in enumerate(model.EDGES):
        current=float(np.vdot(selected,current_operators[edge]@selected).real);flux[start]+=current;flux[end]-=current
    result['actual_Qdot_vs_minus_divJ_error']=float(np.linalg.norm(qdot+flux))
    if max(ward)>2e-9 or result['actual_Qdot_vs_minus_divJ_error']>2e-10:raise AssertionError('Fresh native physical Q/current continuity failed')
    for weak in (hop0,hopphi):
        eps=1e-4;plus=peierls(weak,data,3,0,eps);minus=peierls(weak,data,3,0,-eps)
        jweak=peierls(weak,data,3,1);contact=peierls(weak,data,3,2)
        finite_checks.append(dict(current_difference=float(sparse.linalg.norm(-(plus-minus)/(2*eps)-jweak)),contact_difference=float(sparse.linalg.norm((plus+minus)/(eps*eps)-contact))))
    result['finite_Peierls_derivative_checks']=finite_checks
    snapshots={'prepared':prepared,'fresh_eigenvectors_y':vectors,'fresh_eigenvectors_x':(t@vectors.reshape(len(keep),-1)).reshape(vectors.shape),'fresh_energies':energies,'sector_global_indices':keep,'times':np.array(times)}
    for name,states in trajectories.items():snapshots[name+'_states_y']=np.array(states)
    if j==1:
        # Plausible wrong OFF: removing only Phi/scalar hops leaves real vector transport.
        wrong=FieldHamiltonian(c0,sparse.csr_matrix(g.shape),scalar['H'],scalar['grid']);wrong_state=evolve(wrong,prepared,[.25])[0];wrong_tv=total_variation(count_distribution(wrong_state,data),initial_count)
        try:
            assert wrong_tv<2e-10,'Claimed ALLhopOFF retained vector hops: actual material count PVM moves'
        except AssertionError as error:wrong_rejection=str(error)
        else:raise AssertionError('Wrong scalar-only OFF did not reach discriminating transport consumer')
        result['wrong_scalar_only_ALLhopOFF']=dict(t=.25,material_count_TV=wrong_tv,expected_physical_assertion_rejected=wrong_rejection);snapshots['wrong_scalar_only_OFF_y']=wrong_state
        config=json.loads((ROOT/(RUN+'_input.json')).read_text());theta=config['material_EM_angle'];duration=config['material_EM_duration']
        pulse=FieldHamiltonian(c0+t@peierls(hop0,data,3,0,theta)@t,cp+t@peierls(hopphi,data,3,0,theta)@t,scalar['H'],scalar['grid'])
        after=evolve(pulse,prepared,[duration])[0];same_age=evolve(h,prepared,[duration])[0]
        kicked=(prepared.reshape(len(keep),nf)@expm(-1j*meta['p']['field_kick']*scalar['P']).T).reshape(-1)
        event_states={'EM_pulse':evolve(h,after,times),'same_age_no_EM':evolve(h,same_age,times),'Phi_only_kick':evolve(h,kicked,times)}
        result['events']={}
        for name,states in event_states.items():
            result['events'][name]=[dict(t=float(timepoint),absolute_age=float(timepoint+(duration if name!='Phi_only_kick' else 0)),**readout(state,data,physical,scalar,h)) for timepoint,state in zip(times,states)];snapshots[name+'_states_y']=np.array(states)
        result['field_only_initial_count_TV']=total_variation(count_distribution(kicked,data),initial_count)
        result['electric_OFF_retained_colour']=dict(plaquette_Frobenius=float(sparse.linalg.norm(physical['plaquette'])),quantum_loopC_initial=result['trajectories']['electric_OFF'][0]['loopC'],quantum_loopB_initial=result['trajectories']['electric_OFF'][0]['loopB'],meaning='ONLY op_electric removed; plaquette and quantum U remain. Not allcolourOFF.')
        result['field_kick_scope']='Ibody tensor exp(-i*.3Pphi) before t0, same compact0 source; no matter conditioning, no frozen Phi/R/PW or receiver change.'
    result['elapsed_seconds']=time.time()-started
    np.savez(ROOT/(P+f'J{j}_raw.npz'),**snapshots)
    print(json.dumps(dict(stage='fresh_native_physics',twice_J=j,full_final=result['trajectories']['full'][-1],off_final=result['trajectories']['electric_OFF'][-1],allhopOFF_max_count_TV=nohopTV)),flush=True)
    return result


def main():
    started=time.time();meta=load(ROOT/(RUN+'_body_basis.pkl'));ops={name:sparse.load_npz(ROOT/(RUN+'_op_'+name+'.npz')).tocsr() for name in NAMES}
    raw_errors={}
    for name,matrix in ops.items():
        raw=sparse.load_npz(ROOT/(RUN+'_raw_'+name+'.npz'));raw_errors[name]=dict(raw_antihermitian_Frobenius=float(sparse.linalg.norm(raw-raw.conj().T)),stored_average_error=float(sparse.linalg.norm(matrix-(raw+raw.conj().T)/2)))
    result=dict(kind='INDEPENDENT_FRESH_NATIVE_RICH_COMPOSITE_STATE_TRANSPORT_NOT_PARTICLE_ADMISSION',primitive_audit=audit(meta,ops),raw_operator_epochs=raw_errors,sectors={})
    write_result(result)
    for spin in (1,3):
        result['sectors'][str(spin)]=sector(meta,ops,spin);write_result(result);gc.collect()
    result['threehalf_minus_half']=result['sectors']['3']['fresh_energies'][0]-result['sectors']['1']['fresh_energies'][0]
    result['limits']='Fixed rich native source/sea/R/PW body span tensor full independentPhi7. No1328 reconstruction, author CLI replay, solved/evolved state import, tagged-three-particle PVM, lab convection/Maxwell, fullprimitive lifetime, breakup threshold/binding/proton/mass or derived primitive quark-spin/charge law. Negative offsets to hot reference remain raw preparations, not mass gaps. Chosen J=L_left+S_CAR is distinct from deriving primitive spin from field-only Noether origin.'
    result['resource']=dict(elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);write_result(result)
    print(json.dumps(dict(kind=result['kind'],threehalf_minus_half=result['threehalf_minus_half'],resource=result['resource'])),flush=True)


if __name__=='__main__':main()

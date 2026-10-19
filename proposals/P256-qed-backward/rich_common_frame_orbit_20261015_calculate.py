"""Actual NEW radial/frame/Gauss/colour/source spectrum and physical response."""
import json,resource,sys
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import eigh,expm
from scipy.sparse.linalg import eigsh,expm_multiply,spsolve
from rich_common_frame_orbit_20261015_model import build,law,observable,coupled,source,FormOperator,full_raw_source_gram,route_current
from rich_common_frame_orbit_20261015_frame import coordinate_noether,radial_form_readout
from rich_common_frame_orbit_20261015_reduce import explicit,factorized,csr_snapshot,snapshot_operator


def mean(v,op):return float(np.vdot(v,op@v).real)
def length(v):return float(np.linalg.norm(v))
def norm(op):return float(sparse.linalg.norm(op))


def lowest(matrix,p):
    dimension=matrix.shape[0]
    if sparse.issparse(matrix) and dimension<=p['dense_threshold']:
        values,vectors=eigh(matrix.toarray(),check_finite=False);complete=True
    else:
        values,vectors=eigsh(matrix,k=min(p['low_eigenstates'],dimension-2),which='SA',tol=p['eigen_tolerance'],v0=np.linspace(.2,1.,dimension));order=np.argsort(values);values=values[order];vectors=vectors[:,order];complete=False
    residual=max(length(matrix@vectors[:,i]-values[i]*vectors[:,i]) for i in range(len(values)))
    meta=dict(dimension=dimension,nonzeros=getattr(matrix,'nnz',None),complete_spectrum=complete,eigenstates=len(values),eigen_residual=residual,Hermiticity=norm(matrix-matrix.conj().T) if sparse.issparse(matrix) else None)
    if not sparse.issparse(matrix):
        rng=np.random.default_rng(222315);u=rng.normal(size=dimension)+1j*rng.normal(size=dimension);v=rng.normal(size=dimension)+1j*rng.normal(size=dimension);u/=length(u);v/=length(v);hu=matrix@u;hv=matrix@v
        meta['Hermiticity_bilinear_relative_probe']=float(abs(np.vdot(u,hv)-np.vdot(hu,v))/max(length(hu)+length(hv),np.finfo(float).tiny))
        meta['representation']='Exact-J factorized matrix-free; full norm Hermiticity not evaluated'
    return values,vectors,meta


def j_operators(model,name):
    s=model['sectors'][name];field=[observable(model,name,matter=a) for a in s['frame']['left']];material=[observable(model,name,orientation=a) for a in model['rotor']['L']]
    return field,material


def j2_apply(v,field,material):
    result=np.zeros_like(v)
    for a,b in zip(field,material):
        moved=a@v+b@v;result+=a@moved+b@moved
    return result


def snapshot(model,path):
    arrays={};meta={}
    for name,s in model['sectors'].items():
        arrays[name+'_spin_labels']=np.array(s['spin']['labels']);csr_snapshot(arrays,name+'_frame_P',s['frame']['P']);snapshot_operator(law(model,name),arrays,name+'_H')
        if s.get('representation')=='native_charge_orbit':
            csr_snapshot(arrays,name+'_spinU',s['spin']['U']);arrays[name+'_native_charge3_site0']=s['native']['charge3_site0'];arrays[name+'_native_twice_Sz']=s['native']['twice_Sz']
            native_error=float(s['native_isometry'].isometry_error);raw_dimension=None
        else:
            arrays[name+'_masks']=np.array(s['native']['masks'],dtype=np.uint64);arrays[name+'_spinU']=s['spin']['U'];csr_snapshot(arrays,name+'_colour_P',s['native']['P'])
            native_error=None;raw_dimension=s['native']['raw_dimension']
        meta[name]=dict(representation=s.get('representation','band_pair_control'),matter_dimension=len(s['spin']['j']),full_dimension=len(s['spin']['j'])*10*model['radial'].count,raw_colour_dimension=raw_dimension,native_raw_seed_Gram_isometry_error=native_error,raw_frame_dimension=s['frame']['P'].shape[0],right_spin_multiplets=s['spin']['multiplets'],right_Spin_Gauss=s['frame']['Gauss'],colour_Gauss=s['colour_Gauss'],colour_Gauss_provenance=s.get('colour_Gauss_provenance','Actual finite native raw G matrix measured'),frame_isometry=s['frame']['isometry'],frame_left_equals_reduced=s['frame']['left_equals_reduced'],spin_basis_unitarity=s['spin']['unitarity'],complete_spin_basis=s['spin']['completion'],right_Spin_charge_commutator=s['spin'].get('right_spin_charge_commutator'))
        meta[name]['spin_sparse_roundoff']=s['spin'].get('sparse_roundoff')
    arrays['material_U']=model['material']['U'];arrays['material_labels']=np.array(model['material']['labels']);arrays['material_quadrature_R']=model['rotor']['R'];arrays['material_quadrature_weights']=model['rotor']['weights']
    np.savez_compressed(path.with_suffix('.npz'),**arrays)
    result=dict(kind='NEW_EXACT_FACTOR_H_SNAPSHOT_BEFORE_EXPENSIVE_SPECTRA',parameters=model['parameters'],sectors=meta,radial=[radial_form_readout(model['radial'],j) for j in model['frame_js']],M_rad=model['radial'].mass,I='M_rad*r^2/2',matrix_metric=model['parameters']['matrix_metric'],frame_js=model['frame_js'],peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sparse_roundoff=model['sparse'].snapshot(),scope='Actual new factors/isometries are saved, not old H/state/cost replay. This selected band-pair/PW19/material/radial Galerkin control is NOT primary charge-orbit/EM integration.')
    result['representation']=model.get('native_payload',{'meaning':'Selected band-pair conditional control; not primary charge orbit'})
    if 'native_payload' in model:result['scope']='Actual charge-orbit BODY numeric/isometry factors consumed read-only, but NEW frame/radial/Phi-vector H/state/source/current rebuilt; NOT primary EM or old spatial H/state integration.'
    path.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    inp=Path(sys.argv[1]);out=Path(sys.argv[2]);p=json.loads(inp.read_text());model=build(p);base=out.with_name(out.stem.replace('_result',''))
    construction=snapshot(model,base.with_name(base.name+'_construction.json'));print(json.dumps(dict(stage='new_factor_snapshot',dimensions={k:v['full_dimension'] for k,v in construction['sectors'].items()},peak_rss_KiB=construction['peak_rss_KiB'])),flush=True)
    neutral=model['sectors']['neutral'];h0=law(model,'neutral');maxj=int(round(max(neutral['spin']['j'])+1));reference_candidates=[];neutral_spectra={}
    for J in range(maxj+1):
        v=coupled(model,'neutral',float(J),0.)
        if not v.shape[1]:continue
        matrix,ledger=explicit(h0,v);energies,vectors,meta=lowest(matrix,p);neutral_spectra[J]=(matrix,energies,vectors,v)
        meta['sparse_projection_roundoff']=ledger
        reference_candidates.append(dict(J=J,energy=float(energies[0]),numerics=meta))
    ref=min(reference_candidates,key=lambda x:x['energy']);J0=ref['J'];matrix,e0,vec0,v0=neutral_spectra[J0];ground=v0@vec0[:,0];energy=float(e0[0]);field0,material0=j_operators(model,'neutral')
    np.savez_compressed(base.with_name(base.name+'_reference.npz'),neutral_ground=ground,reference_energy=energy,reference_J=J0)
    field_side={name:coordinate_noether(s,int(round(s['op']['N'].diagonal().real.mean()))) for name,s in model['sectors'].items()}
    sources={};source_states={};spectra={};frame_controls={};source_checks={}
    for name,kind,spins in [('lepton','lepton',[.5]),('antilepton','lepton',[.5]),('baryon','baryon',[.5,1.5]),('antibaryon','baryon',[.5,1.5])]:
        s=model['sectors'][name];h=law(model,name);field,material=j_operators(model,name);spectra[name]={};frame_controls[name]={}
        for J in spins:
            family={};selected=None
            for M in np.arange(-J,J+1):
                operator=source(model,name,kind,J,float(M));family[str(float(M))]=operator.readout
                if M==J:selected=operator
            key=f'{name}:{J}';source_checks[key]=family
            raw=selected@ground;strength=float(np.vdot(raw,raw).real);created=raw/np.sqrt(strength);source_states[key]=created
            np.savez_compressed(base.with_name(base.name+'_source_checkpoint.npz'),neutral_ground=ground,reference_energy=energy,**{k.replace(':','_'):v for k,v in source_states.items()})
            gram=selected.matrix.conj().T@selected.matrix
            middle=FormOperator(neutral,neutral,model['rotor'],model['radial'],[(gram,np.eye(10),0)])
            full_gram,count=full_raw_source_gram(model,name,kind,J)
            physical_strength=mean(ground,full_gram);matter_strength=mean(ground,middle)
            joint=j2_apply(created,field,material);source_j2=float(np.vdot(created,joint).real)
            sources[key]=dict(raw_retained_strength=strength,full_unprojected_polynomial_strength=physical_strength,retained_matter_unprojected_radial_strength=matter_strength,omitted_matter_source_strength=physical_strength-matter_strength,omitted_radial_source_strength=matter_strength-strength,raw_image_masks=count,unfiltered_total_J2=source_j2,J2_variance=length(joint-source_j2*created),conditional_insertion_cost=mean(created,h)-energy,frame_kinetic=mean(created,observable(model,name,field_kind='frame')),radial_kinetic=mean(created,observable(model,name,field_kind='radial')),mean_r2=mean(created,observable(model,name,field_kind='r2')),charge=float(s['op']['Q'].diagonal().real.mean()),fermion_number=float(s['op']['N'].diagonal().real.mean()))
            projection=factorized if p.get('matrix_free_composite_spectra',False) and kind=='baryon' else explicit
            v=coupled(model,name,J,J);hm,ledger=projection(h,v);energies,vectors,meta=lowest(hm,p);component=v.conj().T@raw;amplitudes=vectors.conj().T@component;weights=abs(amplitudes)**2
            meta['sparse_projection_roundoff']=ledger
            bright=next((i for i,w in enumerate(weights) if w>p['bright_tolerance']),None)
            spectra[name][str(J)]=dict(lowest_energy=float(energies[0]),lowest_sector_gap=float(energies[0]-energy),lowest_computed_bright_gap=None if bright is None else float(energies[bright]-energy),low_poles=[dict(gap=float(x-energy),strength=float(w)) for x,w in zip(energies,weights)],source_projection_to_J_strength=float(np.vdot(component,component).real),unresolved_source_strength=strength-float(np.sum(weights)),numerics=meta)
            frame=observable(model,name,field_kind='frame');fm,_=projection(frame,v);off=hm-fm;off_e,off_v,off_meta=lowest(off,p)
            frame_controls[name][str(J)]=dict(frame_off_lowest_energy=float(off_e[0]),full_lowest_frame_energy=mean(vectors[:,0],fm),full_lowest_r2=mean(v@vectors[:,0],observable(model,name,field_kind='r2')),control_numerics=off_meta,meaning='Different-law control removes ONLY the derived field-orbit kinetic, not a fitted subtraction. All couplings/radial measure/trial subspaces unchanged; source/reference/costs are not borrowed from old scalar H.')
            print(json.dumps(dict(stage='new_source_sector',source=key,strength=strength,gap=float(energies[0]-energy),source_J2=source_j2,Gauss=max(x['raw_source_Gauss'] for x in family.values()),peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)
            del hm,fm,off,off_v,vectors
        del h
    np.savez_compressed(base.with_name(base.name+'_source_checkpoint.npz'),neutral_ground=ground,**{k.replace(':','_'):v for k,v in source_states.items()})
    ordering={}
    for name in ['baryon','antibaryon']:
        ordering[name]=dict(full_E_threehalf_minus_half=spectra[name]['1.5']['lowest_energy']-spectra[name]['0.5']['lowest_energy'],frame_off_E_threehalf_minus_half=frame_controls[name]['1.5']['frame_off_lowest_energy']-frame_controls[name]['0.5']['frame_off_lowest_energy'])
    # Full retained neutral current response in the exact J=1 symmetry fibre
    # when the actual neutral reference is invariant. No low-pole truncation.
    currents=[law(model,'neutral',derivative=(a,)) for a in range(6)];derivative=[];contact=[];response=[];step=1e-3
    for a in range(6):
        ext=np.zeros(6);ext[a]=step;hp=law(model,'neutral',ext);hm=law(model,'neutral',-ext);numerical=(hp@ground-hm@ground)/(2*step)
        derivative.append(length(numerical+currents[a]@ground)/max(1.,length(currents[a]@ground)))
        second=law(model,'neutral',derivative=(a,a));contact.append(length((hp@ground+hm@ground-2*(h0@ground))/step**2-second@ground)/max(1.,length(second@ground)))
        if J0==0:
            hc,unused_e,unused_v,v=neutral_spectra[1];probe=v.conj().T@(currents[a]@ground)
            # z Cartesian channels carry M0; x/y have M±1 and are separately
            # retained in derivative/source norms, not falsely projected as z.
            if a%3==2:
                rows=[]
                for omega in p['frequencies']:
                    frequency=omega+1j*p['eta'];zplus=energy+frequency;zminus=energy-frequency
                    plus=spsolve(zplus*sparse.eye(hc.shape[0])-hc,probe);minus=spsolve(zminus*sparse.eye(hc.shape[0])-hc,probe)
                    rp=np.vdot(probe,plus);rm=np.vdot(probe,minus);retarded=rp+rm
                    rows.append(dict(omega=omega,positive_order_real=float(rp.real),positive_order_imag=float(rp.imag),negative_order_real=float(rm.real),negative_order_imag=float(rm.imag),retarded_commutator_real=float(retarded.real),retarded_commutator_imag=float(retarded.imag),full_linear_solve_residual=max(length((zplus*plus-hc@plus)-probe),length((zminus*minus-hc@minus)-probe))/max(1.,length(probe))))
                response.append(dict(channel=a,full_retained_virtual_dimension=hc.shape[0],current_strength=float(np.vdot(probe,probe).real),positive_contact=mean(ground,second),signed_retarded_resolvent=rows,semantics='Both frequency orders retained: 1/(omega-Delta+i eta)-1/(omega+Delta+i eta). Finite eta is not an adiabatic-limit claim.'))
    # Current/dipole compression defect remains a declared band-pair and
    # material Galerkin issue, not repaired by changing the spin supplier.
    dipole=observable(model,'neutral',matter=model['sectors']['neutral']['op']['density_delta'],orientation=model['rotor']['integrate'](model['rotor']['R'][:,2,0]),power=0)
    dipole_scale=p['vertex_separation']/2
    ward=1j*dipole_scale*(h0@(dipole@ground)-dipole@(h0@ground))-currents[2]@ground
    rho0=observable(model,'neutral',matter=neutral['op']['rho0']);routes=[route_current(model,'neutral',edge) for edge in range(2)]
    graph_ward=1j*(h0@(rho0@ground)-rho0@(h0@ground))+sum((a@ground for a in routes),np.zeros_like(ground))
    no_hops=law(model,'neutral',hop_factor=0.)
    no_hops_density=1j*(no_hops@(rho0@ground)-rho0@(no_hops@ground))
    # Independent physical MATRIX-FIELD orientation/radial pulse; NOT an EM
    # photon kick and not primitive-CAR physical-spin rotation.
    name=p.get('event_sector','baryon');s=model['sectors'][name];state0=source_states[name+':0.5'];shape=(len(s['spin']['j']),10,model['radial'].count)
    rotated=state0.reshape(shape)*np.exp(-1j*p['frame_rotation_kick']*s['spin']['m'])[:,None,None]
    kicked=np.zeros_like(rotated)
    for j in set(s['spin']['j']):
        rows=s['spin']['j']==j;kicked[rows]=np.einsum('ian,mn->iam',rotated[rows],expm(-1j*p['radial_kick']*model['radial'].moment(float(j),float(j),1)),optimize=True)
    kicked=kicked.ravel();states=[np.zeros_like(kicked) for _ in p['times']];baselines=[np.zeros_like(kicked) for _ in p['times']];event_components=[]
    for twice in range(1,int(round(2*(max(s['spin']['j'])+1)))+1,2):
        J=twice/2;v=coupled(model,name,J,.5)
        if not v.shape[1]:continue
        hm,_=explicit(law(model,name),v);x=v.conj().T@kicked;b=v.conj().T@state0;event_components.append(dict(J=J,prepared_weight=float(np.vdot(x,x).real),dimension=hm.shape[0]))
        for i,t in enumerate(p['times']):
            pair=expm_multiply(-1j*t*hm,np.column_stack([x,b]),traceA=-1j*t*hm.diagonal().sum());states[i]+=v@pair[:,0];baselines[i]+=v@pair[:,1]
        del hm
    hb=law(model,name);fb,lb=j_operators(model,name);event=[]
    frame_energy=observable(model,name,field_kind='frame');radial_r2=observable(model,name,field_kind='r2');loop_B=observable(model,name,matter=s['op']['B']);colour_energy=observable(model,name,matter=s['op']['electric']);density=observable(model,name,matter=s['op']['density_delta'])
    for t,state,baseline in zip(p['times'],states,baselines):
        record=dict(time=t,norm=float(np.vdot(state,state).real),energy=mean(state,hb),energy_drift=mean(state,hb)-mean(kicked,hb),field_left_Noether=[mean(state,a) for a in fb],material_orbital=[mean(state,a) for a in lb],total_J=[mean(state,a)+mean(state,b) for a,b in zip(fb,lb)],field_left_J2=float(np.sum([np.vdot(a@state,a@state).real for a in fb])),frame_kinetic=mean(state,frame_energy),frame_kinetic_excess=mean(state,frame_energy)-mean(baseline,frame_energy),radial_r2=mean(state,radial_r2),radial_r2_excess=mean(state,radial_r2)-mean(baseline,radial_r2),colour_electric_excess=mean(state,colour_energy)-mean(baseline,colour_energy),loop_plaquette_excess=mean(state,loop_B)-mean(baseline,loop_B),site_charge_delta_excess=mean(state,density)-mean(baseline,density))
        hstate=hb@state
        record['field_left_torque_z']=float(2*np.vdot(fb[2]@state,hstate).imag)
        record['material_torque_z']=float(2*np.vdot(lb[2]@state,hstate).imag)
        record['total_torque_z']=record['field_left_torque_z']+record['material_torque_z']
        if s.get('representation')=='native_charge_orbit':
            charge_by_id={x['id']:x['charge3'] for x in s['spin']['multiplets']};labels=np.array([charge_by_id[x[0]] for x in s['spin']['labels']])
            weights=np.sum(abs(state.reshape(shape))**2,axis=(1,2));baseweights=np.sum(abs(baseline.reshape(shape))**2,axis=(1,2))
            probabilities={str(int(h)):float(np.sum(weights[labels==h])) for h in sorted(set(labels))};baseprob={str(int(h)):float(np.sum(baseweights[labels==h])) for h in sorted(set(labels))}
            record['material_charge_PVM']=dict(h_equals_3rho0=probabilities,probability_sum=sum(probabilities.values()),TV_from_untouched_baseline=sum(abs(probabilities[h]-baseprob[h]) for h in probabilities)/2,meaning='Exact native material TOTAL site0 charge, physical rho=h/3; not added-particle position, source-overlap capture or lab COM.')
        event.append(record)
    # Field/source Noether is distinct from physical electromagnetic current.
    rng=np.random.default_rng(221015);probe=rng.normal(size=h0.shape[0])+1j*rng.normal(size=h0.shape[0]);probe/=length(probe)
    noether=[length(h0@(a@probe+b@probe)-a@(h0@probe)-b@(h0@probe))/max(1.,length(h0@probe)) for a,b in zip(field0,material0)]
    result=dict(kind='NEW_ORDERED_MATRIX_FRAME_ORBIT_QUANTUM_CONSUMER_CONDITIONAL_CONTROL',parameters=p,construction=construction,chosen_gauging='Time-local a0, L=Mr rdot²/2+I(r)(omega-a0)²/2+i cdag cdot-a0·S_CAR-H; variation gives R_frame+S_CAR=0. Gauging/CAR/topology are inputs, not derived from trace metric.',reference=dict(energy=energy,total_J=J0,all_retained_neutral_J_candidates=reference_candidates,field_kinetic=mean(ground,observable(model,'neutral',field_kind='frame')),r2=mean(ground,observable(model,'neutral',field_kind='r2'))),field_side_Noether=field_side,source_Wigner_Gauss=source_checks,sources=sources,spectra=spectra,derived_frame_kinetic_controls=frame_controls,ordering=ordering,current=dict(physical_definition='-dH/dexternal A/B, positive contact+d²H, full route products and mechanical material rotation. No identification of field Noether spin with EM current.',same_state_first_derivative=derivative,positive_contact_second_derivative=contact,neutral_full_virtual_response=response,actual_new_dipole_compression_defect=length(ward),scope='Selected band-pair control still need not commute local EM density; no charge-continuity repair is claimed.'),physical_rotation=dict(full_tensor_HJ_random_probe_relative=noether,primitive_CAR_physical_spin_is_zero=True),event=dict(preparation='Unfiltered actual Bhalf source; LEFT matrix-field g rotation about z and actual radial-field unitary; NOT an EM photon preparation, no source/species/J filter. Same-state untouched baseline.',components=event_components,samples=event),runtime=dict(peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sparse_roundoff=model['sparse'].snapshot()),limits=['Restricted positive-amplitude ordered Clifford orbit, not full Phi/Z quantum phase or derived ordering','Primitive CAR32, gauged right Spin/time connection, colour SU3/PW19, flavour/Q, skin/core/binding parameters remain inputs','Derived r-dependent field kinetic via actual Noether/Gauss embedding is not fitted S² exchange or elementary proton','Band-pair/PW/material/radial Galerkin control; source leakage/cost/pole strength distinct, no global all-N ground theorem','No freeCOM, integrated quantum EM photons, Lorentz/Dirac or Nature claim; independent of primary charge-orbit/EM law'])
    result['representation']=model.get('native_payload',{'meaning':'Selected band-pair conditional control, not primary charge orbit'})
    result['current']['actual_new_material_graph_Ward_ground']=length(graph_ward)
    result['current']['all_hops_off_material_density_motion_ground']=length(no_hops_density)
    result['event']['sector']=name
    result['event']['preparation']=f'Unfiltered actual {name} half-source; LEFT matrix-field g z rotation and radial-field unitary, NOT EM photons or primitive-CAR physical-spin rotation. Same-state untouched baseline; complete M=1/2 fibre, all total J retained.'
    if 'native_payload' in model:
        result['limits']=[x for x in result['limits'] if not x.startswith('Band-pair/')]
        result['limits'].append('Native exact physical-charge orbit/seed truncation, PW19/material/radial Galerkin; not full H closure or a global all-N vacuum theorem.')
        result['current']['scope']='Material graph continuity is measured from independent Peierls route derivatives, never defined as a commutator. Finite material orientation/lab dipole and native packet/H omissions remain distinct possible limits.'
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),neutral_ground=ground,reference_energy=energy,event_times=p['times'],event_states=np.array(states),baseline_states=np.array(baselines),field_prepared=kicked,**{k.replace(':','_'):v for k,v in source_states.items()})
    print(json.dumps(dict(reference_J=J0,source_gaps={name:{j:data['lowest_sector_gap'] for j,data in entries.items()} for name,entries in spectra.items()},ordering=ordering,field_Noether_max=max(x['actual_left_g_derivative'] for x in field_side.values()),raw_Spin_Gauss=max(x['right_Spin_Gauss'] for x in construction['sectors'].values()),HJ=noether,final_r2_excess=event[-1]['radial_r2_excess'],final_frame_energy_excess=event[-1]['frame_kinetic_excess'],peak_rss_KiB=result['runtime']['peak_rss_KiB'])))


if __name__=='__main__':main()

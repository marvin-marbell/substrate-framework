"""Actual same-law native charge-PVM/source/current/field consumer, no fits."""
import json,resource,sys,time
from pathlib import Path
from itertools import product
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from rich_spatial_colour_charge_20261015_model import (
    observable,BodyLift,TensorOperator,charge_site0_image,
    spin_image,clean,packet_matrix,generators,mode)
from rich_spatial_colour_charge_20261015_snapshot import load
from rich_spatial_colour_charge_20261015_factors import law,route_current,stationary,matrix
from rich_spatial_colour_root_20261015_calculate import mean,norm,cjson,response_tensor,spin_component_response,solve_resolvent
from rich_spatial_colour_root_20261015_model import changed_masks,one_body,law as collapsed_law,route_current as collapsed_route


def propagate(h,state,t):
    return state.copy() if t==0 else expm_multiply(-1j*t*h,state,traceA=-1j*t*h.trace())


def commutator_norm(a,b,added=(),factor=1.):
    """Exact all-row Frobenius witness, bounded scratch; never redefine current."""
    total=0.
    for start in range(0,a.shape[0],128):
        rows=slice(start,min(start+128,a.shape[0]))
        difference=factor*(a[rows]@b-b[rows]@a)
        for matrix in added:difference+=matrix[rows]
        total+=float(np.vdot(difference.data,difference.data).real)
    return float(np.sqrt(total))


def hermiticity_norm(matrix):
    total=0.;transpose=matrix.T
    for start in range(0,matrix.shape[0],128):
        rows=slice(start,min(start+128,matrix.shape[0]))
        difference=matrix[rows]-transpose[rows].conj()
        total+=float(np.vdot(difference.data,difference.data).real)
    return float(np.sqrt(total))




def pvm(state,basis,nf):
    weights=np.sum(abs(state.reshape(basis.dimension,10,nf))**2,axis=(1,2))
    labels=sorted(set(basis.h));values=np.array([sum(weights[basis.h==h]) for h in labels])
    return dict(h3=[int(x) for x in labels],probabilities=values.tolist(),sum=float(sum(values)),rho0=float(np.dot(np.array(labels)/3,values)))


def distances(a,b):
    delta=np.array(a['probabilities'])-np.array(b['probabilities']);l1=float(sum(abs(delta)))
    return dict(L1=l1,L2=float(np.linalg.norm(delta)),TV=l1/2,rho0_excess=a['rho0']-b['rho0'])


def pole(energies,vectors,raw,h,p):
    weights=abs(vectors.conj().T@raw)**2;i=0
    while i<len(energies):
        ids=np.flatnonzero(abs(energies-energies[i])<p['degeneracy_tolerance']);weight=float(sum(weights[ids]))
        if weight>p['bright_threshold']:
            state=vectors[:,ids]@(vectors[:,ids].conj().T@raw);state/=np.linalg.norm(state)
            return state,dict(energy=mean(state,h),raw_pole_residue=weight,computed_low_raw_weights=weights.tolist(),uncomputed_high_raw_strength=float(np.vdot(raw,raw).real-sum(weights)),eigen_residual=norm(h@state-mean(state,h)*state))
        i=int(ids[-1])+1
    raise RuntimeError('No bright pole in explicitly computed low eigenspan; no fabricated source pole')


def state_observables(data,name,state,p):
    op=data['operators'][name];nf=len(data['scalar']['grid']);grid=data['scalar']['grid']
    weights=np.sum(abs(state.reshape(-1,nf))**2,axis=0)
    out={key:mean(state,BodyLift(op[key],10*nf)) for key in ['N','Q','rho0','rho1','S2','B','C','PW_boundary','electric']}
    positive=op['N']/2+sum((m[0]+m[0].conj().T for _,m in op['hops'][0]))/2
    out.update(positive_band_count=mean(state,BodyLift(positive,10*nf)),phi=float(weights@grid),phi2=float(weights@(grid**2)),scalar_boundary=float(weights[0]+weights[-1]),J2=mean(state,TensorOperator(op['J2'],sparse.csr_matrix(op['J2'].shape),data['scalar'])))
    out['mu']=[mean(state,law(op,data['rotor'],data['scalar'],p,data['geometry'],derivative=(3+a,))) for a in range(3)]
    out['phi_variance']=out['phi2']-out['phi']**2
    return out


def gauss_witness(basis,loop):
    masks,columns=packet_matrix(basis.seeds)
    terms=[[(mode(f,m,s,a),mode(f,m,s,b),t[a,b]) for f,m,s,a,b in product(range(2),range(2),range(2),range(3),range(3)) if t[a,b]!=0] for t in generators()]
    targets=changed_masks(masks,[x for tt in terms for x in tt]);lookup={m:i for i,m in enumerate(targets)}
    inclusion=sparse.csr_matrix((np.ones(len(masks)),([lookup[m] for m in masks],range(len(masks)))),shape=(len(targets),len(masks)))
    raw=[]
    for a,tt in enumerate(terms):
        image=sparse.kron(one_body(masks,tt,targets),sparse.eye(19),format='csr')@columns
        image+=sparse.kron(inclusion,loop['L'][a]+loop['R'][a],format='csr')@columns
        raw.append(norm(image))
    amplification=max(float(np.linalg.norm(w,2)) for w in basis.blocks.values())
    return dict(full_augmented_raw_masks=len(targets),raw_seed_Gauss_Frobenius=raw,max_native_Gauss_norm_upper=max(raw)*amplification,native_W_norm_upper=amplification,proof='G commutes physical P_h; raw untrimmed full one-colour-step G images bound EVERY normalized native/field state by ||GV||F max_h||W_h||2, not old compressed Gauss or assumed identity Gram. Local G1=0 from solved tree; G0 is this global G.')


def main():
    ip=Path(sys.argv[1]);prefix=Path(sys.argv[2]);request=json.loads(ip.read_text());producer=ip.parent/request['body_prefix'];data=load(str(producer)+'_body.pkl');p=data['parameters'];start=time.time()
    scalar=data['scalar'];rotor=data['rotor'];g=data['geometry'];nf=len(scalar['grid']);h={name:law(op,rotor,scalar,p,g) for name,op in data['operators'].items()}
    arrays=np.load(str(producer)+'_H0.npz')
    captured=np.load(str(producer)+'_neutral_state.npz');omega=captured['vectors'][:,0];e0=float(captured['energies'][0]);states={'neutral_reference':omega};results={}
    result=dict(kind='ACTUAL_PHYSICAL_SITE_CHARGE_ORBIT_COMMON_QUANTUM_FIELD_SOURCE_CURRENT_CONSUMER',producer_prefix=request['body_prefix'],parameters=p,neutral_energy=e0,neutral_J2=mean(omega,TensorOperator(data['operators']['neutral']['J2'],sparse.csr_matrix(data['operators']['neutral']['J2'].shape),scalar)),native={},sources={},C={},responses={},events={},limitations='Finite chosen gauge-orbit packet space, PW19, phi7 and SO3 j0+j1; not full H closure/global vacuum, unique proton/electron, full Cartesian gauge completeness, COM or quantum photons. Material charge PVM is NOT added-particle position/capture.')
    raw_snapshot={'neutral_reference':omega}
    for name,values in data['sources'].items():
        for key,source in values.items():raw_snapshot[name+':'+key]=source@omega
    np.savez_compressed(str(prefix)+'_raw_source_states.npz',**raw_snapshot)
    del raw_snapshot
    result['reference_observables']=state_observables(data,'neutral',omega,p)
    for name in ['neutral','lepton','baryon']:
        basis=data['native'][name];op=data['operators'][name];rho=op['rho0'];images=[charge_site0_image(x) for x in basis.seeds]
        direct=basis.projected(images,basis,0)
        # Full Q0^2 moment from derivative of ACTUAL raw-site projector polynomial,
        # not a square of compressed Q: Q0 P_h=(h/3)P_h in the original CAR space.
        metric=sparse.lil_matrix((basis.dimension,)*2,dtype=complex)
        for hq,w in basis.blocks.items():
            metric[basis.slices[hq],basis.slices[hq]]=(hq/3)**2*(w.conj().T@basis.grams[hq]@w)
        metric=clean(metric)
        spin_errors=[]
        for axis in range(3):
            si=[spin_image(x,axis) for x in basis.seeds];full=basis.image_metric(si,0)
            spin_errors.append(norm(full-op['S'][axis].conj().T@op['S'][axis]))
        charge0=observable(rho,rotor,scalar);routes=[route_current(op,rotor,scalar,p,g,e) for e in range(2)]
        original_routes=[collapsed_route(op,rotor,scalar,p,g,e) for e in range(2)]
        probe=np.random.default_rng(222).normal(size=h[name].shape[0])+1j*np.random.default_rng(223).normal(size=h[name].shape[0]);probe/=norm(probe)
        x=probe.reshape(-1,nf);one=matrix(arrays,name+'_H0_one')
        ward_one=commutator_norm(one,charge0.one,[j.one for j in original_routes],1j)
        rotation_one=[commutator_norm(j,one) for j in op['J']];hermitian_one=hermiticity_norm(one);expected=one@x;del one
        phi=matrix(arrays,name+'_H0_phi')
        ward_phi=commutator_norm(phi,charge0.one,[j.phi for j in original_routes],1j)
        rotation_phi=[commutator_norm(j,phi) for j in op['J']];hermitian_phi=hermiticity_norm(phi)
        expected+=(phi@x)*scalar['grid'][None,:]+x@scalar['H'].T;del phi
        action_error=norm(h[name]@probe-expected.ravel());route_errors=[norm(a@probe-b@probe) for a,b in zip(routes,original_routes)];del original_routes
        if action_error>1e-8 or max(route_errors)>1e-8:raise RuntimeError('Compact action differs from OWN immutable full H/current snapshot')
        results[name]=dict(charge=charge0,routes=routes)
        result['native'][name]=dict(dimension=basis.dimension,isometry_error=basis.isometry_error,physical_charge_projection_error=norm(direct-rho),full_raw_charge_closure_error=norm(metric-rho@rho),raw_charge_moment_scope='Exact full-site Laurent projector derivative, independently checked against raw CAR first-charge image; NOT compressed-Q functional calculus.',spin_closure_errors=spin_errors,Sz_label_error=norm(op['S'][2]-sparse.diags(basis.m/2)),SU2_commutator_error=norm(op['S'][0]@op['S'][1]-op['S'][1]@op['S'][0]-1j*op['S'][2]),Ward_one_norm=ward_one,Ward_phi_norm=ward_phi,all_hops_off_colour_charge_commutator=norm(op['electric']@rho-rho@op['electric']),charge_PVM_sum_error=norm(sum(data['sectors'][name]['charge_PVM'].values())-sparse.eye(basis.dimension)))
        result['native'][name].update(rotation_one_norms=rotation_one,rotation_phi_norms=rotation_phi,Hermitian_one_error=hermitian_one,Hermitian_phi_error=hermitian_phi,snapshot_H0_factor_action_error=action_error,owning_route_factor_action_errors=route_errors,witness_scope='Original captured H0 and original owning derivative route matrices; complete128-row Frobenius plus full-state factor action controls, not current replacement or virtual cut.')
        result['native'][name]['colour_Gauss']=gauss_witness(basis,data['loop'])
        if name=='neutral':
            controls=[]
            for derivative in [(),(0,),(3,),(0,0)]:
                fields=np.array(p['A_pulse']+p['B_pulse']) if not derivative else np.zeros(6)
                original=collapsed_law(op,rotor,scalar,p,g,fields,derivative)
                compact=law(op,rotor,scalar,p,g,fields,derivative)
                controls.append(dict(derivative=list(derivative),fields=fields.tolist(),full_state_action_error=norm(original@probe-compact@probe)))
                del original,compact
            result['native'][name]['finite_source_factor_controls']=controls
    # Native/source/operator snapshots and actual reference already exist before this diagnostic.
    del images,si,full,direct,metric
    Path(str(prefix)+'_native.json').write_text(json.dumps(dict(producer=request['body_prefix'],native=result['native']),indent=2)+'\n')
    for name in ['lepton','baryon']:
        op=data['operators'][name]
        if request.get('eigenspace_prefix'):
            previous=ip.parent/request['eigenspace_prefix'];prior_input=json.loads(Path(str(previous)+'_input.json').read_text())
            if prior_input['body_prefix']!=request['body_prefix']:raise ValueError('Different-H eigenspace checkpoint is forbidden')
            cached=np.load(str(previous)+'_states.npz');energies=cached[name+'_low_energies'];vectors=cached[name+'_low_vectors']
            einfo=dict(same_H_checkpoint=request['eigenspace_prefix'],residuals=[norm(h[name]@vectors[:,j]-energies[j]*vectors[:,j]) for j in range(len(energies))],orthogonality=norm(vectors.conj().T@vectors-np.eye(len(energies))),scope='Own same-body same-coupling immutable-H0 eigen checkpoint, not old/extended-law energy inheritance; full compact-H residual measured.')
        else:energies,vectors,einfo=stationary(arrays,name,h[name],op,rotor,scalar,.5,p['low_eigenstates'],p)
        states[name+'_low_vectors']=vectors;states[name+'_low_energies']=energies;result['sources'][name]={}
        for key,source in data['sources'][name].items():
            raw=source@omega;strength=float(np.vdot(raw,raw).real);metric=data['source_metric_full'][name][key];full=mean(omega,BodyLift(metric,10*nf))
            result['sources'][name][key]=dict(raw_retained_strength=strength,full_native_CAR_strength=full,omitted_source_strength=full-strength,preparation_cost=mean(raw,h[name])/strength-e0 if strength>1e-20 else None)
            if key in ['0:0.5:0.5','0:1.5:0.5']:
                chosen,info=pole(energies,vectors,raw,h[name],p);info.update(gap=info['energy']-e0,conditional_pole_fraction=info['raw_pole_residue']/strength,J2=mean(chosen,TensorOperator(op['J2'],sparse.csr_matrix(op['J2'].shape),scalar)),eigen=einfo)
                result['sources'][name][key].update(info);states[name+'_'+key+'_pole']=chosen;states[name+'_'+key+'_raw']=raw
        states[name+'_preparation']=(data['sources'][name]['0:0.5:0.5']@omega)/np.sqrt(result['sources'][name]['0:0.5:0.5']['raw_retained_strength'])
    np.savez_compressed(str(prefix)+'_states.npz',**states)
    Path(str(prefix)+'_sources.json').write_text(json.dumps(dict(neutral_energy=e0,sources=result['sources']),indent=2)+'\n')
    cn=data['C']['neutral'];result['C']['neutral_overlap']=float(abs(np.vdot(omega,cn@omega)))
    for name,anti in [('lepton','antilepton'),('baryon','antibaryon')]:
        c=data['C'][name];result['C'][name]={}
        anti_vectors=c@states[name+'_low_vectors'];anti_energies=states[name+'_low_energies']
        for key,src in data['source_matter'][name].items():
            direct=data['source_matter'][anti][key];transformed=src@data['C_matter']['neutral'].conj().T
            ar=data['sources'][anti][key]@omega;full=mean(omega,BodyLift(data['source_metric_full'][anti][key],10*nf));strength=float(np.vdot(ar,ar).real)
            result['C'][name][key]=dict(direct_source_operator_error=norm(direct-transformed),raw_anti_strength=strength,full_raw_anti_strength=full,anti_omission=full-strength,anti_preparation_cost=mean(ar,h[anti])/strength-e0 if strength>1e-20 else None)
            if key in ['0:0.5:0.5','0:1.5:0.5']:
                chosen,info=pole(anti_energies,anti_vectors,ar,h[anti],p)
                info.update(gap=info['energy']-e0,conditional_pole_fraction=info['raw_pole_residue']/strength,N=mean(chosen,BodyLift(data['operators'][anti]['N'],10*nf)),Q=mean(chosen,BodyLift(data['operators'][anti]['Q'],10*nf)))
                info['observables']=state_observables(data,anti,chosen,p)
                result['C'][name][key].update(info);states[anti+'_'+key+'_pole']=chosen;states[anti+'_'+key+'_raw']=ar
        probe=states[name+'_preparation'];finite=np.array(p['A_pulse']+p['B_pulse'])
        result['C'][name]['owning_law_probe']=dict(H0=norm(h[anti]@(c@probe)-c@(h[name]@probe)),finite_field=norm(law(data['operators'][anti],rotor,scalar,p,g,fields=-finite)@(c@probe)-c@(law(data['operators'][name],rotor,scalar,p,g,fields=finite)@probe)),current=[norm(law(data['operators'][anti],rotor,scalar,p,g,derivative=(a,))@(c@probe)+c@(law(data['operators'][name],rotor,scalar,p,g,derivative=(a,))@probe)) for a in range(3)])
    np.savez_compressed(str(prefix)+'_C_states.npz',**{key:value for key,value in states.items() if key.startswith(('antilepton','antibaryon'))})
    Path(str(prefix)+'_C.json').write_text(json.dumps(result['C'],indent=2)+'\n')
    for name in ['neutral','lepton','baryon']:
        basis=data['native'][name];op=data['operators'][name];state=omega if name=='neutral' else states[name+'_preparation'];rho=results[name]['charge'];routes=results[name]['routes'];j=[law(op,rotor,scalar,p,g,derivative=(a,)) for a in range(3)];mu=[law(op,rotor,scalar,p,g,derivative=(3+a,)) for a in range(3)]
        h_off=law(op,rotor,scalar,p,g,path_scale=(0.,0.));event={};initial=pvm(state,basis,nf);baseline=[];off=[]
        for t in p['times']:
            full=propagate(h[name],state,t);control=propagate(h_off,state,t)
            baseline.append(dict(t=t,charge_PVM=pvm(full,basis,nf),route_currents=[mean(full,x) for x in routes],phi=float(np.vdot(full.reshape(-1,nf),full.reshape(-1,nf)*scalar['grid']).real),colour_E=mean(full,observable(op['electric'],rotor,scalar))))
            measured=pvm(control,basis,nf);off.append(dict(t=t,charge_PVM=measured,change=distances(measured,initial),mechanical_current=[mean(control,law(op,rotor,scalar,p,g,derivative=(a,),path_scale=(0.,0.))) for a in range(3)]))
        for label,fields in [('A',p['A_pulse']+[0.,0.,0.]),('B',[0.,0.,0.]+p['B_pulse'])]:
            duration=p['pulse_duration'];driven=propagate(law(op,rotor,scalar,p,g,fields=fields),state,duration);base_after=propagate(h[name],state,duration);history=[]
            for t in p['times']:
                a=propagate(h[name],driven,t);b=propagate(h[name],base_after,t);pa=pvm(a,basis,nf);pb=pvm(b,basis,nf)
                history.append(dict(t=t,charge_PVM=pa,excess=distances(pa,pb),route_current_excess=[mean(a,x)-mean(b,x) for x in routes],Cartesian_current_excess=[mean(a,x)-mean(b,x) for x in j],moment_excess=[mean(a,x)-mean(b,x) for x in mu],phi_excess=float(np.sum((abs(a.reshape(-1,nf))**2-abs(b.reshape(-1,nf))**2)*scalar['grid'])),colour_E_excess=mean(a,observable(op['electric'],rotor,scalar))-mean(b,observable(op['electric'],rotor,scalar))))
            event[label]=history
        kicked=(state.reshape(-1,nf)@expm(-1j*p['field_kick']*scalar['P']).T).ravel();field_history=[]
        for t in p['times']:
            a=propagate(h[name],kicked,t);b=propagate(h[name],state,t)
            field_history.append(dict(t=t,charge_PVM=pvm(a,basis,nf),excess=distances(pvm(a,basis,nf),pvm(b,basis,nf)),route_current_excess=[mean(a,x)-mean(b,x) for x in routes],phi_excess=float(np.sum((abs(a.reshape(-1,nf))**2-abs(b.reshape(-1,nf))**2)*scalar['grid'])),colour_E_excess=mean(a,observable(op['electric'],rotor,scalar))-mean(b,observable(op['electric'],rotor,scalar))))
        electric_retained=norm(observable(op['electric'],rotor,scalar)@state)**2
        electric_upper=p['electric']*(g['lengths'][0]*(40/3)+g['lengths'][1]*(4/3))/2
        electric_tail_upper_squared=electric_upper*mean(state,observable(op['electric'],rotor,scalar))-electric_retained
        event.update(initial_charge_PVM=initial,baseline=baseline,ALL_hops_off=off,field_only=field_history,Ward_on_preparation=norm(1j*(h[name]@(rho@state)-rho@(h[name]@state))+sum((x@state for x in routes))),electric_omitted_norm_squared_upper=electric_tail_upper_squared,electric_operator_upper=electric_upper,electric_scope='Rigorous LOOSE colour-electric H COMPONENT tail:0<=E<=M from local four triplets plus PW3 gives treeC2<=40/3,cycleC2<=4/3; ||(1-P)Epsi||²<=M<E>-||PEpsi||². NOT total H leakage/error bound; SO3/PW/DVR/hopping tails remain uncontrolled. Signed floating bound is not clamped.')
        if name!='neutral':
            receiver=data['sources'][name]['1:0.5:0.5']@omega;receiver_norm=norm(receiver);raw0=data['sources'][name]['0:0.5:0.5']@omega;receiver_history=[]
            for t in p['times']:
                a=propagate(h[name],state,t);amp=np.vdot(receiver,a)*norm(raw0)
                receiver_history.append(dict(t=t,raw_CAR_transfer=cjson(amp),conditional_rank1_capture=float(abs(np.vdot(receiver/receiver_norm,a))**2)))
            event['CAR_preparation_receiver']=receiver_history
        result['events'][name]=event
        Path(str(prefix)+'_'+name+'_events.json').write_text(json.dumps(event,indent=2)+'\n')
        stationary_state=omega if name=='neutral' else states[name+'_0:0.5:0.5_pole']
        result['responses'][name]=response_tensor(h[name],j,lambda a,b:law(op,rotor,scalar,p,g,derivative=(a,b)),stationary_state,p)
        Path(str(prefix)+'_'+name+'_response.json').write_text(json.dumps(result['responses'][name],indent=2)+'\n')
    op=data['operators']['baryon'];result['responses']['baryon_threehalf_Iz']=spin_component_response(h['baryon'],law(op,rotor,scalar,p,g,derivative=(2,)),law(op,rotor,scalar,p,g,derivative=(2,2)),states['baryon_0:1.5:0.5_pole'],p)
    for name in ['antilepton','antibaryon']:
        op=data['operators'][name];currents=[law(op,rotor,scalar,p,g,derivative=(a,)) for a in range(3)]
        result['responses'][name]=response_tensor(h[name],currents,lambda a,b:law(op,rotor,scalar,p,g,derivative=(a,b)),states[name+'_0:0.5:0.5_pole'],p)
    result['raw_source_Greens']={};raw_g={}
    for name in ['lepton','baryon','antilepton','antibaryon']:
        for key in (['0:0.5:0.5','0:1.5:0.5'] if 'baryon' in name else ['0:0.5:0.5']):
            raw_state=states[name+'_'+key+'_raw'];samples=[];solves=[];raw_g[(name,key)]={}
            for w in p['frequencies']:
                solved,diagnostic=solve_resolvent(h[name],e0,raw_state,w,p);value=np.vdot(raw_state,solved)
                samples.append(value);solves.append(diagnostic);raw_g[(name,key)][w]=value
            result['raw_source_Greens'][name+':'+key]=dict(frequencies=p['frequencies'],eta=p['eta'],raw_strength=norm(raw_state)**2,G=cjson(samples),solves=solves,retained_virtual_dimension=h[name].shape[0],meaning='Unnormalized full native retained source resolvent, no low-pole virtual replacement.')
    result['C_paired_odd_source_G']={}
    for name,anti in [('lepton','antilepton'),('baryon','antibaryon')]:
        for key in (['0:0.5:0.5','0:1.5:0.5'] if name=='baryon' else ['0:0.5:0.5']):
            result['C_paired_odd_source_G'][name+':'+key]=dict(G=cjson([raw_g[(name,key)][w]-np.conj(raw_g[(anti,key)][-w]) for w in p['frequencies']]),meaning='Addition minus conjugated C-removal at negative frequency. Raw odd-source Lehmann combination, NOT elementary baryon/Dirac normalization.')
    result['spin_gap_difference']=result['sources']['baryon']['0:1.5:0.5']['gap']-result['sources']['baryon']['0:0.5:0.5']['gap']
    result['pole_observables']={name:{key:state_observables(data,name,states[name+'_'+key+'_pole'],p) for key in (['0:0.5:0.5','0:1.5:0.5'] if name=='baryon' else ['0:0.5:0.5'])} for name in ['lepton','baryon']}
    result['metric_definitions']=dict(L1='sum_h |p_h-driven-p_h-baseline|',L2='sqrt(sum_h(delta_p_h)^2)',TV='L1/2; normalized inclusive material physical-charge PVM distributions, NOT capture or added-particle locator',off='ALL quark/lepton hopping off fixes MATERIAL site-charge PVM. Rigid R can still rotate lab density/dipole and mechanical EM current need not vanish.')
    result['resource']=dict(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.time()-start)
    Path(str(prefix)+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(result=str(prefix)+'_result.json',resource=result['resource'],spin_gap_difference=result['spin_gap_difference'])),flush=True)

if __name__=='__main__':main()

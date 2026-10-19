"""Fresh independent first-law reciprocal consumer; no producer state/result replay."""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

ROOT=Path(__file__).resolve().parent;P='rich_common_quantum_em_review_20261016_'


def normalized(value):
    norm=float(np.vdot(value,value).real)
    if norm<=0:raise ValueError('cannot normalize a zero source/state')
    return value/np.sqrt(norm),norm


def source_pair(neutral,pair,reference,prefix):
    S=sparse.load_npz(prefix+'_source_orth.npz');gram=sparse.load_npz(prefix+'_source_full_CAR_gram_orth.npz')
    x=reference.reshape(neutral.n,len(neutral.grid),len(neutral.flux));flat=x.reshape(neutral.n,-1)
    shift=np.asarray(neutral.geometry['Wilson_transverse_shift']);where={tuple(v):i for i,v in enumerate(pair.flux)}
    valid=[i for i,m in enumerate(neutral.flux) if tuple(m+shift) in where];target=[where[tuple(neutral.flux[i]+shift)] for i in valid]
    full=float(np.vdot(flat,gram@flat).real);body_image=S@flat;body=float(np.vdot(body_image,body_image).real)
    selected=x[:,:,valid].reshape(neutral.n,-1);raw_after_flux=float(np.vdot(selected,gram@selected).real)
    result=np.zeros((pair.n,len(pair.grid),len(pair.flux)),complex)
    result[:,:,target]=(S@selected).reshape(pair.n,len(pair.grid),len(valid));state,retained=normalized(result.reshape(-1))
    coo=S.tocoo();delta=np.array([1.,0.,-1.,0.]);ward=max(float(np.linalg.norm(coo.data*(pair.q[coo.row,s]-neutral.q[coo.col,s]-delta[s]))) for s in range(4))
    assert ward<1e-11 and retained>0 and full>=body-1e-10 and body>=retained-1e-10
    return state,dict(full_CAR_Gram_strength_NOT_probability=full,body_before_flux_strength=body,raw_CAR_after_flux_cut_strength=raw_after_flux,retained_body_and_flux_strength=retained,retained_fraction=retained/full,body_lost_strength=full-body,raw_flux_lost_strength=full-raw_after_flux,Wilson_shift=shift.tolist(),source_charge_Ward_error=ward,source_grade_N=4,source_total_J=1,source_M=1,meaning='Complete canonical Bdag0 W Ldag2 operator; L first; no intermediate isolated charged/Gauss projection. Own actual native orthonormal reference, one final conditional normalization only.')


def observe(model,state,theta,control):
    norm=float(np.vdot(state,state).real);x=state.reshape(model.n,len(model.grid),len(model.flux));prob=abs(x)**2;body=prob.sum(axis=(1,2))/norm;field=prob.sum(axis=(0,1))/norm
    current={}
    for species,edge in (('quark',0),('white',2)):
        op=model.observer('current',theta=theta,species=species,edge=edge,source_species='white',source_edge=2,decoupled=control)
        current[species]=float(np.vdot(state,op@state).real/norm)
    boundary=np.max(abs(model.flux),axis=1)==6
    return dict(norm2=norm,physical_Q_site=(body@model.q).tolist(),quark_Q_site=(body@model.quark_q).tolist(),white_Q_site=(body@model.white_q).tolist(),physical_transverse_generator=(field@model.flux/3).tolist(),integer_m_mean=(field@model.flux).tolist(),physical_current=current,Phi=float(np.dot(prob.sum(axis=(0,2))/norm,model.grid)),field_boundary_weight=float(field[boundary].sum()),field_probability=field.tolist())


def difference(row,base):
    result={key:(np.asarray(row[key])-np.asarray(base[key])).tolist() for key in ('physical_Q_site','quark_Q_site','white_Q_site','physical_transverse_generator','integer_m_mean')}
    result['physical_current']={s:row['physical_current'][s]-base['physical_current'][s] for s in ('quark','white')};result['Phi']=row['Phi']-base['Phi'];return result


def contrast(pair,neutral):
    return {key:({s:pair[key][s]-neutral[key][s] for s in pair[key]} if isinstance(pair[key],dict) else (np.asarray(pair[key])-np.asarray(neutral[key])).tolist()) for key in pair}


def support(model,state):
    x=state.reshape(model.n,len(model.grid),len(model.flux));where={tuple(v):i for i,v in enumerate(model.flux)};ranks=[]
    for n in (2,-1,-3):
        for edge in (0,2):
            shift=n*model.profile[edge];indices=[i for i,m in enumerate(model.flux) if tuple(m+shift) in where]
            ranks.append(dict(integer_charge=n,physical_charge=n/3,edge=edge,shift=shift.tolist(),retained_shift_rank=len(indices),flux_dimension=len(model.flux),state_field_shift_retained_norm2=float(np.vdot(x[:,:,indices],x[:,:,indices]).real)))
    legs=[]
    for part in model.parts:
        if part['edge'] not in (0,2) or part['integer_charge'] not in (2,-1,-3):continue
        for backward in (False,True):
            a=part['c0'].conj().T if backward else part['c0'];b=part['cphi'].conj().T if backward else part['cphi'];shift=np.asarray(part['shift'])*(-1 if backward else 1)
            y=(a@x.reshape(model.n,-1)).reshape(x.shape)
            if b.nnz:y+=(b@x.reshape(model.n,-1)).reshape(x.shape)*model.grid[None,:,None]
            valid=[i for i,m in enumerate(model.flux) if tuple(m+shift) in where]
            raw=float(np.vdot(y,y).real);kept=float(np.vdot(y[:,:,valid],y[:,:,valid]).real)
            legs.append(dict(edge=part['edge'],category=part['category'],integer_charge=part['integer_charge'],backward=backward,shift=shift.tolist(),selected_body_full_flux_virtual_norm2=raw,retained_native_hop_image_norm2=kept,field_cut_virtual_lost_norm2=raw-kept))
    W=np.asarray(model.geometry['Wilson_transverse_shift']);Wvalid=[i for i,m in enumerate(model.flux) if tuple(m+W) in where]
    loops=[]
    for charge in (2,-1,-3):
        for cycle in model.geometry['actual_cycle_profiles']:
            shift=charge*np.asarray(cycle);indices=[i for i,m in enumerate(model.flux) if tuple(m+shift) in where]
            loops.append(dict(integer_charge=charge,shift=shift.tolist(),retained_shift_rank=len(indices),state_field_shift_retained_norm2=float(np.vdot(x[:,:,indices],x[:,:,indices]).real)))
    white=[v for v in legs if v['edge']==2 and v['integer_charge']==-3]
    assert all(r['retained_shift_rank']>0 for r in ranks) and sum(v['retained_native_hop_image_norm2'] for v in white)>1e-12
    return dict(primitive_harmonics=ranks,selected_state_virtual_legs=legs,Wilson_shift=W.tolist(),Wilson_rank=len(Wvalid),Wilson_field_image_norm2=float(np.vdot(x[:,:,Wvalid],x[:,:,Wvalid]).real),loop_harmonics=loops,WRONG_Ward_implies_support_rejected='An identically-zero white shift would have zero image norm and vacuously satisfy its own Ward equation; actual native white edge2 images above are nonzero. Structural support and actual state weight are checked separately, not inferred from clamped loops.')


def identities(model,state):
    H=model.hamiltonian();hx=H@state;x=state.reshape(model.n,len(model.grid),len(model.flux));hs=hx.reshape(x.shape);errors={};rhs=[np.zeros_like(state),np.zeros_like(state)];density=[np.zeros_like(state) for _ in range(4)]
    for edge,(start,end) in enumerate(model.geometry['matter_edges']):
        current=sum((model.observer('current',species=species,edge=edge,source_species='white',source_edge=2)@state for species in ('quark','white')),start=np.zeros_like(state))
        density[start]-=current;density[end]+=current
        for axis in range(2):rhs[axis]+=model.profile[edge,axis]*current
    wrong=[]
    for axis in range(2):
        G=model.flux[None,None,:,axis]/3
        comm=(1j*((H@(x*G).reshape(-1)).reshape(x.shape)-hs*G)).reshape(-1)
        force=(model.observer('force',axis=axis)@state)/3
        scale=max(np.linalg.norm(comm),np.linalg.norm(rhs[axis]),1e-15)
        errors['physical_generator_current_'+str(axis)]=float(np.linalg.norm(comm-rhs[axis])/scale)
        errors['canonical_force_'+str(axis)]=float(np.linalg.norm(force-comm)/scale)
        wrong.append(float(np.linalg.norm(3*comm-rhs[axis])/scale))
    for site in range(4):
        q=model.q[:,site,None,None];comm=(1j*((H@(x*q).reshape(-1)).reshape(x.shape)-hs*q)).reshape(-1)
        errors['Gauss_charge_continuity_'+str(site)]=float(np.linalg.norm(comm-density[site])/max(np.linalg.norm(comm),1e-15))
    h=2e-5;Hp=model.hamiltonian(theta=h);Hm=model.hamiltonian(theta=-h);J=model.observer('current',species='white',edge=2);K=model.observer('contact',species='white',edge=2)
    fd=(Hp@state-Hm@state)/(2*h);contact=(Hp@state-2*hx+Hm@state)/(h*h)
    errors['fixed_source_current_fd']=float(np.linalg.norm(fd+J@state)/max(np.linalg.norm(J@state),1e-15))
    errors['fixed_source_contact_fd']=float(np.linalg.norm(contact-K@state)/max(np.linalg.norm(K@state),1e-15))
    assert max(v for k,v in errors.items() if k!='fixed_source_contact_fd')<2e-7 and errors['fixed_source_contact_fd']<2e-3
    assert min(wrong)>1.9
    q_single=np.array([3.,0,0,0]);single_residual=q_single@model.lp@model.d@model.d.T-q_single
    return dict(errors=errors,WRONG_unnormalized_m_vs_m_over3_relative_errors=wrong,isolated_integer_charge3_Gauss_residual=single_residual.tolist(),isolated_physical_charge1_Gauss_residual_norm=float(np.linalg.norm(single_residual/3)),meaning='Physical G=m/3 and Q=Ninteger/3. Current is actual external physical-phase derivative, force is i[H,m]/3, not relabelled oscillator momentum. Closed Gauss requires total Q0; isolated B/L controls are not legal in these blocks.')


def trajectory(model,initial,config,out,control):
    theta=config['theta'];beta=config['physical_generator_kick_beta'];axis=config['kick_axis'];H=model.hamiltonian(decoupled=control);Hp=model.hamiltonian(decoupled=control,theta=theta)
    kicked=(initial.reshape(model.n,len(model.grid),len(model.flux))*np.exp(1j*beta*model.flux[None,None,:,axis]/3)).reshape(-1)
    activation=float(np.linalg.norm(Hp@initial-H@initial));assert activation>1e-10
    state=np.column_stack((initial,kicked));pulse=initial.copy();events={k:[] for k in ('baseline','field_kick','white_pulse')};previous=0.;energies={};arrays={}
    for index,t in enumerate(config['times']):
        if t>previous:
            state=expm_multiply((-1j*(t-previous))*H,state,traceA=(-1j*(t-previous))*H.trace())
            pulse=expm_multiply((-1j*(t-previous))*Hp,pulse,traceA=(-1j*(t-previous))*Hp.trace())
        for label,value,operator,phase in (('baseline',state[:,0],H,0.),('field_kick',state[:,1],H,0.),('white_pulse',pulse,Hp,theta)):
            event=observe(model,value,phase,control);event['time']=t;event['own_H_energy']=float(np.vdot(value,operator@value).real/event['norm2']);events[label].append(event)
            if index==0:energies[label]=event['own_H_energy']
            event['energy_drift_from_own_H_initial']=event['own_H_energy']-energies[label]
            if index in (0,len(config['times'])-1):arrays[label+'_t'+str(index)]=value.copy()
        Path(out+'_checkpoint.json').write_text(json.dumps(dict(events=events,completed_time=t,active_source_norm=activation,transverse_identity=control),indent=2)+'\n')
        np.savez_compressed(out+'_raw.npz',**arrays);previous=t
    responses={label:[dict(time=row['time'],response=difference(row,base)) for row,base in zip(events[label],events['baseline'])] for label in ('field_kick','white_pulse')}
    return dict(events=events,same_age_responses=responses,active_white_source_norm=activation,initial_field_kick_work=events['field_kick'][0]['own_H_energy']-events['baseline'][0]['own_H_energy'],physical_generator_beta=beta,canonical_angle_translation_beta_over3=beta/3,transverse_identity=control)


def main():
    config=json.loads(Path(sys.argv[1]).read_text());stem=Path(sys.argv[1]).name.removesuffix('_input.json');started=time.time()
    sys.path.insert(0,config['frozen_source_bundle'])
    import rich_common_quantum_em_20261016_operator as frozen
    if config['frozen_source_bundle'] not in frozen.__file__:raise ValueError('physics was not imported from frozen bundle')
    neutral=frozen.Model(config['neutral_prefix']);pair=frozen.Model(config['pair_prefix'])
    S=sparse.load_npz(config['source_prefix']+'_source_orth.npz');weights=np.asarray(S.conj().multiply(S).sum(axis=0)).real.ravel();seed_index=int(np.argmax(weights));assert weights[seed_index]>0
    values,vectors=np.linalg.eigh(neutral.hphi);phi=vectors[:,0];field=np.zeros(len(neutral.flux),complex);where={tuple(m):i for i,m in enumerate(neutral.flux)}
    for m,c in zip(config['initial_fluxes'],config['initial_flux_amplitudes']):field[where[tuple(m)]]=complex(*c)
    field,_=normalized(field);seed=np.zeros((neutral.n,len(neutral.grid),len(neutral.flux)),complex);seed[seed_index]=phi[:,None]*field[None,:];seed=seed.reshape(-1)
    Hn=neutral.hamiltonian();reference=expm_multiply((-1j*config['preparation_time'])*Hn,seed,traceA=(-1j*config['preparation_time'])*Hn.trace())
    charged,source=source_pair(neutral,pair,reference,config['source_prefix'])
    np.savez_compressed(ROOT/(stem+'_own_preparation.npz'),seed=seed,neutral=reference,pair=charged,neutral_seed_index=seed_index,initial_field=field,initial_Phi_coefficients=phi)
    result=dict(kind='OWN_FRESH_FIRST_COMMON_GAUSS_COVER_RECIPROCAL_FINITE_CONSUMER',config=config,neutral_seed_index=seed_index,seed_counts=neutral.counts[seed_index].tolist(),own_scalar_only_eigenvalue=float(values[0]),source=source,models=dict(neutral=neutral.receipt,pair=pair.receipt),Gauss_basis_errors={name:float(np.max(abs(model.longitudinal@model.d.T-model.integer_q))) for name,model in (('neutral',neutral),('pair',pair))},operator_identities=identities(pair,charged),initial_native_support=support(pair,charged),sectors={})
    for name,model,state in (('neutral',neutral,reference),('pair',pair,charged)):
        result['sectors'][name]={}
        for control in (False,True):
            mode='active_transverse_identity' if control else 'full_quantum_connection';out=str(ROOT/(stem+'_'+name+'_'+mode))
            result['sectors'][name][mode]=trajectory(model,state,config,out,control)
            (ROOT/(stem+'_partial_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    result['added_source_double_contrast']={}
    for mode in ('full_quantum_connection','active_transverse_identity'):
        result['added_source_double_contrast'][mode]={label:[dict(time=p['time'],response=contrast(p['response'],n['response'])) for p,n in zip(result['sectors']['pair'][mode]['same_age_responses'][label],result['sectors']['neutral'][mode]['same_age_responses'][label])] for label in ('field_kick','white_pulse')}
    result['final_native_support']=support(pair,np.load(str(ROOT/(stem+'_pair_full_quantum_connection_raw.npz')))['baseline_t2'])
    rows=result['sectors']['pair'];full=rows['full_quantum_connection']['same_age_responses'];off=rows['active_transverse_identity']['same_age_responses']
    checks=dict(white_pulse_changes_physical_generator=float(np.linalg.norm(full['white_pulse'][-1]['response']['physical_transverse_generator'])),physical_field_kick_changes_matter_charge=float(np.linalg.norm(full['field_kick'][-1]['response']['quark_Q_site'])),active_identity_white_to_field=float(np.linalg.norm(off['white_pulse'][-1]['response']['physical_transverse_generator'])),active_identity_field_to_matter=float(np.linalg.norm(off['field_kick'][-1]['response']['quark_Q_site'])),residual_white_to_quark_other_channels=float(np.linalg.norm(off['white_pulse'][-1]['response']['quark_Q_site'])))
    result['physical_checks']=checks;result['elapsed_seconds']=time.time()-started
    result['meaning']='Own bounded nonstationary Gauss-valid source preparation, not author stationary/evolved states. Added-source doublecontrast is NOT connected B/L cumulant or scattering. Active transverse-identity control retains white source and other rich channels; residual white-to-quark effect is not all EM. Fundamental selected CAR64/Phi7/R10/PW19 body law, two Cartan-cover modes, not independent integer links or lab photon/mass/primitive-charge/spin closure.'
    (ROOT/(stem+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    assert checks['white_pulse_changes_physical_generator']>1e-10 and checks['physical_field_kick_changes_matter_charge']>1e-10
    assert checks['active_identity_white_to_field']<1e-9 and checks['active_identity_field_to_matter']<1e-9
    print(json.dumps(dict(kind=result['kind'],physical_checks=checks,elapsed_seconds=result['elapsed_seconds'])),flush=True)

if __name__=='__main__':main()

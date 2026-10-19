"""NEW same-span direct-TT physical consumer; no old H skew/checker replay.
Independent native spin probabilities contract canonical local projectors through TT,
not CP/Gram sandwiches or eigenprojectors of compressed spin.
"""
from itertools import product
from pathlib import Path
from typing import Any
import hashlib,json,resource,sys,time
import numpy as np
from scipy import sparse
from scipy.linalg import expm,svd
from scipy.sparse.linalg import eigsh,expm_multiply
from rich_spatial_orthogonal_20261015_model import build,source_matrices,neutral_occupation,rotation_structure,install_operator
from rich_spatial_translation_20261015_model import packet,source_state,apply_terms
from rich_spatial_helicity_20261015_model import field_structure,field_drive
ROOT=Path(__file__).resolve().parent


def real(x: Any)->float:return float(np.real(x))


def cjson(x: Any)->Any:return np.stack([np.real(x),np.imag(x)],axis=-1).tolist()


def sha(path: Path)->str:
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def fold(core: Any,e: Any,operator: Any=None)->Any:
    weighted=np.tensordot(e,core,axes=(1,0))
    if operator is not None:weighted=np.einsum('pq,lqj->lpj',operator,weighted,optimize=True)
    return core.reshape(-1,core.shape[-1]).conj().T@weighted.reshape(-1,core.shape[-1])


def direct_sum(train: Any,local: Any,wrong_prefix_identity: bool=False)->Any:
    norm=np.ones((1,1),complex);result=np.zeros((1,1),complex)
    for core,site in zip(train['cores'],train['order']):
        result=fold(core,result)+fold(core,np.eye(core.shape[0]) if wrong_prefix_identity else norm,local[site]);norm=fold(core,norm)
    return result


def basis_evidence(model: Any)->Any:
    result={};worst_local=0.;worst_reconstruction=0.
    for local in model['locals']:
        for data in local.values():worst_local=max(worst_local,float(np.linalg.norm(data['frame'].conj().T@data['frame']-np.eye(data['frame'].shape[1]))));worst_reconstruction=max(worst_reconstruction,data['reconstruction_norm'])
    for name in ['neutral','positive','negative']:
        rows=[]
        for b,t in enumerate(model[name]['trains']):
            coordinate_norm=np.ones((1,1),complex);physical_norm=np.ones((1,1),complex)
            for core,site in zip(t['cores'],t['order']):
                number=6 if name=='neutral' or site!=b else 7 if name=='positive' else 5;u=model['locals'][site][number]['frame']
                coordinate_norm=fold(core,coordinate_norm);physical_norm=fold(core,physical_norm,u.conj().T@u)
            rows.append(dict(rank=t['rank'],smallest_state_singular=t['smallest_state_singular'],TT_coordinate_norm_defect=float(np.linalg.norm(coordinate_norm-np.eye(t['rank']))),actual_physical_Gram_defect=float(np.linalg.norm(physical_norm-np.eye(t['rank']))),right_inverse_defect=float(np.linalg.norm(t['R']@t['inverse']-np.eye(t['rank']))),discarded_singular_sum=sum(s['discarded_norm'] for s in t['steps'])))
        result[name]=rows
    return dict(local_isometry_defect_max=worst_local,local_reconstruction_max=worst_reconstruction,sectors=result,meaning='Actual Fock-frame Gram included in physical TT norm; no identity assumption, polar replacement, rank loss or old Gram-H reconstruction.')


def quality(model: Any,name: str)->Any:
    sec=model[name];ks=[k.toarray() for k in model['field']['coupling']];boson=model['field']['boson'].toarray()-model['offset']*np.eye(model['field']['dimension']);norm2=0.;skew2=0.
    def squared(fs: Any,operators: Any)->float:
        a=np.array([f.ravel() for f in fs]);b=np.array([k.ravel() for k in operators]);return real(np.sum((a.conj()@a.T)*(b.conj()@b.T)))
    for n,components in zip(sec['sizes'],sec['matrices']):
        fs=list(components)+[np.eye(n)];terms=ks+[boson];norm2+=squared(fs,terms)
        skew2+=squared([f-f.conj().T for f in fs]+[f.conj().T for f in fs],terms+[k-k.conj().T for k in terms])
    return dict(H_Frobenius=float(np.sqrt(norm2)),skew_Frobenius=float(np.sqrt(skew2)),relative_skew=float(np.sqrt(skew2/norm2)),method='Small direct Kronecker Grams of explicit skew terms; no dense joint H or subtraction of large H norm squares.')


def gauge_wrongcase(model: Any)->Any:
    t=model['neutral']['trains'][0];local=[x[6]['operators'][1] for x in model['locals']];base=direct_sum(t,local);cores=list(t['cores']);bond=cores[2].shape[-1];d=np.diag(np.linspace(.7,1.5,bond));di=np.diag(1/np.diag(d));cores[2]=np.einsum('apr,rs->aps',cores[2],d);cores[3]=np.einsum('ab,bpr->apr',di,cores[3]);changed=dict(t,cores=cores)
    correct=direct_sum(changed,local);wrong=direct_sum(changed,local,True)
    return dict(same_state_internal_bond_gauge_operator_error=float(np.linalg.norm(correct-base)),wrong_identity_prefix_operator_error=float(np.linalg.norm(wrong-base)),meaning='Non-isometric internal bond gauge leaves physical TT states/span unchanged. Correct prefix norm+operator environments are invariant; assuming identity at each prefix is a meaningful wrongcase. No physical coupling change.')


def spin_native(model: Any,powers: int=3)->Any:
    """Direct native local spin/T products before projection; prefix Fourier TT."""
    ops={};prefix={};points=56;phases=np.exp(2j*np.pi*np.outer(np.arange(points),np.arange(7))/points)
    for site,local in enumerate(model['locals']):
        for number in [6,7]:
            u=local[number]['frame'];masks=__import__('rich_spatial_collective_20261015_model').exterior_basis(number)[1];up=np.array([sum(bool(mask&(1<<i)) for i in range(0,12,2)) for mask in masks]);translated=u;rows=[]
            for power in range(powers):
                rows.append(np.array([u[up==n].conj().T@translated[up==n] for n in range(7)]));translated=model['local_translations'][number]@translated
            ops[site,number]=rows
    for b,t in enumerate(model['positive']['trains']):
        values=[]
        for power in range(powers):
            evalues=[]
            for k in range(points):
                e=np.ones((1,1),complex)
                for core,site in zip(t['cores'][:-1],t['order'][:-1]):e=fold(core,e,np.einsum('u,uij->ij',phases[k],ops[site,6][power]))
                evalues.append(e)
            values.append(np.array(evalues))
        prefix[b]=values
    return dict(ops=ops,prefix=prefix,phases=phases)


def spin_distribution(model: Any,receiver: Any,state: Any,structure: Any,full_joint: bool=False)->Any:
    points=56;channels=structure['channels'] if full_joint else [dict(angular_z=0,momentum=0,vectors=np.eye(model['field']['shell_dimension']))];powers=3 if full_joint else 1;character=np.zeros((points,powers,len(channels)),complex);sec=model['positive']
    for b,t in enumerate(sec['trains']):
        first,last=sec['offsets'][b:b+2];a=state[first:last].reshape(t['rank'],model['field']['nphi'],model['field']['shell_dimension']);core=t['cores'][-1]
        for c,channel in enumerate(channels):
            conditional=np.einsum('rpe,ej->rpj',a,channel['vectors'].conj()).reshape(t['rank'],-1);last_wave=np.einsum('apr,rf->apf',core,conditional,optimize=True)
            for power in range(powers):
                bonds=[]
                for op in receiver['ops'][b,7][power]:
                    acted=np.einsum('pq,aqf->apf',op,last_wave,optimize=True);bonds.append(last_wave.reshape(core.shape[0],-1).conj()@acted.reshape(core.shape[0],-1).T)
                overlaps=np.einsum('kab,uab->ku',receiver['prefix'][b][power],np.array(bonds),optimize=True);character[:,power,c]+=np.sum(receiver['phases']*overlaps,axis=1)
    spinchar=np.fft.fft(character,axis=0)/points;bychannel=np.fft.ifft(spinchar,axis=1) if full_joint else spinchar;spin=np.real(bychannel.sum(axis=(1,2)));values=np.arange(56)-27.5
    result=dict(probabilities=spin.tolist(),spin_values=values.tolist(),mean=float(values@spin),second_moment=float(values**2@spin),variance=float(values**2@spin-(values@spin)**2),outside_single_half=float(spin[abs(values)>.5].sum()),raw_sum=float(spin.sum()),minimum_unclipped=float(np.real(bychannel).min()),imaginary_max=float(abs(np.imag(bychannel)).max()))
    if full_joint:
        joint=np.zeros((56,3,5,3),complex)
        for c,channel in enumerate(channels):joint[:,:,channel['angular_z']+2,channel['momentum']]+=bychannel[:,:,c]
        result['joint_spin_matter_momentum_field_angular_field_momentum']=np.real(joint).tolist()
    return result


def fieldact(model: Any,op: Any,state: Any)->Any:return (op@state.reshape(sum(model['positive']['sizes']),-1).T).T.ravel()


def rotate(model: Any,rotations: Any,name: str,state: Any)->Any:
    sec=model[name];data=rotations[name];out=np.zeros_like(state)
    for b,n in enumerate(sec['sizes']):
        first,last=sec['offsets'][b:b+2];r=data['mapping'][b];start,stop=sec['offsets'][r:r+2];a=data['matrices'][b]@state[first:last].reshape(n,-1);out[start:stop]=(rotations['field']@a.T).T.ravel()
    return out


def rotation_quality(model: Any,rotations: Any,name: str,state: Any)->Any:
    sec=model[name];data=rotations[name];order=0.;unitarity=0.
    for b,matrix in enumerate(data['matrices']):
        unitary=matrix.conj().T@matrix;unitarity=max(unitarity,float(np.linalg.norm(unitary-np.eye(len(unitary)))));a=np.eye(matrix.shape[1],dtype=complex);site=b
        for _ in range(4):a=data['matrices'][site]@a;site=data['mapping'][site]
        order=max(order,float(np.linalg.norm(a-(-1 if name!='neutral' else 1)*np.eye(len(a)))))
    comm=sec['h']@rotate(model,rotations,name,state)-rotate(model,rotations,name,sec['h']@state)
    return dict(matter_unitarity=unitarity,fourth_power_parity=order,state_H_commutator=float(np.linalg.norm(comm)))


def current_vectors(model: Any,state: Any)->Any:
    sec=model['positive'];return [-apply_terms(sec,d,state) for d in model['field']['derivatives']]+[fieldact(model,q,state) for q in model['field']['coordinates']]


def response(model: Any,state: Any,lag: float)->Any:
    sec=model['positive'];seeds=np.column_stack([state]+current_vectors(model,state));end=expm_multiply(-1j*lag*sec['h'],seeds,traceA=-1j*lag*sec['trace']);c=np.column_stack(current_vectors(model,end[:,0])).conj().T@end[:,1:]
    return dict(lag=lag,ordered=cjson(c),signed_commutator=cjson(-1j*(c-c.conj())),contact=[[real(np.vdot(state,apply_terms(sec,d,state))) for d in row] for row in model['field']['contacts']],nonstationarity=float(np.linalg.norm(sec['h']@state-state*np.vdot(state,sec['h']@state))),meaning='New charged nonstationary finite two-time current/field/mixed8x8 response; same direct H with actual adjoint. Not stationary Lehmann, independent bare polarization or identified photon frequency.')


def field_products(model: Any,law: Any,state: Any)->Any:
    states=model['field']['shell_states'];extended=sorted([s for s in product(range(4),repeat=4) if sum(s)<=3],key=lambda s:(sum(s),s));lookup={s:i for i,s in enumerate(extended)};cut=[lookup[tuple(s)] for s in states];momenta=[]
    for axis in range(4):
        a=np.zeros((len(extended),len(extended)),complex)
        for j,s in enumerate(extended):
            if s[axis]:target=list(s);target[axis]-=1;a[lookup[tuple(target)],j]=np.sqrt(s[axis])
        momenta.append(-1j*(a-a.conj().T)/(np.sqrt(2)*law['shell_width']))
    before=sum((p@p)[np.ix_(cut,cut)] for p in momenta)/law['volume'];wrong=sum(p[np.ix_(cut,cut)]@p[np.ix_(cut,cut)] for p in momenta)/law['volume']
    direction=np.array([.2,-.3,.4,.7]);epsilon=5e-4;profiles=[];component=2
    for axis in range(3):
        for level in (range(3) if axis<2 else [0]):
            profile=np.zeros(4)
            if axis<2:profile[2*axis:2*axis+2]=law['charge']*np.array([np.cos(2*np.pi*level/3),np.sin(2*np.pi*level/3)])
            profiles.append((component,profile));component+=2
    def offset(amount: float)->Any:
        terms=list(model['field']['coupling'])
        for component,profile in profiles:
            angle=amount*float(profile@direction);c,s=terms[component:component+2];terms[component]=np.cos(angle)*c-np.sin(angle)*s;terms[component+1]=np.sin(angle)*c+np.cos(angle)*s
        return apply_terms(model['positive'],terms,state)
    forward,backward,base=offset(epsilon),offset(-epsilon),offset(0.)
    first=sum(direction[a]*apply_terms(model['positive'],d,state) for a,d in enumerate(model['field']['derivatives']))
    second=sum(direction[a]*direction[b]*apply_terms(model['positive'],d,state) for a,row in enumerate(model['field']['contacts']) for b,d in enumerate(row))
    u=np.exp(.113j*np.arange(len(state)));u/=np.linalg.norm(u);v=np.exp(.073j*np.arange(len(state)));v/=np.linalg.norm(v);h=model['positive']['h']
    return dict(beforecut_kinetic_error=float(np.linalg.norm(before-model['field']['em_kinetic'])),wrong_square_cut_momentum_error=float(np.linalg.norm(wrong-model['field']['em_kinetic'])),
        original_source_current_directional_error=float(np.linalg.norm((forward-backward)/(2*epsilon)-first)/np.linalg.norm(first)),original_source_contact_directional_error=float(np.linalg.norm((forward+backward-2*base)/epsilon**2-second)/np.linalg.norm(second)),
        actual_adjoint_bilinear_error=float(abs(np.vdot(u,h@v)-np.vdot(h.rmatvec(u),v))),meaning='Same original before-cut electric kinetic and Peierls external-source derivative/contact on direct TT H; actual adjoint called, no commutator-current substitution or old-H skew reconfirmation.')


def main()->None:
    started=time.monotonic();request:Any=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);config:Any=json.loads((ROOT/request['owner_input']).read_text());law:Any=json.loads((ROOT/config['law_input']).read_text());author:Any=json.loads((ROOT/request['owner_json']).read_text());saved=np.load(ROOT/request['owner_reconstruction']);author_states=np.load(ROOT/request['owner_npz']);model:Any=build(config,law);nb=model['field']['dimension'];neutral=model['neutral'];sec=model['positive'];binding={};maps={}
    for name in ['neutral','positive','negative']:
        differences=[];rows=[]
        for b,t in enumerate(model[name]['trains']):
            oldr=saved[f'{name}_physical_CP_R_{b}'];difference=float(np.linalg.norm(t['R']-oldr));differences.append(difference)
            if difference==0:m=np.eye(t['rank'],dtype=complex)
            else:
                u,s,vh=svd(oldr,full_matrices=False);m=t['R']@((vh.conj().T/s[None])@u.conj().T)
            rows.append(m)
        maps[name]=rows;binding[name]=dict(CP_R_rebuild_differences=differences,component_matrix_difference=float(np.linalg.norm(np.array(model[name]['matrices'])-saved[name+'_projected_components'])),
            native_cache_map_unitarity=[float(np.linalg.norm(m.conj().T@m-np.eye(len(m)))) for m in rows],
            native_component_intertwining_error=max(float(np.linalg.norm(f@m-m@g)) for fs,gs,m in zip(model[name]['matrices'],saved[name+'_projected_components'],rows) for f,g in zip(fs,gs)))
    def map_state(name: str,state: Any)->Any:
        return np.concatenate([(m@a).ravel() for m,a in zip(maps[name],state.reshape(len(maps[name]),model[name]['sizes'][0],nb))])
    seed=np.exp(.17j*np.arange(neutral['h'].shape[0]));seed/=np.linalg.norm(seed);energy,vectors=eigsh(neutral['h'],k=1,which='SA',tol=law['eigensolver_tolerance'],maxiter=5000,v0=seed);neutral['ground']=vectors[:,0];neutral['energy']=float(energy[0]);packetf=packet(model,law);sp=source_matrices(model,1,packetf);sm=source_matrices(model,-1,packetf);plus=source_state(model,1,sp);minus=source_state(model,-1,sm);pn=real(np.vdot(plus,plus));mn=real(np.vdot(minus,minus));initial=plus/np.sqrt(pn);a0=neutral['ground'].reshape(27,nb);occ=neutral_occupation(model,packetf);rawminus=real(np.vdot(a0,occ@a0));car=sum(m.conj().T@m for m in sp)+sum(m.conj().T@m for m in sm)
    binding.update(neutral_H_state_residual=float(np.linalg.norm(neutral['h']@neutral['ground']-neutral['energy']*neutral['ground'])),fresh_neutral_energy=neutral['energy'])
    binding['input_state_hashes']={name:sha(ROOT/request[name]) for name in ['owner_input','owner_json','owner_reconstruction','owner_npz']}
    structure=field_structure(model,law);initials={'baseline':initial};drives={}
    for sigma,name in [(1,'circular_plus'),(-1,'circular_minus')]:initials[name],drives[name]=field_drive(model,structure,initial,sigma,config['drive_Q_amplitude'],config['drive_P_amplitude'])
    quality_data={name:quality(model,name) for name in ['neutral','positive','negative']};basis=basis_evidence(model);wrongcase=gauge_wrongcase(model);times=np.linspace(0,request['duration'],request['time_points']);trajectories={}
    for frozen,label in [(False,'coupled'),(True,'frozen')]:
        install_operator(sec,model,frozen)
        for name,state in initials.items():trajectories[label+'_'+name]=expm_multiply(-1j*sec['h'],state,start=0,stop=times[-1],num=len(times),traceA=-1j*sec['trace'])
    install_operator(sec,model);checkpoint=output.with_name(output.name.replace('_result.json','_checkpoint.npz'));np.savez_compressed(checkpoint,neutral_ground=neutral['ground'],charged_initial=initial,**trajectories);checkpoint.with_suffix('.json').write_text(json.dumps(dict(kind='NEW18_ACTUAL_STATES_BEFORE_NATIVE_PHYSICAL_RECEIVERS',binding=binding,quality=quality_data,basis=basis,times=times.tolist()),indent=2)+'\n')
    receiver=spin_native(model);initial_spin=spin_distribution(model,receiver,initial,structure);own={name:spin_distribution(model,receiver,trajectories['coupled_'+name][-1],structure) for name in initials};frozen={};rho0=initial.reshape(1458,nb)@initial.reshape(1458,nb).conj().T
    for label in ['coupled','frozen']:
        base=trajectories[label+'_baseline'][-1].reshape(1458,nb);rho=base@base.conj().T;frozen[label]={}
        for name in ['circular_plus','circular_minus']:
            a=trajectories[label+'_'+name][-1].reshape(1458,nb);frozen[label][name]=float(np.linalg.norm(a@a.conj().T-rho))
    cached={}
    for name in initials:
        state=map_state('positive',author_states['coupled_'+name][-1]);measured=spin_distribution(model,receiver,state,structure,full_joint=name=='circular_plus');published=author['arms']['coupled_'+name][-1]['physical_receiver'];entry=dict(physical=measured,spin_L1_difference=float(np.sum(abs(np.array(measured['probabilities'])-published['spin_distribution']))))
        if name=='circular_plus':entry['full_joint_maximum_difference']=float(np.max(abs(np.array(measured['joint_spin_matter_momentum_field_angular_field_momentum'])-published['joint_spin_matter_momentum_field_angular_field_momentum'])))
        cached[name]=entry
    rotations=rotation_structure(model,structure);rotation={name:rotation_quality(model,rotations,name,state) for name,state in [('neutral',neutral['ground']),('positive',initial),('negative',minus/np.sqrt(mn))]};current={name:[-real(np.vdot(state,apply_terms(sec,d,state))) for d in model['field']['derivatives']] for name,state in initials.items()}
    response_data={name:response(model,trajectories['coupled_'+name][-1],request['response_lag']) for name in ['baseline','circular_plus']}
    result=dict(kind='NEW_INDEPENDENT_DIRECT_TT_SAME_SPAN_PHYSICAL_CONSUMER',request=request,binding=binding,quality=quality_data,basis=basis,meaningful_wrongcase=wrongcase,
        source=dict(raw_addition=1-rawminus,raw_removal=rawminus,retained_addition=pn,retained_removal=mn,omitted_addition=1-rawminus-pn,omitted_removal=rawminus-mn,packet_CAR_operator_defect=float(np.linalg.norm(car-np.eye(27)))),
        original_field_operator_evidence=field_products(model,law,initial),
        initial_spin=initial_spin,new_time=times.tolist(),new_spin=own,new_spin_L1={name:float(np.sum(abs(np.array(data['probabilities'])-own['baseline']['probabilities']))) for name,data in own.items()},matched_matter_reduced_state_difference=frozen,initial_field_only_matter_reduced_state_difference={name:float(np.linalg.norm(state.reshape(1458,nb)@state.reshape(1458,nb).conj().T-rho0)) for name,state in initials.items()},
        independently_measured_author_states=cached,rotation=rotation,initial_original_source_currents=current,nonstationary_response=response_data,
        limits=['New direct H quality is not old-matrix repair or all-Fock accuracy; old skew accepted, never recomputed.','Direct physical TT spin measurement uses full canonical count/T products BEFORE local projection, no CP/Gram operator sandwich, source-index spin or compressed-Sz eigen-PVM.','Sources/C4 one-sided coordinate maps retain explicit~1e-10/-9 uncertainty, distinct from direct H relative1e-16.','Same severe shell2/oneq/three-cell/scalar5 law, primitive CAR/frame/charge and absent longitudinal coupling; no continuous Noether, isolated particle spin, photon helicity/bulk/QCD/proton/Nature claim.'],elapsed_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(output.with_suffix('.npz'),neutral_ground=neutral['ground'],charged_initial=initial,**trajectories)
    print(json.dumps(dict(result=output.name,quality=quality_data,wrongcase=wrongcase,new_spin_L1=result['new_spin_L1'],cached_spin_L1={name:data['spin_L1_difference'] for name,data in cached.items()},peak_RSS_KiB=result['peak_RSS_KiB'],elapsed=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()

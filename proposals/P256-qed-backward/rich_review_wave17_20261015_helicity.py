"""NEW Wave17 physical spin/C3/C4 consumer of frozen helical and actual3 sources.
Spin probabilities use canonical occupation POLYNOMIAL convolution, not the author's
Fourier receiver and never the spectral PVM of compressed Sz. No source repair.
"""
from itertools import product
from pathlib import Path
from typing import Any
import json
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.linalg import eigh,expm
from scipy.sparse.linalg import expm_multiply
from rich_review_wave17_20261015_translation import (build,bind_exact_cached_frames,
    real,complex_array,digest,packet,source_matrices,source_state,apply_terms,
    install_operator,exterior_basis,multiply_except,matter_density,field_action)
ROOT=Path(__file__).resolve().parent


def field_generators(model: Any,law: Any)->Any:
    states=model['field']['shell_states'];cutoff=law['shell_cutoff'];width=law['shell_width']
    ambient=sorted([s for s in product(range(cutoff+2),repeat=4) if sum(s)<=cutoff+1],key=lambda s:(sum(s),s));lookup={s:i for i,s in enumerate(ambient)};cut=[lookup[tuple(s)] for s in states]
    q=[];p=[]
    for axis in range(4):
        a=np.zeros((len(ambient),len(ambient)),complex)
        for j,s in enumerate(ambient):
            if s[axis]:target=list(s);target[axis]-=1;a[lookup[tuple(target)],j]=np.sqrt(s[axis])
        q.append(width*(a+a.conj().T)/np.sqrt(2));p.append(-1j*(a-a.conj().T)/(width*np.sqrt(2)))
    lpol=(q[0]@p[2]-q[2]@p[0]+q[1]@p[3]-q[3]@p[1])[np.ix_(cut,cut)]
    phase=(q[0]@p[1]-q[1]@p[0]+q[2]@p[3]-q[3]@p[2])[np.ix_(cut,cut)]
    qcut=[x[np.ix_(cut,cut)] for x in q];pcut=[x[np.ix_(cut,cut)] for x in p]
    wrong=qcut[0]@pcut[2]-qcut[2]@pcut[0]+qcut[1]@pcut[3]-qcut[3]@pcut[1]
    values,vectors=eigh(lpol);channels=[]
    for angular in range(-cutoff,cutoff+1):
        sub=vectors[:,abs(values-angular)<1e-8]
        if not sub.shape[1]:continue
        ev,uv=eigh(sub.conj().T@phase@sub);joint=sub@uv
        for momentum in range(3):
            selected=[j for j,k in enumerate(ev) if int(np.rint(k))%3==momentum]
            if selected:channels.append(dict(angular=angular,momentum=momentum,vectors=joint[:,selected]))
    return dict(q=qcut,p=pcut,lpol=lpol,phase=phase,channels=channels,
        evidence=dict(beforecut_Lpol_minus_wrong_cut_products_frobenius=float(np.linalg.norm(lpol-wrong)),
            beforecut_phase_minus_owner_number_conserving_generator=float(np.linalg.norm(phase-model['field']['em_lz'])),
            polarization_translation_commutator=float(np.linalg.norm(lpol@phase-phase@lpol)),
            channel_completeness=float(np.linalg.norm(sum(c['vectors']@c['vectors'].conj().T for c in channels)-np.eye(len(states)))),
            channels=[dict(angular=c['angular'],momentum=c['momentum'],rank=c['vectors'].shape[1]) for c in channels],
            meaning='Fixed+z Lpol and sine/cos translation generators formed from FULL QPi products before shell cut. Simultaneous physical field PVM; counterpropagating channels can have the SAME fixed-axis Lpol. Not sign(totalP)*Lpol or oscillator number.'))


def drive(model: Any,field: Any,state: Any,sigma: int,config: Any)->Any:
    dq=np.array([config['drive_Q_amplitude'],0,0,sigma*config['drive_Q_amplitude']]);dp=np.array([0,config['drive_P_amplitude'],-sigma*config['drive_P_amplitude'],0])
    phase=expm(1j*sum(c*q for c,q in zip(dp,field['q'])));shift=expm(-1j*sum(c*p for c,p in zip(dq,field['p'])))
    def act(u: Any,s: Any)->Any:return field_action(model,model['positive'],sparse.kron(sparse.eye(model['field']['nphi']),u,format='csr'),s)
    return act(shift,act(phase,state)),dict(dq=dq,dp=dp,phase=phase,shift=shift,phase_state=act(phase,state),shift_state=act(shift,state),
        unitary_defect=float(np.linalg.norm((shift@phase).conj().T@(shift@phase)-np.eye(len(phase)))))


def convolution(left: Any,right: Any)->Any:
    result=np.zeros((len(left)+len(right)-1,)+left.shape[1:],complex)
    for k,value in enumerate(right):result[k:k+len(left)]+=left*value[None]
    return result


def spin_polynomials(model: Any)->Any:
    local=[];charged=[]
    for block in model['blocks']:
        entries={}
        for number,key in [(6,'v0'),(7,'vp')]:
            masks=exterior_basis(number)[1];up=np.array([sum(bool(mask&(1<<i)) for i in range(0,12,2)) for mask in masks]);v=block[key];t=model['local_translations'][number];tv=v;polys=[]
            for power in range(3):
                polys.append(np.array([v[up==n].conj().T@tv[up==n] for n in range(7)]));tv=t@tv
            entries[number]=polys
        local.append(entries[6]);charged.append(entries[7])
    outside=[];neutral=[]
    for power in range(3):
        total=np.ones((1,model['gaussian_count'],model['gaussian_count']),complex)
        for b in range(9):total=convolution(total,local[b][power])
        neutral.append(total)
    for b in range(9):
        rows=[]
        for power in range(3):
            total=np.ones((1,model['gaussian_count'],model['gaussian_count']),complex)
            for r in range(9):
                if r!=b:total=convolution(total,local[r][power])
            rows.append(total)
        outside.append(rows)
    return dict(local=local,charged=charged,outside=outside,neutral=neutral)


def physical_joint(model: Any,field: Any,polynomials: Any,state: Any,neutral: bool=False)->Any:
    sec=model['neutral' if neutral else 'positive'];cutoff=max(sum(s) for s in model['field']['shell_states']);character=np.zeros((56,3,len(field['channels'])),complex);count=model['gaussian_count']
    for b,n in enumerate(sec['sizes']):
        first,last=sec['offsets'][b:b+2];a=state[first:last].reshape(n,-1);raw=(sec['whiteners'][b]@a).reshape(-1,model['field']['nphi'],model['field']['shell_dimension'])
        for c,channel in enumerate(field['channels']):
            conditioned=np.einsum('mpe,ej->mpj',raw,channel['vectors'].conj()).reshape(len(raw),-1);rho=conditioned@conditioned.conj().T
            for power in range(3):
                if neutral:
                    coefficients=np.einsum('uij,ji->u',polynomials['neutral'][power],rho);character[:len(coefficients),power,c]+=coefficients
                else:
                    for up,coefficient in enumerate(polynomials['charged'][b][power]):
                        weighted=(coefficient*rho.T).reshape(count,6,count,6).sum(axis=(1,3));coefficients=np.einsum('uij,ij->u',polynomials['outside'][b][power],weighted)
                        character[up:up+len(coefficients),power,c]+=coefficients
    by_channel=np.fft.ifft(character,axis=1);joint=np.zeros((56,3,2*cutoff+1,3),complex)
    for c,channel in enumerate(field['channels']):joint[:,:,channel['angular']+cutoff,channel['momentum']]+=by_channel[:,:,c]
    values=np.real(joint);spin_values=np.arange(56)-(27. if neutral else 27.5);spin=values.sum(axis=(1,2,3));field_joint=values.sum(axis=(0,1));total=np.array([sum(values[:,m,:,f].sum() for m,f in product(range(3),repeat=2) if (m+f)%3==r) for r in range(3)])
    return dict(spin_values=spin_values.tolist(),joint=values.tolist(),spin_distribution=spin.tolist(),spin_mean=float(spin_values@spin),spin_second_moment=float(spin_values**2@spin),
        spin_variance=float(spin_values**2@spin-(spin_values@spin)**2),probability_outside_single_spin_half=None if neutral else float(spin[abs(spin_values)>.5].sum()),
        field_angular_momentum=field_joint.tolist(),field_angular_mean=float(np.arange(-cutoff,cutoff+1)@field_joint.sum(axis=1)),matter_momentum=values.sum(axis=(0,2,3)).tolist(),total_momentum=total.tolist(),
        raw_probability_sum=float(values.sum()),minimum_unclipped_probability=float(values.min()),maximum_imaginary_artifact=float(abs(np.imag(joint)).max()),
        meaning='Independent occupation-polynomial convolution of all N54/N55 canonical CAR spin/translation projectors BEFORE compression, tensor field angular/translation PVM. Whole occupied sea, not isolated particle or compressed-Sz eigenprojectors.')


def physical_rotations(model: Any,field: Any,neutral: bool=False)->Any:
    sec=model['neutral' if neutral else 'positive'];number,key=(6,'v0') if neutral else (7,'vp');momenta=model['momenta'];lookup={tuple(k):b for b,k in enumerate(momenta)};g=model['gindex'];result=[]
    masks6=exterior_basis(6)[1];masksn=exterior_basis(number)[1]
    spin6=np.array([sum(.5 if i%2==0 else -.5 for i in range(12) if mask&(1<<i)) for mask in masks6]);spinn=np.array([sum(.5 if i%2==0 else -.5 for i in range(12) if mask&(1<<i)) for mask in masksn])
    for power in range(4):
        def rotated(k: Any)->Any:
            x,y=k
            for _ in range(power):x,y=-y%3,x
            return (x,y)
        permutation=[lookup[rotated(k)] for k in momenta];inverse=np.argsort(permutation);phase6=np.exp(-.5j*np.pi*power*spin6);phasen=np.exp(-.5j*np.pi*power*spinn)
        local=[model['blocks'][r]['v0'].conj().T@(phase6[:,None]*model['blocks'][int(inverse[r])]['v0']) for r in range(9)];matrices=[]
        if neutral:
            x=sec['whiteners'][0];matrices=[x.conj().T@multiply_except(local,[])@x];mapping=[0]
        else:
            mapping=permutation
            for b,r in enumerate(permutation):
                raw=(model['blocks'][r][key].conj().T@(phasen[:,None]*model['blocks'][b][key]))*multiply_except(local,[r])[g[:,None],g[None,:]]
                matrices.append(sec['whiteners'][r].conj().T@raw@sec['whiteners'][b])
        result.append(dict(mapping=mapping,matrices=matrices,field=sparse.kron(sparse.eye(model['field']['nphi']),expm(-.5j*np.pi*power*field['lpol']),format='csr')))
    return result


def rotate(model: Any,rotations: Any,state: Any,power: int,neutral: bool=False)->Any:
    sec=model['neutral' if neutral else 'positive'];rotation=rotations[power];out=np.zeros_like(state,complex)
    for b,n in enumerate(sec['sizes']):
        first,last=sec['offsets'][b:b+2];a=state[first:last].reshape(n,-1);r=rotation['mapping'][b];start,stop=sec['offsets'][r:r+2]
        out[start:stop]=(rotation['field']@(rotation['matrices'][b]@a).T).T.ravel()
    return out


def angular_distribution(model: Any,rotations: Any,state: Any,neutral: bool=False)->Any:
    labels=np.arange(4)+(0 if neutral else .5);character=[np.vdot(state,rotate(model,rotations,state,power,neutral)) for power in range(4)]
    probability=np.array([sum(np.exp(.5j*np.pi*j*l)*character[l] for l in range(4))/4 for j in labels])
    return dict(classes_mod4=labels.tolist(),probabilities=np.real(probability).tolist(),raw_sum=real(sum(probability)),maximum_imaginary_artifact=float(max(abs(np.imag(probability)))),
        meaning='Full spatial+primitive-frame+field physical quarter-turn powers formed before Gaussian compression; odd Rm^4=-I. Finite angular classes only, not continuous Noether spin.')


def bare_rotation_control(model: Any)->Any:
    cells=model['geo']['cells'];lookup={tuple(r):i for i,r in enumerate(cells)};spin=np.diag(np.exp(-.5j*np.pi*np.array([.5,-.5,.5,-.5])));physical=np.zeros((108,108),complex)
    for i,(x,y,z) in enumerate(cells):
        j=lookup[(-int(y)%3,int(x),int(z))];physical[4*j:4*j+4,4*i:4*i+4]=spin
    wrong=np.kron(np.eye(27),spin);h=model['geo']['reference']
    return dict(one_body_physical_fourth_power_plus_identity=float(np.linalg.norm(np.linalg.matrix_power(physical,4)+np.eye(108))),
        physical_reference_H_commutator=float(np.linalg.norm(h@physical-physical@h)),wrong_spin_only_reference_H_commutator=float(np.linalg.norm(h@wrong-wrong@h)),
        meaning='Actual 27-site spatial permutation plus primitive spin rotation, independently in canonical108 one-body space. Wrong spin-only failure exists BEFORE Gaussian compression, not merely projected norm loss. Exterior parity is (-1)^N; frame/CAR are supplied inputs.')


def adjoint_terms(sec: Any,terms: Any,state: Any)->Any:
    result=np.zeros_like(state,complex)
    for b,n in enumerate(sec['sizes']):
        first,last=sec['offsets'][b:b+2];a=state[first:last].reshape(n,-1);out=np.zeros_like(a,complex)
        for f,k in zip(sec['matrices'][b],terms):
            if k.nnz:out+=(k.conj().T@(f.conj().T@a).T).T
        result[first:last]=out.ravel()
    return result


def adjoint_H(model: Any,sec: Any,state: Any)->Any:
    out=adjoint_terms(sec,model['field']['coupling'],state);nb=model['field']['dimension']
    for b,n in enumerate(sec['sizes']):
        first,last=sec['offsets'][b:b+2];a=state[first:last].reshape(n,nb)
        out[first:last]+=((model['field']['boson'].conj().T@a.T).T-model['offset']*a).ravel()
    return out


def operators(model: Any,state: Any,adjoint: bool=False)->Any:
    sec=model['positive'];action=adjoint_terms if adjoint else apply_terms
    return [-action(sec,d,state) for d in model['field']['derivatives']]+[field_action(model,sec,q,state) for q in model['field']['coordinates']]


def two_time(model: Any,state: Any,anchor: float,lag: float)->Any:
    sec=model['positive'];seeds=np.column_stack([state]+operators(model,state)+operators(model,state,True));propagated=expm_multiply(-1j*lag*sec['h'],seeds,traceA=-1j*lag*sec['trace'])
    regular=np.column_stack(operators(model,propagated[:,0]));adjoint=np.column_stack(operators(model,propagated[:,0],True))
    ordered=adjoint.conj().T@propagated[:,1:9];reverse=(propagated[:,9:17].conj().T@regular).T;commutator=-1j*(ordered-reverse)
    w=np.array([np.array([1,1j*d,1j*c,-c*d])/2 for c in [-1,1] for d in [-1,1]]);basis=np.zeros((8,8),complex);basis[:4,:4]=w;basis[4:,4:]=w
    circular=basis.conj()@commutator@basis.T
    return dict(anchor_time=anchor,lag=lag,ordered=complex_array(ordered),reverse_ordered=complex_array(reverse),signed_commutator=complex_array(commutator),circular_adjoint_receiver_matrix=complex_array(circular),
        exact_adjoint_minus_forced_conjugate_shortcut=float(np.linalg.norm(reverse-ordered.conj())),
        H_minus_true_adjoint_action_on_anchor=float(np.linalg.norm(sec['h']@state-adjoint_H(model,sec,state))),
        current_minus_true_adjoint_action_norms=[float(np.linalg.norm(a-b)) for a,b in zip(operators(model,state),operators(model,state,True))],
        instantaneous_current_contact=[[real(np.vdot(state,apply_terms(sec,d,state))) for d in row] for row in model['field']['contacts']],
        anchor_energy_mean=complex_array(np.vdot(state,sec['h']@state)),
        anchor_nonstationarity_action_norm=float(np.linalg.norm(sec['h']@state-state*np.vdot(state,sec['h']@state))),
        interpretation='Genuine nonstationary charged-anchor two-time current/field/mixed8x8 response. Both operator orders explicitly propagated using true matrix adjoints, not forced Hermiticity or stationary Lehmann reuse. Physical commutator interpretation inherits measured numerical non-Hermiticity; no whole-unitarity claim.')


def spin_compression_wrongcase(model: Any,state: Any,physical: Any)->Any:
    g=model['gindex'];local=[]
    for block in model['blocks']:
        masks=exterior_basis(6)[1];spin=np.array([sum(.5 if i%2==0 else -.5 for i in range(12) if mask&(1<<i)) for mask in masks]);v=block['v0'];local.append(v.conj().T@(spin[:,None]*v))
    squared=0.;mean=0.
    for b,block in enumerate(model['blocks']):
        v=block['vp'];masks=exterior_basis(7)[1];spin=np.array([sum(.5 if i%2==0 else -.5 for i in range(12) if mask&(1<<i)) for mask in masks]);gram=v.conj().T@v
        raw=(v.conj().T@(spin[:,None]*v))*multiply_except(model['grams'],[b])[g[:,None],g[None,:]]
        for r in range(9):
            if r!=b:raw+=local[r][g[:,None],g[None,:]]*gram*multiply_except(model['grams'],[b,r])[g[:,None],g[None,:]]
        x=model['positive']['whiteners'][b];s=x.conj().T@raw@x;first,last=model['positive']['offsets'][b:b+2];a=state[first:last].reshape(s.shape[0],-1);sa=s@a
        squared+=real(np.vdot(sa,sa));mean+=real(np.vdot(a,sa))
    return dict(physical_full_Sz_mean=physical['spin_mean'],compressed_Sz_first_moment=mean,physical_full_Sz_second_moment=physical['spin_second_moment'],
        wrong_compressed_Sz_squared_expectation=squared,missing_spin_complement_weight=physical['spin_second_moment']-squared,
        meaning='P Sz^2 P differs from (P Sz P)^2. First-moment compressed operator is valid; its eigenprojectors are NOT the physical ambient spin PVM. No source Pauli-label measurement substituted.')


def shifted_current(model: Any,law: Any,state: Any,dq: Any)->Any:
    nodes,weights=np.polynomial.hermite.hermgauss(32);poly=np.array([np.ones_like(nodes),np.sqrt(2)*nodes,(4*nodes*nodes-2)/np.sqrt(8)]);indices=np.array(model['field']['shell_states']);nb=model['field']['dimension'];terms=[[sparse.csr_matrix((nb,nb)) for _ in range(16)] for _ in range(4)];component=2
    for axis in range(3):
        for level in (range(3) if axis<2 else [0]):
            profile=np.zeros(4)
            if axis<2:profile[2*axis:2*axis+2]=law['charge']*np.array([np.cos(2*np.pi*level/3),np.sin(2*np.pi*level/3)])
            d=np.ones((len(indices),len(indices)),complex)*np.exp(1j*float(profile@dq))
            for a,c in enumerate(profile):
                integral=(poly*(weights*np.exp(1j*c*law['shell_width']*nodes))[None])@poly.T/np.sqrt(np.pi);d*=integral[indices[:,a,None],indices[None,:,a]]
            cosine=sparse.kron(sparse.eye(model['field']['nphi']),(d+d.conj().T)/2,format='csr');sine=sparse.kron(sparse.eye(model['field']['nphi']),(d-d.conj().T)/(2j),format='csr')
            for a,c in enumerate(profile):terms[a][component]=-c*sine;terms[a][component+1]=c*cosine
            component+=2
    return np.array([-real(np.vdot(state,apply_terms(model['positive'],d,state))) for d in terms])


def neutral_raw_stage(model: Any)->Any:
    count=model['gaussian_count'];raw=np.zeros((16,count,count),complex);local_skew=[]
    for b,block in enumerate(model['blocks']):
        v=block['v0'];matrices=np.array([v.conj().T@(op@v) for op in block['operators'][6]]);raw+=matrices*multiply_except(model['grams'],[b])[None]
        local_skew.append(max(float(np.linalg.norm((op-op.conj().T).data)) for op in block['operators'][6]))
    x=model['neutral']['whiteners'][0];white=model['neutral']['matrices'][0];transport=np.array([x.conj().T@(f-f.conj().T)@x for f in raw]);skew=white-np.swapaxes(white.conj(),-1,-2)
    def tensor_norm(fs: Any,ks: Any)->float:
        flatf=np.array([f.ravel() for f in fs]);flatk=np.array([k.ravel() for k in ks]);value=np.sum((flatf.conj()@flatf.T)*(flatk.conj()@flatk.T));return float(np.sqrt(real(value)))
    fs=list(white)+[np.eye(len(white[0]))];ks=[k.toarray() for k in model['field']['coupling']]+[(model['field']['boson']-model['offset']*sparse.eye(model['field']['dimension'])).toarray()]
    ds=[f-f.conj().T for f in fs]+[f.conj().T for f in fs];dk=ks+[k-k.conj().T for k in ks]
    norm=tensor_norm(fs,ks);defect=tensor_norm(ds,dk)
    return dict(local_canonical_operator_skew_maxima=local_skew,raw_Gaussian_F_skew_frobenius=[float(np.linalg.norm(f-f.conj().T)) for f in raw],
        raw_skew_transport_through_saved_X=[float(np.linalg.norm(f)) for f in transport],white_F_skew_frobenius=[float(np.linalg.norm(f)) for f in skew],
        congruence_skew_minus_transported_raw_skew=[float(np.linalg.norm(a-b)) for a,b in zip(skew,transport)],
        neutral_H_frobenius_via_Kronecker_Gram=norm,neutral_H_skew_frobenius_via_explicit_skew_terms=defect,neutral_H_relative_skew_frobenius=defect/norm,
        meaning='Bounded existing-source stage/action control, no repair. Splits raw canonical/assembly skew, transported raw skew and numerical congruence remainder. Supports conditioning-compatible numerical attribution, not an exhaustive assembly/root-cause proof or trajectory error bound.')


def main()->None:
    started=time.monotonic();request:Any=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);protocol:Any=json.loads((ROOT/request['owner_input']).read_text());law:Any=json.loads((ROOT/protocol['law_input']).read_text());owner:Any=json.loads((ROOT/request['owner_json']).read_text());cache=np.load(ROOT/protocol['cache_states']);author_states=np.load(ROOT/request['owner_npz'])
    model:Any=build(law);frames=bind_exact_cached_frames(model,cache);neutral=model['neutral'];sec=model['positive'];neutral['ground']=cache['neutral_ground'];initial=cache['charged_initial'];neutral['energy']=json.loads((ROOT/protocol['cache_result']).read_text())['energies']['neutral']
    source=source_state(model,1,source_matrices(model,1,packet(model,law)));source/=np.linalg.norm(source);binding=dict(frames=frames,neutral_H_state_residual=float(np.linalg.norm(neutral['h']@neutral['ground']-neutral['energy']*neutral['ground'])),source_direct_residual=float(np.linalg.norm(initial-source)),
        owner_input_sha256=digest(ROOT/request['owner_input']),owner_json_sha256=digest(ROOT/request['owner_json']),owner_npz_sha256=digest(ROOT/request['owner_npz']))
    field=field_generators(model,law);initials={'baseline':initial};drives={}
    for sigma,name in [(1,'circular_plus'),(-1,'circular_minus')]:initials[name],drives[name]=drive(model,field,initial,sigma,protocol)
    binding['literal_author_preparation_state_residuals']={name:float(np.linalg.norm(state-author_states['coupled_'+name][0])) for name,state in initials.items()}
    stage=neutral_raw_stage(model)
    trajectories={};times=np.linspace(0,request['duration'],request['time_points'])
    if 'resume_checkpoint_npz' in request:
        previous=json.loads((ROOT/request['resume_checkpoint_json']).read_text());saved=np.load(ROOT/request['resume_checkpoint_npz'])
        for key in ['owner_input_sha256','owner_json_sha256','owner_npz_sha256']:
            if previous['binding'][key]!=binding[key]:raise ValueError('Checkpoint is not bound to the selected literal author input/cache: '+key)
        if previous['times']!=times.tolist():raise ValueError('Checkpoint trajectory times differ from requested physical times')
        trajectories={label+'_'+name:saved[label+'_'+name] for label in ['coupled','frozen'] for name in initials}
        binding['resumed_own_checkpoint']=dict(npz=request['resume_checkpoint_npz'],npz_sha256=digest(ROOT/request['resume_checkpoint_npz']),metadata=request['resume_checkpoint_json'],
            metadata_sha256=digest(ROOT/request['resume_checkpoint_json']),prepared_state_residuals={key:float(np.linalg.norm(states[0]-initials[key.split('_',1)[1]])) for key,states in trajectories.items()},
            meaning='Own previous attempt saved actual same-H trajectories before a later sparse-norm diagnostic exception. No author/old producer CLI replay and no failed-attempt approval.')
    else:
        for frozen,label in [(False,'coupled'),(True,'frozen')]:
            install_operator(sec,model,frozen)
            for name,state in initials.items():trajectories[label+'_'+name]=expm_multiply(-1j*sec['h'],state,start=0,stop=times[-1],num=len(times),traceA=-1j*sec['trace'])
    install_operator(sec,model);checkpoint_npz=output.with_name(output.name.replace('_result.json','_checkpoint.npz'));checkpoint_json=output.with_name(output.name.replace('_result.json','_checkpoint.json'))
    np.savez_compressed(checkpoint_npz,neutral_ground=neutral['ground'],**trajectories);checkpoint_json.write_text(json.dumps(dict(kind='NEW17_HELICAL_STATES_BEFORE_PHYSICAL_RECEIVERS_RESPONSE',binding=binding,times=times.tolist(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')
    polynomials=spin_polynomials(model);rotations=physical_rotations(model,field);neutral_rotations=physical_rotations(model,field,True);observations={};cached_checks={}
    first_observations={name:dict(physical=physical_joint(model,field,polynomials,state),angular=angular_distribution(model,rotations,state)) for name,state in initials.items()}
    for label,states in trajectories.items():
        observations[label]=[first_observations[label.split('_',1)[1]],dict(physical=physical_joint(model,field,polynomials,states[-1]),angular=angular_distribution(model,rotations,states[-1]))]
    for name in ['baseline','circular_plus','circular_minus']:
        state=author_states['coupled_'+name][-1];measured=physical_joint(model,field,polynomials,state);published=owner['arms']['coupled_'+name][-1]['physical_receiver']
        angular=angular_distribution(model,rotations,state);published_angular=owner['arms']['coupled_'+name][-1]['total_discrete_angular_class']['probabilities']
        cached_checks[name]=dict(physical=measured,angular=angular,full_joint_maximum_absolute_difference=float(np.max(abs(np.array(measured['joint'])-published['joint_spin_matter_momentum_field_angular_field_momentum']))),
            total_angular_class_maximum_absolute_difference=float(np.max(abs(np.array(angular['probabilities'])-published_angular))))
    comparisons={}
    for label in ['coupled','frozen']:
        base=observations[label+'_baseline'];comparisons[label]={}
        for name in ['circular_plus','circular_minus']:
            pulse=observations[label+'_'+name];comparisons[label][name]=dict(spin_L1_paired=[float(np.sum(abs(np.array(p['physical']['spin_distribution'])-b['physical']['spin_distribution']))) for b,p in zip(base,pulse)],
                spin_mean_paired=[p['physical']['spin_mean']-b['physical']['spin_mean'] for b,p in zip(base,pulse)],
                total_momentum_drift_l2=float(np.linalg.norm(np.array(pulse[-1]['physical']['total_momentum'])-pulse[0]['physical']['total_momentum'])),
                total_angular_drift_l2=float(np.linalg.norm(np.array(pulse[-1]['angular']['probabilities'])-pulse[0]['angular']['probabilities'])))
    initial_physical=observations['coupled_baseline'][0]['physical'];calibration={};j0=shifted_current(model,law,initial,np.zeros(4))
    for name,data in drives.items():
        full=initials[name];shifted=shifted_current(model,law,initial,data['dq']);shifted_phase=shifted_current(model,law,data['phase_state'],data['dq']);actual=np.array([-real(np.vdot(full,apply_terms(sec,d,full))) for d in model['field']['derivatives']])
        calibration[name]=dict(phase_and_shift_unitary_defect=data['unitary_defect'],baseline_current=j0.tolist(),beforecut_shifted_current=shifted.tolist(),finite_full_current=actual.tolist(),legitimate_coordinate_shift=(shifted-j0).tolist(),
            finite_phase_at_shift_remainder=(shifted_phase-shifted).tolist(),finite_shift_product_remainder=(actual-shifted_phase).tolist(),
            matter_reduced_state_changes={leg:float(np.linalg.norm(matter_density(model,s)-matter_density(model,initial))) for leg,s in [('phase',data['phase_state']),('shift',data['shift_state']),('full',full)]})
    responses={name:two_time(model,trajectories['coupled_'+name][-1],float(times[-1]),request['response_lag']) for name in ['baseline','circular_plus']}
    result=dict(kind='NEW_INDEPENDENT_FULL_SEA_SPIN_POLYNOMIAL_C4_FIXED_AXIS_POLARIZATION_PHYSICAL_CONSUMER',request=request,protocol=protocol,binding=binding,field=field['evidence'],bare_spatial_rotation=bare_rotation_control(model),
        initial_physical_spin=initial_physical,compressed_spin_wrongcase=spin_compression_wrongcase(model,initial,initial_physical),neutral_physical_receiver=physical_joint(model,field,polynomials,neutral['ground'],True),
        neutral_discrete_angular_class=angular_distribution(model,neutral_rotations,neutral['ground'],True),times=times.tolist(),observations=observations,comparisons=comparisons,
        independently_measured_author_final_states=cached_checks,beforecut_initial_current_calibration=calibration,nonstationary_two_time_response=responses,whole_H_bounded_source_stage=stage,
        limits=['All physical spin probabilities concern the whole occupied sea. Spin0.5 frame and CAR are primitive inputs, not derived intrinsic single-particle Noether spin.',
            'Lpol is fixed+z polarization angular/Stokes momentum; field global modes are not bulk photons, no sign(totalP)*Lpol or HO-occupation helicity substitution.',
            'C4/C8 physical space+frame+field angular classes only; no continuous SO3/Noether theorem or Compton scattering claim.',
            'Whole numerical H is NOT exactly Hermitian/unitary. Independent neutral Frobenius skew5.617985676e-4 and Gram/T/C4/POVM defects remain; selected state conservation is not a whole-operator bound. Attribution remains bounded, no source repair.',
            'Shell2, one q, three cells, primitive Clifford/frame/CAR/charge/scalar potential, zero nonuniform longitudinal coupling, large omitted-H residual and absent all-Fock/continuum convergence remain.'],
        elapsed_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(output.with_suffix('.npz'),neutral_ground=neutral['ground'],**trajectories)
    print(json.dumps(dict(result=output.name,comparisons=comparisons,initial_spin_mean=initial_physical['spin_mean'],initial_spin_variance=initial_physical['spin_variance'],cached_joint_maxima={name:x['full_joint_maximum_absolute_difference'] for name,x in cached_checks.items()},elapsed=result['elapsed_seconds'],peak_RSS_KiB=result['peak_RSS_KiB'])),flush=True)


if __name__=='__main__':main()

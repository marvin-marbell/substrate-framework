"""NEW independent actual3 consumer: ambient products, recoil, source and signed response.
No producer CLI, polar-unitary repair, H symmetrization, or source-index measurement.
"""
from itertools import product
from pathlib import Path
from typing import Any
import hashlib
import json
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.linalg import expm,lu_factor,lu_solve
from scipy.sparse.linalg import expm_multiply
from rich_spatial_translation_20261015_model import (build,exterior_basis,multiply_except,
    packet,source_matrices,source_state,apply_terms,install_operator,momentum_probabilities)
from rich_spatial_collective_20261015_calculate import neutral_one_body
ROOT=Path(__file__).resolve().parent


def real(value: Any)->float:return float(np.real(value))


def complex_array(value: Any)->Any:return np.stack([np.real(value),np.imag(value)],axis=-1).tolist()


def digest(path: Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def field_action(model: Any,sec: Any,operator: Any,state: Any)->Any:
    return (operator@state.reshape(sum(sec['sizes']),model['field']['dimension']).T).T.ravel()


def bind_exact_cached_frames(model: Any,cache: Any)->Any:
    """Rebuild raw ambient congruences using SAVED frames, not a second projection."""
    grams=model['grams'];g=model['gindex'];local=[];raw_neutral=np.zeros((16,model['gaussian_count'],model['gaussian_count']),complex)
    for b,block in enumerate(model['blocks']):
        v=block['v0'];f=np.array([v.conj().T@(op@v) for op in block['operators'][6]])
        local.append(f);raw_neutral+=f*multiply_except(grams,[b])[None]
    evidence={}
    for name,number,key in [('neutral',6,'v0'),('positive',7,'vp'),('negative',5,'vm')]:
        sec=model[name];difference=[];whitening=[];matrices=[];frames=[]
        for b,fresh in enumerate(sec['whiteners']):
            x=cache[f'{name}_whitener_{b}'];frames.append(x);difference.append(float(np.linalg.norm(fresh-x)))
            whitening.append(float(np.linalg.norm(x.conj().T@sec['grams'][b]@x-np.eye(x.shape[1]))))
            if name=='neutral':raw=raw_neutral
            else:
                block=model['blocks'][b];v=block[key];localg=v.conj().T@v
                outside=multiply_except(grams,[b])[g[:,None],g[None,:]]
                raw=np.array([v.conj().T@(op@v) for op in block['operators'][number]])*outside[None]
                for r in range(9):
                    if r!=b:raw+=local[r][:,g[:,None],g[None,:]]*localg[None]*multiply_except(grams,[b,r])[g[:,None],g[None,:]][None]
            matrices.append(np.array([x.conj().T@f@x for f in raw]))
        sec['whiteners']=frames;sec['matrices']=matrices
        sec['matter_translation']=[cache[f'{name}_translation_{b}'] for b in range(len(frames))]
        install_operator(sec,model)
        evidence[name]=dict(fresh_minus_saved_whitener_frobenius=difference,saved_frame_ambient_Gram_defect=whitening,
            shapes=[list(x.shape) for x in frames],meaning='Saved actual3 frames used for ALL raw H/source/operator congruences; no fresh-frame cache relabelling.')
    return evidence


def canonical_translation_evidence(model: Any)->Any:
    out={}
    for number in [5,6,7]:
        _,masks,lookup=exterior_basis(number);t=model['local_translations'][number];wrong=0;negative=0
        for column,mask in enumerate(masks):
            occupied=[i for i in range(12) if mask&(1<<i)]
            images=[(i+4)%12 for i in occupied];ordered=sorted(images);swaps=0
            work=list(images)
            for j,target in enumerate(ordered):
                k=work.index(target,j);swaps+=k-j;work.insert(j,work.pop(k))
            expected=(-1)**swaps;row=lookup[sum(1<<i for i in ordered)]
            wrong+=int(t[row,column]!=expected);negative+=int(expected<0)
        identity=sparse.eye(len(masks),format='csr');delta=t@t@t-identity
        out[str(number)]=dict(canonical_masks=len(masks),independent_bubble_sort_sign_mismatches=wrong,
            negative_CAR_translation_signs=negative,order3_maximum_entry=float(np.max(abs(delta.data))) if delta.nnz else 0.)
    return out


def ambient_momentum_effects(model: Any,sector_name: str)->Any:
    """Pambient(T)^l is formed before Gaussian compression, including l=0 Gram."""
    sec=model[sector_name];number,key=(6,'v0') if sector_name=='neutral' else (7,'vp');g=model['gindex'];characters=[]
    powers={n:[sparse.eye(t.shape[0],format='csr'),t,t@t] for n,t in model['local_translations'].items()}
    for l in range(3):
        local=[b['v0'].conj().T@(powers[6][l]@b['v0']) for b in model['blocks']];rows=[]
        for b,x in enumerate(sec['whiteners']):
            if sector_name=='neutral':raw=multiply_except(local,[])
            else:
                v=model['blocks'][b][key];raw=(v.conj().T@(powers[number][l]@v))*multiply_except(local,[b])[g[:,None],g[None,:]]
            rows.append(x.conj().T@raw@x)
        characters.append(rows)
    effects=[[sum(np.exp(2j*np.pi*r*l/3)*characters[l][b] for l in range(3))/3 for r in range(3)] for b in range(len(sec['sizes']))]
    field=model['field']['field_translation'].toarray();field_effects=[sum(np.exp(2j*np.pi*r*l/3)*np.linalg.matrix_power(field,l) for l in range(3))/3 for r in range(3)]
    evidence=dict(maximum_raw_l1_minus_saved_translation=max(float(np.linalg.norm(characters[1][b]-sec['matter_translation'][b])) for b in range(len(sec['sizes']))),
        maximum_beforecut_l2_minus_cut_translation_squared=max(float(np.linalg.norm(characters[2][b]-sec['matter_translation'][b]@sec['matter_translation'][b])) for b in range(len(sec['sizes']))),
        maximum_completeness_Gram_defect=max(float(np.linalg.norm(sum(es)-np.eye(len(es[0])))) for es in effects),
        maximum_effect_hermiticity_defect=max(float(np.linalg.norm(e-e.conj().T)) for es in effects for e in es),
        meaning='Ambient canonical C3 projector products BEFORE compression; no clipping, Euclidean renormalization, or polar-unitary repair. Ideal PVM, finite Gram-precision defects reported.')
    return effects,field_effects,evidence


def physical_momentum(model: Any,sec: Any,state: Any,effects: Any,field_effects: Any)->Any:
    joint=np.zeros((3,3),complex)
    for b,n in enumerate(sec['sizes']):
        first,last=sec['offsets'][b:b+2];a=state[first:last].reshape(n,-1)
        for m,f in product(range(3),repeat=2):joint[m,f]+=np.vdot(a,effects[b][m]@a@field_effects[f].T)
    real_joint=np.real(joint);total=np.array([sum(real_joint[m,f] for m,f in product(range(3),repeat=2) if (m+f)%3==r) for r in range(3)])
    return dict(joint_matter_field=real_joint.tolist(),matter=real_joint.sum(axis=1).tolist(),field=real_joint.sum(axis=0).tolist(),total=total.tolist(),
        raw_probability_sum=float(np.sum(real_joint)),minimum_unclipped_probability=float(np.min(real_joint)),maximum_imaginary_artifact=float(np.max(abs(np.imag(joint)))))


def field_product_evidence(model: Any,config: Any,initial: Any)->Any:
    states=model['field']['shell_states'];extended=sorted([x for x in product(range(config['shell_cutoff']+2),repeat=4) if sum(x)<=config['shell_cutoff']+1],key=lambda x:(sum(x),x));lookup={x:i for i,x in enumerate(extended)}
    ps=[];qs=[];cut=[lookup[tuple(x)] for x in states]
    for axis in range(4):
        a=np.zeros((len(extended),len(extended)),complex)
        for j,state in enumerate(extended):
            if state[axis]:target=list(state);target[axis]-=1;a[lookup[tuple(target)],j]=np.sqrt(state[axis])
        ps.append(-1j*(a-a.conj().T)/(config['shell_width']*np.sqrt(2)));qs.append(config['shell_width']*(a+a.conj().T)/np.sqrt(2))
    before=sum((p@p)[np.ix_(cut,cut)] for p in ps)/(config['volume'])
    after=sum(p[np.ix_(cut,cut)]@p[np.ix_(cut,cut)] for p in ps)/(config['volume'])
    q0=qs[0][np.ix_(cut,cut)];unitary=expm(1j*config['kick_strength']*q0)
    # Exact Hermite integral for exp(i k Q0), independent scipy matrix exponent.
    from scipy.special import eval_genlaguerre
    from math import factorial
    displacement=np.zeros((len(states),len(states)),complex);alpha=config['kick_strength']*config['shell_width']/np.sqrt(2)
    for i,n in enumerate(states):
        for j,m in enumerate(states):
            if tuple(n[1:])==tuple(m[1:]):
                lo,hi=min(n[0],m[0]),max(n[0],m[0]);displacement[i,j]=np.exp(-alpha*alpha/2)*np.sqrt(factorial(lo)/factorial(hi))*(1j*alpha)**(hi-lo)*eval_genlaguerre(lo,hi-lo,alpha*alpha)
    pfull=sparse.kron(sparse.eye(model['field']['nphi']),displacement,format='csr');projected_phase=field_action(model,model['positive'],pfull,initial)
    return dict(beforecut_kinetic_frobenius_error=float(np.linalg.norm(before-model['field']['em_kinetic'])),wrong_cut_momentum_squared_frobenius_error=float(np.linalg.norm(after-model['field']['em_kinetic'])),
        highest_shell_wrongcase_missing_diagonal=np.real(np.diag(before-after)).tolist(),finite_Q_drive_unitarity_defect=float(np.linalg.norm(unitary.conj().T@unitary-np.eye(len(unitary)))),
        full_phase_projection_minus_finite_unitary=float(np.linalg.norm(displacement-unitary)),projected_full_phase_source_norm2=real(np.vdot(projected_phase,projected_phase)),
        meaning='Before-cut Pi squared includes omitted raising paths. A projected full phase is a contraction and is NOT the prescribed finite drive; no conditional renormalization is used.')


def peierls_source_evidence(model: Any,config: Any,state: Any)->Any:
    nodes,weights=np.polynomial.hermite.hermgauss(32);polynomial=np.array([np.ones_like(nodes),2*nodes,(4*nodes*nodes-2)/np.sqrt(8)])
    polynomial[1]/=np.sqrt(2);indices=np.array(model['field']['shell_states']);profiles=[];errors=[];wrong=[];component=2
    qs=[q.toarray()[:len(indices),:len(indices)] for q in model['field']['coordinates']]
    for axis in range(3):
        for level in (range(3) if axis<2 else [0]):
            profile=np.zeros(4)
            if axis<2:profile[2*axis:2*axis+2]=config['charge']*np.array([np.cos(2*np.pi*level/3),np.sin(2*np.pi*level/3)])
            profiles.append((component,profile));quadrature=np.ones((len(indices),len(indices)),complex)
            for a,c in enumerate(profile):
                integral=(polynomial*(weights*np.exp(1j*c*config['shell_width']*nodes))[None])@polynomial.T/np.sqrt(np.pi)
                quadrature*=integral[indices[:,a,None],indices[None,:,a]]
            full=np.kron(np.eye(model['field']['nphi']),quadrature);provided=(model['field']['coupling'][component]+1j*model['field']['coupling'][component+1]).toarray()
            errors.append(float(np.linalg.norm(full-provided)))
            wrong.append(float(np.linalg.norm(quadrature-expm(1j*sum(c*q for c,q in zip(profile,qs))))))
            component+=2
    direction=np.array([.2,-.3,.4,.7]);epsilon=5e-4
    def shifted(amount: float)->Any:
        terms=list(model['field']['coupling'])
        for component,profile in profiles:
            angle=amount*float(profile@direction);c,s=terms[component],terms[component+1]
            terms[component]=np.cos(angle)*c-np.sin(angle)*s;terms[component+1]=np.sin(angle)*c+np.cos(angle)*s
        return apply_terms(model['positive'],terms,state)
    base=shifted(0.);forward=shifted(epsilon);backward=shifted(-epsilon)
    first=sum(direction[a]*apply_terms(model['positive'],d,state) for a,d in enumerate(model['field']['derivatives']))
    second=sum(direction[a]*direction[b]*apply_terms(model['positive'],d,state) for a,row in enumerate(model['field']['contacts']) for b,d in enumerate(row))
    return dict(full_Peierls_Hermite_quadrature_frobenius_errors=errors,wrong_exponential_of_cut_Q_errors=wrong,
        external_source_direction=direction.tolist(),finite_difference_epsilon=epsilon,
        current_first_derivative_relative_action_error=float(np.linalg.norm((forward-backward)/(2*epsilon)-first)/np.linalg.norm(first)),
        full_contact_directional_second_derivative_relative_action_error=float(np.linalg.norm((forward+backward-2*base)/epsilon**2-second)/np.linalg.norm(second)),
        meaning='Independent Hermite integrals form P exp(i c.Q) P. Current/contact differentiate physical external field offsets before projection; the Gaussian/field regulator is fixed.')


def matter_density(model: Any,state: Any)->Any:
    a=state.reshape(sum(model['positive']['sizes']),-1);return a@a.conj().T


def observe(model: Any,state: Any,effects: Any,field_effects: Any)->Any:
    sec=model['positive'];a=state.reshape(sum(sec['sizes']),-1);rho=a.conj().T@a
    return dict(norm=real(np.vdot(state,state)),physical_momentum=physical_momentum(model,sec,state,effects,field_effects),
        cut_power_momentum=momentum_probabilities(model,sec,state),currents=[-real(np.vdot(state,apply_terms(sec,d,state))) for d in model['field']['derivatives']],
        contact_matrix=[[real(np.vdot(state,apply_terms(sec,d,state))) for d in row] for row in model['field']['contacts']],
        phi_mean=real(np.vdot(state,field_action(model,sec,model['field']['phi_operator'],state))),field_purity=real(np.trace(rho@rho)))


def full_retained_neutral_response(model: Any,request: Any)->Any:
    sec=model['neutral'];g=sec['ground'];energy=sec['energy'];dimension=len(g);identity=np.eye(dimension,dtype=complex)
    h=sec['h']@identity;identity=None
    vectors=[-apply_terms(sec,d,g) for d in model['field']['derivatives']]+[field_action(model,sec,q,g) for q in model['field']['coordinates']]
    v=np.column_stack([x-g*np.vdot(g,x) for x in vectors]);k=h-energy*np.eye(dimension);answers=[]
    for omega in request['response_frequencies']:
        z=omega+1j*request['eta'];minus=z*np.eye(dimension)-k;plus=z*np.eye(dimension)+k
        first=lu_solve(lu_factor(minus),v);second=lu_solve(lu_factor(plus),v)
        pos=v.conj().T@first;neg=v.conj().T@second;response=pos-neg.T
        answers.append(dict(frequency=omega,positive_ordered=complex_array(pos),negative_ordered=complex_array(neg),signed_retarded=complex_array(response),
            positive_solve_relative_residual=float(np.linalg.norm(minus@first-v)/np.linalg.norm(v)),negative_solve_relative_residual=float(np.linalg.norm(plus@second-v)/np.linalg.norm(v)),
            wrong_positive_only_error=float(np.linalg.norm(pos-response))))
    contacts=np.array([[real(np.vdot(g,apply_terms(sec,d,g))) for d in row] for row in model['field']['contacts']])
    return dict(kind='WHOLE_RETAINED_2025_NEUTRAL_RESOLVENT_NOT_48STEP_KRYLOV',dimension=dimension,order=['Jxc','Jxs','Jyc','Jys','Qxc','Qxs','Qyc','Qys'],
        unsymmetrized_H_hermiticity_frobenius_defect=float(np.linalg.norm(h-h.conj().T)),neutral_stationarity_residual=float(np.linalg.norm(h@g-energy*g)),
        raw_connected_Gram=complex_array(v.conj().T@v),instantaneous_current_contact=contacts.tolist(),responses=answers,
        meaning='Full complex current/field/mixed signed response, BOTH frequency orders, of stationary neutral retained H. No H symmetrization. Not bare polarization, no static counterterm, not nonstationary charged-packet Lehmann or continuum convergence.')


def main()->None:
    started=time.monotonic();inputpath=Path(sys.argv[1]);request:Any=json.loads(inputpath.read_text());output=Path(sys.argv[2])
    config:Any=json.loads((ROOT/request['owner_input']).read_text());owner:Any=json.loads((ROOT/request['owner_json']).read_text());cache=np.load(ROOT/request['owner_npz']);model:Any=build(config)
    frame=bind_exact_cached_frames(model,cache);neutral=model['neutral'];sec=model['positive'];neutral['ground']=cache['neutral_ground'];neutral['energy']=owner['energies']['neutral']
    packet_state=packet(model,config);plus=source_state(model,1,source_matrices(model,1,packet_state));minus=source_state(model,-1,source_matrices(model,-1,packet_state));pn=real(np.vdot(plus,plus));mn=real(np.vdot(minus,minus))
    initial=plus/np.sqrt(pn);cached_initial=cache['charged_initial'];source_overlap=np.vdot(initial,cached_initial);aligned=initial*source_overlap/abs(source_overlap)
    kicked=field_action(model,sec,model['field']['q_drive'],initial);a0=neutral['ground'].reshape(neutral['sizes'][0],-1)
    occupied=neutral_one_body(model,[np.outer(x,x.conj()) for x in packet_state]);raw_minus=real(np.vdot(a0,occupied@a0))
    binding=dict(saved_frame_reconstruction=frame,neutral_H_state_residual=float(np.linalg.norm(neutral['h']@neutral['ground']-neutral['energy']*neutral['ground'])),
        cached_source_phase_aligned_residual=float(np.linalg.norm(cached_initial-aligned)),cached_source_direct_residual=float(np.linalg.norm(cached_initial-initial)),
        cached_drive_direct_residual=float(np.linalg.norm(cache['charged_kicked']-kicked)),owner_input_sha256=digest(ROOT/request['owner_input']),owner_json_sha256=digest(ROOT/request['owner_json']),owner_npz_sha256=digest(ROOT/request['owner_npz']))
    print('Exact actual3 source/H cache bound',binding['neutral_H_state_residual'],binding['cached_source_phase_aligned_residual'],flush=True)
    trajectories={};times=np.linspace(0,request['duration'],request['time_points'])
    for frozen,label in [(False,'coupled'),(True,'frozen')]:
        install_operator(sec,model,frozen)
        for name,state in [('baseline',initial),('Q_drive',kicked)]:
            trajectories[label+'_'+name]=expm_multiply(-1j*sec['h'],state,start=0,stop=times[-1],num=len(times),traceA=-1j*sec['trace'])
    install_operator(sec,model)
    checkpoint_npz=output.with_name(output.name.replace('_result.json','_checkpoint.npz'));checkpoint_json=output.with_name(output.name.replace('_result.json','_checkpoint.json'))
    np.savez_compressed(checkpoint_npz,neutral_ground=neutral['ground'],charged_source=initial,charged_Q_drive=kicked,**trajectories)
    checkpoint_json.write_text(json.dumps(dict(kind='NEW_ACTUAL17_STATES_SAVED_BEFORE_RECEIVERS_RESPONSE',binding=binding,times=times.tolist(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')
    effects,field_effects,projector_evidence=ambient_momentum_effects(model,'positive');_,_,neutral_projector_evidence=ambient_momentum_effects(model,'neutral')
    observations={name:[observe(model,s,effects,field_effects) for s in states] for name,states in trajectories.items()};comparison={}
    for label in ['coupled','frozen']:
        base=trajectories[label+'_baseline'];pulse=trajectories[label+'_Q_drive'];ob=observations[label+'_baseline'];op=observations[label+'_Q_drive']
        comparison[label]=dict(matter_reduced_state_difference_Frobenius=[float(np.linalg.norm(matter_density(model,p)-matter_density(model,b))) for b,p in zip(base,pulse)],
            driven_total_distribution_drift_l2=[float(np.linalg.norm(np.array(x['physical_momentum']['total'])-op[0]['physical_momentum']['total'])) for x in op],
            baseline_total_distribution_drift_l2=[float(np.linalg.norm(np.array(x['physical_momentum']['total'])-ob[0]['physical_momentum']['total'])) for x in ob],
            driven_matter_distribution_change_l2=float(np.linalg.norm(np.array(op[-1]['physical_momentum']['matter'])-op[0]['physical_momentum']['matter'])),
            driven_field_distribution_change_l2=float(np.linalg.norm(np.array(op[-1]['physical_momentum']['field'])-op[0]['physical_momentum']['field'])))
    cached_final={name:physical_momentum(model,sec,cache[key][-1],effects,field_effects) for name,key in [('baseline_t5','baseline_states'),('driven_t5','kicked_states')]}
    result=dict(kind='NEW_INDEPENDENT_ACTUAL3_TRANSLATION_PHYSICAL_CONSUMER_NOT_OLD_PRODUCER_REPLAY',request=request,law_input=config,binding=binding,
        physical_source=dict(raw_addition=1-raw_minus,raw_removal=raw_minus,raw_CAR_sum=1.,retained_addition=pn,retained_removal=mn,omitted_addition=1-raw_minus-pn,omitted_removal=raw_minus-mn),
        canonical_CAR_translation=canonical_translation_evidence(model),ambient_projector_evidence=projector_evidence,neutral_ambient_projector_evidence=neutral_projector_evidence,
        beforecut_field_product=field_product_evidence(model,config,initial),times=times.tolist(),observations=observations,comparisons=comparison,
        physical_Peierls_current_contact=peierls_source_evidence(model,config,initial),
        independently_measured_cached_author_final_states=cached_final,full_current_field_response=full_retained_neutral_response(model,request),
        limits=['27 nonnull orbit directions retained at1e-12; WHOLE Gram/translation conditioning defects remain, no precision inferred from selected trajectory conservation.',
            'Whole occupied matter C3 momentum, not isolated-electron recoil. All probabilities raw/unclipped; exact ambient physical products precede compression.',
            'Q-only finite drive is unitary. Full-space Q phase commutes multiplication current; finite t0 current changes remain cutoff/product artifacts, not instantaneous matter reaction.',
            'Full neutral resolvent is stationary-source response only. It is not a Lehmann response of moving charged packet.',
            'Shell2, one q, 3cells, supplied primitive frame/CAR/scalar potential/charge, no nonuniform longitudinal field, large omitted-H residual and no all-Fock/Noether/Lorentz/photons remain.'],
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(output.with_suffix('.npz'),neutral_ground=neutral['ground'],charged_source=initial,charged_Q_drive=kicked,**trajectories)
    print(json.dumps(dict(result=output.name,source=result['physical_source'],comparisons=comparison,peak_RSS_KiB=result['peak_RSS_KiB'],elapsed=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()

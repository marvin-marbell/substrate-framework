"""Independent literal CAR/Hadamard and full physical receivers; no producer CLI."""
from itertools import product
from math import comb
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, expm, solve


def length(value):
    return float(sparse.linalg.norm(value) if sparse.issparse(value) else np.linalg.norm(value))


def mean(state, operator):
    return float(np.vdot(state, operator @ state).real)


def cj(value):
    value=np.asarray(value)
    return dict(real=value.real.tolist(),imag=value.imag.tolist())


def apply_string(mask, operations):
    amplitude=1
    for index,create in reversed(operations):
        if bool(mask & (1 << index))==create:
            return None
        amplitude*=(-1)**((mask & ((1 << index)-1)).bit_count())
        mask^=1 << index
    return mask,amplitude


def coordinate(index, members):
    f,rest=divmod(index,4*members);m,rest=divmod(rest,2*members);s,c=divmod(rest,members)
    return f,m,s,c


def mode(f,m,s,c,members):
    return 4*members*f+2*members*m+members*s+c


def hadamard(mask, members):
    states={0:1.}
    for index in reversed([i for i in range(8*members) if mask & (1 << i)]):
        f,m,s,c=coordinate(index,members);following={}
        for current,value in states.items():
            for original in [0,1]:
                step=apply_string(current,[(mode(f,original,s,c,members),True)])
                if step is not None:
                    target,sign=step
                    following[target]=following.get(target,0)+value*sign*(-1 if m==original==1 else 1)/np.sqrt(2)
        states={key:value for key,value in following.items() if abs(value)>1e-15}
    return states


def epsilon(colours):
    if len(set(colours))!=3:
        return 0
    return (-1)**sum(colours[i]>colours[j] for i in range(3) for j in range(i+1,3))


def insertion_terms(members,kind,spin):
    if kind=='lepton':
        return [([(mode(1,0,0,3,members),True)],1.)]
    cg={(0,0,1):np.sqrt(2/3),(0,1,0):-np.sqrt(1/6),(1,0,0):-np.sqrt(1/6)} if spin==.5 else {(0,0,1):1/np.sqrt(3),(0,1,0):1/np.sqrt(3),(1,0,0):1/np.sqrt(3)}
    return [([(mode(f,0,s,c,members),True) for f,s,c in zip((0,0,1),spins,colours)],epsilon(colours)*value/np.sqrt(6)) for colours in product(range(3),repeat=3) if epsilon(colours) for spins,value in cg.items()]


def act(states,terms):
    out={}
    for operations,coefficient in terms:
        for mask,value in states.items():
            step=apply_string(mask,operations)
            if step is not None:
                target,sign=step;out[target]=out.get(target,0)+coefficient*value*sign
    return {key:value for key,value in out.items() if abs(value)>1e-15}


def source_witness(model,members,kind,spin,target,source):
    neutral=model['sectors']['neutral'];sea=neutral['masks'][0]
    original=hadamard(sea,members);terms=insertion_terms(members,kind,spin)
    literal=act(original,terms);rotated=act({sea:1.},terms)
    factor=2**(-.5*(1 if kind=='lepton' else 3));predicted={}
    for mask,value in rotated.items():
        for original_mask,coefficient in hadamard(mask,members).items():
            predicted[original_mask]=predicted.get(original_mask,0)+factor*value*coefficient
    keys=sorted(set(literal)|set(predicted));aa=np.array([literal.get(mask,0) for mask in keys]);bb=np.array([predicted.get(mask,0) for mask in keys])
    raw=np.array([rotated.get(mask,0)*factor for mask in target['masks']])[:,None]
    rawmap=sparse.kron(sparse.csr_matrix(raw),sparse.eye(19),format='csr')
    projected=(target['P'].conj().T @ rawmap @ neutral['P']).toarray()
    nf=len(model['scalar']['grid']);expected=np.kron(projected,np.eye(nf))
    pp=np.zeros(4*members+1)
    for mask,value in original.items():
        count=sum(bool(mask & (1 << mode(f,0,s,c,members))) for f,s,c in product(range(2),range(2),range(members)))
        pp[count]+=abs(value)**2
    result=dict(original_reference_determinants=len(original),literal_source_determinants=len(keys),original_source_error=length(aa-bb),literal_strength=float(np.vdot(aa,aa).real),rotated_seed_norm_squared=sum(abs(v*factor)**2 for v in rotated.values()),projected_source_error=length(expected-source),Gauss_source_omission=length(rawmap @ neutral['P']-target['P'] @ sparse.csr_matrix(projected)),original_reference_pair_probabilities=pp.tolist(),binomial_error=length(pp-np.array([comb(4*members,k)/2**(4*members) for k in range(4*members+1)])))
    return result,dict(original_masks=np.array(keys,np.uint64),literal=aa,predicted=bb)


def spin_spectrum(h,spin):
    values,vectors=eigh(spin);labels=np.rint(np.sqrt(np.maximum(0,1+4*values))-1).astype(int)
    energies=[];states=[];spins=[]
    for label in np.unique(labels):
        p=vectors[:,labels==label];e,v=eigh(p.conj().T @ h @ p)
        energies.extend(e);states.append(p @ v);spins.extend([label/2]*len(e))
    order=np.argsort(energies)
    return np.asarray(energies)[order],np.column_stack(states)[:,order],np.asarray(spins)[order]


def spectral_response(h,j,k,e,v,state,frequencies,eta):
    estate=mean(state,h);delta=e-estate;weights=abs(v.conj().T @ j @ state)**2
    first=[];second=[];direct=[]
    identity=np.eye(len(e));a=j @ state
    for w in frequencies:
        first.append(sum(weights/(w-delta+1j*eta)));second.append(-sum(weights/(w+delta+1j*eta)))
        direct.append(np.vdot(a,solve((w+estate+1j*eta)*identity-h,a))-np.vdot(a,solve((w-estate+1j*eta)*identity+h,a)))
    chi=np.array(first)+second
    return dict(virtual_states=len(e),gaps=delta.tolist(),all_current_weights=weights.tolist(),sum_rule=float(weights.sum()),direct_J_squared=mean(state,j @ j),state_stationarity=length(h @ state-estate*state),both_orders=cj(chi),first=cj(first),second=cj(second),contact=mean(state,k),kernel=cj(chi+mean(state,k)),independent_resolvent_error=length(chi-np.array(direct)),wrong_positive_only_error=length(chi-np.array(first)))


def evolve(state,e,v,times):
    return np.array([v @ (np.exp(-1j*e*t)*(v.conj().T @ state)) for t in times])


def projectors(operator):
    values,vectors=eigh(operator);labels=np.round(values,8)
    return [(float(label),vectors[:,labels==label] @ vectors[:,labels==label].conj().T) for label in np.unique(labels)]


def physical_events(h,j,k,e,v,initial,op,scalar,law_at,case):
    nf=len(scalar['grid']);ng=len(h)//nf;times=case['times']
    obs={key:np.kron(value,np.eye(nf)) for key,value in op.items() if key in ['Q','N','S2','C','B','boundary','octet','matter_octet','positive']}
    obs.update(current=j,contact=k,phi=np.kron(np.eye(ng),scalar['phi']),phi2=np.kron(np.eye(ng),scalar['phi'] @ scalar['phi']),scalar_boundary=np.kron(np.eye(ng),scalar['boundary']))
    fieldU=np.kron(np.eye(ng),expm(-1j*case['field_kick']*scalar['P']))
    conditional=expm(-1j*case['field_kick']*np.kron(op['S2'],scalar['P']))
    sv,sb=eigh(op['S2']);low=sb[:,np.argmin(sv)];high=sb[:,np.argmax(sv)]
    coherence=np.kron(np.outer(low,high.conj())+np.outer(high,low.conj()),np.eye(nf))
    hp,_,_=law_at(case['pulse_angle']);preparations=dict(fieldonly=fieldU @ initial,EMprobe=expm(-1j*case['pulse_duration']*hp) @ initial)
    baseline=evolve(initial,e,v,times);base={key:[mean(s,value) for s in baseline] for key,value in obs.items()}
    pvm=[]
    for sj,sp in projectors(op['S2']):
        for lc,lp in projectors(op['C']):
            for sign in [-1,0,1]:
                fp=np.diag((np.sign(scalar['grid'])==sign).astype(float));pvm.append((dict(frameS2=sj,loopC=lc,phi_sign=sign),np.kron(sp @ lp,fp)))
    result=dict(initial_norm=length(initial),preparation_semantics='Unfiltered normalized literal source with coherent half/threehalf admixture where available. Not a pole/spin-projector-filtered event. Baseline evolves EXACT SAME initial state under SAME H0.',field_unitary_body_coherence_commutator=length(fieldU @ coherence-coherence @ fieldU),wrong_body_conditioner_commutator=length(conditional @ coherence-coherence @ conditional),PVM_commutator=length(op['S2'] @ op['C']-op['C'] @ op['S2']),baseline=base,events={});arrays=dict(baseline=baseline,initial=initial)
    for name,prepared in preparations.items():
        states=evolve(prepared,e,v,times);data={key:[mean(s,value) for s in states] for key,value in obs.items()}
        probability=abs(v.conj().T @ prepared)**2
        outcomes=[dict(**labels,probability=[mean(s,projector) for s in states]) for labels,projector in pvm]
        result['events'][name]=dict(norm=length(prepared),energy_injection=mean(prepared,h)-mean(initial,h),t0_change={key:mean(prepared,value)-mean(initial,value) for key,value in obs.items()},excess={key:(np.array(data[key])-base[key]).tolist() for key in data},inclusive_energy_probabilities=probability.tolist(),inclusive_energy_sum=float(probability.sum()),joint_PVM=outcomes,joint_PVM_sums=[sum(row['probability'][i] for row in outcomes) for i in range(len(times))])
        arrays[name+'_states']=states;arrays[name+'_prepared']=prepared
    offh,offj,offk=law_at(0.,True);oe,ov=eigh(offh);offbase=evolve(initial,oe,ov,times);offpulse=evolve(preparations['fieldonly'],oe,ov,times)
    result['same_initial_couplingoff_control']=dict(current_norm=length(offj),contact_norm=length(offk),loopC_excess=[mean(a,obs['C'])-mean(b,obs['C']) for a,b in zip(offpulse,offbase)],field_preparation_unchanged=True,meaning='Control retains actual scalar and loop kinetic/plaquette terms and SAME initial state; turns scalar-matter-loop coupling off. Not a new fitted production law.')
    arrays['couplingoff_baseline']=offbase;arrays['couplingoff_field_states']=offpulse
    return result,arrays

"""Historical helicity prefix: fixed+z polarization angular momentum, NOT photons.
Physical full-sea spin POVMs, actual cubic spatial+frame rotation and Q/P drives.
"""
from typing import Any
from itertools import product
import numpy as np
from scipy import sparse
from scipy.linalg import eigh,expm
from rich_spatial_translation_20261015_model import (build,apply_terms,install_operator,packet,
    source_matrices,source_state,translate,momentum_probabilities)
from rich_spatial_collective_20261015_model import exterior_basis,multiply_except


def field_structure(model: Any,config: Any)->Any:
    field=model['field'];states=field['shell_states'];lookup={tuple(s):i for i,s in enumerate(states)};ns=len(states)
    annihilation=[]
    for axis in range(4):
        a=np.zeros((ns,ns),complex)
        for j,state in enumerate(states):
            if state[axis]:
                target=list(state);target[axis]-=1;a[lookup[tuple(target)],j]=np.sqrt(state[axis])
        annihilation.append(a)
    width=config['shell_width'];q=[width*(a+a.conj().T)/np.sqrt(2) for a in annihilation]
    p=[-1j*(a-a.conj().T)/(width*np.sqrt(2)) for a in annihilation]
    # Number-conserving full rotation action stays within the COMPLETE shell.
    lpol=sum(1j*(annihilation[y].conj().T@annihilation[x]-annihilation[x].conj().T@annihilation[y]) for x,y in [(0,2),(1,3)])
    lphase=field['em_lz'];e,u=eigh(lpol+np.sqrt(2)*lphase);channels=[]
    for angular in range(-config['shell_cutoff'],config['shell_cutoff']+1):
        for momentum in range(3):
            indices=[]
            for j in range(ns):
                h=float(np.real(np.vdot(u[:,j],lpol@u[:,j])));k=float(np.real(np.vdot(u[:,j],lphase@u[:,j])))
                if int(np.rint(h))==angular and int(np.rint(k))%3==momentum:indices.append(j)
            if indices:channels.append(dict(angular_z=angular,momentum=momentum,vectors=u[:,indices]))
    rotation=expm(-.5j*np.pi*lpol)
    return dict(q=q,p=p,lpol=lpol,lphase=lphase,rotation=rotation,channels=channels,
        commutator_norm=float(np.linalg.norm(lpol@lphase-lphase@lpol)),
        angular_integer_variance=max(float(np.linalg.norm(lpol@u[:,j]-np.rint(np.real(np.vdot(u[:,j],lpol@u[:,j])))*u[:,j])) for j in range(ns)),
        momentum_integer_variance=max(float(np.linalg.norm(lphase@u[:,j]-np.rint(np.real(np.vdot(u[:,j],lphase@u[:,j])))*u[:,j])) for j in range(ns)))


def field_drive(model: Any,structure: Any,state: Any,chirality: int,q_amplitude: float,p_amplitude: float)->Any:
    # Profile Q=(a,0,0,sigma*a), P=(0,b,-sigma*b,0). No basis-frequency input.
    dq=np.array([q_amplitude,0,0,chirality*q_amplitude])
    dp=np.array([0,p_amplitude,-chirality*p_amplitude,0])
    shift=sum(c*p for c,p in zip(dq,structure['p']));kick=sum(c*q for c,q in zip(dp,structure['q']))
    unitary=expm(-1j*shift)@expm(1j*kick)
    a=state.reshape(sum(model['positive']['sizes']),model['field']['nphi'],model['field']['shell_dimension'])
    transformed=np.einsum('ij,mpj->mpi',unitary,a).ravel()
    return transformed,dict(chirality=chirality,Q_profile=dq.tolist(),P_profile=dp.tolist(),
        unitary_defect=float(np.linalg.norm(unitary.conj().T@unitary-np.eye(len(unitary)))),
        norm_error=abs(float(np.linalg.norm(transformed))-float(np.linalg.norm(state))),
        meaning='Finite projected field-only Q/P drive, both polarizations/sin-cos; injected distributions measured. Chirality label is profile handedness, NOT identified photon helicity.')


def rotation_structure(model: Any,structure: Any)->Any:
    momenta=model['momenta'];lookup={tuple(k):i for i,k in enumerate(momenta)};permutation=[lookup[(-ky%3,kx)] for kx,ky in momenta]
    inverse=np.argsort(permutation);n=model['gaussian_count'];g=model['gindex'];out={}
    phases={}
    for number in [5,6,7]:
        masks=exterior_basis(number)[1]
        spin=np.array([sum(.5 if i%2==0 else -.5 for i in range(12) if mask&(1<<i)) for mask in masks])
        phases[number]=np.exp(-.5j*np.pi*spin)
    for name,number,key in [('neutral',6,'v0'),('positive',7,'vp'),('negative',5,'vm')]:
        sec=model[name];correct=[];wrong=[]
        cross=[model['blocks'][r]['v0'].conj().T@(phases[6][:,None]*model['blocks'][int(inverse[r])]['v0']) for r in range(9)]
        direct=[b['v0'].conj().T@(phases[6][:,None]*b['v0']) for b in model['blocks']]
        if name=='neutral':
            x=sec['whiteners'][0];correct=[x.conj().T@multiply_except(cross,[])@x];wrong=[x.conj().T@multiply_except(direct,[])@x]
            mapping=[0]
        else:
            mapping=permutation
            for b in range(9):
                r=permutation[b];bra=model['blocks'][r][key];ket=model['blocks'][b][key]
                raw=(bra.conj().T@(phases[number][:,None]*ket))*multiply_except(cross,[r])[g[:,None],g[None,:]]
                correct.append(sec['whiteners'][r].conj().T@raw@sec['whiteners'][b])
                v=model['blocks'][b][key];raw=(v.conj().T@(phases[number][:,None]*v))*multiply_except(direct,[b])[g[:,None],g[None,:]]
                wrong.append(sec['whiteners'][b].conj().T@raw@sec['whiteners'][b])
        out[name]=dict(matrices=correct,mapping=mapping,spin_only=wrong,particle_parity=0 if name=='neutral' else 1)
    out['field']=sparse.kron(sparse.eye(model['field']['nphi']),structure['rotation'],format='csr')
    return out


def rotate(model: Any,rotations: Any,sector_name: str,state: Any,wrong: bool=False)->Any:
    sec=model[sector_name];rotation=rotations[sector_name];result=np.zeros_like(state,complex)
    matrices=rotation['spin_only' if wrong else 'matrices'];mapping=list(range(len(sec['sizes']))) if wrong else rotation['mapping']
    for b,n in enumerate(sec['sizes']):
        r=mapping[b];start,stop=sec['offsets'][b:b+2];a=state[start:stop].reshape(n,-1)
        a=matrices[b]@a;a=(rotations['field']@a.T).T
        first,last=sec['offsets'][r:r+2];result[first:last]+=a.ravel()
    return result


def rotation_evidence(model: Any,rotations: Any,sector_name: str,state: Any)->Any:
    sec=model[sector_name];rotation=rotations[sector_name];parity=rotation['particle_parity'];worst=0.;order=0.
    for b,t in enumerate(rotation['matrices']):
        worst=max(worst,float(np.linalg.norm(t.conj().T@t-np.eye(t.shape[1]))));a=np.eye(t.shape[1],dtype=complex);r=b
        for _ in range(4):a=rotation['matrices'][r]@a;r=rotation['mapping'][r]
        order=max(order,float(np.linalg.norm(a-(-1)**parity*np.eye(len(a)))))
    acted=rotate(model,rotations,sector_name,state);h=sec['h'];comm=h@acted-rotate(model,rotations,sector_name,h@state)
    wrong=rotate(model,rotations,sector_name,state,True);bad=h@wrong-rotate(model,rotations,sector_name,h@state,True)
    return dict(matter_unitarity=worst,matter_fourth_power_parity_defect=order,
        physical_joint_H_commutator=float(np.linalg.norm(comm)),relative_physical_joint_H_commutator=float(np.linalg.norm(comm)/max(np.linalg.norm(h@state),1e-30)),
        spin_only_wrong_H_commutator=float(np.linalg.norm(bad)),spin_only_projected_state_norm=float(np.linalg.norm(wrong)),
        interpretation='Actual spatial quarter-turn PLUS primitive frame spin PLUS field polarization rotation. Odd matter Rm^4=-I is a finite C8 action, not continuous Noether halfspin. Spin-only wrong control may also leak from the Gaussian span.')


def angular_class_probabilities(model: Any,rotations: Any,sector_name: str,state: Any)->Any:
    start=.5*rotations[sector_name]['particle_parity'];characteristic=[];v=state
    for l in range(4):characteristic.append(np.vdot(state,v));v=rotate(model,rotations,sector_name,v)
    labels=np.arange(4)+start
    probability=[float(np.real(sum(np.exp(.5j*np.pi*j*l)*characteristic[l] for l in range(4))/4)) for j in labels]
    return dict(angular_classes_mod4=labels.tolist(),probabilities=probability,
        meaning='TOTAL discrete cubic spatial+frame+field angular class modulo4, NOT physical continuous total Jz or isolated-particle helicity.')


def spin_receiver(model: Any,points: int=56)->Any:
    """Compression of full canonical CAR spin/translation PVM, never Pauli labels."""
    count=model['gaussian_count'];g=model['gindex'];neutral_local=[];charged_local=[]
    for block in model['blocks']:
        entries={}
        for number,key in [(6,'v0'),(7,'vp')]:
            masks=exterior_basis(number)[1];up=np.array([sum(1 for i in range(0,12,2) if mask&(1<<i)) for mask in masks]);v=block[key]
            t=model['local_translations'][number];translated=v;values=[]
            for power in range(3):
                values.append(np.array([v[up==n].conj().T@translated[up==n] for n in range(7)]));translated=t@translated
            entries[number]=np.array(values)
        neutral_local.append(entries[6]);charged_local.append(entries[7])
    phase=np.exp(2j*np.pi*np.outer(np.arange(points),np.arange(7))/points)
    neutral_fourier=[np.einsum('ku,luij->klij',phase,local) for local in neutral_local]
    neutral_char=np.empty((points,3,len(model['neutral']['whiteners'][0][0]),len(model['neutral']['whiteners'][0][0])),complex)
    xn=model['neutral']['whiteners'][0]
    for k,l in product(range(points),range(3)):
        raw=multiply_except([v[k,l] for v in neutral_fourier],[]);neutral_char[k,l]=xn.conj().T@raw@xn
    neutral_effects=np.fft.ifft(np.fft.fft(neutral_char,axis=0)/points,axis=1)
    effects=[];normalization=[];hermiticity=[]
    for b,secx in enumerate(model['positive']['whiteners']):
        ch=np.einsum('ku,luij->klij',phase,charged_local[b]);nw=secx.shape[1];character=np.empty((points,3,nw,nw),complex)
        for k,l in product(range(points),range(3)):
            outside=multiply_except([v[k,l] for v in neutral_fourier],[b])[g[:,None],g[None,:]]
            character[k,l]=secx.conj().T@(ch[k,l]*outside)@secx
        e=np.fft.ifft(np.fft.fft(character,axis=0)/points,axis=1)
        normalization.append(float(np.linalg.norm(np.sum(e,axis=(0,1))-np.eye(nw))))
        hermiticity.append(float(np.max(np.linalg.norm(e-np.swapaxes(e.conj(),-1,-2),axis=(-2,-1)))))
        effects.append(e)
    return dict(effects=effects,neutral_effects=neutral_effects,spin_values=(np.arange(points)-27.5).tolist(),neutral_spin_values=(np.arange(points)-27.).tolist(),
        normalization_defects=normalization,maximum_effect_hermiticity_defects=hermiticity,
        meaning='E_s,m=Xdag P_ambient(Sz=s,Tmatter=m) X from exact full occupied-sea CAR spin-count generating functions. Positive compression gives POVM, not spectral PVM of compressed Sz. All bins retained, no negative clipping.')


def physical_receivers(model: Any,structure: Any,receiver: Any,state: Any,neutral: bool=False)->Any:
    sec=model['neutral' if neutral else 'positive'];nspin=len(receiver['spin_values']);maximum=model['field']['shell_states'][-1]
    cutoff=max(sum(s) for s in model['field']['shell_states']);joint=np.zeros((nspin,3,2*cutoff+1,3),complex)
    for b,n in enumerate(sec['sizes']):
        first,last=sec['offsets'][b:b+2];a=state[first:last].reshape(n,model['field']['nphi'],model['field']['shell_dimension'])
        e=receiver['neutral_effects'] if neutral else receiver['effects'][b];rhos=[]
        for channel in structure['channels']:
            conditional=np.einsum('mpe,ej->mpj',a,channel['vectors'].conj()).reshape(n,-1);rhos.append(conditional@conditional.conj().T)
        values=(e.reshape(nspin*3,n*n)@np.array(rhos).transpose(0,2,1).reshape(len(rhos),n*n).T).reshape(nspin,3,-1)
        for j,channel in enumerate(structure['channels']):joint[:,:,channel['angular_z']+cutoff,channel['momentum']]+=values[:,:,j]
    real_joint=np.real(joint);spin_values=np.array(receiver['neutral_spin_values'] if neutral else receiver['spin_values']);spin_momentum=np.sum(real_joint,axis=(2,3));field_angular_momentum=np.sum(real_joint,axis=(0,1))
    return dict(spin_values=spin_values.tolist(),field_angular_values=list(range(-cutoff,cutoff+1)),
        joint_spin_matter_momentum_field_angular_field_momentum=real_joint.tolist(),
        joint_matter_spin_momentum=spin_momentum.tolist(),joint_field_polarization_angular_momentum=field_angular_momentum.tolist(),
        spin_distribution=spin_momentum.sum(axis=1).tolist(),spin_mean=float(spin_values@spin_momentum.sum(axis=1)),
        field_polarization_angular_mean=float(np.arange(-cutoff,cutoff+1)@field_angular_momentum.sum(axis=1)),
        probability_sum=float(np.sum(real_joint)),minimum_unclipped_probability=float(np.min(real_joint)),maximum_imaginary_artifact=float(np.max(abs(np.imag(joint)))),
        interpretation='Full physical occupied-sea spin/translation POVM tensor commuting field fixed+z POLARIZATION ANGULAR MOMENTUM/translation PVM. Lpol is NOT automatically particle helicity; no sign(totalP)*Lpol or auxiliary occupation identification.')

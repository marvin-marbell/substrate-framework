"""Exact full-Slater even-CAR words for the ordinary photon contour.

The full occupied and empty light spaces plus Ref4 are retained. Factored
SOURCE matrices only accelerate traces, never compress the intermediate
Hilbert space. Outside flat P, use the separate actual Q transfer, not CAR.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import itertools
from typing import Any
import numpy as np
from scipy.linalg import block_diag


@dataclass(frozen=True)
class Factor:
    left: np.ndarray
    right: np.ndarray

    def apply(self,value: np.ndarray) -> np.ndarray:
        return self.left@(self.right.conj().T@value)


@lru_cache(maxsize=6)
def patterns(order: int) -> tuple:
    rows=[]
    for p in itertools.permutations(range(order)):
        seen=set();cycles=[]
        for i in range(order):
            if i in seen:continue
            cycle=[];j=i
            while j not in seen:cycle.append(j);seen.add(j);j=p[j]
            cycles.append(cycle)
        parity=(order-len(cycles)+sum(i<p[i] for i in range(order)))%2
        rows.append(((-1)**parity,tuple(tuple(c) for c in cycles)))
    return tuple(rows)


def car_word(occupied: np.ndarray,operators: list[np.ndarray|Factor]) -> complex:
    """General ordered bilinear Wick word, including all PH intermediates.

    Contraction i->j is P for i>=j and -Q for i<j. The permutation
    determinant supplies the other fermion signs. No projected product.
    """
    n=len(occupied);P=np.asarray(occupied,dtype=bool);Q=~P
    def mask(i: int,j: int) -> np.ndarray:return P if i>=j else Q
    def trace(cycle: tuple) -> complex:
        ids=list(cycle);position=next((k for k,i in enumerate(ids) if isinstance(operators[i],Factor)),None)
        if position is None:
            if len(ids)==1:
                a=operators[ids[0]];assert isinstance(a,np.ndarray)
                return complex(np.diag(a)[P].sum())
            if len(ids)!=2:raise ValueError('Nonfactored cycles longer than2 not used by this source contour')
            i,j=ids;a,b=operators[i],operators[j];assert isinstance(a,np.ndarray) and isinstance(b,np.ndarray)
            return complex(np.einsum('ij,ji->',a[np.ix_(mask(j,i),mask(i,j))],b[np.ix_(mask(i,j),mask(j,i))]))
        ids=ids[position:]+ids[:position];source=operators[ids[0]];assert isinstance(source,Factor)
        x=source.left*mask(ids[-1],ids[0])[:,None]
        for k in range(len(ids)-1,0,-1):
            op=operators[ids[k]];x=op.apply(x) if isinstance(op,Factor) else op@x
            x=x*mask(ids[k-1],ids[k])[:,None]
        return complex(np.trace(source.right.conj().T@x))
    total=0j
    for sign,cycles in patterns(len(operators)):
        value=complex(sign)
        for cycle in cycles:value*=trace(cycle)
        total+=value
    return total


def contact_matrix(law: Any,fields: np.ndarray,links: np.ndarray,A: np.ndarray,
                   incoming_leg: np.ndarray,outgoing_leg: np.ndarray) -> np.ndarray:
    """Complex-bilinear literal second Peierls derivative, not J squared."""
    n=4*law.volume;out=np.zeros((n,n),complex)
    for x,y,b,path in law.terms(fields,links,A):
        factor=-sum(incoming_leg[e] for e in path)*sum(outgoing_leg[e] for e in path)
        sx,sy=slice(4*x,4*x+4),slice(4*y,4*y+4)
        out[sx,sy]+=factor*b;out[sy,sx]+=factor*b.conj().T
    return out


def native_closed_contour(model: Any,data: dict[str,Any],profile: str,quadrature: int=6,
                          legal_size: int=5*2**96) -> dict[str,Any]:
    """Actual conditional native SOURCE/true-g reference coefficient.

    Ordinary T6 Omega=0; full108 one-body evolution is only the EXACT
    quadratic free interaction picture. Currents/contact act in whole CAR,
    including neutral-backward and charged-forward branches. FullCoulomb
    remains in the reference; its leading-coefficient correction is priced
    separately. This point diagnostic is not a newly normalized detector or
    a LEGAL whole-state Born probability.
    """
    from functional_source_photon_bank import FunctionalSourcePhotonBank
    from reference_photon_current_scattering import directed_currents
    from prepared_neutral_changing_paired_pulse import source_point
    bank=FunctionalSourcePhotonBank(model,data);fibre=bank.fibre;n=fibre.n;T=6.
    q,em,_=source_point(model,data)
    sample=model.basis_at(np.zeros_like(q),np.zeros_like(em));point=sample['point'];A=np.zeros((model.law.volume,3))
    original=model.original.basis_at(sample['q'],sample['em'],point)
    amplitudes=original['neutral']@data['preparation_state14']
    source=fibre.source_vectors_at(point['phi'],point['links'],original['theta'],A,model.source_bank.profiles[profile])
    S=np.asarray(source['full108_plus_coefficients']);Sm=np.asarray(source['full108_minus_coefficients'])
    # X+ey is the PHOTON receiver centre, not a translation of SOURCE eta.
    # The atomic template has only the original tiny recoil; SOURCE19 stays.
    recoil=2*np.pi/ legal_size*np.array([1.,-1.,0.])
    output_profile=model.source_bank.profiles[profile]*np.exp(1j*((model.law.sites-1)@recoil))
    out_source=fibre.source_vectors_at(point['phi'],point['links'],original['theta'],A,output_profile)
    Sout=np.asarray(out_source['full108_plus_coefficients']);Tout=np.asarray(out_source['full108_minus_coefficients'])
    currents=directed_currents(model.law,point['phi'],point['links'],A,bank.velocity)
    h=currents['H'];Jedge=currents['J_original'];omega=np.asarray(model.em['omega'])
    gin,gout=np.asarray(bank.packets[0]),np.asarray(bank.packets[2])
    if abs(np.vdot(gout,gin))>1e-12:raise ValueError('Actual native incoming/outgoing modes must be orthogonal')
    # The SAME fixed SOURCE normalizer, not conditional or output whitening.
    from prepared_neutral_phi_word_channels import coefficient_channels
    _,metadata=coefficient_channels(data,'charged',profile);Z=float(metadata['global_SOURCE_normalizer'])
    time,weight=np.polynomial.legendre.leggauss(quadrature);times=(time+1)*T/2;weights=weight*T/2
    rows=[]
    for charge in (1,-1):
        O=fibre.orbitals if charge==1 else fibre.orbitals.conj()
        # Hole basis order is unchanged; occupation is complement of electron sea58.
        occupied=np.r_[np.arange(n)<fibre.sea_count if charge==1 else np.arange(n)>=fibre.sea_count+4,np.ones(4,dtype=bool)]
        hq=h if charge==1 else -h.T;Jq=Jedge if charge==1 else -Jedge.transpose(0,2,1)
        sourceq=S if charge==1 else -Sm.T
        outputq=Sout if charge==1 else -Tout.T
        hcoords=O.conj().T@hq@O;energies,modes=np.linalg.eigh(hcoords)
        def unitary(t: float) -> np.ndarray:return (modes*np.exp(-1j*energies*t))@modes.conj().T
        U=unitary(T);left=np.vstack((O.conj().T@sourceq,np.zeros((4,4),complex)))
        ref=np.vstack((np.zeros((n,4),complex),np.eye(4)))
        Bin=Factor(left,ref);Bout=Factor(np.vstack((U.conj().T@(O.conj().T@outputq),np.zeros((4,4),complex))),ref)
        Bd=Factor(Bout.right,Bout.left)
        arrays=[]
        for t,w in zip(times,weights):
            u=unitary(float(t));legin=(model.em['legs']@(gin*np.exp(-1j*omega*t))/np.sqrt(2)).reshape(model.law.volume,3)
            legout=(model.em['legs']@(gout.conj()*np.exp(-1j*omega*(T-t)))/np.sqrt(2)).reshape(model.law.volume,3)
            Ji=np.einsum('e,eij->ij',legin.ravel(),Jq);Jo=np.einsum('e,eij->ij',legout.ravel(),Jq)
            C=contact_matrix(model.law,point['phi'],point['links'],A,legin,legout)
            if charge==-1:C=-C.T
            def evolve(a: np.ndarray) -> np.ndarray:return np.asarray(block_diag(u.conj().T@(O.conj().T@a@O)@u,np.zeros((4,4))))
            arrays.append((float(t),float(w),evolve(Ji),evolve(Jo),evolve(C)))
        contact=0j;cross=0j;forward=0j
        for t,w,Ji,Jo,C in arrays:
            contact+=-1j*w*car_word(occupied,[Bd,C,Bin])
            for s,v,Jis,Jos,_ in arrays:
                cross+=w*v*car_word(occupied,[Jis,Bd,Jo,Bin])
        # Ordered time triangle, with independent Gauss rule on s in[0,t].
        for t,w,Ji,Jo,_ in arrays:
            subtime=(time+1)*t/2;subweight=weight*t/2
            for s,v in zip(subtime,subweight):
                u=unitary(float(s));fi=(model.em['legs']@(gin*np.exp(-1j*omega*s))/np.sqrt(2)).ravel();fo=(model.em['legs']@(gout.conj()*np.exp(-1j*omega*(T-s)))/np.sqrt(2)).ravel()
                ai=np.einsum('e,eij->ij',fi,Jq);ao=np.einsum('e,eij->ij',fo,Jq)
                ai=np.asarray(block_diag(u.conj().T@(O.conj().T@ai@O)@u,np.zeros((4,4))));ao=np.asarray(block_diag(u.conj().T@(O.conj().T@ao@O)@u,np.zeros((4,4))))
                forward-=w*v*(car_word(occupied,[Bd,Jo,ai,Bin])+car_word(occupied,[Bd,Ji,ao,Bin]))
        amplitude=contact+forward+cross;scale=abs(amplitudes[0 if charge==1 else 1])**2/Z
        rows.append(dict(charge=charge,source_global_Z=Z,contact=contact*scale,charged_forward_JJ=forward*scale,
            neutral_backward_cross_JJ=cross*scale,raw_conditional_transition=amplitude*scale,
            dropping_neutral_endpoint_defect=abs(cross*scale),dropping_contact_defect=abs(contact*scale),
            complete_occupied_light_count=int(occupied[:n].sum()),complete_Ref_count=4,
            fullCAR_intermediate_PH_retained=True,new_probability_or_point_receiver=False))
    return dict(rows=rows,profile=profile,quadrature=quadrature,clock=T,Omega=0,
        photon_external_overlap=complex(np.vdot(gout,gin)),
        actual_original_true_photon_g=gin,actual_OUT_g=gout,OUT_order='a†gout B U_N(T)Xi',
        ordinary_vs_long_driven_experiments_distinct=True,
        scope='conditional N3 full108+Ref4 flatCAR CTC leading coefficient; no source range invariance, vacuum reset or LEGAL probability claim')

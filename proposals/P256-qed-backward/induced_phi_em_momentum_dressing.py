#!/usr/bin/env python3
"""Exact tiled onsite-orbital PH/momentum source for the SAME noncompact Href.

Ownership was changed before implementation: this is NOT a Kato/adiabatic sea.
The unitary translates photons conditionally on commuting local CAR generators.
All64 species, original Peierls paths, harmonics and connected Coulomb remain.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from induced_phi_em_gaussian_variational import (build_native, evaluate_gaussian,
    evaluate_old_packet, optimize_variance, matter_hamiltonian, SOURCE_BYTES as GAUSSIAN_BYTES)
from induced_phi_backreaction import SELECTED

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = dict(GAUSSIAN_BYTES)
SOURCE_BYTES[Path(__file__).name] = Path(__file__).read_bytes()


def build_dressing(native: dict[str, Any], axis: int = 0) -> dict[str, Any]:
    """U_k=exp[-ik sum_x Pi_(x,axis) sum_species c_x^dag Gamma_axis c_x]."""
    if axis not in (0, 1, 2):
        raise ValueError('axis must be a spatial orientation')
    labels, W = np.linalg.eigh(SELECTED[axis+1])
    labels = np.rint(labels)
    profiles = []
    V = native['V']
    for profile in native['Pprofiles']:
        O = np.einsum('ab,xbk->xak', W.conj().T, profile['Pocc'].reshape(V,4,2*V)).reshape(4*V,2*V)
        paths = [p for p in profile['paths'] if p['source'] == 0]
        profiles.append(dict(load=profile['load'], label=profile['label'], O=O,
            P=O@O.conj().T, onsite=W.conj().T@profile['onsite']@W,
            paths=[dict(p, hop=W.conj().T@p['hop']@W) for p in paths]))
    Gamma=SELECTED[axis+1]
    Bmean=np.zeros(V); Bcov=np.zeros((V,V))
    for profile in native['Pprofiles']:
        blocks=profile['blocks']; load=profile['load']
        Bmean+=load*np.einsum('ab,xba->x',Gamma,blocks[np.arange(V),np.arange(V)]).real
        Bcov-=load*np.einsum('ab,xybc,cd,yxda->xy',Gamma,blocks,Gamma,blocks,optimize=True).real
        Bcov[np.arange(V),np.arange(V)]+=load*np.einsum('ab,bc,xca->x',Gamma,Gamma,
            blocks[np.arange(V),np.arange(V)]).real
    embedding=np.zeros((3*V,V))
    embedding[axis::3]=np.eye(V)
    shiftcov=native['PT']@embedding@Bcov@embedding.T@native['PT']
    return dict(native=native, axis=axis, labels=labels, W=W, profiles=profiles,
        Bmean=Bmean,Bcov=Bcov,shiftcov=shiftcov)


def _adjugate(matrix: np.ndarray) -> tuple[complex, np.ndarray]:
    """SVD cofactor product: no inverse or division by a full-sea overlap."""
    U, singular, Vh = np.linalg.svd(matrix)
    phase = complex(np.linalg.det(U)*np.linalg.det(Vh))
    prefix = np.r_[1., np.cumprod(singular)]
    suffix = np.r_[np.cumprod(singular[::-1])[::-1], 1.]
    cofactors = prefix[:-1]*suffix[1:]
    return phase*float(prefix[-1]), phase*(Vh.conj().T*cofactors)@U.conj().T


def _fcs(model: dict[str, Any], ell: np.ndarray, kappa: float) -> list[dict[str, Any]]:
    native = model['native']; axis = model['axis']
    v = (native['PT']@ell)[axis::3]
    R = np.exp(1j*kappa*v[:,None]*model['labels'][None,:]).ravel()
    result = []
    for profile in model['profiles']:
        O = profile['O']; RO = R[:,None]*O
        matrix = O.conj().T@RO
        chi, adj = _adjugate(matrix)
        result.append(dict(chi=chi, adj=adj, R=R, RO=RO, matrix=matrix,
            numerator=RO@adj@O.conj().T))
    return result


def _spectator(model: dict[str, Any], fcs: list[dict[str, Any]], index: int) -> complex:
    result = 1.+0j
    for q, profile in enumerate(model['profiles']):
        result *= fcs[q]['chi']**(profile['load']-(q == index))
    return result


def _delta(model: dict[str, Any], left: int, right: int) -> np.ndarray:
    native=model['native']; axis=model['axis']; labels=model['labels']
    delta=np.zeros(3*native['V'])
    delta[3*(left//4)+axis]+=labels[left%4]
    delta[3*(right//4)+axis]-=labels[right%4]
    return delta


def _gaussian_factor(model: dict[str, Any], ell: np.ndarray, delta: np.ndarray,
                     kappa: float, variance: float) -> complex:
    PT=model['native']['PT']
    return complex(np.exp(-variance*(ell@PT@ell)/2
        -kappa*kappa*(delta@PT@delta)/(8*variance)
        +.5j*kappa*(ell@PT@delta)))


def evaluate_dressing(model: dict[str, Any], kappa: float, variance: float) -> dict[str, Any]:
    """Exact Gaussian integrations and full64 determinant/cofactor CAR insertion."""
    if variance <= 0 or not np.isfinite(variance) or not np.isfinite(kappa):
        raise ValueError('finite kappa and positive finite density variance required')
    native=model['native']; V=native['V']; PT=native['PT']; axis=model['axis']
    matter=0.; onsite=0.; hopping=0.; rows=[]
    current=np.zeros(3*V); contact=np.zeros((3*V,3*V))
    # Local generators preserve every local total occupation: FULL rho/Coulomb unchanged.
    for profile in model['profiles']:
        P=profile['P']; amount=0.
        for r in range(4):
            for s in range(4):
                delta=_delta(model,r,s)
                amount += float((profile['onsite'][r,s]*P[s,r]
                    *_gaussian_factor(model,np.zeros(3*V),delta,kappa,variance)).real)
        onsite += V*profile['load']*amount
    for number, template in enumerate(model['profiles'][0]['paths']):
        ell=np.zeros(3*V)
        for edge in template['edges']:
            ell[edge]+=1
        fcs=_fcs(model,ell,kappa)
        path_total=0j; contributions=[]
        for q, profile in enumerate(model['profiles']):
            path=profile['paths'][number]; y=path['target']
            C=fcs[q]['numerator']; numerator=0j
            for r in range(4):
                for s in range(4):
                    left=r; right=4*y+s
                    numerator += path['hop'][r,s]*C[right,left]*_gaussian_factor(
                        model,ell,_delta(model,left,right),kappa,variance)
            contribution=profile['load']*_spectator(model,fcs,q)*numerator
            path_total += contribution
            contributions.append(dict(label=profile['label'], contribution=contribution,
                determinant=fcs[q]['chi']))
        hopping+=2*V*path_total.real
        rows.append(dict(axis=template['axis'],length=template['length'],representative=path_total,
            profiles=contributions))
        # Translate source0 to each site, preserving wrapped path multiplicity.
        for x, site in enumerate(native['sites']):
            translated=np.zeros(3*V)
            for step in range(template['length']):
                vertex=site.copy(); vertex[template['axis']]=(vertex[template['axis']]+step)%native['N']
                n=int(np.ravel_multi_index(tuple(vertex),(native['N'],)*3))
                translated[3*n+template['axis']]+=1
            current+=-2*path_total.imag*translated
            contact+=-2*path_total.real*np.outer(translated,translated)
    matter=onsite+hopping
    kinetic=native['UE']/native['a']*native['M']/(4*variance)
    energy=kinetic+native['Coulomb']+matter
    return dict(kappa=kappa,variance=variance,axis=axis,normalization=1.,energy=float(energy),
        energy_per_site=float(energy/V),energy_parts=dict(EMkinetic=float(kinetic),Coulomb=native['Coulomb'],
            matter=float(matter),onsite=float(onsite),hopping=float(hopping)),
        current=current,contact=contact,charge_covariance=native['charge_covariance'],
        mean_charge=np.zeros(V),momentum_covariance=native['PT']/(4*variance),
        EMcov=variance*PT+kappa*kappa*model['shiftcov'],local_generator_covariance=model['Bcov'],paths=rows,
        scope='Exact finite-box analytic Gaussian/FCS expectation, binary64 arithmetic; not a stationary/ground/photon-vacuum claim')


def momentum_section(model: dict[str, Any], momentum: np.ndarray, kappa: float,
                     variance: float) -> dict[str, Any]:
    """Actual normalized occupied frames and scalar Fourier-Gaussian amplitude.

    Returns Gamma(R(p)) Omega for each profile; the auxiliary frame represents
    63 distinct tensor factors, NOT one multiply occupied fake Slater.
    """
    native=model['native']; p=np.asarray(momentum,dtype=float)
    if p.shape!=(native['M'],):
        raise ValueError('momentum must have one real entry per FULL physical Q mode')
    local=(native['Q']@p)[model['axis']::3]
    R=np.exp(-1j*kappa*local[:,None]*model['labels'][None,:]).ravel()
    frames=[]
    for profile in model['profiles']:
        rotated=(R[:,None]*profile['O']).reshape(native['V'],4,2*native['V'])
        original=np.einsum('ab,xbk->xak',model['W'],rotated).reshape(4*native['V'],2*native['V'])
        frames.append(dict(label=profile['label'],load=profile['load'],occupied=original,
            orthonormal_error=float(np.linalg.norm(original.conj().T@original-np.eye(2*native['V'])))))
    log_amplitude=native['M']/4*math.log(2*variance/math.pi)-variance*float(p@p)
    return dict(log_scalar_amplitude=log_amplitude,profiles=frames,
        state='normalized momentum Gaussian times all64 Gamma(local unitary) occupied seas')


def fcs_expect(model: dict[str, Any], events: list[tuple[int,int,int]], phase: np.ndarray,
               kappa: float, variance: float, *, copy_relation: str = 'all',
               prepared: list[dict[str, Any]] | None = None) -> complex:
    """One/two literal CAR events times exp(i phase.theta), profile sums explicit.

    Events are (profile,left,right) in model.W's common orbital eigenbasis.
    'same' selects the SAME auxiliary tensor factor, 'distinct' the different
    factors, and 'all' both. Different profile labels are always distinct.
    A density insertion is simply a second diagonal event, with delta=0.
    Optional prepared=_fcs(model,phase,kappa) reuses exact determinant data.
    """
    if len(events) not in (1,2) or copy_relation not in ('all','same','distinct'):
        raise ValueError('one/two events and a valid species-copy relation required')
    ell=np.asarray(phase,dtype=float)
    fcs=_fcs(model,ell,kappa) if prepared is None else prepared
    q,i,j=events[0]; profile=model['profiles'][q]; load=profile['load']
    delta=_delta(model,i,j)
    first1=fcs[q]['numerator'][j,i]
    if len(events)==1:
        return load*_spectator(model,fcs,q)*first1*_gaussian_factor(model,ell,delta,kappa,variance)
    r,k,l=events[1]; other=model['profiles'][r]; delta+=_delta(model,k,l)
    first2=fcs[r]['numerator'][l,k]
    total=0j
    if q==r and copy_relation!='distinct':
        O=profile['O']; RO=fcs[q]['RO']; M=fcs[q]['matrix']
        K1=np.outer(O[i].conj(),RO[j]); K2=np.outer(O[k].conj(),RO[l])
        K12=np.outer(O[i].conj(),RO[l]) if j==k else np.zeros_like(M)
        second=0j
        for t,u in itertools.product((-1,1),repeat=2):
            second+=t*u*np.linalg.det(M+t*K1+u*K2+t*u*K12)/4
        total+=load*_spectator(model,fcs,q)*second
    if copy_relation!='same' and (q!=r or load>1):
        spectator=1.+0j
        for z,p in enumerate(model['profiles']):
            spectator*=fcs[z]['chi']**(p['load']-(z==q)-(z==r))
        copies=load*(load-1) if q==r else load*other['load']
        total+=copies*spectator*first1*first2
    return total*_gaussian_factor(model,ell,delta,kappa,variance)


def directed_bilinear_pair(model: dict[str, Any], kappa: float, variance: float,
                          ell1: np.ndarray, indices1: tuple[int,int], ell2: np.ndarray,
                          indices2: tuple[int,int]) -> complex:
    """Exact all64 equal-time bilinear word; no free-band retarded inference."""
    ell=np.asarray(ell1)+np.asarray(ell2); prepared=_fcs(model,ell,kappa)
    return sum(fcs_expect(model,[(q,*indices1),(r,*indices2)],ell,kappa,variance,
        prepared=prepared) for q,r in itertools.product(range(len(model['profiles'])),repeat=2))


def current_direction(model: dict[str, Any], profile_index: int, theta: np.ndarray,
                      direction: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Literal original h'(theta)[direction], h''(theta)[direction,direction].

    Returned matrices are in ORIGINAL site-major orbitals, not model.W basis.
    State differentiation is deliberately absent: kappa is a source parameter.
    """
    native=model['native']; profile=native['Pprofiles'][profile_index]
    theta=np.asarray(theta,dtype=float); direction=np.asarray(direction,dtype=float)
    if theta.shape!=(3*native['V'],) or direction.shape!=theta.shape:
        raise ValueError('theta and direction must be physical real edge vectors')
    current=np.zeros_like(profile['hcentre']); contact=np.zeros_like(current)
    for path in profile['paths']:
        x,y=path['source'],path['target']
        coefficient=float(np.sum(direction[list(path['edges'])]))
        hop=path['hop']*np.exp(1j*np.sum(theta[list(path['edges'])]))
        sx=slice(4*x,4*x+4); sy=slice(4*y,4*y+4)
        J=1j*coefficient*hop; C=-coefficient*coefficient*hop
        current[sx,sy]+=J; current[sy,sx]+=J.conj().T
        contact[sx,sy]+=C; contact[sy,sx]+=C.conj().T
    return current,contact


def independent_fock_control() -> dict[str, Any]:
    """Independent 4-orbital,2-particle wedge and real Gaussian quadrature."""
    rng=np.random.default_rng(2220810)
    O=np.linalg.qr(rng.normal(size=(4,2))+1j*rng.normal(size=(4,2)))[0]
    basis=list(itertools.combinations(range(4),2))
    amplitudes=np.array([np.linalg.det(O[list(b)]) for b in basis])
    b=np.array([-1.,1.,-1.,1.]); i,j=0,1; kappa=.37; variance=.29; ell=.61
    B=np.array([sum(b[list(pair)]) for pair in basis]); A=np.zeros((6,6),complex)
    for n,pair in enumerate(basis):
        occupied=list(pair)
        if j not in occupied or (i in occupied and i!=j):
            continue
        pos=occupied.index(j); occupied.pop(pos); sign=(-1)**pos
        pos=sum(x<i for x in occupied); occupied.insert(pos,i); sign*=(-1)**pos
        A[basis.index((occupied[0],occupied[1])),n]=sign
    norm=(2*math.pi*variance)**-.25
    def integrand(theta: float) -> complex:
        state=amplitudes*norm*np.exp(-(theta-kappa*B)**2/(4*variance))
        return complex(state.conj()@A@state*np.exp(1j*ell*theta))
    numerical=quad(lambda x: integrand(x).real,-np.inf,np.inf,epsabs=1e-11)[0]+1j*quad(
        lambda x: integrand(x).imag,-np.inf,np.inf,epsabs=1e-11)[0]
    R=np.exp(1j*kappa*ell*b); M=O.conj().T@(R[:,None]*O); chi,adj=_adjugate(M)
    C=(R[:,None]*O)@adj@O.conj().T; delta=b[i]-b[j]
    analytic=C[j,i]*np.exp(-variance*ell**2/2-kappa**2*delta**2/(8*variance)+.5j*kappa*ell*delta)
    wrong=C[j,i]*np.exp(-variance*ell**2/2-kappa**2*delta**2/(8*variance)-.5j*kappa*ell*delta)
    # Independent tensor-Fock control of same/different species and two CAR events.
    A2=np.zeros_like(A)
    for n,pair in enumerate(basis):
        occupied=list(pair)
        if 3 not in occupied or 2 in occupied:
            continue
        pos=occupied.index(3); occupied.pop(pos); sign=(-1)**pos
        pos=sum(x<2 for x in occupied); occupied.insert(pos,2); sign*=(-1)**pos
        A2[basis.index((occupied[0],occupied[1])),n]=sign
    amplitudes2=np.kron(amplitudes,amplitudes); B2=(B[:,None]+B[None,:]).ravel()
    Aall=np.kron(A,np.eye(6))+np.kron(np.eye(6),A)
    A2all=np.kron(A2,np.eye(6))+np.kron(np.eye(6),A2)
    word=Aall@A2all
    def pair_integrand(theta: float) -> complex:
        state=amplitudes2*norm*np.exp(-(theta-kappa*B2)**2/(4*variance))
        return complex(state.conj()@word@state*np.exp(1j*ell*theta))
    pair_numerical=quad(lambda x: pair_integrand(x).real,-np.inf,np.inf,epsabs=1e-11)[0]+1j*quad(
        lambda x: pair_integrand(x).imag,-np.inf,np.inf,epsabs=1e-11)[0]
    oscillator=dict(native=dict(V=1,PT=np.diag([1.,0.,0.])),axis=0,labels=b,
        profiles=[dict(load=2,O=O)])
    phase=np.array([ell,0.,0.]); events=[(0,0,1),(0,2,3)]
    pair_analytic=fcs_expect(oscillator,events,phase,kappa,variance)
    wrong_same_only=fcs_expect(oscillator,events,phase,kappa,variance,copy_relation='same')
    return dict(gaussian_wedge_defect=float(abs(numerical-analytic)),wrong_endpoint_phase_defect=float(abs(numerical-wrong)),
        wedge_norm=float(np.real(amplitudes.conj()@amplitudes)),numerical=numerical,analytic=analytic,
        cofactor_identity=float(np.linalg.norm(M@adj-chi*np.eye(2))),
        two_species_CAR_word_defect=float(abs(pair_analytic-pair_numerical)),
        wrong_dropping_distinct_species_defect=float(abs(wrong_same_only-pair_numerical)))


def descriptor(model: dict[str, Any], kappa: float, variance: float) -> dict[str, Any]:
    native=model['native']; V=native['V']; M=native['M']; load=sum(p['load'] for p in model['profiles'])
    Bcap=4*load
    rowcaps=[]
    for profile in native['Pprofiles']:
        rowcap=float(np.linalg.norm(profile['onsite'],2))+2*sum(float(np.linalg.norm(p['hop'],2))
            for p in profile['paths'] if p['source']==0)
        rowcaps.append(profile['load']*2*V*rowcap)
    carcap=sum(rowcaps); ccap=native['UE']/native['a']*128**2*V/native['laplacian_gap']
    kinetic2=(native['UE']/native['a'])**2*M*(M+2)/(16*variance**2)
    shifted2=variance*M+kappa**2*float(np.trace(model['shiftcov']))
    scale_error=max(p['exact_scale_error_upper'] for p in native['frame_image']['scales'])
    epsilon_h=[]; state_error2=0.; direct_car_error=0.
    for profile in native['Pprofiles']:
        onsite,paths=profile['model'].couplings()
        primitive_cap=abs(float(np.trace(onsite@SELECTED[0]).real/4))+2*sum(
            float(np.sum(abs(np.einsum('mij,ji->m',SELECTED,hop)/4))) for _,hop in paths)
        error=scale_error*primitive_cap
        gap=profile['onebody_gap']-error
        if gap<=0:
            raise ArithmeticError('Native frame-image price loses the seed spectral gap')
        projector_error=2*error/gap
        state_error2+=profile['load']*4*V*projector_error**2
        direct_car_error+=profile['load']*2*V*error
        epsilon_h.append(dict(label=profile['label'],onebody_image_error_upper=error,
            projector_image_error_upper=projector_error))
    frame_energy_error=direct_car_error+2*(carcap+ccap)*min(2.,math.sqrt(state_error2))
    return dict(N=native['N'],V=V,a=native['a'],UE=native['UE'],M=M,axis=model['axis'],kappa=kappa,
        s_density_variance=variance,Q_sha256=hashlib.sha256(native['Q'].tobytes()).hexdigest(),
        chargedloads=[dict(load=p['load'],occupation=2*V,label=p['label']) for p in model['profiles']],
        normalized_state_formula='Psi(p)=(2s²/pi)^(M/4) exp(-s²|p|²) tensor_species Gamma[R_x(p)] Omega0; R_x=exp[-i kappa (Qp)_(x,j) Gamma_j]',
        exact_normalization=1.,lambda_dressing=None,source_parameter='kappa changes state ONLY, original charge/law unchanged',
        Gauss='Physical real Q quotient retains full harmonic3; longitudinal real Gauss lift is an isometry with operator rho, not TF density',
        Coulomb_invariance='Each B_x conserves every species local number. [U,rho_x]=0, hence entire connected rho Lplus rho unchanged including Ref4/background cancellation',
        kinetic_invariance='U is momentum-fiber unitary; [U,sum Pi_a²]=0. No omitted Born-Huang/Berry cost: exact original kinetic invariant; CAR nonadiabatic cost is included in onsite/path endpoint overlaps',
        CAR_stat='Independent64 species; auxiliary load63 counts tensor copies, spectator determinants to power63 and connected Wick covariance linear63',
        tails=dict(momentum_probability_outside_radius_R_upper='M/(4s² R²)',
            coordinate_probability_outside_radius_R_upper=f'{shifted2}/R²',coordinate_second_moment_upper=shifted2),
        graph=dict(CAR_norm_upper=carcap,Coulomb_norm_upper=ccap,EMkinetic_second_moment=kinetic2,
            H_squared_norm_upper=3*(kinetic2+(carcap+ccap)**2),domain='Gaussian Schwartz section times bounded finite-CAR conditional translations; in D(H) and all finite graph powers'),
        frame_image=dict(native['frame_image'],seed_projector_prices=epsilon_h,
            complete_energy_image_error_upper=frame_energy_error,
            scope='Conservative finite native image comparison at same source unitary; not macro centre promotion'),
        scope='Native finite diagnostic ONLY; macro222 mean remains exact implicit, no N2/N6 promotion')


def _json(value: Any) -> Any:
    if isinstance(value,np.ndarray):
        return [_json(x) for x in value.tolist()]
    if isinstance(value,complex):
        return dict(real=value.real,imag=value.imag)
    if isinstance(value,np.generic):
        return _json(value.item())
    if isinstance(value,dict):
        return {k:_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_json(x) for x in value]
    return value


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=2)
    parser.add_argument('--axis',type=int,default=0)
    parser.add_argument('--kappa',type=float,help='Evaluate one specified pure source without repeating optimization')
    parser.add_argument('--mean',type=Path,default=ROOT/'induced_phi_ir_mean_certificate_normal_transfer_final_receipt.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); started=time.monotonic()
    native=build_native(args.mean,args.size); model=build_dressing(native,args.axis)
    optimum=optimize_variance(native); variance=optimum['variance']
    if args.kappa is None:
        scan=[evaluate_dressing(model,kappa,variance) for kappa in (-.03,-.01,-.003,-.001,0.,.001,.003,.01,.03)]
        zero=scan[4]
        best=min(scan,key=lambda r:r['energy']); centre=best['kappa']
        fit: Any=minimize_scalar(lambda k:evaluate_dressing(model,float(k),variance)['energy'],
            bounds=(centre-.01,centre+.01),method='bounded',options={'xatol':1e-10})
        dressed=evaluate_dressing(model,float(fit.x),variance)
    else:
        zero=evaluate_dressing(model,0.,variance)
        dressed=evaluate_dressing(model,args.kappa,variance)
        scan=[zero,dressed]
    seed=evaluate_gaussian(native,variance,correlator=False)
    controls=independent_fock_control()
    controls['zero_dressing_energy_defect']=abs(zero['energy']-seed['energy'])
    controls['zero_current_defect']=float(np.linalg.norm(zero['current']-seed['current']))
    controls['zero_contact_defect']=float(np.linalg.norm(zero['contact']-seed['contact']))
    section=momentum_section(model,np.linspace(-.3,.4,native['M']),dressed['kappa'],variance)
    controls['section_orthonormal_error']=max(p['orthonormal_error'] for p in section['profiles'])
    zeroell=np.zeros(3*native['V'])
    pair=directed_bilinear_pair(model,dressed['kappa'],variance,zeroell,(0,0),zeroell,(0,0))
    controls['local_number_squared_moment']=pair
    expected_number=sum(p['load']*p['P'][0,0].real for p in model['profiles'])
    expected_number2=expected_number**2+sum(p['load']*p['P'][0,0].real*(1-p['P'][0,0].real) for p in model['profiles'])
    controls['local_number_squared_Wick_defect']=float(abs(pair-expected_number2))
    p1=np.zeros(native['M']); p1[0]=1.
    first=momentum_section(model,np.zeros(native['M']),dressed['kappa'],variance)
    second=momentum_section(model,p1,dressed['kappa'],variance)
    fidelity=1.
    for a,b in zip(first['profiles'],second['profiles']):
        fidelity*=float(abs(np.linalg.det(a['occupied'].conj().T@b['occupied']))**(2*a['load']))
    controls['different_momentum_CAR_fidelity']=fidelity
    controls['Coulomb_invariance_defect']=abs(dressed['energy_parts']['Coulomb']-seed['Coulomb'])
    controls['kinetic_invariance_defect']=abs(dressed['energy_parts']['EMkinetic']-seed['energy_parts']['EMkinetic'])
    theta=np.linspace(-.03,.02,3*native['V']); direction=native['Q'][:,0]
    J,C=current_direction(model,0,theta,direction); profile=native['Pprofiles'][0]
    step=1e-3
    plus=matter_hamiltonian(native,profile,theta+step*direction)
    minus=matter_hamiltonian(native,profile,theta-step*direction)
    centre_h=matter_hamiltonian(native,profile,theta)
    controls['literal_current_FD_defect']=float(np.linalg.norm((plus-minus)/(2*step)-J))
    controls['literal_contact_FD_defect']=float(np.linalg.norm((plus-2*centre_h+minus)/(step*step)-C))
    controls['zero_source_relative_energy_defect']=controls['zero_dressing_energy_defect']/max(1.,abs(seed['energy']))
    if (controls['gaussian_wedge_defect']>1e-9 or controls['two_species_CAR_word_defect']>1e-9
        or controls['wrong_endpoint_phase_defect']<1e-4 or controls['zero_source_relative_energy_defect']>1e-9):
        raise ArithmeticError(f'Independent source control failed: {controls}')
    result: dict[str, Any]=dict(branch='NEW exact tiled onsiteGamma PH-momentum coupled source, SAME original noncompactHref',
        descriptor=descriptor(model,dressed['kappa'],variance),seed=seed,old_packet=evaluate_old_packet(native,correlator=False),
        scan=[{key:row[key] for key in ('kappa','variance','energy','energy_per_site','energy_parts')} for row in scan],
        optimized=dressed,improvement=dressed['energy']-seed['energy'],controls=controls,
        uncertainty=dict(method='Exact Gaussian integration + deterministic finite determinant/cofactor linear algebra; no MC/quadrature energy estimate',
            arithmetic='binary64, independent zero-source/wedge checks; no interval-certified minimization or macro proof',
            observed_zero_source_energy_arithmetic_defect=controls['zero_dressing_energy_defect'],
            acceptance_relative_arithmetic_tolerance=1e-9,
            qualification='Tolerance is a numerical consistency criterion, NOT a rigorous interval error enclosure; observed baseline defect is retained, not special-cased away'),
        input=dict(mean_name=args.mean.name,sha256=hashlib.sha256(args.mean.read_bytes()).hexdigest()),
        source_at_import={name:dict(sha256=hashlib.sha256(raw).hexdigest(),bytes_utf8=raw.decode()) for name,raw in SOURCE_BYTES.items()},
        runtime=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),elapsed_seconds=time.monotonic()-started,natural_exit_expected=0))
    args.output.write_text(json.dumps(_json(result),indent=2)+'\n')
    print(json.dumps(_json(dict(size=args.size,variance=variance,kappa=dressed['kappa'],seed_energy=seed['energy'],
        dressed_energy=dressed['energy'],improvement=result['improvement'],controls=controls,elapsed_seconds=result['runtime']['elapsed_seconds'])),indent=2))


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""NEW covariant positive-T preparation; frozen 61d690/Law05 is untouched.

Native: invariant two-spin slice of the actual Phi Clifford frame, a four-link
square, exact integer U1 Gauss and positive Wilson Fock/Haar range. Both signed
quantum normal-field/Ref sources and their finite-energy spectral preparations
are executed; this slice is NOT the full macro SU4/Phi source admission.
Macro: exact finite-box photon energy sums, all-statistical current support,
canonical determinant bounds and conditional Root projective/BKS transfer.
Its microscopic photon/source retained Gram remains unproved. A scalar
Gaussian action ratio is explicitly rejected as an inserted-Haar Gram theorem.
Quantum transfer, parameters and Root field completion are declared inputs.
"""
from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction
import importlib.abc
import importlib.machinery
import importlib.util
import itertools
import json
import math
import marshal
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any,cast

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
SNAPSHOT={p.stem:p.read_bytes() for p in ROOT.glob('*.py')}
COMPILED: dict[str,str]={Path(__file__).stem:hashlib.sha256(SNAPSHOT[Path(__file__).stem]).hexdigest()}
LOADED_MAIN_CODE=sys._getframe().f_code
MAIN_CODE_MATCH=LOADED_MAIN_CODE==compile(SNAPSHOT[Path(__file__).stem],LOADED_MAIN_CODE.co_filename,'exec',dont_inherit=True,optimize=sys.flags.optimize)
if not MAIN_CODE_MATCH:raise ArithmeticError('Loaded __main__ code does not match captured source')
class RecordedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str):
        data=SNAPSHOT[fullname]
        COMPILED[fullname]=hashlib.sha256(data).hexdigest()
        return compile(data,self.path,'exec')
class RecordedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname: str,path: Any=None,target: Any=None):
        if fullname not in SNAPSHOT:return None
        return importlib.util.spec_from_loader(fullname,RecordedLoader(fullname,str(ROOT/(fullname+'.py'))))
finder=RecordedFinder();sys.meta_path.insert(0,finder)
import numpy as np
from scipy.linalg import block_diag,eigh
from scipy.special import roots_legendre
from scipy.optimize import brentq
import mpmath as mp
from covariant_induced_photon import Parameters,phi_frame,WilsonConsumer,finite_photon_packet,connected_transfer_kernel,bounds as photon_bounds
import covariant_root_transfer as root_transfer
for _name,_data in root_transfer.SOURCE_BYTES_AT_IMPORT.items():
    _stem=Path(_name).stem
    _module=sys.modules.get(_stem)
    if _module is not None and isinstance(getattr(_module,'__loader__',None),root_transfer.CapsuleLoader):
        _digest=hashlib.sha256(_data).hexdigest()
        if _stem in COMPILED and COMPILED[_stem]!=_digest:
            raise RuntimeError('Root loader disagrees with already compiled source: '+_name)
        SNAPSHOT[_stem]=_data
        COMPILED[_stem]=_digest
sys.meta_path.remove(finder)
PLAN='issue-222-induced-photon-delta13-continuation-20261008'
PLAN_REVISION='a7f0d27e781ceb4f644fb6a3a24ccd597441268e6a4bd1b32887dee65ec6a710'
EPOCH='kappa400-sigma2000-radius2100-microscopic-boundary-minors'


def check(ok: bool,why: str) -> None:
    if not ok:raise ArithmeticError(why)


def occupations(modes: int,number: int):
    return [tuple(c) for c in itertools.combinations(range(modes),number)]


def canonical_reflected_boundary_kernel(left: np.ndarray,middle: np.ndarray,
                                        right: np.ndarray,number: int,
                                        coefficients: np.ndarray) -> dict[str,Any]:
    """Exact fixed-number reflected CAR insertion, conditional on a field path.

    This is a coefficient/minor identity, NOT Wick factorization of an
    interacting density. Haar and normal-field integrals remain outside it.
    """
    modes=len(left);cycle=middle@right@left
    check(0<=number<=modes,'Canonical occupation must be literal')
    adjugate=np.eye(modes,dtype=complex);coefficient=1.+0j
    for degree in range(1,number+1):
        coefficient=sum((np.linalg.det(cycle[np.ix_(occupied,occupied)])
            for occupied in occupations(modes,degree)),0j)
        adjugate=coefficient*np.eye(modes)-cycle@adjugate
    return dict(canonical_closed_trace=coefficient,
        inserted_cross_kernel=coefficients.conj().T@adjugate@middle@coefficients,
        canonical_number=number,modes=modes,
        identity='[z^N] det(1+z C) udag(1+z C)^-1 middle v; C=middle right left. The adjugate polynomial is evaluated by actual principal minors, so neither grand trace nor a singular inverse replaces fixed N.',
        integration_scope='Multiply ALL spectator fixed-number trace coefficients and detB prefactors on the SAME field history, with every centre/statistical branch and Ref sign, then take JOINT full Haar/normal integrals. This conditional kernel need not be pointwise positive; only its reflected integral has OS positivity.')


def canonical_reflected_boundary_control(gamma4: np.ndarray) -> dict[str,Any]:
    """Compare the exact fugacity minor to a full canonical Fock insertion."""
    gamma=two_spin_frame(gamma4);half,pref,_,_,_=square_wilson(gamma,.09,.37)
    # Scaling is displayed, not an energy-zero convention: all matrix factors
    # on both sides of this algebraic identity are the same scaled matrices.
    one=half@half;scale=float(np.linalg.norm(one,2));one/=scale
    left=one;middle=one;right=one
    packet=np.array([.1,.8,.4,.1]);packet/=np.linalg.norm(packet)
    coefficients=source_coefficients(gamma,.09,packet,1)
    result=canonical_reflected_boundary_kernel(left,middle,right,4,coefficients)
    neutral=occupations(8,4);charged=occupations(8,5)
    def exterior(matrix,states):
        rows=np.asarray(states)
        return np.linalg.det(matrix[rows[:,None,:,None],rows[None,:,None,:]])
    index={occupied:i for i,occupied in enumerate(charged)}
    creators=[]
    for ref in range(2):
        creation=np.zeros((len(charged),len(neutral)),complex)
        for column,occupied in enumerate(neutral):
            for mode in range(8):
                if mode in occupied:continue
                target=tuple(sorted((*occupied,mode)))
                creation[index[target],column]=(-1)**sum(z<mode for z in occupied)*coefficients[mode,ref]
        creators.append(creation)
    L=exterior(left,neutral);M=exterior(middle,charged);R=exterior(right,neutral)
    direct=np.array([[np.trace(L@a.conj().T@M@b@R) for b in creators] for a in creators])
    closed=float(np.trace(exterior(left@middle@right,neutral)).real)
    error=float(np.linalg.norm(direct-result['inserted_cross_kernel']))
    check(error<1e-12 and abs(closed-result['canonical_closed_trace'])<1e-12,
        'Canonical Green/adjugate kernel agrees with actual fixed-N CAR reflected boundary')
    result.update(actual_Fock_cross_kernel=direct,absolute_comparison_error=error,
        determinant_prefactor=pref,displayed_onebody_scaling=scale,
        actual_occupations=[4,5],source_recipe='Actual native canonical stable-band coefficients and both Ref labels; all offdiagonal complex insertions retained.',
        macro_admission=False)
    return result


def two_spin_embedding(gamma: np.ndarray) -> np.ndarray:
    values,vectors=eigh(1j*gamma[0]@gamma[1]@gamma[2])
    u=vectors[:,values>0]
    _,v=eigh(u.conj().T@gamma[0]@u)
    return u@v[:,::-1]


def two_spin_frame(gamma: np.ndarray) -> np.ndarray:
    u=two_spin_embedding(gamma)
    reduced=np.array([u.conj().T@g@u for g in gamma[:3]])
    check(max(np.linalg.norm(a@b+b@a-2*np.eye(2)*(i==j)) for i,a in enumerate(reduced) for j,b in enumerate(reduced))<1e-12,'Actual invariant Clifford spin block')
    return reduced


def square_wilson(gamma: np.ndarray,kappa: float,theta: float):
    """Actual positive spatial B/F blocks, before any gauge integration."""
    B=np.eye(4,dtype=complex)/(2*kappa)
    F=np.zeros((4,4),complex)
    for e,(x,y,mu,sign) in enumerate(((0,1,1,1),(1,2,2,1),(2,3,1,-1),(3,0,2,-1))):
        phase=np.exp(1j*theta) if e==3 else 1.
        B[x,y]-=.5*phase;B[y,x]-=.5*phase.conjugate()
        z=sign*gamma[mu,0,1]/2
        F[x,y]+=z*phase;F[y,x]-=z*phase.conjugate()
    L=np.block([[np.eye(4),-F],[np.zeros((4,4)),np.eye(4)]])
    t=L.conj().T@np.block([[np.linalg.inv(B),np.zeros((4,4))],[np.zeros((4,4)),B]])@L
    perm=[0,4,1,5,2,6,3,7];t=t[np.ix_(perm,perm)]
    values,vectors=eigh(t)
    check(values[0]>0,'Positive canonical Wilson onebody transfer')
    half=(vectors*np.sqrt(values))@vectors.conj().T
    h=-(vectors*np.log(values))@vectors.conj().T
    return half,float(np.linalg.det(B).real),h,vectors,values


def full_spin_auxiliary_boundary(gamma: np.ndarray) -> dict[str,Any]:
    """Full four-spin instantaneous canonical kernel with actual chirality."""
    eig,basis=eigh(gamma[0]);plus=basis[:,eig>0]
    gamma5=gamma[0]@gamma[1]@gamma[2]@gamma[3]
    minus=gamma5@plus
    spin=[plus.conj().T@gamma[mu]@minus for mu in range(4)]
    check(max(np.linalg.norm(s+s.conj().T) for s in spin[1:])<1e-12,
        'Actual gamma5-aligned spatial offblocks are anti-Hermitian')
    J=np.block([[np.zeros((8,8)),np.eye(8)],[-np.eye(8),np.zeros((8,8))]])
    rows=[]
    for kappa in (.04,.02,.01):
        frames=[];halves=[];lambdas=[];prefactors=[];dual_errors=[]
        for theta in (0.,.1,.3):
            F=np.zeros((8,8),complex)
            B=np.eye(8,dtype=complex)/(2*kappa)
            for edge,(x,y,mu,sign) in enumerate(((0,1,1,1),(1,2,2,1),(2,3,1,-1),(3,0,2,-1))):
                phase=np.exp(1j*theta) if edge==3 else 1.
                a=slice(2*x,2*x+2);b=slice(2*y,2*y+2)
                B[a,b]-=.5*phase*np.eye(2);B[b,a]-=.5*phase.conjugate()*np.eye(2)
                F[a,b]+=sign*spin[mu]*phase/2;F[b,a]-=sign*spin[mu]*phase.conjugate()/2
            check(bool(np.linalg.norm(F-F.conj().T)<1e-12),'Actual full-spin F is Hermitian')
            inverse=np.linalg.inv(B)
            L=np.block([[np.eye(8),-F],[np.zeros((8,8)),np.eye(8)]])
            T=L.conj().T@block_diag(inverse,B)@L
            values,vectors=eigh(T);half=(vectors*np.sqrt(values))@vectors.conj().T
            pref=float(np.linalg.det(B).real);occupied=vectors[:,values>1]
            check(occupied.shape[1]==8,'Literal full-spin canonical number2V')
            graph=occupied[:8]@np.linalg.inv(occupied[8:])
            check(np.linalg.norm(graph-graph.conj().T)<1e-12,'Inverse duality makes the actual occupied graph Hermitian')
            norm_values,norm_vectors=eigh(np.eye(8)+graph.conj().T@graph)
            inverse_norm=(norm_vectors/np.sqrt(norm_values))@norm_vectors.conj().T
            occupied=np.vstack((graph,np.eye(8)))@inverse_norm
            frames.append(occupied);halves.append(half);prefactors.append(pref)
            lambdas.append(pref*float(np.prod(values[values>1])))
            dual_errors.append(float(np.linalg.norm(J@T@J.conj().T-np.linalg.inv(T))))
        overlap=np.array([[np.linalg.det(a.conj().T@b) for b in frames] for a in frames])
        check(eigh((overlap+overlap.conj().T)/2,eigvals_only=True)[0]>-1e-10,
            'Full four-spin occupied minors give an actual PSD Gram kernel')
        check(bool(np.max(abs(np.imag(overlap)))<1e-12 and np.min(np.real(overlap))>0),
            'Hermitian small canonical graph gives a genuinely real positive overlap in its actual phase gauge')
        U,singular,Vh=np.linalg.svd(halves[1]@halves[0])
        weight=math.sqrt(prefactors[1]*prefactors[0])*float(np.prod(singular[:8]))
        amplitude=math.sqrt(lambdas[1]*lambdas[0])*overlap[1,0]
        left=abs(np.linalg.det(frames[1].conj().T@U[:,:8]))
        right=abs(np.linalg.det(frames[0].conj().T@Vh.conj().T[:,:8]))
        rows.append(dict(kappa=kappa,canonical_number=8,canonical_dimension=math.comb(16,8),
            chirality_inverse_duality_residual=max(dual_errors),
            exact_static_occupied_Gram=overlap,
            actual_full_canonical_next_singular_ratio=float(singular[8]/singular[7]),
            actual_dominant_weight_relative_excess=weight/abs(amplitude)-1,
            actual_dominant_left_distance=math.sqrt(max(0.,2-2*min(1.,left))),
            actual_dominant_right_distance=math.sqrt(max(0.,2-2*min(1.,right))),
            electric_log_overlap_over_kappa4_dtheta2=-math.log(abs(overlap[0,1]))/(kappa**4*.1**2),
            magnetic_log_ratio_over_kappa4_one_minus_cos=-math.log(lambdas[2]/lambdas[0])/(kappa**4*(1-math.cos(.3)))))
    return dict(rows=rows,mean_gamma4=gamma,all_canonical_occupations_retained=True,
        construction='Exact16-mode Wilson T1; canonical8 SVD spectrum from exterior power of its actual onebody product; no full12870-square allocation or rank1 state substitution.',
        scalar_Gauss_claim=False,macro_admission=False,
        positive_phase_proof='J T1 J^-1=T1^-1 and the upper/lower spectral gap imply occupied graph X=Xdag. Use U(X)=[X;I](I+X²)^-1/2. det[U(Xprime)dag U(X)]=det(1+Xprime X)/sqrt(det(1+Xprime²)det(1+X²)). Products of Hermitian graphs have real characteristic polynomial (trace powers real); ||Xprime X||<1 makes this determinant positive. This is a TRUE phase-fixed positive Gram, not the absolute value of a generic complex kernel.',
        U1_background_covariance='Under site phases g, X->gXgdag and U->diag(g,g)U gdag. Its Slater transforms Gamma(g) with compensating det(gdag_lower)=exp(-2i sum_x chi_x), exactly canonical background2/site. Integer source Wilson path and Refhole background cancel body+1/Ref-1; colour/centre transport must use the actual Root Gauss bundle.',
        scope='Full four-spin static auxiliary boundary object and J T1 J^-1=T1^-1, not an interacting Slater or a completed fullmacro inserted numerator.')


def rho_sites(occupied,body_charge: int):
    r=np.bincount(np.asarray(occupied)//2,minlength=4)-1
    r[0]-=body_charge
    return tuple(int(x) for x in r)


def integer_flux(rho,n: int):
    # Edges x -> x+1: E_x-E_(x-1)=rho_x, E_3=n.
    result=[];s=n
    for x in range(4):s+=rho[x];result.append(s)
    check(result[-1]==n,'Integer source total charge must vanish')
    return tuple(result)


def physical_basis(states,cutoff: int,body_charge: int):
    physical=[]
    for i,c in enumerate(states):
        r=rho_sites(c,body_charge)
        for n in range(-cutoff,cutoff+1):
            flux=integer_flux(r,n)
            if max(abs(x) for x in flux)<=cutoff:physical.append((i,n,flux))
    return physical


def native_range(gamma: np.ndarray,kappa: float,grid: int,cutoff: int,body_charge: int) -> dict[str,Any]:
    number=4+body_charge
    states=occupations(8,number)
    middle=[c for c in states if rho_sites(c,body_charge)==(0,0,0,0)]
    physical=physical_basis(states,cutoff,body_charge)
    samples=[]
    for theta in 2*math.pi*np.arange(grid)/grid:
        half,pref,_,_,_=square_wilson(gamma,kappa,float(theta))
        row=np.asarray(states);col=np.asarray(middle)
        minors=half[row[:,None,:,None],col[None,:,None,:]]
        samples.append(math.sqrt(pref)*np.linalg.det(minors))
    coefficients=np.fft.fft(np.asarray(samples),axis=0)/grid
    V=np.array([coefficients[n%grid,i,:] for i,n,_ in physical])
    u,s,_=np.linalg.svd(V,full_matrices=False)
    check(s[-1]>s[0]*1e-10,'Use nonzero bare-Haar range, not a fictitious global Tmin')
    return dict(body_charge=body_charge,states=states,physical=physical,V=V,u=u,values=s*s,
                middle_dimension=len(middle),basis_index={(i,n):j for j,(i,n,_) in enumerate(physical)})


def auxiliary_static_positive_boundary(gamma4: np.ndarray) -> dict[str,Any]:
    """Actual canonical spectral comparison bundle, not an interacting Slater."""
    gamma=two_spin_frame(gamma4);states=np.asarray(occupations(8,4))
    angles=np.linspace(-.3,.3,7);rows=[]
    for kappa in (.09,.04,.02,.01):
        slaters=[];boundaries=[];lambdas=[];gaps=[];halves=[];prefactors=[]
        for theta in angles:
            half,pref,_,vectors,values=square_wilson(gamma,kappa,float(theta))
            occupied=vectors[:,values>1];check(occupied.shape[1]==4,'Actual top canonical4 sector')
            sea=np.linalg.det(occupied[states,:])
            canonical_half=np.linalg.det(half[states[:,None,:,None],states[None,:,None,:]])
            lam=pref*float(np.prod(values[values>1]))
            boundary=math.sqrt(pref)*(canonical_half@sea)
            check(np.linalg.norm(boundary-math.sqrt(lam)*sea)<1e-9*math.sqrt(lam),
                'Actual detB canonical halfstep acts on its exact instantaneous top Slater')
            slaters.append(sea);boundaries.append(boundary);lambdas.append(lam)
            halves.append(canonical_half);prefactors.append(pref)
            gaps.append(math.log(float(values[values>1].min()/values[values<1].max())))
        comparison_scale=lambdas[len(angles)//2]
        measured=np.asarray(boundaries).conj()@np.asarray(boundaries).T/comparison_scale
        overlap=np.asarray(slaters).conj()@np.asarray(slaters).T
        exact=np.sqrt(np.outer(lambdas,lambdas))/comparison_scale*overlap
        error=float(np.linalg.norm(measured-exact))
        check(error<1e-8,'Reflected canonical boundary kernel is EXACT occupied-minor Gram')
        check(eigh((exact+exact.conj().T)/2,eigvals_only=True)[0]>-1e-10,'Actual static boundary kernel is PSD')
        # Unlike the rank-one comparison, this matrix retains ALL70 actual
        # canonical occupation states of BOTH halfsteps.
        full=math.sqrt(prefactors[4]*prefactors[3])*(halves[4]@halves[3])
        amplitude=math.sqrt(lambdas[4]*lambdas[3])*overlap[4,3]
        rank_one=amplitude*np.outer(slaters[4],slaters[3].conj())
        U,singular,Vh=np.linalg.svd(full,full_matrices=False)
        left_alignment=abs(np.vdot(slaters[4],U[:,0]))
        right_alignment=abs(np.vdot(slaters[3],Vh[0].conj()))
        top_amplitude=singular[0]*np.vdot(slaters[4],U[:,0])*np.vdot(Vh[0].conj(),slaters[3])
        relative_phase=(top_amplitude/abs(top_amplitude))*(amplitude.conjugate()/abs(amplitude))
        op_error=float(np.linalg.norm(full-rank_one,2)/abs(amplitude))
        tensor_rows=[]
        for copies in (1,16,1024):
            weight_ratio=(singular[0]/abs(amplitude))**copies
            left_distance=math.sqrt(max(0.,2-2*min(1.,left_alignment)**copies))
            right_distance=math.sqrt(max(0.,2-2*min(1.,right_alignment)**copies))
            tensor_rows.append(dict(copies=copies,dominant_singular_weight_ratio=float(weight_ratio),
                exact_dominant_left_tensor_distance=left_distance,
                exact_dominant_right_tensor_distance=right_distance,
                full_tensor_operator_relative_error_upper=float(
                    abs(weight_ratio*relative_phase**copies-1)+weight_ratio*(left_distance+right_distance+singular[1]/singular[0]))))
        rows.append(dict(kappa=kappa,angles=angles,actual_canonical_sea_eigenvalues=lambdas,
            exact_canonical_pair_gap=gaps,occupied_minor_Gram=overlap,
            actual_compressed_positive_boundary_kernel=measured,
            occupied_minor_identity_absolute_error=error,common_display_scale=comparison_scale,
            phase_covariant_kernel_not_absolute_overlap=True,
            full_canonical_boundary_operator_relative_error=op_error,
            actual_next_canonical_singular_ratio=float(singular[1]/singular[0]),
            dominant_weight_relative_excess=float(singular[0]/abs(amplitude)-1),
            dominant_left_endpoint_distance=math.sqrt(max(0.,2-2*min(1.,left_alignment))),
            dominant_right_endpoint_distance=math.sqrt(max(0.,2-2*min(1.,right_alignment))),
            retained_all_occupation_tensor_bounds=tensor_rows,
            small_angle_fidelity_log_over_kappa4=-math.log(abs(overlap[3,4])**2)/kappa**4))
    return dict(rows=rows,canonical_number=4,canonical_Fock_dimension=70,
        exact_kernel='sqrt(lambda_aux(Aprime)lambda_aux(A)) det[Uocc(Aprime)dag Uocc(A)]; lambda_aux=detBplus product(top T1 eigenvalues). NF flavours tensor this TRUE Gram kernel before shared Haar/Gauss integration.',
        all_occupations_physically_retained=True,
        comparison_projector_only='Pi_aux(A)=|instantaneous exact canonical top Slater><...| is a proof decomposition, NOT a state/filter or a Gaussian-after-interacting-U claim.',
        scope='Actual native Wilson spectral object and positive Fock/Haar boundary factor identity. Physical macro all-link/fullGauss covariance and complement virtual excursions still need bounds.',
        collective_price='Frame jets start kappa_aux²: sqrt(Naux)*kappa_aux²=sqrt(beta)/4. Multiplying by natural deltaA~beta^-1/2 is ORDER ONE, not negligible. NF thermal price alone does NOT control these virtual excursions.',
        macro_admission=False)


def auxiliary_canonical_kernel_budget(p: Parameters) -> dict[str,Any]:
    """Uniform full-canonical mixed-halfstep estimate, not an occupation gate."""
    with mp.workdps(100):
        k=mp.power(10,-p.kappa_power);volume=mp.mpf(p.spatial_size)**3
        copies_exact=10**(p.beta_power+4*p.kappa_power)//16
        copies=mp.mpf(copies_exact)
        constant=mp.mpf(2)**48
        # A deliberately loose constant absorbs the two graph/SVD endpoints,
        # canonical exterior determinant and phase alignment inequalities.
        endpoint=constant*mp.sqrt(copies*volume)*k**3
        weight=constant*copies*volume*k**6
        excited=64*k*k
        check(bool(k<mp.mpf(1)/64 and weight<mp.mpf(1)/64),'Actual branch lies in the uniform graph/SVD gap regime')
        total=2*endpoint+2*mp.expm1(weight)+2*excited
        # Canonical particle-hole counts are binomial(2V,j)^2, not 2^(4V).
        thermal_parameter_log=mp.log(copies*(2*volume)**2)+p.time_slices*mp.log(excited)
        check(bool(thermal_parameter_log<-mp.log(4)),'Canonical excitation exponential remainder has a controlled small parameter')
        thermal_log=mp.log(2)+thermal_parameter_log
        excited_schatten_log=thermal_log/p.time_slices
        scaled_riesz_remainder=mp.mpf(2)**20*volume*(64*k)**6/(1-64*k)
        all_slab_remainder=copies*(p.time_slices+3)*scaled_riesz_remainder
        result=dict(auxiliary_count_exact=str(copies_exact),spatial_volume_exact=str(int(volume)),
            actual_auxiliary_kappa=f'1e-{p.kappa_power}',canonical_number_per_flavour=str(int(2*volume)),
            endpoint_collective_log10_upper=float(mp.log10(endpoint)),
            singular_weight_collective_log10_upper=float(mp.log10(weight)),
            full_canonical_kernel_relative_operator_error_log10_upper=float(mp.log10(total)),
            allNF_canonical_excitation_trace_log_upper=float(thermal_log),
            excitation_Schatten_Nt_relative_log10_upper=float(excited_schatten_log/mp.log(10)),
            allNF_slab_static_kappa6_log10_upper=float(mp.log10(all_slab_remainder)),
            constants=dict(onebody_graph_norm_over_kappa2=32,
                graph_to_mixed_singular_endpoint_constant=2**20,
                canonical_tensor_kernel_constant=2**48,mixed_sameN_next_singular_ratio_over_kappa2=64,
                analytic_scaled_Riesz_radius='1/64',analytic_log_kernel_bound_over_V=2**20))
    result.update(
        exact_full_operator='At every pair of literal compact link fields, use detBpref Gamma_N(sqrtT1(Aprime)sqrtT1(A)), N=2V for EACH auxiliary. This keeps every canonical occupation. The instantaneous Omega(A) is only its exact static comparison vector.',
        kernel_bound='After true canonical phase alignment, ||Kfull(Aprime,A)-K_Omega(Aprime,A)||op <= [2C sqrt(Naux V)kappa³ +2expm1(C Naux V kappa6)+128kappa²] K_Omega(Aprime,A), C=2^48. K_Omega is the ACTUAL positive occupied-minor Gram, not a guessed G.',
        uniform_gap_proof='Bmin=(1-6k)/(2k), ||F||<=3. Minmax gives stable T1<=Bmin^-1 and unstable T1>=Bmin. Offblock<=3/Bmin, so the separated invariant top graph has ||X||<=2(3/Bmin)/(Bmin-Bmin^-1)<=32k² at k<1/64. The canonical pair gap is>=2log Bmin.',
        mixed_singular_proof='In the two instantaneous eigenframes hprime h has blocks Duprime Ruu Du, Duprime Rus Ds, Dsprime Rsu Du, Dsprime Rss Ds. ||Rsu||<=2||X||/(1-||X||²), Ds/Du<=4k. Thus mixed top singular graph endpoints are O(k³), its log singular determinant correction O(V k6), and the next canonical singular ratio<=64k². The 2^20 endpoint bound follows by the separated-block Sylvester/tan-angle estimate; 2^48 absorbs its square, both endpoints, the 2V trace and phase alignment.',
        collective_proof='Canonical dominant vectors are actual Slaters of the mixed onebody SVD; tensor alignment is a product of their cosines, so vector distance<=sqrt(Naux) times a single-flavour distance, while log singular weight adds Naux. The largest discarded tensor singular ratio is the single-flavour next ratio, not Naux times it. These are operator decomposition identities, never Gaussian-after-interacting-U.',
        exact_phase_proof='In gamma5-aligned chirality J T1 J^-1=T1^-1 makes X=Xdag. U(X)=[X;I](1+X²)^-1/2 has strictly positive determinant overlap because ||Xprime X||<1 and the characteristic polynomial is real. Thus the comparison kernel is both PSD and pointwise nonnegative in its TRUE background-covariant phase gauge.',
        static_remainder_proof='At complex |k|=1/64, ||2kT1-diag(0,I)||<=6|k|+64|k|²/(1-6|k|)<1/8. The radius1/2 Riesz contour separates the two bands, and graph/determinant logarithms are analytic with norm<=2^20 V. Bipartite even L removes odd closed words; the explicit fourth-order spatial/temporal Wilson traces give beta=16Naux k4. Cauchy bounds all remaining k6 terms by 2^20 V(64k)^6/(1-64k). The exact common vacuum constant is retained, not independently shifted in charged sectors.',
        Schatten_proof='For EVEN p=Nt the dominant-minus-comparison kernel has pointwise rank<=2, so |Tr E1...Ep|<=2 product||Ei||; integrate against the TRUE positive K_Omega cyclic measure. It is not an assertion that the integral operator has rank2. Excitation cyclic Holder uses sum_j binomial(2V,j)^2 r^(jp); all flavours give <=exp[Naux(2V)^2 r^p]-1. No Fock-dimension bound substitutes for the thermal prices. Odd p+1 requires interpolation and the ACTUAL reference trace ratio, not a silently identical Nt denominator.',
        Gauss_scope='The U1 graph frame transforms with Gamma(g) times the exact background2/site character. Full SU4/centre and statistical sectors must be transported through the existing Root physical bundle; this sublemma never projects them away.',
        scope='Uniform frozen-Clifford auxiliary positive bare-Haar boundary kernel sublemma. Its scalar reference trace and actual active/source/photon toron-conditioned halfsteps remain separate obligations.',
        microscopic_fullpacket_admission_proved=False)
    return result

def active_harmonic_canonical_budget(p: Parameters) -> dict[str,Any]:
    """Full active canonical eigenbundle estimate; never freeze global torons."""
    with mp.workdps(100):
        volume=p.spatial_size**3;mass=mp.power(10,p.beta_power)*p.spatial_size
        connection=mp.mpf(2)**20*volume/p.spatial_size
        heat_price=8*connection/mp.sqrt(mass)
        thermal_log=mp.log(2)+2*mp.log(2*volume)-mp.mpf('.02')*p.time_slices
        check(bool(mass/p.time_slices>=100 and thermal_log<-mp.log(2)),
            'Physical branch satisfies the Poisson and canonical excitation remainder regimes')
        toron_logZ=mp.mpf('1.5')*mp.log(mp.pi*mass/p.time_slices)
        result=dict(canonical_number_neutral=str(2*volume),
            canonical_numbers_charged=[str(2*volume-1),str(2*volume+1)],
            neutral_pair_gap_lower='.02',charged_onebody_gap_lower='.01',
            global_canonical_connection_norm_upper=mp.nstr(connection,20),
            Gaussian_toron_connection_operator_price_log10=float(mp.log10(heat_price)),
            active_canonical_excitation_trace_log10_upper=float(thermal_log/mp.log(10)),
            doubled_heat_three_toron_logZ_upper=float(toron_logZ+mp.mpf('1e-20')))
    result.update(
        exact_reference='Keep the actual full-spin mean-frame T1(k+theta/L) and detB vacuum prefactor for EVERY momentum, all8 statistical torons and literal canonical number. Use only a comparison eigenbundle unitary; do not identify the interacting state with its instantaneous Slater.',
        uniform_gap_proof='B=exp(.01)+sum(1-cos p)>=exp(.01), ||F||<=3. The lower 2-spin block of T1 is B+Fdag F/B>=B. Minmax and J T1 J^-1=T1^-1 imply unstable>=exp(.01), stable<=exp(-.01) at ALL torons, not just the narrow source window.',
        connection_proof='Each momentum has two degenerate stable and two degenerate unstable eigenvalues. Kato A_j=[partial_theta_j P,P] transports the two bands without resolving a degenerate spin basis. ||partial_theta_j T1||<=240/L and their separation>=.019 imply ||A_j||<=480/(.019 L). dGamma(A_j) on N=2V or2V+-1 has norm<=N||A_j||<2^20 V/L. Keep the actual within-band eigenvalues at theta; they are NOT frozen to theta0.',
        positive_kernel_proof='After this physical eigenbundle pullback the active factor is Dhalf(theta_prime) Gamma[U(theta_prime)dag U(theta)] Dhalf(theta). Removing only the connection gives a genuine positive Gaussian-heat sandwich, with the exact theta-dependent Dhalf. It is not a scalar RN-to-Loewner implication.',
        heat_Schatten_proof='For three Gaussian torons, |Delta theta|1 h_t<=8/sqrt(beta L) h_2t, summing all periodic images. Thus the canonical connection kernel norm is bounded by the stated price. For EVEN p, cyclic Fock Holder bounds its loop trace by Tr(Dgap^p) times the scalar doubled-heat loop. This retains ALL occupations: Tr(Dgap^p)<=exp[(2V)^2 exp(-.02p)], and charged traces add the actual2V multiplicity and exp(-.01p). No exponential Fock-dimension or fictitious integral-operator-rank bound is used.',
        winding_and_Gauss='Large-gauge seams must use the actual background2/site character and integer Wilson/Ref intertwiners. The doubled periodic heat sum is valid for every fractional Floquet coset; no coset is fixed to zero. The existing spatial winding theorem bounds the sea vacuum prefactor, while its instantaneous eigenbasis is retained globally.',
        scope='Proved canonical mean-frame/eigenbundle and Gaussian-toron comparison sublemma. Joining it to the actual full-Gauss Wilson photon kernel, Nt+1 filtered trace and the real charged halfsteps is still required.',
        microscopic_fullpacket_admission_proved=False)
    return result


def canonical_gaussian_Gauss_budget(p: Parameters) -> dict[str,Any]:
    """Integer-Gauss Gaussian reference; no free longitudinal entropy."""
    with mp.workdps(100):
        L=p.spatial_size;volume=L**3;beta=mp.power(10,p.beta_power)
        qmin=2*mp.sin(mp.pi/L);mass=beta*L
        coulomb=8*volume/(beta*qmin*qmin)
        flux_shift=32*volume/mp.sqrt(mass)
        result=dict(canonical_background_per_site=2,
            Ref_hole_or_particle_integer_charge=1,
            Coulomb_transfer_error_log10_upper=float(mp.log10(coulomb)),
            all_active_Floquet_heat_shift_operator_error_log10_upper=float(mp.log10(flux_shift)),
            actual_three_dimensional_harmonic_mass=mp.nstr(mass,20))
    result.update(
        exact_projection='Normalized local Haar gives delta_divE,rho with LITERAL integer E and rho, including the body+1/Refhole-1 or adjoint source. It fixes the longitudinal character; it does not leave V-1 independent free longitudinal thermal modes.',
        auxiliary_intertwiner='The normalized actual Omega_aux(A) transforms as Gamma(g) times exp(-2i sum chi). Its tensor isometry pulls the PHYSICAL gauge-average P_Gauss back to the active+Ref background Gauss projector. The distinct rank-null ELECTRIC P_Haar stays in Jdag h_aux P_Haar h_aux J, whose full-field kernel is sqrt(lambda_prime lambda)<Omega(Aprime)|Omega(A)>. Gauge-average endpoints still require joint integration. Collective auxiliary density fluctuations remain in that actual occupied-minor electric kernel; they are NOT replaced by a bare product sea of fixed onsite occupations.',
        canonical_reference='In the ground-comparison frame only, all occupied auxiliary/active lower labels are filled and carry their exact background character. Canonical excitations and both charged sectors remain in the full trace with their actual occupation weights. They are not dropped from the microscopic state.',
        Coulomb_proof='Gaussian Hodge decomposition gives Hlong=rhoT Laplacianplus rho/(2beta). For one active4-spin species and literal Ref4->3 or Ref0->1, sum rho=0, ||rho||²<=4V+5<=8V, so Hlong<=4V/(beta qmin²). The stated8V bound pays both reflected sides. Complementary auxiliary occupations keep the positive Gauss contraction; no small-Coulomb claim is made for their huge collective charges.',
        Floquet_proof='Keep every integer cycle. Active+Ref charges give each dipole shift |kappa_j|<=2V+1<4V. Poisson heat writes h_kappa as phase-weighted periodic Gaussian images. |exp(-i kappa dot Delta)-1|<=4V |Delta|1 and |Delta|1 h_t<=8/sqrt(beta L) h_2t, so the stated32V price bounds the ACTUAL shift versus zero reference. No Floquet coset is declared identical or fixed to zero.',
        thermal_trace_proof='The shifted rotor trace Z(kappa)<=Z(0) follows from Poisson dual cos<=1. For noncommuting canonical matter/charge factors use cyclic Fock Schatten Holder and the same positive scalar heat loop, not factorization of an interacting density. Every longitudinal character contributes a contraction, not a free longitudinal partition function. Canonical excitation counts retain binomial(2V,j)^2, with charged2V multiplicity.',
        scope='Physical integer-Gauss Gaussian comparison sublemma. The actual Wilson-to-Gaussian kernel, full SU4/Root/statistical join and filtered Nt+1/Nt+2 source moments still require their own bounds.',
        microscopic_fullpacket_admission_proved=False)
    return result


def compact_photon_kernel_budget(p: Parameters) -> dict[str,Any]:
    """Positive Wilson/Gaussian kernel estimate, not closed-action RN order."""
    with mp.workdps(100):
        L=p.spatial_size;volume=mp.mpf(L)**3;beta=mp.power(10,p.beta_power)
        radius=mp.mpf('1e-50');qmin=2*mp.sin(mp.pi/L)
        check(bool(beta>=L**4 and qmin*qmin*radius*radius<1/(256*L)),
            'Chosen compact chart obeys the global zero-flux Hodge regime')
        inside=128*volume*beta*radius**4
        entropy=256*volume*(1+mp.log(beta))
        outside_log=-beta*qmin*qmin*radius*radius/128+entropy
        increment_log=-beta*radius*radius/128+entropy
        check(bool(outside_log<0 and increment_log<0),'Physical compact barriers dominate their full photon entropy')
        operator_error=mp.expm1(inside)+8*mp.exp(outside_log/2)+8*mp.exp(increment_log)
        result=dict(coexact_chart_radius='1e-50',local_kernel_log_ratio_upper=mp.nstr(inside,20),
            normalized_positive_kernel_operator_error_log10_upper=float(mp.log10(operator_error)),
            compact_outside_block_log_upper=float(outside_log),
            temporal_long_increment_log_upper=float(increment_log),
            Nt_local_kernel_log_ratio_upper=mp.nstr(p.time_slices*inside,20),
            reference_normalization='One EXACT common vacuum constant and Haar/Hodge Jacobian; neither kernel is independently rescaled.')
    result.update(
        actual_positive_Wilson_reference='KW=c M_W C_beta M_W, with M_W=exp[-beta sum_spatial_faces(1-cos F)/2] and C_beta(Aprime,A)=exp[-beta sum_spatial_links(1-cos(Aprime-A))]. Positive integer Fourier coefficients I_n(beta)>0 prove C_beta PSD. KG is the genuine Gaussian heat/potential sandwich on coexact real coordinates and periodic harmonic/gradient coordinates.',
        local_kernel_proof='In the zero-flux Coulomb chart |Acoexact|infty<=delta, keep ALL flat holonomies and gauge gradients. Split only the temporal kernel into increments |Delta A|infty<=4delta and its long-jump remainder; this is a comparison decomposition, not a state/source gate. |F|<=4delta and 0<=x²/2-(1-cos x)<=x4/24 give |log(KW/KG)|<=128V beta delta4, with the SAME prefactor. Periodic Gaussian-image tails are paid in the long-jump price.',
        compact_Hodge_proof='If magnetic action S<1/(256L), principal face curvatures have no monopole (a2pi cube requires order1 action) and no nonzero flux period (a2pi period costs at least8/L over all parallel planes). Then dF=0 and all periods vanish. Real Fourier Hodge gives ||Acoexact||2<=||F||2/qmin, with ||F||²<=pi² S/2. Thus leaving delta implies S>=qmin² delta²/16; larger-action/topological sectors obey the same weaker threshold. Every flat holonomy remains.',
        actual_operator_proof='For the INSIDE scalar kernel, |EWG(Aprime,A)|<=expm1(price) KG(Aprime,A). Weighted Schur with the actual positive Gaussian ground function gives the operator norm; EVEN Schatten loop integration gives Tr|EWG|^p<=expm1(price)^p Tr KG^p. This is kernel-level domination, NOT an assertion KW-KG is Loewner-positive and NOT scalar action RN lifted through matter insertions.',
        outside_block_proof='KW and KG have positive sandwich form. QKQ is bounded by its magnetic barrier and free electric heat; normalized heat/vacuum/Hodge entropy is <=256V(1+log beta), since beta>=L4. Integer Gaussian heat trace is <=1+sqrt(2pi beta/p). Normalized Wilson trace for p>=1 is <=e^beta/I0(beta)<=6sqrt(beta), by the actual central |theta|<=beta^-1/2 integral; no false quadratic lower bound at arbitrarily high integer n is assumed. Photon ground energy is<3V and finite Hodge determinants cost O(V log L). PSD block Cauchy bounds PKQ by sqrt(PKP QKQ), in operator and Schatten norms. The stated8sqrt(outside) pays both kernels and both cross blocks. A long temporal increment costs beta delta²/128 with the same entropy price.',
        projection_scope='Physical gauge averaging and actual Root/centre/statistical intertwiners must be applied to this positive scalar comparison; rank-null bare electric P_Haar was already retained in its auxiliary compression. Noncommuting active canonical/source factors are not scalar densities and need their separate reflected bounds.',
        scope='Finite-volume positive scalar compact-Wilson/Gaussian kernel sublemma with explicit outside and long-increment payment, not microscopic photon/source admission or all-time coordinate confinement.',
        microscopic_fullpacket_admission_proved=False)
    return result


def positive_scalar_kernel_control() -> dict[str,Any]:
    """Actual PSD kernels reject pointwise-comparison-to-Loewner promotion."""
    beta=80.;grid=192;radius=.3
    angle=2*math.pi*np.arange(grid)/grid-math.pi
    difference=angle[:,None]-angle[None,:]
    electric_w=np.exp(-beta*(1-np.cos(difference)))
    electric_g=sum((np.exp(-beta*(difference+2*math.pi*j)**2/2) for j in (-1,0,1)),np.zeros((grid,grid)))
    rows=[]
    for magnetic in (False,True):
        wilson_potential=np.exp(-beta*(1-np.cos(angle))/2) if magnetic else np.ones(grid)
        gaussian_potential=np.exp(-beta*angle*angle/4) if magnetic else np.ones(grid)
        wilson=wilson_potential[:,None]*electric_w*wilson_potential[None,:]/grid
        gaussian=gaussian_potential[:,None]*electric_g*gaussian_potential[None,:]/grid
        eigen_w=eigh(wilson,eigvals_only=True);eigen_g=eigh(gaussian,eigvals_only=True)
        common_scale=float(eigen_g[-1]);inside=abs(angle)<=radius
        delta=wilson-gaussian;inside_delta=delta[np.ix_(inside,inside)]
        local_price=beta*17*radius**4/24
        error=float(np.linalg.norm(inside_delta,2))/common_scale
        signed=eigh(delta,eigvals_only=True)/common_scale
        check(error<=math.expm1(local_price)*(1+1e-12),
            'Actual positive-kernel Schur comparison controls the projected operator difference')
        rows.append(dict(magnetic_potential=magnetic,
            actual_Wilson_min_eigenvalue=float(eigen_w[0]),
            actual_Gaussian_min_eigenvalue=float(eigen_g[0]),
            common_Gaussian_scale=common_scale,
            actual_inside_operator_error_relative=error,
            actual_kernel_Schur_operator_bound=math.expm1(local_price),
            actual_signed_difference_min=float(signed[0]),actual_signed_difference_max=float(signed[-1])))
    check(rows[0]['actual_signed_difference_min']<-1e-6 and rows[0]['actual_signed_difference_max']>1e-6,
        'Pointwise Wilson/Gaussian rotor comparison is NOT operator Loewner order')
    return dict(beta=beta,grid=grid,coordinate_chart_radius=radius,rows=rows,
        exact_positive_kernel_proof='Wilson heat has strictly positive Bessel Fourier coefficients; periodic Gaussian heat has positive Gaussian Fourier coefficients; multiplying both endpoints by the actual positive potentials preserves PSD.',
        display_scope='Observed Nyström double matrices, not interval enclosures. Gaussian images beyond+-1 are below exp(-2pi² beta) before endpoint potentials. Tiny minimum-eigenvalue rounding is displayed, never promoted to a global Tmin.',
        macro_admission=False)


def canonical_gaussian_Gauss_control() -> dict[str,Any]:
    """Exact integer source/cycle/Hodge identities for every native occupation."""
    beta,tau=20.,8.;vertices=4
    laplacian=2*np.eye(vertices)-np.roll(np.eye(vertices),1,axis=0)-np.roll(np.eye(vertices),-1,axis=0)
    inverse=np.linalg.pinv(laplacian,hermitian=True)
    cycles=np.arange(-48,49,dtype=float);mass=beta/vertices
    zero_trace=float(np.exp(-tau*cycles*cycles/(2*mass)).sum())
    rows=[];maximum_hodge_error=0.
    for body in (0,1,-1):
        physical_trace=0.;worst_rotor_ratio=0.;count=0
        for occupied in occupations(8,4+body):
            rho=np.asarray(rho_sites(occupied,body),dtype=float)
            electric=np.asarray(integer_flux(rho,0),dtype=float)
            check(bool(np.array_equal(electric-np.roll(electric,1),rho)),
                'Every literal integer-cycle/source component satisfies canonical Gauss')
            shift=float(electric.mean());long=float(rho@inverse@rho/(2*beta))
            harmonic=vertices*(cycles+shift)**2/(2*beta)
            direct=np.sum((electric[None,:]+cycles[:,None])**2,axis=1)/(2*beta)
            error=float(np.max(abs(direct-long-harmonic)));maximum_hodge_error=max(maximum_hodge_error,error)
            check(error<2e-12,'Integer Gaussian heat has the exact Coulomb plus shifted-cycle decomposition')
            rotated_trace=float(np.exp(-tau*harmonic).sum())
            check(rotated_trace<=zero_trace*(1+1e-12),
                'Actual retained fractional Floquet heat trace obeys its Poisson upper bound')
            physical_trace+=float(np.exp(-tau*direct).sum())
            worst_rotor_ratio=max(worst_rotor_ratio,rotated_trace/zero_trace);count+=1
        check(physical_trace<=count*zero_trace*(1+1e-12),
            'Full canonical Gauss heat has no independent longitudinal electric entropy')
        rows.append(dict(body_charge=body,canonical_number=4+body,
            all_occupation_components=count,actual_physical_heat_trace=physical_trace,
            canonical_trace_upper=count*zero_trace,actual_maximum_Floquet_trace_ratio=worst_rotor_ratio))
    independent_edge_trace=float(np.exp(-tau*cycles*cycles/(2*beta)).sum())**vertices
    return dict(rows=rows,exact_Hodge_identity_error=maximum_hodge_error,
        integer_cycle_range=[-48,48],native_ring_harmonic_mass=mass,
        native_mass_convention='ONE-dimensional4-edge ring has beta/L, not the three-dimensional beta L.',
        wrong_unprojected_independent_edge_trace=independent_edge_trace,
        actual_zero_character_cycle_trace=zero_trace,
        wrong_free_longitudinal_entropy_ratio=independent_edge_trace/zero_trace,
        scope='All native canonical occupations and integer body/Ref source characters for the Gaussian electric reference, not a microscopic photon/source Gram.',
        macro_admission=False)


def active_harmonic_boundary_control(gamma: np.ndarray) -> dict[str,Any]:
    """Actual four-mode, fixed-N control rejecting a globally frozen band."""
    eig,basis=eigh(gamma[0]);plus=basis[:,eig>0]
    minus=(gamma[0]@gamma[1]@gamma[2]@gamma[3])@plus
    spin=[plus.conj().T@gamma[mu]@minus for mu in range(4)]
    states=np.asarray(occupations(4,2));rows=[]
    for momentum in ((0.,0.,0.),(.3,.2,0.),(math.pi,math.pi,math.pi)):
        halves=[];frames=[]
        for theta in (0.,math.pi/2,math.pi):
            q=np.asarray(momentum)+np.array([theta/4,0.,0.])
            b=math.exp(.01)+float(np.sum(1-np.cos(q)))
            F=sum((1j*spin[j+1]*math.sin(q[j]) for j in range(3)),np.zeros((2,2),complex))
            triangular=np.block([[np.eye(2),-F],[np.zeros((2,2)),np.eye(2)]])
            diagonal=block_diag(np.eye(2)/b,b*np.eye(2))
            T=triangular.conj().T@diagonal@triangular
            values,vectors=eigh(T);occupied=vectors[:,values>1]
            half=(vectors*np.sqrt(values))@vectors.conj().T
            canonical=np.linalg.det(half[states[:,None,:,None],states[None,:,None,:]])
            canonical/=math.sqrt(float(np.prod(values[values>1])))
            dtriangular=np.zeros((4,4),complex)
            dtriangular[:2,2:]=-1j*spin[1]*math.cos(q[0])/4
            db=math.sin(q[0])/4
            ddiagonal=block_diag(-db*np.eye(2)/b**2,db*np.eye(2))
            jet=dtriangular.conj().T@diagonal@triangular+triangular.conj().T@diagonal@dtriangular+triangular.conj().T@ddiagonal@triangular
            projector=occupied@occupied.conj().T;derivative=np.zeros((4,4),complex)
            transformed=vectors.conj().T@jet@vectors
            for lower in range(2):
                for upper in range(2,4):
                    term=transformed[lower,upper]/(values[upper]-values[lower])*np.outer(vectors[:,lower],vectors[:,upper].conj())
                    derivative+=term+term.conj().T
            connection=derivative@projector-projector@derivative
            check(float(values[2])>=math.exp(.01)-1e-12 and float(values[1])<=math.exp(-.01)+1e-12,
                'Actual active four-spin transfer has a uniform global harmonic band gap')
            check(float(np.linalg.norm(connection,2))<=480/(.019*4),
                'Exact Peierls/Kato connection obeys the analytic global gap bound')
            halves.append(canonical);frames.append(occupied)
            rows.append(dict(momentum=list(momentum),theta=theta,native_spatial_size=4,
                actual_canonical_number=2,actual_canonical_dimension=6,
                actual_stable_max=float(values[1]),actual_unstable_min=float(values[2]),
                actual_Kato_connection_norm=float(np.linalg.norm(connection,2)),
                actual_instantaneous_ground_overlap_with_theta0=float(abs(np.linalg.det(frames[0].conj().T@occupied))),
                actual_full_canonical_half_difference_from_theta0=float(np.linalg.norm(canonical-halves[0],2))))
    wrong=max(row['actual_full_canonical_half_difference_from_theta0'] for row in rows)
    check(wrong>.01,'Unconditioned physical harmonic fields cannot be frozen to the theta0 active band')
    return dict(rows=rows,wrong_global_frozen_halfstep_defect=wrong,
        all_canonical_occupations_retained=True,macro_admission=False,
        display_normalization='Each static half is displayed divided by its instantaneous canonical top singular value; this isolates eigenbundle change, not an energy-zero convention or the common lambda0 of a prepared source.',
        scope='Actual full four-spin four-mode canonical2 transfer and analytic Peierls/Kato jets, not a macro full-Gauss simulation.')



def matter_operator(one: np.ndarray,states):
    index={c:i for i,c in enumerate(states)}
    result=np.zeros((len(states),len(states)),complex)
    for j,c in enumerate(states):
        mask=sum(1<<x for x in c)
        for old in c:
            sign0=(-1)**sum(x<old for x in c);removed=mask^(1<<old)
            for new in range(8):
                if removed&(1<<new):continue
                sign=sign0*(-1)**((removed&((1<<new)-1)).bit_count())
                target=tuple(x for x in range(8) if (removed|(1<<new))&(1<<x))
                result[index[target],j]+=sign*one[new,old]
    return result


def native_current(gamma: np.ndarray,kappa: float,grid: int,space):
    epsilon=2e-4;jets=[]
    for theta in 2*math.pi*np.arange(grid)/grid:
        _,p0,h0,_,_=square_wilson(gamma,kappa,float(theta))
        _,pp,hp,_,_=square_wilson(gamma,kappa,float(theta+epsilon))
        _,pm,hm,_,_=square_wilson(gamma,kappa,float(theta-epsilon))
        current=matter_operator((hp-hm)/(2*epsilon),space['states'])-(math.log(pp)-math.log(pm))/(2*epsilon)*np.eye(len(space['states']))
        contact=matter_operator((hp+hm-2*h0)/epsilon**2,space['states'])-(math.log(pp)+math.log(pm)-2*math.log(p0))/epsilon**2*np.eye(len(space['states']))
        jets.append((current,contact))
    coeff=np.fft.fft(np.asarray(jets),axis=0)/grid
    answer=[]
    for order in range(2):
        matrix=np.array([[coeff[(n-m)%grid,order,i,j] for j,m,_ in space['physical']] for i,n,_ in space['physical']])
        check(bool(np.linalg.norm(matrix-matrix.conj().T)<1e-6),'Actual field/current/contact Fourier Hermiticity')
        answer.append(space['u'].conj().T@matrix@space['u'])
    return answer


def source_coefficients(gamma: np.ndarray,kappa: float,packet: np.ndarray,body_charge: int) -> np.ndarray:
    """Declared canonical electron/antiparticle filter, not bare gamma0."""
    _,_,_,vectors,values=square_wilson(gamma,kappa,0.)
    selected=values<1 if body_charge==1 else values>1
    frame=vectors[:,selected]
    raw=np.kron(packet[:,None],np.eye(2))
    result=frame@frame.conj().T@raw
    result/=np.linalg.norm(result,'fro')
    return result


def source_matrix(neutral,charged,coefficients: np.ndarray):
    """Integer dressed canonical a_plus†r or r†a_minus, charge1 Ref unchanged."""
    source=np.zeros((2*len(charged['physical']),len(neutral['physical'])),complex)
    paths=((0,0,0,0),(-1,0,0,0),(-1,-1,0,0),(0,0,0,1))
    outindex={c:i for i,c in enumerate(charged['states'])};bad=0.
    charge=charged['body_charge'];check(charge in (-1,1),'Source body charge must be plus/minus1')
    for j,(i,n,flux) in enumerate(neutral['physical']):
        occupied=neutral['states'][i];mask=sum(1<<x for x in occupied)
        for mode in range(8):
            present=bool(mask&(1<<mode))
            if present==(charge==1):continue
            x=mode//2
            out=tuple(sorted((*occupied,mode))) if charge==1 else tuple(z for z in occupied if z!=mode)
            target=outindex[out]
            newflux=tuple(a+charge*b for a,b in zip(flux,paths[x]));newn=newflux[3]
            row=charged['basis_index'].get((target,newn))
            if row is None:continue
            check(charged['physical'][row][2]==newflux,'Separated source integer-flux dressing')
            actual=rho_sites(out,charge)
            defect=[newflux[y]-newflux[(y-1)%4]-actual[y] for y in range(4)]
            bad=max(bad,max(abs(z) for z in defect))
            for ref in range(2):
                sign=(-1)**((mask&((1<<mode)-1)).bit_count()+(ref if charge==1 else len(out)))
                weight=coefficients[mode,ref] if charge==1 else coefficients[mode,ref].conjugate()
                source[ref*len(charged['physical'])+row,j]+=sign*weight
    return source,bad


def native_preparation(gamma4: np.ndarray,grid: int,cutoff: int,steps: int,body_charge: int=1) -> dict[str,Any]:
    gamma=two_spin_frame(gamma4);kappa=.09
    neutral=native_range(gamma,kappa,grid,cutoff,0)
    charged=native_range(gamma,kappa,grid,cutoff,body_charge)
    scale=float(neutral['values'][0]);ln=neutral['values']/scale
    lc=np.tile(charged['values']/scale,2)
    Un=neutral['u'];Uc=np.kron(np.eye(2),charged['u'])
    weights=ln**steps;weights/=weights.sum();rho=np.diag(weights)
    # Gauge-invariant magnetic plaquette; physical flux shifts ALL4 edges.
    holonomy=np.zeros((len(neutral['physical']),)*2,complex)
    for j,(i,n,_) in enumerate(neutral['physical']):
        out=neutral['basis_index'].get((i,n+1))
        if out is not None:holonomy[out,j]=1
    curl=(holonomy-holonomy.conj().T)/(2j);B=Un.conj().T@curl@Un
    half=np.diag(np.sqrt(ln));phi=half@B.conj().T@rho@B@half
    photon_gram=float(np.trace(phi).real);check(photon_gram>1e-12,'Actual retained photon Gram must be nonzero')
    rho_photon=phi/photon_gram
    extended_neutral=physical_basis(neutral['states'],cutoff+1,0)
    extn_index={(i,n):j for j,(i,n,_) in enumerate(extended_neutral)}
    raw_Bpsi=np.zeros((len(extended_neutral),len(ln)),complex)
    for j,(i,n,_) in enumerate(neutral['physical']):
        raw_Bpsi[extn_index[i,n+1]]-=.5j*Un[j]*np.sqrt(weights)
        raw_Bpsi[extn_index[i,n-1]]+=.5j*Un[j]*np.sqrt(weights)
    raw_photon_gram=float(np.linalg.norm(raw_Bpsi,'fro')**2)
    range_photon_gram=float(np.linalg.norm(B.conj().T@np.diag(np.sqrt(weights)),'fro')**2)
    f=np.array([.1,.8,.4,.1]);f/=np.linalg.norm(f)
    extended=dict(charged)
    extended['physical']=physical_basis(charged['states'],cutoff+1,body_charge)
    extended['basis_index']={(i,n):j for j,(i,n,_) in enumerate(extended['physical'])}
    embedding=np.zeros((len(extended['physical']),len(charged['physical'])))
    for j,(i,n,_) in enumerate(charged['physical']):embedding[extended['basis_index'][i,n],j]=1
    Uc_extended=np.kron(np.eye(2),embedding)@Uc
    coefficients=source_coefficients(gamma,kappa,f,body_charge)
    full_S,defect=source_matrix(neutral,extended,coefficients)
    bare_S,_=source_matrix(neutral,extended,np.kron(f[:,None],np.eye(2)))
    bare_compressed=Uc_extended.conj().T@bare_S@Un
    bare_source_gram=float(np.trace(np.diag(lc)@bare_compressed@rho_photon@bare_compressed.conj().T).real)
    raw_S=full_S@Un
    S=Uc_extended.conj().T@raw_S
    raw_source_gram=float(np.trace(raw_S.conj().T@raw_S@rho_photon).real)
    range_source_gram=float(np.trace(S.conj().T@S@rho_photon).real)
    source_half=np.diag(np.sqrt(lc))@S
    source=source_half@rho_photon@source_half.conj().T
    source_gram=float(np.trace(source).real);check(source_gram>1e-8,'Actual retained charged source Gram must be positive')
    rho_source=source/source_gram
    J,C=native_current(gamma,kappa,grid,neutral)
    energies=-np.log(ln);charged_energies=-np.log(lc)
    mean_thermal_energy=float(weights@energies)
    mean_photon_energy=float(np.real(np.diag(rho_photon))@energies)
    mean_source_energy=float(np.real(np.diag(rho_source))@charged_energies)
    check(raw_source_gram>=range_source_gram-1e-12 and range_source_gram>source_gram,'Native source discriminates null-range loss and positive halfstep retention')
    X=half@B.conj().T@np.diag(np.sqrt(weights))
    photon_terms=abs(X)**2/photon_gram
    rvalues,rvec=eigh(rho_photon);rhalf=(rvec*np.sqrt(np.maximum(rvalues,0)))@rvec.conj().T
    source_X=source_half@rhalf
    source_terms=abs(source_X)**2/source_gram
    constructive_joint_X=source_half@X/math.sqrt(photon_gram)
    joint_terms=abs(constructive_joint_X)**2/source_gram
    observations=[]
    for t in (0.,2.,8.,float(steps)):
        U=np.diag(np.exp(-1j*energies*t));evolved=U@rho_photon@U.conj().T
        observations.append(dict(time=t,photon_clock=complex(np.sum(photon_terms*np.exp(-1j*(energies[:,None]-energies[None,:])*t))),
            charged_clock=complex(np.sum(source_terms*np.exp(-1j*(charged_energies[:,None]-energies[None,:])*t))),
            constructive_joint_clock=complex(np.sum(joint_terms*np.exp(-1j*(charged_energies[:,None]-energies[None,:])*t))),
            density_trace=float(np.trace(evolved).real),current=float(np.trace(evolved@J).real),current_squared=float(np.trace(evolved@J@J).real),contact=float(np.trace(evolved@C).real)))
    _,_,_,v,e=square_wilson(gamma,kappa,0.)
    occupied=v[:,e>1];check(occupied.shape[1]==4,'Canonical unstable-eigenvalue sea')
    slater=np.linalg.det(occupied[np.asarray(neutral['states']),:])
    bare_defect=sum(abs(amplitude)**2*sum(z*z for z in rho_sites(c,False)) for amplitude,c in zip(slater,neutral['states']))
    check(defect==0 and bare_defect>1e-6,'Projected positive-T Gauss state must discriminate wrong bare sea times Haar')
    check(abs(np.trace(rho_photon)-1)<1e-12 and abs(np.trace(rho_source)-1)<1e-12,'Positive prepared photon/charged densities normalize by retained Gram')
    return dict(native_law='Invariant2-spin Phi block, open square spatial B/F, actual detBplus Gamma(T1), Tfhalf P_Haar Tfhalf; integer U1 Gauss canonical sector',
        native_kappa=kappa,grid=grid,electric_cutoff=cutoff,time_steps=steps,
        physical_neutral_dimension=len(neutral['physical']),neutral_nonzero_transfer_rank=len(ln),charged_nonzero_transfer_rank=len(lc),
        bare_Haar_range_rank=neutral['middle_dimension'],global_Tmin_asserted=False,
        nonzero_T_min=float(ln[-1]),thermal_trace=float(np.trace(rho).real),photon_trace=float(np.trace(rho_photon).real),charged_trace=float(np.trace(rho_source).real),
        photon_density_min=float(eigh(rho_photon,eigvals_only=True)[0]),charged_density_min=float(eigh(rho_source,eigvals_only=True)[0]),
        photon_raw_Gram=raw_photon_gram,photon_range_Gram=range_photon_gram,photon_retained_Gram=photon_gram,
        photon_range_loss=1-range_photon_gram/raw_photon_gram,
        source_raw_Gram=raw_source_gram,source_range_Gram=range_source_gram,source_retained_Gram=source_gram,
        source_range_loss=1-range_source_gram/raw_source_gram,source_boundary_retention=source_gram/range_source_gram,
        finite_H_means=dict(thermal=mean_thermal_energy,photon=mean_photon_energy,charged=mean_source_energy),
        adverse_unfiltered_source_H='Infinite: measured nonzero null-range component; positive-halfstep boundary removes it',
        source_packet=f.tolist(),source_Gauss_defect=defect,wrong_unstringed_separated_source_Gauss_defect=math.sqrt(2),
        declared_body_charge=body_charge,canonical_source_coefficients=coefficients.tolist(),
        adverse_bare_unfiltered_source_retained_Gram=bare_source_gram,
        wrong_free_canonical_sea_times_Haar_Gauss_squared=float(bare_defect),
        determinant_prefactor_retained=True,Ref=('nativeRef2->1/fullRef4->3' if body_charge==1 else 'nativeRef0->1/fullRef0->1')+'; operatorcharge1unchanged',
        current_operator_norm=float(np.linalg.norm(J,2)),contact_operator_norm=float(np.linalg.norm(C,2)),
        observations=observations,positive_range_boundary='Tn^1/2 Bdag rho B Tn^1/2 /actualGram; Tch^1/2 S rho_ph Sdag Tch^1/2 /actualGram',
        scope='Actual interacting field/Fock positive density, integerGauss/source/photon/current/unitary clocks on the native Galerkin law. Not a small-simulation renaming of the macro torus.')


def native_normal_preparation(gamma4: np.ndarray,steps: int,nodes: int=7,return_runtime: bool=False,*,body_charge: int=1) -> dict[str,Any]:
    """Root radial normal coordinate coupled to the ACTUAL charged square sea."""
    gamma=two_spin_frame(gamma4);sigma=.03;radius=.12;kappa=.09;cutoff=2;grid=32
    y=np.linspace(-.75*radius,.75*radius,nodes);dy=float(y[1]-y[0])
    logk=-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y))
    k=np.exp(logk);k/=np.linalg.norm(k)
    derivative=np.zeros((nodes-1,nodes))
    for j in range(nodes-1):
        derivative[j,j]=math.sqrt(k[j+1]/k[j])/dy
        derivative[j,j+1]=-math.sqrt(k[j]/k[j+1])/dy
    normal_H=2*sigma*sigma*derivative.T@derivative
    normal_E,normal_U=eigh(normal_H)
    check(abs(float(normal_E[0]))<1e-12 and float(normal_E[1])>.5,'Actual compact normal ground and nonzero massive gap')
    normal_half=(normal_U*np.exp(-normal_E/2))@normal_U.T
    neutral=[];charged=[]
    for value in y:
        actual=gamma.copy();actual[1]*=1+float(value)
        neutral.append(native_range(actual,kappa,grid,cutoff,0))
        charged.append(native_range(actual,kappa,grid,cutoff,body_charge))
    nn=len(neutral[0]['physical']);nc=len(charged[0]['physical'])
    Vn=np.kron(normal_half,np.eye(nn))@block_diag(*(s['V'] for s in neutral))
    Vc=np.kron(normal_half,np.eye(nc))@block_diag(*(s['V'] for s in charged))
    Un,sn,_=np.linalg.svd(Vn,full_matrices=False)
    Uc,sc,_=np.linalg.svd(Vc,full_matrices=False);Uc=np.kron(np.eye(2),Uc)
    scale=float(sn[0]**2);ln=sn*sn/scale;lc=np.tile(sc*sc/scale,2)
    check(bool(ln.min()>0 and lc.min()>0),'Normal-field proof uses only its actual nonzero range')
    ground=np.outer(k,k);Pn=Un.conj().T@np.kron(ground,np.eye(nn))@Un
    weights=ln**steps;weights/=weights.sum();half=np.diag(np.sqrt(ln))
    base=half@Pn@np.diag(weights)@Pn@half
    normal_gram=float(np.trace(base).real);base/=normal_gram
    holonomy=np.zeros((nn,nn),complex)
    for j,(i,n,_) in enumerate(neutral[0]['physical']):
        out=neutral[0]['basis_index'].get((i,n+1))
        if out is not None:holonomy[out,j]=1
    curl=(holonomy-holonomy.conj().T)/(2j)
    B=Un.conj().T@np.kron(np.eye(nodes),curl)@Un
    ph=half@B.conj().T@base@B@half;ph_gram=float(np.trace(ph).real);ph/=ph_gram
    packet=np.array([.1,.8,.4,.1]);packet/=np.linalg.norm(packet)
    coefficients=source_coefficients(gamma,kappa,packet,body_charge)
    single,defect=source_matrix(neutral[0],charged[0],coefficients)
    full=np.zeros((2*nodes*nc,nodes*nn),complex)
    for ref in range(2):
        for j in range(nodes):full[(ref*nodes+j)*nc:(ref*nodes+j+1)*nc,j*nn:(j+1)*nn]=single[ref*nc:(ref+1)*nc]
    source=np.diag(np.sqrt(lc))@Uc.conj().T@full@Un
    ch=source@ph@source.conj().T;ch_gram=float(np.trace(ch).real);ch/=ch_gram
    Qn=np.eye(len(ln))-Pn
    Pc=Uc.conj().T@np.kron(np.eye(2),np.kron(ground,np.eye(nc)))@Uc
    Qc=np.eye(len(lc))-Pc
    outside=np.diag((abs(y)>radius/2).astype(float))
    outside_n=Un.conj().T@np.kron(outside,np.eye(nn))@Un
    outside_c=Uc.conj().T@np.kron(np.eye(2),np.kron(outside,np.eye(nc)))@Uc
    En=-np.log(ln);Ec=-np.log(lc);observations=[]
    base_X=half@Pn@np.diag(np.sqrt(weights))/math.sqrt(normal_gram)
    photon_X=half@B.conj().T@base_X/math.sqrt(ph_gram)
    # Keep raw S on its entire integer-Gauss codomain, including the exact
    # null space of charged K.  No inverse halfstep reconstructs this vector.
    extended=dict(charged[0])
    extended['physical']=physical_basis(charged[0]['states'],cutoff+1,body_charge)
    extended['basis_index']={(i,n):j for j,(i,n,_) in enumerate(extended['physical'])}
    ne=len(extended['physical']);embedding=np.zeros((ne,nc))
    for j,(i,n,_) in enumerate(charged[0]['physical']):
        embedding[extended['basis_index'][i,n],j]=1
    extended_isometry=np.kron(np.eye(2),np.kron(np.eye(nodes),embedding))@Uc
    boundary_factor=np.kron(np.eye(2),Vc)/math.sqrt(scale)
    extended_factor=np.kron(np.eye(2),np.kron(np.eye(nodes),embedding))@boundary_factor
    raw_packet=[];range_packet=[];half_packet=[]
    for site in range(4):
        component=np.zeros_like(coefficients)
        component[2*site:2*site+2]=coefficients[2*site:2*site+2]
        local,local_defect=source_matrix(neutral[0],extended,component)
        check(local_defect==0,'Every coherent packet component has literal integer Gauss')
        operator=np.zeros((2*nodes*ne,nodes*nn),complex)
        for ref in range(2):
            for j in range(nodes):
                operator[(ref*nodes+j)*ne:(ref*nodes+j+1)*ne,j*nn:(j+1)*nn]=local[ref*ne:(ref+1)*ne]
        raw_component=operator@Un@photon_X
        raw_packet.append(raw_component)
        range_packet.append(extended_isometry.conj().T@raw_component)
        half_packet.append(extended_factor.conj().T@raw_component)
    raw_source_X=sum(raw_packet);range_source_X=sum(range_packet)
    minor_source_X=sum(half_packet)
    check(abs(float(np.linalg.norm(minor_source_X,'fro')**2)-ch_gram)<1e-10,
        'Cauchy-Binet charged boundary minors reproduce the actual retained source Gram')
    check(float(np.linalg.norm(raw_source_X,'fro')**2)<=float(np.linalg.norm(coefficients,'fro')**2)+1e-10,
        'Actual raw sameK source satisfies the canonical CAR cap without a range/occupation assumption')
    raw_gram=np.array([[np.vdot(a,b) for b in raw_packet] for a in raw_packet])
    range_gram=np.array([[np.vdot(a,b) for b in range_packet] for a in range_packet])
    half_gram=np.array([[np.vdot(a,b) for b in half_packet] for a in half_packet])
    source_X=source@photon_X/math.sqrt(ch_gram)
    check(bool(np.linalg.norm(source_X@source_X.conj().T-ch)<1e-10),'Constructive normal/source purification matches actual density')
    for time in (0.,float(steps),4.*steps):
        un=np.exp(-1j*En*time);uc=np.exp(-1j*Ec*time)
        pn=un[:,None]*ph*un.conj()[None,:];pc=uc[:,None]*ch*uc.conj()[None,:]
        observations.append(dict(time=time,photon_normal_excited_weight=float(np.trace(Qn@pn).real),
            constructive_joint_clock=complex(np.sum(abs(source_X)**2*np.exp(-1j*(Ec[:,None]-En[None,:])*time))),
            charged_normal_excited_weight=float(np.trace(Qc@pc).real),
            photon_inner_shell_tail=float(np.trace(outside_n@pn).real),
            charged_inner_shell_tail=float(np.trace(outside_c@pc).real)))
    check(ph_gram>0 and ch_gram>0 and defect==0,'Actual coupled normal/Gauss/photon/source preparation')
    summary:dict[str,Any]=dict(radial_law='One-dimensional Galerkin of H=2Omega sigma² A†A, A k=0; same normalized compact radial k, Omega1',
        primitive_Phi_normal='Phi1(Y)=(1+Y)gamma1: changes meanGram, NOT a spin/gauge rotation',
        sigma=sigma,radius=radius,normal_nodes=nodes,normal_energies=normal_E.tolist(),
        exact_discrete_ground_residual=float(np.linalg.norm(normal_H@k)),
        full_positive_T='Knormal½ block[Tfhalf(Y) P_Haar Tfhalf(Y)] Knormal½, actual Wilson CAR minors and Gauss',
        nonzero_neutral_rank=len(ln),nonzero_charged_rank=len(lc),normal_preparation_Gram=normal_gram,
        exact_preparation='K½ Pphi K^Nt Pphi K½ /actualtrace, Pphi=|normalizedk><k|; positive photon/source boundaries follow',
        photon_retained_Gram=ph_gram,source_retained_Gram=ch_gram,
        microscopic_boundary_minors=dict(raw_wholepacket_M0=float(np.linalg.norm(raw_source_X,'fro')**2),
            nonzero_range_wholepacket_M0=float(np.linalg.norm(range_source_X,'fro')**2),
            positive_charged_boundary_M1=float(np.linalg.norm(minor_source_X,'fro')**2),
            actual_reference_CAR_source_norm_squared_upper=float(np.linalg.norm(coefficients,'fro')**2),
            CAR_bound='Filled Ref neutral: orthogonal holes give SdagS=sum_alpha a(f_alpha)a†(f_alpha)<=sum_alpha||f_alpha||². Empty Ref neutral: orthogonal particles give SdagS=sum_alpha a†(f_alpha)a(f_alpha)<=same cap. Integer flux/Wilson strings preserve coefficient norm; no occupation or source/range admission is assumed.',
            raw_packet_cross_Gram=raw_gram,range_packet_cross_Gram=range_gram,
            charged_boundary_packet_cross_Gram=half_gram,
            packet_labels=['site0','site1','site2','site3'],
            neutral_lambda0=scale,photon_normalization=ph_gram,normal_normalization=normal_gram,
            construction='w=Sraw Un Xph; Kch=Vch Vchdag; M1=||Vchdag w||HS²/lambda0. All offdiagonal source-packet minors retained.',
            native_photon_number_coherence='curl=(shift4flux-shift4fluxdag)/(2i); both orientations enter the SAME purifier before Gram.',
            boundary_factor_scope='Actual native joint normal/Wilson detB CAR Fourier-minor factor, extended by exact K-null electric boundary; not a guessed covariance.'),
        base_normal_excited_weight=float(np.trace(Qn@base).real),
        photon_H_mean=float(np.real(np.diag(ph))@En),charged_H_mean=float(np.real(np.diag(ch))@Ec),
        prepared_radial_ground_inner_tail=float(k@outside@k),source_integer_Gauss_defect=defect,
        observations=observations,scope='Actual quantum normal-field and photon/electron/Ref Gauss density, not a macro ground-state identification. All native normal excited states remain.')
    macro_cut=.25
    macro_projector=np.diag((Ec<=macro_cut).astype(float))
    macro_retention=float(np.trace(macro_projector@ch).real)
    summary['adverse_macro_energy_cut_on_wrong_native_parameters']=dict(cut=macro_cut,retained_Gram=macro_retention,
        minimum_actual_charged_energy=float(Ec.min()),
        meaning='This kappa=.09 square has hard transfer energies, NOT the macro a=.01 induced-photon law; a fixed-number bare source norm cannot assert its low-energy admission.')
    if not return_runtime:return summary
    block_frame=two_spin_embedding(gamma4)
    # The native source already includes the two-mode Ref CAR sign.  This
    # compensates it while embedding annihilation into the actual Ref4 holes.
    hole_embedding=np.array([[(-1)**(a-c)*block_frame[a,c].conjugate() for c in range(2)] for a in range(4)])
    return dict(summary=summary,runtime=dict(source_density=ch,photon_density=ph,base_density=base,
        source_purification=source_X,initial_thermal_eigenvalues=ln,
        raw_source_purification=raw_source_X,range_source_purification=range_source_X,
        raw_packet_cross_Gram=raw_gram,range_packet_cross_Gram=range_gram,
        boundary_packet_cross_Gram=half_gram,neutral_lambda0=scale,
        range_packet_purifications=range_packet,
        charged_eigenvalues=lc,neutral_eigenvalues=ln,charged_range_isometry=Uc,
        neutral_range_isometry=Un,source_half=source,source_retained_Gram=ch_gram,
        charged_normal_ground_projector=Pc,charged_inner_shell_projector=outside_c,
        normal_coordinates=y,normal_ground=k,normal_hamiltonian=normal_H,
        charged_physical_basis=charged[0]['physical'],charged_Fock_occupations=charged[0]['states'],
        physical_ordering=('Refhole2' if body_charge==1 else 'Refparticle2')+' major, normal-node major, (chargedFockIndex,integerE3,all4flux)',
        nonzero_ordering=('Refhole2' if body_charge==1 else 'Refparticle2')+' major, jointnormal/field K singular-vector index',
        primitive_Clifford_embedding=block_frame,Ref4_hole_embedding=hole_embedding,
        gamma4=gamma4,body_charge=body_charge,full_Ref_operator_charge=1,full_Ref_background=4 if body_charge==1 else 0,
        Ref4_particle_embedding=block_frame,source_coefficients=coefficients,
        angular_limit='One anisotropic normal coordinate is not the full45 angular Cartesian body law; do not call a naked Gamma rotation physicalNoether. Ref4 hole SU2 is exact independently.'))


def positive_boundary_rayleigh(runtime: dict[str,Any],cut: float) -> dict[str,Any]:
    """Same-lambda0 exact spectral estimate, with explicit K-null subtraction."""
    lc=runtime['charged_eigenvalues'];threshold=math.exp(-cut)
    raw=runtime['raw_source_purification'];ranged=runtime['range_source_purification']
    half=np.sqrt(lc)[:,None]*ranged
    M0=float(np.linalg.norm(raw,'fro')**2)
    M0range=float(np.linalg.norm(ranged,'fro')**2)
    M1=float(np.linalg.norm(half,'fro')**2)
    admitted=(lc>=threshold)&(lc<=1.)
    above=lc>1.
    # If charged K has spectrum above lambda0, subtract that contribution;
    # positivity alone never licenses treating it as PE[0,cut].
    upper_contribution=float(np.linalg.norm(half[above],'fro')**2)
    upper_raw=float(np.linalg.norm(ranged[above],'fro')**2)
    naive=M1-threshold*M0-upper_contribution
    sharper=M1-threshold*(M0range-upper_raw)-upper_contribution
    numerator=float(np.linalg.norm(half[admitted],'fro')**2)
    family=[np.sqrt(lc[admitted])[:,None]*v[admitted] for v in runtime['range_packet_purifications']]
    pe_gram=np.array([[np.vdot(a,b) for b in family] for a in family])
    above_family=[np.sqrt(lc[above]-threshold)[:,None]*v[above]
        for v in runtime['range_packet_purifications']]
    above_gram=np.array([[np.vdot(a,b) for b in above_family] for a in above_family])
    family_lower=runtime['boundary_packet_cross_Gram']-threshold*runtime['range_packet_cross_Gram']-above_gram
    family_residual=(pe_gram-family_lower)
    family_residual=(family_residual+family_residual.conj().T)/2
    family_residual_min=float(eigh(family_residual,eigvals_only=True)[0])
    check(family_residual_min>=-1e-10,'Actual whole coherent source family obeys spectral Gram Loewner inequality')
    check(abs(pe_gram.sum().real-numerator)<1e-10,'Wholepacket PE Gram retains every source cross term')
    check(numerator+1e-10>=max(naive,sharper),'Actual PE numerator obeys exact positiveK Rayleigh lower bound')
    check(abs(M1-runtime['source_retained_Gram'])<1e-10,'SameK charged halfstep/minor normalization')
    return dict(energy_cut=cut,neutral_lambda0=runtime['neutral_lambda0'],
        charged_threshold=runtime['neutral_lambda0']*threshold,
        raw_wholepacket_M0=M0,range_wholepacket_M0=M0range,
        exact_Knull_source_norm_squared=M0-M0range,positive_boundary_M1=M1,
        M1_over_raw_M0=M1/M0,M1_over_range_M0=M1/M0range,
        charged_above_neutral_lambda0_numerator=upper_contribution,
        raw_Rayleigh_PEGram_lower=naive,range_Rayleigh_PEGram_lower=sharper,
        actual_PEGram=numerator,wholepacket_PE_cross_Gram=pe_gram,
        sameK_wholepacket_PE_cross_Gram_lower_matrix=family_lower,
        sameK_cross_Gram_Loewner_residual_min=family_residual_min,
        actual_PE_cross_Gram_eigenvalues=eigh((pe_gram+pe_gram.conj().T)/2,eigvals_only=True),
        sameK_source_normalization=M1,PE_retention_probability=numerator/M1,
        sufficient_raw_Rayleigh_admission=naive>0,
        sufficient_range_Rayleigh_admission=sharper>0,
        observed_native_PE_admission=numerator>0,
        proof='K/lambda0=UUdag positive spectral factor; w=Sraw Xi_ph. Q=(0,K/lambda0<exp(-cut)) excludes exact K-null, so ||PE sqrt(K/lambda0)w||² >= M1-exp(-cut)||P_nonzero w||², after explicit subtraction of K/lambda0>1. Raw bound is weaker but valid. No inverse-log, source/PE commutation or charged own-max normalization.',
        numeric_scope='Observed double-precision canonical boundary-minor calculation with exact-arithmetic spectral inequality; not an interval-arithmetic enclosure or macro theorem.',
        macro_admission=False,finiteSCREEN_source_unchanged=True)


def energy_cut_source(runtime: dict[str,Any],cut: float) -> dict[str,Any]:
    """Actual spectral preparation on the common-neutral nonzero range."""
    levels=-np.log(runtime['charged_eigenvalues'])
    admitted=(levels>=0)&(levels<=cut)
    projector=np.diag(admitted.astype(float))
    raw=projector@runtime['source_density']@projector
    gram=float(np.trace(raw).real)
    if gram<=0:
        raise ArithmeticError('Actual charged source has ZERO retained energy-band Gram')
    density=raw/gram
    purification=projector@runtime['source_purification']/math.sqrt(gram)
    check(np.linalg.norm(purification@purification.conj().T-density)<1e-10,'Energy-cut state uses actual constructive Gauss purifier')
    mean=float(np.real(np.diag(density))@levels)
    check(mean<=cut and abs(np.trace(density)-1)<1e-10,'Actual admitted quantum source has finite energy')
    return dict(density=density,purification=purification,projector=projector,
        summary=dict(energy_cut=cut,retained_probability=gram,
            total_source_retained_Gram=gram*runtime['source_retained_Gram'],
            energy_mean=mean,maximum_admitted_energy=float(levels[admitted].max()),
            actual_high_energy_expectation=0.,body_charge=runtime['body_charge'],
            common_neutral_normalization=True,full_Ref_operator_charge=1,
            full_Ref_background=runtime['full_Ref_background'],
            Gauss_scope='Spectral projector of actual native positiveK on SAME literal integerU1-Gauss range; Ref3/1 and all native quantum normal/source fields retained. Not fullmacroSU4 admission.',
            scope='Actual native kappa=.09 hard square; energycut6 is NOT admitted macro .25 nor a finiteSCREEN source change'))


def native_toron(cutoff: int):
    beta,L,Nt,R,duration=20.,4,8,.7,.1
    n=np.arange(-cutoff,cutoff+1);E=n*n/(2*beta*L)
    weights=np.exp(-Nt*E);weights/=weights.sum()
    z,w=roots_legendre(240);theta=R*z;g=(1-z*z)**2
    diff=n[:,None]-n[None,:]
    F=R*np.einsum('q,ijq,q->ij',w,np.exp(-1j*diff[:,:,None]*theta),g)/(2*math.pi)
    raw=(F*weights)@F.conj().T;p=float(np.trace(raw).real);rho=raw/p
    exterior=np.eye(len(n))-R/math.pi*np.sinc(diff*R/math.pi)
    U=np.diag(np.exp(-1j*E*duration));evolved=U@rho@U.conj().T
    p2=float(weights@(n*n));bound2=duration**2*(63/(8*(beta*L)**2*R**4)+3*p2/((beta*L)**2*R**2))
    correct=float(E[cutoff+1]);wrong=1/(2*beta*L**3)
    qoperator=np.zeros((len(n),len(n)),complex)
    qoperator[np.arange(1,len(n)),np.arange(len(n)-1)]=1
    qoperator-=np.eye(len(n))
    initial_q2=float(np.trace(rho@qoperator.conj().T@qoperator).real)
    evolved_q2=float(np.trace(evolved@qoperator.conj().T@qoperator).real)
    q_rms_upper=R/math.sqrt(11)+duration*(math.sqrt(p2+3/R**2)+.5)/(beta*L)
    check(math.sqrt(evolved_q2)<=q_rms_upper+1e-8,'Periodic holonomy observable has a real-time RMS bound')
    check(abs(correct/wrong-L*L)<1e-12,'Holonomy kinetic uses beta L, not beta V')
    overlap_rows=[]
    for separation in (1,2,3):
        overlap=float(np.sum(abs(F)**2*np.exp(-(Nt-separation)*E[:,None]-separation*E[None,:]))/np.exp(-Nt*E).sum())
        continuum_probability=128*R/(315*math.pi)
        ratio=overlap/continuum_probability
        lower=1-3*separation/(2*beta*L*R**2)
        check(ratio>=lower-1e-8,'Actual reflected toron overlap obeys its Fourier lower bound')
        overlap_rows.append(dict(separation=separation,actual_reflected_overlap=overlap,
            continuum_overlap_over_Haar_probability=ratio,continuum_ratio_lower=lower))
    return dict(beta=beta,L=L,Nt=Nt,R=R,duration=duration,Fourier_cutoff=cutoff,
        trace=float(np.trace(rho).real),postselection_probability=p,exact_Haar_filter_probability=128*R/(315*math.pi),
        initial_Fourier_projection_leakage=float(np.trace(rho@exterior).real),evolved_outside_window_probability=float(np.trace(evolved@exterior).real),
        continuum_thermal_commutator_norm_squared_upper=bound2,
        thermal_momentum_second_moment=p2,filtered_momentum_second_moment=float(np.trace(rho@np.diag(n*n)).real),
        exact_continuum_filtered_momentum_second_moment=p2+3/R**2,
        reflected_boundary_overlap=overlap_rows,
        rarity_cancellation_proof='In a stationary free toron bridge BOTH time marginals are normalized Haar, so E|f_L f_R F|<=||F||infty sqrt(E f_L² E f_R²)=||F||infty p. Fourier e^-x>=1-x and symmetric thermal n kill the cross nm term: E f_L f_R>=p[1-3ell/(2mR²)] in one toron, >=p3[1-9ell/(2mR²)] in three. This is scalar Cauchy plus actual heat overlap, not RN-to-OS positivity or an unproved cancellation of tiny p.',
        initial_periodic_holonomy_second_moment=initial_q2,evolved_periodic_holonomy_second_moment=evolved_q2,
        continuum_periodic_holonomy_RMS_upper=q_rms_upper,
        RMS_proof='Q=exp(itheta)-1 is global periodic observable; [H,Q]=exp(itheta)(2n+1)/(2m), n commutes H. Filter theta² mean R²/11; principal |theta|<=pi|Q|/2. No probability-to-amplitude or ground substitution.',
        wrong_Haar_inside_window_probability=R/math.pi,wrong_volume_kinetic_ratio=correct/wrong)


def toron_static_response(p: Parameters,gamma: np.ndarray) -> dict[str,Any]:
    """Measure actual canonical poles/projectors/jets below double precision."""
    with mp.workdps(130):
        Rh=mp.mpf('1e-50');L=p.spatial_size;T=p.time_slices
        _,frame=eigh(gamma[0]);frame=frame[:,::-1]
        transformed=np.array([frame.conj().T@g@frame for g in gamma])
        gammas=[mp.matrix([[mp.mpc(str(z.real),str(z.imag)) for z in row] for row in g]) for g in transformed]
        def point(momentum):
            b=mp.exp(mp.mpf('.01'))+sum(1-mp.cos(x) for x in momentum)
            F=sum((1j*gammas[j+1]*mp.sin(momentum[j]) for j in range(3)),mp.zeros(4))
            triangular=mp.eye(4)
            for a in range(2):
                for c in range(2):triangular[a,c+2]=-F[a,c+2]
            diagonal=mp.diag([1/b,1/b,b,b])
            one=triangular.H*diagonal*triangular
            values,vectors=mp.eighe(one);stable=vectors[:,:2]
            projector=stable*stable.H;currents=[];contacts=[]
            for axis in range(3):
                sine,cosine=mp.sin(momentum[axis]),mp.cos(momentum[axis])
                first=mp.zeros(4);second=mp.zeros(4)
                for a in range(2):
                    for c in range(2):
                        first[a,c+2]=-1j*gammas[axis+1][a,c+2]*cosine
                        second[a,c+2]=1j*gammas[axis+1][a,c+2]*sine
                d1=mp.diag([-sine/b**2]*2+[sine]*2)
                d2=mp.diag([2*sine*sine/b**3-cosine/b**2]*2+[cosine]*2)
                jet1=first.H*diagonal*triangular+triangular.H*d1*triangular+triangular.H*diagonal*first
                jet2=(second.H*diagonal*triangular+triangular.H*diagonal*second+
                    2*first.H*diagonal*first+2*first.H*d1*triangular+
                    2*triangular.H*d1*first+triangular.H*d2*triangular)
                j1=vectors.H*jet1*vectors;j2=vectors.H*jet2*vectors
                currents.append(mp.re(sum(-j1[s,s]/values[s] for s in range(2))/2))
                contact=mp.mpf(0)
                for s in range(2):
                    contact-=mp.re(j2[s,s])/values[s]/2
                    for t in range(4):
                        ratio=(values[t]-values[s])/values[s]
                        if abs(ratio)<mp.mpf('1e-20'):
                            divided=sum((-1)**n*ratio**(n-2)/n for n in range(2,20))/values[s]**2
                        else:divided=(-mp.log1p(ratio)+ratio)/(values[s]**2*ratio**2)
                        contact+=divided*abs(j1[s,t])**2
                contacts.append(contact)
            return values,stable,projector,currents,contacts
        rows=[];extrema=dict(pole=mp.mpf(0),projector=mp.mpf(0),Gram=mp.mpf(0),current=mp.mpf(0),contact=mp.mpf(0))
        for eta in itertools.product((0,1),repeat=3):
            for offset in (-1,0,1):
                base=[mp.pi*(2*offset+eta[0])/L,mp.pi*eta[1]/L,mp.pi*eta[2]/L]
                e0,v0,P0,j0,c0=point(base)
                source=v0*mp.diag([1/mp.sqrt(2*e0[s]) for s in range(2)])
                local={key:mp.mpf(0) for key in extrema}
                for signs in itertools.product((-1,1),repeat=3):
                    e1,v1,P1,j1,c1=point([x+sign*Rh for x,sign in zip(base,signs)])
                    delta=mp.matrix(P1-P0)
                    projector=mp.sqrt(sum(abs(z)**2 for z in delta))
                    transfer=v1*mp.diag([e1[s] for s in range(2)])*v1.H
                    gram_matrix=source.H*transfer*source
                    gram=mp.re(sum(gram_matrix[s,s] for s in range(2)))
                    values=dict(pole=max(abs(mp.log(e1[s]/e0[s])) for s in range(2)),
                        projector=projector,Gram=abs(gram-1),
                        current=max(abs(a-b) for a,b in zip(j0,j1)),
                        contact=max(abs(a-b) for a,b in zip(c0,c1)))
                    for key,value in values.items():local[key]=max(local[key],value);extrema[key]=max(extrema[key],value)
                rows.append(dict(statistical_toron=list(eta),electron_offset=offset,
                    measured_corner_maxima={key:mp.nstr(value,18) for key,value in local.items()}))
        phase=float(abs(mp.exp(1j*Rh*(3*L/2))-1))
        # Uniform interior bounds are independent of the corner probes.
        # On |p|_1<.1, B~1 and F<=.2, so first three log jets are <30720.
        # ||dT/dp_j||<512 and the stable/unstable gap is >=.019.
        band=2*3*512*float(Rh)/.019
        pole=3*30720*float(Rh)
        source_change=8*(phase+band+pole)
        gram_change=2*8*source_change+64*pole
        mode_count=100;leg=20.;C=30720.;duration=T+4
        H1=mode_count*C*leg;H2=mode_count*C*leg*leg
        dH0=3*mode_count*C*float(Rh);dH1=dH0*leg;dH2=dH0*leg*leg
        word=duration*H2+duration**2*H1*H1/2
        changed=(duration*dH2+duration**2*H1*dH1+duration**2*H2*dH0+
            duration**3*H1*H1*dH0+2*(source_change+gram_change)*word)
        amplitude_log=-p.beta_power-3*math.log10(L)+math.log10(64*changed)
        return dict(link_angle_radius='1e-50',mpmath_digits=130,source_frame='Actual positive canonical T1 band; charge1 Ref4 pole conditional probe, NOT fullGauss Gram',
            rows=rows,measured_maxima={key:mp.nstr(value,18) for key,value in extrema.items()},
            integer_string_phase_change_upper=phase,positive_band_projector_change_upper=band,
            source_operator_change_upper=source_change,retained_Gram_static_correction_upper=gram_change,
            toron_selected_grade2_absolute_error_log10=amplitude_log,
            grading='Static toron correction changes grade2 RELATIVELY; it is not a coexact g² Gram grade4 effect',
            actual_operator_jets='Current/contact are first/second Loewner -logT1 Peierls jets in its charged positive eigenband, neutral/vacuum prefactor subtracted',
            uniform_interior_proof='|p|1<.1, B>=exp(.01), ||F||<=.2, canonical eigenvalue gap>=.019; ||dT1||<=512 and first3 log jets<=30720. Davis-Kahan projector bound, integerstring telescoping, positive Duhamel time-word bound on executed100-CAR support. Corners are probes, not the interior proof.',
            auxiliary_toron_semantics='ALL noncontractible auxiliary loops separately bounded; no Naux operator-norm toron factor and no unprepared free-sea replacement',
            cancellation='No odd-in-toron cancellation required for these conservative positive bounds.')


def finite_box_photon_partition(p: Parameters,cube_radius: int=64) -> dict[str,Any]:
    """Actual lattice dispersion sum with a positive finite-box tail bound."""
    L,Nt=p.spatial_size,p.time_slices
    n=np.arange(-cube_radius,cube_radius+1,dtype=float)
    s=np.sin(math.pi*n/L)**2
    energy=2*np.arcsinh(np.sqrt(s[:,None,None]+s[None,:,None]+s[None,None,:]))
    energy[cube_radius,cube_radius,cube_radius]=np.inf
    gamma=4*math.asinh(math.sqrt(3))/(3*L)
    start=cube_radius+1
    def positive_tail(tau):
        z=math.exp(-tau*gamma);d=1-z;a=start
        second=z**a*(a*a/d+2*a*z/d**2+z*(1+z)/d**3)
        zeroth=z**a/d
        return 2*(4*second+2*zeroth)/(1-z**a)
    finite_energy=np.where(np.isfinite(energy),energy,0.)
    def positive_energy_tail(tau):
        z=math.exp(-tau*gamma);d=1-z;a=start
        first=z**a*(a/d+z/d**2)
        third=z**a*(a**3/d+3*a*a*z/d**2+3*a*z*(1+z)/d**3+z*(1+4*z+z*z)/d**4)
        return 2*gamma*(4*third+2*first)/(1-z**a)
    rows=[]
    for fraction in (1.,.5,.75,.8,.85):
        tau=Nt*fraction
        central=float(-2*np.log1p(-np.exp(-tau*energy)).sum())
        tail=positive_tail(tau)
        guard=2**-30*(1+central+tail)
        mean_actual=float(2*np.sum(finite_energy/np.expm1(np.minimum(tau*energy,700))))
        mean_tail=positive_energy_tail(tau)
        rows.append(dict(inverse_temperature=tau,fraction=fraction,
            actual_finite_box_central_logZ=central,positive_omitted_box_logZ_upper=tail,
            rounding_guard=guard,logZ_lower=central-guard,logZ_upper=central+tail+guard,
            mean_energy_actual_central=mean_actual,mean_energy_positive_tail_upper=mean_tail,
            mean_energy_upper=mean_actual+mean_tail+2**-40))
    mean_central=rows[0]['mean_energy_actual_central']
    mean_tail=rows[0]['mean_energy_positive_tail_upper']
    mean_upper=rows[0]['mean_energy_upper']
    return dict(cube_radius=cube_radius,actual_nonzero_modes=(2*cube_radius+1)**3-1,
        physical_polarizations=2,rows=rows,mean_energy_actual_central=mean_central,
        mean_energy_positive_tail_upper=mean_tail,mean_energy_upper=mean_upper,
        dispersion='E(n)=2asinh(sqrt(sum sin²(pi n_j/L))) on the ACTUAL finite periodic box',
        tail_proof='E(n)>=gamma |n|1, gamma=4asinh(sqrt3)/(3L). Outside evaluated cube |n|1>=N+1; integer Z3 shell count4s²+2. Positive geometric moments bound all omitted finite-box modes by extending to infinite shells. No continuum thermal number or derivative of logZ upper used.',
        neutral_mass_normalization='ALL charged halfsteps retain the SAME neutral transfer lambda0; no charged-sector own-max rescale.')


def prepared_energy_support(p: Parameters,gamma: np.ndarray) -> dict[str,Any]:
    """Execute actual prepared first/second-current kernel; bound its whole graph."""
    c=WilsonConsumer(gamma,.01,1.)
    step=2*math.pi/p.spatial_size
    central=np.array([0.,0.,2*math.asin(math.sinh(.005/2))])
    direction=np.array([math.sin(1.2),0.,math.cos(1.2)])
    radius=cast(float,brentq(lambda r:c.energy(central-r*direction)+c.photon_energy(r*direction)-.015,0.,.01,xtol=1e-15))
    qin=step*np.rint(central/step);qout=step*np.rint(radius*direction/step)
    rows=[]
    for eta,pe,pi,po in (((0,0,0),-1,-2,-1),((1,1,1),1,2,1)):
        pin=math.pi*np.array(eta)/p.spatial_size+np.array([pe*step,0.,0.])
        qi=qin+np.array([pi*step,0.,0.]);qo=qout+np.array([po*step,0.,0.])
        kernel=connected_transfer_kernel(p,gamma,pin,qi,qo)
        rows.append(dict(statistical_toron=list(eta),packet_offsets=[pe,pi,po],pin=pin.tolist(),qin=qi.tolist(),qout=qo.tolist(),
            actual_first_second_virtual_energy_max=kernel['maximum_grade2_virtual_energy_lattice'],
            actual_source_receiver_energy=kernel['energies'],virtual_counts=kernel['virtual_counts'],
            prepared_response=kernel['prepared_response'],actual_neutral_endpoint=kernel['prepared_neutral_endpoint_correction'],
            prepared_contract=kernel['prepared_state_contract']))
    maximum=0.;plane_maximum=0.;hard_photon=0.
    incoming_offsets=list(itertools.product(range(-2,3),repeat=3))
    for eta,pe,incoming,po in itertools.product(itertools.product((0,1),repeat=3),range(-1,2),incoming_offsets,range(-1,2)):
        electron=math.pi*np.array(eta)/p.spatial_size+np.array([pe*step,0.,0.])
        qi=qin+step*np.array(incoming);qo=qout+np.array([po*step,0.,0.])
        hard_photon=max(hard_photon,c.photon_energy(qi),c.photon_energy(qo))
        for i,j in itertools.product(range(-2,3),repeat=2):
            value=c.energy(electron+i*qi+j*qo);maximum=max(maximum,value)
            if incoming[1:]==(0,0):plane_maximum=max(plane_maximum,value)
    upper=3*maximum+2*hard_photon+2**-40
    check(max(r['actual_first_second_virtual_energy_max'] for r in rows)<=upper,'Actual current/endpoint support obeys finite graph bound')
    check(upper<.2,'Finite-energy source cutoff reserves actual virtual support')
    return dict(actual_first_second_kernel_rows=rows,uniform_all8_full125_grade2_energy_upper=upper,
        uniform_all8_all45_selected_receiver_grade2_energy_upper=3*plane_maximum+2*hard_photon+2**-40,
        actual_25momentum_100CAR_graph_onebody_energy_max=maximum,hard_photon_energy_max=hard_photon,
        exact_enumerated_packet_rows=8*3*125*3,exact_selected_receiver_plane_rows=8*3*5*3,
        proof='Actual25-momentum/100-CAR graph for ALL8 statistical sectors,3 electron x offsets, FULL125 incoming offsets and3 outgoing offsets. First/second current and neutral/source endpoints have <=3 particle/hole excitations plus <=2 hard photons. Selected receiver leading attachments use45 on-plane beam triples per sector, with ABSOLUTE B125 amplitudes and no renormalization or PH gate. All125 are retained in this wholepacket energy bound.',
        grading='PEcut boundary alters grade2 only by rare spectator/canonical-field corrections below; no assertion that all source endpoint jets are grade4.')


def macro_preparation(p: Parameters,gamma: np.ndarray,*,partition: dict[str,Any]|None=None,
                      support: dict[str,Any]|None=None,static_toron: dict[str,Any]|None=None,
                      root_parameters: Any=None) -> dict[str,Any]:
    L,Nt,beta=p.spatial_size,p.time_slices,10.**p.beta_power
    volume=L**3;V4=volume*Nt;N=10**(p.beta_power+4*p.kappa_power)//16
    R=L*1e-50;mass=beta*L
    # Poisson theta identity gives <n²>=m/Nt minus a positive dual sum,
    # hence the EXACT upper m/Nt; no floating outward epsilon is lost.
    p2=mass/Nt
    mf=Fraction(10**p.beta_power*L);rf=Fraction(str(R))
    comm2_exact=9*Nt**2*(Fraction(63,8)/(mf**2*rf**4)+3/(mf*Nt*rf**2))
    comm2=float(comm2_exact)
    probability=(128*R/(315*math.pi))**3
    qmin=2*math.sin(math.pi/L)
    soft_coth=1/math.tanh(Nt*qmin/2)
    pkt=finite_photon_packet(p,gamma)
    c=WilsonConsumer(gamma,.01,1.)
    center=np.asarray(pkt['integer_center']);offset=np.array(list(itertools.product(range(-2,3),repeat=3)))
    q=(center+offset)*(2*math.pi/L);weights=np.exp(-np.sum(offset**2,axis=1)/2);weights/=weights.sum()
    energy=np.array([c.photon_energy(x) for x in q]);khat2=np.sum((2*np.sin(q/2))**2,axis=1)
    log_thermal=-Nt*energy/math.log(10)
    # Hard band is cold; compute exponents, never replace the other modes by vacuum.
    Bgram_beta=float(weights@(khat2/(2*np.sinh(energy))))
    Bboundary_beta=float(weights@(khat2*np.exp(-energy)/(2*np.sinh(energy))))
    # BOTH positive-sector boundaries act on the photon.  The extra exp(E/2)
    # here compensates the second, charged halfstep, not just electron energy.
    photon_weights=weights.copy()
    standalone_weights=weights*np.exp(energy);standalone_weights/=standalone_weights.sum()
    carrier=np.array([L/4,0.,0.]);phases=np.exp(-1j*q@carrier)
    selected=(offset[:,1]==0)&(offset[:,2]==0)&(abs(offset[:,0])<=1)
    selected_probability=float(weights[selected].sum())
    selected_weights=weights[selected]/selected_probability
    prefilter=np.sqrt(2*beta*np.sinh(energy))/np.sqrt(khat2)*np.exp(energy)*phases
    electron_offsets=np.arange(-1,2)*2*math.pi/L
    electron_energies=np.array([c.energy(np.array([x,0.,0.])) for x in electron_offsets])
    source_prefilter=np.exp(electron_energies/2)*np.exp(-1j*electron_offsets*L/4)
    statistical_prefilters=[]
    for eta in itertools.product((0,1),repeat=3):
        beam=np.column_stack((electron_offsets,np.zeros(3),np.zeros(3)))+math.pi*np.array(eta)/L
        statistical_prefilters.append(dict(statistical_toron=list(eta),momenta=beam.tolist(),
            factors=[complex(np.exp(c.energy(k)/2-1j*k@carrier)) for k in beam]))
    packet_clip=2e6
    clip_variance=float(weights@(np.exp(2*energy)/np.tanh(Nt*energy/2)))
    clip_tail_log10=math.log10(packet_clip**2+clip_variance)-packet_clip**2/(clip_variance*math.log(10))
    clip_amplitude_log10=math.log10(4*8)-.5*math.log10(.5*selected_probability)+.5*clip_tail_log10
    clocks=[complex(photon_weights@np.exp(-1j*energy*t)) for t in (0,4096,65536,Nt)]
    kmax=max(1/(2*(math.expm1(a*m)+4)) for a in p.spacings for m in p.masses)
    aux_winding=(math.log10(8*V4)+math.log10(N)+Nt*(math.log10(16)-p.kappa_power)-math.log10(Nt))
    aux_thermal=math.log10(4*volume)+math.log10(N)-Nt*(p.kappa_power-math.log10(2))
    active_thermal=math.log10(4*volume)-Nt*min(a*m for a in p.spacings for m in p.masses)/math.log(10)
    spatial_winding=math.log10(8*V4)+L*math.log10(8*kmax)-math.log10(L)-math.log10(1-8*kmax)
    loop_tail=math.log10(8*V4)+math.log10(N)+6*(math.log10(8)-p.kappa_power)-math.log10(6)
    # Full bounded active-Fock + exact photon-sandwich body log derivative,
    # source string <=3L/2. On the retained chart Y=sqrt(beta)A<=2e6.
    field=2e6;Cbody=8*30720.;string_length=3*L/2
    perturb=Nt*(Cbody*volume*field+4*string_length*field)
    source_even_second=(10.**(-p.beta_power))*64*perturb**2
    check(source_even_second<1e-40,'Coexact source Gram correction is small; torons are separate')
    partition=finite_box_photon_partition(p) if partition is None else partition
    support=prepared_energy_support(p,gamma) if support is None else support
    static_toron=toron_static_response(p,gamma) if static_toron is None else static_toron
    if root_parameters is None:
        root_parameters=root_transfer.RootParameters(normal_width_power=2000,tube_radius_power=2100)
    root_input=dict(root=root_parameters.__dict__,photon=p.__dict__)
    root_input_sha256=hashlib.sha256(json.dumps(root_input,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    root_budget=root_transfer.charge_weighted_phi_budget(root_parameters,p)
    root_family=root_transfer.family_bounds(root_parameters,p)
    soft_mean=partition['mean_energy_upper']
    toron_mean=3/(2*Nt)+9/(2*mass*R**2)
    string_energy=root_family['separated_integer_string_induced_energy_upper']
    hard_mean=float(max(electron_energies)+max(energy))
    energy_cut=.25;toron_allocation=.02
    reference_photon_Gram_lower=1-soft_mean-toron_mean-string_energy
    reference_source_Gram_lower=reference_photon_Gram_lower/float(weights@np.exp(energy))
    reference_cut_retention_lower=1-(soft_mean+toron_mean+hard_mean+string_energy)/energy_cut
    harmonic=photon_bounds(p)
    rejected_lift_price=4*64*packet_clip**2*harmonic['normalized_Hermite_transfer_error_budget']
    rejected_lift_lower=reference_source_Gram_lower*reference_cut_retention_lower-rejected_lift_price
    check(reference_photon_Gram_lower>.99 and reference_cut_retention_lower>.9,'Actual finitebox controlled REFERENCE source/energy calculations')
    source_lower=Fraction(1,2)
    photon_lower=Fraction(9,10)
    reserve=energy_cut-support['uniform_all8_full125_grade2_energy_upper']-toron_allocation-1e-6
    partition_base=partition['rows'][0]['logZ_lower']
    chernoff_rows=[]
    for row in partition['rows'][1:]:
        shift=Nt-row['inverse_temperature']
        exponent=row['logZ_upper']-partition_base-shift*reserve
        chernoff_rows.append(dict(fraction=row['fraction'],positive_log_probability_upper=exponent,
            probability_log10_upper=exponent/math.log(10),reserved_soft_energy=reserve,
            actual_tilted_energy_upper=row['mean_energy_upper'],
            first_energy_moment_log10_upper=exponent/math.log(10)+math.log10(row['mean_energy_upper']+hard_mean+toron_mean)))
    photon_energy_tail=min(row['probability_log10_upper'] for row in chernoff_rows)
    check(photon_energy_tail<-1000,'ACTUAL finitebox energy Chernoff resolves the Born scale')
    # f has f=f'=0 at its endpoints, BV(f'')=40/R².  Fourier tail is
    # <=(393.75/pi)(R K)^-5 after its ACTUAL Haar normalization.
    threshold=math.sqrt(2*mass*toron_allocation/3)/2
    filter_tail=math.log10(393.75/math.pi)-5*math.log10(R*threshold)
    warm_tail=math.log10(4)-.9*Nt*toron_allocation/(12*math.log(10))
    toron_tail=math.log10(3)+max(filter_tail,warm_tail)+math.log10(1+10.**(-abs(filter_tail-warm_tail)))
    # Positive band/source/string derivatives through order3 are bounded on
    # the SAME globally small holonomy interval by512/gap,30720 andlength/L.
    # This factor pays their conditional density normalization and Fourier tail.
    source_jet_cap=8*16*(1+1.5+3*512/(.019*L)+100*30720/L)**3
    check(source_jet_cap<2**20,'Third source/band/string derivative cap is derived on actual selected100CAR support')
    toron_source_jet_factor_log10=math.log10(2**40/float(source_lower))
    source_toron_tail=toron_tail+toron_source_jet_factor_log10
    eta=toron_allocation/12
    filter_first=math.log10(6)+filter_tail+math.log10(5*eta/3+3/(2*Nt)+3/(mass*R**2))
    warm_first=math.log10(6)+warm_tail+math.log10(5/Nt+1/Nt+9/(2*mass*R**2))
    toron_first_tail=max(filter_first,warm_first)+math.log10(2)+toron_source_jet_factor_log10
    photon_first_tail=min(row['first_energy_moment_log10_upper'] for row in chernoff_rows)
    root_trace=root_budget['projective_slab_density_trace_error_log10']
    root_step=root_budget['projective_step_colour_log10']
    cp_norm_log10=2*math.log10(8*packet_clip)
    retained_log10=math.log10(probability*float(photon_lower)*float(source_lower))
    # The selected beam weight is a reference receiver component, not an
    # instrument or a physical photon input postselection.
    post_source_trace_log10=math.log10(2)+cp_norm_log10+root_trace-retained_log10
    thermal_purifier_log10=root_budget['prepared_boundary_purifier_error_log10']+math.log10(8*packet_clip)-.5*retained_log10
    boundary_purifier_log10=math.log10(8)+.5*root_step+math.log10(8*packet_clip)-.5*retained_log10
    full_phi_purifier_log10=max(thermal_purifier_log10,boundary_purifier_log10)+math.log10(2)
    spectral_gap=math.exp(-energy_cut)-math.exp(-1.)
    phi_excited_probability_log10=2*(math.log10(4)+root_step-math.log10(spectral_gap))
    check(full_phi_purifier_log10<-190,'Whole normalized postsource Phi purifier controls the Born scale')
    rms_link=math.pi/(2*L)*(R/math.sqrt(11)+Nt*(math.sqrt(p2+3/R**2)+.5)/mass)
    check(rms_link<1e-50,'Global periodic toron observable controls evolved link RMS, not escape probability alone')
    check(comm2<1e-60 and soft_coth>1 and aux_winding<-1e6,'Actual toron/thermal/source fullfamily prices')
    return dict(spatial_size=L,time_slices=Nt,beta_power=p.beta_power,kappa_power=p.kappa_power,auxiliary_count_exact=str(N),
        positive_density='R=Pphi f_physical_torons; rho0=K½ R K^Nt R† K½/actualTrace, fullGauss and all8statistical sectors. Tf=detBplus Gamma(Ldag diag(Bplus^-1,Bminus)L); photon/source positive-sector-halfstep boundaries then EXPLICIT Pe energy preparation. No free-sea replacement.',
        reflected_toron_normalization=dict(
            exact_three_toron_Haar_probability=probability,
            overlap_relative_defect_three_steps_exact=str(Fraction(27,2)/(mf*rf**2)),
            overlap_relative_defect_three_steps_upper=float(Fraction(27,2)/(mf*rf**2)),
            absolute_reflected_moment_bound='E|f_L f_R F|<=cap*p by ordinary Cauchy and equal stationary Haar marginals. The actual free heat overlap is>=p[1-9ell/(2 beta L R²)], ell<=3; this proves reference rarity cancellation before inserted moments are compared.',
            scope='Exact reference free-toron normalization and scalar moment theorem. Actual microscopic kernel/power comparison still must be proved before applying it to macro source admission.'),
        trace_class_and_nonzero='Both conditioned chart density and repaired outside e^-Hout/2 P_U1Haar e^-Hout/2 are traceclass, with fullCartesian kinetic/quartic gap and actual Root Schatten proof. CompactU1 alone would NOT suffice. FixednumberGauss/Haar trial proves nonzero denominator; no globalTmin asserted.',
        actual_auxiliary_elimination='Exact closed thermal Wilson determinant with detB prefactor and fugacity canonical projections; source kernels are actual occupied minors. All virtual auxiliary density/current fluctuations retained.',
        actual_auxiliary_canonical_positive_kernel=auxiliary_canonical_kernel_budget(p),
        actual_active_harmonic_canonical_eigenbundle=active_harmonic_canonical_budget(p),
        physical_canonical_Gaussian_Gauss=canonical_gaussian_Gauss_budget(p),
        positive_compact_photon_kernel=compact_photon_kernel_budget(p),
        canonical_auxiliary_winding_log10=aux_winding,all_auxiliary_thermal_population_upper_log10=aux_thermal,
        active_thermal_population_upper_log10=active_thermal,
        omitted_auxiliary_loop_logdet_upper_log10=loop_tail,
        finite_box_thermal_partition=partition,
        exact_energy_preparation=dict(operator='Pe=1_[exp(-.25)lambda0,lambda0](Kcharged); rho_source=Pe Tch½ S_pre rho_ph S_pre† Tch½ Pe/actualTrace',
            common_neutral_lambda0=True,energy_cut=energy_cut,formal_projected_energy_mean_upper=energy_cut,
            actual_high_energy_expectation=None,microscopic_numerator_nonzero_proved=False,
            preparation_not_ground_or_sharp_pole=True,
            independent_newlaw_not_finiteSCREEN_gate=True,actual_grade2_support=support,
            positive_photon_Chernoff_rows=chernoff_rows,photon_high_energy_probability_log10_upper=photon_energy_tail,
            photon_high_energy_first_moment_log10_upper=photon_first_tail,
            toron_high_energy_probability_log10_upper=source_toron_tail,
            toron_high_energy_first_moment_log10_upper=toron_first_tail,
            reference_source_energy_mean_upper=soft_mean+toron_mean+hard_mean+string_energy,
            reference_energy_cut_retained_probability_lower=reference_cut_retention_lower,
            price_scope='REFERENCE energy/grade2 support only. If the microscopic Pe source numerator is NONZERO, its normalized energy is<=.25 exactly; nonzero/fullGram have NOT been proved by these reference calculations. Selected recoil grade0 vanishes in reference; its cutoff correction is relative sqrt(tail). Clipped multiplier is not asserted bareH2; positiveT½/Pe give a domain only after admission.'),
        toron=dict(coordinate='theta=L A_h, three physicalU1 holonomies',radius=R,link_chart_radius=1e-50,
            kinetic='sum n²/(2 beta L)',thermal_coordinate_density='Haar, before filter',filter='product periodized (1-(theta/R)^2)^2 supported|theta|<R; f rho f /actualTr',
            actual_Haar_postselection_probability=probability,kinetic_momentum_second_upper=p2+3/R**2,
            three_window_purified_unitary_error_squared_upper=comm2,physical_window_escape_probability_upper=comm2,
            unitary_error_squared_upper_exact=str(comm2_exact),
            proof='Thermal purifier U tensor U* cancels stationary rho. ||[H,f]rho^1/2||²/p=63/(8m²R⁴)+3<p²>/(m²R²). Three-window triangle gives9; no small static||Hrho^1/2|| premise.',
            mean_link_shift_bound=1e-50+math.pi/L*comm2,global_periodic_observable_evolved_link_RMS_upper=rms_link,
            RMS_proof='Reference induced quantum toron: Q=exp(itheta)-1, [H,Q]=exp(itheta)(2n+1)/(2m), initial filteredtheta²=R²/11, conservedn. Global principal|theta|<=pi|Q|/2. Full departures are separately paid by source/Root/Photon response; never infer small RMS from escape probability.',
            static_canonical_pole_source_current_response=static_toron,
            spatial_winding_potential_upper_log10=spatial_winding,
            statistical_torons='ALL8 primitivePauli spin structures remain separate and retained'),
        photon=dict(packet_modes=125,physical_polarizations_available=2,prepared_polarization='normalized coexact projection of y at eachq',momenta=q.tolist(),weights=photon_weights.tolist(),
            raw_CurlB_Gram_times_beta=Bgram_beta,positive_range_boundary_Gram_times_beta=Bboundary_beta,
            standalone_hard_leg_boundary_Gram=float(weights@np.exp(energy)),joint_hard_leg_reference_Gram=1,
            reference_soft_spectator_boundary_Gram_lower=reference_photon_Gram_lower,
            full_photon_Gram_lower_exact=None,microscopic_photon_admission_proved=False,
            Gaussian_action_reference=harmonic,
            standalone_precharged_weights=standalone_weights.tolist(),actual_carrier_center=carrier.tolist(),
            Fourier_translation_phases=phases.tolist(),joint_reference_spectator_inverse_temperature=Nt+3,
            explicit_Curl_Fourier_filter=prefilter.tolist(),
            Fourier_filter_convention='Listed creation coefficient of Bdag at positiveq, Curl_-q; B itself is adjoint. Thus preparedket phaseexp(-iqxX), not itsconjugate.',
            selected_consumer3_probability=selected_probability,
            exact_bounded_packet_operator='B125clip=B125 min(1,M/abs(B125)), normal complex gauge-invariant Curl multiplier; M=2e6',
            packet_operator_norm_upper=packet_clip,circular_complex_reference_variance=clip_variance,
            clipped_reference_Hilbert_tail_squared_log10=clip_tail_log10,
            charged_selected3_clip_amplitude_error_log10=clip_amplitude_log10,
            clip_Gram_lower_condition='CONDITIONAL on microscopic gph>0 and sourceGram>0; reference Gaussian/compact action ratio is NOT a positive inserted-Haar/source-Gram theorem',
            selected_consumer3_weights=selected_weights.tolist(),
            minimum_Nt_energy=float(Nt*energy.min()),hardband_thermal_upper_log10=float(log_thermal.max()),
            soft_spectator_coth_upper=soft_coth,nonzero_mode_coordinate_variance_scaled_upper=soft_coth/(2*qmin),
            actual_thermal_Liouville_clock=clocks,clock_times=[0,4096,65536,Nt],
            state='Tn½ B125clip† rho0 B125clip Tn½, followed byTch½ S_pre; B125 exp(Eph) compensates BOTH photon halfsteps andS exp(Ee/2) compensates electron leg. Standalone photonweights differ, finaljoint125targetweights listed. Fouriertranslationapplied: xRef=0 withINITIALRef4/background4 (plus), notoccupationRef0; all other modes thermalNt+3.',
            fullT_relative_kernel_price='Exact determinant nonplaquette logtail above + compact harmonic chart action/weighted Hermite tails from the current Photon source + independently renewed Root field bound'),
        source=dict(charge=1,Ref_charge=1,neutral_Ref_background=4,body_plus_Ref_hole='body+1 at x, Ref-1 at0',
            exact_operator='S_pre=sum_k f_k exp(-ipxL/4)exp(Eelectron(k)/2) a_plus^dag(k;integerSU4/U1Refstrings) r(0); a_plus usescanonicalT1 stablelambda<1, notgamma0plus. CovariantP_x(k)=V(Phi_x)P_el(k)Vdag beforeWilsonpath. Pe Tch½ S_pre rho_ph S_pre† Tch½ Pe/actualretainedGram.',
            source_Gram_not_local_two=True,bare_positive_band_pole_compensated_norm=1,
            reference_soft_spectator_charged_boundary_Gram_lower=reference_source_Gram_lower,
            reference_energy_prepared_source_Gram_lower=reference_source_Gram_lower*reference_cut_retention_lower,
            full_energy_prepared_source_Gram_lower=None,microscopic_source_admission_proved=False,
            explicit_free_boundary_prefilter=source_prefilter.tolist(),
            source_operator_cap=8,
            filledRef4_CAR_chart_operator_norm_squared_upper=math.exp(.011),
            filledRef4_CAR_operator_bound_proof='On the actual transported chart, S(A)=sum_alpha a†(f_alpha(A))r_alpha on filledRef4. SdagS=sum_alpha a(f_alpha)a†(f_alpha)<=Tr f†f. At each site the exact integer Wilson/U1 and transported SU4 matrices are left/right unitaries independent of beam k. Their removal preserves Frobenius norm of the WHOLE coherent band sum. Fourier Parseval gives Tr f†f=sum_k|a_k|² exp(Ee(k))*Tr(P_el(k)^2)/2<=exp(.011), with literal source rank2 normalization and beam l2 norm1. For all8statistical p, |p_j|<1e-6 and the actual positive Wilson dispersion gives Ee<.011. This proves a norm CAP only, never Gram1. The separate global outside source cap8 remains in force.',
            coexact_even_grade2_Gram_correction_upper=source_even_second,
            independent_static_toron_Gram_correction_upper=static_toron['retained_Gram_static_correction_upper'],
            normalized_Gram_lower_exact=None,
            all8statistical_electron_boundary_prefilters=statistical_prefilters,
            bound_conditions='REFERENCE positive Wilson/induced-Gaussian source minors and spectator partition ratio, NOT Gram1. Microscopic fullRoot positiveT/Haar inserted sourceGram remains unproved. CanonicalNF*kappa4 chargevariance does NOT imply its Fourier/minor local-limit ratio. Coexact even source normalization and static toron/source endpoint corrections are separate. No invented interacting-Slater Wick claim.',
            connected_price='Grade2 Gram normalization changesgrade2amplitude atgrade4, BUT neutral/vacuum/source/receiver halfstep endpoint jets canchangegrade2 itself. Photon prepared_response executes them withthedeclaredcanonicalpositiveband source; no quietendpoint→grade4 claim.'),
        full_SU4_normal_Root=dict(source_sha256=hashlib.sha256(SNAPSHOT['covariant_root_transfer']).hexdigest(),
            root_input_sha256=root_input_sha256,root_parameters=root_parameters.__dict__,
            normal_width_parameter=f'1e-{root_parameters.normal_width_power}',
            tube_radius=f'1e-{root_parameters.tube_radius_power}',normal45V=True,colour45V=True,
            radial_ground='Root exactcompact k, not Gaussian width equality when R<sigma; curvature/gap bound survives and support is exact',
            explicit_preparation_projector='Pphi=tensor |normalized continuous radial k><k| for45normal/site and15colour/link, covariantly transported overSU4orbit; tensorIDENTITY on ALL centre/flat/nonflatstat sectors. Never a unique flatcentre projector.',
            exact_prepared_density='R=Pphi f_three_physical_torons; rho0=K½ R K^Nt R† K½/actualTrace, all8 statistical sectors retained; photon/source have subsequent positive-sector-halfstep boundaries',
            actual_species_resolved_projective_budget=root_budget,
            conditional_example_normalized_postsource_trace_error_log10=post_source_trace_log10,
            conditional_example_normalized_postsource_purifier_error_log10=full_phi_purifier_log10,
            conditional_example_boundary_halfstep_HS_price_log10=boundary_purifier_log10,
            CPfilter_norm_upper=8*packet_clip,
            conditional_example_Grams=dict(photon=str(photon_lower),source=str(source_lower),toron=probability),
            actual_retained_Gram_lower=None,
            whole_energy_projector_normal_excited_probability_log10_upper=phi_excited_probability_log10,
            whole_projector_proof='P=Pe, A=Qphi K0 Qphi<=exp(-1)lambda0, B=Pe Knew Pe>=exp(-.25)lambda0. X=Qphi Pe obeys AX-XB=-Qphi deltaK Pe, so ||X||<=||deltaK||/[lambda0(exp(-.25)-exp(-1))] with NO sqrt(rank). Any normalized Pe density and every real-time clock retain this bound. Same all8stat/defect reference; colourgap1 and normalgaplog401.8.',
            outside_quartic_gap_log10=root_family['outside_quartic_gap_log10'],
            outside_kinetic_coefficient=root_family['outside_Cartesian_kinetic_coefficient'],
            outside_normalized_weight_exponents=dict(negative=root_family['outside_thermal_negative_exponent_log10'],
                positive_trial=root_family['outside_thermal_chart_trial_positive_exponent_log10'],
                primitive_entropy=root_family['outside_full_primitive_log_dimension_log10']),
            source_receiver_current_weighted_budgets={str(n):root_transfer.charge_weighted_phi_budget(root_parameters,p,electromagnetic_order=n) for n in (1,2)},
            normalization_proof='Actual projective coefficient sum, not operator supremum rebranded. Root supplies all-p Schatten, positive radialheat+primitiveHaar powers and BKS/Holder boundaryXi. Pphi ground thermal retention >=prod(1-exp(-(Nt+1)Omega))^90V; matterpartition cancels, then bounded toron/photonclip/source CP insertion pays ACTUAL Grams and honest square-root. No globalTmin or exp-closeH Loewner claim.',
            actual_scalar_angular_heat_operator_norm_upper=root_budget['actual_scalar_angular_heat_operator_norm_upper'],
            unprepared_free_sea_admitted=False),
        admission=dict(microscopic_source_Gram_proved=False,microscopic_photon_Gram_proved=False,
            microscopic_energy_band_nonzero_proved=False,macro_source_admitted=False,
            valid_scope='Actual signed native integerU1-Gauss quantum sources; finitebox reference thermal/source/energy calculations; Root weighted/projective/BKS theorem CONDITIONAL on actual retained Grams',
            rejected_Gaussian_action_lift_control=dict(would_be_source_Gram_lower=rejected_lift_lower,
                invalid_step='Scalar EuclideanGaussian/action ratio is NOT positive microscopic inserted-Haar/source norm comparison; actual bareHaar null/commonrange survives'),
            missing_theorem='Same positiveK canonical fullGauss wholepacket cross-source minor/denominator after photon and charged halfsteps, with ALL species, centres,statisticaltorons and integerdipole. Neither co-local norm1, unprojectedNF*kappa4 covariance nor scalarGaussian action bound supplies this.',
            exact_boundary_moment_construction=dict(
                neutral_purifier='Xi0=Kneutral½ Pphi f Kneutral^(Nt/2); Z0=||Xi0||HS²',
                photon_purifier='Xph=Kneutral½ B125clipdag Xi0; Zph=||Xph||HS²; BOTH Fourier orientations and all coherent125 modes stay inside this operator',
                charged_boundary_factor='Kcharged=Vcharged Vchargeddag, Vcharged=Tfcharged½ P_Haar (including declared Root electric/normal factors); each species matrix element is sqrt(detBplus) det[(sqrt T1)_(I,J)] at its ACTUAL fixed canonical occupation; joint Haar integrates product of ALL species minors, not product of separately projected Grams',
                M0='||Sraw Xph||HS²/Zph',
                M1='||Vchargeddag Sraw Xph||HS²/(lambda0 Zph)',
                range_M0='||P_nonzero(Kcharged) Sraw Xph||HS²/Zph; cannot identify this with M0 when PHaar has null range',
                exact_sufficient_PE='M1-exp(-.25)*M0>0, or sharper range_M0 version after exact K-null subtraction; remove any charged spectrum above lambda0 explicitly',
                reflected_absolute_error='If actual absolute HS distances are eps0,eps1 from reference w0,V0dag w0/sqrt(lambda0), then M0<=(||w0||+eps0)^2 and M1>=max(0,||V0dag w0/sqrt(lambda0)||-eps1)^2. Scalar action RN prices supply NEITHER distance.',
                toron_normalization='An unconditioned HS error eps requires division by sqrt(actual Zph), including rare physical-toron filtering. Rarity cancels ONLY if a reflected conditional kernel proves a relative bound uniform on that filter; no such theorem is presently available.',
                actual_macro_M0=None,actual_macro_M1=None,actual_macro_PEGram=None,
                actual_macro_photon_normalization=None,fullGauss_kernel_estimate_proved=False,
                all_centres_statistics_species_and_integer_dipole_required=True),
            finiteSCREEN_source_unchanged=True))


def portable(value):
    if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    if isinstance(value,complex):return [value.real,value.imag]
    if isinstance(value,np.ndarray):return portable(value.tolist())
    if isinstance(value,np.generic):return value.item()
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid',type=int,default=64);parser.add_argument('--cutoff',type=int,default=3)
    parser.add_argument('--steps',type=int,default=12);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,default=ROOT/'covariant_prepared_gauss_state_input_kappa400_sigma2000_r2100.json')
    args=parser.parse_args();start=time.monotonic()
    input_bytes=args.input.read_bytes();configuration=json.loads(input_bytes)
    p=Parameters(**configuration['photon_parameters'])
    root_parameters=root_transfer.RootParameters(**configuration['root_parameters'])
    gamma,_=phi_frame(p)
    native=native_preparation(gamma,args.grid,args.cutoff,args.steps)
    refined=native_preparation(gamma,2*args.grid,args.cutoff+2,args.steps)
    minus_native=native_preparation(gamma,args.grid,args.cutoff,args.steps,-1)
    toron=native_toron(48);toron_refined=native_toron(96)
    normal_native=native_normal_preparation(gamma,args.steps)
    plus_runtime=native_normal_preparation(gamma,args.steps,11,True)
    minus_runtime=native_normal_preparation(gamma,args.steps,11,True,body_charge=-1)
    normal_refined=plus_runtime['summary'];minus_normal=minus_runtime['summary']
    plus_energy=energy_cut_source(plus_runtime['runtime'],6.)
    minus_energy=energy_cut_source(minus_runtime['runtime'],6.)
    rayleigh=[positive_boundary_rayleigh(r['runtime'],cut)
        for r in (plus_runtime,minus_runtime) for cut in (6.,.25)]
    static_toron=toron_static_response(p,gamma)
    partition=finite_box_photon_partition(p)
    support=prepared_energy_support(p,gamma)
    macro=macro_preparation(p,gamma,partition=partition,support=support,static_toron=static_toron,
        root_parameters=root_parameters)
    root_native_bridge=root_transfer.projective_heat_native(gamma)
    root_probe=ROOT/'covariant_root_transfer_aux_density_probe.json';root_probe_bytes=root_probe.read_bytes()
    unchanged=all((ROOT/(name+'.py')).read_bytes()==SNAPSHOT[name] for name in COMPILED)
    check(unchanged,'Compiled source changed during scientific execution; no source retag')
    result=dict(plan=PLAN,governing_revision=PLAN_REVISION,assignment_revision=EPOCH,
        inputs=dict(parameters=p.__dict__,argv=sys.argv,native_steps=args.steps,
            native_grid=args.grid,native_cutoff=args.cutoff,normal_nodes=[7,11],
            native_energy_cut=6.,macro_reference_energy_cut=.25,gamma=gamma),
        input_configuration=dict(path=str(args.input),sha256=hashlib.sha256(input_bytes).hexdigest(),
            captured_bytes=input_bytes.decode(),configuration=configuration),
        branch='NEW positiveT signed native quantum Gauss state; macro source admission remains OPEN',
        native=native,refined=refined,native_refinement=dict(photon_Gram_relative=abs(refined['photon_retained_Gram']/native['photon_retained_Gram']-1),source_Gram_relative=abs(refined['source_retained_Gram']/native['source_retained_Gram']-1)),
        toron_native=toron,toron_refined=toron_refined,macro=macro,
        normal_native=normal_native,normal_refined=normal_refined,
        minus_native=minus_native,minus_normal=minus_normal,
        native_energy_prepared_plus=plus_energy,native_energy_prepared_minus=minus_energy,
        microscopic_signed_wholepacket_Rayleigh=rayleigh,
        conditional_canonical_reflected_boundary=canonical_reflected_boundary_control(gamma),
        auxiliary_static_positive_boundary=auxiliary_static_positive_boundary(gamma),
        full_spin_auxiliary_boundary=full_spin_auxiliary_boundary(gamma),
        active_harmonic_boundary=active_harmonic_boundary_control(gamma),
        canonical_Gaussian_Gauss=canonical_gaussian_Gauss_control(),
        actual_positive_scalar_kernel=positive_scalar_kernel_control(),
        actual_Root_projective_and_boundary_native=root_native_bridge,
        static_toron=static_toron,
        canonical_aux_density_input=dict(path=str(root_probe),sha256=hashlib.sha256(root_probe_bytes).hexdigest(),rows=json.loads(root_probe_bytes)['rows'],exponents=json.loads(root_probe_bytes)['actual_small_kappa_exponents'],free_product_Gauss_admitted=False,multiplicity_weighted_variance_asymptotic='96 Naux kappa^4=6 beta, NOT small'),
        compiled_source_hashes=COMPILED,compiled_source_unchanged=unchanged,
        loaded_main_code_sha256=hashlib.sha256(marshal.dumps(LOADED_MAIN_CODE)).hexdigest(),main_code_matches_captured_source=MAIN_CODE_MATCH,
        captured_source_bytes={name:SNAPSHOT[name].decode('utf-8') for name in COMPILED},
        runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__,platform=platform.platform()),elapsed_seconds=time.monotonic()-start)
    args.output.write_text(json.dumps(portable(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),native_photon_Gram=refined['photon_retained_Gram'],native_source_Gram=refined['source_retained_Gram'],native_signed_energy_cut_Grams=[plus_energy['summary']['total_source_retained_Gram'],minus_energy['summary']['total_source_retained_Gram']],integer_Gauss_defect=refined['source_Gauss_defect'],macro_source_admitted=macro['admission']['macro_source_admitted'],macro_toron_clock_squared=macro['toron']['three_window_purified_unitary_error_squared_upper'],elapsed_seconds=result['elapsed_seconds'])))
    print('Natural exit: actual signed native positiveT/range/integerGauss/energy source; macro inserted-Gram admission UNPROVED')


if __name__=='__main__':main()

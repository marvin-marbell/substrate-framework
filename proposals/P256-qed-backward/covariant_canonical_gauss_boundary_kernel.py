#!/usr/bin/env python3
"""Boundary-resolved canonical/Gauss successor; frozen Source2a0 stays unchanged.

All species have their own fixed-number fugacity coefficient BEFORE identical
auxiliary coefficients are raised to their actual multiplicity. Character
terms need not be positive. The native positive kernel is assembled from the
joint Haar projector, not a product of separately projected species kernels.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import marshal
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence, cast

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
LOADED_MAIN_CODE=sys._getframe().f_code
MAIN_CODE_MATCH=LOADED_MAIN_CODE==compile(OWN_BYTES,LOADED_MAIN_CODE.co_filename,'exec',dont_inherit=True,optimize=sys.flags.optimize)
if not MAIN_CODE_MATCH:raise ArithmeticError('Loaded boundary-kernel code differs from captured source')
import numpy as np
from scipy.linalg import block_diag, eigh, expm
from scipy.special import roots_legendre
import mpmath as mp
import covariant_prepared_gauss_state as frozen
import covariant_root_transfer as physical_root
from covariant_induced_photon import Parameters, phi_frame

PLAN='issue-222-induced-photon-delta13-continuation-20261008'
REVISION='84912954a78dbe9771de62d6a75cfc83496ddbfa0ee7d6e77fa525fb6663efe8'
FROZEN_SOURCE='2a0ecfb2341fceb7866d5062cf33d1da71b2fafa4b148829d21d14f0650fe307'
ROOT_SOURCE='6911f0283dc0f84f4cc05202deb67dcd3266d5c64ca1a05e149f1567451e0f9c'


def require(ok: bool,reason: str) -> None:
    if not ok: raise ArithmeticError(reason)


def occupations(modes: int,number: int) -> list[tuple[int,...]]:
    return list(itertools.combinations(range(modes),number))


def exterior(matrix: np.ndarray,states: Sequence[tuple[int,...]]) -> np.ndarray:
    indices=np.asarray(states)
    return np.linalg.det(matrix[indices[:,None,:,None],indices[None,:,None,:]])


def coefficients(matrix: np.ndarray,degree: int) -> list[complex]:
    """Literal principal-minor coefficients; no spectral inverse or grand trace."""
    return [1.+0j]+[sum((complex(np.linalg.det(matrix[np.ix_(state,state)]))
        for state in occupations(len(matrix),n)),0j) for n in range(1,degree+1)]


def fixed_trace(matrix: np.ndarray,number: int) -> complex:
    return coefficients(matrix,number)[number]


def inserted_minor(left: np.ndarray,middle: np.ndarray,right: np.ndarray,
                   number: int,u: np.ndarray,v: np.ndarray) -> np.ndarray:
    """Tr_N Gamma(L) a(u) Gamma_(N+1)(M) a†(v) Gamma(R)."""
    cycle=middle@right@left
    polynomial=coefficients(cycle,number)
    adjugate=np.eye(len(cycle),dtype=complex)
    for n in range(1,number+1):
        adjugate=polynomial[n]*np.eye(len(cycle))-cycle@adjugate
    return u.conj().T@adjugate@middle@v


def ordered_product(factors: Sequence[np.ndarray]) -> np.ndarray:
    require(bool(factors),'A boundary path must contain actual operators')
    result=factors[0]
    for factor in factors[1:]: result=result@factor
    return result


@dataclass(frozen=True)
class Species:
    canonical_number: int
    copies: int
    left_halfsteps: tuple[np.ndarray,...]
    right_halfsteps: tuple[np.ndarray,...]
    left_scalar_prefactors: tuple[float,...]
    right_scalar_prefactors: tuple[float,...]


def canonical_path_weight(species: Sequence[Species],gauges: Sequence[np.ndarray],
                          background_character: complex=1.+0j) -> complex:
    """Executable time-ordered coefficient integrand at one full field history.

    Every temporal K has TWO microscopic SF factors: onebody sqrt(T4) and
    scalar sqrt(detBplus) on EACH side of its shared Gauss/Haar insertion.
    The normal heat is integrated between the physical endpoint histories.
    Gauges are actual joint local-Gauss/centre matrices at EACH boundary. The
    caller integrates this ONE product; individual character terms are signed.
    Copies are separate canonical flavours, never one total-number coefficient.
    """
    value=background_character
    for item in species:
        require(len(item.left_halfsteps)==len(item.right_halfsteps)==len(gauges)
            ==len(item.left_scalar_prefactors)==len(item.right_scalar_prefactors),
            'Every WHOLE temporal K retains both microscopic halfsteps, shared gauge and half vacuum prefactors')
        factors=[left@gauge@right for left,right,gauge in zip(item.left_halfsteps,item.right_halfsteps,gauges)]
        canonical=fixed_trace(ordered_product(factors),item.canonical_number)
        value*= (math.prod(item.left_scalar_prefactors)*math.prod(item.right_scalar_prefactors)*canonical)**item.copies
    return value


def physical_spatial_onebody(gamma: np.ndarray,kappa: float,links: Sequence[np.ndarray],
                            remainder: Sequence[np.ndarray],field_frame: np.ndarray) -> tuple[np.ndarray,float,np.ndarray]:
    """Actual physical Root b/c on 4V components; no spin×colour doubling.

    The three links and four block-local normal remainders are in the SAME
    Dminus pullback frame. The physical field frame Q is explicit and acts on
    the entire primitive, not only the source or the gamma matrices.
    """
    modes=len(links[0]);sites=modes//4
    component=physical_root.wilson_component_basis(gamma)
    plus=np.kron(np.eye(sites),component[:,:2]);minus=np.kron(np.eye(sites),component[:,2:])
    theta=np.column_stack((plus,minus))
    gamma_space=[np.kron(np.eye(sites),g) for g in gamma[1:]]
    cost=np.zeros(modes)
    for site in range(sites):
        price=sum(float(np.linalg.norm(y[4*site:4*site+4,4*site:4*site+4],'fro')**2) for y in remainder)
        cost[4*site:4*site+4]=price
    b=np.eye(modes)/(2*kappa)-sum((u+u.conj().T for u in links),np.zeros((modes,modes),complex))/2
    b+=kappa*remainder[0]+kappa*np.diag(cost)
    bp=plus.conj().T@b@plus;bm=minus.conj().T@b@minus
    c=sum((g@u-u.conj().T@g for g,u in zip(gamma_space,links)),np.zeros((modes,modes),complex))/2
    c+=kappa*remainder[0]+kappa*sum((y@u-u.conj().T@y for y,u in zip(remainder[1:],links)),np.zeros((modes,modes),complex))/2
    f=plus.conj().T@c@minus
    triangular=np.block([[np.eye(2*sites),-f],[np.zeros((2*sites,2*sites)),np.eye(2*sites)]])
    transfer=field_frame@theta@triangular.conj().T@cast(np.ndarray,block_diag(np.linalg.inv(bp),bm))@triangular@theta.conj().T@field_frame.conj().T
    values,vectors=eigh(transfer)
    require(bool(values[0]>0),'Actual physical 4V primitive must be positive')
    half=(vectors*np.sqrt(values))@vectors.conj().T
    return half,float(np.linalg.det(bp).real),transfer


def full_component_canonical_control(gamma: np.ndarray) -> dict[str,Any]:
    """Literal 4-component Root match and 70/56 canonical source coefficients."""
    rng=np.random.default_rng(222)
    def hermitian(size: int,scale: float) -> np.ndarray:
        matrix=rng.normal(size=(size,size))+1j*rng.normal(size=(size,size))
        return scale*(matrix+matrix.conj().T)/2
    momentum=np.array([.13,-.21,.34]);colour=[expm(1j*hermitian(4,.04)) for _ in range(3)]
    normal=[hermitian(4,.001) for _ in range(4)]
    actual,_,_=physical_spatial_onebody(gamma,.03,[np.exp(1j*p)*u for p,u in zip(momentum,colour)],normal,np.eye(4))
    supplied,_,_=physical_root.phi_wilson_step(gamma,.03,momentum=tuple(float(p) for p in momentum),
        remainder=np.asarray(normal),colour_links=tuple(colour))
    supplied_half_values,supplied_half_frame=eigh(supplied)
    supplier_half=(supplied_half_frame*np.sqrt(supplied_half_values))@supplied_half_frame.conj().T
    error=float(np.linalg.norm(actual-supplier_half))
    require(error<1e-12,'Generic physical 4V primitive is exactly Root6911, including noncommuting Phi/colour')
    link=np.block([[np.zeros((4,4)),np.exp(.31j)*np.eye(4)],[np.eye(4),np.zeros((4,4))]])
    rem=[cast(np.ndarray,block_diag(.007*g,-.009*g)) for g in gamma]
    h0,prefactor,T0=physical_spatial_onebody(gamma,.03,[link,np.eye(8),np.eye(8)],rem,np.eye(8))
    Q=cast(np.ndarray,block_diag(np.eye(4),expm(1j*hermitian(4,.3))))
    hQ,_,_=physical_spatial_onebody(gamma,.03,[link,np.eye(8),np.eye(8)],rem,Q)
    neutral=occupations(8,4);charged=occupations(8,5)
    Gn=exterior(Q,neutral);Gc=exterior(Q,charged)
    covariance=float(np.linalg.norm(exterior(hQ,neutral)-Gn@exterior(h0,neutral)@Gn.conj().T))
    require(covariance<1e-9,'One joined physical frame transports ALL70 occupations, not a separate spin/colour representation')
    values,vectors=eigh(T0);stable=vectors[:,values<1]
    one=stable@stable.conj().T@np.kron(np.array([1.,1.])[:,None]/math.sqrt(2),np.eye(4))
    one/=np.linalg.norm(one)
    creators=creation(one,neutral,charged);transported=creation(Q@one,neutral,charged)
    source_covariance=max(float(np.linalg.norm(b-Gc@a@Gn.conj().T)) for a,b in zip(creators,transported))
    require(source_covariance<1e-11,'Body creator and Ref4->Ref3 string transform consistently with the same frame')
    g0=np.diag(np.exp(1j*np.r_[np.ones(4)*.17,np.zeros(4)]))
    g1=np.diag(np.exp(1j*np.r_[np.ones(4)*-.23,np.zeros(4)]))
    g2=np.diag(np.exp(1j*np.r_[np.ones(4)*.39,np.zeros(4)]))
    L=h0@g0@h0;M=h0@g1@h0;R=h0@g2@h0
    EL=exterior(L,neutral);ER=exterior(R,neutral);EM=exterior(M,charged)
    direct=np.array([[np.trace(EL@a.conj().T@EM@b@ER) for b in creators] for a in creators])
    signed=one*((-1.)**np.arange(4))[None,:]
    coefficient=inserted_minor(L,M,R,4,signed,signed)
    minor_error=float(np.linalg.norm(direct-coefficient)/max(1.,np.linalg.norm(direct)))
    require(minor_error<1e-11,'Actual70/56 full-component canonical source/Ref cross matrix equals its adjugate coefficient')
    return dict(full_component_modes=8,neutral_canonical_components=70,charged_canonical_components=56,Ref4_holes=4,
        actual_noncommuting_Root6911_halfstep_error=error,actual_full_canonical_frame_covariance_error=covariance,
        actual_full_source_Ref_frame_covariance_error=source_covariance,
        actual_full70_56_source_minor_relative_error=minor_error,
        full_source_cross_Fock=direct,full_source_cross_adjugate=coefficient,
        actual_detB_vacuum_prefactor=prefactor,
        scope='Physical 4V primitive is executable for any finite supplied link/normal history. Full macro histories are the analytic loop integrand, not invented dense arrays or a macro evolved-state backend.')


def frozen_time_step_control(gamma: np.ndarray) -> dict[str,Any]:
    """Actual frozen primitive versus Gamma(T4) and the wrong Gamma(T4)^2."""
    T,bp,_=physical_root.phi_wilson_step(gamma,.03,(.3,.2,.1),np.zeros((4,4,4),complex),(np.eye(4),)*3)
    determinant=float(np.linalg.det(bp).real);values,vectors=eigh(T)
    logT=(vectors*np.log(values))@vectors.conj().T
    model=physical_root.BosonicEdgeModel(4,((0,1),(0,2),(0,3)))
    H=-math.log(determinant)*model.identity-physical_root.native_lift(model,logT)
    SF=expm(-H/2)
    number=sum((model.n(c) for c in range(4)),np.zeros_like(model.identity))
    selected=np.flatnonzero(abs(np.diag(number)-2)<1e-10)
    actual=SF[np.ix_(selected,selected)]@SF[np.ix_(selected,selected)]
    canonical=occupations(4,2);one=determinant*exterior(T,canonical)
    two=determinant*exterior(T@T,canonical)
    spectrum=eigh(actual,eigvals_only=True)
    correct=float(np.linalg.norm(spectrum-eigh(one,eigvals_only=True))/np.linalg.norm(spectrum))
    wrong=float(np.linalg.norm(spectrum-eigh(two,eigvals_only=True))/np.linalg.norm(spectrum))
    require(correct<1e-11 and wrong>1.,'Actual Root native primitive completes ONE T4 step, not T4 squared')
    reduced=frozen.two_spin_frame(gamma)
    h,pref,h_one,_,_=frozen.square_wilson(reduced,.07,.3)
    full=expm(-h_one);source_rows=[]
    for number in (4,5):
        states=occupations(8,number);native_SF=math.sqrt(pref)*exterior(h,states)
        native_K=native_SF@native_SF
        target=pref*exterior(full,states);doubled=pref*exterior(full@full,states)
        error=float(np.linalg.norm(native_K-target)/np.linalg.norm(native_K))
        wrong_error=float(np.linalg.norm(native_K-doubled)/np.linalg.norm(native_K))
        require(error<1e-11 and wrong_error>1.,'Frozen Source neutral AND charged sectors use the SAME one-step convention')
        source_rows.append(dict(canonical_number=number,canonical_dimension=len(states),
            actual_native_complete_transfer_Gamma_T_relative_error=error,
            wrong_Gamma_T_squared_relative_error=wrong_error,
            wrong_full_detB_per_microscopic_half_relative_error=abs(pref-1)))
    half=(vectors*np.sqrt(values))@vectors.conj().T
    item=Species(2,3,(half,half),(half,half),(math.sqrt(determinant),)*2,(math.sqrt(determinant),)*2)
    closed=canonical_path_weight((item,),(np.eye(4),)*2)
    expected=complex(np.trace(one@one))**3
    coefficient_error=float(abs(closed-expected)/abs(expected))
    require(coefficient_error<1e-11,'Executed coefficient integrand preserves whole-step clock, BOTH halves and full auxiliary power')
    return dict(frozen_Source_hash=FROZEN_SOURCE,frozen_Root_hash=ROOT_SOURCE,
        convention='Microscopic SF=sqrt(detBplus) Gamma_N(sqrt(T4)); K=SF P_Haar SF with actual normal-heat sandwiches. With BOTH Haar and normal heat trivial, K=detBplus Gamma_N(T4). Nt counts complete K steps, not microscopic halves.',
        Root_actual_lifted_native_Gamma_T_spectrum_relative_error=correct,
        Root_wrong_Gamma_T_squared_spectrum_relative_error=wrong,
        Source_actual_neutral_and_charged_complete_transfer=source_rows,
        actual_canonical_integrand_two_steps_three_auxiliary_flavours_relative_error=coefficient_error,
        charged_boundary_distinction='The actual charged K_C^1/2 is the square root of the COMPLETE field/Haar/Gauss transfer. It is NOT the microscopic SF; M1 uses Sdag K_C S and retains BOTH SFs and normal heat.')




def wilson_spatial(gamma: np.ndarray,kappa: float,theta: float,y: float,
                   common_toron: float=0.) -> tuple[np.ndarray,float,np.ndarray]:
    """Actual Root b/c prescription restricted to the invariant two-spin plane.

    Two cells, one periodic two-edge hopping direction; other spatial links are
    identities. The normal perturbation has the actual EXTRA kappa factor.
    This is a declared spatial native restriction, not an eight-spin macro law.
    """
    temporal=np.kron(np.eye(2),gamma[0])
    eig,frame=eigh(temporal);plus=frame[:,eig>0];minus=frame[:,eig<0]
    basis=np.column_stack((plus,minus))
    phase=np.exp(1j*theta)
    link=np.exp(1j*common_toron)*np.block([[np.zeros((2,2)),phase*np.eye(2)],[np.eye(2),np.zeros((2,2))]])
    links=(link,np.eye(4),np.eye(4))
    spatial=(np.kron(np.eye(2),gamma[1]),np.kron(np.eye(2),gamma[2]),np.zeros((4,4)))
    remainder=(y*temporal,y*spatial[0],y*spatial[1],np.zeros((4,4)))
    b=np.eye(4)/(2*kappa)-sum((u+u.conj().T for u in links),np.zeros((4,4),complex))/2
    b+=kappa*remainder[0]+kappa*sum(float(np.linalg.norm(a,'fro')**2) for a in remainder)*np.eye(4)
    c=sum((g@u-u.conj().T@g for g,u in zip(spatial,links)),np.zeros((4,4),complex))/2
    c+=kappa*remainder[0]+kappa*sum((r@u-u.conj().T@r for r,u in zip(remainder[1:],links)),np.zeros((4,4),complex))/2
    bp=plus.conj().T@b@plus;bm=minus.conj().T@b@minus;f=plus.conj().T@c@minus
    triangular=np.block([[np.eye(2),-f],[np.zeros((2,2)),np.eye(2)]])
    block=triangular.conj().T@cast(np.ndarray,block_diag(np.linalg.inv(bp),bm))@triangular
    transfer=basis@block@basis.conj().T
    values,vectors=eigh(transfer)
    require(bool(values[0]>0),'Actual positive Wilson primitive')
    half=(vectors*np.sqrt(values))@vectors.conj().T
    return half,float(np.linalg.det(bp).real),transfer


def joint_basis(active: Sequence[tuple[int,...]],auxiliary: Sequence[tuple[int,...]],
                copies: int=1) -> list[tuple[int,...]]:
    return list(itertools.product(range(len(active)),*([range(len(auxiliary))]*copies)))


def local_number(state: tuple[int,...]) -> int:
    return sum(mode<2 for mode in state)


def joint_projector(active,auxiliary,copies: int,target: int) -> tuple[np.ndarray,list[tuple[int,...]]]:
    basis=joint_basis(active,auxiliary,copies)
    selected=[local_number(active[row[0]])+sum(local_number(auxiliary[j]) for j in row[1:])==target for row in basis]
    return np.diag(np.asarray(selected,float)),basis


def creation(coefficients: np.ndarray,neutral,charged) -> list[np.ndarray]:
    index={state:i for i,state in enumerate(charged)};result=[]
    for ref in range(coefficients.shape[1]):
        matrix=np.zeros((len(charged),len(neutral)),complex)
        for col,state in enumerate(neutral):
            for mode in range(len(coefficients)):
                if mode in state:continue
                row=index[tuple(sorted((*state,mode)))]
                matrix[row,col]+=(-1)**(sum(j<mode for j in state)+ref)*coefficients[mode,ref]
        result.append(matrix)
    return result


def gauss_character_control(gamma: np.ndarray) -> dict[str,Any]:
    """Exact coefficient products vs full Fock, including auxiliary powers."""
    kactive=1/(2*(math.exp(.06)+3));kaux=.04
    ha,pa,_=wilson_spatial(gamma,kactive,.31,.007)
    hb,pb,_=wilson_spatial(gamma,kaux,-.22,-.009)
    h2a,p2a,_=wilson_spatial(gamma,kactive,-.13,-.011)
    h2b,p2b,_=wilson_spatial(gamma,kaux,.41,.012)
    neutral=occupations(4,2);charged=occupations(4,3);auxiliary=occupations(4,2)
    rows=[]
    for copies in (1,3):
        target=1+copies;quadrature=2*(1+copies)+1
        projector,basis=joint_projector(neutral,auxiliary,copies,target)
        Fa=exterior(ha,neutral);Fb=exterior(hb,auxiliary)
        Ma=exterior(h2a@h2a,neutral);Mb=exterior(h2b@h2b,auxiliary)
        full=Fa;middle=Ma
        for _ in range(copies): full=np.kron(full,Fb);middle=np.kron(middle,Mb)
        direct=complex(np.trace(full@projector@middle@projector@full))
        value=0j;terms=[]
        for a,b in itertools.product(range(quadrature),repeat=2):
            angles=2*math.pi*np.array([a,b])/quadrature
            g0=np.diag(np.exp(1j*angles[0]*np.array([1,1,0,0])))
            g1=np.diag(np.exp(1j*angles[1]*np.array([1,1,0,0])))
            ca=fixed_trace(ha@g0@h2a@h2a@g1@ha,2)
            cb=fixed_trace(hb@g0@h2b@h2b@g1@hb,2)
            term=np.exp(-1j*target*angles.sum())*ca*cb**copies
            terms.append(term);value+=term/(quadrature**2)
        error=abs(value-direct)/max(1.,abs(direct))
        require(error<2e-11,'Joint local Haar and separate canonical auxiliary coefficients agree with exact Fock')
        rows.append(dict(auxiliary_copies=copies,actual_joint_occupations=len(basis),joint_Haar_rank=int(np.trace(projector)),
            phase_quadrature=quadrature,actual_Fock_reflected_trace=direct,coefficient_character_integral=value,
            relative_error=error,individual_character_real_min=float(min(z.real for z in terms)),
            individual_character_imaginary_max=float(max(abs(z.imag) for z in terms))))
    pn,_=joint_projector(neutral,auxiliary,1,2);pc,_=joint_projector(charged,auxiliary,1,3)
    one=np.array([[.4,0.,.1,0.],[0.,.4,0.,.1],[.5,0.,.2,0.],[0.,.5,0.,.2]],complex)
    one/=np.linalg.norm(one)
    signed_one=one*((-1.)**np.arange(one.shape[1]))[None,:]
    creators=creation(one,neutral,charged)
    Sn=[np.kron(a,np.eye(6)) for a in creators]
    Fna=np.kron(exterior(ha,neutral),exterior(hb,auxiliary))
    Fca=np.kron(exterior(h2a,charged),exterior(h2b,auxiliary))
    An=Fna@pn@Fna;Kc=Fca@pc@Fca
    direct_source=np.array([[np.trace(An@a.conj().T@Kc@b@An) for b in Sn] for a in Sn])
    total=np.zeros_like(direct_source);terms=[];quadrature=6
    for a,b,c in itertools.product(range(quadrature),repeat=3):
        angles=2*math.pi*np.array([a,b,c])/quadrature
        gauges=[np.diag(np.exp(1j*x*np.array([1,1,0,0]))) for x in angles]
        L=ha@gauges[0]@ha;M=h2a@gauges[1]@h2a;R=ha@gauges[2]@ha
        auxL=hb@gauges[0]@hb;auxM=h2b@gauges[1]@h2b;auxR=hb@gauges[2]@hb
        term=np.exp(-1j*(2*angles[0]+3*angles[1]+2*angles[2]))*fixed_trace(auxM@auxR@auxL,2)*inserted_minor(L,M,R,2,signed_one,signed_one)
        total+=term/(quadrature**3);terms.append(complex(np.trace(term)))
    source_error=float(np.linalg.norm(total-direct_source)/max(1.,np.linalg.norm(direct_source)))
    require(source_error<2e-11,'Charged/source minor stays INSIDE the joint time-ordered Gauss product')
    # The actual global centre includes the INITIAL full Ref4 or charged Ref3.
    centre=lambda charge,ref:sum((complex(1j**(k*(2+charge+2+ref))) for k in range(4)),0j)/4
    require(abs(centre(0,4)-1)<1e-14 and abs(centre(1,3)-1)<1e-14 and abs(centre(1,4))<1e-14,
        'Centre-character projection retains the true Ref transition and rejects omission')
    return dict(closed_trace_rows=rows,source_cross_Fock=direct_source,source_cross_coefficient_integral=total,
        source_minor_relative_error=source_error,source_character_real_min=float(min(z.real for z in terms)),
        source_character_imaginary_max=float(max(abs(z.imag) for z in terms)),
        true_neutral_Ref4_centre=centre(0,4),true_charged_Ref3_centre=centre(1,3),wrong_unchanged_Ref4_centre=centre(1,4),
        prefactors_retained=dict(active_half=math.sqrt(pa),auxiliary_half=math.sqrt(pb),active_middle=p2a,auxiliary_middle=p2b),
        positivity='Only the COMPLETE reflected joint-Haar/source trace is positive. Individual character terms are complex and may have negative real part.',
        all_auxiliary_flavours='Each flavour takes [z_f^N_f] separately; identical coefficients are then raised to copies. This is NOT [z^(copies*N)]det(1+zM)^copies.')


def physical_states(active,auxiliary,cutoff: int,charge: int) -> list[tuple[int,int,int,int]]:
    result=[]
    for ia,state in enumerate(active):
        for ib,aux in enumerate(auxiliary):
            rho=local_number(state)+local_number(aux)-2-charge
            for cycle in range(-cutoff,cutoff+1): result.append((ia,ib,cycle,cycle+rho))
    return result


def sector_factor(gamma: np.ndarray,kactive: float,kaux: float,normal: np.ndarray,
                  grid: int,cutoff: int,charge: int) -> dict[str,Any]:
    active=occupations(4,2+charge);auxiliary=occupations(4,2)
    projector,joint=joint_projector(active,auxiliary,1,2+charge)
    selected=np.flatnonzero(np.diag(projector));physical=physical_states(active,auxiliary,cutoff,charge)
    factors=[];halfsteps=[]
    for y in normal:
        samples=[];physical_halves=[]
        for t in 2*math.pi*np.arange(grid)/grid:
            ha,pa,_=wilson_spatial(gamma,kactive,float(t),float(y))
            hb,pb,_=wilson_spatial(gamma,kaux,float(t),float(y))
            full=math.sqrt(pa*pb)*np.kron(exterior(ha,active),exterior(hb,auxiliary))
            samples.append(full[:,selected]);physical_halves.append(full)
        transform=np.fft.fft(np.asarray(samples),axis=0)/grid
        # U[0,1]=exp(i theta) uses A0=theta,A1=0. The physical Fourier
        # index is E0=n+rho0, NOT the cycle n; the latter deletes Gauss dressing.
        factors.append(np.array([transform[e0%grid,ia*len(auxiliary)+ib,:] for ia,ib,_,e0 in physical]))
        halfsteps.append(physical_halves)
    return dict(active=active,auxiliary=auxiliary,physical=physical,factors=factors,
        halfsteps=halfsteps,middle_Haar_rank=len(selected),joint_projector=projector,charge=charge,
        basis_index={(ia,ib,n):i for i,(ia,ib,n,_) in enumerate(physical)})


def normal_heat() -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    coordinates=np.array([-.025,0.,.025]);sigma=.04;radius=.12
    logk=-coordinates**2/(4*sigma*sigma)-sigma*sigma*coordinates**2/(radius*radius*(radius*radius-coordinates**2))
    k=np.exp(logk);k/=np.linalg.norm(k);spacing=float(coordinates[1]-coordinates[0])
    derivative=np.zeros((2,3))
    for i in range(2):
        derivative[i,i]=math.sqrt(k[i+1]/k[i])/spacing
        derivative[i,i+1]=-math.sqrt(k[i]/k[i+1])/spacing
    H=2*sigma*sigma*derivative.T@derivative
    spectrum,frame=eigh(H)
    half=(frame*np.exp(-spectrum/2))@frame.T
    require(bool(np.linalg.norm(H@k)<1e-11 and half.min()>0),'Actual compact normal A†A heat is positive and noncommuting with matter')
    return coordinates,k,H,half


def raw_source(neutral: dict[str,Any],charged: dict[str,Any],one: np.ndarray) -> tuple[np.ndarray,float]:
    Sn=creation(one,neutral['active'],charged['active'])
    rows=len(charged['physical']);columns=len(neutral['physical'])
    result=np.zeros((one.shape[1]*rows,columns),complex);defect=0.
    # Body at cell1 -> Ref0 uses the literal integer path (E0,E1)=(0,1).
    for column,(ia,ib,n,e0) in enumerate(neutral['physical']):
        for target,state in enumerate(charged['active']):
            for ref,S in enumerate(Sn):
                weight=S[target,ia]
                if abs(weight)==0:continue
                added=set(state)-set(neutral['active'][ia]);require(len(added)==1,'Literal CAR insertion changes ONE occupied mode')
                site=next(iter(added))//2;newn=n+(1 if site==1 else 0)
                row=charged['basis_index'].get((target,ib,newn))
                if row is None:continue
                newe0=charged['physical'][row][3]
                require(newe0==e0,'Separated integer path preserves E0 and shifts E1, including Refhole charge at0')
                result[ref*rows+row,column]+=weight
                defect=max(defect,abs((newe0-newn)-(local_number(state)+local_number(charged['auxiliary'][ib])-3)))
    return result,defect


def native_boundary(gamma4: np.ndarray,grid: int=32,cutoff: int=2,steps: int=8) -> dict[str,Any]:
    gamma=frozen.two_spin_frame(gamma4)
    kactive=1/(2*(math.exp(.06)+3));kaux=.04
    coordinates,k,Hphi,Nhalf=normal_heat()
    neutral=sector_factor(gamma,kactive,kaux,coordinates,grid,cutoff,0)
    charged=sector_factor(gamma,kactive,kaux,coordinates,grid,cutoff,1)
    nn=len(neutral['physical']);nc=len(charged['physical'])
    Vn=np.kron(Nhalf,np.eye(nn))@cast(np.ndarray,block_diag(*neutral['factors']))
    Vc=np.kron(Nhalf,np.eye(nc))@cast(np.ndarray,block_diag(*charged['factors']))
    Un,sn,_=np.linalg.svd(Vn,full_matrices=False);Uc,sc,_=np.linalg.svd(Vc,full_matrices=False)
    lambda0=float(sn[0]**2);ln=sn**2/lambda0;lc=sc**2/lambda0
    Kn=(Un*ln)@Un.conj().T;Kh=(Un*np.sqrt(ln))@Un.conj().T
    cycles=np.array([state[2] for state in neutral['physical']]);joints=[state[:2] for state in neutral['physical']]
    z,w=roots_legendre(160);radius=.7;theta=radius*z;f=(1-z*z)**2
    difference=cycles[:,None]-cycles[None,:]
    same=np.array([[a==b for b in joints] for a in joints])
    filter_matrix=radius*np.einsum('q,ijq,q->ij',w,np.exp(-1j*difference[:,:,None]*theta),f)/(2*math.pi)*same
    R=np.kron(np.outer(k,k),filter_matrix)
    Xi=Kh@R@(Un*ln**(steps/2));Z=float(np.linalg.norm(Xi,'fro')**2)
    shift=np.zeros((nn,nn),complex)
    for column,(ia,ib,n,_) in enumerate(neutral['physical']):
        row=neutral['basis_index'].get((ia,ib,n+1))
        if row is not None:shift[row,column]=1
    B=np.kron(np.eye(len(coordinates)),(shift-shift.conj().T)/(2j))
    Xph=Kh@B.conj().T@Xi;Zph=float(np.linalg.norm(Xph,'fro')**2)
    h0,_,T0=wilson_spatial(gamma,kactive,0.,0.)
    values,vectors=eigh(T0);stable=vectors[:,values<1]
    embedding=frozen.two_spin_embedding(gamma4)
    packet=np.array([1.,1.])/math.sqrt(2)
    seed=np.kron(packet[:,None],embedding.conj().T)
    coefficients=stable@stable.conj().T@seed
    coefficients/=np.linalg.norm(coefficients)
    extended=dict(charged);extended['physical']=physical_states(charged['active'],charged['auxiliary'],cutoff+1,1)
    extended['basis_index']={(ia,ib,n):i for i,(ia,ib,n,_) in enumerate(extended['physical'])}
    source,defect=raw_source(neutral,extended,coefficients)
    ne=len(extended['physical']);embedding_c=np.zeros((ne,nc))
    for column,(ia,ib,n,_) in enumerate(charged['physical']):embedding_c[extended['basis_index'][ia,ib,n],column]=1
    source=np.kron(np.eye(len(coordinates)),source)
    # Reorder node-major Ref4 outputs to Ref-major for the exact charged factor.
    permutation=np.arange(len(coordinates)*4*ne).reshape(len(coordinates),4,ne).transpose(1,0,2).reshape(-1)
    source=source[permutation]
    charged_factor=np.kron(np.eye(4),np.kron(np.eye(len(coordinates)),embedding_c)@Vc)/math.sqrt(lambda0)
    charged_isometry=np.kron(np.eye(4),np.kron(np.eye(len(coordinates)),embedding_c)@Uc)
    raw=source@Xph/math.sqrt(Zph);ranged=charged_isometry.conj().T@raw
    boundary=charged_factor.conj().T@raw
    M0=float(np.linalg.norm(raw,'fro')**2);M0range=float(np.linalg.norm(ranged,'fro')**2);M1=float(np.linalg.norm(boundary,'fro')**2)
    cvalues=np.tile(lc,4);spectral=charged_isometry.conj().T@raw
    band=(cvalues>=math.exp(-.25))&(cvalues<=1.)
    above=cvalues>1.;price=float(np.sum((cvalues[above,None]-math.exp(-.25))*abs(spectral[above])**2))
    PE=float(np.sum(cvalues[band,None]*abs(spectral[band])**2));lower=M1-math.exp(-.25)*M0range-price
    require(PE>=lower-1e-9,'Exact full-Gauss source moments use COMMON neutral scale and explicit charged-above/null prices')
    phi_noncommutator=float(np.linalg.norm(np.kron(Hphi,np.eye(nn))@Kn-Kn@np.kron(Hphi,np.eye(nn)),2))
    auxiliary_number_variance=float(np.var([local_number(state) for state in charged['auxiliary']]))
    diagonal=np.sum(abs(Xph)**2,axis=1)/Zph
    aux_number=np.tile([local_number(neutral['auxiliary'][ib]) for _,ib,_,_ in neutral['physical']],len(coordinates))
    aux_mean=float(diagonal@aux_number)
    aux_prepared_variance=float(diagonal@(aux_number-aux_mean)**2)
    require(aux_prepared_variance>0,'The actual photon-prepared auxiliary charge variance is retained')
    summary=dict(scope='Exact two-cell invariant-spin spatial Wilson/Gauss restriction with one auxiliary flavour; not full macro SU4 admission.',
        parameters=dict(grid=grid,cutoff=cutoff,steps=steps,active_kappa=kactive,auxiliary_kappa=kaux,normal_coordinates=coordinates,normal_H=Hphi),
        canonical_counts=dict(active_neutral=6,active_charged=4,auxiliary=6,joint_neutral=36,joint_charged=24,
            neutral_joint_Haar_rank=neutral['middle_Haar_rank'],charged_joint_Haar_rank=charged['middle_Haar_rank'],Ref4_holes=4),
        common_neutral_lambda0=lambda0,neutral_nonzero_rank=len(sn),charged_nonzero_rank=4*len(sc),
        exact_neutral_conditioned_denominator=Z,exact_photon_conditioned_denominator=Zph,
        full_normalized_true_M0=M0,range_M0=M0range,Knull_M0=M0-M0range,full_normalized_true_M1=M1,
        exact_common_neutral_PE25_Gram=PE,actual_PE25_nonzero_rank=int(np.sum(band)),
        transfer_energy_cut=.25,range_Rayleigh_PE25_lower=lower,charged_above_neutral_price=price,
        charged_over_neutral_top=float(lc[0]),actual_Gauss_defect=defect,
        canonical_occupation_counting_variance=auxiliary_number_variance,
        prepared_photon_auxiliary_onsite_charge_mean=aux_mean,
        prepared_photon_auxiliary_onsite_charge_variance=aux_prepared_variance,Phi_heat_noncommutator=phi_noncommutator,
        true_canonical_halfsteps='At every theta,Y, sqrt(detB_active detB_aux) Gamma_Nactive(sqrt(Tactive)) tensor Gamma_Naux(sqrt(Taux)); one JOINT local-Haar projector. Actual normal heat sandwiches these exact field-dependent factors.',
        all_cycle_and_null_scope='Finite electric Galerkin [-cutoff,cutoff], source codomain extended by one cycle. Every retained cycle and exact K-null source component remains; no charged own-max or source-commutation shortcut.',
        macro_source_admitted=False)
    return dict(summary=summary,runtime=dict(gamma=gamma,kactive=kactive,kaux=kaux,
        coordinates=coordinates,k=k,Nhalf=Nhalf,neutral=neutral,Kn=Kn,lambda0=lambda0,
        R=R,Un=Un,ln=ln,grid=grid,cutoff=cutoff,steps=steps,
        source=source,extended_charged=extended,charged_factor=charged_factor,Xph=Xph,Zph=Zph))


def exact_determinant_majorant(runtime: dict[str,Any]) -> dict[str,Any]:
    """Positive scalar majorant of the ACTUAL joint canonical/Haar kernel.

    No individual Gauss-character sign or inverse normal heat is used. The
    electric Galerkin projector's Dirichlet-kernel cost is displayed rather
    than pretending the projection has a positive position kernel.
    """
    gamma=runtime['gamma'];coordinates=runtime['coordinates'];grid=runtime['grid']
    cutoff=runtime['cutoff'];steps=runtime['steps'];m=steps//2
    active=runtime['kactive'];aux=runtime['kaux'];neutral=runtime['neutral']
    Nhalf=runtime['Nhalf'];N=Nhalf@Nhalf;N2=N@N
    scalar_heat_ratio=float(np.max(N/N2))
    zero=sector_factor(gamma,active,aux,np.zeros(len(coordinates)),grid,cutoff,0)
    nn=len(neutral['physical'])
    V0=np.kron(Nhalf,np.eye(nn))@cast(np.ndarray,block_diag(*zero['factors']))
    A=V0@V0.conj().T/runtime['lambda0'];K=runtime['Kn'];E=K-A
    F0=zero['halfsteps'][0][0];scale=float(np.linalg.norm(F0,2))
    # A scalar full-rank canonical envelope commutes with the longitudinal
    # row gauge e^{-i rho theta}; a generic F0 F0dag would not.
    D=(scale*scale/runtime['lambda0'])*np.eye(len(F0))
    base_inverse=np.eye(len(F0))/scale
    zero_leg=0.;phi_leg=0.;all_leg=0.
    for j in range(grid):
        reference=zero['halfsteps'][0][j]
        zero_leg=max(zero_leg,float(np.linalg.norm(reference@base_inverse,2)))
        for row in neutral['halfsteps']:
            all_leg=max(all_leg,float(np.linalg.norm(row[j]@base_inverse,2)))
            phi_leg=max(phi_leg,float(np.linalg.norm(row[j]@np.linalg.inv(reference)-np.eye(len(reference)),2)))
    # Matrix norm factors sandwich the ONE joint Haar projector of norm1.
    epsilon=(2*phi_leg+phi_leg*phi_leg)*zero_leg*zero_leg
    envelope=max(all_leg*all_leg,zero_leg*zero_leg)
    band=2*cutoff+1
    # |Pcut(theta',theta)|<=band/(2pi); both external projections are paid.
    projected_error=band*band*epsilon*scalar_heat_ratio
    projected_envelope=band*band*envelope*scalar_heat_ratio
    p=128*.7/(315*math.pi)
    internal_trace=float(np.trace(np.linalg.matrix_power(D,steps)).real)
    scalar_trace=float(np.trace(np.linalg.matrix_power(N,2*steps)).real)
    vertex_upper=band*band*p*scalar_trace*internal_trace
    R=runtime['R'];rows=[]
    for a in range(m):
        term=R@np.linalg.matrix_power(K,a)@E@np.linalg.matrix_power(A,m-1-a)
        actual=float(np.linalg.norm(term,'fro')**2)
        upper=projected_error**2*projected_envelope**(2*m-2)*vertex_upper
        require(actual<=upper+1e-10,'Actual exact determinant/Gauss/Phi two-E source loop obeys the single-f² positive majorant')
        rows.append(dict(left_power=a,right_power=m-1-a,actual_two_E_loop=actual,
            proved_single_f_squared_upper=upper))
    actual_error=float(np.linalg.norm(R@(np.linalg.matrix_power(K,m)-np.linalg.matrix_power(A,m)),'fro'))
    bound=m*projected_error*projected_envelope**(m-1)*math.sqrt(vertex_upper)
    require(actual_error<=bound+1e-10,'Whole reflected power uses Minkowski after actual two-E loops')
    trace_rows=[]
    for power in (steps,steps+1,steps+2):
        actual=float(np.trace(np.linalg.matrix_power(K,power)).real)
        reference=float(np.trace(np.linalg.matrix_power(A,power)).real)
        trace_rows.append(dict(power=power,actual_full_canonical_Gauss_trace=actual,
            actual_zero_Phi_canonical_Gauss_trace=reference,actual_over_reference=actual/reference))
    # Independent scalar localized error makes wrong p² visibly false.
    theta=2*math.pi*np.arange(24)/24;principal=(theta+math.pi)%(2*math.pi)-math.pi
    filt=np.where(abs(principal)<.7,(1-(principal/.7)**2)**2,0.)
    p_grid=float(filt@filt/len(filt))
    heat=np.eye(len(filt))*.99+np.ones((len(filt),len(filt)))*(.01/len(filt))
    doubled=heat@heat;window=np.diag((abs(principal)<.7).astype(float))
    error=.01*window@doubled@window
    loop=float(np.linalg.norm(np.diag(filt)@error@np.linalg.matrix_power(heat,m-1),'fro')**2)
    correct=.01**2*p_grid*float(np.trace(np.linalg.matrix_power(heat,2*m+2)))
    require(loop<=correct+1e-12 and loop>p_grid*correct,
        'A positive stationary/doubled scalar majorant rejects independent-filter p²')
    return dict(actual_native_Phi_kernel_relative_leg_error=phi_leg,
        actual_zero_Phi_toron_leg_norm=zero_leg,actual_all_Phi_toron_leg_norm=all_leg,
        positive_stationary_heat='h0(theta_prime,theta)=1/(2pi), a positive Haar heat projector. Actual normal scalar heat N is retained; doubled N² uses the measured POSITIVE entry ratio, never a matrix inverse.',
        positive_doubled_normal_heat_entry_ratio=scalar_heat_ratio,
        canonical_majorant_D_trace_at_Nt=internal_trace,
        actual_electric_Galerkin_Dirichlet_price=band,
        projected_kernel_error_majorant=projected_error,projected_kernel_majorant=projected_envelope,
        single_f_squared_Haar_probability=p,single_vertex_loop_weight_upper=vertex_upper,
        actual_two_E_source_loops=rows,actual_filtered_power_HS_error=actual_error,proved_power_HS_upper=bound,
        actual_trace_ratios_Nt_Nt1_Nt2=trace_rows,
        localized_positive_falsifier=dict(actual_two_E_loop=loop,single_vertex_p_upper=correct,
            wrong_independent_filters_p_squared_upper=p_grid*correct),
        positivity_scope='Whole joint Haar projector and V Vdag only. Character signs remain inside the kernel; ALL active/auxiliary occupations contribute to Tr D^Nt. Scalar/Dirichlet majorants are norm bounds, never kernel Loewner order.',
        macro_scope='This native stationary majorant is intentionally loose and contains an explicit finite electric-projection cost. It is not the macro induced-beta heat, and supplies no numeric macro source Gram.')




def cyclic_toron_trace_probe(runtime: dict[str,Any],parameters: Parameters) -> dict[str,Any]:
    """Completed positive trace boundary, without a constant internal J."""
    K=runtime['Kn'];neutral=runtime['neutral'];nn=len(neutral['physical'])
    normal=len(runtime['coordinates']);cutoff=runtime['cutoff'];band=2*cutoff+1
    m=runtime['steps']//2;k=runtime['k'];P=np.kron(np.outer(k,k),np.eye(nn))
    # Common shift of BOTH oriented links by delta/2 is generated by
    # (E0+E1)/2=n+rho0/2. Retain the Gauss affine part in every occupation.
    harmonic=np.tile([(n+e0)/2 for _,_,n,e0 in neutral['physical']],normal)
    equal=np.isclose(harmonic[:,None],harmonic[None,:],atol=0.,rtol=0.)
    A=K*equal;E=K-A
    # This is the exact positive Haar twirl, not a sign-indefinite worldline
    # truncation. The double cover closes the affine half-integer charges.
    require(float(eigh(A,eigvals_only=True)[0])>-1e-11,'Full-Phi/all-occupation harmonic twirl is genuinely positive')
    z,w=roots_legendre(160);radius=.7;f=(1-z*z)**2;p=128*radius/(315*math.pi)
    modes=np.arange(-2*cutoff,2*cutoff+1)
    f2=np.array([radius*np.sum(w*np.exp(-1j*q*radius*z)*f*f)/(2*math.pi) for q in modes])
    cycles=np.array([n for _,_,n,_ in neutral['physical']]);joints=[state[:2] for state in neutral['physical']]
    same=np.array([[a==b for b in joints] for a in joints])
    F2=np.array([[f2[int(cycles[i]-cycles[j])+2*cutoff]*same[i,j] for j in range(nn)] for i in range(nn)])
    F2=np.kron(np.eye(normal),F2)
    rows=[]
    for a in range(m):
        b=m-1-a;X=np.linalg.matrix_power(K,a)@E@np.linalg.matrix_power(A,b)
        Q=P@(X@X.conj().T)@P
        tensor=Q.reshape(normal,36,band,normal,36,band)
        coefficients=[]
        for q in modes:
            coefficients.append(sum(tensor[y,j,n,y,j,n-q] for y in range(normal)
                for j in range(36) for n in range(band) if 0<=n-q<band))
        coefficients=np.asarray(coefficients)
        actual=float(np.trace(F2@Q).real)
        flat=p*float(np.trace(Q).real)
        tail=float(np.sum(abs(f2*coefficients))-abs(f2[2*cutoff]*coefficients[2*cutoff]))
        projected=float(np.linalg.norm(runtime['R']@X,'fro')**2)
        require(projected<=actual+1e-10 and abs(actual-flat)<=tail+1e-10,
            'Completed positive squared-telescope trace has ONE f² vertex plus its exact winding Fourier tail')
        control_centres=2*math.pi*np.arange(24)/24
        translated=np.array([float(np.sum(f2*coefficients*np.exp(1j*modes*centre)).real) for centre in control_centres])
        peak=int(np.argmax(translated))
        require(float(translated[peak])>flat+1e-10 and
            float(np.max(abs(translated-flat)))<=tail+1e-10,
            'Genuine winding falsifies globally uniform single-p while the priced translated f² identity survives')
        rows.append(dict(left_power=a,right_power=b,full_Phi_canonical_positive_loop_trace=float(np.trace(Q).real),
            exact_one_vertex_f_squared_trace=actual,actual_projected_filter_loop=projected,
            flat_single_p_trace=flat,exact_f_squared_winding_activity_price=tail,
            toron_diagonal_density_Fourier_coefficients=coefficients,
            unpriced_single_p_would_be_valid=actual<=flat+1e-12,
            negative_control_translated_filter_centre=float(control_centres[peak]),
            negative_control_translated_filter_trace=float(translated[peak]),
            global_unpriced_single_p_is_falsified=True))
    reference=P@np.linalg.matrix_power(A,2*m)@P
    reference_slot=float(np.trace(F2@reference).real);reference_p=p*float(np.trace(reference).real)
    require(abs(reference_slot-reference_p)<1e-10,'Genuinely positive harmonic twirl gives the exact single-p boundary identity')
    # Literal determinant/canonical/local-Gauss cycles under common link shift.
    gamma=runtime['gamma'];shifts=2*math.pi*np.arange(24)/24;det_rows=[]
    for copies in (1,3):
        quad=2*(1+copies)+1;target=1+copies;values=[]
        for shift in shifts:
            ha,pa,_=wilson_spatial(gamma,runtime['kactive'],.31,.007,float(shift))
            hb,pb,_=wilson_spatial(gamma,runtime['kaux'],-.13,-.011,float(shift))
            ha2,pa2,_=wilson_spatial(gamma,runtime['kactive'],-.22,-.009,float(shift))
            hb2,pb2,_=wilson_spatial(gamma,runtime['kaux'],.41,.012,float(shift))
            value=0j
            for j,l in itertools.product(range(quad),repeat=2):
                angles=2*math.pi*np.array([j,l])/quad
                g0=np.diag(np.exp(1j*angles[0]*np.array([1,1,0,0])))
                g1=np.diag(np.exp(1j*angles[1]*np.array([1,1,0,0])))
                value+=np.exp(-1j*target*angles.sum())*fixed_trace(ha@g0@ha2@ha2@g1@ha,2)*fixed_trace(hb@g0@hb2@hb2@g1@hb,2)**copies/quad**2
            value*=pa*pa2*(pb*pb2)**copies
            require(value.real>0 and abs(value.imag)<1e-10*value.real,'Only the COMPLETE reflected Gauss/canonical cycle is positive')
            values.append(value)
        values=np.asarray(values);fourier=np.fft.fft(values)/len(values)
        require(float(np.ptp(values.real)/np.mean(values.real))>1e-4,'Literal short-ring winding is NOT falsely called contractible/uniform')
        require(float(np.linalg.norm(values-values[np.mod(np.arange(len(values))+len(values)//2,len(values))])/np.linalg.norm(values))<1e-11,
            'Closed complete cycle sees holonomy theta+2alpha, not an open single-link phase')
        det_rows.append(dict(auxiliary_copies=copies,canonical_number_per_flavour=2,
            exact_joint_character_quadrature=quad,common_link_shifts=shifts,
            actual_closed_determinant_canonical_Gauss_trace=values,
            relative_shift_variation=float(np.ptp(values.real)/np.mean(values.real)),
            actual_winding_Fourier_coefficients=fourier,
            positive_reference='Haar average of the complete positive reflected cycle; NEVER truncate individual character/worldline terms.'))
    source=runtime['source'];charged=runtime['extended_charged']
    ch=np.tile([(n+e0)/2 for _,_,n,e0 in charged['physical']],4*normal)
    delta=.6*math.pi
    shifted=np.exp(1j*delta*ch)[:,None]*source*np.exp(-1j*delta*harmonic)[None,:]
    difference=ch[:,None]-harmonic[None,:]
    zero=source*(difference==0);one=source*(difference==.5)
    require(float(np.linalg.norm(source-zero-one))<1e-12 and
        float(np.linalg.norm(shifted-zero-np.exp(.5j*delta)*one))<1e-11,
        'True body/Ref integer strings retain open harmonic phases under common toron shifts')
    moments=[]
    for shift in (0.,delta,math.pi):
        S=np.exp(1j*shift*ch)[:,None]*source*np.exp(-1j*shift*harmonic)[None,:]
        boundary=runtime['charged_factor'].conj().T@S@runtime['Xph']/math.sqrt(runtime['Zph'])
        moments.append(dict(common_toron_shift=shift,actual_source_only_shifted_M1=float(np.linalg.norm(boundary,'fro')**2)))
    # Absolute activity bound for a LOCAL Dirac grandcanonical log determinant.
    # This is NOT a relative canonical/Haar/source trace or an RN bound.
    with mp.workdps(100):
        L=mp.mpf(parameters.spatial_size);V=L**3;Nt=mp.mpf(parameters.time_slices)
        copies_exact=10**(parameters.beta_power+4*parameters.kappa_power)//16
        ka=mp.power(10,-parameters.kappa_power);qa=16*ka
        aux_log=(mp.log10(4*V*(Nt+2))+mp.log10(copies_exact)
            +L*mp.log10(qa)-mp.log10(1-qa))
        body=max(1/(2*(math.expm1(a*mass)+4)) for a in parameters.spacings for mass in parameters.masses)
        qb=8*mp.mpf(body)
        ideal_body_log=(mp.log10(4*V*(Nt+2))+L*mp.log10(qb)-mp.log10(1-qb)) if qb<1 else None
    return dict(harmonic_translation='Common absolute holonomy delta shifts both links by delta/2; generator=(E0+E1)/2=n+rho0/2, with full affine occupation/Ref charge.',
        full_Phi_positive_twirl_reference_Nt_trace=float(np.trace(reference).real),
        exact_reference_one_f_squared_trace=reference_slot,exact_reference_single_p_trace=reference_p,
        actual_completed_squared_telescope_cycles=rows,literal_determinant_canonical_Gauss_common_shift=det_rows,
        source_open_string_control=dict(actual_nonzero_string_harmonic_generator=.5,
            actual_deleted_harmonic_phase_operator_error=float(np.linalg.norm(shifted-source,2)),
            actual_shifted_source_M1=moments,
            scope='Source alone is shifted while actual charged/neutral kernels and purifier stay fixed. These genuine open phases are NOT bounded by a spatial-winding tail.'),
        full_family_winding_activity=dict(actual_auxiliary_count_exact=str(copies_exact),
            actual_auxiliary_kappa_power=parameters.kappa_power,conservative_general_auxiliary_hopping_norm=16.,
            absolute_auxiliary_grandcanonical_logdet_winding_log10_upper=float(aux_log),
            frozen_Root_ideal_body_kappa=body,frozen_Root_ideal_body_8k=8*body,
            frozen_Root_ideal_commuting_projector_winding_log10_upper=float(ideal_body_log) if ideal_body_log is not None else None,
            C_completed_actual_critical_body_kappa=.1256305951044483,
            C_completed_actual_8k=8*.1256305951044483,C_completed_8k_winding_bound=None,
            theorem='For ||kH||<=q<1, logdet(I-kH)=-sum_n Tr(kH)^n/n. A genuine spatial winding needs n>=L, giving |winding activity|<=4V(Nt+2) sum_species q_species^L/(1-q_species). Unit-circle fugacity/Gauss twists stay unitary. Use norm16 for general noncommuting projectors; norm8 requires commuting/intertwining projectors and is recorded ONLY for the frozen ideal body law.',
            limitation='A grandcanonical absolute activity bound is NOT a relative canonical/Gauss/conditioned-source trace bound. Character terms need not be positive; the positive complete canonical trace denominator must still be joined. The actual C-completed critical body has8k>1 and needs a different positive/completed-law body winding proof.'),
        macro_single_vertex_uniform_density_admitted=False)


def phi_relative_control(gamma: np.ndarray) -> dict[str,Any]:
    rows=[]
    for kappa in (.06,.03,.015,.0075):
        actual=0.;form=0.;norm_y=.001
        for theta in (0.,.31,1.7,math.pi):
            h0,_,T0=wilson_spatial(gamma,kappa,theta,0.)
            h,_,T=wilson_spatial(gamma,kappa,theta,norm_y)
            actual=max(actual,float(np.linalg.norm(h@np.linalg.inv(h0)-np.eye(4),2)))
            form=max(form,float(np.linalg.norm(np.linalg.solve(h0,T-T0)@np.linalg.inv(h0),2)))
        # Deliberately loose separated-band halfstep constant, uniform in links.
        upper=512*kappa*kappa*(norm_y+8*norm_y*norm_y)/(1-6*kappa)**3
        require(actual<=upper,'Actual extra-kappa Phi halfstep comparison is in the uniform separated regime')
        rows.append(dict(kappa=kappa,normal_amplitude=norm_y,actual_halfstep_relative_operator_error=actual,
            actual_relative_positive_form_error=form,actual_error_over_kappa_squared_Y=actual/(kappa*kappa*norm_y),
            uniform_small_Phi_halfstep_upper=upper))
    return dict(rows=rows,law='Root phi_wilson_step b=b0+kappa Y0+kappa||Y||²I and c=c0+kappa Y0+kappa sum(Yj Uj-Uj†Yj)/2; the extra kappa is retained.',
        no_inverse_Phi_heat=True,macro_admission=False)


def macro_phi_budget(parameters: Parameters) -> dict[str,Any]:
    """Explicit full-flavour price for the actual extra-kappa normal remainder."""
    copies_exact=10**(parameters.beta_power+4*parameters.kappa_power)//16
    with mp.workdps(100):
        volume=mp.mpf(parameters.spatial_size)**3;nt=mp.mpf(parameters.time_slices)
        copies=mp.mpf(copies_exact)
        k=mp.power(10,-parameters.kappa_power);radius=mp.power(10,-2100)
        # V-scaled canonical count and detB half-prefactor are BOTH retained.
        coefficient_constant=mp.mpf(2)**24*(1+parameters.kappa_power*mp.log(10)+mp.log(64))**2
        half=coefficient_constant*k*k*(radius+8*radius*radius)/(1-6*k)**3
        # The actual separable analytic/projective coefficient sum pays V²;
        # an arbitrary uniform bounded multiplier would NOT suffice here.
        log_tensor=mp.mpf(2)**20*8*volume**2*copies*half
        delta=mp.expm1(log_tensor)
        step=2*delta+delta*delta
        powered=2*(nt+2)*step*mp.exp((nt+1)*mp.log1p(step))
        result=dict(auxiliary_count_exact=str(copies_exact),canonical_number_per_auxiliary=str(2*parameters.spatial_size**3),
            actual_tube_normal_radius='1e-2100',onebody_halfstep_kappa_squared_Phi_log10_upper=float(mp.log10(half)),
            projective_coefficient_constant=mp.nstr(coefficient_constant,30),
            projective_volume_squared_and_coordinate_factor='2^20 * 8 * V^2',
            all_auxiliary_canonical_and_detB_collective_log10_upper=float(mp.log10(delta)),
            all_auxiliary_projective_step_relative_Schatten_log10_upper=float(mp.log10(step)),
            Nt_plus2_unnormalized_power_trace_relative_log10_upper=float(mp.log10(powered)))
    result.update(proof='The positive b blocks have bmin=(1-6k)/(2k), ||delta b||<=k(||Y0||+||Y||²), ||delta f||<=4k||Y||. The ONE-body relative square-root Sylvester integral has only the logarithmic condition factor C=2^24(1+kpower log10+log64)^2, giving C k²(||Y||+8||Y||²)/(1-6k)^3. Exterior-N multiplication and detB prefactors pay <=8V per flavour. A Cauchy/separable coefficient expansion, NOT an arbitrary bounded multiplier, pays the additional 2^20 V coordinate/projective factor. Tensor ALL flavours before the joint Haar projector and scalar normal heat. The retained analytic coefficient sum gives the displayed relative Schatten/power price by the physical Root projective theorem. NO inverse Phi heat, source commutation or character-term positivity is used.',
        limitation='This supplies the small-Phi auxiliary boundary price only. Active charged/current kernels, full SU4/centre descent and the zero-Phi conditioned determinant/heat denominator must be joined before any macro M0/M1/PE or moving receiver Gram is admitted.',
        macro_source_admitted=False)
    return result


def portable(value: Any) -> Any:
    if isinstance(value,dict):return {key:portable(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [portable(item) for item in value]
    if isinstance(value,np.ndarray):return portable(value.tolist())
    if isinstance(value,np.generic):return portable(value.item())
    if isinstance(value,complex):return [value.real,value.imag]
    return value


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path,default=ROOT/'covariant_prepared_gauss_state_input_kappa400_sigma2000_r2100.json')
    args=parser.parse_args();start=time.monotonic()
    input_bytes=args.input.read_bytes();configuration=json.loads(input_bytes);parameters=Parameters(**configuration['photon_parameters'])
    require(hashlib.sha256(frozen.SNAPSHOT['covariant_prepared_gauss_state']).hexdigest()==FROZEN_SOURCE,'Frozen Source ABI must NOT be retagged')
    require(hashlib.sha256(physical_root.SOURCE_BYTES_AT_IMPORT['covariant_root_transfer.py']).hexdigest()==ROOT_SOURCE,'Physical Root6911 must be the actual executed supplier')
    gamma4,_=phi_frame(parameters);gamma=frozen.two_spin_frame(gamma4)
    characters=gauss_character_control(gamma)
    native_bundle=native_boundary(gamma4)
    full_components=full_component_canonical_control(gamma4)
    clock=frozen_time_step_control(gamma4)
    native=native_bundle['summary']
    majorant=exact_determinant_majorant(native_bundle['runtime'])
    cyclic=cyclic_toron_trace_probe(native_bundle['runtime'],parameters)
    phi=phi_relative_control(gamma)
    macro=macro_phi_budget(parameters)
    compiled=dict(frozen.COMPILED);compiled['covariant_canonical_gauss_boundary_kernel']=hashlib.sha256(OWN_BYTES).hexdigest()
    captured={name:frozen.SNAPSHOT[name].decode() for name in frozen.COMPILED};captured['covariant_canonical_gauss_boundary_kernel']=OWN_BYTES.decode()
    unchanged=all((ROOT/(name+'.py')).read_bytes().decode()==captured[name] for name in compiled)
    require(unchanged,'Scientific sources changed during exact-kernel execution')
    result=dict(plan=PLAN,governing_revision=REVISION,branch='NEW exact canonical/Gauss boundary representation, frozen Source2a0 ABI unchanged',
        input_configuration=dict(path=str(args.input),sha256=hashlib.sha256(input_bytes).hexdigest(),captured_bytes=input_bytes.decode()),
        actual_full_component_canonical_Root_source_control=full_components,
        actual_frozen_microscopic_time_step_convention=clock,
        actual_exact_determinant_single_vertex_majorant=majorant,
        actual_cyclic_toron_trace_winding_and_open_strings=cyclic,
        exact_character_and_source_minors=characters,actual_native_boundary=native,actual_extra_kappa_Phi=phi,
        full_family_auxiliary_Phi_budget=macro,
        full_family_loop_representation=dict(closed='Integral over ALL physical Phi/colour/U1 histories of scalar normal/electric heat and actual vacuum prefactors, times product_f [z_f^N_f]det(I+z_f ordered_product_t[sqrt(Tf(Phi_t,links_t)) G_t sqrt(Tf(Phi_(t+1),links_(t+1)))]). Identical auxiliary coefficients are raised to Naux ONLY after their separate canonical extractions.',
            source='At source boundaries use u_ref_dag Adj_N(C) M v_ref, C=M R L, Adj_0=I and Adj_j=e_j(C)I-C Adj_(j-1). This polynomial minor has no inverse(I+M). Keep integer SU4/U1 Wilson/Ref strings INSIDE u,v and actual charged microscopic factors/heat in M. Integrate the ONE product with ALL spectators, local Gauss characters and all centre/statistical branches; never separately average species.',
            Phi='Keep actual Dminus scalar/colour heat between EVERY Phi endpoint; actual sqrt(T(Phi)) and source projectors remain inside the time-ordered product. Small-Phi comparison prices the explicit kappa*Y primitive, not inverse Phi heat.',
            positivity='Reflected full trace/factor V Vdag after complete joint character integration is positive. No positivity/cancellation of separate determinant or character terms is asserted.'),
        compiled_source_hashes=compiled,captured_source_bytes=captured,compiled_source_unchanged=unchanged,
        loaded_main_code_sha256=hashlib.sha256(marshal.dumps(LOADED_MAIN_CODE)).hexdigest(),
        main_code_matches_captured_source=MAIN_CODE_MATCH,
        runtime=dict(executable=sys.executable,executable_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            python=sys.version,numpy=np.__version__,scipy=sys.modules['scipy'].__version__,mpmath=mp.__version__,
            numeric_extension_hashes={name:hashlib.sha256(Path(cast(str,module.__file__)).read_bytes()).hexdigest()
                for name,module in sys.modules.items() if name in ('numpy._core._multiarray_umath',
                    'numpy.linalg._umath_linalg','scipy.linalg._fblas','scipy.linalg._flapack','scipy.linalg._matfuncs_expm')},
            platform=platform.platform()),
        macro_conditioned_boundary=dict(photon_Gram=None,true_M0=None,true_M1=None,PE_Gram=None,
            high_energy_weight=None,moving_receiver_mixed_Gram=None,admitted=False,
            missing_join='Positive complete canonical/Gauss cyclic winding price and actual conditioned charged/photon/source denominator; the C-completed body cannot use8k expansion.'),
        elapsed_seconds=time.monotonic()-start,macro_source_admitted=False)
    args.output.write_text(json.dumps(portable(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),source_sha256=compiled['covariant_canonical_gauss_boundary_kernel'],native_M0=native['full_normalized_true_M0'],native_M1=native['full_normalized_true_M1'],native_PE25=native['exact_common_neutral_PE25_Gram'],elapsed_seconds=result['elapsed_seconds'],macro_source_admitted=False)))
    print('Natural exit: exact canonical coefficients, joint Gauss/centre characters, source minors and true native boundary; no macro admission')


if __name__=='__main__':main()

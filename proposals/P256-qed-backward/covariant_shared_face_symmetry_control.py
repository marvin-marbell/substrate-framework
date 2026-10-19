#!/usr/bin/env python3
"""Full-column/source reflection control of the NEW shared3path primitive.

This is not the affine Gaussian D-sign reference. Its native light tree
has a different explicit common phase. No all-Pi-free claim is made.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any
import numpy as np
from scipy.linalg import eigh

ROOT=Path(__file__).resolve().parent
SOURCE_NAMES=('covariant_shared_face_symmetry_control.py','covariant_shared_plaquette_photon.py',
 'covariant_native_collective_two_auxiliary.py','covariant_native_closed_auxiliary_loop.py',
 'covariant_native_source_gram_moments.py','covariant_native_haar_range_jets.py',
 'covariant_source_band_halfstep_control.py','covariant_induced_photon.py',
 'covariant_photon_energy_control.py','sea_selection.py','rotation_band.py')
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in SOURCE_NAMES}
PLAN_PATH=Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_SHA256_AT_IMPORT=hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
from covariant_native_closed_auxiliary_loop import auxiliary_path_half
from covariant_native_source_gram_moments import matched_tree_half
from covariant_induced_photon import Parameters,phi_frame
import covariant_native_haar_range_jets as native


def columns_at(gamma: np.ndarray,charge: int,thetaL: float,thetaR: float) -> tuple[np.ndarray,np.ndarray]:
    light=list(itertools.combinations(range(8),4+charge));aux=list(itertools.combinations(range(4),2))
    states=list(itertools.product(light,aux));rho=np.array([native.density(l,a,charge) for l,a in states])
    middle=np.flatnonzero(np.all(rho==0,axis=1))
    L=np.array(light);A=np.array(aux);half,pref=matched_tree_half(gamma,thetaL,0.)
    LM=pref*np.linalg.det(half[L[:,None,:,None],L[None,:,None,:]])
    phase=complex((np.exp(1j*thetaL)+1+np.exp(1j*(thetaL-thetaR)))/3)
    half,pref=auxiliary_path_half(gamma,.04,phase)
    AM=pref*np.linalg.det(half[A[:,None,:,None],A[None,:,None,:]])
    trip=np.array([(i,j) for i in range(len(light)) for j in range(6)]);cols=trip[middle]
    full=LM[trip[:,0,None],cols[None,:,0]]*AM[trip[:,1,None],cols[None,:,1]]
    return full,rho[:,3]


def source_at(gamma: np.ndarray,embedding: np.ndarray,packet: np.ndarray,theta: float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    lightN=list(itertools.combinations(range(8),4));lightC=list(itertools.combinations(range(8),5));aux=list(itertools.combinations(range(4),2))
    statesN=list(itertools.product(lightN,aux));statesC=list(itertools.product(lightC,aux))
    index:dict[tuple[tuple[int,...],tuple[int,...]],int]={s:i for i,s in enumerate(statesC)}
    weights=native.source_coefficients(gamma,packet)@embedding.conj().T
    S=np.zeros((4*len(statesC),len(statesN)),complex)
    for column,(occupied,a) in enumerate(statesN):
        mask=sum(1<<m for m in occupied)
        for mode in range(8):
            if mode in occupied:continue
            row=index[(tuple(sorted((*occupied,mode))),a)];below=(mask&((1<<mode)-1)).bit_count()
            phase=np.exp(1j*theta*(mode//2==3))
            for ref in range(4):S[ref*len(statesC)+row,column]=(-1)**(below+ref)*weights[mode,ref]*phase
    rhoN=np.array([native.density(l,a,0)[3] for l,a in statesN]);rhoC=np.array([native.density(l,a,1)[3] for l,a in statesC])
    return S,rhoN,rhoC


def run() -> dict[str,Any]:
    gamma4,metric=phi_frame(Parameters(kappa_power=400));ev,U=eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]);embedding=U[:,ev>0]
    _,frame=eigh(embedding.conj().T@gamma4[0]@embedding);embedding=embedding@frame[:,::-1]
    gamma=np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    packet=np.array([.1,.8,.4,.1]);packet/=np.linalg.norm(packet)
    outpacket=np.array([.5,.4,.1,.6]);outpacket/=np.linalg.norm(outpacket)
    rows=[]
    for normal in (-.09,0.,.09):
        actual=gamma.copy();actual[1]*=1+normal
        for thetaL,thetaR in ((.37,-.29),(-1.13,.91),(2.71,-2.43)):
            for charge in (0,1):
                F,rho=columns_at(actual,charge,thetaL,thetaR)
                swapped,_=columns_at(actual,charge,thetaR,thetaL)
                expected=np.exp(1j*(thetaL-thetaR)*rho)[:,None]*swapped
                defect=float(np.linalg.norm(F-expected)/np.linalg.norm(F))
                periodicL,_=columns_at(actual,charge,thetaL+2*math.pi,thetaR)
                periodicR,_=columns_at(actual,charge,thetaL,thetaR+2*math.pi)
                rows.append(dict(normal=normal,thetaL=thetaL,thetaR=thetaR,charge=charge,
                    all_output_canonical_rows=F.shape[0],all_joint_Gauss_middle_columns=F.shape[1],
                    actual_reflection_even_fullcolumn_relative_defect=defect,
                    actual_literal_two_large_gauge_period_relative_defects=[float(np.linalg.norm(F-periodicL)/np.linalg.norm(F)),float(np.linalg.norm(F-periodicR)/np.linalg.norm(F))]))
    sources=[]
    thetaL=.37;thetaR=-.29
    for name,p in (('literal_source',packet),('independent_fixed_target',outpacket)):
        S,rhoN,rhoC=source_at(gamma,embedding,p,thetaL);swapped,_,_=source_at(gamma,embedding,p,thetaR)
        expected=np.tile(np.exp(1j*(thetaL-thetaR)*rhoC),4)[:,None]*swapped*np.exp(-1j*(thetaL-thetaR)*rhoN)[None,:]
        sources.append(dict(boundary=name,all_neutral_canonical_columns=S.shape[1],all_charged_Ref4_rows=S.shape[0],
            actual_same_reflection_source_or_target_relative_defect=float(np.linalg.norm(S-expected)/np.linalg.norm(S))))
    max_defect=max(row['actual_reflection_even_fullcolumn_relative_defect'] for row in rows)
    if max_defect>1e-11 or max(row['actual_same_reflection_source_or_target_relative_defect'] for row in sources)>1e-11:
        raise ArithmeticError('Actual full-column/shared-source reflection statement failed')
    descriptor=dict(plan_id='issue-222-induced-photon-delta13-continuation-20261008',plan_sha256_at_IMPORT=PLAN_SHA256_AT_IMPORT,
        primitive='Same NEW shared3path B/F primitive as connected supplier; no separate law',
        all_active_canonical_numbers=[4,5],all_active_Fock_counts=[70,56],all_auxiliary_canonical_states=6,
        normal_nodes=[-.09,0.,.09],auxiliary_kappa=.04,active_kappa=.09,light_homotopy=0.,original_Ref=4,prepared_Ref=3,
        reflection='(R Psi)(thetaL,thetaR)=exp[i(thetaL-thetaR) rho3] Psi(thetaR,thetaL), SAME charged/neutral total rho including Ref/background',
        exact_full_column_factor='F(thetaL,thetaR)=exp(i thetaL rho3) Gamma_light(Smean) Gamma_aux[S((1+exp(-i thetaL)+exp(-i thetaR))/3)] P_totalrho0, with detBpref and Phi heat retained',
        exact_reflection_involution='R²=I, R is unitary on declared two-cycle L2 physicalGauss Hilbert; R F=F for EVERY complete middle column, so R=+1 onrangeK',
        exact_antisymmetric_curl='R (sin thetaL-sin thetaR) R=-(sin thetaL-sin thetaR); hence sqrtK Bminus sqrtK=0 on untruncated declaredgraph two-cycle model',
        physical_cut_limit='ALL7edges |E|<=12 need not be R invariant because tree density support differs. Tiny finite-cut Bminus residual is not a radiative mode; connected supplier reports numerical resolution rather than normalizing it.',
        scope='Complete finite Pauli/Phi3 and analytic compact TWO-CYCLE graph statement; not a complete 4D/untruncated EM theory. No affine-reference all-boundaries-free retag or physicalPi-free claim.')
    return dict(input_descriptor=descriptor,input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        physical_mean_descriptor=metric,actual_fullcolumn_reflection_and_period_rows=rows,actual_same_reflection_source_and_fixed_target_rows=sources,
        physical_decision='Three equal path SF restores shared-face response but has a reflection-even positive range only; antisymmetric face curl is Knull in the declared untruncated two-cycle graph. A second independent inducing channel, not a quartercut or noisy normalized Bminus, is the material repair.',
        massless_photon_or_common_cone_or_macro_source_admitted=False)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    start=time.monotonic();result=run();result['runtime_seconds']=time.monotonic()-start
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('input_descriptor_sha256','source_at_import','physical_decision','runtime_seconds')}))


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Actual even-box Coulomb pairs and exact selected Source2a Gram contract.

The auxiliary occupied graph is a priced comparison, not an interacting state.
The full physical photon/charged Grams remain honest unpriced fields.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes();LOADED=sys._getframe().f_code
MATCH=LOADED==compile(OWN_BYTES,LOADED.co_filename,'exec',dont_inherit=True,optimize=sys.flags.optimize)
if not MATCH:raise ArithmeticError('Loaded Coulomb-pair bridge differs from captured bytes')
import numpy as np
from scipy.linalg import eigh
import covariant_auxiliary_charge_clock_bridge as clock
ring=clock.ring
SOURCE_SHA='2a0ecfb2341fceb7866d5062cf33d1da71b2fafa4b148829d21d14f0650fe307'
SELECTED='covariant_prepared_gauss_state_conditioned_source_physical_Dminus_verified_20261008.json'
SELECTED_SHA='ae1c756921a90752bf580d4c149d8c4c258224f291303dcc969245f9d543db1b'


def literal_geometry(length: int) -> tuple[np.ndarray,np.ndarray,np.ndarray,list[tuple[int,int,int]],dict[tuple[int,int,int],int]]:
    vertices=list(itertools.product(range(length),range(length),range(length)));lookup={x:i for i,x in enumerate(vertices)}
    volume=len(vertices);D=np.zeros((volume,3*volume));harmonic=np.zeros((3*volume,3))
    for axis in range(3):
        harmonic[axis*volume:(axis+1)*volume,axis]=1/math.sqrt(volume)
        for site in vertices:
            neighbour=list(site);neighbour[axis]=(neighbour[axis]+1)%length
            target=(neighbour[0],neighbour[1],neighbour[2]);edge=axis*volume+lookup[site]
            D[lookup[site],edge]=1;D[lookup[target],edge]=-1
    values,frame=eigh(D@D.T);positive=values>1e-10
    inverse=(frame[:,positive]/values[positive])@frame[:,positive].T
    coexact=np.eye(3*volume)-D.T@inverse@D-harmonic@harmonic.T
    ring.require(float(np.linalg.norm(D@coexact,2))<1e-11,'Actual literal cubic coexact quotient annihilates divergence')
    return D,coexact,harmonic,vertices,lookup


def literal_links(field: np.ndarray,length: int,vertices: list[tuple[int,int,int]],lookup: dict[tuple[int,int,int],int]) -> list[np.ndarray]:
    volume=len(vertices);links=[]
    for axis in range(3):
        shift=np.zeros((volume,volume),complex)
        for site in vertices:
            neighbour=list(site);neighbour[axis]=(neighbour[axis]+1)%length
            target=(neighbour[0],neighbour[1],neighbour[2]);index=lookup[site]
            shift[index,lookup[target]]=np.exp(1j*field[axis*volume+index])
        links.append(np.kron(shift,np.eye(4)))
    return links


def actual_ground_frame(gamma: np.ndarray,kappa: float,links: list[np.ndarray]) -> tuple[np.ndarray,dict[str,Any]]:
    modes=len(links[0]);volume=modes//4;number=2*volume
    half,whole_prefactor,physical=ring.supplier.physical_spatial_onebody(gamma,kappa,links,[np.zeros((modes,modes),complex) for _ in range(4)],np.eye(modes))
    O,J=clock.chiral_coordinates(gamma,volume);T=O.conj().T@physical@O
    _,vectors=eigh(T);ground=vectors[:,-number:]
    upper=np.array([4*x+s for x in range(volume) for s in (0,1)])
    lower=np.array([4*x+s for x in range(volume) for s in (2,3)])
    X=ground[upper]@np.linalg.inv(ground[lower]);hermitian=float(np.linalg.norm(X-X.conj().T,2))
    ring.require(hermitian<1e-10,'Actual central-link Root graph is Hermitian in physical gamma5-aligned coordinates')
    values,frame=eigh(np.eye(number)+X@X);normal=(frame/np.sqrt(values))@frame.conj().T
    occupied=np.zeros((modes,number),complex);occupied[upper]=X@normal;occupied[lower]=normal
    return occupied,dict(physical_modes=modes,canonical_number=number,graph_Hermitian_residual=hermitian,
        graph_norm_over_kappa2=float(np.linalg.norm(X,2)/kappa**2),SF_HALF_scalar=math.sqrt(whole_prefactor),WHOLE_detBplus=whole_prefactor,
        actual_physical_onebody_HALF_composition_error=float(np.linalg.norm(half@half-physical,2)/np.linalg.norm(physical,2)),
        stable_actual_T_J_T_residual=float(np.linalg.norm(physical@J@physical-J,2)/(1+np.linalg.norm(physical,2)**2)))


def character_log(prime: np.ndarray,base: np.ndarray,chi: np.ndarray) -> tuple[float,float]:
    g=np.repeat(np.exp(1j*chi),4)
    sign,logarithm=np.linalg.slogdet(prime.conj().T@(g[:,None]*base))
    phase=sign*np.exp(-2j*sum(chi))
    ring.require(bool(np.real(phase)>0 and abs(np.imag(phase))<1e-10),'TRUE background-covariant occupied minor remains positive; no absolute-value replacement')
    return float(logarithm+math.log(float(np.real(phase)))),float(abs(np.imag(phase)))


def native_pairs(gamma: np.ndarray,length: int,kappas: list[float]) -> dict[str,Any]:
    ring.require(length>=4 and length%2==0,'Native expansion uses a genuine even simple-edge box')
    D,coexact,harmonic,vertices,lookup=literal_geometry(length);volume=len(vertices);rng=np.random.default_rng(22220261008)
    delta=.025;first=coexact@rng.normal(size=3*volume);second=coexact@rng.normal(size=3*volume)
    first*=delta/float(max(abs(first)));second*=.8*delta/float(max(abs(second)))
    flat=np.array([1.41,-.73,2.18]);flat_increment=np.array([.019,-.013,.021])
    A=first+np.repeat(flat/length,volume);Aprime=second+np.repeat((flat+flat_increment)/length,volume)
    difference=Aprime-A;divergence=float(np.max(abs(D@difference)))
    ring.require(divergence<1e-12,'BOTH actual fields lie in the same literal Coulomb quotient with nonstationary harmonics')
    raw_chi=rng.normal(size=volume);raw_chi-=raw_chi[0];raw_chi/=float(max(abs(raw_chi)))
    chi_cases=[size*raw_chi for size in (.03,.23,.63)]
    base_action=float(np.sum(1-np.cos(difference)));rows=[]
    links=literal_links(A,length,vertices,lookup);prime_links=literal_links(Aprime,length,vertices,lookup)
    for kappa in kappas:
        occupied,base_metadata=actual_ground_frame(gamma,kappa,links)
        prime,prime_metadata=actual_ground_frame(gamma,kappa,prime_links)
        zero,_=character_log(prime,occupied,np.zeros(volume));cases=[]
        for chi in chi_cases:
            gradient=D.T@chi;cross=float(difference@gradient)
            ring.require(abs(cross)<1e-12,'Actual Coulomb quadratic cross term cancels, harmonic difference INCLUDED')
            actual,imag=character_log(prime,occupied,chi);opposite,_=character_log(prime,occupied,-chi)
            predicted=-16*kappa**4*(float(np.sum(1-np.cos(difference-gradient)))-base_action)
            predicted_opposite=-16*kappa**4*(float(np.sum(1-np.cos(difference+gradient)))-base_action)
            gaussian=-8*kappa**4*float(gradient@gradient)
            cases.append(dict(actual_chi_infinity=float(max(abs(chi))),actual_mixed_log_ratio=actual-zero,
                original_kappa4_cosine_log_ratio=predicted,Gaussian_REFERENCE_log_ratio=gaussian,
                actual_cosine_remainder_over_V_kappa6=abs(actual-zero-predicted)/(volume*kappa**6),
                actual_Coulomb_cross_term=cross,actual_mixed_evenness_log_defect=actual-opposite,
                original_cosine_evenness_log_defect=predicted-predicted_opposite,
                actual_background_phase_imaginary_error=imag))
        ring.require(max(c['actual_cosine_remainder_over_V_kappa6'] for c in cases)<2**20,
            'Actual full-component mixed Coulomb reference must fit its original kappa6 remainder, not a substituted even Gaussian')
        rows.append(dict(kappa=kappa,base=base_metadata,prime=prime_metadata,actual_zero_character_log=zero,cases=cases))
    return dict(length=length,spatial_volume=volume,physical_components=4*volume,
        actual_D_shape=D.shape,coexact_rank=int(round(np.trace(coexact))),harmonic_rank=3,
        actual_divergence_of_difference=divergence,absolute_flat_holonomies=flat,
        actual_nonstationary_harmonic_increment=flat_increment,
        actual_harmonic_edge_increment=harmonic.T@difference/math.sqrt(volume),
        actual_max_absolute_link_angle=float(max(max(abs(A)),max(abs(Aprime)))),
        actual_max_temporal_increment=float(max(abs(difference))),native_coexact_chart_radius=delta,rows=rows,
        scope='Actual full4V mean-Clifford occupied-minor comparison on literal coexact+ALL3harmonic pairs. This is not an interacting Slater, a fullPhi statistical admission, or the normalized macro photon/charged Gram.')


def selected_source_contract() -> dict[str,Any]:
    data=(ROOT/SELECTED).read_bytes();value=json.loads(data)
    ring.require(hashlib.sha256(data).hexdigest()==SELECTED_SHA,'Select ONLY the actual compiled Source2a capsule, not the differently hashed final-named alternative')
    ring.require(value['compiled_source_hashes']['covariant_prepared_gauss_state']==SOURCE_SHA,'Exact selected boundary/source strings belong to Source2a')
    macro=value['macro'];boundary=macro['admission']['exact_boundary_moment_construction']
    return dict(path=SELECTED,sha256=SELECTED_SHA,source_sha256=SOURCE_SHA,
        exact_selected_positive_density=macro['positive_density'],exact_selected_source_operator=macro['source']['exact_operator'],
        exact_selected_photon_packet_operator=macro['photon']['exact_bounded_packet_operator'],
        exact_selected_photon_Fourier_filter=macro['photon']['explicit_Curl_Fourier_filter'],
        exact_selected_boundary_moment_strings=boundary,
        actual_normalized_photon_Gram=macro['photon']['full_photon_Gram_lower_exact'],
        actual_normalized_charged_Gram=macro['source']['normalized_Gram_lower_exact'],
        actual_energy_prepared_charged_Gram=macro['source']['full_energy_prepared_source_Gram_lower'],
        actual_macro_M0=boundary['actual_macro_M0'],actual_macro_M1=boundary['actual_macro_M1'],
        actual_macro_PEGram=boundary['actual_macro_PEGram'],actual_macro_photon_normalization=boundary['actual_macro_photon_normalization'],
        macro_source_admitted=macro['admission']['macro_source_admitted'],
        honest_unpriced_fields=['Actual fullPhi/statistical physical-Gauss neutral/charged normalized kernel envelope and trace Tr J^p at Nt,Nt+1,Nt+2.',
            'Exact scalar/mean-band gauge-seam/source-string transport inside the SAME interacting Source2a conditioned operator.',
            'Actual normalized Z0,Zph,M0,M1 including charged Knull and spectrum above neutral lambda0; absolute HS distances if used.',
            'Actual same-U_N evolved density and fixed bounded outgoing receiver Gram. No macro U_N backend currently exists.'],
        guard='Neither the scalar Fourier lemma nor actual native Gram controls are promoted to these NULL macro fields; no RN-to-Loewner, weightedBKS, source commutation, Tmin or isolated-pole gate.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--length',type=int,default=4);parser.add_argument('--kappas',type=float,nargs='+',default=[.02,.01,.005]);args=parser.parse_args()
    start=time.monotonic();gamma,_=ring.supplier.phi_frame(ring.supplier.Parameters())
    pairs=native_pairs(gamma,args.length,args.kappas);macro=clock.macro_bound();selected=selected_source_contract()
    captured={name:ring.supplier.frozen.SNAPSHOT[name] for name in ring.supplier.frozen.COMPILED}
    captured['covariant_canonical_gauss_boundary_kernel']=ring.supplier.OWN_BYTES
    captured['covariant_relative_ring_boundary']=ring.OWN_BYTES;captured['covariant_auxiliary_charge_clock_bridge']=clock.OWN_BYTES
    captured[Path(__file__).stem]=OWN_BYTES
    unchanged=all((ROOT/(name+'.py')).read_bytes()==data for name,data in captured.items())
    ring.require(unchanged,'Scientific Coulomb-pair sources changed during execution')
    result=dict(plan=ring.PLAN,governing_revision=clock.PLAN_REVISION,parameters=vars(args)|{'output':str(args.output)},
        actual_native_Coulomb_pairs=pairs,actual_frozen_Source2a_macro_reference_prices=macro,
        exact_selected_Source2a_boundary_and_honest_Gram_contract=selected,
        compiled_source_hashes={name:hashlib.sha256(data).hexdigest() for name,data in captured.items()},
        captured_source_bytes={name:data.decode() for name,data in captured.items()},compiled_source_unchanged=unchanged,
        main_code_matches_captured_source=MATCH,elapsed_seconds=time.monotonic()-start,
        actual_macro_photon_Gram=None,actual_macro_charged_Gram=None,macro_source_admitted=False,
        scope='Current kappa4/all-canonical clock/scalar price continuation and actual even-box Coulomb-pair control; normalized Source2a photon/charged Grams remain honestly unpriced, not a scalar-lemma-only admission.')
    args.output.write_text(json.dumps(ring.supplier.portable(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),elapsed_seconds=result['elapsed_seconds'],
        rows=[dict(kappa=row['kappa'],maximum_remainder_over_V_kappa6=max(case['actual_cosine_remainder_over_V_kappa6'] for case in row['cases']),mixed_evenness=max(abs(case['actual_mixed_evenness_log_defect']) for case in row['cases'])) for row in pairs['rows']],
        actual_normalized_photon_Gram=None,actual_normalized_charged_Gram=None,macro_source_admitted=False)))
    print('Natural exit: actual even-box Coulomb pairs, original mixed kappa4 kernel and exact Source2a strings; macro Grams honestly unpriced')

if __name__=='__main__':main()

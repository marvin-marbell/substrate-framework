#!/usr/bin/env python3
"""Full-sector MinSet trial excess and matched SOURCE/creator window proof.

The charged filter below is ONLY a proof comparator. Physical preparation is
ONE neutral gamma*f_n*tau, followed by the original raw SOURCE and a†.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import itertools
import json
import math
import os
import sys
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from improved_phi_band import C,NSTAR,UE,selected_band
from noncompact_quantitative_normal_budget import ceil_log2
from prepared_minset_normal_proof import local_normal_constants,union_barrier_ledger,exact_portable
from prepared_guard_colour_chart import primary_source_derivative_bounds,covariant_normal_source_bounds
from prepared_neutral_full_field import l1
from prepared_neutral_local_well import exact_capsule
from sea_selection import SELECTED,h_from_couplings

IDENTITY='PNFF-MinSet-matched-full-sector-neutral-window-RAW-SOURCE-v1'


def sumlog(*values:int) -> int:
    """A power-of-two upper for a sum of positive bounded coefficients."""
    return max(values)+(len(values)-1).bit_length()


def trial_and_window_ledger(normal:dict[str,Any]) -> dict[str,Any]:
    if normal['N']!=NSTAR:raise AssertionError('Wrong selected legal N')
    f=local_normal_constants();V=NSTAR**3;d=105*V
    # These are actual regular-tube guards, not a native-node extrapolation.
    gram=min(F(1,40),f['gram_lower']/2)
    jets=primary_source_derivative_bounds(NSTAR,F(math.isqrt(V)+1),gram_min=gram)
    Jlog=ceil_log2(f['quotient_chart_jet_upper'])
    cov=covariant_normal_source_bounds(jets,Jlog)
    CF1=cov['frame_normal_log2_upper'][1]
    rep=4*V+4
    CL=(4*V+8)*8*l1(selected_band()[0]);CC=9*UE*V*NSTAR**2;beta=F(1,16)
    cl=ceil_log2(CL);cc=ceil_log2(max(CC,F(1)))
    m0=ceil_log2(jets['CB']);m1=(cov['canonical_covariant_M1_squared_log2_upper']+1)//2
    m2=cov['covariant_source_log2_upper'][2]
    # Full light/ref Slater transport Fphi: ||A_L|| <= (4V+4)*CF1.
    # Form triangle pays 2*K_field + 8*chi*d*||A_L||². chi<=1.
    light_connection=3+ceil_log2(d)+2*(ceil_log2(rep)+CF1)
    neutral=sumlog(ceil_log2(f['Eex_coefficient'])+1,light_connection,
                   cl+1,cc,ceil_log2(max(4*C*V,F(1))),
                   ceil_log2(max(3*UE/(2*NSTAR),F(1))),0)
    # Trial Ztau>=1/2 is proved below; spending the weaker1/16 is safe.
    # Source form uses FULL ambient Phi+Lie, NOT an unknown Berry replacement.
    ambient1=ceil_log2(jets['canonical_M1_squared_upper'])
    theta2=ceil_log2(jets['harmonic_gradient_squared_upper'])
    source=4+sumlog(1+2*m0+neutral,1+ambient1,
                       1+ceil_log2(UE/NSTAR)+theta2,
                       1+2*m0+sumlog(cl,cc,0))
    creator=sumlog(source,ceil_log2(max(4*C,F(1))))
    common=max(neutral,source,creator)+1
    # Exact lower tanh36 >= 1-2/E72; e72>=its positive Taylor polynomial.
    E72=sum((F(72**j,math.factorial(j)) for j in range(257)),F())
    word=3*NSTAR+2;cword=1-F(2*word,E72)
    if cword<F(3,4):raise AssertionError('Original full legal word lower failed')
    # Point fullRef/chiral raw beta norm²>=4/5; sqrt>=4/5.
    # Off-centre comparison-module variation uses actual Gaussian Wick bound.
    trial_variation=m1+(ceil_log2(f['gaussian_covariance_majorant'])+1)//2+3
    # Neutral/SOURCE matched Duhamel on tau and separately on EXACT a†tau.
    # E_a <= Ccommon Lambda^3/8, Kphi/harmonic bounded by that same floor.
    # Cf_1<=2^2048. Coefficients below include all summands and harmless factors.
    rootE=(common+1)//2
    source_terms=[dict(log=2048+m1+rootE-common+8,exponent=F(-7,16)),
                  dict(log=2048+m2-common+8,exponent=F(-7,8)),
                  dict(log=2048+ceil_log2(UE/NSTAR)//2+m0+rootE-common+10,exponent=F(-3,16)),
                  dict(log=2048+sumlog(cl,cc,0)+m0-common+8,exponent=F(-3,8))]
    omega_min=4*C/NSTAR
    current_leg=cl+(ceil_log2(16*UE/omega_min)+1)//2
    # [H,a†] = omega*a†+Jg. a†Un tau is UNBOUNDED but its norm
    # is <=sqrt(1+E_tau/omega_min), not1. This is a real creator graph.
    photon_terms=[dict(log=2048+m0+ceil_log2(4*C)+rootE+
                       (ceil_log2(1/omega_min)+1)//2-common+8,exponent=F(-3,16)),
                  dict(log=2048+m0+current_leg-common+8,exponent=F(-3,8))]
    epsilon=UE/(V*2**40);targetbits=ceil_log2(1/epsilon)+16
    onset=max(f['required_log2_lambda'],(8*(common+8)+4)//5,
              (8*(trial_variation+6)+2)//3)
    for term in source_terms+photon_terms:
        onset=max(onset,math.ceil(F(term['log']+targetbits+8)/(-term['exponent'])))
    selected=2**315
    if onset>=selected:raise AssertionError('Unchanged Lambda does not pay full-sector window')
    # Actual binding of the operator IMS/Agmon input, not its former defaults.
    union=union_barrier_ledger(exact_portable(f),F(6),common,m0,m1,F(1,4),targetbits)
    return dict(identity=IDENTITY,neutral_trial='FIXED-m CONDITIONAL fullGauss correlated90V field/whole-heavy Kato + actual centre light negative2V sea + Ref4; PH Ref0 belongs historical dual-family Gauss control, NOT same-fixed-reference physical coherent bank; coexactVAC/freeharmonicGaussian/actual MinSet ties',
        actualH='UNCHANGED full microscopic LAW05; all512 Root histories, beta1/16, all coexact and free harmonics retained',
        common_floor='e_m=Lambda min_s v(m,s)-CL-beta; SAME worst CL for neutral and charged; E_ph_vac paid ONCE inside neutral trial excess',
        CL_upper=CL,C_C_upper=CC,light_frame_connection_log2_upper=light_connection,
        neutral_Eex_coefficient_log2_upper=neutral,raw_SOURCE_trial_Eex_coefficient_log2_upper=source,
        true_creator_trial_Eex_coefficient_log2_upper=creator,actual_common_coefficient_log2_upper=common,
        neutral_window='FIXED-m L=4*2^actual_common_coefficient_log2_upper*Lambda^(3/8); branch floor e_m, NOT coherent64 global preparation',
        Ecut='e_m+2L; 8Ccommon Lambda^(3/8)<=V Lambda; Ecut<5V Lambda<unchanged Root Utrial',
        trial_word_lower=cword,raw_point_squared_coefficient_lower=F(4,5),
        trial_variation_log2_upper=trial_variation,trial_variation_exponent=F(-3,8),
        Ztau_lower=F(1,2),neutral_filter_squared_norm_lower=F(3,4),
        raw_filtered_SOURCE_norm_lower=F(1,4),raw_true_creator_norm_lower=F(1,4),
        actual_source_terms=source_terms,actual_creator_photon_terms=photon_terms,
        creator_identity='a†Bfn tau-fqBa†tau=B[a†,fn]tau+(Bfn-fqB)a†tau; direct SOURCE intertwining on exact onephoton trial, NEVER a† times epsilon_n',
        creator_energy_graph='H_D-e>=Hph>=omega_min*N; ||a†Un tau||<=sqrt(1+E_tau/omega_min); actual onephoton tau energy paid +4c',
        photon_omega_min=omega_min,current_creator_leg_log2_upper=current_leg,
        charged_filter_is_PROOF_comparator_ONLY=True,physical_input='FIXED-m conditional a†B gamma fn tau / actual branch norm; no charged filter/raw-quartet whitening. Physical coherent bank successor is prepared_neutral_global_filter',
        full_source_jets_raw_log2_upper=[ceil_log2(v) for v in jets['source_operator_mixed_derivative_upper']],
        full_ambient_M1_squared_log2_upper=ambient1,full_covariant_M1_squared_log2_upper=cov['canonical_covariant_M1_squared_log2_upper'],
        actual_uniform_guard=jets['guards'],required_log2_Lambda=onset,unchanged_selected_log2_Lambda_lower=selected,
        selected_tower_pays_FIXED_m_trial_filter_creator_and_Root_window=True,union_boundary=union,
        coherent64_admitted=False,coherent64_global_three_quarters_claim_withdrawn=True,
        admission_scope='FIXED-m CONDITIONAL branch-local e_m trial/SOURCE/creator estimates ONLY. Old coherent64 global3/4/ordinaryadmission WITHDRAWN. Old Ref0-only minus is historical dual-Gauss family, not new same-fixedGauss physical branch',
        remaining=['ONE physical global-floor fn and actual coherentbank/source norms belong prepared_neutral_global_filter','Born/physical drivenNoether','fundamental quantum law/parameter debt'])


def native_full_ref_chiral_gram(field_receipt:Path) -> dict[str,Any]:
    """NEW N5 full one-particle SEA construction, not the old8-active NPZ."""
    old=json.loads(field_receipt.read_bytes())
    G=np.asarray(old['centre_probe'][0]['actual_full10_gram']['G'],float)
    vals,U=np.linalg.eigh(G);frame=np.einsum('am,mij->aij',(U*np.sqrt(vals))@U.T,SELECTED)
    size=5;sites=np.array(list(itertools.product(range(size),repeat=3)))
    k=2*np.pi*sites/size;V=len(sites);band=selected_band()[0]
    eta=np.exp(1j*(.37*sites[:,0]-.21*sites[:,1]+.13*sites[:,2]))*(1+.11*sites[:,2]);eta/=np.linalg.norm(eta)
    ref=np.array([0,2,2]);ell=np.sum(abs(sites-ref),axis=1);c=math.tanh(36.)
    profile=eta*c**ell*np.exp(-1j*((sites-ref)@np.array([.21,-.13,.09]))/size)
    labels=np.stack([np.eye(4),SELECTED[2]@SELECTED[3],SELECTED[3]@SELECTED[1],SELECTED[1]@SELECTED[2]])/2
    beta=np.array([1,.3j,-.2+.4j,.6]);chiral=SELECTED[0]@SELECTED[1]@SELECTED[2]@SELECTED[3]
    worst=0.;anti=0.;gap=math.inf;sample=None
    expected=.5*np.vdot(profile,profile).real*np.eye(4)
    for m in itertools.product(range(4),repeat=3):
        for s in itertools.product(range(2),repeat=3):
            theta=np.pi*(np.asarray(m)/2+np.asarray(s));momenta=k+theta/size
            H=h_from_couplings(band,momenta,frame);evals,evecs=np.linalg.eigh(H)
            if np.any(np.sum(evals>0,axis=1)!=2):raise AssertionError('Point trial lacks2V occupied negative modes')
            P=np.einsum('kir,kjr->kij',evecs[:,:,2:],evecs[:,:,2:].conj())
            Fourier=np.exp(-1j*(momenta@sites.T))/math.sqrt(V);weights=abs(Fourier@profile)**2
            gram=np.einsum('k,air,kij,bjr->ab',weights,labels.conj(),P,labels)
            error=float(np.linalg.norm(gram-expected));worst=max(worst,error)
            anti=max(anti,float(np.linalg.norm(H@chiral+chiral@H)))
            gap=min(gap,float(np.min(abs(evals))))
            if sample is None:sample=float(np.vdot(beta,gram@beta).real)
    if worst>1e-12 or anti>1e-12:raise AssertionError('Actual fullRef/chiral trial Gram failed')
    return dict(size=size,oneparticle_dimension=4*V,occupied_negative_modes=2*V,reference_modes=4,phase_histories=512,
        frozen_actual_Gram=G.tolist(),frozen_Gram_scope='one original N5 exact Gram tested acrossall512 LIGHT boundary phases; not other512 heavy argmin simulations',
        raw_SOURCE_squared_norm=sample,raw_beta_unnormalized=True,mean_Gram_defect=worst,chiral_defect=anti,negative_cluster_gap=gap,
        wrong_Ref_average_squared_norm=cast(float,sample)/4,unbounded_creator_on_n20_norm=math.sqrt(21),
        scope='NEW full500 oneparticle/quasifree trial with actual negative sea/fullRef, not old8active arbitraryRDM or full90V quantum native state')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--normal-receipt',type=Path,required=True);p.add_argument('--field-receipt',type=Path,required=True)
    p.add_argument('--guard-capsule',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent;guard=json.loads(a.guard_capsule.read_bytes())
    guard_hash=hashlib.sha256((root/'prepared_guard_colour_chart.py').read_bytes()).hexdigest()
    if guard_hash!=guard['source']['sha256']:raise AssertionError('Corrected q6 helper bytes differ from supplied capsule')
    names=('prepared_neutral_minset_sectors.py','prepared_neutral_source_moments.py','prepared_minset_normal_proof.py','prepared_guard_colour_chart.py','prepared_neutral_canonical_source.py')
    hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    loaded={Path(cast(str,m.__file__)).resolve() for m in sys.modules.values()
            if getattr(m,'__file__',None) and Path(cast(str,m.__file__)).suffix=='.py'
            and Path(cast(str,m.__file__)).resolve().parent==root}
    loaded_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in loaded}
    inputs={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.normal_receipt,a.field_receipt,a.guard_capsule)}
    result:dict[str,Any]=dict(source_sha256=hashes,inputs=inputs,ledger=trial_and_window_ledger(json.loads(a.normal_receipt.read_bytes())),native=native_full_ref_chiral_gram(a.field_receipt))
    result['loaded_scientific_source_sha256']=loaded_hashes
    result['loaded_sources_unchanged']=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in loaded_hashes.items())
    result['source_unchanged']=all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    result['positive_AST']=[n.name for n in ast.parse((root/names[0]).read_text()).body if isinstance(n,ast.FunctionDef)]
    print(json.dumps(exact_capsule(result),sort_keys=True,indent=2))

if __name__=='__main__':main()

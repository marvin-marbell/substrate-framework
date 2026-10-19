#!/usr/bin/env python3
"""ONE physical global filter, actual Riccati RAW trial and retained Root weights.

Generic-chart alignment cubic may grow Lambda^(1/16); it is retained, not
silently replaced by the stronger flat-adapted constant-residual premise.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from fractions import Fraction as F
from pathlib import Path
import sys
from typing import Any, cast
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.linalg import expm
from numpy.polynomial.hermite import hermgauss
from prepared_minset_normal_proof import union_barrier_ledger,hex_fraction_log2_upper
from noncompact_quantitative_normal_budget import ceil_log2
from prepared_neutral_local_well import window_values,exact_capsule
from prepared_neutral_minset_sectors import sumlog
from local_orbital_stability import scalar_paths
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE
from sea_selection import SELECTED
from improved_phi_band import NSTAR,UE,C,selected_band
from prepared_neutral_coherent_bank import IDENTITY,coherent_bank_descriptor,fourier_window_majorant
from prepared_neutral_full_field import l1

PLAN='issue-222-fullfield-closedrotation-continuation-20261006'
PLAN_REVISION='7954ba4c18ed7d90fdc3e6cbd3bf75c3dc80d568cae4781d73dadd61e041a718'


def global_shadow_weights(branch_potential:np.ndarray,zero_point:np.ndarray,
                          initial:np.ndarray,light_energy:np.ndarray,lam:float,
                          floor:float,width:float,photon_omega:float=0.,time:float=0.) -> dict[str,Any]:
    """Actual D including bank energies/phases; ONE common floor/norm."""
    D=lam*np.asarray(branch_potential)+np.asarray(zero_point)+np.asarray(light_energy)+photon_omega
    raw=np.asarray(initial,complex)*window_values(D,floor,width)
    norm=float(np.linalg.norm(raw))
    if norm==0:raise ValueError('Global window annihilated every initial branch')
    return dict(raw=raw,normalized=raw/norm,norm=norm,D=D,
                ordinary_amplitudes_at_time=raw/norm*np.exp(-1j*time*D),
                ordinary_phase_law='scalar exp(-i t D_ms,b) factor; actual normal remainder operator MUST also evolve, neverT6 stationaryRiccati propagation',
                required_normal_evolution='exp[-it(Hnormal_ms-Dheavy_ms)] with scalarfactor; actualcommonfieldwave retained/cancels onlymatched SOURCE/current contractions')


def coherent_bank_shadow(branch_potential:np.ndarray,zero_point:np.ndarray,
                         negative_light_energy:np.ndarray,extra_four_energy:np.ndarray,
                         minset:np.ndarray,lam:float,floor:float,width:float,
                         photon_omega:float=0.,time:float=0.) -> dict[str,Any]:
    """Actual64x8x2 seed/filter weights; finite numeric counterpart of operator law."""
    arrays=(branch_potential,zero_point,negative_light_energy,extra_four_energy,minset)
    if any(np.shape(value)!=(64,8) for value in arrays):
        raise ValueError('Actual ALL512 branch arrays/MinSet mask required')
    ties=np.asarray(minset,bool);count=np.sum(ties,axis=1)
    if np.any(count==0):raise ValueError('Every rawm must have an actual MinSet')
    initial=ties[:,:,None]/(8*np.sqrt(2*count[:,None,None]))*np.ones((1,1,2))
    light=np.asarray(negative_light_energy)[:,:,None]+np.stack(
        (np.zeros_like(extra_four_energy),np.asarray(extra_four_energy)),axis=-1)
    result=global_shadow_weights(np.asarray(branch_potential)[:,:,None],
        np.asarray(zero_point)[:,:,None],initial,light,lam,floor,width,photon_omega,time)
    result['identity']=IDENTITY;result['actual_MinSet_mask']=ties
    result['law_scope']='only numerical finite controls; legal huge Lambda ordinary phases remain EXACT symbolic exp(-itD), notfloat-overflow/reset'
    return result


def global_filter_ledger(sector:dict[str,Any],f:dict[str,Any]) -> dict[str,Any]:
    V=NSTAR**3;d=90*V
    if sector['actual_common_coefficient_log2_upper']!=1787662:
        raise ValueError('Exact frozen common-L coefficient binding required')
    common=sector['actual_common_coefficient_log2_upper']
    J=hex_fraction_log2_upper(f['quotient_chart_jet_upper']);cmp=hex_fraction_log2_upper(f['finite_chart_compare'])
    M=hex_fraction_log2_upper(f['gaussian_covariance_majorant']);r=hex_fraction_log2_upper(f['quotient_radius'])
    dim=ceil_log2(d);rep=ceil_log2(4*V+8)
    CL=(4*V+8)*8*l1(selected_band()[0])
    # All-root Pauli coefficient l1, NOT an unproved fullQ ordinaryCAR norm.
    # Moment-majorants include mass-normal inverse and eighth Wick/order terms.
    p0=10+dim+cmp+J
    wick8=16+4*dim+4*max(M,p0)
    frame2=rep+4*J+16
    field=cmp+wick8+4*J+32
    berry=cmp+frame2+p0+dim+16
    bounded=sumlog(ceil_log2(CL)+J+4,ceil_log2(max(F(1),UE*NSTAR**2*F(4*V+32,16))),
                   ceil_log2(max(F(1),32*UE/NSTAR)),ceil_log2(max(F(1),4*C*V)),0)
    # sqrt512 and radial-cutoff normalization included. No positive Lambda
    # power is hidden in this coefficient: the explicit +1/16 is separate.
    CN=sumlog(field,berry,bounded)+8
    source_CN=max(CN+153,445470+p0+dim+16,890487+16)+16
    photon_CN=max(source_CN+4,ceil_log2(CL)+J+153+16)+8
    weighted_CN=photon_CN+4096 # exact fixed-sigma Gaussian theta/polynomial moments through7.
    rows=[dict(term='heavy/local cubic',power=F(-1,8)),
          dict(term='alignment cubic: two soft, ONE stiff (generic chart)',power=F(1,16)),
          dict(term='alignment cubic: one soft, two stiff',power=F(0)),
          dict(term='alignment/plaquette fourth remainder',power=F(-1,4)),
          dict(term='linear-soft cometric times stiff momentum squared',power=F(0)),
          dict(term='quadratic cometric remainder',power=F(-3,8)),
          dict(term='heavy/Light/Ref Kato-Berry first derivative',power=F(-1,16)),
          dict(term='half-density, light, HC, harmonic, beta',power=F(0))]
    if max(row['power'] for row in rows)!=F(1,16):raise AssertionError('Lost growing alignment branch')
    # The exact absolute ZP bound comes from the actual TRUE quadratic Hessian.
    zp=hex_fraction_log2_upper(f['zero_point_upper'])
    if zp+6>=common:raise AssertionError('Existing common L fails exact ZP plateau')
    CB=ceil_log2(16*F(33,32)*(math.isqrt(V)+1)) # homogeneous fullQ bound; NEVER3.
    target=ceil_log2(V*2**40/UE);wanted=target+CB+24
    # Quasimode time argument t/L gives Lambda^(1/16-3/8)=-5/16.
    error_constant=max(CN,source_CN,photon_CN,weighted_CN)+2048-common+14
    selector=math.ceil(F(error_constant+wanted+8,F(5,16)))
    # Actual unchanged radial cutoff:1 inside r/3, support<=2r/3.
    # Interior gamma radius r0 Lambda^-1/4, r0=r/128; Gaussian tail.
    rlower=r-1;r0lower=rlower-7;action=2*r0lower-M-9
    gamma_selector=4*(math.ceil(math.log2(wanted+CN+64))-action+4)
    cutoff_selector=8*(CN+2*(-rlower)+M+64)
    selected=2**315
    required=max(f['required_log2_lambda'],selector,gamma_selector,cutoff_selector)
    if required>=selected:raise AssertionError('Unchanged Lambda fails actual global-filter selector')
    # ONE global floor improves, never weakens, each local union lower form.
    union=union_barrier_ledger(f,F(6),common,153,445470,F(1,128),target+26)
    return dict(identity=IDENTITY,actual_common_coefficient_log2_upper=common,
        CL_upper=CL,full_source_jets_raw_log2_upper=[CB,289,426,564,701,839,976],
        actual_global_floor='e_global=Lambda min_(m,s) v_ms - CLcommon - beta',
        actual_selected_window='L=4*2^1787662*Lambda^(3/8), EXACT physical definition unchanged from supplied sector window; coefficient upper bound is used to CHOOSE this power-of-two coefficient',
        actual_selected_window_coefficient_log2=1787664,
        actual_window_lower='L>=2^1787662 Lambda^(3/8)>=Lambda^(3/8), by DEFINITION not upper-to-lower inference',
        legal_Hamiltonian_energy_origin=dict(common_identity_shift=0,authority='sealed unshifted LAW05/Root106 H_R',filter_floor_is_not_H_subtraction=True),
        physical_preparation='Xi=zeroextend(gamma fn((H_D-e_global)/L) RAWtau)/ONE actual global norm',
        RAWtau='64 coefficients1/8; actual MinSet tie blocks normalized; whole-heavy Kato/Riccati, centre fullnegative2V Slater with actual FnPhi transport; coherent(SeaRef4+SeaPlus4positiveBlochRef0)/sqrt2 on SAME fixedqRef=Nref-4, coexactVAC/fixedsigmaTheta1/exactRootP',
        quadratic_only='TRUE original heavy/scalar/colour full90V Hessian incl SEA+CONTACT and Lambda^5/4 plaquettes; massnorm chi A^2=K; not entire-H ground',
        raw_cutoff='1 inside r/3, zero outside2r/3, Haar/half-density retained',
        branch_D='Lambda v_ms + .5 Tr sqrt(chi K90_ms) + actual fullnegative light energy + bank-specific actual4positive spectator energy; SAME static reference Hamiltonian, no reset',
        photon_zero_point='absolute sum omega/2 counted ONCE in fixed CN remainder (<=4cV), NOT also in D_ms',
        shadow_weights='a_ms,b=(1/(8sqrt(2*tiecount_m)))*f((D_ms,b-e_global)/L); no equal64/bank replacement; ONE normalization',
        photon_shadow_weights='same formula with actual D_ms,b+omega_g; NEVER assume f=1 for all classes or banks',
        ordinary_phases='exp(-i t D_ms,b), exp(-i t(D_ms,b+omega_g)) kept; no physical m/s superselection',
        residual_rows=rows,residual_log2_coefficient_upper=CN,residual_Lambda_power=F(1,16),
        neutral_filter_error_Lambda_power=F(-5,16),normalized_error_log2_coefficient_upper=error_constant,
        source_quasimode_coefficient_log2_upper=source_CN,true_creator_quasimode_coefficient_log2_upper=photon_CN,
        source_quasimode='R_B=B R_tau+[H,B]tau: massivefield/wholeheavy/code commute in accepted physicalRootP SOURCE; kineticchi*M1*pstiff Lambda^-1/16, chi*M2 Lambda^-1/2, light/HC/harm/beta fixed; conservativeCBQ retained',
        charged_window='fq is PROOF ONLY: Bfn tau and fq Btau approximate SAMEexactD-weighted RAWsource; creator uses initiala†tau/a†Btau and exactD+omega, nevera†eta',
        weighted_theta_Pi_quasimode_coefficient_log2_upper=weighted_CN,
        weighted_theta_Pi='applydirectquasimode ALSOto theta^r ptheta^s tau, theta^r ptheta^s Btau andtrueinitiala†variants, r+s<=7; exactGaussianmoments/Weylpullthrough; no unboundedoperatoroneta',
        covariance_log2_upper=M,momentum_majorant_log2_upper=p0,eighth_Wick_majorant_log2_upper=wick8,
        zero_point_log2_coefficient_upper=zp,global_minimum_m_initial_norm_lower=F(1,8),
        neutral_preparation_norm_lower=F(1,32),neutral_inverse_norm_upper=32,
        source_total_unfiltered_norm_lower=F(1,128),source_relative_to_actual_Xi_norm_lower=F(1,4),
        SOURCE19_shadow_raw_relative_lower_squared=F(49,128),
        SOFT_shadow_raw_absolute_relative_lower_squared=F(11749,20480)**2/2,
        raw_bank_source_conservative_lower=F(3,8),
        normalized_source_comparison='64 CB_eta delta_neutral plus Cf1 Eextra/L bank imbalance; same actual fn weights, positive anti rank4 Gram; actual each source lower1/4 after selected selector',
        true_creator='a†BXi: canonical PHOTON CCR gives norm>=||BXi||; not norm1. Approximation uses direct quasimode on initial a†tau plus [a†,fn], true allmodeWeyl force; actualD+omega retained',
        source_CB_full_auxQ_log2_upper=CB,neutral_fn_cloud_extra_squared_log2_bits=10,
        photon_force='allmode path displacement squared<=16UE/c, no1/omega_min for physical moments; Phi/light regular tube l1 bound CL',
        same_common_L_not_retuned=True,required_log2_lambda=required,selected_log2_lambda_lower=selected,
        selected_error_target='delta_neutral, delta_trueCreator <= (UE/(V2^40))*2^(-CB_eta_log-24)',
        global_union=union,
        coherent_bank=coherent_bank_descriptor(),Fourier_window=fourier_window_majorant(),
        bank_balance='orthogonal bank rawcoeff equal1/sqrt2; fullnegative+fourpositive energies differ<=CL. Uniform f Lipschitz<=Cf1 gives ||a0-a1||<=Cf1 CL/(sqrt2 L); ONE total norm>=1/32; banknorm>=1/sqrt2 minus32 Cf1 CL/L. No physical reweighting',
        bank_balance_log2_coefficient_upper=2048+ceil_log2(CL)+6-common,
        bank_balance_Lambda_power=F(-3,8),
        actual_prepared_bank_probability_interval=[F(49,100),F(51,100)],
        actual_bank_weight_proof='||a0-a1||<=Cf1 CL/(sqrt2 L); |p0-1/2|<=16Cf1 CL/L beforephysicalquasimode error. BoundedP_ref measurement adds<=2 normalizedneutralerror, gamma paid. Requiredonset belowselectedtower; Nref invariantordinary/Noe afterprep off',
        bank_weight_required_log2_lambda=max(0,math.ceil(F(2048+ceil_log2(CL)+16-common,F(3,8))),math.ceil(F(error_constant+12,F(5,16)))),
        failed_scope='7ba/f980 global3/4/admissionWITHDRAWN; fixedm conditional pieces only. CN0 also NOTassumed in generic chart.',
        remaining=['actual onceGLOBAL meanCAR/light/ph/harm/HC Born coefficient ownerjoin','encoded SOURCE/globaloddCAR and density/beta in fullauxQ','physical drivenSO3/highergraphs','fundamental quantum law/parameters'])


def light_normal_shadow_ledger(state:dict[str,Any],f:dict[str,Any]) -> dict[str,Any]:
    """Retain the real normal/right dynamics; remove only paid light coupling."""
    V=NSTAR**3;d=90*V
    common=state['actual_common_coefficient_log2_upper'];J=hex_fraction_log2_upper(f['quotient_chart_jet_upper'])
    dim=ceil_log2(d);rep=ceil_log2(4*V+8);cmp=hex_fraction_log2_upper(f['finite_chart_compare'])
    kappa=state['global_union']['kappa_log2_lower']
    rootE=(common+5)//2
    CL=state['CL_upper']
    DL=ceil_log2(CL)+dim+4*J+12
    AL=rep+2*J+(dim+1)//2+12
    terms=[dict(term='fullPauli light Phi/U potential difference',log=DL+rootE+max(0,math.ceil(-kappa/2))+12,power=F(-5,16)),
           dict(term='actual light/Ref frame connection cross full normal kinetic',log=AL+rootE+12,power=F(-1,16)),
           dict(term='frame derivative/square/heavy-connection bounded remainder',log=cmp+2*AL+dim+16,power=F(-1,2)),
           dict(term='source meanmark field difference',log=445470+rootE+max(0,math.ceil(-kappa/2))+12,power=F(-5,16))]
    target=ceil_log2(V*2**40/UE)+state['source_CB_full_auxQ_log2_upper']+32
    onset=max(math.ceil(F(row['log']+target+32)/(-row['power'])) for row in terms)
    if onset>=2**315:raise AssertionError('Unchanged tower fails actual light/normal shadow')
    return dict(identity=IDENTITY,terms=terms,required_log2_lambda=onset,selected_log2_lambda_lower=2**315,
        graph='ONLYactualPglob[H_D-e_global<=2L]: localcoercivity gives ||q Pglob||<=sqrt(2L/kappa/Lambda); fullkinetic gives ||sqrtchi Pi Pglob||<=sqrt(2L)',
        ordinary_Duhamel='UseactualU_HD lowcomponent on right and shadowunitarity on left; physicalsource/truecreator remainder paidinNORMbeforeDuhamel, neverunboundedoperatoroneta',
        shadow='frame-conjugated centre meanCAR + exactALLmodeEM/freeharm/HC/Root + SAMEactualnormal-heavy operator; doNOTapproximate fullneutralnormalwavebyD_ms atT6',
        massive_cancellation='potential/wholeheavycode/sourcecomm cancelonlyinsource-relativeLEFT/neutralRIGHT comparison; no neutral-Sigma return gate',
        all_roots='all actual512weights/ordinaryphases retained; Root betaP=0/Q beta/G kept with conservativeCBQ, not CQ3',
        source_harmonic='theta=N Ah, Hh=UE/N ptheta² and exp(-i theta.(Birth-Ref)/N) retained, order1D/N derivative',
        residual_not_physical_freeze='quasimode D_ms is ONLYshortfilter-time comparator, not an Hreplacement or T6 stationaryPhi hypothesis',
        remaining='Transport owns onceGLOBAL fixedP0/noncommharmonic/HC structured leading coefficient; encodedRoot SOURCE/current bindings stillindependent')


def initial_energy_ledger(state:dict[str,Any],f:dict[str,Any]) -> dict[str,Any]:
    """First quadratic-form energies of actual Xi, BXi and true a†BXi."""
    V=NSTAR**3;CB=state['source_CB_full_auxQ_log2_upper']
    common=state['actual_common_coefficient_log2_upper']
    cmp=hex_fraction_log2_upper(f['finite_chart_compare']);dim=ceil_log2(90*V)
    grad=state['global_union']['cutoff_gradient_coefficient_log2']
    gamma_kin=2*grad+cmp+dim+14
    neutral:list[dict[str,Any]]=[dict(term='ONEglobal spectral cap:32²*2L',log=common+13,power=F(3,8)),
             dict(term='scalar gamma IMS:chi*(Lambda^1/4)^2',log=gamma_kin,power=F(0))]
    ratio=2*CB+5 # 2 CB² / Zsource², Zsource>=1/4.
    source:list[dict[str,Any]]=[dict(term='positive Dirichlet massive/kinetic/Root forms through B',log=row['log']+ratio,power=row['power']) for row in neutral]
    source.extend([dict(term='Phi/colour SOURCE derivative kinetic form',log=cmp+dim+2*445470+12,power=F(0)),
                   dict(term='bounded light and actual beta commutator',log=ceil_log2(state['CL_upper'])+2*CB+8,power=F(0)),
                   dict(term='charged Coulomb/actual harmonic derivative',log=max(0,ceil_log2(UE*NSTAR**2*F(4*V+32,16)),ceil_log2(UE/NSTAR))+2*CB+16,power=F(0))])
    creator=[dict(term='true creator form plus allmode current pullthrough',log=row['log']+2*CB+32,power=row['power']) for row in source]
    onset=max(math.ceil(F(row['log']-ceil_log2(V)+16)/(1-row['power'])) for row in neutral+source+creator)
    if onset>=2**315:raise AssertionError('Actual source/creator initial first energy exceeds8V Lambda')
    photon_zero=F(7,2)*C*(V-1);creator_photon=16*C
    source_harmonic=64*UE/NSTAR
    # Fullsource coexact derivative is exactly zero: Green*div Pcoex=0.
    # Initial photon cloud is proved by fn pullthrough on true raw VAC,
    # including second factorial moment needed AFTER true creator.
    photon_margin=4*C*V+3*UE/NSTAR-photon_zero-creator_photon-source_harmonic
    if photon_margin<=C:raise AssertionError('Initial exact EM zero point/creator margin failed')
    return dict(identity=IDENTITY,neutral_relative_energy_rows=neutral,
        source_relative_energy_rows=source,true_creator_relative_energy_rows=creator,
        required_log2_lambda=onset,selected_log2_lambda_lower=2**315,
        absolute_initial_energy_upper='8V Lambda for actual normalized Xi, each BplusXi/BminusXi, pairedSOURCE and truea† variants',
        absolute_floor='e_global<=0: classical zeroPhi trial has v=0, global min<=0; actual ordinary negative floor NOT removed from Hamiltonian/phases',
        norm_inputs=dict(actual_neutral=F(1,32),each_SOURCE_relative=F(1,4),true_creator_relative=F(1,4)),
        form_proof='fn has actualH_D spectral support0<=H_D-e_global<=2L; scalar gamma IMS adds chi|gradgamma|². Onactualregularunion massive potential minusglobalclassicalfloor nonnegative; positivekinetic/Root forms throughB bounded by2CB² timesneutralform plusSOURCEderivative squares. Lightandactualbeta bounded, HC positive/charged bound explicit. Never infer source energy from norm-close shadow',
        true_creator_form_proof='h=H_D-e_global>=0, x=f(h/L)tau, a tau=0. Exact [a,h]=omega a+F, ||F_tensor||<=CLd and ||[a,F]||<=CLd². Fourier pullthrough for f and u*f gives ||a x||<=Cf1 CLd/L, ||a h x||<=Cf1(uf) CLd. Then <a†x,h a†x> is <x,hx> plus <ax,a hx>, omega||a†x||² and <x,aF†x>; bounded by2L+4c+fixedCf/CL/d terms. gamma commutesa and adds SAMEIMS; B commutesa, so source form lemma applies to this actual neutralcreator, with CCR/source lower and extra conservative bits. No unbounded-energy norm transfer',
        Root_requirement='PAID by exact21922587 original even SOURCE/cycle binding: all106 projector penalties commute BOTH literal maps/allRef/fullP+Q/all512; actual beta NONcommuting retained and CQ16sqrtV retained. Initial H_R energy uses SAME unshiftedC_identity=0 law',
        EM_initial_energy_upper=4*C*V+2*UE*NSTAR**2*(V+2)+3*UE/NSTAR,
        EM_zero_point_upper=photon_zero,EM_true_creator_excitation_upper=creator_photon,
        actual_SOURCE_harmonic_energy_upper=source_harmonic,
        Coulomb_operator_upper=UE*NSTAR**2*F(4*V+32,16),
        Coulomb_proof='each bulk site(n_x-2)^2<=4; fixedqRef=Nref-4 in[-4,0] addsatmost32 atFAR; sumrho=0 BOTHbanks/chargedstates, Green_norm<=N²/16. Before square/projection',
        coexact_SOURCE_commutator='[a_m,B±]=0 exactly for literal Dirac Coulomb phase: div Pcoex=0; SU4 field/colour word independent ofcoexactU1 coordinates. Macrotheta phase retained',
        allmode_initial_graph='J1,J2 ofactual fnXi via trueWeyl Fourier pullthrough <=inverse32 Cf CL d/L and higherTouchard; SOURCE ratioCB/Z; a†Gamma uses J1²+J2², notnorm-transfer. Selectedtower makes sourcecloud<c and creatorN<=4',
        actual_harmonic_graph='weighted samefn/fq quasimodes onptheta*tau andtruecreator; norm<=8, hence64UE/N. Unused exact7c(V-1)/2 zero-point margin pays extra61UE/N in stated4cV+3UE/N total',
        initial_only=True,not_claimed='driven graph/work/Noether/source-relative/spatialLoop stillB; Born/current commonreceiver stillTransport')


def ordinary_weighted_OUT_ledger(state:dict[str,Any],f:dict[str,Any]) -> dict[str,Any]:
    """Actual a† B U(6)Xi transfer: regular N² graph and whole Agmon separately."""
    V=NSTAR**3;J=hex_fraction_log2_upper(f['quotient_chart_jet_upper'])
    radius=hex_fraction_log2_upper(f['quotient_radius'])
    if J+radius+8>=0:raise ValueError('Regular raw fields cannot be bounded by PhiMax4')
    common=state['actual_common_coefficient_log2_upper']
    dim=ceil_log2(105*V);cmp=hex_fraction_log2_upper(f['finite_chart_compare'])
    CB=state['source_CB_full_auxQ_log2_upper'];CLlog=ceil_log2(state['CL_upper'])
    dglog=math.ceil(F(ceil_log2(64*UE/C),2))
    # Regular exact Weyl hierarchy; no photon-energy/omega_min used here.
    cloud_log=2048+CLlog+32+CB
    cloud_onset=math.ceil(F(cloud_log+1024,F(3,8)))
    if CLlog+dglog+6>=-1024:raise ValueError('Ordinary6 Touchard graph needs explicit repricing')
    lapW=ceil_log2(2**32*V**2*(1+l1(RotationBand(**BARE))))+2*J+16
    # POINTWISE positive W includes actual Root penalties and beta+betaI.
    # RootDelta/beta are coordinate-constant: their Laplacian is ZERO.
    elliptic=cmp+4*J+dim+32
    kinetic_log=max(common+4,(lapW+1)//2)+elliptic+16
    rootE=(common+5)//2;kappa=state['global_union']['kappa_log2_lower']
    qlog=rootE+max(0,math.ceil(-kappa/2))+16
    rows:list[dict[str,Any]]=[dict(term='q sqrt(N+1) actual normal endpoint',log=(qlog+8)//2+16,power=F(-5,32)),
          dict(term='chi Pi sqrt(N+1) actual normal endpoint',log=(kinetic_log+8)//2+16,power=F(-1,16))]
    target=ceil_log2(V*2**40/UE)+CB+64
    weighted_terms:list[dict[str,Any]]=[dict(term='weighted fullPauli light potential / actual q',log=rows[0]['log']+CLlog+4*J+dim+CB+16,power=F(-5,32)),
                    dict(term='weighted actual light/Ref Berry kinetic cross',log=rows[1]['log']+4*J+dim+CB+32,power=F(-1,16)),
                    dict(term='bounded Berry derivative and square',log=cmp+8*J+dim+CB+32,power=F(-1,2))]
    onset=max(cloud_onset,*[math.ceil(F(row['log']+target+32)/(-row['power'])) for row in weighted_terms])
    if onset>=2**315:raise AssertionError('Unchanged Lambda does not pay weighted true OUT')
    return dict(identity=IDENTITY,duration=6,physical_Phi_operator_upper=4,
        Phi_bound='centre gram_upper=1/2 gives each Phi norm<=sqrt2; raw quotient variation<=J*r<1, so operator cap4 on actual entire regular union',
        physical_window_lower='L>=Lambda^(3/8) by fixed physical L=4*2^1787662*Lambda^(3/8)',
        initial_annihilator_tensor_bounds_neutral_and_normalized_SOURCE=[1,*[F(1,2**512)]*5],
        initial_annihilator_tensor_bounds_normalized_true_creator=[1,2,*[F(1,2**500)]*4],
        initial_domain='A_k=||a_tensor^k psi|| on actual gamma f(H_D) RAWvacuum trial, through5; boundary/gamma only Phi/SU4, not coexactA',
        initial_proof='A_k<=32 d_g^k sum_r Stirling(k,r)(CL/L)^r C_f,r; exact compact Fourier moments<=2^2048 through8. SOURCE multiplies byCQ/Z (CQ16sqrtV, Z>=1/4); CCR truecreator division by actualnorm>=source norm. No resetvac or a†eta',
        initial_cloud_required_log2_lambda=cloud_onset,
        exact_Weyl_force_coefficient_upper=state['CL_upper'],
        displacement_log2_upper=dglog,
        ordinary_hierarchy='A_k(t)<=sum_l binom(k,l) A_l(0) d_g^(k-l) Touchard_(k-l)(CL*t), free frequencies removed unitarily; all untruncated coexact modes. Rest fast/root/HC/harm commute Ncoexact',
        actual_uniform_regular_Nplus1_squared_graph_upper=dict(neutral=16,normalized_SOURCE=16,normalized_true_creator=64),
        graph_polynomial='||(N+1)^2 psi||²=A0²+15 A1²+25 A2²+10 A3²+A4²; actual initialA5 pays truecreator A4',
        fast_kinetic_graph_log2_coefficient_upper=kinetic_log,fast_kinetic_graph_Lambda_power=F(3,8),
        fast_elliptic='Kf=Tfast+W>=0 on actual regular domain, W pointwise Hermitian positive including Root/beta shifts. Kf commutes H_EM and Kf+H_EM=H_D-e_global-hL (+bounded fixed shifts), hence graph<=2L+CL+fixed. 2Re<Tfast psi,Wpsi>=2chi<gradpsi,W gradpsi>-chi<DeltaW>. Actual raw Cartesian/left-SU4 DeltaW<=C_N Lambda^(5/4); chiDeltaW<=C_N Lambda^(3/4). Thus ||Tfastpsi||<=C_N Lambda^(3/8); ROOT Delta and beta derivatives ZERO. Interior gamma/Caccioppoli keeps compact core; no component-kinetic commutation assumed across curved Dirichlet boundary',
        fast_pointwise_laplacian_log2_coefficient_upper=lapW,
        normal_photon_mixed_rows=rows,
        mixed_proof='q commutes N: ||q sqrt(N+1)psi||²<=||qpsi|| r||(N+1)psi||. Pi commutes N: ||chi Pi sqrt(N+1)psi||²<=chi²||Pi²psi||||(N+1)psi||. These are exact mixed endpoint bounds, never products of Phi and photon marginals',
        weighted_light_normal_Duhamel_terms=weighted_terms,
        actual_OUT='a†_g B_total U_HD(6)Xi, pairedSOURCE and singleGLOBALreceiver; B coexact commutator zero. Weighted Duhamel differentiates endpoint graph, includes [a_g,hL] bounded Weyl force and q coefficients. Berry force has zero coexact derivative. SAME actual normal-heavy operator retained, not D_ms frozen evolution',
        whole_Dirichlet_graph='KEM=coexact plus FREE harmonic, HC belongs positive W. H_R+BN=KEM+Tother+W, Tother commutes KEM. Elliptic cross gives ||KEMpsi||²<=||(H_R+BN)psi||²+UE<C_N(1+Phi²)>; only light Peierls depends A and every actual length2 charge vector has norm²<=2. Global quartic controls Phi². Actual initial gamma fn Hgraph <=C_N Lambda via spectral cap+IMS; autonomous wholeH conserves this Hgraph. No global N² claim',
        whole_creator_interpolation='Ncoexact<=omega_min^-1 KEM with omega_min>=c_N/N; BOTH ordinary endpoints have polynomial firstNgraph. For SUPEREXP Agmon delta only: ||a†delta||²<=||delta||²+||delta||||Ndelta||, so exp(-c_N Lambda^(1/4)) pays ANY polynomial Lambda factor. B afterwards costs conservativeCQ. NEVER apply this to polynomial fn-shadow error; that uses direct truecreator/Fourier proof',
        whole_graph_scale='||N delta||<=C_N Lambda with N/c inside fixed coefficient; unshiftedH_R/BN actual, not tensor energy origin',
        required_log2_lambda=onset,selected_log2_lambda_lower=2**315,
        not_claimed='driven Noether N²/whole mixed high moments/native quantum spatial closure or Born receiver coefficient')

def finite_resolution_photon_response(state:dict[str,Any],duration:F=F(6)) -> dict[str,Any]:
    """Photon phase/current observable on the actual moving neutral right state."""
    dglog=math.ceil(F(ceil_log2(64*UE/C),2))
    force_log=ceil_log2(state['CL_upper'])+dglog
    phase_log=force_log+ceil_log2(max(F(1),abs(duration)))+2
    if phase_log>=-20:raise AssertionError('Actual finite-resolution photon phase bound fails')
    return dict(identity=IDENTITY,duration=duration,
        actual_neutral_norm_lower=state['neutral_preparation_norm_lower'],
        actual_SOURCE_relative_norm_lower=state['source_relative_to_actual_Xi_norm_lower'],
        actual_bank_probabilities=state['actual_prepared_bank_probability_interval'],
        mode_scope='normalized single mode or degenerate-frequency shell; general packets use a†_(exp(+iOmega t)g), NOT a fabricated single omega for unequal frequencies',
        observable='[U_D(t)† a_g† U_D(t)-exp(+i omega_g t) a_g†] Xi; SAME actual neutral U_D(t) on RIGHT, not Xi energy eigenstate or vacuum-return',
        physical_SOURCE_observable='same expression on actual normalized Bplus/minus Xi and pairedSOURCE input, physical fn only; allRef/RAWquartet and ONE source norm',
        exact_response_identity='time integral of exp(+i omega_g(t-s)) U_D(s)† i[H_D,a_g†]_interaction U_D(s); free omega_g a_g† removed exactly',
        interaction_force_log2_norm_upper=force_log,
        finite_time_phase_vector_error_log2_upper=phase_log,
        finite_resolution_vector_norm_target=F(1,2**20),
        error_units='ABSOLUTE photon-creation vector norm on NORMALIZED actual neutral/source states; NOT a Born leading-relative error or source/scattering probability',
        exact_photon_dispersion='omega(k)=c sqrt(sum_j 4 sin²(pi wrapped_k_j/N)); all coexact modes k!=0, harmonic treated as FREE momentum separately',
        canonical_link_creation_coefficient='sqrt(UE/(V omega_g)), fixed LAW05 electricUE Pi²; no extra1/sqrt2 or1/sqrtV',
        current_identity='[a_g,H_D]-omega_g a_g is EXACT Peierls current insertion sum_e d_e,g R_e W_e; full4V/encodedPauli light matrix and allmode Weyl retained, regularPhiMax4 gives norm<=CL*8sqrt(UE/c)',
        shared_response='same actual H/current/UE/c/normal state for BOTH physical SOURCE maps, photons and photon contact; source density/harm/Root coefficients kept',
        finite_soft_continuum_frequency_relative_bound='for |q|<=.05, 0<=1-omega(q)/(c|q|)<=|q|²/24; hardSOURCE19 modes remain exact lattice, NOT softretagged',
        ordinary_offsets='all64 RAW amplitudes/all512 actual histories, bankEextra and ordinary phases remain in U_D and actualfn; no scalar energy reset',
        whole_H_transfer='use ordinary_weighted_OUT creator interpolation for neutral right SUPEREXP Agmon and direct actual fn truecreator lowwindow for LEFT evolution. The regular response remainder is bounded before unitary comparison; never a†eta or whole photon N²',
        limitations='finite-time quantum photon spectral/current response, NOT sharp dressed spectral pole, eternal photon vacuum, quantum-law derivation, spatial quantum closure or calibrated Born coefficient')


def native_alignment_quasimode(normal:dict[str,Any]) -> dict[str,Any]:
    """Actual original N5 field/colour potential, new two-coordinate proof."""
    G=np.asarray(normal['centre_probe'][0]['actual_full10_gram']['G'],float)
    Phi=np.einsum('mn,nij->mij',np.linalg.cholesky(G),np.asarray(SELECTED[:4]))
    N=5;V=N**3;T=1j*SELECTED[1]@SELECTED[2];Tc=SELECTED[0]@T
    soft=Phi.copy();soft[0]+=T;soft/=math.sqrt(V*np.sum(abs(soft)**2))
    generator=Tc/math.sqrt(V*np.linalg.norm(Tc)**2)
    onsite,paths=scalar_paths(RotationBand(**BARE))
    momenta=2*np.pi*(np.asarray(list(itertools.product(range(N),repeat=3)))+np.array([0,0,.25]))/N
    def potential(s:float,t:float) -> tuple[float,float]:
        phi=Phi+s*soft;U=expm(1j*t*generator);gram=np.einsum('mij,nji->mn',phi,phi).real/4
        scalar=V*np.sum(gram**2);H=np.tile(onsite*phi[0],(V,1,1)).astype(complex)
        for displacement,coeff in paths:
            axis=int(np.flatnonzero(displacement)[0]);length=int(displacement[axis])
            Fphi=np.einsum('m,mij->ij',coeff,phi);word=np.linalg.matrix_power(U,length) if axis==2 else np.eye(4)
            block=(Fphi@word+word@Fphi)/2;phase=np.exp(1j*(momenta@displacement))
            H+=phase[:,None,None]*block+phase.conj()[:,None,None]*block.conj().T
        scalar-=np.linalg.eigvalsh(H)[:,-2:].sum()
        alignment=.2/8*V*np.sum(abs(phi-np.einsum('ij,mjk,kl->mil',U,phi,U.conj().T))**2)
        return float(scalar),float(alignment)
    v0=potential(0,0)[0];eps=2e-3
    A=np.empty((2,2));A[0,0]=(potential(eps,0)[0]-2*v0+potential(-eps,0)[0])/eps**2
    A[1,1]=(potential(0,eps)[0]-2*v0+potential(0,-eps)[0])/eps**2
    A[0,1]=A[1,0]=(potential(eps,eps)[0]-potential(eps,-eps)[0]-potential(-eps,eps)[0]+potential(-eps,-eps)[0])/(4*eps**2)
    comm=generator@Phi-Phi@generator;b=.2/4*V*np.sum(abs(comm)**2)
    if A[0,0]<=0 or b<=0:raise AssertionError('Actual soft/stiff Hessian failed')
    mass=np.diag([1.,.6]);sqrtmass=np.diag(np.sqrt([1.,.6]));invsqrtmass=np.diag(1/np.sqrt([1.,.6]))
    nodes,weights=hermgauss(7);rows=[]
    for exponent in (8,16,24,32):
        lam=2.**exponent;chi=lam**(-.5);K=lam*A+np.diag([0.,b*lam**1.25])
        e,U=np.linalg.eigh(sqrtmass@K@sqrtmass)
        if min(e)<=0:raise AssertionError('Actual quadratic has no positive Gaussian')
        S=invsqrtmass@(U*np.sqrt(e/chi))@U.T@invsqrtmass
        cov=.5*np.linalg.inv(S);L=np.linalg.cholesky(cov);aligned=curved=0.
        for i,j in itertools.product(range(len(nodes)),repeat=2):
            q=math.sqrt(2)*L@np.array([nodes[i],nodes[j]]);s,t=q
            scalar,align=potential(s,t);quad=.5*q@A@q
            rflat=lam*(scalar-v0-quad)+lam**1.25*(align-.5*b*t*t)
            # SAME tangent/Hessian but genuinely curved forward chart.
            # Its alignment cubic includes ONE stiff and TWO soft legs.
            curved_scalar,curved_align=potential(s,t+s*s)
            Sq=S@q
            kinetic_curve=chi/2*(-4*s*(S[0,1]-Sq[0]*Sq[1])-2*Sq[1]+4*s*s*(S[1,1]-Sq[1]**2))
            rcurve=rflat+lam*(curved_scalar-scalar)+lam**1.25*(curved_align-align)+kinetic_curve
            weight=weights[i]*weights[j]/np.pi
            aligned+=weight*rflat*rflat;curved+=weight*rcurve*rcurve
        rows.append(dict(log2_lambda=exponent,actual_soft_scaled_covariance=float(cov[0,0]*lam**.75),
            actual_stiff_scaled_covariance=float(cov[1,1]*lam**.875),
            Riccati_relative_residual=float(np.linalg.norm(chi*S@mass@S-K)/np.linalg.norm(K)),
            actual_two_mode_zero_point=float(chi*np.trace(mass@S)/2),
            actual_original_potential_residual_norm=math.sqrt(aligned),curved_chart_residual_norm=math.sqrt(curved),
            curved_residual_over_Lambda_1over16=math.sqrt(curved)/lam**(1/16)))
    flat_zero=max(potential(s,0)[1] for s in (-.2,.1,.3))
    if flat_zero>1e-25 or max(row['Riccati_relative_residual'] for row in rows)>1e-12:
        raise AssertionError('Actual alignment-flat family or true correlated Gaussian failed')
    # Public API control uses actual frozen classical branch energies, not
    # fabricated512 quantum amplitudes or per-branch physical filters.
    branch_v=np.array([V*r['actual_full10_gram']['value'] for r in normal['centre_probe']])
    control=global_shadow_weights(branch_v,np.zeros(2),np.ones(2)/math.sqrt(2),np.zeros(2),2.**32,
                                  2.**32*min(branch_v),32*(2.**32)**.375,time=6.)
    if abs(control['norm']-1/math.sqrt(2))>1e-12:
        raise AssertionError('A per-branch floor reset contaminated ONE global filter')
    return dict(size=N,phi_source='ONE frozen trueN5 skewquarter heavy/scalar Gram, NOTnewargmin orold401s rerun',
        actual_heavy_scalar_hessian=A,actual_alignment_stiff_hessian=b,actual_alignment_soft_family_residual=flat_zero,
        cometric=mass,rows=rows,
        one_global_classical_branch_control=control,
        branch_control_scope='two actual frozenN5 classicalminima/zeroquantumZP; APIadversefloor-resetcontrol ONLY, not512 RAWquantumstate',
        scope='new actual original2coordinate field/colour potential and quantum Gaussian multiplier residual; NOTfull90Vnative Hamiltonian evolution')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('normal-receipt','sector-stdout','source19-capsule','soft-capsule','root-source-capsule'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    loaded={Path(cast(str,m.__file__)).name:hashlib.sha256(Path(cast(str,m.__file__)).read_bytes()).hexdigest() for m in tuple(sys.modules.values())
            if getattr(m,'__file__',None) and Path(cast(str,m.__file__)).parent==root and Path(cast(str,m.__file__)).suffix=='.py'}
    normal=json.loads(a.normal_receipt.read_text());sector=json.loads(a.sector_stdout.read_text())
    sector=sector['ledger']
    for path,sha in ((a.source19_capsule,'8a1ec1ad2e826ce97792a12ab525369fa1c8a22f8914a974f831713984a29029'),
                     (a.soft_capsule,'d173f73fdd060eccdfbd0311d498b08ff9b1c9bb61d7204896de3a595d95f8de')):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Wrong exact RAW norm theorem binding')
    root_binding=json.loads(a.root_source_capsule.read_text())
    if hashlib.sha256(a.root_source_capsule.read_bytes()).hexdigest()!='21922587c0996e0c580061c426dc58e21248d40b32e0f43d9014f9c47e7d90f8':
        raise ValueError('Exact original SOURCE/Root-cycle binding required')
    if root_binding['natural_exit']!=0 or root_binding['results']['root_sha256']!='8e107f1fecb8483200c1520b8afd0032a653c24bae5b81d1ecad36a81d0cdd9e':
        raise ValueError('Original unshifted Root authority mismatch')
    constants=normal['legal_N'] # immutable exact receipt; NEVER recompute huge rational majorants.
    if constants['size']!=NSTAR or constants['physical_normal_dimension']!=90*NSTAR**3:
        raise ValueError('Frozen TRUE90V local-normal receipt must bind selected legalN')
    state=global_filter_ledger(sector,constants)
    result:dict[str,Any]=dict(ledger=state,initial_energy=initial_energy_ledger(state,constants),light_normal_shadow=light_normal_shadow_ledger(state,constants),ordinary_weighted_OUT=ordinary_weighted_OUT_ledger(state,constants),finite_resolution_photon=finite_resolution_photon_response(state),native=native_alignment_quasimode(normal),loaded_source_sha256=loaded,
                original_SOURCE_Root_binding=root_binding['formal_original_graph_binding'],
                input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.normal_receipt,a.sector_stdout,a.source19_capsule,a.soft_capsule,a.root_source_capsule)})
    result['plan']=PLAN;result['assignment_revision']=PLAN_REVISION
    result['loaded_sources_unchanged']=all(hashlib.sha256((root/name).read_bytes()).hexdigest()==h for name,h in loaded.items())
    print(json.dumps(exact_capsule(result),indent=2,sort_keys=True))

if __name__=='__main__':main()

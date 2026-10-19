#!/usr/bin/env python3
"""Smooth actual local-well energy-window preparation and boundary ledger.

Explicit nonzero-norm spectral mass replaces the failed Jensen-only filter.
FullN statements name their local-normal/Agmon hypotheses; native is a control.
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
from typing import Any

if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from prepared_neutral_full_field import PreparedQuantumSlice, neutral_descriptor, require, full_field_bounds, l1
from prepared_neutral_hessian import hessian_descriptor
from noncompact_quantitative_normal_budget import (
    native_normal_budget, sqrt_lower, sqrt_upper, ceil_log2, whole_heavy_projector_jets,
    orientation_source_jets,
)
from noncompact_screened_gauss_dressing import orientation_core_halfspin_bound
from original_charged_principal import portable
from improved_phi_band import NSTAR, UE, C, selected_band

ROOT=Path(__file__).resolve().parent
IDENTITY='PNFF-v3-localwell-smooth-window-correlated-Hessian'
MINSET_IDENTITY='PNFF-v3W-MinSet-fullRoot-minfloor-union'
PLAN='issue-222-fullfield-closedrotation-continuation-20261006'
PLAN_REVISION='70a9c264b2e21fa58d9afa900a3c4ce49348fe8057ee57a5bee389b83ffe57b6'


def exact_capsule(value: Any) -> Any:
    if isinstance(value,F):
        return dict(exact_numerator_hex=hex(value.numerator),exact_denominator_hex=hex(value.denominator))
    if isinstance(value,dict):
        return {str(k):exact_capsule(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [exact_capsule(v) for v in value]
    return portable(value)


def local_well_descriptor() -> dict[str,Any]:
    d=hessian_descriptor()
    d.update(identity=MINSET_IDENTITY,predecessor_identity=IDENTITY,
        candidate_not_current_join_state=True,
        full_state='ALL64 initial m amplitudes1/8; per m trial populates ONLY exact finiteN classical minimizerSET A_m=argmin_s v(m,s), equal normalized tie weights1/sqrt|A_m|; smooth fullRoot H_D window and union cutoff; actualH/Un/SOURCE retain ALL512 histories and ordinary phases',
        branches='ALL64 initial colour amplitudes1/8, exact MinSet s/ties; ALL512 actualH/Un/SOURCE histories; no assumed disjoint centres, gaugeorbits or conserved branches; actual common normalization required',
        H_Dtheta='actual microscopic fullGauss fullRoot H_D(m) on union-of-well tubes, NOT independently conserved H_D(m,s); all ordinary heavy/scalar/light/root/EM/harmonic sectors retained',
        well_centres='exact implicit finiteN full10 original-heavy/scalar argmin G_N(m,s), NOT N5 ordering/template/G_infinity; whole domain includes tubes around ALL relevant classcentres andGaugeorbits',
        E_floor_theta='e_m=Lambda*min_s v(m,s)-C_L*rho-beta; lawful FULLRoot floor, NOT per-s or unknownquantumground; otherflatcode included',
        branch_floor='FULLRoot min_s floor only; coupled s labels are not independently positive/conserved fibres',
        spectral_window='Ltheta=4*Ccommon*Lambda^3/8 with Ccommon bounding BOTH neutral and source-created trial excess; f=1 on[0,1], smooth taper on[-1,0] and[1,2], zero outside[-1,2]',
        taper='f_taper(t)=exp(-1/(1-t))/(exp(-1/t)+exp(-1/(1-t))), 0<t<1; endpoints1,0',
        gamma_theta='smooth1 on union distance<=rtheta/3, zero for union distance>=2rtheta/3; rtheta=r0 Lambda^-1/4; each actual finiteN/aux local-well center retained; true gamma derivatives/curvature paid',
        norm_lower='UNPAID for CURRENT MinSet: conditional target filter>=sqrt(3)/2, postgamma>=1/2, combined>=sqrt(3)/4 requires fresh actual trial-Eex/source/tube proof; no historical3/4 retag',
        photon='NOT vacuum after smooth fullH filter: actual [H_D,a] pullthrough gives initial cloud <=Cf*relative_leg/L; all coexact modes and harmonic paired history retained',
        reference_particle='neutralRef4, qref=Nref-4; Bplus creates actual light2V+1/ref3 bar4',
        reference_antiparticle='neutralRef0, qref=Nref; PH-conjugate Bminus removes light particle and creates ref1 defining4',
        neutral_C='Phi parity+PH heavy/light+complex conjugation of fields/photons; neutralRef4 orRef0 has trivial K on its1D singlet; kappa maps minus kappa; C_n W=W C_eff',
        charge_C='full PH_ref swaps particleRef4 to antiRef0 and chargedRef3 toRef1 (Hodge); not an invariant Ref4 ket under full Fock PH_ref',
        fullQ_scope='spectral H_D graphs plus gamma/source commutators finite; ordinary offsets retained, no fullQ smallness gate in direct relative Duhamel',
        source_primitive='same whole-Haar Phi/SU4 psi1/psi0 eta18 alpha2^371 mark, PURE longitudinal Coulomb/Dirac EM dressing; no coexact EM Wilson insertion; literal harmonic/reference boundary retained',
        narrow_source_words='all finiteV source words <=3N+2; fresh whole-Haar q0..6 jet costs; original source19 and narrow source profiles distinguished',
        current_gamma_commutator='[J_f,gamma]=[C_ff,gamma]=0 since current/contact and gamma are coordinate multiplication; angular [J_Noether,gamma] is NOT zero',
        covariance_and_window_are_preparation_not_receiver=True,
        local_floor_Agmon_hypotheses_unreviewed=True,
        trial_excess_includes_history_spread='Lambda sum_s w_s(v_m,s-min_sprime v_m,sprime); vanishesforMinSet ONLY; equal8 predecessor failed selectedLambda scaling',
        Root_history_semantics='actualcoupledcode sectors: [H,P_s]=sum_e beta_e[Z_e,P_s], generallynonzero; notexternalBC/exactfibres',
        source_cycle_derivation='conditionalactualencodedEvenCAR: cycleS_j central inAij/Bi algebra, SOURCE cdagger*cRef pluscoordinatewords even ⇒[B,P_s]=0; betaZ outsideevenalgebra',
        Root_code_beta_offdiag_and_leakage_paid=False,
        native_historical_identity=IDENTITY,native_receipts_not_MinSet=True,
        full_field_transfer_admitted=False,actual_equilibrium_photon_phase_paid=False)
    return d


def conditional_local_ledger(normal: dict[str,Any], time: F, radius: F) -> dict[str,Any]:
    """Exact outward composition AFTER the explicitly stated normal lemmas.

    A computed selector is not proof of those hypotheses. In particular the
    old static-ground moment is never used as this prepared-state energy.
    """
    b=native_normal_budget(NSTAR)
    D,V=b.dimension,b.volume
    light_l1=l1(selected_band()[0])
    light_gap=F.from_float(float(selected_band()[0].m))*sqrt_lower(b.gram.lower)/2
    light_jets={q:(2*V+1)*math.factorial(q)*(64*light_l1/light_gap)**q for q in range(7)}
    chart=b.chart_majorant
    heavy_jets=whole_heavy_projector_jets(b,6)['Kato_section_jet_upper']
    core=orientation_core_halfspin_bound(3*NSTAR+2,18.,.025,2.,.25)
    core_exponent=int(core['alpha_expression'].split('^')[1])
    require(core_exponent<=371,'fixed actual alpha covers fresh max-word core norm')
    # Fresh all-V Frechet jets. The selected fixed alpha is never retuned.
    fresh=orientation_source_jets(3*NSTAR+2,18,2**371,2*V,F(1,16),6)
    source_jets=fresh['normalized_quartet_source_jet_upper']
    kappa=min(b.alpha,b.stiff)/(2**32*chart**8)
    r0=min(b.radius/16,kappa/(2**16*chart**8))
    # Both neutral and source-created trial graph coefficients are included;
    # this majorant does NOT certify the local form lemmas by arithmetic.
    trial=(2**64)*(D+1)**8*chart**32*(1+1/kappa+sum(heavy_jets.values(),F())+sum(light_jets.values(),F())+sum(source_jets.values(),F()))**8
    covariance_trace=(D+1)*chart*(1+1/kappa)
    # Lower positive Agmon action, after potential/kinetic comparison in H_D.
    agmon=sqrt_lower(kappa/chart)*r0*r0/(2**16)
    require(agmon>0,'positive stated local Agmon action')
    graph_log=ceil_log2(trial+covariance_trace+1/r0+1/kappa)
    allmode=2*(V-1)
    # Deliberately loose low-energy mode-count prefactor, represented only
    # as logs. Never allocate a dyadic integer with an exponent proportionalV.
    P=8*(D+allmode+3)
    omega_min=4*C/NSTAR
    base=1+trial+chart+1/r0+1/kappa+1/UE+1/omega_min
    log_prefactor=64*V+P*ceil_log2(base)+graph_log
    desired=16
    power=4*(P+desired+8)
    log_factorial=power*(ceil_log2(power)+max(0,ceil_log2(1/agmon)))
    # exp(-x)<=k^k/x^k, so the Agmon polynomial envelope is <=Lambda^-16
    # once the displayed coefficient is absorbed by an additional Lambda^-8.
    leakage_selector=(log_prefactor+log_factorial+7)//8
    barrier_selector=8*ceil_log2(2**16*trial/(kappa*r0*r0))
    gaussian_selector=4*ceil_log2(64*covariance_trace/(r0*r0))
    # Jets above already include the fresh conservative lower1/4. A new
    # prepared-state lower still needs the stated C_n/source-value argument.
    normalized_source_lower=F(1,4)
    source_constants=source_jets
    momentum_coefficient=2*(1+chart)*trial
    distance_coefficient=4*trial/kappa
    first_coefficient=time*(source_constants[1]*momentum_coefficient+source_constants[2]+source_constants[0]*distance_coefficient)
    angle_coefficient=F(7)*(radius+1)*source_constants[1]*distance_coefficient
    first_log=ceil_log2(first_coefficient+angle_coefficient)
    local_lambda_log=int(normal['time6_operator_capability']['local_log2_Lambda'])
    required=max(leakage_selector,barrier_selector,gaussian_selector,16*(first_log+64))
    expression=normal['time6_operator_capability']['global_log2_Lambda_expression']
    inner=int(expression['coercivity_reciprocal_inner_exponent'])
    # The known global tower dominates any coefficient whose bitlength is
    # smaller than inner; this comparison is integer arithmetic, not infinity.
    global_tower_dominates=inner>required.bit_length()+16
    photon=full_field_bounds(normal,time)
    return dict(identity=MINSET_IDENTITY,scope='conditional MinSet actual-fullRoot union-well theorem, NOT an independently accepted global proof',
        hypotheses=[
            'actual union-well H_D floor includes light/beta/off-code comparison; accepted actual local chart forms and exact finiteN centres, not numeric template',
            'correlated trial Wick/remainder bounds apply to BOTH neutral and source-created trials with displayed Ccommon',
            'fullH_D low-window packet satisfies genuine weighted counting/Agmon estimate, INCLUDING photon/harmonic/root degeneracy and curvature',
            'new prepared C_n-real source-value lower>=1/4; fresh max3N+2 Frechet jets still require actual physical Noether contractions',
            'local well ordinary offsets and Dirichlet boundary extension are actual microscopic H forms, not independent superselection labels'],
        hypotheses_independently_paid=False,
        kinetic='Phi Lambda^-1/2 p²/2; colour .3 Lambda^-1/2 E4²; LAW05 UEΠ²+(K/2)curl², UEρGreenρ, harmonicUE/Vp_Ah²',
        normal_cometric_upper=portable(chart),kappa=portable(kappa),r0=portable(r0),Ctrial=portable(trial),
        actual_local_floor='e_m=Lambda*min_s v(m,s)-C_light*rho_max-beta; lawful FULLRoot floor including otherflatcodes, not per-s floor',
        nonzero_filter_norm_squared_lower='UNPAID CURRENT MinSet; conditional target3/4 ONLY after fresh fullRoot trialEex<=L/4',
        post_cutoff_norm_lower='UNPAID CURRENT MinSet; conditional target1/2 requires union Agmon/boundary proof',
        combined_norm_lower='UNPAID CURRENT MinSet; conditional target sqrt(3)/4',squared_amplification_upper='CONDITIONAL target16/3, not an admitted norm bound',
        proof_nonzero='UNPAID: actual new MinSet trial and source Eex<=L/4 plus fullRoot floor required before Markov; old equal8/oldnative3/4 proof is not reused',
        uniform_chiPi_squared='<=8*chart*Ctrial*Lambda^-1/8 +boundary graph; norm Lambda^-1/16',
        uniform_distance_squared='<=8*Ctrial/kappa*Lambda^-5/8 +boundary graph; norm Lambda^-5/16',
        source_first_relative_error='T*(C1 chiPi+C2 chi/2+C0 distance); angular error<=2pi*(C1 R distance+colour/EM/Gauss/ref terms)',
        actual_time=str(time),actual_radius=str(radius),source_constants={str(q):portable(v) for q,v in source_constants.items()},fresh_core_norm_control=core,fresh_source_jet_binding=portable(fresh),
        first_error_coefficient_log2_upper=first_log,
        first_error_Lambda_power='-1/16 (coarse sum of faster negative terms)',
        fullQ_source_graph='finite polynomialLambda graph from smooth energywindow and differentiated gamma/B; coefficient ZERO in direct first-moment Duhamel; not required absolutely small',
        agmon_action_lower=portable(agmon),mode_count_Lambda_power=P,
        cutoff_commutator='[H,gamma]=-(chi/2)(Delta_g gamma+2grad_g gamma.Pi), actual SU4/Gauss metric; [J_Noether,gamma] includes orbital/curvature boundary; [J_f,gamma]=[C_ff,gamma]=0',
        actual_time_leakage='T Cpref Lambda^P exp(-agmon Lambda^1/4); angular total2pi Cang Lambda^P exp(-agmon Lambda^1/4)',
        branch_coherence='64 amplitudes1/8; triangle bound factor8; no sector conservation assumption',
        required_new_log2_Lambda=required,known_local_log2_Lambda=local_lambda_log,
        local_selector_admitted=local_lambda_log>=required,
        global_tower_comparison_inner=inner,global_tower_dominates_new_selector=global_tower_dominates,
        selector_admitted_if_hypotheses_hold=(local_lambda_log>=required or global_tower_dominates),
        initial_photon_cloud='smooth-filter allmode pullthrough Fouriermoments Cf,j<=2^2048 forj<=6 dividedbyL; NOT zero, no static ground photons',
        finite_time_photon_cloud_log2_bound=photon['allmode_photon_cloud_squared_log2_upper'],
        all64_offsets_retained=True,phase_not_ground_or_equilibrium=True,
        first_second_source_field_ref_curvature_proof_unpaid=True)


def window_photon_moments(normal: dict[str,Any], time: F, radius: F) -> dict[str,Any]:
    """Actual all-mode Weyl commutators and source-weighted factorial moments."""
    b=native_normal_budget(NSTAR);V=b.volume
    # Conserved whole-H quartic energy, not only initial chart support.
    field=full_field_bounds(normal,time)
    rho=F(field['rho_norm_upper'])
    light_l1=l1(selected_band()[0])
    K=F(2**12)*V*light_l1*rho
    omega_min=4*C/NSTAR
    d2=4*UE/omega_min
    Cf=2**2048
    def touchard(p: int,x: F) -> F:
        rows=[[1],[0,1],[0,1,1],[0,1,3,1],[0,1,7,6,1]]
        return sum((F(c)*x**j for j,c in enumerate(rows[p])),F())
    epsilon=UE/(V*2**40)
    source_transfer_bits=2*(ceil_log2(1/epsilon)+4096)
    photon_selector=(8*(2048+ceil_log2(K)+source_transfer_bits//2+32)+2)//3
    initial={p:d2**p*F(1,2**source_transfer_bits) for p in range(1,5)}
    evolved={p:d2**p*touchard(p,K*(time+1))**2 for p in range(1,5)}
    N2=evolved[2]+evolved[1]
    Nplus4=1+15*evolved[1]+25*evolved[2]+10*evolved[3]+evolved[4]
    CB2=256*V;Z2=F(1,16)
    hardware2=(F(7)*2**20*V*radius)**2*d2**3*(C/UE)*Nplus4
    epsilon=UE/(V*2**40)
    scattering_log=(8*(ceil_log2(F(Cf)*(2*K+1)/epsilon)+32)+2)//3
    inner=int(normal['time6_operator_capability']['global_log2_Lambda_expression']['coercivity_reciprocal_inner_exponent'])
    return dict(state_identity=MINSET_IDENTITY,regime='CONDITIONAL MinSet actualWeyl/Root moment comparison; not nativeallV execution',hypotheses_independently_paid=False,
        rho_max=rho,actual_light_l1=light_l1,K_current=K,allmode_word_displacement_squared=d2,
        omega_min=omega_min,Cf_j_upper=Cf,Cf_scope='fixed smoothwindow Fouriermomentsj<=6; flat-exponential derivatives through8 boundedby2^2048',
        photon_initial_suppression_bits=source_transfer_bits,rho_scope='conserved full-H quartic comparison; requiresnewstateenergy<=CseedLambda, notinitialnormalchartonly',
        recurrence='dM_p/dt<=sum_q binom(p,q) K d^q M_(p-q); M_p=sqrt<fallingfactorialN_p>',
        vacuum_solution='M_p<=d^p Touchard_p(K|t|); allmodeorthonormal projector diagonal<=1; NOextra modecount',
        initial_factorial_moment_upper=initial,evolved_factorial_moment_upper=evolved,
        N_squared_upper=N2,Nplus1_fourth_upper=Nplus4,
        source_operator_squared_upper=CB2,normalized_source_squared_lower=Z2,
        source_weighted_N_squared_upper=CB2/Z2*N2,
        neutral_sine_hardware_error_squared_upper=hardware2,
        source_sine_hardware_error_squared_upper=CB2/Z2*hardware2,
        neutral_sine_hardware_error_log2_upper=(ceil_log2(hardware2)+1)//2,
        source_sine_hardware_error_log2_upper=(ceil_log2(CB2/Z2*hardware2)+1)//2,
        exact_time=time,exact_radius=radius,scattering_time=6,scattering_absolute_amplitude_target=epsilon,
        photon_required_log2_Lambda=photon_selector,scattering_required_log2_Lambda=scattering_log,
        global_tower_dominates=inner>max(photon_selector,scattering_log).bit_length()+16,
        root_floor_authority='ed25add3f7d678d8bde74a104f2b5e850577e172aaf00ae2cc2bb8ae424e6ae4',
        unchanged_beta='1/16',unchanged_Delta='BN+Utrial+window_old+beta+G',
        root_window_proof='exactvtheta<=4V CliffordGramItest; 8Ccommon Lambda^3/8<=V Lambda =>newE<=12V Lambda<Utrial<=Q diagonalH-G',
        source_norm_proof='CONDITIONAL actualHermitian creatorG pluswholeHaarCT andwholelight/state/normal comparison; neutralMarkov/C_nalone doesnotproveZ',
        comparator='wholelightA0 Kato negativecluster blockdiagonal actualGausskinetic; beta removedCOMPARISONonly; actualH keeps offdiagkinetic,beta,allEM',
        new_SC19_leading_coefficient_not_evaluated=True,harmonic_not_oscillator=True,old_state_Root_caps_not_reused=True,
        whole_harmonic_Pi_mixed_hardware_bound_paid=False,hardware_scope='ordinarycoexactPi clover bound only; physicalSOURCErelative firstorder commutator requiresseparatebefore-projection contraction; NOTdrivenwholeGerr')


def taper(values: np.ndarray) -> np.ndarray:
    values=np.asarray(values,dtype=float)
    out=np.ones_like(values)
    out[values>=1]=0.
    mid=(values>0)&(values<1)
    a=np.exp(-1/values[mid]);b=np.exp(-1/(1-values[mid]))
    out[mid]=b/(a+b)
    return out


def window_values(energy: np.ndarray, floor: float, width: float) -> np.ndarray:
    u=(energy-floor)/width
    value=np.ones_like(u)
    value[u<0]=taper(-u[u<0])
    value[u>1]=taper(u[u>1]-1)
    return value


def actual_window(H,trial: np.ndarray,floor: float,width: float,steps: int):
    basis=[];aa=[];bb=[];v=trial/np.linalg.norm(trial);previous=np.zeros_like(v);beta=0.
    for _ in range(steps):
        basis.append(v.copy());w=H@v-beta*previous
        alpha=float(np.vdot(v,w).real);w-=alpha*v
        for q in basis:w-=np.vdot(q,w)*q
        beta=float(np.linalg.norm(w));aa.append(alpha)
        if beta<1e-12:break
        bb.append(beta);previous,v=v,w/beta
    tri=np.diag(aa)+np.diag(bb[:len(aa)-1],1)+np.diag(bb[:len(aa)-1],-1)
    energy,unit=np.linalg.eigh(tri)
    coefficients=unit@(window_values(energy,floor,width)*unit[0].conj())
    unnormalized=coefficients@np.asarray(basis)
    norm=float(np.linalg.norm(unnormalized));require(norm>=math.sqrt(.75)-1e-6,'native Markov window norm lower')
    return unnormalized/norm,norm


def charge_conjugate_native(m: PreparedQuantumSlice, state: np.ndarray, time: float) -> dict[str,Any]:
    """Actual primitive PH/ref conjugate and literal dual coefficient flight.

    The folded comparison has nonzero traces. They are retained, not
    misrepresented as the zero trace of the full108 microscopic onebody.
    """
    from improved_phi_band import exterior
    masks,W=m.model.code_isometry()
    lookup={mask:i for i,mask in enumerate(masks)}
    ph=np.zeros((len(masks),len(masks)),complex)
    for j,mask in enumerate(masks):
        ph[lookup[255^mask],j]=(-1)**sum(i for i in range(8) if mask>>i&1)
    off=m.model.defect@W
    Cr=W@ph@W.T+off@ph@off.T
    _,hmasks=exterior(np.zeros((4,4)),2)
    hlookup={mask:i for i,mask in enumerate(hmasks)}
    Ch=np.zeros((6,6),complex)
    for j,mask in enumerate(hmasks):
        Ch[hlookup[15^mask],j]=(-1)**sum(i for i in range(4) if mask>>i&1)
    unit=float(np.linalg.norm(Cr@Cr.conj().T-np.eye(m.rootdim)))
    require(unit<1e-8,'full primitive root charge-conjugation unitary')
    def conjugate(v):
        wave=np.asarray(v).reshape(-1,6,m.rootdim).conj()
        return np.einsum('ab,nbc,dc->nad',Ch,wave,Cr).ravel()
    I=m.model.identity;loop=m.model.stabilizers[0]
    ref=sum((m.model.n(i) for i in range(4,8)),np.zeros_like(I))
    p=2*math.pi*m.law.sites/m.law.size
    lap=4*np.sum(np.sin(p/2)**2,axis=1)
    green=float(np.sum(1/lap[lap>1e-12])/m.law.volume)
    blocks=[];jblocks=[];cblocks=[];bblocks=[];traces=[]
    for phi,u,em,scalar,b in zip(m.fields,m.links,m.asites,m.scalar,m.sources):
        h=m.fold.conj().T@m.law.onebody(phi,u,em)@m.fold
        hh=m.fold.conj().T@m.heavy.onebody(phi,u,np.zeros_like(em))@m.fold
        j=m.fold.conj().T@m.law.onebody(phi,u,em,m.f)@m.fold
        c=m.fold.conj().T@m.law.onebody(phi,u,em,m.f,m.f)@m.fold
        onsite=m.law.onsite*np.mean(phi[:,0],axis=0)
        rh=-m.model.encode_even(onsite,m.model.bulk_modes)-loop@m.model.encode_even(h-onsite,m.model.bulk_modes)
        rh+=np.trace(onsite)*I+np.trace(h-onsite)*loop
        rh+=m.aux_penalty*(I-loop)/2+m.beta*m.model.defect+m.ue*green*ref@ref
        heavy,_=exterior(m.Lambda*hh,2)
        heavy-=m.Lambda*np.trace(hh)*sparse.eye(6)
        blocks.append(sparse.kron(sparse.eye(6),sparse.csr_matrix(rh))+sparse.kron(heavy,sparse.eye(m.rootdim))+scalar*sparse.eye(m.internal))
        jblocks.append(sparse.kron(sparse.eye(6),sparse.csr_matrix(-loop@m.model.encode_even(j,m.model.bulk_modes)+np.trace(j)*loop)))
        cblocks.append(sparse.kron(sparse.eye(6),sparse.csr_matrix(-loop@m.model.encode_even(c,m.model.bulk_modes)+np.trace(c)*loop)))
        bblocks.append(sparse.kron(sparse.eye(6),-b.conj().T))
        traces.append([complex(np.trace(h)),complex(np.trace(hh))])
    H=sparse.block_diag(blocks,format='csr')-m.energy*sparse.eye(len(state),format='csr')
    coefficients=(m.chi/(2*m.normal_scale**2),.3*m.chi/m.colour_scale**2,m.ue/m.phscale**2,m.ue/m.law.size)
    for k,coefficient in enumerate(coefficients):H+=coefficient*m.axis_operator(k,m.p2[k])
    J=sparse.block_diag(jblocks,format='csr');Q=sparse.block_diag(cblocks,format='csr')
    B=sparse.block_diag(bblocks,format='csr')
    anti=conjugate(state)
    h_error=float(np.linalg.norm(H@anti-conjugate(m.H@state)))
    b_error=float(np.linalg.norm(B@anti-conjugate(m.B@state)))
    j_error=float(np.linalg.norm(J@anti-conjugate(m.J@state)))
    q_error=float(np.linalg.norm(Q@anti-conjugate(m.contact@state)))
    require(max(h_error,b_error,j_error,q_error)<2e-7,'literal actual particle/hole/ref/current/contact conjugate mapping')
    source=B@anti;norm=float(np.linalg.norm(source));source/=norm
    source_plus=m.B@state;source_plus/=np.linalg.norm(source_plus)
    evolve=lambda mat,v: np.asarray(expm_multiply(-1j*time*mat,v,traceA=-1j*time*complex(mat.diagonal().sum())))
    anti_flight=evolve(H,source)
    forward_plus=evolve(m.H,source_plus)
    # Antiunitarity changes the time sign: C Uplus(-T)=Uminus(T) C.
    backward_plus=np.asarray(expm_multiply(1j*time*m.H,source_plus,traceA=1j*time*complex(m.H.diagonal().sum())))
    mapped=conjugate(backward_plus)
    error=float(np.linalg.norm(anti_flight-mapped))
    require(error<2e-7,'actual backward-time particle/hole conjugate flight')
    wrong_time=float(np.linalg.norm(anti_flight-conjugate(forward_plus)))
    wrong_ref=float(np.linalg.norm(m.B@anti))
    require(wrong_ref<1e-8,'wrong particle Ref4 source annihilates actual antiRef0 neutral')
    return dict(root_unitarity=unit,H_mapping_error=h_error,B_mapping_error=b_error,
        current_mapping_error=j_error,contact_mapping_error=q_error,
        actual_anti_source_norm=norm,actual_neutral_anti_reference=float(np.vdot(anti,m.nref@anti).real),
        actual_charged_anti_reference=float(np.vdot(source,m.nref@source).real),
        actual_charged_anti_bulk=float(np.vdot(source,m.nbulk@source).real),
        actual_antiparticle_current_norm=float(np.linalg.norm(J@anti_flight)),
        actual_antiparticle_contact_norm=float(np.linalg.norm(Q@anti_flight)),
        actual_anti_source_first=float(np.linalg.norm(H@(B@anti)-B@(H@anti))),
        actual_backward_time_conjugate_flight_error=error,wrong_forward_time_conjugate_defect=wrong_time,
        wrong_particle_source_on_antineutral_norm=wrong_ref,
        actual_auxiliary_representation_retained=True,full_Fock_PH_ref_not_neutral_Cn=True,
        folded_trace_offsets_retained=traces,
        full108_microscopic_trace_zero_not_folded_trace=True,
        native_conjugate_field_charts='Phi plus/minus dual finite slices; no physical fullfield phase inference')


def native(time: float) -> dict[str,Any]:
    m=PreparedQuantumSlice();H=m.H
    trial,hessian=m.correlated_hessian_control()
    # Finite-dimensional lawful termwise lower, not an actual-ground fit.
    block_min=min(float(np.linalg.eigvalsh(r.toarray())[0])+float(np.linalg.eigvalsh(h.toarray())[0])+s
        for r,h,s in zip(m.root_h,m.heavy_h,m.scalar))
    coefficients=(m.chi/(2*m.normal_scale**2),.3*m.chi/(2*m.colour_scale**2),m.ue/(2*m.phscale**2),m.ue/(2*m.law.size))
    floor=block_min+sum(c*float(np.linalg.eigvalsh(p)[0]) for c,p in zip(coefficients,m.p2))-m.energy-1e-7
    expectation=float(np.vdot(trial,H@trial).real)
    width=4*(expectation-floor)
    require(width>0,'lawful native lower floor below trial')
    state,norm=actual_window(H,trial,floor,width,80)
    check,norm_check=actual_window(H,trial,floor,width,110)
    convergence=float(np.linalg.norm(state-check))
    require(convergence<1e-7,'actual smooth local window native Krylov convergence')
    evolve=lambda mat,v: np.asarray(expm_multiply(-1j*time*mat,v,traceA=-1j*time*complex(mat.diagonal().sum())))
    nt=evolve(H,state);photon=m.creator@state
    flight=evolve(H,photon);right=np.exp(-1j*m.phomega*time)*(m.creator@nt)
    source=m.B@state;source_norm=float(np.linalg.norm(source));source/=source_norm
    J,Q=m.J,m.contact;zero=sparse.csr_matrix(H.shape)
    driven=sparse.bmat([[H,zero,zero],[J,H,zero],[Q,2*J,H]],format='csr')
    init=np.concatenate([source,np.zeros_like(source),np.zeros_like(source)])
    response=evolve(driven,init).reshape(3,-1)
    step=2e-4
    Hp=m.hamiltonian(step)-m.energy*sparse.eye(len(state),format='csr')
    Hm=m.hamiltonian(-step)-m.energy*sparse.eye(len(state),format='csr')
    plus=evolve(Hp,source);minus=evolve(Hm,source)
    first=float(np.linalg.norm((plus-minus)/(2*step)-response[1]))
    second=float(np.linalg.norm((plus+minus-2*response[0])/(step*step)-response[2]))
    require(first<2e-6 and second<2e-4,'new window actual same-current/contact response')
    momentum=m.axis_operator(0,m.p2[0])/m.normal_scale**2
    initial_cloud=float(np.linalg.norm(m.annihilator@state)**2)
    final_cloud=float(np.linalg.norm(m.annihilator@nt)**2)
    no_contact=sparse.bmat([[H,zero,zero],[J,H,zero],[zero,2*J,H]],format='csr')
    wrong=evolve(no_contact,init).reshape(3,-1)
    contact_defect=float(np.linalg.norm(wrong[2]-response[2]))
    require(contact_defect>1e-7,'new prepared-state contact adverse discriminator')
    anti=charge_conjugate_native(m,state,time)
    return dict(identity=IDENTITY,dimension=len(state),time=time,
        actual_particle_hole_reference_path=anti,
        native_scope=PreparedQuantumSlice.__doc__,native_fullDirichlet_chart_not_simulated=True,
        native_floor=floor,actual_trial_expectation=expectation,width=width,
        floor_not_fitted_to_spectrum_or_ground=True,
        spectral_preparation_norm=norm,check_norm=norm_check,norm_squared_lower=.75,Krylov_convergence=convergence,
        correlated_to_window_state_difference=float(np.linalg.norm(state-trial)),
        v2_to_window_state_difference=float(np.linalg.norm(state-m.seed)),
        actual_H_energy=float(np.vdot(state,H@state).real),
        actual_H_variance=float(np.linalg.norm(H@state-float(np.vdot(state,H@state).real)*state)**2),
        initial_prepared_photon_cloud=initial_cloud,final_prepared_photon_cloud=final_cloud,
        prepared_photon_cloud_not_erased=True,
        exact_same_neutral_rightaction_photon_error=float(np.linalg.norm(flight-right)),
        wrong_frozen_rightaction=float(np.linalg.norm(flight-np.exp(-1j*m.phomega*time)*photon)),
        source_norm=source_norm,first_response_difference_error=first,second_response_difference_error=second,
        first_response_norm=float(np.linalg.norm(response[1])),second_response_norm=float(np.linalg.norm(response[2])),
        missing_contact_defect=contact_defect,
        chiPi_after=m.chi*math.sqrt(float(np.vdot(nt,momentum@nt).real)),
        source_first_norm=float(np.linalg.norm(H@(m.B@state)-m.B@(H@state))),
        actual_auxiliary_complement=float(np.linalg.norm(m.complement@nt)),
        actual_native_Hessian=hessian,selected_UE_not_native_UE=True,
        normal_boundary_Agmon_not_native_proof=True,full2pi_not_native_execution=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normal-receipt',type=Path,required=True)
    parser.add_argument('--time',type=float,default=1.)
    parser.add_argument('--rotation-time',type=str,default='1008124871316563/2')
    parser.add_argument('--rotation-radius',type=str,default='1342177280000')
    parser.add_argument('--analytic-only',action='store_true',help='execute only legal-N arithmetic/proof capsule; no native evidence implied')
    args=parser.parse_args()
    data=args.normal_receipt.read_bytes();normal=json.loads(data)
    names=('prepared_neutral_local_well.py','prepared_neutral_local_well.md','prepared_neutral_hessian.py','prepared_neutral_full_field.py')
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    result=dict(plan=PLAN,plan_revision=PLAN_REVISION,work_item='prepared-neutral-full-field',
        state=local_well_descriptor(),conditional_fullN=conditional_local_ledger(normal,F(args.rotation_time),F(args.rotation_radius)),
        native=(None if args.analytic_only else native(args.time)),allmode_moments=window_photon_moments(normal,F(args.rotation_time),F(args.rotation_radius)),inputs={str(args.normal_receipt):hashlib.sha256(data).hexdigest()},source_sha256=hashes,
        source_unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==s for n,s in hashes.items()),
        conclusion='CURRENT MinSet definition and conditional floor/Eex/Agmon/source/fullQ ledger, with explicitly unpaid norm targets; optional corrected historical native control is not MinSet or full-field/equilibrium/unconditional closedrotation proof')
    print(json.dumps(exact_capsule(result),indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()

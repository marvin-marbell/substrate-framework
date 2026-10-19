#!/usr/bin/env python3
"""Source-relative full-Fock driven packet/group comparison on the original law.

The free rotating packet is evaluated on the RIGHT of a Duhamel identity;
no free lattice estimate is applied to an EM-distorted charged state.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
import numpy as np
from scipy.linalg import expm
from improved_phi_band import NSTAR, UE, C, exact
from original_charged_principal import portable
from original_total_angular_transfer import scalar_orbital, primitive_flow
from prepared_neutral_full_field import BETA
from prepared_neutral_shared_photon_state import load_shared_native_cache, loaded_source_epoch
from prepared_neutral_mean_packet_intertwiner import (
    MeanPacketIntertwiner, apply_onebody, coulomb_packet_action, transfer_matrix,
)
from functional_source_photon_bank import Full108ReferenceMarks

PLAN = 'issue-222-mercury-synthesis-continuation-20261007'
REVISION = 'a515ec257394da01e219692c75076125c6903434fb5e6d64edfa547d3156c6d9'
INITIAL_SHA = '64612effb995b3715d7dbc7d49cf0a1b7d3daef415986e730e32198924df47f4'
CORE_SHA = '71bc2f634993a6a63049609af2cae849e0b3a1e40dc98dd8a9ab20d586fb0e86'
HARMONIC_SHA = '953c41379268fcd8b40588effc3f233966e6054772245d49e5acdf83aa87f646'
REFERENCE_SHA = '66b475ce7a259418a74d62790c1eef2d9173d37fec1b64fe4b895d5696512b2e'
CLOCK = 1008124871316564
ROUT = 2684354560000


def ceil_log2(value: F | int) -> int:
    value = F(value)
    if value <= 0:
        raise ValueError('Positive finite bound required')
    k = value.numerator.bit_length() - value.denominator.bit_length()
    p = F(2**k) if k >= 0 else F(1, 2**(-k))
    return k if p >= value else k + 1


def log_sum(*logs: int) -> int:
    return max(logs) + ceil_log2(len(logs))


def finite_mean_screen_bound() -> dict[str, Any]:
    """Original eta18 centre product M(x)=tanh(36)^ell_x, NEVER set to one."""
    # e>sum_(k=0)^12 1/k!, hence delta=2/(exp(72)+1)<2/e_lower^72.
    e_lower = sum((F(1, math.factorial(k)) for k in range(13)), F())
    delta = 2/e_lower**72
    # At nonwinding edges |ell_x-ell_y|<=path length<=2 and
    # |c^ell_x-c^ell_y|<=2 delta. Mean light hopping row<=8,
    # Ref0/4 SAME-Z source norm<=4q0<256: 2^12 dominates16*256.
    H = F(2**12*CLOCK)*delta
    # Nearest covariant orbital row<=3Rout, same source bound and
    # integral|Omega|<8. Its winding edges vanish outside original guard.
    G = F(2**12*8*(1+3*ROUT))*delta
    total = 4*H+2*G
    return dict(original_eta=18, exact_c0='tanh(36)',
        original_full_mean_coefficient='M_c(x)=tanh(36)^ell_x; full original centre characters, core/Ref and harmonic factors multiply the ideal free envelope',
        delta_upper_exact=str(delta), H_integral_exact=str(H), G_integral_exact=str(G),
        total_comparison_exact=str(total), H_log2_upper=ceil_log2(H),
        G_log2_upper=ceil_log2(G), total_log2_upper=ceil_log2(total),
        interior_length_difference_upper=2, unchanged_Z_source_bound_upper=256,
        periodic_jump='No per-edge delta bound at a winding seam: |M_x-M_y|<=1 there, covered by separately paid far-packet seam. Orbital velocity is zero at that seam.',
        initial_scalar_discarded=False, original_mean_product_assumed_unitary=False)


def packet_residual(marks: Full108ReferenceMarks, physical: np.ndarray,
                    comparison: np.ndarray, physical_ref: np.ndarray | None = None,
                    comparison_ref: np.ndarray | None = None) -> Full108ReferenceMarks:
    """Exact arbitrary full108/Ref4 ABI, both maps and true neutral RIGHT."""
    left = MeanPacketIntertwiner.onebody_action(marks, physical, physical_ref)
    right = MeanPacketIntertwiner.onebody_action(marks, comparison, comparison_ref)
    return Full108ReferenceMarks(left.plus - right.plus, left.minus - right.minus)


def harmonic_source_jet(marks: Full108ReferenceMarks, displacement: np.ndarray,
                        N: int, direction: np.ndarray) -> Full108ReferenceMarks:
    """theta=N Ah: derivative of exp[-i (x-Ref).theta/N], NOT zero."""
    displacement = np.asarray(displacement, float)
    if displacement.shape != (marks.plus.shape[0] // 4, 3) or N <= 0:
        raise ValueError('All site displacements and the original period required')
    scalar = np.repeat(displacement @ np.asarray(direction) / N, 4)
    return Full108ReferenceMarks(-1j * scalar[:, None] * marks.plus,
                                1j * marks.minus * scalar[None, :])


def driven_packet_bound(energy: dict[str, Any], initial: dict[str, Any],
                        core: dict[str, Any]) -> dict[str, Any]:
    """Absolute source/group residue; HC payment is NOT a Born-relative bound.

    Constants deliberately use site/Pauli triangles. The number graph belongs
    to the actual slow neutral reference, not a vacuum or a norm-close shadow.
    """
    N = int(NSTAR); V = N**3; v = ceil_log2(V); n = ceil_log2(N)
    t = ceil_log2(CLOCK); r = ceil_log2(1 + 3*ROUT)
    lam = int(initial['initial_energy']['selected_log2_lambda_lower'])
    missing = list(energy.get('missing_prerequisites', []))
    if not energy.get('closed', False):
        missing.append('Actual slow-reference first-number/harmonic momentum bound is not closed')
    if int(energy['original_N']) != N or int(energy['original_T']) != CLOCK:
        raise ValueError('Unchanged LEGAL period and long quintic clock required')
    if (int(energy['selected_log2_Lambda_lower']) != lam or
            exact(energy['original_UE']) != UE or exact(energy['original_c']) != C or
            exact(energy['original_K']) != C*C/(2*UE)):
        raise ValueError('Energy graph belongs to another original law or Lambda')
    # q0<64, source denominator>=1/4. Generic operator-valued marks use the
    # entrywise triangle, NOT the scalar Ref0/4 CAR Frobenius shortcut.
    q = 6
    u = ceil_log2(UE); c_inverse = ceil_log2(1/C)
    contract: dict[str, Any] = dict(
        original_N=N, original_T=CLOCK, original_Rout=ROUT,
        original_UE=str(UE), original_c=str(C), selected_log2_lambda_lower=lam,
        packet='Unchanged selected SOFT scale1e-4, support |p|<=3e-7, same f/Z',
        reference='Original fullFock slow Href with actual neutral RIGHT; full EM/HC/normal common phase',
        Duhamel='U_C(t) B_f Xi - B_free(t) U_N(t) Xi = -i integral U_C(t,s) ([Href,B_free(s)]-B_hfree(s)-Omega(s)([Gref,B_free(s)]-B_gfree(s))) U_N(s) Xi ds',
        bounded_group='Original driven quintic trajectory only: compare bounded propagator/source products by this residual; no independent post-flight G flow and never apply unbounded G to state error',
        Weyl='||(W_delta-I)psi||<=2 ||delta|| sqrt(<Ncoeq+1>); d_edge^2<=16 UE/c, length2<=64 UE/c',
        harmonic='Peierls exp(+i length Ah) cancels exp[-i (x-Ref)Ah] away winding seams; winding factor kept at modulus<=2 and priced by the ORIGINAL free packet tail',
        finite_mean_source='Bfree is FULL original M_c(x)f_free(t,x), M_c=tanh(36)^ell_x, NOT ideal eta/Ppositive alone; its H/orbital commutators are explicitly paid',
        macrotheta='[UE/N ptheta^2,B]=-UE/N Delta_theta B-2i UE/N grad_theta B.ptheta; |(x-Ref)/N|<=sqrt3, no D/N erasure',
        Noether='Literal sine clover, FULL scalar_orbital(A), anchored rho.lambda_C and harmonic SOURCE derivative; lambda=-Lplus Ddagger rawFlow, no independent second density insertion',
        Coulomb='[HC,B]=B_(+/-2UE(G_x-G_Ref)rho+UE W_x), density on RIGHT; multipair retained in identity, only absolute norm payment below',
        mean_vertical='||Hvertical||<=43200 V Lambda^-1/2; retained explicitly',
        initial_Bq_to_B0_count=1, initial_Bq_to_B0_reused_from_A_Core=True,
        inherited_free_lattice_scope='Free f(t,s) on Duhamel RIGHT only; NOT interacting charged packet derivative',
        canonical_CAR='ALL8 statistical-flat P only; all64 colour aliases independent',
        Q='Encoded4096sqrtV source/64V angular prices and OWN truephot306a transfer remain inherited from Core; no physical projection',
        Born_relative_HC_admission=False, neutral_return_or_ground_required=False,
        common_IC='beta=Omega-Omega_start; neutral fast/ordinary phases and start scalar remain common, not reset',
        missing_prerequisites=missing, closed=not missing)
    if missing:
        contract['physical_halfspin_admission'] = False
        return contract
    number = int(energy['first_number_log2_upper'])
    pi = int(energy['harmonic_Pi_squared_log2_upper'])
    # Every edge phase costs <=8sqrt(UE/c)*sqrt(Ncoeq+1).
    # 2^16 V q0 dominates all4x4 marks, both maps, Z and paths of length1/2.
    weyl = (u+c_inverse+number+6+1)//2
    light = t+16+v+q+weyl
    orbit = 3+16+v+r+q+weyl
    # |sin curlA|<=|curlA|. At most12 oriented clover incidences, curl norm<=8.
    # Green<=N²/16, div<=4, |v|<=3Rout, |D/N|<=sqrt3. This loose triangle
    # covers BOTH anchored lambda_C and harmonic derivative, all sites/Ref.
    photon_gauss = 3+40+2*v+2*n+r+q+weyl
    # Hh source commutator: gradTheta<=sqrt3 B, DeltaTheta<=3 B.
    macro = t+u-n+1+16+v+q+max(0, (pi+1)//2)+3
    # HC<=UE N²(4V+32)/16, CB<=64sqrtV on proofP. This is
    # an explicitly absolute source error, never a linked scattering ratio.
    coulomb = t+u+2*n+v+6+(v+1)//2+3
    vertical = t+ceil_log2(43200)+v+7+(v+1)//2-lam//2
    # A harmonic winding seam is near N/2, NOT at the local plateau boundary.
    # Use the SAME frozen q6<2e60/free travel<2e8 at the smaller radius N/4.
    # This is a new far-seam payment, not T times the old plateau-tail estimate.
    far_tail = 2*F(2, N//4-200000000-2)**6*F(2*10**60)
    seam = t+16+q+ceil_log2(far_tail)
    # At each fixed theta the harmonic phase is a light-mode diagonal unitary,
    # so the original Ref0/4 CAR-Frobenius bound is uniform in theta. No V
    # triangle is needed for this scalar-coordinate multiplication operator.
    field_h = log_sum(light, macro, coulomb, vertical, seam)
    field_g = log_sum(orbit, photon_gauss)
    field_total = log_sum(field_h+2, field_g+1)
    screen = finite_mean_screen_bound()
    h = log_sum(field_h, screen['H_log2_upper'])
    g = log_sum(field_g, screen['G_log2_upper'])
    # Full original scalar marks on the RHS; retain4H+2G comparison factors.
    total = log_sum(h+2, g+1)
    inherited = F(core['NEW_fullencoded_bounded_angular_reprice']['full_receipt'][
        'conditional_OLD_bridge_plus_NEW_group_plus_Q_exact'])
    admitted = False
    if field_total < -100:
        field_cap = F(1, 2**100)  # positive cap for field residue ONLY
        cap = field_cap+F(screen['total_comparison_exact'])
        joined = inherited + cap
        admitted = joined < F(1,10000)
        finite = dict(new_interacting_residue_exact_cap=str(cap),
                      joined_exact_upper=str(joined),
                      joined_outward_upper=math.nextafter(float(joined), math.inf),
                      fixed_target=1e-4, margin_lower=float(F(1,10000)-joined))
    else:
        finite = dict(new_interacting_residue_exact_cap=None,
                      joined_exact_upper=None, joined_outward_upper=None,
                      fixed_target=1e-4, margin_lower=None)
    contract.update(
        energy_provenance=energy.get('provenance'),
        finite_mean_screen=screen,
        field_residue_only_log2_upper=field_total,
        field_residue_only_positive_cap='1/1267650600228229401496703205376' if field_total < -100 else None,
        harmonic_winding_far_tail_exact=str(far_tail),
        harmonic_winding_far_tail_radius=N//4-200000000-2,
        harmonic_winding_uses_ORIGINAL_q6_upper='2000000000000000000000000000000000000000000000000000000000000',
        independent_postflight_group_moment_claimed=False,
        residual_log2_upper=dict(Hlight_Weyl=light, full_orbital_A=orbit,
            photon_source_and_rho_lambda_C=photon_gauss,
            macrotheta_SOURCE_kinetic=macro, absolute_Coulomb=coulomb,
            mean_vertical_Gauss=vertical, harmonic_winding_seam=seam,
            H_integral=h, G_integral=g,
            bounded_packet_group_new_total=total),
        **finite,
        physical_halfspin_admission=admitted,
        massive_mobile='SAME ideal massive-band mobile envelope inside the FULL original mean source M_c f_free; lattice/group free costs contract under M_c and the additional M_c commutators are explicitly paid. No interacting invariant one-particle subspace is asserted.',
        physical_twopi_relation='At the SAME original pulse endpoint: U_C,K(T) B_f Xi is within the paid ledger of -B_free_massive(T) U_N,K(T) Xi; the minus is the derived Core physical2pi factor, not a reset of the evolving neutral',
        own_truephoton_unbounded_creator_transfer=False,
        fundamental_quantum_law_and_parameters_still_debt=True)
    return contract


def native_weyl_control() -> dict[str, Any]:
    # Finite oscillator is an independent adverse witness; the analytic bound
    # above is infinite Fock and does not infer a tail from this restriction.
    a = np.diag(np.sqrt(np.arange(1,64)),1).astype(complex)
    displacement = .055
    W = expm(displacement*(a.conj().T-a))
    psi = np.zeros(64,complex); psi[12]=1
    observed = float(np.linalg.norm((W-np.eye(64))@psi))
    bound = 2*displacement*math.sqrt(13)
    vacuum_reset = math.sqrt(2*(1-math.exp(-displacement**2/2)))
    if not observed <= bound or not observed > 3*vacuum_reset:
        raise AssertionError('Number-graph Weyl/vacuum-reset discriminator failed')
    return dict(number=12,displacement=displacement,actual_nonvacuum_norm=observed,
                first_number_bound=bound,wrong_vacuum_reset=vacuum_reset,
                vacuum_reset_rejected=True,finite_oscillator_scope=64)


def native_finite_mean_screen(model: Any) -> dict[str, Any]:
    """Resolve the genuine eta18 nonunitary scalar below binary64 epsilon."""
    import mpmath as mp
    from noncompact_screened_gauss_dressing import source_paths
    law=model.law;api=MeanPacketIntertwiner(model)
    lengths=np.array([len(path) for path in source_paths(law,model.reference_site)])
    with mp.workdps(100):
        c=mp.tanh(36);delta=1-c
        # M=I+delta Z, evaluated by expm1, not by subtracting rounded1.
        scaled=np.repeat([float(mp.expm1(int(ell)*mp.log(c))/delta) for ell in lengths],4)
        actual_delta=float(delta);delta_text=str(mp.nstr(delta,80))
    hopping_rows=np.zeros(law.volume)
    for x,y,coefficient,path in law.terms(model.chart.mean_phi,
                                        model.chart.mean_links,np.zeros((law.volume,3))):
        coefficient_norm=float(np.linalg.norm(coefficient,2))
        hopping_rows[x]+=coefficient_norm;hopping_rows[y]+=coefficient_norm
        if len(path)>2:
            raise AssertionError('Finite-scalar proof requires the ORIGINAL length<=2 stencil')
    if law.band.a!=1 or float(hopping_rows.max())>=8:
        raise AssertionError('Original mean hopping/spacing bound changed')
    raw=model.source_bank.seed_fibre.source_vectors_at(model.chart.mean_phi,
        model.chart.mean_links,np.zeros(3),np.zeros((law.volume,3)),
        model.source_bank.profiles['source19'],BETA)
    marks=Full108ReferenceMarks(raw['full108_plus_coefficients'],raw['full108_minus_coefficients'])
    weighted=Full108ReferenceMarks(scaled[:,None]*marks.plus,marks.minus*scaled[None,:])
    rows=[]
    for name,body in (('Hmean0',api.light(np.zeros((law.volume,3)))),
                      ('orbital0',scalar_orbital(law,model.chart.mean_links,
                       np.zeros((law.volume,3)),model.noether_bank.velocity))):
        action=api.onebody_action(marks,body)
        comparison=Full108ReferenceMarks(scaled[:,None]*action.plus,action.minus*scaled[None,:])
        hcomm=body*(scaled[None,:]-scaled[:,None])
        predicted=Full108ReferenceMarks(hcomm@marks.plus,marks.minus@hcomm)
        full=np.zeros((112,112),complex);full[:108,:108]=body
        errors=[];defects=[]
        for charge in (1,-1):
            T=transfer_matrix(weighted,charge)
            lhs=full@T-T@full-transfer_matrix(comparison,charge)
            rhs=transfer_matrix(predicted,charge)
            errors.append(float(np.linalg.norm(lhs-rhs)))
            defects.append(float(np.linalg.norm(rhs)))
        rows.append(dict(operator=name,scaled_commutator_identity_errors=errors,
            wrong_unit_mean_product_scaled_defects=defects,
            original_eta18_commutator_norms=[actual_delta*x for x in defects]))
    upper=F(finite_mean_screen_bound()['delta_upper_exact'])
    if F(delta_text)>upper or max(max(row['scaled_commutator_identity_errors']) for row in rows)>3e-7:
        raise AssertionError('Finite original eta18 scalar commutator control failed')
    if min(rows[0]['wrong_unit_mean_product_scaled_defects'])<=1e-5:
        raise AssertionError('Unit mean-product adverse control did not discriminate')
    return dict(original_eta=18,precision_digits=100,delta_1_minus_tanh36=delta_text,
        original_mean_hopping_block_row_upper=float(hopping_rows.max()),
        mean_path_lengths=lengths.tolist(),rows=rows,
        scope='Original eta18 resolved by high precision and exact delta-scaled native112 coefficient algebra; no source/profile/eta replacement')


def native_controls(model: Any, guard: dict[str, Any]) -> dict[str, Any]:
    """NEW native arbitrary-Ref commutator and covariant harmonic witnesses."""
    law=model.law; api=MeanPacketIntertwiner(model); rng=np.random.default_rng(22271007)
    y=rng.normal(size=len(model.em['omega']))*.43
    co=(model.em['legs']@y).reshape(law.volume,3)
    harmonic=np.array([.37,-.29,.23]); A=co+harmonic
    raw=model.source_bank.seed_fibre.source_vectors_at(model.chart.mean_phi,
        model.chart.mean_links,law.size*harmonic,A,
        model.source_bank.profiles['source19'],BETA)
    marks=Full108ReferenceMarks(raw['full108_plus_coefficients'],raw['full108_minus_coefficients'])
    flow=model.noether_bank.evaluate(y); velocity=model.noether_bank.velocity
    raw_direct=primitive_flow(law,model.chart.mean_phi,model.chart.mean_links,A,
                               velocity,np.zeros(law.volume))[2]
    lam=-model.green@model.noether_bank.gradient.T@raw_direct.ravel()
    lam-=lam[model.reference_site]
    bank_error=float(np.linalg.norm(lam-flow['lambda_C']))
    scalar=(law.sites-law.sites[model.reference_site])@flow['harmonic']/law.size
    orbital=scalar_orbital(law,model.chart.mean_links,A,velocity)
    # Native exact G SOURCE includes physical orbital and rho lambda_C.
    # Nonzero trueRef comes from the actual mean screened Gauss solve.
    from physical_core_spin_module import screened_noether
    solved=screened_noether(law,model.chart.mean_phi,model.chart.mean_links,A,
                            np.asarray(guard['alpha']),velocity,model.reference_site,
                            model.reference_frame)
    core=np.zeros_like(orbital)
    for x in range(law.volume): core[4*x:4*x+4,4*x:4*x+4]=solved['matrices'][x]
    ref=solved['Ref_RIGHT']; body=orbital+core+np.diag(np.repeat(lam,4))
    deriv=harmonic_source_jet(marks,law.sites-law.sites[model.reference_site],law.size,flow['harmonic'])
    action=api.onebody_action(marks,body,ref)
    exact=Full108ReferenceMarks(action.plus-1j*deriv.plus,action.minus-1j*deriv.minus)
    # -i derivative contributes -scalar for plus and +scalar for minus.
    target=api.onebody_action(marks,orbital+core+np.diag(np.repeat(lam-scalar,4)),ref)
    direction=flow['harmonic']; eps=2.**-17
    def at(theta: np.ndarray) -> Full108ReferenceMarks:
        shifted=A+(theta-law.size*harmonic)/law.size
        value=model.source_bank.seed_fibre.source_vectors_at(model.chart.mean_phi,
            model.chart.mean_links,theta,shifted,model.source_bank.profiles['source19'],BETA)
        return Full108ReferenceMarks(value['full108_plus_coefficients'],value['full108_minus_coefficients'])
    plus=at(law.size*harmonic+eps*direction);minus=at(law.size*harmonic-eps*direction)
    derivative_error=max(float(np.linalg.norm((plus.plus-minus.plus)/(2*eps)-deriv.plus)),
                         float(np.linalg.norm((plus.minus-minus.minus)/(2*eps)-deriv.minus)))
    gerror=max(float(np.linalg.norm(exact.plus-target.plus)),float(np.linalg.norm(exact.minus-target.minus)))
    # The physical macrotheta kinetic commutator on a nonconstant neutral
    # Gaussian. Native finite differences reach the cross term, not just B''.
    probe=np.array([.31,-.47,.53]);theta=law.size*harmonic
    jet=harmonic_source_jet(marks,law.sites-law.sites[model.reference_site],law.size,probe)
    ss=np.repeat((law.sites-law.sites[model.reference_site])@probe/law.size,4)
    second=Full108ReferenceMarks(-ss[:,None]**2*marks.plus,-marks.minus*ss[None,:]**2)
    psi0=math.exp(-float(theta@theta)/2);psip=-float(theta@probe)*psi0
    psipp=(float(theta@probe)**2-float(probe@probe))*psi0
    hstep=2.**-11
    pp=at(theta+hstep*probe);pm=at(theta-hstep*probe)
    wp=math.exp(-float((theta+hstep*probe)@(theta+hstep*probe))/2)
    wm=math.exp(-float((theta-hstep*probe)@(theta-hstep*probe))/2)
    macro_errors=[];macro_missing=[]
    for b,bp,bpp,sp,sm in ((marks.plus,jet.plus,second.plus,pp.plus,pm.plus),
                          (marks.minus,jet.minus,second.minus,pp.minus,pm.minus)):
        numeric=-model.ue/law.size*((sp*wp-2*b*psi0+sm*wm)/hstep**2-b*psipp)
        rhs=-model.ue/law.size*(bpp*psi0+2*bp*psip)
        macro_errors.append(float(np.linalg.norm(numeric-rhs)))
        macro_missing.append(float(np.linalg.norm(rhs)))
    # Actual improved-law terms independently expose every winding seam.
    displacement=law.sites-law.sites[model.reference_site]
    phase=np.exp(-1j*displacement@harmonic)
    diagonal=np.repeat(phase,4)
    hfree=(diagonal[:,None]*api.light(np.zeros_like(A)))*diagonal.conj()[None,:]
    hres=packet_residual(marks,api.light(A),hfree)
    relative_errors=[]
    for charge in (1,-1):
        whole=np.zeros((112,112),complex);whole[:108,:108]=api.light(A)-hfree
        T=transfer_matrix(marks,charge)
        relative_errors.append(float(np.linalg.norm(whole@T-T@whole-transfer_matrix(hres,charge))))
    seam_rows=[]; interior=[]
    zero=np.zeros_like(A)
    terms0=list(law.terms(model.chart.mean_phi,model.chart.mean_links,zero))
    termsh=list(law.terms(model.chart.mean_phi,model.chart.mean_links,np.tile(harmonic,(law.volume,1))))
    for (x,z,c0,path),(xx,zz,ch,_path) in zip(terms0,termsh):
        if x!=xx or z!=zz: raise AssertionError('Original path order changed')
        path_displacement=np.zeros(3)
        for _site,axis in path:path_displacement[axis]+=1
        winding=path_displacement-(displacement[z]-displacement[x])
        observed=ch*phase[z]-phase[x]*c0
        predicted=phase[x]*c0*(np.exp(1j*winding@harmonic)-1)
        error=float(np.linalg.norm(observed-predicted))
        if np.any(winding):seam_rows.append((error,float(np.linalg.norm(observed))))
        else:interior.append(float(np.linalg.norm(observed)))
    # Genuine small fullFock strong commutator on an EM-distorted multipair
    # vector, with true neutral RIGHT, NOT an expectation-density replacement.
    sites=[int(model.reference_site),int(law.target[model.reference_site,0])]
    ids=np.concatenate([np.arange(4*x,4*x+4) for x in sites])
    small=Full108ReferenceMarks(marks.plus[ids],marks.minus[:,ids])
    h=api.light(A)[np.ix_(ids,ids)];g=body[np.ix_(ids,ids)]
    HCaction=coulomb_packet_action(small,model.green[np.ix_(sites,sites)],model.ue,0)
    psi=rng.normal(size=4096)+1j*rng.normal(size=4096)
    # Retain a genuine neutral Ref4 multipair superposition on every even bulk
    # occupation; never substitute the interacting density by its mean.
    masks=np.arange(4096); bulk_even=np.array([(int(i)&255).bit_count()%2==0 for i in masks])
    psi[(masks>>8!=15)|~bulk_even]=0;psi/=np.linalg.norm(psi)
    rho=np.array([[sum((int(mask)>>(4*x+j))&1 for j in range(4))-2+
        (sum((int(mask)>>(8+j))&1 for j in range(4))-4 if x==0 else 0)
        for x in range(2)] for mask in masks],float)
    HC=model.ue*np.einsum('mi,ij,mj->m',rho,model.green[np.ix_(sites,sites)],rho)
    rows=[]
    for charge in (1,-1):
        state=psi if charge==1 else np.roll(psi,-(15<<8))
        T=transfer_matrix(small,charge);source=apply_onebody(T,state)
        hc=HC*source-apply_onebody(T,HC*state);expected=np.zeros_like(state);wrong=np.zeros_like(state)
        mean=np.einsum('m,mi->i',abs(state)**2,rho)
        for mask in np.flatnonzero(state):
            single=np.zeros_like(state);single[mask]=state[mask]
            expected+=apply_onebody(transfer_matrix(HCaction.on_density_eigenvector(rho[mask]),charge),single)
        wrong=apply_onebody(transfer_matrix(HCaction.on_density_eigenvector(mean),charge),state)
        errors={}
        for label,light,reference in (('H',h,np.zeros((4,4))),('G',g,ref)):
            full=np.zeros((12,12),complex);full[:8,:8]=light;full[8:,8:]=reference
            predicted=api.onebody_action(small,light,reference)
            lhs=apply_onebody(full,source)-apply_onebody(T,apply_onebody(full,state))
            rhs=apply_onebody(transfer_matrix(predicted,charge),state)
            errors[label]=float(np.linalg.norm(lhs-rhs))
            if label=='G':
                omitted=api.onebody_action(small,light)
                errors['wrong_Ref_omission']=float(np.linalg.norm(lhs-apply_onebody(transfer_matrix(omitted,charge),state)))
        rows.append(dict(charge=charge,HC_identity_error=float(np.linalg.norm(hc-expected)),
            wrong_HC_mean_density=float(np.linalg.norm(hc-wrong)),
            wrong_HC_half=float(np.linalg.norm(hc-expected/2)),fullFock_errors=errors))
    good=[bank_error,gerror,derivative_error,max(interior),max(x[0] for x in seam_rows),
          *macro_errors,*relative_errors]
    good.extend(row['HC_identity_error'] for row in rows)
    good.extend(row['fullFock_errors'][key] for row in rows for key in ('H','G'))
    if max(good)>3e-7 or any(row['wrong_HC_mean_density']<1e-7 or row['fullFock_errors']['wrong_Ref_omission']<1e-7 for row in rows):
        raise AssertionError('Native interacting/Ref/covariant-harmonic discriminator failed')
    return dict(full108_Ref4=True,source_profile='Literal original native SOURCE19; not new LEGAL SOFT data',
        original_sine_flow_norm=float(np.linalg.norm(raw_direct)),anchored_lambda_sign_error=bank_error,
        macrotheta_source_derivative_error=derivative_error,source_G_exact_jet_error=gerror,
        macrotheta_kinetic_commutator_errors=macro_errors,
        wrong_drop_D_over_N_commutator_defects=macro_missing,
        Hfull108_physical_minus_covariant_free_residual_errors=relative_errors,
        Hfull108_actual_interacting_residual_norms=[float(np.linalg.norm(hres.plus)),float(np.linalg.norm(hres.minus))],
        missing_macrotheta_defect=float(np.linalg.norm(deriv.plus)),
        harmonic_interior_error=max(interior),winding_identity_error=max(x[0] for x in seam_rows),
        wrong_drop_seam_defect=max(x[1] for x in seam_rows),small_fullFock_rows=rows,
        scope='NEW exact native12mode fullFock/108mark witnesses; uniform LEGAL packet cost comes from analytic bound, not native spatial mass')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-npz','cache-manifest','initial-receipt','core-capsule',
                 'harmonic-receipt','reference-capsule','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();start=time.monotonic()
    inputs={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (args.cache_npz,args.cache_manifest,args.initial_receipt,args.core_capsule,
             args.harmonic_receipt,args.reference_capsule)}
    if inputs[str(args.initial_receipt)]!=INITIAL_SHA or inputs[str(args.core_capsule)]!=CORE_SHA:
        raise ValueError('Exact inherited original initial-energy and Core capsules required')
    if inputs[str(args.harmonic_receipt)]!=HARMONIC_SHA or inputs[str(args.reference_capsule)]!=REFERENCE_SHA:
        raise ValueError('Exact original harmonic graph and actual full-Fock reference required')
    initial=json.loads(args.initial_receipt.read_bytes());core=json.loads(args.core_capsule.read_bytes())
    from driven_slow_field_energy import driven_energy_bound
    harmonic=json.loads(args.harmonic_receipt.read_bytes())
    reference=json.loads(args.reference_capsule.read_bytes())
    energy=driven_energy_bound(initial,harmonic,reference)
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    argv=data['manifest']['argv'];guard_path=Path(argv[argv.index('--noether-guard')+1])
    guard=json.loads(guard_path.read_bytes())
    inputs[str(guard_path)]=hashlib.sha256(guard_path.read_bytes()).hexdigest()
    epoch=loaded_source_epoch();epoch[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    omitted=dict(energy,closed=False,missing_prerequisites=['actual first-number graph unavailable'])
    omitted_result=driven_packet_bound(omitted,initial,core)
    large=dict(energy,first_number_log2_upper=10000)
    large_result=driven_packet_bound(large,initial,core)
    wrong_law=dict(energy,original_c='1')
    wrong_rejected=False
    try:
        driven_packet_bound(wrong_law,initial,core)
    except ValueError:
        wrong_rejected=True
    if (omitted_result['physical_halfspin_admission'] or
            large_result['physical_halfspin_admission'] or not wrong_rejected):
        raise AssertionError('Missing/enlarged-number/wrong-law source bound must not admit')
    result=dict(plan=PLAN,scientific_execution_epoch_revision=REVISION,
        analytic=driven_packet_bound(energy,initial,core),energy=energy,
        native=native_controls(model,guard),Weyl=native_weyl_control(),
        finite_mean_screen_native=native_finite_mean_screen(model),
        adverse_API=dict(missing_number_graph_retained=omitted_result['missing_prerequisites'],
            enlarged_number_graph_log2=10000,
            enlarged_number_residual_log2=large_result['residual_log2_upper']['bounded_packet_group_new_total'],
            enlarged_number_admitted=large_result['physical_halfspin_admission'],
            wrong_photon_c_rejected=wrong_rejected),
        input_sha256=inputs,loaded_sources=epoch,program_wall_seconds=time.monotonic()-start,
        source_sha256=epoch[Path(__file__).name],finished_supplier_science_rerun=False)
    after=loaded_source_epoch();after[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['loaded_sources_stable']=all(after.get(key)==value for key,value in epoch.items())
    result['input_bytes_stable']=all(hashlib.sha256(Path(key).read_bytes()).hexdigest()==value for key,value in inputs.items())
    if not result['loaded_sources_stable'] or not result['input_bytes_stable']:
        raise ValueError('Scientific source/input epoch changed during execution')
    text=json.dumps(portable(result),sort_keys=True)
    args.output.write_text(text);os.chmod(args.output,0o600);print(text)


if __name__=='__main__':
    main()

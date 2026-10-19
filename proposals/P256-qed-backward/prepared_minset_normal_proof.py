#!/usr/bin/env python3
"""Exact endpoint-centre mechanism, legal-volume local normals and native controls.

No ground-state or full SOURCE/Agmon transfer claim is made by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import eigh, null_space

from original_charged_principal import (
    OriginalSpatial, original_frame, endpoint_average_controls, portable, plaquette_hessian,
)
from noncompact_quantitative_normal_budget import (
    native_normal_budget, sqrt_lower, sqrt_upper, ceil_log2, coefficient_design,
)
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis
from unrestricted_field_minimum import coordinates
from improved_phi_band import C, NSTAR, UE, selected_band

ROOT = Path(__file__).resolve().parent
IDENTITY = 'PNFF-MinSet-exact-endpoint-centre-local-normal-trial'
PLAN = 'issue-222-fullfield-closedrotation-continuation-20261006'
PLAN_REVISION = '7954ba4c18ed7d90fdc3e6cbd3bf75c3dc80d568cae4781d73dadd61e041a718'


def local_normal_constants(size: int = NSTAR) -> dict[str, Any]:
    """Rational full-q Hessian and explicit finite-chart/trial majorants.

    The exact finite-N centre is an implicit finite sum, never G_infinity.
    Domain is a regular local quotient tube, not the entire field space.
    """
    b = native_normal_budget(size)
    V, d = size**3, 105*size**3
    g, upper = b.gram.lower, b.gram.upper
    # Reuse the weaker gap2 certificate, rather than the older gap4 convention.
    cx2 = F(23, 25)**2 + 3*F(3, 10)**2
    cx = sqrt_upper(cx2)
    rmax = sqrt_upper(upper*cx2)
    eta = coefficient_design()['richness_lower']
    alpha = 2*g*g/upper*min(F(2), eta/(2*rmax**3))
    stiff = F(1, 5)*min(F(16, size**2), F(3, 10)*g)
    leading = b.leading_upper
    schur = 4*(leading+2*leading*leading/alpha)/stiff
    delta = 2*F(1, 5)*sqrt_lower(g)
    rho = sqrt_upper(4*V*upper)
    s = min(F(1, 4), sqrt_lower(g/2), sqrt_lower(stiff), delta/8)
    h = F(2**20)*d*d*(1+rho+cx)
    chart = (h/s)**256
    # Includes quotient half-density, all raw matter representations and Kato.
    compare = F(2**32)*d**8*chart**8*(1+1/alpha+1/stiff)**8*(4*V+3)**8
    radius = min(F(1, 64), min(alpha, stiff, delta)/(2**20*d*compare), delta/(64*V*h))
    nonlinear_schur = 16*(compare+2*compare*compare/alpha)/stiff
    covariance = 4*d*(1+1/sqrt_lower(alpha)+1/sqrt_lower(stiff))
    zero_point = 90*V*sqrt_upper(leading+10+upper)
    excess = 16*(zero_point+compare*(1+covariance)**4+V*size**2+1)
    cutoff_constant = 2**24*d*chart**2*(1+1/radius)**4
    # Gaussian radial cutoff tail exp(-radius^2 Lambda^(3/4)/(512 covariance)).
    # Its inverse-power bound pays every potential/kinetic cutoff contribution.
    # j=16 suffices: tail <=16! x^-16 and Lambda^(5/4)*Lambda^-12.
    tail = F(math.factorial(16))*cutoff_constant*(512*covariance/radius**2)**16
    required_log2_lambda = max(
        4*ceil_log2(schur), 4*ceil_log2(nonlinear_schur), 4*ceil_log2(64*compare),
        2*ceil_log2(4096*covariance/radius**2),
        ceil_log2(64*tail), 2*ceil_log2(64*excess),
    )
    # The selected native tower is read from the existing legal-budget authority.
    selected_log2_lower = 2**315
    if required_log2_lambda >= selected_log2_lower:
        raise AssertionError('Selected unchanged tower does not pay these local constants')
    light_bound = 2*(2*V+1)*cx*(rho+radius)
    return dict(
        size=size, volume=V, unreduced_dimension=d, physical_normal_dimension=90*V,
        soft_dimension=45, stiff_dimension=90*V-45,
        gram_lower=g, gram_upper=upper, uniform_alpha=alpha, all_q_stiff=stiff,
        endpoint_hessian_norm_upper=leading, heavy_gap=delta,
        nonlinear_local_onset=nonlinear_schur,
        quotient_chart_jet_upper=chart, quotient_radius=radius,
        quotient_cometric_lower=F(1, 4), quotient_cometric_upper=F(4),
        quotient_half_density_included=True, finite_chart_compare=compare,
        gaussian_covariance_majorant=covariance, zero_point_upper=zero_point,
        Eex_coefficient=excess, Eex_exponent=F(3, 8),
        cutoff_tail_coefficient=tail, required_log2_lambda=required_log2_lambda,
        selected_log2_lambda_conservative_lower=selected_log2_lower,
        selected_tower_pays_local_onset=True,
        floor_light_subtraction=light_bound, floor_beta_subtraction=F(1, 16),
        exact_centre='finiteN full10 Gram minimizer, Clifford lift, central flat links; endpoint colour derivative identically zero',
        fullRoot_floor='Lambda min_s v(m,s) - floor_light_subtraction - 1/16',
        Ecut='Lambda min_s v(m,s) + 8 Eex_coefficient Lambda^(3/8)',
        unchangedRoot_window_admission='8 Eex_coefficient Lambda^(3/8) <= V Lambda; original Utrial >12V Lambda',
        global_field_coercivity_proved=False, source_created_trial_bound_proved=False,
        whole_H_boundary_leakage_proved=False,
        scope='local full90V quotient and neutral correlated trial; not SOURCE/current/whole-domain Agmon',
    )


def native_centre_probe(size: int = 5) -> list[dict[str, Any]]:
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    basis = matrix_unit_basis()
    rows = []
    for label, angles in (
        ('skew-theta001-quarter', np.array([0., 0., np.pi/2])),
        ('symmetric-control', np.zeros(3)),
    ):
        selected, frame = original_frame(band, size, angles, True)
        model = OriginalSpatial(band, size, basis, coordinates(frame, basis).ravel(), angles)
        gradient = endpoint_average_controls(model)
        # Alignment and magnetic are squared residuals at a central flat link.
        alignment_first = np.zeros((3, 15))
        magnetic_first = np.zeros((3, 15))
        total = np.asarray(gradient['actual_endpoint_averaged_link_gradient'])
        total += alignment_first+magnetic_first
        if np.linalg.norm(total) > 1e-10:
            raise AssertionError('Actual Clifford endpoint colour centre is not stationary')
        rows.append(dict(
            case=label, size=size, centre_angles=angles, actual_full10_gram=selected,
            actual_full60_field_gradient_norm=float(np.linalg.norm(model.gradient[:60])),
            all45_colour_first_variations=total,
            alignment_first_variation=alignment_first, magnetic_first_variation=magnetic_first,
            native_scope='125 momentum blocks/500 onebody modes, original total top250 occupied sea',
            exact_colour_stationarity=True, **gradient,
        ))
    if not rows[0]['wrong_unsym_vertex_discriminates_at_this_input']:
        raise AssertionError('Skew positive adverse endpoint control did not discriminate')
    return rows


def native_full_q_probe(size: int = 5, tower_scale: float = 1e8) -> dict[str, Any]:
    """All discrete modes of one native original Hessian, not an all-N proof."""
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    basis = matrix_unit_basis()
    angles = np.array([0., 0., np.pi/2])
    _, frame = original_frame(band, size, angles, True)
    model = OriginalSpatial(band, size, basis, coordinates(frame, basis).ravel(), angles)
    minima, gauge_errors = [], []
    for mode in np.ndindex(size, size, size):
        q = 2*np.pi*np.asarray(mode)/size
        parts = model.hessian_parts(q)
        heavy = parts['local']+parts['sea']+parts['contact']
        alignment = model.alignment(q)
        H = heavy+tower_scale*alignment
        gauge = model.gauge_columns(q)
        horizontal = null_space(gauge.conj().T)
        eigenvalues = np.linalg.eigvalsh(horizontal.conj().T@H@horizontal)
        minima.append(float(eigenvalues[0]))
        gauge_errors.append(float(np.linalg.norm(H@gauge)/max(1., np.linalg.norm(H))))
    if min(minima) <= 0 or max(gauge_errors) > 1e-8:
        raise AssertionError('Native all-q horizontal positivity/Ward failed')
    return dict(size=size, modes=size**3, physical_modes_per_q=90,
                minimum_horizontal_hessian=min(minima), maximum_relative_Ward=max(gauge_errors),
                Lambda_quarter=tower_scale, colour_cometric=.6,
                omitted_positive_magnetic_term=True, all_N_claim=False)


def riccati_probe() -> dict[str, Any]:
    """A real correlated oscillator with noncommuting cometric/Hessian."""
    G = np.diag([1., .6, .6])
    K = np.array([[3., .4, -.2], [.4, 7., .7], [-.2, .7, 11.]])
    rootG = np.diag(np.sqrt(np.diag(G)))
    invroot = np.diag(1/np.sqrt(np.diag(G)))
    ev, vec = np.linalg.eigh(rootG@K@rootG)
    A = invroot@(vec*np.sqrt(ev))@vec.T@invroot
    covariance = np.linalg.inv(A)/2
    energy = (np.trace(G@A)+np.trace(K@np.linalg.inv(A)))/4
    exact = np.sqrt(ev).sum()/2
    fourth = float(np.trace(covariance)**2+2*np.trace(covariance@covariance))
    residual = float(np.linalg.norm(A@G@A-K))
    wrong = np.diag(np.sqrt(np.diag(K)/np.diag(G)))
    wrong_energy = float((np.trace(G@wrong)+np.trace(K@np.linalg.inv(wrong)))/4)
    if residual > 1e-12 or abs(energy-exact) > 1e-12 or wrong_energy <= energy:
        raise AssertionError('Correlated Riccati energy/adverse control failed')
    return dict(precision=A, covariance=covariance, Riccati_residual=residual,
                actual_energy=float(energy), exact_zero_point=float(exact),
                fourth_radial_moment=fourth, diagonal_trial_excess=wrong_energy-energy,
                scope='noncommuting covariance algebra; not native full-H energy execution')


def hex_fraction_log2_upper(encoded: dict[str, str]) -> int:
    """Read exact receipt bounds without recomputing huge Fraction gcds."""
    numerator = int(encoded['numerator_hex'], 16)
    denominator = int(encoded['denominator_hex'], 16)
    power = numerator.bit_length()-denominator.bit_length()
    fits = numerator <= denominator << power if power >= 0 else numerator << -power <= denominator
    return power if fits else power+1


def union_barrier_ledger(normal: dict[str, Any], time: F = F(6),
                         energy_coefficient_log2: int | None = None,
                         source_norm_log2: int = 0, source_first_log2: int = 0,
                         source_lower: F = F(1), target_bits: int | None = None) -> dict[str, Any]:
    """Operator-norm IMS shells: no rank, trace, eigenvector or ground premise.

    Actual full local form floor and the common neutral/charged energy cap must
    be bound by A. Inputs are encoded exact local_normal_constants receipt data.
    This ledger does not admit a driven H-omega G window.
    """
    if source_lower <= 0:
        raise ValueError('The actual source lower norm must be positive')
    d = normal['physical_normal_dimension']
    count = 512
    alpha = hex_fraction_log2_upper(normal['uniform_alpha'])-1
    stiff = hex_fraction_log2_upper(normal['all_q_stiff'])-1
    kappa_log = min(alpha, stiff)-4
    r0_log = hex_fraction_log2_upper(normal['quotient_radius'])-7
    chart_log = hex_fraction_log2_upper(normal['quotient_chart_jet_upper'])
    # Actual centre clustering inside regular r, not inside shrinking R_out.
    alias_power = 2**20
    alias_x_log_lower = NSTAR.bit_length()-1-20
    alias_log = 32+ceil_log2(NSTAR**3)+chart_log+alias_power*(20-alias_x_log_lower)
    regular_r_lower = hex_fraction_log2_upper(normal['quotient_radius'])-1
    if alias_log >= regular_r_lower-8:
        raise AssertionError('Actual shifted-grid centres do not fit the common regular tube')
    field_energy_log = hex_fraction_log2_upper(normal['Eex_coefficient'])
    common_log = field_energy_log if energy_coefficient_log2 is None else energy_coefficient_log2
    cut_log = common_log+3
    # R=r0 Lambda^-1/4; B>=kappa Lambda R^2/2^10.
    barrier_log = kappa_log+2*r0_log-10
    # n=floor(a Lambda^1/4); each nested IMS contraction is <=1/4.
    action_log = kappa_log//2+2*r0_log-16-ceil_log2(count)
    gradient_log = 16+ceil_log2(count)+2*chart_log-r0_log
    laplacian_log = 32+2*ceil_log2(count)+2*ceil_log2(d)+4*chart_log-2*r0_log
    commutator_log = max(laplacian_log, gradient_log+(cut_log+1)//2)+8
    mixed_log = max(source_norm_log2+commutator_log,
                    source_first_log2+gradient_log)+4
    amplification_log = max(0, ceil_log2(1/source_lower))
    time_log = max(0, ceil_log2(max(F(1), abs(time))))
    bits = ceil_log2(NSTAR**3*2**40/UE)+4096 if target_bits is None else target_bits
    # Tail <=2^[-a Lambda^1/4/2]; absorb Lambda^(3/16) and time/source.
    selector = max(128, 8*(2-action_log),
                   8*max(0, cut_log-barrier_log+2),
                   16*(mixed_log+amplification_log+time_log+bits+4))
    selected_lower = normal['selected_log2_lambda_conservative_lower']
    if selector >= selected_lower:
        raise AssertionError('Unchanged tower does not pay the explicit shell selector')
    return dict(
        identity='MinSet-union-operator-IMS-shells-no-counting', centres=count,
        physical_normal_dimension=d, kappa_log2_lower=kappa_log,
        actual_centre_clustering_log2_upper=alias_log,
        regular_r_log2_lower=regular_r_lower,
        colour_holonomy_separation_log2_lower=-16-ceil_log2(NSTAR)-chart_log,
        r0_log2=r0_log, R_out='r0 Lambda^(-1/4)',
        barrier_log2_coefficient_lower=barrier_log, barrier_Lambda_power='1/2',
        common_energy_coefficient_log2=common_log, Ecut_Lambda_power='3/8',
        action_log2_lower=action_log, shell_steps='floor(a Lambda^(1/4))',
        shell_operator_bound='||1_shell P_[0,Ecut]|| <= 2^(-floor(a Lambda^(1/4)))',
        cutoff_gradient_coefficient_log2=gradient_log,
        cutoff_Laplacian_coefficient_log2=laplacian_log,
        cutoff_commutator_coefficient_log2=commutator_log,
        cutoff_commutator_Lambda_power='3/16',
        mixed_source_commutator_coefficient_log2=mixed_log,
        source_norm_log2=source_norm_log2, source_first_log2=source_first_log2,
        supplied_source_lower=str(source_lower), time=str(time), target_bits=bits,
        required_log2_Lambda=selector, unchanged_selected_log2_Lambda_lower=selected_lower,
        spectral_rank_or_count_not_used=True, arbitrary_EM_and_harmonic_spectators=True,
        all_Bloch_fibres_and_paired_kappa=True, full_Root_space_not_projected=True,
        centre_separation_not_assumed=True,
        actual_sector_inputs_bound=False,
        common_energy_coefficient_supplied=energy_coefficient_log2 is not None,
        required_bindings=[
            'A actual whole local matrix form H_D-e_m >= normal kinetic + kappa Lambda dist_union^2; fullRoot/min_s floor and all light/ref/photon/harmonic terms',
            'actual neutral or charged P_[0,Ecut] and full-sector common Eex coefficient',
            'actual source norm lower and covariant normal first jet, not an assumed creator lower',
            'charged source spectral tail from actual commutator/higher-source graph if B P_neutral is not in charged window',
        ],
        driven_H_minus_omega_G_admitted=False,
        driven_requirement='Actual time-uniform shell/kinetic/source graph for driven flow, or separate SOURCE-relative driven comparison; ordinary spectral invariance is not inherited',
    )


def raw_correlated_trial_ledger(normal: dict[str, Any]) -> dict[str, Any]:
    """Exact Gaussian/Wick constants for A's SAME raw-trial quasimode.

    Momentum constants refer to the scalar half-density chart Gaussian.
    Metric and whole-heavy/light-frame connections enter the residual majorant,
    not an invented diagonal covariance or a whole-H vacuum claim.
    """
    soft_dimension = normal['soft_dimension']
    stiff_dimension = normal['stiff_dimension']
    alpha_floor = hex_fraction_log2_upper(normal['uniform_alpha'])-1
    stiff_floor = hex_fraction_log2_upper(normal['all_q_stiff'])-1
    heavy_upper_log = hex_fraction_log2_upper(normal['endpoint_hessian_norm_upper'])
    zero_point_log = hex_fraction_log2_upper(normal['zero_point_upper'])
    covariance_log = hex_fraction_log2_upper(normal['gaussian_covariance_majorant'])
    compare_log = hex_fraction_log2_upper(normal['finite_chart_compare'])
    chart_log = hex_fraction_log2_upper(normal['quotient_chart_jet_upper'])
    soft_covariance_log = ceil_log2(soft_dimension)+1+(-alpha_floor+1)//2
    stiff_covariance_log = ceil_log2(stiff_dimension)+1+(-stiff_floor+1)//2
    soft_momentum_log = ceil_log2(soft_dimension)+(heavy_upper_log+1)//2+4
    stiff_momentum_log = zero_point_log+4
    moment_log = max(covariance_log, soft_momentum_log, stiff_momentum_log)+2
    # Eighth Wick moments cover quartic norms, and sixth mixed moments cover
    # xi² z and z p_xi². The contracted compare includes metric/frame jets.
    residual_log = 65+compare_log+4*moment_log
    cutoff_log = hex_fraction_log2_upper(normal['cutoff_tail_coefficient'])+2*compare_log+64
    cutoff_selector = (cutoff_log+65)//2
    selected = normal['selected_log2_lambda_conservative_lower']
    if cutoff_selector >= selected:
        raise AssertionError('Unchanged tower does not pay raw operator cutoff residual')
    return dict(
        identity='same-raw-correlated-full90V-field-trial-quasimode-constants',
        raw_preparation='all64 m amplitudes1/8, actual finiteN min_s normalizedties; (Sea Ref4 + Sea+4orthogonal positive spectators Ref0)/sqrt2; no ground imposed',
        actual_quadratic='Lambda*(LOCAL+SEA+CONTACT)+Lambda^1.25*(alignment+plaquette(.7)), all90V',
        precision='A Gkin A=Kactual; Gkin includes chi, so mass-normal A=sqrt(Kactual/chi)',
        cutoff='unchanged even radial: 1 inside r/3, support<=2r/3, half-density included',
        soft_dimension=soft_dimension, stiff_dimension=stiff_dimension,
        uncut_position_trace_coefficients_log2=dict(soft=soft_covariance_log, stiff=stiff_covariance_log),
        cut_position_even_moments='<=4(2k-1)!! trace(Sigma)^k, k<=4; soft Lambda^-3k/4, stiff Lambda^-7k/8',
        scalar_cut_momentum_trace_coefficients_log2=dict(soft=soft_momentum_log, stiff=stiff_momentum_log),
        scalar_momentum_trace_exponents=dict(soft='3/4', stiff='7/8'),
        gaussian_position_momentum_cross='expectation y_i p_j=i delta_ij/2; correlated covariance retained',
        uniform_Wick_majorant_log2=moment_log, Wick_orders='through8; mixed contractions retained',
        forward_chart_jets=f'q! 2^({chart_log})^(q+1), q<=8; ambient displacement<=4 J^2 |y|',
        metric_density_frame_compare_log2=compare_log,
        stiff_flat_identity='geometric uniform-flat alignment kernel retained; do not erase one-stiff/two-soft cubic in the actual trial chart',
        field_operator_residual_log2=residual_log, field_operator_residual_Lambda_exponent='1/16',
        residual_scalings=dict(raw_chart_one_stiff_two_soft='1/16', adapted_two_stiff_one_soft='0',
                               metric_soft_pstiff_squared='0', heavy_cubic='-1/8', Berry_pstiff='-1/16'),
        normal_filter_residual_exponent='-5/16 for L proportional to Lambda^3/8',
        cutoff_operator_residual=f'<=2^({cutoff_log}) Lambda^-2',
        cutoff_selector=cutoff_selector, unchanged_selected_log2_Lambda_lower=selected,
        scalar_reference='Lambda v_ms + exact full90V E0_ms + bank-dependent light/spectator/Coul/harmonic/photon references; retain all offsets and phases',
        physical_preparation_filter='ONE f((H_D-e_global)/L), NOT separate physical f_m or bank filters',
        shadow_weights='(1/8)/sqrt(2*tiecount_m) times f((mu_ms,b-e_global)/L), b=Ref4 or Ref0+four spectators',
        fixed_reference_charge='qRef=nRef-4 in BOTH banks; no charge reset',
        one_SOURCE_branch_bank_factor='1/sqrt2; plus uses Ref4 bank, minus uses Ref0 bank',
        local_graph_usage='direct-sum H_D(m)-e_m may control proof graphs; never replace ONE e_global filter or whole-H mixing',
        owner_remaining_remainders='A adds actual light/Ref/Coul/photon/harmonic/beta residuals and scalar reference',
        global_prepared_SOURCE_admission_proved=False,
    )


def global_driven_phi_moment_ledger(normal: dict[str, Any], time: F,
                                    outer_radius: F,
                                    initial_energy_log: int | None) -> dict[str, Any]:
    """Whole-field energy proof, not transfer of a Dirichlet norm error.

    C_E0 is the actual normalized SOURCE relative-energy coefficient, supplied
    by its preparation owner. No absolute alignment floor is substituted.
    """
    from prepared_neutral_full_field import l1
    from bosonic_original_law_join import quartic_vacuum_coefficients_from_l1
    from unrestricted_field_minimum import BARE

    if time <= 0 or outer_radius <= 0:
        raise ValueError('Positive physical clock and support radius are required')
    V = normal['volume']
    heavy_l1 = 2*l1(RotationBand(**BARE))
    light_l1 = 2*l1(selected_band()[0])
    a, b, cc = quartic_vacuum_coefficients_from_l1(V, F(1), heavy_l1, light_l1)
    Bhat = (a+b+cc)/V+2**80
    CR = 2**16*(1+outer_radius)
    peak = F(165, 14)/time  # pi <=22/7, max w=15/8.
    speed_ratio = peak*outer_radius/C
    # The actual surface has the stronger c/4 guard. This rational envelope
    # is deliberately wider and does not change its clock/profile.
    if speed_ratio > F(65, 256):
        raise AssertionError('Whole-field energy envelope exceeds its speed guard')
    if CR*peak >= 2**16:
        raise AssertionError('Boost absorption coefficient exceeds 2^80')
    selector = max(0, 4*ceil_log2(512*peak*CR))
    selected = normal['selected_log2_lambda_conservative_lower']
    if selector >= selected:
        raise AssertionError('Selected tower does not pay global moment variation')
    bhat_log = ceil_log2(Bhat)
    coefficient_log = None
    if initial_energy_log is not None:
        # 64 sqrt(C_E0+Bhat+1), without allocating 2**C_E0.
        coefficient_log = 6+(max(initial_energy_log, bhat_log, 0)+3)//2
    return dict(
        identity='actual-whole-field-driven-Phi-second-moment-energy-lemma',
        actual_clock=time, actual_outer_radius=outer_radius,
        actual_heavy_l1_outward=heavy_l1, actual_selected_light_l1_outward=light_l1,
        quartic_BN_coefficients=dict(a=a, b=b, c=cc),
        CR=CR, Bhat=Bhat, actual_quintic_peak_upper=peak,
        rational_speed_ratio_upper=speed_ratio,
        EM_positive_form_fraction_lower='5/8',
        global_quartic_form='H_R >=Tphi+Tcolour+H_EM+Lambda*rho^4/(8V)-B_N-beta; rho^2=Tr(Phi^2)/4',
        Noether_global_bounds='|Gphi|<=CR*rho*sqrt(2Tphi/chi); |Gcolour|<=CR*sqrt(V)*sqrt(2Tcolour/chi); |Gorb|<=CR*V; |GEM|<=sqrt2*Rout*H_EM/c',
        driven_coercivity='H_R-Omega G >=Tphi/2+Tcolour/2+(5/8)H_EM+Lambda*rho^4/(16V)-Bhat*Lambda*V',
        retained_energy_reference='H_R has no added scalar; if preparation adds Cscalar, use H-Cscalar in BOTH initial energy and evolution; retain every m/s Lambda floor offset and ordinary J term',
        shifted_energy='S=<H_R-Omega G>+Bhat*Lambda*V >=0',
        exact_energy_derivative="Sprime=-Omega_prime*<G>; no commuting rotation/H assumption",
        normalized_differential_inequality='F=S/(Lambda V); |Fprime|<=|Omega_prime|*(4CR Lambda^-1/4 F^3/4+3CR Lambda^-1/4 F^1/2+CR/Lambda+sqrt2 Rout F/((5/8)c))',
        quintic_total_variation='int|Omega_prime|=2Omega_peak; int|Omega Omega_prime|=Omega_peak^2',
        integrated_energy_bound='S(t)<=8*(C_E0+Bhat+1)*Lambda*V',
        whole_Phi_trace_bound='sup_t <sum_x Tr Phi_x^2> <=64 sqrt(C_E0+Bhat+1)*V',
        whole_Phi_trace_coefficient_log2_upper=coefficient_log,
        actual_initial_SOURCE_energy_coefficient_log2=initial_energy_log,
        required_log2_Lambda=selector, unchanged_selected_log2_Lambda_lower=selected,
        initial_energy_binding_owner='PreparedNeutralTransfer',
        no_Dirichlet_error_moment_transfer=True,
        no_weighted_photon_cubic_graph_claim=True,
        initial_energy_coefficient_argument_supplied=initial_energy_log is not None,
    )


def fast_graph_boundary_ledger(normal: dict[str, Any], driver: dict[str, Any],
                               time: F, target_bits: int) -> dict[str, Any]:
    """Second fast graph and boosted IMS on the actual regular union.

    This is an unweighted normal boundary proof; it never commutes a photon
    weight through fn, SOURCE, or the boundary evolution.
    """
    Cslow_log = driver['contracted_fast_constant_log2']
    radius_log = hex_fraction_log2_upper(normal['quotient_radius'])-1
    coercivity_log = min(hex_fraction_log2_upper(normal['uniform_alpha']),
                         hex_fraction_log2_upper(normal['all_q_stiff']))-8
    initial_graph_log = max(driver['driven_common_window_coefficient_log2'],
                            Cslow_log)+16
    peak_log = ceil_log2(F(165, 14)/time)
    time_log = max(0, ceil_log2(time))
    # sqrt(F) variation is polynomial, not exp(Cslow*T). Selected Lambda
    # absorbs the suppressed T Lambda^-1/2 term before the square.
    graph_log = 2*max((initial_graph_log+1)//2,
                     Cslow_log+max(0, peak_log)+8)+4
    shell_log = (graph_log-coercivity_log-2*radius_log+3)//2+4
    boundary_log = (graph_log+shell_log+3)//2-2*radius_log+32
    selector = max(2*(Cslow_log+time_log+16),
                   8*(boundary_log+time_log+target_bits+8))
    selected = normal['selected_log2_lambda_conservative_lower']
    if selector >= selected:
        raise AssertionError('Unchanged tower does not pay second-fast-graph IMS')
    return dict(
        identity='actual-regular-union-second-fast-graph-boosted-Caccioppoli',
        Kfast='Hfast(includes heavy,allRoot,beta)-Omega Gfast-e_global+Cdrive Lambda^1/2 >=Lambda^1/2+chi/4 Pboost^2+kappa Lambda distance_union^2',
        scalar_floor_only='ONE e_global; no S-dependent Lambda v_s subtraction, since beta need not commute it',
        exact_EM_commutation='Kfast commutes H_EM-Omega G_EM: heavy neutral, Phi/colour primitives and local SU4 body preserve rho, betaZ preserves rho',
        initial_second_graph='Kfast0 and H_EM are positive commuting; ||Kfast0 fn tau|| <=||(Kfast0+H_EM)fn tau|| <=(2L+CL+Cdrive Lambda^1/2)||fn tau||',
        actual_Gamma_SOURCE_initial_graph='Gamma adds chi*dGamma*P and chi*d2Gamma; SOURCE adds chi*DB*P+chi*D2B+Omega DYB+[beta,B]; use actual global SOURCE norm >=1/32',
        slow_operator='hLight-Omega GorbLight; any heavy orbital generator belongs INSIDE Gfast, not a fictitiously bounded [GorbHeavy,Hheavy]',
        slow_commutator='[hLight,Hheavy]=0 on full ambient; bounded normal derivatives of actual Phi/U coefficients and unitary Peierls words; [hLight,beta] retained',
        exact_graph_derivative='Dfast=-Omega_prime Gfast+i[hLight-Omega GorbLight,Kfast]',
        Dfast_norm_bound='Cslow*(|Omega_prime| Lambda^1/4+Lambda^-1/4)*||Kfast^1/2 psi||+Cslow*(|Omega Omega_prime| Lambda^1/2+1+|Omega|)',
        second_graph_propagation='F=||Kfast psi||/Lambda^1/2>=1; |d sqrt(F)| <=Cslow*(|Omega_prime|+|Omega Omega_prime|+(1+|Omega|)Lambda^-1/2)',
        second_graph_norm_coefficient_log2=graph_log,
        second_graph_norm_Lambda_exponent='1/2',
        fixed_regular_shell='proof cutoff1 to r/2, support2r/3 in original domain3r/4; NOT initial thin Gamma replacement',
        shell_amplitude=f'<=2^({shell_log}) Lambda^-1/4',
        magnetic_IMS='chi/4||Pboost eta psi||^2 <=||eta^2 psi|| ||Kfast psi||+C chi ||deta psi||^2',
        boosted_boundary_momentum='||chi Pboost eta psi|| <=C Lambda^-1/8',
        exact_boost_cancellation='Pboost=P-Omega*g(y)^-1*b(y)/chi; [Hdrive,gamma] uses chi*g(y)*Pboost, including actual g(y), not just g0',
        whole_vs_Dirichlet_unweighted_error=f'<=T*2^({boundary_log}) Lambda^-1/8',
        boundary_coefficient_log2=boundary_log, target_bits=target_bits,
        required_log2_Lambda=selector, unchanged_selected_log2_Lambda_lower=selected,
        no_photon_weighted_boundary_claim=True, no_nonlinear_ground_return_needed=True,
        physical_consumer_binding_owner='ClosedSpatialRotation',
    )


def massive_normal_driver_ledger(normal: dict[str, Any], time: F,
                                 guard: dict[str, Any], common_log: int,
                                 target_bits: int = 22) -> dict[str, Any]:
    """All90V branch-normal comparator and finite selectors.

    The frozen Gaussian endpoint is a component proof, not nonlinear physical
    clock admission. Complete Noether/EM/Root/source-relative binding is B's.
    """
    if time <= 0:
        raise ValueError('The actual driver clock must be positive')
    dlog = ceil_log2(normal['unreduced_dimension'])
    vlog = ceil_log2(normal['volume'])
    nlog = ceil_log2(normal['size']+1)
    jlog = hex_fraction_log2_upper(normal['quotient_chart_jet_upper'])
    compare_log = hex_fraction_log2_upper(normal['finite_chart_compare'])
    alpha_log = hex_fraction_log2_upper(normal['uniform_alpha'])-1
    stiff_log = hex_fraction_log2_upper(normal['all_q_stiff'])-1
    radius_log = hex_fraction_log2_upper(normal['quotient_radius'])-1
    rho_log = (vlog+hex_fraction_log2_upper(normal['gram_upper'])+3)//2+1
    # q! CY^(q+1): covariant nearest differences, Cartesian row action,
    # four-link colour clovers, and the mass-Gauss implicit map, q<=8.
    noether_jet_log = 32+3*dlog+nlog+rho_log+4*jlog
    # Finite products, Gaussian Wick contractions through8 and three reduced
    # normal resolvents. No allocated mode list, rank count or photon gap.
    fast_log = 512+32*dlog+32*noether_jet_log+32*compare_log+64*(max(-alpha_log, -stiff_log)+2)
    soft_gap_log = alpha_log//2-4
    stiff_gap_log = stiff_log//2-4
    time_log = max(0, ceil_log2(time), ceil_log2(1/time))
    source_log = max(guard['comparison_frame_covariant_M0_through_M6_log2_upper'])
    norm_log = fast_log+8*(time_log+2)+source_log+5
    core_selector = max(4*(fast_log-radius_log+8),
                        32*(fast_log-soft_gap_log+8))
    # The deliberately weaker -1/32 envelope includes centre translation,
    # static resolvent dressing, squeezing, three adiabatic steps and DB[Y].
    selector = max(core_selector, 32*(norm_log+target_bits+8))
    selected = normal['selected_log2_lambda_conservative_lower']
    if selector >= selected:
        raise AssertionError('Unchanged tower does not pay massive-normal return')
    peak_omega_log = max(0, ceil_log2(F(165, 14)/time))
    boost_log = 2*noether_jet_log+8+2*peak_omega_log
    driven_window_log = max(common_log+3, boost_log+3)
    # Thin ordinary Gamma has barrier Lambda^1/2, not enough for the NEW
    # common-worst-floor Lambda^1/2 excess. A wider INTERIOR comparison cutoff
    # r0 Lambda^-3/16 has barrier Lambda^5/8; actual preparation stays unchanged.
    r0_log = radius_log-6
    barrier_log = min(alpha_log, stiff_log)-4+2*r0_log-10
    driven_barrier_selector = max(8*(driven_window_log-barrier_log+4),
                                  2*(driven_window_log-vlog+8))
    selector = max(selector, driven_barrier_selector)
    if selector >= selected:
        raise AssertionError('Driven common-floor offset exceeds interior barrier')
    # Nonconstant supported alpha(Birth)=1, alpha(Ref)=0 gives nonzero link
    # inertia; near the actual cubic reference ||b0||²>=1/(2^16 N).
    mid_ratio_log = selected//8-common_log-2*time_log-nlog-20
    if mid_ratio_log <= 0:
        raise AssertionError('Midclock ordinary-window adverse did not discriminate')
    return dict(
        identity='actual-Phi-colour-massive-normal-zeroended-driver',
        actual_clock=time,
        physical_coordinates=normal['physical_normal_dimension'],
        original_Phi_kinetic='chi/2 PiPhi^2', original_colour_cometric='0.6chi',
        actual_Noether_jet_C_log2=noether_jet_log, contracted_fast_constant_log2=fast_log,
        jet_orders='q! CY^(q+1), q<=8; original Cartesian/covariant/clover before Gauss',
        soft_frequency_lower=f'2^({soft_gap_log}) Lambda^(1/4)',
        stiff_frequency_lower=f'2^({stiff_gap_log}) Lambda^(3/8)',
        reduced_resolvent='branch Qnormal(Hfast_ms-Efast_ms)^-1Qnormal <=2/omega_soft; tensor identity on photons/harmonics, not a coupled-Root gap',
        fast_gap_core='radius Lambda^-9/32; relative Taylor C Lambda^-5/32 times E0 Lambda^3/8 =C Lambda^7/32 <<gap Lambda^1/4; IMS Lambda^1/16',
        static_cubic_vacuum_residual='O(C Lambda^1/16), resolved to O(C Lambda^-3/16), not multiplied by T',
        driven_mid_momentum='Omega b0/chi, Lambda^1/2; ordinary Lambda^3/8 window NOT inherited',
        driven_common_floor='e_global-Cdrive Lambda^1/2; retain actual finiteN m/s offsets and branch boost floors',
        driven_common_excess_exponent='1/2, NOT 3/8',
        driven_common_window_coefficient_log2=driven_window_log,
        regular_H_D_domain='unchanged union_(m,s) B(c_m,s,3r/4); all512 histories remain in the operator',
        driven_INTERIOR_comparison_cutoff='r0 Lambda^-3/16; does NOT replace initial Gamma/preparation',
        driven_barrier_exponent='5/8', driven_barrier_selector=driven_barrier_selector,
        boost_geometric_centre_shift='soft O(C Omega^2 Lambda^-1/2), stiff O(C Omega^2 Lambda^-3/4)',
        retained_common_phases=['Omega^2 Lambda^1/2 inertial floor', 'soft Omega^4 and Omega_prime^2 Lambda^0 corrections', 'actual branch-local Efast(m,s,t)'],
        exact_SOURCE_boost_commutation='scalar coordinate boost commutes actual B and true coexact a†',
        SOURCE_relative_first_order_cancellation='chi g0 DB*pboost-Omega b0.DB=0; curved remainder Omega*(g(y)M0b0-b(y)).DB',
        constant_force_endpoint='||alpha(T)|| <=600pi/(sqrt(2chi)T^3) sqrt(sum_j |b_j|^2/omega_j^5)',
        endpoint_Lambda_powers=dict(soft='-3/8', stiff='-11/16'),
        normal_quasimode_envelope='BRANCH COMPARATOR ONLY: Cfast (1+T+1/T)^8 Csource Lambda^-1/32 after common scalar phase; not actual nonlinear physical-clock admission',
        normalized_error_coefficient_log2=norm_log, target_bits=target_bits,
        normalized_SOURCE_application_scope='normal envelope pays inverse32 for the two-bank1/sqrt2 factor; A must bind actual ONE-global-filter SOURCE norm',
        required_log2_Lambda=selector, unchanged_selected_log2_Lambda_lower=selected,
        midclock_ordinary_window_ratio_log2_lower=mid_ratio_log,
        common_floor_Root_window='Cdrive Lambda^1/2 <=V Lambda; no Delta retuning or alias erasure',
        whole_joint_physical_clock_proved=False,
        consumer_owner_obligations=[
            'B actual complete Gphys/Phi-colour-EM-current-Ref and coupled Root binding',
            'photon/harmonic exchange graphs use their own energy, NOT total Lambda^1/2 driven energy',
            'actual SOURCE-relative ordered path/body/orbital comparison and final once-prepared neutral right action',
        ],
    )


def native_physical_S3_driver() -> dict[str, Any]:
    """NEW actual G_N, all15 Gauss and full90 complex normal q-plane.

    The +/-q realification has180 oscillators. The constant linear force is
    the actual Phi Cartesian row-spin at the flat mean; DY/off-mode nonlinear
    evolution is covered by the separate all90V ledger, not simulated here.
    """
    import mpmath as mp
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    size, duration = 5, 6.
    angles = np.zeros(3)
    selected, frame = original_frame(band, size, angles, True)
    basis = matrix_unit_basis()
    model = OriginalSpatial(band, size, basis, coordinates(frame, basis).ravel(), angles)
    tangent = np.zeros_like(frame)
    tangent[1], tangent[2] = -frame[2], frame[1]
    field = coordinates(tangent, basis).ravel()
    spin_coeff = np.linalg.solve(model.D.T@model.D, model.D.T@field)
    spin = np.einsum('a,aij->ij', spin_coeff, model.generators)
    field_error = float(np.linalg.norm(model.D@spin_coeff-field))
    mass = np.r_[np.ones(60), np.full(45, 1/.6)]
    root_mass, invroot_mass = np.sqrt(mass), 1/np.sqrt(mass)
    q = np.array([0., 0., 2*np.pi/size])
    gauge = model.gauge_columns(q)
    weighted_gauge = root_mass[:, None]*gauge
    horizontal = null_space(weighted_gauge.conj().T)
    vector = np.r_[field, np.zeros(45)]
    coefficient = np.linalg.solve(weighted_gauge.conj().T@weighted_gauge,
                                  weighted_gauge.conj().T@(root_mass*vector))
    residual = root_mass*vector-weighted_gauge@coefficient
    iph = float(np.vdot(field, field).real)
    ilink = float(np.vdot(gauge[60:]@spin_coeff, mass[60:]*(gauge[60:]@spin_coeff)).real)
    fraction = iph/(iph+ilink)
    ired = iph*ilink/(iph+ilink)
    projection_error = float(np.linalg.norm(coefficient-fraction*spin_coeff))
    norm_error = abs(float(np.vdot(residual, residual).real)-ired)
    _, rest_vectors = eigh(band.m*frame[0])
    rest = rest_vectors[:, 2:].conj().T@spin@rest_vectors[:, 2:]
    rest_values, rest_modes = eigh(rest)
    wrong_minus = float(np.linalg.norm(
        np.eye(2)+(rest_modes*np.exp(2j*np.pi*fraction*rest_values))@rest_modes.conj().T))
    uniform_minus = float(np.linalg.norm(
        np.eye(2)+(rest_modes*np.exp(2j*np.pi*rest_values))@rest_modes.conj().T))
    no_partner = field-model.D@np.linalg.solve(model.D.T@model.D, model.D.T@field)
    parts = model.hessian_parts(q)
    heavy = parts['local']+parts['sea']+parts['contact']
    alignment = model.alignment(q)
    # Canonical K4=.7 Lambda^5/4, NOT an unscaled .7 plaquette control.
    colour_plaquette = plaquette_hessian(q, .7)
    bcomplex = horizontal.conj().T@(root_mass*vector)
    beta = math.sqrt(size**3/2)*np.r_[bcomplex.real, bcomplex.imag]
    rows = []
    with mp.workdps(110):
        def polynomial_integral(z, upper):
            iz, endpoint = 1j*z, mp.exp(1j*z*upper)
            polynomials = ([0, 0, 30, -60, 30], [0, 60, -180, 120],
                           [60, -360, 360], [-360, 720], [720])
            return sum((-1)**j*(endpoint*sum(mp.mpf(c)*upper**k for k,c in enumerate(p))
                                -mp.mpf(p[0]))/iz**(j+1)
                       for j,p in enumerate(polynomials))

        for Lambda in (1e4, 1e16, 1e32):
            chi = Lambda**-.5
            raw = Lambda*heavy+Lambda**1.25*(alignment+colour_plaquette)
            maps = invroot_mass[:, None]*horizontal
            reduced = maps.conj().T@raw@maps
            stiffness = np.block([[reduced.real, -reduced.imag],
                                  [reduced.imag, reduced.real]])
            stiffness = (stiffness+stiffness.T)/2
            values, modes = eigh(stiffness)
            if values[0] <= 0:
                raise AssertionError('Actual original joint normal Hessian not positive')
            frequencies = np.sqrt(chi*values)
            forcing = modes.T@beta
            precision = (modes*(frequencies/chi))@modes.T
            riccati_error = float(np.linalg.norm(chi*precision@precision-stiffness)
                                 /np.linalg.norm(stiffness))
            p_mid, endpoint_alpha, phase_correction = [], [], []
            chi_mp, time_mp = mp.mpf(chi), mp.mpf(duration)
            i0 = 40*mp.pi**2/(7*time_mp)
            i1 = 480*mp.pi**2/(7*time_mp**3)
            i2 = 2880*mp.pi**2/time_mp**5
            for omega, force in zip(frequencies, forcing):
                w, b = mp.mpf(float(omega)), mp.mpf(float(force))
                z = w*time_mp
                integral = 2*mp.pi*polynomial_integral(z, mp.mpf(1))
                alpha = -b*mp.sqrt(w/(2*chi_mp))*mp.exp(-1j*z)*integral
                endpoint_alpha.append(float(abs(alpha)))
                half = 2*mp.pi*polynomial_integral(z, mp.mpf('.5'))
                alpha_half = -b*mp.sqrt(w/(2*chi_mp))*mp.exp(-.5j*z)*half
                p_mid.append(float(mp.sqrt(2*w/chi_mp)*alpha_half.imag))
                correction = i1/w**2+i2/w**4
                correction += (120*mp.pi/(time_mp**3*w**2)
                               -1440*mp.pi/(time_mp**5*w**4))*integral.real
                correction += (-720*mp.pi/(time_mp**4*w**3))*integral.imag
                phase_correction.append(b*b*correction/(2*chi_mp))
            endpoint_norm = float(np.linalg.norm(endpoint_alpha))
            endpoint_upper = float(600*np.pi/(math.sqrt(2*chi)*duration**3)
                                   *np.sqrt(np.sum(forcing**2/frequencies**5)))
            kinetic_mid = float(chi*np.dot(p_mid, p_mid)/2)
            zero_point = float(frequencies.sum()/2)
            omega_peak = 15*np.pi/(4*duration)
            floor = -float(np.dot(beta, beta))*omega_peak**2/(2*chi)
            common_phase = float(np.dot(beta, beta))*float(i0)/(2*chi)
            # ACTUAL Gaussian graph of physical Gphi=pnormal+aQbody.
            variance = float(np.sum(forcing**2*frequencies/(2*chi)))
            mean = fraction/2
            first_squared = mean**2+variance
            second_graph = math.sqrt(mean**4+6*mean**2*variance+3*variance**2)
            if riccati_error > 1e-9 or endpoint_norm > endpoint_upper or second_graph <= first_squared:
                raise AssertionError('Actual driven Gaussian/endpoint/second-graph control failed')
            rows.append(dict(
                Lambda=Lambda, chi=chi, normal_oscillators=len(values),
                frequencies=[float(frequencies[0]), float(frequencies[-1])],
                Riccati_relative_residual=riccati_error,
                exact_quintic_endpoint_coherent_norm=endpoint_norm,
                zeroended_three_IBP_upper=endpoint_upper,
                midpoint_ordinary_kinetic_energy=kinetic_mid,
                ordinary_zero_point=zero_point, ordinary_Ecut_control=8*zero_point,
                midpoint_exceeds_ordinary_control=kinetic_mid>8*zero_point,
                actual_common_inertial_floor=floor, retained_leading_common_phase=common_phase,
                retained_exact_phase_correction=str(sum(phase_correction)),
                first_graph_squared=first_squared, second_graph_norm=second_graph,
                wrong_q1_squared_cap_deficit=second_graph-first_squared,
            ))
    if max(field_error, projection_error, norm_error, uniform_minus) > 1e-10 or wrong_minus < 1:
        raise AssertionError('Actual mass Gauss/body-fraction adverse failed')
    if not rows[-1]['midpoint_exceeds_ordinary_control']:
        raise AssertionError('Midclock ordinary-window adverse did not discriminate')
    return dict(
        scope='NEW actual originalN5 G_N(000), full105/all15 Gauss, +/-q180 canonical quadratic normal restriction; not full90V dynamic simulation or full SOURCE/physical soft loop',
        native_clock=duration, profile='cos(2pi z/5), unit peak; norm²=V/2',
        actual_Gram=selected['G'], full10_stationarity=selected['gradient_residual'],
        field_flow='Phi1dot=-Phi2, Phi2dot=Phi1; flat colour orbital0 BEFORE Gauss',
        actual_S3=spin, physical_rest_spin_eigenvalues=rest_values,
        I_Phi=iph, I_colour_link=ilink, I_reduced=ired, derived_body_fraction=fraction,
        mass_Gauss_projection_error=projection_error, horizontal_inertia_error=norm_error,
        Phi_gauge_flow_error=field_error, wrong_no_link_horizontal_norm=float(np.linalg.norm(no_partner)),
        actual_joint_horizontal_norm=float(np.linalg.norm(residual)),
        wrong_automatic_hardmode_minus_defect=wrong_minus,
        correct_uniform_rest_minus_defect=uniform_minus,
        hardware='Gphi=pnormal-aQGaussMatter=pnormal+aQbody; NO added GU/SU2 rotor',
        source_boost_commutes='scalar field multiplication; actual coordinate SOURCE and coexact creator',
        literal_colour_K4='.7 Lambda^5/4 in ALL normal frequencies',
        high_frequency_phase_evaluation='mpmath110 digits on exact binary64 native mode inputs; legalN uses bounds, not huge-phase floating extrapolation',
        native=rows,
    )


def native_union_boundary() -> dict[str, Any]:
    """Original N5 heavy/scalar curve, two actual centres and coupled histories.

    A one-normal finite-difference consumer; not the full90V H or formal Root.
    The union covers overlapping tubes; beta is retained, not sector-projected.
    """
    from prepared_neutral_local_well import window_values, taper
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    basis = matrix_unit_basis()
    cases = [np.array([0., 0., np.pi/2]), np.array([0., 0., 3*np.pi/2])]
    selected_frames = [original_frame(band, 5, angles, True) for angles in cases]
    frames = [item[1] for item in selected_frames]
    difference = frames[1]-frames[0]
    distance = float(np.sqrt(125*np.einsum('mij,mji->', difference, difference).real/4))
    if distance <= 1e-8:
        raise AssertionError('Actual skew-centre pair did not discriminate')
    direction = difference/distance
    midpoint = (frames[0]+frames[1])/2
    centre_coordinates = np.array([-distance/2, distance/2])
    models = [OriginalSpatial(band, 5, basis, coordinates(f, basis).ravel(), a)
              for f, a in zip(frames, cases)]
    Lambda, chi, beta = 1e4, 1e-2, 1/16
    dx = .0015
    whole_grid = np.arange(-240, 241)*dx
    local_indices = np.flatnonzero(abs(whole_grid) < .24-1e-10)
    potentials = np.empty((len(whole_grid), 2))
    floors = np.array([125*item[0]['value'] for item in selected_frames])
    for j, coordinate in enumerate(whole_grid):
        phi = midpoint+coordinate*direction
        gram = np.einsum('mij,nji->mn', phi, phi).real/4
        elastic = np.sum(gram*gram)
        for s, model in enumerate(models):
            h = np.einsum('pm,mij->pij', model.x, phi)
            sea = np.linalg.eigvalsh(h)[:, 2:].sum()/125
            potentials[j, s] = Lambda*(125*(elastic-sea)-min(floors))+beta
    if abs(floors[0]-floors[1]) > 1e-10:
        raise AssertionError('Reflection-related native classical tie failed')
    kinetic = chi/(2*dx*dx)
    def operator(indices: np.ndarray):
        n = len(indices)
        lap = cast(Any, sparse.diags)([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1], format='csr')
        field = sparse.kron(lap*kinetic, sparse.eye(2, format='csr'), format='csr')
        root = sparse.kron(sparse.eye(n, format='csr'),
                           sparse.csr_matrix([[0., beta], [beta, 0.]]), format='csr')
        return field+root+sparse.diags(potentials[indices].ravel(), format='csr')
    Hdir = operator(local_indices)
    Hwhole = operator(np.arange(len(whole_grid)))
    energies, vectors = eigh(Hdir.toarray())
    x = whole_grid[local_indices]
    trial = np.stack([np.exp(-((x-c)/.023)**2/2) for c in centre_coordinates], axis=1).ravel()
    trial /= np.linalg.norm(trial)
    excess = float(np.vdot(trial, Hdir@trial).real)
    width = 4*excess
    filtered = vectors@(window_values(energies, 0., width)*(vectors.T@trial))
    filter_norm = float(np.linalg.norm(filtered))
    filtered /= filter_norm
    inner, outer = .08, .15
    individual = np.stack([taper((abs(x-c)-inner)/(outer-inner)) for c in centre_coordinates])
    gamma = 1-np.prod(1-individual, axis=0)
    gamma = np.repeat(gamma, 2)
    low = vectors[:, energies <= 2*width]
    shell_gram = low.T@(((1-gamma)**2)[:, None]*low)
    shell_norm = float(np.sqrt(max(0., np.linalg.eigvalsh(shell_gram)[-1])))
    residual = Hdir@(gamma[:, None]*low)-gamma[:, None]*(Hdir@low)
    commutator_norm = float(np.linalg.norm(residual, ord=2))
    local_state = gamma*filtered
    local_norm = float(np.linalg.norm(local_state))
    extension = np.zeros(Hwhole.shape[0])
    rows = (2*local_indices[:, None]+np.arange(2)).ravel()
    extension[rows] = local_state
    whole_e, whole_v = eigh(Hwhole.toarray())
    time = 6.
    flown = whole_v@(np.exp(-1j*time*whole_e)*(whole_v.T@extension))
    local_flown = vectors@(np.exp(-1j*time*energies)*(vectors.T@filtered))
    comparison = np.zeros(Hwhole.shape[0], dtype=complex)
    comparison[rows] = gamma*local_flown
    defect = float(np.linalg.norm(flown-comparison))
    predicted = time*commutator_norm
    wrong = np.repeat((abs(x) < .02).astype(float), 2)
    wrong_norm = float(np.linalg.norm(Hdir@(wrong[:, None]*low)-wrong[:, None]*(Hdir@low), ord=2))
    if filter_norm**2 < .75 or local_norm < .95 or defect > predicted+1e-9 or wrong_norm <= commutator_norm:
        raise AssertionError('Actual native union filter/boundary/adverse control failed')
    return dict(
        scope='one uniform original N5 heavy/scalar normal, two actual reflection-related Root-history comparator curves; not formal fullRoot or full90V',
        Lambda=Lambda, chi=chi, beta=beta, actual_centre_coordinates=centre_coordinates,
        overlapping_tubes=True, floor_spread=float(abs(floors[0]-floors[1])),
        total_onebody_modes=500, occupied_heavy_modes=250,
        local_dimension=Hdir.shape[0], whole_dimension=Hwhole.shape[0],
        trial_excess=excess, filter_norm_squared=filter_norm**2, post_gamma_norm=local_norm,
        actual_spectral_shell_operator_norm=shell_norm,
        actual_cutoff_commutator_operator_norm=commutator_norm,
        exact_whole_H_vs_extended_Dirichlet_time6_defect=defect,
        Duhamel_upper=predicted, wrong_sharp_inner_cutoff_commutator=wrong_norm,
        no_sector_superselection=True, no_ground_state_premise=True,
        harmonic_or_EM_spectator_native_execution=False,
    )


def exact_portable(value: Any) -> Any:
    if isinstance(value, F):
        return dict(numerator_hex=hex(value.numerator), denominator_hex=hex(value.denominator))
    if isinstance(value, dict):
        return {str(k): exact_portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact_portable(v) for v in value]
    return portable(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-q', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--normal-momentum-driver', action='store_true')
    parser.add_argument('--driver-ledger-only', action='store_true')
    parser.add_argument('--driver-guard-capsule', type=Path)
    parser.add_argument('--barrier-normal-receipt', type=Path)
    parser.add_argument('--clock', type=F, default=F(6))
    parser.add_argument('--driver-outer-radius', type=F)
    parser.add_argument('--initial-energy-coefficient-log2', type=int)
    parser.add_argument('--driver-energy-receipt', type=Path)
    parser.add_argument('--energy-coefficient-log2', type=int)
    parser.add_argument('--source-norm-log2', type=int, default=0)
    parser.add_argument('--source-first-log2', type=int, default=0)
    parser.add_argument('--source-lower', type=F, default=F(1))
    parser.add_argument('--target-bits', type=int)
    args = parser.parse_args()
    if args.driver_ledger_only and (not args.normal_momentum_driver or args.full_q):
        parser.error('driver-ledger-only requires normal-momentum-driver and excludes frozen full-q execution')
    names = ('prepared_minset_normal_proof.py', 'prepared_minset_normal_proof.md',
             'original_charged_principal.py', 'full_spatial_gauge_stability.py',
             'noncompact_quantitative_normal_budget.py', 'sea_selection.py',
             'prepared_neutral_local_well.py', 'rotation_band.py',
             'unrestricted_field_minimum.py', 'improved_phi_band.py',
             'periodic_bulk_frame.py', 'local_orbital_stability.py',
             'bulk_flat_holonomy_bound.py', 'bulk_quantum_selection.py',
             'prepared_neutral_full_field.py', 'bosonic_original_law_join.py')
    result: dict[str, Any] = dict(identity=IDENTITY, plan=PLAN, plan_revision=PLAN_REVISION,
                  source_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names})
    if args.normal_momentum_driver:
        if not args.barrier_normal_receipt or not args.driver_guard_capsule or args.energy_coefficient_log2 is None:
            parser.error('normal driver requires frozen normal, corrected Guard and actual common coefficient')
        receipt = json.loads(args.barrier_normal_receipt.read_text())
        guard = json.loads(args.driver_guard_capsule.read_text())
        result.update(
            frozen_normal_input_sha256=hashlib.sha256(args.barrier_normal_receipt.read_bytes()).hexdigest(),
            corrected_Guard_input_sha256=hashlib.sha256(args.driver_guard_capsule.read_bytes()).hexdigest(),
            raw_correlated_trial_constants=raw_correlated_trial_ledger(receipt['legal_N']),
            massive_normal_driver=massive_normal_driver_ledger(receipt['legal_N'], args.clock, guard,
                                                               args.energy_coefficient_log2,
                                                               args.target_bits if args.target_bits is not None else 22),
            ledger_only=args.driver_ledger_only)
        result['fast_graph_boundary'] = fast_graph_boundary_ledger(
            receipt['legal_N'], result['massive_normal_driver'], args.clock,
            args.target_bits if args.target_bits is not None else 22)
        if args.driver_outer_radius is not None:
            result['global_driven_Phi_moment'] = global_driven_phi_moment_ledger(
                receipt['legal_N'], args.clock, args.driver_outer_radius,
                args.initial_energy_coefficient_log2)
            if args.driver_energy_receipt is not None:
                energy_receipt = json.loads(args.driver_energy_receipt.read_text())
                energy = energy_receipt['initial_energy']
                if (energy['identity'] != 'PNFF-RAW64-coherent-Ref4-sea-Ref0-seaPlus4-ONE-globalfilter-v2'
                        or not energy['initial_only']
                        or not energy['absolute_initial_energy_upper'].startswith('8V Lambda ')
                        or energy['selected_log2_lambda_lower'] != receipt['legal_N']['selected_log2_lambda_conservative_lower']
                        or energy['required_log2_lambda'] >= energy['selected_log2_lambda_lower']):
                    raise ValueError('Actual coherent-bank absolute first-energy binding is incompatible')
                if args.initial_energy_coefficient_log2 is None or args.initial_energy_coefficient_log2 < 3:
                    raise ValueError('Use the executed absolute first-energy coefficient 8 or larger')
                result['actual_SOURCE_initial_energy_input_sha256'] = hashlib.sha256(
                    args.driver_energy_receipt.read_bytes()).hexdigest()
                result['global_driven_Phi_moment']['actual_initial_energy_binding'] = dict(
                    identity=energy['identity'], absolute_coefficient='8',
                    root_primitive_requirement=energy['Root_requirement'],
                    actual_neutral_and_SOURCE_norms=energy['norm_inputs'])
                result['global_driven_Phi_moment']['executed_initial_energy_binding_checked'] = True
        if not args.driver_ledger_only:
            result['native_physical_S3_driver'] = native_physical_S3_driver()
    elif args.barrier_normal_receipt:
        receipt = json.loads(args.barrier_normal_receipt.read_text())
        normal = receipt['legal_N']
        result.update(
            frozen_normal_input_sha256=hashlib.sha256(args.barrier_normal_receipt.read_bytes()).hexdigest(),
            union_barrier=union_barrier_ledger(normal, args.clock, args.energy_coefficient_log2,
                                               args.source_norm_log2, args.source_first_log2,
                                               args.source_lower, args.target_bits),
            native_union_boundary=native_union_boundary())
    else:
        result.update(centre_probe=native_centre_probe(), legal_N=local_normal_constants(),
                      correlated_trial_control=riccati_probe())
    if args.full_q:
        result['native_all_q'] = native_full_q_probe()
    payload = json.dumps(exact_portable(result), indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(payload+'\n')
    print(payload)


if __name__ == '__main__':
    main()

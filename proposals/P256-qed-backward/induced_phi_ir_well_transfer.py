#!/usr/bin/env python3
"""Finite macrobox original-law Phi/colour prepared transfer, exact-root shapes.

All bound arithmetic is rational. Square roots use integer-isqrt enclosures,
not floating point; receipts are conservative exact dyadic endpoints. This is
not a photon-vacuum, ground-phase, continuum or Compton construction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
from math import isqrt
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from induced_phi_backreaction import PhiConfig, models, scalar_paths, IMPORT_BYTES

sys.set_int_max_str_digits(0)
ROOT = Path(__file__).resolve().parent
OWN_BYTES = Path(__file__).read_bytes()
SOURCE_BYTES = dict(IMPORT_BYTES)
for _module in tuple(sys.modules.values()):
    _file = getattr(_module, '__file__', None)
    if _file is not None:
        _path = Path(_file).resolve()
        if _path.parent == ROOT and _path.suffix == '.py' and _path.name not in SOURCE_BYTES:
            SOURCE_BYTES[_path.name] = _path.read_bytes()
SOURCE_BYTES[Path(__file__).name] = OWN_BYTES
SOURCE_HASHES = {name: hashlib.sha256(value).hexdigest() for name, value in SOURCE_BYTES.items()}


def rational(value: Any) -> F:
    """The scientific bound ABI deliberately has no missing-field fallback."""
    if not isinstance(value, dict):
        raise TypeError('Expected exact numerator/denominator bound object')
    return F(int(value['numerator']), int(value['denominator']))


def binary_exponent(x: F) -> int:
    if x <= 0:
        raise ValueError('Positive rational required')
    k = x.numerator.bit_length() - x.denominator.bit_length()
    if x < power(k):
        k -= 1
    return k


def power(k: int) -> F:
    return F(1 << k) if k >= 0 else F(1, 1 << -k)


def sqrt_bounds(x: F) -> tuple[F, F]:
    """80 significant binary bits; n^2 <= scaled x < (n+1)^2 exactly."""
    if x < 0:
        raise ArithmeticError('Negative radicand')
    if x == 0:
        return F(0), F(0)
    shift = binary_exponent(x) // 2 - 80
    scaled = x / power(2 * shift)
    n = isqrt(scaled.numerator // scaled.denominator)
    return n * power(shift), (n + 1) * power(shift)


def sl(x: F) -> F:
    return sqrt_bounds(x)[0]


def su(x: F) -> F:
    return sqrt_bounds(x)[1]


def endpoint(x: F, upper: bool = True) -> dict[str, str]:
    if x == 0:
        out = F(0)
    elif x < 0:
        positive = endpoint(-x, not upper)
        return {'numerator': '-' + positive['numerator'], 'denominator': positive['denominator']}
    else:
        k = binary_exponent(x)
        out = power(k + 1) if upper and x != power(k) else power(k)
    return {'numerator': str(out.numerator), 'denominator': str(out.denominator),
            'rounding': 'exact outward power-of-two enclosure; all preceding arithmetic rational and isqrt-certified'}


def exact_endpoint(x: F) -> dict[str, str]:
    return {'numerator': str(x.numerator), 'denominator': str(x.denominator)}


def dyadic_bounds(x: F) -> tuple[F, F]:
    """Exact outward 96-bit dyadic bounds, certified by rational floor."""
    step = power(binary_exponent(x) - 96)
    scaled = x / step
    n = scaled.numerator // scaled.denominator
    return n * step, (n + 1) * step


def pi_bounds() -> tuple[F, F]:
    """Machin identity, alternating rational arctangent series with remainder."""
    def atan(x: F) -> tuple[F, F]:
        total = F(0)
        for n in range(64):
            total += (-1) ** n * x ** (2 * n + 1) / (2 * n + 1)
        next_term = x ** 129 / 129
        return total, total + next_term
    a0, a1 = atan(F(1, 5))
    b0, b1 = atan(F(1, 239))
    return 16 * a0 - 4 * b1, 16 * a1 - 4 * b0


def path_bounds(band: Any) -> dict[str, Any]:
    onsite, paths = scalar_paths(band)
    # |Re z|+|Im z| >= |z|. Every scalar binary64 is preserved exactly.
    cap = F.from_float(abs(float(onsite)))
    j = F(0)
    jj = F(0)
    rows = []
    for displacement, scalars in paths:
        row = sum((F.from_float(abs(float(z.real))) + F.from_float(abs(float(z.imag))) for z in scalars), F(0))
        length = int(np.sum(np.abs(displacement)))
        cap += 2 * row
        j += 2 * length * row
        jj += 2 * length * length * row
        rows.append({'displacement': displacement.tolist(), 'absolute_scalar_sum_upper': endpoint(row)})
    return {'cap': cap, 'j': j, 'jj': jj, 'rows': rows}


def certify(mean_capsule: dict[str, Any], clock: int, cutoff_bits: int) -> dict[str, Any]:
    start = time.monotonic()
    required = ('branch', 'input_descriptor', 'input_sha256', 'source_at_import',
                'own_source_at_import', 'runtime', 'elapsed_seconds', 'finite_size',
                'Lambda_exponent', 'microscopic_config', 'actual_finite_grid_quadrature_certificate',
                'exact_centre_definition', 'centre_enclosure', 'slow_stiffness_lower',
                'slow_stiffness_upper', 'all64_sector_rows', 'sector_policy')
    for key in required:
        mean_capsule[key]
    N = mean_capsule['finite_size']
    exponent = mean_capsule['Lambda_exponent']
    if (N, exponent, clock, cutoff_bits) != (262144, 8192, 4096, 2048):
        raise ValueError('This branch certifies only the assigned macrobox and clock/cutoff')
    config = mean_capsule['microscopic_config']
    expected = asdict(PhiConfig())
    del expected['Lambda']
    if config != expected:
        raise ValueError('Microscopic parameters differ from declared frozen original-law branch')
    cfg = PhiConfig(**config, Lambda=1.0)
    sectors = mean_capsule['all64_sector_rows']
    twists = {tuple(row['twist']) for row in sectors}
    if len(sectors) != 64 or twists != {(i, j, k) for i in range(4) for j in range(4) for k in range(4)}:
        raise ValueError('All64 actual finite-root sector rows are required')
    for row in sectors:
        row['exact_root_definition']; row['energy_interval']; row['selection_status']
    if mean_capsule['sector_policy']['kind'] != 'prepared_uniform_64_mixture':
        raise ValueError('This state explicitly prepares the unresolved-selector uniform64 mixture')
    enc = mean_capsule['centre_enclosure']
    bounds = {key: rational(enc[key]) for key in (
        'g0_lower', 'g0_upper', 'gs_lower', 'gs_upper', 'all64_Gram_radius_upper',
        'minimum_Gram_eigenvalue_lower', 'trace_Gram_upper')}
    amin = rational(mean_capsule['slow_stiffness_lower'])
    amax = rational(mean_capsule['slow_stiffness_upper'])
    if not (bounds['minimum_Gram_eigenvalue_lower'] > 0 and 0 < amin <= amax):
        raise ArithmeticError('No positive all64 full-root stiffness enclosure')
    V = N ** 3
    D = 90 * V
    Dfast = D - 45
    L = power(exponent)
    eta = power(-cutoff_bits)
    chi = power(-exponent // 2)
    rho = su(bounds['trace_Gram_upper'])
    rmin = bounds['minimum_Gram_eigenvalue_lower'] / 2
    rmax = bounds['trace_Gram_upper'] / 2
    pi_lo, pi_hi = pi_bounds()
    xlo, xhi = pi_lo / N, pi_hi / N
    # Alternating sin Taylor bounds at the ACTUAL first periodic mode.
    sin_lo = xlo - xlo ** 3 / 6 + xlo ** 5 / 120 - xlo ** 7 / 5040
    sin_hi = xhi - xhi ** 3 / 6 + xhi ** 5 / 120
    lapmin = dyadic_bounds(4 * sin_lo ** 2)[0]
    lapmin_upper = dyadic_bounds(4 * sin_hi ** 2)[1]
    hfast = min(lapmin / 5, rmin / 5)
    omega_min = sl(3 * rmin / 25)
    kmax = 1 + 12 / rmin
    bands = [(load, model, role, path_bounds(model)) for load, model, role in models(cfg)]
    heavy = bands[0][3]['cap']
    light = sum((F.from_float(load) * p['cap'] for load, _, role, p in bands if not role.startswith('neutral')), F(0))
    light_j = sum((F.from_float(load) * p['j'] for load, _, role, p in bands if not role.startswith('neutral')), F(0))
    light_jj = sum((F.from_float(load) * p['jj'] for load, _, role, p in bands if not role.startswith('neutral')), F(0))
    heavy_gap = 2 * F.from_float(cfg.heavy_m) * sl(bounds['minimum_Gram_eigenvalue_lower'])
    K1 = 8 * heavy * (1 + rho)
    K2 = 16 * heavy * (1 + rho)
    K3 = 32 * heavy * (1 + rho)
    kappa = F(2) / F.from_float(cfg.a)
    # Uniform q, including twisted fundamental CAR: Parseval removes V.
    C = 12 * kappa * (1 + rho) ** 2 + 16 * K2 + 64 * K1 ** 2 / heavy_gap
    charged_static = F(0)
    charged_third = F(0)
    charged_cluster_rows = []
    for load, model, role, paths in bands[1:]:
        species_gap = 2 * F.from_float(model.m) * sl(bounds['minimum_Gram_eigenvalue_lower'])
        s1 = 8 * paths['cap'] * (1 + rho)
        s2 = 16 * paths['cap'] * (1 + rho)
        s3 = 32 * paths['cap'] * (1 + rho)
        weight = F.from_float(load) / L
        charged_static += weight * (16 * s2 + 64 * s1 ** 2 / species_gap)
        charged_third += weight * (2 * V) * (128 * s1 ** 3 / species_gap ** 2 + 32 * s1 * s2 / species_gap + s3)
        charged_cluster_rows.append({'role': role, 'load': load, 'gap_lower': endpoint(species_gap, False),
            'K1_upper': endpoint(s1), 'K2_upper': endpoint(s2), 'K3_upper': endpoint(s3)})
    C += charged_static
    guards: dict[str, Any] = {}
    def guard(name: str, margin: F) -> None:
        if margin <= 0:
            raise ArithmeticError(f'Failed {name}: exact rational margin nonpositive')
        guards[name] = {'passed': True, 'positive_margin_lower': endpoint(margin, False)}
    guard('orbit_force_enclosure_regime', 1 - rmin)
    guard('actual_first_mode_sin_Taylor_regime', min(xlo, F(1, 1000) - xhi))
    perturbation = power(-exponent // 4) * C / hfast
    schur_den = 3 * hfast / 5 - power(-exponent // 4) * (amax + C)
    guard('full_fast_slow_Schur_denominator', schur_den)
    schur = C ** 2 / schur_den
    guard('relative_fast_perturbation_below_one_percent', F(1, 100) - perturbation)
    guard('slow_Schur_mass', amin / 4 - power(-exponent // 4) * schur)
    separate = hfast / 2 - power(-exponent // 4) * (C + 4 * C ** 2 / amin)
    guard('separate_Young_slow_fast_masses', separate)
    guard('fast_mass_dominates_slow_mass', power(exponent // 4) * hfast - amin)
    gap0 = power(exponent // 4) * sl(amin / 2)
    phi_variance = 1 / (2 * sl(lapmin / 5))
    transverse_variance = F(3, 10) / omega_min
    longitudinal_variance = (F(3, 5) + 12 / rmin) / (2 * omega_min)
    fast_variance = max(phi_variance, transverse_variance, longitudinal_variance)
    slow_variance = 1 / (2 * sl(amin))
    radius2 = Dfast * fast_variance * power(-7 * exponent // 8) + 45 * slow_variance * power(-3 * exponent // 4)
    deficit = 4 * radius2 / eta ** 2
    guard('exact_full_Gaussian_cut_normalization', F(1, 100) - deficit)
    cutnorm = 1 - deficit
    analytic_radius = min(F(1, 1000), rmin / (16 * su(rmax)))
    guard('chart_contains_cut_support', analytic_radius - eta)
    M = 4 + 512 / rmin
    M1 = 8 * M / analytic_radius
    log1 = 8 / analytic_radius
    log2 = 32 / analytic_radius ** 2
    U = chi * (D ** 2 * (M1 * log1 + M * log2) / 4 + D * M * log1 ** 2 / 8)
    metric_relative = M1 * eta / F(3, 5)
    guard('all64_heavy_cluster_remains_gapped', heavy_gap - 4 * K1 * eta)
    occupied = 2 * V
    third_heavy = occupied * (128 * K1 ** 3 / heavy_gap ** 2 + 32 * K1 * K2 / heavy_gap + K3)
    third_quartic = 24 * kappa * (rho + 1)
    potential_relative = eta * (8 * (third_heavy + charged_third + third_quartic) / amin + 256 * (1 + rho) ** 2 / hfast + 4096 / rmin)
    eps = metric_relative + potential_relative
    # Every periodic q lies at lap<=12. Roots and their first G derivative
    # are bounded by the actual orbit spectral enclosure, not sampled N6 rows.
    fast_energy = Dfast * power(3 * exponent // 8) * su(3 * rmax / 25 + F(63, 25))
    slow_energy = 45 * power(exponent // 4) * su(max(C, amax))
    light_lipschitz = 8 * su(F(V)) * light * (1 + rho)
    Fforce = 100 * D * (1 + rho) ** 2 * power(3 * exponent // 8) / (rmin * sl(rmin)) + light_lipschitz
    harmonic_excess = power(exponent // 8) * Dfast * C * (fast_variance + kmax / omega_min) + 45 * schur / sl(amin)
    trial_energy = fast_energy + slow_energy + harmonic_excess
    norm_excess = deficit * trial_energy / cutnorm
    cut_derivative = 8 * chi * M * radius2 / eta ** 4
    cut_excess = norm_excess + 2 * su(trial_energy * cut_derivative) + cut_derivative
    shape_price = F(0)  # Defined exact spectral functions, no numerical shape.
    scalar_perturbation = eps * (trial_energy + gap0) + Fforce * eta + U
    baseline = harmonic_excess + cut_excess + shape_price
    gap = gap0 - 4 * scalar_perturbation - 2 * baseline
    guard('full_metric_potential_well_gap', gap - gap0 / 2)
    excess = baseline + 2 * scalar_perturbation
    spectral_distance = su(2 * excess / gap)
    well_upper = trial_energy + cut_excess + scalar_perturbation
    budget = well_upper + Fforce * eta + U
    barrier = L * amin * eta ** 2 / 1024 - Fforce * eta - U
    shell = (1 - eps) * L * amin * eta ** 2 / 256 - Fforce * eta - U
    slope = eta * su(L * amin / (chi * M)) / 64
    ramp_price = chi * M * slope ** 2 / 2
    guard('nonlinear_form_below_one_percent', F(1, 100) - eps)
    guard('Agmon_barrier_dominates_well_energy', barrier - 16 * well_upper)
    guard('actual_shell_pays_ramp_kinetic_price', shell / 8 - ramp_price)
    guard('actual_shell_retains_weaker_half_barrier', shell - well_upper - ramp_price - barrier / 2)
    Agmon = eta ** 2 * sl(L * amin / (chi * M)) / 1024
    weighted_norm2 = 1 + 2 * budget / barrier
    weighted_grad2 = 4 * budget / (chi * F(3, 5)) + 2 * slope ** 2 * weighted_norm2
    boundary_prefactor = 2 * (chi * M * 8 / eta * su(weighted_grad2) + chi / 2 * (D ** 2 * M1 * 8 / eta + D * M * 96 / eta ** 2) * su(weighted_norm2))
    boundary_log2 = binary_exponent(boundary_prefactor) + 1
    guard('normalized_full_D_squared_commutator_tail', Agmon - boundary_log2 - 4096)
    boundary = power(-4096)  # e^-A <= 2^-A, since ln2<1.
    D1 = 4 * su(F(D)) * occupied * K1 / heavy_gap
    D2 = D * (64 * occupied ** 2 * K1 ** 2 / heavy_gap ** 2 + 16 * occupied * K2 / heavy_gap)
    derivative_norm = su(8 * budget / (chi * F(3, 5)))
    charge_norm = 66 * su(F(5 * V, 2))
    fibre_action = chi * (4 * M * derivative_norm * (D1 + charge_norm / rmin) + M * D2 + D * M1 * (D1 + charge_norm) + M * charge_norm ** 2 / rmin)
    charged_action = light_lipschitz * eta
    action = fibre_action + charged_action + boundary
    norm_distance = 2 * spectral_distance + clock * action
    guard('nontrivial_prepared_state_transfer', F(1, 1000) - norm_distance)
    em_modes = 2 * V + 1
    em_radius = F(1, 8)
    UElink = F.from_float(cfg.inertia) / F.from_float(cfg.a)
    em_graph = UElink * su(F(63 * em_modes, 2) + 9 * em_modes * (em_modes - 1)) / em_radius ** 2
    Coulomb_graph = UElink * V * 132 ** 2 / lapmin
    charged_graph = 8 * V * light * rho
    reference_graph = em_graph + Coulomb_graph + charged_graph
    c = L / (8 * F.from_float(cfg.a) * V)
    matter_L = 8 * su(F(V)) * (L * heavy + light)
    guard('Young_lower_shift_ratio', matter_L - c)
    # Young maximization 3 L^(4/3)/(4^(4/3)c^(1/3)) <= 1+L^2/c
    B = 1 + matter_L ** 2 / c
    packet_rho = su(F(V)) * rho + eta
    inverse_covariance = max(2 * su(F(12, 5)), 2 * su(3 * rmax / 25 + F(63, 25)) / F(3, 5))
    kinetic_graph = M * D * (power(3 * exponent // 8) * (1 + inverse_covariance) + power(exponent // 4) * (1 + 2 * su(amax))) + 16 * M * D * chi / eta ** 2
    potential_graph = L / F.from_float(cfg.a) * V * (rho + eta) ** 4 + 8 * L * V * heavy * (rho + eta) + 128 * V * power(5 * exponent // 4) * (1 + rho) ** 2 * eta ** 2
    graph = kinetic_graph + potential_graph + reference_graph + action
    difference_graph = 2 * (graph + B)
    weighted_error = su(su(difference_graph / c)) * su(su(norm_distance ** 3))
    current_cap = 8 * F.from_float(cfg.a) * su(F(V)) * light_j
    contact_cap = 8 * F.from_float(cfg.a) ** 2 * su(F(V)) * light_jj
    current_error = current_cap * weighted_error
    contact_error = contact_cap * weighted_error
    current_expectation = current_cap * (weighted_error + norm_distance * packet_rho + (1 + rho) * eta)
    contact_expectation = contact_cap * (weighted_error + norm_distance * packet_rho + (1 + rho) * eta)
    q_modes = []
    for target in (F(1, 100), F(3, 100), F(6, 100), F(12, 100)):
        n = int(target * N * F.from_float(cfg.a) / (2 * pi_hi))
        if 2 * pi_lo * n / (N * F.from_float(cfg.a)) < F(1, 100):
            n += 1
        if 2 * pi_hi * n / (N * F.from_float(cfg.a)) > F(12, 100):
            n -= 1
        q_lower = 2 * pi_lo * n / (N * F.from_float(cfg.a))
        q_upper = 2 * pi_hi * n / (N * F.from_float(cfg.a))
        guard(f'physical_q_{n}_above_point01', q_lower - F(1, 100))
        guard(f'physical_q_{n}_below_point12', F(12, 100) - q_upper)
        q_modes.append({'integer_q': [n, 0, 0], 'physical_q_lower': exact_endpoint(q_lower), 'physical_q_upper': exact_endpoint(q_upper),
                        'witness': 'Machin pi=16atan(1/5)-4atan(1/239), 64 alternating rational terms plus exact next-term remainder'})
    electron_modes = []
    for row in sectors:
        integers = [0 if t < 2 else -1 for t in row['twist']]
        reduced = [F(n) + F(t, 4) for n, t in zip(integers, row['twist'])]
        components = []
        for shift in reduced:
            endpoints = (2 * pi_lo * shift / (N * F.from_float(cfg.a)), 2 * pi_hi * shift / (N * F.from_float(cfg.a)))
            components.append({'lower': exact_endpoint(min(endpoints)), 'upper': exact_endpoint(max(endpoints))})
        electron_modes.append({'twist': row['twist'], 'integer_p': integers,
            'sector_reduced_momentum': [exact_endpoint(s) for s in reduced],
            'physical_p_component_intervals': components,
            'meaning': 'Actual finite sector twisted fundamentalCAR nearest-rest momentum; t2 half-mode tie chooses negative, so222 is anti-periodic in all3 axes; bosonicq remains periodic'})
    values = dict(volume=F(V), dimension=F(D), fast_dimension=F(Dfast), eta=eta,
        orbit_rmin=rmin, orbit_rmax=rmax, scalar_path_heavy_cap=heavy, scalar_path_all_charged_cap=light,
        K1=K1, K2=K2, K3=K3, universal_all_q_static_C=C, heavy_gap_lower=heavy_gap,
        smallest_laplacian_lower=lapmin, smallest_laplacian_upper=lapmin_upper,
        Phi_fast_scaled_variance_upper=phi_variance, colour_transverse_scaled_variance_upper=transverse_variance,
        colour_longitudinal_scaled_variance_upper=longitudinal_variance, fast_variance_upper=fast_variance,
        fast_mass_lower=hfast, Schur_denominator_lower=schur_den, separate_mass_margin_lower=separate,
        normal_gap_lower=gap, fast_energy_upper=fast_energy, slow_energy_upper=slow_energy,
        harmonic_reprepare_energy_excess=harmonic_excess, cut_norm_squared_deficit_upper=deficit,
        shape_rounding_energy_excess=shape_price, cut_energy_excess=cut_excess,
        principal_metric_cap=M, Haar_FP_scalar_action_upper=U, nonlinear_form_relative=eps,
        mean_force_action_upper=Fforce * eta, well_spectral_state_distance_upper=spectral_distance,
        well_energy_upper=well_upper, weighted_energy_budget=budget, shell_potential_lower=shell,
        weaker_barrier_lower=barrier, ramp_slope_upper=slope, ramp_kinetic_price_upper=ramp_price,
        Agmon_exponent_lower=Agmon, weighted_norm_squared_excess_upper=weighted_norm2 - 1,
        weighted_gradient_squared_upper=weighted_grad2, normalized_boundary_prefactor_upper=boundary_prefactor,
        boundary_action_upper=boundary, Berry_D1_upper=D1, Berry_D2_upper=D2,
        Gauss_charge_norm_upper=charge_norm, heavy_Berry_Gauss_affine_action_upper=fibre_action,
        charged_EM_mean_freezing_action_upper=charged_action, reference_EM_CAR_graph_norm_upper=reference_graph,
        full_Hamiltonian_lower_shift=B, quartic_graph_coercivity=c, difference_graph_norm_upper=difference_graph,
        weighted_Phi_action_error=weighted_error, all_flavour_current_action_error=current_error,
        all_flavour_contact_action_error=contact_error, all_flavour_current_expectation_error=current_expectation,
        all_flavour_contact_expectation_error=contact_expectation, physical_state_norm_distance_upper=norm_distance,
        density_trace_norm_distance_upper=2 * norm_distance)
    lower_names = {'orbit_rmin', 'heavy_gap_lower', 'smallest_laplacian_lower', 'fast_mass_lower',
        'Schur_denominator_lower', 'separate_mass_margin_lower', 'normal_gap_lower',
        'shell_potential_lower', 'weaker_barrier_lower', 'Agmon_exponent_lower', 'quartic_graph_coercivity'}
    exact_counts = {'volume', 'dimension', 'fast_dimension'}
    prices = {name: exact_endpoint(value) if name in exact_counts else endpoint(value, name not in lower_names)
              for name, value in values.items()}
    descriptor = {'mean_input_sha256': mean_capsule['input_sha256'], 'mean_capsule_sha256': hashlib.sha256(json.dumps(mean_capsule, sort_keys=True).encode()).hexdigest(),
        'finite_size': N, 'Lambda_exponent': exponent, 'clock': clock, 'cutoff_bits': cutoff_bits,
        'microscopic_config': config, 'sector_policy': mean_capsule['sector_policy'],
        'state': 'prepared_uniform_64_density_mixture_exact_implicit_full_G', 'bound_arithmetic': 'exact rational/isqrt80; outward dyadic receipts'}
    state = {
        'centre': 'Each sector has its own exact implicit finite10 root G_t; C_t,mu=sum_nu [sqrt(G_t)]_mu,nu SELECTED_nu with full quarter-cross terms; never a decimal seed or cubic replacement.',
        'slow_covariance': 'Let O(C_t) be the 60x15 commutator orbit and N0 an exact orthonormal basis of ker O(C_t)^T. Hslow(G_t) is the full45 zero-q classical heavy+charged+quartic Cartesian Hessian in N0. Cov_slow=Lambda^(-3/4) Hslow(G_t)^(-1/2)/2, exact spectral calculus.',
        'fast_covariance': 'For periodic q=2pi n/N, Phi45(q!=0) covariance Lambda^(-7/8) I45/[2sqrt(lap/5)]. For each exact R0=O(C_t)^T O(C_t) eigenvalue r, colour covariance Lambda^(-7/8)[.3/sqrt(.12r+.21lap) PT+(.6+lap/r)/(2sqrt(.12r+.2lap)) PL]. At q0 use three identical .3/sqrt(.12r) modes; q0 Phi45 is slow only. Exact spectral projectors including degeneracies, not numerically rounded eigenvectors.',
        'normalization': 'All45 slow and 90V-45 fast real orthogonal Fourier modes. Radial b=(1-r^2/eta^2)^2_+, normalized by its exact positive integral. Density is (1/64)sum_t |Psi_t><Psi_t|; exact scalar E_D,t phases cancel separately.',
        'joint_map': 'Phi_x=V_x(C_t+N0Y_x)V_x^dagger; U_xy=V_x z_xy,t exp(iTheta_xy.T) V_y^dagger. Central-seam loops are z_j=exp(2pi i t_j/4). Only diagonalZ4 residual. w_joint=Vol(SU4)^(V-3V) w_local/4 with normalized Haar; W=w_joint^(-1/2) equivariant extension is isometric since z^Ntotal=1. No local4^V projector.',
        'heavy': 'NH=2V upper-h occupied normalized gapped Slater section; Hamiltonian -Lambda RotationBand(.2,.12,.3,improved=False), neutral EM.',
        'charged_EM': 'Exact all64 charged CAR Hamiltonian at actual implicit fullG_t and actual SU4 central seams, active1 m.2 plus aux63 m2. Initial matter is the exact negative-energy spectral-projector Slater for all64 species at the sector centre and seams, tensor actual onsite Ref4=r1dag r2dag r3dag r4dag|0> at0. Original integer background b_x=128+4delta_x0 neutralizes this fixed total-number state, with totalnumbermod4=0. EM L2(R^(2V+1)) product normalized (1-(theta_j/(1/8))^2)^2_+; real-gradient Gauss/Coulomb, UEphys1 and UElink1/a. Every one/two-link Peierls factor is unitary for arbitrary real quantum A; no compactcircle.',
        'prepared_density_policy': 'Arbitrary uniform64 preparation, not energy degeneracy, thermal equilibrium or global ground. Each component comparison uses its own exact E_D,t and reference; convexity gives density distance <=2 max component norm distance.',
        'consumer_electron_p_t': 'In sector t take n_j=0 for t_j=0,1 and n_j=-1 for t_j=2,3; p_t,j=2pi(n_j+t_j/4)/(Na). This is the actual twisted fundamentalCAR nearest-rest electron;222 has negative anti-periodic half-modes. No extra baseline halfshift; bosonic q remains periodic.'}
    ledger = {
        'uniform_static_C': 'Scalar-path row/column K1,K2 bound actual derivatives. For normalized Fourier displacement Parseval gives sum_x |v_x|^2=1; squared matrix-element trace <=4K1^2, onsite/contact trace <=2K2, and resolvent gap>=Delta. Thus occupied2V trace and Fourier1/V cancel. C=12kappa(1+rho)^2+16K2heavy+64K1heavy^2/Deltaheavy plus Lambda^-1 times the active1+aux63 analogues bounds every native all3-species+quartic block, allq and all64 twists. No N6 sample maximum.',
        'orbit': 'Generic Clifford C=sqrtG: R0 eigenvalues are half nonempty subset sums of four eigenvaluesG with multiplicity15. Therefore rmin>=lambda_min(G)/2, rmax<=traceG/2.',
        'fast_covariance_bound': 'Periodic bosonic ACTUAL lapmin=4sin(pi/N)^2 is enclosed using Machin rational pi intervals and alternating sin Taylor x-x^3/6+x^5/120-x^7/5040 <=sin(x)<=x-x^3/6+x^5/120 at x=pi/N, then exact outward96-bit dyadic floors. lap<=12. Phi variance upper uses this actual first-mode lower enclosure, unlike colour-only N6 bound. Canonical original .3chi E4^2 gives .6chi cometric.',
        'mass_sandwich': 'Young 2Lambda C |s||f| <=Lambda amin |s|^2/4+4Lambda C^2 |f|^2/amin. Executed separate masses and Schur guard precede minmax. Heavy third derivative uses occupied2V[128K1^3/Delta^2+32K1K2/Delta+K3]; quartic third24kappa(rho+1). Alignment cubic acts on fast coordinates only; 256(1+rho)^2 eta/hfast; curl4096eta/rmin.',
        'exact_root_force': 'Full coupled finite10 stationarity makes classical mean force the negative fast-zero-point gradient. Generic R0 eigenvalues r are half subset sums, hence their Frobenius G Lipschitz constants <=1; each root omega=sqrt(.12r+c lap) has |domega/dr|=.06/omega<=1/sqrt(rmin). ||dG/dCart||<=2rho and allD real modes therefore give the conservative displayed100D(1+rho)^2/[rmin sqrt(rmin)] bound on this branch. Charged scalar-path Lipschitz additionally overbounds charged freeze. No assumed residual 1e-90 and no numerical rounding centre.',
        'spectral_minmax': 'Identical exact Vmean,t, Efast,t and Eslow,t cancel algebraically BEFORE comparing excess. Positive fast/slow energy bounds are not huge rounded phases. X=harmonic+cut excess, P=eps(Etrial+gap0)+Feta+U; gap>=gap0-4P-2X, excess<=X+2P; sqrt(2excess/gap) bounds trial-ground distance. Exact spectral shapes incur ZERO truncation price by definition, not absent data.',
        'metric_measure': 'Complex chart radius min(.001,rmin/[16sqrt rmax]); metric M=4+512/rmin, M1=8M/R; logmeasure first8/R, second32/R^2. Cauchy per-site bounds summed with fullD^2 scalar-price. Real homogeneous ellipticity>=3/5; original colour Casimir maintained.',
        'Agmon': 'Ramp begins eta/8, first flat-cut derivatives eta/4, separationeta/8. Slope upper eta sqrt(Lambda amin/(chi M))/64; conservative exponent lower eta^2 sqrt(Lambda amin/(chi M))/1024. Actual shell=(1-eps)Lambda amin eta^2/256-Feta-U. Execute shell/8>=rampprice, shell-Ewell-rampprice>=barrier/2, barrier>16Ewell. Weighted norm1+2budget/barrier; fullD^2 commutator prefactor includes normalization2, grad8/eta and Hess96/eta^2. e^-A<=2^-A prices entire tail.',
        'Berry_Gauss': 'D1=4sqrtD(2V)K1/Delta; D2=D[64(2V)^2K1^2/Delta^2+16(2V)K2/Delta]; charge66sqrt(5V/2). Full dimension and volume are explicitly paid.',
        'graphs': 'Reference graph is conserved at ALLtimes by exact Href. Full graph of lifted reference is bounded at ALLtimes by unitary paths, cut support and Berry bounds; full graph of actual evolved state is conserved. BOTH graphs plus lower shift enter difference cap. Primitive UE/a retained; real Coulomb inverse norm<=1/[4sin(pi/N)^2]. Exact H2 product has p^2 mean3/R^2, norm(p^2f)^2=63/(2R^4).',
        'Heinz': 'Retain half quartic: H+B>=c rhoPhi^4, c=Lambda/(8aV). Executed Lmatter>=c implies Young lower shift B=1+Lmatter^2/c dominates max_r(Lmatter r-c r^4). Square-root monotonicity and spectral interpolation give ||rhoPhi delta||<=c^-1/4 ||delta||^3/4 ||(H+B)delta||^1/4. Current/contact caps use actual allflavour pathlength and pathlength^2 sums, not mock forwarding.',
        'bound_witness': 'All numerical operations use Fraction; sqrt endpoints use isqrt floor on an exactly scaled rational with80 binary significant bits. Recorded endpoints are outward power2 enclosures. All positivity guards compare exact rational numbers. No floating exponential, sampled minimum, rounded centre or numericalphase.'}
    stable = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in SOURCE_HASHES.items()}
    if not all(stable.values()):
        raise RuntimeError('An imported dependency changed during physical calculation')
    return {'branch': 'NEW actual finite macroN262144 original-law fullG all64 prepared Phi/colour transfer',
        'input_descriptor': descriptor, 'input_sha256': hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(),
        'finite_size': N, 'Lambda_exponent': exponent, 'microscopic_config': config,
        'exact_mode_counts': {'volume': V, 'total_real_bosonic': D, 'uniform_slow': 45,
            'fast': Dfast, 'noncompact_EM': em_modes, 'heavy_occupied': occupied},
        'exact_centre_definition': mean_capsule['exact_centre_definition'], 'centre_enclosure': enc,
        'mean_prerequisite_provenance': {key: mean_capsule[key] for key in (
            'branch', 'input_descriptor', 'input_sha256', 'source_at_import', 'own_source_at_import',
            'runtime', 'elapsed_seconds', 'actual_finite_grid_quadrature_certificate')},
        'physical_state_definition': state, 'all64_state_policy': mean_capsule['sector_policy'],
        'all64_exact_root_rows': [{'twist': row['twist'], 'exact_root_definition': row['exact_root_definition']} for row in sectors],
        'prices': prices, 'executed_positive_guards': guards, 'derivation_ledger': ledger,
        'actual_scalar_path_rows': [{'role': role, 'load': load, 'rows': p['rows']} for load, _, role, p in bands],
        'charged_cluster_bounds': charged_cluster_rows,
        'compatible_integer_physical_q_modes': q_modes,
        'all64_near_rest_electron_momenta': electron_modes,
        'actual_periodic_first_mode_certificate': {'integer_q': [1, 0, 0],
            'exact_laplacian_definition': '4sin(pi/262144)^2',
            'lower': exact_endpoint(lapmin), 'upper': exact_endpoint(lapmin_upper),
            'witness': 'Exact Machin pi bounds, alternating sin Taylor with explicit remainder, outward96-bit dyadic floor; these endpoints are USED in every Phi variance, mass, Schur and Coulomb bound'},
        'source_at_import': SOURCE_HASHES, 'own_source_at_import': {'path': Path(__file__).name, 'sha256': SOURCE_HASHES[Path(__file__).name], 'text': OWN_BYTES.decode()},
        'source_unchanged_since_import': stable,
        'runtime': {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__, 'platform': platform.platform()},
        'elapsed_seconds': time.monotonic() - start,
        'proof_status': 'Executed volume-uniform exact-rational constants and guards for explicit exact-root mathematical state; comparison lemmas displayed for independent parent mathematical review, not manybody numerical simulation.',
        'limitations': ['Prepared local-well density mixture, not ground or selected thermodynamic phase.', 'Finite macrobox, not bulk substitution or continuum limit.', 'Transfer only to exact charged/noncompactEM reference at actual fullG_t; no Gaussian-RPA photonvacuum/commoncone/Compton admission.', 'Canonical hbar1/CAR/CCR and microscopic parameters are declared inputs.', 'Old finiteSCREEN/delta13 receives no new state or gate.']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mean', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clock', type=int, default=4096)
    parser.add_argument('--cutoff-bits', type=int, default=2048)
    args = parser.parse_args()
    raw = args.mean.read_bytes()
    report = certify(json.loads(raw), args.clock, args.cutoff_bits)
    report['mean_file_sha256'] = hashlib.sha256(raw).hexdigest()
    report['mean_file_path'] = str(args.mean.resolve())
    if args.mean.read_bytes() != raw:
        raise RuntimeError('Mean input changed during physical calculation')
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'receipt': str(args.output), 'input_sha256': report['input_sha256'],
        'own_source_sha256': SOURCE_HASHES[Path(__file__).name], 'guards': len(report['executed_positive_guards']),
        'physical_state_norm_distance_upper': report['prices']['physical_state_norm_distance_upper'],
        'elapsed_seconds': report['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()

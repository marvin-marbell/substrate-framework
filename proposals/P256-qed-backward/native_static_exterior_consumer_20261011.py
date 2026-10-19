"""Owned native static-exterior gates; no native eigensolver or U backend.
Exact algebra/scaling, deterministic native radial integration and raw Gauss
geometry are distinct from uncomputed actual-ground dynamics.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
import sys
import time
import zipfile
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc, gammaincc, gammaln

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_record(value: Fraction) -> dict[str, Any]:
    return {'numerator_bits': value.numerator.bit_length(),
            'denominator_bits': value.denominator.bit_length(),
            'sha256': digest(str(value).encode())}


def coefficients(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    lam = Fraction(2**8192)
    chi = Fraction(1, 2**4096)
    fast = Fraction(2**10240)
    assert [int(p['load']) for p in profiles] == [1, 1, 63]
    w = [abs(Fraction(p['onsite'])) + 2*sum(
         abs(Fraction(path['c0']))+abs(Fraction(path['ci']))
         for path in p['paths']) for p in profiles]
    b = 4*(lam*w[0]+w[1]+63*w[2])
    assert b < 512*lam
    return {'all65_loads': [1, 1, 63], 'B': exact_record(b),
            'B_lt_512lambda': True, 'alignment_alpha': exact_record(fast/40),
            'kinetic_Phi': exact_record(4*chi),
            'kinetic_Haar': exact_record(3*chi/20), 'kinetic_fullE': 64,
            'onsite_and_literal_path_loads': [str(x) for x in w],
            'bounded_intercell_envelope': 'J_M=4alpha M^2+2B M+8beta',
            'same_original_coefficients': True}


def cutoff_gates(request: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for m in request['field_cutoffs']:
        # Best-case field cone after absorbing fixed law/time constants;
        # the actual native constants cannot improve its M exponent.
        radius = m*m
        patch = (2*radius+1)**3
        probability_bound = Fraction(patch, m**4)
        rows.append({'M': m, 'radius_div_fixed_native_time_constant': radius,
                     'cube_patch_sites': patch,
                     'union_tail_squared_div_C4': str(probability_bound),
                     'norm_tail_bound_div_sqrt_C4': math.sqrt(float(probability_bound)),
                     'single_site_quadratic_action_tail_squared_div_C4': 1,
                     'linear_hop_tail_squared_div_B2_C4': str(Fraction(1, m*m)),
                     'fullE_kinetic_action_tail_from_second_moment': 'unbounded'})
    assert rows[-1]['norm_tail_bound_div_sqrt_C4'] > rows[0]['norm_tail_bound_div_sqrt_C4']
    # Exact integer scaling of the competing constraints R=M^a.
    exponents = [{'a': str(a), 'norm_union_M_exponent': str(3*a/2-2),
                  'cone_requires_a_ge_2': a >= 2,
                  'union_requires_a_lt_4over3': a < Fraction(4, 3)}
                 for a in (Fraction(1), Fraction(4, 3), Fraction(2), Fraction(3))]
    return {'status': 'FAILED_AVAILABLE_ESTIMATE_NOT_ACTUAL_GROUND_NO_GO',
            'scope': 'best-case bounded-interaction field cone; kinetic cutoff and static-core membership not granted',
            'rows': rows, 'exact_exponents': exponents,
            'higher_moment_norm_action_total_triangle_M_exponent': '8-p/2; p>16 needed for this estimate',
            'higher_moment_union_M_exponent': '3-p/2; p>6 needed',
            'time': request['times'],
            'N_requirement': 'even N>4(R+r_A+2), no wrap; larger N allowed; small N not used to infer limit',
            'actual_cutoff_evolved_moment_budget_computed': False}


def native_quadratic_tail(cutoffs: list[int]) -> dict[str, Any]:
    # Invariant 60-coordinate native Phi Gaussian at one site has s=|q|^2/2
    # distributed Gamma(30, variance). Shell mixture probability M^-4.
    # Neighbour Gaussian has mean s=1. On s_x>M^2/4, s_y<=M^2/16,
    # ||Phi_x-Ad(U)Phi_y||^4 >= s_x^2/16 for EVERY SU4 U.
    q32 = float(gammaincc(32, 7.5))
    rows = []
    for m in cutoffs:
        neighbour_probability = float(gammainc(30, 30*m*m/16))
        lower = Fraction(31, 480)*Fraction.from_float(q32*neighbour_probability)
        total_align_squared = Fraction(31, 30)*Fraction((m*m+1)**2, m**4)
        rows.append({'M': m, 'mixture_high_probability': str(Fraction(1, m**4)),
                     'high_variance': str(Fraction(m*m, 30)),
                     'high_local_s2_mixture': str(Fraction(31, 30)),
                     'complete_single_alignment_action_squared_div_alpha2_high_component': str(total_align_squared),
                     'alignment_tail_squared_div_alpha2_lower': float(lower),
                     'gamma_Q32_at_7point5': q32,
                     'neighbour_event_probability': neighbour_probability})
    assert rows[-1]['alignment_tail_squared_div_alpha2_lower'] > 0.06
    assert rows[-1]['high_local_s2_mixture'] == rows[0]['high_local_s2_mixture']
    return {'status': 'NATIVE_NONSTATIONARY_MOMENT_IMPLICATION_COUNTERCASE',
            'dimension_per_site': 60, 'state': 'normalized physical Gaussian/Haar/affine canonical mixture; NOT a minimizing ground',
            'source_norm': 1, 'all65_CAR_and_fullE_remain': True,
            'complete_alignment': 'alpha sum_mu ||Phi_mu,x-U Phi_mu,y U*||HS^2',
            'SU4_uniform_lower_bound_not_frozen_link': True,
            'rows': rows, 'analytic_limit_lower': '(31/480)*Q(32,7.5)>0',
            'tail_scope': 'single retained native alignment action; not total-H cancellation or actual-ground failure',
            'kinetic_and_local_form_budgets': 'Gaussian Kphi=60chi/variance; mixture uniformly bounded; all other cells variance1/30; fullE/checkerboard and Haar retained',
            'actual_ground_failure': False}


def raw_geometry(n: int) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    v = n**3
    index = {x: j for j, x in enumerate(sites)}
    d = np.zeros((v, 3*v))
    for x in sites:
        j = index[x]
        for axis in range(3):
            y = list(x)
            y[axis] = (y[axis]+1) % n
            d[j, 3*j+axis] = 1
            d[index[tuple(y)], 3*j+axis] = -1
    parity = np.array([(-1)**sum(x) for x in sites], dtype=float)
    rho = 128*parity
    k = np.zeros(3*v)
    k[::3] = 64*parity
    # Pure checkerboard has a pointwise zero weighted cross under equal
    # link-tail weights. Add an actual closed integer circulation whose two
    # opposite tails have different distance to the declared weight centre.
    loop = np.zeros(3*v)
    base = (n-1, 0, 0)
    loop[3*index[base]] = 1
    loop[3*index[(0, 0, 0)]+1] = 1
    loop[3*index[(n-1, 1, 0)]] = -1
    loop[3*index[base]+1] = -1
    assert np.max(abs(d@loop)) == 0
    k += loop
    assert np.max(abs(d@k-rho)) == 0
    assert int(sum(sum(x) % 2 == 0 for x in sites))*4 == 2*v
    # Solve the actual connected lattice Poisson equation with its constant
    # mode fixed, not a Hamiltonian eigenproblem or producer rerun.
    lap = d@d.T
    fix = np.ones((v, v))/v
    t = np.zeros(3*v)
    t[0] = -1
    tl = d.T@np.linalg.solve(lap+fix, d@t)
    tt = t-tl
    exterior = np.ones(3*v, dtype=bool)
    exterior[0] = False
    cancellation = float(np.max(abs(tt[exterior]+tl[exterior])))
    assert cancellation < 1e-12
    theta = np.array([math.exp(-math.log(2)*sum(min(z, n-z) for z in x)/4)
                      for x in sites])
    edge_theta = np.repeat(theta, 3)
    kl = d.T@np.linalg.solve(lap+fix, rho)
    kt = k-kl
    cross = 128*float(np.dot(edge_theta*kt, kl))
    weighted = 64*float(np.dot(edge_theta*k, k))
    separated = 64*float(np.dot(edge_theta*kt, kt)+np.dot(edge_theta*kl, kl))
    assert abs(weighted-separated-cross) < 1e-7
    assert abs(cross) > 1
    direct_delta = 64*float(2*np.dot(k, t)+np.dot(t, t))
    transverse_delta = 64*float(2*np.dot(kt, tt)+np.dot(tt, tt))
    longitudinal_delta = 64*float(2*np.dot(kl, tl)+np.dot(tl, tl))
    assert abs(direct_delta-transverse_delta-longitudinal_delta) < 1e-7
    return {'N': n, 'V': v, 'each_species_canonical': 2*v, 'species': 65,
            'Ref4_inert': True, 'fullrho_collective64': True,
            'Gauss_max_defect': float(np.max(abs(d@k-rho))),
            'path_affine_delta_fullE': direct_delta,
            'separate_transverse_delta': transverse_delta,
            'separate_Coulomb_delta': longitudinal_delta,
            'nonzero_exterior_longitudinal_norm': float(np.linalg.norm(tl[exterior])),
            'fullE_exterior_cancel_max': cancellation,
            'weighted_fullE': weighted, 'weighted_cross': cross,
            'closed_integer_circulation_added': True,
            'pure_checkerboard_cross_zero_is_special_not_generic': True,
            'wrong_separated_weighted_energy': separated,
            'geometry_only_not_quantum_evolution': True}


def weighted_exterior(request: dict[str, Any]) -> dict[str, Any]:
    eps = math.log(2)/4
    rows = []
    for radius in request['radii']:
        # On Z3 the l1 shell has 4r^2+2 sites. This exact geometric sum is
        # a uniform periodic upper envelope; it is NOT excitation leakage.
        z = math.exp(-eps)
        start = radius+1
        mass = z**start*((4*start*start+2)/(1-z)
                         +8*start*z/(1-z)**2+4*z*(1+z)/(1-z)**3)
        inverse = math.exp(eps*(radius+2))
        rows.append({'radius': radius, 'weighted_background_tail_mass': mass,
                     'inverse_weight_boundary_distance_Rplus2': inverse,
                     'one_shell_unweighted_action_form_cost_div_E': inverse,
                     'background_subtraction_is_not_positive_energy_density': True})
    assert rows[-1]['weighted_background_tail_mass'] < rows[0]['weighted_background_tail_mass']
    assert rows[-1]['one_shell_unweighted_action_form_cost_div_E'] > rows[0]['one_shell_unweighted_action_form_cost_div_E']
    return {'status': 'WEIGHTED_FORM_PROPAGATION_VALID_BUT_EXTERIOR_GATE_NOT_CLOSED',
            'epsilon': eps, 'actual_paid_budget': 'qQtheta(U(t)A Omega)<=exp(Cstar|t|)[66a^2 B0+66d_A+16384lambda Sstar a^2]',
            'no_prior_closed_form_debt_reopened': True,
            'rows': rows,
            'quadratic_alignment_unweighted_boundary_norm_cost': 'at least the available extraction pays sqrt(2E/lambda)*exp(epsilon(R+2)/2) per endpoint; sums add boundary incidence',
            'scalar_multiplication_weighted_tail_is_not_static_patch_projection': True,
            'missing': 'positive excitation-local density/current or conditional physical ground-jet exterior decay; subtracting ground local energy from Qtheta is not a positive form'}


def conditional_ground_jet(request: dict[str, Any]) -> dict[str, Any]:
    # Full native [H,A] for scalar radial A=exp(-s0) is exactly [Kphi,A].
    # All remaining native kinetic/potential/matter terms commute with A.
    # Physical radial two-site entangled state uses normalized 60-D Gaussians
    # f_a(x)f_a(y)+f_b(x)f_b(y), not a ground approximation.
    shape = 30
    va, vb = 1/30, 2/30
    overlap = (2*math.sqrt(va*vb)/(va+vb))**shape
    norm2 = 2+2*overlap*overlap
    def density(s: float, variance: float) -> float:
        if s <= 0:
            return 0.0
        return math.exp((shape-1)*math.log(s)-s/variance-gammaln(shape)-shape*math.log(variance))
    def residual_integrand(s: float) -> float:
        pa, pb = density(s, va), density(s, vb)
        denominator = pa+pb+2*overlap*math.sqrt(pa*pb)
        if denominator == 0:
            return 0.0
        return 4*s*s*math.exp(-2*s)*(1/va-1/vb)**2*(1-overlap**2)*pa*pb/denominator/norm2
    def complete_integrand(s: float) -> float:
        pa, pb = density(s, va), density(s, vb)
        # Delta A=(2s-60)exp(-s), and score kinetic term
        # -2s exp(-s)(pa amplitude/va+pb amplitude/vb).
        ca = 60-2*s-2*s/va
        cb = 60-2*s-2*s/vb
        return math.exp(-2*s)*(ca*ca*pa+cb*cb*pb+2*ca*cb*overlap*math.sqrt(pa*pb))/norm2
    residual, err = quad(residual_integrand, 0, 20, epsabs=1e-11, epsrel=1e-11, limit=200)
    complete, complete_err = quad(complete_integrand, 0, 20, epsabs=1e-10, epsrel=1e-11, limit=200)
    assert residual > 1e-4 and complete > residual
    def damped_tail_moment(variance: float, degree: int) -> float:
        rising = math.prod(range(shape, shape+degree))
        return ((1+2*variance)**(-shape)
                *(variance/(1+2*variance))**degree*rising
                *float(gammaincc(shape+degree, (1/variance+2)*20)))
    residual_tail_bound = (4*(1/va-1/vb)**2*(1-overlap**2)/norm2
                           *damped_tail_moment(vb, 2))
    complete_tail_bound = ((1+overlap)/norm2*sum(
        7200*damped_tail_moment(variance, 0)
        +2*(2+2/variance)**2*damped_tail_moment(variance, 2)
        for variance in (va, vb)))
    vc = 2*va*vb/(va+vb)
    local_s2 = shape*(shape+1)*(va*va+vb*vb+2*overlap**2*vc*vc)/norm2
    local_phi_kinetic_div_kphi = (60/(4*va)+60/(4*vb)
                                 +2*overlap**2*60/(2*(va+vb)))/norm2
    controls = [{'radius': radius, 'remote_site_distance': radius+4,
                 'admissible_even_N_min': 4*(radius+6),
                 'source_norm_squared': 1,
                 'best_static_patch_commutator_residual_squared_div_kphi2': residual,
                 'complete_fullH_commutator_action_squared_div_kphi2': complete}
                for radius in request['radii']]
    return {'status': 'SECOND_MATERIALLY_ATTEMPTED_NATIVE_STATIC_CONDITIONAL_PROJECTION',
            'native_fullH_commutator': '[H,exp(-s0)]=-kphi(2s0-60)exp(-s0)-2kphi grad(exp(-s0)).grad',
            'all65_Gauss_fullE_Haar_alignment_Jordan_plaquette_retained_by_exact_commutation': True,
            'state_kind': 'nonstationary normalized invariant radial two-site superposition times original canonical/Haar/affine section; NOT actual ground',
            'local_multipliers_full_declared_MW_cannot_change_remote_Phi': True,
            'gaussian_overlap': overlap, 'normalization_squared_before_division': norm2,
            'residual_squared_div_kphi2': residual, 'quadrature_absolute_error_estimate': err,
            'complete_action_squared_div_kphi2': complete, 'complete_action_quadrature_error_estimate': complete_err,
            'residual_tail_after_s20_squared_div_kphi2_upper_formula_value': residual_tail_bound,
            'complete_action_tail_after_s20_squared_div_kphi2_upper_formula_value': complete_tail_bound,
            'native_local_s2_exact_formula_value': local_s2,
            'native_local_Phi_kinetic_div_kphi_exact_formula_value': local_phi_kinetic_div_kphi,
            'positive_residual_analytic': '(1-z^2)(va^-1-vb^-1)^2 pa pb/(pa+pb+2z sqrt(pa pb))>0 on s>0',
            'analytic_tail_after_s20': 'Explicit damped Gamma tail bound retained; floating gamma evaluation and quadrature are not interval certificates',
            'rows': controls,
            'native_local_moments_uniform_in_distance_N': 'finite fixed va/vb; <s^2> and kinetic <=finite two-Gaussian Cauchy envelope; canonical fullE density262144; Haar0 expectation but operator retained',
            'inference': 'weighted/higher fixed moments alone do not force uniform static conditional-jet approximation on arbitrary physical states',
            'actual_ground_failure': False,
            'unpaid_actual_ground_inequality': 'sup_admissible_N inf_B_in_THIS_core(K_R) ||[H_N,A]Omega_N-B Omega_N|| ->0 plus iterated local graph/propagation stability; must use exact-ground equation or native ground correlations'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    request = json.loads(args.request.read_text())
    before = {name: digest((ROOT/name).read_bytes()) for name in request['pins']}
    if before != request['pins']:
        raise ValueError('source or input pin mismatch')
    with zipfile.ZipFile(ROOT/request['epoch_archive']) as bundle:
        profile_raw = bundle.read(request['profile'])
        profiles = json.loads(profile_raw)['primitive_profiles']
    result = {'schema': 1, 'scope': 'NATIVE_STATIC_EXTERIOR_TWO_ATTEMPTED_GATES_WITH_EXPLICIT_UNPAID_GROUND_INEQUALITY',
              'coefficients': coefficients(profiles), 'cutoff_gate': cutoff_gates(request),
              'native_quadratic_action_tails': native_quadratic_tail(request['field_cutoffs']),
              'raw_fullE_Gauss': [raw_geometry(n) for n in request['N']],
              'weighted_exterior_attempt': weighted_exterior(request),
              'conditional_ground_jet_attempt': conditional_ground_jet(request),
              'proof_sha256': before[request['proof']], 'profile_member_sha256': digest(profile_raw),
              'source_input_sha256_before': before,
              'source_input_sha256_after': {name: digest((ROOT/name).read_bytes()) for name in request['pins']},
              'runtime': {'python': sys.executable, 'numpy': np.__version__, 'elapsed_seconds': time.monotonic()-start},
              'actual_ground_U_numerics': False, 'same_static_GNS_exterior_closed': False,
              'same_static_resolvent_range_proved': False, 'global_no_go': False,
              'phase_stiffness_IR_atom': False}
    if result['source_input_sha256_after'] != before:
        raise RuntimeError('input mutation')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'cutoff_gate': result['cutoff_gate']['status'],
                      'second_native_residual_div_kphi2': result['conditional_ground_jet_attempt']['residual_squared_div_kphi2'],
                      'actual_ground_failure_claimed': False, 'static_exterior_closed': False,
                      'elapsed_seconds': result['runtime']['elapsed_seconds']}))


if __name__ == '__main__':
    main()

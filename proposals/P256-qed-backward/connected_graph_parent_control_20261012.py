"""Independent exact review-boundary controls; no native ground or phase backend."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(name: str) -> str:
    with (ROOT / name).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lag_cumulant() -> dict[str, Any]:
    # Unique ground, norm-one insertion. Enumerate all 24 literal label orders.
    import itertools
    gamma, eta = sp.symbols('gamma eta', positive=True)
    u1, u2, u3 = sp.symbols('u1 u2 u3', nonnegative=True)
    L = sp.diag(0, gamma)
    omega = sp.Matrix([1, 0])
    V = sp.Matrix([[0, 1], [1, 0]])
    heat = lambda u: sp.diag(1, sp.exp(-gamma * u))
    full = cast(Any, (omega.T * V * heat(u3) * V * heat(u2) * V * heat(u1) * V * omega))[0]
    two_point = lambda u: sp.exp(-gamma * u)
    disconnected = (
        two_point(u1) * two_point(u3)
        + two_point(u1 + u2) * two_point(u2 + u3)
        + two_point(u1 + u2 + u3) * two_point(u2)
    )
    connected = sp.simplify(full - disconnected)
    assert connected == -2 * sp.exp(-gamma * (u1 + 2 * u2 + u3))
    orders = list(itertools.permutations(range(4)))
    summed = sum((connected for _ in orders), sp.Integer(0))
    assert sp.simplify(summed + 48 * sp.exp(-gamma * (u1 + 2 * u2 + u3))) == 0
    # Each exponential integral is exact, over the whole positive octant.
    integral = -48 / ((eta + gamma)**2 * (eta + 2 * gamma))
    example = sp.factor(integral.subs({gamma: sp.Rational(1, 16), eta: 1}))
    assert example == -sp.Rational(32768, 867) and abs(example) > 26
    assert abs(example) < 624
    t = sp.symbols('t', real=True)
    ground_energy = (gamma - sp.sqrt(gamma**2 + 4*t**2)) / 2
    fourth_energy = sp.diff(ground_energy, t, 4).subs(t, 0)
    assert sp.simplify(sp.limit(integral, eta, 0, dir='+') + fourth_energy) == 0
    # Exact partition-coefficient absolute total; finite cube has 1/24 simplex.
    stirling = (1, 7, 6, 1)
    coefficient_total = sum(stirling[k-1] * sp.factorial(k-1) for k in range(1, 5))
    assert coefficient_total == 26
    assert len(orders) * coefficient_total == 624
    assert len(orders) * sp.Rational(1, 24) == 1
    return {'method_only_unique_ground': True, 'ordered_connected': str(connected),
            'all24_lag_integral': str(integral), 'counter_gamma': '1/16', 'counter_eta': '1',
            'counter_value': str(example), 'summed_26_bound_rejected': True,
            'safe_summed_lag_coefficient': 624, 'averaged_lag_coefficient': 26,
            'averaging_changes_target': True, 'finite_cube_26_tau4_unaffected': True,
            'zero_eta_matches_negative_ground_energy_fourth_derivative': str(fourth_energy),
            'physical1_phase_or_no_go': False}


def closed_electric_ground_boundary() -> dict[str, Any]:
    # Distinguish the accepted native closed-E antecedent from general degeneracy.
    L = sp.diag(0, 0, 3)
    P0 = sp.diag(1, 1, 0)
    Q0 = sp.eye(3) - P0
    omega = sp.Matrix([1, 0, 0])
    E = sp.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    F = sp.Matrix([[0, 1, sp.I], [1, 0, 0], [-sp.I, 0, 0]])
    J = -sp.I * (L * E - E * L)
    assert P0 * E * omega == sp.zeros(3, 1)
    assert L * E * omega == sp.I * J * omega
    epsilon = sp.symbols('epsilon', positive=True)
    source_cross = cast(Any, (F * omega).H * Q0 * E * omega)[0]
    current = cast(Any, (Q0 * F * omega).H * (L + epsilon*sp.eye(3)).inv() * Q0 * J * omega)[0]
    assert sp.simplify(sp.limit(current, epsilon, 0, dir='+') + sp.I * source_cross) == 0
    stationary = cast(Any, (F * omega).H * P0 * E * omega)[0]
    assert stationary == 0
    # Generic hopping/observable source may still have a stationary component.
    V = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    assert P0 * V * omega != sp.zeros(3, 1)
    E_bad = E + V
    assert P0 * E_bad * omega != sp.zeros(3, 1)
    inverse = sp.diag(0, 0, sp.Rational(1, 3))
    inverse_value = cast(Any, (E * omega).H * inverse * E * omega)[0]
    assert inverse_value == sp.Rational(1, 3)
    return {'nonnative_boundary_control': True, 'closed_source_wholeP0_zero': True,
            'same_L_LE_equals_iJ': True, 'static_current': str(sp.limit(current, epsilon, 0, dir='+')),
            'generic_V_stationary_block_not_deleted': True,
            'arbitrary_degenerate_E_bad_does_not_satisfy_native_antecedent': True,
            'inverse_matrix_control': str(inverse_value),
            'native_ground_or_native_inverse_computed': False}


def circle_and_complex_controls() -> dict[str, Any]:
    r = sp.symbols('r', positive=True)
    theta = sp.symbols('theta', real=True)
    # Normalized periodic real section; a twist multiplies its phase, not density.
    density = (1 + sp.cos(theta))**2 / (3 * sp.pi)
    normalization = sp.integrate(density, (theta, -sp.pi, sp.pi))
    kinetic = sp.integrate(sp.sin(theta)**2 / (3 * sp.pi), (theta, -sp.pi, sp.pi))
    assert normalization == 1 and kinetic == sp.Rational(1, 3)
    marginal = 1 + sp.Rational(4, 3)*r + sp.Rational(1, 3)*r**2
    squared = (1+r*r)/(1-r*r) + sp.Rational(4, 3)*r*(2+2*r*r/(1-r*r)) + sp.Rational(1, 3)*r*r*(3+2*r*r/(1-r*r))
    assert sp.limit((1-r)*squared, r, 1, dir='-') == sp.Rational(8, 3)
    assert marginal.subs(r, sp.Rational(1, 2)) == sp.Rational(7, 4)
    assert sp.Rational(8, 3)/sp.pi < 2/sp.sqrt(3)
    G = sp.Matrix([[1, sp.I], [-sp.I, 1]])
    vector = sp.Matrix([1, -sp.I])
    full = cast(Any, vector.H * G * vector)[0]
    deleted = cast(Any, vector.H * sp.eye(2) * vector)[0]
    assert full == 4 and deleted == 2
    return {'circle_normalization': str(normalization), 'circle_p2': str(kinetic),
            'one_Poisson_expectation': str(marginal), 'two_Poisson_expectation': str(sp.factor(squared)),
            'two_Poisson_divergence_residue': '8/3', 'local_H1_does_not_suppress_seam': True,
            'complete_complex_quadratic': str(full), 'wrong_real_only_quadratic': str(deleted),
            'nonnative_circle_and_Gram_not_ground_order': True}


def accepted_joint_moment_counter() -> dict[str, Any]:
    rows = []
    for n in (1, 2, 4, 8, 16, 32):
        probability = Fraction(1, 4*n*2**(9*n))
        radius = Fraction(2**n)
        electric2 = Fraction(2**(4*n), n*n)
        electric4_term = probability * electric2**2
        mixed_term = probability * radius**8 * electric2
        assert electric4_term == Fraction(1, 4*n**5*2**n)
        assert mixed_term == Fraction(2**(3*n), 4*n**3)
        if n > 1:
            prior = Fraction(2**(3*(n-1)), 4*(n-1)**3)
            assert mixed_term >= prior
        rows.append({'n': n, 'p': str(probability), 'E4_term': str(electric4_term),
                     'Phi8_E2_term': str(mixed_term)})
    # Infinite-series proofs, not inference from these finite samples:
    # E4 <= (1/4) sum 2^-n = 1/4; radial t<9 is geometric/n.
    # Phi8 E2 terms = 2^(3n)/(4n^3) do not tend to zero.
    return {'method_probability_not_native_ground': True,
            'p_n': '2^(-9n)/(4n)', 'R_n': '2^n', 'E_n_squared': '2^(4n)/n^2',
            'all_radial_t_less_than_9_finite': True,
            'h_p_for_all_p_less_than_9_over_4_finite': True,
            'accepted_E4_finite_upper': '1/4', 'Phi8_E2_diverges_by_term_test': True,
            'old_E2_only_counter_not_all_accepted_budgets': True, 'sample_terms': rows,
            'operator_cancellations_or_ground_equation_not_excluded': True,
            'native_fullgraph_impossibility': False}


def virial_direction_control() -> dict[str, Any]:
    rows = []
    for variance in (1, 4, 16):
        a = sp.Rational(variance)
        # Gaussian variance a with quadratic phase k*x^2/2.
        k2 = sp.Rational(18, 7)*a - 1/(28*a*a)
        assert k2 > 0
        x4, x8, x10 = 3*a*a, 105*a**4, 945*a**5
        weighted_kinetic = (k2 + 1/(4*a*a))*x8
        positive_rhs = 14*weighted_kinetic
        quantum = 105*x4
        potential = 4*x10
        assert sp.simplify(positive_rhs - quantum - potential) == 0
        assert potential > 0 and -quantum < 0
        rows.append({'variance': variance, 'phase_k_squared': str(k2),
                     'positive_weighted_kinetic': str(positive_rhs),
                     'quantum_correction': str(quantum), 'potential_virial': str(potential),
                     'identity_residual': '0', 'drop_positive_kinetic_upper_false': True})
    return {'nonnative_Schwartz_not_ground': True, 'H': 'p^2+x^4', 'v': 'x^7',
            'i_H_D': '14 p x^6 p -105 x^4 -4 x^10', 'rows': rows,
            'single_virial_stationarity_not_ground': True,
            'parent_radial10_upper_inference_withdrawn': True,
            'native_monotone_alignment_or_form_compatibility_not_withdrawn': True}


def fractional_controls() -> dict[str, Any]:
    s = sp.Rational(1, 34)
    assert 0 < s < sp.Rational(1, 17)
    beta = (1-17*s)/4
    assert beta == sp.Rational(1, 8)
    # Exact scalar gapped source: all upper graph/positive-band regularity can
    # coexist with zero spectral mass near zero.
    L = sp.diag(0, 0, 16)
    P0 = sp.diag(1, 1, 0)
    u = sp.Matrix([1, 2, 1])
    j = (sp.eye(3)-P0)*u
    assert j == sp.Matrix([0, 0, 1])
    assert cast(Any, j.H * j)[0] == 1
    assert cast(Any, j.H * sp.diag(0, 0, sp.Rational(1, 17)) * j)[0] == sp.Rational(1, 17)
    # Exact residual endpoint error demonstrates why the K^2 term is needed.
    G, delta, eta = sp.Rational(1), sp.Rational(1, 2), sp.Rational(1)
    actual_error = ((G+delta)**2-G**2)/eta
    linear_only = 2*G*delta/eta
    safe = (2*G*delta+delta**2)/eta
    assert actual_error == safe and actual_error > linear_only
    return {'s': str(s), 'dyadic_beta': str(beta), 'strict_endpoint_not_paid': True,
            'wholeP0_not_mean': True, 'gapped_source_satisfies_upper_bounds': True,
            'no_positive_IR_mass_from_fractional_graph': True,
            'weighted_Gram_error': str(actual_error), 'wrong_linear_only': str(linear_only),
            'quadratic_cutoff_error_required': True, 'nonnative_spectral_control': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    before = {name: sha(name) for name in request['pins']}
    assert before == request['pins']
    result = {'kind': 'INDEPENDENT_PARENT_CONNECTED_WARD_CUMULANT_JOINT_MOMENT_VIRIAL_FRACTIONAL_BOUNDARY_CONTROLS_NOT_PHASE',
              'lag': lag_cumulant(), 'ground_boundary': closed_electric_ground_boundary(),
              'circle_complex': circle_and_complex_controls(), 'joint_moments': accepted_joint_moment_counter(),
              'virial': virial_direction_control(), 'fractional': fractional_controls(),
              'source_input_sha256_before': before,
              'source_input_sha256_after': {name: sha(name) for name in request['pins']},
              'native_ground_or_native_fullH_evolution_computed': False,
              'physical1_phase_stiffness_IR_atom': False, 'campaign_complete': False}
    assert result['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print('Exact Ward/cumulant/joint-moment/virial/fractional adverse boundaries exercised; no phase inference.')


if __name__ == '__main__':
    main()

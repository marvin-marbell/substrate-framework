"""Bind frozen physical-history/reset epochs and exercise joined spectral boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exact(v: F) -> dict[str, Any]:
    return {'sha256': hashlib.sha256(str(v).encode()).hexdigest(),
            'numerator_bits': v.numerator.bit_length(),
            'denominator_bits': v.denominator.bit_length()}


def epoch(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    for key in ('pre', 'executed', 'receipt', 'capsule', 'exit'):
        assert digest(ROOT / spec[key]) == spec[key + '_sha256'], (key, spec[key])
    with zipfile.ZipFile(ROOT / spec['pre']) as pre, zipfile.ZipFile(ROOT / spec['executed']) as exe:
        names = pre.namelist()
        assert len(names) == len(set(names)) and set(names) == set(spec['member_map'])
        for name in names:
            assert pre.read(name) == exe.read(spec['member_map'][name]), name
        raw = exe.read(spec['receipt_member'])
        assert raw == (ROOT / spec['receipt']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == spec['receipt_sha256']
        exit_raw = exe.read(spec['exit'])
        assert hashlib.sha256(exit_raw).hexdigest() == spec['exit_sha256']
        run: dict[str, Any] = json.loads(exit_raw)
        assert run['natural_exit'] is True and run['returncode'] == 0
        payload: dict[str, Any] = json.loads(raw)
    capsule: dict[str, Any] = json.loads((ROOT / spec['capsule']).read_text())
    stability = spec.get('stability', {})
    if stability.get('mode') == 'capsule_flag':
        assert capsule[stability['flag']] is True
    for owner in (payload, capsule, run):
        for before, after in (('source_input_sha256_before', 'source_input_sha256_after'),
                              ('before', 'after')):
            if before in owner:
                assert owner[before] == owner[after]
    return payload, {'members_compared': len(names), 'natural_exit': True, 'returncode': 0,
                     'elapsed_seconds': run.get('elapsed_seconds'),
                     'immutable_source_inputs_unchanged': True}


def native_constants(request: dict[str, Any], payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    with zipfile.ZipFile(ROOT / request['profile_archive']) as archive:
        raw = archive.read(request['profile_member'])
    profiles: list[dict[str, Any]] = json.loads(raw)['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    # Different contraction order from the producers: collect the two charged
    # roles per path before summing lengths. Never average active and auxiliary.
    role_terms = [sum((F(p['load']) * F(t['length']) *
                      (abs(F(t['c0'])) + abs(F(t['ci'])))
                      for t in p['paths'] if t['axis'] == 0), F(0))
                  for p in profiles[1:]]
    ce = 8 * sum(role_terms, F(0))
    assert str(ce) == payloads['electric']['native_costs']['C_e']
    assert str(ce) == payloads['current_domain']['native_per_word_current_cutoff_cost']['charged_C_e']
    assert ce < 2**14
    lam, chi, fast = F(2**8192), F(1, 2**4096), F(2**10240)
    widths = [abs(F(p['onsite'])) + 2 * sum((abs(F(t['c0'])) + abs(F(t['ci']))
                                           for t in p['paths']), F(0)) for p in profiles]
    b = 4 * (lam * widths[0] + widths[1] + 63 * widths[2])
    r = 60 * chi + 11192 * lam + F(129, 10) * fast + 262144
    b0 = 33 * (1 + 10 * r * 4096) + 16384 * lam * 4096
    assert b < 512 * lam and widths[1] != widths[2]
    common_hop_ce = 8 * 64 * role_terms[0]
    assert common_hop_ce == ce  # Literal hopping profiles coincide; onsite roles do not.
    wrong_active_body = 4 * (lam * widths[0] + 64 * widths[1])
    assert wrong_active_body != b
    return {'scope': 'exact native coefficient and inherited budget recomposition, not measured moments',
            'C_e': str(ce), 'charged_role_terms': [str(v) for v in role_terms],
            'all65_B': exact(b), 'translation_averaged_B0': exact(b0), 'R0': exact(r),
            'common_hop_current_64_active_identity': True,
            'wrong_64_active_body_budget_rejected': True,
            'profile_member_sha256': hashlib.sha256(raw).hexdigest()}


def reset_join(payload: dict[str, Any]) -> dict[str, Any]:
    stages = [F(1), F(5, 4), F(3, 2), F(7, 4), F(2)]
    stored = payload['ground_variational_and_cutoff_gates']['two_threshold_bootstrap_strict_stages']
    rows = []
    for i, p in enumerate(stages):
        gamma = min(p - F(2, 3), p / 2 + F(1, 4))
        supremum = 1 + gamma
        assert F(stored[i]['two_threshold_result_h_moment_supremum_not_endpoint']) == supremum
        if i + 1 < len(stages):
            assert stages[i + 1] < supremum
        rows.append({'input_p': str(p), 'tail_gamma': str(gamma),
                     'next_moment_strict_supremum': str(supremum)})
    assert rows[-1]['next_moment_strict_supremum'] == '9/4'
    # Correct h-weighted tail integration, with an exactly integrable density.
    x, m = sp.symbols('x m', nonnegative=True)
    density: Any = sp.exp(-x)
    moment: Any = sp.integrate((1 + x)**2 * density, (x, 0, sp.oo))
    first: Any = sp.integrate((1 + x) * density, (x, 0, sp.oo))
    tail: Any = (m + 1) * sp.exp(-(m - 1))
    contribution: Any = sp.integrate(tail, (m, 1, sp.oo))
    assert moment == first + contribution == 5
    assert first + 2 * contribution == 8
    # At |K|~M^6: radial8 crop norm2 M^-2, ground form M^-4;
    # inserted-source graph antecedent gives form M^-1, not graph error.
    ims = {'patch_power': 6, 'radial_moment': 8, 'ground_norm2_power': 6 - 8,
           'ground_form_power': 6 - 10, 'inserted_form_power': F(6, 2) - 4,
           'quadratic_action_triangle_power': 4, 'linear_action_triangle_power': 3}
    assert ims['ground_form_power'] < 0 and ims['inserted_form_power'] < 0
    assert ims['quadratic_action_triangle_power'] > 0
    return {'conditional_native_bootstrap': rows, 'radial_order_paid': 8,
            'Casimir_order_paid': 2, 'higher_graph_lift_not_assumed': True,
            'exact_tail_factor': 'p-1', 'tail_factor_method_exact_moment': str(moment),
            'wrong_tail_equality_method_value': '8',
            'IMS_exponents': {k: str(v) for k, v in ims.items()},
            'U_is_excitation_unitary': 'exp(-it(H-Eglob))',
            'static_exterior_range_paid': False, 'numerical_ground_moments': False}


def electric_band_budget(request: dict[str, Any], payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    with zipfile.ZipFile(ROOT / request['profile_archive']) as archive:
        profiles: list[dict[str, Any]] = json.loads(archive.read(request['profile_member']))['primitive_profiles']
    w = sum((F(p['load']) * F(t['length']) * (abs(F(t['c0'])) + abs(F(t['ci'])))
             for p in profiles[1:] for t in p['paths']), F(0))
    ce = 8 * sum((F(p['load']) * F(t['length']) *
                  (abs(F(t['c0'])) + abs(F(t['ci'])))
                 for p in profiles[1:] for t in p['paths'] if t['axis'] == 0), F(0))
    assert str(w) == payloads['electric_band']['reset_exact_and_adverse']['W_exact']
    assert str(ce) == payloads['electric']['native_costs']['C_e']
    # Accepted sharp budget is HS radial fourth, not Cartesian fourth:
    # (sum ||Phi_mu||HS^2)^2 = |qCart|^4/4 < 2^15.
    cphi4 = F(2**15)
    assert 16**4 > cphi4 and 256**2 > cphi4
    d = 128 * w * 16
    c1 = 1025 + d  # R=4 variational high-only reset, p_R <= 1.

    def three_quarter_upper(c: F) -> F:
        b = c.numerator.bit_length() - c.denominator.bit_length()
        if c > F(2)**b:
            b += 1
        u = F(2)**((3*b + 3)//4)
        assert u**4 >= c**3
        return u

    assert 33**2 > 1025 and F(16)**4 > F(65, 4)**3
    a1 = d * 16 * three_quarter_upper(c1)
    c3half = 33*c1 + 2*a1
    assert F(64)**8 > F(65, 4)**9
    a3half = d * 64 * three_quarter_upper(c3half)
    denominator1 = 1 + 3*F(1)/4 - F(3, 2)
    denominator2 = 1 + 3*F(3, 2)/4 - 2
    assert denominator1 == F(1, 4) and denominator2 == F(1, 8)
    assert 1 + 3*F(1)/4 - F(7, 4) == 0
    c2 = 1025*c1 + 8*a3half
    ce4 = c2/256
    qj = (4*ce)**2 * 16 * 256

    def sqrt_upper(c: F) -> int:
        u = math.isqrt(c.numerator//c.denominator)
        if F(u*u) < c:
            u += 1
        assert F(u*u) >= c
        return u

    harmonic_a, harmonic_b = sqrt_upper(ce4), F(sqrt_upper(qj), 128)
    # Exact nonnative source-loss witness. Uniform E2 alone allows lost mass.
    n = sp.symbols('n', positive=True, integer=True)
    rho: Any = sp.diag(1 - 1/n**2, 1/n**2)
    electric: Any = sp.diag(0, n)
    second = sp.simplify(sp.trace(rho*electric**2))
    fourth = sp.simplify(sp.trace(rho*electric**4))
    assert second == 1 and fourth == n**2
    # Scalar Taylor bound from E4: ||(E-X_s)Omega|| <= |s| sqrt(CE4)/2.
    # At e=4,s=pi/4 the actual squared error is16+64/pi^2.
    assert F(16) + F(64, 9) < F(36)  # pi>3 bounds both sides outward.
    quotient_error2: Any = 16 + 64/sp.pi**2
    # Uniform E4 still cannot identify finite excitation mass after collapse.
    omega: Any = sp.Matrix([1, 0])
    bounded_e: Any = sp.Matrix([[0, 1], [1, 0]])
    zero_block: Any = sp.eye(2)
    finite_zero: Any = sp.diag(1, 0)
    source: Any = bounded_e*omega
    assert (bounded_e**4*omega) == omega
    assert ((sp.eye(2)-finite_zero)*source).norm()**2 == 1
    assert ((sp.eye(2)-zero_block)*source).norm()**2 == 0
    return {'scope': 'conditional same-ground sharp electric reset budget and total source-kernel passage, not measured moments',
            'W_exact': str(w), 'Cphi4_HS_upper': str(cphi4),
            'D_upper': str(d), 'C1prime_upper': str(c1),
            'C3half_upper': str(c3half), 'C2prime_upper': str(c2),
            'CE4prime_upper': str(ce4), 'QJ_face_upper': str(qj),
            'harmonic_A_upper': str(harmonic_a), 'harmonic_B_upper': str(harmonic_b),
            'strict_denominators': [str(denominator1), str(denominator2)],
            'uniform_source_quotient_error': '|s| sqrt(CE4)/2',
            'total_EE_kernel_error': '(|sa|+|sb|) CE4^(3/4)/2 for equal face caps',
            'method_uniform_E2_lost_mass': str(second),
            'method_fourth_moment_diverges': str(fourth),
            'method_quotient_error2': str(quotient_error2),
            'finite_Q_mass': '1', 'limit_Q_mass_after_zero_collapse': '0',
            'finite_P0_equals_limit_P0': False,
            'direct_E2_source_norm_preserved_from_E4': False,
            'actual_ground_IR_covariance_or_atom_computed': False}


def same_generator_controls() -> dict[str, Any]:
    # Exact nonnative method, with a TWO-dimensional zero block and complex
    # mixed moments. No finite matrix is substituted for the native ground.
    h: Any = sp.diag(0, 0, 3, 5, 7)
    p0: Any = sp.diag(1, 1, 0, 0, 0)
    q0: Any = sp.eye(5) - p0
    e: Any = sp.Matrix.hstack(sp.Matrix([0, 1, 1, sp.I, 1]),
                             sp.Matrix([0, 2, sp.I, 1, -1]))
    xi: Any = q0 * e
    j: Any = -sp.I * h * e
    s: Any = xi.H * xi
    m1: Any = xi.H * h * xi
    m2: Any = xi.H * h * h * xi
    assert j.H * j == m2
    ej: Any = xi.H * j
    assert sp.simplify(ej + sp.I * m1) == sp.zeros(2)
    assert sp.simplify(j.H * xi - sp.I * m1) == sp.zeros(2)
    contact: Any = 2 * m1.applyfunc(sp.re)
    assert sp.im(m1[0, 1]) != 0 and contact[0, 1] != 2 * m1[0, 1]
    wrong_mean: Any = sp.eye(5) - sp.diag(1, 0, 0, 0, 0)
    assert e.H * wrong_mean * e - s == sp.Matrix([[1, 2], [2, 4]])
    projection: Any = xi * s.inv() * xi.H
    bath: Any = q0 - projection
    force: Any = bath * h * xi
    c: Any = bath * h * bath
    assert force != sp.zeros(5, 2)
    schur_rows = []
    for z in (sp.Rational(-2), sp.Rational(-7, 2)):
        sigma: Any = force.H * (z * sp.eye(5) - c).inv() * force
        direct: Any = xi.H * (z * sp.eye(5) - h).inv() * xi
        compressed: Any = s * (z * s - m1 - sigma).inv() * s
        assert sp.simplify(direct - compressed) == sp.zeros(2)
        assert sp.simplify(direct - s * (z * s - m1).inv() * s) != sp.zeros(2)
        schur_rows.append({'z': str(z), 'resolvent_rows': [[str(v) for v in row]
                                                         for row in direct.tolist()],
                           'omitted_orthogonal_force_rejected': True})
    epsilon = sp.Rational(1, 8)
    filt: Any = epsilon * (h + epsilon * sp.eye(5)).inv() * q0
    principal: Any = sp.Matrix([0, 1, 2, -sp.I, 1])
    filtered: Any = filt * principal
    one_filter: Any = principal.H * filt * j[:, 0]
    two_filters: Any = principal.H * filt * filt * j[:, 0]
    assert one_filter != two_filters and filtered.H * j[:, 0] == one_filter
    assert (filt * sp.Matrix([0, 1, 0, 0, 0])).norm() == 0
    delta: Any = sp.Matrix([0, 1, 2, 3, 4])
    graph_error: Any = (filt * delta).norm()**2 + (h * filt * delta).norm()**2
    assert graph_error <= (1 + epsilon**2) * delta.norm()**2
    return {'scope': 'exact nonnative joint-spectral/domain/compression method controls',
            'whole_zero_block_vs_mean_Gram': [['1', '2'], ['2', '4']],
            'm1_rows': [[str(v) for v in row] for row in m1.tolist()],
            'real_contact_does_not_fix_imaginary_m1': True,
            'EJ_sign_minus_i_and_JE_plus_i': True,
            'nonorthogonal_Schur': schur_rows,
            'single_filtered_cross': str(one_filter[0]),
            'wrong_double_filtered_cross': str(two_filters[0]),
            'fixed_epsilon_graph_error2': str(graph_error),
            'native_Gram_IR_atom_or_ground_computed': False}


def reviewed_scopes_and_typed_transfer(request: dict[str, Any]) -> dict[str, Any]:
    reviews: dict[str, Any] = {}
    for label, item in request['reviews'].items():
        proof: dict[str, Any] = json.loads((ROOT/item['path']).read_text())
        assert digest(ROOT/item['path']) == item['sha256']
        verdict = proof[item['verdict_field']]
        assert 'VALID' in verdict and 'INVALID' not in verdict
        reviews[label] = {'proof_sha256': item['sha256'], 'verdict': verdict}
    transfers: dict[str, Any] = {}
    for label, pair in request['typed_payload_pairs'].items():
        original: dict[str, Any] = json.loads((ROOT/pair['original']).read_text())
        successor: dict[str, Any] = json.loads((ROOT/pair['successor']).read_text())
        excluded = set(pair['excluded_metadata_only'])
        assert excluded <= {'runtime', 'source_input_sha256_before', 'source_input_sha256_after',
                            'before', 'after', 'request_sha256'}
        left = {k: v for k, v in original.items() if k not in excluded}
        right = {k: v for k, v in successor.items() if k not in excluded}
        assert left == right, label
        transfers[label] = {'original_payload_sha256': digest(ROOT/pair['original']),
                            'successor_payload_sha256': digest(ROOT/pair['successor']),
                            'mathematical_payload_exact_equal': True}
    return {'independent_frozen_reviews': reviews, 'six_typed_transfers': transfers,
            'original_reviews_targets_not_rebound': True,
            'ground_scope': 'Inherited analytic globally minimizing compatible characters with translation averaging',
            'magnetic_source': 'F=principal_angle(U)/2, norm<=pi/2, first harmonic1; not doubled',
            'original_all64_control': 'Representative native CAR factor repeated as role bookkeeping, not64 independent physical-state executions',
            'tagged_branch_fix': 'NEW source-selected actual3; original far-label bug and actual2 request ambiguity preserved',
            'TOTAL_EE_kernel_identified': True, 'finite_P0_equals_limiting_P0': False,
            'direct_E_squared_source_norm_conserved': False,
            'Ec_squared_Omega_in_D_L': False, 'J_Omega_in_D_E': False,
            'physical1_phase_atom_or_commoncone': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    request: dict[str, Any] = json.loads(args.request.read_text())
    before = {name: digest(ROOT / name) for name in request['pins']}
    assert before == request['pins']
    payloads: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for label, spec in request['epochs'].items():
        payloads[label], bindings[label] = epoch(spec)
    assert payloads['principal']['physical1_phase_stiffness_pole_admission'] is False
    receipt = {'kind': 'JOINED_FULL_HISTORY_OPERATOR_RESET_NATIVE_RECOMPOSITION_NOT_PHASE',
               'epochs': bindings, 'members_compared': sum(v['members_compared'] for v in bindings.values()),
               'native_constants': native_constants(request, payloads),
               'reset_bootstrap_and_IMS_join': reset_join(payloads['reset']),
               'electric_band_sharp_budget_and_total_kernel_join': electric_band_budget(request, payloads),
               'same_generator_method': same_generator_controls(),
               'reviewed_scopes_and_exact_typed_transfer': reviewed_scopes_and_typed_transfer(request),
               'full_history_operator_reviews_pending': False,
               'electric_band_magnetic_defect_reviews_pending': False,
               'native_fullH_numerical_evolution_or_ground_computed': False,
               'same_static_GNS_invariance_onto_range': False,
               'physical1_phase_stiffness_IR_atom': False,
               'original_SOURCE_adopted': False, 'campaign_complete': False,
               'source_input_sha256_before': before,
               'source_input_sha256_after': {name: digest(ROOT / name) for name in before},
               'runtime': {'elapsed_seconds': time.monotonic() - started}}
    assert receipt['source_input_sha256_before'] == receipt['source_input_sha256_after']
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'epochs': len(bindings),
                      'members': receipt['members_compared'], 'phase_or_atom': False}))


if __name__ == '__main__':
    main()

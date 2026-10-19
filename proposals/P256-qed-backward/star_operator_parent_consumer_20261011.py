"""Integrate immutable native star/charged/central enclosures, not a physical phase."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from math import isqrt
from pathlib import Path
import time
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(name: str) -> Any:
    return json.loads((ROOT / name).read_text())


def interval(encoded: dict[str, Any]) -> tuple[F, F]:
    lower, upper = map(F, encoded['exact'])
    assert lower <= upper
    return lower, upper


def add(left: tuple[F, F], right: tuple[F, F]) -> tuple[F, F]:
    return left[0] + right[0], left[1] + right[1]


def scale(value: tuple[F, F], factor: F) -> tuple[F, F]:
    products = value[0] * factor, value[1] * factor
    return min(products), max(products)


def sqrt_upper(value: F) -> F:
    assert value >= 0
    denominator = 1 << 256
    floor = isqrt((value.numerator * denominator * denominator) // value.denominator)
    result = F(floor, denominator)
    if result * result < value:
        result += F(1, denominator)
    assert result * result >= value
    return result


def snapshot(capsule_name: str) -> dict[str, Any]:
    capsule = load(capsule_name)
    assert capsule.get('exit_code', capsule.get('returncode')) == 0
    assert capsule.get('natural_subprocess_exit', capsule.get('natural_exit')) is True
    if 'executed' in capsule:
        archive_name, expected_archive = capsule['executed']['file'], capsule['executed']['sha256']
    else:
        archive_name = capsule.get('executed_zip', capsule_name.replace('_capsule.json', '_executed.zip'))
        expected_archive = capsule.get('executed_sha256', capsule.get('executed_zip_sha256'))
    assert digest(ROOT / archive_name) == expected_archive
    pins = capsule.get('input_and_source_sha256', capsule.get('input_sha256', {}))
    pins = pins | capsule.get('artifacts_sha256', {}) | capsule.get('output_sha256', {})
    checked = []
    with zipfile.ZipFile(ROOT / archive_name) as archive:
        names = archive.namelist()
        for name, expected in pins.items():
            candidates = [member for member in names if Path(member).name == name]
            assert candidates, (archive_name, name)
            member = next((m for m in candidates if hashlib.sha256(archive.read(m)).hexdigest() == expected), None)
            assert member is not None, (archive_name, name, expected)
            checked.append({'name': name, 'member': member, 'sha256': expected,
                            'live_equal': (ROOT / name).exists() and digest(ROOT / name) == expected})
    return {'capsule': capsule_name, 'capsule_sha256': digest(ROOT / capsule_name),
            'archive': archive_name, 'archive_sha256': expected_archive,
            'original_members': checked, 'natural_exit': True, 'original_epochs_not_rebound': True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    request = json.loads(args.request.read_text())
    assert all(digest(ROOT / name) == pin for name, pin in request['pins'].items())
    capsules = ('charged_outer_vector_actual1_20261011_exit.json',
                'central_neutral_vertex_actual2_20261011_capsule.json',
                'star_factor_review_actual2_exit.json',
                'star_resolvent_review_capsule_20261011.json')
    assert all(name in request['pins'] for name in capsules)
    snapshots = [snapshot(name) for name in capsules]
    outer = load('charged_outer_vector_interface_20261011.json')
    central = load('central_neutral_vertex_interface_20261011.json')
    factor = load('star_factor_review_findings_20261011.json')
    resolvent = load('star_resolvent_review_verdict_20261011.json')
    normalized = load('normalized_defect_parent_actual2_20261011_receipt.json')
    quantities = load('matching_defect_interface_20261011.json')['quantities']
    gamma, eg = F(quantities['gamma_exact']), F(quantities['E_g_upper_exact'])
    b, two_cell = -5120 - eg, 64 - 10240 - 2 * eg
    assert two_cell == F(1249546155809, 17179869184) > 0
    errors = []
    for row in outer['axes']:
        a = F(row['anchor_exact'])
        coefficients = [F(x) for x in row['polynomial_coefficients_exact']]
        assert coefficients == [2 / a, -1 / (a * a)]
        m0, m1, m2, _, _ = [interval(x) for x in row['all1_plus63_both_orientations_moments_g']]
        variance_upper = m2[1] - 2 * a * m1[0] + a * a * m0[1]
        epsilon, previous = F(row['epsilon_upper_exact']), F(row['frozen_scalar_epsilon_exact'])
        assert epsilon * epsilon >= max(F(0), variance_upper) * max(F(1), a / two_cell - 1) ** 2 / a ** 4
        assert 0 < epsilon < previous and two_cell < a < 2 * two_cell
        errors.append({'axis': row['axis'], 'epsilon_exact': str(epsilon), 'previous_scalar_exact': str(previous),
                       'whole_spectrum_variance_upper_exact': str(variance_upper), 'strict_improvement': True})
    responses = {}
    for name, row in central['central_q_resolvents'].items():
        m0, m1, m2 = [interval(x) for x in row['moments_exact_intervals']]
        c = F(row['inverse_candidate_exact'])
        residual = add(add(m0, scale(m1, -2 * c)), scale(m2, c * c))
        assert residual == interval(row['residual_square'])
        response = interval(row['sourceweighted_response'])
        assert 0 <= response[0] <= response[1] <= m0[1] / gamma
        assert row['all_spectrum_tail_paid'] is True
        responses[name] = response
    real_polynomial = scale(add(responses['plus'], scale(responses['minus'], F(-1))), F(1, 2))
    imaginary_polynomial = scale(add(responses['minusi'], scale(responses['plusi'], F(-1))), F(1, 2))
    assert real_polynomial == interval(central['polynomial_neutral_2Re_cross'])
    assert imaginary_polynomial == interval(central['polynomial_neutral_2Im_cross'])
    radius = F(central['actual_cross_transfer_radius_exact'])
    for component, polynomial in [('2Re', real_polynomial), ('2Im', imaginary_polynomial)]:
        assert add(polynomial, (-radius, radius)) == interval(central['actual_outer_resolvent_' + component + '_cross'])
    single_floor = 64 + 2 * b - 35
    assert single_floor == F(central['excited_transfer']['uniform_operator_single_floor_exact']) > 0
    norms = list(map(F, central['outer_ground_row_norm_caps_exact']))
    relative = list(map(F, central['excited_transfer']['uniform_excited_form_relative_caps_exact']))
    assert relative == [norm * norm / (single_floor * gamma) for norm in norms]
    assert all(cap > 1 for cap in relative)
    profiles = load('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json')['profiles']
    upper = [sqrt_upper(F(p['e_squared_exact'])) for p in profiles]
    lower = [F(p['e_squared_exact']) / u for p, u in zip(profiles, upper)]
    free = -4 * upper[0] - 252 * upper[1]
    removal = min(l - abs(F(p['c0_exact'])) for p, l in zip(profiles, lower))
    mixed = central['mixed_nonzero_flow_domains']
    assert F(mixed['opposite_sign_floor_exact']) == 128 + 2 * b
    assert F(mixed['same_sign_floor_exact']) <= 128 + 2 * b + free + 2 * removal - eg
    assert mixed['centerPgB_allowed_and_not_deleted'] is True
    role_control = central['controls']['actual_unequal_role_commutator'][1]
    assert role_control['wrong_template_active_h_difference'] > role_control['conditional_formation_allowance']
    assert role_control['rational_vs_frozen_native_action_L1_difference'] <= role_control['conditional_formation_allowance']
    assert central['formation']['cell_residual_including_formation'] == '1/65536'
    assert factor['overall_verdict'] == 'CONDITIONAL_VALID_ON_FROZEN_MATCHING_SOURCE_GROUND_DOMAIN_AND_BINARY64_OPERATION_MODEL'
    real_cross = interval(central['actual_outer_resolvent_2Re_cross'])
    assert real_cross[0] < 0 < real_cross[1]
    assert normalized['physical1']['sufficient_package_passes'] is False
    result = {'kind': 'ACTUAL_IMMUTABLE_NATIVE_OPERATOR_PARENT_COMPOSITION', 'snapshots': snapshots,
              'charged_errors': errors, 'charged_reducing_domain_floor_exact': str(two_cell),
              'central_sourceweighted_responses': {n: [str(x) for x in v] for n, v in responses.items()},
              'neutral_cross_exact': [str(x) for x in real_cross], 'neutral_sign_proved': False,
              'wholecentral_single_floor_exact': str(single_floor), 'uniform_excited_relative_caps_exact': [str(x) for x in relative],
              'uniform_excited_contraction_at_physical1': False,
              'scopes': {'sourceweighted_matrix_is_not_whole_excited_bound': True,
                         'charged_floor_not_arbitrary_charged_Fock': True, 'mixed_center_ground_retained': True,
                         'whole_P0_not_sourceQ_or_flow0': True, 'ground_branch_not_physicalPhi': True},
              'inherited_resolvent_verdict': resolvent['overall_verdict'], 'conditional_machine_model_not_attestation': True,
              'historical_solver_or_producer_runs': 0, 'campaign_complete': False,
              'open': ['physicalPhi/lattice stationary ground', 'physical1 normalized linked phase',
                       'stiffness/IR atom/sharp photon', 'common cone', 'original SOURCE'],
              'elapsed_seconds': time.monotonic() - started}
    assert all(digest(ROOT / name) == pin for name, pin in request['pins'].items())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'neutral_cross': [float(x) for x in real_cross], 'relative_caps': [float(x) for x in relative],
                      'immutable_member_count': sum(len(s['original_members']) for s in snapshots),
                      'physical1': False, 'campaign_complete': False}), flush=True)


if __name__ == '__main__':
    main()

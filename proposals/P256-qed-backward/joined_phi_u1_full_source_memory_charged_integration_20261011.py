"""Immutable-epoch composition, not a native solver or phase certificate."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

sys.set_int_max_str_digits(0)
ROOT = Path(__file__).resolve().parent


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_positive_ldl(matrix: list[list[int]]) -> list[F]:
    """Independent rational congruence of a symmetric lower enclosure."""
    n = len(matrix)
    assert all(len(row) == n for row in matrix)
    assert all(matrix[i][j] == matrix[j][i] for i in range(n) for j in range(n))
    lower = [[F(int(i == j)) for j in range(n)] for i in range(n)]
    pivots: list[F] = []
    for j in range(n):
        d = F(matrix[j][j]) - sum(lower[j][k] ** 2 * pivots[k] for k in range(j))
        assert d > 0, ('nonpositive exact pivot', j, d)
        pivots.append(d)
        for i in range(j + 1, n):
            lower[i][j] = (F(matrix[i][j]) - sum(lower[i][k] * lower[j][k] * pivots[k] for k in range(j))) / d
    assert all(sum(lower[i][k] * pivots[k] * lower[j][k] for k in range(n)) == matrix[i][j]
               for i in range(n) for j in range(n))
    return pivots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    own = digest(Path(__file__).read_bytes())
    manifest_raw = (ROOT / args.manifest).read_bytes()
    manifest: dict[str, Any] = json.loads(manifest_raw)
    expected: dict[str, str] = manifest['files']
    captured = {name: (ROOT / name).read_bytes() for name in expected}
    assert all(digest(raw) == expected[name] for name, raw in captured.items())

    def data(name: str) -> Any:
        return json.loads(captured[name])

    checked_members = 0
    for binding in manifest['archive_members']:
        with zipfile.ZipFile(ROOT / binding['archive']) as z:
            for member, name in binding['members'].items():
                assert z.read(member) == captured[name], (binding['archive'], member, name)
                checked_members += 1

    physical = data('joined_phi_u1_full_physical_source_local_ground_parent_consumer_20261011_receipt.json')
    memory = data('local_so5_ground_electric_memory_consumer_20261011_receipt.json')
    moments = data('local_so5_ground_current_moment_bounds_20261011_receipt.json')
    composed = data('joined_phi_u1_same_local_ground_response_moment_composition_20261011_receipt.json')
    lower = data('charged_output_lower_attempt3_20261011_receipt.json')
    image = data('charged_image_proof_control_20261011_receipt.json')
    root = data('charged_rooted_source_inverse_guard_20261011_receipt.json')
    assert physical['physical_source']['domain'] == 'D(H), not D(H^2)'
    assert physical['physical_source']['neutral_driven_not_charged_only']
    assert memory['standard_binary64_model_assumed']
    assert memory['rho_zero_is_not_spectral_ground'] and memory['local_positive_response_NOT_bulk_stiffness']
    assert moments['selected_local_only'] and moments['not_absence_of_spectral_atoms']
    assert moments['input_sha256']['local_so5_ground_electric_memory_consumer_20261011_receipt.json'] == expected['local_so5_ground_electric_memory_consumer_20261011_receipt.json']
    intervals = [[F(x) for x in pair] for pair in moments['moment_intervals_exact']]
    inverse = [F(x) for x in memory['source_resolvent_interval_exact']]
    cauchy = intervals[0][0] ** 2 / intervals[1][1]
    response = [max(inverse[0], cauchy), inverse[1]]
    curvature = [F(128) - 32768 * response[1], F(128) - 32768 * response[0]]
    assert curvature == [F(x) for x in composed['incoming_electric_curvature_interval_exact']]
    assert 0 < curvature[0] <= curvature[1]
    variance = intervals[2][0] / intervals[0][1] - (intervals[1][1] / intervals[0][0]) ** 2
    assert variance == F(moments['positive_energy_variance_lower_exact']) > 0

    independent_ldl = []
    for branch in lower['branches']:
        matrix = data(branch['exact_matrix_path'])
        assert expected[branch['exact_matrix_path']] == branch['exact_matrix_sha256']
        pivots = exact_positive_ldl(matrix['lower_integer'])
        assert branch['lambda1_lower_certified'] and len(pivots) == 18
        independent_ldl.append({'branch': branch['branch'], 'dimension': len(pivots),
                                'all_exact_pivots_positive': True, 'exact_reconstruction': True})
    assert lower['both_u_signs_exact_unitary_equivalent']
    E0upper = F(physical['restricted_local_ground']['energy_interval_exact'][1])
    floors = {b['branch']: F(b['floor_exact']) for b in lower['branches']}
    gamma = 64 + floors['active'] + floors['aux'] - 2 * E0upper
    assert gamma == F(lower['induced_two_cell_gap_lower_exact']) == F(root['one_external_edge_two_cell_gap_lower_exact']) > 0
    assert image['source_and_inputs_unchanged']
    assert image['crosscopy_number_label_control']['actual_source_restricted_Gram_additivity']
    q = sum(F(row['allocation_exact']) * F(row['inverse_source_norm_square_upper_exact']) for row in root['paths'])
    assert q == F(root['candidate_root_inverse_square_sum_exact']) > 1
    assert not root['candidate_guard_less_than_one']
    assert root['candidate_guard_is_not_an_allorder_theorem']
    if manifest.get('charged_independent_review'):
        review = data(manifest['charged_independent_review'])
        assert review['status'] == 'VALID_SCOPED_LOWER_AND_IMAGE_CONDITIONAL_APPLICATION'
        assert all(expected[name] == pin for name, pin in review['revision_bound_input_sha256'].items())
        assert F(review['root_guard']['gamma_exact']) == gamma
        assert F(review['root_guard']['q_exact']) == q
        assert not review['root_guard']['guard_less_than_one']
        for branch in review['branches']:
            assert branch['status'] == 'VALID'
            assert F(branch['lambda1_floor_exact']) == floors[branch['branch']]
            assert expected[branch['proof_path']] == branch['proof_sha256']
            proof = data(branch['proof_path'])
            pivots = exact_positive_ldl(proof['independent_lower_integer'])
            independent_ldl.append({'branch': branch['branch'], 'independent_reviewer_matrix': True,
                                    'dimension': len(pivots), 'all_exact_pivots_positive': True,
                                    'exact_reconstruction': True})
    recovery = manifest['image_recovery_archive']
    with zipfile.ZipFile(ROOT / recovery) as z:
        reconstructed = json.loads(z.read('recovery_manifest.json'))
        assert reconstructed['all_original_preexecution_source_input_pins_match']
        assert reconstructed['stream_provenance']['missing'] == ['Original raw stdout pipe bytes', 'Original raw stderr pipe bytes']
        for name, pin in reconstructed['member_sha256'].items():
            assert digest(z.read(name)) == pin
            checked_members += 1

    result = {'kind': 'BOUND_FULL_SOURCE_LOCAL_GROUND_MEMORY_CHARGED_COMPOSITION',
              'task_governor': manifest['task_governor'], 'accepted_science_before_packet': manifest['accepted_science'],
              'physical_source': physical['physical_source'], 'selected_local_ground': physical['restricted_local_ground'],
              'local_curvature_interval_exact': [str(x) for x in curvature],
              'local_curvature_interval_float': [float(x) for x in curvature],
              'moment_intervals_float': [[float(x) for x in pair] for pair in intervals],
              'positive_variance_lower_float': float(variance), 'positive_variance_not_no_atoms': True,
              'charged_containing_space_floor': {k: str(x) for k, x in floors.items()},
              'charged_two_cell_gamma_exact': str(gamma), 'charged_two_cell_gamma_float': float(gamma),
              'independent_exact_ldl': independent_ldl,
              'root_candidate_upper_exact': str(q), 'root_candidate_upper_float': float(q),
              'root_candidate_failed': True, 'firstshell_not_allorder_criterion': True,
              'charged_review_stage': manifest['charged_review_stage'],
              'image_byte_archive_reconstructed_after_execution': True, 'image_raw_pipe_bytes_missing': True,
              'verified_archive_members': checked_members, 'input_sha256': expected,
              'own_source_sha256': own, 'manifest_sha256': digest(manifest_raw),
              'inputs_unchanged': all((ROOT / name).read_bytes() == raw for name, raw in captured.items()),
              'source_unchanged': digest(Path(__file__).read_bytes()) == own,
              'fullPhi_stationary_ground': False, 'bulk_stiffness': False, 'N_uniform_phase': False,
              'IRatom': False, 'sharp_photon': False, 'commoncone': False, 'originalSOURCE': False,
              'campaign_complete': False, 'elapsed_seconds': time.monotonic() - start}
    assert result['inputs_unchanged'] and result['source_unchanged']
    (ROOT / args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['input_sha256', 'local_curvature_interval_exact', 'root_candidate_upper_exact']}, indent=2))


if __name__ == '__main__':
    main()

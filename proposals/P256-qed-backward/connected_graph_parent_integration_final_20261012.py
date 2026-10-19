"""Revision-bound connected-response/native-loop graph join, not ground IR closure."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha(name: str) -> str:
    with (ROOT / name).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text())


def at(value: Any, path: list[str]) -> Any:
    for key in path:
        value = value[key]
    return value


def verified_epoch(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    for key in ('capsule', 'pre', 'executed', 'receipt', 'exit'):
        assert sha(spec[key]) == spec[key + '_sha256'], (key, spec[key])
    capsule = load(spec['capsule'])
    assert at(capsule, spec['stability_path']) is True
    for pair in spec['equal_hash_paths']:
        assert at(capsule, pair[0]) == at(capsule, pair[1])
    with zipfile.ZipFile(ROOT / spec['pre']) as pre, zipfile.ZipFile(ROOT / spec['executed']) as exe:
        names = pre.namelist()
        assert len(names) == len(set(names))
        assert set(names) == set(spec['pre_member_sha256']) == set(spec['member_map'])
        assert len(set(spec['member_map'].values())) == len(names)
        for name in names:
            raw = pre.read(name)
            assert sha_bytes(raw) == spec['pre_member_sha256'][name], name
            assert raw == exe.read(spec['member_map'][name]), name
        for name, expected in spec['capsule_declared_member_sha256'].items():
            assert name in names and sha_bytes(pre.read(name)) == expected
        for name, expected in spec['capsule_declared_output_sha256'].items():
            assert sha_bytes(exe.read(name)) == expected, name
        raw_receipt = exe.read(spec['receipt_member'])
        raw_exit = exe.read(spec['exit_member'])
        assert raw_receipt == (ROOT / spec['receipt']).read_bytes()
        assert raw_exit == (ROOT / spec['exit']).read_bytes()
        run = json.loads(raw_exit)
        assert run['natural_exit'] is True and run['returncode'] == 0
        payload = json.loads(raw_receipt)
    for name, expected in spec['captured_live_source_sha256'].items():
        assert sha(name) == expected, name
    return payload, {'members_compared': len(names), 'natural_exit': True,
                     'returncode': 0, 'elapsed_seconds': run['elapsed_seconds'],
                     'pre_to_executed_exact': True, 'live_source_hashes_unchanged': True,
                     'original_preexecution_not_backfilled': True}


def science_join(payloads: dict[str, dict[str, Any]], request: dict[str, Any]) -> dict[str, Any]:
    old, corrected = payloads['connected_original'], payloads['connected_corrected']
    assert old['attempt2']['assignment_total'] == corrected['attempt2']['assignment_total'] == 64**4
    assert corrected['attempt2']['mixed_matrix_element'] == -6
    assert corrected['attempt2']['samecopy_wrong_matrix_element'] == 0
    old_rows = old['attempt2']['physical1_cutoff_connected_bounds']
    new_rows = corrected['attempt2']['physical1_cutoff_connected_bounds']
    assert len(old_rows) == len(new_rows) == 4
    for old_row, new_row in zip(old_rows, new_rows):
        assert new_row['N_label'] == old_row['N_label']
        assert Fraction(new_row['M1_physical1_cumulant_upper']) == 24*Fraction(old_row['M1_physical1_cumulant_upper'])
        assert new_row['safe_lag_coefficient'] == 624 and new_row['literal_sum_all24'] is True
    # All numerical geometry/CAR/matrix outputs are unchanged except the paid lag budget.
    for field in ('native_geometry', 'attempt1', 'normalized_Ward_method_controls', 'full_quantum_matrix_order_control'):
        assert corrected[field] == old[field], field
    native = corrected['native_Ward_specialization']
    assert native['source_wholeP0_zero'] is True and native['native_Z'] == 0
    assert native['generic_V_F_stationary_blocks_not_deleted'] is True
    control = payloads['parent_controls']
    assert Fraction(control['lag']['counter_value']) == Fraction(-32768, 867)
    assert control['lag']['summed_26_bound_rejected'] is True
    assert control['joint_moments']['accepted_E4_finite_upper'] == '1/4'
    assert control['joint_moments']['Phi8_E2_diverges_by_term_test'] is True
    assert control['virial']['parent_radial10_upper_inference_withdrawn'] is True
    assert control['fractional']['quadratic_cutoff_error_required'] is True
    graph = payloads['loop_original']
    expansion = graph['allcopy_expansion']
    Csum = Fraction(expansion['Csum'])
    b = 24*(4*Csum)**4
    assert b == Fraction(expansion['b'])
    assert expansion['all64_fourleg_assignments_actually_streamed'] == 64**4
    assert expansion['all24_temporal_orders'] == 24
    assert expansion['total_distributive_words'] == 24*64**4
    budgets = graph['complete_graph_budgets']
    assert budgets['support_sites_upper'] == 52 and budgets['centre_radius_upper'] == 4
    assert budgets['tau_lower'] == '1/2' and budgets['graph_growth_M_exponent'] == 4
    assert budgets['source_tail_M_exponent'] == '-1/4'
    assert budgets['fractional_s_range'] == '0<s<1/17'
    # Read frozen mathematical reviews; these claims remain analytical, not native numeric trajectories.
    reviews = {role: load(name) for role, name in request['proof_reviews'].items()}
    assert reviews['cumulant']['verdict'].startswith('ORIGINAL_SUMMED_LAG_BOUND_INVALID')
    assert reviews['ward']['formal_verdict']['native_stationary_electric_debt'].startswith('INVALID_AT_DECLARED')
    assert reviews['primitive']['verdict'].startswith('VALID_CONDITIONAL_COMPLETE_COMMUTATOR')
    assert reviews['fractional']['overall_verdict'].startswith('VALID_PAID_FRACTIONAL_SOURCE')
    corrections = {name: load(name) for name in request['corrections']}
    energy = payloads['parent_energy_band']
    assert energy['normalization']['safe_complex_upper'] == '6144'
    assert all(row['certifies_one_over_16'] is False and row['actual_residual_lower'] is False
               for row in energy['normalization']['volume_removed_not_IR_closure'])
    return {'Csum_exact': str(Csum), 'b_exact': str(b),
            'sum24_lag_coefficient': 624, 'finite_cube_coefficient_unchanged': 26,
            'native_closed_E_Z_zero_not_generic_V_F_zero': True,
            'whole_operator_cutoff_graph_growth': 'M^4', 'uncut_fractional_s': '0<s<1/17',
            'uncut_Q_L_or_D_L_paid': False, 'weighted_kernel_quadratic_error_retained': True,
            'strict_joint_moment_E4_corrected_not_native_ground_counterexample': True,
            'finite_fixed_epsilon_ground_form_virial_not_radial10_upper': True,
            'magnetic_energy_l2_upper': 6144, 'TOTAL_l2_or_principal_r1_not_paid': True,
            'available_upper_estimates_not_actual_residual_lower': True,
            'signed_ground_current_seam_and_supported_Gram_lower_unpaid': True,
            'frozen_review_files': request['proof_reviews'], 'correction_files': list(corrections),
            'native_actual_ground_fullH_evolution_or_IR_measured': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    before = {name: sha(name) for name in request['pins']}
    assert before == request['pins']
    payloads: dict[str, dict[str, Any]] = {}
    verified: dict[str, dict[str, Any]] = {}
    for role, spec in request['epochs'].items():
        payloads[role], verified[role] = verified_epoch(spec)
    transfer = load(request['typed_transfer'])
    for row in transfer['transfers']:
        old = payloads[row['role']]
        new = payloads[row['typed_role']]
        ignored = set(row['ignored_METADATA_fields_only'])
        assert {k: v for k, v in old.items() if k not in ignored} == {
            k: v for k, v in new.items() if k not in ignored
        }, row['role']
    science = science_join(payloads, request)
    science['three_new_typed_payloads_exact_original_math'] = True
    after = {name: sha(name) for name in request['pins']}
    assert after == before
    result = {'kind': 'IMMUTABLE_CONNECTED_MAGNETIC_NATIVE_PRIMITIVE_FRACTIONAL_GRAPH_PARENT_JOIN_NOT_PHASE',
              'science': science, 'verified_epochs': verified,
              'members_compared': sum(row['members_compared'] for row in verified.values()),
              'original_targets_and_reviews_frozen_not_rebound': True,
              'fullphysical_ground_phase_stiffness_photon_admitted': False,
              'unconditional_machine_attestation': False, 'original_SOURCE_adopted': False,
              'campaign_complete': False, 'source_input_sha256_before': before,
              'source_input_sha256_after': after}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'epochs': len(verified), 'members': result['members_compared'],
                      'literal_sum_lag': 624, 'phase_or_atom': False}))


if __name__ == '__main__':
    main()

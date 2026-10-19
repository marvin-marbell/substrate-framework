"""Consume frozen K-energy and full physical-source evidence without old producers."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, IO
import zipfile

ROOT = Path(__file__).resolve().parent


def stream_hash(stream: IO[bytes]) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1048576), b''):
        h.update(block)
    return h.hexdigest()


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return stream_hash(stream)


def load(name: str) -> Any:
    return json.loads((ROOT / name).read_text())


def verify_epoch(spec: dict[str, Any]) -> dict[str, Any]:
    metadata = load(spec['metadata'])
    exit_record = metadata.get('execution', metadata)
    assert exit_record['returncode'] == 0 and exit_record['natural_exit']
    if spec.get('exit'):
        exit_record = load(spec['exit'])
        assert exit_record['returncode'] == 0 and exit_record['natural_exit']
    compared = 0
    if spec.get('pre'):
        with zipfile.ZipFile(ROOT / spec['pre']) as before, zipfile.ZipFile(ROOT / spec['exe']) as after:
            after_names = after.namelist()
            if spec['pre'] in after_names:
                # This producer embeds its complete preexecution ZIP, not flattened members.
                with after.open(spec['pre']) as archived_before:
                    assert stream_hash(archived_before) == digest(ROOT / spec['pre'])
                compared = len(before.namelist())
            else:
                for name in before.namelist():
                    target = name if name in after_names else spec['prefix'] + '_request.json'
                    assert name == target or name == 'request.json', (name, target)
                    assert target in after_names
                    with before.open(name) as a, after.open(target) as b:
                        assert stream_hash(a) == stream_hash(b), (name, target)
                    compared += 1
            candidates = [n for n in after_names if n == spec['receipt'] or n.endswith('/' + spec['receipt'])]
            assert len(candidates) == 1
            assert hashlib.sha256(after.read(candidates[0])).hexdigest() == digest(ROOT / spec['receipt'])
    elif spec.get('archive'):
        with zipfile.ZipFile(ROOT / spec['archive']) as archive:
            for name, pin in metadata['archive_member_sha256'].items():
                with archive.open(name) as member:
                    assert stream_hash(member) == pin
                compared += 1
    if spec.get('embedded_result'):
        assert metadata['result'] == load(spec['receipt'])
    return {'metadata': spec['metadata'], 'members_compared': compared,
            'elapsed_seconds': exit_record.get('elapsed_seconds'),
            'input_postrun_audit': spec['input_postrun_audit'], 'natural_exit': True}


def mathematical_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: mathematical_payload(item) for key, item in value.items()
                if key not in {'source_sha256', 'input_sha256', 'elapsed_seconds', 'epoch', 'governing_plan'}}
    if isinstance(value, list):
        return [mathematical_payload(item) for item in value]
    return value


def ceil_log2(n: int) -> int:
    return (n - 1).bit_length()


def main() -> None:
    sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    assert all(digest(ROOT / name) == pin for name, pin in request['pins'].items())
    epochs = [verify_epoch(spec) for spec in request['epochs']]
    assert digest(ROOT / 'canonical_ground_review_proof_20261011.json') == 'ea69bfb67bfffc8eb5a9b5a4efbaf0b54edbb9a8ba06839b49c03faa3e60fd65'
    refined = load('k_resolved_refined_actual1_20261011_receipt.json')
    centered = load('k_resolved_centered_actual1_20261011_receipt.json')
    selection = load('k_joint_review_actual1_20261011_receipt.json')['exact_selections']
    prices = []
    for axis, old_axis in zip(refined['axes'], centered['axes'], strict=True):
        assert axis['axis'] == old_axis['axis']
        for row, old in zip(axis['energies'], old_axis['energies'], strict=True):
            energy = row['E_exact']
            assert energy == old['E_exact']
            assert F(row['every_Q_upper_exact']) == F(row['centered_G_Q_upper_exact']) + F(row['joint_and_negative_scalar_Q_exact'])
            assert row['every_Q_lower_exact'] == old['every_Q_lower_exact']
            assert F(row['every_Q_lower_exact']) == min(sum((F(v) for v in row['complete_lower_components'].values()), F(0)), F(old['every_Q_lower_exact']))
            for weight, upper, lower, absolute in [('Q', 'every_Q_upper_exact', 'every_Q_lower_exact', 'every_Q_absolute_exact'), ('S', 'allcentral_S_upper_exact', 'allcentral_S_lower_exact', 'allcentral_S_absolute_exact')]:
                assert F(row[absolute]) == max(F(row[upper]), F(row[lower]))
                selected = next(s for s in selection if s['family'] == 'refined' and s['axis'] == axis['axis'] and s['E'] == energy and s['weight'] == weight)
                assert F(selected['selected_slope']) == 88
            if F(energy) == 0:
                assert F(row['every_Q_upper_exact']) > 1 and F(row['every_Q_absolute_exact']) > 1
                prices.append({'axis': axis['axis'], 'Q_upper_exact': row['every_Q_upper_exact'], 'Q_ABS_exact': row['every_Q_absolute_exact'], 'S_upper_exact': row['allcentral_S_upper_exact'], 'S_ABS_exact': row['allcentral_S_absolute_exact'], 'Q_upper_display': float(F(row['every_Q_upper_exact'])), 'Q_ABS_display': float(F(row['every_Q_absolute_exact']))})
    proof = load('physical_ground_lift_actual3_20261011_proof.json')
    controls = load('physical_ground_lift_actual3_20261011_receipt.json')
    assert proof['controls'] == controls
    withdrawal = load('lift_gauss_review_F1_withdrawal_20261011.json')
    assert withdrawal['exact_pinned_evidence']['g_rho_moments_1_2_4'] == controls['g_rho_moments_1_2_4']
    assert controls['g_rho_moments_1_2_4'][1] > .64
    assert digest(ROOT / withdrawal['original_formal']) == withdrawal['original_formal_sha256']
    corrected = load('lift_gauss_review_formal_corrected_20261011.json')
    assert corrected['findings'] == []
    repair = load('physical_ground_lift_parent_action_derivative_repair_20261011.json')
    with zipfile.ZipFile(ROOT / 'physical_ground_lift_actual3_20261011_preexecution.zip') as archive:
        with archive.open(repair['pinned_source']['path']) as original_source:
            assert stream_hash(original_source) == repair['pinned_source']['sha256']
    transfer = load('k_lift_parent_typed_payload_transfer_20261011.json')
    for pair in transfer['transfers']:
        old = load(pair.get('original', pair.get('original_capsule')))
        if 'original_capsule' in pair:
            old = old['result']
        assert mathematical_payload(old) == mathematical_payload(load(pair['new']))
    R = F(1, 2**24)
    s1 = 2 * R * 2**40 / R**2
    s2 = 2 * (F(2**80) + R * 2**90) / R**2
    assert s1 == 2**65 and s2 < 2**130
    b1 = 4 * s1
    b2 = 12 * s1**2 + 4 * s2
    assert b1 == 2**67 and b2 < 2**134
    joined = b2 + 2 * b1 * 2**40 + 2**90
    assert joined < 2**135 < 2**240
    V = 64
    ratio = 16384 * V
    terms = {'Phi_complete_scalar_frame_Berry': -4096 + ceil_log2(4 * 60 * V) + ratio + 240,
             'Haar_complete_transport_scalar_cross': -4096 + ceil_log2(45 * V) + ratio + 240,
             'fullE64_longitudinal_and_transverse': 6 + ceil_log2(3 * V) + ratio + 240,
             'neutral_all_canonical2V_onsite_paths': 8192 + 30 + ceil_log2(V),
             'charged_actual1_aux63_all_onsite_one_two_paths': 30 + ceil_log2(V),
             'Jordan_full_support': 8192 + 30 + ceil_log2(V),
             'alignment_full_support': 10240 + 30 + ceil_log2(V),
             'colour_plaquette_full_Haar_support': 10240 + 30 + ceil_log2(V)}
    assert terms == proof['N4_complete_action_upper']['term_log2_upper']
    assert max(terms.values()) + ceil_log2(len(terms)) == 1048833
    eta = F(274877906944, 2204801412260314559041)
    gamma = F(46955312929, 34359738368)
    assert eta == (F(1, 65536) / gamma)**2
    assert F(135, 2048) < F(1, 8) and 72**15 < 2**93
    assert (360 * V + 10094 * (V // 2)) // 2 < ratio
    result = {'kind': 'ACTUAL_K_WHOLE_FORM_PHYSICAL_SOURCE_INTEGRATION_NOT_GROUND_OR_PHASE',
              'verified_epochs': epochs, 'E0_prices': prices,
              'wholecanonical_antecedent_sha256': digest(ROOT / 'canonical_ground_review_proof_20261011.json'),
              'physical1_sufficient_upper_and_absolute_guards_failed_not_no_go': True,
              'Gauss_F1_withdrawn_review_indexing_not_source_bug': True,
              'typed_payload_transfers_exact_equal_original': len(transfer['transfers']),
              'frozen_action_repair_source_resolved_original_archive_not_live_typed_source': True,
              'complete_join_second_derivative_upper_exponent': 135,
              'repaired_weaker_derivative_exponent': 240,
              'complete_H_norm_upper_exponent': 1048833,
              'not_variance_not_stationarity_not_N_uniform': True,
              'binary_x_distinct_from_exact_g_graph_sqrt_infidelity': str(eta),
              'conditional_ambient_realization_for_compact_source_nonstationarity': True,
              'fullphysical_ground_phase_stiffness_photon_admitted': False,
              'historical_formation_machine_and_ground_premises_retained': True,
              'original_SOURCE_adopted': False, 'campaign_complete': False}
    assert all(digest(ROOT / name) == pin for name, pin in request['pins'].items())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'epochs': len(epochs), 'members': sum(e['members_compared'] for e in epochs), 'Q_upper': [p['Q_upper_display'] for p in prices], 'physical1_or_ground': False}))


if __name__ == '__main__':
    main()

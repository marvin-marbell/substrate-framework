"""Revision-bound higher radial source and contiguous Wilson/current mechanism join."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import zipfile

import sympy as sp

from jordan_ground_parent_integration_20261012 import verified_epoch

ROOT = Path(__file__).resolve().parent


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text())


def sha(name: str) -> str:
    with (ROOT / name).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_selected_manifest(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    # This original executor writes a PRE manifest and a different EXIT manifest.
    # All selected members are byte-identical; both metadata objects are checked,
    # rather than pretending that the two manifest byte strings are identical.
    for key in ('pre', 'executed', 'receipt', 'capsule', 'exit'):
        assert sha(spec[key]) == spec[key + '_sha256']
    capsule = load(spec['capsule'])
    with zipfile.ZipFile(ROOT / spec['pre']) as pre, zipfile.ZipFile(ROOT / spec['executed']) as exe:
        pre_manifest = json.loads(pre.read('manifest.json'))
        post_manifest = json.loads(exe.read('manifest.json'))
        selected: dict[str, str] = pre_manifest['selected_hashes']
        assert selected == capsule['selected_hashes']
        expected = {'selected/' + name.lstrip('/') for name in selected}
        assert set(pre.namelist()) == expected | {'manifest.json'}
        assert len(pre.namelist()) == len(set(pre.namelist()))
        assert len(exe.namelist()) == len(set(exe.namelist()))
        for name, pinned in selected.items():
            member = 'selected/' + name.lstrip('/')
            raw = pre.read(member)
            assert hashlib.sha256(raw).hexdigest() == pinned
            assert raw == exe.read(member)
            assert sha(name) == pinned
        exit_raw = exe.read('selected/' + spec['exit'])
        assert exit_raw == (ROOT / spec['exit']).read_bytes()
        runtime = json.loads(exit_raw)
        assert runtime == post_manifest
        assert runtime['command'] == pre_manifest['command'] == capsule['command']
        for flag in ('natural_exit', 'selected_bytes_unchanged', 'python_executable_unchanged'):
            assert runtime[flag] is True
        assert runtime['returncode'] == 0
        assert runtime['preexecution_sha256'] == spec['pre_sha256']
        raw = exe.read('selected/' + spec['receipt'])
        assert raw == (ROOT / spec['receipt']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == runtime['result_sha256']
        payload = json.loads(raw)
    return payload, {'members_compared': len(selected), 'metadata_objects_verified': 2,
                     'pre_manifest_is_not_exit_manifest': True, 'natural_exit': True,
                     'returncode': 0, 'elapsed_seconds': runtime['elapsed_seconds']}


def native_source_price(request: dict[str, Any], loop: dict[str, Any], wilson: dict[str, Any]) -> dict[str, Any]:
    config = load(request['configuration'])['input_descriptor']['original_micro_config_UNCHANGED']
    assert config['auxiliary_count'] == 63 and config['a'] == 1 / 64
    c0 = abs(F(-float(config['light_r']) / float(config['a'])))
    ci = abs(F(-float(config['light_v']) / (2 * float(config['a']))))
    total = (1 + 63) * (c0 + ci)
    b = 24 * (4 * total)**4
    interface = loop['interface']
    assert str(total) == interface['Csum_1_plus_63']
    assert str(b) == interface['bare_b']
    assert interface['all64_fourleg_assignment_count'] == 64**4
    patterns = wilson['mixed_carrier']['active_aux_profile_pattern_weights']
    assert sum(patterns.values()) == 64**4
    assert len(patterns) == 16
    assert 64**4 != 64 and total**4 != 64 * (c0 + ci)**4
    w = F(request['W_native_all_literal_roles'])
    price = 2048 * w
    assert F(wilson['first_gate']['complete_carrier_gain_interval'][1]) == price
    assert wilson['first_gate']['complete_HE_HC_increment'] == 320
    assert wilson['first_gate']['ground_variational_intersection'] == [str(-price), '320']
    assert not wilson['first_gate']['state_independent_positive_K_paid']
    assert all(row['certifies_target'] is False for row in wilson['second_gate']['paid_envelope_rows'])
    return {'Csum_1_plus_63_exact': str(total), 'bare_full64_fourleg_b_exact': str(b),
            'external_assignments': 64**4, 'wrong_singlecopy_assignments': 64,
            'source_norm_only': 'S=2b sqrt(C85), delta(M)=S M^(-1/4)',
            'whole_limiting_P0_eta_error_only': '2S delta/eta after fixing SAME L',
            'full_H_image_energy_or_graph': False,
            'signed_gain_interval_physical1': [str(-price), '320'],
            'positive_carrier_coercivity_paid': False,
            'normalized_current_seam_gate_certified': False}


def moment_and_scaling_controls() -> dict[str, Any]:
    # Degree-four source norm can be completed from the NEW radial 8.5 moment.
    # A finite first spectral moment would be a different, unpaid assertion.
    rows = []
    for cutoff, fourth_root in ((16, 2), (256, 4), (4096, 8)):
        delta_over_s = F(1, fourth_root)
        assert delta_over_s**4 == F(1, cutoff)
        rows.append({'M': cutoff, 'delta_over_S': str(delta_over_s),
                     'bounded_middle_error_over_cS2': str(2 * delta_over_s)})
    q, epsilon = sp.symbols('q epsilon', positive=True)
    residual2 = epsilon**2 * q**5
    chebyshev_over_q = sp.simplify(residual2 / q**4 / q)
    assert chebyshev_over_q == epsilon**2
    # Missing either source normalization or a parallel seam is not harmless.
    k = sp.Matrix([[1, sp.I], [-sp.I, 3]])
    source = sp.Matrix([1, 0])
    parallel = (source.conjugate().T * k * source)[0]
    outside = k * source - parallel * source
    assert parallel == 1 and outside == sp.Matrix([0, -sp.I])
    wrong_entrywise_real = sp.Matrix([[1, 0], [0, 3]])
    assert (wrong_entrywise_real - k).det() == -1
    return {'cutoff_source_norm_rows': rows,
            'conditional_velocity_Chebyshev_mass_over_q': str(chebyshev_over_q),
            'complex_parallel_and_orthogonal_force_control_method_only': True,
            'wrong_deleted_imaginary_force_rejected': True,
            'positive_source_Gram_or_actual_force_computed': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request = json.loads(args.request.read_text())
    before = {name: sha(name) for name in request['pins']}
    assert before == request['pins']
    outputs: dict[str, dict[str, Any]] = {}
    runtime: dict[str, dict[str, Any]] = {}
    for label, spec in request['epochs'].items():
        if spec['archive_layout'] == 'selected_pre_and_exit_manifest':
            payload, verified = verify_selected_manifest(spec)
        else:
            assert spec['archive_layout'] == 'byte_identical_pre_members'
            payload, verified = verified_epoch(spec)
        capsule = load(spec['capsule'])
        if 'external_contract_original_request' in spec:
            original_request = load(spec['external_contract_original_request'])
            assert spec['live_source_path_map'] == {
                'wilson_ground_kernel_governing_contract_20261012.json': original_request['contract_path']}
            assert sha(original_request['contract_path']) == original_request['contract_sha256']
            verified['external_source_path_map_from_original_request'] = spec['live_source_path_map']
        if 'preexecution_input_hashes' in capsule:
            assert capsule['preexecution_input_hashes'] == capsule['postexecution_input_hashes']
            for name, pinned in capsule['preexecution_input_hashes'].items():
                # Some original archive identifiers name externally located
                # sources; the pinned original request supplies this mapping.
                actual_path = spec.get('live_source_path_map', {}).get(name, name)
                assert sha(actual_path) == pinned
        outputs[label], runtime[label] = payload, verified
    loop, wilson, contact = outputs['loop'], outputs['wilson'], outputs['parent_contact']
    assert loop['scientific_admission'] is False and wilson['scientific_admission'] is False
    assert loop['radial_jets']['C2'] == '(2+744*kphi)*C1^2'
    assert loop['allcopy_CAR_adverse']['two_copy_all16_assignments_offdiagonal'] == -6
    assert loop['allcopy_CAR_adverse']['same_copy_sum_offdiagonal'] == 0
    assert contact['real_contact_complex_Gram_boundary']['full_complex_moment'] == '3'
    assert contact['real_contact_complex_Gram_boundary']['wrong_deleted_imaginary_moment'] == '5'
    assert len(contact['geometry']) == 15
    transfers = load(request['typed_transfer_record'])['transfers']
    for transfer in transfers:
        original = load(transfer['original_result'])
        successor = load(transfer['prefix'] + '_receipt.json')
        if transfer['label'] == 'higher':
            # These lists describe a set of archived inputs; only iteration
            # order changed when the new request serialized sorted keys.
            for name, fields in original['producer_archives'].items():
                if 'selected_members_bound' in fields:
                    other = successor['producer_archives'][name]
                    assert sorted(fields['selected_members_bound']) == sorted(other['selected_members_bound'])
                    fields['selected_members_bound'] = sorted(fields['selected_members_bound'])
                    other['selected_members_bound'] = sorted(other['selected_members_bound'])
        omitted = set(transfer['omitted_metadata'])
        assert {k: v for k, v in original.items() if k not in omitted} == {
            k: v for k, v in successor.items() if k not in omitted}
    results = {'kind': 'JOINED_NATIVE_HIGHER_GRAPH_SOURCE_AND_CONTIGUOUS_CARRIER_CURRENT_MECHANISM_NOT_PHASE',
               'verified_epochs': runtime,
               'pre_to_executed_selected_members': sum(row['members_compared'] for row in runtime.values()),
               'explicit_transformed_metadata_objects': sum(row.get('metadata_objects_verified', 0) for row in runtime.values()),
               'native_source_and_carrier_prices': native_source_price(request, loop, wilson),
               'parent_exact_scaling_controls': moment_and_scaling_controls(),
               'parent_contact_filter_countercontrols': {'native_rawgeometry_cases': 15,
                   'O1_total_UV_contact_not_necessary_photon_obstruction': True,
                   'inverse_upper_and_filtered_IR_weight_do_not_force_atom': True,
                   'SymREAL_contact_not_full_complex_Gram': True},
               'review_proof_inputs': request['review_proofs'],
               'higher_graph_and_source_claims_pending_review': request['claims_pending_review'],
               'exact_type_only_mathematical_transfers': len(transfers),
               'metadata_order_qualification_explicit': True,
               'source_only_not_fullH_graph': True, 'finiteP0_equals_limitingP0': False,
               'native_fullH_evolution_or_ground_computed': False,
               'positive_IR_Gram_stiffness_phase_atom_admitted': False,
               'original_SOURCE_adopted': False, 'campaign_complete': False,
               'source_input_sha256_before': before,
               'source_input_sha256_after': {name: sha(name) for name in request['pins']},
               'elapsed_seconds': time.monotonic() - started}
    assert results['source_input_sha256_before'] == results['source_input_sha256_after']
    with args.output.open('x') as stream:
        json.dump(results, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps({'kind': results['kind'], 'verified_epochs': runtime,
                      'members_compared': results['pre_to_executed_selected_members']}))


if __name__ == '__main__':
    main()

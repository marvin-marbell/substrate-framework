"""Immutable canonical-path/carrier integration and exact ground-response method controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_epoch(spec: dict[str, Any]) -> dict[str, Any]:
    prefix = spec['epoch']
    capsule = json.loads((ROOT / spec['capsule']).read_text())
    protocol = spec['protocol']
    if protocol == 'canonical_path':
        files = capsule['artifacts']
        pre_name = prefix + '_preexecution.zip'
        exe_name = prefix + '_executed.zip'
        pins = capsule['source_input_pins']
        output_hashes = {name: files[name] for name in spec['outputs']}
        member_sources = {name: name for name in pins}
        member_sources[prefix + '_request.json'] = prefix + '_request.json'
    elif protocol == 'carrier':
        pre_name = capsule['preexecution']['path']
        exe_name = capsule['executed']['path']
        files = {capsule[key]['path']: capsule[key]['sha256']
                 for key in ('preexecution', 'executed', 'request')}
        files.update(capsule['outputs'])
        pins = capsule['source_input_pins']
        output_hashes = capsule['outputs']
        member_sources = {member: source for member, source in capsule['archive_member_sources'].items()
                          if member not in output_hashes}
    elif protocol == 'parent':
        pre_name = capsule['preexecution']['path']
        exe_name = capsule['executed']['path']
        files = {capsule[key]['path']: capsule[key]['sha256']
                 for key in ('preexecution', 'executed')}
        files.update(capsule['outputs'])
        pins = capsule['source_input_sha256_before']
        assert pins == capsule['source_input_sha256_after']
        assert capsule['sources_inputs_unchanged']
        output_hashes = capsule['outputs']
        member_sources = {member: name for name, member in capsule['archive_member_map'].items()}
    elif protocol == 'review':
        pre_name = spec['preexecution']
        exe_name = spec['executed']
        files = spec['artifact_hashes']
        pins = capsule[spec['pin_key']]
        output_hashes = spec['output_hashes']
        member_sources = spec['member_sources']
    else:
        raise ValueError(protocol)
    for name, expected in files.items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    for name, expected in pins.items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    output_sources = spec.get('output_member_sources', {name: name for name in output_hashes})
    manifest = spec.get('append_output_manifest')
    extracted_inputs = spec.get('archive_member_inputs', {})
    nested_pre = spec.get('nested_preexecution', False)
    with zipfile.ZipFile(ROOT / pre_name) as before, zipfile.ZipFile(ROOT / exe_name) as after:
        pre_members = before.namelist()
        assert len(pre_members) == len(set(pre_members))
        assert set(pre_members) == set(member_sources) | set(extracted_inputs) | ({manifest} if manifest else set())
        expected_inputs = {pre_name} if nested_pre else set(pre_members)
        assert set(after.namelist()) == expected_inputs | set(output_hashes)
        if nested_pre:
            assert after.read(pre_name) == (ROOT / pre_name).read_bytes()
        executed_inputs = before if nested_pre else after
        for member in pre_members:
            raw = before.read(member)
            if member == manifest:
                expected_manifest = dict(pins)
                request_name = prefix + '_request.json'
                expected_manifest[request_name] = sha((ROOT / request_name).read_bytes())
                assert json.loads(raw) == expected_manifest
                added = {output_sources[name]: expected for name, expected in output_hashes.items()}
                assert json.loads(executed_inputs.read(member)) == {**expected_manifest, **added}
            else:
                assert raw == executed_inputs.read(member), member
                if member in extracted_inputs:
                    selected = extracted_inputs[member]
                    with zipfile.ZipFile(ROOT / selected['archive']) as archive:
                        assert raw == archive.read(selected['member'])
                    assert sha(raw) == selected['sha256']
                else:
                    source = member_sources[member]
                    assert raw == (ROOT / source).read_bytes(), source
        for member, expected in output_hashes.items():
            assert sha(after.read(member)) == expected
            assert after.read(member) == (ROOT / output_sources[member]).read_bytes()
    exit_data = json.loads((ROOT / (prefix + '_exit.json')).read_text())
    assert exit_data[spec.get('returncode_key', 'returncode')] == 0 and exit_data['natural_exit']
    elapsed_key = spec.get('elapsed_seconds_key', 'elapsed_seconds')
    elapsed = exit_data[elapsed_key] if elapsed_key is not None else None
    interval = None if elapsed_key is not None else {key: exit_data[key] for key in ('started', 'finished')}
    assert (ROOT / (prefix + '_stderr.txt')).read_bytes() == b''
    return {'epoch': prefix, 'source_input_pins': len(pins),
            'exact_pre_members': len(pre_members) - int(manifest is not None),
            'append_only_output_manifest_verified': manifest is not None,
            'nested_preexecution_bytes_identical': nested_pre,
            'selected_archive_members_verified': len(extracted_inputs),
            'natural_exit': True, 'returncode': 0, 'elapsed_seconds': elapsed,
            'reported_wall_interval_without_monotonic_duration': interval,
            'capsule_sha256': sha((ROOT / spec['capsule']).read_bytes())}


def exact_ground_response() -> dict[str, Any]:
    # Complete four-dimensional positive Hamiltonian; not a native quantum-ground surrogate.
    hamiltonian = sp.diag(0, 2, 5, 11)
    ground = sp.Matrix([1, 0, 0, 0])
    projection = sp.diag(0, 1, 1, 1)
    eta = sp.Integer(3)
    resolvent = sp.diag(0, sp.Rational(1, 5), sp.Rational(1, 8), sp.Rational(1, 14))
    columns = sp.Matrix([[0, 0], [1, 2], [sp.I, 1], [1, -sp.I]])
    gram = columns.conjugate().T * columns
    energy = columns.conjugate().T * hamiltonian * columns
    moment = energy + eta * gram
    response = columns.conjugate().T * resolvent * columns
    lower = gram * moment.inv() * gram
    difference: Any = sp.simplify(response - lower)
    assert difference[0, 0] > 0 and difference[1, 1] > 0
    assert difference.det() == 0 and difference.rank() == 1
    assert sp.im(response[0, 1]) != 0
    # Omitting eta in the energy matrix violates the claimed resolvent lower.
    wrong_lower = gram * energy.inv() * gram
    assert sp.simplify((response - wrong_lower)[0, 0]) == -sp.Rational(99, 280)
    source = sp.Matrix([[sp.Rational(1, 7), 1, -sp.I, 1],
                        [1, 2, 1, 0], [sp.I, 1, -1, sp.I], [1, 0, -sp.I, 3]])
    assert source == source.conjugate().T
    x = projection * source * ground
    y = resolvent * x
    j = sp.Rational(1, 3)
    norm_u = (y.conjugate().T * y)[0]
    t = (x.conjugate().T * resolvent * x)[0]
    k = t - eta * norm_u
    a = (ground.conjugate().T * source * ground)[0]
    cubic = (y.conjugate().T * source * y)[0]
    unnormalized = ground + j * y
    direct = sp.simplify((unnormalized.conjugate().T * (hamiltonian - j * source)
                          * unnormalized)[0] / (unnormalized.conjugate().T * unnormalized)[0])
    numerator = j**2 * k - j * a - 2 * j**2 * t - j**3 * cubic
    formula = sp.simplify(numerator / (1 + j**2 * norm_u))
    assert direct == formula
    assert direct != numerator
    assert direct != sp.simplify((numerator + j**3 * cubic) / (1 + j**2 * norm_u))
    # Whole ground block, not rank-one mean, is the spectral response projection.
    degenerate_h = sp.diag(0, 0, 5)
    zero_source = sp.Matrix([0, 1, 0])
    whole_q = sp.diag(0, 0, 1)
    mean_q = sp.diag(0, 1, 1)
    assert whole_q * zero_source == sp.zeros(3, 1)
    assert mean_q * zero_source == zero_source
    assert degenerate_h * zero_source == sp.zeros(3, 1)
    # Nonzero complementary response branches may cancel completely.
    branch_gram: Any = sp.Matrix([[1, -1], [-1, 1]])
    assert branch_gram[0, 0] > 0 and branch_gram[1, 1] > 0
    assert (sp.ones(1, 2) * branch_gram * sp.ones(2, 1))[0] == 0
    return {'scope': 'EXACT_NONNATIVE_COMPLETE_GROUND_RESPONSE_METHOD_ONLY',
            'response': str(response), 'energy_support_lower': str(lower),
            'positive_semidefinite_rank_one_lower_difference': str(difference),
            'complex_cross_retained': True, 'wrong_omitted_eta_rejected': True,
            'normalized_complete_source_energy': str(direct), 'cubic': str(cubic),
            'wrong_deleted_denominator_rejected': True, 'wrong_deleted_cubic_rejected': True,
            'whole_ground_block_not_mean_rejected': True,
            'nonzero_signed_branches_need_not_give_response': True,
            'native_ground_computed': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    before = {name: (ROOT / name).read_bytes() for name in request['pins']}
    for name, raw in before.items():
        assert sha(raw) == request['pins'][name], name
    epochs = [verify_epoch(spec) for spec in request['epochs']]
    transfers = []
    for binding in request['typed_transfers']:
        original = json.loads((ROOT / binding['original_receipt']['path']).read_text())
        typed = json.loads((ROOT / binding['typed_receipt']['path']).read_text())
        ignored = set(binding['excluded_metadata_fields'])
        assert ignored == {'input_pins', 'source_input_pins'}
        assert {key: value for key, value in original.items() if key not in ignored} == {
            key: value for key, value in typed.items() if key not in ignored}
        transfers.append({'original_receipt': binding['original_receipt'],
                          'typed_receipt': binding['typed_receipt'],
                          'mathematical_payload_exact_equal': True})
    result = {'kind': 'CANONICAL_PATH_CARRIER_IMMUTABLE_JOIN_AND_EXACT_RESPONSE_NOT_PHASE',
              'epochs': epochs, 'pre_members_verified': sum(e['exact_pre_members'] for e in epochs),
              'typed_transfers': transfers,
              'exact_ground_response_method': exact_ground_response(),
              'independent_reviews_pending': request['independent_reviews_pending'],
              'originals_not_rerun_or_rebound': True,
              'native_fullH_ground_or_evolution_computed': False,
              'physical1_positive_carrier_or_phase_stiffness_IR_atom': False,
              'campaign_complete': False, 'input_pins': request['pins']}
    assert all((ROOT / name).read_bytes() == raw for name, raw in before.items())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Canonical path/carrier immutable join and exact response controls completed')


if __name__ == '__main__':
    main()

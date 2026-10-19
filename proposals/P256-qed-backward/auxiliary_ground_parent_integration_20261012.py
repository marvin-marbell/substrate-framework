"""Immutable auxiliary/adjoint/full-form join; no ground-sector or IR promotion."""
from __future__ import annotations

import argparse
import copy
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
    capsule = json.loads((ROOT / spec['capsule']).read_text())
    for selector in spec['stability_fields']:
        value = capsule
        for key in selector:
            value = value[key]
        assert value is True
    if spec['capsule_after_key'] is not None:
        assert capsule[spec['capsule_after_key']] == capsule[spec['capsule_pins_key']]
    for key, member in spec['pin_members'].items():
        assert capsule[spec['capsule_pins_key']][key] == spec['input_members'][member]['sha256']
    assert capsule == spec['captured_capsule']
    for path, expected in spec['artifacts'].items():
        assert sha((ROOT / path).read_bytes()) == expected, path
    with zipfile.ZipFile(ROOT / spec['preexecution']) as before, zipfile.ZipFile(ROOT / spec['executed']) as after:
        assert len(before.namelist()) == len(set(before.namelist()))
        assert len(after.namelist()) == len(set(after.namelist()))
        assert set(before.namelist()) == set(spec['input_members'])
        assert set(after.namelist()) == set(spec['input_members']) | set(spec['output_members'])
        differences = []
        for member, binding in spec['input_members'].items():
            raw = before.read(member)
            assert raw == after.read(member), member
            assert sha(raw) == binding['sha256'], member
            live = sha((ROOT / binding['path']).read_bytes())
            if live != binding['sha256']:
                differences.append({'path': binding['path'], 'captured': binding['sha256'], 'live': live})
        for member, binding in spec['output_members'].items():
            raw = after.read(member)
            assert sha(raw) == binding['sha256'], member
            assert raw == (ROOT / binding['path']).read_bytes(), member
    runtime = json.loads((ROOT / spec['exit_path']).read_text())
    assert runtime['natural_exit']
    assert runtime[spec['exit_code_key']] == spec['returncode']
    assert (ROOT / spec['stderr_path']).read_bytes() == b'' if spec['returncode'] == 0 else (ROOT / spec['stderr_path']).read_bytes() != b''
    return {'epoch': spec['epoch'], 'pre_members_verified': len(spec['input_members']),
            'returncode': spec['returncode'], 'natural_exit': True,
            'elapsed_seconds': runtime['elapsed_seconds'],
            'historical_live_differences_not_rebound': differences}


def matrix_budget_control() -> dict[str, Any]:
    # Nonnative exact implication control: two identical columns saturate p*U.
    u = sp.Matrix([[1, 1], [0, 0]])
    s = u.conjugate().T * u
    assert s == sp.ones(2)
    z = sp.Matrix([1, 1]) / sp.sqrt(2)
    assert sp.simplify((z.conjugate().T * s * z)[0]) == 2
    assert (2 * sp.eye(2) - s).det() == 0
    assert (sp.eye(2) - s).det() == -1
    # Full complex first-energy and second-energy matrices, not real contact.
    l = sp.diag(2, 5)
    columns = sp.Matrix([[1, sp.I], [0, 1]])
    t = columns.conjugate().T * l * columns
    second = columns.conjugate().T * l**2 * columns
    assert t == sp.Matrix([[2, 2 * sp.I], [-2 * sp.I, 7]])
    assert t.det() == 10 and (t - sp.eye(2)).det() == 2
    assert second == sp.Matrix([[4, 4 * sp.I], [-4 * sp.I, 29]])
    assert (32 * sp.eye(2) - second).det() == 68
    eta = sp.Rational(2, 3)
    resolvent = columns.conjugate().T * (l + eta * sp.eye(2)).inv() * columns
    lower = sp.Rational(1, 32) / (32 + eta)
    difference = resolvent - lower * sp.eye(2)
    assert difference[0, 0] > 0 and difference.det() > 0
    polarization = sp.Matrix([1, sp.I])
    assert (polarization.conjugate().T * t * polarization)[0] == 5
    assert (polarization.conjugate().T * sp.diag(2, 7) * polarization)[0] == 9
    return {'scope': 'NONNATIVE_EXACT_FULL_MATRIX_BUDGET_IMPLICATION',
            'diagonal_U_one_requires_pU_two_for_identical_sources': True,
            'same_U_matrix_from_diagonals_rejected': True,
            'full_complex_T': [[str(x) for x in row] for row in t.tolist()],
            'conditional_tau': 1, 'conditional_matrix_U': 32,
            'conditional_G_lower': str(lower),
            'imaginary_cross_deletion_rejected': True,
            'native_first_energy_lower_proved': False}


def reviewed_boundaries(request: dict[str, Any]) -> dict[str, Any]:
    active = json.loads((ROOT / request['active_receipt']).read_text())
    adjoint = json.loads((ROOT / request['adjoint_receipt']).read_text())
    energy = json.loads((ROOT / request['energy_delivery']).read_text())
    return {'active_number_review_receipt_sha256': sha((ROOT / request['active_receipt']).read_bytes()),
            'integrated_quantum_G': active['integrated_quantum_G'],
            'adjoint_review_receipt_sha256': sha((ROOT / request['adjoint_receipt']).read_bytes()),
            'physical_zero_transverse_ranks': {
                key: value['rank'] for key, value in adjoint['observations']['physical_zero_transverse']['ranks'].items()
                if key in ('End16_A', 'End16_joint_B_Bdag', 'cross16')},
            'full_complex_matrix_budget_caveat': energy['supporting_rationale_limits'][1],
            'integrated_G_reused_not_reexecuted': True,
            'physical_quotient_zero_transverse_witness_reviewed': True,
            'upper_versus_upper_does_not_select_auxiliary_ground_sector': True,
            'adjoint_a_e_kernel_exclusion_does_not_pay_originating_weight_or_wholeP0': True,
            'signed_native_tau_and_full_matrix_remain_unpaid': True}


def verify_typed_payloads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified = []
    for row in rows:
        original = json.loads(Path(row['original_receipt']['path']).read_text())
        typed = json.loads(Path(row['typed_receipt']['path']).read_text())
        if ['pins'] in row['excluded_metadata_paths']:
            original_request = json.loads((ROOT / (row['old_epoch'] + '_request.json')).read_text())
            typed_request = json.loads((ROOT / row['request']).read_text())
            assert original['pins'] == original_request and typed['pins'] == typed_request
            expected = copy.deepcopy(original_request)
            expected['consumer'] = row['fork']
            for field in ('capture_pins', 'typed_epoch', 'extra_args'):
                expected[field] = typed_request[field]
            assert expected == typed_request
        for selector in row['excluded_metadata_paths']:
            assert selector in (['pins'], ['runtime', 'elapsed_seconds'])
            left, right = original, typed
            for key in selector[:-1]:
                left, right = left[key], right[key]
            left.pop(selector[-1])
            right.pop(selector[-1])
        assert original == typed, row['typed_epoch']
        verified.append({'typed_epoch': row['typed_epoch'],
                         'excluded_metadata_paths': row['excluded_metadata_paths'],
                         'mathematical_payload_exact_equal': True})
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for path, expected in request['pins'].items():
        assert sha((ROOT / path).read_bytes()) == expected, path
    epochs = [verify_epoch(spec) for spec in request['epochs']]
    result = {'kind': 'EXACT_AUXILIARY_TRIAL_ADJOINT_PHYSICAL_QUOTIENT_FULL_ENERGY_PARENT_JOIN_NOT_PHASE',
              'epochs': epochs, 'pre_members_verified': sum(row['pre_members_verified'] for row in epochs),
              'preserved_failed_epochs': sum(row['returncode'] != 0 for row in epochs),
              'matrix_budget_control': matrix_budget_control(),
              'reviewed_boundaries': reviewed_boundaries(request),
              'typed_payload_transfers': verify_typed_payloads(request['typed_payload_transfers']),
              'original_epochs_modified_or_rebound': False,
              'native_ground_or_fullH_numerical_evolution_computed': False,
              'originating_auxiliary_sector_weight_or_wholeP0_loop_Gram_proved': False,
              'physical1_phase_stiffness_IR_atom': False,
              'unconditional_machine_attestation': False,
              'original_SOURCE_adopted': False, 'campaign_complete': False,
              'input_pins': request['pins']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Immutable auxiliary/adjoint/full-energy join; signed native ground response remains unpaid')


if __name__ == '__main__':
    main()

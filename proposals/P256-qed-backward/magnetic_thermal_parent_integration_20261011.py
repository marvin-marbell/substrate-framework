"""Consume immutable thermal/bulk epochs without rerunning their producers."""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def epoch_payload(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    assert digest(ROOT / spec['pre']) == spec['pre_sha256']
    assert digest(ROOT / spec['executed']) == spec['executed_sha256']
    with zipfile.ZipFile(ROOT / spec['pre']) as pre, zipfile.ZipFile(ROOT / spec['executed']) as exe:
        names = [name for name in pre.namelist() if not name.endswith('/')]
        assert len(names) == len(set(names))
        assert set(names) == set(spec['member_map'])
        for name in names:
            assert pre.read(name) == exe.read(spec['member_map'][name]), name
        raw = exe.read(spec['receipt_member'])
        assert hashlib.sha256(raw).hexdigest() == spec['receipt_sha256']
        assert raw == (ROOT / spec['receipt']).read_bytes()
        receipt = json.loads(raw)
        exit_record = json.loads(exe.read(spec['exit_member']))
        assert exit_record['natural_exit'] is True and exit_record['returncode'] == 0
        stable = exit_record.get('sources_inputs_unchanged', exit_record.get('source_input_unchanged',
                                 exit_record.get('source_inputs_unchanged', exit_record.get('source_request_unchanged'))))
        assert stable is True, spec['receipt']
    return receipt, {'members_compared': len(names), 'elapsed_seconds': exit_record['elapsed_seconds'],
                     'natural_exit': True, 'returncode': 0, 'source_inputs_unchanged': stable}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, pin in request['pins'].items():
        assert digest(ROOT / name) == pin, name
    payloads, evidence = {}, {}
    for name, spec in request['epochs'].items():
        payloads[name], evidence[name] = epoch_payload(spec)
    transfer = json.loads((ROOT / request['transfer']).read_text())
    exact_transfers = []
    for row in transfer['transfers']:
        old = json.loads((ROOT / row['original']).read_text())
        new = json.loads((ROOT / row['typed']).read_text())
        for key in row['metadata_excluded']:
            old.pop(key, None)
            new.pop(key, None)
        assert old == new
        exact_transfers.append({'original': row['original'], 'typed': row['typed'], 'exact_equal': True})
    thermal, bulk = payloads['thermal_original'], payloads['bulk_original']
    assert thermal['native_exact_response_enclosures'] is True
    assert thermal['native_numeric_expectations'] is False
    assert thermal['magnetic_response_sign_paid'] is False
    assert bulk['actual_native_state_spectral_convergence'] is False
    assert bulk['contrary_equal_pressure']['equal_pressure_implies_state_or_spectrum'] is False
    joined = payloads['parent_joined']
    assert joined['derived_conditional_response']['numeric_native_partition_or_response'] is False
    assert all(row['selected_site_from_average_rejected'] for row in joined['adverse_concentration'])
    affine = json.loads((ROOT / request['affine_addendum']).read_text())
    witness = affine['exact_reviewed_witness']
    assert F(witness['longitudinal_turns']) + F(witness['Peierls_turns']) == 1
    assert (F(witness['Peierls_turns']) - F(witness['longitudinal_turns'])) % 1 == F(witness['invalid_forward_plus_residual_turns'])
    assert affine['source_implementation_changed'] is False
    assert all(digest(ROOT / name) == pin for name, pin in request['pins'].items())
    result = {
        'kind': 'ACTUAL_IMMUTABLE_CANONICAL_THERMAL_BULK_INTEGRATION_NOT_GROUND_PHASE_OR_PHOTON',
        'source_sha256': digest(Path(__file__)), 'request_sha256': digest(args.request),
        'input_sha256': request['pins'], 'epochs': evidence,
        'epochs_verified': len(evidence), 'preexecution_members_compared': sum(row['members_compared'] for row in evidence.values()),
        'typed_payload_exact_transfer': exact_transfers,
        'corrected_affine_convention': {'full_section_and_translation_operator': 'PLUS', 'forward_reduced_gluing': 'MINUS',
                                       'ambiguity_not_executed_native_bug': True, 'original_proof_epoch_retained': True},
        'boundaries': {
            'ordinary_cover_partition': 'infinite',
            'native_ensemble': 'finite-N trace per primitive projected-dual-deck cell with ALL Bloch characters',
            'native_thermal_response': 'actual broad exact symbolic enclosure, sign unresolved',
            'native_numerical_crop': 'failed tails; not a converged native partition',
            'common_character_memory': 'one b for whole word, raw dynamic flow must close',
            'boundary_bulk': 'averaged fixed-beta moment bounds; no local state/dynamics identity',
            'Coulomb_surface_operator_route': 'rejected by exact canonical stripe',
            'qN_Gauss_decay_route': 'rejected; centre tends sech(pi), not magnetic stiffness',
            'joined_response': 'Holder upper perarea grows N^(5/2), not uniform IR control',
            'whole_ground_block': 'thermal zero blocks retained; limiting P0 cannot be removed before limit',
        },
        'fullphysical_ground_phase_stiffness_photon_admitted': False,
        'new_boundary_law_adopted_as_original_periodic': False, 'unconditional_machine_attestation': False,
        'original_SOURCE_adopted': False, 'campaign_complete': False,
        'old_producer_solver_root_checker_rerun': False,
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'epochs': result['epochs_verified'], 'members': result['preexecution_members_compared'],
                      'exact_transfers': len(exact_transfers), 'phase_pole': False}))


if __name__ == '__main__':
    main()

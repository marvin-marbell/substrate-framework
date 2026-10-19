"""Pin before execution, preserve every epoch, archive natural-exit scientific consumers."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
PROFILE = 'joined_phi_u1_uniform_gauss_limit_epoch4_20261010_actual_executed_inputs.zip'
MEMBER = 'joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json'
CONTRACT = Path('/home/administrator/.omp/agent/sessions/-/2026-10-09T16-26-59-293Z_01a1217d-415d-7376-a17d-16be721cbe28/local/issue222-common-singlet-energy-native-loop-dark-contract-20261012.json')


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    prefix = f'common_singlet_energy_{args.epoch}_20261012'
    request_path = ROOT / f'{prefix}_request.json'
    assert not request_path.exists(), 'Epochs are immutable: choose a new name'
    names = ['common_singlet_energy_consumer_20261012.py', Path(__file__).name,
             'common_singlet_energy_proof_20261012.json', 'common_singlet_energy_language_20261012.json',
             'canonical_ground_parent_normalized_filter_form_gate_20261012.json',
             'canonical_ground_parent_common_singlet_proof_20261012.json',
             'joined_phi_u1_canonical_path_signed_carrier_parent_packet_20261012.json',
             'joined_phi_u1_canonical_path_signed_carrier_parent_review_20261012.json',
             'joined_phi_u1_uniform_sharp_gauss_proof_20261010.json',
             'native_history_ground_proof_20261011.json', 'ground_weighted_form_review_proof_20261011.json',
             'canonical_quantum_path_boundary_proof_20261012.json', PROFILE]
    for failed in ('actual1', 'actual2', 'actual4'):
        names.extend(f'common_singlet_energy_{failed}_20261012_{suffix}'
                     for suffix in ('capsule.json', 'executed.zip'))
    names.extend(['common_singlet_energy_epoch1_correction_20261012.json',
                  'common_singlet_energy_epoch2_correction_20261012.json',
                  'common_singlet_energy_epoch4_correction_20261012.json'])
    captured = {name: (ROOT / name).read_bytes() for name in names}
    captured[str(CONTRACT)] = CONTRACT.read_bytes()
    assert sha(captured[str(CONTRACT)]) == '0905db8ba29d879692420e17154d34a96e5aab9473d93550ef36c496ee81b8e0'
    assert sha(captured[PROFILE]) == '8af45d28d48ae0ab8302dcea11cdb9356af972a5373ae2568008ff0a99587649'
    pins = {name: sha(raw) for name, raw in captured.items()}
    request = {'consumer': names[0], 'profile_archive': PROFILE, 'profile_member': MEMBER,
               'pins': pins, 'claims': 'Conditional analytic full-H comparison; exact native finite controls separate from native ground computation.'}
    request_path.write_text(json.dumps(request, indent=2) + '\n')
    pre = ROOT / f'{prefix}_preexecution.zip'
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in captured.items():
            archive.writestr(name.lstrip('/'), raw)
        archive.write(request_path, request_path.name)
    started = datetime.now(timezone.utc).isoformat()
    tick = time.monotonic()
    receipt = ROOT / f'{prefix}_receipt.json'
    command = [PYTHON, str(ROOT / names[0]), '--request', str(request_path), '--output', str(receipt)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic() - tick
    stdout, stderr, exitpath = [ROOT / f'{prefix}_{suffix}' for suffix in ('stdout.txt', 'stderr.txt', 'exit.json')]
    stdout.write_bytes(completed.stdout); stderr.write_bytes(completed.stderr)
    exitdata = {'command': command, 'returncode': completed.returncode, 'natural_exit': True,
                'started_at': started, 'completed_at': datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': elapsed}
    exitpath.write_text(json.dumps(exitdata, indent=2) + '\n')
    after = {name: sha((ROOT / name).read_bytes()) for name in captured}
    unchanged = pins == after
    outputs = [request_path, pre, stdout, stderr, exitpath]
    if receipt.exists():
        outputs.append(receipt)
    executed = ROOT / f'{prefix}_executed.zip'
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in captured.items():
            archive.writestr(name.lstrip('/'), raw)
        for path in outputs:
            archive.write(path, path.name)
    capsule = {'kind': 'IMMUTABLE_EXECUTED_NATIVE_ENERGY_CONSUMER_CAPSULE',
               'pins_before': pins, 'pins_after': after, 'unchanged': unchanged,
               'preexecution': {'path': pre.name, 'sha256': sha(pre.read_bytes())},
               'executed': {'path': executed.name, 'sha256': sha(executed.read_bytes())},
               'outputs': {path.name: sha(path.read_bytes()) for path in outputs},
               'runtime': exitdata, 'failed_epoch_preserved': completed.returncode != 0,
               'native_ground_computed': False, 'campaign_complete': False}
    (ROOT / f'{prefix}_capsule.json').write_text(json.dumps(capsule, indent=2) + '\n')
    for path in outputs + [executed, ROOT / f'{prefix}_capsule.json']:
        path.chmod(0o444)
    print(json.dumps({'epoch': prefix, 'returncode': completed.returncode, 'natural_exit': True,
                      'unchanged': unchanged, 'elapsed_seconds': elapsed}))
    if not unchanged:
        raise RuntimeError('Pinned input changed during epoch; preserved capsule is not admissible')
    sys.exit(completed.returncode)


if __name__ == '__main__':
    main()

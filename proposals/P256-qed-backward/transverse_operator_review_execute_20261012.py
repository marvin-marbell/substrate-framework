"""Freeze and execute only the NEW independent review consumer once."""
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request_path = ROOT/args.request
    request = json.loads(request_path.read_bytes())
    epoch = request['epoch']
    assert epoch.startswith('transverse_operator_review_')
    before = {name: sha(ROOT/name) for name in request['pins']}
    assert before == request['pins']
    request_before = sha(request_path)
    pre = ROOT/f'{epoch}_preexecution.zip'
    executed = ROOT/f'{epoch}_executed.zip'
    receipt = ROOT/f'{epoch}_receipt.json'
    stdout = ROOT/f'{epoch}_stdout.txt'
    stderr = ROOT/f'{epoch}_stderr.txt'
    exit_path = ROOT/f'{epoch}_exit.json'
    capsule_path = ROOT/f'{epoch}_capsule.json'
    for path in (pre, executed, receipt, stdout, stderr, exit_path, capsule_path):
        assert not path.exists(), f'Immutable epoch already exists: {path.name}'
    started_freeze = stamp()
    with zipfile.ZipFile(pre, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in request['pins']:
            z.write(ROOT/name, name)
        z.write(request_path, request_path.name)
    frozen_at = stamp()
    argv = [sys.executable, str(ROOT/request['consumer']), '--request', request_path.name,
            '--output', str(receipt)]
    started = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    stdout.write_bytes(result.stdout); stderr.write_bytes(result.stderr)
    completed = stamp()
    exit_data = {'returncode': result.returncode, 'natural_exit': True,
                 'completed_at': completed, 'elapsed_seconds': elapsed}
    exit_path.write_text(json.dumps(exit_data, indent=2)+'\n')
    after = {name: sha(ROOT/name) for name in request['pins']}
    outputs = [stdout, stderr, exit_path]+([receipt] if receipt.exists() else [])
    with zipfile.ZipFile(executed, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in request['pins']:
            z.write(ROOT/name, name)
        z.write(request_path, request_path.name)
        for path in outputs:
            z.write(path, path.name)
    capsule = {'kind': 'IMMUTABLE_INDEPENDENT_TRANSVERSE_OPERATOR_PROOF_REVIEW_EXECUTION',
               'epoch': epoch, 'argv': argv, 'cwd': str(ROOT), 'python': sys.version,
               'freeze_started_at': started_freeze, 'preexecution_completed_at': frozen_at,
               'runtime': exit_data, 'source_input_sha256_before': before,
               'source_input_sha256_after': after, 'request_sha256_before': request_before,
               'request_sha256_after': sha(request_path), 'source_inputs_unchanged': before == after,
               'preexecution': {'path': pre.name, 'sha256': sha(pre)},
               'executed': {'path': executed.name, 'sha256': sha(executed)},
               'outputs': {path.name: sha(path) for path in outputs},
               'original_producer_functions_imported_or_executed': False,
               'native_fullH_ground_backend': False, 'scientific_admission': False,
               'campaign_complete': False}
    capsule_path.write_text(json.dumps(capsule, indent=2)+'\n')
    print(json.dumps({'capsule': capsule_path.name, **exit_data,
                      'pins_unchanged': before == after}))
    if before != after or sha(request_path) != request_before:
        raise RuntimeError('Pinned immutable inputs changed')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

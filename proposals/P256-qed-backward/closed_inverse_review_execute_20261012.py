"""Exclusive capture-before-execution launcher for the independent review CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    path.chmod(0o444)


def capture(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 12, 0, 0, 0))
            info.external_attr = 0o444 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    assert args.epoch.startswith('closed_inverse_review_') and '/' not in args.epoch
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    members = {name: Path(row['path']).read_bytes() for name, row in request['inputs'].items()}
    before = {name: sha(raw) for name, raw in members.items()}
    assert before == {name: row['sha256'] for name, row in request['inputs'].items()}
    assert request['consumer'].startswith('closed_inverse_review_')
    assert request['consumer'] in members and Path(__file__).name in members
    members[args.request.name] = request_raw
    paths = {key: ROOT / (args.epoch + '_' + suffix) for key, suffix in {
        'pre': 'preexecution.zip', 'exe': 'executed.zip', 'receipt': 'receipt.json',
        'exit': 'exit.json', 'stdout': 'stdout.txt', 'stderr': 'stderr.txt', 'capsule': 'capsule.json'}.items()}
    assert all(not p.exists() for p in paths.values())
    captured_at = datetime.now(timezone.utc).isoformat()
    capture(paths['pre'], members)
    command = [request['python'], str(ROOT / request['consumer']), '--request', str(args.request), '--output', str(paths['receipt'])]
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    # No deadline, signal, original executable import, or original program launch.
    process = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic() - start
    finished_at = datetime.now(timezone.utc).isoformat()
    for label, raw in [('stdout', process.stdout), ('stderr', process.stderr)]:
        with paths[label].open('xb') as stream:
            stream.write(raw)
        paths[label].chmod(0o444)
    after = {name: sha(Path(row['path']).read_bytes()) for name, row in request['inputs'].items()}
    stable = after == before and sha(args.request.read_bytes()) == sha(request_raw)
    exit_data = {'schema': 1, 'argv': command, 'returncode': process.returncode,
                 'natural_exit': process.returncode >= 0, 'elapsed_seconds': elapsed,
                 'preexecution_captured_at': captured_at, 'child_started_at': started_at,
                 'child_finished_at': finished_at, 'source_input_bytes_unchanged': stable,
                 'source_input_sha256_before': before, 'source_input_sha256_after': after,
                 'request_sha256': sha(request_raw)}
    save(paths['exit'], exit_data)
    for label in ['receipt', 'exit', 'stdout', 'stderr']:
        if paths[label].exists():
            paths[label].chmod(0o444)
            members[paths[label].name] = paths[label].read_bytes()
    capture(paths['exe'], members)
    capsule = {'schema': 1, 'kind': 'INDEPENDENT_CLOSED_INVERSE_REVIEW_EXECUTION_CAPSULE',
               'epoch': args.epoch, 'exit': exit_data,
               'preexecution': {'path': paths['pre'].name, 'sha256': sha(paths['pre'].read_bytes()), 'meaning': 'Captured before this independent child execution'},
               'executed': {'path': paths['exe'].name, 'sha256': sha(paths['exe'].read_bytes())},
               'receipt': {'path': paths['receipt'].name, 'sha256': sha(paths['receipt'].read_bytes()) if paths['receipt'].exists() else None},
               'members_sha256': {name: sha(raw) for name, raw in members.items()},
               'original_program_rerun': False, 'native_ground_or_fullH_backend': False,
               'scientific_admission': False, 'campaign_complete': False}
    save(paths['capsule'], capsule)
    print(json.dumps({'capsule': paths['capsule'].name, 'returncode': process.returncode,
                      'natural_exit': process.returncode >= 0, 'source_inputs_unchanged': stable}))
    if process.returncode or not stable:
        raise SystemExit(process.returncode if process.returncode > 0 else 1)


if __name__ == '__main__':
    main()
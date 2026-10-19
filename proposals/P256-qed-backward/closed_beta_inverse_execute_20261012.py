"""Capture exclusive immutable inputs before the NEW inverse-construction CLI."""
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


def dump(path: Path, payload: Any) -> None:
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2)
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
    assert args.epoch.startswith('closed_beta_inverse_') and '/' not in args.epoch
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    members = {name: Path(row['path']).read_bytes() for name, row in request['inputs'].items()}
    before = {name: sha(raw) for name, raw in members.items()}
    assert before == {name: row['sha256'] for name, row in request['inputs'].items()}
    assert request['consumer'].startswith('closed_beta_inverse_')
    assert request['consumer'] in members and Path(__file__).name in members
    members[args.request.name] = request_raw
    paths = {label: ROOT/(args.epoch+'_'+suffix) for label, suffix in {
        'pre': 'preexecution.zip', 'executed': 'executed.zip', 'receipt': 'receipt.json',
        'exit': 'exit.json', 'stdout': 'stdout.txt', 'stderr': 'stderr.txt',
        'capsule': 'capsule.json'}.items()}
    assert all(not path.exists() for path in paths.values())
    captured_at = datetime.now(timezone.utc).isoformat()
    capture(paths['pre'], members)
    command = [request['python'], str(ROOT/request['consumer']), '--request', str(args.request),
               '--output', str(paths['receipt'])]
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    # No external child deadline/signal and no old producer/root/solver/checker rerun.
    process = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    finished_at = datetime.now(timezone.utc).isoformat()
    for label, raw in [('stdout', process.stdout), ('stderr', process.stderr)]:
        with paths[label].open('xb') as stream:
            stream.write(raw)
        paths[label].chmod(0o444)
    after = {name: sha(Path(row['path']).read_bytes()) for name, row in request['inputs'].items()}
    stable = before == after and sha(args.request.read_bytes()) == sha(request_raw)
    exit_data = {'schema': 1, 'argv': command, 'returncode': process.returncode,
                 'natural_exit': process.returncode >= 0, 'elapsed_seconds': elapsed,
                 'preexecution_captured_at': captured_at, 'child_started_at': started_at,
                 'child_finished_at': finished_at, 'source_input_bytes_unchanged': stable,
                 'source_input_sha256_before': before, 'source_input_sha256_after': after,
                 'request_sha256': sha(request_raw), 'python': request['python']}
    dump(paths['exit'], exit_data)
    for label in ['receipt', 'exit', 'stdout', 'stderr']:
        if paths[label].exists():
            paths[label].chmod(0o444)
            members[paths[label].name] = paths[label].read_bytes()
    capture(paths['executed'], members)
    capsule = {'schema': 1, 'kind': 'NEW_CLOSED_BETA_INVERSE_IMMUTABLE_CONSTRUCTION_CAPSULE',
               'epoch': args.epoch, 'exit': exit_data,
               'preexecution': {'path': paths['pre'].name, 'sha256': sha(paths['pre'].read_bytes()),
                                'meaning': 'Exact bytes captured BEFORE own child execution'},
               'executed': {'path': paths['executed'].name, 'sha256': sha(paths['executed'].read_bytes())},
               'receipt': {'path': paths['receipt'].name,
                           'sha256': sha(paths['receipt'].read_bytes()) if paths['receipt'].exists() else None},
               'all_executed_members_sha256': {name: sha(raw) for name, raw in members.items()},
               'native_ground_or_fullH_backend': False, 'scientific_admission': False,
               'campaign_complete': False}
    dump(paths['capsule'], capsule)
    print(json.dumps({'capsule': paths['capsule'].name, 'returncode': process.returncode,
                      'natural_exit': process.returncode >= 0, 'source_input_bytes_unchanged': stable}))
    if process.returncode != 0 or not stable:
        raise SystemExit(process.returncode if process.returncode > 0 else 1)


if __name__ == '__main__':
    main()

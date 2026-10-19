"""Exclusive immutable epochs for the independent electric history review CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
    path.chmod(0o444)


def archive(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as target:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 12, 0, 0, 0))
            info.external_attr = 0o444 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(info, raw)
    path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    assert args.epoch.startswith('electric_history_review_')
    request_raw = args.request.read_bytes()
    request = json.loads(request_raw)
    members = {key: Path(row['path']).read_bytes() for key, row in request['inputs'].items()}
    assert {key: sha(raw) for key, raw in members.items()} == {
        key: row['sha256'] for key, row in request['inputs'].items()}
    members[args.request.name] = request_raw
    prefix = ROOT/args.epoch
    pre = prefix.with_name(prefix.name+'_preexecution.zip')
    executed = prefix.with_name(prefix.name+'_executed.zip')
    output = prefix.with_name(prefix.name+'_receipt.json')
    exit_path = prefix.with_name(prefix.name+'_exit.json')
    stdout_path = prefix.with_name(prefix.name+'_stdout.txt')
    stderr_path = prefix.with_name(prefix.name+'_stderr.txt')
    capsule_path = prefix.with_name(prefix.name+'_capsule.json')
    assert all(not p.exists() for p in (pre, executed, output, exit_path, stdout_path, stderr_path, capsule_path))
    archive(pre, members)
    command = [request['python'], str(ROOT/request['consumer']), '--request', str(args.request), '--output', str(output)]
    started = time.monotonic()
    # No external deadline, signal, watcher, or original-producer execution.
    process = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    for path, raw in ((stdout_path, process.stdout), (stderr_path, process.stderr)):
        with path.open('xb') as stream:
            stream.write(raw)
        path.chmod(0o444)
    after = {key: sha(Path(row['path']).read_bytes()) for key, row in request['inputs'].items()}
    stable = after == {key: sha(raw) for key, raw in members.items() if key in request['inputs']}
    exit_data = {'schema': 1, 'command': command, 'returncode': process.returncode,
                 'natural_exit': process.returncode >= 0, 'elapsed_seconds': elapsed,
                 'source_input_bytes_unchanged': stable, 'preexecution_sha256': sha(pre.read_bytes()),
                 'stdout_sha256': sha(process.stdout), 'stderr_sha256': sha(process.stderr),
                 'executable': request['python']}
    dump(exit_path, exit_data)
    for path in (stdout_path, stderr_path, exit_path, output):
        if path.exists():
            path.chmod(0o444)
            members[path.name] = path.read_bytes()
    archive(executed, members)
    capsule = {'schema': 1, 'kind': 'INDEPENDENT_ELECTRIC_HISTORY_REVIEW_IMMUTABLE_SCIENTIFIC_CONSUMER_CAPSULE',
               'epoch': args.epoch, 'exit': exit_data,
               'preexecution': {'path': pre.name, 'sha256': sha(pre.read_bytes()),
                                'meaning': 'Captured before child execution, not frozen-parent snapshot backfill'},
               'executed': {'path': executed.name, 'sha256': sha(executed.read_bytes())},
               'receipt': {'path': output.name, 'sha256': sha(output.read_bytes()) if output.exists() else None},
               'review': {'path': request['review'], 'sha256': request['inputs'][request['review']]['sha256']},
               'all_executed_members_sha256': {key: sha(raw) for key, raw in members.items()},
               'scientific_admission': False, 'native_ground_or_fullH_backend': False,
               'campaign_complete': False}
    dump(capsule_path, capsule)
    print(json.dumps({'capsule': capsule_path.name, 'returncode': process.returncode,
                      'natural_exit': process.returncode >= 0, 'source_input_bytes_unchanged': stable}))
    if process.returncode != 0 or not stable:
        raise SystemExit(process.returncode if process.returncode > 0 else 1)


if __name__ == '__main__':
    main()

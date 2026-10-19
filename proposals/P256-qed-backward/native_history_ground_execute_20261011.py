"""Capture a NEW native-history consumer epoch before execution, preserving failures."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, raw: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    assert args.epoch.startswith('native_history_ground_')
    request_bytes = args.request.read_bytes()
    request: dict[str, Any] = json.loads(request_bytes)
    snapshots = {name: (ROOT/name).read_bytes() for name in request['pins']}
    before = {name: digest(raw) for name, raw in snapshots.items()}
    assert before == request['pins']
    snapshots[args.request.name] = request_bytes
    archive_names = {name: Path(name).name for name in snapshots}
    assert len(set(archive_names.values())) == len(archive_names)
    pre = ROOT/(args.epoch+'_preexecution.zip')
    executed = ROOT/(args.epoch+'_executed.zip')
    receipt = ROOT/(args.epoch+'_receipt.json')
    stdout_path = ROOT/(args.epoch+'_stdout.txt')
    stderr_path = ROOT/(args.epoch+'_stderr.txt')
    exit_path = ROOT/(args.epoch+'_exit.json')
    capsule_path = ROOT/(args.epoch+'_capsule.json')
    assert not any(path.exists() for path in (pre, executed, receipt, stdout_path, stderr_path, exit_path, capsule_path))
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr(archive_names[name], raw)
    captured = dt.datetime.now(dt.timezone.utc).isoformat()
    pre_hash = digest(pre.read_bytes())
    argv = [sys.executable, str(ROOT/request['consumer']), '--request', str(args.request), '--output', str(receipt)]
    started = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    save(stdout_path, result.stdout); save(stderr_path, result.stderr)
    after = {name: digest((ROOT/name).read_bytes()) for name in request['pins']}
    exit_record = {'returncode': result.returncode, 'natural_exit': True, 'elapsed_seconds': elapsed,
                   'completed_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    save(exit_path, (json.dumps(exit_record, indent=2)+'\n').encode())
    generated = [stdout_path, stderr_path, exit_path]
    if receipt.exists():
        generated.append(receipt)
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr(archive_names[name], raw)
        for path in generated:
            archive.writestr(path.name, path.read_bytes())
    capsule = {'schema': 1, 'kind': 'IMMUTABLE_NEW_NATIVE_HISTORY_EPOCH', 'epoch': args.epoch,
               'captured_before_execution_at': captured, 'argv': argv, 'cwd': str(ROOT),
               'source_input_sha256_before': before, 'source_input_sha256_after': after,
               'request_sha256_before': digest(request_bytes), 'request_sha256_after': digest(args.request.read_bytes()),
               'archive_member_map': archive_names,
               'preexecution': {'path': pre.name, 'sha256': pre_hash},
               'executed': {'path': executed.name, 'sha256': digest(executed.read_bytes())},
               'outputs': {path.name: digest(path.read_bytes()) for path in generated},
               'runtime': exit_record,
               'sources_inputs_unchanged': before == after and request_bytes == args.request.read_bytes(),
               'native_fullH_numerical_evolution_computed': False,
               'old_producers_solvers_checkers_imported_or_run': False}
    save(capsule_path, (json.dumps(capsule, indent=2)+'\n').encode())
    print(json.dumps({'capsule': capsule_path.name, 'returncode': result.returncode,
                      'natural_exit': True, 'elapsed_seconds': elapsed,
                      'source_input_unchanged': capsule['sources_inputs_unchanged']}))
    if result.returncode:
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()

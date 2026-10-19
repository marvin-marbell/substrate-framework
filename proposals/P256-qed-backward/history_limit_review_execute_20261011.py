"""Freeze NEW review inputs before naturally executing the independent consumer."""
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


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, raw: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    assert args.epoch.startswith('history_limit_review_')
    request_raw = args.request.read_bytes()
    request: dict[str, Any] = json.loads(request_raw)
    snapshots = {name: (ROOT / name).read_bytes() for name in request['pins']}
    before = {name: sha(raw) for name, raw in snapshots.items()}
    assert before == request['pins']
    snapshots[args.request.name] = request_raw
    pre = ROOT / (args.epoch + '_preexecution.zip')
    executed = ROOT / (args.epoch + '_executed.zip')
    receipt = ROOT / (args.epoch + '_receipt.json')
    stdout = ROOT / (args.epoch + '_stdout.txt')
    stderr = ROOT / (args.epoch + '_stderr.txt')
    exit_path = ROOT / (args.epoch + '_exit.json')
    capsule_path = ROOT / (args.epoch + '_capsule.json')
    assert not any(p.exists() for p in (pre, executed, receipt, stdout, stderr, exit_path, capsule_path))
    captured = dt.datetime.now(dt.timezone.utc).isoformat()
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr(Path(name).name, raw)
    argv = [sys.executable, str(ROOT / request['consumer']), '--request', str(args.request), '--output', str(receipt)]
    started = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    runtime = {'returncode': result.returncode, 'natural_exit': True, 'elapsed_seconds': time.monotonic() - started, 'completed_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    save(stdout, result.stdout)
    save(stderr, result.stderr)
    save(exit_path, (json.dumps(runtime, indent=2) + '\n').encode())
    outputs = [stdout, stderr, exit_path] + ([receipt] if receipt.exists() else [])
    after = {name: sha((ROOT / name).read_bytes()) for name in request['pins']}
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr(Path(name).name, raw)
        for path in outputs:
            archive.writestr(path.name, path.read_bytes())
    capsule = {'schema': 1, 'kind': 'IMMUTABLE_NEW_INDEPENDENT_HISTORY_LIMIT_METHOD_EPOCH', 'epoch': args.epoch, 'captured_before_execution_at': captured, 'argv': argv, 'cwd': str(ROOT), 'source_input_sha256_before': before, 'source_input_sha256_after': after, 'request_sha256_before': sha(request_raw), 'request_sha256_after': sha(args.request.read_bytes()), 'preexecution': {'path': pre.name, 'sha256': sha(pre.read_bytes())}, 'executed': {'path': executed.name, 'sha256': sha(executed.read_bytes())}, 'outputs': {p.name: sha(p.read_bytes()) for p in outputs}, 'runtime': runtime, 'sources_inputs_unchanged': before == after and request_raw == args.request.read_bytes(), 'old_producers_solvers_checkers_imported_or_run': False, 'method_only': True, 'native_ground_or_H_evolution_computed': False}
    save(capsule_path, (json.dumps(capsule, indent=2) + '\n').encode())
    print(json.dumps({'capsule': capsule_path.name, 'runtime': runtime, 'sources_inputs_unchanged': capsule['sources_inputs_unchanged']}))
    if result.returncode:
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()

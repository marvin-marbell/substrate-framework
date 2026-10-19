"""Immutable natural-exit capsule for the new electric-history consumer only."""
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
    assert args.epoch.startswith('electric_history_')
    request_raw = args.request.read_bytes()
    request: dict[str, Any] = json.loads(request_raw)
    snapshots = {name: (ROOT/name).read_bytes() for name in request['pins']}
    before = {name: digest(raw) for name, raw in snapshots.items()}
    assert before == request['pins']
    snapshots[args.request.name] = request_raw
    members = {name: Path(name).name for name in snapshots}
    assert len(set(members.values())) == len(members)
    pre = ROOT/(args.epoch+'_preexecution.zip'); executed = ROOT/(args.epoch+'_executed.zip')
    receipt = ROOT/(args.epoch+'_receipt.json'); capsule = ROOT/(args.epoch+'_capsule.json')
    stdout = ROOT/(args.epoch+'_stdout.txt'); stderr = ROOT/(args.epoch+'_stderr.txt')
    exit_path = ROOT/(args.epoch+'_exit.json')
    assert not any(p.exists() for p in (pre, executed, receipt, capsule, stdout, stderr, exit_path))
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr(members[name], raw)
    captured = dt.datetime.now(dt.timezone.utc).isoformat()
    argv = [sys.executable, str(ROOT/request['consumer']), '--request', str(args.request), '--output', str(receipt)]
    started = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    save(stdout, result.stdout); save(stderr, result.stderr)
    after = {name: digest((ROOT/name).read_bytes()) for name in request['pins']}
    run = {'returncode': result.returncode, 'natural_exit': True, 'elapsed_seconds': elapsed,
           'completed_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    save(exit_path, (json.dumps(run, indent=2)+'\n').encode())
    generated = [stdout, stderr, exit_path]+([receipt] if receipt.exists() else [])
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in snapshots.items():
            archive.writestr(members[name], raw)
        for path in generated:
            archive.writestr(path.name, path.read_bytes())
    data = {'schema': 1, 'kind': 'IMMUTABLE_NEW_ELECTRIC_HISTORY_SCIENTIFIC_EPOCH',
            'epoch': args.epoch, 'argv': argv, 'cwd': str(ROOT), 'captured_before_execution': captured,
            'source_input_sha256_before': before, 'source_input_sha256_after': after,
            'request_sha256_before': digest(request_raw), 'request_sha256_after': digest(args.request.read_bytes()),
            'archive_member_map': members, 'preexecution': {'path': pre.name, 'sha256': digest(pre.read_bytes())},
            'executed': {'path': executed.name, 'sha256': digest(executed.read_bytes())},
            'outputs': {path.name: digest(path.read_bytes()) for path in generated}, 'runtime': run,
            'sources_inputs_unchanged': before == after and request_raw == args.request.read_bytes(),
            'native_ground_or_fullH_evolution_computed': False, 'old_producer_root_solver_checker_run': False}
    save(capsule, (json.dumps(data, indent=2)+'\n').encode())
    print(json.dumps({'capsule': capsule.name, 'returncode': result.returncode, 'natural_exit': True,
                      'sources_inputs_unchanged': data['sources_inputs_unchanged'], 'elapsed_seconds': elapsed}))
    if result.returncode:
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()

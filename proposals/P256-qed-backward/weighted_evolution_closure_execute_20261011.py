"""Immutable preexecution/executed capsules for the NEW weighted closure CLI only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parent


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def immutable_json(path: Path, value: dict[str, Any]) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def immutable_zip(path: Path, payload: dict[str, bytes]) -> str:
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in payload.items():
            archive.writestr(name, raw)
    return digest(path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--stem', required=True)
    args = parser.parse_args()
    if not args.stem.startswith('weighted_evolution_closure_'):
        raise ValueError('Owned output prefix required')
    request = json.loads(args.request.read_text())
    payload = {name: (ROOT/name).read_bytes() for name in request['pins']}
    before = {name: digest(raw) for name, raw in payload.items()}
    if before != request['pins']:
        raise ValueError('Source/input revision mismatch before execution')
    payload[args.request.name] = args.request.read_bytes()
    pre = ROOT/f'{args.stem}_preexecution.zip'
    pre_sha = immutable_zip(pre, payload)
    receipt = ROOT/f'{args.stem}_receipt.json'
    command = [sys.executable, '-B', str(ROOT/request['consumer']),
               '--request', str(args.request), '--output', str(receipt)]
    started = time.monotonic()
    # No deadline, wrapper kill, shell pipeline or reused producer: await the
    # child's actual natural process exit, whether success or scientific error.
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    stdout = ROOT/f'{args.stem}_stdout.txt'
    stderr = ROOT/f'{args.stem}_stderr.txt'
    with stdout.open('xb') as stream:
        stream.write(completed.stdout)
    with stderr.open('xb') as stream:
        stream.write(completed.stderr)
    after = {name: digest((ROOT/name).read_bytes()) for name in request['pins']}
    exit_path = ROOT/f'{args.stem}_exit.json'
    exit_data = {'command': command, 'returncode': completed.returncode,
                 'natural_exit': True, 'elapsed_seconds': elapsed,
                 'source_input_sha256_before': before,
                 'source_input_sha256_after': after,
                 'source_input_unchanged': before == after,
                 'request_sha256_before': digest(payload[args.request.name]),
                 'request_sha256_after': digest(args.request.read_bytes()),
                 'no_old_producer_or_solver_run': True}
    immutable_json(exit_path, exit_data)
    executed_payload = dict(payload)
    for path in (stdout, stderr, exit_path, receipt):
        if path.exists():
            executed_payload[path.name] = path.read_bytes()
    executed = ROOT/f'{args.stem}_executed.zip'
    executed_sha = immutable_zip(executed, executed_payload)
    capsule = {'schema': 1, 'kind': 'NEW_WEIGHTED_EVOLUTION_IMMUTABLE_NATURAL_EXIT_CAPSULE',
               'preexecution': {'path': pre.name, 'sha256': pre_sha,
                                'members': {name: digest(raw) for name, raw in payload.items()}},
               'executed': {'path': executed.name, 'sha256': executed_sha,
                            'members': {name: digest(raw) for name, raw in executed_payload.items()}},
               'exit': exit_data,
               'scientific_scope': 'native exact form/graph controls plus synthetic mechanism; analytic proof separate',
               'parent_final_diagnostics_only': True}
    immutable_json(ROOT/f'{args.stem}_capsule.json', capsule)
    print(json.dumps({'returncode': completed.returncode, 'natural_exit': True,
                      'elapsed_seconds': elapsed, 'stable': before == after,
                      'capsule': f'{args.stem}_capsule.json'}))
    if before != after or completed.returncode != 0:
        raise SystemExit(completed.returncode or 1)


if __name__ == '__main__':
    main()

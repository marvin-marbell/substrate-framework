"""Capture a NEW independent review execution, never rerun the frozen producer."""
from __future__ import annotations
import argparse
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


def save(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    members = sorted(set(request['pins']) | {args.request.name})
    assert all(sha(ROOT/name) == expected for name, expected in request['pins'].items())
    pre = ROOT/(args.prefix+'_preexecution.zip')
    with zipfile.ZipFile(pre, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for name in members:
            bundle.write(ROOT/name, name)
    pre_hash = sha(pre)
    receipt = ROOT/(args.prefix+'_receipt.json')
    command = [sys.executable, str(ROOT/request['consumer']), '--request', str(args.request), '--output', str(receipt)]
    started = time.time_ns()
    child = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    finished = time.time_ns()
    stdout = ROOT/(args.prefix+'_stdout.txt'); stdout.write_bytes(child.stdout)
    stderr = ROOT/(args.prefix+'_stderr.txt'); stderr.write_bytes(child.stderr)
    exit_path = ROOT/(args.prefix+'_exit.json')
    save(exit_path, {'command': command, 'returncode': child.returncode, 'natural_exit': True,
                     'started_ns': started, 'finished_ns': finished, 'elapsed_seconds': (finished-started)/1e9,
                     'preexecution_closed_before_child_start': True,
                     'stdout_sha256': sha(stdout), 'stderr_sha256': sha(stderr)})
    executed_members = members+[stdout.name, stderr.name, exit_path.name]
    if receipt.exists():
        executed_members.append(receipt.name)
    executed = ROOT/(args.prefix+'_executed.zip')
    with zipfile.ZipFile(executed, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for name in executed_members:
            bundle.write(ROOT/name, name)
    capsule_path = ROOT/(args.prefix+'_capsule.json')
    capsule = {'kind': 'INDEPENDENT_STATIC_BOUNDARY_NATURAL_EXIT_CAPSULE',
               'preexecution': {'path': pre.name, 'sha256': pre_hash,
                                'members': {name: sha(ROOT/name) for name in members}},
               'executed': {'path': executed.name, 'sha256': sha(executed),
                            'members': {name: sha(ROOT/name) for name in executed_members}},
               'returncode': child.returncode, 'natural_exit': True,
               'scientific_admission': False, 'original_producer_rerun': False}
    save(capsule_path, capsule)
    print(json.dumps({'capsule': capsule_path.name, 'returncode': child.returncode,
                      'natural_exit': True, 'preexecution_sha256': pre_hash,
                      'executed_sha256': sha(executed)}))
    sys.exit(child.returncode)


if __name__ == '__main__':
    main()

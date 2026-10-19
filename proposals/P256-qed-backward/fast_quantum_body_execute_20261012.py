"""Capture a new scientific CLI epoch; preserve every natural failure unchanged."""
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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(path: Path, members: list[Path]) -> list[dict[str, Any]]:
    if path.exists():
        raise FileExistsError(path)
    names: set[str] = set()
    manifest = []
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            if member.name in names:
                raise ValueError('duplicate immutable member: ' + member.name)
            names.add(member.name)
            content = member.read_bytes()
            archive.writestr(member.name, content)
            manifest.append({'member': member.name, 'path': str(member),
                             'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--epoch', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    root = Path(__file__).resolve().parent
    consumer = root / request['new_scientific_consumer']
    paths = [args.request, Path(__file__), consumer]
    paths.extend(Path(pin['path']) for pin in request['pins'].values())
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    for name, pin in request['pins'].items():
        if digest(Path(pin['path'])) != pin['sha256']:
            raise ValueError('pin changed before execution: ' + name)
    pre = args.epoch.with_name(args.epoch.name + '_preexecution.zip')
    before = capture(pre, unique)
    receipt = args.epoch.with_name(args.epoch.name + '_receipt.json')
    command = [request['python'], str(consumer), '--request', str(args.request), '--output', str(receipt)]
    environment = dict(os.environ)
    environment.update({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'PYTHONHASHSEED': '0'})
    start = time.monotonic()
    completed = subprocess.run(command, cwd=root, env=environment, capture_output=True, check=False)
    wall = time.monotonic() - start
    stdout = args.epoch.with_name(args.epoch.name + '_stdout.txt')
    stderr = args.epoch.with_name(args.epoch.name + '_stderr.txt')
    stdout.write_bytes(completed.stdout)
    stderr.write_bytes(completed.stderr)
    unchanged = all(digest(Path(item['path'])) == item['sha256'] for item in before)
    exit_path = args.epoch.with_name(args.epoch.name + '_exit.json')
    exit_data = {'kind': 'NEW_SCIENTIFIC_NATURAL_EXIT', 'argv': command, 'cwd': str(root),
                 'python': request['python'], 'executor_python': sys.executable, 'python_version': sys.version,
                 'environment': {name: environment[name] for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'PYTHONHASHSEED')},
                 'wall_seconds': wall, 'returncode': completed.returncode, 'natural_exit': completed.returncode >= 0,
                 'stderr_empty': not completed.stderr, 'input_source_unchanged': unchanged,
                 'preexecution': {'path': str(pre), 'sha256': digest(pre)},
                 'immutable_pre_members': before, 'stdout_sha256': digest(stdout), 'stderr_sha256': digest(stderr)}
    exit_path.write_text(json.dumps(exit_data, indent=2) + '\n')
    produced = [stdout, stderr, exit_path] + ([receipt] if receipt.exists() else [])
    executed = args.epoch.with_name(args.epoch.name + '_executed.zip')
    after = capture(executed, unique + produced)
    capsule = {'kind': 'IMMUTABLE_NEW_FAST_QUANTUM_BODY_SCIENTIFIC_EPOCH', 'governor': request['governor'],
               'natural_exit': exit_data['natural_exit'], 'returncode': completed.returncode,
               'wall_seconds': wall, 'input_source_unchanged': unchanged,
               'preexecution': {'path': str(pre), 'sha256': digest(pre), 'members': before},
               'executed': {'path': str(executed), 'sha256': digest(executed), 'members': after},
               'exit': {'path': str(exit_path), 'sha256': digest(exit_path)},
               'receipt': {'path': str(receipt), 'sha256': digest(receipt)} if receipt.exists() else None,
               'scientific_admission': False, 'campaign_complete': False}
    capsule_path = args.epoch.with_name(args.epoch.name + '_capsule.json')
    capsule_path.write_text(json.dumps(capsule, indent=2) + '\n')
    print(json.dumps({'capsule': str(capsule_path), 'natural_exit': exit_data['natural_exit'],
                      'returncode': completed.returncode, 'wall_seconds': wall, 'source_input_unchanged': unchanged}), flush=True)
    if not unchanged:
        raise RuntimeError('immutable epoch source/input changed; captured output retained')
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()

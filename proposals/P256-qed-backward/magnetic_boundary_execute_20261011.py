"""Freeze and execute ONLY the new magnetic boundary scientific consumer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def archive(path: Path, payload: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as output:
        for name, raw in payload.items():
            output.writestr(name, raw)


def run(epoch: str, request_path: Path) -> int:
    if not epoch.startswith('magnetic_boundary_') or '/' in epoch:
        raise ValueError('Owned immutable magnetic_boundary epoch required')
    paths = {key: ROOT/(epoch+'_'+key) for key in
             ('preexecution.zip', 'executed.zip', 'stdout.txt', 'stderr.txt',
              'receipt.json', 'exit.json', 'capsule.json')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Immutable epoch already exists')
    request = json.loads(request_path.read_text())
    producer = request.get('producer', 'magnetic_boundary_scientist_20261011.py')
    if not producer.startswith('magnetic_boundary_') or Path(producer).name != producer or not producer.endswith('.py'):
        raise ValueError('Only a NEW owned boundary scientific producer may execute')
    names = list(request['input_sha256']) + [request_path.name, Path(__file__).name, producer]
    captured = {name: (ROOT/name).read_bytes() for name in dict.fromkeys(names)}
    for name, pin in request['input_sha256'].items():
        if sha(captured[name]) != pin:
            raise ArithmeticError('Input hash changed: '+name)
    argv = [PYTHON, '-B', str(ROOT/producer),
            '--request', str(request_path), '--output', str(paths['receipt.json'])]
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    manifest: dict[str, Any] = dict(kind='IMMUTABLE_PREEXECUTION_MAGNETIC_BOUNDARY_DOMAIN',
        argv=argv, cwd=str(ROOT), source_input_sha256={name: sha(raw) for name, raw in captured.items()},
        old_main_solver_eigensolver_root_checker_executed=False,
        imported_old_pure_helpers_only=['incidence', 'car_transfer'],
        scientific_scope='Exact structural native operator/boundary controls, no ground or stiffness',
        planned_request=request)
    manifest_name = 'magnetic_boundary_preexecution_manifest.json'
    captured[manifest_name] = (json.dumps(manifest, indent=2)+'\n').encode()
    archive(paths['preexecution.zip'], captured)
    started = time.monotonic()
    proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    paths['stdout.txt'].write_bytes(proc.stdout)
    paths['stderr.txt'].write_bytes(proc.stderr)
    unchanged = all((ROOT/name).read_bytes() == raw for name, raw in captured.items() if name != manifest_name)
    info = dict(kind='NEW_SCIENTIFIC_NATURAL_EXIT', returncode=proc.returncode,
        natural_exit=proc.returncode >= 0, elapsed_seconds=elapsed,
        sources_inputs_unchanged=unchanged, argv=argv,
        preexecution_sha256=sha(paths['preexecution.zip'].read_bytes()),
        stdout_sha256=sha(proc.stdout), stderr_sha256=sha(proc.stderr),
        receipt_sha256=sha(paths['receipt.json'].read_bytes()) if paths['receipt.json'].exists() else None)
    paths['exit.json'].write_text(json.dumps(info, indent=2)+'\n')
    executed = dict(captured)
    for key in ('stdout.txt', 'stderr.txt', 'receipt.json', 'exit.json'):
        if paths[key].exists():
            executed[paths[key].name] = paths[key].read_bytes()
    archive(paths['executed.zip'], executed)
    capsule = dict(epoch=epoch, **info, executed_sha256=sha(paths['executed.zip'].read_bytes()),
        files={key: path.name for key, path in paths.items()}, failed_epochs_preserved=True,
        actual_physical_ground_stiffness_or_IR_atom=False)
    paths['capsule.json'].write_text(json.dumps(capsule, indent=2)+'\n')
    print(json.dumps(capsule, indent=2))
    return proc.returncode if unchanged else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    parser.add_argument('--request', required=True, type=Path)
    args = parser.parse_args()
    sys.exit(run(args.epoch, args.request))

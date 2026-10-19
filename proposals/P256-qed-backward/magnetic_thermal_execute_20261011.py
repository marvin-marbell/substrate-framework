"""Freeze and naturally execute only the new owned magnetic thermal CLI."""
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
    if not epoch.startswith('magnetic_thermal_') or epoch.startswith('magnetic_thermal_parent_') or Path(epoch).name != epoch:
        raise ValueError('Disjoint owned thermal epoch required')
    paths = {key: ROOT/(epoch+'_'+key) for key in
             ('preexecution.zip', 'executed.zip', 'stdout.txt', 'stderr.txt',
              'receipt.json', 'exit.json', 'capsule.json')}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError('Immutable scientific epoch exists')
    request = json.loads(request_path.read_text())
    producer = request['producer']
    if producer != 'magnetic_thermal_scientist_20261011.py':
        raise ValueError('Only the assigned new thermal source can execute')
    names = list(request['pins'])+[request_path.name, Path(__file__).name, producer]
    captured = {name: (ROOT/name).read_bytes() for name in dict.fromkeys(names)}
    assert all(sha(captured[name]) == pin for name, pin in request['pins'].items())
    argv = [PYTHON, '-B', str(ROOT/producer), '--request', str(request_path), '--output', str(paths['receipt.json'])]
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    manifest: dict[str, Any] = dict(kind='IMMUTABLE_PREEXECUTION_NATIVE_CANONICAL_MAGNETIC_THERMAL',
        argv=argv, cwd=str(ROOT), source_input_sha256={name: sha(raw) for name, raw in captured.items()},
        old_producer_eigensolver_root_checker_executed=False,
        scientific_scope='Full native ensemble/operator/exact enclosure; finite matrices METHOD only; failed analytic crop retained',
        planned_request=request)
    manifest_name = 'magnetic_thermal_preexecution_manifest.json'
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
        source_sha256=sha(captured[producer]), request_sha256=sha(captured[request_path.name]),
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
        native_response_sign_phase_or_photon=False)
    paths['capsule.json'].write_text(json.dumps(capsule, indent=2)+'\n')
    print(json.dumps(capsule, indent=2))
    return proc.returncode if unchanged else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    parser.add_argument('--request', required=True, type=Path)
    args = parser.parse_args()
    sys.exit(run(args.epoch, args.request))

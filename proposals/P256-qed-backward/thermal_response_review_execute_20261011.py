"""Freeze only the owned review control and execute it to natural exit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
EPOCH = 'thermal_response_review_actual1_20261011'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def freeze(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)


def main() -> None:
    request_path = ROOT / (EPOCH + '_request.json')
    request = json.loads(request_path.read_text())
    producer = request['producer']
    assert producer == 'thermal_response_review_control_20261011.py'
    paths = {suffix: ROOT / (EPOCH + '_' + suffix) for suffix in (
        'preexecution.zip', 'executed.zip', 'receipt.json', 'stdout.txt',
        'stderr.txt', 'exit.json', 'capsule.json')}
    assert not any(path.exists() for path in paths.values())
    names = list(request['pins']) + request['original_archive_members'] + [
        producer, request_path.name, Path(__file__).name]
    captured = {name: (ROOT / name).read_bytes() for name in dict.fromkeys(names)}
    assert all(sha(captured[name]) == pin for name, pin in request['pins'].items())
    argv = [PYTHON, '-B', str(ROOT / producer), '--request', str(request_path),
            '--output', str(paths['receipt.json'])]
    manifest_name = 'thermal_response_review_preexecution_manifest.json'
    manifest = {'kind': 'IMMUTABLE_INDEPENDENT_CONTROL_INPUTS', 'argv': argv,
                'cwd': str(ROOT), 'source_input_sha256': {name: sha(raw) for name, raw in captured.items()},
                'old_entrypoint_executed': False, 'scope': request['scope']}
    captured[manifest_name] = (json.dumps(manifest, indent=2) + '\n').encode()
    freeze(paths['preexecution.zip'], captured)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start = time.monotonic()
    completed = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, check=False)
    elapsed = time.monotonic() - start
    paths['stdout.txt'].write_bytes(completed.stdout)
    paths['stderr.txt'].write_bytes(completed.stderr)
    stable = all((ROOT / name).read_bytes() == raw for name, raw in captured.items() if name != manifest_name)
    result = {'kind': 'INDEPENDENT_CONTROL_NATURAL_EXIT', 'argv': argv,
              'returncode': completed.returncode, 'natural_exit': completed.returncode >= 0,
              'elapsed_seconds': elapsed, 'source_inputs_unchanged': stable,
              'source_sha256': sha(captured[producer]), 'request_sha256': sha(captured[request_path.name]),
              'preexecution_sha256': sha(paths['preexecution.zip'].read_bytes()),
              'stdout_sha256': sha(completed.stdout), 'stderr_sha256': sha(completed.stderr),
              'receipt_sha256': sha(paths['receipt.json'].read_bytes()) if paths['receipt.json'].exists() else None}
    paths['exit.json'].write_text(json.dumps(result, indent=2) + '\n')
    executed = dict(captured)
    for suffix in ('receipt.json', 'stdout.txt', 'stderr.txt', 'exit.json'):
        if paths[suffix].exists():
            executed[paths[suffix].name] = paths[suffix].read_bytes()
    freeze(paths['executed.zip'], executed)
    capsule = dict(result, epoch=EPOCH, executed_sha256=sha(paths['executed.zip'].read_bytes()),
                   files={suffix: path.name for suffix, path in paths.items()}, native_admission=False)
    paths['capsule.json'].write_text(json.dumps(capsule, indent=2) + '\n')
    print(json.dumps(capsule, indent=2))
    if completed.returncode or not stable:
        raise SystemExit(completed.returncode or 2)


if __name__ == '__main__':
    main()

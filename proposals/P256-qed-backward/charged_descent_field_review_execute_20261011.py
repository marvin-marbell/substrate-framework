"""Exclusive immutable byte epochs and observed natural exit for the NEW reviewer."""
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

import bulk_electric_phase_consumer as primitive

ROOT = Path(__file__).resolve().parent
CONTROL = 'charged_descent_field_review_control_20261011.py'
REQUEST = 'charged_descent_field_review_request_20261011.json'


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exclusive(path: Path, raw: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(raw)


def archive(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as stream:
        for name, raw in sorted(files.items()):
            stream.writestr(name, raw)
    path.chmod(0o444)


def encode(value: Any) -> bytes:
    return (json.dumps(value, indent=2)+'\n').encode()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--epoch', required=True)
    args = parser.parse_args(); prefix = f'charged_descent_field_review_{args.epoch}_20261011_'
    request = json.loads((ROOT/REQUEST).read_bytes())
    sources = {Path(__file__).name: Path(__file__).read_bytes(), CONTROL: (ROOT/CONTROL).read_bytes(), REQUEST: (ROOT/REQUEST).read_bytes(), 'charged_descent_field_review_language_20261011.json': (ROOT/'charged_descent_field_review_language_20261011.json').read_bytes()}
    sources.update(primitive.CAPTURED)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).parent == ROOT and str(filename).endswith('.py'):
            sources[Path(filename).name] = Path(filename).read_bytes()
    sources.update({name: (ROOT/name).read_bytes() for name in request['pins']})
    assert all(digest(sources[name]) == value for name, value in request['pins'].items())
    manifest = {'kind': 'EXCLUSIVE_NEW_FIELD_REVIEW_PREEXECUTION', 'plan_id': request['plan_id'], 'task_governing_revision': request['task_governing_revision'], 'source_input_hashes': {name: digest(raw) for name, raw in sources.items()}, 'no_target_main_or_old_producer': True}
    manifest_name = prefix+'preexecution.json'; exclusive(ROOT/manifest_name, encode(manifest))
    pre_name = prefix+'preexecution.zip'; archive(ROOT/pre_name, {**sources, manifest_name: (ROOT/manifest_name).read_bytes()})
    command = [sys.executable, CONTROL, '--request', REQUEST, '--output', prefix+'proof.json']
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, check=False)
    elapsed = time.monotonic()-started
    stdout_name, stderr_name = prefix+'stdout.bin', prefix+'stderr.bin'
    exclusive(ROOT/stdout_name, completed.stdout); exclusive(ROOT/stderr_name, completed.stderr)
    unchanged = {name: (ROOT/name).read_bytes() == raw for name, raw in sources.items()}
    outputs = {stdout_name: completed.stdout, stderr_name: completed.stderr}
    proof_path = ROOT/(prefix+'proof.json')
    if proof_path.exists():
        outputs[proof_path.name] = proof_path.read_bytes()
    exit_value = {'kind': 'OBSERVED_NEW_REVIEWER_NATURAL_EXIT', 'command': command, 'cwd': str(ROOT), 'returncode': completed.returncode, 'natural_exit': completed.returncode >= 0, 'elapsed_seconds': elapsed, 'all_source_inputs_unchanged': all(unchanged.values()), 'source_input_unchanged': unchanged, 'source_input_hashes': manifest['source_input_hashes'], 'output_hashes': {name: digest(raw) for name, raw in outputs.items()}, 'preexecution_archive': {'name': pre_name, 'sha256': digest((ROOT/pre_name).read_bytes())}, 'verification_scope': 'NEW independent finite/rational controls only; no static lint/tests/oldroot/ground/checkers'}
    exit_name = prefix+'exit.json'; exclusive(ROOT/exit_name, encode(exit_value))
    outputs[exit_name] = (ROOT/exit_name).read_bytes()
    executed_name = prefix+'executed.zip'; archive(ROOT/executed_name, {**sources, manifest_name: (ROOT/manifest_name).read_bytes(), **outputs})
    delivery = {'exit': exit_name, 'preexecution': pre_name, 'executed': executed_name, 'executed_sha256': digest((ROOT/executed_name).read_bytes()), 'proof': proof_path.name if proof_path.exists() else None, 'returncode': completed.returncode, 'all_source_inputs_unchanged': all(unchanged.values())}
    exclusive(ROOT/(prefix+'delivery.json'), encode(delivery))
    print(json.dumps(delivery))
    if completed.stdout:
        sys.stdout.buffer.write(completed.stdout)
    if completed.stderr:
        sys.stderr.buffer.write(completed.stderr)
    raise SystemExit(completed.returncode if completed.returncode >= 0 else 1)


if __name__ == '__main__':
    main()

"""Capture only NEW independent consumers with complete local import closure."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def execute(suffix, label=None):
    prefix = 'rich_review_wave7_20261014_' + suffix
    entry = ROOT / (prefix + '.py')
    inputs = ROOT / (prefix + '_input.json')
    paths = {entry, inputs, Path(__file__).resolve()}
    pending = [entry]
    while pending:
        current = pending.pop()
        for node in ast.walk(ast.parse(current.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                     else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                path = ROOT / (name + '.py')
                if path.is_file() and path not in paths:
                    paths.add(path)
                    pending.append(path)
    source = {path.name: path.read_bytes() for path in paths}
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    command = [PYTHON, str(entry)]
    start = time.monotonic()
    done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True)
    artifact_prefix = prefix if label is None else prefix + '_' + label
    stdout = ROOT / (artifact_prefix + '_stdout.txt')
    stderr = ROOT / (artifact_prefix + '_stderr.txt')
    stdout.write_bytes(done.stdout)
    stderr.write_bytes(done.stderr)
    result = ROOT / (prefix + '_result.json')
    outputs = [stdout, stderr] + ([result] if result.is_file() else [])
    digest = lambda value: hashlib.sha256(value).hexdigest()
    receipt = {'kind': 'NEW_WHOLE_H_PHYSICAL_CONSUMER_EXACT_BINDING', 'command': command,
               'cwd': str(ROOT), 'natural_exit': done.returncode, 'elapsed_seconds': time.monotonic() - start,
               'environment': {key: env[key] for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')},
               'sources_and_inputs_sha256': {name: digest(value) for name, value in sorted(source.items())},
               'outputs_sha256': {path.name: digest(path.read_bytes()) for path in outputs},
               'sources_unchanged_during_run': all((ROOT / name).read_bytes() == value for name, value in source.items())}
    exitpath = ROOT / (artifact_prefix + '_exit.json')
    exitpath.write_text(json.dumps(receipt, indent=2) + '\n')
    with zipfile.ZipFile(ROOT / (artifact_prefix + '_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as capsule:
        for name, value in sorted(source.items()):
            capsule.writestr(name, value)
        for path in outputs + [exitpath]:
            capsule.write(path, path.name)
    print(json.dumps(receipt, indent=2))
    if done.returncode:
        print(done.stderr.decode())
        raise SystemExit(done.returncode)


if __name__ == '__main__':
    execute(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

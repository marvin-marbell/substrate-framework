"""Exclusive finite/rational reviewer execution; preserve every failure epoch."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, raw: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(raw)
    path.chmod(0o444)


def archive(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as capsule:
        for name, raw in members.items():
            capsule.writestr(name, raw)
    path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    assert args.epoch.replace('_', '').isalnum()
    prefix = 'charged_descent_energy_review_' + args.epoch + '_20261011_'
    inputs_path = ROOT / 'charged_descent_energy_review_inputs_20261011.json'
    input_pins = json.loads(inputs_path.read_bytes())['pins']
    names = list(input_pins) + ['charged_descent_energy_review_control_20261011.py',
                              Path(__file__).name, inputs_path.name,
                              'charged_descent_energy_review_derivation_20261011.json',
                              'charged_descent_energy_review_language_20261011.json']
    members = {name: (ROOT / name).read_bytes() for name in names}
    before = ROOT / (prefix + 'preexecution.zip')
    archive(before, members)
    command = [PYTHON, 'charged_descent_energy_review_control_20261011.py', '--prefix', prefix]
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, capture_output=True)
    save(ROOT / (prefix + 'stdout.txt'), result.stdout)
    save(ROOT / (prefix + 'stderr.txt'), result.stderr)
    unchanged = all((ROOT / name).read_bytes() == raw for name, raw in members.items())
    outputs = {name: (ROOT / name).read_bytes() for name in (prefix + 'proof.json', prefix + 'receipt.json') if (ROOT / name).is_file()}
    for name in outputs:
        (ROOT / name).chmod(0o444)
    exit_path = ROOT / (prefix + 'exit.json')
    record = {'command': command, 'cwd': str(ROOT), 'natural_returncode': result.returncode,
              'termination': 'subprocess.run returned naturally; no timeout, signal or inferred exit',
              'seconds': time.monotonic() - started, 'source_inputs_unchanged': unchanged,
              'source_input_hashes': {name: sha(raw) for name, raw in members.items()},
              'output_hashes': {name: sha(raw) for name, raw in outputs.items()},
              'stdout_sha256': sha(result.stdout), 'stderr_sha256': sha(result.stderr),
              'preexecution_sha256': sha(before.read_bytes()), 'old_producer_runs': 0}
    save(exit_path, (json.dumps(record, indent=2) + '\n').encode())
    archive(ROOT / (prefix + 'executed.zip'), {**members, **outputs,
            prefix + 'stdout.txt': result.stdout, prefix + 'stderr.txt': result.stderr,
            exit_path.name: exit_path.read_bytes()})
    print(json.dumps({'prefix': prefix, 'natural_returncode': result.returncode,
                      'source_inputs_unchanged': unchanged, 'stdout': result.stdout.decode(errors='replace'),
                      'stderr': result.stderr.decode(errors='replace')}))
    raise SystemExit(result.returncode if unchanged else 2)


if __name__ == '__main__':
    main()

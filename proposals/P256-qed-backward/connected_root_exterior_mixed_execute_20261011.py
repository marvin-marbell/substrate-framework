"""Exclusive captured-byte execution of the overlapping unequal-root consumer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import zipfile

from connected_root_exterior_execute_20261011 import archive, digest

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def main(epoch: str) -> None:
    manifest = json.loads((ROOT/'connected_root_exterior_mixed_inputs_20261011.json').read_bytes())
    selected = list(manifest['sha256'])+['connected_root_exterior_mixed_inputs_20261011.json',
        'connected_root_exterior_mixed_scientist_20261011.py', 'connected_root_exterior_mixed_execute_20261011.py']
    captured = {name: (ROOT/name).read_bytes() for name in selected}
    dependency_archive = ROOT/'connected_root_exterior_actual1_20261011_preexecution.zip'
    dependency_pin = '55f66e6041cdf98557701d2da47ea3a126998d7b3c588dc7fc944439b52fe465'
    assert digest(dependency_archive.read_bytes()) == dependency_pin
    with zipfile.ZipFile(dependency_archive) as immutable:
        original = json.loads(immutable.read('connected_root_exterior_inputs_20261011.json'))
        for name in original['sha256']:
            captured[name] = immutable.read(name)
    assert all(digest(captured[n]) == pin for n, pin in manifest['sha256'].items())
    stem = f'connected_root_exterior_{epoch}'
    pre = ROOT/f'{stem}_preexecution.zip'
    archive(pre, captured)
    work = Path(tempfile.mkdtemp(prefix=stem+'_frozen_'))
    start = time.monotonic()
    try:
        with zipfile.ZipFile(pre) as frozen:
            frozen.extractall(work)
        env = os.environ | {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        command = [PYTHON, '-B', str(work/'connected_root_exterior_mixed_scientist_20261011.py'), '--epoch', epoch]
        process = subprocess.run(command, cwd=work, env=env, capture_output=True)
        additions = {p.name: p.read_bytes() for p in work.iterdir() if p.is_file() and p.name not in captured}
        for name, raw in additions.items():
            if name.startswith(stem):
                with (ROOT/name).open('xb') as out:
                    out.write(raw)
        with (ROOT/f'{stem}_stdout.txt').open('xb') as out:
            out.write(process.stdout)
        with (ROOT/f'{stem}_stderr.txt').open('xb') as out:
            out.write(process.stderr)
        result = {'epoch': epoch, 'command': command, 'python': PYTHON,
                  'natural_returncode': process.returncode, 'natural_exit': True, 'elapsed_seconds': time.monotonic()-start,
                  'captured_sha256': {n: digest(raw) for n, raw in captured.items()},
                  'preexecution_archive_sha256': digest(pre.read_bytes()),
                  'frozen_sources_inputs_unchanged': all((work/n).read_bytes() == raw for n, raw in captured.items()),
                  'immutable_dependency_archive': {'file': dependency_archive.name, 'sha256': dependency_pin},
                  'live_sources_inputs_unchanged': all((ROOT/n).read_bytes() == raw for n, raw in captured.items()),
                  'output_sha256': {n: digest(raw) for n, raw in additions.items()},
                  'stdout_sha256': digest(process.stdout), 'stderr_sha256': digest(process.stderr)}
        raw = (json.dumps(result, indent=2)+'\n').encode()
        with (ROOT/f'{stem}_exit.json').open('xb') as out:
            out.write(raw)
        archive(ROOT/f'{stem}_executed.zip', captured | additions | {f'{stem}_stdout.txt': process.stdout,
            f'{stem}_stderr.txt': process.stderr, f'{stem}_exit.json': raw})
        print(json.dumps(result, indent=2), flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)
    finally:
        shutil.rmtree(work)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    main(parser.parse_args().epoch)

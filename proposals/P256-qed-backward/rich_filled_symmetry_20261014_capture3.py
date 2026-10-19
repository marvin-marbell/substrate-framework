"""Capture actual allowed-grid finite filled reference consumer."""
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from rich_filled_symmetry_20261014_capture import digest

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_filled_symmetry_20261014_actual3'
SOURCES = ['rich_filled_symmetry_20261014_model.py', 'rich_filled_symmetry_20261014_grid_consumer.py', 'rich_filled_symmetry_20261014_grid_input.json', 'rich_filled_symmetry_20261014_calculate.py', 'rich_filled_symmetry_20261014_capture.py', 'rich_filled_symmetry_20261014_capture3.py', 'rich_spinor_rot_20261014_model.py', 'rich_filled_symmetry_20261014_actual1_result.json', 'rich_filled_symmetry_20261014_actual1_exit.json']


def main():
    before = {a: digest(ROOT / a) for a in SOURCES}
    command = [sys.executable, str(ROOT / SOURCES[1])]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start = time.time(); p = subprocess.run(command, cwd=ROOT, env=env, capture_output=True); elapsed = time.time() - start
    (ROOT / (PREFIX + '_stdout.txt')).write_bytes(p.stdout); (ROOT / (PREFIX + '_stderr.txt')).write_bytes(p.stderr)
    names = [PREFIX + '_stdout.txt', PREFIX + '_stderr.txt']
    if p.returncode == 0:
        names.append(PREFIX + '_result.json')
    receipt = {'kind': 'NEW_LITERAL_FINITE_GRID_FILLED_PAIR_CONSUMER', 'command': command, 'cwd': str(ROOT), 'natural_exit': True, 'returncode': p.returncode, 'wall_seconds': elapsed, 'python': sys.version, 'platform': platform.platform(), 'thread_environment': {a: env[a] for a in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}, 'source_before': before, 'source_after': {a: digest(ROOT / a) for a in SOURCES}, 'outputs': {a: digest(ROOT / a) for a in names}, 'reference_reuse': 'actual1 finite reference Esea/N description only; NEW allowed-grid same-H pair calculation, no old CLI replay'}
    name = PREFIX + '_exit.json'; (ROOT / name).write_text(json.dumps(receipt, indent=2) + '\n')
    with zipfile.ZipFile(ROOT / (PREFIX + '_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for a in SOURCES + names + [name]:
            z.write(ROOT / a, arcname=a)
    print(json.dumps(receipt, indent=2)); print(p.stdout.decode()); print(p.stderr.decode()); raise SystemExit(p.returncode)

if __name__ == '__main__':
    main()

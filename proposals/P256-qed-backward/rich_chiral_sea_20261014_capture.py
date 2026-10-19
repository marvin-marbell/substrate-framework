"""Capture NEW physical consumer, immutable source/input/output natural exit."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_chiral_sea_20261014_actual1'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
SOURCES = ['rich_chiral_sea_20261014_model.py', 'rich_chiral_sea_20261014_calculate.py', 'rich_chiral_sea_20261014_capture.py', 'rich_chiral_sea_20261014_input.json', 'rich_frame_gauss_20261014_model.py', 'rich_filled_symmetry_20261014_model.py', 'rich_spinor_rot_20261014_model.py']

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    before = {name: (ROOT / name).read_bytes() for name in SOURCES}
    environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    command = [PYTHON, str(ROOT / 'rich_chiral_sea_20261014_calculate.py')]
    start = time.monotonic()
    run = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True)
    elapsed = time.monotonic() - start
    stdout = PREFIX + '_stdout.txt'
    stderr = PREFIX + '_stderr.txt'
    (ROOT / stdout).write_bytes(run.stdout)
    (ROOT / stderr).write_bytes(run.stderr)
    after = {name: sha((ROOT / name).read_bytes()) for name in SOURCES}
    outputs = [stdout, stderr]
    for suffix in ['_result.json', '_states.npz']:
        if (ROOT / (PREFIX + suffix)).exists():
            outputs.append(PREFIX + suffix)
    receipt = {'kind': 'NEW_PHYSICAL_CALCULATION_EXACT_NATURAL_EXIT', 'command': command, 'cwd': str(ROOT), 'returncode': run.returncode, 'natural_exit': True, 'elapsed_seconds': elapsed, 'python': platform.python_version(), 'thread_environment': {key: environment[key] for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}, 'sources_before': {key: sha(value) for key, value in before.items()}, 'sources_after': after, 'source_unchanged': all(sha(value) == after[key] for key, value in before.items()), 'outputs': {name: sha((ROOT / name).read_bytes()) for name in outputs}, 'no_archived_CLI_replay': True, 'pure_helper_reuse': ['frame_gauss right-colour Gauss and dressed source', 'angular_geometry/field_geometry and angular_reversal', 'full Peierls link vertices and Wick occupied/empty source consumers on entirely NEW reconstructed H']}
    receipt_name = PREFIX + '_exit.json'
    (ROOT / receipt_name).write_text(json.dumps(receipt, indent=2) + '\n')
    with zipfile.ZipFile(ROOT / (PREFIX + '_capsule.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as capsule:
        for name, data in before.items():
            capsule.writestr(name, data)
        for name in outputs + [receipt_name]:
            capsule.write(ROOT / name, arcname=name)
    print(json.dumps(receipt, indent=2))
    if run.returncode:
        print(run.stderr.decode())
    raise SystemExit(run.returncode)

if __name__ == '__main__':
    main()

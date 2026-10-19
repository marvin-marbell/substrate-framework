"""NEW second consumer exact source/input/output capsule, no old CLI replay."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_chiral_sea_20261014_actual2'
SOURCES = ['rich_chiral_sea_20261014_model.py', 'rich_chiral_sea_20261014_calculate.py', 'rich_chiral_sea_20261014_packet.py', 'rich_chiral_sea_20261014_capture2.py', 'rich_chiral_sea_20261014_input.json', 'rich_chiral_sea_20261014_packet_input.json', 'rich_frame_gauss_20261014_model.py', 'rich_filled_symmetry_20261014_model.py', 'rich_spinor_rot_20261014_model.py']

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    before = {name: (ROOT / name).read_bytes() for name in SOURCES}
    environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    command = ['/home/administrator/substrate-framework/.venv/bin/python', str(ROOT / 'rich_chiral_sea_20261014_packet.py')]
    start = time.monotonic()
    run = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True)
    elapsed = time.monotonic() - start
    outputs = []
    for suffix, data in [('_stdout.txt', run.stdout), ('_stderr.txt', run.stderr)]:
        name = PREFIX + suffix
        (ROOT / name).write_bytes(data)
        outputs.append(name)
    if (ROOT / (PREFIX + '_result.json')).exists():
        outputs.append(PREFIX + '_result.json')
    after = {name: sha((ROOT / name).read_bytes()) for name in SOURCES}
    receipt = {'kind': 'NEW_EXACT_PEIERLS_PACKET_AND_CANONICAL_FOCK_C', 'command': command, 'cwd': str(ROOT), 'returncode': run.returncode, 'natural_exit': True, 'elapsed_seconds': elapsed, 'python': platform.python_version(), 'thread_environment': {key: environment[key] for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}, 'sources_before': {key: sha(value) for key, value in before.items()}, 'sources_after': after, 'source_unchanged': all(sha(value) == after[key] for key, value in before.items()), 'outputs': {name: sha((ROOT / name).read_bytes()) for name in outputs}, 'no_old_producer_CLI_replay': True}
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

"""Capture NEW reviewer consumers without replaying any archived entrypoint."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = Path('/home/administrator/substrate-framework/.venv/bin/python')
PREFIX = 'rich_review_wave4_20261014_'

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def capture(label: str) -> None:
    consumer = ROOT / f'{PREFIX}{label}.py'
    inputs = ROOT / f'{PREFIX}{label}_input.json'
    source_before = consumer.read_bytes()
    input_before = inputs.read_bytes()
    start = time.monotonic()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    local_sources_before = {item.name: item.read_bytes() for item in ROOT.glob('*.py')}
    run = subprocess.run([str(PYTHON), str(consumer), str(inputs)], cwd=ROOT, env=env, capture_output=True)
    elapsed = time.monotonic() - start
    stdout = ROOT / f'{PREFIX}{label}_stdout.txt'
    stderr = ROOT / f'{PREFIX}{label}_stderr.txt'
    stdout.write_bytes(run.stdout)
    stderr.write_bytes(run.stderr)
    result = ROOT / f'{PREFIX}{label}_result.json'
    dependencies = json.loads((ROOT / f'{PREFIX}{label}_dependencies.json').read_text()) if run.returncode == 0 else []
    members = [consumer, inputs, Path(__file__), stdout, stderr]
    if result.exists():
        members.append(result)
    members.extend(ROOT / item for item in dependencies)
    members = list(dict.fromkeys(members))
    receipt = {
        'kind': 'NEW_INDEPENDENT_PHYSICAL_CONSUMER_NATURAL_EXIT',
        'argv': [str(PYTHON), str(consumer), str(inputs)], 'cwd': str(ROOT),
        'returncode': run.returncode, 'elapsed_seconds': elapsed,
        'python': str(PYTHON), 'python_version': subprocess.run([str(PYTHON), '--version'], capture_output=True, text=True).stdout.strip(),
        'consumer_sha256': digest(source_before), 'input_sha256': digest(input_before),
        'consumer_unchanged': consumer.read_bytes() == source_before,
        'input_unchanged': inputs.read_bytes() == input_before,
        'members': {item.name: digest(item.read_bytes()) for item in members},
        'dependency_capture': 'Consumer records actual imported local modules; before/after source bytes checked and pre-execution bytes captured. Author CLI never executed.',
        'imported_sources_unchanged': {name: (ROOT / name).read_bytes() == local_sources_before.get(name) for name in dependencies},
    }
    receipt_path = ROOT / f'{PREFIX}{label}_exit.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    with zipfile.ZipFile(ROOT / f'{PREFIX}{label}_capsule.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for item in members + [receipt_path]:
            if item.name in dependencies:
                archive.writestr(item.name, local_sources_before[item.name])
            else:
                archive.write(item, item.name)
    print(json.dumps(receipt))
    raise SystemExit(run.returncode)
if __name__ == '__main__':
    capture(sys.argv[1])

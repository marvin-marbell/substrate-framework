"""Capture new spatial physical calculations, source bindings and natural exits."""
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def capture(cells, hopping, rail_hopping, tag):
    name = f'rich_spatial_20261014_{tag}'
    command = [PYTHON, str(ROOT/'rich_spatial_20261014_model.py'), '--cells', str(cells), '--hopping', str(hopping), '--rail-hopping', str(rail_hopping), '--name', name]
    sources = [ROOT/'rich_spatial_20261014_model.py', ROOT/'rich_spatial_20261014_capture.py', ROOT/'rich_photon_fullq_20261014_model.py', ROOT/'rich_photon_20261014_loop.py']
    source_before = {str(p): digest(p) for p in sources}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = time.time()
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True)
    stdout = ROOT/f'{name}_stdout.txt'; stdout.write_bytes(completed.stdout)
    stderr = ROOT/f'{name}_stderr.txt'; stderr.write_bytes(completed.stderr)
    inputs = ROOT/f'{name}_input.json'
    inputs.write_text(json.dumps({'command': command, 'cells': cells, 'hopping': hopping, 'rail_hopping': rail_hopping, 'units': 'hbar=charge=link_spacing=1; hopping base scale=1', 'environment': {key: env[key] for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}, indent=2)+'\n')
    files = sources+[inputs, stdout, stderr]
    files += [p for p in [ROOT/f'{name}_result.json', ROOT/f'{name}_states.npz'] if p.exists()]
    capsule = ROOT/f'{name}_execution.zip'
    with zipfile.ZipFile(capsule, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
    receipt = {'candidate_id': 'rich-spatial-open-integer-gauss-two-plane-ladder-20261014', 'command': command, 'cwd': str(ROOT), 'python': PYTHON, 'natural_exit': completed.returncode, 'seconds': time.time()-started, 'source_before': source_before, 'source_after': {str(p): digest(p) for p in sources}, 'artifacts': {str(p): digest(p) for p in files+[capsule]}, 'claim_boundary': 'Author NEW calculation receipt; independent scientific/source review pending. No test/build/lint/formatter run.'}
    (ROOT/f'{name}_natural_exit.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(completed.stdout.decode(), flush=True)
    print(json.dumps({'tag': tag, 'natural_exit': completed.returncode, 'seconds': receipt['seconds']}), flush=True)
    if completed.returncode:
        print(completed.stderr.decode(), file=sys.stderr)
        raise SystemExit(completed.returncode)

if __name__ == '__main__':
    selected = sys.argv[1:]
    runs = {'cells1': (1, 1., 1.), 'cells2': (2, 1., 1.), 'cells3': (3, 1., 1.), 'cells3_openrails': (3, 1., 0.), 'cells3_nomotion': (3, 0., 1.)}
    for tag in selected:
        capture(*runs[tag], tag)

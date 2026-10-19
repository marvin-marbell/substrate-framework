"""Immutable input snapshots and natural-exit receipt for the NEW event only."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_quantum_collision_20261014_actual1'
names = ['rich_spin_flux_spatial_20261014_model.py', 'rich_spin_flux_spatial_20261014_3d_model.py', 'rich_quantum_collision_20261014_model.py', 'rich_quantum_collision_20261014_calculate.py', 'rich_quantum_collision_20261014_capture.py', 'rich_quantum_collision_20261014_input.json']
inputs = {name: (ROOT/name).read_bytes() for name in names}
command = [sys.executable, str(ROOT/'rich_quantum_collision_20261014_calculate.py')]
started = time.time()
with (ROOT/(PREFIX+'_stdout.txt')).open('wb') as out, (ROOT/(PREFIX+'_stderr.txt')).open('wb') as err:
    process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), stdout=out, stderr=err)
    code = process.wait()
outputs = [PREFIX+suffix for suffix in ['_result.json', '_states.npz', '_stdout.txt', '_stderr.txt'] if (ROOT/(PREFIX+suffix)).exists()]
receipt = {'command': command, 'cwd': str(ROOT), 'returncode': code, 'natural_exit': True, 'elapsed_seconds': time.time()-started, 'python': sys.version, 'platform': platform.platform(), 'input_SHA256': {name: hashlib.sha256(content).hexdigest() for name, content in inputs.items()}, 'output_SHA256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in outputs}, 'only_pure_predecessor_helpers': True, 'no_predecessor_CLI_or_state_import': True}
name = PREFIX+'_exit.json'
(ROOT/name).write_text(json.dumps(receipt, indent=2))
with zipfile.ZipFile(ROOT/(PREFIX+'_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as capsule:
    for input_name, content in inputs.items():
        capsule.writestr(input_name, content)
    for output_name in outputs+[name]:
        capsule.write(ROOT/output_name, output_name)
print(json.dumps(receipt, indent=2))
raise SystemExit(code)

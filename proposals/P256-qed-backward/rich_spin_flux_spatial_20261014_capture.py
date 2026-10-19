"""Exact preexecution source/input bytes and natural subprocess exit receipt."""
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
PREFIX = 'rich_spin_flux_spatial_20261014_actual2'
source_names = ['rich_spin_flux_spatial_20261014_model.py', 'rich_spin_flux_spatial_20261014_calculate.py', 'rich_spin_flux_spatial_20261014_capture.py', 'rich_spin_flux_spatial_20261014_input.json']
inputs = {name: (ROOT/name).read_bytes() for name in source_names}
command = [sys.executable, str(ROOT/'rich_spin_flux_spatial_20261014_calculate.py')]
environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
started = time.time()
with (ROOT/(PREFIX+'_stdout.txt')).open('wb') as stdout, (ROOT/(PREFIX+'_stderr.txt')).open('wb') as stderr:
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    returncode = process.wait()
outputs = [PREFIX+suffix for suffix in ['_result.json', '_states.npz', '_stdout.txt', '_stderr.txt'] if (ROOT/(PREFIX+suffix)).exists()]
receipt = {'command': command, 'cwd': str(ROOT), 'started_unix': started, 'elapsed_seconds': time.time()-started, 'returncode': returncode, 'natural_exit': True, 'process_id': process.pid, 'python': sys.version, 'platform': platform.platform(), 'runtime_thread_environment': {k: environment[k] for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}, 'input_SHA256': {name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()}, 'output_SHA256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in outputs}, 'no_old_scientific_entrypoint': True}
receipt_name = PREFIX+'_exit.json'
(ROOT/receipt_name).write_text(json.dumps(receipt, indent=2))
with zipfile.ZipFile(ROOT/(PREFIX+'_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as capsule:
    for name, data in inputs.items():
        capsule.writestr(name, data)
    for name in outputs+[receipt_name]:
        capsule.write(ROOT/name, name)
print(json.dumps(receipt, indent=2), flush=True)
raise SystemExit(returncode)

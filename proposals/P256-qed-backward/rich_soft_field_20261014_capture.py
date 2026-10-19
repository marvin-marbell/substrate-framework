"""Natural-exit exact-source/input/output capsule, no archived replay."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_soft_field_20261014_'
inputs = [ROOT/(PREFIX+s) for s in ['model.py', 'calculate.py', 'capture.py', 'input.json']]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = {path.name: sha(path) for path in inputs}
environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
command = [sys.executable, str(ROOT/(PREFIX+'calculate.py'))]
start = time.time()
with (ROOT/(PREFIX+'actual1_stdout.txt')).open('wb') as out, (ROOT/(PREFIX+'actual1_stderr.txt')).open('wb') as err:
    process = subprocess.run(command, cwd=ROOT, env=environment, stdout=out, stderr=err)
outputs = [ROOT/(PREFIX+s) for s in ['actual1_stdout.txt', 'actual1_stderr.txt', 'actual1_result.json', 'actual1_geometry.npz']]
record = {'command': command, 'cwd': str(ROOT), 'exit_code': process.returncode, 'natural_exit': True,
          'elapsed_seconds': time.time()-start, 'python': sys.version, 'thread_env': {k: environment[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
          'sources_before': before, 'sources_after': {path.name: sha(path) for path in inputs},
          'outputs': {path.name: sha(path) for path in outputs if path.exists()}}
exitpath = ROOT/(PREFIX+'actual1_exit.json')
exitpath.write_text(json.dumps(record, indent=2))
with zipfile.ZipFile(ROOT/(PREFIX+'actual1_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as capsule:
    for path in inputs+outputs+[exitpath]:
        if path.exists(): capsule.write(path, path.name)
print(json.dumps(record, indent=2))
sys.exit(process.returncode)

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_spin_flux_spatial_20261014_actual3'
names = ['rich_spin_flux_spatial_20261014_model.py', 'rich_spin_flux_spatial_20261014_vertex.py', 'rich_spin_flux_spatial_20261014_capture3.py', 'rich_spin_flux_spatial_20261014_input.json', 'rich_spin_flux_spatial_20261014_actual2_result.json', 'rich_spin_flux_spatial_20261014_actual2_states.npz']
inputs = {n: (ROOT/n).read_bytes() for n in names}
command = [sys.executable, str(ROOT/'rich_spin_flux_spatial_20261014_vertex.py')]
started = time.time()
with (ROOT/(PREFIX+'_stdout.txt')).open('wb') as out, (ROOT/(PREFIX+'_stderr.txt')).open('wb') as err:
    p = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), stdout=out, stderr=err)
    code = p.wait()
outputs = [PREFIX+s for s in ['_result.json', '_states.npz', '_stdout.txt', '_stderr.txt'] if (ROOT/(PREFIX+s)).exists()]
receipt = {'command': command, 'returncode': code, 'natural_exit': True, 'elapsed_seconds': time.time()-started, 'input_SHA256': {n: hashlib.sha256(b).hexdigest() for n,b in inputs.items()}, 'output_SHA256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in outputs}, 'same_law_state_provenance': 'actual2 exact reference outputs captured as inputs; new actual physical consumer, not predecessor CLI'}
name = PREFIX+'_exit.json'
(ROOT/name).write_text(json.dumps(receipt, indent=2))
with zipfile.ZipFile(ROOT/(PREFIX+'_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
    for n,b in inputs.items():
        z.writestr(n,b)
    for n in outputs+[name]:
        z.write(ROOT/n,n)
print(json.dumps(receipt, indent=2))
raise SystemExit(code)

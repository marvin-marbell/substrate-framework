"""Natural-exit exact capsule for the new spin-pulse scientific consumer."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_induced_spin_20261014_actual2'
inputs = [ROOT/f'rich_induced_spin_20261014_{name}.py' for name in ['model', 'calculate', 'spin_dynamics', 'capture2']]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
before = {p.name: digest(p) for p in inputs}
started = time.time()
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
command = [sys.executable, str(inputs[2])]
p = subprocess.run(command, cwd=ROOT, env=env, capture_output=True)
(ROOT/f'{PREFIX}_stdout.txt').write_bytes(p.stdout)
(ROOT/f'{PREFIX}_stderr.txt').write_bytes(p.stderr)
outputs = [ROOT/f'{PREFIX}_{suffix}' for suffix in ['stdout.txt', 'stderr.txt', 'result.json']]
receipt = {'command': command, 'cwd': str(ROOT), 'python': sys.version, 'executable': sys.executable, 'start_unix': started, 'end_unix': time.time(), 'elapsed_seconds': time.time()-started, 'returncode': p.returncode, 'natural_exit': p.returncode >= 0, 'thread_environment': {k: env[k] for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}, 'source_before_sha256': before, 'source_after_sha256': {p.name: digest(p) for p in inputs}, 'outputs_sha256': {o.name: digest(o) for o in outputs if o.exists()}}
exitpath = ROOT/f'{PREFIX}_exit.json'; exitpath.write_text(json.dumps(receipt, indent=2))
with zipfile.ZipFile(ROOT/f'{PREFIX}_capsule.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for item in [*inputs, *outputs, exitpath]:
        if item.exists(): archive.write(item, item.name)
print(json.dumps(receipt, indent=2))
sys.exit(p.returncode)

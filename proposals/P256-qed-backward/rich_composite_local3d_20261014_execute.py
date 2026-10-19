"""Execute a new physical calculation and bind its natural exit to exact bytes."""
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
PREFIX = 'rich_composite_local3d_20261014'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'actual1'
    name = PREFIX+'_'+tag
    inputs = [ROOT/(PREFIX+'_consumer.py'), ROOT/(PREFIX+'_request.json'),
              ROOT/(PREFIX+'_equations.json'), Path(__file__).resolve(),
              ROOT/'rich_composite_20261014_consumer.py']
    program = inputs[0]
    if tag == 'selection1':
        program = ROOT/(PREFIX+'_selection.py')
        inputs.append(program)
    hashes_before = {p.name: digest(p) for p in inputs}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(program), '--request', str(inputs[1]), '--output', str(ROOT/(name+'_receipt'))]
    start = time.time()
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    duration = time.time()-start
    stdout = ROOT/(name+'_stdout.txt'); stdout.write_text(process.stdout)
    stderr = ROOT/(name+'_stderr.txt'); stderr.write_text(process.stderr)
    outputs = [stdout, stderr] + [p for p in [ROOT/(name+'_receipt.json'), ROOT/(name+'_receipt.npz')] if p.exists()]
    after = {p.name: digest(p) for p in inputs}
    record = dict(command=command, python=sys.executable, natural_exit=process.returncode,
                  runtime_seconds=duration, started_unix=start, sources_inputs_sha256=hashes_before,
                  sources_unchanged=hashes_before == after, outputs_sha256={p.name: digest(p) for p in outputs},
                  environment={k: env[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    exit_path = ROOT/(name+'_exit.json'); exit_path.write_text(json.dumps(record, indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(name+'_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in inputs+outputs+[exit_path]:
            archive.write(path, path.name)
    print(json.dumps(record, indent=2))
    print(process.stdout)
    if process.stderr:
        print(process.stderr, file=sys.stderr)
    return process.returncode

if __name__ == '__main__':
    raise SystemExit(main())

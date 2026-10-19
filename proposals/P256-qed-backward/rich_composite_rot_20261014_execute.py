"""Capture natural scientific exit and exact source/input/output bytes."""
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
PREFIX = 'rich_composite_rot_20261014'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'actual1'
    name = PREFIX+'_'+tag
    inputs = [ROOT/(PREFIX+'_consumer.py'), ROOT/(PREFIX+'_request.json'), ROOT/(PREFIX+'_equations.json'), Path(__file__).resolve()]
    program, request = inputs[0], inputs[1]
    if tag == 'magnetic1':
        program = ROOT/(PREFIX+'_magnetic.py')
        request = ROOT/(PREFIX+'_magnetic_request.json')
        inputs.extend([program, request])
    before = {p.name: digest(p) for p in inputs}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(program), '--request', str(request), '--output', str(ROOT/(name+'_receipt'))]
    start = time.time(); completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    stdout = ROOT/(name+'_stdout.txt'); stdout.write_text(completed.stdout)
    stderr = ROOT/(name+'_stderr.txt'); stderr.write_text(completed.stderr)
    outputs = [stdout, stderr]+[p for p in [ROOT/(name+'_receipt.json'), ROOT/(name+'_receipt.npz')] if p.exists()]
    record = dict(command=command, python=sys.executable, natural_exit=completed.returncode, runtime_seconds=time.time()-start,
                  started_unix=start, sources_inputs_sha256=before, sources_unchanged=before == {p.name: digest(p) for p in inputs},
                  outputs_sha256={p.name: digest(p) for p in outputs}, environment={k: env[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    exit_path = ROOT/(name+'_exit.json'); exit_path.write_text(json.dumps(record, indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(name+'_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in inputs+outputs+[exit_path]:
            archive.write(path, path.name)
    print(json.dumps(record, indent=2)); print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())

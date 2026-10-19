"""Preserve actual child exit, source snapshot and scientific logs."""
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
P = 'rich_dressed_local_carrier_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def main():
    stage = sys.argv[1] if len(sys.argv)>1 else 'prepare'
    config = json.loads((ROOT/(P+'input.json')).read_text())
    selected = list(config['source_files'])
    if stage != 'prepare':
        selected += [P+stage+'.py', P+'interface.json', P+'states.npz']
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in selected}
    with zipfile.ZipFile(ROOT/(P+stage+'_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as capsule:
        for name in selected:
            capsule.write(ROOT/name,name)
    command = [PYTHON, P+stage+'.py']
    start = time.monotonic()
    (ROOT/(P+stage+'_start.json')).write_text(json.dumps(dict(command=command,cwd=str(ROOT),source_SHA256=hashes,pid=os.getpid()),indent=2)+'\n')
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (ROOT/(P+stage+'_stdout.txt')).open('w') as stdout, (ROOT/(P+stage+'_stderr.txt')).open('w') as stderr:
        completed = subprocess.run(command,cwd=ROOT,env=environment,stdout=stdout,stderr=stderr)
    result = dict(command=command,natural_exit=True,returncode=completed.returncode,elapsed_seconds=time.monotonic()-start,
        source_SHA256=hashes,inputs_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items()))
    (ROOT/(P+stage+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()

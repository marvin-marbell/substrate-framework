"""Capture actual science process source/input hashes, streams, and natural exit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
from typing import Any

ROOT=Path(__file__).resolve().parent
P='rich_unitary_rotor_photon_20261016_'


def main():
    files=sorted(ROOT.glob(P+'*.py'))+[ROOT/(P+'interface.json')]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    command=['/home/administrator/substrate-framework/.venv/bin/python',str(ROOT/(P+'run.py'))]
    before:dict[str,Any]=dict(command=command,cwd=str(ROOT),source_sha256=hashes,environment={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},plan='issue222-rich-backward-candidates-20261014@b048dbe91b0250f9525e9a07b6d76b95a8080f2f57c071873cad15f8a8fd0e4f')
    (ROOT/(P+'execution_input.json')).write_text(json.dumps(before,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(P+'source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/(P+'execution_input.json')]:z.write(p,p.name)
    env=os.environ.copy();env.update(before['environment']);start=time.time()
    with (ROOT/(P+'stdout.txt')).open('w') as out,(ROOT/(P+'stderr.txt')).open('w') as err:
        child=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
    result=dict(natural_exit=child.returncode,elapsed_seconds=time.time()-start,command=command,source_sha256=hashes)
    (ROOT/(P+'exit.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    raise SystemExit(child.returncode)


if __name__=='__main__':main()

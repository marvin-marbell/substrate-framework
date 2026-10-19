"""Natural-exit/source-byte capture for this new scientific consumer only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='rich_collective_photon_response_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    stage=sys.argv[1]
    own_stage=stage in ('stress','coldrefine','analyze','language','deliver')
    command=[PY,str(ROOT/(PREFIX+stage+'.py'))] if own_stage else [PY,str(ROOT/(PREFIX+'run.py')),stage]
    own=list(ROOT.glob(PREFIX+'*.py'))
    frozen=[ROOT/('rich_unitary_rotor_photon_20261016_'+s) for s in ('thermal.py','bridge.py')]
    sources=own+frozen;hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with zipfile.ZipFile(ROOT/(PREFIX+stage+'_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.name)
    input_data=dict(command=command,cwd=str(ROOT),source_hashes=hashes,environment={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},stage=stage)
    (ROOT/(PREFIX+stage+'_input.json')).write_text(json.dumps(input_data,indent=2)+'\n')
    env=os.environ.copy();env.update(input_data['environment']);start=time.monotonic()
    with (ROOT/(PREFIX+stage+'_stdout.txt')).open('w') as out,(ROOT/(PREFIX+stage+'_stderr.txt')).open('w') as err:
        process=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
    result=dict(natural_exit=process.returncode,elapsed_seconds=time.monotonic()-start,command=command,
                input_sha256=hashlib.sha256((ROOT/(PREFIX+stage+'_input.json')).read_bytes()).hexdigest(),
                source_capsule_sha256=hashlib.sha256((ROOT/(PREFIX+stage+'_source_capsule.zip')).read_bytes()).hexdigest())
    (ROOT/(PREFIX+stage+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    raise SystemExit(process.returncode)


if __name__=='__main__':main()

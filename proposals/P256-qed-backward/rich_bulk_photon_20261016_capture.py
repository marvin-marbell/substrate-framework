"""Freeze exact sources/input before execution; preserve science natural exit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_bulk_photon_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    label=sys.argv[1] if len(sys.argv)>1 else 'attempt1'
    sources=sorted(list(ROOT.glob(P+'*.py'))+list(ROOT.glob(P+'*input.json'))+[ROOT/(P+'interface.json')])
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with zipfile.ZipFile(ROOT/(P+label+'_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.name)
    command=[PY,str(ROOT/(P+'run.py'))]
    thread_environment={k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}
    before=dict(command=command,source_sha256=hashes,start=time.time(),cwd=str(ROOT),environment=thread_environment)
    if len(sys.argv)>2:command.append(sys.argv[2])
    (ROOT/(P+label+'_input.json')).write_text(json.dumps(before,indent=2)+'\n')
    env=os.environ.copy();env.update(thread_environment)
    with (ROOT/(P+label+'_stdout.txt')).open('w') as out,(ROOT/(P+label+'_stderr.txt')).open('w') as err:
        proc=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
    after=dict(before,natural_exit=proc.returncode,end=time.time(),sources_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in hashes.items()))
    (ROOT/(P+label+'_exit.json')).write_text(json.dumps(after,indent=2)+'\n')
    print(json.dumps(after),flush=True)
    return proc.returncode


if __name__=='__main__':raise SystemExit(main())

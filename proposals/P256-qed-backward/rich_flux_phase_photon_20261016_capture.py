"""Capture exact new input, imported pure sources and scientific natural exit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_flux_phase_photon_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    sources=sorted(list(ROOT.glob(P+'*.py'))+[ROOT/(P+'input.json'),ROOT/(P+'interface.json'),ROOT/'rich_bulk_photon_20261016_state.py',ROOT/'rich_bulk_photon_20261016_run.py'])
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with zipfile.ZipFile(ROOT/(P+'source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as capsule:
        for path in sources:capsule.write(path,path.name)
    env=os.environ.copy();threads={k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')};env.update(threads)
    command=[PY,str(ROOT/(P+'run.py'))]
    before=dict(command=command,cwd=str(ROOT),environment=threads,source_sha256=hashes,start=time.time())
    (ROOT/(P+'execution_input.json')).write_text(json.dumps(before,indent=2)+'\n')
    with (ROOT/(P+'stdout.txt')).open('w') as out,(ROOT/(P+'stderr.txt')).open('w') as err:
        child=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
    result=dict(before,natural_exit=child.returncode,end=time.time(),sources_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in hashes.items()))
    (ROOT/(P+'exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return child.returncode


if __name__=='__main__':raise SystemExit(main())

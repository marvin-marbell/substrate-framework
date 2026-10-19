"""Capture exact inputs/dependencies and natural scientific subprocess exit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_pi_flux_dynamics_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    files=sorted(ROOT.glob(P+'*.py'))+[ROOT/(P+n+'.json') for n in ('input','interface','equations')]
    files += [ROOT/'rich_bulk_photon_20261016_state.py',ROOT/'rich_flux_phase_photon_20261016_state.py']
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    archive=ROOT/(P+'source_capsule.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,path.name)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    command=[PY,str(ROOT/(P+'run.py'))]
    request=dict(command=command,cwd=str(ROOT),thread_environment={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')},source_sha256=hashes,source_capsule_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (ROOT/(P+'execution_input.json')).write_text(json.dumps(request,indent=2)+'\n')
    start=time.time()
    with (ROOT/(P+'stdout.txt')).open('w') as out,(ROOT/(P+'stderr.txt')).open('w') as err:
        child=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
    result=dict(natural_science_exit=child.returncode,elapsed_seconds=time.time()-start,source_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name] for p in files),source_capsule_sha256=request['source_capsule_sha256'])
    (ROOT/(P+'exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    raise SystemExit(child.returncode)


if __name__=='__main__':main()

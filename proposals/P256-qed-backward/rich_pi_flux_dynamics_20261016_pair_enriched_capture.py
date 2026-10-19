"""Exact source/input capsule and scientific exit for material enriched attempt."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_pi_flux_dynamics_20261016_pair_enriched_'


def main():
    previous=ROOT/(P+'exit.json')
    if previous.exists() and json.loads(previous.read_text())['natural_science_exit']!=0:
        failed=ROOT/(P+'failed_compile_capsule.zip')
        if failed.exists():raise RuntimeError('Preserve another failure under a distinct name before retrying')
        with zipfile.ZipFile(failed,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ('source_capsule.zip','execution_input.json','exit.json','stdout.txt','stderr.txt'):
                path=ROOT/(P+name);z.write(path,path.name)
    files=sorted(ROOT.glob('rich_pi_flux_dynamics_20261016_*.py'))+[ROOT/'rich_bulk_photon_20261016_state.py',ROOT/'rich_flux_phase_photon_20261016_state.py',ROOT/'rich_pi_flux_dynamics_20261016_input.json',ROOT/'rich_pi_flux_dynamics_20261016_equations.json',ROOT/(P+'interface.json')]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    archive=ROOT/(P+'source_capsule.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
    command=['/home/administrator/substrate-framework/.venv/bin/python',str(ROOT/(P+'run.py'))]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    request=dict(command=command,cwd=str(ROOT),source_sha256=hashes,source_capsule_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),threads=1,method='76-dimensional coherent delta.8/delta1.2 full64 pair-state family, same H',seeds=[9101671,9101687],samples_per_chain=768,burn=18000,thin=160,epsilon=.35,T=1.5,nt=151,gram_cut=1e-7)
    (ROOT/(P+'execution_input.json')).write_text(json.dumps(request,indent=2)+'\n');start=time.time()
    with (ROOT/(P+'stdout.txt')).open('w') as out,(ROOT/(P+'stderr.txt')).open('w') as err:
        child=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
    result=dict(natural_science_exit=child.returncode,elapsed_seconds=time.time()-start,source_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name] for p in files),source_capsule_sha256=request['source_capsule_sha256'])
    (ROOT/(P+'exit.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    raise SystemExit(child.returncode)


if __name__=='__main__':main()

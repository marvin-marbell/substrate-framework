"""Preserve second material tensor ansatz provenance, raw outputs and natural exit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_periodic_tensor_20261015_'
if __name__=='__main__':
    selected=[P+s for s in ['model.py','run.py','loopgas.py','loopcapture.py','input.json','loopgas_input.json']]+['rich_periodic_photon_20261015_model.py']
    hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in selected}
    with zipfile.ZipFile(ROOT/(P+'loopgas_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for s in selected:z.write(ROOT/s,s)
    # Preserve the first construction's response weights before the second writes.
    with zipfile.ZipFile(ROOT/(P+'first_ansatz_outputs.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for s in [P+'q100_response.npz',P+'q110_response.npz',P+'result.json',P+'state_checkpoint.json']:
            if (ROOT/s).exists():z.write(ROOT/s,s)
    cmd=[str(ROOT/(P+'env/bin/python')),str(ROOT/(P+'loopgas.py'))]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.time();(ROOT/(P+'loopgas_start.json')).write_text(json.dumps({'command':cmd,'source_SHA256':hashes,'start_unix':start},indent=2))
    with (ROOT/(P+'loopgas_stdout.txt')).open('w') as out,(ROOT/(P+'loopgas_stderr.txt')).open('w') as err:
        r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=out,stderr=err)
    result={'actual_science_returncode':r.returncode,'natural_exit':True,'elapsed_seconds':time.time()-start,'source_unchanged':all(hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h for s,h in hashes.items())}
    (ROOT/(P+'loopgas_exit.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    raise SystemExit(r.returncode)

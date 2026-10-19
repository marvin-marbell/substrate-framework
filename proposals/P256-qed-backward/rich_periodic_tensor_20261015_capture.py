"""Run actual new producer with natural exit, hashes, stdout/stderr and capsule."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_periodic_tensor_20261015_'
if __name__=='__main__':
    tag=sys.argv[1] if len(sys.argv)>1 else 'attempt1'
    selected=[P+s for s in ['model.py','run.py','capture.py','input.json']]+['rich_periodic_photon_20261015_model.py']
    hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in selected}
    with zipfile.ZipFile(ROOT/(P+tag+'_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for s in selected:z.write(ROOT/s,s)
    cmd=[str(ROOT/(P+'env/bin/python')),str(ROOT/(P+'run.py'))]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.time()
    (ROOT/(P+tag+'_start.json')).write_text(json.dumps({'command':cmd,'source_SHA256':hashes,'start_unix':start,'environment':{k:env[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}},indent=2))
    with (ROOT/(P+tag+'_stdout.txt')).open('w') as out,(ROOT/(P+tag+'_stderr.txt')).open('w') as err:
        r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=out,stderr=err)
    result={'actual_science_returncode':r.returncode,'natural_exit':True,'elapsed_seconds':time.time()-start,'source_unchanged':all(hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h for s,h in hashes.items())}
    (ROOT/(P+tag+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    sys.exit(r.returncode)

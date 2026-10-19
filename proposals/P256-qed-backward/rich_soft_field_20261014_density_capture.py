import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

root=Path(__file__).resolve().parent
prefix='rich_soft_field_20261014_'
names=['model.py','calculate.py','density.py','density_calculate.py','density_capture.py','density_input.json']
inputs=[root/(prefix+name) for name in names]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={p.name:sha(p) for p in inputs}
command=[sys.executable,str(root/(prefix+'density_calculate.py'))]
environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
start=time.time()
with (root/(prefix+'density_actual1_stdout.txt')).open('wb') as out,(root/(prefix+'density_actual1_stderr.txt')).open('wb') as err:
    process=subprocess.run(command,cwd=root,env=environment,stdout=out,stderr=err)
outputs=[root/(prefix+'density_actual1_'+name) for name in ['stdout.txt','stderr.txt','result.json']]
record={'command':command,'natural_exit':True,'exit_code':process.returncode,'elapsed_seconds':time.time()-start,
        'sources_before':before,'sources_after':{p.name:sha(p) for p in inputs},'outputs':{p.name:sha(p) for p in outputs if p.exists()},
        'python':sys.version,'thread_env':{k:environment[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}}
exitpath=root/(prefix+'density_actual1_exit.json');exitpath.write_text(json.dumps(record,indent=2))
with zipfile.ZipFile(root/(prefix+'density_actual1_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for p in inputs+outputs+[exitpath]:
        if p.exists():z.write(p,p.name)
print(json.dumps(record,indent=2))
sys.exit(process.returncode)

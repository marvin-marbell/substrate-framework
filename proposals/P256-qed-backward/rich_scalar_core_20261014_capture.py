"""Exact source/input/output binding for a NEW natural-exit scalar consumer."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='rich_scalar_core_20261014'
OUTPUT=PREFIX+('_'+sys.argv[1] if len(sys.argv)>1 else '')
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
files=[ROOT/(PREFIX+s) for s in ('_model.py','_calculate.py','_capture.py','_input.json')]
files += [ROOT/'rich_frame_holonomy_20261014_model.py',ROOT/'rich_frame_gauss_20261014_model.py']
before={x.name:digest(x) for x in files}
cmd=[sys.executable,str(ROOT/(PREFIX+'_calculate.py')),str(ROOT/(PREFIX+'_input.json')),str(ROOT/(OUTPUT+'_result.json'))]
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
start=time.time(); run=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True); end=time.time()
(ROOT/(OUTPUT+'_stdout.txt')).write_bytes(run.stdout)
(ROOT/(OUTPUT+'_stderr.txt')).write_bytes(run.stderr)
after={x.name:digest(x) for x in files}
outputs=[ROOT/(OUTPUT+s) for s in ('_result.json','_result.npz','_stdout.txt','_stderr.txt') if (ROOT/(OUTPUT+s)).exists()]
receipt={'kind':'NEW_QUANTUM_PHYSICAL_CONSUMER_NOT_ARCHIVED_CLI_REPLAY','command':cmd,'cwd':str(ROOT),'python':platform.python_version(),'runtime_executable':sys.executable,'thread_environment':{x:env[x] for x in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},'start_unix':start,'end_unix':end,'elapsed_seconds':end-start,'returncode':run.returncode,'natural_exit':True,'source_input_before':before,'source_input_after':after,'unchanged':before==after,'output_hashes':{x.name:digest(x) for x in outputs},'old_producer_replay':False,'pure_helper_reuse':'raw_frame/paths and transitive angular/cg algebra only; old scientific bytes immutable'}
exitpath=ROOT/(OUTPUT+'_exit.json'); exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
with zipfile.ZipFile(ROOT/(OUTPUT+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for x in files+outputs+[exitpath]:
        z.write(x,x.name)
print(json.dumps(receipt)); print(run.stdout.decode()); print(run.stderr.decode(),file=sys.stderr)
raise SystemExit(run.returncode)

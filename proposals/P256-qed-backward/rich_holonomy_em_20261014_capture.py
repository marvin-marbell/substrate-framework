"""Exact source/input/output capsule for the NEW physical calculation."""
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
PREFIX='rich_holonomy_em_20261014'
LABEL=sys.argv[1] if len(sys.argv)>1 else 'actual1'
INPUT=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/(PREFIX+'_input.json')
OUTPUT=ROOT/(PREFIX+'_'+LABEL)
files=[ROOT/(PREFIX+s) for s in ('_model.py','_calculate.py','_capture.py')]+[INPUT,ROOT/'rich_frame_holonomy_20261014_model.py',ROOT/'rich_frame_gauss_20261014_model.py']
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
before={x.name:digest(x) for x in files}
command=[sys.executable,str(ROOT/(PREFIX+'_calculate.py')),str(INPUT),str(OUTPUT)]
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
start=time.time()
run=subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
end=time.time()
stdout=Path(str(OUTPUT)+'_stdout.txt');stderr=Path(str(OUTPUT)+'_stderr.txt')
stdout.write_bytes(run.stdout);stderr.write_bytes(run.stderr)
after={x.name:digest(x) for x in files}
outputs=[Path(str(OUTPUT)+s) for s in ('_result.json','_states.npz','_stdout.txt','_stderr.txt') if Path(str(OUTPUT)+s).exists()]
receipt={'command':command,'cwd':str(ROOT),'runtime_executable':sys.executable,'python_version':platform.python_version(),'thread_environment':{k:env[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},'start_unix':start,'end_unix':end,'elapsed_seconds':end-start,'returncode':run.returncode,'natural_exit':True,'source_input_before':before,'source_input_after':after,'source_input_unchanged':before==after,'output_hashes':{x.name:digest(x) for x in outputs},'old_producer_replay':False,'dependencies':'Pure raw_frame/canonical_c encoder plus angular/cg helpers only; no supplied oscillator called.'}
exitpath=Path(str(OUTPUT)+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
with zipfile.ZipFile(str(OUTPUT)+'_capsule.zip','w',zipfile.ZIP_DEFLATED) as capsule:
    for path in files+outputs+[exitpath]:
        capsule.write(path,path.name)
print(json.dumps(receipt));print(run.stdout.decode());print(run.stderr.decode(),file=sys.stderr)
raise SystemExit(run.returncode)

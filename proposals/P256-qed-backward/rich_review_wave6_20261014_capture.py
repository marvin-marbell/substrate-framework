"""Run only NEW review consumers; bind complete local pure-helper sources and bytes."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'

def execute(suffix):
    prefix='rich_review_wave6_20261014_'+suffix
    entry=ROOT/(prefix+'.py'); input_path=ROOT/(prefix+'_input.json')
    files={entry,input_path,Path(__file__).resolve()}
    pending=[entry]
    while pending:
        path=pending.pop()
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.ImportFrom) and node.module: names=[node.module]
            elif isinstance(node,ast.Import):names=[x.name for x in node.names]
            for name in names:
                candidate=ROOT/(name+'.py')
                if candidate.exists() and candidate not in files:
                    files.add(candidate);pending.append(candidate)
    sources={p.name:p.read_bytes() for p in files}
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    start=time.time()
    completed=subprocess.run([PYTHON,str(entry)],cwd=ROOT,env=env,capture_output=True)
    duration=time.time()-start
    stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt')
    stdout.write_bytes(completed.stdout);stderr.write_bytes(completed.stderr)
    result=ROOT/(prefix+'_result.json')
    outputs=[stdout,stderr]+([result] if result.exists() else [])
    digest=lambda b:hashlib.sha256(b).hexdigest()
    receipt={'entry':str(entry),'python':PYTHON,'cwd':str(ROOT),'natural_exit':completed.returncode,'duration_seconds':duration,
             'environment':{k:env[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
             'sources_and_inputs_sha256':{name:digest(data) for name,data in sorted(sources.items())},
             'outputs_sha256':{p.name:digest(p.read_bytes()) for p in outputs},
             'sources_unchanged_during_run':all((ROOT/name).read_bytes()==data for name,data in sources.items())}
    exitpath=ROOT/(prefix+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(sources.items()):z.writestr(name,data)
        for p in outputs+[exitpath]:z.write(p,p.name)
    print(json.dumps(receipt,indent=2))
    if completed.returncode: print(completed.stderr.decode());sys.exit(completed.returncode)

if __name__=='__main__':execute(sys.argv[1])

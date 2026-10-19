"""Exact natural capture; persist source/input recipe BEFORE scientific execution."""
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
prefix=sys.argv[1];inp=ROOT/(prefix+'_input.json');request=json.loads(inp.read_text());result=ROOT/(prefix+'_result.json');producer=ROOT/request['consumer']
paths={producer,Path(__file__).resolve(),inp}|{ROOT/request[key] for key in ['state_json','state_npz','native_body_cache','event_json','event_npz'] if key in request};pending=[producer]
while pending:
    path=pending.pop()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node,ast.ImportFrom) and node.module:
            helper=ROOT/(node.module+'.py')
            if helper.exists() and helper not in paths:paths.add(helper);pending.append(helper)
hash_paths=lambda:{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
before=hash_paths();command=[sys.executable,str(producer),str(inp),str(result)]
startpath=ROOT/(prefix+'_start.json');startpath.write_text(json.dumps(dict(command=command,source_input_sha256_before=before,meaning='Recipe binding before execution, NOT natural exit or result'),indent=2)+'\n')
start=time.time();environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (ROOT/(prefix+'_stdout.txt')).open('w') as stdout,(ROOT/(prefix+'_stderr.txt')).open('w') as stderr:
    process=subprocess.run(command,cwd=ROOT,env=environment,stdout=stdout,stderr=stderr)
after=hash_paths()
outputs=[ROOT/(prefix+suffix) for suffix in ['_result.json','_result.npz','_stdout.txt','_stderr.txt'] if (ROOT/(prefix+suffix)).exists()]
receipt=dict(command=command,natural_exit=process.returncode,elapsed_seconds=time.time()-start,source_input_sha256_before=before,source_input_sha256_after=after,immutable_during_run=before==after,output_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs})
exitpath=ROOT/(prefix+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(paths)+outputs+[startpath,exitpath]:archive.write(path,path.name)
print(json.dumps(receipt),flush=True)
sys.exit(process.returncode)

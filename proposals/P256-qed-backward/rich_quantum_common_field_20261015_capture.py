"""Natural-exit capture of NEW consumer and recursively pinned local pure helpers."""
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
prefix=sys.argv[1];inp=ROOT/(prefix+'_input.json');result=ROOT/(prefix+'_result.json')
producer=ROOT/'rich_quantum_common_field_20261015_calculate.py'
paths={producer,Path(__file__).resolve(),inp};pending=[producer]
while pending:
    path=pending.pop()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node,ast.ImportFrom) and node.module:
            helper=ROOT/(node.module+'.py')
            if helper.exists() and helper not in paths:paths.add(helper);pending.append(helper)
before={str(path.name):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
start=time.time();environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (ROOT/(prefix+'_stdout.txt')).open('w') as stdout,(ROOT/(prefix+'_stderr.txt')).open('w') as stderr:
    process=subprocess.run([sys.executable,str(producer),str(inp),str(result)],cwd=ROOT,env=environment,stdout=stdout,stderr=stderr)
after={str(path.name):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
outputs=[ROOT/(prefix+suffix) for suffix in ['_result.json','_result.npz','_stdout.txt','_stderr.txt'] if (ROOT/(prefix+suffix)).exists()]
receipt=dict(command=[sys.executable,str(producer),str(inp),str(result)],natural_exit=process.returncode,elapsed_seconds=time.time()-start,source_input_sha256_before=before,source_input_sha256_after=after,immutable_during_run=before==after,output_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs})
exitpath=ROOT/(prefix+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(paths)+outputs+[exitpath]:archive.write(path,path.name)
print(json.dumps(receipt),flush=True)
sys.exit(process.returncode)

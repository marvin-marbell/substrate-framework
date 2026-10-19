"""Exact NEW Wave12 consumers only; no earlier source/result retagging."""
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
    prefix='rich_review_wave12_20261015_'+suffix
    entry=ROOT/(prefix+'.py');inputs=ROOT/(prefix+'_input.json');parameters=json.loads(inputs.read_text())
    paths={entry,inputs,Path(__file__).resolve(),*(ROOT/name for name in parameters.get('captured_inputs',[]))};pending=[entry]
    while pending:
        path=pending.pop()
        for node in ast.walk(ast.parse(path.read_text())):
            modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else [alias.name for alias in node.names] if isinstance(node,ast.Import) else [])
            for module in modules:
                child=ROOT/(module+'.py')
                if child.is_file() and child not in paths:
                    paths.add(child);pending.append(child)
    source={path.name:path.read_bytes() for path in paths}
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    command=[PYTHON,str(entry)];start=time.monotonic();done=subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
    stdout,stderr=[ROOT/(prefix+ending) for ending in ('_stdout.txt','_stderr.txt')]
    stdout.write_bytes(done.stdout);stderr.write_bytes(done.stderr)
    outputs=[stdout,stderr]+[path for ending in ('_result.json','_states.npz') if (path:=ROOT/(prefix+ending)).is_file()]
    digest=lambda value:hashlib.sha256(value).hexdigest()
    receipt=dict(kind='NEW_WAVE12_PHYSICAL_CONSUMER_EXACT_BINDING_NOT_OLD_REPLAY',command=command,cwd=str(ROOT),natural_exit=True,returncode=done.returncode,elapsed_seconds=time.monotonic()-start,
                 environment={key:env[key] for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE')},
                 sources_and_inputs_sha256={name:digest(value) for name,value in sorted(source.items())},outputs_sha256={path.name:digest(path.read_bytes()) for path in outputs},
                 sources_unchanged_during_run=all((ROOT/name).read_bytes()==value for name,value in source.items()))
    exitpath=ROOT/(prefix+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as capsule:
        for name,value in sorted(source.items()):capsule.writestr(name,value)
        for path in outputs+[exitpath]:capsule.write(path,path.name)
    print(json.dumps(receipt,indent=2))
    if done.returncode:
        print(done.stderr.decode());raise SystemExit(done.returncode)


if __name__=='__main__':
    execute(sys.argv[1])

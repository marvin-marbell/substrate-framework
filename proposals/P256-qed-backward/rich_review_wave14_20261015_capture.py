"""Exact fresh review source/input capsules, natural versus signal exit explicit."""
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


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def closure(path,seen):
    if path in seen:return
    seen.add(path)
    for node in ast.walk(ast.parse(path.read_text())):
        names=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
        for name in names:
            candidate=ROOT/(str(name).split('.')[0]+'.py')
            if candidate.is_file():closure(candidate,seen)


def main():
    inp=ROOT/sys.argv[1];prefix=ROOT/sys.argv[2];request=json.loads(inp.read_text());consumer=ROOT/request.get('consumer','rich_review_wave14_20261015_consumer.py');sources=set();closure(consumer,sources);sources|={Path(__file__).resolve(),inp}
    for key in ['owner_input','owner_json','owner_npz']:
        if key in request:sources.add(ROOT/request[key])
    before={p.name:sha(p) for p in sorted(sources)}
    # Capture exact pre-run bytes; active author changes are detected, never retroactively substituted.
    capsule=Path(str(prefix)+'_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sources):z.write(p,p.name)
    command=[PYTHON,str(consumer),str(inp),str(prefix)+'_result.json'];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');start=time.monotonic()
    with open(str(prefix)+'_stdout.txt','wb') as stdout,open(str(prefix)+'_stderr.txt','wb') as stderr:done=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    after={p.name:sha(p) for p in sorted(sources)};outputs=[Path(str(prefix)+s) for s in ['_result.json','_result.npz','_stdout.txt','_stderr.txt']];outputs=[p for p in outputs if p.exists()]
    receipt=dict(command=command,returncode=done.returncode,natural_exit=done.returncode>=0,success=done.returncode==0,signal=-done.returncode if done.returncode<0 else None,elapsed_seconds=time.monotonic()-start,source_and_input_before=before,source_and_input_after=after,source_unchanged=before==after,outputs={p.name:sha(p) for p in outputs})
    exitpath=Path(str(prefix)+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(capsule,'a',zipfile.ZIP_DEFLATED) as z:
        for p in outputs+[exitpath]:z.write(p,p.name)
    print(json.dumps(dict(returncode=done.returncode,natural_exit=receipt['natural_exit'],source_unchanged=before==after,capsule_sha256=sha(capsule))))


if __name__=='__main__':main()

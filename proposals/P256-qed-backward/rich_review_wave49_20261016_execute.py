"""Run independent consumers and preserve immutable inputs, source bytes, natural exits."""
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
P='rich_review_wave49_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def dependencies(path,found):
    if path in found:return
    found.add(path)
    for node in ast.walk(ast.parse(path.read_text())):
        names=[]
        if isinstance(node,ast.Import):names=[n.name for n in node.names]
        if isinstance(node,ast.ImportFrom) and node.module:names=[node.module]
        for name in names:
            candidate=ROOT/(name+'.py')
            if candidate.is_file():dependencies(candidate,found)


def main():
    case=sys.argv[1];attempt=sys.argv[2];own=ROOT/(P+case+'.py');tag=P+attempt
    deps=set();dependencies(own,deps);deps.add(Path(__file__).resolve())
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(deps)}
    with zipfile.ZipFile(ROOT/(tag+'_source_capsule.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(deps):z.write(p,p.name)
    cmd=[PY,str(own),*sys.argv[3:]]
    inputs=dict(plan_id='issue222-rich-backward-candidates-20261014',captured_revision='6facdc39fc29c4b59769e6335b1c31bc95f322e3bf55184de7505d83d9008b08',template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',work_item='independent-review',command=cmd,source_sha256=hashes)
    inputs['capture_epoch']='zero_source_repair2'
    inputs['continuation_binding']=P+'continuation_binding.json'
    (ROOT/(tag+'_input.json')).write_text(json.dumps(inputs,indent=2)+'\n')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',REVIEW49_INPUT_RECEIPT=tag+'_input.json')
    start=time.time();child=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,env=env)
    (ROOT/(tag+'_stdout.txt')).write_text(child.stdout);(ROOT/(tag+'_stderr.txt')).write_text(child.stderr)
    result=dict(natural_exit=child.returncode,elapsed_seconds=time.time()-start,sources_unchanged={p.name:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name] for p in sorted(deps)})
    (ROOT/(tag+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),child.stdout,child.stderr[-2500:],flush=True);sys.exit(child.returncode)

if __name__=='__main__':main()

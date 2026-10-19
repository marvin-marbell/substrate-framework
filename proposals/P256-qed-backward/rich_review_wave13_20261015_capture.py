"""Natural-exit execution with recursive exact local source closure and owner audit."""
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


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def source_closure(path,seen):
    if path in seen:return
    seen.add(path)
    for node in ast.walk(ast.parse(path.read_text())):
        names=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
        for name in names:
            candidate=ROOT/(str(name).split('.')[0]+'.py')
            if candidate.is_file():source_closure(candidate,seen)


def main():
    input_path=ROOT/sys.argv[1];prefix=ROOT/sys.argv[2];p=json.loads(input_path.read_text());consumer=ROOT/'rich_review_wave13_20261015_consumer.py';sources=set();source_closure(consumer,sources);sources.add(Path(__file__).resolve());sources.add(input_path);sources.add(ROOT/p['owner_input'])
    owner_exit=ROOT/p['owner_exit'];owner=json.loads(owner_exit.read_text());audit={}
    with zipfile.ZipFile(ROOT/p['owner_capsule']) as z:
        members={Path(name).name:name for name in z.namelist()}
        for name,expected in owner.get('source_before',owner.get('source_and_input_before',{})).items():
            live=ROOT/name;data=z.read(members[name]) if name in members else None
            audit[name]=dict(expected=expected,live=sha(live),capsule=hashlib.sha256(data).hexdigest() if data is not None else None)
    before={x.name:sha(x) for x in sorted(sources)};command=[PYTHON,str(consumer),str(input_path),str(prefix)+'_result.json'];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.monotonic()
    with open(str(prefix)+'_stdout.txt','wb') as stdout,open(str(prefix)+'_stderr.txt','wb') as stderr:
        done=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    after={x.name:sha(x) for x in sorted(sources)};outputs=[Path(str(prefix)+suffix) for suffix in ['_result.json','_result.npz','_stdout.txt','_stderr.txt']];outputs=[x for x in outputs if x.is_file()]
    record=dict(command=command,returncode=done.returncode,natural_exit=True,elapsed_seconds=time.monotonic()-start,source_before=before,source_after=after,source_unchanged=before==after,owner_exit_sha256=sha(owner_exit),owner_capsule_sha256=sha(ROOT/p['owner_capsule']),owner_source_audit=audit,owner_exact=all(x['expected']==x['live']==x['capsule'] for x in audit.values()),outputs={x.name:sha(x) for x in outputs})
    exit_path=Path(str(prefix)+'_exit.json');exit_path.write_text(json.dumps(record,indent=2)+'\n')
    with zipfile.ZipFile(str(prefix)+'_capsule.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(sources)+outputs+[exit_path,owner_exit]:z.write(path,path.name)
    print(json.dumps(dict(returncode=done.returncode,natural_exit=True,seconds=record['elapsed_seconds'],stable=before==after,owner_exact=record['owner_exact'],capsule_SHA256=sha(Path(str(prefix)+'_capsule.zip')))))


if __name__=='__main__':main()

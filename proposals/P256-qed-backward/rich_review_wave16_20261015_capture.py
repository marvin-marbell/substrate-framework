"""Owned natural execution, no old producer replay; exact recursive sources/case/states."""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def closure(path,seen):
    if path in seen:return
    seen.add(path)
    for node in ast.walk(ast.parse(path.read_text())):
        names=[node.module] if isinstance(node,ast.ImportFrom) else [x.name for x in node.names] if isinstance(node,ast.Import) else []
        for name in names:
            target=ROOT/(str(name).split('.')[0]+'.py')
            if target.is_file():closure(target,seen)


def limit():
    maximum=3800*1024**2;resource.setrlimit(resource.RLIMIT_AS,(maximum,maximum))


def main():
    prefix=sys.argv[1];inp=ROOT/(prefix+'_input.json');p=json.loads(inp.read_text());consumer=ROOT/'rich_review_wave16_20261015_consumer.py';sources=set();closure(consumer,sources);sources.update([Path(__file__).resolve(),inp,ROOT/p['owner_input']]);before={path.name:sha(path) for path in sorted(sources)}
    owner_exit=ROOT/p['owner_exit'];owner=json.loads(owner_exit.read_text());audit={}
    with zipfile.ZipFile(ROOT/p['owner_capsule']) as z:
        members={Path(n).name:n for n in z.namelist()}
        for name,entry in owner['files'].items():
            expected=entry['sha256'];payload=hashlib.sha256(z.read(members[name])).hexdigest();live=sha(ROOT/name);audit[name]=dict(expected=expected,payload=payload,live=live,exact=expected==payload==live)
    handoff_sha=sha(ROOT/p['owner_handoff'])
    if handoff_sha!=p['owner_handoff_SHA256'] or not all(x['exact'] for x in audit.values()):raise RuntimeError('Frozen owner byte mismatch')
    result=ROOT/(prefix+'_result.json');command=[PYTHON,str(consumer),str(inp),str(result)];start=time.monotonic();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt')
    with stdout.open('wb') as out,stderr.open('wb') as err:done=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err,preexec_fn=limit)
    after={path.name:sha(path) for path in sorted(sources)};files=sorted(sources)+[stdout,stderr,owner_exit]
    for path in [result,result.with_suffix('.npz')]:
        if path.is_file():files.append(path)
    metadata=dict(kind='NEW_WAVE16_PHYSICAL_CONSUMER_NATURAL_EXIT',command=command,returncode=done.returncode,natural_exit=done.returncode>=0,elapsed_seconds=time.monotonic()-start,source_input_before=before,source_input_after=after,source_unchanged=before==after,address_space_limit_MiB=3800,owner_handoff_SHA256=handoff_sha,owner_capsule_SHA256=sha(ROOT/p['owner_capsule']),owner_exact_byte_audit=audit,files={path.name:dict(bytes=path.stat().st_size,sha256=sha(path)) for path in files})
    ep=ROOT/(prefix+'_exit.json');ep.write_text(json.dumps(metadata,indent=2)+'\n')
    cap=ROOT/(prefix+'_capsule.zip')
    with zipfile.ZipFile(cap,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files+[ep]:z.write(path,path.name)
    print(json.dumps(dict(returncode=done.returncode,natural_exit=metadata['natural_exit'],seconds=metadata['elapsed_seconds'],source_unchanged=before==after,capsule_SHA256=sha(cap))))
    sys.exit(done.returncode)


if __name__=='__main__':main()

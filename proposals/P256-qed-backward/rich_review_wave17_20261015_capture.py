"""Exact NEW Wave17 physical-consumer capsule; failure is not natural success."""
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
from typing import Any
ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def sha(path: Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def closure(path: Path,seen: set[Path])->None:
    if path in seen:return
    seen.add(path)
    for node in ast.walk(ast.parse(path.read_text())):
        names=[node.module] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names] if isinstance(node,ast.Import) else []
        for name in names:
            local=ROOT/(str(name).split('.')[0]+'.py')
            if local.is_file():closure(local,seen)


def main()->None:
    inputpath=ROOT/sys.argv[1];prefix=ROOT/sys.argv[2];request:Any=json.loads(inputpath.read_text());consumer=ROOT/request['consumer']
    sources:set[Path]=set();closure(consumer,sources);sources|={Path(__file__).resolve(),inputpath}
    sources.update(ROOT/name for name in request.get('bindings',[]))
    before={p.name:sha(p) for p in sorted(sources)};capsule=Path(str(prefix)+'_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources):archive.write(path,path.name)
    command=[PYTHON,str(consumer),str(inputpath),str(prefix)+'_result.json'];start=time.monotonic();wall=time.time()
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with open(str(prefix)+'_stdout.txt','wb') as stdout,open(str(prefix)+'_stderr.txt','wb') as stderr:
        child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stdout,stderr=stderr);pid=child.pid;code=child.wait()
    after={p.name:sha(p) for p in sorted(sources)}
    paths=[Path(str(prefix)+suffix) for suffix in ['_result.json','_result.npz','_checkpoint.json','_checkpoint.npz','_stdout.txt','_stderr.txt']]
    outputs=[path for path in paths if path.is_file()]
    receipt=dict(kind='NEW_INDEPENDENT_WAVE17_NATURAL_OR_FAILED_ACTUAL_ATTEMPT',command=command,cwd=str(ROOT),child_pid=pid,
        started_unix=wall,finished_unix=time.time(),returncode=code,natural_exit=code>=0,success=code==0,signal=-code if code<0 else None,
        elapsed_seconds=time.monotonic()-start,maximum_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        source_and_input_before=before,source_and_input_after=after,source_unchanged=before==after,
        outputs={path.name:sha(path) for path in outputs},environment={key:environment[key] for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
    exitpath=Path(str(prefix)+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(capsule,'a',zipfile.ZIP_DEFLATED) as archive:
        for path in outputs+[exitpath]:archive.write(path,path.name)
    print(json.dumps(dict(returncode=code,natural_exit=code>=0,success=code==0,source_unchanged=before==after,capsule_sha256=sha(capsule),elapsed_seconds=receipt['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()

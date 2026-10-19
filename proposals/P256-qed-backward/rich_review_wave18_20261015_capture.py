"""NEW Wave18 exact source/input/state snapshots, streaming large trusted inputs."""
import ast,hashlib,json,os,resource,subprocess,sys,time,zipfile
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def sha(path: Path)->str:
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def closure(path: Path,seen: set[Path])->None:
    if path in seen:return
    seen.add(path)
    for node in ast.walk(ast.parse(path.read_text())):
        names=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
        for name in names:
            local=ROOT/(str(name).split('.')[0]+'.py')
            if local.is_file():closure(local,seen)


def main()->None:
    inp=ROOT/sys.argv[1];prefix=ROOT/sys.argv[2];request:Any=json.loads(inp.read_text());consumer=ROOT/request['consumer'];sources:set[Path]=set();closure(consumer,sources);sources|={Path(__file__).resolve(),inp};sources.update(ROOT/name for name in request['bindings'])
    before={p.name:sha(p) for p in sorted(sources)};cap=Path(str(prefix)+'_capsule.zip')
    with zipfile.ZipFile(cap,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources):archive.write(path,path.name)
    command=[PYTHON,str(consumer),str(inp),str(prefix)+'_result.json'];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');start=time.monotonic();wall=time.time()
    with open(str(prefix)+'_stdout.txt','wb') as stdout,open(str(prefix)+'_stderr.txt','wb') as stderr:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr);pid=child.pid;code=child.wait()
    after={p.name:sha(p) for p in sorted(sources)};suffixes=['_result.json','_result.npz','_checkpoint.json','_checkpoint.npz','_reconstruction.json','_reconstruction.npz','_stdout.txt','_stderr.txt'];outputs=[Path(str(prefix)+s) for s in suffixes];outputs=[p for p in outputs if p.is_file()]
    receipt=dict(kind='NEW_INDEPENDENT_WAVE18_ACTUAL_NATURAL_OR_FAILED_ATTEMPT',command=command,cwd=str(ROOT),child_pid=pid,returncode=code,natural_exit=code>=0,success=code==0,signal=-code if code<0 else None,started_unix=wall,finished_unix=time.time(),elapsed_seconds=time.monotonic()-start,maximum_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,source_and_input_before=before,source_and_input_after=after,source_unchanged=before==after,outputs={p.name:sha(p) for p in outputs})
    exitpath=Path(str(prefix)+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(cap,'a',zipfile.ZIP_DEFLATED) as archive:
        for path in outputs+[exitpath]:archive.write(path,path.name)
    print(json.dumps(dict(returncode=code,natural_exit=code>=0,success=code==0,source_unchanged=before==after,capsule_sha256=sha(cap),elapsed_seconds=receipt['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()

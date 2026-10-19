"""Archive exact new source/input bytes before launch and natural terminal evidence."""
from pathlib import Path
from typing import Any
import ast,hashlib,json,os,resource,subprocess,sys,time,zipfile
ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def digest(path: Path)->str:
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def closure(entry: str)->Any:
    seen=set();pending=[entry,Path(__file__).name]
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);tree=ast.parse((ROOT/name).read_text())
        for node in ast.walk(tree):
            modules=[alias.name for alias in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for module in modules:
                child=module.replace('.','/')+'.py'
                if (ROOT/child).is_file() and child not in seen:pending.append(child)
    return seen


def main()->None:
    request_file=Path(sys.argv[1]);prefix=sys.argv[2];request=json.loads(request_file.read_text());sources=closure(request['consumer']);names=sorted(sources|set(request['bindings'])|{request_file.name});before={name:digest(ROOT/name) for name in names};capsule=ROOT/(prefix+'_capsule.zip');started=time.time();clock=time.monotonic();environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONHASHSEED='0');stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt');result=ROOT/(prefix+'_result.json');command=[PYTHON,str(ROOT/request['consumer']),str(ROOT/request_file.name),str(result)]
    with zipfile.ZipFile(capsule,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as archive:
        for name in names:archive.write(ROOT/name,arcname=name)
    with stdout.open('w') as out,stderr.open('w') as err:child=subprocess.run(command,cwd=ROOT,env=environment,stdout=out,stderr=err)
    after={name:digest(ROOT/name) for name in names};outputs={}
    for suffix in ['_result.json','_result.npz','_checkpoint.json','_checkpoint.npz','_stdout.txt','_stderr.txt']:
        path=ROOT/(prefix+suffix)
        if path.is_file():outputs[path.name]=digest(path)
    receipt=dict(kind='NEW_INDEPENDENT_WAVE20_ACTUAL_NATURAL_OR_FAILED_ATTEMPT',command=command,cwd=str(ROOT),returncode=child.returncode,natural_exit=child.returncode>=0,success=child.returncode==0,signal=-child.returncode if child.returncode<0 else None,started_unix=started,finished_unix=time.time(),elapsed_seconds=time.monotonic()-clock,maximum_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,source_and_input_before=before,source_and_input_after=after,source_unchanged=before==after,outputs=outputs);exitfile=ROOT/(prefix+'_exit.json');exitfile.write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(capsule,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as archive:
        for name in outputs:archive.write(ROOT/name,arcname=name)
        archive.write(exitfile,arcname=exitfile.name)
    print(json.dumps(dict(returncode=child.returncode,natural_exit=receipt['natural_exit'],source_unchanged=before==after,capsule_sha256=digest(capsule),elapsed_seconds=receipt['elapsed_seconds'])),flush=True)
    raise SystemExit(child.returncode if child.returncode>=0 else 1)


if __name__=='__main__':main()

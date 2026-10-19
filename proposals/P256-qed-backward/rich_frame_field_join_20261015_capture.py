"""Immutable new-law attempts with sources/state checkpoints and natural receipts."""
import json,os,resource,subprocess,sys,time,zipfile
from pathlib import Path
from rich_review_wave19_20261015_capture import closure,sha
ROOT=Path(__file__).resolve().parent

def limit():
    maximum=3800*1024**2;resource.setrlimit(resource.RLIMIT_AS,(maximum,maximum))

def main():
    prefix=sys.argv[1];inp=ROOT/(prefix+'_input.json');p=json.loads(inp.read_text());consumer=ROOT/p['consumer'];sources=set();closure(consumer,sources);sources.update([inp,Path(__file__).resolve(),ROOT/p['native_body']]);sources.update(ROOT/p[key] for key in ['state_file','result_file'] if key in p);before={x.name:sha(x) for x in sources};cap=ROOT/(prefix+'_capsule.zip')
    if cap.exists():raise RuntimeError('Select a new immutable attempt')
    with zipfile.ZipFile(ROOT/(prefix+'_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for x in sources:z.write(x,x.name)
    command=['/home/administrator/substrate-framework/.venv/bin/python',str(consumer),str(inp),str(ROOT/(prefix+'_result.json'))];start=time.time();stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt')
    with stdout.open('wb') as out,stderr.open('wb') as err:
        process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=out,stderr=err,preexec_fn=limit);code=process.wait()
    after={x.name:sha(x) for x in sources};files=list(sources)+list(ROOT.glob(prefix+'_result*'))+[stdout,stderr];record=dict(command=command,pid=process.pid,returncode=code,natural_exit=code>=0,successful=code==0,seconds=time.time()-start,peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,address_space_limit_MiB=3800,source_before=before,source_after=after,source_unchanged=before==after,files={x.name:dict(bytes=x.stat().st_size,sha256=sha(x)) for x in files})
    ep=ROOT/(prefix+'_exit.json');ep.write_text(json.dumps(record,indent=2)+'\n')
    with zipfile.ZipFile(cap,'w',zipfile.ZIP_DEFLATED) as z:
        for x in files+[ep]:z.write(x,x.name)
    print(json.dumps(record));sys.exit(code)
if __name__=='__main__':main()

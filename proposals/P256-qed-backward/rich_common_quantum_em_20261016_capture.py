"""Own finite scientific-run receipt, immutable input epoch and executable capsule."""
import ast,hashlib,json,os,resource,subprocess,sys,time,zipfile
from pathlib import Path


def digest(path):
    result=hashlib.sha256()
    with open(path,'rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def sources(entry):
    found=set();pending=[Path(entry),Path(__file__)]
    while pending:
        path=pending.pop().resolve()
        if path in found:continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[node.module]if isinstance(node,ast.ImportFrom)else[a.name for a in node.names]if isinstance(node,ast.Import)else[]
            for name in names:
                if name and name.startswith('rich_'):
                    candidate=path.parent/(name+'.py')
                    if candidate.exists():pending.append(candidate)
    return sorted(found)


def main():
    configuration=Path(sys.argv[1]);config=json.loads(configuration.read_text());out=config['out'];inputs=[configuration.resolve(),Path('rich_common_quantum_em_20261016_law.json').resolve()];inputs+=sources(config['argv'][0]);patterns=config.get('input_patterns',[])
    for pattern in patterns:
        matched=sorted(Path.cwd().glob(pattern))
        if not matched:raise FileNotFoundError('No actual pinned inputs for '+pattern)
        inputs.extend(path.resolve()for path in matched if path.is_file())
    inputs=list(dict.fromkeys(inputs));before={str(path):{'sha256':digest(path),'bytes':path.stat().st_size}for path in inputs};Path(out+'_before.json').write_text(json.dumps(before,indent=2)+'\n');started=time.time()
    with zipfile.ZipFile(out+'_source_inputs.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3)as archive:
        for path in inputs:archive.write(path,path.name)
    environment=os.environ.copy();environment.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'})
    def limit():resource.setrlimit(resource.RLIMIT_AS,(3800*1024**2,3800*1024**2))
    with open(out+'_stdout.txt','w')as stdout,open(out+'_stderr.txt','w')as stderr:
        process=subprocess.Popen(['/home/administrator/substrate-framework/.venv/bin/python',*config['argv']],stdout=stdout,stderr=stderr,env=environment,preexec_fn=limit);Path(out+'_running_receipt.json').write_text(json.dumps({'pid':process.pid,'command':config['argv'],'started_unix':started,'address_space_MiB':3800},indent=2)+'\n');code=process.wait()
    changed=[str(path)for path in inputs if not path.exists()or digest(path)!=before[str(path)]['sha256']];usage=resource.getrusage(resource.RUSAGE_CHILDREN);receipt={'returncode':code,'natural_exit':code>=0,'natural_zero':code==0,'elapsed_seconds':time.time()-started,'child_peak_RSS_KiB':usage.ru_maxrss,'child_user_seconds':usage.ru_utime,'child_system_seconds':usage.ru_stime,'input_epoch_changed':changed,'command':config['argv'],'pid':process.pid,'address_space_MiB':3800};Path(out+'_exit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(out+'_capsule.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True)as archive:
        for path in sorted(Path.cwd().glob(out+'*')):
            if path.is_file()and path.name!=out+'_capsule.zip':archive.write(path,path.name)
    receipt['capsule_sha256']=digest(out+'_capsule.zip');Path(out+'_capsule_sha256.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    if changed:raise RuntimeError('Captured input epoch changed')
    if code:sys.exit(code if code>0 else 128-code)


if __name__=='__main__':main()

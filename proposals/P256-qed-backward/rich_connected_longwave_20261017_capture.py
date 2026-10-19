"""Own scientific receipts; frozen predecessor deliveries are never rewritten."""
from pathlib import Path
import ast,hashlib,json,os,resource,subprocess,sys,time,zipfile


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def sources(entry):
    found=set();pending=[Path(entry),Path(__file__)]
    while pending:
        p=pending.pop().resolve()
        if p in found:continue
        found.add(p)
        for n in ast.walk(ast.parse(p.read_text())):
            names=[n.module]if isinstance(n,ast.ImportFrom)else[a.name for a in n.names]if isinstance(n,ast.Import)else[]
            for name in names:
                if name and name.startswith('rich_'):
                    q=p.parent/(name+'.py')
                    if q.exists():pending.append(q)
    return sorted(found)


def main():
    configpath=Path(sys.argv[1]);config=json.loads(configpath.read_text());out=config['out'];inputs=[configpath.resolve(),Path('rich_connected_longwave_20261017_law.json').resolve(),*sources(config['argv'][0])]
    for pattern in config.get('input_patterns',[]):
        paths=sorted(Path.cwd().glob(pattern))
        if not paths:raise FileNotFoundError(pattern)
        inputs.extend(p.resolve()for p in paths if p.is_file())
    inputs=list(dict.fromkeys(inputs));before={str(p):{'sha256':digest(p),'bytes':p.stat().st_size}for p in inputs};Path(out+'_before.json').write_text(json.dumps(before,indent=2)+'\n')
    with zipfile.ZipFile(out+'_source_inputs.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3)as z:
        for p in inputs:z.write(p,p.name)
    env=os.environ.copy();env.update({key:'1'for key in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')});env['PYTHONUNBUFFERED']='1'
    def limit():resource.setrlimit(resource.RLIMIT_AS,(3800*1024**2,3800*1024**2))
    started=time.time()
    with open(out+'_stdout.txt','w')as stdout,open(out+'_stderr.txt','w')as stderr:
        child=subprocess.Popen(['/home/administrator/substrate-framework/.venv/bin/python',*config['argv']],stdout=stdout,stderr=stderr,env=env,preexec_fn=limit);Path(out+'_running_receipt.json').write_text(json.dumps({'pid':child.pid,'argv':config['argv'],'started':started,'address_space_MiB':3800},indent=2)+'\n');code=child.wait()
    changed=[str(p)for p in inputs if not p.exists()or digest(p)!=before[str(p)]['sha256']];usage=resource.getrusage(resource.RUSAGE_CHILDREN);receipt={'returncode':code,'natural_exit':code>=0,'natural_zero':code==0,'elapsed_seconds':time.time()-started,'child_peak_RSS_KiB':usage.ru_maxrss,'child_user_seconds':usage.ru_utime,'child_system_seconds':usage.ru_stime,'input_epoch_changed':changed,'command':config['argv'],'pid':child.pid,'address_space_MiB':3800};Path(out+'_exit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(out+'_capsule.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True)as z:
        for p in sorted(Path.cwd().glob(out+'*')):
            if p.is_file()and p.name!=out+'_capsule.zip':z.write(p,p.name)
    receipt['capsule_sha256']=digest(out+'_capsule.zip');Path(out+'_capsule_sha256.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    if changed:raise RuntimeError('Input epoch changed')
    if code:sys.exit(code if code>0 else 128-code)


if __name__=='__main__':main()

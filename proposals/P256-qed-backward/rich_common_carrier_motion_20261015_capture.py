"""Exact sources/input/body before launch; natural or interrupted process capsule."""
import ast,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    ip=Path(sys.argv[1]).resolve();prefix=Path(sys.argv[2]).resolve();p=json.loads(ip.read_text());consumer=ROOT/p['consumer'];files={ip,consumer,Path(__file__).resolve(),ROOT/p['native_body']};queue=[consumer,Path(__file__).resolve()]
    for key in ['own_motion_states','own_motion_result','own_factor_result','own_factor_states']:
        if key in p:files.add((ROOT/p[key]).resolve())
    while queue:
        path=queue.pop()
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                local=ROOT/((module or '').split('.')[0]+'.py')
                if local.is_file() and local not in files:files.add(local);queue.append(local)
    manifest={f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in sorted(files)}
    sourcecap=Path(str(prefix)+'_source_capsule.zip')
    with zipfile.ZipFile(sourcecap,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for f in sorted(files):z.write(f,f.name)
    Path(str(prefix)+'_start.json').write_text(json.dumps({'inputs':manifest,'source_capsule_sha256':hashlib.sha256(sourcecap.read_bytes()).hexdigest(),'command':[sys.executable,str(consumer),str(ip),str(prefix)+'_result'],'deadline_seconds':p['deadline_seconds']},indent=2)+'\n')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');start=time.monotonic();interrupted=False
    with Path(str(prefix)+'_stdout.txt').open('xb') as stdout,Path(str(prefix)+'_stderr.txt').open('xb') as stderr:
        process=subprocess.Popen([sys.executable,str(consumer),str(ip),str(prefix)+'_result'],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        try:rc=process.wait(timeout=p['deadline_seconds'])
        except subprocess.TimeoutExpired:
            interrupted=True;process.terminate()
            try:rc=process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();rc=process.wait()
    receipt={'returncode':rc,'natural_exit':not interrupted,'deadline_interrupted':interrupted,'elapsed_seconds':time.monotonic()-start,'inputs':manifest,'input_bytes_unchanged':all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v['sha256'] for n,v in manifest.items()),'outputs':{}}
    for f in sorted(ROOT.glob(prefix.name+'*')):
        if f.is_file() and f.suffix!='.zip':receipt['outputs'][f.name]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size}
    exitpath=Path(str(prefix)+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(str(prefix)+'_capsule.zip','x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        z.write(sourcecap,sourcecap.name)
        for n in receipt['outputs']:z.write(ROOT/n,n)
        z.write(exitpath,exitpath.name)
    print(json.dumps({'returncode':rc,'natural_exit':not interrupted,'elapsed_seconds':receipt['elapsed_seconds'],'capsule':str(prefix)+'_capsule.zip'}),flush=True)
if __name__=='__main__':main()

"""Bind actual science execution to immutable source and output capsule."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,zipfile,platform,os
ROOT=Path(__file__).resolve().parent
prefix='rich_spinor_gauss_20261014_actual3'
sources=[ROOT/'rich_spinor_gauss_20261014_model.py',ROOT/'rich_spinor_gauss_20261014_calculate.py',ROOT/'rich_spinor_20261014_model.py',Path(__file__).resolve()]
start={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
command=[sys.executable,str(ROOT/'rich_spinor_gauss_20261014_calculate.py')];t=time.monotonic()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
(ROOT/(prefix+'_stdout.txt')).write_text(process.stdout);(ROOT/(prefix+'_stderr.txt')).write_text(process.stderr)
receipt={'command':command,'cwd':str(ROOT),'runtime':sys.version,'platform':platform.platform(),'returncode':process.returncode,'natural_exit':True,'elapsed_seconds':time.monotonic()-t,'source_sha256':start,'source_unchanged':all(hashlib.sha256(p.read_bytes()).hexdigest()==start[p.name] for p in sources),'output_sha256':{}}
for suffix in ['_stdout.txt','_stderr.txt','_result.json','_states.npz']:
    p=ROOT/(prefix+suffix)
    if p.exists():receipt['output_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/(prefix+'_exit.json')).write_text(json.dumps(receipt,indent=2))
with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for p in sources+[ROOT/(prefix+s) for s in ['_stdout.txt','_stderr.txt','_result.json','_states.npz','_exit.json']]:
        if p.exists():z.write(p,p.name)
print(json.dumps(receipt));print(process.stdout);print(process.stderr,file=sys.stderr)
sys.exit(process.returncode)

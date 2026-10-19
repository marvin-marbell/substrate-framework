"""Capture NEW physical flux-source discriminator with source snapshots."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,zipfile,os
ROOT=Path(__file__).resolve().parent;prefix='rich_spinor_gauss_20261014_flux_actual2'
sources=[ROOT/'rich_spinor_gauss_20261014_model.py',ROOT/'rich_spinor_gauss_20261014_flux_probe.py',ROOT/'rich_spinor_20261014_model.py',Path(__file__).resolve()]
sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
cmd=[sys.executable,str(ROOT/'rich_spinor_gauss_20261014_flux_probe.py')];start=time.monotonic();p=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),capture_output=True,text=True)
(ROOT/(prefix+'_stdout.txt')).write_text(p.stdout);(ROOT/(prefix+'_stderr.txt')).write_text(p.stderr)
receipt={'command':cmd,'runtime':sys.version,'cwd':str(ROOT),'returncode':p.returncode,'natural_exit':True,'elapsed_seconds':time.monotonic()-start,'source_sha256':sha,'source_unchanged':all(hashlib.sha256(x.read_bytes()).hexdigest()==sha[x.name] for x in sources),'output_sha256':{}}
outputs=[ROOT/(prefix+s) for s in ['_stdout.txt','_stderr.txt','_result.json']]
for x in outputs:
    if x.exists():receipt['output_sha256'][x.name]=hashlib.sha256(x.read_bytes()).hexdigest()
receiptpath=ROOT/(prefix+'_exit.json');receiptpath.write_text(json.dumps(receipt,indent=2))
with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for x in sources+outputs+[receiptpath]:
        if x.exists():z.write(x,x.name)
print(json.dumps(receipt));print(p.stdout);print(p.stderr,file=sys.stderr);sys.exit(p.returncode)

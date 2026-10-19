"""Immutable input-state/source capsule for a NEW polarization consumer."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent;prefix='rich_extensive_photon_20261014_'+sys.argv[1]
    inp=root/(prefix+'_input.json');out=root/(prefix+'_result.json');request=json.loads(inp.read_text())
    names=['rich_extensive_photon_20261014_model.py','rich_extensive_photon_20261014_tensor.py','rich_extensive_photon_20261014_tensor_capture.py',
           'rich_extensive_photon_20261014_equations.json',inp.name,request['physical_input'],request['result'],request['states'],'rich_extensive_photon_20261014_actual1_exit.json']
    paths=[root/name for name in names];before={p.name:sha(p) for p in paths};frozen={p.name:p.read_bytes() for p in paths}
    environment=dict(os.environ)
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:environment[key]='1'
    command=[sys.executable,str(root/'rich_extensive_photon_20261014_tensor.py'),str(inp),str(out)];start=time.time()
    process=subprocess.run(command,cwd=root,env=environment,capture_output=True,text=True)
    stdout=root/(prefix+'_stdout.txt');stderr=root/(prefix+'_stderr.txt');stdout.write_text(process.stdout);stderr.write_text(process.stderr)
    after={p.name:sha(p) for p in paths};outputs=[stdout,stderr]+[p for p in [out,out.with_suffix('.npz')] if p.exists()]
    receipt=dict(command=command,returncode=process.returncode,natural_exit=True,elapsed_seconds=time.time()-start,source_before=before,source_after=after,
                 source_unchanged=before==after,outputs={p.name:sha(p) for p in outputs},executable=sys.executable,python=sys.version)
    exitpath=root/(prefix+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n');capsule=root/(prefix+'_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in frozen.items():archive.writestr(name,data)
        for p in outputs+[exitpath]:archive.write(p,p.name)
    print(process.stdout)
    if process.stderr:print(process.stderr,file=sys.stderr)
    print(json.dumps(dict(returncode=process.returncode,elapsed_seconds=receipt['elapsed_seconds'],capsule=str(capsule),capsule_sha256=sha(capsule))))
    raise SystemExit(process.returncode)


if __name__=='__main__':main()

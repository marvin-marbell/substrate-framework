"""Freeze exact NEW gauge consumer and complete pure-source dependencies."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    root=Path(__file__).resolve().parent;prefix='rich_rotcov_gauge_20261015_'+sys.argv[1];inp=root/(prefix+'_input.json');out=root/(prefix+'_result.json')
    names=['rich_rotcov_gauge_20261015_model.py','rich_rotcov_gauge_20261015_calculate.py','rich_rotcov_gauge_20261015_capture.py','rich_rotcov_gauge_20261015_equations.json','rich_rotcov_gauge_20261015_electrostatics.py','rich_rotcov_gauge_20261015_interface.json','rich_rotcov_induced_20261015_model.py','rich_rotcov_induced_20261015_calculate.py','rich_rotcov_induced_20261015_connections.py','rich_physical_noether_20261014_model.py','rich_frame_holonomy_20261014_model.py','rich_frame_gauss_20261014_model.py',inp.name]
    names+=['rich_rotcov_gauge_20261015_sparse.py','rich_rotcov_gauge_20261015_source_contract.json']
    names+=['rich_rotcov_gauge_20261015_primitive.py']
    paths=[root/name for name in names];frozen={p.name:p.read_bytes() for p in paths};before={p.name:sha(p) for p in paths};environment=dict(os.environ)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:environment[key]='1'
    command=[sys.executable,str(root/'rich_rotcov_gauge_20261015_calculate.py'),str(inp),str(out)];started=time.time();process=subprocess.run(command,cwd=root,env=environment,capture_output=True,text=True)
    stdout=root/(prefix+'_stdout.txt');stderr=root/(prefix+'_stderr.txt');stdout.write_text(process.stdout);stderr.write_text(process.stderr)
    after={p.name:sha(p) for p in paths};outputs=[stdout,stderr]+[p for p in [out,out.with_suffix('.npz')] if p.exists()]
    receipt=dict(command=command,cwd=str(root),returncode=process.returncode,natural_exit=True,elapsed_seconds=time.time()-started,source_before=before,source_after=after,source_unchanged=before==after,outputs={p.name:sha(p) for p in outputs},python=sys.version,executable=sys.executable,platform=platform.platform(),packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','sympy']})
    receipt['natural_exit']=process.returncode>=0
    receipt['termination_signal']=-process.returncode if process.returncode<0 else None
    exitpath=root/(prefix+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n');capsule=root/(prefix+'_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in frozen.items():archive.writestr(name,content)
        for path in outputs+[exitpath]:archive.write(path,path.name)
    print(process.stdout)
    if process.stderr:print(process.stderr,file=sys.stderr)
    print(json.dumps(dict(returncode=process.returncode,elapsed_seconds=receipt['elapsed_seconds'],capsule=str(capsule),capsule_sha256=sha(capsule))))
    raise SystemExit(process.returncode)
if __name__=='__main__':main()

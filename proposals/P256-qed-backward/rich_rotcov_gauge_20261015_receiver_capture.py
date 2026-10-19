"""Exact same-state physical-receiver capsule, separate from frozen dynamics."""
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
    root=Path(__file__).resolve().parent;tag=sys.argv[1];prefix='rich_rotcov_gauge_20261015_'+tag
    inp=root/(prefix+'_input.json');states=root/(prefix+'_result.npz');out=root/(prefix+'_receiver_result.json')
    names=['rich_rotcov_gauge_20261015_receiver.py','rich_rotcov_gauge_20261015_receiver_capture.py','rich_rotcov_gauge_20261015_model.py','rich_rotcov_gauge_20261015_electrostatics.py','rich_rotcov_induced_20261015_model.py','rich_rotcov_induced_20261015_calculate.py','rich_rotcov_induced_20261015_connections.py','rich_physical_noether_20261014_model.py','rich_frame_holonomy_20261014_model.py','rich_frame_gauss_20261014_model.py','rich_rotcov_gauge_20261015_source_contract.json',inp.name,states.name]
    names+=['rich_rotcov_gauge_20261015_sparse.py']
    names+=['rich_rotcov_gauge_20261015_primitive.py']
    paths=[root/n for n in names];frozen={p.name:p.read_bytes() for p in paths};before={p.name:sha(p) for p in paths};environment=dict(os.environ)
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:environment[k]='1'
    command=[sys.executable,str(root/'rich_rotcov_gauge_20261015_receiver.py'),str(inp),str(states),str(out)];started=time.time();process=subprocess.run(command,cwd=root,env=environment,capture_output=True,text=True)
    stdout=root/(prefix+'_receiver_stdout.txt');stderr=root/(prefix+'_receiver_stderr.txt');stdout.write_text(process.stdout);stderr.write_text(process.stderr)
    after={p.name:sha(p) for p in paths};outputs=[stdout,stderr]+([out] if out.exists() else [])
    receipt=dict(command=command,returncode=process.returncode,natural_exit=process.returncode>=0,termination_signal=-process.returncode if process.returncode<0 else None,elapsed_seconds=time.time()-started,source_before=before,source_after=after,source_unchanged=before==after,outputs={p.name:sha(p) for p in outputs})
    exitpath=root/(prefix+'_receiver_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n');capsule=root/(prefix+'_receiver_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as z:
        for n,content in frozen.items():z.writestr(n,content)
        for path in outputs+[exitpath]:z.write(path,path.name)
    print(process.stdout)
    if process.stderr:print(process.stderr,file=sys.stderr)
    print(json.dumps(dict(returncode=process.returncode,elapsed_seconds=receipt['elapsed_seconds'],capsule_sha256=sha(capsule))))
    raise SystemExit(process.returncode)
if __name__=='__main__':main()

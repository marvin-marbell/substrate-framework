"""Bind exact NEW physical run sources, input, outputs and natural exit."""
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent
    label=sys.argv[1]
    prefix='rich_common_mobile_20261014_'+label
    inp=root/(prefix+'_input.json');out=root/(prefix+'_result.json')
    paths=[root/name for name in ['rich_common_mobile_20261014_model.py','rich_common_mobile_20261014_calculate.py','rich_common_mobile_20261014_capture.py','rich_common_mobile_20261014_equations.json','rich_frame_photon_20261014_model.py','rich_frame_gauss_20261014_model.py',inp.name]]
    before={p.name:sha(p) for p in paths};raw={p.name:p.read_bytes() for p in paths}
    env=dict(os.environ)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env[key]='1'
    command=[sys.executable,str(root/'rich_common_mobile_20261014_calculate.py'),str(inp),str(out)]
    started=time.time();proc=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True)
    stdout=root/(prefix+'_stdout.txt');stderr=root/(prefix+'_stderr.txt')
    stdout.write_text(proc.stdout);stderr.write_text(proc.stderr)
    after={p.name:sha(p) for p in paths}
    outputs=[stdout,stderr]+[p for p in [out,out.with_suffix('.npz')] if p.exists()]
    receipt=dict(command=command,cwd=str(root),python=sys.version,executable=sys.executable,platform=platform.platform(),started_unix=started,elapsed_seconds=time.time()-started,returncode=proc.returncode,natural_exit=True,source_before=before,source_after=after,source_unchanged=before==after,outputs={p.name:sha(p) for p in outputs},packages={n:importlib.metadata.version(n)for n in ('numpy','scipy','sympy')},environment={k:env[k]for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
    exitpath=root/(prefix+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
    capsule=root/(prefix+'_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in raw.items():archive.writestr(name,content)
        for p in outputs+[exitpath]:archive.write(p,p.name)
    print(proc.stdout)
    if proc.stderr:print(proc.stderr,file=sys.stderr)
    print(json.dumps(dict(returncode=proc.returncode,seconds=receipt['elapsed_seconds'],capsule=str(capsule),capsule_sha256=sha(capsule),sources_unchanged=before==after)))
    raise SystemExit(proc.returncode)


if __name__=='__main__':main()

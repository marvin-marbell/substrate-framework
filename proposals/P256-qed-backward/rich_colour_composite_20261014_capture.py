"""Freeze exact NEW sources/input and capture actual scientific consumer exit."""
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
    root=Path(__file__).resolve().parent;prefix='rich_colour_composite_20261014_'+sys.argv[1]
    inp=root/(prefix+'_input.json');out=root/(prefix+'_result.json')
    names=['rich_colour_composite_20261014_model.py','rich_colour_composite_20261014_calculate.py','rich_colour_composite_20261014_capture.py','rich_colour_composite_20261014_equations.json',inp.name]
    paths=[root/name for name in names];raw={p.name:p.read_bytes() for p in paths};before={p.name:sha(p) for p in paths}
    environment=dict(os.environ)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:environment[key]='1'
    command=[sys.executable,str(root/'rich_colour_composite_20261014_calculate.py'),str(inp),str(out)]
    started=time.time();process=subprocess.run(command,cwd=root,env=environment,capture_output=True,text=True)
    stdout=root/(prefix+'_stdout.txt');stderr=root/(prefix+'_stderr.txt');stdout.write_text(process.stdout);stderr.write_text(process.stderr)
    after={p.name:sha(p) for p in paths};outputs=[stdout,stderr]+[p for p in [out,out.with_suffix('.npz')] if p.exists()]
    receipt=dict(command=command,cwd=str(root),returncode=process.returncode,natural_exit=True,elapsed_seconds=time.time()-started,
                 source_before=before,source_after=after,source_unchanged=before==after,outputs={p.name:sha(p) for p in outputs},
                 python=sys.version,executable=sys.executable,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ['numpy','scipy']})
    exitpath=root/(prefix+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n');capsule=root/(prefix+'_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in raw.items():archive.writestr(name,content)
        for path in outputs+[exitpath]:archive.write(path,path.name)
    print(process.stdout)
    if process.stderr:print(process.stderr,file=sys.stderr)
    print(json.dumps(dict(returncode=process.returncode,seconds=receipt['elapsed_seconds'],capsule=str(capsule),capsule_sha256=sha(capsule))))
    raise SystemExit(process.returncode)


if __name__=='__main__':
    main()

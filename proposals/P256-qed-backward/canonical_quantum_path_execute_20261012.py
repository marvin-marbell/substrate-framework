"""Archive exact own source/input bytes and preserve every natural execution epoch."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--request',required=True)
    args=parser.parse_args(); request=ROOT/args.request
    req=json.loads(request.read_text()); prefix=req['epoch']
    pre=ROOT/(prefix+'_preexecution.zip'); executed=ROOT/(prefix+'_executed.zip')
    if pre.exists() or executed.exists():
        raise FileExistsError('Immutable epoch already exists')
    captures={name:(ROOT/name).read_bytes() for name in req['pins']}
    for name,data in captures.items():
        if hashlib.sha256(data).hexdigest()!=req['pins'][name]:
            raise ValueError('Source/input pin mismatch: '+name)
    with zipfile.ZipFile(pre,'x',zipfile.ZIP_DEFLATED) as z:
        for name,data in captures.items(): z.writestr(name,data)
        z.writestr(request.name,request.read_bytes())
    env=dict(os.environ); env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    command=[PYTHON,str(ROOT/req['consumer']),'--request',request.name]
    start=time.monotonic()
    proc=subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
    elapsed=time.monotonic()-start
    stdout=ROOT/(prefix+'_stdout.txt'); stderr=ROOT/(prefix+'_stderr.txt')
    stdout.write_bytes(proc.stdout); stderr.write_bytes(proc.stderr)
    exitpath=ROOT/(prefix+'_exit.json')
    exitpath.write_text(json.dumps({'returncode':proc.returncode,'natural_exit':True,
        'elapsed_seconds':elapsed,'command':command},indent=2)+'\n')
    stable=all((ROOT/name).read_bytes()==data for name,data in captures.items())
    with zipfile.ZipFile(executed,'x',zipfile.ZIP_DEFLATED) as z:
        for name in captures: z.writestr(name,(ROOT/name).read_bytes())
        for path in [request,stdout,stderr,exitpath,ROOT/req['receipt']]:
            if path.exists():z.writestr(path.name,path.read_bytes())
    paths=[request,pre,executed,stdout,stderr,exitpath]
    if (ROOT/req['receipt']).exists():paths.append(ROOT/req['receipt'])
    capsule={'kind':'NATURAL_EXIT_SOURCE_INPUT_BOUND_CANONICAL_QUANTUM_PATH_CAPSULE',
        'natural_returncode':proc.returncode,'elapsed_seconds':elapsed,
        'inputs_stable':stable,'artifacts':{p.name:sha(p) for p in paths},
        'source_input_pins':req['pins'],'full_quantum_ground_backend':False,
        'campaign_complete':False}
    (ROOT/(prefix+'_capsule.json')).write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps({'returncode':proc.returncode,'elapsed_seconds':elapsed,
                      'capsule':prefix+'_capsule.json','inputs_stable':stable}))
    raise SystemExit(proc.returncode)

if __name__=='__main__':main()

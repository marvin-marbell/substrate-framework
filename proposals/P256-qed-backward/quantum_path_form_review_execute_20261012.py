"""Capture exact selected inputs before executing the new independent CLI."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);args=ap.parse_args()
    request_path=ROOT/args.request;request_raw=request_path.read_bytes();req=json.loads(request_raw)
    captured={name:(ROOT/name).read_bytes() for name in req['pins']}
    for name,raw in captured.items():
        if digest(raw)!=req['pins'][name]:raise ValueError('Selected input drift: '+name)
    with zipfile.ZipFile(ROOT/req['profile_archive']) as z:member=z.read(req['profile_member'])
    assert digest(member)==req['profile_sha256']
    with zipfile.ZipFile(ROOT/req['review_snapshot']) as z:
        for name in req['snapshot_members']:
            assert z.read(name)==captured[name],name
    epoch=req['epoch'];pre=ROOT/(epoch+'_preexecution.zip')
    with zipfile.ZipFile(pre,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr(request_path.name,request_raw)
        for name,raw in captured.items():
            if name==req['profile_archive']:continue
            z.writestr(Path(name).name,raw)
        z.writestr('selected_profile_member/'+req['profile_member'],member)
    command=[sys.executable,str(ROOT/req['consumer']),'--request',request_path.name]
    start=time.monotonic();run=subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
    elapsed=time.monotonic()-start
    stdout=ROOT/(epoch+'_stdout.txt');stderr=ROOT/(epoch+'_stderr.txt')
    stdout.write_bytes(run.stdout);stderr.write_bytes(run.stderr)
    stable=all((ROOT/name).read_bytes()==raw for name,raw in captured.items()) and request_path.read_bytes()==request_raw
    exit_path=ROOT/(epoch+'_exit.json')
    exit_payload={'command':command,'returncode':run.returncode,'natural_exit':True,
                  'elapsed_seconds':elapsed,'stable_inputs':stable,'timeout_used':False}
    exit_path.write_text(json.dumps(exit_payload,indent=2)+'\n')
    executed=ROOT/(epoch+'_executed.zip')
    with zipfile.ZipFile(executed,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(pre,pre.name)
        for p in [stdout,stderr,exit_path,ROOT/req['receipt']]:
            if p.exists():z.write(p,p.name)
    versions=subprocess.run([sys.executable,'-c','import sys,sympy;print(sys.version);print("sympy",sympy.__version__)'],capture_output=True,check=True,text=True).stdout
    members=[request_path,pre,executed,stdout,stderr,exit_path]
    if (ROOT/req['receipt']).exists():members.append(ROOT/req['receipt'])
    capsule={'kind':'SOURCE_SELECTED_INDEPENDENT_REVIEW_CAPSULE_NOT_GROUND',
             'epoch':epoch,'source':{'path':req['consumer'],'sha256':req['pins'][req['consumer']]},
             'executor':{'path':Path(__file__).name,'sha256':req['pins'][Path(__file__).name]},
             'proof':{'path':req['proof'],'sha256':req['pins'][req['proof']]},
             'request':{'path':request_path.name,'sha256':digest(request_raw)},
             'runtime':{'executable':sys.executable,'executable_sha256':digest(Path(sys.executable).resolve().read_bytes()),
                        'platform':platform.platform(),'versions':versions},
             'input_pins':req['pins'],'profile_member':req['profile_member'],
             'profile_sha256':digest(member),'preexecution_backfill':False,
             'snapshot_is_postexecution_original':True,'execution':exit_payload,
             'artifacts':{p.name:digest(p.read_bytes()) for p in members},
             'original_functions_executed':False,'originals_modified':False,
             'shared_plan_modified':False,'scientific_admission':False,'campaign_complete':False}
    (ROOT/(epoch+'_capsule.json')).write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps({'returncode':run.returncode,'stable_inputs':stable,'elapsed_seconds':elapsed,'capsule':epoch+'_capsule.json'}))
    if run.returncode or not stable:raise SystemExit(run.returncode or 1)


if __name__=='__main__':
    main()

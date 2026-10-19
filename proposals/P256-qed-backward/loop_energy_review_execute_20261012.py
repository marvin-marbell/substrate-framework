"""Archive exact NEW scientific inputs before execution, including failed exits."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_bytes());prefix=request['epoch_prefix']
    assert prefix.startswith('loop_energy_review_')
    inputs={name:(ROOT/name).read_bytes() for name in request['pins']}
    before={name:sha(raw) for name,raw in inputs.items()};assert before==request['pins']
    pre=ROOT/(prefix+'_preexecution.zip');executed=ROOT/(prefix+'_executed.zip')
    output=ROOT/(prefix+'_receipt.json');stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt')
    exit_path=ROOT/(prefix+'_exit.json');capsule=ROOT/(prefix+'_capsule.json')
    assert not any(p.exists() for p in (pre,executed,output,stdout,stderr,exit_path,capsule))
    with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in inputs.items():archive.writestr(Path(name).name,raw)
        archive.writestr(args.request.name,args.request.read_bytes())
    pre_hash=sha(pre.read_bytes());started_at=dt.datetime.now(dt.timezone.utc).isoformat()
    command=[PYTHON,str(ROOT/request['consumer']),'--request',str(args.request),'--output',str(output)]
    start=time.monotonic();process=subprocess.run(command,capture_output=True,check=False)
    runtime={'command':command,'returncode':process.returncode,'natural_exit':process.returncode>=0,
             'started_at':started_at,'completed_at':dt.datetime.now(dt.timezone.utc).isoformat(),
             'elapsed_seconds':time.monotonic()-start,'timeout_or_signal_exit_used':False}
    stdout.write_bytes(process.stdout);stderr.write_bytes(process.stderr);save(exit_path,runtime)
    after={name:sha((ROOT/name).read_bytes()) for name in inputs}
    outputs=[args.request,stdout,stderr,exit_path]
    if output.exists():outputs.append(output)
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in inputs.items():archive.writestr(Path(name).name,raw)
        for path in outputs:archive.writestr(path.name,path.read_bytes())
    save(capsule,{'kind':'INDEPENDENT_LOOP_ENERGY_EXACT_REVIEW_IMMUTABLE_CAPSULE',
         'pins_before':before,'pins_after':after,'inputs_unchanged':before==after,
         'preexecution':{'path':pre.name,'sha256':pre_hash,'created_before_child_launch':True},
         'executed':{'path':executed.name,'sha256':sha(executed.read_bytes())},
         'outputs':{p.name:sha(p.read_bytes()) for p in outputs},'runtime':runtime,
         'failed_epoch_preserved':process.returncode!=0,'preexecution_backfill':False,
         'original_producer_main_root_solver_checker_eigensolver_executed':False,
         'scientific_admission':False,'campaign_complete':False})
    print(json.dumps({'capsule':capsule.name,'returncode':process.returncode,'natural_exit':runtime['natural_exit'],
                      'inputs_unchanged':before==after,'elapsed_seconds':runtime['elapsed_seconds']}))
    if process.returncode:raise SystemExit(process.returncode)
    assert before==after


if __name__=='__main__':
    main()

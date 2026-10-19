"""Freeze selected bytes then launch only the independent carrier review consumer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parent

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def save(path: Path, data) -> None:
    with path.open('x') as f:
        f.write(json.dumps(data,indent=2)+'\n')

def snapshot(path: Path, frozen: dict[str,bytes]) -> None:
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for name,raw in frozen.items():
            member=name if not Path(name).is_absolute() else 'absolute/'+name.lstrip('/')
            z.writestr(member,raw)
        z.writestr('carrier_action_ward_review_manifest.json',json.dumps({n:sha(r) for n,r in frozen.items()},indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();request_raw=args.request.read_bytes();request=json.loads(request_raw)
    assert request['consumer']=='carrier_action_ward_review_consumer_20261012.py'
    frozen={n:(ROOT/n).read_bytes() for n in request['pins']}
    assert all(sha(raw)==request['pins'][n] for n,raw in frozen.items())
    frozen[args.request.name]=request_raw
    prefix='carrier_action_ward_review_actual1_20261012'
    pre=ROOT/(prefix+'_preexecution.zip');executed=ROOT/(prefix+'_executed.zip')
    receipt=ROOT/(prefix+'_receipt.json');stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt');exitpath=ROOT/(prefix+'_exit.json')
    snapshot(pre,frozen)
    command=[request['python'],str(ROOT/request['consumer']),'--request',str(args.request),'--output',str(receipt)]
    start=time.monotonic()
    with stdout.open('xb') as out,stderr.open('xb') as err:
        child=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,check=False)
    elapsed=time.monotonic()-start
    stable=all((ROOT/n).read_bytes()==raw for n,raw in frozen.items())
    exitdata={'command':command,'returncode':child.returncode,'natural_exit':True,'timeout':False,'signal':None if child.returncode>=0 else -child.returncode,'elapsed_seconds':elapsed,'inputs_unchanged':stable,'stdout_sha256':sha(stdout.read_bytes()),'stderr_sha256':sha(stderr.read_bytes()),'receipt_exists':receipt.exists()}
    save(exitpath,exitdata)
    final=dict(frozen)
    for path in [stdout,stderr,exitpath,receipt]:
        if path.exists():final[path.name]=path.read_bytes()
    snapshot(executed,final)
    capsule={'kind':'NEW_SOURCE_SELECTED_INDEPENDENT_CARRIER_REVIEW_NATURAL_EXIT_CAPSULE','source':request['consumer'],'source_sha256':request['pins'][request['consumer']],'input_pins':request['pins'],'profile_member_sha256':request['profile_member_sha256'],'preexecution':{'path':pre.name,'sha256':sha(pre.read_bytes())},'executed':{'path':executed.name,'sha256':sha(executed.read_bytes())},'exit':{'path':exitpath.name,'sha256':sha(exitpath.read_bytes()),'data':exitdata},'receipt':{'path':receipt.name,'sha256':sha(receipt.read_bytes()) if receipt.exists() else None},'request':{'path':args.request.name,'sha256':sha(request_raw)},'scientific_admission':False,'campaign_complete':False,'old_producers_run':False}
    save(ROOT/(prefix+'_capsule.json'),capsule)
    print(json.dumps({'returncode':child.returncode,'natural_exit':True,'inputs_unchanged':stable,'receipt':receipt.name}))
    raise SystemExit(child.returncode if stable else 99)

if __name__=='__main__':
    main()

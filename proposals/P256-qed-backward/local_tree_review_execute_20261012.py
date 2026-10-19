"""Capture before running NEW scientific consumer, retaining every natural outcome."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2);f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_text());root=args.request.parent
    stem=request['epoch'];files=[Path(p) for p in request['pins']]+[args.request]
    for p,sha in request['pins'].items():
        assert digest(Path(p))==sha,p
    contract=json.loads(Path(request['contract_path']).read_text())
    checked={}
    for name,target in contract['target_pins'].items():
        assert digest(Path(target['path']))==target['sha256'],name
        checked[name]=target['sha256']
    before={str(p):digest(p) for p in files}
    pre=root/(stem+'_preexecution.zip')
    with zipfile.ZipFile(pre,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
    assert not (root/(stem+'_receipt.json')).exists()
    command=[sys.executable,str(root/request['consumer']),'--request',str(args.request),'--output',str(root/(stem+'_receipt.json'))]
    start=time.time()
    process=subprocess.run(command,cwd=root,capture_output=True)
    duration=time.time()-start
    stdout=root/(stem+'_stdout.txt');stderr=root/(stem+'_stderr.txt')
    with stdout.open('xb') as f:f.write(process.stdout)
    with stderr.open('xb') as f:f.write(process.stderr)
    after={str(p):digest(p) for p in files}
    exitpath=root/(stem+'_natural_exit.json')
    runtime={'executable':sys.executable,'python':sys.version,'platform':platform.platform(),'sympy':__import__('sympy').__version__,'numpy':__import__('numpy').__version__}
    save(exitpath,{'kind':'NEW_INDEPENDENT_NATIVE_CONSUMER_NATURAL_EXIT','command':command,'exit_code':process.returncode,'natural_exit':True,'timeout':None,'runtime_seconds':duration,'runtime':runtime,'preexecution':{'path':pre.name,'sha256':digest(pre)},'before':before,'after':after,'immutable_inputs_unchanged':before==after,'all_original_contract_target_pins_checked':checked,'stdout_sha256':digest(stdout),'stderr_sha256':digest(stderr),'scientific_admission':False,'campaign_complete':False})
    executed=root/(stem+'_executed.zip')
    outputs=[stdout,stderr,exitpath]
    receipt=root/(stem+'_receipt.json')
    if receipt.exists():outputs.append(receipt)
    with zipfile.ZipFile(executed,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files+outputs:z.write(p,p.name)
        z.write(pre,pre.name)
    verdict={'kind':'REVISION_BOUND_INDEPENDENT_LOCAL_TREE_KYFAN_REVIEW_VERDICT','status':'exact_native_controls_passed' if process.returncode==0 else 'natural_scientific_consumer_failure_preserved','exit_code':process.returncode,'analytical_proof':request['proof'],'analytical_verdict':'all-field inequality valid with explicit average/measurable translated-partition quantum bracket','first_gate':'failed capped boundary gate, not no-go','native_receipt':receipt.name if receipt.exists() else None,'originals_unchanged':before==after,'scope':'Independent analytical N-uniform comparison plus exact noncommuting cube/cap/tree/face-conjugation/AP/open-h/rank/global canonical controls; no ground or patch minimum/phase/IR adoption.','runtime':runtime,'scientific_admission':False,'campaign_complete':False}
    verdictpath=root/(stem+'_verdict.json');save(verdictpath,verdict)
    delivery={'kind':'INDEPENDENT_NATIVE_REVIEW_DELIVERY','exit_code':process.returncode,'runtime_seconds':duration,'runtime':runtime,'governor':request['governor'],'contract_sha256':request['pins'][request['contract_path']],'original_snapshot_sha256':request['snapshot_sha256'],'artifacts':{p.name:{'path':str(p),'sha256':digest(p)} for p in files+outputs+[pre,executed,verdictpath]},'frozen_targets_checked':checked,'no_original_cli_called':True,'no_claims':['body rigidity','Clifford support','patch minimum','phase','sharp IR','ground stationarity','scientific admission','SOURCE adoption'],'scientific_admission':False,'campaign_complete':False}
    deliverypath=root/(stem+'_delivery.json');save(deliverypath,delivery)
    print(json.dumps({'delivery':str(deliverypath),'delivery_sha256':digest(deliverypath),'executed_sha256':digest(executed),'exit_code':process.returncode,'runtime_seconds':duration}))
    raise SystemExit(process.returncode)


if __name__=='__main__':
    main()

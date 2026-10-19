"""Capture immutable NEW all-time analytic CLI epochs and naturally completed exit."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def capsule(path, members):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in dict.fromkeys(members):
            archive.write(ROOT/name,name)

def import_closure(seeds):
    seen, pending = set(), list(seeds)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name.endswith('.py'):
            for node in ast.walk(ast.parse((ROOT/name).read_text())):
                modules = [v.name for v in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
                for module in modules:
                    local = module.split('.')[0]+'.py'
                    if (ROOT/local).is_file() and local not in seen:
                        pending.append(local)
    return sorted(seen)

def run(epoch):
    prefix = f'analytic_flux_time_{epoch}_'
    if any(ROOT.glob(prefix+'*')):
        raise RuntimeError('Immutable analytic epoch exists; use NEW epoch')
    interface_name = 'charged_cell_heat_interface_20261011.json'
    interface = json.loads((ROOT/interface_name).read_text())
    sources = import_closure(['analytic_flux_time_scientist_20261011.py','analytic_flux_time_execute_20261011.py'])
    assert not any(name.startswith('flux_laplace_') for name in sources)
    names = list(dict.fromkeys([*sources,'analytic_flux_time_proof_20261011.json',interface_name,'charged_cell_heat_proof_20261011.json','flux_laplace_proof_20261011.json',*interface['input_sha256']]))
    hashes = {name:sha(ROOT/name) for name in names}
    assert hashes[interface_name]=='41d524542ded29c183426646bbdecf2f34b613bee7dcd13bf98e2a427c9c330f'
    for name, expected in interface['input_sha256'].items():
        assert hashes[name]==expected, ('input changed',name)
    receipt = prefix+'receipt.json'
    argv = [sys.executable,str(ROOT/'analytic_flux_time_scientist_20261011.py'),'--output',str(ROOT/receipt)]
    request = {'kind':'NEW_ANALYTIC_ALLTIME_NATIVE_EXECUTION','epoch':epoch,'argv':argv,'source_input_sha256':hashes,'governor':'09fba2e1e86dc1f15f08e797760118b6cd91aa01c1f8567cba541f706e11afa1','no_old_scientific_CLI_reruns':True,'time_started':time.time()}
    req = prefix+'request.json'
    (ROOT/req).write_text(json.dumps(request,indent=2)+'\n')
    pre = ROOT/(prefix+'preexecution.zip')
    capsule(pre,[*names,req])
    env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (ROOT/(prefix+'stdout.txt')).open('xb') as stdout,(ROOT/(prefix+'stderr.txt')).open('xb') as stderr:
        process = subprocess.Popen(argv,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        code = process.wait()
    after = {name:sha(ROOT/name) for name in names}
    artifacts = [p.name for p in ROOT.glob(prefix+'*') if p.is_file() and p.suffix!='.zip']
    exit_name = prefix+'exit.json'
    record = {'kind':'NEW_ANALYTIC_NATIVE_NATURAL_EXIT','pid':process.pid,'returncode':code,'natural_exit':code>=0,'source_input_sha256':hashes,'sources_inputs_unchanged':after==hashes,'preexecution_sha256':sha(pre),'artifact_sha256':{name:sha(ROOT/name) for name in artifacts},'receipt_present':(ROOT/receipt).is_file(),'scientific_execution_completed':code==0 and (ROOT/receipt).is_file(),'elapsed_seconds':time.time()-request['time_started']}
    (ROOT/exit_name).write_text(json.dumps(record,indent=2)+'\n')
    executed = ROOT/(prefix+'executed.zip')
    capsule(executed,[*names,*artifacts,exit_name])
    print(json.dumps({'exit':exit_name,'returncode':code,'natural_exit':code>=0,'executed_sha256':sha(executed),'preexecution_sha256':sha(pre),'sources_inputs_unchanged':after==hashes}),flush=True)
    return code

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.epoch))

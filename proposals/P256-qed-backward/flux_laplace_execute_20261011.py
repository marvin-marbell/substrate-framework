"""Immutable NEW native Laplace consumer epochs; preserve every natural failure."""
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

def capsule(path,members):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in dict.fromkeys(members):
            archive.write(ROOT/name,name)

def imported_sources(seeds):
    """AST-only local import closure; never execute a historical module CLI."""
    seen = set()
    pending = list(seeds)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if not name.endswith('.py'):
            continue
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules = [v.name for v in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for module in modules:
                local = module.split('.')[0]+'.py'
                if (ROOT/local).is_file() and local not in seen:
                    pending.append(local)
    return sorted(seen)

def run(epoch,interface_name,helper_name):
    prefix = f'flux_laplace_{epoch}_'
    if any(ROOT.glob(prefix+'*')):
        raise RuntimeError('Immutable epoch exists; select a NEW epoch')
    interface = json.loads((ROOT/interface_name).read_text())
    sources = imported_sources(['flux_laplace_scientist_20261011.py','flux_laplace_execute_20261011.py',helper_name+'.py'])
    inputs = ['flux_laplace_time_request_20261011.json','flux_laplace_proof_20261011.json',interface_name,'charged_first_shell_actual04_20261011_receipt.json','joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz','local_so5_ground_upper_complete_state_consumer_20261011_receipt.json',*interface['input_sha256']]
    names = list(dict.fromkeys([*sources,*inputs]))
    hashes = {name:sha(ROOT/name) for name in names}
    for name,expected in interface['input_sha256'].items():
        assert hashes[name]==expected, ('heat input changed',name)
    assert hashes[helper_name+'.py']==interface['helper_sha256']
    receipt = prefix+'receipt.json'
    argv = [sys.executable,str(ROOT/'flux_laplace_scientist_20261011.py'),'--interface',str(ROOT/interface_name),'--helper',helper_name,'--output',str(ROOT/receipt)]
    request = {'kind':'NEW_NATIVE_SIGNED_COHERENT_FLUX_LAPLACE_EXECUTION','epoch':epoch,'argv':argv,'source_input_sha256':hashes,'captured_governor':'8115649c915772d522c604be80732ccebdb58383b34cfa1738e0466466d429d9','accepted_scoped_science':'6f65d328398ea35d60e7e9b64d8c1631832a3a7746b0d22f797b8cafc6f88693','no_old_scientific_CLI_reruns':True,'time_started':time.time()}
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
    ex = prefix+'exit.json'
    record = {'kind':'NEW_NATIVE_FLUX_LAPLACE_NATURAL_EXIT','pid':process.pid,'returncode':code,'natural_exit':code>=0,'source_input_sha256':hashes,'sources_inputs_unchanged':after==hashes,'preexecution_sha256':sha(pre),'artifact_sha256':{name:sha(ROOT/name) for name in artifacts},'receipt_present':(ROOT/receipt).is_file(),'scientific_execution_completed':code==0 and (ROOT/receipt).is_file(),'elapsed_seconds':time.time()-request['time_started']}
    (ROOT/ex).write_text(json.dumps(record,indent=2)+'\n')
    executed = ROOT/(prefix+'executed.zip')
    capsule(executed,[*names,*artifacts,ex])
    print(json.dumps({'exit':ex,'returncode':code,'natural_exit':code>=0,'executed_sha256':sha(executed),'preexecution_sha256':sha(pre),'sources_inputs_unchanged':after==hashes}),flush=True)
    return code

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',required=True)
    parser.add_argument('--interface',default='charged_cell_heat_interface_20261011.json')
    parser.add_argument('--helper',default='charged_cell_heat_helper_20261011')
    args = parser.parse_args()
    raise SystemExit(run(args.epoch,args.interface,args.helper))

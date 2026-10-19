"""Immutable per-epoch execution of NEW four-edge mathematical consumer only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import ground_flux_cycle_scientist_20261011 as scientist
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def capsule(path,members):
 with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
  for name in dict.fromkeys(members):archive.write(ROOT/name,name)

def run(epoch):
 prefix=f'ground_flux_cycle_{epoch}_'
 if any(ROOT.glob(prefix+'*')):raise RuntimeError('Immutable epoch already exists; choose a new epoch')
 sources=['ground_flux_cycle_scientist_20261011.py','ground_flux_cycle_execute_20261011.py']
 inputs=scientist.INPUTS
 hashes={name:sha(ROOT/name) for name in [*sources,*inputs]}
 request={'kind':'NEW_ACTUAL_G_FOUR_EDGE_FLUX_EXECUTION_REQUEST','epoch':epoch,
  'python':sys.executable,'argv':[sys.executable,str(ROOT/sources[0]),'--epoch',epoch],
  'source_input_sha256':hashes,'contract_sha256':'499ee3972a8f8c88b6a3e98148af3299c53d65cb9c74a561bb05ea3313d947d6',
  'captured_plan_revision':'2f78f9876289c8d9411f62e04f8a4f295cb829cca71d92c6aa87a6d44b19723d',
  'launch_bookkeeping_only':'8115649c915772d522c604be80732ccebdb58383b34cfa1738e0466466d429d9',
  'forbidden_old_cli_reruns':True,'time_started':time.time()}
 req=prefix+'request.json';(ROOT/req).write_text(json.dumps(request,indent=2)+'\n')
 pre=ROOT/(prefix+'preexecution.zip');capsule(pre,[*sources,*inputs,req])
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 with (ROOT/(prefix+'stdout.txt')).open('xb') as stdout,(ROOT/(prefix+'stderr.txt')).open('xb') as stderr:
  process=subprocess.Popen(request['argv'],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
  code=process.wait()
 after={name:sha(ROOT/name) for name in hashes}
 artifacts=[p.name for p in ROOT.glob(prefix+'*') if p.is_file() and p.suffix!='.zip']
 receipt=ROOT/(prefix+'receipt.json')
 exit_record={'kind':'NEW_FLUX_CLI_NATURAL_EXIT','epoch':epoch,'pid':process.pid,
  'returncode':code,'natural_exit':code>=0,'elapsed_seconds':time.time()-request['time_started'],
  'sources_inputs_unchanged':after==hashes,'preexecution_sha256':sha(pre),
  'source_input_sha256':hashes,'artifact_sha256':{name:sha(ROOT/name) for name in artifacts},
  'receipt_present':receipt.exists(),'scientific_pass':code==0 and receipt.exists()}
 ex=prefix+'exit.json';(ROOT/ex).write_text(json.dumps(exit_record,indent=2)+'\n')
 executed=ROOT/(prefix+'executed.zip');capsule(executed,[*sources,*inputs,*artifacts,ex])
 print(json.dumps({'exit':ex,'returncode':code,'natural_exit':code>=0,'executed_sha256':sha(executed),'preexecution_sha256':sha(pre),'sources_inputs_unchanged':after==hashes}),flush=True)
 return code

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True);raise SystemExit(run(parser.parse_args().epoch))

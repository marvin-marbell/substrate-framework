"""Exclusive NEW charged-vector scientific execution with immutable before/after ZIPs."""
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
import charged_outer_vector_scientist_20261011 as science
ROOT=Path(__file__).resolve().parent
PREFIX='charged_outer_vector_'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def capsule(p,members):
 with zipfile.ZipFile(p,'x',zipfile.ZIP_DEFLATED) as z:
  for n in sorted(set(members)):z.write(ROOT/n,n)
def run(epoch):
 t=time.monotonic();modules=[]
 for m in tuple(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if f:
   p=Path(f).resolve()
   if p.parent==ROOT and p.suffix=='.py':modules.append(p.name)
 members=sorted(set(science.INPUTS+modules+[Path(__file__).name,'charged_outer_vector_language_20261011.json']))
 pins={n:sha(ROOT/n) for n in members}
 paths={k:ROOT/f'{PREFIX}{epoch}_{k}{suffix}' for k,suffix in [('request','.json'),('preexecution','.zip'),('executed','.zip'),('stdout','.txt'),('stderr','.txt'),('receipt','.json'),('exit','.json')]}
 cmd=[PYTHON,'charged_outer_vector_scientist_20261011.py','--epoch',epoch]
 dump(paths['request'],{'kind':'NEW_NATIVE_CHARGED_OPERATOR_VECTOR_REQUEST','command':cmd,'input_and_source_sha256':pins,'no_historical_entrypoint':True,'epoch':epoch})
 capsule(paths['preexecution'],members+[paths['request'].name])
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
 with paths['stdout'].open('x') as out,paths['stderr'].open('x') as err:
  done=subprocess.run(cmd,cwd=ROOT,env=env,stdout=out,stderr=err,check=False)
 unchanged=all(sha(ROOT/n)==v for n,v in pins.items())
 outputs=[paths[k].name for k in ('request','stdout','stderr')]
 for n in (paths['receipt'].name,'charged_outer_vector_interface_20261011.json'):
  if (ROOT/n).exists():outputs.append(n)
 capsule(paths['executed'],members+outputs)
 result={'kind':'NEW_CHARGED_VECTOR_NATURAL_EXECUTION_EXIT','exit_code':done.returncode,'natural_subprocess_exit':True,'unchanged':unchanged,
 'input_and_source_sha256':pins,'preexecution_zip':paths['preexecution'].name,'preexecution_sha256':sha(paths['preexecution']),
 'executed_zip':paths['executed'].name,'executed_sha256':sha(paths['executed']),
 'output_sha256':{n:sha(ROOT/n) for n in outputs},'elapsed_seconds':time.monotonic()-t}
 dump(paths['exit'],result);print(json.dumps(result),flush=True)
 if done.returncode or not unchanged:raise SystemExit(done.returncode or 2)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epoch',required=True);run(p.parse_args().epoch)

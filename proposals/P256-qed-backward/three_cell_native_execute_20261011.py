"""Immutable NEW scientific CLI epochs with natural exit and exact byte capsules."""
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
ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def archive(path,members):
 with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
  for name,raw in sorted(members.items()):z.writestr(name,raw)
 path.chmod(0o444)
def main():
 p=argparse.ArgumentParser();p.add_argument('--epoch',required=True);p.add_argument('--N',type=int,default=8);a=p.parse_args()
 stem='three_cell_native_'+a.epoch
 paths={Path(__file__).name, 'three_cell_native_scientist_20261011.py'};pending=list(paths)
 while pending:
  name=pending.pop();tree=ast.parse((ROOT/name).read_bytes())
  for node in ast.walk(tree):
   names=[x.name for x in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
   for module in names:
    child=module.split('.')[0]+'.py'
    if (ROOT/child).is_file() and child not in paths:paths.add(child);pending.append(child)
 scientific=ast.parse((ROOT/'three_cell_native_scientist_20261011.py').read_bytes())
 # Explicit immutable input set, independent of executing the science module.
 for node in scientific.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in node.targets):
   for key in node.value.keys:
    if isinstance(key,ast.Constant):paths.add(key.value)
 paths.add('connected_root_exterior_actual1_20261011_moments.npz')
 paths.add('local_so5_resum_actual04_20261011_state.npz')
 paths.add('local_so5_resum_actual04_20261011_receipt.json')
 members={name:(ROOT/name).read_bytes() for name in sorted(paths)}
 before={name:sha(raw) for name,raw in members.items()}
 command=[PYTHON,'three_cell_native_scientist_20261011.py','--epoch',a.epoch,'--N',str(a.N)]
 request={'command':command,'cwd':str(ROOT),'SHA_before':before,'plan_id':'issue-222-induced-photon-delta13-continuation-20261008',
 'governor':'4512d705460cfed254b28d79ab32f44dd1f64701a043d859d3f62cb2c0af0f0c','N':a.N}
 raw=(json.dumps(request,indent=2)+'\n').encode();members[stem+'_request.json']=raw
 (ROOT/(stem+'_request.json')).write_bytes(raw)
 pre=ROOT/(stem+'_preexecution.zip');archive(pre,members)
 started=time.monotonic()
 with (ROOT/(stem+'_stdout.txt')).open('xb') as out,(ROOT/(stem+'_stderr.txt')).open('xb') as err:
  env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
  child=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,env=env,check=False)
 after={name:sha((ROOT/name).read_bytes()) for name in before}
 receipt=ROOT/(stem+'_receipt.json')
 for name in [stem+'_stdout.txt',stem+'_stderr.txt',stem+'_receipt.json']:
  path=ROOT/name
  if path.is_file():members[name]=path.read_bytes()
 exitinfo={'natural_exit':True,'returncode':child.returncode,'elapsed_seconds':time.monotonic()-started,
 'SHA_before':before,'SHA_after':after,'all_sources_inputs_unchanged':before==after,
 'preexecution_archive':pre.name,'preexecution_SHA':sha(pre.read_bytes()),'receipt_exists':receipt.is_file()}
 members[stem+'_exit.json']=(json.dumps(exitinfo,indent=2)+'\n').encode()
 executed=ROOT/(stem+'_executed.zip');archive(executed,members)
 exitinfo['executed_archive']=executed.name;exitinfo['executed_SHA']=sha(executed.read_bytes())
 (ROOT/(stem+'_exit.json')).write_text(json.dumps(exitinfo,indent=2)+'\n')
 print(json.dumps({k:v for k,v in exitinfo.items() if k not in ('SHA_before','SHA_after')}),flush=True)
 sys.exit(child.returncode)
if __name__=='__main__':main()

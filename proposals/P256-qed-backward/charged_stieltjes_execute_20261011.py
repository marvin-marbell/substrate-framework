"""Immutable preexecution/executed capsules for one genuinely new charged CLI."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='charged_stieltjes_'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(path,value):
 with path.open('x') as f:f.write(json.dumps(value,indent=2)+'\n')

def archive(path,files):
 with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for p in files:z.write(p,p.name)

def run(epoch):
 spec=importlib.util.spec_from_file_location('charged_stieltjes_source',ROOT/'charged_stieltjes_scientist_20261011.py')
 assert spec and spec.loader
 source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
 inputs=[ROOT/n for n in source.INPUTS]
 files=[*inputs,ROOT/'charged_stieltjes_scientist_20261011.py',Path(__file__),ROOT/'charged_stieltjes_proof_20261011.json',ROOT/'charged_stieltjes_lsp_ast_20261011.json']
 base=ROOT/f'{PREFIX}{epoch}'
 request=Path(str(base)+'_request.json');pre=Path(str(base)+'_preexecution.zip')
 stdout=Path(str(base)+'_stdout.txt');stderr=Path(str(base)+'_stderr.txt');exitfile=Path(str(base)+'_exit.json')
 receipt=Path(str(base)+'_receipt.json');executed=Path(str(base)+'_executed.zip')
 assert not any(p.exists() for p in [request,pre,stdout,stderr,exitfile,receipt,executed]), 'epoch is immutable; choose a new epoch'
 command=[sys.executable,str(ROOT/'charged_stieltjes_scientist_20261011.py'),'--epoch',epoch]
 dump(request,{'command':command,'input_and_source_sha256':{p.name:sha(p) for p in files},'old_main_execution':False,'scope':'one actual charged dynamical edge only'})
 archive(pre,[*files,request]);started=time.time()
 environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with stdout.open('xb') as out,stderr.open('xb') as err:
  result=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,env=environment,check=False)
 natural={'command':command,'returncode':result.returncode,'natural_exit':True,'elapsed_seconds':time.time()-started,
  'preexecution_sha256':sha(pre),'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr),
  'source_and_inputs_unchanged':all(sha(p)==json.loads(request.read_text())['input_and_source_sha256'][p.name] for p in files),
  'receipt_sha256':sha(receipt) if receipt.exists() else None}
 dump(exitfile,natural)
 archive(executed,[pre,request,stdout,stderr,exitfile,*([receipt] if receipt.exists() else [])])
 print(json.dumps({'exit':natural,'executed_zip':executed.name,'executed_sha256':sha(executed)},indent=2),flush=True)
 return result.returncode

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True)
 sys.exit(run(parser.parse_args().epoch))

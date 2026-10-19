"""Preserve independent audit source/input/pre/executed/natural-exit bindings."""
import argparse
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import zipfile
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--epoch',required=True)
STEM=f'charged_kernel_review_{parser.parse_args().epoch}'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs=json.loads((ROOT/'charged_stieltjes_actual1_20261011_request.json').read_text())['input_and_source_sha256']
inputs.update({n:sha(ROOT/n) for n in ('charged_stieltjes_actual1_20261011_request.json','charged_stieltjes_actual1_20261011_receipt.json','charged_stieltjes_actual1_20261011_exit.json','charged_stieltjes_actual1_20261011_executed.zip','charged_kernel_review_control_20261011.py','charged_kernel_review_execute_20261011.py','charged_kernel_review_language_20261011.json','joined_phi_u1_physical_charged_canonical_neutral_scoped_acceptance_20261011.json','affine_lower_mathreview_proof_20261011.json','affine_lower_mathreview_verdict_20261011.json')})
inputs.update({n:sha(ROOT/n) for n in ('charged_kernel_review_actual1_20261011_capsule.json','charged_kernel_review_actual1_20261011_executed.zip','charged_kernel_review_actual2_20261011_capsule.json','charged_kernel_review_actual2_20261011_executed.zip')})
inputs.update({n:sha(ROOT/n) for n in ('charged_kernel_review_actual3_20261011_capsule.json','charged_kernel_review_actual3_20261011_executed.zip')})
assert all(sha(ROOT/n)==v for n,v in inputs.items())
command=[PYTHON,str(ROOT/'charged_kernel_review_control_20261011.py'),'--output',str(ROOT/(STEM+'_receipt.json'))]
request={'command':command,'input_and_source_sha256':inputs,'captured_governor':'2f78f9876289c8d9411f62e04f8a4f295cb829cca71d92c6aa87a6d44b19723d','owner_bookkeeping_only':'8115649c915772d522c604be80732ccebdb58383b34cfa1738e0466466d429d9','old_cli_execution':False,'independent_new_control':True}
def jsonnew(name,data):
 p=ROOT/name
 with p.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
 return p
request_path=jsonnew(STEM+'_request.json',request)
pre=ROOT/(STEM+'_preexecution.zip')
with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for n in inputs:z.write(ROOT/n,n)
 z.write(request_path,request_path.name)
os.chmod(pre,0o444)
start=time.monotonic()
stdout=ROOT/(STEM+'_stdout.txt');stderr=ROOT/(STEM+'_stderr.txt')
with stdout.open('xb') as out,stderr.open('xb') as err:
 p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err)
 returncode=p.wait()
receipt=ROOT/(STEM+'_receipt.json')
exit_data={'command':command,'returncode':returncode,'natural_exit':True,'elapsed_seconds':time.monotonic()-start,'preexecution_sha256':sha(pre),'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr),'source_and_inputs_unchanged':all(sha(ROOT/n)==v for n,v in inputs.items()),'receipt_sha256':sha(receipt) if receipt.exists() else None}
exitpath=jsonnew(STEM+'_exit.json',exit_data)
executed=ROOT/(STEM+'_executed.zip')
with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for n in inputs:z.write(ROOT/n,n)
 for p in (request_path,stdout,stderr,exitpath):z.write(p,p.name)
 if receipt.exists():z.write(receipt,receipt.name)
os.chmod(executed,0o444)
summary={'returncode':returncode,'natural_exit':True,'preexecution_sha256':sha(pre),'executed_sha256':sha(executed),'exit_sha256':sha(exitpath),'receipt_sha256':exit_data['receipt_sha256'],'source_and_inputs_unchanged':exit_data['source_and_inputs_unchanged']}
jsonnew(STEM+'_capsule.json',summary)
print(json.dumps(summary,indent=2),flush=True)
raise SystemExit(returncode)

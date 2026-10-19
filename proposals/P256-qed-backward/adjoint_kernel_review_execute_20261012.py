"""Archive before running only the new independent scientific consumer."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parent
EPOCH='adjoint_kernel_review_actual1_20261012'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
CONTRACT=Path('/home/administrator/.omp/agent/sessions/-/2026-10-09T16-26-59-293Z_01a1217d-415d-7376-a17d-16be721cbe28/local/issue222-native-adjoint-kernel-full-loop-energy-form-review-contract-20261012.json')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    names=['adjoint_kernel_review_consumer_20261012.py','adjoint_kernel_review_execute_20261012.py',
           'adjoint_kernel_review_language_20261012.json','ground_loop_support_parent_review_postexecution_snapshot_20261012.zip',
           'closed_inverse_review_proof_20261012.json','closed_beta_inverse_proof_20261012.json','reset_gauss_review_proof_20261011.json',
           'native_loop_dark_auxiliary_proof_20261012.json','full_loop_graph_proof_20261012.json']
    payload={name:(ROOT/name).read_bytes() for name in names};payload['contract.json']=CONTRACT.read_bytes()
    assert sha(payload['contract.json'])=='b0f8c1185eebf5550b1b4ccc623868582bd9abd0064f3214a2afb17926930330'
    assert sha(payload['ground_loop_support_parent_review_postexecution_snapshot_20261012.zip'])=='5a7adf2cc6f8721a2fbcf14e7cdf850b0583c5fbab1aeb36cdd58dd48001cb17'
    pre=ROOT/(EPOCH+'_preexecution.zip');request_path=ROOT/(EPOCH+'_request.json');output=ROOT/(EPOCH+'_receipt.json')
    request={'epoch':EPOCH,'preexecution':str(pre),'pins':{name:sha(raw) for name,raw in payload.items()},
             'plan_id':'issue-222-induced-photon-delta13-continuation-20261008','revision':'c65d409ab76891a52043391a5637e80041f453aa60d4a0634ffe87f265d716d6',
             'template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
             'work_item':'induced-photon','requirements':['outcome','verification'],'no_historical_main':True}
    request_raw=(json.dumps(request,indent=2)+'\n').encode();request_path.write_bytes(request_raw)
    with zipfile.ZipFile(pre,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in payload.items():archive.writestr(name,raw)
        archive.writestr('request.json',request_raw)
    before=sha(pre.read_bytes());start=time.monotonic();env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    run=subprocess.run([PYTHON,str(ROOT/'adjoint_kernel_review_consumer_20261012.py'),'--request',str(request_path),'--output',str(output)],cwd=ROOT,env=env,capture_output=True)
    stdout=ROOT/(EPOCH+'_stdout.txt');stderr=ROOT/(EPOCH+'_stderr.txt');exit_path=ROOT/(EPOCH+'_exit.json')
    stdout.write_bytes(run.stdout);stderr.write_bytes(run.stderr)
    unchanged=all((CONTRACT if name=='contract.json' else ROOT/name).read_bytes()==raw for name,raw in payload.items())
    exit_data={'returncode':run.returncode,'natural_exit':True,'elapsed_seconds':time.monotonic()-start,'inputs_unchanged':unchanged,'preexecution_unchanged':before==sha(pre.read_bytes())}
    exit_path.write_text(json.dumps(exit_data,indent=2)+'\n')
    executed=ROOT/(EPOCH+'_executed.zip')
    with zipfile.ZipFile(executed,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in payload.items():archive.writestr(name,raw)
        archive.writestr('request.json',request_raw);archive.writestr(pre.name,pre.read_bytes())
        for f in (stdout,stderr,exit_path,output):
            if f.exists():archive.writestr(f.name,f.read_bytes())
    capsule={'epoch':EPOCH,'preexecution_sha256':before,'executed_sha256':sha(executed.read_bytes()),'exit':exit_data,
             'pins':request['pins'],'source':names[0],'stdout_sha256':sha(run.stdout),'stderr_sha256':sha(run.stderr),
             'receipt_sha256':sha(output.read_bytes()) if output.exists() else None}
    (ROOT/(EPOCH+'_capsule.json')).write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps(capsule));raise SystemExit(run.returncode)


if __name__=='__main__':main()

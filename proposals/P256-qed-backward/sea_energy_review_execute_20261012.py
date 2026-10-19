"""Capture independent review inputs before executing their exact bytes; retain failures."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
CONTRACT=Path('/home/administrator/.omp/agent/sessions/-/2026-10-09T16-26-59-293Z_01a1217d-415d-7376-a17d-16be721cbe28/local/issue222-connected-sea-local-body-independent-review-contract-20261012.json')
CONTRACT_SHA='8a74e4413b8bb6a8de63ad55f685f0ec2fede2b7568ae14e9acc5adf7af96840'
SOURCE='sea_energy_review_source_20261012.py'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(payload):return (json.dumps(payload,indent=2)+'\n').encode()
def capsule(path,members):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as out:
        for name,raw in sorted(members.items()):out.writestr(name,raw)
    path.chmod(0o444)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True);args=parser.parse_args()
    prefix='sea_energy_review_'+args.epoch+'_20261012'
    assert not list(ROOT.glob(prefix+'*')), 'Use a new immutable epoch after every attempt.'
    raw=CONTRACT.read_bytes();assert sha(raw)==CONTRACT_SHA;contract=json.loads(raw)
    snapshot=(ROOT/contract['snapshot']['path']).read_bytes();assert sha(snapshot)==contract['snapshot']['sha256']
    members={CONTRACT.name:raw,SOURCE:(ROOT/SOURCE).read_bytes(),Path(__file__).name:Path(__file__).read_bytes()}
    archive=zipfile.ZipFile(io.BytesIO(snapshot))
    for name in ['connected_quantum_sea_actual2_20261012_proof.json','connected_quantum_sea_actual2_20261012_interface.json','connected_quantum_sea_source_20261012.py','connected_quantum_sea_actual2_20261012_natural_exit.json','issue222-connected-quantum-sea-local-leading-contract-20261012.json']:
        data=archive.read(name);assert sha(data)==contract['target_pins'][name]['sha256'];members[name]=data
    original_executed=archive.read('connected_quantum_sea_actual2_20261012_executed.zip')
    assert sha(original_executed)==contract['target_pins']['connected_quantum_sea_actual2_20261012_executed.zip']['sha256']
    original=zipfile.ZipFile(io.BytesIO(original_executed));profile='neutral_body_parent_primitive_profiles_20261012.json'
    members[profile]=original.read(profile)
    assert sha(members[profile])==json.loads(members['connected_quantum_sea_actual2_20261012_proof.json'])['bound_pins'][profile]
    language=ROOT/'sea_energy_review_language_20261012.json'
    if language.exists():members[language.name]=language.read_bytes()
    pins={name:sha(data) for name,data in members.items()}
    plan={k:contract[k] for k in ['plan_id','captured_revision','accepted_scoped_science','template_id','template_revision','work_item_id','requirement_ids']}
    request={'pins':pins,'plan':plan,'prefix':prefix,'original_snapshot_sha256':sha(snapshot),'original_executed_zip_sha256':sha(original_executed),'no_original_functions_or_CLI_invoked':True}
    members[prefix+'_request.json']=encoded(request)
    versions=subprocess.run([PYTHON,'-c','import platform,sympy,numpy; print(platform.python_version()); print(sympy.__version__); print(numpy.__version__)'],capture_output=True,check=True)
    manifest={'true_preexecution':True,'input_sha256':{name:sha(data) for name,data in members.items()},'python':PYTHON,'Python_SymPy_NumPy_versions':versions.stdout.decode().splitlines(),'plan':plan,'original_postexecution_snapshot':{'sha256':sha(snapshot),'name':contract['snapshot']['path']}}
    members[prefix+'_manifest.json']=encoded(manifest)
    pre=ROOT/(prefix+'_preexecution.zip');capsule(pre,members)
    for name in [prefix+'_request.json',prefix+'_manifest.json']:(ROOT/name).write_bytes(members[name])
    with tempfile.TemporaryDirectory(prefix='sea_energy_review_') as temporary:
        work=Path(temporary)
        for name,data in members.items():(work/name).write_bytes(data)
        command=[PYTHON,'-B',str(work/SOURCE),'--request',str(work/(prefix+'_request.json')),'--output-prefix',str(work/prefix)]
        start=time.time();run=subprocess.run(command,cwd=work,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,check=False);end=time.time()
        logs={prefix+'_stdout.txt':run.stdout,prefix+'_stderr.txt':run.stderr}
        natural={'natural_exit':True,'returncode':run.returncode,'elapsed_seconds':end-start,'started_unix':start,'ended_unix':end,'command':command,'python':PYTHON,'no_timeout_or_signal':run.returncode>=0,'preexecution_sha256':sha(pre.read_bytes()),'source_sha256':pins[SOURCE],'stdout_sha256':sha(run.stdout),'stderr_sha256':sha(run.stderr),'input_sha256':manifest['input_sha256']}
        logs[prefix+'_natural_exit.json']=encoded(natural)
        outputs={p.name:p.read_bytes() for p in work.glob(prefix+'*') if p.name not in members}
        for name,data in {**outputs,**logs}.items():(ROOT/name).write_bytes(data)
        executed=ROOT/(prefix+'_executed.zip');capsule(executed,{**members,**outputs,**logs})
        delivery={'kind':'INDEPENDENT_SCOPED_REVIEW_DELIVERY_NOT_ADMISSION','natural_exit':natural,'preexecution':{'path':pre.name,'sha256':sha(pre.read_bytes())},'executed':{'path':executed.name,'sha256':sha(executed.read_bytes()),'mode':'0444'},'inputs':pins,'outputs':{name:sha(data) for name,data in {**outputs,**logs}.items()},'plan':plan,'failed_attempt_preserved':run.returncode!=0,'original_postexecution_snapshot_sha256':sha(snapshot),'original_executed_zip_sha256':sha(original_executed),'scientific_admission':False}
        (ROOT/(prefix+'_delivery.json')).write_bytes(encoded(delivery));print(json.dumps(delivery,indent=2),flush=True)
        return run.returncode

if __name__=='__main__':sys.exit(main())

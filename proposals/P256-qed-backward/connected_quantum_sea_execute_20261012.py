"""One new connected-sea epoch: snapshot first, execute captured bytes, retain natural exit."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
CONTRACT=Path('/home/administrator/.omp/agent/sessions/-/2026-10-09T16-26-59-293Z_01a1217d-415d-7376-a17d-16be721cbe28/local/issue222-connected-quantum-sea-local-leading-contract-20261012.json')
CONTRACT_SHA='adb0bba26fd5dee8e86ff0bd5ab96acb0df27359b9ac7049c89162ec127dbf11'
SOURCE='connected_quantum_sea_source_20261012.py'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def dump(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def capsule(path,files):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as out:
        for name,raw in sorted(files.items()):out.writestr(name,raw)
    path.chmod(0o444)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True);args=parser.parse_args()
    prefix='connected_quantum_sea_'+args.epoch+'_20261012'
    assert not list(ROOT.glob(prefix+'*')), 'epochs are write-once; select a new epoch after a failed attempt'
    contract_raw=CONTRACT.read_bytes();assert sha(contract_raw)==CONTRACT_SHA
    contract=json.loads(contract_raw)
    inputs={CONTRACT.name:contract_raw,SOURCE:(ROOT/SOURCE).read_bytes(),Path(__file__).name:Path(__file__).read_bytes()}
    for name,descriptor in contract['shared_inputs'].items():
        raw=(ROOT/descriptor['path']).read_bytes();assert sha(raw)==descriptor['sha256'],name;inputs[name]=raw
    pins={name:sha(raw) for name,raw in inputs.items()}
    request={'kind':'NEW_CONNECTED_GLOBAL_SEA_EXACT_CONTROL','epoch':args.epoch,'profiles':'neutral_body_parent_primitive_profiles_20261012.json','pins':pins,'governing_plan':{k:contract[k] for k in ('plan_id','captured_revision','template_id','template_revision','work_item_id','requirement_ids')},'no_original_producer_rerun':True,'ground_backend':False}
    request_name=prefix+'_request.json';inputs[request_name]=(json.dumps(request,indent=2)+'\n').encode()
    versions=subprocess.run([PYTHON,'-c','import platform,numpy,sympy; print(platform.python_version()); print(numpy.__version__); print(sympy.__version__)'],capture_output=True,check=True)
    manifest={'captured_before_execution':True,'input_sha256':{name:sha(raw) for name,raw in inputs.items()},'python_executable':PYTHON,'package_versions_python_numpy_sympy':versions.stdout.decode().splitlines(),'run_source_bytes':SOURCE,'executed_from_snapshot_not_live_tree':True}
    inputs[prefix+'_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    pre=ROOT/(prefix+'_preexecution.zip');capsule(pre,inputs)
    for name in (request_name,prefix+'_manifest.json'):(ROOT/name).write_bytes(inputs[name])
    with tempfile.TemporaryDirectory(prefix='connected_quantum_sea_') as tmp:
        work=Path(tmp)
        for name,raw in inputs.items():(work/name).write_bytes(raw)
        output=work/prefix
        command=[PYTHON,'-B',str(work/SOURCE),'--request',str(work/request_name),'--output-prefix',str(output)]
        start=time.time(); result=subprocess.run(command,cwd=work,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,check=False);end=time.time()
        logs={prefix+'_stdout.txt':result.stdout,prefix+'_stderr.txt':result.stderr}
        natural={'natural_exit':True,'returncode':result.returncode,'started_unix':start,'ended_unix':end,'elapsed_seconds':end-start,'command':command,'preexecution_zip_sha256':sha(pre.read_bytes()),'executed_source_sha256':pins[SOURCE],'executed_input_sha256':manifest['input_sha256'],'stdout_sha256':sha(result.stdout),'stderr_sha256':sha(result.stderr),'signal_or_timeout':False}
        logs[prefix+'_natural_exit.json']=(json.dumps(natural,indent=2)+'\n').encode()
        outputs={path.name:path.read_bytes() for path in work.glob(prefix+'*') if path.name not in inputs}
        for name,raw in {**outputs,**logs}.items():(ROOT/name).write_bytes(raw)
        executed=ROOT/(prefix+'_executed.zip');capsule(executed,{**inputs,**outputs,**logs})
        receipt={'epoch':args.epoch,'natural_exit':natural,'preexecution_zip':{'name':pre.name,'sha256':sha(pre.read_bytes())},'executed_zip':{'name':executed.name,'sha256':sha(executed.read_bytes()),'mode':'0444'},'source_and_input_pins':pins,'output_sha256':{name:sha(raw) for name,raw in {**outputs,**logs}.items()},'capsules_preserved_on_failure':True}
        dump(ROOT/(prefix+'_execution_receipt.json'),receipt)
        print(json.dumps(receipt,indent=2),flush=True)
        return result.returncode

if __name__=='__main__':sys.exit(main())

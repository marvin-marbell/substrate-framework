"""Exclusive immutable independent-review epoch; no historical scientific entrypoint."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
import star_factor_review_scientist_20261011 as review

ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(path,value):
    with path.open('x') as out:json.dump(value,out,indent=2);out.write('\n')
def capture(path,members):
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as archive:
        for member in sorted(set(members)):archive.write(ROOT/member,member)
def run(epoch):
    start=time.monotonic();prefix='star_factor_review_'+epoch+'_'
    modules=[]
    for module in tuple(sys.modules.values()):
        p=Path(getattr(module,'__file__','') or '.').resolve()
        if p.parent==ROOT and p.suffix=='.py':modules.append(p.name)
    members=sorted(set(modules+review.frozen.INPUTS+[Path(__file__).name,'star_factor_review_scientist_20261011.py','star_factor_review_proof_20261011.json','shared_star_dynamic_proof_20261011.json','shared_star_dynamic_actual2_20261011_receipt.json','shared_star_dynamic_actual2_20261011_exit.json','shared_star_dynamic_actual1_20261011_exit.json']))
    pins={name:digest(ROOT/name) for name in members}
    # Validate historical binding before an import-consuming science process.
    assert pins['shared_star_dynamic_scientist_20261011.py']==review.PIN
    assert pins['shared_star_dynamic_actual2_20261011_receipt.json']=='3f690a820fbc84095804d0470386911dd8a9298c396fb9f2c915da870fdfe789'
    request=ROOT/(prefix+'request.json');pre=ROOT/(prefix+'preexecution.zip');executed=ROOT/(prefix+'executed.zip')
    out=ROOT/(prefix+'stdout.txt');err=ROOT/(prefix+'stderr.txt');receipt=ROOT/f'star_factor_review_{epoch}_20261011_receipt.json'
    command=['/home/administrator/substrate-framework/.venv/bin/python','star_factor_review_scientist_20261011.py',epoch]
    dump(request,{'kind':'INDEPENDENT_REVIEW_IMMUTABLE_REQUEST','command':command,'input_and_source_sha256':pins,'no_historical_main_invoked':True,'governing_plan':{'id':'issue-222-induced-photon-delta13-continuation-20261008','captured_revision':'c6e1ec4424d6c94ad36f5ceca2d5bd72fbbff3ec25d964a0c15a2fe9e565f060','accepted_science':'2ec1fe2f4ebfad5032c59dd9ed0d6ef3a57d15e58247f175c670c5034c7d6208','template':'general','template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe','work_item':'induced-photon'}})
    capture(pre,members+[request.name])
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with out.open('x') as stdout,err.open('x') as stderr:
        completed=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,check=False)
    unchanged=all(digest(ROOT/name)==pin for name,pin in pins.items())
    outputs=[request.name,out.name,err.name]+([receipt.name] if receipt.exists() else [])
    capture(executed,members+outputs)
    result={'kind':'INDEPENDENT_REVIEW_NATURAL_EXIT','exit_code':completed.returncode,'natural_subprocess_exit':True,'unchanged':unchanged,'input_and_source_sha256':pins,'preexecution_zip':pre.name,'preexecution_sha256':digest(pre),'executed_zip':executed.name,'executed_sha256':digest(executed),'receipt_sha256':digest(receipt) if receipt.exists() else None,'stdout_sha256':digest(out),'stderr_sha256':digest(err),'elapsed_seconds':time.monotonic()-start}
    dump(ROOT/(prefix+'exit.json'),result);print(json.dumps(result),flush=True)
    if completed.returncode or not unchanged:raise SystemExit(completed.returncode or 2)

if __name__=='__main__':run(sys.argv[1])

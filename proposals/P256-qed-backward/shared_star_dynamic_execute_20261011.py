"""Single NEW scientific epoch with exclusive immutable before/after capsules."""
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
import shared_star_dynamic_scientist_20261011 as science

ROOT=Path(__file__).resolve().parent
PREFIX='shared_star_dynamic_'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n')


def capture(path,members):
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as archive:
        for member in sorted(set(members)):
            archive.write(ROOT/member,member)


def run(epoch):
    started=time.monotonic()
    source='shared_star_dynamic_scientist_20261011.py'
    modules=[]
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.parent==ROOT and path.suffix=='.py':
                modules.append(path.name)
    members=sorted(set(science.INPUTS+modules+[source,Path(__file__).name,'shared_star_dynamic_proof_20261011.json']))
    pins={name:digest(ROOT/name) for name in members}
    command=[PYTHON,source,'--epoch',epoch]
    request=ROOT/f'{PREFIX}{epoch}_request.json'
    pre=ROOT/f'{PREFIX}{epoch}_preexecution.zip'
    executed=ROOT/f'{PREFIX}{epoch}_executed.zip'
    stdout=ROOT/f'{PREFIX}{epoch}_stdout.txt'
    stderr=ROOT/f'{PREFIX}{epoch}_stderr.txt'
    receipt=ROOT/f'{PREFIX}{epoch}_receipt.json'
    exit_path=ROOT/f'{PREFIX}{epoch}_exit.json'
    dump(request,{'kind':'NEW_NATIVE_DYNAMIC_SHARED_STAR_EXECUTION_REQUEST','command':command,'input_and_source_sha256':pins,'no_historical_entrypoint_invoked':True,'epoch':epoch})
    capture(pre,members+[request.name])
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with stdout.open('x') as out,stderr.open('x') as err:
        completed=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err,check=False)
    unchanged=all(digest(ROOT/name)==pin for name,pin in pins.items())
    outputs=[stdout.name,stderr.name,request.name]
    if receipt.exists():
        outputs.append(receipt.name)
    capture(executed,members+outputs)
    result={'kind':'NATURALLY_COMPLETED_NEW_SHARED_STAR_SCIENTIFIC_CLI','epoch':epoch,
            'command':command,'exit_code':completed.returncode,'natural_subprocess_exit':True,
            'input_and_source_sha256':pins,'unchanged':unchanged,
            'preexecution_zip':pre.name,'preexecution_sha256':digest(pre),
            'executed_zip':executed.name,'executed_sha256':digest(executed),
            'stdout_sha256':digest(stdout),'stderr_sha256':digest(stderr),
            'receipt_sha256':digest(receipt) if receipt.exists() else None,
            'elapsed_seconds':time.monotonic()-started}
    dump(exit_path,result)
    print(json.dumps(result),flush=True)
    if completed.returncode or not unchanged:
        raise SystemExit(completed.returncode or 2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch',required=True)
    run(parser.parse_args().epoch)

"""Exclusive attempt records for the independent lift/Gauss review controls."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epoch',required=True)
    args=parser.parse_args()
    prefix=f'lift_gauss_review_{args.epoch}_20261011'
    request_path=ROOT/f'{prefix}_request.json'
    request=json.loads(request_path.read_text())
    before={n:digest(ROOT/n) for n in request['pins']}
    assert before==request['pins']
    output=ROOT/f'{prefix}_receipt.json'
    command=[PYTHON,str(ROOT/'lift_gauss_review_control_20261011.py'),'--request',str(request_path),'--output',str(output)]
    pre=ROOT/f'{prefix}_preexecution.zip'
    with zipfile.ZipFile(pre,'x',zipfile.ZIP_DEFLATED) as archive:
        for name in ['lift_gauss_review_control_20261011.py',Path(__file__).name,request_path.name,'lift_gauss_review_language_20261011.json']:
            archive.writestr(name,(ROOT/name).read_bytes())
    started=time.monotonic()
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    with (ROOT/f'{prefix}_stdout.txt').open('xb') as stdout, (ROOT/f'{prefix}_stderr.txt').open('xb') as stderr:
        process=subprocess.run(command,stdout=stdout,stderr=stderr,cwd=ROOT,env=env,check=False)
    elapsed=time.monotonic()-started
    after={n:digest(ROOT/n) for n in request['pins']}
    capsule={'kind':'NEW_INDEPENDENT_REVIEW_NATURAL_EXIT_CAPSULE_NOT_SCIENCE_ADMISSION','command':command,'returncode':process.returncode,'natural_exit':process.returncode>=0,'elapsed_seconds':elapsed,'source_inputs_unchanged':before==after,'source_input_sha256':before,'preexecution_zip':pre.name,'preexecution_sha256':digest(pre),'stdout_sha256':digest(ROOT/f'{prefix}_stdout.txt'),'stderr_sha256':digest(ROOT/f'{prefix}_stderr.txt'),'receipt':output.name if output.exists() else None,'receipt_sha256':digest(output) if output.exists() else None}
    capsule_path=ROOT/f'{prefix}_capsule.json'
    with capsule_path.open('x') as file:
        json.dump(capsule,file,indent=2);file.write('\n')
    executed=ROOT/f'{prefix}_executed.zip'
    with zipfile.ZipFile(executed,'x',zipfile.ZIP_DEFLATED) as archive:
        for path in [pre,capsule_path,ROOT/f'{prefix}_stdout.txt',ROOT/f'{prefix}_stderr.txt']+([output] if output.exists() else []):
            archive.writestr(path.name,path.read_bytes())
    print(json.dumps({**capsule,'executed_zip':executed.name,'executed_sha256':digest(executed)},indent=2),flush=True)


if __name__=='__main__':
    main()

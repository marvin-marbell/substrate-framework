"""Run only the new bulk flux consumer and preserve exact pre/post bytes."""
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
import joined_phi_u1_bulk_flux_ring_consumer as consumer

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
PREFIX=consumer.PREFIX

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def run(epoch: str) -> dict:
    label=PREFIX+epoch
    pre=ROOT/(label+'_preexecution.zip')
    assert not pre.exists(), 'Immutable epoch already exists'
    captured=consumer.snapshot()
    captured[Path(__file__).name]=Path(__file__).read_bytes()
    for name,pin in consumer.PINS.items():
        assert sha(captured[name])==pin
    with zipfile.ZipFile(pre,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in captured.items():
            archive.writestr('preexecution/'+name,raw)
    command=[PYTHON,str(ROOT/(PREFIX+'consumer.py')),'--output',str(ROOT/(label+'_actual.json'))]
    started=time.monotonic()
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    child=subprocess.run(command,cwd=ROOT,capture_output=True,check=False,env=environment)
    elapsed=time.monotonic()-started
    (ROOT/(label+'_stdout.txt')).write_bytes(child.stdout)
    (ROOT/(label+'_stderr.txt')).write_bytes(child.stderr)
    unchanged=all((ROOT/name).read_bytes()==raw for name,raw in captured.items() if '/' not in name)
    result=dict(kind='NEW_NATIVE_BULK_FLUX_SCIENTIFIC_NATURAL_EXIT',command=command,cwd=str(ROOT),project_python=PYTHON,
                natural_exit=child.returncode,exit_was_natural=True,elapsed_seconds=elapsed,stderr_empty=not child.stderr,
                local_source_input_bytes_unchanged=unchanged,preexecution_archive=pre.name,preexecution_sha256=sha(pre.read_bytes()),
                preexecution_members_sha256={name:sha(raw) for name,raw in captured.items()},
                stdout_sha256=sha(child.stdout),stderr_sha256=sha(child.stderr),checks_run='Only new assigned scientific consumer; no prior producers/checkers, builds, lint or pytest')
    if child.returncode==0:
        actual=ROOT/(label+'_actual.json')
        result['actual_receipt_sha256']=sha(actual.read_bytes())
        result['actual_executed_source_input_archive_sha256']=sha(actual.with_suffix('.zip').read_bytes())
        result['actual_executed_source_input_archive']=actual.with_suffix('.zip').name
    execution=ROOT/(label+'_execution.zip')
    with zipfile.ZipFile(execution,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in captured.items():
            archive.writestr('preexecuted_bytes/'+name,raw)
        archive.writestr('stdout.txt',child.stdout)
        archive.writestr('stderr.txt',child.stderr)
        archive.writestr('natural_exit.json',json.dumps(result,indent=2))
    result['execution_archive']=execution.name
    result['execution_sha256']=sha(execution.read_bytes())
    (ROOT/(label+'_natural_exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    assert unchanged
    print(json.dumps({k:result[k] for k in ('natural_exit','elapsed_seconds','stderr_empty','local_source_input_bytes_unchanged','preexecution_sha256','execution_sha256')},indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch',required=True)
    args=parser.parse_args()
    result=run(args.epoch)
    raise SystemExit(result['natural_exit'])

"""Run NEW actual electric-sector science once, preserving natural exit and bytes."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_stationary_electric_flux_sector'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def run() -> dict:
    target=ROOT/(PREFIX+'.py');own=Path(__file__).read_bytes();before=target.read_bytes()
    output=ROOT/(PREFIX+'_rational_interval_actual_receipt_20261010.json')
    execution=ROOT/(PREFIX+'_rational_interval_execution_bytes_20261010.zip')
    with zipfile.ZipFile(execution,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(target.name,before);z.writestr(Path(__file__).name,own)
    command=[PYTHON,str(target),'--output',str(output)]
    environment=os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    started=time.monotonic()
    process=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True)
    receipt=dict(command=command,cwd=str(ROOT),executable=PYTHON,exit_code=process.returncode,natural_exit=True,stdout_utf8=process.stdout.decode(),stderr_utf8=process.stderr.decode(),stdout_sha256=hashlib.sha256(process.stdout).hexdigest(),stderr_sha256=hashlib.sha256(process.stderr).hexdigest(),source_before_sha256=hashlib.sha256(before).hexdigest(),source_unchanged=target.read_bytes()==before,runner_sha256=hashlib.sha256(own).hexdigest(),execution_archive=dict(path=execution.name,sha256=hashlib.sha256(execution.read_bytes()).hexdigest()),elapsed_seconds=time.monotonic()-started)
    if process.returncode==0:
        raw=output.read_bytes();science=json.loads(raw)
        receipt['science_receipt']=dict(path=output.name,sha256=hashlib.sha256(raw).hexdigest())
        receipt['science_input_archive']=science['immutable_source_input_capture']
        receipt['claim_scope']='Full operator identities; exact BODY basis controls and offset-retaining fixedPhi bracket, NOT characterized ground or photon pole'
    caps=ROOT/(PREFIX+'_rational_interval_natural_exit_capsule_20261010.json')
    caps.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(capsule=caps.name,capsule_sha256=hashlib.sha256(caps.read_bytes()).hexdigest(),**receipt),indent=2))
    return receipt


if __name__=='__main__':
    result=run()
    sys.exit(result['exit_code'] if result['source_unchanged'] else 99)

"""Natural execution, immutable byte epochs and complete stdout/stderr capture."""
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
from typing import Any
ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_stationary_unitary_ground'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'

def sha(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()

def run(label:str)->dict[str,Any]:
    target=ROOT/(PREFIX+'.py');own=Path(__file__).read_bytes();before=target.read_bytes()
    output=ROOT/(PREFIX+'_'+label+'_receipt_20261010.json')
    execution=ROOT/(PREFIX+'_'+label+'_execution_bytes_20261010.zip')
    inputs={n:(ROOT/n).read_bytes() for n in ('induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json','induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json')}
    with zipfile.ZipFile(execution,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(target.name,before);z.writestr(Path(__file__).name,own)
        for n,b in inputs.items():z.writestr(n,b)
    command=[PYTHON,str(target),'--output',str(output)]
    environment=os.environ.copy();environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    start=time.monotonic();process=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True)
    result=dict(command=command,cwd=str(ROOT),executable=PYTHON,exit_code=process.returncode,natural_exit=True,stdout_utf8=process.stdout.decode(),stderr_utf8=process.stderr.decode(),stdout_sha256=sha(process.stdout),stderr_sha256=sha(process.stderr),source_before_sha256=sha(before),source_after_sha256=sha(target.read_bytes()),runner_before_sha256=sha(own),runner_after_sha256=sha(Path(__file__).read_bytes()),input_before_sha256={n:sha(b) for n,b in inputs.items()},input_after_sha256={n:sha((ROOT/n).read_bytes()) for n in inputs},execution_archive=dict(path=execution.name,sha256=sha(execution.read_bytes())),elapsed_seconds=time.monotonic()-start)
    result['source_inputs_unchanged']=before==target.read_bytes() and own==Path(__file__).read_bytes() and all((ROOT/n).read_bytes()==b for n,b in inputs.items())
    if output.exists():
        raw=output.read_bytes();science=json.loads(raw);result['science_receipt']=dict(path=output.name,sha256=sha(raw));result['science_input_archive']=science['immutable_source_input_capture']
    cap=ROOT/(PREFIX+'_'+label+'_natural_exit_capsule_20261010.json');cap.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(capsule=cap.name,capsule_sha256=sha(cap.read_bytes()),**result),indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',default='final');args=parser.parse_args();r=run(args.label);sys.exit(r['exit_code'] if r['source_inputs_unchanged'] else 99)

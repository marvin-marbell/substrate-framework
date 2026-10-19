"""Natural-exit capture for the selected full-source density jet consumer."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_initial_density_higher_jets'

def sha(path: Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--term-cap',type=int,default=1000000)
    args=parser.parse_args()
    producer=ROOT/(PREFIX+'.py');receipt=ROOT/(PREFIX+'_actual_receipt_20261010.json')
    stdout=ROOT/(PREFIX+'_actual_stdout_20261010.txt');stderr=ROOT/(PREFIX+'_actual_stderr_20261010.txt')
    command=[sys.executable,str(producer),'--output',str(receipt),'--term-cap',str(args.term_cap),'--maximum-order','6']
    before=sha(producer);start=time.monotonic();environment=os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):environment[key]='1'
    with stdout.open('wb') as out,stderr.open('wb') as err:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=out,stderr=err)
        exit_code=process.wait()
    capsule: dict[str,Any]=dict(command=command,numeric_exit=exit_code,natural_exit=True,elapsed_seconds=time.monotonic()-start,
                 producer_sha256_before=before,producer_sha256_after=sha(producer),source_stable=before==sha(producer),
                 runtime_executable=sys.executable,runtime_version=sys.version,selected_term_cap=args.term_cap,
                 stdout=dict(path=stdout.name,sha256=sha(stdout)),stderr=dict(path=stderr.name,sha256=sha(stderr)))
    if receipt.exists():
        result=json.loads(receipt.read_bytes());capsule['receipt']=dict(path=receipt.name,sha256=sha(receipt),status=result['status'],actual12=len(result['receiver_rows']))
        capsule['input_archive']=result['inputs'];capsule['numerical']=result['numerical']
    path=ROOT/(PREFIX+'_natural_exit_capsule_20261010.json');path.write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps(dict(numeric_exit=exit_code,elapsed_seconds=capsule['elapsed_seconds'],capsule=path.name)),flush=True)
    return exit_code

if __name__=='__main__':
    raise SystemExit(run())

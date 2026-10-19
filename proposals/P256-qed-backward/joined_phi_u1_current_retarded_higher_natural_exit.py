"""Run the actual whole129 current CLI and bind its natural exit to bytes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_current_retarded_higher'

def sha(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


def run(term_cap:int,seed_cap:int,label:str,trace_budget:bool=False,reuse_jacobi:Path|None=None)->dict:
    prefix=PREFIX+('_trace_budget' if trace_budget else '')
    source=ROOT/(prefix+'.py');before=source.read_bytes()
    output=ROOT/(prefix+'_'+label+'_receipt_20261010.json')
    stdout=ROOT/(prefix+'_'+label+'_stdout_20261010.txt')
    stderr=ROOT/(prefix+'_'+label+'_stderr_20261010.txt')
    command=[sys.executable,str(source),'--output',str(output)]
    command+=['--jacobi-cap',str(term_cap)] if trace_budget else ['--term-cap',str(term_cap),'--seed-cap',str(seed_cap)]
    if reuse_jacobi is not None:
        if not trace_budget:raise ValueError('Jacobi reuse requires the trace budget consumer')
        command+=['--reuse-jacobi',str(reuse_jacobi)]
    snapshot=ROOT/(prefix+'_'+label+'_source_bytes_20261010.zip')
    runner_before=Path(__file__).read_bytes()
    with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(source.name,before)
        bundle.writestr(Path(__file__).name,runner_before)
    started=time.monotonic()
    with stdout.open('wb') as out,stderr.open('wb') as err:
        process=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,check=False)
    after=source.read_bytes();receipt=output.read_bytes() if output.exists() else None
    capsule=dict(command=command,natural_numeric_exit=process.returncode,elapsed_seconds=time.monotonic()-started,source_sha256_before=sha(before),source_sha256_after=sha(after),source_stable=before==after,runner_sha256=sha(Path(__file__).read_bytes()),receipt_sha256=sha(receipt) if receipt else None,stdout_sha256=sha(stdout.read_bytes()),stderr_sha256=sha(stderr.read_bytes()),runtime_executable=sys.executable)
    capsule['preexecution_source_bytes_zip_sha256']=sha(snapshot.read_bytes())
    capsule['runner_stable']=runner_before==Path(__file__).read_bytes()
    if receipt:
        record=json.loads(receipt);capsule['receipt_source_matches']=record['source_sha256']==sha(before);capsule['input_sha256']=record['input_sha256'];capsule['input_archive_sha256']=record['input_archive_sha256'];capsule['status']=record['status']
    (ROOT/(prefix+'_'+label+'_natural_exit_capsule_20261010.json')).write_text(json.dumps(capsule,indent=2)+'\n')
    return capsule


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--term-cap',type=int,default=200000)
    parser.add_argument('--seed-cap',type=int,default=0)
    parser.add_argument('--label',default='actual')
    parser.add_argument('--trace-budget',action='store_true')
    parser.add_argument('--reuse-jacobi',type=Path)
    args=parser.parse_args()
    if not args.label.replace('_','').isalnum():parser.error('Alphanumeric label required')
    print(json.dumps(run(args.term_cap,args.seed_cap,args.label,args.trace_budget,args.reuse_jacobi)),flush=True)

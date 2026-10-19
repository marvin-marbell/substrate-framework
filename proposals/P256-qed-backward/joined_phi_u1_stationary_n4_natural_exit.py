"""Run the NEW scientific stationary construction and bind natural process exit."""
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
PREFIX='joined_phi_u1_stationary_n4'
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',required=True)
    args=parser.parse_args()
    PREFIX=PREFIX+'_'+args.label
    source=ROOT/'joined_phi_u1_stationary_n4.py'
    receipt=ROOT/(PREFIX+'_receipt.json')
    command=[sys.executable,str(source),'--output',str(receipt)]
    captured={source.name:source.read_bytes(),Path(__file__).name:Path(__file__).read_bytes()}
    start=time.monotonic()
    result=subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
    logs={PREFIX+'_stdout.txt':result.stdout,PREFIX+'_stderr.txt':result.stderr}
    for name,raw in logs.items():(ROOT/name).write_bytes(raw)
    capsule=dict(command=command,cwd=str(ROOT),natural_exit_code=result.returncode,wall_seconds=time.monotonic()-start,source_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in captured.items()},source_unchanged={n:(ROOT/n).read_bytes()==raw for n,raw in captured.items()},stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),scientific_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest() if receipt.exists() else None)
    archive=ROOT/(PREFIX+'_execution_bytes.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in {**captured,**logs}.items():z.writestr(name,raw)
    capsule['execution_bytes_archive']=dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (ROOT/(PREFIX+'_natural_exit_capsule.json')).write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps(capsule,indent=2))
    if result.returncode:sys.exit(result.returncode)

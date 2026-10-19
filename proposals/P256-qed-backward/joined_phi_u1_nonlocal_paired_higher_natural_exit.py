"""Finite NEW paired kernel producer with natural-exit and exact-byte capsule."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile
from typing import Any
ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_nonlocal_paired_higher'

def limits()->None:
    resource.setrlimit(resource.RLIMIT_CPU,(10800,10800))
    resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))

def run(output:Path,order:int,controls:bool,lane:str='cubic',coefficient:Path|None=None)->dict[str,Any]:
    suffix={'cubic':'','tail':'_tail','controls':'_controls'}[lane]
    producer=ROOT/(PREFIX+suffix+'.py');source=producer.read_bytes();driver=Path(__file__).read_bytes()
    receipt=output.with_name(output.stem+'_receipt.json');out=output.with_name(output.stem+'_stdout.txt');err=output.with_name(output.stem+'_stderr.txt')
    cmd=[sys.executable,str(producer),'--output',str(receipt)]
    if lane=='cubic':
        cmd.extend(['--order',str(order)])
        if controls:cmd.append('--controls')
    elif lane=='tail' and coefficient:
        cmd.extend(['--coefficient',str(coefficient.resolve())])
    started=time.monotonic()
    with out.open('wb') as stdout,err.open('wb') as stderr:
        child=subprocess.run(cmd,cwd=ROOT,stdout=stdout,stderr=stderr,preexec_fn=limits,check=False)
    elapsed=time.monotonic()-started
    archive=output.with_name(output.stem+'_runtime_bytes.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr(producer.name,source);z.writestr(Path(__file__).name,driver)
        z.writestr(out.name,out.read_bytes());z.writestr(err.name,err.read_bytes())
    result={'command':cmd,'cwd':str(ROOT),'returncode':child.returncode,'natural_numeric_exit':child.returncode==0,'elapsed_seconds':elapsed,'caps':{'cpu_seconds':10800,'address_space_bytes':12*1024**3},'source_sha256':hashlib.sha256(source).hexdigest(),'driver_sha256':hashlib.sha256(driver).hexdigest(),'source_unchanged':producer.read_bytes()==source,'runtime_bytes':{'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},'stdout':{'path':out.name,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},'stderr':{'path':err.name,'sha256':hashlib.sha256(err.read_bytes()).hexdigest()}}
    if receipt.exists():result['receipt']={'path':receipt.name,'sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--order',type=int,choices=(3,5),default=3);p.add_argument('--controls',action='store_true');p.add_argument('--lane',choices=('cubic','tail','controls'),default='cubic');p.add_argument('--coefficient',type=Path);a=p.parse_args();run(a.output,a.order,a.controls,a.lane,a.coefficient)

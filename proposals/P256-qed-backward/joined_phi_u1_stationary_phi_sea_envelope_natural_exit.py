"""Run the new native sea-envelope scientific CLI and preserve its natural exit."""
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
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_stationary_phi_sea_envelope'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.name.startswith(PREFIX):raise ValueError('Output outside assigned prefix')
    sources={name:(ROOT/name).read_bytes() for name in (PREFIX+'.py',Path(__file__).name)}
    archive=out.with_name(out.stem+'_execution_bytes.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in sources.items():z.writestr(name,raw)
    command=[sys.executable,str(ROOT/(PREFIX+'.py')),'--output',str(out)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    start=time.monotonic();timed_out=False
    try:
        child=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,timeout=300)
        code=child.returncode;stdout=child.stdout;stderr=child.stderr
    except subprocess.TimeoutExpired as error:
        code=None;timed_out=True;stdout=error.stdout or b'';stderr=error.stderr or b''
    stdout_path=out.with_name(out.stem+'_stdout.txt');stderr_path=out.with_name(out.stem+'_stderr.txt')
    stdout_path.write_bytes(stdout);stderr_path.write_bytes(stderr)
    capsule: dict[str,Any]=dict(schema=1,command=command,cwd=str(ROOT),runtime=str(Path(sys.executable).resolve()),runtime_version=sys.version,deadline_seconds=300,natural_exit=code==0 and not timed_out,returncode=code,timed_out=timed_out,elapsed_seconds=time.monotonic()-start,execution_source_manifest={name:dict(sha256=sha(raw),bytes=len(raw),unchanged=(ROOT/name).read_bytes()==raw) for name,raw in sources.items()},execution_bytes_archive=dict(path=str(archive),sha256=sha(archive.read_bytes())),stdout=dict(path=str(stdout_path),sha256=sha(stdout)),stderr=dict(path=str(stderr_path),sha256=sha(stderr)))
    if code==0:
        raw=out.read_bytes();receipt=json.loads(raw)
        capsule['receipt']=dict(path=str(out),sha256=sha(raw))
        capsule['input_source_bytes_archive']=receipt['input_source_bytes_archive']
        capsule['input_source_stable']=all(receipt['input_source_unchanged'].values())
        proof=out.with_name(out.stem+'_proof.txt')
        proof.write_text('\n\n'.join(key+'\n'+value for key,value in receipt['proof'].items())+'\n')
        capsule['proof']=dict(path=str(proof),sha256=sha(proof.read_bytes()))
    cap=out.with_name(out.stem+'_natural_exit_capsule.json')
    cap.write_text(json.dumps(capsule,indent=2)+'\n');print(json.dumps(capsule,indent=2))
    if code!=0 or not all(v['unchanged'] for v in capsule['execution_source_manifest'].values()):raise SystemExit(code or 1)

if __name__=='__main__':main()

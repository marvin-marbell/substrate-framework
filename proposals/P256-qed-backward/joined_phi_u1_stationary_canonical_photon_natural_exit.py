"""Capture immutable canonical-functional source/input bytes and natural process exit."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_stationary_canonical_photon'
def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
def main() -> None:
    source=ROOT/(PREFIX+'.py');output=ROOT/(PREFIX+'_actual_receipt.json')
    names=[source.name,Path(__file__).name,'induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json','induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json']
    before={name:(ROOT/name).read_bytes() for name in names}
    archive=ROOT/(PREFIX+'_execution_bytes.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in before.items():z.writestr(name,raw)
    command=[sys.executable,str(source),'--output',str(output)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    p=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,check=False)
    stdout=ROOT/(PREFIX+'_stdout.txt');stderr=ROOT/(PREFIX+'_stderr.txt');stdout.write_bytes(p.stdout);stderr.write_bytes(p.stderr)
    unchanged={name:(ROOT/name).read_bytes()==raw for name,raw in before.items()}
    capsule=dict(command=command,exit_code=p.returncode,natural_exit=True,timeout_used=False,source_input_before_sha256={k:sha(v) for k,v in before.items()},source_input_after_sha256={k:sha((ROOT/k).read_bytes()) for k in before},unchanged=unchanged,immutable_execution_archive_sha256=sha(archive.read_bytes()),stdout_sha256=sha(p.stdout),stderr_sha256=sha(p.stderr),stdout=p.stdout.decode(),stderr=p.stderr.decode(),runtime_executable=sys.executable)
    if output.exists():capsule['receipt_sha256']=sha(output.read_bytes())
    (ROOT/(PREFIX+'_natural_exit_capsule.json')).write_text(json.dumps(capsule,indent=2))
    print(json.dumps(capsule,indent=2));sys.exit(p.returncode)
if __name__=='__main__':main()

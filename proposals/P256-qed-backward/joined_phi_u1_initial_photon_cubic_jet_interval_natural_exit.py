"""Capture the outward initial cubic-jet consumer's natural numeric exit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_initial_photon_cubic_jet_interval'
source=ROOT/(PREFIX+'.py')
before=source.read_bytes();runner=Path(__file__).read_bytes()
command=[sys.executable,str(source),'--output',str(ROOT/(PREFIX+'_receipt.json'))]
start=time.monotonic();process=subprocess.run(command,cwd=ROOT,capture_output=True)
(ROOT/(PREFIX+'_stdout.txt')).write_bytes(process.stdout)
(ROOT/(PREFIX+'_stderr.txt')).write_bytes(process.stderr)
result={'command':command,'numeric_exit':process.returncode,'natural_exit':True,'source_stable':source.read_bytes()==before,'source_sha256':hashlib.sha256(before).hexdigest(),'runner_sha256':hashlib.sha256(runner).hexdigest(),'elapsed_seconds':time.monotonic()-start,'runtime_executable':sys.executable,'runtime_version':sys.version,'stdout_sha256':hashlib.sha256(process.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(process.stderr).hexdigest(),'scope':'Actual outward fixed-mean full129 initial coefficient consumer, not numeric/primary producer rerun or evolution','outputs':{}}
for suffix in ('_receipt.json','_receipt.npz','_receipt.zip'):
    path=ROOT/(PREFIX+suffix)
    if path.exists():result['outputs'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
(ROOT/(PREFIX+'_natural_exit_capsule.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
if not result['source_stable']:raise SystemExit(2)
raise SystemExit(process.returncode)

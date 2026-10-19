"""Execute the final scientific CLI and selected checker, recording natural exits."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
NAMES = ['induced_phi_primitive_nonlinear_action_certificate.py','induced_phi_primitive_nonlinear_action_certificate_controls.py',Path(__file__).name]
RECEIPT = 'induced_phi_primitive_nonlinear_action_certificate_receipt.json'


def run() -> int:
    captured = {name:(ROOT/name).read_bytes() for name in NAMES}
    commands = [[PYTHON,NAMES[0],'--output',RECEIPT],['/home/administrator/.local/bin/pyright','--pythonpath',PYTHON,'--stats',*NAMES]]
    records = []
    for command in commands:
        start = time.monotonic()
        result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=False)
        records.append({'argv':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'wall_seconds':time.monotonic()-start,'natural_exit':True})
        if result.returncode:
            break
    stable = {name:{'sha256':hashlib.sha256(raw).hexdigest(),'bytes_utf8':raw.decode(),'unchanged':(ROOT/name).read_bytes()==raw} for name,raw in captured.items()}
    receipts = {}
    if records[0]['returncode']==0:
        archive = json.loads((ROOT/RECEIPT).read_text())['immutable_capsule']['path']
        receipts = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (RECEIPT,archive)}
    ok = len(records)==2 and all(r['returncode']==0 for r in records) and all(r['unchanged'] for r in stable.values())
    packet = {'status':'NATURAL_EXIT_FINAL_SELECTED_CHECKER' if ok else 'FAILED_NO_ADMISSION','project_python':PYTHON,'runtime_executable':sys.executable,'records':records,'source_at_start':stable,'receipt_sha256':receipts,'all_sources_stable':all(r['unchanged'] for r in stable.values()),'final_selected_files':NAMES}
    (ROOT/'induced_phi_primitive_nonlinear_action_certificate_natural_exit_capsule.json').write_text(json.dumps(packet,indent=2)+'\n')
    print(json.dumps({key:value for key,value in packet.items() if key!='source_at_start'},indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(run())

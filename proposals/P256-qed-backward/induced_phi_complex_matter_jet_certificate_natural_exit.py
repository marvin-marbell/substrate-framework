"""Run the actual scientific producers and final selected checker to natural exit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
NAMES=['induced_phi_complex_matter_jet_certificate.py','induced_phi_complex_matter_jet_certificate_force_consumer.py',Path(__file__).name]

def run() -> int:
 captured={name:(ROOT/name).read_bytes() for name in NAMES}
 commands=[
 [PYTHON,NAMES[0],'--output','induced_phi_complex_matter_jet_certificate_receipt.json'],
 [PYTHON,NAMES[1],'--jet','induced_phi_complex_matter_jet_certificate_receipt.json','--output','induced_phi_complex_matter_jet_certificate_force_consumer_receipt.json'],
 ['pyright','--pythonpath',PYTHON,'--stats',*NAMES]]
 records=[]
 for command in commands:
  started=time.monotonic();result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=False)
  records.append({'argv':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'elapsed_seconds':time.monotonic()-started,'natural_exit':True})
  if result.returncode:break
 stable={name:{'sha256':hashlib.sha256(raw).hexdigest(),'bytes_utf8':raw.decode(),'unchanged':(ROOT/name).read_bytes()==raw} for name,raw in captured.items()}
 receipts={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('induced_phi_complex_matter_jet_certificate_receipt.json','induced_phi_complex_matter_jet_certificate_force_consumer_receipt.json') if (ROOT/name).is_file()}
 ok=len(records)==3 and all(r['returncode']==0 for r in records) and all(x['unchanged'] for x in stable.values())
 packet={'status':'NATURAL_EXIT_FINAL_SELECTED_CHECKER' if ok else 'FAILED_NO_ADMISSION','project_python':PYTHON,'runtime_executable':sys.executable,'records':records,'source_at_start':stable,'receipt_sha256':receipts,'all_sources_stable':all(x['unchanged'] for x in stable.values()),'final_selected_files':NAMES}
 (ROOT/'induced_phi_complex_matter_jet_certificate_natural_exit_capsule.json').write_text(json.dumps(packet,indent=2)+'\n')
 print(json.dumps({'status':packet['status'],'records':[{k:v for k,v in r.items() if k not in ('argv',)} for r in records],'receipt_sha256':receipts},indent=2))
 return 0 if ok else 1

if __name__=='__main__':raise SystemExit(run())

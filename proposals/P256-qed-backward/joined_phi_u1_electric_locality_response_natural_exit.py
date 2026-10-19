"""Run the selected NEW locality consumer once, retain numeric exit and bytes."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_electric_locality_response'

def run()->int:
 started=time.monotonic();source=ROOT/(PREFIX+'.py');own=Path(__file__);raw=source.read_bytes();ownraw=own.read_bytes()
 output=ROOT/(PREFIX+'_receipt.json')
 command=[sys.executable,str(source),'--output',str(output)]
 env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
 process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
 stdout=ROOT/(PREFIX+'_stdout.txt');stderr=ROOT/(PREFIX+'_stderr.txt')
 stdout.write_bytes(process.stdout);stderr.write_bytes(process.stderr)
 capsule=dict(command=command,natural_numeric_exit=process.returncode,elapsed_seconds=time.monotonic()-started,source_sha256=hashlib.sha256(raw).hexdigest(),runner_sha256=hashlib.sha256(ownraw).hexdigest(),interpreter_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),stdout=dict(path=stdout.name,sha256=hashlib.sha256(process.stdout).hexdigest()),stderr=dict(path=stderr.name,sha256=hashlib.sha256(process.stderr).hexdigest()),source_stable=source.read_bytes()==raw and own.read_bytes()==ownraw)
 if process.returncode==0:
  result=json.loads(output.read_bytes())
  if result['clock_exact']!='1/65536' or set(result['all129_states'])!={'background','raw_source'}:raise ArithmeticError('Selected whole129/clock scope missing')
  controls=result['operator_controls']
  if controls['full_cross_exact_defect']!=0 or controls['faithful_Weyl_operator_words']['full']['phase_angle_exact']!='0':raise ArithmeticError('Exact cancellation failed')
  if any(controls['faithful_Weyl_operator_words'][key]['phase_angle_exact']=='0' for key in ('omittedHC','HE32_HC64')):raise ArithmeticError('Adverse operator controls blind')
  if controls['E0_omittedRef_commutator_energy_shift']=='0':raise ArithmeticError('Ref control blind')
  if any(shape!=[129,129] and shape!=[192,192] for shape in result['full129_capsule']['arrays'].values()):raise ArithmeticError('Whole129 output narrowed')
  capsule['receipt_sha256']=hashlib.sha256(output.read_bytes()).hexdigest();capsule['input_bytes']=result['input_bytes'];capsule['all129_capsule']=result['full129_capsule'];capsule['runtime']=result['runtime']
  capsule['scientific_scope']='Exact locality passed; response-ball construction sign resolution assessed by actual all129 price, failure retained rather than checker suppressed.'
 (ROOT/(PREFIX+'_natural_exit_capsule.json')).write_text(json.dumps(capsule,indent=2)+'\n')
 print(json.dumps(capsule),flush=True)
 return process.returncode

if __name__=='__main__':sys.exit(run())

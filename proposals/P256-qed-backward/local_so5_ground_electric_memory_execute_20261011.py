from __future__ import annotations
import hashlib,json,resource,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PREFIX='local_so5_ground_electric_memory_typed_adapter_attempt4_20261011'
SCRIPT='local_so5_ground_electric_memory_consumer_20261011.py'
NAMES=[SCRIPT,Path(__file__).name,'local_so5_resum_source_20261011.py','local_so5_resum_actual04_20261011_receipt.json','local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json','local_so5_positive_output_row_inertia_certificate_20261011_receipt.json']
PINS={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in NAMES}
def archive(suffix,names):
 with zipfile.ZipFile(ROOT/(PREFIX+suffix),'x',compression=zipfile.ZIP_DEFLATED) as z:
  for n in names:z.write(ROOT/n,n)
archive('_preexecution.zip',NAMES)
command=[sys.executable,SCRIPT];start=time.monotonic();r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
(ROOT/(PREFIX+'_stdout.txt')).write_text(r.stdout);(ROOT/(PREFIX+'_stderr.txt')).write_text(r.stderr)
exitrecord=dict(command=command,returncode=r.returncode,natural_exit=r.returncode>=0,elapsed_seconds=time.monotonic()-start,source_and_input_sha256=PINS,unchanged={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in PINS.items()},stdout_sha256=hashlib.sha256(r.stdout.encode()).hexdigest(),stderr_sha256=hashlib.sha256(r.stderr.encode()).hexdigest(),maxrss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,stage='NEW projected electric resolvent/moments, not original source solver rerun, parent review pending')
(ROOT/(PREFIX+'_exit.json')).write_text(json.dumps(exitrecord,indent=2)+'\n')
outputs=[PREFIX+'_stdout.txt',PREFIX+'_stderr.txt',PREFIX+'_exit.json']
if r.returncode==0:outputs.append('local_so5_ground_electric_memory_consumer_20261011_receipt.json')
archive('_executed.zip',NAMES+outputs)
print(json.dumps(exitrecord,indent=2));print(r.stdout);print(r.stderr,file=sys.stderr);sys.exit(r.returncode)

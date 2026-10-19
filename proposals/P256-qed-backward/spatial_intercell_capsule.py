"""Run only the NEW owned scientific source and persist natural exit evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--epoch',required=True);args=ap.parse_args()
stem=f'spatial_intercell_{args.epoch}_20261010';output=ROOT/(stem+'_receipt.json')
command=[sys.executable,'-B',str(ROOT/'spatial_intercell_source.py'),'--output',str(output)]
start=time.monotonic();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
p=subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
stdout=ROOT/(stem+'_stdout.txt');stderr=ROOT/(stem+'_stderr.txt')
stdout.write_bytes(p.stdout);stderr.write_bytes(p.stderr)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
receipt=dict(command=command,python=sys.executable,returncode=p.returncode,natural_exit=True,elapsed_seconds=time.monotonic()-start,stdout=dict(path=str(stdout),sha256=sha(stdout)),stderr=dict(path=str(stderr),sha256=sha(stderr)),source_sha256=sha(ROOT/'spatial_intercell_source.py'),capsule_sha256=sha(Path(__file__)),output_exists=output.exists())
if output.exists():receipt['output']=dict(path=str(output),sha256=sha(output))
exitpath=ROOT/(stem+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
with zipfile.ZipFile(ROOT/(stem+'_execution.zip'),'x',compression=zipfile.ZIP_DEFLATED) as z:
    for path in (ROOT/'spatial_intercell_source.py',Path(__file__),stdout,stderr,exitpath,output):
        if path.exists():z.write(path,path.name)
print(json.dumps(receipt,indent=2));print(p.stdout.decode());print(p.stderr.decode(),file=sys.stderr)
sys.exit(p.returncode)

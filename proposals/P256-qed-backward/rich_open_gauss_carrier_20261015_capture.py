"""Natural-exit capture for this owning experiment, never historical CLI replay."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_open_gauss_carrier_20261015_'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    stage=sys.argv[1]
    selected=[ROOT/(P+n) for n in ('model.py','input.json',stage+'.py','capture.py')]
    if stage=='charged':
        selected.append(ROOT/(P+'neutral_reference.npz'))
    hashes={p.name:sha(p) for p in selected}
    command=[sys.executable,str(ROOT/(P+stage+'.py'))]
    (ROOT/(P+stage+'_start.json')).write_text(json.dumps(dict(command=command,input_source_SHA256=hashes,environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},python=sys.version),indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(P+stage+'_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in selected:
            z.write(p,p.name)
    start=time.monotonic()
    with (ROOT/(P+stage+'_stdout.txt')).open('w') as out,(ROOT/(P+stage+'_stderr.txt')).open('w') as err:
        done=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err)
    result=dict(returncode=done.returncode,natural_exit=True,elapsed_seconds=time.monotonic()-start,sources_unchanged=all(sha(ROOT/n)==h for n,h in hashes.items()),source_SHA256=hashes)
    (ROOT/(P+stage+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    sys.exit(done.returncode)
if __name__=='__main__':
    main()

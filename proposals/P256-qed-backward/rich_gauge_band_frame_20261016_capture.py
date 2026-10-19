"""Capture owning source/input, actual child exits and immutable capsules."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from typing import Any

ROOT=Path(__file__).resolve().parent
P='rich_gauge_band_frame_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    stamp=sys.argv[1] if len(sys.argv)>1 else 'actual1'
    files=[ROOT/(P+x) for x in ['model.py','run.py','connections.py','capture.py','interface.json','equations.json']]
    script=sys.argv[2] if len(sys.argv)>2 else 'run'
    inputs: dict[str, Any] = dict(command=[PY,str(ROOT/(P+script+'.py'))],cwd=str(ROOT),
                env={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                plan='issue222-rich-backward-candidates-20261014',captured_revision='b048dbe91b0250f9525e9a07b6d76b95a8080f2f57c071873cad15f8a8fd0e4f')
    ip=ROOT/(P+stamp+'_input.json');ip.write_text(json.dumps(inputs,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(P+stamp+'_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for path in files+[ip]:z.write(path,path.name)
    start=time.time()
    with (ROOT/(P+stamp+'_stdout.txt')).open('w') as out,(ROOT/(P+stamp+'_stderr.txt')).open('w') as err:
        child=subprocess.run(inputs['command'],cwd=ROOT,env={**os.environ,**inputs['env']},stdout=out,stderr=err)
    result=dict(returncode=child.returncode,natural_exit=True,seconds=time.time()-start,input_sha256=hashlib.sha256(ip.read_bytes()).hexdigest())
    (ROOT/(P+stamp+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    sys.exit(child.returncode)


if __name__=='__main__':main()

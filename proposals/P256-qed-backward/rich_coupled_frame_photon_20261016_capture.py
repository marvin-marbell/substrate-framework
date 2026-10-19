"""Capture source inputs, natural child exit and immutable attempt capsule."""
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
P='rich_coupled_frame_photon_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    stamp=sys.argv[1] if len(sys.argv)>1 else 'actual1'
    script=sys.argv[2] if len(sys.argv)>2 else 'run'
    files=sorted(ROOT.glob(P+'*.py'))+[ROOT/(P+'equations.json'),ROOT/'rich_gauge_band_frame_20261016_model.py']
    inputs: dict[str, Any] =dict(command=[PY,str(ROOT/(P+script+'.py'))],cwd=str(ROOT),env={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},plan='issue222-rich-backward-candidates-20261014',captured_revision='673b0253d8fdda68dcb25d751a1b8f757476f04a8c4c749e92387deca47cd627')
    ip=ROOT/(P+stamp+'_input.json');ip.write_text(json.dumps(inputs,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(P+stamp+'_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for f in files+[ip]:z.write(f,f.name)
    start=time.time()
    with (ROOT/(P+stamp+'_stdout.txt')).open('w') as out,(ROOT/(P+stamp+'_stderr.txt')).open('w') as err:
        child=subprocess.run(inputs['command'],cwd=ROOT,env={**os.environ,**inputs['env']},stdout=out,stderr=err)
    result=dict(returncode=child.returncode,natural_exit=True,seconds=time.time()-start,input_sha256=hashlib.sha256(ip.read_bytes()).hexdigest())
    (ROOT/(P+stamp+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    sys.exit(child.returncode)


if __name__=='__main__':main()

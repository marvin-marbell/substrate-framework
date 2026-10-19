"""Immutable source/input capsules, natural exits and failed attempts retained."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_spatial_frame_photon_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    stamp=sys.argv[1];script=sys.argv[2] if len(sys.argv)>2 else 'run';args=sys.argv[3:] or ['actual1']
    files=sorted(ROOT.glob(P+'*.py'))+[ROOT/(P+'equations.json'),ROOT/'rich_gauge_band_frame_20261016_model.py',ROOT/'rich_coupled_frame_photon_20261016_language.py',ROOT/'rich_local_frame_spatial_carrier_20261016_angular.py',ROOT/'rich_frame_gauss_20261014_model.py']
    command=[PY,str(ROOT/(P+script+'.py')),*args]
    ip=ROOT/(P+stamp+'_input.json')
    if ip.exists():raise FileExistsError('attempts immutable: '+str(ip))
    data: dict[str,Any]
    data=dict(command=command,cwd=str(ROOT),environment={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},plan='issue222-rich-backward-candidates-20261014',revision='6facdc39fc29c4b59769e6335b1c31bc95f322e3bf55184de7505d83d9008b08')
    ip.write_text(json.dumps(data,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(P+stamp+'_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for f in files+[ip]:z.write(f,f.name)
    start=time.time()
    with (ROOT/(P+stamp+'_stdout.txt')).open('w') as out,(ROOT/(P+stamp+'_stderr.txt')).open('w') as err:
        child=subprocess.run(command,cwd=ROOT,env={**os.environ,**data['environment']},stdout=out,stderr=err)
    result=dict(returncode=child.returncode,natural_exit=child.returncode>=0,termination_signal=-child.returncode if child.returncode<0 else None,seconds=time.time()-start,input_sha256=hashlib.sha256(ip.read_bytes()).hexdigest())
    (ROOT/(P+stamp+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    sys.exit(child.returncode)


if __name__=='__main__':main()

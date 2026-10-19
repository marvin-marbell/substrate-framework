"""Exact initial-basis case sources, natural child exits and immutable failures."""
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
P='rich_exact_electric_picture_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    stamp=sys.argv[1];script=sys.argv[2];command=[PY,str(ROOT/(P+script+'.py')),*sys.argv[3:]]
    ip=ROOT/(P+stamp+'_input.json')
    if ip.exists():raise FileExistsError(str(ip))
    files=sorted(ROOT.glob(P+'*.py'))+sorted(ROOT.glob('rich_spatial_frame_photon_20261016_*.py'))
    files += [ROOT/'rich_gauge_band_frame_20261016_model.py',ROOT/'rich_local_frame_spatial_carrier_20261016_angular.py',ROOT/'rich_frame_gauss_20261014_model.py',ROOT/'rich_coupled_frame_photon_20261016_language.py']
    metadata: dict[str,Any]=dict(command=command,cwd=str(ROOT),environment={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},plan='issue222-rich-backward-candidates-20261014',revision='0b8d134d38e4e77ca0c7cc6af36b4ac042b307918af703b8658b8ace0a36feeb',contract='issue222-exact-electric-interaction-picture-contract-20261016 snapshot367C',physical_H='UNCHANGED frozen48; exact U0 moving basis before Galerkin')
    ip.write_text(json.dumps(metadata,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(P+stamp+'_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for f in files+[ip]:z.write(f,f.name)
    start=time.time()
    with (ROOT/(P+stamp+'_stdout.txt')).open('w') as out,(ROOT/(P+stamp+'_stderr.txt')).open('w') as err:
        child=subprocess.run(command,cwd=ROOT,env={**os.environ,**metadata['environment']},stdout=out,stderr=err)
    result=dict(returncode=child.returncode,natural_exit=child.returncode>=0,termination_signal=-child.returncode if child.returncode<0 else None,seconds=time.time()-start,input_sha256=hashlib.sha256(ip.read_bytes()).hexdigest())
    (ROOT/(P+stamp+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    sys.exit(child.returncode)


if __name__=='__main__':main()

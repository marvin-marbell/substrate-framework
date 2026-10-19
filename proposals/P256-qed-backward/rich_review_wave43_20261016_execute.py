"""Capture independent science commands, inputs, source hashes and natural exits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
P='rich_review_wave43_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def execute(case):
    script=ROOT/(P+case+'.py')
    inputs=sorted(ROOT.glob(P+case+'*input*.json'))
    sources=[script,Path(__file__).resolve()]
    if case in ('band','band_connection'):
        sources += [ROOT/'rich_gauge_band_frame_20261016_model.py',ROOT/'rich_gauge_band_frame_20261016_interface.json']
    if case=='thermal_audit':
        sources += [ROOT/'rich_unitary_rotor_photon_20261016_thermal.py',ROOT/'rich_unitary_rotor_photon_20261016_interface.json',ROOT/(P+'rotor.py')]
    if case=='bridge_owner':
        sources += [ROOT/'rich_unitary_rotor_photon_20261016_bridge.py',ROOT/'rich_unitary_rotor_photon_20261016_thermal.py',ROOT/'rich_unitary_rotor_photon_20261016_interface.json']
    before={str(x.name):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources+inputs}
    archive=ROOT/(P+case+'_source_capsule.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for x in sources+inputs:z.write(x,x.name)
    cmd=[PY,str(script)]
    start=time.time()
    with (ROOT/(P+case+'_stdout.txt')).open('w') as out,(ROOT/(P+case+'_stderr.txt')).open('w') as err:
        child=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=err,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    receipt=dict(command=cmd,cwd=str(ROOT),returncode=child.returncode,natural_exit=child.returncode>=0,elapsed_seconds=time.time()-start,source_input_sha256=before,source_capsule_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (ROOT/(P+case+'_exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return child.returncode


if __name__=='__main__':sys.exit(execute(sys.argv[1]))

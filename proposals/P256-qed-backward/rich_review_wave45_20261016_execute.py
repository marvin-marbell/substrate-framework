"""Capture each independent bounded physics consumer, including natural exit and sources."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent;P='rich_review_wave45_20261016_'


def main():
    case=sys.argv[1];own=ROOT/(P+case+'.py')
    deps=[own,ROOT/(P+'execute.py'),ROOT/'rich_gauge_band_frame_20261016_model.py',ROOT/'rich_spatial_gauge_source_20261016_family.py',ROOT/'rich_unitary_rotor_photon_20261016_thermal.py',ROOT/'rich_unitary_rotor_photon_20261016_bridge.py',ROOT/'rich_coupled_frame_photon_20261016_model.py',ROOT/'rich_collective_photon_response_20261016_model.py']
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in deps}
    with zipfile.ZipFile(ROOT/(P+case+'_source_capsule.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for p in deps:z.write(p,p.name)
    command=['/home/administrator/substrate-framework/.venv/bin/python',str(own)]
    inp=dict(plan_id='issue222-rich-backward-candidates-20261014',captured_revision='673b0253d8fdda68dcb25d751a1b8f757476f04a8c4c749e92387deca47cd627',template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',work_item='independent-review',command=command,source_sha256=hashes)
    (ROOT/(P+case+'_input.json')).write_text(json.dumps(inp,indent=2)+'\n')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.time();child=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,env=env)
    (ROOT/(P+case+'_stdout.txt')).write_text(child.stdout);(ROOT/(P+case+'_stderr.txt')).write_text(child.stderr)
    unchanged={p.name:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name] for p in deps}
    result=dict(natural_exit=child.returncode,elapsed=time.time()-start,sources_unchanged=unchanged)
    (ROOT/(P+case+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),child.stdout,child.stderr[-1000:],flush=True)
    sys.exit(child.returncode)


if __name__=='__main__':main()

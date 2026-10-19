"""Capture only the NEW reviewer consumer, exact source bytes and natural exit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave5_20261014'
NAMES=[PREFIX+'_consumer.py',PREFIX+'_capture.py',PREFIX+'_input.json','rich_frame_gauss_20261014_model.py','rich_frame_gauss_20261014_input.json','rich_quantum_collision_20261014_model.py','rich_quantum_collision_20261014_input.json','rich_spin_flux_spatial_20261014_model.py','rich_spin_flux_spatial_20261014_3d_model.py']

def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    before={name:(ROOT/name).read_bytes() for name in NAMES}
    command=[sys.executable,str(ROOT/(PREFIX+'_consumer.py'))]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.time()
    run=subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
    elapsed=time.time()-start
    (ROOT/(PREFIX+'_stdout.txt')).write_bytes(run.stdout)
    (ROOT/(PREFIX+'_stderr.txt')).write_bytes(run.stderr)
    after={name:digest((ROOT/name).read_bytes()) for name in NAMES}
    previous={}
    for name,key in [('rich_frame_gauss_20261014_actual6_exit.json','sources_before'),('rich_quantum_collision_20261014_actual1_exit.json','input_SHA256')]:
        receipt=json.loads((ROOT/name).read_text())
        previous[name]={'natural_exit':receipt['natural_exit'],'returncode':receipt['returncode'],'sha256':digest((ROOT/name).read_bytes()),'source_matches_current':{source:after[source]==sha for source,sha in receipt[key].items() if source in after},'source_hashes':receipt[key]}
    outputs=[PREFIX+'_stdout.txt',PREFIX+'_stderr.txt',PREFIX+'_result.json',PREFIX+'_frame_states.npz',PREFIX+'_collision_states.npz']
    receipt={'command':command,'cwd':str(ROOT),'python':sys.version,'natural_exit':run.returncode>=0,'returncode':run.returncode,'elapsed_seconds':elapsed,'sources_before':{name:digest(data) for name,data in before.items()},'sources_after':after,'previous_scientific_capture_binding':previous,'outputs':{name:digest((ROOT/name).read_bytes()) for name in outputs if (ROOT/name).exists()}}
    (ROOT/(PREFIX+'_exit.json')).write_text(json.dumps(receipt,indent=2))
    with zipfile.ZipFile(ROOT/(PREFIX+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in before.items():
            archive.writestr('source_before/'+name,data)
        for name in outputs+[PREFIX+'_exit.json']:
            if (ROOT/name).exists():
                archive.write(ROOT/name,name)
    print(json.dumps(receipt,indent=2))
    raise SystemExit(run.returncode)


if __name__=='__main__':
    main()

"""Immutable attempt inputs/source capsule and honest natural process exits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_local_frame_spatial_carrier_20261016_'


def main():
    name=sys.argv[1];target=sys.argv[2] if len(sys.argv)>2 else P+'run.py'
    files=sorted(ROOT.glob(P+'*.py'))+[ROOT/'rich_gauge_band_frame_20261016_model.py',ROOT/'rich_spatial_frame_photon_20261016_kernel.py',ROOT/'rich_spatial_frame_photon_20261016_quadrature.py',ROOT/'rich_spatial_frame_photon_20261016_fields.py',ROOT/'rich_frame_gauss_20261014_model.py']
    record=dict(command=['/home/administrator/substrate-framework/.venv/bin/python',str(ROOT/target),*sys.argv[3:]],cwd=str(ROOT),env={'OPENBLAS_NUM_THREADS':'3','OMP_NUM_THREADS':'3','MKL_NUM_THREADS':'3'},source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    path=ROOT/(P+name+'_input.json')
    if path.exists():raise FileExistsError('Immutable attempt already exists: '+str(path))
    path.write_text(json.dumps(record,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(P+name+'_source_capsule.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,f.name)
    if '--resume' in sys.argv[3:] and (ROOT/(P+'result_partial.json')).exists():
        with zipfile.ZipFile(ROOT/(P+name+'_prior_outputs.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(ROOT.glob(P+'*.pkl')):
                if f.name not in (P+'model.pkl',P+'build_checkpoint.pkl'):z.write(f,f.name)
            for suffix in ('result.json','result_partial.json','build.json'):
                f=ROOT/(P+suffix)
                if f.exists():z.write(f,f.name)
    start=time.time()
    with (ROOT/(P+name+'_stdout.txt')).open('x') as out,(ROOT/(P+name+'_stderr.txt')).open('x') as err:
        child=subprocess.run(record['command'],cwd=ROOT,env=dict(os.environ,**record['env']),stdout=out,stderr=err)
    result=dict(natural_exit=child.returncode,elapsed_seconds=time.time()-start,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (ROOT/(P+name+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

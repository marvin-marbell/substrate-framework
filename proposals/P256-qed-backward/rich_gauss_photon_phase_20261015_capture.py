"""Pin exact shared INPUT and new consumer bytes, retain natural process exit."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_gauss_photon_phase_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    selected = [PREFIX+name for name in ('spectral.py','consume.py','capture.py','actual_input.json','equations.json')]
    selected += ['rich_open_gauss_carrier_20261015_model.py','rich_open_gauss_carrier_20261015_interface.json',
                 'rich_open_gauss_carrier_20261015_neutral_reference.npz',
                 'rich_photon_fullq_20261014_model.py','rich_photon_20261014_loop.py']
    pins = {name:dict(sha256=digest(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in selected}
    command = [PYTHON,PREFIX+'consume.py',PREFIX+'actual_input.json']
    env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start = time.time()
    (ROOT/(PREFIX+'actual_start.json')).write_text(json.dumps(dict(command=command,cwd=str(ROOT),platform=platform.platform(),started_unix=start,source_input_pins=pins,threads={k:env[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(PREFIX+'actual_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as capsule:
        for name in selected:
            capsule.write(ROOT/name,name)
    with (ROOT/(PREFIX+'actual_stdout.txt')).open('w') as out,(ROOT/(PREFIX+'actual_stderr.txt')).open('w') as err:
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err)
        code = process.wait()
    result = dict(command=command,returncode=code,natural_exit=code>=0,terminated_by_signal=-code if code<0 else None,
                  elapsed_seconds=time.time()-start,sources_and_shared_inputs_unchanged=all(digest(ROOT/n)==p['sha256'] for n,p in pins.items()),source_input_pins=pins)
    (ROOT/(PREFIX+'actual_exit.json')).write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(PREFIX+'actual_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as capsule:
        for name in selected:
            capsule.write(ROOT/name,name)
        for path in ROOT.glob(PREFIX+'*.json'):
            if path.name not in selected:
                capsule.write(path,path.name)
        for suffix in ('actual_stdout.txt','actual_stderr.txt','spectral_states.npz','actual_states.npz'):
            path = ROOT/(PREFIX+suffix)
            if path.exists():
                capsule.write(path,path.name)
    result['capsule_SHA256'] = digest(ROOT/(PREFIX+'actual_capsule.zip'))
    print(json.dumps(result),flush=True)
    return int(code!=0)


if __name__ == '__main__':
    sys.exit(main())

"""Capture one new consumer process, immutable input binding and natural exit."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
P = 'rich_review_wave28_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(name: str) -> str:
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def main() -> None:
    inputs = [P+'input.json',P+'consume.py',P+'capture.py',
              'rich_open_gauss_carrier_20261015_model.py',
              'rich_open_gauss_carrier_20261015_neutral_reference.npz',
              'rich_photon_fullq_20261014_model.py','rich_photon_20261014_loop.py']
    hashes = {name:digest(name) for name in inputs}
    command = [PYTHON,P+'consume.py']
    start = time.monotonic()
    (ROOT/(P+'start.json')).write_text(json.dumps(dict(command=command,cwd=str(ROOT),input_SHA256=hashes),indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(P+'source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as capsule:
        for name in inputs:
            capsule.write(ROOT/name,name)
    env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (ROOT/(P+'stdout.txt')).open('w') as out,(ROOT/(P+'stderr.txt')).open('w') as err:
        done = subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
    receipt = dict(command=command,returncode=done.returncode,natural_exit=done.returncode>=0,
                   elapsed_seconds=time.monotonic()-start,input_SHA256=hashes,
                   sources_and_inputs_unchanged=all(digest(name)==value for name,value in hashes.items()))
    (ROOT/(P+'exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    raise SystemExit(done.returncode)


if __name__ == '__main__':
    main()

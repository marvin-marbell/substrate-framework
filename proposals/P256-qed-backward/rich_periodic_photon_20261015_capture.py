"""Capture exact own sources/input, scientific subprocess and natural exit."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_periodic_photon_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def main():
    sources = [PREFIX+s for s in ('model.py','experiment.py','capture.py','input.json')]
    hashes = {s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}
    with zipfile.ZipFile(ROOT/(PREFIX+'source_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(ROOT/source, source)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    versions = subprocess.run([PYTHON,'-c','import sys,numpy,scipy; print(sys.version); print("numpy",numpy.__version__,"scipy",scipy.__version__)'],cwd=ROOT,env=environment,capture_output=True,text=True,check=True).stdout
    command = [PYTHON, '-u', PREFIX+'experiment.py']
    started = time.time()
    start = dict(command=command, cwd=str(ROOT), started=started, python=PYTHON, versions=versions, platform=platform.platform(), source_SHA256=hashes,
                 provenance='NEW OWN-H PERIODIC GRAPH/STATE/SOURCES; no ancestor producer/checker or evolved outputs executed')
    (ROOT/(PREFIX+'start.json')).write_text(json.dumps(start,indent=2)+'\n')
    with (ROOT/(PREFIX+'stdout.txt')).open('w') as out, (ROOT/(PREFIX+'stderr.txt')).open('w') as err:
        result = subprocess.run(command,cwd=ROOT,env=environment,stdout=out,stderr=err)
    receipt = dict(returncode=result.returncode,natural_exit=True,started=started,finished=time.time(),elapsed=time.time()-started,
                   source_SHA256=hashes,sources_unchanged=all(hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h for s,h in hashes.items()))
    (ROOT/(PREFIX+'exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()

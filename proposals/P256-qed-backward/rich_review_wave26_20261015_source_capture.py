"""Capture actual source-aligned continuation, including owning reference bytes."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
from rich_review_wave26_20261015_capture import digest, save, ROOT, PREFIX, PYTHON


def main() -> None:
    stem = PREFIX+'source_mode'
    sources = [PREFIX+'source_mode.py',PREFIX+'source_capture.py',PREFIX+'source_input.json',PREFIX+'consume.py',PREFIX+'capture.py',PREFIX+'input.json',
               PREFIX+'mode_states.npz',PREFIX+'mode_result.json',
               'rich_neutral_mott_gauge_20261015_model.py','rich_photon_fullq_20261014_model.py','rich_photon_20261014_loop.py']
    pins = {name:digest(ROOT/name) for name in sources}
    command = [PYTHON,PREFIX+'source_mode.py']
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    start = time.monotonic()
    save(ROOT/(stem+'_start.json'),dict(command=command,cwd=str(ROOT),source_SHA256=pins,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),single_threaded_blas=True))
    with zipfile.ZipFile(ROOT/(stem+'_source_capsule.zip'),'x',compression=zipfile.ZIP_DEFLATED) as capsule:
        for name in sources:
            capsule.write(ROOT/name,name)
    with (ROOT/(stem+'_stdout.txt')).open('xb') as stdout,(ROOT/(stem+'_stderr.txt')).open('xb') as stderr:
        done = subprocess.run(command,cwd=ROOT,env=environment,stdout=stdout,stderr=stderr)
    receipt = dict(command=command,returncode=done.returncode,natural_exit=done.returncode>=0,exited_successfully=done.returncode==0,
                   terminated_by_signal=-done.returncode if done.returncode<0 else None,elapsed_seconds=time.monotonic()-start,
                   source_SHA256=pins,sources_unchanged=all(digest(ROOT/name)==pin for name,pin in pins.items()),no_timeout_or_interruption_requested=True)
    save(ROOT/(stem+'_exit.json'),receipt)
    members = [stem+suffix for suffix in ('_start.json','_exit.json','_source_capsule.zip','_stdout.txt','_stderr.txt','_result.json','_states.npz') if (ROOT/(stem+suffix)).exists()]
    with zipfile.ZipFile(ROOT/(stem+'_capsule.zip'),'x',compression=zipfile.ZIP_DEFLATED) as capsule:
        for name in members:
            capsule.write(ROOT/name,name)
    save(ROOT/(stem+'_receipt.json'),dict(capsule=stem+'_capsule.zip',SHA256=digest(ROOT/(stem+'_capsule.zip')),members={name:digest(ROOT/name) for name in members},science=receipt))
    print(json.dumps(receipt),flush=True)
    raise SystemExit(done.returncode)


if __name__=='__main__':
    main()

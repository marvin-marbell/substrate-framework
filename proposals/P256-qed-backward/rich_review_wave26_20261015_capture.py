"""Exclusive-output science launch with exact source/input and natural-exit receipts."""
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
PREFIX = 'rich_review_wave26_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main() -> None:
    phase = sys.argv[1]
    if phase not in ('finite', 'mode'):
        raise ValueError('finite or mode required')
    stem = PREFIX+phase
    sources = [PREFIX+'consume.py', PREFIX+'capture.py', PREFIX+'input.json',
               'rich_neutral_mott_gauge_20261015_model.py', 'rich_photon_fullq_20261014_model.py', 'rich_photon_20261014_loop.py']
    pins = {name:digest(ROOT/name) for name in sources}
    command = [PYTHON, PREFIX+'consume.py', phase, PREFIX+'input.json']
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    start = time.monotonic()
    save(ROOT/(stem+'_start.json'), dict(command=command, cwd=str(ROOT), UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         source_SHA256=pins, platform=platform.platform(), python=PYTHON, single_threaded_blas=True))
    with zipfile.ZipFile(ROOT/(stem+'_source_capsule.zip'),'x',compression=zipfile.ZIP_DEFLATED) as capsule:
        for name in sources:
            capsule.write(ROOT/name,name)
    with (ROOT/(stem+'_stdout.txt')).open('xb') as stdout, (ROOT/(stem+'_stderr.txt')).open('xb') as stderr:
        done = subprocess.run(command,cwd=ROOT,env=environment,stdout=stdout,stderr=stderr)
    receipt = dict(command=command, returncode=done.returncode, natural_exit=done.returncode>=0,
                   exited_successfully=done.returncode==0, terminated_by_signal=-done.returncode if done.returncode<0 else None,
                   elapsed_seconds=time.monotonic()-start, source_SHA256=pins,
                   sources_unchanged=all(digest(ROOT/n)==p for n,p in pins.items()),
                   no_timeout_or_interruption_requested=True)
    save(ROOT/(stem+'_exit.json'),receipt)
    members = [stem+'_start.json',stem+'_exit.json',stem+'_source_capsule.zip',stem+'_stdout.txt',stem+'_stderr.txt']
    members += [name for name in (stem+'_result.json',stem+'_states.npz') if (ROOT/name).exists()]
    if phase=='finite' and (ROOT/(PREFIX+'provisional_synthesis.json')).exists():
        members.append(PREFIX+'provisional_synthesis.json')
    with zipfile.ZipFile(ROOT/(stem+'_capsule.zip'),'x',compression=zipfile.ZIP_DEFLATED) as capsule:
        for name in members:
            capsule.write(ROOT/name,name)
    save(ROOT/(stem+'_receipt.json'),dict(capsule=stem+'_capsule.zip', SHA256=digest(ROOT/(stem+'_capsule.zip')),
                                         members={name:digest(ROOT/name) for name in members}, science=receipt))
    print(json.dumps(receipt),flush=True)
    raise SystemExit(done.returncode)


if __name__ == '__main__':
    main()

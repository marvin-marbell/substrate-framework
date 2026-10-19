"""Exact source/input capture, natural or interrupted exit; no old CLI replay."""
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
PREFIX = 'rich_neutral_mott_gauge_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selected = [PREFIX+n for n in ('model.py', 'consume.py', 'capture.py', 'actual_input.json')]
    selected += ['rich_photon_fullq_20261014_model.py', 'rich_photon_20261014_loop.py']
    pins = {name: dict(sha256=digest(ROOT/name), bytes=(ROOT/name).stat().st_size) for name in selected}
    command = [PYTHON, PREFIX+'consume.py', PREFIX+'actual_input.json']
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start = time.time()
    (ROOT/(PREFIX+'actual_start.json')).write_text(json.dumps(dict(command=command, cwd=str(ROOT), platform=platform.platform(), started_unix=start, source_pins=pins, threads={k:env[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}), indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(PREFIX+'actual_source_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as capsule:
        for name in selected:
            capsule.write(ROOT/name, name)
    interruption = None
    with (ROOT/(PREFIX+'actual_stdout.txt')).open('w') as out, (ROOT/(PREFIX+'actual_stderr.txt')).open('w') as err:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err)
        try:
            code = process.wait(timeout=1800)
            natural = True
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            interruption = type(error).__name__
            process.terminate()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            natural = False
    result = dict(command=command, returncode=code, natural_exit=natural, interruption=interruption, elapsed_seconds=time.time()-start, sources_unchanged=all(digest(ROOT/n) == p['sha256'] for n, p in pins.items()), source_pins=pins)
    (ROOT/(PREFIX+'actual_exit.json')).write_text(json.dumps(result, indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(PREFIX+'actual_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as capsule:
        for name in selected:
            capsule.write(ROOT/name, name)
        for path in ROOT.glob(PREFIX+'actual_*'):
            if path.is_file() and path.suffix != '.zip' and path.name not in selected:
                capsule.write(path, path.name)
    result['capsule_sha256'] = digest(ROOT/(PREFIX+'actual_capsule.zip'))
    print(json.dumps(result), flush=True)
    return 0 if natural and code == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

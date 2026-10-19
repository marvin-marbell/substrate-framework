"""Seal source/input bytes before the one new physical consumer, record natural exit."""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent
P = 'rich_gauss_transverse_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def closure(names: list[str]) -> list[str]:
    found = set(names)
    pending = list(names)
    while pending:
        name = pending.pop()
        if not name.endswith('.py'):
            continue
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules = [node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            for module in modules:
                if module is None:
                    continue
                dependency = module.split('.')[0]+'.py'
                if (ROOT/dependency).is_file() and dependency not in found:
                    found.add(dependency)
                    pending.append(dependency)
    return sorted(found)


def main() -> None:
    names = closure([P+'consume.py',P+'capture.py',P+'input.json',
                     'rich_open_gauss_carrier_20261015_neutral_reference.npz'])
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    command = [PYTHON,P+'consume.py']
    with zipfile.ZipFile(ROOT/(P+'source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as cap:
        for name in names:
            cap.write(ROOT/name,name)
    start: dict[str, Any] = dict(command=command,input_SHA256=hashes,source_capsule_SHA256=hashlib.sha256((ROOT/(P+'source_capsule.zip')).read_bytes()).hexdigest(),
        interpreter=PYTHON,environment=dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
        reference_scope='Declared shared stationary Omega INPUT; no producer/checker replay, independent ground claim, or evolved ancestor data read')
    (ROOT/(P+'start.json')).write_text(json.dumps(start,indent=2)+'\n')
    clock = time.monotonic()
    with (ROOT/(P+'stdout.txt')).open('w') as out,(ROOT/(P+'stderr.txt')).open('w') as err:
        done = subprocess.run(command,cwd=ROOT,env=dict(os.environ,**start['environment']),stdout=out,stderr=err)
    receipt = dict(command=command,returncode=done.returncode,natural_exit=done.returncode>=0,
        elapsed_seconds=time.monotonic()-clock,input_SHA256=hashes,
        sources_and_inputs_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in hashes.items()))
    (ROOT/(P+'exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    raise SystemExit(done.returncode)


if __name__ == '__main__':
    main()

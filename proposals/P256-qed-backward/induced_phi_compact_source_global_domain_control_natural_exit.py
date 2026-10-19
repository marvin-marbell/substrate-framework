"""Persist actual global-domain science and final selected-checker natural exits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PREFIX = 'induced_phi_compact_source_global_domain_control'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def execute(command: list[str]) -> dict[str, Any]:
    start = time.monotonic()
    result = subprocess.run(command,cwd=PROJECT,text=True,capture_output=True,check=False)
    return {'command':command,'cwd':str(PROJECT),'natural_exit':result.returncode,
            'wall_seconds':time.monotonic()-start,'stdout':result.stdout,'stderr':result.stderr}


if __name__=='__main__':
    sources = [ROOT/(PREFIX+'.py'),Path(__file__).resolve()]
    before = {p.name:p.read_bytes() for p in sources}
    receipt = ROOT/(PREFIX+'_receipt.json')
    science = execute([PYTHON,str(sources[0]),'--output',str(receipt)])
    # Only after the science entrypoint and all edits; explicit file arguments.
    checker = execute(['/home/administrator/.local/bin/pyright','--pythonpath',PYTHON,'--stats',*[str(p) for p in sources]])
    stable = all(p.read_bytes()==before[p.name] for p in sources)
    capsule: dict[str, Any] = {'kind':'actual_GLOBAL_domain_science_and_selected_checker_natural_exit',
        'source_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in before.items()},
        'source_utf8':{name:raw.decode() for name,raw in before.items()},
        'science':science,'selected_pyright':checker,'selected_files':[str(p) for p in sources],
        'source_unchanged':stable,'runtime':{'executable':sys.executable,'python':sys.version},
        'stage':'conditional NEW packet domain closure; no original SOURCE, global ground or currenttransfer'}
    if science['natural_exit']==0:
        raw = receipt.read_bytes()
        data = json.loads(raw)
        capsule['receipt'] = {'path':receipt.name,'sha256':hashlib.sha256(raw).hexdigest()}
        capsule['input_sha256'] = data['input_sha256']
        capsule['immutable_input_archive'] = data['immutable_input_archive']
    output = ROOT/(PREFIX+'_natural_exit_capsule.json')
    output.write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps({'capsule':str(output),'capsule_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'science_natural_exit':science['natural_exit'],'checker_natural_exit':checker['natural_exit'],
        'source_unchanged':stable,'science_stdout':science['stdout'],'science_stderr':science['stderr'],
        'checker_stdout':checker['stdout'],'checker_stderr':checker['stderr']},indent=2))
    if not stable or science['natural_exit']!=0 or checker['natural_exit']!=0:
        raise SystemExit(1)

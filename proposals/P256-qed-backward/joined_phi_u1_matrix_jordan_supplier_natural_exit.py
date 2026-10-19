"""Bind this NEW Jordan supplier's actual CLI, natural exit and immutable bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_matrix_jordan_supplier'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    script = ROOT/(PREFIX+'.py')
    before = {path.name:digest(path) for path in (script,Path(__file__))}
    command = [PYTHON,str(script),'--output',str(output)]
    process = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=False)
    stdout_path = output.with_name(PREFIX+'_stdout_20261010.txt')
    stderr_path = output.with_name(PREFIX+'_stderr_20261010.txt')
    stdout_path.write_text(process.stdout)
    stderr_path.write_text(process.stderr)
    capsule:dict[str,Any] = {'command':command,'cwd':str(ROOT),'exit_code':process.returncode,
        'natural_exit':process.returncode>=0,'stdout':process.stdout,'stderr':process.stderr,
        'source_before':before,'source_after':{path.name:digest(path) for path in (script,Path(__file__))},
        'elapsed_seconds':time.monotonic()-started,'runtime_python':sys.executable}
    if process.returncode == 0:
        result = json.loads(output.read_bytes())
        capsule['receipt'] = {'path':output.name,'sha256':digest(output)}
        capsule['immutable_input_source_archive'] = result['immutable_archive']
        capsule['selected_science_input'] = result['selected_science_input']
        capsule['arrays'] = {'path':output.with_suffix('.npz').name,'sha256':digest(output.with_suffix('.npz'))}
    execution = output.with_name(PREFIX+'_execution_bytes_'+before[script.name][:16]+'.zip')
    with zipfile.ZipFile(execution,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (script,Path(__file__),stdout_path,stderr_path):
            archive.write(path,path.name)
        if process.returncode == 0:
            archive.write(output,output.name)
    capsule['execution_archive'] = {'path':execution.name,'sha256':digest(execution)}
    capsule_path = output.with_name(PREFIX+'_natural_exit_capsule_20261010.json')
    capsule_path.write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps({'capsule':capsule_path.name,'exit_code':process.returncode,
        'natural_exit':capsule['natural_exit'],'elapsed_seconds':capsule['elapsed_seconds']}))
    if process.returncode != 0:
        raise SystemExit(process.returncode)
    if capsule['source_before'] != capsule['source_after']:
        raise ArithmeticError('Owned execution source changed')
    return capsule


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.output)

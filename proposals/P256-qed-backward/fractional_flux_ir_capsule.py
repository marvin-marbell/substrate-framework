"""Capture preexecution bytes and a new natural-exit fractional-flux epoch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def archive(path: Path, payload: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in payload.items():
            z.writestr(name,raw)


def run(epoch: str, request_path: Path) -> int:
    assert epoch.startswith('fractional_flux_ir_') and '/' not in epoch
    paths = {k:ROOT/(epoch+'_'+k) for k in
             ('preexecution.zip','executed.zip','stdout.txt','stderr.txt','exit.json','receipt.json','capsule.json')}
    assert not any(p.exists() for p in paths.values()), 'Immutable epoch already exists'
    request = json.loads(request_path.read_text())
    names = list(request['input_sha256']) + [request_path.name,Path(__file__).name,'fractional_flux_ir_consumer.py']
    captured = {name:(ROOT/name).read_bytes() for name in dict.fromkeys(names)}
    for name,pin in request['input_sha256'].items():
        assert sha(captured[name]) == pin, name
    argv = [PYTHON,str(ROOT/'fractional_flux_ir_consumer.py'),'--request',str(request_path),
            '--output',str(paths['receipt.json'])]
    env = os.environ.copy()
    env.update({'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    pre: dict[str, Any] = {'kind':'ACTUAL_PREEXECUTION_NEW_SCIENTIFIC_CLI', 'argv':argv,
                          'cwd':str(ROOT),'source_and_input_sha256':{k:sha(v) for k,v in captured.items()},
                          'boundary':'Only new fractional consumer executed. Parent snapshots/receipts are inputs, not runtime preexecution or native ground proof.',
                          'python':PYTHON,'env_overrides':{k:env[k] for k in ('PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}}
    captured['fractional_flux_ir_preexecution_manifest.json'] = (json.dumps(pre,indent=2)+'\n').encode()
    archive(paths['preexecution.zip'],captured)
    started=time.monotonic()
    proc=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,check=False)
    elapsed=time.monotonic()-started
    paths['stdout.txt'].write_bytes(proc.stdout); paths['stderr.txt'].write_bytes(proc.stderr)
    unchanged = all((ROOT/name).read_bytes()==raw for name,raw in captured.items()
                    if name!='fractional_flux_ir_preexecution_manifest.json')
    exit_info={'kind':'NEW_SCIENTIFIC_NATURAL_EXIT','returncode':proc.returncode,
               'natural_exit':proc.returncode>=0,'elapsed_seconds':elapsed,'argv':argv,
               'sources_inputs_unchanged':unchanged,'preexecution_sha256':sha(paths['preexecution.zip'].read_bytes()),
               'stdout_sha256':sha(proc.stdout),'stderr_sha256':sha(proc.stderr),
               'receipt_present':paths['receipt.json'].exists(),
               'receipt_sha256':sha(paths['receipt.json'].read_bytes()) if paths['receipt.json'].exists() else None}
    paths['exit.json'].write_text(json.dumps(exit_info,indent=2)+'\n')
    executed = dict(captured)
    for key in ('stdout.txt','stderr.txt','exit.json','receipt.json'):
        if paths[key].exists():executed[paths[key].name]=paths[key].read_bytes()
    archive(paths['executed.zip'],executed)
    capsule={'epoch':epoch,**exit_info,'executed_sha256':sha(paths['executed.zip'].read_bytes()),
             'files':{k:p.name for k,p in paths.items()},'failed_epochs_preserved':True,
             'build_lint_tests_selected_diagnostics_run':False,
             'sharp_photon':False,'campaign_complete':False}
    paths['capsule.json'].write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps(capsule,indent=2))
    return proc.returncode if unchanged else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch',required=True)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args()
    sys.exit(run(args.epoch,args.request))

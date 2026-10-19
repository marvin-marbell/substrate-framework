"""Natural-exit capture for the independent review, preserving unsuccessful epochs."""
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


def save_zip(path: Path, payload: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(payload):
            info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info,payload[name])


def run(epoch: str) -> int:
    if not epoch.startswith('weyl_ground_review_') or '/' in epoch:
        raise ValueError('Own prefix only')
    paths = {k: ROOT/(epoch+'_'+k) for k in ('preexecution.zip','executed.zip','stdout.txt','stderr.txt','exit.json','receipt.json','capsule.json')}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError('Immutable epoch exists')
    names = ['weyl_ground_review_control_20261011.py',Path(__file__).name,
             'weyl_ground_review_verdict_20261011.json',
             'vacuum_weyl_parent_review_original_targets_20261011.zip',
             'fractional_flux_ir_actual4_20261011_preexecution.zip',
             'fractional_flux_ir_actual4_20261011_executed.zip']
    captured = {n:(ROOT/n).read_bytes() for n in names}
    argv = [PYTHON,str(ROOT/names[0]),'--output',str(paths['receipt.json'])]
    env = os.environ.copy()
    env.update({'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    manifest = {'kind':'RUNTIME_PREEXECUTION_INDEPENDENT_REVIEW','argv':argv,'cwd':str(ROOT),
                'sha256':{n:sha(raw) for n,raw in captured.items()},
                'no_old_main_solver_checker':True,'fixed_zip_metadata':True}
    payload = dict(captured)
    payload['weyl_ground_review_runtime_manifest.json'] = (json.dumps(manifest,indent=2)+'\n').encode()
    save_zip(paths['preexecution.zip'],payload)
    start = time.monotonic()
    proc = subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,check=False)
    elapsed = time.monotonic()-start
    unchanged = all((ROOT/n).read_bytes()==raw for n,raw in captured.items())
    paths['stdout.txt'].write_bytes(proc.stdout)
    paths['stderr.txt'].write_bytes(proc.stderr)
    exit_info: dict[str,Any] = {'returncode':proc.returncode,'natural_exit':proc.returncode>=0,
                 'elapsed_seconds':elapsed,'argv':argv,'sources_inputs_unchanged':unchanged,
                 'stdout_sha256':sha(proc.stdout),'stderr_sha256':sha(proc.stderr),
                 'receipt_sha256':sha(paths['receipt.json'].read_bytes()) if paths['receipt.json'].exists() else None,
                 'preexecution_sha256':sha(paths['preexecution.zip'].read_bytes()),
                 'python':PYTHON,'scientific_failure_preserved':proc.returncode != 0}
    paths['exit.json'].write_text(json.dumps(exit_info,indent=2)+'\n')
    for k in ('stdout.txt','stderr.txt','exit.json','receipt.json'):
        if paths[k].exists():
            payload[paths[k].name] = paths[k].read_bytes()
    save_zip(paths['executed.zip'],payload)
    capsule = {'kind':'REVISION_BOUND_WEYL_REVIEW_NATURAL_EXIT_CAPSULE','epoch':epoch,**exit_info,
               'executed_sha256':sha(paths['executed.zip'].read_bytes()),
               'files':{k:p.name for k,p in paths.items()},'failed_epochs_preserved':True,
               'selected_checker_build_lint_tests_formatters_run':False,'native_ground_or_pole_admitted':False}
    paths['capsule.json'].write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps(capsule,indent=2))
    return proc.returncode if unchanged else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch',required=True)
    sys.exit(run(parser.parse_args().epoch))

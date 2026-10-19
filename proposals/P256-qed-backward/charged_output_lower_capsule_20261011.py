from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
SOURCE = ROOT/'charged_output_lower_certificate_20261011.py'
INPUTS = [
    ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json',
    ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json',
    ROOT/'charged_spinor_joint_free_phase_control_20261011_receipt.json',
    ROOT/'charged_spinor_joint_free_phase_control_20261011.py',
]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze(p, data):
    with p.open('x') as f:
        json.dump(data,f,indent=2)
        f.write('\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag',required=True)
    p.add_argument('--floor',default='-5110')
    p.add_argument('--output-budget',default='240')
    p.add_argument('--support-budget',default='322')
    p.add_argument('--inventory',action='store_true')
    a = p.parse_args()
    assert a.tag.startswith('charged_output_lower_')
    prefix = ROOT/a.tag
    archive = ROOT/f'{a.tag}_frozen.zip'
    paths = [SOURCE,Path(__file__),*INPUTS]
    hashes = {x.name:digest(x) for x in paths}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for x in paths:z.writestr(x.name,x.read_bytes())
    command = [PYTHON,str(SOURCE),'--tag',a.tag,'--floor',a.floor,'--output-budget',a.output_budget,'--support-budget',a.support_budget]
    if a.inventory:command.append('--inventory')
    pre = {'kind':'IMMUTABLE_PREEXECUTION','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'command':command,'cwd':str(ROOT),'source_input_sha256':hashes,'archive_path':archive.name,'archive_sha256':digest(archive),
           'python_realpath':os.path.realpath(PYTHON),'plan_id':'issue-222-induced-photon-delta13-continuation-20261008',
           'plan_governor':'bc2b76efb96c6337de1ed0a5c4ef650824f52609a59c5a048f674725c9e38f6f',
           'template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe','work_item':'induced-photon'}
    prepath = ROOT/f'{a.tag}_preexecution.json'
    freeze(prepath,pre)
    start = time.monotonic()
    logfile = ROOT/f'{a.tag}_stdout.log'
    with logfile.open('x') as stream:
        child = subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                 env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        execpath = ROOT/f'{a.tag}_executed.json'
        freeze(execpath,{'kind':'IMMUTABLE_EXECUTED','pid':child.pid,'command':command,'preexecution_sha256':digest(prepath),
                         'source_input_sha256':hashes,'archive_sha256':digest(archive)})
        code = child.wait()
    outputs = list(ROOT.glob(f'{a.tag}_*receipt.json'))+list(ROOT.glob(f'{a.tag}_*witness.npz'))+list(ROOT.glob(f'{a.tag}_*exact_matrix.json'))
    exitpath = ROOT/f'{a.tag}_natural_exit.json'
    freeze(exitpath,{'kind':'IMMUTABLE_NATURAL_EXIT','returncode':code,'elapsed_seconds':time.monotonic()-start,
                     'preexecution_path':prepath.name,'preexecution_sha256':digest(prepath),'executed_path':execpath.name,'executed_sha256':digest(execpath),
                     'archive_path':archive.name,'archive_sha256':digest(archive),'archive_unchanged':digest(archive)==pre['archive_sha256'],
                     'source_input_sha256':hashes,'source_inputs_unchanged':all(digest(x)==hashes[x.name] for x in paths),
                     'stdout_path':logfile.name,'stdout_sha256':digest(logfile),'output_sha256':{x.name:digest(x) for x in outputs}})
    print(json.dumps({'exit_path':exitpath.name,'exit_sha256':digest(exitpath),'returncode':code,'elapsed_seconds':time.monotonic()-start}))
    sys.exit(code)


if __name__ == '__main__':
    main()

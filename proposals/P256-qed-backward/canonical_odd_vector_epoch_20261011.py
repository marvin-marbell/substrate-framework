from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
SOURCE = ROOT/'canonical_odd_vector_control_20261011.py'
INPUTS = ['neutral_return_lower_control_20261011.py', 'neutral_return_lower_boundary_20261011.py', 'neutral_return_lower_boundary_actual2_20261011_receipt.json', 'neutral_return_image_API_20261011.json', 'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json', 'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json', 'local_so5_resum_actual04_20261011_state.npz']


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    assert args.tag.startswith('canonical_odd_vector_')
    tag = args.tag
    cmd = [PYTHON, str(SOURCE), '--tag', tag]
    pins = {f:digest(ROOT/f) for f in INPUTS}
    source_sha = digest(SOURCE)
    epoch_sha = digest(Path(__file__))
    pre = ROOT/f'{tag}_preexecution.zip'
    with zipfile.ZipFile(pre, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(SOURCE, SOURCE.name)
        z.write(Path(__file__), Path(__file__).name)
        z.write(ROOT/'canonical_odd_vector_language_20261011.json', 'canonical_odd_vector_language_20261011.json')
        for f in INPUTS:
            if not f.endswith('.npz'):
                z.write(ROOT/f, f)
        z.writestr('manifest.json', json.dumps({'command':cmd, 'inputs_sha256':pins, 'source_sha256':source_sha, 'epoch_sha256':epoch_sha, 'state_capture':'Existing immutable source NPZ bound by hash at exact root, not duplicated; not used for sector lower form.'}, indent=2))
    started = time.time()
    with (ROOT/f'{tag}_stdout.log').open('xb') as out, (ROOT/f'{tag}_stderr.log').open('xb') as err:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'), check=False)
    natural = {'command':cmd, 'started_unix':started, 'ended_unix':time.time(), 'natural_exit_code':proc.returncode, 'source_sha256':source_sha, 'epoch_sha256':epoch_sha, 'inputs_sha256':pins, 'source_unchanged':digest(SOURCE)==source_sha, 'epoch_unchanged':digest(Path(__file__))==epoch_sha, 'inputs_unchanged':all(digest(ROOT/f)==h for f,h in pins.items()), 'preexecution_zip_sha256':digest(pre)}
    natural_path = ROOT/f'{tag}_natural_exit.json'
    with natural_path.open('x') as f:
        json.dump(natural, f, indent=2)
    executed = ROOT/f'{tag}_executed.zip'
    with zipfile.ZipFile(executed, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pre, pre.name)
        for f in ROOT.glob(tag+'_*'):
            if f not in (pre, executed):
                z.write(f, f.name)
    print(json.dumps({'tag':tag, 'natural_exit_code':proc.returncode, 'elapsed_seconds':natural['ended_unix']-started, 'source_sha256':source_sha, 'preexecution_zip_sha256':digest(pre), 'executed_zip_sha256':digest(executed), 'natural_exit_sha256':digest(natural_path)}), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()

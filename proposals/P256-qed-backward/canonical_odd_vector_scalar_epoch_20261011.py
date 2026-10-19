from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'canonical_odd_vector_scalar_transfer_20261011.py'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tag',required=True)
    args=p.parse_args(); tag=args.tag
    assert tag.startswith('canonical_odd_vector_')
    parent_name='canonical_odd_vector_actual1_20261011_receipt.json'
    parent=json.loads((ROOT/parent_name).read_text())
    names=set(parent['input_sha256']) | {parent_name,'canonical_odd_vector_control_20261011.py','canonical_odd_vector_proof_20261011.json'}
    for role in parent['role_assignments']:
        for b in role['parity_blocks']:
            names.update((b['matrix_path'],b['witness_path']))
    pins={f:digest(ROOT/f) for f in sorted(names)}
    own=digest(Path(__file__)); source=digest(SOURCE)
    cmd=[PYTHON,str(SOURCE),'--tag',tag]
    pre=ROOT/f'{tag}_preexecution.zip'
    with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(SOURCE,SOURCE.name); z.write(Path(__file__),Path(__file__).name)
        for f in sorted(names):
            if f!='local_so5_resum_actual04_20261011_state.npz': z.write(ROOT/f,f)
        z.writestr('manifest.json',json.dumps({'command':cmd,'source_sha256':source,'epoch_sha256':own,'input_sha256':pins,'state_capture':'Original harmonic NPZ SHA-bound at project path, not used for the exact scalar restriction.'},indent=2))
    start=time.time()
    with (ROOT/f'{tag}_stdout.log').open('xb') as out,(ROOT/f'{tag}_stderr.log').open('xb') as err:
        proc=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=err,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),check=False)
    natural={'command':cmd,'natural_exit_code':proc.returncode,'started_unix':start,'ended_unix':time.time(),'source_sha256':source,'epoch_sha256':own,'input_sha256':pins,'source_unchanged':digest(SOURCE)==source,'epoch_unchanged':digest(Path(__file__))==own,'inputs_unchanged':all(digest(ROOT/f)==h for f,h in pins.items()),'preexecution_zip_sha256':digest(pre)}
    natural_path=ROOT/f'{tag}_natural_exit.json'
    with natural_path.open('x') as f: json.dump(natural,f,indent=2)
    executed=ROOT/f'{tag}_executed.zip'
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pre,pre.name)
        for f in ROOT.glob(tag+'_*'):
            if f not in (pre,executed): z.write(f,f.name)
    print(json.dumps({'natural_exit_code':proc.returncode,'source_sha256':source,'preexecution_zip_sha256':digest(pre),'executed_zip_sha256':digest(executed),'natural_exit_sha256':digest(natural_path)}),flush=True)
    raise SystemExit(proc.returncode)


if __name__=='__main__':
    main()

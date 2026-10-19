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
SOURCE = ROOT/'neutral_return_lower_control_20261011.py'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag',required=True)
    p.add_argument('--kind',required=True)
    p.add_argument('--roles',required=True)
    p.add_argument('--output-budget',type=int,default=120)
    p.add_argument('--support-budget',type=int,default=202)
    p.add_argument('--inventory',action='store_true')
    args = p.parse_args()
    cmd = [PYTHON,str(SOURCE),'--kind',args.kind,'--roles',args.roles,'--tag',args.tag,'--output-budget',str(args.output_budget),'--support-budget',str(args.support_budget)]
    if args.inventory: cmd.append('--inventory')
    tag = args.tag+'_'+args.kind+'_'+args.roles
    inputs = ['local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json','local_so5_ground_upper_complete_state_consumer_20261011_receipt.json','local_so5_resum_actual04_20261011_state.npz']
    pins = {f:digest(ROOT/f) for f in inputs}
    pre = ROOT/f'{tag}_preexecution.zip'
    with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(SOURCE,SOURCE.name)
        z.write(Path(__file__),Path(__file__).name)
        for f in inputs[:2]: z.write(ROOT/f,f)
        z.writestr('manifest.json',json.dumps({'command':cmd,'inputs_sha256':pins,'source_sha256':digest(SOURCE),'state_capture':'Existing immutable NPZ remains at pinned absolute project path; SHA256 bound, not duplicated in every ZIP.'},indent=2))
    started = time.time()
    with (ROOT/f'{tag}_stdout.log').open('xb') as out, (ROOT/f'{tag}_stderr.log').open('xb') as err:
        env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        proc = subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=err,env=env,check=False)
    natural = {'command':cmd,'started_unix':started,'ended_unix':time.time(),'natural_exit_code':proc.returncode,'source_sha256':digest(SOURCE),'input_sha256':pins,'inputs_unchanged':all(digest(ROOT/f)==pin for f,pin in pins.items()),'preexecution_zip_sha256':digest(pre)}
    natural_path = ROOT/f'{tag}_natural_exit.json'
    with natural_path.open('x') as f: json.dump(natural,f,indent=2)
    executed = ROOT/f'{tag}_executed.zip'
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pre,pre.name)
        for f in ROOT.glob(tag+'_*'):
            if f != executed and f != pre: z.write(f,f.name)
    print(json.dumps({'tag':tag,'natural_exit_code':proc.returncode,'elapsed_seconds':natural['ended_unix']-started,'executed_zip_sha256':digest(executed)}),flush=True)
    raise SystemExit(proc.returncode)


if __name__=='__main__':
    main()

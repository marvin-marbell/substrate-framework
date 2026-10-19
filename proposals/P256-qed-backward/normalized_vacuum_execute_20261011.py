"""Archive a new normalized-vacuum source/input epoch, execute, record natural exit."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True);args=parser.parse_args()
    prefix='normalized_vacuum_'+args.epoch
    source=ROOT/'normalized_vacuum_scientist_20261011.py'
    files={source,Path(__file__),ROOT/'normalized_vacuum_proof_20261011.json'}
    pending=[source]
    while pending:
        path=pending.pop()
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                local=ROOT/(name.split('.')[0]+'.py')
                if local.is_file() and local not in files:
                    files.add(local);pending.append(local)
    inputs=['charged_cell_heat_typed3_20261011_interface.json','charged_first_shell_actual04_20261011_receipt.json',
            'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json',
            'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz',
            'matching_defect_interface_20261011.json','matching_defect_projector_notation_addendum_20261011.json']
    files.update(ROOT/name for name in inputs)
    bindings={
        'analytic_flux_time_scientist_20261011.py':'analytic_flux_time_typed4_20261011_executed.zip',
        'charged_cell_heat_typed3_20261011_interface.json':'charged_cell_heat_typed3_20261011_executed.zip',
        'charged_first_shell_actual04_20261011_receipt.json':'charged_first_shell_actual04_20261011_receipt_executed.zip',
        'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json':'local_so5_rep_review_recovered_capsule_20261011.zip'}
    archive_bind=[]
    for name,archive in bindings.items():
        with zipfile.ZipFile(ROOT/archive) as z:
            candidates=[m for m in z.namelist() if Path(m).name==name]
            matched=[m for m in candidates if hashlib.sha256(z.read(m)).hexdigest()==sha(ROOT/name)]
            assert matched,(name,archive,candidates)
        archive_bind.append({'source_or_input':name,'sha256':sha(ROOT/name),'archive':archive,'member':matched[0],'member_equal':True})
    pins={path.name:sha(path) for path in sorted(files)}
    request=ROOT/(prefix+'_request.json');receipt=ROOT/(prefix+'_receipt.json')
    command=[PYTHON,str(source),'--request',str(request),'--output',str(receipt)]
    payload={'kind':'NEW_NORMALIZED_VACUUM_IMMUTABLE_EXECUTION_REQUEST','pins':pins,'archive_revision_bindings':archive_bind,'command':command,
             'plan_revision':'2ec1fe2f4ebfad5032c59dd9ed0d6ef3a57d15e58247f175c670c5034c7d6208','no_historical_main_execution':True}
    with request.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    pre=ROOT/(prefix+'_preexecution.zip')
    with zipfile.ZipFile(pre,'x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files|{request}):z.write(path,path.name)
    start=time.monotonic()
    stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt')
    env=dict(os.environ);env.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
    with stdout.open('xb') as out,stderr.open('xb') as err:
        completed=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err,check=False)
    core={'kind':'NEW_NORMALIZED_VACUUM_NATURAL_EXIT','returncode':completed.returncode,'natural_exit':True,
          'elapsed_seconds':time.monotonic()-start,'preexecution_zip':pre.name,'preexecution_sha256':sha(pre),
          'source_and_input_unchanged':all(sha(ROOT/name)==pin for name,pin in pins.items()),
          'receipt':receipt.name,'receipt_sha256':sha(receipt) if receipt.exists() else None,
          'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr)}
    exitpath=ROOT/(prefix+'_exit.json')
    with exitpath.open('x') as f:json.dump(core,f,indent=2);f.write('\n')
    executed=ROOT/(prefix+'_executed.zip')
    with zipfile.ZipFile(executed,'x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files|{request,stdout,stderr,exitpath}|({receipt} if receipt.exists() else set())):z.write(path,path.name)
    capsule=core|{'executed_zip':executed.name,'executed_sha256':sha(executed),'exit_core_sha256':sha(exitpath)}
    with (ROOT/(prefix+'_capsule.json')).open('x') as f:json.dump(capsule,f,indent=2);f.write('\n')
    print(json.dumps(capsule,indent=2))
    raise SystemExit(completed.returncode)


if __name__=='__main__':
    main()

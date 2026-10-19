"""Immutable before-execution source/input capsules for NEW scientific epochs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
CONTRACT=Path('/home/administrator/.omp/agent/sessions/-/2026-10-09T16-26-59-293Z_01a1217d-415d-7376-a17d-16be721cbe28/local/issue222-all-auxiliary-ground-sector-fullquantum-loop-support-contract-20261012.json')
PROFILE='joined_phi_u1_uniform_gauss_limit_epoch4_20261010_actual_executed_inputs.zip'
MEMBER='joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True);args=parser.parse_args()
    prefix=f'auxiliary_sector_energy_{args.epoch}_20261012'
    request_path=ROOT/f'{prefix}_request.json'
    assert not request_path.exists(),'Immutable epoch already exists'
    names=['auxiliary_sector_energy_consumer_20261012.py',Path(__file__).name,
           'auxiliary_sector_energy_proof_20261012.json','auxiliary_sector_energy_language_20261012.json',
           'joined_phi_u1_uniform_sharp_gauss_proof_20261010.json','common_energy_bound_review_proof_20261012.json',
           'common_singlet_energy_proof_20261012.json','native_loop_dark_auxiliary_proof_20261012.json',
           'canonical_ground_parent_auxiliary_schur_scope_20261012.json',PROFILE]
    if args.epoch != 'actual1':
        names.extend(['auxiliary_sector_energy_epoch1_correction_20261012.json',
                      'auxiliary_sector_energy_actual1_20261012_capsule.json',
                      'auxiliary_sector_energy_actual1_20261012_executed.zip'])
    if args.epoch not in ('actual1','actual2'):
        names.extend(['auxiliary_sector_energy_epoch2_correction_20261012.json',
                      'auxiliary_sector_energy_actual2_20261012_capsule.json',
                      'auxiliary_sector_energy_actual2_20261012_executed.zip'])
    if args.epoch not in ('actual1','actual2','actual3'):
        names.extend(['auxiliary_sector_energy_epoch3_correction_20261012.json',
                      'auxiliary_sector_energy_actual3_20261012_capsule.json',
                      'auxiliary_sector_energy_actual3_20261012_executed.zip'])
    if args.epoch not in ('actual1','actual2','actual3','actual4'):
        names.extend(['auxiliary_sector_energy_physical_sewing_successor_20261012.json',
                      'auxiliary_sector_energy_actual4_20261012_capsule.json',
                      'auxiliary_sector_energy_actual4_20261012_executed.zip'])
    captured={name:(ROOT/name).read_bytes() for name in names}
    captured[str(CONTRACT)]=CONTRACT.read_bytes()
    assert sha(captured[str(CONTRACT)])=='e37aaf344e17cb2bf2bb07623fdf6fd5571b0e840df74a3dcb3f51b751c38254'
    assert sha(captured[PROFILE])=='8af45d28d48ae0ab8302dcea11cdb9356af972a5373ae2568008ff0a99587649'
    contract=json.loads(captured[str(CONTRACT)])
    request={'consumer':names[0],'profile_archive':PROFILE,'profile_member':MEMBER,
             'profile_member_sha256':contract['profile_member_sha256'],
             'governor':{key:contract[key] for key in ('plan_id','captured_revision','accepted_scoped_science','template_id','template_revision','work_item_id','requirement_ids')},
             'pins_before':{name:sha(raw) for name,raw in captured.items()},'old_main_or_ground_solver_run':False}
    with request_path.open('x') as stream:json.dump(request,stream,indent=2);stream.write('\n')
    pre=ROOT/f'{prefix}_preexecution.zip'
    with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in captured.items():archive.writestr(name.lstrip('/'),raw)
        archive.write(request_path,request_path.name)
    start=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
    receipt=ROOT/f'{prefix}_receipt.json'
    command=[PYTHON,str(ROOT/names[0]),'--request',str(request_path),'--output',str(receipt)]
    completed=subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
    elapsed=time.monotonic()-tick
    stdout,stderr,exitpath=[ROOT/f'{prefix}_{suffix}' for suffix in ('stdout.txt','stderr.txt','exit.json')]
    with stdout.open('xb') as stream:stream.write(completed.stdout)
    with stderr.open('xb') as stream:stream.write(completed.stderr)
    exitdata={'command':command,'returncode':completed.returncode,'natural_exit':True,'started_at':start,
              'completed_at':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':elapsed}
    with exitpath.open('x') as stream:json.dump(exitdata,stream,indent=2);stream.write('\n')
    after={name:sha((ROOT/name).read_bytes()) for name in captured}
    unchanged=after==request['pins_before']
    outputs=[request_path,pre,stdout,stderr,exitpath]
    if receipt.exists():outputs.append(receipt)
    executed=ROOT/f'{prefix}_executed.zip'
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in captured.items():archive.writestr(name.lstrip('/'),raw)
        for path in outputs:archive.write(path,path.name)
    capsule={'kind':'NEW_NATIVE_AUXILIARY_SECTOR_ENERGY_TWO_CONSTRUCTION_CAPSULE',
             'pins_before':request['pins_before'],'pins_after':after,'unchanged':unchanged,
             'preexecution':{'path':pre.name,'sha256':sha(pre.read_bytes())},
             'executed':{'path':executed.name,'sha256':sha(executed.read_bytes())},
             'outputs':{path.name:sha(path.read_bytes()) for path in outputs},'runtime':exitdata,
             'failed_epoch_preserved':completed.returncode!=0,'native_ground_backend_run':False,'campaign_complete':False}
    capsule_path=ROOT/f'{prefix}_capsule.json'
    with capsule_path.open('x') as stream:json.dump(capsule,stream,indent=2);stream.write('\n')
    for path in outputs+[executed,capsule_path]:path.chmod(0o444)
    print(json.dumps({'epoch':prefix,'returncode':completed.returncode,'natural_exit':True,'unchanged':unchanged,'elapsed_seconds':elapsed}))
    if not unchanged:raise RuntimeError('Pinned sources changed during epoch; capsule not admissible')
    sys.exit(completed.returncode)


if __name__=='__main__':
    main()

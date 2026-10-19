"""Exclusive native execution epochs; preserve pre/executed bytes and exit."""
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
import physical_ground_lift_source_20261011 as source
ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
INPUTS=[source.STATE,source.primitive.INPUTS[0],source.primitive.INPUTS[1],
 'local_so5_resum_actual04_20261011_receipt.json',
 'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json',
 'physical_singlet_circuit_delivery_20261011.json',
 'physical_singlet_circuit_epoch4_proof_20261011.json',
 'joined_phi_u1_full_physical_source_local_ground_parent_packet_20261011.json',
 'harmonic_car_bridge_derivation_20261011.json',
 'physical_source_binding_parent_actual1_20261011_receipt.json']

def archive(path:Path,members:dict[str,bytes])->None:
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,raw in members.items():z.writestr(name,raw)
    path.chmod(0o444)

def run(epoch:str)->None:
    prefix='physical_ground_lift_'+epoch+'_20261011'
    members={Path(__file__).name:Path(__file__).read_bytes()}
    members.update({name:(ROOT/name).read_bytes() for name in INPUTS})
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).parent==ROOT and str(path).endswith('.py'):
            members[Path(path).name]=Path(path).read_bytes()
    pins={name:hashlib.sha256(raw).hexdigest() for name,raw in members.items()}
    manifest={'kind':'NEW_PHYSICAL_LIFT_IMMUTABLE_EXECUTION_REQUEST','epoch':epoch,'source_input_sha256':pins,'Python':PYTHON,'import_policy':'only guarded pure helper modules; old complete-state consumer never imported or executed','governor':{'plan':'issue-222-induced-photon-delta13-continuation-20261008','revision':'c66121a88fb191d0f878c16e6fc6a5611e2d1260c4927d1d022d57149106137b','work_item':'induced-photon'}}
    manifest_raw=(json.dumps(manifest,indent=2)+'\n').encode()
    (ROOT/(prefix+'_request.json')).write_bytes(manifest_raw)
    members[prefix+'_request.json']=manifest_raw
    archive(ROOT/(prefix+'_preexecution.zip'),members)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[PYTHON,str(ROOT/'physical_ground_lift_source_20261011.py'),'--epoch',epoch]
    began=time.monotonic();p=subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
    (ROOT/(prefix+'_stdout.txt')).write_bytes(p.stdout)
    (ROOT/(prefix+'_stderr.txt')).write_bytes(p.stderr)
    unchanged=all((ROOT/name).read_bytes()==raw for name,raw in members.items())
    result={'command':command,'returncode':p.returncode,'natural_exit':True,'elapsed_seconds':time.monotonic()-began,'source_inputs_unchanged':unchanged,'source_input_sha256':pins,'stdout_sha256':hashlib.sha256(p.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(p.stderr).hexdigest()}
    raw=(json.dumps(result,indent=2)+'\n').encode()
    (ROOT/(prefix+'_natural_exit.json')).write_bytes(raw)
    executed=dict(members)
    for suffix in ['_stdout.txt','_stderr.txt','_natural_exit.json','_proof.json','_receipt.json']:
        path=ROOT/(prefix+suffix)
        if path.exists():executed[path.name]=path.read_bytes()
    archive(ROOT/(prefix+'_executed.zip'),executed)
    print(json.dumps(result,indent=2),flush=True)
    if p.returncode!=0:print(p.stderr.decode(),file=sys.stderr)
    if not unchanged:raise RuntimeError('source/input mutation during epoch')
    raise SystemExit(p.returncode)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--epoch',required=True);run(a.parse_args().epoch)

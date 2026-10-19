"""Capture this new consumer's exact selected source/input and natural exit."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='ground_magnetic_order_'
EPOCH=PREFIX+'actual1_20261012'
CONTRACT=Path('/home/administrator/.omp/agent/sessions/-/2026-10-09T16-26-59-293Z_01a1217d-415d-7376-a17d-16be721cbe28/local/issue222-jordan-loop-ground-magnetic-order-contract-20261012.json')
NAMES=[PREFIX+'consumer_20261012.py',PREFIX+'execute_20261012.py',PREFIX+'proof_20261012.json',PREFIX+'language_20261012.json',
       'rotation_band.py','magnetic_mechanism_proof_20261011.json','joined_phi_u1_closed_inverse_transverse_parent_packet_20261012.json',
       'joined_phi_u1_global_ground_frobenius_actual_20261010.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json']

def digest(data: bytes) -> str:return hashlib.sha256(data).hexdigest()
def save(name: str, value: object) -> None:(ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
def archive(name: str, members: dict[str,bytes]) -> str:
    with zipfile.ZipFile(ROOT/name,'w',zipfile.ZIP_DEFLATED) as z:
        for key,data in members.items():z.writestr(key,data)
    return digest((ROOT/name).read_bytes())

def main() -> None:
    members={name:(ROOT/name).read_bytes() for name in NAMES}
    members['shared_contract.json']=CONTRACT.read_bytes()
    assert digest(members['shared_contract.json'])=='9ac8ed02318d385c540d3ea2c451fdd0f6330900923cdecccc33cdd250b6bd99'
    pre=EPOCH+'_preexecution.zip';prepin=archive(pre,members)
    actual=PREFIX+'actual_20261012.json'
    command=[sys.executable,str(ROOT/(PREFIX+'consumer_20261012.py')),'--output',str(ROOT/actual)]
    request={'command':command,'cwd':str(ROOT),'input_sha256':{k:digest(v) for k,v in members.items()},'preexecution':{'path':pre,'sha256':prepin},'no_old_producer_or_solver_run':True}
    save(EPOCH+'_request.json',request)
    start=time.monotonic();child=subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
    elapsed=time.monotonic()-start
    (ROOT/(EPOCH+'_stdout.txt')).write_bytes(child.stdout);(ROOT/(EPOCH+'_stderr.txt')).write_bytes(child.stderr)
    exit_record={'returncode':child.returncode,'natural_exit':True,'elapsed_seconds':elapsed,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save(EPOCH+'_exit.json',exit_record)
    unchanged=all((ROOT/name).read_bytes()==members[name] for name in NAMES) and CONTRACT.read_bytes()==members['shared_contract.json']
    assert unchanged
    executed={**members,**{name:(ROOT/name).read_bytes() for name in (EPOCH+'_request.json',EPOCH+'_stdout.txt',EPOCH+'_stderr.txt',EPOCH+'_exit.json')}}
    if (ROOT/actual).exists():executed[actual]=(ROOT/actual).read_bytes()
    executed_name=EPOCH+'_executed.zip';executed_pin=archive(executed_name,executed)
    capsule={'kind':'SOURCE_SELECTED_NATIVE_GROUND_ORDER_NATURAL_EXIT_CAPSULE','exit':exit_record,'source_input_unchanged':unchanged,
             'preexecution':{'path':pre,'sha256':prepin},'executed':{'path':executed_name,'sha256':executed_pin},
             'pins':{name:digest(raw) for name,raw in executed.items()},'project_python':sys.executable,
             'full_ground_numeric_backend':False,'successful_native_execution':child.returncode==0}
    save(EPOCH+'_capsule.json',capsule)
    if child.returncode!=0:raise SystemExit(child.returncode)
    result=json.loads((ROOT/actual).read_bytes())
    interface={'kind':'GROUND_MAGNETIC_ORDER_CONSUMER_INTERFACE_NOT_IR_ORDER','proof':{'path':PREFIX+'proof_20261012.json','sha256':digest(members[PREFIX+'proof_20261012.json'])},
               'consumer':{'path':PREFIX+'consumer_20261012.py','sha256':digest(members[PREFIX+'consumer_20261012.py'])},
               'actual_ground_gate':result['actual_derived_ground_gate'],'construction_boundaries':result['two_distinct_attempt_boundaries'],
               'P0':'wholeP0','scope':'finite N4 local paired magnetic covariance lower; finite block constant first moment, not N-uniform low-q order',
               'needed_ground_kernel':'N-uniform Q0 covariance >= c kappa and gradient budget <= K kappa^2 with controlled carrier/charge/fullPhi correlations',
               'deck_broken_raw_lift_paid':False,'low_energy_IR_lower_paid':False,'isolated_atom_paid':False,
               'capsule':{'path':EPOCH+'_capsule.json','sha256':digest((ROOT/(EPOCH+'_capsule.json')).read_bytes())}}
    save(PREFIX+'interface_20261012.json',interface)
    receipt={'natural_exit':exit_record,'sources_inputs_unchanged':unchanged,'interface':{'path':PREFIX+'interface_20261012.json','sha256':digest((ROOT/(PREFIX+'interface_20261012.json')).read_bytes())},'capsule':interface['capsule']}
    save(EPOCH+'_receipt.json',receipt)
    delivery_names=[actual,PREFIX+'interface_20261012.json',EPOCH+'_receipt.json',EPOCH+'_capsule.json',pre,executed_name]
    delivery_zip=PREFIX+'delivery_20261012.zip'
    pin=archive(delivery_zip,{name:(ROOT/name).read_bytes() for name in delivery_names})
    save(PREFIX+'delivery_20261012.json',{'delivery':{'path':delivery_zip,'sha256':pin},'receipt':{'path':EPOCH+'_receipt.json','sha256':digest((ROOT/(EPOCH+'_receipt.json')).read_bytes())},'complete_assignment':True,'campaign_complete':False,'claims':interface})
    print(json.dumps({'natural_exit':exit_record,'delivery_sha256':pin,'source_input_unchanged':unchanged}))

if __name__=='__main__':main()

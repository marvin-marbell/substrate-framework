"""Canonical price-only binding of the completed whole129 current enclosure.

Keep the 4ba247 execution epoch immutable. New names distinguish connected
operator and conservative nuclear-row prices; no numerical producer rerun.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from typing import Any

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_current_retarded_higher_binding'
RECEIPT='joined_phi_u1_current_retarded_higher_trace_budget_endpoint_final_receipt_20261010.json'
RECEIPT_SHA='4ba2473482d7668300973ce20ef45645b72c2f4b048a5d3fc231b8592fd121ef'
CAPSULE='joined_phi_u1_current_retarded_higher_trace_budget_endpoint_final_natural_exit_capsule_20261010.json'
CAPSULE_SHA='c1ddc27cedffea862567d93dffc1e67bae45cc8ad60b164e2113419215cc64ad'
PRODUCER='joined_phi_u1_current_retarded_higher_trace_budget.py'
PRODUCER_SHA='839b929c9a8c0452a0f936f6c2d75464da0ea29f0b79b27ffacded2f3f9632fa'


def sha(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


def bind(output:Path)->dict[str,Any]:
    start=time.monotonic();own=Path(__file__).read_bytes()
    captured={name:(ROOT/name).read_bytes() for name in (RECEIPT,CAPSULE,PRODUCER)}
    for name,pin in ((RECEIPT,RECEIPT_SHA),(CAPSULE,CAPSULE_SHA),(PRODUCER,PRODUCER_SHA)):
        if sha(captured[name])!=pin:raise ArithmeticError(f'Pinned producer binding changed:{name}')
    receipt=json.loads(captured[RECEIPT]);capsule=json.loads(captured[CAPSULE])
    if capsule['natural_numeric_exit']!=0 or not capsule['source_stable'] or not capsule['receipt_source_matches'] or not capsule['runner_stable']:
        raise ArithmeticError('Completed producer capsule is not a stable natural0')
    if receipt['source_sha256']!=PRODUCER_SHA or capsule['receipt_sha256']!=RECEIPT_SHA:
        raise ArithmeticError('Producer source/receipt mismatch')
    archive=(ROOT/RECEIPT).with_suffix('.zip');archive_raw=archive.read_bytes()
    if sha(archive_raw)!=receipt['input_archive_sha256']:raise ArithmeticError('Producer input archive changed')
    checked=[]
    for name,expected in receipt['input_sha256'].items():
        raw=(ROOT/name).read_bytes()
        if sha(raw)!=expected:raise ArithmeticError(f'Current live producer input changed:{name}')
        captured[name]=raw;checked.append(name)
    if F(receipt['clock_exact'])!=F(1,65536) or receipt['clock_adopted'] or receipt['measured_evolution'] or receipt['physical_transfer_admitted']:
        raise ArithmeticError('Wrong declared clock/admission scope')
    if set(receipt['states'])!={'background','raw_source'}:raise ArithmeticError('Original two-state scope changed')
    for label,mass in (('background',F(1)),('raw_source',F(1,2))):
        state=receipt['states'][label]
        if F(state['raw_mass_exact'])!=mass or not state['resolved_BODY'] or F(state['BODY_whole129_HS_lower_exact'])<=0:
            raise ArithmeticError('Original mass/positive whole129 lower missing')
        if state['initial_R3_is_signed_measured_coefficient']:raise ArithmeticError('Initial enclosure relabeled measured coefficient')
    q=F(receipt['exact_nuclear_Dcar_upper_exact']);shadow=4*q;old=F(85284055065947197,35184372088832)
    if not q>0 or shadow>old:raise ArithmeticError('Connected/nuclear shadow rescue fails')
    source_path=Path(__file__);captured[source_path.name]=own
    for name,raw in captured.items():
        if (ROOT/name).read_bytes()!=raw:raise ArithmeticError(f'Input changed while binding:{name}')
    outzip=output.with_suffix('.zip')
    with zipfile.ZipFile(outzip,'w',zipfile.ZIP_DEFLATED) as bundle:
        for name,raw in captured.items():bundle.writestr(name,raw)
    result:dict[str,Any]={
      'status':'CURRENT_BOUND_WHOLE129_BODY_CURRENT_HS_LOWER_CANONICAL_PRICE_API',
      'governing_plan':receipt['governing_plan'],'assignment_revision':receipt['assignment_revision'],'template_revision':receipt['template_revision'],
      'parent_governor_observed':'975dfbde0ca9b3e1c9b91ef7491002e007843c737800c588d575ee41a55312ae',
      'clock_exact':receipt['clock_exact'],'clock_adopted':False,'coordinates':129,
      'kernel_convention':'+i raw<[J_i(t),J_j]>; R1=-F and R3=<[adH^3J,J]>; no M3 substitution',
      'Dcar_connected_operator_upper_exact':str(q),
      'Dcar_conservative_nuclear_shadow_upper_exact':str(shadow),
      'old_paid_unroot_Dcar_exact':str(old),'old_paid_minus_nuclear_shadow_exact':str(old-shadow),
      'J_native_carrier_nuclear_directsum_operator_upper_exact':receipt['exact_nuclear_J_directsum_operator_upper_exact'],
      'Dcar_endpoint_scope':receipt['Dcar_endpoint_scope'],
      'states':receipt['states'],
      'third_positive_hierarchy_exact':receipt['third_positive_hierarchy_exact'],
      'fifth_positive_hierarchy_exact':receipt['fifth_positive_hierarchy_exact'],
      'proof':receipt['proof'],
      'nuclear_shadow_proof':'Each actual fixed-BODY Clifford hop ||B||trace=4max(a,b)<=4(a+b)=4||B||op. Positive incoming/outgoing sums and role maxima cover ALL64x64 generated CAR endpoints, including long/diagonal blocks. Hence global nuclear-row multiplier<=4q. This rescues the old PAID2423.92 value with matching-law global proof, not its insufficient old native-local rationale.',
      'legacy_field_translation':'Only immutable producer839b has field exact_nuclear_Dcar_upper_exact; its proof defines connected operator q. This NEW canonical receipt emits q only as Dcar_connected_operator_upper_exact and4q separately as Dcar_conservative_nuclear_shadow_upper_exact. No compatibility alias is emitted.',
      'declared_source_boundary':'Same declared compact-Phi T for parent integration. BODY kernel only; no actual physical E0/full-Phi kernel unless parent graph endpoint and left/right transfer are separately paid. Physical owninitial R0 upper stays in each original-state entry.',
      'producer_execution':{'source':PRODUCER,'source_sha256':PRODUCER_SHA,'receipt':RECEIPT,'receipt_sha256':RECEIPT_SHA,'capsule':CAPSULE,'capsule_sha256':CAPSULE_SHA,'natural_numeric_exit':0,'elapsed_seconds':capsule['elapsed_seconds'],'input_archive_sha256':receipt['input_archive_sha256']},
      'retained_failures':receipt['Jacobi_native_attempt'],
      'exact_Clifford_norm_controls':receipt['exact_Clifford_norm_controls'],
      'signed_R3_tensor_debt_retained':True,'physical_transfer_admitted':False,'measured_evolution':False,
      'source_sha256':sha(own),'input_sha256':{name:sha(raw) for name,raw in captured.items()},
      'input_archive_sha256':sha(outzip.read_bytes()),'live_producer_inputs_verified':checked,
      'elapsed_seconds':time.monotonic()-start,'runtime_executable':sys.executable}
    output.write_text(json.dumps(result,indent=2)+'\n');return result


def natural(output:Path)->dict[str,Any]:
    source=Path(__file__);before=source.read_bytes();start=time.monotonic()
    stdout=output.with_suffix('.stdout.txt');stderr=output.with_suffix('.stderr.txt')
    command=[sys.executable,str(source),'--output',str(output)]
    with stdout.open('wb') as out,stderr.open('wb') as err:
        process=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,check=False)
    after=source.read_bytes();result_raw=output.read_bytes() if output.exists() else None
    capsule:dict[str,Any]={'command':command,'natural_numeric_exit':process.returncode,'elapsed_seconds':time.monotonic()-start,'source_sha256_before':sha(before),'source_sha256_after':sha(after),'source_stable':before==after,'receipt_sha256':sha(result_raw) if result_raw else None,'stdout_sha256':sha(stdout.read_bytes()),'stderr_sha256':sha(stderr.read_bytes()),'runtime_executable':sys.executable}
    if result_raw:
        record=json.loads(result_raw);capsule.update(receipt_source_matches=record['source_sha256']==sha(before),input_sha256=record['input_sha256'],input_archive_sha256=record['input_archive_sha256'],status=record['status'])
    output.with_name(output.stem.replace('_receipt','_natural_exit_capsule')+output.suffix).write_text(json.dumps(capsule,indent=2)+'\n')
    return capsule


if __name__=='__main__':
    sys.set_int_max_str_digits(0)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/(PREFIX+'_receipt_20261010.json'));parser.add_argument('--natural-exit',action='store_true');args=parser.parse_args()
    result=natural(args.output) if args.natural_exit else bind(args.output)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'natural_numeric_exit':result.get('natural_numeric_exit')}),flush=True)

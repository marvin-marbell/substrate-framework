"""Run actual graph consumer to natural exit; bind exact bytes and prices."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_full_current_graph_transfer'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
sys.set_int_max_str_digits(0)

def run(parent:Path)->dict[str,Any]:
    own=Path(__file__).read_bytes(); target=ROOT/(PREFIX+'.py'); before=target.read_bytes(); normbytes=parent.read_bytes()
    output=ROOT/(PREFIX+'_domain_corrected_receipt_20261010.json')
    command=[PYTHON,str(target),'--parent',str(parent),'--output',str(output)]
    started=time.monotonic(); completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=False)
    result:dict[str,Any]=dict(command=command,natural_exit_code=completed.returncode,stdout=completed.stdout,stderr=completed.stderr,
        elapsed_seconds=time.monotonic()-started,source_sha256=hashlib.sha256(before).hexdigest(),parent_receipt_sha256=hashlib.sha256(normbytes).hexdigest(),checker_sha256=hashlib.sha256(own).hexdigest())
    if completed.returncode==0:
        receipt=json.loads(output.read_bytes())
        if receipt['status']!='ACTUAL_PARENT_BOUND_CONDITIONAL_FULL129_GRAPH_TRANSFER':raise ArithmeticError('Pending theorem is NOT completion')
        if set(receipt['states'])!={'background','raw_source'}:raise ArithmeticError('Missing physical endpoint sector')
        for state,entry in receipt['states'].items():
            prices=entry['prices']
            if F(prices['physical_full129_retarded_transfer_error']['upper'])>=F(prices['existing_body_signal_margin']['lower']):raise ArithmeticError('Physical full129 transfer not useful: '+state)
            if F(prices['wrong_Lambda_chargedscale_transfer']['lower'])<=F(prices['existing_body_signal_margin']['upper']):raise ArithmeticError('Wrong charged scale consumer control blind')
            if F(prices['endpoint_graph_norm']['upper'])<=0 or F(prices['propagated_endpoint_current']['upper'])<=0:raise ArithmeticError('Endpoint graph price omitted')
        constants=receipt['global_constants'];bare=F(constants['bare_quartic']['lower']);graph=F(constants['graph_quartic']['lower'])
        if bare!=F(2)**8192/64 or graph!=bare/2:raise ArithmeticError('Strong bare/graph coercivity ABI violated')
        C0=F(constants['Young_C0']['lower'])
        if C0<=0 or F(constants['additive_C']['lower'])!=F(2)**8192*C0:raise ArithmeticError('Useful O(Lambda) Young price missing')
        if not receipt['stable_input_source_bytes']:raise ArithmeticError('Unstable runtime')
        result.update(receipt_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),verified_states={s:{k:v['prices'][k]['upper_decimal'] for k in ('physical_full129_retarded_transfer_error','existing_body_signal_margin','endpoint_graph_norm','propagated_endpoint_current')} for s,v in receipt['states'].items()},actual_epsilon=receipt['actual_epsilon'],conditional_not_full_SOURCE=True)
    result['stable_exactbytes']=before==target.read_bytes() and normbytes==parent.read_bytes() and own==Path(__file__).read_bytes()
    capsule=ROOT/(PREFIX+'_domain_corrected_natural_exit_capsule_20261010.json');capsule.write_text(json.dumps(result,indent=2)+'\n')
    if completed.returncode!=0:raise ArithmeticError('Actual graph consumer failed; natural exit capsule retained')
    if not result['stable_exactbytes']:raise ArithmeticError('Changed runtime bytes')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True);args=parser.parse_args()
    result=run(args.parent)
    print(json.dumps(dict(natural_exit_code=result['natural_exit_code'],states=result['verified_states'],stable=result['stable_exactbytes'],receipt_sha256=result['receipt_sha256']),indent=2))

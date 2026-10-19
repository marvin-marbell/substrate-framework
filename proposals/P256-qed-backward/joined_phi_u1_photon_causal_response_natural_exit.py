"""Capture a naturally completed photon consumer; never overwrite a receipt."""
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
PREFIX='joined_phi_u1_photon_causal_response'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
sys.set_int_max_str_digits(0)


def run(tag: str) -> dict[str,Any]:
    if not tag.replace('_','').isalnum():raise ValueError('Use alphanumeric receipt tag')
    output=ROOT/(PREFIX+'_'+tag+'_receipt.json');capsule=ROOT/(PREFIX+'_'+tag+'_natural_exit_capsule.json')
    if any(p.exists() for p in (output,output.with_suffix('.npz'),output.with_suffix('.zip'),capsule)):
        raise FileExistsError('Immutable capture exists; choose a distinct tag')
    source=ROOT/(PREFIX+'.py');before=source.read_bytes();own=Path(__file__).read_bytes()
    command=[PYTHON,str(source),'--output',str(output)]
    started=time.monotonic();completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=False)
    result:dict[str,Any]=dict(command=command,natural_exit_code=completed.returncode,natural_completion=True,stdout=completed.stdout,stderr=completed.stderr,elapsed_seconds=time.monotonic()-started,source_sha256=hashlib.sha256(before).hexdigest(),runner_sha256=hashlib.sha256(own).hexdigest(),source_stable=before==source.read_bytes() and own==Path(__file__).read_bytes())
    if completed.returncode==0:
        receipt=json.loads(output.read_bytes())
        for label,row in receipt['all129_states'].items():
            if F(row['signed_contact_post_body_and_Phi_HS_margin'])<=0:raise ArithmeticError('Unresolved photon response '+label)
            if F(row['Pi_post_bound_HS_margin'])<=0:raise ArithmeticError('Unresolved Pi response '+label)
            if row['material_remainder_improvement_observed']<=10:raise ArithmeticError('No material sharpening of mixed route')
            if F(row['minimum_directional_D_signed_margin'])<=0:raise ArithmeticError('Signed directional correction unresolved '+label)
        result.update(receipt_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),numerical_capsule=receipt['all129_capsule'],input_bytes=receipt['input_bytes'],runtime=receipt['runtime'],state_outcomes={s:{k:r[k] for k in ('D_remainder_to_signal','cheap_to_signal_observed','material_remainder_improvement_observed')} for s,r in receipt['all129_states'].items()})
    capsule.write_text(json.dumps(result,indent=2)+'\n')
    if completed.returncode!=0:raise ArithmeticError('Scientific consumer failed; original stderr retained in immutable capsule')
    if not result['source_stable']:raise ArithmeticError('Source mutation')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='actual_20261010');args=parser.parse_args()
    result=run(args.tag);print(json.dumps(dict(natural_exit_code=result['natural_exit_code'],receipt_sha256=result['receipt_sha256'],states=result['state_outcomes'],elapsed_seconds=result['elapsed_seconds'])),flush=True)

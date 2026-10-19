"""One selected actual transport consumer, complete natural numeric exit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_electric_hop_transport'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def run(tag: str) -> dict:
    if not tag.replace('_','').isalnum():raise ValueError('Alphanumeric capture tag required')
    source=ROOT/(PREFIX+'.py');output=ROOT/(PREFIX+'_'+tag+'_receipt.json');capsule=ROOT/(PREFIX+'_'+tag+'_natural_exit_capsule.json')
    if any(path.exists() for path in (output,capsule,output.with_suffix('.npz'),output.with_suffix('.zip'))):raise FileExistsError('Immutable capture exists')
    before=source.read_bytes();own=Path(__file__).read_bytes();command=[PYTHON,str(source),'--output',str(output),'--panels','4'];start=time.monotonic()
    completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=False)
    result=dict(command=command,natural_exit_code=completed.returncode,natural_completion=True,stdout=completed.stdout,stderr=completed.stderr,source_sha256=hashlib.sha256(before).hexdigest(),runner_sha256=hashlib.sha256(own).hexdigest(),source_stable=before==source.read_bytes() and own==Path(__file__).read_bytes(),elapsed_seconds=time.monotonic()-start)
    if completed.returncode==0:
        receipt=json.loads(output.read_bytes())
        if len(receipt['receiver_rows'])!=12 or receipt['raw_mass_exact']!='1/2' or receipt['clock_exact']!='1/65536':raise ArithmeticError('Wrong literal acceptance inputs')
        controls=receipt['controls']['electric_Weyl_number_transition_controls']
        if controls['full']['commutator_modulus']>1e-8 or controls['omittedHC']['commutator_modulus']<.01 or controls['halfHE']['commutator_modulus']<.005:raise ArithmeticError('Cross controls not reached')
        result.update(status=receipt['status'],receipt_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),input_bytes=receipt['input_bytes'],numerical_capsule=receipt['numerical_capsule'],all12_controlled=all(row['physical_controlled'] for row in receipt['receiver_rows']),minimum_physical_margin=min(row['physical_margin_conditional'] for row in receipt['receiver_rows']),local_prices=receipt['local_prices'])
    capsule.write_text(json.dumps(result,indent=2)+'\n')
    if completed.returncode!=0:raise ArithmeticError('Natural consumer failure retained in '+capsule.name)
    if not result['source_stable']:raise ArithmeticError('Source mutation')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='actual_20261010');args=parser.parse_args();result=run(args.tag)
    print(json.dumps(result),flush=True)

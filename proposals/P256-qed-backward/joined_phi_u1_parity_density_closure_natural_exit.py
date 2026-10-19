"""One selected same-source parity density consumer and immutable natural exit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_parity_density_closure'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def run(tag: str) -> dict:
    if not tag.replace('_','').isalnum():raise ValueError('Immutable tag must be alphanumeric')
    source=ROOT/(PREFIX+'.py');output=ROOT/(PREFIX+'_'+tag+'_receipt.json');capsule=ROOT/(PREFIX+'_'+tag+'_natural_exit_capsule.json')
    if any(p.exists() for p in (output,capsule,output.with_suffix('.npz'),output.with_suffix('.zip'))):raise FileExistsError('Immutable capture exists')
    own=Path(__file__).read_bytes();before=source.read_bytes();start=time.monotonic()
    command=[PYTHON,str(source),'--output',str(output)]
    completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=False)
    result=dict(command=command,natural_completion=True,natural_exit_code=completed.returncode,stdout=completed.stdout,stderr=completed.stderr,source_sha256=hashlib.sha256(before).hexdigest(),runner_sha256=hashlib.sha256(own).hexdigest(),source_stable=source.read_bytes()==before and Path(__file__).read_bytes()==own,elapsed_seconds=time.monotonic()-start)
    if completed.returncode==0:
        receipt=json.loads(output.read_bytes());rows=receipt['receiver_rows'];parity=receipt['controls']['parity'];coul=receipt['controls']['Coulomb']
        if len(rows)!=12 or receipt['clock_exact']!='1/65536' or receipt['raw_mass_exact']!='1/2' or receipt['clock_adopted']:raise ArithmeticError('Source/clock/receiver scope')
        if parity['wrong_spin_K_only_defect']<.1 or parity['wrong_single_channel_state_defect']<.01:raise ArithmeticError('Symmetry controls absent')
        if parity['actual12_nohop_onsite_commutator_exact_defect']!=0 or any(r['nohop_density_change_exact']!='0' for r in rows):raise ArithmeticError('Nohop density control')
        if parity['independent_initial_second_CAR_oracle']['wrong_Wick_plus_defect']<.01:raise ArithmeticError('Wrong initial CAR coefficient not rejected')
        phase=coul['electric_Weyl_number_transition_controls']
        if phase['full']['commutator_modulus']>1e-8 or phase['omittedHC']['commutator_modulus']<.01 or phase['halfHE']['commutator_modulus']<.005:raise ArithmeticError('Coulomb/HE controls absent')
        with zipfile.ZipFile(ROOT/receipt['input_bytes']['path']) as z:
            if z.read(source.name)!=before:raise ArithmeticError('Consumer bytes not bound')
        result.update(status=receipt['status'],receipt_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),input_bytes=receipt['input_bytes'],numerical_capsule=receipt['numerical_capsule'],all12_controlled_conditional=all(r['physical_controlled_conditional'] for r in rows),minimum_physical_margin_conditional=min(r['physical_margin_conditional'] for r in rows),maximum_fourth_error_to_signal_conditional=max(r['body_error_to_signal_conditional'] for r in rows),selected_assertions_reached=True)
    capsule.write_text(json.dumps(result,indent=2)+'\n')
    if completed.returncode!=0:raise ArithmeticError('Natural failure retained: '+capsule.name)
    if not result['source_stable']:raise ArithmeticError('Source mutation')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='selected_20261010');args=parser.parse_args();result=run(args.tag)
    print(json.dumps({k:v for k,v in result.items() if k!='input_bytes'}),flush=True)

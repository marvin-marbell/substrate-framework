from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from neutral_return_lower_control_20261011 import bareiss, digest

ROOT = Path(__file__).resolve().parent
PARENT_SOURCE = 'canonical_odd_vector_control_20261011.py'
PARENT_SOURCE_PIN = '1f181bacb0f0c18e444817dc09cc22694edf604e19a60f100cc24f6ff23203b1'
PARENT_RECEIPT = 'canonical_odd_vector_actual1_20261011_receipt.json'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag',required=True)
    args = p.parse_args()
    assert args.tag.startswith('canonical_odd_vector_')
    assert digest(ROOT/PARENT_SOURCE)==PARENT_SOURCE_PIN
    original = json.loads((ROOT/PARENT_RECEIPT).read_text())
    assert original['source_sha256']==PARENT_SOURCE_PIN
    assert original['inputs_unchanged'] and original['source_unchanged']
    assert original['joint_and_adverse_controls']['generic_full_wedge2_scalar_plus_vector6_joint_pair']
    assert original['joint_and_adverse_controls']['generic_full_number1_number3_joint_pair']
    bindings = {PARENT_SOURCE:PARENT_SOURCE_PIN, PARENT_RECEIPT:digest(ROOT/PARENT_RECEIPT)}
    for f,h in original['input_sha256'].items():
        assert digest(ROOT/f)==h
        bindings[f]=h
    results=[]
    for role in original['role_assignments']:
        blocks=[]
        for block in role['parity_blocks']:
            for prefix in ('matrix','witness'):
                f=block[prefix+'_path']; h=block[prefix+'_sha256']
                assert digest(ROOT/f)==h
                bindings[f]=h
            cert=json.loads((ROOT/block['matrix_path']).read_text())
            with np.load(ROOT/block['witness_path']) as witness:
                # State word is (five active occupancies, five auxiliary,
                # full6 index, number1 index, number3 index).
                positions=[i for i,j in enumerate(witness['negative']) if witness['states'][j,10]==5]
            lo=[[cert['lower_integer'][i][j] for j in positions] for i in positions]
            hi=[[cert['upper_integer'][i][j] for j in positions] for i in positions]
            il,ih=bareiss(lo),bareiss(hi)
            blocks.append({'parity':block['parity'],'scalar_negative_free_positions':positions,'lower_integer':lo,'upper_integer':hi,'bits':cert['bits'],'lower_inertia':il,'upper_inertia':ih})
        nl=sum(b['lower_inertia']['negative'] for b in blocks)
        nu=sum(b['upper_inertia']['negative'] for b in blocks)
        certified=nl==nu==0
        results.append({'roles_trivial_odd_odd':role['roles_vector_number1_number3'],'spectator_counts':role['spectator_counts'],'lower_inertia_negative':nl,'upper_inertia_negative':nu,'scalar_lambda1_above_minus5123_certified':certified,'canonical_transferred_numbers':[[0,3,3],[4,1,1]] if certified else [],'blocks':blocks,'conclusion':'Scalar restriction positive; exact transfer proves both canonical triples at the same roles' if certified else 'Scalar positive-output lower form does not prove -5123; the canonical transferred sectors remain unpaid'})
    out=ROOT/f'{args.tag}_receipt.json'
    own=digest(Path(__file__))
    result={'kind':'EXACT_SCALAR_RESTRICTION_OF_EXISTING_MIXED_OUTWARD_SCHUR_NO_NEW_SOLVE','source_sha256':own,'input_sha256':bindings,'floor_exact':'-5123','role_results':results,'all_scalar_roles_certified':all(r['scalar_lambda1_above_minus5123_certified'] for r in results),'restriction_proof':'Scalar index5 is a reducing Hfree/rho block, so its supported negative inputs restrict the full Schur S. Principal restrictions of the already proven rational low<=S<=high retain their Loewner bounds (global radius/error may overbound but remains valid). Scalar D-positive supported and omitted complement still positive by the original full-input/free-spectrum proof. Bareiss inertias of those exact integer principal submatrices certify the scalar branch, independently of the full6 lower-form failure. No numerical solve or gap assumed.','canonical_transfer_proof':'At the distinguished trivial role, n2 scalar,n0,n4 all have h=rho=0. At each odd role the exact signed number3 isometry gives the same joint D4/R4 as number1. Therefore scalar+1+3 restricted H0 is unitarily equivalent to 0+3+3 and 4+1+1 at identical role assignments/counts. Each triple has number6, keeping total N256.','scope':'Only symmetric vector5 spectators with one distinguished trivial slot and two distinct odd slots. No fullcanonical, physicalPhi, phase or pole claim.','inputs_unchanged':all(digest(ROOT/f)==h for f,h in bindings.items()),'source_unchanged':digest(Path(__file__))==own}
    with out.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps({'receipt':out.name,'sha256':digest(out),'all_scalar_roles_certified':result['all_scalar_roles_certified'],'roles':[{'roles':r['roles_trivial_odd_odd'],'negative_lower':r['lower_inertia_negative'],'negative_upper':r['upper_inertia_negative']} for r in results]}),flush=True)


if __name__=='__main__':
    main()

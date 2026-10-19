"""Consume immutable operator/magnetic epochs without rebinding original reviews."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

ROOT=Path(__file__).resolve().parent


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(n: str) -> Any:
    return json.loads((ROOT/n).read_text())


def verify_epoch(prefix: str, metadata: str) -> dict[str, Any]:
    c=load(metadata)
    assert c.get('returncode',c.get('exit_code'))==0 and c['natural_exit']
    assert c.get('sources_inputs_unchanged',c.get('all_input_bytes_unchanged'))
    pre=ROOT/(prefix+'_preexecution.zip');exe=ROOT/(prefix+'_executed.zip')
    if 'artifacts_sha256' in c:
        assert all(digest(ROOT/n)==h for n,h in c['artifacts_sha256'].items())
    else:
        assert digest(pre)==c['preexecution_sha256'] and digest(exe)==c['executed_sha256']
        assert digest(ROOT/(prefix+'_receipt.json'))==c['receipt_sha256']
    with zipfile.ZipFile(pre) as a, zipfile.ZipFile(exe) as b:
        for n in a.namelist():
            target=prefix+'_request.json' if n=='request.json' else n
            assert a.read(n)==b.read(target)
        assert b.read(prefix+'_receipt.json')==(ROOT/(prefix+'_receipt.json')).read_bytes()
        return {'prefix':prefix,'preexecution_sha256':digest(pre),'executed_sha256':digest(exe),'receipt_sha256':digest(ROOT/(prefix+'_receipt.json')),'members':len(a.namelist()),'natural_exit':True}


def main() -> None:
    sys.set_int_max_str_digits(0)
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads(a.request.read_text());assert all(digest(ROOT/n)==h for n,h in r['pins'].items())
    epochs=[verify_epoch(x,r['metadata'][x]) for x in r['epochs']]
    original=load('magnetic_mechanism_original_review_proof_20261011.json')
    corrected=load('magnetic_mechanism_proof_20261011.json')
    changed=[(k,j) for k in original for j in original[k] if isinstance(original[k],dict) and original[k][j]!=corrected[k][j]]
    assert changed==[('law','Gauss'),('magnetic_symmetry_exact','regular_raw_algebra')]
    assert 'NOT the simultaneous Gauss' in corrected['law']['Gauss'] and '128/(2pi)' in corrected['magnetic_symmetry_exact']['regular_raw_algebra']
    old=load('magnetic_mechanism_actual2_20261011_receipt.json');new=load('magnetic_mechanism_parent_corrected3_20261011_receipt.json')
    assert old['algebra']==new['algebra'] and old['N_controls']==new['N_controls']
    ar=load('magnetic_algebra_review_verdict_20261011.json');ir=load('magnetic_ir_review_verdict_20261011.json')
    assert ar['verdict']=='CONDITIONAL' and ir['overall_verdict']=='CONDITIONAL'
    assert ar['original_target']['proof_sha256']==digest(ROOT/'magnetic_mechanism_original_review_proof_20261011.json')
    assert load('magnetic_algebra_review_parenttyped4_payload_transfer_20261011.json')['mathematical_payload_exact_equal']
    native=load('positive_residual_parent_native_actual1_20261011_receipt.json')
    assert native['endpoint_inertia_checks']==2944 and native['every_Q_guard_failed']
    rows=native['whole_domain_rows'];assert len(rows)==54 and all(Fraction(x['Q_absolute'])>1 for x in rows)
    best={str(axis):min(Fraction(x['Q_absolute']) for x in rows if x['axis']==axis and Fraction(x['E'])==0) for axis in (1,2)}
    assert all(x>1 for x in best.values())
    result={'kind':'ACTUAL_PARENT_OPERATOR_MAGNETIC_SCOPE_INTEGRATION_NOT_PHASE','verified_epochs':epochs,'independent_native_LDL_checks':2944,'independent_whole_form_rows':54,'E0_best_every_Q_absolute_exact':{k:str(v) for k,v in best.items()},'corrected_magnetic_fields':changed,'original_reviews_frozen_not_rebound':True,'corrected_numeric_payload_exact_original':True,'physical1_sufficient_guard_failed_not_no_go':True,'raw_closed_charge_trivial_not_no_photon':True,'sharp_deck_raw_field_nonregular_not_global_no_go':True,'IR_implication_conditional_ground_domain_wholeP0_covariance_not_atom':True,'historical_machine_formation_conditional':True,'fullphysical_ground_phase_stiffness_photon_admitted':False,'original_SOURCE_adopted':False,'campaign_complete':False}
    with a.output.open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    assert all(digest(ROOT/n)==h for n,h in r['pins'].items())
    print(json.dumps({'epochs':len(epochs),'members':sum(e['members'] for e in epochs),'fullphysical_phase_or_pole':False}))


if __name__=='__main__':main()

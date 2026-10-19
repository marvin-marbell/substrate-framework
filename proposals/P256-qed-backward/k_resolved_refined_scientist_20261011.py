"""Resolve the actual joint-energy supporting dual between its coarse slope nodes.
Fixed native Galerkin anchor, SAME K and every canonical vector; no new solver.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import time
import k_resolved_scientist_20261011 as joint

ROOT = Path(__file__).resolve().parent
PREFIX = 'k_resolved_'
native, frozen, GAMMA = joint.native, joint.frozen, joint.GAMMA


def run(epoch):
    began = time.monotonic()
    request = json.loads((ROOT/f'{PREFIX}{epoch}_request.json').read_bytes())
    assert all(native.sha(ROOT/name) == pin for name,pin in request['runtime_sha256'].items())
    old = json.loads((ROOT/'positive_residual_actual1_20261011_receipt.json').read_bytes())
    centered = json.loads((ROOT/'k_resolved_centered_actual1_20261011_receipt.json').read_bytes())
    eg = F(json.loads((ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_bytes())['ground_energy_interval_exact'][1])
    hfree = joint.free_matrices()
    axes = []
    for oldaxis, centeraxis in zip(old['axes'], centered['axes']):
        assert oldaxis['axis'] == centeraxis['axis']
        original = oldaxis['scalar_anchor_sweep'][3]
        g0lo,g0hi = [F(x) for x in oldaxis['G_ground_expectation']['exact']]
        rows = []
        for oldrow, centerrow in zip(original['energies'], centeraxis['energies']):
            E, A = F(oldrow['E_exact']), F(original['anchor_exact'])-F(oldrow['E_exact'])
            operators = joint.decode(oldrow['joint_operator_onebody_exact'])
            error = F(oldrow['joint_operator_error_exact'])
            candidates = [joint.pencil(operators, hfree, error, F(u), F(0), eg) for u in (72,80,88,96,104,112,120,128)]
            wlo = F(oldrow['vacuum_w0']['exact'][0])
            mismatch = g0hi/A-wlo
            for c in candidates:
                for weight in ('Q','S'):
                    c[f'joint_and_scalar_{weight}_exact'] = str(joint.scalar_price(F(c['slope_exact']),F(c['intercept_exact']),A,weight,g0lo,mismatch))
            bestQ = min(candidates,key=lambda c:F(c['joint_and_scalar_Q_exact']))
            bestS = min(candidates,key=lambda c:F(c['joint_and_scalar_S_exact']))
            jointQ = min(F(bestQ['joint_and_scalar_Q_exact']),F(centerrow['joint_and_negative_scalar_Q_exact']))
            jointS = min(F(bestS['joint_and_scalar_S_exact']),F(centerrow['joint_and_negative_scalar_S_exact']))
            upperQ = F(centerrow['centered_G_Q_upper_exact'])+jointQ
            upperS = F(centerrow['centered_G_S_upper_exact'])+jointS
            lowerQ, lowerS = F(centerrow['every_Q_lower_exact']),F(centerrow['allcentral_S_lower_exact'])
            rows.append(dict(E_exact=str(E), every_Q_upper_exact=str(upperQ), every_Q_upper_display=float(upperQ), every_Q_lower_exact=str(lowerQ), every_Q_lower_display=float(lowerQ), every_Q_absolute_exact=str(max(upperQ,lowerQ)), every_Q_absolute_display=float(max(upperQ,lowerQ)), allcentral_S_upper_exact=str(upperS), allcentral_S_upper_display=float(upperS), allcentral_S_lower_exact=str(lowerS), allcentral_S_absolute_exact=str(max(upperS,lowerS)), allcentral_S_absolute_display=float(max(upperS,lowerS)), best_Q_slope_exact=bestQ['slope_exact'], best_S_slope_exact=bestS['slope_exact'], joint_and_negative_scalar_Q_exact=str(jointQ), joint_and_negative_scalar_S_exact=str(jointS), centered_G_Q_upper_exact=centerrow['centered_G_Q_upper_exact'], complete_lower_components={k:centerrow[k] for k in ('centered_G_Q_lower_exact','signed_B_lower_Q_exact','scalar_counterterm_lower_exact','mismatch_lower_exact')}, whole_positive_residual_Q_exact=centerrow['whole_positive_residual_Q_exact'], whole_positive_residual_Q_display=centerrow['whole_positive_residual_Q_display'], candidates=candidates))
        axes.append(dict(axis=oldaxis['axis'], energies=rows))
    result = dict(kind='NEW_REFINED_K_ENERGY_DUAL_SIGNED_WHOLE_FORM_NOT_PHYSICAL_ADMISSION', epoch=epoch, governing_plan=request['governing_plan'], source_sha256=native.sha(Path(__file__)), input_sha256=request['input_sha256'], formation=old['formation'], domains=old['domains'], axes=axes, controls=joint.controls(), all512_totalN256_anyQ=True, full_PgQg_allcentral_weight='K+gamma', physical_hopping_exact='1', historical_premises_conditional=True, no_old_main_eigensolver_root_checker=True, phase_or_root_admitted=False, campaign_complete=False, elapsed_seconds=time.monotonic()-began)
    result['physical1_Q_upper_pass'] = all(F(a['energies'][1]['every_Q_upper_exact']) < 1 for a in axes)
    result['physical1_Q_absolute_pass'] = all(F(a['energies'][1]['every_Q_absolute_exact']) < 1 for a in axes)
    result['sources_inputs_unchanged'] = all(native.sha(ROOT/n) == p for n,p in request['runtime_sha256'].items())
    assert result['sources_inputs_unchanged']
    native.dump(ROOT/f'{PREFIX}{epoch}_receipt.json',result)
    print(json.dumps(dict(receipt=f'{PREFIX}{epoch}_receipt.json', axes=[dict(axis=a['axis'], Qupper=a['energies'][1]['every_Q_upper_display'], Qlower=a['energies'][1]['every_Q_lower_display'], Qabsolute=a['energies'][1]['every_Q_absolute_display'], Supper=a['energies'][1]['allcentral_S_upper_display'], slope=a['energies'][1]['best_Q_slope_exact']) for a in axes], Qupper_pass=result['physical1_Q_upper_pass'], Qabsolute_pass=result['physical1_Q_absolute_pass'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch',required=True)
    run(parser.parse_args().epoch)

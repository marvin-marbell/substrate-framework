"""NEW joint K resolution of centered G as well as signed B/T and whole residual.
Second material construction: price the entire K-relative G operator before
using its NONCOMMUTING anticommutator with F. No source-column theorem.
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
GSLOPES = (F(0), F(1, 16), F(1, 8), F(1, 4), F(1, 2), F(1), F(2), F(4), F(8), F(16), F(32))


def centered_role(roles, sign, center):
    out = []
    for role, (scalar, h) in enumerate(roles):
        out.append((sign*scalar+(center if role == 0 else F(0)), [[joint.car.cs(z, sign) for z in row] for row in h]))
    return out


def relative_pencil(certificate, weight):
    u, v = F(certificate['slope_exact']), F(certificate['intercept_exact'])
    # k>=gamma on Q; k>=0 for positive S=K+gamma on whole central.
    return u+max(v, F(0))/GAMMA if weight == 'Q' else max(u, v/GAMMA)


def run(epoch):
    began = time.monotonic()
    request = json.loads((ROOT/f'{PREFIX}{epoch}_request.json').read_bytes())
    assert all(native.sha(ROOT/name) == pin for name, pin in request['runtime_sha256'].items())
    old = json.loads((ROOT/'positive_residual_actual1_20261011_receipt.json').read_bytes())
    primary = json.loads((ROOT/'k_resolved_actual1_20261011_receipt.json').read_bytes())
    energy = json.loads((ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_bytes())
    eg = F(energy['ground_energy_interval_exact'][1])
    hfree = joint.free_matrices()
    axes = []
    for oldaxis, firstaxis in zip(old['axes'], primary['axes']):
        assert oldaxis['axis'] == firstaxis['axis']
        g0lo, g0hi = [F(x) for x in oldaxis['G_ground_expectation']['exact']]
        gspec = [F(x) for x in oldaxis['G_spectrum']['exact']]
        G = joint.decode(oldaxis['G_onebody_exact'])
        Gerror = F(oldaxis['G_operator_error_exact'])
        plus = centered_role(G, F(1), -g0lo)
        minus = centered_role(G, F(-1), g0hi)
        gcandidates = {}
        for name, roles in (('Gc_upper', plus), ('minus_Gc_upper', minus)):
            gcandidates[name] = [joint.pencil(roles, hfree, Gerror, u, F(0), eg) for u in GSLOPES]
            for c in gcandidates[name]:
                c['relative_Q_exact'] = str(relative_pencil(c, 'Q'))
                c['relative_S_exact'] = str(relative_pencil(c, 'S'))
        upperQ = min([max(gspec[1]-g0lo, F(0))/GAMMA]+[relative_pencil(c, 'Q') for c in gcandidates['Gc_upper']])
        minusQ = min([max(g0hi-gspec[0], F(0))/GAMMA]+[relative_pencil(c, 'Q') for c in gcandidates['minus_Gc_upper']])
        upperS = min([max(gspec[1]-g0lo, F(0))/GAMMA]+[relative_pencil(c, 'S') for c in gcandidates['Gc_upper']])
        minusS = min([max(g0hi-gspec[0], F(0))/GAMMA]+[relative_pencil(c, 'S') for c in gcandidates['minus_Gc_upper']])
        original = oldaxis['scalar_anchor_sweep'][3]
        # A separate actual K-resolved TT price pays the WHOLE positive remainder.
        TT = joint.decode(original['TdagT_onebody_exact'])
        Terror = F(original['TdagT_error_exact'])
        tcandidates = [joint.pencil(TT, hfree, Terror, u, F(0), eg) for u in (F(0), F(64), F(128), F(256), F(512), F(1024), F(2048), F(4096))]
        # Actual signed B retains its joint K correlation on the lower side.
        B = joint.decode(original['signed_B_onebody_exact'])
        Berror = F(original['signed_B_error_exact'])
        bcandidates = [joint.pencil(B, hfree, Berror, u, F(0), eg) for u in (F(0), F(1), F(4), F(16), F(64), F(256), F(1024))]
        rows = []
        for oldrow, firstrow in zip(original['energies'], firstaxis['energies']):
            E, A = F(oldrow['E_exact']), F(original['anchor_exact'])-F(oldrow['E_exact'])
            assert oldrow['E_exact'] == firstrow['E_exact']
            de = F(old['domains']['whole_operator_two_cell_floor_exact'])-E
            wlo, whi = [F(x) for x in oldrow['vacuum_w0']['exact']]
            centered_Q_upper = (upperQ+(upperQ+minusQ)/8)/(A+GAMMA)
            centered_S_upper = (upperS+(upperS+minusS)/8)/A
            centered_Q_lower = min(max(upperQ, minusQ), minusQ+(upperQ+minusQ)/8)/(A+GAMMA)
            centered_S_lower = min(max(upperS, minusS), minusS+(upperS+minusS)/8)/A
            residualQ = min(joint.scalar_price(F(c['slope_exact']), F(c['intercept_exact']), A, 'Q') for c in tcandidates)/de
            residualS = min(joint.scalar_price(F(c['slope_exact']), F(c['intercept_exact']), A, 'S') for c in tcandidates)/de
            bQ = min(joint.scalar_price(F(c['slope_exact']), F(c['intercept_exact']), A, 'Q') for c in bcandidates)
            bS = min(joint.scalar_price(F(c['slope_exact']), F(c['intercept_exact']), A, 'S') for c in bcandidates)
            jointQ = F(firstrow['K_resolved_joint_and_negative_scalar_Q_exact'])
            jointS = F(firstrow['K_resolved_joint_and_negative_scalar_S_exact'])
            Qupper, Supper = centered_Q_upper+jointQ, centered_S_upper+jointS
            scalar_lower = g0hi/(A*(A+GAMMA))
            mismatch_lower = max(whi-g0lo/A, F(0))/GAMMA
            Qlower = min(centered_Q_lower+scalar_lower+mismatch_lower+bQ, whi/GAMMA)
            Slower = min(centered_S_lower+scalar_lower+mismatch_lower+bS, whi/GAMMA)
            rows.append(dict(E_exact=str(E), every_Q_upper_exact=str(Qupper), every_Q_upper_display=float(Qupper), every_Q_lower_exact=str(Qlower), every_Q_lower_display=float(Qlower), every_Q_absolute_exact=str(max(Qupper, Qlower)), every_Q_absolute_display=float(max(Qupper, Qlower)), allcentral_S_upper_exact=str(Supper), allcentral_S_upper_display=float(Supper), allcentral_S_lower_exact=str(Slower), allcentral_S_absolute_exact=str(max(Supper, Slower)), allcentral_S_absolute_display=float(max(Supper, Slower)), centered_G_Q_upper_exact=str(centered_Q_upper), centered_G_Q_lower_exact=str(centered_Q_lower), centered_G_S_upper_exact=str(centered_S_upper), centered_G_S_lower_exact=str(centered_S_lower), joint_and_negative_scalar_Q_exact=str(jointQ), joint_and_negative_scalar_S_exact=str(jointS), whole_positive_residual_Q_exact=str(residualQ), whole_positive_residual_Q_display=float(residualQ), whole_positive_residual_S_exact=str(residualS), signed_B_lower_Q_exact=str(bQ), signed_B_lower_S_exact=str(bS), scalar_counterterm_lower_exact=str(scalar_lower), mismatch_lower_exact=str(mismatch_lower), inherited_first_design=firstrow['best_Q_design'], inherited_first_slope=firstrow['best_Q_slope_exact']))
        axes.append(dict(axis=oldaxis['axis'], Gc_K_relative_interval_exact=[str(-minusQ), str(upperQ)], Gc_S_relative_interval_exact=[str(-minusS), str(upperS)], centered_G_joint_pencils=gcandidates, whole_TdagT_joint_pencils=tcandidates, signed_B_joint_pencils=bcandidates, energies=rows))
    result = dict(kind='NEW_K_RESOLVED_G_B_T_WHOLE_SIGNED_FORM_NOT_PHYSICAL_ADMISSION', epoch=epoch, governing_plan=request['governing_plan'], source_sha256=native.sha(Path(__file__)), input_sha256=request['input_sha256'], formation=old['formation'], domains=old['domains'], axes=axes, controls=joint.controls(), all512_totalN256_anyQ=True, full_PgQg_allcentral_weight='K+gamma', physical_hopping_exact='1', historical_premises_conditional=True, no_old_main_eigensolver_root_checker=True, phase_or_root_admitted=False, campaign_complete=False, elapsed_seconds=time.monotonic()-began)
    result['physical1_Q_upper_pass'] = all(F(a['energies'][1]['every_Q_upper_exact']) < 1 for a in axes)
    result['physical1_Q_absolute_pass'] = all(F(a['energies'][1]['every_Q_absolute_exact']) < 1 for a in axes)
    result['sources_inputs_unchanged'] = all(native.sha(ROOT/n) == p for n,p in request['runtime_sha256'].items())
    assert result['sources_inputs_unchanged']
    native.dump(ROOT/f'{PREFIX}{epoch}_receipt.json', result)
    print(json.dumps(dict(receipt=f'{PREFIX}{epoch}_receipt.json', axes=[dict(axis=a['axis'], Qupper=a['energies'][1]['every_Q_upper_display'], Qlower=a['energies'][1]['every_Q_lower_display'], Qabsolute=a['energies'][1]['every_Q_absolute_display'], Supper=a['energies'][1]['allcentral_S_upper_display'], residual=a['energies'][1]['whole_positive_residual_Q_display']) for a in axes], Qupper_pass=result['physical1_Q_upper_pass'], Qabsolute_pass=result['physical1_Q_absolute_pass'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    run(parser.parse_args().epoch)

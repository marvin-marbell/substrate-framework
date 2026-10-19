"""Compose NEW unrestricted body source bounds with the conditional Phi packet."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import mpmath as mp
import joined_phi_u1_body_word as body
import joined_phi_u1_unrestricted_source_phase_control as phase

ROOT = Path(__file__).resolve().parent
INPUTS = {
    'induced_phi_global_normal_source_reprice_receipt.json': '14f50709506112ee0f9957d59d1d0661e66133cbdf27665c324095ba6359581a',
    'joined_phi_u1_unrestricted_source_phase_control_receipt.json': '7b19e9f5e500d8346ba758a5221a067ffba2eaf42584b01e96a207703389d7e4',
    'joined_phi_u1_body_word_actual_receipt.json': 'f558ec753eb2f855a3ccbbf3ccac4a31cb9da86b4226f9399e2ec405d28f6275',
}


def record(value: F) -> dict[str, str]:
    return {'exact': str(value), 'decimal': str(mp.nstr(mp.mpf(value.numerator)/value.denominator, 35))}


def run() -> dict[str, Any]:
    started = time.monotonic()
    own = Path(__file__).read_bytes()
    raws = {name: (ROOT/name).read_bytes() for name in INPUTS}
    for name, raw in raws.items():
        if hashlib.sha256(raw).hexdigest() != INPUTS[name]:
            raise ArithmeticError(f'Pinned input epoch changed: {name}')
    prices, proof, actual = (json.loads(raws[name]) for name in INPUTS)
    mp.mp.dps = 100
    phase.field.full.iv.dps = 100
    core = F(prices['repriced_no_U1_Phi_source']['reference_residual_upper']['exact_upper'])
    coupling = F(prices['new_U1_extension']['coupling_residual_upper']['exact_upper'])
    endpoint = F(prices['repriced_no_U1_Phi_source']['endpoint_upper']['exact_upper'])
    if min(core, coupling, endpoint) < 0:
        raise ArithmeticError('Physical comparison requires nonnegative upward prices')
    rows = []
    for row in proof['unrestricted_source_clocks']:
        t = F(row['time_exact'])
        embedding = 8*abs(t)*(core+coupling)+8*endpoint
        body_error = F(row['quadratic_cubic_error_upper']['exact'])
        body_imag_lower = F(row['actual_negative_imag_lower']['exact'])
        real_lower = F(row['actual_real_lower']['exact'])-embedding
        imag_lower = body_imag_lower-embedding
        if real_lower <= 0:
            raise ArithmeticError('Conditional joined source modulus lower is vacuous')
        rows.append({'time_exact': str(t), 'conditional_Phi_body_complex_error_upper': record(embedding),
                     'unrestricted_body_Taylor_error_upper': record(body_error),
                     'conditional_joined_negative_imag_lower': record(imag_lower),
                     'conditional_joined_nonzero_phase_proven': imag_lower > 0,
                     'conditional_joined_real_lower': record(real_lower),
                     'conditional_joined_raw_probability_lower': record(real_lower**2)})
    # Reject transplanting the old no-U1 free-band reference into the U1 body.
    # This is a NEW interval calculation of the comparison's initial moment,
    # not a rerun of the archived source or its photon precision producer.
    data, _, native_inputs, _ = body.full_inputs()
    pars = json.loads(native_inputs[body.ARCHIVE])['input_descriptor']['exact_NEW_allq_BO_mean']['parameters']
    g0, gs = [phase.field.full.enclosure(F(pars[k]['lower']), F(pars[k]['upper'])) for k in ('g0', 'gs')]
    ex = phase.field.full.exact
    frozen = phase.moment(data['profiles'][1], [g0, gs, gs, gs], {1: ex(1), 2: ex(1)})
    frozen_lower = phase.lower(frozen)
    actual_upper = F(proof['initial_m1_interval']['upper'])
    hmax = F(proof['profile_bounds'][0]['all_real_Theta_Hmax_upper']['exact'])
    first = rows[0]
    t = F(first['time_exact'])
    old_free_remainder = hmax*hmax*t*t/4
    separation = t*(frozen_lower-actual_upper)-old_free_remainder-F(first['unrestricted_body_Taylor_error_upper']['exact'])-F(first['conditional_Phi_body_complex_error_upper']['exact'])
    if separation <= 0:
        raise ArithmeticError('Selected clock cannot reject old free-band phase substitution')
    descriptor = actual['input_descriptor']
    if descriptor['allTheta'] != 129 or descriptor['copy_loads'] != [1, 63] or descriptor['Ref_before'] != 4 or descriptor['Ref_after'] != 3:
        raise ArithmeticError('Prepared-body epoch is not the allcharged co-local Ref source')
    if actual['authoritative_coefficient_binding']['HE_coefficient'] != 64 or actual['authoritative_coefficient_binding']['HC_coefficient'] != 64:
        raise ArithmeticError('Joined body requires model-owned HE=HC=64')
    stable = {name: raw == (ROOT/name).read_bytes() for name, raw in raws.items()}
    if not all(stable.values()) or own != Path(__file__).read_bytes():
        raise ArithmeticError('Source/input changed during composition')
    return {'input_sha256': INPUTS, 'inputs_unchanged': stable, 'source_sha256': hashlib.sha256(own).hexdigest(),
            'source_utf8': own.decode(), 'runtime': {'executable': sys.executable, 'python': sys.version},
            'conditional_joined_source_clocks': rows,
            'wrong_old_free_band_transfer_control': {'time_exact': str(t), 'old_no_U1_initial_m1_interval': phase.bounds(frozen),
                'old_free_remainder_upper': record(old_free_remainder), 'separation_lower': record(separation),
                'status': 'Derived disjoint phase intervals at2^-22. Old no-U1 free-band reference is NOT the U1 interacting-body amplitude.'},
            'derivation': [
                'NEW Phi packet is the compact corrected Gaussian/cubic packet with localSU4 reconstruction and occupied Wphi atTheta0; tensor the NEW fixed-mean charged-Slater/realThetaGaussian body preparation. Wphi is independentTheta, so HE commutes Wphi exactly.',
                'Parent uniform comparison first removes old charged-sea scalar cancellation and then prices arbitrary-state/allTheta chargedPhi variation. Full allchargedCoulomb conjugation is bounded separately. No evolved-state Slater Wick or old free-band probability is transferred.',
                'On any normalized body vector, the conditional Phi residual upper is core+coupling; endpoint is the existing physical Etilde vs baremeanE comparison. Uniformity permits the actual unrestricted Hbody evolution on both legs. Duhamel/source Cauchy gives8|t|(core+coupling)+8endpoint. Common scalar reference phase cancels in the same-H pair.',
                'Triangle inequality transfers the BODY analytic Taylor phase and real-part lower by subtracting this positive comparison price. Finiteword Monte Carlo amplitudes are NOT substituted for the unrestricted body amplitude.',
                'Old free-band mean initial m1 exceeds body m1; spectral support<=Hmax bounds oldfree Taylor remainder by .25Hmax²t² because its raw spectral measure has totalmass.5. Comparison with the derived body/embedding remainder gives a disjoint initial phase interval at2^-22.'],
            'remaining_hypotheses': [
                'Exact selected root-box/Clifford and full Gaussian/corrector supplier coefficient/action hypotheses, not root existence or independently derived whole source certification.',
                'Parent-repriced core Gaussian/Holder/Hermite/action descriptors remain conditional; derived geometry and real Kato only remove their own premises.',
                'Declared NEW compact Phi quantum law/preparation, not adoption of originalSOURCE or CURRENT427.'],
            'not_admitted': ['Original SOURCE', 'whole global source certification', 'body ground', 'physical dressed pole or mass', 'photon kernel/common cone', 'current/Compton closure', 'finiteword Trotter or MC confidence transfer'],
            'governing_plan': 'issue-222-induced-photon-delta13-continuation-20261008',
            'captured_parent_revision': '944a367c79e90d22207696f47a944bbaf062720e24559f0b4b21c823e04f2402',
            'template_revision': '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
            'elapsed_seconds': time.monotonic()-started}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'receipt_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'rows': [{k: (v['decimal'] if isinstance(v, dict) and 'decimal' in v else v) for k, v in row.items()} for row in result['conditional_joined_source_clocks']],
                      'old_free_band_separation_lower': result['wrong_old_free_band_transfer_control']['separation_lower']['decimal'],
                      'elapsed_seconds': result['elapsed_seconds']}, indent=2))

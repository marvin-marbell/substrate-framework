"""Exact SAME-minimum body composition; no ground-vector, phase or atom claim.

Consume immutable sources and independent controls without replaying producers.
A measurable fixed-site matrix event is not an operator-domain assertion.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import io
import json
from math import isqrt
from pathlib import Path
from typing import Any
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text())


def sqrt_upper(value: F) -> F:
    scale = 1 << 512
    n = isqrt(value.numerator * scale * scale // value.denominator)
    assert F(n, scale) ** 2 <= value < F(n + 1, scale) ** 2
    return F(n + 1, scale)


def verify_epoch(spec: dict[str, Any]) -> dict[str, Any]:
    pre, exe, timing = (spec[key] for key in ('preexecution', 'executed', 'runtime_artifact'))
    for artifact in (pre, exe, timing):
        assert sha(Path(artifact['path']).read_bytes()) == artifact['sha256']
    runtime = json.loads(Path(timing['path']).read_text())
    assert runtime['natural_exit'] is True and runtime['returncode'] == spec['returncode']
    with ZipFile(pre['path']) as before, ZipFile(exe['path']) as after:
        assert len(before.namelist()) == len(set(before.namelist()))
        assert len(after.namelist()) == len(set(after.namelist()))
        assert set(before.namelist()) == set(spec['members'])
        embedded_name = Path(pre['path']).name
        nested = ZipFile(io.BytesIO(after.read(embedded_name))) if embedded_name in after.namelist() else None
        if nested is not None:
            assert after.read(embedded_name) == Path(pre['path']).read_bytes()
        for name, binding in spec['members'].items():
            data = before.read(name)
            assert sha(data) == binding['sha256']
            if binding['executed_location'] == 'direct':
                assert after.read(name) == data
            else:
                assert binding['executed_location'] == 'embedded_preexecution' and nested is not None
                assert nested.read(name) == data
            if binding['live_path'] is not None:
                assert sha(Path(binding['live_path']).read_bytes()) == binding['sha256']
        for name, binding in spec['outputs'].items():
            assert sha(after.read(name)) == binding['sha256']
            assert sha(Path(binding['path']).read_bytes()) == binding['sha256']
        if nested is not None:
            nested.close()
    # Raw timing bytes, including records without an elapsed field, remain the authority.
    return {'epoch': spec['epoch'], 'natural_exit': True, 'returncode': runtime['returncode'],
            'members_compared': len(spec['members']), 'raw_runtime_artifact': timing,
            'all_declared_members_verified': True,
            'embedded_only_members': [name for name, binding in spec['members'].items()
                                      if binding['executed_location'] == 'embedded_preexecution']}


def joined_gate() -> dict[str, Any]:
    profile = load('neutral_body_parent_primitive_profiles_20261012.json')['primitive_profiles'][0]
    m = F(profile['onsite'])
    c0, ci = (F(profile['paths'][0][key]) for key in ('c0', 'ci'))
    b, D = abs(c0) + abs(ci), 2 * abs(m) + 12 * (abs(c0) + abs(ci))
    R, cap, N, M = 2 ** 256, 2 ** 64, 2 ** 257, F(2 ** 14)
    a, c = F(40, 2 ** 2035), F(10, 7 * 2 ** 2035)
    field = (1024 * cap ** 3 + D) * sqrt_upper(48 * R ** 4 * a)
    curvature = 4 * b * cap * sqrt_upper(1728 * R ** 5 * c)
    boundary = (18 + 3 * F(22, 7)) * b * cap / R
    tail = D * M / cap ** 3
    total = field + curvature + boundary + tail
    assert total < F(2) ** -160
    supplier = load('quantum_minimum_sea_actual3_20261012_interface.json')
    expected = {'neutral_Jordan_support': F(2) ** -3304,
                'Phi_SU4_kinetic': F(2) ** -4820,
                'alignment': F(2) ** -4840, 'colour': F(2) ** -4840,
                'shared_HEHC_packet': F(2) ** -8173}
    assert {k: F(v) for k, v in supplier['fullH_prices']['fullH_correction_terms'].items()} == expected
    epsilon = sum(expected.values(), F(0))
    assert F(supplier['epsilon_quantum']) == epsilon < F(2) ** -3000
    epsilon_upper = F(2 ** 16, N) + epsilon
    assert epsilon_upper < F(2) ** -240
    Dch = F(855444290449274477, 35184372088832)
    assert F(load('open_box_body_actual2_20261012_receipt.json')['Dch']) == Dch
    charged = 12 * Dch / 2 ** 8192
    eta = total + epsilon_upper + charged
    assert eta < F(2) ** -159
    deviation = 2 * M / cap ** 2 + 96 * R ** 4 * a
    assert deviation < F(2) ** -112
    # The EXACT sqrt(M) is128; M^(1/4)<12, so six safely prices .5*M^(1/4).
    norm_L1 = 6 * sqrt_upper(deviation) + sqrt_upper(eta / 64)
    assert norm_L1 < F(2) ** -52 and 256 * norm_L1 < F(2) ** -44
    # Body-norm, body-Jordan, physical radial and SAME-tree deviation events.
    failures = [F(2) ** -37, F(2) ** -79, F(2) ** -18, F(2) ** -28]
    assert sum(failures, F(0)) < F(2) ** -17
    self_bound = (256 + 5) * F(2) ** -42 + F(2) ** -42 + 2 * F(2) ** -64
    anti_bound = 2 * (256 + 5) * F(2) ** -42 + F(2) ** -40
    assert self_bound < F(2) ** -32 and anti_bound < F(2) ** -31
    original = load('open_box_body_quantum_join_20261012_receipt.json')
    assert F(original['quantum_upper_excess_exact']) == epsilon_upper
    assert F(original['charged_price_exact_per_Lambda_site']) == charged
    assert original['eta_lt_2power_minus159'] is True
    independent = load('body_probability_review_actual3_20261012_receipt.json')
    controls = independent['native_CAR_and_matrices']
    assert controls['canonical'] == 128 and controls['density_not_projector'] and controls['density_noncommuting']
    assert controls['first_site_traces'] == ['1106/625', '1138/625']
    assert F(controls['Jordan_excess']) == F(338, 441) and controls['wrong_raw_trace_rejected']
    volume = independent['all_partitions_volume_bias']
    assert volume['all_partition_shifts'] == 512 and volume['box_volumes'] == [27, 45, 75, 125]
    assert F(volume['volume_biased_mean_Bnorm2']) != F(volume['wrong_unweighted_mean_Bnorm2'])
    ui = independent['fourth_tail_UI_adverse']['family']
    for row in ui:
        radius, mass = int(row['r']), F(row['mass'])
        assert mass * radius ** 4 == 1 and mass * radius ** 2 == F(row['second_transfer'])
        assert F(row['fourth_tail_above_one']) == 1
    return {'domain': 'Every even N>=2^257; R2^256,L2^64; ALL translated partitions and volume-biased boxes.',
            'Etotal_upper_exact': str(total), 'eta_upper_exact': str(eta),
            'epsilon_quantum_exact': str(epsilon), 'charged_price_per_Lambda_site_exact': str(charged),
            'site_quarter_norm_L1_upper_exact': str(norm_L1),
            'site_component_window_failure_upper_exact': str(256 * norm_L1),
            'joint_fixed_site_failure_upper': str(sum(failures, F(0))),
            'four_self_defects_HS_upper': str(self_bound),
            'twelve_ordered_anticommutators_HS_upper': str(anti_bound),
            'retained_before_optional_drop': 'Q/(Lambda V)+(F/Lambda)<W> and complete charged −Dch sqrt(<s>)/Lambda',
            'fixed_site_event_not_mean_Jordan_or_sharp_domain': True,
            'UI_adverse_is_nonnative_family_not_ground_no_go': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, binding in request['inputs'].items():
        assert sha(Path(binding['path']).read_bytes()) == binding['sha256'], name
    epochs = [verify_epoch(spec) for spec in request['epochs']]
    result = {'kind': 'EXACT_COMMON_MINIMUM_FULL_QUANTUM_BODY_AND_UNCAPPED_LOCAL_MATRIX_PROBABILITY_JOIN_NOT_PHASE',
              'governor': request['governor'], 'epochs': epochs,
              'members_compared': sum(item['members_compared'] for item in epochs),
              'joined_gate': joined_gate(), 'independent_reviews': request['independent_reviews'],
              'original_targets_reviews_or_epochs_rebound': False,
              'actual_ground_vector_or_fullH_numerical_evolution_computed': False,
              'mean_Jordan_UI_or_sharp_projector_domain_proved': False,
              'wholeP0_magnetic_Gram_or_physical1_phase_stiffness_IR_atom': False,
              'original_SOURCE_or_law_adopted': False, 'unconditional_machine_attestation': False,
              'campaign_complete': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'epochs': len(epochs), 'members': result['members_compared'],
                      'eta_lt_2power_minus159': True, 'fixed_site_matrix_event_only': True}))


if __name__ == '__main__':
    main()

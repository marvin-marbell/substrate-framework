"""Immutable full-quantum sea/local-body join; no ground or phase adoption.

Recompute the full-H trial price and conditional actual-family marginal bounds.
The trial, actual minimizing-family implications and POINT countercontrols stay
separate. No original producer, solver or review is executed by this consumer.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from fractions import Fraction as F
import hashlib
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


def verify_epoch(spec: dict[str, Any]) -> dict[str, Any]:
    pre, exe = spec['preexecution'], spec['executed']
    assert sha((ROOT / pre['path']).read_bytes()) == pre['sha256']
    assert sha((ROOT / exe['path']).read_bytes()) == exe['sha256']
    r = spec['runtime_artifact']
    assert sha((ROOT / r['path']).read_bytes()) == r['sha256']
    runtime = load(r['path'])
    natural = runtime.get('natural_exit', runtime.get('natural_exit_observed'))
    rc = runtime.get('returncode', runtime.get('exit_code'))
    assert natural is True and rc == spec['runtime']['returncode']
    elapsed = runtime.get('elapsed_seconds', runtime.get('runtime_seconds'))
    if elapsed is None:
        elapsed = (datetime.fromisoformat(runtime['ended']) - datetime.fromisoformat(runtime['started'])).total_seconds()
    assert elapsed == spec['runtime']['elapsed_seconds']
    with ZipFile(ROOT / pre['path']) as before, ZipFile(ROOT / exe['path']) as after:
        assert len(before.namelist()) == len(set(before.namelist()))
        assert len(after.namelist()) == len(set(after.namelist()))
        declared = set(spec['input_members']) | set(spec['archive_only_inputs'])
        assert declared == set(before.namelist()), spec['epoch']
        for member in before.namelist():
            data = before.read(member)
            assert after.read(member) == data, (spec['epoch'], member)
            if member in spec['input_members']:
                binding = spec['input_members'][member]
                assert sha(data) == binding['sha256']
                assert sha(Path(binding['path']).read_bytes()) == binding['sha256']
            else:
                assert sha(data) == spec['archive_only_inputs'][member]
        if spec['nested_preexecution']:
            assert after.read(pre['path']) == (ROOT / pre['path']).read_bytes()
        for member, binding in spec['outputs'].items():
            assert sha(after.read(member)) == binding['sha256']
            assert sha(Path(binding['path']).read_bytes()) == binding['sha256']
        count = len(before.namelist())
    return {'epoch': spec['epoch'], 'runtime': spec['runtime'],
            'preexecution_members_compared': count,
            'all_declared_members_verified': True,
            'archive_only_historical_members': list(spec['archive_only_inputs'])}


def outward_sqrt(value: F) -> F:
    scale = 1 << 256
    n = isqrt(value.numerator * scale * scale // value.denominator)
    assert F(n, scale)**2 <= value < F(n + 1, scale)**2
    return F(n + 1, scale)


def exact_quantum_trial() -> dict[str, Any]:
    profile = load('neutral_body_parent_primitive_profiles_20261012.json')['primitive_profiles'][0]
    m, c0, ci = (F(profile['onsite']), F(profile['paths'][0]['c0']), F(profile['paths'][0]['ci']))
    assert m - 6 * abs(c0) > 0
    x = F(3, 2) * (2 * ci / 3)**2
    xm2 = x * m * m + 4 * c0 * c0 * F(9, 4) * (2 * ci / 3)**2
    x2 = F(21, 8) * (2 * ci / 3)**4
    gain = x * x / (outward_sqrt(x * (xm2 / 4 + x2)) + m * x / 2)
    point = F(172, 27) - m - 2 * gain
    terms = {'neutral_and_Jordan_support_error': F(2)**(-3344),
             'Phi_and_SU4_complete_covariant_Dirichlet': F(2)**(-4860),
             'endpoint_alignment': F(2)**(-4880),
             'colour_plaquette': F(2)**(-4880),
             'charged_all64_AP1_AP2_Ref_expectation': F(0),
             'shared_HE_HC_and_original_cover_packet': F(2)**(-8173)}
    total = point + sum(terms.values(), F(0))
    claimed = load('connected_quantum_sea_actual2_20261012_interface.json')['energy']
    assert F(claimed['point_sea_plus_ordered_Jordan_upper_per_Lambda_site']) == point
    assert {k: F(v) for k, v in claimed['full_H_corrections_per_Lambda_site'].items()} == terms
    assert F(claimed['total_upper_exact']) == total < -21
    assert sum(terms.values(), F(0)) < F(2)**-3000
    control = load('connected_quantum_sea_actual2_20261012_receipt.json')
    assert control['normal_slice_dimension'] == 45 and control['Cart_dimension_per_site'] == 60
    assert F(control['normalized_first_Dirichlet_constant']) == F(34, 7)
    assert F(control['normalized_second_jet_constant']) == F(459, 7)
    for row in control['connected_cover_and_shared_Gauss']:
        assert row['combined_cover_degree'] == 4 and row['divE_minus_rho_exact'] == 0
        assert row['canonical_neutral'] == 2 * row['V']
        assert row['full_HE_plus_HC_per_site'] == 262144
        assert F(row['wrong_normalized_norm_squared']) != 1
    assert load('sea_energy_review_actual5_20261012_verdict.json')['energy_below_minus21'] is True
    return {'full_H_per_Lambda_site_upper_exact': str(total),
            'full_corrections_per_Lambda_site_exact': {k: str(v) for k, v in terms.items()},
            'trial_scope': 'Full physical normalized source, every even N>=4; NOT minimizing ground.',
            'primitive_projected_deck': '2pi B (B^T B)^-1 Z^d; unit-character normalized Haar d beta',
            'normalization': 'Single global Z4 cover degree4; prefactor2; local Jacobians cancel before estimates.',
            'global_minimizer_or_phase_identified': False}


def exact_localization_and_marginals() -> dict[str, Any]:
    local = load('local_body_leading_actual2_20261012_receipt.json')
    coeff = local['neutral_coefficients']
    m, c0, ci = F(coeff['m']), F(coeff['c0']), F(coeff['ci'])
    b = abs(c0) + abs(ci)
    assert F(coeff['b']) == b
    R, cap, moment = 4096, 2**32, F(2**14)
    failed_boundary, ims_boundary = 12 * b * cap / R, 144 * b / R
    assert failed_boundary == F(local['first_gate']['boundary_error']) > 21
    second = local['material_second_route']
    assert ims_boundary == F(second['boundary_error_per_Lambda_site_upper']) < 1
    assert F(second['actual_family_cutoff_failure_upper']) == R**3 * moment / cap**4
    a, c = F(40, 2**2035), F(10, 7 * 2**2035)
    assert F(second['actual_family_mean_field_deviation_failure_upper']) == 48 * R**4 * a * 2**200
    assert F(second['actual_family_mean_flat_link_failure_upper']) == 1728 * R**5 * c * 2**200
    canonical = local['canonical_compensation']
    assert F(canonical['complete_top4']) == F(canonical['dual_exact'])
    assert F(canonical['complete_top4']) != F(canonical['wrong_two_each'])
    assert F(canonical['complete_top4']) != F(canonical['wrong_positive_only'])
    radial = load('history_operator_reset_parent_ground_radial_window_corrected2_20261012_receipt.json')
    a0, ai = m*m + 6*c0*c0, 2*ci*ci
    assert F(radial['neutral_trace_square_coefficients']['A0']) == a0 < F(51, 2)**2
    assert F(radial['neutral_trace_square_coefficients']['Ai']) == ai
    assert 3*ai**2/128 < F(28, 3)**3
    dch = F(local['charged64_full_length12_linear_coefficient'])
    assert dch == F(radial['all64_charged_Dch_exact'])
    assert 12*dch/F(2**8192) < F(1, 2)
    floor = F(21, 26)**2
    probability = (floor-F(1, 4))**2/moment - moment/F(2**16)**2
    phi0floor = F(13, 51)**2
    phi0probability = (phi0floor-F(1, 64))**2/moment - moment/F(2**20)**2
    assert F(radial['actual_family_mean_s_lower_exact']) == floor
    assert F(radial['conditional_window_probability_lower_exact']) == probability > 0
    assert F(radial['Phi0_static_component']['actual_family_mean_TrPhi0_squared_lower_exact']) == phi0floor
    assert F(radial['Phi0_static_component']['conditional_probability_lower_exact']) == phi0probability > 0
    return {'failed_first_boundary_exact': str(failed_boundary),
            'distinct_IMS_boundary_exact': str(ims_boundary),
            'IMS_displayed_price_range': 'R4096 only N>=8192; smaller N use admissible windows.',
            'tree_quantum_bracket': 'Average ALL translated partitions, not arbitrary fixed cut.',
            'conditional_actual_family_mean_s_lower': str(floor),
            'conditional_s_window_probability_lower': str(probability),
            'conditional_actual_family_raw_Phi0_squared_mean_lower': str(phi0floor),
            'conditional_Phi0_window_probability_lower': str(phi0probability),
            'measure': 'Inherited physical Gauss/sewn-cover Phi marginal; translation-averaged globally minimizing family.',
            'raw_vs_quarter': 'Raw q=meanTrPhi^2: J>=4sumq^2. Quarter X=q/4: J>=64sumX^2.',
            'simultaneous_patch_or_Clifford_or_loop_Gram_inference': False,
            'untruncated_patch_minimum_or_body_rigidity_paid': False}


def _drop_metadata_path(payload: dict[str, Any], path: str) -> None:
    """Remove one declared metadata key, including literal absolute-path keys."""
    current = payload
    remaining = path.removeprefix('/')
    while '/' in remaining and remaining not in current:
        key, remaining = remaining.split('/', 1)
        if key not in current:
            return
        child = current[key]
        assert isinstance(child, dict), path
        current = child
    current.pop(remaining, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for member, expected in request['pins'].items():
        assert sha(Path(request['pin_paths'][member]).read_bytes()) == expected, member
    epochs = [verify_epoch(spec) for spec in request['epochs']]
    transfers = []
    for transfer in request['typed_transfers']:
        old, new = load(transfer['original']), load(transfer['typed'])
        for field in transfer['metadata_fields']:
            old.pop(field, None)
            new.pop(field, None)
        for path in transfer.get('metadata_paths', []):
            _drop_metadata_path(old, path)
            _drop_metadata_path(new, path)
        assert old == new, transfer
        transfers.append({**transfer, 'mathematical_payload_exact_equal': True})
    jensen = load('history_operator_reset_parent_neutral_Jordan_Jensen_parenttyped2_20261012_receipt.json')
    assert F(jensen['second_directional_derivative_exact']) == F(-28936637, 8388608)
    assert F(jensen['mean_endpoint_J_minus_midpoint_J_exact']) < 0
    result = {'kind': 'EXACT_CONNECTED_QUANTUM_SEA_LOCAL_BODY_ACTUAL_FAMILY_IMMUTABLE_JOIN_NOT_GROUND_OR_PHASE',
              'governor': request['governor'], 'epochs': epochs,
              'preexecution_members_compared': sum(x['preexecution_members_compared'] for x in epochs),
              'typed_transfers': transfers, 'quantum_trial': exact_quantum_trial(),
              'localization_and_marginals': exact_localization_and_marginals(),
              'Jensen_POINT_counter_not_ground_or_local_Lipschitz_no_go': True,
              'original_targets_and_reviews_not_rebound': True,
              'unconditional_machine_attestation': False,
              'actual_minimizing_ground_or_fullH_numerical_evolution_computed': False,
              'physical1_phase_stiffness_IR_atom_or_original_SOURCE_adopted': False,
              'campaign_complete': False}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'epochs': len(epochs), 'members_compared': result['preexecution_members_compared'],
                      'full_H_trial_upper_below_minus21': True, 'ground_or_phase_admitted': False}))


if __name__ == '__main__':
    main()

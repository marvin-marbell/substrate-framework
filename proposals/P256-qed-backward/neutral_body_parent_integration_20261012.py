"""Immutable neutral/charged join with exact independent boundary arithmetic.

No original producer is executed. Uniform-body, quantum-trial and FLOAT shell
response remain distinct; no body or projector is transported by this join.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from fractions import Fraction as F
from pathlib import Path
from typing import Any
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text())


def verify_epoch(spec: dict[str, Any]) -> dict[str, Any]:
    cap = load(spec['capsule'])
    pre_name, exe_name = spec['preexecution'], spec['executed']
    assert sha((ROOT / pre_name).read_bytes()) == spec['pre_sha256']
    assert sha((ROOT / exe_name).read_bytes()) == spec['exe_sha256']
    with ZipFile(ROOT / pre_name) as before, ZipFile(ROOT / exe_name) as after:
        assert len(before.namelist()) == len(set(before.namelist()))
        assert len(after.namelist()) == len(set(after.namelist()))
        container = spec.get('executed_input_archive_member')
        after_inputs = ZipFile(BytesIO(after.read(container))) if container else after
        for member in before.namelist():
            data = before.read(member)
            assert after_inputs.read(member) == data, (spec['epoch'], member)
        if container:
            after_inputs.close()
        if spec['nested_preexecution']:
            assert after.read(pre_name) == (ROOT / pre_name).read_bytes()
        for locator, member in spec['input_members'].items():
            path = Path(locator)
            if not path.is_absolute():
                path = ROOT / path
            assert sha(before.read(member)) == spec['input_pins'][locator]
            assert sha(path.read_bytes()) == spec['input_pins'][locator]
        for name, expected in spec['outputs'].items():
            assert sha(after.read(name)) == expected
            assert sha((ROOT / name).read_bytes()) == expected
        for member, expected in spec.get('archive_only_inputs', {}).items():
            assert sha(before.read(member)) == expected
        for member, expected in spec.get('archive_only_outputs', {}).items():
            assert sha(after.read(member)) == expected
        count = len(before.namelist())
    runtime = spec['runtime']
    assert runtime['natural_exit'] is True
    assert runtime['returncode'] == spec['expected_returncode']
    # A capsule's omitted post-contract field is not a fabricated equality.
    # All members, including the authoritative contract, were checked above.
    return {'epoch': spec['epoch'], 'members_compared': count,
            'runtime': runtime, 'all_preexecution_members_exact': True,
            'all_output_pins_exact': True, 'live_selected_inputs_exact': True,
            'historical_archive_only_inputs': list(spec.get('archive_only_inputs', {})),
            'historical_archive_only_outputs': list(spec.get('archive_only_outputs', {})),
            'capsule_kind': cap['kind']}


def exact_join_controls() -> dict[str, Any]:
    cert = load('history_operator_reset_parent_neutral_uniform_class_actual2_20261012_receipt.json')
    interior = []
    for row in cert['certificates']:
        x = [F(value) for value in row['norm_squared_candidate_exact']]
        g2 = F(row['gradient_norm2_upper_exact'])
        assert F(row['candidate_energy_excess_upper_exact']) == g2 / 256
        assert F(row['candidate_norm_squared_distance2_upper_exact']) == g2 / 128**2
        assert g2 / 128**2 < min(v*v for v in x)
        candidate = list(map(F, row['candidate_energy_interval_exact']))
        minimum = list(map(F, row['unique_uniform_class_minimum_interval_exact']))
        assert minimum == [candidate[0]-g2/256, candidate[1]]
        interior.append({'N': row['N'], 'certified_interior_by_distance': True,
                         'strong_convex_excess_exact': str(g2/256)})
    seam = load('history_operator_reset_parent_neutral_holonomy_actual1_20261012_receipt.json')
    assert seam['constant_i_links_winding'] == '1'
    assert seam['literal_i_seam_winding'] == 'i'
    assert seam['AP_momenta_pi_over_8']['literal_i_seam'] == [3,7,11,15]
    old = list(map(F, seam['complete_Clifford_canonical2V_point_energy_intervals_exact']['identity']))
    new = list(map(F, seam['complete_Clifford_canonical2V_point_energy_intervals_exact']['literal_i_seam']))
    delta = list(map(F, seam['literal_seam_minus_identity_point_energy_interval_exact']))
    assert delta == [new[0]-old[1], new[1]-old[0]] and delta[0] > 0
    charged = load('native_charged_body_four_edge_actual3_20261012_receipt.json')
    shell = charged['shell']
    total = F(shell['C_internal_exact']) + F(shell['C_boundary_all_native_exact'])
    current = F(charged['source_complement']['each_current_operator_norm_bound_exact'])
    assert current == F(shell['C_boundary_all_native_exact'])
    scalar64 = total*total / 64**3
    assert scalar64 > 1
    assert charged['scope']['actual_ground_response_or_magnetic_stiffness'] is False
    assert charged['ground_compression']['not_fullH_originating_P0'] is True
    native = load('native_neutral_body_actual1_20261012_receipt.json')
    assert native['leading']['canonical_negative_levels_also_required'] == 64
    assert native['quantum']['ground_BO_Berry_error_paid'] is False
    assert native['quantum']['uniform_all_even_N_ge4'] is True
    return {'uniform_norm_controls': interior,
            'genuine_centre_seam_fixed_point_delta_exact': list(map(str,delta)),
            'conditional_same_centre_Schur_scalar_eta64_exact': str(scalar64),
            'conditional_full_current_source_remainder_eta64_exact': str(current*current*(scalar64+F(3,64))),
            'Schur_bound_hypothesis': 'Ctotal bounds B=QHP only in the declared four-edge module, not full torus with frozen exterior sea. Reducing exterior terms is a separate unpaid transition.',
            'local_incident_price_bounds_global_frozen_exterior_QHP': False,
            'canonical_negative_compensation_retained': True,
            'uniform_minimum_IS_NOT_full_quantum_ground': True,
            'trial_kinetic_price_IS_NOT_ground_BO_remainder': True,
            'shell_P0_IS_NOT_originating_whole_P0': True,
            'originating_ground_or_body_or_P0_transport_paid': False}


def reviewed_join_controls() -> dict[str, Any]:
    schur = load('charged_schur_review_actual2_20261012_receipt.json')['native']
    profile = load('neutral_body_parent_primitive_profiles_20261012.json')['primitive_profiles'][1]
    ci = F(profile['paths'][0]['ci'])
    total2 = F(schur['C_total_squared_exact'])
    witnesses = []
    for row in schur['remote_frozen_sea_controls']:
        n = row['N']
        paths = 3*n**3-20
        gram = 64*4*paths*(ci/3)**2
        assert paths == row['disjoint_positive_NN_paths']
        assert gram == F(row['remote_QHP_sea_norm_squared_exact'])
        witnesses.append({'N': n, 'exact_remote_Gram': str(gram), 'exceeds_local_price': gram > total2})
    assert witnesses[-1]['exceeds_local_price']
    assert all(64*4*(3*n**3-20)*(ci/3)**2 <= total2 for n in range(4,100,2))
    assert schur['fixed_sea_full_torus_local_Ctotal_refuted']
    return {'remote_frozen_exterior_witnesses': witnesses,
            'first_even_N_local_price_failure': 100,
            'local_QHP_full_torus_claim_withdrawn': True,
            'large_IR_upper_is_not_no_go': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, expected in request['pins'].items():
        assert sha((ROOT/name).read_bytes()) == expected, name
    epochs = [verify_epoch(spec) for spec in request['epochs']]
    transfers = []
    for row in request['typed_transfers']:
        original, typed = load(row['original_receipt']), load(row['typed_receipt'])
        assert original == typed
        transfers.append({**row, 'mathematical_payload_exact_equal': True})
    result = {'kind': 'EXACT_NEUTRAL_UNIFORM_POLYNOMIAL_AND_CHARGED_SCHUR_IMMUTABLE_JOIN_NOT_GROUND',
              'governor': request['governor'], 'epochs': epochs,
              'members_compared': sum(row['members_compared'] for row in epochs),
              'exact_mathematical_join': exact_join_controls(),
              'independent_scientific_reviews_pending': False,
              'reviewed_exact_controls': reviewed_join_controls(),
              'typed_transfers': transfers,
              'unconditional_machine_attestation': False,
              'native_fullH_or_ground_numerical_backend': False,
              'physical1_phase_stiffness_IR_atom_admitted': False,
              'original_SOURCE_or_law_adopted': False, 'campaign_complete': False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'epochs':len(epochs),'members_compared':result['members_compared'],
                      'physical1_phase_stiffness_IR_atom_admitted':False}))


if __name__ == '__main__':
    main()

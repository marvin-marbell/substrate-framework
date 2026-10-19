"""Revision-bound parent integration with independent canonical/form/character arithmetic."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent
GAMMA = F(46955312929, 34359738368)


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    return value


def load(name: str) -> Any:
    return json.loads((ROOT / name).read_text())


def verify_capsule(name: str, layout: str, record_key: str = 'receipt') -> dict[str, Any]:
    cap = load(name)
    assert cap['returncode'] == 0 and cap['natural_exit'] and cap['sources_inputs_unchanged']
    if layout == 'flat':
        pre_name, exe_name = cap['files']['preexecution.zip'], cap['files']['executed.zip']
        record_name = cap['files']['receipt.json']
        assert digest(ROOT / pre_name) == cap['preexecution_sha256']
        assert digest(ROOT / exe_name) == cap['executed_sha256']
        assert digest(ROOT / record_name) == cap['receipt_sha256']
    else:
        pre_name, exe_name = cap['preexecution']['path'], cap['executed']['path']
        record_name = cap[record_key]['path']
        assert digest(ROOT / pre_name) == cap['preexecution']['sha256']
        assert digest(ROOT / exe_name) == cap['executed']['sha256']
        assert digest(ROOT / record_name) == cap[record_key]['sha256']
    records = 0
    with zipfile.ZipFile(ROOT / pre_name) as pre, zipfile.ZipFile(ROOT / exe_name) as exe:
        for item in pre.namelist():
            target = item
            if layout == 'native':
                target = ('executed_inputs/' + item.removeprefix('inputs/')) if item.startswith('inputs/') else pre_name.removesuffix('_preexecution.zip') + '_request.json'
            assert pre.read(item) == exe.read(target), (name, item, target)
            records += 1
        result_member = 'result.json' if record_key == 'result' else record_name
        assert hashlib.sha256(exe.read(result_member)).hexdigest() == digest(ROOT / record_name)
    return {'capsule': name, 'natural_exit': True, 'pre_to_executed_members': records,
            'record': record_name, 'live_source_equality_not_inferred_from_old_epoch': True}


def exact_spectrum(axis: dict[str, Any], which: str, error_name: str) -> tuple[F, F]:
    roles = axis[which + '_onebody_exact_operator']
    assert sorted(r['physical_count'] for r in roles) == [1, 63]
    lower: list[F] = []
    upper: list[F] = []
    scalar = F(0)
    for role in roles:
        count = role['physical_count']
        scalar += count * F(role['scalar_exact'])
        for interval in role['eigenvalue_intervals']:
            low, high = map(F, interval['exact'])
            assert low <= high
            lower.extend([low] * count)
            upper.extend([high] * count)
    assert len(lower) == len(upper) == 512
    error = F(axis['outer_ground_coefficient_errors_exact'][error_name])
    answer = (scalar + sum(sorted(lower)[:256], F(0)) - error,
              scalar + sum(sorted(upper)[-256:], F(0)) + error)
    # Mode allocation is GLOBAL N256, not a per-copy filling convention.
    assert answer == tuple(map(F, axis[which + '_spectrum_entire_canonical256']['exact']))
    return answer


def native_prices(value: dict[str, Any]) -> list[dict[str, Any]]:
    d = F(value['domains']['whole_operator_two_cell_floor_exact'])
    b = F(value['domains']['source_two_cell_floor_exact'])
    assert b - d == 35 and d > 0
    formation = value['formation']
    infidelity = (F(1, 65536) / GAMMA) ** 2
    assert F(formation['infidelity_from_accepted_cap_exact']) == infidelity
    assert F(formation['ground_distance_cap_exact']) ** 2 >= 2 * infidelity
    assert F(formation['charged_multicell_ground_distance_exact']) ** 2 >= 6 * infidelity
    output = []
    for axis in value['axes']:
        glo, ghi = exact_spectrum(axis, 'G', 'G')
        _, tcap = exact_spectrum(axis, 'TdagT', 'TdagT')
        gl, gu = map(F, axis['G_ground_expectation']['exact'])
        centered = max(abs(glo - gu), abs(ghi - gl))
        assert centered == F(axis['signed_centered_G_operator_norm_cap_exact'])
        for row in axis['projection_accounting']:
            proj = row['projection']
            pair = F(proj['pre_normal_order_term_pair_L1_exact'])
            normal = F(proj['post_normal_order_pre_outer_collection_L1_exact'])
            assert F(proj['formation_L1_exact']) == max(pair, normal)
        for row in axis['energies']:
            E = F(row['E_exact'])
            A = F(axis['anchor_exact']) - E
            assert A > 0 and d > E
            wl, wu = map(F, row['vacuum_w0']['exact'])
            independent = {'signed_centered_G': centered / (GAMMA * (A + GAMMA)),
                           'scalar_energy_counterterm': gu / (A * (A + GAMMA)),
                           'vacuum_scalar_mismatch': max(abs(gl / A - wu), abs(gu / A - wl)) / GAMMA}
            saved = row['whole_excited_connected_form_components_exact']
            assert all(v == F(saved[k]) for k, v in independent.items())
            remainder = F(saved['entire_spectrum_intertwiner_remainder'])
            radical = remainder * (d - E) * GAMMA * (A + GAMMA)
            assert radical > 0 and radical ** 2 >= max(abs(glo), abs(ghi)) * tcap
            bound = sum(independent.values(), F(0)) + remainder
            assert bound == F(row['whole_excited_connected_relative_cap_exact'])
            metric = 1 + max(abs(glo), abs(ghi)) / (d - E) ** 2
            assert metric == F(row['metric_I_plus_Wprime_upper_exact'])
            assert bound > 1 and not row['reaches_one']
            output.append({'axis': axis['axis'], 'E_exact': str(E),
                           'independently_reconstructed_bound_exact': str(bound),
                           'bound_display': float(bound), 'metric_exact': str(metric),
                           'whole_spectrum_radical_outward': True,
                           'physical1_contraction': False})
    assert not value['higher_charge_intercell_domains_paid']
    return output


def character_geometry(n: int) -> dict[str, Any]:
    def add(p: tuple[int, int, int], axis: int) -> tuple[int, int, int]:
        return ((p[0] + int(axis == 0)) % n, (p[1] + int(axis == 1)) % n,
                (p[2] + int(axis == 2)) % n)
    signatures = set()
    for x in range(n):
        for y in range(n):
            for z in range(n):
                p = (x, y, z)
                edges = {(p, 0): F(1, 2), (add(p, 0), 1): F(1, 2),
                         (add(p, 1), 0): F(-1, 2), (p, 1): F(-1, 2)}
                divergence: dict[tuple[int, int, int], F] = {}
                for (start, axis), coefficient in edges.items():
                    end = add(start, axis)
                    divergence[start] = divergence.get(start, F(0)) + coefficient
                    divergence[end] = divergence.get(end, F(0)) - coefficient
                assert all(v == 0 for v in divergence.values())
                assert sum((v * v for v in edges.values()), F(0)) == 1
                signatures.add(tuple(sorted(edges.items())))
    assert len(signatures) == n ** 3
    # Every nonzero difference coefficient is +/-1/2 or +/-1. For 0<s<1,
    # s(b_x-b_y) cannot be an integer character. No ground is computed here.
    return {'N': n, 'closed_distinct_normalized_faces': len(signatures),
            'conditional_total_source_Gram': 'identity',
            'conditional_fourier_m0': '1', 'conditional_fourier_m1': '64s^2',
            'whole_P0_excited_mass': '1-||P0 A_s(q)Omega||^2, unpriced',
            'infinite_volume_pointwise_q_or_ground_or_photon': False}


def main() -> None:
    sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    request = json.loads(args.request.read_text())
    assert all(digest(ROOT / n) == pin for n, pin in request['pins'].items())
    capsule_rows = [verify_capsule('vacuum_connected_native_parent_budget3_20261011_capsule.json', 'native'),
                    verify_capsule('fractional_flux_ir_parenttyped5_20261011_capsule.json', 'flat'),
                    verify_capsule('vacuum_operator_review_actual1_20261011_capsule.json', 'same', 'result'),
                    verify_capsule('weyl_ground_review_actual1_20261011_capsule.json', 'flat')]
    native = load('vacuum_connected_native_parent_budget3_20261011_receipt.json')
    rows = native_prices(native)
    weyl = load('fractional_flux_ir_parenttyped5_20261011_receipt.json')
    assert weyl['HE'] == weyl['HC'] == 64 and weyl['physical_hopping'] == 1
    assert not weyl['ground_or_photon_admitted']
    packet_rows = []
    for row in weyl['original_normalized_packet_controls']['packet_controls']:
        s = F(row['s']); delta = F(row['illustrative_packet_defect_budget'])
        assert delta == (s / 16) ** 2
        bound = (s / 16 + 8 * s) ** 2
        assert bound == F(row['shifted_excess_budget_exact'])
        packet_rows.append({'N': row['N'], 's': str(s), 'packet_excess_budget': str(bound),
                            'native_minimizing_cluster_not_computed': True})
    result = {'kind': 'INTEGRATED_VACUUM_OPERATOR_AND_FULL_FORM_WEYL_PARENT_CONSUMER_NOT_CAMPAIGN_CLOSURE',
              'capsule_bindings': capsule_rows, 'native_form_rows': rows,
              'source_geometry': [character_geometry(n) for n in (4, 6, 8)],
              'original_measure_packet_arithmetic': packet_rows,
              'sharp_total_mass_not_excited_mass': True,
              'regularity_phase_atom_still_unpaid': True,
              'historical_machine_conditions_not_attested': True,
              'entire_interacting_spatial_Schur_not_paid': True,
              'physicalPhi_ground_or_photon_adopted': False, 'campaign_complete': False,
              'elapsed_seconds': time.monotonic() - started}
    assert all(digest(ROOT / n) == pin for n, pin in request['pins'].items())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'native_allvector_guard_failed': True, 'packet_and_character_arithmetic': True,
                      'whole_P0_unpriced': True, 'native_photon': False}), flush=True)


if __name__ == '__main__':
    main()

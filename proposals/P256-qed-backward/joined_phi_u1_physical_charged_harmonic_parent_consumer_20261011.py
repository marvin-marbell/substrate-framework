"""Compose reviewed physical charged gain and distinct frozen spectral diagnostics."""
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

sys.set_int_max_str_digits(0)
ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    own = Path(__file__).read_bytes()
    manifest_bytes = (ROOT / args.manifest).read_bytes()
    manifest: dict[str, Any] = json.loads(manifest_bytes)
    captured = {n: (ROOT / n).read_bytes() for n in manifest['files']}
    assert all(sha(raw) == manifest['files'][n] for n, raw in captured.items())
    members = 0
    historical_sources = []
    for binding in manifest['archives']:
        with zipfile.ZipFile(ROOT / binding['path']) as archive:
            for member, pin in binding['members'].items():
                raw = archive.read(member)
                assert sha(raw) == pin, (binding['path'], member)
                if member in captured and manifest['files'][member] == pin:
                    assert raw == captured[member]
                elif member.endswith('.py'):
                    historical_sources.append({'archive': binding['path'], 'member': member, 'sha256': pin})
                members += 1

    def data(key: str) -> Any:
        return json.loads(captured[manifest[key]])

    energy = data('energy_review')
    field = data('field_review')
    corrected = data('corrected_physical')
    old = data('old_physical')
    increment = corrected['rational_increment']
    assert increment == old['rational_increment']
    absolute = corrected['absolute_source_budget']
    assert absolute == old['absolute_source_budget']
    leading = F(increment['leading_net_increment_per_site_exact'])
    prices = {n: F(v) for n, v in increment['positive_remainder_prices_per_site_exact'].items()}
    upper = leading + sum(prices.values())
    assert leading < 0 and upper == F(increment['increment_upper_per_site_exact']) < leading / 2
    assert F(absolute['neutral_circuited_baseline_energy_upper_per_site_exact']) < -(2 ** 6070)
    assert F(absolute['charged_after_energy_upper_per_site_exact']) < F(absolute['neutral_circuited_baseline_energy_upper_per_site_exact'])
    assert all(prices[n] > 0 for n in prices)
    energy_rows = energy['rational_profiles']
    assert {r['eta_exponent'] for r in energy_rows} == {4000, 4200}
    repaired = next(r for r in energy_rows if r['eta_exponent'] == 4200)
    rejected = next(r for r in energy_rows if r['eta_exponent'] == 4000)
    assert F(repaired['increment_upper_perSITE']) == upper
    assert F(rejected['absolute_after_lower_perSITE']) > 2 ** 6240
    field_rows = field['rational_review']['independent_price_reproductions']
    field_repaired = next(r for r in field_rows if r['eta_exponent'] == 4200)
    assert all(F(v) == prices[n] for n, v in field_repaired['four_field_prices_exact'].items())
    assert field['rational_review']['whole64_G0_cap_verified']
    observed = corrected['observed_controls']
    assert observed['even_Bloch_boost_energy_defect'] < 1e-10
    assert observed['omitted_Bloch_twist_complete_action_error'] > 1e-2
    assert F(observed['coordinate_circle_packet_energy_lower_exact']) > F(observed['Bloch_circle_packet_energy_upper_exact'])
    assert observed['Bloch_beta_is_momentum_not_theta']

    harmonic = data('harmonic')
    inverse = data('feshbach')
    assert harmonic['input_sha256']['local_so5_resum_actual04_20261011_state.npz'] == manifest['selected_state_sha256']
    assert inverse['input_receipt_sha256'] == manifest['files'][manifest['harmonic']]
    flow = data('affine_geometry')
    assert flow['axis0_line_positive_control'] and flow['transverse_common_u_adoption_rejected']
    assert harmonic['input_sha256'][manifest['affine_geometry']] == manifest['files'][manifest['affine_geometry']]
    geometry = {}
    for row in flow['rows']:
        key = (row['axis'], row['length'], tuple(row['sites']), row['qa'])
        offsets = tuple(F(z) for z in row['affine_u'])
        assert geometry.setdefault(key, offsets) == offsets
    source_enclosure = F(harmonic['roundoff_scope']['density_entry_enclosure_exact'])
    ground = data('selected_ground')
    infidelity = F(ground['selected_ground_infidelity_upper_exact'])
    distance = F(1, 60000)
    assert 2 * infidelity < distance * distance
    binary_u = F(1, 2 ** 53)
    operation_count = 4 * harmonic['immutable_source_dimension'] + 32
    gamma = operation_count * binary_u / (1 - operation_count * binary_u)
    assert gamma < F(1, 2 ** 26)
    # Fixed-slot sparse degrees <=2 and normalized CAR maps; this is the
    # stated binary64 formation/reduction assumption, not an interval for T/U.
    numeric_margin = F(1, 2 ** 18)
    assert 2 * distance + numeric_margin < source_enclosure
    lower = data('charged_lower')
    floors = {b['branch']: F(b['floor_exact']) for b in lower['branches']}
    E0_upper = F(ground['ground_energy_interval_exact'][1])
    original_paths = data('old_first_shell')['all64_boundary']['roles']
    q = F(0)
    for path in harmonic['paths']:
        assert [r['count'] for r in path['roles']] == [1, 63]
        term = F(0)
        for role, old_role in zip(path['roles'], original_paths):
            original = next(p for p in old_role['paths'] if (p['axis'], p['length']) == (path['axis'], path['length']))
            allowed_sites = {key[2] for key in geometry if key[:2] == (path['axis'], path['length'])}
            assert {tuple(z['sites']) for z in role['all_sites_both_charge_choices']} == allowed_sites
            for choice in role['all_sites_both_charge_choices']:
                for orientation in choice['orientations']:
                    key = (path['axis'], path['length'], tuple(choice['sites']), orientation['qA'])
                    assert tuple(F(z) for z in orientation['affine_u_exact']) == geometry[key]
            absolute_entries = sum(abs(F(re)) + abs(F(im)) for row in original['matrix_exact'] for re, im in row)
            error = 2 * (2 * source_enclosure + source_enclosure ** 2) * absolute_entries ** 2
            source_upper = F(role['source_weight_float_max_all_sites']) + error
            denominator = 64 * original['external_edges_min'] + 2 * floors['active' if role['count'] == 1 else 'aux'] - 2 * E0_upper
            assert denominator > 0 and denominator == F(role['charged_gap_lower_exact'])
            assert source_upper == F(role['source_weight_upper_exact'])
            term += role['count'] * source_upper / denominator ** 2
        weighted = F(path['allocation_exact']) * term
        assert weighted == F(path['weighted_inverse_square_interval_upper_exact'])
        q += weighted
    assert q == F(harmonic['sourceweighted_gap_guard_upper_exact']) > 1
    assert not harmonic['sourceweighted_gap_guard_less_than_one']
    assert harmonic['moment_heat_guard_float_not_interval'] > 1
    assert inverse['weighted_Cauchy_inverse_square_lower_float'] > 1
    assert inverse['minimum_full_endpoint_Gram_eigenvalue'] > 0
    assert harmonic['source_and_inputs_unchanged']

    result = {
        'kind': 'REVIEW_BOUND_FULLPHI_CHARGED_GAIN_AND_DISTINCT_SELECTED_SOURCE_DIAGNOSTICS',
        'task_governor': manifest['task_governor'],
        'accepted_science_before_packet': manifest['accepted_science'],
        'corrected_physical_source_sha256': manifest['files'][manifest['corrected_source']],
        'charged_per_site_increment_upper_exact': str(upper),
        'strict_half_gain': True,
        'negative_baseline_per_site_upper_less_than_minus2pow6070': True,
        'total_bound_is_V_times_per_site': True,
        'rejected_eta4000_absolute_lower_greater_than2pow6240': True,
        'corrected_Bloch_action_control': observed['omitted_Bloch_twist_complete_action_error'],
        'even_Bloch_boost_energy_defect': observed['even_Bloch_boost_energy_defect'],
        'physical_source_domain': 'D(H), not D(H^2)',
        'selected_source_guard_upper_exact': str(q),
        'selected_source_guard_upper_float': float(q),
        'selected_source_enclosure_conditional_standard_binary64': True,
        'heat_and_Feshbach_FLOAT_NOT_INTERVAL': True,
        'primitive_geometry_corrected_without_charged_gap_or_source_G_change': True,
        'transverse_wrong_common_offset_HE_control': flow['signed_CAR_actual_axis1_wrongcase'],
        'heat_guard_float': harmonic['moment_heat_guard_float_not_interval'],
        'Feshbach_inverse_square_lower_float': inverse['weighted_Cauchy_inverse_square_lower_float'],
        'selected_local_source_IS_NOT_fullphysical_source': True,
        'firstshell_guard_FAILED_NOT_allorder_or_global_no_go': True,
        'historical_source_bindings': historical_sources,
        'verified_archive_members': members,
        'input_sha256': manifest['files'],
        'own_source_sha256': sha(own),
        'manifest_sha256': sha(manifest_bytes),
        'source_inputs_unchanged': all((ROOT / n).read_bytes() == raw for n, raw in captured.items()) and Path(__file__).read_bytes() == own,
        'fullH_numeric_variance': False,
        'fullPhi_or_lattice_stationary_ground': False,
        'normalized_allorder_phase': False,
        'bulk_stiffness': False,
        'IRatom_or_sharp_photon': False,
        'commoncone': False,
        'originalSOURCE': False,
        'campaign_complete': False,
        'elapsed_seconds': time.monotonic() - start,
    }
    assert result['source_inputs_unchanged']
    with (ROOT / args.output).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in {'input_sha256', 'charged_per_site_increment_upper_exact', 'historical_source_bindings'}}, indent=2))


if __name__ == '__main__':
    main()

"""Independent literal boundary-response/moment composition, not a phase theorem."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from fractions import Fraction as F
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def archive_pair(epoch: dict[str, Any]) -> dict[str, Any]:
    pre_path, exe_path = ROOT / epoch['pre'], ROOT / epoch['executed']
    assert digest(pre_path) == epoch['pre_sha256']
    assert digest(exe_path) == epoch['executed_sha256']
    with zipfile.ZipFile(pre_path) as pre, zipfile.ZipFile(exe_path) as exe:
        names = [name for name in pre.namelist() if not name.endswith('/')]
        assert len(names) == len(set(names))
        for name in names:
            assert pre.read(name) == exe.read(name), name
        receipt_bytes = exe.read(epoch['receipt'])
        assert hashlib.sha256(receipt_bytes).hexdigest() == epoch['receipt_sha256']
        receipt = json.loads(receipt_bytes)
        assert receipt_bytes == (ROOT / epoch['receipt']).read_bytes()
    return {'members_compared': len(names), 'receipt': receipt}


def boundary_coefficients(profiles: list[dict[str, Any]]) -> dict[str, F]:
    # ||c_x^dagger B c_y+h.c.|| <=8||B||, endpoint average /2.
    # Each boundary site is one outgoing and one incoming endpoint per z row.
    cj, ck = F(0), F(0)
    for profile in profiles:
        if profile['scale'] != 'charged':
            continue
        for row in profile['paths']:
            if row['axis'] == 2:
                b = abs(F(row['c0'])) + abs(F(row['ci']))
                cj += 8 * profile['load'] * b * row['length']
                ck += 8 * profile['load'] * b * row['length'] ** 2
    assert cj > 0 and ck > cj
    # Wrong one-flavour load changes the physical coefficient, not only a label.
    one_role = [{**p, 'load': 1} for p in profiles if p['scale'] == 'charged']
    wrong_cj = sum((8 * p['load'] * (abs(F(r['c0'])) + abs(F(r['ci']))) * r['length']
                    for p in one_role for r in p['paths'] if r['axis'] == 2), F(0))
    assert wrong_cj < cj
    return {'CJ': cj, 'CK': ck, 'wrong_role_CJ': wrong_cj}


def concentration_control(n: int) -> dict[str, Any]:
    # This scalar HS-budget witness is not an actual Gibbs/ground expectation.
    # n_b boundary sites and uniform z^4=V/n_b saturate the available Holder cap.
    v, nb = n ** 3, 2 * n ** 2
    z4 = F(v, nb)
    sum_z4 = nb * z4
    assert sum_z4 == v
    linear_sum_fourth = F(nb ** 3 * v, 4)  # r=|z|/sqrt(2)
    wrong_uniform_site_fourth = F(nb ** 4, 4)
    assert linear_sum_fourth > wrong_uniform_site_fourth
    # Actual-degree denominator/boundary counting: at fixed mean z^4=1,
    # |J|^2 cap/area grows like N^(5/2), so it is NOT uniform IR control.
    return {'N': n, 'V': v, 'boundary_sites': nb, 'sum_z4_exact': str(sum_z4),
            'saturating_boundary_linear_sum_fourth_exact': str(linear_sum_fourth),
            'wrong_selected_site_bound_fourth_exact': str(wrong_uniform_site_fourth),
            'ratio_exact': str(linear_sum_fourth / wrong_uniform_site_fourth),
            'selected_site_from_average_rejected': True,
            'witness_is_native_thermal_state': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, pin in request['pins'].items():
        assert digest(ROOT / name) == pin, name
    archive_rows = {name: archive_pair(epoch) for name, epoch in request['epochs'].items()}
    thermal = archive_rows['thermal']['receipt']
    bulk = archive_rows['bulk']['receipt']
    assert thermal['native_numeric_expectations'] is False
    assert thermal['magnetic_response_sign_paid'] is False
    assert bulk['actual_native_state_spectral_convergence'] is False
    assert all(row['scope'] == 'fullE longitudinal Gauss reservoir spectral connection ONLY'
               for row in (entry['standing_Gauss'] for entry in bulk['controls']))
    with zipfile.ZipFile(ROOT / request['epochs']['thermal']['executed']) as exe:
        profiles = json.loads(exe.read(request['profile']))['primitive_profiles']
    coeff = boundary_coefficients(profiles)
    # Native producer's global bound must dominate the endpoint-local bound.
    for row in (entry['exact_thermal_enclosures'] for entry in thermal['native_N_controls']):
        n = row['N']
        assert F(row['MJ_N']) == coeff['CJ'] * (2 * n ** 2)
        assert F(row['MK_N']) == coeff['CK'] * (2 * n ** 2)
    result = {
        'kind': 'JOINED_ACTUAL_CANONICAL_THERMAL_MOMENT_RESPONSE_COMPOSITION_NOT_UNIFORM_IR',
        'source_sha256': digest(Path(__file__)), 'request_sha256': digest(args.request),
        'input_sha256': request['pins'],
        'original_archive_members': {k: v['members_compared'] for k, v in archive_rows.items()},
        'boundary_coefficients_exact': {k: str(v) for k, v in coeff.items()},
        'derived_conditional_response': {
            'premise': 'Actual same finiteN fixed-beta all-character physical state with Cbeta=<sum|z|^4>/V upper from bulk heat/entropy theorem, subject to its independent review.',
            'HS_convention': 'r_x^2=Tr sum Phi_mu^2=|z_x|^2/2',
            'n_boundary': '2N^2',
            'J_squared_upper': '(CJ^2/2)*(2N^2)^(3/2)*sqrt(N^3*Cbeta)',
            'contact_absolute_upper': '(CK/sqrt(2))*(2N^2)^(3/4)*(N^3*Cbeta)^(1/4)',
            'chi_upper': 'beta*J_squared_upper; global connected mean, complete degenerate blocks',
            'Fsecond_enclosure': '[-contact_absolute_upper-beta*J_squared_upper,+contact_absolute_upper]',
            'derivation': 'Pointwise four-norm Holder, then concavity/Jensen for actual thermal expectation; no independent-copy/selected-site assumption.',
            'J_squared_per_boundary_area_scaling': 'O(N^(5/2)) at fixed beta,Cbeta; no N-uniform IR/stiffness claim',
            'numeric_native_partition_or_response': False,
        },
        'adverse_concentration': [concentration_control(n) for n in request['N']],
        'new_pressure_sign_paid': False, 'boundary_to_periodic_state_bridge_paid': False,
        'fullphysical_ground_phase_stiffness_photon_admitted': False,
        'old_producer_solver_root_checker_rerun': False, 'campaign_complete': False,
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'members': result['original_archive_members'], 'coefficients': result['boundary_coefficients_exact'],
                      'native_response_sign': False, 'N_uniform_IR': False}))


if __name__ == '__main__':
    main()

"""Independent exact Schur, native incidence/current and exterior controls.

No original producer, eigensolver, ground backend or shell generator is imported.
Finite rational complex matrices test the operator identity, not the campaign.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
from typing import cast

import sympy as sp


def squared(value):
    return sp.simplify(sp.conjugate(value) * value)


def norm_squared(vector):
    return sp.simplify((vector.conjugate().T * vector)[0])


def resolvent_controls():
    i = sp.I
    a = sp.Matrix([[2, 1 + i], [1 - i, -1]])
    d = sp.Matrix([[3, 2 * i, 1], [-2 * i, -2, 1 + i], [1, 1 - i, 4]])
    b = sp.Matrix([[1, i], [1 - i, 2], [i, -1 + i]]) / 3
    h = a.row_join(b.conjugate().T).col_join(b.row_join(d))
    assert h == h.conjugate().T
    assert a * (b.conjugate().T * b) != (b.conjugate().T * b) * a
    assert d * (b * b.conjugate().T) != (b * b.conjugate().T) * d
    c2 = sum(squared(x) for x in b)
    u = sp.Matrix([1, i, 1 - i, 2, i]) / 3
    v = sp.Matrix([i, 1 + i, 2, -i, 1]) / 3
    cj2 = max(norm_squared(u), norm_squared(v))
    rows = []
    for eta in [sp.Rational(1, 3), sp.Integer(2), sp.Integer(7)]:
        center = sp.Rational(5, 7)
        rf = (h - center * sp.eye(5) + i * eta * sp.eye(5)).inv()
        ra = (a - center * sp.eye(2) + i * eta * sp.eye(2)).inv()
        rq = (d - center * sp.eye(3) + i * eta * sp.eye(3)).inv()
        sigma = b.conjugate().T * rq * b
        compressed = rf[:2, :2]
        delta = compressed - ra
        assert sp.simplify(compressed - (a - center * sp.eye(2) + i * eta * sp.eye(2) - sigma).inv()) == sp.zeros(2)
        assert sp.simplify(delta - ra * sigma * compressed) == sp.zeros(2)
        upper = c2 / eta**3
        slack = sp.simplify(upper**2 * sp.eye(2) - delta.conjugate().T * delta)
        assert slack[0, 0] >= 0 and slack[1, 1] >= 0 and slack.det() >= 0
        full_cross = (u.conjugate().T * rf * v)[0]
        local_cross = (u[:2, :].conjugate().T * ra * v[:2, :])[0]
        source_error = sp.simplify(full_cross - local_cross)
        leakage_upper = cj2 * (c2 / eta**3 + 3 / eta)
        assert squared(source_error) <= leakage_upper**2
        assert sp.im(full_cross) != 0
        wrong_shift = (a - (center + 1) * sp.eye(2) + i * eta * sp.eye(2)).inv()
        rows.append({'eta_exact': str(eta), 'Schur_identity_exact': True,
                     'noncommuting_complex_full_complement': True,
                     'operator_error_bound_exact': str(upper),
                     'PSD_bound_slack_principal_minors_exact': [str(slack[0, 0]), str(slack[1, 1]), str(sp.factor(slack.det()))],
                     'complex_full_cross_exact': str(sp.simplify(full_cross)),
                     'source_leakage_error_exact': str(source_error),
                     'source_leakage_upper_exact': str(leakage_upper),
                     'erased_imaginary_cross_error_exact': str(sp.im(full_cross)),
                     'wrong_center_changes_response': ra != wrong_shift})
    # An omitted PHP term is first-order and cannot be paid by B=0.
    k = sp.Matrix([[0, i], [-i, 0]])
    witness = (a + k + 2 * i * sp.eye(2)).inv() - (a + 2 * i * sp.eye(2)).inv()
    assert witness != sp.zeros(2)
    return {'C_B_squared_Frobenius_exact': str(c2), 'C_source_squared_exact': str(cj2),
            'frequencies': rows, 'omitted_PHP_B_zero_error_exact': str(witness),
            'omitted_PHP_wrong_zero_bound_refuted': True}


def native_controls(profiles, lattice_sizes):
    active, aux = profiles[1:]
    assert active['load'] == 1 and aux['load'] == 63 and active['paths'] == aux['paths']
    s, r = Fraction(1, 2), Fraction(1, 3)
    coefficients = {(p['axis'], p['length']): abs(s * Fraction(p['c0'])) + abs(r * Fraction(p['ci'])) for p in active['paths']}
    patch = {(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)}
    selected = [((0, 0, 0), 0), ((1, 0, 0), 1), ((0, 1, 0), 0), ((0, 0, 0), 1)]
    c_internal = 64 * 8 * 2 * sum(coefficients[(axis, 1)] for axis in (0, 1))
    c_boundary = 64 * 4 * 16 * sum(coefficients.values())
    c_total = c_internal + c_boundary
    rows = []
    for n in lattice_sizes:
        incident = []; currents = [0] * 4; endpoint_counts = {}
        only_starts = {}; internal_length2 = 0; distinct_flows = set()
        for axis, length in coefficients:
            count = start_count = 0
            for start in itertools.product(range(n), repeat=3):
                end = list(start); end[axis] = (end[axis] + length) % n; end = tuple(end)
                links = []
                for step in range(length):
                    x = list(start); x[axis] = (x[axis] + step) % n
                    links.append((tuple(x), axis))
                if start in patch: start_count += 1
                if start in patch or end in patch:
                    count += 1; incident.append((axis, length, start, end, links))
                    distinct_flows.add((start, end, tuple(links)))
                if length == 2 and start in patch and end in patch: internal_length2 += 1
                for edge, link in enumerate(selected):
                    currents[edge] += links.count(link)
            endpoint_counts[f'{axis}:{length}'] = count
            only_starts[f'{axis}:{length}'] = start_count
        assert internal_length2 == 0
        unique_incident_bound = 64 * 8 * sum(coefficients[(axis, length)] for axis, length, *_ in incident)
        assert unique_incident_bound <= c_boundary
        current_exact = [64 * 8 * (coefficients[(axis, 1)] + 2 * coefficients[(axis, 2)]) for _, axis in selected]
        assert currents == [3] * 4 and all(x <= c_boundary for x in current_exact)
        ap_seam_paths = sum(any(x[axis] == n - 1 for x, _ in links) for _, _, _, _, links in incident)
        assert ap_seam_paths > 0
        rows.append({'N': n, 'positive_path_incidence_counts': endpoint_counts,
                     'wrong_starts_only_counts': only_starts,
                     'positive_incident_paths': len(incident), 'adjoint_orientations_count': 2 * len(incident),
                     'unique_incident_coefficient_bound_exact': str(unique_incident_bound),
                     'length2_both_endpoints_inside_patch': internal_length2,
                     'distinct_directed_electric_flows': len(distinct_flows),
                     'AP_seam_incident_paths_preserved': ap_seam_paths,
                     'selected_current_path_multiplicities': currents,
                     'selected_current_norm_upper_exact': [str(x) for x in current_exact]})
    # Only two of four alpha entries have occupied negative sources in
    # each orientation; literal CAR below determines four targets per path.
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.diag(1, -1)
    sea_mask = sum(1 << mode for mode in (2, 3, 6, 7))
    alpha_counts = []
    for alpha in [sp.kronecker_product(sx, pauli) for pauli in (sx, sy, sz)]:
        images = {}
        for direction in (1, -1):
            for left, right in itertools.product(range(4), repeat=2):
                value = cast(sp.Matrix, alpha)[left, right]
                if value == 0:
                    continue
                target, source = (left, 4 + right) if direction == 1 else (4 + right, left)
                if not (sea_mask >> source) & 1 or (sea_mask >> target) & 1:
                    continue
                removed = sea_mask ^ (1 << source)
                sign = (-1)**((sea_mask & ((1 << source) - 1)).bit_count() + (removed & ((1 << target) - 1)).bit_count())
                key = (removed | (1 << target), direction)
                images[key] = images.get(key, 0) + sign * value
        gram_one = sum(squared(value) for value in images.values())
        assert len(images) == 4 and gram_one == 4
        alpha_counts.append({'targets': len(images), 'coefficient_free_Gram_exact': str(gram_one)})
    beta_offdiagonal = abs(r * Fraction(next(p['ci'] for p in active['paths'] if p['length'] == 1)))
    remote = []
    for n in lattice_sizes:
        paths = 3 * n**3 - 20
        gram = 64 * 4 * paths * beta_offdiagonal**2
        remote.append({'N': n, 'disjoint_positive_NN_paths': paths,
                       'remote_QHP_sea_norm_squared_exact': str(gram)})
    first_n = 4
    while 64 * 4 * (3 * first_n**3 - 20) * beta_offdiagonal**2 <= c_total**2:
        first_n += 2
    remote.append({'N': first_n, 'disjoint_positive_NN_paths': 3 * first_n**3 - 20,
                   'remote_QHP_sea_norm_squared_exact': str(64 * 4 * (3 * first_n**3 - 20) * beta_offdiagonal**2),
                   'exceeds_declared_Ctotal_squared_exact': True})
    # Literal CAR: occupied negative mode source, empty positive mode target.
    mask = 1 << 6
    target, source = 0, 6
    assert (mask >> source) & 1 and not ((mask >> target) & 1)
    after_annihilation = mask ^ (1 << source)
    car_sign = (-1)**((mask & ((1 << source) - 1)).bit_count() + (after_annihilation & ((1 << target) - 1)).bit_count())
    assert car_sign == 1
    # Alternate Gauss-compatible local compression permits cross-patch
    # transfer: one occupied negative-beta mode at site0 or site2.
    # Relative to reference occupation at site0, site2 has E=(-1,-1).
    # Both states have the same total particle number, but not patch charge8.
    c2_native = -s * Fraction(next(p['c0'] for p in active['paths'] if p['length'] == 2))
    c2_exact = sp.Rational(c2_native.numerator, c2_native.denominator)
    theta0, theta1 = sp.symbols('theta0 theta1', real=True)
    native_php = sp.Matrix([[0, c2_exact * sp.exp(sp.I * (theta0 + theta1))],
                            [c2_exact * sp.exp(-sp.I * (theta0 + theta1)), 0]])
    j0 = native_php.diff(theta0).subs({theta0: 0, theta1: 0})
    j1 = native_php.diff(theta1).subs({theta0: 0, theta1: 0})
    assert j0 == j1 and j0 != sp.zeros(2)
    assert native_php.subs({theta0: 0, theta1: 0})[0, 1] == c2_exact
    return {'C_internal_exact': str(c_internal), 'C_boundary_exact': str(c_boundary),
            'C_total_exact': str(c_total), 'C_total_squared_exact': str(c_total**2),
            'C_J_declared_exact': str(c_boundary), 'native_counts': rows,
            'remote_frozen_sea_controls': remote,
            'literal_negative_sea_alpha_counts_all_axes': alpha_counts,
            'alternate_embedding_length2_PHP_adverse': {
                'same_total_particles': 1, 'patch_charge8_embedding': False,
                'two_basis_masks': [1, 4], 'two_basis_E_vectors': [[0, 0], [-1, -1]],
                'non_diagonal_PHP_entry_exact': str(c2_exact),
                'each_crossed_link_current_matrix_exact': str(j0),
                'length2_current_derivative_multiplicity': 1,
                'B_alone_does_not_pay_changed_PHP': True},
            'remote_CAR_witness': {'before_mask': mask, 'after_mask': after_annihilation | (1 << target), 'sign': car_sign,
                                   'coefficient_modulus_exact': str(beta_offdiagonal), 'shared_flux_shift': 1},
            'remote_Gram_growth': '64*4*(3*N^3-20)*(r*ci_NN)^2',
            'conditional_cross_length2_PHP': 'Zero only when exact patch charge8/species is retained or exterior endpoints have fixed occupations. Outside modes are not automatically diagonal under an arbitrary P.',
            'fixed_sea_full_torus_local_Ctotal_refuted': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    root = args.request.parent
    for name, sha in request['pins'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
    profiles = json.loads((root / request['profiles']).read_text())['primitive_profiles']
    result = {'kind': 'INDEPENDENT_EXACT_CONDITIONAL_SCHUR_AND_NATIVE_CURRENT_REVIEW',
              'resolvent': resolvent_controls(),
              'native': native_controls(profiles, request['lattice_sizes']),
              'ground_backend_used': False, 'original_producer_imported_or_rerun': False,
              'exact_control_not_full_ground_or_phase': True,
              'scientific_admission': False, 'campaign_complete': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'exact_noncommuting_resolvent': 'executed', 'native_incidence_current': 'executed',
                      'fixed_exterior_sea_local_full_torus_bound': 'refuted', 'ground_or_phase_admission': False}))


if __name__ == '__main__':
    main()

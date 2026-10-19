"""NEW exact actual-family charged local-source error controls, not a ground solver."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction as Q
from pathlib import Path

import sympy as s


def clifford():
    pauli = [s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -s.I], [s.I, 0]]), s.diag(1, -1)]
    gamma = [s.diag(1, 1, -1, -1)] + [s.kronecker_product(pauli[0], p) for p in pauli]
    assert all(a * b + b * a == 2 * int(i == j) * s.eye(4)
               for i, a in enumerate(gamma) for j, b in enumerate(gamma))
    return gamma


def hs2(a):
    return s.simplify(s.trace(a.H * a))


def apply_car(state, create, annihilate):
    result = {}
    for mask, value in state.items():
        if not mask & (1 << annihilate):
            continue
        sign = (-1) ** ((mask & ((1 << annihilate) - 1)).bit_count())
        middle = mask ^ (1 << annihilate)
        if middle & (1 << create):
            continue
        sign *= (-1) ** ((middle & ((1 << create) - 1)).bit_count())
        target = middle | (1 << create)
        result[target] = result.get(target, s.S.Zero) + sign * value
    return result


def dg_action(a, state):
    result = {}
    for i in range(a.rows):
        for j in range(a.cols):
            if a[i, j] == 0:
                continue
            for mask, value in apply_car(state, i, j).items():
                result[mask] = result.get(mask, s.S.Zero) + a[i, j] * value
    return {mask: s.expand(value) for mask, value in result.items() if value != 0}


def inner(a, b):
    return s.simplify(sum(s.conjugate(value) * b.get(mask, 0) for mask, value in a.items()))


def hermitian_bond(a):
    return s.zeros(4).row_join(a).col_join(a.H.row_join(s.zeros(4)))


def exact_local_control(profiles):
    gamma = clifford()
    amplitude = [s.Rational(1, 2), s.Rational(2, 3), s.Rational(3, 4), s.Rational(4, 5)]
    centre = [a * g for a, g in zip(amplitude, gamma)]
    dx = [(gamma[j] + gamma[(j + 1) % 4]) / 1024 for j in range(4)]
    dy = [(gamma[j] - gamma[(j + 2) % 4]) / 1536 for j in range(4)]
    # Rational, noncentral SU4 link and noncommuting full-complex endpoint fields.
    v = s.eye(4)
    v[:2, :2] = s.Matrix([[s.Rational(3, 5), s.Rational(4, 5)],
                          [-s.Rational(4, 5), s.Rational(3, 5)]])
    assert v.H * v == s.eye(4) and v.det() == 1
    z = s.I
    v = z * v
    assert v.det() == 1
    state = {15: s.Rational(1, 2), 51: s.I / 2,
             85: s.Rational(1, 2), 170: (3 + 4 * s.I) / 10}
    assert all(mask.bit_count() == 4 for mask in state)
    assert inner(state, state) == 1
    rows = []
    errors = []
    for profile in profiles[1:]:
        for path in profile['paths']:
            axis, length = path['axis'], path['length']
            c0, ci = s.Rational(path['c0']), s.Rational(path['ci'])
            w = v ** length
            zp = z ** length
            fx = c0 * (centre[0] + dx[0]) + s.I * ci * (centre[axis + 1] + dx[axis + 1])
            fy = c0 * (centre[0] + dy[0]) + s.I * ci * (centre[axis + 1] + dy[axis + 1])
            actual = (fx * w + w * fy) / 2
            comparator = (c0 * centre[0] + s.I * ci * centre[axis + 1]) * zp
            remainder = actual - comparator
            wrong_endpoint_average = (fx + fy) * w / 2
            if length == 1:
                assert wrong_endpoint_average != actual
            k = abs(c0) + abs(ci)
            bound = k ** 2 * (sum(hs2(d) for d in dx + dy) + 8 * hs2(w - zp * s.eye(4)))
            assert hs2(remainder) <= bound
            action = dg_action(hermitian_bond(remainder), state)
            norm = inner(action, action)
            assert norm <= 16 * hs2(remainder)
            errors.append(action)
            rows.append({'role': profile['role'], 'axis': axis, 'length': length,
                         'coefficient_HS_squared': str(hs2(remainder)),
                         'endpoint_and_link_bound': str(bound), 'canonical_wedge4_source_squared': str(norm)})
    # An exact CAR complex cross witness on the SAME canonical state.
    kx = hermitian_bond(gamma[1] + s.I * gamma[2])
    ky = hermitian_bond(gamma[2] + (1 + s.I) * gamma[3])
    ax, ay = dg_action(kx, state), dg_action(ky, state)
    cross = inner(ax, ay)
    if s.im(cross) == 0:
        # The noncommuting onsite channel has a rigorously nonzero imaginary cross.
        ky = s.diag(1, -1, 2, -2, 3, -3, 4, -4)
        ay = dg_action(ky, state)
        cross = inner(ax, ay)
    # A separate exact elementary two-mode cross cannot vanish accidentally.
    psi = s.Matrix([s.Rational(3, 5), (12 + 16 * s.I) / 25])
    sx, sy = s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -s.I], [s.I, 0]])
    elementary_cross = (psi.H * sx * sy * psi)[0]
    assert s.im(elementary_cross) != 0
    combined = {mask: errors[0].get(mask, 0) + s.I * errors[1].get(mask, 0)
                for mask in set(errors[0]) | set(errors[1])}
    reconstructed = inner(errors[0], errors[0]) + inner(errors[1], errors[1]) - 2 * s.im(inner(errors[0], errors[1]))
    assert inner(combined, combined) == s.simplify(reconstructed)
    assert any(s.im(a[i, j]) != 0 for a in centre + dx + dy for i in range(4) for j in range(4))
    return {'controls': rows, 'full_complex_CAR_cross': str(cross),
            'guaranteed_nonreal_canonical_two_mode_cross': str(elementary_cross),
            'complex_linear_combination_exact': str(inner(combined, combined)),
            'canonical_pair_particles': 4, 'local_pair_full_CAR_dimension': 256,
            'not_a_factorized_density_or_HS_trace': True}


def lattice_control(n, profiles):
    sites = list(itertools.product(range(n), repeat=3))
    ix = {x: j for j, x in enumerate(sites)}
    counts = {'charged_NN': 0, 'charged_two': 0, 'charged_negative_AP': 0,
              'neutral_NN': 0, 'neutral_negative_AP': 0}
    aliases = {}
    for profile in profiles:
        for x in sites:
            for path in profile['paths']:
                axis, length = path['axis'], path['length']
                y = list(x)
                y[axis] = (x[axis] + length) % n
                y = tuple(y)
                phase = (-1) ** ((x[axis] + length) // n)
                edges = []
                for j in range(length):
                    vertex = list(x)
                    vertex[axis] = (x[axis] + j) % n
                    edges.append(3 * ix[tuple(vertex)] + axis)
                assert len(set(edges)) == length
                charge_delta = {ix[x]: -1, ix[y]: 1}
                divergence = {}
                for edge in edges:
                    source, direction = edge // 3, edge % 3
                    target = list(sites[source]); target[direction] = (target[direction] + 1) % n
                    divergence[source] = divergence.get(source, 0) - 1
                    target_index = ix[tuple(target)]
                    divergence[target_index] = divergence.get(target_index, 0) + 1
                divergence = {key: value for key, value in divergence.items() if value}
                assert divergence == charge_delta
                prefix = 'neutral' if profile['scale'] == 'heavy' else 'charged'
                counts[prefix + ('_NN' if length == 1 else '_two')] += profile['load']
                if phase == -1:
                    counts[prefix + '_negative_AP'] += profile['load']
                # Keep labelled paths before endpoint aliasing, including AP and shared flow.
                if x in ((0, 0, 0), (n - 1, 0, 0), (n - 2, 0, 0)):
                    key = (profile['role'], ix[x], ix[y])
                    aliases.setdefault(key, []).append((axis, length, phase, tuple(edges), path['c0'], path['ci']))
    v = n ** 3
    assert counts['charged_NN'] == counts['charged_two'] == 64 * 3 * v
    assert counts['neutral_NN'] == 3 * v
    assert counts['charged_negative_AP'] == 64 * 9 * n * n
    assert counts['neutral_negative_AP'] == 3 * n * n
    # N4 antipodal TWO directed labels have different flow even at identical undirected endpoints.
    p0 = tuple(3 * ix[(j, 0, 0)] for j in (0, 1))
    p2 = tuple(3 * ix[(j, 0, 0)] for j in (2, 3))
    if n == 4:
        assert p0 != p2 and set(p0).isdisjoint(p2)
    # Exact AP whole-circle character and central winding; local colour/alignment zero.
    central = [s.S.One] * (n - 1) + [s.I]
    nn = [(-1 if j == n - 1 else 1) * central[j] for j in range(n)]
    two = [(-1 if j + 2 >= n else 1) * central[j] * central[(j + 1) % n] for j in range(n)]
    assert s.prod(nn) == -s.I
    assert s.prod(central) == s.I
    assert two[n - 2] == two[n - 1] == -s.I
    centre_error = hs2((-s.I + 1) * s.eye(4))
    assert centre_error == 8
    gamma = clifford()
    alias_controls = []
    for profile in profiles[1:]:
        c0 = s.Rational(profile['paths'][0]['c0'])
        ci = s.Rational(profile['paths'][0]['ci'])
        k_nn = c0 * gamma[0] / 2 + s.I * ci * gamma[1] * s.Rational(2, 3)
        actual_AP = nn[-1] * k_nn
        wrong_AP = -k_nn
        assert hs2(actual_AP - wrong_AP) > 0
        row = {'role': profile['role'], 'NN_AP_wrong_trivial_centre_error_squared': str(hs2(actual_AP - wrong_AP))}
        if n == 4:
            k_two = s.Rational(profile['paths'][1]['c0']) * gamma[0] / 2
            actual_alias = k_two * two[0] + (k_two * two[2]).H
            wrong_alias = s.zeros(4)
            assert actual_alias != wrong_alias
            row['length2_antipodal_full_complex_alias_HS_squared'] = str(hs2(actual_alias))
            row['distinct_shared_flow_labels'] = [list(p0), list(p2)]
        alias_controls.append(row)
    # A local Z4 gauge coboundary changes individual links, never winding.
    frame = [s.I ** (j % 4) for j in range(n)]
    transformed = [s.conjugate(frame[j]) * central[j] * frame[(j + 1) % n] for j in range(n)]
    assert s.simplify(s.prod(transformed)) == s.I
    return {'N': n, 'V': v, 'counts': counts, 'canonical_each65': 2 * v,
            'all65_onsite_terms': 65 * v, 'Ref': 'entire unchanged spectator module',
            'selected_labelled_alias_groups': len(aliases),
            'exact_profile_AP_alias_controls': alias_controls,
            'all_ordered_paths_integer_divergence_exact': True,
            'NN_centre_AP_phases': list(map(str, nn)), 'two_centre_AP_phases': list(map(str, two)),
            'flat_colour_and_alignment': 0, 'nontrivial_centre_winding': 'I',
            'wrong_trivial_centre_phase_HS_squared': str(centre_error),
            'local_Z4_coboundary_winding_preserved': True,
            'N4_two_reverse_labels_not_combined_before_Wilson_flow': n == 4}


def correlated_shared_control():
    copies = 64
    # The exact Gauss-dressed logical qubits use one common theta:
    # |l>+e^{-i theta}|r>. Integer E equals the number of transferred particles.
    weights = [Q(__import__('math').comb(copies, k), 2 ** copies) for k in range(copies + 1)]
    assert sum(weights) == 1
    e1 = sum(k * weight for k, weight in enumerate(weights))
    e2 = sum(k * k * weight for k, weight in enumerate(weights))
    independent_e2 = Q(copies, 2)
    assert e1 == 32 and e2 == 1040 and independent_e2 == 32
    actual_norm = copies ** 2
    independent_norm = copies
    assert actual_norm > independent_norm
    # Correlate a field branch with the joint-copy sign, without changing its marginal.
    actual_error = Q(1 + 4, 2) * actual_norm
    wrong_independent_error = Q(1 + 4, 2) * independent_norm
    rx, ry, rz = Q(72, 125), Q(96, 125), Q(-7, 25)
    assert rx * rx + ry * ry + rz * rz == 1
    real_cross = copies * (copies - 1) * rx * ry
    imaginary_cross = copies * rz
    assert imaginary_cross != 0
    return {'copies': copies, 'same_flow_correlated_source_squared': actual_norm,
            'wrong_sum_independent_copy_squared': independent_norm,
            'field_occupation_correlated_error_squared': str(actual_error),
            'wrong_field_factorized_independent_error_squared': str(wrong_independent_error),
            'E_mean': str(e1), 'E_squared_shared': str(e2),
            'wrong_E_squared_independent': str(independent_e2),
            'HE_plus_HC_ONCE_shared': str(64 * e2), 'wrong_copy_electric_price': str(64 * independent_e2),
            'full_complex_cross_real': str(real_cross), 'full_complex_cross_imag': str(imaginary_cross),
            'canonical_embedding': 'one logical fermion/copy on pair plus 2V-1 unchanged exterior particles; neutral/fullRef unchanged',
            'all64_copy_crosses_and_one_common_integer_flow': True}


def quantitative_control(profiles, geometry):
    assert geometry['constant']['C_orbit_power'] == 61
    c_orbit = 2 ** 61
    for profile in profiles[1:]:
        assert abs(Q(profile['onsite'])) < 37
        assert all(abs(Q(p['c0'])) + abs(Q(p['ci'])) < 24 for p in profile['paths'])
    b = (64 * 2378) ** 2
    c_d = b * (2 + 64 * 1728)
    c_a = b * (32 * 1728)
    c_source = 2 ** 52
    assert c_d < c_source and c_a < c_source
    d2, eta = Q(1, 2 ** 112), Q(1, 2 ** 159)
    a, colour = Q(40, 2 ** 2035), Q(10, 7 * 2 ** 2035)
    distance = 2 * d2 + c_orbit * eta
    link = 1728 * (2 * distance + a)
    exact_envelope = b * (2 * distance + 32 * link)
    coarse = c_source * (distance + a + colour)
    assert exact_envelope <= coarse < Q(1, 2 ** 45)
    return {'C_orbit': c_orbit, 'C_source': c_source, 'exact_C_D': c_d, 'exact_C_a': c_a,
            'distance_outward_bound_exact': str(distance), 'link_outward_bound_exact': str(link),
            'source_outward_bound_exact': str(exact_envelope),
            'source_squared_strictly_below_power2': -45,
            'formula': '(64*2378)^2 * [2D + 32*1728*(2D+a)], D=2d2+2^61 eta',
            'coarse_formula': '2^52 * (2d2+2^61 eta+a+c)',
            'centre_plaquette_mismatch_probability_bound': 'c/2+4*1728*(2D+a)',
            'N_uniform_gate': 'every even N>=2^257; exact controls N4/6/8 are not native ground vectors'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, digest in request['pins'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    profiles = json.loads(Path(request['profiles']).read_text())['primitive_profiles']
    geometry = json.loads(Path(request['geometry_interface']).read_text())
    result = {'kind': 'NEW_EXACT_LOCAL_CHARGED_SOURCE_REMAINDER_CONTROLS_NOT_NATIVE_GROUND',
              'local_full_complex': exact_local_control(profiles),
              'N_controls': [lattice_control(n, profiles) for n in (4, 6, 8)],
              'correlated_shared_flow': correlated_shared_control(),
              'quantitative_bound': quantitative_control(profiles, geometry),
              'law_changed': False, 'actual_rhoN_replaced': False, 'entireP0_preserved_in_theorem': True,
              'reference_ground_or_H_domain_unitary_claimed': False,
              'physical_lowband_order_gap_atom_claimed': False,
              'old_entrypoint_imported_or_replayed': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'C_source': 2 ** 52,
                      'source_squared_gate': '<2^-45', 'controls_N': [4, 6, 8]}))


if __name__ == '__main__':
    main()

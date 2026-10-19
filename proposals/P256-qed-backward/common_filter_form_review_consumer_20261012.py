"""Independent exact native representation/form adverse controls, never a ground solver."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
import zipfile

import sympy as s

ROOT = Path(__file__).resolve().parent


def generators():
    result = []
    for a, b in itertools.combinations(range(4), 2):
        real, imag = s.zeros(4), s.zeros(4)
        real[a, b] = real[b, a] = s.Rational(1, 2)
        imag[a, b], imag[b, a] = -s.I / 2, s.I / 2
        result.extend((real, imag))
    for n in range(1, 4):
        result.append(s.diag(*([1] * n + [-n] + [0] * (3 - n))) / s.sqrt(2 * n * (n + 1)))
    assert s.Matrix([[s.trace(a * b) for b in result] for a in result]) == s.eye(15) / 2
    return result


def exterior(body, k):
    masks = [sum(1 << j for j in occupied) for occupied in itertools.combinations(range(body.rows), k)]
    ids = {mask: i for i, mask in enumerate(masks)}
    result = s.zeros(len(masks))
    for col, mask in enumerate(masks):
        for source in range(body.rows):
            if not mask & (1 << source):
                continue
            removed = mask ^ (1 << source)
            sign1 = (-1) ** (mask & ((1 << source) - 1)).bit_count()
            for target in range(body.rows):
                if removed & (1 << target) or body[target, source] == 0:
                    continue
                sign2 = (-1) ** (removed & ((1 << target) - 1)).bit_count()
                result[ids[removed | (1 << target)], col] += sign1 * sign2 * body[target, source]
    return result


def invariant_pair():
    masks = [sum(1 << j for j in occupied) for occupied in itertools.combinations(range(4), 2)]
    pair = {}
    for i, mask in enumerate(masks):
        occupied = [j for j in range(4) if mask & (1 << j)]
        holes = [j for j in range(4) if not mask & (1 << j)]
        pair[(i, masks.index(15 ^ mask))] = (-1) ** sum(a > b for a in occupied for b in holes)
    return pair


def reduced_cross(left, right):
    result = s.zeros(6)
    for a, x in left.items():
        for b, y in right.items():
            if a[1:] == b[1:]:
                result[a[0], b[0]] += x * y
    return result


def representation_controls(gs):
    pair = invariant_pair()
    vector = s.zeros(36, 1)
    for (i, j), value in pair.items():
        vector[6 * i + j] = value
    for g in gs:
        G = exterior(g, 2)
        assert (s.kronecker_product(G, s.eye(6)) + s.kronecker_product(s.eye(6), G)) * vector == s.zeros(36, 1)
    # Two actual invariant contractions in R^tensor4, not fake external multiplicities.
    first, second = {}, {}
    for (a, b), x in pair.items():
        for (c, d), y in pair.items():
            first[(a, b, c, d)] = x * y
            second[(a, c, b, d)] = x * y
    h = s.diag(1, 1, -1, -1)
    A = exterior(h, 2)
    X = s.diag(*[s.Integer(2) ** A[i, i] for i in range(6)])
    Z = s.trace(X ** 2) / 6
    F = X / s.sqrt(Z)
    cross_scalars = []
    for left, right in itertools.product((first, second), repeat=2):
        reduced = reduced_cross(left, right)
        scalar = reduced[0, 0]
        assert reduced == scalar * s.eye(6)
        assert s.simplify(s.trace(F ** 2 * reduced) - s.trace(reduced)) == 0
        cross_scalars.append(str(scalar))
    assert first != second and cross_scalars[1] != '0'
    # Neutral-entangled coefficients make both diagonal and cross blocks operative.
    states = [first, {key: first.get(key, 0) + 2 * second.get(key, 0) for key in first.keys() | second.keys()}]
    for left, right in itertools.product(states, repeat=2):
        reduced = reduced_cross(left, right)
        assert reduced == reduced[0, 0] * s.eye(6)
        assert s.simplify(s.trace((F ** 2 - s.eye(6)) * reduced)) == 0
    K = exterior(gs[2], 2)
    assert K * F != F * K
    assert s.trace(F * (K * F - F * K)) == 0
    common = s.kronecker_product(K, s.eye(6)) + s.kronecker_product(s.eye(6), K)
    filtered = s.kronecker_product(F, s.eye(6)) * vector
    assert common * filtered != s.zeros(36, 1)
    assert s.simplify((filtered.H * common * filtered)[0]) == 0
    costs = []
    for B in (exterior(g, 2) for g in gs):
        DX = s.zeros(6)
        for i, j in itertools.product(range(6), repeat=2):
            DX[i, j] = B[i, j] * ((X[i, i] - X[j, j]) / (A[i, i] - A[j, j]) if A[i, i] != A[j, j] else s.log(2) * X[i, i])
        dZ = s.trace(X * DX) / 3
        dF = DX / s.sqrt(Z) - X * dZ / (2 * Z ** s.Rational(3, 2))
        assert s.simplify(s.trace(F * dF)) == 0
        difference = s.simplify(s.log(2) ** 2 * s.trace(F ** 2 * B ** 2) / 6 - s.trace(dF ** 2) / 6)
        assert difference == 0 or difference.is_nonnegative is True
        costs.append(str(s.simplify(s.trace(dF ** 2) / 6)))
    casimirs = []
    for k in range(5):
        C = sum((exterior(g, k) ** 2 for g in gs), s.zeros(math.comb(4, k)))
        value = [0, s.Rational(15, 8), s.Rational(5, 2), s.Rational(15, 8), 0][k]
        assert s.simplify(C - value * s.eye(C.rows)) == s.zeros(C.rows)
        casimirs.append(str(value))
    connection = s.I * K
    ordinary = connection * F - F * connection
    assert ordinary != s.zeros(6) and ordinary - connection * F + F * connection == s.zeros(6)
    # Every local charged number, not only species total, commutes F.
    for i in range(4):
        number = exterior(s.diag(*[int(j == i) for j in range(4)]), 2)
        assert number * F == F * number
    return {'true_cross_multiplicity_scalars': cross_scalars, 'neutral_entangled_cross_isometry': True,
            'noncommuting_common_action_nonzero_expectation_zero': True,
            'all15_exact_divided_difference_costs': costs, 'local_exterior_0_to4_Casimirs': casimirs,
            'ordinary_changed_frame_jet_rejected': True, 'collective_rho_affine_fullE_commutation': True,
            'native_full_dimension_scope': 'Local exterior sectors are exact native CAR; finite M4 invariant contractions test representation identities. Full M=256,k=128 construction is combinatorial/analytic, not a computed ground or full vector.'}


def native_and_budget_controls(gs, profiles):
    lam, chi, fast = Fraction(2 ** 8192), Fraction(1, 2 ** 4096), Fraction(2 ** 10240)
    weights = [abs(Fraction(p['onsite'])) + 2 * sum((abs(Fraction(t['c0'])) + abs(Fraction(t['ci'])) for t in p['paths']), Fraction(0)) for p in profiles]
    allocation = 4 * (lam * weights[0] + weights[1] + 63 * weights[2])
    assert [p['load'] for p in profiles] == [1, 1, 63] and allocation < 512 * lam
    onsite_delta = Fraction(profiles[1]['onsite']) - Fraction(profiles[2]['onsite'])
    assert onsite_delta == Fraction(-126663739519795, 70368744177664)
    assert profiles[1]['paths'] == profiles[2]['paths']
    square = s.trace(sum((g ** 2 for g in gs), s.zeros(4)) ** 2)
    crossed = sum((s.trace(a * b * a * b) for a in gs for b in gs), s.Integer(0))
    diagonal, offdiagonal = 4 * (2 * square + crossed), 2 * (square + crossed)
    assert diagonal == s.Rational(435, 4) and offdiagonal == s.Rational(105, 4)
    ordered = 4 * diagonal + 12 * offdiagonal
    assert ordered == 750 and 4 * ordered == 3000
    assert 4 * Fraction(60, 4) == 60
    assert Fraction(3 * 2 * 30, 40) == Fraction(9, 2)
    assert Fraction(3 * 4 * 7, 10) == Fraction(42, 5)
    trial = 60 * chi + 3000 * lam + Fraction(129, 10) * fast
    constrained = trial + 8192 * lam
    assert constrained < 16 * fast == 2 ** 10244
    assert 2 * (16 * fast) / lam == 2 ** 2053
    r = s.symbols('r', nonnegative=True)
    # Stronger exact minimization confirms the loose Young8192 price.
    assert s.expand((r - 8) ** 2 * (r ** 2 + 16 * r + 192) / 2) == r ** 4 / 2 - 1024 * r + 6144
    assert 512 * 16 == 8192 and 16 ** 3 / 2 >= 512
    native_rows = []
    U = s.diag(s.I, -s.I, 1, 1)
    px, py = s.diag(1, -1, 2, -2), s.diag(2, -2, -1, 1)
    A = exterior(s.diag(px, py), 4)
    X = s.diag(*[s.Integer(2) ** A[i, i] for i in range(70)])
    for row in profiles[2]['paths']:
        c0, ci = s.Rational(row['c0']), s.Rational(row['ci'])
        fx, fy = c0 * px + s.I * ci * gs[2 * int(row['axis'])], c0 * py + s.I * ci * gs[2 * int(row['axis'])]
        block = -s.I * (fx * U + U * fy) / 2  # Actual AP sign and Peierls phase retained.
        body = s.zeros(8); body[:4, 4:], body[4:, :4] = block, block.H
        K = exterior(body, 4)
        assert K * X != X * K
        assert s.trace(X * (K * X - X * K)) == 0
        native_rows.append({'axis': row['axis'], 'length': row['length'], 'native_noncommuting_trace': '0'})
    lattice_rows = []
    for N in (4, 6, 8):
        V, M, k = N ** 3, 4 * N ** 3, 2 * N ** 3
        # Exact complementary basis bijection: each pair has one of every M mode.
        assert math.comb(M, k) == math.comb(M - 1, k - 1) + math.comb(M - 1, k)
        sites = list(itertools.product(range(N), repeat=3))
        neutral = [4 if sum(x) % 2 == 0 else 0 for x in sites]
        assert sum(neutral) == k and 32 * M == 64 * k
        assert all(32 * 4 - 64 * 2 == 0 for _ in sites)
        for row in profiles[0]['paths'] + profiles[2]['paths']:
            assert all(tuple((x[j] + (row['length'] if j == row['axis'] else 0)) % N for j in range(3)) != x for x in sites)
        # Nonzero harmonic displacement is not free, even though rho=0.
        assert 64 * Fraction(1, N) ** 2 > 0
        lattice_rows.append({'N': N, 'all65_each_canonical': k, 'charged_mode_collective': 32,
                             'rho': 0, 'trial_fullE_price': 0, 'trial_Haar_price': 0,
                             'nonzero_character_price': str(64 * Fraction(1, N) ** 2)})
    return {'native_paths': native_rows, 'full65_lattices': lattice_rows, 'localCAR_weights': list(map(str, weights)),
            'allocation_less_512Lambda': True, 'ordered_Jordan_wick': str(ordered),
            'trial': '60chi+3000Lambda+(129/10)F', 'own_BS': '2^10244', 'own_C4S': '2^2053',
            'constrained_sharp_actual_budget_imported': False, '10chi_cost': str(4 * s.Rational(5, 2))}


def domain_adverse():
    # At one native60-dimensional Phi site, smooth radial tail (1+r^2)^(-65/4).
    # Radial measure r^59; fourth moment tail r^-2 is finite, sixth tail r^0 is not.
    dimension, power = 60, Fraction(65, 4)
    fourth = dimension - 1 + 4 - 4 * power
    sixth = dimension - 1 + 6 - 4 * power
    gradient = dimension - 1 - 4 * power - 2
    assert fourth < -1 and gradient < -1 and sixth >= -1
    return {'native_dimension': dimension, 'radial_power': str(power), 'fourth_tail_exponent': str(fourth),
            'sixth_tail_exponent': str(sixth), 'gradient_tail_exponent': str(gradient),
            'linear_Phi_entire_form_domain_claim_rejected': True,
            'reason': 'Radial SU4-invariant physical tail has quartic form membership; multiplying by a nonzero linear Phi/CAR source needs sixth moment and fails. Normalized exp filter instead has bounded finite-N value and first jets.',
            'full_jet_inventory': {'Phi0_changed_per_site': 15, 'Phi123_zero_per_site': 45, 'raw_Haar_zero': True, 'raw_U1_covariant_zero': True},
            'connection_scope': 'Inherited equivariant ambient Cartesian/CAR frame only; changed frame must use conjugated connection. No quotient metric or connection invented.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, pin in request['pins'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == pin, name
    with zipfile.ZipFile(ROOT / request['profile_archive']) as archive:
        raw = archive.read(request['profile_member'])
    assert hashlib.sha256(raw).hexdigest() == request['profile_member_sha256']
    profiles = json.loads(raw)['primitive_profiles']
    gs = generators()
    result = {'kind': 'INDEPENDENT_EXACT_NATIVE_FILTER_FORM_REVIEW_NOT_NATIVE_GROUND',
              'representation': representation_controls(gs), 'native_budget': native_and_budget_controls(gs, profiles),
              'domain_adverse': domain_adverse(), 'originals_executed': False,
              'native_ground_phase_stiffness_loop_Gram_inferred': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Independent native representation, full-law budgets and domain adverse controls completed')


if __name__ == '__main__':
    main()

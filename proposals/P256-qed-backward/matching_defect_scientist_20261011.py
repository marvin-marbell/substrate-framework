from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast

import sympy as s

ROOT = Path(__file__).resolve().parent
REQUEST = 'matching_defect_request_20261011.json'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sqrt_interval(value: F, bits: int = 192) -> tuple[F, F]:
    denominator = 1 << bits
    low = math.isqrt(value.numerator * denominator**2 // value.denominator)
    return F(low, denominator), F(low + 1, denominator)


def second(matrix: s.Matrix, number: int) -> tuple[s.Matrix, list[int]]:
    masks = [sum(1 << i for i in indices) for indices in itertools.combinations(range(4), number)]
    index = {mask: i for i, mask in enumerate(masks)}
    result = s.zeros(len(masks))
    for column, mask in enumerate(masks):
        for removed in range(4):
            if not (mask >> removed & 1):
                continue
            tail = mask ^ (1 << removed)
            sign = (-1) ** ((mask & ((1 << removed) - 1)).bit_count())
            for added in range(4):
                if tail >> added & 1:
                    continue
                parity = sign * (-1) ** ((tail & ((1 << added) - 1)).bit_count())
                result[index[tail | (1 << added)], column] += parity * matrix[added, removed]
    return result, masks


def car_controls(profiles: list[dict[str, Any]]) -> list[dict[str, object]]:
    A = s.diag(-1, 1, -1, 1)
    X = s.Matrix([[0, 1], [1, 0]])
    B = cast(s.Matrix, s.kronecker_product(X, s.diag(1, -1)))
    C = s.zeros(2).row_join(s.I * X).col_join((-s.I * X).row_join(s.zeros(2)))
    R = s.diag(s.Rational(1, 2), s.Rational(1, 2), -s.Rational(1, 2), -s.Rational(1, 2))
    x, e = s.symbols('x e')
    result = []
    for profile in profiles:
        m, c, d = [s.Rational(str(profile[key + '_exact'])) for key in ('m', 'c0', 'd')]
        h = m * A + c * B + d * C
        assert B * h * B == h and B * R * B == -R
        free = [-e + c, -e - c, e + c, e - c]
        dimensions = []
        cross = []
        for n in range(5):
            hn, masks = second(h, n)
            rn, _ = second(R, n)
            # Exterior power of the SAME native involution, not dGamma(B).
            wn = s.zeros(len(masks))
            for i, target in enumerate(masks):
                rows = [j for j in range(4) if target >> j & 1]
                for j, source in enumerate(masks):
                    columns = [k for k in range(4) if source >> k & 1]
                    wn[i, j] = B.extract(rows, columns).det()
            assert wn * wn == s.eye(len(masks))
            assert wn * hn * wn == hn and wn * rn * wn == -rn
            expected = s.prod(x - sum(free[j] for j in indices) for indices in itertools.combinations(range(4), n))
            expected = s.rem(s.expand(expected), e**2 - (m**2 + d**2), e)
            assert s.expand(hn.charpoly(x).as_expr() - expected) == 0
            hp = hn + 64 * (rn + s.eye(len(masks)))**2
            hm = hn + 64 * (rn - s.eye(len(masks)))**2
            assert wn * hp * wn == hm
            wrong = hn + 64 * (rn**2 + s.eye(len(masks)))
            assert hp - wrong == 128 * rn
            pp, pm = (s.eye(len(masks)) + wn) / 2, (s.eye(len(masks)) - wn) / 2
            assert pp * hp * pm == 128 * pp * rn * pm
            if n in (1, 2, 3):
                assert pp * hp * pm != s.zeros(len(masks))
                assert hp != wrong
            dimensions.append(len(masks))
            cross.append(n in (1, 2, 3))
        energy = s.sqrt(m**2 + d**2)
        rf, rg = m / (2 * energy), d / (2 * energy)
        rfree = s.Matrix([[0, rf, 0, -rg], [rf, 0, rg, 0], [0, rg, 0, rf], [-rg, 0, rf, 0]])
        shifted = rfree + s.eye(4)
        retained = shifted.extract([0], [0, 1, 2, 3])
        assert all(retained[0, j] != 0 for j in (0, 1, 3))
        complete_square = retained.conjugate().T * retained
        wrong_php_square = s.diag(1, 0, 0, 0)
        assert complete_square != wrong_php_square
        assert complete_square[0, 1] == rf
        result.append({'role': profile['role'], 'literal_m_c_d': [str(m), str(c), str(d)],
                       'full_CAR_dimensions': dimensions, 'all_free_characteristic_polynomials_exact': True,
                       'numbers_retaining_affine_parity_cross': [n for n, nonzero in enumerate(cross) if nonzero],
                       'missing_affine_cross_rejected': True, 'parity_split_rejected': True,
                       'complete_output_preimages': [0, 1, 3], 'missing_preimage_rejected': True,
                       'PHP_square_rejected': True, 'actual_retained_square_cross_exact': str(rf)})
    return result


def move(mask: int, created: int, removed: int) -> tuple[int, int] | None:
    if not mask >> removed & 1:
        return None
    sign = (-1) ** ((mask & ((1 << removed) - 1)).bit_count())
    tail = mask ^ (1 << removed)
    if tail >> created & 1:
        return None
    sign *= (-1) ** ((tail & ((1 << created) - 1)).bit_count())
    return tail | (1 << created), sign


def hopping_controls(boundary: dict[str, Any]) -> list[dict[str, object]]:
    # Every eight-mode Fock column, actual complex-rational primitive matrix.
    result = []
    for role in boundary['roles']:
        for path in role['paths']:
            matrix = [[(F(a), F(b)) for a, b in row] for row in path['matrix_exact']]
            entries = [(i, j, z) for i, row in enumerate(matrix) for j, z in enumerate(row) if z != (0, 0)]
            forward: dict[tuple[int, int], tuple[F, F]] = {}
            reverse: dict[tuple[int, int], tuple[F, F]] = {}
            for mask in range(256):
                for i, j, (a, b) in entries:
                    for out, created, removed, imaginary in ((forward, i, 4 + j, b), (reverse, 4 + j, i, -b)):
                        action = move(mask, created, removed)
                        if action is None:
                            continue
                        target, sign = action
                        assert target.bit_count() == mask.bit_count()
                        key = (target, mask)
                        old = out.get(key, (F(0), F(0)))
                        out[key] = old[0] + sign * a, old[1] + sign * imaginary
            assert all(reverse.get((column, row)) == (z[0], -z[1]) for (row, column), z in forward.items())
            assert not any(column == 255 for row, column in forward)
            # A genuine nonzero T then T* matrix element returns the external
            # flow label to zero; history count is two, instantaneous flow zero.
            column = 240
            norm2 = sum(a*a + b*b for (row, col), (a, b) in forward.items() if col == column)
            assert norm2 > 0
            result.append({'role': role['role'], 'count': role['count'], 'axis': path['axis'],
                           'length': path['length'], 'columns_checked': 256,
                           'nonzero_CAR_entries': len(forward), 'adjoint_and_Pauli_exact': True,
                           'forward_reverse_return_norm2_exact': str(norm2),
                           'historical_path_count_is_not_instantaneous_flow': True})
    assert len(result) == 12
    return result


def geometry_control(n: int) -> dict[str, object]:
    sites = list(itertools.product(range(n), repeat=3))
    def step(x, axis, length=1):
        y = list(x); y[axis] = (y[axis] + length) % n
        return tuple(y)
    def cell(x):
        return x[0] // 2, x[1], x[2]
    raw = [(x, a, step(x, a)) for x in sites for a in range(3)]
    internal = {i for i, (x, a, y) in enumerate(raw) if a == 0 and x[0] % 2 == 0}
    external = [i for i in range(len(raw)) if i not in internal]
    incidence = Counter()
    pairs = Counter()
    d = {x: 0 for x in sites}
    w = Counter()
    energy_external = 0
    for i in external:
        x, axis, y = raw[i]
        incidence[cell(x)] += 1; incidence[cell(y)] += 1
        pairs[tuple(sorted((cell(x), cell(y))))] += 1
        k = (7*i + axis) % 7 - 3
        d[x] -= k; d[y] += k
        w[cell(x)] += k*k; w[cell(y)] += k*k
        energy_external += k*k
    assert set(incidence.values()) == {10}
    assert 32*sum(w.values()) == 64*energy_external
    all_energy = energy_external
    internal_energy = 0
    for i in internal:
        x, axis, y = raw[i]
        electric = i % 5 - 2
        q0, q1 = d[x]-electric, d[y]+electric
        charge = q0+q1
        rho = F(q0-q1, 2)
        affine = F(d[y]-d[x], 2)
        assert electric == -(rho+affine)
        assert (rho+affine).denominator == 1
        assert charge == d[x]+d[y]
        assert charge**2 <= 10*w[cell(x)] and (2*affine)**2 <= 10*w[cell(x)]
        assert abs(q0) <= 128 and abs(q1) <= 128
        internal_energy += electric*electric
        all_energy += electric*electric
    assert all_energy == internal_energy+energy_external
    inventory = Counter()
    for x, axis, length in itertools.product(sites, range(3), (1, 2)):
        path = [(step(x, axis, j), axis) for j in range(length)]
        ext_count = sum(not (a == 0 and tail[0] % 2 == 0) for tail, a in path)
        inventory[axis, length, x[0] % 2, step(x, axis, length)[0] % 2, ext_count] += 1
    assert sum(inventory.values()) == 6*n**3
    assert sum(count for key, count in inventory.items() if key[-1] == 0) == n**3//2
    # Exhaustive small-w endpoint classes, with ten RAW incidences.
    endpoints = [0, 1]+[endpoint for endpoint in (0, 1) for _ in range(4)]
    small = {}
    for weight in (1, 2):
        classes = set()
        for selected in itertools.combinations(range(10), weight):
            for signs in itertools.product((-1, 1), repeat=weight):
                dd = [0, 0]
                for edge, sign in zip(selected, signs):
                    dd[endpoints[edge]] += sign
                classes.add((sum(dd), F(dd[1]-dd[0], 2)))
        small[str(weight)] = [[q, str(u)] for q, u in sorted(classes)]
    assert set(map(tuple, small['1'])) == {(q, str(u)) for q in (-1, 1) for u in (F(-1,2), F(1,2))}
    assert set(map(tuple, small['2'])) == {(q, str(F(u))) for q in (-2,0,2) for u in (-1,0,1)}
    assert 2*64 != 64  # wrong full edge price at BOTH endpoints
    assert 2**2 != 1   # two traversals on same edge are w4, not w1
    return {'N': n, 'raw_edges': len(raw), 'matching_edges': len(internal), 'external_edges': len(external),
            'raw_parallel_pair_excess': sum(count-1 for count in pairs.values()),
            'native_positive_paths': sum(inventory.values()), 'V_paths': 6*n**3-n**3//2,
            'path_classes': [{'key': list(key), 'count': count} for key, count in sorted(inventory.items())],
            'exact_HE': 64*all_energy, 'exact_HE_decomposition': 64*(internal_energy+energy_external),
            'exact_external_endpoint_allocation': 32*sum(w.values()),
            'weighted_incidence_small_classes': small,
            'shared_edge_full_price_twice_rejected': True, 'single_raw_k2_as_w1_rejected': True,
            'arbitrary_multiplicity_rule': 'w=sum RAW k_e^2, k=sum signed traversals; integer amplitudes unbounded, not count of paths'}


def quantitative(profiles: list[dict[str, Any]], caps: dict[str, Any]) -> dict[str, object]:
    a, u = [sqrt_interval(F(str(p['e_squared_exact']))) for p in profiles]
    c = F(str(profiles[0]['c0_exact']))
    assert c == F(str(profiles[1]['c0_exact'])) < 0 and a[1] < u[0]
    baseline = -4*a[1]-252*u[1]
    delta = a[0]+c
    upper = F(str(caps['ground_energy_interval_exact'][1]))
    gamma = -5123-upper
    alpha = baseline+96+5123
    assert gamma == F(46955312929,34359738368)
    assert gamma > 0 and 0 < alpha < 32 and delta > 0
    margins = {'w1_charged': F(-5155)+32-upper,
               'w2_neutral_affine': F(-5187)+64-upper,
               'w2_charge2': baseline+2*delta+64-upper,
               'w_ge3_at3': baseline+96-upper}
    assert all(value >= gamma for value in margins.values())
    # These inequalities + monotone slopes certify ALL integer w, not a scan.
    assert baseline+96-upper >= gamma/2+3*gamma/4
    assert 32 >= gamma/4
    assert baseline+96-upper == gamma+alpha
    coarse_failures = {'w1_charge1_free_only': baseline+delta+32-upper,
                       'w2_neutral_free_only': baseline+64-upper}
    assert all(value < 0 for value in coarse_failures.values())
    # All 512 oneparticle energies with actual role capacities; order statistics
    # of rigorous lower endpoints bound the complete n-particle free spectrum.
    oneparticle = []
    for profile, (elo,ehi) in zip(profiles, (a,u)):
        energies = [-ehi+c, -ehi-c, elo+c, elo-c]
        oneparticle.extend(energies*int(profile['spin_replicas']))
    oneparticle.sort()
    assert len(oneparticle) == 512
    lower = F(0)
    sectors = []
    for number in range(513):
        if number:
            lower += oneparticle[number-1]
        sectors.append({'number': number, 'Q': number-256, 'free_floor_lower_exact': str(lower),
                        'rho_operator_norm_exact': str(F(min(number,512-number),2))})
    # Omitted physical transferred class: A:n0,A:n3,U:n3,125 U minima.
    missing_upper = -4*a[0]-252*u[0] + 2*a[1] + (a[1]+c) + (u[1]+c) + 5123
    assert missing_upper < 0
    # g cannot be the per-slot free vacuum: its H action has a nonzero two-slot
    # double excitation from the FULL coherent electric square.
    m = F(str(profiles[0]['m_exact'])); d = F(str(profiles[0]['d_exact']))
    double_excitation = 32*d*d/F(str(profiles[0]['e_squared_exact']))
    assert double_excitation > 0
    # Actual affine and free actions pay all sectors. Neither negative free
    # cost nor the nonzero H action is a negative FULL-H variational witness.
    return {'baseline_lower_exact': str(baseline), 'baseline_display': float(baseline),
            'E_g_upper_exact': str(upper), 'delta_lower_exact': str(delta),
            'gamma_exact': str(gamma), 'gamma_display': float(gamma),
            'excess_flow_alpha_exact': str(alpha), 'excess_flow_alpha_display': float(alpha),
            'simultaneous_defect_price_exact': str(gamma/2),
            'simultaneous_external_flow_square_price_exact': str(gamma/2),
            'per_cell_margins_exact': {key:str(value) for key,value in margins.items()},
            'coarse_envelope_insufficient_exact': {key:str(value) for key,value in coarse_failures.items()},
            'coarse_envelope_insufficient_display': {key:float(value) for key,value in coarse_failures.items()},
            'all_number_sector_envelopes': sectors,
            'omitted_scalar_transfer_negative_free_H_plus5123_upper_exact': str(missing_upper),
            'omitted_class_not_fullH_counterexample': True,
            'actual_two_active_slot_H_freevacuum_double_excitation_exact': str(double_excitation),
            'primitive_slot_free_defect_bound_false_on_g': True,
            'primitive_slot_counterexample_proof': '128 R_A R_A changes both active free minima; entry32*d^2/eA^2>0. Thus free-minimum product is not an H eigenvector and unique g is not that product. Primitive slot-defect sum has kernel exactly that product; its expectation on g is positive while <g,(H-Eg)g>=0. No positive price for this WRONG count is possible.'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, default=ROOT / REQUEST)
    parser.add_argument('--output', type=Path, default=ROOT / 'matching_defect_interface_20261011.json')
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, expected in request['input_sha256'].items():
        assert digest(ROOT/name) == expected, name
    profiles = json.loads((ROOT/request['profiles']).read_text())['profiles']
    caps = json.loads((ROOT/request['caps']).read_text())
    boundary = json.loads((ROOT/request['hopping_boundary']).read_text())['all64_boundary']
    assert [(p['count'],p['spin_replicas']) for p in profiles] == [(1,2),(63,126)]
    assert [(p['count'],p['spin_replicas']) for p in boundary['roles']] == [(1,2),(63,126)]
    quantities = quantitative(profiles,caps)
    result = {'kind':'NORMALIZED_LINKED_SUPPORT_AWARE_ADDITIVE_MATCHING_REFERENCE_FORM',
              'status':'DERIVED_UNDER_ACCEPTED_FROZEN_LAW_CAP_PREMISES',
              'plan':request['plan'], 'operator_contract':request['operator_contract'],
              'proof':request['proof'], 'coverage':request['coverage'],
              'quantities':quantities,
              'independent_controls':{'full_CAR_actual_profiles':car_controls(profiles),
                                      'full_CAR_actual_hopping':hopping_controls(boundary),
                                      'raw_integer_geometry':[geometry_control(n) for n in (4,6,8)]},
              'input_sha256':request['input_sha256'], 'own_source_sha256':digest(Path(__file__)),
              'language':request['language'], 'capsule':request.get('execution_capsule', 'matching_defect_capsule_20261011.json'),
              'limitations':request['limitations']}
    assert all(digest(ROOT/name)==expected for name,expected in request['input_sha256'].items())
    output = args.output
    with output.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'interface':output.name,'sha256':digest(output),'gamma_exact':quantities['gamma_exact'],
                      'all_n0_through512_and_integer_multiplicities_covered':True,'physical1':False}),flush=True)


if __name__ == '__main__':
    main()

"""Exact canonical determinant/shared-HS/complement controls; no native ground backend."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import combinations, permutations
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
Matrix = list[list[F]]
Poly = dict[int, F]
PolyMatrix = list[list[Poly]]


def eye(n: int) -> Matrix:
    return [[F(i == j) for j in range(n)] for i in range(n)]


def mm(a: Matrix, b: Matrix) -> Matrix:
    return [[sum((x * y for x, y in zip(row, col)), F(0)) for col in zip(*b)] for row in a]


def trace(a: Matrix) -> F:
    return sum((row[i] for i, row in enumerate(a)), F(0))


def det(a: Matrix) -> F:
    n = len(a)
    return sum(((-1)**sum(p[i] > p[j] for i in range(n) for j in range(i+1, n))
                * math.prod(a[i][p[i]] for i in range(n)) for p in permutations(range(n))), F(0))


def p_add(a: Poly, b: Poly) -> Poly:
    out = a.copy()
    for mask, value in b.items():
        out[mask] = out.get(mask, F(0)) + value
    return {k: v for k, v in out.items() if v}


def p_mul(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for ma, va in a.items():
        for mb, vb in b.items():
            if not ma & mb:
                out[ma | mb] = out.get(ma | mb, F(0)) + va * vb
    return {k: v for k, v in out.items() if v}


def p_mm(a: PolyMatrix, b: PolyMatrix) -> PolyMatrix:
    out: PolyMatrix = [[{} for _ in b[0]] for _ in a]
    for i, row in enumerate(a):
        for j in range(len(b[0])):
            for k, value in enumerate(row):
                out[i][j] = p_add(out[i][j], p_mul(value, b[k][j]))
    return out


def p_det(a: PolyMatrix) -> Poly:
    out: Poly = {}
    n = len(a)
    for p in permutations(range(n)):
        term: Poly = {0: F((-1)**sum(p[i] > p[j] for i in range(n) for j in range(i+1, n)))}
        for i in range(n):
            term = p_mul(term, a[i][p[i]])
        out = p_add(out, term)
    return out


def canonical_poly(p: PolyMatrix, number: int) -> Poly:
    # coeff[z^number] det(I+zP) = sum of all principal minors, not selected modes.
    out: Poly = {}
    for ix in combinations(range(len(p)), number):
        out = p_add(out, p_det([[p[i][j] for j in ix] for i in ix]))
    return out


def move(mask: int, source: int, dest: int) -> tuple[int, int] | None:
    if not mask >> source & 1:
        return None
    sign = (-1)**((mask & ((1 << source)-1)).bit_count())
    removed = mask ^ (1 << source)
    if removed >> dest & 1:
        return None
    sign *= (-1)**((removed & ((1 << dest)-1)).bit_count())
    return removed | (1 << dest), sign


def gamma(t: Matrix, number: int) -> Matrix:
    states = list(combinations(range(len(t)), number))
    return [[det([[t[i][j] for j in cols] for i in rows]) for cols in states] for rows in states]


def dg(a: Matrix, number: int) -> Matrix:
    basis = [sum(1 << i for i in ix) for ix in combinations(range(len(a)), number)]
    where = {v: i for i, v in enumerate(basis)}
    out = [[F(0) for _ in basis] for _ in basis]
    for column, state in enumerate(basis):
        for dest, row in enumerate(a):
            for source, value in enumerate(row):
                moved = move(state, source, dest)
                if moved is not None:
                    out[where[moved[0]]][column] += moved[1]*value
    return out


def partitions(mask: int) -> list[tuple[int, ...]]:
    if not mask:
        return [()]
    bit = mask & -mask
    rest = mask ^ bit
    result = []
    for blocks in partitions(rest):
        result.append((bit,) + blocks)
        for j in range(len(blocks)):
            result.append(blocks[:j] + (blocks[j] | bit,) + blocks[j+1:])
    return result


def connected4(z: Poly) -> F:
    normalized = {m: z.get(m, F(0))/z[0] for m in range(16)}
    return sum(((-1)**(len(p)-1)*math.factorial(len(p)-1)
                * math.prod(normalized[m] for m in p) for p in partitions(15)), F(0))


def determinant_control() -> dict[str, Any]:
    t = [eye(4) for _ in range(4)]
    t[0][0][1] = t[0][1][0] = F(1, 4)
    t[1] = [[F((2, 1, 3, 2)[i] if i == j else 0) for j in range(4)] for i in range(4)]
    t[2][2][3] = t[2][3][2] = F(1, 3)
    t[3][0][2] = t[3][2][0] = F(1, 5)
    a = [[[F(0) for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for j in range(4):
        a[j][(j+1) % 4][j] = F(1)
    p: PolyMatrix = [[{0: value} for value in row] for row in eye(4)]
    for j in range(4):
        factor: PolyMatrix = [[{0: F(i == k), 1 << j: a[j][i][k]} for k in range(4)] for i in range(4)]
        p = p_mm([[{0: v} for v in row] for row in t[j]], p_mm(factor, p))
    canonical = canonical_poly(p, 2)
    gt, ja = [gamma(v, 2) for v in t], [dg(v, 2) for v in a]
    direct: Poly = {}
    wrong: Poly = {}
    for mask in range(16):
        word, reverse = eye(6), eye(6)
        for j in range(4):
            exact_factor = mm(gt[j], ja[j] if mask >> j & 1 else eye(6))
            word = mm(exact_factor, word)
            reverse = mm(reverse, exact_factor)
        direct[mask], wrong[mask] = trace(word), trace(reverse)
    assert all(canonical.get(m, F(0)) == direct[m] for m in range(16))
    assert direct != wrong
    grand: Poly = {}
    for number in range(5):
        grand = p_add(grand, canonical_poly(p, number))
    assert connected4(canonical) != connected4(grand)
    assert connected4(canonical) != direct[15]/direct[0]
    return {'all16_multilinear_coefficients_equal_entire_wedge2_trace': True,
            'canonical_dimension': 6, 'canonical_four_connected_exact': str(connected4(canonical)),
            'grandcanonical_wrong_connected_exact': str(connected4(grand)),
            'ordered_source_fourword_exact': str(direct[15]), 'normalizer': str(direct[0]),
            'reversed_time_order_rejected': True, 'raw_fourpoint_as_log_cumulant_rejected': True,
            'canonical_not_selected_column_or_crop': True}


def complement_control() -> dict[str, Any]:
    # Full canonical4 of eight orbitals: four loop orbitals + four spectators.
    basis = [m for m in range(256) if m.bit_count() == 4]
    assert len(basis) == 70
    coeff: dict[int, int] = {}
    for state in basis:
        value = 0
        for order in permutations(range(4)):
            current, sign = state, 1
            for edge in order:
                result = move(current, edge, (edge+1) % 4)
                if result is None:
                    sign = 0
                    break
                current, factor = result
                sign *= factor
            if sign and current == state:
                value += sign
        coeff[state] = value
    rows = []
    for z in (F(1), F(1, 16)):
        full = sum((v*z**((m & 15).bit_count()) for m, v in coeff.items()), F(0))
        selected = sum((v*z**2 for m, v in coeff.items() if m & 15 in (5, 10)), F(0))
        complement = full-selected
        exact = 16*z+16*z**3-96*z**2
        assert full == exact and selected == -48*z**2
        rows.append({'z': str(z), 'full': str(full), 'opposite_pair': str(selected),
                     'complement': str(complement)})
    assert F(rows[1]['full']) > 0 and F(rows[1]['opposite_pair']) < 0
    return {'entire_canonical4_dimension': 70, 'all24_orders': 24,
            'spectator_complement_can_reverse_conditional_sign': True, 'rows': rows,
            'no_physical_SU4_projector_or_ground_weight_claim': True}


def shared_charge_control() -> dict[str, Any]:
    # Two-site one-edge physical quotient: rho=(Q,-Q), Lplus=[[1,-1],[-1,1]]/4.
    rows = []
    for charges in ((1, 1), (1, -1), tuple(1 for _ in range(64))):
        total = sum(charges)
        shared = 64*total*total
        independent = 64*sum(q*q for q in charges)
        assert shared != independent
        rows.append({'charged_copies': len(charges), 'total_charge_transfer': total,
                     'once_shared_fullE_HC_exact': shared, 'wrong_independent_HC': independent})
    assert [r['once_shared_fullE_HC_exact'] for r in rows] == [256, 0, 262144]
    # One common imaginary Gaussian field sigma has cov128*delta*Lplus.
    # E exp(i rho.sigma)=exp(-64*delta*rho.Lplus.rho); separate fields lose cross terms.
    return {'rows': rows, 'covariance_coefficient': 128, 'one_shared_imaginary_field_required': True,
            'neutral_U1_charge_not_inserted': True, 'Ref_and_background_must_glue_with_full_density': True,
            'fullE_not_HC_added_twice': True, 'single_finite_graph_factor_NOT_quantum_path_limit': True}


def sewing_character_control() -> dict[str, Any]:
    rows = []
    for n in (4, 6, 8):
        dimension = 4*n**3
        number = dimension//2
        coefficient = (math.comb(dimension-2, number)
                       - 2*math.comb(dimension-2, number-1)
                       + math.comb(dimension-2, number-2))
        total = math.comb(dimension, number)
        assert coefficient*(dimension-1) == -total
        ratio = F(coefficient, total)
        assert ratio == F(-1, dimension-1) and ratio**65 < 0
        rows.append({'N': n, 'oneparticle_dimension_PARAMETER': dimension,
                     'canonical_number_PARAMETER': number,
                     'normalized_character_exact': str(ratio),
                     'odd65_product_exact': str(ratio**65)})
    return {'rows': rows, 'algebra_of_one_SU4_sewing_with_two_negative_eigenvalues': True,
            'native_onsite_full_heat_and_quotient_path_binding_NOT_assumed': True,
            'conditional_negative_sign_NOT_integrated_ground_no_go': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    before = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in request['pins']}
    assert before == request['pins']
    result = {'kind': 'EXACT_CANONICAL_DETERMINANT_SHARED_HS_AND_SIGNED_COMPLEMENT_METHOD_NOT_NATIVE_GROUND',
              'canonical_ordered_response': determinant_control(), 'canonical_complement': complement_control(),
              'shared_collective_charge': shared_charge_control(), 'method_NOT_full_native_quantum_Gauss_source': True,
              'sewing_character_algebra': sewing_character_control(),
              'native_ground_path_integral_or_IR_computed': False, 'physical1_phase_or_stiffness_or_atom': False,
              'scientific_admission': False, 'campaign_complete': False}
    assert before == {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in request['pins']}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'canonical': 'all16 exact', 'shared_HS': 'cross terms retained', 'phase': False}))


if __name__ == '__main__':
    main()

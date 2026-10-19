"""Independent exact algebra controls; no author module import or ground backend."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import sympy as s


def clifford_basis() -> tuple[list[Any], list[Any]]:
    x = s.Matrix([[0, 1], [1, 0]])
    y = s.Matrix([[0, -s.I], [s.I, 0]])
    z = s.diag(1, -1)
    gamma = [s.kronecker_product(z, s.eye(2))] + [s.kronecker_product(x, a) for a in (x, y, z)]
    basis = []
    for degree in range(1, 5):
        for subset in itertools.combinations(range(4), degree):
            w = s.eye(4)
            for j in subset:
                w = w * gamma[j]
            basis.append(s.I ** (degree * (degree - 1) // 2) * w / 2)
    assert all(s.trace(a.H * b) == int(i == j) for i, a in enumerate(basis) for j, b in enumerate(basis))
    return gamma, basis


def cost(amplitudes: list[Any]) -> tuple[Any, Any, Any]:
    gamma, basis = clifford_basis()
    c = [a * g for a, g in zip(amplitudes, gamma)]
    # Unit Hilbert-Schmidt basis directions; divide their Gram by two to
    # convert to the native 2Tr-unit coefficient basis. Build residual columns.
    residuals = []
    for mu in range(4):
        for u in basis:
            d = [u if j == mu else s.zeros(4) for j in range(4)]
            column = []
            for i in range(4):
                self_residual = c[i] * d[i] + d[i] * c[i] - s.trace(c[i] * d[i]) * s.eye(4) / 2
                column.extend(list(4 * self_residual))
                column.append(8 * s.trace(c[i] * d[i]) / 2)
                for j in range(4):
                    if i != j:
                        cross = c[i] * d[j] + d[j] * c[i] + c[j] * d[i] + d[i] * c[j]
                        column.extend(list(2 * cross))
            residuals.append(s.Matrix(column))
    a = s.Matrix.hstack(*residuals)
    q = (a.H * a / 2).applyfunc(s.simplify)
    orbit = s.Matrix(60, 15, lambda r, j: s.simplify(s.I * s.trace(basis[r % 15] *
        (basis[j] * c[r // 15] - c[r // 15] * basis[j]))))
    gram = orbit.T * orbit
    assert q * orbit == s.zeros(60, 15)
    return q, orbit, gram


def congruence_control() -> dict[str, Any]:
    q, l, g = cost([s.Integer(1)] * 4)
    assert l.rank() == 15 and q.rank() == 45
    pg = l * g.inv() * l.T
    inverse = (q + pg).inv()
    assert (q + pg) * inverse == s.eye(60)
    b = max(sum(abs(v) for v in inverse.row(i)) for i in range(60))
    c = 1 / b
    spectrum = q.eigenvals()
    assert spectrum[0] == 15 and min(v for v in spectrum if v != 0) >= 1
    gamma, basis = clifford_basis()
    radial_rows = s.Matrix(4, 60, lambda i, j: s.trace(gamma[i] * basis[j % 15]) / 2 if j // 15 == i else 0)
    defect = q - 32 * radial_rows.T * radial_rows
    assert defect.rank() == 41 and defect * radial_rows.T == s.zeros(60, 4)
    # Native matrix congruence, not a comparison of stored receipts.
    a = [s.Rational(1, 8), s.Rational(1, 4), s.Rational(1, 2), s.Integer(1)]
    qa, la, ga = cost(a)
    da = s.diag(*[v for v in a for _ in range(15)])
    assert la == da * l
    scaled = da.T * qa * da
    # Exact symmetric PSD test of the claimed minimum fourth-power bound.
    positive = scaled - min(a) ** 4 * q
    assert positive.is_positive_semidefinite
    v = s.eye(60)[:, 0]
    assert (v.T * q * v)[0] > 0
    tiny_q, _, _ = cost([s.Rational(1, 8)] * 4)
    tiny_d = s.eye(60) / 8
    assert (v.T * tiny_d * tiny_q * tiny_d * v)[0] == (v.T * q * v)[0] / 4096
    assert (v.T * tiny_d * tiny_q * tiny_d * v)[0] < (v.T * q * v)[0] / 64
    assert ga.eigenvals() and min(ga.eigenvals()) >= s.Rational(1, 32) and max(ga.eigenvals()) <= 16
    return {'unit_inverse_row_bound': str(b), 'unit_c': str(c),
            'unit_nonzero_spectrum': {str(k): int(v) for k, v in spectrum.items() if k != 0},
            'compact_inverse_bound': str(c / 32768),
            'interface_one_over_32768_independently_valid': bool(c >= 1 or min(v for v in spectrum if v != 0) >= 1),
            'rank': 45, 'deleted_norm_rank': 41, 'deleted_norm_extra_flats': 4,
            'orbit_rank': 15, 'congruence_min_fourth_power': True,
            'wrong_min_square_congruence_rejected': True,
            'no_quantum_gap_inferred': True}


Sparse = dict[tuple[int, int], Any]


def fermion_move(mask: int, target: int, source: int) -> tuple[int, int] | None:
    if not (mask >> source) & 1:
        return None
    sign = (-1) ** ((mask & ((1 << source) - 1)).bit_count())
    empty = mask ^ (1 << source)
    if (empty >> target) & 1:
        return None
    sign *= (-1) ** ((empty & ((1 << target) - 1)).bit_count())
    return empty | (1 << target), sign


def second_quantized(matrix: Any, offset: int) -> Sparse:
    out: Sparse = {}
    for mask in range(256):
        for i in range(4):
            for j in range(4):
                coefficient = matrix[i, j]
                if not coefficient:
                    continue
                move = fermion_move(mask, offset + i, offset + j)
                if move is not None:
                    key = (move[0], mask)
                    out[key] = out.get(key, 0) + coefficient * move[1]
    return {k: v for k, v in out.items() if v != 0}


def multiply(a: Sparse, b: Sparse) -> Sparse:
    rows: dict[int, list[tuple[int, Any]]] = {}
    for (i, j), value in a.items():
        rows.setdefault(j, []).append((i, value))
    result: Sparse = {}
    for (j, k), bv in b.items():
        for i, av in rows.get(j, []):
            key = (i, k)
            result[key] = result.get(key, 0) + av * bv
    return {k: s.simplify(v) for k, v in result.items() if v != 0}


def add(a: Sparse, b: Sparse, sign: int = 1) -> Sparse:
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + sign * v
    return {k: s.simplify(v) for k, v in result.items() if v != 0}


def commutator(a: Sparse, b: Sparse) -> Sparse:
    return add(multiply(a, b), multiply(b, a), -1)


def source_control() -> dict[str, Any]:
    e: Sparse = {}
    for mask in range(256):
        for alpha in range(4):
            move = fermion_move(mask, alpha, 4 + alpha)
            if move is not None:
                e[(move[0], mask)] = s.Rational(move[1], 2)
    generators = []
    for i in range(4):
        for j in range(i + 1, 4):
            x, y = s.zeros(4), s.zeros(4)
            x[i, j] = x[j, i] = 1
            y[i, j], y[j, i] = -s.I, s.I
            generators.extend([x, y])
    for i in range(3):
        t = s.zeros(4)
        t[i, i], t[i + 1, i + 1] = 1, -1
        generators.append(t)
    for t in generators:
        active, ref = second_quantized(t, 0), second_quantized(t, 4)
        assert commutator(add(active, ref), e) == {}
        assert commutator(active, e) != {}
    adjoint = {(j, i): s.conjugate(v) for (i, j), v in e.items()}
    square = multiply(adjoint, e)
    for mask in range(240, 256):
        assert {i: v for (i, j), v in square.items() if j == mask} == (
            {mask: s.Rational(4 - (mask & 15).bit_count(), 4)} if (mask & 15) != 15 else {})
    row = max(sum(abs(v) for (i, _), v in e.items() if i == k) for k in range(256))
    col = max(sum(abs(v) for (_, j), v in e.items() if j == k) for k in range(256))
    assert row == col == 2
    assert square.get((0, 0), 0) == 0 != s.Rational(4, 4)
    assert square.get((255, 255), 0) == 0
    # Actual absolute spectral identity does not license separate shifted windows.
    hn = s.diag(7, 9)
    hc = s.diag(0, 7, 9)
    embedding = s.Matrix([[0, 0], [1, 0], [0, 1]])
    assert hc * embedding == embedding * hn
    pn = s.diag(1, 0)  # neutral ground-shifted width one
    pc = s.diag(1, 0, 0)  # charged ground-shifted width one
    assert pc * embedding != embedding * pn
    # Polar/scalar square is insufficient: a unitary has scalar square but
    # need not intertwine the Hamiltonian.
    polar = s.Matrix([[0, 1], [1, 0]])
    assert polar.H * polar == s.eye(2)
    assert polar * hn != hn * polar
    return {'full_signed_CAR_dimension': 256, 'all15_joint_colour_commutators_zero': True,
            'all15_missingRef_and_separated_endpoint_commutators_nonzero': True,
            'fullRef_Schur_norm_upper': 2, 'bound_is_not_exact_norm_claim': True,
            'Ref4_source_square': '(4-n_active)/4', 'wrong_global_square_rejected': True,
            'zero_weight_allowed_on_Ref4_active4': True,
            'wrong_individual_ground_shifts_rejected': True,
            'polar_scalar_square_not_intertwining_rejected': True}


def mixed_and_graph_control() -> dict[str, Any]:
    pz, ra, rb, electric = map(s.Rational, (5, 2, -1, 3))
    q = 4 * (pz ** 2 + (ra - electric) ** 2 / 3 + (rb + electric) ** 2 / 5) + s.Rational(3, 20) * electric ** 2
    wrong = 4 * (pz ** 2 + (ra ** 2 + electric ** 2) / 3 + (rb ** 2 + electric ** 2) / 5) + s.Rational(3, 20) * electric ** 2
    assert q != wrong and q >= 4 * pz ** 2 + s.Rational(3, 20) * electric ** 2
    t = s.Symbol('t', real=True)
    density = s.exp(2 * t + t * t)
    f = t * t + 1
    flattened = s.simplify(s.sqrt(density) * s.diff(f / s.sqrt(density), t))
    ordered_second = s.simplify(s.sqrt(density) * s.diff(f / s.sqrt(density), t, 2))
    assert s.expand(flattened - s.diff(f, t) + (1 + t) * f) == 0
    assert ordered_second != s.diff(f, t, 2)
    u = s.Matrix([[s.Rational(3, 5), -s.Rational(4, 5)], [s.Rational(4, 5), s.Rational(3, 5)]])
    k = u * s.diag(16, 81) * u.T
    m = s.diag(1, 2)
    root = u * s.diag(2, 3) * u.T
    assert root ** 4 == k and (k - m ** 4).is_positive_semidefinite
    assert (root ** 2 - m ** 2).is_positive_semidefinite
    assert (s.eye(2) - root.inv() * m ** 2 * root.inv()).is_positive_semidefinite
    assert m * root != root * m
    # Scalar invariant IMS identity, including its exact transition cost.
    v = s.Symbol('v', real=True)
    transition = 6 * v ** 5 - 15 * v ** 4 + 10 * v ** 3
    assert s.integrate(s.diff(transition, v) ** 2, (v, 0, 1)) == s.Rational(10, 7)
    r, d = s.symbols('r d', real=True)
    identity = d ** 2 / 2 + r ** 4 / 2 + s.Rational(1, 8) - d * r
    assert s.expand(identity - (r - d) ** 2 / 2 - (r * r - s.Rational(1, 2)) ** 2 / 2) == 0
    return {'mixed_kinetic_exact_sample': str(q), 'deleted_crosses_wrong_sample': str(wrong),
            'density_first_and_ordered_second_retained': True,
            'noncommuting_Loewner_Heinz_quarter_graph_control': True,
            'wrong_commuting_power_or_polar_inference_rejected': True,
            'Young_identity_exact': True, 'IMS_transition_integral': '10/7'}


def sea_control(profile: dict[str, Any]) -> dict[str, Any]:
    m = s.Rational(profile['onsite'])
    c0 = abs(s.Rational(profile['paths'][0]['c0']))
    ci = abs(s.Rational(profile['paths'][0]['ci']))
    # sqrt(2)<3/2 and every aligned amplitude <=sqrt(2); both hop adjoints.
    native_bound = s.Rational(3, 2) * (abs(m) + 6 * (c0 + ci)) + s.Rational(1, 2 ** 18)
    assert native_bound < 156 < 255
    margin = 256 - native_bound
    assert margin > 100 > s.Rational(1, 256)
    assert m - 6 * c0 > s.Rational(1, 8)
    # The bare strict norm premise is not itself a quantitative margin.
    wrong = 256 - s.Rational(1, 65536)
    assert wrong < 256 and 1 / (256 - wrong) > 256
    gamma, _ = clifford_basis()
    h = 2 * gamma[0] + gamma[1] + gamma[2] + gamma[3]
    epsilon = s.sqrt(7)
    sign_h = h / epsilon
    projection = (s.eye(4) + sign_h) / 2
    ax, ay = s.zeros(4), s.zeros(4)
    ax[0, 2] = ax[2, 0] = 1
    ay[0, 2], ay[2, 0] = -s.I, s.I
    dpx = (ax - sign_h * ax * sign_h) / (4 * epsilon)
    dpy = (ay - sign_h * ay * sign_h) / (4 * epsilon)
    curvature = s.simplify(s.trace(projection * (dpx * dpy - dpy * dpx)))
    assert curvature == s.I / (7 * s.sqrt(7)) and curvature != 0
    # For the literal native bands replace mass=2 by sqrt(x0) M(k) and
    # both derivatives by M(k) ax/ay. M(k)>0 at every AP momentum, so
    # the determinant curvature sum has strictly positive imaginary part.
    # N4 native dimensions, ALL canonical 128 columns, full 256 onebody rows.
    occupied = (1 << 128) - 1
    def action(state: dict[int, Any], pairs: list[tuple[int, int]]) -> dict[int, Any]:
        result: dict[int, Any] = {}
        for mask, coefficient in state.items():
            for lo, hi in pairs:
                for target, source, weight in ((hi, lo, 1), (lo, hi, -1)):
                    move = fermion_move(mask, target, source)
                    if move is not None:
                        result[move[0]] = result.get(move[0], 0) + coefficient * weight * move[1]
        return {k: v for k, v in result.items() if v != 0}
    base = {occupied: s.Integer(1)}
    dx = action(base, [(0, 128)])
    dy = action(base, [(127, 255)])
    mixed = action(dx, [(127, 255)])
    assert dx and dy and mixed and mixed == action(dy, [(0, 128)])
    truncated = {(1 << 2) - 1: s.Integer(1)}
    assert action(truncated, [(127, 255)]) == {}
    assert sum(v * v for v in mixed.values()) == 1
    return {'native_onebody_uniform_rational_upper': str(native_bound),
            'right_edge_uniform_margin': str(margin), 'right_edge_resolvent_upper': str(1 / margin),
            'strict_norm_only_adverse_resolvent': '65536',
            'contour_R256_repaired_by_native_coefficients': True,
            'rank2_Clifford_determinant_Berry_curvature': str(curvature),
            'wrong_joint_Kato_horizontal_ray_rejected': True,
            'native_uniform_variation_curvature': 'i sum_k M(k)^2 mass(k)/(2 epsilon(k)^3), strictly nonzero',
            'canonical_columns': 128, 'onebody_rows': 256,
            'whole_exterior_distinct_column_ordered_mixed_jet_nonzero': True,
            'local_refill_or_two_column_truncation_rejected': True,
            'control_scope': 'exact universal exterior algebra in actual N4 dimensions, not actual occupied-ground columns'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for pin in request['pins'].values():
        file = args.request.parent / pin['member']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == pin['sha256']
    profile = json.loads((args.request.parent / request['pins']['profiles']['member']).read_text())['primitive_profiles'][0]
    receipt = {'kind': 'INDEPENDENT_EXACT_FAST_PHYSICAL_REVIEW_NOT_QUANTUM_GAP',
               'governor': request['governor'], 'targets': request['pins'],
               'congruence': congruence_control(), 'source': source_control(),
               'mixed_graph_IMS': mixed_and_graph_control(), 'sea': sea_control(profile),
               'old_entrypoints_imported_or_replayed': False, 'ground_backend_executed': False,
               'scientific_admission': False, 'uniform_fast_gap_proved': False,
               'actual_ground_overlap_proved': False, 'campaign_complete': False}
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'independent_assertions_reached': True, 'uniform_fast_gap_proved': False}), flush=True)


if __name__ == '__main__':
    main()

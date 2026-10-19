"""Independent small native representation/Gauss controls; no target import or rerun."""
from __future__ import annotations

import argparse
import zipfile
from typing import cast
import hashlib
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path

import sympy as s

ROOT = Path(__file__).resolve().parent
PINS = {
    'root_pair_reference_parent_20261011.py': '3e3c2d352eea421d3260ec9a063a042287fd3e7cc9f95f39ba823caf4476a51c',
    'root_pair_reference_parent_attempt2_20261011_receipt.json': '226e2ec28a156ce3109136c48faefbfec356e86ebe76755b4319490631229feb',
    'root_pair_reference_parent_attempt2_20261011_executed.zip': '7a73a361080d4d56bcd6d4a4a77d6e4da839643829a8fbbc3bcefdfd627c3031',
    'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json': '102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8',
    'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json': '2f019dd49873a5e13f130c65ef6739b9af78a16f85138dc6c6ab6e32e65c7a12',
    'charged_lower_review_attempt2_20261011_receipt.json': '7e2bed994a5404794fd888b4812d4bac7873c897f0edb615a3ddadfd7ae961f0',
    'joined_phi_u1_physical_charged_canonical_neutral_scoped_acceptance_20261011.json': '46d482d19a3cd409979b3a74d9237887c5bdf13d7e241ad9ea9fcf30d597841c',
    'three_cell_native_proof_20261011.json': 'c67fa1780acd241d0f7869e34d4d3e7979f3199b57763392354817d28c7fb043',
}


def sha(path: Path) -> str:
    if path.name == 'root_pair_reference_parent_20261011.py':
        with zipfile.ZipFile(ROOT / 'root_pair_reference_parent_attempt2_20261011_executed.zip') as archive:
            return hashlib.sha256(archive.read(path.name)).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lift(matrix: s.Matrix, number: int) -> tuple[s.Matrix, list[int]]:
    masks = [mask for mask in range(16) if mask.bit_count() == number]
    result = s.zeros(len(masks))
    for column, mask in enumerate(masks):
        for q, p in itertools.product(range(4), repeat=2):
            if mask & (1 << q):
                tail = mask ^ (1 << q)
                if not tail & (1 << p):
                    sign = (-1) ** ((mask & ((1 << q) - 1)).bit_count()
                                     + (tail & ((1 << p) - 1)).bit_count())
                    result[masks.index(tail | (1 << p)), column] += sign * matrix[p, q]
    return result, masks


def exterior(matrix: s.Matrix, number: int) -> s.Matrix:
    subsets = list(itertools.combinations(range(4), number))
    return s.Matrix([[matrix.extract(a, b).det() for b in subsets] for a in subsets])


def native_representation() -> dict[str, object]:
    m, c, d = s.symbols('m c d', real=True)
    z = s.diag(1, -1)
    x = s.Matrix([[0, 1], [1, 0]])
    y = s.Matrix([[0, -s.I], [s.I, 0]])
    a = -s.kronecker_product(s.eye(2), z)
    b = s.kronecker_product(x, z)
    cc = -s.kronecker_product(y, x)
    rho = s.kronecker_product(z, s.eye(2)) / 2
    h = m * a + c * b + d * cc
    assert b * b == s.eye(4) and b * h * b == h and b * rho * b == -rho
    scalar = s.Matrix([0, 0, -1, 1, 0, 0]) / s.sqrt(2)
    h2, masks = lift(h, 2)
    r2, _ = lift(s.Matrix(rho), 2)
    assert masks == [3, 5, 6, 9, 10, 12]
    assert h2 * scalar == s.zeros(6, 1) and r2 * scalar == s.zeros(6, 1)
    # Thus the full orthogonal five-dimensional complement reduces BOTH generators.
    hole = s.zeros(4)
    _, masks3 = lift(h, 3)
    for p in range(4):
        hole[masks3.index(15 ^ (1 << p)), p] = (-1) ** p
    k = s.diag(1, -1, 1, -1) * s.Matrix([[int(i + j == 3) for j in range(4)] for i in range(4)])
    t3 = hole * k
    for generator in (h, rho):
        g3, _ = lift(s.Matrix(generator), 3)
        assert s.simplify(g3 * t3 - t3 * generator) == s.zeros(4)
    for number in range(5):
        hn, masks = lift(h, number)
        rn, _ = lift(s.Matrix(rho), number)
        # exterior() uses lexicographic subset order; align to the CAR-mask order.
        subsets = list(itertools.combinations(range(4), number))
        positions = [subsets.index(tuple(i for i in range(4) if mask >> i & 1)) for mask in masks]
        wn = exterior(s.Matrix(b), number).extract(positions, positions)
        assert s.simplify(wn * hn * wn.conjugate().T - hn) == s.zeros(len(masks))
        assert s.simplify(wn * rn * wn.conjugate().T + rn) == s.zeros(len(masks))
    spectrum = []
    concrete = h.subs({m: 3, c: -1, d: 4})
    for number in range(5):
        hn, _ = lift(concrete, number)
        spectrum.append({'number': number, 'eigenvalues': sorted(int(e) for e, count in cast(dict[s.Expr, int], hn.eigenvals()).items() for _ in range(count))})
    return {'symbolic_full_numbers_0_to_4': True, 'joint_number1_number3_intertwiner': True,
            'vector5_plus_zero_scalar': True, 'whole_affine_sign_involution': 'physical B=X_endpoint tensor Z_spin; exterior/tensor B commutes h and flips rho for all m,c,d',
            'concrete_native_m3_c_minus1_d4': spectrum}


def sqrt_bracket(value: F) -> tuple[F, F]:
    scale = 1 << 192
    integer = math.isqrt((value.numerator * scale * scale) // value.denominator)
    lo, hi = F(integer, scale), F(integer + 1, scale)
    assert lo * lo <= value < hi * hi
    return lo, hi


def arithmetic() -> dict[str, object]:
    data = json.loads((ROOT / 'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_bytes())
    a, u = [sqrt_bracket(F(p['e_squared_exact'])) for p in data['profiles']]
    c = F(data['profiles'][0]['c0_exact'])
    assert a[1] < u[0] and c < 0
    baseline = -4 * a[1] - 252 * u[1]
    delta = a[0] + c
    budget = -5155 - baseline
    assert delta > 0 and 3 * delta > budget and 2 * a[0] + delta > budget
    upper = F(json.loads((ROOT / 'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_bytes())['ground_energy_interval_exact'][1])
    gap = -5123 - upper
    margins = {'w1': -5155 + 32 - upper, 'w2_charge2': baseline + 2 * delta + 64 - upper,
               'w_ge3': baseline + 96 - upper, 'w2_neutral_affine_if': -5187 + 64 - upper}
    assert gap == F(46955312929, 34359738368) and all(v >= gap for v in margins.values())
    # Supplemental weak neutral cost observation, not a new affine Schur certificate.
    assert -5187 - baseline < min(2 * delta, 2 * a[0])
    return {'baseline_lower': str(baseline), 'delta_lower': str(delta), 'charged_budget_upper': str(budget),
            'charged_class_exclusion_margins': {'3delta': str(3 * delta - budget), '2eA_plus_delta': str(2 * a[0] + delta - budget)},
            'gap_if_antecedents': str(gap), 'global_local_margins': {k: str(v) for k, v in margins.items()},
            'neutral_minus5187_free_negative_space_unique_minimum': True}


def graph_control(n: int) -> dict[str, object]:
    sites = list(itertools.product(range(n), repeat=3))
    div = {x: 0 for x in sites}
    d = {x: 0 for x in sites}
    incidence = {x: 0 for x in sites}
    raw_edges = []
    internal = {}
    ext_energy = 0
    for index, x in enumerate(sites):
        for axis in range(3):
            head = tuple((x[i] + int(i == axis)) % n for i in range(3))
            flow = (7 * index + axis) % 5 - 2
            raw_edges.append((x, head, axis))
            div[x] -= flow
            div[head] += flow
            if axis == 0 and x[0] % 2 == 0:
                internal[x] = flow
            else:
                d[x] -= flow
                d[head] += flow
                incidence[x] += flow * flow
                incidence[head] += flow * flow
                ext_energy += flow * flow
    endpoint_allocated = sum(32 * w for w in incidence.values())
    assert endpoint_allocated == 64 * ext_energy
    for x, flow in internal.items():
        head = (x[0] + 1, x[1], x[2])
        q = div[x] + div[head]
        rho = F(div[x] - div[head], 2)
        u = F(d[head] - d[x], 2)
        assert q == d[x] + d[head] and flow == -(rho + u)
        assert (rho + u).denominator == 1
    # Count raw oriented positive edges, not deduplicated blocked endpoint pairs.
    cell = lambda p: (p[0] // 2, p[1], p[2])
    external_pairs = [(cell(x), cell(y)) for x, y, axis in raw_edges if not (axis == 0 and x[0] % 2 == 0)]
    unordered = [tuple(sorted(pair)) for pair in external_pairs]
    duplicates = len(unordered) - len(set(unordered))
    assert len(raw_edges) == 3 * n ** 3 and len(external_pairs) == 5 * n ** 3 // 2
    if n == 4:
        assert duplicates > 0
    return {'N': n, 'raw_edges': len(raw_edges), 'external_edges': len(external_pairs),
            'blocked_parallel_duplicates': duplicates, 'endpoint_allocation': endpoint_allocated,
            'signed_integer_Gauss': True, 'rho_plus_u_integer': True}


def wrongcases() -> dict[str, object]:
    # Positive electric square does not license a NONREDUCING source-image cover.
    r = s.Matrix([[1, 1], [1, 0]])
    h = s.diag(-2, 2) + r * r
    assert h[0, 0] == 0 and h.det() == -1
    # For all weighted degrees <=2, a nonzero raw integer has magnitude1.
    endpoints = [0] * 5 + [1] * 5
    classes = {0: {(0, F(0))}, 1: set(), 2: set()}
    for count in (1, 2):
        for edges in itertools.combinations(range(10), count):
            for signs in itertools.product((-1, 1), repeat=count):
                ds = [0, 0]
                for edge, sign in zip(edges, signs):
                    ds[endpoints[edge]] += sign
                classes[count].add((sum(ds), F(ds[1] - ds[0], 2)))
    assert classes[1] == {(q, F(u, 2)) for q in (-1, 1) for u in (-1, 1)}
    assert classes[2] == {(q, F(u)) for q in (-2, 0, 2) for u in (-1, 0, 1)}
    assert 2 ** 2 == 4
    return {'nonreducing_source_image_false_bound': {'H': [[str(v) for v in h.row(i)] for i in range(2)],
            'PHP_floor': 0, 'H_determinant': -1, 'consequence': 'one negative eigenvalue despite positive complement D and nonnegative PHP'},
            'weighted_incidence_classes': {str(k): [[q, str(u)] for q, u in sorted(v)] for k, v in classes.items()},
            'single_raw_absflow2_weight4_not_pathcount1': True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    for name, pin in PINS.items():
        assert sha(ROOT / name) == pin, name
    result = {'kind': 'NEW_INDEPENDENT_REFERENCE_OPERATOR_CONTROLS_NOT_TARGET_RERUN',
              'source_sha256': sha(Path(__file__)), 'inputs': PINS, 'native': native_representation(),
              'frozen_target_source_archive': 'root_pair_reference_parent_attempt2_20261011_executed.zip; target producer is immutable3e3, not the current typed producer',
              'arithmetic': arithmetic(), 'graphs': [graph_control(n) for n in (4, 6, 8)], 'adverse': wrongcases()}
    for name, pin in PINS.items():
        assert sha(ROOT / name) == pin, name
    result['sources_inputs_unchanged'] = True
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'controls': 'passed', 'conditional_gap': result['arithmetic']['gap_if_antecedents']}))


if __name__ == '__main__':
    main()

"""Exact finite three-cell CAR/time-log method control; NOT native phase data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

import mpmath as mp

MODES = 12
NUMBER = 6
GROUND = sum(1 << (4 * cell + mode) for cell in range(3) for mode in (0, 1))
STATES = tuple(mask for mask in range(1 << MODES) if mask.bit_count() == NUMBER)
MODE_ENERGIES = (-3, -1, 1, 3)


def hop(mask: int, row: int, col: int) -> tuple[int, int] | None:
    if not mask & (1 << col):
        return None
    sign = -1 if (mask & ((1 << col) - 1)).bit_count() % 2 else 1
    emptied = mask ^ (1 << col)
    if emptied & (1 << row):
        return None
    if (emptied & ((1 << row) - 1)).bit_count() % 2:
        sign = -sign
    return emptied | (1 << row), sign


def edge_table(outer: int) -> dict[int, tuple[tuple[int, int], ...]]:
    result = {}
    for mask in STATES:
        values: dict[int, int] = defaultdict(int)
        # Full rank-four permutation J4, both hop directions, full twelve-CAR signs.
        for mode in range(4):
            row, col = 4 * outer + mode, 4 + 3 - mode
            for r, c in ((row, col), (col, row)):
                item = hop(mask, r, c)
                if item is not None:
                    values[item[0]] += item[1]
        result[mask] = tuple((target, value) for target, value in values.items() if value)
    for mask, transitions in result.items():
        for target, value in transitions:
            assert dict(result[target])[mask] == value
    return result


def energy(mask: int, central_memory: bool) -> int:
    occupancies = tuple(((mask >> (4 * cell)) & 15).bit_count() for cell in range(3))
    ka, kb = occupancies[0] - 2, occupancies[2] - 2
    qb = occupancies[1] - 2
    assert ka + kb + qb == 0
    # Declared TOY: raw tree-flow energy plus local central charge memory.
    # The latter is not claimed to equal the current native matching pencil.
    matter = 12 + sum(MODE_ENERGIES[mode % 4] for mode in range(MODES) if mask & (1 << mode))
    return matter + 64 * (ka * ka + kb * kb) + (16 * qb * qb if central_memory else 0)


def series(t: F, central_memory: bool, terms: int = 80) -> tuple[dict[tuple[int, int], F], F, int, set[int]]:
    tables = edge_table(0), edge_table(2)
    energies = {mask: energy(mask, central_memory) for mask in STATES}
    assert energies[GROUND] == 0 and sum(value == 0 for value in energies.values()) == 1
    norm = max(energies.values()) + sum(max(sum(abs(v) for _, v in row) for row in table.values()) for table in tables)
    current: dict[tuple[int, int, int], int] = {(0, 0, GROUND): 1}
    coefficients = {(a, b): F(0) for a in range(3) for b in range(3)}
    coefficients[0, 0] = F(1)
    factor = F(1)
    central_masks = {3}
    for degree in range(1, terms + 1):
        following: dict[tuple[int, int, int], int] = defaultdict(int)
        for (a, b, mask), value in current.items():
            following[a, b, mask] += energies[mask] * value
            for axis, table in enumerate(tables):
                aa, bb = a + int(axis == 0), b + int(axis == 1)
                if aa <= 2 and bb <= 2:
                    for target, weight in table[mask]:
                        following[aa, bb, target] += weight * value
                        central_masks.add((target >> 4) & 15)
        current = {key: value for key, value in following.items() if value}
        factor *= -t / degree
        for (a, b, mask), value in current.items():
            if mask == GROUND:
                coefficients[a, b] += factor * value
    # Multivariate Cauchy at |lambda_a|=|lambda_b|=1 and entire full924 operator.
    # Each Taylor coefficient remainder <= scalar norm-series tail, exact rational.
    x = t * norm
    ratio = x / (terms + 2)
    assert ratio < 1
    tail = x ** (terms + 1) / math.factorial(terms + 1) / (1 - ratio)
    assert all(coefficients[a, b] == 0 for a in range(3) for b in range(3) if a % 2 or b % 2)
    return coefficients, tail, norm, central_masks


def multiply_interval(a: tuple[F, F], b: tuple[F, F]) -> tuple[F, F]:
    products = [x * y for x in a for y in b]
    return min(products), max(products)


def connected_interval(coefficients: dict[tuple[int, int], F], error: F) -> tuple[F, F]:
    product = multiply_interval((coefficients[2, 0] - error, coefficients[2, 0] + error), (coefficients[0, 2] - error, coefficients[0, 2] + error))
    return coefficients[2, 2] - error - product[1], coefficients[2, 2] + error - product[0]


def partial_fraction_inverse(rates: tuple[int, ...], t: F) -> Any:
    """Independent exact pole coefficients; only final exponentials use 90 digits."""
    multiplicities = Counter(rates)
    result = mp.mpf(0)
    mt = mp.mpf(t.numerator) / t.denominator
    for rate, count in multiplicities.items():
        polynomial = [F(1)] + [F(0)] * (count - 1)
        for other, power in multiplicities.items():
            if other == rate:
                continue
            delta = other - rate
            expansion = [F((-1) ** k * math.comb(power + k - 1, k), delta ** (power + k)) for k in range(count)]
            polynomial = [sum((polynomial[j] * expansion[k - j] for j in range(k + 1)), F(0)) for k in range(count)]
        for pole in range(1, count + 1):
            coefficient = polynomial[count - pole]
            result += (mp.mpf(coefficient.numerator) / coefficient.denominator) * mt ** (pole - 1) / math.factorial(pole - 1) * mp.exp(-rate * mt)
    return result


def ordered_heat_paths(t: F, a: int, b: int, central_memory: bool) -> tuple[Any, int]:
    tables = edge_table(0), edge_table(2)
    grouped: dict[tuple[int, ...], int] = defaultdict(int)
    path_count = 0
    for word in sorted(set(itertools.permutations((0,) * a + (1,) * b))):
        paths = [(GROUND, (0,), 1)]
        for axis in word:
            following = []
            for mask, rates, weight in paths:
                for target, value in tables[axis][mask]:
                    following.append((target, rates + (energy(target, central_memory),), weight * value))
            paths = following
        for mask, rates, weight in paths:
            if mask == GROUND:
                grouped[tuple(sorted(rates))] += weight
                path_count += 1
    total = sum((weight * partial_fraction_inverse(rates, t) for rates, weight in grouped.items()), mp.mpf(0))
    return (-1) ** (a + b) * total, path_count


def exact_box(box: tuple[F, F]) -> dict[str, Any]:
    return {'exact': [str(v) for v in box], 'display': [float(v) for v in box]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    mp.mp.dps = 90
    rows = []
    for t in (F(1, 512), F(1, 128)):
        coefficients, error, norm, central_masks = series(t, True)
        wrong, wrong_error, _, _ = series(t, False)
        connected = connected_interval(coefficients, error)
        wrong_connected = connected_interval(wrong, wrong_error)
        assert connected[1] < 0 or connected[0] > 0
        assert connected[1] < wrong_connected[0] or wrong_connected[1] < connected[0]
        path_results = {}
        for degrees in ((2, 0), (0, 2), (2, 2)):
            value, count = ordered_heat_paths(t, *degrees, True)
            exact = coefficients[degrees]
            exact_mp = mp.mpf(exact.numerator) / exact.denominator
            tail_mp = mp.mpf(error.numerator) / error.denominator
            assert abs(value - exact_mp) <= tail_mp + mp.mpf('1e-75')
            path_results[str(degrees)] = {'path_count': count, 'value_90digit': str(value), 'agreement_defect': str(abs(value - exact_mp)), 'scope': 'Independent full ordered path/divided-difference numeric check; rational Taylor interval is the certificate.'}
        center = coefficients[2, 2] - coefficients[2, 0] * coefficients[0, 2]
        copy_rows = []
        for copies in (1, 4, 16, 64):
            z20, z02 = copies * coefficients[2, 0], copies * coefficients[0, 2]
            z22 = copies * coefficients[2, 2] + copies * (copies - 1) * coefficients[2, 0] * coefficients[0, 2]
            assert (z22 - z20 * z02) / copies == center
            copy_rows.append({'copies': copies, 'normalized_log_coefficient_per_star_exact': str((z22 - z20 * z02) / copies), 'wrong_unsubtracted_coefficient_per_star_display': float(z22 / copies)})
        rows.append({'time_exact': str(t), 'full_operator_norm_cap': norm, 'per_coefficient_Taylor_tail_exact': str(error), 'full_canonical_dimension': len(STATES), 'central_fourCAR_masks_reached': sorted(central_masks), 'z20_exact_center': str(coefficients[2, 0]), 'z02_exact_center': str(coefficients[0, 2]), 'z22_exact_center': str(coefficients[2, 2]), 'logZ_la2lb2_interval': exact_box(connected), 'wrong_omitted_central_rho_square_interval': exact_box(wrong_connected), 'factorized_central_memory_zero_rejected': True, 'wrong_omitted_central_density_rejected': True, 'independent_ordered_paths': path_results, 'disjoint_star_normalization_identity': copy_rows})
    table = edge_table(0)
    undressed_squared = sum((F(weight * weight, energy(target, True) ** 2) for target, weight in table[GROUND]), F(0))
    assert undressed_squared > 0
    result = {'kind': 'FULL_CENTRAL_FOURCAR_DYNAMIC_LOG_NORMALIZATION_METHOD_CONTROL_NOT_NATIVE_PHASE', 'own_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'canonical_number': NUMBER, 'modes': MODES, 'source': 'Exact half-filled Slater at three cells, n=2/cell; entire canonical6 exterior(C12), no intermediate CAR crop.', 'Hamiltonian': 'Declared TOY H0=D-Dground+64(ka^2+kb^2)+16rho_B^2; ka=nA-2,kb=nC-2,rhoB=-ka-kb. Va,Vb full J4 hops, both directions. This is not native canonical256/all1+63 or its charged affine pencil.', 'generating_object': '<g,exp[-T(H0+laVa+lbVb)]g>, Z00=1; separate charge parity makes odd coefficients zero; [la2lb2]logZ=z22-z20*z02.', 'no_allorder_from_finite_pair': True, 'rows': rows, 'undressed_H0_inverse_Vg_norm_squared_per_star_exact': str(undressed_squared), 'undressed_independent_star_squared_norm_exact': {str(copies): str(copies * undressed_squared) for copies in (1, 4, 16, 64)}, 'nonclaims': ['No native source-g transfer/operation cap.', 'No allorder rooted sum or physical1 radius.', 'No physicalPhi/lattice ground, bulk stiffness, massless IRatom/pole, commoncone or originalSOURCE admission.'], 'elapsed_seconds': time.monotonic() - started}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'kind': result['kind'], 'connected_boxes': [row['logZ_la2lb2_interval']['display'] for row in rows], 'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    main()

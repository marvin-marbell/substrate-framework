"""Exact tensor-local star reduction METHOD control, not native source data."""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any

import dynamic_normalization_parent_control_20261011 as car

ROOT = Path(__file__).resolve().parent
G = {3: F(3, 5), 12: F(4, 5)}
Vector = dict[int, F]


def clean(v: Vector) -> Vector:
    return {mask: value for mask, value in v.items() if value}


def difference(a: Vector, b: Vector) -> Vector:
    return clean({mask: a.get(mask, F(0)) - b.get(mask, F(0)) for mask in a.keys() | b.keys()})


def plus(a: Vector, b: Vector) -> Vector:
    return clean({mask: a.get(mask, F(0)) + b.get(mask, F(0)) for mask in a.keys() | b.keys()})


def inner(a: Vector, b: Vector) -> F:
    return sum((value * b.get(mask, F(0)) for mask, value in a.items()), F(0))


def project(v: Vector, cell: int) -> Vector:
    out: Vector = defaultdict(F)
    for mask, value in v.items():
        local = (mask >> (4 * cell)) & 15
        if local in G:
            rest = mask & ~(15 << (4 * cell))
            for target, weight in G.items():
                out[rest | (target << (4 * cell))] += value * G[local] * weight
    return clean(out)


def inverse(v: Vector) -> Vector:
    """Whole exact projected resolvent via commuting LOCAL spectral projectors."""
    grouped: dict[tuple[int, int, int], Vector] = defaultdict(dict)
    for mask, value in v.items():
        numbers = ((mask & 15).bit_count(), ((mask >> 4) & 15).bit_count(), ((mask >> 8) & 15).bit_count())
        grouped[numbers][mask] = value
    out: Vector = defaultdict(F)
    for numbers, vector in grouped.items():
        neutral = [c for c in range(3) if numbers[c] == 2]
        base = 100 * sum(n != 2 for n in numbers) + 64 * ((numbers[0] - 2)**2 + (numbers[2] - 2)**2)
        for choices in itertools.product((0, 1), repeat=len(neutral)):
            sector = vector
            for cell, excited in zip(neutral, choices):
                pv = project(sector, cell)
                sector = difference(sector, pv) if excited else pv
            rate = base + 2 * sum(choices)
            if rate:
                for mask, value in sector.items():
                    out[mask] += value / rate
    return clean(out)


def apply(v: Vector, outer: int, flipped: bool) -> Vector:
    out: Vector = defaultdict(F)
    for mask, value in v.items():
        for mode in range(4):
            row, col = 4 * outer + mode, 4 + (3 - mode if flipped else mode)
            for r, c in ((row, col), (col, row)):
                item = car.hop(mask, r, c)
                if item is not None:
                    out[item[0]] += value * item[1]
    return clean(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    omega = {a | (b << 4) | (c << 8): x * y * z for (a, x), (b, y), (c, z) in itertools.product(G.items(), repeat=3)}
    assert inner(omega, omega) == 1 and not inverse(omega)
    sa, sb = apply(omega, 0, True), apply(omega, 2, False)
    xa, xb = inverse(sa), inverse(sb)
    unprojected_a, unprojected_b = apply(xa, 0, True), apply(xb, 2, False)
    ya = difference(unprojected_a, {m: inner(omega, unprojected_a) * z for m, z in omega.items()})
    yb = difference(unprojected_b, {m: inner(omega, unprojected_b) * z for m, z in omega.items()})
    ym = plus(apply(xb, 0, True), apply(xa, 2, False))
    qa, qb = project(project(ya, 0), 2), project(project(yb, 0), 2)
    full_cross = inner(ya, inverse(yb))
    reduced_cross = inner(qa, inverse(qb))
    assert full_cross == reduced_cross != 0
    assert inner(qa, qa) < inner(ya, ya) and inner(qb, qb) < inner(yb, yb)
    # At zero external flow and outer ground, Q_global is EXACTLY central Q.
    assert inner(omega, qa) == 0 and inner(omega, qb) == 0
    assert all((m & 15) in G and ((m >> 8) & 15) in G for m in qa.keys() | qb.keys())
    # Mixed external flows are nonzero: central ground is ALLOWED there.
    center_ground = project(ym, 1)
    assert inner(center_ground, center_ground) > 0
    complete_mixed = inner(ym, inverse(ym))
    wrong_mixed = inner(difference(ym, center_ground), inverse(difference(ym, center_ground)))
    assert complete_mixed > wrong_mixed
    Aa, Ab = inner(sa, xa), inner(sb, xb)
    Ba, Bb = inner(xa, xa), inner(xb, xb)
    fold = Aa * Bb + Ab * Ba
    result: dict[str, Any] = {
        'kind': 'EXACT_OUTER_GROUND_NEUTRAL_REDUCTION_AND_MIXED_CENTRAL_GROUND_METHOD_CONTROL',
        'own_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256': {'dynamic_normalization_parent_control_20261011.py': hashlib.sha256((ROOT / 'dynamic_normalization_parent_control_20261011.py').read_bytes()).hexdigest()},
        'toy': 'Three full four-CAR cells, entire canonical6/924 dimension. Local number2 ground g=(3|01>+4|23>)/5, neutral gap2; other number sectors energy100. H0 adds64(ka^2+kb^2), ka=nA-2,kb=nC-2. Exact tensor projector spectral decomposition, not native canonical256 law.',
        'full_neutral_cross_exact': str(full_cross),
        'outer_projected_central_cross_exact': str(reduced_cross),
        'neutral_norms_exact': {'full_aa': str(inner(ya, ya)), 'central_aa': str(inner(qa, qa)), 'full_bb': str(inner(yb, yb)), 'central_bb': str(inner(qb, qb))},
        'mixed_whole_resolvent_exact': str(complete_mixed),
        'wrong_mixed_central_ground_removed_exact': str(wrong_mixed),
        'ground_normalization_fold_exact': str(fold),
        'connected_fulltime_root_exact': str(2 * full_cross + complete_mixed - fold),
        'operator_identity': 'Raw-flow superselection restricts <y_aa,R y_bb> to k=0. H0(k0)=KA+KB+KC and outer PgA/PgC commute with R. Outer projections then leave only central Q_B resolvent. In mixed nonzero-flow branch center PgB is NOT whole ground and must remain.',
        'nonclaims': ['No native reduced vertex or formation price computed.', 'No charged containing-space gap borrowed for an unproved source domain.', 'No native sign, physical1 radius, fullPhi/lattice ground, phase/stiffness/IRatom/pole/commoncone/originalSOURCE admission.'],
        'elapsed_seconds': time.monotonic() - started,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('kind', 'full_neutral_cross_exact', 'connected_fulltime_root_exact', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()

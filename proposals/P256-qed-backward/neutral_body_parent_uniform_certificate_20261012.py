"""Uniform untwisted-AP neutral canonical/Jordan class: exact interval certificate.

Global here means all CONSTANT traceless Hermitian four-matrix bodies with
identity links. Inhomogeneous fields, nontrivial centre holonomies and quantum
Phi/Haar/Gauss ground selection are outside this class.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
from typing import Any, cast

import sympy as sp

from neutral_body_parent_method_20261012 import gammas, jordan

ROOT = Path(__file__).resolve().parent
BITS = 384
Interval = tuple[F, F]


def add(a: Interval, b: Interval) -> Interval:
    return a[0] + b[0], a[1] + b[1]


def scale(c: F, a: Interval) -> Interval:
    return (c * a[0], c * a[1]) if c >= 0 else (c * a[1], c * a[0])


def square(a: Interval) -> Interval:
    lo, hi = a
    return (F(0) if lo <= 0 <= hi else min(lo * lo, hi * hi), max(lo * lo, hi * hi))


def sqrt_out(a: Interval) -> Interval:
    assert 0 <= a[0] <= a[1]
    unit = 1 << BITS
    def root_floor(value: F) -> F:
        return F(isqrt((value.numerator << (2 * BITS)) // value.denominator), unit)
    lo = root_floor(a[0])
    top = root_floor(a[1])
    hi = top if top * top == a[1] else top + F(1, unit)
    assert lo * lo <= a[0] and hi * hi >= a[1]
    return lo, hi


def div_positive(a: Interval, b: Interval) -> Interval:
    assert 0 < b[0] <= b[1]
    products = [x / y for x, y in itertools.product(a, b)]
    return min(products), max(products)


def outward_dyadic(a: Interval) -> Interval:
    # Bound denominator growth without weakening exact outward inclusion.
    unit = 1 << BITS
    lo = F((a[0].numerator * unit) // a[0].denominator, unit)
    hi = F(-((-a[1].numerator * unit) // a[1].denominator), unit)
    assert lo <= a[0] <= a[1] <= hi
    return lo, hi


def point(a: F) -> Interval:
    return a, a


def ap_trig(n: int) -> tuple[list[Interval], list[Interval]]:
    root2 = sqrt_out(point(F(2)))
    if n == 4:
        a = scale(F(1, 2), root2)
        return [a, scale(F(-1), a), scale(F(-1), a), a], [a, a, scale(F(-1), a), scale(F(-1), a)]
    if n == 6:
        b = scale(F(1, 2), sqrt_out(point(F(3))))
        return [b, point(F(0)), scale(F(-1), b), scale(F(-1), b), point(F(0)), b], [point(F(1, 2)), point(F(1)), point(F(1, 2)), point(F(-1, 2)), point(F(-1)), point(F(-1, 2))]
    assert n == 8
    a = scale(F(1, 2), sqrt_out(add(point(F(2)), root2)))
    b = scale(F(1, 2), sqrt_out(add(point(F(2)), scale(F(-1), root2))))
    return [a, b, scale(F(-1), b), scale(F(-1), a), scale(F(-1), a), scale(F(-1), b), b, a], [b, a, a, b, scale(F(-1), b), scale(F(-1), a), scale(F(-1), a), scale(F(-1), b)]


def interval_certificate(profile: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    n = candidate['N']
    body = candidate['Clifford_optimizer_FLOAT_ONLY_NOT_GLOBAL_PROOF']
    s, r = F(body['s']), F(body['r'])  # exact binary inputs, not an optimizer proof
    x = [s * s, r * r, r * r, r * r]
    cs, ss = ap_trig(n)
    paths = {p['axis']: p for p in profile['paths']}
    sums = [point(F(0)) for _ in range(4)]
    eps_sum = point(F(0))
    for indices in itertools.product(range(n), repeat=3):
        mass = point(F(profile['onsite']))
        for axis in range(3):
            mass = add(mass, scale(2 * F(paths[axis]['c0']), cs[indices[axis]]))
        a2 = [square(mass)]
        a2 += [square(scale(-2 * F(paths[axis]['ci']), ss[indices[axis]])) for axis in range(3)]
        epsilon2 = point(F(0))
        for xx, aa in zip(x, a2):
            epsilon2 = add(epsilon2, scale(xx, aa))
        epsilon = sqrt_out(epsilon2)
        assert epsilon[0] > 0
        eps_sum = add(eps_sum, epsilon)
        for mu in range(4):
            sums[mu] = outward_dyadic(add(sums[mu], div_positive(a2[mu], epsilon)))
    gradient = [add(point(128 * xx), scale(F(-1, n**3), aa)) for xx, aa in zip(x, sums)]
    grad2_upper = sum(max(abs(g[0]), abs(g[1]))**2 for g in gradient)
    energy = add(point(64 * sum((xx * xx for xx in x), F(0))), scale(F(-2, n**3), eps_sum))
    gap = grad2_upper / 256  # strong convexity m=128: f(x)-f*<=||grad f||^2/(2m)
    distance2 = grad2_upper / (128**2)
    assert gap < F(1, 1_000_000)
    return {'N': n, 'norm_squared_candidate_exact': [str(xx) for xx in x],
            'candidate_energy_interval_exact': [str(v) for v in energy],
            'unique_uniform_class_minimum_interval_exact': [str(energy[0] - gap), str(energy[1])],
            'gradient_intervals_exact': [[str(v) for v in g] for g in gradient],
            'gradient_norm2_upper_exact': str(grad2_upper),
            'candidate_energy_excess_upper_exact': str(gap),
            'candidate_norm_squared_distance2_upper_exact': str(distance2),
            'minimum_energy_display_NOT_CERTIFICATE': [float(energy[0] - gap), float(energy[1])],
            'root_enclosure_method': '384-bit exact integer-isqrt rational enclosures, no floating transcendental enclosure',
            'optimizer_reexecuted': False, 'inhomogeneous_or_quantum_ground_minimum': False}


def exact_reflection_controls(profile: dict[str, Any]) -> dict[str, Any]:
    beta, alpha = gammas()
    fields = [sp.diag(3, -1, -1, -1) / 4, (beta + alpha[0]) / 3,
              sp.diag(1, 2, -1, -2) / 4, (alpha[2] + beta / 3) / 2]
    fields = [cast(sp.Matrix, p) for p in fields]
    x = [cast(sp.Expr, sp.trace(p * p) / 4) for p in fields]
    assert all(p.trace() == 0 and p == p.H for p in fields)
    mass = cast(sp.Expr, sp.Rational(profile['onsite']) + sum(2 * sp.Rational(p['c0']) / sp.sqrt(2) for p in profile['paths']))
    d = [cast(sp.Expr, -2 * sp.Rational(p['ci']) / sp.sqrt(2)) for p in profile['paths']]
    moment = 0
    for signs in itertools.product((-1, 1), repeat=3):
        h = cast(sp.Matrix, mass * fields[0] + sum((sign * c * phi for sign, c, phi in zip(signs, d, fields[1:])), sp.zeros(4)))
        moment += sp.trace(h * h)
    expected = 4 * (mass**2 * x[0] + sum(c**2 * xx for c, xx in zip(d, x[1:])))
    assert sp.simplify(moment / 8 - expected) == 0
    ordered = jordan(fields)
    norm_floor = 64 * sum(xx * xx for xx in x)
    residual = sp.simplify(ordered - norm_floor)
    assert residual > 0
    assert sp.simplify(ordered - 16 * sum(sp.trace(p**4) for p in fields)) > 0
    return {'arbitrary_traceless_nonClifford_reflection_second_moment_exact': str(sp.simplify(expected)),
            'ordered_Jordan_minus_norm_floor_exact': str(residual),
            'omitted_ordered_cross_terms_rejected': True,
            'canonical_occupation_rule': 'KyFan_top2V(h)<=Tr(h_positive)=0.5Tr|h| is an upper, not a generic occupation identity',
            'Clifford_saturates_canonical_upper': 'exactly2V positive states, each momentum+epsilon twice',
            'generic_positive_only_or_per_k_occupation_inference': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for path, h in request['pins'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == h
    profile = json.loads((ROOT / request['profiles']).read_text())['primitive_profiles'][0]
    candidates = json.loads((ROOT / request['candidates']).read_text())['finite_sea_point_diagnostics']
    result = {'kind': 'EXACT_UNIFORM_UNTWISTED_AP_CANONICAL_NEUTRAL_JORDAN_CLASS_MINIMUM_NOT_FULL_QUANTUM_GROUND',
              'controls': exact_reflection_controls(profile),
              'certificates': [interval_certificate(profile, candidate) for candidate in candidates],
              'analytic_reduction': 'At x_mu=Tr(Phi_mu^2)/4, sign-reflection Jensen/Schatten Cauchy bounds the canonical sea from below by -2 mean_k sqrt(sum a_mu(k)^2 x_mu); ordered Jordan>=64sum x_mu^2. Four Clifford matrices simultaneously saturate both. Thus all uniform traceless-Hermitian matrices reduce exactly to f(x)=64sum x_mu^2-2mean sqrt(sum a_mu^2 x_mu).',
              'strong_convexity': 'Hessian>=128I on positive x; extension convex at boundary. Unique minimizer is interior because each coefficient has nonzero AP weight; x1=x2=x3 by spatial permutations.',
              'scope': 'all constant traceless Hermitian four-matrix bodies, identity links, untwisted antiperiodic grid. Inhomogeneous/flat-centre-twist/quantum Phi-Haar-Gauss competitors NOT proved.',
              'whole_unchanged_physical_ground_or_phase': False, 'original_SOURCE_adopted': False,
              'old_optimizer_or_solver_rerun': False, 'campaign_complete': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Exact uniform-class canonical/Jordan reduction and outward norm-minimum enclosures; no quantum ground')


if __name__ == '__main__':
    main()

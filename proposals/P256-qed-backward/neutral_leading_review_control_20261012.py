"""Independent exact leading/uniform/holonomy controls; no producer imports."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import cmp_to_key
from math import isqrt
from pathlib import Path
from typing import Any

import sympy as s

BITS = 400
UNIT = 1 << BITS


@dataclass(frozen=True)
class Box:
    lo: Q
    hi: Q

    @staticmethod
    def point(v: Q | int) -> Box:
        return Box(Q(v), Q(v))

    def rounded(self) -> Box:
        return Box(Q(self.lo * UNIT // 1, UNIT), Q(-((-self.hi * UNIT) // 1), UNIT))

    def __add__(self, other: Box) -> Box:
        return Box(self.lo + other.lo, self.hi + other.hi).rounded()

    def times(self, other: Box) -> Box:
        ends = [a * b for a, b in itertools.product((self.lo, self.hi), (other.lo, other.hi))]
        return Box(min(ends), max(ends)).rounded()

    def scale(self, v: Q | int) -> Box:
        return self.times(Box.point(v))

    def square(self) -> Box:
        lower = Q(0) if self.lo <= 0 <= self.hi else min(self.lo**2, self.hi**2)
        return Box(lower, max(self.lo**2, self.hi**2)).rounded()

    def root(self) -> Box:
        assert 0 <= self.lo <= self.hi
        a = isqrt(self.lo.numerator * UNIT**2 // self.lo.denominator)
        b = isqrt(self.hi.numerator * UNIT**2 // self.hi.denominator)
        if Q(b, UNIT)**2 < self.hi:
            b += 1
        result = Box(Q(a, UNIT), Q(b, UNIT))
        assert result.lo**2 <= self.lo and result.hi**2 >= self.hi
        return result

    def divide(self, denominator: Box) -> Box:
        assert denominator.lo > 0
        return self.times(Box(1 / denominator.hi, 1 / denominator.lo))

    def strings(self) -> list[str]:
        return [str(self.lo), str(self.hi)]


def trig(n: int) -> tuple[list[Box], list[Box]]:
    # Build the first-quadrant algebraic values; all other entries by symmetry.
    rt2 = Box.point(2).root()
    if n == 4:
        c = [rt2.scale(Q(1, 2))] * 2
        q = c
    elif n == 6:
        c = [Box.point(3).root().scale(Q(1, 2)), Box.point(0), Box.point(3).root().scale(Q(-1, 2))]
        q = [Box.point(Q(1, 2)), Box.point(1), Box.point(Q(1, 2))]
        return c + [v.scale(-1) for v in c], q + [v.scale(-1) for v in q]
    else:
        assert n == 8
        a = (Box.point(2) + rt2).root().scale(Q(1, 2))
        b = (Box.point(2) + rt2.scale(-1)).root().scale(Q(1, 2))
        c, q = [a, b], [b, a]
    positive_c = c if n == 8 else [c[0]]
    positive_q = q if n == 8 else [q[0]]
    cs = positive_c + [v.scale(-1) for v in reversed(positive_c)]
    ss = positive_q + list(reversed(positive_q))
    return cs + list(reversed(cs)), ss + [v.scale(-1) for v in ss]


def sums(profile: dict[str, Any], x: list[Q], n: int, seam: bool = False) -> tuple[Box, list[Box]]:
    cs, ss = trig(n)
    caxes, saxes = [cs] * 3, [ss] * 3
    if seam:
        assert n == 4
        c8, s8 = trig(8)
        caxes = [[c8[j] for j in (1, 3, 5, 7)], cs, cs]
        saxes = [[s8[j] for j in (1, 3, 5, 7)], ss, ss]
    m, c0, ci = [Q(profile[k]) for k in ('m', 'c0', 'ci')]
    energy = Box.point(64 * sum(v*v for v in x))
    gradient = [Box.point(128*v) for v in x]
    for k in itertools.product(range(n), repeat=3):
        mass = Box.point(m)
        for a in range(3):
            mass = mass + caxes[a][k[a]].scale(2*c0)
        weights = [mass.square()] + [saxes[a][k[a]].scale(-2*ci).square() for a in range(3)]
        eps2 = Box.point(0)
        for v, w in zip(x, weights):
            eps2 = eps2 + w.scale(v)
        eps = eps2.root()
        energy = energy + eps.scale(Q(-2, n**3))
        for a in range(4):
            gradient[a] = gradient[a] + weights[a].divide(eps).scale(Q(-1, n**3))
    return energy, gradient


def norm_certificates(request: dict[str, Any], parent_receipt: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for candidate, old in zip(request['candidates'], parent_receipt['certificates'], strict=True):
        n, x = candidate['N'], [Q(v) for v in candidate['x']]
        assert n == old['N'] and [str(v) for v in x] == old['norm_squared_candidate_exact']
        energy, gradient = sums(request['profile'], x, n)
        grad2 = sum(max(abs(g.lo), abs(g.hi))**2 for g in gradient)
        distance2, excess = grad2 / 128**2, grad2 / 256
        assert distance2 < min(v*v for v in x)
        old_energy = [Q(v) for v in old['candidate_energy_interval_exact']]
        assert old_energy[0] <= energy.lo <= energy.hi <= old_energy[1]
        for g, previous in zip(gradient, old['gradient_intervals_exact'], strict=True):
            assert max(g.lo, Q(previous[0])) <= min(g.hi, Q(previous[1]))
        result.append({'N': n, 'candidate_energy': energy.strings(), 'gradient': [g.strings() for g in gradient],
                       'minimum_energy': [str(energy.lo - excess), str(energy.hi)],
                       'gradient_norm2_upper': str(grad2), 'distance2_upper': str(distance2),
                       'nearest_boundary_distance2': str(min(v*v for v in x)), 'interior': True,
                       'independent_candidate_energy_contained_in_parent': True})
    # Adverse interval exercises both sign-crossing square and non-dyadic root.
    assert Box(Q(-2), Q(3)).square() == Box(Q(0), Q(9))
    a = Box.point(Q(7, 11)).root()
    assert a.lo*a.lo <= Q(7, 11) <= a.hi*a.hi and a.lo < a.hi
    return result


def jordan(fields: list[s.Matrix]) -> s.Expr:
    return s.simplify(4 * sum(s.trace((a*b+b*a)**2) for a, b in itertools.product(fields, repeat=2)))


def matrices() -> tuple[s.Matrix, list[s.Matrix]]:
    paulis = [s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -s.I], [s.I, 0]]), s.diag(1, -1)]
    return s.kronecker_product(paulis[2], s.eye(2)), [s.kronecker_product(paulis[0], p) for p in paulis]


def matrix_controls(profile: dict[str, Any]) -> dict[str, Any]:
    beta, alpha = matrices()
    # A new commuting adverse four-matrix body: nonzero cross traces before reflection.
    fields = [s.diag(4, -2, 1, -3)/7, s.diag(1, 3, -2, -2)/5,
              s.diag(-3, 2, 4, -3)/9, s.diag(2, -1, -4, 3)/8]
    coeff = [s.Integer(v) for v in (3, 2, 5, 7)]
    x = [s.trace(p*p)/4 for p in fields]
    hs = [coeff[0]*fields[0] + sum((sign*c*p for sign,c,p in zip(signs,coeff[1:],fields[1:])), s.zeros(4))
          for signs in itertools.product((-1, 1), repeat=3)]
    moments = [s.trace(h*h) for h in hs]
    expected = 4*sum(c*c*v for c,v in zip(coeff,x))
    assert s.simplify(sum(moments)/8-expected) == 0 and moments[0] != expected
    absolute_traces = [sum(abs(v) for v in h.diagonal()) for h in hs]
    assert all(t*t <= 4*mom for t,mom in zip(absolute_traces,moments))
    mean_absolute = sum(absolute_traces)/8
    assert mean_absolute**2 < 16*sum(c*c*v for c,v in zip(coeff,x))
    j = jordan(fields); floor = 64*sum(v*v for v in x)
    assert j > floor and j > 16*sum(s.trace(p**4) for p in fields)
    clifford = [s.Rational(2, 5)*beta, *[s.Rational(3, 7)*a for a in alpha]]
    xc = [s.trace(p*p)/4 for p in clifford]
    hc = sum((c*p for c,p in zip(coeff,clifford)),s.zeros(4))
    e2 = sum(c*c*v for c,v in zip(coeff,xc))
    assert (hc*hc-e2*s.eye(4)).applyfunc(s.simplify) == s.zeros(4)
    assert jordan(clifford) == 64*sum(v*v for v in xc)
    # Complex noncommuting endpoint matrices and a literal SU4 diagonal transport.
    u = s.diag(s.I, -s.I, -1, -1)
    assert u.det() == 1 and u*u.H == s.eye(4)
    p0x, p0y, pix, piy = beta+alpha[0]/3, alpha[2]/2, alpha[1]+beta/7, alpha[0]-alpha[1]/5
    a, b = (p0x+u*p0y*u.H)/2, (pix+u*piy*u.H)/2
    c0, ci = s.Rational(profile['c0']), s.Rational(profile['ci'])
    hop = (c0*a+s.I*ci*b)*u
    exact = s.simplify(s.trace(hop*hop.H))
    assert s.simplify(exact-c0*c0*s.trace(a*a)-ci*ci*s.trace(b*b)) == 0
    assert a*b != b*a
    endpoint_cap = (c0*c0*(s.trace(p0x*p0x)+s.trace(p0y*p0y)) + ci*ci*(s.trace(pix*pix)+s.trace(piy*piy)))/2
    assert exact <= endpoint_cap
    # Rank-half KyFan Cauchy adverse: nonsymmetric spectrum makes it strict.
    h = s.diag(9, 4, 2, 1, -1, -3, -5, -7)
    assert h.trace() == 0
    top, trace2, rankhalf = 16, s.trace(h*h), 4
    assert top*top < rankhalf*trace2/2
    return {'reflection_mean_trace_h2': str(expected), 'unreflected_cross_trace_not_zero': True,
            'reflection_mean_trace_abs': str(mean_absolute), 'reflection_Schatten_Jensen_strict': True,
            'ordered_Jordan_minus_norm_floor': str(j-floor), 'ordered_cross_omission_rejected': True,
            'Clifford_simultaneous_saturation': True, 'complex_endpoint_trace_square': str(exact),
            'complex_endpoint_bound': str(endpoint_cap), 'rankhalf_KyFan_Cauchy_strict': True}


def canonical_control(profile: dict[str, Any]) -> dict[str, Any]:
    m, c0 = s.Rational(profile['m']), s.Rational(profile['c0'])
    # Different from frozen producer D=(3,-1,-1,-1)/4.
    diagonal = [s.Rational(5,5),s.Rational(-1,5),s.Rational(-1,5),s.Rational(-3,5)]
    masses = [(m+c0*s.sqrt(2)*v,count) for v,count in ((3,8),(1,24),(-1,24),(-3,8))]
    groups = [(s.simplify(d*a),count) for a,count in masses for d in diagonal]
    def compare(a: tuple[s.Expr,int], b: tuple[s.Expr,int]) -> int:
        difference = s.simplify(a[0]-b[0])
        return 0 if difference == 0 else (-1 if difference > 0 else 1)
    ordered = sorted(groups,key=cmp_to_key(compare))
    remaining, total, negative = 128,s.Integer(0),0
    for value,count in ordered:
        use = min(remaining,count)
        total += use*value
        negative += use if value < 0 else 0
        remaining -= use
        if not remaining:break
    total = s.simplify(total)
    local_wrong = s.Rational(256,5)*m
    positive_wrong = 64*m
    assert negative == 64 and local_wrong < total < positive_wrong
    assert s.simplify(total-(s.Rational(256,5)*m-s.Rational(96,5)*c0*s.sqrt(2))) == 0
    return {'diagonal_Phi0': [str(v) for v in diagonal], 'complete_top128': str(total),
            'negative_levels_required': negative, 'per_k_two_wrong': str(local_wrong),
            'positive_only_wrong': str(positive_wrong), 'both_wrong_rules_rejected': True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    request = json.loads(args.request.read_text())
    root = args.request.parent
    for name,digest in request['pins'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    profile = request['profile']
    m,c0,ci = [Q(profile[k]) for k in ('m','c0','ci')]
    ch = max(m*m+6*c0*c0,2*ci*ci)
    assert ch < 1024 and m-6*abs(c0) > Q(1,8)
    t = s.Symbol('t')
    assert s.expand((t-2)**2*(t*t+4*t+12)-(t**4-32*t+48)) == 0
    upper = Q(172,27)-m
    assert -48 < upper < -16
    old = json.loads((root/request['parent_uniform_receipt']).read_text())
    certificates = norm_certificates(request,old)
    untwisted,_ = sums(profile,[Q(1,4),Q(1,9),Q(1,9),Q(1,9)],4)
    seam,_ = sums(profile,[Q(1,4),Q(1,9),Q(1,9),Q(1,9)],4,True)
    delta = seam + untwisted.scale(-1)
    assert delta.lo > 0
    # Actual scalar AP translation with one seam: S^4=-zI, not -(z^4)I.
    z=s.I; shift=s.zeros(4)
    for j in range(3):shift[j,j+1]=1
    shift[3,0]=-z
    assert shift**4 == -z*s.eye(4)
    constant = s.zeros(4)
    for j in range(3):constant[j,j+1]=z
    constant[3,0]=-z
    assert constant**4 == -s.eye(4)
    result = {'kind':'INDEPENDENT_EXACT_LEADING_UNIFORM_HOLONOMY_REVIEW_NOT_ADMISSION',
              'target_pins':request['pins'], 'all_field_constants':{'Ch':str(ch),'Wilson_floor':str(m-6*abs(c0)),
              'universal_lower':'-48','Clifford_upper':str(upper),'global_minimum_identified':False},
              'matrix_controls':matrix_controls(profile),'generic_canonical_control':canonical_control(profile),
              'norm_certificates':certificates,'root_precision_bits':BITS,
              'holonomy':{'constant_i_links_winding':'1','single_i_seam_winding':'i',
              'literal_AP_shift_fourth_power':'-i I','constant_AP_shift_fourth_power':'-I',
              'point_energy_difference':delta.strings(),'difference_display_NOT_PROOF':[float(delta.lo),float(delta.hi)],
              'optimized_centre_winner':False},
              'old_producer_optimizer_solver_executed':False,'full_quantum_ground_support':False,
              'phase_source_law_adoption':False,'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Independent exact adverse controls completed; uniform norm vector only; no global quantum body.')


if __name__ == '__main__':
    main()

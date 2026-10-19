"""Independent review controls, never importing any original producer.
Native geometry/factors and NONNATIVE spectral falsifiers are separate.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import zipfile
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast
import sympy as s

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RootRing:
    """Exact monomial multiplication table in Q[z]/Phi_(2N)."""
    def __init__(self, n: int):
        self.n = n
        z = s.Symbol('z')
        modulus = s.Poly(s.cyclotomic_poly(2*n, z), z)
        self.d: int = cast(int, modulus.degree())
        self.roots = [tuple(int(s.rem(s.Poly(z**k, z), modulus).nth(j))
                            for j in range(self.d)) for k in range(2*n)]
        self.zero = (0,)*self.d
        self.one = self.roots[0]
        self.table = [[self.roots[(i+j) % (2*n)] for j in range(self.d)]
                      for i in range(self.d)]

    def root(self, k: int) -> tuple[int, ...]:
        return self.roots[k % (2*self.n)]

    def plus(self, *values: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(sum(v[j] for v in values) for j in range(self.d))

    def scale(self, k: int, a: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(k*x for x in a)

    def times(self, a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
        out = [0]*self.d
        for i, x in enumerate(a):
            if x:
                for j, y in enumerate(b):
                    if y:
                        for k, c in enumerate(self.table[i][j]):
                            out[k] += x*y*c
        return tuple(out)

    def star(self, a: tuple[int, ...]) -> tuple[int, ...]:
        return self.plus(*(self.scale(x, self.root(-j)) for j, x in enumerate(a)))


def geometry(n: int) -> dict[str, Any]:
    r = RootRing(n)
    add, mul, neg = r.plus, r.times, lambda x: r.scale(-1, x)
    # Direct characters, not sampled trigonometric floating approximations.
    for m in range(n):
        assert add(*(r.root(2*m*x) for x in range(n))) == r.scale(n if m == 0 else 0, r.one)
        for x in range(n):
            for length in (1, 2):
                y = (x+length) % n
                sign = (-1)**((x+length)//n)
                assert r.scale(sign, r.root((2*m+1)*y)) == r.root((2*m+1)*(x+length))
                assert r.root(2*m*y) == r.root(2*m*(x+length))
    minima = []
    qmin2 = add(r.scale(2, r.one), neg(r.root(2)), neg(r.root(-2)))
    ranks = Counter()
    for raw in itertools.product(range(n), repeat=3):
        m = tuple(x if x < n//2 else x-n for x in raw)
        d = [add(r.one, neg(r.root(-2*x))) for x in m]
        t = [mul(r.root(x), a) for x, a in zip(m, d)]
        q2 = add(*(mul(a, r.star(a)) for a in d))
        # Uncentered q^2 projector: q^2 I - d^*d.
        scaled = [[add(q2 if i == j else r.zero, neg(mul(r.star(d[i]), d[j])))
                   for j in range(3)] for i in range(3)]
        for i in range(3):
            assert t[i] == add(r.root(m[i]), neg(r.root(-m[i])))
            for j in range(3):
                assert scaled[i][j] == r.star(scaled[j][i])
                assert add(*(mul(scaled[i][k], scaled[k][j]) for k in range(3))) == mul(q2, scaled[i][j])
                # Conjugate pair INCLUDING edge-center sign at Nyquist.
                partner = -m[i]
                wrapped = partner if partner < n//2 else partner-n
                seam_gauge = (-1)**((partner-wrapped)//n)
                assert r.root(wrapped) == r.scale(seam_gauge, r.root(partner))
            assert add(*(mul(d[j], scaled[j][i]) for j in range(3))) == r.zero
        if raw == (0, 0, 0):
            assert q2 == r.zero
            ranks[3] += 1  # Harmonic toron block, distinct from local curl rank0.
        else:
            assert q2 != r.zero
            assert add(*(scaled[i][i] for i in range(3))) == r.scale(2, q2)
            ranks[2] += 1
            for i, j in ((0, 1), (0, 2), (1, 2)):
                c: list[tuple[int, ...]] = [r.zero]*3
                c[i], c[j] = t[j], neg(t[i])
                assert add(*(mul(t[k], c[k]) for k in range(3))) == r.zero
        if q2 == qmin2:
            minima.append(raw)
    assert len(minima) == 6 and ranks == {3: 1, 2: n**3-1}
    # AP electric p0 and uncentered/centered mixing must fail.
    assert add(r.one, neg(r.root(-1))) != r.zero
    d1, d2 = add(r.one, neg(r.root(-2))), add(r.one, neg(r.root(-4)))
    t1, t2 = add(r.root(1), neg(r.root(-1))), add(r.root(2), neg(r.root(-2)))
    assert add(mul(d1, t2), neg(mul(d2, t1))) != r.zero
    return {'scope': 'NEW_NATIVE_EXACT_GEOMETRY_NOT_SPECTRUM', 'N': n,
            'all_discrete_modes': n**3, 'ranks': dict(ranks), 'qN': str(2*s.sin(s.pi/n)),
            'six_minimum_modes': minima, 'qN_squared_ring': qmin2,
            'centered_gauge_Nyquist_conjugates_checked': True,
            'AP_one_two_hop_all_characters': True,
            'AP_electric_and_wrong_centering_rejected': True}


def factors(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    edges = [(x, i) for x in sites for i in range(3)]
    def nextsite(x: tuple[int, ...], i: int) -> tuple[int, ...]:
        y = list(x); y[i] = (y[i]+1) % n; return tuple(y)
    def div(flow: dict) -> dict:
        out = dict.fromkeys(sites, F(0))
        for (x, i), value in flow.items():
            out[x] += value; out[nextsite(x, i)] -= value
        return out
    occupied = {x: 4 if sum(x) % 2 == 0 else 0 for x in sites}
    assert sum(occupied.values()) == 2*n**3
    rho = {x: sum(occupied[x]-2 for _ in range(64)) for x in sites}
    long = {(x, i): F(rho[x], 6) for x, i in edges}
    assert div(long) == rho and sum(rho.values()) == 0
    x = (n-1, n-1, 1)
    face = {(x, 0): F(1), (nextsite(x, 0), 1): F(1),
            (nextsite(x, 1), 0): F(-1), (x, 1): F(-1)}
    assert not any(div(face).values())
    transverse = {e: F(5, 13)*face.get(e, 0) for e in edges}
    full = {e: long[e]+transverse[e] for e in edges}
    assert div(full) == rho
    def dot(a: dict, b: dict) -> F:
        return sum((a.get(e, 0)*b.get(e, 0) for e in edges), F(0))
    he, hc = 64*dot(transverse, transverse), 64*dot(long, long)
    assert dot(transverse, long) == 0 and 64*dot(full, full) == he+hc
    boost = F(3, 11)
    shifted = {e: full[e]+boost*face.get(e, 0) for e in edges}
    assert div(shifted) == rho
    delta = 64*(dot(shifted, shifted)-dot(full, full))
    assert delta == 128*boost*dot(face, full)+64*boost**2*dot(face, face)
    eachlong = {e: value/64 for e, value in long.items()}
    independent_hc = 64*64*dot(eachlong, eachlong)
    assert hc == 64*independent_hc
    counts, seams, aliases = [0]*3, [0]*3, [0]*3
    coefficient = F(0)
    for role, profile in enumerate(profiles):
        for path in profile['paths']:
            axis, length = path['axis'], path['length']
            if role:
                coefficient += profile['load']*length*(abs(F(path['c0']))+abs(F(path['ci'])))
            for start in sites:
                end, flow, seam = start, {}, 1
                for _ in range(length):
                    edge = (end, axis); flow[edge] = flow.get(edge, F(0))+1
                    seam *= -1 if end[axis] == n-1 else 1
                    end = nextsite(end, axis)
                assert div(flow) == {z: F(int(z == start)-int(z == end)) for z in sites}
                assert seam == (-1)**((start[axis]+length)//n)
                counts[role] += 1; seams[role] += int(seam < 0)
                aliases[role] += int(n == 4 and length == 2)
    assert counts == [3*n**3, 6*n**3, 6*n**3]
    assert seams == [3*n*n, 9*n*n, 9*n*n]
    # All distinct charged canonical CAR factors, neutral+Ref remain fixed.
    width = 4*n**3
    local = sum(15 << (4*j) for j, z in enumerate(sites) if occupied[z])
    state = sum(local << (role*width) for role in range(65)) | (15 << (65*width))
    empty = sites.index(next(z for z in sites if not occupied[z]))
    filled = sites.index(nextsite(sites[empty], 0))
    def hop(bits: int, dest: int, src: int) -> tuple[int, int]:
        assert bits & (1 << src) and not bits & (1 << dest)
        sign = (-1)**((bits & ((1 << src)-1)).bit_count())
        bits ^= 1 << src
        sign *= (-1)**((bits & ((1 << dest)-1)).bit_count())
        return bits | (1 << dest), sign
    for role in range(1, 65):
        for color in range(4):
            a, b = role*width+4*empty+color, role*width+4*filled+color
            moved, sign = hop(state, a, b)
            restored, back = hop(moved, b, a)
            assert restored == state and sign*back == 1
            assert ((moved >> (role*width)) & ((1 << width)-1)).bit_count() == 2*n**3
            assert moved & ((1 << width)-1) == local and moved >> (65*width) == 15
    # Noncommuting ordered two-link and full-matrix Jordan covariance.
    g = s.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    h = s.diag(s.I, -s.I, 1, 1)
    phi = s.diag(1, 2, -1, -2)
    psi = s.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert g.det() == h.det() == 1 and g*h != h*g
    u, v = g, h
    gauges = (h, g, g*h)
    transformed = (gauges[0]*u*gauges[1].H)*(gauges[1]*v*gauges[2].H)
    assert transformed == gauges[0]*(u*v)*gauges[2].H
    assert u*v != v*u
    jordan = phi*psi+psi*phi
    assert jordan != s.zeros(4)
    assert (h*phi*h.H)*(h*psi*h.H)+(h*psi*h.H)*(h*phi*h.H) == h*jordan*h.H
    return {'scope': 'NEW_NATIVE_FACTORS_NOT_FULLH_OR_GROUND', 'N': n,
            'each65_population': 2*n**3, 'charged64_color4_CAR_checks': 256,
            'Ref4_neutral_unchanged': True, 'rho': 'sum64(n_f-2)=128 parity',
            'collectiveHC': str(hc), 'HE': str(he), 'closed_fullE_boost_delta': str(delta),
            'wrong_independent_HC_ratio': '64', 'literal_path_counts': counts,
            'AP_seams': seams, 'N4_alias_literals': aliases, 'charged_coefficient_load': str(coefficient),
            'ordered_two_link_SU4_fullmatrix_Jordan': True}


def spectral() -> dict[str, Any]:
    k, q, v, epsilon = s.symbols('k q v epsilon', positive=True)
    density = v/(128*k)
    inverse = s.integrate(density/v, (v, 0, k))
    assert inverse == s.Rational(1, 128)
    moments = [s.integrate(v**j*density, (v, 0, k)) for j in range(7)]
    assert all(s.simplify(m-k**(j+1)/(128*(j+2))) == 0 for j, m in enumerate(moments))
    physical = epsilon/(128*k*q)
    assert s.integrate(physical/epsilon, (epsilon, 0, k*q)) == inverse
    assert s.simplify(s.integrate(physical, (epsilon, 0, k*q))/q) == k/256
    # Preserve conjugate-pair real norm, not merely the pointwise complex bound.
    weights = (F(1, 128), F(0))
    assert sum(weights)/2 == F(1, 256)
    assert F(1, 128) > F(1, 256)  # Both halves saturated would fail the real bound.
    gram = s.Matrix([[1, s.I], [-s.I, 1]])/256
    vector = s.Matrix([1, -s.I])/s.sqrt(2)
    assert (vector.H*gram*vector)[0] == s.Rational(1, 128)
    x, y = s.symbols('x y', real=True)
    assert s.expand((s.Matrix([[x, y]])*gram*s.Matrix([x, y]))[0]) == (x*x+y*y)/256
    # Compressed positive effects of disjoint energies may fail to commute.
    effect1 = s.Matrix([[1, 0], [0, 0]])/1024
    effect2 = s.Matrix([[1, 1], [1, 1]])/1024
    assert effect1*effect2 != effect2*effect1
    assert (s.eye(2)/256-effect1-effect2).is_positive_definite
    oversized = s.Matrix([s.Rational(1, 8), s.Rational(1, 8)])
    selected = s.diag(1, 0)*oversized/2
    assert (oversized.H*oversized)[0] > s.Rational(1, 128)
    assert (selected.H*selected)[0] < s.Rational(1, 128)
    assert s.FiniteSet(s.Rational(1, 3)).measure == 0
    models = []
    for j in range(1, 8):
        qv, kv = F(1, 2**j), F(7, 3)
        velocity, residue = kv/3, kv/(3*256)
        mass = qv*residue
        assert mass/(qv*velocity) == F(1, 256)
        slow = kv*qv
        assert slow/256/slow == F(1, 256)
        assert slow/256 <= kv/256
        energy = 7*kv
        assert energy > kv*qv
        models.append({'q': str(qv), 'atomic_sigma_mass': str(residue),
                       'atomic_velocity': str(velocity), 'atomic_inverse': '1/256',
                       'collapsing_velocity': str(slow), 'collapsing_mass': str(slow/256),
                       'inverse_weight_escape': '1/256', 'massive_energy': str(energy),
                       'massive_inverse': '1/256', 'massive_lowband': '0', 'zero_lowband': '0'})
    # B=I/16, L e_n=n^2 e_n: energy dual bounded, total source unbounded.
    unbounded = [{'n': n, 'B_unit_norm2': '1/256', 'total_unit_norm2': str(F(n*n, 256))}
                 for n in (2, 4, 16, 64)]
    assert all(F(item['total_unit_norm2']) > F(item['B_unit_norm2']) for item in unbounded)
    return {'scope': 'NEW_NONNATIVE_SPECTRAL_COUNTERMODELS_NOT_NATIVE_LAW',
            'atomless_inverse': str(inverse), 'atomless_moments_j0to6': [str(m) for m in moments],
            'models': models, 'half_BZ_conjugate_pair_real_inverse': '1/256',
            'symmetric_complex1over128_rejected_by_real_pair': True,
            'false_complex1over256_rejected': True, 'noncommuting_compressed_effects': True,
            'source_projection_and_rescaling_false_pass_rejected': True,
            'pointwise_singleton_upgrade_rejected': True,
            'face_amplitude_q_squared_not_unit_source': 'mu_face=q^2 mu_unit',
            'atom_from_upper_rejected_by_atomless_massive_zero': True,
            'zero_velocity_MASS_atom_excluded_not_inverse_weight_atom': True,
            'total_source_extension_countermodel': {'B': 'I/16', 'L': 'diag(n^2)',
                'local_total': 'diag(n/16) on finite sequences',
                'completed_h': 'h_n=1/n is l2 but total coefficients1/16 are not l2',
                'controls': unbounded}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    before = {name: digest(ROOT/name) for name in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:
        profiles = json.loads(archive.read(request['profile_member']))['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    receipt = {'kind': 'INDEPENDENT_TRANSVERSE_OPERATOR_REVIEW_CONTROLS',
               'native_geometry': [geometry(n) for n in request['N']],
               'native_factors': [factors(n, profiles) for n in request['N']],
               'nonnative_spectral': spectral(), 'before': before,
               'after': {name: digest(ROOT/name) for name in request['pins']},
               'scalar_dependency': 'CONDITIONAL_UNTIL_INDEPENDENT_REVIEW',
               'scientific_admission': False, 'campaign_complete': False,
               'original_functions_imported_or_executed': False,
               'native_ground_fullH_solver_backend': False}
    assert receipt['after'] == before
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps({'completed': True, 'all_native_modes': sum(n**3 for n in request['N']),
                      'receipt': args.output.name, 'native_spectrum_computed': False}))


if __name__ == '__main__':
    main()

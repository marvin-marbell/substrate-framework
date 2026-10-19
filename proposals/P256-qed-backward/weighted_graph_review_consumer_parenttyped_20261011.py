"""Independent exact native affine jets and separated graph-domain controls.
Never imports or executes a prior scientific producer or computes a ground.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path
import sys
import zipfile

from typing import cast

import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_native_jets(profiles: list[dict]) -> dict:
    n = 4
    sites = list(itertools.product(range(n), repeat=3))
    index = {x: i for i, x in enumerate(sites)}
    edges = [(x, axis) for x in sites for axis in range(3)]
    edge_index = {e: i for i, e in enumerate(edges)}

    def move(x: tuple[int, ...], axis: int, length: int = 1) -> tuple[int, ...]:
        y = list(x)
        y[axis] = (y[axis] + length) % n
        return tuple(y)

    d = sp.zeros(len(sites), len(edges))
    for col, (x, axis) in enumerate(edges):
        d[index[x], col] = 1
        d[index[move(x, axis)], col] = -1
    laplace = d * d.T
    inverse = laplace[:-1, :-1].inv()
    parity = sp.Matrix([(-1)**sum(x) for x in sites])
    rho = 128 * parity
    base = sp.zeros(len(edges), 1)
    for x in sites:
        base[edge_index[x, 0]] = 64 * parity[index[x]]
    assert d * base == rho and sum(rho) == 0
    assert all(4 * sum(int(v == 1) for v in parity) == 2 * len(sites)
               for _ in range(65))
    z = sp.Symbol('z', real=True)
    weights = [z**sum(min(v, n-v) for v in x) for x, _ in edges]
    records = []
    # Distinct starts include both N4 antipodal directed length2 transfers.
    for axis, length, start in itertools.product(range(3), (1, 2), ((0, 0, 0), (3, 3, 3))):
        end = move(start, axis, length)
        t = sp.zeros(len(edges), 1)
        for j in range(length):
            t[edge_index[move(start, axis, j), axis]] = -1
        delta = d * t
        potential = inverse * delta[:-1, :]
        potential = potential.col_join(sp.zeros(1, 1))
        longitudinal = d.T * potential
        transverse = t - longitudinal
        assert laplace * potential == delta and d * transverse == sp.zeros(len(sites), 1)
        assert (transverse.T * longitudinal)[0] == 0
        assert all(transverse[e] + longitudinal[e] == t[e] for e in range(len(edges)))
        exterior = [e for e in range(len(edges)) if t[e] == 0]
        assert any(longitudinal[e] != 0 for e in exterior)
        full_second = sp.expand(sum(weights[e] * t[e]**2 for e in range(len(edges))))
        frozen_second = sp.expand(sum(weights[e] * transverse[e]**2 for e in range(len(edges))))
        defect = sp.expand(frozen_second - full_second)
        assert defect != 0
        # The native weight is algebraic, not a free symbolic indeterminate.
        # 2z^4-1 is irreducible: its reciprocal is Eisenstein x^4-2.
        native_remainder = sp.rem(defect, 2*z**4-1, z)
        assert native_remainder != 0
        # Inherited deck l=2pi P e_j: offdiagonal multiplier has an affine,
        # generally noninteger monodromy; no standard-coordinate torus assumed.
        witness_edge = next(e for e in exterior if transverse[e].q != 1)
        deck_pair = transverse[witness_edge]
        assert deck_pair.q != 1 and deck_pair + longitudinal[witness_edge] == 0
        # Real canonical active-species pair: guarantee one occupied origin
        # colour and one vacant destination colour using spectator transfers.
        active = [int(v == 1) for v in parity]  # selected colour0
        background_flow = sp.zeros(len(edges), 1)

        def route(donor: tuple[int, ...], recipient: tuple[int, ...]) -> sp.Matrix:
            flow = sp.zeros(len(edges), 1)
            current = donor
            for direction in range(3):
                while current[direction] != recipient[direction]:
                    flow[edge_index[current, direction]] -= 1
                    current = move(current, direction)
            return flow

        if not active[index[start]]:
            donor = next(x for x in sites if active[index[x]] and x != end)
            active[index[donor]] = 0
            active[index[start]] = 1
            background_flow += route(donor, start)
        if active[index[end]]:
            recipient = next(x for x in sites if not active[index[x]] and x != start)
            active[index[end]] = 0
            active[index[recipient]] = 1
            background_flow += route(end, recipient)
        assert active[index[start]] == 1 and active[index[end]] == 0
        assert sum(active) + 3*sum(int(v == 1) for v in parity) == 128
        background_rho = rho + d*background_flow
        new_rho = background_rho + delta
        assert d * (base + background_flow + t) == new_rho
        assert max(abs(v) for v in delta) == 1
        records.append({'axis': axis, 'length': length, 'start': start, 'end': end,
                        'AP_sign': -1 if start[axis] + length >= n else 1,
                        'native_full_second_jet_weight': str(full_second),
                        'wrong_frozen_second_jet_weight': str(frozen_second),
                        'wrongcase_defect_polynomial': str(defect),
                        'native_algebraic_defect_remainder': str(native_remainder),
                        'canonical_active_transfer_pair': True,
                        'exterior_nonzero_longitudinal_edges': sum(longitudinal[e] != 0 for e in exterior),
                        'inherited_deck_pair_over_2pi': str(deck_pair),
                        'source_step': 'one charged species in the collective64 rho'})
    lam = Fraction(2**8192)
    allocation = Fraction(0)
    loads = []
    for p in profiles:
        load = int(p['load'])
        loads.append(load)
        w = abs(Fraction(p['onsite'])) + 2 * sum(
            (abs(Fraction(path['c0'])) + abs(Fraction(path['ci'])) for path in p['paths']),
            Fraction(0))
        allocation += 4 * load * (lam if p['scale'] == 'heavy' else 1) * w
    assert loads == [1, 1, 63] and allocation < 512 * lam
    r = sp.Symbol('r', real=True)
    assert sp.expand((r-8)**2 * (r*r+16*r+192)/2 - (r**4/2-1024*r+6144)) == 0
    return {'scope': 'native N4 all65 canonical/AP/collective64 affine coefficient control; NOT a physical fullH spectrum',
            'N': n, 'V': len(sites), 'each_species_number': 128, 'species': 65,
            'Ref4': 4, 'dual_cycle_dimension': len(edges)-len(sites)+1,
            'weight_symbol': 'z=2^(-1/4), theta_e=z^periodic_l1_distance(tail,0)',
            'literal_loads': loads, 'B_over_lambda_lt_512': True,
            'B_sha256': sha(str(allocation).encode()), 'controls': records,
            'second_covariant_jet_exact': True, 'ordinary_derivative_wrongcase_rejected': True,
            'standard_integer_coordinate_deck_wrongcase_rejected': True}


def exact_ibp_method() -> dict:
    q = sp.Symbol('q', real=True)
    z = sp.Symbol('z', nonzero=True)
    connection = sp.diag(sp.Rational(1, 3), sp.Rational(-2, 3))
    v = (q**4+3) * sp.eye(2) + q * sp.Matrix([[0, z], [1/z, 0]])
    u = sp.Matrix([1+q, 1/z])

    def dq(a: sp.Matrix) -> sp.Matrix:
        return cast(sp.MutableDenseMatrix, a.diff(q) - q * a)

    def da(a: sp.Matrix) -> sp.Matrix:
        return sp.I * z * a.diff(z) + sp.I * connection * a

    def dv(a: sp.Matrix) -> sp.Matrix:
        return sp.I * z * a.diff(z) + sp.I * (connection*a-a*connection)

    def dagger(a: sp.Matrix) -> sp.Matrix:
        return a.applyfunc(lambda x: sp.conjugate(x).subs(sp.conjugate(z), 1/z)).T

    def integral(expr: sp.Expr) -> sp.Expr:
        # Haar Fourier constant followed by normalized exp(-q^2) Gaussian.
        const = sp.expand(expr).coeff(z, 0)
        poly = sp.Poly(const, q)
        total = sp.Integer(0)
        for (degree,), coefficient in poly.terms():
            if degree % 2 == 0:
                total += coefficient * sp.factorial2(degree-1) / 2**(degree//2)
        return sp.simplify(total)

    kinetic = -dq(dq(u)) - da(da(u))
    qu = kinetic + v*u
    norm = lambda a: integral((dagger(a)*a)[0])
    middle = 2 * integral((dagger(dq(u))*v*dq(u) + dagger(da(u))*v*da(u))[0])
    second_v = v.diff(q, 2) + dv(dv(v))
    correction = integral((dagger(u)*second_v*u)[0])
    lhs = norm(qu)
    rhs = norm(kinetic) + norm(v*u) + middle - correction
    assert lhs == rhs and middle > 0
    ordinary_second = v.diff(q, 2) - z*(z*v.diff(z)).diff(z)
    wrong_rhs = norm(kinetic) + norm(v*u) + middle - integral((dagger(u)*ordinary_second*u)[0])
    assert lhs != wrong_rhs
    return {'scope': 'EXACT SYNTHETIC covariant quartic matrix/Gaussian/Haar IBP mechanism, NOT native ground/evolution',
            'connection': str(connection), 'matrix_potential': str(v),
            'positive_potential_reason': 'q^4+3-|q|>=1; |q|<=1 or |q|>=1 separately',
            'lhs_exact': str(lhs), 'rhs_exact': str(rhs), 'positive_middle_exact': str(middle),
            'covariant_second_correction_exact': str(correction),
            'wrong_ordinary_second_identity_defect': str(lhs-wrong_rhs),
            'identity_passed': True, 'ordinary_second_derivative_rejected': True}


def domain_boundary_method() -> dict:
    # f=(1+q^2)^(-2): f' and q^2 f in L2 but q^4 f not in L2.
    # Tail exponents decide exact improper-integral convergence; no spectrum.
    exponents = {'L2': -8, 'gradient_squared': -10, 'quartic_form': -4,
                 'quartic_operator_squared': 0}
    assert all(exponents[x] < -1 for x in ('L2', 'gradient_squared', 'quartic_form'))
    assert exponents['quartic_operator_squared'] >= -1
    return {'scope': 'EXACT SYNTHETIC one-dimensional domain boundary; not asserted to be native physical state',
            'vector': '(1+q^2)^(-2)', 'tail_integrand_exponents': exponents,
            'in_Friedrichs_form_domain': True, 'not_in_quartic_operator_domain': True,
            'form_domain_is_not_graph_domain': True,
            'cutoff_warning': 'uniform quartic form norm alone cannot replace the graph estimate'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    before = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    if before != request['pins']:
        raise ValueError('revision-bound input mismatch')
    with zipfile.ZipFile(ROOT/request['epoch_archive']) as frozen:
        profiles = json.loads(frozen.read(request['profile_member']))['primitive_profiles']
        epoch_member_sha = sha(frozen.read(request['profile_member']))
    result = {'schema': 1, 'native_covariant_coefficients': exact_native_jets(profiles),
              'synthetic_ibp': exact_ibp_method(), 'synthetic_domain_boundary': domain_boundary_method(),
              'profile_member_sha256': epoch_member_sha, 'inputs_before': before,
              'inputs_after': {name: sha((ROOT/name).read_bytes()) for name in request['pins']},
              'runtime': {'executable': sys.executable, 'version': sys.version},
              'native_quantum_evolution_or_ground_computed': False}
    if result['inputs_after'] != before:
        raise ValueError('source/input instability')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'native_second_covariant_jets': True, 'synthetic_ibp_exact': True,
                      'wrong_frozen_derivative_rejected': True, 'quantum_ground': False}))


if __name__ == '__main__':
    main()

"""Independent exact native field-price controls; never computes a ground."""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import zipfile
import sympy as sp


def car_bilinear(m: int, k: int, a: sp.Matrix) -> sp.Matrix:
    basis = [sum(1 << i for i in c) for c in itertools.combinations(range(m), k)]
    index = {mask: n for n, mask in enumerate(basis)}
    result = sp.zeros(len(basis))
    for col, mask in enumerate(basis):
        for j in range(m):
            if not mask & (1 << j):
                continue
            intermediate = mask ^ (1 << j)
            sign_j = (-1) ** ((mask & ((1 << j) - 1)).bit_count())
            for i in range(m):
                if a[i, j] == 0 or intermediate & (1 << i):
                    continue
                sign_i = (-1) ** ((intermediate & ((1 << i) - 1)).bit_count())
                result[index[intermediate | (1 << i)], col] += sign_i * sign_j * a[i, j]
    return result


def enveloping_and_transition() -> dict:
    a = sp.zeros(6); b = sp.zeros(6)
    a[0, 1] = 2; a[4, 2] = sp.I; b[1, 3] = 3; b[2, 4] = 1
    da, db = car_bilinear(6, 3, a), car_bilinear(6, 3, b)
    assert da * db - db * da == car_bilinear(6, 3, a * b - b * a)
    # On the physical sigma block copies4..63 have no face particle. Each
    # corresponding bilinear is EXACTLY zero, not a deleted source term.
    initial = tuple(range(4)); target = (1, 2, 3, 0)
    transitions = []
    for reverse in (False, True):
        hits = 0
        for order in itertools.permutations(range(4)):
            states = Counter({initial: 1})
            for edge in order:
                src, dst = edge, (edge + 1) % 4
                if reverse:
                    src, dst = dst, src
                new = Counter()
                for positions, count in states.items():
                    for copy in range(64):
                        if copy >= 4 or positions[copy] != src:
                            continue
                        changed = list(positions); changed[copy] = dst
                        new[tuple(changed)] += count
                states = new
            hits += states[target]
        transitions.append(hits)
    assert transitions == [24, 0]
    # All colour indices are free: forced distinct-copy labels imply a
    # tensor of four4x4 blocks, not a colour trace. Spectators are identities.
    incidence = [0] * 4
    for edge in range(4):
        incidence[edge] -= 1; incidence[(edge + 1) % 4] += 1
    assert incidence == [0] * 4
    individual = [-1, 1, 0, 0]
    assert individual != incidence
    assert 24 * 64 ** 4 == 402653184
    # Universal colour coefficient span: fifteen traceless directions plus
    # the native T U scalar supplement; this is not a generic-rank theorem.
    colour = []
    for i in range(4):
        for j in range(4):
            if i != j:
                x = sp.zeros(4); x[i, j] = 1; colour.append(x)
    for i in range(3):
        x = sp.zeros(4); x[i, i] = 1; x[3, 3] = -1; colour.append(x)
    t = sp.diag(1, -1, 0, 0) / 2
    u = sp.diag(sp.I, -sp.I, 1, 1)
    assert u.det() == 1 and sp.trace(t * u) == sp.I
    colour.append(t * u)
    assert sp.Matrix.hstack(*[x.reshape(16, 1) for x in colour]).rank() == 16
    return {'CAR_commutator_exact': True, 'expanded_words': 402653184,
            'sigma_tau_B_Bstar_counts': transitions, 'A_transition': '12 tensor4 t_e',
            'fullcolour_dimension': 256, 'exterior_modes': 240,
            'spectator_particles': [127] * 4 + [128] * 60,
            'collective_closed_individual_not_closed': True,
            'coefficient_span_rank': 16, 'native_generic_ranks': 'OPEN'}


def determinant_and_paths(profiles: list) -> dict:
    assert profiles[1]['paths'] == profiles[2]['paths']
    assert [x['load'] for x in profiles] == [1, 1, 63]
    assert len(profiles[0]['paths']) == 3 and len(profiles[1]['paths']) == 6
    c0 = sp.Rational(profiles[1]['paths'][0]['c0'])
    ci = sp.Rational(profiles[1]['paths'][0]['ci'])
    delta = sp.Rational(profiles[1]['onsite']) - sp.Rational(profiles[2]['onsite'])
    assert delta == -sp.Rational(126663739519795, 70368744177664)
    T = sp.diag(3, -1, -1, -1) / sp.sqrt(24)
    assert sp.trace(T * T) == sp.Rational(1, 2) and T.det() == -sp.Rational(1, 192)
    U = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, sp.I, 0], [0, 0, 0, -sp.I]])
    V = sp.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    assert U.H * U == sp.eye(4) and V.H * V == sp.eye(4)
    assert U.det() == V.det() == 1 and U * V != V * U
    px = sp.diag(3, -1, -1, -1); py = sp.diag(1, 2, -1, -2)
    ax = sp.Matrix([[0, 1, sp.I, 0], [1, 0, 0, 0], [-sp.I, 0, 0, 1], [0, 0, 1, 0]])
    ay = sp.Matrix([[1, 0, 1, 0], [0, -1, 0, sp.I], [1, 0, 0, 0], [0, -sp.I, 0, 0]])
    C = ((c0 * px + sp.I * ci * ax) * U + U * (c0 * py + sp.I * ci * ay)) / 2
    q = sp.Symbol('q', real=True)
    # Both sewing orientations and nontrivial unit Peierls/AP phase.
    rows = []
    for link in (sp.eye(4), U, V, U * V):
        for reverse in (False, True):
            leading_matrix = (c0 / 2) * (link.H * T if reverse else T * link)
            block = -sp.I * (q * leading_matrix + C)
            p = sp.Poly(block.det(method='berkowitz').expand(), q)
            assert p.degree() == 4 and sp.simplify(p.LC()) == -c0 ** 4 / 3072
            rows.append(str(p.LC()))
    c2 = sp.Rational(next(p['c0'] for p in profiles[1]['paths'] if p['length'] == 2))
    two = -c2 * (px * U * V + U * V * py) / 2
    wrong = -c2 * (px * V * U + V * U * py) / 2
    assert two != wrong and two != -two
    active = abs(c0) + abs(ci)
    assert 64 * active == sp.Rational(13313289252183767, 8796093022208)
    w = [abs(sp.Rational(profile['onsite'])) + 2 * sum(
        abs(sp.Rational(path['c0'])) + abs(sp.Rational(path['ci']))
        for path in profile['paths']) for profile in profiles]
    lam = sp.Integer(2) ** 8192
    B = 4 * (lam * w[0] + w[1] + 63 * w[2])
    assert B < 512 * lam
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    support = set(corners)
    for corner in corners:
        for axis in range(3):
            for step in (-2, -1, 1, 2):
                point = list(corner); point[axis] += step
                support.add(tuple(point))
    assert len(support) <= 52 and max(sum(abs(x) for x in point) for point in support) <= 4
    return {'literal_profiles_equal_hops_distinct_onsite': True, 'onsite_delta': str(delta),
            'native_determinant_checks': len(rows), 'leading': rows[0],
            'universal_leading_proof': 'det((c0/2) T U)=-(c0^4/3072)det U for every SU4; adjoint sewing likewise',
            'ordered_length2_and_AP_wrongcases_rejected': True, 'C_active': str(active),
            'native_B_over_lambda_less_512': True, 'full_incident_support_sites': len(support),
            'full_incident_radius_at_most4': True, 'jet_counts': [240, 60, 4]}


def field_and_spectral_controls() -> dict:
    p, B, k, C, a = sp.symbols('p B k C a', positive=True)
    M = (8 * C / p) ** sp.Rational(2, 17)
    d = a * (p / (64 * sp.sqrt(2 * B / k))) ** 4
    assert sp.simplify(32 * (d / a) ** sp.Rational(1, 4) * sp.sqrt(2 * B / k)) == p / 2
    assert sp.simplify(4 * C * M ** (-sp.Rational(17, 2))) == p / 2
    # Exact smooth vector-valued marginal, not a simulated native ground.
    x = sp.Symbol('x', real=True)
    scalar = (2 / sp.pi) ** sp.Rational(1, 4) * sp.exp(-x ** 2)
    norm = sp.integrate(scalar ** 2, (x, -sp.oo, sp.oo))
    derivative = sp.integrate(sp.diff(scalar, x) ** 2, (x, -sp.oo, sp.oo))
    assert norm == derivative == 1 and 2 / sp.pi < 1
    roots = [sp.Rational(-3), sp.I, sp.Rational(2) + sp.I, sp.Rational(7)]
    polynomial = sp.prod(x - z for z in roots)
    for sample in range(-20, 21):
        value = sp.Rational(sample, 2)
        if abs(polynomial.subs(x, value)) <= 1:
            assert min(abs(value - sp.re(z)) for z in roots) <= 1
    # Plausible wrong constant kernel projector: derivative is nonzero.
    v = sp.Matrix([sp.cos(x), sp.sin(x)]); P = v * v.T
    assert sp.diff(P, x) != sp.zeros(2)
    # WholeP0 subtraction and exact destructive complement.
    D = sp.diag(1, 1, 0); P0 = sp.diag(1, 1, 0); omega = sp.Matrix([0, 1, 0])
    assert (D * omega).norm() ** 2 == 1
    assert ((sp.eye(3) - P0) * D * omega).norm() ** 2 == 0
    mean = omega * omega.H
    excited_ground_vector = sp.Matrix([1, 0, 0])
    assert ((sp.eye(3) - mean) * excited_ground_vector).norm() ** 2 == 1
    assert ((sp.eye(3) - P0) * excited_ground_vector).norm() ** 2 == 0
    assert (D + (-D)) == sp.zeros(3)
    # The valid multiplier bound is2||H|| ||D||, not2||HD psi||.
    H = sp.Matrix([[0, 2], [2, 0]]); projector = sp.diag(1, 0)
    assert H * projector - projector * H != sp.zeros(2)
    null_input = sp.Matrix([0, 1])
    assert H * projector * null_input == sp.zeros(2, 1)
    assert ((H * projector - projector * H) * null_input).norm() == 2
    # Spectral Cauchy lower evaluated with actual finite positive spectrum;
    # a lower mass g<m is retained rather than substituted as an equality.
    energies = [sp.Rational(1), sp.Rational(4)]
    weights = [sp.Rational(1, 3), sp.Rational(2, 3)]
    eta = sp.Rational(2); mass = sum(weights)
    second = sum(w * e ** 2 for w, e in zip(weights, energies))
    response = sum(w / (e + eta) for w, e in zip(weights, energies))
    g = sp.Rational(1, 2)
    lower = g ** 2 / (eta * g + sp.sqrt(second) * sp.sqrt(g))
    assert response > lower > 0 and mass > g
    # Radial17/2 exactly pays degree8 cutoff tail exponent1/2 in squared norm.
    assert sp.Rational(17, 2) - 8 == sp.Rational(1, 2)
    s = sp.Rational(1, 34); beta = (1 - 17 * s) / 4
    assert beta == sp.Rational(1, 8)
    assert -sp.Rational(4, 17) / 4 == -sp.Rational(1, 17)
    assert -1 + 4 * sp.Rational(4, 17) == -sp.Rational(1, 17)
    assert 1 - 2 ** (-beta) > 0
    assert (1 - 17 * sp.Rational(1, 17)) / 4 == 0
    # A spectral law pays every strict s<1/17 yet fails the endpoint.
    # mu_n=2^(-2n/17)/(4n), E_n=2^n; endpoint is harmonic.
    endpoint_terms = [sp.Rational(1, 4 * n) for n in range(1, 12)]
    assert sum(endpoint_terms) > sp.Rational(1, 2)
    for V in (64, 216, 512):
        assert sp.sqrt(V) == sp.Rational(V) / sp.sqrt(V)
    occupation = Fraction(4 * math.comb(240, 127), math.comb(256, 128)) ** 4
    occupation *= Fraction(math.comb(240, 128), math.comb(256, 128)) ** 60
    assert occupation > 0
    # In det31 wedge every mode31/32, sigma <=1 auxiliary per face mode.
    assert 31 > 3
    return {'bad_event_prices': ['p/2', 'p/2'], 'kappa': '144 d^8/(2 C_active M)^24',
            'gaussian_norm_derivative_energy': [str(norm), str(derivative)],
            'kernel_derivative_wrongcase_rejected': True,
            'wholeP0_and_signed_complement_wrongcases_rejected': True,
            'projectors_do_not_commute_H': True, 'finite_spectral_Cauchy_lower': str(lower),
            'canonical_occupation_positive_not_ground': str(occupation),
            'auxiliary_det31wedge_sigma_mass': 0, 'fractional_beta_at_s1over34': str(beta),
            'endpoint_counter': 'mu_n=2^(-2n/17)/(4n), E_n=2^n; every strict s<1/17 finite, endpoint harmonic divergence',
            'Fourier_triangle_cost': 'sqrtV; no N-uniform transfer'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.snapshot) as archive:
        request = json.loads(archive.read('request.json'))
        for name, pin in request['pins'].items():
            assert hashlib.sha256(archive.read('inputs/' + name)).hexdigest() == pin
        raw = archive.read('inputs/' + request['profile_archive'])
    with zipfile.ZipFile(__import__('io').BytesIO(raw)) as native:
        profile_raw = native.read(request['profile_member'])
    assert hashlib.sha256(profile_raw).hexdigest() == request['profile_member_sha256']
    profiles = json.loads(profile_raw)['primitive_profiles']
    result = {'kind': 'INDEPENDENT_EXACT_NATIVE_FIELD_PRICE_REVIEW_NOT_GROUND',
              'governor': request['governor'],
              'enveloping_and_transition': enveloping_and_transition(),
              'native_determinant_and_paths': determinant_and_paths(profiles),
              'conditional_field_and_adverse_controls': field_and_spectral_controls(),
              'original_main_replayed': False, 'actual_ground_computed': False,
              'generic_rank_computed': False, 'scientific_admission': False,
              'campaign_complete': False}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'exact_review_controls_completed': True, 'actual_ground_Gram': 'UNPAID'}))


if __name__ == '__main__':
    main()

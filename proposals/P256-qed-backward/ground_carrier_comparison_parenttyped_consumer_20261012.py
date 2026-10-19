"""Exact full-law comparison construction and separated finite algebra controls; no ground solver."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import zipfile
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast

import sympy as s

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)
FACES = ((0, 1, 2, 3), (1, 4, 5, 2))


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def hop(mask: int, target: int, source: int):
    if not mask & (1 << source):
        return None
    sign = (-1) ** ((mask & ((1 << source) - 1)).bit_count())
    mask ^= 1 << source
    if mask & (1 << target):
        return None
    sign *= (-1) ** ((mask & ((1 << target) - 1)).bit_count())
    return mask | (1 << target), sign


def complete_canonical_count(profiles: list[dict]) -> dict:
    V, local_sites = 64, 6
    by_local_total = Counter()
    opposite_rank = 0
    for pattern in itertools.product(range(5), repeat=local_sites):
        local_number = sum(pattern)
        internal = math.prod(math.comb(4, n) for n in pattern)
        spectator_number = 2 * V - local_number
        assert 0 <= spectator_number <= 4 * (V - local_sites)
        rank = internal * math.comb(4 * (V - local_sites), spectator_number)
        by_local_total[local_number] += rank
        if pattern[:4] == (4, 0, 4, 0):
            opposite_rank += rank
    dimension = math.comb(4 * V, 2 * V)
    assert sum(by_local_total.values()) == dimension
    assert 0 < opposite_rank < dimension
    roles = [{'id': 0, 'profile': profiles[0], 'canonical_particles': 128, 'scale': '-2^8192'}]
    roles += [{'id': i, 'profile': profiles[1 if i == 1 else 2],
               'canonical_particles': 128, 'scale': '1'} for i in range(1, 65)]
    assert len({r['id'] for r in roles}) == 65
    assert len({r['profile']['onsite'] for r in roles}) == 3
    return {'N': 4, 'V': V, 'all_local_total_patterns_per_role': 5 ** 6,
            'all_local_internal_exterior_modes_per_role': 2 ** 24,
            'all65_distinct_role_product_pattern_count': str(5 ** (6 * 65)),
            'all65_distinct_roles': roles, 'one_role_canonical_dimension': str(dimension),
            'all65_canonical_dimension_sha256': sha(str(dimension ** 65).encode()),
            'local_total_sector_dimensions': {str(k): str(v) for k, v in sorted(by_local_total.items())},
            'opposite_branch_rank': str(opposite_rank), 'signed_complement_rank': str(dimension - opposite_rank),
            'local_rank_counting_weight_NOT_ground_probability': str(F(opposite_rank, dimension)),
            'canonical_empty_local_spectators': 128, 'canonical_full_local_spectators': 104,
            'proof': 'Vandermonde identity, all colour subspaces counted and all65 tensor factors retained.',
            'actualground_branch_weights': 'UNPROVED; canonical dimension ratios never substituted.'}


def loop_action(mask: int, face: tuple[int, ...], copies: int = 1, samecopy: bool = False) -> dict[int, int]:
    result = Counter()
    edges = [(face[(j + 1) % 4], face[j]) for j in range(4)]
    for order in itertools.permutations(range(4)):
        for assignment in itertools.product(range(copies), repeat=4):
            if samecopy and len(set(assignment)) != 1:
                continue
            current, coefficient = mask, 1
            for e, copy in zip(order, assignment):
                target, source = edges[e]
                moved = hop(current, 6 * copy + target, 6 * copy + source)
                if moved is None:
                    coefficient = 0
                    break
                current, sign = moved
                coefficient *= sign
            if coefficient:
                result[current] += coefficient
    return {m: c for m, c in result.items() if c}


def native_adverse_complete_canonical(profiles: list[dict]) -> dict:
    # The finite witness is an exact native diagonal-Phi section, NOT a selected ground.
    # Phi0=diag(3,-1,-1,-1), all spatial Phi=0,U=I,A=0; colour0 populated.
    # The other three colours and all63 auxiliary local regions are empty on this witness.
    # Their transfers annihilate the witness exactly; each factor still has128 particles outside.
    c0 = F(next(p for p in profiles[1]['paths'] if p['length'] == 1)['c0'])
    amplitude = (3 * c0) ** 4
    signed = []
    sectors = Counter()
    for mask in range(64):
        counts = []
        for face in FACES:
            action = loop_action(mask, face)
            assert all(m == mask for m in action)
            counts.append(action.get(mask, 0))
        cross = counts[0] * counts[1]
        signed.append({'mask': mask, 'local_n': mask.bit_count(), 'face_coefficients': counts,
                       'adjacent_product_CAR_count': cross,
                       'outside_spectator_number_per_active_role': 128 - mask.bit_count()})
        sectors[(mask.bit_count(), cross)] += 1
    positive = [r for r in signed if r['adjacent_product_CAR_count'] > 0]
    negative = [r for r in signed if r['adjacent_product_CAR_count'] < 0]
    # Each selected colour occupation has a native canonical spectator completion.
    assert all(0 <= r['outside_spectator_number_per_active_role'] <= 232 for r in signed)
    assert positive and negative
    # TrF=0 would erase these nonzero native occupation coefficients: spintrace is wrong.
    assert 3 - 1 - 1 - 1 == 0 and amplitude > 0
    # All local one-colour ranks0..6 exercised; internal colours accounted in exhaustive rank proof.
    assert {r['local_n'] for r in signed} == set(range(7))
    # Non-real phase distinguishes ordered complex adjacent cross from absolute-value pairing.
    ordered_phase = s.exp(s.I * s.pi / 3)
    signed_cross = s.simplify(negative[0]['adjacent_product_CAR_count'] * ordered_phase)
    assert s.im(signed_cross) != 0
    mixed_witness = None
    for a in range(64):
        for b in range(64):
            mask = a | (b << 6)
            full = loop_action(mask, FACES[0], copies=2)
            same = loop_action(mask, FACES[0], copies=2, samecopy=True)
            if full != same and any(m != mask for m in full):
                mixed_witness = {'active_mask': a, 'auxiliary2_mask': b,
                                 'full_all_assignments': {str(m): c for m, c in full.items()},
                                 'wrong_samecopy': {str(m): c for m, c in same.items()}}
                break
        if mixed_witness:
            break
    assert mixed_witness
    # Globally each copy has128 particles: fill all unlisted particles among232 external modes.
    mixed_witness['canonical_spectators'] = [128 - mixed_witness['active_mask'].bit_count(),
                                            128 - mixed_witness['auxiliary2_mask'].bit_count()]
    assert all(0 <= n <= 232 for n in mixed_witness['canonical_spectators'])
    return {'native_allcolour_diagonal_section': {'Phi0': [3, -1, -1, -1], 'Phi123': '0',
                                                'SU4_links': 'I', 'U1_closed_phases': '0'},
            'source_4leg_amplitude': str(amplitude), 'all64_local_colour0_masks': signed,
            'positive_witness': positive[0], 'adverse_signed_complement_witness': negative[0],
            'ordered_complex_adverse_cross_unit_amplitude': str(signed_cross),
            'mixed_copy_offdiagonal_projector_witness': mixed_witness,
            'all65_canonical_completion': 'Neutral and every charged copy128; remaining62 auxiliaries plus neutral retained as external spectators. Empty local transfer action is exact, not factor deletion.',
            'ground_expectation_computed': False,
            'decision': 'Complete-canonical coefficient sign gate fails; no native ground sign/order inferred.'}


def native_complete_body_jets(profiles: list[dict]) -> dict:
    I = s.I
    generators = []
    for a in range(4):
        for b in range(a + 1, 4):
            re, im = s.zeros(4), s.zeros(4)
            re[a, b] = re[b, a] = s.Rational(1, 2)
            im[a, b], im[b, a] = -I / 2, I / 2
            generators += [re, im]
    for n in range(1, 4):
        generators.append(s.diag(*([1] * n + [-n] + [0] * (3 - n))) / s.sqrt(2 * n * (n + 1)))
    assert len(generators) == 15
    casimir = s.simplify(cast(Any, sum((t * t for t in generators), s.zeros(4))))
    assert casimir == s.eye(4) * s.Rational(15, 8)
    phi = [generators[k] for k in (0, 3, 6, 9)]
    U = s.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    V = s.diag(I, -I, 1, 1)
    q = s.symbols('q', real=True)
    path_checks, phi_checks, haar_checks = 0, 0, 0
    onsite_nonzero = 0
    for profile in profiles:
        for path in profile['paths']:
            c0, ci = s.Rational(path['c0']), I * s.Rational(path['ci'])
            axis = path['axis'] + 1
            Fx, Fy = c0 * phi[0] + ci * phi[axis], c0 * phi[1] + ci * phi[(axis + 1) % 4]
            W = U if path['length'] == 1 else U * V
            leg = (Fx * W + W * Fy) / 2
            for mu in range(4):
                for t in generators:
                    inserted = (c0 if mu == 0 else ci if mu == axis else 0) * t * W / 2
                    varied = leg + q * inserted
                    assert s.diff(varied, q) == inserted and s.diff(varied, q, 2) == s.zeros(4)
                    phi_checks += 1
            for t in generators:
                d = (Fx * (I * t * W) + (I * t * W) * Fy) / 2
                dd = (Fx * (-t * t * W) + (-t * t * W) * Fy) / 2
                jet = s.eye(4) + q * I * t - q * q * t * t / 2
                varied = (Fx * jet * W + jet * W * Fy) / 2
                assert s.simplify(s.diff(varied, q).subs(q, 0) - d) == s.zeros(4)
                assert s.simplify(s.diff(varied, q, 2).subs(q, 0) - dd) == s.zeros(4)
                haar_checks += 1
            onsite = s.Rational(profile['onsite']) * phi[0]
            if onsite * leg - leg * onsite != s.zeros(4):
                onsite_nonzero += 1
            if path['length'] == 2:
                assert I * U * generators[0] * V != I * generators[0] * U * V
            path_checks += 1
    assert onsite_nonzero > 0 and [len(p['paths']) for p in profiles] == [3, 6, 6]
    # Differentiate ordered fourleg products before summing; cross jets are indispensable.
    legs = [U, V, U * V, V * U]
    firsts = [generators[k] for k in (0, 3, 6, 9)]
    word_checks = 0
    for order in itertools.permutations(range(4)):
        polynomial = [s.eye(4)]
        for e in order:
            nxt = [s.zeros(4) for _ in range(len(polynomial) + 1)]
            for degree, matrix in enumerate(polynomial):
                nxt[degree] += legs[e] * matrix
                nxt[degree + 1] += firsts[e] * matrix
            polynomial = nxt
        dword, ddword = s.zeros(4), s.zeros(4)
        for slot in range(4):
            term = s.eye(4)
            for position, e in enumerate(order):
                term = (firsts[e] if position == slot else legs[e]) * term
            dword += term
        for p, k in itertools.combinations(range(4), 2):
            term = s.eye(4)
            for position, e in enumerate(order):
                term = (firsts[e] if position in (p, k) else legs[e]) * term
            ddword += 2 * term
        assert s.simplify(dword - polynomial[1]) == s.zeros(4)
        assert s.simplify(ddword - 2 * polynomial[2]) == s.zeros(4)
        word_checks += 1
    # Exact adjacent native incidence and raw-electric right-derivative control.
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 5), (5, 2)]
    incidence = s.zeros(6, 7)
    for e, (tail, head) in enumerate(edges):
        incidence[tail, e], incidence[head, e] = -1, 1
    ca, cb = s.Matrix([1, 1, 1, 1, 0, 0, 0]), s.Matrix([0, -1, 0, 0, 1, 1, 1])
    assert incidence * ca == incidence * cb == s.zeros(6, 1)
    assert (ca.T * cb)[0] == -1 and (ca.T * ca)[0] == (cb.T * cb)[0] == 4
    m = s.symbols('m', real=True)
    electric_action = sum(((m + c) ** 2 - m ** 2 for c in ca), s.Integer(0))
    assert s.expand(electric_action - 4 - 2 * sum(cast(Any, ca)) * m) == 0
    assert s.expand(electric_action - 4) != 0
    # Full covariant source action evaluated in a noncommuting exact differential core.
    # This is a method control, not a discretized Phi/Haar/ground quantum Hamiltonian.
    x = s.symbols('x', real=True)
    X = s.Matrix([[x, I * x ** 2], [-I * x ** 2, -x]])
    connection = I * s.Matrix([[0, 1], [1, 0]])
    psi = s.Matrix([1 + x ** 2, I * x])
    def D(vector):
        return vector.diff(x) + connection * vector
    dX = X.diff(x) + connection * X - X * connection
    ddX = dX.diff(x) + connection * dX - dX * connection
    kinetic_lhs = -D(D(X * psi)) + X * D(D(psi))
    kinetic_rhs = -ddX * psi - 2 * dX * D(psi)
    assert s.simplify(kinetic_lhs - kinetic_rhs) == s.zeros(2, 1)
    naive = -X.diff(x, 2) * psi - 2 * X.diff(x) * psi.diff(x)
    assert s.simplify(kinetic_lhs - naive) != s.zeros(2, 1)
    # Every matter role carries its literal onsite and all native path coefficients in free algebra.
    # These are full-body symbolic commutator terms; no finite field expectation is used as ground.
    # h_path labels below denote the complete directed-plus-adjoint native path block,
    # with endpoint averaging and the native i/AP/ordered Wilson factor inside the label.
    role_terms = []
    for role in range(65):
        profile = profiles[0 if role == 0 else 1 if role == 1 else 2]
        symbols = [s.Symbol(f'h{role}_onsite', commutative=False)]
        coefficients = [s.Rational(profile['onsite'])]
        for path in profile['paths']:
            symbols.extend([s.Symbol(f'h{role}_{path["axis"]}_{path["length"]}_c0', commutative=False),
                            s.Symbol(f'h{role}_{path["axis"]}_{path["length"]}_ci', commutative=False)])
            coefficients.extend([s.Rational(path['c0']), s.Rational(path['ci'])])
        h = sum(c * t for c, t in zip(coefficients, symbols))
        A = s.Symbol('A_aM', commutative=False)
        commutator = s.expand(h * A - A * h)
        role_terms.append({'role': role, 'scale': '-2^8192' if role == 0 else '1',
                           'h_symbolic': str(h), 'commutator_symbolic': '0 (independent neutral CAR factor)' if role == 0 else str(commutator)})
    # Scalar closed Poisson source exact mixed double commutator has sign-changing integrand.
    r, theta = s.symbols('r theta', real=True)
    k = (s.cos(theta) - r) / (1 - 2 * r * s.cos(theta) + r ** 2)
    derivative = -(1 - r ** 2) * s.sin(theta) / (1 - 2 * r * s.cos(theta) + r ** 2) ** 2
    assert s.simplify(s.diff(k, theta) - derivative) == 0
    r0 = s.Rational(1, 2)
    values = [s.simplify(k.subs({r: r0, theta: t})) for t in (0, s.pi)]
    assert values == [2, -s.Rational(2, 3)]
    return {'primitive_path_profiles': path_checks, 'all60_Phi_endpoint_jet_checks': phi_checks,
            'all15_Haar_first_second_jet_checks': haar_checks, 'full65_symbolic_matter_body': role_terms,
            'native_noncommuting_onsite_controls': onsite_nonzero,
            'all24_ordered_cross_jet_controls': word_checks, 'adjacent_native_incidence_pairing': '-1',
            'fullE_right_first_derivative_omission_rejected': True,
            'covariant_Berry_kinetic_identity': True, 'ordinary_derivative_shortcut_rejected': True,
            'fullE_affine_scope': 'Closed mixed words require tT+tL=t; collective rho,Ref4 shared once, not independently electric copies.',
            'Poisson_k_r_at_0_pi_rhalf': [str(t) for t in values],
            'adjacent_double_commutator_conditional_integrands': ['-128*2*2=-512', '-128*2*(-2/3)=512/3'],
            'full_normalized_quantum_ground_integral': 'UNPROVED; these opposite conditional signs do not choose ground sign.'}


def normalized_signed_response() -> dict:
    # Exact finite spectral method control, deliberately NOT native H or native ground.
    L = s.diag(0, 0, 2, 5)
    P0 = s.diag(1, 1, 0, 0)
    Q = s.eye(4) - P0
    omega = s.Matrix([1, 0, 0, 0])
    A = s.Matrix([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    B = s.Matrix([[0, 0, -1 - s.I, 2], [0, 0, 0, 0], [-1 + s.I, 0, 0, 0], [2, 0, 0, 0]])
    assert A == A.H and B == B.H
    X = s.Matrix.hstack(Q * A * omega, Q * B * omega)
    wrong_mean = s.Matrix.hstack(A * omega, B * omega)
    assert wrong_mean.H * wrong_mean != X.H * X
    G = X.H * X
    eta = s.Rational(1, 3)
    R = Q * (L + eta * s.eye(4)).inv() * Q
    Y = R * X
    T, U, K = X.H * R * X, Y.H * Y, Y.H * L * Y
    assert s.simplify(K - T + eta * U) == s.zeros(2)
    M = X.H * (L + eta * s.eye(4)) * X
    lower = s.simplify(G * M.inv() * G)
    # Here X spans the two positive spectral directions, so bound is exactly saturated.
    assert s.simplify(T - lower) == s.zeros(2)
    assert s.re(T[0, 1]) < 0 and s.im(T[0, 1]) != 0 and s.simplify(T.det()) > 0
    j = s.symbols('j', real=True)
    y = Y[:, 0]
    # A shifted by scalar creates a nonzero ground mean and nonzero cubic source term.
    source = A + 2 * s.eye(4)
    psi = omega + j * y
    denominator = (psi.H * psi)[0]
    exact = s.simplify((psi.H * (L - j * source) * psi)[0] / denominator)
    a = (omega.H * source * omega)[0]
    cubic = (y.H * source * y)[0]
    formula = (j ** 2 * K[0, 0] - j * a - 2 * j ** 2 * T[0, 0] - j ** 3 * cubic) / (1 + j ** 2 * U[0, 0])
    assert s.simplify(exact - formula) == 0 and cubic != 0
    wrong = j ** 2 * K[0, 0] - j * a - 2 * j ** 2 * T[0, 0]
    assert s.simplify(exact - wrong) != 0
    # Ground-directed first variation includes complete form; energy never infers source order.
    za, zb = s.symbols('za zb', real=True)
    normalized = omega + za * A * omega + zb * B * omega
    norm = (normalized.H * normalized)[0]
    energy = s.simplify((normalized.H * L * normalized)[0] / norm)
    assert s.simplify(energy - (s.Matrix([za, zb]).T * (X.H * L * X) * s.Matrix([za, zb]))[0] / norm) == 0
    def record(matrix):
        return [[str(matrix[i, j]) for j in range(matrix.cols)] for i in range(matrix.rows)]
    return {'method_control_NOT_native_ground': True, 'L': record(L), 'wholeP0_rank': 2,
            'actual_method_Gram': record(G), 'eta': str(eta), 'resolvent_response_T': record(T),
            'complete_energy_M_eta': record(M), 'nontrivial_variational_lower': record(lower),
            'positive_response_matrix_negative_complex_adjacent_entry': str(T[0, 1]),
            'normalized_complete_source_energy': str(exact), 'retained_cubic_source_expectation': str(cubic),
            'mean_only_P0_shortcut_rejected': True, 'denominator_and_cubic_omission_rejected': True,
            'deformation_energy': str(energy),
            'physical1_native_gate': 'OPEN: actualground G/branch/coherence + N-uniform spectral/spatial scale not supplied by this method example.'}


def native_symbolic_ground_deformation(profiles: list[dict]) -> dict:
    """Evaluate the exact full-H source action in its actualground moment interface.

    The uncomputed objects are explicitly quantum-ground functionals, not decimal
    field-point stand-ins. This consumes the full body rather than a local h.
    """
    channels = []
    for face in ('a', 'b'):
        kinetic = []
        for site in range(6):
            for mu in range(4):
                for generator in range(15):
                    label = f'Phi[{site},{mu},{generator}]'
                    kinetic.append({'family': 'Phi', 'direction': label, 'coefficient': '-4chi',
                                    'action': f'd2_{label}(A_{face}M)+2*d_{label}(A_{face}M)*nabla_{label}'})
        for link in range(7):
            for generator in range(15):
                label = f'Haar[{link},{generator}]'
                kinetic.append({'family': 'Haar', 'direction': label, 'coefficient': '-3chi/20',
                                'action': f'd2_{label}(A_{face}M)+2*d_{label}(A_{face}M)*nabla_{label}'})
            label = f'fullE[{link}]'
            kinetic.append({'family': 'fullE', 'direction': label, 'coefficient': '64',
                            'action': f'[E_{link},[E_{link},A_{face}M]]+2*[E_{link},A_{face}M]*E_{link}'})
        matter = []
        for role in range(65):
            profile = profiles[0 if role == 0 else 1 if role == 1 else 2]
            matter.append({'role': role, 'scale': '-Lambda' if role == 0 else '1',
                           'onsite': profile['onsite'], 'all_native_directed_plus_adjoint_paths': profile['paths'],
                           'action': '0, exact independent neutral CAR commutation' if role == 0 else
                                     f'[dGamma_{role}(h_{role}),A_{face}M]',
                           'complement_action': f'A_{face}M*[dGamma_{role}(h_{role}),P_alpha]'})
        assert len(kinetic) == 360 + 105 + 7 and len(matter) == 65
        channels.append({'face': face, 'complete_kinetic_action': kinetic, 'complete_matter_action': matter,
                         'scalar_Jordan_alignment_plaquette': 'Commutator0; retained in Omega,H,Eglob and all ground functionals.',
                         'cutoff_and_word_jets': 'dchi*A+chi*dA and d2chi*A+2*dchi*dA+chi*d2A; all24 ordered fourleg slot/cross derivatives, all64 independent species; no CAR factor commute.'})
    # Exact complex twoface normalized deformation, not a norm estimate.
    za, zb, zc, ma, mb, ga, gb, gu, gv, ka, kb, ku, kv = s.symbols(
        'za zb zc mean_a mean_b Gaa Gbb ReGab ImGab Kaa Kbb ReKab ImKab', real=True)
    z = s.Matrix([za, zb + s.I * zc])
    G = s.Matrix([[ga, gu + s.I * gv], [gu - s.I * gv, gb]])
    K = s.Matrix([[ka, ku + s.I * kv], [ku - s.I * kv, kb]])
    norm = s.expand(1 + 2 * za * ma + 2 * zb * mb + (z.H * G * z)[0])
    numerator = s.expand((z.H * K * z)[0])
    assert numerator.coeff(kv) == -2 * za * zc
    assert norm.coeff(gv) == -2 * za * zc
    eta = s.symbols('eta', positive=True)
    g, m = s.symbols('actualground_Gram actualground_energy', positive=True)
    conditional_lower = g ** 2 / (m + eta * g)
    assert conditional_lower > 0
    return {'representation': 'Exact actualground symbolic functional consumer; no quantum-ground moments assigned numerical values.',
            'primitive_constants': {'chi': '2^-4096', 'Lambda': '2^8192', 'F': '2^10240', 'a': '1/64'},
            'twoface_full_body_source_actions': channels,
            'complete_energy_moment_interface': 'Kij=omega(A_iM*[H,A_jM]), four ordered complex entries; omega is normalized globally minimizing all65 quantum ground, not conditional field point.',
            'normalization_moment_interface': 'mean_i=omega(A_iM),Gij=omega(A_iM*A_jM); not wholeP0-projected until Q is explicitly applied.',
            'actual_normalization_polynomial': str(norm), 'actual_complete_energy_polynomial': str(numerator),
            'exact_complete_form_energy_excess': str(numerator / norm),
            'imaginary_cross_coefficient_in_both_form_and_norm': str(-2 * za * zc),
            'second_route_scalar_actualground_variational_lower': str(conditional_lower),
            'second_route_complete_domain': 'Y_eta=(SAME L+eta)^-1 Q[A_M] belongs to D(SAME L); exact normalized source deformation in proof.',
            'unknown_ground_inputs': ['All full-body ground functionals above', 'Individual-species projector weights and interferences', 'wholeP0 projected Gram and N-uniform spectral/spatial lower'],
            'native_moment_values_invented': False, 'native_physical1_sign_gate': 'NOT PROVED'}


def inherited_ground_budget_and_support_attempt(profiles: list[dict]) -> dict:
    # SAME bare CAR-loop quadratures, not a scalar flux replacement.
    lam, chi, fast = F(2 ** 8192), F(1, 2 ** 4096), F(2 ** 10240)
    R0 = 60 * chi + 11192 * lam + F(129, 10) * fast + 262144
    B0 = 33 * (1 + 10 * R0 * 4096) + 16384 * lam * 4096
    beta = B0 / 32  # raw fullE budget and theta>=1/2 on both face supports
    ca, cb = s.Matrix([1, 1, 1, 1, 0, 0, 0]), s.Matrix([0, -1, 0, 0, 1, 1, 1])
    J = s.Matrix.hstack(ca, cb).T * s.Matrix.hstack(ca, cb)
    assert J == s.Matrix([[4, -1], [-1, 4]])
    assert sorted(J.eigenvals()) == [3, 5]
    mu_a, mu_b, u_a, u_b = s.symbols('ground_mean_Ca ground_mean_Cb ua ub', real=True)
    vector = u_a * mu_a * ca + u_b * mu_b * cb
    numerator = s.expand((vector.T * vector)[0])
    assert numerator == 4 * mu_a ** 2 * u_a ** 2 - 2 * mu_a * mu_b * u_a * u_b + 4 * mu_b ** 2 * u_b ** 2
    # The actualground Robertson/Ward lower is nontrivial only after signed means are paid.
    assert numerator.subs({mu_a: 0, mu_b: 0}) == 0
    assert F(1, 4) / beta == 8 / B0
    # Exact native canonical source-kernel control: all65 copies full/empty z planes.
    stripes = []
    for shift in range(4):
        filled = {(shift + k) % 4 for k in (0, 1)}
        mask = sum(1 << (4 * (x + 4 * y + 16 * z) + colour)
                   for z in filled for y in range(4) for x in range(4) for colour in range(4))
        assert mask.bit_count() == 128
        # A vertical NN boundary hop creates a unique outgoing CAR pattern.
        outgoing_plane = (shift + 1) % 4
        incoming_plane = (shift + 2) % 4
        moved = hop(mask, 4 * (16 * incoming_plane), 4 * (16 * outgoing_plane))
        assert moved is not None and moved[0].bit_count() == 128
        state = tuple(moved[0] if role == 1 else mask for role in range(65))
        stripes.append({'shift': shift, 'mask': mask, 'outgoing': state, 'CAR_sign': moved[1]})
    assert len({r['outgoing'] for r in stripes}) == 4
    # Each local face is entirely occupied or empty, hence its first hop is blocked.
    assert all(not loop_action(mask, face) for mask in (0, 63) for face in FACES)
    # After one vertical hop, a face on the occupied plane has one colour0 hole.
    opened = loop_action(62, FACES[0])
    assert opened.get(62, 0) == 1
    c0 = F(next(p for p in profiles[1]['paths'] if p['axis'] == 2 and p['length'] == 1)['c0'])
    assert 3 * c0 != 0
    # Exact collective affine Gauss cost for unshifted all64 charged stripe.
    rho = [128, 128, -128, -128]
    flow = [0, 128, 0, -128]
    assert [flow[z] - flow[(z - 1) % 4] for z in range(4)] == rho
    fullE_energy = 64 * 16 * sum(t * t for t in flow)
    assert fullE_energy == 33554432 and fullE_energy < B0
    return {'same_native_bare_source': 'A_fM=(B_fM+B_fM*)/2,C_fM=(B_fM-B_fM*)/(2i), no flux source substituted.',
            'ground_Q_budget_B0_sha256': sha(str(B0).encode()),
            'budget_inverse_coefficient_8_over_B0': {'numerator_bits': (8 / B0).numerator.bit_length(),
                                                   'denominator_bits': (8 / B0).denominator.bit_length()},
            'native_closed_E_wholeP0_zero_dependency': 'magnetic_ward_review_proof_20261012.json; SAME limiting wholeP0, no finiteP0 identification.',
            'actualground_uncertainty_lower': 'Re G_A >= (8/B0) diag(omega(CaM),omega(CbM))*[[4,-1],[-1,4]]*diag(omega(CaM),omega(CbM)); for C swap signed A means.',
            'exact_lower_numerator': str(numerator), 'source_specific_gate_at_unpaid_means': '0, not a positive physical1 lower',
            'adjacent_incidence_eigenvalues': [3, 5],
            'canonical_translation_mean_n_each_role_site': '2, from equal mixture of4 stripe translates; not ground probability',
            'source_kernel_stripe_ground_support_result': 'Actual finite ground cannot have weight1 in span of these4 all65 stripe patterns: a unique active NN outgoing CAR block has a nonzero Phi-linear coefficient almost everywhere. Full kinetic/scalar/affine terms do not change these CAR patterns. L2 Phi ground cannot be supported on its polynomial zero set.',
            'strict_actualground_complement_weight': '1-omega(P_stripe)>0 at finiteN4; no quantitative uniform lower and not opposite-pair weight.',
            'native_stationarity_discriminant': {'stripe_loop_action': 0, 'after_one_vertical_hop_loop_count': 1,
                                                'literal_active_NN_colour0_coefficient_at_native_section': str(3 * c0),
                                                'all65_outgoing_pattern_digests': [sha(str(r['outgoing']).encode()) for r in stripes]},
            'integer_collective_Gauss_flow': flow, 'collective_rho': rho, 'Ref4_background_contribution': 0,
            'once_fullE_energy': fullE_energy,
            'scope': 'Budget/Ward lower is on inherited actual ground. Stripe arithmetic is native canonical coefficient/symmetry control, explicitly not a physical ground or a complete-Q-budget counterexample. Stationarity excludes pure stripe support, but complement may still lie in bare-loop kernel.',
            'unpaid_source_specific_quantity': 'Signed full-body means of cutoff CAR loop quadratures and lower of wholeP0-projected loop norm on complement, with normalized covariant Phi occupation weight; no generic ground existence debt.'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    captured = {name: (ROOT / name).read_bytes() for name in request['pins']}
    assert all(sha(raw) == request['pins'][name] for name, raw in captured.items())
    with zipfile.ZipFile(ROOT / request['profile_archive']) as archive:
        profile_raw = archive.read(request['profile_member'])
    profiles = json.loads(profile_raw)['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    result = {'kind': 'NEW_COMPLETE_BODY_ACTUALGROUND_COMPARISON_ALGEBRA_AND_SEPARATED_CONTROLS',
              'canonical_all65_exhaustive': complete_canonical_count(profiles),
              'first_native_canonical_sign_attempt': native_adverse_complete_canonical(profiles),
              'full_body_covariant_source_action': native_complete_body_jets(profiles),
              'second_material_resolvent_adapted_signed_response': normalized_signed_response(),
              'executed_fullbody_actualground_functional_deformation': native_symbolic_ground_deformation(profiles),
              'actual_inherited_ground_budget_support_attempt': inherited_ground_budget_and_support_attempt(profiles),
              'profile_member_sha256': sha(profile_raw), 'source_input_pins': request['pins'],
              'native_full_quantum_ground_response_numerically_evaluated': False,
              'actualground_complete_form_formula': 'Proof supplies exact H-dependent normalized action, not a numerical backend.',
              'physical1_carrier_lower': False, 'controlled_physical1_gapped_branch': False,
              'scientific_admission': False, 'campaign_complete': False}
    assert all((ROOT / name).read_bytes() == raw for name, raw in captured.items())
    result['inputs_unchanged'] = True
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'roles': 65, 'first_route': 'signed_complement',
                      'second_route': 'resolvent_adapted_complete_variation',
                      'native_ground_numeric': False, 'inputs_unchanged': True}))


if __name__ == '__main__':
    main()

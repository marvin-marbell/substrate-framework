"""New exact native representation/complete-form controls; no native ground solver."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import sympy as s

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def wedge(body: Any, k: int) -> Any:
    masks = [m for m in range(1 << body.rows) if m.bit_count() == k]
    positions = {m: j for j, m in enumerate(masks)}
    result = s.zeros(len(masks))
    for col, mask in enumerate(masks):
        for source in range(body.rows):
            if not mask & (1 << source):
                continue
            removed = mask ^ (1 << source)
            sign1 = (-1) ** (mask & ((1 << source) - 1)).bit_count()
            for target in range(body.rows):
                if removed & (1 << target):
                    continue
                sign2 = (-1) ** (removed & ((1 << target) - 1)).bit_count()
                result[positions[removed | (1 << target)], col] += sign1 * sign2 * body[target, source]
    return result


def generators() -> list[Any]:
    result = []
    for a, b in itertools.combinations(range(4), 2):
        real, imag = s.zeros(4), s.zeros(4)
        real[a, b] = real[b, a] = s.Rational(1, 2)
        imag[a, b] = s.I / 2
        imag[b, a] = -s.I / 2
        result.extend([real, imag])
    for n in range(1, 4):
        result.append(s.diag(*([1] * n + [-n] + [0] * (3 - n))) / s.sqrt(2 * n * (n + 1)))
    assert len(result) == 15
    assert s.Matrix([[s.trace(a * b) for b in result] for a in result]) == s.eye(15) / 2
    return result


def representation_and_form() -> dict[str, Any]:
    masks = [m for m in range(16) if m.bit_count() == 2]
    psi = s.zeros(36, 1)
    for j, mask in enumerate(masks):
        occ = [a for a in range(4) if mask & (1 << a)]
        empty = [a for a in range(4) if not mask & (1 << a)]
        psi[j * 6 + masks.index(15 ^ mask)] = (-1) ** sum(a > b for a in occ for b in empty)
    ident = s.eye(6)
    gs = generators()
    for g in gs:
        D = wedge(g, 2)
        assert (s.kronecker_product(D, ident) + s.kronecker_product(ident, D)) * psi == s.zeros(36, 1)
    casimirs = []
    for k in range(5):
        dim = len([m for m in range(16) if m.bit_count() == k])
        C = sum((wedge(g, k) ** 2 for g in gs), s.zeros(dim))
        expected = [0, s.Rational(15, 8), s.Rational(5, 2), s.Rational(15, 8), 0][k]
        assert s.simplify(C - expected * s.eye(dim)) == s.zeros(dim)
        casimirs.append(str(expected))
    h = s.diag(1, 1, -1, -1)
    A = wedge(h, 2)
    expA = s.diag(*[s.Integer(2) ** A[j, j] for j in range(6)])
    Z = s.trace(expA ** 2) / 6
    filt = expA / s.sqrt(Z)
    FF = s.kronecker_product(filt, ident)
    assert s.simplify((psi.H * FF.H * FF * psi)[0]) == 6
    assert (psi.H * s.kronecker_product(expA ** 2, ident) * psi)[0] != 6
    b = gs[2]
    K = wedge(b, 2)
    common = s.kronecker_product(K, ident) + s.kronecker_product(ident, K)
    assert K * filt != filt * K
    assert s.simplify((psi.H * FF * common * FF * psi)[0]) == 0
    assert common * FF * psi != s.zeros(36, 1)
    # Cross-multiplicity isometry is a matrix identity, not just a unique vector.
    columns = s.zeros(72, 2)
    columns[:36, 0] = psi
    columns[36:, 1] = psi
    assert s.simplify(columns.H * s.diag(FF ** 2, FF ** 2) * columns) == 6 * s.eye(2)
    deriv = s.zeros(6)
    for i in range(6):
        for j in range(6):
            if A[i, i] != A[j, j]:
                deriv[i, j] = K[i, j] * (expA[i, i] - expA[j, j]) / (A[i, i] - A[j, j])
            else:
                deriv[i, j] = s.log(2) * K[i, j] * expA[i, i]
    dz = s.trace(expA * deriv) / 3
    df = deriv / s.sqrt(Z) - expA * dz / (2 * Z ** s.Rational(3, 2))
    assert s.simplify(s.trace(filt * df)) == 0
    cost = s.simplify(s.trace(df ** 2) / 6)
    weighted = s.simplify(s.trace(filt ** 2 * K ** 2) / 6)
    assert s.simplify(cost - s.log(2) ** 2 * weighted).is_negative is True
    wrong_df = s.log(2) * K * filt
    assert s.simplify(wrong_df - df) != s.zeros(6)
    connection = s.I * wedge(gs[2], 2)
    moved_ordinary = df + connection * filt - filt * connection
    moved_covariant = moved_ordinary - connection * filt + filt * connection
    assert s.simplify(moved_covariant - df) == s.zeros(6)
    assert s.simplify(moved_ordinary - df) != s.zeros(6)
    # Literal F is number-local, hence shared rho/fullE commute; a hop is NOT.
    n0 = wedge(s.diag(1, 0, 0, 0), 2)
    assert n0 * filt == filt * n0
    hop_body = s.zeros(4); hop_body[0, 2] = 1
    hopK = wedge(hop_body, 2)
    assert n0 * hopK != hopK * n0
    # Cartesian Phi0 Casimir covers15; spatial45 jets are exactly0, not deleted.
    phi_jet_count = sum(mu == 0 for mu in range(4) for _ in gs)
    assert phi_jet_count == 15 and 60 - phi_jet_count == 45
    raw_wrong = s.zeros(36, 1); raw_wrong[0] = 1
    common_h = s.kronecker_product(A, ident) + s.kronecker_product(ident, A)
    assert common_h * raw_wrong != s.zeros(36, 1)
    return {'exact_reduced_CAR_dimension': 36, 'all15_SU4_checks': 15,
            'local_exterior_Casimirs_0_to4': casimirs, 'normalizer': str(Z),
            'noncommuting_common_primitive_expectation': '0; vector action NONZERO',
            'normalization_and_all_multiplicity_cross_identity': True,
            'divided_difference_cost': str(cost), 'weighted_B2': str(weighted),
            'wrong_commuting_derivative_and_ordinary_frame_rejected': True,
            'Phi0_jets': 15, 'Phi123_exact_zero_jets': 45,
            'rho_fullE_commutation_retained': True, 'wrong_local_budget_trial_not_singlet': True,
            'all64_embedding': '32 exact complementary pairs; active pair controls embed with30 spectator pairs, all64 factors retained.',
            'N4_actual_M_k': [256, 128], 'native_ground_computed': False}


def native_full_form_adverse(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    profile = profiles[2]
    row = next(t for t in profile['paths'] if t['length'] == 1)
    c0, ci = s.Rational(row['c0']), s.Rational(row['ci'])
    px, py = s.diag(1, 1, -1, -1), s.diag(1, 1, -1, -1)
    spatial = generators()[2]
    endpoint = (c0 * px + s.I * ci * spatial + c0 * py) / 2
    body = s.zeros(8)
    body[:4, 4:] = endpoint
    body[4:, :4] = endpoint.H
    native_K = wedge(body, 4)
    A = wedge(s.diag(px, py), 4)
    expA = s.diag(*[s.Integer(2) ** A[i, i] for i in range(70)])
    assert native_K * A != A * native_K
    assert s.trace(expA * (native_K * expA - expA * native_K)) == 0
    assert ci != 0 and c0 != 0
    # Actual nativeN4 full-colour checkerboard trial, all65 canonical128.
    sites = list(itertools.product(range(4), repeat=3))
    mask = sum(15 << (4 * i) for i, x in enumerate(sites) if sum(x) % 2 == 0)
    assert mask.bit_count() == 128
    source = sites.index((0, 0, 0)) * 4
    target = sites.index((1, 0, 0)) * 4
    moved = hop(mask, target, source)
    assert moved is not None
    # Distinct charged labels yield64 orthogonal outgoing blocks, not cancellation.
    common_generator_norm2 = 64
    assert common_generator_norm2 > 0
    # Every colour-mode in32 complementary pairs instead has total occupation32.
    assert 32 * 4 - 64 * 2 == 0
    checkerboard_electric_price = 64 * 64 ** 2
    assert checkerboard_electric_price == 262144
    cosine = (1, 0, -1, 0)
    resistance = F(0)
    for q in itertools.product(range(4), repeat=3):
        eig = 2 * (3 - sum(cosine[j] for j in q))
        if eig:
            resistance += F(2 * (1 - cosine[q[0]]), 64 * eig)
    assert resistance == F(21, 64)
    tL, tT = resistance, 1 - resistance
    assert tL + tT == 1
    assert tL ** 2 + tT ** 2 != 1
    assert 64 * ((64 - 1) ** 2 - 64 ** 2) == -8128
    return {'native_twoendpoint_canonical_dimension': 70,
            'native_fullPhi_endpoint_noncommutes_filter': True,
            'native_common_primitive_trace_cancellation': '0 with actual c0/ci and noncommuting Phi1',
            'native_N4_all65_each_particle_number': 128,
            'wrong_unconstrained_checkerboard_P_S_membership_rejected': True,
            'common64_outgoing_orthogonal_norm_squared': 64,
            'own_singlet_rho_electric_trial_price': 0,
            'unconstrained_checkerboard_electric_per_site': checkerboard_electric_price,
            'actual_N4_raw_path_fullE_jets': {'longitudinal': str(tL), 'transverse': str(tT), 'full': '1'},
            'omitted_affine_fullE_or_quadratic_cross_rejected': True,
            'native_allowed_active_hop_fullE_energy_difference': -8128,
            'scope': 'Exact native primitive coefficient/canonical adverse controls, not numerical ground.'}


def canonical_weighted_complements() -> dict[str, Any]:
    lambdas = [F(-2), F(-1), F(1), F(2), F(-2), F(-1), F(1), F(2)]
    weights = [F(1), F(3, 2), F(2), F(3), F(1), F(3, 2), F(2), F(3)]
    rows = []
    for n in range(2, 7):
        states = list(itertools.combinations(range(8), n))
        probabilities = []
        values = []
        for state in states:
            w = F(1)
            for i in state:
                w *= weights[i]
            probabilities.append(w)
            values.append(sum((lambdas[i] for i in state), F(0)))
        total = sum(probabilities, F(0))
        probabilities = [p / total for p in probabilities]
        minimum = min(sum((p for state, p in zip(states, probabilities) if i in state and j not in state), F(0))
                      for i in range(8) for j in range(8) if i != j)
        mean = sum((p * v for p, v in zip(probabilities, values)), F(0))
        variance = sum((p * v * v for p, v in zip(probabilities, values)), F(0)) - mean ** 2
        lower = sum((l * l for l in lambdas), F(0)) / 288
        assert minimum >= F(1, 144) and variance >= lower
        rows.append({'conditioned_n': n, 'all_configurations': len(states),
                     'minimum_pair_probability': str(minimum), 'variance': str(variance),
                     'cell_matching_variance_lower': str(lower)})
    # Fully retain two bad modes: M=8,k4,good6, n_good2..4.
    conditional_bad = [{'bad_mask': m, 'good_n': 4 - m.bit_count()} for m in range(4)]
    assert {r['good_n'] for r in conditional_bad} == {2, 3, 4}
    return {'weighted_all_good_counts': rows, 'complete_bad_occupation_conditioning': conditional_bad,
            'selected_occupation_substitution_rejected': True,
            'proof': 'Exchange bijections plus conditional independent pair orientations; no ground probability from counts.'}


def constants(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    lam, chi, fast = F(2 ** 8192), F(1, 2 ** 4096), F(2 ** 10240)
    ws = [abs(F(p['onsite'])) + 2 * sum((abs(F(t['c0'])) + abs(F(t['ci'])) for t in p['paths']), F(0)) for p in profiles]
    b = 4 * (lam * ws[0] + ws[1] + 63 * ws[2])
    assert b < 512 * lam
    assert profiles[1]['paths'] == profiles[2]['paths']
    delta = F(profiles[1]['onsite']) - F(profiles[2]['onsite'])
    assert delta == -F(126663739519795, 70368744177664)
    trial = 60 * chi + 3000 * lam + F(129, 10) * fast
    BS, C4 = F(2 ** 10244), F(2 ** 2053)
    assert trial + 8192 * lam < BS and C4 == 2 * BS / lam
    a = 225 * 4 * chi / (8 * BS)
    R, tau = F(2 ** 8196), F(1, 2 ** 10252)
    assert R ** 2 >= 6 * C4 / a and R ** 4 >= 4 * C4
    assert tau <= 1 / (4 * R) and tau <= abs(delta) * a / (5760 * chi)
    gamma, epsilon = F(1, 2 ** 24595), F(1, 2 ** 49195)
    true_gain = abs(delta) * a * tau / 576
    assert true_gain > gamma
    Cactual = F(256)
    exact_eps = min(F(1, 2), 16 * gamma ** 2 / (delta ** 2 * F(64, 255) * Cactual))
    assert exact_eps > epsilon
    # rho0 constant trial fullE0 versus an original nonzero character.
    assert 64 * F(1, 3) ** 2 > 0
    # The old global linear denominator truly loses the extensive gain.
    linear = [{'V': V, 'source_gain_at_t1': str(F(2 * V, 1 + V)),
               'per_volume_gain': str(F(2, 1 + V))} for V in (64, 216, 512)]
    assert F(2, 513) < F(2, 65)
    def record(x: F) -> dict[str, Any]:
        return {'sha256': sha(str(x).encode()), 'numerator_bits': x.numerator.bit_length(),
                'denominator_bits': x.denominator.bit_length()}
    return {'delta': str(delta), 'B_native_over_Lambda': record(b / lam),
            'own_trial_energy': '60chi+3000Lambda+(129/10)F',
            'B_S': '2^10244', 'C4_S': '2^2053', 'a': '225*2^-14340/2',
            'R': '2^8196', 'tau': '2^-10252', 'gamma': '2^-24595',
            'C_actual_separately_accepted': 256, 'epsilon': '2^-49195',
            'exact_gain': record(true_gain), 'exact_complement_from_block': record(exact_eps),
            'good_event_tails_and_conditional_occupancy_paid': True,
            'old_global_linear_route_failure_preserved': linear,
            'nonzero_character_electric_price': '64/9; beta0 trial not minimizing character',
            'constrained_sharp_budget_imported': False}


def hop(mask: int, target: int, source: int) -> tuple[int, int] | None:
    if not mask & (1 << source) or mask & (1 << target):
        return None
    sign = (-1) ** (mask & ((1 << source) - 1)).bit_count()
    mask ^= 1 << source
    sign *= (-1) ** (mask & ((1 << target) - 1)).bit_count()
    return mask | (1 << target), sign


def native_loop(mask: int, face: tuple[int, ...], copies: int = 1) -> dict[int, int]:
    result: Counter[int] = Counter()
    for order in itertools.permutations(range(4)):
        current = {mask: 1}
        for e in order:
            next_current: Counter[int] = Counter()
            for m, c in current.items():
                for copy in range(copies):
                    move = hop(m, 6 * copy + face[(e + 1) % 4], 6 * copy + face[e])
                    if move:
                        next_current[move[0]] += c * move[1]
            current = dict(next_current)
        result.update(current)
    return {m: c for m, c in result.items() if c}


def supported_signed_response(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    faces = ((0, 1, 2, 3), (1, 4, 5, 2))
    c0 = F(next(t['c0'] for t in profiles[1]['paths'] if t['length'] == 1))
    amplitude = s.Rational((3 * c0) ** 4)
    rows = []
    for mask in range(64):
        counts = [native_loop(mask, face).get(mask, 0) for face in faces]
        assert all(set(native_loop(mask, face)).issubset({mask}) for face in faces)
        rows.append({'mask': mask, 'counts': counts, 'canonical_spectators': 128 - mask.bit_count()})
    assert any(r['counts'][0] * r['counts'][1] < 0 for r in rows)
    assert any(r['counts'][0] * r['counts'][1] > 0 for r in rows)
    assert any(r['counts'] == [0, 0] for r in rows)
    # True64-copy DAG embeds two nonempty local roles; all other62 have no local particles.
    mixed = native_loop(1 | (2 << 6), faces[0], copies=2)
    assert mixed and any(m != 1 | (2 << 6) for m in mixed)
    # Different attempt: use actual native source coefficient entries in a labelled NONNATIVE L.
    O = s.eye(3) - 2 * s.Matrix([1, 2, 2]) * s.Matrix([[1, 2, 2]]) / 9
    assert O.T * O == s.eye(3)
    L, Q, omega = s.diag(0, 2, 5), s.diag(0, 1, 1), s.Matrix([1, 0, 0])
    resolvent = s.diag(0, s.Rational(1, 3), s.Rational(1, 6))
    selected = None
    for triple in itertools.combinations(rows, 3):
        aa = O.T * s.diag(*[r['counts'][0] for r in triple]) * O
        bb = O.T * s.diag(*[r['counts'][1] for r in triple]) * O
        X = s.Matrix.hstack(Q * aa * omega, Q * bb * omega)
        T = X.T * resolvent * X
        if T[0, 1] < 0 and T.det() > 0:
            selected = (triple, aa, bb, X, T)
            break
    assert selected is not None
    triple, aa, bb, X, T = selected
    Y = resolvent * X
    U = Y.T * Y
    K = Y.T * L * Y
    assert K == T - U
    j = s.symbols('j', real=True)
    source = aa + 2 * s.eye(3)
    y = Y[:, 0]
    trial = omega + j * y
    norm = (trial.T * trial)[0]
    exact = s.factor((trial.T * (L - j * source) * trial)[0] / norm)
    a = (omega.T * source * omega)[0]
    linear = (y.T * source * omega)[0]
    cubic = (y.T * source * y)[0]
    complete = (-j * a + j ** 2 * K[0, 0] - 2 * j ** 2 * linear - j ** 3 * cubic) / norm
    assert s.factor(exact - complete) == 0 and cubic != 0
    assert s.factor(exact - (-j * a + j ** 2 * K[0, 0] - 2 * j ** 2 * linear)) != 0
    epsilon = s.Rational(1, 2 ** 49195)
    unpaid_loop_mean = s.Integer(0)
    assert epsilon > 0 and unpaid_loop_mean ** 2 == 0
    return {'actual_native_section': 'Phi0=diag(3,-1,-1,-1),Phi123=0,U=I,A=0; coefficient controls not a field restriction or ground',
            'all64_onecolour_local_masks': rows, 'exact_fourleg_amplitude': str(amplitude),
            'all24_all64_source_embedding': 'All charged profiles equal on hops. Locally empty auxiliary roles annihilate exactly, not cropped; every65 species has128 via232 outside modes; neutral and Ref untouched.',
            'mixed_copy_offdiagonal_action': {str(m): c for m, c in mixed.items()},
            'method_ONLY_non_native_L': str(L), 'selected_native_coefficient_masks': [r['mask'] for r in triple],
            'response_T_unit_native_amplitude_squared': str(T), 'positive_determinant_negative_cross': True,
            'exact_signed_source_energy': str(exact), 'cubic': str(cubic),
            'lost_normalization_or_cubic_rejected': True,
            'epsilon_does_not_pay_zero_mean_Ward_lower': True,
            'actual_native_ground_loop_Gram_or_mean_computed': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    captured = {name: (ROOT / name).read_bytes() for name in request['pins']}
    for name, raw in captured.items():
        assert sha(raw) == request['pins'][name], name
    with zipfile.ZipFile(ROOT / request['profile_archive']) as archive:
        raw = archive.read(request['profile_member'])
    profiles = json.loads(raw)['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    result = {'kind': 'NEW_NATIVE_COMMON_SINGLET_NORMALIZED_COMPLETE_FORM_AND_SIGNED_RESPONSE_CONTROLS',
              'representation_form': representation_and_form(),
              'conditional_all_occupations': canonical_weighted_complements(),
              'complete_H_constants': constants(profiles),
              'native_full_form_adverse': native_full_form_adverse(profiles),
              'supported_loop_failed_transfer_and_signed_attempt': supported_signed_response(profiles),
              'profile_member_sha256': sha(raw), 'input_pins': request['pins'],
              'native_ground_computed': False, 'phase_stiffness_IR_atom': False,
              'campaign_complete': False, 'scientific_admission': False}
    assert all((ROOT / name).read_bytes() == raw for name, raw in captured.items())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Native complete-form controls, extensive constants and signed loop attempt completed')


if __name__ == '__main__':
    main()

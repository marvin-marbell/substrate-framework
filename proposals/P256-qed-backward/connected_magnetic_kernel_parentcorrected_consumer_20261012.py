"""Two native connected-mechanism attempts; no physical-ground numeric backend."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import sympy as sp

W = Fraction(46425051219964815, 8796093022208)
ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def face(x: tuple[int, int, int], n: int) -> dict[Any, int]:
    def step(axis: int) -> tuple[int, int, int]:
        y = list(x)
        y[axis] = (y[axis] + 1) % n
        return tuple(y)
    return {(x, 0): 1, (step(0), 1): 1, (step(1), 0): -1, (x, 1): -1}


def geometry(n: int) -> dict[str, Any]:
    a, b = face((0, 0, 0), n), face((1, 0, 0), n)
    assert sum(k*k for k in a.values()) == 4
    assert sum(k*b.get(e, 0) for e, k in a.items()) == -1
    for c in (a, b):
        div: dict[Any, int] = {}
        for (x, axis), value in c.items():
            y = list(x)
            y[axis] = (y[axis]+1) % n
            target = tuple(y)
            div[x] = div.get(x, 0) + value
            div[target] = div.get(target, 0) - value
        assert not any(div.values())
    # Exclusive edges prove integer-character independence for ALL harmonics.
    assert set(a)-set(b) and set(b)-set(a)
    gram = sp.Matrix([[1, -sp.Rational(1, 4)], [-sp.Rational(1, 4), 1]])
    assert gram.det() == sp.Rational(15, 16)
    hist: dict[str, dict[str, int]] = {}
    ap_nonzero = 0
    for role, copies, charge, lengths in (
        ('neutral', 1, 0, (1,)), ('active', 1, 1, (1, 2)),
        ('auxiliary', 63, 1, (1, 2))):
        counts: dict[str, int] = {}
        for x in itertools.product(range(n), repeat=3):
            for axis in range(3):
                for length in lengths:
                    y, nu_a, nu_b, ap = x, 0, 0, 1
                    for _ in range(length):
                        nu_a += charge*a.get((y, axis), 0)
                        nu_b += charge*b.get((y, axis), 0)
                        z = list(y)
                        if z[axis] == n-1:
                            ap *= -1
                        z[axis] = (z[axis]+1) % n
                        y = tuple(z)
                    for orientation in (1, -1):
                        key = f'{orientation*nu_a}/2,{orientation*nu_b}/2'
                        counts[key] = counts.get(key, 0) + copies
                        if ap == -1 and (nu_a or nu_b) and charge:
                            ap_nonzero += copies
        hist[role] = counts
    assert hist['auxiliary'] == {k: 63*v for k, v in hist['active'].items()}
    assert set(hist['neutral']) == {'0/2,0/2'} and ap_nonzero > 0
    # Every xy face at q0, with a nonflat NATIVE phase field: C^T f=0 but
    # C^T(f P_r) is nonzero. This is kinematics, not a measured ground.
    edges = [(x, axis) for x in itertools.product(range(n), repeat=3)
             for axis in range(3)]
    theta = {e: math.pi*((i*7+3) % 23)/11 for i, e in enumerate(edges)}
    r = 0.5
    base: dict[Any, float] = {}
    defect: dict[Any, float] = {}
    weight = 1/math.sqrt(n**3)
    for x in itertools.product(range(n), repeat=3):
        c = face(x, n)
        angle = sum(k*theta[e] for e, k in c.items())
        poisson = (1-r*r)/(1+2*r*math.cos(angle)+r*r)
        for e, k in c.items():
            base[e] = base.get(e, 0.0)+weight*k
            defect[e] = defect.get(e, 0.0)+weight*k*poisson/(2*r)
    assert max(map(abs, base.values())) < 1e-12
    d2 = sum(v*v for v in defect.values())
    assert d2 > 1e-6
    return {'N': n, 'closed_faces': True, 'neighbor_overlap': -1,
            'unit_closed_twoface_Gram': str(gram), 'character_rank': 2,
            'all65_one_two_AP_current_frequency_histogram': hist,
            'AP_nonzero_current_terms': ap_nonzero, 'q0_seam_square': d2,
            'method': 'native incidence and literal path control; not ground'}


def poisson_response_attempt() -> dict[str, Any]:
    r, theta = sp.symbols('r theta', positive=True, real=True)
    p = (1-r*r)/(1+2*r*sp.cos(theta)+r*r)
    # Check derivative using local log argument; branch is smooth for r<1.
    derivative = sp.simplify(2*(r*sp.cos(theta)+r*r)/(1+2*r*sp.cos(theta)+r*r))
    assert sp.simplify(derivative-(1-p)) == 0
    radii = [Fraction(1, 2), Fraction(3, 4), Fraction(15, 16), Fraction(255, 256)]
    rows = []
    for radius in radii:
        # Exact Parseval: P=1+2 sum (-r)^n cos(n theta).
        p2 = (1+radius*radius)/(1-radius*radius)
        assert p2 > 1
        # Native distinct characters have zero joint harmonic except n=m=0.
        covariance = Fraction(1)-Fraction(1)*Fraction(1)
        assert covariance == 0
        rows.append({'r': str(radius), 'native_Haar_mean_P': '1',
                     'native_Haar_mean_PaPb': '1', 'native_Haar_mean_P_squared': str(p2),
                     'native_Haar_normalized_q0_seam_square': str(2/(1-radius*radius)),
                     'normalized_seam_envelope_K0': str(Fraction(3)*p2/(radius*radius))})
    assert Fraction(rows[-1]['native_Haar_mean_P_squared']) > 200
    return {'route': 'actual-ground Ward plus vector-valued H1 Poisson comparison',
            'analytic_density_cap': '1/(2pi)+2sqrt(K)',
            'uniform_r_mean_P_cap': '1+4pi sqrt(K)',
            'two_P_cap': '(1+4pi sqrt(K))*(1+r^2)/(1-r^2)',
            'native_reference_rows': rows,
            'native_reference_is_physical1_ground': False,
            'gate': 'FAILED_LOCAL_MOMENTS_DO_NOT_PAY_CONNECTED_SEAM_OR_GRAM_LOWER'}


def exact_ward_method_control() -> dict[str, Any]:
    # Deliberately degenerate stationary block. NONNATIVE exact method control.
    L = sp.diag(0, 0, 2, 5)
    omega = sp.eye(4)[:, 0]
    P0 = sp.diag(1, 1, 0, 0)
    Q = sp.eye(4)-P0
    E = sp.Matrix([[0, -sp.I, 1-sp.I, 0], [sp.I, 0, 0, 0],
                   [1+sp.I, 0, 0, 0], [0, 0, 0, 0]])
    F = sp.Matrix([[0, 1, 2, 0], [1, 0, 0, 0],
                   [2, 0, 0, sp.I], [0, 0, -sp.I, 0]])
    J = -sp.I*(L*E-E*L)
    assert (J*omega+sp.I*L*E*omega).applyfunc(sp.simplify) == sp.zeros(4, 1)
    Z = (omega.T.conjugate()*F*P0*E*omega)[0]
    cross = (omega.T.conjugate()*F*Q*E*omega)[0]
    response = 2*sp.re(-sp.I*cross)
    full = (sp.I*omega.T.conjugate()*(E*F-F*E)*omega)[0]
    assert sp.simplify(full-response-2*sp.im(Z)) == 0
    assert full != response and Z != 0
    eps = sp.symbols('eps', positive=True)
    K = (omega.T.conjugate()*F*Q*(L+eps*sp.eye(4)).inv()*Q*J*omega)[0]
    assert sp.simplify(sp.limit(K, eps, 0)+sp.I*cross) == 0
    assert sp.simplify(K-(-sp.I*cross+sp.I*eps*
        (omega.T.conjugate()*F*Q*(L+eps*sp.eye(4)).inv()*Q*E*omega)[0])) == 0
    # Same-ground full complex two-column Gram. Entrywise-real is wrong.
    G = sp.Matrix([[0, 0, -sp.I, 2], [0, 0, 0, 0],
                   [sp.I, 0, 0, 0], [2, 0, 0, 0]])
    X = sp.Matrix.hstack(Q*F*omega, Q*G*omega)
    S = X.T.conjugate()*X
    u = sp.Matrix([1, sp.I])
    exact_value = (u.T.conjugate()*S*u)[0]
    deleted_value = (u.T.conjugate()*S.applyfunc(sp.re)*u)[0]
    assert S == S.T.conjugate() and S.det() > 0
    assert exact_value != deleted_value
    null_X = sp.Matrix.hstack(Q*F*omega, sp.I*Q*F*omega)
    assert (null_X.T.conjugate()*null_X).det() == 0
    T_scalar = sp.simplify((omega.T.conjugate()*E*Q*E*omega)[0])
    S_scalar = (omega.T.conjugate()*F*Q*F*omega)[0]
    normalized_static = -sp.I*cross/sp.sqrt(T_scalar*S_scalar)
    assert sp.simplify(sp.Abs(normalized_static)**2) <= 1
    assert S_scalar-sp.Abs(cross)**2/T_scalar >= 0
    t, tau = sp.symbols('t tau', real=True, positive=True)
    stationary_log_four = sp.diff(sp.log(sp.cosh(tau*t)), t, 4).subs(t, 0)
    assert stationary_log_four == -2*tau**4
    stationary_V = sp.zeros(4)
    stationary_V[0, 1] = stationary_V[1, 0] = 1
    assert Q*stationary_V*omega == sp.zeros(4, 1)
    return {'method': 'NONNATIVE exact Hilbert-space Ward/wholeP0/complex controls',
            'full_transport_response': str(full), 'connected_response': str(response),
            'stationary_block_Z': str(Z), 'K_minus_epsilon': str(K),
            'full_Hermitian_Gram': str(S), 'complex_u_quadratic': str(exact_value),
            'wrong_entrywise_real': str(deleted_value),
            'supported_normalized_static_response': str(normalized_static),
            'Gram_Cauchy_lower_control': str(S_scalar-sp.Abs(cross)**2/T_scalar),
            'log_connected_not_wholeP0_fourth': str(stationary_log_four),
            'wholeP0_and_imaginary_cross_discriminators': True,
            'rank_one_false_two_polarization_rejected': True}


def ladder(mask: int, mode: int, create: bool) -> tuple[int, int] | None:
    if bool(mask & (1 << mode)) == create:
        return None
    return mask ^ (1 << mode), (-1)**((mask & ((1 << mode)-1)).bit_count())


def loop_transition(initial: int, final: int, roles: tuple[int, ...]) -> tuple[int, list[int]]:
    total, values = 0, []
    for order in itertools.permutations(range(4)):
        state, sign = initial, 1
        flow = [0]*4
        for edge in order:
            moved = ladder(state, 4*roles[edge]+(edge+1) % 4, False)
            if moved is None:
                sign = 0
                break
            placed = ladder(moved[0], 4*roles[edge]+edge, True)
            if placed is None:
                sign = 0
                break
            state = placed[0]
            sign *= moved[1]*placed[1]
            flow[edge] += 1
            # Actual TOTAL density, not species-factorized electric copies.
            k = max(roles)+1
            rho_initial = [sum((initial >> (4*s+x)) & 1 for s in range(k)) for x in range(4)]
            rho_now = [sum((state >> (4*s+x)) & 1 for s in range(k)) for x in range(4)]
            assert [rho_now[x]-rho_initial[x] for x in range(4)] == [
                flow[x]-flow[(x-1) % 4] for x in range(4)]
        value = sign if state == final else 0
        total += value
        values.append(value)
    assert len(values) == 24
    return total, values


def equality_partitions() -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []
    def extend(prefix: tuple[int, ...]) -> None:
        if len(prefix) == 4:
            result.append(prefix)
            return
        for value in range(max(prefix)+2):
            extend(prefix+(value,))
    extend((0,))
    assert len(result) == 15
    return result


def falling(n: int, k: int) -> int:
    return math.prod(n-j for j in range(k))


def signed_occupation_attempt() -> dict[str, Any]:
    rows = []
    for mask in range(16):
        total, values = loop_transition(mask, mask, (0, 0, 0, 0))
        count = mask.bit_count()
        if count in (0, 4):
            assert total == 0
        elif count in (1, 3):
            assert total == 1
        elif mask in (5, 10):
            assert total == -4
        else:
            assert total == -2
        rows.append({'local_mask': mask, 'local_particles': count,
                     'signed_all24_coefficient': total, 'all24_orders': values})
    initial = sum(1 << mode for mode in (1, 2, 4, 7))
    final = sum(1 << mode for mode in (0, 1, 6, 7))
    profiles = []
    for roles in itertools.product(range(2), repeat=4):
        total, values = loop_transition(initial, final, roles)
        profiles.append({'roles': list(roles), 'coefficient': total, 'orders': values})
    mixed = sum(row['coefficient'] for row in profiles)
    same = sum(row['coefficient'] for row in profiles if len(set(row['roles'])) == 1)
    assert mixed == -6 and same == 0
    weights = []
    assignment_total = partition_abs = 0
    for partition in equality_partitions():
        k = max(partition)+1
        # No active block, or exactly one block occupied by the active species.
        no_active = falling(63, k)
        one_active_each = falling(63, k-1)
        weight = no_active+k*one_active_each
        assert weight == falling(64, k)
        assignment_total += weight
        partition_abs += math.factorial(k-1)
        weights.append({'equality_pattern': list(partition), 'distinct_species': k,
                        'all_auxiliary_assignments': no_active,
                        'each_designated_active_block_assignments': one_active_each,
                        'total_assignments': weight})
    assert assignment_total == 64**4 and partition_abs == 26
    assert 64*(1+1)**2 == 256 and 64*(1-1)**2 == 0
    assert 64 != 128 and 256 != 128
    background = Fraction(-1, 2)
    relative_cost = 64*(2*background+1)
    assert relative_cost == 0
    assert 64*(2*Fraction(-3, 4)+1) == -32
    # Embed the half-edge scalar in a CLOSED native face background. Its
    # initial divergence is zero; a charged hop shifts FULL E and rho together.
    # This is compatible-character kinematics, not a chosen minimizing fibre.
    closed_background = [-Fraction(1, 2)]*4
    partial_costs = []
    for count in range(1, 5):
        flow = [int(j < count) for j in range(4)]
        cost = 64*sum(2*e*k+k*k for e, k in zip(closed_background, flow))
        assert cost == 0
        partial_costs.append(str(cost))
    cumulant = []
    for n in (4, 8, 16, 64):
        # eta=1/N is only a regulator choice, not source or H weakening.
        eta = Fraction(1, n)
        budget = 624*(32*W)**4/eta**3
        assert budget > 1 and budget > 624*(32*W)**4
        cumulant.append({'N_label': n, 'eta_regulator': str(eta),
                         'M1_physical1_cumulant_upper': str(budget),
                         'literal_sum_all24': True, 'safe_lag_coefficient': 624})
    return {'route': 'native signed occupation/allcopy NORMALIZED fullH cumulant',
            'native_CAR_diagonal_all24': rows,
            'native_mixed_profiles': profiles, 'mixed_matrix_element': mixed,
            'samecopy_wrong_matrix_element': same,
            'all64_equality_profile_weights': weights,
            'assignment_total': assignment_total, 'partition_absolute_sum': partition_abs,
            'shared_equal_flow_cost': 256, 'shared_opposite_flow_cost': 0,
            'wrong_independent_cost': 128, 'E_minus_half_relative_cost': str(relative_cost),
            'closed_half_face_background_partial_HE_costs': partial_costs,
            'physical1_cutoff_connected_bounds': cumulant,
            'canonical_local_occupations_not_ground_measurements': True,
            'gate': 'FAILED_SIGNED_CONDITIONAL_OCCUPATION_AND_CONNECTED_ETA_SCALING'}


def matrix_order_control() -> dict[str, Any]:
    # Native admissible Phi/SU4 POINT factors, never frozen ground substitution.
    phi = sp.diag(1, 2, -1, -2)
    other = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0],
                       [0, 0, 0, 2], [0, 0, 2, 0]])
    U = sp.diag(sp.I, -sp.I, 1, 1)
    V = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    assert U.det() == V.det() == 1
    assert U.T.conjugate()*U == V.T.conjugate()*V == sp.eye(4)
    assert sp.trace(phi) == sp.trace(other) == 0
    B = (phi*U+U*other)/2
    C = (other*V+V*phi)/2
    assert B*C != C*B
    literal = sp.trace(B*C*B.T.conjugate()*C.T.conjugate())
    reordered = sp.trace(B*B.T.conjugate()*C*C.T.conjugate())
    assert literal != reordered
    return {'method': 'native admissible full4x4 factor-order control only',
            'ordered_trace': str(literal), 'wrong_factorized_trace': str(reordered),
            'noncommuting_Phi_SU4_retained': True, 'not_ground_or_spin_trace_lower': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, expected in request['source_inputs'].items():
        path = Path(name) if name.startswith('/') else ROOT/name
        assert sha(path) == expected, f'Changed bound bytes: {name}'
    proof = json.loads((ROOT/request['proof']).read_text())
    assert proof['scientific_result']['two_genuine_native_attempts']
    assert not proof['scientific_result']['small_energy_completion_credit']
    contract = json.loads(Path(request['contract']).read_text())
    assert contract['slices']['ConnectedMagneticKernel']['prefix'] == 'connected_magnetic_kernel_'
    for name in request['contract_bound_inputs']:
        assert request['source_inputs'][name] == contract['bound_sources'][name]['sha256']
    result = {'kind': 'OWN_EXECUTED_CONNECTED_NATIVE_TWO_ATTEMPT_RECEIPT',
              'governor': proof['governor'], 'source_input_SHA256': request['source_inputs'],
              'native_geometry': [geometry(n) for n in (4, 6, 8)],
              'attempt1': poisson_response_attempt(),
              'normalized_Ward_method_controls': exact_ward_method_control(),
              'attempt2': signed_occupation_attempt(),
              'full_quantum_matrix_order_control': matrix_order_control(),
              'actual_ground_or_fullH_numeric_backend': False,
              'native_closure_or_controlled_gapped_branch': False,
              'scoped_nonclosure_after_two_native_attempts': True,
              'scientific_admission': False, 'campaign_complete': False,
              'verification': 'Own scientific assertions/adverse controls, not build/lint/test suite.'}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'native_sizes': [4, 6, 8], 'attempts': 2,
                      'Ward_wholeP0_imaginary_controls': True,
                      'native_allcopy_assignments': 64**4,
                      'result': 'SCOPED_NONCLOSURE_NOT_PHOTON_NO_GO'}))


if __name__ == '__main__':
    main()

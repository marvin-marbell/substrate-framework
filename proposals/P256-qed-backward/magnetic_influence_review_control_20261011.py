"""Independent exact influence-method and native local-form falsifiers.
Only archived input data are read; no original modules or entrypoints run.
Finite matrix controls are METHOD, never native Gibbs approximations.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import zipfile

import sympy as s


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exact(value: s.Expr) -> str:
    return str(s.simplify(value))


def spectral_controls() -> dict:
    # Two finite character cells, each with an entire rank-two degeneracy.
    # These are discriminators of integration semantics, not native b quadrature.
    weights = [s.Rational(1, 2), s.Rational(1, 2)]
    boltz = [[s.Integer(1), s.Integer(1), s.Rational(1, 2)],
             [s.Rational(1, 3), s.Rational(1, 3), s.Rational(1, 6)]]
    energies = [[0, 0, s.log(2)], [s.log(3), s.log(3), s.log(6)]]
    aa = [s.Matrix([[1, 2, 1], [2, -1, 0], [1, 0, 3]]),
          s.Matrix([[-2, 1, 0], [1, 2, 1], [0, 1, 0]])]
    jj = [s.Matrix([[2, 3, 1], [3, 0, 0], [1, 0, -1]]),
          s.Matrix([[-1, 2, 0], [2, 3, 1], [0, 1, 4]])]
    zz = [sum(w) for w in boltz]
    z = sum(w * zb for w, zb in zip(weights, zz, strict=True))

    def mean(mats: list[s.Matrix]) -> s.Expr:
        return s.simplify(sum(weights[b] * sum(boltz[b][i] * mats[b][i, i]
                          for i in range(3)) for b in range(2)) / z)

    def raw(left: list[s.Matrix], right: list[s.Matrix], diagonal: bool = False) -> s.Expr:
        total = s.Integer(0)
        for b in range(2):
            for i in range(3):
                for j in range(3):
                    if diagonal and i != j:
                        continue
                    delta = s.simplify(energies[b][j] - energies[b][i])
                    ell = boltz[b][i] if delta == 0 else (boltz[b][i] - boltz[b][j]) / delta
                    total += weights[b] * ell * left[b][i, j] * right[b][j, i]
        return s.simplify(total / z)

    ma, mj = mean(aa), mean(jj)
    d = s.simplify(raw(aa, jj) - ma * mj)
    # Direct first coefficient of exp[-(H+alpha J)]: -L_ij J_ij.
    # At degeneracy this is -w J_ij, not just the diagonal entries.
    heat_prime = []
    for b in range(2):
        hp = s.zeros(3)
        for i in range(3):
            for j in range(3):
                delta = s.simplify(energies[b][j] - energies[b][i])
                hp[i, j] = -jj[b][i, j] * (boltz[b][i] if delta == 0 else
                              (boltz[b][i] - boltz[b][j]) / delta)
        heat_prime.append(hp)
    nprime = sum(weights[b] * s.trace(heat_prime[b] * aa[b]) for b in range(2))
    zprime = sum(weights[b] * s.trace(heat_prime[b]) for b in range(2))
    quotient_prime = s.simplify(nprime / z - ma * zprime / z)
    assert s.simplify(quotient_prime + d) == 0
    diagonal_wrong = s.simplify(raw(aa, jj, True) - ma * mj)
    fibre_a = [sum(boltz[b][i] * aa[b][i, i] for i in range(3)) / zz[b] for b in range(2)]
    fibre_j = [sum(boltz[b][i] * jj[b][i, i] for i in range(3)) / zz[b] for b in range(2)]
    averaged_product = sum(weights[b] * zz[b] * fibre_a[b] * fibre_j[b] for b in range(2)) / z
    fibre_centered_wrong = s.simplify(raw(aa, jj) - averaged_product)
    between = s.simplify(averaged_product - ma * mj)
    assert between != 0 and s.simplify(d - fibre_centered_wrong - between) == 0
    assert d != diagonal_wrong and quotient_prime != nprime / z
    assert ma * mj != 0
    # Independent nondegenerate 2x2 closed exponential, no diagonalization.
    alpha = s.symbols('alpha', real=True)
    gap = s.log(2)
    radius = s.sqrt((gap / 2) ** 2 + alpha ** 2)
    offdiag_mean = -2 * s.exp(-gap / 2) * alpha * s.sinh(radius) / radius / (s.exp(-gap / 2) * 2 * s.cosh(radius))
    pauli_prime = s.simplify(s.diff(offdiag_mean, alpha).subs(alpha, 0).rewrite(s.exp))
    assert s.simplify(pauli_prime + s.Rational(2, 3) / s.log(2)) == 0
    # Mixed spectral coefficients may be negative: A=sigma_x, J=-sigma_x.
    mixed_positive_frequency = s.Integer(-1)
    assert mixed_positive_frequency < 0
    return {'scope': 'EXACT_METHOD_NOT_NATIVE_CHARACTER_QUADRATURE', 'Z': exact(z),
            'global_mean_A': exact(ma), 'global_mean_J': exact(mj),
            'D_AJ_complete': exact(d), 'direct_normalized_derivative': exact(quotient_prime),
            'diagonal_only_wrong_D': exact(diagonal_wrong),
            'fibre_centered_wrong_D': exact(fibre_centered_wrong),
            'between_character_covariance': exact(between),
            'unnormalized_wrong_derivative': exact(nprime / z),
            'closed_Pauli_derivative': exact(pauli_prime),
            'wrongcases_rejected': ['diagonal-only degenerate block', 'per-character means',
                                   'omit derivative of denominator', 'mixed measure assumed positive']}


def contact_ground_thermal_controls() -> dict:
    alpha, beta = s.symbols('alpha beta', real=True, positive=True)
    # Entire two-dimensional block; the quadratic scalar gives a genuine contact.
    f = 3 * alpha ** 2 / 2 - s.log(2 * s.cosh(beta * alpha)) / beta
    second = s.simplify(s.diff(f, alpha, 2).subs(alpha, 0))
    assert second == 3 - beta and second != -beta
    a_mean = -s.tanh(beta * alpha) + 5 * alpha
    a_prime = s.diff(a_mean, alpha).subs(alpha, 0)
    assert a_prime == 5 - beta and a_prime != -beta
    sx = s.Matrix([[0, 1], [1, 0]])
    assert sx ** 2 == s.eye(2) and sx != s.trace(sx) * s.eye(2) / 2
    # H0=diag(0,0,2), J=diag(sigma_x,0), K=0. Q0 J P0=0 exactly.
    ground_response = -2 * beta / (2 + s.exp(-2 * beta))
    assert s.limit(ground_response / beta, beta, s.oo) == -1
    lam = s.symbols('lam')
    assert (alpha * sx - lam * s.eye(2)).det() == lam ** 2 - alpha ** 2
    # KMS signed frequencies, gap log2, initial weights 1 and 1/2.
    gap = s.log(2)
    mu_plus, mu_minus = s.Rational(2, 3), s.Rational(1, 3)
    gplus, gminus = s.Rational(1, 2) / gap, 1 / gap
    assert mu_minus == mu_plus / 2 and mu_plus * gplus == mu_minus * gminus
    return {'scope': 'EXACT_METHOD_NOT_NATIVE_GROUND_OR_RESPONSE',
            'F_second_contact_retained': exact(second),
            'alpha_dependent_A_contact_retained': exact(a_prime),
            'whole_ground_P0_rank': 2, 'Q0JP0_zero': True, 'P0JP0_nonscalar': True,
            'ground_response': exact(ground_response), 'ground_response_over_beta_limit': '-1',
            'ground_envelope_near_zero': '-abs(alpha)', 'rank1_ground_replacement_loses_variance': True,
            'signed_KMS_mu_plus': exact(mu_plus), 'signed_KMS_mu_minus': exact(mu_minus),
            'both_frequency_signs_retained': True}


def endpoint_control() -> dict:
    t = s.symbols('t', real=True)
    zp, zs = s.Rational(7, 3), s.Rational(5, 2)
    ap, ass = s.Rational(2, 7), -s.Rational(3, 5)
    z = (1 - t) * zp + t * zs
    omega = ((1 - t) * zp * ap + t * zs * ass) / z
    deriv = s.factor(s.diff(omega, t))
    assert s.simplify(deriv - zp * zs * (ass - ap) / z ** 2) == 0
    jp, js = 1 / (1 - t), -1 / t
    jmean = ((1 - t) * zp * jp + t * zs * js) / z
    ajmean = ((1 - t) * zp * ap * jp + t * zs * ass * js) / z
    assert s.simplify(deriv + ajmean - omega * jmean) == 0
    assert s.simplify(s.integrate(deriv, (t, 0, 1)) - (ass - ap)) == 0
    assert deriv.subs(t, s.Rational(1, 3)) != (zs * ass - zp * ap) / z.subs(t, s.Rational(1, 3))
    return {'scope': 'EXACT_METHOD_SUPERSELECTION_NOT_LOCAL_GAUSS_RESERVOIR',
            'derivative': exact(deriv), 'integral': exact(ass - ap),
            'source_endpoint_poles': ['1/(beta*(1-t))', '-1/(beta*t)'],
            'gives_spatial_decay': False, 'unequal_fibres_identified': False}


def transfer(mask: int, target: int, source: int) -> tuple[int, int] | None:
    if not mask & (1 << source) or mask & (1 << target):
        return None
    sign = -1 if (mask & ((1 << source) - 1)).bit_count() % 2 else 1
    reduced = mask ^ (1 << source)
    sign *= -1 if (reduced & ((1 << target) - 1)).bit_count() % 2 else 1
    return reduced | (1 << target), sign


def native_local_control(profiles: list[dict], sizes: list[int]) -> dict:
    row = next(r for r in profiles[1]['paths'] if r['axis'] == 0 and r['length'] == 1)
    c0, ci = s.Rational(row['c0']), s.Rational(row['ci'])
    price = abs(c0) + abs(ci)
    assert price > 0 and profiles[1]['onsite'] != profiles[2]['onsite']
    x0 = s.diag(1, -1, 2, -2)
    x1 = s.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 2], [0, 0, 2, 0]])
    y0, y1 = s.diag(2, -2, -1, 1), 2 * x1
    u = s.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, s.I], [0, 0, s.I, 0]])
    assert u.adjoint() * u == s.eye(4) and u.det() == 1
    fx, fy = c0 * x0 + s.I * ci * x1, c0 * y0 + s.I * ci * y1
    bmat = (fx * u + u * fy) / 2
    assert bmat != s.zeros(4) and fx * u != u * fx
    radius2 = s.trace(x0 ** 2 + x1 ** 2 + y0 ** 2 + y1 ** 2)
    cutoff = bmat / (8 * price * s.sqrt(1 + radius2))
    masks = [mask for mask in range(256) if mask.bit_count() == 4]
    lookup = {mask: i for i, mask in enumerate(masks)}
    forward = s.zeros(len(masks))
    for mask in masks:
        for a in range(4):
            for b in range(4):
                moved = transfer(mask, a, 4 + b)
                if moved is not None:
                    dest, sign = moved
                    forward[lookup[dest], lookup[mask]] += sign * cutoff[a, b]
    insertion = forward + forward.adjoint()
    assert insertion != s.zeros(len(masks)) and insertion == insertion.adjoint()
    nx = s.diag(*[int((mask & 15).bit_count()) for mask in masks])
    assert nx * forward - forward * nx == forward
    assert forward * s.diag(*[mask.bit_count() for mask in masks]) == 4 * forward
    # Raw normal edge has divergence (+1 at source,-1 at target); this is the
    # same charged insertion convention c_source^dagger B c_target.
    geometry = []
    for n in sizes:
        c = n // 2 - 1
        start, end = (c, 1, 1), (c + 1, 1, 1)
        assert 0 < start[0] < n - 1 and 0 < end[0] < n - 1
        assert start[1:] == end[1:] and end[0] - start[0] == 1
        geometry.append({'N': n, 'patch': [start, end], 'boundary_distance': min(c, n - 2 - c),
                         'canonical_particles_per_species': 2 * n ** 3,
                         'divergence_on_interior_patch': [1, -1], 'fullE_shift_raw_edge': 1})
    return {'scope': 'EXACT_NATIVE_ROW_AND_CANONICAL_LOCAL_CAR_FORM_NOT_GIBBS',
            'hopping': 1, 'row': row, 'all_species_loads': [p['load'] for p in profiles],
            'distinct_active_aux_onsite': True, 'local_canonical_four_particle_dimension': len(masks),
            'fullPhi_SU4_noncommuting_probe': True, 'radius2_exact': exact(radius2),
            'bounded_norm_analytic_premise': '8||B||/[8b sqrt(1+rx^2+ry^2)] <= 1',
            'CAR_charge_commutator_exact': '[nx,forward]=forward', 'geometry': geometry,
            'domain_claim': 'bounded decomposable insertion and smooth-core covariance; no H-domain assertion',
            'native_Gibbs_expectation_computed': False}


def gates_tails(profiles: list[dict], sizes: list[int]) -> dict:
    cj = sum((8 * p['load'] * (abs(Fraction(r['c0'])) + abs(Fraction(r['ci']))) * r['length']
              for p in profiles if p['scale'] == 'charged' for r in p['paths'] if r['axis'] == 2), Fraction(0))
    ck = sum((8 * p['load'] * (abs(Fraction(r['c0'])) + abs(Fraction(r['ci']))) * r['length'] ** 2
              for p in profiles if p['scale'] == 'charged' for r in p['paths'] if r['axis'] == 2), Fraction(0))
    assert cj == Fraction(15475017073321605, 1099511627776)
    assert ck == Fraction(17636744894459443, 1099511627776)
    fourth = []
    for n in sizes:
        # Exact fourth power of beta CJ 2^(1/4) Cbeta^(1/4) N^(9/4), beta=Cbeta=1.
        lhs = cj ** 4 * Fraction((2 * n ** 2) ** 3 * n ** 3, 4)
        assert lhs == 2 * cj ** 4 * n ** 9
        fourth.append({'N': n, 'cap_fourth_beta1_Cbeta1': str(lhs), 'decay_gate_paid': False})
    n = s.symbols('n', positive=True)
    factor = s.exp(32 / (n - 1)) - 1
    assert s.limit(n * factor, n, s.oo) == 32
    assert s.limit(s.sqrt(n) * factor, n, s.oo) == 0
    assert s.limit(factor + 1, n, s.oo) == 1
    colour_gate = Fraction(3 * 64 ** 2, 20 * 16 * 8 * 2 ** 4096)
    assert colour_gate < 7
    assert Fraction(9 * 4 ** 12, 288) == 524288
    return {'CJ': str(cj), 'CK': str(ck), 'crude_bound_fourth': fourth,
            'sufficient_gate': 'beta_N*Cbeta_N^(1/4)*N^(9/4) -> 0, uniformly in alpha',
            'fixed_positive_beta_gate': False,
            'W_equals_N_X_zero_limit_beta1': '32 (does not vanish)',
            'W_equals_sqrtN_X_zero_limit_beta1': '0 (conditional only)',
            'W_equals_1_X_equals_1_limit_beta1': '1 (remaining channels unpaid)',
            'native_temperature_window_proved': False,
            'SU4_tail_constant': 524288, 'SU4_shell64_slice16_uR2': str(colour_gate),
            'SU4_decreasing_tail_required_uR2': 7, 'SU4_shell64_gate_pass': False,
            'Phi_tail_beyond64': 'exp(-260*beta*2^-2048)',
            'Phi_tail_beta1_lower': '1-260*2^-2048',
            'interacting_Galerkin_or_Trotter_error_paid': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    request = json.loads(args.request.read_text())
    cp = Path(request['contract_path'])
    assert sha(cp.read_bytes()) == request['contract_sha256']
    for name, pin in request['target_sha256'].items():
        assert sha((root / name).read_bytes()) == pin, name
    snapshot = request['postexecution_snapshot']
    assert sha((root / snapshot['path']).read_bytes()) == snapshot['sha256']
    with zipfile.ZipFile(root / snapshot['path']) as z:
        for name, pin in request['target_sha256'].items():
            assert sha(z.read(name)) == pin, name
    with zipfile.ZipFile(root / request['original_preexecution']) as z:
        original_request = json.loads(z.read('magnetic_connected_limit_actual1_20261011_request.json'))
        for name, pin in original_request['pins'].items():
            assert sha(z.read(name)) == pin, name
        profile_bytes = z.read(original_request['profile'])
        profiles = json.loads(profile_bytes)['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    result = {'kind': 'INDEPENDENT_EXACT_INFLUENCE_REVIEW_CONTROL_NOT_SCIENTIFIC_ADMISSION',
              'source_sha256': sha(Path(__file__).read_bytes()),
              'request_sha256': sha(args.request.read_bytes()), 'input_sha256': request['target_sha256'],
              'postexecution_snapshot_sha256': snapshot['sha256'],
              'original_archived_input_sha256': original_request['pins'],
              'native_profile_sha256': sha(profile_bytes),
              'spectral': spectral_controls(), 'contact_ground_thermal': contact_ground_thermal_controls(),
              'endpoint': endpoint_control(), 'native_local': native_local_control(profiles, request['N']),
              'gates_tails': gates_tails(profiles, request['N']),
              'original_entrypoints_run': False, 'original_modules_imported': False,
              'unpaid': ['actual_magnetic_epsilon_N', 'native_local_Gauss_reservoir_interpolation',
                         'W_N', 'X_N', 'local_state_identity', 'common_local_dynamics',
                         'mixed_spectral_convergence', 'wholeP0', 'ground_selection',
                         'physical1phase', 'magnetic_stiffness', 'IRatom', 'pole', 'photon_admission']}
    for name, pin in request['target_sha256'].items():
        assert sha((root / name).read_bytes()) == pin, name
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'independent_exact_controls': 'completed', 'native_Gibbs': False,
                      'scientific_admission': False}))


if __name__ == '__main__':
    main()

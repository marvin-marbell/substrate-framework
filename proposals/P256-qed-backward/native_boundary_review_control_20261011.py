"""Independent exact CAR, native cost and signed all-time boundary review.
No historical source/checker/solver is imported or executed. All writes use own prefix.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import zipfile
from typing import cast

import numpy as np
import sympy as sy

sys.set_int_max_str_digits(0)
ROOT = Path(__file__).resolve().parent
Q = 1 << 100
ZERO = (F(0), F(0))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def gmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def conjugate(a):
    return (a[0], -a[1])


def scale(a, q):
    return (q * a[0], q * a[1])


def vector_sum(*vectors):
    out = defaultdict(lambda: ZERO)
    for vector in vectors:
        for key, value in vector.items():
            out[key] = gadd(out[key], value)
    return {key: value for key, value in out.items() if value != ZERO}


def inner(a, b):
    out = ZERO
    for key in a.keys() & b.keys():
        out = gadd(out, gmul(conjugate(a[key]), b[key]))
    return out


def car(mask, mode, creation):
    occupied = bool(mask & (1 << mode))
    if occupied == creation:
        return None
    sign = -1 if (mask & ((1 << mode) - 1)).bit_count() % 2 else 1
    return mask ^ (1 << mode), sign


def hop(vector, terms, shift):
    out = defaultdict(lambda: ZERO)
    for (mask, k), value in vector.items():
        for p, q, coefficient in terms:
            a = car(mask, q, False)
            if a is None:
                continue
            b = car(a[0], p, True)
            if b is not None:
                key = (b[0], k + shift)
                out[key] = gadd(out[key], scale(gmul(coefficient, value), a[1] * b[1]))
    return {key: value for key, value in out.items() if value != ZERO}


def exact_car_control(profiles):
    c = F(profiles[0]['c0_exact'])
    d = F(profiles[0]['d_exact'])
    local = {(5 | (10 << 4)): (F(1), F(0)),
             (10 | (5 << 4)): (F(0), F(2)),
             (3 | (12 << 4)): (F(3), F(0)),
             (12 | (3 << 4)): (F(-1), F(1))}
    omega = {(a | (b << 8), 0): gmul(x, y)
             for a, x in local.items() for b, y in local.items()}
    norm = inner(omega, omega)[0]
    terms = []
    for role in range(2):
        offset = 4 * role
        # Actual four-mode signed native endpoint block in the profile basis.
        terms.append([(offset + 2 + p, 8 + offset + q,
                       (-c, F(0)) if p == q else (F(0), d))
                      for p, q in itertools.product(range(2), repeat=2)])
    forward = [hop(omega, row, 1) for row in terms]
    reverse_terms = [[(q, p, conjugate(z)) for p, q, z in row] for row in terms]
    backward = [hop(omega, row, -1) for row in reverse_terms]
    images = forward + backward
    gram = [[inner(a, b) for b in images] for a in images]
    assert all(gram[i][j] == ZERO for i in range(4) for j in range(4) if i != j)
    assert all(gram[i][i][0] > 0 for i in range(4))
    assert inner(vector_sum(*images), vector_sum(*images))[0] == sum(gram[i][i][0] for i in range(4))
    assert all({k for _, k in image} == {1} for image in forward)
    assert all({k for _, k in image} == {-1} for image in backward)
    # Apply a later return event to an excited input, not to a fresh product.
    excited = vector_sum(*forward)
    returns = [hop(excited, row, -1) for row in reverse_terms]
    cross = inner(returns[0], returns[1])
    assert cross != ZERO
    projections = [inner(omega, row) for row in returns]
    neutral = vector_sum(*returns)
    projected = {key: scale(value, inner(omega, neutral)[0] / norm) for key, value in omega.items()}
    folded = vector_sum(neutral, {key: scale(value, -1) for key, value in projected.items()})
    assert inner(folded, folded)[0] > 0
    # Actual unequal onsite h and collective electric rho^2 preserve each label.
    onebody = []
    for cell in range(2):
        for role, profile in enumerate(profiles):
            off = 8 * cell + 4 * role
            m = F(profile['m_exact'])
            for p, sign in enumerate((-1, 1, -1, 1)):
                onebody.append((off + p, off + p, (sign * m, F(0))))
            for p, q, z in ((0, 2, (c, F(0))), (1, 3, (-c, F(0))),
                             (0, 3, (F(0), d)), (1, 2, (F(0), d))):
                onebody.extend(((off + p, off + q, z), (off + q, off + p, conjugate(z))))
    evolved = hop(omega, onebody, 0)
    counts = lambda mask: tuple(((mask >> (4 * j)) & 15).bit_count() for j in range(4))
    assert all(counts(mask) == (2, 2, 2, 2) for mask, _ in evolved)
    assert profiles[0]['m_exact'] != profiles[1]['m_exact']
    # CAR anticommutators on every one of the 2^16 masks, not a slot crop.
    for mask in range(1 << 16):
        for p in range(16):
            a, b = car(mask, p, False), car(mask, p, True)
            assert (a is None) != (b is None)
            first = a if a is not None else b
            assert first is not None
            second = car(first[0], p, a is not None)
            assert second == (mask, first[1])
        a = car(mask, 1, True)
        b = car(mask, 5, True)
        ab = None if a is None else car(a[0], 5, True)
        ba = None if b is None else car(b[0], 1, True)
        if ab is not None:
            assert a is not None and b is not None and ba is not None and ab[0] == ba[0] and a[1] * ab[1] == -b[1] * ba[1]
    return {'method_scope': 'Complete 16CAR, two cells, two labelled roles; fixed n2 per initial label; actual unequal native onsite/hopping values. Explicit constructed reference, not an eigenground, g256 or spectral substitution.',
            'canonical8_dimension': math.comb(16, 8), 'full_CAR_masks_exercised': 1 << 16,
            'omega_norm_squared_exact': str(norm),
            'boundary_image_gram_normalized_exact': [[[str(z[0] / norm), str(z[1] / norm)] for z in row] for row in gram],
            'shared_integer_flow_forward': 1, 'shared_integer_flow_reverse': -1,
            'interior_return_cross_exact': [str(z / norm) for z in cross],
            'both_return_reference_overlaps_exact': [[str(z / norm) for z in row] for row in projections],
            'neutral_reference_orthogonal_return_norm_squared_exact': str(inner(folded, folded)[0] / norm),
            'wrong_interior_label_orthogonality_rejected': True,
            'neutral_to_method_reference_reset_rejected': True, 'label_number_preserved_by_unequal_onsite': True}


def support_control(profiles, boundary):
    c, d = abs(F(profiles[0]['c0_exact'])), abs(F(profiles[0]['d_exact']))
    K1, K2 = 64 * 8 * (c + d), 64 * 2 * c
    J = 10 * K1 + 16 * K2
    for row in boundary['all64_boundary']['roles']:
        assert row['count'] in (1, 63)
        for path in row['paths']:
            entry_sum = sum(abs(F(z[0])) + abs(F(z[1])) for rr in path['matrix_exact'] for z in rr)
            assert 2 * entry_sum == F(path['fullCAR_hop_norm_upper_exact'])
            assert entry_sum == (4 * (c + d) if path['length'] == 1 else c)
    observations = []
    for n in (8, 10, 12):
        counts = defaultdict(lambda: defaultdict(int))
        root = (0, 0, 0)
        example = None
        for x in itertools.product(range(n), repeat=3):
            for axis, length in itertools.product(range(3), (1, 2)):
                if axis == 0 and length == 1 and x[0] % 2 == 0:
                    continue
                sites = [tuple((x[j] + (r if j == axis else 0)) % n for j in range(3)) for r in range(length + 1)]
                cells = {(s[0] // 2, s[1], s[2]) for s in sites}
                if root in cells:
                    counts[axis, length][len(cells)] += 1
                if axis == 1 and length == 2 and len(cells) == 3:
                    example = {'raw_sites': sites, 'blocked_cells': sorted(cells)}
        totals = {key: sum(value.values()) for key, value in counts.items()}
        assert totals == {(0, 1): 2, (0, 2): 4, (1, 1): 4, (1, 2): 6, (2, 1): 4, (2, 2): 6}
        observations.append({'N': n, 'incidences': {f'{a},{l}': value for (a, l), value in sorted(totals.items())}, 'middle_cell_witness': example})
    return {'NN_all64_both_orientation_exact': str(K1), 'two_link_all64_both_orientation_exact': str(K2),
            'J_exact': str(J), 'one_orientation_has_half_K': True, 'enumerations': observations,
            'analytic_all_even_N_ge8': 'For axis0 the two root raw sites meet two excluded NN starts and four length2 starts. On a transverse axis two raw sites times two NN starts give4; two raw sites times three length2 starts give6. N>=8 prevents modulo alias; length2 includes the middle blocked cell. Costs already include both orientations.'}


def radd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def rscale(a, q):
    return tuple(sorted((a[0] * q, a[1] * q)))


def rmul(a, b):
    values = [x * y for x in a for y in b]
    return min(values), max(values)


def zadd(a, b):
    return radd(a[0], b[0]), radd(a[1], b[1])


def zscale(a, q):
    return rscale(a[0], q), rscale(a[1], q)


def zmul(a, b):
    return radd(rmul(a[0], b[0]), rscale(rmul(a[1], b[1]), -1)), radd(rmul(a[0], b[1]), rmul(a[1], b[0]))


def zsum(values):
    result = ((F(0), F(0)), (F(0), F(0)))
    for z in values:
        result = zadd(result, z)
    return result


def mm(a, b):
    return [[zsum(zmul(a[i][k], b[k][j]) for k in range(len(a))) for j in range(len(a))] for i in range(len(a))]


def madd(*matrices):
    return [[zsum(a[i][j] for a in matrices) for j in range(len(matrices[0]))] for i in range(len(matrices[0]))]


def mscale(a, q):
    return [[zscale(z, q) for z in row] for row in a]


def transpose(a):
    return [list(row) for row in zip(*a)]


def dagger(a):
    return [[(a[j][i][0], rscale(a[j][i][1], -1)) for j in range(len(a))] for i in range(len(a))]


def sqrt_upper(value):
    n = value.numerator * Q * Q
    q = math.isqrt(n // value.denominator)
    if q * q * value.denominator < n:
        q += 1
    return F(q, Q)


def op_upper(a):
    squared = lambda z: max(abs(x) for x in z) ** 2
    frobenius = sqrt_upper(sum(squared(z[0]) + squared(z[1]) for row in a for z in row))
    off = sqrt_upper(squared(a[0][1][0]) + squared(a[0][1][1]))
    return min(frobenius, max(max(abs(x) for x in a[i][i][0]) + off for i in range(2)))


def lift(a):
    zero = ((F(0), F(0)), (F(0), F(0)))
    return [[a[i % 2][j % 2] if i // 2 == j // 2 else zero for j in range(4)] for i in range(4)]


def heat_polynomial(interface, row):
    moments = [[[tuple(tuple(F(x) for x in z[k]) for k in ('re', 'im')) for z in rr] for rr in row['moments'][name]] for name in ('M0', 'M1', 'M2')]
    if row['orientation'] == 'annihilation':
        moments = [transpose(a) for a in moments]
    b = F(interface['charged_floor_exact'])
    a = max(b + 1, sum(sum(moments[1][i][i][0]) for i in range(2)) / sum(sum(moments[0][i][i][0]) for i in range(2)))
    r = a - b
    v = madd(moments[2], mscale(moments[1], -2 * a), mscale(moments[0], a * a))
    B = mscale(v, 1 / (2 * r * r))
    C0 = madd(moments[0], mscale(B, -1))
    C1 = madd(mscale(moments[0], a), mscale(moments[1], -1), mscale(B, -r))
    eta = F(interface['ground_distance_exact'][1])
    transfer = 2 * eta  # Ys is normalized; raw binary64 N is not ||s||².
    h = min(F(1), op_upper(moments[0]) + transfer)
    errors = [(b, 0, op_upper(v) / (2 * r * r) + transfer),
              (a, 0, -op_upper(v) / (2 * r * r)), (a, 1, -op_upper(v) / (2 * r))]
    return {'terms': [(a, 0, lift(C0)), (a, 1, lift(C1)), (b, 0, lift(B))],
            'error': errors, 'h': [(b, 0, h)], 'a': a, 'b': b, 'vop': op_upper(v)}


def scalar_integral(a, b, moment):
    return sum(c * d * F(math.factorial(p + q + moment), (64 + r + s) ** (p + q + moment + 1))
               for r, p, c in a for s, q, d in b)


def signed_convolution(plus, minus, matrix, moment):
    out = ((F(0), F(0)), (F(0), F(0)))
    for rp, pp, ap in plus['terms']:
        for rm, pm, am in minus['terms']:
            trace = zsum(mm(mm(mm(ap, matrix), am), dagger(matrix))[j][j] for j in range(4))
            weight = F(math.factorial(pp + pm + moment), (64 + rp + rm) ** (pp + pm + moment + 1))
            out = zadd(out, zscale(trace, weight))
    return out


def exact_native_matrices(receipt, phi_path):
    with np.load(phi_path, allow_pickle=False) as data:
        raw = data['Phi_binary_section'][1]
    phi = sy.Matrix([[sy.Rational(F(float(complex(v).real))) + sy.I * sy.Rational(F(float(complex(v).imag))) for v in row] for row in raw])
    had = sy.Matrix([[1, 0, 1, 0], [0, 1, 0, 1], [-1, 0, 1, 0], [0, -1, 0, 1]]) / sy.sqrt(2)
    frame = sy.simplify(had.T * phi * had)
    gamma = sy.simplify(frame / sy.sqrt(sy.trace(frame * frame) / 4))
    S = sy.Matrix.hstack(sy.eye(4)[:, 0], gamma * sy.eye(4)[:, 0], sy.eye(4)[:, 1], gamma * sy.eye(4)[:, 1])
    P = sy.diag(1, -1, 1, -1)
    assert sy.simplify(S.conjugate().T * S) == sy.eye(4)
    matrices = {}
    for role, row in zip(('active', 'aux'), receipt['all64_boundary']['roles']):
        for axis in (1, 2):
            path = next(p for p in row['paths'] if p['axis'] == axis and p['length'] == 1)
            physical = sy.Matrix([[sy.Rational(a) + sy.I * sy.Rational(b) for a, b in rr] for rr in path['matrix_exact']])
            native = sy.simplify(P * S.conjugate().T * physical * S * P)
            matrices[role, axis] = [[((F(str(sy.re(z))), F(str(sy.re(z)))), (F(str(sy.im(z))), F(str(sy.im(z))))) for z in native.row(i)] for i in range(4)]
    return matrices


def alltime_control(interface, boundary, profiles, published):
    matrices = exact_native_matrices(boundary, ROOT / 'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz')
    rows = []
    for role, axis in itertools.product(('active', 'aux'), (1, 2)):
        plus = heat_polynomial(interface, next(r for r in interface['roles'] if r['role'] == role and r['orientation'] == 'creation'))
        minus = heat_polynomial(interface, next(r for r in interface['roles'] if r['role'] == role and r['orientation'] == 'annihilation'))
        matrix = matrices[role, axis]
        cost = abs(F(profiles[0]['c0_exact'])) + abs(F(profiles[0]['d_exact']))
        bounds = []
        for moment in (0, 1):
            center = signed_convolution(plus, minus, matrix, moment)
            reversed_center = signed_convolution(plus, minus, dagger(matrix), moment)
            radius = 4 * cost * cost * (scalar_integral(plus['error'], minus['h'], moment) + scalar_integral(plus['h'], minus['error'], moment) + scalar_integral(plus['error'], minus['error'], moment))
            assert radius >= 0
            independent = max(F(0), center[0][0] - radius), max(F(0), center[0][1] + radius)
            old = next(r for r in published['rows'] if r['role'] == role and r['axis'] == axis)['bounds'][moment]
            claimed = tuple(F(x) for x in old['one_orientation_one_copy_exact'])
            assert claimed[0] <= independent[0] <= independent[1] <= claimed[1]
            reverse = max(F(0), reversed_center[0][0] - radius), max(F(0), reversed_center[0][1] + radius)
            assert claimed[0] <= reverse[0] <= reverse[1] <= claimed[1]
            bounds.append({'moment': moment, 'independent_one_orientation_exact': list(map(str, independent)),
                           'independent_reverse_orientation_exact': list(map(str, reverse)),
                           'claimed_rectangle_contains_both': True, 'independent_error_exact': str(radius),
                           'copy_multiplicity': 1 if role == 'active' else 63})
        rows.append({'role': role, 'axis': axis, 'bounds': bounds})
    # Exact noncommuting complex witness; source Gram minus must be transposed.
    P = sy.Matrix([[sy.Rational(2, 3), sy.I / 7], [-sy.I / 7, sy.Rational(1, 3)]])
    H = sy.Matrix([[sy.Rational(3, 5), sy.I / 11], [-sy.I / 11, sy.Rational(2, 5)]])
    M = sy.Matrix([[1, 2 + sy.I], [-sy.I, sy.Rational(1, 2)]])
    correct = sy.trace(P * M * H.T * M.conjugate().T)
    wrong = sy.trace(P * M * H * M.conjugate().T)
    assert correct != wrong and P * H != H * P
    # Direct tensor spectral resolvent versus polynomial Laplace identity.
    vectors = [sy.Matrix([1, sy.I]), sy.Matrix([2, 1])]
    energies = [F(3), F(8)]
    coefficients = [v * v.conjugate().T / (v.conjugate().T * v)[0] for v in vectors]
    spectral = sum(sy.trace(a * M * b.T * M.conjugate().T) / (64 + x + y)
                   for a, x in zip(coefficients, energies) for b, y in zip(coefficients[::-1], energies[::-1]))
    spectral2 = sum(sy.trace(a * M * b.T * M.conjugate().T) / (64 + x + y) ** 2
                    for a, x in zip(coefficients, energies) for b, y in zip(coefficients[::-1], energies[::-1]))
    spectral, spectral2 = sy.simplify(cast(sy.Expr, spectral)), sy.simplify(cast(sy.Expr, spectral2))
    assert spectral > 0 and spectral2 > 0
    source = sy.Matrix([(u.conjugate().T * M * v.conjugate())[0] /
                        sy.sqrt((u.conjugate().T * u)[0] * (v.conjugate().T * v)[0])
                        for u in vectors for v in vectors[::-1]])
    tensor_energy = sy.diag(*(64 + x + y for x in energies for y in energies[::-1]))
    direct_inverse = (source.conjugate().T * tensor_energy.inv() * source)[0]
    direct_inverse_square = (source.conjugate().T * tensor_energy.inv() ** 2 * source)[0]
    assert sy.simplify(direct_inverse - spectral) == 0
    assert sy.simplify(direct_inverse_square - spectral2) == 0
    t = sy.Symbol('t', nonnegative=True)
    signed = sy.integrate(sy.exp(-7 * t) * (1 - 2 * t) * (3 + t), (t, 0, sy.oo))
    signed_square = sy.integrate(t * sy.exp(-7 * t) * (1 - 2 * t) * (3 + t), (t, 0, sy.oo))
    absolute_wrong = sy.integrate(sy.exp(-7 * t) * (3 + 5 * t + 2 * t * t), (t, 0, sy.oo))
    assert signed == sy.Rational(108, 343) and signed != absolute_wrong
    assert signed_square == sy.Rational(65, 2401)
    return {'actual_native_input_rows': rows, 'all_spin_pair_blocks': 'Native4x4 contraction sums all independent sigma,tau; block zero only when actual matrix says zero.',
            'external_rate_exact': '64', 'charged_phase_replacement': False,
            'complex_transpose_wrongcase_exact': {'correct': str(correct), 'wrong': str(wrong)},
            'finite_spectral_inverse_witness_exact': str(spectral), 'finite_spectral_inverse_square_witness_exact': str(spectral2),
            'direct_tensor_resolvents_equal': True,
            'signed_exponential_polynomial_exact': str(signed),
            'signed_exponential_polynomial_square_exact': str(signed_square),
            'wrong_absolute_coefficients_exact': str(absolute_wrong),
            'whole_time_domain': '[0,infinity)', 'tail_policy': 'No charged upper cutoff. Floor b>0 controls integral tails and Hermite divided difference for every x>=b.'}


def source_binding(interface, formation, upper):
    with np.load(ROOT / 'local_so5_resum_actual04_20261011_state.npz', allow_pickle=False) as data:
        state = data['state']
    assert state.dtype == np.float64 and np.isfinite(state).all()
    buckets = defaultdict(int)
    for bits in state.reshape(-1).view(np.uint64):
        word = int(bits)
        exponent = (word >> 52) & 2047
        mantissa = word & ((1 << 52) - 1)
        if exponent:
            mantissa |= 1 << 52
            power = exponent - 1075
        else:
            power = -1074
        if mantissa:
            buckets[power] += mantissa * mantissa
    exact_norm = sum((F(value << (2 * power)) if power >= 0 else F(value, 1 << (-2 * power)) for power, value in buckets.items()), F(0))
    assert exact_norm == F(formation['source_norm_squared_exact']) == F(interface['source_norm_squared_exact'])
    assert exact_norm > 1  # Existing eta*(1+sqrt(rawN)) is conservative for THESE bytes.
    I = F(upper['selected_ground_infidelity_upper_exact'])
    eta = tuple(F(x) for x in interface['ground_distance_exact'])
    assert eta[0] ** 2 <= 2 * I <= eta[1] ** 2
    residual, shift = F(1, 65536), F(46955575073, 1538631694057472)
    assert F(interface['rigorous_residual_exact']) == residual == F(upper['normalized_exact_source_residual_upper_exact'])
    assert interface['formation_included_in_residual_exact'] == upper['coefficient_and_action_error_upper_exact'] == '1/262144'
    assert F(interface['source_action_error_exact']) == residual + shift
    assert F(interface['Rayleigh_ground_shift_exact']) == shift
    assert F(interface['charged_floor_exact']) == -5120 - F(upper['ground_energy_interval_exact'][1])
    return {'exact_binary64_source_norm_independently_recomputed': str(exact_norm), 'entries': int(state.size), 'shape': list(state.shape),
            'normalization': 'Raw x has norm²=N; s=x/sqrt(N) has norm1. Heat M0,M1,M2 divide raw contractions by exactN.',
            'rawN_vs_normalized_s_semantic_note': 'Analytic helper calls sqrt(rawN) source_norm; it is not ||s||. Here rawN>1, so eta*(1+sqrt(rawN))>=2eta and valid but unnecessarily loose; not a general formula.',
            'eta_sqrt_2_infidelity_upper_exact': str(eta[1]), 'rigorous_residual_exact': str(residual),
            'formation_already_included_exact': '1/262144', 'Rayleigh_shift_exact': str(shift),
            'historical_operation_model_status': 'CONDITIONAL; byte binding does not attest historical intermediate FTZ/DAZ or rounding.'}


def run(request_path, output_path):
    request = json.loads(request_path.read_text())
    pins = request['pins']
    assert all(sha(ROOT / name) == value for name, value in pins.items())
    for binding in request['archive_bindings']:
        with zipfile.ZipFile(ROOT / binding['archive']) as archive:
            archived = archive.read(binding['member'])
        archived_sha = hashlib.sha256(archived).hexdigest()
        name = binding['input']
        transfer = request.get('changed_typed_dependency_bindings', {}).get(name)
        if transfer is None:
            assert archived_sha == pins[name]
            assert archived == (ROOT / name).read_bytes()
        else:
            assert archived_sha == transfer['original']
            assert pins[name] == transfer['successor']
            assert sha(ROOT / name) == transfer['successor']
    load = lambda name: json.loads((ROOT / name).read_text())
    profile = load('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json')['profiles']
    boundary = load('charged_first_shell_actual04_20261011_receipt.json')
    interface = load('charged_cell_heat_typed3_20261011_interface.json')
    result = {'kind': 'NEW_INDEPENDENT_NATIVE_BOUNDARY_EXACT_CAR_AND_ALLTIME_REVIEW_CONTROL',
              'source_sha256': sha(Path(__file__)), 'request_sha256': sha(request_path), 'input_sha256': pins,
              'archive_bindings_observed': request['archive_bindings'],
              'fullCAR_sharedflow_boundary_interior': exact_car_control(profile),
              'native_support_orientation_norms': support_control(profile, boundary),
              'sourceformation': source_binding(interface, load('three_cell_density_formation_typed3_20261011_interface.json'), load('local_so5_ground_upper_complete_state_consumer_20261011_receipt.json')),
              'native_alltime_convolution': alltime_control(interface, boundary, profile, load('normalized_vacuum_actual1_20261011_receipt.json')['native_alltime_source_attempt'])}
    assert pins == {name: sha(ROOT / name) for name in pins}
    result['source_and_inputs_unchanged'] = True
    output_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'result': output_path.name, 'fullCAR_masks': 65536, 'actual_source_rows': 4, 'source_entries': result['sourceformation']['entries'], 'independent_controls_completed': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.request, args.output)

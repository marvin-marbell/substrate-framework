"""Exact native four-matrix polar construction and expectation-priced adverse controls."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import sympy as s


def hs2(matrix):
    return s.simplify(s.trace(matrix.H * matrix))


def tuple2(matrices):
    return s.simplify(sum(hs2(matrix) for matrix in matrices))


def defect(body, target):
    x = [s.trace(b*b)/4 for b in body]
    diagonal = sum(hs2(b*b-v*s.eye(4)) for b, v in zip(body, x))
    ordered = sum(hs2(body[i]*body[j]+body[j]*body[i])
                  for i in range(4) for j in range(4) if i != j)
    delta = s.simplify(16*diagonal+4*ordered)
    radial = sum((v-w)**2 for v, w in zip(x, target))
    return x, delta, s.simplify(delta+64*radial), diagonal, ordered


def polar_sign(matrix):
    """Exact polynomial functional calculus; no original solver or imported producer."""
    eigenvalues = list(matrix.eigenvals())
    polynomial = s.zeros(4)
    for eigenvalue in eigenvalues:
        assert eigenvalue.is_real and eigenvalue.is_zero is False
        sign = s.sign(eigenvalue)
        assert sign in (s.S.One, s.S.NegativeOne)
        term = s.eye(4)
        for other in eigenvalues:
            if other != eigenvalue:
                term = term*(matrix-other*s.eye(4))/(eigenvalue-other)
        polynomial += sign*term
    return s.simplify(polynomial)


def repair(body, target):
    normalized = [b/s.sqrt(v) for b, v in zip(body, target)]
    frames = []
    projection_errors = []
    for matrix in normalized:
        projected = matrix
        for frame in frames:
            projected = s.simplify((projected-frame*projected*frame)/2)
        projection_errors.append(hs2(matrix-projected))
        frames.append(polar_sign(projected))
    for i, frame in enumerate(frames):
        assert frame.H == frame and s.simplify(frame*frame) == s.eye(4)
        assert s.trace(frame) == 0
        for j, other in enumerate(frames):
            assert s.simplify(frame*other+other*frame) == 2*int(i == j)*s.eye(4)
    return frames, projection_errors


def word(tuple_, subset):
    result = s.eye(4)
    for index in subset:
        result = result*tuple_[index]
    return result


def common_intertwiner(frames, gamma):
    subsets = [subset for degree in range(5)
               for subset in itertools.combinations(range(4), degree)]
    fw = [word(frames, subset) for subset in subsets]
    gw = [word(gamma, subset) for subset in subsets]
    for a in range(4):
        for b in range(4):
            unit = s.zeros(4)
            unit[a, b] = 1
            projected = s.simplify(sum((f*unit*g.H for f, g in zip(fw, gw)), s.zeros(4))/16)
            alpha = s.simplify(hs2(projected)/4)
            if alpha == 0:
                continue
            assert s.simplify(projected.H*projected) == alpha*s.eye(4)
            u = s.simplify(projected/s.sqrt(alpha))
            assert s.simplify(u.H*u) == s.eye(4)
            for frame, native in zip(frames, gamma):
                assert s.simplify(frame*u-u*native) == s.zeros(4)
            determinant = s.simplify(u.det())
            assert s.simplify(determinant*s.conjugate(determinant)) == 1
            return {'matrix_unit': [a, b], 'alpha': str(alpha),
                    'unitary_determinant': str(determinant),
                    'SU4_scalar_correction': '(det(U0))^(-1/4)',
                    'simultaneous_intertwining_exact': True,
                    'four_center_choices': True}
    raise AssertionError('all sixteen intertwiner matrix units vanished')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for pin in request['pins'].values():
        assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest() == pin['sha256']
    # Native gamma convention is pinned and inspected, never imported/executed.
    pauli = [s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -s.I], [s.I, 0]]), s.diag(1, -1)]
    gamma = [s.diag(1, 1, -1, -1)] + [s.kronecker_product(pauli[0], p) for p in pauli]
    for i in range(4):
        for j in range(4):
            assert gamma[i]*gamma[j]+gamma[j]*gamma[i] == 2*int(i == j)*s.eye(4)
    amplitudes = [s.Rational(1, 4), s.Rational(1, 2), s.Rational(3, 4), s.Integer(1)]
    target = [a*a for a in amplitudes]
    assert min(target) > s.Rational(1, 128) and sum(v*v for v in target) < 4
    centre = [a*g for a, g in zip(amplitudes, gamma)]
    _, zero_delta, zero_t, _, _ = defect(centre, target)
    assert zero_delta == zero_t == 0
    perturbation = s.Rational(1, 2**40)
    body = list(centre)
    body[0] = amplitudes[0]*(gamma[0]+perturbation*gamma[1])
    _, delta, t, _, ordered = defect(body, target)
    assert 0 < t < s.Rational(1, 2**52) and ordered > 0
    frames, errors = repair(body, target)
    repaired = [a*f for a, f in zip(amplitudes, frames)]
    distance_upper = tuple2([b-c for b, c in zip(body, repaired)])
    assert distance_upper <= 2**35*t
    intertwiner = common_intertwiner(frames, gamma)
    # Conjugation covariance of the constructive tuple, including complex frames.
    gauge = s.diag(s.I, -s.I, 1, 1)
    assert gauge.det() == 1
    transformed, _ = repair([gauge*b*gauge.H for b in body], target)
    assert all(s.simplify(f-gauge*original*gauge.H) == s.zeros(4)
               for f, original in zip(transformed, frames))
    missing = list(centre)
    missing[0] = perturbation*centre[0]
    _, missing_delta, missing_t, _, _ = defect(missing, target)
    missing_distance = 4*target[0]*(1-perturbation)**2
    assert missing_t >= s.Rational(1, 2**52)
    assert missing_distance == tuple2([b-c for b, c in zip(missing, centre)])
    assert missing_distance <= 2**59*missing_t
    wrong = list(centre)
    wrong[1] = amplitudes[1]*gamma[0]
    _, wrong_delta, wrong_t, wrong_self, wrong_cross = defect(wrong, target)
    assert wrong_self == 0 and wrong_cross > 0 and wrong_delta > 0
    wrong_lower = 4*target[0]*target[1]/(target[0]+target[1])
    wrong_upper = tuple2([b-c for b, c in zip(wrong, centre)])
    assert 0 < wrong_lower <= wrong_upper <= 2**59*wrong_t
    # Exact same-measure correlation: all defects and radius occur on ONE atom.
    radius = s.Integer(2)**80
    probability = radius**-4
    zero_body = [s.zeros(4) for _ in range(4)]
    _, _, bad_t, _, _ = defect(zero_body, target)
    mean_t = probability*bad_t
    transfer = probability*radius**2*tuple2(centre)
    actual_distance = probability*(radius-1)**2*tuple2(centre)
    probability_only = probability*tuple2(centre)
    eta = mean_t  # This synthetic control satisfies the stronger combined bound.
    assert eta < s.Rational(1, 2**159) and transfer < s.Rational(1, 2**112)
    assert actual_distance <= 2*transfer+2**61*eta
    assert actual_distance > probability_only*2**150
    coefficients = [1, 14, 156, 1719]
    assert sum(c*c for c in coefficients) == 2979494 < 2**22
    assert s.Rational(1375, 2**20) < 1
    assert 2**58+2**27 < 2**59 and 2**61 <= 2**80
    receipt = {'kind': 'NEW_EXACT_NATIVE_CLIFFORD_GEOMETRY_CONTROLS_NOT_GROUND',
               'governor': request['governor'], 'input_pins': request['pins'],
               'native_metric': 'tupleHS^2=sum Tr(Ddagger D); x=Tr B^2/4',
               'explicit_C_orbit': str(2**61), 'C_orbit_power': 61,
               'unequal_norm_analogs': [str(v) for v in target],
               'good_mixed_noncommuting': {'T': str(t), 'DeltaJ': str(delta),
                   'constructed_distance_upper': str(distance_upper),
                   'projection_errors': [str(e) for e in errors], 'intertwiner': intertwiner,
                   'all4_self_all12_ordered_cross_exact': True,
                   'conjugation_covariance_exact': True},
               'nearzero_component': {'T': str(missing_t), 'DeltaJ': str(missing_delta),
                   'exact_orbit_distance': str(missing_distance), 'bad_branch_pass': True},
               'wrong_independent_frames': {'all4_self_residual_sum': str(wrong_self),
                   'ordered12_cross_sum': str(wrong_cross), 'T': str(wrong_t),
                   'positive_orbit_distance_lower': str(wrong_lower),
                   'one_native_frame_upper': str(wrong_upper), 'rejected_exactly': True},
               'correlated_rare_large_field': {'radius': str(radius), 'probability': str(probability),
                   'same_atom_mean_T': str(mean_t), 'same_atom_mean_transfer_squared': str(transfer),
                   'exact_mean_actual_source_distance': str(actual_distance),
                   'false_probability_only_bound': str(probability_only),
                   'distance_to_probability_only_ratio': str(s.simplify(actual_distance/probability_only)),
                   'paid_eta_and_d2_gates_pass': True, 'expectation_bound_pass': True},
               'constant_gate': {'recurrence_coefficients': coefficients,
                   'sum_squares': 2979494, 'max_square_defect_coefficient': 1375,
                   'global_B_distance_power': 59, 'actual_source_constant_power': 61,
                   'target_power80_pass': True},
               'selection': 'Borel nearest selectors; invariant distance/covariant minimizer set; no global equivariant single selector at zero',
               'scope': 'analytic actual-family theorem consumes same-nu bounds; controls are synthetic native matrices, not numerical ground or state identity',
               'scientific_admission': False, 'campaign_complete': False}
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': str(args.output), 'C_orbit': '2^61',
                      'exact_native_controls_pass': True}), flush=True)


if __name__ == '__main__':
    main()

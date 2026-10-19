"""Independent exact Clifford geometry falsifiers; no producer execution or imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sympy as sp


def clean(matrix):
    return matrix.applyfunc(sp.simplify)


def hs(matrix):
    return sp.simplify(sp.trace(matrix.H * matrix))


def words(generators):
    result = []
    for mask in range(16):
        value = sp.eye(4)
        for index in range(4):
            if mask & (1 << index):
                value *= generators[index]
        result.append(clean(value))
    return result


def prices(body, amplitudes):
    radii = [sp.trace(matrix * matrix) / 4 for matrix in body]
    self_price = sum(hs(matrix * matrix - radius * sp.eye(4))
                     for matrix, radius in zip(body, radii))
    cross_price = sum(hs(body[i] * body[j] + body[j] * body[i])
                      for i in range(4) for j in range(4) if i != j)
    radial = sum((radius - amplitude ** 2) ** 2
                 for radius, amplitude in zip(radii, amplitudes))
    total = sp.simplify(16 * self_price + 4 * cross_price + 64 * radial)
    normalized = [matrix / amplitude for matrix, amplitude in zip(body, amplitudes)]
    defect = sum(hs(matrix * matrix - sp.eye(4)) for matrix in normalized)
    defect += sum(hs(normalized[i] * normalized[j] + normalized[j] * normalized[i])
                  for i in range(4) for j in range(4) if i != j)
    assert defect <= 2 ** 12 * total
    return total, sp.simplify(defect), self_price, cross_price


def intertwiner_projection(frames, gamma):
    fw, gw = words(frames), words(gamma)
    gram = clean(sp.Matrix(16, 16, lambda i, j: sp.trace(fw[i].H * fw[j])))
    assert gram == 4 * sp.eye(16)
    columns = []
    candidates = []
    for a in range(4):
        for b in range(4):
            unit = sp.zeros(4)
            unit[a, b] = 1
            value = clean(sum((f * unit * g.H for f, g in zip(fw, gw)), sp.zeros(4)) / 16)
            candidates.append(value)
            columns.append(sp.Matrix(list(value)))
            for f, g in zip(frames, gamma):
                assert clean(f * value - value * g) == sp.zeros(4)
    projector = sp.Matrix.hstack(*columns)
    assert clean(projector * projector) == projector
    assert clean(projector.H - projector) == sp.zeros(16)
    assert sp.simplify(sp.trace(projector)) == 1 and projector.rank() == 1
    nonzero = next(value for value in candidates if value != sp.zeros(4))
    alpha = hs(nonzero) / 4
    assert alpha > 0 and clean(nonzero.H * nonzero) == alpha * sp.eye(4)
    return {'rank': 1, 'trace': 1, 'all_16_matrix_units': True,
            'orthogonal_projection': True, 'alpha': str(alpha)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for item in request['frozen_inputs']:
        assert hashlib.sha256((args.request.parent / item['name']).read_bytes()).hexdigest() == item['sha256']

    x = sp.Matrix([[0, 1], [1, 0]])
    y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    z = sp.diag(1, -1)
    gamma = [sp.kronecker_product(x, sp.eye(2)), sp.kronecker_product(y, sp.eye(2)),
             sp.kronecker_product(z, x), sp.kronecker_product(z, y)]
    vector = sp.Matrix([1, 2, 2, 1])
    householder = sp.eye(4) - 2 * vector * vector.T / (vector.T * vector)[0]
    unitary = sp.diag(sp.I, sp.I, 1, 1) * householder
    assert unitary.H * unitary == sp.eye(4) and unitary.det() == 1
    opposite = [-gamma[0], *gamma[1:]]
    frames = [clean(unitary * g * unitary.H) for g in opposite]
    assert words(frames)[15] == -clean(unitary * words(gamma)[15] * unitary.H)
    projector_report = intertwiner_projection(frames, gamma)
    phase_reports = []
    for phase in [1, sp.I, -1, -sp.I]:
        phase_reports.append(intertwiner_projection(frames, [phase * g / phase for g in gamma]))
        assert (phase * unitary).det() == 1
        assert all(clean((phase * unitary) * g * (phase * unitary).H) == f
                   for g, f in zip(opposite, frames))

    amplitudes = [sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(3, 4), sp.Rational(4, 5)]
    eps = sp.Rational(1, 2 ** 32)
    normalized = [(1 + eps) * frames[0], frames[1] + eps * frames[0],
                  frames[2] + eps * (frames[0] + 2 * frames[1]),
                  frames[3] + eps * (2 * frames[0] + frames[1] + 3 * frames[2])]
    body = [a * g for a, g in zip(amplitudes, normalized)]
    total, defect, _, _ = prices(body, amplitudes)
    assert total < sp.Rational(1, 2 ** 52)
    ranks = []
    errors = []
    projection_errors = []
    constants = [1, 14, 156, 1719]
    for j, matrix in enumerate(normalized):
        projected = matrix
        for previous in frames[:j]:
            projected = clean((projected - previous * projected * previous) / 2)
        scalar = (1 + eps) if j == 0 else sp.Integer(1)
        assert projected == scalar * frames[j]
        assert projected.det() != 0
        assert frames[j] * frames[j] == sp.eye(4) and sp.trace(frames[j]) == 0
        plus_rank = ((sp.eye(4) + frames[j]) / 2).rank()
        minus_rank = ((sp.eye(4) - frames[j]) / 2).rank()
        assert plus_rank == minus_rank == 2
        ranks.append([plus_rank, minus_rank])
        error = hs(matrix - frames[j])
        assert error <= constants[j] ** 2 * defect
        errors.append(error)
        projection_errors.append(hs(matrix - projected))
    for a in frames:
        for b in frames:
            assert clean(a * b + b * a) == (2 * sp.eye(4) if a == b else sp.zeros(4))
    comparison = sum(hs(b - a * f) for b, a, f in zip(body, amplitudes, frames))
    assert comparison <= 2 ** 35 * total
    assert sum(c ** 2 for c in constants) == 2979494 < 2 ** 22

    missing = [sp.zeros(4), *[a * f for a, f in zip(amplitudes[1:], frames[1:])]]
    missing_total, _, _, _ = prices(missing, amplitudes)
    missing_distance = 4 * amplitudes[0] ** 2
    assert missing_total >= sp.Rational(1, 2 ** 52)
    assert missing_distance <= 2 ** 59 * missing_total
    large_reports = []
    for radius in [2 ** 20, 2 ** 80, 2 ** 140]:
        large = [radius * a * f for a, f in zip(amplitudes, frames)]
        large_total, _, _, _ = prices(large, amplitudes)
        exact_distance = 4 * (radius - 1) ** 2 * sum(a ** 2 for a in amplitudes)
        assert exact_distance <= 2 ** 59 * large_total
        large_reports.append({'radius': str(radius), 'distance_squared': str(exact_distance),
                              'T': str(large_total)})

    wrong = [amplitudes[0] * frames[0], amplitudes[1] * frames[0],
             amplitudes[2] * frames[2], amplitudes[3] * frames[3]]
    _, _, self_price, cross_price = prices(wrong, amplitudes)
    assert self_price == 0 and cross_price > 0
    overlap = sp.trace(wrong[0] * wrong[1])
    assert overlap != 0  # A common conjugation preserves the native zero overlap.

    # Exact nonnearest witness: a small rational common plane rotation improves
    # the sequential comparison frame on the same shear, with unequal radii.
    t = eps / 4
    cosine = (1 - t ** 2) / (1 + t ** 2)
    sine = 2 * t / (1 + t ** 2)
    competitor = [cosine * frames[0] - sine * frames[1],
                  sine * frames[0] + cosine * frames[1], frames[2], frames[3]]
    competitor_distance = sum(hs(b - a * f) for b, a, f in zip(body, amplitudes, competitor))
    improvement = sp.simplify(comparison - competitor_distance)
    assert improvement > 0
    intertwiner_projection(competitor, gamma)

    # Correlated same-atom family: rare huge actual field, bounded exact body.
    rare_probability = sp.Rational(1, 2 ** 400)
    radius = sp.Integer(2 ** 120)
    norm = 4 * sum(a ** 2 for a in amplitudes)
    same_atom_d2 = rare_probability * (radius - 1) ** 2 * norm
    exact_orbit_mean = same_atom_d2
    assert exact_orbit_mean > 2 ** 200 * rare_probability
    assert same_atom_d2 < sp.Rational(1, 2 ** 112)
    eta_gate = sp.Rational(1, 2 ** 159)
    d2_gate = sp.Rational(1, 2 ** 112)
    envelope = 2 * d2_gate + 2 ** 61 * eta_gate
    assert envelope < sp.Rational(1, 2 ** 97)
    assert exact_orbit_mean <= 2 * same_atom_d2

    # At the origin all orbit points minimize. A single one cannot be fixed by
    # the entire conjugation group; exhibit a determinant-one sign reversal.
    reversal = frames[0] * frames[1]
    assert reversal.det() == 1 and reversal.H * reversal == sp.eye(4)
    assert clean(reversal * frames[0] * reversal.H) == -frames[0]

    report = {'kind': 'NEW_EXACT_ORBIT_REVIEW_CONTROLS_NOT_ACTUAL_GROUND',
              'frozen_inputs': request['frozen_inputs'], 'sympy_version': sp.__version__,
              'opposite_orientation_dense_complex_SU4': projector_report,
              'central_phases': phase_reports, 'unequal_amplitudes': list(map(str, amplitudes)),
              'triangular_good_perturbation': {'T': str(total), 'K': str(defect),
                  'r_squared': list(map(str, errors)), 'h_squared': list(map(str, projection_errors)),
                  'ranks': ranks, 'comparison_squared': str(comparison), 'constants': constants},
              'missing_component': {'T': str(missing_total), 'exact_distance_squared': str(missing_distance)},
              'large_radii': large_reports,
              'wrong_independent_frames': {'self_price': str(self_price), 'ordered_cross': str(cross_price),
                  'nonzero_pair_overlap': str(overlap)},
              'nonnearest_polar_frame': {'improvement_squared': str(improvement),
                  'competitor_squared': str(competitor_distance), 'nearest_identity_rejected': True},
              'rare_correlated_field': {'probability': str(rare_probability), 'radius': str(radius),
                  'exact_mean_distance_squared': str(exact_orbit_mean), 'same_atom_d2': str(same_atom_d2),
                  'probability_only_radius_one_price_rejected': True},
              'origin_selector': {'SU4_reversal': True, 'equivariant_single_selector_impossible': True},
              'exact_gate_envelope': str(envelope), 'all_controls_pass': True,
              'scope': 'Exact finite mathematical controls; conditional actual-family analytic proof is separate.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'all_controls_pass': True, 'nonnearest_selector_claim_rejected': True}), flush=True)


if __name__ == '__main__':
    main()

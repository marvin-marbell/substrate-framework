"""Independent exact matrix method controls; no native spectrum or ground computation."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def matrix_unit(n: int, row: int, column: int) -> Any:
    result = sp.zeros(n)
    result[row, column] = 1
    return result


def norm2(vector: Any) -> Any:
    return sp.simplify((vector.conjugate().T * vector)[0])


def exact_controls(depth: int) -> dict[str, Any]:
    l: Any = sp.diag(0, 0, 2, 5)
    omega: Any = sp.eye(4)[:, 0]
    a = sum((matrix_unit(4, r, c) for r, c in ((2, 0), (0, 2), (3, 2), (2, 3))), sp.zeros(4))
    b = sum((matrix_unit(4, r, c) for r, c in ((1, 0), (2, 1), (0, 3))), sp.zeros(4))
    assert a * b != b * a
    phase: Any = sp.diag(1, 1, -1, sp.I)
    letters = [u * z * u.conjugate().T for z in (a, b, b.conjugate().T) for u in (sp.eye(4), phase)]
    vectors = [omega]
    transcript = hashlib.sha256()
    count = 0
    left_inequalities = 0
    null_words = 0
    for size in range(depth + 1):
        for word in itertools.product(range(len(letters)), repeat=size):
            operator: Any = sp.eye(4)
            for index in word:
                operator = letters[index] * operator
            vector = operator * omega
            value = norm2(vector)
            graph = norm2(l * vector)
            spectral_second = sum(l[j, j] ** 2 * abs(vector[j]) ** 2 for j in range(4))
            assert sp.simplify(graph - spectral_second) == 0
            for letter in letters:
                frobenius2 = sum(abs(z) ** 2 for z in letter)
                assert sp.simplify(frobenius2 * value - norm2(letter * vector)) >= 0
                left_inequalities += 1
            if value == 0:
                null_words += 1
                assert all(letter * vector == sp.zeros(4, 1) for letter in letters)
            if len(vectors) < 13:
                vectors.append(vector)
            transcript.update(str((word, tuple(vector), value, graph)).encode())
            count += 1
    columns: Any = sp.Matrix.hstack(*vectors)
    gram: Any = columns.conjugate().T * columns
    coeff: Any = sp.Matrix([sp.I ** j * (j + 1) for j in range(columns.cols)])
    assert sp.simplify((coeff.conjugate().T * gram * coeff)[0] - norm2(columns * coeff)) == 0

    # Positive compression remains positive, but full multiplication is lost.
    projection: Any = sp.diag(1, 1, 0, 1)
    x = matrix_unit(4, 0, 2) + matrix_unit(4, 2, 0)
    y = matrix_unit(4, 2, 3) + matrix_unit(4, 3, 2)
    compressed_x = projection * x * projection
    compressed_y = projection * y * projection
    full_xy = projection * x * y * projection
    wrong_xy = compressed_x * compressed_y
    assert compressed_x == compressed_y == sp.zeros(4)
    assert full_xy[0, 3] == 1 and wrong_xy[0, 3] == 0
    feature_columns: Any = sp.Matrix.hstack(projection * omega, projection * x * omega, projection * y * omega)
    feature_gram: Any = feature_columns.conjugate().T * feature_columns
    assert feature_gram == sp.diag(1, 0, 0)
    assert (projection * x.conjugate().T * x * projection)[0, 0] == 1
    assert (compressed_x.conjugate().T * compressed_x)[0, 0] == 0

    # Zero block, mean and an arbitrary source projection discriminate exactly.
    p0: Any = sp.diag(1, 1, 0, 0)
    mean: Any = omega * omega.conjugate().T
    source_q: Any = sp.diag(0, 1, 0, 1)
    test: Any = sp.Matrix([0, 1, 1, 0])
    whole_mass = norm2((sp.eye(4) - p0) * test)
    mean_mass = norm2((sp.eye(4) - mean) * test)
    source_mass = norm2(source_q * test)
    assert whole_mass == 1 and mean_mass == 2
    assert source_q * test != (sp.eye(4) - p0) * test
    assert source_mass == whole_mass  # Same scalar mass is not the same vector or measure.
    assert norm2(l * source_q * test) == 0 and norm2(l * (sp.eye(4) - p0) * test) == 4

    # Positive finite gaps can collapse into the entire limiting zero block.
    gap_rows = []
    e1: Any = sp.eye(3)[:, 1]
    for n in (2, 4, 8, 16):
        ln: Any = sp.diag(0, sp.Rational(1, n), 2)
        finite_p0: Any = sp.diag(1, 0, 0)
        assert finite_p0 * e1 == sp.zeros(3, 1)
        gap_rows.append({'n': n, 'finite_positive_energy': str(ln[1, 1]), 'finite_deleted_vector_norm2': str(norm2((sp.eye(3)-finite_p0)*e1))})
    limit_p0: Any = sp.diag(1, 1, 0)
    assert (sp.eye(3) - limit_p0) * e1 == sp.zeros(3, 1)

    # A unital static algebra can be isometric but not time-invariant/onto.
    static_l: Any = sp.diag(0, 1, 2)
    static_omega: Any = sp.eye(3)[:, 0]
    v: Any = sp.ones(3, 1)
    static_p: Any = v * v.T / 3
    assert static_p * static_p == static_p
    basis: Any = sp.Matrix.hstack(static_omega, static_p * static_omega)
    evolved: Any = sp.diag(1, -1, 1) * static_p * static_omega
    assert basis.rank() == 2
    assert basis.row_join(evolved).rank() == 3
    resolvent_vector: Any = (static_l - sp.I * sp.eye(3)).inv() * static_p * static_omega
    assert basis.row_join(resolvent_vector).rank() == 3
    restricted_domain: Any = basis.row_join(static_l * basis)
    assert restricted_domain.rank() == 3

    # Canonical purification of an invariant mixed ground, not an invariant pure vector.
    rho: Any = sp.diag(sp.Rational(1, 2), sp.Rational(1, 2), 0, 0)
    sqrt_rho: Any = sp.diag(1 / sp.sqrt(2), 1 / sp.sqrt(2), 0, 0)
    translation: Any = sp.Matrix([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, sp.I]])
    assert translation * rho * translation.conjugate().T == rho
    assert translation * sqrt_rho * translation.conjugate().T == sqrt_rho
    assert l * sqrt_rho == sp.zeros(4)
    assert sp.trace(sqrt_rho.conjugate().T * sqrt_rho) == 1
    assert (rho * (a.conjugate().T * b)).trace() == (sqrt_rho.conjugate().T * a.conjugate().T * b * sqrt_rho).trace()
    assert translation * omega != omega

    # Small first energy alone does not give a small excitation graph norm.
    packet_rows = []
    for n in (2, 4, 8, 16):
        packet_l: Any = sp.diag(0, n ** 3)
        packet_rho: Any = sp.diag(1-sp.Rational(1, n**4), sp.Rational(1, n**4))
        first = (packet_rho * packet_l).trace()
        second = (packet_rho * packet_l ** 2).trace()
        assert first == sp.Rational(1, n) and second == n ** 2
        packet_rows.append({'n': n, 'first_energy': str(first), 'graph_norm_squared': str(second), 'support_upper': n**3})
    return {
        'scope': 'NON_NATIVE_EXACT_MATRIX_METHOD_ONLY',
        'words': {'count': count, 'depth': depth, 'letters_including_adjoints_and_mixed_times': len(letters), 'noncommuting': True, 'null_words': null_words, 'left_inequalities': left_inequalities, 'transcript_sha256': transcript.hexdigest(), 'gram_factorization': 'C.adjoint()*C; exact complex quadratic identity', 'gram_rank': columns.rank()},
        'positive_feature_not_multiplication': {'gram': str(feature_gram), 'full_xy_03': str(full_xy[0,3]), 'compressed_xy_03': str(wrong_xy[0,3]), 'full_xstarx_00': '1', 'compressed_xstarx_00': '0'},
        'whole_zero_block': {'whole_mass': str(whole_mass), 'mean_mass': str(mean_mass), 'source_same_scalar_mass_but_wrong_energy': True, 'finite_gap_rows': gap_rows, 'limit_deleted_norm2': '0'},
        'static_isometry_not_invariant_onto_or_range': {'static_dimension': 2, 'history_dimension_after_one_evolved_letter': 3, 'resolvent_vector_outside_static': True, 'static_algebra': 'span{I,P}, P=ones(3,3)/3'},
        'density_purification': {'trace': '1', 'mixed_invariant_ground': True, 'conjugate_translation_lift_invariant': True, 'chosen_ground_vector_not_invariant': True},
        'first_energy_not_graph_packet': packet_rows,
        'native_ground_or_H_evolution_computed': False,
        'eigensolver_used': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request: dict[str, Any] = json.loads(args.request.read_bytes())
    before = {name: sha((ROOT / name).read_bytes()) for name in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT / request['snapshot']) as archive:
        for name, expected in request['frozen_members'].items():
            assert sha(archive.read(name)) == expected
    proof = json.loads((ROOT / request['proof']).read_bytes())
    receipt = {'schema': 1, 'kind': 'INDEPENDENT_HISTORY_LIMIT_REVIEW_EXACT_METHOD_RECEIPT', 'targets': proof['targets'], 'verdict': proof['verdict'], 'findings': proof['findings'], 'new_exact_method_controls': exact_controls(request['depth']), 'source_input_sha256_before': before, 'source_input_sha256_after': {name: sha((ROOT / name).read_bytes()) for name in request['pins']}, 'runtime': {'elapsed_seconds': time.monotonic()-started}}
    assert receipt['source_input_sha256_before'] == receipt['source_input_sha256_after']
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'scope': 'NON_NATIVE_EXACT_MATRIX_METHOD_ONLY', 'words': receipt['new_exact_method_controls']['words']['count'], 'elapsed_seconds': receipt['runtime']['elapsed_seconds']}))


if __name__ == '__main__':
    main()

"""Exact native geometry and energy-completion controls, not a phase calculation."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from wilson_ground_parent_contact_filter_parenttyped_control_20261012 import (
    divergence, face, move,
)

ROOT = Path(__file__).resolve().parent


def sha(name: str) -> str:
    with (ROOT / name).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cube_control(n: int) -> dict[str, Any]:
    origin = (0, 0, 0)
    faces = [(face(axis, move(origin, axis, n), n), 1) for axis in range(3)]
    faces += [(face(axis, origin, n), -1) for axis in range(3)]
    linear: dict[Any, Fraction] = {}
    for flow, sign in faces:
        for edge, coefficient in flow.items():
            linear[edge] = linear.get(edge, Fraction(0)) + sign * coefficient
    assert all(value == 0 for value in linear.values())
    # Quarter-turn phases on native raw edges; no synthetic ground expectation.
    angles = {(1, origin): 1, (2, move(origin, 1, n)): 2}
    cosines = {0: Fraction(1), 1: Fraction(0), 2: Fraction(-1), 3: Fraction(0)}
    gradient: dict[Any, Fraction] = {}
    phases: list[int] = []
    for flow, sign in faces:
        phase = sum(coefficient * angles.get(edge, 0) for edge, coefficient in flow.items()) % 4
        phases.append(phase)
        cosine = cosines[phase]
        derivative = (cosine + Fraction(1, 2)) / (Fraction(5, 4) + cosine)
        assert abs(derivative) <= 2
        for edge, coefficient in flow.items():
            gradient[edge] = gradient.get(edge, Fraction(0)) + sign * derivative * coefficient
    gradient = {edge: value for edge, value in gradient.items() if value}
    assert gradient and not divergence(gradient, n)
    norm2 = sum((value * value for value in gradient.values()), Fraction(0))
    assert 0 < 64 * norm2 <= 3072 * 6
    return {'N': n, 'cube_Cdag_sigma_zero': True, 'nonlinear_weighted_gradient_norm2': str(norm2),
            'closed_U1_gradient': True, 'quarter_face_phases': phases,
            'cochain_quotient_or_automatic_P_T_rejected': True,
            'point_geometry_not_ground_contact_measurement': True}


def complex_energy_control() -> dict[str, Any]:
    H = sp.diag(0, 1)
    omega = sp.Matrix([1, 0])
    A = sp.Matrix([[0, 1], [1, 0]])
    B = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    X = sp.Matrix.hstack(A * omega, B * omega)
    G = X.H * H * X
    u = sp.Matrix([1, -sp.I])
    full = sp.simplify((u.H * G * u)[0])
    deleted = sp.simplify((u.H * sp.eye(2) * u)[0])
    assert G == sp.Matrix([[1, sp.I], [-sp.I, 1]])
    assert full == 4 and deleted == 2 and full == 2 * (u.H * u)[0]
    return {'full_complex_energy': str(full), 'wrong_deleted_imaginary': str(deleted),
            'safe_real_to_complex_factor2_sharp_method': True,
            'nonnative_matrix_control': True}


def band_control() -> dict[str, Any]:
    L = sp.diag(0, 0, 1, 4)
    P0 = sp.diag(1, 1, 0, 0)
    Q0 = sp.eye(4) - P0
    X = sp.Matrix([[0, 0], [1, 2], [1, 0], [0, 1]])
    root = sp.diag(0, 0, 1, 2)
    T = root * Q0 * X
    band_inverse_root = sp.diag(0, 0, 1, sp.Rational(1, 2))
    BI = band_inverse_root * T
    assert P0 * T == sp.zeros(4, 2)
    assert BI == Q0 * X
    energy_gram = T.H * T
    assert energy_gram == sp.diag(1, 4)
    assert BI.H * BI == sp.eye(2)
    mean_Q = sp.eye(4) - sp.diag(1, 0, 0, 0)
    assert X.H * mean_Q * X == sp.Matrix([[2, 2], [2, 5]])
    assert X.H * mean_Q * X != BI.H * BI
    collapsing = []
    for denominator in (8, 64, 512):
        epsilon = sp.Rational(1, denominator)
        source_mass = 1 / epsilon
        assert epsilon * source_mass == 1 and source_mass > 4
        collapsing.append({'energy': str(epsilon), 'energy_mass': '1', 'TOTAL_mass': str(source_mass)})
    return {'energy_Gram': [[str(v) for v in row] for row in energy_gram.tolist()],
            'whole_P0_before_supported_positiveband': True,
            'mean_ground_removal_rejected': True,
            'energy_completion_NOT_TOTAL_completion': collapsing,
            'nonnative_matrix_control': True}


def normalization_control() -> dict[str, Any]:
    r = Fraction(1, 2)
    real = Fraction(768) / (1-r)**2
    complex_bound = 2 * real
    assert real == 3072 and complex_bound == 6144
    rows = []
    for n in (4, 6, 8, 16, 64):
        q = sp.Rational(4, n)  # analytic lower for first native q; not a measured q value
        upper = 8 * sp.sqrt(2) + 128 * sp.sqrt(3) / q
        assert upper > sp.Rational(1, 16)
        rows.append({'N': n, 'q_lower': str(q), 'available_residual_upper': str(upper),
                     'certifies_one_over_16': False, 'actual_residual_lower': False})
    return {'reused_real_upper': str(real), 'safe_complex_upper': str(complex_bound),
            'unit_closed_h_sigma2': '1/q^2', 'b_band_mass_upper': '6144/(alpha*q)',
            'S_b_over_q_upper': '6144/(alpha*q^2)',
            'volume_removed_not_IR_closure': rows,
            'principal_r1_or_positive_Gram_NOT_proved': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    before = {name: sha(name) for name in request['pins']}
    assert before == request['pins']
    result = {'kind': 'PARENT_SAME_HISTORY_MAGNETIC_ENERGY_BAND_NATIVE_GEOMETRY_AND_METHOD_CONTROLS_NOT_PHASE',
              'native_cube': [cube_control(n) for n in (4, 6, 8)],
              'complex_energy': complex_energy_control(), 'positiveband': band_control(),
              'normalization': normalization_control(), 'proof': request['proof'],
              'source_input_sha256_before': before,
              'source_input_sha256_after': {name: sha(name) for name in request['pins']},
              'native_ground_or_fullH_numerical_evolution_computed': False,
              'physical1_IR_lower_stiffness_phase_atom': False, 'scientific_admission': False,
              'campaign_complete': False}
    assert result['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print('Exact native cochain/complex/wholeP0/positiveband/scaling boundaries exercised; no IR mechanism.')

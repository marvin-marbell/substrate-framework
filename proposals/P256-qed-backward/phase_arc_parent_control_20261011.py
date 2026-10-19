"""Uniform periodic phase arcs and filtered principal vectors, not raw IR order."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import sympy as sp
from magnetic_mechanism_scientist_20261011 import incidence

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def arc_sequence(ell: int) -> dict[str, Any]:
    # Accepted sharp coercivity: Rnew<8193Lambda<2^14Lambda, outgoing E²<Rnew/64.
    assert 8193 < 2 ** 14
    K = 2 ** 8200
    rootK = 2 ** 4100
    assert rootK * rootK == K
    M = 2 ** ell
    a = F(1, 2 ** (2 * ell) * rootK)
    delta = a / 2
    arc_upper = F(1, M) + rootK * a * (M - 1)
    assert M * a < 1 and arc_upper < F(2, M)
    s = delta / M
    r = 1 - s
    assert F(1, 2) < r < 1 and 1 - r > 0
    # Normalized source b_r/(2r), first harmonic1; not q-dependent weakening.
    error_squared = 64 * s * s / (delta * delta) + 16 * arc_upper
    assert error_squared < F(128, M)
    # Honest precision wrongcase: binary64 collapses this legal rational r to1.
    assert float(r) == 1.0 and not (0 < float(r) < 1)
    # Point phase atom would violate the finite-electric regularity premise.
    assert not (F(1) <= arc_upper)
    return {'ell': ell, 'arc_length_exact': str(a), 'rotated_disjoint_arc_count': M,
            'arc_probability_upper_exact': str(arc_upper),
            'one_minus_r_exact': str(s), 'normalized_principal_L2_error_squared_upper_exact': str(error_squared),
            'error_squared_dyadic_cap': f'2^(7-{ell})',
            'finite_E_regular_phase_atom_wrongcase_rejected': True,
            'binary64_r_equal_one_wrongcase_rejected': True,
            'actual_ground_arc_probability_measured': False}


def face_generator(n: int) -> dict[str, Any]:
    D, C, _ = incidence(n)
    c = np.asarray(C[2].toarray()).ravel()
    assert int(c @ c) == 4 and np.array_equal(D @ c, np.zeros(n ** 3, dtype=np.int64))
    # p_B=c.E_full/4 gives [p_B,e^{iB}]=e^{iB}; HC/rho not deleted.
    assert F(int(c @ c), 4) == 1
    assert F(int(c @ c), 16) * 4 == 1
    return {'N': n, 'closed_native_face': True, 'phase_rotation_generator': 'p_B=Cface.E_full/4',
            'phase_rotation_coefficient_exact': '1',
            'p_B_second_moment_upper': '(1/4)sum_four_face_edges<E_e²> <= Rnew/64 <2^8200',
            'finite_geometry_only_not_ground_measure': True}


def branch_operator_control() -> dict[str, Any]:
    U = sp.diag(1, -1)
    omega = sp.Matrix([0, 1])
    # Every sine Fourier term is zero AT the negative-real branch point.
    for k in range(1, 9):
        assert (U ** k - U ** (-k)) / (2 * sp.I) == sp.zeros(2)
    midpoint = sp.zeros(2)
    standard_principal = sp.diag(0, -sp.pi / 2)
    assert (midpoint - standard_principal) * omega == sp.pi * omega / 2
    return {'unitary_branch_eigenvalue': '-1', 'Abel_radial_midpoint_exact': '0',
            'standard_principal_value_exact': '-pi/2',
            'unconditional_standard_Borel_point_limit_rejected': True,
            'scope': 'Exact two-dimensional atom control; finite-electric arc premise is absent here.'}


def averaged_momentum_control(n: int) -> dict[str, Any]:
    volume = n ** 3
    x = np.array(list(np.ndindex(n, n, n)), dtype=float)
    vectors = np.column_stack((x[:, 0] / 7, x[:, 1] / 11, x[:, 2] / 13)).reshape(n, n, n, 3)
    modes = np.fft.fftn(vectors, axes=(0, 1, 2)) / np.sqrt(volume)
    averaged_modes = float(np.sum(np.abs(modes) ** 2) / volume)
    averaged_local = float(np.sum(vectors * vectors) / volume)
    assert abs(averaged_modes - averaged_local) < 1e-12 * max(1, averaged_local)
    constant = np.ones((n, n, n))
    coherent_mode = np.fft.fftn(constant) / np.sqrt(volume)
    assert abs(float(abs(coherent_mode[0, 0, 0]) ** 2) - volume) < 1e-10
    assert abs(float(np.sum(np.abs(coherent_mode) ** 2) / volume) - 1) < 1e-12
    return {'N': n, 'averaged_Parseval_defect_float': averaged_modes - averaged_local,
            'single_mode_can_equal_volume_times_average': True,
            'scope': 'Nonorthogonal finite-vector FFT method control; no native ground covariance/interval claim.'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    assert all(digest(ROOT / name) == pin for name, pin in request['pins'].items())
    result = {'kind': 'UNIFORM_FULL_E_PERIODIC_PHASE_ARC_AND_FILTERED_PRINCIPAL_DOMAIN_NOT_IR_ORDER',
              'face_generators': [face_generator(n) for n in (4, 6, 8)],
              'uniform_arc_sequences': [arc_sequence(ell) for ell in (4, 8, 16, 32)],
              'branch_operator': branch_operator_control(),
              'momentum_average': [averaged_momentum_control(n) for n in (4, 6, 8)],
              'source_normalization': 'b_r/(2r), fixed first harmonic1',
              'periodic_phase_not_raw_unwrapped_B': True,
              'wholeP0_ground_limit_still_required': True,
              'pointwise_IR_covariance_not_paid': True,
              'ground_identification_phase_stiffness_or_pole': False,
              'campaign_complete': False}
    assert all(digest(ROOT / name) == pin for name, pin in request['pins'].items())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'N': [4, 6, 8], 'arc_cap_exponent': 8200, 'branch_wrongcase': True,
                      'Parseval_not_pointwise': True, 'IR_order_or_photon': False}))


if __name__ == '__main__':
    main()

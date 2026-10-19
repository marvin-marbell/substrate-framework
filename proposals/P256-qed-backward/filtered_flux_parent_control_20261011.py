"""Native frozen-reference flux filters and whole-ground-block noninference."""
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


def reference_geometry(n: int) -> dict[str, Any]:
    D, C, S = incidence(n)
    assert (S @ C).nnz == 0 and (C @ D.T).nnz == 0
    volume = n ** 3
    yz = C[2::3]
    # Axis-zero matching dimers contain only x edges. Every yz face is external.
    assert yz[:, ::3].nnz == 0
    norms = np.asarray(yz.multiply(yz).sum(axis=1)).ravel()
    assert np.array_equal(norms, np.full(volume, 4))
    assert (D @ yz.T).nnz == 0
    assert (yz.T @ np.ones(volume, dtype=np.int64) == 0).all()
    # Distinct closed character labels remain orthogonal in external k=0.
    rows = {tuple(zip(yz[i].indices.tolist(), yz[i].data.tolist())) for i in range(volume)}
    assert len(rows) == volume
    assert all(tuple((edge, -value) for edge, value in row) not in rows for row in rows)
    return {'N': n, 'yz_face_count': volume, 'integer_closed_character_norm_squared': 4,
            'matching_internal_edges_in_yz_character': 0,
            'linear_q0_curl_gradient_zero': True,
            'source_domain': 'N4/6/8 integer geometry controls only. Spectral claims use accepted all1+63 matching-product g with external k=0 and frozen beta0 reference premises at even N>=8, not physical1/fullPhi ground.'}


def reference_spectrum(r: F, cut: int) -> dict[str, Any]:
    x = r * r
    weights = [2 * x ** k / (k * k) for k in range(1, cut + 1)]
    energies = [256 * k * k for k in range(1, cut + 1)]
    m0 = sum(weights, F(0))
    m0_tail = 2 * x ** (cut + 1) / ((cut + 1) ** 2 * (1 - x))
    m1 = 512 * x / (1 - x)
    m2 = 131072 * x * (1 + x) / (1 - x) ** 3
    retained_m1 = sum((w * e for w, e in zip(weights, energies, strict=True)), F(0))
    m1_tail = 512 * x ** (cut + 1) / (1 - x)
    assert retained_m1 + m1_tail == m1
    assert m2 >= sum((w * e * e for w, e in zip(weights, energies, strict=True)), F(0))
    # Abel smoothing does not erase the matching reference's gap.
    assert min(energies) == 256 and m0 > 0 and m0_tail > 0
    assert m1 > 0  # rejects substituting the linear q=0 curl gradient
    lam = F(32)
    filtered = sum((w * (lam / (e + lam)) ** 2 for w, e in zip(weights, energies, strict=True)), F(0))
    filtered_tail = m0_tail * (lam / (256 * (cut + 1) ** 2 + lam)) ** 2
    assert filtered > 0 and filtered < m0
    return {'r_exact': str(r), 'retained_harmonics': cut,
            'harmonic_energies_exact': energies, 'm0_interval_exact': [str(m0), str(m0 + m0_tail)],
            'm1_exact': str(m1), 'm2_exact': str(m2),
            'resolvent_filter32_m0_interval_exact': [str(filtered), str(filtered + filtered_tail)],
            'sharp_lowpass_below256_excited_weight_exact': '0',
            'scope': 'Conditional entire matching-reference ground at even N>=8 and external yz characters. Full periodic Abel source, not actual physical1 spectral data.',
            'raw_unwrapping_r_to1_contact_diverges': True,
            'UV_contact_not_a_photon_no_go': True}


def filtered_IR_boundary(q: F) -> dict[str, Any]:
    # Actual measures: q delta_q + delta_256 versus Lebesgue[q,2q] + delta_256.
    lam = 4 * q
    ultraviolet = (lam / (256 + lam)) ** 2
    atom_IR = q * (lam / (q + lam)) ** 2
    continuum_IR = lam ** 2 * (1 / (q + lam) - 1 / (2 * q + lam))
    assert atom_IR == 16 * q / 25 and continuum_IR == 8 * q / 15
    assert ultraviolet < q * q / 4096
    atom_total_m1 = 256 + q * q
    continuum_total_m1 = 256 + 3 * q * q / 2
    assert atom_total_m1 > 256 and continuum_total_m1 > 256
    assert atom_IR > 0 and continuum_IR > 0
    return {'q_exact': str(q), 'filter_lambda_exact': str(lam),
            'UV_filtered_m0_exact': str(ultraviolet),
            'atom_filtered_IR_m0_exact': str(atom_IR),
            'atomless_filtered_IR_m0_exact': str(continuum_IR),
            'nonzero_UV_total_contact_does_not_exclude_massless_atom': True,
            'filtered_weight_linear_in_q_does_not_force_atom': True,
            'scope': 'Exact spectral method control, NOT a measured native ground/IR distribution.'}


def whole_ground_control() -> dict[str, Any]:
    q = sp.Rational(1, 16)
    L = sp.diag(0, 0, q, 256)
    omega = sp.Matrix([1, 0, 0, 0])
    P0 = sp.diag(1, 1, 0, 0)
    A = sp.Matrix([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    insertion = A * omega
    mean_connected = insertion - omega * (omega.dot(insertion))
    excited = (sp.eye(4) - P0) * insertion
    lam = 4 * q
    filt = sp.diag(1, 1, lam / (q + lam), lam / (256 + lam))
    wrong = (filt * mean_connected).dot(filt * mean_connected)
    correct = (filt * excited).dot(filt * excited)
    assert wrong - correct == 1
    assert P0 * L == sp.zeros(4)
    return {'wholeP0_vs_mean_filtered_mass_difference_exact': '1',
            'correct_filtered_excited_mass_exact': str(correct),
            'rho_or_mean_is_not_P0': True, 'scope': 'Independent exact degenerate-ground method control.'}


def full_law_contact() -> dict[str, Any]:
    theta, r = sp.symbols('theta r', real=True)
    f = 2 * r * sp.sin(theta) - r * r * sp.sin(2 * theta)
    psi = sp.exp(theta) * (1 + theta * theta)
    def kinetic(value: Any) -> Any:
        return -64 * sp.diff(value, theta, 2)
    commutator = kinetic(f * psi) - f * kinetic(psi)
    double = f * commutator - (kinetic(f * f * psi) - f * kinetic(f * psi))
    assert sp.simplify(double - 128 * sp.diff(f, theta) ** 2 * psi) == 0
    assert sp.simplify(double - 64 * sp.diff(f, theta) ** 2 * psi) != 0
    return {'core_double_commutator_coefficient_exact': 128,
            'ground_excited_first_form_moment_coefficient_exact': 64,
            'wrong_half_contact_rejected': True,
            'commuting_terms': 'All native charged Peierls CAR/Phi/SU4 multiplier terms and collective HC commute with scalar F(Theta); other kinetic variables commute. This does not compute their ground state.',
            'spectral_antecedent': 'Actual stationary minimizing GROUND, closed insertion form domain and ENTIRE P0 removal. No source/trial substitution.'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    assert all(digest(ROOT / name) == pin for name, pin in request['pins'].items())
    result = {'kind': 'NATIVE_REFERENCE_FILTERED_FLUX_AND_WHOLEP0_METHOD_NOT_PHYSICAL1_PHASE',
              'geometry': [reference_geometry(n) for n in (4, 6, 8)],
              'reference_spectra': [reference_spectrum(r, 24) for r in (F(1, 4), F(1, 2), F(3, 4))],
              'IR_filter_boundary': [filtered_IR_boundary(F(1, 2 ** k)) for k in range(1, 9)],
              'whole_ground': whole_ground_control(), 'full_law_contact': full_law_contact(),
              'native_actual_ground_IR_measure_computed': False,
              'physical1_phase_stiffness_or_pole': False, 'campaign_complete': False}
    assert all(digest(ROOT / name) == pin for name, pin in request['pins'].items())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'N': [4, 6, 8], 'Abel_reference_harmonics': 24, 'wholeP0_rejection_exact': 1,
                      'UV_does_not_exclude_atom': True, 'phase_or_photon': False}))


if __name__ == '__main__':
    main()

"""Actual same-law uniform variational bounds; no phase, source or pole adoption."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any
import zipfile

import numpy as np
import sympy as sp
from scipy.linalg import polar

ROOT = Path(__file__).resolve().parent
PROOF_NAME = "joined_phi_u1_uniform_sharp_gauss_proof_20261010.json"
PROFILE_NAME = "joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json"
PINNED = {
    PROOF_NAME: "fb99f8b2accadee1f868922a972d0dbbb46dacc06b88fef61dfe4bc8e3d44dfb",
    PROFILE_NAME: "1c79b1a74c4236079645ebf9765a48c1c2263ec7b7209e9f103704c98752da94",
    "joined_phi_u1_uniform_gauss_limit_consumer.py": "37086f9ed5faa3b14dbb3fa9061393c527d61226b5288349bc6be04d07cbfafd",
    "joined_phi_u1_uniform_gauss_limit_epoch4_20261010_actual.json": "3661b8e5f6420da11313ae5d282acdaeba37fb45f9c555f9f6f6484a739f964d",
    "joined_phi_u1_uniform_electric_current_bridge_proof_20261010.json": "b68accf76f67d368bf2e747ac633a899b020963b5742bb152dc8c11c6bef6a0e",
    "joined_phi_u1_ir_uniform_charged_parent_packet_20261010.json": "ac143879fab8af1469c12b14cf0781c9bf84bcc87239ae078e329c2a1f8af75c",
}
SOURCE_BYTES = {Path(__file__).name: Path(__file__).read_bytes()}
INPUT_BYTES = {name: (ROOT / name).read_bytes() for name in PINNED}
for _name, _expected in PINNED.items():
    if hashlib.sha256(INPUT_BYTES[_name]).hexdigest() != _expected:
        raise RuntimeError(f"Input binding changed: {_name}")
from joined_phi_u1_uniform_gauss_limit_consumer import basis

SOURCE_BYTES["joined_phi_u1_uniform_gauss_limit_consumer.py"] = INPUT_BYTES[
    "joined_phi_u1_uniform_gauss_limit_consumer.py"
]
sys.set_int_max_str_digits(0)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_prices() -> dict[str, Any]:
    lam, chi, fast = F(2) ** 8192, F(2) ** -4096, F(2) ** 10240
    eta, t = F(2) ** -7168, F(2) ** 7168
    assert chi / eta == chi * t == fast * eta == fast / t == F(2) ** 3072
    assert lam * eta**2 == F(2) ** -6144
    # Exact exponential-series upper, not a rounded numerical assertion.
    exp5_partial = sum((F(5**n, math.factorial(n)) for n in range(41)), F(0))
    exp5_tail = F(5**41, math.factorial(41)) / (1 - F(5, 42))
    assert exp5_partial + exp5_tail < 2**8
    assert math.factorial(16) < 2**45
    assert F(17, 8) ** 2 < 5
    assert 8 + 45 + 32 * 3 == 149 < 150
    # log2<1, log(t)=7168log2; this is an outward rational upper.
    colour_log_upper = F(450 + 48 * 7168 + 24)
    coefficient = F(60) + F(9, 2) + F(27, 8) + F(7, 5) * colour_log_upper
    assert coefficient < 2**19
    upper = coefficient * F(2) ** 3072 + 3000 * lam * eta**2 + 262144
    assert upper < F(2) ** 3092 < lam
    R = upper + 8192 * lam
    assert R / lam < 8193 < 2**14
    assert 2 * R / lam < 2**15
    assert R / fast < F(2) ** -2034
    # Squared current bound uses exactly the archived four-site support.
    current_squared_upper = F(2) ** 28 * 4 * sp.sqrt(sp.Integer(2) ** 15)
    assert current_squared_upper < 2**38
    assert F(2) ** 28 * 4**2 * F(2) ** 15 == F(2) ** 47
    contact_squared_upper = F(2) ** 28 * 4 * sp.sqrt(sp.Integer(2) ** 15)
    assert contact_squared_upper < F(2) ** 38
    return {
        "arithmetic": "Exact Fraction comparisons and symbolic roots; displays are bounds, not measured Gibbs/ground energies.",
        "eta_Phi": "2^-7168", "t_colour": "2^7168",
        "Haar_ball_lower": "2^-150 r^32", "log_Z_negative_per_V_upper": "450log2+48logt+24",
        "colour_Dirichlet_per_V_upper": "(27/8) chi t",
        "trial_coefficient_log_upper": str(coefficient),
        "trial_energy_per_V_upper": "2^3092", "R_over_lambda_upper": "8193",
        "Phi_HS_fourth_per_site_upper": "2^15", "colour_alignment_W_per_site_upper": "2^-2034",
        "local_current_squared_upper": "2^38", "current_cutoff_squared_error_upper": "2^47/M",
        "diagonal_contact_mean_abs_upper": "2^19",
        "no_pointwise_momentum_or_U1_flux_bound": True,
        "old_constant_Haar_price": "(42/5)F; incorrectly setting this to0 would invalidate the trial",
        "e5_exact_series_upper_below256": True,
    }


def haar_and_wick_controls() -> dict[str, Any]:
    matrices, _ = basis()
    zero = sp.zeros(4)
    casimir_T = sum((m * m / 8 for m in matrices), zero)
    casimir_B = sum((m * m for m in matrices), sp.zeros(4))
    assert casimir_T == sp.Rational(15, 8) * sp.eye(4)
    assert casimir_B == 15 * sp.eye(4)
    same = 4 * (2 * sp.trace(casimir_T * casimir_T) + sum(
        (sp.trace(a * b * a * b) / 64 for a in matrices for b in matrices), sp.Integer(0)))
    different = 2 * (sp.trace(casimir_T * casimir_T) + sum(
        (sp.trace(a * b * a * b) / 64 for a in matrices for b in matrices), sp.Integer(0)))
    assert 4 * same + 12 * different == 750
    assert 4 * same + 6 * different != 750
    # Four distinct edges/face; Delta_total C=(15/2)sum ReTr Uface.
    U = sp.diag(sp.I, -sp.I, 1, 1)
    face_laplacian = 4 * sp.re(sp.trace(casimir_T * U))
    wrong_B_laplacian = 4 * sp.re(sp.trace(casimir_B * U))
    assert face_laplacian == 15 and wrong_B_laplacian == 120
    assert 4 * sp.re(sp.trace(casimir_T * -sp.eye(4))) == -30
    assert 4 * sp.re(sp.trace(casimir_T * sp.eye(4))) == 30
    assert F(3, 20) * F(1, 4) * 90 == F(27, 8)
    rng = np.random.default_rng(604918)
    points = []
    for delta in (1 / 128, 1 / 32, 1 / 8):
        noise = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        G = np.eye(4) + delta * noise / np.linalg.norm(noise)
        Q, _ = polar(G)
        determinant = np.linalg.det(Q)
        Uactual = Q * np.exp(-0.25j * np.angle(determinant))
        input_distance = float(np.linalg.norm(G - np.eye(4)))
        unitary_defect = float(np.linalg.norm(Uactual.conj().T @ Uactual - np.eye(4)))
        determinant_defect = float(abs(np.linalg.det(Uactual) - 1))
        assert input_distance <= delta * (1 + 1e-12)
        assert np.linalg.norm(Uactual - np.eye(4)) <= 6 * delta
        assert unitary_defect < 1e-12 and determinant_defect < 1e-12
        points.append(dict(delta=delta,polar_I_distance=float(np.linalg.norm(Q-np.eye(4))),
                           SU4_I_distance=float(np.linalg.norm(Uactual-np.eye(4))),
                           unitary_defect=unitary_defect,determinant_defect=determinant_defect))
    Qwrong, _ = polar((1 + 1j / 16) * np.eye(4))
    wrong_det = float(abs(np.linalg.det(Qwrong) - 1))
    assert wrong_det > 0.2
    return dict(scope="Exact generator/Wick controls plus finite float polar-ball points; not Haar/Gibbs sampling.",
                Haar_Casimir_T="15/8",wrong_B_normalization_factor=8,
                negative_pointwise_face_Laplacian=-30,positive_pointwise_face_Laplacian=30,
                pointwise_Laplacian_positivity_not_assumed=True,wrong_determinant_correction_omission=wrong_det,
                ordered_Jordan_Wick="750 eta^2",polar_SU4_points=points)


def gauss_controls(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for N in (4, 6, 8, 12, 16):
        coords = np.indices((N, N, N))
        parity = np.where(np.sum(coords, axis=0) % 2 == 0, 1, -1)
        rho = 128 * parity
        k0 = 64 * parity
        divergence = k0 - np.roll(k0, 1, axis=0)
        assert np.array_equal(divergence, rho)
        V = N**3
        assert int(np.count_nonzero(parity == 1)) * 4 == 2 * V
        assert int(np.sum(rho)) == 0
        whole = F(64 * int(np.sum(k0**2)), V)
        coulomb = F(64 * int(np.sum(rho**2)), 12 * V)
        transverse = whole - coulomb
        assert whole == 262144 and coulomb == F(262144, 3) and transverse == F(524288, 3)
        assert 2 * V != 63
        rows.append(dict(N=N,V=V,each_species_particles=2*V,charged_flavours=64,
                         electric_per_V=str(whole),HC_per_V=str(coulomb),HE_per_V=str(transverse)))
    current = [8 * sum((F(p['load']) * sum((int(a['length']) * (abs(F(a['c0']))+abs(F(a['ci'])))
                   for a in p['paths'] if int(a['axis']) == j), F(0)) for p in profiles[1:]), F(0)) for j in range(3)]
    assert all(0 < c < 2**14 for c in current)
    return dict(rows=rows,actual_profile_current_coefficients=list(map(str,current)),species_not_volume_dependent=True,
                independent_HC_deletion_changes_total=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    proof = json.loads(INPUT_BYTES[PROOF_NAME])
    data = json.loads(INPUT_BYTES[PROFILE_NAME])
    profiles = data['primitive_profiles']
    assert [int(p['load']) for p in profiles] == [1, 1, 63]
    result: dict[str, Any] = dict(kind='NEW_UNIFORM_SHARP_PHYSICAL_GAUSS_TRIAL_BOUND_CONSUMER',
                  own_source_sha256=sha(SOURCE_BYTES[Path(__file__).name]),
                  source_sha256={name:sha(raw) for name,raw in SOURCE_BYTES.items()},
                  input_sha256={name:sha(raw) for name,raw in INPUT_BYTES.items()},
                  governor=proof['governor'],proof=proof,prices=exact_prices(),
                  Haar_Wick_controls=haar_and_wick_controls(),Gauss_controls=gauss_controls(profiles),
                  runtime=dict(executable=sys.executable,Python=platform.python_version(),numpy=np.__version__,sympy=sp.__version__),
                  normalizer='Exact positive Z_N integral with analytic lower/upper, not numerically evaluated.',
                  campaign_complete=False,no_U1_B2_or_source_adoption=True)
    result['elapsed_seconds'] = time.monotonic()-started
    result['sources_unchanged'] = all((ROOT/name).read_bytes()==raw for name,raw in SOURCE_BYTES.items())
    result['inputs_unchanged'] = all((ROOT/name).read_bytes()==raw for name,raw in INPUT_BYTES.items())
    if not result['sources_unchanged'] or not result['inputs_unchanged']:
        raise RuntimeError('Sources/inputs changed during execution')
    payload = (json.dumps(result,indent=2)+'\n').encode()
    args.output.write_bytes(payload)
    with zipfile.ZipFile(args.output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in SOURCE_BYTES.items(): archive.writestr('sources/'+name,raw)
        for name,raw in INPUT_BYTES.items(): archive.writestr('inputs/'+name,raw)
        archive.writestr('result/'+args.output.name,payload)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Half-filled sea selects a Clifford four-frame from an HS budget, not a penalty.

The proof is in sea_selection.md. This CLI computes actual occupied eigenvalues
for unrestricted traceless Hermitian orthonormal frames and the existing local
RotationBand couplings. No Dirac spectrum is assumed for the alternatives.
"""
from __future__ import annotations

import argparse
import itertools
import json
from typing import Any

import numpy as np
from scipy.linalg import expm

from rotation_band import HOPPING, I4, MIXING, RotationBand, spatial_rotation

SELECTED = np.concatenate((MIXING[None], HOPPING))
QUADRATIC_PAIRS = list(itertools.combinations_with_replacement(range(4), 2))


def matrix_unit_basis():
    """All 15 traceless Hermitian matrix-unit directions, tr(B_a B_b)=4 delta."""
    units = np.eye(4, dtype=complex)
    basis = []
    for i, j in itertools.combinations(range(4), 2):
        unit = np.outer(units[:, i], units[:, j])
        basis.extend((np.sqrt(2) * (unit + unit.conj().T),
                      -1j * np.sqrt(2) * (unit - unit.conj().T)))
    for k in range(1, 4):
        diagonal = np.zeros(4)
        diagonal[:k], diagonal[k] = 1, -k
        basis.append(np.sqrt(4 / (k * (k + 1))) * np.diag(diagonal))
    return np.array(basis)


def gram(frame):
    return np.einsum('aij,bji->ab', frame, frame).real / 4


def random_frame(rng, basis):
    coefficients, _ = np.linalg.qr(rng.normal(size=(15, 4)))
    return np.einsum('am,aij->mij', coefficients, basis)


def coefficients(band, momenta):
    return np.column_stack(([band.mass(p) for p in momenta],
                            band.v * np.sin(band.a * momenta) / band.a))


def quadratic_design(x):
    # Off-diagonal columns multiply anticommutators, diagonal ones A_mu^2-I.
    return np.column_stack([x[:, i] * x[:, j] for i, j in QUADRATIC_PAIRS])


def rank_report(x) -> dict[str, Any]:
    design = quadratic_design(x)
    singular = np.linalg.svd(design, compute_uv=False)
    tolerance = max(design.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    return dict(momentum_count=len(x), quadratic_rank=rank,
                columns=[list(pair) for pair in QUADRATIC_PAIRS],
                singular_values=singular.tolist(),
                rank_tolerance=float(tolerance),
                saturation_forces_full_clifford=(rank == 10))


def minimal_rank_witness(band) -> dict[str, Any]:
    """Ten momenta with an analytic nonzero determinant, even on the L=3 cube."""
    k = 2 * np.pi / (3 * band.a)
    axes = np.eye(3)
    momenta = np.array([np.zeros(3)]
                       + [sign * k * axes[i] for i in range(3) for sign in [1, -1]]
                       + [k * (axes[i] + axes[j]) for i, j in itertools.combinations(range(3), 2)])
    x = coefficients(band, momenta)
    measured = float(abs(np.linalg.det(quadratic_design(x))))
    mu, t = x[1, 0], x[1, 1]
    predicted = float(8 * band.m**2 * mu**3 * abs(t)**15)
    return dict(**rank_report(x), momenta=momenta.tolist(),
                determinant_absolute=measured, analytic_determinant_absolute=predicted,
                determinant_relative_error=abs(measured / predicted - 1))


def reframe(matrix, frame):
    """Keep RotationBand's exact local scalar couplings, vary only orbital frame."""
    weights = np.einsum('aij,ji->a', SELECTED, matrix) / 4
    return np.einsum('a,aij->ij', weights, frame)


def h_from_couplings(band, momenta, frame):
    onsite, terms = band.couplings()
    h = np.broadcast_to(reframe(onsite, frame), (len(momenta), 4, 4)).copy()
    for displacement, coupling in terms:
        phase = np.exp(1j * band.a * (momenta @ displacement))
        varied = reframe(coupling, frame)
        h += phase[:, None, None] * varied
        h += phase.conj()[:, None, None] * varied.conj().T
    return h


def algebra_defects(frame):
    return np.array([frame[i] @ frame[j] + frame[j] @ frame[i]
                     - (2 * I4 if i == j else 0)
                     for i, j in QUADRATIC_PAIRS])


def evaluate(frame, band, momenta) -> dict[str, Any]:
    x = coefficients(band, momenta)
    h = h_from_couplings(band, momenta, frame)
    eigenvalues = np.linalg.eigvalsh(h)
    occupied = eigenvalues[:, :2].sum(axis=1)
    q = np.einsum('pij,pji->p', h, h).real
    lower_bound = -np.sqrt(np.maximum(q, 0))
    excess = occupied - lower_bound
    pair_splitting = ((eigenvalues[:, 1] - eigenvalues[:, 0])**2
                     + (eigenvalues[:, 3] - eigenvalues[:, 2])**2) / 2
    saturation = h @ h - (q / 4)[:, None, None] * I4
    defects = algebra_defects(frame)
    return dict(sea_energy=float(occupied.sum()),
                sea_lower_bound=float(lower_bound.sum()),
                excess_energy=float(excess.sum()),
                minimum_pointwise_excess=float(excess.min()),
                maximum_pointwise_excess=float(excess.max()),
                maximum_saturation_error=float(np.max(np.linalg.norm(saturation, axis=(1, 2)))),
                maximum_clifford_defect=float(np.max(np.linalg.norm(defects, axis=(1, 2)))),
                gram_error=float(np.linalg.norm(gram(frame) - np.eye(4))),
                trace_error=float(np.max(abs(np.trace(frame, axis1=1, axis2=2)))),
                hermiticity_error=float(np.linalg.norm(frame - frame.conj().transpose(0, 2, 1))),
                local_coupling_error=float(np.max(np.linalg.norm(
                    h - np.einsum('pa,aij->pij', x, frame), axis=(1, 2)))),
                fixed_budget_error=float(np.max(abs(q - 4 * np.sum(x**2, axis=1)))),
                exact_pair_splitting_identity_error=float(np.max(abs(q - occupied**2 - pair_splitting))),
                sample_eigenvalues=eigenvalues[:3].tolist())


def rotated_mass_frame(theta, basis):
    """An exact SO(15) plane rotation, NOT Gram restoration after perturbation."""
    # This matrix unit is orthogonal to the four selected directions. It commutes
    # with M,T3 and anticommutes with T1,T2; hence strictly positive curvature.
    b = np.diag([1., -1., 1., -1.]).astype(complex)
    u = np.einsum('aij,ji->a', basis, MIXING).real / 4
    w = np.einsum('aij,ji->a', basis, b).real / 4
    generator = np.outer(w, u) - np.outer(u, w)
    rotation = expm(theta * generator)
    coordinates = np.einsum('aij,mji->am', basis, SELECTED).real / 4
    frame = np.einsum('am,aij->mij', rotation @ coordinates, basis)
    return frame, float(np.linalg.norm(rotation.T @ rotation - np.eye(15)))


def curvature_probe(band, momenta, basis) -> dict[str, Any]:
    x = coefficients(band, momenta)
    radius = np.linalg.norm(x, axis=1)
    predicted = float(np.sum(2 * x[:, 0]**2 * (x[:, 0]**2 + x[:, 3]**2) / radius**3))
    selected_energy = evaluate(SELECTED, band, momenta)['sea_energy']
    rows = []
    for theta in [0.1, 0.03, 0.01, 0.003]:
        plus, orthogonality = rotated_mass_frame(theta, basis)
        minus, _ = rotated_mass_frame(-theta, basis)
        plus_result, minus_result = evaluate(plus, band, momenta), evaluate(minus, band, momenta)
        measured = (plus_result['sea_energy'] + minus_result['sea_energy']
                    - 2 * selected_energy) / theta**2
        rows.append(dict(theta=theta, excess_energy=plus_result['excess_energy'],
                         gram_error=plus_result['gram_error'],
                         so15_orthogonality_error=orthogonality,
                         clifford_defect=plus_result['maximum_clifford_defect'],
                         measured_second_derivative=float(measured),
                         relative_curvature_error=float(abs(measured / predicted - 1))))
    return dict(analytic_second_derivative=predicted, path=rows)


def selected_consumer_probe(band, momenta) -> dict[str, Any]:
    """Use unchanged RotationBand spinors, link current and SU(2) rotations."""
    p = np.array([0.23, -0.17, 0.31]) / band.a
    pout = np.array([-0.11, 0.28, 0.19]) / band.a
    axis, angle = np.array([1., -2., 3.]), 0.47
    j = band.generators
    commutator = max(np.linalg.norm(j[i] @ j[k] - j[k] @ j[i]
                                    - 1j * j[l])
                     for i, k, l in [(0, 1, 2), (1, 2, 0), (2, 0, 1)])
    rotation = band.rotation(axis, angle)
    spatial = spatial_rotation(axis, angle)
    vector_residual = max(np.linalg.norm(rotation @ band.alpha[i] @ rotation.conj().T
                                        - np.einsum('j,jab->ab', spatial[:, i], band.alpha))
                          for i in range(3))
    sine_transfer = 2 * np.sin(band.a * (pout - p) / 2) / band.a
    ward = band.h(pout) - band.h(p) - np.einsum('i,ijk->jk', sine_transfer, band.current(pout, p))
    return dict(coupling_to_existing_h_error=max(float(np.linalg.norm(
                    h - band.h(p0))) for p0, h in zip(momenta, h_from_couplings(band, momenta, SELECTED))),
                spinor_eigenstate_error=max(float(np.linalg.norm(
                    band.h(p) @ band.spinor(p, label) - band.energy(p) * band.spinor(p, label)))
                    for label in range(2)),
                su2_commutator_error=float(commutator),
                casimir_error=float(np.linalg.norm(np.sum(j @ j, axis=0) - 0.75 * I4)),
                full_turn_minus_identity_error=float(np.linalg.norm(band.rotation(axis, 2 * np.pi) + I4)),
                vector_rotation_error=float(vector_residual),
                finite_link_current_ward_error=float(np.linalg.norm(ward)),
                rest_generator_spectrum=np.linalg.eigvalsh(j[2]).tolist())


def run(args):
    band = RotationBand(a=args.a, m=args.m, r=args.r, v=args.v, improved=args.improved)
    basis = matrix_unit_basis()
    rng = np.random.default_rng(args.seed)
    n = args.lattice_size
    momenta = np.array(list(itertools.product(range(n), repeat=3))) * (2 * np.pi / (n * band.a))
    # Independent noncyclic input guards against silently treating rank-poor
    # cubic grids (notably L=2) as full identifiability of ten quadratic products.
    rich = rng.uniform(-np.pi / band.a, np.pi / band.a, (48, 3))
    diagonal = np.column_stack([np.linspace(-2, 2, 13) / band.a] * 3)
    selected = evaluate(SELECTED, band, momenta)
    alternatives = [evaluate(random_frame(rng, basis), band, momenta)
                    for _ in range(args.alternatives)]
    rich_selected = evaluate(SELECTED, band, rich)
    curvature = curvature_probe(band, momenta, basis)
    consumer = selected_consumer_probe(band, momenta)
    rank = rank_report(coefficients(band, momenta))
    rich_rank = rank_report(coefficients(band, rich))
    witness = minimal_rank_witness(band)
    checks = dict(matrix_unit_basis_orthonormal=np.linalg.norm(gram(basis) - np.eye(15)) < 1e-12,
                  selected_attains_sea_bound=abs(selected['excess_energy']) < 1e-9,
                  alternatives_respect_budget=all(row['gram_error'] < 1e-12 and row['trace_error'] < 1e-12
                                                  and row['hermiticity_error'] < 1e-12 for row in alternatives),
                  actual_occupied_energy_obeys_bound=all(row['minimum_pointwise_excess'] > -1e-10
                                                        for row in alternatives + [selected]),
                  alternatives_have_nonzero_excess=all(row['excess_energy'] > 1e-6 for row in alternatives),
                  rich_inputs_identify_full_algebra=rich_rank['quadratic_rank'] == 10,
                  analytic_ten_point_rank_witness=witness['quadratic_rank'] == 10
                      and witness['determinant_relative_error'] < 1e-10,
                  rich_selected_attains_bound=abs(rich_selected['excess_energy']) < 1e-9,
                  coupling_and_budget_identities=all(row['local_coupling_error'] < 1e-10
                                                      and row['fixed_budget_error'] < 1e-9
                                                      and row['exact_pair_splitting_identity_error'] < 1e-9
                                                      for row in alternatives + [selected]),
                  perturbation_has_positive_curvature=curvature['analytic_second_derivative'] > 0
                      and all(row['excess_energy'] > 0 for row in curvature['path'])
                      and curvature['path'][-1]['relative_curvature_error'] < 1e-4,
                  existing_spinor_rotation_current_consumer=all(value < 1e-10 for key, value in consumer.items()
                                                                if key.endswith('error')))
    report = dict(scope='Conditional uniform-orbital mean-field energy theorem; no anticommutator penalty, no full-QED origin claim.',
                  parameters=vars(args), basis='15 Hermitian traceless matrix-unit directions',
                  cubic_momentum_identifiability=rank, noncyclic_momentum_identifiability=rich_rank,
                  adverse_diagonal_momentum_identifiability=rank_report(coefficients(band, diagonal)),
                  exact_ten_momentum_rank_witness=witness,
                  selected=selected, alternatives=alternatives, noncyclic_selected=rich_selected,
                  gram_preserving_curvature=curvature, existing_rotation_band_consumer=consumer,
                  assertions={key: bool(value) for key, value in checks.items()})
    print(json.dumps(report, indent=2))
    return 0 if all(checks.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=float, default=0.5)
    parser.add_argument('--m', type=float, default=1.)
    parser.add_argument('--r', type=float, default=1.)
    parser.add_argument('--v', type=float, default=1.)
    parser.add_argument('--improved', action='store_true')
    parser.add_argument('--lattice-size', type=int, default=3)
    parser.add_argument('--alternatives', type=int, default=8)
    parser.add_argument('--seed', type=int, default=22291)
    args = parser.parse_args()
    if (not all(np.isfinite([args.a, args.m, args.r, args.v]))
            or min(args.a, args.m, args.v) <= 0 or args.r < 0
            or args.lattice_size < 2 or args.alternatives < 1):
        parser.error('Require finite a,m,v>0, r>=0, lattice-size>=2 and alternatives>=1.')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())

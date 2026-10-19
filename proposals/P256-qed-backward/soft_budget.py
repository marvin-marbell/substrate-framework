#!/usr/bin/env python3
"""Unconstrained orbital matrices: elastic Gram energy plus actual occupied sea.

A soft quartic bosonic potential selects the Gram matrix; the sea selects the
Clifford algebra. No HS orthogonality or anticommutator penalty is imposed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from scipy.optimize import minimize
from rotation_band import RotationBand
from sea_selection import SELECTED, coefficients, gram, h_from_couplings, matrix_unit_basis, rank_report


def elastic(gram_matrix: np.ndarray, stiffness: float, preferred: float) -> float:
    return .5 * stiffness * float(np.sum((gram_matrix - preferred * np.eye(4))**2))


def select_gram(x: np.ndarray, stiffness: float, preferred: float,
                weights: np.ndarray | None = None) -> dict:
    """Unique convex Gram minimum for cubic inputs and normalized BZ weights."""
    if stiffness <= 0 or preferred < 0:
        raise ValueError('Positive elastic stiffness and nonnegative preferred scale are required.')
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if (weights.shape != (len(x),) or not np.all(np.isfinite(weights))
                or not np.all(weights > 0) or abs(float(weights.sum()) - 1.) > 1e-12):
            raise ValueError('BZ weights must be positive and normalized to one.')
    mass_squared = x[:, 0]**2
    kinetic_squared = np.sum(x[:, 1:]**2, axis=1)

    def average(values: np.ndarray) -> float:
        return float(values.mean() if weights is None else np.dot(weights, values))

    def functional(g: np.ndarray) -> tuple[float, np.ndarray]:
        radii = np.sqrt(g[0] * mass_squared + g[1] * kinetic_squared)
        value = -2 * average(radii) + .5 * stiffness * (
            (g[0] - preferred)**2 + 3 * (g[1] - preferred)**2)
        derivative = np.array([stiffness * (g[0] - preferred) - average(mass_squared / radii),
                               3 * stiffness * (g[1] - preferred) - average(kinetic_squared / radii)])
        return value, derivative

    def logarithmic_functional(log_excess: np.ndarray) -> tuple[float, np.ndarray]:
        excess = np.exp(log_excess)
        value, derivative = functional(preferred + excess)
        return value, excess * derivative

    # The minimum is strictly above the preferred scale on the rich cubic
    # input. Logarithmic coordinates keep line searches away from the
    # nondifferentiable all-zero Gram corner when the bare preference is zero.
    minimum = minimize(logarithmic_functional, np.array([0., np.log(.1)]), jac=True,
                       method='L-BFGS-B',
                       options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 500})
    estimate = preferred + np.exp(minimum.x)
    value, derivative = functional(estimate)
    # L-BFGS supplies an interior estimate, not the stationarity certificate.
    # Its relative-energy stopping rule may precede the Newton refinement.
    if not np.all(np.isfinite(estimate)) or np.any(estimate <= preferred):
        raise RuntimeError(f'Convex initial estimate left the domain: {minimum.message}')
    g_values = estimate.copy()
    # Refine the actual stationarity equations. O(q^2) orbital curvatures must
    # not be compared with a larger optimizer stopping-error floor.
    for _ in range(3):
        if np.max(np.abs(derivative)) < 1e-13:
            break
        radii = np.sqrt(g_values[0] * mass_squared + g_values[1] * kinetic_squared)
        inverse_cube = 1 / radii**3
        cross = .5 * average(mass_squared * kinetic_squared * inverse_cube)
        hessian = np.array([
            [stiffness + .5 * average(mass_squared**2 * inverse_cube), cross],
            [cross, 3 * stiffness + .5 * average(kinetic_squared**2 * inverse_cube)]])
        g_values -= np.linalg.solve(hessian, derivative)
        if np.any(g_values <= preferred):
            raise RuntimeError('Convex stationary refinement left the proved interior domain.')
        value, derivative = functional(g_values)
    if np.max(np.abs(derivative)) > 1e-11:
        raise RuntimeError(f'Convex stationarity refinement failed: {derivative}')
    g = np.diag([g_values[0], *([g_values[1]] * 3)])
    return dict(G=g, value=float(value), gradient_residual=float(np.max(np.abs(derivative))))


def run_case(stiffness: float, preferred: float = 1.) -> dict:
    band = RotationBand(a=1., m=.2, e=1., improved=True)
    momenta = np.array(list(itertools.product(range(3), repeat=3))) * (2 * np.pi / 3)
    x = coefficients(band, momenta)
    solution = select_gram(x, stiffness, preferred)
    g = solution['G']
    frame = np.sqrt(np.diag(g))[:, None, None] * SELECTED
    energies = np.linalg.eigvalsh(h_from_couplings(band, momenta, frame))
    actual = float(energies[:, :2].sum(axis=1).mean()) + elastic(gram(frame), stiffness, preferred)
    reference = solution['value']
    effective = RotationBand(a=band.a, m=band.m * np.sqrt(g[0, 0]),
                             r=band.r * np.sqrt(g[0, 0]), v=band.v * np.sqrt(g[1, 1]),
                             e=1., improved=True)
    mapping = max(float(np.linalg.norm(matrix - effective.h_from_couplings(p)))
                  for matrix, p in zip(h_from_couplings(band, momenta, frame), momenta))
    rng = np.random.default_rng(222)
    basis = matrix_unit_basis()
    alternatives = []
    for _ in range(8):
        # No normalization, orthogonalization or Gram restoration: unrestricted
        # real adjoint orbital fields compete in the original physical energy.
        fields = rng.normal(size=(4, 15)) / np.sqrt(15)
        alternative = np.einsum('ma,aij->mij', fields, basis)
        alternative_g = gram(alternative)
        alternative_energies = np.linalg.eigvalsh(h_from_couplings(band, momenta, alternative))
        # The actual fixed TOTAL half-filled ground, allowing particle transfer
        # between momentum blocks; no stronger per-block filling constraint.
        occupied = np.sort(alternative_energies.ravel())[:2 * len(momenta)]
        energy = float(occupied.sum()) / len(momenta) + elastic(alternative_g, stiffness, preferred)
        lower = -2 * float(np.sqrt(np.einsum('pi,ij,pj->p', x, alternative_g, x)).mean())
        lower += elastic(alternative_g, stiffness, preferred)
        alternatives.append(dict(actual_energy=energy, gram_lower_bound=lower,
                                 excess_above_selected=energy - reference,
                                 non_clifford_excess=energy - lower,
                                 gram_defect=float(np.linalg.norm(alternative_g - preferred * np.eye(4)))))
    matrix_identity = max(float(np.linalg.norm(frame[i] @ frame[j] + frame[j] @ frame[i]
                                               - 2 * g[i, j] * np.eye(4)))
                          for i in range(4) for j in range(4))
    direction = np.array([.2, -.3, .4])
    incoming = direction + np.array([.1, .05, -.02])
    outgoing = direction
    khat = 2 * np.sin((outgoing - incoming) / 2)
    gauge = effective.vertex(outgoing, incoming, np.r_[0., khat])
    ward = float(np.linalg.norm(gauge - (effective.h(outgoing) - effective.h(incoming))))
    checks = dict(soft_gram_stationary=solution['gradient_residual'] < 1e-7,
                  selected_saturates_global_energy=abs(actual - reference) < 1e-11,
                  unconstrained_alternatives_above_global_bound=all(
                      row['actual_energy'] >= row['gram_lower_bound'] - 1e-11
                      and row['gram_lower_bound'] >= reference - 1e-11 for row in alternatives),
                  alternatives_have_positive_excess=all(row['excess_above_selected'] > 1e-3 for row in alternatives),
                  derived_metric_clifford=matrix_identity < 1e-11,
                  exact_same_link_effective_map=mapping < 1e-11,
                  same_current_ward=ward < 1e-11,
                  rich_inputs=rank_report(x)['quadratic_rank'] == 10)
    return dict(stiffness=stiffness, preferred=preferred, selected_gram=g.tolist(),
                energy=actual, global_lower_bound=reference,
                stationary_residual=solution['gradient_residual'],
                effective_mass=float(effective.m), effective_velocity=float(effective.v),
                effective_wilson_coefficient=float(effective.r), mapping_residual=mapping,
                current_ward_residual=ward, unrestricted_alternatives=alternatives,
                checks=checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preferred', type=float, default=1.)
    args = parser.parse_args()
    rows = [run_case(stiffness, args.preferred) for stiffness in (.1, 1., 10.)]
    result = dict(status='soft_elastic_law_selects_metric_clifford_and_cubic_isotropy',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=rows, all_checks_pass=all(all(row['checks'].values()) for row in rows))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Thermodynamic electric susceptibility and its derived infrared logarithm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from numpy.polynomial.legendre import leggauss
from rotation_band import RotationBand, HOPPING, MIXING
from vacuum_limit import momentum_response


def electric_density(p: np.ndarray, ma: float, improved: bool = True) -> np.ndarray:
    """Trace of actual spectral projectors; no eigenvector-basis convention.

    Tr(P+ J_y P- J_y)=2[|u|^2-(d.u)^2/E^2], d=(M,sin p), u=d_y d.
    Dividing by 4E^3 gives the occupied-to-empty frequency coefficient.
    """
    sine, delta = np.sin(p), 1 - np.cos(p)
    mass = ma + np.sum(delta**2 if improved else delta, axis=-1)
    mass_derivative = (2 * delta[..., 1] if improved else 1.) * sine[..., 1]
    velocity = np.cos(p[..., 1])
    energy_squared = mass**2 + np.sum(sine**2, axis=-1)
    tangent_squared = mass_derivative**2 + velocity**2
    projection = mass * mass_derivative + sine[..., 1] * velocity
    return (tangent_squared - projection**2 / energy_squared) / (2 * energy_squared**1.5)


def thermodynamic_electric(ma: float, order: int, improved: bool = True) -> float:
    # Resolve the small-mass region explicitly; do not confuse a missing zero
    # mode on a coarse uniform quadrature with a converged thermodynamic integral.
    edges = np.array([0., .0001, .0004, .0016, .0064, .0256, .1024, .4096, 1.6384, np.pi])
    nodes, weights = leggauss(order)
    widths = np.diff(edges)
    axis = (edges[:-1, None] + widths[:, None] * (nodes[None, :] + 1) / 2).ravel()
    axis_weights = (widths[:, None] * weights[None, :] / 2).ravel()
    total = 0.
    # One x slab at a time bounds memory independently of quadrature order.
    yz = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    yz_weights = np.outer(axis_weights, axis_weights).ravel()
    points = np.empty((len(yz), 3))
    points[:, 1:] = yz
    for x, weight in zip(axis, axis_weights):
        points[:, 0] = x
        total += weight * float(np.dot(yz_weights, electric_density(points, ma, improved)))
    # Eight even octants divided by (2pi)^3.
    return total / np.pi**3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    controls = []
    for improved in (False, True):
        band = RotationBand(a=1., m=.2, e=1., improved=improved)
        rng = np.random.default_rng(222)
        matrix_errors = []
        for point in rng.uniform(-np.pi, np.pi, (24, 3)):
            energies, states = np.linalg.eigh(band.h_from_couplings(point))
            delta = 1 - np.cos(point[1])
            mass_derivative = (2 * delta if improved else 1.) * np.sin(point[1])
            current = np.cos(point[1]) * HOPPING[1] + mass_derivative * MIXING
            weights = abs(states[:, 2:].conj().T @ current @ states[:, :2])**2
            gaps = energies[2:, None] - energies[None, :2]
            spectral = 2 * float(np.sum(weights / gaps**3))
            matrix_errors.append(abs(spectral - float(electric_density(point, .2, improved))))
        length = 12
        grid = np.moveaxis(np.indices((length,) * 3), 0, -1).reshape(-1, 3)
        analytic_finite = float(electric_density(grid * (2 * np.pi / length), .2, improved).mean())
        reference = momentum_response(length, band)['DeltaChi_electric']
        controls.append(dict(improved=improved, matrix_trace_error=max(matrix_errors),
                             finite_torus_error=abs(analytic_finite - reference),
                             all_pass=max(matrix_errors) < 1e-12 and abs(analytic_finite - reference) < 1e-12))
    coefficient = 1 / (6 * np.pi**2)
    rows = []
    for improved in (False, True):
        previous = None
        for exponent in range(11):
            ma = .4 / 2**exponent
            coarse = thermodynamic_electric(ma, 8, improved)
            fine = thermodynamic_electric(ma, 12, improved)
            slope = None if previous is None else (fine - previous) / np.log(2)
            rows.append(dict(improved=improved, ma=ma, DeltaChi_electric_infinite_volume=fine,
                             quadrature_difference=abs(fine - coarse),
                             logarithmic_mass_slope=slope,
                             slope_over_derived_coefficient=None if slope is None else slope / coefficient))
            previous = fine
    result = dict(status='derived_and_executed_universal_logarithmic_electric_response',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  derived_log_coefficient=float(coefficient), controls=controls, rows=rows,
                  all_controls_pass=all(row['all_pass'] for row in controls))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_controls_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

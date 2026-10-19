#!/usr/bin/env python3
"""Analytic infrared coefficients and the actual continuous photon poles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from angular_photon_spectrum import ContinuousSea


def run(orders, quadrature, mass):
    rows = []
    for order in orders:
        sea = ContinuousSea(order, ma=mass, quadrature=quadrature)
        ir = sea.infrared()
        speed = float(np.sqrt((sea.chi_b + ir['magnetic']) / (sea.chi_e + ir['electric'])))
        source = float(1 / np.sqrt(sea.chi_e + ir['electric']))
        sequence = []
        for fraction in (.08, .04, .02, .01):
            q = mass * fraction
            tensor = sea.spectrum(np.array([q, 0., 0.]))
            basis = tensor.basis
            magnetic = float(np.trace(basis.T @ tensor.static @ basis) / 2
                             / np.dot(tensor.khat, tensor.khat))
            electric = float(np.trace(basis.T @ tensor.spectral(0., 'derivative') @ basis) / 2)
            mode = tensor.modes()['branches'][0]
            pole_speed = float(mode['energy'] / q)
            pole_source = float(mode['source'])
            sequence.append(dict(q=q, q_over_mass=fraction,
                                 magnetic=magnetic, electric=electric,
                                 magnetic_error=magnetic - ir['magnetic'],
                                 electric_error=electric - ir['electric'],
                                 pole_speed=pole_speed,
                                 pole_speed_error=pole_speed - speed,
                                 pole_source=pole_source,
                                 pole_source_error=pole_source - source))
        coarse, fine = sequence[0], sequence[-1]
        checks = dict(
            cubic_electric_and_magnetic=ir['electric_cubic_error'] < 1e-12
                and ir['magnetic_cubic_error'] < 1e-12,
            magnetic_limit=abs(fine['magnetic_error']) < abs(coarse['magnetic_error']) / 20,
            electric_limit=abs(fine['electric_error']) < abs(coarse['electric_error']) / 20,
            actual_pole_speed_limit=abs(fine['pole_speed_error']) < abs(coarse['pole_speed_error']) / 20,
            actual_pole_source_limit=abs(fine['pole_source_error']) < abs(coarse['pole_source_error']) / 20)
        rows.append(dict(order=order, quadrature=quadrature, points=len(sea.p),
                         ma=mass, infrared=ir, infrared_speed=speed, infrared_source=source,
                         finite_transfer_sequence=sequence, checks=checks))
    return dict(status='analytic_same_sea_photon_infrared_and_actual_poles',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                spectrum_source_sha256=hashlib.sha256(Path(__file__).with_name(
                    'angular_photon_spectrum.py').read_bytes()).hexdigest(),
                scope='Analytic transfer-zero response; actual pole and residue approach it. BZ quadrature error is separate.',
                units='hbar=a=1; q,ma energies in lattice units; electric/magnetic coefficients dimensionless',
                rows=rows, all_checks_pass=all(all(row['checks'].values()) for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', nargs='+', type=int, default=[24, 32, 48])
    parser.add_argument('--quadrature', choices=['midpoint', 'split_gauss'], default='midpoint')
    parser.add_argument('--mass', type=float, default=.4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.orders, args.quadrature, args.mass)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

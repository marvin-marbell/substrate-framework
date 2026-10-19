#!/usr/bin/env python3
"""Derive compact-pole and source corrections, then observe the actual rotor."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import sympy as sp

from quantum_compact_photon import neutral_quartic_report, spectral_measure


def perturbation_coefficients() -> dict:
    # These intermediate states cover every path used by X^6 and second-order
    # X^4 for n=0,1,3; this is exact oscillator algebra, not a rotor cutoff.
    size = 10
    lowering = sp.zeros(size)
    for n in range(1, size):
        lowering[n - 1, n] = sp.sqrt(n)
    X = lowering + lowering.T
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    X6 = X3 * X3

    def first(n: int):
        return -X4[n, n] / 24

    def second(n: int):
        return sp.simplify(X6[n, n] / 360 + sum(
            X4[l, n] ** 2 / (576 * (n - l)) for l in range(size) if l != n))

    def state(n: int, l: int):
        return -X4[l, n] / (48 * (n - l)) if l != n else sp.Integer(0)

    def B_correction(m: int):
        return sp.simplify(sum(state(m, l) * X[l, 0] + X[m, l] * state(0, l)
                               for l in range(size)) - X3[m, 0] / 6)

    return dict(gap_UE=sp.simplify(first(1) - first(0)),
                gap_UE2_over_omega=sp.simplify(second(1) - second(0)),
                weight1_v2=sp.simplify(2 * B_correction(1)),
                weight3_v3=sp.simplify(B_correction(3) ** 2),
                level3_gap_UE=sp.simplify(first(3) - first(0)))


def rotor_response(cutoff: int, electric: float, omega: float = 2.) -> dict:
    magnetic = omega ** 2 / (4 * electric)
    v = 2 * electric / omega
    fields = np.arange(-cutoff, cutoff + 1)
    # W is the actual neutral Gauss loop shift. Compression is noncyclic.
    W = np.diag(np.ones(2 * cutoff), -1)
    H = np.diag(2 * electric * fields ** 2) - magnetic * (W + W.T) / 2
    B = (W - W.T) / (2j)
    energies, vectors = np.linalg.eigh(H)
    _, report = spectral_measure(H, B, energies, vectors)
    weights = np.abs(vectors.conj().T @ (B @ vectors[:, 0])) ** 2
    shell = float(abs(vectors[0, 0]) ** 2 + abs(vectors[-1, 0]) ** 2)
    return dict(cutoff=cutoff, UE=electric, K=magnetic, omega=omega, v=v,
                gap=float(energies[1] - energies[0]),
                level3_gap=float(energies[3] - energies[0]),
                weight1=float(weights[1]), weight3=float(weights[3]),
                ground_energy=float(energies[0]), ground_shell_probability=shell,
                equal_time_variance=report['connected_equal_time'],
                static_susceptibility=report['static_susceptibility'],
                equal_time_sum_rule_residual=report['equal_time_sum_rule_residual'],
                energy_sum_rule_residual=report['energy_sum_rule_residual'])


def run(cutoffs: list[int], electric_values: list[float]) -> dict:
    if len(cutoffs) < 2 or sorted(set(cutoffs)) != cutoffs or min(cutoffs) < 2:
        raise ValueError('At least two increasing noncyclic cutoffs >=2 are required')
    if not electric_values or any(e <= 0 for e in electric_values):
        raise ValueError('Positive electric couplings are required')
    exact = perturbation_coefficients()
    numeric = {name: float(value) for name, value in exact.items()}
    # Compare with the existing actual raw-Gauss Pauli model, not a second
    # Gaussian oscillator. Its occupation-dependent constant cancels in gaps.
    embedded = neutral_quartic_report(2, .2)
    direct = rotor_response(2, .2)
    embedding_residual = max(abs(direct['gap'] - embedded['gap']),
                             abs(direct['static_susceptibility']
                                 - embedded['B_response']['static_susceptibility']))
    sequences = []
    for electric in electric_values:
        rows = [rotor_response(c, electric) for c in cutoffs]
        row = rows[-1]
        omega, v = row['omega'], row['v']
        first_gap = omega + numeric['gap_UE'] * electric
        second_gap = first_gap + numeric['gap_UE2_over_omega'] * electric ** 2 / omega
        row.update(first_order_gap=first_gap, second_order_gap=second_gap,
                   first_order_gap_error=row['gap'] - first_gap,
                   second_order_gap_error=row['gap'] - second_gap,
                   measured_second_gap_coefficient=(row['gap'] - first_gap) * omega / electric ** 2,
                   measured_weight1_coefficient=(row['weight1'] - v) / v ** 2,
                   measured_weight3_coefficient=row['weight3'] / v ** 3,
                   harmonic_sine_weight1=math.exp(-v) * v,
                   corrected_weight1=v + numeric['weight1_v2'] * v ** 2)
        sequences.append(dict(UE=electric, cutoffs=rows,
                              last_gap_change=row['gap'] - rows[-2]['gap'],
                              last_weight1_change=row['weight1'] - rows[-2]['weight1']))
    smallest = min(sequences, key=lambda sequence: sequence['UE'])['cutoffs'][-1]
    checks = dict(
        same_actual_neutral_Gauss_response=embedding_residual < 1e-11,
        resolved_final_pole_cutoff=all(abs(s['last_gap_change']) < 1e-8 for s in sequences),
        derived_second_order_improves_pole=all(
            abs(s['cutoffs'][-1]['second_order_gap_error'])
            < abs(s['cutoffs'][-1]['first_order_gap_error']) / 2 for s in sequences),
        derived_second_pole_coefficient=abs(smallest['measured_second_gap_coefficient']
                                           - numeric['gap_UE2_over_omega']) < .003,
        derived_source_coefficient=abs(smallest['measured_weight1_coefficient']
                                       - numeric['weight1_v2']) < .01,
        derived_third_pole_weight=abs(smallest['measured_weight3_coefficient']
                                      - numeric['weight3_v3']) < .01,
        harmonic_source_counterpart_rejected=all(
            abs(s['cutoffs'][-1]['weight1'] - s['cutoffs'][-1]['corrected_weight1'])
            < abs(s['cutoffs'][-1]['weight1'] - s['cutoffs'][-1]['harmonic_sine_weight1'])
            for s in sequences))
    return dict(status='compact_rotor_pole_and_source_asymptotics',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                domain='Exact neutral invariant Gauss block; small-angle asymptotics of its nonlinear cosine Hamiltonian',
                units='hbar=1; UE,K,omega energies; phi,n,B=sin(phi) dimensionless',
                exact_coefficients={name: str(value) for name, value in exact.items()},
                neutral_embedding_residual=embedding_residual, sequences=sequences,
                checks=checks, all_checks_pass=all(checks.values()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs', nargs='+', type=int, default=[20, 32, 48])
    parser.add_argument('--electric', nargs='+', type=float, default=[.2, .1, .05, .025, .0125])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.cutoffs, args.electric)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

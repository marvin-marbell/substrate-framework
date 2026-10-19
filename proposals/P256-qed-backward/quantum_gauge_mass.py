#!/usr/bin/env python3
"""Infinite-Gauss-rotor geometric mass theorem and scalar Galerkin consumers.

The analytic bound is an infinite-domain theorem. Fourier, radial-box and
physical electric-cutoff computations are explicitly separate diagnostics.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import eigh

from compact_charge_gap import GapParameters
from quantum_radial_spinor import RadialBasis, RadialSpinor
import quantum_spinor_plaquette as spinor
from wilson_mass_geometry import (plaquette_one_body, sea_projector,
                                  gauss_diagonals, parallel_sea_frame)


@dataclass(frozen=True)
class GeometryParameters:
    m: float = .8
    r: float = .04
    v: float = .08
    electric: float = .0072
    magnetic: float = .08 / .3**2

    def validate(self):
        if not all(math.isfinite(x) for x in
                   (self.m, self.r, self.v, self.electric, self.magnetic)):
            raise ValueError('Require finite parameters')
        if self.m <= 0 or min(self.r, self.v, self.magnetic) < 0 or self.electric <= 0:
            raise ValueError('Require m,UE>0 and r,v,K>=0; no hopping/mass restriction')

    @property
    def edge_norm(self):
        return (self.r + self.v) / 2

    @property
    def G_bound(self):
        # Rank-eight projector derivative plus neutral Gauss offsets (0,2,4,2).
        return 16 * self.edge_norm**2 / self.m**2 + 48

    @property
    def infinite_gap_bound(self):
        return self.m + self.electric / 2 * (1 - self.G_bound)


def geometry_at(phi, parameters):
    h, dh = plaquette_one_body(phi, parameters.m, parameters.r, parameters.v)
    P, dP = sea_projector(h, dh)
    occupied_energy = float(np.linalg.eigvalsh(h)[:8].sum())
    g = float(np.trace(P @ dP @ dP).real)
    identity = np.eye(16)
    edges = []
    for D, background in gauss_diagonals():
        mean = float((np.trace(P @ D) + background).real)
        variance = float(np.trace(P @ D @ (identity - P) @ D).real)
        z = complex(np.trace(P @ dP @ D))
        value = g + variance - 2 * z.imag
        edges.append({'g': g, 'mean_a': mean, 'variance_a': variance,
                      'Im_z': float(z.imag), 'G_e': value})
    if min(e['G_e'] for e in edges) < -2e-10 or max(abs(e['mean_a']) for e in edges) > 2e-10:
        raise AssertionError('Chiral density or nonnegative covariant metric failed')
    return occupied_energy, float(sum(e['G_e'] for e in edges)), edges


def sample_geometry(parameters, points):
    phases = 2 * np.pi * np.arange(points) / points
    energies, metric = [], []
    for phi in phases:
        energy, value, _ = geometry_at(float(phi), parameters)
        energies.append(energy); metric.append(value)
    metric = np.asarray(metric)
    if np.max(metric) > parameters.G_bound + 2e-9:
        raise AssertionError('Sample violates the independent analytic upper bound')
    return phases, np.asarray(energies), metric


def multiplication(samples, cutoff):
    """Trapezoidal Fourier coefficients; never represented as exact integration."""
    modes = np.arange(-cutoff, cutoff + 1)
    phases = 2 * np.pi * np.arange(len(samples)) / len(samples)
    differences = modes[:, None] - modes[None, :]
    coefficients = {q: np.mean(samples * np.exp(-1j * q * phases))
                    for q in range(-2 * cutoff, 2 * cutoff + 1)}
    result = np.asarray([[coefficients[int(q)] for q in row] for row in differences])
    return (result + result.conj().T) / 2


def cosine_matrix(cutoff):
    count = 2 * cutoff + 1
    return (np.diag(np.ones(count - 1), 1) + np.diag(np.ones(count - 1), -1)) / 2


def rotor_ground(parameters, cutoff, phases, sea, metric):
    modes = np.arange(-cutoff, cutoff + 1)
    matrix = (np.diag(2 * parameters.electric * modes**2)
              + multiplication(sea, cutoff) - parameters.magnetic * cosine_matrix(cutoff))
    values, vectors = eigh(matrix, subset_by_index=(0, 0))
    state = vectors[:, 0]
    state *= np.exp(-1j * np.angle(state[cutoff]))
    wave = np.exp(1j * np.outer(phases, modes)) @ state / np.sqrt(2 * np.pi)
    weight = 2 * np.pi * abs(wave)**2
    weighted_G = float(np.mean(weight * metric))
    energy = float(values[0])
    return {'fourier_cutoff': cutoff, 'dimension': len(modes), 'scalar_ground_Ritz': energy,
            'eigen_residual': float(np.linalg.norm(matrix @ state - energy * state)),
            'quadrature_wave_norm': float(np.mean(weight)),
            'sampled_real_wave_minimum': float(np.min(wave.real)),
            'sampled_wave_imaginary_maximum': float(np.max(abs(wave.imag))),
            'sampled_weighted_G': weighted_G,
            'infinite_trial_energy_estimate': energy + parameters.electric / 2 * weighted_G,
            'scope': 'Ritz/quadrature diagnostics of Hs; neither a certified scalar ground lower bound nor an infinite-domain gap estimate.'}


def joint_ground(parameters, basis, cutoff, phases, sea, metric, charge_mass):
    """Actual coupled two-dimensional scalar Hs+cR, radial-major ordering."""
    modes = np.arange(-cutoff, cutoff + 1)
    count = len(modes)
    phase_part = (np.diag(2 * parameters.electric * modes**2)
                  - parameters.magnetic * cosine_matrix(cutoff))
    potential = multiplication(sea, cutoff)
    R = basis.moment(1)
    matrix = (np.kron(basis.hamiltonian(), np.eye(count))
              + np.kron(np.eye(basis.modes), phase_part)
              + np.kron(R, potential + charge_mass * np.eye(count)))
    values, vectors = eigh(matrix, subset_by_index=(0, 0))
    state = vectors[:, 0].reshape(basis.modes, count)
    phi_wave = state @ np.exp(1j * np.outer(modes, phases)) / np.sqrt(2 * np.pi)
    density = 2 * np.pi * np.sum(abs(phi_wave)**2, axis=0)
    radial_density = state @ state.conj().T
    mean_R = float(np.trace(radial_density @ R).real)
    weighted_G = float(np.mean(density * metric))
    energy = float(values[0])
    if mean_R <= 0:
        raise AssertionError('Positive radial-coordinate expectation failed')
    return {'charge_mass': charge_mass, 'ground_Ritz': energy, 'mean_R': mean_R,
            'sampled_weighted_G': weighted_G,
            'quadrature_wave_norm': float(np.mean(density)),
            'radial_phase_purity': float(np.trace(radial_density @ radial_density).real),
            'eigen_residual': float(np.linalg.norm(matrix @ state.reshape(-1) - energy * state.reshape(-1)))}, state


def joint_consumer(parameters, modes, boxes, cutoffs, phases, sea, metric, inertia, quartic):
    records = []
    for box in boxes:
        for number in modes:
            basis = RadialBasis(number, box, inertia, quartic)
            for cutoff in cutoffs:
                neutral, _ = joint_ground(parameters, basis, cutoff, phases, sea, metric, 0.)
                charged, _ = joint_ground(parameters, basis, cutoff, phases, sea, metric, parameters.m)
                difference = charged['ground_Ritz'] - neutral['ground_Ritz']
                if difference <= 0:
                    raise AssertionError('Positive projected mR perturbation did not raise scalar ground')
                records.append({'radial_modes': number, 'box': box, 'fourier_cutoff': cutoff,
                                'dimension': number * (2 * cutoff + 1),
                                'neutral_scalar': neutral, 'charged_scalar': charged,
                                'Fjoint_Ritz_difference': difference,
                                'Ritz_difference_minus_analytic_geometry': difference
                                    + parameters.electric / 2 * (1 - parameters.G_bound),
                                'neutral_trial_energy_quadrature': neutral['ground_Ritz']
                                    + parameters.electric / 2 * neutral['sampled_weighted_G'],
                                'scope': 'Actual 2D scalar consumer, but difference of two Ritz upper bounds is NOT a certified infinite-rotor/half-line lower bound.'})
    return records


def wedge(frame, derivative, occupations):
    minors = frame[occupations]
    wave = np.linalg.det(minors)
    dwave = np.zeros(len(occupations), dtype=complex)
    for column in range(8):
        replaced = minors.copy()
        replaced[:, :, column] = derivative[occupations, column]
        dwave += np.linalg.det(replaced)
    return wave, dwave


def sea_wave(phi, parameters, occupations, code_phases, with_derivative=True):
    h, dh = plaquette_one_body(phi, parameters.m, parameters.r, parameters.v)
    frame, dframe = parallel_sea_frame(h, dh)
    if with_derivative:
        wave, derivative = wedge(frame, dframe, occupations)
        derivative *= code_phases
    else:
        wave = np.linalg.det(frame[occupations])
        derivative = None
    return wave * code_phases, derivative


def wave_controls(parameters, selected):
    """Direct determinants in actual Pauli-code coordinates, no CAR builder."""
    code = spinor.PauliCode()
    phases = code.intertwiner(8)
    occupations = np.asarray(list(itertools.combinations(range(16), 8)))
    masks = [sum(1 << int(i) for i in row) for row in occupations]
    code_phases = np.asarray([phases[mask] for mask in masks])
    connections = np.asarray([np.diag(D)[occupations].sum(axis=1) + background
                              for D, background in gauss_diagonals()])
    records = []
    epsilon = 2e-5
    for phi in selected:
        wave, derivative = sea_wave(phi, parameters, occupations, code_phases)
        assert derivative is not None
        forward, _ = sea_wave(phi + epsilon, parameters, occupations, code_phases, False)
        backward, _ = sea_wave(phi - epsilon, parameters, occupations, code_phases, False)
        finite_difference = (forward - backward) / (2 * epsilon)
        direct = np.asarray([np.linalg.norm(derivative + 1j * a * wave)**2 for a in connections])
        _, total, edges = geometry_at(phi, parameters)
        target = np.asarray([e['G_e'] for e in edges])
        wrong = np.asarray([np.linalg.norm(derivative - 1j * a * wave)**2 for a in connections])
        report = {'phi': phi, 'norm_residual': abs(float(np.vdot(wave, wave).real) - 1),
                  'parallel_connection': float(abs(np.vdot(wave, derivative))),
                  'derivative_norm': float(np.linalg.norm(derivative)),
                  'determinant_derivative_fd_residual': float(np.linalg.norm(derivative - finite_difference)),
                  'direct_G_edges': direct.tolist(), 'trace_G_edges': target.tolist(),
                  'metric_sign_residual': float(np.max(abs(direct - target))),
                  'wrong_connection_sign_residual': float(np.max(abs(wrong - target))),
                  'sampled_G': total}
        if max(report['norm_residual'], report['parallel_connection'], report['metric_sign_residual']) > 2e-8:
            raise AssertionError('Actual Pauli Slater determinant/connection identity failed')
        if report['determinant_derivative_fd_residual'] > 2e-7:
            raise AssertionError('Actual determinant derivative finite-difference control failed')
        records.append(report)
    first, _ = sea_wave(0., parameters, occupations, code_phases, False)
    last, _ = sea_wave(2 * np.pi, parameters, occupations, code_phases, False)
    periodic = float(np.linalg.norm(last - first))
    wrong_connection = max(r['wrong_connection_sign_residual'] for r in records)
    if periodic > 2e-8 or (parameters.v + parameters.r > 0 and
                           max(r['derivative_norm'] for r in records) < 1e-8):
        raise AssertionError('Periodic phase or nonconstant Slater positive control failed')
    # With real decoupled matter the sign can coincide: distinguish observable
    # sensitivity from the independent direct plus-sign identity, not force it.
    odd_mask = (1 << 9) - 1
    odd_state = code.state(odd_mask)
    marker = all(spinor.MARKER in code.words.occupation(q) for q, _ in odd_state)
    counts = [sum(odd_mask >> (4 * c + j) & 1 for j in range(4)) for c in range(4)]
    charges = [counts[c] - 2 - (1 if c == 0 else 0) for c in range(4)]
    fields = (0, charges[1], charges[1] + charges[2], -charges[0])
    gauss = tuple(fields[c] - fields[(c - 1) % 4] + (1 if c == 0 else 0)
                  - (counts[c] - 2) for c in range(4))
    if not marker or gauss != (0, 0, 0, 0):
        raise AssertionError('Actual odd Pauli marker/Gauss positive control failed')
    return {'selected_points': records, 'code_intertwiner_residual': code.law_residual,
            'periodic_wave_residual': periodic,
            'wrong_connection_sign_detected': wrong_connection > 1e-8,
            'wrong_connection_scope': 'If Im z vanishes, this model cannot discriminate the two signs; direct plus-sign identity is still checked.',
            'fractional_loop_boundary_mismatch': float(abs(np.exp(1j * np.pi) - 1)),
            'integer_loop_boundary_mismatch': float(abs(np.exp(2j * np.pi) - 1)),
            'actual_odd_marker_present': marker, 'actual_odd_gauss': list(gauss),
            'wrong_boundary_gauss': [g + (1 if c == 0 else 0) for c, g in enumerate(gauss)],
            'charged_marker_gauss': [g - (1 if c == 0 else 0) for c, g in enumerate(gauss)]}


def physical_comparison(args, parameters, scalar):
    """Same existing coupled physical consumer, not the infinite theorem's domain."""
    if not args.physical_cutoffs:
        return []
    if parameters.magnetic <= 0:
        raise ValueError('Existing physical consumer requires K>0')
    delta = args.delta if args.delta is not None else 2 * parameters.magnetic + 1
    physical_parameters = GapParameters(electric=parameters.electric, magnetic=parameters.magnetic,
                                        delta=delta, m=parameters.m, r=parameters.r, v=parameters.v)
    physical_parameters.validate()
    code = spinor.PauliCode()
    result = []
    for cutoff in args.physical_cutoffs:
        for radial_modes in args.physical_radial_modes:
            basis = RadialBasis(radial_modes, max(args.boxes), args.inertia, args.quartic)
            energies, reports = {}, []
            for number in (8, 7, 9):
                physical = spinor.SpinorPlaquette(code, number, cutoff, magnetic=parameters.magnetic,
                    electric=parameters.electric, delta=delta, m=parameters.m, r=parameters.r, v=parameters.v)
                model = RadialSpinor(physical, basis, physical_parameters)
                eigenvalues, ground, low = spinor.solve_low(model, norm_bound=model.certified_norm_bound)
                energies[number] = float(eigenvalues[0])
                reports.append({'N': number, 'dimension': model.dimension,
                                'energy': energies[number], 'low_spectrum': low,
                                'mean_R': float(np.vdot(ground[:, 0].reshape(basis.modes, physical.dimension),
                                    basis.moment(1) @ ground[:, 0].reshape(basis.modes, physical.dimension)).real)})
            reference = [s for s in scalar if s['radial_modes'] == radial_modes and s['box'] == basis.box]
            result.append({'electric_cutoff': cutoff, 'radial_modes': radial_modes, 'box': basis.box,
                           'cases': reports, 'physical_mass_N7': energies[7] - energies[8],
                           'physical_mass_N9': energies[9] - energies[8],
                           'same_radial_scalar_diagnostics': reference,
                           'scope': 'Actual quantum_radial_spinor Hrad+R Hmatter+Hph on its finite per-edge electric and radial-sine domains. No infinite-rotor bound is asserted for this compression.'})
    return result


def run(args):
    parameters = GeometryParameters(args.m, args.r, args.v, args.electric, args.magnetic)
    parameters.validate()
    if args.phase_points <= 4 * max(args.fourier_cutoffs) or args.phase_points < 32:
        raise ValueError('Use phase_points>=32 and >4*max Fourier cutoff to resolve Toeplitz quadrature')
    if min(args.fourier_cutoffs) < 1 or min(args.radial_modes + args.physical_radial_modes) < 2:
        raise ValueError('Require positive Fourier cutoffs and radial modes>=2')
    phases, sea, metric = sample_geometry(parameters, args.phase_points)
    rotors = [rotor_ground(parameters, c, phases, sea, metric) for c in args.fourier_cutoffs]
    scalar_modes = sorted(set(args.radial_modes + (args.physical_radial_modes if args.physical_cutoffs else [])))
    joint = joint_consumer(parameters, scalar_modes, args.boxes, args.fourier_cutoffs,
                           phases, sea, metric, args.inertia, args.quartic)
    controls = wave_controls(parameters, args.control_phases)
    return {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'parameters': vars(args) | {'output': str(args.output) if args.output else None},
            'units': 'hbar=a=1; phi,R dimensionless; UE,K,m,r,v,quartic energy; inertia inverse-energy',
            'infinite_fixed_amplitude_theorem': {'bare_mass_lower_bound': parameters.m,
                'actual_edge_singular_norm': parameters.edge_norm,
                'certified_G_sup_upper_bound': parameters.G_bound,
                'charged_gap_lower_bound': parameters.infinite_gap_bound,
                'old_hopping_bound': parameters.m + 2 * parameters.r + parameters.electric / 2
                                    - 8 * max(parameters.r, parameters.v),
                'scope': 'Complete N7/8/9 occupation fibers tensor L2(T), ell in Z, plus code and actual boundary electric cost. No finite Fourier or per-edge cutoff claim.'},
            'sampled_geometry': {'points': args.phase_points, 'sampled_G_max': float(np.max(metric)),
                                'sampled_G_min': float(np.min(metric)),
                                'sea_energy_range': [float(np.min(sea)), float(np.max(sea))],
                                'scope': 'Sampled extrema are not certified suprema.'},
            'actual_pauli_slater_controls': controls, 'scalar_rotor_Galerkin': rotors,
            'scalar_joint_radial_phase_Galerkin': joint,
            'same_consumer_physical_comparison': physical_comparison(args, parameters, joint),
            'radial_theorem': 'Exact gap >= Fjoint(m)-Fjoint(0)+UE/2*(1-Gmax), positive for sufficiently weak UE for every fixed m,kappa,inertia>0 and finite K,r,v>=0.',
            'scope': 'Constructive quantum mass for the selected Clifford shape on one compact loop and one radial amplitude. Not a derivation of the full 60-coordinate field, microscopic quantization law, 3D phase or continuum QED.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m', type=float, default=.8)
    parser.add_argument('--r', type=float, default=.04)
    parser.add_argument('--v', type=float, default=.08)
    parser.add_argument('--electric', type=float, default=.0072)
    parser.add_argument('--magnetic', type=float, default=.08 / .3**2)
    parser.add_argument('--fourier-cutoffs', nargs='+', type=int, default=[6, 10])
    parser.add_argument('--phase-points', type=int, default=256)
    parser.add_argument('--radial-modes', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--boxes', nargs='+', type=float, default=[3., 4.])
    parser.add_argument('--inertia', type=float, default=1.)
    parser.add_argument('--quartic', type=float, default=.5)
    parser.add_argument('--control-phases', nargs='+', type=float, default=[.23, 1.17, 2.61])
    parser.add_argument('--physical-cutoffs', nargs='+', type=int, default=[])
    parser.add_argument('--physical-radial-modes', nargs='+', type=int, default=[2, 4])
    parser.add_argument('--delta', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    payload = json.dumps(run(args), sort_keys=True, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Full 50-dimensional orbital Hessian and finite-sea stability modulo PU(4).

See representation_stability.md for the proof, metric and explicit local tube.
The consumer uses RotationBand's actual links and primitive FilledSeaConsumer
state/current conventions; no QED spectrum supplies the energetic alternatives.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
from scipy.linalg import expm, null_space

from lattice_compton import WilsonLink
from pauli_compton import FilledSeaConsumer, block_terms, boundary_ladder, overlap
from rotation_band import I4, RotationBand
from sea_selection import (SELECTED, QUADRATIC_PAIRS, coefficients, evaluate,
                           gram, h_from_couplings, matrix_unit_basis, rank_report, reframe)


def coordinates(frame, basis):
    return np.einsum('aij,mji->am', basis, frame).real / 4


def matrices(coordinate_frame, basis):
    return np.einsum('am,aij->mij', coordinate_frame, basis)


def inverse_sqrt(matrix):
    values, vectors = np.linalg.eigh(matrix)
    if np.min(values) <= 0:
        raise ValueError('Polar chart requires a positive Gram matrix')
    return (vectors / np.sqrt(values)) @ vectors.conj().T


def retract(origin, tangent, scale):
    """Stiefel polar retraction; derivative at zero is exactly tangent."""
    candidate = origin + scale * tangent
    return candidate @ inverse_sqrt(candidate.T @ candidate)


def tangent_geometry(basis):
    origin = coordinates(SELECTED, basis)
    elementary = np.eye(60).reshape(60, 15, 4)
    constraints = np.array([[np.dot(origin[:, i], direction[:, j])
                             + np.dot(direction[:, i], origin[:, j])
                             for direction in elementary]
                            for i, j in QUADRATIC_PAIRS])
    tangents = null_space(constraints)
    # Genuine simultaneous orbital conjugations, not rotations of four labels.
    orbit_matrices = np.array([1j * (generator @ SELECTED - SELECTED @ generator)
                               for generator in basis])
    orbit = np.column_stack([coordinates(frame, basis).reshape(-1)
                             for frame in orbit_matrices])
    orbit_in_tangent = tangents.T @ orbit
    normal = null_space(orbit_in_tangent.T)
    return origin, constraints, tangents, orbit, normal


def hessian(band, momenta, tangent_columns, basis):
    x = coefficients(band, momenta)
    radius = np.linalg.norm(x, axis=1)
    directions = np.array([matrices(column.reshape(15, 4), basis)
                           for column in tangent_columns.T])
    h0 = h_from_couplings(band, momenta, SELECTED)
    variations = np.einsum('pm,tmij->tpij', x, directions)
    anticommutators = h0[None] @ variations + variations @ h0[None]
    weighted = anticommutators / np.sqrt(8 * radius[None, :, None, None]**3)
    features = weighted.reshape(len(directions), -1)
    result = (features.conj() @ features.T).real
    defects = []
    for variation in directions:
        defects.append(np.array([SELECTED[i] @ variation[j] + variation[i] @ SELECTED[j]
                                 + SELECTED[j] @ variation[i] + variation[j] @ SELECTED[i]
                                 for i, j in QUADRATIC_PAIRS]))
    defects = np.array(defects).reshape(len(directions), -1)
    real_defects = np.concatenate((np.real(defects), np.imag(defects)), axis=1).T
    return result, real_defects


def spectral_rank(matrix):
    values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * values[0]
    return int(np.sum(values > tolerance)), values, float(tolerance)


def polar_sector_frame(h, reference):
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    blocks = []
    for sector in (slice(0, 2), slice(2, 4)):
        projector = eigenvectors[:, sector] @ eigenvectors[:, sector].conj().T
        original = reference[:, sector]
        blocks.append(projector @ original @ inverse_sqrt(original.conj().T @ projector @ original))
    return np.column_stack(blocks), eigenvalues


def linear_ladder(consumer, offset, vector, state, creation=True):
    output = {}
    for orbital, coefficient in enumerate(vector):
        if coefficient == 0:
            continue
        for bits, amplitude in boundary_ladder(consumer.words, consumer.boundary,
                                               offset + orbital, state, creation).items():
            output[bits] = output.get(bits, 0j) + coefficient * amplitude
    return {bits: amplitude for bits, amplitude in output.items() if abs(amplitude) > 1e-15}


def physical_state_map_probe(band, perturbed_frame):
    """Transport one actual primitive occupied block and its charged excitations."""
    photons = WilsonLink(band.a, m=band.m, e=band.e, r=band.r, v=band.v, improved=band.improved)
    consumer = FilledSeaConsumer(band, photons, photons.kinematics(.2, .7))
    block = 1
    momentum = consumer.momenta[block]
    offset = 4 * block
    reference = consumer.frames[block]
    varied_h = h_from_couplings(band, np.array([momentum]), perturbed_frame)[0]
    transported, spectrum = polar_sector_frame(varied_h, reference)
    transform = reference.conj().T @ transported
    empty = consumer.vacuum
    for occupied in (0, 1):
        empty = boundary_ladder(consumer.words, consumer.boundary, offset + occupied, empty, False)
    vacuum = empty
    for occupied in (0, 1):
        vacuum = linear_ladder(consumer, offset, transform[:, occupied], vacuum)
    particles = [linear_ladder(consumer, offset, transform[:, 2 + label], vacuum)
                 for label in (0, 1)]
    base_projector = reference[:, :2] @ reference[:, :2].conj().T
    new_projector = transported[:, :2] @ transported[:, :2].conj().T
    delta_h = np.linalg.norm(varied_h - band.h(momentum), 2)
    radius = band.energy(momentum)
    delta = float(np.linalg.norm(coordinates(perturbed_frame - SELECTED, matrix_unit_basis())))
    spin = np.array([-0.25j * (perturbed_frame[j] @ perturbed_frame[k]
                              - perturbed_frame[k] @ perturbed_frame[j])
                     for j, k in [(2, 3), (3, 1), (1, 2)]])
    current = np.array([reframe(operator, perturbed_frame)
                        for operator in band.current(momentum, momentum)])
    lifted_errors = []
    spin_changes, current_changes = [], []
    for operators, base, changes in ((spin, band.generators, spin_changes),
                                     (current, band.current(momentum, momentum), current_changes)):
        for operator, old in zip(operators, base):
            terms = block_terms(reference.conj().T @ operator @ reference, offset, offset)
            sea_value = overlap(vacuum, consumer.words.apply(terms, vacuum))
            actual = np.array([[overlap(bra, consumer.words.apply(terms, ket))
                                for ket in particles] for bra in particles]) - sea_value * np.eye(2)
            predicted = transported[:, 2:].conj().T @ operator @ transported[:, 2:]
            lifted_errors.append(float(np.linalg.norm(actual - predicted)))
            changes.append(float(np.linalg.norm(predicted - reference[:, 2:].conj().T @ old @ reference[:, 2:])))
    return dict(actual_occupied_eigenvalues=spectrum[:2].tolist(),
                actual_sea_block_energy=float(spectrum[:2].sum()),
                primitive_vacuum_norm_error=float(abs(overlap(vacuum, vacuum) - 1)),
                primitive_particle_orthonormality_error=float(np.linalg.norm(
                    np.array([[overlap(bra, ket) for ket in particles] for bra in particles]) - np.eye(2))),
                physical_particle_number_increase=[float(sum(abs(amplitude)**2 *
                    (len(consumer.physical_occupation(bits)) - len(consumer.negative))
                    for bits, amplitude in state.items())) for state in particles],
                primitive_spin_and_current_lift_error=max(lifted_errors),
                occupied_projector_operator_change=float(np.linalg.norm(new_projector - base_projector, 2)),
                spectral_projector_gap_bound=float(delta_h / (2 * radius - delta_h)),
                orbital_frame_distance_upper_bound=delta,
                minimum_block_gap=float(spectrum[2] - spectrum[1]),
                aligned_spin_matrix_changes=spin_changes, aligned_current_matrix_changes=current_changes,
                state_map='Same EdgeWords and neutral-boundary creation law as FilledSeaConsumer; polar sector transport, no individually pinned degenerate eigenvectors.')


def run(args):
    band = RotationBand(a=args.a, m=args.m, r=args.r, v=args.v, improved=True)
    basis = matrix_unit_basis()
    momenta = np.array(list(itertools.product(range(3), repeat=3))) * (2 * np.pi / (3 * band.a))
    origin, constraints, tangents, orbit, normal = tangent_geometry(basis)
    full, linear_defects = hessian(band, momenta, tangents, basis)
    quotient = normal.T @ full @ normal
    positive, eigenvectors = np.linalg.eigh(quotient)
    all_values = np.linalg.eigvalsh(full)
    orbit_rank, orbit_singular, _ = spectral_rank(orbit)
    defect_rank, defect_singular, defect_tolerance = spectral_rank(linear_defects)
    constraint_rank, _, _ = spectral_rank(constraints)
    lam = float(positive[0])
    x = coefficients(band, momenta)
    radius = np.linalg.norm(x, axis=1)
    k_bound = float(np.sqrt(2 * radius.sum()))
    tube = float(min(.25, np.sqrt(lam) / (np.sqrt(lam) + 3 * k_bound)))
    local_constant = 12 / lam
    modes = tangents @ normal @ eigenvectors
    # All 35 quotient directions are probed using exact budget-preserving paths.
    curvature_rows = []
    for index, direction in enumerate(modes.T):
        step = .001
        plus = evaluate(matrices(retract(origin, direction.reshape(15, 4), step), basis), band, momenta)
        minus = evaluate(matrices(retract(origin, direction.reshape(15, 4), -step), basis), band, momenta)
        measured = (plus['excess_energy'] + minus['excess_energy']) / step**2
        curvature_rows.append(dict(mode=index, analytic_curvature=float(positive[index]),
                                   measured_curvature=float(measured),
                                   relative_error=float(abs(measured / positive[index] - 1)),
                                   maximum_gram_error=max(plus['gram_error'], minus['gram_error'])))
    rng = np.random.default_rng(args.seed)
    mixed = modes @ rng.normal(size=35)
    mixed /= np.linalg.norm(mixed)
    path_rows = []
    selected_paths = [(name, direction) for name, direction in
                      [('softest', modes[:, 0]), ('middle', modes[:, 17]),
                       ('hardest', modes[:, -1]), ('mixed_nonorbit', mixed)]]
    for name, direction in selected_paths:
        tangent = direction.reshape(15, 4)
        for step in (.1, .03, .01, tube / 4):
            frame = matrices(retract(origin, tangent, step), basis)
            receipt = evaluate(frame, band, momenta)
            chord = float(np.linalg.norm(coordinates(frame, basis) - origin))
            path_rows.append(dict(name=name, step=float(step), selected_chord=chord,
                                  inside_certified_tube=chord < tube,
                                  selected_chord_squared_over_excess=float(chord**2 / receipt['excess_energy']),
                                  **receipt))
    generator = np.einsum('a,aij->ij', rng.normal(size=15), basis)
    unitary = expm(.17j * generator)
    conjugated = unitary @ SELECTED @ unitary.conj().T
    orbit_receipt = evaluate(conjugated, band, momenta)
    # Actual rank-poor L=2 input has only six restoring directions, not 35.
    adverse_momenta = np.array(list(itertools.product(range(2), repeat=3))) * np.pi / band.a
    adverse_hessian, _ = hessian(band, adverse_momenta, tangents, basis)
    adverse_values = np.linalg.eigvalsh(adverse_hessian)
    scale = max(float(all_values[-1]), 1.)
    eigen_tolerance = 1e-9 * scale
    state_frame = matrices(retract(origin, mixed.reshape(15, 4), .01), basis)
    physical = physical_state_map_probe(band, state_frame)
    checks = dict(constraint_rank_ten=constraint_rank == 10,
                  full_tangent_dimension_fifty=tangents.shape == (60, 50),
                  tangent_constraints_satisfied=np.linalg.norm(constraints @ tangents) < 1e-12,
                  genuine_unitary_orbit_dimension_fifteen=orbit_rank == 15,
                  orbit_lies_in_tangent=np.linalg.norm(constraints @ orbit) < 1e-11,
                  orbit_exact_hessian_kernel=np.linalg.norm(full @ (tangents.T @ orbit)) < 1e-9 * scale,
                  linearized_clifford_rank_thirtyfive=defect_rank == 35,
                  full_hessian_fifteen_zero_thirtyfive_positive=int(np.sum(abs(all_values) < eigen_tolerance)) == 15
                      and int(np.sum(all_values > eigen_tolerance)) == 35,
                  quotient_positive_thirtyfive=len(positive) == 35 and lam > eigen_tolerance,
                  finite_design_rank_ten=rank_report(x)['quadratic_rank'] == 10,
                  every_normal_mode_restoring=all(row['measured_curvature'] > 0 and row['relative_error'] < .002
                                                 and row['maximum_gram_error'] < 1e-12 for row in curvature_rows),
                  nonorbit_paths_budget_and_energy=all(row['gram_error'] < 1e-12 and row['excess_energy'] > 0
                                                       and row['local_coupling_error'] < 1e-10 for row in path_rows),
                  small_normal_paths_obey_local_estimate=all(row['selected_chord_squared_over_excess'] < local_constant
                        for row in path_rows if row['inside_certified_tube']),
                  actual_orbit_energy_flat=abs(orbit_receipt['excess_energy']) < 1e-9
                      and orbit_receipt['maximum_clifford_defect'] < 1e-12,
                  adverse_mass_only_six_positive=int(np.sum(adverse_values > eigen_tolerance)) == 6,
                  physical_primitive_state_map=physical['primitive_vacuum_norm_error'] < 1e-11
                      and physical['primitive_particle_orthonormality_error'] < 1e-11
                      and physical['primitive_spin_and_current_lift_error'] < 1e-10
                      and all(abs(number - 1) < 1e-11 for number in physical['physical_particle_number_increase']),
                  projector_stability_gap_bound=physical['occupied_projector_operator_change']
                      <= physical['spectral_projector_gap_bound'] + 1e-12)
    report = dict(scope='Finite positive-weight uniform-orbital half-filled sea, improved L=3 band; no thermodynamic or full-QED claim.',
                  parameters=dict(a=args.a, m=args.m, r=args.r, v=args.v, improved=True, lattice_size=3, seed=args.seed),
                  metric='sum_mu tr(delta_A_mu^2)/4; coordinates are real 15x4 HS frames',
                  geometry=dict(ambient_dimension=60, independent_gram_constraints=constraint_rank,
                                tangent_dimension=tangents.shape[1], unitary_orbit_dimension=orbit_rank,
                                normal_dimension=normal.shape[1], orbit_singular_values=orbit_singular.tolist()),
                  momentum_design=rank_report(x),
                  hessian=dict(all_fifty_eigenvalues=all_values.tolist(), quotient_thirtyfive_eigenvalues=positive.tolist(),
                               linearized_clifford_rank=defect_rank, linearized_clifford_singular_values=defect_singular.tolist(),
                               linearized_clifford_rank_tolerance=defect_tolerance, eigenvalue_tolerance=eigen_tolerance),
                  local_coercivity=dict(exact_definition='lambda=min eig(H restricted to orbit-normal tangent)',
                                       lambda_numerical=lam, sum_radii=float(radius.sum()), K=k_bound,
                                       certified_distance_tube_numerical=tube,
                                       dist_squared_over_energy_constant_numerical=local_constant,
                                       bound='d <= min(1/4,sqrt(lambda)/(sqrt(lambda)+3K)) implies d^2 <= (12/lambda)(E-Emin)',
                                       qualification='Constants defined by the exact finite matrix; displayed floats are not interval-arithmetic certificates.'),
                  all_normal_mode_curvature=curvature_rows, nonorbit_budget_preserving_paths=path_rows,
                  exact_unitary_orbit=orbit_receipt,
                  adverse_L2=dict(design=rank_report(coefficients(band, adverse_momenta)),
                                  hessian_eigenvalues=adverse_values.tolist(),
                                  positive_directions=int(np.sum(adverse_values > eigen_tolerance)),
                                  flat_directions=int(np.sum(abs(adverse_values) < eigen_tolerance))),
                  existing_primitive_sea_state_map=physical,
                  assertions={name: bool(result) for name, result in checks.items()})
    print(json.dumps(report, indent=2))
    return 0 if all(checks.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=float, default=.5)
    parser.add_argument('--m', type=float, default=1.)
    parser.add_argument('--r', type=float, default=1.)
    parser.add_argument('--v', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=22297)
    args = parser.parse_args()
    if not all(np.isfinite([args.a, args.m, args.r, args.v])) or min(args.a, args.m, args.v) <= 0 or args.r < 0:
        parser.error('Require finite a,m,v>0 and r>=0.')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())

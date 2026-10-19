#!/usr/bin/env python3
"""Unrestricted R^15 adjoint oscillator mass supplier; no Dirac/photon claim.

The JSON separates an infinite-Hilbert-space variational theorem from finite
SU(4)-invariant Ritz calculations using the actual primitive edge code.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh

from pauli_compton import EdgeWords, boundary_ladder
from spin_orbital_locking import exact_units, primitive_matrix


DIMENSIONS = {0: 1, 1: 4, 2: 6, 3: 4, 4: 1}
CASIMIRS = {n: 5 * n * (4 - n) / 8 for n in range(5)}


def norm(value):
    return float(np.linalg.norm(value))


def sparse_norm(value):
    return norm(value.data)


def primitive_code() -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray],
                              list[np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    """All physical parities, with the neutral marker carrying only parity."""
    words = EdgeWords(5, [(0, 1), (1, 2), (2, 3), (0, 4)])
    bits, numbers = [], []
    for mask in range(16):
        physical = tuple(i for i in range(4) if mask >> i & 1)
        bits.append(words.tree_basis_state(physical + ((4,) if len(physical) % 2 else ())))
        numbers.append(len(physical))
    locations = {bit: index for index, bit in enumerate(bits)}
    creators = []
    for flavour in range(4):
        matrix = np.zeros((16, 16), complex)
        for column, bit in enumerate(bits):
            for target, amplitude in boundary_ladder(words, 4, flavour, {bit: 1.}, True).items():
                matrix[locations[target], column] = amplitude
        creators.append(matrix)
    # This is a complete orthonormal su(4) basis, not a chosen Clifford frame.
    basis = np.array([np.array(t, complex) / math.sqrt(8) for t in exact_units()])
    q_full = np.array([primitive_matrix(words, bits, t) for t in basis])
    f = np.empty((15, 15, 15))
    for b, a, c in itertools.product(range(15), repeat=3):
        f[b, a, c] = (-2j * np.trace((basis[b] @ basis[a] - basis[a] @ basis[b]) @ basis[c])).real
    f[abs(f) < 1e-13] = 0
    sectors = {n: np.flatnonzero(np.array(numbers) == n) for n in range(5)}
    q = {n: q_full[:, sectors[n][:, None], sectors[n]] for n in range(5)}
    bilinear_error = max(norm(q_full[a] - sum(
        (basis[a, i, j] * creators[i] @ creators[j].conj().T
         for i, j in itertools.product(range(4), repeat=2)), np.zeros((16, 16), complex)))
        for a in range(15))
    car_error = max(norm(creators[i].conj().T @ creators[j]
                         + creators[j] @ creators[i].conj().T - (np.eye(16) if i == j else 0))
                    for i, j in itertools.product(range(4), repeat=2))
    nilpotent_error = max(norm(creators[i] @ creators[j] + creators[j] @ creators[i])
                          for i, j in itertools.product(range(4), repeat=2))
    algebra = {n: max(norm(q[n][a] @ q[n][b] - q[n][b] @ q[n][a]
                           - 1j * np.einsum('c,cij->ij', f[a, b], q[n]))
                      for a, b in itertools.product(range(15), repeat=2)) for n in range(5)}
    casimir = {n: norm(sum((t @ t for t in q[n]), np.zeros((DIMENSIONS[n],) * 2, complex))
                       - CASIMIRS[n] * np.eye(DIMENSIONS[n])) for n in range(5)}
    return basis, f, q, creators, sectors, dict(
        edge_qubits=4, physical_flavours=4, full_physical_dimension=16,
        neutral_boundary_marker_is_not_a_spectator_spin=True,
        basis_trace_normalization_residual=norm(np.einsum('aij,bji->ab', basis, basis) - np.eye(15) / 2),
        actual_transfer_vs_boundary_bilinear_residual=bilinear_error,
        full_CAR_residual=car_error, full_creation_nilpotence_residual=nilpotent_error,
        sector_dimensions=DIMENSIONS, primitive_Casimirs=CASIMIRS,
        primitive_Casimir_residuals=casimir, primitive_su4_algebra_residuals=algebra)


def oscillator_basis(cutoff, coordinates=15):
    """Total excitation <= cutoff: includes every adjoint direction equally."""
    states = []
    for shell in range(cutoff + 1):
        for occupied in itertools.combinations_with_replacement(range(coordinates), shell):
            state = [0] * coordinates
            for a in occupied:
                state[a] += 1
            states.append(tuple(state))
    return states, {state: i for i, state in enumerate(states)}


def annihilators(states, locations):
    operators = []
    for a in range(len(states[0])):
        rows, columns, values = [], [], []
        for column, state in enumerate(states):
            if state[a]:
                target = list(state)
                target[a] -= 1
                rows.append(locations[tuple(target)])
                columns.append(column)
                values.append(math.sqrt(state[a]))
        operators.append(sparse.csr_matrix((values, (rows, columns)), shape=(len(states),) * 2))
    return operators


def oscillator_generators(states, locations, f):
    """L_b = sum_ca i f_bac a_c^dagger a_a; [L_b,phi_a]=i f_bac phi_c."""
    result = []
    for b in range(15):
        rows, columns, values = [], [], []
        for a, c in np.argwhere(f[b] != 0):
            for column, state in enumerate(states):
                if state[a]:
                    target = list(state)
                    target[a] -= 1
                    amplitude = math.sqrt(state[a] * (target[c] + 1))
                    target[c] += 1
                    rows.append(locations[tuple(target)])
                    columns.append(column)
                    values.append(1j * f[b, a, c] * amplitude)
        result.append(sparse.csr_matrix((values, (rows, columns)), shape=(len(states),) * 2))
    return result


def deterministic_vector(size, phase=0.):
    index = np.arange(size) + 1
    value = np.sin(index * math.sqrt(2) + phase) + 1j * np.cos(index * math.sqrt(3) + phase)
    return value / norm(value)


def recover_multiplet(seed, generators, dimension, threshold=2e-8):
    """Recover the actual SU(4) orbit, never an eigsh degeneracy count."""
    columns = [seed / norm(seed)]
    cursor = 0
    while cursor < len(columns):
        for generator in generators:
            candidate = generator @ columns[cursor]
            frame = np.column_stack(columns)
            # Reorthogonalize to prevent numerical noise becoming a new state.
            for _ in range(2):
                candidate -= frame @ (frame.conj().T @ candidate)
            length = norm(candidate)
            if length > threshold:
                if len(columns) == dimension:
                    raise RuntimeError('Ground orbit exceeds expected SU(4) multiplet dimension')
                columns.append(candidate / length)
        cursor += 1
    if len(columns) != dimension:
        raise RuntimeError(f'Incomplete ground orbit: {len(columns)} != {dimension}')
    return np.column_stack(columns)


def solve_sector(hamiltonian, generators, number, omega, cutoff, tolerance):
    dimension = DIMENSIONS[number]
    if number in (0, 4):
        ground = np.zeros((hamiltonian.shape[0], 1), complex)
        ground[0, 0] = 1
        energy = 15 * omega / 2
    else:
        values, vectors = eigsh(hamiltonian, k=1, which='SA', tol=tolerance,
                               v0=deterministic_vector(hamiltonian.shape[0]), maxiter=10000)
        energy = float(values[0])
        ground = recover_multiplet(vectors[:, 0], generators, dimension)
    ground_residual = norm(hamiltonian @ ground - energy * ground)
    closure = max(norm(j @ ground - ground @ (ground.conj().T @ (j @ ground))) for j in generators)
    casimir_action = sum((j @ (j @ ground) for j in generators), np.zeros_like(ground))
    casimir_residual = norm(casimir_action - CASIMIRS[number] * ground)
    # The ground projector is lifted, so a separate lowest solve probes its
    # orthogonal complement. A missed ground partner remains unlifted here.
    penalty = 4 * omega
    class LiftedGroundProjector(LinearOperator):
        def _matvec(self, x):
            return hamiltonian @ x + penalty * ground @ (ground.conj().T @ x)
    complement_operator = LiftedGroundProjector(dtype=np.dtype(np.complex128), shape=hamiltonian.shape)
    _, candidate = eigsh(complement_operator, k=1, which='SA', tol=tolerance,
                         v0=deterministic_vector(hamiltonian.shape[0], .31), maxiter=10000)
    candidate = candidate[:, 0]
    overlap = norm(ground.conj().T @ candidate)
    candidate -= ground @ (ground.conj().T @ candidate)
    if norm(candidate) < .5:
        raise RuntimeError('Complement probe selected lifted ground, not a complement eigenstate')
    candidate /= norm(candidate)
    complement_energy = float(np.vdot(candidate, hamiltonian @ candidate).real)
    complement_residual = norm(hamiltonian @ candidate - complement_energy * candidate)
    probe = deterministic_vector(hamiltonian.shape[0], .77)
    commutator_residual = max(norm(hamiltonian @ (j @ probe) - j @ (hamiltonian @ probe))
                              for j in generators)
    report = dict(N=number, total_dimension=hamiltonian.shape[0],
                  actual_ground_multiplet_dimension=ground.shape[1], energy_Ritz=energy,
                  ground_residual=ground_residual, ground_orthonormality_residual=norm(
                      ground.conj().T @ ground - np.eye(dimension)),
                  total_SU4_ground_orbit_closure_residual=closure,
                  total_SU4_ground_Casimir=CASIMIRS[number],
                  total_SU4_ground_Casimir_residual=casimir_residual,
                  total_SU4_H_commutator_probe_residual=commutator_residual,
                  complement_energy_Ritz=complement_energy,
                  complement_separation_Ritz=complement_energy - energy,
                  complement_ground_overlap=overlap, complement_eigen_residual=complement_residual)
    return ground, report


def field_observables(ground, number, states, extended_states, extended_locations,
                      mass, omega, mu, kappa, q):
    """Moments use full phi action into shell K+1, not (P phi P)^2."""
    d = DIMENSIONS[number]
    averaged = ground.shape[1]
    tensor = ground.reshape(len(states), d, averaged)
    extended = np.zeros((len(extended_states), d, averaged), complex)
    extended[:len(states)] = tensor
    phi_actions, pi_actions = [], []
    scale = 1 / math.sqrt(2 * mass * omega)
    for a in range(15):
        lower = np.zeros_like(extended)
        raise_ = np.zeros_like(extended)
        for index, state in enumerate(states):
            if state[a]:
                target = list(state)
                target[a] -= 1
                lower[extended_locations[tuple(target)]] += math.sqrt(state[a]) * tensor[index]
            target = list(state)
            target[a] += 1
            raise_[extended_locations[tuple(target)]] += math.sqrt(state[a] + 1) * tensor[index]
        phi_actions.append(scale * (lower + raise_))
        pi_actions.append(1j * math.sqrt(mass * omega / 2) * (raise_ - lower))
    means = np.array([np.vdot(extended, value).real / averaged for value in phi_actions])
    second = np.array([[np.vdot(left, right).real / averaged for right in phi_actions]
                       for left in phi_actions])
    covariance = second - np.outer(means, means)
    pi_second = np.array([norm(value) ** 2 / averaged for value in pi_actions])
    matter_mean = np.array([np.einsum('idm,de,iem->', tensor.conj(), t, tensor).real / averaged
                            for t in q])
    coupling = float(mu * sum(np.einsum('idm,de,iem->', extended.conj(), q[a], phi_actions[a]).real
                              for a in range(15)) / averaged)
    shells = np.array([sum(state) for state in states])
    weights = np.sum(abs(tensor) ** 2, axis=(1, 2)) / averaged
    probabilities = [float(weights[shells == shell].sum()) for shell in range(int(shells.max()) + 1)]
    # H maps K into K+1 only through its linear coupling. Missing shell is a
    # full-H eigen-defect, not an error bound on the Ritz energy.
    outside = np.zeros_like(extended[len(states):])
    for a in range(15):
        outside += mu * np.einsum('de,iem->idm', q[a], phi_actions[a][len(states):])
    return dict(field_means=means.tolist(), field_covariance=covariance.tolist(),
                field_second_moments=second.tolist(), averaged_matter_Q_means=matter_mean.tolist(),
                ensemble_mean_A0_Frobenius=norm(means) / math.sqrt(2),
                bare_field_variance=scale * scale,
                covariance_isotropy_residual=norm(covariance - np.eye(15) * np.trace(covariance) / 15),
                induced_variance_per_coordinate=float(np.trace(covariance) / 15 - scale * scale),
                oscillator_excitation_mean=float(np.dot(shells, weights)),
                total_excitation_shell_probabilities=probabilities,
                cutoff_shell_probability=probabilities[-1],
                full_H_outside_cutoff_ground_defect=norm(outside),
                kinetic_expectation=float(pi_second.sum() / (2 * mass)),
                bare_zero_preferring_potential_expectation=float(kappa * np.trace(second) / 2),
                microscopic_coupling_expectation=coupling,
                bare_oscillator_ground_energy=15 * omega / 2)


def residues(grounds, creators, sectors, oscillator_count):
    report = {}
    initial = grounds[2]
    for creation, target in ((True, 3), (False, 1)):
        rows = []
        for flavour, creator in enumerate(creators):
            full = creator if creation else creator.conj().T
            ladder = full[np.ix_(sectors[target], sectors[2])]
            action = np.einsum('de,iem->idm', ladder,
                               initial.reshape(oscillator_count, 6, 6)).reshape(-1, 6)
            amplitude = grounds[target].conj().T @ action
            rows.append(dict(flavour=flavour,
                             averaged_ground_residue=float(norm(amplitude) ** 2 / 6),
                             averaged_all_final_weight=float(norm(action) ** 2 / 6),
                             ground_transition_singular_values=np.linalg.svd(amplitude, compute_uv=False).tolist()))
        report['addition' if creation else 'removal'] = dict(
            initial_N=2, final_N=target, neutral_ensemble_dimension=6,
            charged_ground_dimension=4, per_flavour=rows,
            summed_ground_residue=sum(row['averaged_ground_residue'] for row in rows),
            summed_all_final_weight=sum(row['averaged_all_final_weight'] for row in rows),
            definition='sum_flavour Tr(P2 c Pcharged cdag P2)/6 (addition uses cdag first); actual boundary-marker code')
    return report


def exact_theorem(mu, mass, kappa) -> dict[str, Any]:
    omega = math.sqrt(kappa / mass)
    s = mu * mu / (2 * mass * omega ** 3)
    c1, c2 = CASIMIRS[1], CASIMIRS[2]
    e0 = 15 * omega / 2
    optimized_upper = e0 + omega / 2 * (1 - math.sqrt(1 + 4 * s * c2))
    charged_lower = e0 - omega * s * c1
    return dict(omega=omega, s=s, C1=c1, C2=c2,
                bare_zero_preferring_oscillator_energy=e0,
                charged_infinite_lower_bound=charged_lower,
                neutral_two_level_infinite_upper_bound=optimized_upper,
                infinite_charge_gap_lower_bound=charged_lower - optimized_upper,
                optimized_positive_interval='0 < s < 8/45',
                optimized_bound_strictly_positive=bool(0 < s < 8 / 45),
                earlier_trial_norm=1 + s * c2,
                earlier_trial_energy=e0 - omega * s * c2 / (1 + s * c2),
                earlier_trial_gap_lower=omega * s * (c2 / (1 + s * c2) - c1),
                earlier_positive_interval='0 < s < 2/15',
                sector_square_lower_bounds={n: e0 - omega * s * CASIMIRS[n] for n in range(5)})


def compressed_trial_control(q2, phi, oscillator_count, omega, mass, mu):
    # This two-level subspace is available already at total cutoff one. The
    # compression is checked on all six primitive neutral vectors, not one.
    seed = np.zeros((oscillator_count * 6, 6), complex)
    seed[:6] = np.eye(6)
    xseed = sum((sparse.kron(p, sparse.csr_matrix(t), format='csr') @ seed
                 for p, t in zip(phi, q2)), np.zeros_like(seed))
    length = math.sqrt(CASIMIRS[2] / (2 * mass * omega))
    frame = np.column_stack((seed, xseed / length))
    diagonal = np.repeat(omega * (np.array([0] + [1] * 15) + 7.5), 6)
    # Only seed and first shell have support; the remaining rows stay zero.
    hframe = np.zeros_like(frame)
    hframe[:96] = diagonal[:, None] * frame[:96]
    for p, t in zip(phi, q2):
        hframe += mu * (sparse.kron(p, sparse.csr_matrix(t), format='csr') @ frame)
    observed = frame.conj().T @ hframe
    block = np.array([[7.5 * omega, mu * length], [mu * length, 8.5 * omega]])
    expected = np.kron(block, np.eye(6))
    return dict(primitive_neutral_two_level_dimension=12,
                orthonormality_residual=norm(frame.conj().T @ frame - np.eye(12)),
                exact_compression_residual=norm(observed - expected),
                odd_parity_triple_term_residual=norm((xseed / length).conj().T @ (
                    hframe[:, 6:] - (8.5 * omega) * frame[:, 6:])),
                compressed_ground_energy=float(np.linalg.eigvalsh(observed)[0]),
                compressed_ground_multiplicity=6)


def single_field_control(basis, q, mass, kappa, mu, cutoff=32) -> dict[str, Any]:
    t = np.diag([3., -1., -1., -1.]) / math.sqrt(24)
    coefficients = 2 * np.einsum('aij,ji->a', basis, t).real
    omega = math.sqrt(kappa / mass)
    lower = np.diag(np.sqrt(np.arange(1, cutoff + 1)), 1)
    phi = (lower + lower.T) / math.sqrt(2 * mass * omega)
    h0 = np.diag(omega * (np.arange(cutoff + 1) + .5))
    rows = {}
    for n in (1, 2, 3):
        weights = np.linalg.eigvalsh(np.einsum('a,aij->ij', coefficients, q[n]))
        energies = [float(np.linalg.eigvalsh(h0 + mu * weight * phi)[0]) for weight in weights]
        largest = float(np.max(weights ** 2))
        rows[n] = dict(actual_primitive_weights=weights.tolist(), largest_weight_squared=largest,
                       exact_energy=omega / 2 - mu ** 2 * largest / (2 * kappa),
                       one_oscillator_energy_Ritz=min(energies))
    exact_gap = -5 * mu ** 2 / (48 * kappa)
    return dict(field='diag(3,-1,-1,-1)/sqrt(24)', oscillator_cutoff=cutoff, sectors=rows,
                exact_charged_gap=exact_gap,
                finite_charged_gaps={n: rows[n]['one_oscillator_energy_Ritz'] - rows[2]['one_oscillator_energy_Ritz']
                                     for n in (1, 3)},
                exact_gap_derivation='max Q_N^2 = 3/8, 1/6, 3/8 for N=1,2,3; -(mu^2/2kappa)(3/8-1/6)',
                implication='A single traceless field can favor charged sectors; it is not the full15 Casimir supplier.')


def source_hashes():
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in (
        'quantum_adjoint_mass.py', 'quantum_adjoint_mass.md',
        'spin_orbital_locking.py', 'pauli_compton.py')}


def run(args):
    basis, f, q, creators, sectors, primitive = primitive_code()
    mass, kappa = args.M, args.kappa
    omega = math.sqrt(kappa / mass)
    reports = []
    previous = {}
    structural_errors = []
    checks = [primitive['basis_trace_normalization_residual'] < 1e-11,
              primitive['actual_transfer_vs_boundary_bilinear_residual'] < 1e-11,
              primitive['full_CAR_residual'] < 1e-11, primitive['full_creation_nilpotence_residual'] < 1e-11,
              max(primitive['primitive_Casimir_residuals'].values()) < 1e-11,
              max(primitive['primitive_su4_algebra_residuals'].values()) < 1e-11]
    for cutoff in args.cutoffs:
        states, locations = oscillator_basis(cutoff)
        extended_states, extended_locations = oscillator_basis(cutoff + 1)
        annihilation = annihilators(states, locations)
        phi = [(a + a.T) / math.sqrt(2 * mass * omega) for a in annihilation]
        orbital = oscillator_generators(states, locations, f)
        # These exact sparse identities on the entire cutoff establish total
        # SU(4) covariance, including its outer shell, without random-only proof.
        covariance_error = 0.
        for b, a in itertools.product(range(15), repeat=2):
            desired = sum((1j * f[b, a, c] * phi[c] for c in np.flatnonzero(f[b, a])),
                          sparse.csr_matrix(phi[a].shape, dtype=complex))
            covariance_error = max(covariance_error, sparse_norm(orbital[b] @ phi[a] - phi[a] @ orbital[b] - desired))
        structural_errors.append(dict(cutoff=cutoff, full_cutoff_oscillator_adjoint_covariance_residual=covariance_error))
        checks.append(covariance_error < 1e-9)
        h0 = sparse.diags(omega * (np.array([sum(state) for state in states]) + 7.5), format='csr')
        for mu in args.mu:
            grounds, spectrum = {}, {}
            for n in range(5):
                d = DIMENSIONS[n]
                matter_identity = sparse.eye(d, format='csr')
                generators = [sparse.kron(l, matter_identity, format='csr')
                              + sparse.kron(sparse.eye(len(states), format='csr'), sparse.csr_matrix(t), format='csr')
                              for l, t in zip(orbital, q[n])]
                h = sparse.kron(h0, matter_identity, format='csr')
                for p, t in zip(phi, q[n]):
                    h += mu * sparse.kron(p, sparse.csr_matrix(t), format='csr')
                ground, row = solve_sector(h, generators, n, omega, cutoff, args.tolerance)
                row['field'] = field_observables(ground, n, states, extended_states, extended_locations,
                                                  mass, omega, mu, kappa, q[n])
                row['bare_energy_lowering_Ritz'] = 7.5 * omega - row['energy_Ritz']
                key = mu, n
                row['previous_cutoff_energy_Ritz'] = previous.get(key)
                row['nested_energy_change'] = None if key not in previous else row['energy_Ritz'] - previous[key]
                previous[key] = row['energy_Ritz']
                field = row['field']
                checks.extend([row['ground_residual'] < 1e-7,
                               row['complement_eigen_residual'] < 1e-7,
                               row['complement_ground_overlap'] < 1e-6,
                               row['complement_separation_Ritz'] > 1e-6,
                               row['total_SU4_ground_orbit_closure_residual'] < 1e-6,
                               row['total_SU4_ground_Casimir_residual'] < 1e-6,
                               row['total_SU4_H_commutator_probe_residual'] < 1e-8,
                               row['ground_orthonormality_residual'] < 1e-8,
                               norm(field['field_means']) < 1e-6,
                               field['covariance_isotropy_residual'] < 1e-6,
                               abs(field['kinetic_expectation'] + field['bare_zero_preferring_potential_expectation']
                                   + field['microscopic_coupling_expectation'] - row['energy_Ritz']) < 1e-7,
                               row['nested_energy_change'] is None or row['nested_energy_change'] < 1e-8])
                if mu and n in (1, 2, 3):
                    checks.extend([row['bare_energy_lowering_Ritz'] > 0,
                                   field['oscillator_excitation_mean'] > 0,
                                   field['induced_variance_per_coordinate'] > 0])
                grounds[n], spectrum[n] = ground, row
            trial = compressed_trial_control(q[2], phi, len(states), omega, mass, mu)
            theorem = exact_theorem(mu, mass, kappa)
            residue = residues(grounds, creators, sectors, len(states))
            checks.extend([trial['orthonormality_residual'] < 1e-10,
                           trial['exact_compression_residual'] < 1e-10,
                           trial['odd_parity_triple_term_residual'] < 1e-10,
                           abs(trial['compressed_ground_energy'] - theorem['neutral_two_level_infinite_upper_bound']) < 1e-9])
            for direction in residue.values():
                checks.extend([abs(direction['summed_all_final_weight'] - 2) < 1e-8,
                               0 < direction['summed_ground_residue'] <= 2 + 1e-8])
            gaps = {n: spectrum[n]['energy_Ritz'] - spectrum[2]['energy_Ritz'] for n in (1, 3)}
            reports.append(dict(total_excitation_cutoff=cutoff, oscillator_dimension=len(states), mu=mu,
                                sectors=spectrum, actual_primitive_ground_residues=residue,
                                exact_infinite_theorem=theorem, actual_primitive_two_level_control=trial,
                                charged_gap_Ritz=gaps,
                                fixed_boundary_flux_charged_gap_Ritz={n: gap + args.UE / 2 for n, gap in gaps.items()},
                                fixed_boundary_flux_infinite_gap_lower=theorem['infinite_charge_gap_lower_bound'] + args.UE / 2,
                                finite_Ritz_differences_are_not_infinite_gap_certificates=True))
    controls = [single_field_control(basis, q, mass, kappa, mu) for mu in args.mu]
    for control in controls:
        checks.extend(abs(gap - control['exact_charged_gap']) < 1e-9 for gap in control['finite_charged_gaps'].values())
    return dict(model='unrestricted_quantum_su4_adjoint_mass_supplier',
                units={'hbar': 1, 'phi_a': 'field amplitude F, all real R^15',
                       'Pi_a': 'action/F; [phi_a,Pi_b]=i delta_ab',
                       'M': '1/(energy F^2) with hbar=1', 'kappa': 'energy/F^2',
                       'mu': 'energy/F', 'omega': 'energy with hbar=1',
                       'A0': 'sum_a phi_a T_a (field amplitude), not a prescribed mass'},
                parameters=dict(M=mass, kappa=kappa, omega=omega, mu=args.mu, cutoffs=args.cutoffs,
                                eigensolver_tolerance=args.tolerance, fixed_U1_boundary_UE=args.UE),
                primitive=primitive, total_SU4_structural_checks=structural_errors,
                spectra=reports, adverse_single_field_controls=controls,
                all_checks_pass=bool(all(checks)), source_sha256=source_hashes(),
                scope=['Distinct microscopic R^15 field supplier, not the positive-halfline fixed-Clifford radial/sea Hamiltonian.',
                       'Bare harmonic potential prefers zero; quantum dressing comes from microscopic mu phi.Q coupling.',
                       'Neutral ground multiplet six; charged ground multiplets four, checked at each reported finite cutoff.',
                       'SU(4)-averaged mean A0 is zero: no ordered beta frame or unique vacuum.',
                       'Infinite gap follows operator square bounds and a variational neutral upper, not finite Ritz differences.',
                       'Optional outward U(1) flux N-2 adds UE*(N-2)^2/2 only; no dynamical photon spectrum.',
                       'No chosen Clifford frame, spacetime spin, Dirac propagation, 3D phase, or full electron-photon scattering transfer.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cutoffs', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--mu', type=float, nargs='+', default=[.2, .4])
    parser.add_argument('--M', type=float, default=1.)
    parser.add_argument('--kappa', type=float, default=1.)
    parser.add_argument('--UE', type=float, default=0., help='Fixed one-cell boundary flux energy coefficient, not photon dynamics')
    parser.add_argument('--tolerance', type=float, default=1e-11)
    args = parser.parse_args()
    if not (math.isfinite(args.M) and args.M > 0 and math.isfinite(args.kappa) and args.kappa > 0):
        parser.error('M and kappa must be finite and positive')
    if not all(math.isfinite(mu) for mu in args.mu):
        parser.error('mu must be finite')
    if not math.isfinite(args.UE) or args.UE < 0:
        parser.error('UE must be finite and nonnegative')
    if args.cutoffs != sorted(set(args.cutoffs)) or not args.cutoffs or min(args.cutoffs) < 1:
        parser.error('cutoffs must be strictly increasing positive integers')
    if not math.isfinite(args.tolerance) or not 0 < args.tolerance < 1:
        parser.error('tolerance must lie between zero and one')
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), all_checks_pass=result['all_checks_pass'],
                          model=result['model']), sort_keys=True))
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

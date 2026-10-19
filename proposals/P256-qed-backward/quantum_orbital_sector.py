#!/usr/bin/env python3
"""Quantum orbital hopping in both actual Gauss/code sectors of CoupledSquare.

The oscillator coordinate does not commute with its kinetic Hamiltonian. Every
coupling uses the primitive transfer_matrix path, including static improved
paths. Direct execution reports exact finite identities and cutoff diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from code_phase import CoupledSquare, norm


def oscillator(dimension: int, frequency: float, quartic: float) -> dict:
    """Galerkin compression of omega(N+1/2)+lambda X_infinite**4.

    The padded fourth power retains excursions above the retained Fock space;
    X_L**4 would instead define a different, nonnested boundary Hamiltonian.
    """
    if dimension < 2 or frequency <= 0 or quartic < 0:
        raise ValueError('Require dimension>=2, omega>0 and quartic>=0')
    padded = dimension + 4
    a = np.diag(np.sqrt(np.arange(1, padded, dtype=float)), 1)
    x = (a + a.T) / np.sqrt(2 * frequency)
    x4 = np.linalg.matrix_power(x, 4)[:dimension, :dimension]
    x = x[:dimension, :dimension]
    a = a[:dimension, :dimension]
    p = -1j * np.sqrt(frequency / 2) * (a - a.T)
    h = frequency * np.diag(np.arange(dimension, dtype=float) + .5) + quartic * x4
    boundary = np.zeros((dimension, dimension))
    boundary[-1, -1] = 1
    return dict(X=x, P=p, H=h, boundary=boundary,
                noncommuting_coordinate_norm=norm(h @ x - x @ h),
                truncated_CCR_identity_residual=norm(
                    x @ p - p @ x - 1j * (np.eye(dimension) - dimension * boundary)))


def primitive_hopping_derivative(model: CoupledSquare, phase: float):
    """Differentiate only hopping-dependent one-link paths, not improved paths."""
    derivative = np.zeros((model.dimension, model.dimension), dtype=complex)
    paths = []
    for index, edge in enumerate(model.edges):
        transfer = model.transfer_matrix(edge)
        amplitude = np.exp(1j * phase * (index + 1))
        hermitian_path = amplitude * transfer + amplitude.conjugate() * transfer.conj().T
        derivative += hermitian_path
        paths.append((edge, hermitian_path))
    return derivative, paths


def expectation(state, operator):
    return float(np.vdot(state, operator @ state).real)


def ground_energy(matrix):
    return float(eigh(matrix, subset_by_index=(0, 0), eigvals_only=True)[0])


def run_case(oscillator_cutoff=8, coupling=.15, rotor_cutoff=1,
             magnetic=.7, electric=.8, delta=2., hopping=.6,
             improved=.19, phase=.23, frequency=1.3, quartic=.08) -> dict:
    model = CoupledSquare(rotor_cutoff, magnetic, electric, delta, hopping, improved, phase)
    static, stabilizer, cosine, matter = model.operators()
    derivative, paths = primitive_hopping_derivative(model, phase)
    shifted_model = CoupledSquare(rotor_cutoff, magnetic, electric, delta,
                                  hopping + 1, improved, phase)
    shifted_static = shifted_model.operators()[0]
    derivative_residual = norm(shifted_static - static - derivative)
    aux = oscillator(oscillator_cutoff, frequency, quartic)
    x, haux = aux['X'], aux['H']
    physical_identity = np.eye(model.dimension)
    orbital_identity = np.eye(oscillator_cutoff)
    # The static Hamiltonian retains EVERY improved transfer unchanged.
    hfull = (np.kron(static, orbital_identity) + np.kron(physical_identity, haux)
             + coupling * np.kron(derivative, x))
    plus, labels = model.sector_basis(1)
    minus, minus_labels = model.sector_basis(-1)
    if labels != minus_labels:
        raise AssertionError('Unmatched physical occupation/electric domains')
    phases = model.phase_intertwiner()
    factors = np.array([phases[occupation] * (-1.) ** fields[0]
                        for occupation, fields in labels])
    mapped = minus * factors[np.newaxis, :]
    exact_qr = model.pauli_matrix(model.Q) @ np.diag(
        [(-1.) ** fields[0] for _, fields in model.raw])
    plus_full = np.kron(plus, orbital_identity)
    minus_full = np.kron(minus, orbital_identity)
    mapped_full = np.kron(mapped, orbital_identity)
    hplus = plus_full.conj().T @ hfull @ plus_full
    hminus = minus_full.conj().T @ hfull @ minus_full
    transformed = mapped_full.conj().T @ hfull @ mapped_full
    cplus = plus.conj().T @ cosine @ plus
    correction = 2 * magnetic * np.kron(cplus, orbital_identity)
    sector_identity = np.eye(hplus.shape[0])
    identity_error = transformed - hplus - correction - delta * sector_identity
    energy_plus, vectors_plus = eigh(hplus)
    energy_minus = eigh(hminus, eigvals_only=True)
    full_energies = eigh(hfull, eigvals_only=True)
    ground = vectors_plus[:, 0]
    raw_ground = plus_full @ ground
    psi = raw_ground.reshape(model.dimension, oscillator_cutoff)
    rho_orbital = psi.conj().T @ psi
    orbital_eigenvalues = np.linalg.eigvalsh(rho_orbital)
    orbital_eigenvalues = orbital_eigenvalues[orbital_eigenvalues > 1e-14]
    orbital_entropy = float(-np.sum(orbital_eigenvalues * np.log(orbital_eigenvalues)))
    orbital_linear_entropy = float(1 - np.trace(rho_orbital @ rho_orbital).real)
    xfull = np.kron(physical_identity, x)
    dfull = np.kron(derivative, orbital_identity)
    cfull = np.kron(cosine, orbital_identity)
    electric_operator = np.diag([electric / 2 * sum(e * e for e in fields)
                                 for _, fields in model.raw])
    efull = np.kron(electric_operator, orbital_identity)
    xmean = expectation(raw_ground, xfull)
    dmean = expectation(raw_ground, dfull)
    cmean = expectation(raw_ground, cfull)
    emean = expectation(raw_ground, efull)
    xd = expectation(raw_ground, np.kron(derivative, x))
    xc = expectation(raw_ground, np.kron(cosine, x))
    xe = expectation(raw_ground, np.kron(electric_operator, x))
    quantum_tangent = plus_full.conj().T @ np.kron(derivative, x) @ plus_full
    step = 2e-5
    finite_difference = (ground_energy(hplus + step * quantum_tangent)
                         - ground_energy(hplus - step * quantum_tangent)) / (2 * step)
    derivative_error = abs(finite_difference - xd)
    probabilities = np.abs(psi)**2
    orbital_boundary = float(probabilities[:, -1].sum())
    rotor_boundary = np.array([any(abs(e) == rotor_cutoff for e in fields)
                               for _, fields in model.raw])
    rotor_boundary_probability = float(probabilities[rotor_boundary].sum())
    # A stabilizer-dressed primitive path has the WRONG code-sector path sign.
    # Uniformly negating a coefficient in BOTH sectors would not be an adversary:
    # the theorem deliberately permits arbitrary common coefficients.
    wrong_path, right_path_operator = paths[0]
    wrong_derivative = (derivative - right_path_operator
                        + stabilizer @ right_path_operator)
    wrong_h = (np.kron(static, orbital_identity) + np.kron(physical_identity, haux)
               + coupling * np.kron(wrong_derivative, x))
    wrong_path_residual = norm(mapped_full.conj().T @ wrong_h @ mapped_full
                               - plus_full.conj().T @ wrong_h @ plus_full
                               - correction - delta * sector_identity)
    wrong_rotor = model.pauli_matrix(model.Q) @ np.diag(
        [(-1.) ** fields[1] for _, fields in model.raw])
    wrong_mapped = np.kron(wrong_rotor @ plus, orbital_identity)
    wrong_rotor_residual = norm(wrong_mapped.conj().T @ hfull @ wrong_mapped
                                - hplus - correction - delta * sector_identity)
    extended_stabilizer = np.kron(stabilizer, orbital_identity)
    aux_operator = np.kron(physical_identity, haux)
    exact_qr_full = np.kron(exact_qr, orbital_identity)
    quantum_matter = (np.kron(matter, orbital_identity)
                      + coupling * np.kron(derivative, x))
    path_errors = [norm(mapped.conj().T @ model.transfer_matrix(term.path) @ mapped
                        - plus.conj().T @ model.transfer_matrix(term.path) @ plus)
                   for term in model.transfers]
    lower = delta - 2 * magnetic
    operator_margin = float(eigh(transformed - hplus - lower * sector_identity,
                                eigvals_only=True)[0])
    ordered_margin = float(np.min(energy_minus - energy_plus - lower))
    residuals = dict(
        hamiltonian_hermiticity=norm(hfull - hfull.conj().T),
        physical_code_sector_commutator=norm(extended_stabilizer @ hfull
                                            - hfull @ extended_stabilizer),
        actual_hopping_derivative=derivative_residual,
        derived_QR_tensor_identity=norm(mapped_full - exact_qr_full @ plus_full),
        full_quantum_transformed_identity=norm(identity_error),
        all_present_primitive_path_QR_cancellation=max(path_errors),
        quantum_matter_cancellation=norm(mapped_full.conj().T @ quantum_matter @ mapped_full
                                        - plus_full.conj().T @ quantum_matter @ plus_full),
        unchanged_auxiliary_H=norm(exact_qr_full.conj().T @ aux_operator @ exact_qr_full
                                  - aux_operator),
        full_spectrum_union=norm(full_energies - np.sort(np.r_[energy_plus, energy_minus])),
        truncated_CCR=aux['truncated_CCR_identity_residual'])
    checks = dict(
        exact_operator_identity=max(residuals.values()) < 2e-9,
        defect_operator_bound=operator_margin >= -2e-9,
        ordered_spectrum_bound=ordered_margin >= -2e-9,
        ground_defect_bound=float(energy_minus[0] - energy_plus[0]) >= lower - 2e-9,
        noncommuting_quantum_coefficient=aux['noncommuting_coordinate_norm'] > 1e-6,
        orbital_not_spectator=abs(coupling) < 1e-14 or (
            orbital_linear_entropy > 1e-10 and abs(xd - xmean * dmean) > 1e-10),
        coupled_energy_Hellmann_Feynman=derivative_error < 3e-7,
        wrong_code_path_sign_rejected=abs(coupling) < 1e-14 or wrong_path_residual > 1e-6,
        wrong_rotor_factor_rejected=wrong_rotor_residual > 1e-6,
        microscopic_Gauss_preserved=model.constraint_violations == 0
                                      and shifted_model.constraint_violations == 0)
    return dict(
        inputs=dict(oscillator_dimension=oscillator_cutoff, rotor_electric_cutoff=rotor_cutoff,
                    g=coupling, omega=frequency, quartic=quartic, K=magnetic, UE=electric,
                    Delta=delta, hopping=hopping, improved_hopping=improved, phase=phase,
                    background=model.background, N=2, eta=model.eta),
        dimensions=dict(full_two_code_sectors=model.dimension * oscillator_cutoff,
                        each_actual_sector=hplus.shape[0], physical_raw=model.dimension),
        primitive_paths=[dict(path=term.path, coefficient=[float(term.coefficient.real),
                                                          float(term.coefficient.imag)],
                              quantum_X_coupling=len(term.path) == 2)
                         for term in model.transfers],
        residuals=residuals,
        spectrum=dict(plus_ground=float(energy_plus[0]), minus_ground=float(energy_minus[0]),
                      defect_ground_gap=float(energy_minus[0] - energy_plus[0]),
                      guaranteed_Delta_minus_2K=lower,
                      operator_margin_above_bound=operator_margin,
                      ordered_spectrum_margin_above_bound=ordered_margin,
                      plus_first_excitation_gap=float(energy_plus[1] - energy_plus[0]),
                      cosine_operator_norm=float(np.max(np.abs(np.linalg.eigvalsh(cplus))))),
        quantum_state=dict(aux_H_X_commutator_norm=aux['noncommuting_coordinate_norm'],
                           orbital_entropy_nats=orbital_entropy,
                           orbital_linear_entropy=orbital_linear_entropy,
                           mean_X=xmean, mean_hopping_derivative=dmean,
                           mean_photon_cosine=cmean, mean_electric_energy=emean,
                           covariance_X_hopping=xd - xmean * dmean,
                           covariance_X_photon_cosine=xc - xmean * cmean,
                           covariance_X_electric_energy=xe - xmean * emean,
                           energy_derivative_g=xd, centered_energy_difference_g=finite_difference,
                           Hellmann_Feynman_residual=derivative_error),
        cutoff_diagnostics=dict(orbital_top_Fock_probability=orbital_boundary,
                                expected_CCR_boundary_defect=oscillator_cutoff * orbital_boundary,
                                rotor_any_boundary_probability=rotor_boundary_probability,
                                not_a_cutoff_error_bound=True),
        adverse=dict(wrong_sector_sign_primitive_path=wrong_path,
                     stabilizer_dressed_quantum_path_identity_residual=wrong_path_residual,
                     wrong_rotor_edge=1, wrong_rotor_identity_residual=wrong_rotor_residual),
        gauss=dict(max_raw_residual=max(max(abs(g) for g in model.gauss(q, fields))
                                       for q, fields in model.raw),
                   primitive_constraint_violations=model.constraint_violations,
                   silent_projection_used=False),
        checks=checks, all_checks_pass=all(checks.values()))


def run(cutoffs=(4, 8, 12), couplings=(.15, .9), rotor_cutoff=1,
        frequency=1.3, quartic=.08) -> dict:
    if len(set(cutoffs)) != len(cutoffs) or not cutoffs:
        raise ValueError('Use distinct nonempty oscillator cutoffs')
    rows = [run_case(cutoff, g, rotor_cutoff=rotor_cutoff,
                     frequency=frequency, quartic=quartic)
            for g in couplings for cutoff in sorted(cutoffs)]
    sequences = []
    for g in couplings:
        sequence = [row for row in rows if row['inputs']['g'] == g]
        previous = None
        for row in sequence:
            current = row['spectrum']['plus_ground']
            sequences.append(dict(g=g, oscillator_dimension=row['inputs']['oscillator_dimension'],
                                  plus_ground_change=None if previous is None else current - previous,
                                  nested_variational_energy_nonincrease=previous is None
                                      or current <= previous + 2e-9))
            previous = current
    checks = dict(all_finite_consumers=all(row['all_checks_pass'] for row in rows),
                  nested_Galerkin_variational_order=all(
                      item['nested_variational_energy_nonincrease'] for item in sequences),
                  photon_orbital_correlation_observed=any(
                      abs(row['quantum_state']['covariance_X_photon_cosine']) > 1e-10
                      for row in rows))
    return dict(status='quantum_orbital_coefficients_share_actual_bosonic_statistics_selector',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                primitive_source_sha256=hashlib.sha256(
                    Path(__file__).with_name('code_phase.py').read_bytes()).hexdigest(),
                domain='Full physical N=2 raw Pauli square, two code sectors, noncyclic photon rotors, quantum quartic oscillator',
                theorem='T_s=Q_eta R_eta tensor I_aux: T_s* H_s T_s=H_plus+2K sum_def C_p+Delta Ndef; inf(spec H_s)-inf(spec H_plus)>=(Delta-2K)Ndef',
                rows=rows, cutoff_sequences=sequences, checks=checks,
                all_checks_pass=all(checks.values()),
                input_debt=['Primitive quantum tensor/Born/dynamics and scalar couplings remain model inputs',
                            'Finite square is not a Coulomb phase or a full fluctuating Clifford-selection theorem',
                            'Fock/electric boundary weights are diagnostics, not certified continuum errors'])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oscillator-cutoffs', type=int, nargs='+', default=[4, 8, 12])
    parser.add_argument('--couplings', type=float, nargs='+', default=[.15, .9])
    parser.add_argument('--rotor-cutoff', type=int, default=1)
    parser.add_argument('--frequency', type=float, default=1.3)
    parser.add_argument('--quartic', type=float, default=.08)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(tuple(args.oscillator_cutoffs), tuple(args.couplings), args.rotor_cutoff,
                 args.frequency, args.quartic)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Ground/low-energy robustness of the actual Gauss-physical Pauli/rotor square.

All matrices retain both code sectors. A local code Z changes the code sector
without changing occupation, electric flux, total charge, or any Gauss value.
Run directly for exact finite-spectrum comparisons and adversarial strong mixing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from code_phase import CoupledSquare
from pauli_compton import PauliWord


def operator_norm(matrix):
    """Spectral norm, never the Frobenius norm in a perturbation bound."""
    return float(np.linalg.norm(matrix, ord=2))


def mixing_lowering(gap, coupling):
    """Positive lowering of the bottom eigenvalue of [[0,-c],[-c,gap]]."""
    return float(2 * coupling * coupling
                 / (np.hypot(gap, 2 * coupling) + gap))


def outside_bound(coupling, separation):
    if separation <= 0:
        raise ValueError('Leakage theorem requires positive spectral separation')
    ratio = coupling / separation
    return float(ratio * ratio / (1 + ratio * ratio))


def run(cutoff=1, magnetic=0.7, electric=0.8, delta=2.0,
        hopping=0.6, improved=0.19, phase=0.23):
    model = CoupledSquare(cutoff, magnetic, electric, delta, hopping, improved, phase)
    H0, S, _, _ = model.operators()
    plus, plus_labels = model.sector_basis(1)
    minus, minus_labels = model.sector_basis(-1)
    if plus_labels != minus_labels:
        raise AssertionError('Different physical rotor/occupation domains')
    Hp = plus.conj().T @ H0 @ plus
    Hq = minus.conj().T @ H0 @ minus
    ep, up = np.linalg.eigh(Hp)
    eq = np.linalg.eigvalsh(Hq)
    full0 = np.sort(np.r_[ep, eq])
    E0 = float(ep[0])
    gamma = float(eq[0] - E0)
    g0 = float(full0[1] - E0)
    tolerance = 2e-10 * max(1.0, operator_norm(H0))
    if gamma <= 0 or g0 <= tolerance:
        raise ValueError('This consumer follows an isolated unperturbed ground state')

    # A genuinely allowed local microscopic operator, not P_G X P_G = 0.
    word = PauliWord(z=1)
    if not word.hermitian() or any(not word.commutes(b) for b in model.words.B):
        raise AssertionError('Local Z must preserve every occupation')
    if word.commutes(model.S):
        raise AssertionError('Chosen local Z must change the loop sign')
    V = model.pauli_matrix(word)
    norm_v = operator_norm(V)
    Vqp = minus.conj().T @ V @ plus
    Vpp = plus.conj().T @ V @ plus
    Vqq = minus.conj().T @ V @ minus
    ident = np.eye(model.dimension)
    residuals = {
        'H0_hermitian': operator_norm(H0 - H0.conj().T),
        'H0_commutes_S': operator_norm(H0 @ S - S @ H0),
        'Z_hermitian': operator_norm(V - V.conj().T),
        'Z_involution': operator_norm(V @ V - ident),
        'Z_anticommutes_S': operator_norm(V @ S + S @ V),
        'Z_plus_diagonal_block': operator_norm(Vpp),
        'Z_minus_diagonal_block': operator_norm(Vqq),
        'physical_sector_completeness': operator_norm(plus @ plus.conj().T
                                                     + minus @ minus.conj().T - ident),
        'H0_sector_union': float(np.max(np.abs(np.linalg.eigvalsh(H0) - full0))),
    }

    # Diagnose why an undressed X is not an alternative permitted perturbation.
    bare_x = PauliWord(x=1)
    bare_x_gauss_changes = []
    z_gauss_changes = []
    for q, fields in model.raw:
        qx, _ = bare_x.action(q)
        qz, _ = word.action(q)
        bare_x_gauss_changes.append(max(abs(v) for v in model.gauss(qx, fields)))
        z_gauss_changes.append(max(abs(v) for v in model.gauss(qz, fields)))

    # Retain EVERY unperturbed selected-sector eigenstate below the defect edge,
    # rather than throwing away legitimate excited matter/photon states.
    low_count = int(np.count_nonzero(ep < eq[0] - tolerance))
    low_top = float(full0[low_count - 1])
    next_energy = float(full0[low_count])
    cluster_gap = next_energy - low_top
    threshold = (next_energy + low_top) / 2
    low_basis = plus @ up[:, :low_count]
    scale = min(g0, cluster_gap, gamma, float(eq[0] - low_top))
    amplitudes = [factor * scale for factor in (0.1, 0.05, 0.025, 0.0125)]
    ground0 = plus @ up[:, 0]
    response = np.linalg.solve(Hq - E0 * np.eye(len(Hq)), Vqp @ up[:, 0])
    leakage_coefficient = float(np.vdot(response, response).real)
    shift_coefficient = float(np.vdot(Vqp @ up[:, 0], response).real)
    rows = []
    for amplitude in amplitudes:
        H = H0 + amplitude * V
        energies, states = np.linalg.eigh(H)
        energy = float(energies[0])
        state = states[:, 0]
        x = plus.conj().T @ state
        y = minus.conj().T @ state
        leakage = float(np.vdot(y, y).real)
        eps = amplitude * norm_v
        c = amplitude * operator_norm(Vqp)
        low_states = states[:, :low_count]
        low_leakages = np.sum(np.abs(minus.conj().T @ low_states) ** 2, axis=0)
        low_bounds = [outside_bound(c, float(eq[0] - energy_j))
                      for energy_j in energies[:low_count]]
        # Exact Feshbach elimination of the retained complementary sector.
        resolvent_y = np.linalg.solve(Hq - energy * np.eye(len(Hq)), Vqp @ x)
        feshbach_residual = float(np.linalg.norm(
            (Hp - energy * np.eye(len(Hp))) @ x
            - amplitude * amplitude * Vqp.conj().T @ resolvent_y))
        principal_cosines = np.linalg.svd(low_basis.conj().T @ low_states,
                                         compute_uv=False)
        negative_energies = np.linalg.eigvalsh(H0 - amplitude * V)
        row = {
            'amplitude': amplitude, 'total_norm_epsilon': eps,
            'ground_energy': energy, 'ground_energy_shift': energy - E0,
            'outside_sector_weight': leakage,
            'pure_offdiagonal_weight_bound': outside_bound(c, gamma),
            'a_posteriori_weight_bound': outside_bound(c, float(eq[0] - energy)),
            'general_norm_weight_bound': outside_bound(eps, gamma - 2 * eps),
            'energy_shift_lower_bound': -mixing_lowering(gamma, c),
            'energy_shift_upper_bound': 0.0,
            'ground_fidelity_to_unperturbed': float(abs(np.vdot(ground0, state)) ** 2),
            'isolated_ground_fidelity_loss_bound': outside_bound(eps, g0 - eps),
            'perturbed_ground_gap': float(energies[1] - energies[0]),
            'Weyl_ground_gap_lower_bound': g0 - 2 * eps,
            'retained_low_energy_count': int(np.count_nonzero(energies < threshold)),
            'retained_low_energies': energies[:low_count].tolist(),
            'retained_low_state_outside_weights': low_leakages.tolist(),
            'retained_low_state_weight_bounds': low_bounds,
            'low_cluster_min_principal_cosine': float(principal_cosines.min()),
            'low_cluster_principal_cosine_squared_lower_bound': max(
                0.0, 1 - low_count * outside_bound(eps, cluster_gap - 2 * eps)),
            'low_cluster_ordered_energy_shift_max': float(np.max(
                np.abs(energies[:low_count] - full0[:low_count]))),
            'ground_leakage_over_amplitude_squared': leakage / amplitude ** 2,
            'ground_lowering_over_amplitude_squared': (E0 - energy) / amplitude ** 2,
            'Feshbach_residual': feshbach_residual,
            'complement_equation_residual': float(np.linalg.norm(y + amplitude * resolvent_y)),
            'lambda_sign_spectrum_residual': float(np.max(np.abs(energies - negative_energies))),
            'full_spectrum': energies.tolist(),
        }
        row['acceptance'] = bool(
            energy - E0 >= row['energy_shift_lower_bound'] - tolerance
            and energy <= E0 + tolerance
            and leakage <= row['a_posteriori_weight_bound'] + tolerance
            and leakage <= row['general_norm_weight_bound'] + tolerance
            and 1 - row['ground_fidelity_to_unperturbed']
                <= row['isolated_ground_fidelity_loss_bound'] + tolerance
            and row['perturbed_ground_gap'] >= g0 - 2 * eps - tolerance
            and row['retained_low_energy_count'] == low_count
            and row['low_cluster_ordered_energy_shift_max'] <= eps + tolerance
            and row['low_cluster_min_principal_cosine'] ** 2
                >= row['low_cluster_principal_cosine_squared_lower_bound'] - tolerance
            and np.all(low_leakages <= np.array(low_bounds) + tolerance)
            and feshbach_residual < tolerance
            and row['complement_equation_residual'] < tolerance
            and row['lambda_sign_spectrum_residual'] < tolerance)
        rows.append(row)

    slopes = [float(np.log(a['outside_sector_weight'] / b['outside_sector_weight'])
                    / np.log(a['amplitude'] / b['amplitude']))
              for a, b in zip(rows[:-1], rows[1:])]

    # Strong coupling is observed on the SAME full physical Hamiltonian. Center
    # H0 only for a norm estimate; this does not change states or the experiment.
    center = float((full0[-1] + full0[0]) / 2)
    centered_norm = operator_norm(H0 - center * ident)
    strong_amplitude = 10000 * max(centered_norm, gamma)
    strong_energies, strong_states = np.linalg.eigh(H0 + strong_amplitude * V)
    strong_state = strong_states[:, 0]
    strong_y = minus.conj().T @ strong_state
    strong_weight = float(np.vdot(strong_y, strong_y).real)
    strong_half_bound = float(np.sqrt(centered_norm / strong_amplitude))
    strong = {
        'amplitude': strong_amplitude,
        'total_norm_epsilon': strong_amplitude * norm_v,
        'weak_general_norm_theorem_applicable': False,
        'isolated_unperturbed_cluster_theorem_applicable': False,
        'ground_energy_shift': float(strong_energies[0] - E0),
        'outside_sector_weight': strong_weight,
        'half_weight_distance_bound': strong_half_bound,
        'centered_unperturbed_norm': centered_norm,
        'Z_expectation': float(np.vdot(strong_state, V @ strong_state).real),
        'full_spectrum': strong_energies.tolist(),
        'acceptance': bool(abs(strong_weight - 0.5) <= strong_half_bound + tolerance
                           and strong_weight > 0.45),
    }
    smallest = rows[-1]
    asymptotic_error = abs(smallest['ground_leakage_over_amplitude_squared']
                           / leakage_coefficient - 1)
    acceptance = bool(
        max(residuals.values()) < tolerance
        and gamma >= delta - 2 * magnetic - tolerance
        and max(z_gauss_changes) == 0 and min(bare_x_gauss_changes) > 0
        and model.constraint_violations == 0
        and all(row['acceptance'] for row in rows)
        and all(1.8 < slope < 2.2 for slope in slopes)
        and asymptotic_error < 0.03
        and strong['acceptance'])
    report = {
        'inputs': {'cutoff': cutoff, 'K': magnetic, 'UE': electric, 'Delta': delta,
                   'hopping': hopping, 'improved': improved, 'phase': phase,
                   'background': model.background, 'perturbation': 'lambda Z_code_edge_0',
                   'perturbed_edge': model.edges[0]},
        'domain': 'actual full raw Gauss-physical Pauli/rotor square; no code-sector projection',
        'physical_dimension': model.dimension,
        'gauss': {'Z_max_change': max(z_gauss_changes),
                  'undressed_X_min_violation': min(bare_x_gauss_changes),
                  'undressed_X_max_violation': max(bare_x_gauss_changes),
                  'term_constraint_violations': model.constraint_violations},
        'unperturbed': {'E0': E0, 'complement_ground_gap_gamma': gamma,
                        'code_phase_lower_bound': delta - 2 * magnetic,
                        'isolated_full_ground_gap_g0': g0,
                        'whole_band_separation_not_assumed': float(eq[0] - ep[-1]),
                        'plus_spectrum': ep.tolist(), 'minus_spectrum': eq.tolist(),
                        'retained_low_count': low_count, 'retained_low_top': low_top,
                        'low_cluster_gap': cluster_gap, 'fixed_low_threshold': threshold},
        'residuals': residuals,
        'weak_perturbations': rows,
        'small_coupling': {'weight_logarithmic_slopes': slopes,
                           'derived_leakage_coefficient': leakage_coefficient,
                           'derived_energy_lowering_coefficient': shift_coefficient,
                           'smallest_amplitude_relative_coefficient_error': asymptotic_error},
        'strong_perturbation': strong,
        'acceptance': acceptance,
        'scope': 'Finite-spectrum evidence for the stated theorem; not infinite-volume stability or photon-phase selection',
    }
    if not acceptance:
        raise AssertionError(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff', type=int, default=1)
    parser.add_argument('--magnetic', type=float, default=0.7)
    parser.add_argument('--electric', type=float, default=0.8)
    parser.add_argument('--delta', type=float, default=2.0)
    parser.add_argument('--hopping', type=float, default=0.6)
    parser.add_argument('--improved', type=float, default=0.19)
    parser.add_argument('--phase', type=float, default=0.23)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run(args.cutoff, args.magnetic, args.electric, args.delta,
                 args.hopping, args.improved, args.phase)
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()

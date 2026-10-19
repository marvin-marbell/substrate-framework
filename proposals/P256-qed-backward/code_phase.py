#!/usr/bin/env python3
"""Actual Pauli-square + noncyclic quantum rotors, restricted only by Gauss.

No CAR constructor, unconstrained matter spectrum, or projected-away mutation.
Direct execution emits evidence as JSON; failures raise rather than being hidden.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass

import numpy as np

from pauli_compton import EdgeWords, PauliWord


@dataclass(frozen=True)
class Transfer:
    path: tuple[int, ...]
    coefficient: complex


class CoupledSquare:
    """Enumerate physical raw states before allocating any Hamiltonian matrix."""

    def __init__(self, cutoff=1, magnetic=0.7, electric=0.8, delta=2.0,
                 hopping=0.6, improved=0.19, phase=0.23):
        if cutoff < 1 or magnetic < 0 or electric < 0 or delta <= 2 * magnetic:
            raise ValueError('Require cutoff>=1, K,UE>=0 and Delta>2K')
        self.cutoff = cutoff
        self.K, self.UE, self.delta = magnetic, electric, delta
        self.background = (1, 0, 1, 0)  # integer, fixed, sum b=N=2
        self.words = EdgeWords(4, ((0, 1), (1, 2), (2, 3), (0, 3)))
        self.edges = self.words.edges
        self.cycle = (0, 1, 2, 3)
        self.S = self.words.cycle(self.cycle)
        self.eta = (-1, 1, 1, 1)
        self.Q = PauliWord(z=1)  # physical code-qubit Z, not rotor R
        self.transfers = tuple(
            Transfer(edge, hopping * np.exp(1j * phase * (index + 1)))
            for index, edge in enumerate(self.edges)
        ) + (
            Transfer((0, 1, 2), improved * np.exp(1j * 1.7 * phase)),
            Transfer((0, 3, 2), improved * np.exp(-1j * 0.9 * phase)),
            Transfer((0, 1, 2, 3), improved * 0.47 * np.exp(1j * phase)),
        )
        self.onsite = (0.13, -0.07, 0.04, -0.11)
        self.raw = []
        for q in range(1 << len(self.edges)):
            occupied = self.words.occupation(q)
            if len(occupied) != sum(self.background):
                continue
            for fields in itertools.product(range(-cutoff, cutoff + 1), repeat=4):
                if self.gauss(q, fields) == (0, 0, 0, 0):
                    self.raw.append((q, fields))
        self.index = {state: index for index, state in enumerate(self.raw)}
        if not self.raw:
            raise ValueError('No physical states in selected electric cutoff')
        self.dimension = len(self.raw)
        self.constraint_violations = 0

    def gauss(self, q, fields):
        divergence = [0] * 4
        for (source, target), field in zip(self.edges, fields):
            divergence[source] += field
            divergence[target] -= field
        return tuple(divergence[v] - (int(self.words.occupied(q, v))
                                     - self.background[v]) for v in range(4))

    def shifts(self, path, closing=False):
        shifts = [0] * 4
        pairs = zip(path, (*path[1:], path[0])) if closing else zip(path[:-1], path[1:])
        for source, target in pairs:
            edge = self.edges.index((min(source, target), max(source, target)))
            shifts[edge] += -1 if source < target else 1
        return tuple(shifts)

    def shift(self, fields, shifts):
        result = tuple(field + step for field, step in zip(fields, shifts))
        return result if all(abs(field) <= self.cutoff for field in result) else None

    def insert(self, matrix, column, q, fields, value):
        row = self.index.get((q, fields))
        if row is None:
            self.constraint_violations += 1
            raise AssertionError('Microscopic term escaped physical Gauss basis')
        matrix[row, column] += value

    def pauli_matrix(self, word):
        result = np.zeros((self.dimension, self.dimension), dtype=complex)
        for column, (q, fields) in enumerate(self.raw):
            target, value = word.action(q)
            self.insert(result, column, target, fields, value)
        return result

    def transfer_matrix(self, path, bare=False):
        """-i A_path n_source(1-n_target) times directed U_lower shifts."""
        source, target = path[0], path[-1]
        if bare:
            if len(path) != 2:
                raise ValueError('Bare-X adversary is a single microscopic edge')
            edge = self.edges.index((min(source, target), max(source, target)))
            word = PauliWord(x=1 << edge).times_i(0 if source < target else 2)
        else:
            word = self.words.path_A(path)
        shifts = self.shifts(path)
        result = np.zeros((self.dimension, self.dimension), dtype=complex)
        for column, (q, fields) in enumerate(self.raw):
            if not self.words.occupied(q, source) or self.words.occupied(q, target):
                continue
            shifted = self.shift(fields, shifts)
            if shifted is not None:
                new_q, amplitude = word.action(q)
                self.insert(result, column, new_q, shifted, -1j * amplitude)
        return result

    def operators(self, mutate=False):
        S = self.pauli_matrix(self.S)
        electric_diagonal = [self.UE / 2 * sum(e * e for e in fields)
                             for _, fields in self.raw]
        onsite_diagonal = [sum(self.onsite[v] for v in self.words.occupation(q))
                           for q, _ in self.raw]
        matter = np.diag(onsite_diagonal).astype(complex)
        for term in self.transfers:
            bare = mutate and term.path == (0, 3)
            transfer = self.transfer_matrix(term.path, bare)
            matter += term.coefficient * transfer + term.coefficient.conjugate() * transfer.conj().T
        W = np.zeros_like(matter)
        shifts = self.shifts(self.cycle, closing=True)
        for column, (q, fields) in enumerate(self.raw):
            shifted = self.shift(fields, shifts)
            if shifted is not None:
                self.insert(W, column, q, shifted, 1)
        cosine = (W + W.conj().T) / 2
        H = (matter + np.diag(electric_diagonal) - self.K * cosine
             + self.delta * (np.eye(self.dimension) - S) / 2)
        return H, S, cosine, matter

    def sector_basis(self, sign):
        """Actual joint projector columns, with independently canonicalized phases."""
        columns, labels = [], []
        seen = set()
        for q, fields in self.raw:
            label = (self.words.occupation(q), fields)
            if label in seen:
                continue
            seen.add(label)
            target, amplitude = self.S.action(q)
            column = np.zeros(self.dimension, dtype=complex)
            column[self.index[q, fields]] += 0.5
            column[self.index[target, fields]] += 0.5 * sign * amplitude
            norm = np.linalg.norm(column)
            if norm == 0:
                continue
            columns.append(column / norm)
            labels.append(label)
        return np.column_stack(columns), labels

    def occupation_state(self, occupation, sign):
        """Pauli-only occupation/S projector, no fermion basis phase convention."""
        q = next(q for q in range(16) if self.words.occupation(q) == occupation)
        target, amplitude = self.S.action(q)
        state = np.zeros(16, dtype=complex)
        state[q] += 0.5
        state[target] += 0.5 * sign * amplitude
        return state / np.linalg.norm(state)

    def phase_intertwiner(self):
        """Derive f_n=<n,-|Q_eta|n,+>, not an assumed identity between bases."""
        phases = {}
        for occupation in itertools.combinations(range(4), 2):
            plus = self.occupation_state(occupation, 1)
            minus = self.occupation_state(occupation, -1)
            image = np.zeros(16, dtype=complex)
            for q, coefficient in enumerate(plus):
                target, amplitude = self.Q.action(q)
                image[target] += amplitude * coefficient
            phase = np.vdot(minus, image)
            if abs(abs(phase) - 1) > 1e-12:
                raise AssertionError('Q_eta failed to intertwine actual code eigenspaces')
            phases[occupation] = phase
        return phases

    def all_path_residual(self, phases):
        """Check oriented simple paths, including both competing two-link routes."""
        worst = 0.0
        count = 0
        for length in (2, 3, 4):
            for path in itertools.permutations(range(4), length):
                if any(tuple(sorted(pair)) not in self.edges for pair in zip(path[:-1], path[1:])):
                    continue
                word = self.words.path_A(path)
                eta_path = 1
                for a, b in zip(path[:-1], path[1:]):
                    eta_path *= self.eta[self.edges.index(tuple(sorted((a, b))))]
                for occupied in phases:
                    if path[0] not in occupied or path[-1] in occupied:
                        continue
                    target_occ = tuple(sorted((set(occupied) - {path[0]}) | {path[-1]}))
                    amplitudes = []
                    for sign in (1, -1):
                        state = self.occupation_state(occupied, sign)
                        image = np.zeros(16, dtype=complex)
                        for q, coefficient in enumerate(state):
                            target, amplitude = word.action(q)
                            image[target] += -1j * amplitude * coefficient
                        amplitudes.append(np.vdot(self.occupation_state(target_occ, sign), image))
                    transformed = phases[target_occ].conjugate() * amplitudes[1] * phases[occupied]
                    worst = max(worst, abs(transformed - eta_path * amplitudes[0]))
                    count += 1
        return float(worst), count


def norm(matrix):
    return float(np.linalg.norm(matrix))


def rotor_report(cutoff):
    """Noncyclic lowering: no wraparound electric matrix element."""
    values = np.arange(-cutoff, cutoff + 1)
    E = np.diag(values)
    U = np.diag(np.ones(2 * cutoff), 1)
    R = np.diag([(-1.0) ** int(e) for e in values])
    return {
        'electric_values': values.tolist(),
        'lowering_commutator_residual': norm(E @ U - U @ E + U),
        'rotor_pi_conjugation_residual': norm(R @ U @ R + U),
        'cyclic_wrap_element': float(U[-1, 0]),
        'unitarity_defect_expected': norm(U.conj().T @ U - np.eye(len(values))),
    }


def run(cutoff=1, magnetic=0.7, electric=0.8, delta=2.0,
        hopping=0.6, improved=0.19, phase=0.23):
    model = CoupledSquare(cutoff, magnetic, electric, delta, hopping, improved, phase)
    H, S, C, matter = model.operators()
    plus, plus_labels = model.sector_basis(1)
    minus, minus_labels = model.sector_basis(-1)
    if plus_labels != minus_labels:
        raise AssertionError('Physical sectors have different occupation/rotor domains')
    phases = model.phase_intertwiner()
    # T=Q_eta R_eta; the phase is occupation dependent AND electric-field dependent.
    factors = np.array([phases[occupation] * (-1.0) ** fields[0]
                        for occupation, fields in plus_labels])
    T = minus * factors[np.newaxis, :]
    Hplus = plus.conj().T @ H @ plus
    Hminus = minus.conj().T @ H @ minus
    Cplus = plus.conj().T @ C @ plus
    transformed = T.conj().T @ H @ T
    difference = transformed - Hplus - delta * np.eye(Hplus.shape[0])
    identity_residual = norm(difference - 2 * magnetic * Cplus)
    exact_QR = model.pauli_matrix(model.Q) @ np.diag(
        [(-1.0) ** fields[0] for _, fields in model.raw])
    projector_plus = (np.eye(model.dimension) + S) / 2
    projector_minus = (np.eye(model.dimension) - S) / 2
    path_residual, path_count = model.all_path_residual(phases)
    full_spectrum = np.linalg.eigvalsh(H)
    energies_plus = np.linalg.eigvalsh(Hplus)
    energies_minus = np.linalg.eigvalsh(Hminus)
    mutated, mutated_S, _, _ = model.operators(mutate=True)
    leakage_block = minus.conj().T @ mutated @ plus
    commutator = norm(mutated_S @ mutated - mutated @ mutated_S)
    lower = delta - 2 * magnetic
    residuals = {
        'hamiltonian_hermiticity': norm(H - H.conj().T),
        'commuting_sector': norm(S @ H - H @ S),
        'stabilizer_square': norm(S @ S - np.eye(model.dimension)),
        'plus_projector_reconstructed': norm(plus @ plus.conj().T - projector_plus),
        'minus_projector_reconstructed': norm(minus @ minus.conj().T - projector_minus),
        'sector_basis_orthogonality': norm(plus.conj().T @ minus),
        'derived_occupation_rotor_intertwiner': norm(T - exact_QR @ plus),
        'all_oriented_matter_paths': path_residual,
        'coupled_transformed_identity': identity_residual,
        'matter_eta_rotor_cancellation': norm(T.conj().T @ matter @ T - plus.conj().T @ matter @ plus),
        'full_spectrum_sector_union': norm(full_spectrum - np.sort(np.r_[energies_plus, energies_minus])),
    }
    min_operator_margin = float(np.linalg.eigvalsh(transformed - Hplus - lower * np.eye(len(Hplus)))[0])
    spectrum_margins = energies_minus - energies_plus - lower
    accepted = (max(residuals.values()) < 1e-10
                and min_operator_margin >= -1e-10
                and float(spectrum_margins.min()) >= -1e-10
                and commutator > 1e-8 and norm(leakage_block) > 1e-8
                and model.constraint_violations == 0)
    report = {
        'inputs': {'cutoff': cutoff, 'K': magnetic, 'UE': electric, 'Delta': delta,
                   'hopping': hopping, 'multilink_hopping': improved, 'phase': phase,
                   'background': model.background, 'N': 2, 'fixed_parity': 1,
                   'edges': model.edges, 'eta': model.eta, 'onsite': model.onsite},
        'domain': 'open single square; fixed N=2; raw Pauli qubits and four noncyclic rotors; physical Gauss only',
        'dimensions': {'unconstrained_product_not_allocated': 16 * (2 * cutoff + 1) ** 4,
                       'physical_raw': model.dimension, 'each_actual_sector': plus.shape[1]},
        'rotors': rotor_report(cutoff),
        'gauss': {'physical_raw_max_residual': max(max(abs(x) for x in model.gauss(q, fields)) for q, fields in model.raw),
                  'term_constraint_violations': model.constraint_violations,
                  'QR_changes_occupation_or_electric_field': False},
        'occupation_phase_intertwiner': [
            {'occupation': occupied, 'phase': [float(value.real), float(value.imag)]}
            for occupied, value in phases.items()],
        'oriented_simple_path_amplitude_checks': path_count,
        'residuals': residuals,
        'actual_sector_spectra': {'plus': energies_plus.tolist(), 'minus': energies_minus.tolist()},
        'bound': {'Delta_minus_2K': lower, 'operator_margin_min': min_operator_margin,
                  'ordered_spectrum_margin_min': float(spectrum_margins.min()),
                  'undesired_minus_desired_ground': float(energies_minus[0] - energies_plus[0]),
                  'not_claimed_full_band_separation': float(energies_minus[0] - energies_plus[-1])},
        'bare_X_microscopic_mutation': {'edge': [0, 3], 'commutator_norm': commutator,
                                      'actual_cross_sector_block_norm': norm(leakage_block),
                                      'sector_proof_rejected': commutator > 1e-8},
        'acceptance': accepted,
        'input_debt': ['selected primitive Pauli law and quantum Hilbert/dynamics postulates',
                       'K, UE, Delta, hopping, onsite and integer background are inputs',
                       'no photon Coulomb phase, continuum limit, thermal preparation or dressed-charge matching proof'],
    }
    if not accepted:
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
    args = parser.parse_args()
    print(json.dumps(run(args.cutoff, args.magnetic, args.electric, args.delta,
                         args.hopping, args.improved, args.phase), indent=2))


if __name__ == '__main__':
    main()

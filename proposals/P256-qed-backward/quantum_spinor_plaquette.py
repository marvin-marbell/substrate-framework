#!/usr/bin/env python3
"""Sparse, fully quantum planar four-orbital Pauli/code/compact-rotor join.

No canonical CAR matrices construct this Hamiltonian. Run for finite-model
JSON evidence; the accompanying derivation distinguishes proof from measurement.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import itertools
import json
import os
from pathlib import Path

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, eigsh
from scipy.linalg import eigh_tridiagonal

from pauli_compton import EdgeWords
from rotation_band import RotationBand, TRANSITIONS


CELLS = ((0, 0), (1, 0), (1, 1), (0, 1))
ROOTS = (0, 4, 8, 12)
CYCLE = (0, 4, 8, 12)
CYCLE_EDGES = ((0, 4), (4, 8), (8, 12), (12, 0))
MARKER = 16


def occupied(mask):
    return tuple(i for i in range(16) if mask >> i & 1)


def wedge_transfer(mask, source, target):
    """Comparator/intertwiner sign ONLY, never a microscopic matrix builder."""
    low, high = sorted((source, target))
    return -1 if sum(mask >> i & 1 for i in range(low + 1, high)) % 2 else 1


def max_abs(matrix):
    return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0


class PauliCode:
    """Two raw edge-bit words per occupation, projected by the ACTUAL S word."""

    def __init__(self):
        internal = [(4 * c, 4 * c + j) for c in range(4) for j in (1, 2, 3)]
        edges = internal + list(CYCLE_EDGES) + [(0, MARKER)]
        self.words = EdgeWords(17, edges)
        self.S = self.words.cycle(CYCLE)
        self.cut_edge = (0, 12)
        tree_edges = [e for e in self.words.edges if e != self.cut_edge]
        self.tree = EdgeWords(17, tree_edges)
        self.tree_to_full = [self.words.edges.index(e) for e in self.tree.edges]
        self.cache = {}
        self.phases = {}
        self.law_residual = 0.0
        self.Q_edge = self.words.edges.index(self.cut_edge)
        # Derived from the existing two-label matrix units, not a new Dirac input.
        self.local_T = np.kron(np.eye(2), 1j * TRANSITIONS[1])
        self.local_perm = np.argmax(abs(self.local_T), axis=0)
        self.local_phase = self.local_T[self.local_perm, np.arange(4)]
        band = RotationBand(a=1.0)
        U = self.local_T
        identities = [U @ U.conjugate() + np.eye(4),
                      U @ band.beta.conjugate() @ U.conj().T - band.beta]
        identities.extend(U @ a.conjugate() @ U.conj().T + a for a in band.alpha[:2])
        if max(np.linalg.norm(a) for a in identities) > 1e-12:
            raise AssertionError('Scalar orbital law does not admit the derived local T')
        if self.S.x == 0 or not self.S.hermitian():
            raise AssertionError('Cycle projector is not a two-word Hermitian projector')
        if self.S @ self.S != type(self.S)():
            raise AssertionError('S squared is not identity')
        for word in self.words.A.values():
            if not word.commutes(self.S):
                raise AssertionError('Primitive matter transfer mixes code sectors')

    def state(self, mask, sign=1):
        key = mask, sign
        if key not in self.cache:
            occ = occupied(mask)
            full = occ + ((MARKER,) if len(occ) % 2 else ())
            qt = self.tree.tree_basis_state(full)
            q = sum(((qt >> k) & 1) << e for k, e in enumerate(self.tree_to_full))
            other, _ = self.S.action(q)
            q = min(q, other)
            other, amplitude = self.S.action(q)
            if self.words.occupation(q) != full or other == q:
                raise AssertionError('Raw occupation/code projector failed')
            self.cache[key] = ((q, 1 / np.sqrt(2)),
                               (other, sign * amplitude / np.sqrt(2)))
        return self.cache[key]

    def element(self, mask, new_mask, word, sign=1):
        target = dict(self.state(new_mask, sign))
        value = 0j
        for q, coefficient in self.state(mask, sign):
            out, phase = word.action(q)
            value += np.conjugate(target.get(out, 0)) * phase * coefficient
        if abs(abs(value) - 1) > 1e-12:
            raise AssertionError('Actual Pauli transition escaped normalized code state')
        return value

    def transfer(self, mask, source, target, path=None, sign=1):
        new = mask ^ (1 << source) ^ (1 << target)
        path = self.words.path(source, target) if path is None else path
        word = self.words.path_A(path).times_i(-1)
        return new, self.element(mask, new, word, sign)

    def intertwiner(self, number, sign=1):
        """Derive occupation phases by actual tree-edge Pauli transitions."""
        key = number, sign
        if key in self.phases:
            return self.phases[key]
        first = (1 << number) - 1
        phases = {first: 1 + 0j}
        queue = deque([first])
        tree_edges = [e for e in self.tree.edges if MARKER not in e]
        while queue:
            mask = queue.popleft()
            for a, b in tree_edges:
                if ((mask >> a) ^ (mask >> b)) & 1:
                    source, target = (a, b) if mask >> a & 1 else (b, a)
                    new, amplitude = self.transfer(mask, source, target, (source, target), sign)
                    phase = amplitude * phases[mask] / wedge_transfer(mask, source, target)
                    if new not in phases:
                        phases[new] = phase
                        queue.append(new)
                    else:
                        self.law_residual = max(self.law_residual, abs(phase - phases[new]))
        if len(phases) != len(list(itertools.combinations(range(16), number))):
            raise AssertionError('Pauli phase propagation did not reach every occupation')
        # The omitted edge is the only real Z2 twist in S=-; S=+ has no twist.
        for mask in phases:
            a, b = self.cut_edge
            if ((mask >> a) ^ (mask >> b)) & 1:
                source, target = (a, b) if mask >> a & 1 else (b, a)
                new, amplitude = self.transfer(mask, source, target, (source, target), sign)
                expected = sign * wedge_transfer(mask, source, target) * phases[new] / phases[mask]
                self.law_residual = max(self.law_residual, abs(amplitude - expected))
        if self.law_residual > 2e-10:
            raise AssertionError('Actual primitive path signs violate matrix-unit code law')
        self.phases[key] = phases
        return phases

    def time_reversal(self, number, sign=1):
        phases = self.intertwiner(number, sign)
        permutation, coefficients = {}, {}
        for mask, phase in phases.items():
            labels = [4 * (i // 4) + int(self.local_perm[i % 4]) for i in occupied(mask)]
            inversions = sum(labels[i] > labels[j] for i in range(number)
                             for j in range(i + 1, number))
            wedge_phase = (-1) ** inversions * np.prod(
                [self.local_phase[i % 4] for i in occupied(mask)])
            target = sum(1 << i for i in labels)
            permutation[mask] = target
            coefficients[mask] = phases[target] * wedge_phase / np.conjugate(phase)
        return permutation, coefficients


class SpinorPlaquette:
    def __init__(self, code, number, cutoff, magnetic=0.25, electric=0.6,
                 delta=1.2, m=0.8, r=0.04, v=0.08, sign=1):
        if number not in (7, 8, 9) or cutoff < 1 or sign not in (-1, 1):
            raise ValueError('Require N=7/8/9, cutoff>=1, S=+/-')
        if magnetic <= 0 or electric <= 0 or delta <= 2 * magnetic:
            raise ValueError('Require K,UE>0 and Delta>2K')
        self.code, self.N, self.cutoff, self.sign = code, number, cutoff, sign
        self.K, self.UE, self.delta = magnetic, electric, delta
        self.boundary = number - 8
        self.raw = []
        for occ in itertools.combinations(range(16), number):
            mask = sum(1 << i for i in occ)
            counts = [sum(mask >> (4 * c + j) & 1 for j in range(4)) for c in range(4)]
            charges = [counts[c] - 2 - (self.boundary if c == 0 else 0) for c in range(4)]
            offsets = (0, charges[1], charges[1] + charges[2], -charges[0])
            low = max(-cutoff - q for q in offsets)
            high = min(cutoff - q for q in offsets)
            for loop in range(low, high + 1):
                fields = tuple(loop + q for q in offsets)
                if self.gauss(mask, fields) != (0, 0, 0, 0):
                    raise AssertionError('Occupation/one-loop Gauss solver failed')
                self.raw.append((mask, fields))
        self.index = {state: i for i, state in enumerate(self.raw)}
        self.dimension = len(self.raw)
        if not self.dimension:
            raise ValueError('Electric cutoff contains no Gauss states')
        band = RotationBand(a=1.0, m=m, r=r, v=v)
        onsite3, links3 = band.couplings()
        # Honest planar reduction: remove the absent z-link Wilson onsite r beta.
        self.onsite = onsite3 - r * band.beta
        self.terms = []
        for c in range(4):
            for target, source in zip(*np.nonzero(abs(self.onsite) > 1e-14)):
                if source != target:
                    self.add_term(4 * c + source, 4 * c + target,
                                  self.onsite[target, source], None)
        for source_cell, target_cell, axis in ((0, 1, 0), (1, 2, 1), (3, 2, 0), (0, 3, 1)):
            coupling = links3[axis][1].conj().T
            for target, source in zip(*np.nonzero(abs(coupling) > 1e-14)):
                a, b = 4 * source_cell + source, 4 * target_cell + target
                path = tuple(dict.fromkeys((a, ROOTS[source_cell], ROOTS[target_cell], b)))
                self.add_term(a, b, coupling[target, source], path)
                self.add_term(b, a, coupling[target, source].conjugate(), path[::-1])
        self.H, self.W = self.operators()
        self.B = (self.W - self.W.conjugate().T) / (2j)
        self.Tperm, self.Tphase = self.time_reversal()
        self.hermiticity = max_abs(self.H - self.H.conjugate().T)
        square = self.Tphase[self.Tperm] * self.Tphase.conjugate()
        self.Tsquare = float(np.max(abs(square - (-1) ** number)))
        # T = U K, with one nonzero per column. Verify the complete sparse operator.
        U = sparse.coo_matrix((self.Tphase, (self.Tperm, np.arange(self.dimension))),
                              shape=self.H.shape).tocsr()
        self.Tsymmetry = max_abs(U @ self.H.conjugate() @ U.conjugate().T - self.H)
        self.B_Todd = max_abs(U @ self.B.conjugate() @ U.conjugate().T + self.B)
        if max(self.hermiticity, self.Tsquare, self.Tsymmetry, self.B_Todd) > 2e-10:
            raise AssertionError('Actual Hamiltonian/code time-reversal identity failed')

    def gauss(self, mask, fields, boundary=None, marker_charge=0):
        boundary = self.boundary if boundary is None else boundary
        return tuple(fields[c] - fields[(c - 1) % 4]
                     + (boundary if c == 0 else 0)
                     - (sum(mask >> (4 * c + j) & 1 for j in range(4)) - 2)
                     - (marker_charge * (self.N % 2) if c == 0 else 0)
                     for c in range(4))

    def add_term(self, source, target, coefficient, path):
        source, target = int(source), int(target)
        path = self.code.words.path(source, target) if path is None else tuple(int(v) for v in path)
        shifts = [0] * 4
        for a, b in zip(path[:-1], path[1:]):
            if (a, b) in CYCLE_EDGES:
                shifts[CYCLE_EDGES.index((a, b))] -= 1
            elif (b, a) in CYCLE_EDGES:
                shifts[CYCLE_EDGES.index((b, a))] += 1
        word = self.code.words.path_A(path).times_i(-1)
        if not word.commutes(self.code.S):
            raise AssertionError('Current/transfer can couple discarded code sector')
        self.terms.append((source, target, coefficient, word, tuple(shifts)))

    def operators(self):
        rows, cols, values = [], [], []
        wrows, wcols = [], []
        by_mask = {}
        for column, (mask, fields) in enumerate(self.raw):
            by_mask.setdefault(mask, []).append((column, fields))
            diagonal = self.UE / 2 * (sum(e * e for e in fields) + self.boundary ** 2)
            diagonal += self.delta * (1 - self.sign) / 2
            diagonal += sum(self.onsite[i % 4, i % 4].real for i in occupied(mask))
            rows.append(column); cols.append(column); values.append(diagonal)
            shifted = tuple(e + 1 for e in fields)
            row = self.index.get((mask, shifted))
            if row is not None:
                wrows.append(row); wcols.append(column)
        for mask, columns in by_mask.items():
            for source, target, coefficient, word, shifts in self.terms:
                if not mask >> source & 1 or mask >> target & 1:
                    continue
                new = mask ^ (1 << source) ^ (1 << target)
                amplitude = self.code.element(mask, new, word, self.sign)
                for column, fields in columns:
                    shifted = tuple(e + q for e, q in zip(fields, shifts))
                    if any(abs(e) > self.cutoff for e in shifted):
                        continue
                    row = self.index.get((new, shifted))
                    if row is None:
                        raise AssertionError('Pauli/rotor microscopic hop escaped full Gauss')
                    rows.append(row); cols.append(column); values.append(coefficient * amplitude)
        H = sparse.coo_matrix((values, (rows, cols)), shape=(self.dimension,) * 2).tocsr()
        W = sparse.coo_matrix((np.ones(len(wrows)), (wrows, wcols)), shape=H.shape).tocsr()
        H -= self.K / 2 * (W + W.conjugate().T)
        return H, W

    def time_reversal(self):
        perm, phase = self.code.time_reversal(self.N, self.sign)
        return (np.array([self.index[perm[mask], fields] for mask, fields in self.raw]),
                np.array([phase[mask] for mask, _ in self.raw]))

    def apply_T(self, vector):
        result = np.empty_like(vector)
        result[self.Tperm] = self.Tphase * vector.conjugate()
        return result

    def adverse_controls(self):
        mask, fields = self.raw[0]
        wrong_boundary = any(self.gauss(mask, fields, boundary=self.boundary + 1))
        # Marker=0 in N8, so test its charge in an ACTUAL odd-N sector only.
        wrong_marker = any(self.gauss(mask, fields, marker_charge=1)) if self.N % 2 else None
        rotor_rejected = False
        rotor_example = None
        for actual_mask, actual_fields in self.raw:
            for source, target, _, _, shifts in self.terms:
                if not any(shifts) or not actual_mask >> source & 1 or actual_mask >> target & 1:
                    continue
                new = actual_mask ^ (1 << source) ^ (1 << target)
                correct = tuple(e + q for e, q in zip(actual_fields, shifts))
                if (new, correct) not in self.index:
                    continue
                wrong = tuple(e - q for e, q in zip(actual_fields, shifts))
                violations = self.gauss(new, wrong)
                rotor_rejected = any(violations)
                rotor_example = {'source': source, 'target': target,
                                 'initial_fields': actual_fields, 'wrong_fields': wrong,
                                 'wrong_Gauss_residual': violations}
                break
            if rotor_example is not None:
                break
        # Reverse one primitive A sign in the oriented product: S becomes -S.
        wrong_cycle = self.code.S.times_i(2)
        code_sign_rejected = all(abs(sum(np.conjugate(dict(self.code.state(mask)).get(wrong_cycle.action(q)[0], 0))
                                               * wrong_cycle.action(q)[1] * coefficient
                                               for q, coefficient in self.code.state(mask)) + 1) < 1e-12
                                 for mask in (self.raw[0][0], self.raw[-1][0]))
        phases = self.code.intertwiner(self.N)
        a, b = self.code.cut_edge
        path_rejected = False
        path_example = None
        for actual_mask, _ in self.raw:
            if not ((actual_mask >> a) ^ (actual_mask >> b)) & 1:
                continue
            source, target = (a, b) if actual_mask >> a & 1 else (b, a)
            new, actual = self.code.transfer(actual_mask, source, target, (source, target))
            expected = wedge_transfer(actual_mask, source, target) * phases[new] / phases[actual_mask]
            correct_residual = abs(actual - expected)
            wrong_residual = abs(-actual - expected)
            path_rejected = correct_residual < 1e-10 and wrong_residual > 1.9
            path_example = {'source': source, 'target': target,
                            'correct_code_path_residual': float(correct_residual),
                            'reversed_path_sign_residual': float(wrong_residual)}
            break
        result = {'wrong_boundary_flux_rejected': bool(wrong_boundary),
                  'wrong_marker_EM_charge_rejected': bool(wrong_marker) if wrong_marker is not None else None,
                  'wrong_rotor_transport_sign_rejected': bool(rotor_rejected),
                  'wrong_primitive_cycle_sign_rejected': bool(code_sign_rejected),
                  'wrong_code_path_matrix_element_rejected': bool(path_rejected)}
        if not all(v for v in result.values() if v is not None):
            raise AssertionError('Adverse physical control failed to reject wrong construction')
        return {**result, 'actual_wrong_rotor_example': rotor_example,
                'actual_wrong_code_path_example': path_example}


def solve_low(model, tol=2e-9, norm_bound=None):
    rng = np.random.default_rng(222 + model.N)
    start = rng.normal(size=model.dimension) + 1j * rng.normal(size=model.dimension)
    energies, vectors = eigsh(model.H, k=1, which='SA', tol=tol, v0=start,
                              ncv=28, maxiter=4000)
    energy = float(energies[0])
    vector = vectors[:, 0]
    if model.N % 2:
        partner = model.apply_T(vector)
        overlap = float(abs(np.vdot(vector, partner)))
        partner_residual = float(np.linalg.norm(model.H @ partner - energy * partner))
        if overlap > 1e-9 or partner_residual > 2e-6:
            raise AssertionError('Derived T partner is not an orthogonal ground eigenstate')
        ground = np.column_stack((vector, partner))
    else:
        overlap, partner_residual = None, None
        ground = vectors
    actions = model.H @ ground
    ground_energies = np.einsum('ij,ij->j', ground.conj(), actions).real
    residuals = np.linalg.norm(actions - ground * ground_energies, axis=0)
    if np.max(residuals) > 2e-6:
        raise AssertionError('Low-energy sparse eigenpair residual failed')

    # A single-vector Krylov solve need not return all vectors of a degenerate
    # eigenvalue. Use the physical T partner, then probe its ENTIRE complement.
    # Gershgorin for CSR; tensor operators provide a certified spectral norm.
    if norm_bound is None:
        nonempty = model.H.indptr[1:] > model.H.indptr[:-1]
        row_sums = np.add.reduceat(np.abs(model.H.data), model.H.indptr[:-1][nonempty])
        norm_bound = float(row_sums.max())
    if not np.isfinite(norm_bound) or norm_bound <= 0:
        raise ValueError('A finite positive Hamiltonian norm bound is required')
    penalty = 2 * norm_bound + 1.
    def deflated_action(x):
        return model.H @ x + penalty * ground @ (ground.conj().T @ x)
    deflated = LinearOperator(model.H.shape, matvec=deflated_action, dtype=complex)
    start = rng.normal(size=model.dimension) + 1j * rng.normal(size=model.dimension)
    start -= ground @ (ground.conj().T @ start)
    start /= np.linalg.norm(start)
    next_values, next_vectors = eigsh(deflated, k=1, which='SA', tol=tol, v0=start,
                                      ncv=28, maxiter=4000)
    next_vector = next_vectors[:, 0]
    next_action = model.H @ next_vector
    next_energy = float(np.vdot(next_vector, next_action).real)
    complement_leakage = float(np.linalg.norm(ground.conj().T @ next_vector))
    next_residual = float(np.linalg.norm(next_action - next_energy * next_vector))
    if complement_leakage > 2e-6 or next_residual > 2e-6:
        raise AssertionError('Deflated complement eigenpair did not resolve the physical gap')
    gap = next_energy - energy
    if gap <= 1e-7:
        raise AssertionError('Additional ground states remain outside the proposed multiplet')
    observed = np.r_[ground_energies, next_energy]
    return observed, ground, {
        'resolved_ground_energies': ground_energies.tolist(),
        'ground_eigen_residuals': residuals.tolist(),
        'ground_multiplicity': ground.shape[1], 'gap_above_ground_multiplet': gap,
        'first_complement_energy': next_energy,
        'first_complement_eigen_residual': next_residual,
        'complement_ground_leakage': complement_leakage,
        'deflation_energy_penalty': penalty,
        'deflated_ritz_energy': float(next_values[0]),
        'T_partner_self_overlap': overlap, 'T_partner_eigen_residual': partner_residual}


def response(model, energy, ground, steps=36):
    """Positive Lanczos measure + converged projected static resolvent, no dense H."""
    vector = ground[:, 0]
    source = model.B @ vector
    elastic = ground.conj().T @ source
    seed = source - ground @ elastic
    norm2 = float(np.vdot(seed, seed).real)
    expectation = np.vdot(vector, source)
    variance = float(np.vdot(source, source).real - abs(expectation) ** 2)
    zero_connected = float(np.vdot(elastic, elastic).real - abs(expectation) ** 2)
    if norm2 <= 1e-10:
        raise AssertionError('No constructive inelastic compact photon weight')
    basis, diagonal, off = [], [], []
    q = seed / np.sqrt(norm2)
    previous = np.zeros_like(q)
    beta = 0.0
    for j in range(steps):
        basis.append(q)
        z = model.H @ q - energy * q - beta * previous
        alpha = float(np.vdot(q, z).real)
        z -= alpha * q
        # Explicit two-pass full reorthogonalization controls ghost spectral poles.
        Q = np.column_stack(basis)
        for _ in range(2):
            z -= Q @ (Q.conj().T @ z)
            z -= ground @ (ground.conj().T @ z)
        diagonal.append(alpha)
        new_beta = float(np.linalg.norm(z))
        if j == steps - 1 or new_beta < 1e-12:
            break
        off.append(new_beta)
        previous, q, beta = q, z / new_beta, new_beta
    poles, eigenvectors = eigh_tridiagonal(diagonal, off)
    weights = norm2 * abs(eigenvectors[0]) ** 2
    if poles.min() <= 0 or weights.min() < 0:
        raise AssertionError('Lanczos positive spectral measure failed')
    moment = float(np.vdot(seed, model.H @ seed - energy * seed).real)
    # Direct half double commutator: <BHB> - Re<B² H>.
    half_double = float(np.vdot(source, model.H @ source).real
                        - np.vdot(model.B @ source, model.H @ vector).real)
    action = lambda x: model.H @ x - energy * x + ground @ (ground.conj().T @ x)
    A = LinearOperator(model.H.shape, matvec=action, dtype=complex)
    solution, status = cg(A, seed, rtol=2e-9, atol=1e-12, maxiter=2500)
    resolvent_residual = float(np.linalg.norm(action(solution) - seed))
    susceptibility = float(2 * np.vdot(seed, solution).real)
    sum_residual = abs(variance - zero_connected - weights.sum())
    moment_residual = abs(np.dot(poles, weights) - moment)
    commutator_residual = abs(moment - half_double)
    if status != 0 or resolvent_residual > 2e-6 or susceptibility <= 0:
        raise AssertionError('Projected physical static resolvent did not converge positively')
    if max(sum_residual, moment_residual, commutator_residual) > 2e-6:
        raise AssertionError('Sparse connected photon sum rules failed')
    return {'connected_variance': variance, 'zero_energy_connected_weight': zero_connected,
            'inelastic_weight': norm2, 'energy_moment': moment,
            'half_double_commutator': half_double, 'equal_time_sum_rule_residual': float(sum_residual),
            'energy_sum_rule_residual': moment_residual, 'double_commutator_residual': commutator_residual,
            'static_susceptibility': susceptibility, 'static_resolvent_residual': resolvent_residual,
            'lanczos_steps': len(diagonal), 'lanczos_static_susceptibility': float(np.sum(2 * weights / poles)),
            'spectral_scope': 'positive Lanczos quadrature, NOT an enumerated exact full spectrum',
            'poles': [{'energy': float(e), 'weight': float(w)} for e, w in zip(poles, weights)
                      if w > 1e-12]}


def entanglement(model, ground):
    """Trace in actual occupation-code x electric-field embedding; no factor assumption."""
    masks = list(dict.fromkeys(mask for mask, _ in model.raw))
    fields = list(dict.fromkeys(f for _, f in model.raw))
    mi, fi = {m: i for i, m in enumerate(masks)}, {f: i for i, f in enumerate(fields)}
    A = sparse.coo_matrix((ground, ([mi[m] for m, _ in model.raw],
                                   [fi[f] for _, f in model.raw])),
                          shape=(len(masks), len(fields))).tocsr()
    rho = (A.conjugate().T @ A).toarray()
    purity = float(np.vdot(rho, rho).real)
    shell = np.array([any(abs(e) == model.cutoff for e in f) for _, f in model.raw])
    return {'matter_rotor_purity': purity, 'linear_entanglement_entropy': 1 - purity,
            'cutoff_shell_probability': float(np.sum(abs(ground[shell]) ** 2)),
            'loop_raise_loss': float(1 - np.linalg.norm(model.W @ ground) ** 2),
            'loop_lower_loss': float(1 - np.linalg.norm(model.W.conjugate().T @ ground) ** 2)}


def creator_overlap(neutral, neutral_ground, charged, charged_ground):
    reports = []
    for orbital in range(4):
        image = np.zeros(charged.dimension, dtype=complex)
        source, target = (MARKER, orbital) if charged.N == 9 else (orbital, MARKER)
        path = neutral.code.words.path(source, target)
        word = neutral.code.words.path_A(path).times_i(-1)
        for column, (mask, fields) in enumerate(neutral.raw):
            filled = bool(mask >> orbital & 1)
            if filled == (charged.N == 9):
                continue
            new = mask ^ (1 << orbital)
            # Toggle marker too: actual Pauli word does so, marker carries zero EM charge.
            amplitude = neutral.code.element(mask, new, word)
            row = charged.index.get((new, fields))
            if row is None:
                raise AssertionError('Boundary Gauss string escaped charged sector')
            image[row] += amplitude * neutral_ground[column]
        norm2 = float(np.vdot(image, image).real)
        weight = float(np.linalg.norm(charged_ground.conj().T @ image) ** 2)
        reports.append({'orbital': orbital, 'creator_norm_squared': norm2,
                        'charged_ground_multiplet_weight': weight,
                        'normalized_ground_overlap': weight / norm2 if norm2 > 1e-14 else 0.0})
    if sum(row['charged_ground_multiplet_weight'] for row in reports) <= 1e-9:
        raise AssertionError('Actual boundary creator has no charged-ground residue')
    return reports


def cutoff_comparison(small, large):
    sm, se, sg, sr = small
    lm, le, _, lr = large
    embedded = np.zeros(lm.dimension, dtype=complex)
    for coefficient, state in zip(sg[:, 0], sm.raw):
        embedded[lm.index[state]] = coefficient
    residual = float(np.linalg.norm(lm.H @ embedded - se * embedded))
    if le > se + 2e-6:
        raise AssertionError('Nested electric cutoff violated variational monotonicity')
    return {'N': sm.N, 'from': sm.cutoff, 'to': lm.cutoff,
            'ground_energy_change': le - se, 'embedded_ground_residual': residual,
            'large_spectrum_distance_upper_bound': residual,
            'B_static_change': lr['B_response']['static_susceptibility'] - sr['B_response']['static_susceptibility'],
            'B_weight_change': lr['B_response']['inelastic_weight'] - sr['B_response']['inelastic_weight']}


def defect_intertwiner(plus, minus):
    """Actual Q_eta R_eta phases and Delta-2K operator proof, both sectors kept."""
    rows, phases = [], []
    for mask, fields in plus.raw:
        target = dict(plus.code.state(mask, -1))
        coefficient = 0j
        for q, amplitude in plus.code.state(mask, 1):
            coefficient += np.conjugate(target.get(q, 0)) * (-1) ** (q >> plus.code.Q_edge & 1) * amplitude
        # Chosen omitted edge has canonical orientation 0->12 opposite E_30.
        coefficient *= (-1.0) ** fields[3]
        if abs(abs(coefficient) - 1) > 1e-12:
            raise AssertionError('Q_eta R_eta is not a code-sector unitary')
        rows.append(minus.index[mask, fields]); phases.append(coefficient)
    F = sparse.coo_matrix((phases, (rows, np.arange(plus.dimension))), shape=plus.H.shape).tocsr()
    # F† H_- F = H_+ + Delta I + K(W+W†).
    residual = max_abs(F.conjugate().T @ minus.H @ F - plus.H
                       - plus.delta * sparse.eye(plus.dimension) - plus.K * (plus.W + plus.W.conjugate().T))
    if residual > 2e-10:
        raise AssertionError('Exact defect-sector intertwining identity failed')
    return {'operator_identity_residual': residual,
            'rigorous_defect_ground_margin_lower_bound': plus.delta - 2 * plus.K,
            'response_cross_sector_coupling': 0.0,
            'cross_sector_reason': 'actual S commutes with all Pauli transfers; B acts only on electric loop'}


def run(cutoffs=(1, 2), magnetic=0.25, electric=0.6, delta=1.2,
        m=0.8, r=0.04, v=0.08, lanczos_steps=36, check_minus=False):
    if len(cutoffs) < 2 or sorted(set(cutoffs)) != list(cutoffs) or cutoffs[0] < 1:
        raise ValueError('Require at least two strictly increasing positive electric cutoffs')
    if m <= 0 or r < 0 or v <= 0 or lanczos_steps < 4:
        raise ValueError('Require m,v>0, r>=0 and at least four Lanczos steps')
    code = PauliCode()
    cases, comparisons, joins, defects = [], [], [], []
    previous = {}
    kwargs = dict(magnetic=magnetic, electric=electric, delta=delta, m=m, r=r, v=v)
    # Each spatial link has trace-norm 2 max(r,v). The actual integer-Gauss
    # photon Holder theorem cancels the common neutral photon ground energy;
    # no -K loss remains. See compact_charge_gap.md.
    charged_lower_bound = m + 2 * r + electric / 2 - 8 * max(r, v)
    for cutoff in cutoffs:
        current = {}
        for number in (8, 7, 9):
            model = SpinorPlaquette(code, number, cutoff, **kwargs)
            energies, ground, spectrum = solve_low(model)
            energy = float(energies[0])
            B_report = response(model, energy, ground, lanczos_steps)
            entropy = entanglement(model, ground[:, 0])
            report = {'N': number, 'cutoff': cutoff, 'dimension': model.dimension,
                      'Hamiltonian_nonzeros': model.H.nnz, 'boundary_flux': model.boundary,
                      'marker_occupation': number % 2, 'marker_EM_charge': 0,
                      'ground_energy': energy, 'hermiticity_residual': model.hermiticity,
                      'T_squared_residual': model.Tsquare, 'H_time_reversal_residual': model.Tsymmetry,
                      'B_time_reversal_odd_residual': model.B_Todd,
                      'spectrum': spectrum, 'B_response': B_report, **entropy,
                      'adverse_controls': model.adverse_controls()}
            if number == 8 and entropy['linear_entanglement_entropy'] <= 1e-8:
                raise AssertionError('Neutral actual vacuum is not matter-photon entangled')
            current[number] = (model, energy, ground, report)
            cases.append(report)
            if number in previous:
                comparisons.append(cutoff_comparison(previous[number], current[number]))
        neutral, E8, G8, R8 = current[8]
        join = {'cutoff': cutoff}
        for number, label in ((7, 'removal'), (9, 'addition')):
            charged, energy, ground, report = current[number]
            gap = energy - E8
            if gap <= 1e-7:
                raise AssertionError('Positive finite charged gap failed for supplied couplings')
            if gap < charged_lower_bound - 2e-6:
                raise AssertionError('Finite charged energy violates constructive analytic gap bound')
            join[label] = {'charged_gap': gap, 'boundary_creator': creator_overlap(neutral, G8[:, 0], charged, ground),
                           'B_static_difference_from_neutral': report['B_response']['static_susceptibility'] - R8['B_response']['static_susceptibility'],
                           'B_weight_difference_from_neutral': report['B_response']['inelastic_weight'] - R8['B_response']['inelastic_weight']}
        joins.append(join)
        # Prove selection at every cutoff in the full neutral occupation space.
        minus = SpinorPlaquette(code, 8, cutoff, sign=-1, **kwargs)
        defect = {'N': 8, 'cutoff': cutoff, **defect_intertwiner(neutral, minus)}
        if check_minus:
            # A pi-twisted neutral sector can legitimately have extra degeneracy;
            # selection needs its bottom energy, not an imposed singlet claim.
            rng = np.random.default_rng(1222 + cutoff)
            start = rng.normal(size=minus.dimension) + 1j * rng.normal(size=minus.dimension)
            minus_energies, minus_vectors = eigsh(minus.H, k=1, which='SA',
                                                  tol=2e-9, v0=start, maxiter=4000)
            defect_residual = float(np.linalg.norm(minus.H @ minus_vectors[:, 0]
                                                  - minus_energies[0] * minus_vectors[:, 0]))
            if defect_residual > 2e-6:
                raise AssertionError('Discarded-sector bottom eigenpair residual failed')
            defect['discarded_ground_eigen_residual'] = defect_residual
            defect['measured_defect_ground_margin'] = float(minus_energies[0] - E8)
            if defect['measured_defect_ground_margin'] < delta - 2 * magnetic - 2e-6:
                raise AssertionError('Measured discarded-sector margin violates exact bound')
        defects.append(defect)
        previous = current
    return {'scope': 'finite open four-cell planar p_z=0 spinor/compact-photon quantum join; NOT continuum QED or a 3D phase',
            'units': 'hbar=a=1; m,r,v,K,UE,Delta and frequency energies; E,B,n dimensionless; static susceptibility inverse energy',
            'couplings': {'m': m, 'r': r, 'v': v, 'K': magnetic, 'UE': electric, 'Delta': delta},
            'boundary_conditions': 'open square; four internal oriented noncyclic electric rotors; fixed outward boundary E=N-8 at cell0; no exterior dynamical photon; background2 per cell; neutral marker leaf',
            'operator_owner': 'actual EdgeWords Pauli code states and primitive paths + compact Gauss electric loop shift',
            'state_owner': 'sparse eigsh ground multiplets of same nonlinear interacting Hamiltonian',
            'code_matrix_unit_law_residual': code.law_residual,
            'selection_theorem': 'Q_eta R_eta gives H_- equivalent H_+ + Delta + K(W+Wdag); ||W||<=1; margin>=Delta-2K>0 in every N sector',
            'charged_gap_status': 'finite sparse energies plus a rigorous sufficient onsite/trace-norm bound; not a continuum theorem',
            'rigorous_charged_gap_lower_bound': charged_lower_bound,
            'analytic_positive_charged_gap_condition_satisfied': charged_lower_bound > 0,
            'photon_response': 'H_source=-j B; exact connected sum rules; positive Lanczos quadrature; converged projected static resolvent',
            'cases': cases, 'charged_joins': joins, 'cutoff_comparisons': comparisons, 'defect_sector_checks': defects}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cutoffs', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--magnetic', type=float, default=0.25)
    parser.add_argument('--electric', type=float, default=0.6)
    parser.add_argument('--delta', type=float, default=1.2)
    parser.add_argument('--m', type=float, default=0.8)
    parser.add_argument('--r', type=float, default=0.04)
    parser.add_argument('--v', type=float, default=0.08)
    parser.add_argument('--lanczos-steps', type=int, default=36)
    parser.add_argument('--check-minus', action='store_true')
    args = parser.parse_args()
    result = run(tuple(args.cutoffs), args.magnetic, args.electric, args.delta,
                 args.m, args.r, args.v, args.lanczos_steps, args.check_minus)
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.write_text(payload)
    print(payload, end='')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Full native periodic edge-Pauli even algebra with exact compressed code action.

No exponential edge Hilbert matrix is allocated. Pauli words are literal 355-bit
words; flat-history decoding uses their central cycle algebra, not fitted CAR
signs. Encoded Slater preparation and auxiliary syndrome images are separate.
"""
from __future__ import annotations
from collections import deque
from functools import lru_cache
from dataclasses import dataclass
import itertools
from typing import Any
import numpy as np
from scipy.linalg import logm, null_space
from bosonic_statistics import primitive_pauli_words


@dataclass(frozen=True)
class PauliWord:
    """i**phase X**x Z**z, with edge zero the least significant bit."""
    x: int = 0
    z: int = 0
    phase: int = 0

    def __matmul__(self, other: PauliWord) -> PauliWord:
        return PauliWord(self.x ^ other.x, self.z ^ other.z,
                         (self.phase + other.phase + 2*(self.z & other.x).bit_count()) % 4)

    def scale_i(self, phase: int) -> PauliWord:
        return PauliWord(self.x, self.z, (self.phase + phase) % 4)

    def adjoint(self) -> PauliWord:
        return PauliWord(self.x, self.z, (-self.phase + 2*(self.x & self.z).bit_count()) % 4)

    def anticommutes(self, other: PauliWord) -> bool:
        return bool(((self.x & other.z).bit_count() + (self.z & other.x).bit_count()) % 2)

    @classmethod
    def letters(cls, letters: tuple[str, ...], sign: int = 1) -> PauliWord:
        x = z = 0
        phase = 0 if sign == 1 else 2
        for edge, letter in enumerate(letters):
            if letter in ('X', 'Y'): x |= 1 << edge
            if letter in ('Z', 'Y'): z |= 1 << edge
            if letter == 'Y': phase += 1
        return cls(x, z, phase % 4)


@dataclass
class EvenTerm:
    target: int
    source: int
    coefficient: complex
    path: tuple[int, ...]
    words: tuple[tuple[complex, PauliWord], ...]


class NativePauliGraph:
    """H/L onsite stars, H-L bridges, periodic anchor paths, attached Ref tree."""
    def __init__(self, law: Any, reference_site: int):
        self.law = law
        self.volume = law.volume
        self.modes = 8*self.volume + 4
        self.reference_site = int(reference_site)
        edges = []
        for species in range(2):
            for site in range(self.volume):
                anchor = self.mode(species, site, 0)
                edges.extend((anchor, self.mode(species, site, colour)) for colour in range(1, 4))
                edges.extend((anchor, self.mode(species, int(law.target[site, axis]), 0)) for axis in range(3))
        edges.extend((self.mode(0, site, 0), self.mode(1, site, 0)) for site in range(self.volume))
        self.ref = tuple(range(8*self.volume, self.modes))
        edges.extend((self.ref[0], self.ref[colour]) for colour in range(1, 4))
        edges.append((self.mode(1, self.reference_site, 0), self.ref[0]))
        self.edges = tuple(sorted(tuple(sorted(edge)) for edge in edges))
        if len(set(self.edges)) != len(self.edges): raise ValueError('Native graph must be simple')
        self.edge_index = {edge: i for i, edge in enumerate(self.edges)}
        self.incident = tuple(tuple(e for e, pair in enumerate(self.edges) if vertex in pair)
                              for vertex in range(self.modes))
        b, a = primitive_pauli_words(self.edges, self.incident)
        self.B = tuple(PauliWord.letters(word, sign) for sign, word in b)
        self.A = {pair: PauliWord.letters(word) for pair, word in a.items()}
        self.parent: dict[int, int | None] = {0: None}
        queue = deque([0])
        tree = set()
        while queue:
            vertex = queue.popleft()
            for e in self.incident[vertex]:
                pair = self.edges[e]; neighbour = pair[1] if pair[0] == vertex else pair[0]
                if neighbour not in self.parent:
                    self.parent[neighbour] = vertex; tree.add(pair); queue.append(neighbour)
        if len(self.parent) != self.modes: raise ValueError('Connected graph required')
        self.cycles = []
        self.cycle_windings = []
        for pair in self.edges:
            if pair not in tree:
                path = self.tree_path(pair[1], pair[0])
                cycle = (pair[0], *path[:-1])
                self.cycles.append(self.cycle(cycle))
                self.cycle_windings.append(self.winding((*cycle, cycle[0])))
        self.local_cycles = []
        for species in range(2):
            for site in range(self.volume):
                for axis, other in itertools.combinations(range(3), 2):
                    a1 = int(law.target[site, axis]); b1 = int(law.target[site, other]); ab = int(law.target[a1, other])
                    self.local_cycles.append(self.cycle(tuple(self.mode(species, x, 0) for x in (site, a1, ab, b1))))
        for site in range(self.volume):
            for axis in range(3):
                target = int(law.target[site, axis])
                self.local_cycles.append(self.cycle((self.mode(0, site, 0), self.mode(0, target, 0),
                                                    self.mode(1, target, 0), self.mode(1, site, 0))))
        self.spatial_edges = tuple(self.edge_index[tuple(sorted((self.mode(species, site, 0),
                                  self.mode(species, int(law.target[site, axis]), 0))))]
                                  for species in range(2) for site in range(self.volume) for axis in range(3))
        self.epsilon = 1/(96*self.volume)
        self.central_bases: dict[tuple[int, ...], dict[int, tuple[PauliWord, complex]]] = {
            history: self._central_basis(history) for history in itertools.product((1, -1), repeat=3)}
        self.syndromes = {e: sum(int(bool(word.x & (1 << e))) << j for j, word in enumerate(self.cycles))
                          for e in self.spatial_edges}
        self._decoded_terms: dict[tuple[int, tuple[int, ...]], tuple[list[EvenTerm], list[tuple[int, int, complex, int]]]] = {}
        if len(set(self.syndromes.values())) != len(self.spatial_edges) or any(x == 0 for x in self.syndromes.values()):
            raise ValueError('This one-defect orthogonal bundle requires distinct nonzero native syndromes')

    @lru_cache(maxsize=None)
    def mode(self, species: int, site: int, colour: int) -> int:
        return species*4*self.volume + 4*site + colour

    @lru_cache(maxsize=None)
    def tree_path(self, source: int, target: int) -> tuple[int, ...]:
        left = [source]
        while (parent := self.parent[left[-1]]) is not None:
            left.append(parent)
        right = [target]
        while right[-1] not in left:
            parent = self.parent[right[-1]]
            if parent is None: raise ValueError('Tree paths have no common root')
            right.append(parent)
        return tuple(left[:left.index(right[-1])+1] + right[-2::-1])

    def edge_A(self, source: int, target: int) -> PauliWord:
        return self.A[tuple(sorted((source, target)))].scale_i(0 if source < target else 2)

    @lru_cache(maxsize=None)
    def path_A(self, path: tuple[int, ...]) -> PauliWord:
        out = PauliWord()
        for source, target in zip(path[:-1], path[1:]): out = out @ self.edge_A(source, target)
        return out.scale_i(len(path)-2)
    def cycle(self, vertices: tuple[int, ...]) -> PauliWord:
        out = PauliWord()
        for source, target in zip(vertices, (*vertices[1:], vertices[0])): out = out @ self.edge_A(source, target)
        return out.scale_i(len(vertices))

    def winding(self, path: tuple[int, ...]) -> tuple[int, int, int]:
        counts = [0, 0, 0]
        for source, target in zip(path[:-1], path[1:]):
            if source >= 8*self.volume or target >= 8*self.volume: continue
            if source//(4*self.volume) != target//(4*self.volume): continue
            x, y = (source % (4*self.volume))//4, (target % (4*self.volume))//4
            if x == y: continue
            delta = self.law.sites[y] - self.law.sites[x]
            for axis in range(3):
                if abs(int(delta[axis])) == self.law.size-1: counts[axis] ^= 1
        return counts[0], counts[1], counts[2]

    def _central_basis(self, history: tuple[int, ...]) -> dict[int, tuple[PauliWord, complex]]:
        basis = {}
        for original, winding in zip(self.cycles, self.cycle_windings):
            word = original
            value: complex = complex(np.prod([history[j] for j in range(3) if winding[j]]))
            while word.x or word.z:
                mask = word.x | (word.z << len(self.edges)); pivot = mask.bit_length()-1
                if pivot not in basis: basis[pivot] = (word, value); break
                old, eigenvalue = basis[pivot]; word = word @ old; value *= eigenvalue
            else:
                if abs(value - 1j**word.phase) > 1e-12: raise ValueError('Inconsistent central cycle assignment')
        return basis

    def central_value(self, word: PauliWord, history: tuple[int, ...]) -> complex:
        value = 1+0j
        basis = self.central_bases[history]
        while word.x or word.z:
            pivot = (word.x | (word.z << len(self.edges))).bit_length()-1
            if pivot not in basis: raise ValueError('Word is not a central cycle')
            old, eigenvalue = basis[pivot]; word = word @ old; value *= eigenvalue
        return value * 1j**word.phase

    def bilinear(self, target: int, source: int, coefficient: complex,
                 path: tuple[int, ...] | None = None) -> EvenTerm:
        if target == source:
            words = ((coefficient/2, PauliWord()), (-coefficient/2, self.B[source]))
            return EvenTerm(target, source, coefficient, (source,), words)
        path = self.tree_path(source, target) if path is None else path
        if path[0] != source or path[-1] != target: raise ValueError('Path endpoints disagree')
        a = self.path_A(path).scale_i(3)
        words = tuple((coefficient*sign/4, word) for sign, word in
                      ((1, a), (-1, a @ self.B[source]), (1, a @ self.B[target]),
                       (-1, a @ self.B[source] @ self.B[target])))
        return EvenTerm(target, source, coefficient, path, words)

    def lift_matrix(self, matrix: np.ndarray, modes: tuple[int, ...]) -> list[EvenTerm]:
        if matrix.shape != (len(modes), len(modes)): raise ValueError('Full matrix/modes mismatch')
        return [self.bilinear(modes[i], modes[j], complex(matrix[i, j].item())) for i, j in zip(*np.nonzero(matrix))]

    def path_terms(self, law: Any, fields: np.ndarray, links: np.ndarray, a: np.ndarray,
                   species: int, first: np.ndarray | None = None, second: np.ndarray | None = None,
                   drop_two: bool = False) -> list[EvenTerm]:
        terms = []
        if first is None:
            for x in range(self.volume):
                modes = tuple(self.mode(species, x, j) for j in range(4))
                terms.extend(self.lift_matrix(law.onsite*fields[x, 0], modes))
        for x, y, block, field_path in law.terms(fields, links, a):
            if drop_two and len(field_path) == 2: continue
            forward = 1+0j if first is None else 1j*sum(first[e] for e in field_path)
            backward = 1+0j if first is None else -1j*sum(first[e] for e in field_path)
            if second is not None:
                forward *= 1j*sum(second[e] for e in field_path)
                backward *= -1j*sum(second[e] for e in field_path)
            anchors = [self.mode(species, x, 0)]
            site = x
            for _, axis in field_path:
                site = int(law.target[site, axis]); anchors.append(self.mode(species, site, 0))
            for i, j in itertools.product(range(4), repeat=2):
                target, source = self.mode(species, x, i), self.mode(species, y, j)
                path = tuple(([source] if j else []) + list(reversed(anchors)) + ([target] if i else []))
                if block[i, j] != 0:
                    terms.append(self.bilinear(target, source, block[i, j]*forward, path))
                    terms.append(self.bilinear(source, target, block[i, j].conjugate()*backward, tuple(reversed(path))))
        return terms

    def decode(self, terms: list[EvenTerm], history: tuple[int, ...] = (1, 1, 1),
               defect_edge: int | None = None) -> np.ndarray:
        key = (id(terms), history)
        if key not in self._decoded_terms:
            decoded = []
            for term in terms:
                coefficient = term.coefficient
                actual_x = 0
                if term.target != term.source:
                    actual = self.path_A(term.path)
                    canonical = self.path_A(self.tree_path(term.source, term.target))
                    coefficient *= self.central_value(actual @ canonical.adjoint(), history)
                    actual_x = actual.x
                decoded.append((term.target, term.source, coefficient, actual_x))
            self._decoded_terms[key] = (terms, decoded)
        out = np.zeros((self.modes, self.modes), complex)
        for target, source, coefficient, actual_x in self._decoded_terms[key][1]:
            if defect_edge is not None and actual_x & (1 << defect_edge): coefficient *= -1
            out[target, source] += coefficient
        return out

    def polynomial_normalform(self, words: Any, history: tuple[int, ...] = (1, 1, 1)) -> dict[tuple[int, int], complex]:
        """Actual Pauli multiplication quotient by evaluated cycle eigenvalues."""
        out: dict[tuple[int, int], complex] = {}
        for coefficient, original in words:
            word = original
            for pivot, (cycle, value) in sorted(self.central_bases[history].items(), reverse=True):
                if (word.x | (word.z << len(self.edges))) & (1 << pivot):
                    word = word @ cycle
                    coefficient *= value
            key = word.x, word.z
            out[key] = out.get(key, 0j)+coefficient*1j**word.phase
        return {key:value for key,value in out.items() if abs(value) > 1e-12}

    def bilinear_commutator_error(self, left: EvenTerm, right: EvenTerm) -> float:
        words = [(a*b, x@y) for a,x in left.words for b,y in right.words]
        words += [(-a*b, y@x) for a,x in left.words for b,y in right.words]
        expected = []
        if left.source == right.target:
            expected.extend(self.bilinear(left.target, right.source, left.coefficient*right.coefficient).words)
        if left.target == right.source:
            expected.extend(self.bilinear(right.target, left.source, -left.coefficient*right.coefficient).words)
        actual = self.polynomial_normalform(words)
        target = self.polynomial_normalform(expected)
        return float(sum(abs(actual.get(key, 0)-target.get(key, 0)) for key in actual.keys() | target.keys()))

    def algebra_report(self) -> dict[str, Any]:
        errors = 0
        for pair, a in self.A.items():
            if a @ a != PauliWord(): errors += 1
            for endpoint in pair:
                if not a.anticommutes(self.B[endpoint]): errors += 1
        for (left_pair, left), (right_pair, right) in itertools.combinations(self.A.items(), 2):
            if left.anticommutes(right) != bool(len(set(left_pair) & set(right_pair)) == 1): errors += 1
        parity = PauliWord()
        for word in self.B: parity = parity @ word
        if parity != PauliWord(): errors += 1
        def rank(words):
            basis = {}
            for word in words:
                value = word.x | (word.z << len(self.edges))
                while value:
                    pivot = value.bit_length()-1
                    if pivot not in basis: basis[pivot] = value; break
                    value ^= basis[pivot]
            return len(basis)
        stabilizer_rank = rank((*self.B[:-1], *self.cycles))
        local_rank = rank(self.local_cycles)
        if stabilizer_rank != len(self.edges) or len(self.cycles)-local_rank != 3: errors += 1
        for word in self.cycles:
            if word @ word != PauliWord() or any(word.anticommutes(a) for a in self.A.values()): errors += 1
        flat_local_errors = sum(abs(self.central_value(word, history)-1) > 1e-12
                                for history in self.central_bases for word in self.local_cycles)
        return dict(modes=self.modes, edge_qubits=len(self.edges), independent_cycles=len(self.cycles),
                    local_cycles=len(self.local_cycles), local_constraint_rank=local_rank,
                    joint_occupation_cycle_stabilizer_rank=stabilizer_rank, histories=len(self.central_bases),
                    primitive_algebra_errors=errors, all8_local_flat_errors=flat_local_errors,
                    distinct_nonzero_defect_syndromes=len(set(self.syndromes.values())),
                    beta=len(self.spatial_edges)*self.epsilon,
                    defect_D_on_unit_code_norm_squared=len(self.spatial_edges)*self.epsilon**2,
                    full_edge_Hilbert_dimension=f'2^{len(self.edges)}', allocated_edge_Hilbert_matrix=False)


def quadratic_norm_squared(matrix: np.ndarray, projector: np.ndarray) -> float:
    """Exact precompression dGamma(M) Slater second product, including PH."""
    mean = np.trace(projector @ matrix)
    connected = np.trace(projector @ matrix.conj().T @ (np.eye(len(matrix))-projector) @ matrix)
    return float((abs(mean)**2+connected).real)


def encoded_slater_preparation(graph: NativePauliGraph, occupied: np.ndarray,
                               modes: tuple[int, ...]) -> dict[str, Any]:
    """An executable encoded quadratic rotation, not a new ground selection.

    Seed: the unique joint B occupation/cycle eigenstate. Its phase is fixed by
    tree A=-i gamma_s gamma_t. exp(dGamma(log U)) rotates that seed to the given
    cached orbitals. The returned literal Pauli generator specifies the circuit.
    """
    empty = null_space(occupied.conj().T)
    unitary = np.column_stack((occupied, empty))
    generator = logm(unitary)
    terms = graph.lift_matrix(generator, modes)
    return dict(occupied_modes=len(occupied.T), unitary=unitary, antihermitian_generator=generator,
                encoded_generator=terms, seed_occupied=tuple(modes[:occupied.shape[1]]),
                unitary_error=float(np.linalg.norm(unitary.conj().T@unitary-np.eye(len(unitary)))),
                generator_error=float(np.linalg.norm(generator+generator.conj().T)),
                state_definition='joint B occupation/cycle stabilizer state followed by exp(encoded dGamma(log cached U))')


def defect_commutator(graph: NativePauliGraph, terms: list[EvenTerm],
                       projectors: tuple[np.ndarray, np.ndarray], weights: np.ndarray) -> dict[str, Any]:
    base = graph.decode(terms)
    per_edge = []
    for edge in graph.spatial_edges:
        # [D,B]P = sum epsilon Z_e (B-Z_e B Z_e)P. Distinct syndromes are orthogonal.
        difference = base-graph.decode(terms, defect_edge=edge)
        norms = [quadratic_norm_squared(difference, p) for p in projectors]
        per_edge.append(float(np.dot(abs(weights)**2, norms))*graph.epsilon**2)
    return dict(norm_squared=float(sum(per_edge)), edge_norm_squared=per_edge,
                formula='sum_e epsilon_e^2 ||(B-Z_e B Z_e) Psi||^2; actual orthogonal auxiliary syndromes',
                auxQ_not_PH=True)


def syndrome_dressing_bound(graph: NativePauliGraph, state_norm_squared: float,
                            delta: float, time: float) -> dict[str, Any]:
    """Evaluated first Duhamel syndrome bundle for Delta*local-defect count+D.

    Every nonzero epsilon is kept. The norm remainder bound beta^2 t^2/2 is
    uniform, since conjugation by the penalty is unitary. This proves actual
    auxiliary evolution of the fixed state, not full interacting H evolution.
    """
    amplitudes = []
    costs = []
    for edge in graph.spatial_edges:
        count = sum(bool(word.x & (1 << edge)) for word in graph.local_cycles)
        if count == 0: raise ValueError('An unpenalized spatial defect cannot enter this bundle')
        gap = delta*count
        amplitudes.append(graph.epsilon*np.expm1(-1j*gap*time)/gap)
        costs.append(count)
    norm2 = float(np.vdot(amplitudes, amplitudes).real)*state_norm_squared
    remainder = .5*(len(graph.spatial_edges)*graph.epsilon*time)**2*np.sqrt(state_norm_squared)
    return dict(delta=delta, time=time, local_defect_counts=sorted(set(costs)),
                first_Duhamel_auxQ_norm_squared=norm2, amplitudes=np.asarray(amplitudes),
                exact_evolution_minus_bundle_norm_upper=float(remainder),
                exact_auxQ_norm_lower=max(0.,np.sqrt(norm2)-remainder),
                penalty_energy_first_bundle=state_norm_squared*sum(delta*c*abs(a)**2 for c,a in zip(costs,amplitudes)),
                nonzero_D=True, selected_state_unchanged=True,
                scope='rigorous bounded actual selected-state auxiliary evolution only; not full field/matter evolution')


def selected_syndrome_bundle(graph: NativePauliGraph, state_norm_squared: float,
                             delta_floor_log2: int, scaled_time: float) -> dict[str, Any]:
    """Same actual Delta selector, using tau=Delta*t to avoid false floating zero.

    No numerical Delta is substituted for the full-field dominance expression.
    Amplitudes here are Delta times the true amplitudes; absolute nonzero
    values retain the exact symbolic 1/Delta instead of underflowing to zero.
    """
    scaled = syndrome_dressing_bound(graph, state_norm_squared, 1., scaled_time)
    first_norm = np.sqrt(scaled['first_Duhamel_auxQ_norm_squared'])
    beta = len(graph.spatial_edges)*graph.epsilon
    remainder_coefficient = .5*(beta*scaled_time)**2*np.sqrt(state_norm_squared)
    # Delta>=1 gives a conservative, representable lower bound for Delta||Qpsi||.
    lower = max(0., first_norm-remainder_coefficient)
    return dict(scaled_time_tau=scaled_time, physical_time='tau/actual_Delta',
                Delta_scaled_first_amplitudes=scaled['amplitudes'],
                Delta_squared_times_first_auxQ_norm_squared=scaled['first_Duhamel_auxQ_norm_squared'],
                Delta_times_exact_auxQ_norm_lower=float(lower),
                absolute_exact_auxQ_norm_lower='Delta_times_exact_auxQ_norm_lower/actual_Delta > 0',
                Delta_times_remainder_upper='remainder_coefficient/actual_Delta',
                remainder_coefficient=float(remainder_coefficient),
                log2_Delta_times_remainder_upper=float(np.log2(remainder_coefficient)-delta_floor_log2),
                local_defect_counts=scaled['local_defect_counts'],
                Delta_times_penalty_energy_first_bundle=scaled['penalty_energy_first_bundle'],
                actual_Delta='B_N(Lambda)+U_trial(Lambda)+window+beta+max(2^GAP_LOG2,B_N+U_trial+window+beta+1)',
                gap_floor_log2=delta_floor_log2, gap_floor_is_NOT_actual_Delta=True,
                nonzero_D=True, selected_state_unchanged=True,
                zeroD_wrong_case_Delta_times_auxQ_norm=0.,
                scope='bounded SAME selected-Delta auxiliary evolution at t=tau/Delta; not full interacting field/matter propagation')


def conditional_fullfield_window(graph: NativePauliGraph, graph_norm: float,
                                 lower_H0: float | None, code_trial_upper: float | None,
                                 window: float | None, delta: float, time: float) -> dict[str, Any]:
    """Renewed full-field domination; caller quantities are obligations, not inferred.

    A frozen initial function is NOT presumed spectrally confined by this API.
    Coexact sqrt-field and Phi graphs have different powers; harmonic has none.
    """
    beta = len(graph.spatial_edges)*graph.epsilon
    if lower_H0 is None or code_trial_upper is None or window is None:
        return dict(gap=None, admitted=False, fullfield_transfer_admitted=False,
                    missing=[name for name, value in
                             [('fullfield lower bound', lower_H0), ('actual code trial upper', code_trial_upper),
                              ('original source/driven spectral window', window)] if value is None],
                    point_graph_norm_is_not_fullfield_graph=True,
                    Phi_weighted_scaling='gap^-3/4', coexact_weighted_scaling='gap^-1/2',
                    harmonic='free p_theta^2; no oscillator position coercivity',
                    original_state_window_paid=False, driven_source_window_paid=False)
    gap = delta + lower_H0-code_trial_upper-window-beta
    admitted = gap > 0
    return dict(gap=gap, admitted=admitted,
                aux_leakage_upper=beta/gap if admitted else None,
                Schur_upper=beta**2/gap if admitted else None,
                bounded_graph_transfer_upper=graph_norm*(4*beta/gap+abs(time)*beta**2/gap) if admitted else None,
                Phi_weighted_scaling='beta/gap^(3/4), requires renewed quartic graph constants',
                coexact_weighted_scaling='beta/gap^(1/2), requires actual oscillator graph',
                harmonic='free p_theta^2: no oscillator coercivity or weighted position bound',
                original_state_window_paid=False, driven_source_window_paid=False,
                fullfield_transfer_admitted=False)

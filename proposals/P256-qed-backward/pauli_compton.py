#!/usr/bin/env python3
"""Actual edge-qubit filled-sea Compton, not an empty negative-band resolvent.

A spectral basis is used ONLY for this finite current-response calculation.
The separate decorated-cubic probe establishes bounded microscopic support.
No microscopic CAR operators or gamma matrices enter the Pauli calculation.
WilsonLink supplies external photon kinematics and an independent comparator.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np
from scipy.linalg import expm

from bosonic_statistics import BosonicEdgeModel, primitive_pauli_words
from lattice_compton import WilsonLink
from rotation_band import CHARGE_CONJUGATION, ROTATION, RotationBand


@dataclass(frozen=True)
class PauliWord:
    """i**phase X**x Z**z; edge zero is the least significant state bit."""
    x: int = 0
    z: int = 0
    phase: int = 0

    @classmethod
    def from_letters(cls, letters, sign=1):
        x = z = 0
        phase = 2 if sign == -1 else 0
        for edge, letter in enumerate(letters):
            if letter in ('X', 'Y'):
                x |= 1 << edge
            if letter in ('Z', 'Y'):
                z |= 1 << edge
            if letter == 'Y':
                phase += 1
        return cls(x, z, phase % 4)

    def __matmul__(self, other):
        return PauliWord(self.x ^ other.x, self.z ^ other.z,
                         (self.phase + other.phase
                          + 2 * (self.z & other.x).bit_count()) % 4)

    def times_i(self, power):
        return PauliWord(self.x, self.z, (self.phase + power) % 4)

    def action(self, state):
        phase = (self.phase + 2 * (self.z & state).bit_count()) % 4
        return state ^ self.x, (1, 1j, -1, -1j)[phase]

    def commutes(self, other):
        return ((self.x & other.z).bit_count()
                + (self.z & other.x).bit_count()) % 2 == 0

    def hermitian(self):
        return (self.phase - (self.x & self.z).bit_count()) % 2 == 0

    @property
    def weight(self):
        return (self.x | self.z).bit_count()


def independent_constraints(words, qubits):
    """Eliminate commuting stabilizers, retaining signs of dependent products."""
    pivots = {}
    contradictions = 0
    for original in words:
        word = original
        bits = word.x | (word.z << qubits)
        while bits:
            pivot = bits.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = word
                break
            word = word @ pivots[pivot]
            bits = word.x | (word.z << qubits)
        else:
            contradictions += word.phase != 0
    return len(pivots), contradictions


class EdgeWords:
    """The existing primitive bosonic law, evaluated without dense Hilbert space."""
    def __init__(self, vertices, edges, parity=1):
        self.vertices = vertices
        self.edges = tuple(sorted(tuple(sorted(edge)) for edge in edges))
        self.incident = [[] for _ in range(vertices)]
        self.adjacency = [[] for _ in range(vertices)]
        for index, (source, target) in enumerate(self.edges):
            self.incident[source].append(index)
            self.incident[target].append(index)
            self.adjacency[source].append(target)
            self.adjacency[target].append(source)
        b_words, a_words = primitive_pauli_words(self.edges, self.incident, parity, 0)
        self.B = tuple(PauliWord.from_letters(word, sign) for sign, word in b_words)
        self.A = {edge: PauliWord.from_letters(word) for edge, word in a_words.items()}
        self._paths = {}

    def oriented_A(self, source, target):
        word = self.A[min(source, target), max(source, target)]
        return word.times_i(0 if source < target else 2)

    def path(self, source, target):
        key = source, target
        if key not in self._paths:
            queue: deque[tuple[int, tuple[int, ...]]] = deque([(source, (source,))])
            visited = {source}
            while queue:
                vertex, path = queue.popleft()
                if vertex == target:
                    self._paths[key] = path
                    break
                for neighbor in self.adjacency[vertex]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, (*path, neighbor)))
            else:
                raise ValueError('Disconnected primitive graph')
        return self._paths[key]

    def path_A(self, path):
        product = PauliWord()
        for source, target in zip(path[:-1], path[1:]):
            product = product @ self.oriented_A(source, target)
        return product.times_i(len(path) - 2)

    def cycle(self, path):
        product = PauliWord()
        for source, target in zip(path, (*path[1:], path[0])):
            product = product @ self.oriented_A(source, target)
        return product.times_i(len(path))

    def occupied(self, state, vertex):
        target, value = self.B[vertex].action(state)
        assert target == state and value in (-1, 1)
        return value == -1

    def occupation(self, state):
        return tuple(vertex for vertex in range(self.vertices) if self.occupied(state, vertex))

    def tree_basis_state(self, occupied):
        """Unique occupation eigenstate on a tree; no loop-sector shortcut."""
        if len(self.edges) != self.vertices - 1:
            raise ValueError('Occupation-only state construction requires a tree')
        occupied = set(occupied)
        rows = [[word.z, int(vertex in occupied) ^ (word.phase // 2)]
                for vertex, word in enumerate(self.B)]
        pivots = {}
        for mask, rhs in rows:
            while mask:
                pivot = mask.bit_length() - 1
                if pivot not in pivots:
                    pivots[pivot] = mask, rhs
                    break
                old_mask, old_rhs = pivots[pivot]
                mask ^= old_mask
                rhs ^= old_rhs
            else:
                if rhs:
                    raise ValueError('Occupation incompatible with total parity')
        state = 0
        for pivot in sorted(pivots):
            mask, rhs = pivots[pivot]
            if rhs ^ ((mask & state).bit_count() % 2):
                state |= 1 << pivot
        assert self.occupation(state) == tuple(sorted(occupied))
        return state

    def transfer(self, state, source, target):
        if source == target:
            return (state, 1) if self.occupied(state, source) else None
        if not self.occupied(state, source) or self.occupied(state, target):
            return None
        return self.path_A(self.path(source, target)).times_i(-1).action(state)

    def apply(self, terms, amplitudes):
        result = {}
        for state, amplitude in amplitudes.items():
            for target, source, coefficient in terms:
                moved = self.transfer(state, source, target)
                if moved is not None:
                    output, phase = moved
                    result[output] = result.get(output, 0j) + amplitude * coefficient * phase
        return result


def dense_convention_probe():
    """Compare full actual Pauli actions, including every allowed nonedge hop."""
    edges = [(0, 1), (1, 2), (2, 3)]
    words, dense = EdgeWords(4, edges), BosonicEdgeModel(4, edges)
    error = 0.
    for source in range(4):
        for target in range(4):
            matrix = dense.bilinear(target, source,
                                    None if source == target else words.path(source, target))
            for bits in range(1 << len(edges)):
                expected = np.zeros(dense.dimension, dtype=complex)
                moved = words.transfer(bits, source, target)
                if moved is not None:
                    output, phase = moved
                    expected[output] = phase
                error = max(error, float(np.linalg.norm(matrix[:, bits] - expected)))
    assert error < 1e-14
    return {'all_basis_transition_residual': error, 'qubits': len(edges)}


def locality_probe(side):
    """Four-orbital decorated cubic box: only bounded plaquette penalties."""
    sites = list(itertools.product(range(side), repeat=3))
    site_index = {site: index for index, site in enumerate(sites)}
    edges, plaquettes = [], []
    for position, index in site_index.items():
        edges.extend((4 * index, 4 * index + orbital) for orbital in (1, 2, 3))
        for axis in range(3):
            neighbor = tuple(x + int(i == axis) for i, x in enumerate(position))
            if neighbor in site_index:
                edges.append((4 * index, 4 * site_index[neighbor]))
        for axis in range(3):
            for other in range(axis + 1, 3):
                first = tuple(x + int(i == axis) for i, x in enumerate(position))
                second = tuple(x + int(i == other) for i, x in enumerate(position))
                corner = tuple(x + int(i in (axis, other)) for i, x in enumerate(position))
                if corner in site_index:
                    plaquettes.append(tuple(4 * site_index[p]
                                            for p in (position, first, corner, second)))
    boundary = 4 * len(sites)
    edges.append((0, boundary))
    model = EdgeWords(boundary + 1, edges)
    stabilizers = tuple(model.cycle(cycle) for cycle in plaquettes)
    rank, frustrated = independent_constraints(stabilizers, len(edges))
    generators = (*model.B, *model.A.values())
    commutation_failures = sum(not stabilizer.commutes(generator)
                              for stabilizer in stabilizers for generator in generators)
    parity = PauliWord()
    for word in model.B[:boundary]:
        parity = parity @ word
    assert parity == model.B[boundary]
    assert rank == len(edges) - model.vertices + 1 and frustrated == 0
    assert commutation_failures == 0 and all(word.hermitian() for word in stabilizers)
    max_lengths, max_weights = [], []
    # Every on-site and every range<=2 axial orbital matrix element is represented.
    for position, index in site_index.items():
        targets = [position]
        for axis in range(3):
            for step in (1, 2):
                neighbor = tuple(x + step * int(i == axis) for i, x in enumerate(position))
                if neighbor in site_index:
                    targets.append(neighbor)
        for neighbor in targets:
            for first in range(4):
                for second in range(4):
                    source, target = 4 * index + first, 4 * site_index[neighbor] + second
                    if source == target:
                        continue
                    path = model.path(source, target)
                    max_lengths.append(len(path) - 1)
                    max_weights.append(model.path_A(path).weight)
    assert max(max_lengths) <= 4
    return {'side': side, 'vertices': model.vertices, 'edge_qubits': len(edges),
            'plaquette_penalties': len(stabilizers), 'constraint_rank': rank,
            'cycle_rank': len(edges) - model.vertices + 1,
            'frustrated_dependencies': frustrated, 'primitive_commutation_failures': commutation_failures,
            'max_stabilizer_weight': max(word.weight for word in stabilizers),
            'max_band_bilinear_path_edges': max(max_lengths),
            'max_band_path_A_weight': max(max_weights),
            'code_dimension_power_of_two': len(edges) - rank,
            'full_physical_fermion_fock_dimension_power_of_two': boundary,
            'physical_parity_is_neutral_boundary_marker': True,
            'scope': 'Open simply connected box with neutral hidden boundary parity marker. '
                     'Torus also needs specified global Wilson-loop sectors.'}


def block_terms(matrix, target, source):
    return [(target + row, source + column, complex(value))
            for row in range(4) for column in range(4)
            if (value := matrix[row, column]) != 0]


def overlap(bra, ket):
    return sum(amplitude.conjugate() * ket.get(state, 0j) for state, amplitude in bra.items())


def boundary_ladder(words, boundary, vertex, amplitudes, creation=True):
    """Full physical CAR from bosonic A_boundary,vertex and n_vertex.

    The extra leaf is a neutral hidden parity marker, not a spectator fermion.
    gamma_vertex=A_boundary,vertex; c=gamma n, c^dagger=gamma(1-n).
    Even physical observables retain bounded support. Odd operators need a
    boundary string, as expected for an emergent fermionic charged sector.
    """
    word = words.path_A(words.path(boundary, vertex))
    result = {}
    for state, amplitude in amplitudes.items():
        occupied = words.occupied(state, vertex)
        if (creation and occupied) or (not creation and not occupied):
            continue
        output, phase = word.action(state)
        result[output] = amplitude * phase
    return result


def boundary_algebra_and_gauss_probe(charge):
    """Full odd/even algebra plus actual charged quantum state with boundary flux."""
    words = EdgeWords(5, [(0, 1), (1, 2), (2, 3), (0, 4)])
    annihilation, creation = [], []
    for vertex in range(4):
        operators = []
        for raising in (False, True):
            operator = np.zeros((16, 16), dtype=complex)
            for column in range(16):
                for row, value in boundary_ladder(words, 4, vertex, {column: 1.}, raising).items():
                    operator[row, column] = value
            operators.append(operator)
        annihilation.append(operators[0])
        creation.append(operators[1])
    mixed = max(float(np.linalg.norm(annihilation[i] @ creation[j]
                                    + creation[j] @ annihilation[i]
                                    - (np.eye(16) if i == j else 0)))
                for i in range(4) for j in range(4))
    nilpotent = max(float(np.linalg.norm(annihilation[i] @ annihilation[j]
                                       + annihilation[j] @ annihilation[i]))
                   for i in range(4) for j in range(4))
    # Two physical cells, one neutral hidden leaf, two genuine quantum EM rotors.
    small = EdgeWords(3, [(0, 1), (0, 2)])
    creator = np.zeros((4, 4), dtype=complex)
    for column in range(4):
        for row, value in boundary_ladder(small, 2, 1, {column: 1.}).items():
            creator[row, column] = value
    electric = np.diag([-1., 0., 1.])
    lower = np.diag(np.ones(2), 1)
    n = [np.diag([int(small.occupied(bits, vertex)) for bits in range(4)])
         for vertex in range(3)]
    electric_internal = np.kron(np.eye(4), np.kron(electric, np.eye(3)))
    electric_boundary = np.kron(np.eye(4), np.kron(np.eye(3), electric))
    gauss = [electric_internal + electric_boundary - np.kron(n[0], np.eye(9)),
             -electric_internal - np.kron(n[1], np.eye(9))]
    bare = np.kron(creator, np.eye(9))
    dressed = np.kron(creator, np.kron(lower, lower.conj().T))
    gauss_error = max(float(np.linalg.norm(g @ dressed - dressed @ g)) for g in gauss)
    bare_violation = max(float(np.linalg.norm(g @ bare - bare @ g)) for g in gauss)
    q = -charge
    physical_charge = q * np.kron(n[0] + n[1], np.eye(9))
    charge_error = float(np.linalg.norm(physical_charge @ dressed
                                       - dressed @ physical_charge - q * dressed))
    marker_mischarged = q * np.kron(n[0] + n[1] + n[2], np.eye(9))
    wrong_charge_error = float(np.linalg.norm(marker_mischarged @ dressed
                                             - dressed @ marker_mischarged - q * dressed))
    vacuum = np.zeros(36, dtype=complex)
    vacuum[small.tree_basis_state([]) * 9 + 4] = 1.
    charged = dressed @ vacuum
    state_gauss = max(float(np.linalg.norm(g @ charged)) for g in gauss)
    density_charge = float(np.vdot(charged, physical_charge @ charged).real)
    outgoing_flux_charge = float(q * np.vdot(charged, electric_boundary @ charged).real)
    recovery_error = float(np.linalg.norm(dressed.conj().T @ charged - vacuum))
    assert max(mixed, nilpotent, gauss_error, charge_error, state_gauss, recovery_error) < 1e-13
    assert bare_violation > 1 and wrong_charge_error > abs(q)
    assert abs(density_charge - q) < 1e-13 and abs(outgoing_flux_charge - q) < 1e-13
    return {'primitive_full_CAR_mixed_residual': mixed,
            'primitive_full_CAR_nilpotent_residual': nilpotent,
            'odd_and_even_physical_fock_dimension': 16,
            'physical_gauss_dressed_creation_commutator': gauss_error,
            'bare_creation_gauss_violation': bare_violation,
            'charged_state_gauss_residual': state_gauss,
            'actual_charge_density': density_charge, 'actual_boundary_flux_charge': outgoing_flux_charge,
            'same_charge_creation_commutator': charge_error,
            'mischarging_neutral_marker_rejected': wrong_charge_error,
            'annihilation_recovers_physical_vacuum_residual': recovery_error,
            'scope': 'Finite charged Gauss-physical string state, full matter parity sectors. '
                     'Not a Coulomb-minimized/dressed interacting asymptotic wavepacket.'}


class FilledSeaConsumer:
    """Full physical matter parity via a neutral hidden boundary marker."""
    def __init__(self, band, photons, kinematics):
        self.band, self.photons = band, photons
        self.p, self.k, self.pf, self.kf = kinematics
        self.omega, self.omegaf = photons.photon_energy(self.k), photons.photon_energy(self.kf)
        self.momenta, self.frames, energies = [], [], []
        self.offsets = []
        for momentum_vector in (self.p, self.p + self.k, self.p - self.kf, self.pf):
            match = next((index for index, previous in enumerate(self.momenta)
                          if np.allclose(previous, momentum_vector, rtol=0, atol=1e-12)), None)
            if match is None:
                match = len(self.momenta)
                self.momenta.append(momentum_vector)
                positive = np.column_stack([band.spinor(momentum_vector, spin) for spin in range(2)])
                negative = CHARGE_CONJUGATION @ np.column_stack(
                    [band.spinor(-momentum_vector, spin).conjugate() for spin in range(2)])
                frame = np.column_stack([negative, positive])
                energy = band.energy(momentum_vector)
                assert np.linalg.norm(frame.conj().T @ frame - np.eye(4)) < 2e-13
                assert np.linalg.norm(frame.conj().T @ band.h(momentum_vector) @ frame
                                      - np.diag([-energy, -energy, energy, energy])) < 2e-13
                self.frames.append(frame)
                energies.extend([-energy, -energy, energy, energy])
            self.offsets.append(4 * match)
        self.boundary = len(energies)
        self.energies = np.array(energies)
        self.words = EdgeWords(self.boundary + 1,
                               [*( (i, i + 1) for i in range(self.boundary - 1)),
                                (0, self.boundary)])
        self.negative = tuple(np.flatnonzero(self.energies < 0))
        self.vacuum_bits = self.words.tree_basis_state(self.negative)
        self.vacuum = {self.vacuum_bits: 1 + 0j}
        self.vacuum_energy = sum(self.energies[list(self.negative)])
        self._excitation_cache = {}

    def physical_occupation(self, state):
        return tuple(vertex for vertex in self.words.occupation(state) if vertex < self.boundary)

    def transformed(self, matrix, target_offset, source_offset):
        return (self.frames[target_offset // 4].conj().T @ matrix
                @ self.frames[source_offset // 4])

    def currents(self, eps_in, eps_out):
        initial, direct, crossed, final = self.offsets
        band, eo = self.band, np.conjugate(eps_out)
        vin = (block_terms(self.transformed(band.vertex(self.p + self.k, self.p, eps_in), direct, initial), direct, initial)
               + block_terms(self.transformed(band.vertex(self.pf, self.p - self.kf, eps_in), final, crossed), final, crossed))
        vout = (block_terms(self.transformed(band.vertex(self.pf, self.p + self.k, eo), final, direct), final, direct)
                + block_terms(self.transformed(band.vertex(self.p - self.kf, self.p, eo), crossed, initial), crossed, initial))
        contact = block_terms(self.transformed(band.contact(self.pf, self.p, eps_in, eo, self.k, self.kf), final, initial), final, initial)
        return vin, vout, contact

    def external(self, offset, spin):
        return boundary_ladder(self.words, self.boundary, offset + 2 + spin, self.vacuum)

    def excitation(self, state):
        if state not in self._excitation_cache:
            occupied = self.physical_occupation(state)
            energy = float(sum(self.energies[list(occupied)]) - self.vacuum_energy)
            self._excitation_cache[state] = energy
        return self._excitation_cache[state]

    def response(self, initial, final, first, second, total_matter_energy, pair_filter=None):
        intermediate = self.words.apply(first, initial)
        resolved = {}
        groups = {'electron': {}, 'electron_pair': {}}
        baseline_occupation = self.physical_occupation(next(iter(initial)))
        baseline_excitations = (sum(self.energies[vertex] > 0 for vertex in baseline_occupation)
                                + sum(vertex not in baseline_occupation for vertex in self.negative))
        for state, amplitude in intermediate.items():
            occupied = self.physical_occupation(state)
            positive_count = sum(self.energies[vertex] > 0 for vertex in occupied)
            holes = sum(vertex not in occupied for vertex in self.negative)
            group = ('electron' if positive_count + holes == baseline_excitations
                     else 'electron_pair')
            if pair_filter is not None and group != pair_filter:
                continue
            denominator = total_matter_energy - self.excitation(state)
            if abs(denominator) < 1e-12:
                raise ValueError('Selected kinematics encounters a finite-volume intermediate pole')
            resolved[state] = amplitude / denominator
            groups[group][state] = amplitude / denominator
        amplitude = overlap(final, self.words.apply(second, resolved))
        grouped = {name: overlap(final, self.words.apply(second, state)) for name, state in groups.items()}
        return amplitude, grouped, len(intermediate)

    def amplitude(self, eps_in, eps_out, spin, spin_out, keep_contact=True, keep_pairs=True):
        vin, vout, contact = self.currents(eps_in, eps_out)
        initial = self.external(self.offsets[0], spin)
        final = self.external(self.offsets[-1], spin_out)
        energy = self.band.energy(self.p)
        filt = None if keep_pairs else 'electron'
        absorption, abs_groups, abs_count = self.response(initial, final, vin, vout, energy + self.omega, filt)
        emission, emit_groups, emit_count = self.response(initial, final, vout, vin, energy - self.omegaf, filt)
        seagull = overlap(final, self.words.apply(contact, initial)) if keep_contact else 0j
        vacuum_disconnected = 0j
        external_overlap = overlap(final, initial)
        if abs(external_overlap) > 1e-14:
            vacuum_absorption, _, _ = self.response(self.vacuum, self.vacuum, vin, vout, self.omega, filt)
            vacuum_emission, _, _ = self.response(self.vacuum, self.vacuum, vout, vin, -self.omegaf, filt)
            vacuum_contact = overlap(self.vacuum, self.words.apply(contact, self.vacuum)) if keep_contact else 0j
            vacuum_disconnected = external_overlap * (vacuum_absorption + vacuum_emission + vacuum_contact)
        return {'connected': absorption + emission + seagull - vacuum_disconnected,
                'absorb_first_electron': abs_groups['electron'],
                'emit_first_electron': emit_groups['electron'],
                'absorb_first_electron_pair': abs_groups['electron_pair'],
                'emit_first_electron_pair': emit_groups['electron_pair'],
                'contact': seagull, 'vacuum_disconnected': vacuum_disconnected,
                'absorb_intermediate_states': abs_count, 'emit_intermediate_states': emit_count}

    def comparator(self, eps_in, eps_out, spin, spin_out):
        reference = self.photons.kernel(self.p, self.k, self.pf, self.kf, eps_in, eps_out)
        return np.vdot(self.photons.spinor(self.pf, spin_out), reference @ self.photons.spinor(self.p, spin))


def rest_spin_and_hole_probe(charge):
    """Lift rest-mode rotations onto the same primitive filled-sea Hilbert space."""
    band, photons = RotationBand(.025, e=charge, improved=True), WilsonLink(.025, e=charge, improved=True)
    consumer = FilledSeaConsumer(band, photons, photons.kinematics(.2, .7))
    words, rest = consumer.words, consumer.offsets[0]
    frame = consumer.frames[rest // 4]
    particles = [consumer.external(rest, spin) for spin in range(2)]
    holes = [boundary_ladder(words, consumer.boundary, rest + spin, consumer.vacuum, creation=False)
             for spin in range(2)]
    generators = [block_terms(frame.conj().T @ generator @ frame, rest, rest)
                  for generator in ROTATION]
    def restricted(terms, basis):
        return np.array([[overlap(bra, words.apply(terms, ket)) for ket in basis] for bra in basis])
    je = np.array([restricted(terms, particles) for terms in generators])
    jh = np.array([restricted(terms, holes) for terms in generators])
    expected_e = np.array([frame[:, 2:].conj().T @ generator @ frame[:, 2:]
                           for generator in ROTATION])
    expected_h = np.array([-(frame[:, :2].conj().T @ generator @ frame[:, :2]).T
                           for generator in ROTATION])
    electron_error = float(np.linalg.norm(je - expected_e))
    hole_error = float(np.linalg.norm(jh - expected_h))
    vacuum_spin = max(abs(overlap(consumer.vacuum, words.apply(terms, consumer.vacuum)))
                      for terms in generators)
    su2 = max(float(np.linalg.norm(j[i] @ j[k] - j[k] @ j[i] - 1j * j[l]))
              for j in (je, jh) for i, k, l in ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
    casimir = max(float(np.linalg.norm(sum(generator @ generator for generator in j)
                                      - .75 * np.eye(2))) for j in (je, jh))
    minus_one = max(float(np.linalg.norm(expm(-2j * np.pi * j[2]) + np.eye(2)))
                    for j in (je, jh))
    q = consumer.band.charge
    density_terms = [(vertex, vertex, q) for vertex in range(rest, rest + 4)]
    qe = restricted(density_terms, particles) - 2 * q * np.eye(2)
    qh = restricted(density_terms, holes) - 2 * q * np.eye(2)
    charge_error = max(float(np.linalg.norm(qe - q * np.eye(2))),
                       float(np.linalg.norm(qh + q * np.eye(2))))
    assert max(electron_error, hole_error, vacuum_spin, su2, casimir, minus_one, charge_error) < 2e-13
    return {'same_primitive_hilbert_space': True, 'electron_rotation_lift_residual': electron_error,
            'hole_negative_transpose_rotation_lift_residual': hole_error,
            'filled_rest_sea_spin_expectation': float(vacuum_spin),
            'su2_residual': su2, 'spin_half_casimir_residual': casimir,
            'electron_and_hole_two_pi_minus_identity_residual': minus_one,
            'electron_normal_ordered_charge': float(qe[0, 0].real),
            'hole_normal_ordered_charge': float(qh[0, 0].real),
            'charge_residual': charge_error,
            'scope': 'Rest-mode internal rotations; actual physical electron/hole odd sectors via a neutral boundary parity marker. '
                     'Not a proof of Coulomb-minimized asymptotic dressing or full coordinate rotations at finite lattice spacing.'}


def complex_fields(values):
    return {key: [float(value.real), float(value.imag)] if isinstance(value, complex) else value
            for key, value in values.items()}


def scattering_probe(a, momentum, theta, direction, charge):
    band = RotationBand(a, e=charge, improved=True)
    photons = WilsonLink(a, e=charge, improved=True)
    consumer = FilledSeaConsumer(band, photons, photons.kinematics(momentum, theta, direction))
    residual = pair_deletion = contact_deletion = ward = ward_no_pairs = ward_no_contact = 0.
    reference_squared = computed_squared = no_pair_squared = 0.
    witness = None
    for eps in photons.polarizations(consumer.k):
        for eps_out in photons.polarizations(consumer.kf):
            for spin in range(2):
                for spin_out in range(2):
                    values = consumer.amplitude(eps, eps_out, spin, spin_out)
                    reference = consumer.comparator(eps, eps_out, spin, spin_out)
                    no_pairs = consumer.amplitude(eps, eps_out, spin, spin_out, keep_pairs=False)['connected']
                    no_contact = consumer.amplitude(eps, eps_out, spin, spin_out, keep_contact=False)['connected']
                    residual = max(residual, abs(values['connected'] - reference))
                    pair_deletion = max(pair_deletion, abs(no_pairs - reference))
                    contact_deletion = max(contact_deletion, abs(no_contact - reference))
                    scale = 4 * band.energy(consumer.p) * band.energy(consumer.pf) / 4
                    computed_squared += scale * abs(values['connected'])**2
                    reference_squared += scale * abs(reference)**2
                    no_pair_squared += scale * abs(no_pairs)**2
                    if witness is None or abs(values['absorb_first_electron_pair']) + abs(values['emit_first_electron_pair']) > witness[0]:
                        witness = (abs(values['absorb_first_electron_pair']) + abs(values['emit_first_electron_pair']), values)
            for spin in range(2):
                for spin_out in range(2):
                    for gauge_in, gauge_out in ((np.r_[consumer.omega, photons.khat(consumer.k)], eps_out),
                                               (eps, np.r_[consumer.omegaf, photons.khat(consumer.kf)])):
                        ward = max(ward, abs(consumer.amplitude(gauge_in, gauge_out, spin, spin_out)['connected']))
                        ward_no_pairs = max(ward_no_pairs, abs(consumer.amplitude(gauge_in, gauge_out, spin, spin_out, keep_pairs=False)['connected']))
                        ward_no_contact = max(ward_no_contact, abs(consumer.amplitude(gauge_in, gauge_out, spin, spin_out, keep_contact=False)['connected']))
    assert residual < 2e-11 and ward < 2e-11
    assert pair_deletion > 1e-4 * charge**2
    assert abs(computed_squared - reference_squared) < 1e-10 * max(charge**4, reference_squared)
    assert witness is not None
    return {'a': a, 'momentum': momentum, 'theta': theta, 'direction': direction,
            'charge_magnitude': band.e, 'electron_charge': band.charge,
            'unique_momentum_blocks': len(consumer.momenta), 'bosonic_edge_qubits': len(consumer.words.edges),
            'max_connected_amplitude_reference_residual': residual,
            'max_ward_residual': ward, 'delete_virtual_pairs_ward_residual': ward_no_pairs,
            'delete_contact_ward_residual': ward_no_contact,
            'delete_virtual_pairs_amplitude_error': pair_deletion,
            'delete_contact_amplitude_error': contact_deletion,
            'pauli_spin_polarization_sum': computed_squared,
            'reference_spin_polarization_sum': reference_squared,
            'delete_virtual_pairs_spin_polarization_sum': no_pair_squared,
            'actual_pair_channel_witness': complex_fields(witness[1])}


def run(charge):
    rows = [scattering_probe(a, momentum, theta, direction, charge)
            for a, momentum, theta, direction in [(0.025, 0.2, 0.0, [0., 0., 1.]),
                                                 (0.025, 0.2, 0.7, [0., 0., 1.]),
                                                 (0.025, 0.2, 1.9, [0.3, -0.5, 0.8]),
                                                 (0.05, 0.7, 1.1, [0.4, 0.1, 0.9]),
                                                 (0.0125, 0.7, 2.4, [0., 0., 1.])]]
    return {'claim': 'Bosonic Pauli full-parity matter currents reproduce connected filled-sea tree Compton; virtual pairs supply negative poles.',
            'not_claimed': ['Dynamical selection of the isotropic hopping law or the commuting-code phase.',
                            'Equilibrium compact photon phase, asymptotic charged-state dressing, radiative completeness or parameter values.',
                            'Momentum-basis current terms are microscopic spatially local; locality is proved separately on decorated cubic graphs.'],
            'receipt': {'python': platform.python_version(), 'numpy': np.__version__,
                        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'dense_primitive_convention': dense_convention_probe(),
            'full_parity_boundary_algebra_and_charged_gauss_state': boundary_algebra_and_gauss_probe(charge),
            'bounded_spatial_penalty_and_band_support': [locality_probe(side) for side in (2, 3, 4)],
            'same_pauli_filled_sea_rest_spin_and_opposite_charge': rest_spin_and_hole_probe(charge),
            'connected_pauli_compton': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--charge-magnitude', type=float, default=0.303500410715123,
                        help='Positive e convention; physical electron charge is -e')
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    result = run(arguments.charge_magnitude)
    encoded = json.dumps(result, indent=2)
    if arguments.output is not None:
        arguments.output.write_text(encoded + '\n')
    print(encoded)

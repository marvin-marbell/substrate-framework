#!/usr/bin/env python3
"""Compact U(1) neutrality: exact domains, exhaustive two-loop blocks, grid route.

The accompanying note proves the continuum and covariant-grid comparisons.
A noncyclic Fourier-cube computation is evidence, not an angle-positivity proof.
All charges/fields are integers, angles dimensionless, hbar=1, UE and K energies.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True)
class SpatialGraph:
    """B has +1 at the oriented tail and -1 at the head; B E=q."""

    vertices: int
    edges: tuple[tuple[int, int], ...]
    tree: tuple[int, ...]

    def __post_init__(self):
        if self.vertices < 1 or len(self.tree) != self.vertices - 1:
            raise ValueError('Require a finite connected graph and a spanning tree')
        if len(set(self.tree)) != len(self.tree):
            raise ValueError('Repeated tree edge')
        if any(not (0 <= u < self.vertices and 0 <= v < self.vertices) or u == v
               for u, v in self.edges):
            raise ValueError('Invalid oriented edge')
        if any(not 0 <= k < len(self.edges) for k in self.tree):
            raise ValueError('Invalid tree index')
        # Removing leaves verifies the selected tree, including connectivity.
        self.particular((0,) * self.vertices)

    @property
    def incidence(self):
        matrix = np.zeros((self.vertices, len(self.edges)), dtype=np.int64)
        for k, (u, v) in enumerate(self.edges):
            matrix[u, k], matrix[v, k] = 1, -1
        return matrix

    @property
    def chords(self):
        return tuple(k for k in range(len(self.edges)) if k not in self.tree)

    def particular(self, charges):
        """Exact integer tree solution, with all chord fields zero."""
        charges = integer_charges(charges, self.vertices)
        remaining = list(charges)
        adjacency = [set() for _ in range(self.vertices)]
        for k in self.tree:
            u, v = self.edges[k]
            adjacency[u].add(k)
            adjacency[v].add(k)
        fields = np.zeros(len(self.edges), dtype=np.int64)
        queue = [v for v in range(self.vertices) if len(adjacency[v]) == 1]
        removed = 0
        while queue:
            vertex = queue.pop()
            if len(adjacency[vertex]) != 1:
                continue
            k = next(iter(adjacency[vertex]))
            u, v = self.edges[k]
            other = v if vertex == u else u
            sign = 1 if vertex == u else -1
            fields[k] = sign * remaining[vertex]
            remaining[other] += remaining[vertex]
            remaining[vertex] = 0
            adjacency[vertex].remove(k)
            adjacency[other].remove(k)
            removed += 1
            if len(adjacency[other]) == 1:
                queue.append(other)
        if removed != self.vertices - 1 or any(remaining):
            raise ValueError('Selected edges are not a spanning tree')
        if not np.array_equal(self.incidence @ fields, charges):
            raise AssertionError('Exact tree Gauss solve failed')
        return fields

    @property
    def cycles(self):
        columns = []
        for k in self.chords:
            u, v = self.edges[k]
            charge = [0] * self.vertices
            charge[u], charge[v] = -1, 1
            column = self.particular(charge)
            column[k] = 1
            columns.append(column)
        return np.column_stack(columns) if columns else np.zeros((len(self.edges), 0), dtype=np.int64)


def integer_charges(charges, vertices):
    if len(charges) != vertices:
        raise ValueError('One complete Gauss charge per vertex is required')
    if any(not isinstance(q, (int, np.integer)) for q in charges):
        raise ValueError('Compact U(1) characters require integer charges')
    charges = tuple(int(q) for q in charges)
    if sum(charges):
        raise ValueError('Complete internal Gauss characters must sum to zero')
    return charges


def two_plaquettes():
    """Open 3-by-2 vertices: four horizontal and three vertical links."""
    return SpatialGraph(6, ((0, 1), (1, 2), (3, 4), (4, 5),
                            (0, 3), (1, 4), (2, 5)), (2, 3, 4, 5, 6))


@dataclass(frozen=True)
class WilsonTerm:
    winding: tuple[int, ...]
    coefficient: float
    phase: float = 0.0

    def validate(self, cycles):
        if len(self.winding) != cycles or not any(self.winding):
            raise ValueError('Wilson winding must be a nonzero integer cycle vector')
        if any(not isinstance(n, (int, np.integer)) for n in self.winding):
            raise ValueError('Wilson windings must be integers')
        if not math.isfinite(self.coefficient) or not math.isfinite(self.phase):
            raise ValueError('Wilson potential must be real and bounded')


def electric_coefficients(electric, links):
    values = np.asarray(electric, dtype=float)
    if values.ndim == 0:
        values = np.full(links, float(values))
    if values.shape != (links,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('Every compact link requires a strictly positive UE')
    return values


def cube_blocks(graph, cutoff):
    """Enumerate EVERY field and complete charge; certify integer cycle coordinates."""
    if not isinstance(cutoff, int) or cutoff < 0:
        raise ValueError('A nonnegative integer noncyclic Fourier cutoff is required')
    links, base = len(graph.edges), 2 * cutoff + 1
    total = base ** links
    serial = np.arange(total, dtype=np.int64)
    fields = np.empty((total, links), dtype=np.int64)
    for k in range(links):
        fields[:, k] = serial // base ** k % base - cutoff
    charges, labels, counts = np.unique(fields @ graph.incidence.T, axis=0,
                                        return_inverse=True, return_counts=True)
    order = np.argsort(labels, kind='stable')
    fields = fields[order]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    cycles = graph.cycles
    if np.any(graph.incidence @ cycles):
        raise AssertionError('Integer cycle basis has nonzero complete Gauss charge')
    if not np.array_equal(cycles[list(graph.chords)], np.eye(len(graph.chords), dtype=int)):
        raise AssertionError('Cycle basis is not primitive over the integer lattice')
    for i, charge in enumerate(charges):
        block = fields[offsets[i]:offsets[i + 1]]
        particular = graph.particular(tuple(charge))
        loops = block[:, graph.chords]
        if not np.array_equal(block, particular + loops @ cycles.T):
            raise AssertionError('Complete Gauss block omits an integer cycle configuration')
    if int(counts.sum()) != total:
        raise AssertionError('Complete charge partition loses Fourier fields')
    return charges, fields, offsets


def cube_domain(graph, charges, cutoff):
    """Exact fixed-character domain, without enumerating unrelated Gauss blocks."""
    if not isinstance(cutoff, int) or cutoff < 0:
        raise ValueError('Require a nonnegative integer Fourier cutoff')
    particular = graph.particular(charges)
    rank = len(graph.chords)
    loops = np.asarray(list(itertools.product(range(-cutoff, cutoff + 1), repeat=rank)),
                       dtype=int).reshape((2 * cutoff + 1) ** rank, rank)
    fields = particular + loops @ graph.cycles.T
    return fields[np.all(abs(fields) <= cutoff, axis=1)]


def cube_operator(graph, fields, electric, terms):
    """Actual compressed Wilson shifts: missing targets are discarded, NEVER wrapped."""
    electric = electric_coefficients(electric, len(graph.edges))
    cycles = graph.cycles
    for term in terms:
        term.validate(cycles.shape[1])
    diagonal = np.sum(fields.astype(float) ** 2 * electric, axis=1) / 2
    index = {tuple(field): i for i, field in enumerate(fields)}
    row, col, data = list(range(len(fields))), list(range(len(fields))), list(diagonal.astype(complex))
    for term in terms:
        shift = cycles @ np.asarray(term.winding, dtype=int)
        amplitude = -term.coefficient / 2 * np.exp(1j * term.phase)
        for source, field in enumerate(fields):
            target = index.get(tuple(field + shift))
            if target is not None:
                row.extend((target, source))
                col.extend((source, target))
                data.extend((amplitude, np.conjugate(amplitude)))
    matrix = sparse.coo_matrix((data, (row, col)), shape=(len(fields), len(fields))).tocsr()
    if matrix.nnz and np.max(np.abs((matrix - matrix.conj().T).data), initial=0) > 1e-12:
        raise AssertionError('Noncyclic pure-photon operator is not Hermitian')
    return matrix


def ground(matrix):
    if matrix.shape[0] == 1:
        return float(matrix[0, 0].real)
    if matrix.shape[0] <= 100:
        return float(eigh(matrix.toarray(), subset_by_index=(0, 0), eigvals_only=True,
                          check_finite=False)[0])
    values = eigsh(matrix, k=1, which='SA', tol=0, return_eigenvectors=False,
                   v0=np.linspace(1, 2, matrix.shape[0]))
    return float(values[0])


def occupation_counts(charges, boundary):
    """Four orbitals/site, half-fill background 2; root external flux is boundary."""
    counts = tuple(int(q) + 2 + (boundary if i == 0 else 0)
                   for i, q in enumerate(charges))
    return counts if all(0 <= n <= 4 for n in counts) else None


def exhaustive_cube(cutoffs):
    graph = two_plaquettes()
    records = []
    cases = ((0.6, (0.25, 0.25)), (0.0072, (0.08 / 0.3 ** 2,) * 2),
             (0.11, (3.0, 3.0)), (0.6, (0.01, 2.0)))
    for cutoff in cutoffs:
        charges, fields, offsets = cube_blocks(graph, cutoff)
        neutral_index = int(np.flatnonzero(np.all(charges == 0, axis=1))[0])
        dimensions = np.diff(offsets)
        for electric, magnetic in cases:
            terms = tuple(WilsonTerm(tuple(int(i == j) for i in range(2)), k)
                          for j, k in enumerate(magnetic))
            neutral_fields = fields[offsets[neutral_index]:offsets[neutral_index + 1]]
            neutral = ground(cube_operator(graph, neutral_fields, electric, terms))
            minimum = (math.inf, None)
            counterexamples = []
            occupied = {b: {'patterns': 0, 'dimension_with_orbital_multiplicity': 0,
                            'minimum_charged_margin': None} for b in (-1, 0, 1)}
            # A common diagonal shift does not change the eigenvectors. Reuse only
            # exactly identical ordered diagonals and exactly identical shifts.
            spectra = {}
            for i, charge in enumerate(charges):
                block = fields[offsets[i]:offsets[i + 1]]
                if len(block) == 1:
                    energy = float(electric / 2 * np.sum(block[0] ** 2))
                else:
                    integer_diagonal = np.sum(block ** 2, axis=1)
                    shift = int(integer_diagonal.min())
                    loops = block[:, graph.chords]
                    lookup = {tuple(n): j for j, n in enumerate(loops)}
                    adjacency = tuple((j, lookup[tuple(n + np.eye(2, dtype=int)[p])], p)
                                      for j, n in enumerate(loops) for p in range(2)
                                      if tuple(n + np.eye(2, dtype=int)[p]) in lookup)
                    key = (tuple(integer_diagonal - shift), adjacency)
                    if key not in spectra:
                        matrix = cube_operator(graph, block, electric, terms)
                        spectra[key] = ground(matrix - electric * shift / 2 *
                                              sparse.eye(len(block), format='csr'))
                    energy = spectra[key] + electric * shift / 2
                margin = energy - neutral
                if i != neutral_index:
                    if margin < minimum[0]:
                        minimum = (margin, tuple(int(q) for q in charge))
                    if margin < -2e-10 * (1 + abs(energy) + abs(neutral)):
                        counterexamples.append({'charge': tuple(int(q) for q in charge),
                                                'dimension': len(block), 'margin': margin})
                for boundary, consumer in occupied.items():
                    counts = occupation_counts(charge, boundary)
                    if counts is not None:
                        consumer['patterns'] += 1
                        consumer['dimension_with_orbital_multiplicity'] += int(len(block)) * math.prod(
                            math.comb(4, n) for n in counts)
                        if i != neutral_index:
                            previous = consumer['minimum_charged_margin']
                            consumer['minimum_charged_margin'] = margin if previous is None else min(previous, margin)
            records.append({'R': cutoff, 'UE': electric, 'K_by_plaquette': magnetic,
                            'complete_charge_characters': len(charges),
                            'all_Fourier_fields': len(fields), 'neutral_dimension': len(neutral_fields),
                            'dimensions_histogram': dict(Counter(map(int, dimensions))),
                            'distinct_nontrivial_cached_operators': len(spectra),
                            'neutral_ground_energy': neutral,
                            'minimum_nonzero_charge_margin': minimum[0] if minimum[1] is not None else None,
                            'minimizing_complete_charge': minimum[1],
                            'common_continuum_reference_formula': neutral_reference_bound(
                                neutral, cutoff, electric, terms),
                            'charged_lower_than_neutral': counterexamples,
                            'actual_N11_N12_N13_occupations': occupied,
                            'status': 'counterexample' if counterexamples else 'all_enumerated_blocks_compared',
                            'universal_finite_Fourier_theorem': False})
    return records


def neutral_reference_bound(neutral_galerkin_energy, cutoff, minimum_electric, terms):
    """A block-Schur lower bound valid for every charged Galerkin domain."""
    potential_norm = sum(abs(term.coefficient) for term in terms)
    tail_lower = minimum_electric * (cutoff + 1) ** 2 / 2 - potential_norm
    lower = (neutral_galerkin_energy + tail_lower -
             math.hypot(tail_lower - neutral_galerkin_energy, 2 * potential_norm)) / 2
    lower = max(-potential_norm, lower)
    return {'neutral_continuum_lower_bound': lower,
            'neutral_Galerkin_upper_bound': neutral_galerkin_energy,
            'discarded_neutral_kinetic_lower_bound': tail_lower + potential_norm,
            'potential_norm_upper_bound': potential_norm,
            'all_charged_Galerkin_energies_at_least': lower,
            'floating_evaluation_is_not_interval_certified': True,
            'meaning': 'proved Schur/block-form inequality; floating evaluation uses numerical neutral Ritz energy'}


def grid_operator(graph, charges, sites, electric, terms, *, representative=None):
    """Exact-character quotient grid; covariant link differences, real V(phi)."""
    charges = integer_charges(charges, graph.vertices)
    if not isinstance(sites, int) or sites < 3:
        raise ValueError('At least three sites per quotient-circle are required')
    cycles, particular = graph.cycles, graph.particular(charges)
    if representative is not None:
        representative = np.asarray(representative)
        if representative.shape != particular.shape or not np.issubdtype(representative.dtype, np.integer):
            raise ValueError('The gauge-character representative must be an integer link field')
        if not np.array_equal(graph.incidence @ representative, charges):
            raise ValueError('The representative has the wrong exact Gauss charge')
        particular = representative
    rank = cycles.shape[1]
    for term in terms:
        term.validate(rank)
    electric = electric_coefficients(electric, len(graph.edges))
    dimension, h = sites ** rank, 2 * math.pi / sites
    points = np.asarray(list(itertools.product(range(sites), repeat=rank)), dtype=int).reshape(dimension, rank)
    angles = h * points
    potential = np.zeros(dimension)
    for term in terms:
        potential -= term.coefficient * np.cos(angles @ np.asarray(term.winding) + term.phase)
    row, col, data = list(range(dimension)), list(range(dimension)), list(potential.astype(complex))
    powers = sites ** np.arange(rank - 1, -1, -1, dtype=int)
    indices = np.arange(dimension)
    for link, coefficient in enumerate(electric):
        destination = ((points + cycles[link]) % sites) @ powers
        hopping = coefficient / (2 * h ** 2)
        phase = np.exp(-1j * h * particular[link])
        row.extend(indices)
        col.extend(indices)
        data.extend(np.full(dimension, 2 * hopping, dtype=complex))
        row.extend(destination)
        col.extend(indices)
        data.extend(np.full(dimension, -hopping * phase, dtype=complex))
        row.extend(indices)
        col.extend(destination)
        data.extend(np.full(dimension, -hopping * np.conjugate(phase), dtype=complex))
    matrix = sparse.coo_matrix((data, (row, col)), shape=(dimension, dimension)).tocsr()
    if np.max(np.abs((matrix - matrix.conj().T).data), initial=0) > 1e-11:
        raise AssertionError('Covariant quotient-grid operator is not Hermitian')
    return matrix, points


def grid_consumer(sites_sequence):
    graph = two_plaquettes()
    electric = np.array((0.6, 0.4, 0.7, 0.3, 0.5, 0.9, 0.8))
    terms = (WilsonTerm((1, 0), 0.25), WilsonTerm((0, 1), 2.0),
             WilsonTerm((1, 1), -0.4, 0.37))
    charges = ((1, -1, 0, 0, 0, 0), (1, 0, 0, 0, 0, -1),
               (0, 1, 0, 0, -1, 0), (0, 0, 0, 0, 0, 0))
    galerkin = []
    for cutoff in (2, 4, 6):
        for charge in charges:
            fields = cube_domain(graph, charge, cutoff)
            galerkin.append({'R': cutoff, 'charge': charge, 'dimension': len(fields),
                             'ground_energy': ground(cube_operator(graph, fields, electric, terms))})
    continuum_ritz = {tuple(case['charge']): case['ground_energy'] for case in galerkin if case['R'] == 6}
    rng = np.random.default_rng(222091)
    records = []
    for sites in sites_sequence:
        neutral_matrix, points = grid_operator(graph, charges[-1], sites, electric, terms)
        epsilon = ground(neutral_matrix)
        cases = []
        for charge in charges:
            matrix, _ = grid_operator(graph, charge, sites, electric, terms)
            energy = ground(matrix)
            vector = rng.normal(size=len(points)) + 1j * rng.normal(size=len(points))
            vector /= np.linalg.norm(vector)
            form_margin = float(np.vdot(vector, matrix @ vector).real -
                                np.vdot(abs(vector), neutral_matrix @ abs(vector)).real)
            scale = 1 + abs(energy) + abs(epsilon)
            if energy < epsilon - 2e-9 * scale or form_margin < -2e-9 * scale:
                raise AssertionError('Exact-character grid diamagnetic comparison failed')
            particular = graph.particular(charge)
            h = 2 * math.pi / sites
            probe_mode = np.array((1, 2))
            plane_wave = np.exp(1j * h * (points @ probe_mode)) / math.sqrt(len(points))
            probe_fields = particular + graph.cycles @ probe_mode
            expected_kinetic = float(np.sum(electric * (1 - np.cos(h * probe_fields))) / h ** 2)
            expected_potential = -sum(term.coefficient * float(np.mean(
                np.cos(h * (points @ np.asarray(term.winding)) + term.phase))) for term in terms)
            symbol_residual = abs(float(np.vdot(plane_wave, matrix @ plane_wave).real) -
                                  expected_kinetic - expected_potential)
            if symbol_residual > 1e-9 * (1 + expected_kinetic + abs(expected_potential)):
                raise AssertionError('Constructed covariant grid has the wrong charged kinetic symbol')
            # Every reconstructed Fourier node has the SAME EXACT U(1) charge.
            modes = np.rint(np.fft.fftfreq(sites) * sites).astype(int)
            fourier = np.asarray(list(itertools.product(modes, repeat=2)))
            reconstructed = particular + fourier @ graph.cycles.T
            gauss_error = int(np.max(abs(reconstructed @ graph.incidence.T - np.asarray(charge))))
            if gauss_error:
                raise AssertionError('Quotient grid incorrectly identifies Gauss charges modulo its size')
            # Compare two integer tree representatives by an exact grid unitary.
            coordinate_shift = np.array((1, -2))
            multiplier = np.exp(1j * (2 * math.pi / sites) * (points @ coordinate_shift))
            shifted, _ = grid_operator(
                graph, charge, sites, electric, terms,
                representative=particular + graph.cycles @ coordinate_shift)
            transformed = sparse.diags(multiplier) @ shifted @ sparse.diags(multiplier.conjugate())
            equivalence_error = float(np.max(np.abs((transformed - matrix).data), initial=0))
            if equivalence_error > 1e-9:
                raise AssertionError('Changing integer cycle origin changes the physical grid spectrum')
            cases.append({'charge': charge, 'ground_energy': energy, 'neutral_margin': energy - epsilon,
                          'random_diamagnetic_form_margin': form_margin, 'exact_Gauss_residual': gauss_error,
                          'grid_minus_R6_same_operator_Ritz': energy - continuum_ritz[charge],
                          'actual_operator_plane_wave_symbol_residual': symbol_residual,
                          'integer_cycle_origin_operator_error': equivalence_error})
        records.append({'sites_per_cycle': sites, 'dimension': sites ** 2,
                        'UE_by_link': electric.tolist(), 'real_Wilson_terms': terms_json(terms),
                        'neutral_ground_energy': epsilon, 'cases': cases})
    # Low fixed Fourier modes, not a falsely uniform statement about all charges.
    symbol_errors = []
    particular = graph.particular(charges[0])
    for sites in sites_sequence:
        modes = np.asarray(list(itertools.product(range(-2, 3), repeat=2)))
        fields = particular + modes @ graph.cycles.T
        exact = np.sum(electric * fields ** 2, axis=1) / 2
        h = 2 * math.pi / sites
        discrete = np.sum(electric * (1 - np.cos(h * fields)), axis=1) / h ** 2
        symbol_errors.append({'sites_per_cycle': sites,
                              'maximum_fixed_mode_kinetic_error': float(np.max(abs(discrete - exact))),
                              'bound_h2_sum_UE_field4_over24': float(h ** 2 * np.max(
                                  np.sum(electric * fields ** 4, axis=1)) / 24)})
    return {'comparison': 'proved finite-dimensional diamagnetic inequality',
            'continuum_limit': 'fixed charge and Wilson coefficients; see form-convergence proof',
            'cases': records, 'fixed_mode_symbol_consistency': symbol_errors,
            'independent_noncyclic_same_operator_Galerkin_sequence': galerkin}



def terms_json(terms):
    return [{'winding': term.winding, 'K': term.coefficient, 'phase': term.phase} for term in terms]


def adverse_controls():
    graph = two_plaquettes()
    # Smallest nontrivial Fourier truncation: projected free heat kernel at pi.
    time, electric = 0.1, 1.0
    negative_kernel = (1 - 2 * math.exp(-time * electric / 2)) / (2 * math.pi)
    if negative_kernel >= 0:
        raise AssertionError('Fourier angle-positivity countercontrol is not discriminating')
    cube = np.array((0, -1, 0, 1, 0, 1, -1))
    wrapped = (cube + graph.cycles[:, 0] + 1) % 3 - 1
    charge_change = graph.incidence @ (wrapped - cube)
    if not np.any(charge_change) or np.any(charge_change % 3):
        raise AssertionError('Cyclic electric wrap does not expose exact/modular Gauss mismatch')
    rejected = []
    for name, charge in (('fractional', (0.5, -0.5, 0, 0, 0, 0)),
                         ('uncompensated_boundary', (1, 0, 0, 0, 0, 0))):
        try:
            graph.particular(charge)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Invalid complete compact-Gauss charge was accepted')
    return {'R1_free_projected_heat_kernel_at_pi': negative_kernel, 'heat_time': time, 'UE': electric,
            'cyclic_R1_wrap_exact_Gauss_change': charge_change.tolist(),
            'cyclic_R1_wrap_mod3_Gauss_change': (charge_change % 3).tolist(),
            'rejected_character_controls': rejected,
            'scope': 'These invalidate angle-positivity/modular-Gauss shortcuts, NOT the finite ground bound.'}


def one_plaquette_control():
    # This imports the established one-loop consumer without the coupled solver.
    from compact_charge_gap import neutral_ground, rotor_spectrum
    records = []
    for cutoff in (1, 2, 3):
        for electric, magnetic in ((0.6, 0.25), (0.11, 3.0)):
            _, neutral, _ = neutral_ground(cutoff, electric, magnetic)
            offsets = set()
            for raw in itertools.product(range(-cutoff, cutoff + 1), repeat=4):
                offsets.add(tuple(e - raw[0] for e in raw))
            minimum = math.inf
            for offset in offsets:
                energy = float(rotor_spectrum(offset, cutoff, electric, magnetic)[0])
                minimum = min(minimum, energy - neutral)
            if minimum < -2e-10 * (1 + abs(neutral)):
                raise AssertionError('Established exact one-loop Holder positive control failed')
            records.append({'R': cutoff, 'UE': electric, 'K': magnetic,
                            'complete_integer_affine_domains': len(offsets),
                            'minimum_ground_margin_including_neutral': minimum,
                            'positive_control_source_sha256': hashlib.sha256(
                                Path(__file__).with_name('compact_charge_gap.py').read_bytes()).hexdigest()})
    return records


def exact_counterexample_certificate(graph, neutral_fields, charged_fields, terms, threshold):
    """Rational LDL lower bound and rational charged Rayleigh upper bound."""
    def rational_matrix(fields):
        size = len(fields)
        result = [[Fraction(0) for _ in range(size)] for _ in range(size)]
        lookup = {tuple(field): i for i, field in enumerate(fields)}
        for i, field in enumerate(fields):
            result[i][i] = Fraction(int(np.sum(field ** 2)), 200)
        for term in terms:
            if term.phase:
                raise ValueError('Exact certificate requires rational cosine coefficients, zero phases')
            shift = graph.cycles @ np.asarray(term.winding)
            coefficient = -Fraction(str(term.coefficient)) / 2
            for source, field in enumerate(fields):
                target = lookup.get(tuple(field + shift))
                if target is not None:
                    result[target][source] += coefficient
                    result[source][target] += coefficient
        return result

    neutral = rational_matrix(neutral_fields)
    charged = rational_matrix(charged_fields)
    threshold = Fraction(math.floor(threshold * 1_000_000), 1_000_000)
    size = len(neutral)
    lower = [[Fraction(int(i == j)) for j in range(size)] for i in range(size)]
    pivots = []
    for i in range(size):
        diagonal = neutral[i][i] - threshold - sum(lower[i][k] ** 2 * pivots[k] for k in range(i))
        if diagonal <= 0:
            raise AssertionError('Exact rational neutral lower-bound certificate failed')
        pivots.append(diagonal)
        for j in range(i + 1, size):
            lower[j][i] = (neutral[j][i] - sum(lower[j][k] * lower[i][k] * pivots[k]
                                              for k in range(i))) / diagonal
    numeric = np.array([[float(x) for x in row] for row in charged])
    _, vectors = eigh(numeric, subset_by_index=(0, 0), check_finite=False)
    vector = [Fraction(round(float(x) * 1_000_000), 1_000_000) for x in vectors[:, 0]]
    norm = sum(x ** 2 for x in vector)
    rayleigh = sum(vector[i] * charged[i][j] * vector[j]
                   for i in range(len(vector)) for j in range(len(vector))) / norm
    if rayleigh >= threshold:
        raise AssertionError('Exact rational charged upper-bound certificate failed')
    return {'threshold': str(threshold), 'neutral_H_minus_threshold_LDL_pivots': [str(x) for x in pivots],
            'charged_trial_vector': [str(x) for x in vector], 'charged_Rayleigh_quotient': str(rayleigh),
            'strict_order': 'E_charged <= rational Rayleigh < threshold < E_neutral',
            'arithmetic': 'exact Python Fraction, not floating eigenvalue tolerance'}


def search_real_wilson_counterexample(trials):
    """Bounded search is labelled honestly; every returned witness is re-evaluated."""
    graph = two_plaquettes()
    charges, fields, offsets = cube_blocks(graph, 1)
    neutral_index = int(np.flatnonzero(np.all(charges == 0, axis=1))[0])
    neutral = fields[offsets[neutral_index]:offsets[neutral_index + 1]]
    candidates = [i for i, size in enumerate(np.diff(offsets)) if size >= 2 and i != neutral_index]
    harmonics = tuple((a, b) for a in range(-2, 3) for b in range(-2, 3)
                      if (a > 0 or a == 0 and b > 0) and abs(a - b) <= 2)
    rng = np.random.default_rng(2221001)
    minimum = (math.inf, None)
    for trial in range(trials):
        coefficients = rng.integers(-4, 5, len(harmonics))
        terms = tuple(WilsonTerm(winding, float(k)) for winding, k in zip(harmonics, coefficients))
        electric = 0.01
        epsilon = ground(cube_operator(graph, neutral, electric, terms))
        for index in candidates:
            block = fields[offsets[index]:offsets[index + 1]]
            energy = ground(cube_operator(graph, block, electric, terms))
            margin = energy - epsilon
            if margin < minimum[0]:
                minimum = (margin, tuple(int(q) for q in charges[index]))
            if margin < -1e-7:
                witness = {'R': 1, 'UE': electric, 'trial': trial,
                           'charge': tuple(int(q) for q in charges[index]),
                           'neutral_dimension': len(neutral), 'charged_dimension': len(block),
                           'neutral_fields': neutral.tolist(), 'charged_fields': block.tolist(),
                           'terms': terms_json(terms), 'neutral_energy': epsilon,
                           'charged_energy': energy, 'charged_minus_neutral': margin}
                witness['exact_certificate'] = exact_counterexample_certificate(
                    graph, neutral, block, terms, (energy + epsilon) / 2)
                # R0 has only one zero-field/zero-character state, so R1 is the
                # smallest cutoff IF this search produces a strict counterexample.
                return {'status': 'strict_finite_Fourier_counterexample', 'smallest_nontrivial_R': 1,
                        'witness': witness, 'does_not_apply_to_positive_fundamental_terms': True}
    return {'status': 'not_found_in_bounded_search', 'trials': trials,
            'harmonics': harmonics, 'candidate_complete_characters': len(candidates),
            'minimum_observed_margin': minimum[0] if trials else None,
            'minimizing_character': minimum[1],
            'universal_absence_claim': False, 'constructive_alternative': 'exact-character covariant quotient grid'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--grid-sites', type=int, nargs='+', default=[9, 17, 33])
    parser.add_argument('--counterexample-trials', type=int, default=256)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=('all', 'cube', 'grid', 'counterexample', 'controls'), default='all')
    args = parser.parse_args()
    if args.counterexample_trials < 0:
        parser.error('Counterexample trials must be nonnegative')
    result = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runtime': {'python_executable': sys.executable, 'numpy_version': np.__version__},
              'units': {'hbar': 1, 'angles_fields_charges': 'dimensionless; integer field and character labels',
                        'UE_K_ground_energies': 'same energy unit', 'heat_time': 'inverse energy'},
              'graph': {'vertices': 6, 'edges': two_plaquettes().edges,
                        'tree': two_plaquettes().tree, 'integer_cycles': two_plaquettes().cycles.tolist()},
              'theorem': {'untruncated': 'H_ph(q)>=epsilon_0 I for every admissible complete integer character',
                          'boundary': 'add fixed external electric energy C_b; q=Q-b, sum q=0',
                          'not_claimed': 'full charged-fermion mass gap or finite Fourier angle positivity'}}
    if args.mode in ('all', 'controls'):
        result['adverse_controls'] = adverse_controls()
        result['one_plaquette_Holder_positive_control'] = one_plaquette_control()
    if args.mode in ('all', 'cube'):
        result['noncyclic_two_plaquette_all_characters'] = exhaustive_cube(args.cutoffs)
    if args.mode in ('all', 'grid'):
        result['positivity_preserving_finite_route'] = grid_consumer(args.grid_sites)
    if args.mode in ('all', 'counterexample'):
        result['real_Wilson_finite_cube_search'] = search_real_wilson_counterexample(args.counterexample_trials)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()

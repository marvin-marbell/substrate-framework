"""Primitive edge-qubit statistics supplier; direct execution prints run() as JSON.

The model builds only tensor-product Pauli operators. Fermion occupation actions
occur solely in the explicitly separated reference comparator near the end.
Dense matrices are for small exact probes, not a scalable simulation backend.
See bosonic_statistics.md for signs, sectors, locality and finite-penalty limits.
"""
from __future__ import annotations

import itertools
import json
from collections import deque

import numpy as np
from scipy.linalg import expm

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}


def tensor_pauli(word):
    """Edge 0 is the least significant computational-basis bit."""
    result = np.ones((1, 1), dtype=complex)
    for letter in reversed(word):
        result = np.kron(result, PAULI[letter])
    return result


def residual(matrix):
    return float(np.linalg.norm(matrix))


def complex_json(value):
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def primitive_pauli_words(edges, incident, parity=1, boundary_vertex=0):
    """Local primitive specification, reusable without exponential dense matrices.

    Edges and incident edge indices are already ordered by the graph constructor.
    This is the same bosonic law as BosonicEdgeModel, not a second CAR convention.
    """
    b_words = []
    for vertex, indices in enumerate(incident):
        word = ["I"] * len(edges)
        for edge in indices:
            word[edge] = "Z"
        b_words.append((parity if vertex == boundary_vertex else 1, tuple(word)))
    a_words = {}
    for edge, (source, target) in enumerate(edges):
        word = ["I"] * len(edges)
        word[edge] = "X"
        for vertex in (source, target):
            for earlier in incident[vertex]:
                if earlier >= edge:
                    break
                word[earlier] = "I" if word[earlier] == "Z" else "Z"
        a_words[(source, target)] = tuple(word)
    return b_words, a_words


class BosonicEdgeModel:
    """Connected simple graph with one bosonic qubit per edge.

    Local edge order is sorted edge index. A(i,j) has positive orientation i<j.
    parity=-1 supplies an explicit fixed boundary parity background, NOT a local
    fermion creation operator. Full edge-qubit matrices are exponentially large.
    A_words/B_words are inspectable primitive support specifications.
    """

    def __init__(self, vertices, edges, parity=1, boundary_vertex=0):
        self.vertices = int(vertices)
        if self.vertices < 1 or parity not in (-1, 1):
            raise ValueError("Positive vertex count and parity +/-1 required")
        self.edges = tuple(sorted((min(i, j), max(i, j)) for i, j in edges))
        if len(set(self.edges)) != len(self.edges) or any(
                i == j or i < 0 or j >= self.vertices for i, j in self.edges):
            raise ValueError("Graph must be simple, with vertices 0..vertices-1")
        if not 0 <= boundary_vertex < self.vertices:
            raise ValueError("Boundary vertex is outside the graph")
        self.parity = parity
        self.boundary_vertex = boundary_vertex
        self.edge_index = {edge: e for e, edge in enumerate(self.edges)}
        self.incident = tuple(tuple(e for e, edge in enumerate(self.edges) if i in edge)
                              for i in range(self.vertices))
        self.dimension = 1 << len(self.edges)
        self.identity = np.eye(self.dimension, dtype=complex)
        self.B_words, self.A_words = primitive_pauli_words(
            self.edges, self.incident, parity, boundary_vertex)
        self.B = [sign * tensor_pauli(word) for sign, word in self.B_words]
        self._A = {edge: tensor_pauli(word) for edge, word in self.A_words.items()}
        self._X = {}
        for e, edge in enumerate(self.edges):
            bare = ["I"] * len(self.edges)
            bare[e] = "X"
            self._X[edge] = tensor_pauli(bare)
        self.cycles = self._fundamental_cycles()
        self.stabilizers = tuple(self.cycle_operator(cycle) for cycle in self.cycles)

    def _fundamental_cycles(self):
        adjacency = [[] for _ in range(self.vertices)]
        for i, j in self.edges:
            adjacency[i].append(j)
            adjacency[j].append(i)
        parent: dict[int, int | None] = {0: None}
        queue = deque([0])
        tree_edges = set()
        while queue:
            i = queue.popleft()
            for j in sorted(adjacency[i]):
                if j not in parent:
                    parent[j] = i
                    tree_edges.add((min(i, j), max(i, j)))
                    queue.append(j)
        if len(parent) != self.vertices:
            raise ValueError("Use one connected component per BosonicEdgeModel")
        tree_adjacency = [[] for _ in range(self.vertices)]
        for i, j in tree_edges:
            tree_adjacency[i].append(j)
            tree_adjacency[j].append(i)
        cycles = []
        for i, j in self.edges:
            if (i, j) in tree_edges:
                continue
            paths: deque[tuple[int, tuple[int, ...]]] = deque([(j, (j,))])
            visited = {j}
            while paths:
                vertex, path = paths.popleft()
                if vertex == i:
                    cycles.append((i, *path[:-1]))
                    break
                for neighbor in sorted(tree_adjacency[vertex]):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        paths.append((neighbor, (*path, neighbor)))
        return tuple(cycles)

    def A(self, source, target, dressed=True):
        edge = (min(source, target), max(source, target))
        sign = 1 if source < target else -1
        return sign * (self._A[edge] if dressed else self._X[edge])

    def n(self, vertex):
        return (self.identity - self.B[vertex]) / 2

    def hop(self, source, target, dressed=True):
        """Directed source->target transfer: -i A_st n_s (1-n_t)."""
        return -1j * self.A(source, target, dressed) @ self.n(source) @ (
            self.identity - self.n(target))

    def bilinear(self, target, source, path=None):
        """Even target-creation/source-annihilation; explicit paths for nonedges."""
        if source == target:
            return self.n(source)
        if path is None:
            return self.hop(source, target)
        if len(path) < 2 or path[0] != source or path[-1] != target:
            raise ValueError("Bilinear path must run source to target")
        return -1j * self.path_A(path) @ self.n(source) @ (
            self.identity - self.n(target))

    def quadratic(self, one_particle, modes=None):
        """Linear even-algebra lift; nonzero off-diagonals require graph edges.

        For four orbitals at one spatial site, include its six internal edges.
        The supplied matrix can be Hermitian or a commutator; no spin matrix is
        assumed by this constructor. Same-link gauging remains explicit.
        """
        modes = tuple(range(self.vertices)) if modes is None else tuple(modes)
        matrix = np.asarray(one_particle, dtype=complex)
        if matrix.shape != (len(modes), len(modes)):
            raise ValueError("One-particle matrix must match the selected modes")
        result = np.zeros_like(self.identity)
        for target_index, target in enumerate(modes):
            for source_index, source in enumerate(modes):
                coefficient = matrix[target_index, source_index]
                if coefficient != 0:
                    result += coefficient * self.bilinear(target, source)
        return result

    def region_parity(self, modes):
        """(-1)^N_region; central for region-even observables, not crossing hops."""
        product = self.identity.copy()
        for mode in modes:
            product = product @ self.B[mode]
        return product

    def pair_creation(self, source, target):
        """Ordered pair: create source then target; changes charge by two units."""
        return -1j * self.A(source, target) @ (
            self.identity - self.n(source)) @ (self.identity - self.n(target))

    def path_A(self, path):
        """Endpoint A along a simple open path; code removes contractible path debt."""
        product = self.identity.copy()
        for i, j in zip(path[:-1], path[1:]):
            product = product @ self.A(i, j)
        return 1j ** (len(path) - 2) * product

    def cycle_operator(self, cycle):
        product = self.identity.copy()
        for i, j in zip(cycle, (*cycle[1:], cycle[0])):
            product = product @ self.A(i, j)
        return 1j ** len(cycle) * product

    def sector_projector(self, signs=None):
        if signs is None:
            signs = (1,) * len(self.stabilizers)
        if len(signs) != len(self.stabilizers) or any(s not in (-1, 1) for s in signs):
            raise ValueError("One +/-1 sign required per fundamental cycle")
        projector = self.identity.copy()
        for sign, stabilizer in zip(signs, self.stabilizers):
            projector = projector @ (self.identity + sign * stabilizer) / 2
        return projector

    def occupation_projector(self, occupied):
        occupied = set(occupied)
        projector = self.identity.copy()
        for i in range(self.vertices):
            projector = projector @ (self.n(i) if i in occupied else self.identity - self.n(i))
        return projector

    def state(self, occupied, signs=None):
        """Canonical rank-one joint occupation/loop state, built without a CAR map."""
        projector = self.sector_projector(signs) @ self.occupation_projector(occupied)
        diagonal = np.real(np.diag(projector))
        indices = np.flatnonzero(diagonal > 1e-10)
        if not len(indices):
            raise ValueError("Occupation is incompatible with fixed total parity")
        column = int(indices[0])
        return projector[:, column] / np.sqrt(diagonal[column])

    def occupation_basis(self, number=None, signs=None):
        masks, states = [], []
        numbers = range(self.vertices + 1) if number is None else (number,)
        for count in numbers:
            if (-1) ** count != self.parity:
                continue
            for occupied in itertools.combinations(range(self.vertices), count):
                masks.append(sum(1 << i for i in occupied))
                states.append(self.state(occupied, signs))
        return tuple(masks), (np.column_stack(states) if states else
                              np.empty((self.dimension, 0), dtype=complex))

    def loop_penalty(self, strength):
        result = np.zeros_like(self.identity)
        for stabilizer in self.stabilizers:
            result += strength * (self.identity - stabilizer) / 2
        return result

    def gauged_hop(self, source, target, phase):
        """Classical same-link phase; phase=+q*a*A for forward U-dagger."""
        return np.exp(1j * phase) * self.hop(source, target)

    def quantum_gauged_hop(self, source, target, link_lower):
        """Matter tensor link; [E,link_lower]=-link_lower for oriented electric flux."""
        return np.kron(self.hop(source, target), link_lower)

    def link_derivative(self, source, target, coefficient, phase=0.0, order=0):
        """Derivative of c exp(i phase) T + h.c.; order=0 is the link Hamiltonian."""
        forward = coefficient * (1j ** order) * self.gauged_hop(source, target, phase)
        return forward + forward.conj().T

    def number_current(self, source, target, coefficient, phase=0.0):
        return -self.link_derivative(source, target, coefficient, phase, order=1)

    def hamiltonian(self, coefficients, phases=None, onsite=None, penalty=0.0):
        phases = (0.0,) * len(self.edges) if phases is None else phases
        if len(coefficients) != len(self.edges) or len(phases) != len(self.edges):
            raise ValueError("Coefficients and phases must match the edge order")
        result = self.loop_penalty(penalty)
        for (i, j), coefficient, phase in zip(self.edges, coefficients, phases):
            result += self.link_derivative(i, j, coefficient, phase)
        if onsite is not None:
            if len(onsite) != self.vertices:
                raise ValueError("Onsite potentials must match the vertex count")
            for i, potential in enumerate(onsite):
                result += potential * self.n(i)
        return result

    def bare_X_transfer(self, source, target):
        """Number-conserving primitive-X defect, not a fermionic hopping operator."""
        edge = (min(source, target), max(source, target))
        return self._X[edge] @ self.n(source) @ (self.identity - self.n(target))


def algebra_report(model):
    identity = model.identity
    generators = [*model.B, *model._A.values()]
    involution = max(residual(a @ a - identity) for a in generators)
    hermitian = max(residual(a - a.conj().T) for a in generators)
    bb = max(residual(a @ b - b @ a) for a in model.B for b in model.B)
    ab, aa, hopping = [], [], []
    for i, j in model.edges:
        a = model.A(i, j)
        for k, b in enumerate(model.B):
            sign = -1 if k in (i, j) else 1
            ab.append(residual(a @ b - sign * b @ a))
        for k, l in model.edges:
            sign = (-1) ** len({i, j}.intersection((k, l)))
            b = model.A(k, l)
            aa.append(residual(a @ b - sign * b @ a))
        t = model.hop(i, j)
        initial = model.n(i) @ (identity - model.n(j))
        final = model.n(j) @ (identity - model.n(i))
        hopping.extend((residual(t.conj().T - model.hop(j, i)),
                        residual(t.conj().T @ t - initial),
                        residual(t @ t.conj().T - final)))
    product_b = identity.copy()
    for b in model.B:
        product_b = product_b @ b
    central = max((residual(s @ g - g @ s)
                   for s in model.stabilizers for g in generators), default=0.0)
    projector = model.sector_projector()
    return {
        "vertices": model.vertices, "edges": model.edges,
        "primitive_qubits": len(model.edges), "full_bosonic_dimension": model.dimension,
        "cycles": model.cycles, "independent_cycle_count": len(model.stabilizers),
        "expected_cycle_count": len(model.edges) - model.vertices + 1,
        "code_dimension": int(round(np.trace(projector).real)),
        "expected_code_dimension": 1 << (model.vertices - 1),
        "parity": model.parity, "A_words": {f"{i}->{j}": "".join(word)
                                             for (i, j), word in model.A_words.items()},
        "B_words": [[sign, "".join(word)] for sign, word in model.B_words],
        "involution_residual": involution, "hermitian_residual": hermitian,
        "BB_commutation_residual": bb, "AB_sign_residual": max(ab, default=0.0),
        "AA_sign_residual": max(aa, default=0.0),
        "directed_hop_projector_residual": max(hopping, default=0.0),
        "fixed_parity_residual": residual(product_b - model.parity * identity),
        "stabilizer_central_residual": central,
        "cycle_code_identity_residual": max((residual((s - identity) @ projector)
                                               for s in model.stabilizers), default=0.0),
        "projector_residual": residual(projector @ projector - projector),
        "forbidden_parity_projector_norm": residual(model.occupation_projector(
            (0,) if model.parity == 1 else ())),
    }


# REFERENCE COMPARATOR ONLY. No function above uses a fermion occupation action.
def reference_hop(vertices, masks, source, target, fermionic=True):
    """Independent occupation-basis comparator (CAR signs or hard-core bosons)."""
    lookup = {mask: i for i, mask in enumerate(masks)}
    result = np.zeros((len(masks), len(masks)), dtype=complex)
    for column, mask in enumerate(masks):
        if not (mask & (1 << source)) or mask & (1 << target):
            continue
        removed = mask ^ (1 << source)
        output = removed | (1 << target)
        sign = 1
        if fermionic:
            swaps = (mask & ((1 << source) - 1)).bit_count()
            swaps += (removed & ((1 << target) - 1)).bit_count()
            sign = (-1) ** swaps
        if output in lookup:
            result[lookup[output], column] = sign
    return result


def reference_alignment(model, number):
    """Evidence-only phase alignment of primitive joint states to the comparator."""
    masks, basis = model.occupation_basis(number)
    phases = np.zeros(len(masks), dtype=complex)
    primitive = [basis.conj().T @ model.hop(i, j) @ basis for i, j in model.edges]
    reference = [reference_hop(model.vertices, masks, i, j) for i, j in model.edges]
    for root in range(len(masks)):
        if abs(phases[root]) > 0:
            continue
        phases[root] = 1
        queue = deque([root])
        while queue:
            column = queue.popleft()
            for encoded, expected in zip(primitive, reference):
                for q, f in ((encoded, expected), (encoded.conj().T, expected.conj().T)):
                    for row in np.flatnonzero(np.abs(f[:, column]) > 0.5):
                        if abs(phases[row]) == 0:
                            phases[row] = q[row, column] * phases[column] / f[row, column]
                            queue.append(int(row))
    aligned = basis * phases[None, :]
    error = max((residual(model.hop(i, j) @ aligned - aligned @ f)
                 for (i, j), f in zip(model.edges, reference)), default=0.0)
    return masks, aligned, reference, error


def exchange_report(model, occupied, moves, signs=None, dressed=True):
    """Closed exchange both as directed hops and coherent pi/2 link pulses."""
    initial = model.state(occupied, signs)
    state = initial.copy()
    pulse_state = initial.copy()
    intermediate = []
    for source, target in moves:
        hop = model.hop(source, target, dressed)
        state = hop @ state
        pulse_state = expm(0.5j * np.pi * (hop + hop.conj().T)) @ pulse_state
        intermediate.append([float(np.vdot(state, model.n(i) @ state).real)
                             for i in range(model.vertices)])
    pulse_factor = 1j ** len(moves)
    masks = tuple(mask for mask in range(1 << model.vertices)
                  if mask.bit_count() == len(occupied))
    boson = np.zeros(len(masks), dtype=complex)
    column = masks.index(sum(1 << i for i in occupied))
    boson[column] = 1
    for source, target in moves:
        boson = reference_hop(model.vertices, masks, source, target, False) @ boson
    projector = model.sector_projector(signs)
    amplitude = np.vdot(initial, state)
    coherent_amplitude = np.vdot(initial, pulse_state) / pulse_factor
    return {
        "occupied_initial": occupied, "moves_in_time_order": moves,
        "loop_signs": [1] * len(model.cycles) if signs is None else signs,
        "dressed": dressed, "hop_return_amplitude": complex_json(amplitude),
        "closed_hop_state_residual": residual(state - amplitude * initial),
        "coherent_pulse_return_amplitude_without_common_pulse_phase": complex_json(coherent_amplitude),
        "coherent_return_probability": float(abs(np.vdot(initial, pulse_state)) ** 2),
        "coherent_loop_leakage": float(np.vdot(pulse_state, (model.identity - projector) @ pulse_state).real),
        "hard_core_boson_return_amplitude": complex_json(boson[column]),
        "intermediate_occupations": intermediate,
    }


def coherent_report(model, number, occupied):
    coefficients = tuple(-value for value in (1.0, 0.73, 1.21, 0.92)[:len(model.edges)])
    phases = (0.23, -0.41, 0.37, -0.29)[:len(model.edges)]
    onsite = (0.17, -0.08, 0.13, -0.04)[:model.vertices]
    h = model.hamiltonian(coefficients, phases, onsite)
    masks, aligned, reference, transfer_error = reference_alignment(model, number)
    expected = np.zeros((len(masks), len(masks)), dtype=complex)
    boson_h = expected.copy()
    for (i, j), coefficient, phase, f in zip(model.edges, coefficients, phases, reference):
        term = coefficient * np.exp(1j * phase) * f
        expected += term + term.conj().T
        b = coefficient * np.exp(1j * phase) * reference_hop(model.vertices, masks, i, j, False)
        boson_h += b + b.conj().T
    potentials = np.array([sum(onsite[i] for i in range(model.vertices) if mask & (1 << i))
                           for mask in masks])
    expected += np.diag(potentials)
    boson_h += np.diag(potentials)
    initial_ref = np.zeros(len(masks), dtype=complex)
    initial_ref[masks.index(sum(1 << i for i in occupied))] = 1
    time = 0.73
    primitive_state = expm(-1j * time * h) @ (aligned @ initial_ref)
    reference_state = expm(-1j * time * expected) @ initial_ref
    transformed = aligned.conj().T @ primitive_state
    first, second = [], []
    continuity = []
    for (i, j), coefficient, phase, f in zip(model.edges, coefficients, phases, reference):
        for order, errors in ((1, first), (2, second)):
            term = coefficient * (1j ** order) * np.exp(1j * phase) * f
            derivative = term + term.conj().T
            errors.append(residual(model.link_derivative(i, j, coefficient, phase, order) @ aligned -
                                   aligned @ derivative))
        local_h = model.link_derivative(i, j, coefficient, phase)
        current = model.number_current(i, j, coefficient, phase)
        continuity.extend((residual(1j * (local_h @ model.n(j) - model.n(j) @ local_h) - current),
                           residual(1j * (local_h @ model.n(i) - model.n(i) @ local_h) + current)))
    primitive_spectrum = np.linalg.eigvalsh(aligned.conj().T @ h @ aligned)
    expected_spectrum = np.linalg.eigvalsh(expected)
    return {
        "parity": model.parity, "number": number, "coefficients": coefficients,
        "same_link_phases": phases, "onsite": onsite, "time": time,
        "occupation_masks": masks, "directed_intertwiner_residual": transfer_error,
        "Hamiltonian_intertwiner_residual": residual(h @ aligned - aligned @ expected),
        "primitive_spectrum": primitive_spectrum.tolist(),
        "reference_spectrum": expected_spectrum.tolist(),
        "hard_core_boson_spectrum": np.linalg.eigvalsh(boson_h).tolist(),
        "spectral_residual": residual(primitive_spectrum - expected_spectrum),
        "coherent_state_transfer_residual": residual(primitive_state - aligned @ reference_state),
        "primitive_probabilities": (abs(transformed) ** 2).tolist(),
        "reference_probabilities": (abs(reference_state) ** 2).tolist(),
        "first_phase_derivative_transfer_residual": max(first),
        "second_phase_derivative_transfer_residual": max(second),
        "local_number_continuity_residual": max(continuity),
    }


def gauge_report(model):
    source, target = model.edges[0]
    electric = np.diag(np.arange(-2, 3, dtype=float))
    lower = np.diag(np.ones(4), k=1).astype(complex)
    rotor_identity = np.eye(5)
    transfer = model.quantum_gauged_hop(source, target, lower)
    source_gauss = np.kron(model.identity, electric) - np.kron(model.n(source), rotor_identity)
    target_gauss = -np.kron(model.identity, electric) - np.kron(model.n(target), rotor_identity)
    bare = np.kron(model.hop(source, target), rotor_identity)
    chi = np.array([0.19, -0.23, 0.41, -0.07])[:model.vertices]
    rotation = expm(1j * sum((value * model.n(i) for i, value in enumerate(chi)),
                             np.zeros_like(model.identity)))
    theta = 0.31
    covariance = rotation @ model.gauged_hop(source, target, theta) @ rotation.conj().T
    transformed = model.gauged_hop(source, target, theta + chi[target] - chi[source])
    return {
        "link_orientation": [source, target], "E_bounds": [-2, 2],
        "E_lower_commutator_residual": residual(electric @ lower - lower @ electric + lower),
        "source_Gauss_commutator_residual": residual(source_gauss @ transfer - transfer @ source_gauss),
        "target_Gauss_commutator_residual": residual(target_gauss @ transfer - transfer @ target_gauss),
        "bare_transfer_Gauss_commutator_norm": residual(source_gauss @ bare - bare @ source_gauss),
        "scalar_gauge_covariance_residual": residual(covariance - transformed),
        "rotor_lower_unitarity_defect": residual(lower.conj().T @ lower - rotor_identity),
        "cutoff_wrap_used": False,
    }


def finite_penalty_report(model):
    """Full microscopic sectors, plus a charge-preserving loop-changing X defect."""
    coefficients = (-1.0, -0.73, -1.21, -0.92)
    phases = (0.23, -0.41, 0.37, -0.29)
    onsite = (0.17, -0.08, 0.13, -0.04)
    h0 = model.hamiltonian(coefficients, phases, onsite)
    delta = 12.0
    full = h0 + model.loop_penalty(delta)
    sectors = []
    sector_spectra = []
    for signs in itertools.product((1, -1), repeat=len(model.cycles)):
        _, basis = model.occupation_basis(signs=signs)
        spectrum = np.linalg.eigvalsh(basis.conj().T @ full @ basis)
        sector_spectra.extend(spectrum)
        sectors.append({"loop_signs": signs, "dimension": basis.shape[1],
                        "penalty_energy": delta * signs.count(-1), "full_spectrum": spectrum.tolist()})
    projector = model.sector_projector()
    _, code = model.occupation_basis()
    code_spectrum = np.linalg.eigvalsh(code.conj().T @ h0 @ code)
    outside = model.identity - projector
    initial = model.state((0, 2))
    source, target = model.edges[1]
    bare_forward = model.bare_X_transfer(source, target)
    defect = bare_forward + bare_forward.conj().T
    number = sum((model.n(i) for i in range(model.vertices)), np.zeros_like(model.identity))
    coupling = 0.2
    time = 1.1
    finite = []
    for strength in (12.0, 24.0, 48.0):
        microscopic = h0 + model.loop_penalty(strength) + coupling * defect
        state = expm(-1j * time * microscopic) @ initial
        low_spectrum = np.linalg.eigvalsh(microscopic)[:code.shape[1]]
        effective = code.conj().T @ (h0 - coupling ** 2 / strength * defect @ defect) @ code
        finite.append({"penalty": strength, "defect_coupling": coupling, "time": time,
                       "loop_leakage_probability": float(np.vdot(state, outside @ state).real),
                       "low_spectrum_deviation_from_exact_code": residual(low_spectrum - code_spectrum),
                       "second_order_effective_spectral_residual": residual(
                           low_spectrum - np.linalg.eigvalsh(effective)),
                       "low_spectrum": low_spectrum.tolist()})
    bare_h = np.zeros_like(model.identity)
    for (i, j), coefficient, phase in zip(model.edges, coefficients, phases):
        term = coefficient * np.exp(1j * phase) * model.hop(i, j, dressed=False)
        bare_h += term + term.conj().T
    bare_state = expm(-1j * time * (bare_h + model.loop_penalty(delta))) @ initial
    return {
        "penalty": delta, "sectors": sectors,
        "full_spectrum": np.linalg.eigvalsh(full).tolist(),
        "sector_union_spectral_residual": residual(np.sort(sector_spectra) - np.linalg.eigvalsh(full)),
        "exact_code_invariance_residual": residual(outside @ full @ projector),
        "full_sector_separation_from_entire_code_band": float(
            min(s["full_spectrum"][0] for s in sectors if any(sign < 0 for sign in s["loop_signs"])) -
            max(sectors[0]["full_spectrum"])),
        "defect_edge": [source, target], "defect_charge_commutator_residual": residual(number @ defect - defect @ number),
        "defect_code_to_outside_norm": residual(outside @ defect @ projector),
        "bare_hopping_stabilizer_commutator_norm": max(residual(s @ bare_h - bare_h @ s)
                                                       for s in model.stabilizers),
        "bare_hopping_code_Hamiltonian_difference": residual(projector @ (bare_h - h0) @ projector),
        "bare_hopping_finite_penalty_leakage_probability": float(np.vdot(bare_state, outside @ bare_state).real),
        "finite_defect_dynamics": finite,
        "exact_faithfulness_requires_no_loop_mixing_defect": True,
    }


def run():
    star = BosonicEdgeModel(4, ((0, 1), (0, 2), (0, 3)))
    square = BosonicEdgeModel(4, ((0, 1), (1, 2), (2, 3), (3, 0)))
    odd = BosonicEdgeModel(4, square.edges, parity=-1, boundary_vertex=0)
    star_moves = ((1, 0), (0, 3), (2, 0), (0, 1), (3, 0), (0, 2))
    square_moves = ((0, 1), (2, 3), (1, 2), (3, 0))
    a = star.hop(1, 0) @ star.hop(0, 2) @ star.hop(3, 0)
    b = star.hop(3, 0) @ star.hop(0, 2) @ star.hop(1, 0)
    bare_a = star.hop(1, 0, False) @ star.hop(0, 2, False) @ star.hop(3, 0, False)
    bare_b = star.hop(3, 0, False) @ star.hop(0, 2, False) @ star.hop(1, 0, False)
    algebra = {"star": algebra_report(star), "square": algebra_report(square),
               "odd_boundary_square": algebra_report(odd)}
    exchange = {"star": exchange_report(star, (1, 2), star_moves),
                "star_removed_Z_tails": exchange_report(star, (1, 2), star_moves, dressed=False),
                "square_code": exchange_report(square, (0, 2), square_moves),
                "square_wrong_loop_sector": exchange_report(square, (0, 2), square_moves, signs=(-1,))}
    coherence = {"star_two_particles": coherent_report(star, 2, (1, 2)),
                 "square_two_particles": coherent_report(square, 2, (0, 2)),
                 "odd_square_single_particle": coherent_report(odd, 1, (0,))}
    penalty = finite_penalty_report(square)
    gauge = gauge_report(star)
    checks = {
        "primitive_local_algebra": all(max(report[key] for key in (
            "involution_residual", "hermitian_residual", "BB_commutation_residual",
            "AB_sign_residual", "AA_sign_residual", "directed_hop_projector_residual",
            "fixed_parity_residual", "stabilizer_central_residual", "cycle_code_identity_residual",
            "projector_residual", "forbidden_parity_projector_norm")) < 1e-10 for report in algebra.values()),
        "dimensions": all(report["code_dimension"] == report["expected_code_dimension"] and
                          report["independent_cycle_count"] == report["expected_cycle_count"]
                          for report in algebra.values()),
        "fermion_exchange_and_tail_adversary": all(abs(exchange[name]["hop_return_amplitude"]["real"] - sign) < 1e-10
            for name, sign in (("star", -1), ("star_removed_Z_tails", 1),
                               ("square_code", -1), ("square_wrong_loop_sector", 1))),
        "coherent_exchange": all(abs(exchange[name]["coherent_pulse_return_amplitude_without_common_pulse_phase"]["real"] + 1) < 1e-10
                                  for name in ("star", "square_code")),
        "full_dynamics_and_vertices_transfer": all(max(report[key] for key in (
            "directed_intertwiner_residual", "Hamiltonian_intertwiner_residual", "spectral_residual",
            "coherent_state_transfer_residual", "first_phase_derivative_transfer_residual",
            "second_phase_derivative_transfer_residual", "local_number_continuity_residual")) < 1e-10
            for report in coherence.values()),
        "microscopic_extra_sectors": penalty["sector_union_spectral_residual"] < 1e-10 and
            penalty["exact_code_invariance_residual"] < 1e-10 and
            penalty["full_sector_separation_from_entire_code_band"] > 0,
        "finite_defect_is_not_exact_encoding": penalty["defect_charge_commutator_residual"] < 1e-10 and
            all(row["loop_leakage_probability"] > 1e-9 and row["low_spectrum_deviation_from_exact_code"] > 1e-6
                for row in penalty["finite_defect_dynamics"]),
        "bare_hopping_failure_visible": penalty["bare_hopping_stabilizer_commutator_norm"] > 1e-3 and
            penalty["bare_hopping_code_Hamiltonian_difference"] > 1e-3,
        "same_link_gauge_transfer": max(gauge[key] for key in (
            "E_lower_commutator_residual", "source_Gauss_commutator_residual",
            "target_Gauss_commutator_residual", "scalar_gauge_covariance_residual")) < 1e-10 and
            gauge["bare_transfer_Gauss_commutator_norm"] > 1e-3,
        "Levin_Wen_triple": residual(a + b) < 1e-10 and residual(bare_a - bare_b) < 1e-10 and residual(a) > 0.5,
    }
    return {
        "status": "conditional_bosonic_statistics_supplier_not_spin_Dirac_or_photon_origin",
        "fundamental_inputs": ["complex tensor-product qubit quantum law", "chosen local Pauli Hamiltonian",
                               "loop-sector preparation or energetic selection", "charge/link variables and couplings"],
        "implementation_uses_CAR": False, "reference_comparator_uses_CAR": True,
        "algebra": algebra, "exchange": exchange, "coherence": coherence,
        "Levin_Wen_fermion_triple_residual": residual(a + b),
        "Levin_Wen_undressed_boson_triple_residual": residual(bare_a - bare_b),
        "gauge": gauge, "finite_penalty": penalty, "checks": checks,
        "all_checks_pass": all(checks.values()),
        "limits": ["No physical spin, Dirac mass, Lorentz restoration or Coulomb photon phase is derived here",
                   "Finite loop-mixing microscopic terms change exact code dynamics; corrections and leakage are reported",
                   "Arbitrary bounded degree makes hopping local, not every fundamental-cycle penalty short",
                   "Odd boundary parity is explicit input; no local single fermion creation operator is supplied"],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))

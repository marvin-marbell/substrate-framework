"""NEW finite cube candidate: local trace-matrix/sea and centre colour flux.
Not QCD, not the old global breather, and not an imposed trimer hopping model.
The exact 3-fundamental SU(3)-singlet CAR sector is Sym^3(space x spin x flavour):
the missing antisymmetry is carried by the explicit epsilon colour tensor.
Only the Z3 centre is locally gauged in this declared new regime.
"""
from __future__ import annotations
import itertools
import math
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy import sparse
from scipy.linalg import eigh


def cube() -> tuple[Any, list[tuple[int, int, int]], Any, Any, Any]:
    positions = np.array(list(itertools.product([0, 1], repeat=3)), dtype=float)
    edges = [(i, j, d) for i in range(8) for j in range(i+1, 8)
             for d in range(3) if np.sum(np.abs(positions[j]-positions[i])) == 1 and positions[j, d] != positions[i, d]]
    incidence = np.zeros((8, 12), dtype=int)
    for e, (i, j, _) in enumerate(edges):
        incidence[i, e], incidence[j, e] = 1, -1
    visited, tree = {0}, []
    while len(visited) < 8:
        for e, (i, j, _) in enumerate(edges):
            if (i in visited) != (j in visited):
                tree.append(e); visited.update([i, j])
    chords = [e for e in range(12) if e not in tree]
    loop = []
    for a, b in itertools.combinations(range(3), 2):
        c = next(d for d in range(3) if d not in [a, b])
        for fixed in [0, 1]:
            vertices = []
            for aa, bb in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                x = np.zeros(3); x[a], x[b], x[c] = aa, bb, fixed
                vertices.append(int(np.where(np.all(positions == x, axis=1))[0][0]))
            flow = np.zeros(12, dtype=int)
            for i, j in zip(vertices, vertices[1:]+vertices[:1]):
                e = next(e for e, (s, t, _) in enumerate(edges) if {s, t} == {i, j})
                flow[e] = 1 if i < j else -1
            loop.append(flow)
    return positions, edges, incidence, np.array(tree), np.array(loop)


@dataclass
class LocalCloud:
    p: dict[str, Any]

    def __post_init__(self) -> None:
        n = self.p['boson_levels']
        b = np.diag(np.sqrt(np.arange(1, n)), 1)
        x = (b+b.T)/np.sqrt(2)
        pair = np.diag(np.sqrt(np.arange(1, 7)*(7-np.arange(1, 7))), 1)
        self.X = np.kron(x, np.eye(7))
        x2 = x@x
        x2[-1, -1] += n/2  # P X^2 P, not the wrong square (P X P)^2.
        self.X2 = np.kron(x2, np.eye(7))
        self.sea = np.kron(np.eye(n), np.diag(np.arange(7)))
        self.phonon = np.kron(np.diag(np.arange(n)), np.eye(7))
        self.top = np.kron(np.diag(np.arange(n) == n-1), np.eye(7))
        bare = self.p['omega']*self.phonon + 2*self.p['sea_mass']*self.sea
        bare += self.p['sea_yukawa']*np.kron(x, pair+pair.T)
        self.vectors, self.energies, self.ops = [], [], []
        for occupation in range(4):
            e, v = eigh(bare+self.p['deformation_g']*occupation*self.X, subset_by_index=[0, 1])
            for col in range(2):
                if v[np.argmax(np.abs(v[:, col])), col] < 0:
                    v[:, col] *= -1
            self.vectors.append(v); self.energies.append(e)
            self.ops.append({name: v.T@op@v for name, op in [('X', self.X), ('X2', self.X2), ('sea', self.sea), ('phonon', self.phonon), ('top', self.top)]})
        self.overlaps = {(a, b): self.vectors[a].T@self.vectors[b] for a in range(4) for b in range(4)}
        self.bits = np.array([bits for bits in itertools.product([0, 1], repeat=8) if sum(bits) <= self.p['cloud_excitation_cap']], dtype=np.int8)
        self.lookup = {tuple(bits): k for k, bits in enumerate(self.bits)}
        self.template: dict[tuple[int, ...], tuple[Any, Any, Any, Any]] = {}
        self.cache: dict[tuple[Any, ...], Any] = {}

    def local_product(self, sites: tuple[int, ...], matrices: list[Any]) -> Any:
        if sites not in self.template:
            rr, cc, outputs, inputs = [], [], [], []
            for col, bits in enumerate(self.bits):
                for values in itertools.product([0, 1], repeat=len(sites)):
                    target = bits.copy(); target[list(sites)] = values
                    row = self.lookup.get(tuple(target))
                    if row is not None:
                        rr.append(row); cc.append(col); outputs.append(values); inputs.append(bits[list(sites)])
            self.template[sites] = (np.array(rr), np.array(cc), np.array(outputs), np.array(inputs))
        rr, cc, outputs, inputs = self.template[sites]
        data = np.ones(len(rr))
        for k, matrix in enumerate(matrices):
            data *= matrix[outputs[:, k], inputs[:, k]]
        return sparse.csr_matrix((data, (rr, cc)), shape=(len(self.bits), len(self.bits)))

    def hop(self, counts: tuple[int, ...], i: int, j: int) -> Any:
        key = ('hop', counts[i], counts[j], i, j)
        if key not in self.cache:
            self.cache[key] = self.local_product((i, j), [self.overlaps[counts[i]-1, counts[i]], self.overlaps[counts[j]+1, counts[j]]])
        return self.cache[key]

    def hamiltonian(self, counts: tuple[int, ...], edges: list[tuple[int, int, int]]) -> Any:
        diagonal = sum(self.energies[n][self.bits[:, s]] for s, n in enumerate(counts))
        h = sparse.diags(diagonal, format='csr')
        gradient = self.p['deformation_gradient']
        for i, j, _ in edges:
            h = h + gradient/2*(self.local_product((i,), [self.ops[counts[i]]['X2']]) + self.local_product((j,), [self.ops[counts[j]]['X2']]))
            h = h-gradient*self.local_product((i, j), [self.ops[counts[i]]['X'], self.ops[counts[j]]['X']])
        return h


@dataclass
class ColourFlux:
    p: dict[str, Any]

    def __post_init__(self) -> None:
        self.positions, self.edges, self.incidence, self.tree, loops = cube()
        self.chords = np.array([e for e in range(12) if e not in self.tree])
        self.f = np.array(list(itertools.product(range(3), repeat=5)), dtype=int)
        self.weights = 3**np.arange(4, -1, -1)
        self.inverse = np.rint(np.linalg.inv(self.incidence[1:, self.tree])).astype(int)
        self.plaquette = np.zeros((243, 243))
        for loop in loops:
            target = ((self.f+loop[self.chords]) % 3)@self.weights
            self.plaquette[target, np.arange(243)] -= self.p['colour_magnetic']/2
            self.plaquette[np.arange(243), target] -= self.p['colour_magnetic']/2
        self.cache: dict[tuple[int, ...], dict[str, Any]] = {}
        self.channels = 1 if self.p['colour_electric'] == 0 else 2

    def get(self, counts: tuple[int, ...]) -> dict[str, Any]:
        if counts not in self.cache:
            electric = np.zeros((243, 12), dtype=int)
            electric[:, self.chords] = self.f
            source = np.array(counts)[None, :]-self.f@self.incidence[:, self.chords].T
            electric[:, self.tree] = (source[:, 1:]@self.inverse.T) % 3
            if np.any((electric@self.incidence.T-np.array(counts)) % 3):
                raise ValueError('Literal local/global triality Gauss failed')
            cost = np.sum(electric != 0, axis=1)
            h = self.plaquette+np.diag(self.p['colour_electric']*cost)
            if self.channels == 1:
                v = np.ones((243, 1))/np.sqrt(243)
                ground = float((v.T@h@v)[0, 0])
            else:
                e, ground_vector = eigh(h, subset_by_index=[0, 0])
                first = ground_vector[:, 0]
                if np.sum(first) < 0:
                    first *= -1
                second = cost*first-float(first@(cost*first))*first
                length = np.linalg.norm(second)
                if length < 1e-12:
                    raise ValueError('True null electric response; cannot invent colour channel')
                v = np.column_stack([first, second/length]); ground = float(e[0])
            projected = v.T@h@v
            self.cache[counts] = dict(v=v, electric=electric, cost=cost, H=projected,
                electric_matrix=v.T@(cost[:, None]*v), magnetic_matrix=v.T@self.plaquette@v,
                primitive_ground=ground, exterior_H2=np.sum(np.abs(h@v-v@projected)**2, axis=0))
        return self.cache[counts]

    def hop(self, counts: tuple[int, ...], target: tuple[int, ...], edge: int, direction: int) -> Any:
        source_v, target_v = self.get(counts)['v'], self.get(target)['v']
        shifted = self.f.copy()
        if edge in self.chords:
            shifted[:, int(np.where(self.chords == edge)[0][0])] -= direction
        destination = (shifted % 3)@self.weights
        return target_v[destination].T@source_v


class Body:
    def __init__(self, p: dict[str, Any]):
        self.p = p
        self.positions, self.edges, _, _, _ = cube()
        self.states = [(a, b, d) for a in range(16) for b in range(a, 16) for d in range(16)]
        self.lookup = {state: i for i, state in enumerate(self.states)}
        self.counts = [tuple(np.bincount(np.array(state)//2, minlength=8)) for state in self.states]
        self.sigma = [np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]

    def move(self, state: tuple[int, int, int], flavour: int, source: int, destination: int) -> tuple[int, float] | None:
        if flavour == 1:
            if state[2] != source:
                return None
            return self.lookup[(state[0], state[1], destination)], 1.
        occupation = [state[0], state[1]]
        count = occupation.count(source)
        if count == 0:
            return None
        if source == destination:
            return self.lookup[state], float(count)
        factor = math.sqrt(count*(occupation.count(destination)+1))
        occupation.remove(source); occupation.append(destination); occupation.sort()
        return self.lookup[(occupation[0], occupation[1], state[2])], factor

    def transitions(self, column: int) -> list[tuple[int, complex, int, int, int, int, int]]:
        state = self.states[column]; out = []
        for edge, (i, j, axis) in enumerate(self.edges):
            forward = -self.p['hopping']*np.eye(2)-1j*self.p['spin_orbit']*self.sigma[axis]
            for source_site, target_site, matrix, direction in [(i, j, forward, 1), (j, i, forward.conj().T, -1)]:
                for flavour in [0, 1]:
                    for source_spin, target_spin in itertools.product(range(2), repeat=2):
                        coefficient = matrix[target_spin, source_spin]
                        if coefficient == 0:
                            continue
                        moved = self.move(state, flavour, 2*source_site+source_spin, 2*target_site+target_spin)
                        if moved is not None:
                            row, amplitude = moved
                            out.append((row, coefficient*amplitude, source_site, target_site, edge, direction, flavour))
        return out

    def spin_raising(self) -> Any:
        rows, columns, data = [], [], []
        for col, state in enumerate(self.states):
            for site in range(8):
                for flavour in [0, 1]:
                    moved = self.move(state, flavour, 2*site+1, 2*site)
                    if moved is not None:
                        row, amplitude = moved; rows.append(row); columns.append(col); data.append(amplitude)
        return sparse.csr_matrix((data, (rows, columns)), shape=(len(self.states), len(self.states)))

    def spin_half_source(self, sites: tuple[int, int, int]) -> tuple[Any, float]:
        vector = np.zeros(len(self.states), dtype=complex)
        u = sorted([2*sites[0], 2*sites[1]])
        vector[self.lookup[(u[0], u[1], 2*sites[2]+1)]] = 1
        raising = self.spin_raising()
        sz = np.array([sum(.5 if mode % 2 == 0 else -.5 for mode in state) for state in self.states])
        spin2vector = sz*sz*vector + .5*(raising@(raising.T@vector)+raising.T@(raising@vector))
        result = (3.75*vector-spin2vector)/3
        weight = float(np.vdot(result, result).real)
        return result/np.sqrt(weight), weight


class Candidate:
    def __init__(self, p: dict[str, Any]):
        self.p, self.body, self.local, self.flux = p, Body(p), LocalCloud(p), ColourFlux(p)
        self.ncloud, self.nflux = len(self.local.bits), self.flux.channels
        self.block = self.ncloud*self.nflux
        self.dimension = len(self.body.states)*self.block
        self.field_cache: dict[tuple[int, ...], Any] = {}

    def field_block(self, counts: tuple[int, ...]) -> Any:
        if counts not in self.field_cache:
            local = self.local.hamiltonian(counts, self.body.edges)
            colour = self.flux.get(counts)['H']
            self.field_cache[counts] = sparse.kron(local, sparse.eye(self.nflux), format='csr')+sparse.kron(sparse.eye(self.ncloud), colour, format='csr')
        return self.field_cache[counts]

    def assemble(self, external_A: Any = None, derivative_axis: int | None = None, contact: bool = False) -> Any:
        rows, columns, values = [], [], []
        max_hermiticity = 0.
        for col, counts in enumerate(self.body.counts):
            if derivative_axis is None:
                block = self.field_block(counts).tocoo()
                rows.append((col*self.block+block.row).astype(np.int32))
                columns.append((col*self.block+block.col).astype(np.int32)); values.append(block.data.astype(complex))
            for row, coefficient, i, j, edge, direction, flavour in self.body.transitions(col):
                target = self.body.counts[row]
                phonon = self.local.hop(counts, i, j)
                colour = self.flux.hop(counts, target, edge, direction)
                q = 2/3 if flavour == 0 else -1/3
                displacement = self.body.positions[j]-self.body.positions[i]
                if external_A is not None:
                    coefficient *= np.exp(1j*q*np.dot(external_A, displacement))
                if derivative_axis is not None:
                    # H(A) carries exp(+i q A.dx); J=-d_A H, K=d_A^2 H.
                    coefficient *= -(q*displacement[derivative_axis])**2 if contact else -1j*q*displacement[derivative_axis]
                if coefficient == 0:
                    continue
                block = sparse.kron(phonon, colour, format='coo')
                rows.append((row*self.block+block.row).astype(np.int32))
                columns.append((col*self.block+block.col).astype(np.int32)); values.append(coefficient*block.data)
            if col % 512 == 0:
                print(f'assemble body={col}/{len(self.body.states)} kind={derivative_axis}', flush=True)
        matrix = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(self.dimension, self.dimension)).tocsr()
        matrix.eliminate_zeros()
        difference = matrix-matrix.conj().T
        if difference.nnz:
            max_hermiticity = float(np.max(np.abs(difference.data)))
        if max_hermiticity > 1e-10:
            raise ValueError(f'Owning literal Hamiltonian/current not Hermitian: {max_hermiticity}')
        return matrix

    def source(self, sites: tuple[int, int, int]) -> tuple[Any, float]:
        spin, raw_spin_weight = self.body.spin_half_source(sites)
        vector = np.zeros(self.dimension, dtype=complex)
        # Dressed preparation: conditional local/colour ground basis state.
        # NOT silently identified with the bare canonical source on N=0.
        for body in np.flatnonzero(spin):
            vector[body*self.block] = spin[body]
        return vector, raw_spin_weight

    def observe(self, vector: Any) -> dict[str, Any]:
        amplitudes = vector.reshape(len(self.body.states), self.ncloud, self.nflux)
        body_probability = np.sum(np.abs(amplitudes)**2, axis=(1, 2))
        counts_array = np.array(self.body.counts)
        pair_distance = np.zeros(len(self.body.states))
        centres = counts_array@self.body.positions/3
        for i, j in itertools.combinations(range(8), 2):
            pair_distance += counts_array[:, i]*counts_array[:, j]*np.sum((self.body.positions[i]-self.body.positions[j])**2)/3
        raising = self.body.spin_raising()
        flat = amplitudes.reshape(len(self.body.states), -1)
        sz = np.array([sum(.5 if mode % 2 == 0 else -.5 for mode in state) for state in self.body.states])
        spin2 = np.sum(np.abs(sz[:, None]*flat)**2)+.5*(np.sum(np.abs(raising@flat)**2)+np.sum(np.abs(raising.T@flat)**2))
        result: dict[str, Any] = dict(norm=float(np.sum(body_probability)), charge=1.,
            valence_density=(body_probability@counts_array).tolist(), compact_all_three=float(np.sum(body_probability[np.max(counts_array, axis=1) == 3])),
            all_separated=float(np.sum(body_probability[np.max(counts_array, axis=1) == 1])),
            pair_rms=float(np.sqrt(body_probability@pair_distance)), centre_mean=(body_probability@centres).tolist(),
            spin_squared=float(spin2.real), spin_z=float(body_probability@sz),
            cloud_cap_probability=float(np.sum(np.abs(amplitudes[:, np.sum(self.local.bits, axis=1) == self.p['cloud_excitation_cap'], :])**2)))
        local_values = {name: np.zeros(8) for name in ['X', 'sea', 'phonon', 'top']}
        electric, magnetic, flux_response_weight, exterior_colour_H2 = 0., 0., 0., 0.
        for body, counts in enumerate(self.body.counts):
            state = amplitudes[body]
            if body_probability[body] == 0:
                continue
            colour_density = state.conj().T@state
            flux = self.flux.get(counts)
            electric += float(np.trace(colour_density@flux['electric_matrix'].T).real)
            magnetic += float(np.trace(colour_density@flux['magnetic_matrix'].T).real)
            exterior_colour_H2 += float(np.sum(np.diag(colour_density).real*flux['exterior_H2']))
            if self.nflux > 1:
                flux_response_weight += float(np.sum(np.abs(state[:, 1])**2))
            for site, occupation in enumerate(counts):
                for name in local_values:
                    op = self.local.local_product((site,), [self.local.ops[occupation][name]])
                    local_values[name][site] += float(np.vdot(state, op@state).real)
        result.update(local={k: v.tolist() for k, v in local_values.items()}, electric_link_square=electric,
            magnetic_energy=magnetic, flux_response_weight=flux_response_weight, conditional_colour_exterior_H2=exterior_colour_H2)
        return result

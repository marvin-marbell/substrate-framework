"""Literal edge-sharing triangular-strip CAR/supplier/orientor/Gauss law.
No magnetic/Wilson energy term. All operations act in the actual joint basis.
"""
from dataclasses import dataclass
from itertools import combinations, product
from typing import cast
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply, splu

LAW = 'edge-sharing-triangle-strip-CAR-orientor-supplier-common-Gauss-electric-v2-20261014'

@dataclass(frozen=True)
class Parameters:
    bound: int = 1
    inertia: float = 3.0
    active_hop: float = 1.0
    supplier_hop: float = 1.4
    neutral_hop: float = 0.7
    alignment: float = 1.1
    torque: float = 1.0
    cut: bool = False


def geometry():
    positions = np.array([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)], float)
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 1), (3, 4), (4, 2)]
    incidence = np.zeros((5, 7), int)
    for l, (a, b) in enumerate(edges):
        incidence[a, l], incidence[b, l] = 1, -1
    faces = np.zeros((3, 7), int)
    faces[0, [0, 1, 2]] = 1
    faces[1, [1, 3, 4]] = 1
    faces[2, [3, 5, 6]] = 1
    tree = [0, 1, 3, 5]
    chords = [2, 4, 6]
    return positions, edges, incidence, faces, tree, chords


def car(bits, mode, create):
    occupied = (bits >> mode) & 1
    if occupied == int(create):
        return None
    sign = (-1)**((bits & ((1 << mode)-1)).bit_count())
    return bits ^ (1 << mode), sign


def hop(bits, destination, origin):
    first = car(bits, origin, False)
    if first is None:
        return None
    second = car(first[0], destination, True)
    return None if second is None else (second[0], first[1]*second[1])


def fock_states(up, down, supplier, vertices):
    for u in combinations(range(vertices), up):
        for d in combinations(range(vertices), down):
            for m in combinations(range(vertices), supplier):
                yield sum(1 << (2*x) for x in u) + sum(1 << (2*x+1) for x in d) + sum(1 << (2*vertices+x) for x in m)


class Sector:
    def __init__(self, p=Parameters(), up=1, down=0, supplier=1):
        self.p, self.up, self.down, self.supplier = p, up, down, supplier
        self.positions, self.edges, self.B, self.faces, self.tree, self.chords = geometry()
        vertices, links = len(self.positions), len(self.edges)
        self.vertices, self.links = vertices, links
        self.background = np.zeros(vertices, int)
        self.background[0] = up+down+supplier
        chord_values = np.array(list(product(range(-p.bound, p.bound+1), repeat=3)), int)
        inverse = np.linalg.inv(self.B[np.ix_(range(1, vertices), self.tree)])
        cycle_tree = -self.B[np.ix_(range(1, vertices), self.chords)].T @ inverse.T
        basis, electric = [], []
        for bits in fock_states(up, down, supplier, vertices):
            rho = np.array([((bits >> (2*x)) & 1)+((bits >> (2*x+1)) & 1)+((bits >> (2*vertices+x)) & 1) for x in range(vertices)])-self.background
            fields = np.zeros((len(chord_values), links), int)
            fields[:, self.tree] = np.rint((-rho[1:])@inverse.T+chord_values@cycle_tree).astype(int)
            fields[:, self.chords] = chord_values
            for flux, field in zip(chord_values, fields):
                if np.max(np.abs(field)) > p.bound:
                    continue
                for r in range(3):
                    basis.append((bits, r, *flux)); electric.append(field)
        self.basis = basis
        self.lookup = {b: i for i, b in enumerate(basis)}
        self.dimension = len(basis)
        self.bits = np.array([b[0] for b in basis], np.int64)
        self.r = np.array([b[1] for b in basis])
        self.electric = np.array(electric)
        self.active_density = np.array([[((int(bits) >> (2*x)) & 1)+((int(bits) >> (2*x+1)) & 1) for x in range(vertices)] for bits in self.bits])
        self.supplier_density = np.array([[(int(bits) >> (2*vertices+x)) & 1 for x in range(vertices)] for bits in self.bits])
        self.rho = self.active_density+self.supplier_density-self.background
        self.electric_energy = np.sum(self.electric**2, axis=1)/(2*p.inertia)
        self.alignment_energy = -p.alignment*sum(self.active_density[:, site]*np.cos(2*np.pi*(angle-self.r)/3) for angle, site in enumerate([0, 1, 2]))
        self.H = sparse.diags(self.electric_energy+self.alignment_energy, format='csr', dtype=complex)
        self.transfers = {'active': [], 'supplier': []}
        left = {0, 1, 2}
        self.cut_edges = [l for l, (a, b) in enumerate(self.edges) if (a in left) != (b in left)]
        for species, strength, spins in [('active', p.active_hop, (0, 1)), ('supplier', p.supplier_hop, (0,))]:
            for l, (a, b) in enumerate(self.edges):
                rows, cols, values = [], [], []
                if not (p.cut and l in self.cut_edges):
                    for col, state in enumerate(basis):
                        for spin in spins:
                            origin = 2*a+spin if species == 'active' else 2*vertices+a
                            destination = 2*b+spin if species == 'active' else 2*vertices+b
                            moved = hop(state[0], destination, origin)
                            if moved is None:
                                continue
                            flux = list(state[2:])
                            if l in self.chords:
                                flux[self.chords.index(l)] += 1
                            row = self.lookup.get((moved[0], state[1], *flux))
                            if row is not None:
                                rows.append(row); cols.append(col); values.append(-strength*moved[1])
                t = sparse.csr_matrix((values, (rows, cols)), shape=self.H.shape, dtype=complex)
                self.transfers[species].append(t)
                self.H += t+t.getH()
        rows = [self.lookup[(b[0], (b[1]+1) % 3, *b[2:])] for b in basis]
        orientor = sparse.csr_matrix((np.ones(self.dimension), (rows, range(self.dimension))), shape=self.H.shape)
        self.Ln = 1j*(orientor-orientor.getH())/np.sqrt(3)
        self.H += -p.neutral_hop*(orientor+orientor.getH()) + p.torque*((up-down)/2)*self.Ln
        self.H = self.H.tocsr()
        self.currents = {s: [-1j*(t-t.getH()) for t in ts] for s, ts in self.transfers.items()}
        self.contacts = {s: [-(t+t.getH()) for t in ts] for s, ts in self.transfers.items()}
        self.total_currents = [self.currents['active'][l]+self.currents['supplier'][l] for l in range(links)]
        self.W = [self.wilson(face) for face in self.faces]
        self.F = [(w-w.getH())/(2j) for w in self.W]
        self.C = [(w+w.getH())/2 for w in self.W]
        self.loop_electric = [sparse.diags(self.electric@face/3, format='csr') for face in self.faces]
        self.body_x = sparse.diags(self.active_density@self.positions[:, 0], format='csr')
        self.body_velocity = 1j*(self.H@self.body_x-self.body_x@self.H)
        self.neutral_torque = 1j*(self.H@self.Ln-self.Ln@self.H)

    def wilson(self, face):
        shift = np.asarray(face)[self.chords]
        rows, cols = [], []
        for col, b in enumerate(self.basis):
            row = self.lookup.get((b[0], b[1], *(np.asarray(b[2:])+shift)))
            if row is not None:
                rows.append(row); cols.append(col)
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=self.H.shape)

    def ground(self):
        values, vectors = eigsh(self.H, k=3, which='SA', v0=np.linspace(.7, 1.3, self.dimension).astype(complex))
        order = np.argsort(values)
        return float(values[order[0]]), vectors[:, order[0]], values[order]

    def evolve(self, vectors, times, shift=0.):
        generator = cast(sparse.csr_matrix, -1j*(self.H-shift*sparse.eye(self.dimension)))
        return expm_multiply(generator, vectors, start=float(times[0]), stop=float(times[-1]), num=len(times), traceA=complex(generator.diagonal().sum()))

    def retarded(self, g, energy, ops, z):
        centered = centered_sources(g, ops)
        delta = self.H-energy*sparse.eye(self.dimension)
        eye = sparse.eye(self.dimension, format='csc')
        plus = splu((z*eye-delta).tocsc()).solve(centered)
        minus = splu((z*eye+delta).tocsc()).solve(centered)
        return centered.conj().T@plus-(centered.conj().T@minus).T

    def constraints(self):
        electric_error, density_error = 0., 0.
        for l in range(self.links):
            e = sparse.diags(self.electric[:, l], dtype=float)
            residual = 1j*(self.H@e-e@self.H)-self.total_currents[l]
            if residual.nnz:
                electric_error = max(electric_error, float(np.max(np.abs(residual.data))))
        for a in range(self.vertices):
            rho = sparse.diags(self.rho[:, a], dtype=float)
            residual = 1j*(self.H@rho-rho@self.H)
            for l in range(self.links):
                residual += self.B[a, l]*self.total_currents[l]
            residual = cast(sparse.csr_matrix, residual)
            if residual.nnz:
                density_error = max(density_error, float(np.max(np.abs(residual.data))))
        return {'Gauss_max': int(np.max(np.abs(self.electric@self.B.T+self.rho))), 'electric_current_max': electric_error, 'continuity_max': density_error}


def mean(state, operator):
    return float(np.vdot(state, operator@state).real)


def centered_sources(g, ops):
    raw = np.column_stack([op@g for op in ops])
    return raw-g[:, None]*(g.conj()@raw)[None, :]


def gram_frame(vectors):
    gram = vectors.conj().T@vectors
    values, rotation = np.linalg.eigh(gram)
    keep = values > max(1e-13, float(values.max())*1e-10)
    frame = vectors@rotation[:, keep]/np.sqrt(values[keep])[None, :]
    return frame, gram, values


def charged_source(previous, target, site, spin=0, create=True):
    """Physical CAR +/- with literal tree Wilson and +/- static anchor charge.
    The reference countercharge changes at vertex0 along with active number.
    """
    path: dict[int, tuple[tuple[int, int], ...]] = {0: ()}
    queue = [0]
    for vertex in queue:
        for l in target.tree:
            a, b = target.edges[l]
            other = b if a == vertex else a if b == vertex else None
            if other is not None and other not in path:
                path[other] = (*path[vertex], (l, 1 if a == vertex else -1)); queue.append(other)
    rows, cols, values = [], [], []
    for col, b in enumerate(previous.basis):
        action = car(b[0], 2*site+spin, create)
        if action is None:
            continue
        electric = previous.electric[col].copy()
        for l, direction in path[site]:
            electric[l] += direction*(1 if create else -1)
        row = target.lookup.get((action[0], b[1], *electric[target.chords]))
        if row is not None and np.array_equal(target.electric[row], electric):
            rows.append(row); cols.append(col); values.append(action[1])
    return sparse.csr_matrix((values, (rows, cols)), shape=(target.dimension, previous.dimension), dtype=complex)

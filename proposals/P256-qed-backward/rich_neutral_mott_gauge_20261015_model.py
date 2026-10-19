"""New physical-edge spin-one electric ladder law on the half-filled cube.

No old Sector/state/producer is used. Geometry, Fock enumeration and literal CAR
hop are frozen pure helpers. Every complete hop/loop is formed before projection.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy import sparse
from rich_photon_fullq_20261014_model import geometry, fock_basis
from rich_photon_20261014_loop import hop

LAW_ID = 'neutral-mott-cube-uniform-physical-edge-qlink3-20261015'


def maximum(matrix):
    return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0


def commutator(a, b):
    return a @ b - b @ a


class NeutralCube:
    def __init__(self):
        self.positions, self.edges, self.tree, self.chords, self.incidence, faces = geometry('cube')
        self.fock = np.asarray(fock_basis(8, 4, 4), dtype=np.int64)
        self.fi = {int(bits): i for i, bits in enumerate(self.fock)}
        self.flux = np.asarray(list(itertools.product((-1, 0, 1), repeat=5)), dtype=np.int8)
        self.nflux = len(self.flux)
        self.mi = {tuple(m): i for i, m in enumerate(self.flux)}
        occupations = np.asarray([[(int(b) >> (2*i) & 1) + (int(b) >> (2*i+1) & 1) for i in range(8)] for b in self.fock], dtype=np.int8)
        inverse = np.linalg.inv(self.incidence[np.ix_(range(1, 8), self.tree)])
        tree_base = -(occupations[:, 1:] - 1) @ inverse.T
        tree_cycle = -self.incidence[np.ix_(range(1, 8), self.chords)].T @ inverse.T
        solved = np.rint(tree_base[:, None, :] + (self.flux @ tree_cycle)[None, :, :]).astype(np.int8)
        valid = np.all(np.abs(solved) <= 1, axis=2)
        # This restriction is on ALL twelve physical edges, including every tree.
        self.old_indices = np.flatnonzero(valid.ravel())
        self.dimension = len(self.old_indices)
        self.lookup = np.full(len(self.fock)*self.nflux, -1, dtype=np.int32)
        self.lookup[self.old_indices] = np.arange(self.dimension)
        self.fock_id, self.flux_id = np.divmod(self.old_indices, self.nflux)
        self.electric = np.empty((self.dimension, 12), dtype=np.int8)
        self.electric[:, self.tree] = solved.reshape(-1, 7)[self.old_indices]
        self.electric[:, self.chords] = self.flux[self.flux_id]
        self.density = occupations[self.fock_id]
        self.doubles = np.asarray([[(int(b) >> (2*i) & 1)*(int(b) >> (2*i+1) & 1) for i in range(8)] for b in self.fock], dtype=np.int8)[self.fock_id]
        self.holons = (self.density == 0).astype(np.int8)
        self.left_sites = np.flatnonzero(self.positions[:, 0] == 0)
        self.right_sites = np.flatnonzero(self.positions[:, 0] == 1)
        self.left_cycle = next(c for c in faces if np.all(self.positions[np.flatnonzero(np.any(self.incidence[:, np.flatnonzero(c)] != 0, axis=1)), 0] == 0))
        self.left_edges = np.flatnonzero(self.left_cycle)
        self.right_edges = np.asarray([e for e, (a, b) in enumerate(self.edges) if self.positions[a, 0] == self.positions[b, 0] == 1])
        self.bridges = np.asarray([e for e, (a, b) in enumerate(self.edges) if self.positions[a, 0] != self.positions[b, 0]])
        self.weights = np.asarray([(self.positions[a, 0]+self.positions[b, 0])/2 for a, b in self.edges])
        self.transfers = []
        for edge, (a, b) in enumerate(self.edges):
            shift = np.zeros(5, dtype=int)
            if edge in self.chords:
                shift[self.chords.index(edge)] = 1
            fm = np.asarray([self.mi.get(tuple(m+shift), -1) for m in self.flux])
            rows, cols, values = [], [], []
            for spin in (0, 1):
                target = np.full(len(self.fock), -1, dtype=int)
                signs = np.zeros(len(self.fock), dtype=int)
                for k, bits in enumerate(self.fock):
                    result = hop(int(bits), 2*b+spin, 2*a+spin)
                    if result is not None:
                        target[k], signs[k] = self.fi[result[0]], result[1]
                ok = (target[self.fock_id] >= 0) & (fm[self.flux_id] >= 0)
                source = np.flatnonzero(ok)
                destination = self.lookup[target[self.fock_id[source]]*self.nflux+fm[self.flux_id[source]]]
                good = destination >= 0
                rows.extend(destination[good]); cols.extend(source[good])
                values.extend(-signs[self.fock_id[source[good]]])
            self.transfers.append(sparse.csr_matrix((values, (rows, cols)), shape=(self.dimension, self.dimension), dtype=float))
        self.currents = [-1j*(t-t.T) for t in self.transfers]
        self.HE = np.sum(self.electric.astype(float)**2, axis=1)/2
        self.HU = 4*np.sum(self.doubles, axis=1).astype(float)
        self.Hoff = sparse.diags(self.HE+self.HU, format='csr')
        self.H = self.Hoff + sum((t+t.T for t in self.transfers), sparse.csr_matrix(self.Hoff.shape))
        self.ER = np.sum(self.electric[:, self.right_edges].astype(float)**2, axis=1)/2
        self.HRoff = sparse.diags(np.sum(self.electric.astype(float)**2*self.weights, axis=1)/2+4*np.sum(self.doubles[:, self.right_sites], axis=1), format='csr')
        self.HR = self.HRoff + sum((w*(t+t.T) for w, t in zip(self.weights, self.transfers)), sparse.csr_matrix(self.H.shape))
        self.Hcut = self.H - sum((self.transfers[e]+self.transfers[e].T for e in self.bridges), sparse.csr_matrix(self.H.shape))
        self.HRcut = self.HRoff + sum((self.transfers[e]+self.transfers[e].T for e in self.right_edges), sparse.csr_matrix(self.H.shape))
        self.W = self.loop(self.left_cycle)
        self.source = (self.W+self.W.T)/2
        # C c_(site,spin) C^-1 = (-1)^(x+y+z) c^dagger, E -> -E.
        complemented = np.asarray([self.fi[int(b)^65535] for b in self.fock])
        reversed_flux = np.asarray([self.mi[tuple(-m)] for m in self.flux])
        ctarget = self.lookup[complemented[self.fock_id]*self.nflux+reversed_flux[self.flux_id]]
        phases = []
        for bits in self.fock:
            phase = 1
            for p in range(16):
                if int(bits) >> p & 1:
                    phase *= (-1)**(p+int(np.sum(self.positions[p//2])))
            phases.append(phase)
        self.C = sparse.csr_matrix((np.asarray(phases)[self.fock_id], (ctarget, np.arange(self.dimension))), shape=self.H.shape)

    def loop(self, cycle):
        # Whole physical product first; capacity on every target physical edge.
        mapping = np.asarray([self.mi.get(tuple(m+cycle[self.chords]), -1) for m in self.flux])
        source = np.flatnonzero(mapping[self.flux_id] >= 0)
        target = self.lookup[self.fock_id[source]*self.nflux+mapping[self.flux_id[source]]]
        good = target >= 0
        return sparse.csr_matrix((np.ones(np.sum(good)), (target[good], source[good])), shape=(self.dimension, self.dimension))

    def seed(self):
        """Four literal disjoint z-edge spin singlets, E=0, exactly one/site."""
        state = np.zeros(self.dimension, dtype=complex)
        for choices in itertools.product((0, 1), repeat=4):
            bits, sign = 0, 1
            for k, choice in enumerate(choices):
                a, b = 2*k, 2*k+1
                bits |= (1 << (2*a+choice)) | (1 << (2*b+1-choice))
                sign *= (-1)**choice
            index = self.lookup[self.fi[bits]*self.nflux+self.mi[(0,)*5]]
            state[index] = sign/4
        return state

    def measure(self, states):
        probability = np.abs(states)**2
        def diagonal(values):
            return np.asarray(values.T @ probability).real.T.tolist()
        def expectation(operator):
            return np.sum(states.conj()*(operator @ states), axis=0).real.tolist()
        hstate = self.H @ states
        hrstate = self.HR @ states
        # 2 Re < E J >/2 = Re <E J>; this is the symmetrized power,
        # not the product of symmetry-zero means.
        edge_power = [np.sum(states.conj()*(self.electric[:, e, None]*(j @ states)), axis=0).real for e, j in enumerate(self.currents)]
        right_current_squared = [np.sum(np.abs(self.currents[e] @ states)**2, axis=0) for e in self.right_edges]
        return dict(norm=np.sum(probability, axis=0).tolist(), H=expectation(self.H), H_variance=(np.sum(np.abs(hstate)**2, axis=0)-np.asarray(expectation(self.H))**2).tolist(),
                    HE=diagonal(self.HE), HU=diagonal(self.HU), ER=diagonal(self.ER), HR=expectation(self.HR),
                    powerR=(-2*np.sum(hstate.conj()*hrstate, axis=0).imag).tolist(),
                    powerER=np.sum(np.asarray(edge_power)[self.right_edges], axis=0).tolist(),
                    sym_EJ_power=np.asarray(edge_power).T.tolist(), right_J2=np.asarray(right_current_squared).T.tolist(),
                    density=diagonal(self.density), meanE=diagonal(self.electric), currents=np.asarray([expectation(j) for j in self.currents]).T.tolist(),
                    doublons=diagonal(self.doubles), holons=diagonal(self.holons), charge_variance=diagonal((self.density-1)**2),
                    right_HU=diagonal(4*np.sum(self.doubles[:, self.right_sites], axis=1)),
                    right_owned_HE=diagonal(np.sum(self.electric.astype(float)**2*self.weights, axis=1)/2),
                    ER_right_charge_free=diagonal(self.ER*np.all(self.density[:, self.right_sites] == 1, axis=1)),
                    ER_global_charge_free=diagonal(self.ER*np.all(self.density == 1, axis=1)),
                    right_charge_free_probability=diagonal(np.all(self.density[:, self.right_sites] == 1, axis=1)),
                    no_charge_pair_probability=np.sum(probability[np.all(self.density == 1, axis=1)], axis=0).tolist(),
                    C_expectation=expectation(self.C))

    def algebra(self):
        errors = dict(gauss=float(np.max(np.abs(self.electric @ self.incidence.T+self.density-1))), physical_edge_capacity=int(np.max(np.abs(self.electric))),
                      H_hermiticity=maximum(self.H-self.H.T), C_squared=maximum(self.C @ self.C-sparse.eye(self.dimension)),
                      C_H=maximum(commutator(self.C, self.H)), C_source=maximum(commutator(self.C, self.source)))
        primitive, continuity, power = [], [], []
        for e, t in enumerate(self.transfers):
            rows, cols = t.nonzero()
            shift = np.zeros(12, dtype=int); shift[e] = 1
            primitive.append(float(np.max(np.abs(self.electric[rows]-self.electric[cols]-shift))))
            E = sparse.diags(self.electric[:, e].astype(float), format='csr')
            continuity.append(maximum(1j*commutator(self.H, E)-self.currents[e]))
            power.append(maximum(1j*commutator(self.H, E @ E/2)-(E @ self.currents[e]+self.currents[e] @ E)/2))
        errors.update(hop_physical_shift=max(primitive), dotE_J=max(continuity), electric_power_identity=max(power))
        errors['density_continuity'] = max(maximum(1j*commutator(self.H, sparse.diags(self.density[:, site].astype(float), format='csr'))+sum((self.incidence[site, e]*self.currents[e] for e in range(12)), sparse.csr_matrix(self.H.shape))) for site in range(8))
        rows, cols = self.W.nonzero()
        errors['loop_physical_shift'] = float(np.max(np.abs(self.electric[rows]-self.electric[cols]-self.left_cycle)))
        errors['loop_matter_unchanged'] = int(np.max(np.abs(self.density[rows]-self.density[cols])))
        er = sparse.diags(self.ER, format='csr')
        errors.update(source_ER=maximum(commutator(self.source, er)), source_HR=maximum(commutator(self.source, self.HR)),
                      off_ER=maximum(commutator(self.Hoff, er)), off_HRoff=maximum(commutator(self.Hoff, self.HRoff)),
                      source_HRoff=maximum(commutator(self.source, self.HRoff)),
                      cut_factorization=maximum(commutator(self.Hcut-self.HRcut, self.HRcut)),
                      cut_left_ER=maximum(commutator(self.Hcut-self.HRcut, er)), source_HRcut=maximum(commutator(self.source, self.HRcut)))
        return errors

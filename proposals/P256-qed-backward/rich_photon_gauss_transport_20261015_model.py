"""Actual cube Gauss/Hubbard experiment; no magnetic restoration or CAR36 join.

The physical oriented faces are defined before the chord projection.  The frozen
fullq helper supplies only pure geometry/CAR/Gauss-sector construction, not states.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from rich_photon_fullq_20261014_model import Parameters, Sector, geometry

LAW_ID = 'rich-photon-literal-gauss-cube-transport-20261015'
ROOT = Path(__file__).resolve().parent
FROZEN_SHA256 = 'f3af36d79727738822ed2d19b04bf6905c90de0110265717570e7868828dbf02'


def maximum(matrix):
    return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0


def commutator(left, right):
    return left @ right - right @ left


def oriented_face(positions, edges, side):
    """Counterclockwise yz perimeter viewed along +x, NOT a tree-cycle label."""
    coordinates = [(side, 0, 0), (side, 1, 0), (side, 1, 1), (side, 0, 1)]
    vertices = [int(np.flatnonzero(np.all(positions == point, axis=1))[0]) for point in coordinates]
    cycle = np.zeros(len(edges), dtype=int)
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        edge = edges.index((a, b)) if (a, b) in edges else edges.index((b, a))
        cycle[edge] = 1 if edges[edge] == (a, b) else -1
    return vertices, cycle


class CubeExperiment:
    def __init__(self, cutoff=1):
        helper = ROOT / 'rich_photon_fullq_20261014_model.py'
        if hashlib.sha256(helper.read_bytes()).hexdigest() != FROZEN_SHA256:
            raise ValueError('Frozen pure model identity changed')
        positions, edges, _, _, incidence, _ = geometry('cube')
        # Define literal physical products before creating any truncated operator.
        self.left_vertices, self.left_cycle = oriented_face(positions, edges, 0)
        self.right_vertices, self.right_cycle = oriented_face(positions, edges, 1)
        if np.any(incidence @ self.left_cycle) or np.any(incidence @ self.right_cycle):
            raise ValueError('Face is not closed')
        self.left_edges = np.flatnonzero(self.left_cycle)
        self.right_edges = np.flatnonzero(self.right_cycle)
        self.bridge_edges = np.array([e for e, (a, b) in enumerate(edges) if positions[a, 0] != positions[b, 0]])
        if set(self.left_edges) & set(self.right_edges):
            raise ValueError('Source and receiver are not disjoint')
        self.parameters = Parameters(geometry='cube', cutoff=cutoff, hopping=1.0, repulsion=4.0,
                                     electric_inertia=1.0, reference_particles=2, anchor=0,
                                     background='integer_ions')
        self.sector = Sector(self.parameters, 1, 1)
        s = self.sector
        # flux_operator compresses the COMPLETE physical product in one step:
        # P (product_e U_e^cycle_e) P, not product_e (P U_e P).
        self.Wleft = s.flux_operator(self.left_cycle)
        self.Bleft = (self.Wleft - self.Wleft.T) / (2j)
        self.Cleft = (self.Wleft + self.Wleft.T) / 2
        self.Wright = s.flux_operator(self.right_cycle)
        self.Cright = (self.Wright + self.Wright.T) / 2
        self.density = np.repeat(s.occupations, s.nflux, axis=0)
        doubles = np.array([sum((bits >> (2*i) & 1) * (bits >> (2*i+1) & 1)
                                for i in range(s.vertices)) for bits in s.fock])
        self.hubbard = self.parameters.repulsion * np.repeat(doubles, s.nflux)
        self.Hoff = sparse.diags(s.electric_energy + self.hubbard, format='csr')
        self.ER = np.sum(s.electric[:, self.right_edges] ** 2, axis=1) / 2
        self.EL = np.sum(s.electric[:, self.left_edges] ** 2, axis=1) / 2
        self.Ebridge = np.sum(s.electric[:, self.bridge_edges] ** 2, axis=1) / 2
        # Actual energy ownership: endpoint average assigns each link electric
        # and hopping energy; Hubbard belongs to its vertex.  HL + HR = H.
        self.weights = np.array([(positions[a, 0] + positions[b, 0]) / 2 for a, b in edges])
        right_double = np.array([sum((bits >> (2*i) & 1) * (bits >> (2*i+1) & 1)
                                    for i in self.right_vertices) for bits in s.fock])
        self.HRoff = sparse.diags(np.sum(s.electric**2 * self.weights, axis=1) / 2
                                 + self.parameters.repulsion * np.repeat(right_double, s.nflux), format='csr')
        self.HR = self.HRoff + sum((weight * (transfer + transfer.T)
                                   for weight, transfer in zip(self.weights, s.transfers)),
                                  sparse.csr_matrix(s.H.shape))
        self.HL = s.H - self.HR
        # Plane cut is a second, literal same-state causal control. No new solve.
        self.Hcut = s.H - sum((s.transfers[e] + s.transfers[e].T for e in self.bridge_edges),
                              sparse.csr_matrix(s.H.shape))
        self.HRcut = self.HRoff + sum((s.transfers[e] + s.transfers[e].T for e in self.right_edges),
                                     sparse.csr_matrix(s.H.shape))
        self.HLcut = self.Hcut - self.HRcut
        self.powerR = 1j * commutator(s.H, self.HR)
        self.powerER = 1j * commutator(s.H, sparse.diags(self.ER))

    def seed(self):
        """Finite neutral product preparation, NOT a ground or filled sea claim."""
        s = self.sector
        second = int(np.flatnonzero(np.all(s.positions == (1, 0, 0), axis=1))[0])
        bits = (1 << 0) | (1 << (2 * second + 1))
        state = np.zeros(s.dimension, dtype=complex)
        state[s.fock_index[bits] * s.nflux + s.flux_index[(0,) * len(s.chords)]] = 1
        return state

    def algebra(self):
        s = self.sector
        electric_support = 0.0
        rows, cols = self.Wleft.nonzero()
        if len(rows):
            electric_support = float(np.max(np.abs(s.electric[rows] - s.electric[cols] - self.left_cycle)))
        electric_current = []
        density_current = []
        for edge in range(len(s.edges)):
            electric = sparse.diags(s.electric[:, edge])
            electric_current.append(maximum(1j * commutator(s.H, electric) - s.currents[edge]))
        for site in range(s.vertices):
            density = sparse.diags(self.density[:, site])
            density_current.append(maximum(1j * commutator(s.H, density)
                                          + sum((s.incidence[site, e] * s.currents[e] for e in range(len(s.edges))),
                                                sparse.csr_matrix(s.H.shape))))
        return dict(
            Gauss_all_basis=s.gauss_error, integer_electric=s.integer_electric_error,
            density_continuity=max(density_current), dotE_minus_J=max(electric_current),
            full_face_electric_support=electric_support,
            full_face_matter_support=float(np.max(np.abs(self.density[rows] - self.density[cols]))),
            source_receiver_electric_commutator=maximum(commutator(self.Bleft, sparse.diags(self.ER))),
            source_receiver_owned_energy_commutator=maximum(commutator(self.Bleft, self.HR)),
            source_receiver_current_commutator=max(maximum(commutator(self.Bleft, s.currents[e])) for e in self.right_edges),
            off_receiver_energy_conserved=maximum(commutator(self.Hoff, self.HRoff)),
            off_receiver_electric_conserved=maximum(commutator(self.Hoff, sparse.diags(self.ER))),
            cut_left_right_commutator=maximum(commutator(self.HLcut, self.HRcut)),
            cut_source_right_commutator=maximum(commutator(self.Bleft, self.HRcut)),
            cut_left_receiver_electric_commutator=maximum(commutator(self.HLcut, sparse.diags(self.ER))),
            ownership_sum_error=maximum(self.HL + self.HR - s.H))

    def measure(self, states):
        """Columns are baseline,pulse; every energy uses its disclosed H owner."""
        s = self.sector
        probability = np.abs(states)**2
        def diagonal(values):
            return np.real(values @ probability).tolist()
        def expectation(operator):
            return np.real(np.sum(states.conj() * (operator @ states), axis=0)).tolist()
        return dict(norm=np.sum(probability, axis=0).tolist(), H=expectation(s.H),
                    ER=diagonal(self.ER), EL=diagonal(self.EL), Ebridge=diagonal(self.Ebridge),
                    HE=diagonal(s.electric_energy), HU=diagonal(self.hubbard),
                    HR=expectation(self.HR), HL=expectation(self.HL),
                    powerR=expectation(self.powerR), powerER=expectation(self.powerER),
                    density=np.asarray(self.density.T @ probability).T.tolist(),
                    currents=np.asarray([expectation(j) for j in s.currents]).T.tolist(),
                    Cleft=expectation(self.Cleft), Cright=expectation(self.Cright),
                    boundary=[s.boundary_weight(states[:, i]) for i in range(states.shape[1])])

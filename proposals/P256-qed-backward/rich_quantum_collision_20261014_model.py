"""Full spin/vector-orientor extension of the literal 3D common Gauss law.
Only pure constructors/operator helpers are reused. No predecessor states import.
"""
from dataclasses import replace
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from rich_spin_flux_spatial_20261014_3d_model import Sector as AxialSector
from rich_spin_flux_spatial_20261014_model import Parameters, hop

LAW = '3D-common-Gauss-CAR-electric-full-SdotL-quantum-event-v1-20261014'


def maximum_entry(operator):
    operator = operator.tocsr()
    return float(np.max(np.abs(operator.data))) if operator.nnz else 0.


class Sector(AxialSector):
    """N_active=1, both spin orientations, N_supplier=1, all flux/frame states."""
    def __init__(self, p=Parameters(), vector=True):
        base = AxialSector(replace(p, torque=0.), 1, 0)
        down = AxialSector(replace(p, torque=0.), 0, 1)
        self.__dict__.update(base.__dict__)
        self.p = p
        self.basis = base.basis + down.basis
        self.lookup = {b: i for i, b in enumerate(self.basis)}
        self.dimension = len(self.basis)
        self.bits = np.concatenate([base.bits, down.bits])
        self.r = np.concatenate([base.r, down.r])
        for name in ['electric', 'active_density', 'supplier_density', 'rho', 'electric_energy', 'alignment_energy']:
            setattr(self, name, np.concatenate([getattr(base, name), getattr(down, name)]))
        self.H = sparse.block_diag([base.H, down.H], format='csr')
        self.transfers = {sp: [sparse.block_diag([a, b], format='csr') for a, b in zip(base.transfers[sp], down.transfers[sp])] for sp in base.transfers}
        pauli = [np.array([[0, 1], [1, 0]], complex)/2,
                 np.array([[0, -1j], [1j, 0]], complex)/2,
                 np.diag([.5, -.5])]
        generators = []
        for axis in range(3):
            matrix = np.zeros((3, 3), complex)
            b, c = (axis+1) % 3, (axis+2) % 3
            matrix[b, c], matrix[c, b] = -1j, 1j
            generators.append(matrix)
        self.spin = [self.internal_operator(matrix, spin=True) for matrix in pauli]
        self.frame_L = [self.internal_operator(matrix, spin=False) for matrix in generators]
        self.Ln = sum(self.frame_L)/np.sqrt(3)
        self.vector_coupling = p.torque*sum(s@l for s, l in zip(self.spin, self.frame_L))
        self.axial_coupling = p.torque*self.spin[2]@self.Ln
        self.H = (self.H + (self.vector_coupling if vector else self.axial_coupling)).tocsr()
        self.currents = {sp: [-1j*(t-t.getH()) for t in ts] for sp, ts in self.transfers.items()}
        self.contacts = {sp: [-(t+t.getH()) for t in ts] for sp, ts in self.transfers.items()}
        self.total_currents = [self.currents['active'][l]+self.currents['supplier'][l] for l in range(self.links)]
        self.W = [self.wilson(face) for face in self.faces]
        self.F = [(w-w.getH())/(2j) for w in self.W]
        self.C = [(w+w.getH())/2 for w in self.W]
        self.loop_electric = [sparse.diags(self.electric@f/4, format='csr') for f in self.faces]
        self.body_x = sparse.diags(self.active_density@self.positions[:, 0], format='csr')
        self.body_velocity = 1j*(self.H@self.body_x-self.body_x@self.H)
        self.spin_rate = [1j*(self.H@s-s@self.H) for s in self.spin]
        self.body_spin = [[sparse.diags(self.active_density[:, site])@s for s in self.spin] for site in range(self.vertices)]
        self.spin_currents = [[(s@j+j@s)/2 for s in self.spin] for j in self.currents['active']]

    def internal_operator(self, matrix, spin):
        rows, cols, values = [], [], []
        for col, state in enumerate(self.basis):
            if spin:
                site = int(np.flatnonzero(self.active_density[col])[0])
                origin = int((state[0] >> (2*site+1)) & 1)
                for destination in range(2):
                    value = matrix[destination, origin]
                    if value == 0:
                        continue
                    if destination == origin:
                        moved = (state[0], 1)
                    else:
                        moved = hop(state[0], 2*site+destination, 2*site+origin)
                    if moved is not None:
                        row = self.lookup[(moved[0], state[1], *state[2:])]
                        rows.append(row); cols.append(col); values.append(value*moved[1])
            else:
                for destination in range(3):
                    value = matrix[destination, state[1]]
                    if value != 0:
                        row = self.lookup[(state[0], destination, *state[2:])]
                        rows.append(row); cols.append(col); values.append(value)
        return sparse.csr_matrix((values, (rows, cols)), shape=(self.dimension, self.dimension))

    def polarized_ground(self):
        values, vectors = eigsh(self.H, k=6, which='SA', v0=np.linspace(.7, 1.3, self.dimension).astype(complex), tol=1e-11)
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        keep = np.abs(values-values[0]) < 1e-8
        # ARPACK vectors at an exact degeneracy need explicit orthonormalization.
        subspace, _ = np.linalg.qr(vectors[:, keep])
        spin_values, rotation = np.linalg.eigh(subspace.conj().T@(self.spin[2]@subspace))
        g = subspace@rotation[:, -1]
        return float(values[0]), g, values, spin_values

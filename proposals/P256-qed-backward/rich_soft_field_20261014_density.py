"""Material fork: finite-density independent correlated suppliers sharing flux.
Each supplier retains an explicit low-energy subspace of the exact 64-state
Hubbard cluster. Products of ALL retained supplier/electric states are used.
No response tensor, stiffness, or oscillator frequency is installed.
"""
import numpy as np
from scipy import sparse
from rich_soft_field_20261014_model import Joint, Matter


class DensityJoint(Joint):
    def __init__(self, p, n=1, suppliers=1, retain=8, frozen=False):
        bare = Joint(p, n, no_matter=True)
        self.__dict__.update(bare.__dict__)
        matter = Matter(p)
        values, vectors = np.linalg.eigh(matter.operators(0.)[0].toarray())
        self.projection = vectors[:, :1 if frozen else retain]
        self.retained_energies = values[:self.projection.shape[1]]
        d = self.projection.shape[1]
        if suppliers not in (1, 2):
            raise ValueError('This selected exact retained design has one or two suppliers.')
        self.suppliers, self.retain = suppliers, d
        self.md = d**suppliers
        identity = sparse.eye(d, format='csr')
        def project(operator):
            return sparse.csr_matrix(self.projection.conj().T@(operator@self.projection))
        def sum_suppliers(operator):
            if suppliers == 1:
                return operator
            return sparse.kron(operator, identity, format='csr')+sparse.kron(identity, operator, format='csr')
        blocks = [[], [], []]
        for phi in self.phi:
            for i, operator in enumerate(matter.operators(phi)):
                blocks[i].append(sum_suppliers(project(operator)))
        self.HM, self.J, self.K = [sparse.block_diag(b, format='csr') for b in blocks]
        self.HE = sparse.kron(bare.HE, sparse.eye(self.md), format='csr')
        self.H = self.HE+self.HM
        self.F = sparse.kron(bare.F, sparse.eye(self.md), format='csr')
        self.Q = sparse.kron(bare.Q, sparse.eye(self.md), format='csr')
        self.P = sparse.kron(bare.P, sparse.eye(self.md), format='csr')
        self.electric_leg = sparse.kron(bare.electric_leg, sparse.eye(self.md), format='csr')
        rho = sum_suppliers(project(sparse.diags(matter.density[:, 0]-matter.density[:, 1], dtype=float)))
        self.Rho = sparse.kron(sparse.eye(len(self.phi)), rho, format='csr')
        self.matter = matter

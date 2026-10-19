"""New triangle shared-link charged-supplier architecture, no magnetic bare term.
Finite energy-projected compact electric basis, explicit neutral three-state rotor.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu

LAW = 'triangle-active-neutral-rotor-charged-supplier-shared-gauss-v1-20261014'
@dataclass(frozen=True)
class Parameters:
    cutoff: int = 24
    inertia: float = 3.0
    active_hop: float = 1.0
    medium_hop: float = 1.4
    neutral_hop: float = 0.7
    alignment: float = 1.1
    transfer: float = 1.0
    medium: bool = True
    active: bool = True
    spin: int = 1

class Sector:
    """One spin block; both +/- blocks are retained in the delivered source API."""
    def __init__(self, p):
        self.p = p
        self.B = np.array([[1, 0, -1], [-1, 1, 0], [0, -1, 1]])
        limit = int(np.sqrt(p.cutoff))
        electric = [e for e in product(range(-limit, limit + 1), repeat=3) if sum(a*a for a in e) <= p.cutoff]
        self.basis = []
        count = int(p.active) + int(p.medium)
        for x in (range(3) if p.active else [-1]):
            for y in (range(3) if p.medium else [-1]):
                rho = np.zeros(3, int); rho[0] -= count
                if x >= 0: rho[x] += 1
                if y >= 0: rho[y] += 1
                for e in electric:
                    if np.any(self.B @ np.array(e) + rho): continue
                    for r in range(3): self.basis.append((x, y, r, *e))
        self.lookup = {b: i for i, b in enumerate(self.basis)}
        self.dimension = len(self.basis)
        self.e = np.array([b[3:] for b in self.basis])
        self.x = np.array([b[0] for b in self.basis])
        self.y = np.array([b[1] for b in self.basis])
        self.r = np.array([b[2] for b in self.basis])
        diag = np.sum(self.e**2, axis=1)/(2*p.inertia)
        if p.active: diag += -p.alignment*np.cos(2*np.pi*(self.x-self.r)/3)
        h = sparse.diags(diag.astype(complex), format='csr')
        self.transfers = {'active': [], 'medium': []}
        for species, index, hop, present in [('active', 0, p.active_hop, p.active), ('medium', 1, p.medium_hop, p.medium)]:
            for link in range(3):
                rows = []; cols = []
                if present:
                    for col, b in enumerate(self.basis):
                        if b[index] != link: continue
                        new = list(b); new[index] = (link+1) % 3
                        # rho changes by -B[:,link], hence E must increase.
                        new[3+link] += 1
                        row = self.lookup.get(tuple(new))
                        if row is not None: rows.append(row); cols.append(col)
                t = sparse.csr_matrix((-hop*np.ones(len(rows)), (rows, cols)), shape=h.shape, dtype=complex)
                self.transfers[species].append(t)
                h += t + t.getH()
        rows = []; cols = []
        for col, b in enumerate(self.basis):
            new = list(b); new[2] = (b[2]+1) % 3
            rows.append(self.lookup[tuple(new)]); cols.append(col)
        nr = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=h.shape, dtype=complex)
        self.Ln = 1j*(nr-nr.getH())/np.sqrt(3)
        h += -p.neutral_hop*(nr+nr.getH())
        if p.active: h += p.transfer*(p.spin/2)*self.Ln
        self.H = h.tocsr()
        self.currents = {s: [-1j*(t-t.getH()) for t in ts] for s, ts in self.transfers.items()}
        self.contacts = {s: [-(t+t.getH()) for t in ts] for s, ts in self.transfers.items()}
        rows = []; cols = []
        for col, b in enumerate(self.basis):
            new = (*b[:3], *(a+1 for a in b[3:]))
            row = self.lookup.get(new)
            if row is not None: rows.append(row); cols.append(col)
        self.W = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=h.shape, dtype=complex)
        self.F = (self.W-self.W.getH())/(2j)
        self.C = (self.W+self.W.getH())/2
        self.Ia = sum(self.currents['active'])/3
        self.Im = sum(self.currents['medium'])/3
        self.It = self.Ia + self.Im

    def solve(self, count=8):
        e, v = eigsh(self.H, k=min(count, self.dimension-2), which='SA', tol=2e-12, v0=np.linspace(.7, 1.3, self.dimension).astype(complex))
        order = np.argsort(e)
        self.energies, self.vectors = e[order], v[:, order]
        return self.energies, self.vectors

    def mean(self, g, op):
        return float(np.vdot(g, op@g).real)

    def constraints(self):
        rho = np.zeros((self.dimension, 3), int)
        rho[:, 0] -= int(self.p.active)+int(self.p.medium)
        for i, b in enumerate(self.basis):
            if b[0] >= 0: rho[i, b[0]] += 1
            if b[1] >= 0: rho[i, b[1]] += 1
        electric_current_error = 0.
        continuity_error = 0.
        for l in range(3):
            ee = sparse.diags(self.e[:, l])
            # i[H,E] = Jtotal with the declared current convention.
            err = 1j*(self.H@ee-ee@self.H)-self.currents['active'][l]-self.currents['medium'][l]
            if err.nnz: electric_current_error = max(electric_current_error, float(np.max(np.abs(err.data))))
        for a in range(3):
            nn = sparse.diags(rho[:, a]); err = 1j*(self.H@nn-nn@self.H)
            for l in range(3): err += self.B[a, l]*(self.currents['active'][l]+self.currents['medium'][l])
            if err.nnz: continuity_error = max(continuity_error, float(np.max(np.abs(err.data))))
        return {'gauss_max_error': int(np.max(np.abs(self.e@self.B.T+rho))), 'electric_current_operator_error': electric_current_error, 'continuity_operator_error': continuity_error}

    def state_data(self, g):
        prob = np.abs(g)**2
        return {'norm': float(np.vdot(g, g).real), 'F_sinWilson': self.mean(g, self.F), 'C_cosWilson': self.mean(g, self.C), 'active_circulation': self.mean(g, self.Ia), 'medium_circulation': self.mean(g, self.Im), 'total_circulation': self.mean(g, self.It), 'neutral_L': self.mean(g, self.Ln), 'spin_z': self.p.spin/2 if self.p.active else 0., 'spin_times_flux': (self.p.spin/2)*self.mean(g, self.F) if self.p.active else 0., 'electric_energy': float(prob@np.sum(self.e**2, axis=1)/(2*self.p.inertia)), 'energy_cutoff_shell_weight': float(prob@(np.sum(self.e**2, axis=1) >= self.p.cutoff-2)), 'eigen_residual': float(np.linalg.norm(self.H@g-self.mean(g, self.H)*g))}

    def retarded(self, g, energy, ops, z):
        rhs = np.column_stack([o@g-np.vdot(g, o@g)*g for o in ops])
        dh = self.H-energy*sparse.eye(self.dimension)
        eye = sparse.eye(self.dimension, format='csc')
        a = splu((z*eye-dh).tocsc()).solve(rhs)
        b = splu((z*eye+dh).tocsc()).solve(rhs)
        return rhs.conj().T@a-(rhs.conj().T@b).T


def charged_source(vacuum, charged, site):
    """Adds one active charge plus static -1 anchor reference and a fixed tree string.
    Tree edges 0->1->2; source is not bare charge creation in a fixed Gauss sector.
    """
    rows = []; cols = []
    for col, b in enumerate(vacuum.basis):
        e = list(b[3:])
        for l in range(site): e[l] += 1
        row = charged.lookup.get((site, b[1], b[2], *e))
        if row is not None: rows.append(row); cols.append(col)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(charged.dimension, vacuum.dimension), dtype=complex)

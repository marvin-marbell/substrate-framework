"""Exact retained two-loop CAR matter coupled to one projected 3D electric mode.
Two orthogonal copies give the two transverse legs. Noncompact flux projection;
Dirichlet amplitude cutoff is explicit, never compact-Fourier sewing by fiat.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply

LAW = 'periodic-cubic-curl-electric-two-correlated-loops-joint-20261014'


@dataclass(frozen=True)
class Parameters:
    shape: tuple = (64, 4, 4)
    electric: float = 10.0
    hopping: float = 1.0
    bridge: float = 0.35
    hubbard: float = 3.0
    dx: float = 0.065
    halfwidth: float = np.pi
    count: int = 16


def hop(bits, destination, origin):
    if not bits & (1 << origin) or bits & (1 << destination):
        return None
    sign = (-1) ** ((bits & ((1 << origin)-1)).bit_count())
    removed = bits ^ (1 << origin)
    sign *= (-1) ** ((removed & ((1 << destination)-1)).bit_count())
    return removed | (1 << destination), sign


def geometry(shape, n):
    """Forward-difference periodic cochain curl, with its adjoint divergence."""
    volume = int(np.prod(shape))
    x = np.arange(shape[0])[:, None, None]
    profile = np.broadcast_to(np.sqrt(2/volume)*np.cos(2*np.pi*n*x/shape[0]), shape)
    difference = lambda a, axis: np.roll(a, -1, axis=axis)-a
    legs = []
    for axis in (2, 1):
        m = np.zeros((*shape, 3))
        m[..., axis] = profile
        e = np.stack([difference(m[..., 2], 1)-difference(m[..., 1], 2),
                      difference(m[..., 0], 2)-difference(m[..., 2], 0),
                      difference(m[..., 1], 0)-difference(m[..., 0], 1)], axis=-1)
        div = sum(difference(e[..., a], a) for a in range(3))
        legs.append({'m': m, 'e': e, 'gauss_error': float(np.max(np.abs(div)))})
    gram = np.array([[np.vdot(a['e'], b['e']).real for b in legs] for a in legs])
    return {'q': 2*np.pi*n/shape[0], 'lambda': 4*np.sin(np.pi*n/shape[0])**2,
            'volume': volume, 'local_flux_factor': float(profile[0, 0, 0]),
            'gram': gram, 'legs': legs}


class Matter:
    """Eight sites, two square loops plus a physical bridge, Nup=Ndown=1."""
    def __init__(self, p):
        self.basis = [(1 << (2*u)) | (1 << (2*d+1)) for u in range(8) for d in range(8)]
        lookup = {bits: i for i, bits in enumerate(self.basis)}
        self.dimension = len(self.basis)
        self.density = np.array([[(bits >> (2*j) & 1)+(bits >> (2*j+1) & 1) for j in range(8)] for bits in self.basis])
        self.interaction = sparse.diags([p.hubbard*sum((bits >> (2*j) & 1)*(bits >> (2*j+1) & 1) for j in range(8)) for bits in self.basis], format='csr')
        self.edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4)]
        self.transfers = []
        for edge, (a, b) in enumerate(self.edges):
            rows, cols, vals = [], [], []
            for col, bits in enumerate(self.basis):
                for spin in (0, 1):
                    moved = hop(bits, 2*b+spin, 2*a+spin)
                    if moved:
                        rows.append(lookup[moved[0]]); cols.append(col)
                        vals.append(-(p.bridge if edge == 8 else p.hopping)*moved[1])
            self.transfers.append(sparse.csr_matrix((vals, (rows, cols)), shape=(64, 64)))
        self.flux_edges = {3, 7}
        self.positions = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                   (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], float)

    def operators(self, phi, charge=1., cut=False):
        h = self.interaction.astype(complex).copy()
        j = sparse.csr_matrix(h.shape, dtype=complex)
        k = sparse.csr_matrix(h.shape, dtype=complex)
        for edge, t in enumerate(self.transfers):
            if cut and edge in self.flux_edges:
                continue
            coupled = edge in self.flux_edges
            z = np.exp(1j*charge*phi) if coupled else 1.
            h += z*t+np.conjugate(z)*t.getH()
            if coupled:
                j += 1j*charge*(z*t-np.conjugate(z)*t.getH())
                k -= charge**2*(z*t+np.conjugate(z)*t.getH())
        return h.tocsr(), j.tocsr(), k.tocsr()


class Joint:
    def __init__(self, p=Parameters(), n=1, charge=1., cut=False, no_matter=False):
        self.p, self.n = p, n
        self.geometry = geometry(p.shape, n)
        self.metric = p.electric**2*self.geometry['lambda']*self.geometry['local_flux_factor']**2
        points = int(np.ceil(2*p.halfwidth/p.dx))-1
        self.dx = 2*p.halfwidth/(points+1)
        self.phi = np.linspace(-p.halfwidth+self.dx, p.halfwidth-self.dx, points)
        self.matter = Matter(p)
        self.md = 1 if no_matter else self.matter.dimension
        lap = sparse.diags([-np.ones(points-1), 2*np.ones(points), -np.ones(points-1)], [-1, 0, 1], format='csr')/self.dx**2
        self.HE = sparse.kron(self.metric/2*lap, sparse.eye(self.md), format='csr')
        if no_matter:
            self.HM = sparse.csr_matrix(self.HE.shape)
            self.J = self.K = sparse.csr_matrix(self.HE.shape)
        else:
            operators = [self.matter.operators(phi, charge, cut) for phi in self.phi]
            self.HM = sparse.block_diag([op[0] for op in operators], format='csr')
            self.J = sparse.block_diag([op[1] for op in operators], format='csr')
            self.K = sparse.block_diag([op[2] for op in operators], format='csr')
        self.H = self.HE+self.HM
        self.F = sparse.diags(np.repeat(np.sin(self.phi), self.md), format='csr')
        self.Q = sparse.diags(np.repeat(self.phi, self.md), format='csr')
        derivative = sparse.diags([-np.ones(points-1), np.ones(points-1)], [-1, 1], format='csr')/(2*self.dx)
        self.P = sparse.kron(-1j*derivative, sparse.eye(self.md), format='csr')
        self.electric_leg = p.electric*self.geometry['local_flux_factor']*np.sqrt(self.geometry['lambda'])*self.P
        density = np.ones(self.md) if no_matter else self.matter.density[:, 0]-self.matter.density[:, 1]
        self.Rho = sparse.diags(np.tile(density, points), format='csr')

    def spectrum(self):
        count = min(self.p.count, self.H.shape[0]-2)
        values, vectors = eigsh(self.H, k=count, which='SA', tol=2e-10,
                               v0=np.linspace(.7, 1.3, self.H.shape[0]).astype(complex))
        # ARPACK's complex degeneracies need an orthonormal selected subspace.
        # Rayleigh--Ritz retains the microscopic H, not individual raw columns.
        basis, _ = np.linalg.qr(vectors[:, np.argsort(values)])
        projected = basis.conj().T@(self.H@basis)
        self.values, rotation = np.linalg.eigh((projected+projected.conj().T)/2)
        self.vectors = basis@rotation
        self.g = self.vectors[:, 0]
        return self.values, self.vectors

    def response(self, omega, eta):
        """Full-H sparse resolvents, not the small spectral reporting window."""
        from scipy.sparse.linalg import spsolve
        ground = self.values[0]
        ops = [self.F, self.electric_leg, self.J, self.Rho]
        sources = np.array([op@self.g-self.g*np.vdot(self.g, op@self.g) for op in ops]).T
        identity = sparse.eye(self.H.shape[0], format='csr')
        plus = spsolve((omega+1j*eta+ground)*identity-self.H, sources)
        minus = spsolve((omega+1j*eta-ground)*identity+self.H, sources)
        return sources.conj().T@plus-(sources.conj().T@minus).T

    def pulse(self, times, amplitude=.08):
        state = expm_multiply(-1j*amplitude*self.J, self.g)
        records = []
        for t in times:
            psi = expm_multiply(-1j*t*(self.H-self.values[0]*sparse.eye(self.H.shape[0])), state)
            records.append({'time': float(t), 'electric_energy': float(np.vdot(psi, self.HE@psi).real),
                            'matter_energy': float(np.vdot(psi, self.HM@psi).real),
                            'current': float(np.vdot(psi, self.J@psi).real),
                            'field': float(np.vdot(psi, self.F@psi).real),
                            'density': float(np.vdot(psi, self.Rho@psi).real),
                            'norm': float(np.vdot(psi, psi).real)})
        return records

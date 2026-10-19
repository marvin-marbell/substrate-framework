"""Quantum scalar transfer order on a right-Gauss physical full-Fock frame.

Pure angular/frame encoder reuse only.  No old CLI execution, fixed mass term,
trap, preassigned physical spin, or claimed spatial self-binding.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from rich_frame_holonomy_20261014_model import raw_frame, paths


@lru_cache(maxsize=1)
def physical_frame():
    return raw_frame()


def matter(p, k=None):
    """Full physical16 from raw224; k is dimensionless uniform link phase."""
    frame = physical_frame()
    d = frame['source']
    k = np.asarray(p['k'] if k is None else k)
    bridge = paths(frame, p['theta'])
    forward = [p['hop'] * sum((1 if a < 2 else -1) * d[a] @ bridge[axis][a].conj().T for a in range(4)) / 2 for axis in range(3)]
    links = [np.exp(1j*k[a]) * forward[a] for a in range(3)]
    kinetic = sum(b + b.conj().T for b in links)
    transfer = sum(d[a] @ d[a+2].conj().T + d[a+2] @ d[a].conj().T for a in range(2))
    s2 = sum(s @ s for s in frame['S'])
    return {'frame': frame, 'H0': kinetic + p['frame_cost']*s2,
            'kinetic': kinetic, 'transfer': transfer, 'S2': s2,
            'current': [1j*(b-b.conj().T) for b in links],
            'contact': [-(b+b.conj().T) for b in links], 'forward': forward}


def build(p, k=None, coupling=None, frozen=None):
    """Return common full16 x scalar-DVR H/source/current, all N=0..4.

    `frozen` is an explicitly nonequivalent comparator: phi in the coupling is
    replaced by a number, while the elastic scalar remains an uncoupled quantum
    spectator.  It is never the delivered dynamical construction.
    """
    m = matter(p, k)
    n = p['scalar_points']
    dx = 2*p['wall']/(n+1)
    phi = np.linspace(-p['wall']+dx, p['wall']-dx, n)
    kinetic = sparse.diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1,0,1], format='csr') / (2*p['scalar_inertia']*dx**2)
    potential = .5*p['elastic_quadratic']*phi**2 + .25*p['elastic_quartic']*phi**4
    hs = kinetic + sparse.diags(potential, format='csr')
    ident = sparse.eye(n, format='csr')
    q = sparse.diags(phi, format='csr')
    g = p['coupling'] if coupling is None else coupling
    coupled_q = q if frozen is None else frozen*ident
    h = sparse.kron(m['H0'], ident, format='csr') + sparse.kron(np.eye(16), hs, format='csr') + g*sparse.kron(m['transfer'], coupled_q, format='csr')
    return {'H': h, 'phi': phi, 'dx': dx, 'Hscalar': hs, 'matter': m,
            'source': [sparse.kron(d, ident, format='csr') for d in m['frame']['source']],
            'current': [sparse.kron(j, ident, format='csr') for j in m['current']],
            'contact': [sparse.kron(j, ident, format='csr') for j in m['contact']],
            'Q': sparse.kron(np.eye(16),q,format='csr'),
            'T': sparse.kron(m['transfer'],ident,format='csr'),
            'scalar_kinetic': kinetic, 'scalar_potential': sparse.diags(potential,format='csr'),
            'N': sparse.kron(m['frame']['N'],ident,format='csr'),
            'nphi': n, 'parameters': dict(p), 'coupling': g, 'frozen': frozen}


def sector(model, number):
    n = model['nphi']
    occupations = [a for a in range(16) if a.bit_count() == number]
    indices = np.concatenate([np.arange(a*n,(a+1)*n) for a in occupations])
    return indices, model['H'][indices][:,indices].tocsr()


def ground_sector(model, number, count=8):
    indices, h = sector(model, number)
    # Deterministic initial vector, not a selected sign of the order parameter.
    v0 = np.cos(np.arange(h.shape[0])*.713) + 1j*np.sin(np.arange(h.shape[0])*.419)
    values, vectors = eigsh(h, k=count, which='SA', v0=v0, tol=2e-12)
    order = np.argsort(values)
    return indices, values[order], vectors[:,order]


def full_spectrum(model, number):
    indices, h = sector(model, number)
    values, vectors = eigh(h.toarray(), check_finite=False, driver='evd')
    return indices, values, vectors


def source_block(model, source_index, incoming, outgoing, addition=True):
    op = model['source'][source_index]
    if not addition:
        op = op.conj().T
    return op[outgoing][:,incoming].tocsr()

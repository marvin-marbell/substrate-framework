"""Two spatial material cells, full Gauss-source CAR and shared quantum scalar field.

Literal finite two-edge periodic law. No Bloch-cell G, prescribed mass, trap,
co-moving scalar copy, Maxwell claim, or replacement of phi by its mean/rms.
"""
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
from rich_frame_holonomy_20261014_model import raw_frame, canonical_c


def clean(a):
    a = np.array(a, dtype=complex, copy=True)
    a[np.abs(a) < 2e-13] = 0
    return sparse.csr_matrix(a)


@lru_cache(maxsize=1)
def physical():
    f = raw_frame()
    eye = sparse.eye(16, format='csr')
    parity = sparse.diags([(-1)**i.bit_count() for i in range(16)])
    d = [sparse.kron(clean(a), eye, format='csr') for a in f['source']]
    d += [sparse.kron(parity, clean(a), format='csr') for a in f['source']]
    spins = [[sparse.kron(clean(a), eye, format='csr') for a in f['S']],
             [sparse.kron(eye, clean(a), format='csr') for a in f['S']]]
    numbers = [sparse.kron(clean(f['N']), eye, format='csr'),
               sparse.kron(eye, clean(f['N']), format='csr')]
    cf, u = canonical_c(f['source'])
    return {'raw': f, 'd': d, 'S': spins, 'N': numbers,
            'C': sparse.kron(clean(cf), parity@clean(cf), format='csr'), 'c_intertwiner': u}


def build(p, coupling=None, hop=None, angles=None):
    f = physical()
    d = f['d']
    n = p['scalar_points']
    dx = 2*p['wall']/(n+1)
    phi = np.linspace(-p['wall']+dx, p['wall']-dx, n)
    k = sparse.diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1], format='csr')/(2*p['scalar_inertia']*dx**2)
    v = sparse.diags(.5*p['elastic_quadratic']*phi**2+.25*p['elastic_quartic']*phi**4)
    iq = sparse.eye(n, format='csr')
    field_identity = sparse.eye(n*n, format='csr')
    q = [sparse.kron(sparse.diags(phi), iq, format='csr'), sparse.kron(iq, sparse.diags(phi), format='csr')]
    gradient = .5*p['gradient']*(q[0]-q[1])@(q[0]-q[1])
    hs = sparse.kron(k+v, iq, format='csr')+sparse.kron(iq, k+v, format='csr')+gradient
    # Symmetric conjugate LEFT-frame paths are the rotational-scalar geometric
    # edge: (R d R† + R† d R)/2, not an inserted physical spin matrix.
    r = expm(-1j*p['theta']*f['raw']['S'][2])
    local_paths = [(r@a@r.conj().T+r.conj().T@a@r)/2 for a in f['raw']['source']]
    parity = sparse.diags([(-1)**i.bit_count() for i in range(16)])
    right_paths = [sparse.kron(parity, clean(a), format='csr') for a in local_paths]
    t = p['hop'] if hop is None else hop
    forward = t*sum(d[a]@right_paths[a].conj().T for a in range(4))
    forwards = [forward, forward.conj().T]
    angles = p['angles'] if angles is None else angles
    links = [np.exp(1j*a)*b for a, b in zip(angles, forwards)]
    hkin = sum(b+b.conj().T for b in links)
    casimir = p['frame_cost']*sum(s@s for site in f['S'] for s in site)
    transfers = [sum(d[o+a]@d[o+a+2].conj().T+d[o+a+2]@d[o+a].conj().T for a in range(2)) for o in (0, 4)]
    g = p['coupling'] if coupling is None else coupling
    h = sparse.kron(hkin+casimir, field_identity, format='csr')+sparse.kron(sparse.eye(256), hs, format='csr')
    h += g*sum(sparse.kron(a, b, format='csr') for a, b in zip(transfers, q))
    # A is a charge-twist probe with H(A)=e^{iA}F+h.c.; convention J=-dH/dA.
    currents = [sparse.kron(-1j*(b-b.conj().T), field_identity, format='csr') for b in links]
    contacts = [sparse.kron(-(b+b.conj().T), field_identity, format='csr') for b in links]
    return {'H': h.tocsr(), 'source': [sparse.kron(a, field_identity, format='csr') for a in d],
            'current': currents, 'contact': contacts, 'Q': [sparse.kron(sparse.eye(256), a, format='csr') for a in q],
            'N': [sparse.kron(a, field_identity, format='csr') for a in f['N']],
            'S': [sparse.kron(f['S'][0][a]+f['S'][1][a], field_identity, format='csr') for a in range(3)],
            'C': sparse.kron(f['C'], field_identity, format='csr'), 'phi': phi, 'nfield': n*n,
            'Hfield': hs, 'field_q': q, 'frame': f, 'parameters': dict(p), 'coupling': g, 'hop': t,
            'angles': list(angles), 'dx': dx, 'casimir': sparse.kron(casimir, field_identity, format='csr'),
            'scalar_transfer': g*sum(sparse.kron(a, b, format='csr') for a, b in zip(transfers, q))}


def sector(model, number):
    nf = model['nfield']
    states = [a for a in range(256) if a.bit_count() == number]
    indices = np.concatenate([np.arange(a*nf, (a+1)*nf) for a in states])
    return indices, model['H'][indices][:, indices].tocsr()


def ground(model, number, count=8):
    indices, h = sector(model, number)
    # Real-symmetric ARPACK preserves real-law multiplets more reliably than its
    # complex nonsymmetric fallback. Wider windows also retain exact degeneracy.
    count = max(count, 24)
    real_law = np.linalg.norm(h.data.imag) < 1e-12
    solver_h = h.real if real_law else h
    v0 = np.cos(np.arange(len(indices))*.713)
    if not real_law:
        v0 = v0+1j*np.sin(np.arange(len(indices))*.419)
    e, u = eigsh(solver_h, k=count, ncv=max(100, 3*count), which='SA', v0=v0, tol=3e-11)
    order = np.argsort(e)
    e, u = e[order], u[:, order]
    deg = np.count_nonzero(e-e[0] < 2e-8)
    if deg == count:
        raise RuntimeError('Ground multiplet unresolved; increase requested spectrum.')
    projector_basis, _ = np.linalg.qr(u[:, :deg])
    return indices, h, e, u, projector_basis


def block(op, rows, cols):
    return op[rows][:, cols].tocsr()

"""Literal filled CAR sea on a new bipartite molecular bridge graph.
All molecular levels are internal fermionic species, including neutral memory;
there is no independent/shared oscillator or photon in this candidate.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.linalg import eigh
from rich_spinor_rot_20261014_model import angular_geometry, field_geometry

LAW = 'bipartite-molecular-rung-odd-extended-bridge-filled-CAR-v1'

@dataclass(frozen=True)
class Parameters:
    charged_lmax: int = 2
    neutral_lmax: int = 1
    field_cutoff: int = 1
    rung_gap: float = 2.0
    inverse_inertia: float = 1.4
    neutral_inverse_inertia: float = 2.0
    charged_spin_orbit: float = 1.1
    neutral_spin_orbit: float = .25
    orientation: float = .45
    internal_frequency: float = 1.15
    internal_coupling: float = .16
    hopping: float = .55
    spacing: float = 1.0
    charge: float = -1.0


def angular_reversal(basis):
    u = np.zeros((len(basis), len(basis)), complex)
    lookup = {b: i for i, b in enumerate(basis)}
    for j, (l, m) in enumerate(basis):
        u[lookup[(l, -m)], j] = (-1.) ** m
    return u


def build(params=Parameters()):
    c = angular_geometry(params.charged_lmax)
    n = angular_geometry(params.neutral_lmax)
    f = field_geometry(params.field_cutoff)
    nc, nn, nf = len(c['basis']), len(n['basis']), len(f['basis'])
    eye = np.eye(nc * 2 * nn * nf)
    spin2 = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex) / 2
    m = params.rung_gap * eye.astype(complex)
    m += np.kron(np.kron(np.kron(params.inverse_inertia * c['L2'] / 2, np.eye(2)), np.eye(nn)), np.eye(nf))
    m += np.kron(np.kron(np.eye(nc * 2), params.neutral_inverse_inertia * n['L2'] / 2), np.eye(nf))
    m += np.kron(np.eye(nc * 2 * nn), params.internal_frequency * f['number'])
    r, s, j = [], [], []
    for i in range(3):
        rc = np.kron(np.kron(np.kron(c['R'][i], np.eye(2)), np.eye(nn)), np.eye(nf))
        si = np.kron(np.kron(np.kron(np.eye(nc), spin2[i]), np.eye(nn)), np.eye(nf))
        lc = np.kron(np.kron(np.kron(c['L'][i], np.eye(2)), np.eye(nn)), np.eye(nf))
        ln = np.kron(np.kron(np.eye(nc * 2), n['L'][i]), np.eye(nf))
        lf = np.kron(np.eye(nc * 2 * nn), f['L'][i])
        m += params.charged_spin_orbit * lc @ si + params.neutral_spin_orbit * ln @ si
        m += params.orientation * np.kron(np.kron(np.kron(c['R'][i], np.eye(2)), n['R'][i]), np.eye(nf))
        # Dynamic carried internal vector, position parity even product; not EM.
        m += params.internal_coupling * np.kron(np.kron(np.kron(c['R'][i], np.eye(2)), np.eye(nn)), f['Q'][i])
        r.append(rc); s.append(si); j.append(lc + si + ln + lf)
    dim = len(m)
    zero = np.zeros_like(m)
    rung = np.block([[zero, m], [m, zero]])
    tunnels = []
    for ri in r:
        a = params.hopping * ri
        # Actual oriented A_x -> B_{x+d} +a, A_x -> B_{x-d} -a.
        tunnels.append(np.block([[zero, a], [-a, zero]]))
    grade = np.diag(np.r_[np.ones(dim), -np.ones(dim)])
    tr0 = np.kron(np.kron(np.kron(angular_reversal(c['basis']), np.array([[0, 1], [-1, 0]])), angular_reversal(n['basis'])), np.eye(nf))
    tr = np.kron(np.eye(2), tr0)
    local = np.zeros((dim, nc * 2), complex)
    vacuum = f['basis'].index((0, 0, 0))
    for a in range(nc * 2):
        local[a * nn * nf + vacuum, a] = 1
    source = np.kron(np.eye(2), local)
    return {'law': LAW, 'params': params, 'parameters': asdict(params), 'M': m,
            'H0': rung, 'T': tunnels, 'directions': params.spacing * np.eye(3),
            'grade': grade, 'TR': tr, 'C': grade @ tr,
            'spin': np.array([np.kron(np.eye(2), x) for x in s]),
            'J': np.array([np.kron(np.eye(2), x) for x in j]),
            'parity': np.kron(np.eye(2), np.kron(np.kron(np.kron(c['parity'], np.eye(2)), n['parity']), f['parity'])),
            'source': source, 'internal_number': np.kron(np.eye(2), np.kron(np.eye(nc * 2 * nn), f['number']))}


def hamiltonian(model, k):
    h = model['H0'].copy()
    for d, t in zip(model['directions'], model['T']):
        phase = np.exp(1j * np.dot(k, d))
        h += phase * t + phase.conjugate() * t.conj().T
    return h


def spectrum(model, k):
    e, v = eigh(hamiltonian(model, k), check_finite=False)
    # Resolve Sz within actual degenerate eigenspaces, never split Kramers pairs.
    a = 0
    while a < len(e):
        b = a + 1
        while b < len(e) and abs(e[b] - e[a]) < 1e-9:
            b += 1
        w = v[:, a:b]
        _, u = eigh(w.conj().T @ model['spin'][2] @ w)
        v[:, a:b] = w @ u
        a = b
    return e, v


def vertex(model, kout, kin, epsilon):
    midpoint = (np.asarray(kout) + np.asarray(kin)) / 2
    q = np.asarray(kout) - np.asarray(kin)
    j = np.zeros_like(model['H0'])
    for d, t in zip(model['directions'], model['T']):
        phase = np.exp(1j * np.dot(midpoint, d))
        j += 1j * model['params'].charge * np.dot(epsilon, d) * np.sinc(np.dot(q, d) / (2 * np.pi)) * (phase * t - phase.conjugate() * t.conj().T)
    return j


def contact(model, kout, kin, qin, epsin, qout, epsout):
    midpoint = (np.asarray(kout) + np.asarray(kin)) / 2
    c = np.zeros_like(model['H0'])
    for d, t in zip(model['directions'], model['T']):
        phase = np.exp(1j * np.dot(midpoint, d))
        factor = np.dot(np.conjugate(epsout), d) * np.dot(epsin, d) * np.sinc(np.dot(qin, d) / (2 * np.pi)) * np.sinc(np.dot(qout, d) / (2 * np.pi))
        c -= model['params'].charge ** 2 * factor * (phase * t + phase.conjugate() * t.conj().T)
    return c


def charged_green(model, k, z):
    """Full zero-temperature time-ordered G: addition + removal from sea.
    z has positive imaginary part; removal uses conjugate z, giving -i eta.
    Source convention removal residue is <sea|c†|hole><hole|c|sea>.
    """
    e, v = spectrum(model, k)
    a = v.conj().T @ model['source']
    positive = e > 0
    addition = a[positive].conj().T @ ((1 / (z - e[positive]))[:, None] * a[positive])
    removal = a[~positive].conj().T @ ((1 / (np.conjugate(z) - e[~positive]))[:, None] * a[~positive])
    return {'addition': addition, 'removal': removal, 'time_ordered': addition + removal,
            'retarded': a.conj().T @ ((1 / (z - e))[:, None] * a)}


def pair_response(model, k, q, z):
    """Exact one-k sea pair tensor; caller supplies BZ measure for bulk Pi.
    Retains every occupied-to-empty matrix element and antiresonant time order.
    """
    e, v = spectrum(model, k)
    ep, vp = spectrum(model, np.asarray(k) + np.asarray(q))
    em, vm = spectrum(model, np.asarray(k) - np.asarray(q))
    occ, empty = e < 0, ep > 0
    vertices = [vertex(model, np.asarray(k) + q, k, np.eye(3)[i]) for i in range(3)]
    amplitudes = np.array([vp[:, empty].conj().T @ a @ v[:, occ] for a in vertices])
    delta = ep[empty, None] - e[None, occ]
    occm, empty0 = em < 0, e > 0
    reverse = np.array([v[:, empty0].conj().T @ vertex(model, k, np.asarray(k) - q, np.eye(3)[i]) @ vm[:, occm] for i in range(3)])
    deltam = e[empty0, None] - em[None, occm]
    tensor = np.einsum('iab,jab,ab->ij', amplitudes.conj(), amplitudes, 1 / (z - delta))
    tensor -= np.einsum('iab,jab,ab->ij', reverse, reverse.conj(), 1 / (z + deltam))
    return {'tensor': tensor, 'energies': delta, 'amplitudes': amplitudes,
            'summed_weight': np.einsum('iab,jab->ij', amplitudes.conj(), amplitudes),
            'occupied': int(np.sum(occ)), 'empty': int(np.sum(empty))}


def pair_creation(model, kh, qin, epsin, qout, epsout, omega, particle=0, hole=-1, eta=.025):
    """Connected sea -> one real particle-hole pair, two external-source legs.
    Complete Fock intermediate sector: all one-pair states accessible by J.
    Both particle scattering and minus-sign hole scattering, both time orders.
    This is NOT a photon-normalized cross section.
    """
    kh = np.asarray(kh); qin = np.asarray(qin); qout = np.asarray(qout)
    kp = kh + qin - qout
    eh, vh = spectrum(model, kh); ep, vp = spectrum(model, kp)
    hid = np.flatnonzero(eh < 0)[hole]; pid = np.flatnonzero(ep > 0)[particle]
    h, p = vh[:, hid], vp[:, pid]
    gap = ep[pid] - eh[hid]; wo = omega - gap
    direct = p.conj() @ contact(model, kp, kh, qin, epsin, qout, epsout) @ h
    contributions = []
    for qfirst, efirst, wfirst, qsecond, esecond in [
        (qin, epsin, omega, -qout, np.conjugate(epsout)),
        (-qout, np.conjugate(epsout), -wo, qin, epsin)]:
        ka = kh + qfirst; ea, va = spectrum(model, ka); un = ea > 0
        a = va[:, un]
        pa = vertex(model, kp, ka, esecond); ah = vertex(model, ka, kh, efirst)
        termp = np.sum((p.conj() @ pa @ a) * (a.conj().T @ ah @ h) / (wfirst - (ea[un] - eh[hid]) + 1j * eta))
        kb = kp - qfirst; eb, vb = spectrum(model, kb); oc = eb < 0
        b = vb[:, oc]
        pb = vertex(model, kp, kb, efirst); bh = vertex(model, kb, kh, esecond)
        termh = -np.sum((p.conj() @ pb @ b) * (b.conj().T @ bh @ h) / (wfirst - (ep[pid] - eb[oc]) + 1j * eta))
        contributions.append({'particle': termp, 'hole': termh})
    return {'amplitude': direct + sum(x['particle'] + x['hole'] for x in contributions),
            'contact': direct, 'time_orders': contributions, 'pair_energy': gap,
            'incoming_energy': omega, 'outgoing_energy': wo, 'kh': kh, 'kp': kp,
            'particle_index': int(pid), 'removed_index': int(hid)}

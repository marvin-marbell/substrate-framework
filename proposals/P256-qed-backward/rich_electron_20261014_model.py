"""Exact finite spinful dimerized ring with a quantum Peierls cavity.

Units: hbar=|electron charge|=nearest-neighbor distance=1. Sites 0..3
form an oriented ring; orbital A/B is site parity, cell is site//2.
N=4 reference sea; electric charge relative to it is -(N-4).
This is a nonrelativistic candidate, not a derivation of QED.
"""
from dataclasses import dataclass, asdict
from itertools import combinations
import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class Parameters:
    t_strong: float = 1.0
    t_weak: float = 0.65
    omega: float = 0.8
    coupling: float = 0.45
    boson_levels: int = 7


def basis(number):
    return np.array([sum(1 << p for p in occupied)
                     for occupied in combinations(range(8), number)], dtype=np.int64)


def annihilate(bits, mode):
    if not (bits >> mode) & 1:
        return None
    return bits ^ (1 << mode), (-1) ** ((bits & ((1 << mode) - 1)).bit_count())


def create(bits, mode):
    if (bits >> mode) & 1:
        return None
    return bits | (1 << mode), (-1) ** ((bits & ((1 << mode) - 1)).bit_count())


def bilinear(states, destination, origin):
    index = {int(b): i for i, b in enumerate(states)}
    out = np.zeros((len(states), len(states)), complex)
    for column, raw in enumerate(states):
        result = annihilate(int(raw), origin)
        if result is not None:
            removed, sign = result
            result = create(removed, destination)
            if result is not None:
                inserted, sign2 = result
                out[index[inserted], column] = sign * sign2
    return out


def sector_sources(origin_number, destination_number, params):
    """Eight local CAR source matrices destination <- origin, boson unchanged."""
    origin, destination = basis(origin_number), basis(destination_number)
    index = {int(b): i for i, b in enumerate(destination)}
    operation = create if destination_number == origin_number + 1 else annihilate
    if abs(destination_number - origin_number) != 1:
        raise ValueError('Source changes fermion number by exactly one')
    matrices = []
    for mode in range(8):
        out = np.zeros((len(destination), len(origin)), complex)
        for column, raw in enumerate(origin):
            result = operation(int(raw), mode)
            if result is not None:
                bits, sign = result
                out[index[bits], column] = sign
        matrices.append(np.kron(out, np.eye(params.boson_levels)))
    return matrices


def one_body_matrix(params, theta=0.0):
    h = np.zeros((4, 4), complex)
    for site in range(4):
        t = params.t_strong if site % 2 == 0 else params.t_weak
        h[(site+1) % 4, site] -= t * np.exp(1j * theta / 4)
    return h + h.conj().T


def build_sector(params, number, theta=0.0, spin_field=0.0):
    """Pure matrix interface. Ordering: lexicographic occupation combos, boson fast.

    H includes exact exp(i*g*Q) in the declared finite oscillator space.
    J=-dH/dtheta is flux-conjugate number current; charge-current is -J.
    C=d2H/dtheta2. A uniform link probe A=theta/4 has J_A=4J,C_A=16C.
    spin_field has energy units and H_B=-spin_field*S_z; it is not a derived
    magnetic gyromagnetic ratio. Q is the cavity coordinate, not charge.
    """
    states = basis(number)
    levels = params.boson_levels
    a = np.diag(np.sqrt(np.arange(1, levels)), 1)
    q = (a + a.T) / np.sqrt(2)
    qvalues, qvectors = eigh(q)
    link = (qvectors * np.exp(1j * params.coupling * qvalues)) @ qvectors.T
    identity_f, identity_b = np.eye(len(states)), np.eye(levels)
    h = np.kron(identity_f, params.omega * (a.T @ a)).astype(complex)
    current, contact = np.zeros_like(h), np.zeros_like(h)
    edge_current = [np.zeros_like(h) for _ in range(4)]
    edge_contact = [np.zeros_like(h) for _ in range(4)]
    densities = []
    sz_f = np.zeros((len(states), len(states)), complex)
    sx_f, sy_f = np.zeros_like(sz_f), np.zeros_like(sz_f)
    for site in range(4):
        up = bilinear(states, 2*site, 2*site)
        down = bilinear(states, 2*site+1, 2*site+1)
        densities.append(np.kron(up+down, identity_b))
        sz_f += (up-down) / 2
        flip = bilinear(states, 2*site, 2*site+1)
        sx_f += (flip+flip.conj().T) / 2
        sy_f += (flip-flip.conj().T) / (2j)
        t = params.t_strong if site % 2 == 0 else params.t_weak
        for spin in range(2):
            hop = bilinear(states, 2*((site+1) % 4)+spin, 2*site+spin)
            term = -t * np.exp(1j*theta/4) * np.kron(hop, link)
            h += term + term.conj().T
            current += -(1j/4) * (term-term.conj().T)
            contact += -(term+term.conj().T) / 16
            edge_current[site] += -1j * (term-term.conj().T)
            edge_contact[site] += -(term+term.conj().T)
    spin = [np.kron(s, identity_b) for s in (sx_f, sy_f, sz_f)]
    h -= spin_field * spin[2]
    positive_values, positive_vectors = eigh(one_body_matrix(params))
    positive = positive_vectors[:, positive_values > 0]
    projector = positive @ positive.conj().T
    pair_f = np.zeros_like(sz_f)
    for site in range(4):
        for other in range(4):
            for spin_index in range(2):
                pair_f += projector[site, other] * bilinear(states, 2*site+spin_index, 2*other+spin_index)
    return {'H': h, 'J': current, 'C': contact, 'basis': states,
            'Q': np.kron(identity_f, q), 'nb': np.kron(identity_f, a.T @ a),
            'density': densities, 'spin': spin,
            'edge_current': edge_current, 'edge_contact': edge_contact,
            'positive_band_number': np.kron(pair_f, identity_b),
            'charge': -(number-4), 'number': number, 'parameters': asdict(params)}


def current(sector, direction4):
    """Derivative -dH/ds for external link phases A_l += s*direction4_l."""
    return sum(direction4[l] * sector['edge_current'][l] for l in range(4))


def contact(sector, left4, right4):
    """Mixed link probe derivative d2H/dsdt; bare link cross terms vanish."""
    return sum(left4[l] * right4[l] * sector['edge_contact'][l] for l in range(4))


def solve(params, theta=0.0):
    sectors = {n: build_sector(params, n, theta) for n in (3, 4, 5)}
    for value in sectors.values():
        value['energies'], value['vectors'] = eigh(value['H'], check_finite=False)
    return sectors


def momentum_sources(params, k, addition=True):
    local = sector_sources(4, 5 if addition else 3, params)
    sources = []
    for orbital in range(2):
        for spin in range(2):
            phases = np.exp((1j if addition else -1j) * k * np.arange(2)) / np.sqrt(2)
            sources.append(sum(phases[cell] * local[2*(2*cell+orbital)+spin] for cell in range(2)))
    return sources


def spectral_data(sectors, params, k):
    """Full 4x4 orbital/spin retarded spectral amplitudes, not pole fitting.

    G_ab(z)=sum_n conj(v_na)*v_nb/(z-dEplus_n)
            +sum_m conj(w_mb)*w_ma/(z+dEminus_m).
    v=<N5,n|c†_a(k)|Omega>; w=<N3,m|c_a(k)|Omega>.
    Canonical c Fourier normalization 1/sqrt(2); Omega normalized N4 state.
    """
    ground = sectors[4]['vectors'][:, 0]
    e0 = sectors[4]['energies'][0]
    values = {}
    for number, tag, addition in ((5, 'plus', True), (3, 'minus', False)):
        source = np.column_stack([s @ ground for s in momentum_sources(params, k, addition)])
        values[tag+'_amplitudes'] = sectors[number]['vectors'].conj().T @ source
        values[tag+'_energies'] = sectors[number]['energies'] - e0
    values['k'] = k
    return values


def retarded_green(data, omega, eta=0.04):
    if eta <= 0:
        raise ValueError('Retarded regulator eta must be positive')
    v, w = data['plus_amplitudes'], data['minus_amplitudes']
    plus = 1 / (omega+1j*eta-data['plus_energies'])
    minus = 1 / (omega+1j*eta+data['minus_energies'])
    return np.einsum('na,nb,n->ab', v.conj(), v, plus) + np.einsum('nb,na,n->ab', w.conj(), w, minus)

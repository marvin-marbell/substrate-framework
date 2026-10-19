"""New complete-paired-shell geometric path law and literal filled CAR sea.

The physical halfspin matrices are reduced LEFT field-coordinate Noether
operators after right-colour Gauss, never primitive physical Pauli inputs.
CAR/fundamental colour/centre sewing and signed coherent path weights remain
hypotheses. Internal neutral/vector modes are carried fermionic species,
not an independent shared boson and not photons.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.linalg import eigh
from rich_spinor_rot_20261014_model import angular_geometry, field_geometry
from rich_frame_gauss_20261014_model import frame_gauss
from rich_filled_symmetry_20261014_model import angular_reversal
from rich_filled_symmetry_20261014_model import hamiltonian, spectrum, vertex, contact, charged_green, pair_response, pair_creation

LAW = 'field-frame-gauss-complete-j-signed-ordered-rotation-cap-path-filled-sea-v1'

@dataclass(frozen=True)
class Parameters:
    jmax: float = 1.5
    neutral_lmax: int = 1
    field_cutoff: int = 1
    torque_gap: float = 1.6
    higher_torque: float = .65
    neutral_transfer: float = .7
    vector_transfer: float = .85
    orientation: float = .14
    vector_feedback: float = .18
    frame_inertia: float = 2.0
    angle: float = .29
    hopping: float = .7
    spacing: float = 1.0
    charge: float = -1.0


def build(params=Parameters()):
    frame = frame_gauss()
    odd = np.flatnonzero(abs(frame['N'] - 1) < 1e-8)
    rawspin = [a[np.ix_(odd, odd)] for a in frame['S']]
    _, basis = eigh(rawspin[2])
    off = (basis.conj().T @ rawspin[0] @ basis)[0, 1]
    basis[:, 1] *= np.conjugate(off) / abs(off)
    spin = [basis.conj().T @ a @ basis for a in rawspin]
    c = angular_geometry(int(params.jmax + .5))
    nc = len(c['basis'])
    lraw = [np.kron(a, np.eye(2)) for a in c['L']]
    sraw = [np.kron(np.eye(nc), a) for a in spin]
    jraw = [a + b for a, b in zip(lraw, sraw)]
    j2 = sum(a @ a for a in jraw)
    je, jv = eigh(j2)
    projection = jv[:, je < params.jmax * (params.jmax + 1) + 1e-8]
    d_raw = np.eye(nc * 2) + 2 * sum(a @ b for a, b in zip(lraw, sraw))
    x_raw = 2 * sum(np.kron(r, s) for r, s in zip(c['R'], spin))
    project = lambda a: projection.conj().T @ a @ projection
    d = project(d_raw)
    x = project(x_raw)
    r = [project(np.kron(a, np.eye(2))) for a in c['R']]
    js = [project(a) for a in jraw]
    ss = [project(a) for a in sraw]
    trraw = np.kron(angular_reversal(c['basis']), np.array([[0, 1], [-1, 0]]))
    tr = projection.conj().T @ trraw @ projection.conjugate()
    values, vectors = eigh(d)
    u = (vectors * np.exp(1j * params.angle * values)) @ vectors.conj().T
    # Primitive two signed ordered paths: +U R_i U^-1 and -U^-1 R_i U.
    # These are finite angular transfers followed by a geometric cap aperture.
    # The current is subsequently derived by link Peierls differentiation.
    path_plus = [u @ a @ u.conj().T for a in r]
    path_minus = [u.conj().T @ a @ u for a in r]
    bridge = [(a - b) / 2 for a, b in zip(path_plus, path_minus)]
    n = angular_geometry(params.neutral_lmax)
    f = field_geometry(params.field_cutoff)
    nn, nf, ns = len(n['basis']), len(f['basis']), len(d)
    memories = nn * nf
    lift = lambda a: np.kron(a, np.eye(memories))
    # Each rung is a SIGNED torque transfer, not positive onsite kinetic energy.
    # A neutral/vector kinetic transfer occurs along the same torque-oriented
    # rung. This is a material correlated-transfer hypothesis, not mu(k) fitting.
    memory_cost = params.neutral_transfer * np.kron(n['L2'], np.eye(nf)) + params.vector_transfer * np.kron(np.eye(nn), f['number'])
    frame_cost = .75 / (2 * params.frame_inertia)
    h0 = lift((params.torque_gap + frame_cost) * d + params.higher_torque * (d @ d @ d - d)) + np.kron(d, memory_cost)
    pfield = [1j * (f['number'] @ a - a @ f['number']) for a in f['Q']]
    for a in range(3):
        h0 += params.orientation * np.kron((d @ r[a] + r[a] @ d) / 2, np.kron(n['R'][a], np.eye(nf)))
        h0 += params.vector_feedback * np.kron(bridge[a], np.kron(np.eye(nn), pfield[a]))
    tunnels = [params.hopping * lift(a) / (2j) for a in bridge]
    totalj = [lift(js[a]) + np.kron(np.eye(ns), np.kron(n['L'][a], np.eye(nf))) + np.kron(np.eye(ns * nn), f['L'][a]) for a in range(3)]
    trfull = np.kron(tr, np.kron(angular_reversal(n['basis']), np.eye(nf)))
    xfull = lift(x)
    source = np.zeros((ns * memories, ns), complex)
    vacuum = f['basis'].index((0, 0, 0))
    for a in range(ns):
        source[a * memories + vacuum, a] = 1
    parity = np.kron(project(np.kron(c['parity'], np.eye(2))), np.kron(n['parity'], f['parity']))
    # Adverse real conventional kinetics on the SAME domain. These commute
    # with chirality and therefore cannot be hidden by the exact C claim.
    conventional_kinetic = frame_cost * np.eye(ns * memories) + np.kron(np.eye(ns), memory_cost)
    return {'law': LAW, 'params': params, 'parameters': asdict(params), 'H0': h0, 'T': tunnels,
            'directions': params.spacing * np.eye(3), 'spin': np.array([lift(a) for a in ss]),
            'J': np.array(totalj), 'TR': trfull, 'C': xfull @ trfull, 'grade': xfull,
            'parity': parity, 'source': source, 'internal_number': np.kron(np.eye(ns * nn), f['number']),
            'D': d, 'X': x, 'R': r, 'bridge': bridge, 'path_plus': path_plus, 'path_minus': path_minus,
            'shell_J': js, 'shell_spin': ss, 'projection': projection, 'frame': frame,
            'canonical_frame_basis': basis, 'conventional_kinetic': conventional_kinetic,
            'ns': ns, 'nn': nn, 'nf': nf}

"""SO(3) material orientation, body-frame spin connection, and literal charge motion.

The entire 49152-state body is retained. No old ground state is imported.
The only predecessor import is its pure finite-body Hamiltonian construction.
"""
from __future__ import annotations
from functools import lru_cache
from typing import cast
import numpy as np
from scipy import sparse
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan
from rich_composite_rot_20261014_consumer import build, COORDS, current_contact

SPHERICAL = np.array([[1/np.sqrt(2), -1j/np.sqrt(2), 0], [0, 0, 1], [-1/np.sqrt(2), -1j/np.sqrt(2), 0]], complex)


def spin_matrices(j):
    numbers = np.arange(-j, j+1)
    plus = np.diag(np.sqrt((j-numbers[:-1])*(j+numbers[:-1]+1)), -1)
    return [(plus+plus.T)/2, (plus-plus.T)/(2j), np.diag(numbers)]


@lru_cache(None)
def cg(j, m, n, jp):
    if abs(m+n) > jp:
        return 0.
    return float(clebsch_gordan(Rational(int(2*j), 2), 1, Rational(int(2*jp), 2), Rational(int(2*m), 2), n, Rational(int(2*(m+n)), 2)))


@lru_cache(None)
def rotation_matrices(jp, mp, j, m):
    """<j' M' K'|R_ia|j M K>, exact Haar product, Cartesian lab/body axes.

    Orientation functions are sqrt(2j+1) D^{j*}_{MK}; all K retained.
    Half-integer functions are antiperiodic, because three primitive spins
    change sign under 2pi body-frame rotation. Lab wavefunctions are single
    valued on SO(3). N_a=J_a^T is the negative-algebra right generator.
    """
    ns = np.arange(-j, j+1)
    nps = np.arange(-jp, jp+1)
    out = [[np.zeros((len(nps), len(ns)), complex) for _ in range(3)] for _ in range(3)]
    spherical_m = mp-m
    if abs(spherical_m) > 1 or abs(spherical_m-round(spherical_m)) > 1e-10:
        return out
    sm = int(round(spherical_m))
    left = np.sqrt((2*j+1)/(2*jp+1))*cg(j, m, sm, jp)
    for sn in (-1, 0, 1):
        element = np.zeros((len(nps), len(ns)), complex)
        for column, k in enumerate(ns):
            kp = k+sn
            if abs(kp) <= jp:
                row = int(round(kp+jp))
                element[row, column] = left*cg(j, float(k), sn, jp)
        for i in range(3):
            for a in range(3):
                out[i][a] += SPHERICAL[sm+1, i].conjugate()*SPHERICAL[sn+1, a]*element
    return out


def material(p, flavors=(0, 1, 2)):
    h, body = build(p, flavors=flavors)
    size = h.shape[0]
    moment, hop_contact = current_contact(body, [0, 0, 0], [0, 0, 0], magnetic=True)
    # C_ab = 1/2 sum_i q_i (r_i^2 delta_ab-r_ia r_ib).
    # The charge positions are literal material coordinates, not spin labels.
    positions = COORDS[cast(np.ndarray, body['positions'])]
    charge_tensor = [[sparse.diags(np.repeat(.5*np.sum(body['charges'][None, :]*(np.sum(positions**2, axis=2)*(a == b)-positions[:, :, a]*positions[:, :, b]), axis=1), body['ns']*body['nf']), format='csr') for b in range(3)] for a in range(3)]
    return dict(H=h, body=body, T=body['J'], S=body['S'], L=body['L'], size=size, C=charge_tensor, hop_moment=moment, hop_contact=hop_contact)


def covariant_derivatives(material_body, j):
    cache = material_body.setdefault('derivative_cache', {})
    if j not in cache:
        eye = sparse.eye(material_body['size'], format='csr')
        ik = sparse.eye(int(2*j+1), format='csr')
        cache[j] = [sparse.kron(sparse.csr_matrix(a.T), eye, format='csr')-sparse.kron(ik, material_body['T'][axis], format='csr') for axis, a in enumerate(spin_matrices(j))]
    return cache[j]


def sector(material_body, p, j):
    """One conserved lab M block, EVERY K and EVERY internal state.

    H is independent of M, so every lab multiplet is restored by tensoring
    its solved K/internal vector with all M. Different j are compared, never
    selected as an input ground sector. High virtuals are not low-state projected.
    """
    ik = sparse.eye(int(2*j+1), format='csr')
    derivatives = covariant_derivatives(material_body, j)
    h = sparse.kron(ik, material_body['H'], format='csr')
    for a in range(3):
        h += (derivatives[a] @ derivatives[a])/(2*p['inertia'][a])
    return h.tocsr(), derivatives


def vector_component(material_body, jp, mp, j, m, components, axis):
    rotations = rotation_matrices(jp, mp, j, m)
    shape = (int(2*jp+1)*material_body['size'], int(2*j+1)*material_body['size'])
    operator = sparse.csr_matrix(shape, dtype=complex)
    for a in range(3):
        operator += sparse.kron(sparse.csr_matrix(rotations[axis][a]), components[a], format='csr')
    return operator


def magnetic_projected(material_body, p, jp, mp, j, m, axis, part='total'):
    """Literal magnetic source -dH/dB_axis from continuum kinetic + Peierls.

    Lrot=I Omega^2/2+Omega.a, a_a=sum_i q_i[r_i x R^-1 A(Rr_i)]_a.
    For A=B x x/2, a=C R^-1 B. Thus Hrot=(N-T-a)^2/(2I)
    BEFORE angular projection. Symmetric ordering and the full contact are kept.
    """
    dj = covariant_derivatives(material_body, j)
    dp = covariant_derivatives(material_body, jp)
    rotations = rotation_matrices(jp, mp, j, m)
    shape = (int(2*jp+1)*material_body['size'], int(2*j+1)*material_body['size'])
    mechanical = sparse.csr_matrix(shape, dtype=complex)
    for a in range(3):
        f = sparse.csr_matrix(shape, dtype=complex)
        for b in range(3):
            f += sparse.kron(sparse.csr_matrix(rotations[axis][b]), material_body['C'][b][a], format='csr')
        mechanical += (dp[a] @ f+f @ dj[a])/(2*p['inertia'][a])
    hopping = sparse.kron(sparse.csr_matrix(rotations[axis][2]), material_body['hop_moment'], format='csr')
    if part == 'mechanical':
        return mechanical
    if part == 'peierls':
        return hopping
    if part != 'total':
        raise ValueError(part)
    return mechanical+hopping


def magnetic_contact(material_body, p, j, m, axis):
    """Exact projection of continuum d2H/dB_axis2, not (P F P)^2.

    Every rank-one intermediate orientation multiplet j-1,j,j+1 is included.
    Its entire K/internal state space is kept. This is a quadratic contact,
    not an imported magnetic field-energy term.
    """
    shape = (int(2*j+1)*material_body['size'],)*2
    result = sparse.csr_matrix(shape, dtype=complex)
    for jp in (j-1, j, j+1):
        if jp < .5:
            continue
        for mp in np.arange(-jp, jp+1):
            rotations = rotation_matrices(jp, float(mp), j, m)
            for a in range(3):
                f = sparse.csr_matrix((int(2*jp+1)*material_body['size'], shape[0]), dtype=complex)
                for b in range(3):
                    f += sparse.kron(sparse.csr_matrix(rotations[axis][b]), material_body['C'][b][a], format='csr')
                result += (f.conj().T @ f)/p['inertia'][a]
            rz = sparse.csr_matrix(rotations[axis][2])
            result += sparse.kron(rz.conj().T @ rz, material_body['hop_contact'], format='csr')
    return result.tocsr()


def lab_internal(material_body, jp, mp, j, m, axis, kind='T'):
    return vector_component(material_body, jp, mp, j, m, material_body[kind], axis)


def full_multiplet_moment(material_body, p, j, axis, part='total'):
    numbers = np.arange(-j, j+1)
    return sparse.bmat([[magnetic_projected(material_body, p, j, float(mp), j, float(m), axis, part) for m in numbers] for mp in numbers], format='csr')

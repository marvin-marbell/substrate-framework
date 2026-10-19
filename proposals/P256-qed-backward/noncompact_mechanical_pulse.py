#!/usr/bin/env python3
"""Native mechanical pulse: full mass-horizontal modes and common normal RIGHT.

The quantum normal calculation is exact for the complete affine-force harmonic
Hamiltonian. The full non-skew source jet and nonlinear/whole-heavy/slow terms
are retained in separately reported native interfaces and analytic bounds, not
misrepresented as a numerical interacting scalar ground calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
import mpmath as mp
from scipy import sparse
from scipy.integrate import quad
from scipy.linalg import expm, null_space, norm as stable_vector_norm
from scipy.special import betainc, comb, gamma, jv

from bulk_quantum_selection import BulkQuantumLaw
from noncompact_finite_packet_spin import (
    Controller, ENDS, GraphQuantum, creator_maps, gram, initial_whiten,
    mechanical_geometry, prepare, propagate,
)
from noncompact_frame_pulse_transfer import endpoint_blocks, slater
from noncompact_marked_metric import spin_fields
from noncompact_native_current import accept, adverse
from original_charged_principal import OriginalSpatial, original_frame, plaquette_hessian, portable
from original_intrinsic_angular_transfer import MASS
from original_total_angular_transfer import (
    clover, endpoint_flow, gauge_move, oriented_step, primitive_flow, scalar_orbital,
)
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE, coordinates

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
REVISION = '609326d553757bde7ce67ee32f84492a05e6fe6705dd0705018c68d5c8f37b09'
TEMPLATE = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
SOURCES = (
    'noncompact_mechanical_pulse.py', 'noncompact_mechanical_pulse.md',
    'original_intrinsic_angular_transfer.py', 'original_intrinsic_angular_transfer.md',
    'original_total_angular_transfer.py', 'original_total_angular_transfer.md',
    'original_finite_packet_echo.py', 'original_finite_packet_echo.md',
    'original_slow_kinetic_bo.py', 'original_slow_kinetic_bo.md',
    'original_normal_gap_bound.py', 'original_normal_gap_bound.md',
    'original_charged_principal.py', 'full_spatial_gauge_stability.py',
    'noncompact_frame_pulse_transfer.py', 'noncompact_marked_metric.py',
    'noncompact_finite_packet_spin.py', 'bulk_quantum_selection.py',
)


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def linear_clover(law, A):
    """Actual noncompact curl, ALL anchored loops/reverse seams, never sin curl."""
    electromagnetic = np.zeros((law.volume, 3, 3))
    identity = np.tile(np.eye(4, dtype=complex), (law.volume, 3, 1, 1))
    for x in range(law.volume):
        for j, i in itertools.combinations(range(3), 2):
            for s, t in itertools.product((-1, 1), repeat=2):
                y, flux = x, 0.
                for axis, sign in ((j, s), (i, t), (j, -s), (i, -t)):
                    y, _, phase = oriented_step(law, y, axis, sign, identity, A)
                    flux += phase
                if y != x:
                    raise AssertionError('Noncompact anchored loop did not close')
                electromagnetic[x, j, i] += s*t*flux/(4*law.band.a**2)
            electromagnetic[x, i, j] = -electromagnetic[x, j, i]
    return electromagnetic


def mechanical_flow(law, fields, links, A, velocity, torque):
    """UNCHANGED Cartesian Phi/compact-colour/scalar spatial primitive, real EM."""
    df, du, _ = primitive_flow(law, fields, links, A, velocity, torque)
    B = linear_clover(law, A)
    da = np.zeros_like(A)
    for x in range(law.volume):
        for i in range(3):
            y = law.target[x, i]
            for j in range(3):
                da[x, i] -= law.band.a*(velocity[x, j]+velocity[y, j])*(B[x, j, i]+B[y, j, i])/4
    return df, du, da


def laplacian(law, b):
    b = np.asarray(b).reshape(law.volume)
    result = np.zeros_like(b)
    for j in range(3):
        previous = np.array([law.index[tuple((x-np.eye(3, dtype=int)[j]) % law.size)] for x in law.sites])
        result += 2*b-b[law.target[:, j]]-b[previous]
    return result


def profiles(law):
    """A fixed finite velocity graph, strict one-edge guard, two-row packet, outside R."""
    cells = (law.sites-law.size//2).astype(float)
    anchor = np.array([1, 0, 0])
    distance = np.abs(cells-anchor).sum(axis=1)
    b = (distance <= 2).astype(float)
    g = .25*(distance == 0)
    velocity = g[:, None]*np.cross(np.array([0., 0., 1.]), cells*law.band.a)
    packet = np.array([law.index[tuple(anchor+law.size//2)],
                       law.index[tuple(anchor+np.array([0, 1, 0])+law.size//2)]])
    eta = np.zeros(law.volume, complex)
    eta[packet] = [math.sqrt(.6), math.sqrt(.4)*np.exp(.37j)]
    R = law.index[(0, 0, 0)]
    if b[R] != 0 or not np.all(b[packet] == 1):
        raise AssertionError('Physical packet/halo and SAME outside reference overlap')
    return b, velocity, eta, R, packet


def geometry(law, frame, torque):
    model = OriginalSpatial(law.band, law.size, law.basis, coordinates(frame, law.basis).reshape(-1), np.zeros(3))
    change = np.zeros_like(frame)
    change[1], change[2] = -frame[2], frame[1]
    v = coordinates(change, law.basis).reshape(-1).real
    fhat = np.fft.fftn(torque.reshape((law.size,)*3), norm='ortho')
    ellhat = np.zeros((law.size,)*3+(15,), complex)
    chat = np.zeros((law.size,)*3+(105,), complex)
    maximum = 0.
    for n in itertools.product(range(law.size), repeat=3):
        q = 2*np.pi*np.array(n)/law.size
        G = model.gauge_columns(q)
        Y = np.r_[v*fhat[n], np.zeros(45)]
        ell = np.linalg.solve(G.conj().T@(MASS[:, None]*G), G.conj().T@(MASS*Y))
        normal = Y-G@ell
        maximum = max(maximum, float(np.linalg.norm(G.conj().T@(MASS*normal))))
        ellhat[n], chat[n] = ell, np.sqrt(MASS)*normal
    ell = np.asarray(np.fft.ifftn(ellhat, axes=(0, 1, 2), norm='ortho'), dtype=np.complex128).reshape(law.volume, 15)
    accept(np.linalg.norm(ell.imag), 'actual real mechanical Gauss connection', 2e-7)
    accept(maximum, 'all15 mass-horizontal native source modes', 2e-7)
    return model, ell.real, chat, maximum


def packet_marks(law, fields, links, A, eta, R):
    """Full field-dependent CT-even quaternion, real Coulomb phase, true Wilson word.

    Returns coefficients only; the whole-CAR creator and all four opposite
    reference components own their standard exterior-functor implementation.
    No occupation projection or photon Gaussian enters the preparation.
    """
    _, _, _, labels = spin_fields(fields)
    M = np.zeros((4, 4*law.volume, 4), complex)
    momenta = 2*np.pi*np.arange(law.size)/law.size
    spectrum = sum(2*(1-np.cos(momenta)).reshape(tuple(law.size if k == j else 1 for k in range(3))) for j in range(3))
    for x in np.flatnonzero(eta):
        charge = np.zeros(law.volume); charge[x], charge[R] = 1., -1.
        transformed = np.fft.fftn(charge.reshape((law.size,)*3))
        potential = np.asarray(np.fft.ifftn(np.divide(transformed, spectrum, out=np.zeros_like(transformed), where=spectrum > 1e-14)), dtype=np.complex128).real.reshape(-1)
        word = (potential[:, None]-potential[law.target])
        phase = np.exp(1j*np.sum(word*A))
        y, W = int(x), np.eye(4, dtype=complex)
        for axis in range(3):
            while law.sites[y, axis] != law.sites[R, axis]:
                forward = (int(law.sites[R, axis])-int(law.sites[y, axis])) % law.size
                sign = 1 if forward <= law.size//2 else -1
                y, edge, _ = oriented_step(law, y, axis, sign, links, A)
                W = W@edge
        if y != R:
            raise AssertionError('Actual colour Wilson path did not end at SAME R')
        M[:, 4*x:4*x+4] = eta[x]*phase*(labels[x]@W)
    return M


def immutable_array(value):
    """A bytes-backed array: callers cannot re-enable write access."""
    a = np.ascontiguousarray(value)
    backing = a
    while isinstance(backing, np.ndarray) and backing.base is not None:
        backing = backing.base
    payload = backing if isinstance(backing, bytes) and len(backing) == a.nbytes else a.tobytes()
    return np.frombuffer(payload, dtype=a.dtype).reshape(a.shape)


class ArraySnapshot:
    """Immutable bytes plus detached NumPy metadata on every public access."""
    __slots__ = ()

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = object.__getattribute__(self, name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, name, immutable_array(value))

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if not isinstance(value, np.ndarray):
            return value
        backing = value
        while isinstance(backing, np.ndarray):
            backing = backing.base
        assert isinstance(backing, bytes), 'Snapshot arrays must have an immutable bytes owner'
        return np.frombuffer(backing, dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, slots=True, eq=False)
class PacketMetric(ArraySnapshot):
    """Measured initial metric, never a final-time renormalization."""
    gram: np.ndarray
    inverse_root: np.ndarray
    minimum: float
    scope: str


def packet_metric_from_prepared(rows):
    """Actual TWO fixed joint receiver vectors; quartet/bar4 remain internal."""
    rows = np.asarray(rows)
    if rows.shape[0] != 2 or rows.shape[-1] != 4:
        raise ValueError('Need TWO fixed receiver rows with complete internal bar4')
    G = gram(rows)
    values, vectors = np.linalg.eigh(G)
    if values[0] <= 0:
        raise ValueError('Actual prepared initial source metric is singular')
    return PacketMetric(immutable_array(G),
                        immutable_array((vectors/np.sqrt(values))@vectors.conj().T),
                        float(values[0]), 'actual joint TWO-vector GLOBAL receiver Gram')


def coefficient_remainder(delta, metric):
    """State-independent CAR upper bound, normalized by ACTUAL initial G0."""
    delta = initial_whiten(np.asarray(delta), metric.inverse_root)
    return float(np.sqrt(np.linalg.eigvalsh(gram(delta))[-1]))


@dataclass(frozen=True, slots=True, eq=False)
class RadialPacket(ArraySnapshot):
    """Full-law coefficient supplier; neither a species nor an occupation cut."""
    size: int
    spacing: float
    origin: tuple[float, float, float]
    reference: int
    sigma: float
    bandwidth: float
    core_radius: float
    outer_radius: float
    filter_epsilon: float
    mark_epsilon: float
    sites: np.ndarray
    target: np.ndarray
    envelope: np.ndarray
    guard: np.ndarray
    coefficients: np.ndarray
    unguarded_coefficients: np.ndarray
    link_couplings: np.ndarray
    transporters: np.ndarray
    paths: tuple[tuple[tuple[int, int, int], ...], ...]
    tree_displacement: np.ndarray
    source_vertices: tuple[int, ...]
    halo_vertices: tuple[int, ...]
    filter_dependency: str
    body_toron: tuple[float, float, float] | None

    def scalar_orbital_action(self, velocity, coefficients=None):
        """Literal primitive scalar_orbital action, without allocating (4V)^2."""
        velocity = np.asarray(velocity)
        rows = self.coefficients if coefficients is None else np.asarray(coefficients)
        C = rows.reshape(4, self.size**3, 4, 4)
        result = np.zeros_like(C)
        for axis in range(3):
            y = self.target[:, axis]
            weight = (velocity[:, axis]+velocity[y, axis])/(4j*self.spacing)
            U = self.transporters[:, axis]
            forward = weight[:, None, None]*U
            backward = -weight[:, None, None]*U.conj().transpose(0, 2, 1)
            result += forward[None]@C[:, y]
            result[:, y] += backward[None]@C
        return immutable_array(result.reshape(rows.shape))

    def fixed_configuration_metric(self, vacancy: Callable[[np.ndarray], np.ndarray]):
        """Internal quartet Wick diagnostic ONLY; never the GLOBAL two-vector Ei G."""
        rows = self.coefficients
        image = np.asarray([vacancy(row) for row in rows])
        G = np.einsum('air,bir->ab', rows.conj(), image)
        G = (G+G.conj().T)/2
        values, vectors = np.linalg.eigh(G)
        if values[0] <= 0:
            raise ValueError('Actual frozen-state vacancy gives singular initial G0')
        return PacketMetric(immutable_array(G),
                            immutable_array((vectors/np.sqrt(values))@vectors.conj().T),
                            float(values[0]), 'frozen internal quaternion coefficient covariance; NOT GLOBAL Ei G')

    def photon_current(self, wavevector, polarization):
        """Per-unit PHYSICAL A current; no implicit charge or 1/sqrt(V) factor."""
        return self._peierls_vertex(wavevector, polarization, None)

    def photon_contact(self, wavevector_sum, polarization1, polarization2):
        """Exact second PHYSICAL A derivative; both link legs retained."""
        return self._peierls_vertex(wavevector_sum, polarization1, polarization2)

    def link_current(self, link_mode):
        """Exact native LINK-variable derivative; accept the actual modal map."""
        return self._link_vertex(np.asarray(link_mode), second=False)

    def link_contact(self, link_mode1, link_mode2):
        """Exact native LINK-variable contact, with no modal normalization."""
        return self._link_vertex(np.asarray(link_mode1)*np.asarray(link_mode2), second=True)

    def _peierls_vertex(self, wavevector, polarization1, polarization2):
        k, e = np.asarray(wavevector), np.asarray(polarization1)
        phase = np.exp(-1j*self.spacing*(self.sites@k))[:, None]*np.exp(-1j*self.spacing*k/2)[None, :]
        weight = self.spacing*phase*e
        if polarization2 is None:
            return self._link_vertex(weight, second=False)
        return self._link_vertex(weight*self.spacing*np.asarray(polarization2), second=True)

    def _link_vertex(self, weights, *, second):
        if weights.shape != (self.size**3, 3):
            raise ValueError('Supply the actual full native link-mode array (V,3)')
        blocks, rr, cc = [], [], []
        colour_row, colour_column = np.indices((4, 4))
        for axis in range(3):
            weight = weights[:, axis]
            if second:
                forward, backward = -self.link_couplings[:, axis], -self.link_couplings[:, axis].conj().transpose(0, 2, 1)
            else:
                forward, backward = 1j*self.link_couplings[:, axis], -1j*self.link_couplings[:, axis].conj().transpose(0, 2, 1)
            x, y = np.arange(self.size**3), self.target[:, axis]
            for left, right, matrix in ((x, y, forward), (y, x, backward)):
                rr.append((4*left[:, None, None]+colour_row).reshape(-1))
                cc.append((4*right[:, None, None]+colour_column).reshape(-1))
                blocks.append((weight[:, None, None]*matrix).reshape(-1))
        return sparse.coo_matrix((np.concatenate(blocks),
                                  (np.concatenate(rr), np.concatenate(cc))),
                                 shape=(4*self.size**3,)*2).tocsr()


def radial_profile(law, origin, sigma, bandwidth, core_radius, outer_radius):
    """Radial in native Wilson radius; a PREPARATION, never a Hamiltonian cut."""
    a, N = law.band.a, law.size
    if not (0 < sigma < bandwidth < np.pi/a and
            0 < core_radius < outer_radius < N*a/2-a):
        raise ValueError('Need positive radial width/bandwidth and a genuine guard+halo')
    p = 2*np.pi*np.fft.fftfreq(N, d=a)
    momentum = np.stack(np.meshgrid(p, p, p, indexing='ij'), axis=-1)
    r2 = np.sum((2*np.sin(a*momentum/2)/a)**2, axis=-1)
    s2 = r2/bandwidth**2
    weights = np.zeros((N,)*3)
    inside = s2 < 1
    weights[inside] = np.exp(-r2[inside]/(4*sigma**2)-s2[inside]/(1-s2[inside]))
    shifted = weights*np.exp(-1j*np.einsum('...j,j->...', momentum, origin))
    eta = np.fft.ifftn(shifted).reshape(-1)
    eta /= np.linalg.norm(eta)
    displacement = (a*law.sites-np.asarray(origin)+N*a/2) % (N*a)-N*a/2
    radius = np.linalg.norm(displacement, axis=-1)
    guard = betainc(9, 9, np.clip((outer_radius-radius)/(outer_radius-core_radius), 0, 1))
    return eta, guard, momentum


def regular_mark_labels(fields, epsilon, field_dot=None):
    """Globally smooth preparation marks, not a change to the Gauss generator."""
    if epsilon <= 0:
        raise ValueError('A positive preparation regularizer is required')
    flat = fields.reshape(len(fields), 4, 16)
    features = np.concatenate((flat.real, flat.imag), axis=-1)/2
    U, singular, _ = np.linalg.svd(features, full_matrices=False)
    roots = np.sqrt(singular**2+epsilon**2)
    Q = np.einsum('xmi,xi,xni->xmn', U, 1/roots, U)
    gammas = np.einsum('xmn,xnij->xmij', Q, fields)
    labels = np.empty(fields.shape, dtype=complex)
    labels[:, 0] = np.eye(4)
    for j, (k, l) in enumerate(((2, 3), (3, 1), (1, 2)), 1):
        labels[:, j] = (gammas[:, k]@gammas[:, l]-gammas[:, l]@gammas[:, k])/2
    if field_dot is None:
        return labels
    dG = (np.einsum('xmij,xnji->xmn', field_dot, fields)
          + np.einsum('xmij,xnji->xmn', fields, field_dot))/4
    L = -1/(roots[:, :, None]*roots[:, None, :]*(roots[:, :, None]+roots[:, None, :]))
    transformed = U.transpose(0, 2, 1)@dG@U
    dQ = U@(L*transformed)@U.transpose(0, 2, 1)
    dg = np.einsum('xmn,xnij->xmij', dQ, fields)+np.einsum('xmn,xnij->xmij', Q, field_dot)
    dlabels = np.zeros_like(labels)
    for j, (k, l) in enumerate(((2, 3), (3, 1), (1, 2)), 1):
        dlabels[:, j] = (dg[:, k]@gammas[:, l]+gammas[:, k]@dg[:, l]
                        -dg[:, l]@gammas[:, k]-gammas[:, l]@dg[:, k])/2
    return labels, dlabels


def coulomb_source_phase(law, A, reference):
    """Same Green word as packet_marks, with one solve rather than V solves."""
    divergence = np.sum(A, axis=1).reshape((law.size,)*3)
    for axis in range(3):
        divergence -= np.roll(A[:, axis].reshape((law.size,)*3), 1, axis=axis)
    p = 2*np.pi*np.fft.fftfreq(law.size)
    laplacian = sum(2*(1-np.cos(p)).reshape(tuple(law.size if j == axis else 1 for j in range(3))) for axis in range(3))
    transformed = np.fft.fftn(divergence)
    potential = np.fft.ifftn(np.divide(transformed, laplacian, out=np.zeros_like(transformed), where=laplacian > 1e-14)).reshape(-1)
    if not np.iscomplexobj(A):
        potential = np.real(potential)
    return potential-potential[reference]


def source_tree(law, links, reference, link_dot=None):
    """The existing genuine shortest-axis Wilson convention, with its full jet."""
    words, paths, displacement, derivatives = [], [], [], []
    dummy_phases = np.zeros((law.volume, 3))
    for x in range(law.volume):
        y, W, dW = x, np.eye(4, dtype=complex), np.zeros((4, 4), complex)
        path, dx = [], np.zeros(3)
        for axis in range(3):
            while law.sites[y, axis] != law.sites[reference, axis]:
                forward = (int(law.sites[reference, axis])-int(law.sites[y, axis])) % law.size
                sign = 1 if forward <= law.size//2 else -1
                old = y
                y, edge, _ = oriented_step(law, y, axis, sign, links, dummy_phases)
                edge_origin = old if sign == 1 else y
                path.append((edge_origin, axis, sign))
                dx[axis] -= sign
                if link_dot is not None:
                    de = link_dot[edge_origin, axis]
                    if sign == -1:
                        de = -edge@de@edge
                    dW = dW@edge+W@de
                W = W@edge
        words.append(W); derivatives.append(dW); paths.append(tuple(path)); displacement.append(dx)
    return np.asarray(words), tuple(paths), np.asarray(displacement), np.asarray(derivatives)


def primitive_couplings(law, fields, links, A):
    result = np.empty((law.volume, 3, 4, 4), complex)
    for axis, scalar in law.paths:
        F = np.einsum('m,xmij->xij', scalar, fields)
        y = law.target[:, axis]
        result[:, axis] = (F@links[:, axis]+links[:, axis]@F[y])/2*np.exp(1j*A[:, axis, None, None])
    return result


def smooth_band_value(values, epsilon):
    return (1+values/np.hypot(values, epsilon))/2


def smooth_band_divided_difference(values, epsilon):
    """Stable f[x,y], including opposite signs and exactly repeated eigenvalues."""
    x, y = values[..., :, None], values[..., None, :]
    sx, sy = np.hypot(x, epsilon), np.hypot(y, epsilon)
    opposite = x*y < 0
    denominator = np.where(opposite,
                           1+(x*x+y*y+epsilon**2)/(sx*sy+abs(x*y)),
                           (sx*sy+x*y+epsilon**2)/epsilon**2)
    return (sx+sy)/(2*sx*sy*denominator)


def radial_packet_source(law, fields, links, A, *, origin, reference, sigma,
                         bandwidth, core_radius, outer_radius,
                         filter_epsilon=1e-5, mark_epsilon=1e-5, body_toron=None):
    """Exact full finite-lattice F_e(h) mark; explicit body FFT or full eigensystem."""
    fields, links, A = np.asarray(fields), np.asarray(links), np.asarray(A)
    if filter_epsilon <= 0:
        raise ValueError('The globally smooth particle preparation needs epsilon>0')
    eta, guard, momentum = radial_profile(law, origin, sigma, bandwidth, core_radius, outer_radius)
    active = tuple(int(x) for x in np.flatnonzero(guard))
    halo = set(active)
    for axis in range(3):
        halo.update(int(x) for x in law.target[list(active), axis])
        halo.update(int(x) for x in np.flatnonzero(np.isin(law.target[:, axis], active)))
    if reference in halo:
        raise ValueError('Complete opposite reference must lie outside source+halo')
    W, paths, dx, _ = source_tree(law, links, reference)
    flat = A.mean(axis=0)
    phase = np.exp(1j*coulomb_source_phase(law, A, reference)-1j*(dx@flat))
    if body_toron is not None:
        theta = np.asarray(body_toron)
        if not (np.allclose(fields, fields[0], atol=1e-13, rtol=0) and
                np.allclose(links, law.toron_links(theta), atol=1e-13, rtol=0) and
                np.allclose(A, flat, atol=1e-13, rtol=0)):
            raise ValueError('Body FFT requires the declared uniform/true central-seam background')
        labels = np.broadcast_to(regular_mark_labels(fields[:1], mark_epsilon), fields.shape)
    else:
        labels = regular_mark_labels(fields, mark_epsilon)
    raw = (eta*phase)[:, None, None, None]*(labels@W[:, None])
    raw = raw.transpose(0, 2, 1, 3).reshape(law.volume, 4, 16)
    if body_toron is None:
        values, vectors = np.linalg.eigh(law.onebody(fields, links, A))
        filtered = vectors@(smooth_band_value(values, filter_epsilon)[:, None]*(vectors.conj().T@raw.reshape(4*law.volume, 16)))
        filtered = filtered.reshape(law.volume, 4, 16)
    else:
        # A Bloch basis change, NOT an asserted scalar SU4 gauge transformation.
        theta = np.asarray(body_toron)
        D = np.exp(-1j*(law.sites@theta)/law.size)
        p = momentum.reshape(-1, 3)+(theta/law.size+flat)/law.band.a
        h = np.broadcast_to(law.onsite*fields[0, 0], (law.volume, 4, 4)).copy()
        for axis, scalar_path in law.paths:
            K = np.einsum('m,mij->ij', scalar_path, fields[0])
            z = np.exp(1j*law.band.a*p[:, axis])
            h += z[:, None, None]*K+z.conj()[:, None, None]*K.conj().T
        values, vectors = np.linalg.eigh(h)
        transformed = np.fft.fftn((D[:, None, None]*raw).reshape((law.size,)*3+(4, 16)), axes=(0, 1, 2)).reshape(law.volume, 4, 16)
        filtered = vectors@(smooth_band_value(values, filter_epsilon)[..., None]*(vectors.conj().transpose(0, 2, 1)@transformed))
        filtered = np.fft.ifftn(filtered.reshape((law.size,)*3+(4, 16)), axes=(0, 1, 2)).reshape(law.volume, 4, 16)/D[:, None, None]
    unguarded = filtered.reshape(law.volume, 4, 4, 4).transpose(2, 0, 1, 3).reshape(4, 4*law.volume, 4)
    coefficients = (filtered*guard[:, None, None]).reshape(law.volume, 4, 4, 4).transpose(2, 0, 1, 3).reshape(4, 4*law.volume, 4)
    return RadialPacket(law.size, law.band.a, (float(origin[0]), float(origin[1]), float(origin[2])), reference,
                       sigma, bandwidth, core_radius, outer_radius, filter_epsilon, mark_epsilon,
                       immutable_array(law.sites), immutable_array(law.target), immutable_array(eta),
                       immutable_array(guard), immutable_array(coefficients), immutable_array(unguarded),
                       immutable_array(primitive_couplings(law, fields, links, A)),
                       immutable_array(links*np.exp(1j*A[:, :, None, None])), paths,
                       immutable_array(dx), active, tuple(sorted(halo)),
                       'FULL periodic h/filter and every raw Wilson/field dependency retained',
                       None if body_toron is None else
                       (float(body_toron[0]), float(body_toron[1]), float(body_toron[2])))


def radial_source_directional_derivative(law, packet, fields, links, A,
                                        field_dot, link_dot, A_dot):
    """Full Phi mark, horizontal Wilson, Coulomb/harmonic and band-filter jet."""
    labels, dlabels = regular_mark_labels(fields, packet.mark_epsilon, field_dot)
    W, paths, dx, dW = source_tree(law, links, packet.reference, link_dot)
    if paths != packet.paths:
        raise ValueError('Directional derivative must use the SAME reference tree')
    flat, dflat = np.mean(A, axis=0), np.mean(A_dot, axis=0)
    phase_angle = coulomb_source_phase(law, A, packet.reference)-dx@flat
    phase_dot = coulomb_source_phase(law, A_dot, packet.reference)-dx@dflat
    scalar = packet.envelope*np.exp(1j*phase_angle)
    raw = scalar[:, None, None, None]*(labels@W[:, None])
    draw = scalar[:, None, None, None]*(dlabels@W[:, None]+labels@dW[:, None]+1j*phase_dot[:, None, None, None]*(labels@W[:, None]))
    raw = raw.transpose(0, 2, 1, 3).reshape(4*law.volume, 16)
    draw = draw.transpose(0, 2, 1, 3).reshape(4*law.volume, 16)
    H = law.onebody(fields, links, A)
    dH = np.zeros_like(H)
    for x in range(law.volume):
        sx = slice(4*x, 4*x+4)
        dH[sx, sx] += law.onsite*field_dot[x, 0]
        for axis, scalar_path in law.paths:
            y = int(law.target[x, axis]); sy = slice(4*y, 4*y+4)
            Fx, Fy = np.einsum('m,mij->ij', scalar_path, fields[x]), np.einsum('m,mij->ij', scalar_path, fields[y])
            dFx, dFy = np.einsum('m,mij->ij', scalar_path, field_dot[x]), np.einsum('m,mij->ij', scalar_path, field_dot[y])
            U, dU = links[x, axis], link_dot[x, axis]
            K = (Fx@U+U@Fy)/2
            dK = ((dFx@U+Fx@dU+dU@Fy+U@dFy)/2+1j*A_dot[x, axis]*K)*np.exp(1j*A[x, axis])
            Ud = U.conj().T
            dUd = -Ud@dU@Ud
            dFxd = np.einsum('m,mij->ij', scalar_path.conj(), field_dot[x])
            dFyd = np.einsum('m,mij->ij', scalar_path.conj(), field_dot[y])
            dKd = ((dUd@Fx.conj().T+Ud@dFxd+dFyd@Ud+Fy.conj().T@dUd)/2
                   -1j*A_dot[x, axis]*K.conj().T)*np.exp(-1j*A[x, axis])
            dH[sx, sy] += dK; dH[sy, sx] += dKd
    values, vectors = np.linalg.eigh(H)
    f = smooth_band_value(values, packet.filter_epsilon)
    eigen_raw = vectors.conj().T@raw
    dF = smooth_band_divided_difference(values, packet.filter_epsilon)*(vectors.conj().T@dH@vectors)
    result = vectors@(dF@eigen_raw+f[:, None]*(vectors.conj().T@draw))
    result = result.reshape(law.volume, 4, 4, 4)*packet.guard[:, None, None, None]
    return immutable_array(result.transpose(2, 0, 1, 3).reshape(4, 4*law.volume, 4))


@dataclass(frozen=True, slots=True, eq=False)
class ForcedHarmonicNormalClock(ArraySnapshot):
    """Immutable complete scalar/SU4 affine-then-free comparator, shared by RIGHT."""
    Lambda: float
    duration: float
    smoothness: int
    frequencies: np.ndarray
    force_squared: np.ndarray
    source_chat: np.ndarray
    counts: tuple[tuple[str, Any], ...]
    affine_B_bound: float

    @property
    def relative_form_delta(self):
        pulse = SmoothPulse(self.smoothness)
        return 2*np.pi*float(pulse.value(.5))*self.affine_B_bound/(self.duration*float(self.frequencies.min()))

    def at(self, time_fraction):
        if not 0 <= time_fraction <= 2:
            raise ValueError('The actual two-leg clock runs over 0<=t/tau<=2')
        pulse = SmoothPulse(self.smoothness)
        report = coherent_history(self.frequencies, self.force_squared,
                                  self.Lambda, self.duration, pulse, time_fraction)
        delta = self.relative_form_delta
        report.update(normal_scope='ALL90V scalar/SU4 harmonic axes; second leg affine force ZERO',
                      affine_non_skew_covariance_majorant=math.exp(4*delta/(1-delta))*math.sqrt((1+delta)/(1-delta)),
                      relative_form_delta=delta,
                      photon_history_owner='actual separate photon sector on THIS same time',
                      exact_scalar_well_and_whole_heavy_offsets_owner='caller retains their SAME exact common propagator',
                      interacting_scalar_ground_computed=False)
        if time_fraction >= 1:
            positive = self.force_squared > 0
            if np.any(positive):
                log_integral = pulse.endpoint_integral_log_upper(self.frequencies[positive]*self.duration)
                log_occupation = float(np.logaddexp.reduce(
                    np.log(self.force_squared[positive])+np.log(self.frequencies[positive])
                    -math.log(2)-math.log(self.Lambda**-.5)+2*math.log(2*np.pi)+2*log_integral))
                report['endpoint_coherent_occupation_log_upper'] = log_occupation
                report['endpoint_return_norm_log_upper'] = min(math.log(2), log_occupation/2)
            else:
                report['endpoint_coherent_occupation'] = 'exactly zero FOR THIS zero-force harmonic model'
        return report


def forced_harmonic_normal_clock(law, frame, centre, chat, scale, duration,
                                 *, affine_B_bound, smoothness=8):
    """Actual source geometry in; no fabricated frequency, gap or frozen RIGHT."""
    if scale < 1 or duration <= 0 or affine_B_bound < 0:
        raise ValueError('Need finite positive scale/time and nonnegative actual B bound')
    omega, force, counts = native_modes(law, frame, centre, chat, scale)
    clock = ForcedHarmonicNormalClock(scale, duration, smoothness,
                                     immutable_array(omega), immutable_array(force),
                                     immutable_array(chat), tuple(counts.items()), affine_B_bound)
    if clock.relative_form_delta >= .25:
        raise ValueError('Actual full affine form is outside the admitted delta<1/4 regime')
    return clock


def stationary_normal_coefficient_factors(law, frame, centre, scale):
    """Complete source-free harmonic factors; NOT an interacting C_N/N0 proof."""
    omega, _, counts = native_modes(law, frame, centre, np.zeros((law.size,)*3+(105,), complex), scale)
    chi = scale**-.5
    Q, P = chi/(2*omega), omega/(2*chi)
    trQ, trP = float(Q.sum()), float(P.sum())
    return dict(Lambda=scale, centre=tuple(centre), scope='complete native harmonic ground only',
                axes=len(omega), counts=counts, omega_min=float(omega.min()),
                omega_max=float(omega.max()), coordinate_RMS=math.sqrt(trQ),
                chi_momentum_RMS=chi*math.sqrt(trP),
                coordinate_fourth=trQ**2+2*float(Q@Q),
                momentum_fourth=trP**2+2*float(P@P),
                mixed_Weyl=trQ*trP,
                ordered_xp_norm_squared=trQ*trP+len(omega)/2,
                coefficient_bound='T*(L0*coordinate_RMS+L1*chi_momentum_RMS+chi*L2/2)/sqrt(actual_initial_G0_min)',
                interacting_initial_preparation_and_nonlinear_remainder='NOT certified by these factors',
                ordinary_scalar_and_heavy_offsets='same exact history, never replaced by harmonic factors')


def all64(law, b, velocity, eta, R, packet):
    zero, frame0 = original_frame(law.band, law.size, np.zeros(3), True)
    base = OriginalSpatial(law.band, law.size, law.basis, coordinates(frame0, law.basis).reshape(-1), np.zeros(3))
    _, _, spins, _ = spin_fields(frame0[None])
    s = 2*np.einsum('aij,ji->a', law.generators, spins[0, 2]).real
    inertia = base.D.T@base.D
    lam = float(s@inertia@s/(s@s))
    torque = b+laplacian(law, b)/(.6*lam)
    rows, native = [], {}
    maximum = 0.
    for centre in itertools.product(range(4), repeat=3):
        angles = np.pi/2*np.array(centre)
        selected, frame = original_frame(law.band, law.size, angles, True)
        model, ec, chat, ward = geometry(law, frame, torque)
        ell = np.einsum('xa,aij->xij', ec, law.generators)
        _, _, spin, _ = spin_fields(frame[None])
        target = b[:, None, None]*spin[0, 2]
        if centre == (0, 0, 0):
            accept(np.linalg.norm(ell-target), 'precompensated actual isometric mechanical plateau', 3e-7)
        links = law.toron_links(angles)
        o = scalar_orbital(law, links, np.zeros((law.volume, 3)), velocity)
        active = np.flatnonzero(np.any(abs(o) > 1e-14, axis=0) | np.any(abs(o) > 1e-14, axis=1))
        active = np.unique(np.r_[active, np.concatenate([np.arange(4*x, 4*x+4) for x in packet])])
        q = sparse.block_diag(list(ell)).toarray()
        oa, qa = o[np.ix_(active, active)], q[np.ix_(active, active)]
        E = expm(-2j*np.pi*oa)@expm(2j*np.pi*(oa+qa))
        fields = np.tile(frame, (law.volume, 1, 1, 1))
        M = packet_marks(law, fields, links, np.zeros((law.volume, 3)), eta, R)
        defect = np.array([E@m[active]@expm(-2j*np.pi*ell[R])+m[active] for m in M])
        # Final neutral density need not remain CT. Use the ENTIRE CAR upper,
        # Tr(delta†delta), divided ONLY by the initial G=2I.
        residual = np.array([[np.vdot(a, z) for z in defect] for a in defect])/2
        error = math.sqrt(max(0., float(np.linalg.eigvalsh((residual+residual.conj().T)/2)[-1])))
        maximum = max(maximum, error)
        initial_hs = np.array([[np.vdot(a, z) for z in M] for a in M])/2
        accept(np.linalg.norm(initial_hs-2*np.eye(4)), 'CT full coherent packet initial metric all true Grams', 2e-7)
        rows.append(dict(centre=centre, G=selected['G'], classical_offset=law.volume*(selected['value']-zero['value']),
                         exact_scalar_offset_owner='actual scalar-well E_c, not this classical diagnostic',
                         true_mechanical_plateau_defect=float(np.linalg.norm(ell-target)),
                         reference_leakage=float(np.linalg.norm(ell[R])),
                         source_relative_CAR_HS_upper=error, source_error_matrix_upper=residual,
                         active_onebody_rows=len(active), all15_Gauss_error=ward,
                         q0_full_source_force_not_assumed_zero=float(np.linalg.norm(chat[0, 0, 0]))))
        native[centre] = (selected, frame, ec, chat)
    # No weights are assigned: any actual CT-neutral well mixture is bounded by max.
    return rows, native, torque, maximum


def native_modes(law, frame, centre, chat, scale):
    model = OriginalSpatial(law.band, law.size, law.basis, coordinates(frame, law.basis).reshape(-1), np.pi/2*np.array(centre))
    chi = scale**(-.5)
    frequencies, forces = [], []
    ward = source_reconstruction = kernel_error = riccati_error = 0.
    soft_force = 0.
    for n in itertools.product(range(law.size), repeat=3):
        q = 2*np.pi*np.array(n)/law.size
        G = model.gauge_columns(q)
        Z = null_space((np.sqrt(MASS)[:, None]*G).conj().T, rcond=1e-12)
        if Z.shape != (105, 90):
            raise AssertionError('Full scalar+colour quotient must have90 modes per q')
        parts = model.hessian_parts(q)
        H0 = parts['local']+parts['sea']+parts['contact']
        H1 = parts['alignment']+plaquette_hessian(q)
        transform = Z/np.sqrt(MASS)[:, None]
        A = transform.conj().T@H0@transform
        B = transform.conj().T@H1@transform
        rho = scale**.25
        if n == (0, 0, 0):
            # The actual q0 alignment/curl form has45 EXACT scalar kernel axes
            # and45 link axes. Keep all90 and all H0 cross couplings. A naive
            # eigh(A+rho B) loses the soft spectrum once rho exceeds float precision.
            bvalues, U = np.linalg.eigh((B+B.conj().T)/2)
            kernel = U[:, abs(bvalues) <= 1e-9*max(1., abs(bvalues[-1]))]
            if kernel.shape[1] != 45:
                raise AssertionError('q0 has45 soft plus45 stiff; not45 total normals')
            kernel_error = float(np.linalg.norm(B@kernel)/max(1., np.linalg.norm(B)))
            accept(kernel_error, 'actual analytic q0 alignment/curl kernel', 3e-13)
            layered = U.conj().T@A@U
            S, C, T = layered[:45, :45], layered[:45, 45:], layered[45:, 45:]
            Kt = T+rho*np.diag(bvalues[45:])
            X = np.zeros((45, 45), complex)
            for _ in range(16):
                X = np.linalg.solve(Kt, -C.conj().T+X@S+X@C@X)
                residual = C.conj().T+Kt@X-X@S-X@C@X
                riccati_error = float(np.linalg.norm(residual)/max(1., np.linalg.norm(S), np.linalg.norm(C)))
                if riccati_error <= 3e-13:
                    break
            accept(riccati_error, 'complete q0 soft/stiff H0 Riccati coupling', 3e-13)
            rv, rz = np.linalg.eigh(np.eye(45)+X.conj().T@X)
            Rs = (rz/np.sqrt(rv))@rz.conj().T
            rv, rz = np.linalg.eigh(np.eye(45)+X@X.conj().T)
            Rt = (rz/np.sqrt(rv))@rz.conj().T
            Ks = Rs@(S+C@X+X.conj().T@C.conj().T+X.conj().T@Kt@X)@Rs
            Kf = Rt@(Kt-C.conj().T@X.conj().T-X@C+X@S@X.conj().T)@Rt
            vs, us = np.linalg.eigh((Ks+Ks.conj().T)/2)
            vf, uf = np.linalg.eigh((Kf+Kf.conj().T)/2)
            rotation = np.block([[Rs, -X.conj().T@Rt], [X@Rs, Rt]])
            basis = U@rotation
            vectors = np.column_stack((basis[:, :45]@us, basis[:, 45:]@uf))
            values = chi*scale*np.r_[vs, vf]
            soft_force = float(np.linalg.norm(kernel.conj().T@Z.conj().T@chat[n]))
        else:
            # Every nonzero-q B is positive on ALL90 physical axes.
            K = B+A/rho
            values, vectors = np.linalg.eigh((K+K.conj().T)/2)
            values *= chi*scale**1.25
        if values.min() <= 0:
            raise AssertionError('Native complete normal form is not positive at chosen Lambda')
        modes = Z@vectors
        c = modes.conj().T@chat[n]
        source_reconstruction = max(source_reconstruction, float(np.linalg.norm(chat[n]-modes@c)))
        ward = max(ward, float(np.linalg.norm(G.conj().T@(np.sqrt(MASS)[:, None]*modes))))
        frequencies.extend(np.sqrt(values))
        forces.extend(abs(c)**2)
    accept(source_reconstruction, 'ALL90V native quantum source reconstruction', 3e-7)
    accept(ward, 'ALL90V quantum mode mass Gauss', 3e-7)
    return np.array(frequencies), np.array(forces), dict(
        real_normal_modes=90*law.volume, q0_soft=45, q0_stiff=45,
        nonzero_q_stiff=90*(law.volume-1), complete_source_reconstruction=source_reconstruction,
        all15_mass_Gauss=ward, actual_uniform_soft_force=soft_force,
        q0_exact_H1_kernel_error=kernel_error, q0_full_H0_Riccati_error=riccati_error,
        no_q_or_occupation_projection=True)


class SmoothPulse:
    """Normalized beta polynomial; r-1 endpoint derivatives vanish exactly."""
    def __init__(self, order=8):
        self.order = order
        self.normalization = float((2*order+1)*math.comb(2*order, order))
        self.polynomial = np.polynomial.Polynomial(
            [0.]*order+[self.normalization*(-1)**k*float(comb(order, k, exact=True)) for k in range(order+1)])
        self.derivatives = [self.polynomial.deriv(k) for k in range(2*order+1)]

    def value(self, u):
        return self.normalization*np.asarray(u)**self.order*(1-np.asarray(u))**self.order

    def derivative_value(self, k, u):
        """Factorized beta derivative: no Horner cancellation at the midpoint."""
        r = self.order
        terms = []
        for j in range(max(0, k-r), min(k, r)+1):
            coefficient = (math.comb(k, j)*(math.factorial(r)//math.factorial(r-j))
                           *(math.factorial(r)//math.factorial(r-k+j))*(-1)**(k-j))
            terms.append(coefficient*u**(r-j)*(1-u)**(r-k+j))
        return self.normalization*math.fsum(terms)

    def integral(self, z, u=1.):
        """Integral_0^u w(s) exp(i z s) ds, with no oscillatory time stepping."""
        return self._integral_kernel(z, u, 0.)

    def retarded_integral(self, z, time_fraction):
        """Integral_0^min(u,1) w(s) exp(i z(s-u)) ds; cancel phases BEFORE rounding."""
        return self._integral_kernel(z, min(time_fraction, 1.), time_fraction)

    def _integral_kernel(self, z, stop, observation):
        z = np.asarray(z, float)
        result = np.empty(z.shape, complex)
        low = abs(z) < 80
        if np.any(low):
            nodes, weights = np.polynomial.legendre.leggauss(96)
            points = (nodes+1)*stop/2
            result[low] = np.exp(1j*z[low, None]*(points-observation))@(weights*self.value(points)*stop/2)
        high = ~low
        zh = z[high]
        total = np.zeros_like(zh, complex)
        stop_phase = 1. if stop == observation else np.exp(1j*zh*(stop-observation))
        initial_phase = 1. if observation == 0. else np.exp(-1j*zh*observation)
        inverse = 1/(1j*zh)
        inverse_power = inverse.copy()
        for k in range(2*self.order+1):
            at0 = (math.factorial(k)*self.normalization*(-1)**(k-self.order)*float(comb(self.order, k-self.order, exact=True))
                   if self.order <= k <= 2*self.order else 0.)
            atstop = (-1)**k*at0 if stop == 1. else self.derivative_value(k, stop)
            total += (-1)**k*(atstop*stop_phase-at0*initial_phase)*inverse_power
            inverse_power *= inverse
        result[high] = total
        return result

    def endpoint_integral_log_upper(self, z):
        """Exact integration-by-parts triangle bound, valid below float range."""
        terms = []
        for k in range(self.order, 2*self.order+1):
            at0 = math.factorial(k)*self.normalization*float(comb(self.order, k-self.order, exact=True))
            terms.append(math.log(2*at0)-(k+1)*np.log(z))
        return np.minimum(0., np.logaddexp.reduce(terms))

    def response_action(self, z, u):
        """Integral_0^u w(t) Im[int_0^t w(s)e^(-iz(t-s))ds]dt.

        Its negative is the forced-oscillator Magnus phase kernel. High-frequency
        terms use EXACT finite polynomial antiderivatives, not sampled fast sines.
        """
        z = np.asarray(z, float)
        result = np.zeros_like(z)
        high = abs(z) >= 80
        zh = z[high]
        nodes, weights = np.polynomial.legendre.leggauss(96)
        points = (nodes+1)*u/2
        measure = weights*self.value(points)*u/2
        integral = self.integral(-zh, u)
        value = np.zeros(zh.shape, dtype=np.complex128)
        inverse = 1/(1j*zh)
        inverse_power = inverse.copy()
        for k, derivative in enumerate(self.derivatives):
            product = float(measure@derivative(points))
            at0 = (math.factorial(k)*self.normalization*(-1)**(k-self.order)*float(comb(self.order, k-self.order, exact=True))
                   if self.order <= k <= 2*self.order else 0.)
            value += (-1)**k*(product-at0*integral)*inverse_power
            inverse_power *= inverse
        result[high] = value.imag
        low = ~high
        zl = z[low]
        for t, weight in zip(points, measure):
            response = self.retarded_integral(zl, float(t))
            result[low] += weight*np.imag(response)
        return result


def switching_controls(pulse):
    """Independent Fourier identity and the force-centroid midpoint invariant."""
    normalization = quad(lambda u: float(pulse.value(u)), 0, 1, epsabs=1e-12)[0]
    accept(abs(normalization-1), 'smooth source has actual total angle2pi', 2e-10)
    sample = np.array([.1, 3., 90., 500.])
    order = pulse.order
    exact = (np.exp(1j*sample/2)*gamma(order+1.5)
             *(4/sample)**(order+.5)*jv(order+.5, sample/2))
    fourier_error = float(np.linalg.norm(pulse.integral(sample)-exact))
    accept(fourier_error, 'independent smooth switching Fourier solution', 2e-8)
    large_z = np.array([1e26, 1e35])
    real = abs(pulse.retarded_integral(large_z, .5).real)
    # At the symmetric midpoint every odd derivative vanishes. Even
    # derivatives give imaginary terms; ONLY the initial endpoint contributes
    # to the real force-centroid kernel. Its ONE-endpoint triangle bound is
    # half the existing two-endpoint bound, independent of phase arithmetic.
    log_upper = pulse.endpoint_integral_log_upper(large_z)-math.log(2)
    positive = real > 0
    margin = float(np.max(np.log(real[positive])-log_upper[positive])) if np.any(positive) else None
    if margin is not None:
        accept(max(0., margin), 'actual harmonic midpoint real-centroid bound', 2e-10)
    return dict(normalization_error=abs(normalization-1), Fourier_error=fourier_error,
                midpoint_real_observed=real, midpoint_real_log_upper=log_upper,
                midpoint_real_log_bound_margin=margin,
                underflow_does_not_mean_exact_zero=True)


def coherent_history(omega, force_squared, scale, duration, pulse, u, angle=2*np.pi) -> dict[str, Any]:
    chi, tau = scale**(-.5), duration
    z = omega*tau
    # Leg2 has no affine force: retain its actual free evolution and phase,
    # while its nonzero B2 squeezing remains in the full two-leg majorant.
    stop = min(u, 1.)
    integral = angle*pulse.retarded_integral(z, u)
    x2 = force_squared*integral.real**2
    p2 = force_squared*(omega/chi)**2*integral.imag**2
    Q, P = chi/(2*omega), omega/(2*chi)
    meanx2, meanp2 = float(x2.sum()), float(p2.sum())
    trq, trp = float(Q.sum()), float(P.sum())
    occupation = float((force_squared*omega/(2*chi)*abs(integral)**2).sum())
    # Exact ordered Wick contraction, including the noncommutative d/2 term.
    xp2 = (meanx2+trq)*(meanp2+trp)+len(omega)/2
    x4 = (meanx2+trq)**2+2*float((Q*Q).sum())+4*float((Q*x2).sum())
    p4 = (meanp2+trp)**2+2*float((P*P).sum())+4*float((P*p2).sum())
    action = -angle**2*float((force_squared*omega/(2*chi)*pulse.response_action(z, stop)).sum())
    return dict(time_fraction=u, physical_time=tau*u, coordinate_second_moment=meanx2+trq,
                coordinate_fourth_moment=x4, momentum_second_moment=meanp2+trp,
                momentum_fourth_moment=p4, ordered_xp_norm_squared=xp2,
                centroid_coordinate_norm=float(stable_vector_norm(np.sqrt(force_squared)*integral.real)),
                centroid_momentum_norm=float(stable_vector_norm(np.sqrt(force_squared)*omega/chi*integral.imag)),
                coherent_occupation=occupation, frozen_vacuum_overlap_modulus=math.exp(-min(occupation/2, 750)),
                common_RIGHT_overlap=1., shared_history_norm_error=0.,
                completion_square_energy_at_time=(
                    -angle**2*float(pulse.value(u))**2*float(force_squared.sum())/(2*chi*tau**2)
                    if u <= 1. else 0.),
                harmonic_zero_point_exact=float(omega.sum()/2),
                retained_forced_Magnus_action=action,
                retained_harmonic_common_phase=action-float(omega.sum()/2)*tau*u,
                phase_representation='unwrapped action in radians; no inaccurate reduction of a huge common phase to a final metric',
                common_scalar_phase_not_deleted=True)


def quantum_normal_case(law, data, centre, scale, power, pulse, B_bound):
    _, frame, _, chat = data[centre]
    requested = scale
    while True:
        omega, force, counts = native_modes(law, frame, centre, chat, scale)
        duration = scale**(-power)
        delta = 2*np.pi*float(pulse.value(.5))*B_bound/(duration*float(omega.min()))
        if delta <= .25:
            break
        # A MATERIAL scaling repair, not deleting B or calling a failed
        # positive-form estimate a proof. Every chosen Lambda stays finite.
        scale *= 1e4
        if not np.isfinite(scale):
            raise ValueError('Native affine onset exceeds floating range; use analytic joint family')
    histories = [coherent_history(omega, force, scale, duration, pulse, u) for u in (.25, .5, .75, 1., 2.)]
    probe = histories[1]
    # A single common factor is exactly normalized; compare at the SAME Gram/cross boundary.
    frozen_error_lower = 1-probe['frozen_vacuum_overlap_modulus']
    rejection = adverse(frozen_error_lower, 'same common normal RIGHT Gram/cross', 3e-7)
    positive = force > 0
    log_integral_upper = pulse.endpoint_integral_log_upper(omega[positive]*duration)
    log_occupation_upper = float(np.logaddexp.reduce(
        np.log(force[positive])+np.log(omega[positive])-math.log(2)-math.log(scale**(-.5))
        +2*math.log(2*np.pi)+2*log_integral_upper))
    log_return_upper = min(math.log(2), log_occupation_upper/2)
    return dict(centre=centre, requested_Lambda=requested, Lambda=scale, power=power,
                duration=duration, total_history_time=2*duration,
                peak_physical_source_strength=2*np.pi*float(pulse.value(.5))/duration,
                affine_B_norm_upper=B_bound, affine_relative_positive_form_delta=delta,
                full_two_leg_squeezed_covariance_upper_factor=math.exp(4*delta/(1-delta))*math.sqrt((1+delta)/(1-delta)),
                counts=counts, min_frequency=float(omega.min()), max_frequency=float(omega.max()),
                actual_source_norm_squared=float(force.sum()), histories=histories,
                endpoint_coherent_occupation_log_upper=log_occupation_upper,
                endpoint_return_norm_log_upper=log_return_upper,
                endpoint_return_norm_upper=math.exp(max(log_return_upper, math.log(np.nextafter(0., 1.)))),
                endpoint_float_underflow_owner='coherent occupation may round to0; log upper retains the actual nonzero return payment',
                frozen_normal_same_boundary=rejection,
                frozen_RIGHT_cross_error_lower=frozen_error_lower,
                moving_RIGHT_same_boundary='derived exact Weyl unitarity on ALL90V axes; no numerical scalar-ground claim',
                common_phase_owner='exact forced-oscillator Magnus action plus EACH actual scalar-well/heavy offset, identical in both histories',
                normal_quantum_scope='complete infinite-axis forced harmonic coherent history; full affine squeezing has the proved covariance majorant, not a computed scalar-well ground')


def native_normal_jet(law, frame, centre, torque, velocity, rng):
    """Complete RNC first jet with native moving Gauss, ALL axes and BCH contact."""
    from scipy.sparse.linalg import splu
    V = law.volume
    links = law.toron_links(np.pi/2*np.array(centre))
    fields = np.tile(frame, (V, 1, 1, 1))
    mass = np.tile(MASS, V)
    root = np.sqrt(mass)

    def gauss(phi, u):
        Df = np.stack([coordinates(1j*(t@phi-phi@t), law.basis).reshape(V, 60).real for t in law.generators], axis=-1)
        rows, columns, values = [], [], []
        def block(r, c, a):
            ii, jj = np.nonzero(a)
            rows.extend(r+ii); columns.extend(c+jj); values.extend(a[ii, jj])
        for x in range(V):
            block(105*x, 15*x, Df[x])
            for axis in range(3):
                y = int(law.target[x, axis])
                transported = u[x, axis]@law.generators@u[x, axis].conj().T
                Ad = 2*np.einsum('aij,bji->ab', law.generators, transported).real
                block(105*x+60+15*axis, 15*x, np.eye(15))
                block(105*x+60+15*axis, 15*y, -Ad)
        G = sparse.coo_matrix((values, (rows, columns)), shape=(105*V, 15*V)).tocsr()
        Gt = sparse.csr_matrix(sparse.diags(root)@G)
        return Gt, splu(sparse.csc_matrix(Gt.T@Gt))

    def pack(df, h):
        fc = coordinates(df, law.basis).real.reshape(V, 60)
        hc = 2*np.einsum('aij,xeji->xea', law.generators, h).real
        return root*np.concatenate((fc, hc.reshape(V, 45)), axis=1).reshape(-1)

    def unpack(q):
        raw = (q/root).reshape(V, 105)
        return (np.einsum('xma,aij->xmij', raw[:, :60].reshape(V, 4, 15), law.basis),
                np.einsum('xea,aij->xeij', raw[:, 60:].reshape(V, 3, 15), law.generators))

    Gt, factor = gauss(fields, links)
    def project(q):
        return q-Gt@factor.solve(np.asarray(Gt.T@q))
    flow = mechanical_flow(law, fields, links, np.zeros((V, 3)), velocity, torque)
    left = -1j*flow[1]@links.conj().swapaxes(-1, -2)
    source = pack(flow[0], left)
    coefficients = factor.solve(np.asarray(Gt.T@source)).reshape(V, 15)
    ell = np.einsum('xa,aij->xij', coefficients, law.generators)
    affine = project(source)
    q = project(rng.normal(size=105*V))
    q /= np.linalg.norm(q)
    df, h = unpack(q)

    expected_field = np.zeros_like(df)
    expected_field[:, 1] -= torque[:, None, None]*df[:, 2]
    expected_field[:, 2] += torque[:, None, None]*df[:, 1]
    for axis in range(3):
        plus = law.target[:, axis]
        minus = np.array([law.index[tuple((x-np.eye(3, dtype=int)[axis]) % law.size)] for x in law.sites])
        hsum = h[:, axis]+h[minus, axis]
        difference = (df[plus]-df[minus]+1j*(hsum[:, None]@frame-frame@hsum[:, None]))/(2*law.band.a)
        expected_field -= velocity[:, axis, None, None, None]*difference
    expected_field -= 1j*(ell[:, None]@df-df@ell[:, None])
    b4 = np.zeros((V, 3, 3, 4, 4), complex)
    for x in range(V):
        for j, i in itertools.combinations(range(3), 2):
            for s, t in itertools.product((-1, 1), repeat=2):
                y, flux = x, np.zeros((4, 4), complex)
                for axis, sign in ((j, s), (i, t), (j, -s), (i, -t)):
                    if sign == 1:
                        flux += h[y, axis]; y = int(law.target[y, axis])
                    else:
                        site = law.sites[y].copy(); site[axis] = (site[axis]-1) % law.size
                        y = law.index[tuple(site)]; flux -= h[y, axis]
                b4[x, j, i] += s*t*flux/(4*law.band.a**2)
            b4[x, i, j] = -b4[x, j, i]
    expected_link = np.zeros_like(h)
    bch = np.zeros_like(h)
    for x in range(V):
        for i in range(3):
            y = law.target[x, i]
            for j in range(3):
                expected_link[x, i] -= law.band.a*(velocity[x, j]+velocity[y, j])*(b4[x, j, i]+b4[y, j, i])/4
            both = ell[x]+ell[y]
            bch[x, i] = -.5j*(both@h[x, i]-h[x, i]@both)
    expected = project(pack(expected_field, expected_link+bch))

    def curve(t):
        moved = np.array([[expm(1j*t*z)@u for z, u in zip(zh, uh)] for zh, uh in zip(h, links)])
        phi = fields+t*df
        ff, fu, _ = mechanical_flow(law, phi, moved, np.zeros((V, 3)), velocity, torque)
        fl = -1j*fu@moved.conj().swapaxes(-1, -2)
        gt, fac = gauss(phi, moved)
        raw = pack(ff, fl)
        horizontal = raw-gt@fac.solve(np.asarray(gt.T@raw))
        hf, hl = unpack(horizontal)
        pulled = np.empty_like(hl)
        for x in range(V):
            for axis in range(3):
                val, vec = np.linalg.eigh(t*h[x, axis])
                difference = val[:, None]-val[None, :]
                multiplier = np.exp(-.5j*difference)/np.sinc(difference/(2*np.pi))
                pulled[x, axis] = vec@((vec.conj().T@hl[x, axis]@vec)*multiplier)@vec.conj().T
        return project(pack(hf, pulled))

    step = 2e-5
    numerical = (curve(step)-curve(-step))/(2*step)
    error = float(np.linalg.norm(numerical-expected))
    accept(error, 'SAME FULL native mass-horizontal normal RNC jet', 3e-6)
    wrong = project(pack(expected_field, expected_link))
    bch_error = float(np.linalg.norm(numerical-wrong))
    Dfield = np.stack([coordinates(1j*(t@frame-frame@t), law.basis).reshape(60).real for t in law.generators], axis=-1)
    vsum = float(np.max(abs(velocity), axis=0).sum())
    ellmax = max(float(np.linalg.norm(e, ord=2)) for e in ell)
    # Central differences, field-link incidence and every anchored clover.
    B_bound = float(np.max(abs(torque)))+vsum/law.band.a*(17+math.sqrt(.6)*float(np.linalg.norm(Dfield, ord=2)))+2*ellmax
    return dict(raw_coordinates=105*V, colour_Gauss_rows=15*V, physical_axes=90*V,
                affine_source_norm=float(np.linalg.norm(affine)), full_source_jet_norm=float(np.linalg.norm(expected)),
                complete_source_jet_error=error,
                source_B_operator_norm_upper=B_bound,
                omit_BCH=(adverse(bch_error, 'SAME FULL native mass-horizontal normal RNC jet', 3e-6)
                          if np.any(torque) else 'exactly zero at torqueless BODY; entire B remains nonzero'),
                omit_entire_B=adverse(float(np.linalg.norm(numerical)), 'SAME FULL native mass-horizontal normal RNC jet', 3e-6),
                half_divergence_and_squeezing_owner='sym(By.p), not removed by a sampled trace or zero source mean',
                no_selected_normal_direction_as_Hamiltonian=True)


def native_flow_controls(law, frame, centre, torque, velocity, eta, R, rng):
    links = law.toron_links(np.pi/2*np.array(centre))
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    fields += 3e-4*np.einsum('xma,aij->xmij', rng.normal(size=(law.volume, 4, 15)), law.basis)
    A = rng.normal(scale=.6, size=(law.volume, 3))
    flow = mechanical_flow(law, fields, links, A, velocity, torque)
    gauges = np.array([expm(.03j*np.einsum('a,aij->ij', r, law.generators)) for r in rng.normal(size=(law.volume, 15))])
    angles = rng.normal(scale=.2, size=law.volume)
    moved = gauge_move(law, fields, links, A, gauges, angles)
    moved_flow = mechanical_flow(law, *moved, velocity, torque)
    expected = (gauges[:, None]@flow[0]@gauges[:, None].conj().swapaxes(-1, -2),
                gauges[:, None]@flow[1]@gauges[law.target].conj().swapaxes(-1, -2), flow[2])
    gauge_error = max(float(np.linalg.norm(a-z)) for a, z in zip(moved_flow, expected))
    accept(gauge_error, 'actual full mechanical stress gauge covariance', 3e-7)
    o = scalar_orbital(law, links, A, velocity)
    flow_h = endpoint_flow(law, fields, links, A, flow)
    step = 2e-5
    left = -1j*flow[1]@links.conj().swapaxes(-1, -2)
    plus = np.array([[expm(1j*step*h)@u for h, u in zip(rowh, rowu)] for rowh, rowu in zip(left, links)])
    minus = np.array([[expm(-1j*step*h)@u for h, u in zip(rowh, rowu)] for rowh, rowu in zip(left, links)])
    derivative = (law.onebody(fields+step*flow[0], plus, A+step*flow[2])-law.onebody(fields-step*flow[0], minus, A-step*flow[2]))/(2*step)
    current_error = float(np.linalg.norm(derivative-flow_h)/max(1., np.linalg.norm(flow_h)))
    accept(current_error, 'native complete mechanical endpoint current', 3e-7)
    wrong = primitive_flow(law, fields, links, A, velocity, torque)[2]
    sine_error = float(np.linalg.norm(flow[2]-wrong))
    M = packet_marks(law, fields, links, A, eta, R)
    moved_M = packet_marks(law, *moved, eta, R)
    expected_M = np.array([np.concatenate([np.exp(1j*(angles[x]-angles[R]))*gauges[x]@m[4*x:4*x+4]@gauges[R].conj().T for x in range(law.volume)]) for m in M])
    mark_error = float(np.linalg.norm(moved_M-expected_M))
    accept(mark_error, 'actual whole packet SAME-R colour and real U1 Gauss intertwiner', 3e-7)
    return dict(full_flow_Gauss_error=gauge_error, native_endpoint_current_error=current_error,
                packet_reference_covariance_error=mark_error,
                compact_sine_rejected=adverse(sine_error, 'same noncompact photon stress', 3e-7),
                scalar_orbital_Hermiticity=float(np.linalg.norm(o-o.conj().T)),
                full_Ward_expression_norm=float(np.linalg.norm(flow_h+1j*(o@law.onebody(fields, links, A)-law.onebody(fields, links, A)@o))),
                continuous_rotation_conservation_not_assumed=True)


def full_packet_case(args, selected, frame, normal_rows):
    """Actual quantum photon/light histories on EACH admitted normal pulse time.

    The graph is the isometric native interface, not the N^3 interacting ground.
    All64 nonisometric geometry remains a separate, unchanged local norm payment.
    H is unreversed in both legs/sectors; no quadrature increase pays its error.
    """
    spins, labels, geometry_report = mechanical_geometry(frame, selected)
    neutral = GraphQuantum(args.levels, args.harmonics, 0., 6, args.UE, args.K, frame)
    charged = GraphQuantum(args.levels, args.harmonics, 0., 7, args.UE, args.K, frame)
    E0, state, cluster = neutral.solve(args.eigenpairs)
    eta = np.array([math.sqrt(.6), math.sqrt(.4)*np.exp(.37j)])
    maps = creator_maps(neutral, charged, eta, labels)
    initial = np.asarray([prepare(neutral, charged, state, m) for m in maps])
    G = gram(initial); values, vectors = np.linalg.eigh(G)
    if values[0] <= 1e-9:
        raise AssertionError('Actual unfiltered initial packet metric is singular')
    inverse = (vectors/np.sqrt(values))@vectors.conj().T
    pulse = SmoothPulse(args.smoothness)
    nc, cc = Controller(neutral, spins[2], args.photon_shear), Controller(charged, spins[2], args.photon_shear)
    badn, badc = Controller(neutral, spins[2], args.photon_shear, halo=0.), Controller(charged, spins[2], args.photon_shear, halo=0.)
    intrinsic = [(Controller(neutral, s, args.photon_shear), Controller(charged, s, args.photon_shear)) for s in spins]
    label_columns = np.column_stack([v.ravel() for v in labels])
    left = [np.linalg.lstsq(label_columns, np.column_stack([(s@v).ravel() for v in labels]), rcond=None)[0] for s in spins]
    links = np.tile(np.eye(4, dtype=complex), (len(ENDS), 1, 1))
    _, _, h = endpoint_blocks(np.tile(frame, (3, 1, 1, 1)), links)
    precision = max(50, int(math.ceil(math.log10(max(row['Lambda'] for row in normal_rows))))+40)
    with mp.workdps(precision):
        h_mp = mp.matrix([[mp.mpc(float(z.real), float(z.imag)) for z in row] for row in h])
        hv_mp, hz_mp = mp.eighe(h_mp)
        occupied_mp = hz_mp[:, 6:]
        heavy_energy_mp = mp.fsum(hv_mp[j] for j in range(6, 12))
        occupied = np.array([[complex(occupied_mp[i, j]) for j in range(6)] for i in range(12)])
    heavy0 = slater(occupied, neutral.rows)
    oh = np.zeros_like(h)
    for edge, velocity in ((0, .28), (1, -.11)):
        x, y = ENDS[edge]
        block = velocity*np.eye(4)/(2j)
        oh[4*x:4*x+4, 4*y:4*y+4] += block
        oh[4*y:4*y+4, 4*x:4*x+4] += block.conj().T

    def operator_error(raw):
        whitened = initial_whiten(raw, inverse)
        return math.sqrt(max(0., float(np.linalg.eigvalsh(gram(whitened))[-1])))

    def physical_spin_interface(c, n):
        """Two-sector physical intrinsic action, NOT an imported Pauli action.

        At the relative-minus endpoint C=-I n, so the neutral term has a PLUS
        sign. Keep it on the actual evolved n, never on the initial vacuum.
        """
        casimir = np.zeros_like(c)
        errors = []
        for (nj, cj), L in zip(intrinsic, left):
            n1 = nj.intrinsic(n)
            i1 = np.asarray([prepare(neutral, charged, n1, m) for m in maps])
            charged_image = np.asarray([cj.intrinsic(x) for x in c])
            image = charged_image+i1
            target = np.einsum('ba,bij->aij', L, c)
            errors.append(operator_error(image-target))
            n2 = nj.intrinsic(n1)
            i2 = np.asarray([prepare(neutral, charged, n2, m) for m in maps])
            casimir += np.asarray([cj.intrinsic(x) for x in charged_image])
            casimir += 2*np.asarray([cj.intrinsic(x) for x in i1])-i2
        orbital_n, photon_n = nc.orbital(n), nc.photon(n)
        orbital = np.asarray([cc.orbital(x) for x in c])+np.asarray([
            prepare(neutral, charged, orbital_n, m) for m in maps])
        photon = np.asarray([cc.photon(x) for x in c])+np.asarray([
            prepare(neutral, charged, photon_n, m) for m in maps])
        z_neutral = intrinsic[2][0].intrinsic(n)
        z_image = np.asarray([intrinsic[2][1].intrinsic(x) for x in c])+np.asarray([
            prepare(neutral, charged, z_neutral, m) for m in maps])
        z_target = np.einsum('ba,bij->aij', left[2], c)
        return dict(intrinsic_three_axis_intertwining_errors=errors,
                    propagated_relative_intrinsic_Casimir_error=operator_error(casimir-.75*c),
                    physical_orbital_relative_image=operator_error(orbital),
                    physical_photon_relative_image=operator_error(photon),
                    full_first_leg_source_spin_image_defect=operator_error(z_image+orbital+photon-z_target),
                    neutral_connection_owner='actual evolved neutral photon/light state, all four opposite reference components',
                    scope='relative intrinsic ad_J Casimir only; localized orbital/EM remainder retained, no global spatial J2 claim')

    def evolve(ncontroller, ccontroller, duration, scale, steps, first_fraction=None,
               window=(1., 1., 0.), measure_spin=False):
        n, c, ho = state.copy(), initial.copy(), occupied_mp.copy()
        qh = np.kron(np.diag(window), spins[2])
        fraction = 1. if first_fraction is None else float(first_fraction)
        elapsed_time = 0.
        heavy_elapsed_mp = mp.mpf(0)
        decimal_precision = max(50, int(math.ceil(math.log10(scale)))+40)
        heavy_sources = [oh+qh, -oh]
        for leg in ((1, 2) if first_fraction is None else (1,)):
            for k in range(steps):
                angle = 2*np.pi*float(betainc(pulse.order+1, pulse.order+1, fraction*(k+1)/steps)-betainc(pulse.order+1, pulse.order+1, fraction*k/steps))
                dt = duration*fraction/steps
                elapsed_time += dt
                ns = ncontroller.first if leg == 1 else ncontroller.second
                cs = ccontroller.first if leg == 1 else ccontroller.second
                n = propagate(neutral, n, dt, angle, ns, neutral.trace)
                c = np.asarray([propagate(charged, x, dt, angle, cs, charged.trace) for x in c])
                # Actual Lambda h + gamma J, NOT a small surrogate heavy scale.
                # Diagonalize h+epsilon J at enough digits to retain epsilon;
                # only mpmath reduces its enormous eigenphases modulo2pi.
                with mp.workdps(decimal_precision):
                    heavy_elapsed_mp += mp.mpf(dt)
                    source_mp = mp.matrix([[mp.mpc(float(z.real), float(z.imag)) for z in row]
                                           for row in heavy_sources[leg-1]])
                    if dt:
                        time_strength = mp.mpf(dt)*mp.mpf(scale)
                        energies, modes = mp.eighe(h_mp+mp.mpf(angle)/time_strength*source_mp)
                    else:
                        time_strength = mp.mpf(angle)
                        energies, modes = mp.eighe(source_mp)
                    phases = mp.diag([mp.exp(1j*time_strength*energies[j]) for j in range(12)])
                    ho = modes*phases*modes.H*ho
        right = np.asarray([prepare(neutral, charged, n, m) for m in maps])
        x, y = initial_whiten(c, inverse), initial_whiten(right, inverse)
        cross = np.array([[np.vdot(a, z) for z in x] for a in y])
        whole_frame = np.array([[complex(ho[i, j]) for j in range(6)] for i in range(12)])
        common = slater(whole_frame, neutral.rows)
        with mp.workdps(decimal_precision):
            bare_phase = mp.nstr(mp.mpf(scale)*heavy_elapsed_mp*heavy_energy_mp, decimal_precision)
        common_error = accept(abs(np.vdot(common, common).real-1), 'entire evolved heavy CAR common RIGHT norm', 2e-7)
        report = dict(endpoint_operator_error=operator_error(c+right),
                      charged_Gram=gram(x), RIGHT_Gram=gram(y), raw_cross=cross,
                      physical_time=elapsed_time, heavy_Lambda=scale,
                      heavy_decimal_precision=decimal_precision,
                      heavy_unwrapped_bare_energy_phase=bare_phase,
                      heavy_phase_owner='actual ordered MP matrix exponential product; bare phase retained separately, not a replacement for source phase',
                      heavy_initial_final_overlap=np.vdot(heavy0, common),
                      whole_heavy_dimension=len(neutral.rows), full_reference_components=4,
                      whole_heavy_common_cancellation_error=common_error,
                      no_final_metric_inverse=True)
        if measure_spin:
            report['physical_spin_interface'] = physical_spin_interface(c, n)
        return report

    source = evolve(nc, cc, 0., normal_rows[0]['Lambda'], args.steps)
    accept(source['endpoint_operator_error'], 'same actual full packet relative-minus endpoint', 1e-5)
    bad = evolve(badn, badc, 0., normal_rows[0]['Lambda'], args.steps, window=(1., 0., 0.))
    source_rejection = adverse(bad['endpoint_operator_error'], 'same actual full packet relative-minus endpoint', 1e-5)
    cases = []
    for normal in normal_rows:
        duration = normal['duration']
        finite = evolve(nc, cc, duration, normal['Lambda'], args.steps, measure_spin=True)
        midpoint = evolve(nc, cc, duration, normal['Lambda'], args.steps, first_fraction=.5)
        wrong_midpoint = evolve(badn, badc, duration, normal['Lambda'], args.steps, first_fraction=.5, window=(1., 0., 0.))
        cases.append(dict(centre=normal['centre'], Lambda=normal['Lambda'], duration=duration,
                          total_history_time=2*duration, ordinary_H_unreversed=True,
                          source_angle_per_leg=2*np.pi, actual_first_leg_midpoint=midpoint,
                          cut_halo_first_leg_midpoint=wrong_midpoint, full_H=finite,
                          observed_error_over_duration=finite['endpoint_operator_error']/duration,
                          heavy_Lambda=normal['Lambda'],
                          heavy_scope='actual whole924-dimensional heavy exterior history under the SAME Lambda h and pulse clock; no source or occupied mode dropped'))
    return dict(native_geometry=geometry_report, actual_neutral_energy=E0, actual_neutral_cluster=cluster,
                initial_G=G, initial_G_eigenvalues=values, source_only=source,
                cut_packet_halo=source_rejection, history_cases=cases,
                all_light_sectors=[neutral.d, charged.d], quadrature_steps=args.steps,
                photon_scope='finite Galerkin real quadratic curl and harmonic direct-integral fibre zeta0; no full3D phase claim',
                normal_join='same actual physical first/second leg times, source profile and body frame; all native normal axes common, affine B squeezing majorant retained',
                nonisometric_scope='all64 local geometric norm payment retained separately; tilted normal cases are not labelled an isometric small-graph join')


def joined_history_boundary(packet, normal):
    """Both adverse inputs reach the SAME complete initial-G Gram/cross surface."""
    first = packet['actual_first_leg_midpoint']
    wrong = packet['cut_halo_first_leg_midpoint']
    surface = 'same full-history initial-G Gram/cross'
    relative_time_error = abs(first['physical_time']-normal['histories'][1]['physical_time'])/normal['duration']
    accept(relative_time_error, 'same actual native-normal/light physical pulse time', 3e-13)
    # No common phase has been assumed zero: reverse triangle bounds its error.
    modulus = normal['histories'][1]['frozen_vacuum_overlap_modulus']
    frozen_lower = (1-modulus)*float(np.linalg.norm(first['raw_cross'], ord=2))
    window_error = max(float(np.linalg.norm(first[key]-wrong[key], ord=2))
                       for key in ('charged_Gram', 'RIGHT_Gram', 'raw_cross'))
    return dict(centre=normal['centre'], Lambda=normal['Lambda'], time_fraction=.5,
                duration=normal['duration'], physical_time=first['physical_time'],
                normal_light_physical_time_error=relative_time_error,
                endpoint_physical_time=packet['full_H']['physical_time'],
                endpoint_error=packet['full_H']['endpoint_operator_error'],
                physical_spin_interface=packet['full_H']['physical_spin_interface'],
                frozen_normal_full_history_error_lower=frozen_lower,
                cut_source_window_full_history_error=window_error,
                frozen_normal_rejected=adverse(frozen_lower, surface, 1e-5),
                cut_source_window_rejected=adverse(window_error, surface, 1e-5),
                common_moving_recovery='actual coherent/whole-heavy unitary factor is identical in both histories, so all THREE raw forms are unchanged',
                physical_surface='charged Gram, complete neutral RIGHT Gram and raw cross, normalized ONCE by initial G',
                interface_scope='native3D complete normal factor joined to explicit native small-graph packet interface; no numerical full3D interacting ground claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--scales', nargs='+', type=float, default=[1e24, 1e96, 1e128])
    parser.add_argument('--centres', nargs='+', default=['000', '123'])
    parser.add_argument('--power', type=float, default=1/32)
    parser.add_argument('--smoothness', type=int, default=8)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--harmonics', type=int, default=2)
    parser.add_argument('--eigenpairs', type=int, default=6)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--UE', type=float, default=.37)
    parser.add_argument('--K', type=float, default=.9)
    parser.add_argument('--photon-shear', type=float, default=.17)
    args = parser.parse_args()
    if args.size < 5 or args.size % 2 != 1 or not 0 < args.spacing <= 1:
        raise ValueError('Odd N>=5 native unaliased source and fixed0<a<=1 required')
    if not 0 < args.power < 1/20 or args.smoothness < 4 or min(args.scales) <= 1:
        raise ValueError('Constructive shrinking-tube family requires0<p<1/20, smoothness>=4, Lambda>1')
    if args.steps < args.smoothness+1 or args.levels < 2 or args.harmonics < 1:
        raise ValueError('Smooth quadrature and nontrivial quantum photon space required')
    law = BulkQuantumLaw(args.size, band=RotationBand(**{**BARE, 'a': args.spacing}))
    b, velocity, eta, R, packet = profiles(law)
    rows, data, torque, geometric = all64(law, b, velocity, eta, R, packet)
    pulse = SmoothPulse(args.smoothness)
    switching_report = switching_controls(pulse)
    centres = [tuple(int(c) for c in word) for word in args.centres]
    if any(len(c) != 3 or c not in data for c in centres):
        raise ValueError('Native diagnostic centres must be three digits0..3')
    if (0, 0, 0) not in centres:
        raise ValueError('Explicit native isometric small-graph/normal join requires centre000; all64 geometry remains')
    selected, frame, _, _ = data[(0, 0, 0)]
    flow = native_flow_controls(law, frame, (0, 0, 0), torque, velocity, eta, R, np.random.default_rng(222100602))
    jet = [native_normal_jet(law, frame, (0, 0, 0), torque, velocity, np.random.default_rng(222100603)),
           native_normal_jet(law, frame, (0, 0, 0), np.zeros_like(torque), -velocity, np.random.default_rng(222100604))]
    B_bound = max(j['source_B_operator_norm_upper'] for j in jet)
    # A tilted true Gram can change BOTH D and ell. The covariance majorant uses
    # the maximum native coefficient bound, not the isometric diagnostic alone.
    vsum = float(np.max(abs(velocity), axis=0).sum())
    for _, frame_c, ec_c, _ in data.values():
        D_c = np.stack([coordinates(1j*(t@frame_c-frame_c@t), law.basis).reshape(60).real for t in law.generators], axis=-1)
        ell_c = np.einsum('xa,aij->xij', ec_c, law.generators)
        ellmax = max(float(np.linalg.norm(e, ord=2)) for e in ell_c)
        B_bound = max(B_bound, float(np.max(abs(torque)))+vsum/law.band.a*(17+math.sqrt(.6)*float(np.linalg.norm(D_c, ord=2)))+2*ellmax)
    normal = [quantum_normal_case(law, data, c, scale, args.power, pulse, B_bound)
              for c in centres for scale in args.scales]
    joined_normal = [row for row in normal if row['centre'] == (0, 0, 0)]
    packet_report = full_packet_case(args, selected, frame, joined_normal)
    joined = [joined_history_boundary(case, row)
              for case, row in zip(packet_report['history_cases'], joined_normal)]
    distinct = sorted({case['Lambda']: case for case in packet_report['history_cases']}.values(),
                      key=lambda case: case['Lambda'])
    scaling = [dict(Lambda_pair=[a['Lambda'], b['Lambda']],
                    duration_pair=[a['duration'], b['duration']],
                    error_pair=[a['full_H']['endpoint_operator_error'], b['full_H']['endpoint_operator_error']],
                    observed_duration_exponent=math.log(a['full_H']['endpoint_operator_error']/b['full_H']['endpoint_operator_error'])/math.log(a['duration']/b['duration']),
                    observed_Lambda_decay_exponent=math.log(a['full_H']['endpoint_operator_error']/b['full_H']['endpoint_operator_error'])/math.log(b['Lambda']/a['Lambda']))
               for a, b in zip(distinct, distinct[1:])]
    result: dict[str, Any] = dict(plan=PLAN, revision=REVISION, template=TEMPLATE,
        law='original Phi/SU4/whole-heavy/light plus explicitly different noncompact quadratic EM',
        all64_true_Grams=rows, all64_source_error_max=geometric,
        well_weights='actual neutral CT ensemble; never assigned equal weights',
        all64_exact_scalar_offsets='kept symbolically in the SAME scalar propagator; classical offsets above are NOT replacements',
        physical_reference_site=int(R), complex_finite_packet_sites=packet, complex_eta=eta[packet],
        complete_native_flow=flow, full_native_normal_jet=jet, full_quantum_normal=normal, native_quantum_packet=packet_report,
        smooth_switching_controls=switching_report,
        joined_full_history_adverse=joined,
        physical_duration_scaling=scaling, distinct_admitted_packet_Lambdas=[row['Lambda'] for row in distinct],
        pulse_trajectory=dict(power=args.power, duration='Lambda^-p',
                              angle_per_leg=2*np.pi, ordinary_H_unreversed=True,
                              strength='2pi w(t/tau)/tau, same normal/heavy/light/photon time in both legs'),
        proof_error_powers=dict(initial_scalar_norm=-3/80, smooth_duration=-args.power,
                               metric=-1/4+args.power, moving_centroid=-1/2+2*args.power,
                               fluctuation_preparation=-1/16-args.power, heavy_riesz=args.power-1),
        fundamental_CAR_and_canonical_quantization_assumed=True,
        full_nonlinear_scalar_ground_numerically_computed=False,
        interacting_total_spatial_half_spin_proved=False, native_N_globality_certified=False,
        campaign_complete=False, source_hashes=source_hashes())
    print(json.dumps(portable(result), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

"""One-H finite Gaussian transverse field / full cube / frame-Gauss model.

Pure construction APIs. No old producer, magnetic oscillator or response import.
See rich_frame_photon_20261014_equations.json for physical scope.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, gmres
from scipy.special import erf

from rich_frame_gauss_20261014_model import frame_gauss

Array = NDArray[Any]


@dataclass(frozen=True)
class Parameters:
    grid: int = 19
    wall: float = 2.4
    ell: float = 1.2
    epsilon: float = 0.35
    cube_half: float = 0.75
    cube_shift: tuple[float, float, float] = (0.24, 0.11, -0.17)
    supplier_width: float = 0.26
    ion_width: float = 0.36
    core_width: float = 0.58
    hopping: float = 1.0
    hubbard: float = 2.0
    common_inertia: float = 1.8
    relative_inertia: float = 3.2
    frame_inertia: float = 1.5

    def json(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(None)
def physical_frame() -> dict[str, Any]:
    return frame_gauss()


def mode_values(points: Array, ell: float, smear: float = 0.0) -> Array:
    """u_z,u_y, optionally convolved with normalized Gaussian charge width."""
    width2 = ell * ell + smear * smear
    factor = (ell * ell / width2) ** 2.5
    f = factor * np.exp(-np.sum(points * points, axis=-1) / (2 * width2))
    return np.stack([np.cross(axis, points) * f[..., None]
                     for axis in (np.array([0., 0., 1.]), np.array([0., 1., 0.]))])


def curl_values(points: Array, ell: float) -> Array:
    radius2 = np.sum(points * points, axis=-1)
    f = np.exp(-radius2 / (2 * ell * ell))
    return np.stack([(2 * axis - (radius2[..., None] * axis
                      - points * (points @ axis)[..., None]) / ell ** 2) * f[..., None]
                     for axis in (np.array([0., 0., 1.]), np.array([0., 1., 0.]))])


def coulomb_kernel(positions: Array, widths: Array) -> Array:
    distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    width2 = widths[:, None] ** 2 + widths[None, :] ** 2
    kernel = np.sqrt(2 / np.pi) / np.sqrt(width2)
    mask = distance > 1e-14
    kernel[mask] = erf(distance[mask] / np.sqrt(2 * width2[mask])) / distance[mask]
    return kernel


def cube(p: Parameters) -> dict[str, Any]:
    signs = np.array(list(product((-1., 1.), repeat=3)))
    sites = p.cube_half * signs + np.asarray(p.cube_shift)
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if np.count_nonzero(signs[i] != signs[j]) == 1]
    bits = [(1 << i) | (1 << (8 + j)) for i in range(8) for j in range(8)]
    index = {state: k for k, state in enumerate(bits)}
    density = np.array([[(state >> i & 1) + (state >> (8 + i) & 1)
                         for i in range(8)] for state in bits], dtype=float)
    hops = []
    for i, j in edges:
        rows, cols, data = [], [], []
        for col, state in enumerate(bits):
            for flavour in range(2):
                source, target = i + 8 * flavour, j + 8 * flavour
                if state & (1 << source) and not state & (1 << target):
                    sign = (-1) ** ((state & ((1 << source) - 1)).bit_count())
                    intermediate = state ^ (1 << source)
                    sign *= (-1) ** ((intermediate & ((1 << target) - 1)).bit_count())
                    rows.append(index[intermediate | (1 << target)])
                    cols.append(col)
                    data.append(sign)
        hops.append(sp.csr_matrix((data, (rows, cols)), shape=(64, 64), dtype=complex))
    nodes, weights = np.polynomial.legendre.leggauss(40)
    phase = []
    for i, j in edges:
        delta = sites[j] - sites[i]
        positions = sites[i] + (nodes[:, None] + 1) * delta / 2
        phase.append(np.einsum('aik,k,i->a', mode_values(positions, p.ell, p.supplier_width),
                               delta, weights / 2))
    return dict(sites=sites, edges=edges, bits=bits, density=density,
                hops=hops, phase=np.asarray(phase))


class Joined:
    """Exact N-core and l=0 invariant sector of the specified finite regulator.

    Array ordering: field coordinate pair, physical frame, all64 cube states.
    induction=False removes supplier hopping, not states or Coulomb charge.
    metric=False sets only the common inverse angular inertia to zero.
    """

    def __init__(self, p: Parameters, occupation: int = 1,
                 induction: bool = True, metric: bool = True):
        self.p, self.occupation = p, occupation
        self.induction, self.metric = induction, metric
        self.frame = physical_frame()
        self.ix = np.flatnonzero(abs(np.asarray(self.frame['N']) - occupation) < 1e-8)
        self.ns = len(self.ix)
        self.spin = [np.asarray(a)[np.ix_(self.ix, self.ix)] for a in self.frame['S']]
        self.cube = cube(p)
        n = p.grid
        self.dx = 2 * p.wall / (n + 1)
        axis = np.linspace(-p.wall + self.dx, p.wall - self.dx, n)
        aa, bb = np.meshgrid(axis, axis, indexing='ij')
        self.coordinates = np.stack([aa.ravel(), bb.ravel()])
        self.nfield, self.internal = n * n, self.ns * 64
        self.size = self.nfield * self.internal
        self.I = sp.eye(self.size, format='csr', dtype=complex)
        self.I_field = sp.eye(self.nfield, format='csr')
        self.I_internal = sp.eye(self.internal, format='csr')
        ident = sp.eye(n, format='csr')
        lap = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)],
                       offsets=[-1, 0, 1], format='csr') / self.dx ** 2
        momentum = sp.diags([1j * np.ones(n - 1), -1j * np.ones(n - 1)],
                            offsets=[-1, 1], format='csr') / (2 * self.dx)
        self.mode_norm = np.pi ** 1.5 * p.ell ** 5
        self.mass = p.epsilon * self.mode_norm
        lap2 = sp.kron(lap, ident, format='csr') + sp.kron(ident, lap, format='csr')
        self.E = sp.kron(lap2 / (2 * self.mass), self.I_internal, format='csr')
        self.A = [sp.kron(sp.diags(x), self.I_internal, format='csr') for x in self.coordinates]
        self.P = [sp.kron(m, self.I_internal, format='csr')
                  for m in (sp.kron(momentum, ident), sp.kron(ident, momentum))]
        self.S = [sp.kron(self.I_field, sp.kron(sp.csr_matrix(s), sp.eye(64)), format='csr')
                  for s in self.spin]
        self.C = -2 * occupation * p.core_width ** 2 * (1 + p.core_width ** 2 / p.ell ** 2) ** -2.5
        self.ic = 1 / p.common_inertia if metric else 0.0
        self.ir = 1 / p.relative_inertia
        self.HL_values, self.charge_data = self.longitudinal()
        self.HL = sp.kron(self.I_field, sp.diags(np.tile(self.HL_values, self.ns)), format='csr')
        self.Hbody = self.body()
        self.Hmatter = self.matter()
        self.H = (self.E + self.HL + self.Hbody + self.Hmatter).tocsr()
        self.Jcore = [self.C * self.ic * self.S[a]
                      - self.C ** 2 * (self.ic + self.ir) * self.A[b]
                      for b, a in enumerate((2, 1))]
        self.Jsupplier = [self.supplier_derivative((a,)) * -1 for a in range(2)]
        self.J = [(a + b).tocsr() for a, b in zip(self.Jcore, self.Jsupplier)]
        self.K = [[(self.supplier_derivative((a, b))
                    + (self.C ** 2 * (self.ic + self.ir) * self.I if a == b else 0)).tocsr()
                   for b in range(2)] for a in range(2)]

    def longitudinal(self) -> tuple[Array, dict[str, Any]]:
        p = self.p
        positions = np.vstack([self.cube['sites'], self.cube['sites'], np.zeros((1, 3))])
        widths = np.array([p.supplier_width] * 8 + [p.ion_width] * 8 + [p.core_width])
        charges = np.hstack([self.cube['density'], np.full((64, 8), -0.25),
                             np.full((64, 1), -self.occupation)])
        kernel = coulomb_kernel(positions, widths) / (4 * np.pi * p.epsilon)
        energy = np.einsum('bi,ij,bj->b', charges, kernel, charges) / 2
        return energy, dict(positions=positions, widths=widths, charges=charges, kernel=kernel)

    def body(self, shift: tuple[float, float] = (0., 0.)) -> sp.csr_matrix:
        s2 = sum(s @ s for s in self.spin)
        spin_energy = s2 * (self.ic + 1 / self.p.frame_inertia) / 2
        result = sp.kron(self.I_field, sp.kron(sp.csr_matrix(spin_energy), sp.eye(64)), format='csr')
        for a, component in enumerate((2, 1)):
            x = self.coordinates[a] + shift[a]
            coordinate = sp.kron(sp.diags(x), self.I_internal, format='csr')
            result = result - self.C * self.ic * coordinate @ self.S[component]
            result = result + self.C ** 2 * (self.ic + self.ir) / 2 * sp.kron(
                sp.diags(x * x), self.I_internal, format='csr')
        return result.tocsr()

    def matter(self, shift: tuple[float, float] = (0., 0.)) -> sp.csr_matrix:
        doublon = np.sum(self.cube['density'] == 2, axis=1)
        result = sp.kron(self.I_field, sp.diags(np.tile(self.p.hubbard * doublon, self.ns)), format='csr').astype(complex)
        return (result + self.supplier_derivative((), shift)).tocsr()

    def supplier_derivative(self, axes: tuple[int, ...],
                            shift: tuple[float, float] = (0., 0.)) -> sp.csr_matrix:
        result = sp.csr_matrix((self.size, self.size), dtype=complex)
        if not self.induction:
            return result
        for w, hop in zip(self.cube['phase'], self.cube['hops']):
            phase = w @ (self.coordinates + np.asarray(shift)[:, None])
            coefficient = -self.p.hopping * np.exp(1j * phase)
            for a in axes:
                coefficient *= 1j * w[a]
            forward = sp.kron(sp.diags(coefficient), sp.kron(sp.eye(self.ns), hop), format='csr')
            result = result + forward + forward.getH()
        return result.tocsr()

    def shifted(self, shift: tuple[float, float]) -> sp.csr_matrix:
        return (self.E + self.HL + self.body(shift) + self.matter(shift)).tocsr()

    def source_to(self, target: Joined, source_index: int) -> sp.csr_matrix:
        raw = np.asarray(self.frame['source'][source_index])
        block = raw[np.ix_(target.ix, self.ix)]
        return sp.kron(self.I_field, sp.kron(sp.csr_matrix(block), sp.eye(64)), format='csr')

    def density_fourier(self, wavevector: Array) -> sp.csr_matrix:
        d = self.charge_data
        phase = np.exp(-1j * d['positions'] @ wavevector
                       - d['widths'] ** 2 * np.dot(wavevector, wavevector) / 2)
        rho = d['charges'] @ phase
        return sp.kron(self.I_field, sp.diags(np.tile(rho, self.ns)), format='csr')

    def current_fourier(self, wavevector: Array) -> list[sp.csr_matrix]:
        """Gaussian-smeared bond current plus actual rotating-cloud current."""
        result = [sp.csr_matrix((self.size, self.size), dtype=complex) for _ in range(3)]
        nodes, weights = np.polynomial.legendre.leggauss(40)
        if self.induction:
            for (i, j), w, hop in zip(self.cube['edges'], self.cube['phase'], self.cube['hops']):
                delta = self.cube['sites'][j] - self.cube['sites'][i]
                points = self.cube['sites'][i] + (nodes[:, None] + 1) * delta / 2
                form = np.dot(weights / 2, np.exp(-1j * points @ wavevector))
                form *= np.exp(-self.p.supplier_width ** 2 * np.dot(wavevector, wavevector) / 2)
                coefficient = -self.p.hopping * np.exp(1j * (w @ self.coordinates))
                forward = sp.kron(sp.diags(coefficient), sp.kron(sp.eye(self.ns), hop), format='csr')
                flow = -1j * (forward - forward.getH())
                for a in range(3):
                    result[a] += delta[a] * form * flow
        omega = [self.ic * s for s in self.S]
        for a, component in enumerate((2, 1)):
            omega[component] -= self.C * (self.ic + self.ir) * self.A[a]
        coefficient = 1j * self.occupation * self.p.core_width ** 2
        coefficient *= np.exp(-self.p.core_width ** 2 * np.dot(wavevector, wavevector) / 2)
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            result[a] += coefficient * (omega[b] * wavevector[c] - omega[c] * wavevector[b])
        return [a.tocsr() for a in result]

    def boundary_probability(self, state: Array) -> float:
        pfield = np.sum(abs(state.reshape(self.p.grid, self.p.grid, self.ns, 64)) ** 2, axis=(2, 3))
        mask = np.zeros_like(pfield, dtype=bool)
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
        return float(np.sum(pfield[mask]))

    def supplier_purity(self, state: Array) -> float:
        matrix = state.reshape(-1, 64)
        rho = matrix.conj().T @ matrix
        return float(np.real(np.trace(rho @ rho)))


def spectrum(h: sp.csr_matrix, count: int = 8) -> tuple[Array, Array, dict[str, float]]:
    rng = np.random.default_rng(27041)
    seed = rng.normal(size=h.shape[0]) + 1j * rng.normal(size=h.shape[0])
    _, raw = eigsh(h, k=count, which='SA', v0=seed, tol=2e-11, maxiter=15000)
    orthogonal, _ = np.linalg.qr(raw)
    values, rotation = eigh(orthogonal.conj().T @ (h @ orthogonal))
    vectors = orthogonal @ rotation
    residuals = np.linalg.norm(h @ vectors - vectors * values, axis=0)
    return values, vectors, dict(residual=float(max(residuals)),
                                orthogonality=float(np.linalg.norm(vectors.conj().T @ vectors - np.eye(count))))


def full_resolvent(h: sp.csr_matrix, energy: complex, vectors: Array) -> tuple[Array, list[float]]:
    operator = energy * sp.eye(h.shape[0], format='csr') - h
    diagonal = operator.diagonal()
    preconditioner = sp.diags(1 / diagonal, format='csr')
    answer, residuals = [], []
    for column in vectors.T:
        solution, info = gmres(operator, column, M=preconditioner, rtol=2e-10,
                               atol=1e-12, restart=100, maxiter=1000)
        if info != 0:
            raise RuntimeError(f'full-space resolvent did not converge: {info}')
        answer.append(solution)
        residuals.append(float(np.linalg.norm(operator @ solution - column)))
    return np.column_stack(answer), residuals


def gram(vectors: Array) -> Array:
    return vectors.conj().T @ vectors


def expectation(operator: sp.spmatrix, state: Array) -> float:
    return float(np.real(np.vdot(state, operator @ state)))

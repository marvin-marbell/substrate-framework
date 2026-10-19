#!/usr/bin/env python3
"""Actual finite-bulk field/link Hessian and Gauss-linear principal dynamics.

No optimization, quantum normal-confinement, phase or all-q certificate. The
parent runs this scientific CLI after the source/proof revisions are stable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import expm, null_space

from local_orbital_stability import (
    elastic_hessian, fourier_sea_hessian, local_hamiltonian, periodic_grid,
    scalar_paths,
)
from periodic_bulk_frame import (
    CURVATURE_TOLERANCE, GAP_TOLERANCE, RANK_RCOND, PeriodicFunctional,
    occupation_report, quadrature_report, source_hashes as bulk_source_hashes,
)
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis
from unrestricted_field_minimum import ALIGNMENT, BARE, STIFFNESS, coordinates, matrices

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = 'd66d469ecc0e4e487bb104dc866f7fd679ae0e868c1a5d3abf38c1ede747183b'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
DEFAULT_BRANCH = 'environmental_Clifford_L3'
DEFAULT_INPUT_SHA256 = '4a34efc4053ffbdf42f74ff492960d55109a5a359f2c367a24b14af740872a62'


def source_hashes():
    names = ('full_spatial_gauge_stability.py', 'full_spatial_gauge_stability.md',
             'cartesian_angular_quotient.py', 'cartesian_angular_quotient.md',
             'full_haar_field_packet.py', 'full_haar_field_packet.md',
             'full_band_packet_bound.md')
    return {**bulk_source_hashes(), **{
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}


def check_error(error, label, tolerance=3e-7):
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error} > {tolerance}')


def hermitian(matrix, label, *, relative=False):
    error = float(np.linalg.norm(matrix - matrix.conj().T))
    if relative:
        error /= max(1., float(np.linalg.norm(matrix)))
    check_error(error, label + ' Hermiticity', 1e-12 if relative else 3e-7)
    return (matrix + matrix.conj().T) / 2


def serializable(value) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return dict(real=value.real.tolist(), imaginary=value.imag.tolist())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


class SpatialFunctional:
    """Uniform mean, all 60 fields and three genuine 15-coordinate links.

    The input law is unimproved. Every executed scalar path must be a positive
    nearest-axis path; a changed longer-path law is rejected, not silently given
    a single link. h_xy=(F_x U_xy+U_xy F_y)/2 is the same endpoint law as the
    inherited identity-link field consumer, now with its actual colour link.
    """
    def __init__(self, band, size, basis, value):
        self.uniform = PeriodicFunctional(band, size, basis)
        self.band, self.size, self.basis = band, size, basis
        self.generators = basis / np.sqrt(8)
        self.value = np.asarray(value, dtype=float).reshape(60)
        self.frame = self.uniform.fields(self.value)
        self.sites = periodic_grid(size)
        self.k = band.a * self.uniform.momenta
        self.x, self.count = self.uniform.x, self.uniform.count
        self.h = np.einsum('pm,mij->pij', self.x, self.frame)
        self.values, self.vectors, self.occupied, self.gap = self.uniform.spectral(self.value)
        if self.gap <= GAP_TOLERANCE:
            raise ValueError(f'No separated TOTAL occupied/empty cluster: {self.gap}')
        self.projector = ((self.vectors * self.occupied[:, None, :])
                          @ self.vectors.conj().transpose(0, 2, 1))
        self.onsite, raw_paths = scalar_paths(band)
        self.paths = []
        for displacement, scalar in raw_paths:
            displacement = np.asarray(displacement, dtype=int)
            axes = np.flatnonzero(displacement)
            if len(axes) != 1 or displacement[axes[0]] != 1:
                raise ValueError('Supported receipt law needs actual nearest positive-axis paths')
            axis = int(axes[0])
            F = np.einsum('m,mij->ij', scalar, self.frame)
            self.paths.append((axis, displacement, np.asarray(scalar), F))
        if band.improved or {row[0] for row in self.paths} != {0, 1, 2}:
            raise ValueError('This full-link supplier supports the actual unimproved three-axis input law')
        self.D = np.array([coordinates(1j * (T @ self.frame - self.frame @ T), basis)
                           .reshape(-1) for T in self.generators]).T
        self.elastic = elastic_hessian(self.frame, basis, STIFFNESS, 0.)
        energy, field_gradient = self.uniform.evaluate(self.value)
        link_gradient = np.zeros(45)
        for start in range(0, self.count, 128):
            indices = np.arange(start, min(start+128, self.count))
            V0 = self.vertices(indices, np.zeros(3), indices)
            link_gradient -= np.einsum('pij,paji->a', self.projector[indices],
                                       V0[:, 60:]).real / self.count
        self.gradient = np.concatenate((field_gradient, link_gradient))
        self.energy = energy

    def shifted_indices(self, q):
        shift = q * self.size / (2 * np.pi)
        rounded = np.rint(shift).astype(int)
        check_error(float(np.linalg.norm(shift-rounded)), 'q belongs to the Bloch quadrature', 1e-10)
        return np.ravel_multi_index(((self.sites + rounded) % self.size).T, (self.size,) * 3)

    def vertices(self, indices, q, targets):
        """<k+q|d h/dz_i(q)|k>, with the source-owned Fourier sign."""
        k = self.k[indices]
        t = (self.x[indices] + self.x[targets]) / 2
        result = np.zeros((len(indices), 105, 4, 4), dtype=complex)
        result[:, :60] = (t[:, :, None, None, None]
                           * self.basis[None, None]).reshape(-1, 60, 4, 4)
        for axis, displacement, _, F in self.paths:
            anti = F @ self.generators + self.generators @ F
            anti_dagger = F.conj().T @ self.generators + self.generators @ F.conj().T
            forward = np.exp(1j * (k @ displacement))
            backward = np.exp(-1j * ((k+q) @ displacement))
            result[:, 60+15*axis:75+15*axis] += .5j * (
                forward[:, None, None, None] * anti
                - backward[:, None, None, None] * anti_dagger)
        return result

    def contact(self, q):
        """-Tr P d^2 h: mixed field/link AND link/link diamagnetism."""
        result = np.zeros((105, 105), dtype=complex)
        BT = self.basis[:, None] @ self.generators[None]
        TB = self.generators[None] @ self.basis[:, None]
        TT = (self.generators[:, None] @ self.generators[None]
              + self.generators[None] @ self.generators[:, None])
        for axis, displacement, scalar, F in self.paths:
            phase = np.exp(1j * (self.k @ displacement))
            rho_plus = np.einsum('p,pij->ij', phase, self.projector) / self.count
            rho_minus = np.einsum('p,pij->ij', phase.conj(), self.projector) / self.count
            z = np.exp(-1j * (q @ displacement))
            forward = .5j * scalar[:, None, None, None, None] * (BT + z*TB)[None]
            backward = -.5j * scalar.conj()[:, None, None, None, None] * (TB + z*BT)[None]
            mixed = -(np.einsum('ij,mabji->mab', rho_plus, forward)
                      + np.einsum('ij,mabji->mab', rho_minus, backward)).reshape(60, 15)
            link = slice(60+15*axis, 75+15*axis)
            result[:60, link] += mixed
            result[link, :60] += mixed.conj().T
            second_forward = -.25 * (F @ TT + TT @ F)
            second_backward = second_forward.conj().swapaxes(-1, -2)
            result[link, link] -= (np.einsum('ij,abji->ab', rho_plus, second_forward)
                                  + np.einsum('ij,abji->ab', rho_minus, second_backward))
        return hermitian(result, 'whole-heavy contact')

    def alignment(self, q):
        result = np.zeros((105, 105), dtype=complex)
        for axis in range(3):
            L = np.zeros((60, 105), dtype=complex)
            L[:, :60] = (1-np.exp(1j*q[axis])) * np.eye(60)
            L[:, 60+15*axis:75+15*axis] = -self.D
            result += ALIGNMENT * L.conj().T @ L
        return result

    def hessian_parts(self, q):
        targets = self.shifted_indices(q)
        sea = np.zeros((105, 105), dtype=complex)
        for start in range(0, self.count, 128):
            indices = np.arange(start, min(start+128, self.count))
            target = targets[indices]
            V = self.vertices(indices, q, target)
            feature = np.einsum('pti,paij,pjs->pats',
                                self.vectors[target].conj().transpose(0, 2, 1),
                                V, self.vectors[indices])
            target_e, source_e = self.values[target], self.values[indices]
            target_n, source_n = self.occupied[target], self.occupied[indices]
            mask = target_n[:, :, None] != source_n[:, None, :]
            denominator = target_e[:, :, None] - source_e[:, None, :]
            numerator = target_n[:, :, None].astype(float) - source_n[:, None, :]
            weight = np.zeros_like(denominator)
            np.divide(numerator, denominator, out=weight, where=mask)
            if np.any(weight[mask] <= 0) or np.any(abs(denominator[mask]) < self.gap):
                raise AssertionError('Cross-cluster denominators must retain the actual global gap')
            weighted = (feature * np.sqrt(weight)[:, None]).transpose(1, 0, 2, 3).reshape(105, -1)
            sea -= weighted.conj() @ weighted.T / self.count
        local = np.zeros((105, 105), dtype=complex)
        local[:60, :60] = self.elastic
        alignment = self.alignment(q)
        sea = hermitian(sea, 'whole-heavy cluster response')
        contact = self.contact(q)
        H = hermitian(local + alignment + sea + contact, 'full105 potential Hessian')
        return dict(H=H, local=local, alignment=alignment, sea=sea, contact=contact)

    def gauge_columns(self, q):
        return np.vstack((self.D, *[(1-np.exp(1j*q[axis])) * np.eye(15) for axis in range(3)]))

    def ward_rhs(self, q):
        """-(d tau)^T gradient, including the exponential-link chart Jacobian."""
        gf = matrices(self.gradient[:60].reshape(4, 15), self.basis)
        fields = np.array([coordinates(1j*(T @ gf-gf @ T), self.basis).reshape(-1)
                           for T in self.generators]).T
        links = []
        for axis in range(3):
            gl = np.einsum('a,aij->ij', self.gradient[60+15*axis:75+15*axis], self.generators)
            action = np.array([2*np.einsum('aij,ji->a', self.generators,
                                           1j*(T @ gl-gl @ T)).real for T in self.generators]).T
            links.append((1+np.exp(1j*q[axis])) * action / 2)
        return np.vstack((fields, *links))

    def ward(self, q, parts, *, adverse=False):
        G, rhs = self.gauge_columns(q), self.ward_rhs(q)
        actual = parts['H'] @ G
        error = float(np.linalg.norm(actual-rhs))
        check_error(error, 'differentiated local gauge Ward')
        targets = self.shifted_indices(q)
        indices = np.arange(min(13, self.count))
        contracted = np.einsum('paij,at->ptij', self.vertices(indices, q, targets[indices]), G)
        expected = 1j*(self.generators[None] @ self.h[indices, None]
                       - self.h[targets[indices], None] @ self.generators[None])
        vertex_error = float(np.linalg.norm(contracted-expected))
        check_error(vertex_error, 'local gauge onebody vertex')
        result = dict(gauge_columns=G, differentiated_rhs=rhs,
                      differentiated_Ward_residual=error, gauge_Hessian_action_norm=float(np.linalg.norm(actual)),
                      off_stationary_rhs_norm=float(np.linalg.norm(rhs)), onebody_gauge_vertex_residual=vertex_error,
                      identity='H(q) G(q)=-(d tau)^T grad; H G is not forced to zero')
        if adverse:
            no_contact = float(np.linalg.norm((parts['H']-parts['contact']) @ G-rhs))
            wrong_sea = float(np.linalg.norm((parts['local']+parts['alignment']
                                             -parts['sea']-parts['contact']) @ G-rhs))
            S = np.concatenate((np.ones(60), -np.ones(45)))
            wrong_link = float(np.linalg.norm((parts['local']+parts['alignment']+parts['contact']
                                              +S[:, None]*parts['sea']*S[None]) @ G-rhs))
            if min(no_contact, wrong_sea, wrong_link) <= 1e-5:
                raise AssertionError('Local Ward adverse controls did not reject missing contact/sign')
            result.update(missing_diamagnetic_contact_Ward_residual=no_contact,
                          wrong_heavy_sea_sign_Ward_residual=wrong_sea,
                          wrong_linear_link_vertex_sign_Ward_residual=wrong_link,
                          adverse_controls_reach_same_differentiated_local_Ward=True)
        return result

    def fixed_link_controls(self, q, parts):
        integer_q = np.rint(q*self.size/(2*np.pi)).astype(int)
        result: dict[str, Any] = {}
        if np.all(self.occupied[:, 2:]) and not np.any(self.occupied[:, :2]):
            data = dict(sites=self.sites, x=self.x, values=self.values, vectors=self.vectors)
            inherited = fourier_sea_hessian(data, integer_q, self.size, self.basis)
            expected = self.elastic + inherited + ALIGNMENT*float(np.sum(abs(1-np.exp(1j*q))**2))*np.eye(60)
            error = float(np.linalg.norm(parts['H'][:60, :60]-expected))
            check_error(error, 'independent inherited field60 Fourier control')
            result.update(inherited_field60_Fourier_error=error,
                          inherited_scope='fixed links; verified two globally occupied modes per momentum')
        else:
            result['inherited_scope'] = 'two-per-block helper inapplicable; exact global-cluster full105 still used'
        if np.linalg.norm(q) == 0:
            error = float(np.linalg.norm(parts['H'][:60, :60]-self.uniform.hessian(self.value)))
            check_error(error, 'independent periodic uniform60 Hessian')
            result['independent_q0_uniform60_error'] = error
        return result


def principal_modes(H, G, *, Lambda, ue4):
    """Gauss G^*p=0, M-horizontal velocities, restoring potential Lambda*V."""
    mass = np.concatenate((np.full(60, 1/np.sqrt(Lambda)), np.full(45, 1/(2*ue4))))
    _, singular, right = np.linalg.svd(G, full_matrices=False)
    tolerance = RANK_RCOND * max(1., float(singular[0]))
    rank = int(np.sum(singular > tolerance))
    # Resolve the actual coordinate stabilizer BEFORE mass weighting. The mass
    # scale is not permitted to change the input's stabilizer classification.
    supported_gauge = G @ right[:rank].conj().T
    weighted_gauge = np.sqrt(mass)[:, None] * supported_gauge
    Z = null_space(weighted_gauge.conj().T, rcond=1e-12)
    if Z.shape[1] != 105-rank:
        raise ValueError('Mass-weighted gauge complement lost a resolved gauge direction')
    inverse_root = 1/np.sqrt(mass)
    K = Lambda * (inverse_root[:, None]*H*inverse_root[None])
    reduced = hermitian(Z.conj().T @ K @ Z, 'Gauss-linear principal restoring matrix', relative=True)
    values, vectors = np.linalg.eigh(reduced)
    kinetic_vectors = Z @ vectors
    coordinate_vectors = inverse_root[:, None] * kinetic_vectors
    coordinate_norm = np.sum(abs(coordinate_vectors)**2, axis=0)
    potential_curvature = values / (Lambda*coordinate_norm)
    gauss_error = float(np.linalg.norm(G.conj().T @ (mass[:, None]*coordinate_vectors)))
    check_error(gauss_error, 'actual Gauss-linear mass horizontality', 2e-6)
    projected_error = float(np.linalg.norm(reduced @ vectors-vectors*values[None]) / max(1., np.linalg.norm(reduced)))
    check_error(projected_error, 'principal normal eigenproblem', 2e-10)
    return dict(Cartesian_field_mass=mass[0], colour_link_mass=mass[60],
                potential_multiplier=Lambda, gauge_rank=rank, stabilizer_dimension=15-rank,
                gauge_singular_values=singular, gauge_rank_tolerance=tolerance,
                normal_dimension=len(values), squared_frequencies=values,
                oscillation_frequencies=np.sqrt(np.maximum(values, 0.)),
                instability_growth_rates=np.sqrt(np.maximum(-values, 0.)),
                potential_curvature_per_coordinate_norm=potential_curvature,
                negative_squared_frequency_count_raw=int(np.sum(values < 0)),
                resolved_negative_normal_count=int(np.sum(potential_curvature < -CURVATURE_TOLERANCE)),
                resolved_flat_normal_count=int(np.sum(abs(potential_curvature) <= CURVATURE_TOLERANCE)),
                field_kinetic_participation=np.sum(abs(kinetic_vectors[:60])**2, axis=0),
                Gauss_mass_horizontal_residual=gauss_error, projected_eigenproblem_residual=projected_error,
                scope='classical quadratic / adiabatic principal spectrum at this frozen mean; not quantum eigenstates')


def real_space_hamiltonian(fields, links, model, size, *, left=None, right=None):
    """Independent actual endpoint implementation and identity-link derivatives.

    left/right hold real sitewise Cartesian/link tangent arrays. Derivatives are
    assembled in real space, not by inverse-transforming the tested vertices.
    """
    sites = periodic_grid(size)
    index = {tuple(site): number for number, site in enumerate(sites)}
    count = len(sites)
    result = np.zeros((4*count, 4*count), dtype=complex)
    for x, site in enumerate(sites):
        sx = slice(4*x, 4*x+4)
        if left is None:
            result[sx, sx] += model.onsite*fields[x, 0]
        elif right is None:
            result[sx, sx] += model.onsite*left[0][x, 0]
        for axis, displacement, scalar, _ in model.paths:
            y = index[tuple((site+displacement) % size)]
            sy = slice(4*y, 4*y+4)
            Fx, Fy = (np.einsum('m,mij->ij', scalar, fields[z]) for z in (x, y))
            if left is None:
                U = expm(1j*links[x, axis])
                coupling = (Fx @ U+U @ Fy)/2
            else:
                if np.linalg.norm(links) != 0:
                    raise ValueError('Independent analytic probe derivatives use the identity-link chart')
                Lx, Ly = (np.einsum('m,mij->ij', scalar, left[0][z]) for z in (x, y))
                A = left[1][x, axis]
                if right is None:
                    coupling = (Lx+Ly+1j*(Fx @ A+A @ Fy))/2
                else:
                    Rx, Ry = (np.einsum('m,mij->ij', scalar, right[0][z]) for z in (x, y))
                    B = right[1][x, axis]
                    second_U = -(A @ B+B @ A)/2
                    coupling = (Fx @ second_U+second_U @ Fy
                                +1j*(Lx @ B+B @ Ly+Rx @ A+A @ Ry))/2
            result[sx, sy] += coupling
            result[sy, sx] += coupling.conj().T
    return result


def real_space_energy(fields, links, model, size):
    h = real_space_hamiltonian(fields, links, model, size)
    values = np.linalg.eigvalsh(h)
    count = len(fields)
    gap = float(values[2*count]-values[2*count-1])
    if gap <= GAP_TOLERANCE:
        raise ValueError('Real-space variation lost its exact global half-filled cluster')
    sea = -float(values[2*count:].sum())
    gram = np.einsum('xmij,xnji->xmn', fields, fields).real/4
    elastic = STIFFNESS/2*float(np.sum(gram**2))
    sites = periodic_grid(size)
    index = {tuple(site): number for number, site in enumerate(sites)}
    alignment = 0.
    for x, site in enumerate(sites):
        for axis in range(3):
            y = index[tuple((site+np.eye(3, dtype=int)[axis]) % size)]
            U = expm(1j*links[x, axis])
            difference = fields[x]-U @ fields[y] @ U.conj().T
            alignment += ALIGNMENT/8*float(np.sum(abs(difference)**2))
    return (elastic+alignment+sea)/count


def real_space_probes(band, basis, value):
    """Small periodic consumer, nontrivial oblique wave and actual SU4 generators."""
    size = 3  # Its midpoint grid has periodic, not antiperiodic, seam.
    off = np.asarray(value)+.017*np.sin(np.arange(60)*.43+.31)
    model = SpatialFunctional(band, size, basis, off)
    sites, count = model.sites, model.count
    fields = np.broadcast_to(model.frame, (count, 4, 4, 4)).copy()
    links = np.zeros((count, 3, 4, 4), dtype=complex)
    F = np.exp(1j*(sites @ model.k.T))/np.sqrt(count)
    transform = np.kron(F, np.eye(4))
    h0 = real_space_hamiltonian(fields, links, model, size)
    inherited = local_hamiltonian(fields, band, sites, size)
    inherited_error = float(np.linalg.norm(h0-inherited))
    check_error(inherited_error, 'actual identity-link finite real-space h')
    diagonal = np.zeros_like(h0)
    for k in range(count):
        diagonal[4*k:4*k+4, 4*k:4*k+4] = model.h[k]
    h_error = float(np.linalg.norm(transform.conj().T @ h0 @ transform-diagonal))
    check_error(h_error, 'actual finite real-space h / Fourier law')
    q = 2*np.pi*np.array([1, 1, 0])/size
    targets = model.shifted_indices(q)
    theta = .37
    profile = np.cos(sites @ q+theta)
    u = np.sin(np.arange(105)*.37+.21); u /= np.linalg.norm(u)
    v = np.cos(np.arange(105)*.53+.41); v /= np.linalg.norm(v)
    uniform = np.zeros(105); uniform[:60] = off/np.linalg.norm(off)*.31
    def tangent(direction, constant=None):
        z = profile[:, None]*direction[None]
        if constant is not None:
            z = z+constant[None]
        return (matrices(z[:, :60].reshape(count, 4, 15), basis),
                np.einsum('xda,aij->xdij', z[:, 60:].reshape(count, 3, 15), model.generators))
    left, right = tangent(u, uniform), tangent(v)
    dh = real_space_hamiltonian(fields, links, model, size, left=left)
    d2h = real_space_hamiltonian(fields, links, model, size, left=left, right=right)
    transformed = transform.conj().T @ dh @ transform
    expected = np.zeros_like(h0)
    V = model.vertices(np.arange(count), q, targets)
    V0 = model.vertices(np.arange(count), np.zeros(3), np.arange(count))
    for k, target in enumerate(targets):
        block = .5*np.exp(1j*theta)*np.einsum('a,aij->ij', u, V[k])
        expected[4*target:4*target+4, 4*k:4*k+4] += block
        expected[4*k:4*k+4, 4*target:4*target+4] += block.conj().T
        expected[4*k:4*k+4, 4*k:4*k+4] += np.einsum('a,aij->ij', uniform, V0[k])
    vertex_error = float(np.linalg.norm(transformed-expected))
    check_error(vertex_error, 'all105 nontrivial real-wave linear vertex')
    values, vectors = np.linalg.eigh(h0)
    P = vectors[:, 2*count:] @ vectors[:, 2*count:].conj().T
    sea_first = -float(np.trace(P @ dh).real)/count
    elastic_first = float(model.uniform.elastic_gradient(off)[1] @ uniform[:60])
    first = sea_first+elastic_first
    Hq = model.hessian_parts(q)
    H0 = model.hessian_parts(np.zeros(3))
    second = .5*float(np.vdot(u, Hq['H'] @ v).real)
    first_from_uniform_gradient = float(model.gradient @ uniform)
    check_error(abs(first-first_from_uniform_gradient), 'finite real-space nonzero first energy variation')
    eo_left = vectors[:, :2*count].conj().T @ dh @ vectors[:, 2*count:]
    right_h = real_space_hamiltonian(fields, links, model, size, left=right)
    eo_right = vectors[:, :2*count].conj().T @ right_h @ vectors[:, 2*count:]
    denominator = values[None, 2*count:]-values[:2*count, None]
    sea_second = (-2*float(np.sum(eo_left.conj()*eo_right/denominator).real)
                  -float(np.trace(P @ d2h).real))/count
    expected_sea = .5*float(np.vdot(u, (Hq['sea']+Hq['contact']) @ v).real)
    sea_second_error = abs(sea_second-expected_sea)
    check_error(sea_second_error, 'finite real-space spectral response INCLUDING contact')
    rows = []
    for eps in (2e-4, 1e-4):
        plus_h = real_space_hamiltonian(fields+eps*left[0], links+eps*left[1], model, size)
        minus_h = real_space_hamiltonian(fields-eps*left[0], links-eps*left[1], model, size)
        first_h_error = float(np.linalg.norm((plus_h-minus_h)/(2*eps)-dh))
        mixed_h = sum(sign_l*sign_r*real_space_hamiltonian(
            fields+eps*(sign_l*left[0]+sign_r*right[0]),
            links+eps*(sign_l*left[1]+sign_r*right[1]), model, size)
            for sign_l in (-1, 1) for sign_r in (-1, 1))/(4*eps**2)
        second_h_error = float(np.linalg.norm(mixed_h-d2h))
        plus_e = real_space_energy(fields+eps*left[0], links+eps*left[1], model, size)
        minus_e = real_space_energy(fields-eps*left[0], links-eps*left[1], model, size)
        first_e_error = abs((plus_e-minus_e)/(2*eps)-first)
        mixed_e = sum(sign_l*sign_r*real_space_energy(
            fields+eps*(sign_l*left[0]+sign_r*right[0]),
            links+eps*(sign_l*left[1]+sign_r*right[1]), model, size)
            for sign_l in (-1, 1) for sign_r in (-1, 1))/(4*eps**2)
        second_e_error = abs(mixed_e-second)
        check_error(first_h_error, 'actual h first finite variation', 2e-7)
        check_error(second_h_error, 'actual h mixed/link contact finite variation', 2e-6)
        check_error(first_e_error, 'actual energy first finite variation', 3e-7)
        check_error(second_e_error, 'actual energy second finite variation', 3e-6)
        rows.append(dict(step=eps, h_first_error=first_h_error, h_second_error=second_h_error,
                         energy_first_error=first_e_error, energy_second_error=second_e_error))
    if abs(first) <= 1e-5 or np.linalg.norm(d2h) <= 1e-4:
        raise AssertionError('Real-space variation probe lacks a nonzero first derivative/contact')
    # Gauge perturbations use nonuniform site rotations and actual, unrelated
    # SU4 generators. Neither h covariance nor finite energy is forced by G.
    gauge_rows = []
    for generator in (model.generators[0], model.generators[7],
                      np.einsum('a,aij->ij', .19*np.sin(np.arange(15)*.47+.21), model.generators)):
        Usite = np.array([expm(1j*.23*p*generator) for p in profile])
        rotated_fields = Usite[:, None] @ fields @ Usite[:, None].conj().transpose(0, 1, 3, 2)
        rotated_h = np.zeros_like(h0)
        index = {tuple(site): number for number, site in enumerate(sites)}
        for x, site in enumerate(sites):
            sx = slice(4*x, 4*x+4)
            rotated_h[sx, sx] += model.onsite*rotated_fields[x, 0]
            for _, displacement, scalar, _ in model.paths:
                y = index[tuple((site+displacement) % size)]; sy = slice(4*y, 4*y+4)
                U = Usite[x] @ Usite[y].conj().T
                Fx, Fy = (np.einsum('m,mij->ij', scalar, rotated_fields[z]) for z in (x, y))
                coupling = (Fx @ U+U @ Fy)/2
                rotated_h[sx, sy] += coupling; rotated_h[sy, sx] += coupling.conj().T
        gauge_matrix = np.zeros_like(h0)
        for x in range(count):
            gauge_matrix[4*x:4*x+4, 4*x:4*x+4] = Usite[x]
        covariance = float(np.linalg.norm(rotated_h-gauge_matrix @ h0 @ gauge_matrix.conj().T))
        check_error(covariance, 'finite nonuniform SU4 real-space h covariance')
        generator_coeff = 2*np.einsum('aij,ji->a', model.generators, generator).real
        gauge_direction = model.gauge_columns(q) @ generator_coeff
        onebody_tangent = np.einsum('a,paij->pij', gauge_direction, V)
        exact_tangent = 1j*(generator @ model.h-model.h[targets] @ generator)
        differential = float(np.linalg.norm(onebody_tangent-exact_tangent))
        check_error(differential, 'actual nonuniform SU4 generator differential')
        gauge_rows.append(dict(generator_coordinates=generator_coeff, finite_h_covariance_error=covariance,
                               infinitesimal_wave_vertex_error=differential))
    return dict(real_space_grid=size, off_stationary_coordinate_values=off.reshape(4, 15),
                q_angular=q, wave_phase=theta, actual_first_energy_variation=first,
                actual_second_energy_variation=second, real_space_second_h_norm=float(np.linalg.norm(d2h)),
                identity_link_inherited_h_error=inherited_error, finite_h_Fourier_error=h_error,
                all105_wave_vertex_error=vertex_error, spectral_second_with_contact_error=sea_second_error,
                central_differences=rows, actual_SU4_generator_probes=gauge_rows,
                off_stationary_local_Ward=model.ward(q, Hq, adverse=True),
                q0_control=model.fixed_link_controls(np.zeros(3), H0),
                scopes='actual real-space endpoint h, exact TOTAL occupation; generic off-stationary mean and oblique wave')


def singular_sector(model, H):
    _, singular, right = np.linalg.svd(model.D, full_matrices=False)
    tolerance = RANK_RCOND*max(1., float(singular[0]))
    rank = int(np.sum(singular > tolerance))
    kernel = right[rank:].conj().T
    result: dict[str, Any] = dict(q0_field_orbit_dimension=rank,
        numerical_stabilizer_Lie_dimension=15-rank, stabilizer_generator_columns=kernel,
        stabilizer_commutator_residual=float(np.linalg.norm(model.D @ kernel)),
        commutator_resolution=tolerance, exact_stabilizer_not_certified=True)
    if kernel.shape[1]:
        C = np.zeros((105, 3*kernel.shape[1]), dtype=complex)
        for axis in range(3):
            C[60+15*axis:75+15*axis, axis*kernel.shape[1]:(axis+1)*kernel.shape[1]] = kernel
        curvature = hermitian(C.conj().T @ H @ C, 'q0 unbroken holonomy curvature')
        result.update(q0_unbroken_link_potential_matrix=curvature,
            q0_unbroken_link_curvatures=np.linalg.eigvalsh(curvature),
            unbroken_alignment_mass_matrix=C.conj().T @ model.alignment(np.zeros(3)) @ C,
            interpretation='resolved U1 or larger singular stratum: q0 stabilizer is not a missing normal mode; '
                           'unbroken constant links have zero alignment mass, while finite-mesh sea holonomy curvature '
                           'is retained. Exactly commuting continuum Bloch integrals are shift-invariant, not a finite-grid zero theorem')
    return result


def select_inputs(receipt, selected_name, candidate, identities, all_branches):
    expected = {**BARE, 'stiffness': STIFFNESS, 'g0': 0., 'Jpref': ALIGNMENT}
    if receipt.get('bare_coefficients') != expected:
        raise ValueError('Input receipt must identify the actual same unimproved zero-preference bulk law')
    for owner in ('sea_selection.py', 'local_orbital_stability.py', 'rotation_band.py',
                  'unrestricted_field_minimum.py', 'periodic_bulk_frame.py'):
        if receipt.get('source_sha256', {}).get(owner) != identities[owner]:
            raise ValueError(f'Input receipt has a different executed coordinate/path owner: {owner}')
    branches = receipt.get('same_law_seed_branches', [])
    matches = [row for row in branches if row.get('name') == selected_name]
    if len(matches) != 1 or candidate not in matches[0]:
        raise ValueError(f'Exact branch/candidate unavailable: {selected_name}/{candidate}')
    chosen = [matches[0]]+[row for row in branches if row is not matches[0]] if all_branches else matches
    result = []
    for branch in chosen:
        if candidate not in branch:
            raise ValueError(f'Requested same-law branch has no actual {candidate}: {branch.get("name")}')
        item = branch[candidate]
        value = np.asarray(item['coordinate_values'], dtype=float)
        if value.shape != (4, 15) or not np.all(np.isfinite(value)):
            raise ValueError('Actual receipt needs all60 finite Cartesian coordinates')
        branch_index = branches.index(branch)
        result.append(dict(name=branch['name'], value=value.reshape(-1), input_grid=int(item['grid']),
            pointer=f'/same_law_seed_branches/{branch_index}/{candidate}/coordinate_values',
            candidate=candidate, selected=branch is matches[0]))
    return result


def q_samples(input_grid, coarse, fine, requested):
    common = math.gcd(math.gcd(input_grid, coarse), fine)
    if requested:
        indices = []
        for text in requested:
            parts = text.split(',')
            if len(parts) != 3:
                raise ValueError('--q-index requires three comma-separated integers on the input grid')
            indices.append(np.array([int(x) for x in parts], dtype=int) % input_grid)
    else:
        if common < 4 or common % 2:
            raise ValueError('Default zero/soft/edge/corner probes require an even common grid >=4')
        step, edge = input_grid//common, input_grid//2
        indices = [np.array(row, dtype=int) for row in
                   ((0, 0, 0), (step, 0, 0), (step, step, 0), (step, 2*step, step),
                    (edge, 0, 0), (edge, edge, 0), (edge, edge, edge))]
    # Keep q0 first even when a user listed it after another requested wave.
    indices = [np.zeros(3, dtype=int)]+[row for row in indices if np.any(row != 0)]
    unique = []
    for row in indices:
        if not any(np.array_equal(row, prior) for prior in unique):
            unique.append(row)
    for row in unique:
        for size in (coarse, fine):
            if np.any((row*size) % input_grid):
                raise ValueError('q must belong to BOTH Bloch quadratures and the actual input grid')
    return unique


def run(*, input_receipt=ROOT/'periodic_bulk_frame_receipt.json', input_sha256=DEFAULT_INPUT_SHA256,
        branch=DEFAULT_BRANCH, candidate='validation_stationary_candidate', grid=8, validation_grid=16,
        Lambda=1e6, ue4=.3, all_branches=True, q_indices=None):
    if grid < 4 or validation_grid <= grid:
        raise ValueError('Need coarse grid>=4 and a strictly finer Bloch quadrature')
    if not all(np.isfinite(x) and x > 0 for x in (Lambda, ue4)):
        raise ValueError('Positive finite Lambda and ue4 required')
    input_receipt = Path(input_receipt).resolve()
    identities = source_hashes()
    payload = input_receipt.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != input_sha256:
        raise ValueError('Actual input bytes do not match --input-sha256')
    receipt = json.loads(payload)
    inputs = select_inputs(receipt, branch, candidate, identities, all_branches)
    basis, band = matrix_unit_basis(), RotationBand(**BARE)
    check_error(float(np.linalg.norm(np.einsum('aij,bji->ab', basis, basis)/4-np.eye(15))), 'actual Cartesian basis')
    selected = inputs[0]
    probes = real_space_probes(band, basis, selected['value'])
    results = []
    for supplied in inputs:
        samples = q_samples(supplied['input_grid'], grid, validation_grid, q_indices)
        mesh_results = []
        for size in (grid, validation_grid):
            model = SpatialFunctional(band, size, basis, supplied['value'])
            rows = []
            for index in samples:
                q = 2*np.pi*index/supplied['input_grid']
                parts = model.hessian_parts(q)
                spectrum = np.linalg.eigvalsh(parts['H'])
                G = model.gauge_columns(q)
                principal = principal_modes(parts['H'], G, Lambda=Lambda, ue4=ue4)
                scaling = dict(
                    fixed_ue4_vary_Lambda=[dict(Lambda=Lambda*factor,
                        principal=principal_modes(parts['H'], G, Lambda=Lambda*factor, ue4=ue4))
                        for factor in (.25, 4.)],
                    fixed_Lambda_vary_ue4=[dict(ue4=ue4*factor,
                        principal=principal_modes(parts['H'], G, Lambda=Lambda, ue4=ue4*factor))
                        for factor in (.5, 2.)])
                rows.append(dict(q_input_grid_index=index, q_angular=q, q_physical=q/band.a,
                    full105_potential_Hessian=parts['H'], potential_eigenvalues=spectrum,
                    negative_potential_count=int(np.sum(spectrum < -CURVATURE_TOLERANCE)),
                    flat_potential_count=int(np.sum(abs(spectrum) <= CURVATURE_TOLERANCE)),
                    local_gauge_Ward=model.ward(q, parts),
                    independent_field60_controls=model.fixed_link_controls(q, parts),
                    diamagnetic_contact_norm=float(np.linalg.norm(parts['contact'])),
                    principal=principal, scaling=scaling,
                    stationarity_required_for_equilibrium_interpretation=True))
            mesh_results.append(dict(grid=size, energy_per_cell=model.energy,
                full105_gradient=model.gradient, gradient_norm=float(np.linalg.norm(model.gradient)),
                field_gradient_norm=float(np.linalg.norm(model.gradient[:60])),
                link_gradient_norm=float(np.linalg.norm(model.gradient[60:])),
                scaled_Lambda_gradient_norm=float(Lambda*np.linalg.norm(model.gradient)),
                actual_stationary_at_2e_minus8=bool(np.linalg.norm(model.gradient) <= 2e-8),
                exact_global_occupation=occupation_report(model.uniform, model.value),
                finite_quadrature_cover=quadrature_report(model.uniform, model.value),
                q0_stabilizer_and_unbroken_links=singular_sector(model, rows[0]['full105_potential_Hessian']),
                q_rows=rows))
        coarse, fine = mesh_results
        convergence = [dict(q_input_grid_index=left['q_input_grid_index'],
            potential_Hessian_difference_norm=float(np.linalg.norm(right['full105_potential_Hessian']-left['full105_potential_Hessian'])),
            minimum_potential_coarse=float(left['potential_eigenvalues'][0]),
            minimum_potential_fine=float(right['potential_eigenvalues'][0]),
            principal_squared_frequencies_difference_norm=float(np.linalg.norm(
                right['principal']['squared_frequencies']-left['principal']['squared_frequencies'])))
            for left, right in zip(coarse['q_rows'], fine['q_rows'])]
        results.append(dict(name=supplied['name'], selected=supplied['selected'],
            coordinate_pointer=supplied['pointer'], coordinate_values=supplied['value'].reshape(4, 15),
            input_stationary_grid=supplied['input_grid'], meshes=mesh_results,
            coarse_finer_comparison=convergence, frozen_coordinates_not_reoptimized=True))
    if source_hashes() != identities or input_receipt.read_bytes() != payload:
        raise RuntimeError('Source/proof/input bytes changed during execution; refusing mixed-bound output')
    return serializable(dict(schema=1, source_sha256=identities,
        source_bindings_unchanged_during_run=True, input_receipt_unchanged_during_run=True,
        input_binding=dict(path=str(input_receipt), sha256=actual_hash,
            selected_branch=branch, selected_candidate=candidate, coordinate_pointer=selected['pointer'],
            role='actual finite-bulk coordinate bytes only; every reported property recomputed'),
        governing_plan=PLAN, assignment_plan_revision=PLAN_REVISION,
        pinned_template_revision=TEMPLATE_REVISION, work_item='campaign', requirement_ids=['outcome', 'verification'],
        bare_coefficients=receipt['bare_coefficients'], parameters=dict(Lambda=Lambda, ue4=ue4),
        potential='Lambda/N * [kappa/2 sum_x ||C_x C_x^T||_F^2 + J/8 sum_bonds,mu ||Phi_x-U Phi_y U^dagger||_HS^2 - sum_global_top_half eigenvalues(h)]',
        kinetic='sqrt(Lambda)/2 sum_field p_c^2 + ue4 sum_colour_link p_A^2; Gauss G(q)^dagger p=0',
        all105_real_space_controls=probes, input_branches=results,
        scientific_controls_pass=True, scientific_controls_scope='executed derivation/probe identities, not stability assertions',
        distinctions=[
            'Every negative potential/principal mode is retained; horizontal restriction removes gauge velocities, not unstable modes.',
            'Off-stationary H G has the actual differentiated-gradient RHS; no forced zero or gauge projection of the potential Hessian.',
            'q0 stabilizers change the singular stratum and normal dimension; a stabilizer is not a physical zero mode by counting alone.',
            'Finite quadrature convergence comparisons are observations, not continuum Hessian or all-q certification.',
            'Potential is multiplied by Lambda; mass is 1/sqrt(Lambda) per Cartesian coordinate and 1/(2 ue4) per genuine colour link.',
            'Pure field restoring frequencies conditionally scale as Lambda^(3/4); pure link as sqrt(Lambda ue4); mixed modes are computed separately.',
            'No second field rotor, optimizer-Hessian mass, borrowed spin, quantum phase or quantum normal-confinement claim.',
            'Next join needs actual moving normal bundle, quantum measure/drift/Berry terms, uniform gap/control and the same-input discrete flux/current boundary.']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-receipt', type=Path, default=ROOT/'periodic_bulk_frame_receipt.json')
    parser.add_argument('--input-sha256', default=DEFAULT_INPUT_SHA256)
    parser.add_argument('--branch', default=DEFAULT_BRANCH)
    parser.add_argument('--candidate', choices=('coarse_stationary_candidate', 'validation_stationary_candidate'),
                        default='validation_stationary_candidate')
    parser.add_argument('--grid', type=int, default=8)
    parser.add_argument('--validation-grid', type=int, default=16)
    parser.add_argument('--Lambda', type=float, default=1e6)
    parser.add_argument('--ue4', type=float, default=.3)
    parser.add_argument('--all-branches', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--q-index', action='append', help='comma-separated integer q index on the actual candidate input grid')
    args = parser.parse_args()
    if args.output.resolve() == args.input_receipt.resolve():
        raise ValueError('Output must not overwrite the immutable input receipt')
    result = run(input_receipt=args.input_receipt, input_sha256=args.input_sha256,
                 branch=args.branch, candidate=args.candidate, grid=args.grid,
                 validation_grid=args.validation_grid, Lambda=args.Lambda, ue4=args.ue4,
                 all_branches=args.all_branches, q_indices=args.q_index)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output.resolve() in {(ROOT/name).resolve() for name in result['source_sha256']}:
        raise ValueError('Output must not overwrite a bound source/proof')
    if (source_hashes() != result['source_sha256']
            or hashlib.sha256(args.input_receipt.read_bytes()).hexdigest() != result['input_binding']['sha256']):
        raise RuntimeError('Source/proof/input changed before output; refusing mixed-bound receipt')
    args.output.write_text(text+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()), selected_branch=args.branch,
                         scientific_controls_pass=result['scientific_controls_pass'],
                         stability_claim=False), indent=2))


if __name__ == '__main__':
    main()

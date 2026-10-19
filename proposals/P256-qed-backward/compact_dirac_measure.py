#!/usr/bin/env python3
"""Frozen actual-bulk-mean compact U1 measure with one positive Dirac determinant.

This finite anisotropic Euclidean regulator has its own temporal Wilson term.
It is not the finite-dt primitive SU4/Pauli Hamiltonian and proves no phase.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp

from bulk_frame_compton import (band_parameters, raw_paths, read_bulk,
                                reconstruct_frame, source_hashes as bulk_sources)
from rotation_band import HOPPING, MIXING, RotationBand

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = 'c182aa82519fba2d52c8112f1b29cda63c055871502bb2d3177c1ea337331624'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
I4 = np.eye(4, dtype=complex)
GAMMA = np.array([MIXING, *[-1j * MIXING @ alpha for alpha in HOPPING]])
GAMMA5 = GAMMA[0] @ GAMMA[1] @ GAMMA[2] @ GAMMA[3]
PAIRS = tuple(itertools.combinations(range(4), 2))


def norm(x):
    return float(np.linalg.norm(x))


def require(value, message, tolerance=2e-9):
    if not np.isfinite(value) or value > tolerance:
        raise ValueError(f'{message}: {value} exceeds {tolerance}')


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def source_hashes():
    names = ('compact_dirac_measure.py', 'compact_dirac_measure.md',
             'photon_matching.py', 'quantum_su4_photon.py', 'gauss_photon_charge.py')
    return {**bulk_sources(), **{
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}


def geometry(shape):
    coordinates = np.array(list(np.ndindex(shape)), dtype=int)
    neighbours = np.empty((len(coordinates), 4), dtype=int)
    for site, coordinate in enumerate(coordinates):
        for axis in range(4):
            target = coordinate.copy()
            target[axis] = (target[axis] + 1) % shape[axis]
            neighbours[site, axis] = np.ravel_multi_index(tuple(target), shape)
    return coordinates, neighbours


def links_from_angles(angles):
    return np.exp(1j * np.asarray(angles))


def gauge_transform(links, phases, neighbours):
    return np.exp(1j * phases)[:, None] * links * np.exp(-1j * phases[neighbours])


def plaquettes(links, neighbours):
    return np.array([np.angle(links[:, mu] * links[neighbours[:, mu], nu]
                              * links[neighbours[:, nu], mu].conj() * links[:, nu].conj())
                     for mu, nu in PAIRS])


def heat_kernel_images(t, tolerance=1e-14):
    """Uniform relative error bound for the positive wrapped Gaussian sum."""
    count = 1
    while True:
        distance = 2 * np.pi * (count + 1) - np.pi
        log_bound = (np.log(2 * (1 + t / (2 * np.pi * distance)))
                     + (np.pi**2 - distance**2) / (2 * t))
        if log_bound <= np.log(tolerance):
            return count, float(np.exp(log_bound))
        count += 1


def heat_kernel_log(angles, t):
    """Haar-normalized rotor kernel sum_n exp(-t n^2/2) exp(i n angle)."""
    count, _ = heat_kernel_images(t)
    images = np.arange(-count, count + 1)
    distances = wrap(angles)[..., None] + 2 * np.pi * images
    return .5 * np.log(2 * np.pi / t) + logsumexp(-distances**2 / (2 * t), axis=-1)


def photon_log(links, neighbours, ue1, k1, dt):
    angles = plaquettes(links, neighbours)
    temporal = float(np.sum(heat_kernel_log(angles[:3], ue1 * dt)))
    spatial = float(k1 * dt * np.sum(np.cos(angles[3:])))
    return temporal + spatial, dict(temporal_heat_kernel_log=temporal,
                                    spatial_Wilson_log=spatial)


def dirac_matrix(links, shape, neighbours, band, dt, *,
                 spatial_wilson=True, temporal_wilson=True, time_dirac=True,
                 mass=True, gauge_links=True):
    """D=m+W_space+W_time+sum_mu Gamma_mu nabla_mu, one four-orbital field.

    Links multiply row x -> column x+mu. Time wrap is antiperiodic for fermions
    only. Flags are term-deletion adverse probes, never the sampled measure.
    """
    volume = int(np.prod(shape))
    if links.shape != (volume, 4) or not np.all(np.isfinite(links)):
        raise ValueError('Require finite links[volume,4]')
    require(float(np.max(np.abs(np.abs(links) - 1))), 'Genuine compact U1 links')
    wt = 1 / dt if temporal_wilson else 0.
    ws = band.r / band.a if spatial_wilson else 0.
    derivatives = [1 / dt if time_dirac else 0., *([band.v / band.a] * 3)]
    diagonal = (band.m if mass else 0.) + wt + 3 * ws
    result = np.kron(np.eye(volume), diagonal * I4)
    coordinates, _ = geometry(shape)
    for site in range(volume):
        row = slice(4 * site, 4 * site + 4)
        for mu in range(4):
            target = neighbours[site, mu]
            column = slice(4 * target, 4 * target + 4)
            phase = links[site, mu] if gauge_links else 1.
            if mu == 0 and coordinates[site, 0] == shape[0] - 1:
                phase = -phase
            wilson = wt if mu == 0 else ws
            result[row, column] += phase * (derivatives[mu] * GAMMA[mu] - wilson * I4) / 2
            result[column, row] += phase.conjugate() * (-derivatives[mu] * GAMMA[mu] - wilson * I4) / 2
    return result


def positive_logdet(matrix):
    """Use the actual complex determinant sign: never abs(det) or det(D†D)."""
    sign, logdet = np.linalg.slogdet(matrix)
    require(float(abs(sign - 1)), 'Single-flavour determinant is real positive', 2e-8)
    if not np.isfinite(logdet):
        raise ValueError('Finite nonsingular determinant required')
    return float(logdet), dict(sign_real=float(sign.real), sign_imag=float(sign.imag))


def joint_log(links, shape, neighbours, band, dt, ue1, k1) -> tuple[float, dict[str, Any]]:
    photon, terms = photon_log(links, neighbours, ue1, k1, dt)
    fermion, sign = positive_logdet(dirac_matrix(links, shape, neighbours, band, dt))
    return photon + fermion, dict(**terms, single_Dirac_logdet=fermion,
                                 determinant_sign=sign, log_weight=photon + fermion)


def pole_energy(band, momentum, dt, temporal_wilson=True):
    ms = band.mass(momentum)
    pv2 = float(np.sum((band.v * np.sin(band.a * momentum) / band.a)**2))
    if not temporal_wilson:
        return float(np.arcsinh(dt * np.sqrt(ms**2 + pv2)) / dt)
    # cosh(x)-1 evaluated without subtracting two nearly equal numbers.
    delta = dt**2 * (ms**2 + pv2) / (2 * (1 + ms * dt))
    return float(2 * np.arcsinh(np.sqrt(delta / 2)) / dt)


def pole_controls(band, dt):
    rows = []
    for angles in ([.31, -.47, .59], [.91, .23, -.71], [np.pi, .19, -.29]):
        p = np.array(angles) / band.a
        exact = band.energy(p)
        convergence = []
        for step in (dt, dt / 2, dt / 4, dt / 8):
            energy = pole_energy(band, p, step)
            ms = band.mass(p)
            pv2 = float(np.sum((band.v * np.sin(band.a * p) / band.a)**2))
            formula = ((1 + ms * step)**2 + 1 + step**2 * pv2) / (2 * (1 + ms * step))
            residual = abs(np.cosh(energy * step) - formula)
            require(residual, 'Exact temporal pole equation')
            # Direct analytic continuation of the executing four-by-four symbol.
            symbol = (ms + (1 - np.cosh(energy * step)) / step) * I4
            symbol += -np.sinh(energy * step) * GAMMA[0] / step
            symbol += 1j * np.einsum('i,ijk->jk', band.v * np.sin(band.a * p) / band.a, GAMMA[1:])
            singular = float(np.linalg.svd(symbol, compute_uv=False)[-1])
            require(singular, 'Continued free Dirac symbol has the claimed pole')
            convergence.append(dict(dt=step, E_dt=energy, error_to_actual_band=abs(energy - exact),
                                    cosh_equation_residual=residual, pole_singular_value=singular))
        errors = [row['error_to_actual_band'] for row in convergence]
        if not all(left > right for left, right in zip(errors, errors[1:])):
            raise ValueError('Free temporal refinement did not converge to actual RotationBand energy')
        no_wilson = pole_energy(band, p, dt, False)
        missing_spatial_mass = np.sqrt(band.m**2 + pv2)
        rows.append(dict(momentum=p.tolist(), actual_RotationBand_energy=exact, refinement=convergence,
                         deleted_temporal_Wilson_energy=no_wilson,
                         deleted_temporal_Wilson_finite_dt_defect=abs(no_wilson - convergence[0]['E_dt']),
                         deleted_spatial_Wilson_continuous_time_defect=abs(missing_spatial_mass - exact)))
    if min(row['deleted_temporal_Wilson_finite_dt_defect'] for row in rows) <= 1e-10:
        raise ValueError('Temporal-Wilson deletion did not discriminate the finite regulator')
    return rows


def current_controls(bare, frame, band, unitary):
    """Raw mean-field Peierls derivatives, beta H and source band comparisons."""
    _, raw = raw_paths(bare, frame)
    _, reduced = band.couplings()
    beta_raw = unitary @ MIXING @ unitary.conj().T
    gamma_raw = np.array([unitary @ g @ unitary.conj().T for g in GAMMA[1:]])
    eps, eps_out = np.array([.31, -.23, .41]), np.array([-.19, .37, .29])
    rows = []
    for in_angles, out_angles in (([.19, -.31, .47], [-.23, .17, .53]),
                                   ([.63, .29, -.37], [.11, -.41, .71])):
        pin, pout = np.array(in_angles) / band.a, np.array(out_angles) / band.a
        midpoint = (pin + pout) / 2
        raw_j, raw_c = np.zeros((3, 4, 4), complex), np.zeros((4, 4), complex)
        dirac_j, dirac_c = np.zeros_like(raw_j), np.zeros_like(raw_c)
        omit_mass_j, omit_contact = np.zeros_like(raw_j), np.zeros_like(raw_c)
        for displacement, coupling in raw:
            axis = int(np.flatnonzero(displacement)[0])
            phase = np.exp(1j * band.a * np.dot(midpoint, displacement))
            forward, backward = beta_raw @ coupling, beta_raw @ coupling.conj().T
            raw_j[axis] += 1j * band.a * (phase * coupling - phase.conjugate() * coupling.conj().T)
            raw_c -= band.e**2 * band.a**2 * eps[axis] * eps_out[axis] * (
                phase * coupling + phase.conjugate() * coupling.conj().T)
            dirac_j[axis] += 1j * band.a * (phase * forward - phase.conjugate() * backward)
            dirac_c -= band.e**2 * band.a**2 * eps[axis] * eps_out[axis] * (
                phase * forward + phase.conjugate() * backward)
            omit_mass_j[axis] = 1j * band.v * np.cos(band.a * midpoint[axis]) * gamma_raw[axis]
        source_j = unitary @ band.current(pout, pin) @ unitary.conj().T
        source_c = unitary @ band.contact(pout, pin, np.r_[0., eps], np.r_[0., eps_out],
                                          pout - pin, pout - pin) @ unitary.conj().T
        errors = dict(raw_D_current_equals_beta_raw_H_current=norm(dirac_j - beta_raw @ raw_j),
                      raw_D_contact_equals_beta_raw_H_contact=norm(dirac_c - beta_raw @ raw_c),
                      raw_D_current_equals_beta_source_current=norm(dirac_j - beta_raw @ source_j),
                      raw_D_contact_equals_beta_source_contact=norm(dirac_c - beta_raw @ source_c))
        for key, value in errors.items():
            require(value, key, 2e-6)
        # Independent dense link derivatives at nonzero compact phases below.
        finite_rows = []
        for axis, ((_, t_raw), (_, t)) in enumerate(zip(raw, reduced)):
            theta, step = .137 + .071 * axis, 2e-4
            f_raw, b_raw = beta_raw @ t_raw, beta_raw @ t_raw.conj().T
            def block(angle):
                return np.exp(1j * angle) * f_raw + np.exp(-1j * angle) * b_raw
            first = 1j * (np.exp(1j * theta) * f_raw - np.exp(-1j * theta) * b_raw)
            second = -(np.exp(1j * theta) * f_raw + np.exp(-1j * theta) * b_raw)
            first_error = norm((block(theta + step) - block(theta - step)) / (2 * step) - first)
            second_error = norm((block(theta + step) - 2 * block(theta) + block(theta - step)) / step**2 - second)
            reduced_error = norm(f_raw - unitary @ (MIXING @ t) @ unitary.conj().T)
            for key, value in (('link first derivative', first_error), ('link contact', second_error),
                               ('raw to mean-owned path', reduced_error)):
                require(value, key, 2e-6)
            finite_rows.append(dict(axis=axis, theta=theta, first_derivative_error=first_error,
                                    contact_error=second_error, raw_reduced_forward_error=reduced_error))
        wrong_current = norm(dirac_j - omit_mass_j)
        wrong_contact = norm(dirac_c - omit_contact)
        wrong_beta = norm(dirac_j - source_j)
        if min(wrong_current, wrong_contact, wrong_beta) <= 1e-6:
            raise ValueError('Nontrivial current/contact term-deletion controls failed')
        rows.append(dict(pin=pin.tolist(), pout=pout.tolist(), errors=errors,
                         nonzero_link_derivatives=finite_rows,
                         deleted_spatial_Wilson_vertex_defect=wrong_current,
                         deleted_contact_defect=wrong_contact, deleted_beta_defect=wrong_beta))
    return dict(rows=rows, charge=band.charge, theta_equals_e_a_A=True,
                current_is_charge_divided_midpoint_link_vertex=True,
                primitive_owner='full_band_gauss_section.peierls_action and local_orbital_stability.scalar_paths',
                raw_frame_reduction_tolerance=2e-6,
                two_path_owner_note='Primitive two-cell source averages two oriented paths; this 3D nearest-axis band uses each source path with its original coefficient.')


def spatial_hamiltonian(links, spatial_shape, band):
    _, neighbours = geometry((1, *spatial_shape))
    sites = int(np.prod(spatial_shape))
    onsite, paths = band.couplings()
    result = np.kron(np.eye(sites), onsite)
    for site in range(sites):
        row = slice(4 * site, 4 * site + 4)
        for axis, (_, hopping) in enumerate(paths):
            target = neighbours[site, axis + 1]
            column = slice(4 * target, 4 * target + 4)
            result[row, column] += links[site, axis] * hopping
            result[column, row] += links[site, axis].conjugate() * hopping.conj().T
    return result


def neutral_transfer_controls(shape, band, dt, seed):
    """Static and time-dependent Wilson/canonical neutral determinant reductions."""
    rng = np.random.default_rng(seed)
    spatial_shape, nt = shape[1:], shape[0]
    sites = int(np.prod(spatial_shape))
    spatial_links = links_from_angles(rng.uniform(-.71, .83, (sites, 3)))
    h = spatial_hamiltonian(spatial_links, spatial_shape, band)
    beta_values, beta_basis = eigh(MIXING)
    order = np.r_[np.flatnonzero(beta_values > 0), np.flatnonzero(beta_values < 0)]
    change = np.kron(np.eye(sites), beta_basis[:, order])
    # Group all positive-beta orbitals, then all negative-beta orbitals.
    grouped = np.r_[np.array([4 * x + i for x in range(sites) for i in (0, 1)]),
                    np.array([4 * x + i for x in range(sites) for i in (2, 3)])]
    change = change[:, grouped]
    hh = change.conj().T @ h @ change
    rank = 2 * sites
    w, f = hh[:rank, :rank], hh[:rank, rank:]
    require(norm(hh[rank:, rank:] + w), 'Equal spatial Wilson blocks in beta basis')
    b = np.eye(rank) + dt * w
    inv_b = np.linalg.solve(b, np.eye(rank))
    c = dt * f
    transfer = np.block([[inv_b, -inv_b @ c],
                         [-c.conj().T @ inv_b, b + c.conj().T @ inv_b @ c]])
    transfer_eigenvalues = np.linalg.eigvalsh(transfer)
    if transfer_eigenvalues[0] <= 0:
        raise ValueError('Static Wilson transfer must be strictly positive')
    log_b_value, _ = positive_logdet(b)
    transfer_power = np.linalg.matrix_power(transfer, nt)
    _, neighbours = geometry(shape)
    rows = []
    for holonomy in (.0, .37, -.61):
        links = np.ones((nt * sites, 4), complex)
        links[:, 1:] = np.tile(spatial_links, (nt, 1))
        links[:, 0] = np.exp(1j * holonomy / nt)
        d = dirac_matrix(links, shape, neighbours, band, dt)
        sign_d, log_d = np.linalg.slogdet(d)
        z = np.exp(-1j * holonomy)  # Canonical fugacity is inverse temporal-link holonomy.
        sign_t, log_t = np.linalg.slogdet(np.eye(4 * sites) + z * transfer_power)
        reduced_sign = z**(-rank) * sign_t
        reduced_log = -4 * nt * sites * np.log(dt) + nt * log_b_value + log_t
        require(abs(log_d - reduced_log), 'Wilson/canonical transfer logdet identity', 2e-7)
        require(abs(sign_d - reduced_sign), 'Neutral-background Wilson transfer phase identity', 2e-7)
        wrong_phase = abs(sign_d - sign_t)
        rows.append(dict(temporal_holonomy=holonomy, actual_D_sign=[float(sign_d.real), float(sign_d.imag)],
                         neutral_reduced_sign=[float(reduced_sign.real), float(reduced_sign.imag)],
                         logdet_identity_error=float(abs(log_d - reduced_log)),
                         sign_identity_error=float(abs(sign_d - reduced_sign)),
                         deleted_background_phase_defect=float(wrong_phase)))
    if min(row['deleted_background_phase_defect'] for row in rows[1:]) <= .05:
        raise ValueError('Nonzero holonomy must expose the canonical N versus N-2*sites mismatch')
    # General temporal gauge: independent spatial links on every slice and
    # site-dependent temporal holonomy on the final seam, not only a free pole.
    dynamic_links = np.ones((nt * sites, 4), complex)
    dynamic_links[:, 1:] = links_from_angles(rng.uniform(-.81, .92, (nt * sites, 3)))
    holonomies = rng.uniform(-.73, .67, sites)
    dynamic_links[(nt - 1) * sites:, 0] = np.exp(1j * holonomies)
    product = np.eye(4 * sites, dtype=complex)
    normalization = 0.
    for time_slice in range(nt):
        h_t = spatial_hamiltonian(dynamic_links[time_slice * sites:(time_slice + 1) * sites, 1:],
                                  spatial_shape, band)
        h_t = change.conj().T @ h_t @ change
        w_t, f_t = h_t[:rank, :rank], h_t[:rank, rank:]
        b_t = np.eye(rank) + dt * w_t
        inv_t = np.linalg.solve(b_t, np.eye(rank)); c_t = dt * f_t
        t_t = np.block([[inv_t, -inv_t @ c_t],
                        [-c_t.conj().T @ inv_t, b_t + c_t.conj().T @ inv_t @ c_t]])
        product = t_t @ product
        log_t_b, _ = positive_logdet(b_t)
        normalization += log_t_b
    fugacities = np.r_[np.repeat(np.exp(-1j * holonomies), 2),
                      np.repeat(np.exp(-1j * holonomies), 2)]
    dynamic_sign, dynamic_log = np.linalg.slogdet(dirac_matrix(
        dynamic_links, shape, neighbours, band, dt))
    canonical_sign, canonical_log = np.linalg.slogdet(
        np.eye(4 * sites) + fugacities[:, None] * product)
    neutral_sign = np.exp(2j * holonomies.sum()) * canonical_sign
    neutral_log = -4 * nt * sites * np.log(dt) + normalization + canonical_log
    require(abs(dynamic_log - neutral_log), 'Time-dependent neutral transfer logdet identity', 2e-7)
    require(abs(dynamic_sign - neutral_sign), 'Local-background transfer phase identity', 2e-7)
    dynamic_reduction = dict(logdet_identity_error=float(abs(dynamic_log - neutral_log)),
                             sign_identity_error=float(abs(dynamic_sign - neutral_sign)),
                             site_temporal_holonomies=holonomies.tolist(),
                             Wilson_normalization_log=normalization,
                             deleted_normalization_logweight_defect=abs(normalization),
                             deleted_local_background_phase_defect=float(abs(dynamic_sign - canonical_sign)),
                             canonical_fugacity_is_inverse_link_holonomy=True)
    refinements = []
    for step in (dt, dt / 2, dt / 4, dt / 8):
        bb = np.eye(rank) + step * w
        ib = np.linalg.solve(bb, np.eye(rank)); cc = step * f
        tt = np.block([[ib, -ib @ cc], [-cc.conj().T @ ib, bb + cc.conj().T @ ib @ cc]])
        values, vectors = np.linalg.eigh(tt)
        h_dt = -(vectors * (np.log(values) / step)) @ vectors.conj().T
        log_bb, _ = positive_logdet(bb)
        refinements.append(dict(dt=step, H_dt_to_actual_frozen_H_error=norm(h_dt - hh),
                                normalization_log_per_dt=log_bb / step,
                                limiting_gauge_independent_Tr_W=float(np.trace(w).real),
                                normalization_limit_error=abs(log_bb / step - np.trace(w).real)))
    if not all(left['H_dt_to_actual_frozen_H_error'] > right['H_dt_to_actual_frozen_H_error']
               for left, right in zip(refinements, refinements[1:])):
        raise ValueError('Static finite-time transfer does not converge to the source Hamiltonian')
    return dict(spatial_link_seed=seed, temporal_beta_rank_each=rank,
                transfer_eigenvalue_min=float(transfer_eigenvalues[0]),
                transfer_determinant_log=float(np.linalg.slogdet(transfer)[1]),
                static_nontrivial_link_holonomies=rows, temporal_refinement=refinements,
                time_dependent_spatial_links_local_holonomy=dynamic_reduction,
                canonical_charge='Q=N_light-2*spatial_sites; local generator n_light(x)-2',
                exact_reduction='det(D)=dt^(-4*V) prod_t det(B_t) prod_x z_x^(-2) det(I+Z prod_t T_t); z_x=inverse temporal holonomy',
                flavour_count=1, heavy_species_in_measure=False,
                full_primitive_Hamiltonian_identity=False,
                missing_join='The frozen conditional U1 light Hamiltonian transfer is derived. An identity to the unfrozen primitive neutral SU4/Pauli Gauss trace, including heavy orbitals, dynamical fields, centre constraints and their integration/Jacobian, has not been constructed.')


def measure_controls(shape, band, dt, ue1, k1, seed):
    rng = np.random.default_rng(seed)
    volume = int(np.prod(shape))
    phases = rng.uniform(-.9, .8, volume)
    links = links_from_angles(rng.uniform(-1.1, 1.3, (volume, 4)))
    _, neighbours = geometry(shape)
    d = dirac_matrix(links, shape, neighbours, band, dt)
    transformed = gauge_transform(links, phases, neighbours)
    dg = dirac_matrix(transformed, shape, neighbours, band, dt)
    g = np.repeat(np.exp(1j * phases), 4)
    expected = g[:, None] * d * g.conj()[None, :]
    gamma5 = np.kron(np.eye(volume), GAMMA5)
    covariance = norm(dg - expected)
    chiral = norm(d.conj().T - gamma5 @ d @ gamma5)
    hermitian = (d + d.conj().T) / 2
    minimum = float(np.linalg.eigvalsh(hermitian)[0])
    require(covariance, 'Local compact gauge covariance')
    require(chiral, 'gamma5 Hermiticity')
    if minimum < band.m - 2e-9:
        raise ValueError('Hermitian part fell below the actual positive mean mass')
    weight, terms = joint_log(links, shape, neighbours, band, dt, ue1, k1)
    changed_weight, _ = joint_log(transformed, shape, neighbours, band, dt, ue1, k1)
    require(abs(weight - changed_weight), 'Actual combined log-measure gauge invariance', 2e-7)
    no_links = dirac_matrix(links, shape, neighbours, band, dt, gauge_links=False)
    no_links_g = dirac_matrix(transformed, shape, neighbours, band, dt, gauge_links=False)
    deleted_links_defect = norm(no_links_g - g[:, None] * no_links * g.conj()[None, :])
    if deleted_links_defect <= .01:
        raise ValueError('Deleting genuine fermion gauge links must fail local covariance')
    # Positive-Hermitian/gamma5 alone require BOTH. Delete the mass on a fully
    # periodic (including time) cold probe: an exact constant zero mode remains.
    cold = np.ones_like(links)
    # A seam -1 cancels the fermion antiperiodicity, making the zero mode reachable.
    coordinates, _ = geometry(shape)
    cold[coordinates[:, 0] == shape[0] - 1, 0] = -1
    massless = dirac_matrix(cold, shape, neighbours, band, dt, mass=False)
    zero_mode = float(np.linalg.svd(massless, compute_uv=False)[-1])
    require(zero_mode, 'Mass deletion permits an actual zero mode', 2e-8)
    # Deleting just one directed term, rather than squaring a determinant, breaks
    # gamma5 Hermiticity and produces a genuinely complex one-flavour determinant.
    broken = d.copy()
    site, mu = 0, 1
    target = neighbours[site, mu]
    broken[4 * site:4 * site + 4, 4 * target:4 * target + 4] -= (
        links[site, mu] * (band.v * GAMMA[mu] - band.r * I4) / (2 * band.a))
    broken_gamma5 = norm(broken.conj().T - gamma5 @ broken @ gamma5)
    broken_sign, _ = np.linalg.slogdet(broken)
    if broken_gamma5 <= .01 or abs(broken_sign.imag) <= 1e-8:
        raise ValueError('One-direction deletion must expose gamma5/sign failure')
    deletions = []
    for label, flags in (('spatial_Wilson', dict(spatial_wilson=False)),
                         ('temporal_Wilson', dict(temporal_wilson=False)),
                         ('temporal_Dirac', dict(time_dirac=False))):
        modified = dirac_matrix(links, shape, neighbours, band, dt, **flags)
        altered, _ = positive_logdet(modified)
        defect = abs(altered - terms['single_Dirac_logdet'])
        if defect <= 1e-8:
            raise ValueError(f'{label} deletion must change the actual determinant weight')
        deletions.append(dict(deleted_term=label, logdet_defect=defect,
                              note='Deletion need not destroy positivity; it changes the required physical/regulator law.'))
    photons = plaquettes(links, neighbours)
    spatial_without = weight - terms['spatial_Wilson_log']
    # An isotropic cosine in time is not the heat kernel.
    guessed_photon = k1 * dt * float(np.sum(np.cos(photons)))
    if abs(guessed_photon - terms['temporal_heat_kernel_log'] - terms['spatial_Wilson_log']) <= .01:
        raise ValueError('Isotropic guessed photon action did not discriminate')
    t = ue1 * dt
    grid_size = max(512, int(np.ceil(64 / np.sqrt(t))))
    theta = (np.arange(grid_size) + .5) * 2 * np.pi / grid_size - np.pi
    kernel = np.exp(heat_kernel_log(theta, t))
    normalization = float(kernel.mean())
    first_fourier = float(np.mean(kernel * np.cos(theta)))
    require(abs(normalization - 1), 'Haar rotor heat-kernel normalization', 2e-9)
    require(abs(first_fourier - np.exp(-t / 2)), 'Actual UE1/2 rotor energy Fourier coefficient', 2e-9)
    images, bound = heat_kernel_images(t)
    return dict(nontrivial_link_seed=seed, local_gauge_covariance_residual=covariance,
                gamma5_Hermiticity_residual=chiral, Hermitian_part_min=minimum,
                actual_positive_mass=band.m, Hermitian_lower_bound_satisfied=minimum >= band.m - 2e-9,
                joint_gauge_invariance_error=abs(weight - changed_weight), actual_terms=terms,
                heat_kernel=dict(t_UE1_dt=t, images_each_side=images, uniform_relative_tail_bound=bound,
                                 Haar_normalization=normalization, first_Fourier=first_fourier,
                                 exact_first_Fourier=float(np.exp(-t / 2))),
                badcontrols=dict(deleted_gauge_link_covariance_defect=deleted_links_defect,
                                 deleted_mass_periodic_zero_singular_value=zero_mode,
                                 deleted_directed_term_gamma5_defect=broken_gamma5,
                                 deleted_directed_term_determinant_sign=[float(broken_sign.real), float(broken_sign.imag)],
                                 determinant_term_deletions=deletions,
                                 deleted_spatial_photon_weight_defect=abs(weight - spatial_without),
                                 deleted_temporal_photon_weight_defect=abs(terms['temporal_heat_kernel_log']),
                                 guessed_isotropic_photon_log_weight_defect=abs(guessed_photon - terms['temporal_heat_kernel_log'] - terms['spatial_Wilson_log'])),
                doubled_flavour_logweight_defect=abs(terms['single_Dirac_logdet']),
                sign_statement='The primary measure uses slogdet(D) with sign checked +1, not absolute value, det(D†D), a second species or a rooted determinant.')


def observables(links, shape, neighbours):
    angles = plaquettes(links, neighbours)
    lookup = {pair: angle for pair, angle in zip(PAIRS, angles)}
    def oriented(a, b):
        return lookup[a, b] if a < b else -lookup[b, a]
    currents = []
    for mu in range(4):
        a, b, c = [axis for axis in range(4) if axis != mu]
        flux = (oriented(b, c)[neighbours[:, a]] - oriented(b, c)
                + oriented(c, a)[neighbours[:, b]] - oriented(c, a)
                + oriented(a, b)[neighbours[:, c]] - oriented(a, b)) / (2 * np.pi)
        # epsilon(mu,a,b,c): moving mu to first requires mu transpositions.
        currents.append((-1)**mu * flux)
    currents = np.array(currents)
    integers = np.rint(currents)
    require(float(np.max(abs(currents - integers))), 'Compact cube flux quantization', 2e-9)
    divergence = sum(integers[mu, neighbours[:, mu]] - integers[mu] for mu in range(4))
    require(float(np.max(abs(divergence))), 'Compact dual current conservation')
    coordinates, _ = geometry(shape)
    sectors: dict[str, list[int]] = {}
    for mu, nu in PAIRS:
        other = [axis for axis in range(4) if axis not in (mu, nu)]
        fluxes = []
        for fixed in itertools.product(*(range(shape[axis]) for axis in other)):
            keep = np.ones(len(coordinates), dtype=bool)
            for axis, value in zip(other, fixed):
                keep &= coordinates[:, axis] == value
            flux = float(lookup[mu, nu][keep].sum() / (2 * np.pi))
            require(abs(flux - round(flux)), 'Periodic two-plane flux quantization')
            fluxes.append(round(flux))
        sectors[f'{mu}{nu}'] = fluxes
    rectangles = {}
    for mu, nu in PAIRS:
        # An actual 1x2 rectangular Wilson loop, with periodic repeated links.
        xmu, xnu = neighbours[:, mu], neighbours[:, nu]
        x2nu = neighbours[xnu, nu]
        loop = (links[:, mu] * links[xmu, nu] * links[neighbours[xmu, nu], nu]
                * links[x2nu, mu].conj() * links[xnu, nu].conj() * links[:, nu].conj())
        rectangles[f'{mu}{nu}'] = [float(loop.real.mean()), float(loop.imag.mean())]
    return dict(temporal_plaquette_cosine=float(np.cos(angles[:3]).mean()),
                spatial_plaquette_cosine=float(np.cos(angles[3:]).mean()),
                plaquette_sine_mean=float(np.sin(angles).mean()),
                Wilson_1x2_by_plane=rectangles,
                monopole_dual_current_abs_sum=int(np.abs(integers).sum()),
                monopole_dual_current_max=int(np.abs(integers).max()),
                monopole_integer_residual=float(np.max(abs(currents - integers))),
                monopole_dual_current_divergence_max=int(np.max(abs(divergence))),
                periodic_two_plane_flux_integers=sectors,
                topology_scope='DeGrand-Toussaint compact cube currents and plane fluxes, not a continuum topological charge or phase diagnosis')


def metropolis(start, shape, band, dt, ue1, k1, seed, burn, samples, updates, proposal):
    rng = np.random.default_rng(seed)
    volume = int(np.prod(shape))
    angles = np.zeros((volume, 4)) if start == 'cold' else rng.uniform(-np.pi, np.pi, (volume, 4))
    _, neighbours = geometry(shape)
    links = links_from_angles(angles)
    weight, terms = joint_log(links, shape, neighbours, band, dt, ue1, k1)
    initial = dict(terms=terms, observables=observables(links, shape, neighbours),
                   links_sha256=hashlib.sha256(links.tobytes()).hexdigest())
    accepted, attempted = 0, 0
    rows = []
    for sweep in range(burn + samples):
        for _ in range(updates):
            site, axis = int(rng.integers(volume)), int(rng.integers(4))
            old_angle, old_link = angles[site, axis], links[site, axis]
            angles[site, axis] = float(wrap(old_angle + rng.uniform(-proposal, proposal)))
            links[site, axis] = np.exp(1j * angles[site, axis])
            candidate, candidate_terms = joint_log(links, shape, neighbours, band, dt, ue1, k1)
            attempted += 1
            if np.log(rng.random()) < min(0., candidate - weight):
                weight, terms = candidate, candidate_terms
                accepted += 1
            else:
                angles[site, axis], links[site, axis] = old_angle, old_link
        if sweep >= burn:
            rows.append(dict(sample=sweep - burn, terms=terms,
                             observables=observables(links, shape, neighbours)))
    final_weight, _ = joint_log(links, shape, neighbours, band, dt, ue1, k1)
    require(abs(final_weight - weight), 'Metropolis retained actual joint log measure')
    return dict(start=start, seed=seed, initial=initial, attempts=attempted, accepted=accepted,
                acceptance_fraction=accepted / attempted, burn_sweeps=burn, sample_sweeps=samples,
                updates_per_sweep=updates, proposal_width=proposal, samples=rows,
                final_link_angles=angles.tolist(), final_links_sha256=hashlib.sha256(links.tobytes()).hexdigest(),
                natural_bounded_loop_completed=True,
                interpretation='Finite raw correlated observations only: no equilibrium, hot/cold agreement, thermodynamic Coulomb or compact-phase assertion.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bulk-receipt', type=Path, required=True)
    parser.add_argument('--branch', required=True)
    parser.add_argument('--ue1', type=float, required=True)
    parser.add_argument('--k1', type=float, required=True)
    parser.add_argument('--dt', type=float, required=True)
    parser.add_argument('--lattice', type=int, nargs=4, metavar=('NT', 'NX', 'NY', 'NZ'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--burn', type=int, required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--updates-per-sweep', type=int)
    parser.add_argument('--proposal-width', type=float, default=.45)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    shape = tuple(args.lattice)
    if any(x < 2 for x in shape) or int(np.prod(shape)) > 64:
        parser.error('Require each lattice extent>=2 and volume<=64 (dense Dirac dimension<=256)')
    if not all(np.isfinite(x) and x > 0 for x in (args.ue1, args.k1, args.dt, args.proposal_width)):
        parser.error('Require finite positive UE1, K1, dt and proposal width')
    if not 1e-4 <= args.ue1 * args.dt <= 20:
        parser.error('Supported numerical heat-kernel range: 1e-4<=UE1*dt<=20')
    if args.burn < 0 or args.samples < 1 or args.seed < 0:
        parser.error('Require burn>=0, samples>=1 and seed>=0')
    updates = 4 * int(np.prod(shape)) if args.updates_per_sweep is None else args.updates_per_sweep
    if updates < 1 or 2 * (args.burn + args.samples) * updates > 20000:
        parser.error('Require updates>=1 and at most20000 total two-chain proposals')
    before = source_hashes()
    bare, frame, binding = read_bulk(args.bulk_receipt, args.branch)
    band, unitary, reconstruction = reconstruct_frame(bare, frame)
    if band.improved or not band.m > 0 or band.r <= 0 or band.v <= 0 or band.e != 1:
        raise ValueError('Require actual unimproved massive band, r>0, v>0 and primitive e=1')
    if args.dt * (band.m + 6 * band.r / band.a + 3 * band.v / band.a) > 2:
        raise ValueError('Supported bounded temporal transfer probe requires dt*(m+6r/a+3v/a)<=2')
    algebra = dict(Hermitian_gamma_residual=max(norm(g - g.conj().T) for g in GAMMA),
                   Clifford_residual=max(norm(GAMMA[i] @ GAMMA[j] + GAMMA[j] @ GAMMA[i]
                                               - (2 * I4 if i == j else 0))
                                          for i in range(4) for j in range(4)),
                   gamma5_Hermitian_residual=norm(GAMMA5 - GAMMA5.conj().T),
                   gamma5_square_residual=norm(GAMMA5 @ GAMMA5 - I4),
                   gamma5_anticommutation_residual=max(norm(GAMMA5 @ g + g @ GAMMA5) for g in GAMMA))
    for key, value in algebra.items():
        require(value, key)
    current = current_controls(bare, frame, band, unitary)
    poles = pole_controls(band, args.dt)
    transfer = neutral_transfer_controls(shape, band, args.dt, args.seed + 11)
    controls = measure_controls(shape, band, args.dt, args.ue1, args.k1, args.seed + 23)
    chains = [metropolis(start, shape, band, args.dt, args.ue1, args.k1,
                         args.seed + offset, args.burn, args.samples, updates, args.proposal_width)
              for start, offset in (('cold', 101), ('hot', 211))]
    input_unchanged = hashlib.sha256(args.bulk_receipt.read_bytes()).hexdigest() == binding['sha256']
    seed_binding = binding['original_seed_binding']
    seed_unchanged = hashlib.sha256(Path(seed_binding['path']).read_bytes()).hexdigest() == seed_binding['sha256']
    source_unchanged = before == source_hashes()
    if not (input_unchanged and seed_unchanged and source_unchanged):
        raise ValueError('Source/input bindings changed during the finite consumer')
    result: dict[str, Any] = dict(
        status='conditional_actual_bulk_mean_compact_U1_single_Dirac_measure',
        plan=PLAN, plan_revision=PLAN_REVISION, template_revision=TEMPLATE_REVISION,
        source_sha256=before, source_bindings_unchanged_during_run=source_unchanged,
        input_receipt_binding=binding, input_receipt_unchanged_during_run=input_unchanged,
        original_seed_unchanged_during_run=seed_unchanged,
        parameters=dict(lattice=shape, dt=args.dt, UE1=args.ue1, K1=args.k1, seed=args.seed,
                        burn=args.burn, samples=args.samples, updates_per_sweep=updates,
                        proposal_width=args.proposal_width, physical_spatial_spacing=band.a,
                        temporal_extent=args.dt * shape[0], band=band_parameters(band)),
        reconstruction=reconstruction, gamma_derivation=algebra,
        raw_current_contact=current, temporal_free_poles=poles,
        neutral_charge_and_temporal_transfer=transfer, joint_measure_controls=controls,
        photon_normalization=dict(H='UE1/2 sum E_link^2-K1 sum cos(curl theta)',
                                  temporal_Villain_quadratic_coefficient=1 / (args.ue1 * args.dt),
                                  spatial_cosine_coefficient=args.k1 * args.dt,
                                  theta_equals_a_A_at_e_one=True,
                                  physical_bare_photon_speed=band.a * np.sqrt(args.ue1 * args.k1)),
        extra_regulator=dict(temporal_Wilson_coefficient=1 / args.dt,
                             exact_finite_dt_Hamiltonian_identity=False,
                             dt_to_zero_conditional_light_transfer=True),
        chains=chains, all_construction_controls_pass=True,
        full_primitive_neutral_measure_join_claim=False, thermodynamic_phase_claim=False,
        charged_B_completion_claim=False, mechanical_spin_completion_claim=False,
        normal_field_completion_claim=False, radiative_completion_claim=False,
        Lorentz_completion_claim=False, campaign_complete=False)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        # The scientific run's output is caller-selected; suppliers/inputs are not overwritten.
        protected = {args.bulk_receipt.resolve(), Path(seed_binding['path']).resolve(),
                     *((ROOT / name).resolve() for name in before)}
        if args.output.resolve() in protected:
            raise ValueError('Output cannot overwrite a bound input or source')
        args.output.write_text(payload + '\n')
    print(payload)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

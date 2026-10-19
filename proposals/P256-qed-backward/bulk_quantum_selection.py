#!/usr/bin/env python3
"""Local source-sampled CAR sea selects a full quantum 3D Clifford field.

This is an explicitly CHANGED bare matter law, not a theorem for the original
itinerant heavy sea. All Cartesian fields, colour/EM links and aggregate-half
Fock sectors remain. Centre torons are retained, never fixed to identity.
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
from scipy.linalg import expm
from scipy.optimize import minimize

from local_orbital_stability import local_hamiltonian, periodic_grid, scalar_paths
from rotation_band import RotationBand
from sea_selection import SELECTED, coefficients, matrix_unit_basis, minimal_rank_witness, rank_report
from soft_budget import select_gram
from unrestricted_field_minimum import BARE, STIFFNESS, matrices

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = 'c504463f58f37719bbfeefd0954e9eee09f1b34d13b163861a609c104a10e8fa'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
PAIRS = tuple(itertools.combinations(range(4), 2))


def source_hashes():
    names = ('bulk_quantum_selection.py', 'bulk_quantum_selection.md',
             'local_orbital_stability.py', 'rotation_band.py', 'sea_selection.py',
             'soft_budget.py', 'unrestricted_field_minimum.py')
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def check(error, label, tolerance=2e-9):
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error} > {tolerance}')


def exterior(matrix):
    """Actual exterior-square representation; also the rank-one Slater projector."""
    return np.array([[matrix[i, k] * matrix[j, l] - matrix[i, l] * matrix[j, k]
                      for k, l in PAIRS] for i, j in PAIRS])


def car_two(matrix):
    """dGamma(matrix) on the actual six-dimensional two-particle CAR sector."""
    result = np.zeros((6, 6), dtype=complex)
    for column, occupied in enumerate(PAIRS):
        for source in occupied:
            remainder = list(occupied)
            sign = (-1) ** remainder.index(source)
            remainder.remove(source)
            for target in range(4):
                if target in remainder:
                    continue
                new = tuple(sorted((target, *remainder)))
                insert = sum(item < target for item in remainder)
                result[PAIRS.index(new), column] += sign * (-1) ** insert * matrix[target, source]
    return result


def epsilon_two():
    result = np.zeros((6, 6))
    for i, left in enumerate(PAIRS):
        for j, right in enumerate(PAIRS):
            word = left + right
            if len(set(word)) == 4:
                inversions = sum(word[a] > word[b] for a in range(4) for b in range(a + 1, 4))
                result[i, j] = (-1) ** inversions
    return result


def paired_section(h, *, complementary=False):
    """Phase-free equivariant normalized Slater pair, not a retained Fock cut."""
    eigenvalues, vectors = np.linalg.eigh(h)
    if eigenvalues[2] <= 0 or eigenvalues[1] >= 0:
        raise ValueError('The physical trial section needs its proved local sign gap')
    projector = vectors[:, 2:] @ vectors[:, 2:].conj().T
    state = exterior(projector)
    if complementary:
        state = state @ epsilon_two()
    check(abs(float(np.linalg.norm(state)) - 1), 'paired Slater norm')
    return state


class BulkQuantumLaw:
    """All fields/links on a genuine periodic N^3 lattice, N>=3.

    27 on-site fundamental/antifundamental selector pairs use source coefficients
    at the actual periodic 3^3 momenta. These coefficients define a new local law
    independent of the physical box size. Selector filling is aggregate-half.
    """
    def __init__(self, size=3, *, alignment=.2, colour_plaquette=.7, band=None):
        if size < 3 or min(alignment, colour_plaquette) <= 0:
            raise ValueError('Need N>=3 and strictly positive J/K4')
        self.size, self.alignment, self.colour_plaquette = size, alignment, colour_plaquette
        self.band = RotationBand(**BARE) if band is None else band
        self.basis = matrix_unit_basis()
        self.generators = self.basis / np.sqrt(8)
        self.sites = periodic_grid(size)
        self.volume = len(self.sites)
        self.channels = coefficients(self.band, 2 * np.pi * periodic_grid(3) / (3 * self.band.a))
        self.flavours = len(self.channels)
        self.minimum = select_gram(self.channels, STIFFNESS, 0.)
        self.gram = self.minimum['G']
        self.frame = np.sqrt(np.diag(self.gram))[:, None, None] * SELECTED
        radii = np.sqrt(np.einsum('fm,mn,fn->f', self.channels, self.gram, self.channels))
        self.gradient = STIFFNESS * self.gram - np.einsum(
            'fm,fn,f->mn', self.channels, self.channels, 1 / radii) / self.flavours
        self.index = {tuple(site): n for n, site in enumerate(self.sites)}
        self.target = np.array([[self.index[tuple((site + np.eye(3, dtype=int)[axis]) % size)]
                                 for axis in range(3)] for site in self.sites])
        self.onsite, raw_paths = scalar_paths(self.band)
        self.paths = []
        for displacement, scalar in raw_paths:
            axes = np.flatnonzero(displacement)
            if len(axes) != 1 or displacement[axes[0]] != 1:
                raise ValueError('The source endpoint law must remain unimproved nearest-axis')
            self.paths.append((int(axes[0]), np.asarray(scalar)))

    def fields(self, coordinates):
        return matrices(np.asarray(coordinates).reshape(self.volume, 4, 15), self.basis)

    def identity_links(self):
        return np.tile(np.eye(4, dtype=complex), (self.volume, 3, 1, 1))

    def uniform_fields(self):
        return np.tile(self.frame, (self.volume, 1, 1, 1))

    def toron_links(self, angles):
        links = self.identity_links()
        for x, site in enumerate(self.sites):
            for axis in range(3):
                if site[axis] == self.size - 1:
                    links[x, axis] = np.exp(1j * angles[axis]) * np.eye(4)
        return links

    def onebody(self, fields, links, phases=None):
        """Native endpoint-average source, colour only for H, colour+EM for L."""
        h = np.zeros((4 * self.volume, 4 * self.volume), dtype=complex)
        for x in range(self.volume):
            sx = slice(4 * x, 4 * x + 4)
            h[sx, sx] += self.onsite * fields[x, 0]
            for axis, scalar in self.paths:
                y = self.target[x, axis]
                sy = slice(4 * y, 4 * y + 4)
                fx = np.einsum('m,mij->ij', scalar, fields[x])
                fy = np.einsum('m,mij->ij', scalar, fields[y])
                coupling = (fx @ links[x, axis] + links[x, axis] @ fy) / 2
                if phases is not None:
                    coupling *= np.exp(1j * phases[x, axis])
                h[sx, sy] += coupling
                h[sy, sx] += coupling.conj().T
        return h

    def grams(self, fields):
        return np.einsum('xmij,xnji->xmn', fields, fields).real / 4

    def gram_functional(self, grams):
        radii = np.sqrt(np.maximum(0., np.einsum('fm,xmn,fn->xf', self.channels, grams, self.channels)))
        return STIFFNESS / 2 * np.sum(grams**2, axis=(1, 2)) - 2 * radii.mean(axis=1)

    def link_defects(self, fields, links):
        W, P = 0., 0.
        for x in range(self.volume):
            for axis in range(3):
                y, u = self.target[x, axis], links[x, axis]
                W += float(np.linalg.norm(fields[x] - u @ fields[y] @ u.conj().T)**2)
            for a, b in itertools.combinations(range(3), 2):
                loop = (links[x, a] @ links[self.target[x, a], b]
                        @ links[self.target[x, b], a].conj().T @ links[x, b].conj().T)
                # Positive form avoids subtracting two nearly equal numbers.
                P += float(np.linalg.norm(loop - np.eye(4))**2) / 2
        return W, P

    def selector(self, fields):
        blocks = np.einsum('fm,xmij->xfij', self.channels, fields)
        values = np.linalg.eigvalsh(blocks)
        # h^T anti-fundamental partner has identical eigenvalues. No site,
        # flavour or momentum occupation is fixed by this executing consumer.
        all_values = np.concatenate((values.reshape(-1), values.reshape(-1)))
        selected = np.argsort(all_values, kind='stable')[len(all_values) // 2:]
        occupied = np.zeros(len(all_values), dtype=bool)
        occupied[selected] = True
        sea = -float(all_values[selected].sum()) / (2 * self.flavours)
        counts = occupied.reshape(2, self.volume, self.flavours, 4).sum(axis=-1)
        forced = -float(values[:, :, 2:].sum()) / self.flavours
        return sea, dict(total_modes=len(all_values), total_occupied=len(selected),
            occupation_histogram={str(n): int(np.sum(counts == n)) for n in range(5)},
            incorrect_forced_two_per_block_energy=forced,
            forced_block_error=forced - sea)

    def leading(self, fields, links) -> dict[str, Any]:
        grams = self.grams(fields)
        elastic = STIFFNESS / 2 * float(np.sum(grams**2))
        sea, filling = self.selector(fields)
        W, P = self.link_defects(fields, links)
        local_lower = float(self.gram_functional(grams).sum())
        slack = elastic + sea - local_lower
        D = float(np.linalg.norm(grams - self.gram)**2)
        bound = (self.volume * self.minimum['value'] + STIFFNESS / 2 * D
                 + float(np.einsum('mn,xmn->', self.gradient, grams - self.gram))
                 + slack + self.alignment * W / 8 + self.colour_plaquette * P)
        actual = elastic + sea + self.alignment * W / 8 + self.colour_plaquette * P
        if slack < -1e-9 or actual < bound - 2e-9 * max(1., abs(actual)):
            raise AssertionError('Full-domain aggregate CAR/strong Gram inequality failed')
        return dict(actual_leading_energy=actual, exact_local_Gram_lower=local_lower,
            spectral_slack=slack, Gram_distance_squared_sum=D, full_alignment_W=W,
            colour_plaquette_P=P, strong_Gram_lower_with_residual=bound,
            lower_margin=actual - bound, filling=filling)


def original_toron_competition(law):
    k = 2 * np.pi * law.sites / (law.size * law.band.a)
    rows = []
    for angle in (0., np.pi / 2, np.pi, 3 * np.pi / 2):
        x = coefficients(law.band, k + angle / (law.size * law.band.a))
        candidate = select_gram(x, STIFFNESS, 0.)
        frame = np.sqrt(np.diag(candidate['G']))[:, None, None] * SELECTED
        fields = np.tile(frame, (law.volume, 1, 1, 1))
        links = law.toron_links([angle] * 3)
        spectrum = np.linalg.eigvalsh(law.onebody(fields, links))
        native = (STIFFNESS / 2 * float(np.sum(candidate['G']**2))
                  - float(spectrum[2 * law.volume:].sum()) / law.volume)
        check(abs(native - candidate['value']), 'actual central seam/Bloch source identity')
        W, P = law.link_defects(fields, links)
        check(W + P, 'allowed central toron zero local penalties')
        rows.append(dict(central_holonomy_angle_all_axes=angle, native_energy_per_site=native,
            reoptimized_Gram=candidate['G'].tolist(), W=W, P=P))
    improvement = rows[0]['native_energy_per_site'] - min(row['native_energy_per_site'] for row in rows)
    if law.size == 3 and improvement <= .001:
        raise AssertionError('Original identity-link global-supplier falsification was not reached')

    # Reducible Cartan fields commute with every diagonal toron. A twist can
    # improve their aggregate filling without ANY alignment/plaquette debt.
    rng = np.random.default_rng(222702)
    cartan = rng.normal(size=(4, 3)) * .3
    cartan = np.column_stack((cartan, -cartan.sum(axis=1)))
    gram = cartan @ cartan.T / 4
    def energy(theta):
        t = np.asarray(theta).reshape(3, 3)
        t = np.column_stack((t, -t.sum(axis=1))).T
        x = np.stack([coefficients(law.band, k + t[j] / (law.size * law.band.a)) for j in range(4)], axis=1)
        values = np.einsum('pjm,mj->pj', x, cartan)
        return STIFFNESS / 2 * float(np.sum(gram**2)) - float(np.sort(values.reshape(-1))[-2 * law.volume:].sum()) / law.volume
    zero = energy(np.zeros(9))
    fit = minimize(energy, np.linspace(-1.1, 1.3, 9), method='BFGS',
                   options={'gtol': 1e-8, 'maxiter': 400})
    t = fit.x.reshape(3, 3)
    t = np.column_stack((t, -t.sum(axis=1))).T
    links = law.identity_links()
    for x, site in enumerate(law.sites):
        for axis in range(3):
            if site[axis] == law.size - 1:
                links[x, axis] = np.diag(np.exp(1j * t[:, axis]))
    fields = np.tile(np.array([np.diag(row) for row in cartan]), (law.volume, 1, 1, 1))
    W, P = law.link_defects(fields, links)
    eigenvalues = np.linalg.eigvalsh(law.onebody(fields, links))
    actual = STIFFNESS / 2 * float(np.sum(gram**2)) - float(eigenvalues[2 * law.volume:].sum()) / law.volume
    check(abs(actual - fit.fun), 'actual noncentral Cartan seam consumer')
    check(W + P, 'reducible noncentral toron zero local penalties')
    if zero - actual <= 1e-6:
        raise AssertionError('Noncentral reducible-filling twist control failed')
    return dict(original_uniform_Clifford_central_torons=rows,
        identity_supplier_energy_improvement=improvement,
        identity_global_claim_rejected=bool(improvement > 1e-10),
        noncentral_reducible_competitor=dict(Cartan_fields=cartan.tolist(),
            seam_angles_by_colour_axis=t.tolist(), identity_energy_per_site=zero,
            actual_twisted_energy_per_site=actual, improvement=zero - actual,
            W=W, P=P, optimization_success=bool(fit.success),
            claim='A fixed reducible filling improves, not a global optimizer or Clifford-beating claim'))


def trial_constants(law, radius=.001, ue4=.3, ue1=.4, magnetic=.7) -> dict[str, Any]:
    g, V = law.gram, law.volume
    rho = float(np.sqrt(np.trace(g)))
    normg = float(np.linalg.norm(g))
    norms = np.linalg.norm(law.channels, axis=1)
    radii = np.sqrt(np.einsum('fm,mn,fn->f', law.channels, g, law.channels))
    r0 = min(radius, .01, float(np.min(radii) / (8 * np.max(norms))), float(np.sqrt(g[0, 0]) / 8))
    Lfield = V * (STIFFNESS * normg * (2 * rho + r0)
        + STIFFNESS / 2 * (2 * rho + r0)**2 * r0 + 4 * float(norms.mean()))
    w_slope = 4 + 2 * np.sqrt(2) * (rho + r0)
    Lalign = law.alignment / 8 * 3 * V * w_slope**2 * r0
    Lcolour = 12 * V * law.colour_plaquette * r0
    metric = V * (960 * float(np.sum(norms**2 / radii**2)) + 960 / float(g[0, 0]))
    b = abs(law.onsite) + 2 * sum(float(np.linalg.norm(scalar)) for _, scalar in law.paths)
    B0 = 4 * b * np.sqrt(2 * V * V * float(np.trace(g)))
    B1 = 8 * b * V**.75
    young = .75 * B1**(4 / 3) / STIFFNESS**(1 / 3)
    return dict(chart_radius=r0, L_leading_Lipschitz=Lfield + Lalign + Lcolour,
        field_bump_kinetic_A=680 * V, section_kinetic_metric_M=metric / 2,
        colour_bump_electric_B=ue4 * 3 * V * (920 + 20000 * r0**4),
        original_CAR_anchor_B0=float(B0), original_CAR_Young_B1=float(B1),
        original_CAR_Young_C=float(young), original_source_HS_b=float(b),
        photon_positive_potential_upper=6 * V * magnetic,
        ue4=ue4, ue1=ue1, magnetic=magnetic,
        gap_min=float(np.min(radii)), Gram_target_min=float(np.min(np.diag(g))),
        trial='square root of gauge-averaged normalized product density, paired Slater projector purification; EM Haar-constant',
        numerical_status='floating coefficients in proved formulas; no interval endpoints or solved quantum ground')


def quantum_bound(law, constants, Lambda):
    if not np.isfinite(Lambda) or Lambda < 1:
        raise ValueError('Finite Lambda>=1 required')
    L, A = constants['L_leading_Lipschitz'], constants['field_bump_kinetic_A']
    r = min(constants['chart_radius'], (2 * A / L)**(1 / 3) * Lambda**(-1 / 6))
    trial = (L * r + A / (np.sqrt(Lambda) * r**2)
        + constants['section_kinetic_metric_M'] / np.sqrt(Lambda)
        + constants['colour_bump_electric_B'] / (Lambda * r**2)
        + (constants['original_CAR_anchor_B0']
           + constants['original_CAR_Young_B1'] * law.volume**.25 * np.sqrt(2 * r * (2 * np.sqrt(np.trace(law.gram)) + r))
           + constants['photon_positive_potential_upper']) / Lambda)
    lower_cost = constants['original_CAR_anchor_B0'] / Lambda + constants['original_CAR_Young_C'] * Lambda**(-4 / 3)
    S = 2 * (trial + lower_cost)
    return dict(Lambda=Lambda, physical_packet_radius=r,
        scaled_trial_energy_excess_upper=float(trial), expected_selection_remainder_upper=float(S),
        expected_sum_Gram_distance_squared_upper=float(2 * S / STIFFNESS),
        expected_full_alignment_W_upper=float(8 * S / law.alignment),
        expected_colour_plaquette_P_upper=float(S / law.colour_plaquette),
        expected_actual_global_CAR_spectral_slack_upper=float(S),
        expected_actual_selector_Fock_excitation_upper=float(S),
        expected_selection_plus_Fock_excitation_upper=float(S),
        theorem_scope='any exact full physical neutral ground; exact Gstar in proof, numerical Ghat here',
        leading_rate='O(Lambda^-1/6) at each fixed finite N; constants grow with volume/flavour count',
        no_numerical_ground_state_claim=True)


def paired_section_controls(law):
    perturbation = .0001 * np.sin(np.arange(60) * .41).reshape(4, 15)
    control_frame = law.frame + matrices(perturbation, law.basis)
    h = np.einsum('m,mij->ij', law.channels[5], control_frame)
    state = paired_section(h)
    two = car_two(h)
    H = -(np.kron(two, np.eye(6)) + np.kron(np.eye(6), car_two(h.T)))
    vector = state.reshape(-1)
    selected_energy = -2 * float(np.linalg.eigvalsh(h)[2:].sum())
    error = float(np.linalg.norm(H @ vector - selected_energy * vector))
    check(error, 'actual paired fundamental/antifundamental CAR Slater eigenstate')
    generator = np.einsum('a,aij->ij', .2 * np.sin(np.arange(15) + .3), law.generators)
    unitary = expm(1j * generator)
    w = exterior(unitary)
    moved = paired_section(unitary @ h @ unitary.conj().T)
    covariance = float(np.linalg.norm(moved - w @ state @ w.conj().T))
    check(covariance, 'phase-free paired colour equivariance')
    original = paired_section(control_frame[0], complementary=True)
    original_moved = paired_section(unitary @ control_frame[0] @ unitary.conj().T, complementary=True)
    original_covariance = float(np.linalg.norm(original_moved - w @ original @ w.T))
    check(original_covariance, 'original heavy/light complementary Gauss section')
    Horig = (-np.kron(car_two(control_frame[0]), np.eye(6))
             + np.kron(np.eye(6), car_two(control_frame[0])))
    original_values = np.linalg.eigvalsh(control_frame[0])
    energy = float(original_values[:2].sum() - original_values[2:].sum())
    original_error = float(np.linalg.norm(Horig @ original.reshape(-1) - energy * original.reshape(-1)))
    check(original_error, 'original two-heavy/two-light trial onsite CAR energy')
    # Wrong anti-fundamental h (using h instead of h.T) fails this exact state.
    wrong = -(np.kron(two, np.eye(6)) + np.kron(np.eye(6), two))
    wrong_error = float(np.linalg.norm(wrong @ vector - selected_energy * vector))
    if wrong_error < 1e-4:
        raise AssertionError('Anti-fundamental transpose adverse control failed')
    direction = np.sin(np.arange(60) * .67 + .11).reshape(4, 15)
    direction /= np.linalg.norm(direction)
    dfields = matrices(direction, law.basis)
    dh = np.einsum('m,mij->ij', law.channels[5], dfields)
    eps = 1e-6
    derivative = (paired_section(h + eps * dh) - paired_section(h - eps * dh)) / (2 * eps)
    metric = float(np.linalg.norm(derivative)**2)
    bound = 960 * float(np.linalg.norm(law.channels[5])**2 / (np.linalg.eigvalsh(h)[2]**2))
    if metric > bound:
        raise AssertionError('Physical packet section derivative bound failed')
    gauss_errors, original_gauss_errors = [], []
    for T in law.generators:
        d = 1j * (T @ h - h @ T)
        d0 = 1j * (T @ control_frame[0] - control_frame[0] @ T)
        dstate = (paired_section(h + eps * d) - paired_section(h - eps * d)) / (2 * eps)
        dorig = (paired_section(control_frame[0] + eps * d0, complementary=True)
                 - paired_section(control_frame[0] - eps * d0, complementary=True)) / (2 * eps)
        charge = car_two(T)
        gauss_errors.append(float(np.linalg.norm(dstate - 1j * (charge @ state - state @ charge))))
        original_gauss_errors.append(float(np.linalg.norm(dorig - 1j * (charge @ original + original @ charge.T))))
    check(max(gauss_errors + original_gauss_errors), 'all15 actual differential Cartesian/matter Gauss', 3e-8)
    # The light number is the actual CAR number operator, not a singlet label.
    em_number = car_two(np.eye(4)) - 2 * np.eye(6)
    em_gauss = float(np.linalg.norm(original @ em_number.T))
    check(em_gauss, 'actual original light-background EM Gauss')
    actual_metric_sum = 0.
    for x in law.channels:
        block = np.einsum('m,mij->ij', x, control_frame)
        for mu, B in itertools.product(range(4), law.basis):
            derivative = (paired_section(block + eps * x[mu] * B)
                          - paired_section(block - eps * x[mu] * B)) / (2 * eps)
            actual_metric_sum += float(np.linalg.norm(derivative)**2)
    for B in law.basis:
        derivative = (paired_section(control_frame[0] + eps * B, complementary=True)
                      - paired_section(control_frame[0] - eps * B, complementary=True)) / (2 * eps)
        actual_metric_sum += float(np.linalg.norm(derivative)**2)
    analytic_metric_sum = 2 * trial_constants(law)['section_kinetic_metric_M'] / law.volume
    if actual_metric_sum > analytic_metric_sum:
        raise AssertionError('Actual all-flavour/all60 physical section kinetic bound failed')
    return dict(actual_36state_CAR_eigenstate_residual=error,
        fundamental_antifundamental_covariance_error=covariance,
        original_complementary_covariance_error=original_covariance,
        original_CAR_onsite_residual=original_error,
        wrong_antifundamental_transpose_residual=wrong_error,
        finite_difference_section_metric=metric, analytic_metric_upper=bound,
        nonClifford_trial_field_perturbation_norm=float(np.linalg.norm(perturbation)),
        actual_all60_allflavour_metric_sum=actual_metric_sum,
        analytic_all60_allflavour_metric_upper=analytic_metric_sum,
        all15_selector_Gauss_residual_max=max(gauss_errors),
        all15_original_Gauss_residual_max=max(original_gauss_errors),
        original_EM_number_Gauss_residual=em_gauss,
        physical_Gauss_scope='equivariant full-field section and invariant scalar density; not a Fock-sector truncation')


def retained_light_current_controls(law):
    rng = np.random.default_rng(222704)
    fields = law.uniform_fields() + law.fields(rng.normal(size=(law.volume, 4, 15)) * .03)
    links = np.array([[expm(1j * np.einsum('a,aij->ij', rng.normal(size=15) * .13, law.generators))
                       for _ in range(3)] for _ in range(law.volume)])
    phases = rng.uniform(-.7, .7, (law.volume, 3))
    phi = .3 * np.sin(np.arange(law.volume) * .79)
    direction = phi[:, None] - phi[law.target]
    h = law.onebody(fields, links, phases)
    current, contact = np.zeros_like(h), np.zeros_like(h)
    for x in range(law.volume):
        sx = slice(4*x, 4*x+4)
        for axis in range(3):
            y = law.target[x, axis]
            sy = slice(4*y, 4*y+4)
            coupling, d = h[sx, sy], direction[x, axis]
            current[sx, sy] += 1j * d * coupling
            current[sy, sx] += (1j * d * coupling).conj().T
            contact[sx, sy] -= d*d * coupling
            contact[sy, sx] -= d*d * coupling.conj().T
    charge = np.diag(np.repeat(phi, 4))
    commutator = charge @ h - h @ charge
    first_ward = float(np.linalg.norm(current - 1j * commutator))
    second_ward = float(np.linalg.norm(contact + charge @ commutator - commutator @ charge))
    check(first_ward + second_ward, 'retained actual light current/contact full U1 Ward')
    eps = 2e-5
    plus, minus = law.onebody(fields, links, phases + eps * direction), law.onebody(fields, links, phases - eps * direction)
    first_fd = float(np.linalg.norm((plus - minus) / (2 * eps) - current))
    second_fd = float(np.linalg.norm((plus + minus - 2*h) / eps**2 - contact))
    check(first_fd, 'retained native light phase current finite difference', 2e-8)
    check(second_fd, 'retained native light phase contact finite difference', 2e-5)
    gauge = np.diag(np.exp(1j * np.repeat(phi, 4)))
    finite_ward = float(np.linalg.norm(law.onebody(fields, links, phases + direction) - gauge @ h @ gauge.conj().T))
    check(finite_ward, 'retained finite full U1 gauge identity')
    wrong_contact = float(np.linalg.norm(charge @ commutator - commutator @ charge))
    if wrong_contact < .01:
        raise AssertionError('Deleted light current contact adverse control failed')
    return dict(current_Ward_error=first_ward, contact_Ward_error=second_ward,
        native_current_difference_error=first_fd, native_contact_difference_error=second_fd,
        finite_U1_covariance_error=finite_ward, omitted_contact_violation=wrong_contact,
        scope='actual full nonuniform 3D endpoint light CAR current/contact; no quantum pole/photon-phase implication')


def run(*, size=3, Lambdas=(1e24, 1e36, 1e48, 1e60)) -> dict[str, Any]:
    identities = source_hashes()
    law = BulkQuantumLaw(size)
    rank = rank_report(law.channels)
    if rank['quadratic_rank'] != 10:
        raise AssertionError('Local source samples lack full Clifford equality richness')
    check(float(np.linalg.norm(law.gradient)), 'source-selected full PSD Gram residual', 2e-10)
    fields, links = law.uniform_fields(), law.identity_links()
    native_error = float(np.linalg.norm(law.onebody(fields, links)
        - local_hamiltonian(fields, law.band, law.sites, size)))
    check(native_error, 'actual read-only native periodic endpoint consumer')
    attainment = law.leading(fields, links)
    check(abs(attainment['actual_leading_energy'] - law.volume * law.minimum['value']),
          'actual aggregate total-half selector attainment')
    failed_original = original_toron_competition(law)
    controls = []
    rng = np.random.default_rng(222703)
    for scale in (.001, .03, .2, .7):
        arbitrary = fields + law.fields(rng.normal(size=(law.volume, 4, 15)) * scale)
        arbitrary_links = np.array([[expm(1j * np.einsum('a,aij->ij', rng.normal(size=15) * scale, law.generators))
                                     for _ in range(3)] for _ in range(law.volume)])
        phases = rng.uniform(-np.pi, np.pi, (law.volume, 3))
        report = law.leading(arbitrary, arbitrary_links)
        Z = float(np.trace(law.grams(arbitrary), axis1=1, axis2=2).sum())
        b = abs(law.onsite) + 2 * sum(float(np.linalg.norm(scalar)) for _, scalar in law.paths)
        hH = law.onebody(arbitrary, arbitrary_links)
        hL = law.onebody(arbitrary, arbitrary_links, phases)
        actual_matter = -float(np.linalg.eigvalsh(hH)[2 * law.volume:].sum()) + float(np.linalg.eigvalsh(hL)[:2 * law.volume].sum())
        lower_matter = -4 * b * np.sqrt(2 * law.volume * Z)
        if actual_matter < lower_matter - 1e-8:
            raise AssertionError('Retained whole-half original H/L CAR lower bound failed')
        local_g = np.array([expm(1j * np.einsum('a,aij->ij', rng.normal(size=15) * .2, law.generators)) for _ in range(law.volume)])
        moved_fields = local_g[:, None] @ arbitrary @ local_g[:, None].conj().swapaxes(-1, -2)
        moved_links = np.array([[local_g[x] @ arbitrary_links[x, a] @ local_g[law.target[x, a]].conj().T
                                for a in range(3)] for x in range(law.volume)])
        moved = law.leading(moved_fields, moved_links)
        check(abs(moved['actual_leading_energy'] - report['actual_leading_energy']),
              'all-site colour covariance of changed law', 2e-8)
        full_g = np.zeros_like(hH)
        for x in range(law.volume):
            full_g[4*x:4*x+4, 4*x:4*x+4] = local_g[x]
        check(float(np.linalg.norm(law.onebody(moved_fields, moved_links, phases) - full_g @ hL @ full_g.conj().T)),
              'actual nonuniform colour/EM native onebody covariance')
        controls.append(dict(scale=scale, **report, retained_original_matter_ground=actual_matter,
            retained_original_matter_lower=lower_matter, local_gauge_covariance_error=abs(moved['actual_leading_energy'] - report['actual_leading_energy'])))
    adverse = np.zeros_like(fields)
    adverse[:, 0] = np.diag([5., 1., 1., -7.])
    adverse_report = law.leading(adverse, links)
    if adverse_report['filling']['forced_block_error'] <= 1e-5:
        raise AssertionError('Aggregate selector occupation-transfer adverse control failed')
    torons = []
    for angles in ([np.pi, 0., 0.], [np.pi / 2, np.pi, 3 * np.pi / 2]):
        result = law.leading(fields, law.toron_links(angles))
        check(abs(result['actual_leading_energy'] - attainment['actual_leading_energy']), 'retained exact centre toron degeneracy')
        torons.append(dict(angles=angles, **result))
    noncentral_links = links.copy()
    seam = np.diag(np.exp(1j * np.array([.7, -.7, .2, -.2])))
    for x, site in enumerate(law.sites):
        if site[0] == size - 1:
            noncentral_links[x, 0] = seam
    noncentral = law.leading(fields, noncentral_links)
    if noncentral['full_alignment_W'] <= 1e-3:
        raise AssertionError('Irreducible-field noncentral holonomy exclusion failed')
    check(noncentral['colour_plaquette_P'], 'flat noncentral holonomy has zero plaquette debt')
    centre_flux = links.copy()
    centre_flux[0, 0] = -np.eye(4)
    flux = law.leading(fields, centre_flux)
    check(flux['full_alignment_W'], 'centre flux commutes with full field')
    if flux['colour_plaquette_P'] <= 1:
        raise AssertionError('Positive colour plaquette distinguishes centre flux')
    constants = trial_constants(law)
    paired = paired_section_controls(law)
    current = retained_light_current_controls(law)
    quantum = [quantum_bound(law, constants, float(value)) for value in Lambdas]
    # Actual packet-support points check the global Lipschitz leading bound;
    # wrong removal of a local selector block cannot pass the aggregate control.
    packet_checks = []
    for fraction in (1., .4, .1):
        r = constants['chart_radius'] * fraction
        c = rng.normal(size=(law.volume, 60))
        c *= (r / np.linalg.norm(c, axis=1))[:, None]
        perturbed = fields + law.fields(c)
        z = rng.normal(size=(law.volume, 3, 15))
        z *= (r / np.linalg.norm(z, axis=-1))[..., None]
        us = np.array([[expm(1j * np.einsum('a,aij->ij', z[x, a], law.generators)) for a in range(3)] for x in range(law.volume)])
        support = law.leading(perturbed, us)
        energy = support['actual_leading_energy'] + support['filling']['forced_block_error']
        upper = law.volume * law.minimum['value'] + constants['L_leading_Lipschitz'] * r
        if energy > upper + 1e-8:
            raise AssertionError('Actual full-coordinate physical packet potential bound failed')
        packet_checks.append(dict(radius=r, actual_paired_trial_leading_energy=energy, packet_uniform_upper=upper))
    if source_hashes() != identities:
        raise RuntimeError('Quantum supplier/proof changed during actual discriminating run')
    return dict(schema=1, source_sha256=identities, source_bindings_unchanged=True,
        governing_plan=PLAN, assignment_plan_revision=PLAN_REVISION, template_revision=TEMPLATE_REVISION,
        bare_original=BARE, lattice_size=size, volume=law.volume,
        changed_law=dict(stiffness=STIFFNESS, alignment=law.alignment,
            colour_plaquette=law.colour_plaquette, selector_sample_grid=3,
            selector_pairs=law.flavours, selector_normalization=1 / (2 * law.flavours),
            selector_occupation=4 * law.flavours * law.volume,
            selector_single_particle_modes=8 * law.flavours * law.volume,
            original_heavy_occupation=2 * law.volume, original_light_occupation=2 * law.volume,
            field_kinetic='sqrtLambda/2 p_C^2', original_heavy_scale=1., original_light_scale=1.,
            leading_scale='Lambda on local quartic, on-site selector CAR, alignment and colour plaquette',
            colour_electric=constants['ue4'], EM_electric=constants['ue1'], EM_magnetic=constants['magnetic'],
            selector_EM_charge=0, selector_colour='paired fundamental and antifundamental',
            domain=f'R^{60*law.volume} x SU4^{3*law.volume} x U1^{3*law.volume}, full aggregate CAR sectors, all primitive Gauss'),
        selected_Gram=law.gram.tolist(), source_richness=rank,
        exact_rank10_witness=minimal_rank_witness(law.band),
        full_Gram_gradient_norm=float(np.linalg.norm(law.gradient)),
        residual_energy_error_formula=float(np.linalg.norm(law.gradient)**2 / (2 * STIFFNESS)),
        residual_status='floating evaluation, not interval certification',
        native_periodic_source_error=native_error, original_failed_route=failed_original,
        actual_changed_law_attainment=attainment, full_nonuniform_controls=controls,
        adverse_global_filling=adverse_report, allowed_centre_torons=torons,
        excluded_noncentral_holonomy=noncentral, penalized_centre_flux=flux,
        physical_paired_CAR_section=paired, physical_packet_constants=constants,
        retained_actual_light_current=current,
        packet_support_controls=packet_checks, full_quantum_ground_bounds=quantum,
        frozen_emergent_carrier=dict(mass=law.band.m * np.sqrt(law.gram[0, 0]),
            Wilson_r=law.band.r * np.sqrt(law.gram[0, 0]),
            velocity=law.band.v * np.sqrt(law.gram[1, 1]),
            statement='source light band on limiting covariantly uniform Clifford field, central colour torons and dynamical U1 retained'),
        theorem='Exact full-domain operator and physical variational bound selects uniform full-rank Clifford field modulo local SU4 and unfixed centre torons at fixed finite N.',
        all_checks_pass=True,
        limits=['Explicit new on-site selector flavours; not the original itinerant-heavy quantum law.',
            'Formal exact-ground concentration, not an observed numerical full quantum eigenstate.',
            'Bounds use exact unique Gstar in proof; receipt constants are non-interval floating values.',
            'Fixed finite-volume large-Lambda limit; no uniform thermodynamic concentration or phase/mixing theorem.',
            'No physical spin, charged lower gap, isolated electron, completed photon/electron interaction or Lorentz closure.',
            'No toron fixing or U1 normal/Fock cuts; original whole-heavy/light and compact photon remain actual operators.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=3)
    parser.add_argument('--Lambda', type=float, nargs='+', default=[1e24, 1e36, 1e48, 1e60])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(size=args.size, Lambdas=args.Lambda)
    payload = json.dumps(result, indent=2, allow_nan=False)
    args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Full-SU4 scalar transport and globally Cartesian covariant frame cloud.

Haar grids and orientation-node integrals are diagnostics, not vacuum samples.
The analytic energy/domain bounds use the original slow Cartesian law.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import expm

from original_centre_charge import CENTRES, C_F, potential, check
from original_charged_principal import original_frame
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis, SELECTED
from bulk_flat_holonomy_bound import BARE
from bulk_quantum_selection import BulkQuantumLaw
from original_normal_gap_bound import peierls_derivatives
from local_orbital_stability import scalar_paths

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
BASIS = matrix_unit_basis()
GENERATORS = BASIS / math.sqrt(8)


def scalar_jet(matrices: np.ndarray, eta: float, generators: np.ndarray = GENERATORS,
               charge: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """s_q, its entire 15-vector left derivative, and the same single Casimir Laplacian."""
    if eta <= 0 or charge not in range(4):
        raise ValueError('positive eta and an original four-centre character required')
    trace = np.trace(matrices, axis1=-2, axis2=-1)
    r = (trace[..., None] * CENTRES.conj()).real
    logits = eta*r
    seeds = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p = seeds / seeds.sum(axis=-1, keepdims=True)
    phases = CENTRES**charge
    value = np.sum(p*phases, axis=-1)
    t = np.einsum('aij,...ji->...a', generators, matrices)
    dr = (1j*t[..., None, :]*CENTRES.conj()[:, None]).real
    mean_dr = np.sum(p[..., :, None]*dr, axis=-2)
    gradient = eta*np.sum(p[..., :, None]*(phases-value[..., None])[..., :, None]*dr, axis=-2)
    log_seed_laplacian = -eta*C_F*r + eta**2*np.sum(dr**2, axis=-1)
    laplacian = np.sum(p*(phases-value[..., None])*log_seed_laplacian, axis=-1)
    laplacian -= 2*eta*np.sum(mean_dr*gradient, axis=-1)
    return value, gradient, laplacian


def centre_scalar(matrix: np.ndarray, eta: float,
                  generators: np.ndarray = GENERATORS) -> tuple[complex, np.ndarray]:
    value, gradient, _ = scalar_jet(matrix, eta, generators)
    return complex(value), gradient


def centre_distance(matrix: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(matrix[None]-CENTRES[:, None, None]*np.eye(4), axis=(1, 2))**2))


def frame_constants(frame: np.ndarray) -> dict[str, Any]:
    """Cartesian orbit metric HS/4, not a declared rotor inertia."""
    tangents = 1j*(GENERATORS[:, None]@frame[None]-frame[None]@GENERATORS[:, None])
    metric = np.einsum('amij,bmij->ab', tangents.conj(), tangents).real/4
    eig = np.linalg.eigvalsh(metric)
    if eig[0] <= 0:
        raise ValueError('Original full-rank irreducible Clifford frame required')
    return dict(Cartesian_orbit_metric=metric.tolist(), inertia_eigenvalues=eig.tolist(),
                ell_min=8*float(eig[0]), ell_max=8*float(eig[-1]),
                Cartesian_frame_squared=float(np.linalg.norm(frame)**2/4),
                minimal_one_endpoint_fundamental_gradient_squared=float(np.trace(np.linalg.inv(metric))/8),
                inertia_law='actual g/chi, chi=Lambda^-1/2; no additional endpoint rotor')


def haar_nodes(samples: int, sites: int, seed: int) -> np.ndarray:
    """Independent exact-distribution Haar SU4 draws, not a finite-irrep law."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(samples, sites, 4, 4))+1j*rng.normal(size=(samples, sites, 4, 4))
    unitary, triangular = np.linalg.qr(raw)
    phase = np.diagonal(triangular, axis1=-2, axis2=-1)
    unitary *= (phase/abs(phase))[..., None, :]
    unitary[..., :, 0] /= np.linalg.det(unitary)[..., None]
    return unitary


def orientation_cloud(fields_on_path: np.ndarray, oriented_links: np.ndarray,
                      body_mark: np.ndarray, eta: float, alpha: float,
                      nodes: np.ndarray, templates: np.ndarray) -> dict[str, np.ndarray]:
    """Normalized whole-Haar orientation-cloud diagnostic and exact node derivatives.

    nodes=(samples,L+1,4,4), fields/templates=(L+1,4,4,4), links=(L,4,4).
    Returns q/2 convention, ALL60 Cartesian derivatives per site and ALL15 per
    edge. The continuum definition is the Haar integral, not this finite rule.
    A correlated product-Haar rule is normalized by its joint posterior weight.
    """
    count, sites = nodes.shape[:2]
    if (alpha <= 0 or len(oriented_links) != sites-1 or len(fields_on_path) != sites
            or templates.shape != fields_on_path.shape or count < 1):
        raise ValueError('positive cloud alpha and full, consistent original path data required')
    conjugate = nodes.conj().swapaxes(-1, -2)
    rotated = nodes[:, :, None]@templates[None]@conjugate[:, :, None]
    coordinates = np.einsum('aij,sxmji->sxma', BASIS, rotated).real/4
    logits = alpha*np.einsum('xmij,sxmij->s', fields_on_path.conj(), rotated).real/2
    weights = np.exp(logits-logits.max())
    weights /= weights.sum()
    relative = conjugate[:, :-1]@oriented_links[None]@nodes[:, 1:]
    values, _, _ = scalar_jet(relative, eta)
    product = np.prod(values, axis=1)
    endpoint = nodes[:, 0]@body_mark@conjugate[:, -1]/2
    integrand = product[:, None, None]*endpoint
    mean = np.einsum('s,sij->ij', weights, integrand)
    c = coordinates.reshape(count, sites, 60)
    mc = np.einsum('s,sxa->xa', weights, c)
    centered = c-mc[None]
    gradient = 2*alpha*np.einsum('s,sij,sxa->xaij', weights, integrand, centered)
    variance = np.sum(centered**2, axis=-1)
    laplacian_sites = 4*alpha**2*(np.einsum('s,sij,sx->xij', weights, integrand, variance)
                         -mean[None]*np.einsum('s,sx->x', weights, variance)[:, None, None])
    link_gradient = np.zeros((sites-1, 15, 4, 4), complex)
    link_laplacian = np.zeros((sites-1, 4, 4), complex)
    for edge in range(sites-1):
        # Actual link variation exp(i t T_a)U: its relative generator is g_x^dagger T_a g_x.
        body_generators = conjugate[:, edge, None]@GENERATORS[None]@nodes[:, edge, None]
        r = (np.trace(relative[:, edge], axis1=-2, axis2=-1)[:, None]*CENTRES.conj()).real
        seeds = np.exp(eta*r-np.max(eta*r, axis=1, keepdims=True))
        p = seeds/seeds.sum(axis=1, keepdims=True)
        traces = np.einsum('saij,sji->sa', body_generators, relative[:, edge])
        dr = (1j*traces[:, None, :]*CENTRES.conj()[None, :, None]).real
        s = values[:, edge]
        ds = eta*np.sum(p[:, :, None]*(CENTRES-s[:, None])[:, :, None]*dr, axis=1)
        mean_dr = np.sum(p[:, :, None]*dr, axis=1)
        ak = -eta*C_F*r+eta**2*np.sum(dr**2, axis=-1)
        dd = np.sum(p*(CENTRES-s[:, None])*ak, axis=1)-2*eta*np.sum(mean_dr*ds, axis=1)
        others = np.prod(np.delete(values, edge, axis=1), axis=1)
        link_gradient[edge] = np.einsum('s,sa,sij->aij', weights*others, ds, endpoint)
        link_laplacian[edge] = np.einsum('s,s,sij->ij', weights*others, dd, endpoint)
    return dict(mean=mean, Cartesian_gradient=gradient, Cartesian_Laplacian=laplacian_sites.sum(axis=0),
                Cartesian_site_Laplacian=laplacian_sites, link_gradient=link_gradient,
                link_Laplacian=link_laplacian, posterior_weights=weights)


def source_energy_majorant(length: int, eta: float, alpha: float, frame_squared: float,
                           strong: float, bad_probability: float, norm_lower: float,
                           normal_momentum_squared: float, link_momentum_squared: float,
                           flux_coefficient: float = 1.) -> dict[str, float]:
    """Caller supplies ACTUAL scalar guard probability/normal P moment, in original units."""
    if min(length, eta, alpha, frame_squared, strong, norm_lower) <= 0 or not 0 <= bad_probability <= 1:
        raise ValueError('positive source/normal data and genuine probability required')
    if min(normal_momentum_squared, link_momentum_squared, flux_coefficient) < 0:
        raise ValueError('nonnegative actual normal and flux graph inputs required')
    chi = strong**-.5
    field_d2 = 16*alpha**2*(length+1)*frame_squared
    link_d2 = length*(1152*eta**2*math.exp(-6*eta)+8*eta**2*bad_probability)
    field_lap = 32*alpha**2*(length+1)*frame_squared
    link_lap = length*(15*eta+12*eta**2)
    return dict(Cartesian_gradient_squared_upper=field_d2, full15_link_gradient_squared_upper=link_d2,
                extra_source_Dirichlet_energy_upper=(chi*field_d2/2+.3*chi*link_d2)/norm_lower**2,
                Cartesian_source_commutator_upper=chi*(math.sqrt(field_d2*normal_momentum_squared)+field_lap/2)/norm_lower,
                full15_link_Laplacian_upper=link_lap,
                colour_source_commutator_upper=.3*chi*(2*math.sqrt(link_d2*link_momentum_squared)+link_lap)/norm_lower,
                boundary_colour_flux_upper=flux_coefficient*math.sqrt(link_d2)/norm_lower,
                ratio_normalization_lower=norm_lower,
                # This is a source-only form/graph budget; the common vacuum form is not erased.
                chi=chi)


def whole_haar_consumer(eta: float, grid: int, scales_sum: float) -> dict[str, Any]:
    angles = 2*np.pi*np.arange(grid)/grid-np.pi
    b, c = np.meshgrid(angles, angles, indexing='ij')
    mass = norm = kinetic = well = neutral_norm = 0.
    sectors = np.zeros(4)
    for a in angles:
        eigen = np.stack((np.full_like(b, np.exp(1j*a), dtype=complex), np.exp(1j*b),
                          np.exp(1j*c), np.exp(-1j*(a+b+c))), axis=-1)
        matrices = np.zeros((*eigen.shape[:-1], 4, 4), complex)
        matrices[..., np.arange(4), np.arange(4)] = eigen
        vandermonde = np.ones_like(b)
        for i, j in itertools.combinations(range(4), 2):
            vandermonde *= abs(eigen[..., i]-eigen[..., j])**2
        weight = vandermonde/(24*grid**3)
        mass += float(weight.sum())
        trace = eigen.sum(axis=-1)
        seeds = np.exp(eta*((trace[..., None]*CENTRES.conj()).real-4))
        density = seeds.sum(axis=-1)**2
        neutral_norm += float(np.sum(weight*density))
        value, gradient, _ = scalar_jet(matrices, eta)
        norm += float(np.sum(weight*density*abs(value)**2))
        kinetic += float(np.sum(weight*density*np.sum(abs(gradient)**2, axis=-1)))
        wbar = 8/15*scales_sum*(16-abs(trace)**2)
        well += float(np.sum(weight*density*wbar))
        for q in range(4):
            packet = np.sum(seeds*CENTRES**q, axis=-1)
            sectors[q] += float(np.sum(weight*abs(packet)**2))
    check(abs(mass-1), 'native entire Weyl Haar measure', 2e-10)
    if min(sectors) <= 0:
        raise AssertionError('all four centre packets must retain nonzero norm')
    return dict(grid=grid, eta=eta, Haar_mass=mass, all4_sector_norms=sectors.tolist(),
                ratio_norm_squared=norm/neutral_norm, ratio_full15_derivative_squared=kinetic/neutral_norm,
                original_frozen_frame_W=well/neutral_norm,
                scope='entire-Haar quadrature of an explicit q0 neutral trial, NOT original scalar ground or interval bound')


def scalar_controls(frame: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(222105)
    u = haar_nodes(1, 1, 222106)[0, 0]
    epsilon = 2e-5
    errors, laps, sectors, inequalities = [], [], [], []
    constants = frame_constants(frame)
    for matrix in (u, expm(.2j*GENERATORS[3]), np.diag(np.exp(1j*np.array([.7, -.1, .2, -.8])))):
        value, derivative, lap = scalar_jet(matrix, 2.)
        fd, second = [], []
        for t in GENERATORS:
            vp = scalar_jet(expm(1j*epsilon*t)@matrix, 2.)[0]
            vm = scalar_jet(expm(-1j*epsilon*t)@matrix, 2.)[0]
            fd.append((vp-vm)/(2*epsilon))
            h2 = .001
            values = [scalar_jet(expm(1j*k*h2*t)@matrix, 2.)[0] for k in (-2, -1, 1, 2)]
            second.append((-values[3]+16*values[2]-30*value+16*values[1]-values[0])/(12*h2**2))
        errors.append(float(np.linalg.norm(derivative-np.asarray(fd))))
        laps.append(abs(complex(lap)-sum(second)))
        sectors.append(max(abs(complex(scalar_jet(z*matrix, 2.)[0])-z*complex(value)) for z in CENTRES))
        d = centre_distance(matrix)
        w = potential(frame, frame, matrix)
        inequalities.append(max(0., constants['ell_min']*d/18-w))
        for scale in (.001, .1):
            nu = scale*np.einsum('xma,aij->xmij', rng.normal(size=(2, 4, 15)), BASIS)
            actual = potential(frame+nu[0], frame+nu[1], matrix)
            lower = constants['ell_min']*d/36-2*float(np.linalg.norm(nu)**2)
            inequalities.append(max(0., lower-actual))
    check(max(errors), 'all15 source quotient derivatives', 3e-8)
    check(max(laps), 'all15 source quotient Laplacian', 2e-5)
    check(max(sectors), 'all4 centre descent', 1e-12)
    check(max(inequalities), 'original-law perturbed full Cartesian centre barrier', 2e-10)
    missing = abs(complex(scalar_jet(1j*u, 2.)[0])-1j)
    if missing < .1:
        raise AssertionError('omitting three centre seeds did not discriminate centre descent')
    guard_matrix = np.diag(np.exp(1j*np.array([1., 1., 1., -3.])*np.pi/4))
    _, guard_gradient, _ = scalar_jet(guard_matrix, 4.)
    unpaid_guard = float(np.linalg.norm(guard_gradient)**2)-1152*16*math.exp(-24)
    if unpaid_guard <= 1e-3:
        raise AssertionError('wrong near-centre derivative outside guard did not discriminate')
    return dict(full15_derivative_error=max(errors), full15_Laplacian_error=max(laps),
                all4_character_error=max(sectors), original_coercivity_error=max(inequalities),
                missing_three_centres_descent_defect=float(missing),
                unpaid_guard_derivative_excess=unpaid_guard,
                global_derivative_squared_bound=8*2.**2,
                normal_cost_retained=True)


def cloud_consumer(frame: np.ndarray, samples: int) -> dict[str, Any]:
    templates = np.tile(frame, (2, 1, 1, 1))
    fields = templates.copy()
    fields[0, 2] += .017*BASIS[7]
    fields[1, 0] -= .011*BASIS[2]
    links = np.array([expm(.21j*GENERATORS[4])])
    nodes = haar_nodes(samples, 2, 222107)
    alpha, eta, epsilon = 2., 3., 2e-5
    q = SELECTED[2]@SELECTED[3]
    result = orientation_cloud(fields, links, q, eta, alpha, nodes, templates)
    fd = np.zeros_like(result['Cartesian_gradient'])
    lap = np.zeros((4, 4), complex)
    for site in range(2):
        for component in range(4):
            for axis, basis in enumerate(BASIS):
                plus, minus = fields.copy(), fields.copy()
                plus[site, component] += epsilon*basis
                minus[site, component] -= epsilon*basis
                vp = orientation_cloud(plus, links, q, eta, alpha, nodes, templates)['mean']
                vm = orientation_cloud(minus, links, q, eta, alpha, nodes, templates)['mean']
                fd[site, 15*component+axis] = (vp-vm)/(2*epsilon)
                h2 = .001
                second_values = []
                for k in (-2, -1, 1, 2):
                    displaced = fields.copy()
                    displaced[site, component] += k*h2*basis
                    second_values.append(orientation_cloud(displaced, links, q, eta, alpha, nodes, templates)['mean'])
                lap += (-second_values[3]+16*second_values[2]-30*result['mean']+
                        16*second_values[1]-second_values[0])/(12*h2**2)
    field_error = float(np.linalg.norm(fd-result['Cartesian_gradient']))
    lap_error = float(np.linalg.norm(lap-result['Cartesian_Laplacian']))
    link_fd = []
    for t in GENERATORS:
        vp = orientation_cloud(fields, np.array([expm(1j*epsilon*t)@links[0]]), q, eta, alpha, nodes, templates)['mean']
        vm = orientation_cloud(fields, np.array([expm(-1j*epsilon*t)@links[0]]), q, eta, alpha, nodes, templates)['mean']
        link_fd.append((vp-vm)/(2*epsilon))
    link_error = float(np.linalg.norm(np.asarray(link_fd)-result['link_gradient'][0]))
    check(field_error, 'actual120 Cartesian cloud derivatives', 2e-8)
    check(lap_error, 'actual120 Cartesian cloud Laplacian', 4e-5)
    check(link_error, 'actual15 original-link cloud derivatives', 2e-8)
    h = np.array([expm(.17j*GENERATORS[1]), expm(-.23j*GENERATORS[10])])
    moved = h[:, None]@fields@h[:, None].conj().swapaxes(-1, -2)
    moved_links = np.array([h[0]@links[0]@h[1].conj().T])
    moved_nodes = h[None]@nodes
    gauge = orientation_cloud(moved, moved_links, q, eta, alpha, moved_nodes, templates)['mean']
    covariance = float(np.linalg.norm(gauge-h[0]@result['mean']@h[1].conj().T))
    check(covariance, 'whole-integrand local gauge change of variables', 2e-12)
    centred_nodes = nodes.copy()
    centred_nodes[:, 0] *= 1j
    descent = float(np.linalg.norm(orientation_cloud(fields, links, q, eta, alpha, centred_nodes, templates)['mean']-result['mean']))
    check(descent, 'orientation cloud independent frame centre sheet', 2e-12)
    zero = orientation_cloud(np.zeros_like(fields), links, q, eta, alpha, nodes, templates)
    field_form = float(np.linalg.norm(result['Cartesian_gradient'])**2)
    link_form = float(np.linalg.norm(result['link_gradient'])**2)
    if np.linalg.norm(result['Cartesian_gradient'][-1]) < 1e-5:
        raise AssertionError('full reference endpoint derivative adverse control did not discriminate')
    constants = frame_constants(frame)
    bound = 16*alpha**2*2*constants['Cartesian_frame_squared']
    check(max(0., field_form-bound), 'actual Cartesian posterior derivative bound')
    return dict(samples=samples, path_sites=2, Cartesian_axes=120, link_axes=15,
                alpha=alpha, eta=eta, complete_bar4_reference=True,
                source_norm_squared=float(np.linalg.norm(result['mean'])**2),
                Cartesian_gradient_squared=field_form, all15_link_gradient_squared=link_form,
                original_slow_extra_energy_per_chi=.5*field_form+.3*link_form,
                Cartesian_derivative_error=field_error, Cartesian_Laplacian_error=lap_error,
                original_link_derivative_error=link_error, local_gauge_change_of_variables_error=covariance,
                centre_sheet_descent_error=descent,
                omitted_reference_endpoint_derivative_defect=float(np.linalg.norm(result['Cartesian_gradient'][-1])),
                degenerate_Phi_zero_source_norm=float(np.linalg.norm(zero['mean'])),
                degenerate_Phi_zero_gradient_norm=float(np.linalg.norm(zero['Cartesian_gradient'])),
                normalized_nodes_are_not_a_continuum_norm_certificate=True,
                scope='full-Cartesian/full15 finite product-Haar importance quadrature; exact analytic continuum bounds are separate')


def native_current_consumer(band: RotationBand, frame: np.ndarray) -> dict[str, Any]:
    law = BulkQuantumLaw(3, band=band)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.pi/2*np.array([1, 2, 3]))
    rng = np.random.default_rng(222108)
    phases = rng.normal(scale=.3, size=(law.volume, 3))
    direction = rng.normal(size=(law.volume, 3))
    current, contact = peierls_derivatives(law, fields, links, phases, direction)
    epsilon = 2e-5
    h = law.onebody(fields, links, phases)
    plus = law.onebody(fields, links, phases+epsilon*direction)
    minus = law.onebody(fields, links, phases-epsilon*direction)
    first_error = float(np.linalg.norm((plus-minus)/(2*epsilon)-current))
    second_error = float(np.linalg.norm((plus-2*h+minus)/epsilon**2-contact))
    check(first_error, 'unchanged original same-current first derivative', 1e-7)
    check(second_error, 'unchanged original same-current contact', 2e-5)
    # Real Coulomb source, same fixed R, Laplacian zero-mode quotient retained.
    incidence = np.zeros((law.volume, 3*law.volume))
    for x in range(law.volume):
        for axis in range(3):
            incidence[x, 3*x+axis] = 1
            incidence[law.target[x, axis], 3*x+axis] = -1
    green = np.linalg.pinv(incidence@incidence.T, hermitian=True)
    charge = np.zeros(law.volume); charge[0], charge[-1] = 1., -1.
    word = incidence.T@green@charge
    check(float(np.linalg.norm(incidence@word-charge)), 'same Coulomb field and complete reference Gauss')
    harmonic = np.array([.11, -.07, .09])
    displacement = (law.sites[0]-law.sites[-1]).astype(float)
    cloud_phase = np.exp(1j*np.dot(word, phases.reshape(-1))-1j*np.dot(displacement, harmonic))
    if abs(np.linalg.norm(contact)) < 1e-3:
        raise AssertionError('deleted actual contact adverse input did not discriminate')
    onsite, paths = scalar_paths(band)
    return dict(original_modes=4*law.volume, whole_heavy_number=2*law.volume,
                current_error=first_error, contact_error=second_error,
                deleted_contact_defect=float(np.linalg.norm(contact)),
                Coulomb_word_squared=float(np.dot(word, word)),
                Coulomb_dressing_energy_coefficient_per_UE=float(np.dot(word, word)),
                harmonic_electric_coefficient_per_UE=float(np.dot(displacement, displacement))/law.volume,
                combined_phase_modulus=float(abs(cloud_phase)),
                noncompact_energy_units='HC=UE rho^T G rho; Hh=(UE/V) p_a^2, NOT UE/2',
                actual_onsite=float(onsite), actual_path_coefficient_norms=[float(np.linalg.norm(c)) for _, c in paths],
                theta_in_true_SU4_links_only=True, harmonic_and_Coulomb_not_added_as_colour_holonomy=True,
                no_numeric_coupled_neutral_ground=True)


def run(spacing: float, grid: int, samples: int, haar_grids: list[int]) -> dict[str, Any]:
    band = RotationBand(**{**BARE, 'a': spacing})
    hashes = source_bindings()
    rows = []
    frame = None
    for centre in itertools.product(range(4), repeat=3):
        selected, phi = original_frame(band, grid, np.pi/2*np.array(centre), True)
        constants = frame_constants(phi)
        rows.append(dict(centre=list(centre), native_Gram=np.asarray(selected['G']).tolist(),
                         original_Gram_gradient=selected['gradient_residual'],
                         ell_min=constants['ell_min'], Cartesian_inertia_min=constants['inertia_eigenvalues'][0],
                         original_whole_heavy_energy=float(selected['value'])*grid**3))
        if centre == (0, 0, 0):
            frame = phi
    if frame is None:
        raise AssertionError('identity member must be retained alongside all63 other holonomies')
    constants = frame_constants(frame)
    scales_sum = constants['Cartesian_frame_squared']
    result = dict(plan=PLAN, template_revision=TEMPLATE_REVISION, a=spacing, native_frame_grid=grid,
                  original_law=dict(Cartesian_kinetic='.5 Lambda^-.5 sum_all60V P^2',
                                    colour_electric='.3 Lambda^-.5 single full15 Casimir per edge',
                                    alignment='.2 Lambda^1.25 W/8', colour_plaquette='.7 Lambda^1.25',
                                    whole_heavy='original endpoint-average NH=2V, multiplier Lambda',
                                    extra_local_selector_used=False),
                  all64_spatial_holonomies=rows, frame=constants,
                  scalar_source=scalar_controls(frame),
                  whole_Haar=[whole_haar_consumer(2., n, scales_sum) for n in haar_grids],
                  full_Cartesian_cloud=cloud_consumer(frame, samples),
                  same_current_Coulomb_harmonic=native_current_consumer(band, frame),
                  sequence=dict(eta='log(6 L/epsilon)/3 or larger',
                                orientation_radius='r=epsilon/8 <=1/16',
                                alpha='finite smallest doubled alpha>=1 with posterior tail bound <=epsilon^2/(16(L+1))',
                                Lambda='max(original scalar onset, explicit normal/centre/energy/commutator inequalities in proof); every coefficient finite at fixed N>=N0(a)',
                                actual_scalar_constants_numeric=False,
                                N0_numeric_certified=False),
                  scientific_scope='original-law analytic domain/coercivity/diagonal source bounds plus native finite diagnostics; no local physical half-spin or neutral scalar vacuum computed',
                  source_sha256=hashes)
    result['source_bindings_unchanged_during_run'] = hashes == source_bindings()
    if not result['source_bindings_unchanged_during_run']:
        raise RuntimeError('executing supplier/source bytes changed during run')
    return result


def source_bindings() -> dict[str, str]:
    paths = {ROOT/'noncompact_centre_source_energy.py', ROOT/'noncompact_centre_source_energy.md',
             ROOT/'original_slow_kinetic_bo.md', ROOT/'original_centre_charge.md',
             ROOT/'original_charged_principal.md', ROOT/'noncompact_marked_metric.md',
             ROOT/'noncompact_microscopic_charge_window.md',
             ROOT/'noncompact_charged_packet_transport.md', ROOT/'noncompact_coulomb_carrier.md'}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.parent == ROOT and path.suffix == '.py':
                paths.add(path)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=float, default=1.)
    parser.add_argument('--frame-grid', type=int, default=5)
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--haar-grids', type=int, nargs='+', default=[24, 40])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.a, args.frame_grid, args.samples, args.haar_grids)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload+'\n')
    print(payload)


if __name__ == '__main__':
    main()

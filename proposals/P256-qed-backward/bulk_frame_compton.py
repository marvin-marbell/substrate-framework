#!/usr/bin/env python3
"""Actual finite bulk frame, light Peierls sea pole and primitive Pauli response.

This is a finite-background Gaussian/quadratic-sea calculation. It does not
supply a charged Gauss ground, normal confinement or a compact photon phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from full_band_gauss_section import FullBandGaussSection, source_hashes as section_hashes
from local_orbital_stability import scalar_paths
from photon_matched_compton import forward_at_pole
from photon_matching import match_photon
from rotation_band import RotationBand
from sea_selection import SELECTED, coefficients, matrix_unit_basis
from vacuum_limit import momentum_response

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = 'd66d469ecc0e4e487bb104dc866f7fd679ae0e868c1a5d3abf38c1ede747183b'
FRAME_TOLERANCE = 5e-6


def require(error, name, tolerance=FRAME_TOLERANCE):
    if not np.isfinite(error) or error > tolerance:
        raise ValueError(f'{name}: residual {error} exceeds {tolerance}')


def band_parameters(band):
    return dict(a=band.a, m=band.m, r=band.r, v=band.v, e=band.e, improved=band.improved)


def source_hashes():
    names = ('bulk_frame_compton.py', 'bulk_frame_compton.md', 'periodic_bulk_frame.py',
             'periodic_bulk_frame.md', 'vacuum_limit.py', 'photon_matching.py',
             'photon_matched_compton.py', 'photon_matching.md', 'vacuum_limit.md')
    return {**section_hashes(), **{
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}


def read_bulk(path: Path, branch: str) -> tuple[RotationBand, np.ndarray, dict[str, Any]]:
    payload = path.read_bytes()
    receipt = json.loads(payload)
    identities = receipt['source_sha256']
    if not identities or any(Path(name).name != name for name in identities):
        raise ValueError('Bulk source bindings must be nonempty local owner filenames')
    mismatches = [name for name, expected in identities.items()
                  if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected]
    if mismatches:
        raise ValueError(f'Bulk input is not bound to the current supplier bytes: {mismatches}')
    if not receipt['source_bindings_unchanged_during_run'] or not receipt['input_receipt_unchanged_during_run']:
        raise ValueError('Bulk receipt did not retain immutable supplier/input bindings')
    rows = [(i, row) for i, row in enumerate(receipt['same_law_seed_branches']) if row['name'] == branch]
    if len(rows) != 1:
        raise ValueError(f'Exactly one explicitly named bulk branch required: {branch}')
    index, row = rows[0]
    candidate = row['validation_stationary_candidate']
    if not (candidate['actual_finite_mesh_stationary']
            and candidate['finite_numerical_local_minimum_modulo_orbit']
            and candidate['orbit_dimension'] == 15
            and candidate['common_complex_commutant_dimension'] == 1):
        raise ValueError('Selected finite candidate must be a stationary irreducible rank15 local minimum')
    values = np.asarray(candidate['coordinate_values'], dtype=float)
    if values.shape != (4, 15) or not np.all(np.isfinite(values)):
        raise ValueError('Actual finite all60 Cartesian input coordinates required')
    bare = receipt['bare_coefficients']
    band = RotationBand(**{key: bare[key] for key in ('a', 'm', 'r', 'v', 'improved')}, e=1.)
    if band.improved:
        raise ValueError('This bulk/current join derives the actual unimproved nearest-path law only')
    seed_binding = receipt['input_receipt_binding']
    seed_path = Path(seed_binding['path'])
    if hashlib.sha256(seed_path.read_bytes()).hexdigest() != seed_binding['sha256']:
        raise ValueError('Original bulk seed input bytes changed')
    frame = np.einsum('ma,aij->mij', values, matrix_unit_basis())
    binding: dict[str, Any] = dict(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest(),
                   branch=branch, coordinate_pointer=f'/same_law_seed_branches/{index}/validation_stationary_candidate/coordinate_values',
                   supplier_source_sha256=identities, original_seed_binding=seed_binding,
                   finite_grid=candidate['grid'], energy_per_cell=candidate['energy_per_cell'],
                   finite_normal_Hessian_min=candidate['normal_Hessian_min'],
                   continuum_energy_quadrature=candidate['quadrature'],
                   role='actual selected finite coordinates; no global-ground or continuum energy ranking')
    return band, frame, binding


def reconstruct_frame(bare, frame):
    gram = np.einsum('mij,nji->mn', frame, frame).real / 4
    amplitudes = np.sqrt(np.diag(gram))
    if np.any(amplitudes <= 0) or not np.all(np.isfinite(amplitudes)):
        raise ValueError('All four actual frame amplitudes must be nonzero')
    gamma = frame / amplitudes[:, None, None]
    clifford = max(float(np.linalg.norm(gamma[i] @ gamma[j] + gamma[j] @ gamma[i]
                     - (2 * np.eye(4) if i == j else 0))) for i in range(4) for j in range(4))
    isotropy = float(np.max(abs(amplitudes[1:] / amplitudes[1:].mean() - 1)))
    require(clifford, 'actual normalized Clifford identities')
    require(isotropy, 'actual spatial amplitude isotropy')
    # vec_F(gamma U-U Gamma)=0. The polar part is reconstructed, not selected
    # from a preinstalled frame or a guessed unitary gauge transformation.
    system = np.concatenate([np.kron(np.eye(4), g) - np.kron(s.T, np.eye(4))
                             for g, s in zip(gamma, SELECTED)])
    _, singular, vh = np.linalg.svd(system)
    intertwiner = vh[-1].conj().reshape(4, 4, order='F')
    left, polar_values, right = np.linalg.svd(intertwiner)
    unitary = left @ right
    error = max(float(np.linalg.norm(g - unitary @ s @ unitary.conj().T))
                for g, s in zip(gamma, SELECTED))
    require(error, 'reconstructed simultaneous unitary intertwiner')
    commutant = np.concatenate([np.kron(np.eye(4), g) - np.kron(g.T, np.eye(4)) for g in gamma])
    commutant_values = np.linalg.svd(commutant, compute_uv=False)
    dimension = int(np.sum(commutant_values < FRAME_TOLERANCE))
    if dimension != 1 or singular[-2] <= .1 or polar_values[-1] <= .1:
        raise ValueError('Actual frame does not supply a unique nonsingular irreducible intertwiner')
    b0, spatial = float(amplitudes[0]), float(amplitudes[1:].mean())
    band = RotationBand(a=bare.a, m=bare.m * b0, r=bare.r * b0,
                        v=bare.v * spatial, e=bare.e, improved=bare.improved)
    generators = np.array([-.5j * gamma[j] @ gamma[k] for j, k in ((2, 3), (3, 1), (1, 2))])
    su2 = max(float(np.linalg.norm(generators[i] @ generators[j] - generators[j] @ generators[i]
                                  - 1j * generators[k])) for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
    casimir = float(np.linalg.norm(sum(s @ s for s in generators) - .75 * np.eye(4)))
    rotation_vector = max(float(np.linalg.norm(generators[i] @ gamma[j+1] - gamma[j+1] @ generators[i]
                                - 1j * gamma[k+1])) for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
    half_angle = float(np.linalg.norm(expm(-2j * np.pi * generators[2]) + np.eye(4)))
    for name, value in (('SU2', su2), ('spin Casimir', casimir), ('vector action', rotation_vector), ('2pi', half_angle)):
        require(value, name, 5 * FRAME_TOLERANCE)
    inertial_inverse = band.v**2 / band.m + band.a * band.r
    report = dict(actual_field_Gram=gram.tolist(), amplitudes=amplitudes.tolist(), temporal_amplitude=b0,
                  spatial_amplitude=spatial, normalized_Clifford_residual=clifford,
                  spatial_isotropy_residual=isotropy, unitary_real=np.real(unitary).tolist(),
                  unitary_imag=np.imag(unitary).tolist(), intertwiner_residual=error,
                  intertwiner_singular_values=singular.tolist(), commutant_dimension=dimension,
                  approximation_tolerance=FRAME_TOLERANCE,
                  reconstruction='fields determine amplitudes and unitary; no imposed Gamma amplitude',
                  effective_band=band_parameters(band), internal_spin=dict(SU2_residual=su2,
                  Casimir_residual=casimir, two_pi_minus_one_residual=half_angle, vector_action_residual=rotation_vector),
                  low_momentum=dict(rest_energy=band.m, linear_matter_speed=band.v,
                  Wilson_mass_quadratic_coefficient=band.a * band.r / 2,
                  energy_quadratic_coefficient=inertial_inverse / 2,
                  inertial_mass=1 / inertial_inverse, relativistic_mass_without_Wilson=band.m / band.v**2,
                  corner_masses=[dict(pi_axes=n, multiplicity=(3 if n in (1, 2) else 1),
                                      mass=band.m + 2 * n * band.r / band.a) for n in (1, 2, 3)]))
    return band, unitary, report


def raw_paths(bare, frame):
    onsite, paths = scalar_paths(bare)
    return onsite * frame[0], [(d, np.einsum('m,mij->ij', scalars, frame)) for d, scalars in paths]


def raw_h(bare, frame, p):
    onsite, paths = raw_paths(bare, frame)
    result = onsite.astype(complex).copy()
    for displacement, coupling in paths:
        phase = np.exp(1j * bare.a * np.dot(p, displacement))
        result += phase * coupling + phase.conjugate() * coupling.conj().T
    return result


def primitive_frame_controls(bare, band, frame, unitary):
    rows, wrong_default, wrong_unscaled, wrong_frame = [], [], [], []
    wrong_current, wrong_contact = [], []
    onsite, paths = raw_paths(bare, frame)
    del onsite
    wrong = frame.copy(); wrong[2] += .07 * matrix_unit_basis()[0]
    try:
        reconstruct_frame(bare, wrong)
    except ValueError as error:
        nonclifford_rejection = str(error)
    else:
        raise ValueError('Non-Clifford field mutation was incorrectly accepted as a spinor frame')
    eps, eps_out = np.array([.31, -.23, .41]), np.array([-.19, .37, .29])
    for pin_angles, pout_angles in (([.19, -.31, .47], [-.23, .17, .53]), ([.63, .29, -.37], [.11, -.41, .71])):
        pin, pout = np.array(pin_angles) / band.a, np.array(pout_angles) / band.a
        midpoint = (pin + pout) / 2
        current = np.zeros((3, 4, 4), dtype=complex)
        contact = np.zeros((4, 4), dtype=complex)
        for d, coupling in paths:
            axis = int(np.flatnonzero(d)[0])
            phase = np.exp(1j * band.a * np.dot(midpoint, d))
            current[axis] += 1j * band.a * (phase * coupling - phase.conjugate() * coupling.conj().T)
            contact -= band.e**2 * band.a**2 * eps[axis] * eps_out[axis] * (
                phase * coupling + phase.conjugate() * coupling.conj().T)
        expected_h = unitary @ band.h(pin) @ unitary.conj().T
        expected_current = unitary @ band.current(pout, pin) @ unitary.conj().T
        expected_contact = unitary @ band.contact(pout, pin, np.r_[0., eps], np.r_[0., eps_out],
                                                 pout-pin, pout-pin) @ unitary.conj().T
        step = 2e-5
        finite_current = (raw_h(bare, frame, midpoint + step * eps)
                          - raw_h(bare, frame, midpoint - step * eps)) / (2 * step)
        finite_contact = (raw_h(bare, frame, midpoint + step * eps)
                          - 2 * raw_h(bare, frame, midpoint)
                          + raw_h(bare, frame, midpoint - step * eps)) / step**2
        contact_same = band.e**2 * sum(-band.a**2 * eps[int(np.flatnonzero(d)[0])]**2 * (
            np.exp(1j * band.a * np.dot(midpoint, d)) * t
            + np.exp(-1j * band.a * np.dot(midpoint, d)) * t.conj().T) for d, t in paths)
        errors = dict(raw_h_intertwiner=float(np.linalg.norm(raw_h(bare, frame, pin)-expected_h)),
                      raw_current_intertwiner=float(np.linalg.norm(current-expected_current)),
                      raw_contact_intertwiner=float(np.linalg.norm(contact-expected_contact)),
                      raw_h_vs_field_coefficients=float(np.linalg.norm(raw_h(bare, frame, pin)
                          - np.einsum('m,mij->ij', coefficients(bare, pin[None])[0], frame))),
                      owner_current_finite_difference=float(np.linalg.norm(finite_current-np.einsum('i,ijk->jk', eps, current))),
                      owner_contact_finite_difference=float(np.linalg.norm(band.e**2 * finite_contact-contact_same)))
        for name, error in errors.items():
            require(error, name, 1e-5)
        rows.append(dict(pin=pin.tolist(), pout=pout.tolist(), errors=errors))
        wrong_default.append(float(np.linalg.norm(raw_h(bare, frame, pin)-bare.h(pin))))
        wrong_unscaled.append(float(np.linalg.norm(raw_h(bare, frame, pin)-unitary @ bare.h(pin) @ unitary.conj().T)))
        wrong_frame.append(float(np.linalg.norm(raw_h(bare, wrong, pin)-expected_h)))
        wrong_current.append(float(np.linalg.norm(current-bare.current(pout, pin))))
        wrong_contact.append(float(np.linalg.norm(contact-bare.contact(
            pout, pin, np.r_[0., eps], np.r_[0., eps_out], pout-pin, pout-pin))))
    if min(*wrong_default, *wrong_unscaled, *wrong_frame, *wrong_current, *wrong_contact) <= 1e-3:
        raise ValueError('Actual nontrivial-momentum controls failed to reject wrong/default frames')
    return dict(nontrivial_momenta=rows, default_frame_errors=wrong_default,
                unscaled_parameters_errors=wrong_unscaled, nonClifford_frame_errors=wrong_frame,
                default_current_errors=wrong_current, default_contact_errors=wrong_contact,
                nonClifford_mutation_rejected=nonclifford_rejection,
                bad_controls_rejected=True, law_owner='local_orbital_stability.scalar_paths(bare) with actual receipt fields')


def actual_sea_pole(length, bare, frame, band, reduced_pole):
    """Independent raw-field channels; root/legs fed to the actual Pauli consumer.

    Parameters of the consumer are the verified field-derived reduction. Raw
    channels retain the unrounded actual matrices, not the normalized Gamma fit.
    """
    onsite, paths = raw_paths(bare, frame)
    grid = np.moveaxis(np.indices((length,) * 3), 0, -1).reshape(-1, 3)
    p = grid * (2 * np.pi / (bare.a * length))
    cells, angle = length**3, reduced_pole['link_momentum_angle']
    polarization = reduced_pole['polarization_axis']
    h = np.broadcast_to(onsite, (cells, 4, 4)).copy()
    for d, t in paths:
        phase = np.exp(1j * bare.a * (p @ d))
        h += phase[:, None, None] * t + phase.conj()[:, None, None] * t.conj().T
    values, vectors = np.linalg.eigh(h)
    if not (np.all(values[:, 1] < 0) and np.all(values[:, 2] > 0)):
        raise ValueError('Actual raw light sea is not separated two-negative/two-positive')
    occupied = vectors[:, :, :2]
    contact_matrix = np.zeros_like(h)
    channels = []
    for sign in (-1, 1):
        current = np.zeros_like(h)
        for d, t in paths:
            axis = int(np.flatnonzero(d)[0])
            if axis != polarization:
                continue
            phase = np.exp(1j * bare.a * (p @ d))
            reverse = np.exp(-1j * (bare.a * (p @ d) + sign * angle * d[0]))
            current += -.5j * bare.charge * (phase[:, None, None] * t - reverse[:, None, None] * t.conj().T)
            if sign == 1:
                contact_matrix -= .5 * bare.charge**2 * (phase[:, None, None] * t + phase.conj()[:, None, None] * t.conj().T)
        shift = -sign * reduced_pole['mode']
        shifted_vectors = np.roll(vectors.reshape((length,) * 3 + (4, 4)), shift, axis=0).reshape(cells, 4, 4)
        shifted_values = np.roll(values.reshape((length,) * 3 + (4,)), shift, axis=0).reshape(cells, 4)
        gaps = shifted_values[:, 2:, None] - values[:, None, :2]
        weights = abs(shifted_vectors[:, :, 2:].conj().swapaxes(1, 2) @ current @ occupied)**2
        channels.append((gaps, weights))
    norm = cells / 2
    gradient_contact_matrix = np.zeros_like(h)
    gradient_pair = 0.
    for sign in (-1, 1):
        vertex = np.zeros_like(h)
        path = np.exp(1j * sign * angle)-1
        for d, t in paths:
            if d[0] != 1:
                continue
            phase = np.exp(1j * bare.a * (p @ d))
            reverse = np.exp(-1j * (bare.a * (p @ d)+sign * angle))
            vertex += -.5j * bare.charge * path * (
                phase[:, None, None] * t-reverse[:, None, None] * t.conj().T)
            if sign == 1:
                gradient_contact_matrix -= .5 * bare.charge**2 * abs(path)**2 * (
                    phase[:, None, None] * t+phase.conj()[:, None, None] * t.conj().T)
        shifted = np.roll(vectors.reshape((length,) * 3 + (4, 4)),
                          -sign * reduced_pole['mode'], axis=0).reshape(cells, 4, 4)
        gaps = channels[(sign+1)//2][0]
        weights = abs(shifted[:, :, 2:].conj().swapaxes(1, 2) @ vertex @ occupied)**2
        gradient_pair -= 2 * float(np.sum(weights/gaps))/norm
    gradient_contact = float(np.trace(occupied.conj().swapaxes(1, 2) @ gradient_contact_matrix @ occupied,
                                     axis1=1, axis2=2).real.sum())/norm
    gradient_residual = abs(gradient_contact+gradient_pair)/max(1., abs(gradient_contact), abs(gradient_pair))
    require(gradient_residual, 'actual raw same-current longitudinal Ward', 1e-11)
    contact = float(np.trace(occupied.conj().swapaxes(1, 2) @ contact_matrix @ occupied,
                            axis1=1, axis2=2).real.sum()) / norm
    pair = -2 * sum(float(np.sum(w / g)) for g, w in channels) / norm
    electric = 2 * sum(float(np.sum(w / g**3)) for g, w in channels) / norm
    epsilon, magnetic = reduced_pole['lagrangian_epsilon'], reduced_pole['hamiltonian_K1']
    khat = reduced_pole['lattice_momentum']
    static = magnetic * khat**2 + contact + pair
    gap_squared = min(float(g.min())**2 for g, _ in channels)
    upper = static / (epsilon + electric)
    if not 0 < upper < gap_squared:
        raise ValueError('Actual raw field sea has no certified positive below-pair-gap pole')
    def inverse(s):
        nonlinear = 2 * s**2 * sum(float(np.sum(w / (g**3 * (g**2-s))))
                                   for g, w in channels) / norm
        return static * (1-s/upper) - nonlinear
    fraction, convergence = brentq(lambda value: inverse(value*upper), 0., 1.,
                                  xtol=1e-15, rtol=np.float64(1e-13), full_output=True)
    root = float(fraction*upper)
    residue = epsilon + 2 * sum(float(np.sum(w * g / (g**2-root)**2)) for g, w in channels) / norm
    lower = static / (epsilon + electric / (1-upper/gap_squared))
    errors = dict(contact=abs(contact-reduced_pole['static_contact_per_profile_norm']),
                  pair=abs(pair-reduced_pole['static_pair_per_profile_norm']),
                  electric=abs(electric-reduced_pole['DeltaChi_electric_transverse']),
                  pole_energy=float(abs(np.sqrt(root)-reduced_pole['photon_energy'])),
                  link_residue_inverse=abs(residue-reduced_pole['link_residue_inverse']))
    for name, error in errors.items():
        require(error, 'raw/reduced sea ' + name, 2e-5)
    pole = dict(reduced_pole)
    pole.update(photon_energy=float(np.sqrt(root)), pair_threshold=float(np.sqrt(gap_squared)),
                phase_velocity=float(np.sqrt(root) / reduced_pole['momentum']),
                lattice_phase_velocity=float(band.a * np.sqrt(root) / khat),
                static_contact_per_profile_norm=contact, static_pair_per_profile_norm=pair,
                static_restoring_force=static, relative_pole_residual=abs(inverse(root))/static,
                DeltaChi_magnetic=(contact+pair)/khat**2,
                DeltaChi_electric_transverse=electric, link_residue_inverse=residue,
                field_residue_inverse=residue/band.a, canonical_link_leg_factor=1/np.sqrt(residue),
                canonical_field_leg_factor=float(np.sqrt(band.a/residue)),
                canonical_cell_field_leg_factor=1/(band.a*np.sqrt(residue)),
                canonical_current_coupling=float(band.charge*np.sqrt(band.a/residue)),
                low_frequency_pole_upper=float(np.sqrt(upper)), rigorous_pole_lower=float(np.sqrt(lower)),
                source_owner='actual raw receipt frame times source-owned bare scalar_paths',
                source_bare_band_parameters=band_parameters(bare), raw_reduced_residuals=errors,
                consumer_mapping='field-derived RotationBand; finite intertwiner residual declared, not exact machine identity')
    pole['actual_raw_longitudinal_control'] = dict(contact=gradient_contact, pair=gradient_pair,
        relative_residual=gradient_residual, no_contact_residual=abs(gradient_pair),
        no_pairs_residual=abs(gradient_contact))
    expansion = []
    for fraction in (.1, .25, .5):
        squared = fraction**2 * gap_squared
        exact = 2 * sum(float(np.sum(w / (g * (g**2-squared)))) for g, w in channels) / norm
        expansion.append(dict(omega_over_pair_gap=fraction, exact_electric=exact,
                              lower=electric, upper=electric/(1-fraction**2),
                              bounded=electric-1e-12 <= exact <= electric/(1-fraction**2)+1e-12))
    pole['frequency_bounds'] = expansion
    pole['checks'] = dict(actual_raw_positive_below_gap=0 < root < gap_squared,
        actual_raw_positive_residue=residue >= epsilon > 0, actual_raw_pole_equation=pole['relative_pole_residual'] < 1e-10,
        actual_raw_bracket=lower-1e-12 <= root <= upper+1e-12,
        raw_reduced_frame_transfer=max(errors.values()) <= 2e-5,
        actual_raw_spectral_bounds=all(x['bounded'] for x in expansion),
        actual_raw_root_converged=convergence.converged,
        actual_raw_longitudinal_Ward=gradient_residual < 1e-11)
    return pole


def current_excited_gauss(bare, frame):
    model = FullBandGaussSection(bare, heavy_scale=1.)
    fields = np.tile(frame, (2, 1, 1, 1))
    link_generator = np.einsum('a,aij->ij', .13 * np.sin(np.arange(15)*.37+.29), model.generators)
    link = expm(1j * link_generator)
    phases, direction = np.array([.137, -.081]), np.array([.31, -.27])
    section = model.state(fields, link, phases)
    psi = section.state
    def current(vector, f=fields, u=link, a=phases):
        return model.peierls_action(vector, f, u, a, direction)
    excited = current(psi)
    excitation_norm = float(np.linalg.norm(excited))
    if excitation_norm <= 1e-7:
        raise ValueError('Actual primitive source current does not excite this neutral section')
    onebody_current = model.onebody_derivative(fields, link, phases, np.zeros_like(fields),
                                              np.zeros_like(link), direction)[1]
    owner_error = float(np.linalg.norm(excited-model.lift(onebody_current, 1) @ psi))
    require(owner_error, 'primitive current versus owner derivative', 3e-8)
    undressed = np.tile(SELECTED, (2, 1, 1, 1))
    bad_excited = model.peierls_action(psi, undressed, link, phases, direction)
    rows, bad_rows = [], []
    for site in range(2):
        for index, generator in enumerate(model.generators):
            df = np.zeros_like(fields)
            df[site] = 1j * (generator @ fields[site]-fields[site] @ generator)
            du = 1j * generator @ link if site == 0 else -1j * link @ generator
            dpsi = model.derivative(section, fields, link, phases, df, du)
            q = model.matter.Q[site][index]
            errors, bad_errors = [], []
            for step in (2e-4, 1e-4):
                gplus, gminus = expm(1j*step*generator), expm(-1j*step*generator)
                fp, fm = fields.copy(), fields.copy()
                fp[site], fm[site] = gplus @ fields[site] @ gplus.conj().T, gminus @ fields[site] @ gminus.conj().T
                up = gplus @ link if site == 0 else link @ gplus.conj().T
                um = gminus @ link if site == 0 else link @ gminus.conj().T
                # Independent finite owner derivative of J at fixed psi: NOT
                # i[Q,J], and no differentiated covariance identity as supplier.
                djpsi = (current(psi, fp, up)-current(psi, fm, um))/(2*step)
                errors.append(float(np.linalg.norm(djpsi+current(dpsi)-1j*(q @ excited))))
                bad_dj = (model.peierls_action(psi, undressed, up, phases, direction)
                          - model.peierls_action(psi, undressed, um, phases, direction))/(2*step)
                bad_errors.append(float(np.linalg.norm(bad_dj+model.peierls_action(
                    dpsi, undressed, link, phases, direction)-1j*(q @ bad_excited))))
            require(errors[-1], 'actual current-excited SU4 Gauss', 3e-7)
            rows.append(dict(site=site, generator=index, steps=[2e-4, 1e-4], residuals=errors))
            bad_rows.append(dict(site=site, generator=index, residual=bad_errors[-1]))
    em_rows = []
    for site in range(2):
        gradient = np.array([1., -1.]) * (1 if site == 0 else -1)
        dpsi = model.derivative(section, fields, link, phases, dphases=gradient)
        step = 1e-4
        djpsi = (current(psi, a=phases+step*gradient)-current(psi, a=phases-step*gradient))/(2*step)
        q = model.number[1][site]-2*model.identity
        error = float(np.linalg.norm(djpsi+current(dpsi)-1j*(q @ excited)))
        require(error, 'actual current-excited light-only EM Gauss', 3e-7)
        em_rows.append(dict(site=site, residual=error))
    undressed_error = max(row['residual'] for row in bad_rows)
    if undressed_error <= 1e-5:
        raise ValueError('Real undressed current failed to break combined field/link Gauss covariance')
    return dict(primitive_dimension=model.matter.dimension, NH=4, NL=4,
                fields_real=np.real(fields).tolist(), fields_imag=np.imag(fields).tolist(),
                link_real=np.real(link).tolist(), link_imag=np.imag(link).tolist(), phases=phases.tolist(),
                source_direction=direction.tolist(), current_excitation_norm=excitation_norm,
                primitive_current_owner_residual=owner_error, SU4_current_excited_all30=rows,
                light_EM_current_excited_both_sites=em_rows, undressed_current_controls=bad_rows,
                undressed_current_max_defect=undressed_error, checks=dict(
                    actual_current_owner=owner_error < 3e-8,
                    all30_current_excited_Gauss=max(row['residuals'][-1] for row in rows) < 3e-7,
                    both_EM_current_excited_Gauss=max(row['residual'] for row in em_rows) < 3e-7,
                    undressed_current_rejected=undressed_error > 1e-5),
                geometry='inherited one-active-x-axis two-cell owner (onsite m+r/a), same actual bulk frame; not the full periodic 3D Hamiltonian',
                scope='neutral NH=NL=4 section; dJ psi+J dpsi=iQ Jpsi on the actual field/link gauge path; not external charged B4/bar4')


def run(input_bulk_receipt, branch, lengths, electric, magnetic):
    identities = source_hashes()
    bare, frame, binding = read_bulk(input_bulk_receipt, branch)
    identities = {**binding['supplier_source_sha256'], **identities}
    band, unitary, reconstruction = reconstruct_frame(bare, frame)
    controls = primitive_frame_controls(bare, band, frame, unitary)
    gauss = current_excited_gauss(bare, frame)
    rows = []
    for length in lengths:
        poles = []
        for polarization in (1, 2):
            reduced = match_photon(length, band, electric_coefficient=electric,
                                   magnetic_coefficient=magnetic, polarization_axis=polarization)
            poles.append(actual_sea_pole(length, bare, frame, band, reduced))
        polarization_error = max(abs(poles[0][key]-poles[1][key]) for key in (
            'photon_energy', 'link_residue_inverse', 'DeltaChi_electric_transverse'))
        require(polarization_error, 'independent transverse polarization degeneracy', 2e-5)
        # Use one measured axis-x profile in Compton; the independently computed
        # z profile controls the second physical transverse polarization.
        response = forward_at_pole(band, poles[0])
        gradient = momentum_response(length, band, gradient=True)
        checks = dict(independent_two_transverse_profiles=polarization_error < 2e-5,
                      actual_raw_poles=all(all(p['checks'].values()) for p in poles),
                      actual_pauli_response=all(response['checks'].values()),
                      # Deleted-channel sizes remain in the raw diagnostics;
                      # they are not universal lower bounds on a valid sea.
                      same_current_longitudinal_Ward=gradient['gradient_relative_residual'] < 1e-11)
        rows.append(dict(L=length, actual_raw_photon_poles=poles, independent_polarization_error=polarization_error,
                         actual_pauli_forward=response, longitudinal_sea_control=gradient, checks=checks))
    final = {**{name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
               for name in binding['supplier_source_sha256']}, **source_hashes()}
    if final != identities or hashlib.sha256(input_bulk_receipt.read_bytes()).hexdigest() != binding['sha256']:
        raise RuntimeError('Source/proof/input bytes changed during run; refusing mixed-revision receipt')
    original = binding['original_seed_binding']
    if hashlib.sha256(Path(original['path']).read_bytes()).hexdigest() != original['sha256']:
        raise RuntimeError('Original bulk seed bytes changed during run')
    passed = all(gauss['checks'].values()) and all(all(row['checks'].values()) for row in rows)
    return dict(schema=1, governing_plan=PLAN, assignment_plan_revision=PLAN_REVISION,
                source_sha256=final, input_bulk_binding=binding, source_and_input_unchanged=True,
                bare_band_parameters=band_parameters(bare), actual_fields_real=np.real(frame).tolist(),
                actual_fields_imag=np.imag(frame).tolist(), frame_reconstruction=reconstruction,
                primitive_frame_controls=controls, primitive_current_excited_neutral_Gauss=gauss,
                rows=rows, all_checks_pass=bool(passed),
                approximation='actual finite uniform bulk mean; Gaussian links plus exact quadratic filled light sea; field-derived approximate Clifford matter reduction',
                limits=dict(spacing=band.a, Hamiltonian_UE1=electric, Hamiltonian_K1=magnetic,
                    Lagrangian_epsilon=1/electric, bare_photon_speed=band.a*np.sqrt(electric*magnetic),
                    matter_speed=band.v, rest_energy=band.m, photon_phase=None,
                    continuum='requires |a p|<<1 and a*r_eff*m_eff << v_eff^2 for relativistic inertial relation; fixed-frame a->0 not a solved cutoff flow',
                    UV='unimproved Wilson corners m_eff+2n*r_eff/a; no borrowed improved hopping or silently c=1'),
                unresolved=['Finite supplier mesh/energy bounds do not select a global or thermodynamic ground.',
                    'Internal spin algebra and low-momentum envelope rotations do not establish exact continuous spatial rotations of the finite cubic lattice.',
                    'Neutral current-excited Gauss covariance is not a physical external B4/bar4 charged gap, charged self-energy, normal confinement or selected phase.',
                    'Gaussian positive poles do not prove a compact Coulomb phase; photon_phase is absent/null, not zero.',
                    'Connected forward Pauli tree response with measured external legs is not full radiative Compton or Lorentz parameter closure.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-bulk-receipt', type=Path, required=True)
    parser.add_argument('--branch', required=True)
    parser.add_argument('--lengths', type=int, nargs='+', required=True)
    parser.add_argument('--electric', type=float, required=True, help='Hamiltonian UE1, not epsilon')
    parser.add_argument('--magnetic', type=float, required=True, help='Hamiltonian K1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.lengths or any(length < 3 for length in args.lengths):
        raise ValueError('At least one physical finite torus with L>=3 required')
    result = run(args.input_bulk_receipt, args.branch, args.lengths, args.electric, args.magnetic)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload+'\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

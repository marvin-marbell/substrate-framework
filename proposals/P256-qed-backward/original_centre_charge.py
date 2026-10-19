#!/usr/bin/env python3
"""Original native SU4 centre packets and exact external-colour charge transfer.

No Z4 Hamiltonian replaces SU4. Haar quadrature is a finite diagnostic;
analytic bounds and operator product identities are distinguished from it.
Neither a many-body ground nor a deconfined phase is numerically solved.
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
from scipy.linalg import expm

from bulk_flat_holonomy_bound import (PLAN, PLAN_REVISION, BARE, STIFFNESS,
                                    RotationBand, SELECTED, periodic_grid,
                                    selected_gram)
from sea_selection import matrix_unit_basis

ROOT = Path(__file__).resolve().parent
CENTRES = (1j) ** np.arange(4)
C_F = 15 / 8
ALIGNMENT = .2
COLOUR_PLAQUETTE = .7
OVERLAP_EXPONENT = 8 - 4 * math.sqrt(2)


def check(error: float, label: str, tolerance: float = 2e-9) -> None:
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error} > {tolerance}')


def source_hashes() -> dict[str, str]:
    names = ('original_centre_charge.py', 'original_centre_charge.md',
             'bulk_flat_holonomy_bound.py', 'bulk_flat_holonomy_bound.md',
             'bulk_quantum_selection.py', 'sea_selection.py', 'soft_budget.py',
             'rotation_band.py', 'local_orbital_stability.py',
             'unrestricted_field_minimum.py')
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def source_frame(spacing: float, reference_grid: int) -> tuple[np.ndarray, dict[str, Any]]:
    band = RotationBand(**{**BARE, 'a': spacing})
    momenta = 2 * np.pi * periodic_grid(reference_grid) / (reference_grid * spacing)
    selected = selected_gram(band, momenta, STIFFNESS / spacing)
    g = np.asarray(selected['G'], dtype=float)
    if np.linalg.eigvalsh(g)[0] <= 0:
        raise AssertionError('Original source selected Gram is not full rank')
    frame = np.sqrt(np.diag(g))[:, None, None] * SELECTED
    check(float(np.linalg.norm(g - np.diag(np.diag(g)))), 'source diagonal Gram')
    return frame, dict(spacing=spacing, grid=reference_grid, stiffness=STIFFNESS / spacing,
                      Gram=g.tolist(), value=float(selected['value']),
                      status='source finite-grid full-rank frame; not interval continuum Gstar',
                      added_local_CAR_selector_energy_used=False)


def potential(left: np.ndarray, right: np.ndarray, u: np.ndarray) -> float:
    """The actual complete covariant native link W, not a centre-only surrogate."""
    difference = left - u @ right @ u.conj().T
    return float(np.linalg.norm(difference) ** 2)


def potential_derivatives(frame: np.ndarray, u: np.ndarray,
                          generators: np.ndarray) -> np.ndarray:
    transported = u @ frame @ u.conj().T
    difference = frame - transported
    return np.array([-2 * np.vdot(difference,
        1j * (t @ transported - transported @ t)).real for t in generators])


def frame_control(frame: np.ndarray, generators: np.ndarray) -> dict[str, Any]:
    scales = np.einsum('mij,mji->m', frame, frame).real / 4
    residual = max(float(np.linalg.norm(a @ b + b @ a -
                   (2 * scales[i] * np.eye(4) if i == j else 0)))
                   for i, a in enumerate(frame) for j, b in enumerate(frame))
    check(residual, 'actual source weighted Clifford frame')
    units = np.eye(16, dtype=complex).reshape(16, 4, 4)
    images = np.array([sum(a @ (a @ e - e @ a) - (a @ e - e @ a) @ a
                           for a in frame) for e in units])
    superoperator = images.reshape(16, 16).T
    values = np.linalg.eigvalsh(superoperator)
    check(float(np.linalg.norm(superoperator - superoperator.conj().T)), 'commutator Hermiticity')
    predicted = []
    for mask in range(16):
        grade = mask.bit_count()
        predicted.append(4 * sum(scales[j] for j in range(4)
                         if bool(mask & (1 << j)) == (grade % 2 == 0)))
    check(float(np.linalg.norm(values - np.sort(predicted))), 'full native Clifford commutator spectrum')
    tangents = np.array([1j * (t @ frame - frame @ t) for t in generators])
    quadratic = np.einsum('amij,bmij->ab', tangents.conj(), tangents).real
    local_values = np.linalg.eigvalsh(quadratic)
    check(float(np.linalg.norm(local_values - values[1:] / 2)), '15 Casimir local normal curvatures')
    if values[1] <= 0:
        raise AssertionError('Actual Clifford frame must have scalar-only commutant')
    check(float(np.linalg.norm(2 * np.einsum('aij,bji->ab', generators, generators) - np.eye(15))),
          'canonical SU4 generator metric')
    check(float(np.linalg.norm(sum(t @ t for t in generators) - C_F * np.eye(4))),
          'native fundamental SU4 Casimir')
    return dict(Gram_diagonal=scales.tolist(), Clifford_residual=residual,
                commutator_eigenvalues=values.tolist(), scalar_commutant_dimension=1,
                ell_min=float(values[1]), ell_max=float(values[-1]),
                local_W_quadratic_eigenvalues=local_values.tolist(),
                W_zero_set=['I', 'iI', '-I', '-iI'], centre_minima_not_deleted=True)


def packet(u: np.ndarray, eta: float, charge: int) -> complex:
    seeds = np.exp(eta * ((CENTRES.conj() * np.trace(u)).real - 4))
    return complex(np.dot(CENTRES ** charge, seeds))


def packet_derivatives(u: np.ndarray, eta: float, charge: int,
                       generators: np.ndarray) -> np.ndarray:
    seeds = np.exp(eta * ((CENTRES.conj() * np.trace(u)).real - 4))
    traces = np.einsum('aij,ji->a', generators, u)
    seed_derivatives = eta * (1j * CENTRES.conj()[:, None] * traces[None, :]).real * seeds[:, None]
    return np.einsum('k,ka->a', CENTRES ** charge, seed_derivatives)


def native_controls(frame: np.ndarray, generators: np.ndarray,
                    eta: float, seed: int = 222970) -> dict[str, Any]:
    """Exercise actual noncentral matrices, all 15 derivatives and full Gauss action."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    u, triangular = np.linalg.qr(raw)
    phases = np.diag(triangular) / abs(np.diag(triangular))
    u = u * phases[None, :]
    u[:, 0] /= np.linalg.det(u)
    direction = np.einsum('a,aij->ij', rng.normal(size=15), generators)
    near = expm(.25j * direction)
    controls = [u, near, np.diag(np.exp(1j * np.array([.6, -.2, .1, -.5])))]
    centre_w = [potential(frame, frame, z * np.eye(4)) for z in CENTRES]
    check(max(centre_w), 'all four actual centre minima')
    derivative_errors, kinetic_errors, characters, w_values = [], [], [], []
    w_derivative_errors = []
    all_derivatives = []
    epsilon = 1e-6
    ell = frame_control(frame, generators)
    for matrix in controls:
        w = potential(frame, frame, matrix)
        if w <= 1e-6:
            raise AssertionError('Noncentral control failed to reach positive W')
        w_values.append(w)
        scalar_distance = 4 - abs(np.trace(matrix)) ** 2 / 4
        check(max(0., ell['ell_min'] * scalar_distance - w), 'global noncentral W lower')
        check(max(0., w - ell['ell_max'] * scalar_distance), 'global noncentral W upper')
        w_gradient = potential_derivatives(frame, matrix, generators)
        w_fd = np.array([(potential(frame, frame, expm(1j * epsilon * t) @ matrix) -
                          potential(frame, frame, expm(-1j * epsilon * t) @ matrix)) /
                         (2 * epsilon) for t in generators])
        w_derivative_errors.append(float(np.linalg.norm(w_gradient - w_fd)))
        for q in range(4):
            value = packet(matrix, eta, q)
            analytic = packet_derivatives(matrix, eta, q, generators)
            numerical = np.array([(packet(expm(1j * epsilon * t) @ matrix, eta, q) -
                                   packet(expm(-1j * epsilon * t) @ matrix, eta, q)) /
                                  (2 * epsilon) for t in generators])
            scale = max(1e-12, abs(value), float(np.linalg.norm(analytic)))
            derivative_errors.append(float(np.linalg.norm(analytic - numerical)) / scale)
            characters.append(max(abs(packet(z * matrix, eta, q) - z ** q * value)
                                  for z in CENTRES) / max(1e-12, abs(value)))
            traces = np.trace(matrix)
            trace2 = np.trace(matrix @ matrix)
            a = packet(matrix, eta, (q - 1) % 4)
            b = packet(matrix, eta, (q + 1) % 4)
            c = 2 - abs(traces) ** 2 / 8
            d = (trace2 - traces ** 2 / 4) / 2
            fierz = eta ** 2 / 4 * ((abs(a) ** 2 + abs(b) ** 2) * c -
                                   2 * (a * b.conjugate() * d).real)
            kinetic_errors.append(abs(float(np.vdot(analytic, analytic).real) - fierz) /
                                  max(1e-18, float(np.vdot(analytic, analytic).real)))
            if matrix is near:
                all_derivatives.append(dict(charge=q, real=np.real(analytic).tolist(),
                                            imaginary=np.imag(analytic).tolist()))
    check(max(derivative_errors), 'all15 actual packet left-Casimir derivatives', 2e-7)
    check(max(w_derivative_errors), 'all15 actual native potential derivatives', 2e-8)
    check(max(kinetic_errors), 'all15 derivative sum versus exact Fierz formula', 2e-8)
    check(max(characters), 'actual fundamental and antifundamental centre characters', 2e-8)
    # An external fund at L and anti-fund at R are spectator sources, not CAR particles.
    # Their coefficient matrix M=U/2 has unit norm, and transforms hL M hR^dagger.
    h_left = expm(.31j * np.einsum('a,aij->ij', rng.normal(size=15), generators))
    h_right = expm(.27j * np.einsum('a,aij->ij', rng.normal(size=15), generators))
    moved = h_left @ near @ h_right.conj().T
    left_fields = h_left @ frame @ h_left.conj().T
    right_fields = h_right @ frame @ h_right.conj().T
    covariance = abs(potential(left_fields, right_fields, moved) - potential(frame, frame, near))
    check(covariance, 'actual independent endpoint SU4 potential invariance')
    m = near / 2
    derivatives = np.array([1j * t @ m for t in generators])
    cross = np.einsum('ij,aij->a', m.conj(), derivatives)
    extra = float(np.linalg.norm(derivatives) ** 2)
    check(float(np.linalg.norm(cross)), 'all15 transporter kinetic cross cancellation')
    check(abs(extra - C_F), 'normalized reference transporter native Casimir cost')
    # Test the product identity with arbitrary vector-valued neutral derivatives.
    neutral = rng.normal(size=7) + 1j * rng.normal(size=7)
    neutral /= np.linalg.norm(neutral)
    d_neutral = rng.normal(size=(15, 7)) + 1j * rng.normal(size=(15, 7))
    product_derivatives = (derivatives[:, :, :, None] * neutral[None, None, None, :] +
                           m[None, :, :, None] * d_neutral[:, None, None, :])
    difference = float(np.linalg.norm(product_derivatives) ** 2 - np.linalg.norm(d_neutral) ** 2)
    check(abs(difference - C_F), 'matrix-valued neutral section exact kinetic product identity', 2e-8)
    gauss_left, gauss_right = [], []
    for t in generators:
        left_fd = (expm(1j * epsilon * t) @ near - expm(-1j * epsilon * t) @ near) / (4 * epsilon)
        right_fd = (near @ expm(-1j * epsilon * t) - near @ expm(1j * epsilon * t)) / (4 * epsilon)
        gauss_left.append(float(np.linalg.norm(left_fd - 1j * t @ m)))
        gauss_right.append(float(np.linalg.norm(right_fd + 1j * m @ t)))
    check(max(gauss_left + gauss_right), 'all15 actual SU4 Gauss reference pair', 2e-8)
    wrong_reference_residual = float(np.linalg.norm(1j * generators[0] @ m + 1j * m @ generators[0]))
    if wrong_reference_residual <= 1e-3:
        raise AssertionError('Wrong fundamental/fundamental pairing control failed')
    # Full SU4 packet section on the actual two-frame orbit, not just its Z4 label.
    vector_left = np.eye(4, dtype=complex)[:, 0]
    vector_right = np.eye(4, dtype=complex)[:, 1]
    def charged_section(gl: np.ndarray, gr: np.ndarray, link: np.ndarray) -> np.ndarray:
        body = gl.conj().T @ link @ gr
        return packet(body, eta, 1) * np.outer(gl @ vector_left, (gr @ vector_right).conj())
    section = charged_section(h_left, h_right, moved)
    section_scale = max(1e-12, float(np.linalg.norm(section)))
    section_isotropy = max(
        float(np.linalg.norm(charged_section(z * h_left, h_right, moved) - section)) / section_scale
        for z in CENTRES)
    section_isotropy = max(section_isotropy, max(
        float(np.linalg.norm(charged_section(h_left, z * h_right, moved) - section)) / section_scale
        for z in CENTRES))
    section_gauss = []
    for t in generators:
        plus, minus = expm(1j * epsilon * t), expm(-1j * epsilon * t)
        dl = (charged_section(plus @ h_left, h_right, plus @ moved) -
              charged_section(minus @ h_left, h_right, minus @ moved)) / (2 * epsilon)
        dr = (charged_section(h_left, plus @ h_right, moved @ plus.conj().T) -
              charged_section(h_left, minus @ h_right, moved @ minus.conj().T)) / (2 * epsilon)
        section_gauss.extend((float(np.linalg.norm(dl - 1j * t @ section)) / section_scale,
                              float(np.linalg.norm(dr + 1j * section @ t)) / section_scale))
    check(section_isotropy, 'packet/reference descent under both independent frame centres', 2e-8)
    check(max(section_gauss), 'packet/reference complete field-link SU4 Gauss action', 2e-7)
    # Two-edge noncentral path: interior Gauss cancels; both edge Casimirs remain.
    path_links = (near, u)
    path = path_links[0] @ path_links[1] / 2
    path_derivatives = np.concatenate((
        np.array([1j * t @ path for t in generators]),
        np.array([path_links[0] @ (1j * t) @ path_links[1] / 2 for t in generators])))
    path_cross = np.einsum('ij,aij->a', path.conj(), path_derivatives)
    path_casimir = float(np.linalg.norm(path_derivatives) ** 2)
    check(float(np.linalg.norm(path_cross)), '30 distinct-edge path derivative cross cancellations')
    check(abs(path_casimir - 2 * C_F), 'two-edge fullSU4 path Casimir cost')
    interior = expm(.17j * direction)
    check(float(np.linalg.norm((path_links[0] @ interior.conj().T) @
          (interior @ path_links[1]) / 2 - path)), 'noncentral interior Gauss path cancellation')
    return dict(seed=seed, all15_derivatives_at_noncentral_controls=True,
                noncentral_W=w_values, centre_W=centre_w,
                packet_derivative_relative_error=max(derivative_errors),
                potential_derivative_error=max(w_derivative_errors),
                kinetic_Fierz_relative_error=max(kinetic_errors),
                centre_character_relative_error=max(characters),
                native_near_point_derivatives=all_derivatives,
                endpoint_SU4_covariance_error=float(covariance),
                reference_pair_norm=float(np.linalg.norm(m) ** 2),
                reference_pair_Casimir_extra=extra,
                all15_cross_inner_products_real=cross.real.tolist(),
                all15_cross_inner_products_imaginary=cross.imag.tolist(),
                arbitrary_neutral_section_kinetic_extra=difference,
                full_left_right_Gauss_derivative_error=max(gauss_left + gauss_right),
                packet_reference_frame_isotropy_error=section_isotropy,
                packet_reference_all15_Gauss_error=max(section_gauss),
                two_distinct_edge_path_Casimir_extra=path_casimir,
                path_all30_cross_inner_product_error=float(np.linalg.norm(path_cross)),
                wrong_reference_representation_residual=wrong_reference_residual)


def haar_packet_integrals(eta: float, grid: int, scales_sum: float) -> dict[str, Any]:
    """Entire SU4 Weyl torus; no omitted group angles or finite irrep cutoff.

    Class packet permits exact conjugation averaging of W. This quadrature
    estimates exact infinite-Haar formulas; its grid difference is not a
    rigorous interval error or a solved rotor ground.
    """
    angles = 2 * np.pi * np.arange(grid) / grid - np.pi
    theta2, theta3 = np.meshgrid(angles, angles, indexing='ij')
    norms = np.zeros(4)
    gradients = np.zeros(4)
    w_numerators = np.zeros(4)
    seed_norm = seed_gradient = seed_w = haar_mass = 0.
    inverse_centres = CENTRES.conj()
    for theta1 in angles:
        eigenvalues = np.stack((np.full_like(theta2, np.exp(1j * theta1), dtype=complex),
                               np.exp(1j * theta2), np.exp(1j * theta3),
                               np.exp(-1j * (theta1 + theta2 + theta3))), axis=-1)
        trace = eigenvalues.sum(axis=-1)
        trace2 = (eigenvalues ** 2).sum(axis=-1)
        vandermonde2 = np.ones_like(theta2)
        for i in range(4):
            for j in range(i + 1, 4):
                vandermonde2 *= abs(eigenvalues[..., i] - eigenvalues[..., j]) ** 2
        weight = vandermonde2 / (24 * grid ** 3)
        haar_mass += float(weight.sum())
        seeds = np.exp(eta * ((inverse_centres[:, None, None] * trace[None, :, :]).real - 4))
        packets = 4 * np.fft.ifft(seeds, axis=0)
        c = np.maximum(0., 2 - abs(trace) ** 2 / 8)
        d = (trace2 - trace ** 2 / 4) / 2
        averaged_w = 8 / 15 * scales_sum * np.maximum(0., 16 - abs(trace) ** 2)
        density = abs(packets) ** 2
        for q in range(4):
            a, b = packets[(q - 1) % 4], packets[(q + 1) % 4]
            kinetic_density = eta ** 2 / 4 * ((abs(a) ** 2 + abs(b) ** 2) * c -
                                             2 * (a * b.conj() * d).real)
            if float(kinetic_density.min()) < -1e-12:
                raise AssertionError('Exact Casimir kinetic density became negative')
            norms[q] += float(np.sum(weight * density[q]))
            gradients[q] += float(np.sum(weight * np.maximum(0., kinetic_density)))
            w_numerators[q] += float(np.sum(weight * density[q] * averaged_w))
        f2 = seeds[0] ** 2
        # |grad ReTr U|^2=(C-Re D)/2 with the same canonical all15 generators.
        seed_norm += float(np.sum(weight * f2))
        seed_gradient += float(np.sum(weight * f2 * eta ** 2 * (c - d.real) / 2))
        seed_w += float(np.sum(weight * f2 * averaged_w))
    check(abs(haar_mass - 1), 'complete SU4 Weyl Haar normalization', 2e-10)
    if min(norms) <= 0 or seed_norm <= 0:
        raise AssertionError('Whole-Haar sector packet has zero sampled norm')
    return dict(grid=grid, eta=eta, Haar_mass=haar_mass,
                sector_norms=norms.tolist(), sector_Casimir= (gradients / norms).tolist(),
                sector_W=(w_numerators / norms).tolist(),
                one_well_norm=seed_norm, one_well_Casimir=seed_gradient / seed_norm,
                one_well_W=seed_w / seed_norm,
                status='full-torus deterministic quadrature estimate, not rigorous interval')


def rigorous_packet_bound(eta: float, u4: float, v_alignment: float,
                          ell_max: float) -> dict[str, Any]:
    """Explicit analytic bound for global analytic exp-seed packets, no support cut."""
    if eta < 1:
        raise ValueError('Analytic packet bound uses eta>=1')
    # Canonical metric 2Tr(dX^2), Haar normalized to one; proof derives this volume.
    group_volume = 2 ** 7.5 * (2 * math.pi) ** 9 / 6
    ball_volume = math.pi ** 7.5 / math.gamma(8.5)
    jacobian_lower = float(np.sinc(1 / (2 * np.pi)) ** 12)
    seed_norm_lower = ball_volume / group_volume * math.exp(-.5) * jacobian_lower * eta ** -7.5
    overlap = math.exp(-OVERLAP_EXPONENT * eta)
    overlap_ratio = overlap / seed_norm_lower
    log_tail = math.log(8 * eta) - math.log(seed_norm_lower)
    deficit_upper = min(8., (log_tail / 2 + 1) / eta)
    seed_kinetic_upper = 15 / 4 * u4 * eta
    seed_potential_upper = 2 * v_alignment * ell_max * deficit_upper
    seed_energy_upper = seed_kinetic_upper + seed_potential_upper
    cross_form_ratio = (2 * u4 * eta ** 2 + 4 * v_alignment * ell_max) * overlap_ratio
    certificate = 3 * overlap_ratio < 1
    common_upper = ((seed_energy_upper + 3 * cross_form_ratio) / (1 - 3 * overlap_ratio)
                    if certificate else None)
    deviation = ((3 * cross_form_ratio + 3 * overlap_ratio * seed_energy_upper) /
                 (1 - 3 * overlap_ratio) if certificate else None)
    return dict(eta=eta, normalized_Haar_volume=group_volume,
                seed_norm_rigorous_lower=seed_norm_lower,
                Haar_Jacobian_lower_in_unit_ball=jacobian_lower,
                offdiagonal_seed_overlap_rigorous_upper=overlap,
                overlap_to_norm_rigorous_upper=overlap_ratio,
                trace_deficit_expectation_upper=deficit_upper,
                one_well_kinetic_upper=seed_kinetic_upper,
                one_well_potential_upper=seed_potential_upper,
                one_well_trial_energy_upper=seed_energy_upper,
                four_sector_common_trial_upper=common_upper,
                each_sector_trial_deviation_from_one_well_upper=deviation,
                four_sector_trial_spread_upper=2 * deviation if deviation is not None else None,
                positive_norm_certificate=certificate,
                status='analytic inequalities; floating coefficients are not outward-rounded intervals',
                actual_ground_energy_computed=False)


def rotation_control(frame: np.ndarray) -> dict[str, Any]:
    """Derive compensator from the actual Clifford frame; distinguish gauge restoration."""
    scales = np.einsum('mij,mji->m', frame, frame).real / 4
    gamma1 = frame[1] / math.sqrt(float(scales[1]))
    gamma2 = frame[2] / math.sqrt(float(scales[2]))
    generator = gamma1 @ gamma2 / 2
    endpoint = expm(2 * math.pi * generator)
    check(float(np.linalg.norm(endpoint + np.eye(4))), 'Clifford compensator endpoint minusI')
    # A moving normalized fundamental basis has A=S^dagger dS=generator.
    # Parallel transporter cancels the endpoint overlap, giving the restored loop +I.
    connection_transport = expm(-2 * math.pi * generator)
    restored = endpoint @ connection_transport
    check(float(np.linalg.norm(restored - np.eye(4))), 'compact pure gauge/Berry restored loop plusI')
    return dict(compensator_endpoint_real=np.real(endpoint).tolist(),
                compensator_endpoint_imaginary=np.imag(endpoint).tolist(),
                localized_fundamental_centre_character_at_minusI=-1,
                antifundamental_reference_centre_character_at_minusI=-1,
                co_rotated_pair_centre_character=1,
                restored_compact_gauge_Berry_loop_error=float(np.linalg.norm(restored - np.eye(4))),
                full_physical_restored_rotation=1,
                fixed_reference_operation_not_equal_to_co_rotation=True,
                mechanical_half_spin_derived=False,
                limitation='A localized projective factor needs mobile charge and symmetry factorization; not supplied by this matrix lift.')


def run(*, spacing: float = 1., reference_grid: int = 32,
        lambdas: tuple[float, ...] = (1., 16., 256.),
        haar_grids: tuple[int, ...] = (96, 128, 160)) -> dict[str, Any]:
    if not np.isfinite(spacing) or not 0 < spacing <= 1 or reference_grid < 3:
        raise ValueError('Require 0<a<=1 and source grid>=3')
    if not lambdas or any(not np.isfinite(x) or x < 1 for x in lambdas):
        raise ValueError('Require finite Lambda>=1')
    if not haar_grids or min(haar_grids) < 8:
        raise ValueError('Require Haar grids>=8')
    identities = source_hashes()
    frame, selected = source_frame(spacing, reference_grid)
    generators = matrix_unit_basis() / math.sqrt(8)
    geometric = frame_control(frame, generators)
    native = native_controls(frame, generators, eta=4.)
    scales_sum = float(sum(geometric['Gram_diagonal']))
    rows = []
    for stiffness in lambdas:
        u4 = .3 * stiffness ** -.5
        v_alignment = ALIGNMENT / 8 * stiffness ** 1.25
        eta = max(4., math.sqrt(64 * v_alignment * scales_sum / (15 * u4)))
        estimates = []
        for grid in haar_grids:
            data = haar_packet_integrals(eta, grid, scales_sum)
            energies = u4 * np.asarray(data['sector_Casimir']) + v_alignment * np.asarray(data['sector_W'])
            data['sector_trial_energies'] = energies.tolist()
            data['one_well_trial_energy'] = u4 * data['one_well_Casimir'] + v_alignment * data['one_well_W']
            data['fund_antifund_trial_energy_difference'] = float(energies[1] - energies[3])
            check(abs(float(energies[1] - energies[3])), 'inversion fundamental/antifund packet symmetry', 2e-8)
            estimates.append(data)
        refinement = None
        if len(estimates) > 1:
            previous, final = estimates[-2:]
            e_prev = np.asarray(previous['sector_trial_energies'])
            e_final = np.asarray(final['sector_trial_energies'])
            refinement = dict(norm_max_relative_difference=float(np.max(abs(
                np.asarray(previous['sector_norms']) / np.asarray(final['sector_norms']) - 1))),
                energy_max_relative_difference=float(np.max(abs(e_prev - e_final) / np.maximum(1., abs(e_final)))),
                interval_error_certificate=False)
        normal = np.asarray(geometric['local_W_quadratic_eigenvalues'])
        rows.append(dict(Lambda=stiffness, u4=u4, alignment_multiplier=v_alignment, eta=eta,
                         colour_plaquette_multiplier=COLOUR_PLAQUETTE * stiffness ** 1.25,
                         Haar_packet_estimates=estimates, refinement_diagnostic=refinement,
                         rigorous_smooth_packet_bound=rigorous_packet_bound(eta, u4, v_alignment, geometric['ell_max']),
                         frozen_single_link_harmonic_ground=float(np.sqrt(u4 * v_alignment * normal).sum()),
                         frozen_single_link_harmonic_first_gap=float(2 * math.sqrt(u4 * v_alignment * normal[0])),
                         harmonic_scope='frozen-field single-link quadratic form only, not full Gauss/BH spectrum',
                         exact_scalar_fundamental_ground_splitting_interval=[0., u4 * C_F],
                         exact_scalar_antifundamental_ground_splitting_interval=[0., u4 * C_F],
                         full_original_external_pair_energy_difference_upper_per_distinct_edge=u4 * C_F,
                         full_original_pair_difference_lower_bound_proved=False))
    rotations = rotation_control(frame)
    if source_hashes() != identities:
        raise RuntimeError('Source/proof changed during original centre-charge execution')
    return dict(source_sha256=identities, source_bindings_unchanged=True,
                governing_plan=PLAN, assignment_plan_revision=PLAN_REVISION,
                native_law='u4*(-Delta_SU4)+(J/8)*Lambda^(5/4)*W; Tr(TaTb)=delta_ab/2; all SU4 points retained',
                candidate_trajectory=dict(u4_prefactor=.3, u4_power=-.5,
                    original_heavy_coefficient='Lambda', field_kinetic_original='sqrtLambda/2',
                    compatible_alternative_field_kinetic='Lambda^(-1/2)/2; separately changed law, no phase assumption'),
                source_selection=selected, geometric_controls=geometric,
                native_SU4_controls=native, rows=rows, rotation_boundary=rotations,
                full_Gauss_pair_theorem=dict(
                    references='spectator fundamental at L and antifundamental at R, not original light-CAR insertion',
                    section='M_gamma(U)=ordered_product_gamma(U)/2 tensor arbitrary exact neutral physical state',
                    pointwise_norm=1., reference_Casimir=C_F,
                    bound='E_external_fund_antifund(gamma)-E_neutral <= len(gamma)*15*u4/8 for simple paths',
                    original_heavy_light_and_all_field_derivatives_retained=True,
                    global_Gauss_includes_field_link_matter_and_reference=True,
                    external_pair_net_centre_character=0,
                    zero_momentum_Gauss_generators_deleted=False,
                    no_background_charge_deleted=True,
                    exact_neutral_ground_numerically_computed=False),
                free_rotor_adverse=dict(alignment=0., exact_scalar_neutral_ground=0.,
                    exact_scalar_fundamental_ground='15*u4/8',
                    exact_scalar_antifundamental_ground='15*u4/8',
                    implication='Equal localized trial upper energies never imply equal true sector grounds; the fullSU4 free rotor has strictly positive splitting for every u4>0.'),
                analytic_results=['actual weighted-Clifford native W has exactly four centre minima',
                    'global smooth centre-character packets have exact full-Haar normal/energy formulas and analytic trial bounds',
                    'scalar single-link sector ground splitting 0<=E_fund-E_trivial<=15u4/8, same for antifund',
                    'full original non-scalar Gauss Hamiltonian has exact external-reference-pair upper cost per simple-path edge'],
                missing_premises=['No actual original-light creator/propagating charged-sector embedding is derived.',
                    'At fixed Lambda the path bound is linear in separation; it is not deconfinement.',
                    'No uniform thermodynamic path-independent bound or plaquette phase/tunnelling theorem is proved.',
                    'Full original scalar/matrix angular q0, heavy Berry and Born-Huang terms are not discarded; their mobile-light transfer remains open.',
                    'No charged lower gap, photon pole, mechanical half-spin or joined QED completion.'],
                quantum_ground_computed=False, Z4_replacement_law_used=False,
                all_checks_pass=True, campaign_complete=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--reference-grid', type=int, default=32)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[1., 16., 256.])
    parser.add_argument('--haar-grids', type=int, nargs='+', default=[96, 128, 160])
    parser.add_argument('--output', type=Path, default=ROOT / 'original_centre_charge_receipt.json')
    args = parser.parse_args()
    result = run(spacing=args.spacing, reference_grid=args.reference_grid,
                 lambdas=tuple(args.lambdas), haar_grids=tuple(args.haar_grids))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), all_checks_pass=result['all_checks_pass'],
                          quantum_ground_computed=False, campaign_complete=False)))


if __name__ == '__main__':
    main()

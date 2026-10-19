#!/usr/bin/env python3
"""Actual occupied-sea 45-direction uniform SU(4) link Hessian.

Balanced opposite heavy/light bands at the common zero-origin selected saddle.
This is a static q=0 calculation, not SU(4) Gauss projection or a quantum phase.
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

from angular_photon_spectrum import brillouin_quadrature
from heavy_orbital_join import selected_saddle
from local_orbital_stability import scalar_paths
from orbital_gauge_locking import adjoint, gauge_mass, gradient_energy
from rotation_band import RotationBand
from sea_selection import h_from_couplings, matrix_unit_basis


SOURCE_FILES = ('su4_sea_backreaction.py', 'su4_sea_backreaction.md',
                'heavy_orbital_join.py', 'local_orbital_stability.py',
                'orbital_gauge_locking.py', 'rotation_band.py', 'sea_selection.py',
                'soft_budget.py', 'angular_photon_spectrum.py')


def symmetric(matrix):
    return (matrix.real + matrix.real.T) / 2


def spectrum(matrix) -> dict[str, Any]:
    values = np.linalg.eigvalsh(symmetric(matrix))
    tolerance = 128 * np.finfo(float).eps * max(1., float(np.linalg.norm(matrix, 2)))
    return dict(eigenvalues=values.tolist(), minimum=float(values[0]),
                maximum=float(values[-1]), negative=int(np.sum(values < -tolerance)),
                positive=int(np.sum(values > tolerance)),
                numerically_zero=int(np.sum(abs(values) <= tolerance)),
                sign_tolerance=tolerance)


def critical_prefactor(sea, bare, scale) -> dict[str, Any]:
    """Raw generalized spectrum first; clipping defines a nonnegative threshold."""
    values, vectors = np.linalg.eigh(symmetric(bare))
    if values[0] <= 0:
        raise ValueError('The full bare Higgs metric must be positive definite.')
    inverse_root = (vectors / np.sqrt(values)) @ vectors.T
    reduced = symmetric(inverse_root @ ((scale + 1) / scale * sea) @ inverse_root)
    raw = spectrum(reduced)
    return dict(raw_generalized_spectrum=raw,
                raw_generalized_minimum=raw['minimum'],
                Jcritical=max(0., -raw['minimum']),
                definition='max(0,-lambda_min(Mbase^-1/2 ((Lambda+1)/Lambda Hsea) Mbase^-1/2))',
                scope='Necessary and sufficient positive-definiteness threshold of this 45D static q=0 Hessian; strict J>Jcritical is sufficient, not a phase theorem.')


class UniformColourSea:
    """Stream actual 4x4 eigenspaces; only the projected 2x2 features are stacked."""

    def __init__(self, order, rule, band, frame, scale, chunk_size=1024):
        self.points, quadrature_weights = brillouin_quadrature(order, rule)
        self.weights = (np.full(len(self.points), 1 / len(self.points))
                        if quadrature_weights is None else quadrature_weights)
        if np.any(self.weights <= 0) or abs(float(self.weights.sum()) - 1) > 1e-12:
            raise ValueError('Require common positive normalized BZ weights.')
        self.order, self.rule = order, rule
        self.band, self.frame, self.scale = band, np.asarray(frame), scale
        self.chunk_size = chunk_size
        self.onsite, paths = scalar_paths(band)
        self.paths = []
        for displacement, scalars in paths:
            axes = np.flatnonzero(displacement)
            if len(axes) != 1 or displacement[axes[0]] <= 0:
                raise ValueError('This response requires the actual positive single-axis paths.')
            self.paths.append((np.asarray(displacement), int(axes[0]),
                               int(displacement[axes[0]]), np.asarray(scalars)))

    def chunks(self):
        for start in range(0, len(self.points), self.chunk_size):
            stop = min(start + self.chunk_size, len(self.points))
            yield self.points[start:stop], self.weights[start:stop]

    def hamiltonian(self, points, links=None, frame=None):
        frame = self.frame if frame is None else np.asarray(frame)
        h = np.broadcast_to(self.onsite * frame[0], (len(points), 4, 4)).copy()
        for displacement, axis, length, scalars in self.paths:
            orbital = np.einsum('m,mij->ij', scalars, frame)
            if links is not None:
                path_link = np.linalg.matrix_power(links[axis], length)
                orbital = (orbital @ path_link + path_link @ orbital) / 2
            phase = np.exp(1j * self.band.a * (points @ displacement))
            h += phase[:, None, None] * orbital
            h += phase.conj()[:, None, None] * orbital.conj().T
        return h

    def response(self, generators, coupling, frame=None, actual_heavy=False) -> dict[str, Any]:
        frame = self.frame if frame is None else np.asarray(frame)
        count = len(generators)
        dimension = 3 * count
        # Constants retain l for first derivatives and l^2 for contacts.
        first = []
        second = []
        generator_products = (generators[:, None] @ generators[None, :]
                              + generators[None, :] @ generators[:, None])
        for _, _, length, scalars in self.paths:
            orbital = np.einsum('m,mij->ij', scalars, frame)
            first.append(1j * coupling * length / 2
                         * (orbital @ generators + generators @ orbital))
            second.append(-coupling**2 * length**2 / 4
                          * (orbital @ generator_products + generator_products @ orbital))
        pair = np.zeros((dimension, dimension))
        fourier_projectors = np.zeros((len(self.paths), 4, 4), complex)
        density = np.zeros((4, 4), complex)
        heavy_density = np.zeros((4, 4), complex)
        baseline_light = baseline_heavy = trace_average = 0.
        min_gap = float('inf')
        hermiticity = complement_error = 0.
        for points, weights in self.chunks():
            h = self.hamiltonian(points, frame=frame)
            hermiticity = max(hermiticity, float(np.max(np.linalg.norm(
                h - h.conj().transpose(0, 2, 1), axis=(1, 2)))))
            energies, vectors = np.linalg.eigh(h)
            occupied, empty = vectors[:, :, :2], vectors[:, :, 2:]
            projector = occupied @ occupied.conj().transpose(0, 2, 1)
            density += np.einsum('p,pij->ij', weights, projector)
            baseline_light += float(weights @ energies[:, :2].sum(axis=1))
            trace_average += float(weights @ np.trace(h, axis1=1, axis2=2).real)
            gaps = energies[:, None, 2:] - energies[:, :2, None]
            min_gap = min(min_gap, float(gaps.min()))
            if gaps.min() <= 0:
                raise ValueError('The occupied/empty spectral gap closed at the expansion point.')
            # 45 x (batch * 2 * 2), not 45 x batch x 4 x 4 over the full BZ.
            features = np.zeros((dimension, len(points), 2, 2), complex)
            for path_number, (displacement, axis, _, _) in enumerate(self.paths):
                phase = np.exp(1j * self.band.a * (points @ displacement))
                fourier_projectors[path_number] += np.einsum(
                    'p,p,pij->ij', weights, phase, projector)
                derivative = (phase[:, None, None, None] * first[path_number][None]
                              + phase.conj()[:, None, None, None]
                              * first[path_number].conj().transpose(0, 2, 1)[None])
                projected = np.einsum('pio,paij,pje->apoe', occupied.conj(),
                                      derivative, empty, optimize=True)
                features[axis * count:(axis + 1) * count] += projected
            features *= np.sqrt(weights[:, None, None] / gaps)[None]
            flattened = features.reshape(dimension, -1)
            pair -= 2 * np.real(flattened @ flattened.conj().T)
            if actual_heavy:
                negative_energies, negative_vectors = np.linalg.eigh(-self.scale * h)
                heavy_occupied = negative_vectors[:, :, :2]
                heavy_projector = heavy_occupied @ heavy_occupied.conj().transpose(0, 2, 1)
                heavy_density += np.einsum('p,pij->ij', weights, heavy_projector)
                baseline_heavy += float(weights @ negative_energies[:, :2].sum(axis=1))
                complement_error = max(complement_error, float(np.max(np.linalg.norm(
                    projector + heavy_projector - np.eye(4), axis=(1, 2)))))
        contact = np.zeros_like(pair)
        for number, (_, axis, _, _) in enumerate(self.paths):
            block = 2 * np.einsum('ij,abji->ab', fourier_projectors[number], second[number]).real
            section = slice(axis * count, (axis + 1) * count)
            contact[section, section] += block
        contact, pair = symmetric(contact), symmetric(pair)
        result = dict(contact=contact, pair=pair, sea=contact + pair,
                      density=density, minimum_particle_hole_gap=min_gap,
                      baseline_light=baseline_light, trace_average=trace_average,
                      hermiticity_residual=hermiticity)
        if actual_heavy:
            result.update(heavy_density=heavy_density, baseline_heavy=baseline_heavy,
                          complementary_projector_residual=complement_error)
        return result

    def energies(self, links, alignment, frame=None) -> dict[str, float]:
        """Fresh spectra of h and the actual whole negative Hamiltonian -Lambda h."""
        frame = self.frame if frame is None else np.asarray(frame)
        light = heavy = trace = 0.
        maximum_pointwise_trace = 0.
        for points, weights in self.chunks():
            h = self.hamiltonian(points, links, frame)
            light += float(weights @ np.linalg.eigvalsh(h)[:, :2].sum(axis=1))
            heavy += float(weights @ np.linalg.eigvalsh(-self.scale * h)[:, :2].sum(axis=1))
            traces = np.trace(h, axis1=1, axis2=2).real
            trace += float(weights @ traces)
            maximum_pointwise_trace = max(maximum_pointwise_trace, float(np.max(abs(traces))))
        bare = sum(gradient_energy(frame, frame, link, alignment) for link in links)
        return dict(light=light, heavy=heavy, bare_higgs=bare,
                    total_sea_plus_higgs=light + heavy + bare,
                    integrated_trace=trace, maximum_pointwise_trace=maximum_pointwise_trace,
                    heavy_minus_Lambda_light=heavy - self.scale * light,
                    trace_identity_residual=abs(heavy - self.scale * light + self.scale * trace))


def links_for(direction, generators, coupling, step):
    coefficients = np.asarray(direction).reshape(3, len(generators))
    sources = np.einsum('ia,ajk->ijk', coefficients, generators)
    return np.array([expm(1j * coupling * step * source) for source in sources])


def noncommutativity(direction, generators):
    sources = np.einsum('ia,ajk->ijk', np.asarray(direction).reshape(3, -1), generators)
    return [float(np.linalg.norm(sources[i] @ sources[j] - sources[j] @ sources[i]))
            for i, j in ((0, 1), (1, 2), (2, 0))]


def finite_direction(sea, direction, generators, coupling, alignment, analytic,
                     contact, pair, scale, steps) -> dict[str, Any]:
    baseline = sea.energies(np.broadcast_to(np.eye(4), (3, 4, 4)), alignment)
    prediction = float(direction @ analytic @ direction)
    contact_prediction = float(direction @ contact @ direction)
    pair_prediction = float(direction @ pair @ direction)
    rows = []
    for step in steps:
        plus = sea.energies(links_for(direction, generators, coupling, step), alignment)
        minus = sea.energies(links_for(direction, generators, coupling, -step), alignment)
        differences = {key: (plus[key] + minus[key] - 2 * baseline[key]) / step**2
                       for key in ('light', 'heavy', 'bare_higgs')}
        measured = sum(differences.values())
        light_prediction = float(direction @ (contact + pair) @ direction)
        roundoff = float(256 * np.finfo(float).eps
                         * (abs(plus['light']) + abs(minus['light']) + abs(plus['heavy'])
                            + abs(minus['heavy']) + 2 * abs(baseline['light'])
                            + 2 * abs(baseline['heavy']) + 1) / step**2)
        rows.append(dict(step=step, plus=plus, minus=minus, component_second_differences=differences,
                         actual_second_difference=measured, analytic=prediction,
                         absolute_error=abs(measured - prediction), roundoff_allowance=roundoff,
                         light_analytic=light_prediction,
                         heavy_analytic_from_integrated_trace=scale * light_prediction,
                         actual_heavy_error=abs(differences['heavy'] - scale * light_prediction),
                         actual_heavy_minus_Lambda_light_second_difference=(
                             differences['heavy'] - scale * differences['light']),
                         omitted_contact_prediction=prediction - (scale + 1) * contact_prediction,
                         omitted_pair_prediction=prediction - (scale + 1) * pair_prediction,
                         omitted_contact_error=abs(measured - prediction + (scale + 1) * contact_prediction),
                         omitted_pair_error=abs(measured - prediction + (scale + 1) * pair_prediction)))
    coarse, fine = rows
    error_floor = max(fine['absolute_error'], fine['roundoff_allowance'])
    convergence_factor = max(.65, (1 + (steps[1] / steps[0])**2) / 2)
    convergence = (fine['absolute_error'] <= convergence_factor * coarse['absolute_error']
                   + fine['roundoff_allowance'])
    return dict(direction=direction.tolist(), pairwise_axis_commutator_norms=noncommutativity(direction, generators),
                analytic_contact=contact_prediction, analytic_pair=pair_prediction,
                analytic_total=prediction, baseline=baseline, steps=rows,
                coarse_to_fine_error_ratio=(coarse['absolute_error'] / fine['absolute_error']
                                            if fine['absolute_error'] > 0 else None),
                expected_truncation_error_ratio=(steps[0] / steps[1])**2,
                checks=dict(all_three_axes_noncommuting=min(noncommutativity(direction, generators)) > 1e-7,
                            nontrivial_actual_response=abs(fine['actual_second_difference']) > 20 * error_floor,
                            finite_difference_agrees=fine['absolute_error'] < max(
                                1e-5 * (1 + abs(prediction)), fine['roundoff_allowance']),
                            second_difference_converges=convergence,
                            actual_negative_heavy_agrees=fine['actual_heavy_error'] < max(
                                1e-5 * (1 + abs(fine['heavy_analytic_from_integrated_trace'])),
                                fine['roundoff_allowance']),
                            omitted_contact_resolved=fine['omitted_contact_error'] > 20 * error_floor,
                            omitted_pair_resolved=fine['omitted_pair_error'] > 20 * error_floor))


def gauss_mean_receipt(light, heavy, generators) -> dict[str, Any]:
    means = np.einsum('ij,aji->a', light + heavy, generators).real
    covariances = []
    for density in (light, heavy):
        products = generators[:, None] @ generators[None, :]
        first = np.einsum('ij,abji->ab', density, products).real
        second = np.einsum('ij,ajk,kl,bli->ab', density, generators, density, generators,
                           optimize=True).real
        covariances.append(symmetric(first - second))
    variance = covariances[0] + covariances[1]
    return dict(light_local_one_body_density_real=light.real.tolist(),
                light_local_one_body_density_imaginary=light.imag.tolist(),
                heavy_local_one_body_density_real=heavy.real.tolist(),
                heavy_local_one_body_density_imaginary=heavy.imag.tolist(),
                light_mean_number=float(np.trace(light).real),
                heavy_mean_number=float(np.trace(heavy).real),
                total_traceless_colour_mean=means.tolist(), mean_norm=float(np.linalg.norm(means)),
                local_colour_covariance=variance.tolist(), local_colour_variance_trace=float(np.trace(variance)),
                local_colour_covariance_spectrum=spectrum(variance),
                local_gauss_projection_performed=False,
                colour_variance_zero=bool(np.linalg.norm(variance) < 1e-12),
                meaning='Complementary unprojected opposite-band Slater seas cancel local mean traceless colour. Local colour fluctuates: this is a balanced mean constraint, not a Gauss eigenstate or quantum SU4 vacuum.')


def covariance_control(sea, frame, generators, coupling, j_prefactor, sea_hessian, bare) -> dict[str, Any]:
    unitary = expm(1j * np.einsum('a,aij->ij', .29 * np.cos(np.arange(15) * .73 + .4), generators))
    changed_frame = adjoint(frame, unitary)
    coordinates = 2 * np.einsum('aij,bji->ab', generators, adjoint(generators, unitary)).real
    rotation = np.kron(np.eye(3), coordinates)
    changed = sea.response(generators, coupling, changed_frame)
    changed_mass, _ = gauge_mass(changed_frame, generators, 1., coupling)
    changed_bare = np.kron(np.eye(3), symmetric(changed_mass))
    theta = np.sin(np.arange(45) * 1.17 + .63)
    theta /= np.linalg.norm(theta)
    links = links_for(theta, generators, coupling, .21)
    changed_links = adjoint(links, unitary)
    alignment = sea.scale * j_prefactor
    original = sea.energies(links, alignment)
    transformed = sea.energies(changed_links, alignment, changed_frame)
    absent = sea.energies(links, alignment, changed_frame)
    energy_errors = {key: abs(original[key] - transformed[key])
                     for key in ('light', 'heavy', 'bare_higgs', 'total_sea_plus_higgs')}
    points = sea.points[::max(1, len(sea.points) // 13)][:13]
    matrix_error = float(np.linalg.norm(sea.hamiltonian(points, changed_links, changed_frame)
                                      - adjoint(sea.hamiltonian(points, links), unitary)))
    hessian_error = float(np.linalg.norm(changed['sea'] - rotation @ sea_hessian @ rotation.T))
    bare_error = float(np.linalg.norm(changed_bare - rotation @ bare @ rotation.T))
    off_spin = float(np.linalg.norm(changed_frame[0] - frame[0]))
    return dict(adjoint_coordinates=coordinates.tolist(), off_spin_mass_field_change=off_spin,
                adjoint_orthogonality_residual=float(np.linalg.norm(coordinates @ coordinates.T - np.eye(15))),
                actual_hamiltonian_covariance_residual=matrix_error,
                actual_energy_covariance_errors=energy_errors,
                sea_Hessian_covariance_residual=hessian_error,
                bare_Hessian_covariance_residual=bare_error,
                original=original, transformed=transformed,
                absent_link_transform_energy_defect=abs(absent['total_sea_plus_higgs'] - original['total_sea_plus_higgs']),
                checks=dict(genuinely_off_spin_frame=off_spin > 1e-3,
                            actual_frame_and_link_energy_covariance=max(energy_errors.values()) < 1e-10 * (1 + abs(original['total_sea_plus_higgs'])),
                            actual_matrix_covariance=matrix_error < 1e-10,
                            full_45D_adjoint_Hessian_covariance=bool(hessian_error < 1e-10 * (1 + np.linalg.norm(sea_hessian))),
                            bare_adjoint_Hessian_covariance=bool(bare_error < 1e-10 * (1 + np.linalg.norm(bare))),
                            absent_link_transform_discriminates=abs(absent['total_sea_plus_higgs'] - original['total_sea_plus_higgs']) > 1e-7))


def coupling_unit_control(sea, bare, coupling, alternate_sea, alternate_bare,
                          alternate_coupling, threshold, alternate_threshold) -> dict[str, Any]:
    comparisons = (
        ('sea', sea / coupling**2, alternate_sea / alternate_coupling**2),
        ('bare', bare / coupling**2, alternate_bare / alternate_coupling**2),
        ('raw_generalized', np.array(threshold['raw_generalized_spectrum']['eigenvalues']),
         np.array(alternate_threshold['raw_generalized_spectrum']['eigenvalues'])))
    errors, relative, checks = {}, {}, {}
    for name, original, alternate in comparisons:
        errors[name] = float(np.linalg.norm(alternate - original))
        relative[name] = errors[name] / max(float(np.linalg.norm(original)), 1e-30)
        checks[f'g_squared_{name}_invariant'] = relative[name] < 1e-10
    return dict(alternate_g=alternate_coupling,
        alternate_raw_generalized_spectrum=alternate_threshold['raw_generalized_spectrum'],
        alternate_Jcritical=alternate_threshold['Jcritical'],
        absolute_Jcritical_change=abs(threshold['Jcritical'] - alternate_threshold['Jcritical']),
        scaled_sea_matrix_residual=errors['sea'], scaled_bare_matrix_residual=errors['bare'],
        raw_generalized_spectrum_residual=errors['raw_generalized'],
        relative_residuals=relative, checks=checks,
        method='Rebuild derivatives at a second g; enforce g^2-normalized sea/bare matrices and the raw generalized spectrum, not merely clipped thresholds.')


def analyse_case(order, args, generators):
    selected, frame, saddle = selected_saddle(order, args.light_ma, args.scale,
                                            args.kappa, heavy_ma=args.light_ma,
                                            quadrature=args.quadrature)
    band = RotationBand(a=1., m=args.light_ma, r=1., v=1., e=1., improved=True)
    sea = UniformColourSea(order, args.quadrature, band, frame, args.scale, args.chunk_size)
    response = sea.response(generators, args.g, actual_heavy=True)
    mass, _ = gauge_mass(frame, generators, 1., args.g)
    bare = np.kron(np.eye(3), symmetric(mass))
    total = (args.scale + 1) * response['sea'] + args.scale * args.j_prefactor * bare
    threshold = critical_prefactor(response['sea'], bare, args.scale)
    critical = threshold['Jcritical']
    safe = critical + max(.1, .05 * critical)
    safe_total = (args.scale + 1) * response['sea'] + args.scale * safe * bare
    alternate_coupling = 1.37 * args.g
    alternate = sea.response(generators, alternate_coupling)
    alternate_mass, _ = gauge_mass(frame, generators, 1., alternate_coupling)
    alternate_bare = np.kron(np.eye(3), symmetric(alternate_mass))
    alternate_threshold = critical_prefactor(alternate['sea'], alternate_bare, args.scale)
    units = coupling_unit_control(response['sea'], bare, args.g, alternate['sea'],
                                  alternate_bare, alternate_coupling, threshold, alternate_threshold)
    candidates = [np.sin(np.arange(45) * frequency + phase)
                  for frequency, phase in ((.71, .31), (1.17, .63), (1.91, .27))]
    candidates = [vector / np.linalg.norm(vector) for vector in candidates]
    # Chosen without any FD results: dense directions resolve both analytic pieces.
    scores = [min(abs(float(v @ response['contact'] @ v)), abs(float(v @ response['pair'] @ v)))
              for v in candidates]
    witness = int(np.argmax(scores))
    directions = [candidates[witness], candidates[(witness + 1) % len(candidates)]]
    finite = [finite_direction(sea, direction, generators, args.g, args.scale * args.j_prefactor,
                               total, response['contact'], response['pair'], args.scale, args.steps)
              for direction in directions]
    covariance = covariance_control(sea, frame, generators, args.g, args.j_prefactor,
                                    response['sea'], bare)
    mean = gauss_mean_receipt(response['density'], response['heavy_density'], generators)
    points = sea.points[::max(1, len(sea.points) // 17)][:17]
    mapping_error = float(np.linalg.norm(sea.hamiltonian(points) - h_from_couplings(band, points, frame)))
    selected_mapping_error = float(np.linalg.norm(sea.hamiltonian(points)
        - np.array([selected.h_from_couplings(p) for p in points])))
    checks = dict(actual_primitive_zero_link_mapping=mapping_error < 1e-10,
                  actual_selected_band_mapping=selected_mapping_error < 1e-10,
                  half_seas_are_actual_complementary_projectors=response['complementary_projector_residual'] < 1e-10,
                  balanced_mean_colour=mean['mean_norm'] < 1e-10,
                  not_zero_colour_variance=mean['local_colour_variance_trace'] > 1e-6,
                  explicit_safe_parameter_positive=spectrum(safe_total)['minimum'] > 0)
    checks.update(covariance['checks'])
    checks.update(units['checks'])
    for index, item in enumerate(finite):
        checks.update({f'direction_{index}_{key}': value for key, value in item['checks'].items()})
    receipt: dict[str, Any] = dict(
        order=order, quadrature=args.quadrature, points=len(sea.points),
        parameters=dict(light_ma=args.light_ma, heavy_ma=args.light_ma, Lambda=args.scale,
                        kappa=args.kappa, Jpref=args.j_prefactor, coupling_g=args.g,
                        a=1., bare_r=1., bare_v=1., improved=True),
        coordinate_order='axis-major: (x,T0..T14),(y,T0..T14),(z,T0..T14)',
        primitive_scalar_path_law=dict(onsite_scalar=sea.onsite,
            directed_paths=[dict(displacement=displacement.tolist(), axis=axis, length=length,
                                 scalars_real=scalars.real.tolist(), scalars_imaginary=scalars.imag.tolist())
                            for displacement, axis, length, scalars in sea.paths]),
        selected_gram=saddle['selected_gram'],
        selected_saddle_certificate=dict(bare_preferred_gram=0.,
            objective_saturation_residual=saddle['objective_saturation_residual'],
            normalized_stationarity_residual=saddle['normalized_stationarity_residual'],
            selected_light_mass=selected.m, selected_light_wilson=selected.r, selected_light_velocity=selected.v,
            interpretation='Common balanced shape selected by the reused unrestricted Gram saddle. Heavy microscopic law here is the whole negative -Lambda h, not the original same-sign sea occupation.'),
        bare_quartic_energy=float(.5 * args.scale * args.kappa * np.sum(np.asarray(saddle['selected_gram'])**2)),
        minimum_particle_hole_gap=response['minimum_particle_hole_gap'],
        weights=dict(minimum=float(sea.weights.min()), sum=float(sea.weights.sum())),
        sea_contact_Hessian=response['contact'].tolist(), sea_pair_Hessian=response['pair'].tolist(),
        single_light_sea_Hessian=response['sea'].tolist(),
        two_species_sea_Hessian=((args.scale + 1) * response['sea']).tolist(),
        bare_Higgs_Mbase=bare.tolist(), actual_bare_Higgs_Hessian=(args.scale * args.j_prefactor * bare).tolist(),
        total_Hessian=total.tolist(), sea_spectrum=spectrum(response['sea']),
        contact_spectrum=spectrum(response['contact']), pair_spectrum=spectrum(response['pair']),
        bare_Higgs_spectrum=spectrum(args.scale * args.j_prefactor * bare), total_spectrum=spectrum(total),
        threshold=threshold,
        supplied_Jpref_uniform_static_stable=spectrum(total)['minimum'] > 0,
        explicit_safe_scalar_hypothesis=dict(Jpref=safe, total_spectrum=spectrum(safe_total),
            interpretation='An externally chosen scalar J>Jcritical, not dynamically selected and not a finite-q/nonlinear/quantum stability certificate.'),
        coupling_unit_control=units,
        balanced_mean_constraint=mean, finite_direction_controls=finite,
        direction_selection=dict(rule='Two dense noncommuting fixed trigonometric candidates; first maximizes min(|contact|,|pair|) before any FD is evaluated.',
                                 candidate_scores=scores, selected_candidate=witness),
        whole_colour_covariance=covariance,
        baseline_actual_light_sea=response['baseline_light'], baseline_actual_negative_heavy_sea=response['baseline_heavy'],
        baseline_integrated_trace=response['trace_average'],
        baseline_heavy_scaling_residual=abs(response['baseline_heavy'] - args.scale * response['baseline_light']),
        complementary_projector_residual=response['complementary_projector_residual'],
        zero_link_mapping_residual=mapping_error, selected_band_mapping_residual=selected_mapping_error,
        checks={key: bool(value) for key, value in checks.items()})
    return receipt, frame, band


def identity_control(order, args, frame, band) -> dict[str, Any]:
    """Identity generators shift momentum; finite quadrature need not obey Ward exactly."""
    sea = UniformColourSea(order, args.quadrature, band, frame, args.scale, args.chunk_size)
    identity = np.eye(4)[None]
    response = sea.response(identity, 1.)
    direction = np.array([.37, -.51, .78]); direction /= np.linalg.norm(direction)
    theta = .173 * direction
    links = np.array([np.exp(1j * angle) * np.eye(4) for angle in theta])
    points = sea.points[::max(1, len(sea.points) // 19)][:19]
    mapping_error = float(np.linalg.norm(sea.hamiltonian(points, links)
                                        - h_from_couplings(band, points + theta / band.a, frame)))
    base = sea.energies(np.broadcast_to(np.eye(4), (3, 4, 4)), args.scale * args.j_prefactor)
    differences = []
    for step in args.steps:
        plus = sea.energies(links_for(direction, identity, 1., step), args.scale * args.j_prefactor)
        minus = sea.energies(links_for(direction, identity, 1., -step), args.scale * args.j_prefactor)
        measured = (plus['light'] + minus['light'] - 2 * base['light']) / step**2
        differences.append(dict(step=step, light_second_difference=measured,
            analytic=float(direction @ response['sea'] @ direction),
            absolute_error=abs(measured - float(direction @ response['sea'] @ direction)),
            bare_Higgs_second_difference=(plus['bare_higgs'] + minus['bare_higgs'] - 2 * base['bare_higgs']) / step**2))
    bare_identity = sum(gradient_energy(frame, frame, link, 1.) for link in links)
    return dict(order=order, quadrature=args.quadrature, points=len(sea.points),
                fixed_finest_selected_frame=True,
                single_sea_Hessian=response['sea'].tolist(), contact=response['contact'].tolist(), pair=response['pair'].tolist(),
                Hessian_spectrum=spectrum(response['sea']), Hessian_norm=float(np.linalg.norm(response['sea'])),
                actual_momentum_shift_mapping_residual=mapping_error,
                exact_identity_bare_Higgs_energy=bare_identity, finite_differences=differences,
                zero_target='Continuous normalized full-BZ integral only: a uniform scalar identity link shifts momentum, so its sea Hessian is zero. These weighted finite quadratures can have nonzero remainder.',
                finite_grid_exact_Ward_claim=False,
                checks=dict(actual_identity_momentum_shift=mapping_error < 1e-10,
                            identity_bare_Higgs_zero=bare_identity < 1e-24))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', nargs='+', type=int, default=[24, 32])
    parser.add_argument('--quadrature', choices=['midpoint', 'split_gauss'], default='midpoint')
    parser.add_argument('--light-ma', type=float, default=.2)
    parser.add_argument('--Lambda', dest='scale', type=float, default=64.)
    parser.add_argument('--kappa', type=float, default=1.)
    parser.add_argument('--Jpref', dest='j_prefactor', type=float, default=1.)
    parser.add_argument('--g', type=float, default=.7)
    parser.add_argument('--steps', nargs=2, type=float, default=[.005, .0025])
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if (args.orders != sorted(set(args.orders)) or min(args.orders) < 8
            or args.chunk_size < 1 or not all(np.isfinite(x) and x > 0
            for x in (args.light_ma, args.scale, args.kappa, args.g, *args.steps))
            or not np.isfinite(args.j_prefactor) or args.j_prefactor < 0
            or args.steps[0] <= args.steps[1]):
        parser.error('Require increasing distinct orders >=8, positive finite ma/Lambda/kappa/g/descending steps, nonnegative finite Jpref, and positive chunk size.')
    generators = matrix_unit_basis() / (2 * np.sqrt(2))
    generator_metric = np.einsum('aij,bji->ab', generators, generators).real
    rows = []
    finest_frame = finest_band = None
    for order in args.orders:
        row, finest_frame, finest_band = analyse_case(order, args, generators)
        rows.append(row)
    if finest_frame is None or finest_band is None:
        raise ValueError('Require at least one quadrature order.')
    identity_rows = [identity_control(order, args, finest_frame, finest_band) for order in args.orders]
    refinements = [dict(from_order=left['order'], to_order=right['order'],
                        sea_Hessian_change=float(np.linalg.norm(np.asarray(right['single_light_sea_Hessian'])
                                                               - np.asarray(left['single_light_sea_Hessian']))),
                        total_minimum_change=right['total_spectrum']['minimum'] - left['total_spectrum']['minimum'],
                        Jcritical_change=right['threshold']['Jcritical'] - left['threshold']['Jcritical'])
                   for left, right in zip(rows, rows[1:])]
    result = dict(status='balanced_opposite_band_occupied_sea_uniform_SU4_backreaction',
        scope='Full 45 real uniform static single-axis link directions; actual 4x4 half-sea contact and occupied-empty Gram, plus bare three-axis Higgs restoring.',
        microscopic_law=dict(light='Endpoint-averaged primitive scalar_paths with positive light whole h.',
            heavy='EM-neutral SU4 fundamental, whole H=-Lambda*h_light, same bare m/r/v/improved and same frame, independently occupied lowest two levels.',
            bare_potential='Lambda*kappa*||G||_F^2/2, unique bare minimum G=0.',
            uniform_links='Ui=exp(i*g*sum_a theta_i,a Ta); an actual axis path of length l uses ordered Ui^l.',
            sea_energy_identity='Integrated Tr h=0, hence Eheavy=Lambda*Elight and combined half-sea energy=(Lambda+1)*Elight. Individual non-Abelian h(p,U) need not be traceless.'),
        limitations=['Static q=0 Hessian only: no finite-q, global/nonlinear stability, dressed quantum gauge gap, or 3D Higgs phase.',
            'Complementary seas cancel local mean traceless colour but are not Gauss projected and do not have zero local colour variance.',
            'The externally selected safe Jpref is an explicit scalar parameter hypothesis, not dynamical selection.',
            'This occupied global soft-model saddle is distinct from the J=0 one-cell quantum Gauss model; no domain gluing, U1 photon claim, or canonical full-QED claim.'],
        generator_normalization_residual=float(np.linalg.norm(generator_metric - .5 * np.eye(15))),
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in SOURCE_FILES},
        rows=rows, quadrature_refinement=refinements, scalar_identity_U1_controls=identity_rows,
        scalar_identity_refinement=dict(fixed_frame_from_order=args.orders[-1],
            Hessian_norms=[row['Hessian_norm'] for row in identity_rows],
            successive_norm_ratios=[right['Hessian_norm'] / left['Hessian_norm'] if left['Hessian_norm'] > 0 else None
                                   for left, right in zip(identity_rows, identity_rows[1:])],
            interpretation='Measured weighted-quadrature approach to the continuous BZ zero; neither exact finite-grid Ward nor monotonic convergence is assumed.'),
        all_numerical_controls_pass=all(all(row['checks'].values()) for row in rows + identity_rows),
        all_supplied_Jpref_static_stable=all(row['supplied_Jpref_uniform_static_stable'] for row in rows))
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    # A real negative supplied-J Hessian is a reported physical outcome, not an implementation failure.
    return 0 if result['all_numerical_controls_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

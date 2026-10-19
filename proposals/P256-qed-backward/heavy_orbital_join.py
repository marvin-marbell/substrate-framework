#!/usr/bin/env python3
"""Neutral heavy sea selects zero-origin orbitals used by the light photon consumer.

Uniform two-sea saddle and harmonic hierarchy, not a quantum SU4 Higgs run.
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

from angular_photon import axis_scattering
from angular_photon_spectrum import ContinuousSea, brillouin_quadrature
from orbital_gauge_locking import adjoint, analytic_masses, gauge_mass, pauli_covariance
from rotation_band import RotationBand
from sea_selection import SELECTED, coefficients, h_from_couplings, matrix_unit_basis, rank_report
from soft_budget import select_gram
from soft_quantum import angular_coefficient, current_velocity, ir_resolution, rest_coefficient


def radial_derivatives(x, weights, g):
    """Sea gradient/Hessian in (gm,gs); three spatial entries vary together."""
    squared = np.column_stack((x[:, 0]**2, np.sum(x[:, 1:]**2, axis=1)))
    radius = np.sqrt(squared @ np.array([g[0, 0], g[1, 1]]))
    gradient = -np.einsum('p,pi->i', weights / radius, squared)
    hessian = .5 * np.einsum('p,pi,pj->ij', weights / radius**3, squared, squared)
    return gradient, hessian


def _local_centre_population(xh, xl, weights, g):
    """Unprojected Slater sea: half filling is not a fixed local occupation."""
    species = {}
    distributions = []
    for name, x in (('heavy', xh), ('light', xl)):
        radius = np.sqrt(np.einsum('pm,mn,pn->p', x, g, x))
        b = float(np.dot(weights, x[:, 0] * np.sqrt(g[0, 0]) / radius))
        occupations = [(1 + b) / 2] * 2 + [(1 - b) / 2] * 2
        probability = np.array([1.])
        for occupation in occupations:
            probability = np.convolve(probability, [1 - occupation, occupation])
        distributions.append(probability)
        species[name] = dict(mean_local_number=2.,
                             local_density_projector_beta_average=b,
                             local_number_variance=1 - b**2,
                             local_number_probabilities=probability.tolist())
    total = np.convolve(*distributions)
    return dict(species=species, total_local_number_probabilities=total.tolist(),
                local_centre_probabilities=[float(total[k::4].sum()) for k in range(4)],
                local_centre_neutral_probability=float(total[::4].sum()),
                local_gauss_projection_performed=False,
                meaning='Global half filling and formal wedge2 x wedge2 centre bookkeeping do not make the hopping sea a local centre eigenstate.')


def selected_saddle(order, light_ma, scale, stiffness, heavy_ma=.8,
                    quadrature='midpoint') -> tuple[RotationBand, np.ndarray, dict[str, Any]]:
    """Positive common weights and zero bare Gram preference; both seas vary."""
    if (order < 8 or not all(np.isfinite(value) and value > 0
                             for value in (light_ma, scale, stiffness, heavy_ma))):
        raise ValueError('Require order >=8 and finite positive masses, Lambda and kappa.')
    points, quadrature_weights = brillouin_quadrature(order, quadrature)
    weights = (np.full(len(points), 1 / len(points)) if quadrature_weights is None
               else quadrature_weights)
    heavy = RotationBand(a=1., m=heavy_ma, r=1., v=1., e=0., improved=True)
    light = RotationBand(a=1., m=light_ma, r=1., v=1., e=1., improved=True)
    xh, xl = coefficients(heavy, points), coefficients(light, points)
    x = np.concatenate((xh, xl))
    joined_weights = np.concatenate((scale / (scale + 1) * weights,
                                     1 / (scale + 1) * weights))
    minimum = select_gram(x, stiffness * scale / (scale + 1), 0., weights=joined_weights)
    heavy_minimum = select_gram(xh, stiffness, 0., weights=weights)
    g, gh = minimum['G'], heavy_minimum['G']
    frame = np.sqrt(np.diag(g))[:, None, None] * SELECTED
    selected = RotationBand(a=1., m=float(light_ma * np.sqrt(g[0, 0])),
                            r=float(np.sqrt(g[0, 0])), v=float(np.sqrt(g[1, 1])),
                            e=1., improved=True)
    heavy_selected_gap = float(scale * heavy_ma * np.sqrt(g[0, 0]))
    # Actual matrix eigenvalues of both local scalar-coupling laws, not merely
    # the radius expression used to solve the convex lower bound.
    heavy_eigenvalues = np.linalg.eigvalsh(h_from_couplings(heavy, points, frame))
    light_eigenvalues = np.linalg.eigvalsh(h_from_couplings(light, points, frame))
    heavy_sea_energy = float(scale * np.dot(weights, heavy_eigenvalues[:, :2].sum(axis=1)))
    light_sea_energy = float(np.dot(weights, light_eigenvalues[:, :2].sum(axis=1)))
    potential = float(.5 * scale * stiffness * np.sum(g**2))
    energy = heavy_sea_energy + light_sea_energy + potential
    lower_bound = float((scale + 1) * minimum['value'])
    light_gradient_at_heavy, _ = radial_derivatives(xl, weights, gh)
    _, heavy_hessian = radial_derivatives(xh, weights, gh)
    heavy_hessian += stiffness * np.diag([1., 3.])
    first_order = -np.linalg.solve(heavy_hessian, light_gradient_at_heavy) / scale
    # Full Frobenius Gram norm: the radial spatial derivative represents 3 entries.
    feedback_bound = float(np.sqrt(light_gradient_at_heavy[0]**2
                                  + light_gradient_at_heavy[1]**2 / 3) / (scale * stiffness))
    difference = float(np.linalg.norm(g - gh))
    _, hh = radial_derivatives(xh, weights, g)
    _, hl = radial_derivatives(xl, weights, g)
    radial_hessian = scale * (hh + stiffness * np.diag([1., 3.])) + hl
    probes = points[::max(1, len(points) // 8)][:8]
    mapping_error = max(float(np.linalg.norm(matrix - selected.h_from_couplings(p)))
                        for matrix, p in zip(h_from_couplings(light, probes, frame), probes))
    clifford_error = max(float(np.linalg.norm(frame[i] @ frame[j] + frame[j] @ frame[i]
                                             - 2 * g[i, j] * np.eye(4)))
                         for i in range(4) for j in range(4))
    rank = rank_report(x)
    return selected, frame, dict(
        order=order, quadrature=quadrature, quadrature_points=len(points),
        Lambda=scale, kappa=stiffness, bare_light_ma=light_ma, bare_heavy_ma=heavy_ma,
        bare_preferred_gram=0., bare_unique_minimum='All sixty real adjoint orbital fields zero',
        species=dict(light_U1_charge=-1, heavy_U1_charge=0,
                     light_SU4_representation='fundamental', heavy_SU4_representation='fundamental',
                     light_mean_filling=2, heavy_mean_filling=2,
                     fixed_half_filled_cell_representation_centre_charge_mod_four=0),
        unprojected_local_centre_population=_local_centre_population(xh, xl, weights, g),
        common_weight_minimum=float(weights.min()), common_weight_sum=float(weights.sum()),
        joined_weight_sum=float(joined_weights.sum()), selected_gram=g.tolist(),
        heavy_only_gram=gh.tolist(), heavy_only_energy=float(heavy_minimum['value']),
        light_feedback_gram_norm=difference, strong_convexity_feedback_bound=feedback_bound,
        first_order_feedback_radial=first_order.tolist(),
        first_order_feedback_error=float(np.linalg.norm(
            np.array([g[0, 0] - gh[0, 0], g[1, 1] - gh[1, 1]]) - first_order)),
        objective=energy, exact_gram_lower_bound=lower_bound,
        heavy_sea_energy=heavy_sea_energy, light_sea_energy=light_sea_energy,
        bare_quartic_energy=potential, objective_saturation_residual=abs(energy - lower_bound),
        normalized_stationarity_residual=minimum['gradient_residual'],
        objective_stationarity_residual=(scale + 1) * minimum['gradient_residual'],
        quadratic_design=rank, metric_clifford_residual=clifford_error,
        selected_light_mass=selected.m, selected_light_velocity=selected.v,
        selected_light_wilson=selected.r, selected_heavy_one_particle_gap=heavy_selected_gap,
        same_local_link_mapping_residual=mapping_error,
        uniform_radial_gram_hessian=radial_hessian.tolist(),
        uniform_radial_gram_hessian_per_Lambda=(radial_hessian / scale).tolist(),
        checks=dict(zero_origin_positive_selected_metric=float(np.diag(g).min()) > 0,
                    positive_normalized_common_weights=bool(np.all(weights > 0))
                        and abs(float(weights.sum()) - 1) < 1e-12,
                    exact_unrestricted_gram_certificate=rank['quadratic_rank'] == 10
                        and minimum['gradient_residual'] < 1e-10,
                    actual_two_sea_energy_saturates=abs(energy - lower_bound) < 1e-10 * (scale + 1),
                    same_selected_local_light_law=mapping_error < 1e-10,
                    metric_clifford_saturation=clifford_error < 1e-10,
                    finite_light_feedback_retained=difference > 1e-12,
                    feedback_obeys_strong_convexity=difference <= feedback_bound + 1e-10,
                    positive_radial_curvature=float(np.linalg.eigvalsh(radial_hessian)[0]) > 0))


def higgs_receipt(frame, light_ma, scale, alignment_prefactor, coupling,
                  electric_prefactor) -> dict[str, Any]:
    """Exact 15x15 covariant bare restoring matrix, not a dressed quantum gap."""
    basis = matrix_unit_basis()
    generators = basis / (2 * np.sqrt(2))
    alignment = alignment_prefactor * scale
    electric = electric_prefactor / np.sqrt(scale)
    mass, _ = gauge_mass(frame, generators, alignment, coupling)
    eigenvalues = np.linalg.eigvalsh(mass)
    g = np.einsum('mij,nji->mn', frame, frame).real / 4
    predicted = np.sort([row['mass_squared'] for row in analytic_masses(np.diag(g), alignment, coupling)])
    unitary = expm(1j * np.einsum('a,aij->ij', .23 * np.cos(np.arange(15) + .9), basis))
    changed_mass, _ = gauge_mass(adjoint(frame, unitary), generators, alignment, coupling)
    coordinates = 2 * np.einsum('aij,bji->ab', generators, adjoint(generators, unitary)).real
    covariance_error = float(np.linalg.norm(changed_mass - coordinates @ mass @ coordinates.T))
    # Exercise the existing raw two-word Pauli-code covariance consumer for the
    # charged light law. This is an external-link finite fiber, not Gauss projection.
    covariance = pauli_covariance(adjoint(frame, unitary), basis,
                                  RotationBand(a=1., m=light_ma, e=1., improved=True))
    return dict(J=alignment, SU4_coupling=coupling, U_E4=electric,
                shape_inertia=scale**-.5, bare_higgs_matrix_real=mass.real.tolist(),
                bare_higgs_matrix_imaginary=mass.imag.tolist(),
                bare_higgs_eigenvalues=eigenvalues.tolist(),
                harmonic_gauge_frequencies=np.sqrt(electric * eigenvalues).tolist(),
                harmonic_gauge_variances=(.5 * np.sqrt(electric / eigenvalues)).tolist(),
                analytic_spectrum_residual=float(np.linalg.norm(eigenvalues - predicted)),
                adjoint_covariance_residual=covariance_error,
                light_external_link_pauli_covariance=covariance,
                scope='Bare quadratic restoring and finite external-link covariance only; no dressed gap or SU4 Gauss vacuum simulation',
                checks=dict(positive_all_fifteen=float(eigenvalues[0]) > 0,
                            exact_higgs_spectrum=float(np.linalg.norm(eigenvalues - predicted)) < 1e-10 * scale,
                            adjoint_higgs_covariance=covariance_error < 1e-10 * scale,
                            actual_light_pauli_covariance=covariance['actual_pauli_covariance_residual'] < 1e-9,
                            absent_link_transform_control=covariance['absent_link_transform_defect'] > 1e-3))


def joined_case(order, light_ma, scale, stiffness=1., heavy_ma=.8, momentum=.4,
                quadrature='midpoint', alignment_prefactor=1., coupling=.7,
                electric_prefactor=1.) -> dict[str, Any]:
    if not all(np.isfinite(value) and value > 0
               for value in (momentum, alignment_prefactor, coupling, electric_prefactor)):
        raise ValueError('Require finite positive momentum, J prefactor, coupling and U_E4 prefactor.')
    selected, frame, saddle = selected_saddle(order, light_ma, scale, stiffness, heavy_ma, quadrature)
    # The neutral heavy species is never supplied to ContinuousSea: its exact
    # microscopic U1 source is zero, not a suppressed response approximation.
    probe = ContinuousSea(order, band=selected, quadrature=quadrature)
    infrared = probe.infrared()
    chi_b = 1 / .09
    chi_e = (chi_b + infrared['magnetic']) / selected.v**2 - infrared['electric']
    sea = ContinuousSea(order, band=selected, chi_e=chi_e, chi_b=chi_b, quadrature=quadrature)
    a = selected.m  # Physical m=1 is only the energy/length unit conversion.
    scattering = axis_scattering(sea, q=a * momentum, a=a)
    physical = RotationBand(a=a, m=1., r=selected.r, v=selected.v, e=1., improved=True)
    vi, initial = current_velocity(physical, np.zeros(3))
    vf, final = current_velocity(physical, np.array(scattering['electron_recoil']))
    speed = float(np.sqrt((chi_b + infrared['magnetic']) / (chi_e + infrared['electric'])))
    charge = float(1 / np.sqrt(chi_e + infrared['electric']))
    coefficient = float(angular_coefficient(charge, vi, vf, speed))
    independent = rest_coefficient(charge / np.sqrt(speed), float(np.linalg.norm(vf) / speed))
    resolution = ir_resolution(coefficient, .01, .001)
    incoming = scattering['incoming_photon']['branches'][0]
    outgoing = scattering['outgoing_photon']['branches'][0]
    source = float(np.sqrt(incoming['source'] * outgoing['source']))
    mass_ref = scattering['mass'] / selected.v
    omega_ref = incoming['energy'] / a / selected.v
    charge_ref = source / np.sqrt(selected.v)
    ratio = 1 / (1 + omega_ref / mass_ref)
    alpha = charge_ref**2 / (4 * np.pi)
    reference_rate = float(alpha**2 / (2 * mass_ref**2) * ratio**2 * (ratio + 1 / ratio - 1))
    higgs = higgs_receipt(frame, light_ma, scale, alignment_prefactor, coupling, electric_prefactor)
    checks = dict(saddle['checks'])
    checks.update(higgs['checks'])
    checks.update(light_only_photon_stiffness_positive=chi_e > 0,
                  analytic_IR_speed_equals_selected_light=abs(speed / selected.v - 1) < 1e-12,
                  actual_primitive_axis_consumer=all(scattering['checks'].values()),
                  same_primitive_current_soft_coefficient=abs(coefficient - independent) < 1e-12)
    return dict(saddle=saddle, higgs=higgs, physical_mass_unit=1., physical_a=a,
                matching=dict(chi_e=chi_e, chi_b=chi_b, light_only_IR_response=infrared,
                              heavy_U1_response_exactly_zero=True, photon_speed=speed,
                              selected_light_speed=selected.v, IR_velocity_mismatch=speed / selected.v - 1,
                              interpretation='Explicitly tuned relativistic critical surface, not dynamical velocity locking'),
                scattering=scattering,
                matter_speed_unit_reference=dict(mass=mass_ref, omega=omega_ref, charge=charge_ref,
                                                 differential_rate=reference_rate),
                relative_rate_to_reference=scattering['differential_rate'] / reference_rate - 1,
                leading_soft=dict(initial_current=initial, final_current=final,
                                  canonical_charge=charge, photon_speed=speed,
                                  angular_coefficient=coefficient, independent_rest_coefficient=independent,
                                  coefficient_error=abs(coefficient - independent), inclusive_resolution=resolution,
                                  inclusive_differential_rate=scattering['differential_rate']
                                      * resolution['leading_soft_veto_factor']),
                checks={key: bool(value) for key, value in checks.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', nargs='+', type=int, default=[24, 32])
    parser.add_argument('--mass-ratios', nargs='+', type=float, default=[.2])
    parser.add_argument('--lambdas', nargs='+', type=float, default=[1., 8., 64.])
    parser.add_argument('--heavy-ma', type=float, default=.8)
    parser.add_argument('--kappa', type=float, default=1.)
    parser.add_argument('--momentum', type=float, default=.4)
    parser.add_argument('--quadrature', choices=['midpoint', 'split_gauss'], default='midpoint')
    parser.add_argument('--alignment-prefactor', type=float, default=1.)
    parser.add_argument('--su4-coupling', type=float, default=.7)
    parser.add_argument('--electric-prefactor', type=float, default=1.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.orders != sorted(set(args.orders)) or args.lambdas != sorted(set(args.lambdas)):
        parser.error('Orders and Lambdas must each be distinct and increasing.')
    rows = [joined_case(order, ma, scale, args.kappa, args.heavy_ma, args.momentum,
                        args.quadrature, args.alignment_prefactor, args.su4_coupling, args.electric_prefactor)
            for ma in args.mass_ratios for order in args.orders for scale in args.lambdas]
    hierarchy = []
    for ma in args.mass_ratios:
        for order in args.orders:
            sequence = [row for row in rows if row['saddle']['order'] == order
                        and row['saddle']['bare_light_ma'] == ma]
            hierarchy.append(dict(order=order, bare_light_ma=ma, Lambda=args.lambdas,
                                  metric_distance_to_heavy_only=[row['saddle']['light_feedback_gram_norm'] for row in sequence],
                                  Lambda_times_metric_distance=[row['saddle']['Lambda'] * row['saddle']['light_feedback_gram_norm'] for row in sequence],
                                  first_order_feedback_errors=[row['saddle']['first_order_feedback_error'] for row in sequence]))
    refinements = []
    for ma in args.mass_ratios:
        for scale in args.lambdas:
            sequence = [row for row in rows if row['saddle']['Lambda'] == scale
                        and row['saddle']['bare_light_ma'] == ma]
            refinements.append(dict(bare_light_ma=ma, Lambda=scale, orders=args.orders,
                                    selected_metrics=[row['saddle']['selected_gram'] for row in sequence],
                                    rates=[row['scattering']['differential_rate'] for row in sequence],
                                    inclusive_rates=[row['leading_soft']['inclusive_differential_rate'] for row in sequence],
                                    adjacent_metric_changes=[float(np.linalg.norm(np.array(right['saddle']['selected_gram'])
                                                                                - np.array(left['saddle']['selected_gram'])))
                                                             for left, right in zip(sequence, sequence[1:])],
                                    adjacent_relative_rate_changes=[right['scattering']['differential_rate']
                                                                    / left['scattering']['differential_rate'] - 1
                                                                    for left, right in zip(sequence, sequence[1:])]))
    sources = ('heavy_orbital_join.py', 'soft_budget.py', 'orbital_gauge_locking.py',
               'angular_photon.py', 'angular_photon_spectrum.py', 'soft_quantum.py',
               'rotation_band.py', 'sea_selection.py', 'pauli_compton.py')
    result = dict(status='neutral_heavy_zero_origin_orbital_saddle_same_light_photon_consumer',
                  approximation='Uniform two-species half-filled saddle; Gaussian light photon response; primitive tree Compton and leading-soft inclusive resolution',
                  limitations='No continuous SU4 Gauss projection, quantum 3D Higgs phase, dressed SU4 gap, arbitrary-fluctuation +/-m theorem, or physical-spin claim',
                  source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in sources},
                  rows=rows, heavy_only_limit=hierarchy, quadrature_refinement=refinements,
                  all_checks_pass=all(all(row['checks'].values()) for row in rows))
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

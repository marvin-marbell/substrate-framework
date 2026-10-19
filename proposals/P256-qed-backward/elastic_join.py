#!/usr/bin/env python3
"""Soft-selected orbital law, same-sea dressed photons, primitive angular rate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from angular_photon import axis_scattering
from angular_photon_spectrum import ContinuousSea, brillouin_quadrature
from rotation_band import RotationBand
from sea_selection import SELECTED, coefficients, h_from_couplings, rank_report
from soft_budget import select_gram
from soft_quantum import angular_coefficient, current_velocity, ir_resolution, quantum_mode, rest_coefficient


def joined_case(order: int, bare_ma: float, stiffness: float = 1., momentum: float = .4,
                quadrature: str = 'midpoint') -> dict:
    bare = RotationBand(a=1., m=bare_ma, e=1., improved=True)
    points, weights = brillouin_quadrature(order, quadrature)
    x = coefficients(bare, points)
    minimum = select_gram(x, stiffness, 1., weights=weights)
    g = minimum['G']
    selected = RotationBand(a=1., m=float(bare_ma * np.sqrt(g[0, 0])),
                            r=float(np.sqrt(g[0, 0])), v=float(np.sqrt(g[1, 1])),
                            e=1., improved=True)
    # The same thermodynamic quadrature selected the fields and now supplies
    # the photon channels. No L3 selection is silently reused as an infinite sea.
    probe_sea = ContinuousSea(order, band=selected, quadrature=quadrature)
    # Keep the MATCHING physical momentum/mass fixed across cutoff sequences.
    matching_momentum = .04 * selected.m
    reference = probe_sea.spectrum(np.array([matching_momentum, 0., 0.]))
    basis = reference.basis
    magnetic = float(np.trace(basis.T @ reference.static @ basis) / 2
                     / np.dot(reference.khat, reference.khat))
    electric = float(np.trace(basis.T @ reference.spectral(0., 'derivative') @ basis) / 2)
    chi_b = 1 / .09
    chi_e = (chi_b + magnetic) / selected.v**2 - electric
    # Make a NEW model: changing stiffness after a pole cache is populated
    # would mix physical models and invalidate the state normalization.
    sea = ContinuousSea(order, band=selected, chi_e=chi_e, chi_b=chi_b, quadrature=quadrature)
    # Physical mass one is only a length/energy unit choice; the selected
    # dimensionless gap is retained explicitly in this receipt.
    a = selected.m
    q = a * momentum
    scattering = axis_scattering(sea, q, a)
    physical_band = RotationBand(a=a, m=1., r=selected.r, v=selected.v, e=1., improved=True)
    vi, initial_current = current_velocity(physical_band, np.zeros(3))
    vf, final_current = current_velocity(physical_band, np.array(scattering['electron_recoil']))
    infrared = sea.infrared()
    ir_charge = float(1 / np.sqrt(chi_e + infrared['electric']))
    ir_speed = float(np.sqrt((chi_b + infrared['magnetic'])
                             / (chi_e + infrared['electric'])))
    coefficient = float(angular_coefficient(ir_charge, vi, vf, ir_speed))
    independent_coefficient = rest_coefficient(ir_charge / np.sqrt(ir_speed),
                                              float(np.linalg.norm(vf) / ir_speed))
    resolution = ir_resolution(coefficient, .01, .001)
    leading_soft = dict(initial_current=initial_current, final_current=final_current,
                        local_photon_speed=ir_speed, local_canonical_charge=ir_charge,
                        infrared_reference=infrared,
                        infrared_velocity_mismatch=ir_speed / selected.v - 1,
                        angular_coefficient=coefficient, independent_rest_coefficient=independent_coefficient,
                        coefficient_error=abs(coefficient - independent_coefficient),
                        quantum_oscillator=quantum_mode(ir_charge, vi, vf, ir_speed),
                        inclusive_resolution=resolution,
                        inclusive_differential_rate=scattering['differential_rate']
                            * resolution['leading_soft_veto_factor'])
    frame = np.sqrt(np.diag(g))[:, None, None] * SELECTED
    probe_points = points[::max(1, len(points) // 8)][:8]
    mapping_error = max(float(np.linalg.norm(matrix - selected.h_from_couplings(p)))
                        for matrix, p in zip(h_from_couplings(bare, probe_points, frame), probe_points))
    incoming = scattering['incoming_photon']['branches'][0]
    outgoing = scattering['outgoing_photon']['branches'][0]
    source_geometric = float(np.sqrt(incoming['source'] * outgoing['source']))
    # Comparator in the matter-speed natural units: t'=v*t,
    # m'=m/v, omega'=omega/v, e'=e/sqrt(v). This is a reference,
    # not the finite-model recoil or the phase-space denominator.
    mass_reference = scattering['mass'] / selected.v
    omega_reference = incoming['energy'] / a / selected.v
    e_reference = source_geometric / np.sqrt(selected.v)
    ratio = 1 / (1 + omega_reference / mass_reference)
    alpha = e_reference**2 / (4 * np.pi)
    qed_reference = float(alpha**2 / (2 * mass_reference**2) * ratio**2
                          * (ratio + 1 / ratio - 1))
    checks = dict(orbital_metric_dynamically_selected=minimum['gradient_residual'] < 1e-7,
                  selected_gap_positive=selected.m > 0,
                  selected_isotropic_velocity_positive=selected.v > 0,
                  physical_current_same_reframed_law=mapping_error < 1e-10,
                  photon_stiffness_positive=chi_e > 0,
                  full_primitive_angular_consumer=all(scattering['checks'].values()),
                  selection_quadratic_design_rank_ten=rank_report(x)['quadratic_rank'] == 10,
                  same_current_soft_integral=abs(coefficient - independent_coefficient) < 1e-12)
    return dict(order=order, quadrature=quadrature, quadrature_points=len(points),
                bare_ma=bare_ma, elastic_stiffness=stiffness,
                selected_gram=g.tolist(), orbital_stationary_residual=minimum['gradient_residual'],
                selected_dimensionless_mass=selected.m, selected_velocity=selected.v,
                selected_wilson_coefficient=selected.r, physical_mass_unit=1.,
                physical_a=a, same_link_mapping_residual=mapping_error,
                matching=dict(momentum=matching_momentum, physical_momentum_over_mass=.04,
                              DeltaChi_magnetic=magnetic,
                              DeltaChi_electric=electric, chi_e=chi_e, chi_b=chi_b,
                              microscopic_U_E=1 / chi_e, microscopic_K=chi_b,
                              code_Delta=3 * chi_b, code_defect_margin=chi_b),
                matter_speed_unit_reference=dict(mass=mass_reference, omega=omega_reference,
                                                 charge=e_reference, differential_rate=qed_reference),
                relative_rate_to_reference=scattering['differential_rate'] / qed_reference - 1,
                scattering=scattering, leading_soft=leading_soft, checks=checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', nargs='+', type=int, default=[24, 32])
    parser.add_argument('--mass-ratios', nargs='+', type=float, default=[.4, .2])
    parser.add_argument('--quadrature', choices=['midpoint', 'split_gauss'], default='midpoint')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = [joined_case(order, ma, quadrature=args.quadrature)
            for ma in args.mass_ratios for order in args.orders]
    result = dict(status='soft_selected_matter_and_same_sea_photon_primitive_nonforward_rate',
                  approximation='uniform orbital sea saddle, Gaussian rotor plus quadratic sea, tree electron response with dressed legs',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=rows, all_checks_pass=all(all(row['checks'].values()) for row in rows))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

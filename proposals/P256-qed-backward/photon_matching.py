#!/usr/bin/env python3
"""A positive-residue photon pole from the actual primitive filled-sea spectrum.

The Gaussian rotor plus quadratic sea-response approximation is explicit.
Bare coefficients are microscopic Hamiltonian inputs, not pure-gauge receipts.
"""
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
from scipy.optimize import brentq
from vacuum_limit import transverse_spectrum
from rotation_band import RotationBand


def match_photon(length: int, band: RotationBand, mode: int = 1, *,
                 electric_coefficient: float, magnetic_coefficient: float,
                 polarization_axis: int = 1) -> dict:
    """Pole for H=UE1 E²/2+K1 curl(theta)²/2; epsilon=1/UE1.

    band owns physical a,m,r,v,e and its hopping law. This is a compact-angle
    Gaussian/quadratic-sea kernel, not a compact-phase existence assertion.
    """
    if not all(np.isfinite(x) and x > 0 for x in (electric_coefficient, magnetic_coefficient)):
        raise ValueError('Finite positive Hamiltonian UE1 and K1 are required.')
    epsilon = 1 / electric_coefficient
    spectrum = transverse_spectrum(length, band, mode, polarization_axis=polarization_axis)
    channels = spectrum['channels']
    norm = spectrum['cells'] / 2
    contact = spectrum['contact'] / norm
    khat = float(2 * np.sin(spectrum['qangle'] / 2))
    pair_gap_squared = min(float(gaps.min())**2 for gaps, _ in channels)
    pair_zero = -2 * sum(float(np.sum(weights / gaps)) for gaps, weights in channels) / norm
    curvature = contact + pair_zero
    electric = 2 * sum(float(np.sum(weights / gaps**3)) for gaps, weights in channels) / norm
    spatial = magnetic_coefficient * khat**2 + curvature
    if spatial <= 0:
        raise ValueError('This probe has no positive static transverse restoring force.')
    low_squared = spatial / (epsilon + electric)
    if low_squared >= pair_gap_squared:
        raise ValueError('The sufficient below-pair-gap pole bound is not satisfied.')

    def inverse(squared: float) -> float:
        # Remove the linear electric term analytically. At the certified
        # upper endpoint the first term is exactly zero, preserving the
        # bracket even when the nonlinear sea correction is below its ulp.
        nonlinear = 2 * squared**2 * sum(
            float(np.sum(weights / (gaps**3 * (gaps**2 - squared))))
            for gaps, weights in channels) / norm
        return spatial * (1 - squared / low_squared) - nonlinear

    # Solve a dimensionless fraction of the certified upper squared energy.
    root = brentq(lambda fraction: inverse(fraction * low_squared), 0., 1.,
                  xtol=1e-15, rtol=np.float64(1e-13), full_output=True)
    pole_squared = float(root[0] * low_squared)
    residue_inverse = epsilon + 2 * sum(
        float(np.sum(weights * gaps / (gaps**2 - pole_squared)**2))
        for gaps, weights in channels) / norm
    lower_squared = spatial / (epsilon
                                + electric / (1 - low_squared / pair_gap_squared))
    expansion = []
    for fraction in (.1, .25, .5):
        squared = fraction**2 * pair_gap_squared
        exact = 2 * sum(float(np.sum(weights / (gaps * (gaps**2 - squared))))
                        for gaps, weights in channels) / norm
        upper = electric / (1 - fraction**2)
        expansion.append(dict(omega_over_pair_gap=fraction, exact_electric=exact,
                              lower=electric, upper=upper,
                              bounded=electric - 1e-12 <= exact <= upper + 1e-12))
    # The inverse kernel has energy units; compare with its restoring scale.
    # Contact/pair effect sizes may legitimately vanish at weak coupling.
    relative_pole_residual = abs(inverse(pole_squared)) / spatial
    checks = dict(unique_positive_residue=residue_inverse >= epsilon > 0,
                  pole_equation=relative_pole_residual < 1e-10,
                  below_pair_gap=pole_squared < pair_gap_squared,
                  rigorously_bounded=lower_squared - 1e-12 <= pole_squared <= low_squared + 1e-12,
                  spectral_expansion_bounds=all(row['bounded'] for row in expansion),
                  root_converged=root[1].converged)
    return dict(L=length, band_parameters=dict(a=band.a, m=band.m, r=band.r,
                v=band.v, e=band.e, improved=band.improved),
                mode=mode, polarization_axis=polarization_axis,
                momentum=spectrum['qwave'], link_momentum_angle=spectrum['qangle'],
                lattice_momentum=float(khat), physical_lattice_momentum=float(khat / band.a),
                hamiltonian_UE1=electric_coefficient, hamiltonian_K1=magnetic_coefficient,
                lagrangian_epsilon=epsilon,
                bare_photon_speed=band.a * float(np.sqrt(electric_coefficient * magnetic_coefficient)),
                matter_linear_speed=band.v,
                DeltaChi_magnetic=curvature / khat**2,
                DeltaChi_electric_transverse=electric,
                static_contact_per_profile_norm=contact, static_pair_per_profile_norm=pair_zero,
                static_restoring_force=spatial, profile_norm=norm,
                relative_pole_residual=relative_pole_residual,
                pair_threshold=float(np.sqrt(pair_gap_squared)),
                photon_energy=float(np.sqrt(pole_squared)),
                phase_velocity=float(np.sqrt(pole_squared) / spectrum['qwave']),
                lattice_phase_velocity=float(band.a * np.sqrt(pole_squared) / khat),
                link_residue_inverse=residue_inverse, field_residue_inverse=residue_inverse / band.a,
                canonical_link_leg_factor=1 / np.sqrt(residue_inverse),
                canonical_field_leg_factor=float(np.sqrt(band.a / residue_inverse)),
                canonical_cell_field_leg_factor=1 / (band.a * np.sqrt(residue_inverse)),
                canonical_current_coupling=float(band.charge * np.sqrt(band.a / residue_inverse)),
                normalization='theta=a A; field canonical uses continuum volume a^3 per cell',
                low_frequency_pole_upper=float(np.sqrt(low_squared)),
                rigorous_pole_lower=float(np.sqrt(lower_squared)),
                frequency_bounds=expansion, checks=checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', type=int, nargs='+', default=[16, 32, 48])
    parser.add_argument('--ma', type=float, default=.2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    band = RotationBand(a=1., m=args.ma, e=1., r=1., v=1., improved=True)
    rows = [match_photon(length, band, electric_coefficient=.09,
                         magnetic_coefficient=1 / .09) for length in args.lengths]
    # An allowed microscopic electric coefficient tunes the low-frequency
    # photon speed to the matter velocity, without deleting the sea response.
    reference = rows[-1]
    tuned_chi_e = (reference['hamiltonian_K1'] + reference['DeltaChi_magnetic']
                   - reference['DeltaChi_electric_transverse'])
    tuned = match_photon(args.lengths[-1], band, electric_coefficient=1 / tuned_chi_e,
                         magnetic_coefficient=1 / .09)
    result = dict(status='constructed_positive_photon_pole_with_spectral_error_bounds',
                  approximation='Gaussian microscopic rotor plus exact quadratic free-sea response',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=rows, low_frequency_speed_tuning=dict(
                      lagrangian_epsilon=tuned_chi_e, hamiltonian_UE1=1 / tuned_chi_e,
                      result=tuned),
                  all_checks_pass=all(all(row['checks'].values()) for row in rows + [tuned]))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

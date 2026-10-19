#!/usr/bin/env python3
"""Nonforward primitive Pauli scattering with continuous sea-dressed photons."""
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

from angular_photon_spectrum import ContinuousSea
from pauli_compton import FilledSeaConsumer
from rotation_band import RotationBand


class AxisPhotons:
    """Symmetry-protected degenerate cubic-axis branches; no bare dispersion."""
    def __init__(self, sea, a):
        self.sea, self.a = sea, a

    def photon_energy(self, k):
        k = np.asarray(k)
        if np.count_nonzero(k) != 1:
            raise ValueError('This scattering consumer selects nonzero cubic-axis modes.')
        axis = int(np.flatnonzero(k)[0])
        return self.sea.axis_energy(self.a * abs(k[axis]), axis) / self.a


def independent_kernel(band, kinematics, photons, eps, eps_out):
    p, k, pf, kf = kinematics
    eo = eps_out.conjugate()
    direct = np.linalg.inv((band.energy(p) + photons.photon_energy(k)) * np.eye(4) - band.h(p + k))
    crossed = np.linalg.inv((band.energy(p) - photons.photon_energy(kf)) * np.eye(4) - band.h(p - kf))
    return (band.vertex(pf, p + k, eo) @ direct @ band.vertex(p + k, p, eps)
            + band.vertex(pf, p - kf, eps) @ crossed @ band.vertex(p - kf, p, eo)
            + band.contact(pf, p, eps, eo, k, kf))


def axis_scattering(sea: ContinuousSea, q: float, a: float = .4):
    """Right-angle x->z scattering, averaging the exactly degenerate input pair.

    Physical electron spinors have norm one. Squared responses below average
    initial spin and polarization and sum both outgoing spin/polarizations.
    """
    band = RotationBand(a=a, m=sea.band.m / a, e=1., r=sea.band.r,
                        v=sea.band.v, improved=sea.band.improved)
    photons = AxisPhotons(sea, a)
    p = np.zeros(3)
    k = np.array([q / a, 0., 0.])
    omega = photons.photon_energy(k)
    nout = np.array([0., 0., 1.])
    def balance(radius):
        return band.energy(k - radius * nout) + sea.axis_energy(a * radius, 2) / a - band.m - omega
    # The physical branch is explicitly the low-BZ recoil root. At zero the
    # gapless outgoing limit is analytic; no finite-grid photon mass is inferred.
    lower, upper = q / (4 * a), q / a
    if balance(lower) >= 0 or balance(upper) <= 0:
        raise ValueError('Selected low-momentum recoil bracket failed.')
    solved = brentq(balance, lower, upper, xtol=2e-12,
                    rtol=np.float64(1e-12), full_output=True)
    radius = float(solved[0])
    kf = radius * nout
    pf = k - kf
    kinematics = (p, k, pf, kf)
    incoming = sea.modes(a * k)
    outgoing = sea.modes(a * kf)
    consumer = FilledSeaConsumer(band, photons, kinematics)
    squared = reference_squared = no_pairs_squared = bare_squared = 0.
    max_error = ward = no_pairs_ward = no_contact_ward = 0.
    bare_source = 1 / np.sqrt(sea.chi_e)
    polarization_rows = []
    for i, leg in enumerate(incoming['branches']):
        eps = leg['source'] * np.r_[0., leg['polarization']]
        for j, leg_out in enumerate(outgoing['branches']):
            eps_out = leg_out['source'] * np.r_[0., leg_out['polarization']]
            kernel = independent_kernel(band, kinematics, photons, eps, eps_out)
            branch_squared = 0.
            for spin in range(2):
                for spin_out in range(2):
                    amp = consumer.amplitude(eps, eps_out, spin, spin_out)['connected']
                    reference = np.vdot(band.spinor(pf, spin_out), kernel @ band.spinor(p, spin))
                    omitted = consumer.amplitude(eps, eps_out, spin, spin_out, keep_pairs=False)['connected']
                    bare = consumer.amplitude(eps * (bare_source / leg['source']),
                                              eps_out * (bare_source / leg_out['source']), spin, spin_out)['connected']
                    branch_squared += abs(amp)**2 / 2
                    squared += abs(amp)**2 / 4
                    reference_squared += abs(reference)**2 / 4
                    no_pairs_squared += abs(omitted)**2 / 4
                    bare_squared += abs(bare)**2 / 4
                    max_error = max(max_error, abs(amp - reference))
                    # Gauge vectors use the elementary midpoint lattice khat,
                    # not the continuum k or a hopping-displacement sine.
                    gin = leg['source'] * np.r_[omega, 2 * np.sin(a * k / 2) / a]
                    gout = leg_out['source'] * np.r_[consumer.omegaf, 2 * np.sin(a * kf / 2) / a]
                    for ein, eout in ((gin, eps_out), (eps, gout)):
                        ward = max(ward, abs(consumer.amplitude(ein, eout, spin, spin_out)['connected']))
                        no_pairs_ward = max(no_pairs_ward, abs(consumer.amplitude(
                            ein, eout, spin, spin_out, keep_pairs=False)['connected']))
                        no_contact_ward = max(no_contact_ward, abs(consumer.amplitude(
                            ein, eout, spin, spin_out, keep_contact=False)['connected']))
            polarization_rows.append(dict(incoming_branch=i, outgoing_branch=j,
                                          spin_averaged_squared=branch_squared))
    # Axis C4 rotations/reflections forbid transverse velocity components.
    # d(omega_phys)/d(k_phys) = d(omega_lattice)/dQ; no factor of a.
    velocity_in = sea.axis_velocity(q, 0)
    velocity_out = sea.axis_velocity(a * radius, 2)
    velocity_in_fine = sea.axis_velocity(q, 0, 1e-4)
    velocity_out_fine = sea.axis_velocity(a * radius, 2, 1e-4)
    x = a * pf
    mass_gradient = band.r * np.sin(x)
    if band.improved:
        mass_gradient *= 4 * np.sin(x / 2)**2
    electron_velocity = (band.v**2 * np.sin(x) * np.cos(x) / a
                         + band.mass(pf) * mass_gradient) / band.energy(pf)
    flux = abs(velocity_in_fine)
    jacobian = abs(velocity_out_fine - np.dot(electron_velocity, nout))
    phase_space = radius**2 / (16 * np.pi**2 * omega * consumer.omegaf * flux * jacobian)
    rate = float(squared * phase_space)
    # A bare-speed denominator is an independently adverse physical rate,
    # distinct from the omitted external residue control.
    bare_speed = float(np.sqrt(sea.chi_b / sea.chi_e))
    bare_flux = bare_speed * float(np.cos(q / 2))
    bare_jacobian = abs(bare_speed * float(np.cos(a * radius / 2)) - np.dot(electron_velocity, nout))
    bare_velocity_rate = squared * radius**2 / (
        16 * np.pi**2 * omega * consumer.omegaf * bare_flux * bare_jacobian)
    for row in polarization_rows:
        row['differential_rate'] = row['spin_averaged_squared'] * phase_space
    checks = dict(actual_pauli_matches_band_resolvent=max_error < 1e-10,
                  midpoint_ward=ward < 1e-10,
                  omitted_pairs_fail_ward=no_pairs_ward > 1e-5,
                  omitted_contact_fails_ward=no_contact_ward > 1e-8,
                  omitted_pairs_change_rate=abs(no_pairs_squared / squared - 1) > .01,
                  velocity_step_converges=max(abs(velocity_in - velocity_in_fine),
                                              abs(velocity_out - velocity_out_fine)) < 2e-6,
                  exact_energy_recoil=abs(balance(radius)) < 1e-10,
                  symmetry_protected_degeneracy=max(incoming['branch_splitting'],
                                                    outgoing['branch_splitting']) < 1e-10,
                  real_positive_subthreshold_poles=all(
                      0 < branch['energy'] < mode['pair_threshold'] and branch['residue'] > 0
                      and branch['pole_residual'] < 1e-10
                      for mode in (incoming, outgoing) for branch in mode['branches']),
                  positive_flux_and_jacobian=flux > 0 and jacobian > 0)
    return dict(order=sea.order, ma=sea.ma, a=a, q=q, mass=band.m,
                incoming_momentum=k.tolist(), outgoing_momentum=kf.tolist(),
                electron_recoil=pf.tolist(), incoming_photon=incoming, outgoing_photon=outgoing,
                energy_residual=balance(radius), scattering_angle=np.pi / 2,
                pauli_squared=squared, independent_squared=reference_squared,
                omitted_pairs_squared=no_pairs_squared, bare_external_legs_squared=bare_squared,
                bare_source_relative_rate_error=float(bare_squared / squared - 1),
                bare_group_velocity_relative_rate_error=float(bare_velocity_rate / rate - 1),
                amplitude_residual=max_error, ward_residual=ward,
                omitted_pairs_ward=no_pairs_ward, omitted_contact_ward=no_contact_ward,
                incoming_group_flux=flux, outgoing_radial_group_velocity=velocity_out_fine,
                radial_energy_delta_jacobian=jacobian, electron_velocity=electron_velocity.tolist(),
                velocity_stencil_error=max(abs(velocity_in - velocity_in_fine),
                                           abs(velocity_out - velocity_out_fine)),
                phase_space=phase_space, differential_rate=rate,
                omitted_pairs_rate=no_pairs_squared * phase_space,
                bare_external_legs_rate=bare_squared * phase_space,
                bare_velocity_rate=bare_velocity_rate, polarization_rates=polarization_rows,
                checks={key: bool(value) for key, value in checks.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', nargs='+', type=int, default=[24, 32, 48])
    parser.add_argument('--momenta', nargs='+', type=float, default=[.12, .2])
    parser.add_argument('--ma', type=float, default=.4)
    parser.add_argument('--a', type=float, default=.4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if len(args.orders) < 3 or args.orders != sorted(set(args.orders)):
        parser.error('Require at least three increasing quadrature orders for refinement evidence.')
    rows, tensors = [], []
    for order in args.orders:
        sea = ContinuousSea(order, args.ma)
        rows.extend(axis_scattering(sea, q, args.a) for q in args.momenta)
        # An off-axis positive control explicitly retains the tensor; no scalar
        # isotropic replacement is used in general directions.
        q_oblique = np.array([.16, .09, .04])
        tensors.append(dict(order=order, result=sea.modes(q_oblique)))
    refinements = []
    for q in args.momenta:
        sequence = [row for row in rows if row['q'] == q]
        rates = [row['differential_rate'] for row in sequence]
        ward = [row['incoming_photon']['gradient_relative_residual'] for row in sequence]
        coarse_change = abs(rates[-2] / rates[-3] - 1)
        fine_change = abs(rates[-1] / rates[-2] - 1)
        refinements.append(dict(q=q, rates=rates, gradient_residuals=ward,
                                coarse_relative_rate_change=coarse_change,
                                fine_relative_rate_change=fine_change,
                                checks=dict(rate_refines=fine_change < coarse_change,
                                            rate_stable=fine_change < .002,
                                            gradient_ward_refines=ward[-1] < ward[-2] < ward[0],
                                            gradient_ward_small=ward[-1] < 2e-5)))
    tensor_checks = dict(off_axis_tensor_has_positive_modes=all(
        0 < branch['energy'] < row['result']['pair_threshold'] and branch['pole_residual'] < 1e-10
        for row in tensors for branch in row['result']['branches']),
        off_axis_anisotropy_resolved=tensors[-1]['result']['branch_splitting'] > 1e-8)
    passed = (all(all(row['checks'].values()) for row in rows)
              and all(all(row['checks'].values()) for row in refinements)
              and all(tensor_checks.values()))
    result = dict(status='continuous_transfer_nonforward_dressed_primitive_pauli_rate',
                  scope='Gaussian rotor plus quadratic sea; connected tree electron response with dressed external legs',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  spectrum_source_sha256=hashlib.sha256(Path(__file__).with_name('angular_photon_spectrum.py').read_bytes()).hexdigest(),
                  bare_U_E=.09, bare_K=1 / .09, rows=rows, quadrature_refinement=refinements,
                  oblique_tensor_modes=tensors, tensor_checks=tensor_checks, all_checks_pass=passed)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())

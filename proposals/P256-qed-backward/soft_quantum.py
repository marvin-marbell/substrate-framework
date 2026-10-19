#!/usr/bin/env python3
"""Conditional leading-soft quantum dressing of the actual Pauli Compton consumer.

No hard radiative correction, compact Coulomb minimizer, or infraparticle theorem
is claimed. Run directly for the finite quantum-mode and IR-resolution witnesses.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammaln

from lattice_compton import WilsonLink
from pauli_compton import FilledSeaConsumer
from rotation_band import RotationBand


def current_velocity(band, momentum):
    """Charge-divided current expectation; no relativistic velocity is inserted."""
    currents = band.current(momentum, momentum)
    velocities = []
    eigenstate_residual = 0.
    for spin in range(2):
        state = band.spinor(momentum, spin)
        eigenstate_residual = max(eigenstate_residual, float(np.linalg.norm(
            band.h(momentum) @ state - band.energy(momentum) * state)))
        velocities.append(np.array([np.vdot(state, op @ state).real for op in currents]))
    h = 2e-6
    gradient = np.array([(band.energy(momentum + h * axis)
                          - band.energy(momentum - h * axis)) / (2 * h)
                         for axis in np.eye(3)])
    hf_error = float(np.max(np.abs(velocities[0] - gradient)))
    spin_error = float(np.max(np.abs(velocities[0] - velocities[1])))
    assert hf_error < 2e-9 and spin_error < 2e-12 and eigenstate_residual < 2e-12
    return velocities[0], dict(momentum=momentum.tolist(), velocity=velocities[0].tolist(),
                              energy_gradient=gradient.tolist(), hellmann_feynman_error=hf_error,
                              spin_independence_error=spin_error,
                              band_eigenstate_residual=eigenstate_residual)


def transverse_basis(direction):
    n = np.asarray(direction, dtype=float)
    n = n / np.linalg.norm(n)
    seed = np.eye(3)[np.argmin(np.abs(n))]
    first = np.cross(n, seed)
    first /= np.linalg.norm(first)
    return first, np.cross(n, first)


def oscillator_source(charge, velocity, direction, eps, omega, volume, speed):
    """H_v=(omega-k.v) a†a - g(a+a†), canonical A=(a+a†)/sqrt(2omega V)."""
    if omega <= 0 or volume <= 0:
        raise ValueError('Positive mode frequency and normalization volume required')
    doppler = omega * (1 - np.dot(direction, velocity) / speed)
    if doppler <= 0:
        raise ValueError('Stable subluminal asymptotic oscillator required')
    coupling = charge * np.dot(velocity, eps) / np.sqrt(2 * omega * volume)
    return float(doppler), float(coupling), float(coupling / doppler)


def quantum_mode(charge, vi, vf, speed):
    """Diagonalize actual truncated Hamiltonians, not an analytic coherent vector."""
    vhat = vf / np.linalg.norm(vf)
    perpendicular = transverse_basis(vhat)[0]
    n = (vhat + perpendicular) / np.sqrt(2)
    eps = (vhat - perpendicular) / np.sqrt(2)
    omega = 1e-3
    # A rectangular periodic box can have length 2pi*c/omega along n and this
    # transverse area. This single-mode stress box is NOT the bulk IR integral.
    _, _, eta_unit = oscillator_source(charge, vf, n, eps, omega, 1., speed)
    volume = (eta_unit / 1.2)**2
    oi, gi, ai = oscillator_source(charge, vi, n, eps, omega, volume, speed)
    of, gf, af = oscillator_source(charge, vf, n, eps, omega, volume, speed)
    mean = (af - ai)**2
    rows = []
    for cutoff in (4, 12, 32):
        annihilation = np.diag(np.sqrt(np.arange(1, cutoff, dtype=float)), 1)
        number = np.diag(np.arange(cutoff, dtype=float))
        hi = oi * number - gi * (annihilation + annihilation.T)
        hf = of * number - gf * (annihilation + annihilation.T)
        ei, ui = eigh(hi)
        ef, uf = eigh(hf)
        ground_i, ground_f = ui[:, 0], uf[:, 0]
        probabilities = np.abs(uf.T @ ground_i)**2
        levels = np.arange(cutoff)
        poisson = np.exp(-mean + levels * np.log(mean) - gammaln(levels + 1))
        qop = (annihilation + annihilation.T) / np.sqrt(2 * omega)
        measured_q = float(ground_f @ qop @ ground_f)
        variance_q = float(ground_f @ qop @ qop @ ground_f - measured_q**2)
        rows.append(dict(cutoff=cutoff,
                         ground_energy_error=abs(float(ef[0]) + gf**2 / of) / of,
                         displaced_ground_overlap_squared=float(probabilities[0]),
                         exact_overlap_squared=float(np.exp(-mean)),
                         max_first_six_probability_error=float(np.max(
                             np.abs(probabilities[:min(6, cutoff)] - poisson[:min(6, cutoff)]))),
                         probabilities=probabilities[:8].tolist(),
                         poisson_probabilities=poisson[:8].tolist(),
                         initial_ground_energy=float(ei[0]), final_ground_energy=float(ef[0]),
                         coordinate_mean=measured_q,
                         exact_coordinate_mean=float(np.sqrt(2 / omega) * af),
                         coordinate_variance=variance_q,
                         exact_coordinate_variance=1 / (2 * omega),
                         bare_fock_boundary_probability=float(abs(ground_f[-1])**2)))
    assert rows[0]['ground_energy_error'] > 1e-3
    assert rows[-1]['ground_energy_error'] < 2e-12
    assert rows[-1]['max_first_six_probability_error'] < 2e-10
    assert abs(rows[-1]['displaced_ground_overlap_squared'] - np.exp(-mean)) < 2e-12
    assert abs(rows[-1]['coordinate_variance'] * 2 * omega - 1) < 2e-11
    assert abs(rows[-1]['coordinate_mean'] / rows[-1]['exact_coordinate_mean'] - 1) < 2e-12
    assert rows[-1]['bare_fock_boundary_probability'] < 1e-20
    wrong_no_doppler = (gf / omega - gi / omega)**2
    wrong_no_canonical_two = 2 * mean
    assert abs(wrong_no_doppler / mean - 1) > .05
    # Test a truly arbitrarily soft mode with no absolute eigenvalue/frequency floor.
    scaled = []
    for frequency in (1e-3, 1e-9, 1e-18, 1e-30):
        _, _, eta = oscillator_source(charge, vf, n, eps, frequency, volume, speed)
        scaled.append(float(eta * frequency**1.5))
    assert np.max(np.abs(np.array(scaled) / scaled[0] - 1)) < 2e-14
    cubic_volume = (2 * np.pi * speed / omega)**3
    _, _, cubic_eta = oscillator_source(charge, vf, n, eps, omega, cubic_volume, speed)
    return dict(charge=charge, omega=omega, direction=n.tolist(), polarization=eps.tolist(),
                normalization_volume=volume, box_length_along_mode=2 * np.pi * speed / omega,
                box_transverse_area=volume * omega / (2 * np.pi * speed),
                bulk_cubic_box_displacement=cubic_eta, mean_relative_photon_number=mean,
                moving_generator_frequencies=[oi, of], current_sources=[gi, gf],
                cloud_displacements=[ai, af], cutoff_convergence=rows,
                wrong_no_doppler_mean=wrong_no_doppler,
                wrong_missing_canonical_two_mean=wrong_no_canonical_two,
                soft_frequency_scaled_displacements=scaled,
                interpretation='Quantum probabilities and nonzero vacuum variance, not merely a classical transverse wave.')


def angular_coefficient(charge, vi, vf, speed=1., order=64):
    if speed <= 0 or max(np.linalg.norm(vi), np.linalg.norm(vf)) >= speed:
        raise ValueError('Strictly subluminal velocities and positive photon speed required')
    mu, weights = np.polynomial.legendre.leggauss(order)
    phi = np.arange(2 * order) * np.pi / order
    sintheta = np.sqrt(1 - mu**2)[:, None]
    normals = np.stack(np.broadcast_arrays(sintheta * np.cos(phi),
                                           sintheta * np.sin(phi), mu[:, None]), axis=-1)
    difference = (vf / (speed - normals @ vf)[..., None]
                  - vi / (speed - normals @ vi)[..., None])
    transverse = np.sum(difference**2, axis=-1) - np.sum(difference * normals, axis=-1)**2
    integral = float(np.sum(weights[:, None] * transverse) * np.pi / order)
    return charge**2 * integral / (2 * speed * (2 * np.pi)**3)


def rest_coefficient(charge, beta):
    """Independent integrated rest->v comparator at c=1, stable as beta->0."""
    if not 0 <= beta < 1:
        raise ValueError('Rest comparison requires 0 <= beta < 1')
    if beta < 1e-3:
        bracket = sum(beta**(2 * j) / (2 * j + 1) for j in range(1, 7))
    else:
        bracket = np.arctanh(beta) / beta - 1
    return float(charge**2 * bracket / (2 * np.pi**2))


def gauge_probe(charge, vi, vf, speed):
    n = np.array([.3, -.4, np.sqrt(.75)])
    eps = np.r_[0., transverse_basis(n)[0]]
    z = .73
    shifted = eps + z * np.r_[speed, n]
    def soft(epsilon, incoming_charge=charge):
        out = charge * (vf @ epsilon[1:] - epsilon[0]) / (speed - n @ vf)
        inc = incoming_charge * (vi @ epsilon[1:] - epsilon[0]) / (speed - n @ vi)
        return float(out - inc)
    residual = abs(soft(shifted) - soft(eps))
    charge_mismatch_residual = abs(soft(shifted, charge * 1.1) - soft(eps, charge * 1.1))
    spatial_only = lambda epsilon: charge * (vf @ epsilon[1:] / (speed - n @ vf)
                                            - vi @ epsilon[1:] / (speed - n @ vi))
    spatial_only_error = abs(spatial_only(shifted) - spatial_only(eps))
    assert residual < 2e-15 and charge_mismatch_residual > abs(charge) * .01
    assert spatial_only_error > abs(charge) * .001
    return dict(conserved_charge_gauge_residual=residual,
                unequal_in_out_charge_gauge_residual=charge_mismatch_residual,
                missing_density_component_gauge_residual=spatial_only_error,
                substitution='eps -> eps + z*(c,n); scalar density term retained')


def ir_resolution(coefficient, upper, resolution) -> dict:
    """Each unresolved photon <Delta, no photon Delta..Lambda; no total-energy cut."""
    if not 0 < resolution < upper or coefficient <= 0:
        raise ValueError('Require positive C and 0 < Delta < Lambda')
    resolved_number = coefficient * np.log(upper / resolution)
    rows = []
    for log_ratio in (10., 100., 1000., 10000., 100000.):
        total_number = coefficient * log_ratio
        unresolved_number = total_number - resolved_number
        assert unresolved_number > 0
        cutoff = math.ceil(unresolved_number + 20 * math.sqrt(unresolved_number + 1) + 50)
        first_omitted = cutoff + 1
        log_tail_bound = (-unresolved_number + first_omitted
                          * (1 + np.log(unresolved_number / first_omitted)))
        assert log_tail_bound < np.log(1e-25)
        occupations = np.arange(cutoff + 1)
        # The sum is over aggregate occupation of all independent unresolved
        # modes. Multinomial expansion gives N_unresolved^n/n!, not one mode.
        log_real_sum = float(np.logaddexp.reduce(occupations * np.log(unresolved_number)
                                               - gammaln(occupations + 1)))
        log_veto = -total_number + log_real_sum
        assert abs(log_veto + resolved_number) < 2e-11 * max(1., total_number)
        log_lambda = np.log(upper) - log_ratio
        lower = math.exp(log_lambda) if log_lambda > -745 else 0.
        rows.append(dict(log_lambda=log_lambda, log_upper_over_lambda=log_ratio,
                         cloud_number=total_number, unresolved_mean=unresolved_number,
                         resolved_mean=resolved_number, unresolved_sum_cutoff=cutoff,
                         unresolved_poisson_log_tail_bound=float(log_tail_bound),
                         log_cloud_overlap_squared=-total_number,
                         cloud_overlap_squared=float(np.exp(-total_number)),
                         mean_soft_energy=coefficient * (upper - lower),
                         log_veto_probability=log_veto,
                         veto_probability=float(np.exp(log_veto)),
                         incorrect_exclusive_vacuum_probability=float(np.exp(-total_number)),
                         incorrect_omit_virtual_log_probability=log_real_sum))
    expected = float((resolution / upper)**coefficient)
    assert all(abs(row['veto_probability'] / expected - 1) < 2e-10 for row in rows)
    assert rows[-1]['cloud_number'] > 100 * rows[0]['cloud_number']
    assert rows[-1]['log_cloud_overlap_squared'] < -resolved_number - 100 * coefficient
    return dict(coefficient=coefficient, soft_upper_energy=upper,
                detector_resolution_energy=resolution, leading_soft_veto_factor=expected,
                finite_soft_energy_limit=coefficient * upper, regulator_sequence=rows,
                limits='N -> infinity, overlap_squared -> 0 for C>0; E -> C*Lambda. Log-lambda retained even beyond floating-point exp range; zero lower in E is only its controlled limit, not a spectral floor.')


def hard_consumer(band, photons, kinematics) -> dict:
    """Primitive connected Pauli tree amplitude and canonical finite-a phase space."""
    consumer = FilledSeaConsumer(band, photons, kinematics)
    reduced_squared = 0.
    for eps in photons.polarizations(consumer.k):
        for eps_out in photons.polarizations(consumer.kf):
            for spin in range(2):
                for spin_out in range(2):
                    connected = consumer.amplitude(eps, eps_out, spin, spin_out)['connected']
                    reduced_squared += abs(connected)**2 / 4
    incoming_flux = np.linalg.norm(photons.photon_velocity(consumer.k))
    nout = consumer.kf / np.linalg.norm(consumer.kf)
    final_velocity, _ = current_velocity(band, consumer.pf)
    jacobian = abs(np.dot(photons.photon_velocity(consumer.kf) - final_velocity, nout))
    rate = float(reduced_squared * np.dot(consumer.kf, consumer.kf) / (
        16 * np.pi**2 * consumer.omega * consumer.omegaf * incoming_flux * jacobian))
    assert rate > 0
    return dict(pauli_spin_polarization_averaged_reduced_squared=float(reduced_squared),
                tree_dsigma_dwavevector_solid_angle=rate,
                canonical_photon_energies=[consumer.omega, consumer.omegaf],
                incident_flux=float(incoming_flux), radial_phase_space_jacobian=float(jacobian),
                hard_input='FilledSeaConsumer primitive connected vacuum-subtracted tree amplitudes; no gamma-matrix comparator evaluated')


def run(charge=.30350041):
    if not np.isfinite(charge) or charge <= 0:
        raise ValueError('Charge magnitude must be finite and positive')
    band = RotationBand(.025, e=charge, improved=True)
    photons = WilsonLink(band.a, m=band.m, e=band.e, r=band.r, v=band.v,
                         photon_speed=1., improved=band.improved)
    kinematics = photons.kinematics(.2, .7)
    pi, ki, pf, kf = kinematics
    vi, initial = current_velocity(band, pi)
    vf, final = current_velocity(band, pf)
    # Additional actual-band off-axis and non-rest probes reject p/E substitution.
    hf_probes = [current_velocity(model, np.array([.23, -.17, .31]))[1]
                 for model in (band, RotationBand(.2, e=charge, v=.83, improved=False))]
    energy_balance = abs(band.energy(pi) + photons.photon_energy(ki)
                         - band.energy(pf) - photons.photon_energy(kf))
    assert energy_balance < 2e-12
    c = photons.photon_speed
    low = angular_coefficient(band.charge, vi, vf, c, 32)
    coefficient = angular_coefficient(band.charge, vi, vf, c, 64)
    independent = rest_coefficient(band.charge, float(np.linalg.norm(vf)))
    assert abs(coefficient / low - 1) < 2e-12
    assert abs(coefficient / independent - 1) < 2e-12
    angular_controls = []
    for beta in (.0001, .4, .85):
        numerical = angular_coefficient(band.charge, np.zeros(3), np.array([0., 0., beta]), 1., 96)
        analytic = rest_coefficient(band.charge, beta)
        assert abs(numerical / analytic - 1) < 2e-11
        angular_controls.append(dict(beta=beta, numerical=numerical, independent=analytic))
    nonrest_i, nonrest_f = np.array([.13, -.08, .02]), np.array([-.06, .19, .07])
    nonrest = angular_coefficient(band.charge, nonrest_i, nonrest_f)
    swapped = angular_coefficient(band.charge, nonrest_f, nonrest_i)
    unchanged = angular_coefficient(band.charge, nonrest_i, nonrest_i)
    assert nonrest > 0 and abs(nonrest - swapped) < 2e-15 and unchanged == 0
    upper, delta = .001, .0001
    hard = hard_consumer(band, photons, kinematics)
    soft = ir_resolution(coefficient, upper, delta)
    hard_scale = min(photons.photon_energy(ki), photons.photon_energy(kf), band.m)
    assert upper / hard_scale < .01
    return dict(scope='Conditional leading-soft quantum dressing and individually resolved-photon veto, not full radiative Compton or a general infraparticle theorem',
                model=dict(a=band.a, m=band.m, charge=band.charge, charge_magnitude=band.e,
                           matter_speed=band.v, photon_speed=c, improved=band.improved,
                           normalization='e=.30350041 is an input model normalization, not independently derived dressed fine structure'),
                shared_compton_momenta=[p.tolist() for p in kinematics],
                hard_energy_balance_residual=energy_balance,
                actual_band_current=dict(initial=initial, final=final, off_axis_probes=hf_probes),
                angular_integral=dict(coefficient=coefficient, order32=low,
                                      independent_rest_formula=independent,
                                      rest_formula_controls=angular_controls,
                                      distinct_nonrest_coefficient=nonrest,
                                      swapped_velocity_coefficient=swapped,
                                      same_velocity_coefficient=unchanged),
                quantum_mode=quantum_mode(band.charge, vi, vf, c),
                gauge_controls=gauge_probe(band.charge, vi, vf, c),
                infrared_detector=soft, hard_tree=hard,
                joined_leading_soft_dsigma=hard['tree_dsigma_dwavevector_solid_angle'] * soft['leading_soft_veto_factor'],
                soft_upper_over_hard_scale=upper / hard_scale,
                approximation='Continuum soft omega=c|k|; sudden hard event, adiabatic straight asymptotic currents, negligible soft recoil; tree rate multiplied only by leading-soft factor.',
                unresolved=['Coulomb-minimized compact charged state', 'thermodynamic compact photon transfer',
                            'unquenched charge/current normalization and speed equality',
                            'hard loops and finite soft terms', 'origin of quantum postulates and selected microscopic law'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--charge', type=float, default=.30350041,
                        help='Input positive model charge magnitude, not a measured alpha prediction')
    args = parser.parse_args()
    print(json.dumps(run(args.charge), indent=2, allow_nan=False))

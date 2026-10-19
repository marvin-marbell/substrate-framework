#!/usr/bin/env python3
"""Nonrest Pauli filled-sea scattering; boosts select targets, not lattice laws.

Run with the project Python. Default output is a self-checking JSON report.
The invariant normalization is not a finite-a Lorentz-invariant cross section.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from lattice_compton import WilsonLink
from pauli_compton import FilledSeaConsumer, complex_fields
from rotation_band import RotationBand


LOW_ZONE = 0.6  # Bound every external and intermediate component |a p_i|.


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def boost(fourvector, velocity):
    """Active continuum boost, exclusively for selecting target input data."""
    velocity = np.asarray(velocity, dtype=float)
    speed2 = float(velocity @ velocity)
    if speed2 == 0:
        return np.array(fourvector, dtype=float)
    if speed2 >= 1:
        raise ValueError('Target boost must be subluminal')
    energy, momentum = fourvector[0], np.asarray(fourvector[1:])
    gamma = 1 / np.sqrt(1 - speed2)
    dot = float(velocity @ momentum)
    return np.r_[gamma * (energy + dot),
                 momentum + ((gamma - 1) * dot / speed2 + gamma * energy) * velocity]


def minkowski(first, second):
    return float(first[0] * second[0] - np.dot(first[1:], second[1:]))


def target_event(mass, omega_over_m, theta, incoming, velocity):
    """Own exact continuum recoil; never reused as the finite-a candidate recoil."""
    direction = unit(incoming)
    seed = np.eye(3)[np.argmin(abs(direction))]
    tangent = unit(seed - np.dot(seed, direction) * direction)
    outgoing = np.cos(theta) * direction + np.sin(theta) * tangent
    omega = mass * omega_over_m
    recoil = omega / (1 + omega_over_m * (1 - np.cos(theta)))
    p = np.r_[mass, np.zeros(3)]
    k = np.r_[omega, omega * direction]
    kout = np.r_[recoil, recoil * outgoing]
    pf = p + k - kout
    return tuple(boost(vector, velocity) for vector in (p, k, pf, kout))


def invariant_kn(event, mass, charge):
    """Independent scalar continuum target: no kernel, spinor or gamma supplier."""
    p, k, _, kout = event
    x = minkowski(p, k) / mass**2
    y = minkowski(p, kout) / mass**2
    d = 1 / x - 1 / y
    return float(2 * charge**4 * (x / y + y / x + 2 * d + d**2))


def candidate_event(band, photons, target):
    """Solve the candidate law on the bounded low-Brillouin branch.

    Only target incoming three-momenta and outgoing direction are retained.
    Incoming energies, outgoing magnitude and final electron are all candidate.
    """
    p, k = target[0][1:].copy(), target[1][1:].copy()
    direction = unit(target[3][1:])
    total = p + k
    incoming_energy = band.energy(p) + photons.photon_energy(k)

    def balance(magnitude):
        kout = magnitude * direction
        return band.energy(total - kout) + photons.photon_energy(kout) - incoming_energy

    if balance(0.) >= 0:
        raise ValueError('No positive-recoil branch starting below incoming energy')
    # Photon energy is increasing within this bound. Scan all sign crossings,
    # rather than trusting the continuum recoil or a periodic high-zone root.
    cap = LOW_ZONE / (band.a * max(abs(direction)))
    grid = np.linspace(0., cap, 257)
    values = [balance(float(value)) for value in grid]
    brackets = [(left, right) for left, right, fl, fr in
                zip(grid[:-1], grid[1:], values[:-1], values[1:]) if fl < 0 <= fr]
    if len(brackets) != 1:
        raise ValueError(f'Expected one low-zone recoil branch, found {len(brackets)}')
    magnitude = brentq(balance, *brackets[0], xtol=1e-14, rtol=1e-14)
    kout = magnitude * direction
    pf = total - kout
    max_component = max(float(np.max(abs(band.a * vector)))
                        for vector in (p, k, pf, kout, p + k, p - kout))
    if max_component >= LOW_ZONE:
        raise ValueError('External/intermediate momentum leaves scoped low-zone domain')
    return (p, k, pf, kout), {
        'energy_residual': float(balance(magnitude)),
        'momentum_residual': float(np.linalg.norm(p + k - pf - kout)),
        'max_external_or_intermediate_abs_ap_component': max_component,
        'candidate_outgoing_magnitude': float(magnitude),
        'target_outgoing_magnitude': float(np.linalg.norm(target[3][1:])),
        'relative_outgoing_magnitude_correction': float(magnitude / np.linalg.norm(target[3][1:]) - 1),
        'initial_electron_energy': band.energy(p),
        'final_electron_energy': band.energy(pf),
        'incoming_photon_energy': photons.photon_energy(k),
        'outgoing_photon_energy': photons.photon_energy(kout),
        'candidate_momenta_p_k_pf_kout': [vector.tolist() for vector in (p, k, pf, kout)],
    }


def scattering_row(a, case, photon_speed=1.):
    mass, charge = case['mass'], case['charge']
    target = target_event(mass, case['omega_over_m'], case['theta'],
                          case['incoming_direction'], case['boost_velocity'])
    band = RotationBand(a, m=mass, e=charge, v=1., improved=True)
    photons = WilsonLink(a, m=mass, e=charge, v=1., photon_speed=photon_speed, improved=True)
    event, conservation = candidate_event(band, photons, target)
    consumer = FilledSeaConsumer(band, photons, event)
    normalization = 2 * np.sqrt(band.energy(event[0]) * band.energy(event[2]))
    squared = {'full': 0., 'no_pairs': 0., 'no_contact': 0.}
    controls = {'full': {}, 'no_pairs': {'keep_pairs': False},
                'no_contact': {'keep_contact': False}}
    ward = {name: {'incoming': 0., 'outgoing': 0.} for name in controls}
    deletion = {'no_pairs': 0., 'no_contact': 0.}
    witness = None
    pol_in, pol_out = photons.polarizations(event[1]), photons.polarizations(event[3])
    for eps in pol_in:
        for eps_out in pol_out:
            for spin in range(2):
                for spin_out in range(2):
                    channels = consumer.amplitude(eps, eps_out, spin, spin_out)
                    amplitudes = {'full': channels['connected']}
                    for name in deletion:
                        amplitudes[name] = consumer.amplitude(eps, eps_out, spin, spin_out,
                                                             **controls[name])['connected']
                        deletion[name] = max(deletion[name], abs(amplitudes[name] - amplitudes['full']))
                    for name, amplitude in amplitudes.items():
                        # Average two initial electron spins and two initial photon
                        # polarizations; sum all final spins/polarizations.
                        squared[name] += float(abs(normalization * amplitude)**2 / 4)
                    pair_size = (abs(channels['absorb_first_electron_pair'])
                                 + abs(channels['emit_first_electron_pair']))
                    if witness is None or pair_size > witness[0]:
                        witness = (pair_size, {'spin': spin, 'spin_out': spin_out,
                                              'eps_in': eps.tolist(), 'eps_out': eps_out.tolist(),
                                              'channels': complex_fields(channels)})
    # Actual midpoint-link gauge variations, not continuum k substituted into a
    # different dispersion. Exercise each leg and every external spin channel.
    win = np.r_[consumer.omega, photons.khat(event[1])]
    wout = np.r_[consumer.omegaf, photons.khat(event[3])]
    substitutions = [('incoming', win, eps_out) for eps_out in pol_out]
    substitutions += [('outgoing', eps, wout) for eps in pol_in]
    for leg, incoming, outgoing in substitutions:
        for spin in range(2):
            for spin_out in range(2):
                for name, options in controls.items():
                    amplitude = consumer.amplitude(incoming, outgoing, spin, spin_out,
                                                   **options)['connected']
                    ward[name][leg] = max(ward[name][leg], float(abs(normalization * amplitude)))
    reference = invariant_kn(target, mass, charge)
    # Independently reduce the scalar invariant expression in the original rest
    # event to the ordinary KN angular bracket; compare, do not construct one
    # target from the other or feed candidate recoil into either.
    ratio = 1 / (1 + case['omega_over_m'] * (1 - np.cos(case['theta'])))
    angular_kn = float(2 * charge**4 * (ratio + 1 / ratio - np.sin(case['theta'])**2))
    p, k, pf, kout = target
    target_checks = {
        'four_momentum_residual': float(np.linalg.norm(p + k - pf - kout)),
        'initial_mass_shell_residual': abs(minkowski(p, p) - mass**2),
        'final_mass_shell_residual': abs(minkowski(pf, pf) - mass**2),
        'incoming_null_residual': abs(minkowski(k, k)),
        'outgoing_null_residual': abs(minkowski(kout, kout)),
        'rest_angular_kn_relative_difference': abs(reference / angular_kn - 1),
    }
    assert max(target_checks.values()) < 2e-12, target_checks
    assert abs(conservation['energy_residual']) < 2e-11 * mass, conservation
    assert conservation['momentum_residual'] < 2e-13 * mass, conservation
    assert max(ward['full'].values()) < 2e-10 * charge**2 * mass, ward
    assert witness is not None
    return {'a': a, 'a_times_mass': a * mass, 'case': case,
            'matter_speed': band.v, 'photon_speed': photon_speed,
            'external_normalization': float(normalization),
            'conservation': conservation, 'target_event_p_k_pf_kout': [v.tolist() for v in target],
            'target_checks': target_checks, 'independent_continuum_invariant_kn_squared': reference,
            'independent_rest_angular_kn_squared': angular_kn,
            'pauli_spin_polarization_averaged_squared': squared,
            'relative_error': abs(squared['full'] / reference - 1),
            'signed_relative_correction': squared['full'] / reference - 1,
            'actual_ward_normalized_amplitude': ward,
            'deletion_max_normalized_amplitude_change': {key: float(normalization * value)
                                                       for key, value in deletion.items()},
            'actual_virtual_pair_witness': witness[1],
            'bosonic_edge_qubits': len(consumer.words.edges)}


def lorentz_algebra_probe(velocity):
    """Extract continuum gamma/boost algebra from selected hopping coefficients."""
    band = RotationBand(0.1, improved=True)
    onsite, couplings = band.couplings()
    c = np.array([1j * band.a / band.v * (matrix - matrix.conj().T)
                  for displacement, matrix in couplings if sum(abs(displacement)) == 1])
    b = onsite / (band.m + 4.5 * band.r / band.a)
    identity = np.eye(4, dtype=complex)
    gamma = np.array([b, *(b @ operator for operator in c)])
    metric = np.diag([1., -1., -1., -1.])
    clifford = max(float(np.linalg.norm(gamma[mu] @ gamma[nu] + gamma[nu] @ gamma[mu]
                                        - 2 * metric[mu, nu] * identity))
                   for mu in range(4) for nu in range(4))
    j = np.array([-0.5j * c[first] @ c[second] for first, second in ((1, 2), (2, 0), (0, 1))])
    hermitian_boost = 0.5j * c
    def comm(first, second):
        return first @ second - second @ first
    lorentz = max(float(np.linalg.norm(residual))
                  for first, second, third in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
                  for residual in (comm(j[first], j[second]) - 1j * j[third],
                                   comm(j[first], hermitian_boost[second]) - 1j * hermitian_boost[third],
                                   comm(hermitian_boost[first], hermitian_boost[second]) + 1j * j[third]))
    velocity = np.asarray(velocity, dtype=float)
    speed = np.linalg.norm(velocity)
    rapidity = float(np.arctanh(speed))
    s = expm(rapidity / 2 * np.einsum('i,ijk->jk', velocity / speed, c))
    def slash(vector):
        return vector[0] * gamma[0] - np.einsum('i,ijk->jk', vector[1:], gamma[1:])
    probe = np.r_[np.sqrt(1 + np.dot([.31, -.22, .17], [.31, -.22, .17])), [.31, -.22, .17]]
    covariance = float(np.linalg.norm(s @ slash(probe) @ np.linalg.inv(s) - slash(boost(probe, velocity))))
    pseudo_unitary = float(np.linalg.norm(s.conj().T @ b @ s - b))
    rest_basis = np.column_stack([band.spinor(np.zeros(3), spin) for spin in range(2)])
    spinors = s @ (np.sqrt(2 * band.m) * rest_basis)
    momentum = boost(np.r_[band.m, np.zeros(3)], velocity)
    dirac_residual = float(np.linalg.norm((slash(momentum) - band.m * identity) @ spinors))
    covariant_norm = float(np.linalg.norm(spinors.conj().T @ b @ spinors - 2 * band.m * np.eye(2)))
    energy_norm = float(np.linalg.norm(spinors.conj().T @ spinors - 2 * momentum[0] * np.eye(2)))
    residuals = {'clifford_from_hoppings': clifford, 'lorentz_lie_algebra': lorentz,
                 'slash_covariance': covariance, 'pseudo_unitary_mass_metric': pseudo_unitary,
                 'boosted_rest_dirac_equation': dirac_residual,
                 'covariant_spinor_norm': covariant_norm, 'energy_spinor_norm': energy_norm}
    assert max(residuals.values()) < 3e-13, residuals
    return {'velocity': velocity.tolist(), 'rapidity': rapidity, 'residuals': residuals,
            'interpretation': 'Continuum representation consequence of selected hopping coefficients; not dynamics selecting them or microscopic all-frame invariance.'}


def run():
    cases = [
        {'name': 'oblique_low_energy', 'mass': 1., 'charge': .30350,
         'omega_over_m': .35, 'theta': 1.05, 'incoming_direction': [.12, .2, 1.],
         'boost_velocity': [.23, -.17, .11]},
        {'name': 'different_frame_and_scale', 'mass': 1.7, 'charge': .21,
         'omega_over_m': .85, 'theta': 1.4, 'incoming_direction': [.4, -.2, 1.],
         'boost_velocity': [-.14, .27, .19]},
    ]
    refinements = [.16, .08, .04, .02]
    matched = []
    for case in cases:
        rows = [scattering_row(scale / case['mass'], case) for scale in refinements]
        errors = [row['relative_error'] for row in rows]
        assert errors[-1] < errors[0] / 8 and errors[-1] < .004, errors
        assert rows[-1]['deletion_max_normalized_amplitude_change']['no_pairs'] > 1e-4 * case['charge']**2
        matched.append({'case': case['name'], 'rows': rows,
                        'successive_error_ratios': [first / second for first, second in zip(errors[:-1], errors[1:])]})
    # Same v, mass, charge, target and improved action: only photon speed differs.
    # A mismatch is not repaired with a spacing-dependent matter-speed retune.
    mismatch_speed = 1.25
    mismatch = [scattering_row(scale / cases[0]['mass'], cases[0], mismatch_speed)
                for scale in [.08, .04, .02]]
    assert all(row['relative_error'] > .02 for row in mismatch), [row['relative_error'] for row in mismatch]
    assert mismatch[-1]['relative_error'] > .75 * mismatch[0]['relative_error']
    # Rest isotropy is a false positive: rotationally symmetric unequal limiting
    # speeds need not satisfy the boosted, common-speed continuum observable.
    rest = dict(cases[0], name='mismatched_rest_isotropy', boost_velocity=[0., 0., 0.])
    rest_isotropy = []
    for scale in [.08, .02]:
        rows = [scattering_row(scale / rest['mass'], dict(rest, incoming_direction=direction), mismatch_speed)
                for direction in ([0., 0., 1.], [1., 1., 1.])]
        reference = rows[0]['independent_continuum_invariant_kn_squared']
        spread = abs(rows[0]['pauli_spin_polarization_averaged_squared']['full']
                     - rows[1]['pauli_spin_polarization_averaged_squared']['full']) / reference
        rest_isotropy.append({'a_times_mass': scale, 'relative_orientation_spread': spread, 'rows': rows})
    assert rest_isotropy[-1]['relative_orientation_spread'] < rest_isotropy[0]['relative_orientation_spread'] / 4
    return {'scope': 'Actual Pauli filled-sea nonrest tree scattering on selected continuous-time low-zone law',
            'boost_role': 'Continuum target-input selection only; candidate energies and recoil independently solved',
            'normalization_role': '2 sqrt(Ei Ef) per connected amplitude; not finite-a exact invariant phase space',
            'matched_speed_refinement': matched,
            'limiting_speed_mismatch': {'fixed_photon_speed': mismatch_speed, 'fixed_matter_speed': 1., 'rows': mismatch},
            'rest_isotropy_false_positive': rest_isotropy,
            'derived_continuum_lorentz_action': [lorentz_algebra_probe(case['boost_velocity']) for case in cases],
            'remaining_input_debt': [
                'The selected hopping matrices, quantum postulates, continuous time, sector preparation and link Hamiltonian are inputs.',
                'Common limiting speed is selected, not derived; unequal speeds remain a material failure after refinement.',
                'Mass and charge are model inputs. 0.30350 is a model normalization, not a dressed measured fine-structure derivation.',
                'Finite spacing has frame, dispersion, recoil and vertex corrections; this finite event sweep is not an all-frame theorem.',
                'Continuum spinor algebra is not full microscopic physical rotation/Lorentz dynamics or Coulomb-minimized asymptotic preparation.',
                'The tree consumer is not thermodynamic, unquenched, radiative or continuous-time equivalence proof.'
            ]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Optional JSON report path')
    args = parser.parse_args()
    encoded = json.dumps(run(), indent=2, sort_keys=True)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as stream:
            stream.write(encoded + '\n')
    print(encoded)

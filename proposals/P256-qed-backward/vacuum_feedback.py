#!/usr/bin/env python3
"""Filled-sea link feedback; finite matrices, not an imported continuum loop.

Run with the scientific project Python. The default bounds the matrix at 864
rows. CLI exit status rejects gauge/contact/empty-sea false positives, not an
assumed sign or a continuum extrapolation. See vacuum_feedback.md for units.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from typing import Any, TypedDict

# Bound default CLI matrix cost without adding a threadpool-control dependency.
# Library imports leave their host's thread configuration alone.
if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = os.environ.get('VACUUM_FEEDBACK_THREADS', '1')

import numpy as np
from scipy.linalg import eigh

from rotation_band import RotationBand


class PairResponse(TypedDict):
    contact: float
    virtual_pair: float
    curvature: float
    no_contact: float
    no_virtual_pairs: float
    empty_negative_sea_curvature: float
    electric_omega_squared_coefficient: float
    imaginary_frequency_kernel: list[float]


@dataclass
class PeriodicBand:
    """Every coupling follows its straight primitive-link path on a 3-torus."""

    band: RotationBand
    length: int

    def __post_init__(self):
        self.sites = list(itertools.product(range(self.length), repeat=3))
        self.lookup = {site: index for index, site in enumerate(self.sites)}
        self.cells = len(self.sites)
        self.dimension = 4 * self.cells
        onsite, terms = self.band.couplings()
        self.onsite = onsite
        self.edges = []
        for site in self.sites:
            for displacement, coupling in terms:
                axis = int(np.flatnonzero(displacement)[0])
                steps = int(displacement[axis])
                path = []
                point = list(site)
                for _ in range(steps):
                    path.append((self.lookup[tuple(point)], axis))
                    point[axis] = (point[axis] + 1) % self.length
                self.edges.append((self.lookup[site], self.lookup[tuple(point)],
                                   coupling, path))

    def matrices(self, profile, amplitude=0., wrong_two_link=False):
        """Return H(t), dH/dt, d2H/dt2 at the specified amplitude.

        Row x, column x+d contains T_d exp(-i q sum_path theta), q=-1.
        The adverse construction deletes the second primitive link only from
        the improved path, leaving its coupling and all other paths unchanged.
        """
        h = np.kron(np.eye(self.cells), self.onsite)
        first = np.zeros_like(h)
        second = np.zeros_like(h)
        charge = self.band.charge
        for source, target, coupling, path in self.edges:
            selected = path[:1] if wrong_two_link and len(path) == 2 else path
            slope = sum(profile[index, axis] for index, axis in selected)
            phase = np.exp(-1j * charge * amplitude * slope)
            block = coupling * phase
            derivative = -1j * charge * slope * block
            contact = -charge**2 * slope**2 * block
            rows, cols = slice(4 * source, 4 * source + 4), slice(4 * target, 4 * target + 4)
            for matrix, value in ((h, block), (first, derivative), (second, contact)):
                matrix[rows, cols] += value
                matrix[cols, rows] += value.conj().T
        return h, first, second

    def transverse(self, mode):
        profile = np.zeros((self.cells, 3))
        profile[:, 1] = [np.cos(2 * np.pi * mode * site[0] / self.length)
                         for site in self.sites]
        return profile

    def gradient(self, mode):
        potential = np.array([np.sin(2 * np.pi * mode * site[0] / self.length)
                              + .37 * np.cos(2 * np.pi * mode * site[1] / self.length)
                              for site in self.sites])
        profile = np.zeros((self.cells, 3))
        for index, site in enumerate(self.sites):
            for axis in range(3):
                target = list(site)
                target[axis] = (target[axis] + 1) % self.length
                profile[index, axis] = potential[self.lookup[tuple(target)]] - potential[index]
        return potential, profile

    def curl_norm_squared(self, profile):
        result = 0.
        for index, site in enumerate(self.sites):
            for i, j in ((0, 1), (0, 2), (1, 2)):
                next_i, next_j = list(site), list(site)
                next_i[i] = (next_i[i] + 1) % self.length
                next_j[j] = (next_j[j] + 1) % self.length
                curl = (profile[index, i] + profile[self.lookup[tuple(next_i)], j]
                        - profile[self.lookup[tuple(next_j)], i] - profile[index, j])
                result += curl**2
        return float(result)


def filled_energy(matrix, cells):
    energies = eigh(matrix, eigvals_only=True, check_finite=False, driver='evr')
    if not (energies[2 * cells - 1] < 0 < energies[2 * cells]):
        raise ValueError('Probe closed the filled-negative/empty-positive gap.')
    return float(np.sum(energies[:2 * cells])), energies


def pair_response(energies, states, first, second, cells, frequencies=()) -> PairResponse:
    """Occupied-negative -> empty-positive virtual pairs plus actual contact."""
    occupied, empty = states[:, :2 * cells], states[:, 2 * cells:]
    gaps = energies[2 * cells:, None] - energies[None, :2 * cells]
    weights = np.abs(empty.conj().T @ first @ occupied)**2
    contact = float(np.trace(occupied.conj().T @ second @ occupied).real)
    pair = float(-2 * np.sum(weights / gaps))
    dynamic = [float(contact - 2 * np.sum(weights * gaps / (gaps**2 + omega**2)))
               for omega in frequencies]
    return PairResponse(contact=contact, virtual_pair=pair, curvature=contact + pair,
                no_contact=pair, no_virtual_pairs=contact,
                empty_negative_sea_curvature=0.,
                electric_omega_squared_coefficient=float(2 * np.sum(weights / gaps**3)),
                imaginary_frequency_kernel=dynamic)


def finite_difference(model, profile, step, baseline):
    energies = [filled_energy(model.matrices(profile, sign * step)[0], model.cells)[0]
                for sign in (-1, 1)]
    return (energies[0] + energies[1] - 2 * baseline) / step**2


def run_case(length, mode, mass_ratio, a, improved, step, gauge_amplitude) -> dict[str, Any]:
    band = RotationBand(a=a, m=mass_ratio / a, e=1., improved=improved)
    model = PeriodicBand(band, length)
    zero = np.zeros((model.cells, 3))
    h0 = model.matrices(zero)[0]
    energies, states = eigh(h0, check_finite=False, driver='evr')
    if not (energies[2 * model.cells - 1] < 0 < energies[2 * model.cells]):
        raise ValueError('Unperturbed finite sea is not gapped.')
    baseline = float(np.sum(energies[:2 * model.cells]))
    transverse = model.transverse(mode)
    _, first, contact = model.matrices(transverse)
    response = pair_response(energies, states, first, contact, model.cells)
    curl_squared = model.curl_norm_squared(transverse)
    fd = finite_difference(model, transverse, step, baseline)
    fd_half = finite_difference(model, transverse, step / 2, baseline)
    transverse_energy, transverse_spectrum = filled_energy(
        model.matrices(transverse, gauge_amplitude)[0], model.cells)

    # Exact periodic gauge transformation at finite amplitude, not just a Ward sum.
    potential, gradient = model.gradient(1)
    hg, jg, kg = model.matrices(gradient)
    gauge_response = pair_response(energies, states, jg, kg, model.cells)
    gauge_matrix = model.matrices(gradient, gauge_amplitude)[0]
    unitary = np.repeat(np.exp(1j * band.charge * gauge_amplitude * potential), 4)
    conjugated = unitary[:, None] * hg * unitary.conj()[None, :]
    gauge_energy, gauge_spectrum = filled_energy(gauge_matrix, model.cells)
    wrong_matrix, wrong_first, wrong_contact = model.matrices(
        gradient, wrong_two_link=True)
    wrong_response = pair_response(energies, states, wrong_first, wrong_contact, model.cells)
    wrong_finite = model.matrices(gradient, gauge_amplitude, wrong_two_link=True)[0]
    wrong_energy, wrong_spectrum = filled_energy(wrong_finite, model.cells)

    # A uniform time-varying link probes the missing electric-frequency sector.
    # Its STATIC value is torus holonomy curvature, not a photon mass.
    uniform = np.zeros_like(zero)
    uniform[:, 1] = 1.
    _, electric_first, electric_contact = model.matrices(uniform)
    frequencies = [band.m * .1, band.m * .2]
    electric = pair_response(energies, states, electric_first, electric_contact,
                             model.cells, frequencies)
    delta_electric = electric['electric_omega_squared_coefficient'] / (a * model.cells)
    scale = max(1., abs(gauge_response['contact']), abs(gauge_response['virtual_pair']))
    defect = float(np.max(np.abs(gauge_matrix - conjugated)))
    fd_error = abs(fd_half - response['curvature'])
    checks = dict(
        finite_gauge_unitary=defect < 1e-11 / a,
        finite_gauge_spectrum=float(np.max(np.abs(gauge_spectrum - energies))) < 1e-10 / a,
        finite_gauge_energy=abs(gauge_energy - baseline) < 1e-9 / a,
        gradient_susceptibility_cancels=abs(gauge_response['curvature']) < 1e-10 * scale,
        deleted_contact_rejected=abs(gauge_response['no_contact']) > 1e-5 / a,
        deleted_virtual_pairs_rejected=abs(gauge_response['no_virtual_pairs']) > 1e-5 / a,
        spectral_matches_energy_derivative=fd_error < 2e-4 * max(1., abs(response['curvature'])),
        nonzero_transverse_spectral_control=float(np.max(np.abs(transverse_spectrum - energies))) > 1e-5 / a,
        empty_sea_rejected=abs(response['curvature']) > 1e-6 / a
            and abs(transverse_energy - baseline) > 1e-7 / a,
    )
    if improved:
        checks['wrong_two_link_gauge_rejected'] = (
            float(np.max(np.abs(wrong_finite - conjugated))) > 1e-4 / a
            and float(np.max(np.abs(wrong_spectrum - energies))) > 1e-5 / a
            and abs(wrong_response['curvature']) > 1e-5 / a
            and abs(wrong_energy - baseline) > 1e-6 / a)
    return dict(L=length, cells=model.cells, matrix_dimension=model.dimension,
                mode=mode, a=a, ma=mass_ratio, m=band.m, improved=improved,
                physical_volume=(length * a)**3,
                physical_momentum=2 * np.pi * mode / (length * a),
                lattice_momentum=2 * np.sin(np.pi * mode / length) / a,
                curl_profile_squared=curl_squared,
                stiffness_denominator=curl_squared / a,
                DeltaChi_magnetic=a * response['curvature'] / curl_squared,
                response=response, filled_energy_zero=baseline,
                finite_difference=dict(step=step, curvature=fd, half_step_curvature=fd_half,
                                       spectral_error=fd_error),
                transverse_finite_energy_change=transverse_energy - baseline,
                gradient=dict(amplitude=gauge_amplitude, response=gauge_response,
                              unitary_defect=defect, energy_change=gauge_energy - baseline,
                              curl_profile_squared=model.curl_norm_squared(gradient)),
                wrong_two_link=dict(applicable=improved, response=wrong_response,
                                    unitary_defect=float(np.max(np.abs(wrong_finite - conjugated))),
                                    energy_change=wrong_energy - baseline,
                                    spectrum_defect=float(np.max(np.abs(wrong_spectrum - energies)))),
                electric=dict(DeltaChi_electric_low_frequency=delta_electric,
                              frequencies=frequencies, response=electric,
                              DeltaChi_electric_finite_frequency=[
                                  (kernel - electric['curvature']) / (a * model.cells * omega**2)
                                  for kernel, omega in zip(
                                      electric['imaginary_frequency_kernel'], frequencies)],
                              uniform_static_is_finite_torus_holonomy=True,
                              magnetic_electric_difference=delta_electric
                                  - a * response['curvature'] / curl_squared),
                checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', type=int, nargs='+', default=[4, 6])
    parser.add_argument('--modes', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--mass-ratios', type=float, nargs='+', default=[.05, .2])
    parser.add_argument('--a', type=float, default=1.)
    parser.add_argument('--step', type=float, default=.002)
    parser.add_argument('--gauge-amplitude', type=float, default=.31)
    parser.add_argument('--unimproved', action='store_true')
    parser.add_argument('--pure-gauge-chi', type=float, default=10.85629,
                        help='Conditional pure-sector reference only; no full-charge claim.')
    args = parser.parse_args()
    if (args.a <= 0 or args.step <= 0 or args.gauge_amplitude == 0
            or any(m <= 0 for m in args.mass_ratios)
            or any(length < 4 for length in args.lengths)
            or any(mode < 1 or 2 * mode > length
                   for length in args.lengths for mode in args.modes)):
        parser.error('Require a, step, masses > 0, nonzero gauge amplitude, L>=4, 1<=mode<=L/2.')
    rows = [run_case(length, mode, mass, args.a, not args.unimproved,
                     args.step, args.gauge_amplitude)
            for length in args.lengths for mode in args.modes for mass in args.mass_ratios]
    for row in rows:
        total = args.pure_gauge_chi + row['DeltaChi_magnetic']
        row['conditional_magnetic_normalization_only'] = dict(
            pure_gauge_chi=args.pure_gauge_chi, chi_plus_magnetic_feedback=total,
            inverse_sqrt_if_fully_matched=float(1 / np.sqrt(total)) if total > 0 else None,
            is_full_canonical_charge=False)
    signs = [row['DeltaChi_magnetic'] for row in rows]
    mass_comparisons = []
    momentum_volume_comparisons = []
    for length in sorted(set(args.lengths)):
        for mode in sorted(set(args.modes)):
            samples = sorted((row for row in rows if row['L'] == length and row['mode'] == mode),
                             key=lambda row: row['ma'])
            for light, heavy in zip(samples, samples[1:]):
                change = light['DeltaChi_magnetic'] - heavy['DeltaChi_magnetic']
                mass_comparisons.append(dict(
                    L=length, mode=mode, lighter_ma=light['ma'], heavier_ma=heavy['ma'],
                    lighter_minus_heavier_DeltaChi=change,
                    grows_with_cutoff_to_mass_ratio=change > 0,
                    controlled_continuum_log=False))
    for mass in sorted(set(args.mass_ratios)):
        samples = sorted((row for row in rows if row['ma'] == mass),
                         key=lambda row: (row['physical_momentum'], row['L']))
        momentum_volume_comparisons.append(dict(
            ma=mass, samples=[dict(L=row['L'], mode=row['mode'],
                                  ka=row['a'] * row['physical_momentum'],
                                  mLa=row['ma'] * row['L'],
                                  DeltaChi_magnetic=row['DeltaChi_magnetic'])
                             for row in samples]))
    result = dict(
        mechanism='Actual Peierls-link derivatives of a periodic RotationBand filled negative sea',
        source_charge=-1, imported_dirac_vacuum_polarization=False,
        independent_endpoint_normalization_valid=False,
        all_magnetic_coefficients_positive=all(value > 0 for value in signs),
        weak_photon_support=('Positive finite static stiffness correction only; electric/Lorentz '
                             'and unquenched matching remain unproved.' if all(value > 0 for value in signs)
                             else 'At least one finite static coefficient is nonpositive; no universal '
                                  'weak-photon inference from this finite-volume probe.'),
        rows=rows,
        mass_comparisons=mass_comparisons,
        momentum_volume_comparisons=momentum_volume_comparisons,
        all_discriminators_pass=all(all(row['checks'].values()) for row in rows),
        limitations=[
            'Finite periodic susceptibility, not a controlled continuum logarithm.',
            'The selected hopping, quantum filling, masses and integer charge remain inputs.',
            'Small ma at fixed small L also has mLa small: infrared finite-volume effects are not cutoff scaling.',
            'Uniform electric frequency derivative is calculated, but continuum Lorentz equality is not established.',
            'Pure compact chi and this free-sea determinant must be matched in the same regulator/ensemble; no unquenched measurement.',
            'No radiative Compton amplitude, interacting resummation or equilibrium Coulomb-phase claim.',
        ])
    print(json.dumps(result, indent=2))
    return 0 if result['all_discriminators_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

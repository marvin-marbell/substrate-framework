#!/usr/bin/env python3
"""Actual Pauli forward Compton response at the same sea-dressed photon pole.

Finite monochromatic photon modes have measured spectral energies/residues;
this is a connected tree electron response with dressed external photon legs,
not a complete radiative Compton amplitude.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from pauli_compton import FilledSeaConsumer
from photon_matching import match_photon
from rotation_band import RotationBand


class PhotonMode:
    """A physical oscillator mode; uncatalogued momenta are not a fake dispersion."""
    def __init__(self, momentum: np.ndarray, energy: float):
        self.momentum = momentum
        self.energy = energy

    def photon_energy(self, momentum: np.ndarray) -> float:
        if not (np.allclose(momentum, self.momentum, rtol=0, atol=1e-13)
                or np.allclose(momentum, -self.momentum, rtol=0, atol=1e-13)):
            raise ValueError('Momentum is outside this finite monochromatic mode domain.')
        return self.energy


def forward_at_pole(band: RotationBand, pole: dict) -> dict:
    """Actual Pauli consumer with the measured same-band pole and field residue."""
    parameters = dict(a=band.a, m=band.m, r=band.r, v=band.v, e=band.e, improved=band.improved)
    if pole['band_parameters'] != parameters:
        raise ValueError('The photon pole and Pauli consumer must have the same actual band.')
    k = np.array([pole['momentum'], 0., 0.])
    photons = PhotonMode(k, pole['photon_energy'])
    p = np.zeros(3)
    consumer = FilledSeaConsumer(band, photons, (p, k, p, k))
    source = pole['canonical_field_leg_factor']
    polarizations = [source * np.r_[0., vector] for vector in np.eye(3)[1:]]
    reference_squared = squared = no_pairs_squared = max_error = ward = 0.
    bare_squared = no_contact_squared = no_contact_ward = 0.
    bare_source = float(np.sqrt(band.a * pole['hamiltonian_UE1']))
    for eps in polarizations:
        for eps_out in polarizations:
            # Independent one-particle expression is a comparison, not the
            # Pauli intermediate occupation-state supplier.
            ei = band.energy(p)
            direct = np.linalg.inv((ei + photons.energy) * np.eye(4) - band.h(p + k))
            crossed = np.linalg.inv((ei - photons.energy) * np.eye(4) - band.h(p - k))
            kernel = (band.vertex(p, p + k, eps_out) @ direct @ band.vertex(p + k, p, eps)
                      + band.vertex(p, p - k, eps) @ crossed @ band.vertex(p - k, p, eps_out)
                      + band.contact(p, p, eps, eps_out, k, k))
            for spin in range(2):
                for spin_out in range(2):
                    amplitude = consumer.amplitude(eps, eps_out, spin, spin_out)['connected']
                    reference = np.vdot(band.spinor(p, spin_out), kernel @ band.spinor(p, spin))
                    no_pairs = consumer.amplitude(eps, eps_out, spin, spin_out,
                                                  keep_pairs=False)['connected']
                    # Bare source normalization is a physically different
                    # external state, not another name for the measured residue.
                    bare = consumer.amplitude(eps * (bare_source / source),
                                              eps_out * (bare_source / source),
                                              spin, spin_out)['connected']
                    no_contact = consumer.amplitude(eps, eps_out, spin, spin_out,
                                                    keep_contact=False)['connected']
                    scale = band.energy(p)**2
                    squared += scale * abs(amplitude)**2
                    reference_squared += scale * abs(reference)**2
                    no_pairs_squared += scale * abs(no_pairs)**2
                    bare_squared += scale * abs(bare)**2
                    no_contact_squared += scale * abs(no_contact)**2
                    max_error = max(max_error, abs(amplitude - reference))
                    gauge = source * np.r_[photons.energy, 2 * np.sin(band.a * k / 2) / band.a]
                    no_contact_ward = max(no_contact_ward, abs(consumer.amplitude(
                        gauge, gauge, spin, spin_out, keep_contact=False)['connected']))
                    ward = max(ward, abs(consumer.amplitude(gauge, gauge, spin, spin_out)['connected']),
                               abs(consumer.amplitude(gauge, eps_out, spin, spin_out)['connected']),
                               abs(consumer.amplitude(eps, gauge, spin, spin_out)['connected']))
    # Contact, residue and virtual-pair corrections can become physically small.
    # Their measured sizes are diagnostics, not universal lower-size gates.
    # Independent kernel equality and full Ward retain correctness coverage.
    checks = dict(actual_pauli_matches_independent_kernel=max_error < 1e-11,
                  same_pole_ward=ward < 1e-11,
                  pole_checks=all(pole['checks'].values()))
    checks = {name: bool(value) for name, value in checks.items()}
    soft_coefficient = band.v**2 + (0. if band.improved else band.m * band.a * band.r)
    return dict(L=pole['L'], band_parameters=parameters, photon_pole=pole,
                incoming_momentum=k.tolist(), photon_energy=photons.energy,
                canonical_field_leg_factor=source, bare_field_leg_factor=bare_source,
                pauli_forward_squared=squared, independent_forward_squared=reference_squared,
                no_virtual_pairs_squared=no_pairs_squared, bare_source_squared=bare_squared,
                no_contact_squared=no_contact_squared, no_contact_ward_residual=no_contact_ward,
                max_amplitude_residual=max_error, ward_residual=ward,
                soft_rest_forward_squared=4 * band.e**4 * source**4 * soft_coefficient**2,
                parity_marker='neutral hidden Pauli boundary marker; not external SU4 B4/bar4',
                checks=checks)


def forward_response(length: int, band: RotationBand, *, electric_coefficient: float,
                     magnetic_coefficient: float) -> dict:
    pole = match_photon(length, band, electric_coefficient=electric_coefficient,
                        magnetic_coefficient=magnetic_coefficient)
    return forward_at_pole(band, pole)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', type=int, nargs='+', default=[16, 32, 48])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    # Preserve the previous unit-spacing factory's physical rescaling explicitly:
    # h_phys=h_unit/a, UE1_phys=UE1_unit/a, K1_phys=K1_unit/a.
    a = .2
    band = RotationBand(a=a, m=.2 / a, e=1., r=1., v=1., improved=True)
    rows = [forward_response(length, band, electric_coefficient=.09 / a,
                             magnetic_coefficient=(1 / .09) / a) for length in args.lengths]
    result = dict(status='same_primitive_sea_photon_pole_and_connected_pauli_response',
                  scope='forward monochromatic tree electron response with dressed external photon legs',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=rows, all_checks_pass=all(all(row['checks'].values()) for row in rows))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

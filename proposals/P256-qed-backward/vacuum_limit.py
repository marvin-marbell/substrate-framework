#!/usr/bin/env python3
"""Exact momentum reduction of the same finite filled-sea link Hamiltonian."""
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
from rotation_band import RotationBand


def transverse_spectrum(length: int, band: RotationBand, mode: int = 1,
                        *, gradient: bool = False, polarization_axis: int = 1) -> dict:
    """Compact-link-angle response of the REQUIRED caller-owned band.

    Momenta are physical (1/length); the link profile is theta=t cos(q.x).
    Thus the differentiated vertex is d h/d theta, not d h/d physical A.
    The latter is a times this vertex. Spatial normalization is N/2.
    """
    if not isinstance(band, RotationBand):
        raise TypeError('A caller-owned RotationBand is required.')
    if not all(np.isfinite(x) for x in (band.a, band.m, band.r, band.v, band.e)):
        raise ValueError('Finite band parameters required.')
    if length < 3 or not 0 < mode < length / 2 or band.a <= 0 or band.m <= 0:
        raise ValueError('Require positive spacing/mass and distinct nonzero +/-Q channels.')
    if polarization_axis not in (1, 2):
        raise ValueError('The axis-x profile has transverse polarization y or z.')
    onsite, terms = band.couplings()
    grid = np.moveaxis(np.indices((length,) * 3), 0, -1)
    momenta = grid.reshape(-1, 3) * (2 * np.pi / (band.a * length))
    qangle = 2 * np.pi * mode / length
    qwave = qangle / band.a
    cells = length**3
    h = np.broadcast_to(onsite, (cells, 4, 4)).copy()
    for displacement, coupling in terms:
        phase = np.exp(1j * band.a * (momenta @ displacement))
        h += phase[:, None, None] * coupling + phase.conj()[:, None, None] * coupling.conj().T
    energies, states = np.linalg.eigh(h)
    if not (np.all(energies[:, 1] < 0) and np.all(energies[:, 2] > 0)):
        raise ValueError('The selected negative sea must remain gapped.')
    occupied = states[:, :, :2]
    contact_matrix = np.zeros_like(h)
    channels = []
    for sign in (-1, 1):
        vertex = np.zeros_like(h)
        for displacement, coupling in terms:
            axis = int(np.flatnonzero(displacement)[0])
            steps = int(displacement[axis])
            amplitude = (np.exp(1j * sign * qangle) - 1.) if gradient and axis == 0 else (
                1. if not gradient and axis == polarization_axis else 0.)
            path = amplitude * sum(np.exp(1j * sign * qangle * r * (axis == 0))
                                   for r in range(steps))
            phase = np.exp(1j * band.a * (momenta @ displacement))
            reverse_phase = np.exp(-1j * (band.a * (momenta @ displacement)
                                          + sign * qangle * displacement[0]))
            vertex += (-.5j * band.charge * path) * (
                phase[:, None, None] * coupling - reverse_phase[:, None, None] * coupling.conj().T)
            if sign == 1:
                contact_matrix += (-.5 * band.charge**2 * abs(path)**2) * (
                    phase[:, None, None] * coupling + phase.conj()[:, None, None] * coupling.conj().T)
        shifted_states = np.roll(states.reshape((length,) * 3 + (4, 4)),
                                 -sign * mode, axis=0).reshape(cells, 4, 4)
        shifted_energies = np.roll(energies.reshape((length,) * 3 + (4,)),
                                   -sign * mode, axis=0).reshape(cells, 4)
        gaps = shifted_energies[:, 2:, None] - energies[:, None, :2]
        weights = abs(shifted_states[:, :, 2:].conj().swapaxes(1, 2) @ vertex @ occupied)**2
        channels.append((gaps, weights))
    contact = float(np.trace(occupied.conj().swapaxes(1, 2) @ contact_matrix @ occupied,
                             axis1=1, axis2=2).real.sum())
    return dict(cells=cells, qwave=qwave, qangle=qangle, band=band, terms=terms,
                momenta=momenta, energies=energies, states=states, contact=contact,
                channels=channels, polarization_axis=polarization_axis,
                source_coordinate='compact link theta=a*physical A; charge belongs to band')


def momentum_response(length: int, band: RotationBand, mode: int = 1,
                      *, gradient: bool = False) -> dict:
    """Compact-angle susceptibilities for a caller-owned physical-spacing band.

    Profiles are cos(q.x) e_y, or the exact forward link gradient of cos(q.x).
    Distinct +/-q channels exclude zero/Nyquist. No parameter is reinterpreted.
    """
    spectrum = transverse_spectrum(length, band, mode, gradient=gradient)
    cells, qwave, band = spectrum['cells'], spectrum['qwave'], spectrum['band']
    terms, momenta = spectrum['terms'], spectrum['momenta']
    energies, states = spectrum['energies'], spectrum['states']
    occupied = states[:, :, :2]
    contact = spectrum['contact']
    pair = -2 * sum(float(np.sum(weights / gaps)) for gaps, weights in spectrum['channels'])
    uniform_vertex = np.zeros((cells, 4, 4), dtype=complex)
    uniform_contact = np.zeros_like(uniform_vertex)
    for displacement, coupling in terms:
        steps = displacement[1]
        phase = np.exp(1j * band.a * (momenta @ displacement))
        uniform_vertex += (-1j * band.charge * steps) * (
            phase[:, None, None] * coupling - phase.conj()[:, None, None] * coupling.conj().T)
        uniform_contact += (-band.charge**2 * steps**2) * (
            phase[:, None, None] * coupling + phase.conj()[:, None, None] * coupling.conj().T)
    gaps = energies[:, 2:, None] - energies[:, None, :2]
    weights = abs(states[:, :, 2:].conj().swapaxes(1, 2) @ uniform_vertex @ occupied)**2
    electric = 2 * float(np.sum(weights / gaps**3)) / cells
    uniform_static = float(np.trace(occupied.conj().swapaxes(1, 2) @ uniform_contact @ occupied,
                                    axis1=1, axis2=2).real.sum()) - 2 * float(np.sum(weights / gaps))
    curvature = contact + pair
    denominator = cells * (2 * np.sin(spectrum['qangle'] / 2))**2 / 2
    return dict(L=length, a=band.a, mass=band.m, ma=band.m * band.a,
                mL=band.m * band.a * length, mode=mode,
                momentum_over_mass=qwave / band.m, improved=band.improved, gradient=gradient,
                contact=contact, virtual_pair=pair, curvature=curvature,
                DeltaChi_magnetic=None if gradient else curvature / denominator,
                DeltaChi_electric=electric,
                uniform_static_curvature_per_cell=uniform_static / cells,
                gradient_relative_residual=abs(curvature) / max(1., abs(contact), abs(pair))
                    if gradient else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', nargs='+', type=int, default=[8, 16, 24, 32])
    parser.add_argument('--mass-ratios', nargs='+', type=float, default=[.05, .2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from vacuum_feedback import run_case
    controls = []
    for length in (4, 6):
        dense = run_case(length, 1, .05, 1., True, .002, .31)
        band = RotationBand(a=1., m=.05, e=1., r=1., v=1., improved=True)
        fourier = momentum_response(length, band)
        gradient = momentum_response(length, band, gradient=True)
        errors = dict(magnetic=abs(fourier['DeltaChi_magnetic'] - dense['DeltaChi_magnetic']),
                      electric=abs(fourier['DeltaChi_electric']
                                   - dense['electric']['DeltaChi_electric_low_frequency']),
                      contact=abs(fourier['contact'] - dense['response']['contact']),
                      pair=abs(fourier['virtual_pair'] - dense['response']['virtual_pair']))
        checks = dict(exact_dense_reduction=max(errors.values()) < 1e-8,
                      longitudinal_ward=gradient['gradient_relative_residual'] < 1e-12,
                      deleted_contact_fails=abs(gradient['virtual_pair']) > 1.,
                      deleted_pairs_fail=abs(gradient['contact']) > 1.,
                      dense_adverse_controls=all(dense['checks'].values()))
        controls.append(dict(L=length, errors=errors, checks=checks))
    rows = [momentum_response(length, RotationBand(
        a=1., m=ma, e=1., r=1., v=1., improved=True))
        for ma in args.mass_ratios for length in args.lengths]
    fixed_volume = [momentum_response(length, RotationBand(
        a=1., m=6.4 / length, e=1., r=1., v=1., improved=True))
        for length in args.lengths]
    result = dict(status='exact_finite_fourier_response_and_executed_limit_sequences',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  controls=controls, fixed_cutoff_volume_sequences=rows,
                  fixed_mass_physical_volume_cutoff_sequence=fixed_volume,
                  all_controls_pass=all(all(row['checks'].values()) for row in controls))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_controls_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

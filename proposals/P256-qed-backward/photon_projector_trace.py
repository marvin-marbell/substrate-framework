#!/usr/bin/env python3
"""Exact continuous photon tensor reduction, checked on the actual matrix law."""
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
from angular_photon_spectrum import ContinuousSea
from rotation_band import RotationBand


def matrix_tensor_reference(sea: ContinuousSea, q: np.ndarray) -> dict:
    """Independent occupied/empty eigenstate sum of the primitive matrix links."""
    band = sea.band
    onsite, terms = band.couplings()

    def blocks(points):
        h = np.broadcast_to(onsite, (len(points), 4, 4)).copy()
        for displacement, coupling in terms:
            phase = np.exp(1j * (points @ displacement))
            h += phase[:, None, None] * coupling + phase.conj()[:, None, None] * coupling.conj().T
        return np.linalg.eigh(h)

    energies, states = blocks(sea.p)
    occupied = states[:, :, :2]
    contact = np.zeros((3, 3))
    direct_contact = np.zeros((3, 3))
    projector = occupied @ occupied.conj().swapaxes(1, 2)
    empty = np.eye(4) - projector
    derivatives = np.zeros((len(sea.p), 3, 4, 4), dtype=complex)
    pieces = []
    for displacement, coupling in terms:
        axis = int(np.flatnonzero(displacement)[0])
        steps = int(displacement[axis])
        phase = np.exp(1j * (sea.p @ displacement))[:, None, None]
        derivative = 1j * steps * (phase * coupling - phase.conj() * coupling.conj().T)
        derivatives[:, axis] += derivative
        pieces.append((axis, steps, derivative))
    for axis, steps, derivative in pieces:
        full = derivatives[:, axis]
        Pprime = -(empty @ full @ projector + projector @ full @ empty)
        Pprime /= (energies[:, 2] - energies[:, 0])[:, None, None]
        values = -np.trace(Pprime @ derivative, axis1=1, axis2=2).real
        path = sum(np.cos(q[axis] * (r - (steps - 1) / 2)) for r in range(steps))
        average = float(values.mean() if sea.weights is None else np.dot(sea.weights, values))
        contact[axis, axis] += band.charge**2 * (path / steps)**2 * average
    responses = {kind: np.zeros((3, 3)) for kind in ('static', 'derivative', 'change')}
    for sign in (-1, 1):
        vertex = np.zeros((len(sea.p), 3, 4, 4), dtype=complex)
        for displacement, coupling in terms:
            axis = int(np.flatnonzero(displacement)[0])
            steps = int(displacement[axis])
            path = sum(np.exp(1j * sign * q[axis] * (r + .5)) for r in range(steps))
            phase = np.exp(1j * (sea.p @ displacement))
            reverse = np.exp(-1j * ((sea.p + sign * q) @ displacement))
            vertex[:, axis] += (-.5j * band.charge * path) * (
                phase[:, None, None] * coupling - reverse[:, None, None] * coupling.conj().T)
            if sign == 1:
                c = (-.5 * band.charge**2 * abs(path)**2) * (
                    phase[:, None, None] * coupling + phase.conj()[:, None, None] * coupling.conj().T)
                values = np.trace(occupied.conj().swapaxes(1, 2) @ c @ occupied,
                                  axis1=1, axis2=2).real
                direct_contact[axis, axis] += 2 * float(values.mean() if sea.weights is None
                                                       else np.dot(sea.weights, values))
        shifted_energies, shifted_states = blocks(sea.p + sign * q)
        gaps = shifted_energies[:, 2:, None] - energies[:, None, :2]
        amp = np.einsum('nxa,nixy,nyb->niab', shifted_states[:, :, 2:].conj(),
                        vertex, occupied, optimize=True)
        for kind in responses:
            if kind == 'static':
                factor = 1 / gaps
            elif kind == 'derivative':
                factor = gaps / (gaps**2 - .01)**2
            else:
                factor = .01 / (gaps * (gaps**2 - .01))
            if sea.weights is not None:
                factor *= sea.weights[:, None, None]
            normalization = 1 / len(sea.p) if sea.weights is None else 1.
            responses[kind] += 4 * normalization * np.einsum(
                'niab,njab,nab->ij', amp.conj(), amp, factor, optimize=True).real
    return dict(contact=contact, direct_link_contact=direct_contact, **responses)


def run() -> dict:
    rows = []
    for rule, mass, q in (('midpoint', .4, [.16, .09, .04]),
                          ('split_gauss', .1, [.04, .03, .02])):
        band = RotationBand(a=1., m=mass, r=1.87, v=1.04, e=1., improved=True)
        sea = ContinuousSea(8, band=band, quadrature=rule)
        momentum = np.array(q)
        tensor = sea.spectrum(momentum)
        reference = matrix_tensor_reference(sea, momentum)
        errors = dict(contact=float(np.linalg.norm(tensor.contact - reference['contact'])))
        for kind in ('static', 'derivative', 'change'):
            s = 0. if kind == 'static' else .01
            errors[kind] = float(np.linalg.norm(tensor.spectral(s, kind) - reference[kind]))
        minimum = min(float(np.linalg.eigvalsh(weights).min()) for _, weights in tensor.channels)
        rows.append(dict(quadrature=rule, legendre_or_midpoint_order=8,
                         points=len(sea.p), ma=mass, q=q, matrix_tensor_errors=errors,
                         minimum_point_spectral_eigenvalue=minimum,
                         periodic_contact_quadrature_difference=float(np.linalg.norm(
                             reference['contact'] - reference['direct_link_contact'])),
                         normalized_measure=1. if sea.weights is None else float(sea.weights.sum()),
                         checks=dict(actual_matrix_tensor_identity=max(errors.values()) < 1e-11,
                                     positive_point_spectral_measure=minimum > -1e-12)))
    return dict(status='exact_clifford_projector_photon_tensor',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                spectrum_source_sha256=hashlib.sha256(Path(__file__).with_name('angular_photon_spectrum.py').read_bytes()).hexdigest(),
                scope='Same selected primitive scalar links and occupied sea; exact spectral trace, not a new photon approximation',
                rows=rows, all_checks_pass=all(all(row['checks'].values()) for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

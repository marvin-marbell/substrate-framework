#!/usr/bin/env python3
"""Resolved spin/photon-state map, including a negative-band deletion falsifier."""
from __future__ import annotations

import json
import numpy as np

from compton import amplitude, polarizations
from lattice_compton import I4, WilsonLink


def positive_only_kernel(model, p, k, pf, kf, eps, epsf):
    """Deliberately wrong: delete all internal negative-energy band poles."""
    def resolvent(z, momentum):
        energy = model.energy(momentum)
        return (I4 + model.h(momentum) / energy) / (2 * (z - energy))
    ei = model.energy(p)
    eo = np.conjugate(epsf)
    return (model.vertex(pf, p + k, eo) @ resolvent(ei + model.photon_energy(k), p + k)
            @ model.vertex(p + k, p, eps)
            + model.vertex(pf, p - kf, eps) @ resolvent(ei - model.photon_energy(kf), p - kf)
            @ model.vertex(p - kf, p, eo)
            + model.contact(pf, p, eps, eo, k, kf))


def run():
    rows = []
    for spacing in [0.05, 0.025, 0.0125]:
        model = WilsonLink(spacing, improved=True)
        errors = []
        for omega in [0.05, 0.4, 1., 2.]:
            for theta in [0.2, 0.7, 1.5, 2.6]:
                for direction in [np.array([0., 0., 1.]), np.array([1., 2., 3.])]:
                    p, k, pf, kf = model.kinematics(omega, theta, direction)
                    nin, nout = k / np.linalg.norm(k), kf / np.linalg.norm(kf)
                    omega_out = omega / (1 + omega * (1 - np.cos(theta)))
                    cp = np.array([1., 0., 0., 0.])
                    ck, ckf = np.r_[omega, omega * nin], np.r_[omega_out, omega_out * nout]
                    cpf = cp + ck - ckf
                    # Match basis convention before comparing components. Lattice
                    # transverse space is perpendicular to khat, not continuum k.
                    lep = polarizations(np.r_[model.photon_energy(k), model.khat(k)])
                    lepf = polarizations(np.r_[model.photon_energy(kf), model.khat(kf)])
                    cep, cepf = polarizations(ck), polarizations(ckf)
                    # Circular combinations expose outgoing complex conjugation.
                    for circular in [False, True]:
                        if circular:
                            incoming = [(lep[0] + 1j * lep[1]) / np.sqrt(2)]
                            outgoing = [(lepf[0] - 1j * lepf[1]) / np.sqrt(2)]
                            cin = [(cep[0] + 1j * cep[1]) / np.sqrt(2)]
                            cout = [(cepf[0] - 1j * cepf[1]) / np.sqrt(2)]
                        else:
                            incoming, outgoing, cin, cout = lep, lepf, cep, cepf
                        for i, eps in enumerate(incoming):
                            for j, epsf in enumerate(outgoing):
                                kernel = model.kernel(p, k, pf, kf, eps, epsf)
                                for spin in range(2):
                                    for spinf in range(2):
                                        mapped = (-2 * np.sqrt(model.energy(p) * model.energy(pf))
                                                  * np.vdot(model.spinor(pf, spinf),
                                                            kernel @ model.spinor(p, spin)))
                                        reference = amplitude(cp, ck, cpf, ckf, cin[i], cout[j], spin, spinf)
                                        errors.append(abs(mapped - reference))
        row = dict(a=spacing, max_component_absolute_error_over_e2=float(max(errors)),
                   scope='linear and circular photon states, fixed-z boosted electron spins')
        rows.append(row)
    assert all(x['max_component_absolute_error_over_e2'] > y['max_component_absolute_error_over_e2']
               for x, y in zip(rows, rows[1:])), rows
    assert rows[-1]['max_component_absolute_error_over_e2'] < 0.005, rows
    model = WilsonLink(0.1, improved=True)
    p, k, pf, kf = model.kinematics(0.4, 1.1)
    eps, epsf = model.polarizations(k)[0], model.polarizations(kf)[0]
    full, truncated = model.kernel(p, k, pf, kf, eps, epsf), positive_only_kernel(model, p, k, pf, kf, eps, epsf)
    difference = max(abs(np.vdot(model.spinor(pf, sf), (full - truncated) @ model.spinor(p, s)))
                     for s in range(2) for sf in range(2))
    wrong_ward = positive_only_kernel(model, p, k, pf, kf,
                                     np.r_[model.photon_energy(k), model.khat(k)], epsf)
    ward_residual = max(abs(np.vdot(model.spinor(pf, sf), wrong_ward @ model.spinor(p, s)))
                        for s in range(2) for sf in range(2))
    assert difference > 0.01 and ward_residual > 1e-4, (difference, ward_residual)
    return dict(resolved_state_convergence=rows,
                negative_band_deletion=dict(amplitude_difference=float(difference),
                                            incoming_ward_residual=float(ward_residual)),
                scope='conditional state map; CAR and Clifford origin not explained')


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))

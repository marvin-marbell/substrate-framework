#!/usr/bin/env python3
"""Continuous-time Wilson-link Compton kernel; conditional quantum supplier.

m=e=1 by default. Link fields live at midpoints. Finite-a results are
reference-normalized matrix elements, NOT a continuum cross section.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from compton import GAMMA

I4 = np.eye(4, dtype=complex)
BETA = GAMMA[0]
ALPHA = np.array([GAMMA[0] @ GAMMA[i] for i in range(1, 4)])
SIGMA = np.array([ALPHA[i][:2, 2:] for i in range(3)])


@dataclass(frozen=True)
class WilsonLink:
    a: float
    m: float = 1.0
    e: float = 1.0
    r: float = 1.0
    v: float = 1.0
    photon_speed: float = 1.0
    improved: bool = False

    def mass(self, p):
        delta = 2 * np.sin(self.a * p / 2)**2
        return self.m + self.r / self.a * np.sum(delta**2 if self.improved else delta)

    def h(self, p):
        return (np.einsum('i,ijk->jk', self.v * np.sin(self.a * p) / self.a, ALPHA)
                + self.mass(p) * BETA)

    def energy(self, p):
        return float(np.sqrt(np.sum((self.v * np.sin(self.a * p) / self.a)**2)
                             + self.mass(p)**2))

    def velocity(self, p):
        x = self.a * p
        dm = self.r * np.sin(x)
        if self.improved:
            dm *= 4 * np.sin(x / 2)**2
        return (self.v**2 * np.sin(x) * np.cos(x) / self.a + self.mass(p) * dm) / self.energy(p)

    def spinor(self, p, spin):
        energy, mass = self.energy(p), self.mass(p)
        chi = np.eye(2, dtype=complex)[:, spin]
        lower = np.einsum('i,ijk->jk', self.v * np.sin(self.a * p) / self.a, SIGMA) @ chi
        return np.concatenate((np.sqrt((energy + mass) / (2 * energy)) * chi,
                               lower / np.sqrt(2 * energy * (energy + mass))))

    def khat(self, k):
        return 2 * np.sin(self.a * k / 2) / self.a

    def photon_energy(self, k):
        return float(self.photon_speed * np.linalg.norm(self.khat(k)))

    def photon_velocity(self, k):
        return self.photon_speed**2 * np.sin(self.a * k) / (self.a * self.photon_energy(k))

    def vertex(self, pout, pin, eps):
        midpoint = self.a * (pout + pin) / 2
        mass_vertex = self.r * np.sin(midpoint)
        if self.improved:
            transfer = self.a * (pout - pin) / 2
            mass_vertex = self.r * (2 * np.sin(midpoint)
                                   - np.cos(transfer) * np.sin(2 * midpoint))
        derivatives = (self.v * np.cos(midpoint)[:, None, None] * ALPHA
                       + mass_vertex[:, None, None] * BETA)
        # Charge sign irrelevant to two-photon tree kernel, but same q at every vertex.
        q = -self.e
        return q * (eps[0] * I4 - np.einsum('i,ijk->jk', eps[1:], derivatives))

    def contact(self, pout, pin, eps_in, eps_out, k, kout):
        midpoint = self.a * (pout + pin) / 2
        mass_curvature = self.r * np.cos(midpoint)
        if self.improved:
            mass_curvature = self.r * (2 * np.cos(midpoint)
                                      - 2 * np.cos(self.a * k / 2)
                                      * np.cos(self.a * kout / 2) * np.cos(2 * midpoint))
        curvature = self.a * (-self.v * np.sin(midpoint)[:, None, None] * ALPHA
                             + mass_curvature[:, None, None] * BETA)
        return self.e**2 * np.einsum('i,i,ijk->jk', eps_in[1:], eps_out[1:], curvature)

    def kinematics(self, momentum, theta, direction=None):
        nin = np.array([0., 0., 1.]) if direction is None else np.array(direction, dtype=float)
        nin /= np.linalg.norm(nin)
        # Rotate in a plane containing nin; default is the x-z scattering plane.
        axis = np.array([1., 0., 0.])
        tangent = axis - np.dot(axis, nin) * nin
        if np.linalg.norm(tangent) < 0.1:
            axis = np.array([0., 1., 0.])
            tangent = axis - np.dot(axis, nin) * nin
        tangent /= np.linalg.norm(tangent)
        nout = np.cos(theta) * nin + np.sin(theta) * tangent
        k = momentum * nin
        omega = self.photon_energy(k)
        balance = lambda t: self.energy(k - t * nout) + self.photon_energy(t * nout) - self.m - omega
        # Scoped low-Brillouin-zone branch; no Umklapp or high-momentum root claim.
        kout_norm = brentq(balance, 0., 2 * momentum, xtol=1e-14)
        kout = kout_norm * nout
        return np.zeros(3), k, k - kout, kout

    def polarizations(self, k):
        n = self.khat(k) / np.linalg.norm(self.khat(k))
        seed = np.eye(3)[np.argmin(np.abs(n))]
        first = np.cross(n, seed)
        first /= np.linalg.norm(first)
        second = np.cross(n, first)
        return (np.r_[0., first], np.r_[0., second])

    def kernel(self, p, k, pout, kout, eps_in, eps_out, contact=True):
        ei = self.energy(p)
        gs = np.linalg.inv((ei + self.photon_energy(k)) * I4 - self.h(p + k))
        gu = np.linalg.inv((ei - self.photon_energy(kout)) * I4 - self.h(p - kout))
        eo = np.conjugate(eps_out)
        result = (self.vertex(pout, p + k, eo) @ gs @ self.vertex(p + k, p, eps_in)
                  + self.vertex(pout, p - kout, eps_in) @ gu @ self.vertex(p - kout, p, eo))
        if contact:
            result += self.contact(pout, p, eps_in, eo, k, kout)
        return result

    def squared(self, momentum, theta, contact=True, direction=None):
        p, k, pf, kf = self.kinematics(momentum, theta, direction)
        total = 0.
        for eps in self.polarizations(k):
            for epsf in self.polarizations(kf):
                kernel = self.kernel(p, k, pf, kf, eps, epsf, contact)
                for s in range(2):
                    for sf in range(2):
                        amp = (2 * np.sqrt(self.energy(p) * self.energy(pf))
                               * np.vdot(self.spinor(pf, sf), kernel @ self.spinor(p, s)))
                        total += abs(amp)**2 / 4
        return float(total)

    def cross_section(self, momentum, theta, direction=None):
        """Tree dσ/dΩ_k: canonical photon field, wave-vector solid angle."""
        p, k, pf, kf = self.kinematics(momentum, theta, direction)
        incoming_flux = np.linalg.norm(self.photon_velocity(k))
        nout = kf / np.linalg.norm(kf)
        radial_jacobian = abs(np.dot(self.photon_velocity(kf) - self.velocity(pf), nout))
        reduced_squared = self.squared(momentum, theta, direction=direction) / (
            4 * self.energy(p) * self.energy(pf))
        return float(reduced_squared * np.dot(kf, kf) / (
            16 * np.pi**2 * self.photon_energy(k) * self.photon_energy(kf)
            * incoming_flux * radial_jacobian))

    def ward(self, momentum, theta, contact=True, direction=None):
        p, k, pf, kf = self.kinematics(momentum, theta, direction)
        win = np.r_[self.photon_energy(k), self.khat(k)]
        wout = np.r_[self.photon_energy(kf), self.khat(kf)]
        residuals = []
        for incoming, outgoing in [(win, epsf) for epsf in self.polarizations(kf)] + [
                (eps, wout) for eps in self.polarizations(k)]:
            kernel = self.kernel(p, k, pf, kf, incoming, outgoing, contact)
            for s in range(2):
                for sf in range(2):
                    residuals.append(abs(np.vdot(self.spinor(pf, sf), kernel @ self.spinor(p, s))))
        return float(max(residuals))


def run():
    from compton import unpolarized_squared
    theta, momentum = 1.1, 0.4
    reference = unpolarized_squared(momentum, theta)
    recoil_ratio = 1 / (1 + momentum * (1 - np.cos(theta)))
    reference_rate = reference * recoil_ratio**2 / (64 * np.pi**2)
    rows = []
    for a in [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]:
        model = WilsonLink(a)
        squared = model.squared(momentum, theta)
        row = dict(a=a, squared=squared, reference=reference,
                   relative_error=abs(squared / reference - 1),
                   ward=model.ward(momentum, theta),
                   no_contact_ward=model.ward(momentum, theta, contact=False))
        assert row['ward'] < 2e-11, row
        assert row['no_contact_ward'] > 1e-5, row
        rows.append(row)
    errors = [row['relative_error'] for row in rows]
    assert all(x > y for x, y in zip(errors, errors[1:])), rows
    assert errors[-1] < 0.02, rows
    # Low-energy target selects inertial mass, not the rest gap independently.
    thomson = 2 * (1 + np.cos(theta)**2)
    soft = []
    for a in [0.2, 0.1, 0.05]:
        model = WilsonLink(a)
        ratio = model.squared(1e-5, theta) / thomson
        expected = (model.v**2 + model.r * a * model.m)**2
        assert abs(ratio / expected - 1) < 5e-5, (a, ratio, expected)
        soft.append(dict(a=a, ratio=ratio, expected=expected))
    # Material repair: tune kinetic speed so inertial mass equals rest mass at finite a.
    repaired = WilsonLink(0.1, v=np.sqrt(0.9))
    soft_repaired = repaired.squared(1e-5, theta) / thomson
    assert abs(soft_repaired - 1) < 5e-5, soft_repaired
    # That repair cannot by itself restore the full relativistic observable.
    hard_repaired = repaired.squared(momentum, theta) / reference
    assert abs(hard_repaired - 1) > 1e-3, hard_repaired
    orientations = [np.array([0., 0., 1.]), np.ones(3) / np.sqrt(3)]
    anisotropy = []
    for a in [0.2, 0.1, 0.05]:
        model = WilsonLink(a)
        values = [model.squared(momentum, theta, direction=n) for n in orientations]
        anisotropy.append(dict(a=a, relative_spread=abs(values[0] - values[1]) / reference))
    assert anisotropy[-1]['relative_spread'] < anisotropy[0]['relative_spread'], anisotropy
    # Better repair: cancel Wilson p^2 curvature using covariant two-link paths,
    # retaining heavy Brillouin-zone corners. Vertices/contact change WITH the band.
    improved_rows = []
    for a in [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]:
        model = WilsonLink(a, improved=True)
        row = dict(a=a, squared=model.squared(momentum, theta),
                   ward=model.ward(momentum, theta),
                   no_contact_ward=model.ward(momentum, theta, contact=False),
                   thomson_ratio=model.squared(1e-5, theta) / thomson)
        row['relative_error'] = abs(row['squared'] / reference - 1)
        row['cross_section'] = model.cross_section(momentum, theta)
        row['cross_section_relative_error'] = abs(row['cross_section'] / reference_rate - 1)
        assert row['ward'] < 2e-11, row
        assert abs(row['thomson_ratio'] - 1) < 5e-5, row
        improved_rows.append(row)
    improved_errors = [row['relative_error'] for row in improved_rows]
    assert all(x > y for x, y in zip(improved_errors, improved_errors[1:])), improved_rows
    assert improved_errors[-1] < errors[-1] / 10, improved_rows
    assert improved_rows[0]['no_contact_ward'] > 1e-5, improved_rows[0]
    rate_errors = [row['cross_section_relative_error'] for row in improved_rows]
    assert all(x > y for x, y in zip(rate_errors, rate_errors[1:])), improved_rows
    assert rate_errors[-1] < 1e-4, improved_rows
    corners = {}
    for name, model in [('naive', WilsonLink(0.1, r=0)),
                        ('wilson', WilsonLink(0.1)),
                        ('two_link', WilsonLink(0.1, improved=True))]:
        corners[name] = [model.energy(np.pi / model.a * np.array([int(i < n) for i in range(3)]))
                         for n in range(4)]
    assert abs(corners['naive'][3] - 1) < 1e-12, corners
    assert corners['wilson'][1] > 20 and corners['two_link'][1] > 40, corners
    # Untuned complementary kinematics, oblique orientations and charge/mass scales.
    sweep = []
    for m, e in [(1., 1.), (2., 0.3)]:
        for omega_over_m in [0.05, 0.4, 1., 2.]:
            for angle in [0.2, 0.7, 1.5, 2.6]:
                omega = m * omega_over_m
                ref_sq = unpolarized_squared(omega, angle, m=m, e=e)
                recoil = 1 / (1 + omega_over_m * (1 - np.cos(angle)))
                ref_rate = ref_sq * recoil**2 / (64 * np.pi**2 * m**2)
                for direction in [np.array([0., 0., 1.]), np.array([1., 2., 3.])]:
                    model = WilsonLink(0.0125 / m, m=m, e=e, improved=True)
                    rate = model.cross_section(omega, angle, direction=direction)
                    error = abs(rate / ref_rate - 1)
                    ward = model.ward(omega, angle, direction=direction)
                    assert error < 0.003, (m, e, omega_over_m, angle, direction, error)
                    assert ward < 2e-10 * e**2, (m, e, omega_over_m, angle, ward)
                    sweep.append(dict(m=m, e=e, omega_over_m=omega_over_m,
                                      theta=angle, direction=direction.tolist(),
                                      am=model.a * m, rate=rate, reference_rate=ref_rate,
                                      relative_error=error, ward=ward))
    return dict(scope='conditional Wilson-link tree scattering, canonical photon normalization',
                reference_cross_section=reference_rate,
                convergence=rows, thomson_inertial_mass=soft,
                soft_only_repair=dict(v=repaired.v, thomson_ratio=soft_repaired,
                                      finite_energy_ratio=hard_repaired),
                orientation_check=anisotropy, two_link_repair=improved_rows,
                doubler_corner_gaps=corners, independent_kinematics=sweep)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))

#!/usr/bin/env python3
"""Local four-orbital band, without imported gamma matrices or spin labels.

The internal matrix units are bosonic qudit transitions. Spectral isotropy and
minimal block dimension SELECT their hopping pattern; they are not a dynamical
prediction of generic bosons. Many-body fermionic filling needs the statistics
supplier. Run directly for deterministic discriminating numerical evidence.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)
E00, E01 = np.outer(I2[:, 0], I2[:, 0]), np.outer(I2[:, 0], I2[:, 1])
E10, E11 = E01.T, np.outer(I2[:, 1], I2[:, 1])
# Matrix units on two initially unassigned internal labels, NOT spatial spin.
TRANSITIONS = np.array([E01 + E10, -1j * (E01 - E10), E00 - E11])
X, Y, Z = TRANSITIONS
HADAMARD = (X + Z) / np.sqrt(2)
TO_WILSON = np.kron(HADAMARD, I2)
HOPPING = np.array([np.kron(Z, operator) for operator in TRANSITIONS])
MIXING = np.kron(X, I2)
# Solve the vector-action commutators; the half-angle is an output of the algebra.
ROTATION = np.array([-0.5j * HOPPING[j] @ HOPPING[k]
                     for j, k in [(1, 2), (2, 0), (0, 1)]])
CHARGE_CONJUGATION = np.kron(Y, Y)


def norm(matrix):
    return float(np.linalg.norm(matrix))


def isotropy_defect(operators):
    """Operator identity needed for two degenerate isotropic linear bands."""
    return max(norm(operators[i] @ operators[j] + operators[j] @ operators[i]
                    - (2 * I4 if i == j else 0))
               for i in range(3) for j in range(3))


def spatial_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0., -axis[2], axis[1]],
                      [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    return (np.cos(angle) * np.eye(3) + (1 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * cross)


@dataclass(frozen=True)
class RotationBand:
    a: float
    m: float = 1.0
    e: float = 1.0
    r: float = 1.0
    v: float = 1.0
    improved: bool = False

    @property
    def charge(self):
        return -self.e

    @property
    def alpha(self):
        return HOPPING

    @property
    def beta(self):
        return MIXING

    @property
    def generators(self):
        return ROTATION

    def couplings(self):
        """h(p)=onsite+sum(T_d exp(i p.d)+T_d^dagger exp(-i p.d))."""
        coefficient = 4.5 if self.improved else 3.
        onsite = (self.m + coefficient * self.r / self.a) * MIXING
        nearest_mass = self.r / self.a if self.improved else self.r / (2 * self.a)
        terms = []
        for i in range(3):
            displacement = np.eye(3, dtype=int)[i]
            terms.append((displacement, -1j * self.v * HOPPING[i] / (2 * self.a)
                          - nearest_mass * MIXING))
            if self.improved:
                terms.append((2 * displacement, self.r * MIXING / (4 * self.a)))
        return onsite, terms

    def h_from_couplings(self, p):
        onsite, terms = self.couplings()
        result = onsite.copy()
        for displacement, coupling in terms:
            phase = np.exp(1j * self.a * np.dot(p, displacement))
            result += coupling * phase + coupling.conj().T * phase.conjugate()
        return result

    def mass(self, p):
        delta = 2 * np.sin(self.a * np.asarray(p) / 2)**2
        return self.m + self.r / self.a * np.sum(delta**2 if self.improved else delta)

    def h(self, p):
        return (np.einsum('i,ijk->jk', self.v * np.sin(self.a * np.asarray(p)) / self.a,
                          HOPPING) + self.mass(p) * MIXING)

    def energy(self, p):
        return float(np.sqrt(np.sum((self.v * np.sin(self.a * np.asarray(p)) / self.a)**2)
                             + self.mass(p)**2))

    def continuum_h(self, p):
        return self.v * np.einsum('i,ijk->jk', p, HOPPING) + self.m * MIXING

    def projector(self, p, sign=1):
        if sign not in (-1, 1):
            raise ValueError('sector sign must be +1 or -1')
        return (I4 + sign * self.h(p) / self.energy(p)) / 2

    def rotation(self, axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        return expm(-1j * angle * np.einsum('i,ijk->jk', axis, ROTATION))

    def spinor(self, p, label):
        """Positive-band basis; labels refer to the derived rest-frame J_z."""
        energy, mass = self.energy(p), self.mass(p)
        chi = I2[:, label]
        lower = np.einsum('i,ijk->jk', self.v * np.sin(self.a * np.asarray(p)) / self.a,
                          TRANSITIONS) @ chi
        wilson_vector = np.concatenate((np.sqrt((energy + mass) / (2 * energy)) * chi,
                                        lower / np.sqrt(2 * energy * (energy + mass))))
        return TO_WILSON.conj().T @ wilson_vector

    def current(self, pout, pin):
        """Finite-transfer midpoint-link current divided by charge."""
        midpoint = self.a * (np.asarray(pout) + np.asarray(pin)) / 2
        mass_vertex = self.r * np.sin(midpoint)
        if self.improved:
            transfer = self.a * (np.asarray(pout) - np.asarray(pin)) / 2
            mass_vertex = self.r * (2 * np.sin(midpoint)
                                    - np.cos(transfer) * np.sin(2 * midpoint))
        return (self.v * np.cos(midpoint)[:, None, None] * HOPPING
                + mass_vertex[:, None, None] * MIXING)

    def vertex(self, pout, pin, eps):
        return self.charge * (eps[0] * I4
                              - np.einsum('i,ijk->jk', eps[1:], self.current(pout, pin)))

    def contact(self, pout, pin, eps_in, eps_out, k, kout):
        midpoint = self.a * (np.asarray(pout) + np.asarray(pin)) / 2
        curvature = self.r * np.cos(midpoint)
        if self.improved:
            curvature = self.r * (2 * np.cos(midpoint)
                                  - 2 * np.cos(self.a * np.asarray(k) / 2)
                                  * np.cos(self.a * np.asarray(kout) / 2)
                                  * np.cos(2 * midpoint))
        matrices = self.a * (-self.v * np.sin(midpoint)[:, None, None] * HOPPING
                            + curvature[:, None, None] * MIXING)
        return self.e**2 * np.einsum('i,i,ijk->jk', eps_in[1:], eps_out[1:], matrices)


def local_bosonic_operators():
    """15-state local qudit: vacuum, four singles, ten bosonic doubles."""
    basis = sorted((occupation for occupation in itertools.product(range(3), repeat=4)
                    if sum(occupation) <= 2), key=lambda state: (sum(state), state))
    lookup = {state: index for index, state in enumerate(basis)}
    lowering = []
    for orbital in range(4):
        operator = np.zeros((len(basis), len(basis)), dtype=complex)
        for column, state in enumerate(basis):
            if state[orbital]:
                target = list(state)
                target[orbital] -= 1
                operator[lookup[tuple(target)], column] = np.sqrt(state[orbital])
        lowering.append(operator)
    return basis, np.array(lowering), np.diag([sum(state) for state in basis])


def bosonic_dimer_probe():
    """Actual finite-penalty two-site sector; double occupations are not hidden."""
    band = RotationBand(0.5)
    onsite, terms = band.couplings()
    hopping = terms[0][1]
    omega = 20.
    single = np.block([[onsite + omega * I4, hopping],
                       [hopping.conj().T, onsite + omega * I4]])
    # Two total bosons on eight modes; each site's local cutoff is still two.
    basis = []
    for first in range(8):
        for second in range(first, 8):
            occupation = [0] * 8
            occupation[first] += 1
            occupation[second] += 1
            basis.append(tuple(occupation))
    lookup = {state: index for index, state in enumerate(basis)}
    quadratic = np.zeros((len(basis), len(basis)), dtype=complex)
    for column, state in enumerate(basis):
        for source in range(8):
            if not state[source]:
                continue
            intermediate = list(state)
            intermediate[source] -= 1
            for target in range(8):
                output = intermediate.copy()
                output[target] += 1
                amplitude = np.sqrt(state[source] * (intermediate[target] + 1))
                quadratic[lookup[tuple(output)], column] += single[target, source] * amplitude
    doubles = np.array([sum(state[:4]) * (sum(state[:4]) - 1) / 2
                        + sum(state[4:]) * (sum(state[4:]) - 1) / 2 for state in basis])
    rows = []
    for penalty in [2., 8., 32.]:
        h = quadratic + penalty * np.diag(doubles)
        energies, states = np.linalg.eigh(h)
        state = states[:, 0]
        rows.append(dict(Delta=penalty, lowest_two_boson_energy=float(energies[0]),
                         double_occupation_probability=float(np.vdot(state, doubles * state).real),
                         actual_eigenstate_residual=norm(h @ state - energies[0] * state)))
    local_basis, lowering, number = local_bosonic_operators()
    return dict(local_dimension=len(local_basis), two_particle_dimension=len(basis),
                local_number_action_residual=max(norm(number @ b - b @ number + b)
                                                 for b in lowering),
                single_sector_energies=np.linalg.eigvalsh(single).tolist(),
                two_particle_hermiticity_residual=norm(quadratic - quadratic.conj().T),
                finite_penalty_extra_sector=rows,
                interpretation='Bare bosonic model only: finite Delta leaves measurable double-occupation admixture; not CAR.')


def magnetic_probe(m=1., v=1., q=-1.):
    """Weak-field spectrum of the derived continuum generator, not lattice g."""
    cutoff = 6
    lowering = np.diag(np.sqrt(np.arange(1, cutoff)), 1).astype(complex)
    rows = []
    for field in [0.02, 0.01, 0.005, 0.0025]:
        scale = np.sqrt(abs(q * field) / 2)
        pix = scale * (lowering + lowering.conj().T)
        piy = -1j * np.sign(q * field) * scale * (lowering - lowering.conj().T)
        h = (np.kron(m * MIXING, np.eye(cutoff))
             + v * np.kron(HOPPING[0], pix) + v * np.kron(HOPPING[1], piy))
        rest_positive = np.array([1., 1.]) / np.sqrt(2)
        energies, variances, spectral_distances = [], [], []
        spectrum = np.linalg.eigvalsh(h)
        for label in [0, 1]:
            state = np.kron(np.kron(rest_positive, I2[:, label]), np.eye(cutoff)[:, 0])
            squared_state = h @ (h @ state)
            squared_energy = float(np.vdot(state, squared_state).real)
            energy = np.sqrt(squared_energy)
            energies.append(float(energy))
            variances.append(norm(squared_state - squared_energy * state))
            spectral_distances.append(float(np.min(abs(spectrum - energy))))
        split = abs(energies[0] - energies[1])
        rows.append(dict(B=field, positive_energies=energies,
                         g_from_weak_field_split=float(2 * m * split / (v*v * abs(q*field))),
                         squared_eigenstate_residual=max(variances),
                         actual_spectrum_distance=max(spectral_distances)))
    return rows


def run():
    # Existing consumer is used ONLY as a comparison target, never a supplier.
    from lattice_compton import ALPHA, BETA, WilsonLink

    p = np.array([0.23, 0.31, 0.17])
    axis, angle = np.array([1., 2., -1.]), 0.71
    rotation = spatial_rotation(axis, angle)
    rows = []
    for improved in [False, True]:
        for a in [0.2, 0.1, 0.05, 0.025]:
            band = RotationBand(a, improved=improved)
            old = WilsonLink(a, improved=improved)
            d = band.rotation(axis, angle)
            plus, minus = band.projector(p, 1), band.projector(p, -1)
            pin, pout = -0.4 * p, 0.6 * p
            k, kout = np.array([0.11, -0.08, 0.19]), np.array([0.04, 0.09, 0.05])
            eps, epsout = np.array([0., 0.3, -0.2, 0.4]), np.array([0., -0.1, 0.5, 0.2])
            khat = 2 * np.sin(a * (pout - pin) / 2) / a
            ward = np.einsum('i,ijk->jk', khat, band.current(pout, pin))
            corners = [band.energy(np.pi / a * np.array(corner))
                       for corner in itertools.product([0, 1], repeat=3)]
            axes_p = np.array([np.linalg.norm(p), 0., 0.])
            continuum_energy = np.sqrt(band.m**2 + band.v**2 * np.dot(p, p))
            rows.append(dict(a=a, improved=improved,
                microscopic_fourier_residual=norm(band.h_from_couplings(p) - band.h(p)),
                spectrum=np.linalg.eigvalsh(band.h(p)).tolist(),
                relativistic_energy_error=float(band.energy(p) - continuum_energy),
                continuum_rotation_residual=norm(d @ band.continuum_h(p) @ d.conj().T
                                                - band.continuum_h(rotation @ p)),
                finite_a_rotation_residual=norm(d @ band.h(p) @ d.conj().T
                                              - band.h(rotation @ p)),
                oblique_minus_axis_energy=float(band.energy(p) - band.energy(axes_p)),
                projector_residual=max(norm(plus @ plus - plus), norm(minus @ minus - minus),
                                       norm(plus @ minus), norm(plus + minus - I4)),
                rest_spin_eigenvalues=np.linalg.eigvalsh(ROTATION[2]).tolist(),
                charge_conjugation_residual=norm(CHARGE_CONJUGATION @ band.h(p).conj()
                                                @ CHARGE_CONJUGATION.conj().T + band.h(-p)),
                spinor_residual=max(norm(band.h(p) @ band.spinor(p, s)
                                        - band.energy(p) * band.spinor(p, s)) for s in [0, 1]),
                current_ward_residual=norm(ward - band.h(pout) + band.h(pin)),
                wilson_h_residual=norm(TO_WILSON @ band.h(p) @ TO_WILSON.conj().T - old.h(p)),
                wilson_vertex_residual=norm(TO_WILSON @ band.vertex(pout, pin, eps)
                                           @ TO_WILSON.conj().T - old.vertex(pout, pin, eps)),
                wilson_contact_residual=norm(TO_WILSON @ band.contact(pout, pin, eps, epsout, k, kout)
                                            @ TO_WILSON.conj().T
                                            - old.contact(pout, pin, eps, epsout, k, kout)),
                corner_gaps=corners,
                naive_r_zero_corner_gaps=[RotationBand(a, r=0).energy(np.pi / a * np.array(c))
                                         for c in itertools.product([0, 1], repeat=3)]))
    band = RotationBand(0.1)
    # An axis-only fit cannot reject this commuting fake: oblique probes must.
    commuting = np.array([HOPPING[2]] * 3)
    anisotropic = HOPPING * np.array([1., 0.8, 1.2])[:, None, None]
    missing_hopping = HOPPING.copy()
    missing_hopping[1] = 0
    false_positives = []
    target_energy = np.sqrt(1 + np.dot(p, p))
    for name, operators in [('commuting', commuting), ('anisotropic', anisotropic),
                            ('missing_y_hopping', missing_hopping)]:
        fake = MIXING + np.einsum('i,ijk->jk', p, operators)
        spectrum = np.linalg.eigvalsh(fake)
        fake_rotation = band.rotation(axis, angle)
        false_positives.append(dict(name=name, isotropy_defect=isotropy_defect(operators),
            positive_energy_error=float(spectrum[-1] - target_energy),
            covariance_error=norm(fake_rotation @ fake @ fake_rotation.conj().T
                - MIXING - np.einsum('i,ijk->jk', rotation @ p, operators))))
    tiny = 0.001
    slopes = []
    for improved in [False, True]:
        for a in [0.1, 0.05]:
            b = RotationBand(a, improved=improved)
            coefficient = (b.energy(np.array([tiny, 0., 0.]))**2 - b.m**2) / tiny**2
            expected = b.v**2 + (0 if improved else b.m * b.r * a)
            slopes.append(dict(a=a, improved=improved, measured_p_squared_coefficient=coefficient,
                               analytic_p_squared_coefficient=expected))
    return dict(
        inputs=dict(quantum_hilbert_law='declared', mass='declared scalar mixing m',
                    charge='declared q=-e shared with link supplier',
                    pattern='selected by spectral isotropy, minimal two-level blocks and a gap',
                    many_body='requires bosonic-statistics encoded sector; bare bosons do not fill a Dirac sea'),
        algebra=dict(isotropy_defect=isotropy_defect(HOPPING),
            mass_anticommutation=max(norm(MIXING @ h + h @ MIXING) for h in HOPPING),
            spin_casimir_residual=norm(sum(j @ j for j in ROTATION) - 0.75 * I4),
            two_pi_minus_identity_residual=norm(band.rotation(axis, 2*np.pi) + I4),
            four_pi_identity_residual=norm(band.rotation(axis, 4*np.pi) - I4),
            alpha_unitary_map_residual=max(norm(TO_WILSON @ HOPPING[i] @ TO_WILSON.conj().T
                                                - ALPHA[i]) for i in range(3)),
            beta_unitary_map_residual=norm(TO_WILSON @ MIXING @ TO_WILSON.conj().T - BETA)),
        bands=rows, mass_and_velocity_coefficients=slopes,
        weak_field_continuum_magnetic_response=magnetic_probe(), rejected_false_positives=false_positives,
        bare_bosonic_extra_sectors=bosonic_dimer_probe(),
        limitations=['No mechanism selects isotropy from generic local couplings.',
                     'Continuous rotations are effective at low momentum, not exact on the cubic lattice.',
                     'The central 2pi phase is a derived internal band response, not by itself a boson-to-fermion proof.',
                     'Negative bands are holes only after the separate fermionic filling construction.',
                     'No measured electron mass, charge, or common photon velocity is generated.'])


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, sort_keys=True))

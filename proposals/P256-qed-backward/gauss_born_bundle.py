#!/usr/bin/env python3
"""Primitive paired-sea Gauss bundle and a negative full-field trial theorem.

The state is a covariant variational section, not a solved quantum ground.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import expm

import full_field_charge_bound as fields
from quantum_su4_gauss import norm, require_small, uniform_data
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis


class PairedGaussState:
    """Actual N_H=N_L=2 Pauli sector; no Slater-minor sign convention."""

    def __init__(self):
        self.primitive, _, units = fields.primitive_singlet_and_ph()
        keep = np.array([mask for mask in range(256)
                         if (mask & 15).bit_count() == 2 and (mask >> 4).bit_count() == 2])
        self.units = {(i, j): matrix[keep][:, keep].toarray()
                      for (i, j), matrix in units.items()
                      if i // 4 == j // 4}
        self.basis = matrix_unit_basis()
        self.T = self.basis / np.sqrt(8)
        self.QH = np.array([self.lift(t, 0) for t in self.T])
        self.QL = np.array([self.lift(t, 1) for t in self.T])
        self.Q = self.QH + self.QL
        casimir = sum(q @ q for q in self.Q)
        values, vectors = np.linalg.eigh(casimir)
        multiplicities = {str(c): int(np.count_nonzero(abs(values - c) < 1e-9))
                          for c in (0, 4, 6)}
        if multiplicities != {'0': 1, '4': 15, '6': 20}:
            raise AssertionError('Primitive paired representation does not contain the unique epsilon singlet')
        self.singlet = vectors[:, 0]
        self.multiplicities = multiplicities
        require_small(norm(self.Q @ self.singlet), 'actual epsilon singlet')

    def lift(self, matrix, species):
        offset = 4 * species
        return sum(matrix[i, j] * self.units[offset + i, offset + j]
                   for i in range(4) for j in range(4))

    @staticmethod
    def projector(h):
        values, vectors = np.linalg.eigh(h)
        gap = float(values[2] - values[1])
        if gap <= 1e-10 * max(1., norm(h)):
            raise ValueError('Rank-two occupied projector requires a separated middle gap')
        occupied = vectors[:, :2]
        return occupied @ occupied.conj().T, values, vectors, gap

    @staticmethod
    def projector_derivative(values, vectors, change):
        occupied, empty = vectors[:, :2], vectors[:, 2:]
        mixing = (empty.conj().T @ change @ occupied) / (values[:2][None, :] - values[2:, None])
        block = empty @ mixing @ occupied.conj().T
        return block + block.conj().T

    def state(self, projector):
        number = self.lift(projector, 1)
        return np.sqrt(6) / 2 * (number @ (number @ self.singlet - self.singlet))

    def derivative(self, projector, change):
        number, derivative = self.lift(projector, 1), self.lift(change, 1)
        return np.sqrt(6) / 2 * (derivative @ (number @ self.singlet - self.singlet)
                                + number @ (derivative @ self.singlet))

    def controls(self, h, steps=(2e-4, 1e-4)):
        projector, values, vectors, gap = self.projector(h)
        state = self.state(projector)
        normalization = abs(float(np.vdot(state, state).real) - 1)
        colour_mean = norm(np.einsum('i,aij,j->a', state.conj(), self.Q, state))
        derivatives = [self.derivative(projector, 1j * (t @ projector - projector @ t)) for t in self.T]
        gauss = max(norm(d - 1j * q @ state) for d, q in zip(derivatives, self.Q))
        missing_heavy = math.sqrt(sum(norm(d - 1j * q @ state)**2 for d, q in zip(derivatives, self.QL)))
        frozen_matter = math.sqrt(sum(norm(q @ state)**2 for q in self.Q))
        field_changes = [self.projector_derivative(values, vectors, b) for b in self.basis]
        state_changes = np.array([self.derivative(projector, p) for p in field_changes], dtype=np.complex128)
        inner_products = state_changes.conj() @ state_changes.T
        actual_metric = inner_products.real
        projector_metric = np.array([[np.trace(a @ b).real for b in field_changes] for a in field_changes])
        metric_error = norm(actual_metric - projector_metric)
        berry = max(abs(np.vdot(state, d)) for d in state_changes)
        paired_curvature = norm(2 * inner_products.imag)
        light_curvature = np.array([[float((1j * np.trace(projector @ (a @ b - b @ a))).real)
                                     for b in field_changes] for a in field_changes])
        heavy_curvature = np.array([[float((1j * np.trace((np.eye(4) - projector) @ (a @ b - b @ a))).real)
                                     for b in field_changes] for a in field_changes])
        fd_errors = []
        for step in steps:
            errors = []
            for b, derivative in zip(self.basis, state_changes):
                plus = self.state(self.projector(h + step * b)[0])
                minus = self.state(self.projector(h - step * b)[0])
                errors.append(norm((plus - minus) / (2 * step) - derivative))
            fd_errors.append(max(errors))
        coordinates = .21 * np.sin(np.arange(15) * .71 + .19)
        g = expm(1j * np.einsum('a,aij->ij', coordinates, self.T))
        transformed = self.state(g @ projector @ g.conj().T)
        primitive_rotation = expm(1j * np.einsum('a,aij->ij', coordinates, self.Q))
        finite_covariance = norm(transformed - primitive_rotation @ state)
        moment_formula = 8 * sum(1 / (values[e] - values[o])**2 for o in (0, 1) for e in (2, 3))
        metric_trace_error = abs(float(np.trace(actual_metric)) - moment_formula)
        light_energy = self.lift(h, 1) @ state
        heavy_energy = self.lift(h, 0) @ state
        energy_error = max(norm(light_energy - sum(values[:2]) * state),
                           norm(heavy_energy - sum(values[2:]) * state))
        for value, name in ((normalization, 'paired normalization'), (colour_mean, 'paired mean'),
                            (gauss, 'covariant differential Gauss'), (finite_covariance, 'finite primitive covariance'),
                            (metric_error, 'projector quantum metric'), (metric_trace_error, 'inverse-gap metric'),
                            (berry, 'globally flat paired Berry connection'), (paired_curvature, 'paired curvature'),
                            (norm(light_curvature + heavy_curvature), 'opposite determinant curvatures'),
                            (energy_error, 'actual onsite occupied energies')):
            require_small(value, name, 2e-9)
        if frozen_matter < 1 or missing_heavy < 1 or norm(light_curvature) < 1e-3:
            raise AssertionError('Gauss/Berry adverse controls are not resolved')
        if fd_errors[1] > .4 * fd_errors[0] + 5e-10 or fd_errors[1] > 5e-7:
            raise AssertionError('Independent finite field derivative is not converged')
        return dict(occupied_gap=gap, paired_norm_residual=normalization,
            mean_colour_norm=colour_mean, matter_Casimir=frozen_matter**2,
            actual_Gauss_covariant_derivative_residual=gauss, finite_primitive_covariance_residual=finite_covariance,
            frozen_matter_Gauss_defect=frozen_matter, missing_heavy_Gauss_defect=missing_heavy,
            paired_Berry_connection_residual=float(berry), paired_curvature_residual=paired_curvature,
            light_determinant_curvature_norm=norm(light_curvature), opposite_curvature_residual=norm(light_curvature + heavy_curvature),
            quantum_metric_residual=metric_error, canonical_metric_trace=float(np.trace(actual_metric)),
            inverse_gap_metric_trace=moment_formula, finite_derivative_steps=list(steps), finite_derivative_errors=fd_errors,
            onsite_occupied_eigenstate_residual=energy_error)


def negative_trial(graph, band, *, sigma, epsilon, kappa, J, K4, KEM, scales):
    """Rigorous conservative Gaussian/orbit trial constants, not MC estimates."""
    if min(sigma, epsilon, kappa) <= 0 or min(J, K4, KEM) < 0:
        raise ValueError('Positive widths/kappa and nonnegative gauge couplings required')
    onsite, edges = fields.primitive_edges(graph, band)
    if onsite <= 0 or min(scales) < 1:
        raise ValueError('Positive common onsite scalar and Lambda>=1 required')
    # For traceless Hermitian4: -sum(two lowest E)>=||h||HS/sqrt3.
    chi15 = math.sqrt(2) * math.gamma(8) / math.gamma(7.5)
    occupied_coefficient = 2 * chi15 / math.sqrt(3)
    quartic = kappa / 2 * (255 * sigma**4 + 90 * sigma**2 * epsilon**2 + 855 * epsilon**4)
    alignment = 15 * J * (sigma**2 + 3 * epsilon**2) * 3
    colour = 2 * K4 * 3
    coefficient = quartic + alignment + colour - onsite * occupied_coefficient * sigma
    # GUE4 inverse-gap sum E sum_(i<j) 1/(xi-xj)^2=3; cross pairs are a subset.
    kinetic = (15 / 8 + 3) / sigma**2 + 45 / (8 * epsilon**2)
    _, paths = fields.scalar_paths(band)
    cH = 8 * abs(onsite) + 16 * sum(float(np.linalg.norm(f)) for _, f in paths)
    rows = []
    for scale in scales:
        upper = scale * coefficient + math.sqrt(scale) * kinetic - onsite * occupied_coefficient * sigma
        amplitude = max(0., (-upper / scale - KEM * 3 / scale) / (cH * (1 + 1 / scale)))
        rows.append(dict(Lambda=scale, neutral_full_field_energy_upper_per_site=upper,
                         invariant_mean_amplitude_lower=amplitude, negative_trial=upper < 0))
    return dict(domain='all60 real canonical fields/site, actual primitive matter, genuine Haar/compact EM; finite bounded-degree cubic graphs',
        common_onsite_scalar=onsite, field_density_widths=dict(Phi0=sigma, spatial=epsilon),
        chi15_mean=chi15, occupied_sum_lower_coefficient=occupied_coefficient,
        quartic_Lambda_coefficient=quartic, alignment_Lambda_coefficient=alignment,
        positive_colour_Lambda_coefficient=colour, net_Lambda_coefficient=coefficient,
        kinetic_sqrt_Lambda_upper=kinetic, whole_species_coercivity_coefficient=cH,
        strict_sufficient_Lambda_for_negative_leading_balance=(kinetic / -coefficient)**2 if coefficient < 0 else None,
        asymptotic_invariant_amplitude_lower=max(0., -coefficient / cH), rows=rows,
        hypotheses=['whole heavy Hamiltonian=-Lambda*h_light; onsite scalar positive',
                    'local N_H=N_L=2 in the variational state, hopping means zero',
                    'bare quartic minimum zero; alignment/colour energies nonnegative',
                    'positive canonical/electric kinetic coefficients; physical Gauss form domain',
                    'graph bond/plaquette density <=3; negative EM magnetic energy >=-3*KEM*V'],
        limits=['variational energy upper and gauge-invariant amplitude lower, not a solved ground wavefunction',
                'does not identify a Clifford quartet, prove SSB/phase, charged lower gap, Lorentz symmetry or physical half-spin'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lambdas', nargs='+', type=float, default=[1e6, 1e7, 1e8])
    parser.add_argument('--sigma', type=float, default=.2)
    parser.add_argument('--epsilon', type=float, default=.1)
    parser.add_argument('--kappa', type=float, default=1.)
    parser.add_argument('--J', type=float, default=.05)
    parser.add_argument('--K4', type=float, default=.01)
    parser.add_argument('--KEM', type=float, default=.02)
    args = parser.parse_args()
    if not all(np.isfinite(v) for v in args.lambdas + [args.sigma, args.epsilon, args.kappa, args.J, args.K4, args.KEM]):
        parser.error('finite parameters required')
    pair = PairedGaussState()
    band = RotationBand(a=1., m=.4, r=.2, v=.3, improved=False)
    data = uniform_data(band, 3, 2., 0., matrix_unit_basis())
    h = data['frame'][0]
    change = np.einsum('a,aij->ij', np.sin(np.arange(15) * .47 + .23), pair.basis) / 31
    result = dict(status='primitive_covariant_paired_Gauss_trial_and_unrestricted_quantum_amplitude_selection_NOT_full_phase',
        primitive=pair.primitive, pair_Casimir_multiplicities=pair.multiplicities,
        cases=[dict(label=label, **pair.controls(matrix)) for label, matrix in (('selected', h), ('deformed', h + change))],
        trial=negative_trial(fields.open_box((2, 4, 4)), band, sigma=args.sigma, epsilon=args.epsilon,
                             kappa=args.kappa, J=args.J, K4=args.K4, KEM=args.KEM, scales=args.lambdas))
    names = ('gauss_born_bundle.py', 'gauss_born_bundle.md', 'full_field_charge_bound.py',
             'gauss_charge_intertwiner.py', 'quantum_su4_gauss.py', 'quantum_spinor_plaquette.py')
    result['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    payload = json.dumps(result, indent=2, allow_nan=False)
    args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()

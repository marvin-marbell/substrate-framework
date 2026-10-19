#!/usr/bin/env python3
"""Whole-heavy frozen-frame potential: global four-centre certificate, not a phase."""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import scipy
from scipy.linalg import expm

from full_band_gauss_section import FullBandGaussSection, source_hashes as section_hashes
from haar_centre_rotor import CENTRE_DISTANCE_CONSTANT, alignment_value, commutator_controls
from local_orbital_stability import uniform_data
from quantum_su4_gauss import norm, require_small
from rotation_band import RotationBand


def source_hashes():
    root = Path(__file__).resolve().parent
    names = ('heavy_sea_centre_potential.py', 'heavy_sea_centre_potential.md',
             'haar_centre_rotor.py', 'haar_centre_rotor.md')
    return {**section_hashes(), **{
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}


def fraction(value):
    return Fraction(str(value))


def sqrt_upper(value: Fraction) -> Fraction:
    """A rational upper bound, with the final inequality checked in exact arithmetic."""
    with localcontext() as context:
        context.prec = 70
        root = (Decimal(value.numerator) / Decimal(value.denominator)).sqrt().next_plus()
        bound = Fraction(str(root))
    # Decimal division is rounded before sqrt; enlarge if that rounding crossed the root.
    if bound * bound < value:
        bound += Fraction(1, 10 ** 65)
    if bound * bound < value:
        raise ArithmeticError('Rational square-root upper bound did not certify')
    return bound


def amplitude_interval(value):
    centre = Fraction(f'{value:.10f}')
    radius = Fraction(1, 10 ** 10)
    lo, hi = centre - radius, centre + radius
    if not lo <= fraction(value) <= hi or lo <= 0:
        raise ArithmeticError('Positive selected amplitude is outside its stated interval')
    return lo, hi


def global_certificate(frame, band, prefactor) -> dict[str, Any]:
    """Exact rational Gershgorin proof covering an amplitude box, not sampled SU4."""
    amplitudes = np.sqrt(np.einsum('mij,mji->m', frame, frame).real / 4)
    require_small(norm(amplitudes[1:] - amplitudes[1]), 'equal spatial amplitudes')
    b = amplitude_interval(float(amplitudes[0]))
    s = amplitude_interval(float(amplitudes[1]))
    m, r, v, j = map(fraction, (band.m, band.r, band.v, prefactor))
    if min(m, r, v, j) <= 0:
        raise ValueError('This certificate requires positive m,r,v,Jpref')
    c = m + r
    a_lo, a_hi = c * b[0], c * b[1]
    d_hi = r * b[1] / 2
    e_lo, e_hi = v * s[0] / 2, v * s[1] / 2
    gamma_lo = (m + r / 2) * b[0]
    actual_gap_lo = 2 * (a_lo - d_hi - e_hi)
    if actual_gap_lo <= 0:
        raise ValueError('Amplitude box does not certify the actual whole-heavy gap')
    R_hi = sqrt_upper(a_hi * a_hi + e_hi * e_hi)
    loss_lo = e_lo * e_lo / (2 * R_hi)
    off_hi = d_hi * e_hi / (2 * gamma_lo)
    rows = []
    margins = []
    for mask in range(1, 16):
        grade = mask.bit_count()
        anti = [bool(mask & (1 << mu)) == (grade % 2 == 0) for mu in range(4)]
        commuting = [not anti[0], not anti[1]]
        ell_lo = 4 * (b[0] ** 2 * anti[0] + s[0] ** 2 * sum(anti[1:]))
        diagonal_lo = (j * ell_lo / 8 + loss_lo
                       - (d_hi ** 2 * commuting[0] + e_hi ** 2 * commuting[1]) / (2 * gamma_lo))
        row_radius_hi = off_hi if commuting[0] != commuting[1] else Fraction(0)
        margin = diagonal_lo - row_radius_hi
        margins.append(margin)
        rows.append(dict(mask=mask, grade=grade, anticommutes=anti,
                         paired_mask=(mask ^ 3) if row_radius_hi else None,
                         diagonal_lower=float(diagonal_lo), off_diagonal_row_sum_upper=float(row_radius_hi),
                         margin_lower=float(margin), exact_margin_rational=str(margin)))
    minimum = min(margins)
    if minimum <= 0:
        raise ValueError('Global heavy-sea quadratic certificate failed; no centre-well claim')
    ell_max_hi = 4 * (b[1] ** 2 + 3 * s[1] ** 2)
    return dict(amplitude_intervals={name: [str(x) for x in values]
                                    for name, values in (('beta', b), ('spatial', s))},
                rational_parameters=dict(m=str(m), r=str(r), v=str(v), Jpref=str(j)),
                reference_absolute_gap_lower=float(gamma_lo), radial_loss_coefficient_lower=float(loss_lo),
                actual_occupied_empty_gap_lower=float(actual_gap_lo),
                hopping_operator_norm_upper=float(d_hi + e_hi),
                off_diagonal_magnitude_upper=float(off_hi), complex_traceless_dimension=15,
                real_dimension=30, repeated_real_imaginary_rows=rows,
                Gershgorin_margin_lower=float(minimum), exact_margin_rational=str(minimum),
                global_Q_coercivity_lower=float(minimum),
                global_W_coercivity_lower=float(minimum / ell_max_hi),
                global_centre_chord_coercivity_lower=float(minimum) / CENTRE_DISTANCE_CONSTANT,
                proof='Exact rational interval inequalities in the Hermitian Clifford-monomial basis; each row has at most one off-diagonal entry. No finite optimization or sampling premise.')


class HeavySeaCentrePotential:
    """Same bare two-cell Hamiltonian, with all heavy onsite and hopping at order Lambda."""

    def __init__(self, m=.2, r=.12, v=.3, stiffness=2., prefactor=.2):
        self.band = RotationBand(a=1., m=m, r=r, v=v, improved=False)
        self.model = FullBandGaussSection(self.band, heavy_scale=1.)
        self.data = uniform_data(self.band, 3, stiffness, 0., self.model.basis)
        self.frame = self.data['frame'].astype(complex)
        self.fields = np.tile(self.frame, (2, 1, 1, 1))
        self.prefactor = float(prefactor)
        self.generators = self.model.generators
        self.c = float(self.model.onsite[0])
        require_small(abs(self.c - (m + r)), 'same-bare onsite m+r')
        self.F = np.einsum('m,mij->ij', self.model.paths[0], self.frame)
        amplitudes = np.sqrt(np.einsum('mij,mji->m', self.frame, self.frame).real / 4)
        self.a, self.d, self.e = self.c * amplitudes[0], r * amplitudes[0] / 2, v * amplitudes[1] / 2
        self.R = float(np.hypot(self.a, self.e))
        require_small(norm(self.F + r * self.frame[0] / 2 + 1j * v * self.frame[1] / 2),
                      'actual extracted Wilson/spatial endpoint coefficients')
        self.hopping_norm = float(np.linalg.norm(self.F, 2))
        self.commutators = commutator_controls(self.frame, self.generators)
        self.certificate = global_certificate(self.frame, self.band, self.prefactor)
        self.gap_lower = self.certificate['actual_occupied_empty_gap_lower']
        self.hopping_upper = self.certificate['hopping_operator_norm_upper']
        if self.hopping_norm > self.hopping_upper:
            raise ArithmeticError('Actual hopping norm exceeds certified amplitude box')
        self.centre_energy = -4 * self.R
        require_small(abs(self.value(np.eye(4))['V'] - self.centre_energy), 'exact centre whole-heavy sum')

    def value(self, U) -> dict[str, Any]:
        h = self.model.onebody(self.fields, U)[0]
        p, spectrum = self.model.spectral(h, 4, top=True)
        sea = float(spectrum[0][4:].sum())
        w = alignment_value(self.frame, U)
        q = max(0., float(4 - abs(np.trace(U)) ** 2 / 4))
        return dict(V=self.prefactor * w / 8 - sea, sea_positive_sum=sea,
                    W=w, Q=q, gap=float(spectrum[-1]), h=h, projector=p, spectrum=spectrum)

    def differential(self, U) -> dict[str, Any]:
        result = self.value(U)
        h, p, spectrum = result['h'], result['projector'], result['spectrum']
        dh, dp, dU = [], [], []
        for t in self.generators:
            tangent = 1j * t @ U
            change = self.model.onebody_derivative(self.fields, U, (0., 0.),
                np.zeros_like(self.fields), tangent, np.zeros(2))[0]
            dU.append(tangent)
            dh.append(change)
            dp.append(self.model.spectral_derivative(spectrum, change))
        dh, dp, dU = np.array(dh), np.array(dp), np.array(dU)
        gradient_sea = np.einsum('ij,aji->a', p, dh).real
        gradient_w = np.zeros(15)
        Hessian_w, Hessian_sea = np.zeros((15, 15)), np.zeros((15, 15))
        for a, ta in enumerate(self.generators):
            for f in self.frame:
                comm = f @ U - U @ f
                first = f @ dU[a] - dU[a] @ f
                gradient_w[a] += 2 * np.vdot(comm, first).real
            for b, tb in enumerate(self.generators):
                d2U = -(ta @ tb + tb @ ta) @ U / 2
                d2T = (self.F @ d2U + d2U @ self.F) / 2
                d2h = np.zeros_like(h)
                d2h[:4, 4:], d2h[4:, :4] = d2T, d2T.conj().T
                Hessian_sea[a, b] = np.trace(p @ d2h + dp[a] @ dh[b]).real
                for f in self.frame:
                    first_a = f @ dU[a] - dU[a] @ f
                    first_b = f @ dU[b] - dU[b] @ f
                    comm = f @ U - U @ f
                    second = f @ d2U - d2U @ f
                    Hessian_w[a, b] += 2 * (np.vdot(first_a, first_b) + np.vdot(comm, second)).real
        Hessian = self.prefactor * Hessian_w / 8 - Hessian_sea
        metric = np.array([[np.trace(p @ x @ y).real for y in dp] for x in dp])
        require_small(norm(Hessian - Hessian.T), 'actual coupled Hessian symmetry')
        return dict(gradient=(self.prefactor * gradient_w / 8 - gradient_sea).tolist(),
                    heavy_sea_gradient=gradient_sea.tolist(), Hessian=Hessian.tolist(),
                    Hessian_eigenvalues=np.linalg.eigvalsh(Hessian).tolist(),
                    heavy_sea_Hessian=Hessian_sea.tolist(), alignment_Hessian=Hessian_w.tolist(),
                    heavy_determinant_Born_metric=metric.tolist(),
                    heavy_determinant_Born_eigenvalues=np.linalg.eigvalsh(metric).tolist(),
                    projector_derivatives=dp, onebody_derivatives=dh)

    def primitive_checks(self, U, include_metric=True):
        section = self.model.state(self.fields, U)
        psi = section.state
        energies = []
        for species, multiplier in ((0, -1.), (1, 1.)):
            operator = multiplier * self.model.lift(section.onebody[species], species)
            occupied = multiplier * sum(section.spectra[species][0][section.spectra[species][2]])
            residual = norm(operator @ psi - occupied * psi)
            require_small(residual, 'actual 4900-state whole occupied energy', 3e-9)
            energies.append(dict(species='heavy' if species == 0 else 'light',
                                 expectation=float(np.vdot(psi, operator @ psi).real),
                                 occupied_energy=float(occupied), eigen_residual=residual))
        centres = []
        nleft = self.model.matter.heavy_left + self.model.matter.light_left
        for k in range(4):
            z = 1j ** k
            target = self.model.state(self.fields, z * U)
            phase = z ** nleft
            residue = norm(target.state - phase * psi)
            block = np.diag([z] * 4 + [1.] * 4)
            onebody_residue = max(norm(target.onebody[s] - block @ section.onebody[s] @ block.conj().T)
                                  for s in range(2))
            projector_residue = max(norm(target.projectors[s] - block @ section.projectors[s] @ block.conj().T)
                                    for s in range(2))
            require_small(max(residue, onebody_residue, projector_residue), 'actual centre covariance', 3e-8)
            centres.append(dict(k=k, centre_real=float(z.real), centre_imaginary=float(z.imag),
                                primitive_state_covariance_residual=residue,
                                onebody_covariance_residual=onebody_residue,
                                projector_covariance_residual=projector_residue,
                                projection_weight=float(target.weight)))
        output: dict[str, Any] = dict(dimension=self.model.matter.dimension, N_heavy=4, N_light=4,
                    projection_weight=section.weight, whole_occupied_energies=energies,
                    centre_covariance=centres,
                    centre_map='D_z=diag(zI4,I4) on each one-body species; M_z=z**(NH_left+NL_left) on the actual Pauli basis')
        if include_metric:
            differentials = self.differential(U)
            tangents = np.array([self.model.derivative(section, self.fields, U, (0., 0.), dU=1j * t @ U)
                                 for t in self.generators], dtype=complex)
            tangents -= np.outer(tangents @ psi.conj(), psi)
            primitive_metric = np.real(tangents.conj() @ tangents.T)
            expected = 2 * np.array(differentials['heavy_determinant_Born_metric'])
            error = norm(primitive_metric - expected)
            require_small(error, 'actual paired Pauli Born metric versus projector derivatives', 3e-8)
            output.update(paired_Pauli_Born_metric=primitive_metric.tolist(),
                          paired_Pauli_Born_metric_eigenvalues=np.linalg.eigvalsh(primitive_metric).tolist(),
                          paired_metric_vs_two_heavy_error=error)
        return output

    def derivative_controls(self, U, seed=222):
        rng = np.random.default_rng(seed)
        coordinates = rng.normal(size=15)
        coordinates /= norm(coordinates)
        X = np.einsum('a,aij->ij', coordinates, self.generators)
        analytic = self.differential(U)
        dp = np.einsum('a,aij->ij', coordinates, analytic['projector_derivatives'])
        gradient = float(coordinates @ np.array(analytic['gradient']))
        Hessian = float(coordinates @ np.array(analytic['Hessian']) @ coordinates)
        v0 = self.value(U)['V']
        probes = []
        for step in (2e-4, 1e-4):
            plus, minus = self.value(expm(1j * step * X) @ U), self.value(expm(-1j * step * X) @ U)
            probes.append(dict(step=step,
                projector_derivative_error=norm((plus['projector'] - minus['projector']) / (2 * step) - dp),
                energy_gradient_error=abs((plus['V'] - minus['V']) / (2 * step) - gradient),
                energy_Hessian_error=abs((plus['V'] + minus['V'] - 2 * v0) / step ** 2 - Hessian)))
        require_small(max(p['projector_derivative_error'] for p in probes), 'projector derivative finite difference', 2e-7)
        require_small(max(p['energy_gradient_error'] for p in probes), 'energy gradient finite difference', 2e-7)
        require_small(max(p['energy_Hessian_error'] for p in probes), 'energy Hessian finite difference', 2e-6)
        return probes

    def uniform_derivative_bounds(self):
        f, g = self.hopping_upper, self.gap_lower
        return dict(whole_heavy_hopping_operator_norm=self.hopping_norm,
                    whole_heavy_hopping_operator_norm_upper=f, whole_heavy_occupied_empty_gap_lower=g,
                    minimum_positive_onebody_energy_lower=g / 2,
                    normalization='Tr(Ta Tb)=delta_ab/2, ||X||HS=|x|/sqrt2, dU=iXU',
                    onebody_first_derivative_HS_per_unit_coordinate=f,
                    projector_first_derivative_HS_per_unit_coordinate=f / g,
                    heavy_determinant_Born_metric_operator_upper=f * f / (2 * g * g),
                    paired_determinant_Born_metric_operator_upper=f * f / (g * g),
                    sea_gradient_per_unit_coordinate_upper=2 * np.sqrt(2) * f,
                    sea_directional_Hessian_absolute_upper=2 * f + f * f / g,
                    domain='All SU4 links at this fixed selected frame. These are projector/ray bounds, not a globally positive epsilon-section weight or full-field kinetic certificate.')


def build(args):
    identities = source_hashes()
    potential = HeavySeaCentrePotential(args.m, args.r, args.v, args.stiffness, args.Jpref)
    rng = np.random.default_rng(args.seed)
    samples, links = [], []
    for _ in range(args.samples):
        U, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        U[:, 0] /= np.linalg.det(U)
        row = potential.value(U)
        delta = row['V'] - potential.centre_energy
        lower = potential.certificate['global_Q_coercivity_lower'] * row['Q']
        if delta < lower - 2e-10:
            raise AssertionError('Observed counterexample to proved frozen-frame coercivity')
        samples.append(dict(V=row['V'], V_minus_centre=delta, Q=row['Q'], W=row['W'],
                            certified_difference_lower=lower, heavy_gap=row['gap']))
        links.append(U)
    centre = potential.differential(np.eye(4, dtype=complex))
    centre.pop('projector_derivatives')
    centre.pop('onebody_derivatives')
    if min(centre['Hessian_eigenvalues']) <= 0:
        raise AssertionError('Observed negative centre curvature; do not discard')
    require_small(norm(centre['gradient']), 'centre actual leading gradient')
    representative = links[int(np.argmin([row['V'] for row in samples]))]
    primitive = [dict(label='identity', **potential.primitive_checks(np.eye(4, dtype=complex))),
                 dict(label='lowest_of_generic_samples', **potential.primitive_checks(representative, False))]
    return dict(model='Actual whole-heavy frozen-frame leading link potential',
                parameters={key: value for key, value in vars(args).items() if key != 'output'},
                leading_law='Lambda*[Jpref*W(U)/8 - sum_top4 h_H(U)]; same m+r onsite and endpoint-averaged full Wilson/spatial hopping',
                selected_gram_diagonal=np.diag(potential.data['selected']['G']).tolist(),
                selected_stationarity_residual=float(potential.data['selected']['gradient_residual']),
                exact_centre_positive_sum=4 * potential.R, exact_centre_V=potential.centre_energy,
                onsite_a=float(potential.a), Wilson_d=float(potential.d), spatial_e=float(potential.e),
                commutator_controls=potential.commutators, global_certificate=potential.certificate,
                uniform_gap_and_derivative_bounds=potential.uniform_derivative_bounds(),
                centre_all15_actual_projector_differentials=centre,
                finite_generic_SU4_observations=dict(seed=args.seed, count=args.samples, points=samples,
                    minimum_V=min(row['V'] for row in samples), interpretation='Discriminating observations only; the global proof is the rational matrix certificate'),
                actual_Pauli_4900_checks=primitive,
                derivative_finite_difference_controls=potential.derivative_controls(representative),
                proved_wells=['I', 'iI', '-I', '-iI'],
                stabilizer='The fixed four weighted Clifford fields have scalar commutant; its SU4 intersection is Z4. This is a mathematical frame stabilizer, not a deconfined gauge phase.',
                outstanding_join=['Actual matter-angular Gauss kinetic operator and its Born/Berry terms',
                    'Unrestricted Cartesian field Born kinetics and radial/frame fluctuations',
                    'Spectral four-band isolation/tunnelling for the coupled operator',
                    'Spatial phase, charged lower gap, and an operational rotation/deconfinement consumer'],
                excluded_claims=['isolated dynamical quartet', 'mechanical half-spin', 'deconfinement',
                                 'scalar Casimir equals matter-angular kinetics', 'canonical/full QED'],
                source_sha256=identities, runtime=dict(numpy=np.__version__, scipy=scipy.__version__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (('m', .2), ('r', .12), ('v', .3), ('stiffness', 2.), ('Jpref', .2)):
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.samples < 1 or not all(np.isfinite(x) and x > 0 for x in (args.m, args.r, args.v, args.stiffness, args.Jpref)):
        parser.error('Positive finite coefficients and at least one sample required')
    result = build(args)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Actual two-cell full occupied-band Gauss section; not a solved field ground."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from gauss_charge_intertwiner import boundary_word
from local_orbital_stability import scalar_paths, uniform_data
from quantum_su4_gauss import (PauliCode, PrimitiveMatter, norm, require_small,
                               source_hashes as supplier_source_hashes)
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis


SOURCE_NAMES = ('full_band_gauss_section.py', 'full_band_gauss_section.md',
                'gauss_born_bundle.py', 'full_field_charge_bound.py',
                'gauss_charge_intertwiner.py', 'quantum_su4_gauss.py',
                'quantum_su4_photon.py', 'quantum_spinor_plaquette.py',
                'local_orbital_stability.py', 'rotation_band.py', 'sea_selection.py')


def source_hashes():
    """Capture identities before constructing primitive operators or selected fields."""
    root = Path(__file__).resolve().parent
    return {**supplier_source_hashes(), **{
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in SOURCE_NAMES}}


@dataclass
class Section:
    state: np.ndarray
    initial: np.ndarray
    raw: np.ndarray
    weight: float
    local: list
    spectra: list
    projectors: list
    onebody: list


class FullBandGaussSection:
    """All 4900 actual Pauli states with N_H=N_L=4; no minor determinants."""

    def __init__(self, light_band, heavy_band=None, heavy_scale=3., chart_tolerance=1e-12):
        if not np.isfinite(heavy_scale) or heavy_scale <= 0:
            raise ValueError('Positive finite whole-heavy scale required')
        if not np.isfinite(chart_tolerance) or chart_tolerance <= 0:
            raise ValueError('Positive finite chart tolerance required')
        self.bands = (light_band if heavy_band is None else heavy_band, light_band)
        self.scale, self.tolerance = float(heavy_scale), float(chart_tolerance)
        self.basis = matrix_unit_basis()
        self.generators = self.basis / np.sqrt(8)
        self.code = PauliCode()
        self.matter = PrimitiveMatter(self.code, 4, self.generators)
        self.identity = sparse.diags(np.ones(self.matter.dimension, dtype=complex), format='csr')
        self.units = []
        for species in range(2):
            modes = tuple(range(4 * species, 4 * species + 4)) + tuple(range(8 + 4 * species, 12 + 4 * species))
            self.units.append({(i, j): self.matter.transfer(self.code, modes[j], modes[i])
                               for i in range(8) for j in range(8)})
        self.onsite, self.paths = [], []
        for band in self.bands:
            onsite, paths = scalar_paths(band)
            xpaths = [p for displacement, p in paths if tuple(displacement) == (1, 0, 0)]
            if band.improved or len(xpaths) != 1:
                raise ValueError('The owner uses one unimproved nearest x path')
            self.onsite.append(onsite - 2 * band.r / band.a)
            self.paths.append(xpaths[0])
        self.singlet = self._primitive_product_singlet()
        self.initial_singlet_gauss = [[norm(q @ self.singlet) for q in qs] for qs in self.matter.Q]
        require_small(max(map(max, self.initial_singlet_gauss)), 'primitive product epsilon singlet')
        self.number = []
        for species in range(2):
            left = sparse.diags(getattr(self.matter, 'heavy_left' if species == 0 else 'light_left'), dtype=float, format='csr')
            self.number.append([left, 4 * self.identity - left])

    def _primitive_product_singlet(self):
        """Embed both epsilon singlets with actual marker-word creation amplitudes."""
        terms = []
        for pair in itertools.combinations(range(4), 2):
            complement = tuple(i for i in range(4) if i not in pair)
            labels = pair + complement
            epsilon = (-1) ** sum(labels[i] > labels[j] for i in range(4) for j in range(i + 1, 4))
            terms.append((epsilon, pair + tuple(4 + i for i in complement)))
        result = np.zeros(self.matter.dimension, dtype=complex)
        words = [boundary_word(self.code, i, True) for i in range(16)]
        for (sign0, orbitals0), (sign1, orbitals1) in itertools.product(terms, repeat=2):
            mask, amplitude = 0, complex(sign0 * sign1)
            for orbital in reversed(orbitals0 + tuple(8 + i for i in orbitals1)):
                new = mask | (1 << orbital)
                amplitude *= self.code.element(mask, new, words[orbital], sign=1)
                mask = new
            result[self.matter.index[mask]] += amplitude / 6
        require_small(abs(norm(result) - 1), 'actual product singlet norm')
        return result

    def lift(self, matrix, species):
        result = sparse.csr_matrix(self.identity.shape, dtype=complex)
        for i, j in zip(*np.nonzero(matrix)):
            result = result + matrix[i, j] * self.units[species][i, j]
        return result.tocsr()

    def local_lift(self, matrix, site):
        block = np.zeros((8, 8), dtype=complex)
        block[4 * site:4 * site + 4, 4 * site:4 * site + 4] = matrix
        return self.lift(block, 1)

    @staticmethod
    @overload
    def filled(number, degree, vector, change: None = None, tangent=None) -> np.ndarray: ...

    @staticmethod
    @overload
    def filled(number, degree, vector, change: object, tangent=None) -> tuple[np.ndarray, np.ndarray]: ...

    @staticmethod
    def filled(number, degree, vector, change=None, tangent=None):
        """prod_(j=0)^(degree-1)(N-j)/degree! and exact noncommuting product rule."""
        value = vector.copy()
        derivative = np.zeros_like(vector) if tangent is None else tangent.copy()
        for j in range(degree):
            if change is not None:
                derivative = (number @ derivative - j * derivative + change @ value) / (j + 1)
            value = (number @ value - j * value) / (j + 1)
        return value if change is None else (value, derivative)

    def onebody(self, fields, U, phases=(0., 0.)):
        """Owner's m+r onsite and endpoint-averaged colour transport, two EM paths."""
        fields, U, phases = np.asarray(fields), np.asarray(U), np.asarray(phases)
        if fields.shape != (2, 4, 4, 4) or U.shape != (4, 4) or phases.shape != (2,):
            raise ValueError('Require fields[2,4,4,4], U[4,4], phases[2]')
        if not all(np.all(np.isfinite(x)) for x in (fields, U, phases)) or np.iscomplexobj(phases):
            raise ValueError('Finite fields/link and real compact phases required')
        if norm(fields - fields.conj().swapaxes(-1, -2)) > 1e-9 or norm(np.trace(fields, axis1=-2, axis2=-1)) > 1e-9:
            raise ValueError('Fields must be traceless Hermitian')
        if norm(U.conj().T @ U - np.eye(4)) > 1e-9 or abs(np.linalg.det(U) - 1) > 1e-9:
            raise ValueError('One genuine SU4 link required')
        factor = (np.exp(1j * phases[0]) + np.exp(-1j * phases[1])) / 2
        result = []
        for species in range(2):
            endpoint = np.einsum('m,xmij->xij', self.paths[species], fields)
            coupling = (endpoint[0] @ U + U @ endpoint[1]) / 2
            if species == 1:
                coupling *= factor
            h = np.zeros((8, 8), dtype=complex)
            h[:4, :4], h[4:, 4:] = self.onsite[species] * fields[0, 0], self.onsite[species] * fields[1, 0]
            h[:4, 4:], h[4:, :4] = coupling, coupling.conj().T
            result.append(h)
        return result

    def onebody_derivative(self, fields, U, phases, dfields, dU, dphases):
        result = []
        a0, a1 = phases
        factor = (np.exp(1j * a0) + np.exp(-1j * a1)) / 2
        dfactor = (1j * dphases[0] * np.exp(1j * a0) - 1j * dphases[1] * np.exp(-1j * a1)) / 2
        for species in range(2):
            endpoint = np.einsum('m,xmij->xij', self.paths[species], fields)
            dend = np.einsum('m,xmij->xij', self.paths[species], dfields)
            coupling = (endpoint[0] @ U + U @ endpoint[1]) / 2
            change = (dend[0] @ U + endpoint[0] @ dU + dU @ endpoint[1] + U @ dend[1]) / 2
            if species == 1:
                change = factor * change + dfactor * coupling
            h = np.zeros((8, 8), dtype=complex)
            h[:4, :4], h[4:, 4:] = self.onsite[species] * dfields[0, 0], self.onsite[species] * dfields[1, 0]
            h[:4, 4:], h[4:, :4] = change, change.conj().T
            result.append(h)
        return result

    def spectral(self, h, rank, top=False):
        values, vectors = np.linalg.eigh(h)
        gap = float(values[rank] - values[rank - 1])
        if gap <= self.tolerance * max(1., norm(h)):
            raise ValueError('Separated occupied-empty spectrum required; no chart fallback')
        occupied = np.arange(rank, 2 * rank) if top else np.arange(rank)
        empty = np.arange(rank) if top else np.arange(rank, 2 * rank)
        v = vectors[:, occupied]
        return v @ v.conj().T, (values, vectors, occupied, empty, gap)

    @staticmethod
    def spectral_derivative(spectrum, change):
        values, vectors, occupied, empty, _ = spectrum
        vo, ve = vectors[:, occupied], vectors[:, empty]
        mixing = (ve.conj().T @ change @ vo) / (values[occupied][None, :] - values[empty, None])
        block = ve @ mixing @ vo.conj().T
        return block + block.conj().T

    def state(self, fields, U, phases=(0., 0.)):
        h = self.onebody(fields, U, phases)
        local = [self.spectral(fields[x, 0], 2) for x in range(2)]
        initial = self.singlet.copy()
        for x, (p, _) in enumerate(local):
            initial = np.sqrt(6) * self.filled(self.local_lift(p, x), 2, initial)
        require_small(abs(norm(initial) - 1), 'local paired initial norm', 2e-9)
        global_data = [self.spectral(h[0], 4, top=True), self.spectral(h[1], 4)]
        raw = initial.copy()
        for species, (p, _) in enumerate(global_data):
            raw = self.filled(self.lift(p, species), 4, raw)
        weight = float(np.vdot(raw, raw).real)
        if not np.isfinite(weight) or weight <= self.tolerance:
            raise ValueError('Positive nonzero projection weight required; no alternative chart or frozen fallback')
        return Section(raw / np.sqrt(weight), initial, raw, weight, local,
                       [s for _, s in global_data], [p for p, _ in global_data], h)

    def derivative(self, section, fields, U, phases, dfields=None, dU=None, dphases=None):
        dfields = np.zeros_like(fields) if dfields is None else dfields
        dU = np.zeros_like(U) if dU is None else dU
        dphases = np.zeros(2) if dphases is None else dphases
        value, tangent = self.singlet.copy(), np.zeros_like(self.singlet)
        for x, (p, spectrum) in enumerate(section.local):
            dp = self.spectral_derivative(spectrum, dfields[x, 0])
            value, tangent = self.filled(self.local_lift(p, x), 2, value,
                                         self.local_lift(dp, x), tangent)
            value *= np.sqrt(6); tangent *= np.sqrt(6)
        changes = self.onebody_derivative(fields, U, phases, dfields, dU, dphases)
        for species in range(2):
            dp = self.spectral_derivative(section.spectra[species], changes[species])
            value, tangent = self.filled(self.lift(section.projectors[species], species), 4, value,
                                         self.lift(dp, species), tangent)
        derivative = tangent / np.sqrt(section.weight)
        derivative -= section.state * np.vdot(section.state, derivative).real
        return derivative

    def peierls_action(self, vector, fields, U, phases, left, right=None, species=1):
        """Independent primitive directed bilinears, not lifted onebody derivative."""
        if species == 0:
            return np.zeros_like(vector)
        endpoint = np.einsum('m,xmij->xij', self.paths[1], fields)
        factors = np.array([np.exp(1j * phases[0]), np.exp(-1j * phases[1])]) / 2
        factors *= np.array([1j * left[0], -1j * left[1]])
        if right is not None:
            factors *= np.array([1j * right[0], -1j * right[1]])
        result = np.zeros_like(vector)
        for i in range(4):
            for j in range(4):
                colour = sum((endpoint[0, i, k] * U[k, j] + U[i, k] * endpoint[1, k, j]) / 2 for k in range(4))
                coefficient = colour * sum(factors)
                directed = self.matter.forward[1, i, j]
                result += coefficient * (directed @ vector) + coefficient.conjugate() * (directed.conjugate().T @ vector)
        return result

    def controls(self, label, fields, U, phases, geometry=False):
        section = self.state(fields, U, phases)
        psi = section.state
        gauss, frozen, broken_heavy = [], [], []
        for site in range(2):
            site_gauss, site_frozen, site_broken = [], [], []
            for a, t in enumerate(self.generators):
                df = np.zeros_like(fields)
                df[site] = 1j * (t @ fields[site] - fields[site] @ t)
                du = 1j * t @ U if site == 0 else -1j * U @ t
                dpsi = self.derivative(section, fields, U, phases, df, du)
                q = self.matter.Q[site][a]
                ql = sum(t[i, j] * self.matter.units[site, 1, i, j] for i in range(4) for j in range(4))
                site_gauss.append(norm(dpsi - 1j * (q @ psi)))
                site_frozen.append(norm(q @ psi))
                site_broken.append(norm(dpsi - 1j * (ql @ psi)))
            gauss.append(site_gauss); frozen.append(site_frozen); broken_heavy.append(site_broken)
        em = []
        for site in range(2):
            gradient = np.array([1., -1.]) * (1 if site == 0 else -1)
            dpsi = self.derivative(section, fields, U, phases, dphases=gradient)
            em.append(norm(dpsi - 1j * ((self.number[1][site] - 2 * self.identity) @ psi)))
        covariance = []
        for site in range(2):
            coordinates = .17 * np.sin(np.arange(15) * .47 + .21 + site)
            t = np.einsum('a,aij->ij', coordinates, self.generators)
            g = expm(1j * t)
            transformed_fields = fields.copy()
            transformed_fields[site] = g @ fields[site] @ g.conj().T
            transformed_link = g @ U if site == 0 else U @ g.conj().T
            transformed = self.state(transformed_fields, transformed_link, phases).state
            q = sum(coordinates[a] * self.matter.Q[site][a] for a in range(15))
            covariance.append(norm(transformed - expm_multiply(1j * q, psi)))
        em_finite = []
        for site, theta in enumerate((.193, -.127)):
            shifted = np.array(phases) + theta * np.array([1., -1.]) * (1 if site == 0 else -1)
            transformed = self.state(fields, U, shifted).state
            expected = np.exp(1j * theta * (self.number[1][site].diagonal().real - 2)) * psi
            em_finite.append(norm(transformed - expected))
        energy, kinetic, variance = [], [], []
        for species in range(2):
            multiplier = -self.scale if species == 0 else 1.
            operator = multiplier * self.lift(section.onebody[species], species)
            eigenvalue = multiplier * sum(section.spectra[species][0][section.spectra[species][2]])
            energy.append(dict(expectation=float(np.vdot(psi, operator @ psi).real),
                               occupied_sum=float(eigenvalue), eigen_residual=norm(operator @ psi - eigenvalue * psi)))
            hopping = section.onebody[species].copy(); hopping[:4, :4] = 0; hopping[4:, 4:] = 0
            kinetic.append(float(np.vdot(psi, multiplier * self.lift(hopping, species) @ psi).real))
            n = self.number[species][0] @ psi
            variance.append(float(np.vdot(n, n).real - np.vdot(psi, n).real**2))
        density_errors, variance_errors = [], []
        for species, projector in enumerate(section.projectors):
            density = np.array([[np.vdot(psi, self.units[species][i, j] @ psi)
                                 for j in range(8)] for i in range(8)])
            density_errors.append(norm(density - projector.T))
            restricted = projector[:4, :4]
            variance_errors.append(abs(variance[species] - float(np.trace(restricted - restricted @ restricted).real)))
        require_small(max(density_errors), 'actual independent filled-plane densities', 3e-8)
        require_small(max(variance_errors), 'actual Slater number fluctuation', 3e-8)
        # Actual independent directions: all field components, colour-link tangent,
        # and both source coordinates; central differences use exact SU4 link paths.
        df = np.einsum('xma,aij->xmij', .031 * np.sin(np.arange(120).reshape(2, 4, 15) * .37 + .2), self.basis)
        link_generator = np.einsum('a,aij->ij', .029 * np.cos(np.arange(15) * .41), self.generators)
        dp = np.array([.23, -.19])
        tangent = self.derivative(section, fields, U, phases, df, 1j * link_generator @ U, dp)
        steps, errors = [2e-4, 1e-4], []
        for step in steps:
            plus = self.state(fields + step * df, expm(1j * step * link_generator) @ U, np.array(phases) + step * dp).state
            minus = self.state(fields - step * df, expm(-1j * step * link_generator) @ U, np.array(phases) - step * dp).state
            errors.append(norm((plus - minus) / (2 * step) - tangent))
        rng = np.random.default_rng(222)
        probe = rng.normal(size=self.matter.dimension) + 1j * rng.normal(size=self.matter.dimension)
        probe /= norm(probe)
        operator = self.lift(section.onebody[1], 1)
        current, contact, ward_j, ward_c = [], [], [], []
        fd_j, fd_c = [], []
        source_directions = [np.array([1., 0.]), np.array([0., 1.]), np.array([1., -1.]), np.array([.31, -.27])]
        zero_fields, zero_link = np.zeros_like(fields), np.zeros_like(U)
        for left in source_directions:
            derivative_h = self.onebody_derivative(fields, U, phases, zero_fields, zero_link, left)[1]
            j = self.peierls_action(probe, fields, U, phases, left)
            current.append(norm(j - self.lift(derivative_h, 1) @ probe))
            epsilon = 2e-4
            hp = self.lift(self.onebody(fields, U, np.array(phases) + epsilon * left)[1], 1)
            hm = self.lift(self.onebody(fields, U, np.array(phases) - epsilon * left)[1], 1)
            fd_j.append(norm((hp @ probe - hm @ probe) / (2 * epsilon) - j))
            for right in source_directions:
                c = self.peierls_action(probe, fields, U, phases, left, right)
                endpoint = np.einsum('m,xmij->xij', self.paths[1], fields)
                coupling = (endpoint[0] @ U + U @ endpoint[1]) / 2
                factor = -(left[0] * right[0] * np.exp(1j * phases[0]) + left[1] * right[1] * np.exp(-1j * phases[1])) / 2
                h2 = np.zeros((8, 8), dtype=complex)
                h2[:4, 4:] = factor * coupling; h2[4:, :4] = h2[:4, 4:].conj().T
                contact.append(norm(c - self.lift(h2, 1) @ probe))
                if np.array_equal(left, right):
                    fd_c.append(norm((hp @ probe - 2 * operator @ probe + hm @ probe) / epsilon**2 - c))
        gradient = np.array([1., -1.])
        for site in range(2):
            n = self.number[1][site]
            signed_gradient = gradient * (1 if site == 0 else -1)
            comm = n @ (operator @ probe) - operator @ (n @ probe)
            ward_j.append(norm(self.peierls_action(probe, fields, U, phases, signed_gradient) - 1j * comm))
            double = n @ (n @ (operator @ probe)) - 2 * n @ (operator @ (n @ probe)) + operator @ (n @ (n @ probe))
            ward_c.append(norm(self.peierls_action(probe, fields, U, phases, signed_gradient, signed_gradient) + double))
        heavy_operator = -self.scale * self.lift(section.onebody[0], 0)
        heavy_wrong = norm(1j * (self.number[0][0] @ (heavy_operator @ probe) - heavy_operator @ (self.number[0][0] @ probe)))
        heavy_source = norm(self.peierls_action(probe, fields, U, phases, gradient, species=0))
        for value, name in ((max(map(max, gauss)), 'all15 each-site SU4 differential Gauss'),
                            (max(em), 'both EM differential Gauss'), (max(covariance), 'finite SU4 covariance'),
                            (max(em_finite), 'finite EM covariance'), (max(e['eigen_residual'] for e in energy), 'whole onsite+hopping eigenstate'),
                            (max(current), 'independent primitive current'), (max(contact), 'independent primitive contact'),
                            (max(ward_j), 'current source Ward'), (max(ward_c), 'contact source Ward'),
                            (heavy_source, 'heavy source neutrality')):
            require_small(value, name, 3e-8)
        require_small(errors[-1], 'analytic normalized state derivative', 2e-6)
        require_small(max(fd_j), 'current finite difference', 2e-7)
        require_small(max(fd_c), 'contact finite difference', 3e-6)
        if errors[1] > .45 * errors[0] + 2e-9:
            raise AssertionError('Normalized state derivative central differences did not converge')
        if min(variance) <= 1e-8 or min(abs(k) for k in kinetic) <= 1e-8:
            raise AssertionError('Probe failed to resolve local fluctuations and both kinetic energies')
        if min(norm(frozen), norm(broken_heavy), heavy_wrong) <= 1e-5:
            raise AssertionError('Frozen-wavefunction/missing-heavy/charged-heavy adverse control unresolved')
        result: dict[str, Any] = dict(label=label, projection_weight=section.weight,
                      local_gaps=[s[-1] for _, s in section.local], global_gaps=[s[-1] for s in section.spectra],
                      onebody_spectra=[s[0].tolist() for s in section.spectra],
                      SU4_differential_Gauss_each_site_all15=gauss, EM_differential_Gauss_each_site=em,
                      finite_SU4_covariance_each_site=covariance, finite_EM_covariance_each_site=em_finite,
                      whole_species_energies=energy, heavy_light_kinetic_energies=kinetic,
                      heavy_light_left_number_variances=variance,
                      frozen_wavefunction_Gauss_defect=norm(frozen), missing_heavy_Gauss_defect=norm(broken_heavy),
                      erroneously_EM_charged_heavy_current_defect=heavy_wrong,
                      state_derivative_steps=steps, state_derivative_errors=errors,
                      primitive_current_vs_onebody_errors=current, primitive_contact_vs_onebody_errors=contact,
                      current_finite_difference_errors=fd_j, contact_finite_difference_errors=fd_c,
                      source_current_Ward_each_site=ward_j, source_contact_Ward_each_site=ward_c,
                      heavy_EM_source_action_norm=heavy_source,
                      actual_filled_plane_density_residuals=density_errors,
                      number_variance_vs_filled_plane_residuals=variance_errors,
                      fields_real=fields.real.tolist(), fields_imag=fields.imag.tolist(),
                      colour_link_real=U.real.tolist(), colour_link_imag=U.imag.tolist(), phases=list(phases))
        if geometry:
            tangents = []
            for site, component, a in itertools.product(range(2), range(4), range(15)):
                direction = np.zeros_like(fields); direction[site, component] = self.basis[a]
                tangents.append(self.derivative(section, fields, U, phases, dfields=direction))
            for direction in np.eye(2):
                tangents.append(self.derivative(section, fields, U, phases, dphases=direction))
            tangents = np.asarray(tangents)
            berry = tangents @ psi.conj()
            tensor = tangents.conj() @ tangents.T - np.outer(berry.conj(), berry)
            result['geometry'] = dict(coordinates='120 real field coordinates then A0,A1; colour link held fixed',
                quantum_metric=tensor.real.tolist(), Berry_connection=(-1j * berry).real.tolist(),
                Berry_curvature=(2 * tensor.imag).tolist(), metric_trace=float(np.trace(tensor.real)),
                EM_curvature_block=(2 * tensor[-2:, -2:].imag).tolist(),
                claim='observed section geometry; no opposite-sea Berry cancellation assumption')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--heavy-scale', type=float, default=3.)
    for name, default in (('m', .8), ('r', .2), ('v', .3), ('stiffness', 2.)):
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--geometry', action='store_true', help='emit full 122-coordinate metric and Berry curvature')
    args = parser.parse_args()
    if not all(np.isfinite(x) for x in (args.m, args.r, args.v, args.stiffness)):
        parser.error('Finite bare coefficients and stiffness required')
    if args.stiffness <= 0:
        parser.error('Positive stiffness required')
    identities = source_hashes()
    band = RotationBand(a=1., m=args.m, r=args.r, v=args.v, improved=False)
    model = FullBandGaussSection(band, heavy_scale=args.heavy_scale)
    data = uniform_data(band, 3, args.stiffness, 0., model.basis)
    fields = np.tile(data['frame'], (2, 1, 1, 1)).astype(complex)
    deformation = np.einsum('xma,aij->xmij', .018 * np.sin(np.arange(120).reshape(2, 4, 15) * .43 + .17), model.basis)
    link_generator = np.einsum('a,aij->ij', .13 * np.sin(np.arange(15) * .37 + .29), model.generators)
    deformed = fields + deformation
    non_gamma = max(norm(deformed[x, m] @ deformed[x, n] + deformed[x, n] @ deformed[x, m]
                         - np.trace(deformed[x, m] @ deformed[x, n] + deformed[x, n] @ deformed[x, m]) / 4 * np.eye(4))
                    for x in range(2) for m in range(4) for n in range(m, 4))
    if non_gamma <= 1e-5:
        raise AssertionError('Deformed probe is not genuinely outside scalar Clifford Gram law')
    cases = [model.controls('selected', fields, np.eye(4, dtype=complex), (0., 0.), args.geometry),
             model.controls('non_gamma_fields_colour_link_and_EM_source', deformed, expm(1j * link_generator), (.137, -.081), args.geometry)]
    invalid_chart_rejected = False
    try:
        model.state(np.zeros_like(fields), np.eye(4, dtype=complex))
    except ValueError:
        invalid_chart_rejected = True
    if not invalid_chart_rejected:
        raise AssertionError('Unseparated occupied chart was not rejected')
    zero_overlap_fields = np.zeros_like(fields)
    zero_overlap_fields[0, 0] = np.diag([-100., -99., -98., 297.])
    zero_overlap_fields[1, 0] = np.diag([-3., -2., 1., 4.])
    zero_overlap_phases = (np.pi, 0.)
    zero_overlap_h = model.onebody(zero_overlap_fields, np.eye(4, dtype=complex), zero_overlap_phases)
    zero_overlap_local_gaps = [model.spectral(zero_overlap_fields[x, 0], 2)[1][-1] for x in range(2)]
    zero_overlap_global_gaps = [model.spectral(zero_overlap_h[0], 4, True)[1][-1],
                                model.spectral(zero_overlap_h[1], 4)[1][-1]]
    zero_overlap_rejected = False
    try:
        model.state(zero_overlap_fields, np.eye(4, dtype=complex), zero_overlap_phases)
    except ValueError as error:
        zero_overlap_rejected = 'projection weight' in str(error)
    if not zero_overlap_rejected:
        raise AssertionError('Separated-spectrum zero-overlap chart was not rejected')
    report = dict(status='actual_full_occupied_two_cell_covariant_Gauss_section_NOT_full_field_ground_or_phase',
        source_sha256=identities, primitive_modes=16, NH=4, NL=4, dimension=model.matter.dimension,
        parameters={key: value for key, value in vars(args).items() if key != 'output'},
        heavy_scale=model.scale, primitive_product_singlet_all15_each_site=model.initial_singlet_gauss,
        owner_onsite_coefficients=model.onsite,
        owner_x_path_coefficients_real=[p.real.tolist() for p in model.paths],
        owner_x_path_coefficients_imag=[p.imag.tolist() for p in model.paths],
        non_gamma_scalar_anticommutator_defect=non_gamma,
        unseparated_chart_rejected=invalid_chart_rejected, cases=cases,
        zero_overlap_chart=dict(rejected=zero_overlap_rejected,
                               local_gaps=zero_overlap_local_gaps, global_gaps=zero_overlap_global_gaps),
        analytic_domain='Separated local rank2 and global rank4 spectra, projection weight strictly positive; gauge-invariant smooth packet support inside this open chart',
        limits=['two cells and one x-axis with two compact parallel EM paths, not a full60 field spectrum or 3D scattering',
                'all60 real field coordinates per site are allowed; section derivatives are not a solved field Hamiltonian',
                'no unconditional arbitrary-field/link overlap positivity; no EM-independent paired Berry cancellation claim',
                'no canonical QED, full phase, Lorentz invariance or mechanical-spin conclusion'])
    if source_hashes() != identities:
        raise RuntimeError('Scientific source changed during execution; refusing mixed-source receipt')
    payload = json.dumps(report, indent=2, allow_nan=False)
    args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()

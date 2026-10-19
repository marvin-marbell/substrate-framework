#!/usr/bin/env python3
"""Exact all-coexact Gaussian/Weyl kernels and four-packet finite Fock chart.

This is a basis/integration core, not a new preparation or a dynamical closure.
Harmonics are excluded: their free UE/N ptheta^2 remains with the state owner.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile
from typing import Any

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy.linalg import expm
from numpy.polynomial.hermite import hermgauss

Monomial = tuple[int, ...]


@dataclass
class ProjectionPolynomial:
    """p(y)=sum c_I product_{i in I}(projections[i] dot y), bilinear dot.

    Terms use sorted factor-index tuples, including () for the constant.
    Only small projection Gram matrices are formed, never dimension^degree.
    """
    projections: np.ndarray
    terms: dict[Monomial, complex]

    def __post_init__(self) -> None:
        self.projections = np.asarray(self.projections, complex)
        if self.projections.ndim != 2:
            raise ValueError('Projection rows must have shape (k, all_mode_dimension)')
        combined: dict[Monomial, complex] = {}
        for factors, coefficient in self.terms.items():
            if any(i < 0 or i >= len(self.projections) for i in factors):
                raise ValueError('Polynomial factor index outside projection bank')
            indices = tuple(sorted(factors))
            combined[indices] = combined.get(indices, 0j) + coefficient
        self.terms = {i: c for i, c in combined.items() if c != 0}

    @classmethod
    def constant(cls, dimension: int, value: complex = 1) -> ProjectionPolynomial:
        return cls(np.empty((0, dimension), complex), {(): value})

    def evaluate(self, y: np.ndarray) -> tuple[complex, np.ndarray]:
        y = np.asarray(y, float)
        if y.shape != (self.projections.shape[1],):
            raise ValueError('All real coexact coordinates required')
        z = self.projections @ y
        value = 0j
        gradient = np.zeros(len(y), complex)
        for factors, coefficient in self.terms.items():
            value += coefficient * np.prod(z[list(factors)])
            for pos, index in enumerate(factors):
                remaining = factors[:pos] + factors[pos + 1:]
                gradient += coefficient * np.prod(z[list(remaining)]) * self.projections[index]
        return complex(value), gradient

    def conjugate(self) -> ProjectionPolynomial:
        return ProjectionPolynomial(self.projections.conj(), {i: c.conjugate() for i, c in self.terms.items()})

    def multiply(self, other: ProjectionPolynomial) -> ProjectionPolynomial:
        if self.projections.shape[1] != other.projections.shape[1]:
            raise ValueError('Different all-mode coordinate charts')
        offset = len(self.projections)
        terms: dict[Monomial, complex] = {}
        for a, ca in self.terms.items():
            for b, cb in other.terms.items():
                indices = a + tuple(i + offset for i in b)
                terms[indices] = terms.get(indices, 0j) + ca * cb
        return ProjectionPolynomial(np.vstack((self.projections, other.projections)), terms)

    def directional_derivative(self, direction: np.ndarray) -> ProjectionPolynomial:
        direction = np.asarray(direction, complex)
        if direction.shape != (self.projections.shape[1],):
            raise ValueError('All-mode direction required')
        slopes = self.projections @ direction
        terms: dict[Monomial, complex] = {}
        for factors, coefficient in self.terms.items():
            for pos, index in enumerate(factors):
                remaining = factors[:pos] + factors[pos + 1:]
                terms[remaining] = terms.get(remaining, 0j) + coefficient * slopes[index]
        return ProjectionPolynomial(self.projections, terms)


def fourier_gaussian_moment(projections: np.ndarray, factors: Monomial,
                            displacement: np.ndarray) -> complex:
    """E[exp(i l.y) product(v_i.y)] for density pi^(-d/2) exp(-y.y).

    Wick recursion uses mean i v.l/2 and covariance v_i.v_j/2, not v†v.
    It supports every finite degree; complexity follows factor/projection count.
    """
    projections = np.asarray(projections, complex)
    displacement = np.asarray(displacement, float)
    if projections.ndim != 2 or projections.shape[1] != len(displacement):
        raise ValueError('Projection/displacement chart mismatch')
    mean = .5j * (projections @ displacement)
    covariance = .5 * (projections @ projections.T)

    @lru_cache(None)
    def moment(indices: Monomial) -> complex:
        if not indices:
            return 1 + 0j
        first, rest = indices[0], indices[1:]
        value = mean[first] * moment(rest)
        for pos, index in enumerate(rest):
            value += covariance[first, index] * moment(rest[:pos] + rest[pos + 1:])
        return complex(value)

    return complex(math.exp(-float(displacement @ displacement) / 4) * moment(tuple(sorted(factors))))


def polynomial_fourier_integral(polynomial: ProjectionPolynomial,
                                displacement: np.ndarray) -> complex:
    """Exact all-mode Fourier Gaussian polynomial integral."""
    return sum((c * fourier_gaussian_moment(polynomial.projections, i, displacement)
                for i, c in polynomial.terms.items()), 0j)


@dataclass
class GaussianFunction:
    """Normalized base Gaussian times polynomial times exp(i phase.y)."""
    polynomial: ProjectionPolynomial
    phase: np.ndarray

    def __post_init__(self) -> None:
        self.phase = np.asarray(self.phase, float)
        if self.phase.shape != (self.polynomial.projections.shape[1],):
            raise ValueError('Real all-mode Weyl phase required')

    def evaluate(self, y: np.ndarray) -> tuple[complex, np.ndarray]:
        y = np.asarray(y, float)
        value, gradient = self.polynomial.evaluate(y)
        gaussian = math.pi ** (-len(y) / 4) * np.exp(-float(y @ y) / 2 + 1j * (self.phase @ y))
        return complex(gaussian * value), gaussian * (gradient + (1j * self.phase - y) * value)

    def directional_derivative(self, direction: np.ndarray) -> GaussianFunction:
        """Return the actual wave derivative, including Gaussian and Weyl jets."""
        direction = np.asarray(direction, float)
        derivative = self.polynomial.directional_derivative(direction)
        linear = ProjectionPolynomial(direction[None, :], {(): complex(1j * self.phase @ direction), (0,): -1})
        product = self.polynomial.multiply(linear)
        # Preserve the original projection bank and append the single y.direction.
        terms = dict(product.terms)
        for i, c in derivative.terms.items():
            terms[i] = terms.get(i, 0j) + c
        return GaussianFunction(ProjectionPolynomial(product.projections, terms), self.phase)


def gaussian_kernel(left: GaussianFunction, right: GaussianFunction,
                    displacement: np.ndarray | None = None) -> complex:
    """<left|exp(i displacement.y)|right>, exact on ALL real modes."""
    delta = right.phase - left.phase
    if displacement is not None:
        displacement = np.asarray(displacement, float)
        if displacement.shape != delta.shape:
            raise ValueError('All-mode displacement required')
        delta = delta + displacement
    return polynomial_fourier_integral(left.polynomial.conjugate().multiply(right.polynomial), delta)


def native_packets(law: Any, em: dict[str, Any]) -> np.ndarray:
    """Literal ez exp(±2pi i x_j/N)/sqrt(V), order +x,-x,+y,-y.

    Physical input is neither whitened nor refiltered. Projection is checked
    against the complete coexact bank, rather than repairing the input.
    """
    vectors = np.asarray(em['vectors'], float)
    packets = []
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        field = np.zeros((law.volume, 3), complex)
        field[:, 2] = np.exp(sign * 2j * math.pi * law.sites[:, axis] / law.size) / math.sqrt(law.volume)
        coordinates = vectors.T @ field.ravel()
        if np.linalg.norm(vectors @ coordinates - field.ravel()) > 2e-10:
            raise ValueError('Literal native packet is not in the full coexact chart')
        packets.append(coordinates)
    result = np.column_stack(packets)
    if not np.allclose(result.conj().T @ result, np.eye(4), atol=2e-10, rtol=0):
        raise ValueError('Literal four native packets are not normalized orthogonal')
    return result


def fock_labels(max_total: int) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted((n for n in itertools.product(range(max_total + 1), repeat=4)
                         if sum(n) <= max_total), key=lambda n: (sum(n), n)))


class HardFockBasis:
    """Four orthonormal complex oscillator packets in a full real EM chart."""
    def __init__(self, em: dict[str, Any], hard_packets: np.ndarray, max_total: int = 2):
        self.omega = np.asarray(em['omega'], float)
        self.packets = np.array(hard_packets, complex, copy=True)
        if self.packets.shape != (len(self.omega), 4):
            raise ValueError('Four complex packets in ALL coexact modes required')
        if not np.allclose(self.packets.conj().T @ self.packets, np.eye(4), atol=2e-10, rtol=0):
            raise ValueError('Unwhitened physical packets must already be orthonormal')
        if not isinstance(max_total, int) or max_total < 0:
            raise ValueError('Nonnegative total Fock cutoff required')
        self.max_total = max_total
        self.labels = fock_labels(max_total)
        self.index = {n: i for i, n in enumerate(self.labels)}
        self.dimension = len(self.labels)
        self.bilinear = self.packets.T @ self.packets
        self.packet_omega = np.real(np.diag(self.packets.conj().T @ (self.omega[:, None] * self.packets)))
        if np.linalg.norm(self.omega[:, None] * self.packets - self.packets * self.packet_omega) > 2e-10:
            raise ValueError('Diagonal free energies require actual monochromatic packets')
        self.zero_point = float(np.sum(self.omega) / 2)
        self.free_energy = self.zero_point + np.asarray(self.labels) @ self.packet_omega
        self.creation = np.zeros((4, self.dimension, self.dimension), complex)
        for column, n in enumerate(self.labels):
            for mode in range(4):
                raised = list(n); raised[mode] += 1
                row = self.index.get(tuple(raised))
                if row is not None:
                    self.creation[mode, row, column] = math.sqrt(n[mode] + 1)
        self.annihilation = self.creation.conj().transpose(0, 2, 1)
        self.coordinate_polynomials = tuple(self._polynomial(n) for n in self.labels)

    def _polynomial(self, n: tuple[int, ...]) -> ProjectionPolynomial:
        terms: dict[Monomial, complex] = {(): 1}
        # a†(g)[p psi0] = [sqrt2(g.y)p - (g.grad p)/sqrt2] psi0.
        for mode, occupation in enumerate(n):
            for count in range(occupation):
                next_terms: dict[Monomial, complex] = {}
                for factors, coefficient in terms.items():
                    raised = tuple(sorted(factors + (mode,)))
                    next_terms[raised] = next_terms.get(raised, 0j) + math.sqrt(2) * coefficient
                    for pos, index in enumerate(factors):
                        remaining = factors[:pos] + factors[pos + 1:]
                        next_terms[remaining] = next_terms.get(remaining, 0j) - self.bilinear[mode, index] * coefficient / math.sqrt(2)
                terms = {i: c / math.sqrt(count + 1) for i, c in next_terms.items()}
        return ProjectionPolynomial(self.packets.T, terms)

    def polynomials(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.empty(self.dimension, complex)
        derivative = np.empty((len(self.omega), self.dimension), complex)
        for i, polynomial in enumerate(self.coordinate_polynomials):
            values[i], derivative[:, i] = polynomial.evaluate(y)
        return values, derivative

    def evaluate(self, y: np.ndarray) -> dict[str, np.ndarray]:
        y = np.asarray(y, float)
        polynomial, derivative = self.polynomials(y)
        gaussian = math.pi ** (-len(y) / 4) * math.exp(-float(y @ y) / 2)
        return dict(polynomial=polynomial, polynomial_derivative=derivative,
                    value=gaussian * polynomial,
                    first_derivative=gaussian * (derivative - y[:, None] * polynomial))

    def function(self, index: int, phase: np.ndarray | None = None) -> GaussianFunction:
        return GaussianFunction(self.coordinate_polynomials[index], np.zeros(len(self.omega)) if phase is None else phase)

    def weyl_matrix(self, displacement_all52: np.ndarray) -> np.ndarray:
        """P exp(i l.y) P, NOT exp(i P(l.y)P); includes outside-hard Debye."""
        displacement = np.asarray(displacement_all52, float)
        if displacement.shape != self.omega.shape:
            raise ValueError('Full real coexact displacement required; no harmonics')
        alpha = self.packets.conj().T @ (1j * displacement / math.sqrt(2))
        result = np.empty((self.dimension, self.dimension), complex)
        common = math.exp(-float(displacement @ displacement) / 4)
        for row, m in enumerate(self.labels):
            for column, n in enumerate(self.labels):
                factor = 1 + 0j
                for mode in range(4):
                    mm, nn = m[mode], n[mode]
                    factor *= math.sqrt(math.factorial(mm) * math.factorial(nn)) * sum(
                        (alpha[mode] ** (mm - k) * (-alpha[mode].conjugate()) ** (nn - k)
                         / (math.factorial(k) * math.factorial(mm - k) * math.factorial(nn - k))
                         for k in range(min(mm, nn) + 1)), 0j)
                result[row, column] = common * factor
        return result

    def arbitrary_creator(self, packet: np.ndarray) -> dict[str, Any]:
        """Exact P a†(packet)P and orthogonal outside norm maps for any packet.

        boundary_map rows are degree cutoff+1 hard labels; outside_hard_map
        rows have the original hard labels and one residual-mode photon.
        Their Gram sum equals (a†P)†(I-P)(a†P), including coherent interference.
        """
        packet = np.asarray(packet, complex)
        if packet.shape != self.omega.shape:
            raise ValueError('Arbitrary creator needs the full coexact packet')
        coefficients = self.packets.conj().T @ packet
        residual = packet - self.packets @ coefficients
        residual_norm = float(np.vdot(residual, residual).real)
        projected = np.einsum('i,ijk->jk', coefficients, self.creation)
        boundary_labels = tuple(n for n in fock_labels(self.max_total + 1) if sum(n) == self.max_total + 1)
        lookup = {n: i for i, n in enumerate(boundary_labels)}
        boundary = np.zeros((len(boundary_labels), self.dimension), complex)
        for column, n in enumerate(self.labels):
            if sum(n) == self.max_total:
                for mode in range(4):
                    raised = list(n); raised[mode] += 1
                    boundary[lookup[tuple(raised)], column] += coefficients[mode] * math.sqrt(n[mode] + 1)
        outside_hard = math.sqrt(residual_norm) * np.eye(self.dimension)
        outside_gram = boundary.conj().T @ boundary + residual_norm * np.eye(self.dimension)
        full_norm_gram = float(np.vdot(packet, packet).real) * np.eye(self.dimension) + projected @ projected.conj().T
        return dict(projected=projected, hard_coefficients=coefficients,
                    outside_hard_packet=residual, outside_hard_norm_squared=residual_norm,
                    boundary_labels=boundary_labels, boundary_map=boundary,
                    outside_hard_map=outside_hard, outside_gram=outside_gram,
                    full_norm_gram=full_norm_gram)


def packet_templates(law: Any, basis: HardFockBasis, time: float = 6) -> dict[str, Any]:
    """Transport-owned native positions/phases, CREATOR coefficients.

    Source19 eta is common to all geometries and is deliberately not shifted
    here. Annihilator coefficients are the conjugates of creator coefficients.
    """
    centre = np.array([1, 1, 1])
    positions = np.array([centre, centre + [1, 0, 0], centre + [0, 1, 0]])
    wavevectors = 2 * math.pi / law.size * np.array([[1, 0, 0], [-1, 0, 0],
                                                   [0, 1, 0], [0, -1, 0]])
    weights = np.array([4, 1]) / math.sqrt(17)
    coefficients = np.zeros((3, 4), complex)
    coefficients[0, :2] = weights * np.exp(-1j * wavevectors[:2] @ positions[0])
    coefficients[1, :2] = weights * np.exp(-1j * wavevectors[:2] @ positions[1])
    coefficients[2, 2:] = weights * np.exp(-1j * wavevectors[2:] @ positions[2]) * np.exp(-1j * basis.packet_omega[2:] * time)
    if law.size != 3:
        raise ValueError('These authoritative positions/templates are native N3 diagnostics only')
    return dict(names=('incoming', 'forward', 'outgoing'), positions=positions,
                native_wavevectors=wavevectors, creator_hard_coefficients=coefficients,
                annihilator_hard_coefficients=coefficients.conj(),
                creator_all52=(basis.packets @ coefficients.T).T,
                annihilator_all52=(basis.packets @ coefficients.T).conj().T,
                outgoing_free_phase=np.exp(-1j * basis.packet_omega[2:] * time),
                time=time, SOURCE_eta_common=True,
                scope='native N3 hard IDs, not the legal-N Source19 photon IDs')


def _quadrature_matrix(basis: HardFockBasis, real_frame: np.ndarray,
                       displacement: np.ndarray, order: int = 12,
                       derivatives: bool = False) -> np.ndarray:
    """Independent tensor Gauss-Hermite on controlled real effective projections."""
    nodes, weights = hermgauss(order)
    result = np.zeros((basis.dimension, basis.dimension), complex)
    for indices in itertools.product(range(order), repeat=real_frame.shape[1]):
        x = nodes[list(indices)]; y = real_frame @ x
        weight = float(np.prod(weights[list(indices)]) / math.pi ** (len(indices) / 2))
        values, gradient = basis.polynomials(y)
        if derivatives:
            derivative = real_frame[:, 0] @ gradient - x[0] * values
            result += weight * np.outer(derivative.conj(), derivative) * np.exp(1j * displacement @ y)
        else:
            result += weight * np.outer(values.conj(), values) * np.exp(1j * displacement @ y)
    return result


def scientific_receipt(state_npz: Path, output_npz: Path, source_archive: Path) -> dict[str, Any]:
    # Supplier H/preparation is never executed. Only its coexact coordinate builder.
    from improved_phi_band import ImprovedPhiLaw, selected_band
    from prepared_neutral_ambient_weak_state import coexact_coordinates
    source_root = Path(__file__).resolve().parent
    sources: dict[str, bytes] = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename)
            if path.parent == source_root and path.suffix == '.py':
                sources[path.name] = path.read_bytes()
    with zipfile.ZipFile(source_archive, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in sorted(sources.items()):
            archive.writestr(name, body)
    band, _, _ = selected_band(); law = ImprovedPhiLaw(3, band)
    em = coexact_coordinates(law, 2. ** -20, .125)
    packets = native_packets(law, em); basis = HardFockBasis(em, packets)
    templates = packet_templates(law, basis)
    frequency_error = float(np.max(abs(basis.packet_omega - .125 * math.sqrt(3))))
    template_norm_error = float(np.max(abs(np.sum(abs(templates['creator_hard_coefficients']) ** 2, axis=1) - 1)))
    real_frame = np.linalg.qr(np.column_stack((np.real(packets[:, 0]), np.imag(packets[:, 0]),
                                             np.real(packets[:, 2]), np.imag(packets[:, 2]))))[0]
    displacement = real_frame @ np.array([.47, -.31, .23, .19])
    exact = basis.weyl_matrix(displacement)
    quadrature = _quadrature_matrix(basis, real_frame, displacement)
    hermite = np.array([[gaussian_kernel(basis.function(i), basis.function(j), displacement)
                         for j in range(basis.dimension)] for i in range(basis.dimension)])
    overlap = np.array([[gaussian_kernel(basis.function(i), basis.function(j))
                         for j in range(basis.dimension)] for i in range(basis.dimension)])
    direction = real_frame[:, 0]
    derivative_exact = np.array([[gaussian_kernel(basis.function(i).directional_derivative(direction),
                                                basis.function(j).directional_derivative(direction), displacement)
                                  for j in range(basis.dimension)] for i in range(basis.dimension)])
    derivative_quad = _quadrature_matrix(basis, real_frame, displacement, derivatives=True)
    outside = np.eye(52)[:, 0] - packets @ (packets.conj().T @ np.eye(52)[:, 0])
    outside = np.real(outside); outside /= np.linalg.norm(outside)
    all_displacement = displacement + .61 * outside
    outside_weyl = basis.weyl_matrix(all_displacement)
    reduced_debye = exact
    compressed = sum((1j * (np.vdot(packets[:, i], displacement) * basis.creation[i]
                           + np.vdot(packets[:, i], displacement).conjugate() * basis.annihilation[i]) / math.sqrt(2)
                      for i in range(4)), np.zeros_like(exact))
    wrong_compressed = expm(compressed)
    creator = basis.arbitrary_creator(packets[:, 0] + .4 * outside)
    n2 = basis.index[(2, 0, 0, 0)]
    boundary_norm = float(creator['outside_gram'][n2, n2].real)
    full_creator_error = np.linalg.norm(creator['projected'].conj().T @ creator['projected']
                                       + creator['outside_gram'] - creator['full_norm_gram'])
    # Wrong complex Wick: treating bilinear as Hermitian subtracts1 for two +x.
    wrong_poly = ProjectionPolynomial(packets.T, {(0, 0): math.sqrt(2), (): -1 / math.sqrt(2)})
    wrong_wick = abs(gaussian_kernel(GaussianFunction(wrong_poly, np.zeros(52)), basis.function(0)))
    # Higher-degree generic outside-mode Weyl polynomial, not a hard-mode echo.
    high_polynomial = ProjectionPolynomial(outside[None, :], {(0,) * 8: 1, (0, 0): .3j})
    high_function = GaussianFunction(high_polynomial, .2 * outside)
    shifted_function = GaussianFunction(ProjectionPolynomial.constant(52), -.4 * outside)
    high_exact = gaussian_kernel(shifted_function, high_function)
    hn, hw = hermgauss(20)
    high_quadrature = np.sum(hw * (hn ** 8 + .3j * hn ** 2) * np.exp(.6j * hn)) / math.sqrt(math.pi)
    y = real_frame @ np.array([.13, -.17, .21, -.11])
    eps = 2. ** -18
    wave = basis.function(n2, .3 * outside)
    _, wave_gradient = wave.evaluate(y)
    finite_derivative = (wave.evaluate(y + eps * outside)[0] - wave.evaluate(y - eps * outside)[0]) / (2 * eps)
    derivative_error = abs(finite_derivative - outside @ wave_gradient)
    errors = dict(weyl_quadrature=float(np.linalg.norm(exact - quadrature)),
                  weyl_hermite=float(np.linalg.norm(exact - hermite)),
                  overlap=float(np.linalg.norm(overlap - np.eye(15))),
                  derivative_quadrature=float(np.linalg.norm(derivative_exact - derivative_quad)),
                  creator_norm_identity=float(full_creator_error),
                  monochromatic_frequency=frequency_error, template_norm=template_norm_error,
                  higher_degree_outside_Weyl=float(abs(high_exact - high_quadrature)),
                  value_first_derivative=float(derivative_error))
    defects = dict(exp_compressed_A=float(np.linalg.norm(exact - wrong_compressed)),
                   reduced_hard_Debye=float(np.linalg.norm(outside_weyl - reduced_debye)),
                   Hermitian_instead_of_complex_Wick=float(wrong_wick),
                   missing_n2_creator_outside_norm_squared=boundary_norm)
    if max(errors.values()) > 2e-9 or min(defects.values()) < .01:
        raise AssertionError(dict(errors=errors, defects=defects))
    with np.load(state_npz, allow_pickle=False) as saved:
        state = saved['neutral_state']; gram = saved['neutral_Gram']
    embedded = np.zeros((len(state), basis.dimension), complex); embedded[:, 0] = state
    norm_before = float(np.vdot(state, gram @ state).real)
    norm_after = float(np.vdot(embedded.ravel(), np.kron(gram, overlap) @ embedded.ravel()).real)
    if abs(norm_before - 1) > 2e-10 or abs(norm_after - norm_before) > 2e-10:
        raise AssertionError('Frozen state vacuum embedding changed its metric norm')
    np.savez_compressed(output_npz, packets=packets, omega=basis.omega,
                        labels=np.asarray(basis.labels), creation=basis.creation,
                        annihilation=basis.annihilation, free_energy=basis.free_energy,
                        weyl=outside_weyl, displacement=all_displacement,
                        neutral_vacuum_embedding=embedded, creator_outside_gram=creator['outside_gram'],
                        template_creator_hard=templates['creator_hard_coefficients'],
                        template_creator_all52=templates['creator_all52'], template_positions=templates['positions'])
    return dict(native_size=3, all_real_coexact_modes=len(basis.omega), hard_packets=4,
                total_cutoff=2, fock_dimension=basis.dimension, hard_order=['+x', '-x', '+y', '-y'],
                packet_Gram_error=float(np.linalg.norm(packets.conj().T @ packets - np.eye(4))),
                coexact_orthogonality_error=float(np.linalg.norm(em['vectors'].T @ em['vectors'] - np.eye(52))),
                coexact_projector_error=float(np.linalg.norm(em['vectors'] @ em['vectors'].T - em['coexact_projector'])),
                native_omega=basis.packet_omega.tolist(), expected_omega=.125 * math.sqrt(3),
                bilinear_Gram={'real': np.real(basis.bilinear).tolist(), 'imag': np.imag(basis.bilinear).tolist()},
                all52_zero_point_once=basis.zero_point, harmonic_excluded=True,
                errors=errors, rejected_defects=defects,
                quadrature=dict(real_effective_dimensions=4, order=12, nodes=12 ** 4,
                                independent_tensor_Gauss_Hermite=True),
                frozen_state_metric_norm=norm_before, vacuum_embedded_metric_norm=norm_after,
                state_input_sha256=hashlib.sha256(state_npz.read_bytes()).hexdigest(),
                output_npz_sha256=hashlib.sha256(output_npz.read_bytes()).hexdigest(),
                source_sha256={name: hashlib.sha256(body).hexdigest() for name, body in sorted(sources.items())},
                loaded_sources_unchanged=all((source_root / name).read_bytes() == body for name, body in sources.items()),
                frozen_source_archive=str(source_archive),
                frozen_source_archive_sha256=hashlib.sha256(source_archive.read_bytes()).hexdigest(),
                templates=dict(names=templates['names'], positions=templates['positions'].tolist(),
                    native_wavevectors=templates['native_wavevectors'].tolist(),
                    creator_hard_real=templates['creator_hard_coefficients'].real.tolist(),
                    creator_hard_imag=templates['creator_hard_coefficients'].imag.tolist(),
                    time=6, SOURCE_eta_common=True, scope=templates['scope']),
                debts=['finite Fock dynamical leakage and outside native Noether partners',
                       'native mode IDs and diagnostic parameters are not legal-N hard IDs',
                       'analytic EM integration may change vacuum H compression; no state refilter',
                       'full Phi/Haar and harmonic integration remain state-owner obligations'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-npz', type=Path, required=True)
    parser.add_argument('--output-npz', type=Path, required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    args = parser.parse_args(); started = time.perf_counter()
    result = scientific_receipt(args.state_npz, args.output_npz, args.source_archive)
    result['wall_seconds'] = time.perf_counter() - started
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

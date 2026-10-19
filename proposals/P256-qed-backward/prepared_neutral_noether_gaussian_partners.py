#!/usr/bin/env python3
"""Literal native sine-clover EM Noether functions, before Gaussian projection."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, cast
import zipfile

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import mpmath as mp
import numpy as np

from original_total_angular_transfer import primitive_flow
from prepared_neutral_weyl_hard_fock import (GaussianFunction, HardFockBasis,
    ProjectionPolynomial, gaussian_kernel, native_packets)
from prepared_neutral_quartet_functions import QuartetSourceFunctionBank, source19_native_profile

GUARD_SHA256 = '5aeceaf4a7cb104346ebc5d1130e22bae6090b5cf96afcd297e02a6facc22128'
STATE_SHA256 = 'eb283954e7ebb13e1b5368deb058214443d9a839b8319161c0306afb2c8b53b6'


@dataclass(frozen=True)
class FiniteGaussianFunction:
    """A finite sum of full-52 polynomial/Weyl waves; never a Fock projection."""
    terms: tuple[GaussianFunction, ...]

    def evaluate(self, y: np.ndarray) -> tuple[complex, np.ndarray]:
        value = 0j
        derivative = np.zeros(len(y), complex)
        for term in self.terms:
            v, d = term.evaluate(y)
            value += v
            derivative += d
        return value, derivative

    def directional_derivative(self, direction: np.ndarray) -> FiniteGaussianFunction:
        return FiniteGaussianFunction(tuple(term.directional_derivative(direction) for term in self.terms))


def function_kernel(left: FiniteGaussianFunction, right: FiniteGaussianFunction,
                    displacement: np.ndarray | None = None) -> complex:
    return sum((gaussian_kernel(a, b, displacement) for a in left.terms for b in right.terms), 0j)


def creator_function(function: FiniteGaussianFunction, packet: np.ndarray) -> FiniteGaussianFunction:
    """a†_g on wave factors: sqrt(2)(g.y)p - (g.grad p+i g.phase p)/sqrt(2).

    g is a complex CREATOR packet (bilinear contraction, not conjugated).
    On the full wave this is (g.y - g.grad)/sqrt(2); the sqrt(2) formula
    applies to the factor divided by the common Gaussian, as in HardFockBasis.
    """
    packet = np.asarray(packet, complex)
    out = []
    for term in function.terms:
        if packet.shape != term.phase.shape:
            raise ValueError('Creator requires the entire real-mode coordinate chart')
        linear = ProjectionPolynomial(packet[None, :], {(0,): math.sqrt(2),
            (): complex(-1j * (packet @ term.phase) / math.sqrt(2))})
        product = term.polynomial.multiply(linear)
        derivative = term.polynomial.directional_derivative(packet)
        terms = dict(product.terms)
        for factors, value in derivative.terms.items():
            terms[factors] = terms.get(factors, 0j) - value / math.sqrt(2)
        out.append(GaussianFunction(ProjectionPolynomial(product.projections, terms), term.phase))
    return FiniteGaussianFunction(tuple(out))


def high_precision_kernel(left: FiniteGaussianFunction, right: FiniteGaussianFunction,
                          digits: int = 60) -> complex:
    """Exact Wick/Fourier moments with high precision SUM, not clipped small norms.

    Input vectors remain their actual native double coefficients. mpmath is
    needed because nearly opposite Weyl terms cancel at the native UE scale.
    """
    with mp.workdps(digits):
        total = mp.mpc(0)
        for a in left.terms:
            for b in right.terms:
                p = a.polynomial.conjugate().multiply(b.polynomial)
                delta = b.phase - a.phase
                rows = [[mp.mpc(complex(z)) for z in row] for row in p.projections]
                d = [mp.mpf(float(z)) for z in delta]
                mean = [mp.j * sum(x * y for x, y in zip(row, d)) / 2 for row in rows]
                cov = [[sum(x * y for x, y in zip(r, s)) / 2 for s in rows] for r in rows]
                cache: dict[tuple[int, ...], Any] = {(): mp.mpc(1)}
                def moment(factors: tuple[int, ...]) -> Any:
                    if factors not in cache:
                        i, rest = factors[0], factors[1:]
                        value = mean[i] * moment(rest)
                        for pos, j in enumerate(rest):
                            value += cov[i][j] * moment(rest[:pos] + rest[pos + 1:])
                        cache[factors] = value
                    return cache[factors]
                total += mp.exp(-sum(x * x for x in d) / 4) * sum(
                    mp.mpc(complex(c)) * moment(i) for i, c in p.terms.items())
        return complex(total)


class NoetherGaussianBank:
    """Own the literal RAW EM vector field and conditional strong SOURCE partners.

    model is the ORIGINAL AmbientWeakState, not a new preparation. Scalar
    coordinates are all real coexact y; harmonic theta is a distinct chart.
    """
    def __init__(self, model: Any, velocity: np.ndarray):
        self.model = model
        self.law = model.law
        law = self.law
        self.velocity = np.asarray(velocity, float)
        if self.velocity.shape != (law.volume, 3):
            raise ValueError('Full native edge-averaged site velocity required')
        v = law.volume
        gradient = np.zeros((3 * v, v))
        for x in range(v):
            for axis in range(3):
                gradient[3 * x + axis, law.target[x, axis]] = 1
                gradient[3 * x + axis, x] = -1
        self.gradient = gradient
        # Canonical integer edge curls: opposite loop vectors share one sine.
        loops: dict[tuple[int, ...], np.ndarray] = {}
        def curl(x: int, j: int, i: int, s: int, t: int) -> tuple[tuple[int, ...], int]:
            row = np.zeros(3 * v, dtype=np.int8)
            site = x
            for axis, sign in ((j, s), (i, t), (j, -s), (i, -t)):
                if sign == 1:
                    row[3 * site + axis] += 1
                    site = int(law.target[site, axis])
                else:
                    coord = law.sites[site].copy()
                    coord[axis] = (coord[axis] - 1) % law.size
                    site = law.index[tuple(coord)]
                    row[3 * site + axis] -= 1
            if site != x:
                raise AssertionError('Oriented native plaquette did not close')
            sign = 1 if row[np.flatnonzero(row)[0]] > 0 else -1
            return tuple(int(z) for z in sign * row), sign
        # This is exactly primitive_flow's endpoint clover and edge velocity
        # averages. No continuum angular pair or linearized curl is inserted.
        for x in range(v):
            for i in range(3):
                endpoint = int(law.target[x, i])
                for j in range(3):
                    if i == j:
                        continue
                    velocity_average = (self.velocity[x, j] + self.velocity[endpoint, j]) / 2
                    low, high = min(i, j), max(i, j)
                    orientation = 1 if j < i else -1
                    for anchor in (x, endpoint):
                        for s, t in itertools.product((-1, 1), repeat=2):
                            key, sign = curl(anchor, low, high, s, t)
                            coefficient = -velocity_average * orientation * s * t * sign / (8 * law.band.a)
                            if key not in loops:
                                loops[key] = np.zeros(3 * v)
                            loops[key][3 * x + i] += coefficient
        keys = sorted(loops)
        self.edge_curls = np.asarray(keys, float)
        raw = np.asarray([loops[key] for key in keys])
        self.raw_coefficients = raw
        legs = np.asarray(model.em['legs'])
        phases = self.edge_curls @ legs
        coexact = (raw @ model.em['vectors']) * np.sqrt(model.em['omega'] / (2 * model.ue))[None, :]
        harmonic = law.size * raw.reshape(-1, v, 3).mean(axis=1)
        lambdas = -(raw @ gradient) @ model.green
        lambdas -= lambdas[:, model.reference_site, None]
        self.coefficients = dict(phases=phases, coexact=coexact,
                                 harmonic=harmonic, lambda_C=lambdas)
        self.dimension = len(model.em['omega'])
        self.hard = HardFockBasis(model.em, native_packets(law, model.em))
        self._gamma: FiniteGaussianFunction | None = None
        self._scalar_gram: np.ndarray | None = None
        self._neutral_rho = np.asarray([[np.trace(p[4*x:4*x+4, 4*x:4*x+4]).real - 2
            for x in range(v)] for p in (model.seed_fibre.projectors[1], model.seed_fibre.projectors[-1])])
        # Ref backgrounds are -4 versus 0 occupation; anchored lambda_R=0.
        self._neutral_rho[1, model.reference_site] -= 4

    def evaluate(self, y: np.ndarray) -> dict[str, Any]:
        y = np.asarray(y, float)
        if y.shape != (self.dimension,):
            raise ValueError('Every real coexact coordinate is required')
        c = self.coefficients
        angle = c['phases'] @ y
        sine, cosine = np.sin(angle), np.cos(angle)
        flow = sine @ c['coexact']
        divergence = float(cosine @ np.einsum('ri,ri->r', c['coexact'], c['phases']))
        gamma = 1j * (flow @ y - divergence / 2)
        return dict(flow=flow, divergence=divergence, gamma=complex(gamma),
            harmonic=sine @ c['harmonic'], lambda_C=sine @ c['lambda_C'],
            raw=(sine @ self.raw_coefficients).reshape(self.law.volume, 3))

    def gaussian_functions(self) -> tuple[FiniteGaussianFunction, ...]:
        """Scalar library: hard15, common EVEN gamma, then R common ODD sines."""
        if self._gamma is None:
            terms = []
            for phase, flow in zip(self.coefficients['phases'], self.coefficients['coexact']):
                dot = float(flow @ phase)
                for sign in (-1, 1):
                    p = ProjectionPolynomial(flow[None, :], {(0,): sign / 2, (): -.25j * dot})
                    terms.append(GaussianFunction(p, sign * phase))
            self._gamma = FiniteGaussianFunction(tuple(terms))
        base = tuple(FiniteGaussianFunction((self.hard.function(i),)) for i in range(self.hard.dimension))
        sines = tuple(FiniteGaussianFunction(tuple(GaussianFunction(
            ProjectionPolynomial.constant(self.dimension, sign / (2j)), sign * phase)
            for sign in (-1, 1))) for phase in self.coefficients['phases'])
        return (*base, self._gamma, *sines)

    def scalar_gram(self) -> np.ndarray:
        """Exact conditional EM Gram; parity and sinh avoid sine cancellation."""
        if self._scalar_gram is None:
            functions = self.gaussian_functions()
            k, phases = self.hard.dimension, self.coefficients['phases']
            out = np.zeros((len(functions), len(functions)), complex)
            out[:k, :k] = np.eye(k)
            # gamma even, sin odd; gamma vacuum mean is exactly zero by IBP.
            gamma = functions[k]
            for i in range(1, k):
                if sum(self.hard.labels[i]) == 2:
                    out[i, k] = high_precision_kernel(functions[i], gamma)
            out[k, :k] = out[:k, k].conj()
            out[k, k] = high_precision_kernel(gamma, gamma)
            norm = np.einsum('ri,ri->r', phases, phases)
            out[k+1:, k+1:] = np.exp(-(norm[:, None] + norm[None, :]) / 4) * np.sinh(phases @ phases.T / 2)
            for i, label in enumerate(self.hard.labels):
                if sum(label) == 1:
                    packet = self.hard.packets[:, label.index(1)]
                    out[i, k+1:] = (phases @ packet.conj()) / math.sqrt(2) * np.exp(-norm / 4)
            out[k+1:, :k] = out[:k, k+1:].conj().T
            self._scalar_gram = out
        return self._scalar_gram.copy()

    def scalar_creator_kernels(self, packet: np.ndarray) -> dict[str, np.ndarray]:
        """<f|a†g f> and <a†g f|a†g f>, BEFORE finite-basis projection.

        Gamma cancellation uses 60-digit moments; sine full norms use the
        exact CCR and sinh/cosh rather than subtracting close Weyl values.
        Returned matrices do not assume Fock<=2 creator closure.
        """
        packet = np.asarray(packet, complex)
        if packet.shape != (self.dimension,):
            raise ValueError('Full52 complex CREATOR packet required')
        functions = self.gaussian_functions()
        created = tuple(creator_function(f,packet) for f in functions)
        k, size = self.hard.dimension, len(functions)
        parity = np.r_[[sum(n)%2 for n in self.hard.labels],0,np.ones(size-k-1,int)]
        cross = np.zeros((size,size),complex)
        full = np.zeros_like(cross)
        for i in range(size):
            for j in range(size):
                if parity[i] != parity[j]:
                    kernel = high_precision_kernel if i == k or j == k else function_kernel
                    cross[i,j] = kernel(functions[i],created[j])
            for j in range(i+1):
                if parity[i] == parity[j] and not (i > k and j > k):
                    kernel = high_precision_kernel if i == k or j == k else function_kernel
                    full[i,j] = kernel(created[i],created[j])
                    full[j,i] = full[i,j].conjugate()
        phases = self.coefficients['phases']
        norm = np.einsum('ri,ri->r',phases,phases)
        cosgram = np.exp(-(norm[:,None]+norm[None,:])/4)*np.cosh(phases@phases.T/2)
        slopes = phases@packet
        full[k+1:,k+1:] = (float(np.vdot(packet,packet).real)*self.scalar_gram()[k+1:,k+1:]
            + .5*slopes[:,None]*slopes.conj()[None,:]*cosgram)
        return dict(cross=cross,full_norm=full)

    def scalar_values(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Values/jets DIVIDED common EM Gaussian, same convention as A17."""
        p, dp = self.hard.polynomials(y)
        c = self.coefficients
        angles = c['phases'] @ y
        sine, cosine = np.sin(angles), np.cos(angles)
        dots = np.einsum('ri,ri->r', c['coexact'], c['phases'])
        fy = c['coexact'] @ y
        gamma = 1j * (fy @ sine - .5 * dots @ cosine)
        dgamma = 1j * (sine @ c['coexact'] + ((fy * cosine + .5 * dots * sine) @ c['phases']))
        return np.r_[p, gamma, sine], np.column_stack((dp, dgamma, c['phases'].T * cosine))

    def conditional_coefficients(self, sample: dict[str, Any], source_bank: QuartetSourceFunctionBank,
                                 prepared14: np.ndarray) -> dict[str, Any]:
        """Coefficient tensors for exact EM integration at fixed full Phi/theta.

        Shapes neutral[2,4,S], charged[448,3*T*4*2,S], S=15+1+R.
        Charged column order category(even/commonodd/relativeodd),template,word,bank.
        SOURCE global2 lives in original unit-word producer, exactly once.
        """
        c14 = np.asarray(prepared14, complex)
        if c14.shape != (14,) or sample['neutral'].shape != (2, 14):
            raise ValueError('The immutable prepared14/original neutral basis is required')
        fibre = source_bank.seed_fibre
        point, theta = sample['point'], sample['theta']
        phi, links = point['phi'], point['links']
        psi = sample['neutral'] @ c14
        R, T, k = len(self.raw_coefficients), len(source_bank.template_order), self.hard.dimension
        S, per = k + 1 + R, T * 8
        neutral = np.zeros((2, 4, S), complex)
        charged = np.zeros((fibre.charged_dimension, 3 * per, S), complex)
        lam, harm = self.coefficients['lambda_C'], self.coefficients['harmonic']
        weak = lam @ self._neutral_rho.T + 1j * (harm @ theta)[:, None]
        for bank in range(2):
            neutral[bank, bank, k] = psi[bank]
            neutral[bank, 2+bank, k+1:] = weak[:, bank] * psi[bank]
        relative_details = []
        displacement = (self.law.sites - self.law.sites[self.model.reference_site]) / self.law.size
        relative_scalar = lam - lam[:, self.model.reference_site, None] - harm @ displacement.T
        for t, profile in enumerate(source_bank.profiles.values()):
            for word in range(4):
                source = fibre.source_vectors_at(phi, links, theta, sample['a'], profile, np.eye(4)[word])
                marks = source['full_SOURCE_marks']
                plus = marks[1].reshape(fibre.n, 4)
                minus = marks[-1].transpose(1, 0, 2).reshape(4, fibre.n)
                relative = np.zeros((R, fibre.charged_dimension, 2), complex)
                scalar = np.repeat(relative_scalar, 4, axis=1)
                relative[:, fibre.charged_slices[1], 0] = np.einsum('an,rn,nb->rba', fibre.empty.conj().T, scalar, plus).reshape(R, -1)
                relative[:, fibre.charged_slices[-1], 1] = -np.einsum('an,rn,nb->rab', minus, scalar, fibre.occupied[-1]).reshape(R, -1)
                for bank in range(2):
                    column = (4*t + word)*2 + bank
                    vector = source['source_matrix'][:, bank] * psi[bank]
                    charged[:, column, k] = vector
                    charged[:, per+column, k+1:] = vector[:, None] * weak[:, bank]
                    charged[:, 2*per+column, k+1:] = relative[:, :, bank].T * psi[bank]
                relative_details.append((source, relative))
        return dict(neutral=neutral, charged=charged, prepared_bank_values=psi,
            weak_odd=weak, relative_details=relative_details,
            scalar_order='hard15,gamma,sinR', charged_order='category,template,word,bank',
            norm_semantics='raw physical functions; no SOURCE/profile/partner normalization')

    def generated_values(self, sample: dict[str, Any], source_bank: QuartetSourceFunctionBank,
                         prepared14: np.ndarray) -> dict[str, np.ndarray]:
        coefficients = self.conditional_coefficients(sample, source_bank, prepared14)
        scalar, _ = self.scalar_values(sample['em'][:self.dimension])
        return dict(neutral_extra=coefficients['neutral'] @ scalar,
                    charged_extra=coefficients['charged'] @ scalar)

    def generated_source_map(self, source_bank: QuartetSourceFunctionBank,
                             beta: np.ndarray, selected_profile: str) -> np.ndarray:
        """Exact SOURCE map on all four new neutral columns; PLUS beta/MINUS beta*."""
        weights = np.asarray(beta, complex)
        if weights.shape != (4,):
            raise ValueError('Original raw four-word quartet required')
        T = len(source_bank.template_order)
        t = source_bank.template_order.index(selected_profile)
        per = 8*T
        out = np.zeros((2, 3*per, 4), complex)
        for bank, word_weights in enumerate((weights, weights.conj())):
            for category in (0, 1):
                for word, coefficient in enumerate(word_weights):
                    out[bank, category*per+(4*t+word)*2+bank, 2*category+bank] = coefficient
        return out

    def conditional_kernels(self, sample: dict[str, Any], source_bank: QuartetSourceFunctionBank,
                            prepared14: np.ndarray) -> dict[str, np.ndarray]:
        coefficients = self.conditional_coefficients(sample, source_bank, prepared14)
        gram = self.scalar_gram()
        result = {}
        for sector in ('neutral', 'charged'):
            c = coefficients[sector]
            result[sector] = np.einsum('acs,st,adt->cd', c.conj(), gram, c, optimize=True)
        return result

    def generated_directional_values(self, sample: dict[str, Any], source_bank: QuartetSourceFunctionBank,
            prepared14: np.ndarray, normal: np.ndarray, harmonic: np.ndarray,
            coexact: np.ndarray, step: float = 2.**-17) -> dict[str, Any]:
        """Native refined FD coefficients plus exact Gaussian/Haar wave derivative.

        This is a finite computation, not a certified analytic Phi/source jet.
        """
        dq, dem = np.asarray(normal, float), np.r_[coexact, harmonic]
        scale = max(1., float(np.linalg.norm(dq)), float(np.linalg.norm(dem)))
        eps = step / scale
        def shifted(t: float) -> dict[str, Any]:
            q, em = sample['q'] + t*dq, sample['em'] + t*dem
            return self.model.basis_at(q, em, self.model.chart.embedding(q))
        def derivative(h: float) -> dict[str, np.ndarray]:
            plus = self.generated_values(shifted(h), source_bank, prepared14)
            minus = self.generated_values(shifted(-h), source_bank, prepared14)
            return {key: (plus[key]-minus[key])/(2*h) for key in plus}
        coarse, fine = derivative(eps), derivative(eps/2)
        density = self.model.chart.directional_jet(sample['q'], dq, sample['point'])['log_density_first'] if np.any(dq) else 0.
        logder = -float(sample['q'] @ dq)/self.model.sigma**2 - float(sample['em'] @ dem) - .5*density
        logder += float(sample['cutoff_gradient'] @ dq)/sample['cutoff']
        values = self.generated_values(sample, source_bank, prepared14)
        out: dict[str, Any] = {key: fine[key]+logder*values[key] for key in fine}
        out['native_error_estimate'] = {key: float(np.linalg.norm(fine[key]-coarse[key])/3) for key in fine}
        out['step'] = eps/2
        return out

    def projection_receipt(self) -> dict[str, Any]:
        gram = self.scalar_gram()
        k = self.hard.dimension
        full = float(gram[k, k].real)
        inside = float(np.vdot(gram[:k, k], gram[:k, k]).real)
        outside = full-inside
        if full <= 0 or outside <= 0:
            raise AssertionError('Literal native gamma has no positive outside-hard function norm')
        augmented = gram[:k+1, :k+1]
        recovered = np.linalg.solve(augmented, gram[:k+1, k])
        target = np.zeros(k+1, complex)
        target[k] = 1
        difference = recovered-target
        recovery_error = float(np.sqrt(np.vdot(difference, augmented@difference).real))
        return dict(full_gamma_norm_squared=full, hard15_norm_squared=inside,
            outside_hard15_norm_squared=outside, outside_fraction=outside/full,
            augmented_gamma_recovery_error=recovery_error,
            recovery_semantics='exact Gram solved on actual appended gamma function, not polynomial closure',
            scalar_dimension=len(gram), unique_oriented_curl_vectors=len(self.edge_curls))


def scientific_receipt(root_lemma: Path, state_npz: Path, descriptor: Path,
                       guard_path: Path, soft_path: Path) -> dict[str, Any]:
    from improved_phi_band import ImprovedPhiLaw, selected_band
    from prepared_neutral_ambient_weak_state import AmbientWeakState
    from prepared_neutral_canonical_source import BETA
    started = time.perf_counter()
    if hashlib.sha256(guard_path.read_bytes()).hexdigest() != GUARD_SHA256:
        raise ValueError('The exact physical native guard input is required')
    if hashlib.sha256(state_npz.read_bytes()).hexdigest() != STATE_SHA256:
        raise ValueError('The frozen A17 state must remain byte-identical')
    guard = json.loads(guard_path.read_bytes())
    band, frame, _ = selected_band()
    law = ImprovedPhiLaw(3, band)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = np.tile(np.eye(4, dtype=complex), (law.volume, 3, 1, 1))
    eta = source19_native_profile(law, descriptor)['profile']
    model = AmbientWeakState(law, fields, links, law.index[(0,1,1)], eta, json.loads(root_lemma.read_bytes()))
    soft_fixture = json.loads(soft_path.read_bytes())
    soft = np.asarray(soft_fixture['coordinate_restriction']['real']) + 1j*np.asarray(soft_fixture['coordinate_restriction']['imag'])
    if soft_fixture['momentum_centre'] != [2e-7, 0., 0.] or soft_fixture['width'] != 1e-7:
        raise ValueError('Original SOFT centre/width must not be retuned')
    source_bank = QuartetSourceFunctionBank(model.seed_fibre, {'source19': eta, 'soft': soft})
    bank = NoetherGaussianBank(model, np.asarray(guard['velocity']))
    if len(bank.edge_curls) != 81:
        raise AssertionError('Exact N3 oriented-clover merging did not produce 81 curls')
    with np.load(state_npz) as data:
        c14 = data['neutral_state'].copy()
        frozen_norm = float(np.vdot(c14, data['neutral_Gram'] @ c14).real)
    rng = np.random.default_rng(222709)
    q = rng.normal(size=model.chart.normal_dim) * model.sigma / math.sqrt(2)
    y = rng.normal(size=bank.dimension) / math.sqrt(2)
    theta = np.array([.31, -.22, .17])
    sample = model.basis_at(q, np.r_[y, theta])
    flow = bank.evaluate(y)
    raw = primitive_flow(cast(Any, law), sample['point']['phi'], sample['point']['links'], sample['a'],
                         bank.velocity, np.asarray(guard['alpha']))[2]
    lam = -model.green @ bank.gradient.T @ raw.ravel()
    lam -= lam[model.reference_site]
    coeq = (model.em['vectors'].T @ raw.ravel()) * np.sqrt(model.em['omega']/(2*model.ue))
    errors = dict(raw_primitive=float(np.linalg.norm(raw-flow['raw'])),
        coexact_projection=float(np.linalg.norm(coeq-flow['flow'])),
        anchored_lambda_sign=float(np.linalg.norm(lam-flow['lambda_C'])),
        harmonic_projection=float(np.linalg.norm(law.size*raw.mean(axis=0)-flow['harmonic'])))
    divergence_fd = 0.
    h = 2.**-16
    for j in range(bank.dimension):
        direction = np.eye(bank.dimension)[j]
        divergence_fd += (bank.evaluate(y+h*direction)['flow'][j]-bank.evaluate(y-h*direction)['flow'][j])/(2*h)
    errors['divergence_analytic'] = abs(divergence_fd-flow['divergence'])
    scalar, scalar_derivative = bank.scalar_values(y)
    gamma_wave, gamma_derivative = bank.gaussian_functions()[15].evaluate(y)
    vacuum = math.pi**(-bank.dimension/4)*math.exp(-float(y@y)/2)
    errors['gamma_finite_Weyl_value'] = abs(gamma_wave/vacuum-flow['gamma'])
    errors['gamma_finite_Weyl_derivative'] = float(np.linalg.norm(gamma_derivative/vacuum-(scalar_derivative[:,15]-y*scalar[15])))
    generated = bank.generated_values(sample, source_bank, c14)
    coefficients = bank.conditional_coefficients(sample, source_bank, c14)
    # Literal strong relative lambda/source harmonic commutator, all 4 words,
    # both +/- and every Ref mode, with projected neutral RIGHT outside retained.
    fibre = source_bank.seed_fibre
    harmonic_body = (law.sites-law.sites[model.reference_site]) @ flow['harmonic'] / law.size
    body = np.diag(np.repeat(flow['lambda_C']-harmonic_body, 4))
    ref = flow['lambda_C'][model.reference_site]*np.eye(4)
    qn = fibre.onebody(body, 'neutral')+fibre.reference(ref, 'neutral')
    qc = fibre.onebody(body)+fibre.reference(ref)
    right_norms, comm_errors, projected_errors = [], [], []
    for source, relative_r in coefficients['relative_details']:
        b, marks = source['source_matrix'], source['full_SOURCE_marks']
        relative = np.einsum('r,rcb->cb', np.sin(bank.coefficients['phases']@y), relative_r)
        right = np.zeros_like(b)
        right[fibre.charged_slices[1],0] = (-fibre.empty.conj().T@body@fibre.projectors[1]@marks[1].reshape(fibre.n,4)).T.reshape(-1)
        mm = marks[-1].transpose(1,0,2).reshape(4,fibre.n)
        right[fibre.charged_slices[-1],1] = (mm@(np.eye(fibre.n)-fibre.projectors[-1])@body@fibre.occupied[-1]).reshape(-1)
        comm_errors.append(float(np.linalg.norm(qc@b-b@qn-right-relative)))
        right_norms.append(float(np.linalg.norm(right)))
        projected_errors.append(float(np.linalg.norm(qc@b-b@qn-relative)))
    errors['strong_relative_CAR_RIGHT'] = max(comm_errors)
    # Source derivative is actually observed, not inferred only from labels.
    coeq_direction = rng.normal(size=bank.dimension)
    def source_shift(dem: np.ndarray, scale: float) -> np.ndarray:
        a, th = model.electromagnetic_field(sample['em']+scale*dem)
        return source_bank.word_sources(sample['point']['phi'], sample['point']['links'], th, a)
    dem = np.r_[coeq_direction, np.zeros(3)]
    errors['coexact_SOURCE_commutation'] = float(np.linalg.norm(source_shift(dem,h)-source_shift(dem,-h))/(2*h))
    demh = np.r_[np.zeros(bank.dimension), flow['harmonic']]
    actual_dtheta = (source_shift(demh,h)-source_shift(demh,-h))/(2*h)
    expected_dtheta = np.zeros_like(actual_dtheta)
    for item, (source, _) in enumerate(coefficients['relative_details']):
        t, word = divmod(item, 4)
        marks = source['full_SOURCE_marks']
        plus = -1j*np.repeat(harmonic_body,4)[:,None]*marks[1].reshape(fibre.n,4)
        minus = 1j*marks[-1].transpose(1,0,2).reshape(4,fibre.n)*np.repeat(harmonic_body,4)[None,:]
        expected_dtheta[t,word] = fibre.source_vectors_from_marks({1:plus.reshape(law.volume,4,4),
            -1:minus.reshape(4,law.volume,4).transpose(1,0,2)})['source_matrix']
    errors['literal_harmonic_SOURCE_derivative'] = float(np.linalg.norm(actual_dtheta-expected_dtheta))
    extra_maps = {name:bank.generated_source_map(source_bank,BETA,name) for name in source_bank.template_order}
    map_errors = []
    for t, name in enumerate(source_bank.template_order):
        actual_source = fibre.source_vectors_at(sample['point']['phi'], sample['point']['links'],
            theta, sample['a'], source_bank.profiles[name], BETA)['source_matrix']
        expected = actual_source @ generated['neutral_extra']
        mapped = np.zeros_like(expected)
        for b, charge in enumerate((1,-1)):
            rows = fibre.charged_slices[charge]
            mapped[rows] = generated['charged_extra'][rows] @ extra_maps[name][b]
        map_errors.append(float(np.linalg.norm(mapped-expected)))
    errors['SOURCE_on_all_new_neutral_columns'] = max(map_errors)
    conditional_norms = bank.conditional_kernels(sample,source_bank,c14)
    if np.min(cast(Any, np.diag(conditional_norms['neutral'])).real) <= 0:
        raise AssertionError('An actual generated neutral function lost its full Gaussian norm')
    per = len(source_bank.template_order)*8
    qn_coul = bank.coefficients['lambda_C']@bank._neutral_rho.T
    # The actual wrong extra rho insertion, not a label-only lambda norm.
    doubled_value = np.zeros_like(generated['charged_extra'][:,per:2*per])
    for t in range(len(source_bank.template_order)):
        for word in range(4):
            for b in range(2):
                col = (4*t+word)*2+b
                source = coefficients['relative_details'][4*t+word][0]['source_matrix'][:,b]
                doubled_value[:,col] = source*coefficients['prepared_bank_values'][b]*(np.sin(bank.coefficients['phases']@y)@qn_coul[:,b])
    projection = bank.projection_receipt()
    linear = (bank.coefficients['phases']@y) @ bank.raw_coefficients
    sins = np.sin(bank.coefficients['phases']@y)
    defects = dict(mean_EM_zero_discard=float(np.linalg.norm(flow['flow'])),
        linear_SIN=float(np.linalg.norm(raw.ravel()-linear)),
        wrong_lambda_sign=2*float(np.linalg.norm(flow['lambda_C'])),
        missing_harmonic=float(np.linalg.norm(expected_dtheta)),
        missing_neutral_RIGHT=max(projected_errors),
        doubled_rho_lambda=float(np.linalg.norm(doubled_value)),
        missing_anchored_Ref=float(np.linalg.norm((-(bank.raw_coefficients@bank.gradient)@model.green)[:,model.reference_site]@sins)))
    # Creator on full functions, not truncated hard matrices; nonzero tail paid.
    creator = creator_function(bank.gaussian_functions()[15], bank.hard.packets[:,0])
    cv, _ = creator.evaluate(y)
    expected_cv = (bank.hard.packets[:,0]@y*gamma_wave-bank.hard.packets[:,0]@gamma_derivative)/math.sqrt(2)
    errors['full_function_creator_value'] = abs(cv-expected_cv)/vacuum
    creator_kernels = bank.scalar_creator_kernels(bank.hard.packets[:,0])
    creator_norm = creator_kernels['full_norm'][15,15]
    errors['creator_vacuum_full_norm'] = abs(creator_kernels['full_norm'][0,0]-1)
    creator_hard_norm = float(np.vdot(creator_kernels['cross'][:15,15],creator_kernels['cross'][:15,15]).real)
    if creator_norm.real <= 0 or abs(creator_norm.imag) > 1e-10:
        raise AssertionError('The actual all52 generated-function creator norm failed')
    creator_product_error = float(np.linalg.norm((source_shift(dem,h)-source_shift(dem,-h))*cv/(2*h)))
    errors['full_creator_SOURCE_commutator'] = creator_product_error
    # No PH norm claims are manufactured by the weak relative identity.
    coul_body = np.diag(np.repeat(flow['lambda_C'],4))
    graph = fibre.onebody_graph_diagnostic(coul_body)
    native_kinematic = dict(phase_norm=float(np.linalg.norm(bank.coefficients['phases'])),
        all52_y_nonzero_count=int(np.count_nonzero(y)), harmonic_theta=theta.tolist(),
        raw_flow_norm=float(np.linalg.norm(raw)), generated_neutral_shape=list(generated['neutral_extra'].shape),
        generated_charged_shape=list(generated['charged_extra'].shape),
        neutral_RIGHT_norm=max(right_norms), neutral_Coulomb_PH_norm_squared=graph['neutral_particle_hole_norm_squared'],
        generated_neutral_full_norm_squared=cast(Any, np.diag(conditional_norms['neutral'])).real.tolist(),
        generated_charged_full_norm_squared=cast(Any, np.diag(conditional_norms['charged'])).real.tolist(),
        gamma_creator_full_norm_squared=float(creator_norm.real),
        gamma_creator_outside_hard15_norm_squared=float(creator_norm.real)-creator_hard_norm)
    if max(errors.values()) > 2e-7 or min(defects.values()) <= 0:
        raise AssertionError(json.dumps(dict(errors=errors, defects=defects)))
    return dict(scope='new actual nativeN3 literal finiteWeyl/all52/Ref4 strong SOURCE functions; not full closed rotation',
        errors=errors, rejected_defects=defects, projection=projection, native=native_kinematic,
        frozen_state_norm=frozen_norm, frozen_state_sha256=STATE_SHA256, guard_sha256=GUARD_SHA256,
        profiles=list(source_bank.template_order), source_profile_norm=float(np.linalg.norm(eta)),
        soft_profile_norm=float(np.linalg.norm(soft)), soft_profile_renormalized=False,
        soft_input_sha256=hashlib.sha256(soft_path.read_bytes()).hexdigest(),
        neutral_preparation_rerun=False, beta_normalization=False, all_words=['I','Gamma23','Gamma31','Gamma12'],
        law=dict(Lambda=model.Lambda, UE=model.ue, c=model.c, sigma=model.sigma, radius=model.radius,
            electric='UE Pi^2', magnetic='K/2 curl^2', Coulomb='UE rho Green rho BEFORE projection'),
        debts=['Phi/harmonic native quadrature and FD jets are finite-computation debts',
            'full Coulomb neutral/charged PH and Q norm are separate, not captured by generated weak functions',
            'infinite native Weyl Fock tails, full G/U2pi and same-current Born integration remain owner obligations',
            'SOFT native restriction loses almost all physical spatial mass; periodic Fourier projection is zero'],
        wall_seconds=time.perf_counter()-started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-lemma', type=Path, required=True)
    parser.add_argument('--state-npz', type=Path, required=True)
    parser.add_argument('--descriptor', type=Path, required=True)
    parser.add_argument('--guard', type=Path, required=True)
    parser.add_argument('--soft-profile', type=Path, required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    __import__('prepared_neutral_ambient_weak_state')
    sources = {Path(cast(str, m.__file__)).name:Path(cast(str, m.__file__)).read_bytes() for m in list(sys.modules.values())
        if getattr(m,'__file__',None) and Path(cast(str, m.__file__)).parent == root and Path(cast(str, m.__file__)).suffix == '.py'}
    result = scientific_receipt(args.root_lemma,args.state_npz,args.descriptor,args.guard,args.soft_profile)
    with zipfile.ZipFile(args.source_archive,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in sorted(sources.items()):
            archive.writestr(name,body)
    result['source_sha256'] = {name:hashlib.sha256(body).hexdigest() for name,body in sorted(sources.items())}
    result['source_archive_sha256'] = hashlib.sha256(args.source_archive.read_bytes()).hexdigest()
    result['loaded_sources_unchanged'] = all((root/name).read_bytes()==body for name,body in sources.items())
    result['inputs'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (args.root_lemma,args.state_npz,args.descriptor,args.guard,args.soft_profile)}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

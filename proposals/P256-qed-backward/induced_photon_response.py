#!/usr/bin/env python3
"""NEW no-bare-magnetic, N charged-sea / primitive electric-inertia branch.

This is the selected scalar Phi mean-field band, not the original full Root law.
The quadratic collective (large-N/RPA) quantum dynamics is specified here; its
microscopic finite-N and fluctuating-Phi validity is not silently presumed.
All spatial vertices are extracted from ImprovedPhiLaw.paths, including length2.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, cast

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
from improved_phi_band import ImprovedPhiLaw, LIGHT_PARAMETERS, TRIAL_G0, TRIAL_GS
from rotation_band import RotationBand
from sea_selection import SELECTED

ROOT = Path(__file__).resolve().parent
LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ResponseConfig:
    a: float = 1.
    m: float = .2
    r: float = .12
    v: float = float(LIGHT_PARAMETERS['v'])
    g0: float = TRIAL_G0
    gs: float = TRIAL_GS
    flavors: int = 64
    inertia: float = 1.
    orders: tuple[int, ...] = (4, 6, 8)
    q_values: tuple[float, ...] = (.05, .1, .2)
    omega_values: tuple[float, ...] = (.02, .04, .08)
    species: tuple[tuple[int, float], ...] = ()

    def __post_init__(self):
        if min(self.a, self.m, self.r, self.v, self.g0, self.gs, self.inertia) <= 0:
            raise ValueError('Positive microscopic scales required')
        if self.flavors < 1 or not self.orders or min(self.orders) < 2:
            raise ValueError('Positive flavor count and quadrature orders >=2 required')
        if self.species:
            if sum(count for count, _ in self.species) != self.flavors:
                raise ValueError('Species multiplicities must sum to total charged flavors')
            if any(count < 1 or mass <= 0 for count, mass in self.species):
                raise ValueError('Positive integer multiplicities and masses required')
            if any(int(count) != count for count, _ in self.species):
                raise ValueError('Flavor multiplicities must be integers')
            if self.species[0][1] != self.m:
                raise ValueError('First species is the active filled flavor with config.m')
            if any(mass < self.m for _, mass in self.species):
                raise ValueError('This branch requires auxiliaries no lighter than the active species')


def make_band(config: ResponseConfig) -> tuple[RotationBand, np.ndarray, ImprovedPhiLaw]:
    band = RotationBand(a=config.a, m=config.m, r=config.r, v=config.v, e=1., improved=True)
    frame = np.sqrt([config.g0, config.gs, config.gs, config.gs])[:, None, None] * SELECTED
    return band, frame, ImprovedPhiLaw(3, band)


class SpectralResponse:
    """Full BZ filled lower doublet, empty upper doublet, exact path vertices.

    theta=a*p; A_link=a*A_phys. The measure is d^3theta/(2pi)^3/a^3.
    Thus physical spatial kernel is N*K_link/a; density is N*chi_link/a^3.
    No sampled K(0,0) is subtracted. Bulk zero follows periodic integration.
    """
    def __init__(self, config: ResponseConfig, order: int):
        self.config = config
        self.band, self.frame, self.law = make_band(config)
        self.paths = [(int(np.flatnonzero(d)[0]), int(np.max(d)),
                       np.einsum('m,mij->ij', s, self.frame)) for d, s in self.law.paths]
        self.onsite_coefficients = np.einsum(
            'mij,ji->m', SELECTED, self.law.onsite*self.frame[0]).real/4
        self.path_coefficients = []
        for axis, length, matrix in self.paths:
            coefficients = np.einsum('mij,ji->m', SELECTED, matrix)/4
            reconstructed = np.einsum('m,mij->ij', coefficients, SELECTED)
            if np.max(np.abs(matrix-reconstructed)) > 1e-12*max(1., np.max(np.abs(matrix))):
                raise ArithmeticError('Actual Peierls path is outside selected-Phi scalar span')
            self.path_coefficients.append((axis, length, coefficients))
        onsite = self.law.onsite*self.frame[0]
        if np.max(np.abs(onsite-np.einsum('m,mij->ij', self.onsite_coefficients, SELECTED))) > 1e-12:
            raise ArithmeticError('Actual onsite term is outside selected-Phi scalar span')
        # Panels resolve the physical gap at every a rather than copying a=1 data.
        scale = min(.4, config.a * config.m * math.sqrt(config.g0) /
                    (config.v * math.sqrt(config.gs)))
        boundaries = [0., scale]
        while boundaries[-1] < math.pi:
            boundaries.append(min(math.pi, 2 * boundaries[-1]))
        self.gauss_nodes, self.gauss_weights = leggauss(order)
        self.boxes = [(np.zeros(3), np.full(3, scale))]
        # Seven boxes tile each positive-octant cubical shell exactly.
        # Avoid a tensor product of every tiny-axis panel with every other panel.
        for inner, outer in zip(boundaries[1:-1], boundaries[2:]):
            for flags in itertools.product((0, 1), repeat=3):
                if any(flags):
                    self.boxes.append((np.asarray([inner if flag else 0. for flag in flags]),
                                       np.asarray([outer if flag else inner for flag in flags])))
        self.signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
        self.order = order
        self.species_models = [
            (count, SpectralResponse(replace(config, m=mass, flavors=count, species=()), order))
            for count, mass in config.species]
        self.quadrature_points = (sum(child.quadrature_points for _, child in self.species_models)
                                  if self.species_models else 8*order**3*len(self.boxes))

    def chunks(self):
        for lo, hi in self.boxes:
            coordinates = [(lo[j]+hi[j])/2+(hi[j]-lo[j])*self.gauss_nodes/2 for j in range(3)]
            axis_weights = [(hi[j]-lo[j])*self.gauss_weights/(4*math.pi) for j in range(3)]
            local = np.stack(np.meshgrid(*coordinates, indexing='ij'), axis=-1).reshape(-1, 3)
            weight = np.prod(np.stack(np.meshgrid(*axis_weights, indexing='ij'), axis=-1), axis=-1).reshape(-1)
            yield (local[None, :, :]*self.signs[:, None, :]).reshape(-1, 3), np.tile(weight, 8)

    def h(self, k: np.ndarray) -> np.ndarray:
        h = np.broadcast_to(self.law.onsite*self.frame[0], (len(k), 4, 4)).copy()
        for axis, length, t in self.paths:
            phase = np.exp(1j*length*k[:, axis])[:, None, None]
            h += phase*t + phase.conj()*t.conj().T
        return h

    def projectors(self, k: np.ndarray):
        h = self.h(k)
        e = np.sqrt(np.einsum('bij,bji->b', h, h).real/4)
        return e, (np.eye(4)-h/e[:, None, None])/2, (np.eye(4)+h/e[:, None, None])/2

    def vertices(self, k: np.ndarray, qtheta: np.ndarray):
        j = np.zeros((len(k), 3, 4, 4), complex)
        c = np.zeros_like(j)
        for axis, length, t in self.paths:
            # Literal sum over every link in the path, midpoint-link convention.
            s = sum(np.exp(1j*qtheta[axis]*(ell-(length-1)/2)) for ell in range(length)).real
            phase = np.exp(1j*length*k[:, axis])[:, None, None]
            j[:, axis] += 1j*s*(phase*t-phase.conj()*t.conj().T)
            c[:, axis] -= s*s*(phase*t+phase.conj()*t.conj().T)
        return j, c

    def coefficient_fields(self, k: np.ndarray, qtheta: np.ndarray):
        """Exact coefficients of the actual path matrices; no band approximation."""
        h0 = np.broadcast_to(self.onsite_coefficients, (len(k), 4)).copy()
        hm, hp = h0.copy(), h0.copy()
        j0 = np.zeros((len(k), 3, 4))
        j = np.zeros_like(j0)
        contact = np.zeros_like(j0)
        contact_difference = np.zeros_like(j0)
        qhat = 2*np.sin(qtheta/2)
        for axis, length, coefficients in self.path_coefficients:
            value = np.exp(1j*length*k[:, axis])[:, None]*coefficients
            shift = np.exp(1j*length*qtheta[axis]/2)
            path_sum = sum(np.exp(1j*qtheta[axis]*(ell-(length-1)/2))
                           for ell in range(length)).real
            real = 2*value.real
            current = -2*value.imag
            h0 += real
            hm += 2*(value*shift.conjugate()).real
            hp += 2*(value*shift).real
            j0[:, axis] += length*current
            j[:, axis] += path_sum*current
            contact[:, axis] -= path_sum**2*real
            if length == 2:
                contact_difference[:, axis] += qhat[axis]**2*real
            elif length != 1:
                contact_difference[:, axis] -= (path_sum**2-length**2)*real
        return hm, hp, h0, j, j0, contact, contact_difference

    def matrix_probe(self, k, qtheta, em, ep, jc, contact, g, density, mixed):
        """Independent literal matrix-eigh/projector checks in every BZ shell."""
        indices = np.linspace(0, len(k)-1, 8, dtype=int)
        probe = k[indices]
        projector_errors = []
        def literal_projectors(momentum):
            eigenvalues, vectors = np.linalg.eigh(self.h(momentum))
            lower, upper = vectors[:, :, :2], vectors[:, :, 2:]
            filled = lower@lower.conj().transpose(0, 2, 1)
            empty = upper@upper.conj().transpose(0, 2, 1)
            energy, analytic_filled, analytic_empty = self.projectors(momentum)
            projector_errors.extend((float(np.max(np.abs(filled-analytic_filled))),
                                     float(np.max(np.abs(empty-analytic_empty)))))
            expected = energy[:, None]*np.array([-1., -1., 1., 1.])
            if np.max(np.abs(eigenvalues-expected)) > 2e-11*max(1., np.max(energy)):
                raise ArithmeticError('Literal matrix eigenvalues disagree with selected-Phi band')
            return eigenvalues[:, 2:].mean(axis=1), filled, empty
        exact_em, pm, um = literal_projectors(probe-qtheta/2)
        exact_ep, pp, up = literal_projectors(probe+qtheta/2)
        _, p0, _ = literal_projectors(probe)
        exact_j, exact_c = self.vertices(probe, qtheta)
        exact_g = np.empty((len(indices), 3, 3))
        for i in range(3):
            for j in range(3):
                exact_g[:, i, j] = np.trace(
                    pm@exact_j[:, i]@up@exact_j[:, j]+
                    um@exact_j[:, j]@pp@exact_j[:, i], axis1=1, axis2=2).real
        exact_density = np.trace(pm@up+um@pp, axis1=1, axis2=2).real
        exact_mixed = np.stack([np.trace(
            pm@up@exact_j[:, i]-um@pp@exact_j[:, i], axis1=1, axis2=2).real
            for i in range(3)], axis=1)
        exact_contact = np.einsum('bij,baji->ba', p0, exact_c).real
        comparisons = dict(
            energy=(np.stack((em[indices], ep[indices])), np.stack((exact_em, exact_ep))),
            current=(np.einsum('bam,mij->baij', jc[indices], SELECTED), exact_j),
            contact=(contact[indices], exact_contact),
            spectral=(g[indices], exact_g), density=(density[indices], exact_density),
            mixed=(mixed[indices], exact_mixed))
        errors = dict(projector=max(projector_errors))
        if errors['projector'] > 2e-11:
            raise ArithmeticError('Literal matrix-eigh filled projectors disagree with Clifford form')
        for name, (coefficients, matrices) in comparisons.items():
            error = float(np.max(np.abs(coefficients-matrices)))
            if error > 2e-11*max(1., float(np.max(np.abs(matrices)))):
                raise ArithmeticError(f'Actual matrix/coefficient {name} disagreement: {error}')
            errors[name] = error
        return errors

    def spectral(self, q: np.ndarray) -> dict[str, Any]:
        if self.species_models:
            return self.mixed_spectral(q)
        qtheta = self.config.a*np.asarray(q, float)
        qhat = 2*np.sin(qtheta/2)
        deltas, spectral_weights, density_weights, mixed_weights = [], [], [], []
        contact = np.zeros(3)
        stable_static = np.zeros((3, 3))
        vertex_error = 0.
        wrong_vertex_error = 0.
        matrix_errors: dict[str, float] = {}
        matrix_points = 0
        for k, w in self.chunks():
            hm, hp, h0, jc, j0c, cc, cdifference = self.coefficient_fields(k, qtheta)
            em, ep, e0 = (np.linalg.norm(h, axis=1) for h in (hm, hp, h0))
            nm, np_, n0 = hm/em[:, None], hp/ep[:, None], h0/e0[:, None]
            contact_density = -2*np.einsum('bm,bam->ba', n0, cc)
            contact += np.einsum('ba,b->a', contact_density, w)
            difference = hp-hm
            vertex_error = max(vertex_error, float(np.max(np.linalg.norm(
                np.einsum('a,bam->bm', qhat, jc)-difference, axis=1))))
            wrong_vertex_error = max(wrong_vertex_error, float(np.max(np.linalg.norm(
                np.einsum('a,bam->bm', qhat, j0c)-difference, axis=1))))
            # Exact Clifford trace, retaining both interband clusters.
            dot = np.einsum('bi,bi->b', nm, np_)
            jj = np.einsum('bim,bjm->bij', jc, jc)
            mj = np.einsum('bm,bim->bi', nm, jc)
            pj = np.einsum('bm,bim->bi', np_, jc)
            g = 2*((1+dot)[:, None, None]*jj-
                   mj[:, :, None]*pj[:, None, :]-pj[:, :, None]*mj[:, None, :])
            density = np.sum((np_-nm)**2, axis=1)
            mixed_density = 2*np.einsum('bm,bim->bi', np_-nm, jc)
            errors = self.matrix_probe(k, qtheta, em, ep, jc, contact_density,
                                       g, density, mixed_density)
            matrix_points += 8
            for name, error in errors.items():
                matrix_errors[name] = max(matrix_errors.get(name, 0.), error)
            # Bulk periodicity proves integral C(0)=integral B(0), before quadrature.
            # Integrate B(0)-B(q)+[C(q)-C(0)] pointwise to resolve small q^2.
            # Keep the independent raw C(q)-B(q) and its alias in the capsule.
            n0j = np.einsum('bm,bim->bi', n0, j0c)
            g0 = 4*(np.einsum('bim,bjm->bij', j0c, j0c)-n0j[:, :, None]*n0j[:, None, :])
            stable_static += np.diag(-2*np.einsum('bm,bam,b->a', n0, cdifference, w))
            stable_static += np.einsum('bij,b->ij',
                g0/(2*e0[:, None, None])-g/(em+ep)[:, None, None], w)
            deltas.append(em+ep)
            spectral_weights.append(g*w[:, None, None])
            density_weights.append(density*w)
            # Signed forward-minus-reverse density/current trace, not a guessed gamma endpoint.
            mixed_weights.append(mixed_density*w[:, None])
        delta = np.concatenate(deltas)
        weights = np.concatenate(spectral_weights)
        density = np.concatenate(density_weights)
        mixed = np.concatenate(mixed_weights)
        bubble = np.einsum('bij,b->ij', weights, 1/delta).real
        static_raw = np.diag(contact)-bubble
        static = stable_static if np.linalg.norm(qtheta) > 0 else static_raw
        eps = np.einsum('bij,b->ij', weights, 1/delta**3).real
        chi = float(np.sum(density/delta))
        longitudinal = float(qhat@static@qhat)
        spectral_ward = float(np.max(np.abs(np.einsum('i,bij,j->b', qhat, weights, qhat)-delta**2*density)))
        mixed_ward = float(np.max(np.abs(mixed@qhat-delta*density)))
        return dict(q=np.asarray(q).tolist(), qtheta=qtheta.tolist(), qhat=qhat, delta=delta,
                    weights=weights, density=density, mixed_weights=np.real(mixed), contact=contact, bubble=bubble,
                    static=static, static_raw=static_raw, epsilon_link=eps, chi=chi, vertex_error=vertex_error,
                    wrong_vertex_error=wrong_vertex_error, spectral_ward=spectral_ward,
                    mixed_ward=mixed_ward, longitudinal_static=longitudinal,
                    backend=dict(kind='Exact actual-path selected-Phi coefficient algebra',
                                 matrix_probe_points=matrix_points, matrix_errors=matrix_errors,
                                 vertex_error_kind='Selected-Phi Clifford operator norm bound'))

    def mixed_spectral(self, q: np.ndarray) -> dict[str, Any]:
        """Add actual mass-resolved filled-sea spectral measures, not fitted endpoints."""
        parts = [(count/self.config.flavors, model.spectral(q))
                 for count, model in self.species_models]
        delta = np.concatenate([data['delta'] for _, data in parts])
        weights = np.concatenate([fraction*data['weights'] for fraction, data in parts])
        density = np.concatenate([fraction*data['density'] for fraction, data in parts])
        mixed = np.concatenate([fraction*data['mixed_weights'] for fraction, data in parts])
        contact = sum(fraction*data['contact'] for fraction, data in parts)
        bubble = np.einsum('bij,b->ij', weights, 1/delta)
        static = sum(fraction*data['static'] for fraction, data in parts)
        static_raw = np.diag(contact)-bubble
        qhat = parts[0][1]['qhat']
        return dict(q=np.asarray(q).tolist(), qtheta=(self.config.a*np.asarray(q)).tolist(),
                    qhat=qhat, delta=delta, weights=weights, density=density,
                    mixed_weights=mixed, contact=contact, bubble=bubble, static=static,
                    static_raw=static_raw,
                    epsilon_link=np.einsum('bij,b->ij', weights, 1/delta**3),
                    chi=float(np.sum(density/delta)),
                    vertex_error=max(data['vertex_error'] for _, data in parts),
                    wrong_vertex_error=max(data['wrong_vertex_error'] for _, data in parts),
                    spectral_ward=max(data['spectral_ward'] for _, data in parts),
                    mixed_ward=max(data['mixed_ward'] for _, data in parts),
                    longitudinal_static=float(qhat@static@qhat),
                    backend=dict(kind='Exact actual-path selected-Phi coefficient algebra',
                                 species=[data['backend'] for _, data in parts]),
                    species=[dict(count=count, m=model.config.m,
                                  gap=model.config.m*math.sqrt(model.config.g0),
                                  quadrature_points=model.quadrature_points,
                                  electric_trace_per_flavor=float(np.trace(data['epsilon_link'])/(3*self.config.a)),
                                  static_link=data['static'].tolist())
                             for (count, model), (_, data) in zip(self.species_models, parts)])

    def mode(self, q: np.ndarray, *, velocity: bool = False,
             data: dict[str, Any] | None = None) -> dict[str, Any]:
        data = self.spectral(q) if data is None else data
        cfg = self.config
        qhat = data['qhat']
        norm = float(np.linalg.norm(qhat))
        if norm == 0:
            raise ValueError('Nonzero q required for a propagating mode')
        direction = qhat/norm
        _, _, vh = np.linalg.svd(direction[None, :])
        transverse = vh[1:].T
        kt = transverse.T@data['static']@transverse
        kvalues = np.linalg.eigvalsh(kt)
        threshold = float(np.min(data['delta']))
        factor = cfg.flavors/cfg.a
        bare = 1/(2*cfg.inertia)
        wt = np.einsum('ia,bij,jc->bac', transverse, data['weights'], transverse).real
        mt = np.einsum('bi,ia->ba', data['mixed_weights'], transverse)
        delta2 = data['delta']**2
        density_delta = data['density']*data['delta']
        wt_over_delta = wt/data['delta'][:, None, None]
        qphysical2 = (norm/cfg.a)**2
        def scalar_constraint(inverse_denominator, *, derivative=False):
            chi = bare*qphysical2+cfg.flavors/cfg.a**3*np.einsum(
                'b,b->', density_delta, inverse_denominator)
            d = cfg.flavors/cfg.a**2*np.einsum('bi,b->i', mt, inverse_denominator)
            chiprime, dprime = 0., np.zeros(2)
            if derivative:
                squared = inverse_denominator**2
                chiprime = cfg.flavors/cfg.a**3*np.einsum('b,b->', density_delta, squared)
                dprime = cfg.flavors/cfg.a**2*np.einsum('bi,b->i', mt, squared)
            return chi, d, chiprime, dprime
        if np.min(kvalues) <= 0:
            raise ArithmeticError(f'Negative/unresolved transverse stiffness {kvalues}; order={self.order}, a={cfg.a}')
        def inverse_matrix(x):
            inverse_denominator = 1/(delta2-x)
            chi, d, _, _ = scalar_constraint(inverse_denominator)
            return bare*x*np.eye(2)-factor*kt + factor*x*np.einsum(
                'bij,b->ij', wt_over_delta, inverse_denominator)-x*np.outer(d, d)/chi
        stiffness_scale = factor*float(np.max(kvalues))
        root_scale = stiffness_scale/(bare+factor*float(np.trace(data['epsilon_link']))/3)
        upper_scaled = threshold**2*(1-1e-10)/root_scale
        poles = []
        for i in range(2):
            def inverse_scaled(y):
                return float(np.linalg.eigvalsh(inverse_matrix(y*root_scale))[i])/stiffness_scale
            scaled_root = cast(float, brentq(inverse_scaled, 0., upper_scaled,
                xtol=1e-13*min(1., upper_scaled), rtol=np.float64(1e-13), full_output=False))
            root = scaled_root*root_scale
            _, eigenvectors = np.linalg.eigh(inverse_matrix(root))
            tv = eigenvectors[:, i]
            vector = transverse@tv
            g = np.einsum('i,bij,j->b', tv, wt, tv)
            static = factor*float(tv@kt@tv)
            chi, d, chiprime, dprime = scalar_constraint(1/(delta2-root), derivative=True)
            dv, dpv = float(tv@d), float(tv@dprime)
            constraint_derivative = dv**2/chi+root*(2*dv*dpv/chi-dv**2*chiprime/chi**2)
            derivative = bare + factor*np.sum(g*data['delta']/(data['delta']**2-root)**2)-constraint_derivative
            chi0, d0, _, _ = scalar_constraint(1/delta2)
            eps = bare+factor*np.sum(g/data['delta']**3)-float(tv@d0)**2/chi0
            poles.append(dict(omega=float(math.sqrt(root)), residue=float(1/derivative),
                              local_omega=float(math.sqrt(static/eps)), epsilon=float(eps),
                              scalar_polarization=float(math.sqrt(root)*dv/chi),
                              kappa=float(static/(norm/cfg.a)**2),
                              polarization=vector.tolist(),
                              inverse_residual=float(np.linalg.eigvalsh(inverse_matrix(root))[i])))
        poles.sort(key=lambda p: p['omega'])
        result = dict(q=np.asarray(q).tolist(), omega=float(np.mean([p['omega'] for p in poles])),
                      residue=float(np.mean([p['residue'] for p in poles])),
                      pair_threshold=threshold, two_polarizations=poles,
                      pair_threshold_kind='Numerical minimum of exact interband gaps on the reported 3D quadrature',
                      pair_threshold_lower=2*cfg.m*math.sqrt(cfg.g0),
                      polarization_speed_splitting=float(poles[1]['omega']/poles[0]['omega']-1),
                      quadrature_order=self.order, quadrature_points=self.quadrature_points)
        if velocity:
            step = max(float(np.linalg.norm(q))*1e-3, 1e-5)
            gradients = np.zeros((2, 3))
            for axis in range(3):
                dq = np.eye(3)[axis]*step
                upper = self.mode(np.asarray(q)+dq)
                lower = self.mode(np.asarray(q)-dq)
                for i in range(2):
                    gradients[i, axis] = (upper['two_polarizations'][i]['omega']-
                        lower['two_polarizations'][i]['omega'])/(2*step)
            result['group_velocity'] = np.mean(gradients, axis=0).tolist()
            for i, pole in enumerate(poles):
                pole['group_velocity'] = gradients[i].tolist()
        return result


def finite_mode(config: ResponseConfig, q: np.ndarray, *, velocity: bool = True) -> dict[str, Any]:
    """Finite-q Coulomb-constrained RPA pole, residue and optional gradient."""
    return SpectralResponse(config, config.orders[-1]).mode(np.asarray(q), velocity=velocity)


def compact(data: dict[str, Any], config: ResponseConfig) -> dict[str, Any]:
    c = config
    qphysical2 = float(data['qhat']@data['qhat'])/c.a**2
    epsilon_longitudinal = (1/(2*c.inertia)+c.flavors*data['chi']/c.a**3/qphysical2
                            if qphysical2 > 0 else None)
    return dict(q=data['q'], qtheta=data['qtheta'], contact_link=data['contact'].tolist(),
                paramagnetic_static_link=data['bubble'].tolist(), static_link=data['static'].tolist(),
                static_raw_link=data['static_raw'].tolist(),
                identity_vs_raw_contact_difference_link=(data['static']-data['static_raw']).tolist(),
                static_evaluation='At nonzero q, exact bulk periodic identity: integral B(0)-B(q)+C(q)-C(0); raw unsubtracted contact-minus-poles retained',
                epsilon_link=data['epsilon_link'].tolist(), density_static_link=data['chi'],
                density_static_physical=c.flavors*data['chi']/c.a**3,
                epsilon_longitudinal_static=epsilon_longitudinal,
                vertex_Ward_error=data['vertex_error'], spectral_density_Ward_error=data['spectral_ward'],
                mixed_density_current_Ward_error=data['mixed_ward'],
                pure_gradient_static_form=data['longitudinal_static'],
                pure_gradient_static_raw_form=float(data['qhat']@data['static_raw']@data['qhat']),
                wrong_short_path_vertex_error=data['wrong_vertex_error'],
                backend=data['backend'],
                species=data.get('species', [dict(count=config.flavors, m=config.m,
                    gap=config.m*math.sqrt(config.g0))]),
                pair_threshold=float(np.min(data['delta'])))


def native_path_proof(config: ResponseConfig) -> dict[str, float]:
    model = SpectralResponse(config, 2)
    law = model.law
    fields = np.tile(model.frame, (law.volume, 1, 1, 1))
    links = np.tile(np.eye(4), (law.volume, 3, 1, 1))
    a = np.zeros((law.volume, 3))
    q = np.asarray([2*math.pi/3, 0., 0.])
    pin = np.asarray([0., 2*math.pi/3, 0.])
    pout = pin+q
    variation = np.zeros_like(a)
    variation[:, 0] = np.cos(law.sites@q+q[0]/2)
    vin = np.kron(np.exp(1j*law.sites@pin)[:, None], np.eye(4))/math.sqrt(law.volume)
    vout = np.kron(np.exp(1j*law.sites@pout)[:, None], np.eye(4))/math.sqrt(law.volume)
    j, _ = model.vertices(((pin+pout)/2)[None, :], q)
    _, ck = model.vertices(pin[None, :], q)
    native_j = vout.conj().T@law.onebody(fields, links, a, variation)@vin
    native_c = vin.conj().T@law.onebody(fields, links, a, variation, variation)@vin
    eps = 1e-5
    finite = (law.onebody(fields, links, a+eps*variation)-
              law.onebody(fields, links, a-eps*variation))/(2*eps)
    return dict(current_Bloch_error=float(np.linalg.norm(native_j-j[0, 0]/2)),
                contact_Bloch_error=float(np.linalg.norm(native_c-ck[0, 0]/2)),
                actual_derivative_error=float(np.linalg.norm(finite-law.onebody(fields, links, a, variation))))


def oscillator_proof(model: SpectralResponse, q: np.ndarray) -> dict[str, Any]:
    """Positive canonical Hamiltonian using actual BZ spectral weights.

    Gap-bin compression is confined to this independent symplectic discriminator;
    the exported photon poles always use the uncompressed spectral sum.
    """
    data = model.spectral(q)
    cfg = model.config
    weights = cfg.flavors/cfg.a*data['weights'][:, 1, 1].real
    stiffness = cfg.flavors/cfg.a*data['static'][1, 1]
    exact = model.mode(q)['two_polarizations'][0]['omega']
    rows = []
    for bins in (16, 32, 64):
        edges = np.geomspace(np.min(data['delta'])*(1-1e-12),
                             np.max(data['delta'])*(1+1e-12), bins+1)
        index = np.clip(np.searchsorted(edges, data['delta'])-1, 0, bins-1)
        g = np.bincount(index, weights=weights, minlength=bins)
        numerator = np.bincount(index, weights=weights/data['delta']**3, minlength=bins)
        keep = g > 1e-20
        # Match the exact electric moment in each gap bin, not a fitted pole.
        delta = (g[keep]/numerator[keep])**(1/3)
        s = np.sqrt(g[keep])/delta
        mass = np.zeros((len(delta)+1, len(delta)+1))
        shift = np.concatenate(([1.], s))
        mass += 2*cfg.inertia*np.outer(shift, shift)
        mass[1:, 1:] += np.diag(delta)
        position = np.diag(np.concatenate(([stiffness], delta)))
        rootposition = np.diag(np.sqrt(np.diag(position)))
        eigenvalues, vectors = np.linalg.eigh(rootposition@mass@rootposition)
        omega = math.sqrt(float(eigenvalues[0]))
        residue = float(eigenvalues[0]*vectors[0, 0]**2/stiffness)
        rows.append(dict(bins=bins, omega=omega, residue=residue,
                         relative_full_spectral_error=abs(omega/exact-1),
                         positive_Hessian_min=float(np.min(eigenvalues))))
    missing_contact = data['static'][1, 1]-data['contact'][1]
    missing_poles = data['contact'][1]
    return dict(q=q.tolist(), canonical='H=1/(2eps_b)(P+s.p)^2+sum Delta(x^2+p^2)/2+K A^2/2',
                full_spectral_omega=exact, compressed_symplectic=rows,
                identity_vs_raw_contact_difference_link=float(data['static'][1, 1]-data['static_raw'][1, 1]),
                quantum_scope='Canonical oscillator discretizes the continuous-BZ identity law; differs from finite-node raw contact by reported alias, not an exact finite-box filled sea',
                missing_contact_static=float(missing_contact),
                missing_poles_static=float(missing_poles),
                missing_contact_positive_stiffness=bool(missing_contact > 0),
                missing_poles_nonzero_static_defect=bool(abs(missing_poles) > 1e-12))


def mean_field_remainder(config: ResponseConfig) -> dict[str, Any]:
    """Deterministic Cartesian Phi ball; NOT a quantum-vacuum concentration claim."""
    if config.species:
        parts = [mean_field_remainder(replace(config, m=mass, flavors=count, species=()))
                 for count, mass in config.species]
        result = dict(parts[0])
        for key in ('epsilon_physical_error_upper', 'static_kernel_physical_error_upper',
                    'kappa_at_smallest_q_error_upper'):
            result[key] = sum(part[key] for part in parts)
        for key in ('onebody_operator_error', 'current_operator_error', 'contact_operator_error',
                    'projector_operator_error'):
            result[key] = max(part[key] for part in parts)
        result['remaining_interband_gap'] = min(part['remaining_interband_gap'] for part in parts)
        result['species_bounds'] = parts
        return result
    _, frame, law = make_band(config)
    eta = 1e-12*config.a
    scalars = [(int(np.max(d)), float(np.sum(np.abs(s)))) for d, s in law.paths]
    h_lipschitz = abs(law.onsite)+2*sum(s for _, s in scalars)
    j_lipschitz = 2*sum(length*s for length, s in scalars)
    c_lipschitz = 2*sum(length**2*s for length, s in scalars)
    delta_h, delta_j, delta_c = eta*np.asarray([h_lipschitz, j_lipschitz, c_lipschitz])
    mass = config.m*math.sqrt(config.g0)
    gap = 2*(mass-delta_h)
    projector = delta_h/(mass-delta_h)
    jmax = 2*sum(length*np.linalg.norm(np.einsum('m,mij->ij', s, frame), 2)
                 for d, s in law.paths for length in [int(np.max(d))])
    cmax = 2*sum(length**2*np.linalg.norm(np.einsum('m,mij->ij', s, frame), 2)
                 for d, s in law.paths for length in [int(np.max(d))])
    trace_error = 16*projector*jmax**2+16*jmax*delta_j+8*delta_j**2
    bubble_error = trace_error/gap+8*jmax**2*2*delta_h/gap**2
    contact_error = 4*projector*cmax+4*delta_c
    electric_error = trace_error/gap**3+8*jmax**2*6*delta_h/gap**4
    return dict(Phi_matrix_radius=eta, law='Each Cartesian endpoint matrix operator-norm perturbation <=eta',
                onebody_operator_error=float(delta_h), current_operator_error=float(delta_j),
                contact_operator_error=float(delta_c), remaining_interband_gap=float(gap),
                projector_operator_error=float(projector),
                epsilon_physical_error_upper=float(config.flavors/config.a*electric_error),
                static_kernel_physical_error_upper=float(config.flavors/config.a*(contact_error+bubble_error)),
                kappa_at_smallest_q_error_upper=float(config.flavors/config.a*(contact_error+bubble_error)/
                    (2*math.sin(config.a*config.q_values[0]/2)/config.a)**2),
                assumptions='Gap-preserving deterministic uniform Phi ball. No original Root-vacuum concentration claim; no old volume bounds copied.')


def source_hashes() -> dict[str, str]:
    names = ('induced_photon_response.py', 'improved_phi_band.py', 'rotation_band.py',
             'local_orbital_stability.py', 'sea_selection.py', 'bosonic_original_law_join.py',
             'reference_photon_current_scattering.py')
    result = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    result['induced_photon_response.py'] = LOADED_SOURCE_SHA256
    return result


def compute_response(config: ResponseConfig | None = None) -> dict[str, Any]:
    cfg = config or ResponseConfig()
    raw = []
    final_zero = None
    final_model = None
    for order in cfg.orders:
        model = SpectralResponse(cfg, order)
        zero = model.spectral(np.zeros(3))
        rows = []
        for q in cfg.q_values:
            data = model.spectral(np.asarray([q, 0., 0.]))
            row = compact(data, cfg)
            row['modes'] = model.mode(np.asarray([q, 0., 0.]), data=data)
            row['dynamic'] = []
            for omega in cfg.omega_values:
                response = np.einsum('bij,b->ij', data['weights'],
                    data['delta']/(data['delta']**2+omega**2)).real
                density = float(np.sum(data['density']*data['delta']/(data['delta']**2+omega**2)))
                k = np.diag(data['contact'])-response
                qhat = data['qhat']
                kidentity = data['static']+data['bubble']-response
                row['dynamic'].append(dict(omega_Euclidean=omega, kernel_diagonal_link=np.diag(k).tolist(),
                    kernel_identity_diagonal_link=np.diag(kidentity).tolist(),
                    density_link=density, temporal_Ward_defect=float(qhat@k@qhat-omega**2*density),
                    temporal_Ward_identity_defect=float(qhat@kidentity@qhat-omega**2*density)))
            rows.append(row)
        eps_dynamic = []
        g0 = np.real(np.diagonal(zero['weights'], axis1=1, axis2=2))
        for omega in cfg.omega_values:
            value = np.sum(g0/(zero['delta'][:, None]*(zero['delta'][:, None]**2+omega**2)), axis=0)
            eps_dynamic.append(dict(omega_Euclidean=omega, epsilon_sea_physical=(value/cfg.a).tolist()))
        raw.append(dict(order=order, BZ_points=model.quadrature_points,
                        quadrature='Gauss tensor cells on exact dyadic cubical shells, all eight octants',
                        zero=compact(zero, cfg), zero_frequency_grid=eps_dynamic, finite_q=rows))
        final_zero, final_model = zero, model
    assert final_zero is not None and final_model is not None
    epsilon_sea = float(np.trace(final_zero['epsilon_link'])/(3*cfg.a))
    # No intercept subtraction: fit K_T/qhat_phys^2 directly, then extrapolate q^2.
    q2 = np.asarray([(2*math.sin(cfg.a*q/2)/cfg.a)**2 for q in cfg.q_values])
    kvals = np.asarray([np.mean(np.diag(np.asarray(row['static_link']))[1:])*cfg.a /
                       (2*math.sin(cfg.a*row['q'][0]/2))**2 for row in raw[-1]['finite_q']])
    kappa_sea = float(np.polynomial.polynomial.polyfit(q2, kvals, min(2, len(q2)-1))[0])
    epsilon = 1/(2*cfg.inertia)+cfg.flavors*epsilon_sea
    kappa = cfg.flavors*kappa_sea
    if epsilon <= 0 or kappa <= 0:
        raise ArithmeticError('Unstable induced branch: epsilon or kappa nonpositive')
    speed = math.sqrt(kappa/epsilon)
    mass = cfg.m*math.sqrt(cfg.g0)
    matter_speed = cfg.v*math.sqrt(cfg.gs)
    aliases = [float(np.max(np.abs(np.asarray(r['zero']['static_link'])))) for r in raw]
    lastq = raw[-1]['finite_q'][0]
    controls = dict(missing_contact_static_defect_link=max(abs(x) for x in np.diag(final_zero['bubble'])),
                    missing_poles_static_defect_link=max(abs(x) for x in final_zero['contact']),
                    half_poles_static_defect_link=float(np.max(np.abs(np.diag(final_zero['contact'])-.5*final_zero['bubble']))),
                    constant_A_alias_by_order=aliases,
                    actual_vertex_Ward_max=max(row['vertex_Ward_error'] for row in raw[-1]['finite_q']),
                    wrong_vertex_max=max(row['wrong_short_path_vertex_error'] for row in raw[-1]['finite_q']),
                    density_mixed_Ward_max=max(row['mixed_density_current_Ward_error'] for row in raw[-1]['finite_q']),
                    pole_remainder_at_smallest_q=abs(lastq['modes']['two_polarizations'][0]['omega']/lastq['modes']['two_polarizations'][0]['local_omega']-1))
    probe = final_model.spectral(np.asarray([.07, .05, .03]))
    controls['oblique_q'] = compact(probe, cfg)
    controls['native_path'] = native_path_proof(cfg)
    controls['canonical_oscillator'] = oscillator_proof(final_model, np.asarray([cfg.q_values[0], 0., 0.]))
    previous = raw[-2] if len(raw) > 1 else raw[-1]
    eps_previous = float(np.mean(np.diag(np.asarray(previous['zero']['epsilon_link'])))/cfg.a)
    convergence = dict(epsilon_relative=abs(epsilon_sea/eps_previous-1))
    previous_kappa = np.mean(np.diag(np.asarray(previous['finite_q'][0]['static_link']))[1:])*cfg.a/(2*math.sin(cfg.a*cfg.q_values[0]/2))**2
    convergence['kappa_smallest_q_relative'] = abs(kvals[0]/previous_kappa-1)
    convergence['constant_A_alias_final'] = aliases[-1]
    hashes = source_hashes()
    inputs = asdict(cfg)
    input_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    omega_check = .4*mass
    species_data = final_zero.get('species', [dict(count=cfg.flavors, m=cfg.m,
        gap=mass, electric_trace_per_flavor=epsilon_sea)])
    decoupling = [dict(count=part['count'], m=part['m'], gap=part['gap'],
        electric_fraction=part['count']*part['electric_trace_per_flavor']/epsilon,
        retarded_relative_frequency_remainder=omega_check**2/(4*part['gap']**2-omega_check**2))
        for part in species_data]
    branch = ('NEW mass-resolved charged seas (one active filled light flavor, massive auxiliaries)'
              if cfg.species else 'NEW equal-mass charged-light seas')
    return dict(schema=1, branch=branch+', no bare magnetic Maxwell, primitive electric rotor, large-N RPA',
        assigned_plan_epoch='03b34911221ba1898315373edaa7c4c3bcac718462e7fe51a842560eb6933f87',
        config=inputs, band=asdict(final_model.band),
        metric=dict(G=np.diag([cfg.g0, cfg.gs, cfg.gs, cfg.gs]).tolist(), g0=cfg.g0, gs=cfg.gs,
                    frame_provenance='selected scalar Phi mean from original heavy full10 Gram; not recalculated new-flavor vacuum'),
        Peierls=dict(onsite=final_model.law.onsite, path_lengths=[length for _, length, _ in final_model.paths],
                     supplier='ImprovedPhiLaw.paths scalar coefficients from RotationBand.couplings, no endpoint Gamma'),
        charged_species=[dict(count=count, m=species_mass, gap=species_mass*math.sqrt(cfg.g0),
            role='Contains tagged active electron/positron' if index == 0 else 'Massive inducing auxiliary')
            for index, (count, species_mass) in enumerate(cfg.species or ((cfg.flavors, cfg.m),))],
        epsilon=epsilon, kappa=kappa, photon_speed=speed, transverse_residue=1/epsilon,
        epsilon_sea=epsilon_sea, kappa_sea=kappa_sea, epsilon_bare=1/(2*cfg.inertia),
        charged_cone=matter_speed, charged_gap=mass, speed_ratio=speed/matter_speed,
        mass_decoupling=dict(omega=omega_check, species=decoupling,
            total_retarded_relative_frequency_remainder_upper=sum(
                part['electric_fraction']*part['retarded_relative_frequency_remainder'] for part in decoupling),
            scope='Heavy production thresholds and low-frequency dynamic decoupling, NOT removal of their static UV loop coefficient or proof of their origin'),
        units=dict(a='physical lattice spacing', momentum='physical inverse length; theta=a*p',
                   omega='energy with hbar=1', A='physical, A_link=a*A',
                   measure='d^3theta/(2pi)^3/a^3', charge=1,
                   inertia='Physical electric-density UE_phys; microscopic link term (inertia/a)*Pi_link^2, epsilon_b=1/(2inertia)',
                   action='1/2 int dt d^3x [epsilon E^2-kappa B^2]',
                   pole='Coulomb-constrained 2x2 F: eps_b*x I-Kstatic+N/a sum W*x/[Delta*(Delta^2-x)]-x*d*d^T/chi',
                   residue='[dF/domega^2]^-1; mode leg sqrt(Z/(2omega))',
                   longitudinal='Coulomb rho(q)rho(-q)/(2 [epsilon_b*qhat_phys^2+N*chi00/a^3])'),
        raw_grids=raw, controls=controls, convergence=convergence,
        validity=dict(quadratic_collective='large-N/RPA only; not exact interacting photon',
                      frequency_gap=2*mass,
                      finite_q_threshold='Numerical exact-band interband gap minima in every raw row/mode; analytic 2M lower bound also returned; no exact continuum threshold claim',
                      frequency_remainder_Euclidean='q0: eps_E=sum g2/[Delta*(Delta^2+omega_E^2)], relative <=omega_E^2/(4M^2)',
                      frequency_remainder_retarded='q0 belowpairgap: relative <=omega^2/(4M^2-omega^2); finite-q numerical spectral denominator evaluated directly, conservative gap lower bound remains 2M',
                      field_remainder='improved mass <=sqrt(g0)*r*a^3*sum p_j^4/4; kinetic <=v*sqrt(gs)*a^2*sum |p_j|^3/6',
                      Cartesian_Phi_remainder=mean_field_remainder(cfg),
                      Root_transfer='No frozen huge-volume/Root theorem transferred to this changed quantum law'),
        assumptions=dict(quantum='CAR charged fields and local canonical tangent oscillators, hbar=1 input. Compact U1 prepared-state completion is a separate microscopic branch, not an admitted stationary photon phase.',
                         flavor='Total N charged species with explicit mass spectrum; tagged excitation in first filled flavor, all N lower doublets filled; no extra species or repeated bubble',
                         dynamics='Physical UE_phys electric density kept, link term UE_phys/a Pi_link^2; bare curl^2 removed. Sea collective oscillator creates stiffness.',
                         vacuum='Filled lower doublets at fixed selected Phi mean; new species backreaction not yet self-consistently admitted',
                         normal_transfer='Original Phi/SU4/source fluctuation bounds do not automatically apply; new-law admission remains coupled debt'),
        source_hashes=hashes, input_sha256=input_hash,
        runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, platform=platform.platform()))


def validate_response(result: dict[str, Any]) -> None:
    controls = result['controls']
    convergence = result['convergence']
    if convergence['epsilon_relative'] > 1e-4 or convergence['kappa_smallest_q_relative'] > 2e-3:
        raise ArithmeticError(f'Unresolved quadrature: {convergence}')
    if controls['actual_vertex_Ward_max'] > 1e-10 or controls['density_mixed_Ward_max'] > 1e-10:
        raise ArithmeticError('Spatial/temporal gauge vertices failed')
    if controls['wrong_vertex_max'] < 100*controls['actual_vertex_Ward_max']:
        raise ArithmeticError('Wrong path control is not discriminating')
    if min(controls['missing_contact_static_defect_link'], controls['missing_poles_static_defect_link'],
           controls['half_poles_static_defect_link']) < 100*convergence['constant_A_alias_final']:
        raise ArithmeticError('Contact/pole controls are not discriminating')
    if max(controls['native_path'].values()) > 1e-7:
        raise ArithmeticError('Actual primitive API/Bloch correspondence failed')
    if controls['canonical_oscillator']['compressed_symplectic'][-1]['relative_full_spectral_error'] > 1e-4:
        raise ArithmeticError('Canonical oscillator/full spectral pole join failed')
    for row in result['raw_grids'][-1]['finite_q']:
        for pole in row['modes']['two_polarizations']:
            if not (pole['residue'] > 0 and 0 < pole['omega'] < row['pair_threshold']):
                raise ArithmeticError('Nonpositive or above-pair-threshold photon mode')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=float, default=1.)
    parser.add_argument('--family', help='Comma-separated actual lattice spacings; fixed remaining microscopic parameters')
    parser.add_argument('--m', type=float, default=.2)
    parser.add_argument('--species', help='Count:mass pairs, e.g. 1:0.2,63:2; total includes the tagged active flavor')
    parser.add_argument('--flavors', type=int, default=64)
    parser.add_argument('--inertia', type=float, default=1.)
    parser.add_argument('--orders', default='4,6,8')
    parser.add_argument('--output', type=Path, default=ROOT/'induced_photon_response_receipt.json')
    args = parser.parse_args()
    species = tuple((int(item.split(':')[0]), float(item.split(':')[1]))
                    for item in args.species.split(',')) if args.species else ()
    spacings = [float(value) for value in args.family.split(',')] if args.family else [args.a]
    results = []
    for spacing in spacings:
        started = time.time()
        result = compute_response(ResponseConfig(a=spacing, m=args.m, species=species,
            flavors=args.flavors, inertia=args.inertia,
            orders=tuple(int(x) for x in args.orders.split(','))))
        result['elapsed_seconds'] = time.time()-started
        path = (args.output if len(spacings) == 1 else args.output.with_name(
            args.output.stem+'_a'+str(spacing).replace('.', 'p')+args.output.suffix))
        path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        validate_response(result)
        results.append((result, path))
        print(json.dumps({key: result[key] for key in ('epsilon', 'kappa', 'photon_speed', 'speed_ratio', 'convergence', 'source_hashes')}, indent=2), flush=True)
        print(f'Checked capsule: {path}; elapsed={result["elapsed_seconds"]:.3f}s', flush=True)
    if len(results) > 1:
        first = results[0][0]
        first['continuum_family_capsules'] = [str(path) for _, path in results]
        args.output.write_text(json.dumps(first, indent=2, allow_nan=False)+'\n')
    print(f'Natural exit: {args.output}; all {len(results)} source-bound responses checked', flush=True)


if __name__ == '__main__':
    main()

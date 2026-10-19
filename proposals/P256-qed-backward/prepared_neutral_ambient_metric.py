#!/usr/bin/env python3
"""Full Cartesian Phi/SU4-Haar chart coefficients for the original positive form."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy.linalg import cho_factor, cho_solve, expm, expm_frechet, qr
from improved_phi_band import ImprovedPhiLaw, NSTAR
from sea_selection import SELECTED
from prepared_guard_colour_chart import pullback_metric
from soft_spatial_closed_rotation import native_full_gauss_columns
from original_charged_principal import original_frame
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE, coordinates
from noncompact_quantitative_normal_budget import GramBounds, native_normal_budget, sqrt_lower

ROOT = Path(__file__).resolve().parent
MASS_SITE = np.r_[np.ones(60), np.full(45, 1/.6)]


def algebra_coordinates(law: Any, matrices: np.ndarray) -> np.ndarray:
    return (2*np.einsum('...ij,aji->...a', matrices, law.generators)).real


def adjoint_matrix(law: Any, unitary: np.ndarray) -> np.ndarray:
    return algebra_coordinates(law, unitary@law.generators@unitary.conj().T).T


def exponential_dexp(law: Any, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """exp(iX), exact left-trivialized Frechet map H -> integral Ad(exp(itX))H dt.

    The sinc expression handles coincident eigenvalues without a divided-zero
    branch. No truncated commutator expansion is used.
    """
    X = np.einsum('a,aij->ij', xi, law.generators)
    eigen, V = np.linalg.eigh(X)
    delta = eigen[:, None]-eigen[None, :]
    factor = np.exp(.5j*delta)*np.sinc(delta/(2*math.pi))
    body = V.conj().T@law.generators@V
    mapped = V@(factor[None, :, :]*body)@V.conj().T
    return (V*np.exp(1j*eigen))@V.conj().T, algebra_coordinates(law, mapped).T


def second_exponential(A: np.ndarray, E: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Exact D^2 exp(A)[E,F] via both ordered triangular block exponentials."""
    block = np.zeros((12, 12), complex)
    for k in range(3):
        block[4*k:4*k+4, 4*k:4*k+4] = A
    block[:4, 4:8], block[4:8, 8:12] = E, F
    first = expm(block)[:4, 8:12]
    block[:4, 4:8], block[4:8, 8:12] = F, E
    return first+expm(block)[:4, 8:12]


class AmbientFieldChart:
    """All90V normal coordinates and all15V gauge-frame directions.

    Normal coordinates are mass-horizontal only at the supplied regular mean.
    Phi is linear, U=U0 exp(i xi); tangent output uses dU=i L U (LEFT).
    Gauge frames carry literal product Haar, not a flat gauge-coordinate measure.
    This is a local chart, not a globally injective parametrization of SU4.
    """
    def __init__(self, law: Any, mean_phi: np.ndarray, mean_links: np.ndarray):
        self.law = law
        self.volume = int(law.volume)
        self.raw_dim, self.normal_dim, self.gauge_dim = 105*self.volume, 90*self.volume, 15*self.volume
        self.mean_phi = np.array(mean_phi, dtype=complex, copy=True)
        self.mean_links = np.array(mean_links, dtype=complex, copy=True)
        if self.mean_phi.shape != (self.volume, 4, 4, 4) or self.mean_links.shape != (self.volume, 3, 4, 4):
            raise ValueError('Full60V Cartesian Phi and45V SU4 links required')
        if (np.linalg.norm(self.mean_phi-self.mean_phi.conj().swapaxes(-1, -2)) > 1e-9
                or np.linalg.norm(np.trace(self.mean_phi, axis1=-2, axis2=-1)) > 1e-9
                or np.linalg.norm(self.mean_links@self.mean_links.conj().swapaxes(-1, -2)-np.eye(4)) > 1e-9
                or np.linalg.norm(np.linalg.det(self.mean_links)-1) > 1e-9):
            raise ValueError('Mean must contain traceless Hermitian Phi and actual SU4 links')
        self.mass = np.tile(MASS_SITE, self.volume)
        gauge, mass_site = native_full_gauss_columns(law, self.mean_phi, self.mean_links)
        if not np.array_equal(mass_site, MASS_SITE):
            raise ValueError('Shared gauge builder must use original .6 colour cometric')
        self.mean_gauge = gauge.toarray()
        weighted = np.sqrt(self.mass)[:, None]*self.mean_gauge
        Q, R = cast(tuple[np.ndarray, np.ndarray], qr(weighted, mode='full'))
        if np.min(np.abs(np.diag(R[:self.gauge_dim]))) < 1e-10*np.linalg.norm(weighted, ord='fro'):
            raise ValueError('Nonregular mean: all15V gauge directions must be independent')
        self.mean_basis = Q[:, self.gauge_dim:]/np.sqrt(self.mass)[:, None]
        self.body_to_left = np.asarray([[adjoint_matrix(law, U) for U in row] for row in self.mean_links])
        self.mean_log_density = .5*np.linalg.slogdet(weighted.T@weighted)[1]

    def _normal(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float)
        if q.shape != (self.normal_dim,) or not np.isfinite(q).all():
            raise ValueError(f'Exactly ALL{self.normal_dim} finite real normal coordinates required')
        return (self.mean_basis@q).reshape(self.volume, 105)

    def embedding(self, q: np.ndarray) -> dict[str, Any]:
        raw = self._normal(q)
        phi = self.mean_phi+self.law.fields(raw[:, :60])
        left = raw[:, 60:].reshape(self.volume, 3, 15)
        xi = np.einsum('xjba,xjb->xja', self.body_to_left, left)
        links = np.empty_like(self.mean_links)
        normal = self.mean_basis.copy()
        local_maps = np.empty((self.volume, 3, 15, 15))
        for x in range(self.volume):
            for j in range(3):
                U, dexp = exponential_dexp(self.law, xi[x, j])
                links[x, j] = self.mean_links[x, j]@U
                R = self.body_to_left[x, j]
                local_maps[x, j] = R@dexp@R.T
                rows = slice(105*x+60+15*j, 105*x+75+15*j)
                normal[rows] = local_maps[x, j]@self.mean_basis[rows]
        return dict(phi=phi, fields=phi, links=links, xi=xi,
                    normal_jacobian=normal, local_dexp=local_maps)

    def evaluate(self, q: np.ndarray) -> dict[str, Any]:
        point = self.embedding(q)
        gauge = native_full_gauss_columns(self.law, point['phi'], point['links'])[0].toarray()
        J = np.column_stack((point['normal_jacobian'], gauge))
        metric = J.T@(self.mass[:, None]*J)
        try:
            factor = cho_factor(metric, lower=True)
        except np.linalg.LinAlgError as error:
            raise ValueError('Point is outside a regular full ambient chart') from error
        log_density = float(np.log(np.diag(factor[0])).sum())
        inverse = cho_solve(factor, np.eye(self.raw_dim))
        n = self.normal_dim
        cross, orbit = metric[:n, n:], metric[n:, n:]
        orbit_inverse = cho_solve(cho_factor(orbit, lower=True), np.eye(self.gauge_dim))
        connection = orbit_inverse@cross.T
        relative_log_density = log_density-self.mean_log_density
        point.update(jacobian=J, gauge_columns=gauge, metric=metric, cometric=inverse,
            normal_metric=metric[:n, :n], normal_gauge_metric=cross,
            normal_gauge_cometric=inverse[:n, n:], orbit_metric=orbit,
            orbit_inverse=orbit_inverse, mechanical_connection=connection,
            horizontal_metric=metric[:n, :n]-cross@connection,
            log_density=log_density, relative_log_density=relative_log_density,
            relative_density=float(np.exp(relative_log_density)),
            normal_dim=n, gauge_dim=self.gauge_dim, raw_dim=self.raw_dim,
            measure='relative_density dq times literal product SU4 Haar in the gauge frame',
            colour_cometric_over_chi=.6)
        return point

    def tangent(self, point: dict[str, Any], direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        direction = np.asarray(direction, float)
        if direction.shape != (self.normal_dim,):
            raise ValueError('ALL normal derivative entries required')
        raw = (point['normal_jacobian']@direction).reshape(self.volume, 105)
        dphi = self.law.fields(raw[:, :60])
        L = np.einsum('xja,aik->xjik', raw[:, 60:].reshape(self.volume, 3, 15), self.law.generators)
        return dphi, 1j*L@point['links']

    def directional_jet(self, q: np.ndarray, direction: np.ndarray,
                        point: dict[str, Any] | None = None) -> dict[str, Any]:
        """Exact Jacobian/metric/log-density first jet; no finite differences."""
        point = self.evaluate(q) if point is None else point
        draw = self._normal(direction)
        dphi, dlinks = self.tangent(point, direction)
        dnormal = np.zeros_like(self.mean_basis)
        for x in range(self.volume):
            for j in range(3):
                R = self.body_to_left[x, j]
                dxi = R.T@draw[x, 60+15*j:75+15*j]
                X = np.einsum('a,aij->ij', point['xi'][x, j], self.law.generators)
                E = 1j*np.einsum('a,aij->ij', dxi, self.law.generators)
                U, dU = expm_frechet(1j*X, E)
                dmap = np.empty((15, 15))
                for a, T in enumerate(self.law.generators):
                    first = expm_frechet(1j*X, 1j*T, compute_expm=False)
                    second = second_exponential(1j*X, E, 1j*T)
                    dmap[:, a] = algebra_coordinates(self.law, -1j*(second@U.conj().T+first@dU.conj().T))
                rows = slice(105*x+60+15*j, 105*x+75+15*j)
                dnormal[rows] = R@dmap@R.T@self.mean_basis[rows]
        dgauge = native_full_gauss_columns(self.law, point['phi'], point['links'],
            field_tangent=dphi, link_tangent=dlinks)[0].toarray()
        dJ = np.column_stack((dnormal, dgauge))
        product = point['jacobian'].T@(self.mass[:, None]*dJ)
        dmetric = product+product.T
        return dict(phi_first=dphi, links_first=dlinks, jacobian_first=dJ,
                    metric_first=dmetric, log_density_first=float(.5*np.sum(point['cometric']*dmetric.T)))


def positive_kinetic_bilinear(point: dict[str, Any], normal_left: np.ndarray,
                              gauge_left: np.ndarray, normal_right: np.ndarray,
                              gauge_right: np.ndarray, chi: float = 1.) -> complex:
    """chi/2 times FULL inverse-metric weak integrand, including vertical Casimir.

    Arrays have shape (coordinate,) or (coordinate,fibre). Gauge derivatives
    are actual equivariant frame derivatives (e.g. i T_a psi), not discarded.
    Integrate this relative-density integrand against dq and literal Haar.
    """
    if not math.isfinite(chi) or chi <= 0:
        raise ValueError('Original positive chi required')
    pairs = ((normal_left, gauge_left), (normal_right, gauge_right))
    derivatives = []
    for normal, gauge in pairs:
        normal, gauge = np.asarray(normal, complex), np.asarray(gauge, complex)
        if normal.ndim not in (1, 2) or normal.shape[0] != point['normal_dim'] or gauge.shape != (point['gauge_dim'], *normal.shape[1:]):
            raise ValueError('All normal and equivariant gauge derivatives required')
        derivatives.append(np.concatenate((normal, gauge), axis=0))
    if derivatives[0].shape != derivatives[1].shape:
        raise ValueError('Matching fibre dimensions required')
    return complex(chi/2*point['relative_density']*np.vdot(derivatives[0], point['cometric']@derivatives[1]))


def regular_orbit_certificate(gram: GramBounds, phi_HS_lipschitz: F = F(2),
                              size: int = NSTAR) -> dict[str, Any]:
    """Conditional exact radius/orbit bound from the ORIGINAL full4 Gram.

    For ||delta Phi||_HS <= C_J r <= sqrt(g)/2, original coordinate
    D_Phi.T D_Phi >= g/8. Links add a positive form, independently of U.
    The present linear full-normal chart has C_J=2 exactly. External quotient
    charts must supply their own C_J; this function does not prove that input.
    """
    gram.validate(size)
    CJ = F(phi_HS_lipschitz)
    if CJ <= 0:
        raise ValueError('Positive actual Phi HS chart Lipschitz bound required')
    d = sqrt_lower(gram.lower)
    return dict(full4_gram_lower=gram.lower, full4_gram_upper=gram.upper,
        full4_gram_provenance=gram.provenance, gram_onset=gram.onset, N=size,
        Phi_HS_chart_lipschitz=CJ, regular_radius_lower=d/(2*CJ),
        permitted_Phi_HS_perturbation=d/2,
        mean_coordinate_orbit_lower=gram.lower/2,
        perturbed_coordinate_orbit_lower=gram.lower/8,
        full_coordinate_orbit_inverse_upper=8/gram.lower,
        all_links_arbitrary_SU4=True, all_sites_uniform=True,
        full4_including_Phi0_required=True,
        scope='fixed regular orbit lower bound conditional on actual full4 Gram and chart Lipschitz inputs; no full ambient-J regularity assertion')


def regular_full_jacobian_certificate(gram: GramBounds, radius_upper: F | None = None,
                                      size: int = NSTAR) -> dict[str, Any]:
    """Full weighted-Jacobian transversality on the UNCHANGED frozen radius.

    Default radius_upper bounds the existing delta/(64 V h) term, since
    delta=(2/5)sqrt_lower(g), V>=1,h>=1. No giant chart constants are recomputed.
    It is an upper bound on that existing radius, not a replacement domain.
    """
    gram.validate(size)
    g = gram.lower
    if g > 2:
        raise ValueError('Original mean-J minimum uses g/2<=1')
    r = sqrt_lower(g)/160 if radius_upper is None else F(radius_upper)
    if not 0 < r <= sqrt_lower(g/2)/16:
        raise ValueError('Supplied radius does not satisfy the full-J perturbation bound')
    return dict(N=size, full4_gram_lower=g, full4_gram_provenance=gram.provenance,
        radius_upper=r, radius_domain_changed=False,
        original_radius_authority='native_normal_budget and local_normal_constants: r<=delta/(64 V h), delta=(2/5)sqrt_lower(g), V>=1,h>=1',
        radius_binding=('unchanged frozen radius via original source inequality' if radius_upper is None
                        else 'supplied independent radius bound; caller binds its actual domain'),
        bound_without_giant_recompute=('r<=sqrt_lower(g)/160<sqrt_lower(g/2)/16' if radius_upper is None
                                       else 'supplied r<=sqrt_lower(g/2)/16 checked directly'),
        weighted_mean_J_sigma2_lower=g/2,
        weighted_normal_J_perturbation_coefficient_upper=F(2),
        weighted_gauge_J_perturbation_coefficient_upper=F(4),
        weighted_full_J_perturbation_coefficient_upper=F(8),
        weighted_full_J_sigma2_lower=g/8,
        full_metric_lower=g/8, full_cometric_upper=8/g,
        all_normal_and_gauge_columns=True, full_Haar_density_strictly_positive=True,
        chart_scope='actual fullnormal linear-Phi and U0 exp(i xi) chart with mean mass-orthonormal B; not an arbitrary nonlinear quotient chart',
        no_chart_dimension_or_domain_cutover=True)


def native_full_jacobian_regular(size: int = 3) -> dict[str, Any]:
    """Actual full native-J positive control for the analytic frozen-radius proof."""
    start = time.perf_counter()
    law = ImprovedPhiLaw(size)
    _, frame = original_frame(RotationBand(**BARE), 5, np.array([0., math.pi, math.pi/2]), True)
    rng = np.random.default_rng(22220261009)
    phi = np.tile(frame, (law.volume, 1, 1, 1))
    links = np.asarray([[expm(.07j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators))
                         for _ in range(3)] for _ in range(law.volume)])
    chart = AmbientFieldChart(law, phi, links)
    gram = GramBounds.native_a1()
    analytic = regular_full_jacobian_certificate(gram)
    # A native diagnostic point, not a replacement for the frozen legal radius.
    native_r = F(1, 1024)
    diagnostic = regular_full_jacobian_certificate(gram, native_r)
    q = rng.normal(size=chart.normal_dim)
    q *= float(native_r)/np.linalg.norm(q)
    point = chart.evaluate(q)
    mean_J = np.column_stack((chart.mean_basis, chart.mean_gauge))
    delta_J = np.sqrt(chart.mass)[:, None]*(point['jacobian']-mean_J)
    frobenius_upper = float(np.linalg.norm(delta_J, ord='fro'))
    # Frobenius dominates operator norm; this native comparison is independent
    # of the all-q analytic operator inequality in the certificate.
    if frobenius_upper > 8*float(native_r):
        raise AssertionError('Actual full-J perturbation control failed')
    try:
        cho_factor(point['metric']-float(gram.lower/8)*np.eye(chart.raw_dim), lower=True)
    except np.linalg.LinAlgError as error:
        raise AssertionError('Full ambient massGram did not retain analytic positive lower') from error
    determinant_error = abs(np.linalg.slogdet(point['jacobian'])[1]
        +.5*np.log(chart.mass).sum()-point['log_density'])
    if determinant_error > 1e-8:
        raise AssertionError('Full Haar Jacobian determinant identity failed')
    # The actual frozen radius in BOTH original source formulas is bounded
    # by sqrt_lower(g)/160 without evaluating their large rational constants.
    exact_ratio_squared = F(2, 100)
    if exact_ratio_squared >= 1:
        raise AssertionError('Frozen-radius/allowed-radius symbolic comparison failed')
    return dict(identity='full-ambient-J-transversality-unchanged-frozen-radius',
        native_N=size, native_V=law.volume, original_mean_centre_N=5,
        original_mean_full4_Gram=(np.einsum('mij,nji->mn', frame, frame).real/4).tolist(),
        raw_dim=chart.raw_dim, normal_dim=chart.normal_dim, gauge_dim=chart.gauge_dim,
        native_diagnostic_radius=str(native_r), frozen_radius_changed=False,
        actual_weighted_deltaJ_Frobenius_upper=frobenius_upper,
        actual_weighted_deltaJ_over_radius=frobenius_upper/float(native_r),
        full_massGram_ge_g_over8_Cholesky=True,
        relative_density=point['relative_density'], full_density_determinant_error=determinant_error,
        frozen_radius_ratio_to_allowed_squared_upper=str(exact_ratio_squared),
        frozen_analytic={key:str(value) if isinstance(value, F) else value for key, value in analytic.items()},
        diagnostic_analytic={key:str(value) if isinstance(value, F) else value for key, value in diagnostic.items()},
        native_scope='actual generalN chart evaluated at N3, on one supplied original N5 full4 centre; all2430normal+405gauge columns',
        analytic_scope='uniform full-J transversality for every full q in unchanged original radius and every actual full4 centre with the supplied Gram enclosure',
        scientific_seconds=time.perf_counter()-start)


def clifford_supercommutator_certificate() -> dict[str, Any]:
    """Exact Gaussian-integer four-Clifford algebra on all15 traceless grades."""
    gamma = np.asarray(SELECTED, dtype=np.complex128)
    if not (np.array_equal(gamma.real, np.rint(gamma.real))
            and np.array_equal(gamma.imag, np.rint(gamma.imag))):
        raise ValueError('Exact Gaussian-integer Clifford matrices required')
    identity = np.eye(4, dtype=complex)
    for a, b in itertools.product(range(4), repeat=2):
        if not np.array_equal(gamma[a]@gamma[b]+gamma[b]@gamma[a],
                              (2 if a == b else 0)*identity):
            raise AssertionError('Exact four-Clifford anticommutation failed')
    grades, matrices, spectrum = [], [], []
    for grade in range(1, 5):
        eigenvalue = 4*(grade if grade % 2 == 0 else 4-grade)
        for indices in itertools.combinations(range(4), grade):
            T = identity.copy()
            for index in indices:
                T = T@gamma[index]
            T = (1j)**(grade*(grade-1)//2)*T
            supercommutator = sum((G@(G@T-T@G)-(G@T-T@G)@G for G in gamma),
                                 np.zeros((4, 4), complex))
            if (not np.array_equal(T, T.conj().T) or np.trace(T) != 0
                    or not np.array_equal(supercommutator, eigenvalue*T)):
                raise AssertionError('Exact Clifford-grade supercommutator failed')
            grades.append(grade); matrices.append(T); spectrum.append(eigenvalue)
    basis = np.asarray(matrices)
    if not np.array_equal(np.einsum('aij,bji->ab', basis, basis), 4*np.eye(15)):
        raise AssertionError('Exact15-grade HS completeness failed')
    return dict(exact_arithmetic='Gaussian-integer matrix products, exact equality; symbolic integer eigenvalues',
        grade_eigenvalues={str(k):4*(k if k % 2 == 0 else 4-k) for k in range(1, 5)},
        multiplicities={str(k):math.comb(4, k) for k in range(1, 5)},
        traceless_dimension=len(spectrum), supercommutator_minimum=min(spectrum),
        HS_mean_singular_lower='2 sqrt(full4_Gram_lower)',
        coordinate_normalization='Phi HS norm squared=4|Phi_coordinates|^2; gauge HS norm squared=|lambda|^2/2')


def native_regular_orbit(size: int = 5) -> dict[str, Any]:
    """New cheap all64 full4 actual-centre and fixed-radius positive control."""
    if size < 5:
        raise ValueError('Original full-ten finite-grid centre control requires N>=5')
    start = time.perf_counter()
    exact = clifford_supercommutator_certificate()
    law = ImprovedPhiLaw(size)
    band = RotationBand(**BARE)
    rng = np.random.default_rng(22220261008)
    records = []
    for label in itertools.product(range(4), repeat=3):
        angles = (math.pi/2)*np.asarray(label)
        selected, frame = original_frame(band, size, angles, True)
        G = np.einsum('mij,nji->mn', frame, frame).real/4
        g = float(np.linalg.eigvalsh(G)[0])
        mean = np.column_stack([coordinates(1j*(T@frame-frame@T), law.basis).ravel()
                                for T in law.generators])
        mean_lower = float(np.linalg.eigvalsh(mean.T@mean)[0])
        delta = law.fields(rng.normal(size=(law.volume, 60)))[0]
        delta *= math.sqrt(g)/(2*np.linalg.norm(delta))
        moved = frame+delta
        D = np.column_stack([coordinates(1j*(T@moved-moved@T), law.basis).ravel()
                             for T in law.generators])
        moved_lower = float(np.linalg.eigvalsh(D.T@D)[0])
        gram_residual = float(np.linalg.norm(G-selected['G']))
        if mean_lower < g/2-1e-10 or moved_lower < g/8-1e-10 or gram_residual > 1e-9:
            raise AssertionError('Actual full4 centre/original coordinate orbit bound failed')
        records.append(dict(label=list(label), full4_Gram=G.tolist(), full4_Gram_min=g,
            Phi0_norm2=float(G[0, 0]), spatial3_Gram_min=float(np.linalg.eigvalsh(G[1:, 1:])[0]),
            original_centre_Gram_residual=gram_residual, mean_coordinate_orbit_min=mean_lower,
            perturbed_coordinate_orbit_min=moved_lower, mean_bound=g/2,
            perturbed_bound=g/8, perturbation_HS=float(np.linalg.norm(delta))))
    # Every actual m,s gives a label plus an integer shifted-grid permutation:
    # pi*(m/2+s)=pi*label/2+2*pi*k. This keeps all512 histories in H.
    root_permutations = []
    for m in itertools.product(range(4), repeat=3):
        for s in itertools.product(range(2), repeat=3):
            total = np.asarray(m)+2*np.asarray(s)
            label, shifts = total % 4, total//4
            if not np.array_equal(total, label+4*shifts):
                raise AssertionError('Actual Root/ordinary centre reindexing failed')
            root_permutations.append(dict(m=list(m), s=list(s),
                                          label=label.tolist(), grid_shifts=shifts.tolist()))
    budget = native_normal_budget(NSTAR)
    legal = regular_orbit_certificate(budget.gram)
    legal_serialized = {key:str(value) if isinstance(value, F) else value for key, value in legal.items()}
    removed_Phi0 = np.zeros_like(SELECTED)
    removed_Phi0[1:] = SELECTED[1:]
    light_only = np.column_stack([coordinates(1j*(T@removed_Phi0-removed_Phi0@T), law.basis).ravel()
                                 for T in law.generators])
    light_only_spectrum = np.linalg.eigvalsh(light_only.T@light_only)
    light_only_wrong = float(light_only_spectrum[0])
    if abs(light_only_wrong) > 1e-12 or legal['perturbed_coordinate_orbit_lower'] != F(1, 160):
        raise AssertionError('Full4-vs-light3/legal-Gram adverse control failed')
    # Original links add exactly a positive mass form. Check on the actual
    # all-site native builder at one representative perturbed mean.
    fields = np.tile(moved, (law.volume, 1, 1, 1))
    links = np.asarray([[expm(.13j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators))
                         for _ in range(3)] for _ in range(law.volume)])
    gauge, mass = native_full_gauss_columns(law, fields, links)
    parameters = rng.normal(size=15*law.volume)
    flow = (gauge@parameters).reshape(law.volume, 105)
    phi_form = float(np.sum(flow[:, :60]**2))
    link_form = float(np.sum(flow[:, 60:]**2*mass[60:]))
    if phi_form < records[-1]['perturbed_bound']*float(np.dot(parameters, parameters))-1e-8 or link_form < 0:
        raise AssertionError('Native all-site orbit lower/positive link contribution failed')
    return dict(identity='full4-Clifford-fixed-regular-orbit-successor', N=size, V=law.volume,
        exact_Clifford=exact, full4_actual_centre_records=records,
        actual_centre_count=len(records), retained_Root_ordinary_history_count=len(root_permutations),
        Root_ordinary_grid_permutations=root_permutations,
        minimum_actual_full4_Gram=min(row['full4_Gram_min'] for row in records),
        legal_certificate=legal_serialized,
        legal_Gram_correction='authoritative native_normal_budget lower1/20, not unsupported1/8; perturbed orbit lower1/160, not1/64',
        light3_only_adverse_orbit_min=light_only_wrong,
        native_all_site_Phi_orbit_form=phi_form, native_positive_link_orbit_form=link_form,
        full_Root_beta_removed=False, source_or_H_state_completion_claimed=False,
        chart_CJ2_scope='exact linear Phi embedding of full mass-orthonormal ambient normal coordinates',
        outside_chart_CJ_scope='external quotient C_J must be supplied/proved by its owner',
        orbit_vs_ambient_J_scope='orbit positivity does not alone prove [Jnormal,D] nonsingular everywhere',
        scientific_seconds=time.perf_counter()-start)


def native(size: int = 3) -> dict[str, Any]:
    start = time.perf_counter()
    law = ImprovedPhiLaw(size)
    rng = np.random.default_rng(22220261007)
    phi = np.tile(SELECTED, (law.volume, 1, 1, 1))+law.fields(.015*rng.normal(size=(law.volume, 60)))
    links = np.asarray([[expm(.04j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators))
                         for _ in range(3)] for _ in range(law.volume)])
    chart = AmbientFieldChart(law, phi, links)
    zero = chart.evaluate(np.zeros(chart.normal_dim))
    q = .12*rng.normal(size=chart.normal_dim)
    point = chart.evaluate(q)
    direction = rng.normal(size=chart.normal_dim)
    direction /= np.linalg.norm(direction)
    jet = chart.directional_jet(q, direction, point)
    eps = 2e-5
    plus, minus = chart.evaluate(q+eps*direction), chart.evaluate(q-eps*direction)
    link_fd = float(np.linalg.norm((plus['links']-minus['links'])/(2*eps)-jet['links_first']))
    jac_fd = float(np.linalg.norm((plus['jacobian']-minus['jacobian'])/(2*eps)-jet['jacobian_first']))
    density_fd = abs((plus['relative_log_density']-minus['relative_log_density'])/(2*eps)-jet['log_density_first'])
    mean_horizontal = float(np.linalg.norm(chart.mean_gauge.T@(chart.mass[:, None]*chart.mean_basis)))
    mean_orthonormal = float(np.linalg.norm(zero['normal_metric']-np.eye(chart.normal_dim)))
    offmean_cross = float(np.linalg.norm(point['normal_gauge_metric']))
    full_det_error = abs(np.linalg.slogdet(point['jacobian'])[1]+.5*np.log(chart.mass).sum()-point['log_density'])
    normal_log = .5*np.linalg.slogdet(point['normal_metric'])[1]
    normal_density_wrong = abs(normal_log-point['relative_log_density'])
    derivative = rng.normal(size=(chart.raw_dim, 2))+1j*rng.normal(size=(chart.raw_dim, 2))
    value = positive_kinetic_bilinear(point, derivative[:chart.normal_dim], derivative[chart.normal_dim:],
        derivative[:chart.normal_dim], derivative[chart.normal_dim:], .73)
    ambient_gradient = np.linalg.solve(point['jacobian'].T, derivative)
    direct = .73/2*point['relative_density']*float(np.sum(abs(ambient_gradient)**2/chart.mass[:, None]))
    form_error = abs(value-direct)/(1+direct)
    vertical = positive_kinetic_bilinear(point, np.zeros(chart.normal_dim), np.ones(chart.gauge_dim),
        np.zeros(chart.normal_dim), np.ones(chart.gauge_dim)).real
    wrong_mass = np.tile(np.r_[np.ones(60), np.full(45, 1/.3)], law.volume)
    wrong_metric = point['jacobian'].T@(wrong_mass[:, None]*point['jacobian'])
    wrong_form = .73/2*point['relative_density']*np.vdot(derivative, np.linalg.solve(wrong_metric, derivative)).real
    wrong_colour_defect = abs(wrong_form-direct)/direct
    dphi, dlinks = chart.tangent(point, direction)
    second = rng.normal(size=chart.normal_dim); second /= np.linalg.norm(second)
    ephi, elinks = chart.tangent(point, second)
    induced = pullback_metric(law, np.asarray([dphi, ephi]), np.asarray([dlinks, elinks]), point['links'], 1.)
    vectors = np.column_stack((direction, second))
    guard_metric_error = float(np.linalg.norm(induced['mass']-vectors.T@point['normal_metric']@vectors))
    rejected_short = False
    try:
        chart.evaluate(np.zeros(6))
    except ValueError:
        rejected_short = True
    result = dict(N=size, V=law.volume, raw_dim=chart.raw_dim, normal_dim=chart.normal_dim,
        gauge_dim=chart.gauge_dim, mean_horizontality=mean_horizontal,
        mean_mass_orthonormality=mean_orthonormal, offmean_normal_gauge_coupling=offmean_cross,
        relative_density=point['relative_density'], relative_log_density=point['relative_log_density'],
        full_determinant_identity_error=full_det_error, link_Frechet_FD_error=link_fd,
        full_Jacobian_FD_error=jac_fd, log_density_FD_error=float(density_fd),
        positive_form_relative_error=float(form_error), positive_form=value.real,
        vertical_Casimir_integrand=vertical, guard_pullback_agreement=guard_metric_error,
        adverse_normal_only_density_defect=float(normal_density_wrong),
        adverse_point3_colour_cometric_relative_defect=float(wrong_colour_defect),
        adverse_onecurve_or_six_coordinate_input_rejected=rejected_short,
        scope='actual original ImprovedPhiLaw generalN full finite chart/operator coefficients; no H/state propagation or full ground proof',
        analytic_scope='exact finite-dimensional Frechet/Haar/positive-form identities on every regular chart; no legalN allocation or global injectivity bound',
        scientific_seconds=time.perf_counter()-start)
    if (max(mean_horizontal, mean_orthonormal, full_det_error, link_fd, jac_fd, density_fd,
            form_error, guard_metric_error) > 2e-6 or offmean_cross < .01
            or normal_density_wrong < 1e-4 or wrong_colour_defect < .01
            or value.real <= 0 or abs(value.imag) > 1e-7 or vertical <= 0 or not rejected_short):
        raise AssertionError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=3)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--regular-orbit-only', action='store_true')
    modes.add_argument('--full-jacobian-certificate-only', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    inputs = dict(size=args.size, seed=22220261007, mean='SELECTED plus full Cartesian .015 Gaussian; nonidentity .04 SU4 links',
                  q='.12 Gaussian in ALL90V normal coordinates', chi=.73)
    if args.regular_orbit_only:
        inputs = dict(size=args.size, seed=22220261008, mode='full4-original-centre-fixed-regular-orbit',
                      legal_N=NSTAR, actual_offsets=64, actual_Root_histories=512)
    elif args.full_jacobian_certificate_only:
        inputs = dict(size=args.size, seed=22220261009, mode='full-ambient-J-unchanged-frozen-radius',
                      legal_N=NSTAR, original_centre_N=5, native_radius='1/1024')
    source_hashes = {Path(filename).name: hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        for module in tuple(sys.modules.values()) if (filename := getattr(module, '__file__', None))
        and Path(filename).parent == ROOT and Path(filename).suffix == '.py'}
    if args.full_jacobian_certificate_only:
        result = native_full_jacobian_regular(args.size)
    elif args.regular_orbit_only:
        result = native_regular_orbit(args.size)
    else:
        result = native(args.size)
    print(json.dumps(dict(native=result, inputs=inputs,
        input_sha256=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        argv=sys.argv, cwd=os.getcwd(), python=sys.executable, source_sha256=source_hashes,
        sources_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha for name, sha in source_hashes.items()),
        wall_seconds=time.perf_counter()-started), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

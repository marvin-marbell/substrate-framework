#!/usr/bin/env python3
"""Original whole-heavy full105 principal form, charge Schur and BO geometry.

Two declared kinetic trajectories are compared. This is not a pole theorem.
The integration owner runs this CLI after all supplier edits have landed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import null_space

from bulk_flat_holonomy_bound import selected_gram
from bulk_quantum_selection import BulkQuantumLaw
from full_spatial_gauge_stability import SpatialFunctional, principal_modes, hermitian
from local_orbital_stability import elastic_hessian, periodic_grid
from periodic_bulk_frame import PeriodicFunctional
from rotation_band import RotationBand
from sea_selection import SELECTED, coefficients, matrix_unit_basis
from unrestricted_field_minimum import ALIGNMENT, BARE, STIFFNESS, coordinates

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = 'e1874737294f261dfc305be3f8d054bd1223a1a3142bdf1a70c6d6cc136b76a4'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
SOURCES = ('original_charged_principal.py', 'original_charged_principal.md',
           'bulk_flat_holonomy_bound.py', 'bulk_flat_holonomy_bound.md',
           'bulk_quantum_selection.py', 'full_spatial_gauge_stability.py',
           'periodic_bulk_frame.py', 'local_orbital_stability.py',
           'unrestricted_field_minimum.py', 'sea_selection.py', 'soft_budget.py',
           'rotation_band.py')


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def check(error, label, tolerance=2e-8):
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{label}: {error} > {tolerance}')


def original_frame(band, size, angles, native_toron):
    """Original convex Gram law; full ten directions for noncubic torons."""
    momenta = 2*np.pi*periodic_grid(size)/(size*band.a)
    selected = selected_gram(band, momenta, STIFFNESS/band.a)
    if native_toron:
        x = band.a*coefficients(band, momenta+angles/(size*band.a))
        symmetric = [np.diag(np.eye(4)[i]) for i in range(4)]
        for i, j in itertools.combinations(range(4), 2):
            E = np.zeros((4, 4))
            E[i, j] = E[j, i] = 1/np.sqrt(2)
            symmetric.append(E)
        symmetric = np.asarray(symmetric)
        design = np.einsum('pi,aij,pj->pa', x, symmetric, x)
        coefficient = np.einsum('aij,ij->a', symmetric, selected['G'])

        def evaluate(c):
            G = np.einsum('a,aij->ij', c, symmetric)
            radii = np.sqrt(np.einsum('pi,ij,pj->p', x, G, x))
            objective = STIFFNESS*np.dot(c, c)/2-2*radii.mean()
            gradient = STIFFNESS*c-np.mean(design/radii[:, None], axis=0)
            hessian = STIFFNESS*np.eye(10)+.5*np.einsum('pa,pb,p->ab', design, design, radii**-3)/len(x)
            return objective, gradient, hessian, G, radii

        for iteration in range(100):
            objective, gradient, hessian, G, radii = evaluate(coefficient)
            if np.max(abs(gradient)) < 2e-12:
                break
            step = -np.linalg.solve(hessian, gradient)
            fraction = 1.
            for _ in range(80):
                trial = coefficient+fraction*step
                trial_G = np.einsum('a,aij->ij', trial, symmetric)
                if np.linalg.eigvalsh(trial_G)[0] > 0:
                    # Stable convex remainder, not subtraction of O(1)
                    # objectives when Newton descent is below their ulp.
                    delta = trial-coefficient
                    dx = design@delta
                    trial_radii = np.sqrt(radii*radii+dx)
                    difference = (np.dot(gradient, delta)+STIFFNESS*np.dot(delta, delta)/2
                        +np.mean(dx**2/(radii*(trial_radii+radii)**2)))
                    if difference <= 1e-4*np.dot(gradient, delta):
                        coefficient = trial
                        break
                fraction /= 2
            else:
                raise RuntimeError('Actual full10 Gram Newton step failed its positive-domain descent')
        else:
            raise RuntimeError('Actual full10 Gram stationarity did not converge')
        objective, gradient, _, G, _ = evaluate(coefficient)
        selected = dict(G=G, value=float(objective/band.a),
                        gradient_residual=float(np.max(abs(gradient))),
                        full10_Gram=True, iterations=iteration,
                        convex_Hessian_lower=STIFFNESS,
                        units='dimensionless aH solved, physical minimum divided by a')
    eigenvalues, vectors = np.linalg.eigh(selected['G'])
    if eigenvalues[0] <= 0:
        raise ValueError('Selected frame lacks a full-rank Clifford Gram matrix')
    root = (vectors*np.sqrt(eigenvalues))@vectors.T
    frame = np.einsum('mn,nij->mij', root, SELECTED)
    return selected, frame


def physical_hessian_controls(model, dimensionless, q, parts):
    """Independent kappa owner and aH unit probes; no stationarity assumption."""
    dimparts = dimensionless.hessian_parts(q)
    actual = parts['local']+parts['sea']+parts['contact']
    expected = dimparts['local']+dimparts['sea']+dimparts['contact']
    unit_error = float(np.linalg.norm(model.band.a*actual-expected)/max(1., np.linalg.norm(expected)))
    check(unit_error, 'physical kappa=2/a and dimensionless aH Hessian', 3e-8)
    uniform_error = None
    if np.linalg.norm(q) == 0:
        independent = model.uniform.hessian(model.value)
        uniform_error = float(np.linalg.norm(parts['H'][:60, :60]-independent)/max(1., np.linalg.norm(independent)))
        check(uniform_error, 'independent actual q0 uniform60 kappa owner', 3e-8)
    return dict(physical_stiffness=STIFFNESS/model.band.a,
                dimensionless_bare_mass=model.band.a*model.band.m,
                dimensionless_aH_Hessian_relative_error=unit_error,
                independent_q0_uniform60_Hessian_relative_error=uniform_error)


def endpoint_average_controls(model):
    """Exact Clifford anticommutator mechanism, plus the wrong unsym vertex."""
    correct, wrong, algebra = [], [], []
    for axis, displacement, _, F in model.paths:
        phase = np.exp(1j*model.k@displacement)
        anti = F@model.generators+model.generators@F
        anti_dagger = F.conj().T@model.generators+model.generators@F.conj().T
        vertex = .5j*(phase[:, None, None, None]*anti[None]
                      -phase.conj()[:, None, None, None]*anti_dagger[None])
        unsym = 1j*(phase[:, None, None, None]*(F@model.generators)[None]
                    -phase.conj()[:, None, None, None]*(model.generators@F.conj().T)[None])
        correct.append(-np.einsum('pij,paji->a', model.projector, vertex).real/model.count)
        wrong.append(-np.einsum('pij,paji->a', model.projector, unsym).real/model.count)
        anticommutator = model.h@F+F@model.h
        scalar = np.trace(anticommutator, axis1=1, axis2=2)/4
        algebra.append(np.linalg.norm(anticommutator-scalar[:, None, None]*np.eye(4)[None]))
    zero = float(np.linalg.norm(correct))
    wrong_norm = float(np.linalg.norm(wrong))
    check(model.band.a*zero, 'endpoint averaged Clifford heavy link gradient', 3e-8)
    check(model.band.a**2*float(np.linalg.norm(algebra)), 'Clifford endpoint anticommutator scalar', 3e-8)
    return dict(actual_endpoint_averaged_link_gradient=correct,
        actual_endpoint_gradient_norm=zero,
        Clifford_anticommutator_scaled_residual=model.band.a**2*float(np.linalg.norm(algebra)),
        wrong_unsymmetrized_endpoint_link_gradient=wrong,
        wrong_unsymmetrized_endpoint_gradient_norm=wrong_norm,
        wrong_unsym_vertex_discriminates_at_this_input=bool(model.band.a*wrong_norm > 1e-7),
        symmetry_scope='Quarter-centre native control supplies a positive adverse witness; symmetry may null the wrong vertex on other grids.')


def spectral_resolution(matrix, values, vectors):
    """Measured eigen-residual plus a conservative floating resolution scale."""
    measured = float(np.linalg.norm(matrix@vectors-vectors*values[None], ord=2))
    rounding = float(10*np.finfo(float).eps*len(values)*max(1., np.max(abs(values))))
    return measured+rounding


def portable(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return dict(real=value.real.tolist(), imag=value.imag.tolist())
        return value.tolist()
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable(item) for item in value]
    return value


class OriginalUniform(PeriodicFunctional):
    """Source periodic momenta, original whole occupation and kappa=2/a."""
    def __init__(self, band, size, basis, centre_angles):
        super().__init__(band, size, basis)
        self.momenta = (2*np.pi*periodic_grid(size)+centre_angles)/(size*band.a)
        self.x = coefficients(band, self.momenta)
        self.stiffness = STIFFNESS/band.a

    def elastic_gradient(self, value):
        energy, gradient = super().elastic_gradient(value)
        return energy/self.band.a, gradient/self.band.a

    def hessian_parts(self, value):
        local, sea = super().hessian_parts(value)
        return local/self.band.a, sea


class OriginalSpatial(SpatialFunctional):
    """Reuse the actual endpoint vertices/contact/G, replacing only quadrature."""
    def __init__(self, band, size, basis, value, centre_angles):
        super().__init__(band, size, basis, value)
        self.uniform = OriginalUniform(band, size, basis, centre_angles)
        self.k = band.a*self.uniform.momenta
        self.x = self.uniform.x
        self.h = np.einsum('pm,mij->pij', self.x, self.frame)
        self.values, self.vectors, self.occupied, self.gap = self.uniform.spectral(self.value)
        if self.gap <= 1e-10:
            raise ValueError('Actual TOTAL whole-heavy cluster is not separated')
        self.projector = ((self.vectors*self.occupied[:, None, :])
                          @ self.vectors.conj().transpose(0, 2, 1))
        self.elastic = elastic_hessian(self.frame, basis, STIFFNESS/band.a, 0.)
        self.energy, gradient = self.uniform.evaluate(self.value)
        link_gradient = np.zeros(45)
        for start in range(0, self.count, 128):
            indices = np.arange(start, min(start+128, self.count))
            vertex = self.vertices(indices, np.zeros(3), indices)
            link_gradient -= np.einsum('pij,paji->a', self.projector[indices],
                                       vertex[:, 60:]).real/self.count
        self.gradient = np.concatenate((gradient, link_gradient))


def plaquette_hessian(q, coefficient=.7):
    """K4*||U_square-I||_HS^2/2: Tr(Ta Tb)=delta/2."""
    d = 1-np.exp(1j*q)
    result = np.zeros((105, 105), dtype=complex)
    for a, b in itertools.combinations(range(3), 2):
        curl = np.zeros((15, 105), dtype=complex)
        curl[:, 60+15*a:75+15*a] = d[b]*np.eye(15)
        curl[:, 60+15*b:75+15*b] = -d[a]*np.eye(15)
        result += coefficient/2*curl.conj().T@curl
    return result


def rank_support(G):
    _, singular, right = np.linalg.svd(G, full_matrices=False)
    tolerance = 1e-7*max(1., float(singular[0]))
    rank = int(np.sum(singular > tolerance))
    return rank, singular, right[:rank].conj().T, right[rank:].conj().T


def restoring_sectors(strong, leading, G):
    """Rank/count discriminant before applying either kinetic trajectory."""
    quotient = null_space(G.conj().T, rcond=1e-12)
    values, vectors = np.linalg.eigh(hermitian(quotient.conj().T@strong@quotient, 'strong quotient'))
    tolerance = 1e-8*max(1., float(np.max(abs(values))))
    kernel = quotient@vectors[:, abs(values) <= tolerance]
    uniform = np.linalg.eigvalsh(hermitian(kernel.conj().T@leading@kernel, 'nonstiff leading quotient'))
    return dict(physical_quotient_dimension=quotient.shape[1],
        strong_quotient_eigenvalues=values, strong_positive_rank=int(np.sum(values > tolerance)),
        strong_negative_count=int(np.sum(values < -tolerance)),
        nonstiff_normal_count=kernel.shape[1], nonstiff_original_heavy_eigenvalues=uniform,
        nonstiff_leading_negative_count=int(np.sum(uniform < -2e-6)),
        rank_tolerance=tolerance,
        scope='Euclidean native Gauss quotient rank; nonstationary Hessian is not an equilibrium claim')


def charge_form(G, field_cometric, u4) -> dict[str, Any]:
    """Complete Cartesian constrained minimum, including resolved stabilizers."""
    a = np.concatenate((np.full(60, field_cometric), np.full(45, 2*u4)))
    mass = 1/a
    rank, singular, support, stabilizer = rank_support(G)
    weighted = G.conj().T@(mass[:, None]*G)
    reduced = hermitian(support.conj().T@weighted@support, 'supported Gauss mass', relative=True)
    inverse = np.linalg.solve(reduced, np.eye(rank))
    S = support@inverse@support.conj().T
    horizontal = hermitian(np.diag(a)-G@S@G.conj().T, 'horizontal cometric', relative=True)
    inertia = hermitian(G[:60].conj().T@G[:60], 'actual Cartesian inertia')
    return dict(a=a, mass=mass, S=S, horizontal=horizontal, weighted=weighted,
                rank=rank, singular=singular, support=support, stabilizer=stabilizer,
                inertia=inertia)




def longitudinal_minimum(G, form, rho):
    """Reject unsupported charge instead of dropping stabilizer constraints."""
    unsupported = float(np.linalg.norm(form['stabilizer'].conj().T@rho))
    if unsupported > 1e-9*max(1., float(np.linalg.norm(rho))):
        raise ValueError('Inhomogeneous Gauss charge has a resolved stabilizer component')
    momentum = form['mass']*(G@form['S']@rho)
    return momentum, float(np.vdot(rho, form['S']@rho).real/2)


def charge_controls(G, form):
    rng = np.random.default_rng(222907)
    rho = form['support']@(rng.normal(size=form['rank'])+1j*rng.normal(size=form['rank']))
    p, schur = longitudinal_minimum(G, form, rho)
    cost = float(np.vdot(p, form['a']*p).real/2)
    Z = null_space(G.conj().T, rcond=1e-12)
    delta = Z@(rng.normal(size=Z.shape[1])+1j*rng.normal(size=Z.shape[1]))
    displaced = float(np.vdot(p+delta, form['a']*(p+delta)).real/2)
    controls = dict(full_Gauss_residual=float(np.linalg.norm(G.conj().T@p-rho)),
        constrained_cost=cost, Schur_cost=schur, completion_error=abs(cost-schur),
        adverse_horizontal_displacement_cost=displaced,
        exact_excess=float(np.vdot(delta, form['a']*delta).real/2),
        completion_excess_error=abs(displaced-cost-float(np.vdot(delta, form['a']*delta).real/2)),
        horizontal_cometric_residual=float(np.linalg.norm(form['horizontal']@(form['mass'][:, None]*G))))
    check(controls['full_Gauss_residual'], 'inhomogeneous complete Gauss')
    check(controls['completion_error']/max(1., cost), 'complete charge Schur')
    check(controls['completion_excess_error']/max(1., displaced), 'complete square adverse')
    check(controls['horizontal_cometric_residual'], 'mass horizontal quotient', 3e-7)
    return controls


def general_principal(potential, G, field_cometric, u4, Lambda, *, source_gauge_null):
    """General kinetic law; independent original interface cross-control."""
    form = charge_form(G, field_cometric, u4)
    mass = form['mass']
    Z = null_space((np.sqrt(mass)[:, None]*G@form['support']).conj().T, rcond=1e-12)
    root = np.sqrt(form['a'])
    reduced = hermitian(Z.conj().T@(root[:, None]*potential*root[None])@Z,
                        'original full105 physical principal', relative=True)
    eigenvalues, vectors = np.linalg.eigh(reduced)
    modes = root[:, None]*(Z@vectors)
    participation = np.sum(abs((Z@vectors)[:60])**2, axis=0)
    # z=R*z_old gives exactly the old principal_modes mass convention.
    R = np.concatenate((np.full(60, np.sqrt(field_cometric/np.sqrt(Lambda))), np.ones(45)))
    inherited = principal_modes(R[:, None]*potential*R[None]/Lambda,
                                G/R[:, None], Lambda=Lambda, ue4=u4)
    residual = float(np.max(abs(inherited['squared_frequencies']-eigenvalues))/max(1., np.max(abs(eigenvalues))))
    check(residual, 'source-owned principal_modes interface', 2e-8)
    curvature = eigenvalues/np.sum(abs(modes)**2, axis=0)
    resolution = spectral_resolution(reduced, eigenvalues, vectors)
    quotient = null_space(G.conj().T, rcond=1e-12)
    quotient_potential = hermitian(quotient.conj().T@potential@quotient, 'unweighted physical potential', relative=True)
    potential_values, potential_vectors = np.linalg.eigh(quotient_potential)
    potential_resolution = spectral_resolution(quotient_potential, potential_values, potential_vectors)
    return dict(gauge_rank=form['rank'], stabilizer_dimension=15-form['rank'],
        source_gauge_null_at_reported_resolution=source_gauge_null,
        scope='Stationary gauge-null normal form' if source_gauge_null else 'Off-stationary slice diagnostics; no mass-independent physical-sign verdict',
        normal_dimension=len(eigenvalues), squared_frequencies=eigenvalues,
        frequencies=np.sqrt(np.maximum(0., eigenvalues)),
        growth_rates=np.sqrt(np.maximum(0., -eigenvalues)),
        field_kinetic_participation=participation,
        physical_potential_curvature=curvature,
        resolved_negative_count=int(np.sum((curvature < -2e-6) & (eigenvalues < -resolution))),
        numerically_unresolved_count=int(np.sum(abs(eigenvalues) <= resolution)),
        numerically_unresolved_frequency_mask=abs(eigenvalues) <= resolution,
        floating_frequency_squared_resolution=resolution,
        euclidean_slice_potential_eigenvalues=potential_values,
        euclidean_slice_potential_resolution=potential_resolution,
        euclidean_slice_potential_resolved_negative_count=int(np.sum(potential_values < -potential_resolution)),
        mass_independent_potential_resolved_negative_count=int(np.sum(potential_values < -potential_resolution)) if source_gauge_null else None,
        mass_independent_potential_positive_at_reported_resolution=bool(np.min(potential_values) > potential_resolution) if source_gauge_null else None,
        raw_negative_count=int(np.sum(eigenvalues < 0)),
        full_mass_Gauss_residual=float(np.linalg.norm(G.conj().T@(mass[:, None]*modes))),
        inherited_principal_interface_relative_error=residual,
        kinetic_normal_basis=Z, coordinate_modes=modes), form


def slater_geometry(model, q):
    """Actual TOTAL Slater QGT, colour density/covariance; no forced block fill."""
    metrics = [np.zeros((105, 105), dtype=np.complex128) for _ in range(2)]
    covariances = [np.zeros((15, 15), dtype=np.complex128) for _ in range(2)]
    targets = model.shifted_indices(q)
    for start in range(0, model.count, 128):
        indices = np.arange(start, min(start+128, model.count))
        target = targets[indices]
        left = model.vectors[target].conj().transpose(0, 2, 1)
        right = model.vectors[indices]
        denominator = model.values[target, :, None]-model.values[indices, None, :]
        vertex = model.vertices(indices, q, target)
        transitions = np.einsum('pei,paij,pjo->paeo', left, vertex, right)
        charges = np.einsum('pei,aij,pjo->paeo', left, model.generators, right)
        for species in range(2):
            occupied = model.occupied if species == 0 else ~model.occupied
            mask = (~occupied[target, :, None]) & occupied[indices, None, :]
            inverse = np.zeros_like(denominator)
            np.divide(1., denominator, out=inverse, where=mask)
            feature = (transitions*inverse[:, None]).transpose(1, 0, 2, 3).reshape(105, -1)
            metrics[species] += feature.conj()@feature.T/model.count
            charge = (charges*mask[:, None]).transpose(1, 0, 2, 3).reshape(15, -1)
            covariances[species] += charge.conj()@charge.T/model.count
    metric, light_metric = metrics
    covariance, light_covariance = covariances
    mean = np.einsum('pij,aji->a', model.projector, model.generators).real/model.count
    light_projector = np.eye(4)[None]-model.projector
    light_mean = np.einsum('pij,aji->a', light_projector, model.generators).real/model.count
    G = model.gauge_columns(q)
    ward = float(np.linalg.norm(G.conj().T@metric@G-covariance))
    check(ward, 'Slater angular covariance equals actual QGT pullback', 3e-7)
    light_ward = float(np.linalg.norm(G.conj().T@light_metric@G-light_covariance))
    check(light_ward, 'complementary light angular QGT covariance', 3e-7)
    berry_cancellation = float(np.linalg.norm((metric+light_metric).imag)) if np.linalg.norm(q) == 0 else None
    if berry_cancellation is not None:
        check(berry_cancellation, 'actual q0 complementary Berry cancellation', 3e-7)
    check(float(np.linalg.norm(mean+light_mean)), 'complement light vacuum mean cancellation')
    return dict(metric=hermitian(metric, 'Slater QGT'), covariance=hermitian(covariance, 'Slater covariance'),
        complementary_light_metric=hermitian(light_metric, 'complement light QGT'),
        complementary_light_covariance=hermitian(light_covariance, 'complement light covariance'),
        heavy_colour_mean=mean, complementary_light_mean=light_mean,
        mean_cancellation_error=float(np.linalg.norm(mean+light_mean)),
        angular_QGT_covariance_error=ward,
        complementary_light_angular_QGT_covariance_error=light_ward,
        q0_complementary_Berry_cancellation_error=berry_cancellation,
        q0_Berry_curvature=2*metric.imag if np.linalg.norm(q) == 0 else None,
        occupied_count=int(np.sum(model.occupied)),
        occupation_histogram={str(n): int(np.sum(model.occupied.sum(axis=1) == n)) for n in range(5)},
        whole_particle_hole_gap=model.gap,
        heavy_sign='-Lambda*dGamma(h); largest TOTAL 2V modes',
        light_complement='negative cluster of the same frozen +dGamma(h), not an independent imposed local sea',
        normalization='per-cell QGT/covariance; Fourier charge operators use 1/sqrt(V)')


def singular_inertia_control(basis):
    """An actual Cartan field keeps its five Lie stabilizers at q0."""
    frame = np.zeros((4, 4, 4), dtype=complex)
    frame[0] = np.diag([5., 1., 1., -7.])
    D = np.array([coordinates(1j*(T@frame-frame@T), basis).reshape(-1)
                  for T in basis/np.sqrt(8)]).T
    G = np.vstack((D, np.zeros((45, 15))))
    form = charge_form(G, 1., .3)
    if form['rank'] != 10 or form['stabilizer'].shape[1] != 5:
        raise AssertionError('Native Cartan stabilizer was deleted by mass weighting')
    rejected = False
    try:
        longitudinal_minimum(G, form, form['stabilizer'][:, 0])
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError('Unsupported inhomogeneous stabilizer charge was silently removed')
    return dict(actual_Cartan_q0_gauge_rank=form['rank'], stabilizer_dimension=5,
        gauge_singular_values=form['singular'], unsupported_charge_rejected=rejected,
        supported_charge_completion=charge_controls(G, form))


def projected_terms(model, q, geometry, form):
    metric, covariance = geometry['metric'], geometry['covariance']
    normal_BH = float(np.trace(form['horizontal']@metric).real/2)
    angular_variance = float(np.trace(form['S']@covariance).real/2)
    full_BH = float(np.einsum('a,aa->', form['a'], metric).real/2)
    light_normal_BH = float(np.trace(form['horizontal']@geometry['complementary_light_metric']).real/2)
    light_angular_variance = float(np.trace(form['S']@geometry['complementary_light_covariance']).real/2)
    check(abs(full_BH-normal_BH-angular_variance)/max(1., full_BH), 'normal/angular BH decomposition')
    # Operator on the actual four-dimensional fundamental; mean is relevant at q0.
    mean = geometry['heavy_colour_mean'] if np.linalg.norm(q) == 0 else np.zeros(15)
    inverse = form['S'].real
    fundamental = sum(inverse[a, b]*model.generators[a]@model.generators[b]/(2*model.count)
                      for a in range(15) for b in range(15))
    linear = sum((inverse@mean)[a]*model.generators[a] for a in range(15))
    charge_operator = hermitian(fundamental+linear, 'projected light charge operator')
    spectrum = np.linalg.eigvalsh(charge_operator)
    # This four-colour probe is an operator diagnostic, not an unreferenced physical state.
    scalar_part = np.trace(charge_operator).real/4
    difference = float(np.linalg.norm(charge_operator-scalar_part*np.eye(4)))
    vertex = model.vertices(np.array([0]), q, model.shifted_indices(q)[:1])[0]
    # Explicit indexing avoids NumPy's mixed-index axis reordering.
    positive = model.vectors[0][:, model.occupied[0]]
    compressed = np.einsum('ie,aij,jf->aef', positive.conj(), vertex, positive)
    traceless = vertex-np.trace(vertex, axis1=1, axis2=2)[:, None, None]/4*np.eye(4)[None]
    normal_vertices = np.einsum('ai,aef->ief', null_space(model.gauge_columns(q).conj().T), traceless)
    return dict(heavy_common_normal_Born_Huang=normal_BH,
        heavy_common_angular_variance=angular_variance, heavy_full_Born_Huang=full_BH,
        BH_decomposition_error=abs(full_BH-normal_BH-angular_variance),
        complementary_vacuum_normal_BH=normal_BH+light_normal_BH,
        complementary_light_normal_BH=light_normal_BH,
        independent_complementary_vacuum_angular_variance=angular_variance+light_angular_variance,
        complementary_light_angular_variance=light_angular_variance,
        complementary_vacuum_Berry='trace connection/curvature cancels at q0; independent species variances ADD',
        heavy_Berry_charge_mean=mean,
        heavy_mean_only_q0_angular_cost_total=float(model.count*np.vdot(mean, form['S']@mean).real/2),
        heavy_variance_is_scalar_on_retained_light_Fock=True,
        light_fundamental_charge_operator=charge_operator,
        light_fundamental_charge_eigenvalues=spectrum,
        light_charge_operator_nonscalar_norm=difference,
        quadratic_fundamental_nonscalar_norm=float(np.linalg.norm(fundamental-np.trace(fundamental).real/4*np.eye(4))),
        heavy_mean_linear_light_eigenvalues=np.linalg.eigvalsh(linear),
        actual_light_normal_vertex_nonscalar_norm=float(np.linalg.norm(normal_vertices)),
        retained_positive_cluster_q0_vertices=compressed if np.linalg.norm(q) == 0 else None,
        charge_operator_scope='single-colour local algebra probe only; physical charged trial needs explicit bar4/negative-charge reference',
        remaining_operator_premise='Uniform isolated normal bundle/resolvent control, bounded covariant whole-heavy projector derivatives and operator-valued light-current transfer with all torons/full Gauss. Not implied by Hessian or shape selection.')




def pair_reference(model, q, form, separation):
    """Exact fund(x) tensor static bar4(y) singlet, no fake scalar reference."""
    singlet = np.eye(4).reshape(-1)/2
    local = np.array([np.kron(T, np.eye(4)) for T in model.generators])
    reference = np.array([-np.kron(np.eye(4), T.T) for T in model.generators])
    total = local+reference
    error = float(np.linalg.norm(total@singlet))
    check(error, 'physical fund plus static bar4 reference Gauss singlet')
    phase = np.exp(1j*np.dot(q, separation))
    rho = (local+phase*reference)/np.sqrt(model.count)
    image = rho@singlet
    C = image.conj()@image.T
    predicted = abs(1-phase)**2/(8*model.count)*np.eye(15)
    check(float(np.linalg.norm(C-predicted)), 'physical reference exact Fourier covariance')
    cost = float(np.trace(form['S']@C).real/2)
    wrong = np.eye(15)/(8*model.count)
    wrong_cost = float(np.trace(form['S']@wrong).real/2)
    return dict(reference='static anti-fundamental bar4 at y; no co-rotation and no scalar colour reservoir',
        separation=np.asarray(separation), exact_pair_singlet_Gauss_residual=error,
        Fourier_charge_covariance=C, exact_reference_Coulomb_cost=cost,
        reference_omitted_adverse_cost=wrong_cost,
        q0_reference_total_charge_exactly_zero=bool(np.linalg.norm(q) == 0 and np.linalg.norm(C) < 1e-20),
        note='This is the added pair charge form, not an eigenstate of the interacting heavy/light/background Hamiltonian.')


def native_control(model, centre_angles):
    law = BulkQuantumLaw(model.size, band=model.band)
    fields = np.tile(model.frame, (law.volume, 1, 1, 1))
    links = law.toron_links(centre_angles)
    h = law.onebody(fields, links)
    values = np.linalg.eigvalsh(h)
    expected = np.sort(model.values.reshape(-1))
    error = float(np.max(abs(values-expected)))
    check(model.band.a*error, 'native endpoint/whole Bloch spectrum', 3e-8)
    return dict(native_endpoint_spectrum_max_error=error,
        native_whole_occupied=int(len(values)//2), native_spectrum=values,
        geometry_owner='BulkQuantumLaw.onebody only; its local selector is NEVER executed or added')


def adverse_whole_filling(band):
    """Native traceless Cartan torons really transfer whole occupation."""
    law = BulkQuantumLaw(3, band=band)
    frame = np.zeros((4, 4, 4), dtype=complex)
    frame[0] = np.diag([5., 1., 1., -7.])
    angles = np.array([[.5, -.3, .7], [-.2, .6, -.5], [.8, -.9, .1]])
    angles = np.vstack((angles, -angles.sum(axis=0)))
    links = law.identity_links()
    for x, site in enumerate(law.sites):
        for axis in range(3):
            if site[axis] == 2:
                links[x, axis] = np.diag(np.exp(1j*angles[:, axis]))
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    values, vectors = np.linalg.eigh(law.onebody(fields, links))
    P = vectors[:, 2*law.volume:]@vectors[:, 2*law.volume:].conj().T
    density = np.array([np.trace(P[4*x:4*x+4, 4*x:4*x+4]@T).real
                        for T in law.generators for x in range(law.volume)]).reshape(15, law.volume).mean(axis=1)
    blocks = np.array([coefficients(band, (2*np.pi*law.sites+angles[i])/(3*band.a))[:, 0]*frame[0, i, i].real
                       for i in range(4)]).T
    sorted_blocks = np.sort(blocks, axis=1)
    wrong = -float(sorted_blocks[:, 2:].sum()/law.volume)
    actual = -float(values[2*law.volume:].sum()/law.volume)
    counts = np.zeros(blocks.size, dtype=bool)
    counts[np.argsort(blocks.reshape(-1), kind='stable')[-2*law.volume:]] = True
    histogram = {str(n): int(np.sum(counts.reshape(-1, 4).sum(axis=1) == n)) for n in range(5)}
    error = float(np.max(abs(np.sort(blocks.reshape(-1))-values)))
    check(band.a*error, 'native arbitrary Cartan toron control', 3e-8)
    if band.a*(wrong-actual) <= 1e-5:
        raise AssertionError('Whole-half adverse filling transfer not reached')
    return dict(native_spectrum_error=error, native_whole_half_energy=actual,
        wrong_two_per_block_energy=wrong, rejected_forced_block_error=wrong-actual,
        actual_occupation_histogram=histogram, actual_heavy_colour_mean=density,
        frame='diag(5,1,1,-7) in orbital0; other orbitals zero',
        traceless_colour_toron_angles=angles, scope='adverse non-Clifford native law, not selected equilibrium')


def run(*, frame_grid=16, grids=(3, 4, 8), lambdas=(1e4, 1e8), spacing=1., p=.5, s=.5,
        q_indices=((0, 0, 0), (1, 0, 0), (1, 1, 1)), centre=(0, 0, 0), separation=(1, 0, 0),
        native_frame_toron=False) -> dict[str, Any]:
    if frame_grid < 3 or any(n < 3 for n in grids):
        raise ValueError('Original native distinct-nearest-path box requires grids >=3')
    if not 0 < spacing <= 1 or not 0 < p < 1.25 or not 0 < s < 1:
        raise ValueError('Need 0<a<=1, 0<p<5/4, 0<s<1 for the declared two trajectories')
    if any(not np.isfinite(L) or L < 1 for L in lambdas):
        raise ValueError('Lambda must be finite >=1')
    if len(centre) != 3 or any(int(c) != c for c in centre):
        raise ValueError('Centre torons use three integer Z4 labels')
    identities = source_hashes()
    band = RotationBand(**{**BARE, 'a': spacing})
    basis = matrix_unit_basis()
    angles = np.pi/2*np.asarray(centre, dtype=int)
    selected, frame = original_frame(band, frame_grid, angles, native_frame_toron)
    value = coordinates(frame, basis).reshape(-1)
    grid_rows = []
    for size in grids:
        model = OriginalSpatial(band, size, basis, value, angles)
        dimensionless_band = RotationBand(**{**BARE, 'a': 1., 'm': spacing*band.m})
        dimensionless = OriginalSpatial(dimensionless_band, size, basis, value, angles)
        rows = []
        for supplied in q_indices:
            index = np.asarray(supplied, dtype=int)
            q = 2*np.pi*index/size
            parts = model.hessian_parts(q)
            G = model.gauge_columns(q)
            strong = parts['alignment']+plaquette_hessian(q)
            leading = parts['local']+parts['sea']+parts['contact']
            # Check the source-scale gradient/H0, not a stiff-dominated Lambda
            # matrix whose relative gauge defect can hide nonstationarity.
            source_gauge_null = bool(np.linalg.norm(band.a*model.gradient) <= 2e-8
                and np.linalg.norm(band.a*leading@G) <= 3e-8)
            geometry = slater_geometry(model, q)
            check(float(np.linalg.norm(strong@G)), 'strong positive gauge Hessian', 3e-8)
            ward = model.ward(q, parts)
            I = G[:60].conj().T@G[:60]
            b = float(np.sum(abs(1-np.exp(1j*q))**2))
            trajectories = []
            for trajectory in ('original_fast_field', 'explicit_slow_field'):
                scales = []
                for Lambda in lambdas:
                    kfield = np.sqrt(Lambda) if trajectory == 'original_fast_field' else Lambda**(-s)
                    u4 = .3*Lambda**(-p)
                    potential = Lambda*leading+Lambda**1.25*strong
                    principal, form = general_principal(potential, G, kfield, u4, Lambda,
                        source_gauge_null=source_gauge_null)
                    matrix_identity = I/kfield+b/(2*u4)*np.eye(15)
                    check(float(np.linalg.norm(form['weighted']-matrix_identity))/max(1., np.linalg.norm(matrix_identity)), 'actual inertia plus complete incidence')
                    modes = principal.pop('coordinate_modes')
                    principal.pop('kinetic_normal_basis')
                    scales.append(dict(Lambda=Lambda, field_cometric=kfield, u4=u4,
                        potential_eigenvalues=np.linalg.eigvalsh(potential),
                        principal=principal, actual_charge_Schur_matrix=form['S'],
                        charge_Schur_eigenvalues=np.linalg.eigvalsh(form['S']),
                        Gauss_mass_matrix=form['weighted'], charge_controls=charge_controls(G, form),
                        projected_BO_terms=projected_terms(model, q, geometry, form),
                        physical_reference=pair_reference(model, q, form, separation),
                        coordinate_mode_norms=np.sum(abs(modes)**2, axis=0)))
                if len(scales) > 1 and len(set(lambdas)) == len(lambdas):
                    low, high = min(scales, key=lambda x: x['Lambda']), max(scales, key=lambda x: x['Lambda'])
                    ratio = np.log(high['Lambda']/low['Lambda'])
                    omega_low = low['principal']['frequencies']
                    omega_high = high['principal']['frequencies']
                    power = np.full_like(omega_low, np.nan)
                    positive = ((omega_low > 0) & (omega_high > 0)
                        & ~low['principal']['numerically_unresolved_frequency_mask']
                        & ~high['principal']['numerically_unresolved_frequency_mask'])
                    power[positive] = np.log(omega_high[positive]/omega_low[positive])/ratio
                    # A missing logarithmic exponent is null, never a fabricated zero or NaN JSON.
                    exponents = [float(x) if np.isfinite(x) else None for x in power]
                else:
                    exponents = None
                trajectories.append(dict(name=trajectory, scales=scales,
                    sorted_frequency_finite_log_slopes=exponents,
                    slopes_scope='finite diagnostics by sorted eigenvalue, not identified branches or asymptotic proof'))
            rows.append(dict(q_index=index, q_angular=q, q_physical=q/spacing,
                native_gauge_columns=G, field_Cartesian_inertia=I,
                inertia_eigenvalues=np.linalg.eigvalsh(I), incidence_norm_squared=b,
                strong_Hessian_eigenvalues=np.linalg.eigvalsh(strong),
                restoring_sectors=restoring_sectors(strong, leading, G),
                physical_unit_controls=physical_hessian_controls(model, dimensionless, q, parts),
                original_local_heavy_Hessian_eigenvalues=np.linalg.eigvalsh(leading),
                whole_heavy_geometry=geometry, actual_Ward=ward, trajectories=trajectories))
        grid_rows.append(dict(grid=size, volume=model.count,
            finite_frame_energy_per_cell=model.energy, gradient=model.gradient,
            gradient_norm=float(np.linalg.norm(model.gradient)),
            stationary_at_this_actual_grid=bool(np.linalg.norm(model.gradient) < 2e-8),
            endpoint_average_controls=endpoint_average_controls(model),
            native_endpoint_control=native_control(model, angles) if size in (3, 4) else None,
            q_rows=rows))
    if source_hashes() != identities:
        raise RuntimeError('Supplier/source/proof changed during execution; receipt rejected')
    return dict(source_sha256=identities, source_bindings_unchanged=True,
        governing_plan=PLAN, assignment_plan_revision=PLAN_REVISION, template_revision=TEMPLATE_REVISION,
        inputs=dict(frame_grid=frame_grid, original_bare={**BARE, 'a': spacing}, stiffness=STIFFNESS/spacing,
            grids=grids, q_indices=q_indices, lambdas=lambdas, p=p, s=s, centre_Z4_labels=centre,
            static_reference_separation=separation, native_frame_toron=native_frame_toron),
        source_owned_original_frame=frame, selected_Gram=selected['G'],
        selected_Gram_diagnostics=selected,
        frame_scope='finite periodic original-heavy Gram minimizer; not continuum/global quantum certificate',
        law=dict(original_heavy_coefficient='Lambda', alignment='.2 Lambda^(5/4)',
            colour_plaquette='.7 Lambda^(5/4)', colour_electric='.3 Lambda^-p Casimir',
            original_field_kinetic='sqrt(Lambda)/2 times Cartesian -Delta',
            explicit_changed_field_kinetic='Lambda^-s/2 times Cartesian -Delta',
            whole_filling='NH=NL=2V vacuum; retained complete light Fock and full15V Gauss',
            extra_local_27CAR_supplier_energy=False, compact_U1='unchanged, not computed by these colour principal diagnostics'),
        symbolic_asymptotics=dict(slow_uniform_frequency=(1-s)/2,
            slow_stiff_field_frequency=(1.25-s)/2, stiff_colour_frequency=(1.25-p)/2,
            slow_uniform_field_width=-(1+s)/4, slow_stiff_field_width=-(1.25+s)/4,
            stiff_colour_width=-(1.25+p)/4,
            original_uniform_field_frequency=.75, original_stiff_field_frequency=.875,
            q0_original_charge='sqrt(Lambda) I^-1', q0_slow_charge='Lambda^-s I^-1',
            nonzero_q_original_charge='2 u4/b times identity plus smaller actual-inertia correction',
            slow_equal_p_s_charge='Lambda^-s [I+b/.6 identity]^-1 when p=s',
            conditions='fixed a,N, resolved positive restoring modes, bounded frame/projector derivatives; not uniform toron/volume theorem'),
        native_stabilizer_adverse_control=singular_inertia_control(basis),
        finite_diagnostics=grid_rows, native_adverse_whole_filling=adverse_whole_filling(band),
        all_checks_pass=True, low_energy_frozen_Dirac_pole_proved=False, campaign_complete=False,
        conclusions=['Full105 principal and complete Cartesian charge Schur are exact quadratic/frozen mean statements.',
            'Original sqrtLambda q0 charge/BH can diverge; deleting q0 or keeping only the heavy mean is wrong.',
            'Explicit slow kinetic trajectory removes this fixed-frame q0 obstruction, subject to bounded derivative and normal-bundle hypotheses.',
            'A complementary light vacuum cancels heavy colour MEAN/Berry at q0, not independent Slater variances.',
            'Reference is a real static bar4 representation paired with the added fundamental, not a scalar reservoir.',
            'No global low-energy embedding, thermodynamic centre phase, physical localized half-spin or compact photon pole follows.'],
        remaining_premises=['Uniform isolated positive normal/whole-heavy bundle over all relevant torons and fluctuations, including degeneracies.',
            'Controlled operator-valued light-current/compact-U1 transfer, with full Gauss/reference and normal Berry/measure terms, after scalar energy subtraction.',
            'Physical charge localization/symmetry factorization and interacting charged/neutral spectral poles.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame-grid', type=int, default=16)
    parser.add_argument('--native-frame-toron', action='store_true',
                        help='Solve all10 original Gram directions at frame-grid and actual centre toron')
    parser.add_argument('--grids', type=int, nargs='+', default=[3, 4, 8])
    parser.add_argument('--lambdas', type=float, nargs='+', default=[1e4, 1e8])
    parser.add_argument('--a', type=float, default=1.)
    parser.add_argument('--p', type=float, default=.5)
    parser.add_argument('--s', type=float, default=.5)
    parser.add_argument('--q', type=int, nargs=3, action='append')
    parser.add_argument('--centre', type=int, nargs=3, default=[0, 0, 0])
    parser.add_argument('--reference-separation', type=int, nargs=3, default=[1, 0, 0])
    parser.add_argument('--output', type=Path, default=ROOT/'original_charged_principal_receipt.json')
    args = parser.parse_args()
    result = run(frame_grid=args.frame_grid, grids=args.grids, lambdas=args.lambdas,
        spacing=args.a, p=args.p, s=args.s, q_indices=args.q or [(0, 0, 0), (1, 0, 0), (1, 1, 1)],
        centre=args.centre, separation=args.reference_separation, native_frame_toron=args.native_frame_toron)
    payload = json.dumps(portable(result), indent=2, allow_nan=False)+'\n'
    args.output.write_text(payload)
    print(json.dumps(dict(receipt=str(args.output), all_checks_pass=True,
        original_frame_grid=args.frame_grid, actual_grids=args.grids, p=args.p, s=args.s,
        native_adverse_forced_block_error=result['native_adverse_whole_filling']['rejected_forced_block_error'],
        low_energy_frozen_Dirac_pole_proved=False, campaign_complete=False), indent=2))


if __name__ == '__main__':
    main()

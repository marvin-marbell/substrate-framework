#!/usr/bin/env python3
"""Continuous, full-complex orbital response of the endpoint-average local law.

The parent owns numerical science execution. This CLI reports refinement, not a
certificate of thermodynamic order. See orbital_hydrodynamics.md for hypotheses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from scipy.linalg import eigvalsh, null_space, orth

from angular_photon_spectrum import brillouin_quadrature
from local_orbital_stability import (elastic_hessian, field_matrices,
                                     local_hamiltonian, periodic_grid, scalar_paths)
from rotation_band import RotationBand
from sea_selection import SELECTED, coefficients, matrix_unit_basis, rank_report
from soft_budget import select_gram


def hermitian(matrix):
    return (matrix + matrix.conj().T) / 2


def normalized_weights(points, supplied):
    weights = (np.full(len(points), 1 / len(points)) if supplied is None
               else np.asarray(supplied, dtype=float))
    if weights.shape != (len(points),) or np.min(weights) <= 0:
        raise ValueError('Require one strictly positive quadrature weight per point.')
    if abs(float(weights.sum()) - 1) > 1e-12:
        raise ValueError('Brillouin weights must already be normalized to one.')
    return weights


def refine_gram(x, weights, stiffness, preferred):
    """Newton solve the actual convex equations; never repair a physical Hessian."""
    initial = select_gram(x, stiffness, preferred, weights=weights)
    g = np.array([initial['G'][0, 0], initial['G'][1, 1]])
    y = np.column_stack((x[:, 0]**2, np.sum(x[:, 1:]**2, axis=1)))

    def equations(value):
        radius = np.sqrt(y @ value)
        derivative = stiffness * np.array([1., 3.]) * (value - preferred)
        derivative -= np.sum(weights[:, None] * y / radius[:, None], axis=0)
        curvature = stiffness * np.diag([1., 3.])
        curvature += .5 * np.einsum('p,pi,pj->ij', weights / radius**3, y, y)
        return derivative, curvature

    iterations = 0
    for iterations in range(24):
        derivative, curvature = equations(g)
        if np.max(np.abs(derivative)) < 3e-14 * max(1., stiffness * np.max(g)):
            break
        step = np.linalg.solve(curvature, derivative)
        scale = 1.
        while np.min(g - scale * step) <= preferred:
            scale /= 2
        old_norm = np.linalg.norm(derivative)
        while np.linalg.norm(equations(g - scale * step)[0]) > old_norm:
            scale /= 2
            if scale < 2**-30:
                raise RuntimeError('Convex equation refinement cannot decrease residual.')
        g -= scale * step
    derivative, curvature = equations(g)
    if np.max(np.abs(derivative)) > 1e-11:
        raise RuntimeError(f'Unrefined convex stationarity: {derivative}')
    selected = np.diag([g[0], g[1], g[1], g[1]])
    return selected, dict(initial_residual=float(initial['gradient_residual']),
                         refined_residual=float(np.max(np.abs(derivative))),
                         newton_steps=iterations,
                         convex_equation_hessian=eigvalsh(curvature).tolist())


def uniform_data(band, points, weights, stiffness, preferred, basis) -> dict[str, Any]:
    x = coefficients(band, points / band.a)
    selected, solution = refine_gram(x, weights, stiffness, preferred)
    frame = np.sqrt(np.diag(selected))[:, None, None] * SELECTED
    h = np.einsum('pm,mij->pij', x, frame)
    radius = np.sqrt(np.einsum('pm,mn,pn->p', x, selected, x))
    projector = (np.eye(4)[None] - h / radius[:, None, None]) / 2
    gradient = np.einsum('p,pij,aji,pm->ma', weights, projector, basis, x).real
    origin = np.einsum('mij,aji->ma', frame, basis).real / 4
    gradient += 2 * stiffness * (selected - preferred * np.eye(4)) @ origin
    commutators = np.array([1j * (generator @ frame - frame @ generator)
                           for generator in basis])
    columns = np.einsum('tmij,aji->mat', commutators, basis).real.reshape(60, 15) / 4
    orbit = orth(columns)
    complement = null_space(orbit.T)
    if orbit.shape != (60, 15) or complement.shape != (60, 45):
        raise RuntimeError('Actual commutator orbit must have dimension fifteen.')
    elastic = elastic_hessian(frame, basis, stiffness, preferred)
    algebra = max(np.linalg.norm(frame[m] @ frame[n] + frame[n] @ frame[m]
                                 - 2 * selected[m, n] * np.eye(4))
                  for m in range(4) for n in range(4))
    return dict(points=points, weights=weights, frame=frame, G=selected, x=x,
                gradient=gradient, orbit=orbit, complement=complement,
                elastic=elastic, solution=solution, radius=radius,
                average_projector=np.einsum('p,pij->ij', weights, projector),
                algebra_residual=float(algebra), design=rank_report(x))


def continuous_response(data, band, q, basis, batch_size=192, frequencies=()) -> dict[str, Any]:
    """Both spectral orientations at genuine continuous k=p-q/2, l=p+q/2.

    S_ab = <l,target|V_a|k,source>* <l,target|V_b|k,source>.
    Each S is positive Hermitian; its imaginary off-diagonal entries are kept.
    K_R(w)=H-w B-w**2 eta+O(w**3), with H=elastic-chi(0).
    """
    positive = [np.zeros((60, 60), dtype=complex) for _ in range(2)]
    berry = np.zeros((60, 60), dtype=complex)
    inertia = np.zeros((60, 60), dtype=complex)
    dynamic = [np.zeros((60, 60), dtype=complex) for _ in frequencies]
    pair_gap = np.inf
    projector_error = 0.
    projector_bilinear = np.zeros(2)
    probe = np.cos(np.arange(60) + .37)
    probe /= np.linalg.norm(probe)
    variation = np.einsum('ma,aij->mij', probe.reshape(4, 15), basis)
    for start in range(0, len(data['points']), batch_size):
        p = data['points'][start:start + batch_size]
        weights = data['weights'][start:start + batch_size]
        xminus = coefficients(band, (p - q / 2) / band.a)
        xplus = coefficients(band, (p + q / 2) / band.a)
        hminus = np.einsum('pm,mij->pij', xminus, data['frame'])
        hplus = np.einsum('pm,mij->pij', xplus, data['frame'])
        eminus, uminus = np.linalg.eigh(hminus)
        eplus, uplus = np.linalg.eigh(hplus)
        t = (xminus + xplus) / 2
        vertex = np.einsum('pm,mij->pij', t, variation)
        for orientation, (target_sector, source_sector) in enumerate(
                ((slice(2, 4), slice(0, 2)), (slice(0, 2), slice(2, 4)))):
            target = uplus[:, :, target_sector]
            source = uminus[:, :, source_sector]
            transition = np.einsum('pie,aij,pjo->paeo', target.conj(), basis, source,
                                   optimize=True)
            feature = np.einsum('pm,paeo->mapeo', t, transition).reshape(60, len(p), 2, 2)
            gap = np.abs(eplus[:, target_sector, None] - eminus[:, None, source_sector])
            pair_gap = min(pair_gap, float(gap.min()))
            if np.min(gap) <= 0 or max((abs(w) for w in frequencies), default=0.) >= gap.min():
                raise ValueError('Spectral response requires an isolated, subthreshold sea.')
            for power, destination, sign in ((1, positive[orientation], 1.),
                                             (2, berry, 1. if orientation == 0 else -1.),
                                             (3, inertia, 1.)):
                weighted = (feature * np.sqrt(weights[:, None, None] / gap**power)[None]).reshape(60, -1)
                destination += sign * (weighted.conj() @ weighted.T)
            for frequency, destination in zip(frequencies, dynamic):
                denominator = gap - frequency if orientation == 0 else gap + frequency
                weighted = (feature * np.sqrt(weights[:, None, None] / denominator)[None]).reshape(60, -1)
                destination += weighted.conj() @ weighted.T
            # An independent projector trace checks complex cross terms, not
            # just eigenvalues or individually gauge-dependent eigenvectors.
            pt = target @ target.conj().swapaxes(1, 2)
            ps = source @ source.conj().swapaxes(1, 2)
            rm = np.sqrt(np.einsum('pm,mn,pn->p', xminus, data['G'], xminus))
            rp = np.sqrt(np.einsum('pm,mn,pn->p', xplus, data['G'], xplus))
            sign_t = 1 if orientation == 0 else -1
            projector_error = max(projector_error, float(np.max(np.abs(
                pt - (np.eye(4)[None] + sign_t * hplus / rp[:, None, None]) / 2))),
                float(np.max(np.abs(ps - (np.eye(4)[None] - sign_t * hminus / rm[:, None, None]) / 2))))
            traces = np.trace(ps @ vertex @ pt @ vertex, axis1=1, axis2=2).real
            projector_bilinear[orientation] += np.sum(weights * traces / (rm + rp))
    positive = [hermitian(matrix) for matrix in positive]
    chi = positive[0] + positive[1]
    berry, inertia = hermitian(berry), hermitian(inertia)
    hessian = hermitian(data['elastic'] - chi)
    hessian += 4 * np.sum(np.sin(q / 2)**2) * data.get('J', 0.) * np.eye(60)
    probe_bilinear = float(np.vdot(probe, chi @ probe).real)
    return dict(H=hessian, chi=chi, B=berry, eta=inertia,
                dynamic=[hermitian(matrix) for matrix in dynamic], pair_gap=pair_gap,
                controls=dict(spectral_positive_minima=[float(eigvalsh(matrix)[0]) for matrix in positive],
                              eta_minimum=float(eigvalsh(inertia)[0]),
                              projector_identity_residual=projector_error,
                              projector_bilinear_residual=float(abs(probe_bilinear - projector_bilinear.sum())),
                              imaginary_static_norm=float(np.linalg.norm(chi.imag, 2)),
                              hermiticity_residual=float(np.linalg.norm(chi - chi.conj().T))))


def analytic_low_q(data, zero, band, direction, basis, batch_size) -> dict[str, Any]:
    """Differentiate the projector identity under the continuous integral.

    No small-q difference or imposed flat projection defines these matrices.
    Path derivatives retain every scalar path, including improved endpoints.
    """
    first = np.zeros((60, 60), dtype=complex)
    second = np.zeros((60, 60), dtype=complex)
    static = np.zeros((60, 60), dtype=complex)
    _, paths = scalar_paths(band)
    for start in range(0, len(data['points']), batch_size):
        p = data['points'][start:start + batch_size]
        weights = data['weights'][start:start + batch_size]
        x = data['x'][start:start + batch_size]
        dx = np.zeros_like(x, dtype=complex)
        ddx = np.zeros_like(x, dtype=complex)
        for displacement, scalars in paths:
            phase = np.exp(1j * p @ displacement)[:, None]
            slope = float(direction @ displacement)
            dx += 1j * slope * (phase * scalars - phase.conj() * scalars.conj())
            ddx -= slope**2 * (phase * scalars + phase.conj() * scalars.conj())
        dx, ddx = np.real(dx), np.real(ddx)
        h = np.einsum('pm,mij->pij', x, data['frame'])
        dh = np.einsum('pm,mij->pij', dx, data['frame'])
        ddh = np.einsum('pm,mij->pij', ddx, data['frame'])
        radius = data['radius'][start:start + batch_size]
        dr = np.einsum('pm,mn,pn->p', dx, data['G'], x) / radius
        ddr = (np.einsum('pm,mn,pn->p', ddx, data['G'], x)
               + np.einsum('pm,mn,pn->p', dx, data['G'], dx) - dr**2) / radius
        n = h / radius[:, None, None]
        dn = dh / radius[:, None, None] - h * (dr / radius**2)[:, None, None]
        ddn = (ddh / radius[:, None, None] - 2 * dh * (dr / radius**2)[:, None, None]
               - h * (ddr / radius**2)[:, None, None]
               + 2 * h * (dr**2 / radius**3)[:, None, None])
        def trace_product(left, right):
            return np.einsum('pij,ajk,pkl,bli->pab', left, basis, right, basis, optimize=True)
        f0 = (4 * np.eye(15)[None] - trace_product(n, n)) / 2
        df = (trace_product(dn, n) - trace_product(n, dn)) / 4
        ddf = -(trace_product(ddn, n) + trace_product(n, ddn)
                - 2 * trace_product(dn, dn)) / 8
        static += np.einsum('p,pm,pn,pab->manb', weights / (2 * radius), x, x, f0).reshape(60, 60)
        first += np.einsum('p,pm,pn,pab->manb', weights / (2 * radius), x, x, df).reshape(60, 60)
        ddt = np.einsum('pm,pn->pmn', ddx, x) + np.einsum('pm,pn->pmn', x, ddx)
        xx = np.einsum('pm,pn->pmn', x, x)
        coefficient = ddt / (8 * radius[:, None, None]) - xx * (ddr / (8 * radius**2))[:, None, None]
        second += np.einsum('p,pmn,pab->manb', weights, coefficient, f0).reshape(60, 60)
        second += np.einsum('p,pm,pn,pab->manb', weights / (2 * radius), x, x, ddf).reshape(60, 60)
    first = -hermitian(first)
    second = -hermitian(second) + 2 * data.get('J', 0.) * np.eye(60)
    orbit, complement = data['orbit'], data['complement']
    cc = hermitian(complement.T @ zero['H'] @ complement)
    if eigvalsh(cc)[0] <= 0:
        return dict(restoring_positive=False)
    oc_first = orbit.T @ first @ complement
    naive = hermitian(orbit.T @ second @ orbit / 2)
    decrement = hermitian(oc_first @ np.linalg.solve(cc, oc_first.conj().T))
    stiffness = hermitian(naive - decrement)
    odd = hermitian(orbit.T @ first @ orbit)
    odd_values, odd_vectors = np.linalg.eigh(odd)
    values, vectors = np.linalg.eigh(stiffness)
    witness = None
    if odd_values[0] < -1e-10 or values[0] < 0:
        odd_branch = odd_values[0] < -1e-10
        u = odd_vectors[:, 0] if odd_branch else vectors[:, 0]
        leading = orbit @ u
        induced_slope = -complement @ np.linalg.solve(cc, oc_first.conj().T @ u)
        witness = dict(unit_wave_direction=direction.tolist(),
                       leading_coordinates_real=np.real(leading).reshape(4, 15).tolist(),
                       leading_coordinates_imaginary=np.imag(leading).reshape(4, 15).tolist(),
                       restoring_slope_real=np.real(induced_slope).reshape(4, 15).tolist(),
                       restoring_slope_imaginary=np.imag(induced_slope).reshape(4, 15).tolist(),
                       wave_direction='q=t*unit_wave_direction, z(t)=leading+t*restoring_slope+O(t^2)',
                       small_q_power=1 if odd_branch else 2,
                       quadratic_energy_coefficient=float(odd_values[0] if odd_branch else values[0]) / 4,
                       field='F+epsilon*(cos(q.x)*Re(z)-sin(q.x)*Im(z))*B',
                       status='Constructed asymptotic modulated candidate; quadrature sign is not interval-certified.')
    return dict(restoring_positive=True, K=stiffness, L=hermitian(orbit.T @ first @ orbit),
                first_H=first, second_H=second,
                stiffness_eigenvalues=eigvalsh(stiffness).tolist(),
                odd_coefficient_norm=float(np.linalg.norm(orbit.T @ first @ orbit, 2)),
                analytic_Hoc_slope_norm=float(np.linalg.norm(oc_first, 2)),
                forbidden_drop_Hoc_stiffness_error=float(np.linalg.norm(decrement, 2)),
                modulated_small_q_candidate=witness,
                full_projector_static_residual=float(np.linalg.norm(static - zero['chi'], 2)))


def relax_response(response, orbit, complement, mass) -> dict[str, Any]:
    """Eliminate all 45 restoring coordinates, including frequency relaxation."""
    h, b = response['H'], response['B']
    eta = response['eta'] + mass * np.eye(60)
    hcc = hermitian(complement.T @ h @ complement)
    gap = float(eigvalsh(hcc)[0])
    if gap <= 0:
        return dict(restoring_gap=gap, restoring_positive=False)
    hco = complement.T @ h @ orbit
    relaxation = np.linalg.solve(hcc, hco)
    lift = orbit - complement @ relaxation
    heff = hermitian(lift.conj().T @ h @ lift)
    beff = hermitian(lift.conj().T @ b @ lift)
    mixed_berry = lift.conj().T @ b @ complement
    eta_static_relaxation = hermitian(lift.conj().T @ eta @ lift)
    eta_berry_relaxation = hermitian(mixed_berry @ np.linalg.solve(hcc, mixed_berry.conj().T))
    eta_eff = eta_static_relaxation + eta_berry_relaxation
    naive = hermitian(orbit.T @ h @ orbit)
    decrement = hermitian(naive - heff)
    # Discarding imaginary response is a separate forbidden approximation.
    real_cc = complement.T @ h.real @ complement
    real_co = complement.T @ h.real @ orbit
    real_eff = hermitian(orbit.T @ h.real @ orbit - real_co.T @ np.linalg.solve(real_cc, real_co))
    return dict(restoring_positive=True, restoring_gap=gap, H=heff, B=beff, eta=eta_eff,
                lift=lift, Hoo=naive, decrement=decrement, real_only_H=real_eff,
                static_eta=eta_static_relaxation, berry_eta=eta_berry_relaxation,
                Hoc_norm=float(np.linalg.norm(hco, 2)),
                physical_lift_norm_squared=float(np.linalg.norm(lift, 2)**2),
                eta_minimum=float(eigvalsh(eta_eff)[0]),
                schur_decrement_minimum=float(eigvalsh(decrement)[0]))


def low_frequency_poles(h, b, eta) -> dict[str, Any]:
    """Actual quadratic pencil, including gyroscopic/type-B modes."""
    dimension = len(h)
    companion = np.block([[np.zeros_like(h), np.eye(dimension)],
                          [np.linalg.solve(eta, h), -np.linalg.solve(eta, b)]])
    roots = np.linalg.eigvals(companion)
    scale = max(1., float(np.max(np.abs(roots))))
    positive = sorted(float(w.real) for w in roots if w.real > 1e-10 * scale
                      and abs(w.imag) < 1e-8 * scale)
    return dict(positive=positive, maximum_imaginary=float(np.max(np.abs(np.imag(roots)))),
                all_real_parts=sorted(float(w.real) for w in roots))


def modulated_candidate(data, response, relaxed, q, basis) -> dict[str, Any] | None:
    """Construct the real cos/sin field when an actual curvature is negative.

    This is a variational small-amplitude witness, not a finite-amplitude trial
    falsely substituted for the thermodynamic response.
    """
    if relaxed['restoring_positive']:
        values, vectors = np.linalg.eigh(relaxed['H'])
        direction = relaxed['lift'] @ vectors[:, 0]
        curvature = float(values[0])
    else:
        values, vectors = np.linalg.eigh(response['H'])
        direction = vectors[:, 0]
        curvature = float(values[0])
    if curvature >= 0:
        return None
    perturbation = np.einsum('ma,aij->mij', direction.reshape(4, 15), basis)
    # Complex coordinates multiply a Hermitian real matrix basis. The physical
    # field is Re(exp(i q.x) z_a) B_a, NOT entrywise Re of a matrix.
    return dict(wavevector_angle=q.tolist(), frame=data['frame'].real.tolist(),
                frame_imaginary=data['frame'].imag.tolist(),
                cosine_coordinates=np.real(direction).reshape(4, 15).tolist(),
                sine_coordinates=(-np.imag(direction)).reshape(4, 15).tolist(),
                quadratic_energy_density_coefficient=curvature / 4,
                complex_matrix_direction_norm=float(np.linalg.norm(perturbation)),
                field='A_mu(x)=frame_mu+epsilon*(cos(q.x)*Re(z_mu,a)-sin(q.x)*Im(z_mu,a))*B_a',
                descent='e(epsilon)-e(0)=epsilon^2*curvature/4+O(epsilon^3); negative for sufficiently small epsilon',
                retains_all_restoring_coordinates=True,
                finite_amplitude_energy_measured=False)


def vertex_controls(band, basis) -> dict[str, Any]:
    """Continuous path identity plus independently constructed local vertices."""
    onsite, paths = scalar_paths(band)
    minus = np.array([.23, -.41, .19])
    plus = np.array([-.17, .36, .28])
    path_vertex = np.array([onsite, 0., 0., 0.], dtype=complex)
    for displacement, scalars in paths:
        phase = (np.exp(1j * minus @ displacement) + np.exp(1j * plus @ displacement)) / 2
        path_vertex += scalars * phase + scalars.conj() * phase.conjugate()
    expected = coefficients(band, np.array([minus, plus]) / band.a).mean(axis=0)
    size = 3
    sites = periodic_grid(size)
    transform = np.kron(np.exp(2j * np.pi * sites @ sites.T / size) / np.sqrt(len(sites)), np.eye(4))
    x = coefficients(band, 2 * np.pi * sites / (size * band.a))
    weights = (x[:, None, :] + x[None, :, :]) / 2
    error = 0.
    for fiber in range(60):
        coordinates = np.zeros((len(sites), 4, 15))
        coordinates[0].reshape(-1)[fiber] = 1.
        actual = transform.conj().T @ local_hamiltonian(
            field_matrices(coordinates, basis), band, sites, size) @ transform
        mu, alpha = divmod(fiber, 15)
        target = np.einsum('kp,ij->kipj', weights[:, :, mu], basis[alpha]).reshape(actual.shape) / len(sites)
        error = max(error, float(np.linalg.norm(actual - target)))
    return dict(path_count=len(paths), longest_endpoint=max(int(np.max(abs(d))) for d, _ in paths),
                continuous_endpoint_path_residual=float(np.linalg.norm(path_vertex - expected)),
                actual_local_all_60_vertex_residual=error,
                actual_local_size=size,
                owner='RotationBand.couplings -> scalar_paths -> local_hamiltonian; every path and adjoint retained')


def exact_frequency_control(data, band, q, basis, mass, batch_size) -> dict[str, Any]:
    delta = min(2e-4, .001 * float(data['radius'].min()))
    response = continuous_response(data, band, q, basis, batch_size, frequencies=(-delta, delta))
    minus, plus = response['dynamic']
    measured_eta = (minus + plus - 2 * response['chi']) / (2 * delta**2)
    measured_b = (plus - minus) / (2 * delta)
    relaxed = relax_response(response, data['orbit'], data['complement'], mass)
    result = dict(frequency_step=delta,
                  full_eta_relative_residual=float(np.linalg.norm(measured_eta - response['eta'], 2)
                                                   / max(1., np.linalg.norm(response['eta'], 2))),
                  full_B_relative_residual=float(np.linalg.norm(measured_b - response['B'], 2)
                                                 / max(1., np.linalg.norm(response['B'], 2))))
    if relaxed['restoring_positive']:
        def exact_schur(frequency, chi):
            kernel = data['elastic'] - chi - mass * frequency**2 * np.eye(60)
            kernel += 4 * np.sum(np.sin(q / 2)**2) * data.get('J', 0.) * np.eye(60)
            cc = data['complement'].T @ kernel @ data['complement']
            co = data['complement'].T @ kernel @ data['orbit']
            return hermitian(data['orbit'].T @ kernel @ data['orbit'] - co.conj().T @ np.linalg.solve(cc, co))
        km = exact_schur(-delta, minus)
        kp = exact_schur(delta, plus)
        induced_eta = -(km + kp - 2 * relaxed['H']) / (2 * delta**2)
        result['relaxed_eta_relative_residual'] = float(np.linalg.norm(induced_eta - relaxed['eta'], 2)
                                                       / max(1., np.linalg.norm(relaxed['eta'], 2)))
    return result


def refinement_record(previous, current) -> dict[str, Any] | None:
    if previous is None:
        return None
    rows = []
    for old, new in zip(previous['directions'], current['directions']):
        if old['extrapolated_stiffness'] is None or new['extrapolated_stiffness'] is None:
            continue
        older = np.asarray(old['extrapolated_stiffness'])
        newer = np.asarray(new['extrapolated_stiffness'])
        analytic_change = None
        if old['analytic_low_q']['restoring_positive'] and new['analytic_low_q']['restoring_positive']:
            analytic_change = float(np.max(abs(
                np.asarray(new['analytic_low_q']['stiffness_eigenvalues'])
                - np.asarray(old['analytic_low_q']['stiffness_eigenvalues']))))
        rows.append(dict(direction=new['direction'], stiffness_eigenvalue_max_change=float(np.max(abs(newer - older))),
                         minimum_change=float(newer[0] - older[0]), analytic_stiffness_max_change=analytic_change))
    return dict(previous_order=previous['order'], current_order=current['order'], directions=rows)


def run(args):
    if args.M <= 0 or args.J < 0 or args.batch < 1 or args.a <= 0 or args.m <= 0:
        raise ValueError('Require M,a,m,batch>0 and scalar J>=0.')
    if len(args.q) < 2 or any(q <= 0 for q in args.q) or any(args.q[i] <= args.q[i + 1] for i in range(len(args.q) - 1)):
        raise ValueError('Supply at least two strictly decreasing positive q magnitudes.')
    band = RotationBand(a=args.a, m=args.m, r=args.r, v=args.v, improved=True)
    basis = matrix_unit_basis()
    vertices = vertex_controls(band, basis)
    directions = [('axis', np.array([1., 0., 0.])),
                  ('face_diagonal', np.array([1., 1., 0.]) / np.sqrt(2)),
                  ('body_diagonal', np.ones(3) / np.sqrt(3))]
    levels = []
    refinements = []
    structural_checks = []
    for order in args.orders:
        points, supplied = brillouin_quadrature(order, rule=args.rule)
        weights = normalized_weights(points, supplied)
        data = uniform_data(band, points, weights, args.kappa, args.g0, basis)
        data['J'] = args.J
        zero = continuous_response(data, band, np.zeros(3), basis, args.batch)
        rz = relax_response(zero, data['orbit'], data['complement'], args.M)
        zero_values = eigvalsh(zero['H'])
        zero_orbit_error = float(np.linalg.norm(zero['H'] @ data['orbit'], 2))
        ward_rhs = np.array([1j * (generator @ np.einsum('ma,aij->mij', data['gradient'], basis)
                                   - np.einsum('ma,aij->mij', data['gradient'], basis) @ generator)
                             for generator in basis])
        ward_columns = np.einsum('tmij,aji->mat', ward_rhs, basis).real.reshape(60, 15) / 4
        orbit_raw = np.array([1j * (generator @ data['frame'] - data['frame'] @ generator) for generator in basis])
        orbit_columns = np.einsum('tmij,aji->mat', orbit_raw, basis).real.reshape(60, 15) / 4
        ward_error = float(np.linalg.norm(zero['H'] @ orbit_columns - ward_columns, 2))
        berry_values = eigvalsh(rz['B']) if rz['restoring_positive'] else np.array([])
        berry_rank = int(np.count_nonzero(abs(berry_values) > 1e-9))
        charge_berry = np.array([[np.trace(data['average_projector'] @ (left @ right - right @ left))
                                 for right in basis] for left in basis])
        berry_charge_error = float(np.linalg.norm(
            orbit_columns.T @ zero['B'] @ orbit_columns - charge_berry, 2))
        level: dict[str, Any] = dict(order=order, rule=args.rule, point_count=len(points), weight_sum=float(weights.sum()),
                     selected_G=data['G'].tolist(), stationarity=data['solution'],
                     full_gradient_norm=float(np.linalg.norm(data['gradient'])), design=data['design'],
                     uniform=dict(all_60_curvatures=zero_values.tolist(), physical_orbit_dimension=15,
                                  restoring_dimension=45, restoring_gap=rz['restoring_gap'],
                                  symmetry_flat_residual=zero_orbit_error, algebraic_Ward_residual=ward_error,
                                  Clifford_identity_residual=data['algebra_residual'],
                                  B_eigenvalues=berry_values.tolist(), B_rank=berry_rank,
                                  B_charge_commutator_residual=berry_charge_error,
                                  type_B_pairs_if_rank_persists=berry_rank // 2,
                                  type_A_modes_if_rank_persists=15 - berry_rank,
                                  eta_eigenvalues=eigvalsh(rz['eta']).tolist() if rz['restoring_positive'] else None,
                                  spectral_controls=zero['controls']), directions=[])
        valid = (data['design']['quadratic_rank'] == 10 and zero_orbit_error < 1e-9
                 and ward_error < 1e-9 and berry_charge_error < 1e-9 and np.linalg.norm(data['gradient']) < 1e-10
                 and np.count_nonzero(abs(zero_values) < 1e-9) == 15
                 and rz['restoring_positive'] and data['algebra_residual'] < 1e-10)
        for name, direction in directions:
            rows, even_matrices = [], []
            analytic = analytic_low_q(data, zero, band, direction, basis, args.batch)
            analytic_report = {key: value for key, value in analytic.items()
                               if key not in ('K', 'L', 'first_H', 'second_H')}
            if analytic['restoring_positive']:
                valid = valid and analytic['full_projector_static_residual'] < 1e-10
            for magnitude in args.q:
                q = magnitude * direction
                responses = [continuous_response(data, band, sign * q, basis, args.batch) for sign in (1, -1)]
                relaxed = [relax_response(response, data['orbit'], data['complement'], args.M) for response in responses]
                candidates = [modulated_candidate(data, response, relaxation, sign * q, basis)
                              for sign, response, relaxation in zip((1, -1), responses, relaxed)]
                row: dict[str, Any] = dict(q_angle=magnitude, physical_q=magnitude / band.a,
                           full_60_minima=[float(eigvalsh(response['H'])[0]) for response in responses],
                           restoring_gaps=[relaxation['restoring_gap'] for relaxation in relaxed],
                           conjugate_q_H_residual=float(np.linalg.norm(responses[0]['H'] - responses[1]['H'].conj(), 2)),
                           spectral_controls=[response['controls'] for response in responses],
                           modulated_candidates=[candidate for candidate in candidates if candidate is not None])
                valid = valid and row['conjugate_q_H_residual'] < 1e-10
                if analytic['restoring_positive']:
                    row['full_H_Taylor_error_over_q_squared'] = float(np.linalg.norm(
                        responses[0]['H'] - zero['H'] - magnitude * analytic['first_H']
                        - magnitude**2 * analytic['second_H'] / 2, 2) / magnitude**2)
                if all(relaxation['restoring_positive'] for relaxation in relaxed):
                    plus, minus = relaxed
                    even = hermitian((plus['H'] + minus['H']) / (2 * magnitude**2))
                    even_matrices.append(even)
                    if analytic['restoring_positive']:
                        row['stiffness_matrix_error_vs_analytic_derivative'] = float(
                            np.linalg.norm(even - analytic['K'], 2))
                    poles = low_frequency_poles(plus['H'], plus['B'], plus['eta'])
                    row.update(stiffness_eigenvalues=eigvalsh(even).tolist(),
                               plus_minus_effective_minima=[float(eigvalsh(r['H'])[0]) for r in relaxed],
                               orbital_odd_norm_over_q=float(np.linalg.norm((plus['H'] - minus['H']) / (2 * magnitude), 2)),
                               naive_Hoo_stiffness_eigenvalues=eigvalsh(plus['Hoo'] / magnitude**2).tolist(),
                               forbidden_drop_Hoc_stiffness_error=float(np.linalg.norm(plus['decrement'], 2) / magnitude**2),
                               forbidden_drop_imaginary_stiffness_error=float(np.linalg.norm(plus['real_only_H'] - plus['H'], 2) / magnitude**2),
                               Hoc_norm_over_q=plus['Hoc_norm'] / magnitude,
                               physical_lift_norm_squared=plus['physical_lift_norm_squared'],
                               eta_eigenvalues=eigvalsh(plus['eta']).tolist(),
                               induced_static_inertia_norm=float(np.linalg.norm(plus['static_eta'] - data['orbit'].T @
                                   (responses[0]['eta'] + args.M * np.eye(60)) @ data['orbit'], 2)),
                               induced_B_inertia_norm=float(np.linalg.norm(plus['berry_eta'], 2)),
                               schur_decrement_minimum=plus['schur_decrement_minimum'],
                               low_frequency_poles=poles,
                               poles_over_pair_gap=[w / responses[0]['pair_gap'] for w in poles['positive']],
                               poles_over_q=[w / magnitude for w in poles['positive']],
                               poles_over_q_squared=[w / magnitude**2 for w in poles['positive']])
                    valid = valid and plus['eta_minimum'] > 0 and plus['schur_decrement_minimum'] > -1e-10
                else:
                    even_matrices.append(None)
                for response in responses:
                    controls = response['controls']
                    valid = valid and min(controls['spectral_positive_minima']) > -1e-10
                    valid = valid and controls['eta_minimum'] > -1e-10
                    valid = valid and controls['projector_bilinear_residual'] < 1e-10
                    valid = valid and controls['projector_identity_residual'] < 1e-10
                rows.append(row)
            extrapolation = None
            q_error = None
            extrapolation_vs_analytic = None
            if even_matrices[-1] is not None and even_matrices[-2] is not None:
                large, small = args.q[-2:]
                extrapolated = (large**2 * even_matrices[-1] - small**2 * even_matrices[-2]) / (large**2 - small**2)
                extrapolation = eigvalsh(extrapolated).tolist()
                q_error = float(np.linalg.norm(extrapolated - even_matrices[-1], 2))
                if analytic['restoring_positive']:
                    extrapolation_vs_analytic = float(np.linalg.norm(extrapolated - analytic['K'], 2))
            level['directions'].append(dict(direction=name, unit_vector=direction.tolist(), q_sequence=rows,
                                            extrapolated_stiffness=extrapolation,
                                            analytic_low_q=analytic_report,
                                            extrapolation_matrix_error_vs_analytic=extrapolation_vs_analytic,
                                            last_q_extrapolation_matrix_error=q_error))
        level['frequency_derivative_control'] = exact_frequency_control(data, band,
            args.q[-1] * directions[-1][1], basis, args.M, args.batch)
        frequency = level['frequency_derivative_control']
        valid = valid and frequency['full_eta_relative_residual'] < 1e-5
        valid = valid and frequency.get('relaxed_eta_relative_residual', 0.) < 1e-5
        structural_checks.append(bool(valid))
        refinements.append(refinement_record(levels[-1] if levels else None, level))
        levels.append(level)
    minima = [direction['analytic_low_q']['stiffness_eigenvalues'][0] for direction in levels[-1]['directions']
              if direction['analytic_low_q']['restoring_positive']]
    sampled_rho = min(minima) if minima else None
    any_candidate = any(row['modulated_candidates'] for direction in levels[-1]['directions']
                        for row in direction['q_sequence'])
    any_candidate = any_candidate or any(
        direction['analytic_low_q'].get('modulated_small_q_candidate') is not None
        for direction in levels[-1]['directions'])
    positive_sample = len(minima) == len(directions) and sampled_rho is not None and sampled_rho > 0 and not any_candidate
    variance = None
    if sampled_rho is not None and positive_sample:
        rho = sampled_rho / 2
        variance = dict(candidate_rho=rho, cutoff_angle=args.ir_cutoff,
                        quantum_trace_bound=15 * args.ir_cutoff**2 / (8 * np.pi**2 * np.sqrt(args.M * rho)),
                        mass_scaling='M^(-1/2) for the quantum upper bound; no full quantum-order theorem',
                        finite_temperature_trace_coefficient=15 * args.ir_cutoff / (2 * np.pi**2 * rho),
                        conditional_on='Exact continuum Schur stiffness >=rho in EVERY direction below cutoff; sampled directions/refinements do not certify this.',
                        same_current_mean_square_bound_multiplier='4*|c|^2*sup_q||R(q)||^2 times orbit variance; uniform co-transported current is invariant')
    checks = dict(actual_endpoint_vertices=vertices['actual_local_all_60_vertex_residual'] < 1e-10
                  and vertices['continuous_endpoint_path_residual'] < 1e-10,
                  all_structural_response_controls=all(structural_checks))
    return dict(model='unrestricted four traceless Hermitian local orbital fields; soft Gram potential; actual half-filled sea',
                units='hbar=1; q_angle=a*q_physical; energy per primitive cell; normalized d^3p/(2pi)^3',
                parameters=dict(a=args.a, m=args.m, r=args.r, v=args.v, kappa=args.kappa, g0=args.g0, M=args.M, J=args.J),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                ownership='Parent executes science; RotationBand owns paths; same weights select Gram and integrate sea',
                full_complex_response=True, vertex_controls=vertices, levels=levels,
                quadrature_refinements=[row for row in refinements if row is not None],
                sampled_positive_stiffness=bool(positive_sample), harmonic_IR_variance=variance,
                limitations=['Quadrature/q refinements are estimates, not certified continuum error bounds.',
                             'Axis and two oblique directions do not prove angular-uniform stiffness.',
                             'Pole pencil is a subthreshold low-frequency derivative expansion, not the full retarded spectrum.',
                             'Berry response is charge/gyroscopic dynamics, not a spin derivation.',
                             'B is the sea-induced vacuum flavor-charge form about a zero-angular-velocity orbital background; total bosonic+matter charge constraints require their own background.',
                             'No nonlinear quantum ordering, canonical promotion, or joined campaign completion.'],
                checks=checks, all_checks_pass=all(checks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=None)
    parser.add_argument('--rule', choices=['midpoint', 'split_gauss'], default='midpoint')
    parser.add_argument('--q', type=float, nargs='+', default=[.16, .08, .04])
    parser.add_argument('--a', type=float, default=1.)
    parser.add_argument('--m', type=float, default=.2)
    parser.add_argument('--r', type=float, default=1.)
    parser.add_argument('--v', type=float, default=1.)
    parser.add_argument('--kappa', type=float, default=1.)
    parser.add_argument('--g0', type=float, default=1.)
    parser.add_argument('--M', type=float, default=1.)
    parser.add_argument('--J', type=float, default=0.)
    parser.add_argument('--batch', type=int, default=192)
    parser.add_argument('--ir-cutoff', type=float, default=.04)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.orders is None:
        args.orders = [6, 12, 24] if args.rule == 'midpoint' else [8, 10]
    result = run(args)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

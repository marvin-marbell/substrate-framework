#!/usr/bin/env python3
"""Same-law finite Bloch/Galerkin photon leg and charged dynamical response.

No native compact phase, continuum Compton amplitude, or full-R ground claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from scipy.integrate import simpson
from scipy.sparse.linalg import LinearOperator, expm_multiply, gmres

from noncompact_native_current import Coupled, accept, adverse, source_operators
from noncompact_photon_spectrum import full_source_measure, oscillator_positive_control
from original_charged_principal import original_frame
from rotation_band import RotationBand

F = np.array([1., -1., 0.]) / math.sqrt(2)
HARMONIC = np.array([1., 1., -2.]) / math.sqrt(6)
INDEPENDENT = (F + HARMONIC) / math.sqrt(2)
GAUGE = np.ones(3)


def encoded(value) -> Any:
    a = np.asarray(value)
    if np.iscomplexobj(a):
        if a.ndim == 0:
            return dict(real=float(a.real), imag=float(a.imag))
        return [encoded(x) for x in a]
    if a.ndim == 0:
        return a.item()
    return a.tolist()


def shifted(model, energy, z):
    constructor = cast(Callable[..., Any], LinearOperator)
    return constructor((model.size, model.size), dtype=np.complex128,
                       matvec=lambda v: model.action(v) - (energy + z) * np.asarray(v).ravel(),
                       rmatvec=lambda v: model.action(v) - (energy + z.conjugate()) * np.asarray(v).ravel())


def solve_resolvent(model, energy, z, rhs):
    solver = cast(Callable[..., Any], gmres)
    operator = shifted(model, energy, z)
    answer, status = solver(operator, rhs, rtol=2e-11, atol=1e-13,
                            restart=100, maxiter=1000)
    if status != 0:
        raise AssertionError(f'actual complex resolvent did not converge: {status}')
    residual = float(np.linalg.norm(operator @ answer - rhs))
    accept(residual, 'actual complex-frequency quantum resolvent', 2e-8)
    return answer, residual


def coordinate(model, vector):
    """P_n Q_f P_n; discarded Q_f output has EXACT one-level leakage."""
    wave = np.asarray(vector).reshape(model.n, model.l, model.d)
    out = np.zeros_like(wave)
    scale = math.sqrt(model.UE / model.omega)
    for level in range(model.n - 1):
        coefficient = scale * math.sqrt(level + 1)
        out[level + 1] += coefficient * wave[level]
        out[level] += coefficient * wave[level + 1]
    leakage = scale * math.sqrt(model.n) * float(np.linalg.norm(wave[-1]))
    return out.ravel(), leakage


def krylov_measure(model, E0, q, maximum):
    """Positive actual-H Krylov spectral measure; Ritz nodes are NOT poles."""
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise AssertionError('Restored-coordinate photon has no connected quantum weight')
    columns = [q / norm]
    for _ in range(1, min(maximum, model.size)):
        new = model.action(columns[-1]) - E0 * columns[-1]
        for _pass in range(2):
            basis = np.column_stack(columns)
            new -= basis @ (basis.conj().T @ new)
        length = float(np.linalg.norm(new))
        if length < 1e-12:
            break
        columns.append(new / length)
    V = np.column_stack(columns)
    HV = np.column_stack([model.action(v) - E0 * v for v in columns])
    T = V.conj().T @ HV
    accept(np.linalg.norm(T - T.conj().T), 'same-H photon spectral compression Hermiticity', 2e-8)
    T = (T + T.conj().T) / 2
    nodes, U = np.linalg.eigh(T)
    weights = norm**2 * abs(U[0])**2
    accept(abs(sum(weights) - norm**2), 'actual connected photon spectral total weight', 2e-8)
    return V, HV, T, nodes, weights, norm


def photon_supplier(neutral, E0, psi, values, vectors, eta, krylov_size, spectral_max_dimension, photon_bands):
    Qpsi, leakage = coordinate(neutral, psi)
    mean = np.vdot(psi, Qpsi)
    q = Qpsi - mean * psi
    V, HV, T, nodes, weights, norm = krylov_measure(neutral, E0, q, krylov_size)
    overlaps = vectors.conj().T @ q
    low_weights = abs(overlaps)**2
    excitation = values - E0
    indices, full_values, X, bands, certificate = full_source_measure(neutral, E0, q, spectral_max_dimension)
    excited = [row for row in bands if row['energy_interval'][0] > 1e-7]
    ranked = sorted(excited, key=lambda row: row['residue'], reverse=True)
    selected = ranked[:photon_bands]
    if len(selected) != photon_bands:
        raise AssertionError('Requested more photon bands than the complete finite excited spectrum')
    for row in selected:
        cluster_vectors = np.zeros((neutral.size, row['multiplicity']), complex)
        cluster_vectors[indices] = X[:, row['begin']:row['end']]
        residuals = [float(np.linalg.norm(neutral.action(u) - full_values[j] * u))
                     for j, u in zip(range(row['begin'], row['end']), cluster_vectors.T)]
        accept(max(residuals), 'resolved photon eigencluster residual in WHOLE H', 2e-8)
        row['whole_H_eigenpair_residuals'] = residuals
        row['enlarged_basis_residuals'] = [
            neutral.full_residual_bound(u, float(full_values[j]))
            for j, u in zip(range(row['begin'], row['end']), cluster_vectors.T)]
    dominant = selected[0]
    competitors = ranked[1:]
    dominance_margin = dominant['residue'] - dominant['residue_error_upper'] - max(
        (row['residue'] + row['residue_error_upper'] for row in competitors), default=0.)
    if dominance_margin <= 0:
        raise AssertionError('Finite residual bounds do not distinguish a unique dominant Q eigencluster')
    probes = [dict(z=complex(scale * row['energy'], eta), band_index=row['index'],
                   energy=row['energy'], energy_interval=row['energy_interval'],
                   residue=row['residue'], residue_error_upper=row['residue_error_upper'],
                   fraction=row['fraction'], fraction_error_upper=row['fraction_error_upper'])
              for row in selected for scale in (1., .6)]
    # Retain earlier low poles as low spectral features, never the dominant leg.
    low_bands = [dict(energy=row['energy'], residue=row['residue'], fraction=row['fraction'])
                 for row in bands if 1e-7 < row['energy'] < float(excitation[-1]-1e-7)]
    iteration_measures = []
    for size in sorted(set((max(4, krylov_size // 2), krylov_size))):
        if size == krylov_size:
            Vk, HVk, Tk, nk, wk = V, HV, T, nodes, weights
        else:
            Vk, HVk, Tk, nk, wk, _ = krylov_measure(neutral, E0, q, size)
        _, Uk = np.linalg.eigh(Tk)
        candidate = int(np.argmax(wk))
        witness = Vk @ Uk[:, candidate]
        eigen_residual = float(np.linalg.norm(HVk @ Uk[:, candidate] - nk[candidate] * witness))
        iteration_measures.append(dict(requested_size=size, actual_size=len(nk),
                                       candidate_energy=float(nk[candidate]),
                                       candidate_weight=float(wk[candidate]),
                                       candidate_full_H_eigen_residual=eigen_residual,
                                       candidate_energy_minus_resolved_dominant=float(nk[candidate]-dominant['energy']),
                                       candidate_weight_minus_resolved_dominant=float(wk[candidate]-dominant['residue']),
                                       nodes=nk.tolist(), weights=wk.tolist(),
                                       status='quadrature candidates only; complete full-H diagonalization supplies poles'))
    Vwrong, HVwrong, Twrong, nwrong, wwrong, _ = krylov_measure(neutral, E0, q, 4)
    _, Uwrong = np.linalg.eigh(Twrong)
    jwrong = int(np.argmax(wwrong))
    wrong_residual = float(np.linalg.norm(HVwrong @ Uwrong[:, jwrong] - nwrong[jwrong]*(Vwrong @ Uwrong[:, jwrong])))
    wrong_ritz = adverse(wrong_residual, 'resolved photon eigencluster residual in WHOLE H', 2e-8)
    spectral = []
    full_weights = abs(X.conj().T @ q[indices])**2
    bare_errors = []
    for probe in probes:
        z = probe['z']
        coefficients = np.linalg.solve(T - z * np.eye(len(T)), V.conj().T @ q)
        approximate = V @ coefficients
        residual = float(np.linalg.norm(HV @ coefficients - z * approximate - q))
        scalar_error_bound = norm * residual / abs(z.imag)
        exact, solve_error = solve_resolvent(neutral, E0, z, q)
        measured = np.vdot(q, exact)
        compressed = np.vdot(q, approximate)
        # Complete eigencluster transform is independent of Krylov quadrature.
        outside_weight = certificate['Q_outside_sector_weight']
        full_transform = sum(full_weights / (full_values - E0 - z))
        complete_tolerance = max(3e-8, certificate['finite_H_backward_error_upper'] * norm**2 / abs(z.imag)**2
                                 + norm * solve_error / abs(z.imag) + outside_weight / abs(z.imag))
        accept(abs(measured - full_transform), 'complete eigencluster measure versus independent full-H resolvent', complete_tolerance)
        tolerance = max(3e-8, scalar_error_bound + norm * solve_error / abs(z.imag))
        accept(abs(measured - compressed), 'same-state photon measure versus independent full-H resolvent', tolerance)
        # The discarded oscillator output is reported separately; the target here
        # is explicitly P_n Q P_n, not an asserted full unbounded-Q limit.
        bare = (neutral.UE / neutral.omega) / (neutral.omega - z)
        bare_errors.append((abs(measured - bare), complete_tolerance))
        spectral.append(dict(z=encoded(z), band_index=probe['band_index'],
                             spectral_resolvent=encoded(compressed), actual_resolvent=encoded(measured),
                             complete_eigencluster_resolvent=encoded(full_transform),
                             complete_eigencluster_acceptance_tolerance=complete_tolerance,
                             Krylov_residual=residual, scalar_error_bound=scalar_error_bound,
                             numerical_acceptance_tolerance=tolerance))
    best = max(bare_errors, key=lambda pair: pair[0] / pair[1])
    wrong_bare = adverse(best[0], 'complete eigencluster measure versus independent full-H resolvent', best[1])
    return dict(Q_expectation=encoded(mean), connected_projected_Q_norm_squared=norm**2,
                exact_unbounded_Q_output_leakage_norm=leakage,
                full_Q_norm_squared=float(np.vdot(Qpsi, Qpsi).real + leakage**2),
                actual_low_spectrum=excitation.tolist(), actual_low_Q_weights=low_weights.tolist(),
                earlier_low_spectral_features=low_bands,
                complete_Q_sector_spectrum=full_values.tolist(),
                complete_Q_sector_bands=bands,
                dominant_bands=selected, dominant_certified_residue_margin=dominance_margin,
                dominant_band_energy=dominant['energy'], dominant_band_residue=dominant['residue'],
                dominant_band_fraction=dominant['fraction'],
                complete_finite_H_certificate=certificate,
                unresolved_after_low_solve_projected_Q_weight=max(0., norm**2-float(sum(low_weights))),
                unresolved_after_complete_sector_projected_Q_weight=max(0., norm**2-certificate['complete_spectral_weight']),
                oscillator_positive_control=oscillator_positive_control(neutral.UE, neutral.omega),
                unconverged_Ritz_pole_same_residual_assertion=wrong_ritz,
                Krylov_iteration_diagnostics=iteration_measures,
                Krylov_nodes=nodes.tolist(), Krylov_weights=weights.tolist(),
                Krylov_scope='positive actual-H compression; nodes alone are not interacting poles',
                spectral_resolvent_discriminants=spectral,
                bare_oscillator_only_comparison=dict(omega=neutral.omega, residue=neutral.UE / neutral.omega),
                omitted_interacting_photon_leg=wrong_bare), probes


def complete_source_charged_ground(model, sector):
    for count in (6, 10, 18, 30):
        available=int(np.count_nonzero(model.triality()==sector))
        values, vectors, residuals = model.source_spectrum(min(count, available - 2), sector)
        cluster = np.flatnonzero(values - values[0] < 1e-7)
        if len(cluster) < len(values):
            G = np.linalg.qr(vectors[:, cluster])[0]
            accept(np.linalg.norm(G.conj().T @ G - np.eye(len(cluster))),
                   'complete orthonormal charged degenerate projector', 2e-8)
            return float(values[0]), G, float(values[len(cluster)] - values[0]), residuals
    raise AssertionError('Charged ground degeneracy reaches every requested spectral edge')


def source_density(neutral, charged, psi, G, E0):
    B = source_operators(neutral, charged)
    expected = np.diag(4 - neutral.nx)
    actual = sum(t.conj().T @ t for t in B[0])
    accept(np.linalg.norm(actual - expected), 'complete-reference source norm operator', 2e-8)
    missing = adverse(np.linalg.norm(B[0][0].conj().T @ B[0][0] - expected),
                      'complete-reference source norm operator', 2e-8)
    wave = neutral.reshape(psi)
    sources = np.array([[ (wave @ t.T).ravel() for t in label] for label in B])
    gram = np.array([[np.vdot(a, b) for b in sources] for a in sources])
    accept(np.linalg.norm(gram - 2 * np.eye(4)), 'same-neutral-state complete-reference complex quartet Gram', 2e-7)
    energy = np.array([[sum(np.vdot(sa, charged.action(sb) - E0 * sb)
                                  for sa, sb in zip(a, b)) / 2
                       for b in sources] for a in sources])
    accept(np.linalg.norm(energy - energy.conj().T), 'actual charged complex source Rayleigh matrix', 2e-7)
    alpha = np.array([1., .3j, -.2 + .4j, .6], complex)
    alpha /= np.linalg.norm(alpha)
    created = np.einsum('a,ars->sr', alpha, sources) / math.sqrt(2)
    accept(abs(np.vdot(created, created).real - 1), 'full-reference actual complex source normalization', 2e-7)
    amplitudes = G.conj().T @ created
    reduced = amplitudes @ amplitudes.conj().T
    weight = float(np.trace(reduced).real)
    if weight < 1e-10:
        raise AssertionError('Actual complete-reference source has no charged-ground spectral weight')
    rho = reduced / weight
    accept(abs(np.trace(rho) - 1), 'actual source-conditioned charged stationary density', 2e-8)
    return rho, dict(complete_reference=4, complex_coefficients=encoded(alpha),
                    actual_source_Gram=encoded(gram), actual_source_Rayleigh_matrix=encoded(energy),
                    actual_source_Rayleigh_excess=float(np.vdot(alpha, energy @ alpha).real),
                    charged_ground_source_weight=weight, charged_ground_density_matrix=encoded(rho),
                    charged_ground_source_amplitudes=encoded(amplitudes), omitted_reference=missing)


def jets(model, G, a, b):
    Ja = np.column_stack([model.current(u, a) for u in G.T])
    Jb = np.column_stack([model.current(u, b) for u in G.T])
    contact = G.conj().T @ np.column_stack([model.contact(u, a, b) for u in G.T])
    return Ja, Jb, contact


def response(model, Ec, G, a, b, z):
    Ja, Jb, contact = jets(model, G, a, b)
    minus = np.column_stack([solve_resolvent(model, Ec, z, v)[0] for v in Jb.T])
    plus = np.column_stack([solve_resolvent(model, Ec, -z, v)[0] for v in Ja.T])
    first = Ja.conj().T @ minus
    second = Jb.conj().T @ plus
    return contact - first - second, contact, first, second


def trace_observable(matrix, rho):
    return complex(np.trace(rho @ matrix))


def quantum_time_response(model, Ec, G, rho, a, b, z, intervals):
    """Independent actual-H retarded Kubo integral, including measured contact."""
    Ja, Jb, contact = jets(model, G, a, b)
    operator = shifted(model, Ec, 0j)
    exact_trace = (model.d * sum(model.phenergy) + model.phdim *
                   (sum(model.coulomb) + np.trace(model.onsite).real) - Ec * model.size)
    end = 16 / z.imag
    grid = np.linspace(0, end, intervals + 1)
    integrand = np.empty(intervals + 1, complex)
    exponential = cast(Callable[..., Any], expm_multiply)
    # Advance bounded-size chunks from their previous endpoint, rather than
    # repropagating from zero or allocating the entire time history.
    forward_state = Jb
    backward_state = Ja
    for start in range(0, intervals, 32):
        stop = min(start + 32, intervals)
        local = grid[start:stop + 1]
        duration = float(local[-1] - local[0])
        forward = exponential(-1j * operator, forward_state, start=0.,
                              stop=duration, num=len(local), endpoint=True,
                              traceA=-1j * exact_trace)
        backward = exponential(1j * operator, backward_state, start=0.,
                               stop=duration, num=len(local), endpoint=True,
                               traceA=1j * exact_trace)
        forward_state = forward[-1]
        backward_state = backward[-1]
        for offset, time in enumerate(local):
            A = Ja.conj().T @ forward[offset]
            B = Jb.conj().T @ backward[offset]
            integrand[start + offset] = np.exp(1j * z * time) * trace_observable(A - B, rho)
    coarse = trace_observable(contact, rho) - 1j * simpson(integrand[::2], x=grid[::2])
    fine = trace_observable(contact, rho) - 1j * simpson(integrand, x=grid)
    tail = 2 * np.linalg.norm(Ja) * np.linalg.norm(Jb) * math.exp(-z.imag * end) / z.imag
    # Difference is a reported quadrature diagnostic, not a theorem for Simpson.
    tolerance = max(2e-7, 4 * abs(fine - coarse) + tail)
    return complex(fine), dict(time_end=end, intervals=intervals,
                              coarse_fine_difference=float(abs(fine - coarse)),
                              rigorous_exponential_tail_upper=float(tail),
                              numerical_acceptance_tolerance=float(tolerance),
                              actual_quantum_time_response=encoded(fine))


def actual_peierls_discriminant(model, G, rho, direction) -> dict[str, Any]:
    delta = 2e-5
    up = Coupled(model.n, model.M, model.zeta, model.occupation, model.UE,
                 model.K, model.frame, delta * direction)
    down = Coupled(model.n, model.M, model.zeta, model.occupation, model.UE,
                   model.K, model.frame, -delta * direction)
    current = np.column_stack([model.current(u, direction) for u in G.T])
    measured = np.column_stack([(up.action(u) - down.action(u)) / (2 * delta) for u in G.T])
    error = accept(np.linalg.norm(current - measured), 'same charged H actual shared-photon Peierls jet', 2e-7)
    frozen = adverse(np.linalg.norm(current), 'same charged H actual shared-photon Peierls jet', 2e-7)
    contact = G.conj().T @ np.column_stack([model.contact(u, direction, direction) for u in G.T])
    measured_contact = G.conj().T @ np.column_stack([
        (up.action(u) + down.action(u) - 2 * model.action(u)) / delta**2 for u in G.T])
    accept(np.linalg.norm(contact - measured_contact), 'same charged H actual Peierls contact', 2e-5)
    return dict(delta=delta, current_jet_error=error, contact_jet_error=float(np.linalg.norm(contact - measured_contact)),
                physical_current_norm=float(np.linalg.norm(current)),
                source_conditioned_contact=encoded(trace_observable(contact, rho)), frozen_hopping=frozen)


def ward(model, Ec, G, rho, z):
    density = np.column_stack([(model.reshape(u) * model.nx[None, :]).ravel() for u in G.T])
    J = np.column_stack([model.current(u, GAUGE) for u in G.T])
    delta_density = np.column_stack([model.action(v) - Ec * v for v in density.T])
    continuity = accept(np.linalg.norm(J + 1j * delta_density), 'actual scalar-density continuity sign', 2e-7)
    Rminus = np.column_stack([solve_resolvent(model, Ec, z, v)[0] for v in density.T])
    Rplus = np.column_stack([solve_resolvent(model, Ec, -z, v)[0] for v in density.T])
    Snn = density.conj().T @ (Rminus + Rplus)
    Pi, _, _, _ = response(model, Ec, G, GAUGE, GAUGE, z)
    temporal = accept(np.linalg.norm(Pi + z**2 * Snn), 'actual temporal-density two-channel Ward sign', 2e-7)
    Ja = np.column_stack([model.current(u, F) for u in G.T])
    Ra = np.column_stack([solve_resolvent(model, Ec, -z, v)[0] for v in Ja.T])
    mixed_density = Ja.conj().T @ Rminus + density.conj().T @ Ra
    mixed, _, _, _ = response(model, Ec, G, F, GAUGE, z)
    scalar = accept(np.linalg.norm(mixed - 1j * z * mixed_density), 'actual mixed scalar Ward sign', 2e-7)
    return dict(continuity_error=continuity, temporal_Ward_error=temporal,
                mixed_scalar_Ward_error=scalar, density_kernel=encoded(-Snn),
                source_conditioned_longitudinal_response=encoded(trace_observable(Pi, rho)),
                sign_convention='J_g=i[N,H]; Pi_gg=-z^2 S_NN; Pi_fg=i z (S_fN+S_Nf)')


def case(args):
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    selected, frame = original_frame(band, 5, np.zeros(3), True)
    neutral = Coupled(args.n, args.M, args.zeta, 4, args.UE, args.K, frame)
    charged = Coupled(args.n, args.M, args.zeta, 5, args.UE, args.K, frame)
    values, vectors, neutral_residuals = neutral.solve(args.neutral_states)
    # Complex ARPACK may return nonorthogonal vectors within a degenerate band.
    # A spectral measure requires its projector, not their raw squared overlaps.
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and abs(values[end] - values[begin]) < 1e-7:
            end += 1
        vectors[:, begin:end] = np.linalg.qr(vectors[:, begin:end])[0]
        begin = end
    accept(np.linalg.norm(vectors.conj().T @ vectors - np.eye(len(values))),
           'actual neutral low-band complete orthonormal projectors', 2e-7)
    neutral_residuals = [float(np.linalg.norm(neutral.action(vectors[:, j]) - values[j] * vectors[:, j]))
                         for j in range(len(values))]
    accept(max(neutral_residuals), 'actual neutral low-band projector eigenpair residual', 2e-7)
    E0 = float(values[0]); psi = vectors[:, 0]
    if values[1] - E0 < 1e-7:
        raise AssertionError('Neutral degeneracy requires a complete neutral projector')
    photon, frequencies = photon_supplier(neutral, E0, psi, values, vectors, args.eta, args.krylov,
                                           args.spectral_max_dimension, args.photon_bands)
    point_source=np.array([neutral.reshape(psi)@t.T for t in source_operators(neutral,charged)[0]])
    sector,sector_weights=charged.source_triality(point_source)
    Ec, G, gap, charged_residuals = complete_source_charged_ground(charged,sector)
    rho, source = source_density(neutral, charged, psi, G, E0)
    global_values,global_vectors,_=charged.solve(6)
    global_cluster=np.flatnonzero(global_values-global_values[0]<1e-7)
    global_G=np.linalg.qr(global_vectors[:,global_cluster])[0]
    try:
        source_density(neutral,charged,psi,global_G,E0)
    except AssertionError as exc:
        if 'no charged-ground spectral weight' not in str(exc):raise
        unrelated_global_ground=str(exc)
    else:
        unrelated_global_ground='Global ground also has prepared-source weight at these inputs; no mismatch claimed'
    outputs = []
    for probe in frequencies:
        z = probe['z']
        for label, a, b in [('restored_restored', F, F), ('restored_independent', F, INDEPENDENT)]:
            matrix, contact, first, second = response(charged, Ec, G, a, b, z)
            actual = trace_observable(matrix, rho)
            independent, time_data = quantum_time_response(charged, Ec, G, rho, a, b, z, args.intervals)
            tolerance = time_data['numerical_acceptance_tolerance']
            accept(abs(actual - independent), 'same charged state finite-frequency response versus actual quantum time consumer', tolerance)
            omissions = {}
            for name, wrong in [('second_ordered_channel', contact - first),
                                ('contact', -first - second),
                                ('frozen_hopping', np.zeros_like(matrix))]:
                omissions[name] = adverse(abs(trace_observable(wrong, rho) - independent),
                                          'same charged state finite-frequency response versus actual quantum time consumer', tolerance)
            photon_factor = probe['residue'] * float(F @ a) * float(F @ b)
            outputs.append(dict(source=label, incoming_edge_source=a.tolist(), outgoing_edge_source=b.tolist(),
                                photon_band_index=probe['band_index'], photon_band_energy=probe['energy'],
                                photon_band_residue=probe['residue'], photon_band_fraction=probe['fraction'],
                                photon_band_energy_interval=probe['energy_interval'],
                                photon_band_residue_error_upper=probe['residue_error_upper'],
                                photon_band_fraction_error_upper=probe['fraction_error_upper'],
                                photon_band_matrix_residue_error_upper=float(abs(float(F @ a)*float(F @ b))
                                                                            * probe['residue_error_upper']*np.linalg.norm(matrix)),
                                z=encoded(z), actual_response_matrix=encoded(matrix),
                                actual_contact_matrix=encoded(contact), actual_first_ordering=encoded(first),
                                actual_second_ordering=encoded(second), source_conditioned_response=encoded(actual),
                                photon_band_external_leg_factor=photon_factor,
                                photon_band_response_matrix=encoded(photon_factor * matrix),
                                actual_quantum_time_discriminant=time_data, omissions=omissions))
    # Zero is rejected by the SAME independent time assertion above, not by an
    # imaginary one-sided static cross term or harmonic translation zero.
    physical = actual_peierls_discriminant(charged, G, rho, F)
    charged_Q = [coordinate(charged, u) for u in G.T]
    QG = np.column_stack([pair[0] for pair in charged_Q])
    JfG = np.column_stack([charged.current(u, F) for u in G.T])
    JhG = np.column_stack([charged.current(u, HARMONIC) for u in G.T])
    physical.update(charged_Q_cluster_matrix=encoded(G.conj().T @ QG),
                    charged_Q_exact_output_leakage_norms=[pair[1] for pair in charged_Q],
                    restored_current_cluster_matrix=encoded(G.conj().T @ JfG),
                    harmonic_current_cluster_matrix=encoded(G.conj().T @ JhG),
                    actual_current_transition_Gram_ff=encoded(JfG.conj().T @ JfG),
                    actual_current_transition_Gram_fh=encoded(JfG.conj().T @ JhG),
                    actual_current_transition_Gram_hh=encoded(JhG.conj().T @ JhG))
    return dict(law='noncompact quadratic curl, exact microscopic exp(iA), reduced Gauss Coulomb sector',
                UE=args.UE, K=args.K, oscillator_levels=args.n, harmonic_labels=charged.ell.tolist(),
                Bloch_label=args.zeta, broadening=args.eta, neutral_dimension=neutral.size,
                charged_dimension=charged.size, frame_native_N5_Gram=selected['G'].tolist(),
                E0=E0, Ec=Ec, charged_multiplicity=G.shape[1], charged_next_gap=gap,
                source_triality=sector,source_triality_weights=sector_weights,
                unconditioned_global_charged_Ec=float(global_values[0]),
                unrelated_global_ground_same_source_assertion=unrelated_global_ground,
                neutral_eigen_residuals=neutral_residuals, charged_eigen_residuals=charged_residuals,
                neutral_full_comparison_residual=neutral.full_residual_bound(psi, E0),
                charged_full_comparison_residuals=[charged.full_residual_bound(u, Ec) for u in G.T],
                photon_supplier=photon, actual_complete_reference_source=source,
                physical_shared_photon_current=physical, finite_frequency_responses=outputs,
                scalar_temporal_Ward=ward(charged, Ec, G, rho, frequencies[0]['z']),
                scope='finite two-vertex three-parallel-edge rank-one-curl graph; finite Hermite/Fourier Galerkin; selected Bloch fibre; actual SOURCE-ADDRESSED conserved-sector complete degenerate charged cluster, not unconditioned global ground/reference-free mass; finite positive broadening; not nonforward spatial Compton',
                remaining_debts=['unbounded Q source/operator and normal/Born-Huang domain transfer',
                                 'photon tails and oscillator/harmonic cutoffs, Krylov and time quadrature convergence',
                                 'Bloch direct-integral infimum and normalizable harmonic wavepacket',
                                 'full native-field/heavy/colour Gauss supplier instead of frozen N5 frame',
                                 'thermodynamic nonlinear phase, mobile isolated charged sector, photon asymptotic scattering',
                                 '3D continuum, Lorentz/spin identification, zero-broadening and radiative limits'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=7)
    parser.add_argument('--M', type=int, default=4)
    parser.add_argument('--UE', type=float, default=.37)
    parser.add_argument('--K', type=float, default=.9)
    parser.add_argument('--zeta', type=float, default=0.)
    parser.add_argument('--eta', type=float, default=.5)
    parser.add_argument('--neutral-states', type=int, default=16)
    parser.add_argument('--krylov', type=int, default=100)
    parser.add_argument('--spectral-max-dimension', type=int, default=3000,
                        help='hard bound for COMPLETE Q-sector dense solve; never substitute incomplete Ritz poles')
    parser.add_argument('--photon-bands', type=int, default=1,
                        help='number of largest-residue complete eigenclusters joined to charged time response')
    parser.add_argument('--cutoff-grid', nargs='+', metavar='N:M',
                        help='run complete consumers at each explicit cutoff; report dominant-band drift')
    parser.add_argument('--intervals', type=int, default=1024)
    args = parser.parse_args()
    if args.n < 3 or args.M < 2 or min(args.UE, args.K, args.eta) <= 0:
        parser.error('require n>=3, M>=2 and positive UE,K,eta')
    if args.intervals < 16 or args.intervals % 4 or args.neutral_states < 4 or args.krylov < 4:
        parser.error('intervals must be a multiple of four >=16; spectral sizes >=4')
    if args.photon_bands < 1 or args.spectral_max_dimension < 4:
        parser.error('positive photon-band count and spectral-max-dimension>=4 required')
    result: dict[str, Any]
    if args.cutoff_grid:
        cuts = []
        for entry in args.cutoff_grid:
            try:
                n, M = map(int, entry.split(':'))
            except ValueError:
                parser.error('cutoff-grid entries must be N:M')
            if n < 3 or M < 2:
                parser.error('cutoff-grid requires N>=3,M>=2')
            cuts.append((n, M))
        runs = []
        for args.n, args.M in cuts:
            runs.append(case(args))
        convergence = []
        for previous, current in zip(runs, runs[1:]):
            a, b = previous['photon_supplier'], current['photon_supplier']
            convergence.append(dict(from_cutoff=[previous['oscillator_levels'], len(previous['harmonic_labels'])//2],
                                    to_cutoff=[current['oscillator_levels'], len(current['harmonic_labels'])//2],
                                    dominant_energy_change=b['dominant_band_energy']-a['dominant_band_energy'],
                                    dominant_residue_change=b['dominant_band_residue']-a['dominant_band_residue'],
                                    dominant_fraction_change=b['dominant_band_fraction']-a['dominant_band_fraction'],
                                    scope='empirical cutoff drift, not rigorous infinite-law convergence; dominant identity may change'))
        result = dict(runs=runs, cutoff_convergence=convergence)
    else:
        result = case(args)
    directory = Path(__file__).resolve().parent
    result['source_sha256'] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                              for name in ['noncompact_photon_response.py', 'noncompact_photon_response.md',
                                           'noncompact_photon_spectrum.py', 'noncompact_native_current.py',
                                           'noncompact_coulomb_carrier.md']}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

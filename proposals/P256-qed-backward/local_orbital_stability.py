#!/usr/bin/env python3
"""Unrestricted spatial orbital Hessian for the actual periodic half-filled sea.

The companion note derives the finite-box theorem. Run only at the parent-owned
verification boundary; the default constructs and diagonalizes all 60*3**3 modes.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from scipy.linalg import expm, orth

from pauli_compton import EdgeWords, block_terms
from rotation_band import RotationBand
from sea_selection import SELECTED, coefficients, gram, matrix_unit_basis, rank_report
from soft_budget import select_gram


def periodic_grid(size):
    return np.array(list(itertools.product(range(size), repeat=3)), dtype=int)


def scalar_paths(band):
    """Extract, rather than duplicate, RotationBand's directed scalar paths."""
    onsite, paths = band.couplings()
    onsite_scalar = float(np.trace(onsite @ SELECTED[0]).real / 4)
    result = []
    for displacement, matrix in paths:
        scalars = np.einsum('mij,ji->m', SELECTED, matrix) / 4
        if np.linalg.norm(matrix - np.einsum('m,mij->ij', scalars, SELECTED)) > 1e-12:
            raise ValueError('Scalar path is outside the four primitive orbital labels')
        result.append((displacement, scalars))
    return onsite_scalar, result


def local_hamiltonian(fields, band, sites, size):
    """Endpoints only, including two-step paths; add every oriented path and adjoint."""
    onsite, paths = scalar_paths(band)
    count = len(sites)
    index = {tuple(site): number for number, site in enumerate(sites)}
    result = np.zeros((4 * count, 4 * count), dtype=complex)
    for x, site in enumerate(sites):
        sx = slice(4 * x, 4 * x + 4)
        result[sx, sx] += onsite * fields[x, 0]
        for displacement, scalars in paths:
            y = index[tuple((site + displacement) % size)]
            sy = slice(4 * y, 4 * y + 4)
            coupling = np.einsum('m,mij->ij', scalars, (fields[x] + fields[y]) / 2)
            result[sx, sy] += coupling
            result[sy, sx] += coupling.conj().T
    return result


def alignment_laplacian(sites, size):
    """One positive-axis nearest bond per site/axis, with periodic multiplicity."""
    index = {tuple(site): number for number, site in enumerate(sites)}
    laplacian = np.zeros((len(sites), len(sites)))
    for x, site in enumerate(sites):
        for axis in np.eye(3, dtype=int):
            y = index[tuple((site + axis) % size)]
            laplacian[x, x] += 1
            laplacian[y, y] += 1
            laplacian[x, y] -= 1
            laplacian[y, x] -= 1
    return laplacian


def field_matrices(coordinates, basis):
    return np.einsum('xma,aij->xmij', coordinates, basis)


def elastic_hessian(frame, basis, stiffness, preferred):
    origin = np.einsum('mij,aji->ma', frame, basis).real / 4
    elementary = np.eye(60).reshape(60, 4, 15)
    dg = np.einsum('tma,na->tmn', elementary, origin)
    dg += dg.transpose(0, 2, 1)
    mismatch = gram(frame) - preferred * np.eye(4)
    return stiffness * dg.reshape(60, 16) @ dg.reshape(60, 16).T + np.kron(2 * stiffness * mismatch, np.eye(15))


def uniform_data(band, size, stiffness, preferred, basis):
    sites = periodic_grid(size)
    momenta = 2 * np.pi * sites / (size * band.a)
    x = coefficients(band, momenta)
    selected = select_gram(x, stiffness, preferred)
    frame = np.sqrt(np.diag(selected['G']))[:, None, None] * SELECTED
    h = np.einsum('pm,mij->pij', x, frame)
    values, vectors = np.linalg.eigh(h)
    orbit = np.array([1j * (generator @ frame - frame @ generator) for generator in basis])
    orbit_columns = np.einsum('tmij,aji->mat', orbit, basis).real.reshape(60, 15) / 4
    orbit_basis = orth(orbit_columns)
    elastic = elastic_hessian(frame, basis, stiffness, preferred)
    return dict(sites=sites, x=x, selected=selected, frame=frame, h=h,
                values=values, vectors=vectors, orbit=orbit_basis, elastic=elastic)


def fourier_sea_hessian(data, q, size, basis):
    """Exact complex 60x60 Hessian block; both occupied/empty orientations."""
    count = len(data['sites'])
    indices = np.ravel_multi_index(((data['sites'] + q) % size).T, (size,) * 3)
    t = (data['x'] + data['x'][indices]) / 2
    result = np.zeros((60, 60), dtype=complex)
    for target_sector, source_sector in ((slice(2, 4), slice(0, 2)),
                                          (slice(0, 2), slice(2, 4))):
        target = data['vectors'][indices, :, target_sector]
        source = data['vectors'][:, :, source_sector]
        transitions = np.einsum('pie,aij,pjo->paeo', target.conj(), basis, source)
        features = np.einsum('pm,paeo->mapeo', t, transitions).reshape(60, count, 2, 2)
        target_e = data['values'][indices, target_sector]
        source_e = data['values'][:, source_sector]
        denominators = np.abs(target_e[:, :, None] - source_e[:, None, :])
        weighted = (features / np.sqrt(denominators)[None]).reshape(60, -1)
        result -= weighted.conj() @ weighted.T / count
    return (result + result.conj().T) / 2


def reconstruct_hessian(blocks, sites, size):
    """Inverse unitary spatial DFT, with untouched 60-coordinate fibers."""
    difference = sites[:, None, :] - sites[None, :, :]
    phases = np.exp(2j * np.pi * np.einsum('qi,xyi->qxy', sites, difference) / size)
    full = np.einsum('qxy,qab->xayb', phases, blocks) / len(sites)
    imaginary_residual = float(np.max(np.abs(full.imag)))
    real = full.real.reshape(60 * len(sites), 60 * len(sites))
    return (real + real.T) / 2, imaginary_residual


def sea_bilinear(left, right, values, vectors):
    half = len(values) // 2
    eo_left = vectors[:, half:].conj().T @ left @ vectors[:, :half]
    eo_right = vectors[:, half:].conj().T @ right @ vectors[:, :half]
    denominator = values[half:, None] - values[None, :half]
    return -2 * float(np.sum(eo_left.conj() * eo_right / denominator).real)


def energy(fields, band, sites, size, stiffness, preferred, alignment, laplacian):
    eigenvalues = np.linalg.eigvalsh(local_hamiltonian(fields, band, sites, size))
    sea = float(eigenvalues[:len(eigenvalues) // 2].sum())
    grams = np.einsum('xmij,xnji->xmn', fields, fields).real / 4
    elastic = .5 * stiffness * float(np.sum((grams - preferred * np.eye(4))**2))
    coordinate_norm = np.einsum('xmij,ymji->xy', fields, fields).real / 4
    aligned = .5 * alignment * float(np.sum(laplacian * coordinate_norm))
    return sea + elastic + aligned


def primitive_sector_probe(matrix):
    """Actual edge-qubit words on one four-orbital block, not a CAR constructor."""
    words = EdgeWords(5, [(0, 1), (1, 2), (2, 3), (0, 4)])
    states = [words.tree_basis_state(pair) for pair in itertools.combinations(range(4), 2)]
    index = {state: i for i, state in enumerate(states)}
    actual = np.zeros((6, 6), dtype=complex)
    for column, state in enumerate(states):
        for target, value in words.apply(block_terms(matrix, 0, 0), {state: 1.}).items():
            actual[index[target], column] += value
    one = np.linalg.eigvalsh(matrix)
    predicted = sorted(one[i] + one[j] for i, j in itertools.combinations(range(4), 2))
    return dict(owner='EdgeWords -> bosonic_statistics.primitive_pauli_words; neutral boundary parity marker',
                two_particle_dimension=6, pair_spectrum_residual=float(np.linalg.norm(
                    np.linalg.eigvalsh(actual) - predicted)))


def small_q_sequence(band, sizes, stiffness, preferred, basis):
    rows = []
    for size in sizes:
        data = uniform_data(band, size, stiffness, preferred, basis)
        q = np.array([1, 0, 0])
        plus = fourier_sea_hessian(data, q, size, basis) + data['elastic']
        minus = fourier_sea_hessian(data, -q, size, basis) + data['elastic']
        h0 = fourier_sea_hessian(data, np.zeros(3, dtype=int), size, basis) + data['elastic']
        lap = 4 * np.sin(np.pi / size)**2
        angle = 2 * np.pi / size
        orbit = data['orbit']
        odd = orbit.conj().T @ ((plus - minus) / 2) @ orbit
        even = orbit.conj().T @ ((plus + minus) / 2 - h0) @ orbit
        minimum = float(np.linalg.eigvalsh(plus)[0])
        gap = float(np.min(data['values'][:, 2]) - np.max(data['values'][:, 1]))
        bound = 4 * float(np.max(np.sum(data['x']**2, axis=1))) / gap
        rows.append(dict(L=size, q_angle=angle, laplacian=lap, minimum_unaligned_curvature=minimum,
                         minimum_over_q=minimum / angle, minimum_over_laplacian=minimum / lap,
                         orbital_odd_norm_over_q=float(np.linalg.norm(odd, 2) / angle),
                         orbital_even_norm_over_q_squared=float(np.linalg.norm(even, 2) / angle**2),
                         axial_required_J=max(0., -minimum / lap), sufficient_J=(bound + 1.) / lap))
    return rows


def run(args):
    band = RotationBand(a=args.a, m=args.m, r=args.r, v=args.v, improved=True)
    basis = matrix_unit_basis()
    data = uniform_data(band, args.L, args.kappa, args.g0, basis)
    sites, frame = data['sites'], data['frame']
    count = len(sites)
    fields = np.broadcast_to(frame, (count, 4, 4, 4)).copy()
    laplacian = alignment_laplacian(sites, args.L)
    lap_values = 4 * np.sum(np.sin(np.pi * sites / args.L)**2, axis=1)
    lambda1 = float(np.min(lap_values[1:]))
    gap = float(np.min(data['values'][:, 2]) - np.max(data['values'][:, 1]))
    frobenius_bound = 4 * float(np.max(np.sum(data['x']**2, axis=1)))
    sea_bound = frobenius_bound / gap
    alignment = (sea_bound + args.margin) / lambda1
    sea_blocks = np.array([fourier_sea_hessian(data, q, args.L, basis) for q in sites])
    unaligned = sea_blocks + data['elastic'][None]
    stable_blocks = unaligned + alignment * lap_values[:, None, None] * np.eye(60)
    full, imaginary = reconstruct_hessian(stable_blocks, sites, args.L)
    full_values, full_vectors = np.linalg.eigh(full)
    block_values = np.linalg.eigvalsh(stable_blocks)
    base_values = np.linalg.eigvalsh(unaligned)
    uniform_values = base_values[0]
    scale = max(1., float(full_values[-1]))
    tolerance = 1e-9 * scale
    global_orbit = np.tile(data['orbit'], (count, 1)) / np.sqrt(count)
    orbit_error = float(np.linalg.norm(full @ global_orbit, 2))
    projector = np.einsum('pij,pkj->pik', data['vectors'][:, :, :2], data['vectors'][:, :, :2].conj())
    sea_gradient = np.einsum('pij,aji,pm->ma', projector, basis, data['x']).real / count
    origin = np.einsum('mij,aji->ma', frame, basis).real / 4
    local_gradient = sea_gradient + 2 * args.kappa * (data['selected']['G'] - args.g0 * np.eye(4)) @ origin
    hlocal = local_hamiltonian(fields, band, sites, args.L)
    actual_values, actual_vectors = np.linalg.eigh(hlocal)
    fourier = np.exp(2j * np.pi * sites @ sites.T / args.L) / np.sqrt(count)
    transform = np.kron(fourier, np.eye(4))
    expected_h = np.zeros_like(hlocal)
    for p in range(count):
        expected_h[4*p:4*p+4, 4*p:4*p+4] = data['h'][p]
    fourier_h_error = float(np.linalg.norm(transform.conj().T @ hlocal @ transform - expected_h))
    endpoint_weights = (data['x'][:, None, :] + data['x'][None, :, :]) / 2
    expected_vertices = np.einsum('kpm,aij->makipj', endpoint_weights, basis).reshape(60, 4 * count, 4 * count) / count
    vertex_residuals = []
    for fiber in range(60):
        coordinate = np.zeros((count, 4, 15))
        coordinate[0].reshape(-1)[fiber] = 1.
        local_vertex = local_hamiltonian(field_matrices(coordinate, basis), band, sites, args.L)
        vertex_residuals.append(float(np.linalg.norm(
            transform.conj().T @ local_vertex @ transform - expected_vertices[fiber])))

    # Independently differentiate actual local matrices, including long endpoints.
    probe_indices = [0, 19, 59, 60, 60 * (count - 1) + 37]
    probes, probe_rows = [], []
    for index in probe_indices:
        coordinate = np.zeros((count, 4, 15))
        coordinate.reshape(-1)[index] = 1.
        probes.append(local_hamiltonian(field_matrices(coordinate, basis), band, sites, args.L))
    for a, left in enumerate(probes):
        for b in range(a, len(probes)):
            i, j = probe_indices[a], probe_indices[b]
            direct_sea = sea_bilinear(left, probes[b], actual_values, actual_vectors)
            elastic_entry = data['elastic'][i % 60, j % 60] if i // 60 == j // 60 else 0.
            aligned_entry = alignment * laplacian[i // 60, j // 60] if i % 60 == j % 60 else 0.
            direct = direct_sea + elastic_entry + aligned_entry
            probe_rows.append(dict(indices=[i, j], direct=direct, reconstructed=float(full[i, j]),
                                   absolute_error=float(abs(direct - full[i, j]))))
    rng = np.random.default_rng(args.seed)
    random_direction = rng.normal(size=60 * count)
    random_direction /= np.linalg.norm(random_direction)
    candidates = [('random_full_space', random_direction), ('uniform_amplitude', np.tile(origin.reshape(-1), count))]
    candidates[1] = (candidates[1][0], candidates[1][1] / np.linalg.norm(candidates[1][1]))
    nonuniform_index = np.unravel_index(np.argmin(base_values[1:]), base_values[1:].shape)[0] + 1
    eig, vec = np.linalg.eigh(unaligned[nonuniform_index])
    texture = np.real(np.exp(2j * np.pi * sites @ sites[nonuniform_index] / args.L)[:, None] * vec[:, 0][None])
    texture = texture.reshape(-1)
    texture /= np.linalg.norm(texture)
    candidates.append(('softest_nonuniform_J0', texture))
    candidates.append(('softest_positive_full', full_vectors[:, np.where(full_values > tolerance)[0][0]]))
    finite_rows = []
    for label, direction in candidates:
        delta_fields = field_matrices(direction.reshape(count, 4, 15), basis)
        perturbation = local_hamiltonian(delta_fields, band, sites, args.L)
        direct = sea_bilinear(perturbation, perturbation, actual_values, actual_vectors)
        direct += float(np.einsum('xa,ab,xb->', direction.reshape(count, 60), data['elastic'], direction.reshape(count, 60)))
        direct += alignment * float(np.einsum('xa,xy,ya->', direction.reshape(count, 60), laplacian, direction.reshape(count, 60)))
        for coupling in (0., alignment):
            analytic = float(direction @ full @ direction) + (coupling - alignment) * float(np.einsum(
                'xa,xy,ya->', direction.reshape(count, 60), laplacian, direction.reshape(count, 60)))
            origin_energy = energy(fields, band, sites, args.L, args.kappa, args.g0, coupling, laplacian)
            for step in (.002, .001):
                plus = energy(fields + step * delta_fields, band, sites, args.L, args.kappa, args.g0, coupling, laplacian)
                minus = energy(fields - step * delta_fields, band, sites, args.L, args.kappa, args.g0, coupling, laplacian)
                measured = (plus + minus - 2 * origin_energy) / step**2
                finite_rows.append(dict(direction=label, J=coupling, step=step, analytic=analytic, measured=measured,
                                        scaled_error=abs(measured - analytic) / max(1., abs(analytic)),
                                        direct_spectral_hessian_error=abs(direct - float(direction @ full @ direction))))

    generator = np.einsum('a,aij->ij', rng.normal(size=15), basis)
    unitary = expm(.17j * generator)
    conjugated = unitary @ fields @ unitary.conj().T
    original_energy = energy(fields, band, sites, args.L, args.kappa, args.g0, alignment, laplacian)
    orbit_energy_error = abs(energy(conjugated, band, sites, args.L, args.kappa, args.g0, alignment, laplacian) - original_energy)
    phases = .1 * np.cos(2 * np.pi * sites[:, 0] / args.L)
    local_unitaries = np.array([expm(1j * phase * basis[0]) for phase in phases])
    local_orbit = local_unitaries[:, None] @ fields @ local_unitaries[:, None].conj().transpose(0, 1, 3, 2)
    defects = local_orbit[:, :, None] @ local_orbit[:, None, :] + local_orbit[:, None, :] @ local_orbit[:, :, None]
    defects -= 2 * data['selected']['G'][None, :, :, None, None] * np.eye(4)
    local_penalty = float(np.sum(np.abs(defects)**2))
    local_alignment = energy(local_orbit, band, sites, args.L, args.kappa, args.g0, alignment, laplacian) - energy(
        local_orbit, band, sites, args.L, args.kappa, args.g0, 0., laplacian)
    accidental = frame.copy()
    accidental[1:] = np.sqrt(np.diag(data['selected']['G'])[1:])[:, None, None] * basis[12:15]
    accidental_fields = np.broadcast_to(accidental, fields.shape).copy()
    accidental_gap = energy(accidental_fields, band, sites, args.L, args.kappa, args.g0, 0., laplacian) - energy(
        fields, band, sites, args.L, args.kappa, args.g0, 0., laplacian)
    accidental_p0_residual = float(np.linalg.norm(np.linalg.eigvalsh(np.einsum(
        'm,mij->ij', data['x'][0], accidental)) - data['values'][0]))
    primitive = primitive_sector_probe(hlocal[:4, :4] + .03 * basis[2])
    observed_required = max(0., float(np.max(-base_values[1:, 0] / lap_values[1:])))
    checks = dict(momentum_design_rank_ten=rank_report(data['x'])['quadratic_rank'] == 10,
                  selected_gram_stationary=data['selected']['gradient_residual'] < 1e-7,
                  actual_local_stationary=float(np.max(np.abs(local_gradient))) < 1e-7,
                  actual_real_space_band_matches_scalar_paths=fourier_h_error < 1e-9,
                  every_local_fiber_vertex_matches_endpoints=max(vertex_residuals) < 1e-9,
                  real_full_hessian=imaginary < 1e-9,
                  real_space_spectral_crosschecks=max(row['absolute_error'] for row in probe_rows) < 1e-8 * scale,
                  full_spectrum_matches_all_fourier_blocks=float(np.max(np.abs(full_values - np.sort(block_values.ravel())))) < 1e-8 * scale,
                  uniform_fifteen_zero_fortyfive_positive=int(np.sum(np.abs(uniform_values) < tolerance)) == 15
                    and int(np.sum(uniform_values > tolerance)) == 45,
                  actual_global_PU4_kernel=data['orbit'].shape[1] == 15 and orbit_error < 1e-8 * scale,
                  full_only_fifteen_zero=int(np.sum(np.abs(full_values) < tolerance)) == 15
                    and int(np.sum(full_values > tolerance)) == 60 * count - 15,
                  nonuniform_margin=float(np.min(block_values[1:])) >= args.margin - 1e-8 * scale,
                  sufficient_J_exceeds_actual_threshold=alignment > observed_required,
                  finite_unrestricted_curvatures=all(row['scaled_error'] < .003 and row['direct_spectral_hessian_error'] < 1e-8 * scale for row in finite_rows),
                  finite_global_conjugation_flat=orbit_energy_error < 1e-8 * scale,
                  accidental_zero_momentum_spinor_rejected=accidental_p0_residual < 1e-12 and accidental_gap > 1e-6,
                  clifford_penalty_blind_to_real_texture=local_penalty < 1e-20 and local_alignment > 1e-6,
                  primitive_bosonic_word_sector=float(primitive['pair_spectrum_residual']) < 1e-10)
    return dict(status='finite_box_local_orbital_stability_modulo_global_PU4',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Full unrestricted spatial fields, zero EM, periodic graph; local minimum, not global spatial or thermodynamic theorem.',
                parameters=dict(L=args.L, sites=count, real_coordinates=60 * count, one_particle_dimension=4 * count,
                                half_filled_particles=2 * count, a=args.a, m=args.m, r=args.r, v=args.v,
                                kappa=args.kappa, g0=args.g0, improved=True, seed=args.seed),
                convention='Every RotationBand directed scalar path uses endpoint mean; length-two path excludes midpoint; periodic wrapping and multiplicities retained.',
                selected_gram=data['selected']['G'].tolist(), gap=gap,
                sufficient_alignment=dict(J=alignment, lambda1=lambda1, frobenius_squared_bound=frobenius_bound,
                                          sea_negative_curvature_bound=sea_bound, certified_nonuniform_margin=args.margin,
                                          actual_finite_box_J_threshold=observed_required,
                                          qualification='Algebraic bound is rigorous; floats are not interval-arithmetic certificates.'),
                uniform_hessian_eigenvalues=uniform_values.tolist(), full_hessian_eigenvalues=full_values.tolist(),
                all_fourier_blocks=[dict(q=q.tolist(), laplacian=float(lap), J0_eigenvalues=ev.tolist(),
                                        stable_eigenvalues=stable.tolist())
                                    for q, lap, ev, stable in zip(sites, lap_values, base_values, block_values)],
                adverse_J0=dict(nonuniform_negative_count=int(np.sum(base_values[1:] < -tolerance)),
                               most_negative_nonuniform=float(np.min(base_values[1:])),
                               interpretation='A negative count is reported only if present; absence on one grid is not a limit theorem.'),
                crosschecks=dict(local_stationary_residual=float(np.max(np.abs(local_gradient))),
                                 local_hamiltonian_fourier_residual=fourier_h_error, full_imaginary_residual=imaginary,
                                 all_sixty_local_vertex_residuals=vertex_residuals,
                                 global_orbit_hessian_residual=orbit_error, local_spectral_entries=probe_rows),
                finite_deformations=finite_rows,
                wrongcases=dict(accidental_p0_spectrum_residual=accidental_p0_residual, accidental_uniform_energy_excess=accidental_gap,
                                exact_local_clifford_penalty=local_penalty, same_texture_alignment_cost=local_alignment,
                                global_conjugation_energy_error=orbit_energy_error),
                primitive_sector=primitive,
                small_q_diagnostics=small_q_sequence(band, args.sequence, args.kappa, args.g0, basis),
                limit_interpretation='The sufficient bound grows as L^2 at fixed massive scalar band. Small-q ratios diagnose, but do not assume, linear-gradient versus q^2 physical behavior.',
                checks={name: bool(value) for name, value in checks.items()}, all_checks_pass=all(checks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--L', type=int, default=3)
    parser.add_argument('--a', type=float, default=1.)
    parser.add_argument('--m', type=float, default=.2)
    parser.add_argument('--r', type=float, default=1.)
    parser.add_argument('--v', type=float, default=1.)
    parser.add_argument('--kappa', type=float, default=1.)
    parser.add_argument('--g0', type=float, default=1.)
    parser.add_argument('--margin', type=float, default=1.)
    parser.add_argument('--sequence', type=int, nargs='+', default=[3, 4, 6, 8, 12])
    parser.add_argument('--seed', type=int, default=222103)
    args = parser.parse_args()
    if args.L < 3 or any(size < 3 for size in args.sequence):
        parser.error('L and sequence sizes must be >=3; L=2 loses rich kinetic inputs.')
    parameters = [args.a, args.m, args.r, args.v, args.kappa, args.g0, args.margin]
    if not all(np.isfinite(parameters)) or min(args.a, args.m, args.v, args.kappa, args.g0, args.margin) <= 0 or args.r < 0:
        parser.error('Require finite a,m,v,kappa,g0,margin>0 and r>=0.')
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

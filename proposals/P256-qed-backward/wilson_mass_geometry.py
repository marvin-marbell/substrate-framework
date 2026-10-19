#!/usr/bin/env python3
"""Gauge-uniform Wilson gap and geometry of the actual Pauli plaquette sea.

The one-body symbol is a derived code-fiber comparator, not a microscopic CAR
builder. The CLI checks it against complete compressed Gauss matrix entries.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse

from quantum_spinor_plaquette import (
    CYCLE_EDGES, ROOTS, PauliCode, SpinorPlaquette, occupied, wedge_transfer,
)
from rotation_band import RotationBand


# EXACT SpinorPlaquette spatial paths, including its reversed x and y edges.
PLAQUETTE_LINKS = ((0, 1, 0), (1, 2, 1), (3, 2, 0), (0, 3, 1))


def operator_norm(matrix):
    return float(np.linalg.norm(matrix, ord=2))


def maximum(matrix):
    return float(np.max(np.abs(matrix))) if matrix.size else 0.0


def loop_transport(source_cell, target_cell):
    """Derive ell'-ell from the actual physical oriented path; ell=E0."""
    edge = (ROOTS[source_cell], ROOTS[target_cell])
    shifts = [0] * 4
    if edge in CYCLE_EDGES:
        shifts[CYCLE_EDGES.index(edge)] = -1
    elif edge[::-1] in CYCLE_EDGES:
        shifts[CYCLE_EDGES.index(edge[::-1])] = 1
    else:
        raise ValueError('Not a physical plaquette edge')
    return shifts[0]


def plaquette_one_body(phi, m, r, v):
    """Return (h, dh/dphi), 16x16, in the ell=E0 Fourier convention.

    |ell> maps to exp(i ell phi). Therefore a physical source0->target1
    transfer has exp(-i phi). Physical edge3 has no phase in this gauge.
    Scalar onsite/link coefficients are obtained from RotationBand.couplings,
    with EXACT SpinorPlaquette's planar subtraction and adjoint orientation.
    """
    if not np.isfinite([phi, m, r, v]).all() or m <= 0 or r < 0:
        raise ValueError('Require finite phi,m,r,v, m>0 and r>=0')
    band = RotationBand(a=1.0, m=m, r=r, v=v)
    onsite3, links3 = band.couplings()
    h = np.kron(np.eye(4), onsite3 - r * band.beta).astype(complex)
    dh = np.zeros_like(h)
    for source_cell, target_cell, axis in PLAQUETTE_LINKS:
        shift = loop_transport(source_cell, target_cell)
        forward = links3[axis][1].conj().T * np.exp(1j * shift * phi)
        source = slice(4 * source_cell, 4 * source_cell + 4)
        target = slice(4 * target_cell, 4 * target_cell + 4)
        h[target, source] += forward
        h[source, target] += forward.conj().T
        dh[target, source] += 1j * shift * forward
        dh[source, target] += (1j * shift * forward).conj().T
    return h, dh


def gauss_diagonals():
    """Four neutral-N8 (16x16 D_e, scalar b_e) pairs, E_e=ell+dGamma(D_e)+b_e.

    Backgrounds come from rho_c=n_c-2, NOT the neutral parity marker.
    Charged N differs only by adding boundary=N-8 to b_3.
    """
    cells = [np.diag(np.repeat(np.eye(4)[c], 4)) for c in range(4)]
    return ((np.zeros((16, 16)), 0.0), (cells[1], -2.0),
            (cells[1] + cells[2], -4.0), (-cells[0], 2.0))


def sea_projector(h, dh):
    """Return the strictly negative spectral projector and its exact derivative.

    Only occupied-empty denominators occur. Degeneracies within either cluster
    are harmless; no eigenvector phase derivative or finite difference is used.
    The actual occupied-empty separation, not a presumed dispersion, is used.
    """
    h, dh = np.asarray(h, dtype=complex), np.asarray(dh, dtype=complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or dh.shape != h.shape:
        raise ValueError('Require square h and matching dh')
    if maximum(h - h.conj().T) > 1e-11 or maximum(dh - dh.conj().T) > 1e-11:
        raise ValueError('Require Hermitian h and dh')
    energies, vectors = np.linalg.eigh(h)
    zero_tolerance = 64 * np.finfo(float).eps * max(1.0, operator_norm(h))
    if np.min(np.abs(energies)) <= zero_tolerance:
        raise ValueError('Sea projector requires an actual nonzero spectral gap')
    valence, empty = energies < 0, energies > 0
    if not np.any(valence) or not np.any(empty):
        raise ValueError('Require both occupied and empty clusters')
    occupied_vectors, empty_vectors = vectors[:, valence], vectors[:, empty]
    p = occupied_vectors @ occupied_vectors.conj().T
    derivative_block = (empty_vectors.conj().T @ dh @ occupied_vectors
                        / (energies[valence][None, :] - energies[empty][:, None]))
    off_diagonal = empty_vectors @ derivative_block @ occupied_vectors.conj().T
    return p, off_diagonal + off_diagonal.conj().T


@lru_cache(maxsize=None)
def orbital_frames(cells):
    """Fixed beta reference and chiral involution from existing matrix units."""
    band = RotationBand(a=1.0)
    gamma = band.beta @ band.alpha[0] @ band.alpha[1] @ band.alpha[2]
    beta = np.kron(np.eye(cells), band.beta)
    beta_values, beta_vectors = np.linalg.eigh(band.beta)
    reference = np.kron(np.eye(cells), beta_vectors[:, beta_values < 0])
    return beta, np.kron(np.eye(cells), gamma), reference


def canonical_sea_frame(p, dp):
    """Global smooth polar frame of P on the fixed beta-negative reference."""
    if p.shape[0] % 4:
        raise ValueError('Require four derived orbitals per cell')
    reference = orbital_frames(p.shape[0] // 4)[2]
    overlap = reference.conj().T @ p @ reference
    derivative = reference.conj().T @ dp @ reference
    values, vectors = np.linalg.eigh(overlap)
    if np.min(values) <= 0:
        raise ValueError('Reference projection is not invertible')
    roots = np.sqrt(values)
    inverse_root = (vectors / roots) @ vectors.conj().T
    divided_difference = -1 / (roots[:, None] * roots[None, :]
                              * (roots[:, None] + roots[None, :]))
    dinverse_root = vectors @ (divided_difference * (vectors.conj().T @ derivative @ vectors)) @ vectors.conj().T
    frame = p @ reference @ inverse_root
    dframe = dp @ reference @ inverse_root + p @ reference @ dinverse_root
    return frame, dframe


def parallel_sea_frame(h, dh):
    """Return (frame,dframe) for the global periodic parallel Slater section.

    The fixed beta-negative-reference polar frame has zero TRACE connection
    by constant chiral symmetry. Its wedge is already determinant-parallel;
    no phase correction is required. Individual orbitals need not be parallel.
    """
    if h.shape[0] % 4:
        raise ValueError('Require four derived orbitals per cell')
    beta, gamma, _ = orbital_frames(h.shape[0] // 4)
    scale = max(1.0, operator_norm(h))
    if max(operator_norm(gamma @ h + h @ gamma),
           operator_norm(gamma @ dh + dh @ gamma)) > 1e-10 * scale:
        raise ValueError('Require constant chiral anticommutation of h and dh')
    mass = (h @ beta + beta @ h) / 2
    if np.min(np.linalg.eigvalsh(mass)) <= 0:
        raise ValueError('Require positive mass on the fixed beta reference')
    return canonical_sea_frame(*sea_projector(h, dh))


def mass_diagnostics(h, m):
    beta = orbital_frames(h.shape[0] // 4)[0]
    mass = (h @ beta + beta @ h) / 2
    q = h - beta @ mass
    energies = np.linalg.eigvalsh(h)
    lower_mass = float(np.min(np.linalg.eigvalsh(mass)))
    gap = float(np.min(np.abs(energies)))
    if lower_mass < m - 2e-10 or gap < m - 2e-10:
        raise AssertionError('Positive Wilson mass or Schur gap bound violated')
    return {'declared_mass_lower_bound': m, 'actual_mass_minimum': lower_mass,
            'actual_gap_about_zero': gap, 'odd_hopping_norm': operator_norm(q),
            'mass_hopping_commutator_norm': operator_norm(mass @ q - q @ mass),
            'mass_beta_commutator': maximum(mass @ beta - beta @ mass),
            'q_beta_anticommutator': maximum(q @ beta + beta @ q),
            'hermiticity': maximum(h - h.conj().T)}


def lattice_shifts(shape, periodic):
    """Actual one-link transports with arbitrary deterministic curved phases."""
    sites = list(itertools.product(*(range(length) for length in shape)))
    index = {site: number for number, site in enumerate(sites)}
    shifts = []
    for axis in range(3):
        shift = np.zeros((len(sites), len(sites)), dtype=complex)
        for source, site in enumerate(sites):
            target = list(site)
            target[axis] += 1
            if target[axis] == shape[axis]:
                if not periodic:
                    continue
                target[axis] = 0
            phase = 0.91 * np.sin(1.17 * (source + 1) * (axis + 1)) + 0.37 * np.cos(source + 2 * axis)
            shift[index[tuple(target)], source] = np.exp(1j * phase)
        shifts.append(shift)
    return shifts


def lattice_one_body(shape, periodic, m, r, v, improved):
    """Primitive 3D nearest/two-link paths and ALL their adjoints, a=1.

    Open improved coefficients are Galerkin compression of the untruncated
    primitive operator, NOT the square of a compressed nearest Laplacian.
    """
    band = RotationBand(a=1.0, m=m, r=r, v=v, improved=improved)
    onsite, couplings = band.couplings()
    shifts = lattice_shifts(shape, periodic)
    h = np.kron(np.eye(int(np.prod(shape))), onsite).astype(complex)
    for displacement, coupling in couplings:
        axis = int(np.flatnonzero(displacement)[0])
        steps = int(displacement[axis])
        transport = shifts[axis] if steps == 1 else shifts[axis] @ shifts[axis]
        path = np.kron(transport, coupling.conj().T)
        h += path + path.conj().T
    return h, shifts


def wilson_compression_control(shift, r):
    identity = np.eye(shift.shape[0])
    laplacian = identity - (shift + shift.conj().T) / 2
    primitive = r * (1.5 * identity - shift - shift.conj().T
                     + (shift @ shift + shift.conj().T @ shift.conj().T) / 4)
    boundary = r * (2 * identity - shift @ shift.conj().T - shift.conj().T @ shift) / 4
    residual = maximum(primitive - r * laplacian @ laplacian - boundary)
    naive_error = operator_norm(primitive - r * laplacian @ laplacian)
    if residual > 1e-11 or np.min(np.linalg.eigvalsh(primitive)) < -1e-11:
        raise AssertionError('Actual improved Wilson compression is not positive')
    return {'identity_residual': residual,
            'primitive_minimum': float(np.min(np.linalg.eigvalsh(primitive))),
            'boundary_minimum': float(np.min(np.linalg.eigvalsh(boundary))),
            'boundary_norm': operator_norm(boundary),
            'wrong_compressed_square_operator_error': naive_error,
            'wrong_compression_rejected': naive_error > 1e-8}


def geometry_control(phi, m, r, v):
    h, dh = plaquette_one_body(phi, m, r, v)
    p, dp = sea_projector(h, dh)
    _, gamma, reference = orbital_frames(4)
    frame, dframe = parallel_sea_frame(h, dh)
    overlap_minimum = float(np.min(np.linalg.eigvalsh(reference.conj().T @ p @ reference)))
    metric = float(np.trace(p @ dp @ dp).real)
    derivative_bound = operator_norm(dh) / (2 * m)
    metric_bound = 8 * operator_norm(dh) ** 2 / (4 * m ** 2)
    cell_traces = [float(np.trace(p[4 * cell:4 * cell + 4, 4 * cell:4 * cell + 4]).real)
                   for cell in range(4)]
    charge_means, charge_derivatives, charge_variances = [], [], []
    for diagonal, background in gauss_diagonals():
        charge_means.append(float((np.trace(p @ diagonal) + background).real))
        charge_derivatives.append(float(np.trace(dp @ diagonal).real))
        charge_variances.append(float(np.trace(p @ diagonal @ (np.eye(16) - p) @ diagonal).real))
    connection = np.trace(frame.conj().T @ dframe)
    # Independent finite difference is applied ONLY to gauge-invariant projectors,
    # never to phased eigenvectors, and is not how the exported derivative is built.
    step = 2e-5
    p_plus = sea_projector(*plaquette_one_body(phi + step, m, r, v))[0]
    p_minus = sea_projector(*plaquette_one_body(phi - step, m, r, v))[0]
    derivative_residual = maximum(dp - (p_plus - p_minus) / (2 * step))
    residuals = {
        'projector_idempotence': maximum(p @ p - p),
        'spectral_commutator': maximum(h @ p - p @ h),
        'derivative_projector_identity': maximum(dp @ p + p @ dp - dp),
        'derivative_sylvester_identity': maximum(h @ dp - dp @ h + dh @ p - p @ dh),
        'projector_derivative_independent_control': derivative_residual,
        'chiral_anticommutator': maximum(gamma @ h + h @ gamma),
        'chiral_projector_complement': maximum(gamma @ p @ gamma - (np.eye(16) - p)),
        'canonical_frame_projector': maximum(frame @ frame.conj().T - p),
        'canonical_frame_derivative': maximum(dframe @ frame.conj().T + frame @ dframe.conj().T - dp),
        'parallel_determinant_connection': float(abs(connection)),
        'cell_half_filling_error': max(abs(value - 2) for value in cell_traces),
        'charge_mean_error': max(abs(value) for value in charge_means),
        'charge_derivative_error': max(abs(value) for value in charge_derivatives),
    }
    if max(residuals.values()) > 2e-7 or overlap_minimum < 0.5 - 1e-10:
        raise AssertionError('Actual smooth sea/charge geometry identity failed')
    if operator_norm(dp) > derivative_bound + 1e-10 or metric > metric_bound + 1e-10:
        raise AssertionError('Gap-controlled derivative or Slater metric bound failed')
    return {'phi': phi, **mass_diagnostics(h, m), 'residuals': residuals,
            'reference_overlap_minimum': overlap_minimum,
            'projector_derivative_norm': operator_norm(dp),
            'projector_derivative_bound': derivative_bound,
            'parallel_slater_metric': metric, 'slater_metric_bound': metric_bound,
            'cell_occupied_traces': cell_traces, 'gauss_offset_means': charge_means,
            'gauss_offset_mean_derivatives': charge_derivatives,
            'gauss_offset_variances': charge_variances,
            'physical_spin_status': 'Berry determinant geometry is not physical Spin3; use prior orbital-spatial locking'}


def actual_fiber_control(code, number, cutoff, phi, m, r, v):
    """Check every surviving full-Gauss matter entry through actual Pauli phases.

    This is a row/column transport-symbol check, NOT an assertion that finite
    electric truncation is a direct integral of independent phi fibers.
    """
    model = SpinorPlaquette(code, number, cutoff, m=m, r=r, v=v)
    phases = code.intertwiner(number)
    h, dh = plaquette_one_body(phi, m, r, v)
    electric = np.array([model.UE / 2 * (sum(e * e for e in fields) + model.boundary ** 2)
                         for _, fields in model.raw])
    matter = (model.H - sparse.diags(electric)
              + model.K / 2 * (model.W + model.W.conjugate().T)).tocoo()
    matter.eliminate_zeros()
    residual = derivative_residual = gauss_residual = wrong_flux_error = wrong_edge_error = 0.0
    observed_entries, transports, markers = set(), set(), set()
    wrong_h = np.kron(np.eye(4), model.onsite).astype(complex)
    band = RotationBand(a=1.0, m=m, r=r, v=v)
    links = band.couplings()[1]
    for source_cell, target_cell, axis in PLAQUETTE_LINKS:
        # Deliberately wrong: place flux on omitted-code edge3, not on E0.
        wrong_shift = 1 if (source_cell, target_cell) == (0, 3) else 0
        source = slice(4 * source_cell, 4 * source_cell + 4)
        target = slice(4 * target_cell, 4 * target_cell + 4)
        forward = links[axis][1].conj().T * np.exp(1j * wrong_shift * phi)
        wrong_h[target, source] += forward
        wrong_h[source, target] += forward.conj().T
    diagonal_pairs = gauss_diagonals()
    for row, column, value in zip(matter.row, matter.col, matter.data):
        mask, fields = model.raw[column]
        new, shifted = model.raw[row]
        for edge, (diagonal, background) in enumerate(diagonal_pairs):
            offset = sum(diagonal[i, i] for i in occupied(mask)) + background
            offset += model.boundary if edge == 3 else 0
            gauss_residual = max(gauss_residual, abs(fields[edge] - fields[0] - offset))
        delta_loop = shifted[0] - fields[0]
        transformed = value * np.conjugate(phases[new]) * phases[mask] * np.exp(1j * delta_loop * phi)
        if mask == new:
            expected = sum(h[i, i] for i in occupied(mask))
            residual = max(residual, abs(transformed - expected))
            continue
        removed, added = mask & ~new, new & ~mask
        if removed.bit_count() != 1 or added.bit_count() != 1:
            raise AssertionError('Actual bilinear changed more than one physical orbital')
        source, target = removed.bit_length() - 1, added.bit_length() - 1
        sign = wedge_transfer(mask, source, target)
        residual = max(residual, abs(transformed - sign * h[target, source]))
        derivative_residual = max(derivative_residual,
                                  abs(1j * delta_loop * transformed - sign * dh[target, source]))
        wrong_flux = value * np.conjugate(phases[new]) * phases[mask] * np.exp(-1j * delta_loop * phi)
        wrong_flux_error = max(wrong_flux_error, abs(wrong_flux - sign * h[target, source]))
        wrong_edge_error = max(wrong_edge_error, abs(transformed - sign * wrong_h[target, source]))
        observed_entries.add((target, source))
        transports.add((source // 4, target // 4, delta_loop))
    for mask, fields in model.raw:
        gauss_residual = max(gauss_residual, max(abs(x) for x in model.gauss(mask, fields)))
        for raw_word, _ in code.state(mask):
            markers.add(int(16 in code.words.occupation(raw_word)))
    expected_entries = set(zip(*np.nonzero(abs(h - np.diag(np.diag(h))) > 1e-14)))
    if observed_entries != expected_entries or max(residual, derivative_residual, gauss_residual) > 2e-10:
        raise AssertionError('Derived static symbol does not equal actual Pauli/Gauss matrix entries')
    if min(wrong_flux_error, wrong_edge_error) < 1e-5:
        raise AssertionError('Wrong Fourier transport or physical phase edge was not discriminated')
    if markers != {number % 2}:
        raise AssertionError('Actual parity marker occupancy failed')
    wrong_marker = (any(model.gauss(mask, fields, marker_charge=1)) for mask, fields in model.raw)
    marker_charge_rejected = any(wrong_marker) if number % 2 else None
    if number % 2 and not marker_charge_rejected:
        raise AssertionError('Actual odd-N marker mischarge escaped Gauss')
    return {'number': number, 'cutoff': cutoff, 'complete_gauss_dimension': model.dimension,
            'actual_matter_matrix_nonzero_entries': int(matter.nnz),
            'covered_one_body_offdiagonal_entries': len(observed_entries),
            'pauli_intertwiner_residual': code.law_residual,
            'full_gauss_fiber_entry_residual': float(residual),
            'full_gauss_fiber_derivative_residual': float(derivative_residual),
            'electric_offset_and_gauss_residual': float(gauss_residual),
            'physical_cell_transports': sorted(transports),
            'actual_marker_occupancies': sorted(markers),
            'wrong_marker_em_charge_rejected': marker_charge_rejected,
            'wrong_fourier_transport_error': float(wrong_flux_error),
            'wrong_flux_on_code_cut_edge3_error': float(wrong_edge_error),
            'finite_cutoff_status': 'compressed Gauss entries checked; not a direct-integral finite-rotor identity'}


def hypothesis_controls(m):
    beta = RotationBand(a=1.0).beta
    identity = np.eye(4)
    bad_mass = np.zeros((4, 4))
    gap_without_mass = float(np.min(abs(np.linalg.eigvalsh(beta @ bad_mass))))
    bad_q = -m * identity
    gap_without_oddness = float(np.min(abs(np.linalg.eigvalsh(m * beta + bad_q))))
    residual = operator_norm(beta @ bad_q + bad_q @ beta)
    if gap_without_mass > 1e-12 or gap_without_oddness > 1e-12 or residual <= 0:
        raise AssertionError('Hypothesis-violation controls did not close the declared gap')
    return {'nonpositive_mass_gap': gap_without_mass,
            'beta_even_hopping_gap': gap_without_oddness,
            'beta_even_hopping_anticommutator': residual,
            'hypothesis_violations_rejected': True}


def run(args):
    geometry = [geometry_control(phi, args.m, args.r, args.v)
                for phi in (0.0, args.phi, np.pi, 2 * np.pi)]
    h_zero, dh_zero = plaquette_one_body(0.0, args.m, args.r, args.v)
    h_periodic, dh_periodic = plaquette_one_body(2 * np.pi, args.m, args.r, args.v)
    frames_zero = parallel_sea_frame(h_zero, dh_zero)
    frames_periodic = parallel_sea_frame(h_periodic, dh_periodic)
    frame_periodicity = maximum(frames_zero[0] - frames_periodic[0])
    if frame_periodicity > 1e-10:
        raise AssertionError('Global determinant-parallel frame is not periodic')
    lattices = []
    for periodic in (False, True):
        for improved in (False, True):
            h, shifts = lattice_one_body(tuple(args.shape), periodic, args.m, args.r, args.v, improved)
            compression = [wilson_compression_control(shift, args.r) for shift in shifts] if improved else []
            if improved and any(item['wrong_compression_rejected'] != (not periodic) for item in compression):
                raise AssertionError('Boundary correction was not correctly distinguished')
            diagnostic = mass_diagnostics(h, args.m)
            if diagnostic['mass_hopping_commutator_norm'] < 1e-5:
                raise AssertionError('Curved arbitrary-link consumer failed to probe noncommuting M,Q')
            lattices.append({'shape': args.shape, 'periodic': periodic, 'improved': improved,
                             **diagnostic, 'improved_compression': compression})
    code = PauliCode()
    fibers = [actual_fiber_control(code, number, args.cutoff, args.phi, args.m, args.r, args.v)
              for number in args.numbers]
    return {'schema': 'wilson-mass-geometry-v1',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'parameters': {'m': args.m, 'r': args.r, 'v': args.v, 'phi': args.phi,
                           'shape': args.shape, 'cutoff': args.cutoff, 'numbers': args.numbers},
            'theorem': 'h=beta M+Q, M>=m>0, [M,beta]=0, {Q,beta}=0 => spectrum excludes (-m,m), without hopping-norm restriction',
            'domain': 'finite static gauge fibers and positive primitive Galerkin Wilson compressions; inherited selected Clifford shape',
            'plaquette_geometry': geometry, 'global_parallel_frame_periodicity_residual': frame_periodicity,
            'three_dimensional_arbitrary_link_consumers': lattices,
            'actual_pauli_gauss_fiber_controls': fibers,
            'hypothesis_controls': hypothesis_controls(args.m),
            'limits': ['Static one-body theorem is not a full charged compact-photon many-body gap theorem.',
                       'Finite electric truncation is not an infinite-rotor Fourier direct integral.',
                       'No quantum orbital symmetry-breaking or continuum/QED origin is claimed.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m', type=float, default=0.03)
    parser.add_argument('--r', type=float, default=0.27)
    parser.add_argument('--v', type=float, default=1.7)
    parser.add_argument('--phi', type=float, default=0.731)
    parser.add_argument('--shape', type=int, nargs=3, default=[3, 3, 3])
    parser.add_argument('--cutoff', type=int, default=1)
    parser.add_argument('--numbers', type=int, nargs='+', choices=(7, 8, 9), default=[7, 8, 9])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.m <= 0 or args.r <= 0 or args.v <= 0 or min(args.shape) < 2 or args.cutoff < 1:
        parser.error('Require m,r,v>0, shape lengths>=2 and cutoff>=1')
    if abs(np.sin(args.phi)) < 1e-4:
        parser.error('Choose a nontrivial flux for the adverse orientation controls')
    result = run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.write_text(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()

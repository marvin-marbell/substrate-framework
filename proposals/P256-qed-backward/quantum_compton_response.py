#!/usr/bin/env python3
"""Full same-N quantum matter Peierls response, not a photon S matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla

from quantum_spinor_plaquette import (
    CELLS, CYCLE_EDGES, PauliCode, SpinorPlaquette, max_abs, occupied, solve_low,
)


def invariant(matrix):
    """Only unitary-ground-frame invariant outputs, even for non-Hermitian kernels."""
    eigenvalues = np.linalg.eigvals(matrix)
    eigenvalues = sorted(eigenvalues, key=lambda z: (float(z.real), float(z.imag)))
    return {'frobenius_norm': float(np.linalg.norm(matrix)),
            'singular_values': np.linalg.svd(matrix, compute_uv=False).tolist(),
            'eigenvalues': [[float(z.real), float(z.imag)] for z in eigenvalues],
            'trace': [float(np.trace(matrix).real), float(np.trace(matrix).imag)]}


def plane_wave_edges(momentum, polarization):
    """Exact straight-link integral of eps exp(i k.x) on the physical square."""
    k, eps = np.asarray(momentum, float), np.asarray(polarization, float)
    result = []
    for source, target in CYCLE_EDGES:
        x, y = np.asarray(CELLS[source // 4], float), np.asarray(CELLS[target // 4], float)
        displacement = y - x
        result.append(np.dot(eps, displacement) * np.exp(1j * np.dot(k, (x + y) / 2))
                      * np.sinc(np.dot(k, displacement) / (2 * np.pi)))
    return np.asarray(result)


class MatterPeierls:
    """Differentiate actual Pauli words/rotor shifts, keeping magnetic W fixed."""

    def __init__(self, model):
        self.model = model
        rows, columns, coefficients, directions = [], [], [], []
        by_mask = {}
        for column, (mask, fields) in enumerate(model.raw):
            by_mask.setdefault(mask, []).append((column, fields))
        term_count = 0
        for mask, fiber in by_mask.items():
            for source, target, coefficient, word, shifts in model.terms:
                if not any(shifts) or not mask >> source & 1 or mask >> target & 1:
                    continue
                new = mask ^ (1 << source) ^ (1 << target)
                amplitude = model.code.element(mask, new, word, model.sign)
                direction = tuple(-q for q in shifts)
                for column, fields in fiber:
                    shifted = tuple(e + q for e, q in zip(fields, shifts))
                    if any(abs(e) > model.cutoff for e in shifted):
                        continue
                    row = model.index.get((new, shifted))
                    if row is None:
                        raise AssertionError('Peierls differentiated hop escaped actual Gauss domain')
                    rows.append(row); columns.append(column)
                    coefficients.append(coefficient * amplitude); directions.append(direction)
                    term_count += 1
        self.rows, self.columns = np.asarray(rows), np.asarray(columns)
        self.coefficients = np.asarray(coefficients, complex)
        self.directions = np.asarray(directions, float)
        self.term_count = term_count
        self.J = [self.matrix(1j * self.directions[:, e]) for e in range(4)]
        self.C = [[self.matrix(-self.directions[:, e] * self.directions[:, f])
                   for f in range(4)] for e in range(4)]
        self.hermiticity = max(max_abs(a - a.conjugate().T)
                               for a in self.J + [a for row in self.C for a in row])
        if self.hermiticity > 2e-12:
            raise AssertionError('Actual Peierls derivative is not Hermitian')

    def matrix(self, factor):
        return sparse.coo_matrix((self.coefficients * factor, (self.rows, self.columns)),
                                 shape=self.model.H.shape).tocsr()

    def current(self, edges):
        return self.matrix(1j * (self.directions @ np.asarray(edges)))

    def contact(self, left, right):
        return self.matrix(-(self.directions @ np.asarray(left))
                           * (self.directions @ np.asarray(right)))

    def ward_operators(self, chi):
        gradient = np.asarray([chi[b // 4] - chi[a // 4] for a, b in CYCLE_EDGES])
        density = np.asarray([sum(chi[i // 4] for i in occupied(mask))
                              for mask, _ in self.model.raw])
        G = sparse.diags(density, format='csr')
        commutator = G @ self.model.H - self.model.H @ G
        J, C = self.current(gradient), self.contact(gradient, gradient)
        double = G @ commutator - commutator @ G
        magnetic = G @ self.model.W - self.model.W @ G
        identities = {'gradient_curl': float(abs(gradient.sum())),
                      'J_minus_i_density_commutator': max_abs(J - 1j * commutator),
                      'C_plus_density_double_commutator': max_abs(C + double),
                      'density_magnetic_commutator': max_abs(magnetic)}
        if max(identities.values()) > 2e-12:
            raise AssertionError('Exact matter-only density/Peierls Ward operators fail')
        return gradient, density, J, C, identities


class CompleteResolvent:
    """Full raw same-N complement, never a truncated list of low eigenstates."""

    def __init__(self, model, energy, ground, gap, tolerance, maxiter):
        self.model, self.energy, self.ground, self.gap = model, energy, ground, gap
        self.tolerance, self.maxiter = tolerance, maxiter
        self.receipts = []

    def project(self, x):
        return x - self.ground @ (self.ground.conj().T @ x)

    def solve(self, source, frequency):
        if abs(frequency) >= self.gap:
            raise ValueError('Complement resolvent requires frequency strictly below measured gap')
        Qsource = self.project(source)
        def action(x):
            q = self.project(x)
            return self.project(self.model.H @ q - (self.energy + frequency) * q) \
                + self.gap * self.ground @ (self.ground.conj().T @ x)
        class ProjectedOperator(sla.LinearOperator):
            def __init__(self):
                super().__init__(np.dtype(complex), self_shape)

            def _matvec(self, x):
                return action(x)

        self_shape = self.model.H.shape
        operator = ProjectedOperator()
        result = np.empty_like(Qsource)
        for column in range(Qsource.shape[1]):
            rhs = Qsource[:, column]
            calls = [0]
            def iteration(_):
                calls[0] += 1
            solution, status = sla.cg(operator, rhs, rtol=self.tolerance, atol=0.,
                                      maxiter=self.maxiter, callback=iteration)
            residual = float(np.linalg.norm(action(solution) - rhs))
            rhs_norm = float(np.linalg.norm(rhs))
            relative = residual / rhs_norm if rhs_norm else residual
            leakage = float(np.linalg.norm(self.ground.conj().T @ solution))
            if status != 0 or relative > 8 * self.tolerance:
                raise AssertionError(f'Complete projected resolvent did not converge: {status}, {relative}')
            result[:, column] = self.project(solution)
            self.receipts.append({'frequency': float(frequency), 'column': column,
                                  'iterations': calls[0], 'cg_status': int(status),
                                  'rhs_norm': rhs_norm, 'absolute_residual': residual,
                                  'relative_residual': relative, 'ground_leakage': leakage})
        return result

    def kernel(self, left, right, contact, omega):
        ground = self.ground
        first = ground.conj().T @ (left @ self.solve(right @ ground, omega))
        second = ground.conj().T @ (right @ self.solve(left @ ground, -omega))
        local = ground.conj().T @ (contact @ ground)
        left_ground, right_ground = ground.conj().T @ (left @ ground), ground.conj().T @ (right @ ground)
        if omega == 0:
            if max(np.linalg.norm(left_ground), np.linalg.norm(right_ground)) > 2e-7:
                raise ValueError('Static ground poles cannot be discarded for this source')
            born = np.zeros_like(local)
        else:
            born = (left_ground @ right_ground - right_ground @ left_ground) / omega
        return local - first - second + born, local, first + second, born


def source_sum_identities(model, ground, energy, source):
    """Complete equal-time closure and energy-weighted double commutator."""
    acted = source @ ground
    elastic = ground.conj().T @ acted
    inelastic = acted - ground @ elastic
    total = acted.conj().T @ acted
    ground_weight = elastic.conj().T @ elastic
    complement_weight = inelastic.conj().T @ inelastic
    moment = inelastic.conj().T @ (model.H @ inelastic - energy * inelastic)
    # Source is Hermitian in this diagnostic. Compute the actual operator action,
    # not a duplicate of the spectral-side expression.
    Hground = model.H @ ground
    S2ground = source @ acted
    half_double = acted.conj().T @ (model.H @ acted) \
        - (S2ground.conj().T @ Hground + Hground.conj().T @ S2ground) / 2
    return {'total_weight': invariant(total), 'ground_weight': invariant(ground_weight),
            'complete_complement_weight': invariant(complement_weight),
            'equal_time_closure_residual': float(np.linalg.norm(total - ground_weight - complement_weight)),
            'energy_weighted_moment': invariant(moment),
            'half_double_commutator': invariant(half_double),
            'energy_sum_residual': float(np.linalg.norm(moment - half_double))}


def analyze_sector(code, number, cutoff, args):
    model = SpinorPlaquette(code, number, cutoff, magnetic=args.magnetic,
                           electric=args.electric, delta=args.delta, m=args.mass,
                           r=args.wilson, v=args.velocity)
    energies, ground, spectrum = solve_low(model, tol=args.eigen_tolerance)
    energy, gap = float(energies[0]), spectrum['gap_above_ground_multiplet']
    if ground.shape[1] != 2:
        raise AssertionError('Charged odd-N response requires the actual complete Kramers frame')
    eigen_residual = max(spectrum['ground_eigen_residuals'] + [spectrum['first_complement_eigen_residual']])
    if eigen_residual > args.identity_tolerance * min(1., gap):
        raise AssertionError('Eigen/complement residual insufficient for requested response accuracy')
    peierls = MatterPeierls(model)
    chi = np.asarray((0., .37, -.23, .61))
    gradient, density, ward_J, ward_C, identities = peierls.ward_operators(chi)
    omega = args.gap_fraction * gap
    incoming_k, outgoing_k = np.asarray(args.incoming_momentum), np.asarray(args.outgoing_momentum)
    def transverse(k):
        if np.linalg.norm(k) == 0:
            raise ValueError('Nonzero plane-wave momentum required')
        return np.asarray((-k[1], k[0])) / np.linalg.norm(k)
    incoming = plane_wave_edges(incoming_k, transverse(incoming_k))
    outgoing = plane_wave_edges(outgoing_k, transverse(outgoing_k))
    left, right = peierls.current(outgoing.conjugate()), peierls.current(incoming)
    contact = peierls.contact(outgoing.conjugate(), incoming)
    results, stage_receipts = [], []
    for tolerance in (args.coarse_tolerance, args.resolvent_tolerance):
        solver = CompleteResolvent(model, energy, ground, gap, tolerance, args.maxiter)
        ward = solver.kernel(ward_J, ward_J, ward_C, 0.)
        physical = solver.kernel(left, right, contact, omega)
        results.append((ward, physical)); stage_receipts.append(solver.receipts)
    (ward, physical), (fine_ward, fine_physical) = results
    convergence = {'static_Ward_kernel_change': float(np.linalg.norm(fine_ward[0] - ward[0])),
                   'nonforward_kernel_change': float(np.linalg.norm(fine_physical[0] - physical[0]))}
    ward_total, ward_contact, ward_pair, _ = fine_ward
    ward_error = float(np.linalg.norm(ward_total))
    scale = max(float(np.linalg.norm(ward_contact)), float(np.linalg.norm(ward_pair)))
    if scale <= args.identity_tolerance:
        raise AssertionError('Pure-gradient Ward probe has no discriminating contact/pair weight')
    if ward_error > args.identity_tolerance or max(convergence.values()) > args.identity_tolerance:
        raise AssertionError('Full same-N response did not satisfy Ward/convergence tolerances')
    ground_current = ground.conj().T @ (ward_J @ ground)
    # Damaged calculation one deletes C. Calculation two keeps only the actual
    # P intermediates: their two denominators are -omega,+omega. For this single
    # gradient source the pole terms cancel identically, including in the
    # static limit; no ill-defined inverse at omega=0 or spectral floor is used.
    missing_contact = -ward_pair
    projected_pole_numerator = ground_current @ ground_current
    frozen_pair = -projected_pole_numerator / omega + projected_pole_numerator / omega
    frozen_intermediates = ground.conj().T @ (ward_C @ ground) - frozen_pair
    adverse_scale = max(100 * ward_error, args.identity_tolerance)
    controls = {'missing_contact_Ward_norm': float(np.linalg.norm(missing_contact)),
                'frozen_to_ground_intermediates_Ward_norm': float(np.linalg.norm(frozen_intermediates)),
                'frozen_ground_pole_numerator_norm': float(np.linalg.norm(projected_pole_numerator)),
                'frozen_ground_pair_static_continuation_norm': float(np.linalg.norm(frozen_pair)),
                'missing_contact_breaks_Ward': bool(np.linalg.norm(missing_contact) > adverse_scale),
                'frozen_to_ground_breaks_Ward': bool(np.linalg.norm(frozen_intermediates) > adverse_scale)}
    if not controls['missing_contact_breaks_Ward'] or not controls['frozen_to_ground_breaks_Ward']:
        raise AssertionError('Adverse Ward controls did not discriminate omitted quantum terms')
    # Independent exact Ward resolvent solution is -i Q G|g>, from J=i[G,H].
    exact_seed_solution = -1j * (density[:, None] * ground
                               - ground @ (ground.conj().T @ (density[:, None] * ground)))
    exact_seed = ward_J @ ground
    solution_identity_residual = float(np.linalg.norm(
        model.H @ exact_seed_solution - energy * exact_seed_solution - exact_seed))
    closure = source_sum_identities(model, ground, energy, ward_J)
    if max(closure['equal_time_closure_residual'], closure['energy_sum_residual'],
           solution_identity_residual) > args.identity_tolerance:
        raise AssertionError('Actual complete current sum/commutator identity failed')
    # A genuine change of doublet coordinates must only conjugate the kernel.
    U = np.asarray(((1., 1j), (1j, 1.))) / np.sqrt(2)
    original = fine_physical[0]
    rotated_solver = CompleteResolvent(model, energy, ground @ U, gap,
                                      args.resolvent_tolerance, args.maxiter)
    rotated = rotated_solver.kernel(left, right, contact, omega)[0]
    covariance_error = float(np.linalg.norm(rotated - U.conj().T @ original @ U))
    basis_change = abs(np.linalg.norm(rotated) - np.linalg.norm(original))
    if covariance_error > args.identity_tolerance:
        raise AssertionError('Independently recomputed doublet-frame covariance failed')
    shell = np.asarray([any(abs(e) == cutoff for e in fields) for _, fields in model.raw])
    counts = [sum(mask >> (4 * c + j) & 1 for j in range(4))
              for mask, _ in model.raw for c in range(4)]
    return {'N': number, 'R': cutoff, 'boundary_charge': model.boundary,
            'full_raw_Gauss_dimension': model.dimension,
            'raw_occupation_masks': len({mask for mask, _ in model.raw}),
            'raw_domain_sha256': hashlib.sha256(repr(model.raw).encode()).hexdigest(),
            'cell_occupation_range': [min(counts), max(counts)],
            'raw_cutoff_shell_states': int(shell.sum()),
            'ground_cutoff_shell_probability': invariant(ground.conj().T @ (shell[:, None] * ground)),
            'parameters': {'K': model.K, 'UE': model.UE, 'Delta': model.delta,
                           'm': args.mass, 'r': args.wilson, 'v': args.velocity},
            'spectrum': spectrum, 'frequency': omega, 'measured_gap_fraction': args.gap_fraction,
            'actual_T_squared_residual': model.Tsquare, 'actual_T_symmetry_residual': model.Tsymmetry,
            'peierls_hop_matrix_entries_before_CSR_sum': peierls.term_count,
            'derivative_hermiticity_residual': peierls.hermiticity,
            'Ward': {'chi': chi.tolist(), 'edge_gradient': gradient.tolist(),
                     'operator_identities': identities, 'ground_current': invariant(ground_current),
                     'contact': invariant(ward_contact), 'complete_pair': invariant(ward_pair),
                     'contact_minus_pair': invariant(ward_total),
                     'exact_density_resolvent_solution_residual': solution_identity_residual,
                     'adverse_controls': controls, 'current_sum_identities': closure},
            'nonforward': {'incoming_momentum': incoming_k.tolist(),
                           'outgoing_momentum': outgoing_k.tolist(),
                           'incoming_edge_amplitudes': [[float(z.real), float(z.imag)] for z in incoming],
                           'outgoing_edge_amplitudes': [[float(z.real), float(z.imag)] for z in outgoing],
                           'contact': invariant(fine_physical[1]),
                           'complete_complement_pair': invariant(fine_physical[2]),
                           'analytic_ground_doublet_Born': invariant(fine_physical[3]),
                           'full_kernel': invariant(original),
                           'recomputed_doublet_covariance_residual': covariance_error,
                           'doublet_unitary_norm_residual': float(basis_change)},
            'tolerance_convergence': convergence,
            'resolvents': [{'requested_relative_tolerance': t, 'solves': r}
                           for t, r in zip((args.coarse_tolerance, args.resolvent_tolerance), stage_receipts)],
            'rotated_frame_resolvents': rotated_solver.receipts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--numbers', nargs='+', type=int, choices=(7, 9), default=[7, 9])
    parser.add_argument('--magnetic', type=float, default=.25)
    parser.add_argument('--electric', type=float, default=.6)
    parser.add_argument('--delta', type=float, default=1.2)
    parser.add_argument('--mass', type=float, default=.8)
    parser.add_argument('--wilson', type=float, default=.04)
    parser.add_argument('--velocity', type=float, default=.08)
    parser.add_argument('--gap-fraction', type=float, default=.23)
    parser.add_argument('--incoming-momentum', nargs=2, type=float, default=[.65, .2])
    parser.add_argument('--outgoing-momentum', nargs=2, type=float, default=[-.25, .55])
    parser.add_argument('--eigen-tolerance', type=float, default=1e-11)
    parser.add_argument('--coarse-tolerance', type=float, default=1e-8)
    parser.add_argument('--resolvent-tolerance', type=float, default=2e-11)
    parser.add_argument('--identity-tolerance', type=float, default=2e-6)
    parser.add_argument('--maxiter', type=int, default=4000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 0 < args.gap_fraction < 1:
        parser.error('Require a strictly positive frequency fraction below the measured gap')
    if not 0 < args.resolvent_tolerance < args.coarse_tolerance < 1:
        parser.error('Require 0 < fine resolvent tolerance < coarse tolerance < 1')
    if args.eigen_tolerance <= 0 or args.identity_tolerance <= 0 or args.maxiter <= 0:
        parser.error('Require positive eigensolver/identity tolerances and maxiter')
    if np.array_equal(args.incoming_momentum, args.outgoing_momentum):
        parser.error('Use distinct momenta for the requested nonforward probe')
    code = PauliCode()
    source = Path(__file__)
    dependencies = ('quantum_spinor_plaquette.py', 'pauli_compton.py', 'rotation_band.py')
    receipt = {'schema': 1,
               'scope': 'finite open planar full interacting same-N matter-only Peierls two-source response',
               'not_claimed': ['asymptotic continuum Compton differential rate',
                               'propagating-photon S matrix', 'total external compact-background response',
                               'infinite electric cutoff convergence'],
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'supplier_sources': {name: hashlib.sha256(source.with_name(name).read_bytes()).hexdigest()
                                    for name in dependencies},
               'source_ownership': 'Actual SpinorPlaquette terms/PauliCode.element/compact shifts; W fixed',
               'intermediate_scope': 'complete raw same-N Gauss complement plus analytic ground-doublet poles',
               'edge_geometry': [{'source': list(CELLS[a // 4]), 'target': list(CELLS[b // 4])}
                                 for a, b in CYCLE_EDGES],
               'cases': [analyze_sector(code, n, cutoff, args)
                         for cutoff in args.cutoffs for n in args.numbers]}
    text = json.dumps(receipt, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()

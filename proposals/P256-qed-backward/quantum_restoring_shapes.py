#!/usr/bin/env python3
"""All 45 harmonic shapes joined to actual Pauli/Gauss compact-photon matter.

The default consumer is a coupled variational state, NOT a quantum ground solve.
Its small arrowhead Schur equation is solved exactly; its full residual is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import cast

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import null_space
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator

import compact_charge_gap as gap
import quantum_spinor_plaquette as spinor
from local_orbital_stability import fourier_sea_hessian, scalar_paths, uniform_data
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis

LINKS = ((0, 1, 0), (1, 2, 1), (3, 2, 0), (0, 3, 1))
ETA = np.array([.19, -.31, .27, -.13])


def real_inner(left, right):
    return float(np.vdot(left, right).real)


def complex_json(value):
    value = np.asarray(value)
    return {'real': value.real.tolist(), 'imaginary': value.imag.tolist()}


def shape_data(code):
    """Actual q=0 Hessian; remove the actual 15 tangents, not named coordinates."""
    basis = matrix_unit_basis()
    data = uniform_data(RotationBand(a=1., m=.2, improved=True), 3, 1., 0., basis)
    hessian = fourier_sea_hessian(data, np.zeros(3, int), 3, basis) + data['elastic']
    if np.linalg.norm(hessian.imag) > 1e-10:
        raise AssertionError('Uniform real-coordinate Hessian is unexpectedly complex')
    complement = null_space(data['orbit'].T)
    curvatures, rotation = np.linalg.eigh(complement.T @ hessian.real @ complement)
    normals = complement @ rotation
    changes = np.einsum('tma,aij->tmij', normals.T.reshape(45, 4, 15), basis)
    if data['orbit'].shape != (60, 15) or complement.shape != (60, 45):
        raise AssertionError('The actual Clifford orbit must have exactly 15 tangents')
    if curvatures[0] <= 0 or np.linalg.norm(hessian @ data['orbit']) > 1e-8:
        raise AssertionError('This selected sea does not supply 45 positive restoring modes')
    signs = np.array([1., -1., -1., -1.])
    transformed_basis = code.local_T @ basis.conjugate() @ code.local_T.conjugate().T
    orbital_T = np.einsum('aij,bji->ba', transformed_basis, basis).real / 4
    fiber_T = np.kron(np.diag(signs), orbital_T)
    normal_T = normals.T @ fiber_T @ normals
    frame_T = signs[:, None, None] * (code.local_T @ data['frame'].conjugate()
                                    @ code.local_T.conjugate().T)
    residuals = {
        'basis_trace_normalization': float(np.linalg.norm(np.einsum('aij,bji->ab', basis, basis) - 4 * np.eye(15))),
        'uniform_orbit_flat': float(np.linalg.norm(hessian @ data['orbit'])),
        'normal_orthonormality': float(np.linalg.norm(normals.T @ normals - np.eye(45))),
        'normal_eigenvectors': float(np.linalg.norm(hessian @ normals - normals * curvatures)),
        'T_frame_even': float(np.linalg.norm(frame_T - data['frame'])),
        'T_restoring_subspace_closed': float(np.linalg.norm(fiber_T @ normals - normals @ normal_T)),
        'T_coordinate_involution': float(np.linalg.norm(normal_T @ normal_T - np.eye(45))),
        'T_hessian_covariance': float(np.linalg.norm(fiber_T @ hessian.real @ fiber_T.T - hessian.real)),
        'T_frequency_covariance': float(np.linalg.norm(normal_T * np.sqrt(curvatures)[None] - np.sqrt(curvatures)[:, None] * normal_T)),
    }
    if max(residuals.values()) > 1e-8:
        raise AssertionError('Restoring basis or derived all-coordinate T law failed')
    return data, curvatures, normals, changes, normal_T, residuals


def planar_blocks(fields, band):
    """RotationBand owns every scalar coefficient; delete absent z Wilson onsite."""
    onsite, paths = scalar_paths(band)
    nearest = {tuple(displacement): coefficients for displacement, coefficients in paths}
    blocks = [(onsite - band.r / band.a) * fields[0]]
    for _, _, axis in LINKS:
        displacement = tuple(np.eye(3, dtype=int)[axis])
        blocks.append(np.einsum('m,mij->ij', nearest[displacement], fields).conjugate().T)
    return np.asarray(blocks)


def coefficient_bound(blocks):
    """Each density/primitive transfer has norm <=1, also on unbounded rotors."""
    return float(4 * np.sum(abs(blocks[0])) + 2 * np.sum(abs(blocks[1:])))


class PrimitiveLift:
    """Shared sparse entry geometry from actual raw Pauli words and rotor shifts.

    No CAR/wedge sign constructs any entry. The reference model is immutable.
    All possible onsite flavour transfers and all 16 flavours of each link exist.
    """
    def __init__(self, model, reference_blocks):
        self.model = model
        specs = []
        for cell in range(4):
            for target, source in itertools.product(range(4), repeat=2):
                if source != target:
                    a, b = 4 * cell + source, 4 * cell + target
                    specs.append((a, b, 4 * target + source, False, None, 0.))
        for link, (source_cell, target_cell, _) in enumerate(LINKS):
            for target, source in itertools.product(range(4), repeat=2):
                a, b = 4 * source_cell + source, 4 * target_cell + target
                path = tuple(dict.fromkeys((a, spinor.ROOTS[source_cell], spinor.ROOTS[target_cell], b)))
                label = 16 * (link + 1) + 4 * target + source
                angle = float(ETA[target_cell] - ETA[source_cell])
                specs.extend(((a, b, label, False, path, angle),
                              (b, a, label, True, path[::-1], -angle)))
        rows, cols, labels, conjugates, amplitudes, angles = [], [], [], [], [], []
        by_mask = {}
        for column, (mask, fields) in enumerate(model.raw):
            if any(model.gauss(mask, fields)):
                raise AssertionError('Supplier domain violates primitive full Gauss')
            by_mask.setdefault(mask, []).append((column, fields))
            for orbital in spinor.occupied(mask):
                rows.append(column); cols.append(column); labels.append(5 * (orbital % 4))
                conjugates.append(False); amplitudes.append(1.); angles.append(0.)
        code_checks = 0
        for source, target, label, conjugate, path, angle in specs:
            path = model.code.words.path(source, target) if path is None else path
            shifts = [0] * 4
            for a, b in zip(path[:-1], path[1:]):
                if (a, b) in spinor.CYCLE_EDGES:
                    shifts[spinor.CYCLE_EDGES.index((a, b))] -= 1
                elif (b, a) in spinor.CYCLE_EDGES:
                    shifts[spinor.CYCLE_EDGES.index((b, a))] += 1
            word = model.code.words.path_A(path).times_i(-1)
            if not word.commutes(model.code.S):
                raise AssertionError('An arbitrary-flavour shape transfer escaped S=+')
            code_checks += 1
            for mask, columns in by_mask.items():
                if not mask >> source & 1 or mask >> target & 1:
                    continue
                new = mask ^ (1 << source) ^ (1 << target)
                amplitude = model.code.element(mask, new, word, model.sign)
                for column, fields in columns:
                    shifted = tuple(e + shift for e, shift in zip(fields, shifts))
                    if any(abs(e) > model.cutoff for e in shifted):
                        continue
                    if any(model.gauss(new, shifted)):
                        raise AssertionError('Actual primitive transfer does not preserve Gauss')
                    row = model.index.get((new, shifted))
                    if row is None:
                        raise AssertionError('Primitive transfer escaped complete supplier domain')
                    rows.append(row); cols.append(column); labels.append(label)
                    conjugates.append(conjugate); amplitudes.append(amplitude); angles.append(angle)
        self.rows, self.cols = np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp)
        self.labels = np.asarray(labels, dtype=np.intp)
        self.conjugates = np.asarray(conjugates, dtype=bool)
        self.amplitudes, self.angles = np.asarray(amplitudes), np.asarray(angles)
        self.phase_factors = {}
        self.code_path_count = code_checks
        self.shape = model.H.shape
        self.reference = self.matrix(reference_blocks)

    def matrix(self, blocks):
        coefficients = np.asarray(blocks).reshape(-1)[self.labels]
        coefficients = np.where(self.conjugates, coefficients.conjugate(), coefficients)
        values = coefficients * self.amplitudes
        result = sparse.coo_matrix((values, (self.rows, self.cols)), shape=self.shape).tocsr()
        result.eliminate_zeros()
        return result

    def peierls(self, operator, derivative=0, theta=0.):
        """Dress an already lifted matrix using primitive PATH angles, not rho."""
        if derivative == 0 and theta == 0:
            return operator
        key = derivative, theta
        if key not in self.phase_factors:
            values = ((1j * self.angles)**derivative if derivative
                      else np.expm1(1j * theta * self.angles))
            factor = sparse.coo_matrix((values, (self.rows, self.cols)), shape=self.shape).tocsr()
            factor.eliminate_zeros()
            self.phase_factors[key] = factor
        changed = operator.multiply(self.phase_factors[key]).tocsr()
        return changed if derivative else operator + changed


def reference_seed(model, parameters):
    """Onsite minimum times neutral rotor, embedded by actual Pauli phases.

    Determinants here only describe a trial vector; they never build H or G_a.
    Charged cell zero has Ncell=1/3 and is balanced by its actual boundary flux.
    """
    if model.N == 8:
        return gap.neutral_trial(model, parameters)[0]
    _, vectors = np.linalg.eigh(model.onsite)
    columns = [0] if model.N == 7 else [0, 1, 2]
    local_occupied = [columns, [0, 1], [0, 1], [0, 1]]
    local = []
    for selected in local_occupied:
        local.append({sum(1 << i for i in pair): np.linalg.det(vectors[np.ix_(pair, selected)])
                      for pair in itertools.combinations(range(4), len(selected))})
    loops, _, rotor = gap.neutral_ground(model.cutoff, model.UE, model.K)
    photons = dict(zip(loops, rotor))
    phases = model.code.intertwiner(model.N, model.sign)
    vector = np.zeros(model.dimension, complex)
    for index, (mask, fields) in enumerate(model.raw):
        masks = [(mask >> (4 * cell)) & 15 for cell in range(4)]
        if len(set(fields)) == 1 and all(masks[c] in local[c] for c in range(4)):
            vector[index] = phases[mask] * math.prod(local[c][masks[c]] for c in range(4)) * photons[fields[0]]
    if abs(real_inner(vector, vector) - 1) > 1e-10:
        raise AssertionError('Charged physical variational seed is not normalized')
    return vector


class QuantumShapes(LinearOperator):
    """Complete total-excitation shells 0 and 1, all 45 modes, matrix-free tensor."""
    def __init__(self, model, operators, curvatures, scale):
        self.physical, self.G = model, operators
        self.scale = scale
        self.omega = scale**.75 * np.sqrt(curvatures)
        # M=Lambda^-1/2, q=(2 M omega)^-1/2.
        self.q = 1 / np.sqrt(2 * scale**-.5 * self.omega)
        self.dimension = 46 * model.dimension
        super().__init__(dtype=np.dtype(complex), shape=(self.dimension, self.dimension))

    def _matvec(self, x):
        x = np.asarray(x).reshape(46, self.physical.dimension)
        result = (self.physical.H @ x.T).T
        result[1:] += self.omega[:, None] * x[1:]
        for mode, operator in enumerate(self.G):
            result[0] += self.q[mode] * (operator @ x[mode + 1])
            result[mode + 1] += self.q[mode] * (operator @ x[0])
        return result.reshape(-1)

    def _rmatvec(self, x):
        return self._matvec(x)

    def _matmat(self, vectors):
        return np.column_stack([self._matvec(vectors[:, j]) for j in range(vectors.shape[1])])

    def gauge_action(self, vector, lift, derivative=0, theta=0.):
        x = np.asarray(vector).reshape(46, self.physical.dimension)
        result = (lift.peierls(lift.reference, derivative, theta) @ x.T).T
        for mode in range(45):
            operator = lift.peierls(self.G[mode], derivative, theta)
            result[0] += self.q[mode] * (operator @ x[mode + 1])
            result[mode + 1] += self.q[mode] * (operator @ x[0])
        return result.reshape(-1)


def schur_trial(quantum, seed):
    """Exact finite Ritz arrowhead solve, explicitly not the full Schur complement."""
    images = np.array([operator @ seed for operator in quantum.G])
    norms = np.linalg.norm(images, axis=1)
    vectors = np.zeros_like(images)
    active = norms > 0
    vectors[active] = images[active] / norms[active, None]
    energy0 = real_inner(seed, quantum.physical.H @ seed)
    offsets = np.array([real_inner(vector, quantum.physical.H @ vector) for vector in vectors])
    offsets += quantum.omega - energy0
    couplings = quantum.q * norms
    if np.min(offsets[active]) <= 0:
        raise ValueError('Selected Lambda is too small for this downward arrowhead root; increase --scales')
    def solve(weights):
        high = float(np.sum(weights / offsets))
        if high == 0:
            return 0.
        return cast(float, brentq(lambda shift: shift - np.sum(weights / (offsets + shift)),
                                 0., high, xtol=1e-300, rtol=4 * np.finfo(float).eps))
    shift = solve(couplings**2)
    amplitudes = -couplings / (offsets + shift)
    vacuum = 1 / np.sqrt(1 + np.sum(amplitudes**2))
    amplitudes *= vacuum
    trial = np.vstack((vacuum * seed, amplitudes[:, None] * vectors))
    applied = quantum @ trial.reshape(-1)
    energy = real_inner(trial.reshape(-1), applied)
    expectation_error = abs(energy - (energy0 - shift))
    if abs(real_inner(trial, trial) - 1) > 1e-10 or expectation_error > 1e-9:
        raise AssertionError('Full coupled trial expectation does not match its Ritz energy')
    reduced_residual = np.linalg.norm(np.concatenate(([shift * vacuum + np.dot(couplings, amplitudes)],
                              couplings * vacuum + (offsets + shift) * amplitudes)))
    if reduced_residual > 1e-9 * (1 + np.max(abs(couplings))):
        raise AssertionError('Finite Ritz Schur solve failed its actual residual')
    mode = int(np.argmax(couplings**2 / offsets))
    omitted_weights = couplings**2
    omitted_weights[mode] = 0
    omitted_shift = solve(omitted_weights)
    omitted_action = np.zeros_like(trial)
    omitted_action[0] = quantum.q[mode] * (quantum.G[mode] @ trial[mode + 1])
    omitted_action[mode + 1] = quantum.q[mode] * (quantum.G[mode] @ trial[0])
    report = {
        'status': 'coupled_variational_trial_NOT_ground',
        'small_solver': 'exact scalar Schur root in the orthonormal active-image Ritz subspace, NOT the full finite Hilbert space',
        'reference_seed_energy': energy0, 'energy_expectation': energy,
        'energy_decrease_stably_evaluated': shift,
        'full_H_expectation_Ritz_energy_residual': expectation_error,
        'reduced_Ritz_residual': float(reduced_residual),
        'full_finite_H_residual': float(np.linalg.norm(applied - energy * trial.reshape(-1))),
        'orthogonal_one_excitation_trial_dimensions': int(np.count_nonzero(active)),
        'Ritz_trial_dimension': 1 + int(np.count_nonzero(active)),
        'complete_oscillator_shell_dimension': 46,
        'omitted_coupling_adverse': {'omitted_mode': mode,
            'full_trial_operator_action_change': float(np.linalg.norm(omitted_action)),
            'omitted_model_optimized_decrease': omitted_shift,
            'lost_energy_lowering': shift - omitted_shift},
    }
    if shift <= 0 or np.linalg.norm(omitted_action) == 0 or shift <= omitted_shift:
        raise AssertionError('Actual shape coupling or omitted-coupling discriminant vanished')
    return trial, report


def coordinate_observables(quantum, trial):
    """P xi^2 P includes excursions OUTSIDE K=1, not (P xi P)^2."""
    populations = np.sum(abs(trial[1:])**2, axis=1)
    overlaps = trial[1:] @ trial[0].conjugate()
    means = 2 * quantum.q * overlaps.real
    seconds = quantum.q**2 * (1 + 2 * populations)
    rho = trial @ trial.conjugate().T
    cross = 2 * quantum.q[:, None] * quantum.q[None, :] * rho[1:, 1:].real
    np.fill_diagonal(cross, seconds)
    couplings = np.array([2 * quantum.q[a] * real_inner(trial[0], quantum.G[a] @ trial[a + 1])
                         for a in range(45)])
    # Norm of the grade-two image: sqrt(2)q_a x_a and q_a x_b
    # for b != a. Do not subtract nearly equal grade-zero moments.
    moment_defect = quantum.q**2 * (np.sum(populations) + populations)
    if np.min(seconds - means**2) <= 0:
        raise AssertionError('Continuum coordinate variance must be positive')
    shell = np.array([any(abs(e) == quantum.physical.cutoff for e in fields)
                      for _, fields in quantum.physical.raw])
    return {'one_excitation_probability_by_mode': populations.tolist(),
            'total_nonvacuum_shape_probability': float(np.sum(populations)),
            'mean_xi': means.tolist(), 'mean_xi_squared': seconds.tolist(),
            'coordinate_covariance': (cross - means[:, None] * means[None, :]).tolist(),
            'xi_G_correlations': couplings.tolist(),
            'total_shape_matter_correlation': float(np.sum(couplings)),
            'shape_physical_linear_entropy': 1 - real_inner(rho, rho),
            'wrong_compressed_coordinate_square_defect_by_mode': moment_defect.tolist(),
            'electric_cutoff_shell_probability': float(np.sum(abs(trial[:, shell])**2)),
            'moment_scope': 'Exact continuum coordinate moments of the represented oscillator wavefunction, including excursions to K=2.'}


def ward_controls(quantum, trial, lift):
    """Peierls derivatives built from primitive coefficients BEFORE any commutator."""
    physical = quantum.physical
    density = np.array([sum(ETA[i // 4] for i in spinor.occupied(mask)) for mask, _ in physical.raw])
    vector = trial.reshape(-1)
    # Oscillator, electric, magnetic and onsite terms commute with cell density;
    # omit them here to avoid subtracting enormous oscillator frequencies.
    h = quantum.gauge_action(vector, lift)
    j = quantum.gauge_action(vector, lift, derivative=1)
    contact = quantum.gauge_action(vector, lift, derivative=2)
    charge_vector = (density[None, :] * trial).reshape(-1)
    hc = quantum.gauge_action(charge_vector, lift)
    hcc = quantum.gauge_action((density[None, :]**2 * trial).reshape(-1), lift)
    expected_j = 1j * ((density[None, :] * h.reshape(46, -1)).reshape(-1) - hc)
    expected_c = (-(density[None, :]**2 * h.reshape(46, -1)).reshape(-1)
                  + 2 * (density[None, :] * hc.reshape(46, -1)).reshape(-1) - hcc)
    reference_j = (lift.peierls(lift.reference, derivative=1) @ trial.T).T.reshape(-1)
    reference_c = (lift.peierls(lift.reference, derivative=2) @ trial.T).T.reshape(-1)
    theta = 2e-4
    plus = quantum.gauge_action(vector, lift, theta=theta)
    minus = quantum.gauge_action(vector, lift, theta=-theta)
    phase = np.exp(1j * theta * density)[None, :]
    unrotated = (phase.conjugate() * trial).reshape(-1)
    conjugated = (phase * quantum.gauge_action(unrotated, lift).reshape(46, -1)).reshape(-1)
    report = {'eta': ETA.tolist(),
        'independent_Peierls_current_commutator_residual': float(np.linalg.norm(j - expected_j)),
        'independent_Peierls_contact_double_commutator_residual': float(np.linalg.norm(contact - expected_c)),
        'finite_pure_gradient_unitary_covariance_residual': float(np.linalg.norm(plus - conjugated)),
        'Peierls_first_derivative_difference_residual': float(np.linalg.norm((plus - minus) / (2 * theta) - j)),
        'Peierls_second_derivative_difference_residual': float(np.linalg.norm((plus + minus - 2 * h) / theta**2 - contact)),
        'omitting_shape_current_action_defect': float(np.linalg.norm(j - reference_j)),
        'omitting_shape_contact_action_defect': float(np.linalg.norm(contact - reference_c)),
        'resolvent_status': 'NOT_RUN: trial is not an eigenstate; no ground Ward cancellation or scattering claim',
        'exact_identity': 'J=i[rho,H], C=-[rho,[rho,H]]; for an exact eigenstate C_expectation-2<J Q(H-E)^(-1)Q J>=0 when reduced resolvent exists',
        'derivative_owner': 'primitive link phases of BOTH reference hopping and every xi-dependent G_a; not defined by the density commutator'}
    if max(report['independent_Peierls_current_commutator_residual'],
           report['independent_Peierls_contact_double_commutator_residual'],
           report['finite_pure_gradient_unitary_covariance_residual']) > 1e-9:
        raise AssertionError('Full shape-dependent Peierls Ward operator identity failed')
    if min(report['omitting_shape_current_action_defect'], report['omitting_shape_contact_action_defect']) <= 0:
        raise AssertionError('Omitting xi-dependent Peierls vertices was not detected')
    return report


def t_controls(model, operators, shape_T):
    rng = np.random.default_rng(920 + model.N)
    vector = rng.normal(size=model.dimension) + 1j * rng.normal(size=model.dimension)
    vector /= np.linalg.norm(vector)
    coordinates = rng.normal(size=45); coordinates /= np.linalg.norm(coordinates)
    image = sum((value * (operator @ vector) for value, operator in zip(coordinates, operators)), np.zeros_like(vector))
    transformed = sum((value * (operator @ model.apply_T(vector))
                      for value, operator in zip(shape_T @ coordinates, operators)), np.zeros_like(vector))
    error = float(np.linalg.norm(model.apply_T(image) - transformed))
    if error > 1e-9:
        raise AssertionError('Derived coordinate-covariant primitive G law failed')
    return {'actual_G_coordinate_covariance_residual': error,
            'physical_reference_T_residual': model.Tsymmetry,
            'physical_T_square_residual': model.Tsquare,
            'law': 'T xi T^-1=S xi, T Pi T^-1=-S Pi; S=V^T diag(+,-,-,-) Ad(local_T K) V; individual arbitrary G_a need not be T-even'}


def run(cutoffs=(1,), scales=(.5, 2., 8.), oscillator_cutoffs=(1,)):
    if not cutoffs or any(c < 1 for c in cutoffs) or not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('Require positive electric cutoffs and finite positive threshold-relative scales')
    if tuple(oscillator_cutoffs) != (1,):
        raise ValueError('This consumer supports the complete K=1 shell only; K=2 is not silently approximated')
    parameters = gap.GapParameters(electric=.6, magnetic=.25, delta=1.2, m=.8, r=.04, v=.08)
    parameters.validate()
    code = spinor.PauliCode()
    data, curvatures, normals, changes, shape_T, shape_checks = shape_data(code)
    gram = np.diag(data['selected']['G'])
    if np.max(abs(gram[1:] - gram[1])) > 1e-10 or np.min(gram) <= 0:
        raise AssertionError('Selected finite cube did not produce a positive cubic Gram')
    bare_band = RotationBand(a=1., m=parameters.m / np.sqrt(gram[0]),
                             r=parameters.r / np.sqrt(gram[0]), v=parameters.v / np.sqrt(gram[1]))
    reference_blocks = planar_blocks(data['frame'], bare_band)
    shape_blocks = np.array([planar_blocks(change, bare_band) for change in changes])
    bounds = np.array([coefficient_bound(blocks) for blocks in shape_blocks])
    constant = float(np.sum(bounds**2 / (2 * curvatures)))
    threshold = constant / parameters.gap_bound
    cases = []
    for cutoff in dict.fromkeys(cutoffs):
        for number in (8, 7, 9):
            model = spinor.SpinorPlaquette(code, number, cutoff, electric=parameters.electric,
                                           magnetic=parameters.magnetic, delta=parameters.delta,
                                           m=parameters.m, r=parameters.r, v=parameters.v)
            lift = PrimitiveLift(model, reference_blocks)
            reference_matter = lift.reference
            onsite, hopping, _, _ = gap.operator_parts(model)
            selected_error = spinor.max_abs(reference_matter - onsite - hopping)
            if selected_error > 1e-10:
                raise AssertionError('Bare coefficients do not recover the immutable reference H')
            operators = [lift.matrix(blocks) for blocks in shape_blocks]
            hermiticity = [spinor.max_abs(operator - operator.conjugate().T) for operator in operators]
            if max(hermiticity) > 1e-10:
                raise AssertionError('An actual arbitrary-flavour restoring G is not Hermitian')
            seed = reference_seed(model, parameters)
            time_reversal = t_controls(model, operators, shape_T)
            primitive_adverse = model.adverse_controls()
            _, epsilon, _ = gap.neutral_ground(cutoff, parameters.electric, parameters.magnetic)
            onsite_lower = (-8 if number == 8 else -7) * parameters.mass
            for factor in scales:
                scale = factor * threshold
                quantum = QuantumShapes(model, operators, curvatures, scale)
                trial, trial_report = schur_trial(quantum, seed)
                delta = constant / scale
                # -K is a rigorous untruncated rotor lower bound; the finite Holder
                # lower epsilon is stronger, but does not become an infinite value.
                infinite_lower = onsite_lower - parameters.magnetic + parameters.electric / 2 * model.boundary**2 - parameters.hopping_bound - delta
                finite_lower = onsite_lower + epsilon + parameters.electric / 2 * model.boundary**2 - parameters.hopping_bound - delta
                trial_report['untruncated_ground_energy_bracket'] = [infinite_lower, trial_report['energy_expectation']]
                trial_report['finite_Gauss_and_complete_K1_ground_energy_bracket'] = [finite_lower, trial_report['energy_expectation']]
                if trial_report['energy_expectation'] < finite_lower - 1e-9:
                    raise AssertionError('Variational energy violates the analytic lower bound')
                cases.append({'N': number, 'electric_cutoff': cutoff, 'oscillator_cutoff': 1,
                    'physical_Gauss_dimension': model.dimension, 'quantum_dimension': quantum.dimension,
                    'threshold_relative_scale': factor, 'Lambda': scale, 'M_Lambda': scale**-.5,
                    'omega_by_mode': quantum.omega.tolist(),
                    'all_45_infinite_charge_gap_bound': parameters.gap_bound - delta,
                    'delta_Lambda': delta, 'positive_gap_certified': bool(scale > threshold),
                    'selected_reference_matrix_residual': selected_error,
                    'primitive_G_hermiticity_residuals': hermiticity,
                    'primitive_arbitrary_flavour_path_count': lift.code_path_count,
                    'all_raw_states_and_transfers_Gauss_valid': True,
                    'primitive_Gauss_adverse_controls': primitive_adverse,
                    'time_reversal': time_reversal, 'trial': trial_report,
                    'coordinates': coordinate_observables(quantum, trial),
                    'Peierls_Ward': ward_controls(quantum, trial, lift)})
                del quantum, trial
            del operators, model, lift, seed, reference_matter, onsite, hopping
    return {'status': '45_shape_Gaussian_EFT_compact_photon_coupled_trials_NOT_full_QED',
        'selection': {'band': {'a': 1., 'm': .2, 'r': 1., 'v': 1., 'improved': True},
            'cube_size': 3, 'stiffness': 1., 'preferred_Gram': 0.,
            'selected_Gram': data['selected']['G'].tolist(), 'selected_fields': complex_json(data['frame']),
            'selected_gradient_residual': float(data['selected']['gradient_residual']),
            'q0_restoring_eigenvalues_all_45': curvatures.tolist(),
            'q0_normal_coordinate_columns_60_by_45': normals.tolist(),
            'normal_field_changes': complex_json(changes), 'derived_coordinate_T_45_by_45': shape_T.tolist(),
            'projection_checks': shape_checks},
        'couplings': {'effective_m': parameters.m, 'effective_r': parameters.r, 'effective_v': parameters.v,
            'bare_m': bare_band.m, 'bare_r': bare_band.r, 'bare_v': bare_band.v,
            'onsite_and_four_reference_link_blocks': complex_json(reference_blocks),
            'analytic_untruncated_G_norm_upper_bounds_Ca': bounds.tolist(),
            'norm_bound_owner': 'sum absolute primitive density and directed transfer coefficients, NOT finite matrix norms'},
        'infinite_gap_theorem': {'reference_gap_bound': parameters.gap_bound,
            'sum_Ca_squared_over_2ka': constant, 'strict_positive_threshold_Lambda': threshold,
            'bound': 'Delta_charge >= Delta_reference_bound - sum_a C_a^2/(2 Lambda k_a)',
            'hypotheses': ['45 measured k_a>0; entire xi in R45; canonical momenta; common vacuum normal ordering',
                'bounded Hermitian primitive G_a independent of xi and Pi; no mutual commutation assumed',
                'untruncated integral Gauss rotors with no cyclic wrap; UE=.6,K=.25,delta=1.2,S=+',
                'reference planar Wilson scalar-U1 links, m=.8,r=.04,v=.08; onsite M=m+2r>0; r,v>=0',
                'charged boundary electric energy UE/2 retained; neutral onsite/photon trial has zero hopping expectation',
                'compact_charge_gap Holder argument on integer offsets; hopping norm <=8 max(r,v)',
                'no arbitrary SU4 links: the scalar-U1 Wilson reference hypothesis is not inherited by SU4 fluctuations'],
            'proof': 'For each a, B_a=a_a+q_a G_a/omega_a gives omega_a B_a^dag B_a-G_a^2/(2 Lambda k_a). Sum positive forms; no simultaneous displaced vacuum is asserted. Neutral common-vacuum product trial has zero xi coupling expectation.'},
        'cases': cases,
        'limitations': ['Harmonic q=0 45-shape EFT around the finite 3^3 one-sea selected Clifford minimum, not the nonlinear 60-field theory.',
            'The zero-preferring Gram elastic energy alone selects zero; the occupied-sea energy supplies this nonzero saddle. Its local uniform restoring Hessian is measured here.',
            'Removing orientation is an explicit gauged-orientation hypothesis or local slice restriction; this CLI has compact U1 photons, NOT dynamical SU4 Gauss/Higgs links.',
            'Every coupled state is variational. The exact small Schur root is NOT a full coupled ground-state solution, and its full finite residual is printed.',
            'Only complete K=1 oscillator shells are implemented; continuum coordinate moments nevertheless include outside-shell excursions.',
            'Finite Ritz energy differences and finite residuals do not certify infinite oscillator/rotor ground gaps; the stated infinite gap comes solely from the operator inequality.',
            'The Ward operator identity is exercised on actual coupled trials, not a full quantum ground resolvent; no scattering, continuum, 3D Higgs or fullQED claim.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs', type=int, nargs='+', default=[1])
    parser.add_argument('--oscillator-cutoffs', type=int, nargs='+', default=[1], choices=[1],
                        help='Complete total-excitation shell cutoff; implemented K=1 includes every mode')
    parser.add_argument('--scales', type=float, nargs='+', default=[.5, 2., 8.],
                        help='Positive multipliers of the derived strict-positive Lambda threshold')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(tuple(args.cutoffs), tuple(args.scales), tuple(args.oscillator_cutoffs))
    names = ('quantum_restoring_shapes.py', 'local_orbital_stability.py', 'orbital_gauge_locking.py',
             'rotation_band.py', 'sea_selection.py', 'soft_budget.py', 'quantum_spinor_plaquette.py',
             'compact_charge_gap.py', 'pauli_compton.py', 'bosonic_statistics.py')
    result['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in names}
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

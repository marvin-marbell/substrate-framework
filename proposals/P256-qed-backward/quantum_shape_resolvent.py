#!/usr/bin/env python3
"""Complete 45-mode K=1 compact-Gauss ground and external Peierls response."""
from __future__ import annotations

import argparse
import hashlib
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
from scipy.optimize import brentq
from scipy.sparse import linalg as sla

import compact_charge_gap as gap
import quantum_compton_response as response
import quantum_restoring_shapes as shapes
import quantum_spinor_plaquette as spinor
from rotation_band import RotationBand


class ActionOperator(sla.LinearOperator):
    """Typed matrix-free action, matching SciPy's subclass constructor."""
    def __init__(self, shape, action):
        self.action = action
        super().__init__(dtype=np.dtype(complex), shape=shape)

    def _matvec(self, x):
        return self.action(np.asarray(x).reshape(-1))


def checked_cg(operator, rhs, preconditioner, tolerance, maxiter):
    iterations = [0]
    def count(_):
        iterations[0] += 1
    solution, status = sla.cg(operator, rhs, M=preconditioner, rtol=tolerance,
                              atol=0., maxiter=maxiter, callback=count)
    residual = float(np.linalg.norm(operator @ solution - rhs))
    norm = float(np.linalg.norm(rhs))
    relative = residual / norm if norm else residual
    if status != 0 or relative > 8 * tolerance:
        raise RuntimeError(f'CG failed: status={status}, measured residual={relative}')
    return solution, {'iterations': iterations[0], 'status': int(status),
                      'rhs_norm': norm, 'absolute_residual': residual,
                      'relative_residual': relative}


class PhysicalInverse:
    """Full physical shifted inverse; diagonal preconditioner, measured residuals."""
    def __init__(self, H, shift, tolerance, maxiter):
        self.H, self.shift = H, shift
        self.tolerance, self.maxiter = tolerance, maxiter
        diagonal = H.diagonal().real + shift
        if np.min(diagonal) <= 0:
            raise ValueError('Physical shifted inverse lacks a positive diagonal')
        self.operator = ActionOperator(H.shape, lambda x: H @ x + shift * x)
        self.preconditioner = ActionOperator(H.shape, lambda x: x / diagonal)
        self.calls, self.iterations, self.max_relative = 0, 0, 0.

    def solve(self, rhs):
        result, receipt = checked_cg(self.operator, rhs, self.preconditioner,
                                     self.tolerance, self.maxiter)
        self.calls += 1
        self.iterations += receipt['iterations']
        self.max_relative = max(self.max_relative, receipt['relative_residual'])
        return result

    def report(self):
        return {'physical_shift': float(self.shift), 'calls': self.calls,
                'iterations': self.iterations, 'maximum_measured_relative_residual': self.max_relative,
                'requested_relative_tolerance': self.tolerance}


class SchurGround:
    """Physical-dimension Schur eigenbranches; no fixed-image Ritz subspace."""
    def __init__(self, quantum, reference_energy, reference_ground, delta, args):
        self.quantum, self.args = quantum, args
        self.reference_energy, self.delta = reference_energy, delta
        self.warm = reference_ground[:, 0].copy()
        self.receipts = []
        self.rank = reference_ground.shape[1]

    def at(self, energy, count=1):
        quantum, H = self.quantum, self.quantum.physical.H
        if energy >= self.reference_energy + float(np.min(quantum.omega)):
            raise ValueError('Schur energy reached the first shell spectrum')
        inverses = [PhysicalInverse(H, float(omega - energy), self.args.inner_tolerance,
                                    self.args.inner_maxiter) for omega in quantum.omega]
        def action(x):
            result = H @ x
            for q, G, inverse in zip(quantum.q, quantum.G, inverses):
                result -= q*q * (G @ inverse.solve(G @ x))
            return result
        # ARPACK's relative tolerance must see H_eff(E), not a Schur eigenvalue
        # crossing zero. Subtract E only from its converged eigenvalues.
        operator = ActionOperator(H.shape, action)
        values, vectors = sla.eigsh(operator, k=1, which='SA', v0=self.warm,
                                    tol=self.args.eigen_tolerance, ncv=28,
                                    maxiter=self.args.eigen_maxiter)
        values -= energy
        self.warm = vectors[:, 0].copy()
        if count > 1:
            # A one-vector Krylov space need not recover a repeated eigenvalue.
            # Reconstruct its actual physical T partner, then solve the ENTIRE
            # orthogonal complement of that freshly computed Schur eigenspace.
            first = vectors[:, 0]
            if quantum.physical.N % 2:
                partner = quantum.physical.apply_T(first)
                if abs(np.vdot(first, partner)) > self.args.identity_tolerance:
                    raise RuntimeError('Schur T partner is not orthogonal')
                vectors = np.column_stack((first, partner))
                values = np.repeat(values, 2)
            nonempty = H.indptr[1:] > H.indptr[:-1]
            Hbound = float(np.add.reduceat(abs(H.data), H.indptr[:-1][nonempty]).max())
            # Below the reference ground, delta itself bounds the inverse sum;
            # using omega_min there would understate modes with larger omega.
            coupling_bound = self.delta / (1-max(energy-self.reference_energy, 0.)/min(quantum.omega))
            penalty = 2*(Hbound+abs(energy)+coupling_bound)+1
            base_frame = vectors
            def deflated(x):
                return action(x)+penalty*base_frame@(base_frame.conj().T@x)
            deflated_operator = ActionOperator(H.shape, deflated)
            rng = np.random.default_rng(829+quantum.physical.N)
            start = rng.normal(size=H.shape[0])+1j*rng.normal(size=H.shape[0])
            start -= base_frame@(base_frame.conj().T@start)
            start /= np.linalg.norm(start)
            next_values, next_vectors = sla.eigsh(
                deflated_operator, k=1, which='SA', v0=start,
                tol=self.args.eigen_tolerance, ncv=28, maxiter=self.args.eigen_maxiter)
            next_vector = next_vectors[:, 0]
            if np.linalg.norm(base_frame.conj().T@next_vector) > self.args.identity_tolerance:
                raise RuntimeError('Complete Schur complement probe leaks into deflated multiplet')
            values = np.r_[values, float(np.vdot(next_vector, action(next_vector)).real) - energy]
            vectors = np.column_stack((base_frame, next_vector))
        residuals = [float(np.linalg.norm(action(vectors[:, j]) - (values[j]+energy)*vectors[:, j]))
                     for j in range(vectors.shape[1])]
        if max(residuals) > self.args.full_residual_target / 4:
            raise RuntimeError('Physical Schur eigenbranch residual exceeds the requested accuracy')
        self.receipts.append({'energy': float(energy), 'schur_eigenvalues': values.tolist(),
                              'schur_eigen_residuals': residuals,
                              'all_45_shifted_inverses': [inverse.report() for inverse in inverses]})
        return values, vectors, inverses

    def reconstruct(self, energy, vacuum, inverses):
        quantum = self.quantum
        state = np.vstack((vacuum, [-q * inverse.solve(G @ vacuum)
                                   for q, G, inverse in zip(quantum.q, quantum.G, inverses)]))
        state = state.reshape(-1)
        state /= np.linalg.norm(state)
        residual = float(np.linalg.norm(quantum @ state - energy * state))
        if residual > self.args.full_residual_target:
            raise RuntimeError(f'Full 46*d eigenresidual {residual} exceeds requested target')
        return state, residual

    def solve(self, reference_next, shape_T):
        reference, delta, args = self.reference_energy, self.delta, self.args
        # The reference energy is numerical. A residual margin accompanies labels;
        # this is not an interval-certified eigensolver or an infinite-spectrum proof.
        margin = 8 * args.full_residual_target
        bracket = (reference - delta - margin, reference + margin)
        low = float(self.at(bracket[0])[0][0]); high = float(self.at(bracket[1])[0][0])
        if low <= 0 or high >= 0:
            raise RuntimeError('Lowest full Schur eigenbranch is not bracketed; no trial fallback')
        def root_value(E):
            # Brent re-requests endpoints already solved for the sign check.
            if E == bracket[0]:
                return low
            if E == bracket[1]:
                return high
            return float(self.at(E)[0][0])
        energy = cast(float, brentq(root_value, *bracket,
                                    xtol=args.full_residual_target / 20, rtol=4*np.finfo(float).eps,
                                    maxiter=args.root_maxiter))
        values, vectors, inverses = self.at(energy)
        state, residual = self.reconstruct(energy, vectors[:, 0], inverses)
        if self.quantum.physical.N % 2:
            partner = full_T(self.quantum, shape_T, state)
            overlap = abs(np.vdot(state, partner))
            if overlap > args.identity_tolerance:
                raise RuntimeError('Actual coupled T partner is not orthogonal')
            frame = np.column_stack((state, partner))
        else:
            overlap = None
            frame = state[:, None]
        ground_residuals = np.linalg.norm(self.quantum @ frame - energy*frame, axis=0)
        if max(ground_residuals) > args.full_residual_target:
            raise RuntimeError('Coupled ground frame fails full Hamiltonian residual')
        # Resolve the next Schur eigenbranch in the complete physical space. At its
        # root exactly rank negative eigenvalues must be present and T-closed.
        count = max(4, self.rank + 2)
        at_ground = self.at(energy, count)[0]
        if np.max(abs(at_ground[:self.rank])) > args.full_residual_target or at_ground[self.rank] <= margin:
            raise RuntimeError('Ground rank is ambiguous in the full Schur spectrum')
        upper = min(reference_next + margin, reference + .9*float(np.min(self.quantum.omega)))
        upper_values = self.at(upper, count)[0]
        if upper_values[self.rank] >= 0:
            raise RuntimeError('Next Schur eigenbranch was not bracketed by reference vacuum upper')
        next_energy = cast(float, brentq(lambda E: float(self.at(E, count)[0][self.rank]),
                                         energy, upper, xtol=args.full_residual_target/20,
                                         rtol=4*np.finfo(float).eps, maxiter=args.root_maxiter))
        next_values, next_vectors, next_inverses = self.at(next_energy, count)
        next_state, next_residual = self.reconstruct(next_energy, next_vectors[:, self.rank], next_inverses)
        if np.linalg.norm(frame.conj().T@next_state) > args.identity_tolerance:
            raise RuntimeError('Next full-H state leaks into the coupled ground frame')
        negative_frame = next_vectors[:, :self.rank]
        transformed = np.column_stack([self.quantum.physical.apply_T(negative_frame[:, j])
                                       for j in range(self.rank)])
        closure = float(np.linalg.norm(transformed-negative_frame@(negative_frame.conj().T@transformed)))
        if self.rank == 2 and closure > args.identity_tolerance:
            raise RuntimeError('Negative Schur inertia frame is not the complete T doublet')
        if np.max(next_values[:self.rank]) >= -margin:
            raise RuntimeError('Next root has unresolved Schur inertia')
        gap_value = next_energy-energy
        usable_gap = gap_value - 4*max(max(ground_residuals), next_residual, args.full_residual_target)
        if usable_gap <= 0:
            raise RuntimeError('No resolved positive complete complement gap')
        return energy, frame, usable_gap, {
            'status': 'numerically_resolved_complete_K1_ground_NOT_trial',
            'full_H_dimension': self.quantum.dimension, 'physical_Schur_dimension': self.quantum.physical.dimension,
            'ground_energy': energy, 'reference_ground_numerical_bracket': list(bracket),
            'requested_full_H_absolute_residual_target': args.full_residual_target,
            'normalized_full_H_ground_residuals': ground_residuals.tolist(),
            'ground_rank': frame.shape[1], 'ground_orthonormality_residual': float(np.linalg.norm(frame.conj().T@frame-np.eye(frame.shape[1]))),
            'coupled_T_partner_overlap': None if overlap is None else float(overlap),
            'next_complete_Schur_root': next_energy, 'next_full_H_residual': next_residual,
            'next_full_state_ground_leakage': float(np.linalg.norm(frame.conj().T@next_state)),
            'next_root_negative_inertia_T_closure': closure,
            'next_root_low_Schur_eigenvalues': next_values.tolist(),
            'measured_complement_gap': gap_value, 'residual_margined_response_gap': usable_gap,
            'spectrum_scope': 'Numerical complete finite-domain eigenbranches with measured residuals and Schur inertia; not interval-certified or infinite-cutoff spectrum.',
            'schur_monotonicity': 'dS/dE=-I-sum q_a^2 G_a R_a(E)^2 G_a < 0 below first shell; inertia(H-E)=inertia(S(E)) there.',
            'schur_solves': self.receipts}


def full_T(quantum, shape_T, vector):
    blocks = np.asarray(vector).reshape(46, quantum.physical.dimension)
    transformed = np.array([quantum.physical.apply_T(block) for block in blocks])
    transformed[1:] = shape_T @ transformed[1:]
    return transformed.reshape(-1)


class QuantumVertex(sla.LinearOperator):
    """External derivative on reference hopping AND each xi G, all 45 modes."""
    def __init__(self, quantum, reference, operators):
        self.quantum, self.reference, self.operators = quantum, reference, operators
        super().__init__(dtype=np.dtype(complex), shape=quantum.shape)

    def _matvec(self, x):
        quantum = self.quantum
        blocks = np.asarray(x).reshape(46, quantum.physical.dimension)
        result = (self.reference @ blocks.T).T
        for a, G in enumerate(self.operators):
            result[0] += quantum.q[a] * (G @ blocks[a+1])
            result[a+1] += quantum.q[a] * (G @ blocks[0])
        return result.reshape(-1)

    def _matmat(self, frame):
        return np.column_stack([self._matvec(frame[:, j]) for j in range(frame.shape[1])])


class ShapePeierls:
    """Primitive labels own oriented link phases; rho only tests the result."""
    def __init__(self, quantum, lift):
        self.quantum, self.lift = quantum, lift

    def angles(self, edges):
        # Geometry is supplied in actual LINKS orientation, not cycle orientation.
        labels = self.lift.labels
        result = np.zeros(len(labels), complex)
        active = labels >= 16
        result[active] = np.asarray(edges)[labels[active]//16-1]
        # Reverse a complex Fourier source by minus, NEVER by complex conjugation.
        result[self.lift.conjugates] *= -1
        return result

    def vertex(self, left, right=None):
        angles = self.angles(left)
        values = 1j*angles if right is None else -angles*self.angles(right)
        # Multiple primitive entries with the same geometric position must own
        # one Hadamard factor, not the sum of repeated copies of that factor.
        positions = self.lift.rows.astype(np.int64)*self.lift.shape[1]+self.lift.cols
        _, indices, inverse = np.unique(positions, return_index=True, return_inverse=True)
        if np.max(abs(values-values[indices][inverse]), initial=0.) > 1e-12:
            raise AssertionError('Conflicting primitive path angles at a shared matrix entry')
        factor = sparse.coo_matrix((values[indices], (self.lift.rows[indices], self.lift.cols[indices])),
                                   shape=self.lift.shape).tocsr()
        reference = self.lift.reference.multiply(factor).tocsr()
        operators = [G.multiply(factor).tocsr() for G in self.quantum.G]
        return QuantumVertex(self.quantum, reference, operators)

    @staticmethod
    def plane_wave(momentum, polarization):
        k, eps = np.asarray(momentum), np.asarray(polarization)
        values = []
        for a, b, _ in shapes.LINKS:
            x, y = np.asarray(spinor.CELLS[a]), np.asarray(spinor.CELLS[b])
            d = y-x
            values.append(np.dot(eps,d)*np.exp(1j*np.dot(k,(x+y)/2))*np.sinc(np.dot(k,d)/(2*np.pi)))
        return np.asarray(values)


class BlockJacobi:
    """All 46 full physical diagonal blocks, positive shifted inverse actions.

    A positive common extra shift is needed because at positive response frequency
    the unprojected vacuum diagonal block need not be positive. No mode is deleted.
    Tight measured inner CG inverses approximate this fixed SPD preconditioner.
    """
    def __init__(self, quantum, energy, frequency, gap_value, args):
        extra = abs(frequency)+gap_value
        shifts = np.r_[0., quantum.omega]-energy+extra
        self.blocks = [PhysicalInverse(quantum.physical.H, float(shift),
                                        args.preconditioner_tolerance, args.inner_maxiter)
                       for shift in shifts]
        self.quantum = quantum
        self.operator = ActionOperator(quantum.shape, self.apply)

    def apply(self, vector):
        blocks = np.asarray(vector).reshape(46, self.quantum.physical.dimension)
        return np.array([inverse.solve(block) for inverse, block in zip(self.blocks, blocks)]).reshape(-1)

    def report(self):
        return {'scope': 'complete 46 physical blocks; fixed positive extra shift; measured inner CG',
                'all_block_inverses': [block.report() for block in self.blocks]}


class CompleteQuantumResolvent(response.CompleteResolvent):
    """Full coupled Q(H-E-frequency)Q+gap P with complete block preconditioning."""
    def __init__(self, quantum, energy, ground, gap_value, tolerance, args):
        super().__init__(quantum, energy, ground, gap_value, tolerance, args.response_maxiter)
        self.args = args
        self.preconditioners = {}

    def solve(self, source, frequency):
        if abs(frequency) >= self.gap:
            raise ValueError('Frequency must be below the actual complete complement gap')
        Qsource = self.project(source)
        def action(x):
            q = self.project(x)
            return self.project(self.model @ q-(self.energy+frequency)*q) \
                + self.gap*self.ground@(self.ground.conj().T@x)
        operator = ActionOperator(self.model.shape, action)
        if frequency not in self.preconditioners:
            self.preconditioners[frequency] = BlockJacobi(self.model, self.energy, frequency, self.gap, self.args)
        block = self.preconditioners[frequency]
        # Project the preconditioner on both sides, retaining a positive P block.
        def precondition(x):
            return self.project(block.operator @ self.project(x)) \
                + self.ground@(self.ground.conj().T@x)/self.gap
        M = ActionOperator(self.model.shape, precondition)
        result = np.empty_like(Qsource)
        for column in range(Qsource.shape[1]):
            result[:, column], receipt = checked_cg(operator, Qsource[:, column], M,
                                                   self.tolerance, self.maxiter)
            receipt.update({'frequency': float(frequency), 'column': column,
                            'ground_leakage': float(np.linalg.norm(self.ground.conj().T@result[:, column]))})
            self.receipts.append(receipt)
            result[:, column] = self.project(result[:, column])
        return result

    def report(self):
        return {'solves': self.receipts,
                'block_Jacobi': [{'frequency': float(frequency), **block.report()}
                                 for frequency, block in self.preconditioners.items()]}


def analyze_response(quantum, lift, energy, frame, gap_value, args):
    peierls = ShapePeierls(quantum, lift)
    chi = np.array((0., .37, -.23, .61))
    gradient = np.array([chi[b]-chi[a] for a,b,_ in shapes.LINKS])
    density = np.array([sum(chi[i//4] for i in spinor.occupied(mask))
                        for mask,_ in quantum.physical.raw])
    density_full = np.tile(density,46)
    J, C = peierls.vertex(gradient), peierls.vertex(gradient,gradient)
    acted, Hframe = J@frame, quantum@frame
    rhoframe = density_full[:,None]*frame
    expected = 1j*(density_full[:,None]*Hframe-quantum@rhoframe)
    double = -density_full[:,None]**2*Hframe + 2*density_full[:,None]*(quantum@rhoframe) \
        - quantum@(density_full[:,None]**2*frame)
    operators = {'independent_current_commutator_residual': float(np.linalg.norm(acted-expected)),
                 'independent_contact_double_commutator_residual': float(np.linalg.norm(C@frame-double))}
    if max(operators.values()) > args.identity_tolerance:
        raise AssertionError('Actual all-shape primitive Ward operators fail')
    def transverse(k):
        k = np.asarray(k)
        if np.linalg.norm(k) == 0:
            raise ValueError('Plane-wave momentum must be nonzero')
        return np.array((-k[1],k[0]))/np.linalg.norm(k)
    incoming = peierls.plane_wave(args.incoming_momentum, transverse(args.incoming_momentum))
    outgoing = peierls.plane_wave(args.outgoing_momentum, transverse(args.outgoing_momentum))
    left, right = peierls.vertex(outgoing.conjugate()), peierls.vertex(incoming)
    contact = peierls.vertex(outgoing.conjugate(), incoming)
    minus = peierls.vertex(incoming.conjugate())
    adjoint = float(np.linalg.norm(frame.conj().T@(minus@frame)-(right@frame).conj().T@frame))
    # Full sparse block adjoints independently exercise J_q^dag=J_-q, not only P.
    adjoint_blocks = max([spinor.max_abs(minus.reference-right.reference.conj().T)] +
                         [spinor.max_abs(a-b.conj().T) for a,b in zip(minus.operators,right.operators)])
    if adjoint_blocks > args.identity_tolerance:
        raise AssertionError('Complex-source Peierls adjoint orientation failed')
    frequency = args.gap_fraction*gap_value
    outputs, receipts = [], []
    for tolerance in (args.coarse_tolerance,args.resolvent_tolerance):
        solver = CompleteQuantumResolvent(quantum,energy,frame,gap_value,tolerance,args)
        ward = solver.kernel(J,J,C,0.)
        physical = solver.kernel(left,right,contact,frequency)
        outputs.append((ward,physical)); receipts.append(solver.report())
    ward, physical = outputs[-1]
    convergence = {'Ward_kernel_change': float(np.linalg.norm(ward[0]-outputs[0][0][0])),
                   'nonforward_kernel_change': float(np.linalg.norm(physical[0]-outputs[0][1][0]))}
    ward_error = float(np.linalg.norm(ward[0]))
    ground_current = frame.conj().T@acted
    ground_pole_numerator = ground_current@ground_current
    frozen_pair = -ground_pole_numerator/frequency+ground_pole_numerator/frequency
    frozen_kernel = ward[1]-frozen_pair
    adverse_scale = max(100*ward_error,args.identity_tolerance)
    deleted_norm = float(np.linalg.norm(-ward[2]))
    frozen_norm = float(np.linalg.norm(frozen_kernel))
    adverse = {'deleted_contact_Ward_norm': deleted_norm,
               'frozen_to_P_intermediate_Ward_norm': frozen_norm,
               'actual_ground_pole_numerator_norm': float(np.linalg.norm(ground_pole_numerator)),
               'frozen_pair_static_continuation_norm': float(np.linalg.norm(frozen_pair)),
               'deleted_contact_breaks_Ward': deleted_norm>adverse_scale,
               'frozen_intermediates_break_Ward': frozen_norm>adverse_scale}
    # Same-source P poles cancel in the exact static continuation; the contact
    # above remains the actual primitive derivative on the actual coupled frame.
    if ward_error > args.identity_tolerance or max(convergence.values()) > args.identity_tolerance:
        raise AssertionError('Complete quantum response fails Ward or tolerance convergence')
    if not adverse['deleted_contact_breaks_Ward'] or not adverse['frozen_intermediates_break_Ward']:
        raise AssertionError('Deleted-contact/frozen-intermediate controls did not actually fail')
    exact_solution = -1j*(rhoframe-frame@(frame.conj().T@rhoframe))
    identity = float(np.linalg.norm(quantum@exact_solution-energy*exact_solution-acted))
    solver = CompleteQuantumResolvent(quantum,energy,frame,gap_value,args.resolvent_tolerance,args)
    actual_solution = solver.solve(acted,0.)
    solution_difference = float(np.linalg.norm(actual_solution-exact_solution))
    if max(identity,solution_difference) > args.identity_tolerance:
        raise AssertionError('Full exact density Ward solution fails')
    rank = frame.shape[1]
    U = np.array(((1.,1j),(1j,1.)))/np.sqrt(2) if rank==2 else np.array([[np.exp(.37j)]])
    rotated_solver = CompleteQuantumResolvent(quantum,energy,frame@U,gap_value,args.resolvent_tolerance,args)
    rotated = rotated_solver.kernel(left,right,contact,frequency)[0]
    covariance = float(np.linalg.norm(rotated-U.conj().T@physical[0]@U))
    if covariance > args.identity_tolerance:
        raise AssertionError('Recomputed coupled ground-frame covariance fails')
    return {'scope': 'finite planar external matter-source response; magnetic compact W fixed; full coupled ground and intermediate domain',
            'frequency': frequency, 'gap_fraction': args.gap_fraction,
            'complex_source_adjoint_full_block_residual': adjoint_blocks,
            'complex_source_adjoint_ground_residual': adjoint,
            'Ward': {'chi': chi.tolist(), 'LINKS_gradient': gradient.tolist(),
                     'operator_identities': operators, 'contact': response.invariant(ward[1]),
                     'complete_resolvent_pair': response.invariant(ward[2]),
                     'contact_minus_pair': response.invariant(ward[0]),
                     'exact_minus_i_QrhoF_equation_residual': identity,
                     'numerical_vs_exact_density_solution': solution_difference,
                     'adverse_controls': adverse, 'density_solution_resolvents': solver.report()},
            'nonforward': {'incoming_momentum': args.incoming_momentum, 'outgoing_momentum': args.outgoing_momentum,
                           'incoming_LINKS_amplitudes': shapes.complex_json(incoming),
                           'outgoing_LINKS_amplitudes': shapes.complex_json(outgoing),
                           'contact': response.invariant(physical[1]),
                           'complete_complement_pair': response.invariant(physical[2]),
                           'ground_Born': response.invariant(physical[3]),
                           'full_kernel': response.invariant(physical[0]),
                           'independently_recomputed_frame_covariance_residual': covariance},
            'tolerance_convergence': convergence, 'resolvents': receipts,
            'rotated_frame_resolvents': rotated_solver.report()}


def run(args):
    parameters = gap.GapParameters(electric=.6,magnetic=.25,delta=1.2,m=.8,r=.04,v=.08)
    parameters.validate()
    code = spinor.PauliCode()
    data, curvatures, normals, changes, shape_T, checks = shapes.shape_data(code)
    gram = np.diag(data['selected']['G'])
    if np.max(abs(gram[1:]-gram[1])) > 1e-10 or np.min(gram)<=0:
        raise AssertionError('Selected Gram is not positive cubic')
    band = RotationBand(a=1.,m=parameters.m/np.sqrt(gram[0]),
                        r=parameters.r/np.sqrt(gram[0]),v=parameters.v/np.sqrt(gram[1]))
    reference_blocks = shapes.planar_blocks(data['frame'],band)
    shape_blocks = [shapes.planar_blocks(change,band) for change in changes]
    bounds = np.array([shapes.coefficient_bound(blocks) for blocks in shape_blocks])
    constant = float(np.sum(bounds**2/(2*curvatures)))
    threshold = constant/parameters.gap_bound
    cases = []
    for cutoff in dict.fromkeys(args.cutoffs):
        for number in dict.fromkeys(args.numbers):
            model = spinor.SpinorPlaquette(code,number,cutoff,electric=parameters.electric,
                                           magnetic=parameters.magnetic,delta=parameters.delta,
                                           m=parameters.m,r=parameters.r,v=parameters.v)
            reference_energies, reference_frame, reference_spectrum = spinor.solve_low(model,tol=args.eigen_tolerance)
            reference_energy = float(reference_energies[0])
            reference_residual = max(reference_spectrum['ground_eigen_residuals']+
                                     [reference_spectrum['first_complement_eigen_residual']])
            if reference_residual > args.full_residual_target:
                raise RuntimeError('Actual reference ground/complement solve is insufficiently accurate')
            lift = shapes.PrimitiveLift(model,reference_blocks)
            onsite,hopping,_,_ = gap.operator_parts(model)
            selected_error = spinor.max_abs(lift.reference-onsite-hopping)
            if selected_error > 1e-10:
                raise AssertionError('Selected reference coefficients do not reproduce physical hopping/onsite')
            operators = [lift.matrix(blocks) for blocks in shape_blocks]
            hermiticity = [spinor.max_abs(G-G.conj().T) for G in operators]
            if max(hermiticity)>1e-10:
                raise AssertionError('Actual G is not Hermitian')
            for factor in dict.fromkeys(args.scales):
                scale = factor*threshold
                quantum = shapes.QuantumShapes(model,operators,curvatures,scale)
                delta = constant/scale
                supplier = SchurGround(quantum,reference_energy,reference_frame,delta,args)
                energy,frame,gap_value,spectrum = supplier.solve(reference_spectrum['first_complement_energy'],shape_T)
                state = frame[:,0].reshape(46,model.dimension)
                coordinates = shapes.coordinate_observables(quantum,state)
                # The represented state is narrow in excitation PROBABILITY, not
                # necessarily in field amplitude. Full continuum coordinate moments
                # include K=2 excursions and every one-excitation population.
                sigmavariance = np.sum(np.diag(np.asarray(coordinates['coordinate_covariance'])))/60
                t_error = float(np.linalg.norm(full_T(quantum,shape_T,quantum@frame[:,0])-
                                               quantum@full_T(quantum,shape_T,frame[:,0])))
                if t_error > args.identity_tolerance:
                    raise RuntimeError('Actual complete coupled Hamiltonian fails shape-covariant T')
                cases.append({'N':number,'electric_cutoff':cutoff,'oscillator_cutoff':1,
                              'physical_dimension':model.dimension,'full_H_dimension':quantum.dimension,
                              'threshold_relative_scale':factor,'Lambda':scale,'delta_Lambda':delta,
                              'infinite_Gaussian_EFT_charge_gap_bound':parameters.gap_bound-delta,
                              'omega_by_mode':quantum.omega.tolist(),'q_by_mode':quantum.q.tolist(),
                              'reference_spectrum':reference_spectrum,'selected_reference_matrix_residual':selected_error,
                              'G_hermiticity_residuals':hermiticity,'coupled_T_H_residual':t_error,
                              'spectrum':spectrum,'coordinates':coordinates,
                              'full_wavefunction_sigma_coordinate_variance':float(sigmavariance),
                              'sigma_definition':'sigma^2=(1/60) sum_a Var(xi_a) in orthonormal 60-coordinate restoring slice; not a nonlinear quartic root width',
                              'response':analyze_response(quantum,lift,energy,frame,gap_value,args)})
                del quantum, supplier, frame, state
            del model,lift,operators,reference_frame
    return {'status':'actual_all45_complete_K1_Gaussian_EFT_ground_and_quantum_Peierls_response_NOT_fullQED',
            'selection':{'selected_Gram':data['selected']['G'].tolist(),
                         'all_45_curvatures':curvatures.tolist(),'projection_checks':checks,
                         'normal_coordinate_columns':normals.tolist(),'shape_T':shape_T.tolist()},
            'infinite_Gaussian_EFT_theorem':{'A':constant,'strict_positive_Lambda_threshold':threshold,
                                           'reference_charge_gap_bound':parameters.gap_bound,
                                           'inequality':'Delta_charge >= .54-A/Lambda',
                                           'scope':'Exact bounded-linear-coupling Gaussian shape EFT on untruncated compact Gauss rotors, not a nonlinear 3D phase.'},
            'cases':cases,'limitations':['Only full total-excitation K=1; no infinite-oscillator or electric-cutoff convergence asserted.',
                                        'Finite external Peierls matter-source response, not asymptotic photons or a photon S matrix.',
                                        'Harmonic q=0 restoring slice around a selected finite-cube sea; not global quartic selection or nonlinear 3D symmetry breaking.',
                                        'Compact U1 Gauss only; removal of 15 orientation modes remains a separate slice/gauged-orientation hypothesis.',
                                        'Numerical eigenspectrum and measured residuals, not interval-certified eigenvalue bounds.',
                                        'Small one-excitation probability does not imply small full coordinate variance or narrow nonlinear quartic wavefunction.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs',nargs='+',type=int,default=[1])
    parser.add_argument('--numbers',nargs='+',type=int,choices=(7,8,9),default=[7,8,9])
    parser.add_argument('--scales',nargs='+',type=float,default=[8.])
    parser.add_argument('--full-residual-target',type=float,default=2e-8)
    parser.add_argument('--eigen-tolerance',type=float,default=2e-12)
    parser.add_argument('--inner-tolerance',type=float,default=2e-13)
    parser.add_argument('--preconditioner-tolerance',type=float,default=2e-12)
    parser.add_argument('--coarse-tolerance',type=float,default=1e-8)
    parser.add_argument('--resolvent-tolerance',type=float,default=2e-11)
    parser.add_argument('--identity-tolerance',type=float,default=2e-6)
    parser.add_argument('--eigen-maxiter',type=int,default=4000)
    parser.add_argument('--inner-maxiter',type=int,default=1000)
    parser.add_argument('--response-maxiter',type=int,default=2000)
    parser.add_argument('--root-maxiter',type=int,default=60)
    parser.add_argument('--gap-fraction',type=float,default=.23)
    parser.add_argument('--incoming-momentum',nargs=2,type=float,default=[.65,.2])
    parser.add_argument('--outgoing-momentum',nargs=2,type=float,default=[-.25,.55])
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    tolerances = (args.full_residual_target,args.eigen_tolerance,args.inner_tolerance,
                  args.preconditioner_tolerance,args.coarse_tolerance,args.resolvent_tolerance,args.identity_tolerance)
    if any(not math.isfinite(t) or t<=0 for t in tolerances):
        parser.error('Require finite positive residual targets and solver tolerances')
    if not 0<args.resolvent_tolerance<args.coarse_tolerance<1 or not 0<args.gap_fraction<1:
        parser.error('Require fine<coarse<1 and a positive strictly subgap frequency fraction')
    if any(c<1 for c in args.cutoffs) or any(not math.isfinite(s) or s<=0 for s in args.scales):
        parser.error('Require positive electric cutoffs and finite positive scale factors')
    if min(args.eigen_maxiter,args.inner_maxiter,args.response_maxiter,args.root_maxiter)<=0:
        parser.error('Require positive solver iteration limits')
    if not all(math.isfinite(k) for k in args.incoming_momentum+args.outgoing_momentum):
        parser.error('Require finite momenta')
    if args.incoming_momentum==args.outgoing_momentum:
        parser.error('Nonforward probe requires distinct momenta')
    result = run(args)
    dependencies = ('quantum_shape_resolvent.py','quantum_restoring_shapes.py','quantum_compton_response.py',
                    'quantum_spinor_plaquette.py','compact_charge_gap.py','local_orbital_stability.py',
                    'orbital_gauge_locking.py','rotation_band.py','sea_selection.py','soft_budget.py',
                    'pauli_compton.py','bosonic_statistics.py')
    result['source_sha256'] = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in dependencies}
    payload = json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(payload)
    print(payload,end='')


if __name__ == '__main__':
    main()

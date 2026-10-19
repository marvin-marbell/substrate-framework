#!/usr/bin/env python3
"""Actual Pauli-word boundary charge intertwiner and SU(4)-Gauss trial cost.

The atomic consumer is the fixed-orbit J=0 EFT, not a 60-field simulation.
The accompanying note proves the separate unrestricted quartic moment theorem.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh

from local_orbital_stability import uniform_data
from quantum_spinor_plaquette import MARKER, PauliCode
from quantum_su4_gauss import (CUTOFFS, HaarLink, PhysicalHamiltonian,
                               PrimitiveMatter, norm, require_small,
                               source_hashes, sparse_sum)
from rotation_band import RotationBand
from sea_selection import matrix_unit_basis

ATOMIC_GAUSS_SOURCE = '9e91c25bc4bcbc5f2306b38d4716b4e3af4664fc22757aedda527c056aea096b'


def boundary_word(code, orbital, creation):
    """Same actual directed marker path as spinor_plaquette.creator_overlap."""
    source, target = (MARKER, orbital) if creation else (orbital, MARKER)
    return code.words.path_A(code.words.path(source, target)).times_i(-1)


def rectangular_ladder(code, source_masks, target_masks, orbital, creation):
    target_index = {mask: row for row, mask in enumerate(target_masks)}
    rows, columns, values = [], [], []
    word = boundary_word(code, orbital, creation)
    for column, mask in enumerate(source_masks):
        if bool(mask >> orbital & 1) == creation:
            continue
        new = mask ^ (1 << orbital)
        if new not in target_index:
            raise AssertionError('Actual boundary word escaped the declared occupation domain')
        rows.append(target_index[new]); columns.append(column)
        values.append(code.element(mask, new, word, sign=1))
    return sparse.coo_matrix((values, (rows, columns)),
        shape=(len(target_masks), len(source_masks)), dtype=complex).tocsr()


class BoundaryChargeIntertwiner:
    """Unnormalized A=sum_i c_Ri tensor|i>, or c_Ri^dagger tensor|bar i>.

    Complete matter fibres and both colour-link indices are retained. The only
    null sectors are the genuine empty/full right light occupations.
    """

    def __init__(self, code, base, charged):
        if base.q != 0 or charged.q not in (-1, 1):
            raise ValueError('Use an NL=4 neutral base and NL=3/5 charged target')
        if base.link is not charged.link or base.bd != 1 or charged.bd != 4:
            raise ValueError('Intertwining requires the same Haar link and prescribed boundary')
        if (base.inertia, base.ue4, base.ue1, base.heavy_multiplier) != (
                charged.inertia, charged.ue4, charged.ue1, charged.heavy_multiplier):
            raise ValueError('The two Hamiltonians must have identical microscopic coefficients')
        self.base, self.charged = base, charged
        self.creation = charged.q == 1
        self.ladders = [rectangular_ladder(code, base.matter.masks,
            charged.matter.masks, 12 + i, self.creation) for i in range(4)]
        self.components, self.matrix = {}, {}
        for name, block in base.blocks.items():
            target = charged.blocks[name]
            if block['shape'][1:3] != target['shape'][1:3]:
                raise AssertionError('Charge dressing changed colour-link indices')
            components = [c[target['keep']][:, block['keep']].tocsr() for c in self.ladders]
            # Matter x boundary ordering matches the actual final tensor axis.
            matrix = sparse_sum(sparse.kron(c, sparse.csr_matrix(np.eye(4)[:, i:i + 1]),
                                           format='csr') for i, c in enumerate(components))
            self.components[name], self.matrix[name] = components, matrix
        counts = 4 - base.matter.light_left
        self.right_number = counts
        self.norm_counts = 4 - counts if self.creation else counts

    def apply(self, vector):
        result = {}
        for name, x in self.base.unpack(vector).items():
            shape = self.charged.blocks[name]['shape']
            y = np.zeros(shape, complex)
            for i, c in enumerate(self.components[name]):
                y[..., i] = (c @ x[..., 0].reshape(x.shape[0], -1)).reshape(shape[:-1])
            result[name] = y
        return self.charged.pack(result)

    def adjoint(self, vector):
        result = {}
        for name, y in self.charged.unpack(vector).items():
            shape = self.base.blocks[name]['shape']
            x = np.zeros(shape, complex)
            for i, c in enumerate(self.components[name]):
                x[..., 0] += (c.getH() @ y[..., i].reshape(y.shape[0], -1)).reshape(shape[:-1])
            result[name] = x
        return self.base.pack(result)

    def weight(self, vector):
        return self.base.pack({name: self.norm_counts[self.base.blocks[name]['keep']][:, None, None, None] * x
                              for name, x in self.base.unpack(vector).items()})

    def normalized(self, vector):
        image = self.apply(vector)
        norm_squared = float(np.vdot(image, image).real)
        if norm_squared <= 0:
            raise ValueError('Boundary partial isometry excludes the actual zero-weight sector')
        return image / np.sqrt(norm_squared), norm_squared




def generator_action(h, name, x, cell, a, boundary_scale=1.):
    result = h.matter_action(h.blocks[name]['Q'][cell][a], x)
    if cell == 1:
        result += boundary_scale * np.einsum('zw,mabw->mabz', h.boundary[a], x, optimize=True)
    return result


def casimir_action(h, name, x, cell, boundary_scale=1.):
    return sum(generator_action(h, name, generator_action(h, name, x, cell, a,
               boundary_scale), cell, a, boundary_scale) for a in range(15))


def angular_action(h, name, x, boundary_scale=1.):
    """Independent sum of all actual (E-Q)^2, including adverse boundary choices."""
    result = np.zeros_like(x)
    rep = h.link.reps[name]
    for a, t in enumerate(rep):
        def left(y):
            return np.einsum('ca,mabz->mcbz', t.T, y, optimize=True) - generator_action(h, name, y, 0, a)
        def right(y):
            return -np.einsum('cb,mabz->macz', t, y, optimize=True) - generator_action(h, name, y, 1, a, boundary_scale)
        result += left(left(x)) + right(right(x))
    return result / (2 * h.inertia)


class HamiltonianParts:
    """Actual onsite/edge restrictions; every existing edge and adjoint is used."""

    def __init__(self, h):
        self.h = h
        self.onsite = {}
        for flavour in range(2):
            coefficient = h.heavy_multiplier * h.heavy_onsite_scalar if flavour == 0 else h.onsite_scalar
            for cells in ((0,), (1,), (0, 1)):
                matrix = sparse_sum(coefficient * h.frame[0, i, j] * h.matter.units[cell, flavour, i, j]
                    for cell in cells for i in range(4) for j in range(4)).tocsr()
                self.onsite[flavour, cells] = {name: matrix[b['keep']][:, b['keep']].tocsr()
                    for name, b in h.blocks.items()}

    def onsite_action(self, vector, flavour, cells=(0, 1)):
        return self.h.pack({name: self.h.matter_action(self.onsite[flavour, cells][name], x)
                           for name, x in self.h.unpack(vector).items()})

    def hopping(self, vector, flavour):
        h = self.h
        x = h.unpack(vector)
        result = {name: np.zeros_like(v) for name, v in x.items()}
        for r, s, f, mm, lm in h.edges:
            if f != flavour:
                continue
            y = h.matter_action(mm, x[s])
            m, _, _, b = y.shape
            dr = h.link.reps[r].shape[1]
            result[r] += (lm @ y.reshape(m, -1, b).transpose(1, 0, 2).reshape(lm.shape[1], -1)).reshape(dr, dr, m, b).transpose(2, 0, 1, 3)
            y = h.matter_action(mm.getH(), x[r])
            m, _, _, b = y.shape
            ds = h.link.reps[s].shape[1]
            result[s] += (lm.getH() @ y.reshape(m, -1, b).transpose(1, 0, 2).reshape(lm.shape[0], -1)).reshape(ds, ds, m, b).transpose(2, 0, 1, 3)
        return h.pack(result)

    def flavour(self, vector, flavour):
        return self.onsite_action(vector, flavour) + self.hopping(vector, flavour)

    def angular(self, vector):
        return self.h.pack({name: self.h.orientation(name, x) for name, x in self.h.unpack(vector).items()})

    def electric(self, vector):
        return self.h.pack({name: self.h.diagonal[name][:, None, None, None] * x
                           for name, x in self.h.unpack(vector).items()})

    def reconstruction(self, vector):
        return norm(self.h.apply(vector) - self.angular(vector) - self.electric(vector)
                    - self.flavour(vector, 0) - self.flavour(vector, 1))


def deterministic_vector(dimension, offset=0):
    k = np.arange(dimension, dtype=float) + 1 + offset
    x = np.sin(k * .317) + 1j * np.cos(k * .193)
    return x / norm(x)


def primitive_one_cell(code, generators):
    """All 16 local occupations, built only with actual code-word transitions."""
    active = tuple(range(12, 16))
    frozen_heavy = 15
    masks = {n: [frozen_heavy | sum(1 << i for i in occ)
                 for occ in itertools.combinations(active, n)] for n in range(5)}
    full_masks = [mask for n in range(5) for mask in masks[n]]
    c = [rectangular_ladder(code, full_masks, full_masks, i, False) for i in active]
    identity = sparse.eye(16, format='csr')
    car = max(norm(c[i] @ c[j].conj().T + c[j].conj().T @ c[i] - (identity if i == j else 0))
              for i in range(4) for j in range(4))
    nilpotent = max(norm(c[i] @ c[j] + c[j] @ c[i]) for i in range(4) for j in range(4))
    require_small(car, 'full actual local boundary CAR')
    require_small(nilpotent, 'full actual local boundary nilpotency')
    q = [sparse_sum(t[i, j] * c[i].conj().T @ c[j] for i in range(4) for j in range(4))
         for t in generators]
    indices = {n: np.array([full_masks.index(mask) for mask in masks[n]]) for n in range(5)}
    neutral_q = [g[indices[2]][:, indices[2]] for g in q]
    neutral_cas = sparse_sum(g @ g for g in neutral_q).toarray()
    require_small(norm(neutral_cas - 2.5 * np.eye(6)), 'primitive wedge2 Casimir')
    reports = []
    for creation, number in ((False, 1), (True, 3)):
        components = [rectangular_ladder(code, masks[2], masks[number], i, creation) for i in active]
        a = sparse_sum(sparse.kron(ci, sparse.csr_matrix(np.eye(4)[:, i:i + 1]), format='csr')
                       for i, ci in enumerate(components)).toarray() / np.sqrt(2)
        qs = [g[indices[number]][:, indices[number]].toarray() for g in q]
        boundary = -generators.conj() if creation else generators
        total_q = [np.kron(g, np.eye(4)) + np.kron(np.eye(4), b) for g, b in zip(qs, boundary)]
        total_cas = sum(g @ g for g in total_q)
        matter_cas = sum(g @ g for g in qs)
        residual = max(norm(g @ a - a @ n.toarray()) for g, n in zip(total_q, neutral_q))
        complement = np.eye(16) - a @ a.conj().T
        image_residual = norm(total_cas @ a - 2.5 * a)
        complement_residual = norm(total_cas @ complement - 4.5 * complement)
        require_small(norm(a.conj().T @ a - np.eye(6)), 'one-cell partial isometry on N=2')
        require_small(residual, 'one-cell all15 generators')
        require_small(image_residual, 'one-cell dressed Casimir')
        require_small(complement_residual, 'one-cell symmetric complement Casimir')
        require_small(norm(matter_cas - 1.875 * np.eye(4)), 'one-cell bare charged matter Casimir')
        reports.append(dict(creation=creation, source_N=2, target_N=number,
            image_dimension=6, complement_dimension=10, all15_generator_residual=residual,
            dressed_image_Casimir=2.5, complement_Casimir=4.5, bare_charged_matter_Casimir=1.875,
            dressed_image_residual=image_residual, complement_residual=complement_residual,
            invalid_matter_only_Casimir_defect=norm(np.kron(matter_cas, np.eye(4)) @ a - 2.5 * a)))
    return dict(full_local_occupations=list(range(5)), full_local_dimension=16,
                actual_word_CAR_residual=car, actual_word_nilpotency_residual=nilpotent, maps=reports)


def spectator_action(h, columns, magnetic=.37):
    """Three independent ell samples; no claim of an infinite-loop truncation bound."""
    ell = np.array([-1., 0., 1.])
    shift = np.diag(np.ones(2), 1)
    magnetic_matrix = -magnetic * (shift + shift.T)
    result = columns @ magnetic_matrix.T
    for k, value in enumerate(ell):
        result[:, k] += h.pack({name: h.ue1 / 2 * (
            (value + h.matter.light_left[b['keep']] - 2)**2 + value**2)[:, None, None, None] * x
            for (name, b), x in zip(h.blocks.items(), h.unpack(columns[:, k]).values())})
    return result


def operator_evidence(a, neutral_parts, charged_parts):
    base, charged = a.base, a.charged
    x = deterministic_vector(base.dimension)
    image = a.apply(x)
    residuals = dict(norm_identity=norm(a.adjoint(image) - a.weight(x)),
                     all15_generators=0., Casimirs=0., angular=0., angular_expansion=0.,
                     wrong_boundary_sign=0., omitted_boundary=0.)
    local_norm = 0.
    for name, matrix in a.matrix.items():
        block = base.blocks[name]
        expected = sparse.diags(a.norm_counts[block['keep']], format='csr', dtype=float)
        local_norm = max(local_norm, norm(matrix.getH() @ matrix - expected))
        for cell in range(2):
            for k in range(15):
                qs = block['Q'][cell][k]
                qt = sparse.kron(charged.blocks[name]['Q'][cell][k], sparse.eye(4), format='csr')
                if cell == 1:
                    qt += sparse.kron(sparse.eye(matrix.shape[0] // 4), sparse.csr_matrix(charged.boundary[k]), format='csr')
                residuals['all15_generators'] = max(residuals['all15_generators'], norm(qt @ matrix - matrix @ qs))
    residuals['norm_operator_identity'] = local_norm
    neutral_blocks, charged_blocks = base.unpack(x), charged.unpack(image)
    for cell in range(2):
        before = base.pack({n: casimir_action(base, n, v, cell) for n, v in neutral_blocks.items()})
        after = charged.pack({n: casimir_action(charged, n, v, cell) for n, v in charged_blocks.items()})
        residuals['Casimirs'] = max(residuals['Casimirs'], norm(after - a.apply(before)))
    residuals['angular'] = norm(charged_parts.angular(image) - a.apply(neutral_parts.angular(x)))
    residuals['full_heavy'] = norm(charged_parts.flavour(image, 0) - a.apply(neutral_parts.flavour(x, 0)))
    residuals['SU4_and_tree_electric'] = norm(charged_parts.electric(image) - a.apply(neutral_parts.electric(x)) - base.ue1 / 2 * image)
    residuals['neutral_full_H_reconstruction'] = neutral_parts.reconstruction(x)
    residuals['charged_full_H_reconstruction'] = charged_parts.reconstruction(image)
    for h, blocks in ((base, neutral_blocks), (charged, charged_blocks)):
        residuals['angular_expansion'] = max(residuals['angular_expansion'],
            norm(h.pack({n: angular_action(h, n, v) - h.orientation(n, v) for n, v in blocks.items()})))
    for scale, label in ((-1., 'wrong_boundary_sign'), (0., 'omitted_boundary')):
        wrong = charged.pack({n: angular_action(charged, n, v, scale) for n, v in charged_blocks.items()})
        residuals[label] = norm(wrong - a.apply(neutral_parts.angular(x)))
    columns = np.column_stack([deterministic_vector(base.dimension, k * 19) for k in range(3)])
    images = np.column_stack([a.apply(columns[:, k]) for k in range(3)])
    action = spectator_action(base, columns)
    residuals['loop_identity_electric_and_magnetic'] = norm(spectator_action(charged, images) -
        np.column_stack([a.apply(action[:, k]) for k in range(3)]))
    for label, value in residuals.items():
        if label not in ('wrong_boundary_sign', 'omitted_boundary'):
            require_small(value, label, 2e-8)
    if min(residuals['wrong_boundary_sign'], residuals['omitted_boundary']) <= 1e-5:
        raise AssertionError('Actual boundary-sign/omission adverse controls vanished')
    occupancy = {str(n): sum(int(np.count_nonzero(a.right_number[b['keep']] == n))
                            * b['shape'][1]**2 for b in base.blocks.values()) for n in range(5)}
    if any(count == 0 for count in occupancy.values()):
        raise AssertionError('The consumer unexpectedly discarded a native right occupation')
    return dict(charge=charged.q, boundary=charged.boundary_name,
        residuals=residuals, native_right_Nlight_dimensions=occupancy,
        zero_weight_sector=4 if a.creation else 0,
        normalization='A^dagger A=4-N_R (particle) or N_R (hole); no all-state unit norm',
        tensor_policy='matter restriction and appended boundary index only; both colour-link indices unchanged',
        loop_scope='independent ell identity, e0=ell+nlight_left-2, e1=ell; bounded three-ell operator probe plus exact algebra')


def neutral_ground(h, tolerance, maxiter, seed):
    rng = np.random.default_rng(seed)
    initial = rng.normal(size=h.dimension) + 1j * rng.normal(size=h.dimension)
    operator_factory: Any = LinearOperator
    operator = operator_factory((h.dimension, h.dimension), matvec=h.apply, rmatvec=h.apply, dtype=complex)
    values, vectors = eigsh(operator, k=2, which='SA', tol=tolerance / 10,
                           maxiter=maxiter, v0=initial, ncv=min(h.dimension, 24))
    vectors, _ = np.linalg.qr(vectors, mode='reduced')
    actions = np.column_stack([h.apply(vectors[:, k]) for k in range(2)])
    projected = vectors.conj().T @ actions
    values, rotation = np.linalg.eigh((projected + projected.conj().T) / 2)
    vectors, actions = vectors @ rotation, actions @ rotation
    residuals = np.linalg.norm(actions - vectors * values, axis=0)
    scaled = residuals / np.maximum(1., np.abs(values))
    orthogonality = norm(vectors.conj().T @ vectors - np.eye(2))
    if (not np.all(np.isfinite(scaled)) or float(scaled.max()) > tolerance
            or orthogonality > 256 * np.finfo(float).eps):
        raise RuntimeError('Actual full-H neutral eigenpair residual/orthogonality exceeds its declared limit')
    return vectors[:, 0], dict(energy=float(values[0]), lowest_returned_energies=values.tolist(),
        full_H_eigen_residuals=residuals.tolist(), scaled_full_H_residuals=scaled.tolist(),
        residual_limit=tolerance, residual_scale='max(1, abs(E))', orthogonality_residual=orthogonality,
        owner='actual sparse eigsh SA on every native centre-allowed physical state; QR/Hermitian Ritz refinement',
        limit='numerical ground estimate, not a certified exact neutral lower energy or infinite-cutoff gap')


def expectation(vector, action):
    return complex(np.vdot(vector, action))


def charge_trial(a, psi, energy, neutral_parts, charged_parts, tolerance):
    image = a.apply(psi)
    norm2 = float(np.vdot(image, image).real)
    actual_weight = float(np.vdot(psi, a.weight(psi)).real)
    require_small(abs(norm2 - actual_weight), 'actual trial normalization')
    if norm2 <= 0:
        raise ValueError('The actual neutral state has zero charge-map weight')
    normalized = image / np.sqrt(norm2)
    charged_energy = float(expectation(normalized, a.charged.apply(normalized)).real)
    neutral_light = neutral_parts.flavour(psi, 1)
    commutator = expectation(image, charged_parts.flavour(image, 1) - a.apply(neutral_light))
    onsite_right = expectation(psi, neutral_parts.onsite_action(psi, 1, (1,)))
    light_hopping = expectation(psi, neutral_parts.hopping(psi, 1))
    independent = -onsite_right - light_hopping / 2
    stationarity = expectation(psi, a.base.apply(a.weight(psi))) - expectation(a.base.apply(psi), a.weight(psi))
    direct_difference = charged_energy - energy
    commutator_prediction = a.base.ue1 / 2 + commutator.real / norm2
    independent_prediction = a.base.ue1 / 2 + independent.real / norm2
    numerical_limit = max(2e-8, 20 * tolerance * max(1., abs(energy)) / norm2)
    require_small(abs(direct_difference - commutator_prediction), 'full charged energy versus commutator', numerical_limit)
    require_small(abs(commutator - independent), 'stationary independent right-onsite/half-hopping identity', numerical_limit)
    preservation = {}
    for label, before, after in (
        ('Gauss_angular', neutral_parts.angular(psi), charged_parts.angular(image)),
        ('entire_heavy_H', neutral_parts.flavour(psi, 0), charged_parts.flavour(image, 0))):
        charged_value = expectation(image, after).real / norm2
        weighted_neutral = expectation(psi, a.weight(before)).real / norm2
        ordinary_neutral = expectation(psi, before).real
        require_small(abs(charged_value - weighted_neutral), label + ' weighted energy identity', 2e-8)
        preservation[label] = dict(charged_normalized=float(charged_value),
            neutral_N_weighted=float(weighted_neutral), neutral_unweighted=float(ordinary_neutral),
            weighted_preservation_residual=float(abs(charged_value - weighted_neutral)))
    probabilities = {}
    for n in range(5):
        probabilities[str(n)] = float(sum(np.sum(abs(x[a.right_number[a.base.blocks[name]['keep']] == n])**2)
                                      for name, x in a.base.unpack(psi).items()))
    nright = sum(int(n) * p for n, p in probabilities.items())
    return dict(charge=a.charged.q, charged_trial_energy=charged_energy,
        charged_trial_Rayleigh_difference=direct_difference,
        cost_interpretation='charged energy is variational UPPER; difference uses residual-enforced numerical neutral ground; never a positive LOWER charge gap',
        actual_norm_squared=norm2, actual_occupation_denominator=actual_weight,
        N_R_mean=float(nright), N_R_probabilities=probabilities, p0=probabilities['0'], p4=probabilities['4'],
        boundary_electric_added=a.base.ue1 / 2,
        commutator_numerator_real=float(commutator.real), commutator_numerator_imaginary=float(commutator.imag),
        onsite_R_mean=float(onsite_right.real), light_hopping_mean=float(light_hopping.real),
        independent_numerator=float(independent.real), independent_numerator_residual=float(abs(commutator - independent)),
        commutator_predicted_cost=float(commutator_prediction), independent_predicted_cost=float(independent_prediction),
        full_energy_formula_residual=float(abs(direct_difference - commutator_prediction)),
        occupancy_stationarity_residual=float(abs(stationarity)), weighted_cost_preservation=preservation,
        warning='intertwining preserves operators; normalized expectations compare to occupation-weighted neutral expectations, not generally the unweighted neutral means')


def quartic_moment_bound(kappa, heavy_coefficient_bound, light_coefficient_bound,
                         neutral_trial_coefficient, negative_photon_bound=0.):
    """Conditional full-field theorem constants, not a numerical field simulation.

    neutral_trial_coefficient must come from an actual neutral Gauss trial as in
    the note. The atomic fixed-orbit ground is not such a full-field certificate.
    """
    constants = (kappa, heavy_coefficient_bound, light_coefficient_bound,
                 neutral_trial_coefficient, negative_photon_bound)
    if not all(np.isfinite(x) for x in constants) or kappa <= 0 or min(constants[1:]) < 0:
        raise ValueError('Use finite kappa>0 and nonnegative proven coefficient/trial/photon bounds')
    c = heavy_coefficient_bound + light_coefficient_bound
    young = .75 * (4 / kappa)**(1 / 3) * c**(4 / 3)
    fourth = 16 / kappa * (neutral_trial_coefficient + young + negative_photon_bound)
    return dict(Lambda_domain='Lambda>=1', rho_fourth_moment_bound=float(fourth),
                Young_constant=float(young), first_moment_bound=float(fourth**.25),
                hypotheses='four unrestricted traceless Hermitian fields with Tr(B_a B_b)=4 delta_ab; full quartic Gram law, nonnegative kinetic/electric, actual neutral Gauss trial upper, bounded negative photon')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff', choices=tuple(CUTOFFS), default='minimal')
    parser.add_argument('--lambdas', nargs='+', type=float, default=[1.])
    parser.add_argument('--inertia', type=float, default=2.)
    parser.add_argument('--heavy-sign', type=int, choices=(-1, 1), default=-1)
    parser.add_argument('--ue4', type=float, default=.3)
    parser.add_argument('--ue1', type=float, default=.4)
    parser.add_argument('--stiffness', type=float, default=2.)
    parser.add_argument('--m', type=float, default=.2)
    parser.add_argument('--r', type=float, default=.12)
    parser.add_argument('--v', type=float, default=.3)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--maxiter', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.lambdas + [args.inertia, args.ue4, args.ue1, args.stiffness, args.tol]) <= 0 or args.maxiter < 1:
        parser.error('Require positive scales/inertia/electric/stiffness/tolerance and maxiter>=1')
    hashes = source_hashes()
    if hashes.get('quantum_su4_gauss.py') != ATOMIC_GAUSS_SOURCE:
        raise RuntimeError('Atomic Gauss owner changed; re-review the pinned charge-intertwiner interface')
    band = RotationBand(a=1., m=args.m, r=args.r, v=args.v, improved=False)
    data = uniform_data(band, 3, args.stiffness, 0., matrix_unit_basis())
    frame = data['frame']
    require_small(abs(np.trace(frame[0])), 'actual light onsite tracelessness')
    code, link = PauliCode(), HaarLink(args.cutoff)
    matter = {number: PrimitiveMatter(code, number, link.T) for number in (3, 4, 5)}
    primitive = primitive_one_cell(code, link.T)
    neutral_matter, hole_matter, particle_matter = matter[4], matter[3], matter[5]
    minus = [rectangular_ladder(code, neutral_matter.masks, hole_matter.masks, 12 + i, False) for i in range(4)]
    plus = [rectangular_ladder(code, neutral_matter.masks, particle_matter.masks, 12 + i, True) for i in range(4)]
    identity = sparse.eye(neutral_matter.dimension, format='csr')
    car = max(norm(minus[i].conj().T @ minus[j] + plus[j].conj().T @ plus[i] - (identity if i == j else 0))
              for i in range(4) for j in range(4))
    require_small(car, 'actual whole neutral-domain right-light CAR')
    result: dict[str, Any] = dict(model='actual atomic SU4 Gauss boundary-charge intertwiner',
        parameters={key: value for key, value in vars(args).items() if key != 'output'},
        pinned_atomic_gauss_sha256=ATOMIC_GAUSS_SOURCE, primitive_one_cell=primitive,
        whole_neutral_domain_CAR_residual=car, runs=[],
        raw_marker_policy='mode16 is unchanged neutral parity marker; no SU4 or EM charge',
        atomic_scope='fixed simultaneous Phi orbit EFT, J=0, actual minimal33 by default; no quantized 60-field or new photon-module consumer',
        full_field_theorem='separate unrestricted quartic law and constructive neutral Gauss trial proved in gauss_charge_intertwiner.md; quartic_moment_bound accepts proven constants only',
        excluded_claims=['positive lower charge gap', 'charged collapse inferred from bare matter Casimir',
                         'full 60-field phase', 'continuum QED', 'integrated new photon-module execution'])
    for heavy_scale in args.lambdas:
        hs = {number: PhysicalHamiltonian(matter[number], link, frame, band, heavy_scale,
              args.inertia, args.ue4, args.ue1, heavy_sign=args.heavy_sign) for number in (3, 4, 5)}
        parts = {number: HamiltonianParts(h) for number, h in hs.items()}
        base = hs[4]
        maps = {number: BoundaryChargeIntertwiner(code, base, hs[number]) for number in (3, 5)}
        evidence = [operator_evidence(a, parts[4], parts[number]) for number, a in maps.items()]
        psi, ground = neutral_ground(base, args.tol, args.maxiter, args.seed)
        trials = [charge_trial(a, psi, ground['energy'], parts[4], parts[number], args.tol)
                  for number, a in maps.items()]
        result['runs'].append(dict(Lambda=heavy_scale, inertia=args.inertia,
            heavy_whole_H_multiplier=base.heavy_multiplier, neutral_NH=4, neutral_NL=4,
            matter_dimensions={str(n): h.matter.dimension for n, h in hs.items()},
            physical_dimensions={str(n): h.dimension for n, h in hs.items()},
            intertwiners=evidence, neutral_ground=ground, charged_trials=trials))
    result['source_sha256'] = source_hashes()
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()

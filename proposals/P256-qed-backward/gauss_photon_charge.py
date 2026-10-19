#!/usr/bin/env python3
"""Primitive boundary charge carrier in the complete aligned SU4/U1 photon EFT."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from exact_haar_alignment import ExactHaarAlignment
from gauss_charge_intertwiner import (ATOMIC_GAUSS_SOURCE, BoundaryChargeIntertwiner,
                                     HamiltonianParts, deterministic_vector)
from quantum_su4_gauss import (CUTOFFS, HaarLink, PauliCode, PhysicalHamiltonian,
                               PrimitiveMatter, RotationBand, matrix_unit_basis,
                               norm, require_small, source_hashes, uniform_data)
from quantum_su4_photon import CompactGaussPhoton, solve_ground

PHOTON_SOURCE = 'ffadad919f74fe510484530037d4ae11b7860046fa384aa721839d2771d5c0bf'


class PhotonChargeIntertwiner:
    """A on (matter, colour-left, colour-right, boundary, independent ell).

    The atomic supplier owns the primitive ladders and their centre restrictions.
    This lift changes only matter and the boundary index; ell is NOT boundary.
    """

    def __init__(self, code, neutral, charged):
        if (neutral.R, neutral.magnetic, neutral.alignment_coefficient) != (
                charged.R, charged.magnetic, charged.alignment_coefficient):
            raise ValueError('Charge sectors require the same loop and alignment coefficients')
        if neutral.alignment_blocks is not charged.alignment_blocks:
            raise ValueError('Charge sectors require the same exact Haar alignment provider')
        if not np.array_equal(neutral.base.frame, charged.base.frame):
            raise ValueError('Charge sectors require the same orbital frame')
        self.neutral, self.charged = neutral, charged
        self.primitive = BoundaryChargeIntertwiner(code, neutral.base, charged.base)
        self.components = self.primitive.components
        self.adjoints = {name: [c.getH().tocsr() for c in components]
                         for name, components in self.components.items()}
        self.right_number = self.primitive.right_number
        self.norm_counts = self.primitive.norm_counts
        for name, block in neutral.blocks.items():
            target = charged.blocks[name]
            if (len(block['shape']) != 5 or len(target['shape']) != 5
                    or block['shape'][3] != 1 or target['shape'][3] != 4
                    or block['shape'][1:3] != target['shape'][1:3]
                    or block['shape'][4] != target['shape'][4]):
                raise AssertionError('Independent colour, boundary and ell axes changed')

    def apply(self, vector):
        result = {}
        for name, x in self.neutral.unpack(vector).items():
            shape = self.charged.blocks[name]['shape']
            y = np.zeros(shape, dtype=complex)
            columns = x[:, :, :, 0, :].reshape(x.shape[0], -1)
            for i, c in enumerate(self.components[name]):
                y[:, :, :, i, :] = (c @ columns).reshape(shape[0], shape[1], shape[2], shape[4])
            result[name] = y
        return self.charged.pack(result)

    def adjoint(self, vector):
        result = {}
        for name, y in self.charged.unpack(vector).items():
            shape = self.neutral.blocks[name]['shape']
            x = np.zeros(shape, dtype=complex)
            for i, c in enumerate(self.adjoints[name]):
                columns = y[:, :, :, i, :].reshape(y.shape[0], -1)
                x[:, :, :, 0, :] += (c @ columns).reshape(shape[0], shape[1], shape[2], shape[4])
            result[name] = x
        return self.neutral.pack(result)

    def weight(self, vector):
        return self.neutral.pack({name: self.norm_counts[self.neutral.blocks[name]['keep']]
            [:, None, None, None, None] * x for name, x in self.neutral.unpack(vector).items()})

    def normalized(self, vector):
        image = self.apply(vector)
        weight = float(np.vdot(image, image).real)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('The actual charge source has zero or invalid occupation weight')
        return image / np.sqrt(weight), weight


class PhotonHamiltonianParts:
    """Lift the existing onsite action; use actual two-path flavour hopping."""

    def __init__(self, model):
        self.model = model
        self.atomic = HamiltonianParts(model.base)

    def onsite_action(self, vector, flavour, cells=(0, 1)):
        model, base = self.model, self.model.base
        blocks = model.unpack(vector)
        slots = []
        for ell_index in range(len(model.loop_values)):
            source = base.pack({name: x[..., ell_index] for name, x in blocks.items()})
            slots.append(base.unpack(self.atomic.onsite_action(source, flavour, cells)))
        return model.pack({name: np.stack([slot[name] for slot in slots], axis=-1)
                           for name in model.blocks})

    def flavour(self, vector, flavour):
        return self.onsite_action(vector, flavour) + self.model.hop_action(vector, flavour=flavour)

    def angular(self, vector):
        return self.model.pack({name: self.model.orientation(name, x)
                                for name, x in self.model.unpack(vector).items()})

    def colour_electric(self, vector):
        return self.model.pack({name: self.model.base.ue4 * self.model.link.casimir[name] * x
                                for name, x in self.model.unpack(vector).items()})

    def em_electric(self, vector):
        return self.model.pack({name: (self.model.diagonal[name] - self.model.base.ue4
            * self.model.link.casimir[name])[:, None, None, None, :] * x
            for name, x in self.model.unpack(vector).items()})

    def alignment(self, vector):
        source = self.model.unpack(vector)
        target = {name: np.zeros_like(x, dtype=complex) for name, x in source.items()}
        self.model._add_alignment(source, target)
        return self.model.pack(target)

    def magnetic(self, vector):
        return -self.model.magnetic * self.model.loop_action(vector, 'cos')

    def reconstruction(self, vector):
        return norm(self.model.apply(vector) - self.angular(vector) - self.colour_electric(vector)
                    - self.em_electric(vector) - self.alignment(vector) - self.magnetic(vector)
                    - self.flavour(vector, 0) - self.flavour(vector, 1))


def gauss_generator(model, name, x, cell, generator, boundary_scale=1.):
    """Actual covariant Gauss derivative, with an independently adverse boundary."""
    result = model.base.matter_action(model.blocks[name]['Q'][cell][generator], x)
    t = model.link.reps[name][generator]
    if cell == 0:
        result -= np.einsum('ca,mabzl->mcbzl', t.T, x, optimize=True)
    else:
        result += np.einsum('cb,mabzl->maczl', t, x, optimize=True)
        result += boundary_scale * np.einsum('zw,mabwl->mabzl',
                                             model.base.boundary[generator], x, optimize=True)
    return result


def gauss_square(model, vector, boundary_scale=1.):
    result = {}
    for name, x in model.unpack(vector).items():
        y = np.zeros_like(x, dtype=complex)
        for cell in range(2):
            for generator in range(15):
                gx = gauss_generator(model, name, x, cell, generator, boundary_scale)
                y += gauss_generator(model, name, gx, cell, generator, boundary_scale)
        result[name] = y / (2 * model.base.inertia)
    return model.pack(result)


def edge_field(model, vector, edge):
    if edge == 0:
        return model.loop_action(vector, 'ell') + model.density_action(vector) - 2 * vector
    if edge == 1:
        return model.loop_action(vector, 'ell')
    raise ValueError('Internal edge is 0 or 1')


def weight_adverse_case(a):
    """Exact occupation witness: unweighted N_L=2, weighted N_L=1.5 or 2.5."""
    model = a.neutral
    blocks = {name: np.zeros(b['shape'], dtype=complex) for name, b in model.blocks.items()}
    for occupation in (1, 3):
        for name, block in model.blocks.items():
            indices = np.flatnonzero(a.right_number[block['keep']] == occupation)
            if len(indices):
                blocks[name][indices[0], 0, 0, 0, model.R] = 1 / np.sqrt(2)
                break
        else:
            raise AssertionError('Native complete source lost N_R=1 or 3')
    witness = model.pack(blocks)
    image, weight = a.normalized(witness)
    ordinary = float(np.vdot(witness, model.density_action(witness)).real)
    weighted = float(np.vdot(witness, a.weight(model.density_action(witness))).real / weight)
    charged = float(np.vdot(image, a.charged.density_action(image)).real)
    expected = 2.5 if a.primitive.creation else 1.5
    require_small(abs(ordinary - 2.), 'unweighted occupation witness')
    require_small(abs(weighted - expected), 'weighted occupation witness')
    require_small(abs(charged - weighted), 'actual normalized image weighted occupation')
    require_small(abs(abs(charged - ordinary) - .5), 'unweighted adverse defect is exactly one half')
    return dict(ordinary_neutral_N_L=ordinary, weighted_neutral_N_L=weighted,
                charged_image_N_L=charged, wrong_unweighted_defect=abs(charged - ordinary),
                witness='Equal probabilities on native N_R=1,3 at ell=0; exact nonconstant A-dagger-A')


def operator_evidence(a, neutral_parts, charged_parts, seed):
    neutral, charged = a.neutral, a.charged
    x = deterministic_vector(neutral.dimension)
    y = deterministic_vector(charged.dimension, 31)
    image = a.apply(x)
    residuals = dict(adjoint_pairing=abs(np.vdot(y, image) - np.vdot(a.adjoint(y), x)),
                     full_norm_operator=norm(a.adjoint(image) - a.weight(x)),
                     primitive_norm_operator=0., primitive_all30_generator_operators=0.,
                     all30_Gauss_generators=0.)
    source_blocks, image_blocks = neutral.unpack(x), charged.unpack(image)
    for name, matrix in a.primitive.matrix.items():
        expected = sparse.diags(a.norm_counts[neutral.blocks[name]['keep']], dtype=float, format='csr')
        residuals['primitive_norm_operator'] = max(residuals['primitive_norm_operator'],
                                                  norm(matrix.getH() @ matrix - expected))
        for cell in range(2):
            for generator in range(15):
                qs = neutral.blocks[name]['Q'][cell][generator]
                qt = sparse.kron(charged.blocks[name]['Q'][cell][generator],
                                 sparse.eye(4, format='csr'), format='csr')
                if cell == 1:
                    qt += sparse.kron(sparse.eye(matrix.shape[0] // 4, format='csr'),
                        sparse.csr_matrix(charged.base.boundary[generator]), format='csr')
                residuals['primitive_all30_generator_operators'] = max(
                    residuals['primitive_all30_generator_operators'], norm(qt @ matrix - matrix @ qs))
    for cell in range(2):
        for generator in range(15):
            before = neutral.pack({name: gauss_generator(neutral, name, v, cell, generator)
                                   for name, v in source_blocks.items()})
            after = charged.pack({name: gauss_generator(charged, name, v, cell, generator)
                                  for name, v in image_blocks.items()})
            residuals['all30_Gauss_generators'] = max(residuals['all30_Gauss_generators'],
                                                     norm(after - a.apply(before)))
    for label, before, after in preserved_parts(neutral_parts, charged_parts):
        residuals[label] = norm(after(image) - a.apply(before(x)))
    residuals['EM_boundary_constant_only'] = norm(charged_parts.em_electric(image)
        - a.apply(neutral_parts.em_electric(x)) - neutral.base.ue1 * charged.q**2 / 2 * image)
    for edge in (0, 1):
        residuals['internal_edge_' + str(edge)] = norm(edge_field(charged, image, edge)
                                                      - a.apply(edge_field(neutral, x, edge)))
    residuals['neutral_parts_reconstruction'] = neutral_parts.reconstruction(x)
    residuals['charged_parts_reconstruction'] = charged_parts.reconstruction(image)
    residuals['neutral_Gauss_square'] = norm(gauss_square(neutral, x) - neutral_parts.angular(x))
    residuals['charged_Gauss_square'] = norm(gauss_square(charged, image) - charged_parts.angular(image))
    adverse = {label: norm(gauss_square(charged, image, scale) - a.apply(neutral_parts.angular(x)))
               for label, scale in (('wrong_boundary_sign', -1.), ('omitted_boundary', 0.))}
    # A path-1 transfer shifts ell oppositely to N_L, and therefore preserves E0.
    # Replacing its shift by the identity demonstrably violates that edge relation.
    path1 = lambda v: neutral.hop_action(v, left=(0., 1.), flavour=1)
    wrong_path1 = lambda v: neutral.hop_action(v, left=(0., 1.), flavour=1, formal_identity=True)
    field_x = edge_field(neutral, x, 0)
    residuals['path1_current_preserves_other_edge_E0'] = norm(
        edge_field(neutral, path1(x), 0) - path1(field_x))
    path0 = lambda v: neutral.hop_action(v, left=(1., 0.), flavour=1)
    residuals['path0_current_preserves_other_edge_E1'] = norm(
        edge_field(neutral, path0(x), 1) - path0(edge_field(neutral, x, 1)))
    adverse['no_ell_shift_wrong_edge_commutator'] = norm(
        edge_field(neutral, wrong_path1(x), 0) - wrong_path1(field_x))
    adverse['no_ell_shift_changes_actual_light_H'] = norm(neutral.hop_action(x, flavour=1)
        - neutral.hop_action(x, flavour=1, formal_identity=True))
    for label, value in residuals.items():
        require_small(float(value), label, 3e-8)
    for label, value in adverse.items():
        if not np.isfinite(value) or value <= 100 * 3e-8:
            raise AssertionError('Actual wrong operator failed to discriminate: ' + label)
    return dict(charge=charged.q, residuals={k: float(v) for k, v in residuals.items()},
                adverse_operators=adverse, occupation_weight_adverse=weight_adverse_case(a),
                charged_current_contact_Ward=charged.operator_evidence(seed),
                policy='All five native axes; actual primitive ladders; no colour/boundary/ell merging')


def preserved_parts(neutral, charged):
    return (('Gauss_angular', neutral.angular, charged.angular),
            ('full_heavy', lambda v: neutral.flavour(v, 0), lambda v: charged.flavour(v, 0)),
            ('colour_electric', neutral.colour_electric, charged.colour_electric),
            ('exact_Haar_alignment', neutral.alignment, charged.alignment),
            ('dynamical_magnetic', neutral.magnetic, charged.magnetic))


def charge_trial(a, psi, energy, neutral_parts, charged_parts, tolerance) -> tuple[np.ndarray, dict[str, Any]]:
    neutral, charged = a.neutral, a.charged
    image = a.apply(psi)
    normalized, weight = a.normalized(psi)
    require_small(abs(weight - np.vdot(psi, a.weight(psi)).real), 'actual A norm equals occupation weight')
    hn, hc = neutral.apply(psi), charged.apply(normalized)
    charged_energy = float(np.vdot(normalized, hc).real)
    ordinary_energy = float(np.vdot(psi, hn).real)
    weighted_energy = np.vdot(psi, a.weight(hn)) / weight
    neutral_light = neutral_parts.flavour(psi, 1)
    commutator = np.vdot(image, charged_parts.flavour(image, 1) - a.apply(neutral_light))
    right_onsite = np.vdot(psi, neutral_parts.onsite_action(psi, 1, (1,)))
    full_light_hopping = np.vdot(psi, neutral.hop_action(psi, flavour=1))
    independent = -right_onsite - full_light_hopping / 2
    exact_weighted_prediction = weighted_energy + neutral.base.ue1 / 2 + commutator / weight
    independent_cost = neutral.base.ue1 / 2 + independent.real / weight
    stationarity = np.vdot(psi, neutral.apply(a.weight(psi))) - np.vdot(hn, a.weight(psi))
    limit = max(3e-8, 32 * tolerance * max(1., abs(energy)) / weight)
    require_small(abs(charged_energy - exact_weighted_prediction), 'full charged weighted commutator identity', limit)
    require_small(abs(weighted_energy - energy), 'neutral eigenstate weighted energy', limit)
    require_small(abs(commutator - independent), 'stationary right onsite plus half full two-path hopping', limit)
    require_small(abs(stationarity), 'actual right-occupation stationarity', limit * weight)
    require_small(abs(charged_energy - energy - independent_cost), 'full Rayleigh cost independent formula', limit)
    preservation = {}
    for label, before, after in preserved_parts(neutral_parts, charged_parts):
        action = before(psi)
        weighted = np.vdot(psi, a.weight(action)) / weight
        ordinary = np.vdot(psi, action)
        image_value = np.vdot(image, after(image)) / weight
        require_small(abs(image_value - weighted), label + ' occupation-weighted preservation', 3e-8)
        preservation[label] = dict(charged_normalized=float(image_value.real),
            neutral_occupation_weighted=float(weighted.real), neutral_ordinary=float(ordinary.real),
            weighted_imaginary=float(weighted.imag), residual=float(abs(image_value - weighted)))
    probabilities = {str(n): float(sum(np.sum(abs(x[a.right_number[neutral.blocks[name]['keep']] == n])**2)
        for name, x in neutral.unpack(psi).items())) for n in range(5)}
    return normalized, dict(charge=charged.q, charged_trial_energy=charged_energy,
        charged_trial_Rayleigh_difference=charged_energy - energy,
        actual_norm_squared=weight, occupation_weight='4-N_R' if a.primitive.creation else 'N_R',
        neutral_ordinary_total_energy=ordinary_energy, neutral_occupation_weighted_total_energy=float(weighted_energy.real),
        weighted_total_imaginary=float(weighted_energy.imag), neutral_reference_energy=energy,
        N_R_probabilities=probabilities, N_R_mean=sum(int(n) * p for n, p in probabilities.items()),
        onsite_right_light=float(right_onsite.real), full_two_path_light_hopping=float(full_light_hopping.real),
        commutator_numerator_real=float(commutator.real), commutator_numerator_imaginary=float(commutator.imag),
        independent_numerator=float(independent.real), independent_numerator_residual=float(abs(commutator - independent)),
        independent_predicted_cost=float(independent_cost), weighted_full_formula_residual=float(abs(charged_energy - exact_weighted_prediction)),
        full_Rayleigh_formula_residual=float(abs(charged_energy - energy - independent_cost)),
        stationarity_residual=float(abs(stationarity)), weighted_part_preservation=preservation,
        interpretation='Actual normalized A-image variational upper energy; never a lower charged-gap bound')


def source_active_pole(model, source, weight, neutral_energy, tolerance, maxiter, source_tolerance) -> dict[str, Any]:
    """A k=1 source-initialized search on the COMPLETE charged Hamiltonian."""
    values, vectors = eigsh(model.linear_operator(), k=1, which='SA', v0=source,
        tol=tolerance / 20, maxiter=maxiter, ncv=min(model.dimension, 32))
    vector = vectors[:, 0] / norm(vectors[:, 0])
    action = model.apply(vector)
    eigen_energy = float(np.vdot(vector, action).real)
    residual = norm(action - eigen_energy * vector)
    if not np.isfinite(residual) or residual > tolerance * max(1., abs(eigen_energy)):
        raise RuntimeError('Charged candidate is not a residual-confirmed full-H eigenstate')
    transfer = eigen_energy - neutral_energy
    transfer_uncertainty = residual + tolerance * max(1., abs(neutral_energy))
    if transfer <= transfer_uncertainty:
        raise RuntimeError('Charged source-active candidate lacks resolved positive energy transfer')
    overlap = np.vdot(vector, source)
    normalized_residue = float(abs(overlap)**2)
    if normalized_residue <= source_tolerance:
        raise RuntimeError('Returned full-H eigenstate is source-dark at the declared source activity threshold')
    if normalized_residue > 1 + 256 * np.finfo(float).eps:
        raise RuntimeError('Creator residue exceeds the normalized total spectral weight')
    source_action = model.apply(source) - neutral_energy * source
    moment0 = weight * float(np.vdot(source, source).real)
    moment1 = weight * float(np.vdot(source, source_action).real)
    moment2 = weight * float(np.vdot(source_action, source_action).real)
    residue = weight * normalized_residue
    unseen = source - vector * overlap
    unseen_weight = weight * float(np.vdot(unseen, unseen).real)
    unseen_first = weight * float(np.vdot(unseen, model.apply(unseen) - neutral_energy * unseen).real)
    zero_partition = abs(moment0 - residue - unseen_weight)
    first_partition = abs(moment1 - residue * transfer - unseen_first)
    require_small(zero_partition, 'creator zeroth moment partition', 3e-8)
    first_limit = max(3e-8, 4 * weight * abs(overlap) * residual)
    require_small(first_partition, 'creator first moment full-H partition', first_limit)
    return dict(charge=model.q, full_H_eigen_energy=eigen_energy, energy_transfer=transfer,
        positive_transfer_uncertainty=transfer_uncertainty, full_H_eigen_residual=residual,
        scaled_full_H_eigen_residual=residual / max(1., abs(eigen_energy)), residual_limit=tolerance,
        lowest_returned_eigsh_value=float(values[0]), creator_residue=residue,
        normalized_creator_residue=normalized_residue, source_activity_threshold=source_tolerance,
        total_zeroth_spectral_moment=moment0, total_first_transfer_moment=moment1,
        total_second_transfer_moment=moment2, normalized_first_transfer_moment=moment1 / moment0,
        unseen_zeroth_spectral_weight=unseen_weight, unseen_normalized_weight=unseen_weight / moment0,
        unseen_signed_first_transfer_moment=moment1 - residue * transfer,
        directly_projected_unseen_first_transfer_moment=unseen_first,
        zeroth_moment_partition_residual=zero_partition, first_moment_partition_residual=first_partition,
        classification='Residual-confirmed full-H eigenstate pole, not a fixed-image or finite-Krylov projection pole',
        search='eigsh k=1 SA, normalized primitive A image v0, full charged LinearOperator',
        limitations=['No completeness or full charged-ground certificate from this source-initialized search',
                     'One resolved eigenvector; degenerate eigenspace residue need not be exhausted',
                     'Unseen first moment is signed without a separately established full charged spectral gap'])


def compare_fullground(path, parameters, hashes, run):
    if path is None:
        return dict(available=False, reason='No parent same-parameter fullground receipt provided')
    raw = path.read_bytes()
    receipt = json.loads(raw)
    carrier_only = {Path(__file__).name, Path(__file__).with_suffix('.md').name,
                    'gauss_charge_intertwiner.py'}
    for filename, digest in hashes.items():
        if filename not in carrier_only and receipt.get('source_sha256', {}).get(filename) != digest:
            raise ValueError('Parent fullground receipt has a different or absent shared source: ' + filename)
    owner_parameters = receipt['parameters']
    for key in ('m', 'r', 'v', 'stiffness', 'ue4', 'ue1', 'magnetic', 'heavy_sign'):
        if owner_parameters.get(key) != parameters[key]:
            raise ValueError('Parent fullground receipt has a different parameter: ' + key)
    for key in ('m', 'r', 'v'):
        if owner_parameters.get('heavy_' + key) not in (None, parameters[key]):
            raise ValueError('Parent fullground receipt uses a different heavy band: ' + key)
    matches = {}
    for candidate in receipt['runs']:
        if all(candidate.get(key) == run[key] for key in ('cutoff', 'R', 'Lambda', 'inertia', 'Jpref')):
            if 'spectrum' in candidate:
                matches[candidate['NL']] = candidate['spectrum']
    if set(matches) != {3, 4, 5}:
        raise ValueError('Parent receipt does not contain all three matching complete-H sectors')
    parent_neutral = float(matches[4]['energies'][0])
    parent_tolerance = float(matches[4]['residual_limit'])
    neutral_limit = 32 * max(parameters['tol'], parent_tolerance) * max(1., abs(parent_neutral))
    require_small(abs(run['neutral_energy'] - parent_neutral), 'same-source parent neutral energy', neutral_limit)
    comparisons = []
    for pole in run['charged_poles']:
        number = pole['charge'] + 4
        spectrum = matches[number]
        charged_ground_estimate = float(spectrum['energies'][0])
        comparisons.append(dict(charge=pole['charge'], parent_charged_ground_Ritz_energy=charged_ground_estimate,
            parent_full_H_residuals=spectrum['full_H_residuals'],
            parent_ground_rank=spectrum['ground_rank'], parent_measured_complement_gap=spectrum['measured_complement_gap'],
            pole_minus_parent_ground=pole['full_H_eigen_energy'] - charged_ground_estimate,
            parent_ground_transfer=charged_ground_estimate - parent_neutral,
            interpretation='Comparison to separately executed complete-domain parent ground/multiplet receipt, not a new gap certificate'))
    return dict(available=True, receipt=str(path), receipt_sha256=hashlib.sha256(raw).hexdigest(),
                parent_neutral_energy=parent_neutral, comparisons=comparisons)


def source_identity():
    hashes = source_hashes()
    if hashes.get('quantum_su4_gauss.py') != ATOMIC_GAUSS_SOURCE:
        raise RuntimeError('Atomic Gauss supplier changed; re-review the primitive charge interface')
    if hashes.get('quantum_su4_photon.py') != PHOTON_SOURCE:
        raise RuntimeError('Aligned photon supplier changed; re-review the full five-axis consumer')
    note = Path(__file__).with_suffix('.md')
    hashes[note.name] = hashlib.sha256(note.read_bytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff', choices=tuple(CUTOFFS), default='minimal')
    parser.add_argument('--R', type=int, nargs='+', default=[1])
    parser.add_argument('--lambdas', type=float, nargs='+', default=[1.])
    parser.add_argument('--inertia', type=float, default=2.)
    parser.add_argument('--heavy-sign', type=int, choices=(-1, 1), default=-1)
    parser.add_argument('--Jpref', type=float, default=.2)
    for name, default in (('m', .2), ('r', .12), ('v', .3), ('stiffness', 2.),
                          ('ue4', .3), ('ue1', .4), ('magnetic', .7)):
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--levels', type=int, default=8)
    parser.add_argument('--max-ground-rank', type=int, default=32)
    parser.add_argument('--tol', type=float, default=1e-9)
    parser.add_argument('--source-tol', type=float, default=1e-10)
    parser.add_argument('--maxiter', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--fullground-receipt', type=Path)
    parser.add_argument('--operators-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    positive = args.lambdas + [args.inertia, args.Jpref, args.stiffness, args.ue4, args.ue1, args.tol, args.source_tol]
    if any(not np.isfinite(x) or x <= 0 for x in positive):
        parser.error('Finite positive Lambda,I,Jpref,stiffness,UE4,UE1 and tolerances required')
    if (any(R < 1 for R in args.R) or args.levels < 4 or args.maxiter < 1
            or args.max_ground_rank < 1 or args.source_tol >= 1 or args.magnetic < 0
            or any(not np.isfinite(x) for x in (args.m, args.r, args.v, args.magnetic))):
        parser.error('R>=1, levels>=4, ranks/iterations>=1, source-tol<1 and finite bare coefficients/K>=0 required')
    hashes = source_identity()  # Before orbital selection, Haar products or eigensolves.
    parameters = {key: value for key, value in vars(args).items() if key not in ('output', 'fullground_receipt')}
    print(json.dumps(dict(stage='source-bound preflight', source_sha256=hashes, parameters=parameters),
                     allow_nan=False), file=sys.stderr, flush=True)
    band = RotationBand(a=1., m=args.m, r=args.r, v=args.v, improved=False)
    data = uniform_data(band, 3, args.stiffness, 0., matrix_unit_basis())
    require_small(abs(np.trace(data['frame'][0])), 'actual traceless right light onsite')
    code, link = PauliCode(), HaarLink(args.cutoff)
    alignment = ExactHaarAlignment(link, data['frame'])
    matters = {number: PrimitiveMatter(code, number, link.T) for number in (3, 4, 5)}
    result: dict[str, Any] = dict(model='Primitive charged carrier in complete aligned continuous SU4 plus compact U1 EFT',
        parameters=parameters, source_sha256=hashes, link=link.evidence(), alignment=alignment.evidence(),
        selected_frame_real=data['frame'].real.tolist(), selected_frame_imaginary=data['frame'].imag.tolist(),
        selected_gram=np.diag(data['selected']['G']).tolist(), runs=[],
        source_chart=dict(orbital='uniform_data same RotationBand, n3 quadrature and stiffness, fixed simultaneous orbit',
            statistics='PauliCode primitive rectangular right ladders from BoundaryChargeIntertwiner.components',
            Gauss='PhysicalHamiltonian complete NH4 and NL3/4/5 centre fibres, same continuous HaarLink',
            photon='CompactGaussPhoton complete independent ell=-R..R, two EM paths, dynamical compressed magnetic shift',
            alignment='ExactHaarAlignment continuous P W P, same provider and Lambda*Jpref/8 in all sectors',
            energy='existing residual-enforced solve_ground for neutral, complete charged-H source-initialized eigsh for poles'),
        excluded_claims=['Full charged ground from source-initialized k1', 'positive lower charge-gap theorem',
            'Dirac doublet', 'lower emergent mass', 'infinite cutoff', 'nonlinear60 quantum phase',
            '3D photon', 'Lorentz invariance', 'mechanical spin', 'S matrix', 'canonical QED'])
    for heavy in args.lambdas:
        bases = {number: PhysicalHamiltonian(matter, link, data['frame'], band, heavy,
            args.inertia, args.ue4, args.ue1, heavy_sign=args.heavy_sign)
            for number, matter in matters.items()}
        for R in args.R:
            models = {number: CompactGaussPhoton(base, R, args.magnetic, alignment, heavy * args.Jpref / 8)
                      for number, base in bases.items()}
            parts = {number: PhotonHamiltonianParts(model) for number, model in models.items()}
            maps = {number: PhotonChargeIntertwiner(code, models[4], models[number]) for number in (3, 5)}
            run: dict[str, Any] = dict(cutoff=args.cutoff, R=R, Lambda=heavy, inertia=args.inertia,
                Jpref=args.Jpref, alignment_coefficient=heavy * args.Jpref / 8,
                physical_dimensions={str(n): model.dimension for n, model in models.items()},
                tensor_shapes={str(n): {name: list(b['shape']) for name, b in model.blocks.items()}
                               for n, model in models.items()},
                neutral_current_contact_Ward=models[4].operator_evidence(args.seed),
                intertwiners=[operator_evidence(a, parts[4], parts[number], args.seed + number)
                              for number, a in maps.items()])
            if not args.operators_only:
                values, _, ground, _, receipt = solve_ground(models[4], args.levels, args.tol,
                    args.maxiter, args.seed, args.max_ground_rank)
                energy, psi = float(values[0]), ground[:, 0]
                run.update(neutral_energy=energy, neutral_ground=receipt,
                    neutral_source_policy='First member of completed measured neutral ground frame; no averaging of degenerate creator sources',
                    charged_trials=[], charged_poles=[])
                for number, a in maps.items():
                    source, trial = charge_trial(a, psi, energy, parts[4], parts[number], args.tol)
                    pole = source_active_pole(models[number], source, trial['actual_norm_squared'], energy,
                                             args.tol, args.maxiter, args.source_tol)
                    require_small(abs(pole['normalized_first_transfer_moment'] - trial['charged_trial_Rayleigh_difference']),
                                  'spectral first moment equals actual full Rayleigh transfer', 3e-8)
                    run['charged_trials'].append(trial)
                    run['charged_poles'].append(pole)
                run['parent_fullground_comparison'] = compare_fullground(args.fullground_receipt, parameters, hashes, run)
            result['runs'].append(run)
    if source_identity() != hashes:
        raise RuntimeError('A source or proof note changed during this execution; discard this unbound receipt')
    result['source_unchanged_at_completion'] = True
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()

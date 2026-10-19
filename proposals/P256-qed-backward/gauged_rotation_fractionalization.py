#!/usr/bin/env python3
"""Conditional rotation fractionalization, with actual Pauli/Z4 Gauss controls.

The finite link is a genuine four-state electric Hilbert space. It is not an
SU4 rotor, a quantum Higgs phase simulation, or a mechanical interferometer.
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
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.sparse import csr_matrix, diags

from pauli_compton import EdgeWords, boundary_ladder
from rotation_band import CHARGE_CONJUGATION, HOPPING, MIXING, RotationBand
from sea_selection import coefficients
from soft_budget import select_gram
from spin_orbital_locking import (binary_octahedral, exact_matrix,
                                 exact_stabilizer, exact_units,
                                 solve_selected_generators)


def norm(value) -> float:
    return float(np.linalg.norm(value))


def sparse_norm(value) -> float:
    return norm(value.data)


def complex_json(value) -> dict[str, Any]:
    value = np.asarray(value)
    return dict(real=value.real.tolist(), imaginary=value.imag.tolist())


def selected_frame(args):
    """Use the existing occupied-sea selection and solve all 18 unknowns."""
    exact_frame = [exact_matrix(MIXING), *(exact_matrix(a) for a in HOPPING)]
    exact_basis = exact_units()
    exact_solve, exact_generators = exact_stabilizer(exact_frame, exact_basis, derive=True)
    if exact_generators is None:
        raise AssertionError('No connected spatial stabilizer')
    commutant_basis = [sp.eye(4), *exact_basis]
    commutant_map = sp.Matrix.hstack(*(sp.Matrix.vstack(*[
        (t * a - a * t).reshape(16, 1) for a in exact_frame]) for t in commutant_basis))
    commutant_kernel = commutant_map.nullspace()
    exact_solve['common_commutant'] = dict(
        complex_matrix_unknowns=16, exact_dimension=len(commutant_kernel),
        scalar_only=commutant_kernel == [sp.Matrix([1] + [0] * 15)],
        su4_determinant_condition='lambda^4=1',
        residual_centre=['1', 'i', '-1', '-i'])
    group, quaternions, rotations = binary_octahedral(exact_generators)
    basis = np.array([np.array(t).astype(complex) for t in exact_basis])
    bare = RotationBand(args.a, m=args.m, r=args.r, v=args.v, e=1., improved=True)
    momenta = np.array(list(itertools.product(range(3), repeat=3))) * (2 * np.pi / (3 * args.a))
    gram = select_gram(coefficients(bare, momenta), args.stiffness, args.preferred)
    scales = np.sqrt(np.diag(gram['G']))
    change = expm(-1j * (.31 * basis[4] - .23 * basis[9] + .17 * basis[12]))
    frame = np.array([change @ (scale * matrix) @ change.conj().T
                      for scale, matrix in zip(scales, [MIXING, *HOPPING])])
    spin, solve = solve_selected_generators(frame, basis)
    expected = np.array([change @ np.array(t).astype(complex) @ change.conj().T
                         for t in exact_generators])
    solve['exact_to_selected_frame_error'] = norm(spin - expected)
    band = RotationBand(args.a, m=args.m * scales[0], r=args.r * scales[0],
                        v=args.v * scales[1], e=1., improved=True)
    return frame, spin, band, change, gram, exact_solve, solve, group, quaternions, rotations


def cubic_section(frame, spin, quaternions, rotations):
    """Choose one of the actual derived connected lifts over each cubic rotation."""
    keys, matrices, lifts, section_q = [], [], [], []
    for q, exact_rotation in zip(quaternions, rotations):
        rotation = np.array(exact_rotation).astype(int)
        key = tuple(rotation.ravel())
        if key in keys:
            continue
        keys.append(key)
        matrices.append(rotation)
        numeric_q = np.array(q).astype(float)
        section_q.append(numeric_q)
        lifts.append(numeric_q[0] * np.eye(4)
                     - 2j * np.einsum('i,ijk->jk', numeric_q[1:], spin))
    index = {key: i for i, key in enumerate(keys)}
    products = np.array([[index[tuple((r @ s).ravel())] for s in matrices] for r in matrices])
    cocycle = np.zeros((24, 24), int)
    central_errors = []
    for r, s in itertools.product(range(24), repeat=2):
        factor = lifts[r] @ lifts[s] @ lifts[products[r, s]].conj().T
        defects = [norm(factor - 1j**k * np.eye(4)) for k in range(4)]
        cocycle[r, s] = int(np.argmin(defects))
        central_errors.append(min(defects))
    failures = sum((cocycle[r, s] + cocycle[products[r, s], t]
                    - cocycle[s, t] - cocycle[r, products[s, t]]) % 4 != 0
                   for r, s, t in itertools.product(range(24), repeat=3))
    # A different gauge-fixed section has a Z4, rather than merely Z2, cocycle.
    phase = np.arange(24) % 4
    rephased = [1j**int(k) * u for k, u in zip(phase, lifts)]
    changed_cocycle = (cocycle + phase[:, None] + phase[None, :] - phase[products]) % 4
    rephase_error = max(norm(rephased[r] @ rephased[s]
                            - 1j**int(changed_cocycle[r, s]) * rephased[products[r, s]])
                        for r, s in itertools.product(range(24), repeat=2))
    rx = index[tuple(np.diag([1, -1, -1]).ravel())]
    ry = index[tuple(np.diag([-1, 1, -1]).ravel())]
    commutator = lifts[rx] @ lifts[ry] @ lifts[rx].conj().T @ lifts[ry].conj().T
    characters = [dict(q=q, centre_i=complex_json(np.array(1j**q)),
                       two_pi_character=int((-1)**q),
                       commuting_pi_rotation_commutator=int((-1)**q)) for q in range(4)]
    covariance = max(norm(sum(r[i, j] * u @ frame[j + 1] @ u.conj().T for j in range(3))
                          - frame[i + 1]) for r, u in zip(matrices, lifts) for i in range(3))
    report = dict(section_rotations=[r.tolist() for r in matrices],
                  section_quaternions=[q.tolist() for q in section_q],
                  actual_derived_lifts=[complex_json(u) for u in lifts],
                  multiplication_table=products.tolist(), z4_cocycle_exponents=cocycle.tolist(),
                  connected_cocycle_values=sorted(set(cocycle.ravel().tolist())),
                  all_13824_cocycle_associativity_failures=int(failures),
                  all_576_lift_product_max_error=max(central_errors),
                  rephased_section_exponents=phase.tolist(),
                  rephased_z4_cocycle_exponents=changed_cocycle.tolist(),
                  rephased_section_multiplication_error=rephase_error,
                  selected_order_covariance_error=covariance,
                  commuting_pi_rotations=[rx, ry],
                  commuting_spatial_rotation_error=norm(matrices[rx] @ matrices[ry]
                                                       - matrices[ry] @ matrices[rx]),
                  actual_lift_commutator_minus_identity_error=norm(commutator + np.eye(4)),
                  charge_characters=characters,
                  continuum_two_pi_error=norm(expm(-2j * np.pi * spin[2]) + np.eye(4)))
    return report, section_q, products, cocycle


class PauliZ4Pair:
    """Full physical-parity eight-mode primitive code times a four-state link.

    One neutral marker supplies all 256 physical Fock states. It is not charged.
    Each cell includes a static conjugate-six centre reference of charge -2.
    Only its exact residual Z4 character is retained, not a fake SU4 Gauss law.
    """
    def __init__(self):
        self.words = EdgeWords(9, [(0, 1), (1, 2), (2, 3), (0, 4),
                                   (4, 5), (5, 6), (6, 7), (0, 8)])
        self.bits = []
        for mask in range(256):
            physical = tuple(i for i in range(8) if mask >> i & 1)
            self.bits.append(self.words.tree_basis_state(
                physical + ((8,) if len(physical) % 2 else ())))
        self.mask_of = {bit: mask for mask, bit in enumerate(self.bits)}
        self.n = np.array([[sum(mask >> i & 1 for i in range(4)),
                            sum(mask >> i & 1 for i in range(4, 8))] for mask in range(256)])
        self.q = self.n - 2
        self.raw = [(mask, ell) for mask in range(256) for ell in range(4)]
        self.g = np.array([((self.q[mask, 0] - ell) % 4,
                            (self.q[mask, 1] + ell) % 4) for mask, ell in self.raw])
        self.physical = np.flatnonzero(np.all(self.g == 0, axis=1))
        self.ell = np.array([ell for _, ell in self.raw])
        self.clock = diags(1j**self.ell).tocsr()
        self.gauss = [diags(1j**self.g[:, x]).tocsr() for x in range(2)]
        # Average every element of BOTH Z4 endpoint gauge groups.
        projector = np.ones(len(self.raw), complex)
        for x in range(2):
            projector *= sum(1j**(power * self.g[:, x]) for power in range(4)) / 4
        self.projector_error = norm(projector - np.isin(np.arange(len(self.raw)), self.physical))

    def transfer(self, target, source, dressed=True):
        rows, columns, values = [], [], []
        shift = (int(target < 4) - int(source < 4)) if dressed else 0
        for column, (mask, ell) in enumerate(self.raw):
            moved = self.words.transfer(self.bits[mask], source, target)
            if moved is not None:
                output, amplitude = moved
                rows.append(4 * self.mask_of[output] + (ell + shift) % 4)
                columns.append(column)
                values.append(amplitude)
        return csr_matrix((values, (rows, columns)), shape=(1024, 1024), dtype=complex)

    def matter_block(self, matrix, cell):
        result = np.zeros((256, 256), complex)
        terms = [(4 * cell + row, 4 * cell + column, complex(matrix[row, column]))
                 for row in range(4) for column in range(4) if abs(matrix[row, column]) > 1e-15]
        for column, bit in enumerate(self.bits):
            for output, amplitude in self.words.apply(terms, {bit: 1.}).items():
                result[self.mask_of[output], column] = amplitude
        return result

    def physical_matter(self, matrix):
        result = np.zeros((len(self.physical), len(self.physical)), complex)
        for row, raw_row in enumerate(self.physical):
            mask, ell = self.raw[raw_row]
            for column, raw_column in enumerate(self.physical):
                other, other_ell = self.raw[raw_column]
                if ell == other_ell:
                    result[row, column] = matrix[mask, other]
        return result

    def vector(self, state, ell):
        result = np.zeros(1024, complex)
        for bit, amplitude in state.items():
            result[4 * self.mask_of[bit] + ell] = amplitude
        return result

    def ladder(self, vector, state, cell, creation=True):
        result = {}
        for orbital, value in enumerate(vector):
            coefficient = value if creation else value.conjugate()
            for bit, amplitude in boundary_ladder(self.words, 8, 4 * cell + orbital,
                                                  state, creation).items():
                result[bit] = result.get(bit, 0j) + coefficient * amplitude
        return {bit: value for bit, value in result.items() if abs(value) > 1e-14}


def finite_consumer(band, change, spin, section_q, products, cocycle) -> dict[str, Any]:
    code = PauliZ4Pair()
    positive = change @ np.column_stack([band.spinor(np.zeros(3), i) for i in range(2)])
    negative = change @ CHARGE_CONJUGATION @ np.column_stack(
        [band.spinor(np.zeros(3), i).conjugate() for i in range(2)])
    sea = {code.bits[0]: 1 + 0j}
    for cell in range(2):
        for vector in negative.T:
            sea = code.ladder(vector, sea, cell)
    vacuum = code.vector(sea, 0)
    pair_vectors = []
    for electron, hole in itertools.product(range(2), repeat=2):
        pair = code.ladder(positive[:, electron], sea, 0)
        pair = code.ladder(negative[:, hole], pair, 1, creation=False)
        pair_vectors.append(code.vector(pair, 1))
    pair_basis = np.column_stack(pair_vectors)
    identity = np.eye(len(code.physical))
    generators = [[code.physical_matter(code.matter_block(t, cell)) for t in spin]
                  for cell in range(2)]
    restrictions = [[pair_basis[code.physical].conj().T @ k @ pair_basis[code.physical]
                     for k in local] for local in generators]
    rotations = [[], []]
    for cell in range(2):
        for q in section_q:
            vector = q[1:]
            sine = norm(vector)
            if sine < 1e-12:
                # The section's identity lift is +I; other central fibres are not selected.
                if q[0] < 0:
                    raise AssertionError('Unexpected negative identity in rotation section')
                rotations[cell].append(np.eye(4, dtype=complex))
            else:
                angle = 2 * np.arctan2(sine, q[0])
                axis_generator = np.einsum('i,ijk->jk', vector / sine, restrictions[cell])
                rotations[cell].append(expm(-1j * angle * axis_generator))
    local_cocycle_error = max(norm(rotations[cell][r] @ rotations[cell][s]
                                    - 1j**int(cocycle[r, s]) * rotations[cell][products[r, s]])
                               for cell in range(2) for r, s in itertools.product(range(24), repeat=2))
    global_rotations = [rotations[0][i] @ rotations[1][i] for i in range(24)]
    global_product_error = max(norm(global_rotations[r] @ global_rotations[s]
                                    - global_rotations[products[r, s]])
                               for r, s in itertools.product(range(24), repeat=2))
    local_two_pi = [expm(-2j * np.pi * local[2]) for local in generators]
    all_two_pi = expm(-2j * np.pi * (generators[0][2] + generators[1][2]))
    link_parity = diags((-1.)**code.ell).tocsr()
    centre_left = code.gauss[0] @ code.gauss[0]
    centre_right = code.gauss[1] @ code.gauss[1]
    # Keep BOTH the compensating gauge operation and the reference-seam inverse.
    # A projective localized symmetry is conditional on this physical seam prescription.
    localized_left = centre_left @ link_parity
    localized_right = centre_right @ link_parity
    physical_pair = pair_basis[code.physical]
    transfers = [code.transfer(target, source) for target in range(8) for source in range(8)]
    gauss_error = max(sparse_norm(g @ t - t @ g) for g in code.gauss for t in transfers)
    bad = code.transfer(0, 4, dressed=False)
    good = code.transfer(0, 4)
    link_clock = code.clock[:4, :4].toarray()
    moved = code.words.transfer(code.bits[16], 4, 0)
    if moved is None:
        raise AssertionError('Primitive one-particle transfer disappeared')
    link_shift = good[4:8, 64:68].toarray() / moved[1]
    link_algebra_error = max(norm(link_clock @ link_shift - 1j * link_shift @ link_clock),
                             norm(np.linalg.matrix_power(link_shift, 4) - np.eye(4)),
                             norm(np.linalg.matrix_power(link_clock, 4) - np.eye(4)))
    bare_defect = max(sparse_norm(g @ bad - bad @ g) for g in code.gauss)
    # The actual rest electron/hole pair is also made by a Wilson-dressed Pauli transfer.
    pair_creation = csr_matrix((1024, 1024), dtype=complex)
    for row, column in itertools.product(range(4), repeat=2):
        pair_creation += (positive[row, 0] * negative[column, 0].conjugate()
                          * code.transfer(row, 4 + column))
    prepared = pair_creation @ vacuum
    selected_pair = pair_basis[:, 0]
    phase = np.vdot(selected_pair, prepared)
    # Raw gauge transformations act on the link endpoint, not just matter.
    full_centre_covariance = sparse_norm(centre_left @ good @ centre_left.conj().T - good)
    matter_only = diags([(-1.)**int(code.q[mask, 0]) for mask, _ in code.raw], format='csr')
    matter_only_defect = sparse_norm(matter_only @ good @ matter_only.conj().T - good)
    local_generator_error = max(norm(local_two_pi[cell] @ physical_pair + physical_pair) for cell in range(2))
    flux_error = max(norm(local - link_parity[code.physical][:, code.physical].toarray())
                     for local in local_two_pi)
    # Constant rest-projector transport and a bare scalar spatial loop both give +I.
    rest_projector = positive @ positive.conj().T
    intermediate = expm(-.73j * spin[2])
    berry_projector_error = norm(intermediate @ rest_projector @ intermediate.conj().T - rest_projector)
    berry_transport = np.eye(2, dtype=complex)
    for _ in range(64):
        left, _, right = np.linalg.svd(positive.conj().T @ positive)
        berry_transport = (left @ right) @ berry_transport
    bare_scalar_generator = np.zeros((2, 2))
    bare_scalar_loop = expm(-2j * np.pi * bare_scalar_generator)
    wrong_background_g = np.array([((code.n[mask, 0] - ell) % 4,
                                    (code.n[mask, 1] + ell) % 4) for mask, ell in code.raw])
    wrong_phase = 1j**wrong_background_g[:, 0]
    marker_charge_defect = norm(np.array([(-1)**len(code.words.occupation(bit)) for bit in code.bits])
                               - np.array([(-1)**int(n.sum()) for n in code.n]))
    report = dict(primitive_owner='pauli_compton.EdgeWords.transfer/boundary_ladder on actual bosonic Pauli words',
                  bosonic_edge_qubits=len(code.words.edges), matter_full_parity_dimension=256,
                  electric_link_dimension=4, electric_basis=[0, 1, 2, 3], raw_dimension=1024,
                  actual_electric_clock=complex_json(link_clock),
                  actual_cyclic_shift_from_primitive_transfer=complex_json(link_shift),
                  z4_link_weyl_and_order_four_error=link_algebra_error,
                  complete_z4_gauss_dimension=len(code.physical),
                  physical_total_N4_dimension=int(sum(code.n[code.raw[i][0]].sum() == 4 for i in code.physical)),
                  static_reference_charge_per_cell=2,
                  reference='Conjugate-six selected sea-pair reference, frozen only after restricting to Z4; its centre character is i^-2=-1 and connected rotation is a singlet.',
                  gauss_convention='q_x=N_x-2 mod4; G_left=i^(q_left-ell), G_right=i^(q_right+ell); P=product_x sum_{k=0}^3 G_x^k/4',
                  gauss_projector_average_error=code.projector_error,
                  all_64_actual_pauli_transfer_gauss_commutator_error=gauss_error,
                  omitted_link_shift_gauss_defect=bare_defect,
                  primitive_pair_creation_state_error=norm(prepared - phase * selected_pair),
                  primitive_pair_creation_norm_error=abs(abs(phase) - 1),
                  pair_basis_orthonormality_error=norm(pair_basis.conj().T @ pair_basis - np.eye(4)),
                  selected_pair_gauss_error=max(norm(g @ pair_basis - pair_basis) for g in code.gauss),
                  selected_pair_charge_expectations=[float(np.vdot(selected_pair,
                      np.array([code.q[mask, cell] for mask, _ in code.raw]) * selected_pair).real) for cell in range(2)],
                  selected_pair_total_charge_error=norm(np.array([code.q[mask].sum() for mask, _ in code.raw]) * selected_pair),
                  sea_gauss_error=max(norm(g @ vacuum - vacuum) for g in code.gauss),
                  selected_sea_connected_rotation_singlet_error=max(
                      norm(k @ vacuum[code.physical]) for local in generators for k in local),
                  absent_centre_two_background_sea_gauss_defect=norm(wrong_phase * vacuum - vacuum),
                  neutral_marker_matter_parity_defect=marker_charge_defect,
                  complete_pure_gauge_centre_pair_error=max(norm(c @ pair_basis - pair_basis)
                                                            for c in (centre_left, centre_right)),
                  wilson_dressed_centre_operation_error=full_centre_covariance,
                  matter_only_centre_wilson_defect=matter_only_defect,
                  full_physical_space_global_two_pi_error=norm(all_two_pi - identity),
                  actual_pair_global_two_pi_error=norm(all_two_pi @ physical_pair - physical_pair),
                  conditional_localized_two_pi_odd_error=max(norm(localized_left @ pair_basis + pair_basis),
                                                             norm(localized_right @ pair_basis + pair_basis)),
                  two_localized_factors_global_plus_error=norm(localized_left @ localized_right @ pair_basis - pair_basis),
                  actual_primitive_local_generator_two_pi_odd_error=local_generator_error,
                  gauss_boundary_flux_localization_error=flux_error,
                  actual_pair_all_576_local_cocycle_product_error=local_cocycle_error,
                  actual_pair_all_576_global_rotation_product_error=global_product_error,
                  no_pure_gauge_control=dict(full_gauge_restored_endpoint_loop_phase=complex_json(
                      np.vdot(selected_pair, centre_left @ selected_pair)),
                      complete_link_action_pair_error=norm(centre_left @ pair_basis - pair_basis),
                      localized_rotation_definition='L_x(2pi)=G_x(-I) times inverse reference-seam matching B_x(-I); on physical states it equals boundary electric parity, not a pure gauge operation.',
                      left_link_endpoint_factor=complex_json(np.vdot(selected_pair, link_parity @ selected_pair)),
                      inverse_reference_seam_factor=complex_json(np.vdot(selected_pair, link_parity @ selected_pair)),
                      inference='The sign requires a spatial/order rotation with a separately held reference seam. Gauge-equivalent Higgs order alone also admits the trivial complete-gauge loop and does not establish that operational prescription.'),
                  berry_and_scalar_controls=dict(bare_spatial_scalar_two_pi=complex_json(bare_scalar_loop),
                      bare_scalar_loop_identity_error=norm(bare_scalar_loop - np.eye(2)),
                      constant_rest_projector_rotation_error=berry_projector_error,
                      constant_rest_basis_berry_holonomy=complex_json(berry_transport),
                      actual_polar_overlap_berry_identity_error=norm(berry_transport - np.eye(2)),
                      compensated_rest_lift_two_pi_error=norm(positive.conj().T
                          @ expm(-2j * np.pi * spin[2]) @ positive + np.eye(2))))
    return report


def run(args):
    frame, spin, band, change, gram, exact_solve, solve, group, quaternions, rotations = selected_frame(args)
    section, section_q, products, cocycle = cubic_section(frame, spin, quaternions, rotations)
    finite = finite_consumer(band, change, spin, section_q, products, cocycle)
    checks = dict(
        exact_18_unknown_stabilizer=exact_solve['real_unknowns'] == 18
            and exact_solve['exact_rank'] == 15 and all(exact_solve['exact_checks'].values()),
        common_commutant_is_z4=exact_solve['common_commutant']['scalar_only']
            and exact_solve['common_commutant']['exact_dimension'] == 1,
        actual_selected_18_unknown_solve=solve['numeric_rank'] == 15 and solve['internal_map_rank'] == 15
            and max(solve['normalized_lift_residual'], solve['exact_to_selected_frame_error']) < 1e-9,
        actual_derived_cubic_lifts=all(group['exact_checks'].values())
            and section['all_576_lift_product_max_error'] < 1e-9
            and section['selected_order_covariance_error'] < 1e-9,
        associative_z4_cocycle=section['all_13824_cocycle_associativity_failures'] == 0
            and section['connected_cocycle_values'] == [0, 2]
            and section['rephased_section_multiplication_error'] < 1e-9,
        nontrivial_odd_rotation_class=section['commuting_spatial_rotation_error'] < 1e-9
            and section['actual_lift_commutator_minus_identity_error'] < 1e-9
            and section['continuum_two_pi_error'] < 1e-9,
        full_z4_gauss_projection=finite['gauss_projector_average_error'] < 1e-10
            and finite['complete_z4_gauss_dimension'] == 72
            and finite['physical_total_N4_dimension'] == 70,
        genuine_z4_electric_link=finite['z4_link_weyl_and_order_four_error'] < 1e-9,
        actual_gauss_preserving_pauli_transfers=finite['all_64_actual_pauli_transfer_gauss_commutator_error'] < 1e-9,
        actual_selected_electron_hole_pair=max(finite['primitive_pair_creation_state_error'],
            finite['primitive_pair_creation_norm_error'], finite['pair_basis_orthonormality_error'],
            finite['selected_pair_gauss_error'], finite['selected_pair_total_charge_error'],
            finite['sea_gauss_error'], finite['selected_sea_connected_rotation_singlet_error']) < 1e-9,
        full_centre_and_wilson_cancellation=max(finite['complete_pure_gauge_centre_pair_error'],
            finite['wilson_dressed_centre_operation_error']) < 1e-9,
        neutral_closed_global_rotation=max(finite['full_physical_space_global_two_pi_error'],
            finite['actual_pair_global_two_pi_error'], finite['actual_pair_all_576_global_rotation_product_error']) < 1e-9,
        conditional_localized_boundary_action=max(finite['conditional_localized_two_pi_odd_error'],
            finite['two_localized_factors_global_plus_error'], finite['actual_primitive_local_generator_two_pi_odd_error'],
            finite['gauss_boundary_flux_localization_error'], finite['actual_pair_all_576_local_cocycle_product_error']) < 1e-9,
        wrong_omitted_link_and_background_detected=min(finite['omitted_link_shift_gauss_defect'],
            finite['matter_only_centre_wilson_defect'], finite['absent_centre_two_background_sea_gauss_defect'],
            finite['neutral_marker_matter_parity_defect']) > 1,
        pure_gauge_counterexample_retained=finite['no_pure_gauge_control']['complete_link_action_pair_error'] < 1e-9,
        berry_not_rotation=max(finite['berry_and_scalar_controls']['constant_rest_projector_rotation_error'],
            finite['berry_and_scalar_controls']['bare_scalar_loop_identity_error'],
            finite['berry_and_scalar_controls']['actual_polar_overlap_berry_identity_error'],
            finite['berry_and_scalar_controls']['compensated_rest_lift_two_pi_error']) < 1e-9)
    return dict(status='conditional_local_gauge_rotation_fractionalization_with_reference_seam',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hypotheses=[
            'Microscopic matter is a spatial scalar SU4 fundamental; A0 is an adjoint spatial scalar and Ai an adjoint spatial vector.',
            'Flavour is a genuine local gauge redundancy, not the previous global flavour symmetry. The selected isotropic Clifford frame has common commutant Z4.',
            'A deconfined Higgs/Z4 charge sector and localized physical rotation symmetry are assumed, not established by this finite consumer.',
            'The localization prescription rotates the spatial source/order and keeps a separate reference seam; its endpoint action contains the inverse boundary matching. Co-rotating the full gauge dressing instead gives the explicit +1 control.',
            'The cubic24 symmetry is exact on the lattice; SO3 and its continuous lift require the rotational longwave limit.'],
        noether_relation=dict(units='hbar=1; gauge coupling absorbed in connection and E',
            gauss_convention='G^a=rho^a-(D_i E_i)^a; rho includes matter and adjoint-Higgs momenta',
            equation='K_diag[T]=J_space[T]+Q_gauge[T]',
            gauge_charge='Q_gauge[alpha]=integral(alpha rho+E_i D_i alpha)=integral(alpha G)+integral_boundary(alpha E_i n_i)',
            neutral_closed='On closed physical states G=0 and no boundary flux, K_diag=J_space; complete physical 2pi=+1.',
            open_charged='Bulk Gauss does not remove the boundary flux. Endpoint rotation factors can be projective under the stated localization prescription; no replacement Q_gauge=0 on a single charged state is made.'),
        selected_gram=gram['G'].tolist(), selected_stationarity_error=gram['gradient_residual'],
        exact_stabilizer=exact_solve, actual_selected_stabilizer=solve,
        cubic_section=section, finite_pauli_z4_consumer=finite,
        operational_limits=[
            'Rotation source: physical rotations of coordinates and adjoint spatial vector order, followed by the derived gauge-frame restoration; not an externally supplied spin generator.',
            'Reference arm: a separately held frame/seam with boundary electric matching is needed to distinguish localized rotation from complete pure gauge; no implemented apparatus or dynamical arm protocol is claimed.',
            'The compensator and deconfinement hypotheses alone do not select that protocol: the complete gauge-restored local loop gives +1 and is an explicit counterexample to inferring mechanical spin from a pure-gauge sign.',
            'This is a residual Z4 electric-link Hilbert space, not full SU4 Peter-Weyl, SU4 Gauss projection, non-Abelian link transport, or a simulated quantum Higgs phase.',
            'No full QED, microscopic quantum-law origin, measured electron spin, or mechanical interferometer claim.'],
        checks=checks, all_checks_pass=all(checks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--a', type=float, default=.2)
    parser.add_argument('--m', type=float, default=.8)
    parser.add_argument('--r', type=float, default=.08)
    parser.add_argument('--v', type=float, default=.7)
    parser.add_argument('--stiffness', type=float, default=1.)
    parser.add_argument('--preferred', type=float, default=0.)
    args = parser.parse_args()
    if min(args.a, args.m, args.v, args.stiffness) <= 0 or min(args.r, args.preferred) < 0:
        parser.error('Require a,m,v,stiffness>0 and r,preferred>=0')
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

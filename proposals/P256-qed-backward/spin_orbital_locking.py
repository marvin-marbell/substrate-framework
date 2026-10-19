#!/usr/bin/env python3
"""Derive the physical diagonal spin group of the selected bosonic orbital phase.

Exact symbolic stabilizers/group closure and actual primitive edge-Pauli state
consumers. This is conditional on local phase stability, not quantum SSB.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.linalg import expm

from pauli_compton import EdgeWords, block_terms, boundary_ladder, overlap
from rotation_band import (CHARGE_CONJUGATION, HOPPING, I4, MIXING,
                           RotationBand, spatial_rotation)
from sea_selection import coefficients
from soft_budget import select_gram


CYCLES = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
# Omega is MINUS the active spatial cross-product matrix. With U=exp(-i theta T)
# the order transforms as A'_i=R_ij U A_j U^dagger, giving the requested equation.
OMEGA = (sp.Matrix([[0, 0, 0], [0, 0, 1], [0, -1, 0]]),
         sp.Matrix([[0, 0, -1], [0, 0, 0], [1, 0, 0]]),
         sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))


def exact_units():
    """The fifteen traceless Hermitian matrix units, without a spin assignment."""
    two = (sp.eye(2), sp.Matrix([[0, 1], [1, 0]]),
           sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.diag(1, -1))
    return [sp.kronecker_product(a, b) for i, a in enumerate(two)
            for j, b in enumerate(two) if (i, j) != (0, 0)]


def exact_matrix(matrix):
    # Existing selected matrix units have integer real/imaginary entries.
    return sp.Matrix([[int(value.real) + sp.I * int(value.imag) for value in row]
                      for row in matrix])


def real_column(matrices):
    entries = [entry for matrix in matrices for entry in matrix]
    return sp.Matrix([sp.re(entry) for entry in entries]
                     + [sp.im(entry) for entry in entries])


def stabilizer_map(frame, basis):
    columns = [real_column([sp.I * (t * a - a * t) for a in frame])
               for t in basis]
    for omega in OMEGA:
        changes = [sp.zeros(4)] + [sum((omega[i, j] * frame[j + 1]
                                       for j in range(3)), sp.zeros(4))
                                   for i in range(3)]
        columns.append(real_column(changes))
    return sp.Matrix.hstack(*columns)


def exact_stabilizer(frame, basis, derive=False) -> tuple[dict[str, Any], list | None]:
    linear = stabilizer_map(frame, basis)
    kernel = linear.nullspace()
    nullity = len(kernel)
    projection = sp.Matrix.hstack(*(v[15:18, :] for v in kernel)) if kernel else sp.zeros(3, 0)
    spatial_dimension = projection.rank()
    result: dict[str, Any] = dict(real_unknowns=18, real_equations=linear.rows,
                  exact_rank=18 - nullity, connected_dimension=nullity,
                  spatial_projection_dimension=spatial_dimension,
                  pure_internal_dimension=nullity - spatial_dimension,
                  spatial_axes=[[str(entry) for entry in projection[:, i]]
                                for i in range(projection.cols)])
    if not derive:
        return result, None
    if nullity != 3 or spatial_dimension != 3:
        raise AssertionError('Selected frame does not have a unique spatial lift')
    kernel_matrix = sp.Matrix.hstack(*kernel)
    normalized = kernel_matrix * projection.inv()
    generators = [sp.zeros(4) for _ in range(3)]
    for i in range(3):
        for a in range(15):
            generators[i] += normalized[a, i] * basis[a]
    zero = sp.zeros(4)
    checks = dict(
        normalized_kernel=linear * normalized == sp.zeros(linear.rows, 3),
        su2=all(generators[i] * generators[j] - generators[j] * generators[i]
                == sp.I * generators[k] for i, j, k in CYCLES),
        spin_half_casimir=sum((t * t for t in generators), zero) == sp.Rational(3, 4) * sp.eye(4),
        clifford_generator_identity=all(generators[k] == -sp.I * frame[i + 1] * frame[j + 1] / 2
                                        for i, j, k in CYCLES),
        generator_anticommutators=all(generators[i] * generators[j] + generators[j] * generators[i]
                                    == (sp.eye(4) / 2 if i == j else zero)
                                    for i in range(3) for j in range(3)))
    result.update(exact_checks=checks,
                  generators=[[[str(entry) for entry in row] for row in t.tolist()]
                              for t in generators],
                  axial_eigenvalues={str(value): multiplicity for value, multiplicity
                                     in generators[2].eigenvals().items()})
    if not all(checks.values()):
        raise AssertionError('Solved stabilizer does not give spin half')
    return result, generators


def quaternion_product(q, r):
    a, x, y, z = q
    b, u, v, w = r
    return tuple(sp.expand(entry) for entry in
                 (a * b - x * u - y * v - z * w,
                  a * u + b * x + y * w - z * v,
                  a * v + b * y + z * u - x * w,
                  a * w + b * z + x * v - y * u))


def quaternion_rotation(q):
    a, x, y, z = q
    return sp.Matrix([[sp.expand(entry) for entry in row] for row in
                      ((1 - 2 * (y*y + z*z), 2 * (x*y - a*z), 2 * (x*z + a*y)),
                       (2 * (x*y + a*z), 1 - 2 * (x*x + z*z), 2 * (y*z - a*x)),
                       (2 * (x*z - a*y), 2 * (y*z + a*x), 1 - 2 * (x*x + y*y)))])


def quaternion_lift(q, generators):
    return (q[0] * sp.eye(4) - 2 * sp.I * sum(
        (q[i + 1] * generators[i] for i in range(3)), sp.zeros(4))).applyfunc(sp.expand)


def proper_cubic_rotations():
    matrices = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = sp.zeros(3)
            for column, row in enumerate(permutation):
                matrix[row, column] = signs[column]
            if matrix.det() == 1:
                matrices.append(matrix)
    return matrices


def matrix_key(matrix):
    return tuple(matrix[row, column] for row in range(matrix.rows)
                 for column in range(matrix.cols))


def binary_octahedral(generators) -> tuple[dict[str, Any], list, list]:
    """Generate lifts from the SOLVED generators, never from a supplied SU(2)."""
    d = sp.sqrt(2) / 2
    quarters = [(d, d, 0, 0), (d, 0, d, 0), (d, 0, 0, d)]
    identity = (sp.Integer(1), sp.Integer(0), sp.Integer(0), sp.Integer(0))
    elements, indices, queue = [identity], {identity: 0}, deque([identity])
    while queue:
        q = queue.popleft()
        for quarter in quarters:
            product = quaternion_product(q, quarter)
            if product not in indices:
                indices[product] = len(elements)
                elements.append(product)
                queue.append(product)
                if len(elements) > 48:
                    raise AssertionError('Quarter-turn lifts exceed the binary octahedral group')
    rotations = [quaternion_rotation(q) for q in elements]
    lifts = [quaternion_lift(q, generators) for q in elements]
    cubic = {matrix_key(r): r for r in proper_cubic_rotations()}
    fibres = {key: [] for key in cubic}
    for index, rotation in enumerate(rotations):
        fibres[matrix_key(rotation)].append(index)
    table = [[indices[quaternion_product(q, r)] for r in elements] for q in elements]
    exact_representation = all(
        (lifts[i] * quaternion_lift(quarter, generators)).applyfunc(sp.expand)
        == lifts[indices[quaternion_product(q, quarter)]]
        for i, q in enumerate(elements) for quarter in quarters)
    kernel = [i for i, r in enumerate(rotations) if r == sp.eye(3)]
    checks = dict(proper_signed_permutations_24=len(cubic) == 24,
                  generated_spinorial_lifts_48=len(elements) == 48,
                  all_2304_products_closed=len(table) == 48 and all(len(row) == 48 for row in table),
                  two_lifts_per_rotation=all(len(fibre) == 2 for fibre in fibres.values()),
                  exact_lift_representation=exact_representation,
                  kernel_is_plus_minus_identity=len(kernel) == 2
                  and {matrix_key(lifts[i]) for i in kernel} == {matrix_key(sp.eye(4)), matrix_key(-sp.eye(4))})
    if not all(checks.values()):
        raise AssertionError('Exact cubic spin lift fails closure or covariance projection')
    report = dict(exact_checks=checks, multiplication_table=table,
                  quaternions=[[str(value) for value in q] for q in elements],
                  proper_signed_permutations=[np.array(r).astype(int).tolist() for r in cubic.values()],
                  full_SU4_stabilizer_caveat='The adjoint order also admits the disconnected Z4 flavour centre; '
                  'the generated 48-element group is the lift inherited from the connected diagonal Spin(3), '
                  'not the entire 96-element SU4 preimage of the cubic group.')
    return report, elements, rotations


def matrix_norm(matrix):
    return float(np.linalg.norm(matrix))


def solve_selected_generators(frame, basis):
    """Solve the actual scaled/conjugated field equations in all 18 real variables."""
    columns = []
    for t in basis:
        changes = 1j * (t @ frame - frame @ t)
        columns.append(np.r_[changes.real.ravel(), changes.imag.ravel()])
    for omega in OMEGA:
        changes = np.zeros_like(frame)
        changes[1:] = np.einsum('ij,jkl->ikl', np.array(omega).astype(float), frame[1:])
        columns.append(np.r_[changes.real.ravel(), changes.imag.ravel()])
    linear = np.column_stack(columns)
    solutions, _, rank, _ = np.linalg.lstsq(linear[:, :15], -linear[:, 15:], rcond=None)
    generators = np.einsum('ak,aij->kij', solutions, basis)
    singular = np.linalg.svd(linear, compute_uv=False)
    tolerance = 1e-10 * singular[0]
    return generators, dict(numeric_rank=int(np.count_nonzero(singular > tolerance)),
                            internal_map_rank=int(rank),
                            normalized_lift_residual=matrix_norm(linear[:, :15] @ solutions + linear[:, 15:]),
                            singular_values=singular.tolist())


def primitive_matrix(words, bits, operator):
    terms = block_terms(operator, 0, 0)
    return np.array([[words.apply(terms, {source: 1 + 0j}).get(target, 0j)
                      for source in bits] for target in bits])


def linear_ladder(words, vector, state, creation=True):
    result = {}
    for orbital, value in enumerate(vector):
        coefficient = value if creation else np.conjugate(value)
        for bits, amplitude in boundary_ladder(words, 4, orbital, state, creation).items():
            result[bits] = result.get(bits, 0j) + coefficient * amplitude
    return {bits: value for bits, value in result.items() if abs(value) > 1e-14}


def state_vector(state, bits):
    return np.array([state.get(bit, 0j) for bit in bits])


def primitive_consumer(band, field_basis, generators) -> dict[str, Any]:
    """Actual edge-Pauli matrices and odd charged states, not CAR comparator arrays."""
    words = EdgeWords(5, [(0, 1), (1, 2), (2, 3), (0, 4)])
    bits, occupations = [], []
    for mask in range(16):
        physical = tuple(i for i in range(4) if mask >> i & 1)
        full = physical + ((4,) if len(physical) % 2 else ())
        bits.append(words.tree_basis_state(full))
        occupations.append(len(physical))
    empty = {bits[0]: 1 + 0j}
    positive = field_basis @ np.column_stack([band.spinor(np.zeros(3), i) for i in range(2)])
    negative = field_basis @ CHARGE_CONJUGATION @ np.column_stack(
        [band.spinor(np.zeros(3), i).conjugate() for i in range(2)])
    vacuum = empty
    for vector in negative.T:
        vacuum = linear_ladder(words, vector, vacuum)
    electrons = [linear_ladder(words, vector, vacuum) for vector in positive.T]
    holes = [linear_ladder(words, vector, vacuum, False) for vector in negative.T]
    pair = linear_ladder(words, positive[:, 1], electrons[0])
    states = [('empty', empty, 0), ('filled_sea', vacuum, 2),
              ('electron_up', electrons[0], 3), ('electron_down', electrons[1], 3),
              ('hole_up', holes[0], 1), ('hole_down', holes[1], 1), ('electron_pair', pair, 4)]
    primitive_generators = [primitive_matrix(words, bits, t) for t in generators]
    axis = np.array([1., 2., -1.]) / np.sqrt(6)
    total_spin = np.einsum('i,ijk->jk', axis, primitive_generators)
    two_pi = expm(-2j * np.pi * total_spin)
    parity = np.diag([(-1) ** number for number in occupations])
    number = primitive_matrix(words, bits, I4)
    raw_number = np.diag([len(words.occupation(bit)) for bit in bits])
    rows = []
    for name, state, expected_number in states:
        vector = state_vector(state, bits)
        marker = sum(abs(value)**2 * words.occupied(bit, 4) for bit, value in state.items())
        rows.append(dict(state=name, physical_N=expected_number,
                         norm_error=abs(overlap(state, state) - 1),
                         number_eigenstate_error=matrix_norm(number @ vector - expected_number * vector),
                         marker_occupation=float(marker),
                         marker_parity_error=abs(marker - expected_number % 2),
                         two_pi_state_error=matrix_norm(two_pi @ vector - (-1) ** expected_number * vector),
                         normal_ordered_charge=float(band.charge * np.vdot(vector, (number - 2 * np.eye(16)) @ vector).real),
                         wrong_marker_charged_charge=float(band.charge * np.vdot(
                             vector, (raw_number - 2 * np.eye(16)) @ vector).real)))
    vacuum_vector = state_vector(vacuum, bits)
    restrictions = []
    for name, state_basis, expected_charge in (('electron', electrons, band.charge),
                                               ('hole', holes, -band.charge)):
        vectors = np.column_stack([state_vector(state, bits) for state in state_basis])
        matrices = [vectors.conj().T @ t @ vectors for t in primitive_generators]
        restrictions.append(dict(sector=name,
            axial_spin_eigenvalues=np.linalg.eigvalsh(matrices[2]).tolist(),
            spin_half_casimir_error=matrix_norm(sum(t @ t for t in matrices) - .75 * np.eye(2)),
            su2_error=max(matrix_norm(matrices[i] @ matrices[j] - matrices[j] @ matrices[i]
                                     - 1j * matrices[k]) for i, j, k in CYCLES),
            charge_error=matrix_norm(band.charge * vectors.conj().T @ (number - 2 * np.eye(16)) @ vectors
                                    - expected_charge * np.eye(2))))
    pin, pout = np.array([.17, -.11, .23]), np.array([-.07, .19, .13])
    current = [field_basis @ j @ field_basis.conj().T for j in band.current(pout, pin)]
    code_current = [primitive_matrix(words, bits, j) for j in current]
    quarter_rows = []
    for axis_index in range(3):
        axis_vector = np.eye(3)[axis_index]
        rotation = np.rint(spatial_rotation(axis_vector, np.pi / 2)).astype(int)
        code_rotation = expm(-.5j * np.pi * primitive_generators[axis_index])
        rotated_current = [primitive_matrix(words, bits, field_basis @ j @ field_basis.conj().T)
                           for j in band.current(rotation @ pout, rotation @ pin)]
        target = np.einsum('ji,jkl->ikl', rotation, rotated_current)
        quarter_rows.append(dict(axis=axis_index,
            full_16_state_current_covariance_error=max(matrix_norm(code_rotation @ code_current[i]
                @ code_rotation.conj().T - target[i]) for i in range(3)),
            filled_sea_rotation_error=matrix_norm(code_rotation @ vacuum_vector - vacuum_vector)))
    parity_error = matrix_norm(two_pi - parity)
    wrong_marker_parity = np.diag([(-1) ** len(words.occupation(bit)) for bit in bits])
    checks = dict(full_physical_parity=parity_error < 1e-10,
                  actual_states=all(max(row['norm_error'], row['number_eigenstate_error'],
                      row['marker_parity_error'], row['two_pi_state_error']) < 1e-10 for row in rows),
                  electron_and_hole_spin_half=all(max(row['spin_half_casimir_error'], row['su2_error'],
                      row['charge_error']) < 1e-10 for row in restrictions),
                  filled_sea_spin_zero=max(matrix_norm(t @ vacuum_vector) for t in primitive_generators) < 1e-10,
                  physical_current_covariance=all(max(row['full_16_state_current_covariance_error'],
                      row['filled_sea_rotation_error']) < 1e-10 for row in quarter_rows),
                  neutral_marker_not_counted_as_matter=matrix_norm(two_pi - wrong_marker_parity) > 1,
                  marker_mischarge_discriminated=max(abs(row['wrong_marker_charged_charge']
                      - row['normal_ordered_charge']) for row in rows) > abs(band.charge) / 2)
    return dict(actual_bosonic_edge_qubits=len(words.edges), physical_dimension=len(bits),
                state_rows=rows, charged_spin_restrictions=restrictions,
                two_pi_equals_physical_parity_error=parity_error,
                wrong_marker_inclusive_parity_error=matrix_norm(two_pi - wrong_marker_parity),
                wrong_marker_inclusive_number_error=matrix_norm(raw_number - number),
                quarter_turn_current_consumers=quarter_rows, checks=checks,
                construction='Same EdgeWords/block_terms/boundary_ladder neutral-marker law as pauli_compton; '
                             'all generators/current matrices evaluated by actual Pauli transfers.')


def run(args):
    basis = exact_units()
    selected = [exact_matrix(MIXING), *(exact_matrix(a) for a in HOPPING)]
    lie, exact_generators = exact_stabilizer(selected, basis, derive=True)
    if exact_generators is None:
        raise AssertionError('No selected generators')
    group, quaternions, rotations = binary_octahedral(exact_generators)
    controls = []
    for name, frame, expected, subgroup in (
            ('commuting_identical_vectors', [selected[0]] + [selected[3]] * 3,
             (14, 4, 1, 3), 'su(2) pure internal plus so(2) physical about (1,1,1); not isotropic locking'),
            ('three_distinct_spatial_scales', [selected[0], selected[1], sp.Rational(4, 5) * selected[2],
                                             sp.Rational(6, 5) * selected[3]],
             (18, 0, 0, 0), 'zero connected stabilizer; only discrete rotations can remain'),
            ('axially_anisotropic', [selected[0], selected[1], selected[2], sp.Rational(6, 5) * selected[3]],
             (17, 1, 1, 0), 'one diagonal so(2) about z; not so(3)')):
        report, _ = exact_stabilizer(frame, basis)
        actual = tuple(report[key] for key in ('exact_rank', 'connected_dimension',
                                               'spatial_projection_dimension', 'pure_internal_dimension'))
        report.update(name=name, remaining_connected_lie_algebra=subgroup, expected_rank_verified=actual == expected)
        controls.append(report)
    bare = RotationBand(args.a, m=args.m, r=args.r, v=args.v, e=args.e, improved=True)
    momenta = np.array(list(itertools.product(range(3), repeat=3))) * (2 * np.pi / (3 * args.a))
    gram_solution = select_gram(coefficients(bare, momenta), args.stiffness, 1.)
    scales = np.sqrt(np.diag(gram_solution['G']))
    band = RotationBand(args.a, m=args.m * scales[0], r=args.r * scales[0],
                        v=args.v * scales[1], e=args.e, improved=True)
    numeric_basis = np.array([np.array(t).astype(complex) for t in basis])
    field_basis = expm(-1j * (.31 * numeric_basis[4] - .23 * numeric_basis[9] + .17 * numeric_basis[12]))
    frame = np.array([field_basis @ (scale * matrix) @ field_basis.conj().T
                      for scale, matrix in zip(scales, np.array([MIXING, *HOPPING]))])
    generators, actual_lie = solve_selected_generators(frame, numeric_basis)
    expected_generators = np.array([field_basis @ np.array(t).astype(complex) @ field_basis.conj().T
                                   for t in exact_generators])
    actual_lie['same_field_basis_generator_error'] = matrix_norm(generators - expected_generators)
    mapping = max(matrix_norm(np.einsum('m,mij->ij', x, frame)
                              - field_basis @ band.h(p) @ field_basis.conj().T)
                  for x, p in zip(coefficients(bare, momenta), momenta))
    p = np.array([.23, -.31, .17])
    pin, pout = -.4 * p, .6 * p
    cubic_rows = []
    for q, exact_rotation in zip(quaternions, rotations):
        rotation = np.array(exact_rotation).astype(int)
        lift = float(q[0]) * I4 - 2j * np.einsum('i,ijk->jk', np.array(q[1:]).astype(float), generators)
        h = field_basis @ band.h(p) @ field_basis.conj().T
        rotated_h = field_basis @ band.h(rotation @ p) @ field_basis.conj().T
        current = np.array([field_basis @ j @ field_basis.conj().T for j in band.current(pout, pin)])
        rotated_current = np.array([field_basis @ j @ field_basis.conj().T
                                    for j in band.current(rotation @ pout, rotation @ pin)])
        target_current = np.einsum('ji,jkl->ikl', rotation, rotated_current)
        cubic_rows.append(dict(rotation=rotation.tolist(),
            h_covariance_error=matrix_norm(lift @ h @ lift.conj().T - rotated_h),
            current_covariance_error=max(matrix_norm(lift @ current[i] @ lift.conj().T - target_current[i])
                                         for i in range(3)),
            order_covariance_error=max(matrix_norm(sum(rotation[i, j] * lift @ frame[j + 1] @ lift.conj().T
                                                      for j in range(3)) - frame[i + 1]) for i in range(3))))
    axis, angle = np.array([1., 2., -1.]), .71
    rotation = spatial_rotation(axis, angle)
    lift = expm(-1j * angle * np.einsum('i,ijk->jk', axis / np.linalg.norm(axis), generators))
    continuum_h = field_basis @ band.continuum_h(p) @ field_basis.conj().T
    continuous_error = matrix_norm(lift @ continuum_h @ lift.conj().T
                                  - field_basis @ band.continuum_h(rotation @ p) @ field_basis.conj().T)
    lattice_rows = []
    for a in (args.a, args.a / 2, args.a / 4, args.a / 8):
        member = RotationBand(a, m=band.m, r=band.r, v=band.v, e=band.e, improved=True)
        lattice_rows.append(dict(a=a, noncubic_covariance_defect=matrix_norm(
            lift @ (field_basis @ member.h(p) @ field_basis.conj().T) @ lift.conj().T
            - field_basis @ member.h(rotation @ p) @ field_basis.conj().T)))
    current = np.array([field_basis @ j @ field_basis.conj().T for j in band.current(pout, pin)])
    khat = 2 * np.sin(args.a * (pout - pin) / 2) / args.a
    ward_error = matrix_norm(np.einsum('i,ijk->jk', khat, current)
                            - field_basis @ (band.h(pout) - band.h(pin)) @ field_basis.conj().T)
    code = primitive_consumer(band, field_basis, generators)
    # A constant rest projector has a constant basis and identity Berry transport.
    # Physical rotation nevertheless acts nontrivially inside that subspace.
    rest_frame = field_basis @ np.column_stack([band.spinor(np.zeros(3), i) for i in range(2)])
    rest_projector = rest_frame @ rest_frame.conj().T
    two_pi = expm(-2j * np.pi * generators[2])
    intermediate_rotation = expm(-.73j * generators[2])
    berry: dict[str, Any] = dict(constant_positive_rest_projector=True,
                 projector_rotation_error=matrix_norm(intermediate_rotation @ rest_projector
                     @ intermediate_rotation.conj().T - rest_projector),
                 projector_generator_commutator_error=max(matrix_norm(t @ rest_projector - rest_projector @ t)
                                                          for t in generators),
                 constant_basis_berry_holonomy=np.eye(2).tolist(),
                 physical_two_pi_in_rest_band_error=matrix_norm(rest_frame.conj().T @ two_pi @ rest_frame + np.eye(2)),
                 inference='No minus sign is inferred from projector/Berry transport; it is the action of the solved unbroken physical symmetry.')
    checks = dict(exact_selected_lie=lie['exact_rank'] == 15 and all(lie['exact_checks'].values()),
                  exact_binary_cubic_group=all(group['exact_checks'].values()),
                  adverse_subgroups=all(row['expected_rank_verified'] for row in controls),
                  actual_selected_field_solve=actual_lie['numeric_rank'] == 15
                  and actual_lie['internal_map_rank'] == 15
                  and max(actual_lie['normalized_lift_residual'], actual_lie['same_field_basis_generator_error']) < 1e-10,
                  same_field_finite_gap_current_map=mapping < 1e-10 and band.m > 0 and ward_error < 1e-10,
                  all_48_lifts_sameband_covariant=all(max(row['h_covariance_error'], row['current_covariance_error'],
                      row['order_covariance_error']) < 1e-10 for row in cubic_rows),
                  continuum_physical_covariance=continuous_error < 1e-10,
                  noncubic_lattice_defect_decreases=all(lattice_rows[i + 1]['noncubic_covariance_defect']
                      < lattice_rows[i]['noncubic_covariance_defect'] for i in range(3)),
                  actual_primitive_states_and_current=all(code['checks'].values()),
                  berry_and_physical_action_distinguished=max(berry['physical_two_pi_in_rest_band_error'],
                      berry['projector_rotation_error'], berry['projector_generator_commutator_error']) < 1e-10)
    return dict(status='conditional_physical_spin_orbital_locking_theorem',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                hypotheses=['Fundamental complex quantum Hilbert law and bosonic microscopic edge/adjoint orbital variables.',
                            'Microscopic matter is a spatial scalar with exact global SU4 flavour symmetry; the derived primitive charge field is fundamental.',
                            'A0 is spatial scalar, Ai a vector; isotropic scalar gradients/hopping respect cubic rotations, continuum gradients SO3.',
                            'The selected uniform SPD metric-Clifford/cubic-isotropic phase is locally stable; quantum SSB is NOT proved here.'],
                units='hbar=1; a is length, m and r/a energies, v velocity, e charge; orbital Gram scales dimensionless.',
                exact_lie=lie, exact_cubic_group=group, adverse_frames=controls,
                selected_gram=gram_solution['G'].tolist(), actual_selected_lie=actual_lie,
                band_map=dict(effective_m=band.m, effective_v=band.v, effective_r=band.r,
                              minimum_positive_energy_bound=band.m, particle_hole_gap_bound=2 * band.m,
                              microscopic_same_field_map_error=mapping, exact_current_ward_error=ward_error),
                cubic_sameband_consumers=cubic_rows, continuum_covariance_error=continuous_error,
                noncubic_lattice_limit=lattice_rows, primitive_consumer=code,
                berry_counterexample=berry, checks=checks, all_checks_pass=all(checks.values()),
                limitations=['No thermodynamic quantum symmetry-breaking theorem or fundamental quantum-law derivation.',
                             'Continuous physical rotations belong to the longwave theory, not arbitrary exact finite-a rotations.',
                             'The connected diagonal Spin(3) is derived only under the scalar/vector and flavour symmetry hypotheses.',
                             'No adiabatic constant-projector spin sign, measured mass/charge, or full asymptotic photon dressing is inferred.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--a', type=float, default=.2)
    parser.add_argument('--m', type=float, default=.8)
    parser.add_argument('--r', type=float, default=.08)
    parser.add_argument('--v', type=float, default=.7)
    parser.add_argument('--e', type=float, default=1.)
    parser.add_argument('--stiffness', type=float, default=1.)
    args = parser.parse_args()
    if min(args.a, args.m, args.v, args.e, args.stiffness) <= 0 or args.r < 0:
        parser.error('Require a,m,v,e,stiffness>0 and r>=0')
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + '\n')
    print(payload)
    return 0 if result['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

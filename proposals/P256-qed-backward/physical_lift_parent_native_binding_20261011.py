"""Actual source-centre CAR/H/density binding; no frozen or physical ground solve."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import cast
import numpy as np
import sympy as s
import physical_ground_lift_source_20261011 as lift
import joined_phi_u1_bulk_flux_ring_consumer as primitive

ROOT = Path(__file__).resolve().parent


def rational(value: F) -> s.Rational:
    return cast(s.Rational, s.Rational(value.numerator, value.denominator))


def exact_matrix(matrix: list[list[primitive.C]]) -> s.Matrix:
    return s.Matrix([[rational(re) + s.I * rational(im) for re, im in row] for row in matrix])


def fock_lift(matrix: s.Matrix) -> s.Matrix:
    operators = []
    for mode in range(4):
        ann = s.zeros(16)
        for mask in range(16):
            if mask >> mode & 1:
                ann[mask ^ (1 << mode), mask] = (-1) ** ((mask & ((1 << mode) - 1)).bit_count())
        operators.append(ann)
    return cast(s.Matrix, sum((operators[i].T * operators[j] * matrix[i, j] for i, j in itertools.product(range(4), repeat=2)), s.zeros(16)))


def main() -> None:
    sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == pin for name, pin in request['pins'].items())
    frame, scale = primitive.exact_frame((ROOT / 'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz').read_bytes())
    phi = [exact_matrix(matrix) for matrix in frame]
    cfg = json.loads((ROOT / 'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json').read_text())['input_descriptor']['original_micro_config_UNCHANGED']
    profiles = primitive.profiles(cfg, frame, scale)
    frozen = json.loads((ROOT / 'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_text())['profiles']
    centre = np.array([np.array(matrix, dtype=complex) for matrix in phi])
    with np.load(ROOT / lift.STATE, allow_pickle=False) as archive:
        stored = archive['state']
    trial = lift.LiftedGround(stored, centre)
    amplitude_scale = rational(frame[1][0][3][0])
    spin = s.Matrix.hstack(s.eye(4)[:, 0], phi[1][:, 0] / amplitude_scale, s.eye(4)[:, 1], phi[1][:, 1] / amplitude_scale)
    assert spin.conjugate().T * spin == s.eye(4)
    assert np.array_equal(np.array(spin, dtype=complex), trial.spin)
    native = s.zeros(8)
    for spin_index in range(2):
        for site in range(2):
            for row in range(4):
                for col in range(2):
                    native[4 * site + row, 4 * spin_index + 2 * site + col] = spin[row, 2 * spin_index + col]
    phase = s.diag(1, -1, -1, 1, 1, -1, -1, 1)
    section = native * phase
    A = s.diag(-1, 1, -1, 1)
    X = s.Matrix([[0, 1], [1, 0]])
    B = s.kronecker_product(X, s.diag(1, -1))
    C = s.zeros(2).row_join(s.I * X).col_join((-s.I * X).row_join(s.zeros(2)))
    R = s.diag(s.Rational(1, 2), s.Rational(1, 2), -s.Rational(1, 2), -s.Rational(1, 2))
    density = s.diag(*([s.Rational(1, 2)] * 4 + [-s.Rational(1, 2)] * 4))
    assert section.conjugate().T * density * section == s.diag(R, R)
    J, _, _, _, _, V = lift.bridge.car_exact()
    slot_phase = s.diag(1, -s.I, 1, 1, s.I)
    embedding = V * J * slot_phase
    assert embedding.conjugate().T * embedding == s.eye(5)
    g1 = lift.bridge.generators(1, True)
    rows = []
    for profile, old in zip(profiles, frozen, strict=True):
        m, c0, d = (rational(F(old[key])) for key in ['m_exact', 'c0_exact', 'd_exact'])
        onsite = cast(s.Matrix, rational(profile['onsite'] / scale) * phi[0])
        hopping = exact_matrix(next(path['matrix'] for path in profile['paths'] if path['axis'] == 0 and path['length'] == 1))
        physical = onsite.row_join(hopping).col_join(hopping.conjugate().T.row_join(onsite))
        source = m * A + c0 * B + d * C
        assert section.conjugate().T * physical * section == s.diag(source, source)
        assert native.conjugate().T * physical * native != s.diag(source, source)
        harmonic_real = slot_phase.conjugate().T * (m * g1[0] + c0 * g1[1] + d * g1[2]) * slot_phase
        assert s.simplify(fock_lift(source) * embedding - embedding * harmonic_real) == s.zeros(16, 5)
        assert s.simplify(fock_lift(R) * embedding - embedding * (slot_phase.conjugate().T * g1[3] * slot_phase)) == s.zeros(16, 5)
        rows.append({'role': profile['role'], 'count': profile['count'], 'entire_native_eight_mode_H_exact': True, 'joint_H_density_slot_intertwiner_exact': True, 'wrong_omitted_oneparticle_phase_rejected': True})
    # One actual native occupation amplitude of the correlated ALL64 source.
    # Its slot kernel is exactly rank1. All normalized extraction steps on the
    # unique extremal l=n,p=q=0 ray have coefficient1, avoiding an exponential
    # branch enumeration while preserving the full stored correlated coefficient.
    occupation = [0, 1, 4, 5]
    pairs = list(itertools.combinations(range(4), 2))
    kernel = s.zeros(5)
    for a in pairs:
        for b in pairs:
            cols = list(a) + [i + 4 for i in b]
            mask_a, mask_b = sum(1 << i for i in a), sum(1 << i for i in b)
            kernel += section.extract(occupation, cols).det() * embedding[mask_a, :].T * embedding[mask_b, :]
    expected = s.zeros(5); expected[1, 1] = 1
    assert kernel == expected
    for n in range(1, 127):
        extremal_squared = F((2 * n + 1) * (2 * n + 2) * n * n, 2 * n * n * (n + 1) * (2 * n + 1))
        assert extremal_squared == 1
    amplitude = float(trial.g[lift.bridge.idx(2, 0, 0), lift.bridge.idx(126, 0, 0)])
    result = {'kind': 'EXACT_PHYSICAL_SOURCE_CENTRE_NATIVE_CAR_H_DENSITY_BINDING_WITH_STORED_ALL64_AMPLITUDE_NOT_GROUND', 'roles': rows, 'native_spin_frame_equals_actual_source': True, 'canonical_all1_plus63_source_kernel_exact': [[str(value) for value in row] for row in kernel.tolist()], 'all126_normalized_extremal_extraction_coefficients_exact_one': True, 'actual_correlated_binary_source_dimension': int(stored.size), 'actual_native_four_occupied_modes_per_species': occupation, 'actual_all64_one_matching_block_amplitude_float': amplitude, 'amplitude_scope': 'Diagnostic normalized stored x amplitude, NOT an interval exactg amplitude/full-field norm integral; exact kernel/isometry/formulas separate.', 'source_x_NOT_exactg': True, 'frozen_selected_ground_NOT_fullphysical_ground': True, 'historical_main_solver_root_checker_executed': False, 'campaign_complete': False}
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == pin for name, pin in request['pins'].items())
    with args.output.open('x') as out:
        json.dump(result, out, indent=2); out.write('\n')
    print(json.dumps({'native_H_density_bindings': len(rows), 'actual_source_dimension': int(stored.size), 'amplitude': amplitude, 'physical_ground': False}))


if __name__ == '__main__':
    main()

"""Native CAR/Jordan POINT loop discriminator, not a ground or stiffness theorem."""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
from itertools import permutations
import json
from pathlib import Path
import time
from typing import Any

from local_orbital_stability import scalar_paths
from rotation_band import RotationBand

ROOT = Path(__file__).resolve().parent
Z = tuple[F, F]
ZERO: Z = (F(0), F(0))
ONE: Z = (F(1), F(0))


def add(a: Z, b: Z) -> Z:
    return a[0] + b[0], a[1] + b[1]


def mul(a: Z, b: Z) -> Z:
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def conjugate(a: Z) -> Z:
    return a[0], -a[1]


def scale(a: Z, r: F) -> Z:
    return a[0] * r, a[1] * r


def multiply(a: list[list[Z]], b: list[list[Z]]) -> list[list[Z]]:
    out = [[ZERO for _ in b[0]] for _ in a]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                out[i][j] = add(out[i][j], mul(a[i][k], b[k][j]))
    return out


def adjoint(a: list[list[Z]]) -> list[list[Z]]:
    return [[conjugate(a[j][i]) for j in range(len(a))] for i in range(len(a[0]))]


def tensor(a: list[list[Z]], b: list[list[Z]]) -> list[list[Z]]:
    return [[mul(a[i // 2][j // 2], b[i % 2][j % 2]) for j in range(4)] for i in range(4)]


def trace(a: list[list[Z]]) -> Z:
    out = ZERO
    for i in range(len(a)):
        out = add(out, a[i][i])
    return out


def car(mask: int, create: int, annihilate: int, unsigned: bool = False) -> tuple[int, int] | None:
    if not mask & (1 << annihilate):
        return None
    sign = -1 if (mask & ((1 << annihilate) - 1)).bit_count() % 2 else 1
    mask ^= 1 << annihilate
    if mask & (1 << create):
        return None
    if (mask & ((1 << create) - 1)).bit_count() % 2:
        sign *= -1
    return mask | (1 << create), 1 if unsigned else sign


def loop(matrix_x: list[list[Z]], matrix_y: list[list[Z]], ref: int, unsigned: bool = False) -> Z:
    matrices = [matrix_x, matrix_y, adjoint(matrix_x), adjoint(matrix_y)]
    terms = [[(4 * edge + i, 4 * ((edge + 1) % 4) + j, a[i][j])
              for i in range(4) for j in range(4) if a[i][j] != ZERO]
             for edge, a in enumerate(matrices)]
    total = ZERO
    for order in permutations(range(4)):
        amplitudes = {ref: ONE}
        for edge in order:
            nxt: dict[int, Z] = defaultdict(lambda: ZERO)
            for mask, amplitude in amplitudes.items():
                for create, annihilate, coefficient in terms[edge]:
                    moved = car(mask, create, annihilate, unsigned)
                    if moved is not None:
                        target, sign = moved
                        nxt[target] = add(nxt[target], scale(mul(amplitude, coefficient), F(sign)))
            amplitudes = {mask: a for mask, a in nxt.items() if a != ZERO}
        total = add(total, amplitudes.get(ref, ZERO))
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request: dict[str, Any] = json.loads(Path(args.request).read_text())
    before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in request['pins']}
    assert before == request['pins']
    cfg = json.loads((ROOT / request['configuration']).read_text())['input_descriptor']['original_micro_config_UNCHANGED']
    assert cfg['a'] == 1 / 64 and cfg['auxiliary_count'] == 63
    i2 = [[ONE, ZERO], [ZERO, ONE]]
    x = [[ZERO, ONE], [ONE, ZERO]]
    y = [[ZERO, (F(0), F(-1))], [(F(0), F(1)), ZERO]]
    z = [[ONE, ZERO], [ZERO, (F(-1), F(0))]]
    gamma = [tensor(z, i2), tensor(x, x), tensor(x, y), tensor(x, z)]
    assert all(a == adjoint(a) for a in gamma)
    identity4 = [[ONE if i == j else ZERO for j in range(4)] for i in range(4)]
    assert all(multiply(a, a) == identity4 for a in gamma)
    defects = []
    for a_index in range(4):
        for b_index in range(a_index + 1, 4):
            a, b = gamma[a_index], gamma[b_index]
            ab, ba = multiply(a, b), multiply(b, a)
            anticommutator = [[add(ab[i][j], ba[i][j]) for j in range(4)] for i in range(4)]
            assert all(v == ZERO for row in anticommutator for v in row)
            defects.append(str(trace(multiply(anticommutator, anticommutator))[0]))
            assert -trace(multiply(ab, ab))[0] == trace(multiply(multiply(a, a), multiply(b, b)))[0] == 4
    commuting_price = -trace(multiply(multiply(gamma[1], gamma[1]), multiply(gamma[1], gamma[1])))[0]
    assert commuting_price == -4
    occupancy = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    patterns = [[pair] * 4 for pair in occupancy]
    patterns += [[occupancy[0], occupancy[5], occupancy[0], occupancy[5]],
                 [occupancy[1], occupancy[4], occupancy[1], occupancy[4]],
                 [(), (0, 1, 2, 3), (), (0, 1, 2, 3)],
                 [(0, 1, 2, 3), (), (0, 1, 2, 3), ()]]
    roles = []
    for role, count, mass in [('active', 1, cfg['active_m']), ('auxiliary', 63, cfg['auxiliary_m'])]:
        band = RotationBand(a=cfg['a'], m=mass, r=cfg['light_r'], v=cfg['light_v'], improved=True)
        _, paths = scalar_paths(band)
        matrices: dict[int, list[list[Z]]] = {}
        for displacement, coefficients in paths:
            if sum(abs(int(v)) for v in displacement) != 1:
                continue
            axis = next(i for i, v in enumerate(displacement) if int(v))
            c = [(F(float(value.real)), F(float(value.imag))) for value in coefficients]
            matrix = [[ZERO for _ in range(4)] for _ in range(4)]
            for i in range(4):
                for j in range(4):
                    for m in range(4):
                        matrix[i][j] = add(matrix[i][j], mul(c[m], gamma[m][i][j]))
            matrices[axis] = matrix
        rows = []
        for pattern in patterns:
            ref = sum(1 << (4 * site + spin) for site, spins in enumerate(pattern) for spin in spins)
            assert ref.bit_count() == 8
            result = loop(matrices[0], matrices[1], ref)
            rho = [64 * (len(spins) - 2) for spins in pattern]
            incoming = [rho[0], rho[0] + rho[1], sum(rho[:3]), 0]
            assert [incoming[e] - incoming[(e - 1) % 4] for e in range(4)] == rho
            rows.append({'occupation': [list(v) for v in pattern], 'coefficient_real': str(result[0]),
                         'coefficient_imag': str(result[1]), 'once_shared_all64_initial_cycle_flow': incoming})
        witness_ref = sum(1 << (4 * site + spin) for site in range(4) for spin in occupancy[0])
        signed = loop(matrices[0], matrices[1], witness_ref)
        unsigned = loop(matrices[0], matrices[1], witness_ref, True)
        assert signed != unsigned
        assert F(rows[0]['coefficient_real']) > 0 and rows[0]['coefficient_imag'] == '0'
        assert rows[2]['coefficient_real'] == rows[2]['coefficient_imag'] == '0'
        assert rows[3]['coefficient_real'] == rows[3]['coefficient_imag'] == '0'
        # All three occupations have exactly two particles/site and zero charge.
        # A positive spin trace is therefore not a strictly positive CAR-word
        # identity lower on this entire canonical sector.
        roles.append({'role': role, 'count': count, 'rows': rows,
                      'signed_reference_word': [str(v) for v in signed], 'unsigned_reference_word': [str(v) for v in unsigned],
                      'CAR_sign_discriminated': signed != unsigned})
    receipt = {'kind': 'EXACT_NATIVE_CAR_JORDAN_POINT_LOOP_OCCUPANCY_DISCRIMINATOR_NOT_GROUND_OR_STIFFNESS',
               'offdiagonal_Jordan_anticommutator_defects': defects, 'spin_anticommuting_price': '4',
               'spin_commuting_price': str(commuting_price), 'roles': roles, 'temporal_orders': 24,
               'local_mode_count': 16, 'local_particle_number': 8,
               'strict_Jordan_spin_and_number_only_CAR_lower_rejected': True,
               'zero_word_occupations_have_two_per_site_and_rho_zero': True,
               'observable': 'Diagonal FOURHOP bare native Fourier-word sum, NOT a charged resolvent/energy/pole coefficient',
               'Phi_point_not_actual_ground': True, 'complete_quantum_H_action_or_N_uniform_mechanism': False,
               'charged_resolvent_sign_or_magnetic_stiffness_inferred': False,
               'source_input_sha256_before': before,
               'source_input_sha256_after': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in request['pins']},
               'elapsed_seconds': time.monotonic() - started}
    assert receipt['source_input_sha256_after'] == before
    with Path(args.output).open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'role_rows': [len(row['rows']) for row in roles], 'ground_or_stiffness': False}))


if __name__ == '__main__':
    main()

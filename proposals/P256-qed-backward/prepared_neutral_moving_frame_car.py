#!/usr/bin/env python3
"""Full112 determinant SOURCE-relative controls on one cached point/local frame."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from prepared_neutral_slater_fibre import slater_expectation
from original_charged_principal import portable

SparseKet = dict[int, complex]


def source_transfer_matrix(source: dict[str, Any], n: int) -> np.ndarray:
    """Physical oneparticle B, BOTH literal maps; minus is not inferred from plus."""
    plus = np.asarray(source['full108_plus_coefficients'], complex)
    minus = np.asarray(source['full108_minus_coefficients'], complex)
    if plus.shape != (n, 4) or minus.shape != (4, n):
        raise ValueError('Full light/all4Ref SOURCE maps required')
    out = np.zeros((n + 4, n + 4), complex)
    out[:n, n:] = plus
    out[n:, :n] = minus
    return out


def apply_dgamma(matrix: np.ndarray, ket: SparseKet) -> SparseKet:
    """Exact c_i†c_j on sparse full-Fock determinants, right annihilator first.

    No small-entry dropping, state normalization, projection or norm floor.
    Diagonal terms are number operators; offdiagonal Pauli/sign rules agree
    with FullSlaterFibre._car_transfer, including the physical Ref mode order.
    """
    matrix = np.asarray(matrix, complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Square oneparticle matrix required')
    n = len(matrix)
    if any(mask < 0 or mask >> n for mask in ket):
        raise ValueError('Determinant outside oneparticle mode set')
    columns = [[(int(i), complex(matrix[i, j])) for i in np.flatnonzero(matrix[:, j]) if i != j]
               for j in range(n)]
    bits = [1 << j for j in range(n)]
    below = [bit - 1 for bit in bits]
    out: SparseKet = {}
    diagonal = np.diag(matrix)
    for mask, value in ket.items():
        if value == 0:
            continue
        occupied = mask
        trace = 0j
        while occupied:
            bit = occupied & -occupied
            j = bit.bit_length() - 1
            occupied ^= bit
            trace += diagonal[j]
            removed = mask ^ bit
            sign_j = (mask & below[j]).bit_count() & 1
            for i, coefficient in columns[j]:
                if removed & bits[i]:
                    continue
                target = removed | bits[i]
                phase = -1 if (sign_j + (removed & below[i]).bit_count()) & 1 else 1
                out[target] = out.get(target, 0j) + phase * coefficient * value
        if trace != 0:
            out[mask] = out.get(mask, 0j) + trace * value
    return out


def sparse_norm_squared(ket: SparseKet) -> float:
    return math.fsum(abs(value) ** 2 for value in ket.values())


def sparse_norm(ket: SparseKet) -> float:
    return math.sqrt(sparse_norm_squared(ket))


def sparse_difference(left: SparseKet, right: SparseKet) -> SparseKet:
    out = {mask: value - right.get(mask, 0j) for mask, value in left.items()}
    out.update((mask, -value) for mask, value in right.items() if mask not in left)
    return out


def _project(fibre: Any, ket: SparseKet) -> np.ndarray:
    return np.asarray([phase * ket.get(int(mask), 0j)
                       for mask, phase in zip(fibre.charged_masks, fibre.charged_car_phases)], complex)


def _outside_norm_squared(ket: SparseKet, represented: set[int]) -> float:
    return math.fsum(abs(value) ** 2 for mask, value in ket.items() if mask not in represented)


def car_source_control(fibre: Any, source: dict[str, Any], neutral_value: np.ndarray,
                       light_body: np.ndarray, reference_body: np.ndarray) -> dict[str, Any]:
    """Decision-ready exact fullCAR LEFT minus RIGHT, complex current supported.

    Inputs are actual point neutral coefficients and physical-coordinate bodies.
    Orbitals are the held/moved once seed, never a newly diagonalized sea.
    """
    neutral_value = np.asarray(neutral_value, complex)
    light_body = np.asarray(light_body, complex)
    reference_body = np.asarray(reference_body, complex)
    if neutral_value.shape != (2,) or light_body.shape != (fibre.n, fibre.n) or reference_body.shape != (4, 4):
        raise ValueError('Actual two-bank point, FULL light and Ref4 bodies required')
    n = fibre.n
    a = np.zeros((n + 4, n + 4), complex)
    a[:n, :n] = fibre.orbitals.conj().T @ light_body @ fibre.orbitals
    a[n:, n:] = reference_body
    physical_b = source_transfer_matrix(source, n)
    b = np.zeros_like(physical_b)
    b[:n, n:] = fibre.orbitals.conj().T @ physical_b[:n, n:]
    b[n:, :n] = physical_b[n:, :n] @ fibre.orbitals
    neutral = {int(mask): complex(value) for mask, value in zip(fibre.neutral_masks, neutral_value)}
    created = apply_dgamma(b, neutral)
    left = apply_dgamma(a, created)
    neutral_action = apply_dgamma(a, neutral)
    right = apply_dgamma(b, neutral_action)
    relative = sparse_difference(left, right)
    direct = apply_dgamma(a @ b - b @ a, neutral)
    represented = set(map(int, fibre.charged_masks))
    point_source = source['source_matrix'] @ neutral_value
    charged_weak = fibre.onebody(light_body) + fibre.reference(reference_body)
    neutral_weak = fibre.onebody(light_body, 'neutral') + fibre.reference(reference_body, 'neutral')
    weak_left = charged_weak @ point_source
    weak_right = source['source_matrix'] @ neutral_weak @ neutral_value
    full_left_projected, full_right_projected = _project(fibre, left), _project(fibre, right)
    full_relative_projected = _project(fibre, relative)
    second = fibre.bilinear_product(light_body.conj().T, light_body,
                                    reference_A=reference_body.conj().T, reference_B=reference_body)
    neutral_second = fibre.bilinear_product(light_body.conj().T, light_body, 'neutral',
                                            reference_body.conj().T, reference_body)
    left_norm = sparse_norm(left)
    right_norm = sparse_norm(right)
    direct_norm = sparse_norm(direct)
    source_norm = sparse_norm(created)
    disagreement = sparse_norm(sparse_difference(relative, direct))
    weak_right_missing_ph = full_right_projected - weak_right
    # Strict floating-point operation-scaled check; never changes input norms.
    scale = left_norm + right_norm + direct_norm
    error_relative = disagreement / scale if scale != 0 else disagreement
    source_error = float(np.linalg.norm(_project(fibre, created) - point_source))
    source_relative_error = source_error / source_norm if source_norm != 0 else source_error
    left_product = complex(np.vdot(point_source, second @ point_source))
    neutral_product = complex(np.vdot(neutral_value, neutral_second @ neutral_value))
    left_product_relative_error = abs(sparse_norm_squared(left) - left_product) / abs(left_product) if left_product != 0 else abs(left_product)
    neutral_product_relative_error = abs(sparse_norm_squared(neutral_action) - neutral_product) / abs(neutral_product) if neutral_product != 0 else abs(neutral_product)
    if max(error_relative, source_relative_error, left_product_relative_error, neutral_product_relative_error) > 2e-11:
        raise AssertionError('Full112 CAR/source/Wick consumer agreement failed')
    return dict(actual_modes=n+4, light_occupied_counts=[len(fibre.occupied_indices[q]) for q in (1,-1)],
                reference_occupied_counts=[4,0], neutral_masks_hex=[hex(m) for m in fibre.neutral_masks],
                point_neutral_coefficients=neutral_value, neutral_norm_squared=sparse_norm_squared(neutral),
                source_norm_squared=sparse_norm_squared(created), source_projection_error=source_error,
                source_projection_relative_error=source_relative_error,
                LEFT_norm=left_norm, RIGHT_norm=right_norm, relative_norm=sparse_norm(relative), direct_commutator_norm=direct_norm,
                fullCAR_commutator_error=disagreement, fullCAR_commutator_relative_error=error_relative,
                LEFT_PH_norm=math.sqrt(_outside_norm_squared(left, represented)),
                RIGHT_PH_norm=math.sqrt(_outside_norm_squared(right, represented)),
                relative_PH_norm=math.sqrt(_outside_norm_squared(relative, represented)),
                direct_PH_norm=math.sqrt(_outside_norm_squared(direct, represented)),
                full_neutral_action_norm=sparse_norm(neutral_action),
                neutral_PH_norm=math.sqrt(_outside_norm_squared(neutral_action, set(map(int,fibre.neutral_masks)))),
                LEFT_weak_projection_error=float(np.linalg.norm(full_left_projected-weak_left)),
                RIGHT_projected_neutral_only_missing_PH_norm=float(np.linalg.norm(weak_right_missing_ph)),
                weak_relative_vs_full_projected_error=float(np.linalg.norm(weak_left-weak_right-full_relative_projected)),
                LEFT_exact_Wick_relative_error=float(left_product_relative_error),
                neutral_exact_Wick_relative_error=float(neutral_product_relative_error),
                full_projected_relative=full_relative_projected,
                weak_projected_relative=weak_left-weak_right,
                wrong_returned_neutral_no_RIGHT_defect=right_norm,
                wrong_returned_neutral_rejected=right_norm != 0,
                PH_statement='absolute LEFT/RIGHT PH cancel in SOURCE-relative commutator; not the cause of historical Bnative1.83 defect',
                scope='actual full112 CAR on one point/local frozen frame; no full trajectory or wholefield bound')


def heavy_common_control(projector: np.ndarray, body: np.ndarray,
                         source_norm_squared: float) -> dict[str, Any]:
    """Full separate heavy Slater mean and positive A†A variance, not mean-only."""
    projector, body = np.asarray(projector, complex), np.asarray(body, complex)
    mean = slater_expectation(projector, body)
    variance = float(np.linalg.norm((np.eye(len(projector))-projector) @ body @ projector) ** 2)
    second = slater_expectation(projector, body.conj().T, body)
    return dict(full_heavy_modes=len(projector), occupied_count=float(np.trace(projector).real),
                mean=mean, variance=variance, second_moment=second,
                LEFT_RIGHT_common_norm_squared=source_norm_squared*(abs(mean)**2+variance),
                SOURCE_relative_contribution=0.,
                reason='Separate even heavy dGamma commutes literal light/Ref SOURCE; common cancellation, not SOURCE defect cause')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    from prepared_neutral_shared_photon_state import load_shared_native_cache
    from prepared_neutral_moving_matter_frame import construct_point_frame
    started = time.perf_counter()
    model, data = load_shared_native_cache(args.cache, args.manifest)
    frame = construct_point_frame(model, data)
    moved, sample = frame['model'], frame['sample']
    neutral_value = sample['neutral'] @ data['neutral_state']
    results = {}
    for profile in ('source19', 'SOFT'):
        source = moved.source_at(sample['point']['phi'], sample['point']['links'], sample['theta'], sample['a'], profile)
        results[profile] = {}
        for label, bodies in (('M', frame['generators']), ('A', frame['bodies'])):
            control = car_source_control(moved.fibre, source, neutral_value, bodies['light'], bodies['reference'])
            control['heavy_common'] = heavy_common_control(moved.heavy_projector, bodies['heavy'], control['source_norm_squared'])
            if not control['wrong_returned_neutral_rejected']:
                raise AssertionError('NoRIGHT branch is not discriminating on actual point')
            results[profile][label] = control
    from prepared_neutral_shared_photon_state import loaded_source_epoch
    result = dict(scope='point/local frozen frame, not trajectory/wholefield', controls=results,
                  no_preparation_H_or_G_recompilation=True, original_fH14_state_unchanged=True,
                  wall_seconds=time.perf_counter()-started,
                  command=[sys.executable, *sys.argv], cwd=os.getcwd(),
                  input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.cache,args.manifest)},
                  source_sha256=loaded_source_epoch(),
                  remaining_debt=['Root','Q','all512','Gauss','infiniteFock','wholefield','full physical trajectory'])
    print(json.dumps(portable(result), sort_keys=True))


if __name__ == '__main__':
    main()

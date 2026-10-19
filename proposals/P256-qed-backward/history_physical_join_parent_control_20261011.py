"""Native closed-flow factors and exact all-word operator-boundary controls."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import sympy as sp

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def closed_face(n: int) -> dict[str, Any]:
    sites = cast(list[tuple[int, int, int]], list(itertools.product(range(n), repeat=3)))
    index = {x: i for i, x in enumerate(sites)}
    origin = (1, 1, 1)

    def shift(x: tuple[int, int, int], axis: int) -> tuple[int, int, int]:
        y = list(x)
        y[axis] = (y[axis] + 1) % n
        return y[0], y[1], y[2]

    c: dict[int, int] = {}
    for x, axis, coefficient in ((origin, 0, 1), (shift(origin, 0), 1, 1),
                                  (shift(origin, 1), 0, -1), (origin, 1, -1)):
        edge = 3 * index[x] + axis
        c[edge] = c.get(edge, 0) + coefficient
    divergence = [0] * len(sites)
    for edge, coefficient in c.items():
        x, axis = sites[edge // 3], edge % 3
        divergence[index[x]] -= coefficient
        divergence[index[shift(x, axis)]] += coefficient
    assert len(c) == 4 and all(x == 0 for x in divergence)
    norm_squared = sum(x*x for x in c.values())
    p_squared_coefficient = Fraction(norm_squared, 16)
    assert norm_squared == 4 and p_squared_coefficient == Fraction(1, 4)
    # Edge-transitive torus: Tr(D^T Lplus D)=V-1, M=3V.
    resistance = Fraction(len(sites)-1, 3*len(sites))
    collective_density_phase = 64 * resistance
    wrong_one_copy_phase = resistance
    assert collective_density_phase != wrong_one_copy_phase
    electric_total = Fraction(64 * 64**2)
    electric_longitudinal = electric_total * resistance
    electric_transverse = electric_total * (1-resistance)
    assert electric_transverse + electric_longitudinal == electric_total
    return {'N': n, 'raw_edges': {str(k): v for k, v in c.items()},
            'D_face_exact_zero': True, 'face_squared_norm': norm_squared,
            'p_B_equals_face_dot_fullE_over_4': True,
            'p_B_Q_word_bound_coefficient': '1/(256*tau_face)',
            'closed_wavepacket_density_connection_exact_zero': True,
            'bare_edge_collective_1plus63_density_phase': str(collective_density_phase),
            'wrong_one_copy_phase': str(wrong_one_copy_phase),
            'all64_number2V_density_pattern_legal': True,
            'HE_transverse_exact': str(electric_transverse),
            'HC_collective_exact': str(electric_longitudinal),
            'HE_plus_HC_exact': str(electric_total),
            'integer_face_character_incoming_cross': '128*n*face_dot_fullE',
            'integer_face_character_self_energy': '256*n^2',
            'full_law_character_double_commutator': '512*n^2',
            'character_on_ground_is_eigenvector_assumed': False,
            'point_geometry_not_fullPhi_SU4_quantum_source': True}


def all_word_arc_budget() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for mass, momentum_squared in ((sp.Rational(1), sp.Rational(1)),
                                    (sp.Rational(9, 4), sp.Rational(8)),
                                    (sp.Rational(4), sp.Rational(25))):
        previous: Any = None
        for level in (2, 4, 8, 16):
            cells = 2**level
            delta = sp.Rational(1, 2**(4*level))
            r = 1-sp.Rational(1, 2**(6*level))
            arc: Any = mass/cells + sp.sqrt(mass*momentum_squared) * 2*delta*(cells-1)
            error: Any = sp.simplify(64*mass*(1-r)**2/delta**2 + 16*arc)
            assert error > 0 and r < 1
            if previous is not None:
                assert sp.simplify(previous-error) > 0
            previous = error
            rows.append({'word_norm_squared': str(mass),
                         'word_p_B_squared': str(momentum_squared),
                         'level': level, 'r_exact': str(r),
                         'arc_mass_upper_exact': str(arc),
                         'normalized_principal_error_squared_upper_exact': str(error)})
    return {'rows': rows, 'same_Abel_sequence_for_every_fixed_word': True,
            'first_sine_harmonic_normalization': '1',
            'q_dependent_source_weakening': False,
            'native_word_moment_premise_not_computed_here': True,
            'vacuum_bound_not_substituted_for_word_bound': True,
            'uniform_operator_bound_and_dense_word_convergence_required': True}


def selected_feature_countercontrols() -> dict[str, Any]:
    omega: Any = sp.Matrix([1, 0])
    word: Any = sp.Matrix([[0, 1], [1, 0]])
    # Finite METHOD: vacuum sees no seam, but a history word reaches it.
    principal: Any = sp.diag(sp.pi/4, -sp.pi/2)
    abel_limit: Any = sp.diag(sp.pi/4, 0)
    assert (abel_limit-principal)*omega == sp.zeros(2, 1)
    seam_error: Any = (abel_limit-principal)*word*omega
    assert sp.simplify((seam_error.H*seam_error)[0]) == sp.pi**2/4
    assert sp.Matrix.hstack(omega, word*omega).rank() == 2
    # A zero vacuum electric moment cannot price the same boost on another word.
    electric: Any = sp.diag(0, 5)
    assert (omega.H*electric**2*omega)[0] == 0
    assert (omega.H*word.H*electric**2*word*omega)[0] == 25
    boost: Any = sp.diag(1, -1)  # s=pi/5.
    difference: Any = (boost-sp.eye(2))*word*omega
    assert (difference.H*difference)[0] == 4
    return {'method_only_not_native_ground': True,
            'vacuum_principal_error_exact': '0',
            'history_word_principal_error_squared_exact': 'pi^2/4',
            'vacuum_E_squared_exact': '0', 'word_E_squared_exact': '25',
            'boost_word_difference_squared_at_pi_over_5': '4',
            'wrong_vacuum_only_operator_upgrade_rejected': True,
            'wrong_vacuum_only_word_regular_budget_rejected': True,
            'small_selected_feature_not_full_history_operator': True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request: dict[str, Any] = json.loads(args.request.read_text())
    before = {n: digest(ROOT/n) for n in request['pins']}
    assert before == request['pins']
    history: dict[str, Any] = json.loads((ROOT/request['history_receipt']).read_text())
    assert history['native_exact_constants']['loads'] == [1, 1, 63]
    receipt = {'kind': 'NATIVE_CLOSED_FLOW_AND_ALL_WORD_OPERATOR_BOUNDARY_METHOD_NOT_PHASE',
               'native_closed_flow': [closed_face(n) for n in request['N']],
               'all_word_arc': all_word_arc_budget(),
               'selected_feature_countercontrols': selected_feature_countercontrols(),
               'whole_P0_after_common_history_L_required': True,
               'same_static_GNS_identification': False,
               'full_history_operator_joins_independent_proof_pending': True,
               'actual_ground_moments_computed': False,
               'physical1_phase_stiffness_IR_atom_photon': False,
               'campaign_complete': False,
               'source_input_sha256_before': before,
               'source_input_sha256_after': {n: digest(ROOT/n) for n in before},
               'runtime': {'elapsed_seconds': time.monotonic()-started}}
    assert receipt['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()

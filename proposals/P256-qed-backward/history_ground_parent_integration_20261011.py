"""Integrate frozen history/exterior epochs and independent scoped corrections."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, cast
import zipfile

import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def epoch(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    for key in ('pre', 'executed', 'receipt', 'capsule'):
        assert digest(ROOT / spec[key]) == spec[key + '_sha256'], (key, spec[key])
    cap: dict[str, Any] = json.loads((ROOT / spec['capsule']).read_text())
    run = cap[spec['run_metadata_key']] if spec['run_metadata_key'] is not None else cap
    assert run['natural_exit'] is True and run['returncode'] == 0
    with zipfile.ZipFile(ROOT / spec['pre']) as pre, zipfile.ZipFile(ROOT / spec['executed']) as exe:
        names = pre.namelist()
        assert len(names) == len(set(names)) and set(names) == set(spec['member_map'])
        for name in names:
            assert pre.read(name) == exe.read(spec['member_map'][name]), name
        raw = exe.read(spec['receipt_member'])
        assert hashlib.sha256(raw).hexdigest() == spec['receipt_sha256']
        assert raw == (ROOT / spec['receipt']).read_bytes()
        payload: dict[str, Any] = json.loads(raw)
    stable = spec['stability']
    if stable['mode'] == 'capsule_flag':
        stable_owner = cap[stable['owner_key']] if stable['owner_key'] is not None else cap
        assert stable_owner[stable['flag']] is True
    else:
        assert stable['mode'] == 'receipt_hash_maps'
        assert payload[stable['before']] == payload[stable['after']]
    return payload, {'members_compared': len(names), 'natural_exit': True,
                     'returncode': 0, 'unchanged_source_inputs_verified': True,
                     'elapsed_seconds': run.get('elapsed_seconds')}


def orbit_metric_from_executed_control(payload: dict[str, Any]) -> dict[str, Any]:
    gram = cast(sp.MutableDenseMatrix, sp.Matrix(
        [[sp.sympify(x) for x in row] for row in payload['trial_colour_orbit']['orbit_Gram_exact']]))
    assert gram.H == gram
    x = sp.Symbol('x')
    polynomial = gram.charpoly('x').as_expr()
    factorization: Any = sp.factor_list(polynomial, x)
    coefficient, factors = factorization
    assert coefficient == 1
    roots: list[tuple[Any, int]] = []
    reconstructed: Any = sp.Integer(1)
    for expression, multiplicity in factors:
        p = sp.Poly(expression, x)
        assert p.degree() == 1
        root = -p.nth(0) / p.nth(1)
        assert root.is_Rational and root >= 2
        roots.append((root, multiplicity))
        reconstructed *= (x - root) ** multiplicity
    assert sp.expand(polynomial - reconstructed) == 0
    assert sum(n for _, n in roots) == 15
    assert payload['Gauss_reset']['covariance_only_reset_Gauss_square_exact'] == '15/4'
    assert payload['complete_H_channel_or_domain_proved'] is False
    return {'native_HS_colour_orbit_metric_lower_exact': '2',
            'exact_characteristic_roots': [{'root': str(r), 'multiplicity': n} for r, n in roots],
            'method_carrier_channel_not_fullPhi_channel': True,
            'ground_high_moment_or_bootstrap_admitted': False}


def corrected_boundaries(request: dict[str, Any]) -> dict[str, Any]:
    word: dict[str, Any] = json.loads((ROOT / request['word_proof']).read_text())
    limit: dict[str, Any] = json.loads((ROOT / request['limit_proof']).read_text())
    static: dict[str, Any] = json.loads((ROOT / request['static_proof']).read_text())
    inherited: dict[str, Any] = json.loads((ROOT / request['inherited_family_addendum']).read_text())
    assert inherited['frozen_target_or_review_rebound'] is False
    assert inherited['original_review_proof_sha256'] == digest(ROOT / request['limit_proof'])
    assert [f['id'] for f in word['findings']] == ['PACKET_INITIAL_PROPAGATOR']
    limit_ids = {f['id'] for f in limit['findings']}
    assert {'HL1', 'HL4'} <= limit_ids
    assert [f['id'] for f in static['findings']] == ['TAIL_THRESHOLD_HALF_RADIUS']
    # Corrected target-radius Gamma tail, not the original half-radius number.
    gamma_sum = sum((sp.Rational(30) ** j / sp.factorial(j) for j in range(32)), sp.Integer(0))
    limit_constant = sp.Rational(31, 480) * sp.exp(-30) * gamma_sum
    assert gamma_sum > 0
    value = float(sp.N(limit_constant, 25))
    assert 0.0399 < value < 0.04
    return {'packet': {'exact_ground_unchanged': True,
                       'factorization': 'U(-tm) Am U(tm-tm-1) ... A1 U(t1) Omega',
                       'first_packet_Q': 'exp(Cstar*abs(t1))*B0', 'initial_graph': 'eta_N',
                       'small_first_energy_not_substitute': True},
            'spectral_endpoints': {'open_increasing_approximants_then_complements_intersections': True,
                                   'singletons_use_decreasing_closed_neighborhoods': True,
                                   'whole_P0_heat_limit_retained': True},
            'static_tail': {'r_definition': 'sqrt(s)=abs(q)/sqrt(2)',
                            'original_bound_radius': 'M/2', 'corrected_event': 's_x>M^2; s_y<=M^2/4',
                            'corrected_positive_limit_div_alpha2_exact': str(limit_constant),
                            'numeric_display_not_interval': value,
                            'nonstationary_not_actual_ground_failure': True},
            'inherited_family': {'analytic_existence_selection_B0_inherited_conditionally': True,
                                 'numerical_ground_computation_required_for_existence': False,
                                 'new_independently_unpaid_family_gate_withdrawn': True,
                                 'ground_characterization_not_supplied': True},
            'frozen_originals_not_rebound': True,
            'same_static_GNS_invariance_onto_range': False}

def typed_transfers(request: dict[str, Any],
                    payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    transfers: list[dict[str, Any]] = []
    for spec in request['typed_transfers']:
        omitted = set(spec['metadata_only_fields'])
        original = {k: v for k, v in payloads[spec['original_epoch']].items() if k not in omitted}
        successor = {k: v for k, v in payloads[spec['typed_epoch']].items() if k not in omitted}
        assert original == successor, spec['typed_epoch']
        transfers.append({**spec, 'mathematical_payload_exact_equal': True,
                          'original_targets_and_reviews_not_rebound': True})
    return transfers



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request: dict[str, Any] = json.loads(args.request.read_text())
    before = {n: digest(ROOT / n) for n in request['pins']}
    assert before == request['pins']
    payloads: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for name, spec in request['epochs'].items():
        payloads[name], bindings[name] = epoch(spec)
    assert payloads['history']['native_exact_constants']['loads'] == [1, 1, 63]
    assert payloads['exterior']['same_static_GNS_exterior_closed'] is False
    receipt = {'kind': 'INDEPENDENT_FROZEN_FULL_HISTORY_EXTERIOR_CORRECTION_INTEGRATION_NOT_PHASE',
               'epochs': bindings, 'total_members_compared': sum(v['members_compared'] for v in bindings.values()),
               'scope_corrections': corrected_boundaries(request),
               'native_colour_metric': orbit_metric_from_executed_control(payloads['parent_reset']),
               'typed_transfers': typed_transfers(request, payloads),
               'positive_history_generator_conditional_proof_reviewed': True,
               'native_fullH_numerical_evolution_computed': False,
               'same_static_GNS_exterior_range': False, 'actual_ground_identification': False,
               'physical1_phase_stiffness_IR_atom': False, 'original_SOURCE_adopted': False,
               'campaign_complete': False, 'source_input_sha256_before': before,
               'source_input_sha256_after': {n: digest(ROOT / n) for n in before},
               'runtime': {'elapsed_seconds': time.monotonic() - started}}
    assert receipt['source_input_sha256_before'] == receipt['source_input_sha256_after']
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()

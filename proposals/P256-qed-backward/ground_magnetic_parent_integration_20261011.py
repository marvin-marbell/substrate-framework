"""Consume frozen word/influence epochs; independently compose their form boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction as F
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
    cap = json.loads((ROOT / spec['capsule']).read_text())
    run_key = spec['run_metadata_key']
    run = cap[run_key] if run_key is not None else cap
    assert isinstance(run, dict)
    natural = run.get('natural_exit', run.get('natural_completed_exit'))
    stable = run.get('sources_inputs_unchanged', run.get('source_inputs_unchanged',
                     run.get('source_input_unchanged')))
    assert natural is True and run['returncode'] == 0 and stable is True
    with zipfile.ZipFile(ROOT / spec['pre']) as pre, zipfile.ZipFile(ROOT / spec['executed']) as exe:
        names = [n for n in pre.namelist() if not n.endswith('/')]
        assert len(names) == len(set(names))
        assert set(names) == set(spec['member_map'])
        for name in names:
            assert pre.read(name) == exe.read(spec['member_map'][name]), name
        raw = exe.read(spec['receipt_member'])
        assert hashlib.sha256(raw).hexdigest() == spec['receipt_sha256']
        assert raw == (ROOT / spec['receipt']).read_bytes()
        payload = json.loads(raw)
    return payload, {'members_compared': len(names), 'natural_exit': True,
                     'returncode': 0, 'source_input_unchanged': True,
                     'elapsed_seconds': run.get('elapsed_seconds')}


def conditional_heat_composition(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Same-background four-heat ACTION, not an interacting Gibbs expectation."""
    a, b, c, p, q, beta, initial = sp.symbols('a b c p q beta initial', real=True)
    # Per-interval longitudinal action and one shared-character transverse variance.
    longitudinal_cross = 128 * beta * c * p
    transverse_cross = 128 * beta * (c - a * b) * q
    action_cross = sp.expand(longitudinal_cross + transverse_cross)
    disjoint = sp.expand(action_cross.subs(q, -p))
    assert sp.expand(disjoint - 128 * beta * a * b * p) == 0
    assert sp.diff(disjoint, c) == 0
    # Common-background baseline and linear terms cancel only in the FOUR-heat ratio.
    background, linear_a, linear_b = sp.symbols('background linear_a linear_b')
    joint = background + linear_a + linear_b + action_cross
    single_a, single_b = background + linear_a, background + linear_b
    assert sp.expand(joint + background - single_a - single_b) == action_cross
    assert sp.expand(joint - single_a - single_b - action_cross) == -background
    rows = []
    for native in payload['native_forms']:
        n = native['N']
        pl, qt = F(native['longitudinal_cross_exact']), F(native['transverse_cross_exact'])
        assert pl == -F(1, n - 1) and qt == -pl
        half = 128 * F(1, 2) * F(1, 2) * pl
        quarter = 128 * F(1, 4) * F(1, 2) * pl
        assert half == F(native['joint_minus_independent_exponent_per_beta'])
        assert quarter == half / 2 and quarter != half
        assert native['boundary_plane_is_tangential_magnetic_source'] is False
        assert native['full_quantum_word_coefficient_integrated'] is False
        rows.append({'N': n, 'half_time_action_per_beta': str(half),
                     'quarter_half_action_per_beta': str(quarter),
                     'overlap_changes_split_not_total_for_disjoint_raw_supports': True,
                     'missing_common_background_rejected': True,
                     'actual_Gibbs_or_magnetic_influence': False})
    return rows


def spectral_cutoff_method() -> dict[str, Any]:
    """Exact noncommuting bounded method control, not native evolution realization."""
    h = cast(sp.MutableDenseMatrix, sp.Matrix([[2, 1, 0], [1, 2, 1], [0, 1, 2]]))
    q = cast(sp.MutableDenseMatrix, sp.diag(1, 2, 4))
    current = sp.I * (h * q - q * h)
    invroot = cast(sp.MutableDenseMatrix, sp.diag(1, 1 / sp.sqrt(2), sp.Rational(1, 2)))
    normalized = invroot * current * invroot
    x = sp.Symbol('x', real=True)
    polynomial = normalized.charpoly(x)
    assert polynomial.all_coeffs() == [1, 0, -1, 0]
    # Thus |i[H,Q]| in Q form has sharp constant1, not an ordinary operator-norm bound.
    cutoffs = [sp.diag(1, 0, 0), sp.diag(1, 1, 0), sp.eye(3)]
    for s in cutoffs:
        hr = s * h * s
        assert hr * q - q * hr == s * (h * q - q * h) * s
    # A rotated projection is not a Q spectral cutoff; the compressed current identity fails.
    rotated = cast(sp.MutableDenseMatrix, sp.Matrix([[sp.Rational(1, 2), sp.Rational(1, 2), 0],
                                                    [sp.Rational(1, 2), sp.Rational(1, 2), 0],
                                                    [0, 0, 1]]))
    wrong_h = rotated * h * rotated
    wrong = wrong_h * q - q * wrong_h - rotated * (h * q - q * h) * rotated
    assert wrong != sp.zeros(3)
    assert cutoffs[-1] * h * cutoffs[-1] == h
    # Volume-independent passage needs a current FORM bound, not mere positivity of H,Q.
    rate = sp.Symbol('rate', positive=True)
    scaled_current = sp.I * (rate * h * q - q * rate * h)
    assert scaled_current == rate * current
    return {'scope': 'EXACT_FINITE_NONCOMMUTING_METHOD_NOT_NATIVE_FIBRE_EVOLUTION',
            'Q_form_current_characteristic': str(polynomial.as_expr()),
            'sharp_form_constant': 1, 'spectral_compression_identity': True,
            'arbitrary_rotated_cutoff_rejected': True,
            'positivity_without_uniform_current_bound_rejected': True,
            'actual_native_Friedrichs_evolution_extension': False}


def current_recomposition(request: dict[str, Any], audited: dict[str, Any],
                          propagation: dict[str, Any]) -> dict[str, Any]:
    """Reprice the reviewed native current from immutable profiles, not copied C."""
    with zipfile.ZipFile(ROOT / request['profile_archive']) as archive:
        profiles = json.loads(archive.read(request['profile_member']))['primitive_profiles']
    lam, chi, fast = F(1 << 8192), F(1, 1 << 4096), F(1 << 10240)
    weights = [abs(F(p['onsite'])) + 2 * sum(
        (abs(F(row['c0'])) + abs(F(row['ci'])) for row in p['paths']), F(0)) for p in profiles]
    assert [p['load'] for p in profiles] == [1, 1, 63]
    b = 4 * (lam * weights[0] + weights[1] + 63 * weights[2])
    assert b < 512 * lam
    kphi, kh, ke = 4 * chi, 3 * chi / 20, F(64)
    alignment, plaquette, sphere = fast / 40, 7 * fast / 10, F(4096)
    z = sphere + 2 / lam
    terms = [F(54), 480 * kphi * b * b * sphere,
             184320 * kphi * alignment * alignment * z, 480 * kh * b * b * z,
             32 * ke * b * b * z, 32 * b * b * z,
             46080 * kh * alignment * alignment / lam,
             2880 * kh * plaquette * plaquette * sphere]
    c = sum(terms, F(0))
    assert c < F(1 << 16448)
    constant = audited['constant_control']
    assert c == F(constant['explicit_C'])
    rational = str(c.numerator) + '/' + str(c.denominator)
    pin = hashlib.sha256(rational.encode()).hexdigest()
    assert pin == constant['explicit_C_sha256']
    cstar = (1 << 32) * 65 * (1 + ((1 << 16) * (1 + fast + b)) ** 2) * z
    certificate = propagation['exact_constants']
    for name, value in [('B', b), ('C_index', c), ('Cstar', cstar)]:
        assert certificate[name] == {
            'sha256': hashlib.sha256(str(value).encode()).hexdigest(),
            'numerator_bits': value.numerator.bit_length(),
            'denominator_bits': value.denominator.bit_length()}
    assert all(term <= cstar / 8 for term in terms)
    # Diagonal core current and actual evolution are different transitions.
    assert audited['actual_quantum_evolution_computed'] is False
    assert audited['closed_form_evolution_extension_verified_by_this_audit'] is False
    return {'explicit_C_sha256': pin, 'C_upper': '2^16448',
            'Cstar_sha256': certificate['Cstar']['sha256'],
            'profiles_active_aux_roles': [1, 63], 'terms_recomputed': len(terms),
            'all_N_character_independent_core_form_claim': True,
            'termwise_C_index_le_Cstar_over_8': True,
            'plaquette_coefficient_is_not_inverse_temperature': True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    pins = request['pins']
    assert all(digest(ROOT / name) == pin for name, pin in pins.items())
    payloads, evidence = {}, {}
    for name, spec in request['epochs'].items():
        payloads[name], evidence[name] = epoch(spec)
    word = payloads['word']
    connected = payloads['connected']
    assert word['flags']['full_core_mixed_word_excitation_transport_all_N'] is False
    assert word['flags']['weighted_local_closed_form_evolution_extension'] is False
    assert word['flags']['full_GNS_dynamics_achieved'] is False
    assert connected['full_normalized_magnetic_epsilon_N_achieved'] is False
    assert connected['native_numeric_Gibbs_expectation'] is False
    for row in connected['native_tails_and_failed_crude_gates']:
        assert F(row['crude_local_influence_cap_fourth_over_beta4_Cbeta']) == 2 * F(connected['CJ']) ** 4 * row['N'] ** 9
        assert row['fixed_beta_epsilon_gate_pass'] is False
    addition = json.loads((ROOT / request['terminology_addendum']).read_text())
    assert addition['original_epochs_rebound'] is False
    heat = conditional_heat_composition(connected)
    method = spectral_cutoff_method()
    current = current_recomposition(request, payloads['weighted_review'], payloads['evolution'])
    transfers = []
    for row in request['typed_transfers']:
        original = json.loads((ROOT / row['original']).read_text())
        typed = json.loads((ROOT / row['typed']).read_text())
        for key in row['mathematical_fields']:
            assert original[key] == typed[key], key
        transfers.append({'original': row['original'], 'typed': row['typed'], 'exact_equal': True})
    assert all(digest(ROOT / name) == pin for name, pin in pins.items())
    result = {'kind': 'IMMUTABLE_NATIVE_WORD_MAGNETIC_INTEGRATION_WITH_EXACT_FORM_METHOD_NOT_PHASE',
              'epochs': evidence, 'preexecution_members_compared': sum(r['members_compared'] for r in evidence.values()),
              'native_conditional_four_heat_composition': heat, 'weighted_cutoff_method': method,
              'native_current_recomposition': current, 'typed_mathematical_transfers': transfers,
              'restricted_word_review': 'VALID_CONDITIONAL_WITH_GRAPH_PACKET_QUALIFICATION_NOT_STATIC_GNS',
              'analytic_finite_native_Q_form_propagation_independently_reviewed': True,
              'native_fullH_numerical_evolution_computed': False,
              'uniform_initial_budget_scope': 'TRANSLATION_AVERAGED_GLOBAL_MIN_FAMILY_OR_SEPARATE_LOCAL_MOMENTS',
              'same_static_GNS_exterior_range': False,
              'uniform_graph_core_constants_asserted': False,
              'full_core_actual_evolution_unpaid_in_original': True,
              'native_magnetic_Gibbs_influence_unpaid': True,
              'wholeP0_state_spectral_convergence': False,
              'original_reviews_frozen_not_rebound': True,
              'fullphysical_ground_phase_stiffness_photon_admitted': False,
              'original_SOURCE_adopted': False, 'campaign_complete': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'members': result['preexecution_members_compared'],
                      'method_uniform_current_constant': 1, 'analytic_native_propagation': True,
                      'native_numerical_evolution_or_phase': False}))


if __name__ == '__main__':
    main()

"""NEW weighted-budget consumer of the completed uniform90 coefficient epoch.

No reconstruction of its expensive matrix function or physical Kato point.
Historical point/coefficients retain their original source and execution capsule.
"""
from __future__ import annotations
import argparse
import ast
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)
PRIMARY = 'induced_phi_car_superadiabatic_transfer_corrected_actual_receipt.json'
BASE = 'induced_phi_full_uniform_colour_fermionic_source_unitary_final_actual_receipt.json'
MAIN = 'induced_phi_car_superadiabatic_transfer.py'
PINNED = {PRIMARY: '480fd3f4c6b6ebd4aba9e163134fef590cb61427a629e50d354399411ddec4d1', BASE: 'e44ea9a373b7d4397ac5357845d5edaf5b5d561aa116d769dbfdb9af93bfdc75'}
UNCHANGED_FUNCTIONS = ('norm', 'upper', 'exact_field_pencil', 'positive_mixed_root', 'resolve_transition', 'all_q_counterrotations', 'point_coefficients', 'uniform_transitions', 'literal_differential_and_CAR', 'literal_rotated_source')

def upper(record: dict[str, Any]) -> F:
    return F(record['upper'])

def encoded(value: F) -> dict[str, str]:
    return {'upper': str(value), 'encoding': 'exact rational upper from frozen declared bounds'}

def functions(raw: bytes) -> dict[str, str]:
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}

def run() -> dict[str, Any]:
    start = time.monotonic()
    raws = {name: (ROOT/name).read_bytes() for name in PINNED}
    assert all(hashlib.sha256(raw).hexdigest() == PINNED[name] for name,raw in raws.items())
    primary, base = [json.loads(raw) for raw in raws.values()]
    assert all(primary['source_unchanged_since_import'].values()) and all(primary['input_unchanged'].values())
    historical = primary['source_at_import'][MAIN]
    old_bytes = historical['bytes_utf8'].encode()
    assert hashlib.sha256(old_bytes).hexdigest() == historical['sha256'] == '0b963c8b3e32753df491e9dd1c17391d078886ac42b9c2a142dd0cda6813cc1a'
    current_bytes = (ROOT/MAIN).read_bytes()
    old_functions, current_functions = functions(old_bytes), functions(current_bytes)
    assert all(old_functions[name] == current_functions[name] for name in UNCHANGED_FUNCTIONS)
    proof = base['actual_full105_corrected_joint90_quantum_CAR_source_proof']
    cmax = upper(proof['covariance_operator_upper'])
    cmin = F(proof['covariance_operator_lower']['lower'])
    p4 = upper(proof['corrector_L4_norm_upper'])
    assert cmax >= cmin > 0 and p4 >= 0
    field = primary['actual_mixed90_positive_field_function']
    root_error = upper(field['root_operator_error_upper'])
    omega_lower = F(field['positive_frequency_lower_divided_2to3072'])/4
    assert 0 <= root_error < omega_lower
    coefficient_margins = []
    for species in primary['actual_all65_gap_loading_and_costs']:
        delta_lower = F(species['physical_gap_lower'])/(2*F(2)**3072)
        # 12 > 2 sqrt(80/3); doubling the archived k sum and halving
        # the gap leave ample space for its declared MP960 rounding.
        uncertainty = 128*24*root_error*F(species['all64_k_Frobenius_sum'])/(delta_lower+omega_lower)
        nominal = F(species['finite_CAR_generator_Lipschitz_upper'])
        relative = uncertainty/nominal
        assert 0 <= relative < F(10)**-62
        coefficient_margins.append({'role': species['role'], 'generator_uncertainty_upper_exact': str(uncertainty), 'relative_uncertainty_upper_exact': str(relative)})
    d = 90
    first_ratio, second_ratio = F(101,100), F(103,100)
    # Exact integer inequalities enclose the L4/L2 radial moment ratios.
    assert first_ratio**4 > F(d+2,d)
    assert second_ratio**4 > F((d+4)*(d+6),d*(d+2))
    # ||A||<=1/(2 Cmin); ||rho^2||_L4 <= (d+6) Cmax.
    score_weight = F(d+6,2)*cmax/cmin
    old_budget = primary['actual_new_prepared_state_budget']
    budget = dict(old_budget)
    terms = {name: upper(value) for name,value in old_budget['remainder_budget'].items()}
    terms['radial_Kato_connection_variation'] = terms['anti_H_connection_second_derivatives']*(1+p4)*score_weight
    terms['actual_hPhiC_gap_variation_T'] *= second_ratio
    terms['actual_compact_bump_derivative_cross'] *= first_ratio
    terms['exact_positive_matrix_function_coefficient_enclosure'] *= first_ratio
    residual = sum(terms.values(), F(0))
    assert residual < F(2)**-400
    budget['remainder_budget'] = {name: encoded(value) for name,value in terms.items()}
    budget['actual_new_state_reference_residual_upper'] = encoded(residual)
    budget['actual_new_state_Rayleigh_residual_upper'] = encoded(2*residual)
    budget['anti_H_generator_on_compact_packet_L2_upper'] = encoded(first_ratio*upper(old_budget['anti_H_generator_on_compact_packet_L2_upper']))
    budget['physical_source_endpoint_change_upper'] = encoded(first_ratio*upper(old_budget['physical_source_endpoint_change_upper']))
    witness = F(2**20+F(1,2**20),2)
    assert witness > 3*d
    captures = {MAIN: current_bytes, Path(__file__).name: Path(__file__).read_bytes()}
    assert all((ROOT/name).read_bytes() == raw for name,raw in {**raws,**captures}.items())
    return {
        'kind': 'NEW_UNIFORM90_GAUSSIAN_WEIGHT_REPRICE_FROM_IMMUTABLE_COEFFICIENT_EPOCH',
        'historical_primary': {'path': PRIMARY, 'sha256': PINNED[PRIMARY], 'coefficient_point_source_sha256': historical['sha256'], 'expensive_primary_science_not_rerun': True, 'original_natural0_not_backfilled': True},
        'underlying_state_and_H_unchanged': True,
        'unchanged_physical_coefficient_and_point_function_ASTs': list(UNCHANGED_FUNCTIONS),
        'actual_new_prepared_state_budget': budget,
        'Gaussian_weight_proof': {
            'first_radial': '||rho(1-P) f||2 <= sqrt(d Cmax)+[d(d+2)]^(1/4) sqrt(Cmax) P4; its ratio to the previous majorant is <=101/100, proven above.',
            'second_radial': '||rho^2(1-P) f||2 <= sqrt(d(d+2)) Cmax+[d(d+2)(d+4)(d+6)]^(1/4) Cmax P4; its ratio to the previous majorant is <=103/100.',
            'connection': 'Sdag K(eta) S+Sdag DS first-connection variation is bounded per coordinate by sqrt(d)[C2+(C1+Ct)^2+Ct^2] rho. Summing against G A eta yields d maxG L ||A|| ||rho^2(1-P)f||2; the completed second-connection upper already contains normalization*d*maxG*L. C=.5 A^-1 gives ||A||<=1/(2 Cmin). All W variation, S conjugation and exponential differential are included.',
            'score_weight_exact_upper': str(score_weight),
            'covariance_ratio_exact_upper': str(cmax/cmin),
            'bound_inputs': 'Historical derived MP960 coefficient/operator bounds, input-width and arithmetic floors remain declared conditional inputs. New rational weighting arithmetic is exact; this is not an independent interval rederivation of every old coefficient or a measured residual.',
            'coefficient_norm_rounding_margin': {'all65_species': coefficient_margins, 'absorbed_by_existing_upper_inflation_exact': '1/10^60', 'maximum_allowed_relative_uncertainty_exact': '1/10^62', 'scope': 'Conditional frozen coefficient enclosures; proves the separately priced Omega uncertainty cannot invalidate generator or squared-generator upper margins. No new physical coefficient evaluation.'},
        },
        'abstract_control': {'exact_radial_Kato_Gaussian_score_variation_norm': str(witness), 'rejected_condition_free_d90_L3': str(3*d), 'not_actual_packet_norm_violation': True},
        'input_sha256': PINNED,
        'source_at_import': {name: {'path': str(ROOT/name), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes_utf8': raw.decode()} for name,raw in captures.items()},
        'source_unchanged_since_import': {name: (ROOT/name).read_bytes() == raw for name,raw in captures.items()},
        'input_unchanged': {name: (ROOT/name).read_bytes() == raw for name,raw in raws.items()},
        'runtime': {'executable': sys.executable, 'python': sys.version},
        'scope': 'Distinct e44/8bfe uniform105/90, no Pi/HC/full5760/EM/originalSOURCE/ground/pole/cone promotion. Same-H source endpoint stability must consume these NEW corrected prices.',
        'elapsed_seconds': time.monotonic()-start,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'kind': result['kind'], 'unchanged_physical_functions': result['unchanged_physical_coefficient_and_point_function_ASTs'], 'elapsed_seconds': result['elapsed_seconds']}))

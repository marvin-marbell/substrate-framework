"""Exercise corrected packet_budget from immutable coefficient captures.

No root solver, positive matrix-function iteration, transition coefficients,
physical Kato point or completed source consumer is rerun.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import induced_phi_car_superadiabatic_transfer as paired

ROOT = Path(__file__).resolve().parent
PRIMARY = 'induced_phi_car_superadiabatic_transfer_corrected_actual_receipt.json'
REPRICE = 'induced_phi_car_superadiabatic_weighted_reprice_verified_actual_receipt_20261010.json'
PINS = {
    PRIMARY: '480fd3f4c6b6ebd4aba9e163134fef590cb61427a629e50d354399411ddec4d1',
    'induced_phi_uniform_relative_colour_quantum_final_actual_receipt.json': 'cda59b31daa41a30fd676ca2422155f80aea2a20c213a592c515193756c954b8',
    'induced_phi_full_uniform_colour_fermionic_source_unitary_final_actual_receipt.json': 'e44ea9a373b7d4397ac5357845d5edaf5b5d561aa116d769dbfdb9af93bfdc75',
}


def run(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    sys.set_int_max_str_digits(0)
    captured = {**paired.SOURCE_BYTES, Path(__file__).name: Path(__file__).read_bytes()}

    def take(name: str, pin: str | None = None) -> Any:
        raw = (ROOT/name).read_bytes()
        if pin is not None and hashlib.sha256(raw).hexdigest() != pin:
            raise ArithmeticError('Frozen cached-budget input mismatch: '+name)
        captured[name] = raw
        return json.loads(raw)

    primary = take(PRIMARY, PINS[PRIMARY])
    supplier = take('induced_phi_uniform_relative_colour_quantum_final_actual_receipt.json', PINS['induced_phi_uniform_relative_colour_quantum_final_actual_receipt.json'])
    base = take('induced_phi_full_uniform_colour_fermionic_source_unitary_final_actual_receipt.json', PINS['induced_phi_full_uniform_colour_fermionic_source_unitary_final_actual_receipt.json'])
    full_name = 'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json'
    full = take(full_name, primary['input_descriptor']['frozen_inputs'][full_name])
    reprice = take(REPRICE)
    if reprice['source_at_import']['induced_phi_car_superadiabatic_transfer.py']['sha256'] != hashlib.sha256(captured['induced_phi_car_superadiabatic_transfer.py']).hexdigest():
        raise ArithmeticError('Cached reprice binds a different corrected producer')
    paired.mp.mp.dps = 960
    paired.iv.dps = 1100
    root = supplier['actual_NEW_colour_BO_feedback_mean']
    _, geom = paired.joined.literal_geometry(root, full)
    g0, gs = paired.frozen.interval(root['parameters']['g0']), paired.frozen.interval(root['parameters']['gs'])
    frozen_field = primary['actual_mixed90_positive_field_function']
    frozen_budget = primary['actual_new_prepared_state_budget']
    field = {'root_error': paired.scalar(F(frozen_field['root_operator_error_upper']['upper']))}
    transitions = {
        'species': primary['actual_all65_gap_loading_and_costs'],
        'before': paired.scalar(F(frozen_budget['observed_actual_linear_Kato_before']['upper'])),
        'after': paired.scalar(F(frozen_budget['observed_actual_linear_kin_plus_matter_after']['upper'])),
    }
    actual_budget = paired.packet_budget(base, supplier, field, transitions, geom, [g0, gs, gs, gs])
    conservative = reprice['actual_new_prepared_state_budget']
    comparisons = {}
    for key in ('actual_new_state_reference_residual_upper', 'actual_new_state_Rayleigh_residual_upper', 'anti_H_generator_on_compact_packet_L2_upper', 'physical_source_endpoint_change_upper'):
        actual, accepted = F(actual_budget[key]['upper']), F(conservative[key]['upper'])
        if actual > accepted:
            raise ArithmeticError('Cached rational reprice underbounds corrected producer: '+key)
        comparisons[key] = {'direct_producer_upper': str(actual), 'conservative_cached_upper': str(accepted)}
    stable = all((ROOT/name).read_bytes() == raw for name, raw in captured.items())
    if not stable:
        raise ArithmeticError('Direct budget source/input mutation')
    result = {
        'kind': 'NEW_DIRECT_CORRECTED_PACKET_BUDGET_FROM_IMMUTABLE_COEFFICIENTS',
        'changed_consumer': 'induced_phi_car_superadiabatic_transfer.packet_budget',
        'historical_primary': {'path': PRIMARY, 'sha256': PINS[PRIMARY], 'coefficient_and_physical_point_not_rerun': True},
        'actual_direct_budget': actual_budget,
        'conservative_weighted_reprice_comparisons': comparisons,
        'cached_observation_fields_not_remeasured': ['observed_actual_linear_Kato_before', 'observed_actual_linear_kin_plus_matter_after'],
        'input_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in captured.items()},
        'all_inputs_and_sources_unchanged': stable,
        'runtime': {'executable': sys.executable, 'python': sys.version, 'mpmath': paired.mp.__version__, 'precision': paired.mp.mp.dps},
        'scope': 'Distinct conditional e44 uniform90 preparation; no Theta/Pi/Coulomb, full5760, ground, pole, commoncone or originalSOURCE admission.',
        'elapsed_seconds': time.monotonic()-started,
    }
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.output)
    print(json.dumps({'kind': receipt['kind'], 'elapsed_seconds': receipt['elapsed_seconds']}), flush=True)

"""Decompose the saved 9f4 cone-PROXY mismatch without rerunning its science.

No electric term, covariance, source, bare mass or microscopic law is changed.
The no-bare-inertia number is an algebraic diagnostic, not an adopted model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'induced_native_photon_marginal_counterterm_response_receipt.json'
INPUT_SHA = '9f4b82e8e5cba007b9dc1878ad8446c5d5b8ec2562762fe19ecf3f74744c7bd3'
PRODUCER_SHA = '2621267880f9c2566c2963b66cf3d68fa7612c91ae9fba4a2b456416232942c5'


def real_entry(value: Any) -> float:
    if isinstance(value, dict):
        if abs(float(value['imag'])) > 1e-12:
            raise ArithmeticError('Electric-response trace has a non-real diagonal')
        return float(value['real'])
    return float(value)


def run() -> dict[str, Any]:
    raw = INPUT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise RuntimeError('The actual fixed6296/9f4 response epoch changed')
    receipt = json.loads(raw)
    if receipt['provenance']['own_sha256'] != PRODUCER_SHA:
        raise RuntimeError('The saved coefficient formula has a different producer')
    cfg = receipt['actual_original_metric_inertia_counts_and_positive_sampling_labels_NOT_counterfactual_bare_input']
    after = receipt['actual_after_counterfactual_response']
    final = after['actual_orders'][-1]
    bare = .5 / cfg['inertia']
    epsilon = after['actual_IR_epsilon']
    kappa = after['actual_IR_kappa']
    current = after['actual_marginal_linear_current_coefficient']
    trace = sum(real_entry(final['actual_zero_epsilon_link'][j][j]) for j in range(3))
    induced = cfg['flavors'] * trace / (3 * cfg['a'])
    reconstructed = bare + induced
    if abs(reconstructed - epsilon) > 1e-11:
        raise ArithmeticError('Saved electric response does not match the actual producer formula')
    if min(bare, induced, kappa, current) <= 0:
        raise ArithmeticError('This diagnostic requires the saved positive response coefficients')
    actual_speed = math.sqrt(kappa / epsilon)
    actual_ratio = actual_speed / current
    if abs(actual_speed - after['actual_IR_photon_speed']) > 1e-14:
        raise ArithmeticError('Saved photon speed and coefficient units differ')
    if abs(actual_ratio - after['actual_photon_to_marginal_linear_ratio']) > 1e-14:
        raise ArithmeticError('Saved cone ratio is not a speed ratio')
    response_only_ratio = math.sqrt(kappa / induced) / current
    bare_factor = math.sqrt(induced / epsilon)
    # A pure-bare-inertia explanation predicts kappa=current^2*induced.
    pure_bare_prediction = bare_factor
    difference = actual_ratio - pure_bare_prediction
    if abs(difference) <= 1e-3 or response_only_ratio >= .99:
        raise ArithmeticError('The proposed discriminator did not reject the pure-bare explanation')
    own = Path(__file__).read_bytes()
    return {
        'scope': 'NEW saved-coefficient discriminator, conditional fixed6296/9f4 Gaussian closure only; not a physical dressed-velocity or common-cone certificate',
        'governing_plan': 'issue-222-induced-photon-delta13-continuation-20261008',
        'governing_revision': '21f3ef4514733e4067279f53582168d126b4b9d0ff01b767727c2a6fd2eac345',
        'template_revision': '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        'input': {'path': str(INPUT), 'sha256': INPUT_SHA, 'actual_producer_sha256': PRODUCER_SHA},
        'runtime': {'executable': sys.executable, 'python': platform.python_version()},
        'source_sha256': hashlib.sha256(own).hexdigest(),
        'source_utf8': own.decode(),
        'actual_saved_coefficients': {
            'epsilon_total': epsilon, 'epsilon_bare_primitive_electric': bare,
            'epsilon_induced_from_saved_link_trace': induced, 'kappa_static': kappa,
            'matter_linear_current_PROXY': current, 'photon_speed_IR_conditional': actual_speed,
            'ratio_semantics': 'photon speed / matter linear-current proxy, NOT squared-speed ratio',
        },
        'observed_decomposition': {
            'actual_speed_ratio': actual_ratio,
            'actual_speed_ratio_deficit_percent': 100*(actual_ratio-1),
            'response_only_speed_ratio_DIAGNOSTIC_not_adopted': response_only_ratio,
            'response_only_deficit_percent_DIAGNOSTIC': 100*(response_only_ratio-1),
            'bare_electric_multiplicative_speed_factor': bare_factor,
            'pure_bare_explanation_predicted_ratio': pure_bare_prediction,
            'pure_bare_prediction_minus_actual_ratio': pure_bare_prediction-actual_ratio,
            'factorization_absolute_defect': abs(actual_ratio-response_only_ratio*bare_factor),
            'epsilon_for_matching_current_proxy': kappa/current**2,
            'induced_electric_excess_relative_to_current_proxy': induced-kappa/current**2,
        },
        'conclusion': 'Primitive electric inertia contributes but cannot alone explain the recorded cone-proxy deficit: the unchanged induced electric/static response still fails the matter-current proxy relation. This separates one explanation, not a unique physical cause.',
        'boundaries': [
            'No original response producer, quadrature or finite-q refinement was rerun.',
            'Both endpoints use the SAME frozen6296 input; no spliced-epoch explanation.',
            'No electric term removed in the Hamiltonian; no covariance iteration/adoption or velocity fit.',
            'Matter current is an operator-cone proxy, not a dressed pole velocity; finite-gap p0 group velocity is zero.',
            'Finite binary arithmetic on saved coefficients is not an outward interval or full model-error bound.',
            'Bare/response factors multiply; their percent deficits must not be added as causal contributions.',
            'No instability, contact, pole or original source/receiver term suppressed; scientific campaign remains OPEN.',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('actual_saved_coefficients', 'observed_decomposition', 'conclusion')}, indent=2))

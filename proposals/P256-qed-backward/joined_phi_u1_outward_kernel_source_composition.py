"""Compose new outward body kernels with the actual compact-Phi map.

The old graph transfer's floating initial-current allowance is replaced by
its independently derived BODY operator envelope. No old science is rerun.
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

ROOT = Path(__file__).resolve().parent
PINS = {
    'joined_phi_u1_current_signal_interval_certificate_receipt.json': '9eb73c38e019cd87be153db01e65058e1bf65e3fa9a4bd6bf503059b2fe3725b',
    'joined_phi_u1_full_current_graph_transfer_verified_source_receipt_20261010.json': '2cc91cbe6b8b64f6d270ff04a9b946b22bd3a9a52ff9627e4bf15325f0c0adf4',
    'joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json': '2a1bbb46852707c82e477f28b5497e303b5ca66453db76d0655379f35cfb9a97',
    'joined_phi_u1_photon_causal_response_verified_20261010_receipt.json': 'c7c3ec787e34d5aa58c806650db7603e9f06c92d388cf590672d3644efbe262f',
}


def upper(record: dict[str, Any]) -> F:
    return F(record['upper'])


def run(output: Path) -> dict[str, Any]:
    start = time.monotonic()
    sys.set_int_max_str_digits(0)
    captured = {Path(__file__).name: Path(__file__).read_bytes()}

    def take(name: str) -> dict[str, Any]:
        raw = (ROOT/name).read_bytes()
        if name in PINS and hashlib.sha256(raw).hexdigest() != PINS[name]:
            raise ArithmeticError('Physical kernel composition input changed: '+name)
        captured[name] = raw
        return json.loads(raw)

    def bind_capsule(receipt_name: str, capsule_name: str, producer: str) -> dict[str, Any]:
        receipt = take(receipt_name)
        capsule = take(capsule_name)
        code = capsule.get('natural_exit_code', capsule.get('returncode'))
        if code != 0 or capsule['receipt_sha256'] != hashlib.sha256(captured[receipt_name]).hexdigest():
            raise ArithmeticError('Missing exact natural0 receipt binding: '+receipt_name)
        own = (ROOT/producer).read_bytes()
        source_sha = capsule.get('source_sha256', capsule.get('pinned_before', {}).get(producer))
        if source_sha != hashlib.sha256(own).hexdigest():
            raise ArithmeticError('Producer does not match completed source epoch: '+producer)
        captured[producer] = own
        return receipt

    current = bind_capsule('joined_phi_u1_current_signal_interval_certificate_receipt.json', 'joined_phi_u1_current_signal_interval_certificate_natural_exit_capsule.json', 'joined_phi_u1_current_signal_interval_certificate.py')
    photon = bind_capsule('joined_phi_u1_photon_causal_response_verified_20261010_receipt.json', 'joined_phi_u1_photon_causal_response_verified_20261010_natural_exit_capsule.json', 'joined_phi_u1_photon_causal_response.py')
    nonlocal_result = bind_capsule('joined_phi_u1_nonlocal_signal_interval_certificate_typed_verified_receipt_20261010.json', 'joined_phi_u1_nonlocal_signal_interval_certificate_typed_verified_natural_exit_capsule_20261010.json', 'joined_phi_u1_nonlocal_signal_interval_certificate.py')
    graph = take('joined_phi_u1_full_current_graph_transfer_verified_source_receipt_20261010.json')
    source = take('joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json')
    root_join = take('joined_phi_u1_current_root_source_composition_receipt_20261010.json')
    if graph['input_sha256']['joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json'] != PINS['joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json']:
        raise ArithmeticError('Graph and compact source use different epochs')
    if not source['all_inputs_and_sources_unchanged'] or not graph['stable_input_source_bytes'] or not photon['stable_source_input_bytes']:
        raise ArithmeticError('Source/graph/causal capsule lacked stable inputs')
    if F(current['clock']) != F(photon['clock_exact']) or current['coordinates'] != 129:
        raise ArithmeticError('Current and causal response clock/coordinate mismatch')
    current_states = {}
    jbody = upper(graph['global_constants']['body_J_operator'])
    for label, mass, norm_upper in (('background', F(1), F(1)), ('raw_source', F(1,2), F(3,4))):
        body = current['all129_states'][label]
        old = graph['states'][label]
        if F(body['raw_norm_squared']) != mass or F(old['raw_norm_squared']) != mass or F(old['clock_exact']) != F(current['clock']):
            raise ArithmeticError('Current body/physical mass or clock mismatch')
        if norm_upper**2 < mass:
            raise ArithmeticError('Invalid exact Hilbert norm upper')
        left, right = [upper(old['prices'][k]) for k in ('current_left_error', 'current_right_error')]
        # Exact global BODY envelope, not a fullfield current bound and not
        # the old floating current-inserted-state norm with a 1ppm allowance.
        transfer = 2*(left*jbody*norm_upper+jbody*norm_upper*right+left*right)
        body_lower = F(body['body_HS_lower'])
        physical_lower = body_lower-transfer
        if physical_lower <= 0:
            raise ArithmeticError('Outward physical retarded signal unresolved: '+label)
        current_states[label] = {'raw_mass_exact': str(mass), 'body_HS_lower_exact': str(body_lower), 'physical_transfer_upper_exact': str(transfer), 'physical_HS_lower_exact': str(physical_lower), 'physical_HS_lower_display': float(physical_lower)}
    for label, state in photon['all129_states'].items():
        signal = F(state['cubic_signal_HS_lower']['lower'])
        error = upper(state['mixed_body_prices']['D_remainder'])+upper(state['physical_transfer']['D_transfer'])
        if signal-error <= 0 or F(state['minimum_directional_D_signed_margin']) <= 0:
            raise ArithmeticError('Whole or directional causal correction unresolved: '+label)
        current_states[label]['photon_correction_HS_lower_exact'] = str(signal-error)
    if len(nonlocal_result['receiver_rows']) != 12 or nonlocal_result['floating_premise_used']:
        raise ArithmeticError('Actual signed receiver interval scope incomplete')
    if any(F(r['change_from_own_physical_initial_lower_exact']) <= 0 for r in nonlocal_result['receiver_rows']):
        raise ArithmeticError('Physical receiver has no own-initial change proof')
    stable = all((ROOT/name).read_bytes() == raw for name, raw in captured.items())
    if not stable:
        raise ArithmeticError('Integrated physical kernel source/input mutated')
    result = {
        'status': 'OUTWARD_CURRENT_PHOTON_AND_PHYSICAL_RECEIVER_COMPOSITION_PROVED_AT_DECLARED_PREPARATION',
        'current_states': current_states,
        'current_transfer_derivation': '2[left*jbody*n_upper+jbody*n_upper*right+left*right], with jbody the exact derived BODY operator envelope. Discards the prior floating initial-current norm allowance and all1percentF signal premises. Global quartic physical current transfer is unchanged.',
        'root_composition': root_join,
        'photon_signed_scope': 'All129 nonstationary coordinate/Pi correction with contact, unequal-time channel, Theta-inserted graph transfer and all3torons. Not a stationary spectral pole or dispersion.',
        'nonlocal_signed_receivers': nonlocal_result['receiver_rows'],
        'declared_preparation': 'Background Tbody psi and raw fixed Tsrc E0body psi for current/photon; nonlocal receiver additionally compares actual physical E0/Ex to these via separate eta0/etax. No originalSOURCE or CURRENT427 adoption.',
        'conditions': 'Inherited exact declared root/static/q0 and compact-source/global-domain/action inequalities; outward body elementary intervals plus binary64 contraction bounds conditional on IEEE arithmetic. No measured full129 time evolution, interacting ground, stationary pole, commoncone or Compton closure.',
        'input_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in captured.items()},
        'all_inputs_and_sources_unchanged': stable,
        'runtime': {'executable': sys.executable, 'python': sys.version},
        'elapsed_seconds': time.monotonic()-start,
    }
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.output)
    print(json.dumps({'status': receipt['status'], 'current': receipt['current_states'], 'elapsed_seconds': receipt['elapsed_seconds']}), flush=True)

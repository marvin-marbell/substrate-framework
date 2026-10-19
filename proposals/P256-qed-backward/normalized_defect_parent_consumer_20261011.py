"""Compose reviewed additive-defect and normalized-root bounds; no old CLI runs."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(name: str) -> Any:
    return json.loads((ROOT / name).read_text())


def archive_epoch(capsule_name: str, pins: dict[str, str]) -> dict[str, Any]:
    capsule = load(capsule_name)
    assert capsule['returncode'] == 0 and capsule['natural_exit'] is True
    archive_name = capsule['executed_zip']
    assert sha(ROOT / archive_name) == capsule['executed_sha256']
    members = []
    with zipfile.ZipFile(ROOT / archive_name) as archive:
        for name, expected in pins.items():
            matches = [m for m in archive.namelist()
                       if Path(m).name == name and hashlib.sha256(archive.read(m)).hexdigest() == expected]
            assert matches, (archive_name, name, expected)
            members.append({'name': name, 'member': matches[0], 'sha256': expected,
                            'live_equal': (ROOT / name).is_file() and sha(ROOT / name) == expected})
    return {'capsule': capsule_name, 'capsule_sha256': sha(ROOT / capsule_name),
            'archive': archive_name, 'archive_sha256': capsule['executed_sha256'],
            'natural_exit': True, 'returncode': 0, 'members': members,
            'historical_bytes_not_rebound': True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request = json.loads(Path(args.request).read_text())
    for name, pin in request['pins'].items():
        assert sha(ROOT / name) == pin, name

    defect = load('matching_defect_interface_20261011.json')
    quantities = defect['quantities']
    normalized = load('normalized_vacuum_interface_20261011.json')
    theorem = normalized['operator_source_bound_theorem']
    mayer = load('normalized_mayer_review_verdict_20261011.json')
    operator_review = load('defect_operator_review_proof_20261011.json')
    price_review = load('defect_price_review_verdict_20261011.json')
    native_review = load('native_boundary_review_actual3_verdict_20261011.json')
    notation = load('matching_defect_projector_notation_addendum_20261011.json')
    assert notation['frozen_target']['sha256'] == sha(ROOT / 'matching_defect_interface_20261011.json')
    assert price_review['verdict'] == 'VALID_CONDITIONAL_WITH_FROZEN_PROJECTOR_NOTATION_CORRECTION'
    assert mayer['verdict'] == 'VALID_CONDITIONAL_WITH_EXPLICIT_CONSTRUCTION_AND_TIME_LIMIT_COMPLETIONS'
    assert native_review['verdict'] == 'CONDITIONAL'
    reviewed_bindings = []
    original_capsule = load('normalized_vacuum_actual1_20261011_capsule.json')
    original_archive = ROOT / original_capsule['executed_zip']
    assert sha(original_archive) == original_capsule['executed_sha256']
    with zipfile.ZipFile(original_archive) as archive:
        for name, expected in mayer['frozen_targets'].items():
            live_equal = sha(ROOT / name) == expected
            archived = [m for m in archive.namelist()
                        if Path(m).name == name and hashlib.sha256(archive.read(m)).hexdigest() == expected]
            assert live_equal or archived, name
            reviewed_bindings.append({'name': name, 'reviewed_sha256': expected,
                                      'live_equal': live_equal, 'original_member': archived[0] if archived else None,
                                      'postexecution_standalone_if_unarchived': not archived})

    matching_capsule = load('matching_defect_capsule_20261011.json')
    matching_pins = matching_capsule['input_source_sha256'] | {
        'matching_defect_interface_20261011.json': matching_capsule['interface_sha256']}
    normalized_request = load('normalized_vacuum_actual1_20261011_request.json')
    normalized_capsule = load('normalized_vacuum_actual1_20261011_capsule.json')
    normalized_pins = normalized_request['pins'] | {
        normalized_capsule['receipt']: normalized_capsule['receipt_sha256']}
    archives = [archive_epoch('matching_defect_capsule_20261011.json', matching_pins),
                archive_epoch('normalized_vacuum_actual1_20261011_capsule.json', normalized_pins)]

    gamma = F(quantities['gamma_exact'])
    eg = F(quantities['E_g_upper_exact'])
    baseline = F(quantities['baseline_lower_exact'])
    delta = F(quantities['delta_lower_exact'])
    alpha = baseline + 96 + 5123
    assert gamma == -5123 - eg and alpha == F(quantities['excess_flow_alpha_exact'])
    assert 0 < alpha < 32
    margins = [F(-5155 + 32) - eg, F(-5187 + 64) - eg,
               baseline + 2 * delta + 64 - eg, baseline + 96 - eg]
    assert margins[0] == gamma and margins[1] == gamma
    assert all(m >= gamma for m in margins)
    # All w>=3 reduce to the w=3 endpoint and positive slope, not a finite flow crop.
    simultaneous_at_three = baseline + 96 - eg - gamma / 2 - 3 * gamma / 4
    assert simultaneous_at_three >= 0 and 32 - gamma / 4 > 0
    assert all(m >= gamma / 2 + gamma * w / 4
               for m, w in zip(margins[:3], (1, 2, 2)))
    excess_at_three = baseline + 96 - eg - gamma - alpha
    assert excess_at_three >= 0 and 32 - alpha > 0
    assert len(quantities['all_number_sector_envelopes']) == 513
    assert [row['number'] for row in quantities['all_number_sector_envelopes']] == list(range(513))

    # Exact diagonal domain witness: ground, zero-flow neutral excitation, nonzero flow.
    p_flow = (1, 1, 0)
    p_ground = (1, 0, 0)
    q_ground = tuple(1 - p for p in p_ground)
    q_flow = tuple(1 - p for p in p_flow)
    neutral_memory = tuple(a - b for a, b in zip(p_flow, p_ground))
    assert q_ground == tuple(a + b for a, b in zip(q_flow, neutral_memory))
    assert q_ground[1] == 1 and q_flow[1] == 0
    # The wrong flow-only inverse deletes a genuine neutral spectral contribution.
    energies = (F(0), gamma, 2 * gamma)
    source = (0, 1, 1)
    correct_inverse = sum(F(source[i] ** 2) / energies[i] for i in (1, 2))
    wrong_flow_inverse = F(1) / energies[2]
    assert correct_inverse - wrong_flow_inverse == 1 / gamma

    params = theorem['parameters']
    assert gamma == F(params['gamma_exact'])
    support = params['s_max_cells']
    split = params['C_projector_splits']
    theta2 = F(params['theta_squared_exact'])
    j = F(params['J_exact'])
    assert (support, split, theta2) == (3, 64, F(1, 64))
    radius = gamma / (72 * split * support * j)
    assert radius == F(theorem['radius_exact']) == F(mayer['exact_constants']['radius_exact'])
    x_over_gamma = split * j * 3 * radius / gamma  # strict upper e<3
    r = 4 * split * support * j * 9 * radius / gamma  # e²<9
    event_guard = theta2 * x_over_gamma / (1 - r)
    overlap_guard = 8 * support ** 2 * theta2 * x_over_gamma * (1 + r) / (1 - r) ** 3
    assert r == F(1, 2) and event_guard == F(1, 2304) and overlap_guard == F(3, 16)
    assert event_guard < F(1, 2) and overlap_guard < 1
    # Reserve controls the entire attached Mayer cluster, not just a boundary rod.
    reserved_r = r * F(50, 49)
    reserved_event = theta2 * x_over_gamma / (1 - reserved_r)
    reserved_overlap = (8 * support ** 2 * theta2 * x_over_gamma *
                        (1 + reserved_r) / (1 - reserved_r) ** 3 * F(50, 49))
    assert reserved_r == F(25, 49)
    assert reserved_event == F(49, 110592)
    assert reserved_overlap == F(45325, 221184) < 1
    physical1_ratio = 36 * split * support * j / gamma
    assert physical1_ratio == F(normalized['physical1']['r_upper_at1_exact']) > 1
    # Distinguish total Taylor tail from a root-component-only geometric tail.
    rho_fraction = F(1, 2)
    order = 4
    total_tail_per_root_norm = 18 * split * rho_fraction ** order / (1 - rho_fraction)
    wrong_old_root_constant = 3 * split
    corrected_root_constant = 9 * split
    assert corrected_root_constant > wrong_old_root_constant
    # A zero-length P/P root support point is incompatible with a crossing vacuum rod.
    rod_start, rod_end, root_time = F(-1), F(2), F(0)
    assert rod_start < root_time < rod_end
    rods_only_wrong_incompatibility = False
    blocked_point_incompatibility = rod_start <= root_time <= rod_end
    assert blocked_point_incompatibility and not rods_only_wrong_incompatibility

    review_fields = {'operator': operator_review.get('verdict', operator_review.get('overall_verdict')),
                     'price': price_review['verdict'], 'Mayer': mayer['verdict'],
                     'native_boundary': native_review['verdict']}
    result = {
        'kind': 'NEW_PARENT_COMPOSED_ADDITIVE_DEFECT_NORMALIZED_ROOT_CONSUMER',
        'stage': 'EXACT_RATIONAL_COMPOSITION_OF_REVIEWED_CONDITIONAL_PROOFS_NOT_MACHINE_ATTESTATION',
        'captured_governor': request['captured_governor'],
        'accepted_scoped_science_before_run': request['accepted_scoped_science'],
        'own_source_sha256': sha(Path(__file__)), 'input_pins': request['pins'],
        'immutable_original_epoch_bindings': archives, 'review_verdicts': review_fields,
        'additive_forms': {'gamma_exact': str(gamma), 'excess_flow_alpha_exact': str(alpha),
                          'count_and_flow_price_exact': str(gamma / 2),
                          'per_class_margin_exact': [str(m) for m in margins],
                          'unbounded_flow_endpoint_and_slope_proof': True,
                          'all_513_number_labels_retained': True},
        'projection_boundary': {'P_flow0': 'entire external-zero superselection block, not ground',
                               'P0': 'entire product-g rank-one reference spectral ground',
                               'Q0': 'nonzero flow plus neutral zero-flow ground complement',
                               'neutral_inverse_wrongcase_defect_exact': str(1 / gamma)},
        'normalized_branch': {'radius_exact': str(radius), 'radius_display': float(radius),
                              'root_bound_constant_per_norm': 9 * split,
                              'root_support_includes_P_P_blocked_point': True,
                              'event_over_gamma_upper_exact': str(event_guard),
                              'overlap_upper_exact': str(overlap_guard),
                              'whole_cluster_length_reserve_exact': str(gamma / 100),
                              'reserved_r_upper_exact': str(reserved_r),
                              'reserved_event_over_gamma_exact': str(reserved_event),
                              'reserved_overlap_exact': str(reserved_overlap),
                              'time_boundary_decay_rate_exact': str(gamma / 200),
                              'T_limit_scope': 'Cauchy normalized matching branch from entire length-weighted root-cluster sum',
                              'total_Taylor_tail_example': {'rho_over_R': str(rho_fraction), 'order': order,
                                                           'upper_per_root_norm_exact': str(total_tail_per_root_norm)},
                              'component_tail_is_not_total_cluster_tail': True},
        'physical1': {'r_upper_exact': str(physical1_ratio), 'r_upper_display': float(physical1_ratio),
                      'sufficient_package_passes': False, 'verdict': 'FAIL_TO_CERTIFY_NOT_MODEL_NO_GO'},
        'conditional_antecedents': mayer['conditional_antecedents_not_reproved_here'],
        'reviewed_original_target_bindings': reviewed_bindings,
        'source_boundary_not_interior_gain': True, 'historical_main_runs': 0,
        'new_scientific_acceptance': False, 'campaign_complete': False,
        'open': ['full physicalPhi/lattice stationary ground', 'physical1 normalized linked phase',
                 'charged exterior', 'gap-closing mechanism', 'bulk stiffness', 'sharp IR atom/photon',
                 'common cone', 'original SOURCE adoption'],
        'elapsed_seconds': time.monotonic() - started,
        'sources_inputs_unchanged': all(sha(ROOT / name) == pin for name, pin in request['pins'].items())}
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'radius': float(radius), 'reserved_overlap': str(reserved_overlap),
                      'physical1_r': float(physical1_ratio), 'historical_main_runs': 0,
                      'campaign_complete': False}))


if __name__ == '__main__':
    main()

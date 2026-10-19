"""Read-only immutable heat/analytic-flux integration, not campaign closure."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time
import sys
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent
BINDING = 'charged_heat_analytic_flux_parent_typed_bindings_20261011.json'
BINDING_SHA = 'aa9966ab22a815887e7382388b6da85dcb6ea137d077877dd32b7834487c7741'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(name: str) -> Any:
    return json.loads((ROOT / name).read_bytes())


def archived_input(name: str, expected: str, stages: list[dict[str, Any]]) -> str | None:
    """Resolve by bytes, never by whichever live producer has the same filename."""
    for stage in stages:
        for archive_name in stage['artifact_sha256']:
            if archive_name.endswith('.zip'):
                with zipfile.ZipFile(ROOT / archive_name) as archive:
                    if name in archive.namelist() and hashlib.sha256(archive.read(name)).hexdigest() == expected:
                        return archive_name + ':' + name
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    sys.set_int_max_str_digits(0)  # SHA-pinned trusted exact proof endpoints.
    assert digest(ROOT / BINDING) == BINDING_SHA
    bindings = load(BINDING)
    stages = []
    verified_members = 0
    for stage in bindings['stages']:
        prefix = stage['prefix']
        assert {n: digest(ROOT / n) for n in stage['artifact_sha256']} == stage['artifact_sha256']
        ex = load(prefix + '_exit.json')
        assert ex.get('returncode', ex.get('exit_code')) == 0 and ex['natural_exit']
        assert all(ex[k] is True for k in stage['expected_success_flags'])
        assert ex['preexecution_sha256'] == stage['artifact_sha256'][prefix + '_preexecution.zip']
        archive_hashed = []
        separately_snapshotted = []
        with zipfile.ZipFile(ROOT / (prefix + '_preexecution.zip')) as pre, zipfile.ZipFile(ROOT / (prefix + '_executed.zip')) as executed:
            for name, expected in stage['source_input_sha256'].items():
                holders = [z for z in (pre, executed) if name in z.namelist()]
                if holders:
                    for z in holders:
                        assert hashlib.sha256(z.read(name)).hexdigest() == expected
                        verified_members += 1
                    archive_hashed.append(name)
                else:
                    found = archived_input(name, expected, bindings['stages'])
                    if found is not None:
                        archive_hashed.append(found)
                        verified_members += 1
                    else:
                        # A NEW parent snapshot is not inserted into a historical ZIP.
                        assert digest(ROOT / name) == expected
                        separately_snapshotted.append(name)
            receipt_name = prefix + '_receipt.json'
            assert hashlib.sha256(executed.read(receipt_name)).hexdigest() == stage['artifact_sha256'][receipt_name]
            archived_exit = json.loads(executed.read(prefix + '_exit.json'))
            assert archived_exit.get('returncode', archived_exit.get('exit_code')) == 0
            assert archived_exit['natural_exit']
        stages.append({'prefix': prefix, 'natural_exit': True, 'original_immutable_member_names': archive_hashed, 'new_parent_snapshot_NOT_historical_backfill': separately_snapshotted})
    rev = bindings['independent_review']
    assert digest(ROOT / rev['file']) == rev['sha256']
    review = load(rev['file'])
    assert review['verdict'] == 'CONDITIONAL' and not review['load_bearing_target_errors']
    assert review['campaign_complete'] is False
    analytic = load('analytic_flux_time_actual3_20261011_receipt.json')
    assert len(analytic['orders']) == 96
    assert F(analytic['quadrature_error_exact']) == 0
    for z in ('0', '64'):
        lo, hi = map(F, analytic['coefficient_intervals'][z]['re']['exact'])
        assert 0 < lo <= hi and analytic['sign_verdict'][z] == 'POSITIVE'
        budget = analytic['error_budgets'][z]
        assert F(budget['coherent_remainder']['exact']) <= F(budget['operator_remainder']['exact']) <= F(budget['trace_baseline']['exact'])
        assert F(budget['source_increment']['exact']) >= 0
    typed = bindings['typed_transfer']
    new_analytic = load(typed['analytic_receipt'])
    assert all(analytic[k] == new_analytic[k] for k in analytic if k not in ('own_sha256', 'elapsed_seconds'))
    old_heat = load('charged_cell_heat_interface_20261011.json')
    new_heat = load(typed['heat_interface'])
    assert all(old_heat[k] == new_heat[k] for k in typed['mathematical_heat_keys'])
    old_control = load('heat_flux_review_actual2_20261011_receipt.json')
    new_control = load(typed['review_receipt'])
    assert all(old_control[k] == new_control[k] for k in ('matrix_functional_calculus', 'native_commutator', 'tensor_factorization', 'coherence_and_neutral_scope'))
    boundary = load('analytic_flux_feshbach_parent_actual1_20261011_receipt.json')
    assert F(boundary['selected_C0_minus_C64_real_lower_exact']) > 0
    assert boundary['energy_independent_selected_fourhop_coefficient_rejected']
    abi = load(typed['abi_receipt'])
    assert abi['complex_canonical2_CAR_wrong_transpose_norm'] > 1
    assert F(abi['wrong_entrywise_Loewner_positivity_rejected']['offdiagonal_real_remainder']) < 0
    result = {
        'kind': 'REVIEWED_HEAT_ANALYTIC_FLUX_IMMUTABLE_PARENT_INTEGRATION',
        'own_source_sha256': digest(Path(__file__)),
        'typed_transfer_mathematical_payloads_exactly_unchanged': True,
        'typed_review_frozen_target_live_source_differences': new_control['source_capsule_mismatches'],
        'bindings_sha256': BINDING_SHA,
        'captured_bookkeeping_revision': bindings['captured_bookkeeping_revision'],
        'accepted_prior_scoped_science': bindings['accepted_scoped_science'],
        'verified_original_archive_member_reads': verified_members,
        'stages': stages,
        'independent_frozen_heat_flux_verdict': review['verdict'],
        'analytic_selected_real_intervals': analytic['coefficient_intervals'],
        'selected_energy_dependence_lower_exact': boundary['selected_C0_minus_C64_real_lower_exact'],
        'formal_Feshbach_sign': '-C(z); this positive z is a resolvent energy shift, not photon frequency.',
        'formation_scope': 'Conditional correctly-rounded historical operation model and accepted source/ground caps. Not machine attestation.',
        'parent_review_status': 'Immutable/composed consumer completed; scientific scope decided separately in parent proof review.',
        'campaign_complete': False,
        'still_open': ['physicalPhi/lattice ground', 'normalized N-uniform physical1 linked phase', 'stiffness', 'gap-closing massless IR atom/sharp photon', 'commoncone', 'originalSOURCE'],
        'elapsed_seconds': time.monotonic() - started,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('kind', 'verified_original_archive_member_reads', 'selected_energy_dependence_lower_exact', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()

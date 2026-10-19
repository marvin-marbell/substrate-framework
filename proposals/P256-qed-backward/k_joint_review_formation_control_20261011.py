"""Independent serialized coefficient-budget and whole-ground interface control."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT/name).read_bytes())


def sha(name):
    with (ROOT/name).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(request_name, output_name):
    request = load(request_name)
    assert all(sha(n) == p for n, p in request['input_sha256'].items())
    old = load('positive_residual_actual1_20261011_receipt.json')
    formation = old['formation']
    price, beta = F(formation['projected_operator_price_exact']), F(formation['beta_exact'])
    result = []
    for axis in old['axes']:
        errors = {name: F(0) for name in ('G', 'JC', 'CC', 'JT', 'Tgram')}
        raw_above_pair = 0
        for row in axis['projection_accounting']:
            mult = 1 if row['role'] == 0 else 63
            assert row['physical_count'] == mult
            account = row['accounting']
            pair = F(account['pre_normal_order_term_pair_L1_exact'])
            raw = F(account['post_normal_order_pre_outer_collection_L1_exact'])
            formed = F(account['formation_L1_exact'])
            outer = F(account['signed_outer_L1_exact'])
            assert formed == max(pair, raw)
            errors[row['operator']] += mult*(outer*price+formed*beta)
            raw_above_pair += raw > pair
        original = axis['scalar_anchor_sweep'][3]
        assert F(axis['G_operator_error_exact']) == errors['G']
        assert F(original['signed_B_error_exact']) == errors['JT']
        assert F(original['TdagT_error_exact']) == errors['Tgram']
        for row in original['energies']:
            de = F(old['domains']['whole_operator_two_cell_floor_exact'])-F(row['E_exact'])
            assert F(row['joint_operator_error_exact']) == errors['JT']+errors['Tgram']/de
        result.append(dict(axis=axis['axis'], coefficient_errors_exact={k: str(v) for k, v in errors.items()},
                           native_raw_above_operand_pair_cases=raw_above_pair,
                           physical_multiplicities=[1, 63], all_orientation_prices_retained=True))
    ground = load('local_so5_ground_upper_complete_state_consumer_20261011_receipt.json')
    gamma = F(46955312929, 34359738368)
    assert gamma == -5123-F(ground['ground_energy_interval_exact'][1])
    assert ground['normalized_exact_source_residual_upper_exact'] == '1/65536'
    assert ground['coefficient_and_action_error_upper_exact'] == '1/262144'
    assert F(formation['infidelity_exact']) == (F(1, 65536)/gamma)**2
    whole = load('canonical_ground_review_proof_20261011.json')
    assert whole['exact_scope']['conclusion'].startswith('g is unique local fullcanonical ground')
    accepted = load('joined_phi_u1_physical_charged_canonical_neutral_parent_packet_20261011.json')['scoped_claims']['frozen_fullcanonical_local_ground']
    assert accepted['proof_sha256'] == sha('canonical_ground_review_proof_20261011.json')
    assert accepted['gap_lower_exact'] == str(gamma) and accepted['H_on_g_perp_lower_exact'] == '-5123'
    assert accepted['unique_exact_ground_g'] and accepted['all_negative_free_classes_covered']
    assert all(sha(n) == p for n, p in request['input_sha256'].items())
    receipt = dict(kind='EXACT_SERIALIZED_FORMATION_AND_CONDITIONAL_GROUND_INTERFACE_CONTROL',
        source_sha256=sha(Path(__file__).name), input_sha256=request['input_sha256'],
        errors=result, source_cap='1/65536', formation_cap='1/262144',
        accepted_fullcanonical_complement_proof_bound=True,
        wholecanonical_ground_rederived_here=False, historical_operation_audit=False,
        observed_small_residual_not_used=True, natural_execution_required_separately=True)
    (ROOT/output_name).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(axes=len(result), exact_G_B_T_joint_error_matches=True,
                         accepted_fullcanonical_ground_antecedent_bound=True)), flush=True)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])

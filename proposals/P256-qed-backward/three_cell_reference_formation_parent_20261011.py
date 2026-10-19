"""Exact source-moment and immutable epoch integration, not a physical1 ground."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
PINS = {'local_so5_resum_actual04_20261011_state.npz': 'dd06400e1b7f70df935cf7ca02cf5ebf598481a8d5e209d1241aba5ed3fea2a2', 'three_cell_density_formation_interface_20261011.json': 'c40506cde408eb241602bd2e79696382c7cf0b3f9b55fee8d913930bbecd9318', 'three_cell_density_formation_proof_20261011.json': '0677b384cb4faa4d47673d2a26d4d35975103501bb074e5db9fb3e1605d91088', 'density_formation_mathreview_verdict_20261011.json': 'f100a7c86f03e04d89a485b62226092759a5a59132bf140eaea8e9b40b31fc0e', 'source_metric_mathreview_verdict_20261011.json': '89f907262c054acf118f0878090867a936b19ff731a0bdbd278a5aa8182274ed', 'reference_h0_mathreview_verdict_20261011.json': 'a26fdc95a0978ef812feaf52fe7aaf9b7975214c2ec5713e2643aeca57346300', 'three_cell_native_proof_20261011.json': 'c67fa1780acd241d0f7869e34d4d3e7979f3199b57763392354817d28c7fb043', 'three_cell_native_actual3_20261011_receipt.json': '1616f84b0a4c09f7ab1905d0b158601cd2fa7e082a4ee4e1312b35e1dcc7a741', 'three_cell_metric_transfer_actual1_20261011_executed.zip': 'be1732b5b202d1f228de76b475f74125443f9c535a73fffd8c149877db771b19', 'three_cell_metric_transfer_semantic2_20261011_executed.zip': 'f187b7bf44488a371557ea122ae5b753527e41f1c9ddb22cfc82d4d35bc9e621', 'reference_matching_flux_spectral_parent_attempt1_20261011_executed.zip': '4e18f54c3bca93b80368557bb47ae547de94b6fb593291eb484634297c814fbe', 'reference_matching_flux_spectral_parent_attempt2_20261011_executed.zip': 'afaed2b4e8b6c25d7234f877fa49bc24edd5d384b7c975c5e21d0453eba44da3', 'affine_canonical_weak_actual1_20261011_executed.zip': '4aa209be85f9bf41aa37f5d9bcf73650f10b4ab8c28072fc881b755f44e965e2', 'affine_canonical_weak_minus_actual1_20261011_executed.zip': '2d7b8b7918f7d1dedcc9876b5d8058376e257ba66b76f3f94d2197c0719fe9b7', 'affine_canonical_weak_proof_20261011.json': 'c7b57c6643f57b4d0b548e1052ffe2ff2d7183ed6c61e9114acba7e0aa186615', 'joined_phi_u1_physical_charged_canonical_neutral_scoped_acceptance_20261011.json': '46d482d19a3cd409979b3a74d9237887c5bdf13d7e241ad9ea9fcf30d597841c'}
PINS.update({
    'affine_lower_mathreview_proof_20261011.json': '055309d7bb6209a1bcd8a45b8b1eaab0272dbc9f40df3d94f6a25d9066e3d265',
    'affine_lower_mathreview_verdict_20261011.json': 'beb9045d8ef12bd9670d30da8777bf523b611c7394294deb89022898500b1a63',
    'affine_lower_mathreview_epoch2_20261011_receipt.json': 'e7cddffe8c232343c2398aa2da7bb5ffb9555099b7125c60bb98bbd8d4397719',
    'affine_lower_mathreview_epoch2_20261011_executed.zip': 'fd289bd7f084fafbd700c19fb20c1516556d5157dd77cea404fcc7906815c3ac',
})
PINS.update({
    'three_cell_density_formation_typed3_20261011_exit.json': '052374c137666cdf64bc9619d5120d2796d92084449a6862bbfe61fcfe74ba24',
    'three_cell_metric_transfer_typed3_20261011_exit.json': 'd6c8422d9142e008051be244b0edd70aa2ce4c3386c37779ed0e7b3d232e65ef',
    'root_pair_reference_parent_typed3_20261011_exit.json': '598db15c1b221d70f63145cd5b10108b63f504c6c8760028929b858d0fde77e5',
    'density_formation_mathreview_control_typed2_20261011_exit.json': 'a8b059c2619a166ae1086138ba9f93e8b647bf56c8df9b64c224bc3e9a15d67e',
    'source_metric_mathreview_control_typed2_20261011_exit.json': 'ee887a2aebc2940434a05e8b2743071d87e129ad385ed09ce18b0027b9d13d54',
    'reference_h0_mathreview_control_typed2_20261011_exit.json': '49fddb115d79137fc961a80c8e7999f580a8e3de10beb4ad81fb5c6486bb249b',
    'root_pair_reference_parent_attempt2_20261011_receipt.json': '226e2ec28a156ce3109136c48faefbfec356e86ebe76755b4319490631229feb',
    'three_cell_reference_kernel_parent_bindings_20261011.json': '4fc7d7c04bf89113cd79241a80850dcc4c8615db905c4f3eaabfb3bb25aec51d',
})


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(name: str) -> Any:
    return json.loads((ROOT / name).read_bytes())


def archived_json(archive: zipfile.ZipFile, name: str) -> Any:
    return json.loads(archive.read(name))


def exact_source_moments() -> dict[str, Any]:
    # The actual native labels are (l,p,q) ordered as in representation(n),
    # with rho=p-q. Neither floating multiplication nor reduction is used.
    ra = [p-q for l in range(3) for p in range(l+1) for q in range(l+1)]
    rb = np.fromiter((p-q for l in range(127) for p in range(l+1)
                      for q in range(l+1)), dtype=np.int16)
    bins = [[0] * 2047 for _ in range(3)]
    with np.load(ROOT / 'local_so5_resum_actual04_20261011_state.npz', allow_pickle=False) as data:
        x = data['state']
        assert x.dtype == np.dtype('float64') and x.size == len(ra)*rb.size == 9672320
        x = x.reshape(len(ra), rb.size)
        for a, row in zip(ra, x):
            for raw, b in zip(row.view(np.uint64), rb):
                word = int(raw)
                exponent = (word >> 52) & 2047
                assert exponent != 2047
                mantissa = word & ((1 << 52)-1)
                if exponent:
                    mantissa += 1 << 52
                square = mantissa * mantissa
                rho2 = (a + int(b))**2
                bins[0][exponent] += square
                bins[1][exponent] += square * rho2
                bins[2][exponent] += square * rho2 * rho2
    values = [F(group[0] + sum(value << (2*(exponent-1))
                for exponent, value in enumerate(group[1:], 1)), 1 << 2148)
              for group in bins]
    assert values[0] > 0
    second, fourth = values[1]/values[0], values[2]/values[0]
    assert second < 1 and fourth < 2
    # A shared two-role state with rhoA=1,rhoU=-1 has total rho^2=0,
    # unlike the incorrect independent/diagonal-only sum, which gives2.
    assert (1 + (-1))**2 == 0 and 1**2 + (-1)**2 == 2
    return {'norm_exact':str(values[0]), 'rho2_exact':str(second),
            'rho4_exact':str(fourth), 'rho2_cap_exact':'1', 'rho4_cap_exact':'2',
            'source':'Exact normalized STORED s, not the true ground g',
            'summation':'IEEE754 significand integer squares including all subnormals; no historical FP positive-reduction premise',
            'role_domain':'H2 tensor H126 native labels, total shared rho=pA-qA+pU-qU; active1+aux63',
            'wrong_missing_cross_rejected':True,
            'normalization_changes_source_or_adopts_covariance':False}


def check_weak(archive_name: str, receipt_name: str, sign: str) -> dict[str, Any]:
    with zipfile.ZipFile(ROOT / archive_name) as archive:
        result = archived_json(archive, receipt_name)
        assert result['u_exact'] == sign and result['floor_exact'] == '-5187'
        assert result['all18_certified'] and len(result['families']) == 18
        pivots = []
        for family in result['families']:
            assert family['number_sector'] == 256 and family['merged_parity']
            assert family['identity_rows'] == family['output_rows']
            assert family['entire_output_preimages_checked']
            assert F(family['omitted_free_H_minus_z_lower_exact']) > 0
            raw = archive.read(family['matrix_path'])
            assert hashlib.sha256(raw).hexdigest() == family['matrix_sha256']
            assert hashlib.sha256(archive.read(family['witness_path'])).hexdigest() == family['witness_sha256']
            matrix = json.loads(raw)
            lower, center, upper = (matrix[key] for key in ('lower_integer','center_integer','upper_integer'))
            assert len(lower) == family['negative_free_columns']
            # The weak free-budget theorem leaves at most ONE column.
            assert len(lower) <= 1
            if lower:
                assert len(lower[0]) == 1 and lower[0][0] > 0
                assert lower[0][0] == center[0][0] - matrix['radius_integer']
                assert upper[0][0] == center[0][0] + matrix['radius_integer'] + 64*matrix['residual_squared_upper_integer']
                pivots.append(str(F(lower[0][0], 2**matrix['bits'])))
            assert family['certified_lambda1_ge_floor']
    return {'u_exact':sign, 'positive_schur_pivots_exact':pivots,
            'lower':'-5187', 'audit':'Exact serialized weak lower-matrix positivity and archive binding; independent affine formation/cover replay055309 closes this antecedent under the pinned profile law',
            'not_original_strong_minus5123_completion':True}


def renamed_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {('source_connected_over_full_norm' if key == 'normalized_covariance' else key):renamed_tree(item)
                for key,item in value.items()}
    if isinstance(value, list):
        return [renamed_tree(item) for item in value]
    return value


def typed_epoch_transfers(metric: Any) -> dict[str, Any]:
    rows = [
        ('density', 'three_cell_density_formation_consumer_20261011.py',
         'three_cell_density_formation_typed3_20261011',
         'three_cell_density_formation_typed3_20261011_interface.json', 'own_source_sha256'),
        ('metric', 'three_cell_metric_transfer_consumer_20261011.py',
         'three_cell_metric_transfer_typed3_20261011',
         'three_cell_metric_transfer_typed3_20261011_receipt.json', 'own_source_SHA'),
        ('reference', 'root_pair_reference_parent_20261011.py',
         'root_pair_reference_parent_typed3_20261011',
         'root_pair_reference_parent_typed3_20261011_receipt.json', 'own_source_sha256'),
        ('density_control', 'density_formation_mathreview_control_20261011.py',
         'density_formation_mathreview_control_typed2_20261011',
         'density_formation_mathreview_control_typed2_20261011_receipt.json', 'own_source_sha256'),
        ('metric_control', 'source_metric_mathreview_control_20261011.py',
         'source_metric_mathreview_control_typed2_20261011',
         'source_metric_mathreview_control_typed2_20261011_receipt.json', 'own_source_SHA'),
        ('reference_control', 'reference_h0_mathreview_control_20261011.py',
         'reference_h0_mathreview_control_typed2_20261011',
         'reference_h0_mathreview_control_typed2_20261011_receipt.json', 'source_sha256'),
    ]
    loaded: dict[str, Any] = {}
    epochs: dict[str, Any] = {}
    for role, script, prefix, output, source_key in rows:
        exit_name = prefix + '_exit.json'
        record = read_json(exit_name)
        assert record['returncode'] == 0 and record['natural_exit']
        assert record['sources_inputs_unchanged'] and record['output'] == output
        source_sha = record['source_and_input_sha256'][script]
        assert digest(ROOT/script) == source_sha
        assert digest(ROOT/output) == record['output_sha256']
        assert digest(ROOT/record['executed_archive']) == record['executed_sha256']
        with zipfile.ZipFile(ROOT/record['executed_archive']) as archive:
            assert hashlib.sha256(archive.read(script)).hexdigest() == source_sha
            assert hashlib.sha256(archive.read(output)).hexdigest() == record['output_sha256']
            loaded[role] = archived_json(archive, output)
            archived_exit = archived_json(archive, exit_name)
            assert archived_exit['returncode'] == 0 and archived_exit['natural_exit']
        assert loaded[role][source_key] == source_sha
        epochs[role] = {'exit':exit_name, 'exit_sha256':PINS[exit_name],
            'source':script, 'source_sha256':source_sha,
            'receipt':output, 'receipt_sha256':record['output_sha256'],
            'executed_archive':record['executed_archive'],
            'executed_sha256':record['executed_sha256'],
            'natural_exit':True, 'elapsed_seconds':record['elapsed_seconds'],
            'frozen_source_bindings':record['frozen_source_bindings']}
    original_density = read_json('three_cell_density_formation_interface_20261011.json')
    for key in ('errors', 'common_N0_exact', 'source_norm_squared_exact',
                'operation_assumptions', 'source_norm_operation_audit'):
        assert loaded['density'][key] == original_density[key]
    assert loaded['metric']['classes'] == metric['classes']
    assert loaded['metric']['domain_prices'] == metric['domain_prices']
    assert loaded['metric']['density_interface_SHA'] == epochs['density']['receipt_sha256']
    original_reference = read_json('root_pair_reference_parent_attempt2_20261011_receipt.json')
    assert {k:v for k,v in loaded['reference'].items() if k != 'own_source_sha256'} == {
        k:v for k,v in original_reference.items() if k != 'own_source_sha256'}
    assert loaded['density_control']['entire_source_norm']['equals_interface_exact_rational']
    assert loaded['metric_control']['all164_interval_fields_exact_match']
    assert loaded['metric_control']['native_formation_is_conditional_not_historical_machine_attestation']
    assert loaded['reference_control']['sources_inputs_unchanged']
    return {'epochs':epochs, 'all164_intervals_and_source_prices_unchanged':True,
        'density_exact_norm_and_operation_prices_unchanged':True,
        'reference_algebra_unchanged':True,
        'original_proofs_and_verdicts_not_rebound':True,
        'controls_replay_frozen_producer_bytes_from_original_archives':True,
        'typing_and_output_path_changes_only':True,
        'scientific_operation_model_remains_conditional':True}


def dynamical_kernel_extension() -> dict[str, Any]:
    pins = read_json('three_cell_reference_kernel_parent_bindings_20261011.json')
    assert {name:digest(ROOT/name) for name in pins} == pins
    records = [
        ('charged_stieltjes_actual1_20261011', 'charged_stieltjes_scientist_20261011.py'),
        ('charged_kernel_review_typed5_20261011', 'charged_kernel_review_control_20261011.py'),
        ('ground_flux_cycle_typed3_20261011', 'ground_flux_cycle_scientist_20261011.py'),
    ]
    epochs = {}
    for prefix, script in records:
        exit_name, receipt_name = prefix+'_exit.json', prefix+'_receipt.json'
        archive_name = prefix+'_executed.zip'
        record, receipt = read_json(exit_name), read_json(receipt_name)
        assert record['returncode'] == 0 and record['natural_exit']
        assert record.get('sources_inputs_unchanged', record.get('source_and_inputs_unchanged'))
        with zipfile.ZipFile(ROOT/archive_name) as archive:
            assert hashlib.sha256(archive.read(receipt_name)).hexdigest() == pins[receipt_name]
            if script in archive.namelist():
                source_sha = hashlib.sha256(archive.read(script)).hexdigest()
            else:
                pre_bytes = archive.read(prefix+'_preexecution.zip')
                assert hashlib.sha256(pre_bytes).hexdigest() == record['preexecution_sha256']
                with zipfile.ZipFile(io.BytesIO(pre_bytes)) as pre:
                    source_sha = hashlib.sha256(pre.read(script)).hexdigest()
            assert source_sha == receipt['own_source_sha256'] == digest(ROOT/script)
        epochs[prefix] = {'exit':exit_name, 'exit_sha256':pins[exit_name],
            'source':script, 'source_sha256':source_sha, 'receipt':receipt_name,
            'receipt_sha256':pins[receipt_name], 'executed_archive':archive_name,
            'executed_sha256':pins[archive_name], 'natural_exit':True,
            'elapsed_seconds':record['elapsed_seconds']}
    review = read_json('charged_kernel_review_typed5_20261011_receipt.json')
    original_review = read_json('charged_kernel_review_actual4_20261011_receipt.json')
    assert {k:v for k,v in review.items() if k != 'own_source_sha256'} == {
        k:v for k,v in original_review.items() if k != 'own_source_sha256'}
    assert review['verdict'] == 'CONDITIONAL' and not review['load_bearing_findings']
    assert review['complete_commutator_cases'] == 32
    assert all(row['exact_labelled_1_plus_63_sum'] for row in review['aggregate_checks'])
    charged = read_json('charged_stieltjes_actual1_20261011_receipt.json')
    assert F(charged['source_residual_cap_exact']) == F(1,65536)
    assert F(charged['Rayleigh_ground_shift_cap_exact']) == F('46955575073/1538631694057472')
    allocation = 2*F(charged['aggregate']['0']['inverse_square']['exact'][1])
    assert allocation > 1
    flux = read_json('ground_flux_cycle_typed3_20261011_receipt.json')
    original_flux = read_json('ground_flux_cycle_actual2_20261011_receipt.json')
    changed_metadata = {'epoch','own_source_sha256','elapsed_seconds'}
    assert {k:v for k,v in flux.items() if k not in changed_metadata} == {
        k:v for k,v in original_flux.items() if k not in changed_metadata}
    controls = flux['controls']
    assert controls['N8_four_distinct_external_edges']
    assert F(controls['final_flow_energy_exact']) == 256
    assert F(controls['incoming_face_electric_difference_exact']) == 768
    assert controls['omitted_intermediate_twoPH_rejected']
    assert controls['exterior_full_norm_squared_exact'] == '1'
    assert F(controls['exterior_vector5_cropped_norm_squared_exact']) == F(1,2)
    real = list(map(F,flux['coefficient_enclosures']['0']['real']['exact']))
    assert real[0] < 0 < real[1]
    assert not flux['rooted_remainder_attempt']['gate_less_than_one']
    return {'epochs':epochs, 'conditional_review_no_load_bearing_findings':True,
        'new_typing_epochs_preserve_scientific_payloads':True,
        'original_review_and_producer_epochs_not_rebound':True,
        'charged_Stieltjes':charged['aggregate'],
        'one_edge_allocation2_inverse_square_exact':str(allocation),
        'one_edge_allocation2_guard_failed':True,
        'four_distinct_edge_flux_transition':flux['coefficient_enclosures'],
        'flux_sign_nonzero_certified':False,
        'rooted_sufficient_gate':flux['rooted_remainder_attempt'],
        'arithmetic_scope':'Exact outward rational bounds under stated operation, source, charged-sector and ground hypotheses. Not historical machine attestation.',
        'incoming_scope':'Only k=0; k=face external increment768, not256.',
        'no_physical1_phase_or_pole_admission':True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    own = digest(Path(__file__))
    assert {name:digest(ROOT/name) for name in PINS} == PINS
    affine_review = read_json('affine_lower_mathreview_verdict_20261011.json')
    assert affine_review['matching_H0_conditional_graph_eligible']
    assert affine_review['target_proof_sha256'] == PINS['affine_canonical_weak_proof_20261011.json']
    weak = [check_weak('affine_canonical_weak_actual1_20261011_executed.zip',
                      'affine_canonical_weak_actual1_20261011_receipt.json','1'),
            check_weak('affine_canonical_weak_minus_actual1_20261011_executed.zip',
                      'affine_canonical_weak_minus_actual1_20261011_receipt.json','-1')]
    source = exact_source_moments()
    with zipfile.ZipFile(ROOT/'three_cell_metric_transfer_actual1_20261011_executed.zip') as archive:
        old = archived_json(archive,'three_cell_metric_transfer_actual1_20261011_receipt.json')
    with zipfile.ZipFile(ROOT/'three_cell_metric_transfer_semantic2_20261011_executed.zip') as archive:
        metric = archived_json(archive,'three_cell_metric_transfer_semantic2_20261011_receipt.json')
        assert hashlib.sha256(archive.read('three_cell_metric_transfer_consumer_20261011.py')).hexdigest() == metric['own_source_SHA']
    assert renamed_tree(old['classes']) == metric['classes']
    assert len(metric['classes']) == 164
    typed = typed_epoch_transfers(metric)
    with zipfile.ZipFile(ROOT/'reference_matching_flux_spectral_parent_attempt2_20261011_executed.zip') as archive:
        spectral = archived_json(archive,'reference_matching_flux_spectral_parent_attempt2_20261011_receipt.json')
        assert hashlib.sha256(archive.read('reference_matching_flux_spectral_parent_20261011.py')).hexdigest() == spectral['own_source_sha256']
    boost = spectral['external_block_boost_control']
    assert boost['nonzero_external_block_shift_exact'] == 384
    assert boost['wrong_unrestricted_internal_only_shift_exact'] == 256
    gamma = -5123-F('-176071894972193/34359738368')
    assert gamma == F('46955312929/34359738368') > 0
    native_proof = read_json('three_cell_native_proof_20261011.json')
    native = read_json('three_cell_native_actual3_20261011_receipt.json')
    addendum = native_proof['complete_first_action_proof']['new_tighter_outward_SVD_bound']
    bounds = native_proof['complete_first_action_proof']['bounds_recorded']
    assert native['complete_action']['outward_sourceweighted_complete_firstH_upper_exact'] == bounds['outward_sourceweighted_complete_firstH_upper_exact']
    assert F(addendum['complete_firstH_norm_upper_exact']) < F(bounds['outward_sourceweighted_complete_firstH_upper_exact'])
    result = {'kind':'EXACT_SOURCE_MOMENTS_AND_REVIEWED_FORMATION_REFERENCE_EPOCH_INTEGRATION',
      'own_source_sha256':own, 'input_sha256':PINS, 'source_moments':source,
      'affine_weak_matrix_audit':weak,
      'conditional_reference_gap_exact':str(gamma),
      'reference_ground_proof_scope':'Independently reviewed frozen compact beta0 matching charged H0, entire affine +/-1 formation closed; unique product g/gap conditional on pinned exact profiles and accepted u0 binary64/source caps. Not physical1/fullPhi.',
      'metric_semantic_migration':{'all164_numeric_intervals_unchanged':True,
          'diagnostic':'t4C/F, NOT normalized-state sector covariance t4C/F^2',
          'historical_native_shared_normalized_covariance_t':'Frozen historical name only; interpret as diagnostic/F, not a probability covariance.'},
      'source_restricted_Sa':metric['domain_prices'],
      'typed_epoch_transfers':typed,
      'dynamical_kernel_extension':dynamical_kernel_extension(),
      'spectral_classes':spectral['spectral_classes'], 'boost_adverse':boost,
      'action_prices':{'primary_executed_price_exact':bounds['outward_sourceweighted_complete_firstH_upper_exact'],
          'separate_analytic_SVD_addendum_exact':addendum['complete_firstH_norm_upper_exact'],
          'scope':'Complete finite N8 frozen CHARGED-body FIRST-H norm upper, not variance, residual, fullPhi action or N-uniform linked bound',
          'old_stale_number_causal_explanation_withdrawn':True,
          'source_moment_caps_now_exact_for_stored_s':True},
      'stage':'PARENT_CONSUMER_NOT_SCIENTIFIC_ACCEPTANCE',
      'physical1_massless_inference':False,
      'mechanism':'A gap-preserving continuation of this gapped reference to physical1 would exclude masslessness, not prove it. Charged dynamical and gap-closing/IR mechanism remain open.',
      'nonclaims':['physicalPhi/lattice ground','physical1 phase','normalized N-uniform linked stability','stiffness','IRatom','commoncone','originalSOURCE','campaign closure'],
      'elapsed_seconds':time.monotonic()-started}
    assert digest(Path(__file__)) == own and {name:digest(ROOT/name) for name in PINS} == PINS
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':args.output,'elapsed_seconds':result['elapsed_seconds'],
                      'rho2':float(F(source['rho2_exact'])),'rho4':float(F(source['rho4_exact'])),
                      'reference_gap':float(gamma),'physical1_massless_inference':False}))


if __name__ == '__main__':
    main()

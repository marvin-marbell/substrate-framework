"""Compact observed physics and exact local source/input/result delivery capsule."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
P = 'rich_review_wave32_20261015_'


def digest(path: Path) -> dict:
    return dict(file=path.name,bytes=path.stat().st_size,SHA256=hashlib.sha256(path.read_bytes()).hexdigest())


def delta(final: dict, initial: dict, keys: list) -> dict:
    return {key:(np.asarray(final[key])-np.asarray(initial[key])).tolist() for key in keys}


def main() -> None:
    result = json.loads((ROOT/(P+'result.json')).read_text())
    receipt = json.loads((ROOT/(P+'exit.json')).read_text())
    if not (receipt['natural_exit'] and receipt['returncode']==0 and receipt['sources_and_inputs_unchanged']):
        raise RuntimeError('Successful exact natural runtime required for observed synthesis')
    axis = np.asarray(result['source_axis'])
    branch_table = {}
    controls = {}
    keys = ['physical_charge','local_frame_spin','local_matrix_norm2','material_angular','electric_flux','physical_link_charge_current','electric_power']
    for branch,sign in [('addition',1),('removal',-1)]:
        item = result['branches'][branch]
        full = item['variants']['full']
        scalar = item['variants']['vector_off']
        initial,final = full['observations'][0],full['observations'][-1]
        excess0,excess = full['excess_over_independently_evolving_neutral'][0],full['excess_over_independently_evolving_neutral'][-1]
        detector = final['receiver_detector']
        receiver = detector['receiver']
        count = 3 if branch=='addition' else 1
        neutral_final = result['neutral']['full']['observations'][-1]
        branch_table[branch] = dict(source=item['source_input'],
            initial_signed_excitation_number=(sign*np.asarray(excess0['local_number'])).tolist(),
            final_signed_excitation_number=(sign*np.asarray(excess['local_number'])).tolist(),
            signed_excitation_number_CHANGE_after_preparation=(sign*(np.asarray(excess['local_number'])-np.asarray(excess0['local_number']))).tolist(),
            initial_twice_excess_local_spin_axis=(2*np.asarray(excess0['local_frame_spin'])@axis).tolist(),
            final_twice_excess_local_spin_axis=(2*np.asarray(excess['local_frame_spin'])@axis).tolist(),
            initial_material_axis=float(np.asarray(excess0['material_angular'])@axis),
            final_material_axis=float(np.asarray(excess['material_angular'])@axis),
            final_total_J_axis=float(np.asarray(excess['total_angular'])@axis),
            initial_excess_matrixnorm2=excess0['local_matrix_norm2'],
            final_excess_matrixnorm2=excess['local_matrix_norm2'],
            raw_same_state_own_t0_change=delta(final,initial,keys),
            final_excess_over_own_neutral={key:excess[key] for key in keys},
            final_excess_over_matched_extra_filter_neutral=full['excess_over_matched_extra_filter_neutral'][-1],
            final_charge_spin_connected_covariance=excess['signed_charge_spin_covariance'],
            final_charge_norm_connected_covariance=excess['signed_charge_matrixnorm_covariance'],
            receiver_detector=dict(probability=detector['probability'],spin_plus_probability=detector['spin_plus_probability'],
                spin_minus_probability=detector['spin_minus_probability'],rest_probability=detector['rest_probability'],
                neutral_same_count_probability=neutral_final['local_count_PVM'][receiver][count],
                count_selected_raw_subnormalized=detector['count']['raw_subnormalized'],
                spin_plus_raw_subnormalized=detector['spin_plus']['raw_subnormalized'],
                spin_minus_raw_subnormalized=detector['spin_minus']['raw_subnormalized'],
                rest_raw_subnormalized=detector['rest']['raw_subnormalized'],
                full_quantum_norm=final['quantum_norm'],
                count_selected_quantum_norm=detector['count']['quantum_norm'],
                spin_plus_quantum_norm=detector['spin_plus']['quantum_norm'],
                rest_quantum_norm=detector['rest']['quantum_norm'],
                qualification='Real detector probabilities; background-contaminated full Fock count/spin channel, NOT probability the added particle is here. Conditional values only in raw result with own probability. No intermediate projection.'),
            full_initial_energy=full['initial_energy'])
        controls[branch] = dict(definition='SAME t0 ray; owning vector hopping removed, scalar hopping retained.',
            charged_full_minus_vectoroff=delta(final,scalar['observations'][-1],keys),
            excess_response_difference=delta(excess,scalar['excess_over_independently_evolving_neutral'][-1],keys),
            full_initial_own_H_rates=initial['rates'],vectoroff_initial_own_H_rates=scalar['observations'][0]['rates'],
            full_actual_edge_contact=full['actual_edge_contact'],vectoroff_actual_edge_contact=scalar['actual_edge_contact'],
            vectoroff_local_spin_own_t0_change=(np.asarray(scalar['observations'][-1]['local_frame_spin'])-np.asarray(scalar['observations'][0]['local_frame_spin'])).tolist(),
            vectoroff_material_own_t0_change=(np.asarray(scalar['observations'][-1]['material_angular'])-np.asarray(scalar['observations'][0]['material_angular'])).tolist())
    trajectories = list(result['neutral'].values())+list(result['matched_neutral'].values())+[case for branch in result['branches'].values() for case in branch['variants'].values()]
    samples = [value for case in trajectories for value in case['observations']]
    charged_samples = [value for branch in result['branches'].values() for case in branch['variants'].values() for value in case['observations']]
    numeric = dict(maximum_state_coarse_fine_difference=max(max(case['accuracy']['same_state_accuracy_comparison']['state_norm_differences']) for case in trajectories),
        accuracy_comparisons={name:case['accuracy'] for name,case in result['neutral'].items()},
        charged_accuracy_comparisons={branch:{name:case['accuracy'] for name,case in item['variants'].items()} for branch,item in result['branches'].items()},
        maximum_norm_error=max(max(case['accuracy']['norm_errors']) for case in trajectories),
        maximum_energy_drift=max(max(abs(value['energy']-case['observations'][0]['energy']) for value in case['observations']) for case in trajectories),
        maximum_actual_state_continuity_residual=max(float(np.max(abs(np.asarray(value['continuity_residual_on_actual_state'])))) for value in samples),
        maximum_detector_spin_idempotence=max(value['receiver_detector']['spin_projector_idempotence_on_state'] for value in charged_samples),
        any_link_Ecap_probability_range=[min(value['any_link_cap_probability'] for value in samples),max(value['any_link_cap_probability'] for value in samples)],
        any_outer_radial_node_probability_range=[min(value['quantum_norm']['any_outer_node_probability'] for value in samples),max(value['quantum_norm']['any_outer_node_probability'] for value in samples)],
        qualification='Finite-H real-time approximation only. High radial-edge or Ecap populations are substantive regulator limits, not a no-particle theorem. No separate absolute filter-strength error bound.')
    language = json.loads((ROOT/(P+'language_final.json')).read_text())
    joint_channels = {}
    for branch,item in result['branches'].items():
        detector = item['variants']['full']['observations'][-1]['receiver_detector']
        channels = {}
        for name in ['spin_plus','spin_minus','rest']:
            value = detector[name]
            probability = value['probability']
            raw = value['raw_subnormalized']
            channels[name] = dict(probability=probability,
                raw_material_axis=float(np.asarray(raw['material_angular'])@axis),
                conditional_material_axis=float(np.asarray(raw['material_angular'])@axis/probability),
                raw_electric_flux=raw['electric_flux'],
                conditional_electric_flux=(np.asarray(raw['electric_flux'])/probability).tolist(),
                raw_matrixnorm2=raw['local_matrix_norm2'],
                conditional_matrixnorm2=(np.asarray(raw['local_matrix_norm2'])/probability).tolist())
        joint_channels[branch] = channels
    synthesis = dict(kind='EXECUTED_INDEPENDENT_NEW_DRESSED_BRANCH_AND_QUANTUM_FIELD_DETECTOR_REVIEW',
        verdict='New full-H energy-filtered states of both charge signs show actual charge AND measured physical local spin redistribution after the dressed t0 preparation, with quantum link-current/field response and separately carried material angular. Same-state vector removal retains scalar spin transport but removes further material-angular exchange and changes currents/norm response. Receiver count/spin/field joints resolve BOTH spin signs with distinct material/electric correlations and substantial rest weight; virtual-sea background means this is not an exclusive whole-particle PVM or unique coherent/asymptotic species identification. Initial filtering-induced spread and extra neutral cooling are quantified separately. Ambient norm need not move rigidly.',
        operator_review=P+'raw_review.json',new_input=result['declared_shared_input'],source_axis=result['source_axis'],times=result['times'],
        source_spinor_real=result['source_spinor_real'],source_spinor_imag=result['source_spinor_imag'],
        charged_branch_findings=branch_table,same_state_causal_H_controls=controls,
        receiver_spin_field_channels=joint_channels,
        neutral={name:dict(initial_energy=case['initial_energy'],own_t0_final_change=delta(case['observations'][-1],case['observations'][0],keys)) for name,case in result['neutral'].items()},
        matched_background_filter=result['matched_background_filter'],
        matched_background_trajectory_difference={name:[delta(matched,original,keys) for matched,original in zip(case['observations'],result['neutral'][name]['observations'])] for name,case in result['matched_neutral'].items()},
        numerical=numeric,diagnostics=dict(summary=language['summary'],returncode=language['returncode'],errors=language['error_counts_by_file'],LSP_exit=language['rooted_LSP']['server_exit'],type_clean_claim=False),
        exit=dict(file=P+'exit.json',natural_exit=True,returncode=0,elapsed_seconds=receipt['elapsed_seconds'],sources_and_inputs_unchanged=True),
        restrictions='No Nature electron/positron, photon, asymptotic species, full-half-line/regulator convergence, unique stable body, continuum, freeCOM, selfbinding or Lorentz claim. r² is not position and all ambient norm need not travel. Retained material angular is not suppressed.',
        artifacts=dict(result=P+'result.json',raw_states=P+'states.npz',source_capsule=P+'source_capsule.zip',language=P+'language_final.json'))
    (ROOT/(P+'synthesis.json')).write_text(json.dumps(synthesis,indent=2)+'\n')
    compact = dict(kind='COMPACT_ACTUAL_NEW_DRESSED_CARRIER_PHYSICAL_FINDINGS',
        verdict=synthesis['verdict'],input=synthesis['new_input'],axis=result['source_axis'],times=result['times'],
        phase_correction='Reviewer found literal Jminus B0=-B1; producer preserved failed attempt1 and corrected actual source. Own actual physical axis residuals retained.',
        matched_background={key:value for key,value in result['matched_background_filter'].items() if key!='accuracy'},
        matched_background_full_final_delta=synthesis['matched_background_trajectory_difference']['full'][-1],
        branches={branch:{key:value for key,value in item.items() if key not in ['receiver_detector','final_charge_spin_connected_covariance','final_charge_norm_connected_covariance','final_excess_over_matched_extra_filter_neutral']} for branch,item in branch_table.items()},
        detector={branch:{key:value for key,value in item['receiver_detector'].items() if key in ['probability','spin_plus_probability','spin_minus_probability','rest_probability','neutral_same_count_probability','count_selected_raw_subnormalized','spin_plus_raw_subnormalized','spin_minus_raw_subnormalized']} for branch,item in branch_table.items()},
        receiver_spin_field_channels=joint_channels,
        controls=controls,numerical=numeric,diagnostics=synthesis['diagnostics'],exit=synthesis['exit'],
        restrictions=synthesis['restrictions'],raw_result=P+'result.json',synthesis=P+'synthesis.json')
    (ROOT/(P+'physical_findings.json')).write_text(json.dumps(compact,indent=2)+'\n')
    selected = sorted(path for path in ROOT.glob(P+'*') if path.is_file() and path.name not in [P+'manifest.json',P+'delivery_capsule.zip',P+'delivery_receipt.json'])
    manifest = dict(kind='WAVE32_EXACT_LOCAL_ARTIFACT_MANIFEST',artifacts=[digest(path) for path in selected],upstream_bound_inside_source_capsule=receipt['input_source'])
    manifest_path = ROOT/(P+'manifest.json')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    capsule = ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in selected+[manifest_path]:
            archive.write(path,path.name)
    delivery = dict(delivery_capsule=digest(capsule),manifest=digest(manifest_path),synthesis=digest(ROOT/(P+'synthesis.json')),
        scope='LOCAL only; parent owns plan/synthesis; no graph/public/merge/push/settings/services changes.')
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(delivery,indent=2)+'\n')
    print(json.dumps(dict(delivery=delivery,numerical=numeric,branches={branch:{key:value for key,value in item.items() if key not in ['receiver_detector','final_charge_spin_connected_covariance','final_charge_norm_connected_covariance']} for branch,item in branch_table.items()},controls=controls)),flush=True)


if __name__ == '__main__':
    main()

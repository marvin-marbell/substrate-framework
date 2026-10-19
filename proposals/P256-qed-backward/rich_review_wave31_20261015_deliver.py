"""Wave31 synthesis plus source/input/result capsules and exact manifest."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave31_20261015_'


def digest(path: Path) -> dict:
    return dict(file=path.name,bytes=path.stat().st_size,SHA256=hashlib.sha256(path.read_bytes()).hexdigest())


def maximum(value: Any) -> float:
    return float(np.max(np.abs(np.asarray(value))))


def main() -> None:
    result = json.loads((ROOT/(PREFIX+'result.json')).read_text())
    exit_receipt = json.loads((ROOT/(PREFIX+'exit.json')).read_text())
    if not exit_receipt['natural_exit'] or exit_receipt['returncode'] != 0 or not exit_receipt['sources_and_inputs_unchanged']:
        raise RuntimeError('Cannot deliver successful-result synthesis without exact successful exit')
    config = result['config']
    axis = np.array(config['source_spin_axis'])
    table = {}
    controls = {}
    for branch,sign in [('addition',1),('removal',-1)]:
        item = result['branches'][branch]
        full = item['variants']['full']
        excess = full['signed_excess_over_evolving_neutral'][-1]
        initial = full['signed_excess_over_evolving_neutral'][0]
        spin_profile = 2*np.array(excess['local_frame_spin']) @ axis
        charge_profile = sign*np.array(excess['local_number'])
        control = {}
        for variant in ['path_cut','vector_off']:
            case = item['variants'][variant]
            change = case['own_initial_changes'][-1]
            delta = case['signed_excess_over_evolving_neutral'][-1]
            control[variant] = dict(source_site_own_number_change=change['local_number'][0],
                excess_number=delta['local_number'],excess_frame_spin=delta['local_frame_spin'],
                excess_matrix_norm2=delta['local_matrix_norm2'],
                own_local_spin_change=change['local_frame_spin'],own_material_angular_change=change['material_angular'])
        table[branch] = dict(physical_total_charge=item['physical_total_charge'],raw_source_strength=item['source']['raw_strength'],
            final_excess_number=excess['local_number'],final_excess_physical_charge=excess['physical_charge'],
            final_excess_local_spin=excess['local_frame_spin'],final_excess_local_casimir=excess['local_frame_casimir'],
            final_excess_matrix_norm2=excess['local_matrix_norm2'],
            excess_matrix_norm_change_from_own_t0=full['excess_changes_from_own_t0'][-1]['local_matrix_norm2'],
            source_axis_spin_twice=spin_profile.tolist(),signed_excitation_number_profile=charge_profile.tolist(),
            signed_spin_charge_profile_L1_discrepancy=float(np.sum(abs(spin_profile-charge_profile))),
            final_excess_material_angular=excess['material_angular'],final_excess_total_angular=excess['total_angular'],
            initial_excess_number=initial['local_number'],initial_excess_spin=initial['local_frame_spin'],
            final_charge_spin_connected_covariance=excess['signed_charge_spin_covariance'],
            final_charge_matrixnorm_connected_covariance=excess['signed_charge_matrixnorm_covariance'],
            final_local_count_PVM=full['samples'][-1]['local_count_PVM'],
            final_joint_local_count_spin=full['samples'][-1]['joint_local_count_spin'],
            final_joint_local_count_casimir=full['samples'][-1]['joint_local_count_spin_casimir'],
            final_root_upper_population=full['samples'][-1]['root_upper_population'],
            final_root_lower_population=full['samples'][-1]['root_lower_population'])
        controls[branch] = control
    field = result['fields']['addition_full']
    final_field = field['pulse_minus_baseline'][-1]
    initial_field = field['pulse_minus_baseline'][0]
    comparison = result['branches']['addition']['variants']['full']['propagation']['same_state_accuracy_comparison']
    trajectories = list(result['neutral'].values())
    trajectories += [case for branch in result['branches'].values() for case in branch['variants'].values()]
    trajectories.append(field)
    numerical = dict(decision_changing_same_state_comparison=comparison,
        maximum_norm_error=max(maximum(case['propagation']['norm_errors']) for case in trajectories),
        maximum_energy_drift=max(maximum([sample['energy']-case['samples'][0]['energy'] for sample in case['samples']]) for case in trajectories),
        maximum_actual_state_charge_continuity_residual=max(maximum(sample['continuity_residual_on_actual_state']) for case in trajectories for sample in case['samples']),
        max_source_local_Noether_residual=max(maximum(branch['source']['local_source_Noether_commutator_residuals']) for branch in result['branches'].values()),
        electric_cap_population_range=[min(sample['any_link_cap_probability'] for case in trajectories for sample in case['samples']),max(sample['any_link_cap_probability'] for case in trajectories for sample in case['samples'])],
        qualification='Accuracy comparison resolves selected same-state finite-H propagation only; no radial, angular, electric-cap or continuum convergence claim.')
    field_summary = dict(pure_field_generator=field['generator'],alpha=field['alpha'],
        initial_own_H_work=field['initial_injected_own_H_work'],
        initial_charge_change=initial_field['physical_charge'],initial_spin_change=initial_field['local_frame_spin'],
        initial_norm_change=initial_field['local_matrix_norm2'],
        final_physical_link_charge_current_response=final_field['physical_link_charge_current'],
        final_electric_flux_response=final_field['electric_flux'],
        final_electric_power_response=final_field['electric_power'],
        final_charge_response=final_field['physical_charge'],
        final_local_spin_response=final_field['local_frame_spin'],
        final_material_angular_response=final_field['material_angular'],
        final_matrix_norm_response=final_field['local_matrix_norm2'])
    synthesis = dict(kind='EXECUTED_INDEPENDENT_PRODUCT_LOCAL_CARRIER_RAW_REVIEW_AND_NEW_RESPONSE',
        governing_contract=dict(plan=config['plan'],captured_revision=config['captured_revision'],parent_sole_plan_writer=True),
        physical_interpretation='Independent physical g_v are present before reduction. Product RIGHT-Haar sewing yields local LEFT_v=S_v; local cone measure r³dr and I=M r²/2 produce [3/8+S_v²]/(M r_v²). Exact raw224→16 LEFT/source operators feed the disclosed orbital constitutive root. This is genuine local provenance, not an arbitrary decomposition of a global spin generator.',
        observed_decision='The specified full-root source evolves into a moving excess charge with distributed physical local frame spin and a separately responding internal matrix-norm cloud. Numerical signed profiles are reported without forcing one rigid spin/charge/norm object. Nonzero spin/charge profile discrepancy or source-local radial memory defeats exact local rigid co-motion in this finite experiment, not every quasiparticle interpretation. Multiple components remain allowed.',
        times=config['times'],phase=config['phase'],parameters=config['parameters'],
        shared_input=result['shared_neutral_input'],
        source_model_input_and_exit_binding=dict(model_SHA256=config['expected_model_SHA256'],neutral_basis_SHA256=result['neutral_basis_SHA256'],exit_file=PREFIX+'exit.json',natural_exit=True,returncode=0,inputs_unchanged=True),
        raw_operator_provenance=result['primitive_operator_provenance'],
        neutral_nonstationarity={key:value['reference_nonstationarity_norm'] for key,value in result['neutral'].items()},
        source_and_local_response=table,physical_controls=controls,field_current_result=field_summary,numerical_resolution=numerical,
        angular_accounting='Material angular expectations are separately reported with all local physical frame spins. U1 links are scalar edge fields attached to material paths; no independent continuum E×B angular or photon-spin operator is asserted. Conserved total J alone is not used as evidence of local co-motion.',
        semantics='P(n_v=k), k0..4 is an actual local count PVM, not the added-particle position PVM. <P(n_v=k)S_v> and connected Q-S/Q-r² values are signed measurements. Initial correlations are never labelled transport; own-t0 and evolving neutral subtractions are retained in raw result. r² is matrix norm, never spatial radius. Hole is not an observed positron.',
        source_inspection='Raw root library PW j0,.5,1 times local16 has right Gauss projection and actual D(g)c source, followed by independently applied site copies. Owning model all masks N5/6/7 and all Ecap2 Gauss solutions; ions2 and boundary6-N. No positive-energy, spinful-Mott, one-pair or fixed-free-Dirac reduction. Both ±root orbitals are retained and upper/lower populations measured.',
        limitations='Conditional finite full-root triangle with chosen frame/sewing/CAR, metric, root coupling, relative-frame constitutive transport and regulators. Prepared sea is not a stationary interacting vacuum. No unique stable elementary particle, spatial selfbinding, freeCOM, Lorentz, continuum Maxwell, Nature electron/positron/proton or periodic3D photons. Matrix norm response is not physical particle size.',
        artifacts=dict(result=PREFIX+'result.json',states=PREFIX+'states.npz',source_capsule=PREFIX+'source_capsule.zip',exit=PREFIX+'exit.json',language=PREFIX+'language_final.json'))
    (ROOT/(PREFIX+'synthesis.json')).write_text(json.dumps(synthesis,indent=2)+'\n')
    selected = sorted(path for path in ROOT.glob(PREFIX+'*') if path.is_file() and path.name not in [PREFIX+'manifest.json',PREFIX+'delivery_capsule.zip',PREFIX+'delivery_receipt.json'])
    manifest = dict(kind='WAVE31_EXACT_ARTIFACT_MANIFEST',artifacts=[digest(path) for path in selected],
                    upstream_bound_inside_source_capsule=exit_receipt['input_source'])
    manifest_path = ROOT/(PREFIX+'manifest.json')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    capsule = ROOT/(PREFIX+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in selected+[manifest_path]:
            archive.write(path,path.name)
    receipt = dict(delivery_capsule=digest(capsule),manifest=digest(manifest_path),synthesis=digest(ROOT/(PREFIX+'synthesis.json')),
                   publication='Local only. No plan, graph, public state, merge, push or service action.')
    (ROOT/(PREFIX+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(delivery=receipt,table=table,field=field_summary,numerical=numerical)),flush=True)


if __name__ == '__main__':
    main()

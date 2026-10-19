"""New independent full-H evolution of declared owning dressed input, both charges."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
from rich_local_frame_carrier_20261015_model import Model
from rich_dressed_local_carrier_20261015_source import cool
from rich_review_wave31_20261015_krylov import propagate
from rich_review_wave31_20261015_observe import subtract
from rich_review_wave32_20261015_observe import observe

ROOT = Path(__file__).resolve().parent
P = 'rich_review_wave32_20261015_'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def energy(h: Any, state: Any) -> dict:
    action = h @ state
    mean = float(np.vdot(state, action).real)
    return dict(mean=mean, variance=float(np.vdot(action-mean*state,action-mean*state).real))


def angular_checks(model: Any, h: Any, state: Any) -> dict:
    lower = model.J[0]-1j*model.J[1]
    return dict(total_Jhalf_state_residual=float(np.linalg.norm(sum(a@(a@state) for a in model.J)-.75*state)),
                owning_H_Jlower_commutator_on_state=float(np.linalg.norm(h@(lower@state)-lower@(h@state))))


def main() -> None:
    start = time.monotonic()
    config = json.loads((ROOT/(P+'input.json')).read_text())
    for name, expected in config['expected_input_SHA256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Input source binding changed: '+name)
    interface = json.loads((ROOT/config['interface_file']).read_text())
    parameters = interface['parameters']
    times = config['times']
    axis = np.asarray(config['source_axis'],float)
    neutral_model = Model(6,parameters)
    one_spin = neutral_model.architecture['one_spin']
    physical_generator = sum(axis[a]*one_spin[a][:2,:2] for a in range(3))
    eigenvalues, eigenvectors = np.linalg.eigh(physical_generator)
    coefficients = eigenvectors[:,np.argmax(eigenvalues)]
    if neutral_model.basis_hash != interface['neutral']['basis_hash']:
        raise ValueError('Neutral physical basis changed')
    with np.load(ROOT/interface['state_file']) as archive:
        neutral = archive['neutral'].astype(complex)
    output: dict[str,Any] = dict(kind='NEW_INDEPENDENT_DRESSED_CHARGE_AND_FIELD_JOINT_DYNAMICS',
        declared_shared_input=dict(interface=config['interface_file'],state_file=interface['state_file'],
            state_SHA256=sha(ROOT/interface['state_file']), provenance='NEW owner-computed input only. Not independently solved vacuum, not old evolved output.'),
        source_axis=axis.tolist(),source_spinor_real=coefficients.real.tolist(),source_spinor_imag=coefficients.imag.tolist(),
        times=times, parameters=parameters, neutral={},matched_neutral={},branches={},
        neutral_total_J_residual=[float(np.linalg.norm(a@neutral)) for a in neutral_model.J])
    saved: dict[str,Any] = dict(declared_neutral_t0=neutral)
    hn, _, _ = neutral_model.law('full')
    matched_states, matched_logs, matched_accuracy = cool(hn,neutral,[interface['selected_source_tau']],
        config['matched_filter_dimension'],config['matched_filter_comparison_dimension'],interface['filter_shift'])
    matched = matched_states[0]
    overlap = complex(np.vdot(neutral,matched))
    trace_distance = float(np.sqrt(max(0.,1-abs(overlap)**2)))
    output['matched_background_filter'] = dict(tau=interface['selected_source_tau'],shift=interface['filter_shift'],
        normalized_ray_distance=float(np.linalg.norm(matched-neutral)),overlap_squared=float(abs(overlap)**2),
        pure_state_trace_distance=trace_distance,any_effect_probability_difference_bound=trace_distance,
        norm1_operator_expectation_difference_bound=2*trace_distance,
        local_matrixnorm2_difference_bound=float((max(neutral_model.r**2)-min(neutral_model.r**2))*trace_distance),
        log_raw_output_norm=matched_logs[0],accuracy=matched_accuracy,
        before=energy(hn,neutral),after=energy(hn,matched),
        meaning='Independent SAME extra imaginary tau on declared neutral with actual H6. Quantifies nonstationary-background cooling, not independent ground or unique attachment decomposition. Bound remains valid after identical unitary evolution.')
    saved['matched_extra_cooled_neutral_t0'] = matched
    del hn,matched_states
    neutral_observations = {}
    matched_observations = {}
    for variant in config['variants']:
        h, currents, _ = neutral_model.law(variant)
        states, accuracy = propagate(h,neutral,times,config['krylov_dimension'],config['comparison_dimension'])
        observations = [observe(state,neutral_model,h,currents,axis,config['receiver']) for state in states]
        neutral_observations[variant] = observations
        output['neutral'][variant] = dict(initial_energy=energy(h,neutral),accuracy=accuracy,observations=observations)
        for index,state in enumerate(states):
            saved[f'neutral_{variant}_t{index}'] = state
        extra_states, extra_accuracy = propagate(h,matched,times,config['krylov_dimension'],config['comparison_dimension'])
        extra_observations = [observe(state,neutral_model,h,currents,axis,config['receiver']) for state in extra_states]
        matched_observations[variant] = extra_observations
        output['matched_neutral'][variant] = dict(initial_energy=energy(h,matched),accuracy=extra_accuracy,observations=extra_observations)
        for index,state in enumerate(extra_states):
            saved[f'matched_neutral_{variant}_t{index}'] = state
        del extra_states
        del h, currents, states
        gc.collect()
    for branch,number in [('addition',7),('removal',5)]:
        model = Model(number,parameters)
        info = interface['branches'][branch]
        if model.basis_hash != info['basis_hash']:
            raise ValueError('Charged physical basis changed')
        with np.load(ROOT/interface['state_file']) as archive:
            components = [archive[f'{branch}_component{component}'].astype(complex) for component in range(2)]
        logs = np.asarray(info['component_log_raw_norm'])
        common_log = float(max(logs))
        raw_scaled = sum(coefficients[i]*np.exp(logs[i]-common_log)*components[i] for i in range(2))
        scaled_norm = float(np.linalg.norm(raw_scaled))
        source = raw_scaled/scaled_norm
        bare_components = [model.source_from(neutral_model,interface['source_site'],branch,component=i,orbital=interface['orbital'])@neutral for i in range(2)]
        bare = sum(coefficients[i]*bare_components[i] for i in range(2))
        bare_strength = float(np.vdot(bare,bare).real)
        dressed_log_strength = 2*(common_log+np.log(scaled_norm))
        lowered = (model.J[0]-1j*model.J[1])@bare_components[0]
        lowering_coefficient = np.vdot(bare_components[1],lowered)/np.vdot(bare_components[1],bare_components[1])
        axis_generator = sum(axis[a]*model.J[a] for a in range(3))
        output['branches'][branch] = dict(number=number,source_input=dict(
            normalized_ray_norm=float(np.vdot(source,source).real), component_log_raw_norm=logs.tolist(),
            shared_component_Gram_real=np.array([[np.vdot(a,b) for b in components] for a in components]).real.tolist(),
            shared_component_Gram_imag=np.array([[np.vdot(a,b) for b in components] for a in components]).imag.tolist(),
            direct_own_CAR_source_strength=bare_strength, filtered_log_strength=float(dressed_log_strength),
            filtered_strength=float(np.exp(dressed_log_strength)),
            physical_meaning='Filter norm is raw operator strength, NOT detector probability or CAR event brightness. No contraction implementation claimed.',
            bare_to_dressed_overlap_squared=float(abs(np.vdot(bare/np.sqrt(bare_strength),source))**2),
            direct_measured_lowering_coefficient=[float(lowering_coefficient.real),float(lowering_coefficient.imag)],
            direct_corrected_partner_residual=float(np.linalg.norm(lowered/lowering_coefficient-bare_components[1])),
            dressed_actual_physical_axis_residual=float(np.linalg.norm(axis_generator@source-.5*source)),
            bare_actual_physical_axis_residual=float(np.linalg.norm(axis_generator@bare-.5*bare)),
            new_preparation_not_causal_control=True),variants={})
        saved[branch+'_source_t0'] = source
        saved[branch+'_raw_scaled_source'] = raw_scaled
        saved[branch+'_raw_scaled_source_log_multiplier'] = np.array(common_log)
        saved[branch+'_direct_bare_source'] = bare
        for variant in config['variants']:
            h, currents, contacts = model.law(variant)
            states, accuracy = propagate(h,source,times,config['krylov_dimension'],config['comparison_dimension'])
            observations = [observe(state,model,h,currents,axis,config['receiver'],branch) for state in states]
            excess = []
            for value,baseline in zip(observations,neutral_observations[variant]):
                # Conditional probabilities are not subtracted into new probabilities.
                primitive_keys = ['physical_charge','local_number','local_frame_spin','local_frame_casimir',
                    'local_matrix_norm2','material_angular','total_angular','electric_flux',
                    'physical_link_charge_current','electric_energy','electric_power',
                    'signed_charge_spin_covariance','signed_charge_matrixnorm_covariance','local_count_PVM']
                excess.append(subtract({key:value[key] for key in primitive_keys},
                                       {key:baseline[key] for key in primitive_keys}))
            matched_excess = [subtract({key:value[key] for key in primitive_keys},{key:baseline[key] for key in primitive_keys})
                              for value,baseline in zip(observations,matched_observations[variant])]
            result = dict(initial_energy=energy(h,source),angular_checks=angular_checks(model,h,source),
                accuracy=accuracy,observations=observations,excess_over_independently_evolving_neutral=excess,
                excess_over_matched_extra_filter_neutral=matched_excess,
                actual_edge_contact=[float(np.vdot(states[-1],op@states[-1]).real) for op in contacts])
            output['branches'][branch]['variants'][variant] = result
            for index,state in enumerate(states):
                saved[f'{branch}_{variant}_t{index}'] = state
            del h, currents, contacts, states
            gc.collect()
        (ROOT/(P+'partial_result.json')).write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps(dict(stage='actual_new_branch',branch=branch,
            full_final_excess=output['branches'][branch]['variants']['full']['excess_over_independently_evolving_neutral'][-1])),flush=True)
        del model,components,bare_components,source,bare,raw_scaled
        gc.collect()
    output['semantics'] = dict(
        detector='Full local count P(n_receiver=3 for N7,1 for N5), spin-axis +/- half. Can trigger on virtual-sea configurations; NOT complete added-particle PVM. Neutral raw same count probability retained.',
        joint='Raw subnormalized detector count/spin, field norm density, E and material angular. Conditional values always accompanied by actual full-state probability and rest weight.',
        current='q=-1 current=-i(T-Tdag), dotE=-current and dotQ=-incidence current. Detector current is final-measurement PJP, not simultaneous count/current eigenvalues.',
        intervention='SAME initial charged and neutral rays under full vs vector_off owning H. Scalar transport remains. Different H is causal vector/contact intervention, unlike prior state energy filtering.',
        numerical='Same-state coarse/fine fully reorthogonalized full-H Lanczos. Not regulator, radial or continuum convergence.',
        field='r2 is quantum matrix norm, not particle position or radius. Ambient/tails need not move rigidly. Material angular momentum retained separately.',
        limits='Finite triangle, cap2 link fields,3 radial nodes/site, material l0+l1. No photon, Nature, asymptotic species, continuum, selfbinding or unique-body claim.')
    output['elapsed_seconds'] = time.monotonic()-start
    np.savez_compressed(ROOT/(P+'states.npz'),**saved)
    (ROOT/(P+'result.json')).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(complete=True,elapsed_seconds=output['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    main()

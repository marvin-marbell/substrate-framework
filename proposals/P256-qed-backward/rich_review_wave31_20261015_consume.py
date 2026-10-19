"""Independent NEW product-local source/field evolution using the declared shared input."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import expm_multiply
from rich_local_frame_carrier_20261015_model import Model, INCIDENCE
from rich_review_wave31_20261015_krylov import propagate
from rich_review_wave31_20261015_observe import measurements, subtract

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave31_20261015_'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def operators(model: Any) -> dict:
    return dict(spin=model.S_local, spin2=model.S2_local, norm2=model.r2_local,
                material=model.material_L, total=model.J, electric=model.E,
                electric_energy=model.HE, electric_inertia=model.parameters['electric_inertia'])


def observe(state: Any, model: Any, h: Any, current: list) -> dict:
    answer = measurements(state, model, h, current, model.sector.density, model.shape, operators(model))
    marginal = np.sum(abs(state.reshape(model.sector.dimension,-1))**2,axis=1)
    cap = (abs(model.sector.electric) == model.parameters['electric_cap']).any(axis=1)
    answer['any_link_cap_probability'] = float(marginal @ cap)
    hstate = h @ state
    direct = np.array([-2*np.vdot(hstate,q @ state).imag for q in model.Q])
    answer['continuity_residual_on_actual_state'] = (direct+INCIDENCE @ np.array(answer['physical_link_charge_current'])).tolist()
    answer['root_upper_population'] = [float(np.vdot(state,model.root_band_count(v,1) @ state).real) for v in range(3)]
    answer['root_lower_population'] = [float(np.vdot(state,model.root_band_count(v,-1) @ state).real) for v in range(3)]
    return answer


def main() -> None:
    start = time.monotonic()
    config = json.loads((ROOT/(PREFIX+'input.json')).read_text())
    if sha(ROOT/config['reference_file']) != config['expected_reference_SHA256']:
        raise ValueError('Shared reference bytes changed from declared owning input')
    if sha(ROOT/'rich_local_frame_carrier_20261015_model.py') != config['expected_model_SHA256']:
        raise ValueError('Owning H/source bytes changed from frozen interface')
    parameters = config['parameters']
    neutral = Model(6,parameters)
    if neutral.basis_hash != config['expected_neutral_basis_SHA256']:
        raise ValueError('Owning neutral basis changed')
    with np.load(ROOT/config['reference_file']) as archive:
        omega = archive[config['reference_key']].astype(complex).reshape(-1)
    if omega.size != neutral.dimension or abs(np.linalg.norm(omega)-1)>1e-10:
        raise ValueError('Shared neutral input incompatible with complete own basis')
    times = config['times']
    output: dict[str,Any] = dict(kind='NEW_INDEPENDENT_PRODUCT_LOCAL_SOURCE_AND_FIELD_TRAJECTORIES',
                                config=config, shared_neutral_input=dict(file=config['reference_file'], key=config['reference_key'],
                                SHA256=sha(ROOT/config['reference_file']), declaration='INPUT from owning new preparation, NOT independent ground solve, NOT evolved output'),
                                neutral_basis_SHA256=neutral.basis_hash,
                                primitive_operator_provenance=neutral.architecture['provenance'],
                                neutral={}, branches={}, fields={})
    saved: dict[str,Any] = dict(shared_neutral_t0=omega)
    neutral_histories = {}
    neutral_observations = {}
    for variant in config['variants']:
        hn, jn, _ = neutral.law(variant,config['phase'],0)
        states, resolution = propagate(hn,omega,times,config['krylov_dimension'])
        observed = [observe(state,neutral,hn,jn) for state in states]
        energy = observed[0]['energy']
        residual = float(np.linalg.norm(hn @ omega-energy*omega))
        neutral_histories[variant] = states
        neutral_observations[variant] = observed
        output['neutral'][variant] = dict(reference_energy=energy, reference_nonstationarity_norm=residual,
                                         samples=observed, propagation=resolution)
        saved['neutral_'+variant+'_final'] = states[-1]
        print(json.dumps(dict(stage='neutral',variant=variant,dimension=neutral.dimension,nonstationarity=residual)),flush=True)
        del hn, jn
    for branch, number in [('addition',7),('removal',5)]:
        model = Model(number,parameters)
        # Choose the physical Noether eigen-spinor from measured one-root LEFT.
        axis = np.array(config['source_spin_axis'])
        one = model.architecture['one_spin']
        generator = sum(axis[a]*one[a][:2,:2] for a in range(3))
        values,vectors = np.linalg.eigh(generator)
        spinor = vectors[:,np.argmax(values)]
        components = [model.source_from(neutral,0,branch,a,config['source_orbital']) for a in range(2)]
        source = sum(spinor[a]*components[a] for a in range(2))
        raw = source @ omega
        strength = float(np.vdot(raw,raw).real)
        if strength <= 1e-14:
            raise ValueError('Chosen physical source has zero raw strength')
        initial = raw/np.sqrt(strength)
        # Independent source locality/covariance witness on the actual input.
        source_covariance = []
        for site in range(3):
            for a in range(3):
                lhs = model.S_local[site][a] @ raw-source @ (neutral.S_local[site][a] @ omega)
                rhs = np.zeros(model.dimension,complex)
                if site == 0:
                    for beta in range(2):
                        rhs += (one[a][:2,:2] @ spinor)[beta]*(components[beta] @ omega)
                source_covariance.append(float(np.linalg.norm(lhs-rhs)))
        output['branches'][branch] = dict(number=number,physical_total_charge=6-number,
            basis_SHA256=model.basis_hash,dimension=model.dimension,shape=model.shape,
            source=dict(site=0,orbital=config['source_orbital'],spinor=[[float(z.real),float(z.imag)] for z in spinor],
                        physical_axis=axis.tolist(),raw_strength=strength,raw_norm=float(np.linalg.norm(raw)),
                        selected_component_raw_strengths=[float(np.linalg.norm(c @ omega)**2) for c in components],
                        local_source_Noether_commutator_residuals=source_covariance,
                        meaning='Addition carries Q=-1; removal is epsilon-contracted hole Q=+1, not measured positron.'),
            variants={})
        saved[branch+'_raw_source'] = raw
        saved[branch+'_initial'] = initial
        for variant in config['variants']:
            h,j,_ = model.law(variant,config['phase'],0)
            states,resolution = propagate(h,initial,times,config['krylov_dimension'],
                                         config['comparison_dimension'] if branch=='addition' and variant=='full' else None)
            observed = [observe(state,model,h,j) for state in states]
            excess = [subtract(sample,reference) for sample,reference in zip(observed,neutral_observations[variant])]
            change = [subtract(sample,observed[0]) for sample in observed]
            excess_change = [subtract(sample,excess[0]) for sample in excess]
            result = dict(samples=observed,signed_excess_over_evolving_neutral=excess,
                          own_initial_changes=change,excess_changes_from_own_t0=excess_change,
                          propagation=resolution,
                          energy_drift=[sample['energy']-observed[0]['energy'] for sample in observed])
            output['branches'][branch]['variants'][variant] = result
            saved[branch+'_'+variant+'_final'] = states[-1]
            print(json.dumps(dict(stage='charged',branch=branch,variant=variant,
                t=times[-1],excess_number=excess[-1]['local_number'],excess_spin=excess[-1]['local_frame_spin'],
                excess_norm2=excess[-1]['local_matrix_norm2'])),flush=True)
            if branch=='addition' and variant=='full':
                # Pure field unitary; its generator is the literal complete Wilson loop.
                pulsed = expm_multiply(-1j*config['field_pulse_alpha']*model.F,initial)
                pulse_states,pulse_resolution = propagate(h,pulsed,times,config['krylov_dimension'])
                pulse_observed = [observe(state,model,h,j) for state in pulse_states]
                pulse_delta = [subtract(a,b) for a,b in zip(pulse_observed,observed)]
                output['fields']['addition_full'] = dict(generator='Im(U0 U1 U2), complete capped Gauss loop; field-only',
                    alpha=config['field_pulse_alpha'],initial_injected_own_H_work=pulse_delta[0]['energy'],
                    samples=pulse_observed,pulse_minus_baseline=pulse_delta,propagation=pulse_resolution,
                    energy_drift=[sample['energy']-pulse_observed[0]['energy'] for sample in pulse_observed])
                saved['addition_field_pulsed_t0'] = pulsed
                saved['addition_field_pulsed_final'] = pulse_states[-1]
                output['first_actual_addition'] = dict(physical_charge=excess[-1]['physical_charge'],
                    physical_spin=excess[-1]['local_frame_spin'],matrix_norm=excess[-1]['local_matrix_norm2'],
                    field_induced_current=pulse_delta[-1]['physical_link_charge_current'],resolution=resolution)
                (ROOT/(PREFIX+'first_actual_result.json')).write_text(json.dumps(output['first_actual_addition'],indent=2)+'\n')
                del pulse_states,pulse_observed,pulsed
            del h,j,states
        del model,source,components,raw,initial
        gc.collect()
    output['elapsed_seconds'] = time.monotonic()-start
    output['semantics'] = dict(count_PVM='P(n_v=k), k0..4 of complete local Fock count, NOT added-particle position',
        joint_count_spin='Unnormalized signed <P(n_v=k) S_v,a> and <P(n_v=k) S_v²>',
        covariances='signed connected Q_v versus physical S_w,a or internal matrix r_w²; initial covariance is not transport',
        angular='material_L separately carried. U1 links are scalar line degrees on material paths; no independent continuum field angular generator is asserted.',
        time='Both source and shared nonstationary neutral propagated from absolute t0 for every own-H variant.',
        controls='path_cut disconnects site0; vector_off removes noncommuting material/frame hopping, scalar spin transport remains legal.',
        matrix_norm='r_v² is internal matrix norm, not spatial radius',
        limits='Finite triangle, constitutive physical frame/right sewing/CAR/cone metric/root geometry. No Nature/Dirac derivation, photons, proton, Lorentz or free COM claim.')
    np.savez_compressed(ROOT/(PREFIX+'states.npz'),**saved)
    (ROOT/(PREFIX+'result.json')).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(complete=True,elapsed_seconds=output['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    main()

"""Final-time physical detectors: raw count/spin/field joints, never a particle PVM."""
from __future__ import annotations
from typing import Any
import numpy as np
from rich_review_wave31_20261015_consume import observe as primitive_observe


def expval(state: Any, operator: Any) -> float:
    return float(np.vdot(state, operator @ state).real)


def radial_readout(state: Any, model: Any) -> dict:
    wave = np.asarray(state).reshape(model.shape)
    probabilities = np.sum(abs(wave)**2, axis=(0, 1))
    local = []
    for site in range(3):
        moved = np.moveaxis(wave, 2+site, -1).reshape(-1, len(model.r))
        density = moved.T @ moved.conj()
        weight = float(np.trace(density).real)
        local.append(dict(density_real=density.real.tolist(), density_imag=density.imag.tolist(),
                          eigenvalue_probabilities=np.diag(density).real.tolist(),
                          unnormalized_purity=float(np.trace(density @ density).real),
                          weight=weight))
    outer = np.indices(probabilities.shape)
    return dict(joint_three_site_norm_eigenvalue_probability=probabilities.tolist(),
                local_quantum_radial_density=local,
                any_outer_node_probability=float(probabilities[(outer == len(model.r)-1).any(axis=0)].sum()),
                norm2_eigenvalues=(model.r**2).tolist())


def conditional_readout(projected: Any, model: Any, currents: list, weight: float) -> dict:
    # P is applied only at final measurement. PJP is an actual postmeasurement
    # current, not a claimed nondestructive simultaneous count/current PVM.
    values = dict(physical_charge=[expval(projected, op) for op in model.Q],
                  local_frame_spin=[[expval(projected, op) for op in site] for site in model.S_local],
                  local_matrix_norm2=[expval(projected, op) for op in model.r2_local],
                  material_angular=[expval(projected, op) for op in model.material_L],
                  total_angular=[expval(projected, op) for op in model.J],
                  electric_flux=[expval(projected, op) for op in model.E],
                  final_Luders_current=[expval(projected, op) for op in currents])
    return dict(probability=weight, raw_subnormalized=values,
                conditional_given_detector={key:(np.array(value)/weight).tolist() if weight > 1e-14 else None
                                            for key,value in values.items()},
                quantum_norm=radial_readout(projected, model))


def observe(state: Any, model: Any, h: Any, currents: list, axis: Any,
            receiver: int, branch: str | None = None) -> dict:
    answer = primitive_observe(state, model, h, currents)
    answer['quantum_norm'] = radial_readout(state, model)
    density = model.sector.density
    flat = np.asarray(state).reshape(model.sector.dimension, -1)
    joint = []
    for count in range(5):
        masked = (flat * (density[:,receiver] == count)[:,None]).reshape(-1)
        joint.append(dict(count=count, probability=float(np.vdot(masked,masked).real),
                          quantum_norm=radial_readout(masked,model)))
    answer['receiver_all_count_quantum_norm_joints'] = joint
    if branch is None:
        return answer
    selected_count = 3 if branch == 'addition' else 1
    selected = (flat * (density[:,receiver] == selected_count)[:,None]).reshape(-1)
    spin_axis = sum(axis[a]*model.S_local[receiver][a] for a in range(3))
    spin_action = spin_axis @ selected
    positive, negative = .5*selected+spin_action, .5*selected-spin_action
    rest = np.asarray(state)-selected
    p = float(np.vdot(selected,selected).real)
    plus = float(np.vdot(positive,positive).real)
    minus = float(np.vdot(negative,negative).real)
    answer['receiver_detector'] = dict(
        definition=f'P(n_receiver={selected_count}) jointly resolved by physical S_receiver.axis=+/-1/2',
        receiver=receiver, source_axis=np.asarray(axis).tolist(),
        meaning='Actual full-Fock detector, not exclusive added-particle position; neutral sea can trigger. No intermediate projection.',
        probability=p, spin_plus_probability=plus, spin_minus_probability=minus,
        rest_probability=float(np.vdot(rest,rest).real),
        spin_joint_sum_residual=plus+minus-p,
        spin_projector_idempotence_on_state=float(np.linalg.norm(.5*positive+spin_axis@positive-positive)),
        count=conditional_readout(selected,model,currents,p),
        spin_plus=conditional_readout(positive,model,currents,plus),
        spin_minus=conditional_readout(negative,model,currents,minus),
        rest=conditional_readout(rest,model,currents,float(np.vdot(rest,rest).real)))
    return answer

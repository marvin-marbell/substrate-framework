"""Local expectations/joint count measurements; never an added-particle n/N PVM."""
from __future__ import annotations
from typing import Any
import numpy as np


def expectation(state: Any, operator: Any) -> float:
    return float(np.vdot(state, operator @ state).real)


def measurements(state: Any, model: Any, hamiltonian: Any, currents: list,
                 density: Any, shape: tuple, operators: dict) -> dict:
    """Gauge/CAR index first; count projectors are diagonal on that raw index."""
    state = np.asarray(state).reshape(-1)
    wave = state.reshape(shape)
    probabilities = np.sum(abs(wave.reshape(shape[0], -1))**2, axis=1)
    count = np.asarray(density, dtype=float)
    charges = 2-count
    count_pvm = np.stack([[(count[:,v] == k) @ probabilities for k in range(5)] for v in range(3)])
    local_n = probabilities @ count
    local_q = probabilities @ charges
    spin = np.zeros((3,3))
    joint_count_spin = np.zeros((3,5,3))
    joint_count_casimir = np.zeros((3,5))
    spin_casimir = np.zeros(3)
    charge_spin_covariance = np.zeros((3,3,3))
    charge_norm_covariance = np.zeros((3,3))
    norm = np.zeros(3)
    for site in range(3):
        for axis in range(3):
            action = np.asarray(operators['spin'][site][axis] @ state).reshape(shape[0],-1)
            gauge_joint = np.sum(wave.reshape(shape[0],-1).conj()*action, axis=1).real
            spin[site,axis] = gauge_joint.sum()
            for k in range(5):
                joint_count_spin[site,k,axis] = (count[:,site] == k) @ gauge_joint
            for charge_site in range(3):
                charge_spin_covariance[charge_site,site,axis] = charges[:,charge_site] @ gauge_joint - local_q[charge_site]*spin[site,axis]
        action = np.asarray(operators['spin2'][site] @ state).reshape(shape[0],-1)
        gauge_joint = np.sum(wave.reshape(shape[0],-1).conj()*action,axis=1).real
        spin_casimir[site] = gauge_joint.sum()
        for k in range(5):
            joint_count_casimir[site,k] = (count[:,site] == k) @ gauge_joint
        action = np.asarray(operators['norm2'][site] @ state).reshape(shape[0],-1)
        gauge_joint = np.sum(wave.reshape(shape[0],-1).conj()*action,axis=1).real
        norm[site] = gauge_joint.sum()
        for charge_site in range(3):
            charge_norm_covariance[charge_site,site] = charges[:,charge_site] @ gauge_joint - local_q[charge_site]*norm[site]
    hstate = np.asarray(hamiltonian @ state).reshape(-1)
    energy = float(np.vdot(state,hstate).real)
    # d<O>/dt = -2 Im <Hpsi|Opsi>, using THIS H, without huge commutators.
    rates = {}
    for key in ['spin', 'norm2']:
        values = operators[key]
        if key == 'spin':
            rates[key] = [[float(-2*np.vdot(hstate,op @ state).imag) for op in site] for site in values]
        else:
            rates[key] = [float(-2*np.vdot(hstate,op @ state).imag) for op in values]
    electric = [expectation(state,op) for op in operators['electric']]
    current = [expectation(state,op) for op in currents]
    electric_power = -sum(float(np.vdot(operators['electric'][edge] @ state, currents[edge] @ state).real)
                          for edge in range(3))/operators['electric_inertia']
    return dict(norm=float(np.vdot(state,state).real), energy=energy,
                local_number=local_n.tolist(), physical_charge=local_q.tolist(),
                local_frame_spin=spin.tolist(), local_frame_casimir=spin_casimir.tolist(),
                local_matrix_norm2=norm.tolist(), local_count_PVM=count_pvm.tolist(),
                joint_local_count_spin=joint_count_spin.tolist(),
                joint_local_count_spin_casimir=joint_count_casimir.tolist(),
                signed_charge_spin_covariance=charge_spin_covariance.tolist(),
                signed_charge_matrixnorm_covariance=charge_norm_covariance.tolist(),
                material_angular=[expectation(state,op) for op in operators['material']],
                total_angular=[expectation(state,op) for op in operators['total']],
                electric_flux=electric, physical_link_charge_current=current,
                electric_flux_rate=(-np.array(current)).tolist(),
                electric_power=electric_power,
                electric_energy=expectation(state,operators['electric_energy']),
                rates=rates)


def subtract(left: dict, right: dict) -> dict:
    result = {}
    for key,value in left.items():
        if isinstance(value,dict):
            result[key] = subtract(value,right[key])
        elif isinstance(value,list):
            result[key] = (np.array(value)-np.array(right[key])).tolist()
        else:
            result[key] = value-right[key]
    return result

"""Quantum energy-filter cloud in the unchanged local-frame Hamiltonian.

This is a preparation operator, not a reduced dynamical law. Every Krylov
vector is a full physical CAR/Gauss/material/radial state.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.linalg import eigh_tridiagonal
from rich_local_frame_carrier_20261015_model import Model, prepared_reference


def expectation(state: Any, operator: Any) -> float:
    return float(np.vdot(state, operator @ state).real)


def energy(h: Any, state: Any) -> dict:
    hs = h @ state
    mean = float(np.vdot(state, hs).real)
    return dict(energy=mean, variance=float(np.vdot(hs-mean*state, hs-mean*state).real))


def lanczos(h: Any, initial: Any, dimension: int) -> tuple:
    """Full-space, two-pass reorthogonalized Hermitian Lanczos."""
    initial = np.asarray(initial, complex)
    basis = np.empty((dimension, initial.size), complex)
    vector = initial / np.linalg.norm(initial)
    diagonal, offdiagonal = [], []
    residual = 0.0
    for index in range(dimension):
        basis[index] = vector
        work = h @ vector
        alpha = float(np.vdot(vector, work).real)
        diagonal.append(alpha)
        work -= alpha*vector
        if index:
            work -= offdiagonal[-1]*basis[index-1]
        for _ in range(2):
            for previous in basis[:index+1]:
                work -= np.vdot(previous, work)*previous
        residual = float(np.linalg.norm(work))
        if residual < 1e-13 or index+1 == dimension:
            break
        offdiagonal.append(residual)
        vector = work/residual
    return basis[:len(diagonal)], np.asarray(diagonal), np.asarray(offdiagonal[:len(diagonal)-1]), residual


def cool(h: Any, initial: Any, taus: list[float], dimension: int,
         comparison_dimension: int, shift: float) -> tuple[list, list, dict]:
    """Return NORMALIZED inputs and log raw norm gains for exp[-tau(H-shift)]."""
    initial_norm = float(np.linalg.norm(initial))
    basis, diagonal, offdiagonal, residual = lanczos(h, initial, dimension)
    values, rotations = eigh_tridiagonal(diagonal, offdiagonal)
    size = len(diagonal)
    coarse = min(comparison_dimension, size)
    cv, cr = eigh_tridiagonal(diagonal[:coarse], offdiagonal[:coarse-1])
    states, gains, differences, measurements = [], [], [], []
    for tau in taus:
        coefficients = rotations @ (np.exp(-tau*(values-values[0]))*rotations[0])
        norm = float(np.linalg.norm(coefficients))
        state = (coefficients/norm) @ basis
        cc = cr @ (np.exp(-tau*(cv-cv[0]))*cr[0])
        alternate = (cc/np.linalg.norm(cc)) @ basis[:coarse]
        states.append(state)
        gains.append(float(np.log(initial_norm)+np.log(norm)-tau*(values[0]-shift)))
        differences.append(float(np.linalg.norm(state-alternate)))
        measurements.append(energy(h, state))
    return states, gains, dict(method='FULL_H_energy_filter_two_pass_Lanczos',
        requested_dimension=dimension, actual_dimension=size, coarse_dimension=coarse,
        shift=shift, taus=taus, normalized_state_comparison=differences,
        log_raw_output_norm=gains, measurements=measurements, final_Lanczos_residual=residual)


def spin_down(model: Any, state: Any) -> Any:
    """Exact total-J lowering; valid doublet partner only with recorded checks."""
    return (model.J[0]-1j*model.J[1]) @ state


def component_source(target: Any, reference: Any, neutral: Any, component: int,
                     site: int = 0, branch: str = 'addition') -> Any:
    return target.source_from(reference, site, branch, component=component, orbital=0) @ neutral


def load_reference(path: str = 'rich_dressed_local_carrier_20261015_states.npz') -> Any:
    with np.load(path) as saved:
        return saved['neutral'].copy()


def load_source(branch: str, component: int = 0,
                path: str = 'rich_dressed_local_carrier_20261015_states.npz') -> Any:
    """Normalized component input; raw filter strength belongs to interface JSON."""
    if branch not in ('addition', 'removal') or component not in (0, 1):
        raise ValueError((branch, component))
    with np.load(path) as saved:
        return saved[f'{branch}_component{component}'].copy()

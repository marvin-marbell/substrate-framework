"""Hermitian Krylov propagation with explicit state-resolution evidence, not an eigensolve."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.linalg import eigh_tridiagonal


def propagate(h: Any, initial: Any, times: list[float], dimension: int,
              comparison_dimension: int | None = None) -> tuple[list, dict]:
    """One Lanczos space, full two-pass reorthogonalization, all declared times."""
    initial = np.asarray(initial, dtype=complex)
    norm = float(np.linalg.norm(initial))
    if abs(norm-1) > 1e-9:
        raise ValueError('Propagation expects normalized state')
    basis = np.empty((dimension, initial.size), dtype=complex)
    diagonal: list[float] = []
    offdiagonal: list[float] = []
    vector = initial / norm
    residual = 0.0
    for index in range(dimension):
        basis[index] = vector
        work = np.asarray(h @ vector).reshape(-1)
        alpha = float(np.vdot(vector, work).real)
        diagonal.append(alpha)
        work -= alpha*vector
        if index:
            work -= offdiagonal[-1]*basis[index-1]
        # Two complete MGS passes keep raw nonorthogonality from hiding error.
        for _ in range(2):
            for previous in basis[:index+1]:
                work -= np.vdot(previous, work)*previous
        residual = float(np.linalg.norm(work))
        if residual < 1e-13 or index+1 == dimension:
            break
        offdiagonal.append(residual)
        vector = work/residual
    size = len(diagonal)
    basis = basis[:size]
    eig, rotations = eigh_tridiagonal(np.array(diagonal), np.array(offdiagonal[:size-1]))
    states = []
    for time in times:
        coefficients = rotations @ (np.exp(-1j*time*eig)*rotations[0, :])
        states.append(coefficients @ basis)
    report: dict[str, Any] = dict(method='two-pass fully reorthogonalized Hermitian Lanczos',
                                   requested_dimension=dimension, actual_dimension=size,
                                   last_residual_norm=residual,
                                   norm_errors=[abs(float(np.vdot(v,v).real)-1) for v in states])
    if comparison_dimension is not None:
        coarse = min(comparison_dimension, size)
        ceig, crot = eigh_tridiagonal(np.array(diagonal[:coarse]), np.array(offdiagonal[:coarse-1]))
        differences = []
        for time, state in zip(times, states):
            coefficients = crot @ (np.exp(-1j*time*ceig)*crot[0, :])
            alternate = coefficients @ basis[:coarse]
            differences.append(float(np.linalg.norm(state-alternate)))
        report['same_state_accuracy_comparison'] = dict(coarse_dimension=coarse,
                                                         fine_dimension=size,
                                                         times=times, state_norm_differences=differences,
                                                         expectation_error_bound_for_norm1_operator=[2*value for value in differences])
    return states, report

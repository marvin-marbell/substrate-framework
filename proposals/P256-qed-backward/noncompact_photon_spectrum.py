"""Complete Q-accessible finite-H measure; no Ritz-node pole substitution."""
from __future__ import annotations

import math
from typing import Any, Callable, cast

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags, eye, kron

from noncompact_native_current import accept, adverse


def cluster_measure(matrix, q, energy, cluster_tolerance=1e-7) -> tuple[Any, Any, list[dict[str, Any]], dict[str, Any]]:
    """All eigenclusters, with a posteriori finite-matrix error bounds.

    The complete diagonalization brackets both sides of every cluster. The
    Frobenius backward error bounds its operator norm. Projector errors use
    the separation from the entire complementary spectrum, not a solve edge.
    These are floating-point residual certificates, not interval arithmetic.
    """
    solver = cast(Callable[..., Any], eigh)
    values, X = solver(matrix.toarray(), check_finite=False, driver='evd')
    orthogonality = float(np.linalg.norm(X.conj().T @ X - np.eye(len(values))))
    accept(orthogonality, 'complete source-sector eigenbasis orthogonality', 2e-8)
    R = matrix @ X - X * values[None, :]
    backward = float(np.linalg.norm(R))
    accept(backward, 'complete source-sector finite-H backward residual', 2e-7)
    # Polar correction bound: ||X-U|| <= ||X*X-I||/(1+sqrt(1-error)).
    polar = orthogonality / (1 + math.sqrt(max(0., 1 - orthogonality)))
    scale = float(np.max(abs(values)))
    delta = (backward + 2 * scale * polar) / (1 - polar)
    total = float(np.vdot(q, q).real)
    weights = abs(X.conj().T @ q)**2
    accept(abs(float(sum(weights)) - total), 'complete Q spectral sum rule', 2e-8)
    bands = []
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[end] - values[end - 1] < cluster_tolerance:
            end += 1
        gaps = []
        if begin:
            gaps.append(float(values[begin] - values[begin - 1]))
        if end < len(values):
            gaps.append(float(values[end] - values[end - 1]))
        gap = min(gaps) if gaps else math.inf
        if gap <= 2 * delta:
            raise AssertionError('Finite spectral cluster not isolated beyond backward error; change cluster tolerance')
        residual = float(np.linalg.norm(R[:, begin:end]))
        projector_error = min(1., (residual + (2 * scale + delta) * polar) / (gap - 2 * delta)) if gaps else polar
        residue = float(sum(weights[begin:end]))
        bands.append(dict(index=len(bands), begin=begin, end=end, multiplicity=end-begin,
                          energy=float(np.mean(values[begin:end]) - energy),
                          energy_interval=[float(values[begin]-energy-delta), float(values[end-1]-energy+delta)],
                          residue=residue, fraction=residue/total,
                          residue_error_upper=total*(2*projector_error+3*polar),
                          fraction_error_upper=2*projector_error+3*polar,
                          complement_gap=gap if gaps else None,
                          projector_error_upper=projector_error, finite_H_residual=residual))
        begin = end
    return values, X, bands, dict(dimension=len(values), eigenbasis_orthogonality=orthogonality,
                                 finite_H_backward_error_upper=delta,
                                 complete_spectral_weight=float(sum(weights)),
                                 spectral_sum_error=abs(float(sum(weights))-total),
                                 bound_scope='a posteriori floating-point finite-matrix residuals; supplied q fixed; not interval arithmetic or infinite-H convergence')


def full_source_measure(model, energy, q, maximum_dimension) -> tuple[Any, Any, Any, list[dict[str, Any]], dict[str, Any]]:
    """Exact invariant triality reduction of the SAME full Galerkin H."""
    sector, sector_weights = model.source_triality(q)
    indices = np.flatnonzero((model.triality() == sector).ravel())
    if len(indices) > maximum_dimension:
        raise AssertionError(f'Complete Q-sector dimension {len(indices)} exceeds explicit dense bound {maximum_dimension}; increase --spectral-max-dimension')
    H = diags(np.repeat(model.phenergy, model.d) + np.tile(model.coulomb, model.phdim), format='csr', dtype=complex)
    H += kron(eye(model.phdim, format='csr'), csr_matrix(model.onsite), format='csr')
    for Z, T in zip(model.phases, model.hops):
        term = kron(Z, csr_matrix(T), format='csr')
        H += term + term.conj().T
    witness = np.random.default_rng(2221057).normal(size=model.size).astype(complex)
    witness += 1j*np.random.default_rng(2221058).normal(size=model.size)
    witness /= np.linalg.norm(witness)
    accept(np.linalg.norm(H @ witness - model.action(witness)), 'assembled SAME full H versus independent engine action', 2e-10)
    outside = np.ones(model.size, bool); outside[indices] = False
    accept(float(np.linalg.norm(H[outside][:, indices].data)), 'complete Q-source sector is invariant in whole H', 2e-10)
    block = H[indices][:, indices].tocsr()
    values, X, bands, certificate = cluster_measure(block, q[indices], energy)
    leakage = float(np.linalg.norm(q[outside]))
    total = float(np.vdot(q, q).real)
    for row in bands:
        row['fraction'] = row['residue'] / total
        row['fraction_error_upper'] = row['residue_error_upper'] / total
    certificate.update(Q_triality=sector, Q_triality_weights=sector_weights,
                       whole_H_dimension=model.size,
                       Q_outside_sector_roundoff_norm=leakage,
                       Q_outside_sector_weight=leakage**2,
                       completeness='ALL eigenclusters in exact Q-accessible triality sector; complementary trialities Q-dark for exact supplied-sector state, measured roundoff leakage retained; not deleted from full H')
    return indices, values, X, bands, certificate


def oscillator_positive_control(UE, omega):
    """Independent soluble finite oscillator checks residue, poles and signs."""
    H = diags(omega*(np.arange(4)+.5), format='csr', dtype=complex)
    q = np.zeros(4, complex); q[1] = math.sqrt(UE/omega)
    values, X, bands, certificate = cluster_measure(H, q, .5*omega)
    bright = max(bands, key=lambda row: row['residue'])
    accept(abs(bright['energy']-omega), 'soluble oscillator actual eigenpole', 2e-10)
    accept(abs(bright['residue']-UE/omega), 'soluble oscillator actual coordinate residue', 2e-10)
    z = complex(.8*omega, .4*omega)
    measured = np.vdot(q, np.linalg.solve(H.toarray()-(.5*omega+z)*np.eye(4), q))
    expected = (UE/omega)/(omega-z)
    accept(abs(measured-expected), 'soluble oscillator independent resolvent', 2e-10)
    wrong = adverse(abs(measured-(UE/omega)/(omega+z)), 'soluble oscillator independent resolvent', 2e-10)
    return dict(energy=bright['energy'], residue=bright['residue'], fraction=bright['fraction'],
                resolvent_error=float(abs(measured-expected)), wrong_resolvent_sign=wrong,
                finite_H_certificate=certificate)

"""Source-aligned continuation of the ONE stationary mode experiment.

The lowest four C- levels were field-dark. Follow the actual electric source
cyclic subspace, rather than calling those levels photons or changing the law.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
from scipy import sparse
from scipy.linalg import eigh_tridiagonal
from rich_neutral_mott_gauge_20261015_model import NeutralCube
from rich_review_wave26_20261015_consume import PREFIX, diagonal_observables, parity_basis, real, save

ROOT = Path(__file__).resolve().parent


def main() -> None:
    settings = json.loads((ROOT/(PREFIX+'source_input.json')).read_text())
    start = time.monotonic()
    model = NeutralCube()
    g = np.load(ROOT/settings['reference_states'])['reference']
    reference = diagonal_observables(model,g)
    eg = reference['H']
    reference_residual = float(np.linalg.norm(model.H @ g-eg*g))
    minus = parity_basis(model,-1)
    h = (minus.T @ model.H @ minus).tocsr()
    coefficients = model.left_cycle.astype(float)/np.linalg.norm(model.left_cycle)
    e = model.electric @ coefficients
    raw_source = e*g
    source = np.asarray(minus.T @ raw_source).ravel()
    source_norm = float(np.linalg.norm(source))
    dimension = h.shape[0]
    count = settings['krylov_dimension']
    basis = np.empty((dimension,count),dtype=float,order='F')
    q = source/source_norm
    previous = np.zeros(dimension)
    beta_previous = 0.0
    alphas, betas = [], []
    orth_corrections = []
    final_beta = 0.0
    used = 0
    for k in range(count):
        basis[:,k] = q
        w = h @ q-beta_previous*previous
        alpha = real(np.vdot(q,w))
        w -= alpha*q
        active = basis[:,:k+1]
        correction_norm = 0.0
        for _ in range(2):
            correction = active.T @ w
            correction_norm = max(correction_norm,float(np.linalg.norm(correction)))
            w -= active @ correction
        beta = float(np.linalg.norm(w))
        alphas.append(alpha)
        orth_corrections.append(correction_norm)
        used = k+1
        final_beta = beta
        if beta < settings['breakdown_tolerance'] or k+1==count:
            break
        betas.append(beta)
        previous,q = q,w/beta
        beta_previous = beta
    basis = basis[:,:used]
    values, vectors = eigh_tridiagonal(np.asarray(alphas),np.asarray(betas))
    weights = source_norm**2*np.abs(vectors[0])**2
    estimates = final_beta*np.abs(vectors[-1])
    eligible = np.flatnonzero((weights >= settings['minimum_raw_weight']) & (estimates <= settings['maximum_residual_estimate']))
    if len(eligible):
        selected = int(eligible[0])
        status = 'CONVERGED_SOURCE_BRIGHT_RAY'
    else:
        selected = int(np.argmax(weights))
        status = 'FINITE_SOURCE_PROBE_NO_CONVERGED_BRIGHT_RAY'
    v = np.asarray(minus @ (basis @ vectors[:,selected])).ravel()
    energy = float(values[selected])
    gap = energy-eg
    obs = diagonal_observables(model,v)
    current = sum((coefficients[k]*model.currents[k] for k in range(12)),sparse.csr_matrix(model.H.shape,dtype=complex))
    jg = current @ g
    loopg = ((model.W-model.W.T)/(2j)) @ g
    d = model.doubles.sum(axis=1)
    e_amplitude = np.vdot(v,raw_source)
    j_amplitude = np.vdot(v,jg)
    residue = float(abs(e_amplitude)**2)
    r = dict(kind='SOURCE_ALIGNED_CONTINUATION_OF_SAME_STATIONARY_MODE_EXPERIMENT',status=status,settings=settings,
             law='neutral-mott-cube-uniform-physical-edge-qlink3-20261015',reference=reference,reference_full_H_residual_l2=reference_residual,
             hypothesis_correction='Lowest four C- eigenvectors were field-dark. Follow the actual field source cyclic sector without changing H, parameters, capacity, reference or matter/gauge feedback.',
             selected=dict(quadrature_index=selected,H=energy,excitation_energy=gap,
                           full_H_residual_l2=float(np.linalg.norm(model.H @ v-energy*v)),
                           tridiagonal_residual_estimate=float(estimates[selected]),C_minus_residual_l2=float(np.linalg.norm(model.C @ v+v)),
                           diagonal=obs,mode_minus_reference={key:obs[key]-reference[key] for key in ('H','HE','HU','hopping','doublon_total','doublon_per_site','holon_total','charge_variance_total','probability_any_pair')},
                           E_residue_raw=residue,E_fraction_of_declared_source=residue/source_norm**2,
                           J_residue_raw=float(abs(j_amplitude)**2),ImW_residue_raw=float(abs(np.vdot(v,loopg))**2),
                           D_transition_squared_SELECTION_RULE=float(abs(np.vdot(v,d*g))**2),
                           J_equals_i_gap_E_amplitude_residual=float(abs(j_amplitude-1j*gap*e_amplitude)),
                           all_edge_E_residue_raw=[float(abs(np.vdot(v,model.electric[:,k]*g))**2) for k in range(12)],
                           all_edge_J_residue_raw=[float(abs(np.vdot(v,model.currents[k] @ g))**2) for k in range(12)]),
             source=dict(E_raw_norm_squared=source_norm**2,J_raw_norm_squared=real(np.vdot(jg,jg)),ImW_raw_norm_squared=real(np.vdot(loopg,loopg)),
                         coefficients=coefficients.tolist(),unrepresented_E_source_norm_squared=source_norm**2-residue),
             finite_Krylov=dict(used_dimension=used,orthogonality_max=float(np.max(np.abs(basis.T @ basis-np.eye(used)))),
                               max_reorthogonalization_correction=max(orth_corrections),terminal_beta=final_beta,
                               quadrature_energies=values.tolist(),quadrature_raw_weights=weights.tolist(),residual_estimates=estimates.tolist(),
                               quadrature_weight_sum=float(weights.sum()),
                               scope='Finite scalar-source Lanczos quadrature, NOT complete spectrum. Only selected converged ray promoted; all other quadrature nodes remain unresolved numerical spectral approximants.'),
             group_scope='A normalized source-selected eigenray, potentially a ray in a degenerate eigenspace; no complete degenerate group or multiplicity claim. Diagonal matter expectations refer to this ray, not group averages. Residual source weight includes all other energies and possible unrepresented group directions.',
             units='hbar=t=I=1. E coefficients have Euclidean norm1. Spectral fraction is raw residue / raw source norm squared; it does not normalize particle populations. Dtot counts expected doublon sites, Dtot/8 per site. Fractions are expectations, not fractional charge quanta.',
             interpretation='C-even total D transition to C- can vanish identically while DIAGONAL D is nonzero. Pair content here is charge-pair admixture; real-on-shell versus virtual dressing remains unproved. A gapped mixed correlated gauge/matter field-bright mode is not a massless photon or bound-exciton theorem.',
             scientific_elapsed_seconds=time.monotonic()-start)
    save('source_mode_result',r)
    np.savez_compressed(ROOT/(PREFIX+'source_mode_states.npz'),reference=g,selected_mode=v,H=energy,source=raw_source,old_indices=model.old_indices)
    print(json.dumps(dict(status=status,selected=r['selected'],source=r['source'],elapsed=r['scientific_elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()

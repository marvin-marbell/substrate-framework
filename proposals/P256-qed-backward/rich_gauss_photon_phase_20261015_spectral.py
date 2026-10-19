"""New small-source joint-H response on the owning I16 cube/reference INPUT."""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy import sparse
from scipy.linalg import eigh


def real(value: Any) -> float:
    return float(np.real(value))


def encoded(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {'real': array.real.tolist(), 'imag': array.imag.tolist()}


def diagonal(model: Any, state: Any) -> dict[str, float]:
    probability = np.abs(state)**2
    h = real(np.vdot(state, model.H @ state))
    he = real(probability @ model.HE)
    hu = real(probability @ model.HU)
    return dict(norm=real(probability.sum()), H=h, HE=he, HU=hu, hopping=h-he-hu,
                doublon_total=real(probability @ model.doubles.sum(axis=1)),
                holon_total=real(probability @ model.holons.sum(axis=1)),
                charge_variance_total=real(probability @ ((model.density-1)**2).sum(axis=1)),
                probability_any_pair=real(probability[model.doubles.sum(axis=1)>0].sum()))


def response(model: Any, omega: Any, settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Only this cube's circulation/loop/density source block, not a pole engine."""
    energy = real(np.vdot(omega, model.H @ omega))
    coefficients = model.left_cycle.astype(float)/np.linalg.norm(model.left_cycle)
    electric = model.electric[:, :12] @ coefficients
    current = sum((coefficients[e]*model.currents[e] for e in range(12)),
                  sparse.csr_matrix(model.H.shape, dtype=complex))
    loop = (model.W-model.W.T)/(2j)
    dipole_coefficients = (1-model.positions[:, 0])*(2*model.positions[:, 2]-1)/2
    density = (model.density-1) @ dipole_coefficients
    source = np.column_stack((electric*omega, current @ omega, loop @ omega, density*omega))
    source -= omega[:, None]*(omega.conj() @ source)[None, :]
    # H and Omega are real in this frozen neutral basis. The imaginary Hermitian
    # source ImW is phase-rotated ONLY for a real Krylov basis, restored in source.
    seeds = np.column_stack((source[:, 0].real, source[:, 2].imag, source[:, 3].real))
    u, singular, _ = np.linalg.svd(seeds, full_matrices=False)
    rank = int(np.sum(singular > 1e-12*singular[0]))
    q = np.empty((model.dimension, settings['krylov_dimension']), order='F')
    q[:, :rank] = u[:, :rank]
    used = rank
    begin = 0
    while used < q.shape[1]:
        block = model.H @ q[:, begin:used]
        active = q[:, :used]
        for _ in range(2):
            block -= active @ (active.T @ block)
        bu, bs, _ = np.linalg.svd(block, full_matrices=False)
        keep = min(int(np.sum(bs > 1e-11)), q.shape[1]-used)
        if not keep:
            break
        begin = used
        q[:, used:used+keep] = bu[:, :keep]
        used += keep
    q = q[:, :used]
    hq = model.H @ q
    reduced = q.T @ hq
    reduced = (reduced+reduced.T)/2
    values, vectors = eigh(reduced)
    gaps = values-energy
    projected_source = q.T @ source
    amplitudes = vectors.T @ projected_source
    residual_matrix = hq-q @ reduced
    residuals = np.linalg.norm(residual_matrix @ vectors, axis=0)
    weights = np.abs(amplitudes[:, 0])**2
    eligible = np.flatnonzero((weights > settings['minimum_raw_weight']) &
                             (residuals < settings['maximum_mode_residual']) & (gaps > 1e-9))
    selected = int(eligible[0]) if len(eligible) else int(np.argmax(weights))
    mode = q @ vectors[:, selected]
    gap = float(gaps[selected])
    before, after = diagonal(model, omega), diagonal(model, mode)
    contact = -sum((coefficients[e]**2*(model.transfers[e]+model.transfers[e].T) for e in range(12)),
                   sparse.csr_matrix(model.H.shape))
    contact_mean = real(np.vdot(omega, contact @ omega))

    def susceptibility(z: complex, amps: Any, frequencies: Any) -> Any:
        return (amps.conj().T @ (amps/(z-frequencies)[:, None])-
                amps.T @ (amps.conj()/(z+frequencies)[:, None]))

    evaluations = []
    coarse = min(settings['coarse_dimension'], used)
    cv, ct = eigh(reduced[:coarse, :coarse])
    ca = ct.T @ projected_source[:coarse]
    for frequency in settings['frequencies']:
        z = complex(frequency, settings['eta'])
        chi = susceptibility(z, amplitudes, gaps)
        coarse_chi = susceptibility(z, ca, cv-energy)
        # Explicit full-H resolvent residuals give ||error|| <= ||r||/eta;
        # stationarity residual is separately disclosed, not silently absorbed.
        positive = np.linalg.solve(z*np.eye(used)-(reduced-energy*np.eye(used)), projected_source)
        negative = np.linalg.solve(z*np.eye(used)+(reduced-energy*np.eye(used)), projected_source)
        leakage = source-q @ projected_source
        rp = leakage+residual_matrix @ positive
        rn = leakage-residual_matrix @ negative
        evaluations.append(dict(frequency=frequency, eta=settings['eta'], chi=encoded(chi),
                                coarse_difference_max=float(np.max(np.abs(chi-coarse_chi))),
                                positive_resolvent_residual_l2=np.linalg.norm(rp, axis=0).tolist(),
                                negative_resolvent_residual_l2=np.linalg.norm(rn, axis=0).tolist(),
                                resolvent_error_l2_bound=(np.maximum(np.linalg.norm(rp, axis=0),np.linalg.norm(rn, axis=0))/settings['eta']).tolist(),
                                primitive_current_response_to_external_phase=encoded(-contact_mean-chi[1, 1])))
    static = susceptibility(0j, amplitudes, gaps)
    eamp = np.asarray([np.vdot(mode, model.electric[:, e]*omega) for e in range(12)])
    jamp = np.asarray([np.vdot(mode, model.currents[e] @ omega) for e in range(12)])
    rhoamp = mode.conj() @ ((model.density-1)*omega[:, None])
    near = np.flatnonzero((np.abs(gaps-gap)<settings['group_energy_tolerance']) &
                         (residuals<settings['maximum_mode_residual']))
    near_gram = amplitudes[near].conj().T @ amplitudes[near]
    result = dict(kind='NEW_SMALL_SOURCE_SIGNED_REDUCIBLE_JOINT_H_RESPONSE',
                  shared_reference_is_INPUT=True, reference=before,
                  reference_residual_l2=float(np.linalg.norm(model.H @ omega-energy*omega)),
                  source_names=['left_loop_canonical_flux', 'same_primitive_loop_current', 'left_ImW', 'left_z_charge_dipole'],
                  source_coefficients=coefficients.tolist(), density_coefficients=dipole_coefficients.tolist(),
                  source_gram=encoded(source.conj().T @ source), real_seed_rank=rank,
                  source_projection_residual_l2=np.linalg.norm(source-q @ projected_source, axis=0).tolist(),
                  finite_response=evaluations, contact_expectation=contact_mean,
                  relaxed_static_twist_curvature=encoded(contact_mean+static[1, 1]),
                  static_semantics='Exact field-basis unitary equivalence makes fully relaxed static curvature zero; this is not absence of induced dynamics. Full joint-H response is reducible and is NOT reinserted into a Dyson kernel.',
                  selected=dict(status='CONVERGED_SOURCE_BRIGHT_RAY' if len(eligible) else 'UNRESOLVED_FINITE_SOURCE_RAY',
                                H=float(values[selected]), gap=gap, full_H_residual_l2=float(np.linalg.norm(model.H @ mode-values[selected]*mode)),
                                raw_source_residues=np.abs(amplitudes[selected]).__pow__(2).tolist(),
                                E_source_fraction=float(weights[selected]/real(np.vdot(source[:, 0],source[:, 0]))),
                                unresolved_E_weight=float(real(np.vdot(source[:, 0],source[:, 0]))-weights[selected]),
                                diagonal=after, mode_minus_reference={key:after[key]-before[key] for key in before},
                                D_even_transition_squared_SELECTION_RULE_ONLY=float(abs(np.vdot(mode,model.doubles.sum(axis=1)*omega))**2),
                                E_amplitudes=encoded(eamp), J_amplitudes=encoded(jamp), density_amplitudes=encoded(rhoamp),
                                primitive_flux_motion_residual=float(np.max(np.abs(jamp-1j*gap*eamp))),
                                density_continuity_residual=float(np.max(np.abs(1j*gap*rhoamp+model.incidence[:, :12] @ jamp))),
                                ImW_transition=encoded(amplitudes[selected, 2])),
                  returned_near_group=dict(count=len(near), source_gram=encoded(near_gram),
                                           scope='Only converged directions represented by this finite cyclic block. Not a complete degeneracy group or two physical polarizations.'),
                  krylov=dict(dimension=used, coarse_dimension=coarse,
                              orthogonality_max=float(np.max(np.abs(q.T @ q-np.eye(used)))),
                              converged_E_bright_ray_count=len(eligible)),
                  limitations='No all-poles/multiplicity completeness. Source-selected eigenray and its diagonal pair content only; no photon, spin, speed, Coulomb or continuum identification.')
    arrays = dict(selected_mode=mode, source=source, reduced_H=reduced,
                  projected_source=projected_source, reduced_eigenvalues=values,
                  reduced_residuals=residuals, selected_index=selected)
    return result, arrays

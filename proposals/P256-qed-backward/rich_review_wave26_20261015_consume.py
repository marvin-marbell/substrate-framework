"""Independent finite transfer and one conditional-stationary joint-H mode probe.

Only frozen law construction/helpers are imported; no author CLI or evolved state.
This executable and its JSON input define the experiment, not a photon theorem.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.special import jv
from rich_neutral_mott_gauge_20261015_model import NeutralCube

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave26_20261015_'


def save(name: str, data: Any) -> None:
    with (ROOT / (PREFIX + name + '.json')).open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def maximum(matrix: Any) -> float:
    return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0


def comm(a: Any, b: Any) -> Any:
    return a @ b - b @ a


def real(value: Any) -> float:
    return float(np.real(value))


def chebyshev(h: Any, states: Any, duration: float, degree: int) -> tuple[Any, dict[str, Any]]:
    diagonal = h.diagonal()
    radius = np.asarray(abs(h).sum(axis=1)).ravel() - np.abs(diagonal)
    low, high = float(np.min(diagonal-radius)), float(np.max(diagonal+radius))
    center, scale = (low+high)/2, (high-low)/2
    def action(v: Any) -> Any:
        return (h @ v-center*v)/scale
    previous = states.copy()
    current = action(previous)
    answer = jv(0, scale*duration)*previous + 2*(-1j)*jv(1, scale*duration)*current
    for k in range(2, degree+1):
        following = 2*action(current)-previous
        answer += 2*((-1j)**k)*jv(k, scale*duration)*following
        previous, current = current, following
    answer *= np.exp(-1j*center*duration)
    # Direct Bessel tail evaluation is a finite numerical tail estimate, not an
    # interval-arithmetic theorem. The independent norm/energy drifts are kept.
    tail = 2*np.sum(np.abs(jv(np.arange(degree+1, degree+501), scale*duration)))
    return answer, dict(enclosure=[low, high], degree=degree, numerical_Bessel_tail_500_terms=float(tail))


def independent_census(model: Any) -> dict[str, Any]:
    flux = np.asarray(list(itertools.product((-1, 0, 1), repeat=12)), dtype=np.int8)
    occupation = 1-flux @ model.incidence.T.astype(np.int8)
    allowed = np.all((occupation >= 0) & (occupation <= 2), axis=1)
    flux, occupation = flux[allowed], occupation[allowed]
    singles = np.sum(occupation == 1, axis=1)
    doubles = np.sum(occupation == 2, axis=1)
    counts = np.asarray([math.comb(int(s), int(4-d)) if 0 <= 4-d <= s else 0 for s, d in zip(singles, doubles)], dtype=np.int64)
    physical, multiplicity = np.unique(model.electric, axis=0, return_counts=True)
    keep = counts > 0
    # product enumeration and np.unique both use ascending lexicographic order.
    return dict(all_physical_assignments=3**12, admissible_patterns=int(np.sum(keep)),
                dimension_from_spin_multiplicity=int(np.sum(counts)),
                exact_physical_flux_set_equal=bool(np.array_equal(flux[keep], physical)),
                every_flux_spin_multiplicity_equal=bool(np.array_equal(counts[keep], multiplicity)))


def finite(model: Any, settings: dict[str, Any]) -> None:
    start = time.monotonic()
    seed = model.seed()
    pulse = expm_multiply(-1j*settings['alpha']*model.source, seed, traceA=0)
    initial = np.column_stack((seed, pulse))
    before = model.measure(initial)
    endpoint, resolution = chebyshev(model.H, initial, settings['duration'], settings['degree'])
    after = model.measure(endpoint)
    er = sparse.diags(model.ER, format='csr')
    leftcut = model.Hcut-model.HRcut
    proof = dict(source_ER=maximum(comm(model.source, er)), source_HR=maximum(comm(model.source, model.HR)),
                 off_ER=maximum(comm(model.Hoff, er)), off_HRoff=maximum(comm(model.Hoff, model.HRoff)),
                 source_HRoff=maximum(comm(model.source, model.HRoff)), cut_factorization=maximum(comm(leftcut, model.HRcut)),
                 cut_left_ER=maximum(comm(leftcut, er)), source_HRcut=maximum(comm(model.source, model.HRcut)))
    e = int(model.right_edges[0])
    E = sparse.diags(model.electric[:, e].astype(float), format='csr')
    J = model.currents[e]
    invariants = dict(gauss_max=int(np.max(np.abs(model.electric @ model.incidence.T+model.density-1))),
                      all_physical_edge_max=int(np.max(np.abs(model.electric))),
                      H_hermiticity=maximum(model.H-model.H.T), C_H=maximum(comm(model.C, model.H)),
                      C_source=maximum(comm(model.C, model.source)),
                      C_seed_l2=float(np.linalg.norm(model.C @ seed-seed)), C_pulse_l2=float(np.linalg.norm(model.C @ pulse-pulse)),
                      selected_edge=e, dotE_minus_J=maximum(1j*comm(model.H, E)-J),
                      power_identity=maximum(1j*comm(model.H, E @ E/2)-(E @ J+J @ E)/2))
    delta_er = after['ER'][1]-after['ER'][0]
    delta_hr = after['HR'][1]-after['HR'][0]
    work = before['H'][1]-before['H'][0]
    result = dict(kind='INDEPENDENT_CHEBYSHEV_FINITE_CASE', law=model.LAW_ID if hasattr(model, 'LAW_ID') else 'neutral-mott-cube-uniform-physical-edge-qlink3-20261015',
                  settings=settings, dimension=model.dimension, H_nnz=model.H.nnz, before=before, after=after,
                  delta_ER=delta_er, delta_HR=delta_hr, preparation_work=work,
                  initial_delta_ER=before['ER'][1]-before['ER'][0], initial_delta_HR=before['HR'][1]-before['HR'][0],
                  resolution=resolution, invariants=invariants, complete_physical_census=independent_census(model),
                  exact_own_H_controls=proof,
                  control_meaning='Same-state OFF conserves its own ER/HRoff. In cut H, HLcut commutes with ER and HRcut; source commutes with both receivers. Thus full right trajectories equal with no extra propagation arms.',
                  norm_drift=np.max(np.abs(np.asarray(after['norm'])-np.asarray(before['norm']))).item(),
                  H_drift=np.max(np.abs(np.asarray(after['H'])-np.asarray(before['H']))).item(),
                  scientific_elapsed_seconds=time.monotonic()-start)
    save('finite_result', result)
    np.savez_compressed(ROOT/(PREFIX+'finite_states.npz'), initial=initial, endpoint=endpoint, old_indices=model.old_indices)
    verdict = dict(kind='EARLY_PROVISIONAL_INDEPENDENT_FINITE_VERDICT', delta_ER=delta_er, delta_HR=delta_hr,
                   own_H_control_max=max(proof.values()), full_space_census=result['complete_physical_census'],
                   verdict='Positive finite same-H mixed neutral field/matter redistribution independently reproduced. Zero mean charge/current is C symmetry, not radiation identification. Pair admixture is observed; real on-shell versus virtual dressing is UNRESOLVED. Author wording real pair populations is too strong if it implies on-shell particles.',
                   scope='One newly computed alpha0.7 t4 endpoint with full joint H. No borrowed evolved outputs or author CLI. Stationary probe has not yet run.',
                   result=PREFIX+'finite_result.json')
    save('provisional_synthesis', verdict)
    print(json.dumps(verdict), flush=True)


def parity_basis(model: Any, parity: int) -> Any:
    c = model.C.tocsc()
    target = c.indices
    phase = c.data
    first = np.flatnonzero(np.arange(model.dimension) < target)
    second = target[first]
    if len(first)*2 != model.dimension:
        raise ValueError('Expected particle-hole two-cycles without fixed Fock states')
    columns = np.arange(len(first))
    values = np.concatenate((np.ones(len(first)), parity*phase[first]))/np.sqrt(2)
    return sparse.csr_matrix((values, (np.concatenate((first, second)), np.concatenate((columns, columns)))), shape=(model.dimension, len(first)))


def diagonal_observables(model: Any, v: Any) -> dict[str, Any]:
    p = np.abs(v)**2
    d = model.doubles.sum(axis=1)
    holons = model.holons.sum(axis=1)
    h = real(np.vdot(v, model.H @ v))
    he, hu = real(p @ model.HE), real(p @ model.HU)
    return dict(norm=real(p.sum()), H=h, HE=he, HU=hu, hopping=h-he-hu,
                doublon_total=real(p @ d), doublon_per_site=real(p @ d)/8,
                holon_total=real(p @ holons), charge_variance_total=real(p @ ((model.density-1)**2).sum(axis=1)),
                probability_any_pair=real(p[d > 0].sum()), probability_no_pair=real(p[d == 0].sum()),
                doublons_by_site=np.asarray(model.doubles.T @ p).real.tolist(),
                density_minus_one_by_site=np.asarray((model.density-1).T @ p).real.tolist(), C=real(np.vdot(v, model.C @ v)))


def mode(model: Any, settings: dict[str, Any]) -> None:
    start = time.monotonic()
    plus, minus = parity_basis(model, 1), parity_basis(model, -1)
    hp, hm = (plus.T @ model.H @ plus).tocsr(), (minus.T @ model.H @ minus).tocsr()
    seed = np.asarray(plus.T @ model.seed()).real
    val, vec = eigsh(hp, k=1, which='SA', v0=seed, tol=settings['reference_tolerance'], ncv=settings['ncv'], maxiter=settings['maxiter'])
    g = np.asarray(plus @ vec[:, 0]).ravel()
    energy = float(val[0])
    reference = diagonal_observables(model, g)
    reference['full_H_residual_l2'] = float(np.linalg.norm(model.H @ g-energy*g))
    reference['C_plus_residual_l2'] = float(np.linalg.norm(model.C @ g-g))
    reference['conditional_scope'] = 'Lowest reference found by seed-start C+ eigsh in complete half-filled finite H. Full-H residual verifies stationarity; no global-ground, unique-vacuum, Mott-phase or continuum claim.'
    print(json.dumps(dict(event='stationary_reference_computed', energy=energy, residual=reference['full_H_residual_l2'])), flush=True)
    # Unit Euclidean coefficient norm; it is a circulation observable, not a
    # division of particle number. Its squared source norm is raw spectral weight.
    coefficients = model.left_cycle.astype(float)/np.linalg.norm(model.left_cycle)
    electric = model.electric @ coefficients
    source = electric*g
    source_norm2 = real(np.vdot(source, source))
    current = sum((coefficients[e]*model.currents[e] for e in range(12)), sparse.csr_matrix(model.H.shape, dtype=complex))
    loop_odd = (model.W-model.W.T)/(2j)
    start_odd = np.asarray(minus.T @ source).real.ravel()
    if source_norm2 < 1e-15:
        raise ValueError('Declared field source has no finite stationary-reference strength')
    values, vectors = eigsh(hm, k=settings['returned_modes'], which='SA', v0=start_odd,
                            tol=settings['mode_tolerance'], ncv=settings['ncv'], maxiter=settings['maxiter'])
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    full_vectors = np.asarray(minus @ vectors)
    d = model.doubles.sum(axis=1).astype(float)
    current_source = current @ g
    loop_source = loop_odd @ g
    d_source = d*g
    records = []
    for i, ev in enumerate(values):
        v = full_vectors[:, i]
        gap = float(ev-energy)
        field_amplitude = np.vdot(v, source)
        current_amplitude = np.vdot(v, current_source)
        obs = diagonal_observables(model, v)
        records.append(dict(index=i, H=float(ev), excitation_energy=gap,
                            full_H_residual_l2=float(np.linalg.norm(model.H @ v-ev*v)),
                            C_minus_residual_l2=float(np.linalg.norm(model.C @ v+v)),
                            electric_residue_raw=float(abs(field_amplitude)**2),
                            electric_fraction_of_declared_source=float(abs(field_amplitude)**2/source_norm2),
                            current_residue_raw=float(abs(current_amplitude)**2), loop_ImW_residue_raw=float(abs(np.vdot(v, loop_source))**2),
                            D_transition_squared_selection_rule_only=float(abs(np.vdot(v, d_source))**2),
                            J_equals_i_gap_E_amplitude_residual=float(abs(current_amplitude-1j*gap*field_amplitude)),
                            all_edge_E_residue_raw=[float(abs(np.vdot(v, model.electric[:, e]*g))**2) for e in range(12)],
                            all_edge_J_residue_raw=[float(abs(np.vdot(v, model.currents[e] @ g))**2) for e in range(12)],
                            diagonal=obs,
                            mode_minus_reference={key:obs[key]-reference[key] for key in ('H','HE','HU','hopping','doublon_total','doublon_per_site','holon_total','charge_variance_total','probability_any_pair')}))
    brightest = max(records, key=lambda r:r['electric_residue_raw'])
    captured = sum(r['electric_residue_raw'] for r in records)
    result = dict(kind='ONE_ACTUAL_SAME_H_CONDITIONAL_STATIONARY_FIELD_BRIGHT_NEUTRAL_PROBE', settings=settings,
                  dimension=model.dimension, parity_dimension=plus.shape[1], reference=reference,
                  parity_exactness=dict(Cplus=maximum(model.C @ plus-plus), Cminus=maximum(model.C @ minus+minus),
                                        Hplus_intertwining=maximum(model.H @ plus-plus @ hp), Hminus_intertwining=maximum(model.H @ minus-minus @ hm)),
                  source=dict(operator='Ecirculation=sum cycle_e E_e / sqrt(sum cycle_e^2) on left face', coefficients=coefficients.tolist(),
                              electric_raw_norm_squared=source_norm2, current_raw_norm_squared=real(np.vdot(current_source,current_source)),
                              ImW_raw_norm_squared=real(np.vdot(loop_source,loop_source)),
                              Codd_source_residual_l2=float(np.linalg.norm(model.C @ source+source)),
                              E_mean=real(np.vdot(g,source)),
                              raw_weight_sum_in_returned_vectors=captured, unrepresented_source_norm_squared=source_norm2-captured,
                              D_even_C_commutator=maximum(comm(model.C,sparse.diags(d,format='csr')))),
                  returned_vectors=records, brightest_returned_index=brightest['index'],
                  spectral_scope='Only four returned low C- Ritz vectors. Single-start ARPACK does NOT establish complete degeneracy groups or all bright poles. Captured weights are projections onto these vectors only; residual source norm includes unreturned groups/tail. No full-pole completeness or polarization count.',
                  units='hbar=t=I=1; gap in hopping-energy units; Dtot is expected number of doublon sites, Dtot/8 is per-site occupancy, holon expectation equals doublon at half filling. A fractional D is an expectation, not fractional elementary charge. Residue fraction normalizes spectral strength, never pair number.',
                  interpretation='Diagonal pair admixture and gauge/current residues decide mixed versus pair-free character here. Vanishing C-even D transition to C- is only a selection rule, NOT evidence of virtual-only radiation. Snapshot pair admixture does not settle real-on-shell versus virtual dressing.',
                  scientific_elapsed_seconds=time.monotonic()-start)
    save('mode_result',result)
    np.savez_compressed(ROOT/(PREFIX+'mode_states.npz'), reference=g, mode_vectors=full_vectors, energies=values, old_indices=model.old_indices)
    print(json.dumps(dict(event='mode_probe_complete', brightest=brightest, raw_source=source_norm2, captured=captured, elapsed=result['scientific_elapsed_seconds'])),flush=True)


def main() -> None:
    settings = json.loads(Path(sys.argv[2]).read_text())
    model = NeutralCube()
    if sys.argv[1] == 'finite':
        finite(model,settings['finite'])
    elif sys.argv[1] == 'mode':
        mode(model,settings['mode'])
    else:
        raise ValueError('Explicit finite or mode phase required')


if __name__ == '__main__':
    main()

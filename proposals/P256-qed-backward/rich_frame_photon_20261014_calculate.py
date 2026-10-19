"""New physical calculation, not a test-suite or archived producer replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply

from rich_frame_photon_20261014_model import (
    Parameters, Joined, physical_frame, mode_values, curl_values,
    spectrum, full_resolvent, gram, expectation,
)


def encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {'real': value.real.tolist(), 'imag': value.imag.tolist()}
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def geometry(p: Parameters) -> dict[str, Any]:
    nodes, weights = np.polynomial.hermite.hermgauss(18)
    xx, yy, zz = np.meshgrid(nodes, nodes, nodes, indexing='ij')
    points = p.ell * np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    ww = (weights[:, None, None] * weights[None, :, None] * weights[None, None, :]).ravel()
    compensation = np.exp(np.sum(points * points, axis=1) / p.ell ** 2)
    u, curl = mode_values(points, p.ell), curl_values(points, p.ell)
    norm = np.einsum('aic,bic,i->ab', u, u, ww * compensation) * p.ell ** 3
    curl_gram = np.einsum('aic,bic,i->ab', curl, curl, ww * compensation) * p.ell ** 3
    probe = np.array([[0.27, -0.41, 0.18], [-0.33, 0.22, 0.57]])
    step = 1e-5
    divergence = np.zeros((2, 2))
    for axis in range(3):
        delta = np.eye(3)[axis] * step
        divergence += (mode_values(probe + delta, p.ell)[:, :, axis]
                       - mode_values(probe - delta, p.ell)[:, :, axis]) / (2 * step)
    exact = np.pi ** 1.5 * p.ell ** 5
    return dict(vector_potential_gram=norm, curl_gram=curl_gram,
                canonical_mass=p.epsilon * norm,
                mode_norm_relative_error=float(np.linalg.norm(norm / exact - np.eye(2))),
                curl_relative_error=float(np.linalg.norm(curl_gram / (exact * 2.5 / p.ell ** 2) - np.eye(2))),
                divergence_difference_residual=float(np.linalg.norm(divergence)))


def frame_record() -> dict[str, Any]:
    f = physical_frame()
    projection = np.asarray(f['P'])
    identity = np.eye(4)
    sources = [np.asarray(s) for s in f['source']]
    residual = max(np.linalg.norm(a.conj().T @ b + b @ a.conj().T - (identity if i == j else 0))
                   for i, a in enumerate(sources) for j, b in enumerate(sources))
    return dict(gauss_residual=max(float(np.linalg.norm(g @ projection)) for g in f['G']),
                bare_physical_gram=[(projection.conj().T @ b @ projection).conj().T
                                    @ (projection.conj().T @ b @ projection) for b in f['bare']],
                car_residual=float(residual), physical_spin_squared=sum(s @ s for s in f['S']),
                number=np.asarray(f['N']))


def physical_spin_state(model: Joined, values: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, int]:
    group = vectors[:, values - values[0] < 2e-8]
    _, rotation = eigh(group.conj().T @ (model.S[2] @ group))
    return group @ rotation[:, -1], group.shape[1]


def event(model: Joined, initial: np.ndarray, source_vectors: np.ndarray,
          times: list[float], eta: float, current_strength: float) -> dict[str, Any]:
    pulse = np.exp(1j * eta * np.repeat(model.coordinates[0], model.internal)) * initial
    electric_columns = np.column_stack([p @ initial for p in model.P])
    preparation = electric_columns @ np.array([1., 1j]) / np.sqrt(2)
    preparation_norm = float(np.linalg.norm(preparation))
    preparation /= preparation_norm
    inserted = expm_multiply(-1j * current_strength * model.J[0], preparation)
    seeds = np.column_stack([initial, pulse, preparation, inserted])
    sz_up = model.I / 2 + model.S[2]
    field_positive = np.repeat(model.coordinates[1] >= 0, model.internal)
    histories = []
    trace = complex(model.H.diagonal().sum())
    for time in times:
        evolved = seeds if time == 0 else expm_multiply(-1j * time * model.H, seeds,
                                                        traceA=-1j * time * trace)
        records = []
        for state in evolved.T:
            spin_up = sz_up @ state
            spin_down = state - spin_up
            probabilities = [float(np.vdot(part[mask], part[mask]).real)
                             for part in (spin_up, spin_down)
                             for mask in (field_positive, ~field_positive)]
            records.append(dict(norm=float(np.vdot(state, state).real),
                                energy=expectation(model.H, state), electric_energy=expectation(model.E, state),
                                matter_energy=expectation(model.Hmatter, state),
                                longitudinal_energy=expectation(model.HL, state),
                                body_energy=expectation(model.Hbody, state),
                                spin=[expectation(s, state) for s in model.S],
                                current=[expectation(j, state) for j in model.J],
                                supplier_current=[expectation(j, state) for j in model.Jsupplier],
                                core_current=[expectation(j, state) for j in model.Jcore],
                                field=[expectation(a, state) for a in model.A],
                                electric=[expectation(v / model.mass, state) for v in model.P],
                                inclusive_Sz_field_probabilities=probabilities,
                                source_return_probability=float(np.sum(abs(source_vectors.conj().T @ state) ** 2)),
                                boundary_probability=model.boundary_probability(state)))
        histories.append(dict(time=time, states=records))
    spin_change = np.array([np.array(row['states'][1]['spin']) - row['states'][0]['spin'] for row in histories])
    return dict(state_order=['source_prepared_baseline', 'local_field_unitary', 'normalized_electric_leg',
                             'same_current_unitary_after_electric_preparation'],
                electric_source_gram=gram(electric_columns),
                electric_preparation_squared_norm=preparation_norm ** 2,
                source_gram=gram(source_vectors), histories=histories,
                unitary_spin_difference=spin_change,
                maximum_spin_difference=float(np.max(np.linalg.norm(spin_change, axis=1))))


def run(p: Parameters, config: dict[str, Any], output: Path) -> dict[str, Any]:
    result: dict[str, Any] = dict(parameters=p.json(), geometry=geometry(p), frame=frame_record())
    arrays: dict[str, np.ndarray] = {}
    print('construct neutral and charged common-H sectors', flush=True)
    neutral = Joined(p, 0)
    odd = Joined(p, 1)
    ev0, vec0, quality0 = spectrum(neutral.H, 8)
    ev1, vec1, quality1 = spectrum(odd.H, 10)
    ground0 = vec0[:, 0]
    ground1, multiplicity = physical_spin_state(odd, ev1, vec1)
    source = np.column_stack([neutral.source_to(odd, i) @ ground0 for i in range(2)])
    _, source_rotation = eigh(source.conj().T @ (odd.S[0] @ source))
    prepared = source @ source_rotation[:, -1]
    prepared /= np.linalg.norm(prepared)
    result['spectra'] = dict(neutral=ev0, odd=ev1, neutral_quality=quality0, odd_quality=quality1,
                             odd_ground_multiplicity=multiplicity, addition_threshold=float(ev1[0] - ev0[0]),
                             neutral_dimension=neutral.size, charged_dimension=odd.size,
                             neutral_boundary=neutral.boundary_probability(ground0),
                             charged_boundary=odd.boundary_probability(ground1),
                             neutral_supplier_purity=neutral.supplier_purity(ground0),
                             charged_supplier_purity=odd.supplier_purity(ground1))
    result['source'] = dict(gram=gram(source), low_subspace_weights=abs(vec1.conj().T @ source) ** 2,
                            physical_spin=[source.conj().T @ (s @ source) for s in odd.S],
                            charge_reference={'supplier_plus_particles': 2, 'ions': -2, 'core': -1, 'total': -1})
    print('full-space source and field resolvents', flush=True)
    z = complex(config['frequency'], config['eta'])
    propagated, residual = full_resolvent(odd.H, ev0[0] + z, source)
    result['source']['G'] = source.conj().T @ propagated
    result['source']['G_full_space_residual'] = residual
    threepoint = []
    for current in odd.J:
        response, r = full_resolvent(odd.H, ev0[0] + z + 0.17, current @ propagated)
        threepoint.append(dict(tensor=source.conj().T @ response, residual=r))
    result['source']['current_threepoint'] = threepoint
    field_columns = np.column_stack([(a @ ground1 - expectation(a, ground1) * ground1) for a in odd.A])
    current_columns = np.column_stack([j @ ground1 for j in odd.J])
    electric_columns = np.column_stack([v @ ground1 / np.sqrt(odd.mass) for v in odd.P])
    positive, rp = full_resolvent(odd.H, ev1[0] + z, field_columns)
    negative, rn = full_resolvent(odd.H, ev1[0] - z, field_columns)
    result['field'] = dict(coordinate_connected_gram=gram(field_columns),
                           electric_energy_normalized_gram=gram(electric_columns),
                           total_current_gram=gram(current_columns),
                           retarded_D=field_columns.conj().T @ positive + (field_columns.conj().T @ negative).T,
                           full_space_residual=rp + rn,
                           contact=[[expectation(k, ground1) for k in row] for row in odd.K])
    result['field']['low_spectral_gaps'] = ev1 - ev1[0]
    result['field']['low_coordinate_weights'] = abs(vec1.conj().T @ field_columns) ** 2
    result['field']['low_current_weights'] = abs(vec1.conj().T @ current_columns) ** 2
    result['field']['spectral_convention'] = 'Sum weights across complete equal-energy groups; columns individually are basis-dependent.'
    k = np.array([0.37, -0.24, 0.29])
    rho = odd.density_fourier(k)
    spatial = odd.current_fourier(k)
    continuity = 1j * (odd.H @ rho - rho @ odd.H) + 1j * sum(k[a] * spatial[a] for a in range(3))
    field_longitudinal = [-1j * k[a] * rho / (p.epsilon * np.dot(k, k)) for a in range(3)]
    gauss = 1j * sum(k[a] * field_longitudinal[a] for a in range(3)) - rho / p.epsilon
    delta = 2e-4
    plus, minus = odd.shifted((delta, 0.)), odd.shifted((-delta, 0.))
    current_error = ((plus - minus) / (2 * delta) + odd.J[0]) @ ground1
    contact_error = ((plus + minus - 2 * odd.H) / delta ** 2 - odd.K[0][0]) @ ground1
    result['identities'] = dict(continuity_on_ground=float(np.linalg.norm(continuity @ ground1)),
                                continuity_matrix_norm=float(sp.linalg.norm(continuity)),
                                Fourier_gauss_matrix_norm=float(sp.linalg.norm(gauss)),
                                current_derivative_error=float(np.linalg.norm(current_error)),
                                contact_derivative_error=float(np.linalg.norm(contact_error)),
                                spatial_current_gram=gram(np.column_stack([j @ ground1 for j in spatial])))
    print('normalized field/current/source events and physical controls', flush=True)
    result['events'] = {}
    times = config['times']
    for name, model in [('full', odd), ('induction_off', Joined(p, 1, induction=False)),
                        ('metric_off', Joined(p, 1, metric=False))]:
        # All variants use the SAME physical prepared vector, but their OWN evolved baseline.
        result['events'][name] = event(model, prepared, source, times, config['pulse'], config['current_pulse'])
    # Neutral induction-off has no core contact; its field ground is exact box product.
    bare_field = np.sin(np.pi * (np.arange(p.grid) + 1) / (p.grid + 1))
    bare_field = np.outer(bare_field, bare_field).ravel()
    bare_field /= np.linalg.norm(bare_field)
    neutral_off = Joined(p, 0, induction=False)
    local_diag = neutral_off.HL_values + p.hubbard * np.sum(neutral_off.cube['density'] == 2, axis=1)
    off_state = np.kron(bare_field, np.eye(64)[:, np.argmin(local_diag)])
    result['induction_control'] = dict(neutral_on_coordinate_variance=[expectation(a @ a, ground0) for a in neutral.A],
                                        neutral_off_coordinate_variance=[expectation(a @ a, off_state) for a in neutral_off.A],
                                        neutral_on_current_gram=gram(np.column_stack([j @ ground0 for j in neutral.J])),
                                        neutral_off_current_gram=gram(np.column_stack([j @ off_state for j in neutral_off.J])),
                                        neutral_off_eigen_residual=float(np.linalg.norm(neutral_off.H @ off_state
                                                               - expectation(neutral_off.H, off_state) * off_state)),
                                        no_bare_magnetic_energy=True)
    # Freeze an actual microscopic supplier vector only as an exposing comparator.
    centre = neutral.nfield // 2
    local_slice = slice(centre * 64, (centre + 1) * 64)
    local_h = (neutral.Hmatter + neutral.HL)[local_slice, local_slice].toarray()
    _, local_vectors = eigh(local_h)
    embedding = sp.kron(neutral.I_field, sp.csr_matrix(local_vectors[:, :1]), format='csr')
    frozen_h = (embedding.getH() @ neutral.H @ embedding).tocsr()
    frozen_values, frozen_vectors, frozen_quality = spectrum(frozen_h, 6)
    frozen_ground = embedding @ frozen_vectors[:, 0]
    discarded_current = []
    for current in neutral.J:
        full_current = current @ frozen_ground
        discarded_current.append(float(np.linalg.norm(full_current - embedding @ (embedding.getH() @ full_current))))
    result['frozen_supplier_comparator'] = dict(
        exact64_neutral_gaps=ev0[1:4] - ev0[0],
        frozen_same_supplier_gaps=frozen_values[1:4] - frozen_values[0],
        quality=frozen_quality, discarded_current_norm=discarded_current,
        explanation='One actual phi0 full64 supplier ground vector held fixed; not delivered physical law or projection convergence.')
    # Same complete finite law also supports the charge-2 source sector, not a positron.
    pair = Joined(p, 2)
    ev2, vec2, quality2 = spectrum(pair.H, 4)
    pair_sources = np.column_stack([odd.source_to(pair, i) @ ground1 for i in range(2)])
    pair_response, pair_residual = full_resolvent(pair.H, ev1[0] + z, pair_sources)
    result['pair'] = dict(spectrum=ev2, quality=quality2, charge=-2,
                          addition_from_odd=float(ev2[0] - ev1[0]),
                          source_gram=gram(pair_sources), G=pair_sources.conj().T @ pair_response,
                          residual=pair_residual)
    removed_sources = np.column_stack([neutral.source_to(odd, i).getH() @ ground1 for i in range(2)])
    removed_response, removed_residual = full_resolvent(neutral.H, ev1[0] - z, removed_sources)
    result['odd_complete_CAR_response'] = dict(
        addition_gram=gram(pair_sources), removal_gram=gram(removed_sources),
        CAR_sum_residual=float(np.linalg.norm(gram(pair_sources) + gram(removed_sources).T - np.eye(2))),
        retarded_fermion_G=pair_sources.conj().T @ pair_response
                            - (removed_sources.conj().T @ removed_response).T,
        removal_full_space_residual=removed_residual,
        convention='Gadd=<d_i R(E1+z,H2) d_jdag>; Gremove=<d_jdag (z+H0-E1)^-1 d_i>, same charged reference E1.')
    result['cross_charge'] = dict(core_changes_supplier_diagonal_range=float(np.ptp(odd.HL_values - neutral.HL_values)),
                                  total_longitudinal_energy=expectation(odd.HL, ground1),
                                  neutral_longitudinal_energy=expectation(neutral.HL, ground0))
    arrays.update(neutral_ground=ground0, charged_ground=ground1, neutral_eigenvalues=ev0,
                  charged_eigenvalues=ev1, neutral_eigenvectors=vec0, charged_eigenvectors=vec1,
                  source_vectors=source, prepared_state=prepared, coordinates=odd.coordinates,
                  cube_sites=odd.cube['sites'], cube_phase=odd.cube['phase'],
                  pair_ground=vec2[:, 0], pair_eigenvalues=ev2)
    np.savez_compressed(output.with_suffix('.npz'), **arrays)
    output.write_text(json.dumps(encode(result), indent=2) + '\n')
    print(json.dumps(encode(dict(addition=result['spectra']['addition_threshold'],
                                  field_gram=result['field']['coordinate_connected_gram'],
                                  spin_events={name: record['maximum_spin_difference']
                                               for name, record in result['events'].items()},
                                  identities=result['identities'])), indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    configuration = json.loads(Path(args.input).read_text())
    run(Parameters(**configuration['parameters']), configuration, Path(args.output))

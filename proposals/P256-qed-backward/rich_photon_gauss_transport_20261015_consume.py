"""One spatial cube experiment, first transport then local algebra/resolution."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.special import gammaln, jv
from rich_photon_gauss_transport_20261015_model import CubeExperiment, LAW_ID

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_photon_gauss_transport_20261015_'


def save(name, data):
    (ROOT / (PREFIX + name + '.json')).write_text(json.dumps(data, indent=2) + '\n')


def chebyshev_action(H, state, duration, tolerance=2e-12):
    """Independent polynomial action with Gershgorin enclosure and Bessel tail.

    For n>N, |Jn(z)| <= (z/2)^n/n! exp(z^2/[4(n+1)]).
    Successive majorants have ratio <= z/[2(N+2)]. This bounds truncation,
    not floating-point recurrence roundoff; actual endpoint distance is reported.
    """
    diagonal = H.diagonal()
    radius_rows = np.asarray(np.abs(H).sum(axis=1)).ravel() - np.abs(diagonal)
    lower = float(np.min(diagonal - radius_rows))
    upper = float(np.max(diagonal + radius_rows))
    center = (lower + upper) / 2
    radius = (upper - lower) / 2
    z = abs(radius * duration)
    degree = max(1, int(math.ceil(z)))
    while True:
        n = degree + 1
        ratio = z / (2 * (n + 1))
        log_bound = math.log(2) + n * math.log(z / 2) - float(gammaln(n + 1)) + z*z/(4*(n+1)) - math.log1p(-ratio)
        if log_bound <= math.log(tolerance):
            break
        degree += 1
    tail_bound = math.exp(log_bound)
    scaled = (H - center * sparse.eye(H.shape[0], format='csr')) / radius
    previous = state
    current = scaled @ state
    result = jv(0, z) * previous + 2 * (-1j) * jv(1, z) * current
    for n in range(2, degree + 1):
        following = 2 * (scaled @ current) - previous
        result += 2 * ((-1j)**n) * jv(n, z) * following
        previous, current = current, following
    return np.exp(-1j * center * duration) * result, dict(
        independent_method='Hermitian Chebyshev recurrence, Bessel coefficients',
        enclosure=[lower, upper], polynomial_degree=degree,
        analytic_operator_norm_tail_bound=tail_bound, floating_roundoff_not_in_bound=True)


def difference(record, key):
    values = record[key]
    return float(values[1] - values[0])


def main():
    data = json.loads((ROOT / sys.argv[1]).read_text())
    started = time.monotonic()
    model = CubeExperiment(data['cutoff'])
    s = model.sector
    seed = model.seed()
    pulse = expm_multiply(-1j * data['pulse_alpha'] * model.Bleft, seed, traceA=0)
    initial = np.column_stack((seed, pulse))
    wave = initial.copy()
    times = np.linspace(0, data['duration'], data['intervals'] + 1)
    records = []
    history = []
    dt = float(times[1] - times[0])
    for index, instant in enumerate(times):
        record = model.measure(wave)
        record['time'] = float(instant)
        records.append(record)
        history.append(wave.copy())
        if index < len(times) - 1:
            wave = expm_multiply(-1j * dt * s.H, wave, traceA=-1j * dt * float(s.H.diagonal().sum()))
    # FIRST genuine inter-region physical discriminator, before mode spectra,
    # charged sectors or any attempt to claim a photon dispersion.
    ER_delta = np.array([difference(row, 'ER') for row in records])
    HR_delta = np.array([difference(row, 'HR') for row in records])
    density_delta = np.array([np.array(row['density'][1]) - row['density'][0] for row in records])
    current_delta = np.array([np.array(row['currents'][1]) - row['currents'][0] for row in records])
    peak = int(np.argmax(np.abs(ER_delta)))
    provisional = dict(
        kind='FIRST_ACTUAL_CUBE_FIELD_TRANSFER_DISCRIMINATOR_PROVISIONAL', law_id=LAW_ID,
        governing_plan='issue222-rich-backward-candidates-20261014',
        actual_launch_revision='b8b88c61c08e2622b9a2b7f5355f5259c1e37ca98c344bccf6263896549353d4',
        work_item='photon-candidate', geometry='ACTUAL eight-site twelve-link spatial cube, five compact chords',
        dimension=s.dimension, cutoff=data['cutoff'], input=data,
        preparation='New neutral up-at(0,0,0), down-at(1,0,0), all physical E=0 seed; exp(-i alpha sin Phi_left) field-only pulse. Not a ground state, Dirac sea or historical evolved state.',
        receiver='Disjoint x=1 yz face electric energy; endpoint-owned right total energy separately retained',
        max_abs_receiver_electric_difference=float(abs(ER_delta[peak])),
        time_of_peak=float(times[peak]), receiver_electric_difference_at_peak=float(ER_delta[peak]),
        max_abs_receiver_owned_energy_difference=float(np.max(np.abs(HR_delta))),
        max_abs_right_edge_current_difference=float(np.max(np.abs(current_delta[:, model.right_edges]))),
        max_abs_right_density_difference=float(np.max(np.abs(density_delta[:, model.right_vertices]))),
        initial_receiver_electric_difference=float(ER_delta[0]),
        initial_receiver_owned_energy_difference=float(HR_delta[0]),
        pulse_injected_full_H_energy=difference(records[0], 'H'),
        decision='Finite spatial coupled response observed; algebraic locality/control and independent numerical endpoint still being completed in final result.',
        scope='No modes, charged G, masslessness, fitted speed or continuum claim. Distinct law from CAR36 and older oscillator/cube-API-only deliveries.')
    save('provisional_synthesis', provisional)
    print(json.dumps(provisional), flush=True)
    algebra = model.algebra()
    independent, numerical = chebyshev_action(s.H, initial, data['duration'])
    independent_record = model.measure(independent)
    numerical['endpoint_state_l2_difference_per_column'] = np.linalg.norm(independent - wave, axis=0).tolist()
    numerical['endpoint_receiver_electric_difference_between_methods'] = abs(difference(independent_record, 'ER') - difference(records[-1], 'ER'))
    numerical['endpoint_owned_energy_difference_between_methods'] = abs(difference(independent_record, 'HR') - difference(records[-1], 'HR'))
    # A full matter reduced state, not just population, checks field-only support.
    seed_matrix = seed.reshape(len(s.fock), s.nflux)
    pulse_matrix = pulse.reshape(len(s.fock), s.nflux)
    matter_reduced_distance = float(np.linalg.norm(pulse_matrix @ pulse_matrix.conj().T
                                                  - seed_matrix @ seed_matrix.conj().T))
    numerical['max_norm_drift'] = max(float(np.max(np.abs(np.array(row['norm']) - 1))) for row in records)
    numerical['max_full_energy_drift'] = max(float(np.max(np.abs(np.array(row['H']) - records[0]['H']))) for row in records)
    numerical['method'] = 'SciPy expm_multiply adaptive scaled Taylor at each sample, independently checked by global Chebyshev action; not a fixed Krylov schedule'
    exact_off = all(algebra[key] == 0 for key in ('off_receiver_energy_conserved', 'off_receiver_electric_conserved', 'source_receiver_electric_commutator'))
    exact_cut = all(algebra[key] == 0 for key in ('cut_left_right_commutator', 'cut_source_right_commutator', 'cut_left_receiver_electric_commutator', 'source_receiver_electric_commutator'))
    result = dict(
        kind='ACTUAL_LITERAL_CUBE_INTERREGION_GAUSS_TRANSPORT_RESULT', law_id=LAW_ID,
        input=data, geometry=dict(positions=s.positions.tolist(), oriented_edges=s.edges,
                                 tree=s.tree, chords=s.chords, background=s.background.tolist(),
                                 source_vertices=model.left_vertices, receiver_vertices=model.right_vertices,
                                 oriented_source_face=model.left_cycle.tolist(), oriented_receiver_face=model.right_cycle.tolist(),
                                 source_edges=model.left_edges.tolist(), receiver_edges=model.right_edges.tolist(), bridge_edges=model.bridge_edges.tolist()),
        dimension=s.dimension, H_nnz=s.H.nnz,
        full_H_new_preparation=dict(seed_fock_indices=np.flatnonzero(seed).tolist(),
                                   seed_E_max=float(np.max(np.abs(s.electric[np.flatnonzero(seed)]))),
                                   matter_reduced_state_initial_distance=matter_reduced_distance,
                                   initial_receiver_E2_probability_invariance=float(np.sum(np.abs(pulse[model.ER != 0])**2)),
                                   pulse_alpha=data['pulse_alpha'], generator='Bleft=(P product physical U around x0 yz face P - adjoint)/(2i)',
                                   ground_state_claim=False, reference='finite prepared state, exact full-H motion'),
        discriminator=provisional, algebra=algebra, numerical=numerical,
        controls=dict(same_state_hopping_off=dict(actual_H='Hoff=sum E_e^2/2 + U sum doubles',
                                                  actual_receiver_energy='HRoff=sum endpoint_weight E_e^2/2 + U doubles_right',
                                                  exact_same_initial_states=True, exact_no_receiver_difference=exact_off,
                                                  proof='All receiver electric/owned energies diagonal and conserved; pulse commutes with receiver energy. Thus their pulse-baseline difference stays its initial zero for all t, without a duplicated trajectory.'),
                      same_state_plane_cut=dict(removed_physical_edges=model.bridge_edges.tolist(),
                                                actual_H='Hcut=H minus literal forward/backward hopping only on four x links',
                                                actual_receiver_energy='HRcut=HRoff + hopping on four right face links',
                                                exact_same_initial_states=True, exact_no_receiver_difference=exact_cut,
                                                proof='[HLcut,HRcut]=0, [HLcut,ER]=0, [Bleft,HRcut]=[Bleft,ER]=0. Receiver evolution is HRcut only and its entire ER trajectory is unchanged by the pulse. HRcut itself is conserved.'),
                      susceptibility_warning='E2 rotors already have finite sin-flux transitions; only spatial pulse-baseline transfer and same-state controls count here.'),
        samples=records, wall_seconds=time.monotonic() - started,
        finite_model_limitations=['cutoff is a declared L=1 five-chord projection, not a convergence/rotor-limit certificate',
                                  'all full retained spinful Fock and flux states evolve; no Maxwell B2 or fitted velocity',
                                  'initial state is finite prepared neutral state, not stationary vacuum or sea',
                                  'finite spatial gauge/matter transport is not massless photons, two polarizations or Nature closure'])
    save('actual_result', result)
    np.savez_compressed(ROOT / (PREFIX + 'actual_states.npz'), initial=initial, endpoint=wave,
                        independent_endpoint=independent, times=times, states=np.asarray(history))
    print(json.dumps(dict(completed=True, wall_seconds=result['wall_seconds'], algebra=algebra, numerical=numerical,
                          exact_off=exact_off, exact_cut=exact_cut)), flush=True)


if __name__ == '__main__':
    main()

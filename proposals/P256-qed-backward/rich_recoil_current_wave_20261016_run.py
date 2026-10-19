"""Fresh coupled 108-CAR current-dressed wavepackets and matched neutral/control."""
from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
import pickle
import resource
import sys
import time
from typing import Any, cast
import numpy as np
from scipy.linalg import expm
from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
from tenpy.linalg import np_conserved as npc
from rich_recoil_current_wave_20261016_model import Law, prepare_current_space
from rich_recoil_current_wave_20261016_observe import observe, CoherentWave

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_recoil_current_wave_20261016_'


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2)+'\n')


def save(path: Path, value: Any) -> None:
    with path.open('wb') as stream:
        pickle.dump(value, stream, protocol=5)


def normalize(wave: dict[int, Any]) -> float:
    norm2 = float(sum(abs(psi.norm)**2 for psi in wave.values()))
    for psi in wave.values():
        psi.norm /= np.sqrt(norm2)
    return norm2


def fixed_charge_unitary(psi: Any, site_index: int, matrix: np.ndarray) -> None:
    site = psi.sites[site_index]
    q = site.leg.to_qflat()
    allowed = (q[:, None, 0] == q[None, :, 0]) & ((q[:, None, 1]-q[None, :, 1]) % 3 == 0)
    if np.linalg.norm(matrix[~allowed]) > 1e-12*max(1., np.linalg.norm(matrix)):
        raise ValueError('The fixed control unitary violated its actual N,C3 charges')
    operator = npc.Array.from_ndarray(np.where(allowed, matrix, 0.), [site.leg, site.leg.conj()],
        qtotal=[0, 0], labels=['p', 'p*'])
    psi.apply_local_op(site_index, operator, unitary=True, renormalize=False)


def frozen_step(law: Law, wave: dict[int, Any], dt: float) -> None:
    field = expm(-1j*dt*law.field['boson'])
    for psi in wave.values():
        fixed_charge_unitary(psi, 0, field)
        for i, energy in enumerate(law.one['energies']):
            fixed_charge_unitary(psi, i+1, np.diag([1., np.exp(-1j*dt*energy)]))
        # Same constant offset as H; retain absolute phase for coherent amplitudes.
        fixed_charge_unitary(psi, 0, np.eye(len(field))*np.exp(1j*dt*law.one['offset']))


def field_only_data(law: Law, wave: dict[int, Any]) -> dict[str, Any]:
    frame = CoherentWave(wave)
    one = frame.conditional_one_body(np.eye(len(law.field['momentum'])))
    u = law.one['orbitals']
    density = np.einsum('ai,ij,aj->a', u.conj(), one, u, optimize=True).real.reshape(27, 4).sum(axis=1)
    return dict(norm=float(np.trace(frame.field_density()).real), density=density.tolist(),
                momentum=frame.momentum(law), reference_depletion_P0_P1_Pge2=frame.sea_depletion(law))


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    started = time.time()
    input_path = Path(sys.argv[1])
    inputs = json.loads(input_path.read_text())
    stamp = inputs['stamp']
    stem = ROOT/(PREFIX+stamp+'_')
    law = Law(inputs['law'])
    dump(Path(str(stem)+'construction.json'), dict(epochs=law.epochs, orbital_unitarity=law.one['orbital_unitarity'],
        single_particle_labels=law.one['labels'], supplied_law=inputs['law'], versions=inputs['versions'],
        physical_states='One full108-mode CAR chain, exact N and physical finite-C3 charges; no auxiliary COM and no old27/162 source-span closure.',
        neutral='Fresh filled negative-band Slater at phi_center,A=0 times scalar kinetic/potential ground and Hermite-shell0 field packet; a PREPARED state, not the ground of the interacting enlarged ansatz.'))
    print(json.dumps(dict(stage='constructed_full_CAR_law', epochs=law.epochs, seconds=time.time()-started)), flush=True)
    save(Path(str(stem)+'constructed_model.pkl'), law)
    prepared = {}
    source_norms = {}
    for number in inputs['numbers']:
        for protocol in inputs['protocols']:
            pol = None if protocol == 'baseline' else {'xcos': 0, 'ycos': 1}[protocol]
            name = f'N{number}_{protocol}'
            prepared[name] = law.prepared(number, pol)
            source_norms[name] = normalize(prepared[name])
    car = {protocol: source_norms[f'N55_{protocol}']+source_norms[f'N53_{protocol}'] for protocol in inputs['protocols']}
    packet = law.packet()
    coefficient_norm2 = float(np.vdot(packet, packet).real)
    expected_car = {protocol: coefficient_norm2*source_norms[f'N54_{protocol}'] for protocol in inputs['protocols']}
    if any(abs(value-expected_car[protocol]) > 1e-10*max(1., expected_car[protocol]) for protocol, value in car.items()):
        raise ValueError(f'The actual canonical packet source failed its coefficient/reference-normalized CAR sum: {car}')
    dump(Path(str(stem)+'prepared_rays.json'), dict(raw_source_norms=source_norms, raw_CAR_sum=car, actual_expected_CAR_sum=expected_car,
        scope='All canonical orders and field-only protocols are physically prepared before any TDVP; source MPS carries explicit global JW/N/C3 charges.'))
    print(json.dumps(dict(stage='prepared_all_actual_rays', raw_source_norms=source_norms, raw_CAR_sum=car)), flush=True)
    results: dict[str, Any] = {}
    for number in inputs['numbers']:
        for protocol in inputs['protocols']:
            name = f'N{number}_{protocol}'
            wave = {r: psi.copy() for r, psi in prepared[name].items()}
            source_norm2 = source_norms[name]
            initial = {r: psi.copy() for r, psi in wave.items()}
            control = {r: psi.copy() for r, psi in wave.items()}
            save(Path(str(stem)+name+'_initial.pkl'), initial)
            initial_observation, initial_arrays = observe(law, wave)
            np.savez_compressed(Path(str(stem)+name+'_initial_arrays.npz'), **cast(Any, initial_arrays))
            initial_frozen = field_only_data(law, control)
            seed = {}
            engines = {}
            for r, psi in list(wave.items()):
                wave[r], seed[r] = prepare_current_space(law, psi, inputs['minimum_bond'], inputs['short_start'])
                options = dict(dt=inputs['dt'], N_steps=1, trunc_params=dict(chi_max=seed[r]['chi_max'], svd_min=1e-12),
                    lanczos_params=dict(N_max=24, P_tol=1e-12), preserve_norm=True)
                engines[r] = TwoSiteTDVPEngine(wave[r], law.model, options)
            seed_overlap = sum(initial[r].overlap(wave[r]) for r in wave)
            short_observation, short_arrays = observe(law, wave, expensive=False)
            frozen_step(law, control, inputs['short_start'])
            observations = [initial_observation, short_observation]
            frozen = [initial_frozen, field_only_data(law, control)]
            evolution = []
            checkpoints = [0., inputs['short_start']]
            np.savez_compressed(Path(str(stem)+name+'_short_start_arrays.npz'), **cast(Any, short_arrays))
            save(Path(str(stem)+name+'_current_seed.pkl'), wave)
            dump(Path(str(stem)+name+'_seed.json'), dict(seed=seed, first_physical_step_overlap=[float(seed_overlap.real), float(seed_overlap.imag)], raw_source_norm2=source_norm2))
            print(json.dumps(dict(stage='current_seeded', name=name, seed=seed, seconds=time.time()-started)), flush=True)
            steps = int(round((inputs['duration']-inputs['short_start'])/inputs['dt']))
            for step in range(1, steps+1):
                for engine in engines.values():
                    engine.run()
                absolute_time = inputs['short_start']+step*inputs['dt']
                frozen_step(law, control, inputs['dt'])
                observation, arrays = observe(law, wave, expensive=step == steps)
                controls = field_only_data(law, control)
                observations.append(observation)
                frozen.append(controls)
                checkpoints.append(absolute_time)
                evolution.append({r: dict(evolved_time=inputs['short_start']+float(np.real(engine.evolved_time)), truncation_eps=float(engine.trunc_err.eps), max_bond=max(engine.psi.chi)) for r, engine in engines.items()})
                save(Path(str(stem)+name+f'_t{step}.pkl'), wave)
                np.savez_compressed(Path(str(stem)+name+f'_t{step}_arrays.npz'), **cast(Any, arrays))
                dump(Path(str(stem)+name+'_checkpoint.json'), dict(absolute_time=absolute_time, observations=observations, control=frozen, evolution=evolution))
                print(json.dumps(dict(stage='evolved', name=name, absolute_time=absolute_time,
                    matter_momentum=observation['momentum']['matter'], field_momentum=observation['momentum']['field'],
                    total_momentum=observation['momentum']['total'], depletion=observation['reference_depletion_P0_P1_Pge2'],
                    spin=observation['primitive_frame_spin'], seconds=time.time()-started)), flush=True)
            results[name] = dict(number=number, protocol=protocol, raw_source_norm2=source_norm2,
                initial_seed=seed, times=checkpoints, observations=observations, frozen_interaction_control=frozen, evolution=evolution)
            dump(Path(str(stem)+'result.json'), dict(kind='NEW_FULL_CAR_CURRENT_DRESSED_RECOIL_CONSTRUCTION', input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                cases=results, elapsed_seconds=time.time()-started, peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    paired = {}
    for number in [n for n in inputs['numbers'] if n != 54]:
        for protocol in inputs['protocols']:
            name = f'N{number}_{protocol}'
            neutral = results[f'N54_{protocol}']
            case = results[name]
            paired[name] = dict(excess_density=[(np.array(c['density'])-n['density']).tolist() for c, n in zip(case['observations'], neutral['observations'])],
                excess_semantics='Physical N55-N54 or N53-N54 density against the SAME-protocol neutral propagated under this new full-CAR MPS law at the SAME absolute clock; signed excess is not n/N probability.')
    dump(Path(str(stem)+'result.json'), dict(kind='NEW_FULL_CAR_CURRENT_DRESSED_RECOIL_CONSTRUCTION',
        input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(), cases=results, matched_neutral=paired,
        elapsed_seconds=time.time()-started, peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        conditioning='Finite MPS/TDVP projection with recorded exact initial current images, bond ranks and truncations. No equality to old finiteH or full-Fock convergence claim.',
        preparation='Finite projected-Q-only unitary, not full-W or a pure/asymptotic photon. Source norms are preserved separately before charged normalization.',
        control='Same chosen initial rays; matter coefficients frozen at phi_center,A=0, actual scalar/EM kinetic retained. Owning coupled current/contact are NOT called frozen-H physical currents.'))
    print(json.dumps(dict(stage='complete', elapsed_seconds=time.time()-started, cases=list(results))), flush=True)


if __name__ == '__main__':
    main()

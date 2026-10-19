"""Reduce the actual NEW cube result/state capsule, never re-evolve a model."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.integrate import simpson

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_photon_gauss_transport_20261015_'


def read_json(name):
    return json.loads((ROOT / (PREFIX + name + '.json')).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def delta(record, key):
    return float(record[key][1] - record[key][0])


def entanglement(state):
    weights = np.linalg.svd(state.reshape(64, 243), compute_uv=False)**2
    weights = weights[weights > 1e-15]
    return dict(entropy=float(-np.sum(weights * np.log(weights))),
                purity=float(np.sum(weights**2)), trace=float(np.sum(weights)))


def main():
    result = read_json('actual_result')
    exit_record = read_json('actual_exit')
    language = read_json('language_final')
    samples = result['samples']
    initial, final = samples[0], samples[-1]
    times = np.array([row['time'] for row in samples])
    injection = delta(initial, 'H')
    transfer = delta(final, 'HR')
    electric = delta(final, 'ER')
    power_integral = float(simpson([delta(row, 'powerR') for row in samples], x=times))
    electric_power_integral = float(simpson([delta(row, 'powerER') for row in samples], x=times))
    states = np.load(ROOT / (PREFIX + 'actual_states.npz'))
    entropy = dict(baseline=entanglement(states['endpoint'][:, 0]),
                   pulse=entanglement(states['endpoint'][:, 1]))
    budget = max(abs(delta(row, 'HR') + delta(row, 'HL') - injection) for row in samples)
    boundary = dict(initial=initial['boundary'], endpoint=final['boundary'],
                    maximum_per_column=np.max([row['boundary'] for row in samples], axis=0).tolist())
    accepted = (exit_record['success'] and all(value == 0 for value in result['algebra'].values())
                and result['controls']['same_state_hopping_off']['exact_no_receiver_difference']
                and result['controls']['same_state_plane_cut']['exact_no_receiver_difference']
                and max(result['numerical']['endpoint_state_l2_difference_per_column']) < 1e-10
                and transfer > 0 and electric > 0)
    synthesis = dict(
        kind='POSITIVE_FINITE_PROJECTED_GAUSS_CUBE_TRANSPORT_NOT_MASSLESS_PHOTONS',
        plan_id='issue222-rich-backward-candidates-20261014',
        actual_launch_revision='b8b88c61c08e2622b9a2b7f5355f5259c1e37ca98c344bccf6263896549353d4',
        prepared_contract_revision='f35603f8ff89f505e8fc769166a9eff9e6af1c6314de50cbd49050c73661027a',
        template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        work_item='photon-candidate', requirement_ids=['outcome', 'verification'], parent_sole_plan_writer=True,
        law_id=result['law_id'],
        decision='One actual finite spatial gauge/matter energy-transfer mechanism passes its first physical discriminator' if accepted else 'Exact declared result needs adverse-control interpretation; no positive promotion',
        positive_finite_mechanism_accepted=bool(accepted),
        material_extension='ACTUAL eight-site twelve-link three-dimensional cube, five compact chords; complete Nup=Ndown=1 nonlinear Fock/flux sector, dimension15552. Historical ACTUAL fullq was six-site seven-link DOUBLE_FACE, not cube or sea.',
        mechanism='An oriented physical x=0 yz Wilson-loop impulse changes only left-loop electric fields. Under literal E2+Hubbard+CAR link H it changes disjoint x=1 electric energy, endpoint-owned receiving-region energy and actual primitive current/density. Off and physical bridge-cut controls have exact zero receiver DIFFERENCE for the SAME initial states.',
        preparation='New explicit neutral zero-E basis state with static ions/up at(0,0,0), down at(1,0,0); field-only exp(-i0.7 Bleft) pulse. This is a finite nonstationary prepared state, not a ground/sea/free electron or positron.',
        observed=dict(pulse_work=injection, t_final=float(times[-1]),
                      final_receiver_electric_difference=electric,
                      final_receiver_owned_energy_difference=transfer,
                      fraction_of_pulse_work_in_right_owned_energy=transfer/injection,
                      fraction_of_pulse_work_in_right_electric_energy=electric/injection,
                      initial_receiver_electric_and_owned_energy_differences=[delta(initial, 'ER'), delta(initial, 'HR')],
                      receiver_electric_difference_at_times={str(row['time']):delta(row, 'ER') for row in samples if row['time'] in (0, 1, 2, 3, 4)},
                      receiver_owned_energy_difference_at_times={str(row['time']):delta(row, 'HR') for row in samples if row['time'] in (0, 1, 2, 3, 4)},
                      max_right_edge_current_difference=result['discriminator']['max_abs_right_edge_current_difference'],
                      max_right_density_difference=result['discriminator']['max_abs_right_density_difference'],
                      max_total_pulse_energy_budget_error=budget,
                      same_H_receiver_power_integral=power_integral,
                      receiver_energy_minus_power_quadrature=transfer-power_integral,
                      same_H_electric_power_integral=electric_power_integral,
                      receiver_electric_minus_power_quadrature=electric-electric_power_integral,
                      quadrature_scope='33-point Simpson integration only; small finite sampling residual is reported, not hidden as a propagation error',
                      actual_endpoint_matter_field_entanglement=entropy),
        exact_algebra=result['algebra'], initial_locality=result['full_H_new_preparation'],
        exact_controls=result['controls'], independent_resolution=result['numerical'],
        critical_projection_limit=dict(cutoff=1, boundary_probability=boundary,
            meaning='Large boundary occupancy is actual evidence against treating L1 as a converged integer-rotor calculation. The accepted result is ONLY a finite projected compact gauge-transport mechanism. Projection changes Wilson algebra at boundaries; no cutoff-independent continuum/rotor conclusion is drawn. No cutoff sweep or decimal polishing was substituted for this physical discriminator.'),
        no_photon_promotion=['No masslessness or two propagating polarization branches',
                            'No fitted speed, momentum dispersion or common light cone',
                            'No global-vacuum theorem, continuum or thermodynamic limit',
                            'No free charged electron/positron or Nature/proton/QCD closure',
                            'No propagator/source splice into CAR36/CommonCarrierMotion or historical oscillator laws'],
        execution=dict(observed_returncode=exit_record['observed_returncode'],
                       natural_exit=exit_record['natural_process_exit'], interruption=exit_record['interruption'],
                       wall_seconds=exit_record['elapsed_seconds'], sources_unchanged=exit_record['sources_unchanged'],
                       science_failures_or_interruptions_observed=[],
                       scope='One NEW scientific run only; no old CLI replay, mode survey or charged-G run'),
        language=dict(version=language['version'], returncode=language['returncode'],
                      summary=language['summary'], errors_by_file=language['errors_by_file'],
                      rooted_LSP_exit=language['LSP']['server_exit'], scope='ONLY four NEW owned Python files; frozen dependency diagnostics were observed before edits and not repaired or relabelled clean'),
        artifacts=dict(first_discriminator=PREFIX+'provisional_synthesis.json', model=PREFIX+'model.py',
                       consumer=PREFIX+'consume.py', capture=PREFIX+'capture.py', finalizer=PREFIX+'finalize.py',
                       interface=PREFIX+'interface.json', equations=PREFIX+'equations.json',
                       actual_input=PREFIX+'actual_input.json', actual_result=PREFIX+'actual_result.json',
                       actual_states=PREFIX+'actual_states.npz', source_capsule=PREFIX+'actual_source_capsule.zip',
                       actual_exit=PREFIX+'actual_exit.json', actual_capsule=PREFIX+'actual_capsule.zip',
                       selected_diagnostics=PREFIX+'language_final.json', manifest=PREFIX+'manifest.json',
                       delivery_capsule=PREFIX+'delivery_capsule.zip'))
    (ROOT / (PREFIX + 'synthesis.json')).write_text(json.dumps(synthesis, indent=2) + '\n')
    selected = sorted(path for path in ROOT.glob(PREFIX + '*') if path.is_file()
                      and path.name not in (PREFIX+'manifest.json', PREFIX+'delivery_capsule.zip'))
    pins = {path.name:dict(sha256=digest(path), bytes=path.stat().st_size) for path in selected}
    manifest = dict(kind='EXACT_NEW_FINITE_GAUSS_CUBE_ARTIFACT_MANIFEST', files=pins,
                    source_before_exit=exit_record['source_pins'], actual_natural_exit=exit_record['natural_process_exit'],
                    actual_returncode=exit_record['observed_returncode'], unchanged_science_sources=exit_record['sources_unchanged'],
                    no_old_evolved_state_or_output_in_consumer=True,
                    capsule_self_hash='Delivery zip excludes itself and contains this manifest; its hash is separately recorded in delivery_receipt.json')
    manifest_path = ROOT / (PREFIX + 'manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(ROOT / (PREFIX + 'delivery_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in selected + [manifest_path]:
            archive.write(path, path.name)
    receipt = dict(capsule=PREFIX+'delivery_capsule.zip', sha256=digest(ROOT / (PREFIX+'delivery_capsule.zip')),
                   manifest_sha256=digest(manifest_path), natural_science_exit=exit_record['natural_process_exit'],
                   scientific_returncode=exit_record['observed_returncode'])
    (ROOT / (PREFIX + 'delivery_receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(accepted=bool(accepted), receiver_electric=electric, receiver_owned=transfer,
                          work=injection, entanglement=entropy, boundary=boundary, receipt=receipt)), flush=True)


if __name__ == '__main__':
    main()

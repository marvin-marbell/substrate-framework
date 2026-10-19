"""Central evolved-current readout and exact primitive TN images, no dynamics.
Optional expanded-image catalogues are not prerequisites or invented metrics.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any
import numpy as np
from scipy.linalg import expm
from rich_spatial_translation_20261015_model import shell_fields

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_recoil_current_wave_20261016_'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def pair(value: complex) -> list[float]:
    return [float(np.real(value)), float(np.imag(value))]


def complex_value(value: list[float]) -> complex:
    return complex(value[0], value[1])


def insertion_coefficients(law: Any) -> list[dict[str, Any]]:
    # Same owning before-cut products; independent of H's primitive span.
    extended: Any = shell_fields(dict(law.parameters, shell_cutoff=law.parameters['shell_cutoff']+1))
    source = law.field['raw']['shell_states']
    target = extended['shell_states']
    indices = np.array([p*len(target)+target.index(s) for p in range(extended['nphi']) for s in source])
    v = law.field['transform']
    result = []
    for vertex in law.emission:
        pol, sign = vertex['polarization'], vertex['profile_sign']
        current = [-(extended['derivatives'][2*pol][c]+1j*sign*extended['derivatives'][2*pol+1][c])/np.sqrt(2) for c in range(16)]
        qplus = (extended['coordinates'][2*pol]+1j*extended['coordinates'][2*pol+1])/np.sqrt(2)
        counter = qplus.conj().T if vertex['current_C3_charge'] == 1 else qplus
        compress = lambda a: v.conj().T @ a.toarray()[np.ix_(indices, indices)] @ v
        result.append(dict(polarization=pol, profile_sign=sign, actual_current_C3_charge=vertex['current_C3_charge'],
            J=[compress(a) for a in current], QJ=[compress(counter@a) for a in current], JQ=[compress(a@counter) for a in current]))
    return result


def frozen_original_law_probe(law: Any, initial_one: np.ndarray, initial_field: np.ndarray,
                              time: float, coefficients: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Analytical readout of the ALREADY specified factorized control generator
    # on its OWN clock0 state. This is not a new preparation or TDVP trajectory.
    energy = law.one['energies']
    one = initial_one*np.exp(1j*time*(energy[:, None]-energy[None, :]))
    u = expm(-1j*time*law.field['boson'])
    rho = u@initial_field@u.conj().T
    matter = np.zeros(16, complex)
    for block in range(9):
        sl = slice(12*block, 12*block+12)
        matter += np.einsum('cij,ij->c', law.one['components'][block], one[sl, sl], optimize=True)
    result = []
    for vertex in coefficients:
        row = {key: vertex[key] for key in ['polarization', 'profile_sign', 'actual_current_C3_charge']}
        for family in ['J', 'QJ', 'JQ']:
            row[family] = pair(sum(matter[c]*np.trace(rho@f) for c, f in enumerate(vertex[family])))
        result.append(row)
    return result


def coupled_insertions(observation: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(observation['signed_current_field_emission']):
        pol, sign = row['polarization'], row['profile_sign']
        j = (observation['currents'][2*pol]+1j*sign*observation['currents'][2*pol+1])/np.sqrt(2)
        result.append(dict(polarization=pol, profile_sign=sign, actual_current_C3_charge=row['actual_current_C3_charge'],
            J=pair(j), QJ=row['QJ'], JQ=row['JQ']))
    return result


def subtract(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for x, y in zip(a, b):
        row = {key: x[key] for key in ['polarization', 'profile_sign', 'actual_current_C3_charge']}
        row.update({family: pair(complex_value(x[family])-complex_value(y[family])) for family in ['J', 'QJ', 'JQ']})
        result.append(row)
    return result


def main() -> None:
    physical_stamp, output_stamp = sys.argv[1:3]
    source_stem = ROOT/(PREFIX+physical_stamp+'_')
    output = ROOT/(PREFIX+output_stamp+'_')
    source_path = Path(str(source_stem)+'result.json')
    completed = json.loads(source_path.read_text())
    if len(completed['cases']) != 9:
        raise ValueError('The completed physical source must contain both CAR branches and matched neutral, both T protocols')
    model_path = Path(str(source_stem)+'constructed_model.pkl')
    with model_path.open('rb') as stream:
        law = pickle.load(stream)
    model_hash = sha(model_path)
    coefficients = insertion_coefficients(law)
    cases = {}
    descriptors = []
    for name, case in completed['cases'].items():
        initial_path = Path(str(source_stem)+name+'_initial_arrays.npz')
        with np.load(initial_path) as saved:
            initial_one, initial_field = saved['one_body_cdag_c'], saved['field_density']
        endpoints = []
        for index in [0, -1]:
            observation = case['observations'][index]
            time = case['times'][index]
            coupled = coupled_insertions(observation)
            frozen = frozen_original_law_probe(law, initial_one, initial_field, time, coefficients)
            endpoints.append(dict(absolute_time=time, normalized_state_norm=observation['norm'],
                coupled=coupled, original_coupled_law_probe_on_frozen_control=frozen,
                coupled_minus_frozen_original_law_probe=subtract(coupled, frozen),
                owning_frozen_H_field_source_derivative='zero by removed feedback; not zero physical particle/convection current'))
        cases[name] = dict(number=case['number'], protocol=case['protocol'], raw_CAR_source_norm2=case['raw_source_norm2'], endpoints=endpoints)
        if case['number'] in [55, 53] and case['protocol'] == 'baseline':
            for endpoint, index in [('initial', 0), (f't{len(case["evolution"])}', -1)]:
                state_path = Path(str(source_stem)+name+'_'+endpoint+'.pkl')
                state_hash = sha(state_path)
                for vertex_index, vertex in enumerate(law.emission):
                    for family in ['current', 'QJ', 'JQ']:
                        for r in case['initial_seed']:
                            source_charge = int(r)
                            output_charge = (source_charge+vertex['current_C3_charge']) % 3 if family == 'current' else source_charge
                            materialized_family = 'J' if family == 'current' else family
                            partial = ROOT/(PREFIX+f'emit2_N{case["number"]}_{endpoint}_{materialized_family}{vertex_index}_r{output_charge}.pkl.primitive.json')
                            descriptors.append(dict(source_model=model_path.name, source_model_sha256=model_hash,
                                source_state=state_path.name, source_state_sha256=state_hash, absolute_time=case['times'][index],
                                source_total_C3=source_charge, output_total_C3=output_charge,
                                operator_index=vertex_index, operator_family=family,
                                representation='EXACT uncontracted two-layer saved MPO x saved MPS primitive image; no old-span projection or new cap.',
                                existing_committed_materialization_receipt=partial.name if partial.exists() else None,
                                optional_expanded_norm_Gram_spin_depletion_status='Not asserted from this descriptor. Only separately completed catalogue records are exercised metrics.'))
    for name, row in cases.items():
        neutral = cases['N54_'+row['protocol']]
        for endpoint, baseline in zip(row['endpoints'], neutral['endpoints']):
            endpoint['charge_minus_propagated_matched_neutral'] = subtract(endpoint['coupled'], baseline['coupled'])
    pulse_contrasts = {}
    for number in [55, 53]:
        base = cases[f'N{number}_baseline']
        for protocol in ['xcos', 'ycos']:
            drive = cases[f'N{number}_{protocol}']
            pulse_contrasts[f'N{number}_{protocol}'] = [dict(absolute_time=a['absolute_time'],
                coupled_pulse_minus_baseline=subtract(a['coupled'], b['coupled']),
                charge_excess_pulse_minus_baseline=subtract(a['charge_minus_propagated_matched_neutral'], b['charge_minus_propagated_matched_neutral']))
                for a, b in zip(drive['endpoints'], base['endpoints'])]
    initial_expanded = json.loads((ROOT/(PREFIX+'emit2_emission_result.json')).read_text())['results']['N55_initial']
    result = dict(kind='EXERCISED_EVOLVED_SIGNED_CURRENT_INSERTIONS_AND_EXACT_PRIMITIVE_IMAGES', physical_source=physical_stamp,
        physical_source_sha256=sha(source_path), cases=cases, pulse_contrasts=pulse_contrasts,
        exact_primitive_images=descriptors, completed_initial_N55_expanded_receiver_metrics=initial_expanded,
        model_units='hbar=lattice spacing=1. J/QJ/JQ are complex current/coordinate-current insertion amplitudes; raw receiver Gram weights are NOT probabilities or density contrasts.',
        exercised_scope='Actual6 already contracted coupled J and BEFORE-CUT QJ/JQ at clocks0/.2 for all9 cases, with matched propagated neutral and explicitly labelled original-law probes on the own-generator frozen controls. No new evolution or pulse.',
        optional_scope='emit1 SIGKILL and emit2 parent-directed catalogue stop preserved. Optional evolved expanded-image norm/Gram/spin/depletion catalogue is UNEXERCISED unless a completed metric record explicitly exists; descriptors are real exact TN images, not approval or guessed metrics.')
    Path(str(output)+'emission_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases), exercised_endpoints=18, signed_vertices_per_endpoint=4,
        exact_primitive_images=len(descriptors), completed_initial_expanded='N55_initial', no_new_evolution=True)), flush=True)


if __name__ == '__main__':
    main()

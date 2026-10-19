"""Actual NEW filled-reference composite sources and inclusive EM/sea/loop receiver."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eigh, expm
from rich_colour_sea_20261014_model import build, law


def mean(state, matrix):
    return float(np.vdot(state, matrix @ state).real)


def norm(matrix):
    return float(np.linalg.norm(matrix))


def complex_json(matrix):
    return dict(real=np.asarray(matrix).real.tolist(), imag=np.asarray(matrix).imag.tolist())


def spectrum(h, spin):
    ss, vectors = eigh(spin)
    labels = np.rint((np.sqrt(np.maximum(0, 1 + 4 * ss)) - 1)).astype(int)
    energies, states, spins = [], [], []
    for label in np.unique(labels):
        projector = vectors[:, labels == label]
        e, v = eigh(projector.conj().T @ h @ projector)
        energies.extend(e); states.append(projector @ v); spins.extend([label / 2] * len(e))
    order = np.argsort(energies)
    return np.array(energies)[order], np.column_stack(states)[:, order], np.array(spins)[order]


def at_angle(model, name, angle):
    p = model['parameters']; scalar = model['scalar']; op = model['operators']
    if name == 'neutral':
        h, j, k = law(op[name], scalar, p, angle)
        hm, jm, km = law(op[name], scalar, p, -angle)
        c = model['Cneutral']
        return (h + c @ hm @ c.conj().T) / 2, (j - c @ jm @ c.conj().T) / 2, (k + c @ km @ c.conj().T) / 2
    hb, jb, kb = law(op['baryon'], scalar, p, angle)
    ha, ja, ka = law(op['antibaryon'], scalar, p, -angle)
    c = model['Cbaryon']
    hb = (hb + c.conj().T @ ha @ c) / 2
    jb = (jb - c.conj().T @ ja @ c) / 2
    kb = (kb + c.conj().T @ ka @ c) / 2
    if name == 'baryon': return hb, jb, kb
    hm, jm, km = at_angle(model, 'baryon', -angle)
    return c @ hm @ c.conj().T, -c @ jm @ c.conj().T, c @ km @ c.conj().T


def dynamics(state, energies, vectors, times, observables):
    coefficients = vectors.conj().T @ state
    states = vectors @ (np.exp(-1j * np.outer(energies, times)) * coefficients[:, None])
    data = {key: [mean(states[:, i], op) for i in range(len(times))] for key, op in observables.items()}
    return states, data


def main():
    inp, out = Path(sys.argv[1]), Path(sys.argv[2]); p = json.loads(inp.read_text()); model = build(p)
    field = model['scalar']; nf = len(field['grid']); spectra = {}; observables = {}; stored = {}; result = {}
    for name, sector in model['sectors'].items():
        op = model['operators'][name]; h, current, contact = model['laws'][name]
        spin = np.kron(op['S2'], np.eye(nf)); e, v, s = spectrum(h, spin); spectra[name] = (e, v, s)
        pairs = op['positive'] - (3 * np.eye(sector['dimension']) if name == 'baryon' else 0)
        observables[name] = dict(positive_occupation=np.kron(op['positive'], np.eye(nf)), pair_number=np.kron(pairs, np.eye(nf)),
                                spin_squared=spin, loop_C=np.kron(op['C'], np.eye(nf)), loop_B=np.kron(op['B'], np.eye(nf)),
                                matter_octet=np.kron(op['octet'], np.eye(nf)), loop_boundary=np.kron(op['boundary'], np.eye(nf)),
                                phi=np.kron(np.eye(sector['dimension']), field['phi']), phi2=np.kron(np.eye(sector['dimension']), field['phi'] @ field['phi']),
                                scalar_boundary=np.kron(np.eye(sector['dimension']), field['boundary']), scalar_energy=np.kron(np.eye(sector['dimension']), field['H']),
                                current=current, contact=contact)
        stored[name + '_energies'] = e; stored[name + '_vectors'] = v; stored[name + '_spins'] = s
        stored[name + '_H'] = h; stored[name + '_J'] = current; stored[name + '_K'] = contact
        stored[name + '_P'] = sector['P'].toarray(); stored[name + '_masks'] = np.array(sector['masks'], np.uint32)
        for key, operator in observables[name].items(): stored[name + '_' + key] = operator
        result[name] = dict(energy=float(e[0]), charge={'neutral':0., 'baryon':1., 'antibaryon':-1.}[name],
                            number={'neutral':12, 'baryon':15, 'antibaryon':9}[name],
                            dimension=len(e), raw_dimension=sector['raw_dimension'], gauge_dimension=sector['dimension'],
                            gauss_error=sector['gauge_error'], ground_residual=norm(h @ v[:, 0] - e[0] * v[:, 0]),
                            raw_route_Gauss_commutators=op['gauge_route_errors'],
                            current_operator_norm=norm(current), contact_operator_norm=norm(contact),
                            spin_counts={str(x):int(sum(s == x)) for x in np.unique(s)},
                            ground={key:mean(v[:, 0], operator) for key, operator in observables[name].items()},
                            all_energies=e.tolist(), all_spins=s.tolist())
    e0, v0, s0 = spectra['neutral']; reference = v0[:, 0]; stored['reference'] = reference
    source_results = {}
    for name, sources in [('baryon', model['sources']), ('antibaryon', model['anti_sources'])]:
        e, v, spins = spectra[name]; source_results[name] = {}
        for spin, source in sources.items():
            state = source @ reference; amplitudes = v.conj().T @ state; weights = abs(amplitudes) ** 2
            gaps = e - e0[0]; bright = np.flatnonzero(weights > 1e-10)
            # Entire degenerate pole subspace residue, never a single arbitrary eigenvector label.
            lowest = float(gaps[bright[0]]) if len(bright) else None
            pole = abs(gaps - lowest) < 1e-8 if lowest is not None else np.zeros(len(gaps), bool)
            source_results[name][spin] = dict(norm_squared=norm(state) ** 2, sumrule=float(weights.sum()),
                lowest_bright_gap=lowest, residue=float(weights[pole].sum()), all_gaps=gaps.tolist(), all_weights=weights.tolist(),
                normalized_residue=float(weights[pole].sum() / weights.sum()) if weights.sum() else None,
                spin_residual=norm(observables[name]['spin_squared'] @ state - float(spin) * (float(spin) + 1) * state))
            stored[name + '_source_' + spin] = source; stored[name + '_source_state_' + spin] = state
    c = model['Cbaryon']; cn = model['Cneutral']; hb, jb, kb = model['laws']['baryon']; ha, ja, ka = model['laws']['antibaryon']
    cchecks = dict(unitarity=norm(c.conj().T @ c - np.eye(c.shape[1])), neutral_unitarity=norm(cn.conj().T @ cn - np.eye(cn.shape[1])),
                   H=norm(ha @ c - c @ hb), current=norm(ja @ c + c @ jb), contact=norm(ka @ c - c @ kb),
                   neutral_reference_C_overlap=float(abs(np.vdot(reference, cn @ reference)) ** 2), raw_path=model['raw_C_defects'])
    stored['Cneutral'] = cn; stored['Cbaryon'] = c
    responses = {}; events = {}; frequencies = p['frequencies']; eta = p['eta']; times = np.array(p['times'])
    for name in ['neutral', 'baryon', 'antibaryon']:
        e, v, spins = spectra[name]; op = observables[name]; h, current, contact = model['laws'][name]
        if name == 'neutral': state = reference
        else:
            source = model['sources']['0.5'] if name == 'baryon' else model['anti_sources']['0.5']
            initial = source @ reference
            coefficients = v.conj().T @ initial
            bright = np.flatnonzero(abs(coefficients) ** 2 > 1e-10)
            pole = abs(e - e[bright[0]]) < 1e-8
            state = v[:, pole] @ coefficients[pole]; state /= norm(state)
        amplitudes = v.conj().T @ current @ state; weights = abs(amplitudes) ** 2
        estate = mean(state, h); gaps = e - estate
        retarded = [sum(weights * (1 / (frequency - gaps + 1j * eta) - 1 / (frequency + gaps + 1j * eta))) for frequency in frequencies]
        responses[name] = dict(reference_energy=estate, all_gaps=gaps.tolist(), all_current_weights=weights.tolist(),
                               virtual_states=len(e), spectral_sum=float(weights.sum()), direct_sum=mean(state, current @ current),
                               contact=mean(state, contact), retarded=complex_json(retarded),
                               frequencies=frequencies, kernel_with_contact=complex_json(np.array(retarded) + mean(state, contact)))
        angle = p['pulse_angle'] * (-1 if name == 'antibaryon' else 1)
        hp, jp, kp = at_angle(model, name, angle)
        kicked = expm(-1j * p['pulse_duration'] * hp) @ state
        baseline, base_data = dynamics(state, e, v, times, op)
        event_states, event_data = dynamics(kicked, e, v, times, op)
        excess = {key:(np.array(event_data[key]) - np.array(base_data[key])).tolist() for key in op}
        outcomes = abs(v.conj().T @ kicked) ** 2
        events[name] = dict(preparation='Finite external EM Wilson angle pulse H(A), then same untouched H(0); no photons or COM claim.',
                            angle=angle, duration=p['pulse_duration'], times=times.tolist(), norm_error=abs(norm(kicked) - 1),
                            initial={key:mean(state, operator) for key, operator in op.items()}, baseline=base_data,
                            pulse=event_data, excess=excess, inclusive_all_eigenstate_probabilities=outcomes.tolist(),
                            energy_injection=mean(kicked, h) - estate, inclusive_sum=float(outcomes.sum()))
        stored[name + '_event_states'] = event_states; stored[name + '_event_baseline'] = baseline
        stored[name + '_receiver_state'] = state; stored[name + '_pulse_H'] = hp
    # Mechanism control recomputes BOTH reference and charged states under scalar decoupling.
    zero = dict(p, scalar_coupling=0.)
    control = {}
    for name in ['neutral', 'baryon', 'antibaryon']:
        h, _, _ = law(model['operators'][name], field, zero)
        control[name] = float(eigh(h, eigvals_only=True)[0])
    control['baryon_gap'] = control['baryon'] - control['neutral']; control['antibaryon_gap'] = control['antibaryon'] - control['neutral']
    # Exact all-pair scalar-only CAR control: X=sum c+dagger c-+h.c. has
    # extremal |X|=24-N (N>=12) or N (N<=12). No fitted mass/recentering.
    full_scalar_control = {}
    for number in range(25):
        extremal = min(number, 24 - number)
        energy = float(eigh(field['H'] + p['scalar_coupling'] * extremal * field['phi'], eigvals_only=True)[0])
        full_scalar_control[str(number)] = energy
    full_scalar_control = dict(all_number_energies=full_scalar_control,
        baryon_gap=full_scalar_control['15'] - full_scalar_control['12'],
        antibaryon_gap=full_scalar_control['9'] - full_scalar_control['12'],
        scope='Exact all-pair/all-N scalar-only CAR24 extremal collective route, no colour-loop coupling. Not the coupled-loop full-sector vacuum proof.')
    fieldonly = np.kron(np.eye(model['sectors']['baryon']['dimension']), expm(-1j * p['field_kick'] * field['P']))
    receiver = stored['baryon_receiver_state']; kicked = fieldonly @ receiver
    ee, vv, _ = spectra['baryon']; fs, fd = dynamics(kicked, ee, vv, times, observables['baryon'])
    field_base, fbase = dynamics(receiver, ee, vv, times, observables['baryon'])
    field_event = dict(preparation='Unitary scalar-field-only exp(-i eta Pphi) on SAME actual baryon reference; matter untouched at preparation.',
                       norm_error=abs(norm(kicked) - 1), times=times.tolist(),
                       initial_body_changes={key:mean(kicked, observables['baryon'][key]) - mean(receiver, observables['baryon'][key]) for key in ['pair_number','spin_squared','matter_octet','loop_C']},
                       excess={key:(np.array(fd[key]) - np.array(fbase[key])).tolist() for key in fd})
    stored['fieldonly_event_states'] = fs
    payload = dict(schema=1, parameters=p, frame={key:value for key, value in model['frame'].items() if key not in ['scalar','vectors','spin']},
                   sectors=result, sources=source_results, C=cchecks, current_response=responses, EM_events=events,
                   scalar_field_only_event=field_event, decoupled_control=control,
                   full_pair_scalar_only_control=full_scalar_control,
                   positive_excitation_gaps={name:result[name]['energy'] - result['neutral']['energy'] for name in ['baryon','antibaryon']},
                   scalar_grid=field['grid'].tolist(),
                   scope='One CAR24; literal negative mirror sea base N12; pair_cut counts extra holes for baryon/neutral and C-related extra particles for anti. Selected variational sectors, NOT full all-number stable vacuum. One quantum PW19 loop and scalar DVR. Primitive quaternion/mirror/multiplicity/CAR/flavour/charge/couplings remain inputs. No spatial COM, continuum QCD or photon claim.')
    out.write_text(json.dumps(payload, indent=2) + '\n'); np.savez_compressed(out.with_suffix('.npz'), **stored)
    print(json.dumps(dict(dimensions={name:item['dimension'] for name,item in result.items()}, gaps=payload['positive_excitation_gaps'],
                          scalar_off=control, sources={name:{spin:{key:value for key,value in data.items() if key not in ['all_gaps','all_weights']} for spin,data in values.items()} for name,values in source_results.items()},
                          C=cchecks, ground={name:item['ground'] for name,item in result.items()}), indent=2))


if __name__ == '__main__':
    main()

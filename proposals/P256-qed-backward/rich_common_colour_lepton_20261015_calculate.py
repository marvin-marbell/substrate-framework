"""Actual common CAR32 source poles, external EM and quantum-field receivers.
No producer CLI is imported or replayed. All eigenstates of each exact invariant
sector are retained, including dark ones, in every response and finite event.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eigh, expm
from rich_common_colour_lepton_20261015_model import build, law, occupation_closure


def mean(state, operator):
    return float(np.vdot(state, operator @ state).real)


def norm(value):
    return float(np.linalg.norm(value))


def complex_json(value):
    return dict(real=np.asarray(value).real.tolist(), imag=np.asarray(value).imag.tolist())


def spectrum(h, spin):
    ss, vectors = eigh(spin)
    labels = np.rint(np.sqrt(np.maximum(0, 1 + 4 * ss)) - 1).astype(int)
    energies, states, spins = [], [], []
    for label in np.unique(labels):
        projector = vectors[:, labels == label]
        e, v = eigh(projector.conj().T @ h @ projector)
        energies.extend(e); states.append(projector @ v); spins.extend([label / 2] * len(e))
    order = np.argsort(energies)
    return np.array(energies)[order], np.column_stack(states)[:, order], np.array(spins)[order]


def dynamics(state, e, v, times, observables):
    states = v @ (np.exp(-1j * np.outer(e, times)) * (v.conj().T @ state)[:, None])
    data = {key:[mean(states[:, i], op) for i in range(len(times))] for key, op in observables.items()}
    return states, data


def field_density(state, matter_dimension, field_dimension):
    tensor = state.reshape(matter_dimension, field_dimension)
    return tensor.T @ tensor.conj()


def main():
    inp, out = Path(sys.argv[1]), Path(sys.argv[2]); p = json.loads(inp.read_text()); model = build(p)
    scalar = model['scalar']; nf = len(scalar['grid']); laws = {}; spectra = {}; observables = {}; stored = {}; sectors = {}
    for name, sector in model['sectors'].items():
        op = model['operators'][name]; h, current, contact = law(op, scalar, p); laws[name] = (h, current, contact)
        spin = np.kron(op['S2'], np.eye(nf)); e, v, s = spectrum(h, spin); spectra[name] = (e, v, s)
        obs = {key:np.kron(op[key], np.eye(nf)) for key in ['positive', 'S2', 'N', 'Q', 'matter_octet', 'C', 'B', 'boundary']}
        obs.update(phi=np.kron(np.eye(sector['dimension']), scalar['phi']),
            phi2=np.kron(np.eye(sector['dimension']), scalar['phi'] @ scalar['phi']),
            scalar_energy=np.kron(np.eye(sector['dimension']), scalar['H']),
            scalar_boundary=np.kron(np.eye(sector['dimension']), scalar['boundary']), current=current, contact=contact)
        observables[name] = obs
        for key, value in dict(H=h, J=current, K=contact, energies=e, vectors=v, spins=s, masks=np.array(sector['masks'], np.uint64)).items():
            stored[name + '_' + key] = value
        for key, value in obs.items():
            stored[name + '_observable_' + key] = value
        for key in ['data', 'indices', 'indptr']:
            stored[name + '_P_' + key] = getattr(sector['P'], key)
        stored[name + '_P_shape'] = sector['P'].shape
        rho = field_density(v[:, 0], sector['dimension'], nf)
        sectors[name] = dict(dimension=len(e), gauge_dimension=sector['dimension'], raw_dimension=sector['raw_dimension'],
            ground_energy=float(e[0]), ground={key:mean(v[:, 0], value) for key, value in obs.items()},
            gauss_error=sector['gauge_error'], route_Gauss_commutators=op['route_errors'],
            closure=occupation_closure(sector), eigen_residual=norm(h @ v - v * e[None, :]),
            spin_commutator=norm(h @ spin - spin @ h), current_spin_commutator=norm(current @ spin - spin @ current),
            current_norm=norm(current), contact_norm=norm(contact),
            scalar_reduced_density=complex_json(rho), scalar_purity=float(np.trace(rho @ rho).real),
            scalar_variance=mean(v[:, 0], obs['phi2']) - mean(v[:, 0], obs['phi'])**2,
            spin_counts={str(x):int(sum(s == x)) for x in np.unique(s)}, all_energies=e.tolist(), all_spins=s.tolist())
    e0, v0, _ = spectra['neutral']; reference = v0[:, 0]; stored['reference'] = reference
    source_results = {}; receivers = {'neutral':reference}
    for name, sources in model['sources'].items():
        e, v, spins = spectra[name]; source_results[name] = {}
        for spin, source in sources.items():
            state = source @ reference; amplitudes = v.conj().T @ state; weights = abs(amplitudes)**2
            bright = np.flatnonzero(weights > 1e-11); gaps = e - e0[0]
            if not len(bright):
                raise ValueError('Actual source has no bright state: ' + name + '/' + spin)
            lowest = float(gaps[bright[0]]); pole = abs(gaps - lowest) < 1e-8
            projected = v[:, pole] @ amplitudes[pole]; receiver = projected / norm(projected)
            source_results[name][spin] = dict(raw_strength=norm(state)**2, spectral_sum=float(weights.sum()),
                lowest_bright_gap=lowest, raw_pole_residue=float(weights[pole].sum()),
                conditional_pole_fraction=float(weights[pole].sum() / weights.sum()),
                all_gaps=gaps.tolist(), all_raw_weights=weights.tolist(),
                source_spin_residual=norm(observables[name]['S2'] @ state - float(spin) * (float(spin) + 1) * state),
                source_charge=mean(state, observables[name]['Q']) / norm(state)**2,
                source_number=mean(state, observables[name]['N']) / norm(state)**2,
                pole_spin=mean(receiver, observables[name]['S2']),
                pole_observables={key:mean(receiver, op) for key, op in observables[name].items()})
            stored[name + '_source_' + spin] = source; stored[name + '_source_state_' + spin] = state
            stored[name + '_pole_' + spin] = receiver
            if spin == '0.5':
                receivers[name] = receiver
    conjugation = {}
    cn = model['C']['neutral']; hn, jn, kn = laws['neutral']
    conjugation['neutral'] = dict(unitarity=norm(cn.conj().T @ cn - np.eye(len(cn))),
        H=norm(hn @ cn - cn @ hn), J=norm(jn @ cn + cn @ jn), K=norm(kn @ cn - cn @ kn),
        reference_overlap=float(abs(np.vdot(reference, cn @ reference))**2))
    stored['Cneutral'] = cn
    for name, anti in [('baryon', 'antibaryon'), ('lepton', 'antilepton')]:
        c = model['C'][name]; h, j, k = laws[name]; ha, ja, ka = laws[anti]
        hp, jp, kp = law(model['operators'][name], scalar, p, p['pulse_angle'])
        ham, jam, kam = law(model['operators'][anti], scalar, p, -p['pulse_angle'])
        conjugation[name] = dict(unitarity=norm(c.conj().T @ c - np.eye(c.shape[1])), H=norm(ha @ c - c @ h),
            J=norm(ja @ c + c @ j), K=norm(ka @ c - c @ k), finite_A_H=norm(ham @ c - c @ hp),
            finite_A_J=norm(jam @ c + c @ jp), finite_A_K=norm(kam @ c - c @ kp),
            Q=norm(observables[anti]['Q'] @ c + c @ observables[name]['Q']),
            N_complement=norm(observables[anti]['N'] @ c + c @ observables[name]['N'] - 32 * c))
        stored['C' + name] = c
    responses = {}; events = {}; field_events = {}; times = np.array(p['times']); frequencies = np.array(p['frequencies'])
    kernels = {}
    for name, state in receivers.items():
        e, v, spins = spectra[name]; h, current, contact = laws[name]; obs = observables[name]
        estate = mean(state, h); gaps = e - estate; weights = abs(v.conj().T @ current @ state)**2
        chi = np.array([sum(weights * (1 / (w - gaps + 1j * p['eta']) - 1 / (w + gaps + 1j * p['eta']))) for w in frequencies])
        kernels[name] = chi + mean(state, contact)
        responses[name] = dict(virtual_states=len(e), all_gaps=gaps.tolist(), all_current_weights=weights.tolist(),
            spectral_sum=float(weights.sum()), direct_sum=mean(state, current @ current),
            contact=mean(state, contact), retarded=complex_json(chi), kernel_with_contact=complex_json(kernels[name]),
            frequencies=frequencies.tolist(), eta=p['eta'], normalization='Normalized stationary lowest bright halfspin pole (or exact neutral reference); every eigenstate included; both retarded orders and physical contact retained.')
        angle = p['pulse_angle'] * (-1 if name.startswith('anti') else 1)
        hp, _, _ = law(model['operators'][name], scalar, p, angle)
        pulse = expm(-1j * p['pulse_duration'] * hp) @ state
        baseline, base = dynamics(state, e, v, times, obs); event, data = dynamics(pulse, e, v, times, obs)
        probabilities = abs(v.conj().T @ pulse)**2
        events[name] = dict(angle=angle, duration=p['pulse_duration'], times=times.tolist(),
            preparation='External physical EM Wilson flux angle in SAME selected Q route, finite H(A) evolution followed by H(0). Not a dynamical photon or spatial COM experiment.',
            baseline=base, pulse=data, excess={key:(np.array(data[key]) - base[key]).tolist() for key in data},
            inclusive_eigenstate_probabilities=probabilities.tolist(), inclusive_sum=float(probabilities.sum()),
            energy_injection=mean(pulse, h) - estate, norm_error=abs(norm(pulse) - 1))
        stored[name + '_EM_event'] = event; stored[name + '_baseline'] = baseline; stored[name + '_receiver'] = state
        field_unitary = np.kron(np.eye(model['sectors'][name]['dimension']), expm(-1j * p['field_kick'] * scalar['P']))
        field_state = field_unitary @ state
        field_states, field_data = dynamics(field_state, e, v, times, obs)
        field_events[name] = dict(preparation='Scalar-field-only unitary I_matter,loop tensor exp(-i kick Pphi), SAME normalized pole and untouched-state baseline.',
            times=times.tolist(), norm_error=abs(norm(field_state) - 1),
            initial_matter_loop_changes={key:mean(field_state, obs[key]) - mean(state, obs[key]) for key in ['positive', 'N', 'Q', 'S2', 'matter_octet', 'C', 'B', 'boundary']},
            baseline=base, pulse=field_data, excess={key:(np.array(field_data[key]) - base[key]).tolist() for key in field_data},
            energy_injection=mean(field_state, h) - estate)
        stored[name + '_field_event'] = field_states
    source_excess = {name:dict(kernel_minus_neutral=complex_json(kernel - kernels['neutral']),
        EM_energy_injection_minus_neutral=events[name]['energy_injection'] - events['neutral']['energy_injection'])
        for name, kernel in kernels.items() if name != 'neutral'}
    decoupled = {}
    for name in sectors:
        h, _, _ = law(model['operators'][name], scalar, dict(p, scalar_coupling=0.))
        decoupled[name] = float(eigh(h, eigvals_only=True)[0])
    decoupled['gaps'] = {name:decoupled[name] - decoupled['neutral'] for name in sectors if name != 'neutral'}
    # Filled opposite rotated sea reverses the ENTIRE scalar density coupling.
    opposite, _, _ = law(model['operators']['neutral'], scalar, dict(p, scalar_coupling=-p['scalar_coupling']))
    reflection = np.kron(np.eye(model['sectors']['neutral']['dimension']), np.eye(nf)[::-1])
    branches = dict(selected='All16 rotated-negative occupied, rotated-positive empty. Its original-mirror expansion includes every pair number through16, coherently; no original pair truncation.',
        opposite='All16 rotated-positive occupied. Separate exactly conserved population sector, with reversed scalar tilt; no tunneling between branches in this scalar-only law.',
        energy=float(eigh(opposite, eigvals_only=True)[0]), parity_similarity_defect=norm(opposite - reflection @ hn @ reflection),
        global_claim='No minimization over all rotated populations or all N. The two branch degeneracy is not a derived unique phase, and invariant scalar closure is not interband Dirac pair feedback.')
    payload = dict(schema=1, parameters=p, root={key:value for key,value in model['root'].items() if key not in ['cliff', 'Phi', 'Z']},
        sectors=sectors, sources=source_results, C=conjugation, current_response=responses, EM_events=events,
        scalar_field_only_events=field_events, source_excess_response=source_excess, scalar_off_control=decoupled,
        occupied_branches=branches, gaps={name:sectors[name]['ground_energy'] - sectors['neutral']['ground_energy'] for name in sectors if name != 'neutral'},
        scalar_grid=scalar['grid'].tolist(),
        physical_scope='One actual occupied CAR32 reference, same common quantum scalar, PW19 non-Abelian loop, Q law/current/contact and C. Lepton N17,Q-1 versus odd colour-white uud N19,Q+1; their partners N15,Q+1 and N13,Q-1 respectively. NO identification of proton with positron. Source-connected closure exact only for this chosen tau_x scalar-density law. Ci=Ymirror tensor Fi and fixed3+1 algebra are kinematics, not an implemented spatial vector kinetic term; no claim of full spatial Noether spin, photons, COM, QCD, rank splitting dynamics, charge derivation or global vacuum.')
    out.write_text(json.dumps(payload, indent=2) + '\n'); np.savez_compressed(out.with_suffix('.npz'), **stored)
    print(json.dumps(dict(dimensions={name:value['dimension'] for name,value in sectors.items()}, gaps=payload['gaps'],
        sources={name:{spin:{k:v for k,v in source.items() if k in ['raw_strength','lowest_bright_gap','raw_pole_residue','conditional_pole_fraction','source_spin_residual','source_charge','source_number']} for spin,source in sources.items()} for name,sources in source_results.items()},
        C=conjugation, scalar_off=decoupled, closure={name:value['closure']['missing_targets'] for name,value in sectors.items()}), indent=2))


if __name__ == '__main__':
    main()

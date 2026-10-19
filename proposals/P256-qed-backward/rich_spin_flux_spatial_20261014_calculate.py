"""New actual same-H/state field preparation, propagation, scattering and G12."""
import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, expm_multiply
from rich_spin_flux_spatial_20261014_model import (LAW, Parameters, Sector, mean, centered_sources, gram_frame, charged_source)

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_spin_flux_spatial_20261014_actual2'


def encode(value):
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def state_observations(sector, state):
    probability = np.abs(state)**2
    return {'norm': float(probability.sum()), 'flux': [mean(state, f) for f in sector.F], 'cosWilson': [mean(state, c) for c in sector.C], 'active_current': [mean(state, j) for j in sector.currents['active']], 'supplier_current': [mean(state, j) for j in sector.currents['supplier']], 'total_current': [mean(state, j) for j in sector.total_currents], 'electric_energy': float(probability@sector.electric_energy), 'orientor_L': mean(state, sector.Ln), 'orientor_torque': mean(state, sector.neutral_torque), 'orientor_torque_norm': float(np.linalg.norm(sector.neutral_torque@state)), 'body_x': mean(state, sector.body_x), 'body_velocity': mean(state, sector.body_velocity), 'bound_occupancy': float(probability@(np.max(np.abs(sector.electric), axis=1) == sector.p.bound))}


def field_run(p, spin, times):
    sector = Sector(p, up=int(spin == 1), down=int(spin == -1))
    energy, ground, lowest = sector.ground()
    incoming_ops = [sector.F[0], sector.loop_electric[0]]
    outgoing_ops = [sector.F[2], sector.loop_electric[2]]
    incoming_raw = np.column_stack([op@ground for op in incoming_ops])
    outgoing_raw = np.column_stack([op@ground for op in outgoing_ops])
    incoming = centered_sources(ground, incoming_ops)
    outgoing = centered_sources(ground, outgoing_ops)
    qin, gin, eig_in = gram_frame(incoming)
    qout, gout, eig_out = gram_frame(outgoing)
    waves = sector.evolve(qin, times, energy)
    amplitudes = np.einsum('ar,tai->tri', qout.conj(), waves)
    probability_out = np.sum(np.abs(amplitudes)**2, axis=1)
    ground_probability = np.abs(np.einsum('a,tai->ti', ground.conj(), waves))**2
    remainder = 1-probability_out-ground_probability
    # Genuine finite quantum impulse; ground/admixture retained rather than renormalized away.
    eta = .7
    impulse = expm_multiply(-1j*eta*sector.F[0], ground, traceA=0.)
    raw_waves = sector.evolve(impulse, times, energy)
    raw_out = np.sum(np.abs(raw_waves@qout.conj())**2, axis=1)
    raw_ground = np.abs(raw_waves@ground.conj())**2
    raw_remainder = 1-raw_ground-raw_out
    electric_face = np.column_stack([np.sum(sector.electric[:, np.flatnonzero(face)]**2, axis=1)/(2*p.inertia) for face in sector.faces])
    density = np.abs(raw_waves)**2
    field_energy = density@electric_face
    baseline_field = np.abs(ground)**2@electric_face
    # Strictly local right-edge detector: edge5=(3,4), not a global channel projector.
    local_receiver_diag = (sector.electric[:, 5]**2 > 0).astype(float)
    receiver = density@local_receiver_diag
    baseline_receiver = float(np.abs(ground)**2@local_receiver_diag)
    charge_remote_diag = sector.active_density[:, [3, 4]].sum(axis=1)
    charge_remote = density@charge_remote_diag
    body_x = density@(sector.active_density@sector.positions[:, 0])
    velocity = np.array([mean(w, sector.body_velocity) for w in raw_waves])
    flux = np.array([[mean(w, f) for f in sector.F] for w in raw_waves])
    electric_energy = density@sector.electric_energy
    total_energy = np.array([mean(w, sector.H) for w in raw_waves])
    operators = [sector.F[0], sector.loop_electric[0], sector.F[2], sector.loop_electric[2], sector.body_velocity, sector.Ln]
    chi = sector.retarded(ground, energy, operators, .43+.11j)
    raw_left = incoming_raw-ground[:, None]*(ground.conj()@incoming_raw)[None, :]
    evolved_left = sector.evolve(raw_left, times, energy)
    retarded_time = 2*np.imag(np.einsum('ar,tai->tri', outgoing.conj(), evolved_left))
    result = {'law': LAW, 'parameters': asdict(p), 'spin_z': spin/2, 'dimension': sector.dimension, 'countercharge_vertex0': -int(sector.background[0]), 'energy': energy, 'lowest_energies': lowest, 'ground': state_observations(sector, ground), 'constraints': sector.constraints(), 'eigen_residual': float(np.linalg.norm(sector.H@ground-energy*ground)), 'sources': ['actual sinWilson left face0', 'actual loop-electric left face0'], 'outchannels': ['actual sinWilson remote face2', 'actual loop-electric remote face2'], 'raw_incoming_Gram': incoming_raw.conj().T@incoming_raw, 'incoming_ground_means': ground.conj()@incoming_raw, 'connected_incoming_Gram': gin, 'incoming_Gram_eigenvalues': eig_in, 'incoming_rank': qin.shape[1], 'raw_outgoing_Gram': outgoing_raw.conj().T@outgoing_raw, 'outgoing_ground_means': ground.conj()@outgoing_raw, 'connected_outgoing_Gram': gout, 'outgoing_Gram_eigenvalues': eig_out, 'outgoing_rank': qout.shape[1], 'times': times, 'normalized_outchannel_probability': probability_out, 'normalized_ground_probability': ground_probability, 'normalized_remainder_probability': remainder, 'outchannel_probability_max': np.max(probability_out, axis=0), 'outchannel_probability_initial': probability_out[0], 'source_norm_drift': float(np.max(np.abs(np.sum(np.abs(waves)**2, axis=1)-1))), 'impulse': {'operator': 'exp(-i eta sinWilson0)', 'eta': eta, 'norm': float(np.vdot(impulse, impulse).real), 'ground_probability': raw_ground, 'out_probability': raw_out, 'remainder_probability': raw_remainder, 'partition_norm_error': float(np.max(np.abs(raw_ground+raw_out+raw_remainder-1))), 'local_remote_E_nonzero_probability': receiver, 'local_remote_E_nonzero_baseline': baseline_receiver, 'local_remote_E_probability_max_change': float(np.max(np.abs(receiver-baseline_receiver))), 'face_electric_energy': field_energy, 'face_electric_baseline': baseline_field, 'remote_face_electric_max_change': float(np.max(np.abs(field_energy[:, 2]-baseline_field[2]))), 'electric_energy': electric_energy, 'electric_energy_exchange_range': float(np.ptp(electric_energy)), 'active_remote_probability': charge_remote, 'active_remote_probability_baseline': float(np.abs(ground)**2@charge_remote_diag), 'body_x': body_x, 'body_x_max_displacement': float(np.max(np.abs(body_x-mean(ground, sector.body_x)))), 'body_velocity': velocity, 'body_velocity_max': float(np.max(np.abs(velocity))), 'flux': flux, 'remote_flux_max_change': float(np.max(np.abs(flux[:, 2]-mean(ground, sector.F[2])))), 'total_energy_drift': float(np.ptp(total_energy)), 'norm_drift': float(np.max(np.abs(density.sum(axis=1)-1)))}, 'mixed_response_z': [.43, .11], 'mixed_response_ops': ['F0', 'Eloop0', 'F2', 'Eloop2', 'bodyvelocity', 'Ln'], 'mixed_retarded_response': chi, 'remote_time_retarded': retarded_time, 'remote_time_retarded_max': np.max(np.abs(retarded_time), axis=0), 'cut_edges': sector.cut_edges, 'outprojector_semantics': 'ground projector plus orthogonal connected remote-source Gram frame plus remainder; globally specified channel measurement, not identical to local electric receiver POVM', 'virtual_semantics': 'full actual finite Hilbert resolvents and exponential, no low-eigenstate virtual projection'}
    arrays = {'ground': ground, 'incoming_frame': qin, 'outgoing_frame': qout, 'impulse_state': impulse, 'incoming_waves': waves, 'impulse_waves': raw_waves}
    return result, arrays, sector


def charged_g12(p, reference, ground, energy):
    removal_sector = Sector(p, 0, 0)
    add_same = Sector(p, 2, 0)
    add_opposite = Sector(p, 1, 1)
    n = reference.vertices
    removal = np.column_stack([charged_source(reference, removal_sector, site, 0, False)@ground for site in range(n)])
    additions = [np.column_stack([charged_source(reference, target, site, spin, True)@ground for site in range(n)]) for target, spin in [(add_same, 0), (add_opposite, 1)]]
    z = .61+.13j
    eye0 = sparse.eye(removal_sector.dimension, format='csc')
    backward = splu((z*eye0+removal_sector.H-energy*eye0).tocsc()).solve(removal)
    grams = [a.conj().T@a for a in additions]
    greens = []
    for target, source in zip([add_same, add_opposite], additions):
        eye = sparse.eye(target.dimension, format='csc')
        forward = splu((z*eye-target.H+energy*eye).tocsc()).solve(source)
        value = source.conj().T@forward
        if target is add_same:
            value += (removal.conj().T@backward).T
        greens.append(value)
    return {'definition': 'G_same(z)=A†(z-H_N2+E_N1)^-1 A+[R†(z+H_N0-E_N1)^-1 R]^T; opposite spin addition only because occupied reference has no down charge', 'z': z, 'sites': list(range(n)), 'reference': {'law': LAW, 'up': 1, 'down': 0, 'supplier': 1, 'countercharge0': -2}, 'sector_dimensions': [removal_sector.dimension, add_same.dimension, add_opposite.dimension], 'countercharges0': [-1, -3, -3], 'same_spin_G12': greens[0], 'opposite_spin_G_add': greens[1], 'removal_Gram': removal.conj().T@removal, 'addition_Grams': grams, 'physical_source_closure_same': grams[0]+(removal.conj().T@removal).T, 'closure_not_assumed_identity': 'projected bounded tree-Wilson charge source has field-domain loss; canonical CAR acts before finite field projection', 'full_virtual_states': True, 'constraints': [s.constraints() for s in [removal_sector, add_same, add_opposite]], 'no_positron_claim': True}, {'removal_sources': removal, 'same_addition_sources': additions[0], 'opposite_addition_sources': additions[1]}


def main():
    request = json.loads((ROOT/'rich_spin_flux_spatial_20261014_input.json').read_text())
    p = Parameters(**request['parameters'])
    times = np.linspace(0., request['time_stop'], request['time_count'])
    result = {'law': LAW, 'input': request, 'cases': {}}
    stored = {}
    reference = None
    for name, params, spin in [('joined_up', p, 1), ('joined_down', p, -1), ('supplier_frozen', replace(p, supplier_hop=0.), 1), ('true_disconnect', replace(p, cut=True), 1), ('orientor_torque_off', replace(p, torque=0.), 1)]:
        data, arrays, sector = field_run(params, spin, times)
        result['cases'][name] = data
        stored.update({name+'_'+key: value for key, value in arrays.items()})
        if name == 'joined_up':
            reference = sector
        print(name, 'dimension', sector.dimension, 'flux', data['ground']['flux'], 'ranks', data['incoming_rank'], data['outgoing_rank'], 'remote_local_receiver_change', data['impulse']['local_remote_E_probability_max_change'], 'remoteelectric', data['impulse']['remote_face_electric_max_change'], 'bodydisplacement', data['impulse']['body_x_max_displacement'], flush=True)
    if reference is None:
        raise RuntimeError('Missing actual reference')
    energy = result['cases']['joined_up']['energy']
    gdata, gsources = charged_g12(p, reference, stored['joined_up_ground'], energy)
    result['charged_G12'] = gdata
    stored.update(gsources)
    print('G12 sector dimensions', gdata['sector_dimensions'], flush=True)
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(encode(result), indent=2))
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'), **stored)


if __name__ == '__main__':
    main()

"""New common-H quantum field event; no response kernel substituted for an event."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from rich_quantum_collision_20261014_model import LAW, Sector, Parameters, maximum_entry
from rich_spin_flux_spatial_20261014_model import centered_sources, gram_frame, mean

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_quantum_collision_20261014_actual1'


def encode(value):
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def expectations(waves, operator):
    # time x basis x preparation, retaining full states without spectral truncation.
    flat = waves.transpose(1, 0, 2).reshape(waves.shape[1], -1)
    return np.sum(flat.conj()*(operator@flat), axis=0).real.reshape(waves.shape[0], waves.shape[2])


def event(s, seeds, times, outgoing, reference_energy):
    waves = s.evolve(seeds, times, reference_energy)
    probability = np.abs(waves)**2
    detector = (np.sum(s.electric[:, [10, 11]]**2, axis=1) > 0).astype(float)
    up = .5+s.spin[2].diagonal().real
    weights = np.column_stack([detector*up, detector*(1-up), (1-detector)*up, (1-detector)*(1-up)])
    joint = np.einsum('tai,ao->tio', probability, weights)
    amplitudes = np.einsum('ar,tai->tri', outgoing.conj(), waves)
    projected = np.einsum('ar,tri->tai', outgoing, amplitudes)
    p_out = np.sum(np.abs(amplitudes)**2, axis=1)
    spin_projected = expectations(projected, s.spin[2])
    source_povm = np.stack([p_out/2+spin_projected, p_out/2-spin_projected, 1-p_out], axis=2)
    far_energy = np.einsum('tai,a->ti', probability, np.sum(s.electric[:, [10, 11]]**2, axis=1)/(2*s.p.inertia))
    spin = np.stack([expectations(waves, op) for op in s.spin], axis=2)
    body_spin = np.stack([expectations(waves, sum(s.body_spin[a][2] for a in range(6, 9))), expectations(waves, sum(s.body_spin[a][2] for a in range(3)))], axis=2)
    active_current = np.stack([expectations(waves, op) for op in s.currents['active']], axis=2)
    supplier_current = np.stack([expectations(waves, op) for op in s.currents['supplier']], axis=2)
    spin_current_z = np.stack([expectations(waves, ops[2]) for ops in s.spin_currents], axis=2)
    energy = expectations(waves, s.H)
    density = np.einsum('tai,av->tiv', probability, s.active_density)
    body_x = expectations(waves, s.body_x)
    velocity = expectations(waves, s.body_velocity)
    field_flux = np.stack([expectations(waves, op) for op in s.F], axis=2)
    baseline = 0
    pulse_columns = [3, 4]
    packet_columns = [1, 2]
    changes = {}
    for label, columns in [('postselected_circular', packet_columns), ('local_unitary', pulse_columns)]:
        changes[label] = {
            'initial_far_detector_change': [float(joint[0, c, :2].sum()-joint[0, baseline, :2].sum()) for c in columns],
            'initial_far_energy_change': [float(far_energy[0, c]-far_energy[0, baseline]) for c in columns],
            'max_far_detector_change': [float(np.max(np.abs(joint[:, c, :2].sum(axis=1)-joint[:, baseline, :2].sum(axis=1)))) for c in columns],
            'max_far_energy_change': [float(np.max(np.abs(far_energy[:, c]-far_energy[:, baseline]))) for c in columns],
            'max_spin_vector_change': [np.max(np.abs(spin[:, c]-spin[:, baseline]), axis=0) for c in columns],
            'max_charge_motion_change': [float(np.max(np.abs(body_x[:, c]-body_x[:, baseline]))) for c in columns],
            'max_actual_spin_current_change': [float(np.max(np.abs(spin_current_z[:, c]-spin_current_z[:, baseline]))) for c in columns],
            'max_helicity_spin_separation': np.max(np.abs(spin[:, columns[0]]-spin[:, columns[1]]), axis=0),
            'max_helicity_far_detector_separation': float(np.max(np.abs(joint[:, columns[0], :2].sum(axis=1)-joint[:, columns[1], :2].sum(axis=1)))),
            'max_out_source_probability': [float(p_out[:, c].max()) for c in columns],
            'energy_injection_above_same_state_baseline': energy[0, columns]-energy[0, baseline]
        }
    result = {
        'dimension': s.dimension, 'parameters': asdict(s.p), 'constraints': s.constraints(),
        'spin_H_commutator_max': [maximum_entry(s.H@op-op@s.H) for op in s.spin],
        'probability_norm_max_error': float(np.max(np.abs(probability.sum(axis=1)-1))),
        'joint_probability_sum_max_error': float(np.max(np.abs(joint.sum(axis=2)-1))),
        'source_POVM_sum_max_error': float(np.max(np.abs(source_povm.sum(axis=2)-1))),
        'minimum_joint_probability': float(joint.min()), 'minimum_source_POVM_probability': float(source_povm.min()),
        'energy_conservation_max_error': float(np.max(np.abs(energy-energy[0]))),
        'changes': changes, 'times': times, 'joint_far_field_spin_PVM': joint,
        'out_source_then_fixed_Sz_POVM': source_povm, 'out_source_amplitudes': amplitudes,
        'far_electric_energy': far_energy, 'primitive_spin_vector': spin, 'left_and_right_body_spin_z': body_spin,
        'active_density': density, 'body_x': body_x, 'body_velocity': velocity,
        'active_current': active_current, 'supplier_current': supplier_current,
        'actual_spin_current_z': spin_current_z, 'field_flux': field_flux,
        'electric_energy': np.einsum('tai,a->ti', probability, s.electric_energy), 'total_energy': energy,
        'contacts_initial': {sp: [[mean(seeds[:, c], op) for op in s.contacts[sp]] for c in range(seeds.shape[1])] for sp in s.contacts}
    }
    return result, waves[-1]


def main():
    request = json.loads((ROOT/'rich_quantum_collision_20261014_input.json').read_text())
    p = Parameters(**request['parameters'])
    times = np.linspace(0., request['time_stop'], request['time_count'])
    s = Sector(p)
    energy, g, low, ground_spin_values = s.polarized_ground()
    incoming = centered_sources(g, s.F[:2])
    outgoing_raw = centered_sources(g, s.F[2:])
    incoming_frame, incoming_gram, incoming_values = gram_frame(incoming)
    outgoing, outgoing_gram, outgoing_values = gram_frame(outgoing_raw)
    packet_raw = incoming@np.array([[1, 1], [1j, -1j]])/np.sqrt(2)
    packet_gram = packet_raw.conj().T@packet_raw
    packets = packet_raw/np.sqrt(packet_gram.diagonal().real)[None, :]
    # Hermitian local field quadratures: circular source selection is exact only
    # for the postselected F combinations above. These unitary controls are
    # coherent signed-quadrature pulses, NOT asserted helicity eigenphotons.
    quadratures = centered_sources(g, [s.F[0], s.loop_electric[1]])
    qgram = quadratures.conj().T@quadratures
    scale = np.sqrt(qgram[0, 0].real/qgram[1, 1].real)
    generators = [s.F[0]+sign*scale*s.loop_electric[1] for sign in [1, -1]]
    pulses = np.column_stack([expm_multiply(-1j*request['pulse_eta']*q, g, traceA=-1j*request['pulse_eta']*q.diagonal().sum()) for q in generators])
    seeds = np.column_stack([g, packets, pulses])
    remote = sparse.diags((np.sum(s.electric[:, [10, 11]]**2, axis=1) > 0).astype(float))
    result = {'law': LAW, 'request': request, 'preparation_labels': ['ground_baseline', 'circular_F_plus_postselected', 'circular_F_minus_postselected', 'unitary_signed_quadrature_plus', 'unitary_signed_quadrature_minus'],
              'energy_reference': energy, 'lowest_energies': low, 'ground_subspace_Sz': ground_spin_values,
              'ground_residual': float(np.linalg.norm(s.H@g-energy*g)), 'ground_spin': [mean(g, op) for op in s.spin],
              'ground_boundary_occupancy': float(np.abs(g)**2@(np.max(np.abs(s.electric), axis=1)==p.bound)),
              'incoming_actual_field_Gram': incoming_gram, 'incoming_Gram_eigenvalues': incoming_values, 'incoming_rank': incoming_frame.shape[1],
              'outgoing_actual_field_Gram': outgoing_gram, 'outgoing_Gram_eigenvalues': outgoing_values, 'outgoing_rank': outgoing.shape[1],
              'postselected_raw_Gram': packet_gram, 'postselected_normalized_overlap': packets.conj().T@packets,
              'ground_overlap_all_preparations': np.abs(g.conj()@seeds)**2,
              'unitary_quadrature_Gram': qgram, 'unitary_quadrature_scale': scale,
              'unitary_remote_detector_commutator_max': [maximum_entry(q@remote-remote@q) for q in generators],
              'primitive_spin_SU2_error': maximum_entry(s.spin[0]@s.spin[1]-s.spin[1]@s.spin[0]-1j*s.spin[2]),
              'orientor_SO3_error': maximum_entry(s.frame_L[0]@s.frame_L[1]-s.frame_L[1]@s.frame_L[0]-1j*s.frame_L[2]),
              'cases': {}}
    stored = {'seeds': seeds, 'incoming_field_sources': incoming, 'outgoing_field_sources': outgoing_raw, 'outgoing_projector_frame': outgoing, 'basis': np.array(s.basis, dtype=np.int64), 'electric': s.electric}
    cases = [('full_vector', s), ('complete_spatial_cut', Sector(replace(p, cut=True))),
             ('current_coupling_off', Sector(replace(p, active_hop=0., supplier_hop=0.))),
             ('orientor_off', Sector(replace(p, torque=0., alignment=0., neutral_hop=0.))),
             ('old_axial_dark_fork', Sector(p, vector=False))]
    for name, sector in cases:
        if sector.basis != s.basis:
            raise RuntimeError('Control changed Hilbert basis')
        data, final = event(sector, seeds, times, outgoing, energy)
        result['cases'][name] = data
        stored[name+'_final'] = final
        print(name, 'dim', sector.dimension, 'unitary', encode(data['changes']['local_unitary']), 'Szcomm', data['spin_H_commutator_max'][2], flush=True)
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(encode(result), indent=2))
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'), **stored)


if __name__ == '__main__':
    main()

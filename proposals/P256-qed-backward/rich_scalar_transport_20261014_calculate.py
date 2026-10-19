"""NEW actual shared-field spatial source/resolvent/packet consumer."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, splu, norm
from rich_scalar_transport_20261014_model import build, ground, sector, block


def avg(op, u):
    return float(np.trace(u.conj().T@(op@u)).real/u.shape[1])


def describe(m, data):
    ix, h, e, vectors, u = data
    nf = m['nfield']
    probability = np.mean(np.abs(u.reshape(-1, nf, u.shape[1]))**2, axis=2).sum(axis=0)
    q = [block(a, ix, ix) for a in m['Q']]
    nums = [block(a, ix, ix) for a in m['N']]
    means = [avg(a, u) for a in q]
    squares = [avg(a@a, u) for a in q]
    s = [block(a, ix, ix) for a in m['S']]
    s2 = sum(a@a for a in s)
    source_spin = np.linalg.eigvalsh(u.conj().T@(s2@u)).real.tolist()
    residual = max(np.linalg.norm(h@u[:, a]-e[0]*u[:, a]) for a in range(u.shape[1]))
    cloud = []
    for x in range(2):
        # Positive actual local charged-occupation projector, not a classical sign.
        px = sparse.diags((nums[x].diagonal().real > 2.5).astype(float))
        weight = avg(px, u)
        cloud.append({'cell': x, 'probability_Nx_greater_than_2': weight,
                      'conditional_phi2_local': avg(px@q[x]@q[x], u)/weight if weight > 1e-10 else None,
                      'conditional_phi2_other': avg(px@q[1-x]@q[1-x], u)/weight if weight > 1e-10 else None,
                      'connected_Nx_phi2_local': avg(nums[x]@q[x]@q[x], u)-avg(nums[x], u)*squares[x],
                      'connected_Nx_phi2_other': avg(nums[x]@q[1-x]@q[1-x], u)-avg(nums[x], u)*squares[1-x]})
    return {'energy': float(e[0]), 'ground_degeneracy': u.shape[1], 'lowest_energies': e.tolist(),
            'N_local': [avg(a, u) for a in nums], 'mean_phi': means, 'phi2': squares,
            'phi0_phi1': avg(q[0]@q[1], u), 'field_distribution': probability.reshape(len(m['phi']), -1).tolist(),
            'charge_conditioned_field': cloud, 'ground_S2_eigenvalues': source_spin,
            'ground_residual': float(residual), 'positive_frame_energy': avg(block(m['casimir'], ix, ix), u),
            'scalar_transfer_energy': avg(block(m['scalar_transfer'], ix, ix), u)}


def spectral(m, ref, charged, hole, p, arrays):
    i0, h0, e0, v0, u0 = ref
    ip, hp, ep, vp, up = charged
    im, hm, em, vm, um = hole
    add = np.stack([block(d, ip, i0)@u0 for d in m['source']], axis=1)
    rem = np.stack([block(d.conj().T, im, i0)@u0 for d in m['source']], axis=1)
    ga = np.einsum('iad,ibd->ab', add.conj(), add)/u0.shape[1]
    gr = np.einsum('iad,ibd->ab', rem.conj(), rem)/u0.shape[1]
    values = []
    for zr, zi in p['resolvent_z']:
        z = zr+1j*zi
        la = splu((z+e0[0])*sparse.eye(hp.shape[0], format='csc')-hp.tocsc())
        lr = splu((z-e0[0])*sparse.eye(hm.shape[0], format='csc')+hm.tocsc())
        a = add.reshape(len(ip), -1)
        r = rem.reshape(len(im), -1)
        xa, xr = la.solve(a), lr.solve(r)
        g = np.einsum('iad,ibd->ab', add.conj(), xa.reshape(add.shape))/u0.shape[1]
        g += (np.einsum('iad,ibd->ab', rem.conj(), xr.reshape(rem.shape))/u0.shape[1]).T
        arrays[f'G_{zr}'] = g
        values.append({'z': [zr, zi], 'real': g.real.tolist(), 'imag': g.imag.tolist(),
                       'addition_resolvent_residual': float(np.linalg.norm((z+e0[0])*xa-hp@xa-a)),
                       'removal_resolvent_residual': float(np.linalg.norm((z-e0[0])*xr+hm@xr-r))})
    wa = np.linalg.norm(np.einsum('ji,iad->jad', up.conj().T, add))**2/u0.shape[1]
    wr = np.linalg.norm(np.einsum('ji,iad->jad', um.conj().T, rem))**2/u0.shape[1]
    # Every virtual scalar/Fock state enters the resolvents above; low vectors only
    # identify the actual threshold eigenspace and are not a G truncation.
    return {'addition_gap': float(ep[0]-e0[0]), 'removal_gap': float(em[0]-e0[0]),
            'CAR_Gram_error': float(np.linalg.norm(ga+gr.T-np.eye(8))),
            'threshold_addition_total_weight': float(wa), 'threshold_removal_total_weight': float(wr),
            'addition_Gram_real': ga.real.tolist(), 'removal_Gram_real': gr.real.tolist(),
            'full_virtual_resolvents': values, 'chemical_potential': 0.0}


def packet(m, ref, charged, p, pulse=False):
    i0, h0, e0, v0, u0 = ref
    ix, h, e, v, u = charged
    initial = block(m['source'][1], ix, i0)@u0
    initial /= np.linalg.norm(initial)
    if pulse:
        # Field-only canonical momentum kick at FIXED cell0, independent of N/X.
        qdiag = block(m['Q'][0], ix, ix).diagonal().real
        initial = np.exp(-1j*p['field_pulse']*qdiag)[:, None]*initial
    times = np.linspace(0, p['time_end'], p['time_points'])
    states = expm_multiply(-1j*h, initial, start=0, stop=p['time_end'], num=p['time_points'], traceA=-1j*h.diagonal().sum())
    nums = [block(a, ix, ix) for a in m['N']]
    q = [block(a, ix, ix) for a in m['Q']]
    currents = [block(a, ix, ix) for a in m['current']]
    def trajectory(op):
        acted = (op@states.transpose(1, 0, 2).reshape(len(ix), -1)).reshape(len(ix), len(times), -1).transpose(1, 0, 2)
        return np.einsum('tid,tid->t', states.conj(), acted).real
    density = np.array([trajectory(a) for a in nums]).T
    phi2 = np.array([trajectory(a@a) for a in q]).T
    j = np.array([trajectory(a) for a in currents]).T
    cov = trajectory(currents[0]@q[0]@q[0])-j[:, 0]*phi2[:, 0]
    field_prob = np.abs(states.reshape(len(times), -1, m['nfield'], u0.shape[1]))**2
    field_prob = field_prob.sum(axis=(1, 3))
    return {'times': times.tolist(), 'N_local': density.tolist(), 'phi2': phi2.tolist(),
            'currents': j.tolist(), 'current_phi0_squared_connected': cov.tolist(),
            'field_probability_initial': field_prob[0].tolist(), 'field_probability_final': field_prob[-1].tolist(),
            'max_remote_number_change': float(np.max(np.abs(density[:, 1]-density[0, 1]))),
            'max_field_phi2_change': float(np.max(np.abs(phi2-phi2[0]))),
            'final_field_probability_L1_change': float(np.sum(np.abs(field_prob[-1]-field_prob[0]))),
            'norm_error': float(np.max(np.abs(np.sum(np.abs(states)**2, axis=(1, 2))-1))),
            'energy': float(np.vdot(initial, h@initial).real)}, states


def main():
    p = json.loads(Path(sys.argv[1]).read_text())
    output = Path(sys.argv[2])
    m = build(p)
    print('Solving actual occupied N4 and adjacent N3/N5', flush=True)
    data = {n: ground(m, n) for n in (3, 4, 5)}
    arrays = {'phi': m['phi'], 'parameters_utf8': np.frombuffer(json.dumps(p).encode(), dtype=np.uint8)}
    for n, datum in data.items():
        arrays[f'N{n}_ground'] = datum[-1]
        arrays[f'N{n}_indices'] = datum[0]
    result = {'candidate': p['candidate'], 'parameters': p, 'dimension': {'full': m['H'].shape[0], 'N3': len(data[3][0]), 'N4': len(data[4][0]), 'N5': len(data[5][0])},
              'sectors': {str(n): describe(m, datum) for n, datum in data.items()}}
    print('Solving full virtual addition/removal G', flush=True)
    result['source_G'] = spectral(m, data[4], data[5], data[3], p, arrays)
    result['packet'], arrays['packet_states'] = packet(m, data[4], data[5], p)
    result['field_pulsed_packet'], arrays['field_pulsed_states'] = packet(m, data[4], data[5], p, True)
    result['field_pulse_effect'] = {'remote_number_max_difference': float(np.max(np.abs(np.array(result['packet']['N_local'])[:, 1]-np.array(result['field_pulsed_packet']['N_local'])[:, 1]))),
                                  'final_field_distribution_L1_difference': float(np.sum(np.abs(np.array(result['packet']['field_probability_final'])-np.array(result['field_pulsed_packet']['field_probability_final']))))}
    currents = m['current']
    n0 = m['N'][0]
    continuity = 1j*(m['H']@n0-n0@m['H'])-(currents[0]-currents[1])
    s2 = sum(a@a for a in m['S'])
    result['operators'] = {'number_continuity_full_norm': float(norm(continuity)),
                           'spin_rotation_commutator_full_norm': float(norm(m['H']@s2-s2@m['H'])),
                           'canonical_C_covariance_full_norm_at_zero_twist': float(norm(m['H']@m['C']-m['C']@m['H']))}
    eps = 2e-4
    derivative = []
    for a in range(2):
        plus, minus = list(p['angles']), list(p['angles'])
        plus[a] += eps
        minus[a] -= eps
        hp = build(p, angles=plus)['H']
        hm = build(p, angles=minus)['H']
        derivative.append({'link': a, 'current_relative_derivative_error': float(norm(-(hp-hm)/(2*eps)-currents[a])/max(1, norm(currents[a]))),
                           'contact_relative_derivative_error': float(norm((hp+hm-2*m['H'])/eps**2-m['contact'][a])/max(1, norm(m['contact'][a])))})
    result['operators']['derivatives'] = derivative
    print('Computing material hopoff and field-coupling-off controls', flush=True)
    result['controls'] = {}
    for name, model in [('coupling_off', build(p, coupling=0)), ('hop_off', build(p, hop=0))]:
        ref, cha, hol = ground(model, 4, count=12), ground(model, 5, count=12), ground(model, 3, count=12)
        baseline, states = packet(model, ref, cha, p)
        pulsed, pulse_states = packet(model, ref, cha, p, True)
        result['controls'][name] = {'neutral': describe(model, ref), 'charged': describe(model, cha),
                                    'addition_gap': float(cha[2][0]-ref[2][0]), 'removal_gap': float(hol[2][0]-ref[2][0]),
                                    'packet': baseline,
                                    'field_pulse_remote_number_effect': float(np.max(np.abs(np.array(baseline['N_local'])[:, 1]-np.array(pulsed['N_local'])[:, 1]))),
                                    'field_pulse_final_field_L1_difference': float(np.sum(np.abs(np.array(baseline['field_probability_final'])-np.array(pulsed['field_probability_final']))))}
    print('Computing literal many-body twist response', flush=True)
    twist = []
    for a in p['twist_steps']:
        mm = m if a == 0 else build(p, angles=[a, a])
        rr = data[4] if a == 0 else ground(mm, 4)
        cc = data[5] if a == 0 else ground(mm, 5)
        twist.append({'uniform_link_twist': a, 'neutral_energy': float(rr[2][0]), 'charged_energy': float(cc[2][0]), 'source_gap': float(cc[2][0]-rr[2][0])})
    result['twist_response'] = twist
    da = p['twist_steps'][-1]
    curvature = (twist[2]['source_gap']+twist[0]['source_gap']-2*twist[1]['source_gap'])/da**2
    result['rest_inertial_debt'] = {'finite_twist_charge_gap_curvature': float(curvature), 'finite_ring_not_continuum_inertial_mass': True,
                                     'scalar_generated_gap_increment': result['source_G']['addition_gap']-result['controls']['coupling_off']['addition_gap']}
    output.write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'), **arrays)
    print(json.dumps({'output': str(output), 'gap': result['source_G']['addition_gap'], 'g0_gap': result['controls']['coupling_off']['addition_gap'], 'motion': result['packet']['max_remote_number_change'], 'hopoff_motion': result['controls']['hop_off']['packet']['max_remote_number_change'], 'operators': result['operators']}), flush=True)


if __name__ == '__main__':
    main()

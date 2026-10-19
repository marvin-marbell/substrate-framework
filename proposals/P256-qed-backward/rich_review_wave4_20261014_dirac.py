"""Contrary actual contact geometry and full-state charged receivers."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import expm
from rich_dirac_contact_20261014_model import Parameters, build, hamiltonian, spectrum, vertex, contact, charged_green, scattering
from rich_mobile_spinor_20261014_model import contact_values
from rich_review_wave4_20261014_tools import complete_resolvent, complex_pair, save


def probe(model: dict, k: np.ndarray, qin: np.ndarray, qout: np.ndarray) -> dict:
    zero = np.zeros(3)
    e0, v0 = spectrum(model, zero)
    block = v0[:, :4]
    projected = np.array([block.conj().T @ vertex(model, zero, zero, direction) @ block / model['params'].charge for direction in np.eye(3)])
    velocity_sq = np.array([np.trace(a @ a).real / 4 for a in projected])
    clifford = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            clifford[a, b] = np.linalg.norm(projected[a] @ projected[b] + projected[b] @ projected[a] - (2 * velocity_sq[a] * np.eye(4) if a == b else 0))
    center = float(np.mean(e0[:4]))
    mass = float((e0[2]-e0[0])/2)
    beta = (np.diag(e0[:4])-center*np.eye(4))/mass
    beta_alpha = [float(np.linalg.norm(beta @ a+a @ beta)) for a in projected]
    momenta = [zero, k, -k, 2*k]
    energy_rows = [spectrum(model, momentum)[0][:8] for momentum in momenta]
    actual_scalar_shift = [float(np.mean(row[:4])-center) for row in energy_rows]
    e, v = spectrum(model, k)
    z = complex(2.41, .17)
    source = model['source']
    direct_g = source.conj().T @ complete_resolvent(hamiltonian(model, k), z, source)
    supplied_g = charged_green(model, k, z)
    spectral_sources = v.conj().T @ source
    ward = sum(qin[a] * vertex(model, k+qin, k, np.eye(3)[a]) for a in range(3))
    ward -= model['params'].charge*(hamiltonian(model, k+qin)-hamiltonian(model, k))
    epsin = np.array([.2, .7, -.4]); epsout = np.array([.6, -.3, .5])
    wi = .71; eta = .043; initial = 3; final = 2; kout = k+qin-qout
    ef, vf = spectrum(model, kout); a = v[:, initial]; b = vf[:, final]
    wo = wi+e[initial]-ef[final]
    jin = vertex(model, k+qin, k, epsin); jout = vertex(model, kout, k+qin, epsout)
    je = vertex(model, k-qout, k, epsout); ji = vertex(model, kout, k-qout, epsin)
    co = contact(model, kout, k, qin, epsin, qout, epsout)
    direct = b.conj() @ (co @ a+jout @ complete_resolvent(hamiltonian(model, k+qin), e[initial]+wi+1j*eta, jin @ a)
                        +ji @ complete_resolvent(hamiltonian(model, k-qout), e[initial]-wo+1j*eta, je @ a))
    full = scattering(model, k, qin, epsin, qout, epsout, wi, initial, final, eta)
    low = scattering(model, k, qin, epsin, qout, epsout, wi, initial, final, eta, np.arange(4))
    low_g = spectral_sources[:4].conj().T @ ((1/(z-e[:4]))[:, None] * spectral_sources[:4])
    jproj = np.array([block.conj().T @ j @ block for j in model['J']])
    vector_errors = []
    for aindex in range(3):
        for bindex in range(3):
            target = np.zeros((4,4), complex)
            if aindex != bindex:
                cindex = 3-aindex-bindex
                sign = 1 if (aindex,bindex,cindex) in [(0,1,2),(1,2,0),(2,0,1)] else -1
                target = 1j*sign*projected[cindex]
            vector_errors.append(float(np.linalg.norm(jproj[aindex] @ projected[bindex]-projected[bindex] @ jproj[aindex]-target)))
    return {'parameters': model['parameters'], 'dimension': len(e), 'k': k.tolist(), 'raw_low8_at_zero_k_minus_k_twok': [row.tolist() for row in energy_rows],
            'kramers_pair_splittings': [float(row[1]-row[0]) for row in energy_rows],
            'constant_parity_center': center, 'mass_half_separation': mass, 'unsifted_low4_center_shift': actual_scalar_shift,
            'projected_velocity_squared_by_axis': velocity_sq.tolist(), 'clifford_absolute_errors': clifford.tolist(),
            'beta_velocity_anticommutators': beta_alpha, 'physical_rotation_vector_errors_max': max(vector_errors),
            'projected_current_traces': [complex_pair(np.trace(op)) for op in projected],
            'source_isometry_error': float(np.linalg.norm(source.conj().T @ source-np.eye(source.shape[1]))),
            'full_G_direct_error': float(np.linalg.norm(supplied_g-direct_g)), 'low4_G_direct_error': float(np.linalg.norm(low_g-direct_g)),
            'total_source_weight': float(np.linalg.norm(spectral_sources)**2), 'low4_source_weight': float(np.linalg.norm(spectral_sources[:4])**2),
            'full_source_closure_error': float(np.linalg.norm(spectral_sources.conj().T @ spectral_sources-source.conj().T @ source)),
            'finite_q_Ward_error': float(np.linalg.norm(ward)), 'full_scattering': complex_pair(full['amplitude']),
            'full_scattering_direct_error': float(abs(full['amplitude']-direct)), 'low4_scattering_error': float(abs(low['amplitude']-direct)),
            'actual_recoil': float(full['recoil_energy']), 'outgoing_frequency': float(full['outgoing_energy']),
            'charged_L2_low4': [float(np.vdot(v0[:,i], model['Lc2'] @ v0[:,i]).real) for i in range(4)]}


def main() -> None:
    inputs = json.loads(Path(sys.argv[1]).read_text())
    p = Parameters(**inputs['parameters'])
    k = np.array(inputs['k']); qin = np.array(inputs['qin']); qout = np.array(inputs['qout'])
    facing = build(replace(p, geometry='facing'))
    interference = build(replace(p, geometry='interference'))
    # A real unequal facing-channel amplitude is an actual alternate microscopic contact,
    # retaining all cavity/neutral/charged states and no momentum-dependent energy redefinition.
    imperfect = build(replace(p, geometry='facing'))
    nn = len(imperfect['neutral']['basis'])
    imperfect['T'] = []
    for index, direction in enumerate(imperfect['directions']):
        facing_values = contact_values(imperfect['charged']['basis'], direction)
        receiving_values = contact_values(imperfect['charged']['basis'], -direction)
        facing_orbital = np.outer(facing_values.conj(), receiving_values)/np.vdot(facing_values,facing_values).real
        orbital = -p.hopping*(facing_orbital-inputs['reverse_channel_amplitude']*facing_orbital.conj().T)/2
        field_link = expm(-1j*p.charge*p.cavity_coupling*p.spacing*imperfect['field']['Q'][index])
        imperfect['T'].append(np.kron(np.kron(np.kron(orbital,np.eye(2)),np.eye(nn)),field_link))
    imperfect['parameters'] = dict(imperfect['parameters'], geometry='unequal-opposite-real-facing-channels', reverse_channel_amplitude=inputs['reverse_channel_amplitude'])
    result = {name: probe(model,k,qin,qout) for name,model in [('facing',facing),('interference',interference),('unequal_channels',imperfect)]}
    save('dirac', {'kind':'INDEPENDENT_NEW_CONTACT_GEOMETRY_AND_COMPLETE_RECEIVER','inputs':inputs,'cases':result,
                   'scope':'Empty-charge neutral/cavity vacuum addition G, not occupied removal or positrons; supplied cavity and primitive isotropic LS/spin/path signs remain inputs. Pair degeneracy is expected. All momentum energies retained without subtraction.'})

if __name__ == '__main__':
    main()

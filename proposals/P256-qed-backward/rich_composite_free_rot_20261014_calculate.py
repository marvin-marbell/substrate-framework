"""NEW free-material angular selection, magnetic source and full virtual response."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, cast
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from rich_composite_free_rot_20261014_model import material, sector, spin_matrices, rotation_matrices, magnetic_projected, magnetic_contact, lab_internal
from rich_composite_free_rot_20261014_current import current_projected, dipole_projected
from rich_composite_rot_20261014_consumer import spectrum, high_angular_sector


def low_states(h, number=4):
    rng = np.random.default_rng(2220414)
    initial = rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0])
    energy, states = eigsh(h, k=number, which='SA', tol=1e-11, v0=initial)
    order = np.argsort(energy)
    energy, states = energy[order], states[:, order]
    block = states[:, abs(energy-energy[0]) < 2e-8]
    q, _ = np.linalg.qr(block)
    return energy, q


def mean(block, operator):
    return float(np.trace(block.conj().T @ (operator @ block)).real/block.shape[1])


def orientation_calculation():
    rows = []
    for j in (.5, 1.5):
        numbers = np.arange(-j, j+1)
        jops = [np.kron(a, np.eye(len(numbers))) for a in spin_matrices(j)]
        nops = [np.kron(np.eye(len(numbers)), a.T) for a in spin_matrices(j)]
        r = [[np.block([[rotation_matrices(j, float(mp), j, float(m))[i][a] for m in numbers] for mp in numbers]) for a in range(3)] for i in range(3)]
        mechanical = [sum(r[i][a] @ nops[a] for a in range(3)) for i in range(3)]
        errors = []
        for i in range(3):
            for axis in range(3):
                # Cartesian physical-vector commutator [J_i,V_a]=i eps_iak V_k.
                expected = np.zeros_like(r[axis][0])
                if i != axis:
                    k = 3-i-axis
                    sign = 1 if (i, axis, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1
                    expected = 1j*sign*r[k][0]
                errors.append(float(np.linalg.norm(jops[i] @ r[axis][0]-r[axis][0] @ jops[i]-expected)))
        rows.append(dict(j=j, full_MK_dimension=len(numbers)**2, R_N_equals_lab_J_errors=[float(np.linalg.norm(mechanical[a]-jops[a])) for a in range(3)], vector_commutator_max=max(errors)))
    return rows


def summarize_branch(p, tag, js):
    mat = material(p)
    rows, stored = [], {}
    for j in js:
        h, derivatives = sector(mat, p, j)
        energies, block = low_states(h)
        rows.append(dict(j=j, M=.5, all_M_multiplicity=int(2*j+1), dimension_per_M=cast(tuple[int, int], h.shape)[0], all_K_internal_states=True, energies=energies.tolist(), ground_copies_found=block.shape[1], eigen_residual_max=float(np.linalg.norm(h @ block-energies[0]*block, axis=0).max()), mechanical_angular_square=mean(block, sum(a @ a for a in derivatives))))
        stored[j] = h, energies, block
        print(json.dumps(dict(branch=tag, angular=rows[-1])), flush=True)
    h, energies, block = stored[.5]
    mz = magnetic_projected(mat, p, .5, .5, .5, .5, 2)
    mech = magnetic_projected(mat, p, .5, .5, .5, .5, 2, 'mechanical')
    hop = magnetic_projected(mat, p, .5, .5, .5, .5, 2, 'peierls')
    iz = lab_internal(mat, .5, .5, .5, .5, 2)
    spin_z = lab_internal(mat, .5, .5, .5, .5, 2, 'S')
    carrier_z = lab_internal(mat, .5, .5, .5, .5, 2, 'L')
    lab_internal_z = mean(block, iz)
    result = dict(tag=tag, inputs=p, angular_sectors=rows, lowest_j_among_compared=min(rows, key=lambda row: row['energies'][0])['j'], source_j=.5, J2=.75, Jz=.5, physical_J_definition='lab mechanical orientation + rotated constituent spin + all rotated carrier momenta; left Noether generator in the body gauge', bare_spin_lab_z=mean(block, spin_z), carrier_lab_z=mean(block, carrier_z), total_internal_lab_z=lab_internal_z, mechanical_lab_z=.5-lab_internal_z, magnetic_total_z=mean(block, mz), magnetic_mechanical_z=mean(block, mech), magnetic_peierls_z=mean(block, hop), magnetic_ground_copy_eigenvalues=np.linalg.eigvalsh(block.conj().T @ (mz @ block)).tolist(), carrier_excitation_number=mean(block, sparse.kron(sparse.eye(2), mat['body']['carrier_energy'], format='csr')))
    result['load_energy'] = mean(block, sparse.kron(sparse.eye(2), mat['H']-material(dict(p, loading_g=0.))['H'], format='csr'))
    return result, mat, stored


def run(p, output):
    geometry = orientation_calculation()
    selected, mat, stored = summarize_branch(p, 'free_material_axial', p['compare_j'])
    h, energies, block = stored[.5]
    psi = block[:, 0]
    moment = magnetic_projected(mat, p, .5, .5, .5, .5, 2)
    contact = magnetic_contact(mat, p, .5, .5, 2)
    selected['magnetic_contact'] = mean(block, contact)
    selected['response'] = []
    for jp in (.5, 1.5):
        hp = stored[jp][0]
        source = magnetic_projected(mat, p, jp, .5, .5, .5, 2) @ psi
        spectral = cast(dict[str, Any], spectrum(hp, energies[0], source, p['krylov_steps']))
        poles, weights = np.asarray(spectral['poles']), np.asarray(spectral['weights'])
        z = complex(*p['response_z'])
        spectral['retarded_source_resolvent_real_imag'] = [float(v) for v in (np.sum(weights/(z-poles)).real, np.sum(weights/(z-poles)).imag)]
        spectral['ground_block_source_weight'] = float(np.linalg.norm(stored[jp][2].conj().T @ source)**2)
        selected['response'].append(dict(j_virtual=jp, M_virtual=.5, dimension=hp.shape[0], magnetic=spectral))
    # The dipole Ward identity reaches literal lab positions and charge motion.
    ward = []
    for jp in (.5, 1.5):
        hp = stored[jp][0]
        for axis in range(3):
            for mp in np.arange(-jp, jp+1):
                d = dipole_projected(mat, jp, float(mp), .5, .5, axis)
                current = current_projected(mat, p, jp, float(mp), .5, .5, axis)
                hop = current_projected(mat, p, jp, float(mp), .5, .5, axis, 'peierls')
                expected = 1j*(hp @ (d @ psi)-d @ (h @ psi))
                ward.append(dict(j_virtual=jp, M_virtual=float(mp), axis=axis, source_norm=float(np.linalg.norm(current @ psi)), continuum_current_ward_error=float(np.linalg.norm(current @ psi-expected)), wrong_hopping_only_ward_error=float(np.linalg.norm(hop @ psi-expected)), mechanical_current_source_norm=float(np.linalg.norm((current-hop) @ psi))))
    selected['actual_lab_charge_current'] = ward
    # All-half-j comparison: numerical isotropic spectral floor, not a rigorous
    # interval certificate. Representation inequality then covers unscanned j.
    isotropic_p = dict(p, axial_bias=0.)
    iso_mat = material(isotropic_p)
    iso_energies, _ = low_states(iso_mat['H'])
    high = high_angular_sector(iso_mat['H'], iso_mat['body'])
    Bmin = min(1/(2*np.asarray(p['inertia'])))
    selected['all_higher_j_lower_comparison'] = dict(isotropic_body_low=float(iso_energies[0]), isotropic_body_high_T_at_least_three_halves=high, axial_operator_norm_bound=2*abs(p['axial_bias']), representation_kinetic_gap_for_T_half_and_j_at_least_three_halves=2*float(Bmin), numerical_floor_all_j_at_least_three_halves=float(min(iso_energies[0]+2*Bmin, high['energy'])-2*abs(p['axial_bias'])), scope='Exact representation/norm inequality with numerically obtained isotropic spectral minima; not a certified global numerical eigenvalue bound. Explicit unrestricted-K/internal j=.5,1.5,2.5 comparisons are direct calculations.')
    iso_result, _, _ = summarize_branch(isotropic_p, 'free_material_isotropic', [.5])
    reverse_result, _, _ = summarize_branch(dict(p, axial_bias=-p['axial_bias']), 'free_material_reverse_pretwist', [.5])
    no_loading_result, _, _ = summarize_branch(dict(p, loading_g=0.), 'free_material_no_spin_load', [.5])
    # Actual external uniform-B receiver, quadratic small-field Hamiltonian in
    # j=.5,1.5. Contact is continuum-before-projection, not squared P mu P.
    # Dropped j>=2.5 enter the ground energy first beyond quadratic response.
    hp = stored[1.5][0]
    m01 = magnetic_projected(mat, p, 1.5, .5, .5, .5, 2)
    m11 = magnetic_projected(mat, p, 1.5, .5, 1.5, .5, 2)
    receiver_h = sparse.block_diag([h, hp], format='csr')
    receiver_mu = sparse.bmat([[moment, m01.conj().T], [m01, m11]], format='csr')
    receiver_contact = sparse.block_diag([contact, magnetic_contact(mat, p, 1.5, .5, 2)], format='csr')
    receiver = []
    for b in p['receiver_B']:
        eh, bh = low_states(receiver_h-b*receiver_mu+.5*b*b*receiver_contact, 2)
        receiver.append(dict(B=b, energy=float(eh[0]), moment=mean(bh, receiver_mu-b*receiver_contact), eigen_residual_max=float(np.linalg.norm((receiver_h-b*receiver_mu+.5*b*b*receiver_contact) @ bh-eh[0]*bh, axis=0).max())))
    selected['external_magnetic_receiver'] = dict(field_is_supplied_not_induced=True, law='H_B=P(H0-B.mu+B^2 contact/2)P; j=.5,1.5 retained; cross-j contact omitted so receiver is correct ground energy through O(B^2), not an exact finite-B angular law.', observations=receiver)
    result = dict(candidate='FREE_SO3_MATERIAL_SPIN_DIFFERENCE_ROTOR_V1', inputs=p, orientation_geometry=geometry, selected=selected, isotropic=iso_result, reverse_pretwist=reverse_result, no_spin_loading=no_loading_result, binding_reference=dict(kind='ANTECEDENT_ONLY', predecessor='rich_composite_rot_20261014_magnetic1_receipt.json', reason='Old fixed-square dressed pair+single reference does not include the new free orientation/mechanical degree; no new free-body breakup margin is asserted. Neutral load feedback is freshly recomputed but rigid material scaffold is input.'), physical_selection_scope='M=.5 contains every half-integer j; complete K/internal space per compared j; all M copies exist by exact lab covariance. .5,1.5,2.5 explicitly compared and all larger j have conditional numerical spectral-floor comparison. This is a rigid-body scaffold, not self-organized geometry or a free proton.')
    Path(output+'.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(output+'.npz', ground_Mhalf=block, ground_energy=energies[0], higher_j_energies=stored[1.5][1], positions=cast(np.ndarray, mat['body']['positions']), charges=cast(np.ndarray, mat['body']['charges']))
    print(json.dumps(dict(result=output+'.json', selected=selected['magnetic_total_z'], mechanical=selected['magnetic_mechanical_z'], peierls=selected['magnetic_peierls_z'], ward_max=max(row['continuum_current_ward_error'] for row in ward), orientation=geometry)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.request).read_text()), args.output)

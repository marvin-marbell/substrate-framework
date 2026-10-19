#!/usr/bin/env python3
"""Reviewer-owned finite physical consumer; no author entrypoint or cached state.

Distinct laws/states remain distinct: cubic Gaussian D, compact neutral Gauss
plaquette, charged one-flux bundle, and neutral-breather trimer. Numerical
controls change actual Hamiltonians/sources, never substitute stored answers.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

import rich_photon_20261014_loop as photon
import rich_electron_20261014_hubbard_charged as electron
import rich_interaction_20261014_coupled as interaction
import rich_composite_20261014_consumer as composite
import rich_composite_20261014_motion as motion


def data(a):
    a = np.asarray(a)
    return {'real': a.real.tolist(), 'imag': a.imag.tolist()}


def chi(energies, states, ground, left, right, z):
    """Independent connected retarded commutator, both frequency signs."""
    gaps = energies[1:]-energies[0]
    excited = states[:, 1:]
    return (np.sum((ground.conj()@left@excited)*(excited.conj().T@right@ground)/(z-gaps))
            - np.sum((ground.conj()@right@excited)*(excited.conj().T@left@ground)/(z+gaps)))


def photon_probe(p):
    matter = photon.LoopMatter(photon.Parameters())
    q = np.asarray(p['photon_q']); z = complex(*p['z'])
    matrix, _ = photon.collective_matrix(matter, q)
    independent = np.linalg.inv(matrix-z*z*np.eye(len(matrix)))[:3, :3]
    direct = photon.propagator(matter, q, z)
    frame = photon.transverse_frame(q)
    values, residues, fields, _ = photon.collective_modes(matter, q)
    selected = np.flatnonzero((values > 1e-8) & (values < matter.gaps.min()**2))
    decoupled = photon.LoopMatter(photon.Parameters(link_charge=0.0))
    decoupled_matrix, _ = photon.collective_matrix(decoupled, q)
    frozen = photon.polarization(matter, q, z, frozen=True)
    full = photon.polarization(matter, q, z)
    response, contact, ward = photon.density_current_response(matter, q, z)
    return {'state': 'isolated N2 correlated ring poles, quantized cubic Gaussian collective state',
            'q': q.tolist(), 'z': [z.real, z.imag], 'D_full': data(direct),
            'D_transverse': data(frame.conj().T@direct@frame),
            'independent_collective_resolvent_error': float(np.linalg.norm(independent-direct)),
            'positive_submatter_omegas': np.sqrt(values[selected]).tolist(),
            'submatter_field_residues': residues[selected].tolist(),
            'field_residue_sum': float(residues.sum()),
            'gauss_null_residual': photon.collective_gauss(matter, q)[1],
            'density_current_ward_error': ward,
            'four_current': data(response), 'native_contact': data(contact),
            'removed_charge_Pi_norm': float(np.linalg.norm(photon.polarization(decoupled, q, z))),
            'removed_charge_field_eigenvalues': np.linalg.eigvalsh(decoupled_matrix).tolist(),
            'frozen_contact_only_Pi_difference': float(np.linalg.norm(frozen-full)),
            'scope': 'Gaussian field response is not the nonlinear compact Gauss ground below.'}


def compact_probe(p):
    model = interaction.CoupledMolecule(cutoff=p['electric_cutoff'])
    frozen = interaction.CoupledMolecule(frozen=True)
    qin, qout = np.array(p['qin']), np.array(p['qout'])
    z = complex(*p['z'])
    incoming = model.vertex(qin).reshape(2, 2, len(model.basis), len(model.basis)).sum(axis=0)
    outgoing = model.vertex(qout).reshape(2, 2, len(model.basis), len(model.basis)).sum(axis=0)
    physical = model.response(qout, qin, z).reshape(2, 2, 2, 2).sum(axis=(0, 2))
    contact = model.two_leg_contact(qout, qin).reshape(2, 2, 2, 2, len(model.basis), len(model.basis)).sum(axis=(0, 2))
    independent = np.empty((2, 2), complex)
    for a in range(2):
        for b in range(2):
            independent[a, b] = np.vdot(model.ground, contact[a, b]@model.ground)+chi(
                model.energies, model.states, model.ground, outgoing[a].conj().T, incoming[b], z)
    frozen_response = frozen.response(qout, qin, z).reshape(2, 2, 2, 2).sum(axis=(0, 2))
    flux = np.diag(model.electric.mean(axis=1))
    field_response = chi(model.energies, model.states, model.ground, flux, flux, z)
    density = np.array([[((bits >> (2*s)) & 1)+((bits >> (2*s+1)) & 1)-.5
                         for s in range(4)] for bits, _ in model.basis])
    gauss = model.electric-np.roll(model.electric, 1, axis=1)+density
    continuity = []
    for s in range(4):
        rho = np.diag(density[:, s])
        js = model.current.sum(axis=1)
        continuity.append(np.linalg.norm(1j*(model.H@rho-rho@model.H)-(js[(s-1) % 4]-js[s])))
    omega, eta = p['incoming_omega'], p['z'][1]
    active = []
    for final in np.flatnonzero((model.gaps > 1e-8) & (model.gaps < omega)):
        tensor = model.raman(qout, qin, omega, eta, int(final)).reshape(2, 2, 2, 2).sum(axis=(0, 2))
        active.append((float(np.linalg.norm(tensor)**2), int(final), tensor))
    weight, final, tensor = max(active, key=lambda row: row[0])
    elastic = model.raman(qout, qin, omega, eta, 0)
    truncated = model.raman(qout, qin, omega, eta, 0, ground_only=True)
    times = model.time_motion(qin)
    return {'law_id': interaction.LAW_ID, 'state_id': model.state_id,
            'dimension': len(model.basis), 'ground_energy': float(model.energies[0]),
            'first_gap': float(model.gaps[1]), 'frozen_first_gap': float(frozen.gaps[1]),
            'gauss_basis_max_error': float(np.max(np.abs(gauss))),
            'operator_continuity_max_error': float(max(continuity)),
            'physical_nonforward_response': data(physical),
            'independent_response_error': float(np.linalg.norm(independent-physical)),
            'physical_response_difference_from_frozen': float(np.linalg.norm(physical-frozen_response)),
            'electric_circulation_retarded_chi': data(field_response),
            'electric_circulation_source_norm': float(np.linalg.norm(flux@model.ground)),
            'electric_energy_time_range': float(np.ptp(times[:, 1])),
            'double_occupancy_time_range': float(np.ptp(times[:, 2])),
            'total_energy_time_range': float(np.ptp(times[:, 3])),
            'elastic_full_minus_ground_only_norm': float(np.linalg.norm(elastic-truncated)),
            'strongest_physical_inelastic': {'final': final, 'weight': weight,
                'energy_transfer': float(model.gaps[final]),
                'outgoing_energy': float(omega-model.gaps[final]), 'tensor': data(tensor)},
            'missing_interface': {'charged_G': 'Absent: fixed background and periodic summed Gauss require N=2. N1/N3 need explicit compensating charge/boundary flux plus a dressed CAR source.',
                                  'cubic_two_polarization_D': 'Absent: one physical plaquette circulation. Electric chi above is a joint-state field probe, not the cubic Gaussian A propagator.'}}


def creation_map(neutral_basis, charged_basis, spin, k, creation=True):
    index = {bits: i for i, bits in enumerate(charged_basis)}
    result = np.zeros((len(charged_basis), len(neutral_basis)), complex)
    site_maps = []
    for site in range(4):
        op = np.zeros_like(result); mode = 2*site+spin
        for j, bits in enumerate(neutral_basis):
            occupied = bool(bits & (1 << mode))
            if occupied != creation:
                target = bits | (1 << mode) if creation else bits ^ (1 << mode)
                op[index[target], j] = (-1)**((bits & ((1 << mode)-1)).bit_count())
        coefficient = np.exp((-1j if creation else 1j)*k*site)/2
        site_maps.append(coefficient*op)
        result += site_maps[-1]
    return result, site_maps


def large_gauge(basis):
    return np.array([np.exp(.5j*np.pi*sum(site*(((bits >> (2*site)) & 1)+((bits >> (2*site+1)) & 1))
                                          for site in range(4))) for bits in basis])


def spin_squared(basis):
    lookup = {bits: i for i, bits in enumerate(basis)}
    raising = np.zeros((len(basis), len(basis)), complex)
    sz = np.zeros(len(basis))
    for j, bits in enumerate(basis):
        for site in range(4):
            sz[j] += .5*(((bits >> (2*site)) & 1)-((bits >> (2*site+1)) & 1))
            moved = electron.hop(bits, 2*site, 2*site+1)
            if moved is not None:
                dest, sign = moved; raising[lookup[dest], j] += sign
    return np.diag(sz**2)+.5*(raising@raising.conj().T+raising.conj().T@raising)


def electron_probe(p):
    parameters = photon.Parameters(); ng = p['flux_grid']; nlow = p['charged_low_states']
    sectors = {n: electron.eigensystem(n, parameters, ng=ng, k=nlow) for n in (1, 2, 3)}
    e0 = sectors[2][0][0]; psi = sectors[2][1][:, 0].reshape(ng, -1)
    mu_plus = float(sectors[3][0][0]-e0); mu_minus = float(e0-sectors[1][0][0])
    rows = []; z = complex(*p['z'])
    for spin in (0, 1):
        for k in (0., np.pi/2):
            row = {'spin': spin, 'k': k}; gs = {}
            for n, creation in ((3, True), (1, False)):
                energies, states, basis, _ = sectors[n]
                op, pieces = creation_map(sectors[2][2], basis, spin, k, creation)
                vn, vc = large_gauge(sectors[2][2]), large_gauge(basis)
                mismatch = op-vc[:, None]*op*vn.conj()[None, :]
                bare = np.concatenate([op@part for part in psi])
                anchored = []
                sign = 1 if creation else -1
                for grid, part in enumerate(psi):
                    Q = 2*np.pi*grid/ng
                    patched = sum(np.exp(sign*1j*site*Q/4)*piece for site, piece in enumerate(pieces))
                    anchored.append(patched@part)
                anchored = np.concatenate(anchored)
                poles = energies-e0 if creation else e0-energies
                bare_weights = np.abs(states.conj().T@bare)**2
                anchored_weights = np.abs(states.conj().T@anchored)**2
                g_bare = np.sum(bare_weights/(z-poles)); g_anchored = np.sum(anchored_weights/(z-poles))
                label = 'addition' if creation else 'removal'; gs[label] = g_bare
                source_s2 = sparse.kron(sparse.eye(ng), sparse.csr_matrix(spin_squared(basis)))
                row[label] = {'poles': poles.tolist(), 'bare_weights': bare_weights.tolist(),
                    'anchored_weights': anchored_weights.tolist(), 'bare_source_norm2': float(np.vdot(bare, bare).real),
                    'captured_bare_weight': float(bare_weights.sum()),
                    'anchored_source_norm2': float(np.vdot(anchored, anchored).real),
                    'captured_anchored_weight': float(anchored_weights.sum()),
                    'bare_large_gauge_patch_relative_error': float(np.linalg.norm(mismatch)/np.linalg.norm(op)),
                    'anchored_patch_error': float(np.linalg.norm(sum(np.exp(sign*.5j*np.pi*site)*piece for site, piece in enumerate(pieces))-vc[:, None]*op*vn.conj()[None, :])),
                    'source_S2': float(np.vdot(bare, source_s2@bare).real/np.vdot(bare, bare).real),
                    'G_bare': data(g_bare), 'G_anchored': data(g_anchored),
                    'source_choice_G_difference': float(abs(g_bare-g_anchored))}
            row['G_bare_add_plus_remove'] = data(gs['addition']+gs['removal'])
            row['full_CAR_source_sum'] = row['addition']['bare_source_norm2']+row['removal']['bare_source_norm2']
            rows.append(row)
    current_blocks = []
    _, _, _, transfers = electron.matter_matrices(2, parameters)
    for grid in range(ng):
        phase = np.exp(1j*2*np.pi*grid/ng/4)
        current_blocks.append(-.25j*np.sum(phase*transfers-phase.conjugate()*transfers.conj().transpose(0, 2, 1), axis=0))
    jq = sparse.block_diag(current_blocks).toarray()
    neutral_e, neutral_u, _, _ = sectors[2]
    current_chi = chi(neutral_e, neutral_u, neutral_u[:, 0], jq, jq, z)
    frozen = {}
    for n in (1, 2, 3):
        _, _, inter, transfers = electron.matter_matrices(n, parameters)
        frozen[n] = np.linalg.eigvalsh(inter+np.sum(transfers+transfers.conj().transpose(0, 2, 1), axis=0))
    return {'law_family': electron.LAW, 'state': 'recomputed compact twisted 32-grid N2 ground; partial N1/N3 spectra',
            'mu_plus': mu_plus, 'mu_minus': mu_minus, 'canonical_charge_gap': mu_plus-mu_minus,
            'midgap_mu': (mu_plus+mu_minus)/2,
            'midgap_particle_and_hole_cost': (mu_plus-mu_minus)/2,
            'frozen_addition_cost': float(frozen[3][0]-frozen[2][0]),
            'dynamical_minus_frozen_addition_cost': mu_plus-float(frozen[3][0]-frozen[2][0]),
            'neutral_same_state_flux_current_chi_partial': data(current_chi),
            'retarded_formula': 'G(z)=sum_add w/(z-(E3-E2))+sum_remove w/(z-(E2-E1)); at chemical potential mu replace z by z+mu.',
            'rows': rows,
            'scope': 'Primitive CAR/SU2 and fixed gauge assumed. Anchored source is an explicitly new reviewer observable, not silently attributed to author. No relativistic positron; no same-state cubic D.'}


def composite_probe(p):
    h, positions, spins, meta = composite.build(p)
    blocks = []; energies = []; multiplicities = []
    for mode in (0, 1):
        q = 2*np.pi*mode/p['N']
        embedding = motion.sector_embedding(p, positions, meta['ns'], meta['levels'], q)
        reduced_h = (embedding.conj().T@h@embedding).tocsr()
        values, vectors = eigsh(reduced_h, k=4, which='SA', tol=1e-11,
                               v0=np.linspace(.3, 1.1, reduced_h.shape[0]))
        order = np.argsort(values); values, vectors = values[order], vectors[:, order]
        multiplicity = int(np.sum(np.abs(values-values[0]) < 1e-8))
        block, _ = np.linalg.qr(embedding@vectors[:, :multiplicity])
        blocks.append(block); energies.append(float(values[0])); multiplicities.append(multiplicity)
    psi = blocks[0][:, 0]; q = 2*np.pi/p['N']; rho = composite.density(q, positions, meta)
    j, contact = composite.current_contact(p, q, positions, spins, meta)
    charge_matrix = blocks[1].conj().T@(rho[:, None]*blocks[0])
    current_matrix = blocks[1].conj().T@(j@blocks[0]); recoil = energies[1]-energies[0]
    spin_op = 2.25*np.eye(meta['ns'])+2*sum(meta['pair_ops'].values())
    s2 = sparse.kron(sparse.eye(len(positions)), sparse.kron(sparse.csr_matrix(spin_op), sparse.eye(meta['levels'])))
    probability = np.abs(psi.reshape(len(positions), meta['ns'], meta['levels']))**2
    intrinsic = composite.intrinsic(p, positions, probability.sum(axis=(1, 2)), meta['q'])
    measure = composite.spectral_measure(h, energies[0], j@psi, 24)
    controls = []
    for g in (p['binding_g'], 0.0):
        params = dict(p, binding_g=g)
        hg, _, _, _ = composite.build(params)
        eg = energies[0] if g else composite.ground(hg)[0]
        pair_rows = []
        for pair in itertools.combinations(range(3), 2):
            hp, _, _, _ = composite.build(params, pair)
            ep, _ = composite.ground(hp)
            pair_rows.append({'pair': list(pair), 'dressed_pair_plus_single_energy': ep-2*p['hopping'],
                              'binding_margin': ep-2*p['hopping']-eg})
        controls.append({'g': g, 'trimer_energy': eg, 'fragments': pair_rows})
    return {'candidate': p['candidate'], 'dimension': h.shape[0], 'ground_energy': energies[0],
            'actual_S2': float(np.vdot(psi, s2@psi).real), 'ground_eigen_residual': float(np.linalg.norm(h@psi-energies[0]*psi)),
            'Q': float(meta['q'].sum()), 'lab_nonzero_q_density': data(np.vdot(psi, rho*psi)),
            'intrinsic': intrinsic, 'translation_ground_block_dimensions': multiplicities,
            'own_recoil': recoil, 'whole_target_block_norm_selected_preparation': float(np.linalg.norm(charge_matrix[:, 0])),
            'both_block_density_singular_values': np.linalg.svd(charge_matrix, compute_uv=False).tolist(),
            'both_block_elastic_continuity_error': float(np.linalg.norm(recoil*charge_matrix-q*current_matrix)),
            'source_continuity_error': float(np.linalg.norm(h@(rho*psi)-rho*(h@psi)-q*(j@psi))),
            'current_24step_response': composite.retarded(measure, [0., .4, 1.1], .04),
            'current_source_norm2': measure['source_norm2'], 'current_weight_sum': float(sum(measure['weights'])),
            'native_contact': float(np.vdot(psi, contact@psi).real), 'actual_removed_coupling_controls': controls,
            'scope': 'Finite N dressed fragment energy sums, not infinite-separation thresholds; trimer is not proton, intrinsic chart not Sachs form factor; no EM field dressing in this law.'}


def run(request, output):
    result = {'kind': 'independent_real_physical_consumer', 'input': request,
              'photon': photon_probe(request), 'compact_interaction': compact_probe(request),
              'charged_flux': electron_probe(request),
              'composite': composite_probe(json.loads(Path(request['composite_request']).read_text()))}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'photon_omegas': result['photon']['positive_submatter_omegas'],
        'compact_gap': result['compact_interaction']['first_gap'], 'canonical_charge_gap': result['charged_flux']['canonical_charge_gap'],
        'composite_S2': result['composite']['actual_S2'], 'composite_recoil': result['composite']['own_recoil']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.input.read_text()), args.output)

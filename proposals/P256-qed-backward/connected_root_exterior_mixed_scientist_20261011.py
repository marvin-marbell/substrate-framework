"""Actual unequal primitive paths sharing an integer edge and matching cells."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

import connected_root_exterior_scientist_20261011 as connected

ROOT = Path(__file__).resolve().parent


def mixed_prototype(gram: np.ndarray, matrices: list[np.ndarray], hs: list[np.ndarray],
                    roles: tuple[int, int], same_copy: bool, local: np.ndarray,
                    masks: list[int], lookup: dict[int, int], orientation: int) -> dict[str, Any]:
    embed = connected.finite_embedding(masks, local, roles[0] == roles[1])
    h = np.zeros((16, 16), complex)
    rs = [np.zeros((16, 16), complex), np.zeros((16, 16), complex)]
    for cell, slot in itertools.product(range(2), repeat=2):
        sl = slice(8*cell+4*slot, 8*cell+4*slot+4)
        h[sl, sl] = hs[roles[slot]]
        rs[cell][sl, sl] = np.diag([.5, .5, -.5, -.5])
    H = connected.lift_onebody(h, masks, lookup)
    rho = [connected.lift_onebody(r, masks, lookup) for r in rs]
    def hop(path: int, slot: int | None, spin_pair: tuple[int, int] | None) -> sparse.csr_matrix:
        one = np.zeros((16, 16), complex)
        sa, sb = [(1, 0), (1, 1)][path]
        for a, b in itertools.product(range(4), repeat=2):
            s, p = divmod(a, 2)
            t, q = divmod(b, 2)
            if spin_pair is not None and (s, t) != spin_pair:
                continue
            i = 4*(s if slot is None else slot)+2*sa+p
            j = 8+4*(t if slot is None else slot)+2*sb+q
            one[i, j] = matrices[path][a, b]
        lifted = connected.lift_onebody(one, masks, lookup)
        return lifted if orientation == 1 else lifted.conjugate().T.tocsr()
    products = []
    if same_copy:
        T1, T2 = hop(0, None, None), hop(1, None, None)
        products.append((T2@T1, T1+T2))
    else:
        for spins in itertools.product(range(2), repeat=4):
            T10, T20 = hop(0, 0, (spins[0], spins[1])), hop(1, 0, (spins[0], spins[1]))
            T11, T21 = hop(0, 1, (spins[2], spins[3])), hop(1, 1, (spins[2], spins[3]))
            # These two assignments reach the SAME canonical final label set.
            # Sum before taking a norm; deleting the cross assignment is wrong.
            products.append((T21@T10+T20@T11, T10+T11+T20+T21))
    rows = []
    for O, Vplus in products:
        maps = connected.kernel_action(H, O, embed, rho[0], rho[1], 256, orientation)
        row = connected.action_moments(maps, gram)
        S = maps[0]
        d = gram.shape[0]//2
        density = gram[:d, :d]
        row['V_selected_full_two_path_exterior_square'] = float((connected.contract(Vplus@S, Vplus@S, density, density)
             +connected.contract(Vplus.conjugate().T@S, Vplus.conjugate().T@S, density, density)).real)
        rows.append(row)
    return {key: sum(r[key] for r in rows) for key in rows[0]}


def run(epoch: str) -> None:
    manifest = json.loads((ROOT/'connected_root_exterior_mixed_inputs_20261011.json').read_bytes())
    assert all(connected.sha(ROOT/name) == pin for name, pin in manifest['sha256'].items())
    first = json.loads((ROOT/'connected_root_exterior_actual1_20261011_receipt.json').read_bytes())
    receipt = json.loads((ROOT/'harmonic_car_bridge_typed_attempt3_20261011_receipt.json').read_bytes())
    boundary = json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_bytes())['all64_boundary']
    profiles = json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_bytes())['profiles']
    with np.load(ROOT/'connected_root_exterior_actual1_20261011_moments.npz', allow_pickle=False) as data:
        grams = {pair: data[key]/receipt['source_norm_float'] for pair, key in
                 [((0, 0), 'active_pair'), ((1, 1), 'auxiliary_pair'), ((0, 1), 'mixed_pair')]}
    hs = []
    for profile in profiles:
        m, c, d = [float(Fraction(profile[key])) for key in ['m_exact', 'c0_exact', 'd_exact']]
        hs.append(m*np.diag([-1, 1, -1, 1])+c*np.kron([[0, 1], [1, 0]], np.diag([1, -1]))
                  +d*np.block([[np.zeros((2, 2)), 1j*np.array([[0, 1], [1, 0]])], [-1j*np.array([[0, 1], [1, 0]]), np.zeros((2, 2))]]))
    with np.load(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz', allow_pickle=False) as data:
        phi = data['Phi_binary_section']
    had = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [-1, 0, 1, 0], [0, -1, 0, 1]], float)/math.sqrt(2)
    frame = np.einsum('ia,mij,jb->mab', had, phi, had)
    gamma = frame[1]/math.sqrt(np.trace(frame[1]@frame[1]).real/4)
    spin = np.column_stack([np.eye(4)[:, 0], gamma@np.eye(4)[:, 0], np.eye(4)[:, 1], gamma@np.eye(4)[:, 1]])
    matrices = []
    P = np.diag([1, -1, 1, -1])
    for length in [1, 2]:
        path = next(p for p in boundary['roles'][0]['paths'] if p['axis'] == 0 and p['length'] == length)
        physical = np.asarray([[complex(float(Fraction(z[0])), float(Fraction(z[1]))) for z in row] for row in path['matrix_exact']])
        matrices.append((-1 if length == 1 else 1)*P@(spin.conjugate().T@physical@spin)@P)
    local = connected.slot_embedding()
    masks = [sum(1 << j for j in inds) for inds in itertools.combinations(range(16), 8)]
    lookup = {mask: i for i, mask in enumerate(masks)}
    counts = {'same_active': 1, 'same_auxiliary': 63, 'different_active_auxiliary': 63, 'different_auxiliary_auxiliary': 1953}
    results = []
    for orientation in [1, -1]:
        rows = {}
        for name, pair, same_copy in [('same_active', (0, 0), True), ('same_auxiliary', (1, 1), True),
                                     ('different_active_auxiliary', (0, 1), False), ('different_auxiliary_auxiliary', (1, 1), False)]:
            rows[name] = mixed_prototype(grams[pair], matrices, hs, pair, same_copy, local, masks, lookup, orientation)
        full = {key: sum(counts[n]*rows[n][key] for n in rows) for key in rows['same_active']}
        overlap = 0j
        for r, count in enumerate([1, 63]):
            role = ['active', 'auxiliary'][r]
            left = next(s for s in receipt['source_roles'] if s['role'] == role and s['orientation'] == ('creation' if orientation == 1 else 'annihilation'))
            right = next(s for s in receipt['source_roles'] if s['role'] == role and s['orientation'] == ('annihilation' if orientation == 1 else 'creation'))
            A = connected.decoded(left['endpoint_reduced_matrices']['0.5']['G'])[2:4, 2:4]/receipt['source_norm_float']
            B = connected.decoded(right['endpoint_reduced_matrices']['0.5']['G'])[0:2, 2:4]/receipt['source_norm_float']
            M1, M2 = matrices if orientation == 1 else [M.conjugate() for M in matrices]
            overlap += count*np.einsum('ij,kl,ik,jl->', M1.conjugate(), M2, np.kron(np.eye(2), A), np.kron(np.eye(2), B))
        sign = '+' if orientation == 1 else '-'
        W1, W2 = [p['material_second_shell_graph_test']['raw_first_oriented_G'][sign]/receipt['source_norm_float']**2 for p in first['paths']]
        t1, t2 = [p['material_second_shell_graph_test']['t'] for p in first['paths']]
        z1, z2 = [p['material_second_shell_graph_test']['first_shell_Z'] for p in first['paths']]
        true = (full['G']-W1*W2-abs(overlap)**2)*t1*t1*t2*t2/(z1*z2)
        wrong = (full['G']-W1*W2)*t1*t1*t2*t2/(z1*z2)
        results.append({'orientation': orientation, 'primitive_roots': [{'axis': 0, 'length': 1, 'sites': [1, 0]}, {'axis': 0, 'length': 2, 'sites': [1, 1]}],
                        'operator': 'O=T_length2*T_length1; T each contains physical1 all64 coherent native Dirac hopping',
                        'all64_multiplicities': counts, 'unordered_copy_assignment_amplitudes_summed_before_norm': True,
                        'source_kernel_by_roles': rows, 'complete_all64_relative_G_T_U_and_selected_V': full,
                        'physical_first_source_weights': [W1, W2], 'actual_cross_path_Gram_D': connected.encoded(overlap),
                        'exact_normalized_log_cumulant': true, 'drop_cross_path_overlap_wrong_cumulant': wrong,
                        'drop_overlap_difference': wrong-true, 'drop_crosscopy_two_root_weight': rows['same_active']['G']+63*rows['same_auxiliary']['G'],
                        'full_shared_graph_norm': 1+W1+W2+2*overlap.real+full['G'],
                        'disconnected_product_graph_norm': (1+W1)*(1+W2),
                        'conditional_scope': 'Actual binary64 source contraction, algebraic all64 reduction, complete H0 moment; no phase/ground certificate.'})
        print(json.dumps({'orientation': orientation, 'Gmixed': full['G'], 'D': [overlap.real, overlap.imag], 'cumulant': true}), flush=True)
    output = {'kind': 'ACTUAL_UNEQUAL_INTEGER_PATH_SHARED_CELL_SHARED_FLOW_ALL64_CONNECTED_CONSUMER',
              'source_and_inputs_unchanged': all(connected.sha(ROOT/name) == pin for name, pin in manifest['sha256'].items()),
              'input_sha256': manifest['sha256'], 'own_source_sha256': connected.sha(Path(__file__)),
              'path_source_orthogonality_may_not_be_assumed': True, 'results': results,
              'log_identity': 'F(z,w)=1+Wa|z|^2+Wb|w|^2+D*z_conj*w+D_conj*z*w_conj+Gab|zw|^2. Mixed log coefficient=Gab-WaWb-|D|^2.',
              'source_normalization': 'Each selected matching source actual04/sqrt(N0); pair Gram divided by N0 before contracting two actual cells. No copy-factorized source.',
              'fullPhi_ground': False, 'phase': False, 'campaign_complete': False}
    assert output['source_and_inputs_unchanged']
    (ROOT/f'connected_root_exterior_{epoch}_receipt.json').write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    run(parser.parse_args().epoch)

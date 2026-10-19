"""Physical integer-flow two-root CAR contractions on the immutable 9.6M source.

No ground/root solver.  All numerical results are binary64 finite observations;
the all-order tree bound is conditional on explicitly listed graph hypotheses.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import sparse

import harmonic_car_bridge_scientist_20261011 as native

ROOT = Path(__file__).resolve().parent
NAMES = ['active', 'auxiliary']


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded(value: Any) -> np.ndarray:
    a = np.asarray(value, float)
    return a[..., 0] + 1j*a[..., 1]


def encoded(value: Any) -> Any:
    a = np.asarray(value)
    return np.stack([a.real, a.imag], axis=-1).tolist()


def selected_pair_gram(state: np.ndarray, rho: np.ndarray, roles: tuple[int, int], epoch: str) -> np.ndarray:
    """Actual two-labelled-slot extraction, not a product of one-slot densities.

    Same-role coordinates use the orthonormal Sym^2(W5) basis (dimension15).
    Unlike-role coordinates use W5_active tensor W5_aux (dimension25).
    Fields are psi and TOTAL physical rho psi; hence one shared E64 field.
    """
    a, b = roles
    ns = [2, 126]
    first = native.extraction(ns[a])
    second = native.extraction(ns[b]-int(a == b))
    pairs = [(i, j) for i in range(5) for j in range(5) if a != b or i <= j]
    size = native.dim(ns[a]-2)*native.dim(ns[1-a]) if a == b else native.dim(1)*native.dim(125)
    paths = []
    for field, original in enumerate([state, rho*state]):
        for i in range(5):
            once = first[i]@original if a == 0 else (first[i]@original.T).T
            for j in range(5):
                if a == b and i > j:
                    continue
                axis = b
                twice = second[j]@once if axis == 0 else (second[j]@once.T).T
                if a == b and i != j:
                    twice *= math.sqrt(2)
                path = ROOT/f'connected_root_exterior_{epoch}_scratch_{a}{b}_{field}_{i}{j}.npy'
                out = np.lib.format.open_memmap(path, mode='w+', dtype=float, shape=(size,))
                out[:] = twice.reshape(-1)
                out.flush()
                del out, twice
                paths.append(path)
            del once
    views = [np.load(p, mmap_mode='r') for p in paths]
    gram = np.zeros((2*len(pairs), 2*len(pairs)))
    for start in range(0, size, 32768):
        block = np.stack([v[start:start+32768] for v in views])
        gram += block@block.T
    del views
    for path in paths:
        path.unlink()
    assert abs(np.trace(gram[:len(pairs), :len(pairs)])-1) < 2e-9
    return gram


def apply_car(mask: int, mode: int, creation: bool) -> tuple[int, int] | None:
    occupied = bool(mask & (1 << mode))
    if occupied == creation:
        return None
    sign = (-1)**((mask & ((1 << mode)-1)).bit_count())
    return mask ^ (1 << mode), sign


def lift_onebody(h: np.ndarray, masks: list[int], lookup: dict[int, int]) -> sparse.csr_matrix:
    rr, cc, vv = [], [], []
    nz = list(zip(*np.nonzero(h)))
    for column, mask in enumerate(masks):
        for p, q in nz:
            right = apply_car(mask, int(q), False)
            if right is None:
                continue
            left = apply_car(right[0], int(p), True)
            if left is not None:
                rr.append(lookup[left[0]])
                cc.append(column)
                vv.append(left[1]*right[1]*h[p, q])
    return sparse.csr_matrix((vv, (rr, cc)), shape=(len(masks), len(masks)), dtype=complex)


def slot_embedding() -> np.ndarray:
    # Pinned all-four-generator native joint embedding and slot gauge.
    receipt = json.loads((ROOT/'harmonic_car_bridge_typed_attempt3_20261011_receipt.json').read_bytes())
    import sympy
    J = np.asarray([[complex(sympy.sympify(z)) for z in row] for row in receipt['controls']['joint_n1_native_embedding']], complex)
    V = np.zeros((16, 5), complex)
    for j, mask in enumerate([5, 3, 12, 10]):
        V[mask, j] = 1
    V[9, 4] = V[6, 4] = 1/math.sqrt(2)
    return V@J@np.diag([1, -1j, 1, 1, 1j])


def pair_basis(same: bool) -> np.ndarray:
    if not same:
        return np.eye(25)
    columns = []
    for i in range(5):
        for j in range(i, 5):
            column = np.zeros(25)
            column[5*i+j] = 1 if i == j else 1/math.sqrt(2)
            if i != j:
                column[5*j+i] = 1/math.sqrt(2)
            columns.append(column)
    return np.column_stack(columns)


def finite_embedding(masks: list[int], local: np.ndarray, same: bool) -> sparse.csr_matrix:
    left = sparse.csr_matrix(np.kron(local, local)@pair_basis(same))
    both = sparse.kron(left, left, format='csr')
    # scipy kron indexes the first slot as the high bits; native CAR uses low bits.
    full_indices = [((m & 15) << 12) | (((m >> 4) & 15) << 8) | (((m >> 8) & 15) << 4) | ((m >> 12) & 15) for m in masks]
    return cast(sparse.csr_matrix, both[full_indices].tocsr())


def contract(left: sparse.csr_matrix, right: sparse.csr_matrix, ga: np.ndarray, gb: np.ndarray) -> complex:
    # Source coefficient covariance is <y_i|y_j>.  The density matrix is its transpose.
    gram = (left.conjugate().T@right).toarray()
    density = np.kron(ga, gb)
    return complex(np.sum(gram*density))


def kernel_action(H: sparse.csr_matrix, O: sparse.csr_matrix,
                  embed: sparse.csr_matrix, rho_left: sparse.csr_matrix, rho_right: sparse.csr_matrix,
                  external: int, orientation: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    S = O@embed
    comm = H@S-O@(H@embed)
    one, two = [], []
    # The RIGHT tree edge carries the accumulated LEFT charge, not q_right.
    # E=-cumsum(q_site) gives both matching offsets u=q_left/2.
    for R, u in zip([rho_left, rho_right], [orientation, orientation]):
        C1 = R@S-O@(R@embed)
        C2 = R@(R@S)-2*R@(O@(R@embed))+O@(R@(R@embed))
        one.append(128*(C1+u*S))
        two.append(64*C2+128*u*C1+64*u*u*S)
    return S, comm+two[0]+two[1]+external*S, one[0], one[1]


def action_moments(maps: tuple[Any, Any, Any, Any], gram: np.ndarray) -> dict[str, float]:
    d = gram.shape[0]//2
    G = gram[:d, :d]
    GR = gram[:d, d:]
    RG = gram[d:, :d]
    RR = gram[d:, d:]
    S, A, L, R = maps
    w = contract(S, S, G, G).real
    t = (contract(S, A, G, G)+contract(S, L, GR, G)+contract(S, R, G, GR)).real
    u = (contract(A, A, G, G)+contract(L, L, RR, G)+contract(R, R, G, RR)
         +2*contract(A, L, GR, G).real+2*contract(A, R, G, GR).real
         +2*contract(L, R, RG, GR).real).real
    assert w >= -1e-8 and u >= -1e-5
    return {'G': float(w), 'T_relative': float(t), 'U_relative_complete_H0': float(u)}


def prototype(gram: np.ndarray, M: np.ndarray, hs: list[np.ndarray], sites: tuple[int, int],
              role_pair: tuple[int, int], same_copy: bool, local: np.ndarray,
              masks: list[int], lookup: dict[int, int], denominator: tuple[float, float], edge_length: int,
              orientation: int = 1) -> dict[str, Any]:
    same = role_pair[0] == role_pair[1]
    embed = finite_embedding(masks, local, same)
    sa, sb = sites
    h = np.zeros((16, 16), complex)
    rho = [np.zeros((16, 16), complex), np.zeros((16, 16), complex)]
    r4 = np.diag([.5, .5, -.5, -.5])
    for cell in range(2):
        for slot in range(2):
            sl = slice(8*cell+4*slot, 8*cell+4*slot+4)
            h[sl, sl] = hs[role_pair[slot]]
            rho[cell][sl, sl] = r4
    rlift = [lift_onebody(r, masks, lookup) for r in rho]
    Hlift = lift_onebody(h, masks, lookup)
    def one_hop(slot: int | None, spin_pair: tuple[int, int] | None, scale: float) -> sparse.csr_matrix:
        hop = np.zeros((16, 16), complex)
        for a, b in itertools.product(range(4), repeat=2):
            s1, p = divmod(a, 2)
            s2, q = divmod(b, 2)
            if spin_pair is not None and (s1, s2) != spin_pair:
                continue
            i = 4*(s1 if slot is None else slot)+2*sa+p
            j = 8+4*(s2 if slot is None else slot)+2*sb+q
            hop[i, j] = M[a, b]*scale
        lifted = lift_onebody(hop, masks, lookup)
        return lifted if orientation == 1 else lifted.conjugate().T.tocsr()
    rows = []
    if same_copy:
        T = one_hop(None, None, 1/denominator[0])
        products = [(T@T, T*denominator[0])]
    else:
        products = []
        for spins in itertools.product(range(2), repeat=4):
            T1 = one_hop(0, (spins[0], spins[1]), 1/denominator[0])
            T2 = one_hop(1, (spins[2], spins[3]), 1/denominator[1])
            # Different physical copies commute as EVEN global CAR words.
            assert sparse.linalg.norm(T1@T2-T2@T1) < 1e-9
            products.append((T2@T1, T1*denominator[0]+T2*denominator[1]))
    for O, T in products:
        maps = kernel_action(Hlift, O, embed, rlift[0], rlift[1], 64*4, orientation)
        row = action_moments(maps, gram)
        S = maps[0]
        d = gram.shape[0]//2
        G = gram[:d, :d]
        # Full physical hopping action into both 1-hop and 3-hop exterior flows.
        plus, minus = T@S, T.conjugate().T@S
        row['V_selected_complete_exterior_square'] = float((contract(plus, plus, G, G)+contract(minus, minus, G, G)).real)
        # Full H, not PHP squared: odd-flow V images are orthogonal to the 2-flow H0 image.
        row['full_selected_H_action_square'] = row['U_relative_complete_H0']+row['V_selected_complete_exterior_square']
        rows.append(row)
    return {key: sum(r[key] for r in rows) for key in rows[0]} | {'canonical_spin_subsectors': len(rows), 'all_contact_cross_ordering_kept': True}


def primitive_flow_controls() -> dict[str, Any]:
    # Physical Dinc is -1 tail,+1 head; T creates LEFT and annihilates RIGHT.
    incidence = np.array([[-1, 0, 0], [1, -1, 0], [0, 1, -1], [0, 0, 1]], int)
    k = np.array([0, -1, 0], int)
    l = np.array([0, -1, -1], int)
    qk, ql = incidence@k, incidence@l
    overlap = 128*int(k@l)
    assert np.array_equal(incidence@(k+l), qk+ql)
    assert 64*int((k+l)@(k+l)) != 64*int(k@k+l@l)
    masks = [sum(1 << j for j in indices) for indices in itertools.combinations(range(16), 8)]
    wrong_offset_max = 0.
    for mask in masks:
        counts = np.asarray([sum((mask >> j) & 1 for j in modes)
                             for modes in [(0, 1, 4, 5), (2, 3, 6, 7), (8, 9, 12, 13), (10, 11, 14, 15)]])
        charges = counts-2
        E = -np.cumsum(charges)[:3]
        qA = charges[0]+charges[1]
        rhoA = (counts[0]-counts[1])/2
        rhoB = (counts[2]-counts[3])/2
        assert np.array_equal(incidence@E, charges)
        assert E[0] == -(rhoA+qA/2) and E[2] == -(rhoB+qA/2)
        wrong_offset_max = max(wrong_offset_max, abs(E[2]+rhoB-qA/2))
    assert wrong_offset_max > 0
    # Actual tree/cycle source cross term survives although Euclidean Hodge spaces differ.
    return {'primitive_integer_paths': [k.tolist(), l.tolist()], 'integer_divergences': [qk.tolist(), ql.tolist()],
            'shared_flow': (k+l).tolist(), 'HE64_shared': 64*int((k+l)@(k+l)),
            'drop_overlap_wrong_value': 64*int(k@k+l@l), 'cross_HE64': overlap,
            'literal_Dinc': incidence.tolist(), 'integer_matching_offsets_by_orientation': {'+': [1, 1], '-': [-1, -1]},
            'same_offset_formula_exact_for_all_12870_fullCAR_masks': True,
            'opposite_right_offset_wrong_max_edgeflow_error': wrong_offset_max,
            'HE_HC64_once': 'integer E64 includes longitudinal Gauss contact, never add a second HC64',
            'treecycle_source_cross_coefficients_retained': {'TT': '0', 'TC': '1/4', 'CT': '1/4', 'CC': '1/2'},
            'W_colour_alignment_is_not_U1_flux_smallness': True}


def run(epoch: str, pair_density: str | None = None) -> None:
    started = time.monotonic()
    manifest = json.loads((ROOT/'connected_root_exterior_inputs_20261011.json').read_bytes())
    assert all(sha(ROOT/name) == pin for name, pin in manifest['sha256'].items())
    receipt = json.loads((ROOT/'harmonic_car_bridge_typed_attempt3_20261011_receipt.json').read_bytes())
    boundary = json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_bytes())['all64_boundary']
    profiles = json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_bytes())['profiles']
    with np.load(ROOT/'local_so5_resum_actual04_20261011_state.npz', allow_pickle=False) as data:
        state = np.asarray(data['state']).reshape(14, 690880)
    rho1 = np.asarray([p-q for _, p, q in native.labels(2)])
    rho2 = np.asarray([p-q for _, p, q in native.labels(126)])
    rho = rho1[:, None]+rho2[None, :]
    pair_grams = {}
    if pair_density is None:
        for pair in [(0, 0), (1, 1), (0, 1)]:
            pair_grams[pair] = selected_pair_gram(state, rho, pair, epoch)
            print(json.dumps({'stage': 'actual_pair_density', 'roles': pair, 'dimension': pair_grams[pair].shape[0]}), flush=True)
    else:
        expected = '4c6c6e0da22e0d9d405c37935332bb9548fbf2df5f429ec66b50aa671f8af67d'
        manifest['sha256'][pair_density] = expected
        assert sha(ROOT/pair_density) == expected
        with np.load(ROOT/pair_density, allow_pickle=False) as data:
            pair_grams = {(0, 0): data['active_pair'], (1, 1): data['auxiliary_pair'], (0, 1): data['mixed_pair']}
        print(json.dumps({'stage': 'consuming_pinned_actual_two_slot_density', 'sha256': expected}), flush=True)
    np.savez_compressed(ROOT/f'connected_root_exterior_{epoch}_moments.npz',
                        active_pair=pair_grams[0, 0], auxiliary_pair=pair_grams[1, 1], mixed_pair=pair_grams[0, 1])
    hs = []
    for profile in profiles:
        m, c, d = [float(Fraction(profile[key])) for key in ['m_exact', 'c0_exact', 'd_exact']]
        hs.append(m*np.diag([-1, 1, -1, 1])+c*np.kron([[0, 1], [1, 0]], np.diag([1, -1]))
                  +d*np.block([[np.zeros((2, 2)), 1j*np.array([[0, 1], [1, 0]])], [-1j*np.array([[0, 1], [1, 0]]), np.zeros((2, 2))]]))
    local = slot_embedding()
    masks = [sum(1 << j for j in inds) for inds in itertools.combinations(range(16), 8)]
    lookup = {m: i for i, m in enumerate(masks)}
    phi = np.load(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz', allow_pickle=False)['Phi_binary_section']
    had = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [-1, 0, 1, 0], [0, -1, 0, 1]], float)/math.sqrt(2)
    frame = np.einsum('ia,mij,jb->mab', had, phi, had)
    gamma = frame[1]/math.sqrt(np.trace(frame[1]@frame[1]).real/4)
    spin = np.column_stack([np.eye(4)[:, 0], gamma@np.eye(4)[:, 0], np.eye(4)[:, 1], gamma@np.eye(4)[:, 1]])
    # Independent computable literal X: select the measured minimum mean over sites,
    # NOT a charged lower bound, and never Hamiltonian root-allocation coefficients.
    chosen = []
    for path in receipt['paths']:
        if path['axis'] != 0:
            continue
        means, weights = [], []
        for role in path['roles']:
            oriented = [site['orientations'][1] for site in role['all_sites_both_charge_choices']]
            means.append(min(float(o['complete_charged_relative_mean']) for o in oriented))
            weights.append(max(float(o['source_weight']) for o in oriented))
        chosen.append((path['length'], means, weights))
    outputs = []
    for length, means, weights in chosen:
        primitive = next(p for p in boundary['roles'][0]['paths'] if p['axis'] == 0 and p['length'] == length)
        physical = np.asarray([[complex(float(Fraction(z[0])), float(Fraction(z[1]))) for z in row] for row in primitive['matrix_exact']])
        M = spin.conjugate().T@physical@spin
        # Matching sublattice source gauge is part of the physical endpoint map.
        P = np.diag([1, -1, 1, -1])
        M = (-1 if length == 1 else 1)*P@M@P
        rows = {}
        for name, pair, same_copy, denominator in [
            ('same_active', (0, 0), True, (means[0], means[0])),
            ('same_auxiliary', (1, 1), True, (means[1], means[1])),
            ('different_active_auxiliary', (0, 1), False, (means[0], means[1])),
            ('different_auxiliary_auxiliary', (1, 1), False, (means[1], means[1])),
            # Both charge signs use the identical shared field and source.
        ]:
            sites = (1, 0) if length == 1 else (1, 1)
            rows[name] = prototype(pair_grams[pair], M, hs, sites, pair, same_copy, local, masks, lookup, denominator, length)
            rows[name+'_reverse'] = prototype(pair_grams[pair], M, hs, sites, pair, same_copy, local, masks, lookup, denominator, length, -1)
        multiplicities = {'same_active': 1, 'same_auxiliary': 63, 'different_active_auxiliary': 252, 'different_auxiliary_auxiliary': 7812}
        forward_names = list(multiplicities)
        full = {key: sum(multiplicities[name]*rows[name][key] for name in forward_names) for key in ['G', 'T_relative', 'U_relative_complete_H0']}
        reverse = {key: sum(multiplicities[name]*rows[name+'_reverse'][key] for name in forward_names) for key in ['G', 'T_relative', 'U_relative_complete_H0']}
        # Compute first-root weight on the SAME fixed matching endpoint, not max-site data.
        w1role = []
        path = next(p for p in receipt['paths'] if p['axis'] == 0 and p['length'] == length)
        for r in range(2):
            site = next(s for s in path['roles'][r]['all_sites_both_charge_choices'] if s['sites'] == list(sites))
            w1role.append(site['orientations'][1]['source_weight']/means[r]**2)
        w1 = w1role[0]+63*w1role[1]
        w2 = full['G']
        Z = 1+w1
        connected = (w2-w1*w1)/(Z*Z)
        # Exact log-norm mixed cumulant includes TWO folds for coincident flows:
        # the disconnected diagonal fold and the one-root exchange contraction.
        normal_ordered = (w2-2*w1*w1)/(Z*Z)
        overlapping = 27 if length == 1 else 39
        first_tree_load = math.e*overlapping*abs(connected)
        wick_tree_load = math.e*overlapping*abs(normal_ordered)
        wrong_crosscopy = rows['same_active']['G']+63*rows['same_auxiliary']['G']
        product_density_error = []
        for r, pair in [(0, (0, 0)), (1, (1, 1))]:
            source_role = next(s for s in receipt['source_roles'] if s['role'] == NAMES[r] and s['orientation'] == 'annihilation')
            G1 = decoded(source_role['complete_streamed_field_slot_Gram'])[:5, :5]
            B = pair_basis(True)
            fake = B.T@np.kron(G1, G1)@B
            actual = pair_grams[pair][:15, :15]
            product_density_error.append(float(np.linalg.norm(actual-fake)))
        # Linked full-V exterior, with no disconnected spectator norm: use exact
        # [Pdag,P^2]=C P+P C and onebody commutator on each actually touched copy.
        # This paid bound is deliberately separate from the computed H0 square.
        copy_commutator = [4*float(np.linalg.norm(M, 2))**2/m for m in means]
        linked_V_paid = 2*sum(n*c for n, c in zip([1, 63], copy_commutator))*math.sqrt(w1)
        additional_commutator_paid = math.sqrt(sum(n*(8*float(np.linalg.norm(M, 2))**3/m**2)**2
                                                   for n, m in zip([1, 63], means)))
        root_norm_paid = 4*float(np.linalg.norm(M, 2))*sum(n/m for n, m in zip([1, 63], means))
        O_norm_paid = root_norm_paid**2
        raw_factors = {'same_active': means[0]**4, 'same_auxiliary': means[1]**4,
                       'different_active_auxiliary': means[0]**2*means[1]**2,
                       'different_auxiliary_auxiliary': means[1]**4}
        raw2 = {sign: {key: sum(multiplicities[n]*raw_factors[n]*rows[n+suffix][key] for n in forward_names)
                      for key in ['G', 'T_relative', 'U_relative_complete_H0']}
                for sign, suffix in [('+', ''), ('-', '_reverse')]}
        raw1 = {'+': 0., '-': 0.}
        raw_first = 0.
        for count, role in zip([1, 63], path['roles']):
            choice = next(s for s in role['all_sites_both_charge_choices'] if s['sites'] == list(sites))
            for sign, ori in zip(['-', '+'], choice['orientations']):
                raw1[sign] += count*ori['source_weight']
                raw_first += count*ori['source_weight']*ori['complete_charged_relative_mean']
        Wtotal = raw1['+']+raw1['-']
        mu = raw_first/Wtotal
        t_graph = 2/(mu+math.sqrt(mu*mu+4*Wtotal))
        u_graph = t_graph*t_graph/2
        Zfirst = 1+t_graph*t_graph*Wtotal
        Zsecond = Zfirst+u_graph*u_graph*(raw2['+']['G']+raw2['-']['G'])
        scalar_pair = t_graph**4*(raw2['+']['G']-2*raw1['+']**2)
        second_shell = {'construction': 'X2=-t(T+Tdag)Omega+u(T^2+(Tdag)^2)Omega, u=t^2/2',
                        't': t_graph, 'u': u_graph, 'first_shell_Z': Zfirst, 'second_shell_Z': Zsecond,
                        'raw_first_oriented_G': raw1, 'raw_second_oriented_G_T_U': raw2,
                        'first_graph_pair_kernel': scalar_pair/Zfirst**2,
                        'second_graph_pair_kernel': scalar_pair/Zsecond**2,
                        'first_constant_envelope_load': math.e*overlapping*abs(scalar_pair)/Zfirst**2,
                        'second_constant_envelope_load': math.e*overlapping*abs(scalar_pair)/Zsecond**2,
                        'second_shell_H0_action_norm_paid': u_graph*(math.sqrt(raw2['+']['U_relative_complete_H0'])+math.sqrt(raw2['-']['U_relative_complete_H0'])),
                        'degree2_graph_is_not_stationary_or_exact_Riccati': True,
                        'literal_additional_hypothesis': 'Uniform adjacent-root kernel upper envelope at least the tested same-edge kernel; all unseen kernels/residuals still require upper bounds.'}
        # Distinct-copy spin sectors are coherently combined before norms.  The
        # finite selected V action is a lower subset, not an all64 V norm equality.
        row = {'axis': 0, 'length': length, 'physical_hopping': 1, 'X_coefficients_by_role': [-1/m for m in means],
               'X_is_candidate_not_Riccati_solution': True, 'first_root_oriented_role_weights': w1role,
               'first_root_all64_G': w1, 'normalizer_Z': Z, 'two_root_all64_complete_H0': full,
               'matching_endpoint_sites': list(sites), 'same_edge_external_flow_E64': 256,
               'independent_external_field_wrong_E64': 128,
               'same_copy_and_distinct_copy_kernels': rows, 'all64_exact_multiplicities': multiplicities,
               'graph_Gram_covariance_kernel': connected, 'exact_normalized_log_cumulant': normal_ordered,
               'material_second_shell_graph_test': second_shell,
               'two_root_reverse_all64_complete_H0': reverse,
               'shared_composed_graph_normalizer': 1+4*w1+w2, 'disconnected_product_normalizer': Z*Z,
               'two_root_tree_test': {'spatial_overlap_count': overlapping, 'graph_covariance_e_weighted_load': first_tree_load,
                                      'graph_covariance_load_less_than_one': first_tree_load < 1,
                                      'true_log_cumulant_e_weighted_load': wick_tree_load,
                                      'true_log_cumulant_load_less_than_one': wick_tree_load < 1,
                                      'finite_pair_test_only_not_allorder_acceptance': True},
               'complete_linked_exterior_bound': {'H0_source_action_norm': math.sqrt(full['U_relative_complete_H0']),
                  'linked_full_V_norm_paid_upper': linked_V_paid,
                  'formula': 'C=[Tphysicaldag,Pcandidate]; [Vphysical,Pcandidate^2]=C Pcandidate+Pcandidate C. norm(C)<=sum_actual64 4||M||^2/mean_role. The linked payment is <=2||C||sqrt(G1)+||[C,Pcandidate]Omega||; this is not the omitted O Vphysical Omega full-action term.',
                  'additional_C_P_commutator_paid_upper': additional_commutator_paid,
                  'additional_payment_derivation': 'Canonical changed-copy sectors are orthogonal: sqrt(sum_role count*(8||M||^3/mean_role^2)^2) bounds [C,P]Omega; no unproved comparison to ||P Omega||.',
                  'complete_bound_including_additional_term': math.sqrt(full['U_relative_complete_H0'])+linked_V_paid+additional_commutator_paid,
                  'source_residual_payment': '2*source_residual*||P^2||; separate from exact-eigenvector conditional H0 formula',
                  'full_physical_single_edge_action_bound': math.sqrt(full['U_relative_complete_H0'])+linked_V_paid+additional_commutator_paid+O_norm_paid*math.sqrt(Wtotal),
                  'full_action_additional_O_VOmega_payment': O_norm_paid*math.sqrt(Wtotal),
                  'full_action_vs_linked': 'Full H O Omega retains O V Omega, bounded here on the physical all64 single-edge Hamiltonian. Disconnected remote edges are not silently deleted: only a normalized connected expectation factorization can cancel them, not this vector norm.',
                  'no_neutral_gap_used_for_higher_charge_or_external_Q': True},
               'wrongcases': {'drop_crosscopy_G': wrong_crosscopy, 'drop_crosscopy_difference': w2-wrong_crosscopy,
                              'pretend_vacuum_cancel_difference': connected,
                              'incorrect_shared_normalizer_difference': (1+4*w1+w2)-Z*Z,
                              'independent_selected_copy_density_norm_errors': product_density_error,
                              'incorrect_role_equalization_difference': abs(w1role[0]-w1role[1]),
                              'drop_overlap_E64_difference': 128*length}}
        assert abs(w2-wrong_crosscopy) > 1e-12
        assert abs(row['shared_composed_graph_normalizer']-Z*Z) > 1e-12
        assert max(product_density_error) > 1e-8
        outputs.append(row)
        print(json.dumps({'stage': 'physical_connected_pair', 'length': length, 'first_load': first_tree_load, 'normal_ordered_load': wick_tree_load}), flush=True)
    outcome = {'kind': 'NEW_ACTUAL_PHYSICAL1_ALL64_NORMALIZED_SHARED_FLOW_TWO_ROOT_COMPLETE_CAR_CONSUMER',
               'input_sha256': manifest['sha256'], 'own_source_sha256': sha(Path(__file__)),
               'stage': 'finite binary64 actual-source observations plus algebraic all64 reduction; allorder theorem conditional',
               'physical_source_dimension': state.size, 'source_norm': float(np.sum(state*state)),
               'pair_density_definition': 'R_ab=<E_i E_j psi,E_k E_l psi>; E=b/sqrt(n), second E=b/sqrt(n-1) on same role. Not R_a tensor R_b.',
               'pair_density_fields': ['psi', 'total physical rho psi'], 'paths': outputs,
               'primitive_flow_controls': primitive_flow_controls(),
               'integer_tree_matching_offsets_by_orientation': {'+': [1, 1], '-': [-1, -1]},
               'affine_error_scope': 'Old actual1 T2/U2 with opposite right offset withdrawn; G2/densities/log kernels unaffected. Pinned harmonic typed one-root source already uses same offsets and is unchanged.',
               'pair_density_dependency': pair_density,
               'literal_allorder_hypotheses': ['X is a localized physical-source graph and Z=I+XdagX is boundedly invertible (automatic lower I, upper bound still needed uniformly).',
                   'Whole-H graph residual R_X=B+DX-XA-XBdagX is paid in the same linked norm, including all higher-charge and all external integer flows.',
                   'For every connected rooted history of every order, its normalized (or exchange-resummed) cumulant admits the proposed tree-product bound using the tested pair kernel; a fourth moment alone DOES NOT prove this.',
                   'The resulting spatial Kotecky-Preiss/tree rooted sum, including all primitive paths, roles, contacts and residual activities, is <=a with uniform a>0 for every N.',
                   'Matching product is a trial state; physicalPhi stationarity/Gauss and SAME interacting ground spectral measure are separate unproved obligations.'],
               'N_uniform_candidate': 'For range<=2 and even N>=8, adjacency degrees27/39 are N-independent. Under literal allorder tree domination by a uniform adjacent-kernel envelope c and paid graph residual, e*degree*c<1 gives a convergent rooted tree majorant. The measured same-edge c is a lower floor for that uniform envelope; failure of e*degree*c<1 excludes this constant-envelope majorant only. It is NOT a measured sum over all distinct neighbors or an allorder no-go.',
               'representation_change': 'After testing first-shell normalized graph, execute the materially different physical degree2 charge-inclusive graph X2=-t(T+Tdag)Omega+(t^2/2)(T^2+(Tdag)^2)Omega; use actual all64 second-shell norms and full H0/exterior actions. Normal-ordered fourth cumulant is diagnostic, never assumed cancelled.',
               'open': ['physicalPhi/global stationary ground', 'allorder tree domination and residual payments', 'N-uniform phase', 'stiffness/IR atom/sharp photon/common cone/original SOURCE'],
               'fullPhi_ground': False, 'phase': False, 'campaign_complete': False,
               'source_and_inputs_unchanged': all(sha(ROOT/name) == pin for name, pin in manifest['sha256'].items()),
               'elapsed_seconds': time.monotonic()-started}
    assert outcome['source_and_inputs_unchanged']
    (ROOT/f'connected_root_exterior_{epoch}_receipt.json').write_text(json.dumps(outcome, indent=2)+'\n')
    print(json.dumps({'kind': outcome['kind'], 'elapsed_seconds': outcome['elapsed_seconds'], 'unchanged': outcome['source_and_inputs_unchanged']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    parser.add_argument('--pair-density')
    args = parser.parse_args()
    run(args.epoch, args.pair_density)

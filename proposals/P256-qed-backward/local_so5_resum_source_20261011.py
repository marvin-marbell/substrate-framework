"""Exact selected SO5 harmonic dimer; full-space numerical Ritz, not ground admission."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable
import zipfile

import numpy as np
from numba import njit
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, lobpcg

ROOT = Path(__file__).resolve().parent
PREFIX = 'local_so5_resum_'
PINS = {
    'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz': 'f8a0793adcbf4909402cd089471aa8c33fc74a31b66e99618f8baaab8d6dfb65',
    'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json': '5b97862fc7fca21b8797d549d7d4dfdde58b101e8c46c7a7cfb6531a669c4354',
    'joined_phi_u1_local_so5_design_control_20261010_receipt.json': '08a4de1a0e8ca18078b2e3f4407fcfa69c23902cb437a80e7655053a503efe5e',
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def car(mask: int, p: int, q: int) -> tuple[int, int] | None:
    if not mask & (1 << q):
        return None
    sign = (-1) ** (mask & ((1 << q) - 1)).bit_count()
    mask ^= 1 << q
    if mask & (1 << p):
        return None
    sign *= (-1) ** (mask & ((1 << p) - 1)).bit_count()
    return mask | (1 << p), sign


def wedge(h: np.ndarray) -> np.ndarray:
    masks = [sum(1 << p for p in pair) for pair in itertools.combinations(range(4), 2)]
    out = np.zeros((6, 6), dtype=complex)
    for j, mask in enumerate(masks):
        for p in range(4):
            for q in range(4):
                transition = car(mask, p, q)
                if transition is not None:
                    out[masks.index(transition[0]), j] += transition[1] * h[p, q]
    return out


def native_controls() -> tuple[list[np.ndarray], dict[str, Any]]:
    z = np.diag([-1., 1.])
    x = np.array([[0., 1.], [1., 0.]])
    a = np.diag([-1., 1., -1., 1.])
    b = np.kron(x, -z)
    c = np.block([[np.zeros((2, 2)), 1j*x], [-1j*x, np.zeros((2, 2))]])
    r = np.diag([.5, .5, -.5, -.5])
    masks = [sum(1 << p for p in pair) for pair in itertools.combinations(range(4), 2)]
    v = np.zeros((6, 5))
    for j, mask in enumerate([5, 3, 12, 10]):
        v[masks.index(mask), j] = 1
    for mask in [9, 6]:
        v[masks.index(mask), 4] = 1 / np.sqrt(2)
    ops6 = [wedge(o) for o in [a, b, c, r]]
    ops5 = [v.T @ o @ v for o in ops6]
    errors = [float(np.linalg.norm(o @ v - v @ q)) for o, q in zip(ops6, ops5)]
    # Independent full16 CAR checks, including all local occupation sectors.
    annih = []
    for p in range(4):
        op = np.zeros((16, 16))
        for mask in range(16):
            if mask & (1 << p):
                op[mask ^ (1 << p), mask] = (-1)**(mask & ((1 << p)-1)).bit_count()
        annih.append(op)
    car_error = max(float(np.linalg.norm(annih[p] @ annih[q].T + annih[q].T @ annih[p] - (np.eye(16) if p == q else 0))) for p in range(4) for q in range(4))
    assert max(errors) < 1e-12 and car_error == 0
    return ops5, {'canonical_wedge_invariance_errors': errors, 'full16_CAR_error': car_error, 'number_sector_dimension': 6, 'vector_dimension': 5, 'inert_singlet_dimension': 1}


def representation(n: int, m: float, c0: float, ci: float, real: bool = True) -> tuple[sparse.csr_matrix, np.ndarray, list[tuple[int, int, int]]]:
    labels = [(l, p, q) for l in range(n+1) for p in range(l+1) for q in range(l+1)]
    dim = len(labels)
    assert dim == (n+1)*(n+2)*(2*n+3)//6
    lookup = {label: i for i, label in enumerate(labels)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    rho = np.empty(dim)
    for j, (l, p, q) in enumerate(labels):
        rho[j] = p-q
        rows.append(j); cols.append(j); data.append(2*m*(p+q-l))
        for axis in [0, 1]:
            k = p if axis == 0 else q
            if k < l:
                dest = (l, p+1, q) if axis == 0 else (l, p, q+1)
                i = lookup[dest]
                value = ci*np.sqrt((l-k)*(k+1)) if real else 1j*ci*np.sqrt((l-k)*(k+1))
                rows.extend([i, j]); cols.extend([j, i]); data.extend([value, np.conjugate(value)])
        if l < n:
            f = np.sqrt((n-l)*(n+l+3)*(l+1)/(2*(l+2)))
            for dp, dq, weight in [(1, 0, np.sqrt(2)), (0, 1, -np.sqrt(2))]:
                # CG(j,m;1/2,+/-1/2 | j+1/2,m+/-1/2).
                left = np.sqrt((p+1 if dp else l-p+1)/(l+1))
                right = np.sqrt((q+1 if dq else l-q+1)/(l+1))
                i = lookup[(l+1, p+dp, q+dq)]
                value = c0*f*left*right*weight
                rows.extend([i, j]); cols.extend([j, i]); data.extend([value, value])
    dtype = float if real else complex
    values = np.real(data) if real else data
    matrix = sparse.csr_matrix(sparse.coo_matrix((np.asarray(values, dtype=dtype), (rows, cols)), shape=(dim, dim)))
    matrix.eliminate_zeros()
    return matrix, rho, labels


def tensor_sum(o: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((5**n, 5**n), dtype=complex)
    for k in range(n):
        out += np.kron(np.kron(np.eye(5**k), o), np.eye(5**(n-k-1)))
    return out


def certify_small(ops: list[np.ndarray], n: int) -> dict[str, Any]:
    native = [tensor_sum(o, n) for o in ops]
    a, rho, labels = representation(n, 1., 0., 0., False)
    b = representation(n, 0., 1., 0., False)[0]
    c = representation(n, 0., 0., 1., False)[0]
    candidate = [a.toarray(), b.toarray(), c.toarray(), np.diag(rho)]
    dim = len(labels)
    v = np.zeros(5**n, dtype=complex); v[0] = 1
    w = np.zeros(dim, dtype=complex); w[labels.index((n, 0, 0))] = 1
    vs = [v]; ws = [w]
    discrepancy = 0.
    cursor = 0
    while cursor < len(vs):
        for t, h in zip(native, candidate):
            x = t @ vs[cursor]; y = h @ ws[cursor]
            # Twice reorthogonalized, same coefficients on both constructions.
            for _ in range(2):
                for u, z in zip(vs, ws):
                    coefficient = np.vdot(u, x)
                    x -= coefficient*u; y -= coefficient*z
            norm = float(np.linalg.norm(x))
            discrepancy = max(discrepancy, abs(norm-float(np.linalg.norm(y))))
            if norm > 1e-9:
                vs.append(x/norm); ws.append(y/norm)
                assert len(vs) <= dim, 'candidate is not complete cyclic representation'
        cursor += 1
    vmat = np.column_stack(vs); wmat = np.column_stack(ws)
    embedding = vmat @ wmat.conj().T
    intertwining = [float(np.linalg.norm(t @ embedding-embedding @ h)) for t, h in zip(native, candidate)]
    gram = float(np.linalg.norm(embedding.conj().T @ embedding-np.eye(dim)))
    assert len(vs) == dim and discrepancy < 1e-8 and max(intertwining) < 1e-8 and gram < 1e-8
    # SO5 generators constructed independently in a true real-vector frame.
    # The antiunitary real structure G below is solved by invariant bilinear form.
    g = np.zeros((5, 5)); g[0, 3] = g[3, 0] = -1; g[1, 2] = g[2, 1] = 1; g[4, 4] = 1
    vals, basis = eigh(g)
    d = basis.astype(complex) @ np.diag(np.where(vals > 0, 1.+0j, 1j))
    # d d^T=G, so O_real=d^dagger O d is purely imaginary antisymmetric.
    reality = max(float(np.linalg.norm((d.conj().T @ o @ d).real)) for o in ops)
    assert reality < 1e-12
    casimir = np.zeros((5**n, 5**n), dtype=complex)
    for i in range(5):
        for j in range(i+1, 5):
            generator = np.zeros((5, 5), dtype=complex)
            generator[i, j] = -1j; generator[j, i] = 1j
            total = tensor_sum(d @ generator @ d.conj().T, n)
            casimir += total @ total
    cas_error = float(np.linalg.norm(casimir @ embedding-n*(n+3)*embedding))
    assert cas_error < 1e-8
    # Wrong forward factor, omitted two replicas, and traceful symmetric sector controls.
    wrong = candidate[1]*np.sqrt(2)
    wrong_error = float(np.linalg.norm(native[1] @ embedding-embedding @ wrong))
    full_cas = np.linalg.eigvalsh(casimir)
    assert wrong_error > 1
    if n > 1:
        assert np.min(full_cas) < n*(n+3)-1
        trace_source = g.reshape(-1) if n == 2 else np.einsum('ij,k->ijk', g, np.eye(5)[0]).reshape(-1)
        if n == 3:
            tensor = trace_source.reshape(5, 5, 5)
            trace_source = (tensor+tensor.transpose(0, 2, 1)+tensor.transpose(2, 1, 0)).reshape(-1)
        trace_source /= np.linalg.norm(trace_source)
        assert np.linalg.norm(casimir @ trace_source-(n-2)*(n+1)*trace_source) < 1e-8
    phase = np.array([1j**(p+q-l) for l, p, q in labels])
    complex_h = representation(n, 1.2, -4.1, -5.2, False)[0].toarray()
    transformed = phase.conj()[:, None]*complex_h*phase[None, :]
    real_h = representation(n, 1.2, -4.1, -5.2, True)[0].toarray()
    gauge_error = float(np.linalg.norm(transformed-real_h))
    assert gauge_error < 1e-12
    return {'n': n, 'cyclic_dimension': len(vs), 'tensor_dimension': 5**n, 'intertwining_errors': intertwining, 'normalization_error': gram, 'CG_action_norm_error': discrepancy, 'Casimir': n*(n+3), 'Casimir_error': cas_error, 'wrong_factor_rejected_error': wrong_error, 'trace_rep_rejected_min_Casimir': float(full_cas[0]), 'complex_to_real_error': gauge_error, 'real_vector_frame_error': reality}


@njit(cache=False)
def full_action(x: np.ndarray, diagonal: np.ndarray, ap: np.ndarray, aj: np.ndarray, av: np.ndarray, bp: np.ndarray, bj: np.ndarray, bv: np.ndarray) -> np.ndarray:
    na, nb = diagonal.shape
    result = np.empty(na*nb)
    for a in range(na):
        offset = a*nb
        for b in range(nb):
            value = diagonal[a, b]*x[offset+b]
            for k in range(ap[a], ap[a+1]):
                value += av[k]*x[aj[k]*nb+b]
            for k in range(bp[b], bp[b+1]):
                value += bv[k]*x[offset+bj[k]]
            result[offset+b] = value
    return result


@njit(cache=False)
def density_moments(state: np.ndarray, rho: np.ndarray) -> tuple[float, float, float]:
    mean = 0.; second = 0.; zero = 0.
    charge = rho.reshape(-1)
    for i in range(state.size):
        probability = state[i]*state[i]
        mean += probability*charge[i]
        second += probability*charge[i]*charge[i]
        if charge[i] == 0:
            zero += probability
    return mean, second, zero


def casimir_recurrence(n: int) -> dict[str, Any]:
    # Four equivalent coset directions, B=2 times a unit SO5 generator.
    # Averaged forward CG norm=(l+2)^2/(l+1)^2; reverse norm=f_(l-1)^2.
    prior = Fraction(0)
    for l in range(n+1):
        forward = Fraction((n-l)*(n+l+3)*(l+1), 2*(l+2))
        actual = forward*Fraction((l+2)**2, (l+1)**2)+prior
        assert actual == n*(n+3)-l*(l+2)
        prior = forward
    return {'n': n, 'exact_rational_Casimir_recurrence': True, 'terminal_forward_zero': prior == 0}


class RealOperator(LinearOperator):
    def __init__(self, dimension: int, action: Callable[[np.ndarray], np.ndarray]):
        self.action = action
        super().__init__(dtype=np.dtype(np.float64), shape=(dimension, dimension))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self.action(x)


class Block:
    def __init__(self, a: sparse.csr_matrix, b: sparse.csr_matrix, ra: np.ndarray, rb: np.ndarray, u: float):
        self.rho = ra[:, None]+rb[None, :]
        self.diagonal = a.diagonal()[:, None]+b.diagonal()[None, :]+64*(self.rho+u)**2
        self.a = a.copy(); self.a.setdiag(0); self.a.eliminate_zeros()
        self.b = b.copy(); self.b.setdiag(0); self.b.eliminate_zeros()
        self.dimension = self.diagonal.size
        self.calls = 0

    def action(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        return full_action(np.asarray(x).reshape(-1), self.diagonal, self.a.indptr, self.a.indices, self.a.data, self.b.indptr, self.b.indices, self.b.data)

    def operator(self) -> LinearOperator:
        return RealOperator(self.dimension, self.action)


def ritz(block: Block, start: np.ndarray, iterations: int, tolerance: float, lower: float) -> tuple[np.ndarray, dict[str, Any]]:
    t = time.monotonic()
    denom = np.maximum(block.diagonal.reshape(-1)-lower, 1.)
    preconditioner = RealOperator(block.dimension, lambda x: np.asarray(x).reshape(-1)/denom)
    vals, vectors, history = lobpcg(block.operator(), start.reshape(-1, 1), M=preconditioner, largest=False, tol=tolerance, maxiter=iterations, retResidualNormsHistory=True)
    state = vectors[:, 0]
    state /= np.linalg.norm(state)
    action = block.action(state)
    energy = float(state @ action)
    action -= energy*state
    residual = float(np.linalg.norm(action))
    norm = float(state @ state)
    mean, second, zero = density_moments(state, block.rho)
    stats = {'energy': energy, 'solver_eigenvalue': float(vals[0]), 'normalization': norm, 'complete_H_residual': residual, 'complete_H_variance': residual**2, 'rho_mean': mean, 'rho2': second, 'whole_zero_block_probability': zero, 'full_basis_dimension': block.dimension, 'matvecs': block.calls, 'iteration_history_count': len(history), 'residual_history': [float(np.asarray(v).reshape(-1)[0]) for v in history], 'seconds': time.monotonic()-t, 'stage': 'NUMERICAL_RITZ_NOT_CERTIFIED_MINIMUM_OR_GAP', 'finite_float': True}
    print(json.dumps({k: v for k, v in stats.items() if k != 'residual_history'}), flush=True)
    return state, stats


def capture(name: str, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(ROOT/name, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, raw in sorted(members.items()):
            archive.writestr(path, raw)


def run(args: argparse.Namespace) -> None:
    began = time.monotonic()
    epoch = args.epoch
    base = PREFIX+epoch+'_20261011'
    own = Path(__file__).read_bytes()
    members = {Path(__file__).name: own}
    for name, expected in PINS.items():
        raw = (ROOT/name).read_bytes(); assert digest(raw) == expected, name
        members[name] = raw
    import bulk_electric_phase_consumer as bound
    members.update(bound.CAPTURED)
    import local_so5_resum_allocation_20261011 as linked
    members[Path(linked.__file__).name] = Path(linked.__file__).read_bytes()
    members['local_so5_resum_exterior_20261011.py'] = (ROOT/'local_so5_resum_exterior_20261011.py').read_bytes()
    if args.state_input:
        raw = (ROOT/args.state_input).read_bytes()
        assert digest(raw) == args.state_sha256
        members[args.state_input] = raw
    frame, scale = bound.frozen.exact_frame(members[bound.INPUTS[0]])
    cfg = json.loads(members[bound.INPUTS[1]])['input_descriptor']['original_micro_config_UNCHANGED']
    profiles = bound.frozen.profiles(cfg, frame, scale)
    receipt = json.loads(members['joined_phi_u1_local_so5_design_control_20261010_receipt.json'])
    native = receipt['Cl4_spin_replicas']
    cx = lambda pair: complex(float(pair[0]), float(pair[1]))
    f = [np.array([[cx(z) for z in row] for row in mat]) for mat in frame]
    gamma = f[1]/np.sqrt(np.trace(f[1] @ f[1]).real/4)
    spin = np.column_stack([np.eye(4)[:, 0], gamma @ np.eye(4)[:, 0], np.eye(4)[:, 1], gamma @ np.eye(4)[:, 1]])
    assert np.linalg.norm(spin.conj().T @ spin-np.eye(4)) < 1e-12
    parameters = []
    for p, expected in zip(profiles, native):
        path = next(z for z in p['paths'] if z['axis'] == 0 and z['length'] == 1)
        hop = np.array([[cx(z) for z in row] for row in path['matrix']])
        t = spin.conj().T @ hop @ spin
        c0 = -float(t[0, 0].real); ci = float((t[0, 1]/1j).real)
        assert c0 == expected['c0_native'] and ci == expected['ci_native']
        assert str(p['onsite']) == expected['onsite_exact'] and p['count'] == expected['count']
        parameters.append({'role': p['role'], 'count': p['count'], 'spin_replicas': 2*p['count'], 'onsite_exact': str(p['onsite']), 'onsite_float': float(p['onsite']), 'c0': c0, 'ci': ci})
    capture(base+'_preexecution.zip', members)
    controls_ops, controls = native_controls()
    controls['higher_harmonic'] = [certify_small(controls_ops, n) for n in [1, 2, 3]]
    controls['exact_Casimir_CG_recurrence'] = [casimir_recurrence(n) for n in [1, 2, 3, 126]]
    controls['exact_linked_support_allocation'] = [linked.count(n) for n in [4, 6, 8]]
    (ROOT/(base+'_controls.json')).write_text(json.dumps(controls, indent=2)+'\n')
    print(json.dumps({'controls': controls}), flush=True)
    ma, mb = [p['onsite_float'] for p in parameters]
    c0, ci = parameters[0]['c0'], parameters[0]['ci']
    active, ra, la = representation(2, ma, c0, ci)
    auxiliary, rb, lb = representation(126, mb, c0, ci)
    assert active.shape == (14, 14) and auxiliary.shape == (690880, 690880)
    block = Block(active, auxiliary, ra, rb, 0.)
    source = np.zeros(block.dimension)
    source[la.index((2, 0, 0))*len(lb)+lb.index((126, 0, 0))] = 1
    source_action = block.action(source)
    source_energy = float(source @ source_action)
    source_action -= source_energy*source
    source_variance = float(source_action @ source_action)
    assert abs(source_energy-(-4*ma-252*mb)) < 1e-9
    assert abs(source_variance-256*ci**2) < 1e-8
    controls['full128_source'] = {'energy': source_energy, 'variance': source_variance, 'independent_native_tensor_variance': 256*ci**2, 'wrong_flavour_as_one_variance': 128*ci**2, 'wrong_fold_rejected_factor': 2, 'active_auxiliary_mass_difference': mb-ma}
    source_action = None
    # Exact free native5 minimum gives a global lower bound after dropping the
    # nonnegative electric square. Used ONLY as a positive preconditioner shift,
    # not a certificate of H-E>=0, stationary ground, or an excitation gap.
    a5, b5, c5, _ = controls_ops
    lower = 2*np.linalg.eigvalsh(ma*a5+c0*b5+ci*c5)[0]+126*np.linalg.eigvalsh(mb*a5+c0*b5+ci*c5)[0]-1
    if args.state_input:
        with np.load(ROOT/args.state_input, allow_pickle=False) as checkpoint:
            source = checkpoint['state']
        assert source.shape == (block.dimension,) and abs(float(source @ source)-1) < 1e-12
    state, zero = ritz(block, source, args.iterations, args.tolerance, lower)
    source = None
    delta = args.delta
    plus_block = Block(active, auxiliary, ra, rb, delta)
    plus, plus_stats = ritz(plus_block, state, args.response_iterations, args.tolerance, lower)
    minus_block = Block(active, auxiliary, ra, rb, -delta)
    minus, minus_stats = ritz(minus_block, state, args.response_iterations, args.tolerance, lower)
    curvature = (plus_stats['energy']+minus_stats['energy']-2*zero['energy'])/delta**2
    rho_state = block.rho.reshape(-1)*state
    connected_norm = float(rho_state @ rho_state-(state @ rho_state)**2)
    # Normalized linked product on a disjoint one-link matching is well-defined;
    # every omitted bond is assigned once, with exact incidence/support counters.
    # Frozen one-link solve has NOT supplied their canonical or field action.
    degree = 6
    matching_sites = 2
    external_one_links = matching_sites*(degree-1)
    two_links_touching = 18
    sum_onsite = 2*ma+126*mb
    hopping_triangle = 2*(2+126)*(abs(c0)+abs(ci))
    bare_gap_proxy = min(2*ma, 2*mb)  # label as proxy, NEVER actual gap
    allocation = {'geometry': 'cubic nearest-neighbour perfect matching: one selected edge per pair', 'sites_per_block': 2, 'selected_one_link_edges_per_block': 1, 'external_one_link_incidence_per_block': external_one_links, 'external_edge_allocation_weight': .5, 'allocated_external_edges_per_block': external_one_links*.5, 'length2_directed_support_incidence_before_overlap_removal': two_links_touching, 'one_link_per_edge_all64_operator_triangle': hopping_triangle, 'charged_onsite_reference_magnitude': 2*sum_onsite, 'optimistic_bare_gap_proxy_NOT_certificate': bare_gap_proxy, 'already_one_link_boundary_ratio_even_with_proxy': external_one_links*.5*hopping_triangle/bare_gap_proxy, 'normalized_matching_product': 'each finite block state has norm1; tensor product norm1 exactly on disjoint matter factors, frozen conditional beta gauge frame only', 'shared_electric_and_Gauss_overlap_paid': False, 'two_link_overlap_counter_paid': False, 'fullPhi_colour_derivatives_paid': False, 'linked_stability_at_hopping1': 'FAILED_PROPOSED_OPERATOR_NORM_BOUND: even optimistic bare onsite gap proxy cannot dominate assigned omitted one-link norm bound; actual ground gap is uncertified and shared electric/two-link/field boundary terms are unpaid', 'not_no_go': True}
    allocation['length2_distinct_touching_paths'] = 16
    allocation['allocated_length2_boundary_paths_per_block'] = 6
    allocation['two_link_overlap_counter_paid'] = True
    allocation['whole_operator_vs_source_weighted'] = 'whole-operator triangle criterion ONLY; source/resolvent connected/rooted or gapless Coulomb stability not rejected'
    allocation['full_two_link_electric_field_cost_condition'] = '5*K_one_link+6*K_two_link+K_shared_electric+K_fullPhi_colour < certified_block_gap; additional costs are nonnegative/unpaid, not set to zero'
    allocation['ground_conditional_response'] = 'Full numerical incoming-u Ritz branch is actual; ground-positive interpretation requires a still-absent ground/exterior certificate.'
    assert allocation['already_one_link_boundary_ratio_even_with_proxy'] > 1
    arrays = ROOT/(base+'_state.npz')
    np.savez_compressed(arrays, state=state, plus_state=plus, minus_state=minus, incoming_rho_state=rho_state, rho_active=ra, rho_auxiliary=rb, active_labels=np.asarray(la, dtype=np.int16), auxiliary_labels=np.asarray(lb, dtype=np.int16))
    response = {'delta': delta, 'u_plus': plus_stats, 'u_minus': minus_stats, 'incoming_u_finite_difference_Ritz_branch_curvature': curvature, 'rho_connected_weight': connected_norm, 'electric_energy_constant': '64u^2 restored exactly once in every full diagonal', 'whole_zero_block_included': True, 'signed_response_not_ground_positive_spectral_measure': True, 'curvature_error_scale_from_residuals_NOT_energy_error_bound': (zero['complete_H_residual']*2+plus_stats['complete_H_residual']+minus_stats['complete_H_residual'])/delta**2, 'not_bulk_stiffness_or_photon': True}
    pins = {name: digest(raw) for name, raw in members.items()}
    unchanged = {name: (ROOT/name).exists() and digest((ROOT/name).read_bytes()) == sha for name, sha in pins.items()}
    assert all(unchanged.values())
    capture(base+'_executed.zip', members)
    output = {'kind': 'FULL128_CANONICAL_REPLICA_SELECTED_HARMONIC_DIMER_PHYSICAL1_NUMERICAL_RITZ', 'epoch': epoch, 'command': [sys.executable, *sys.argv], 'parameters': parameters, 'controls': controls, 'physical_hopping': 1, 'full128_spin_replicas': True, 'active_dimension': 14, 'auxiliary_dimension': 690880, 'paid_full_tensor_dimension': 9672320, 'active_sparse_nnz': active.nnz, 'auxiliary_sparse_nnz': auxiliary.nnz, 'complete_H': 'Hact+Haux+64(rhoact+rhoaux+u)^2; no projected square/no angular truncation', 'zero_u': zero, 'incoming_response': response, 'normalized_linked_attempt': allocation, 'source_and_input_sha256': pins, 'source_and_inputs_unchanged': unchanged, 'npz': arrays.name, 'npz_sha256': digest(arrays.read_bytes()), 'preexecution_capsule': base+'_preexecution.zip', 'executed_capsule': base+'_executed.zip', 'runtime_seconds': time.monotonic()-began, 'selected_sector_minimum_certified': False, 'spectral_gap_certified': False, 'other_spin_number_representation_sectors_paid': False, 'fullPhi_colour_lattice_beta_ground_paid': False, 'stationary_ground_admitted': False, 'campaign_complete': False, 'next_physical_decision': 'Use actual local full-state dressing as conditional ingredient only; operator-norm linked bound fails at physical1. Replace unmatched-bond norm control by explicit connected matrix-element/excitation allocation or larger gauge-compatible clusters; do not claim phase or ground-positive response.'}
    (ROOT/(base+'_receipt.json')).write_text(json.dumps(output, indent=2)+'\n')
    proof = {'kind': 'INVARIANT_REPRESENTATION_PROOF_WITH_FINITE_FLOAT_CONSUMER_NOT_ADMISSION', 'source_sha256': digest(own), 'receipt': base+'_receipt.json', 'derivation': {'canonical': 'Each native Cl2 sector is canonical wedge2(C4), not a two-level flavour surrogate. CAR dGamma splits wedge2 into invariant1+vector5; the actual ac source belongs to5. There are active2 and auxiliary126 spin replicas, same hopping but distinct exact onsite fractions.', 'harmonic': 'Permutation-symmetric n-fold tensor vector5 has invariant trace decomposition Sym^n(5)=direct sum harmonic(n-2k). The ac vector is isotropic for the invariant bilinear G, so ac^tensor n is tracefree and lies wholly in highest harmonic n. Collective SO5 Lie action preserves this exact irreducible sector. Dimension sum_l(l+1)^2=(n+1)(n+2)(2n+3)/6.', 'branching': 'SO4 labels l=0..n, jL=jR=l/2. A=2(JLz+JRz), rho=JLz-JRz, C=-2(JLy+JRy). Wigner-Eckart plus vector CG gives B_(l+1,l)=sqrt((n-l)(n+l+3)(l+1)/(2(l+2))) CG_L CG_R times weights sqrt2 for (+,-), -sqrt2 for (-,+). Native tensor intertwining and independent full SO5 Casimir n(n+3) are exercised at n1,n2,n3. These controls are numerical identities in this execution, not a symbolic certificate for arbitrary n; the analytic harmonic recurrence is the representation argument.', 'real_basis': 'Diagonal unitary D(l,p,q)=i^(p+q-l). D^dagger H D has real symmetric ci ladder entries ci sqrt((l-k)(k+1)); B leaves p+q-l invariant. No Hermitian-part projection or dropping imaginary operators. Independently invariant real vector frame uses Takagi factor of real symmetric invariant G; Casimir uses all10 exact SO5 generators.', 'complete_action': 'Every one of14*690880 entries is allocated. Full onsite+allCG edges+64(rhoact+rhoaux+u)^2 are used directly. Residual vector Hpsi-Epsi gives complete variance; no PHP square/exterior debt within this selected sector.', 'ground_limit': 'Sparse variational solver is not an exterior/ground certificate. Positivity of H-E is not presumed; local signed Ritz branch response cannot certify positive same-ground measure.', 'linked_failure': allocation}, 'scope': 'Frozen two-site one-edge tree selected canonical spin/harmonic sector only; not ambient N4 inverse Green, bulk dielectric stiffness, photon, fullPhi or campaign ground.', 'input_sha256': pins, 'admission': 'PENDING_PARENT_REVIEW'}
    proof['derivation']['exact_reduced_factor_recurrence'] = 'C5=n(n+3), C4=l(l+2). Four SO4-equivalent coset directions and B=2 times a unit generator give averaged B^2=C5-C4. Forward product-CG trace averaged over (l+1)^2 states is (l+2)^2/(l+1)^2 and reverse norm is f_(l-1)^2. Therefore f_l^2*(l+2)^2/(l+1)^2+f_(l-1)^2=n(n+3)-l(l+2), f_-1=0. This uniquely yields f_l^2=(n-l)(n+l+3)(l+1)/(2(l+2)), terminal f_n=0. Exact Fraction consumer checks n1,n2,n3,n126; small canonical tensor/Casimir fixes phases and normalization independently.'
    (ROOT/(PREFIX+'proof_20261011.json')).write_text(json.dumps(proof, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', required=True)
    parser.add_argument('--iterations', type=int, default=250)
    parser.add_argument('--response-iterations', type=int, default=100)
    parser.add_argument('--tolerance', type=float, default=2e-5)
    parser.add_argument('--delta', type=float, default=.1)
    parser.add_argument('--state-input')
    parser.add_argument('--state-sha256')
    run(parser.parse_args())

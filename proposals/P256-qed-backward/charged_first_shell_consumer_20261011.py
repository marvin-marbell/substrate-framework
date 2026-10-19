"""Gauge-complete charged first shell, normalized circuit control, NOT phase admission.

The exact formulas and role accounting apply to all64. The finite fullCAR
calculation has ONE spin replica, not all64, and is labelled accordingly.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import zipfile
from typing import Any

import numpy as np
from scipy.linalg import eigh, expm

ROOT = Path(__file__).resolve().parent
PREFIX = 'charged_first_shell_'
INPUTS = [
    'local_so5_resum_source_20261011.py',
    'local_so5_resum_actual04_20261011_receipt.json',
    'local_so5_resum_actual04_20261011_state.npz',
    'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json',
    'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json',
    'local_so5_positive_output_row_inertia_certificate_20261011_receipt.json',
    'local_so5_ground_electric_memory_consumer_20261011_receipt.json',
    'local_so5_resum_allocation_20261011.py',
    'charged_first_shell_proof_20261011.json',
]
PINS = {
    INPUTS[0]: '7f73170ca3a33782b5b391a6f6d3b6e0e3b7bf385c2f03cc2b5f7696fde8e009',
    INPUTS[1]: 'a31c2d5be2a368b7de703d92461e230af59adf7e8ee8f4eaca0944fe4f556fc1',
    INPUTS[2]: 'dd06400e1b7f70df935cf7ca02cf5ebf598481a8d5e209d1241aba5ed3fea2a2',
    INPUTS[3]: '102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8',
    INPUTS[5]: 'e8fc8c9159b66fca7544d83813a833ba6018151bc863f6396a4ef5666ced227c',
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def archive(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted(members.items()):
            z.writestr(name, raw)
    path.chmod(0o444)


def car(mask: int, p: int, q: int) -> tuple[int, int] | None:
    # Independent CAR implementation, not the selected-block solver.
    if not (mask >> q) & 1:
        return None
    sign = -1 if (mask & ((1 << q)-1)).bit_count() % 2 else 1
    mask ^= 1 << q
    if (mask >> p) & 1:
        return None
    sign *= -1 if (mask & ((1 << p)-1)).bit_count() % 2 else 1
    return mask | (1 << p), sign


def second(h: np.ndarray, masks: list[int]) -> np.ndarray:
    lookup = {mask: i for i, mask in enumerate(masks)}
    out = np.zeros((len(masks), len(masks)), complex)
    for j, mask in enumerate(masks):
        for p, q in zip(*np.nonzero(h)):
            transition = car(mask, int(p), int(q))
            if transition is not None:
                out[lookup[transition[0]], j] += transition[1]*h[p, q]
    return out


def finite_control(m: float, c0: float, ci: float) -> dict[str, Any]:
    """Full fixed-number CAR on four sites, native tree electric, one spin replica."""
    annih = []
    for q in range(4):
        op = np.zeros((16, 16))
        for mask in range(16):
            if (mask >> q) & 1:
                op[mask ^ (1 << q), mask] = (-1)**((mask & ((1 << q)-1)).bit_count())
        annih.append(op)
    car_error = max(float(np.linalg.norm(annih[p] @ annih[q].T+annih[q].T @ annih[p]
                    -(np.eye(16) if p == q else 0))) for p in range(4) for q in range(4))
    assert car_error == 0
    a = np.diag([-1., 1., -1., 1.])
    b = np.kron(np.array([[0., 1.], [1., 0.]]), np.diag([1., -1.]))
    x = np.array([[0., 1.], [1., 0.]])
    c = np.block([[np.zeros((2, 2)), 1j*x], [-1j*x, np.zeros((2, 2))]])
    hlocal = m*a+c0*b+ci*c
    masks4 = [sum(1 << p for p in z) for z in itertools.combinations(range(4), 2)]
    rho4 = np.array([(mask & 3).bit_count()-1 for mask in masks4])
    hl = second(hlocal, masks4)+np.diag(64*rho4**2)
    el, ul = eigh(hl)
    local = ul[:, 0]
    masks = [sum(1 << p for p in z) for z in itertools.combinations(range(8), 4)]
    lookup = {mask: i for i, mask in enumerate(masks)}
    omega = np.zeros(len(masks), complex)
    for i, left in enumerate(masks4):
        for j, right in enumerate(masks4):
            omega[lookup[left | (right << 4)]] = local[i]*local[j]
    occupation = np.array([[(mask >> (2*s)) & 3 for s in range(4)] for mask in masks])
    number = np.array([[int(v).bit_count() for v in row] for row in occupation])
    rho = number-1
    # Literal Dinc=row-col: positive edge j -> j+1 has -1 at tail,+1 at head.
    D = np.zeros((4, 3), int)
    for j in range(3):
        D[j, j] = -1; D[j+1, j] = 1
    fullE = -np.cumsum(rho, axis=1)[:, :3]
    assert np.array_equal(fullE @ D.T, rho)
    qa = number[:, :2].sum(axis=1)-2
    relative = np.column_stack(((number[:, 0]-number[:, 1])/2, (number[:, 2]-number[:, 3])/2))
    affine = np.column_stack((-relative[:, 0]-qa/2, -qa, -relative[:, 1]-qa/2))
    assert np.array_equal(fullE, affine)
    hm = np.zeros((8, 8), complex)
    hm[:4, :4] = hlocal; hm[4:, 4:] = hlocal
    Hm = second(hm, masks)
    HE = np.diag(64*np.sum(fullE**2, axis=1))
    H0 = Hm+HE
    # One native external link connects site1 to site2. No neutral-sector projection.
    external = np.zeros((8, 8), complex)
    external[2:4, 4:6] = hlocal[:2, 2:4]
    forward = second(external, masks)
    V = forward+forward.conj().T
    # Energy-weighted antihermitian root generator: a normalized connected
    # Schrieffer-Wolff trial, not an alteration of the physical Hamiltonian.
    A = (H0 @ V-V @ H0)/64
    H = H0+V
    source = V @ omega
    unsigned = np.zeros_like(forward)
    for j, mask in enumerate(masks):
        for p, q in zip(*np.nonzero(external)):
            transition = car(mask, int(p), int(q))
            if transition is not None:
                unsigned[lookup[transition[0]], j] += external[p, q]
    wrong_car = float(np.linalg.norm((unsigned-forward) @ omega))
    wrong_car_columns = np.linalg.norm(unsigned-forward, axis=0)
    wrong_car_witness = int(np.argmax(wrong_car_columns))
    assert wrong_car_columns[wrong_car_witness] > 1
    first = np.column_stack((omega, forward @ omega, forward.conj().T @ omega))
    gram = first.conj().T @ first
    Q, _ = np.linalg.qr(first)
    acted = H @ first
    exterior = acted-Q @ (Q.conj().T @ acted)
    wrongHE = np.diag(64*(np.sum(relative**2, axis=1)+qa**2))
    charged = np.flatnonzero(np.abs(qa) == 1)
    charged_bottom = float(eigh(H0[np.ix_(charged, charged)], eigvals_only=True)[0])
    charged_gap = charged_bottom-2*float(el[0])
    grid = []
    for tau in (0., -.005, -.01, -.02, -.04, -.08):
        psi = expm(tau*A) @ omega
        action = H @ psi
        energy = float(np.vdot(psi, action).real)
        grid.append(dict(tau=tau, norm=float(np.vdot(psi, psi).real), energy=energy,
                         full_action_norm=float(np.linalg.norm(action)),
                         full_variance=float(np.linalg.norm(action-energy*psi)**2)))
    best = min(grid, key=lambda row: row['energy'])
    # Independent occupation-basis gauge control: every hop obeys literal DeltaE.
    hop_error = 0.
    for j, mask in enumerate(masks):
        for p, q in zip(*np.nonzero(external)):
            transition = car(mask, int(p), int(q))
            if transition is not None:
                i = lookup[transition[0]]
                expected = np.zeros(3); expected[1] = -1
                hop_error = max(hop_error, float(np.linalg.norm(fullE[i]-fullE[j]-expected)))
    assert hop_error == 0 and charged_gap > 0
    assert np.linalg.norm((H0-2*el[0]*np.eye(len(masks))) @ omega) < 1e-10
    assert best['energy'] < grid[0]['energy'] and max(abs(row['norm']-1) for row in grid) < 1e-12
    wrong_cross = float(np.linalg.norm((HE-wrongHE) @ source))
    exterior_norm = float(np.linalg.norm(exterior[:, 1:]))
    assert wrong_cross > 1 and exterior_norm > 1
    # A shared integer edge is NOT copywise independent: electric cost of two
    # same-oriented unit flows is256, not128. Opposite flows cost0, not128.
    shared = {'same_oriented_full': 256, 'factorized_wrong': 128,
              'opposite_oriented_full': 0, 'cross_same': 128, 'cross_opposite': -128}
    # Normalized disjoint unitary circuits: metric folds cancel vacuum bubbles
    # exactly, before any thermodynamic assertion. Numeric check uses fullCAR roots.
    psi = expm(best['tau']*A) @ omega
    norm = float(np.vdot(psi, psi).real)
    e = float(np.vdot(psi, H @ psi).real)
    disjoint_normalized_energy = (e*norm+norm*e)/(norm*norm)
    return dict(scope='NUMERICAL_ONE_SPIN_REPLICA_TWO_DIMERS_70_DIMENSION_FULLCAR_TREE_INTEGERFLOW; NOT ALL64',
                dimension=len(masks), canonical_per_replica_initial=[2, 2],
                charged_numbers=[[1, 3], [3, 1]], local_neutral_energy=float(el[0]),
                charged_sector_ground_gap_float=charged_gap,
                rooted_source_weight=float(np.vdot(source, source).real),
                rooted_inverse_gap_square_upper_control=float(np.vdot(source, source).real/charged_gap**2),
                source_gram_real=np.real(gram).tolist(), source_gram_imag=np.imag(gram).tolist(),
                complete_first_shell_H_action_norms=np.linalg.norm(acted, axis=0).tolist(),
                complete_exterior_norm=exterior_norm,
                PHP_square_is_full_H_square=False,
                missing_rho_affine_cross_action_error=wrong_cross,
                literal_Dinc_row_minus_col=True, affine_E_identity_error=0,
                gauge_hop_DeltaE_error=hop_error, overlap_signs_from_fullCAR=True,
                independent_full16_CAR_error=car_error, wrong_unsigned_CAR_source_error=wrong_car,
                unsigned_CAR_contrary_occupation_mask=masks[wrong_car_witness],
                unsigned_CAR_contrary_action_error=float(wrong_car_columns[wrong_car_witness]),
                unsigned_CAR_neutral_source_is_blind=wrong_car < 1e-10,
                normalized_generator='A=[H0,V]/64; antihermitian, retains charged and electric sectors',
                normalized_root_circuit_grid=grid, best_normalized_root_circuit=best,
                circuit_stationary=False, fullPhi_lift=False,
                disconnected_normalization=dict(norm=norm, two_root_normalized_energy=disjoint_normalized_energy,
                    twice_single_root_energy=2*e, exact_identity='(E Z + Z E)/Z^2 = 2 E/Z; unitary roots have Z=1',
                    vacuum_bubble_in_connected_log_zero=True), shared_flow_contrary_control=shared)


def selected_moments(raw: bytes, state_path: Path) -> dict[str, Any]:
    receipt = json.loads(raw)
    with np.load(state_path, allow_pickle=False) as checkpoint:
        state = checkpoint['state'].reshape(14, 690880)
    ra = np.array([p-q for l in range(3) for p in range(l+1) for q in range(l+1)])
    rb = np.array([p-q for l in range(127) for p in range(l+1) for q in range(l+1)])
    norm = float(np.sum(state*state))
    moments = np.zeros(4)
    for i in range(14):
        probability = state[i]*state[i]
        r = ra[i]+rb
        for k in range(1, 5):
            moments[k-1] += float(probability @ (r.astype(float)**k))/norm
    infidelity = F(receipt['selected_ground_infidelity_upper_exact'])
    assert 2*infidelity < F(1, 60000)**2
    # Positive sums: gamma_(16d) is a deliberately conservative binary64 reduction
    # cap, including products, rho multiplication and normalization. m2<1 then
    # ||rho Omega||<=1+128/60000, whose square is<2. No densityzero=P0.
    gamma = F(16*state.size, 2**53-16*state.size)
    assert moments[1]/(1-float(gamma)) < 1 and (F(1)+F(128, 60000))**2 < 2
    return dict(source_moments_float=moments.tolist(), source_norm_float=norm,
                positive_sum_roundoff_relative_upper_exact=str(gamma),
                ground_norm_distance_upper_exact='1/60000', ground_rho_second_moment_upper_exact='2',
                proof='positive reduction m2<1 with gamma_(16d); norm-distance <=sqrt(2 infidelity)<1/60000; ||rho||<=128; triangle bound gives ground m2<2',
                binary64_model_conditional=True, uses_infidelity_not_unsquared_error=True,
                parent_memory_response_used_as_charged_gap=False)


def analytic_roles(bound: Any, captured: dict[str, bytes], upper: dict[str, Any]) -> dict[str, Any]:
    frame, scale = bound.frozen.exact_frame(captured[bound.INPUTS[0]])
    cfg = json.loads(captured[bound.INPUTS[1]])['input_descriptor']['original_micro_config_UNCHANGED']
    profiles = bound.frozen.profiles(cfg, frame, scale)
    rep = json.loads(captured[INPUTS[3]])
    # Rational sqrt majorants for free neutral lower and charged penalty lower.
    # e_low^2<=m^2+d^2<=e_high^2, outward on a fixed 2^-30 grid.
    intervals = []
    for profile in rep['profiles']:
        e2 = F(profile['e_squared_exact'])
        lo = F(int(np.sqrt(float(e2))*2**30), 2**30)
        while lo*lo > e2:
            lo -= F(1, 2**30)
        hi = lo+F(1, 2**30)
        while hi*hi < e2:
            hi += F(1, 2**30)
        intervals.append((lo, hi))
    free_lower = -4*sum(p['count']*pair[1] for p, pair in zip(profiles, intervals))
    e0_upper = F(upper['ground_energy_interval_exact'][1])
    c0 = abs(F(rep['profiles'][0]['c0_exact']))
    role_rows = []
    root_numerator = F(0)
    for profile, (elo, ehi) in zip(profiles, intervals):
        paths = []
        for path in profile['paths']:
            # Finite CAR hopping operator <=2 sum |matrix entries|; l1 over real
            # and imaginary rational parts is a valid sqrt-free majorant.
            K = 2*sum(abs(z[0])+abs(z[1]) for row in path['matrix'] for z in row)
            # Matching internal edge intersections may remove one edge. All
            # length2 transverse paths retain2 external edges; longitudinal may1.
            ext_min = 1 if path['axis'] == 0 else path['length']
            delta = 2*free_lower+2*(elo-c0)+64*ext_min-2*e0_upper
            # COMPLETE source-weighted electric commutator, never a current
            # operator norm: triangle inequality keeps rho-electric cross terms.
            # At most2 selected tree edges contribute: sum ||rho_j Omega||<3.
            electric_commutator = 64*K*(6+path['length'])
            # Neutral block number changes by +/-1 in a SINGLE spin sector of
            # whichever actual copy the native matrix hits. No neutral gap used.
            paths.append(dict(axis=path['axis'], length=path['length'], matrix_exact=[[[str(v) for v in z] for z in row] for row in path['matrix']],
                              both_orientations=True, all_endpoints_retained=True,
                              fullCAR_hop_norm_upper_exact=str(K),
                              charged_two_block_lower_relative_exact=str(delta),
                              charged_gap_certified_by_this_bound=delta > 0,
                              source_weighted_electric_commutator_upper_exact=str(electric_commutator),
                              external_edges_min=ext_min))
        role_rows.append(dict(role=profile['role'], count=profile['count'], spin_replicas=2*profile['count'],
                              onsite_exact=str(profile['onsite']), free_e_interval_exact=[str(elo), str(ehi)], paths=paths))
    all_path_bounds = []
    for axis in range(3):
        for length in (1, 2):
            totalK = sum(row['count']*F(next(p['fullCAR_hop_norm_upper_exact'] for p in row['paths']
                         if p['axis'] == axis and p['length'] == length)) for row in role_rows)
            allocated = F(1 if axis == 0 else 2) if length == 1 else F(2)
            root_numerator += allocated*totalK*totalK
            # A pointwise once-shared bound, not a fictitious independent-copy
            # operator norm. The same compact field is used by every actual role.
            all_path_bounds.append(dict(axis=axis, length=length,
                complete_all64_shared_pointwise_norm_upper_exact=str(totalK),
                root_allocation_exact=str(allocated), norm_method='triangle over actual1+aux63, NOT sqrt(count)'))
    # Explicit complete overlap support on N8, whose radius2 neighborhoods do not
    # wrap ambiguously. Enumerate every q once; shared vertices/edges never get
    # replicated by copies. This finite enumeration is stable for all even N>=8.
    sites = list(itertools.product(range(8), repeat=3))
    def vertices(x: tuple[int, ...], axis: int, length: int) -> set[tuple[int, ...]]:
        return {tuple((x[j]+(r if j == axis else 0)) % 8 for j in range(3)) for r in range(length+1)}
    complete_overlap = []
    onsite_per_site = 4*sum(p['count']*p['onsite'] for p in profiles)
    for p in all_path_bounds:
        support = vertices((1, 0, 0), p['axis'], p['length'])
        touchedK = F(0); number = 0
        for x in sites:
            for q in all_path_bounds:
                if support & vertices(x, q['axis'], q['length']):
                    touchedK += F(q['complete_all64_shared_pointwise_norm_upper_exact'])
                    number += 1
        K = F(p['complete_all64_shared_pointwise_norm_upper_exact'])
        full_bound = 2*K*(len(support)*onsite_per_site+touchedK)+64*K*(6+p['length'])
        complete_overlap.append(dict(axis=p['axis'], length=p['length'], distinct_overlapping_paths=number,
            once_shared_overlap_norm_sum_exact=str(touchedK),
            full_connected_commutator_source_norm_upper_exact=str(full_bound),
            all64_source_gram_diagonal_upper_exact=str(K*K),
            offdiagonal_gram_upper='|Gpq|<=Kp Kq; signs are retained, not assumed cancelling'))
    assert sum(p['count'] for p in profiles) == 64
    assert any(not path['charged_gap_certified_by_this_bound'] for row in role_rows for path in row['paths'])
    return dict(all64_analytic=True, all64_numerical=False, roles=role_rows,
                all64_once_shared_pointwise_norms=all_path_bounds,
                complete_overlap_enumeration=complete_overlap,
                neutral_free_lower_exact=str(free_lower),
                exact_selected_neutral_E0_upper=str(e0_upper),
                selected_neutral_gap_never_used_for_charged_sector=True,
                rooted_criterion=dict(physical_hopping=1, allocated_norm_square_numerator_exact=str(root_numerator),
                    formula='q_root=sum_paths allocation * (sum_roles count*K_path)^2 / min_role(delta_charged(path))^2 <1, only if every delta_charged>0; first-shell candidate, NOT proved allorders criterion',
                    disconnected_vacuum_normalization='normalized unitary root circuit; log norm=0 exactly, rather than undressed sqrtV resolvent',
                    satisfied=False, value=None, first_failure='one-link charged lower bound is nonpositive; a source-weighted neutral rho resolvent does not repair a missing charged gap',
                    not_a_global_no_go=True),
                complete_exterior_bound=dict(formula='||[H,T_p]Omega|| <= ||[Hm,T_p]|| + ||[64 E^2,T_p]Omega|| + 2 K_p sum_{q overlap p} K_q; local source residual adds K_p residual when replacing exact ground by actual04',
                    matter_commutator='native finite CAR onebody commutator; <=2 K_p * (sum touching onsite and internal path norms)',
                    all_path_choices='both lengths, all3 axes, all64 distinct role matrices; endpoint and intermediate-edge intersection required, not endpoints only',
                    scope='connected rooted commutator on the normalized selected product. NOT global fullaction norm: disconnected spectators belong to normalized metric/log fold. Complete finite fullCAR action is exercised independently.',
                    unbounded_current_operator_norm_asserted=False))


def run(output: Path) -> None:
    started = time.monotonic()
    captured = {name: (ROOT/name).read_bytes() for name in INPUTS}
    for name, expected in PINS.items():
        assert digest(captured[name]) == expected, name
    captured[Path(__file__).name] = Path(__file__).read_bytes()
    import bulk_electric_phase_consumer as bound
    # Import declarations and exact suppliers ONLY; never call old run/solvers.
    captured.update(bound.snapshot())
    import local_so5_resum_allocation_20261011 as allocation
    captured[Path(allocation.__file__).name] = Path(allocation.__file__).read_bytes()
    archive(output.with_name(output.stem+'_preexecution.zip'), captured)
    upper = json.loads(captured[INPUTS[4]])
    actual = json.loads(captured[INPUTS[1]])
    moments = selected_moments(captured[INPUTS[4]], ROOT/INPUTS[2])
    roles = analytic_roles(bound, captured, upper)
    controls = []
    for p in actual['parameters']:
        controls.append(dict(role=p['role'], **finite_control(p['onsite_float'], p['c0'], p['ci'])))
    allocations = [allocation.count(n) for n in (4, 6, 8)]
    result = dict(kind='NEW_CHARGED_CANONICAL_FIRST_SHELL_AFFINE_E_AND_NORMALIZED_ROOT_BOUNDARY_NOT_PHASE',
        command=[sys.executable, *sys.argv], physical_hopping=1,
        native_law_unchanged=dict(a='1/64', auxiliary_count=63, Lambda='2^8192', chi='2^-4096', F='2^10240', HE=64, HC=64, no_bare_U1_B2=True),
        affine_electric_proof=dict(Dinc='row-col, -tail+head',
            rho='occupation minus neutral background',
            E='k_rho+C(n+beta), Dinc k_rho=rho; never C-only',
            external_hop='create at tail, annihilate at head gives DeltaE=-positive path and Delta rho=Dinc DeltaE',
            two_dimer_chain='E=(-rho_relative_A-qA/2, -qA, -rho_relative_B-qA/2), qB=-qA',
            canonical_sector_change='neutral Ncell=2 per spinreplica -> (1,3) or (3,1); native spin-mixing axes also change spin-sector tags; all128 spinreplicas of64 roles accounted',
            shared_energy='64||E+k||^2=64||E||^2+128 E.k+64||k||^2; overlaps add128 k1.k2, either sign',
            cycle_chart='For any integer tree section k_rho and integral cycle basis C, beta=C_chord inverse*(k-k_rho) modulo integer coordinates. Full E independent of section; charged k_rho may not be replaced by0.'),
        actual_selected_source_moments=moments, all64_boundary=roles,
        normalized_matching_allocation=allocations, finite_fullCAR_integerflow_contrary_controls=controls,
        status=dict(fullaction='complete in70-dimensional numerical controls; rigorous connected commutator bound analytic all64, not global fullaction',
            norm='unit normalized root unitary circuit exercised; actual04 source normalized with roundoff/infidelity distinction',
            variance='complete numerical finite-control variance, nonzero; PHP squared rejected by exterior',
            stationarity=False, selected_neutral_ground=True, charged_all64_ground_gap=False,
            physicalPhi_lift=False, N_uniform_overlapping_flow_linked_phase=False, IRatom=False, campaign_complete=False),
        next_constructive_decision=dict(choice='extend the exercised normalized root unitary from ONE replica to symmetry-reduced charged spinor x harmonic branches at both actual masses, retaining affine flux offsets and full exterior',
            changed_material='normalized exact root dressing already lowers fullCAR control energy, preserves norm1 and exposes exterior; replaces undressed neutral-gap resolvent',
            immediate_target='certify lower spectrum or source-weighted charged inverse in (n-1 harmonic)xSO5 spinor branches and shared-flow two-root overlap, then recompute q_root',
            not_an_untried_fork='both actual-mass ONE-replica fullCAR root dressings, charged gaps, fullaction/exterior and normalized disjoint fold exercised here; all64 charged representation lift remains unpaid'),
        source_and_input_sha256={name: digest(raw) for name, raw in captured.items()},
        source_unchanged=True, inputs_unchanged=True, elapsed_seconds=time.monotonic()-started)
    for name, raw in captured.items():
        if '/' not in name:
            assert (ROOT/name).read_bytes() == raw, name
    output.write_text(json.dumps(result, indent=2)+'\n')
    output.chmod(0o444)
    archive(output.with_name(output.stem+'_executed.zip'), {**captured, output.name: output.read_bytes()})
    print(json.dumps(dict(receipt=output.name, sha256=digest(output.read_bytes()),
        charged_all64_criterion=False, control_best=[c['best_normalized_root_circuit'] for c in controls],
        inputs_unchanged=True, elapsed_seconds=result['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)

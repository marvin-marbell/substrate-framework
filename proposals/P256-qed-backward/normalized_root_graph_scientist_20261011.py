"""Physical hopping-one normalized graphs; complete rooted H paid, NOT ground.

Consumes immutable harmonic CAR G/T/U, never runs a source/ground solver.
Second graph includes actual charge-two same-edge CAR source, not a Ritz pole.
The complete finite rooted Hamiltonian is two matching dimers + their link;
all64 roles share the three integer edge flows. Infinite-lattice spectators,
physicalPhi derivatives, phase and a neutral-exterior denominator remain open.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parent
PINS = {
    'harmonic_car_bridge_typed_attempt3_20261011_receipt.json': 'be07db74d79630fd62866e4a44eb32e8b9217eaf59bf2ce1f26fdc247e91ff21',
    'harmonic_car_bridge_typed_attempt3_20261011_executed.zip': '03e36cd18eeb400d98b5cdd25a817d38c6f485a6d128511594eb46050a524906',
    'connected_root_exterior_actual1_20261011_moments.npz': '4c6c6e0da22e0d9d405c37935332bb9548fbf2df5f429ec66b50aa671f8af67d',
    'local_so5_resum_source_20261011.py': '7f73170ca3a33782b5b391a6f6d3b6e0e3b7bf385c2f03cc2b5f7696fde8e009',
    'charged_output_lower_attempt3_20261011_receipt.json': '427f5dbcff0fbfbb126c13b7d6c157a0ac6997248af53ce76f00ddc08384f7c2',
    'charged_first_shell_actual04_20261011_receipt.json': 'b48aed4cb67ae2fd98e0b75ee4c58053244c68e1e4bd47c4cb97c9fdf58e4f02',
    'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json': '102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8',
    'local_so5_resum_actual04_20261011_state.npz': 'dd06400e1b7f70df935cf7ca02cf5ebf598481a8d5e209d1241aba5ed3fea2a2',
    'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz': 'f8a0793adcbf4909402cd089471aa8c33fc74a31b66e99618f8baaab8d6dfb65',
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def exclusive_json(p: Path, x: Any) -> None:
    with p.open('x') as stream:
        json.dump(x, stream, indent=2)
        stream.write('\n')


def decode(v: Any) -> np.ndarray:
    z = np.asarray(v, dtype=float)
    return z[..., 0]+1j*z[..., 1]


def encode(v: Any) -> Any:
    a = np.asarray(v)
    return np.stack([a.real, a.imag], axis=-1).tolist()


def contract(m: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float(np.einsum('ij,kl,ik,jl->', m.conj(), m, a, b).real)


def endpoint(mom: Any, site: int) -> dict[str, np.ndarray]:
    ix = [2*site, 2*site+1]
    return {k: np.kron(np.eye(2), decode(v)[np.ix_(ix, ix)]) for k, v in mom.items()}


def first_graph(harmonic: Any, boundary: Any, spin: np.ndarray) -> dict[str, Any]:
    energy = harmonic['source_energy_float']
    source_norm = harmonic['source_norm_float']
    rows = []
    W = M1 = M2 = 0.0
    K = F(0)
    # One literal external edge: dimers (0,1),(2,3), site1 -> site2.
    for role in boundary['roles']:
        path = next(p for p in role['paths'] if p['axis'] == 0 and p['length'] == 1)
        physical = np.array([[complex(float(F(z[0])), float(F(z[1]))) for z in row] for row in path['matrix_exact']])
        m = spin.conj().T @ physical @ spin
        moments = {r['orientation']: r['endpoint_reduced_matrices'] for r in harmonic['source_roles'] if r['role'] == role['role']}
        choices = []
        for qa in (-1, 1):
            a = endpoint(moments['creation' if qa == 1 else 'annihilation'][str(qa/2)], 1)
            b = endpoint(moments['annihilation' if qa == 1 else 'creation'][str(qa/2)], 0)
            a = {k:v/source_norm for k,v in a.items()}
            b = {k:v/source_norm for k,v in b.items()}
            pa = np.diag([1, -1, 1, -1])*(-1)
            pb = np.diag([1, -1, 1, -1])
            n = pa @ m @ pb
            if qa == -1:
                n = n.conj()
            shift = 64-2*energy
            w = contract(n, a['G'], b['G'])
            t = contract(n, a['T'], b['G'])+contract(n, a['G'], b['T'])+shift*w
            u = (contract(n, a['U'], b['G'])+contract(n, a['G'], b['U'])+2*contract(n, a['T'], b['T'])
                 +2*shift*(t-shift*w)+shift*shift*w)
            choices.append({'qA': qa, 'weight': w, 'relative_first_moment': t, 'relative_second_moment': u,
                            'coherent_spin_matrix': encode(n), 'endpoint_A': {k: encode(v) for k, v in a.items()},
                            'endpoint_B': {k: encode(v) for k, v in b.items()}})
        count = role['count']
        W += count*sum(c['weight'] for c in choices)
        M1 += count*sum(c['relative_first_moment'] for c in choices)
        M2 += count*sum(c['relative_second_moment'] for c in choices)
        K += count*F(path['fullCAR_hop_norm_upper_exact'])
        rows.append({'role': role['role'], 'count': count, 'physical_matrix_exact': path['matrix_exact'], 'choices': choices})
    mu = M1/W
    variance_mass = M2-M1*M1/W
    t = 2/(mu+math.sqrt(mu*mu+4*W))
    Z = 1+t*t*W
    lam = -t*W
    # Orthogonal charge parity: L B is odd, Q V B is even.
    # P V B=W Omega, so the removed P component is subtracted, not dropped.
    exterior_square_upper = float(K*K)*W-W*W
    assert exterior_square_upper > 0 and variance_mass > 0
    riccati_square_upper = t*t*(variance_mass+exterior_square_upper)
    residual = math.sqrt(riccati_square_upper/Z)
    source_error = 2*harmonic['source_residual_float_observation_not_recertification']*(1+t*float(K))/math.sqrt(Z)
    floors = [F(r['charged_gap_lower_exact']) for r in harmonic['paths'][0]['roles']]
    g1 = min(floors)
    g2 = 2*F(boundary['neutral_free_lower_exact'])+256-2*F(boundary['exact_selected_neutral_E0_upper'])
    assert g1 > 0 and g2 > 0
    return {'stage': 'FLOATING_MOMENTS_WITH_EXACT_OPERATOR_AND_ALGEBRAIC_EXTERIOR_MAJORANT_NOT_INTERVAL_GROUND',
            'operator_id': 'physical1_signedCAR64_four_sites_integer_tree_three_edges_matching01_23_external12',
            'hopping_exact': '1', 'root_allocation_in_H': False, 'roles': rows,
            'W': W, 'M1': M1, 'M2': M2, 'mu': mu, 'variance_mass': variance_mass,
            'X': '-t B Omega, B=Q V P actual all64 two-orientation hopping; one-dimensional P=|OmegaOmega><OmegaOmega|',
            't_all_roles': t, 'Z': Z, 'normalizer': 1/math.sqrt(Z), 'U': '(Omega-t B Omega)/sqrt(1+t^2 W)',
            'root_compressed_energy_shift': lam, 'full_H0_action_square': M2,
            'Riccati_equation': 'R=B+DX-XA-X Bdag X; relative A=0, D=Q(H0+V-2E0)Q, X=-tB',
            'projected_Riccati_root_not_full': True,
            'full_Riccati_identity': 'R=t(mu B-L B)-t Q V B; L=H0-2E0, mean relation 1-mu*t-W*t^2=0',
            'K_V_complete_all64_sharedfield_exact': str(K),
            'QV_B_norm_square_upper': exterior_square_upper,
            'QV_B_bound': '||QVB||^2=||VB||^2-W^2 <= K_V^2 W-W^2; includes charge2,4PH,neutral exterior, distinct-copy role crossactions',
            'full_HU_minus_Ulambda_norm_upper_float': residual+source_error,
            'full_HU_minus_Ulambda_square_upper_float_without_source_error': riccati_square_upper/Z,
            'source_binary_residual_addition_float': source_error,
            'paid_odd_H0_residual_norm': t*math.sqrt(variance_mass/Z),
            'paid_even_VQ_residual_norm_upper': t*math.sqrt(exterior_square_upper/Z),
            'actual_charge1_denominator_lower_exact': str(g1),
            'actual_charge2_denominator_lower_exact': str(g2),
            'charge2_denominator_derivation': 'central integer flow |qA|=2 costs256; all matter >=2 neutral free lower, internal shared HE squares nonnegative; subtract2 selected E0 upper. NO neutral selected gap.',
            'neutral_Q_or_higher_charge_global_gap_proved': False,
            'decision': 'FAILS_COMPLETE_RESIDUAL_CONTROL; projected mean pole removed but whole-H exterior remains paid and large; no global no-go',
            'fullground': False, 'physicalPhi_lift': False, 'phase': False}


def second_graph(first: Any, peer: Any, boundary: Any, rep: Any) -> dict[str, Any]:
    # peer supplies literal same-edge CAR charge-two source W2, not W1 squared.
    w2 = float(peer['graph_interface']['two_orientation_charge_two_weight'])
    assert w2 > 0
    t = first['t_all_roles']
    W = first['W']
    m21 = float(peer['graph_interface']['two_orientation_charge_two_relative_first_moment'])
    m22 = float(peer['graph_interface']['two_orientation_charge_two_relative_second_moment'])
    # Minimize the actual WHOLE-H0 charge2 source-action residual. This is
    # not an eigenvalue pole, an inverse heat denominator, or an H coupling.
    fit_denominator = m22+2*t*W*m21+t*t*W*W*w2
    assert fit_denominator > 0
    u = t*(m21+t*W*w2)/fit_denominator
    Z = 1+t*t*W+u*u*w2
    matter = 8*sum(p['count']*(abs(F(p['m_exact']))+abs(F(p['c0_exact']))+abs(F(p['d_exact']))) for p in rep['profiles'])
    # In C=(T^2+Tdag^2)Omega the central flow is exactly +/-2, and each
    # physical end site has 64 four-spin copies: |rho_site|<=128.
    H0_abs = matter+64*(2*128**2+4)
    L_abs = H0_abs+2*abs(F(boundary['exact_selected_neutral_E0_upper']))
    K = F(first['K_V_complete_all64_sharedfield_exact'])
    comm_K = F(0)
    for role, profile in zip(first['roles'], rep['profiles']):
        entries = [[abs(F(z[0]))+abs(F(z[1])) for z in row] for row in role['physical_matrix_exact']]
        hm = np.zeros((16, 16), dtype=object)
        vm = np.zeros((16, 16), dtype=object)
        # The normalized physical onsite Clifford matrix has entries<=1;
        # retain all spin entries instead of assuming an onsite basis.
        for site in range(4):
            for a in range(4):
                for b in range(4):
                    hm[4*site+a, 4*site+b] = abs(F(profile['m_exact']))
        for left, right, target in [(0, 1, hm), (2, 3, hm), (1, 2, vm)]:
            for a in range(4):
                for b in range(4):
                    target[4*left+a, 4*right+b] = entries[a][b]
                    target[4*right+b, 4*left+a] = entries[a][b]
        # cdag_i c_j has norm<=1; absolute entry sum bounds its second
        # quantization. Different-copy matter commutators vanish identically.
        comm_K += role['count']*sum((hm @ vm+vm @ hm).reshape(-1))
    delta = first['source_binary_residual_addition_float']*math.sqrt(first['Z'])/(1+t*float(K))
    comm_source_square = first['M2']-128*first['M1']+4096*W
    assert comm_source_square > 0
    comm_source = math.sqrt(comm_source_square)
    # Internal tree edges depend on site0/site3 occupations and commute V12.
    # The central flow is0 on Omega and +/-2 on C: electric action256 C.
    LC_bound = 256*math.sqrt(w2)+float(comm_K)*math.sqrt(W)+float(K)*comm_source+2*float(K*K)*delta
    LC_bound = min(float(L_abs)*math.sqrt(w2), LC_bound)
    first_R = math.sqrt(first['full_HU_minus_Ulambda_square_upper_float_without_source_error']*first['Z'])
    extra = abs(u)*(LC_bound+(t*W+float(K))*math.sqrt(w2))
    source_price = first['source_binary_residual_addition_float']*math.sqrt(first['Z']/Z)
    residual = (first_R+extra)/math.sqrt(Z)+source_price
    g2 = F(first['actual_charge2_denominator_lower_exact'])
    # The mixed V term is exact: <B,VC>=||C||^2, since C=P_|q|=2 V B.
    numerator_known = -2*t*W+t*t*first['M1']-2*t*u*w2
    energy_low = (numerator_known+u*u*float(g2)*w2)/Z
    energy_high = (numerator_known+u*u*min(float(L_abs)*w2, math.sqrt(w2)*LC_bound))/Z
    coeff = u*t*W-t
    charge2_residual_square = u*u*m22+2*u*coeff*m21+coeff*coeff*w2
    assert charge2_residual_square >= 0
    neutral_exterior_square_upper = float(K*K)*W-W*W-w2
    assert neutral_exterior_square_upper > 0
    odd_residual_upper = t*math.sqrt(first['variance_mass'])+abs(u)*float(K)*math.sqrt(w2)
    replacement_action_price = 2*peer['graph_interface']['source_residual_observation_not_recertification']*float(K*K)
    charge2_paid_square = (math.sqrt(charge2_residual_square)+abs(u)*replacement_action_price)**2
    moment_paid_square = (charge2_paid_square+t*t*neutral_exterior_square_upper+odd_residual_upper**2)/Z
    moment_paid_residual = math.sqrt(moment_paid_square)+source_price
    actual_trial_energy = (numerator_known+u*u*m21)/Z
    g3 = 2*F(boundary['neutral_free_lower_exact'])+576-2*F(boundary['exact_selected_neutral_E0_upper'])
    assert g3 > 0
    first_charge2_lower = t*math.sqrt(w2/first['Z'])
    return {'stage': 'ACTUAL_SOURCE_HIGHER_CHARGE_GRAPH_CANDIDATE_FLOAT_MOMENT_AND_EXACT_ALGEBRAIC_CONTROL_NOT_GROUND',
            'X': '-t B Omega+u C, C=P_|qA|=2 V B Omega=T^2 Omega+(Tdag)^2 Omega',
            't_all_roles': t, 'u_all_roles': u, 'W2': w2, 'Z': Z, 'normalizer': 1/math.sqrt(Z),
            'u_construction': 't*(M21+tW W2)/(M22+2tW M21+t^2W^2 W2); unique minimizer of ||uLC+(utW-t)C||^2, uses full H0 C not PHP^2',
            'taylor_u_unfitted_comparator':t*t/2,
            'taylor_charge2_residual_square_ratio':((t*t/2)**2*m22+2*(t*t/2)*((t*t/2)*t*W-t)*m21+((t*t/2)*t*W-t)**2*w2)/(t*t*w2),
            'source_normalizer_identity': 'Z=1+t^2 W+u^2 W2; charge0, +/-1, +/-2 orthogonal; allcopy products coherent inside C',
            'material_change': 'NONZERO ACTUAL CHARGE2/4PH MULTICOPY COMPONENT in X, not a changed denominator or point normalization',
            'operator_id': first['operator_id'], 'charge2_source_interface': peer['graph_interface'],
            'H0_on_C_absolute_norm_upper_exact': str(H0_abs), 'L_on_C_absolute_norm_upper_exact': str(L_abs),
            'matter_V_commutator_complete_all64_entry_majorant_exact': str(comm_K),
            'L_C_sourceweighted_action_norm_upper_float': LC_bound,
            'L_C_sourceweighted_bound': '256 sqrt(W2)+K_comm sqrt(W)+K_V sqrt(M2-128M1+4096W)+2 K_V^2 delta_source; commute internalHE exactly, centralHE acts256, no neutralgap',
            'source_binary_residual_addition_float': source_price,
            'complete_H_action': 'Hrel(Omega-tB+uC) = B-t L B-t V B+u L C+u V C; neither V B nor V C is compressed',
            'full_Riccati_identity': 'R_second=R_first+u(L+tW)C+u Q V C; P H X=-tW, Bdag C=0',
            'complete_residual_bound': '||R_second|| <= ||R_first||+|u|[LC_sourcebound+(tW+K_V)sqrt(W2)]; ||HU-U(-tW)||=||R||/sqrt(Z), plus retained source error',
            'full_HU_minus_Ulambda_norm_upper_float': residual,
            'complete_H0_charge2_action_moments': {'relative_first':m21,'relative_second':m22,'action_norm':math.sqrt(m22)},
            'physical_first_candidate_charge2_residual_norm_lower_float':first_charge2_lower,
            'charge2_Riccati_residual_square_exact_identity_float':charge2_residual_square,
            'charge2_eigenvector_replacement_LC_action_error_price':replacement_action_price,
            'charge2_residual_square_after_eigenvector_replacement_price':charge2_paid_square,
            'changed_decision_charge2_residual_square_ratio_to_first':charge2_residual_square/(t*t*w2),
            'actual_charge2_residual_reduced_after_source_price':charge2_paid_square < t*t*w2,
            'neutral_QVB_norm_square_upper_after_deducting_P_and_charge2':neutral_exterior_square_upper,
            'odd_charge1_and_charge3_residual_norm_upper':odd_residual_upper/math.sqrt(Z),
            'full_HU_minus_Ulambda_moment_paid_norm_upper_float':moment_paid_residual,
            'full_HU_minus_Ulambda_moment_paid_square_upper_float':moment_paid_square,
            'moment_paid_identity': 'RC=u LC+(utW-t)C; charge0 residual=-t P0QVB; odd residual=t(muB-LB)+uQVC. Square orthogonal charges; bound neutral by K^2W-W^2-W2, retain coherent odd triangle.',
            'actual_variational_energy_shift_float':actual_trial_energy,
            'variational_energy_shift_eigenvector_replacement_error_price':u*u*math.sqrt(w2)*replacement_action_price/Z,
            'charge3_denominator_lower_exact':str(g3),
            'variational_energy_shift_enclosure_float_from_algebraic_bounds': [energy_low, energy_high],
            'exact_mixed_hopping_identity': '<B,V C>=W2, <C,V C>=0 by dimer charge parity',
            'charge2_denominator_lower_exact': str(g2),
            'known_subthreshold_gap_used_as_acceptance': False,
            'decision': 'FAILS_PAID_COMPLETE_CONTROL; highercharge graph was actually normalized and paid, not claimed exact Riccati or ground',
            'global_higher_charge_gap': False, 'physicalPhi_lift': False, 'phase': False}


def signed_second(h: np.ndarray, masks: list[int], unsigned: bool = False) -> Any:
    from scipy.sparse import coo_matrix
    lookup = {m:i for i,m in enumerate(masks)}
    rows, cols, values = [], [], []
    terms = [(int(p), int(q), h[p,q]) for p,q in zip(*np.nonzero(h))]
    for j, mask in enumerate(masks):
        for p,q,value in terms:
            if not (mask>>q)&1:
                continue
            sign = (-1)**((mask&((1<<q)-1)).bit_count())
            changed = mask^(1<<q)
            if (changed>>p)&1:
                continue
            sign *= (-1)**((changed&((1<<p)-1)).bit_count())
            changed |= 1<<p
            rows.append(lookup[changed])
            cols.append(j)
            values.append(value*(1 if unsigned else sign))
    return coo_matrix((values,(rows,cols)),shape=(len(masks),len(masks))).tocsr()


def finite_fullcar(first: Any, second: Any, frame: np.ndarray, epoch: str) -> dict[str, Any]:
    import itertools
    from scipy.sparse import diags
    masks = [sum(1<<p for p in z) for z in itertools.combinations(range(16),8)]
    lookup = {m:i for i,m in enumerate(masks)}
    number = np.array([[(m>>(4*site)&15).bit_count() for site in range(4)] for m in masks])
    rho = number-2
    E = -np.cumsum(rho,axis=1)[:,:3]
    D = np.array([[-1,0,0],[1,-1,0],[0,1,-1],[0,0,1]])
    assert np.array_equal(E@D.T,rho)
    q = number[:,:2].sum(axis=1)-4
    relative = np.column_stack([(number[:,0]-number[:,1])/2,(number[:,2]-number[:,3])/2])
    literal_affine = np.column_stack([-relative[:,0]-q/2,-q,-relative[:,1]-q/2])
    assert np.array_equal(E,literal_affine)
    wrong_affine = np.column_stack([-relative[:,0]-q/2,-q,-relative[:,1]+q/2])
    wrong_HE = 64*np.sum(wrong_affine**2,axis=1)
    # A simple canonical reference is only a direct fullCAR adverse diagnostic;
    # it is NOT substituted for the immutable harmonic source above.
    reference = sum(3<<(4*site) for site in range(4))
    omega = np.zeros(len(masks),complex)
    omega[lookup[reference]] = 1
    gamma0 = frame[0]/math.sqrt(float(np.trace(frame[0]@frame[0]).real/4))
    rep = json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_bytes())
    outputs: dict[str,Any] = {'masks':np.array(masks,np.uint16),'integer_flow':E,'dimer_charge':q}
    results = []
    for role, profile in zip(first['roles'],rep['profiles']):
        M = np.array([[complex(float(F(z[0])),float(F(z[1]))) for z in row] for row in role['physical_matrix_exact']])
        hm = np.zeros((16,16),complex)
        vm = np.zeros_like(hm)
        for site in range(4):
            hm[4*site:4*site+4,4*site:4*site+4] = float(F(profile['m_exact']))*gamma0
        for left,right,target in [(0,1,hm),(2,3,hm),(1,2,vm)]:
            target[4*left:4*left+4,4*right:4*right+4] = M
            target[4*right:4*right+4,4*left:4*left+4] = M.conj().T
        H0 = signed_second(hm,masks)+diags(64*np.sum(E*E,axis=1),dtype=np.float64)
        V = signed_second(vm,masks)
        transitions = V.tocoo(copy=False)
        internal_HE = 64*(E[:,0]**2+E[:,2]**2)
        internal_tree_commutator = (internal_HE[transitions.row]-internal_HE[transitions.col])*transitions.data
        assert np.all(internal_tree_commutator==0)
        assert np.all(abs(q[transitions.row]-q[transitions.col])==1)
        assert np.all(E[transitions.row,1]-E[transitions.col,1]==-(q[transitions.row]-q[transitions.col]))
        H = H0+V
        b = V@omega
        c = np.where(abs(q)==2,V@b,0)
        wrong_affine_action_error = float(np.linalg.norm((wrong_HE-64*np.sum(E*E,axis=1))*c))
        assert wrong_affine_action_error > 0
        t = first['t_all_roles']
        u = second['u_all_roles']
        x = -t*b+u*c
        Z = 1+float(np.vdot(x,x).real)
        psi = (omega+x)/math.sqrt(Z)
        action = H@psi
        energy = float(np.vdot(psi,action).real)
        residual = action-energy*psi
        trial = np.column_stack([omega,b,c])
        column_norm_square = np.array([1.,float(np.vdot(b,b).real),float(np.vdot(c,c).real)])
        # Orthogonal charge sectors make each nonzero column normalization
        # explicit; no matrix-rank crop or invented null threshold is used.
        assert np.all(column_norm_square>0)
        Q = trial/np.sqrt(column_norm_square)
        exterior = action-Q@(Q.conj().T@action)
        unsigned = signed_second(vm,masks,unsigned=True)
        wrongcols = np.asarray(abs(unsigned-V).sum(axis=0)).reshape(-1)
        witness = int(np.argmax(wrongcols))
        assert wrongcols[witness]>0 and np.linalg.norm(exterior)>0
        flow_error = 0
        for p0,p1 in zip(*np.nonzero(vm[:])):
            expected = np.zeros(3,int)
            tail,head = p1//4,p0//4
            expected[min(tail,head):max(tail,head)] = 1 if tail<head else -1
            for mask in [masks[witness],reference]:
                if not (mask>>p1)&1:
                    continue
                changed = mask^(1<<p1)
                if (changed>>p0)&1:
                    continue
                changed |= 1<<p0
                flow_error = max(flow_error,int(np.max(abs(E[lookup[changed]]-E[lookup[mask]]-expected))))
        assert flow_error==0
        key = role['role']
        outputs[key+'_U'] = psi
        outputs[key+'_HU'] = action
        outputs[key+'_residual'] = residual
        outputs[key+'_exterior'] = exterior
        outputs[key+'_physical_onebody'] = hm+vm
        results.append({'role':key,'count_scope':1,'dimension':len(masks),'spinlabels':[0,1,2,3],
                        'physical_hopping':1,'canonical_number':8,'Z':Z,'W':float(np.vdot(b,b).real),
                        'W2':float(np.vdot(c,c).real),'energy':energy,'full_HU_norm':float(np.linalg.norm(action)),
                        'full_residual_norm':float(np.linalg.norm(residual)),'full_exterior_norm':float(np.linalg.norm(exterior)),
                        'wrong_unsigned_CAR_occupation_witness':masks[witness],'wrong_unsigned_column_error':float(wrongcols[witness]),
                        'integer_Gauss_error':0,'integer_hop_flow_error':flow_error,
                        'same_sign_internal_tree_offsets_required':True,
                        'fullCAR_all_occupation_internal_HE_commutator_error':0,
                        'fullCAR_all_occupation_central_integer_flow_action_error':0,
                        'wrong_opposite_internal_affine_action_error_on_charge2':wrong_affine_action_error,
                        'projected_PHP2_is_full_H2':False,'ground':False,'source':'simple canonical diagnostic, NOT immutable harmonic source'})
    path = ROOT/f'normalized_root_graph_{epoch}_fullcar.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream,**outputs)
    return {'scope':'INDEPENDENT_DIRECT_FULL12870_SIGNEDCAR_ACTUAL4SPIN_DIAGNOSTIC; ONE COPY OF EACH ACTUAL ROLE, NOT64GROUND',
            'rows':results,'witness_npz':path.name,'witness_sha256':sha(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', required=True)
    parser.add_argument('--peer-interface', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    captured = {name: sha(ROOT/name) for name in PINS}
    assert captured == PINS
    harmonic = json.loads((ROOT/'harmonic_car_bridge_typed_attempt3_20261011_receipt.json').read_bytes())
    import zipfile
    with zipfile.ZipFile(ROOT/'harmonic_car_bridge_typed_attempt3_20261011_executed.zip') as frozen:
        archived_provider_sha = hashlib.sha256(frozen.read('harmonic_car_bridge_scientist_20261011.py')).hexdigest()
    assert archived_provider_sha==harmonic['own_source_sha256']=='7a637ffc083212a7cb22b250e7ec49aed3f7d4ad133abb6eb296f4d4b340a0f1'
    boundary = json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_bytes())['all64_boundary']
    rep = json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_bytes())
    with np.load(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz', allow_pickle=False) as data:
        phi = data['Phi_binary_section']
    had = np.array([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]], float)/math.sqrt(2)
    frame = np.einsum('ia,mij,jb->mab', had, phi, had)
    gamma = frame[1]/math.sqrt(float(np.trace(frame[1]@frame[1]).real/4))
    spin = np.column_stack([np.eye(4)[:,0], gamma@np.eye(4)[:,0], np.eye(4)[:,1], gamma@np.eye(4)[:,1]])
    assert np.linalg.norm(spin.conj().T@spin-np.eye(4)) < 1e-12
    first = first_graph(harmonic, boundary, spin)
    peerpath = ROOT/args.peer_interface
    peer = json.loads(peerpath.read_bytes())
    for name,pin in peer['sourcepins'].items():
        if name=='harmonic_car_bridge_scientist_20261011.py':
            assert pin==harmonic['own_source_sha256']=='7a637ffc083212a7cb22b250e7ec49aed3f7d4ad133abb6eb296f4d4b340a0f1'
        else:
            assert PINS[name]==pin
    assert peer['graph_interface']['axis']==0 and peer['graph_interface']['length']==1
    assert peer['graph_interface']['matching_endpoint_sites']==[1,0] and peer['graph_interface']['physical_hopping']==1
    assert peer['graph_interface']['integer_tree_matching_offsets_by_orientation']=={'+':[1,1],'-':[-1,-1]}, 'Wrong internal matching affine offsets: literal integer tree requires same signs'
    second = second_graph(first, peer, boundary, rep)
    fullcar = finite_fullcar(first, second, frame, args.epoch)
    # Adverses are computed on the actual physical quantities and accepted only
    # if the falsely cheaper answer differs, not merely a hardcoded flag.
    omit = first['paid_odd_H0_residual_norm']
    full = first['full_HU_minus_Ulambda_norm_upper_float']
    wrongW = sum(sum(c['weight'] for c in r['choices']) for r in first['roles'])
    wrongZ = 1+first['t_all_roles']**2*wrongW
    allocation = F(next(p['root_allocation_exact'] for p in boundary['all64_once_shared_pointwise_norms'] if p['length']==2))
    changed_coupling_weight = float(allocation*allocation)*first['W']
    assert full > omit and abs(first['Z']-wrongZ) > .01 and changed_coupling_weight != first['W']
    controls = {'omit_exterior': {'wrong_residual': omit, 'complete_paid_bound': full, 'rejected': full > omit},
                'omit_aux63_multiplicity': {'wrong_Z': wrongZ, 'actual_Z': first['Z'], 'rejected': abs(first['Z']-wrongZ) > .01},
                'rootallocation_as_coupling': {'allocation': str(allocation), 'wrong_W': changed_coupling_weight, 'actual_W': first['W'], 'rejected': changed_coupling_weight != first['W']},
                'shared_integer_flow': {'same_two_unit_flows_actual_HE':64*(1+1)**2, 'wrong_copywise_HE':64+64,
                                       'opposite_actual_HE':64*(1-1)**2, 'HC_added_again':False},
                'charge_two_independent_copy_square': {'wrong_W2': first['W']**2, 'actual_W2':second['W2'],
                                                      'difference':second['W2']-first['W']**2}}
    result = {'kind':'ACTUAL_PHYSICAL1_NORMALIZED_ROOT_GRAPH_COMPLETE_H_EXTERIOR_TWO_MATERIAL_CANDIDATES',
              'direct_fullcar_action_controls':fullcar,
              'sourcepins':captured, 'peer_interface_sha256':sha(peerpath), 'own_source_sha256':sha(Path(__file__)),
              'immutable_harmonic_provider_source_sha256':archived_provider_sha,
              'immutable_harmonic_provider_archive':'harmonic_car_bridge_typed_attempt3_20261011_executed.zip',
              'source_dimension':harmonic['immutable_source_dimension'], 'source_ground_solver_rerun':False,
              'first_candidate':first, 'second_candidate':second, 'adverse_controls':controls,
              'graph_interface':{'P':'rank1 Omega(actual04) tensor Omega(actual04)', 'root_geometry':{'matching_edges':[[0,1],[2,3]],'external_edge':[1,2], 'Dinc':'row minus column; -1 tail,+1 head'},
                                 'source_binary_norm_squared_local':harmonic['source_norm_float'],
                                 'source_normalization':'Each local G/T/U divided by actual binary norm_squared; Omega=actual04/sqrt(N0), pairedOmega=Omega tensor Omega. No combinatorial allocation in B or H.',
                                 'primitive_path_scope':'ONE actual axis0 length1 edge; no unproved cross-path directsum orthogonality or sameground lattice claim',
                                 'physical_roles':[['active',1],['auxiliary',63]], 'canonical_number_per_copy_two_dimers':8,
                                 'all_spinlabels':[0,1,2,3], 'integer_edgeflow':'E=-cumsum(sum_copy(n_site-2)); HE=64 sum_edges E_edge^2 once; HC not added',
                                 'spin_unitary':encode(spin), 'firstshell_t':first['t_all_roles'], 'secondshell_u':second['u_all_roles'],
                                 'literal_firstshell_offsets':{'+':[0.5,0.5],'-':[-0.5,-0.5]},
                                 'literal_secondshell_offsets':peer['graph_interface']['integer_tree_matching_offsets_by_orientation'],
                                 'firstshell_delta_binding':'At qa+1, left cdag atsa1 has rho shift-1/2 and right ann atsb0 has rho shift-1/2; both u+1/2 cancel. Reverseqa-1 gives both shift+1/2, u-1/2. Internal treeE depends only on unchanged site0/site3; direct fullCAR affine identity/adverse retained.',
                                 'normalizer_Z':second['Z'], 'normalizer_scalar':second['normalizer'],
                                 'charged_masks':'P1: |q_dimerA|=1; P2: |q_dimerA|=2, total fixed canonical number; neutral changed labels retained in Q',
                                 'complete_operator':'H=sum_copy dGamma(actual onsite and matching+external onebody matrix)+64 sum_3edges(sum_copy integerflow)^2',
                                 'unproved_hypotheses':['roundoff enclosure of new contracted moments','global highercharge/neutral-Q spectral denominator','all-lattice connected remainder','physicalPhi derivative action','same full ground','N-uniform phase and sharp photon']},
              'boundaries':{'finite_two_dimer_rooted_H_complete':True,'finite_trial_not_ground':True,'full_lattice_H_action':False,
                            'fullPhi_globalground_allorder_phase_stiffness_IRatom_sharpphoton_commoncone_originalSOURCE':'OPEN'},
              'prior_failed_envelopes_not_global_no_go':[6.4621797912465375,5.666659770035231,3.1510030280826653],
              'legacy_action_diagnostic_status':'Old second-shell opposite-offset T2/U2 and their fitted u/residual/energy are WITHDRAWN; heat/Feshbach physical diagnostics HELD pending parent literal-tree correction. Historical envelope numbers are not current phase or physical-H certificates.',
              'elapsed_seconds':time.monotonic()-start}
    assert captured == {name:sha(ROOT/name) for name in PINS}
    exclusive_json(ROOT/f'normalized_root_graph_{args.epoch}_receipt.json', result)
    print(json.dumps({'epoch':args.epoch,'first_Z':first['Z'],'first_t':first['t_all_roles'],
                      'first_complete_residual_bound':full,'second_Z':second['Z'],
                      'second_complete_residual_bound':second['full_HU_minus_Ulambda_norm_upper_float'],
                      'natural_main_finished':True,'phase':False},indent=2),flush=True)

if __name__ == '__main__':
    main()

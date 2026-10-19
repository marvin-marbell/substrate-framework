"""Independent frozen-body contrary cases; no producer/supplier imports or root calls."""
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import expm, expm_frechet
from threadpoolctl import threadpool_limits

input_path = Path(sys.argv[1])
raw = input_path.read_bytes()
data = json.loads(raw)
t = float(F(data['angle']))
r = float(F(data['loop_r']))
z = 1 + r*r/2
qg0 = F(data['g0_centre_exact_rational'])
qgs = F(data['gs_centre_exact_rational'])
g0, gs = float(qg0), float(qgs)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1., -1.]).astype(complex)
# An independently chosen irreducible Clifford representation; trace invariants
# are unchanged by its common unitary equivalence to the selected native frame.
gamma = np.array([np.kron(Z, I2), np.kron(X, X), np.kron(X, Y), np.kron(X, Z)])
frame = gamma * np.sqrt([g0, gs, gs, gs])[:, None, None]
I = np.eye(16)
p0 = np.kron(np.eye(4), (np.eye(4)-gamma[0])/2)
occupied = np.eye(16)[:, np.diag(p0).real > .5]
links = [(0, 1, 1), (1, 3, 2), (2, 3, 1), (0, 2, 2)]


def exterior_columns(angles, derivatives=False):
    U = np.eye(16, dtype=complex)
    dU = np.zeros((4, 16, 16), complex)
    for edge, (a, b, axis) in enumerate(links):
        S = np.zeros((16, 16), complex)
        dS = np.zeros_like(S)
        block = np.exp(1j*angles[edge])*1j*gamma[0]@gamma[axis]
        aa, bb = slice(4*a, 4*a+4), slice(4*b, 4*b+4)
        S[aa, bb], S[bb, aa] = block, -block.conj().T
        dS[aa, bb], dS[bb, aa] = 1j*block, 1j*block.conj().T
        if derivatives:
            gate, dg = expm_frechet(-t*S, -t*dS)
            dU = np.array([gate@du for du in dU])
            dU[edge] += dg@U
        else:
            gate = expm(-t*S)
        U = gate@U
    W = U@occupied
    return W, U, dU


def local_matter(profile, phi):
    h = np.kron(np.eye(4), float(F(profile['onsite']))*frame[0]).astype(complex)
    for edge, (a, b, axis) in enumerate(links):
        path = next(v for v in profile['paths'] if int(v['axis']) == axis-1 and int(v['length']) == 1)
        hop = np.exp(1j*(phi if edge == 3 else 0))*(float(F(path['c0']))*frame[0]+1j*float(F(path['ci']))*frame[axis])
        aa, bb = slice(4*a, 4*a+4), slice(4*b, 4*b+4)
        h[aa, bb] += hop
        h[bb, aa] += hop.conj().T
    return h


with threadpool_limits(limits=1):
    phi = 2*np.pi*np.arange(data['controls']['new_phi_grid'])/data['controls']['new_phi_grid']
    amplitude = (1+r*np.cos(phi))/np.sqrt(z)
    weight = amplitude**2
    onsite_trace, matter_active, matter_aux, metric, ph, berry = [], [], [], [], [], []
    defects = []
    for angle in phi:
        W, U, dU = exterior_columns([0., 0., 0., angle], derivatives=True)
        P = W@W.conj().T
        R = np.array([-1j*U.conj().T@du for du in dU])
        means = np.array([np.trace(p0@v).real for v in R])
        variances = np.array([np.trace(p0@v@(I-p0)@v).real for v in R])
        onsite_trace.append(np.trace(P@np.kron(np.eye(4), frame[0])).real)
        matter_active.append(np.trace(P@local_matter(data['profiles'][1], angle)).real)
        matter_aux.append(np.trace(P@local_matter(data['profiles'][2], angle)).real)
        metric.append(64*variances.sum()+64**2*np.square(means).sum())
        ph.append(np.trace((I-p0)@P).real)
        berry.append(max(abs(means)))
        defects.append(np.max(abs(W.conj().T@W-np.eye(8))))
    onsite_trace = np.array(onsite_trace)
    matter_active, matter_aux = np.array(matter_active), np.array(matter_aux)
    metric, ph = np.array(metric), np.array(ph)
    q_active, q_aux = F(data['profiles'][1]['onsite']), F(data['profiles'][2]['onsite'])
    delta = q_aux-q_active
    paths = data['profiles'][1]['paths']
    hopping_sum = sum(abs(float(F(v['c0'])))*np.sqrt(g0)+abs(float(F(v['ci'])))*np.sqrt(gs) for v in paths)
    on_a, on_b = float(q_active)*np.sqrt(g0), float(q_aux)*np.sqrt(g0)
    corrected_prices = []
    for old in data['selected_observations']:
        V, cells = old['V'], old['V']//4
        M = 2*V*(1*(2*on_a+2*hopping_sum)+63*(2*on_b+2*hopping_sum))
        new_a1 = cells*np.mean(matter_active+63*matter_aux)+64*cells*np.mean(metric)
        new_imposed = cells*np.mean(weight*(matter_active+63*matter_aux))+64*cells*np.mean(weight*metric+4*r*r*np.sin(phi)**2/z)
        corrected_prices.append({
            'N': old['N'], 'actual_matter_energy_A1': float(cells*np.mean(matter_active+63*matter_aux)),
            'actual_energy_relative_neutral_A1': float(new_a1),
            'actual_energy_relative_neutral_imposed': float(new_imposed),
            'actual_imposed_same_H_cost': float(new_imposed-new_a1),
            'old_minus_actual_A1_energy': float(old['same_H_A_equals_1_control']['complete_body_energy_relative_neutral_eigenenergy']-new_a1),
            'actual_onsite_reference': float(-2*V*(on_a+63*on_b)),
            'species_weighted_matter_action_triangle_bound': float(M),
            'species_weighted_variance_upper_A1': float((M+64*V*(512*np.sin(t)+261632*np.sin(t)**2))**2),
            'species_weighted_variance_upper_imposed': float((M+old['electric_action_norm_bound'])**2),
            'lower_bound_unchanged': old['complete_H_variance_lower_bound'],
            'active_only_matter_receipt_reconstruction_error': float(abs(64*cells*np.mean(weight*matter_active)-old['matter_energy']))
        })
    # Generic disconnected Wick control: rephase each normalized copy by exp(i phi).
    phase_increment = 64*4*64**2
    incorrectly_independent_increment = 64*4*64
    # Ordered native path support control: enumerate all forward AND reverse paths
    # before selecting a cell pair. All four vertical paths contribute coherently
    # to one 16x16 block; no other native path may be ignored inside that sector.
    sector_controls = []
    for N in (4, 6):
        sites = list(np.ndindex((N, N, N)))
        xy = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for za, zb in ((0, 1), (N-1, 0)):
            aa = [(x, y, za) for x, y in xy]
            bb = [(x, y, zb) for x, y in xy]
            all_matches = []
            for site in sites:
                for v in paths:
                    axis, length = int(v['axis']), int(v['length'])
                    target = list(site); target[axis] = (target[axis]+length)%N; target = tuple(target)
                    seam = -1 if site[axis]+length >= N else 1
                    if site in aa and target in bb:
                        all_matches.append((aa.index(site), bb.index(target), axis, length, seam, False, v))
                    if target in aa and site in bb:
                        all_matches.append((aa.index(target), bb.index(site), axis, length, seam, True, v))
            hab = np.zeros((16, 16), complex)
            angle_list = data['controls']['z_pair_angles'][0 if za == 0 else 1]
            for a, b, axis, length, seam, reverse, v in all_matches:
                block = seam*np.exp(1j*angle_list[a])*(float(F(v['c0']))*frame[0]+1j*float(F(v['ci']))*frame[axis+1])
                if reverse: block = block.conj().T
                hab[4*a:4*a+4, 4*b:4*b+4] += block
            Wa = exterior_columns(data['controls']['z_pair_angles'][0])[0]
            Wb = exterior_columns(data['controls']['z_pair_angles'][1])[0]
            Pa, Pb = Wa@Wa.conj().T, Wb@Wb.conj().T
            hs = float(np.linalg.norm((I-Pa)@hab@Pb))
            zs = next(v for v in paths if int(v['axis']) == 2 and int(v['length']) == 1)
            co, ci = F(zs['c0']), F(zs['ci'])
            bracket = np.sqrt(8*float(ci)**2*gs)-64*t*np.sqrt(float(co)**2*g0+float(ci)**2*gs)
            sector_controls.append({'N': N, 'z_pair': [za, zb], 'complete_block_path_count': len(all_matches), 'path_axis_length_set': sorted({(p[2], p[3]) for p in all_matches}), 'complete_block_transition_HS': hs, 'claimed_transition_lower': float(bracket), 'passes_lower': bool(hs >= bracket)})
    hop = gamma[0]+1j*gamma[1]
    output = {
        'kind': 'INDEPENDENT_MATH_CONTROL_NOT_PRODUCER_RERUN',
        'input_sha256': hashlib.sha256(raw).hexdigest(), 'python': sys.executable, 'numpy': np.__version__,
        'grid': len(phi), 'exact_aux_minus_active_onsite': str(delta),
        'g0_float': g0, 'gs_float': gs,
        'exact_lower_bracket_positive': bool(31*ci*ci*qgs > co*co*qg0),
        'Clifford_defect': float(max(np.max(abs(a@b+b@a-(2*np.eye(4) if i == j else 0))) for i, a in enumerate(gamma) for j, b in enumerate(gamma))),
        'exterior_gram_max_defect': float(max(defects)), 'Berry_max': float(max(berry)),
        'A1_contractible_PH_character': float(np.mean(ph*np.cos(phi))),
        'one_shared_amplitude_norm': float(np.mean(weight)),
        'wrong_per_species_amplitude_norm': float(np.mean(amplitude**128)),
        'scalar_gradient_energy_per_cell': float(64*np.mean(4*r*r*np.sin(phi)**2/z)),
        'phase_per_copy_control_per_cell': {'correct_shared_energy_increment': phase_increment, 'wrong_independent_increment': incorrectly_independent_increment, 'missing_disconnected_increment': phase_increment-incorrectly_independent_increment},
        'nonnormal_hop_control': {'equal_components_operator_norm': float(np.linalg.norm(hop, 2)), 'incorrect_euclidean_norm': float(np.sqrt(2)), 'correct_sum_norm': 2., 'commutator_norm': float(np.linalg.norm(hop@hop.conj().T-hop.conj().T@hop, 2))},
        'species_corrected_prices': corrected_prices,
        'complete_adjacent_z_sector_controls': sector_controls,
        'decisive_findings': {
            'active_onsite_not_aux_onsite': bool(delta != 0),
            'every_actual_energy_differs_from_64_active_receipt': all(v['old_minus_actual_A1_energy'] > 1 for v in corrected_prices),
            'cheaper_A1_survives_species_correction': all(v['actual_imposed_same_H_cost'] > 0 for v in corrected_prices),
            'full_H_adjacent_z_lower_survives_all_native_paths_and_AP': all(v['passes_lower'] and v['complete_block_path_count'] == 4 for v in sector_controls)
        },
        'scope': 'Frozen Clifford/identitySU4 beta0 body only. Float observations, not interval enclosures or original full-law lift.'
    }
print(json.dumps(output, indent=2))
# Natural exit 0 means the discriminating control reached its mathematical claims.
if not all(output['decisive_findings'].values()):
    sys.exit(2)

"""Ground-directed neutral band128 lift and untruncated charged/electric action.

Fixed comparison section only. FullPhi differential debt is separately priced;
no shell is promoted to a finite Hamiltonian or physical photon pole.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
from scipy.linalg import block_diag, expm
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field
import joined_phi_u1_ground_photon_collective_graph as previous

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_neutral_ground_graph_'
G = [F('0.28917434559291342'), *[F('0.10475219290309079')]*3]
Flow = tuple[tuple[int, int], ...]
Vertex = tuple[tuple[tuple[int, int], ...], Flow]

PROOF = {
    'neutral': 'Assemble the literal PLUS native256 h, including every onsite, ordered one/two-link and AP seam. Its top128 eigenvectors Wplus define Pplus and the normalized wedge128 state. Physical Hneutral=-Lambda dGamma(h). Full fixedN128 exterior CAR action is E=-Lambda Tr(Pplus h), residual squared=Lambda^2 ||Wminus* h Wplus||HS^2. In exact spectral arithmetic this is ZERO. This is all128 globally occupied eigenvalues, not selected-loop Ritz or fixed2 per momentum occupation prescription.',
    'full_graph': 'Neutral band rotation is a unitary transformation of ALL256 neutral CAR modes. Arbitrary masks of128 band modes remain, so no neutral occupation tail is discarded. Charged graph retains active128 plus63 auxiliary copies each128, Ref4, collective rho on64 sites, all129 electric characters, incomingPi and every native hop. The evaluated seed charged masks are the preceding complementary local baryons, not a charged ground. All64 charged H actions and the exact preceding rational residual survive unchanged; the neutral Lambda^2 residual is removed.',
    'correlated_fork': 'A NEW normalized superposition of the seed and TWO distinct auxiliary hops, with the sum of BOTH electric characters and collective density, not the preceding first-action meson state. Complete H action is summed over EVERY species and every native hop, including onsite CAR terms, on both input vertices. Output includes third-copy and same-copy escapes, and all destination amplitudes interfere before squaring. The support of Hpsi is not identified with the two input vertices; no finite-H substitution. This trial tests electric connected weight, full residual, current/contact and distinct-aux collective energies, not a spectral pole.',
    'derivative': 'Local isolated projector Pplus(Phi,U) is smooth and gauge equivariant. For an occupied/unoccupied gap delta, its off-band derivative is the Sylvester solution X_ai=(Wminus* dh Wplus)_ai/(eplus_i-eminus_a); the wedge horizontal first-derivative norm squared is Tr X*X. For a unit coordinate, ||psi_j||<=sqrt128 b_j/delta. A conservative horizontal-wedge second derivative bound is sqrt128 c_j/delta+(8sqrt128+512)b_j^2/delta^2; it follows by differentiating the Sylvester equation and wedge, retaining its scalar and double-PH terms. Ambient Cart3840 and Haar2880 are ALL retained, with T=M/sqrt8 and TrTT=1/2. FullH acting on scalar-envelope times the lift includes -4chi sum(psi f_jj+2 f_j psi_j+f psi_jj) and -(3chi/20)sum Haar analogues, plus VJ/alignment/plaquette. This consumer prices lift-only second derivative by a uniform analytic sum; it does NOT set envelope derivatives to zero or claim a normalized fullPhi wave.',
    'gauss': 'A local gauge-equivariant determinant-line section can be constructed by wedge(Pplus) applied to an invariant baryon reference wherever its compression is nonsingular. Charged local baryons and Ref4 are singlets. The direct Wplus wedge is a normalized local section, not a globally trivialized determinant line. Finite SU4 covariance of h/P is checked independently; compact chart envelope, exact scalar normalization and global Gauss wave remain unpaid.',
    'photon': 'Bounded transverse electric Weyl uses a physical spatial cosine-weighted plaquette flow, not configuration beta. Connected weight on the correlated state is computed from its TWO characters. The Theta-Peierls derivative h_k(a)=h_k exp(i a v.k) gives phase current/contact with HE unchanged. SEPARATELY Pi translation Pi->Pi+a v gives electric-energy derivatives128 v.Pi and128|v|^2. These are independent parameters, never summed into one phase current. No high-frequency enclosure or pole is constructed.',
    'scope': 'NEW Jordan changes only the bosonic quartic. At frozen fields its value/alignment/plaquette are common subtracted constants; field and Haar differential terms are NOT silently subtracted from a fullH residual. No root/mean solve, source adoption, covariance adoption, vfit or coefficient change.',
    'character_lattice': 'Vertices store literal integer192 edge flows k, not fictitious integer129 orthonormal Theta coordinates. Actual reduced character is Q^T k and Pi shift is Q Q^T k; it lives in the projected lattice/coset. Equivalently saturated cycle coordinates t have integer C^T k characters, with Theta=(C^T Q)^T t. For fixed occupation changes Dk=rho, if two k have the same reduced character then their difference is both transverse-zero and D-zero, hence zero; these evaluated keys do not split one physical character into spurious copies.',
    'numerical_state': 'The EXACT neutral state is the spectral functional Pplus=1_(0,infinity)(literal full-native h) followed by wedge128. Stored numerical Wplus/Pplus are floating witnesses, NOT exact projector bytes and NOT an admitted small physical residual after multiplying by Lambda. The nonzero binary off-band defect/eigensolver error are explicitly reported; exact residual0 is the spectral-identity result for the defined exact projector. FullPhi/Haar/Berry and scalar-envelope residuals remain open.',
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def add_flow(a: Flow, b: Flow) -> Flow:
    values = dict(a)
    for edge, value in b:
        values[edge] = values.get(edge, 0)+value
    return tuple(sorted((edge, value) for edge, value in values.items() if value))


def dense_flow(k: Flow) -> np.ndarray:
    out = np.zeros(192)
    for edge, value in k:
        out[edge] = value
    return out


def terms(profile: dict[str, Any], native: dict[str, Any]) -> list[tuple[int, int, Flow, complex]]:
    out = []
    for x in range(64):
        onsite = float(profile['onsite'])*native['frame'][0]
        for i in range(4):
            for j in range(4):
                if onsite[i, j] != 0:
                    out.append((4*x+i, 4*x+j, (), complex(onsite[i, j])))
    for path in field.native_paths(profile, native):
        hop = np.array([[complex(float(r), float(i)) for r, i in row] for row in previous.exact_hop(path, native['frame'])])
        x, y = int(path['source']), int(path['target'])
        k = tuple((int(edge), 1) for edge in sorted(path['edges']))
        back = tuple((edge, -value) for edge, value in k)
        for i in range(4):
            for j in range(4):
                if hop[i, j] != 0:
                    out.append((4*x+i, 4*y+j, k, complex(hop[i, j])))
                    out.append((4*y+j, 4*x+i, back, complex(hop[i, j].conjugate())))
    return out


def local_action(mask: int, entries: list[tuple[int, int, Flow, complex]]) -> list[tuple[int, Flow, complex]]:
    result: dict[tuple[int, Flow], complex] = defaultdict(complex)
    for a, b, k, amplitude in entries:
        moved = previous.car_source.car(mask, a, b)
        if moved is not None:
            dest, sign = moved
            result[(dest, k)] += sign*amplitude
    return [(dest, k, value) for (dest, k), value in result.items() if value != 0]


def complete_action(state: dict[Vertex, complex], base: list[int], all_terms: list[list[tuple[int, int, Flow, complex]]], incoming: np.ndarray) -> dict[Vertex, complex]:
    result: dict[Vertex, complex] = defaultdict(complex)
    cache: dict[tuple[int, int], list[tuple[int, Flow, complex]]] = {}
    for (changes, k), amplitude in state.items():
        changed = dict(changes)
        flow = dense_flow(k)
        # Dk=rho exactly; Q Q* + D*Lplus D=I and incoming is transverse.
        energy = 64*float(np.dot(incoming+flow, incoming+flow))
        result[(changes, k)] += energy*amplitude
        for species in range(64):
            mask = changed.get(species, base[species])
            role = 0 if species == 0 else 1
            ck = (role, mask)
            if ck not in cache:
                cache[ck] = local_action(mask, all_terms[role])
            for dest, shift, value in cache[ck]:
                next_changed = dict(changed)
                if dest == base[species]:
                    next_changed.pop(species, None)
                else:
                    next_changed[species] = dest
                key = (tuple(sorted(next_changed.items())), add_flow(k, shift))
                result[key] += amplitude*value
    return {key: value for key, value in result.items() if value != 0}


def run(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    names = ['induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json', 'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json', 'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json', 'joined_phi_u1_ground_photon_collective_delivery_receipt_20261010.json', 'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.json', 'joined_phi_u1_stationary_jordan_parent_packet_20261010.json', 'joined_phi_u1_matrix_jordan_supplier_receipt_20261010.json']
    inputs = {name: (ROOT/name).read_bytes() for name in names}
    data = field.load_mean(g_override=[float(g) for g in G], mean_descriptor=json.loads(inputs[names[2]])['input_descriptor'])
    native, geom = data['native'], data['geom']
    qmat, d, lp = native['Q'], native['D'], native['Lplus']
    pt = qmat@qmat.T
    fields = np.broadcast_to(native['frame'], (64, 4, 4, 4)).copy()
    links = np.broadcast_to(np.eye(4, dtype=complex), (192, 4, 4)).copy()
    h = -field.assemble(data['profiles'][0], native, fields, links)
    e, u = np.linalg.eigh(h)
    wm, wp = u[:, :128], u[:, 128:]
    p = wp@wp.conj().T
    gap = float(e[128]-e[127])
    assert e[127] < 0 < e[128] and gap > 1
    off = wm.conj().T@h@wp
    residual = float(np.linalg.norm(off)**2)
    eig_error = float(np.linalg.norm(h@u-u*e, 2))
    unitary_error = float(np.linalg.norm(u.conj().T@u-np.eye(256), 2))
    energy = -float(e[128:].sum())
    wrong_sheet_energy = -float(e[:128].sum())
    assert wrong_sheet_energy-energy > 100
    # Wrong occupied rotated mode is NOT merely the other stationary sheet.
    angle = .23
    wrongw = wp.copy()
    wrongw[:, 0] = math.cos(angle)*wp[:, 0]+math.sin(angle)*wm[:, -1]
    wrongp = wrongw@wrongw.conj().T
    wrong_residual = float(np.trace(wrongp@h@(np.eye(256)-wrongp)@h).real)
    predicted = math.sin(angle)**2*math.cos(angle)**2*gap**2
    assert abs(wrong_residual-predicted) < 1e-8 and wrong_residual > .1
    # Independent noncommuting finite local SU4 assembly, not just conjugating P.
    t = geom['M']/np.sqrt(8)
    angles = .21*np.cos(2*np.pi*native['sites'][:, 0]/4)
    gauges = np.array([expm(1j*(a*t[1]+.13*t[4])) for a in angles])
    rotated_fields = gauges[:, None]@fields@gauges[:, None].conj().transpose(0, 1, 3, 2)
    lookup = {tuple(map(int, pos)): x for x, pos in enumerate(native['sites'])}
    rotated_links = []
    for x, pos in enumerate(native['sites']):
        for axis in range(3):
            dest = pos.copy(); dest[axis] = (dest[axis]+1)%4
            y = lookup[tuple(map(int, dest))]
            rotated_links.append(gauges[x]@gauges[y].conj().T)
    hg = -field.assemble(data['profiles'][0], native, rotated_fields, np.asarray(rotated_links))
    bigg = np.asarray(block_diag(*gauges))
    eg, ug = np.linalg.eigh(hg)
    pg = ug[:, 128:]@ug[:, 128:].conj().T
    gauge_h_error = float(np.linalg.norm(hg-bigg@h@bigg.conj().T))
    gauge_p_error = float(np.linalg.norm(pg-bigg@p@bigg.conj().T))
    assert max(gauge_h_error, gauge_p_error) < 1e-9
    # A local equivariant section exists: projection of invariant even baryon.
    even = previous.mask_on(native['sites'], 0)
    odd = previous.mask_on(native['sites'], 1)
    selected = [j for j in range(256) if (even >> j)&1]
    compression = np.linalg.eigvalsh(p[np.ix_(selected, selected)])
    # May fail at this exact Clifford section; preserve, never divide by zero.
    projected_reference_valid = bool(compression[0] > 1e-12)
    derivative_grams, derivative_rows = [], []
    for qi in field.QINTEGERS:
        row = field.build_species(data['profiles'][0], native, geom, qi)
        x = -row['PH_plus']/row['gap_plus'][:, None]
        gram = np.einsum('kaij,kbij->ab', x.conj(), x)
        derivative_grams.append(gram)
        derivative_rows.append(dict(q_integer=list(map(int, qi)),normalPhi45_gradient_squared=float(np.trace(gram[:45, :45]).real),colour45_gradient_squared=float(np.trace(gram[45:, 45:]).real),PC_norm=float(np.linalg.norm(gram[:45, 45:])),minimum_ph_gap=float(row['gap_plus'].min())))
    assert sum(r['normalPhi45_gradient_squared'] for r in derivative_rows) > 0
    # Conservative FULL ambient derivative bounds from literal finite primitive.
    profile = data['profiles'][0]
    raw_sum = abs(F(profile['onsite']))+2*sum((sum((abs(F(complex(c).real))+abs(F(complex(c).imag)) for c in coefficients), F(0)) for _, coefficients in profile['raw_paths']), F(0))
    # Each endpoint/path has <=2 links; unit Cart/SU4 T has opnorm<1.
    b = 8*raw_sum
    c = 32*raw_sum*(1+sum((F(abs(float(z.real)))+F(abs(float(z.imag))) for z in native['frame'].ravel()), F(0)))
    # Independent outward saved-parameter gap supplier; no floating gap claim.
    bound = field.exact_operator_bounds(profile, data['g'])
    # Reserve an explicit binary sqrt/assembly price instead of comparing an
    # exact algebraic lower endpoint to an unpriced floating eigensolver value.
    binary_gap_price = raw_sum/F(2**30)
    gap_lower = F(bound['minimum_gap_unscaled']['lower'])-binary_gap_price
    assert gap_lower > 0 and float(gap_lower) <= gap, (str(gap_lower), gap)
    delta = gap_lower/2
    first_bound = F(12)*b/delta
    second_bound = F(12)*c/delta+F(608)*b*b/(delta*delta)
    differential_bound = (F(4*3840)+F(3*2880, 20))*second_bound
    tube_radius = min(F(1, 2**20), delta/(4*b))
    # Actual all64 charged exact graph action survives neutral adaptation.
    charged_shells = [previous.shell(data['profiles'][1], native, even, True), previous.shell(data['profiles'][2], native, even, True), previous.shell(data['profiles'][2], native, odd, True)]
    charged_exact = sum((n*s['norm'] for n, s in zip((1, 31, 32), charged_shells)), F(0))
    old = json.loads(inputs[names[4]])
    assert charged_exact == F(old['seed_shell']['charged_norm_squared_exact'])
    assert np.array_equal(native['frame'], np.load(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz')['Phi_binary_section'])
    inputs['joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz'] = (ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz').read_bytes()
    incoming = .37*pt[:, 0]
    assert np.linalg.norm(d@incoming) < 1e-12 and qmat.shape == (192, 129)
    # Material fork has a pair of distinct auxiliary changes AND electric sum.
    loops = previous.plaquettes(native)
    transverse_test = np.cos(2*np.pi*native['sites'][:, 0]/4)@loops/8
    witness = max(charged_shells[1]['rows'], key=lambda r: abs(float(transverse_test@r['k'])))
    k_one: Flow = tuple((int(j), int(witness['k'][j])) for j in np.flatnonzero(witness['k']))
    double_k = add_flow(k_one, k_one)
    vertex0: Vertex = ((), ())
    vertex2: Vertex = (((1, int(witness['mask'])), (2, int(witness['mask']))), double_k)
    lam = .27
    state = {vertex0: complex(1/math.sqrt(1+lam*lam)), vertex2: complex(lam/math.sqrt(1+lam*lam))}
    base = [even]*32+[odd]*32
    entries = [terms(data['profiles'][1], native), terms(data['profiles'][2], native)]
    baseline_action = complete_action({vertex0: 1+0j}, base, entries, incoming)
    baseline_residual = math.fsum(abs(z)**2 for key, z in baseline_action.items() if key != vertex0)
    baseline_error = abs(baseline_residual-float(charged_exact))
    assert baseline_error < 1e-7
    del baseline_action
    action = complete_action(state, base, entries, incoming)
    expected_energy = sum(value.conjugate()*action.get(key, 0) for key, value in state.items())
    assert abs(expected_energy.imag) < 1e-10
    action_norm = math.fsum(abs(z)**2 for z in action.values())
    variance = action_norm-expected_energy.real**2
    assert variance > 1
    exterior_norm = math.fsum(abs(z)**2 for key, z in action.items() if key not in state)
    triple_norm = math.fsum(abs(z)**2 for (changes, _), z in action.items() if len(changes) >= 3)
    assert triple_norm > 1 and exterior_norm > 1
    full_flow = dense_flow(double_k)
    rho = d@full_flow
    pi = incoming+pt@full_flow
    proper_energy = 64*(pi@pi+rho@lp@rho)
    full_e = incoming+full_flow
    wrong_independent_energy = 64*(incoming@incoming+2*np.dot(pt@dense_flow(k_one), pt@dense_flow(k_one))+2*(d@dense_flow(k_one))@lp@(d@dense_flow(k_one)))
    assert abs(proper_energy-64*(full_e@full_e)) < 1e-9
    assert abs(proper_energy-wrong_independent_energy) > 1
    wrong_zeroPi_energy = 64*(np.dot(pt@full_flow, pt@full_flow)+rho@lp@rho)
    assert abs(proper_energy-wrong_zeroPi_energy) > 1
    forks = []
    weights = {key: abs(value)**2 for key, value in state.items()}
    for qi in ((1, 0, 0), (2, 0, 0)):
        v = np.cos(2*np.pi*(native['sites']@np.array(qi))/4)@loops/8
        v /= np.linalg.norm(v)
        characteristic = sum(weights[key]*np.exp(1.3j*np.dot(v, incoming+pt@dense_flow(key[1]))) for key in state)
        connected = 1-abs(characteristic)**2
        current: dict[Vertex, complex] = defaultdict(complex)
        contact: dict[Vertex, complex] = defaultdict(complex)
        # Character derivative of each complete transition, not Haction phase.
        for key, amplitude in state.items():
            single = complete_action({key: amplitude}, base, entries, incoming)
            oldflow = dense_flow(key[1])
            for dest, z in single.items():
                shift = dense_flow(dest[1])-oldflow
                phase = float(v@shift)
                current[dest] += 1j*phase*z
                contact[dest] -= phase*phase*z
        translation_current_norm = sum(weights[key]*(128*float(v@(incoming+pt@dense_flow(key[1]))))**2 for key in state)
        forks.append(dict(q_integer=list(qi),connected_weight=float(connected),Theta_Peierls_current_action_norm_squared=math.fsum(abs(z)**2 for z in current.values()),Theta_Peierls_contact_action_norm_squared=math.fsum(abs(z)**2 for z in contact.values()),Pi_translation_current_action_norm_squared=float(translation_current_norm),Pi_translation_contact=128*float(v@v),Pi_translation_contact_action_norm_squared=(128*float(v@v))**2,Pi_translation_incoming_current=128*float(v@incoming),separate_external_parameters=True))
    assert max(r['connected_weight'] for r in forks) > 1e-5
    np.savez_compressed(output.with_suffix('.npz'), h_unscaled_plus=h, Wplus=wp, Wminus=wm, Pplus=p, bands=e, frame=native['frame'], Q=qmat, D=d, Lplus=lp, incomingPi=incoming, neutral_derivative_Gram_all64=np.asarray(derivative_grams), sites=native['sites'])
    captured = dict(field.CAPTURED)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.parent == ROOT:
                captured.setdefault(path.name, path.read_bytes())
    for name, raw in captured.items():
        assert (ROOT/name).read_bytes() == raw
    for name, raw in inputs.items():
        assert (ROOT/name).read_bytes() == raw
    with zipfile.ZipFile(output.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in captured.items():
            archive.writestr('executed_source/'+name, raw)
        for name, raw in inputs.items():
            archive.writestr('inputs/'+name, raw)
        archive.writestr('arrays/'+output.with_suffix('.npz').name, output.with_suffix('.npz').read_bytes())
    result: dict[str, Any]
    result = dict(kind='NEUTRAL_POSITIVE_BAND128_FULL_CHARGED_GRAPH_AND_CORRELATED_TWO_AUX_ELECTRIC_FORK_NOT_FULLGROUND',proof=PROOF,plan=dict(id='issue-222-induced-photon-delta13-continuation-20261008',revision='55352a0aabbcc1da539641239f48f2807ec1e2a609b5d533ca0f186d8df27b6e',template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',work_item='induced-photon'),law=dict(N=4,V=64,charged_species=64,auxiliary=63,each_charged=128,neutral=128,Ref=4,HE=64,HC=64,Pi_modes=129,Lambda='2^8192',chi='2^-4096',sourceactive129_Ref3='Separate unchanged incoming descriptor, NOT ground sector',Jordan='NEW_NOT_SOURCE_ADOPTED'),neutral=dict(energy_over_Lambda=energy,full256_gap_float=gap,exact_spectral_residual_over_Lambda2='0',observed_binary_offband_residual_over_Lambda2=residual,eigensolver_operator_error=eig_error,orthogonality_operator_error=unitary_error,charged_residual_squared_exact=str(charged_exact),old_neutral_residual_over_Lambda2=old['seed_shell']['neutral_norm_squared_over_Lambda2_exact'],wrong_negative_sheet_energy_over_Lambda=wrong_sheet_energy,wrong_mixed_occupation_residual_over_Lambda2=wrong_residual,wrong_mixed_occupation_prediction=predicted,gauge_h_error=gauge_h_error,gauge_p_error=gauge_p_error,baryon_compression_min=float(compression[0]),baryon_projected_reference_nonsingular=projected_reference_valid),derivatives=dict(all64_PH90=derivative_rows,ambient_Cart=3840,ambient_Haar=2880,rigorous_gap_supplier=bound,isolated_tube_radius_exact=str(tube_radius),tube_gap_lower_exact=str(delta),native_unit_first_operator_bound_exact=str(b),native_unit_second_operator_bound_exact=str(c),lift_unit_first_bound_exact=str(first_bound),lift_unit_second_bound_exact=str(second_bound),lift_only_full_kinetic_norm_upper_over_chi_exact=str(differential_bound),fullH_residual='UNPAID: compact scalar normalization/envelope derivatives and mixed envelope-lift terms, VJ/alignment/plaquette variations; lift-only derivative price is not fullH residual',normal90_not_ambient105=True),correlated_fork=dict(input_vertices=2,complete_action_destinations=len(action),energy=float(expected_energy.real),action_norm_squared=action_norm,residual_squared=variance,exterior_action_norm_squared=exterior_norm,three_or_more_distinct_species_escape_norm_squared=triple_norm,proper_collective_energy=float(proper_energy),wrong_independent_copy_energy=float(wrong_independent_energy),incomingPi_retained=True,closed_input_support=False,probe=forks,numerics='Floating complete sparse action at exact archived binary section, not outward variational interval or spectral replacement'),paid_tails=['Exact neutral ALL256 canonical spectral action, no neutral occupation truncation','Exact charged first action on64-species baryon seed from rational source','Complete correlated-state sparse charged action incl all onsite, AP, one/two paths and distinct-aux escapes','Full129 Pi/full64 collective HC identity with incomingPi','All64 conditional PH90 Kato derivative metric; ambient105 derivative bound separately supplied'],unpaid=['Normalized compact fullPhi/Haar/Gauss wave and total differential residual','NEW Jordan root/proper sectors/off-flat global ground','Stationary charged/electric ground and minimizing beta','Low-energy same-ground photon pole/dispersion and spectral tails','originalSOURCE/common cone/Compton'],next_decision='Neutral Lambda^2 action failure is removed by the exact positive band. Correlated two-aux electric trial has nonzero spatial probe weight but retains large charged/exterior action; construct charged/electric stationary correlated state together with compact normalized Phi/Haar envelope, consuming full mixed Gaussian only as a conditional comparison.',source_sha256={name:sha(raw) for name, raw in captured.items()},input_sha256={name:sha(raw) for name, raw in inputs.items()},archive=dict(path=output.with_suffix('.zip').name,sha256=sha(output.with_suffix('.zip').read_bytes())),elapsed_seconds=time.monotonic()-started,runtime=dict(executable=sys.executable,python=sys.version))
    result['derivatives']['binary_gap_price_exact'] = str(binary_gap_price)
    result['neutral']['independent_complete_graph_charged_residual_squared'] = baseline_residual
    result['neutral']['rational_vs_complete_graph_residual_error'] = baseline_error
    result['correlated_fork']['wrong_zero_incomingPi_energy'] = float(wrong_zeroPi_energy)
    result['correlated_fork']['residual_change_vs_neutral_adapted_seed'] = variance-float(charged_exact)
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        receipt = run(args.output)
    print(json.dumps(dict(kind=receipt['kind'],neutral=receipt['neutral'],correlated_fork=receipt['correlated_fork'],elapsed_seconds=receipt['elapsed_seconds']), indent=2))

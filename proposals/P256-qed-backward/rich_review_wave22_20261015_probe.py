"""Independent same-180CAR saved-state consumer; no author consumer imports."""
from __future__ import annotations
import gc
import hashlib
import json
import resource
import time
from itertools import combinations
from pathlib import Path
from typing import Any
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, expm
from scipy.sparse.linalg import expm_multiply
import rich_mobile_packet_20261015_model as primitives

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave22_20261015_'
SUP = 'rich_mobile_packet_20261015_actual1_'
START = time.time()
CONSUMED: dict[str, str] = {}
RESULT: dict[str, Any] = {'kind': 'INDEPENDENT_SAME_180CAR_PHYSICAL_CONSUMER', 'prohibited_imports': 'No calculate/continue/analyze/evolve/observe/review consumer imported', 'propagation_scope': 'NEW t4->4.5 charged incoming, charged baseline, neutral incoming; SAME initial entangled-state EMoff t0->.5; bare field t0->8. Not a replay of all17points/fivearms.'}


def pin(name: str) -> Path:
    path = ROOT / name
    if name not in CONSUMED:
        CONSUMED[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def js(name: str) -> Any:
    return json.loads(pin(name).read_text())


def array(name: str) -> Any:
    return np.load(pin(name))


def save_json(name: str, value: Any) -> None:
    (ROOT / (PREFIX + name)).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checkpoint(phase: str) -> None:
    RESULT['checkpoint_phase'] = phase
    RESULT['elapsed_seconds'] = time.time() - START
    RESULT['peak_resident_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    RESULT['consumed_sha256'] = CONSUMED
    save_json('checkpoint.json', RESULT)
    print(json.dumps({'phase': phase, 'elapsed': RESULT['elapsed_seconds']}), flush=True)


def norm2(x: np.ndarray) -> float:
    return float(np.vdot(x, x).real)


def field_action(x: np.ndarray, op: np.ndarray) -> np.ndarray:
    return (x.reshape(-1, 45) @ op.T).ravel()


def own_onebody(masks: list[int], h: np.ndarray) -> sparse.csr_matrix:
    lookup = {m: k for k, m in enumerate(masks)}
    rows, cols, values = [], [], []
    nonzero = [[(int(i), h[i, j]) for i in np.flatnonzero(abs(h[:, j]) > 1e-13) if i != j] for j in range(180)]
    for col, mask in enumerate(masks):
        occupied = [j for j in range(180) if (mask >> j) & 1]
        diag = sum(h[j, j] for j in occupied)
        if abs(diag) > 1e-14:
            rows.append(col); cols.append(col); values.append(diag)
        for j in occupied:
            removed = mask ^ (1 << j)
            sign = (-1) ** ((mask & ((1 << j) - 1)).bit_count())
            for i, value in nonzero[j]:
                if (removed >> i) & 1:
                    continue
                target = lookup.get(removed | (1 << i))
                if target is not None:
                    rows.append(target); cols.append(col)
                    values.append(value * sign * (-1) ** ((removed & ((1 << i) - 1)).bit_count()))
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(masks), len(masks)))


config = js(SUP + 'input.json')
raw = array(SUP + 'primitive.npz')
initial = array(SUP + 'initial_states.npz')
producer = js('rich_mobile_packet_20261015_actual2_result.json')
# Pure representation helpers are allowed, never producer propagation/measurement.
geo = primitives.geometry(config)
f = primitives.field(config, geo)
U = raw['orbitals']
component = raw['onebody_components']
terms = [sparse.csr_matrix(t) for t in raw['field_terms']]
boson = sparse.csr_matrix(raw['field_boson'])
offset = float(raw['reference_energies'][:90].sum())
RESULT['primitive_binding'] = {k: float(np.max(abs(a-b))) for k, a, b in [
    ('orbitals', U, geo['orbitals']), ('onebody_components', component, geo['transformed']),
    ('field_terms', raw['field_terms'], f['terms']), ('field_boson', raw['field_boson'], f['boson']),
    ('Q', raw['field_q'], f['q']), ('P', raw['field_p'], f['p'])]}
SECTORS: dict[str, Any] = {}
for sector, count in [('neutral', 90), ('positive', 91), ('negative', 89)]:
    masks = [int(m) for m in js(SUP + sector + '_masks.json')['masks']]
    matrices = [sparse.load_npz(pin(SUP + sector + f'_Hcomponent{k}.npz')) for k in range(24)]
    checks = {}
    for k in [0, 1, 22, 23]:
        delta = own_onebody(masks, component[k]) - matrices[k]
        checks[str(k)] = float(np.max(abs(delta.data))) if delta.nnz else 0.0
    SECTORS[sector] = {'masks': masks, 'matrices': matrices, 'N': count}
    RESULT.setdefault('basis', {})[sector] = {'matter_dimension': len(masks), 'joint_dimension': 225*len(masks), 'all_particle_counts_correct': all(m.bit_count() == count for m in masks), 'independent_component_maxdiff': checks}
checkpoint('exact primitive and selected independently reconstructed component bindings')


def H(sec: dict[str, Any], x: np.ndarray) -> np.ndarray:
    y = x.reshape(-1, 225)
    out = (boson @ y.T).T - offset*y
    for m, t in zip(sec['matrices'], terms):
        out += (t @ (m @ y).T).T
    return out.ravel()


def source_map(masks: list[int], target: list[int], orbital: int, create: bool) -> sparse.csr_matrix:
    lookup = {m: k for k, m in enumerate(target)}
    rr, cc, vv = [], [], []
    for k, mask in enumerate(masks):
        if bool((mask >> orbital) & 1) == create:
            continue
        dest = mask ^ (1 << orbital)
        row = lookup.get(dest)
        if row is None:
            raise ValueError('source family failed complete physical action closure')
        rr.append(row); cc.append(k); vv.append((-1)**((mask & ((1 << orbital)-1)).bit_count()))
    return sparse.csr_matrix((vv, (rr, cc)), shape=(len(target), len(masks)))


omega = initial['neutral']
neutral = SECTORS['neutral']; positive = SECTORS['positive']
RESULT['neutral_stationarity'] = {'energy': float(np.vdot(omega, H(neutral, omega)).real), 'residual': float(np.linalg.norm(H(neutral, omega) - producer['initial']['neutral_energy']*omega))}
indices = np.flatnonzero((raw['momenta'][:, 0] == 1) & (raw['momenta'][:, 1] == 0))
source_ops = {side: [source_map(neutral['masks'], SECTORS[sec]['masks'], int(i), create) for i in indices] for side, sec, create in [('add', 'positive', True), ('remove', 'negative', False)]}
source_grams = {}
for side in ['add', 'remove']:
    images = [(op @ omega.reshape(-1, 225)).ravel() for op in source_ops[side]]
    gram = np.array([[np.vdot(x, y) for y in images] for x in images])
    source_grams[side] = gram
    packet = sum(c*x for c, x in zip(initial['source_coefficients'] if side == 'add' else initial['source_coefficients'].conj(), images))
    RESULT.setdefault('sources', {})[side] = {'raw_packet_weight': norm2(packet), 'saved_raw_packet_maxdiff': float(np.max(abs(packet-initial['raw_' + side + '_packet']))), 'gram_maxdiff': float(np.max(abs(gram-array(SUP+'source_grams.npz')['orbital_'+side])))}
    del images, packet
RESULT['sources']['raw_CAR_gram_defect'] = float(np.max(abs(source_grams['add'] + source_grams['remove'].T - np.eye(20))))
np.savez_compressed(ROOT / (PREFIX + 'source_grams.npz'), **source_grams)
# These matrices are sparse literal layer observables, not commutator-defined currents.
density_h = []
flux_h = []
for z in range(5):
    position = np.repeat(geo['cells'][:, 2] == z, 4)
    density_h.append(U.conj().T @ (position[:, None]*U))
    link = np.zeros((180, 180), complex)
    for site, target, axis, level, block in geo['links']:
        if axis == 2 and geo['cells'][site, 2] == z:
            link[4*site:4*site+4, 4*target:4*target+4] += block
    flux_h.append(U.conj().T @ (1j*(link-link.conj().T)) @ U)
spin_h = U.conj().T @ np.kron(np.eye(45), -.5j*geo['cliff'][1]@geo['cliff'][2]) @ U
for sec in [neutral, positive]:
    sec['density'] = [own_onebody(sec['masks'], h) for h in density_h]
    sec['flux'] = [own_onebody(sec['masks'], h) for h in flux_h]
    sec['spin'] = own_onebody(sec['masks'], spin_h)
    sec['momentum_labels'] = np.array([sum(int(raw['momenta'][i, 2]) for i in range(180) if (mask >> i) & 1) % 5 for mask in sec['masks']])


def observe(sec: dict[str, Any], x: np.ndarray) -> dict[str, Any]:
    X = x.reshape(-1, 225); hx = H(sec, x).reshape(X.shape)
    dens = [float(np.vdot(X, n@X).real) for n in sec['density']]
    flux = [float(np.vdot(X, j@X).real) for j in sec['flux']]
    deriv = [float(2*np.vdot(n@X, hx).imag) for n in sec['density']]
    defect = np.array(deriv) - (np.roll(flux, 1)-flux)
    q = [float(np.vdot(x, field_action(x, op)).real) for op in f['q']]
    p = [float(np.vdot(x, field_action(x, op)).real) for op in f['p']]
    shape = (2, 5, 8)
    profiles = np.zeros(shape)
    for band in range(2):
        for pol in range(2):
            profiles[pol, :, 4*band+2*pol] = np.cos(2*np.pi*(band+1)*np.arange(5)/5)
            profiles[pol, :, 4*band+2*pol+1] = np.sin(2*np.pi*(band+1)*np.arange(5)/5)
    A = np.einsum('pza,a->pz', profiles, q)
    E = np.einsum('pza,a->pz', profiles, p)/f['inertia']
    # Rectangular BEFORE-cut P includes intermediate shell3.
    energy = np.zeros((2, 5))
    for pol in range(2):
        for z in range(5):
            pr = np.einsum('a,aij->ij', profiles[pol, z], f['prect'])/f['inertia']
            energy[pol, z] = .5*norm2(field_action(x, pr))
    amplitudes = (x.reshape(-1, 45)@f['translation_vectors'].conj()).reshape(-1, 5, 45)
    weights = np.sum(abs(amplitudes)**2, axis=1)
    joint = np.array([[weights[sec['momentum_labels']==m][:, f['translation_values']==n].sum() for n in range(5)] for m in range(5)])
    # n_z/N is NOT a projection, despite historical PVM naming.
    nscaled = sec['density'][0]/sec['N']
    defect_pvm = float(np.linalg.norm(nscaled@(nscaled@X)-nscaled@X))
    return {'norm2': norm2(x), 'energy': float(np.vdot(X, hx).real), 'number_density_z': dens, 'number_flux_z': flux, 'density_time_derivative': deriv, 'continuity_defect': defect.tolist(), 'continuity_L2': float(np.linalg.norm(defect)), 'whole_sea_Sz': float(np.vdot(X, sec['spin']@X).real), 'A': A.tolist(), 'E': E.tolist(), 'electric_energy_density': energy.tolist(), 'electric_integral_minus_kinetic': float(9*energy.sum()-np.vdot(x, field_action(x, f['kinetic'])).real), 'joint_C5_momentum': joint.tolist(), 'whole_sea_POVM_idempotence_action_defect_z0': defect_pvm}


def load_state(arm: str, index: int) -> np.ndarray:
    d = array(producer['state_paths'][arm+':'+str(index)])
    if float(d['time']) != index*.5:
        raise ValueError('saved-state time binding mismatch')
    return d['state']


RESULT['anchors'] = {}
for arm, sec in [('charged_no_incoming', positive), ('charged_incoming', positive), ('neutral_incoming', neutral), ('charged_same_state_EMoff_no_incoming', positive), ('charged_same_state_EMoff_incoming', positive)]:
    for index in [0, 8, 16]:
        x = load_state(arm, index)
        obs = observe(sec, x)
        # For off arms H here is only a counterfactual coupled-H insertion.
        obs['Hamiltonian_scope'] = 'coupled H; for EMoff anchors derivative/current continuity is counterfactual, not off EOM'
        ref = producer['arms'][arm][index]
        obs['author_density_maxdiff'] = float(np.max(abs(np.array(obs['number_density_z'])-ref['number_density_z'])))
        obs['author_flux_maxdiff'] = float(np.max(abs(np.array(obs['number_flux_z'])-ref['number_flux_z'])))
        obs['author_joint_C5_maxdiff'] = float(np.max(abs(np.array(obs['joint_C5_momentum'])-ref['momentum']['joint'])))
        RESULT['anchors'][arm+':'+str(index)] = obs
        del x
    checkpoint('independent physical anchors '+arm)
RESULT['source_envelope_vs_charge'] = {'source_envelope_z': [float(np.sum(abs(initial['source_physical'].reshape(45, 4)[geo['cells'][:, 2]==z])**2)) for z in range(5)], 'signed_excess_initial': (np.array(RESULT['anchors']['charged_no_incoming:0']['number_density_z'])-np.array(observe(neutral, omega)['number_density_z'])).tolist()}
drive = initial['field_drive']
RESULT['field_only_preparation'] = {'unitarity_defect': float(np.linalg.norm(drive.conj().T@drive-np.eye(45))), 'saved_full_prepared_state_difference': float(np.linalg.norm(field_action(initial['charged'], drive)-initial['charged_field'])), 'matter_density_L2': float(np.linalg.norm(np.array(RESULT['anchors']['charged_incoming:0']['number_density_z'])-RESULT['anchors']['charged_no_incoming:0']['number_density_z'])), 'all_matter_Phi_observables': 'Exact right field-unitary preserves reduced matter/Phi state; full state comparison and unitarity checked, no product state substituted.'}
checkpoint('all selected full-state anchors and receiver semantics')


def lanczos(sec: dict[str, Any], initial_state: np.ndarray, dt: float, dimension: int) -> tuple[np.ndarray, dict[str, float]]:
    # Own physical Krylov implementation, twice-reorthogonalized; no producer routine.
    initial_norm = np.linalg.norm(initial_state)
    vectors = [initial_state/initial_norm]
    alpha, beta = [], []
    tail = 0.0
    for k in range(dimension):
        v = H(sec, vectors[k])
        a = float(np.vdot(vectors[k], v).real)
        alpha.append(a)
        v -= a*vectors[k]
        if k:
            v -= beta[k-1]*vectors[k-1]
        for _ in range(2):
            for basis in vectors:
                v -= np.vdot(basis, v)*basis
        tail = float(np.linalg.norm(v))
        if k == dimension-1 or tail < 1e-13:
            break
        beta.append(tail); vectors.append(v/tail)
    tridiagonal = np.diag(alpha)+np.diag(beta, 1)+np.diag(beta, -1)
    vals, vec = eigh(tridiagonal)
    coefficients = vec @ (np.exp(-1j*dt*vals)*vec[0].conj())
    out = np.zeros_like(initial_state)
    for c, basis in zip(coefficients, vectors):
        out += initial_norm*c*basis
    return out, {'dimension': len(alpha), 'dt': dt, 'endpoint_Schrodinger_residual': float(initial_norm*tail*abs(coefficients[-1])), 'norm2_change': norm2(out)-norm2(initial_state)}


RESULT['fresh_propagation'] = {}
for arm, sec in [('charged_incoming', positive), ('charged_no_incoming', positive), ('neutral_incoming', neutral)]:
    state = load_state(arm, 8)
    # 28 directions/.5 matches supplier resolution only as an adverse/coarse control.
    coarse, coarse_receipt = lanczos(sec, state, .5, 28)
    half, receipt1 = lanczos(sec, state, .25, 32)
    fine, receipt2 = lanczos(sec, half, .25, 32)
    # Further new interval at .125 with independent dimension checks convergence
    # without duplicating all multi-hour arms or replacing saved histories.
    first, receipt3 = lanczos(sec, state, .125, 24)
    tighter, receipt4 = lanczos(sec, state, .125, 32)
    obs = observe(sec, fine)
    RESULT['fresh_propagation'][arm] = {'start_time': 4.0, 'end_time': 4.5, 'coarse': coarse_receipt, 'fine': [receipt1, receipt2], 'fine_vs_coarse_state_L2': float(np.linalg.norm(fine-coarse)), 'quarter_interval_24vs32_state_L2': float(np.linalg.norm(first-tighter)), 'quarter_interval_receipts': [receipt3, receipt4], 'physical': obs, 'author_t4point5_density_L2': float(np.linalg.norm(np.array(obs['number_density_z'])-producer['arms'][arm][9]['number_density_z']))}
    np.savez_compressed(ROOT/(PREFIX+arm+'_new_state4point5.npz'), state=fine, time=4.5)
    np.savez_compressed(ROOT/(PREFIX+arm+'_new_state4point125.npz'), state=tighter, time=4.125)
    del state, coarse, half, fine, first, tighter
    gc.collect(); checkpoint('naturally completed new physical interval '+arm)

# Literal SAME-entangled-state complete EMoff, independently exponentiated.
mm = positive['matrices']
hmatter = mm[0] + sum(mm[2::2])
left = sparse.kron(hmatter, sparse.eye(5), format='csr') + sparse.kron(mm[1], sparse.diags(config['coupling']*f['phi']), format='csr') + sparse.kron(sparse.eye(len(positive['masks'])), sparse.csr_matrix(f['phi_h']), format='csr') - offset*sparse.eye(5*len(positive['masks']), format='csr')
rightU = expm(-.5j*f['kinetic'])
off_states = []
for key in ['charged', 'charged_field']:
    x = initial[key].reshape(-1, 45)
    evolved = expm_multiply(-.5j*left, x, traceA=-.5j*left.diagonal().sum())@rightU.T
    off_states.append(evolved.ravel())
    np.savez_compressed(ROOT/(PREFIX+'same_initial_EMoff_'+key+'_state0point5.npz'), state=evolved.ravel(), time=.5)
off_dens = [[float(np.vdot(x.reshape(-1, 225), n@x.reshape(-1, 225)).real) for n in positive['density']] for x in off_states]
RESULT['fresh_SAME_initial_EMoff'] = {'time': .5, 'density_noincoming': off_dens[0], 'density_incoming': off_dens[1], 'pulse_minus_baseline_L2': float(np.linalg.norm(np.array(off_dens[1])-off_dens[0])), 'full_state_fieldunitary_relation_L2': float(np.linalg.norm(off_states[1]-field_action(off_states[0], rightU@drive@rightU.conj().T))), 'state_factorization': 'NONE; full initial 9134x5x45 entangled coefficient tensor retained', 'owning_transverse_current': 'zero derivative of complete EMoff; counterfactual coupled current is not an off current'}
del off_states, left
checkpoint('new SAME-entangled-state complete EMoff physical control')

# Bare 45-state P^2 control has no inserted magnetic stiffness.
w, v = eigh(f['kinetic'])
bare0 = v[:, 0]
bare = drive@bare0
bare_records = []
for t in [0., .5, 4., 8.]:
    state = v@(np.exp(-1j*t*w)*(v.conj().T@bare))
    q = np.array([np.vdot(state, op@state).real for op in f['q']])
    bare_records.append({'time': t, 'Q': q.tolist(), 'norm2': norm2(state)})
    np.savez_compressed(ROOT/(PREFIX+'bare_state'+str(t).replace('.', 'point')+'.npz'), state=state, time=t)
bright = f['q'][0]@bare0
RESULT['bare_electric'] = {'dimension': 45, 'ground_energy': float(w[0]), 'coordinate_bright_centroid': float(np.vdot(bright, (f['kinetic']-w[0]*np.eye(45))@bright).real/norm2(bright)), 'records': bare_records, 'Q_motion_L2': float(np.linalg.norm(np.array(bare_records[-1]['Q'])-bare_records[0]['Q'])), 'classification': 'finite P-squared oscillator-basis cutoff alone supplies nonzero gap and coordinate/profile motion, not proof for or against induced collective photon energy'}
checkpoint('new bare-electric45 physical gap/crest control')

# Field EOM action defects, not declared zero by finite-cut CCR.
RESULT['field_EOM'] = {}
for label, sec, state in [('neutral_initial', neutral, omega), ('charged_initial', positive, initial['charged']), ('charged_incoming_t8', positive, load_state('charged_incoming', 16))]:
    hx = H(sec, state)
    entries = []
    for axis in [0, 4]:
        qx = field_action(state, f['q'][axis]); px = field_action(state, f['p'][axis])
        qdot = 1j*(H(sec, qx)-field_action(hx, f['q'][axis]))
        pdot = 1j*(H(sec, px)-field_action(hx, f['p'][axis]))
        current = np.zeros_like(state)
        X = state.reshape(-1, 225)
        for k in range(2, 24, 2):
            coefficient = geo['profiles'][k, axis]
            if coefficient:
                current += coefficient*((terms[k+1]@(sec['matrices'][k]@X).T).T-(terms[k]@(sec['matrices'][k+1]@X).T).T).ravel()
        omitted_q = norm2(field_action(state, f['qrect'][axis]))-norm2(qx)
        entries.append({'axis': axis, 'Qdot_minus_PoverI_norm': float(np.linalg.norm(qdot-px/f['inertia'])), 'Pdot_minus_primitive_current_norm': float(np.linalg.norm(pdot-current)), 'coordinate_omitted_action_norm2': omitted_q, 'coordinate_full_action_norm2': norm2(field_action(state, f['qrect'][axis]))})
    RESULT['field_EOM'][label] = entries
checkpoint('independent literal-current and field-cut EOM defects')

# Raw return-source losses are occupancy minus actual retained return norm.
RESULT['reverse_source'] = {}
for label, state in [('initial', initial['charged']), ('incoming_t8', load_state('charged_incoming', 16))]:
    X = state.reshape(-1, 225)
    population = np.sum(abs(X)**2, axis=1)
    records = []
    for j, orbital in enumerate(indices):
        occupancy = float(sum(weight for mask, weight in zip(positive['masks'], population) if (mask >> int(orbital)) & 1))
        returned = norm2(source_ops['add'][j].conj().T@X)
        records.append({'orbital': int(orbital), 'full_removal_norm2': occupancy, 'retained_neutral_return_norm2': returned, 'omitted_raw_weight': occupancy-returned})
    RESULT['reverse_source'][label] = records

# Literal full z-current action beyond selected matter CI, streamed by boson columns.
def outside_current(sec: dict[str, Any], states: list[tuple[str, np.ndarray]]) -> dict[str, Any]:
    h = sum(flux_h)
    retained = own_onebody(sec['masks'], h)
    lookup = set(sec['masks']); outside: dict[int, int] = {}
    rr, cc, vv = [], [], []
    hops = [[(int(i), h[i, j]) for i in np.flatnonzero(abs(h[:, j]) > 1e-13) if i != j] for j in range(180)]
    for col, mask in enumerate(sec['masks']):
        for j in range(180):
            if not ((mask >> j) & 1):
                continue
            removed = mask ^ (1 << j)
            sign = (-1)**((mask & ((1 << j)-1)).bit_count())
            for i, value in hops[j]:
                if (removed >> i) & 1:
                    continue
                dest = removed | (1 << i)
                if dest in lookup:
                    continue
                row = outside.setdefault(dest, len(outside))
                rr.append(row); cc.append(col); vv.append(sign*value*(-1)**((removed & ((1 << i)-1)).bit_count()))
    op = sparse.csr_matrix((vv, (rr, cc)), shape=(len(outside), len(sec['masks'])))
    records = []
    for label, x in states:
        X = x.reshape(-1, 225)
        keep = norm2(retained@X); loss = 0.0
        for start in range(0, 225, 4):
            loss += norm2(op@X[:, start:start+4])
        records.append({'state': label, 'retained_norm2': keep, 'omitted_norm2': loss, 'full_norm2': keep+loss, 'omitted_fraction': loss/(keep+loss)})
    return {'outside_masks': len(outside), 'records': records, 'units': 'Squared number-current action, particles^2/time^2 (hbar=linklength=1); fraction = ||(1-P)Jz psi||^2 / ||Jz psi||^2. Not state/evolution error; no comparison to density L2 or density-derivative defect as same-unit error.'}


RESULT['current_projection'] = {}
RESULT['current_projection']['neutral'] = outside_current(neutral, [('initial', omega)])
checkpoint('independent full omitted neutral-current action')
RESULT['current_projection']['positive'] = outside_current(positive, [('initial', initial['charged']), ('incoming_t8', load_state('charged_incoming', 16))])
checkpoint('independent full omitted charged-current action')

# Explicit selected two-pair C4 retained weights from physical one-body rotation.
sea = (1 << 90)-1
pairs = []
for mask in neutral['masks']:
    particles = [i for i in range(90, 180) if (mask >> i) & 1]
    holes = [i for i in range(90) if not ((mask >> i) & 1)]
    if len(particles) == 2:
        pairs.append((particles, holes))
spin_rotation = expm(-np.pi/4*geo['cliff'][1]@geo['cliff'][2])
physical_rotation = np.zeros((180, 180), complex)
cell_lookup = {tuple(c): k for k, c in enumerate(geo['cells'])}
for i, (x, y, z) in enumerate(geo['cells']):
    j = cell_lookup[((-int(y))%3, int(x), int(z))]
    physical_rotation[4*j:4*j+4, 4*i:4*i+4] = spin_rotation
rotation = U.conj().T@physical_rotation@U
rotation_weights = []
for ps, hs in pairs:
    rotation_weights.append(float(sum(abs(np.linalg.det(rotation[np.ix_(pt, ps)]) * np.linalg.det(rotation[np.ix_(hs, ht)]))**2 for pt, ht in pairs)))
# Choose C by explicit local matrix identities, not import a positron species.
cliff = geo['cliff']; candidates = []
for n in range(5):
    for subset in combinations(range(4), n):
        B = np.eye(4, dtype=complex)
        for i in subset:
            B = B@cliff[i]
        defect = np.linalg.norm(B@cliff[0].conj()@B.conj().T+cliff[0]) + sum(np.linalg.norm(B@cliff[i].conj()@B.conj().T-cliff[i]) for i in range(1, 4))
        candidates.append((defect, subset, B))
Cdefect, Csubset, B = min(candidates, key=lambda item: item[0])
charge_map = U.conj().T@np.kron(np.eye(45), B)@U.conj()
c_weights = []
for ps, hs in pairs:
    c_weights.append(float(sum(abs(np.linalg.det(charge_map[np.ix_(pt, hs)]) * np.linalg.det(charge_map[np.ix_(ht, ps)]))**2 for pt, ht in pairs)))
RESULT['selected_CI_symmetry'] = {'two_pair_count': len(pairs), 'C4_weights': rotation_weights, 'C4_min_retained': min(rotation_weights), 'C4_positive_negative_mixing': float(np.linalg.norm(rotation[:90, 90:])), 'C_local_matrix_identity_defect': float(Cdefect), 'C_Clifford_product': list(Csubset), 'C_two_pair_weights': c_weights, 'C_min_retained': min(c_weights), 'C_opposite_band_failure_norm': float(np.linalg.norm(charge_map[:90, :90])+np.linalg.norm(charge_map[90:, 90:])), 'scope': 'Local antiunitary first-quantized B conjugation swaps energy signs; associated particle-hole Fock map. Selected neutral two-pair retained weight only; no paired Nplus/Nminus spectrum, field-C law completion or protected particle claim.'}
RESULT['preserved_execution'] = {'timeout': js('rich_mobile_packet_20261015_actual1_timeout_exit.json'), 'continuation': js('rich_mobile_packet_20261015_actual2_exit.json'), 'analysis': js('rich_mobile_packet_20261015_analysis_exit.json'), 'baseline_Lanczos_endpoint_records': 'UNKNOWN/unavailable, never zero; own local interval comparison does not recover baseline history', 'new_science': 'RichPhotonCollective unbuilt work is separate, not this review scope or gated by this review'}
RESULT['elapsed_seconds'] = time.time()-START
RESULT['peak_resident_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
RESULT['consumed_sha256'] = CONSUMED
save_json('result.json', RESULT)
checkpoint('complete independent same-model physical consumer')

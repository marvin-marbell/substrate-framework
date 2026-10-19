"""Independent frozen-law consumers; no producer main, repair, or state splice."""
from __future__ import annotations
import hashlib
import json
from math import factorial
from pathlib import Path
import sys
import zipfile
import numpy as np
from scipy.linalg import eigh, expm
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import expm_multiply, norm as sparse_norm
from rich_extensive_photon_20261014_model import matter, whole_h, molecular_h, vertices, line_integrals, polarizations, selfconsistent, collective
from rich_physical_noether_20261014_model import build, shifted_h, oscillator, geometry_quadrature, electromagnetic_profile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value):
    return float(sparse_norm(value)) if hasattr(value, 'tocsr') else float(np.linalg.norm(value))


def mean(state, operator):
    return float(np.vdot(state, operator @ state).real)


def plain(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return dict(real=value.real.tolist(), imag=value.imag.tolist())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def embed(vector, indices, dimension):
    state = np.zeros(dimension, complex)
    state[indices] = vector
    return state


def fock_rotation(osc, rotation):
    """Actual polynomial rotation of creation coordinates, not exp(generator)."""
    creation = [operator.conj().T for operator in osc['a']]
    columns = []
    for occupation in osc['basis']:
        state = np.eye(len(osc['basis']), dtype=complex)[:, 0]
        for axis, count in enumerate(occupation):
            operator = sum(rotation[i, axis] * creation[i] for i in range(3))
            for _ in range(count):
                state = operator @ state
            state /= np.sqrt(factorial(count))
        columns.append(state)
    return np.column_stack(columns)


def pvm(state, model):
    eig = [eigh(model[name]['L'][2]) for name in ['geometry', 'field']]
    sv, su = eigh(model['frame']['S'][2])
    amplitude = np.einsum('ai,abgf,gj,fk->ibjk', su.conj(), state.reshape(16, model['nb'], model['ng'], model['nf']), eig[0][1].conj(), eig[1][1].conj(), optimize=True)
    probabilities = np.sum(abs(amplitude) ** 2, axis=1)
    joint, marginal = {}, {}
    for i, spin in enumerate(sv):
        for j, orbital in enumerate(eig[0][0]):
            for k, field in enumerate(eig[1][0]):
                probability = float(probabilities[i, j, k])
                label = ','.join(f'{x:.1f}' for x in [spin, orbital, field])
                joint[label] = joint.get(label, 0.) + probability
                label = f'{spin + orbital + field:.1f}'
                marginal[label] = marginal.get(label, 0.) + probability
    return dict(joint=joint, Jz=marginal, sum=float(probabilities.sum()))


def binding_audit(root, request):
    evidence = []
    for name, expected in request['bindings'].items():
        actual = sha(root / name)
        if actual != expected:
            raise ValueError('Exact bound artifact changed: ' + name)
    for prefix in request['frozen_captures']:
        receipt = json.loads((root / (prefix + '_exit.json')).read_text())
        mismatches, current_revision_differences = [], []
        with zipfile.ZipFile(root / (prefix + '_capsule.zip')) as archive:
            for name, expected in {**receipt['source_before'], **receipt['outputs']}.items():
                archived = hashlib.sha256(archive.read(name)).hexdigest()
                current = sha(root / name)
                if archived != expected:
                    mismatches.append(name)
                if current != expected:
                    current_revision_differences.append(name)
        evidence.append(dict(prefix=prefix, producer_natural_exit=receipt['natural_exit'], producer_returncode=receipt['returncode'], capsule_sha256=sha(root / (prefix + '_capsule.zip')), exact_member_mismatches=mismatches, current_vs_capture_revision_differences=current_revision_differences))
        if mismatches:
            raise ValueError('Frozen capsule discrepancy: ' + prefix)
    return evidence


def photon(root, saved):
    cfg = json.loads((root / 'rich_extensive_photon_20261014_actual1_input.json').read_text())
    mat = matter(cfg)
    original = json.loads((root / 'rich_extensive_photon_20261014_actual1_result.json').read_text())
    second = json.loads((root / 'rich_extensive_photon_20261014_actual2_result.json').read_text())
    prior = np.load(root / 'rich_extensive_photon_20261014_actual1_result.npz')
    points = []
    for multiplier, qmag in [(1, .25), (2, .5), (4, 1.)]:
        q = np.array([0., 0., qmag]); law = whole_h(mat, cfg, q)
        if multiplier == 2:
            energy, vectors = eigh(law['h'].toarray(), driver='evd')
            origin = 'NEW independent full diagonalization'
        else:
            energy = prior[f'q{multiplier}_energies']; vectors = prior[f'q{multiplier}_vectors']
            origin = 'exact hash-bound frozen full basis; NEW whole-H/current/contact/event consumer'
        ground = vectors[:, 0]; gaps = energy[1:] - energy[0]
        current = np.array([vectors[:, 1:].conj().T @ a @ ground for a in law['currents']])
        coord = np.array([vectors[:, 1:].conj().T @ a @ ground for a in law['Q']])
        bright = np.flatnonzero(np.sum(abs(coord) ** 2, axis=0) > 1.e-6)
        low = gaps[bright[0]]; window = (gaps > .8 * low) & (gaps < 1.2 * low)
        gram = current[:, window].conj() @ current[:, window].T
        contact = np.array([[mean(ground, a) for a in row] for row in law['contacts']])
        z = .08 + .017j
        signed = (current.conj() / (z - gaps)[None, :]) @ current.T - (current / (z + gaps)[None, :]) @ current.conj().T
        step = 2.e-4; probe = np.array([.37, -.29]); beta = law['coefficients']
        j, c = vertices(mat, cfg, beta, probe)
        derivatives, contacts = [], []
        for a in range(2):
            delta = np.eye(2)[a] * step
            hp = molecular_h(mat, cfg, probe + delta, beta); hm = molecular_h(mat, cfg, probe - delta, beta)
            derivatives.append(norm((hp - hm) / (2 * step) + j[a]))
            contacts.append(norm((hp + hm - 2 * molecular_h(mat, cfg, probe, beta)) / step ** 2 - c[a, a]))
        # NEW two-field kick, different from the producer preparation.
        kicked = np.exp(1j * (.091 * law['Q'][0].diagonal() - .057 * law['Q'][1].diagonal())) * ground
        initial = vectors.conj().T @ kicked
        electric = kron(law['electric'], eye(16), format='csr')
        double = np.tile(mat['double'], len(law['coordinates']))
        events, event_states = [], []
        for t in [0., 7., 19., 37.]:
            state = vectors @ (np.exp(-1j * (energy - energy[0]) * t) * initial)
            event_states.append(state)
            events.append(dict(time=t, norm=float(np.vdot(state, state).real), energy=mean(state, law['h']), current=[mean(state, a) for a in law['currents']], electric_excess=mean(state, electric) - mean(ground, electric), double_change=float(np.vdot(state, double * state).real - np.vdot(ground, double * ground).real)))
        pol = polarizations(q)
        points.append(dict(q=q, origin=origin, dimension=len(energy), all_virtuals=len(gaps), eigenbasis_residual=float(np.max(np.linalg.norm(law['h'] @ vectors - vectors * energy[None, :], axis=0))), transverse_Gram=pol @ pol.T, q_dot_polarizations=pol @ q, field_current_poles=gaps[bright[:2]], current_window_state_count=int(window.sum()), current_Gram_eigenvalues=np.linalg.eigvalsh(gram), derivative_errors=derivatives, contact_errors=contacts, full_contact=contact, all_signed_current_real=signed.real, all_signed_current_imag=signed.imag, first_coordinate_moment=np.sum(abs(coord) ** 2 * gaps, axis=1), event=events))
        saved[f'photon_q{multiplier}_new_event'] = np.array(event_states)
    feedback = selfconsistent(mat, cfg, bulk=True)
    dw = feedback['dw']; returned = []
    for direction in [[0., 0., 1.], [1., 0., 0.], [1., 1., 1.]]:
        direction = np.array(direction); direction /= np.linalg.norm(direction)
        for magnitude in [.0625, .125, .25]:
            q = magnitude * direction; mode = collective(mat, cfg, q, dw, bulk=True)
            returned.append(dict(q=q, frequencies=mode['frequencies'][:2], covariance=mode['covariance'], all_transition_count=mode['transition_count'], contact=mode['contact']))
    before = collective(mat, cfg, np.array([0., 0., .25]), np.ones(6), bulk=True)
    after = collective(mat, cfg, np.array([0., 0., .25]), dw, bulk=True)
    single = collective(mat, cfg, np.array([0., 0., .25]), dw, bulk=False)
    uniform = collective(mat, cfg, np.zeros(3), np.ones(6), bulk=True)
    larger = second['fixed_step_domain']
    return dict(source_density=mat['source_density'], independent_loop_density=mat['independent_loop_density'], volume=mat['volume'], actual_vertex_scale=cfg['charge'] / np.sqrt(cfg['epsilon'] * mat['volume']), points=points, feedback=dict(converged=feedback['converged'], iterations=len(feedback['history']), last_error=feedback['history'][-1]['error'], six_hoppings=dw, one_pass=before['frequencies'][:2], returned=after['frequencies'][:2], three_dimensional=returned), different_law_phase_average=dict(homogenized_static=after['static'], single_sine_static=single['static'], homogenized_frequencies=after['frequencies'][:2], single_sine_frequencies=single['frequencies'][:2]), uniform_A_Ward_norm=norm(uniform['static']), bound_fixed_step_domain=dict(baseline_step=larger['baseline_step'], new_step=larger['new_step'], baseline_pole=points[0]['field_current_poles'][0], larger_pole=larger['gaps'][0], relative_change=larger['gaps'][0] / points[0]['field_current_poles'][0] - 1, larger_boundary=larger['field_boundary_probability']), bound_full_time_approx_current_errors=[item['current_relative_L2_error'] for item in second['finite_gaussian_current_comparison']], bare_projection_gap=original['exact'][0]['bare_electric_gaps'][0], semantics='Finite current-bright rank2 reaction and conditional regulated homogenized continuation; no converged photons, charged G, Noether-spin, or universal light cone.')


def noether(root, saved):
    p = json.loads((root / 'rich_physical_noether_20261014_actual2_input.json').read_text())
    model = build(p); h = model['H']; dim = h.shape[0]; j = model['J']; j2 = sum(a @ a for a in j)
    states = np.load(root / 'rich_physical_noether_20261014_actual2_result.npz')
    numbers = model['N'].diagonal().real
    eigensystems = {}
    for n in [2, 3, 1]:
        indices = np.flatnonzero(abs(numbers - n) < 1.e-8)
        energies, vectors = eigh(h[indices][:, indices].toarray(), driver='evd')
        eigensystems[n] = (indices, energies, vectors)
    i0, e0, v0 = eigensystems[2]; ip, ep, vp = eigensystems[3]; im, em, vm = eigensystems[1]
    ground = embed(v0[:, 0], i0, dim); created = model['source'][0] @ ground; created /= norm(created)
    addition = np.column_stack([vp.conj().T @ (a @ ground)[ip] for a in model['source']])
    removal = np.column_stack([vm.conj().T @ (a.conj().T @ ground)[im] for a in model['source']])
    pole = np.flatnonzero(abs(ep - ep[0]) < 1.e-8); pole_j2 = np.linalg.eigvalsh(vp[:, pole].conj().T @ (j2[ip][:, ip] @ vp[:, pole]))
    z = 1.4 + .09j
    green = addition.conj().T @ (addition / (z - (ep - e0[0]))[:, None]) + (removal.conj().T @ (removal / (z + (em - e0[0]))[:, None])).T
    axis = np.array([.2, -.3, np.sqrt(.87)]); angle = .643
    skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    rotation = expm(angle * skew)
    ug = fock_rotation(model['geometry'], rotation); uf = fock_rotation(model['field'], rotation)
    # Haar frame basis is ordered m=(-1/2,+1/2), not Pauli (+1/2,-1/2).
    spinhalf = [np.array([[0, 1], [1, 0]]) / 2, np.array([[0, 1j], [-1j, 0]]) / 2, np.diag([-.5, .5])]
    fundamental = expm(-1j * angle * sum(axis[a] * spinhalf[a] for a in range(3)))
    symmetric = np.array([[1., 0., 0.], [0., 1 / np.sqrt(2), 0.], [0., 1 / np.sqrt(2), 0.], [0., 0., 1.]])
    raw = np.zeros((14, 14), complex); start = 0
    for block in [np.ones((1, 1)), fundamental.conj(), (symmetric.T @ np.kron(fundamental, fundamental) @ symmetric).conj()]:
        size = len(block); raw[start:start + size ** 2, start:start + size ** 2] = np.kron(block, np.eye(size)); start += size ** 2
    ur = model['frame']['P'].conj().T @ np.kron(raw, np.eye(16)) @ model['frame']['P']
    physical = kron(csr_matrix(ur), kron(eye(model['nb']), kron(csr_matrix(ug), csr_matrix(uf))), format='csr')
    probe = np.random.default_rng(22211).normal(size=dim) + 1j * np.random.default_rng(22212).normal(size=dim); probe /= norm(probe)
    pullback_defect = norm(physical @ probe - expm_multiply(-1j * angle * sum(axis[a] * j[a] for a in range(3)), probe))
    vectors = {name: max(norm(physical.conj().T @ model[name][a] @ physical - sum(rotation[a, b] * model[name][b] for b in range(3))) for a in range(3)) for name in ['X', 'PX', 'Q', 'P']}
    points = np.array([[.21, .32, -.18], [-.4, .1, .7]]); amplitude = np.array([.3, -.2, .6])
    av, bv = electromagnetic_profile(points, amplitude, p['field_width']); ar, br = electromagnetic_profile(points @ rotation.T, rotation @ amplitude, p['field_width'])
    derivative_step = 2.e-4; derivatives, contact_positive, contact_returned = [], [], []
    for a in range(3):
        ext = np.eye(3)[a] * derivative_step; hp = shifted_h(model, ext); hm = shifted_h(model, -ext)
        derivatives.append(norm((hp - hm) / (2 * derivative_step) + model['current'][a]))
        numerical = (hp + hm - 2 * h) / derivative_step ** 2
        positive = kron(model['quadratic'][a][a] * 2, eye(model['nf']), format='csr')
        contact_positive.append(norm(numerical - positive))
        contact_returned.append(dict(returned_plus_H_second_norm=norm(model['contact'][a][a] + positive), returned_minus_H_second_norm=norm(model['contact'][a][a] - positive), H_second_norm=norm(positive)))
    # Independent full Galerkin p^2, including retained shell0<->shell2 terms.
    extended = oscillator(p['geometry_cut'] + 2); selected = [extended['basis'].index(n) for n in model['geometry']['basis']]
    true_kinetic = sum(a @ a for a in extended['P'])[np.ix_(selected, selected)] / (2 * p['reduced_mass'] * p['length'] ** 2)
    implemented = np.diag([(sum(n) + 1.5) / (2 * p['reduced_mass'] * p['length'] ** 2) for n in model['geometry']['basis']])
    defect = true_kinetic - implemented
    changed_h = h + kron(eye(16 * model['nb']), kron(csr_matrix(defect), eye(model['nf'])), format='csr')
    changed_e0 = eigh(changed_h[i0][:, i0].toarray(), eigvals_only=True)
    changed_ep = eigh(changed_h[ip][:, ip].toarray(), eigvals_only=True)
    # NEW field-only angular impulse acting on the exact same unfiltered moving source.
    moving = states['moving_source']; kicked = expm_multiply(-.43j * (model['P'][2] + model['Q'][0]), moving)
    baseline_initial = [mean(moving, a) for a in j]; kicked_initial = [mean(kicked, a) for a in j]
    samples, event_states = [], []
    for t in [0., .9, 2.4, 3.7]:
        state = expm_multiply(-1j * t * h, kicked); baseline = expm_multiply(-1j * t * h, moving); event_states.append(state)
        partition = {key: [mean(state, a) for a in model[key]] for key in ['S', 'L', 'F']}
        torques = {key: [mean(state, 1j * (h @ a - a @ h)) for a in model[key]] for key in ['S', 'L', 'F']}
        samples.append(dict(time=t, norm=float(np.vdot(state, state).real), energy=mean(state, h), J=[mean(state, a) for a in j], partition=partition, torques=torques, torque_sum=np.sum(list(torques.values()), axis=0), current_excess=[mean(state, a) - mean(baseline, a) for a in model['current']], centre_excess=[mean(state, a) - mean(baseline, a) for a in model['X']], radius=mean(state, model['radius']), field_energy_excess=mean(state, model['field_energy']) - mean(baseline, model['field_energy']), inclusive=pvm(state, model)))
    saved['noether_new_event'] = np.array(event_states); saved['noether_new_ground'] = ground
    neutral_current = []
    for current in model['current']:
        amps = v0.conj().T @ (current @ ground)[i0]; neutral_current.append(float(np.sum(abs(amps[1:]) ** 2)))
    integer = np.diag([float(abs(value - round(value)) < 1.e-9) for value, _, _ in model['frame']['basis']]); odd = np.flatnonzero([n.bit_count() % 2 for n in range(16)])
    return dict(dimension=dim, sector_dimensions={n: len(data[0]) for n, data in eigensystems.items()}, independent_neutral_J2=mean(ground, j2), neutral_overlap_with_bound_ground=float(abs(np.vdot(ground, states['neutral_ground'])) ** 2), source_J2=mean(created, j2), moving_J2=mean(moving, j2), lowest_pole_J2=pole_j2, addition_gap=ep[0] - e0[0], removal_gap=em[0] - e0[0], raw_CAR=max(norm(a @ b.conj().T + b.conj().T @ a - (np.eye(16) if x == y else 0)) for x, a in enumerate(model['frame']['source']) for y, b in enumerate(model['frame']['source'])), all_adjacent_spectral_CAR=norm(addition.conj().T @ addition + (removal.conj().T @ removal).T - np.eye(4)), all_G_real=green.real, all_G_imag=green.imag, independent_polynomial_and_Haar_pullback=pullback_defect, all_canonical_vector_covariance=vectors, actual_spatial_fields_covariance=dict(A=norm(ar - av @ rotation.T), B=norm(br - bv @ rotation.T)), H_J_relative=[norm(h @ a - a @ h) / norm(h) for a in j], frame_only_H_relative=[norm(h @ a - a @ h) / norm(h) for a in model['S']], charged_2pi_minus=norm(expm_multiply(-2j * np.pi * j[2], created) + created), trivial_SO3_odd_weight=norm(np.kron(integer, np.eye(16)) @ model['frame']['P'][:, odd]), mechanical_current_derivative=derivatives, contact_H_second_numeric_errors=contact_positive, returned_contact_sign=contact_returned, kinetic_lagrangian_mismatch=dict(matrix_defect=defect, Frobenius_norm=norm(defect), relative_kinetic_norm=norm(defect) / norm(true_kinetic), implemented_frozen_gap=ep[0] - e0[0], explicit_true_kinetic_different_law_gap=changed_ep[0] - changed_e0[0], neutral_energy_change=changed_e0[0] - e0[0], meaning='Frozen H drops nonzero retained shell0<->shell2 p^2 matrix entries; both H laws remain rotation covariant, but original is not the stated full mechanical kinetic Galerkin law.'), neutral_current_full_Lehmann_strength=neutral_current, radial_continuity=norm(1j * (h @ model['Nt'][0] - model['Nt'][0] @ h) + model['radial_current']), supplied_Maxwell=dict(metric=model['metric'], stiffness=model['stiffness'], omega=model['omega']), new_field_only_event=dict(external_angular_impulse=np.array(kicked_initial) - baseline_initial, samples=samples), semantics='Charged Z2 bundle/CAR/binding are inputs; full moving J differs from source/pole halfspin. Current is mechanical -H derivative, not Noether J. Neutral response includes field/contact transitions, not isolated pair yield. Maxwell is supplied, not induced.')


def main():
    inp, out = map(Path, sys.argv[1:3]); root = inp.parent; request = json.loads(inp.read_text())
    bindings = binding_audit(root, request); saved = {}
    print(json.dumps(dict(stage='photon_whole_H')), flush=True)
    photon_result = photon(root, saved)
    print(json.dumps(dict(stage='noether_whole_H')), flush=True)
    noether_result = noether(root, saved)
    result = dict(kind='NEW_INDEPENDENT_WAVE11_FROZEN_SOURCE_PHYSICAL_CONSUMERS', scope=request['scope'], exact_artifact_audit=bindings, photon=photon_result, noether=noether_result)
    out.write_text(json.dumps(result, default=plain, indent=2) + '\n')
    np.savez_compressed(out.with_suffix('.npz'), **saved)
    print(json.dumps(dict(stage='complete', photon_ranks=[int(np.sum(np.array(x['current_Gram_eigenvalues']) > 1.e-12)) for x in photon_result['points']], kinetic_defect=noether_result['kinetic_lagrangian_mismatch']['Frobenius_norm'], source_gap=noether_result['addition_gap'])), flush=True)


if __name__ == '__main__':
    main()

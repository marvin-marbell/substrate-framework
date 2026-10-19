"""Independent oblique-frame / real-loop intervention, not producer replay."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
from scipy import sparse as sp
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.spatial.transform import Rotation
from rich_frame_gauss_join_20261015_model import Model, tensor, VERTICES, EDGES, SIGMA

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave29_20261015_'

def maximum(a: Any) -> float:
    if sp.issparse(a):
        return float(np.max(np.abs(a.data))) if a.nnz else 0.0
    return float(np.max(np.abs(a)))

def comm(a: Any, b: Any) -> Any:
    return a @ b - b @ a

def mean(v: Any, a: Any) -> float:
    return float(np.vdot(v, a @ v).real)

def expectations(v: Any, operators: dict[str, Any]) -> dict[str, float]:
    return {key: mean(v, value) for key, value in operators.items()}

def delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {key: a[key] - b[key] for key in a}

def group_checks(neutral: Any, charged: Any, parameters: Any, config: Any) -> dict[str, Any]:
    # Independent finite group action, not just a declared Noether matrix list.
    rotation_vector = np.array([0.41, -0.27, 0.36])
    angle = np.linalg.norm(rotation_vector)
    axis = rotation_vector / angle
    spatial = Rotation.from_rotvec(rotation_vector).as_matrix()
    spin_q = expm(-1j * sum(rotation_vector[a] * charged.sector.spin[a].toarray() for a in range(3)))
    material_q = expm(-1j * sum(rotation_vector[a] * charged.L[a].toarray() for a in range(3)))
    spin_n = expm(-1j * sum(rotation_vector[a] * neutral.sector.spin[a].toarray() for a in range(3)))
    uq = tensor(spin_q, sp.eye(10), sp.eye(3), sp.eye(3))
    un = tensor(spin_n, sp.eye(10), sp.eye(3), sp.eye(3))
    joint = tensor(spin_q, material_q, sp.eye(3), sp.eye(3))
    hamiltonian = charged.law('full', config['phase'], config['phase_edge'])[0]
    # Build lifted vector hopping directly from physical O(g)^-1 R d.
    lifted = charged.Hstatic.copy().astype(complex)
    for edge, (i, j) in enumerate(EDGES):
        direction = VERTICES[j] - VERTICES[i]
        direction = direction / np.linalg.norm(direction)
        a = charged.scalar_transport[edge].copy().astype(complex)
        for b in range(3):
            relative_tangent = sum(spatial[c, b] * direction[d] * charged.R[3*c+d]
                                   for c in range(3) for d in range(3))
            a += 1j * parameters['vector_hopping'] * tensor(
                charged.sector.hops[edge][b+1], relative_tangent,
                sp.diags(charged.r), sp.eye(3) + parameters['colour_transport'] * charged.colour_X)
        if edge == config['phase_edge']:
            a *= np.exp(1j * config['phase'])
        lifted -= a + a.conj().T
    source = [charged.source_from(neutral, 0, 'addition', a) for a in range(2)]
    source_group = []
    spin_half = expm(-0.5j * sum(rotation_vector[a] * SIGMA[a] for a in range(3)))
    joint_n = tensor(spin_n, material_q, sp.eye(3), sp.eye(3))
    for alpha in range(2):
        source_group.append(maximum(joint @ source[alpha] @ joint_n.conj().T -
                                    sum(source[beta] * spin_half[beta, alpha] for beta in range(2))))
    physical_positions = []
    for vertex in VERTICES:
        x = [sum(vertex[b] * charged.R[3*a+b] for b in range(3)) for a in range(3)]
        physical_positions.extend(maximum(material_q.conj().T @ x[a] @ material_q -
                                          sum(spatial[a, b] * x[b] for b in range(3))) for a in range(3))
    # Coordinate witnesses rotate BOTH actual g and R and therefore the real bonds.
    rg = Rotation.from_rotvec([0.2, 0.31, -0.17])
    rr = Rotation.from_rotvec([-0.31, 0.12, 0.43])
    rh = Rotation.from_rotvec(rotation_vector)
    before_positions = (rr.as_matrix() @ VERTICES.T).T
    after_positions = ((rh * rr).as_matrix() @ VERTICES.T).T
    material_bond = VERTICES[1] - VERTICES[0]
    before_relative = rg.as_matrix().T @ rr.as_matrix() @ material_bond
    after_relative = (rh * rg).as_matrix().T @ (rh * rr).as_matrix() @ material_bond
    # Noncommuting right sewing and source isometry at a second group element.
    kvec = np.array([-0.22, 0.18, 0.11])
    uk = expm(-1j * sum(kvec[a] * charged.sector.spin[a].toarray() for a in range(3)))
    ugk = expm(-1j * sum(Rotation.from_matrix(spatial @ Rotation.from_rotvec(kvec).as_matrix()).as_rotvec()[a]
                         * charged.sector.spin[a].toarray() for a in range(3)))
    bg = [uq.conj().T @ c @ un for c in source]
    return dict(
        finite_joint_H_covariance=maximum(joint.conj().T @ hamiltonian @ joint - hamiltonian),
        independent_lift_H_isometry=maximum(lifted - uq.conj().T @ hamiltonian @ uq),
        frame_only_H_change=maximum(uq.conj().T @ hamiltonian @ uq - hamiltonian),
        source_group_covariance=source_group,
        dressed_source_isometry=[maximum(bg[a] @ un.conj().T - uq.conj().T @ source[a]) for a in range(2)],
        pointwise_Haar_isometry=maximum(uq.conj().T @ uq - charged.identity),
        right_sewing_noncommuting=maximum(ugk.conj().T - uk.conj().T @ spin_q.conj().T),
        actual_position_rotation=max(physical_positions),
        coordinate_witness=dict(g_before_xyzw=rg.as_quat().tolist(), g_after_xyzw=(rh*rg).as_quat().tolist(),
                                R_before=rr.as_matrix().tolist(), R_after=(rh*rr).as_matrix().tolist(),
                                positions_before=before_positions.tolist(), positions_after=after_positions.tolist(),
                                relative_bond_invariance=maximum(after_relative-before_relative)),
        interpretation='Normalized Haar isometry follows pointwise unitary UF, not sampled-Haar quadrature. g/right sewing/CAR/constitutive vector hopping are assumed. J=S+L is physical; S alone is not conserved.')

def main() -> None:
    started = time.monotonic()
    config = json.loads((ROOT / (PREFIX + 'input.json')).read_text())
    parameters = json.loads((ROOT / config['law_input']).read_text())
    neutral, model = Model(2, parameters), Model(3, parameters)
    hneutral = neutral.law('full', config['phase'], config['phase_edge'])[0]
    rng = np.random.default_rng(config['seed'])
    eigenvalues, eigenvectors = eigsh(hneutral, k=2, which='SA', tol=1e-12,
                                    v0=rng.normal(size=neutral.dimension))
    order = np.argsort(eigenvalues)
    omega = eigenvectors[:, order[0]]
    energy = float(eigenvalues[order[0]])
    nx, ny, nz = config['source_spin_axis']
    spinor = np.array([np.sqrt((1+nz)/2), (nx+1j*ny)/np.sqrt(2*(1+nz))])
    c = sum(spinor[a] * model.source_from(neutral, 0, 'addition', a) for a in range(2))
    raw = c @ omega
    strength = float(np.vdot(raw, raw).real)
    initial = raw / np.sqrt(strength)
    w = model.lift_gauge(model.sector.wilson)
    pulse_generator = (w + w.conj().T) / 2
    pulsed = expm_multiply(-1j * config['pulse_alpha'] * pulse_generator, initial)
    shared: dict[str, Any] = {'S'+str(a): model.S[a] for a in range(3)}
    shared.update({'L'+str(a): model.material_L[a] for a in range(3)})
    shared.update({'Jcomponent'+str(a): model.J[a] for a in range(3)})
    shared.update({'Q'+str(a): model.Q[a] for a in range(3)})
    shared.update({'E'+str(a): model.E[a] for a in range(3)})
    shared.update(r2=model.r2, HE=model.HE, colour_loop=model.colour_loop,
                  LEFT2=model.S2, J2=model.J2)
    cap = (np.abs(model.sector.electric) == parameters['electric_cap']).any(axis=1)
    shared['any_electric_cap'] = model.lift_gauge(sp.diags(cap.astype(float)))
    states = dict(neutral_reference=omega, addition_raw=raw, addition_initial=initial,
                  addition_pulsed=pulsed)
    results: dict[str, Any] = {}
    for variant in config['variants']:
        h, flux, _ = model.law(variant, config['phase'], config['phase_edge'])
        operators = dict(shared)
        operators.update({'flux'+str(a): flux[a] for a in range(3)})
        operators.update({'charge_current'+str(a): -flux[a] for a in range(3)})
        operators['frame_torque_n'] = 1j * comm(h, sum(config['source_spin_axis'][a] * model.S[a] for a in range(3)))
        operators['material_torque_n'] = 1j * comm(h, sum(config['source_spin_axis'][a] * model.material_L[a] for a in range(3)))
        operators['electric_power'] = sum((model.E[a] @ flux[a] + flux[a] @ model.E[a]) /
                                           (2 * parameters['electric_inertia']) for a in range(3))
        histories: dict[str, Any] = {}
        resolution: dict[str, Any] = {}
        for label, prepared in [('baseline', initial), ('pulse', pulsed)]:
            history = expm_multiply(-1j*h, prepared, start=0., stop=config['times'][-1],
                                    num=len(config['times']), endpoint=True)
            observed = [expectations(v, operators) for v in history]
            ode = solve_ivp(lambda _t, v: -1j*(h @ v), (0., config['times'][-1]), prepared,
                            method='DOP853', rtol=2e-12, atol=2e-14, t_eval=[config['times'][-1]])
            if not ode.success:
                raise RuntimeError(ode.message)
            end = ode.y[:, -1]
            resolved = expectations(end, operators)
            resolution[label] = dict(method='DOP853 independent ODE vs adaptive sparse exponential',
                                      nfev=ode.nfev, success=ode.success, endpoint_L2=float(np.linalg.norm(end-history[-1])),
                                      max_observable_difference=max(abs(resolved[key]-observed[-1][key]) for key in resolved),
                                      norm_drift=float(abs(np.vdot(history[-1],history[-1])-1)))
            histories[label] = observed
            states[variant+'_'+label+'_trajectory'] = history
        d = delta(histories['pulse'][-1], histories['baseline'][-1])
        dn = sum(config['source_spin_axis'][a]*d['S'+str(a)] for a in range(3))
        ln = sum(config['source_spin_axis'][a]*d['L'+str(a)] for a in range(3))
        # Active-law identities use every actual control's H and current, not full-H relabelling.
        incidence = model.sector.incidence
        algebra = dict(
            flux_Heisenberg=max(maximum(1j*comm(h, model.E[a])-flux[a]) for a in range(3)),
            charge_continuity=max(maximum(1j*comm(h,model.Q[a])-sum(incidence[a,b]*flux[b] for b in range(3))) for a in range(3)),
            electric_power=maximum(1j*comm(h,model.HE)-operators['electric_power']),
            joint_J=max(maximum(comm(h,a)) for a in model.J),
            frame_n=maximum(comm(h,sum(config['source_spin_axis'][a]*model.S[a] for a in range(3)))))
        epsilon = 1e-5
        phase_plus = model.law(variant, config['phase']+epsilon, config['phase_edge'])[0]
        phase_minus = model.law(variant, config['phase']-epsilon, config['phase_edge'])[0]
        algebra['flux_minus_phase_derivative'] = maximum(-(phase_plus-phase_minus)/(2*epsilon)-flux[config['phase_edge']])
        results[variant] = dict(histories=histories, pulse_minus_baseline_endpoint=d,
                               field_induced_frame_n=dn, field_induced_material_n=ln,
                               field_induced_J_n=dn+ln, resolution=resolution, active_H_algebra=algebra,
                               preparation_work=mean(pulsed,h)-mean(initial,h),
                               baseline_own_initial_change=delta(histories['baseline'][-1],histories['baseline'][0]))
        print(json.dumps(dict(variant=variant, frame_n=dn, material_n=ln,
                              charge_current=[d['charge_current'+str(a)] for a in range(3)],
                              r2=d['r2'], HE=d['HE'], resolution=resolution)), flush=True)
    frame = group_checks(neutral, model, parameters, config)
    n_axis = sum(config['source_spin_axis'][a]*model.J[a] for a in range(3))
    result = dict(kind='INDEPENDENT_NEW_OBLIQUE_PHYSICAL_FRAME_REAL_LOOP_RESPONSE',
                  config=config, source=dict(spinor_real_imag=[[float(z.real),float(z.imag)] for z in spinor],
                  raw_strength=strength, charge=mean(initial,sum(model.Q)),
                  J2=mean(initial,model.J2), Jhalf_residual=float(np.linalg.norm((model.J2-.75*model.identity)@initial)),
                  Jaxis=mean(initial,n_axis), LEFT2=mean(initial,model.S2)),
                  neutral=dict(energy=energy,next_observed_eigenvalue=float(eigenvalues[order[1]]),
                               residual=float(np.linalg.norm(hneutral@omega-energy*omega)),J2=mean(omega,neutral.J2)),
                  preparation=dict(max_initial_matter_frame_change=max(abs(x) for key,x in delta(expectations(pulsed,shared),expectations(initial,shared)).items() if key not in ['E0','E1','E2','HE','any_electric_cap']),
                                   commutators={key:maximum(comm(pulse_generator,a)) for key,a in shared.items() if key not in ['E0','E1','E2','HE','any_electric_cap']}),
                  frame=frame,variants=results,elapsed_seconds=time.monotonic()-started,
                  dimensions=dict(neutral=neutral.dimension,addition=model.dimension),
                  source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitations='One finite addition case only. No removal/positron, continuum, full Cartesian/Poynting-current or classical-fluid emergence. LEFT2 not necessarily .75. Fixed cap2/l1/radial3/colour3; substantial cap probability remains. Ancestor69 typing errors not repaired or erased.')
    np.savez_compressed(ROOT/(PREFIX+'actual2_states.npz'), **states)
    (ROOT/(PREFIX+'actual2_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,elapsed_seconds=result['elapsed_seconds'],frame=frame,source=result['source'])),flush=True)

if __name__ == '__main__':
    main()

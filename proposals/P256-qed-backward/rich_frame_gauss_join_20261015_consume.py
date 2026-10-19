"""Fresh owning model, both signed sources, actual field intervention and defeats."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy import sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.special import jv
from rich_frame_gauss_join_20261015_model import Model, SIGMA, EDGES, VERTICES, tensor, clean


def maximum(matrix) -> float:
    a = sp.csr_matrix(matrix)
    return float(np.max(np.abs(a.data), initial=0.))


def norm(vector) -> float:
    return float(np.linalg.norm(vector))


def expectation(state, operator) -> float:
    return float(np.vdot(state, operator@state).real)


def evolve(hamiltonian, state, time_value):
    return expm_multiply((-1j*time_value)*hamiltonian, state, traceA=(-1j*time_value)*hamiltonian.diagonal().sum())


def chebyshev(hamiltonian, state, time_value):
    diagonal = hamiltonian.diagonal().real
    radius = np.asarray(abs(hamiltonian).sum(axis=1)).ravel()-np.abs(hamiltonian.diagonal())
    low, high = float(np.min(diagonal-radius)), float(np.max(diagonal+radius))
    centre, width = (high+low)/2, (high-low)/2
    scaled = (hamiltonian-centre*sp.eye(hamiltonian.shape[0]))/width
    degree = max(50, int(width*abs(time_value)+80))
    coefficients = np.array([jv(k, width*time_value) for k in range(degree+501)])
    t0 = state.copy(); t1 = scaled@state
    answer = coefficients[0]*t0+2*(-1j)*coefficients[1]*t1
    for k in range(2, degree+1):
        t2 = 2*(scaled@t1)-t0
        answer += 2*((-1j)**k)*coefficients[k]*t2
        t0, t1 = t1, t2
    return np.exp(-1j*centre*time_value)*answer, dict(enclosure=[low, high], degree=degree, numerical_Bessel_tail_500=2*float(np.abs(coefficients[degree+1:]).sum()))


def algebra(model, variant):
    h, currents, contacts = model.law(variant)
    sector = model.sector
    commutator = lambda a, b: clean(a@b-b@a)
    result = dict(hermiticity=maximum(h-h.conj().T),
                  joint_J_commutators=[maximum(commutator(h, j)) for j in model.J],
                  LEFT_z_commutator=maximum(commutator(h, model.S[2])),
                  flux_current=[maximum(1j*commutator(h, e)-j) for e, j in zip(model.E, currents)],
                  continuity=[], electric_power=[], source='Actual active-edge primitive law, never full-H probes relabelled OFF/cut')
    for site in range(3):
        result['continuity'].append(maximum(1j*commutator(h, model.Q[site])-sum(sector.incidence[site, e]*currents[e] for e in range(3))))
    for edge in range(3):
        electric_energy = model.E[edge]@model.E[edge]/(2*model.parameters['electric_inertia'])
        power = (model.E[edge]@currents[edge]+currents[edge]@model.E[edge])/(2*model.parameters['electric_inertia'])
        result['electric_power'].append(maximum(1j*commutator(h, electric_energy)-power))
    delta = 1e-5
    hp, jp, _ = model.law(variant, delta)
    hm, _, _ = model.law(variant, -delta)
    result['current_minus_Hprime'] = maximum(-(hp-hm)/(2*delta)-currents[0])
    result['contact_Hsecond'] = maximum((hp+hm-2*h)/(delta**2)-contacts[0])
    result['current_at_own_nonzero_phase'] = maximum(-(model.law(variant, .17+delta)[0]-model.law(variant, .17-delta)[0])/(2*delta)-model.law(variant, .17)[1][0])
    result['cut_receiver_charge_commutator'] = maximum(commutator(h, model.Q[1]+model.Q[2]))
    gauss = sector.electric@sector.incidence.T
    gauss[:, 0] += sector.boundary
    result['Gauss_all_basis'] = float(np.max(np.abs(gauss-sector.charge)))
    return result


def observe(model, variant, state):
    h, currents, _ = model.law(variant)
    he_power = 1j*(h@model.HE-model.HE@h)
    spin_torque = 1j*(h@model.S[2]-model.S[2]@h)
    material_torque = 1j*(h@model.material_L[2]-model.material_L[2]@h)
    sector = model.sector
    electric_cap = model.lift_gauge(sp.diags(np.any(np.abs(sector.electric) == sector.cap, axis=1).astype(float)))
    return dict(norm_squared=float(np.vdot(state, state).real), H=expectation(state, h), HE=expectation(state, model.HE),
                Hcolour=expectation(state, model.Hcolour), LEFT_z=expectation(state, model.S[2]),
                material_Lz=expectation(state, model.material_L[2]), Jz=expectation(state, model.J[2]), J2=expectation(state, model.J2),
                LEFT_spin2=expectation(state, model.S2), charge=[expectation(state, q) for q in model.Q],
                receiver_charge=expectation(state, model.Q[1]+model.Q[2]), flux=[expectation(state, e) for e in model.E],
                electric_field=[expectation(state, e)/model.parameters['electric_inertia'] for e in model.E],
                flux_current=[expectation(state, j) for j in currents], physical_charge_current=[-expectation(state, j) for j in currents],
                electric_power=expectation(state, he_power), LEFT_torque=expectation(state, spin_torque), material_torque=expectation(state, material_torque),
                doubles=expectation(state, model.D), holons=expectation(state, model.holon), r2=expectation(state, model.r2),
                colour_loop=expectation(state, model.colour_loop), colour_electric_C2=expectation(state, model.colour_electric),
                any_link_at_cap_probability=expectation(state, electric_cap))


def difference(a, b):
    answer = {}
    for key in a:
        if isinstance(a[key], list):
            answer[key] = (np.array(a[key])-np.array(b[key])).tolist()
        elif isinstance(a[key], (int, float)):
            answer[key] = a[key]-b[key]
    return answer


def coordinate_frame_checks(model, reference, branch, rng):
    """Independent explicit relative-frame coefficients versus isometric pullback."""
    sector = model.sector
    axis = np.array([.3, -.4, .5]); axis /= np.linalg.norm(axis)
    angle = .63
    spin_generator = sum(axis[a]*sector.spin[a].toarray() for a in range(3))
    ug = expm(-1j*angle*spin_generator)
    ug_full = tensor(sp.csr_matrix(ug), sp.eye(10), sp.eye(len(model.r)), sp.eye(3))
    fund = expm(-.5j*angle*sum(axis[a]*SIGMA[a] for a in range(3)))
    rotation = np.array([[np.trace(SIGMA[b]@fund.conj().T@SIGMA[a]@fund).real/2 for b in range(3)] for a in range(3)])
    colour_transport = sp.eye(3)+model.parameters['colour_transport']*model.colour_X
    lifted = model.Hstatic.copy().astype(complex)
    for edge, (i, j) in enumerate(EDGES):
        direction = VERTICES[j]-VERTICES[i]; direction /= np.linalg.norm(direction)
        a = model.scalar_transport[edge].copy().astype(complex)
        for body_axis, lab_axis in np.ndindex(3, 3):
            tangent = sum(direction[b]*model.R[3*lab_axis+b] for b in range(3))
            a += 1j*model.parameters['vector_hopping']*rotation[lab_axis, body_axis]*tensor(sector.hops[edge][body_axis+1], tangent, sp.diags(model.r), colour_transport)
        lifted -= a+a.conj().T
    h = model.law()[0]
    explicit_lift_residual = maximum(lifted-ug_full.conj().T@h@ug_full)
    refspin = sum(axis[a]*reference.sector.spin[a].toarray() for a in range(3))
    uref = expm(-1j*angle*refspin)
    sources = [model.source_from(reference, 0, branch, c) for c in range(2)]
    uref_full = tensor(sp.csr_matrix(uref), sp.eye(10), sp.eye(len(model.r)), sp.eye(3))
    source_intertwining = [maximum(ug_full.conj().T@sources[a]@uref_full-sum(fund.conj().T[b, a]*sources[b] for b in range(2))) for a in range(2)]
    generator_intertwining = []
    for a in range(3):
        for component in range(2):
            generator_intertwining.append(maximum(model.J[a]@sources[component]-sources[component]@reference.J[a]-sum(SIGMA[a][b, component]*sources[b]/2 for b in range(2))))
    vector = rng.normal(size=sector.dimension)+1j*rng.normal(size=sector.dimension)
    vector /= norm(vector)
    step = 1e-5
    left_derivative = []
    right_equivariance = []
    for a in range(3):
        s = sector.spin[a].toarray()
        # I(h^-1 g) = U(g^-1) U(h): actual LEFT coordinate derivative.
        plus = ug.conj().T@expm(-1j*step*s)@vector
        minus = ug.conj().T@expm(1j*step*s)@vector
        left_derivative.append(norm(1j*(plus-minus)/(2*step)-ug.conj().T@s@vector))
        uk = expm(-1j*.27*s)
        right_equivariance.append(norm((ug@uk).conj().T@vector-uk.conj().T@ug.conj().T@vector))
    noncommuting = max(maximum(a@b-b@a) for a in model.R for b in model.R)
    return dict(explicit_relative_frame_H_intertwiner=explicit_lift_residual,
                Wigner_dressed_source_intertwiner=source_intertwining,
                full_spatial_generator_source_intertwiner=generator_intertwining,
                actual_LEFT_coordinate_derivative=left_derivative, right_Gauss_finite_action=right_equivariance,
                Haar_isometry_pointwise_unitarity=norm(ug.conj().T@ug-np.eye(sector.dimension)),
                quantum_material_coordinate_commutator_max=noncommuting,
                note='Pointwise relative-frame primitive built explicitly, not just conjugating H and reporting its tautological equality. Haar proof and physical geometry are in equations.json. Coordinate derivatives are finite-difference checks, not a proof of arbitrary-function identities.')


def main():
    input_path, output_prefix = Path(sys.argv[1]), Path(sys.argv[2])
    p = json.loads(input_path.read_text())
    rng = np.random.default_rng(p['random_seed'])
    start = time.monotonic()
    models = {name: Model(number, p) for name, number in [('neutral', 2), ('addition', 3), ('removal', 1)]}
    neutral = models['neutral']; hn = neutral.law()[0]
    initial = rng.normal(size=neutral.dimension)+1j*rng.normal(size=neutral.dimension)
    values, vectors = eigsh(hn, k=4, which='SA', tol=p['eigensolver_tolerance'], v0=initial, maxiter=20000)
    order = np.argsort(values); values, vectors = values[order], vectors[:, order]
    omega = vectors[:, 0]; omega /= norm(omega)
    result = dict(kind='NEW_OWNING_PHYSICAL_FRAME_GAUSS_LAW_ACTUAL_RESPONSE', law=p['law_id'],
                  ancestor_science_inputs=[], parameters=p, dimensions={k:dict(gauge_matter=v.sector.dimension, complete=v.dimension, factor_shape=v.shape) for k, v in models.items()},
                  reference=dict(energy=float(values[0]), first_four_eigenvalues=values.tolist(), residual=norm(hn@omega-values[0]*omega), observations=observe(neutral, 'full', omega)),
                  algebra={}, frame={}, branches={}, execution=dict(start_time_unix=time.time(), scientific_runtime=sys.executable))
    npz = dict(neutral_reference=omega)
    if abs(expectation(omega, neutral.J2)) > 1e-8:
        raise RuntimeError('Fresh neutral lowest state is not J0; do not fabricate a spin-half source')
    for name, model in models.items():
        result['algebra'][name] = {variant:algebra(model, variant) for variant in ['full', 'vector_off', 'path_cut']}
    print(json.dumps(dict(stage='new_reference', dimensions=result['dimensions'], energy=float(values[0]), J2=expectation(omega, neutral.J2))), flush=True)
    for branch in ['addition', 'removal']:
        model = models[branch]
        result['frame'][branch] = coordinate_frame_checks(model, neutral, branch, rng)
        sources = [model.source_from(neutral, site, branch) for site in range(3)]
        raw = sources[0]@omega
        strength = float(np.vdot(raw, raw).real)
        base = raw/np.sqrt(strength)
        pulse = evolve(model.F, base, p['field_pulse'])
        branch_result = dict(raw_anchor_source_strength=strength, source_strength_by_site=[float(np.vdot(c@omega, c@omega).real) for c in sources],
                             initial_source_spin_half_residual=norm((model.J2-.75*model.identity)@base),
                             initial_Jz_half_residual=norm((model.J[2]-.5*model.identity)@base),
                             pure_field_matter_commutators=dict(LEFT_z=maximum(model.F@model.S[2]-model.S[2]@model.F), charge=max(maximum(model.F@q-q@model.F) for q in model.Q), r2=maximum(model.F@model.r2-model.r2@model.F), colour=maximum(model.F@model.colour_loop-model.colour_loop@model.F)), variants={})
        for variant in ['vector_off', 'full', 'path_cut']:
            h, currents, _ = model.law(variant)
            neutral_h = neutral.law(variant)[0]
            histories = []
            initial_base = observe(model, variant, base)
            initial_pulse = observe(model, variant, pulse)
            last_base, last_pulse = base, pulse
            for time_value in p['times']:
                state = evolve(h, base, time_value)
                kicked = evolve(h, pulse, time_value)
                neutral_t = evolve(neutral_h, omega, time_value)
                base_obs, pulse_obs = observe(model, variant, state), observe(model, variant, kicked)
                signed_green = []
                for c in sources:
                    z = np.vdot(c@neutral_t, state)*np.sqrt(strength)
                    signed_green.append([float(z.real), float(z.imag)])
                histories.append(dict(time=time_value, baseline=base_obs, pulse=pulse_obs, field_intervention=difference(pulse_obs, base_obs),
                                      baseline_change_from_own_initial=difference(base_obs, initial_base),
                                      pulse_change_from_own_initial=difference(pulse_obs, initial_pulse),
                                      signed_two_sided_source_G=signed_green))
                last_base, last_pulse = state, kicked
            check_base, cheb_info = chebyshev(h, base, p['times'][-1])
            check_pulse, _ = chebyshev(h, pulse, p['times'][-1])
            checks = dict(chebyshev=cheb_info, endpoint_baseline_L2=norm(last_base-check_base), endpoint_pulse_L2=norm(last_pulse-check_pulse),
                          endpoint_LEFT_z_difference=expectation(last_pulse, model.S[2])-expectation(check_pulse, model.S[2]),
                          endpoint_flux_current_difference=[expectation(last_pulse, j)-expectation(check_pulse, j) for j in currents],
                          maximum_norm_drift=max(abs(entry[k]['norm_squared']-1) for entry in histories for k in ['baseline', 'pulse']),
                          baseline_H_drift=max(abs(entry['baseline']['H']-initial_base['H']) for entry in histories),
                          pulse_H_drift=max(abs(entry['pulse']['H']-initial_pulse['H']) for entry in histories))
            branch_result['variants'][variant] = dict(histories=histories, numerical_resolution=checks,
                                                    initial_field_preparation_work=initial_pulse['H']-initial_base['H'])
            npz[branch+'_'+variant+'_baseline_endpoint'] = last_base
            npz[branch+'_'+variant+'_pulse_endpoint'] = last_pulse
            if variant == 'vector_off':
                failed = dict(kind='COMPUTED_SCALAR_RIGHT_HAAR_ROUTE_IS_INSUFFICIENT', branch=branch,
                              endpoint_field_induced_LEFT_z=histories[-1]['field_intervention']['LEFT_z'],
                              endpoint_field_induced_flux_current=histories[-1]['field_intervention']['flux_current'],
                              exact_LEFT_commutator=result['algebra'][branch][variant]['LEFT_z_commutator'],
                              conclusion='Physical frame halfspin alone remains dynamically spectator in its angular channel despite scalar charge/field response. Change material construction to actual relative-frame vector transport; do not end at this no-go.')
                (output_prefix.parent/(output_prefix.name+'_'+branch+'_failed_scalar_route.json')).write_text(json.dumps(failed, indent=2)+'\n')
                print(json.dumps(failed), flush=True)
        npz[branch+'_initial_baseline'] = base
        npz[branch+'_initial_field_pulse'] = pulse
        result['branches'][branch] = branch_result
        endpoint = branch_result['variants']['full']['histories'][-1]
        print(json.dumps(dict(stage='physical_joined_response', branch=branch, endpoint_field_effect=endpoint['field_intervention'], resolution=branch_result['variants']['full']['numerical_resolution'])), flush=True)
        # A compact result exists at the first genuinely completed physical branch.
        (output_prefix.parent/(output_prefix.name+'_first_branch_synthesis.json')).write_text(json.dumps(dict(kind='ACTUAL_SCOPED_FIRST_PHYSICAL_JOIN', reference=result['reference'], branch=branch, frame=result['frame'][branch], algebra=result['algebra'][branch], response=branch_result), indent=2)+'\n')
    result['execution']['elapsed_seconds'] = time.monotonic()-start
    result['execution']['completed'] = True
    np.savez_compressed(str(output_prefix)+'_states.npz', **npz)
    (output_prefix.parent/(output_prefix.name+'_result.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(stage='completed', elapsed_seconds=result['execution']['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()

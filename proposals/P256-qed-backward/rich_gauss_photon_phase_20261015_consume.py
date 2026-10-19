"""One I16 shared-reference spectral experiment and causal local field packet."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.special import jv
from rich_open_gauss_carrier_20261015_model import NeutralCube, LAW_ID
from rich_gauss_photon_phase_20261015_spectral import response, real

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_gauss_photon_phase_20261015_'


def save(name: str, value: Any) -> None:
    (ROOT/(PREFIX+name+'.json')).write_text(json.dumps(value, indent=2)+'\n')


def maximum(matrix: Any) -> float:
    return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0


def comm(a: Any, b: Any) -> Any:
    return a @ b-b @ a


def measure(model: Any, states: Any, h: Any, hr: Any, active_edges: Any = None) -> dict[str, Any]:
    p = np.abs(states)**2
    active = set(range(12) if active_edges is None else active_edges)
    zero = sparse.csr_matrix(model.H.shape, dtype=complex)
    currents = [j if e in active else zero for e,j in enumerate(model.currents)]
    def expectation(a: Any) -> Any:
        return np.sum(states.conj()*(a @ states), axis=0).real.tolist()
    power = [np.sum(states.conj()*(model.electric[:,e,None]*(currents[e] @ states)),axis=0).real/16 for e in range(12)]
    d = model.doubles.sum(axis=1)
    return dict(norm=p.sum(axis=0).tolist(), H=expectation(h), own_HR=expectation(hr),
                ER=(model.ER @ p).tolist(), HE=(model.HE @ p).tolist(), HU=(model.HU @ p).tolist(),
                rho=(model.rho.T @ p).T.tolist(), flux=(model.electric.T @ p).T.tolist(),
                currents=np.asarray([expectation(j) for j in currents]).T.tolist(),
                J2=np.asarray([np.sum(np.abs(j @ states)**2,axis=0) for j in currents]).T.tolist(),
                sym_EJ_electric_power=np.asarray(power).T.tolist(),
                right_powerER=np.asarray(power)[model.right_edges].sum(axis=0).tolist(),
                right_owned_energy_power=(-2*np.sum((h @ states).conj()*(hr @ states),axis=0).imag).tolist(),
                doublon_total=(d @ p).tolist(),
                right_doublons=(model.doubles[:,model.right_sites].sum(axis=1) @ p).tolist(),
                probability_any_pair=p[d>0].sum(axis=0).tolist(),
                right_charge_free_ER=((model.ER*np.all(model.rho[:,model.right_sites]==0,axis=1)) @ p).tolist())


def chebyshev(h: Any, state: Any, duration: float, degree: int) -> tuple[Any, dict[str, Any]]:
    d = h.diagonal()
    radius = np.asarray(abs(h).sum(axis=1)).ravel()-np.abs(d)
    low, high = float(np.min(d-radius)), float(np.max(d+radius))
    center, scale = (low+high)/2, (high-low)/2
    previous = state.copy()
    current = (h @ state-center*state)/scale
    out = jv(0,scale*duration)*previous+2*(-1j)*jv(1,scale*duration)*current
    for k in range(2,degree+1):
        following = 2*(h @ current-center*current)/scale-previous
        out += 2*(-1j)**k*jv(k,scale*duration)*following
        previous,current = current,following
    out *= np.exp(-1j*center*duration)
    tail = 2*np.sum(np.abs(jv(np.arange(degree+1,degree+501),scale*duration)))
    return out,dict(enclosure=[low,high], degree=degree, numerical_Bessel_tail_500_terms=float(tail))


def main() -> None:
    start = time.monotonic()
    settings = json.loads((ROOT/sys.argv[1]).read_text())
    model = NeutralCube()
    model_hash = hashlib.sha256((ROOT/settings['model']).read_bytes()).hexdigest()
    reference_path = ROOT/settings['shared_reference']
    reference = np.load(reference_path)
    omega = reference['psi']
    if str(reference['model_SHA256']) != model_hash or str(reference['basis_SHA256']) != model.basis_hash:
        raise ValueError('Shared input model/basis hash mismatch')
    if not np.array_equal(reference['old_indices'],model.old_indices) or not np.array_equal(reference['electric'],model.electric):
        raise ValueError('Shared input basis arrays mismatch')
    binding = dict(model_SHA256=model_hash, basis_SHA256=model.basis_hash,
                   reference_file=settings['shared_reference'], reference_SHA256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                   owner_input_SHA256=str(reference['input_SHA256']), owner_energy=float(reference['energy']),
                   owner_residual=float(reference['residual']), declared_shared_INPUT_not_independent_ground_solve=True)
    spectral, arrays = response(model, omega, settings['spectral'])
    spectral['input_binding'] = binding
    save('spectral_result',spectral)
    np.savez_compressed(ROOT/(PREFIX+'spectral_states.npz'),**arrays,old_indices=model.old_indices)
    save('first_phase_synthesis',dict(kind='FIRST_ACTUAL_I16_PHASE_MODE_FINDING_BEFORE_PACKET_OR_AUDIT',
         input_binding=binding, selected=spectral['selected'],
         contact=spectral['contact_expectation'], relaxed_static_curvature=spectral['relaxed_static_twist_curvature'],
         finding='Actual finite coupled field/current/loop response computed; a gapped mixed ray is a finding, not a photon. Shared Ω is INPUT. No imported Maxwell D or irreducible-Dyson reinterpretation.',
         finite_response_resolution=[dict(frequency=r['frequency'],coarse_difference=r['coarse_difference_max'],resolvent_error_bound=r['resolvent_error_l2_bound']) for r in spectral['finite_response']]))
    print(json.dumps(dict(event='FIRST_ACTUAL_PHASE_MODE',selected=spectral['selected'],input_binding=binding)),flush=True)
    generator = (model.W-model.W.T)/(2j)
    pulse = expm_multiply(-1j*settings['packet']['alpha']*generator,omega,traceA=0)
    initial = np.column_stack((omega,pulse))
    states = initial.copy()
    samples = []
    dt = settings['packet']['duration']/settings['packet']['intervals']
    for step in range(settings['packet']['intervals']+1):
        samples.append(dict(time=step*dt,observables=measure(model,states,model.H,model.HR)))
        if step<settings['packet']['intervals']:
            states = expm_multiply(-1j*dt*model.H,states,traceA=-1j*dt*model.H.diagonal().sum())
    first,last = samples[0]['observables'],samples[-1]['observables']
    trajectory = dict(kind='FIRST_ACTUAL_SAME_H_LOCAL_UNITARY_FIELD_PACKET',input_binding=binding,settings=settings['packet'],
                      samples=samples,preparation_work=first['H'][1]-first['H'][0],
                      endpoint_delta_ER=last['ER'][1]-last['ER'][0],endpoint_delta_own_HR=last['own_HR'][1]-last['own_HR'][0],
                      prepared_selected_ray_probability=float(abs(np.vdot(arrays['selected_mode'],pulse))**2),
                      source_semantics='exp(-i alpha ImWleft) Ω is a local field UNITARY, not spectral source insertion or postselection. Receiver is opposite face; no directional-eigenpacket or one-way propagation claim. Splitting cannot establish absence of propagation.',
                      covariance_semantics='J2 and sym(EJ)/I are actual equal-time correlations, not products of means. Spectral response separately retains signed unequal-time linear response.',
                      matter_semantics='Diagonal doublon/holon admixture, not an on-shell/virtual classification.')
    save('first_packet_synthesis',trajectory)
    print(json.dumps(dict(event='FIRST_ACTUAL_PACKET',work=trajectory['preparation_work'],delta_ER=trajectory['endpoint_delta_ER'],delta_HR=trajectory['endpoint_delta_own_HR'])),flush=True)
    # New causal controls use exactly the same initial state columns and their
    # own Hamiltonian/receiver. The baseline is NOT regrounded for either arm.
    off = np.exp(-1j*settings['packet']['duration']*(model.HE+model.HU))[:,None]*initial
    cut = expm_multiply(-1j*settings['packet']['duration']*model.Hcut,initial,
                        traceA=-1j*settings['packet']['duration']*model.Hcut.diagonal().sum())
    cut_edges = np.concatenate((model.left_edges,model.right_edges))
    controls = dict(off_before=measure(model,initial,model.Hoff,model.HRoff,()),off_after=measure(model,off,model.Hoff,model.HRoff,()),
                    cut_before=measure(model,initial,model.Hcut,model.HRcut,cut_edges),cut_after=measure(model,cut,model.Hcut,model.HRcut,cut_edges))
    er = sparse.diags(model.ER,format='csr')
    locality = dict(source_ER=maximum(comm(generator,er)),source_HR=maximum(comm(generator,model.HR)),
                    source_HRoff=maximum(comm(generator,model.HRoff)),source_HRcut=maximum(comm(generator,model.HRcut)),
                    off_ER=maximum(comm(model.Hoff,er)),off_HRoff=maximum(comm(model.Hoff,model.HRoff)),
                    cut_factorization=maximum(comm(model.Hcut-model.HRcut,model.HRcut)),
                    cut_left_ER=maximum(comm(model.Hcut-model.HRcut,er)))
    locality['source_right_J2'] = max(maximum(comm(generator,model.currents[e] @ model.currents[e])) for e in model.right_edges)
    units = dict(flux_motion=0.0,physical_field_motion=0.0,electric_power=0.0)
    for e in range(12):
        E = sparse.diags(model.electric[:,e].astype(float),format='csr')
        J = model.currents[e]
        units['flux_motion'] = max(units['flux_motion'],maximum(1j*comm(model.H,E)-J))
        units['physical_field_motion'] = max(units['physical_field_motion'],maximum(1j*comm(model.H,E/16)-J/16))
        units['electric_power'] = max(units['electric_power'],maximum(1j*comm(model.H,E @ E/32)-(E @ J+J @ E)/32))
    independent,resolution = chebyshev(model.H,initial,settings['packet']['duration'],settings['packet']['chebyshev_degree'])
    independent_obs = measure(model,independent,model.H,model.HR)
    resolution.update(endpoint_state_l2_difference=np.linalg.norm(independent-states,axis=0).tolist(),
                      delta_ER_difference=(independent_obs['ER'][1]-independent_obs['ER'][0])-trajectory['endpoint_delta_ER'],
                      delta_HR_difference=(independent_obs['own_HR'][1]-independent_obs['own_HR'][0])-trajectory['endpoint_delta_own_HR'],
                      maximum_norm_drift=max(float(np.max(np.abs(np.asarray(s['observables']['norm'])-1))) for s in samples),
                      maximum_H_drift=max(float(np.max(np.abs(np.asarray(s['observables']['H'])-first['H']))) for s in samples))
    result = dict(law_id=LAW_ID,input_binding=binding,dimension=model.dimension,H_nnz=model.H.nnz,
                  geometry=dict(positions=model.positions.tolist(),edges=model.edges,left_cycle=model.left_cycle.tolist(),
                                right_edges=model.right_edges.tolist(),bridges=model.bridges.tolist()),
                  spectral_result=PREFIX+'spectral_result.json',packet=trajectory,controls=controls,
                  exact_all_basis_locality_and_defeat=locality,actual_primitive_I16_units=units,
                  numerical_resolution=resolution,scientific_elapsed_seconds=time.monotonic()-start,
                  limitation='One cube, one I16 law, one shared stationary Ω, one source block and one unitary packet. No sweep, Coulomb, low-q speed, two-photon polarization, physical Noether spin, continuum or Nature claim.')
    save('actual_result',result)
    np.savez_compressed(ROOT/(PREFIX+'actual_states.npz'),initial=initial,endpoint=states,independent_endpoint=independent,
                        off_endpoint=off,cut_endpoint=cut,old_indices=model.old_indices)
    print(json.dumps(dict(event='COMPLETED_SAME_LAW_RESPONSE',resolution=resolution,locality=locality,units=units,elapsed=result['scientific_elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    main()

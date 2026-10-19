"""New real joint-current consumer and executed wrong-case discriminants."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from rich_soft_field_20261014_model import Parameters, Joint, Matter, LAW


def summarize(joint, label, full=False):
    values, vectors = joint.spectrum()
    g = joint.g
    gaps = values-values[0]
    weights = {name: np.abs(vectors.conj().T@(op@g))**2 for name, op in
               [('flux', joint.F), ('electric', joint.electric_leg), ('current', joint.J), ('density', joint.Rho)]}
    bright = np.flatnonzero((gaps > 1e-7) & (weights['flux'] > max(1e-9, np.max(weights['flux'][1:])*0.01)))
    first = int(bright[0]) if len(bright) else None
    reshaped = g.reshape(len(joint.phi), joint.md)
    probabilities = np.sum(np.abs(reshaped)**2, axis=1)
    reduced = reshaped.conj().T@reshaped
    purity = float(np.trace(reduced@reduced).real)
    geom = joint.geometry
    result = {'label': label, 'n': joint.n, 'q': geom['q'], 'curl_eigenvalue': geom['lambda'],
              'curl_gram': geom['gram'].tolist(), 'gauss_errors': [leg['gauss_error'] for leg in geom['legs']],
              'volume': geom['volume'], 'local_flux_factor': geom['local_flux_factor'],
              'canonical_electric_metric': joint.metric, 'dimension': joint.H.shape[0], 'points': len(joint.phi),
              'dx': joint.dx, 'halfwidth': joint.p.halfwidth, 'ground_energy': float(values[0]),
              'gaps': gaps.tolist(), 'weights': {name: a.tolist() for name, a in weights.items()},
              'first_flux_bright_index': first, 'first_flux_bright_gap': None if first is None else float(gaps[first]),
              'first_flux_velocity': None if first is None else float(gaps[first]/np.sqrt(geom['lambda'])),
              'matter_reduced_purity': purity, 'field_rms': float(np.sqrt(probabilities@joint.phi**2)),
              'outer_tenth_probability': float(np.sum(probabilities[np.abs(joint.phi)>.9*joint.p.halfwidth])),
              'contact': float(np.vdot(g, joint.K@g).real),
              'current_electric_commutator_norm': float(np.linalg.norm((joint.J@joint.electric_leg-joint.electric_leg@joint.J)@g)),
              'ground_residual': float(np.linalg.norm(joint.H@g-values[0]*g)),
              'orthonormal_error': float(np.linalg.norm(vectors.conj().T@vectors-np.eye(len(values))))}
    if full and first is not None:
        w = float(gaps[first]*.8)
        response = joint.response(w, .02)
        result['response'] = {'omega': w, 'eta': .02, 'operators': ['sin_local_flux', 'normalized_electric_leg', 'loop_current', 'density_0_minus_1'],
                              'real': response.real.tolist(), 'imag': response.imag.tolist(),
                              'uses_all_joint_virtual_states': True}
        result['current_insertion'] = joint.pulse([0., 1., 3., 8., 16.])
    print(json.dumps({k: result[k] for k in ['label', 'n', 'dimension', 'first_flux_bright_gap', 'first_flux_velocity', 'matter_reduced_purity', 'field_rms', 'outer_tenth_probability']}), flush=True)
    return result


def main():
    request = json.loads(Path('rich_soft_field_20261014_input.json').read_text())
    p = Parameters(**{**request['parameters'], 'shape': tuple(request['parameters']['shape'])})
    results = []
    for n in request['q_indices']:
        results.append(summarize(Joint(p, n), 'actual_joint', full=n==1))
        results.append(summarize(Joint(p, n, no_matter=True), 'bare_E_no_matter'))
        if n == 1:
            results.append(summarize(Joint(p, n, charge=0.), 'charge_zero_matter_retained'))
            results.append(summarize(Joint(p, n, cut=True), 'both_loop_closures_cut'))
    for width in request['width_controls']:
        pp = replace(p, halfwidth=width*np.pi)
        results.append(summarize(Joint(pp, 1), f'actual_box_{width}pi'))
        results.append(summarize(Joint(pp, 1, no_matter=True), f'bare_E_box_{width}pi'))
    for changed in request['material_designs']:
        pp = replace(p, bridge=changed['bridge'], shape=tuple(changed['shape']))
        results.append(summarize(Joint(pp, 1), changed['label']))
    matter = Matter(p)
    h, j, k = matter.operators(0.)
    ev, vec = np.linalg.eigh(h.toarray())
    curvature_current = float(np.vdot(vec[:, 0], k@vec[:, 0]).real-2*sum(abs(np.vdot(vec[:, r], j@vec[:, 0]))**2/(ev[r]-ev[0]) for r in range(1, len(ev)) if ev[r]-ev[0]>1e-10))
    large_gauge = float(np.linalg.norm((matter.operators(2*np.pi)[0]-h).toarray()))
    sewing = {'gauge': 'all loop flux on closing edge; H(phi+2pi)=H(phi) exactly, with ordinary global CAR signs',
              'matter_large_gauge_error': large_gauge, 'field_amplitude_boundary': 'Dirichlet, not a compact identification or a physical large-gauge quotient',
              'matter_dimension': matter.dimension, 'retained_quantum_matter': 'all 64 eight-site Nup=Ndown=1 CAR configurations',
              'static_curvature_diagnostic_not_installed': curvature_current}
    result = {'law': LAW, 'request': request, 'cases': results, 'sewing': sewing,
              'two_leg_joint': {'law': 'H_total=H_xy tensor I + I tensor H_xz, disjoint matter clusters; both legs fully quantum',
                               'E_gram': results[0]['curl_gram'], 'cross_residue': 0.,
                               'basis': 'xy clusters at z=0,1 and y=0,1; rotated xz clusters at y=2,3 and z=0,1'},
              'scope': 'Finite collective single-q cosine mode projection, actual full retained matter/field feedback. Not full nonlinear lattice gauge theory or demonstrated homogeneous bulk photon phase.'}
    Path('rich_soft_field_20261014_actual1_result.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed('rich_soft_field_20261014_actual1_geometry.npz',
                        m0=Joint(p, 1).geometry['legs'][0]['m'], e0=Joint(p, 1).geometry['legs'][0]['e'],
                        m1=Joint(p, 1).geometry['legs'][1]['m'], e1=Joint(p, 1).geometry['legs'][1]['e'])


if __name__ == '__main__':
    main()

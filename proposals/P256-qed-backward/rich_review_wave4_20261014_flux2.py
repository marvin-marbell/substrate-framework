"""New contrary-parameter actual same-H field/current/body receiver."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, splu
from rich_spin_flux_spatial_20261014_model import Parameters, Sector
from rich_review_wave4_20261014_tools import gram_frame, complex_pair, save


def probe(p: Parameters, times: np.ndarray, kick: float) -> dict:
    sector = Sector(p)
    energy, ground, energies = sector.ground()
    left_ops = [sector.F[0], sector.loop_electric[0], sector.total_currents[0]]
    right_ops = [sector.F[2], sector.loop_electric[2]]
    left = np.column_stack([op @ ground for op in left_ops])
    right = np.column_stack([op @ ground for op in right_ops])
    left -= ground[:, None] * (ground.conj() @ left)[None, :]
    right -= ground[:, None] * (ground.conj() @ right)[None, :]
    incoming, in_weights = gram_frame(left)
    outgoing, out_weights = gram_frame(right)
    states = sector.evolve(incoming, times, energy)
    overlap = np.einsum('di,tdj->tij', outgoing.conj(), states)
    probabilities = np.sum(np.abs(overlap) ** 2, axis=1)
    norms = np.sum(np.abs(states) ** 2, axis=1)
    z = .37 + .13j
    eye = sparse.eye(sector.dimension, format='csc')
    delta = sector.H - energy * eye
    all_sources = np.column_stack([left, right])
    plus = splu((z * eye - delta).tocsc()).solve(all_sources)
    minus = splu((z * eye + delta).tocsc()).solve(all_sources)
    chi = all_sources.conj().T @ plus - (all_sources.conj().T @ minus).T
    native = sector.retarded(ground, energy, left_ops + right_ops, z)
    # A finite physical Wilson impulse, not normalized excitation or serialized D.
    prepared = expm_multiply(1j * kick * sector.F[0], ground)
    evolved = sector.evolve(prepared, times, energy)
    observations = {'remote_flux': sector.F[2], 'remote_electric': sector.loop_electric[2],
                    'remote_density': sparse.diags(sector.active_density[:, 3] + sector.active_density[:, 4]),
                    'remote_electric_energy': sparse.diags(np.sum(sector.electric[:, np.flatnonzero(sector.faces[2])]**2, axis=1)/(2*p.inertia)),
                    'body_position': sector.body_x, 'body_velocity': sector.body_velocity}
    changes = {}
    for name, operator in observations.items():
        values = np.array([np.vdot(state, operator @ state).real for state in evolved])
        baseline = float(np.vdot(ground, operator @ ground).real)
        changes[name] = {'max_absolute_from_ground': float(np.max(np.abs(values - baseline))),
                         'initial_change': float(values[0] - baseline), 'values': (values - baseline).tolist()}
    # Independent operator/source identities, not source-text assertions.
    body_source = 1j * (sector.H @ (sector.body_x @ ground) - sector.body_x @ (sector.H @ ground))
    current_body = sum((sector.positions[b, 0] - sector.positions[a, 0]) * sector.currents['active'][l] @ ground
                       for l, (a, b) in enumerate(sector.edges))
    gauss = int(np.max(np.abs(sector.electric @ sector.B.T + sector.rho)))
    return {'parameters': vars(p), 'law': sector.__class__.__module__, 'dimension': sector.dimension,
            'energy': energy, 'lowest_energies': energies.tolist(), 'ground_residual': float(np.linalg.norm(sector.H @ ground-energy*ground)),
            'Gauss_error': gauss, 'native_constraints': sector.constraints(), 'cut_edges': sector.cut_edges,
            'incoming_gram_eigenvalues': in_weights.tolist(), 'outgoing_gram_eigenvalues': out_weights.tolist(),
            'incoming_frame_error': float(np.linalg.norm(incoming.conj().T @ incoming-np.eye(incoming.shape[1]))),
            'outgoing_frame_error': float(np.linalg.norm(outgoing.conj().T @ outgoing-np.eye(outgoing.shape[1]))),
            'transition_max_by_incoming_channel': np.max(probabilities, axis=0).tolist(),
            'transition_initial_by_channel': probabilities[0].tolist(), 'probability_bound_excess': float(max(0., np.max(probabilities)-1)),
            'unitary_norm_error': float(np.max(np.abs(norms-1))),
            'raw_remote_retarded_max': float(np.max(np.abs(chi[3:, :3]))), 'raw_chi_native_error': float(np.linalg.norm(chi-native)),
            'remote_flux_from_left_flux': complex_pair(chi[3, 0]), 'remote_electric_from_left_current': complex_pair(chi[4, 2]),
            'body_velocity_current_error': float(np.linalg.norm(body_source-current_body)),
            'spin_conditioned_local_flux': [float(np.vdot(ground, f @ ground).real) for f in sector.F],
            'finite_wilson_kick': changes}


def main() -> None:
    inputs = json.loads(Path(sys.argv[1]).read_text())
    p = Parameters(**inputs['parameters'])
    times = np.linspace(0, inputs['time_stop'], inputs['time_count'])
    cases = {'connected': p, 'true_cut': replace(p, cut=True), 'supplier_fixed_hilbert_frozen': replace(p, supplier_hop=0.),
             'neutral_spin_torque_zero': replace(p, torque=0.)}
    result = {name: probe(parameters, times, inputs['kick']) for name, parameters in cases.items()}
    save('flux2', {'kind': 'INDEPENDENT_NEW_SHARED_GAUSS_PHYSICAL_RECEIVER', 'inputs': inputs, 'cases': result,
                  'scope': 'NEW EDGE-SHARING MATERIAL FORK, not predecessor vertex-sharing identity. Literal bounded links and fixed primitive-spin block; raw response and normalized Gram transition are separately reported. Fixed anchor and spin/CAR inputs not derived. Finite kick includes ground and all excited states.'})

if __name__ == '__main__':
    main()

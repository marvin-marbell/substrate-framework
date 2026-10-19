"""Actual new finite quantum consumer plus named self-consistent bulk closure."""
import json
from pathlib import Path
import sys
import numpy as np
from rich_extensive_photon_20261014_model import matter, polarizations, line_integrals, molecular_response, collective, selfconsistent, exact_consumer


def plain(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def groups(gaps, weights, current=None, tolerance=1.e-7):
    """Sum full degenerate eigenspaces; no arbitrary bright eigenvector labels."""
    gaps = np.asarray(gaps); weights = np.asarray(weights)
    current = np.zeros_like(weights) if current is None else np.asarray(current)
    answer = []; start = 0
    while start < len(gaps):
        stop = start + 1
        while stop < len(gaps) and abs(gaps[stop] - gaps[start]) < tolerance:
            stop += 1
        w = weights[:, start:stop].sum(axis=1); j = current[:, start:stop].sum(axis=1)
        if max(w.max(), j.max()) > 1.e-7:
            answer.append(dict(energy=float(gaps[start]), degeneracy=stop-start, coordinate_residue=w.tolist(), current_weight=j.tolist()))
        start = stop
    return answer


def mode_summary(mode):
    return dict(stable=mode['stable'], minimum_squared_frequency=float(mode['eigenvalues'][0]),
                modes=groups(mode['frequencies'], mode['residues']), static_tensor=mode['static'],
                contact_tensor=mode['contact'], full_virtual_count=mode['transition_count'],
                molecular_double_occupancy=mode['response']['double_occupancy'],
                molecular_gaps=mode['response']['gaps'], current_phase_defect=mode['imaginary_phase_defect'])


def main():
    inp, out = map(Path, sys.argv[1:3]); cfg = json.loads(inp.read_text()); mat = matter(cfg)
    result = dict(schema=1, model='correlated spinful tetrahedral Hubbard molecule + actual electric-only transverse field',
                  density=dict(source_count=1, independent_cycle_count=3, cell_volume=mat['volume'], source_density=mat['source_density'],
                               loop_density=mat['independent_loop_density'], box_lengths=mat['lengths'], matter_states=16,
                               ionic_charge=2, electron_number=2, density_in_the_continuation='one identical Nup=Ndown=1 molecule per primitive volume; V_total=M V_cell and N_sources=M'),
                  exact=[], bulk={}, limitations=cfg['limitations'])
    saved = {}
    # Constant A is pure gauge on the full finite graph. Test full contact minus
    # BOTH-sign current response, not just positivity or a fitted stiffness.
    constant = line_integrals(mat, [0, 0, 0], normalized_standing=False) * cfg['charge'] / np.sqrt(cfg['epsilon'] * mat['volume'])
    response = molecular_response(mat, cfg, constant)
    static = response['contact'] - 2 * ((response['transition'].conj() / response['gaps'][None, :]) @ response['transition'].T).real
    result['Ward'] = dict(constant_A_static_tensor=static, defect=float(np.linalg.norm(static)),
                          molecular_connected_density=response['connected_density'], bare_double_occupancy=response['double_occupancy'])
    for multiplier in cfg['exact_q_harmonics']:
        q = np.array([0., 0., 2 * np.pi * multiplier / mat['lengths'][2]])
        print(json.dumps(dict(stage='exact_whole_H', q=q.tolist())), flush=True)
        exact, states = exact_consumer(mat, cfg, q)
        exact['bright_groups'] = groups(exact['gaps'], exact['coordinate_weights'], exact['current_weights'])
        exact['frozen_bright_groups'] = groups(exact['frozen_gaps'], exact['frozen_coordinate_weights'])
        exact['bare_electric_bright_groups'] = groups(exact['bare_electric_gaps'], exact['bare_coordinate_weights'])
        finite = selfconsistent(mat, cfg, q=q)
        exact['finite_gaussian_feedback'] = finite
        exact['finite_gaussian_one_pass'] = mode_summary(collective(mat, cfg, q, np.ones(6)))
        if finite['stable']:
            exact['finite_gaussian_returned'] = mode_summary(collective(mat, cfg, q, finite['dw']))
        exact['polarizations'] = polarizations(q)
        result['exact'].append(exact)
        for key, value in states.items():
            saved[f'q{multiplier}_{key}'] = value
        # Write real intermediate scientific output only; final capsule is
        # created after the producer naturally exits, never from a live result.
        out.write_text(json.dumps(result, default=plain, indent=2) + '\n')
    if cfg['wall_contrast']:
        q = np.array([0., 0., 2 * np.pi * cfg['exact_q_harmonics'][0] / mat['lengths'][2]])
        print(json.dumps(dict(stage='changed_field_wall', wall=cfg['wall_contrast'])), flush=True)
        adverse, states = exact_consumer(mat, cfg, q, box=cfg['wall_contrast'])
        adverse['bright_groups'] = groups(adverse['gaps'], adverse['coordinate_weights'], adverse['current_weights'])
        result['field_wall_contrast'] = adverse
        for key, value in states.items(): saved[f'wall_{key}'] = value
    print(json.dumps(dict(stage='self_consistent_3D_density_response')), flush=True)
    feedback = selfconsistent(mat, cfg, bulk=True); result['bulk']['feedback'] = feedback
    result['bulk']['quadrature'] = dict(spacing=cfg['bulk_mesh_spacing'], points=26,
        weight_each=cfg['bulk_mesh_spacing'] ** 3 / (2*np.pi) ** 3,
        ultraviolet_cube_halfwidth=1.5 * cfg['bulk_mesh_spacing'], excluded_central_voxel_halfwidth=0.5 * cfg['bulk_mesh_spacing'],
        meaning='midpoint d^3q/(2pi)^3 Galerkin fluctuation integral, not 26 particles or an N sweep; central IR voxel omitted explicitly')
    dw = feedback['dw']; dispersion = []
    for direction in cfg['bulk_directions']:
        direction = np.array(direction, float); direction /= np.linalg.norm(direction)
        for magnitude in cfg['bulk_q_magnitudes']:
            q = magnitude * direction
            own = collective(mat, cfg, q, dw, bulk=True)
            frozen = collective(mat, cfg, q, np.ones(6), bulk=True)
            omitted = collective(mat, cfg, q, dw, bulk=True, contact_scale=0.)
            wrong = collective(mat, cfg, q, dw, bulk=True, normalization_scale=mat['volume'])
            # Actual wrong-current normalization: J doubled while contact is
            # unchanged, equivalently four times the paramagnetic subtraction.
            wrong_current_static = own['contact'] - 4 * (own['contact'] - own['static'])
            dispersion.append(dict(q=q, polarizations=polarizations(q), returned=mode_summary(own),
                                   one_pass_undressed_matter=mode_summary(frozen), omitted_contact=mode_summary(omitted),
                                   wrong_missing_inverse_sqrt_V=mode_summary(wrong), wrong_double_current_static=wrong_current_static))
    result['bulk']['dispersion'] = dispersion
    result['bulk']['matter_return'] = dict(undressed_hopping=cfg['hopping'], dressed_bond_hopping=cfg['hopping'] * dw,
        undressed_double_occupancy=response['double_occupancy'], returned_double_occupancy=dispersion[0]['returned']['molecular_double_occupancy'],
        returned_gaps=dispersion[0]['returned']['molecular_gaps'],
        rule='own D energies and residues determine <Q Q>; that covariance changes all six Peierls hoppings, full correlated eigenstate, current/contact vertices and subsequent D at every iteration')
    # Direct decision-level comparisons, not all-positive success flags.
    result['decision_data'] = dict(
        exact_first_field_poles=[dict(q=x['q'], mixed=x['bright_groups'][:3], frozen=x['frozen_bright_groups'][:3], bare=x['bare_electric_bright_groups'][:3],
                                     purity=x['matter_purity'], boundary=x['field_boundary_probability'],
                                     reaction_double_change=max(abs(r['double_occupancy_change']) for r in x['reaction'])) for x in result['exact']],
        gaussian_is_conditional=True, exact_is_all_retained_states=True,
        full_signed_virtuals='all d-1 exact joint eigenstates and all 15 molecule eigenstates at each phase; no selected-pole current approximation')
    np.savez_compressed(out.with_suffix('.npz'), **saved)
    out.write_text(json.dumps(result, default=plain, indent=2) + '\n')
    print(json.dumps(dict(stage='complete', exact_consumers=len(result['exact']), bulk_feedback_stable=feedback['stable'],
                          bulk_feedback_converged=feedback['converged'], Ward=result['Ward']['defect'])), flush=True)


if __name__ == '__main__':
    main()

"""NEW same-law translation-sector dispersion and actual elastic EM vertices."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import rich_composite_20261014_consumer as candidate


def sector_embedding(p, positions, ns, levels, momentum):
    n = p['N']; lookup = {tuple(x): i for i, x in enumerate(positions)}
    relative = list(itertools.permutations(range(1, n), 2))
    rows, cols, values = [], [], []
    for a, rel in enumerate(relative):
        for center in range(n):
            x = (center, (center+rel[0]) % n, (center+rel[1]) % n)
            start = lookup[x]*ns*levels
            for internal in range(ns*levels):
                rows.append(start+internal); cols.append(a*ns*levels+internal)
                values.append(np.exp(1j*momentum*center)/np.sqrt(n))
    return sparse.coo_matrix((values, (rows, cols)),
                             shape=(len(positions)*ns*levels, len(relative)*ns*levels)).tocsr()


def run(p, output):
    h, positions, spins, meta = candidate.build(p)
    states = []; rows = []
    for mode in (0, 1, 2):
        q = 2*np.pi*mode/p['N']
        u = sector_embedding(p, positions, meta['ns'], meta['levels'], q)
        hk = (u.conj().T@h@u).tocsr()
        values, reduced = eigsh(hk, k=4, which='SA', tol=1e-11,
                                v0=np.linspace(0.3, 1.1, hk.shape[0]))
        order = np.argsort(values); values, reduced = values[order], reduced[:, order]
        energy = float(values[0]); multiplicity = int(np.sum(abs(values-energy)<1e-8))
        block = u@reduced[:, :multiplicity]
        # ARPACK complex-Hermitian degenerate columns need explicit orthogonalization.
        block, _ = np.linalg.qr(block)
        states.append(block)
        rows.append({'momentum_mode': mode, 'k': q, 'energy': energy,
                     'ground_block_multiplicity': multiplicity,
                     'next_internal_gap': float(values[multiplicity]-energy),
                     'full_H_residual': float(np.linalg.norm(h@block-energy*block)),
                     'translation_sector_dimension': hk.shape[0]})
    for mode in (1, 2):
        q = rows[mode]['k']; delta = rows[mode]['energy']-rows[0]['energy']
        rho = candidate.density(q, positions, meta)
        j, contact = candidate.current_contact(p, q, positions, spins, meta)
        prepared = states[0][:, 0]
        charge_vertex = states[mode].conj().T@(rho*prepared)
        current_vertex = states[mode].conj().T@(j@prepared)
        rows[mode].update(recoil_energy=delta,
                         small_k_mass_estimator=q*q/(2*delta),
                         elastic_ground_block_density_vertex_real=charge_vertex.real.tolist(),
                         elastic_ground_block_density_vertex_imag=charge_vertex.imag.tolist(),
                         elastic_ground_block_density_vertex_norm=float(np.linalg.norm(charge_vertex)),
                         elastic_ground_block_current_vertex_real=current_vertex.real.tolist(),
                         elastic_ground_block_current_vertex_imag=current_vertex.imag.tolist(),
                         elastic_Ward_vertex_norm_error=float(np.linalg.norm(delta*charge_vertex-q*current_vertex)),
                         contact_ground_expectation=float(np.vdot(prepared, contact@prepared).real))
    no_breather = dict(p); no_breather['binding_g'] = 0.0
    h0, _, _, _ = candidate.build(no_breather)
    e0, _ = candidate.ground(h0)
    controls = []
    for pair in itertools.combinations(range(3), 2):
        hp, _, _, _ = candidate.build(no_breather, pair); ep, _ = candidate.ground(hp)
        threshold = ep-2*p['hopping']
        controls.append({'pair': list(pair), 'fragment_energy': threshold,
                         'binding_below_fragment': threshold-e0})
    result = {'candidate': p['candidate'], 'parameters': p,
              'translation_dispersion_and_elastic_vertices': rows,
              'all_single_fragment_threshold_exact': -6*p['hopping'],
              'binding_below_all_single_fragments': -6*p['hopping']-rows[0]['energy'],
              'no_breather_energy': e0, 'no_breather_pair_single_controls': controls,
              'phase_note': 'one normalized k0 ground preparation; transitions projected into entire degenerate target ground block; vector norm invariant under target basis rotations',
              'mass_note': 'finite lattice dispersion curvature estimator, not relativistic/proton physical mass',
              'neutral_reference_note': 'omega n zero-point convention: vacuum=0 in each domain; pair load of singleton K=0, so a separated singleton does not independently dress the shared oscillator',
              'finite_ring_threshold_note': 'energies of fragments are same N domain; true infinite-separation threshold remains open'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'), psi_k0=states[0], psi_k1=states[1], psi_k2=states[2],
                        positions=positions, energies=np.array([x['energy'] for x in rows]))
    print(json.dumps({'dispersion': rows, 'no_breather_pair_single_controls': controls}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.request.read_text()), args.output)

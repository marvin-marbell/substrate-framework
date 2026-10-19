"""Actual transverse spin pulse and field dynamics in both full spin blocks."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, expm_multiply
from rich_induced_spin_20261014_model import Parameters, Sector, LAW
from rich_induced_spin_20261014_calculate import encode

ROOT = Path(__file__).resolve().parent

def run():
    result = {'law': LAW, 'protocol': 'Prepare lowest Sz=+1/2 eigenstate; ideal external pi pulse sigma_x at t=0 flips spin only, leaving matter/rotor/electric amplitudes unchanged; evolve that exact state with full Sz=-1/2 joint Hamiltonian. Pulse is preparation, not an emergent spin-flip interaction.', 'cases': {}, 'units': 'same as actual1; time inverse energy'}
    for name, p in [('coupled', Parameters()), ('no_medium', Parameters(medium=False)), ('no_transfer', Parameters(transfer=0.))]:
        up = Sector(p); down = Sector(replace(p, spin=-1))
        e, v = up.solve(4); ed, vd = down.solve(4); g = v[:, 0]
        # Hfull = Hup direct-sum Hdown; Sx offdiagonal identity/2.
        sxsource = g/2
        spin_response = {}
        for z in [.05+.02j, .65+.07j, 1.3+.07j]:
            dh = down.H-e[0]*sparse.eye(down.dimension)
            a = splu((z*sparse.eye(down.dimension)-dh).tocsc()).solve(sxsource)
            b = splu((z*sparse.eye(down.dimension)+dh).tocsc()).solve(sxsource)
            value = np.vdot(sxsource, a)-np.vdot(sxsource, b)
            spin_response[str(z)] = {'real': float(value.real), 'imag': float(value.imag)}
        times = np.linspace(0., 18., 37)
        waves = expm_multiply(-1j*down.H, g, start=0., stop=18., num=37)
        trajectory = [{'t': float(t), **down.state_data(w)} for t, w in zip(times, waves)]
        result['cases'][name] = {'dimension_both_spin_blocks': 2*up.dimension, 'transverse_spin_response_chi_xx': spin_response, 'initial_up': up.state_data(g), 'equilibrium_down': down.state_data(vd[:, 0]), 'postpulse_excess_energy': float(np.vdot(g, down.H@g).real-ed[0]), 'trajectory': trajectory, 'flux_min_max': [min(x['F_sinWilson'] for x in trajectory), max(x['F_sinWilson'] for x in trajectory)], 'active_current_min_max': [min(x['active_circulation'] for x in trajectory), max(x['active_circulation'] for x in trajectory)]}
        print(name, json.dumps({k: result['cases'][name][k] for k in ['dimension_both_spin_blocks', 'postpulse_excess_energy', 'flux_min_max', 'active_current_min_max', 'transverse_spin_response_chi_xx']}, default=encode), flush=True)
    (ROOT/'rich_induced_spin_20261014_actual2_result.json').write_text(json.dumps(result, default=encode, indent=2))

if __name__ == '__main__': run()

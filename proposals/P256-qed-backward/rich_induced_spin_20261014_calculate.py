"""New finite physical calculation; prints and saves all actual observables."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from rich_induced_spin_20261014_model import LAW, Parameters, Sector, charged_source

ROOT = Path(__file__).resolve().parent

def encode(x):
    if isinstance(x, np.ndarray):
        return {'real': x.real.tolist(), 'imag': x.imag.tolist()} if np.iscomplexobj(x) else x.tolist()
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x).__name__)

def run():
    base = Parameters()
    result = {'law': LAW, 'units': 'hbar=elementary_charge=triangle_edge=1; all Hamiltonian parameters energies except electric inertia inverse energy', 'cases': {}, 'response_order': ['sinWilson', 'active_circulation', 'medium_circulation', 'neutral_L'], 'source_protocol': 'H(lambda)=H-lambda sinWilson; source energy lambda, dimensionless compact flux receiver; no bare restoring term in H(0)', 'state_semantics': 'lowest eigenstate within the declared conserved active Sz block; +/- spin blocks form time reversal pair; no arbitrary degenerate-frame eigenvectors'}
    stored = {}
    cases = [('coupled_up', base), ('coupled_down', replace(base, spin=-1)), ('no_medium', replace(base, medium=False)), ('no_transfer', replace(base, transfer=0.)), ('larger_electric_basis', replace(base, cutoff=36))]
    for name, p in cases:
        s = Sector(p); e, v = s.solve(14); g = v[:, 0]
        ops = [s.F, s.Ia, s.Im, s.Ln]
        data = {'parameters': asdict(p), 'dimension': s.dimension, 'energies': e, 'state': s.state_data(g), 'constraints': s.constraints(), 'response': {str(z): s.retarded(g, e[0], ops, z) for z in [.05+.02j, .65+.07j, 1.3+.07j]}}
        data['flux_low_poles'] = [{'gap': float(e[i]-e[0]), 'weight': float(abs(np.vdot(v[:, i], s.F@g))**2)} for i in range(1, len(e))]
        if name in ('coupled_up', 'no_medium', 'no_transfer'):
            physical = []
            original = s.H.copy()
            for lam in [-.002, 0., .002]:
                s.H = original-lam*s.F
                pe, pv = s.solve(4)
                physical.append({'lambda': lam, 'energy': float(pe[0]), **s.state_data(pv[:, 0])})
            s.H = original
            data['physical_probe'] = physical
            data['static_flux_susceptibility'] = (physical[2]['F_sinWilson']-physical[0]['F_sinWilson'])/.004
            data['static_active_current_flux_feedback'] = (physical[2]['active_circulation']-physical[0]['active_circulation'])/.004
            data['energy_slope'] = (physical[2]['energy']-physical[0]['energy'])/.004
        if name == 'coupled_up':
            stored['basis'] = np.array(s.basis)
            stored['ground_up'] = g
            stored['energies_up'] = e
            stored['low_vectors_up'] = v
            vac = Sector(replace(base, active=False)); ev, vv = vac.solve(4); gv = vv[:, 0]
            sources = [charged_source(vac, s, a) for a in range(3)]
            incoming = np.column_stack([o@gv for o in sources])
            removal = np.column_stack([o.getH()@g for o in sources])
            z = .5+.07j
            addition = incoming.conj().T@splu(((z+ev[0])*sparse.eye(s.dimension)-s.H).tocsc()).solve(incoming)
            occupied_removal = removal.conj().T@splu(((z-e[0])*sparse.eye(vac.dimension)+vac.H).tocsc()).solve(removal)
            data['charged_source'] = {'vacuum_dimension': vac.dimension, 'vacuum_ground_energy': float(ev[0]), 'charged_ground_energy': float(e[0]), 'vacuum_constraints': vac.constraints(), 'addition_G_from_vacuum': addition, 'addition_source_gram': incoming.conj().T@incoming, 'occupied_removal_G': occupied_removal.T, 'occupied_removal_gram': (removal.conj().T@removal).T, 'z': str(z), 'reference': 'static -1 at vertex0 for each mobile +1; source tree paths 0->site; projection can reduce norm', 'scope': 'single-active-particle architecture: addition vacuum->N1 and occupied removal N1->vacuum are DISTINCT prepared states; NO N2 sector, NO pair/positron interpretation'}
            phases = np.array([.17, -.11, .08]); u = np.exp(1j*(s.e@phases))
            hp = s.H.copy()
            for species in ('active', 'medium'):
                for l, t in enumerate(s.transfers[species]):
                    hp += (np.exp(1j*phases[l])-1)*t+(np.exp(-1j*phases[l])-1)*t.getH()
            diff = hp-sparse.diags(u)@s.H@sparse.diags(u.conjugate())
            data['unconstrained_phase_unitary_max_error'] = float(np.max(np.abs(diff.data))) if diff.nnz else 0.
        result['cases'][name] = data
        print(name, json.dumps({'dimension': s.dimension, 'state': data['state'], 'constraints': data['constraints'], 'static_flux_susceptibility': data.get('static_flux_susceptibility'), 'static_current_feedback': data.get('static_active_current_flux_feedback')}, default=encode), flush=True)
    # A physical, clamped-holonomy supplier comparator only; never grafted into full H.
    clamped = []
    for phi in [-.01, 0., .01, .6]:
        t = np.zeros((3, 3), complex)
        for a in range(3): t[(a+1)%3, a] = -base.medium_hop*np.exp(1j*phi/3)
        bands = np.linalg.eigvalsh(t+t.conj().T)
        clamped.append({'phi': phi, 'supplier_bands': bands})
    result['clamped_supplier_comparator'] = {'meaning': 'actual 3-site charged supplier energy versus prescribed gauge-invariant holonomy; adiabatic diagnostic only, not a term supplied to joint H', 'samples': clamped, 'curvature_zero': float((clamped[0]['supplier_bands'][0]-2*clamped[1]['supplier_bands'][0]+clamped[2]['supplier_bands'][0])/.0001)}
    np.savez_compressed(ROOT/'rich_induced_spin_20261014_actual1_states.npz', **stored)
    (ROOT/'rich_induced_spin_20261014_actual1_result.json').write_text(json.dumps(result, default=encode, indent=2))

if __name__ == '__main__': run()

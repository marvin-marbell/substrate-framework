"""New same-state full current/contact response and actual CAR charge packet."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from rich_spin_flux_spatial_20261014_model import Parameters, Sector, LAW, mean, charged_source, centered_sources

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_spin_flux_spatial_20261014_actual3'


def encode(v):
    if isinstance(v, np.ndarray):
        return encode(v.tolist())
    if isinstance(v, complex):
        return [v.real, v.imag]
    if isinstance(v, dict):
        return {k: encode(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [encode(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def main():
    request = json.loads((ROOT/'rich_spin_flux_spatial_20261014_input.json').read_text())
    p = Parameters(**request['parameters'])
    states = np.load(ROOT/'rich_spin_flux_spatial_20261014_actual2_states.npz')
    previous = json.loads((ROOT/'rich_spin_flux_spatial_20261014_actual2_result.json').read_text())
    sector = Sector(p)
    g = states['joined_up_ground']
    energy = previous['cases']['joined_up']['energy']
    operators = sector.currents['active']+sector.currents['supplier']+sector.F+sector.loop_electric+[sector.body_velocity, sector.Ln]
    response = sector.retarded(g, energy, operators, .47+.13j)
    contact_active = np.array([mean(g, k) for k in sector.contacts['active']])
    contact_supplier = np.array([mean(g, k) for k in sector.contacts['supplier']])
    total_chi = response[:7, :7]+response[:7, 7:14]+response[7:14, :7]+response[7:14, 7:14]
    result = {'law': LAW, 'reference_state': 'exact actual2 joined_up ground npz bytes, no new state import from any other law', 'frequency': [.47, .13], 'operator_order': ['Ja0..6', 'Jm0..6', 'F0..2', 'Eloop0..2', 'Vactive', 'Ln'], 'full_retarded': response, 'active_contact': contact_active, 'supplier_contact': contact_supplier, 'total_current_retarded': total_chi, 'current_response_plus_contact': total_chi+np.diag(contact_active+contact_supplier), 'response_semantics': 'finite spatial Peierls current/contact derivative interface; not a continuum photon propagator or physical asymptotic S matrix', 'source_Gram': centered_sources(g, operators).conj().T@centered_sources(g, operators), 'packets': {}}
    saved = {}
    times = np.linspace(0., 20., 81)
    for name, params, spin in [('up', p, 0), ('down', p, 1), ('true_cut_up', replace(p, cut=True), 0)]:
        vacuum = Sector(params, 0, 0)
        ev, gv, _ = vacuum.ground()
        charged = Sector(params, up=int(spin == 0), down=int(spin == 1))
        source = charged_source(vacuum, charged, 1, spin)
        raw = source@gv
        gram = float(np.vdot(raw, raw).real)
        seed = raw/np.sqrt(gram)
        waves = charged.evolve(seed, times)
        prob = np.abs(waves)**2
        density = prob@charged.active_density
        remote = density[:, [3, 4]].sum(axis=1)
        electric = prob@charged.electric_energy
        current = np.array([[mean(w, j) for j in charged.currents['active']] for w in waves])
        flux = np.array([[mean(w, f) for f in charged.F] for w in waves])
        body = prob@(charged.active_density@charged.positions[:, 0])
        result['packets'][name] = {'source': 'CAR active spin creation atsite1 plus tree electric string edge0 +1 and added static anchor countercharge -1', 'vacuum_supplier_count': 1, 'vacuum_countercharge0': -1, 'charged_countercharge0': -2, 'source_Gram': gram, 'normalization': 'actual projected source norm, not guessed', 'vacuum_energy': ev, 'dimension': charged.dimension, 'spin_z': .5 if spin == 0 else -.5, 'times': times, 'active_density': density, 'remote_charge_probability': remote, 'remote_charge_max': float(remote.max()), 'initial_remote_charge': float(remote[0]), 'electric_energy': electric, 'electric_exchange_range': float(np.ptp(electric)), 'active_link_current': current, 'flux': flux, 'body_x': body, 'body_x_displacement_max': float(np.max(np.abs(body-body[0]))), 'norm_drift': float(np.max(np.abs(prob.sum(axis=1)-1))), 'constraints': charged.constraints(), 'finite_full_state_evolution': True}
        saved[name+'_seed'] = seed
        saved[name+'_vacuum'] = gv
        saved[name+'_waves'] = waves
        print(name, 'CARsourceGram', gram, 'remotecharge', float(remote.max()), 'electricexchange', float(np.ptp(electric)), flush=True)
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(encode(result), indent=2))
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'), **saved)


if __name__ == '__main__':
    main()

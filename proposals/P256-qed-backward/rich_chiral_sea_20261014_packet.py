"""NEW exact-Peierls allowed-q packet and canonical low-module Fock control."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
import rich_chiral_sea_20261014_model as law
from rich_chiral_sea_20261014_calculate import encode

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_chiral_sea_20261014_actual2'

def create_fock(modes):
    matrices = []
    for mode in range(modes):
        a = np.zeros((2 ** modes, 2 ** modes), complex)
        for state in range(2 ** modes):
            if not state & (1 << mode):
                a[state | (1 << mode), state] = (-1) ** ((state & ((1 << mode) - 1)).bit_count())
        matrices.append(a)
    return matrices

def bilinear(matrix, creation):
    return sum(matrix[i, j] * creation[i] @ creation[j].conj().T for i in range(len(creation)) for j in range(len(creation)))

def fock_control(model):
    e, v = law.spectrum(model, np.zeros(3))
    half = len(e) // 2
    ids = np.arange(half - 2, half + 2)
    w = v[:, ids]
    u = w.conj().T @ model['C'] @ w.conjugate()
    creation = create_fock(4)
    cf = np.zeros((16, 16), complex)
    for state in range(16):
        transformed = np.eye(16, dtype=complex)[:, -1]
        for i in reversed([a for a in range(4) if state & (1 << a)]):
            mapped = sum(u[j, i].conjugate() * creation[j].conj().T for j in range(4))
            transformed = mapped @ transformed
        cf[:, state] = transformed
    hf = bilinear(np.diag(e[ids]), creation)
    qf = model['params'].charge * (sum(a @ a.conj().T for a in creation) - 2 * np.eye(16))
    jf = [bilinear(w.conj().T @ a @ w, creation) for a in model['J']]
    current = [bilinear(w.conj().T @ law.vertex(model, np.zeros(3), np.zeros(3), np.eye(3)[i]) @ w, creation) for i in range(3)]
    sea = np.eye(16, dtype=complex)[:, 3]
    return {'modes': 4, 'Fock_dimension': 16, 'scope': 'Exact canonical CAR/C positive control on actual invariant lowest four rest modes; NOT replacement for full192-state response.',
            'C_square_plus_I': np.linalg.norm(model['C'] @ model['C'].conjugate() + np.eye(len(e))),
            'C_square_minus_I': np.linalg.norm(model['C'] @ model['C'].conjugate() - np.eye(len(e))),
            'canonical_unitarity': np.linalg.norm(cf.conj().T @ cf - np.eye(16)),
            'annihilator_map': max(np.linalg.norm(cf @ creation[i].conj().T @ cf.conj().T - sum(u[j, i] * creation[j] for j in range(4))) for i in range(4)),
            'vacuum_to_full': np.linalg.norm(cf[:, 0] - np.eye(16)[:, -1]),
            'charge_reversal': np.linalg.norm(cf @ qf @ cf.conj().T + qf),
            'physical_J_preservation': max(np.linalg.norm(cf @ a @ cf.conj().T - a) for a in jf),
            'H_preservation': np.linalg.norm(cf @ hf @ cf.conj().T - hf),
            'current_reversal': max(np.linalg.norm(cf @ a @ cf.conj().T + a) for a in current),
            'filled_sea_overlap_magnitude': abs(np.vdot(sea, cf @ sea))}

def packet(model, request, amplitude):
    cells = 4
    dim = len(model['H0'])
    h = np.zeros((cells * dim, cells * dim), complex)
    hp = np.zeros_like(h)
    current0 = [np.zeros_like(h) for _ in range(3)]
    currentp = [np.zeros_like(h) for _ in range(3)]
    for x in range(cells):
        sl = slice(x * dim, (x + 1) * dim)
        following = slice(((x + 1) % cells) * dim, (((x + 1) % cells) + 1) * dim)
        ay = amplitude * np.cos(np.pi * x / 2 + request['phase'])
        h[sl, sl] += model['H0']
        hp[sl, sl] += model['H0']
        for axis in (1, 2):
            t = model['T'][axis]
            phase = np.exp(-1j * model['params'].charge * (ay if axis == 1 else 0.))
            h[sl, sl] += t + t.conj().T
            hp[sl, sl] += phase * t + phase.conjugate() * t.conj().T
            current0[axis][sl, sl] += 1j * model['params'].charge * (t - t.conj().T)
            currentp[axis][sl, sl] += 1j * model['params'].charge * (phase * t - phase.conjugate() * t.conj().T)
        t = model['T'][0]
        h[sl, following] += t
        h[following, sl] += t.conj().T
        hp[sl, following] += t
        hp[following, sl] += t.conj().T
        current0[0][sl, following] += 1j * model['params'].charge * t
        current0[0][following, sl] -= 1j * model['params'].charge * t.conj().T
        currentp[0] = current0[0]
    e, v = eigh(h)
    ep, vp = eigh(hp)
    source = np.zeros(cells * dim, complex)
    # A literal local dressed source, followed by positive empty-sea projection.
    source[:dim] = model['source'][:, request['source_column']]
    prepared = v[:, e > 0] @ (v[:, e > 0].conj().T @ source)
    prepared /= np.linalg.norm(prepared)
    evolve = lambda vec, values, basis, t: basis @ (np.exp(-1j * values * t) * (basis.conj().T @ vec))
    pulsed = evolve(prepared, ep, vp, request['pulse_duration'])
    unpulsed = evolve(prepared, e, v, request['pulse_duration'])
    observables = {'current': current0,
                   'physical_Jz': [np.kron(np.eye(cells), model['J'][2])],
                   'frame_Sz': [np.kron(np.eye(cells), model['spin'][2])],
                   'internal_vector_occupation': [np.kron(np.eye(cells), model['internal_number'])],
                   'cell_population': [np.kron(np.diag(np.eye(cells)[a]), np.eye(dim)) for a in range(cells)]}
    records = []
    for time in request['free_times']:
        actual = evolve(pulsed, e, v, time)
        control = evolve(unpulsed, e, v, time)
        values = {key: {'actual': [np.vdot(actual, a @ actual).real for a in ops], 'own_baseline': [np.vdot(control, a @ control).real for a in ops], 'difference': [np.vdot(actual, a @ actual).real - np.vdot(control, a @ control).real for a in ops]} for key, ops in observables.items()}
        records.append({'time_after_pulse': time, 'norm': np.vdot(actual, actual).real, **values})
    return {'amplitude': amplitude, 'Hilbert_dimension': cells * dim, 'initial_source_energy': np.vdot(prepared, h @ prepared).real, 'occupied_reference_modes_in_active_momentum_orbit': int(np.sum(e < 0)), 'records': records,
            'meaning': 'Addition-minus-the-SAME-driven-filled-reference differential is exactly this one extra wavefunction under quadratic CAR evolution. Full4kx orbit at ky=kz=0 retains all192 internal states; other15 momentum orbits cancel in differential, NOT projected out of finite6144-particle sea. Finite Peierls Ay(x), including all contact orders, is external source not photon.'}

def main():
    request = json.loads((ROOT / 'rich_chiral_sea_20261014_packet_input.json').read_text())
    base = json.loads((ROOT / 'rich_chiral_sea_20261014_input.json').read_text())
    model = law.build(law.Parameters(**base['parameters']))
    result = {'law': model['law'], 'parameters': model['parameters'], 'Fock_C': fock_control(model), 'packet': packet(model, request, request['amplitude']), 'source_off': packet(model, request, 0.), 'C_semantics': 'The geometric X reverses frame S separately but preserves TOTAL physical orbital+frame+neutral/vector J under Fock C. C_one_particle^2=+I, not the predecessor endpoint-only C^2=-I; canonical unitary Fock map is explicitly computed. No physical-j relabeling.'}
    (ROOT / (PREFIX + '_result.json')).write_text(json.dumps(result, default=encode, indent=2) + '\n')
    print(json.dumps(result, default=encode, indent=2))

if __name__ == '__main__':
    main()

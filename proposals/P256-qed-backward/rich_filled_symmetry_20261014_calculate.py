"""NEW filled-reference computation, not an old band CLI replay."""
import json
from pathlib import Path
from itertools import product
from typing import cast
from numpy.typing import NDArray
import numpy as np
from scipy.linalg import eigh
import rich_filled_symmetry_20261014_model as law

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_filled_symmetry_20261014_actual1'


def encode(x):
    if isinstance(x, np.ndarray):
        return {'real': x.real.tolist(), 'imag': x.imag.tolist()} if np.iscomplexobj(x) else x.tolist()
    if isinstance(x, complex) or np.iscomplexobj(x):
        return [float(np.real(x)), float(np.imag(x))]
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def primitive_fock_control():
    """All16 occupation states on one A/B spinful rung, explicit canonical C.
    A positive CAR/Fock-map control, not a replacement of full576-mode law.
    """
    n = 4; dim = 2 ** n
    annih = []
    for i in range(n):
        a = np.zeros((dim, dim), complex)
        for ket in range(dim):
            if ket & (1 << i):
                a[ket ^ (1 << i), ket] = (-1) ** ((ket & ((1 << i) - 1)).bit_count())
        annih.append(a)
    tr = np.kron(np.eye(2), np.array([[0, 1], [-1, 0]]))
    grade = np.diag([1, 1, -1, -1]); c = grade @ tr
    canonical = c.T
    transformed = [cast(NDArray[np.complex128], sum(canonical[i, j] * annih[j].conj().T for j in range(n))) for i in range(n)]
    u = np.zeros((dim, dim), complex)
    full = np.zeros(dim, complex); full[-1] = 1
    for ket in range(dim):
        state = full.copy()
        for i in range(n - 1, -1, -1):
            if ket & (1 << i):
                state = transformed[i].conj().T @ state
        u[:, ket] = state
    number = sum(a.conj().T @ a for a in annih)
    spin = (annih[0].conj().T @ annih[0] - annih[1].conj().T @ annih[1] + annih[2].conj().T @ annih[2] - annih[3].conj().T @ annih[3]) / 2
    h = 2 * (annih[0].conj().T @ annih[2] + annih[2].conj().T @ annih[0] + annih[1].conj().T @ annih[3] + annih[3].conj().T @ annih[1])
    return {'dimension': dim, 'unitarity_error': np.linalg.norm(u.conj().T @ u - np.eye(dim)),
            'CAR_map_error': max(np.linalg.norm(u @ annih[i] @ u.conj().T - transformed[i]) for i in range(n)),
            'charge_reversal_error': np.linalg.norm(u @ (number - 2 * np.eye(dim)) @ u.conj().T + number - 2 * np.eye(dim)),
            'spin_preservation_error': np.linalg.norm(u @ spin @ u.conj().T - spin),
            'Hamiltonian_invariance_error': np.linalg.norm(u @ h @ u.conj().T - h)}


def main():
    request = json.loads((ROOT / 'rich_filled_symmetry_20261014_input.json').read_text())
    model = law.build(law.Parameters(**request['parameters']))
    k0 = np.zeros(3); k = np.array(request['generic_k']); q = np.array(request['pair_q'])
    e0, v0 = law.spectrum(model, k0); e, v = law.spectrum(model, k)
    n = len(e0) // 2; c = model['C']; grade = model['grade']
    mmin = float(eigh(model['M'], eigvals_only=True)[0])
    grid = np.array(list(product(request['grid_axis'], repeat=3)))
    bands = np.array([eigh(law.hamiltonian(model, p), eigvals_only=True, check_finite=False) for p in grid])
    sea_energy = float(np.sum(bands[:, :n])); sea_n = int(len(grid) * n)
    errors = {'chiral': np.linalg.norm(grade @ law.hamiltonian(model, k) @ grade + law.hamiltonian(model, k)),
              'C_H': np.linalg.norm(c @ law.hamiltonian(model, -k).conj() @ c.conj().T + law.hamiltonian(model, k)),
              'TR_H': np.linalg.norm(model['TR'] @ law.hamiltonian(model, -k).conj() @ model['TR'].conj().T - law.hamiltonian(model, k)),
              'charge_C': np.linalg.norm(-c @ np.eye(2 * n).T @ c.conj().T + np.eye(2 * n)),
              'spin_C': [np.linalg.norm(-c @ x.T @ c.conj().T - x) for x in model['spin']],
              'rotation_C': [np.linalg.norm(-c @ x.T @ c.conj().T - x) for x in model['J']],
              'C_square': np.linalg.norm(c @ c.conj() + np.eye(2 * n)),
              'M_TR': np.linalg.norm(model['TR'][:n, :n] @ model['M'].conj() @ model['TR'][:n, :n].conj().T - model['M'])}
    step = request['curvature_step']; curvature = []
    for axis in range(3):
        p = step * np.eye(3)[axis]
        ep = eigh(law.hamiltonian(model, p), eigvals_only=True)
        em = eigh(law.hamiltonian(model, -p), eigvals_only=True)
        add = float((np.mean(ep[n:n + 2]) + np.mean(em[n:n + 2]) - 2 * np.mean(e0[n:n + 2])) / step ** 2)
        rem = float((-np.mean(ep[n - 2:n]) - np.mean(em[n - 2:n]) + 2 * np.mean(e0[n - 2:n])) / step ** 2)
        curvature.append({'axis': axis, 'addition': add, 'hole_excitation': rem,
                          'scalar_center': (add - rem) / 2, 'inertial_mass': 1 / add})
    plus = v0[:, n + 1]; negative = c @ plus.conj()
    pspin = np.array([np.vdot(plus, s @ plus).real for s in model['spin']])
    hspin = -np.array([np.vdot(negative, s @ negative).real for s in model['spin']])
    pJ2 = sum(np.vdot(plus, j @ j @ plus).real for j in model['J'])
    hJ2 = sum(np.vdot(negative, j @ j @ negative).real for j in model['J'])
    pair = law.pair_response(model, k, q, complex(*request['response_z']))
    creation = law.pair_creation(model, k, request['pair_in'], request['pair_epsin'], request['pair_out'], request['pair_epsout'], request['incoming_energy'], particle=1, hole=-2)
    green = law.charged_green(model, k, .37 + .08j)
    residue = v.conj().T @ model['source']; occ = e < 0
    add_weight = residue[~occ].conj().T @ residue[~occ]
    rem_weight = residue[occ].conj().T @ residue[occ]
    low = np.r_[np.arange(n - 2, n), np.arange(n, n + 2)]
    currents = [law.vertex(model, k0, k0, np.eye(3)[i]) for i in range(3)]
    low_current = [np.linalg.norm(v0[:, low].conj().T @ j @ v0[:, low]) for j in currents]
    contact_trace = np.array([[np.trace(v[:, occ].conj().T @ law.contact(model, k, k, q, np.eye(3)[i], q, np.eye(3)[j]) @ v[:, occ]) for j in range(3)] for i in range(3)])
    kp = k + q; jk = law.vertex(model, kp, k, q)
    ward = np.linalg.norm(jk - model['params'].charge * (law.hamiltonian(model, kp) - law.hamiltonian(model, k)))
    velocity = np.array([np.vdot(v[:, n + 1], law.vertex(model, k, k, np.eye(3)[i]) @ v[:, n + 1]).real / model['params'].charge for i in range(3)])
    holestate = c @ v[:, n + 1].conj(); kh = -k
    hole_current = -np.array([np.vdot(holestate, law.vertex(model, kh, kh, np.eye(3)[i]) @ holestate).real for i in range(3)])
    result = {'law': model['law'], 'parameters': model['parameters'], 'dimensions': {'one_particle': 2 * n, 'bands_filled_per_k': n, 'molecular': n, 'finite_grid_cells': len(grid), 'filled_particles': sea_n},
              'reference': {'Esea': sea_energy, 'Nsea': sea_n, 'background_per_cell': n, 'chemical_potential': 0., 'sampled_min_addition': float(np.min(bands[:, n])), 'analytic_global_min_addition_bound': mmin, 'analytic_pair_gap_bound': 2 * mmin},
              'symmetry_errors': errors, 'literal_Fock_map_control': primitive_fock_control(),
              'charged_excitations': {'rest_addition': float(e0[n]), 'rest_removal_excitation': float(-e0[n - 1]), 'addition_charge': -1., 'hole_charge': 1., 'addition_spin': pspin, 'C_related_hole_spin': hspin, 'addition_J2': pJ2, 'C_related_hole_J2': hJ2, 'curvature': curvature, 'addition_velocity_at_k': velocity, 'addition_current_at_k': -velocity, 'C_related_hole_current_at_minus_k': hole_current, 'internal_number_addition': np.vdot(plus, model['internal_number'] @ plus).real},
              'adverse_Dirac_test': {'lowest_particle_hole_rest_projected_currents': low_current, 'cause': 'both C partner roots have same charged parity; sphere-vector bridge is odd and cannot connect them at rest; all higher pair channels generate inertia, not low4 alpha'},
              'full_G': green, 'source_addition_residue': add_weight, 'source_removal_residue': rem_weight, 'CAR_sumrule_error': np.linalg.norm(add_weight + rem_weight - model['source'].conj().T @ model['source']),
              'pair_response': {'z': request['response_z'], 'q': q, 'k': k, 'tensor': pair['tensor'], 'summed_weight': pair['summed_weight'], 'min_pair_energy': float(np.min(pair['energies'])), 'all_pair_count': pair['energies'].size},
              'sea_contact': contact_trace, 'finite_q_Ward_error': ward,
              'nonforward_real_pair_creation': creation,
              'allowed_decay': {'rest_addition_to_occupied_negative': 'Pauli blocked by literal filled Slater state', 'rest_addition_to_other_unoccupied': 'no lower energy empty state: global min is Gamma', 'pair_annihilation': 'allowed if a real hole is present; excitation charge conserved', 'within_supplied_law': 'quadratic CAR law conserves every exact eigenmode occupation; no spontaneous radiation/external photon exists', 'compatible_future_shared_field': 'below pair threshold excludes additional fermion pairs but does not alone exclude massless photon emission for moving charge; must recompute common interacting sea'},
              'scope': {'fermionic_Fock': 'full exterior algebra of576 modes/cell; no explicit2^576 enumeration; quadratic Wick formulas exact', 'bosons': 'none shared; orientor and vector oscillator levels are internal molecular species occupied/created per fermion; internal Q changes species dynamically', 'physical_spin': 'primitive spin and physical angular generators supplied; C preserves normal-ordered physical spin, not explained CAR/halfspin', 'Lorentz': 'exact C removes scalar mean of signed poles, but angular selection prevents low-rest Dirac current; lattice/multiband dispersion not Lorentz', 'pair_consumer': 'connected second-order external-source amplitude with contact, particle+hole legs and both time orders; no photon normalization'}}
    (ROOT / (PREFIX + '_result.json')).write_text(json.dumps(result, default=encode, indent=2) + '\n')
    np.savez_compressed(ROOT / (PREFIX + '_states.npz'), grid=grid, bands=bands, rest_e=e0, rest_v=v0, generic_e=e, generic_v=v, source=model['source'], pair_delta=pair['energies'], pair_amplitudes=pair['amplitudes'], sea_contact=contact_trace)
    print(json.dumps({k: result[k] for k in ['dimensions', 'reference', 'symmetry_errors', 'literal_Fock_map_control', 'charged_excitations', 'adverse_Dirac_test', 'CAR_sumrule_error', 'finite_q_Ward_error', 'nonforward_real_pair_creation']}, default=encode, indent=2))

if __name__ == '__main__':
    main()

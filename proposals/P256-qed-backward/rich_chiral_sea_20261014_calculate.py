"""NEW finite-grid full-state filled source/current/pair consumer."""
import json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
import rich_chiral_sea_20261014_model as law

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_chiral_sea_20261014_actual1'

def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, np.generic):
        return encode(value.item())
    raise TypeError(type(value).__name__)

def main():
    request = json.loads((ROOT / 'rich_chiral_sea_20261014_input.json').read_text())
    model = law.build(law.Parameters(**request['parameters']))
    zero = np.zeros(3)
    e, v = law.spectrum(model, zero)
    dim = len(e)
    half = dim // 2
    eps = np.eye(3)
    cn = model['C']
    grade = model['grade']
    k = np.array(request['source_k'])
    h = law.hamiltonian(model, k)
    errors = {'chirality_H': np.linalg.norm(grade @ h + h @ grade),
              'C_H': np.linalg.norm(cn @ law.hamiltonian(model, -k).conjugate() @ cn.conj().T + h),
              'TR_H': np.linalg.norm(model['TR'] @ law.hamiltonian(model, -k).conjugate() @ model['TR'].conj().T - h),
              'C_spin': max(np.linalg.norm(cn @ a.conjugate() @ cn.conj().T + a) for a in model['spin']),
              'C_physical_J': max(np.linalg.norm(cn @ a.conjugate() @ cn.conj().T + a) for a in model['J']),
              'C_square_plus_I': np.linalg.norm(cn @ cn.conjugate() + np.eye(dim)),
              'rest_rotation': max(np.linalg.norm(model['H0'] @ a - a @ model['H0']) for a in model['J']),
              'C_opposite_parity': np.linalg.norm(cn @ model['parity'].conjugate() @ cn.conj().T + model['parity'])}
    x, d = model['X'], model['D']
    shell = {'dimension': model['ns'], 'D_eigenvalues': eigh(d, eigvals_only=True),
             'X_square': np.linalg.norm(x @ x - np.eye(len(x))),
             'anticommutator_D_X': np.linalg.norm(d @ x + x @ d),
             'commutator_R_X': [np.linalg.norm(a @ x - x @ a) for a in model['R']],
             'anticommutator_bridge_X': [np.linalg.norm(a @ x + x @ a) for a in model['bridge']],
             'bridge_vs_infinitesimal': [np.linalg.norm(a - 1j * model['params'].angle * (d @ r - r @ d)) for a, r in zip(model['bridge'], model['R'])]}
    # Colour-Gauss source is reconstructed, not inherited as a result.
    frame = model['frame']
    fp = frame['P']
    gauss = max(np.linalg.norm(g @ fp) for g in frame['G'])
    car = max(np.linalg.norm(a.conj().T @ b + b @ a.conj().T - (np.eye(4) if i == j else np.zeros((4, 4)))) for i, a in enumerate(frame['source']) for j, b in enumerate(frame['source']))
    bare_grams = [(fp.conj().T @ a @ fp).conj().T @ (fp.conj().T @ a @ fp) for a in frame['bare']]
    lowp = v[:, half:half + 2]
    lowh = v[:, half - 2:half]
    physical_j2 = sum(a @ a for a in model['J'])
    rest = {'addition': e[half:half + 2], 'removal': -e[half - 2:half],
            'addition_parity': np.diag(lowp.conj().T @ model['parity'] @ lowp).real,
            'removed_parity': np.diag(lowh.conj().T @ model['parity'] @ lowh).real,
            'addition_J2': np.diag(lowp.conj().T @ physical_j2 @ lowp).real,
            'removed_J2': np.diag(lowh.conj().T @ physical_j2 @ lowh).real,
            'vector_occupation': np.diag(lowp.conj().T @ model['internal_number'] @ lowp).real}
    velocities = [law.vertex(model, zero, zero, a) / model['params'].charge for a in eps]
    interband = [lowp.conj().T @ a @ lowh for a in velocities]
    curvature = []
    p = v[:, half + 1]
    delta = float(request['curvature_delta'])
    for axis in range(3):
        elements = v.conj().T @ velocities[axis] @ p
        terms = np.zeros(dim)
        nondegenerate = abs(e - e[half + 1]) > 1e-8
        terms[nondegenerate] = 2 * abs(elements[nondegenerate]) ** 2 / (e[half + 1] - e[nondegenerate])
        ekp = law.spectrum(model, delta * eps[axis])[0]
        ekm = law.spectrum(model, -delta * eps[axis])[0]
        finite = (ekp[half + 1] + ekm[half + 1] - 2 * e[half + 1]) / delta ** 2
        scalar = ((ekp[half + 1] + ekp[half - 2]) / 2 + (ekm[half + 1] + ekm[half - 2]) / 2 - (e[half + 1] + e[half - 2])) / delta ** 2
        curvature.append({'axis': axis, 'full_spectral': np.sum(terms), 'finite_difference': finite,
                          'negative_all': np.sum(terms[e < 0]), 'positive_high': np.sum(terms[e > 0]),
                          'lowest_opposite_parity_pair': np.sum(terms[half - 2:half]),
                          'higher_occupied': np.sum(terms[:half - 2]), 'scalar_signed_center': scalar,
                          'rest_contact': np.vdot(p, law.contact(model, zero, zero, zero, eps[axis], zero, eps[axis]) @ p).real})
    # Full allowed periodic momentum grid supplies ACTUAL finite filled state.
    size = request['cells_per_axis']
    grid = [2 * np.pi * np.array(a) / size for a in product(range(size), repeat=3)]
    sea_energy = 0.
    global_gap = np.inf
    minima = []
    response = np.zeros((3, 3), complex)
    weight = np.zeros((3, 3), complex)
    sea_contact = np.zeros((3, 3), complex)
    q = np.array(request['response_q'])
    for kg in grid:
        eg, vg = law.spectrum(model, kg)
        sea_energy += np.sum(eg[eg < 0])
        gap = np.min(eg[eg > 0])
        global_gap = min(global_gap, gap)
        minima.append((kg, gap))
        pair = law.pair_response(model, kg, q, complex(*request['response_z']))
        response += pair['tensor'] / len(grid)
        weight += pair['summed_weight'] / len(grid)
        occupied = vg[:, eg < 0]
        for i in range(3):
            for j in range(3):
                sea_contact[i, j] += np.trace(occupied.conj().T @ law.contact(model, kg, kg, q, eps[j], q, eps[i]) @ occupied) / len(grid)
    green = law.charged_green(model, k, complex(*request['source_z']))
    ek, vk = law.spectrum(model, k)
    residues = vk.conj().T @ model['source']
    full_gram = residues.conj().T @ residues
    gr_direct = model['source'].conj().T @ np.linalg.solve(complex(*request['source_z']) * np.eye(dim) - h, model['source'])
    ward = sum(q[a] * law.vertex(model, k + q, k, eps[a]) for a in range(3)) - model['params'].charge * (law.hamiltonian(model, k + q) - h)
    channels = []
    for particle, hole in product(range(4), range(-1, -5, -1)):
        channels.append(law.pair_creation(model, np.array(request['pair_kh']), request['pair_qin'], request['pair_epsin'], request['pair_qout'], request['pair_epsout'], request['pair_omega'], particle=particle, hole=hole, eta=request['pair_eta']))
    selected = max(channels, key=lambda a: abs(a['amplitude']))
    allowed = [k, q, np.array(request['pair_kh']), np.array(request['pair_qin']), np.array(request['pair_qout']), selected['kp']]
    momentum_error = max(np.linalg.norm(a * size / (2 * np.pi) - np.round(a * size / (2 * np.pi))) for a in allowed)
    conventional = h + model['conventional_kinetic']
    conventional_error = np.linalg.norm(cn @ (law.hamiltonian(model, -k) + model['conventional_kinetic']).conjugate() @ cn.conj().T + conventional)
    conventional_rest = eigh(model['H0'] + model['conventional_kinetic'], eigvals_only=True)
    cut = law.build(law.Parameters(**{**request['parameters'], 'orientation': 0., 'vector_feedback': 0.}))
    ecut, vcut = law.spectrum(cut, zero)
    cutinter = [vcut[:, half:half + 2].conj().T @ (law.vertex(cut, zero, zero, a) / cut['params'].charge) @ vcut[:, half - 2:half] for a in eps]
    result = {'law': model['law'], 'parameters': model['parameters'], 'full_dimension': dim,
              'frame_source': {'raw_dimension': len(fp), 'physical_dimension': fp.shape[1], 'Gauss': gauss, 'dressed_CAR': car, 'bare_physical_grams': bare_grams},
              'paired_geometry': shell, 'symmetry': errors, 'rest': rest,
              'low_rest_interband_velocity': interband, 'low_rest_interband_norms': [np.linalg.norm(a) for a in interband],
              'full_curvature': curvature,
              'finite_reference': {'cells': len(grid), 'Nsea': half * len(grid), 'background_per_cell': half, 'Esea': sea_energy, 'chemical_potential': 0., 'global_finite_gap': global_gap, 'minima': [a for a, gap in minima if abs(gap - global_gap) < 1e-8], 'allowed_momentum_error': momentum_error, 'charges_addition_removal': [model['params'].charge, -model['params'].charge]},
              'G': green, 'source': {'CAR_gram_error': np.linalg.norm(full_gram - np.eye(len(full_gram))), 'retarded_direct_error': np.linalg.norm(green['retarded'] - gr_direct), 'addition_residue': residues[ek > 0].conj().T @ residues[ek > 0], 'removal_residue': residues[ek < 0].conj().T @ residues[ek < 0]},
              'finite_BZ_pair': {'q': q, 'z': complex(*request['response_z']), 'retarded_tensor_per_cell': response, 'pair_weight_per_cell': weight, 'contact_per_cell': sea_contact, 'virtuals_per_k_per_time_order': half ** 2, 'ward_error': np.linalg.norm(ward)},
              'nonforward_pair': {'channels': channels, 'largest': selected, 'largest_magnitude': abs(selected['amplitude']), 'source_scope': 'Literal filled finite4^3 grid, both occupied-hole and empty-particle intermediates in both source time orders; no photon normalization.'},
              'adverse_conventional_kinetics': {'C_H_error': conventional_error, 'rest_middle_signed_center': (conventional_rest[half] + conventional_rest[half - 1]) / 2, 'meaning': 'An ordinary positive frame/internal kinetic on this same species domain BREAKS filled C. Final correlated signed rung is a different primitive operator, NOT subtraction of this center.'},
              'dynamic_feedback_cut': {'rest_gap': ecut[half], 'full_rest_gap': e[half], 'low_velocity_norms': [np.linalg.norm(a) for a in cutinter]},
              'scope': 'Full finite quadratic CAR sea with dynamic carried frame/neutral/vector species, exact all-mode G and occupied pair response. Fixed cubic geometry, no independent shared boson, no continuous COM, no emergent photon, no interaction self-energy beyond this common quadratic law.'}
    (ROOT / (PREFIX + '_result.json')).write_text(json.dumps(result, default=encode, indent=2) + '\n')
    np.savez_compressed(ROOT / (PREFIX + '_states.npz'), H0=model['H0'], T=np.array(model['T']), C=cn, J=model['J'], parity=model['parity'], source=model['source'], rest_e=e, rest_v=v, D=d, X=x, bridge=np.array(model['bridge']))
    print(json.dumps({'law': result['law'], 'dimension': dim, 'symmetry': errors, 'rest': rest, 'low_current': result['low_rest_interband_norms'], 'curvature': curvature, 'finite_reference': result['finite_reference'], 'nonforward_largest': selected, 'adverse_conventional': result['adverse_conventional_kinetics']}, default=encode, indent=2))

if __name__ == '__main__':
    main()

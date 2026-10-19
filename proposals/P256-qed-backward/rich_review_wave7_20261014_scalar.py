"""NEW finite quantum scalar/source consumer, no author CLI replay."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply
from rich_scalar_core_20261014_model import build, sector

ROOT = Path(__file__).resolve().parent


def expect(operator, state):
    return float(np.vdot(state, operator @ state).real)


def inspect(p, coupling=None, frozen=None):
    model = build(p, coupling=coupling, frozen=frozen)
    spectra = {}
    for number in range(5):
        ix, h = sector(model, number)
        values, vectors = eigh(h.toarray(), driver='evd', check_finite=False)
        spectra[number] = (ix, h, values, vectors)
    ix, h, energies, vectors = spectra[2]
    ground = vectors[:, energies - energies[0] < 1e-8]
    ensemble = ground / np.sqrt(ground.shape[1])
    q = model['Q'][ix][:, ix]
    t = model['T'][ix][:, ix]
    scalar_probability = np.sum(abs(ensemble.reshape(6, model['nphi'], -1))**2, axis=(0, 2))
    field_parity = sp.csr_matrix(np.eye(model['nphi'])[::-1])
    channel_sign = np.diag([(-1)**((state >> 2).bit_count()) for state in range(16)])
    parity = sp.kron(channel_sign, field_parity, format='csr')[ix][:, ix]
    source_data = {}
    gmats = {}
    car = []
    for mode in range(4):
        addix = spectra[3][0]
        remix = spectra[1][0]
        creation = model['source'][mode]
        plus = creation[addix][:, ix]
        minus = creation.getH()[remix][:, ix]
        car.append(float(sp.linalg.norm(plus.getH() @ plus + minus.getH() @ minus - sp.eye(len(ix)))))
    z = complex(*p['frequency'])
    source_columns = []
    for target, addition in ((3, True), (1, False)):
        outix, _, poles, eigen = spectra[target]
        maps = [model['source'][mode] if addition else model['source'][mode].getH() for mode in range(4)]
        amplitudes = np.stack([eigen.conj().T @ (operator[outix][:, ix] @ ensemble) for operator in maps])
        weights = np.sum(abs(amplitudes)**2, axis=(0, 2))
        delta = poles - energies[0]
        source_data[str(target)] = {'gap': float(delta[0]), 'minimum_weighted_pole': float(np.min(delta[weights > 1e-10])),
                                    'total_weight': float(weights.sum()), 'all_poles': delta.tolist(), 'all_mode_summed_weights': weights.tolist()}
        denominator = z - delta if addition else z + delta
        g = np.einsum('arg,brg,r->ab', amplitudes.conj(), amplitudes, 1 / denominator)
        if not addition:
            g = g.T
        gmats[str(target)] = [[str(value) for value in row] for row in g]
        if addition:
            selected = sum(coefficient * operator[outix][:, ix] for coefficient, operator in zip(p['source_coefficients'], maps)) @ ensemble
            source_columns = selected / np.linalg.norm(selected)
    addix, addh, _, _ = spectra[3]
    spin = sum(axis * sp.kron(sp.csr_matrix(s), sp.eye(model['nphi']), format='csr') for axis, s in zip(p['spin_axis'], model['matter']['frame']['S']))[addix][:, addix]
    sval, su = eigh(spin.toarray(), check_finite=False)
    sproject = su[:, sval > 1e-8] @ su[:, sval > 1e-8].conj().T
    fieldmask = (np.tile(model['phi'], 4) > 0).astype(float)
    qadd = model['Q'][addix][:, addix]
    jadd = sum(coefficient * operator for coefficient, operator in zip(p['current_direction'], model['current']))[addix][:, addix]
    receiver = {}
    for label, operator in (('field_only', qadd), ('actual_current', jadd), ('wrong_Noether_spin', spin)):
        incoming = expm_multiply(1j * p['pulse'] * operator, source_columns)
        outgoing = expm_multiply(-1j * p['time'] * addh, incoming, traceA=-1j * p['time'] * addh.diagonal().sum())
        positive_spin = sproject @ outgoing
        negative_spin = outgoing - positive_spin
        probabilities = [float(np.linalg.norm(mask[:, None] * state)**2) for state in (positive_spin, negative_spin) for mask in (fieldmask, 1-fieldmask)]
        receiver[label] = {'probabilities_spin_field': probabilities, 'sum': sum(probabilities), 'spin': expect(spin, outgoing), 'scalar': expect(qadd, outgoing), 'current': expect(jadd, outgoing)}
    phi = model['phi']
    return {'dimensions': {str(number): len(data[0]) for number, data in spectra.items()},
            'energies_N0_to4': [float(spectra[number][2][0]) for number in range(5)],
            'reference_ground_degeneracy': ground.shape[1], 'reference_residual': float(np.linalg.norm(h @ ensemble - ensemble * energies[0])),
            'scalar_mean': expect(q, ensemble), 'scalar_variance': expect(q @ q, ensemble), 'transfer_mean': expect(t, ensemble),
            'scalar_transfer_correlation': expect(q @ t, ensemble), 'combined_parity_commutator': float(sp.linalg.norm(h @ parity - parity @ h)),
            'field_distribution': scalar_probability.tolist(), 'distribution_asymmetry': float(np.linalg.norm(scalar_probability - scalar_probability[::-1])),
            'positive_field_probability': float(scalar_probability[phi > 0].sum()), 'boundary_probability': float(scalar_probability[[0,-1]].sum()),
            'source': source_data, 'G_add_remove': gmats, 'CAR_residuals': car, 'receiver': receiver,
            'odd_S2_residual': float(sp.linalg.norm(sp.kron(sp.csr_matrix(model['matter']['S2']), sp.eye(model['nphi']), format='csr')[addix][:, addix] - .75 * sp.eye(len(addix)))),
            'input_semantics': 'Positive scalar kinetic/quadratic/quartic and g are supplied. No constant mass; finite one-cell Bloch family is not CM mobility/selfbinding.'}


def main():
    inputs = json.loads((ROOT / 'rich_review_wave7_20261014_scalar_input.json').read_text())
    p = inputs['parameters']
    quantum = inspect(p)
    frozen = float(np.sqrt(quantum['scalar_variance']))
    result = {'kind': 'NEW_FULL_QUANTUM_SCALAR_AND_SOURCE_FALSIFIER', 'inputs': inputs,
              'quantum': quantum, 'coupling_off': inspect(p, coupling=0.), 'wrong_frozen_positive_order': inspect(p, frozen=frozen),
              'frozen_value': frozen, 'meaning': 'Own complete N0..4 spectra and originating N2 ground projector. All adjacent source poles retained. Wrong fixed scalar order remains an uncoupled quantum spectator, not the true quantum law. Scalar receiver is not an EM photon event.'}
    (ROOT / 'rich_review_wave7_20261014_scalar_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: {name: value for name, value in result[key].items() if name in ('energies_N0_to4','scalar_mean','scalar_variance','scalar_transfer_correlation','combined_parity_commutator','distribution_asymmetry','CAR_residuals','receiver')} for key in ('quantum','coupling_off','wrong_frozen_positive_order')}, indent=2))


if __name__ == '__main__':
    main()

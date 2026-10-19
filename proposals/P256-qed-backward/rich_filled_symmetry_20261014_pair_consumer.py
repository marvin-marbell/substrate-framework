"""NEW distinct spin/parity channels and full virtual inertia on same filled H."""
import json
from pathlib import Path
import numpy as np
import rich_filled_symmetry_20261014_model as law
from rich_filled_symmetry_20261014_calculate import encode

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_filled_symmetry_20261014_actual2'


def main():
    request = json.loads((ROOT / 'rich_filled_symmetry_20261014_input.json').read_text())
    model = law.build(law.Parameters(**request['parameters']))
    k = np.array(request['generic_k'])
    channels = []
    for particle, hole in [(0, -1), (0, -2), (1, -1), (1, -2), (2, -1), (2, -2), (3, -1), (0, -3), (0, -4), (2, -3), (3, -4)]:
        value = law.pair_creation(model, k, request['pair_in'], request['pair_epsin'], request['pair_out'], request['pair_epsout'], request['incoming_energy'], particle=particle, hole=hole)
        channels.append({'empty_offset': particle, 'occupied_offset': hole, **value})
    e, v = law.spectrum(model, np.zeros(3)); n = len(e) // 2; p = v[:, n + 1]
    decompositions = []
    for axis in range(3):
        velocity = law.vertex(model, np.zeros(3), np.zeros(3), np.eye(3)[axis]) / model['params'].charge
        elements = v.conj().T @ velocity @ p
        terms = np.zeros(len(e)); ids = abs(e - e[n + 1]) > 1e-8
        terms[ids] = 2 * abs(elements[ids]) ** 2 / (e[n + 1] - e[ids])
        decompositions.append({'axis': axis, 'full': np.sum(terms), 'negative_full': np.sum(terms[e < 0]), 'positive_high': np.sum(terms[e > 0]), 'lowest_partner': np.sum(terms[n - 2:n]), 'higher_negative': np.sum(terms[:n - 2]), 'contact_curvature': np.vdot(p, law.contact(model, np.zeros(3), np.zeros(3), np.zeros(3), np.eye(3)[axis], np.zeros(3), np.eye(3)[axis]) @ p).real})
    pair = law.pair_response(model, k, np.array(request['pair_q']), complex(*request['response_z']))
    result = {'law': model['law'], 'parameters': model['parameters'], 'channels': channels, 'largest_amplitude_magnitude': max(abs(x['amplitude']) for x in channels), 'inertial_virtual_decomposition': decompositions, 'low_pair_transition_weight': np.sum(abs(pair['amplitudes'][:, :2, -2:]) ** 2), 'full_pair_transition_weight': np.sum(abs(pair['amplitudes']) ** 2), 'interpretation': 'Distinct spin-resolved channels use unchanged actual microscopic H and same filled reference. All occupied and empty virtual states in pair_creation, and all signed energy states in inertia. No fitted mass or offset; selected nonzero channel is a finite source consumer, not photon scattering or pair pole claim.'}
    (ROOT / (PREFIX + '_result.json')).write_text(json.dumps(result, default=encode, indent=2) + '\n')
    print(json.dumps(result, default=encode, indent=2))

if __name__ == '__main__':
    main()

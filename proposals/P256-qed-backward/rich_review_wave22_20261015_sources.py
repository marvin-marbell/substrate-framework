"""Independent full owning-H field-source moments and quantum-Phi feedback."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave22_20261015_'
SUP = 'rich_mobile_packet_20261015_actual1_'
start = time.time()
consumed = {}


def pin(name: str) -> Path:
    path = ROOT/name
    consumed[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


config = json.loads(pin(SUP+'input.json').read_text())
primitive = np.load(pin(SUP+'primitive.npz'))
initial = np.load(pin(SUP+'initial_states.npz'))
terms = [sparse.csr_matrix(t) for t in primitive['field_terms']]
boson = sparse.csr_matrix(primitive['field_boson'])
matrices = [sparse.load_npz(pin(SUP+f'neutral_Hcomponent{k}.npz')) for k in range(24)]
offset = float(primitive['reference_energies'][:90].sum())


def H(state: np.ndarray) -> np.ndarray:
    X = state.reshape(970, 225)
    out = (boson@X.T).T-offset*X
    for m, t in zip(matrices, terms):
        out += (t@(m@X).T).T
    return out.ravel()


omega = initial['neutral']
energy = float(np.vdot(omega, H(omega)).real)
records = []
sources = {}
for axis in range(8):
    q = primitive['field_q'][axis]
    full = (omega.reshape(-1, 45)@q.T).ravel()
    mean = np.vdot(omega, full)
    source = full-mean*omega
    image = H(source)-energy*source
    weight = float(np.vdot(source, source).real)
    centroid = float(np.vdot(source, image).real/weight)
    second = float(np.vdot(image, image).real/weight)
    records.append({'axis': axis, 'raw_centered_Q_source_weight': weight, 'centroid': centroid, 'second_moment': second, 'width': float(np.sqrt(max(0.0, second-centroid**2))), 'meaning': 'Complete retained owning-H source moments; not a pole or normalized-away omitted weight.'})
    sources[f'source{axis}'] = source
    sources[f'owning_HminusE_source{axis}'] = image
np.savez_compressed(ROOT/(PREFIX+'field_sources.npz'), **sources)
phi = config['phi_center']+2*config['phi_width']*(np.arange(5)-2)/5
feedback = {}
for label, state, sector in [('neutral_initial', omega, 'neutral'), ('charged_initial', initial['charged'], 'positive'), ('charged_noincoming_t8', np.load(pin(SUP+'charged_no_incoming_state16.npz'))['state'], 'positive'), ('charged_incoming_t8', np.load(pin('rich_mobile_packet_20261015_actual2_charged_incoming_state16.npz'))['state'], 'positive'), ('neutral_incoming_t8', np.load(pin('rich_mobile_packet_20261015_actual2_neutral_incoming_state16.npz'))['state'], 'neutral')]:
    masks = [int(m) for m in json.loads(pin(SUP+sector+'_masks.json').read_text())['masks']]
    X = state.reshape(-1, 5, 45)
    probability = np.sum(abs(X)**2, axis=2)
    pop = np.array([sum((m >> i)&1 for i in range(90, 180)) for m in masks])
    distribution = probability.sum(axis=0)
    mean_phi = float(distribution@phi)
    mean_pop = float(pop@probability.sum(axis=1))
    feedback[label] = {'Phi_distribution': distribution.tolist(), 'Phi_mean': mean_phi, 'Phi_variance': float(distribution@(phi**2)-mean_phi**2), 'positive_reference_band_occupancy': mean_pop, 'Phi_occupancy_covariance': float(np.einsum('mp,m,p->', probability, pop, phi)-mean_phi*mean_pop)}
result = {'kind': 'INDEPENDENT_FULL_218250_COORDINATE_SOURCES_AND_QUANTUM_PHI_FEEDBACK', 'energy': energy, 'field_source_moments': records, 'feedback': feedback, 'consumed_sha256': consumed, 'elapsed_seconds': time.time()-start, 'peak_resident_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'scope': 'No author observe/evolve import, no low-window pole substitute, no frozen quantumPhi or frozen sea. Saved-state feedback at selected anchors only, not full-history accuracy.'}
(ROOT/(PREFIX+'sources_result.json')).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'elapsed_seconds': result['elapsed_seconds'], 'peak_resident_KiB': result['peak_resident_KiB']}), flush=True)

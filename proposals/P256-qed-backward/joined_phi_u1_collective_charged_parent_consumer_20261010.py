"""Consume immutable all63 charged trial; reject projected-Ritz stationarity.

Exact serialized-amplitude covariance is separate from floating operator prices.
This does not rerun the ten-million-destination producer or adopt a Phi state.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits

import joined_phi_u1_collective_charged_ground_consumer as producer

ROOT = Path(__file__).resolve().parent
PINS = {
    'joined_phi_u1_collective_charged_ground_consumer.py': '5a84f580f49607ca945a12bb112c68f3b8cd24569b855f65af6d4b066606a783',
    'joined_phi_u1_collective_charged_ground_controls.py': '738692050c103f36923b6e5ea50800d770694d1a175e04c42ac7c1004bd4be25',
    'joined_phi_u1_collective_charged_ground_natural_exit.py': '721285a05ad30d2c27c27f49886e57b4eaa4d8f648c98b3c6eccf8941f006a1d',
    'joined_phi_u1_collective_charged_ground_delivery_receipt_20261010.json': 'c44ab260702856ae6d9b7b19cd2ab69f950d49d69a31ea4742e989fb49b0ea03',
    'joined_phi_u1_collective_charged_ground_final_20261010.json': 'fab3e2ecd2415a3fe97576a687bd07f1f1d744066c8bc593655339aa068ed0c7',
    'joined_phi_u1_collective_charged_ground_final_20261010_natural_exit.json': 'cc7d92f0fdc85dc8adeb9938bc28e8e9b3457c74bcbf7d545c76464a648146fc',
    'joined_phi_u1_collective_charged_ground_final_20261010.zip': 'a90bdcade5f19da40131691e08ceea60a1761a125cbf29e3aa5f7c41539c39e3',
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def matrix(rows: list[list[list[float]]]) -> np.ndarray:
    return np.asarray([[complex(*entry) for entry in row] for row in rows])


@lru_cache(maxsize=None)
def deviations(mask: int) -> tuple[int, ...]:
    return tuple(((mask >> (4*x)) & 15).bit_count()-2 for x in range(64))


def parse_state(rows: list[dict[str, Any]]) -> producer.State:
    state: producer.State = {}
    for row in rows:
        vertex: producer.Vertex = (
            int(row['active_mask']),
            tuple((int(mask), int(count)) for mask, count in row['auxiliary_multiset']),
            tuple((int(edge), int(count)) for edge, count in row['integer192_flow']),
        )
        assert vertex not in state
        state[vertex] = complex(*row['amplitude'])
    return state


def exact_covariance(state: producer.State) -> dict[str, Any]:
    weights = [F(a.real)**2 + F(a.imag)**2 for a in state.values()]
    denominator = max(weight.denominator for weight in weights)
    assert denominator & (denominator-1) == 0
    integers = [weight.numerator*(denominator//weight.denominator) for weight in weights]
    total = sum(integers)
    means = [0]*64
    pairs = [0]*64
    singles = [0]*64
    for (_, bath, _), weight in zip(state, integers):
        delta = [0]*64
        square = [0]*64
        for mask, count in bath:
            for x, value in enumerate(deviations(mask)):
                delta[x] += count*value
                square[x] += count*value*value
        for x in range(64):
            means[x] += weight*delta[x]
            singles[x] += weight*square[x]
            pairs[x] += weight*(delta[x]**2-square[x])
    covariance = [F(pairs[x], total*63*62)-F(means[x], total*63)**2 for x in range(64)]
    variance = [F(singles[x], total*63)-F(means[x], total*63)**2 for x in range(64)]
    assert all(value < 0 for value in covariance)
    assert all(value > 0 for value in variance)
    return dict(
        scope='EXACT_RATIONAL_EXPECTATION_OF_NORMALIZED_SERIALIZED_BINARY_AMPLITUDES_NOT_OUTWARD_MICROSCOPIC_ROOT',
        serialized_norm=str(F(total, denominator)),
        all63_pair_covariance_rational=[str(value) for value in covariance],
        all63_pair_covariance_float=[float(value) for value in covariance],
        single_copy_density_variance_float=[float(value) for value in variance],
        independent_copy_zero_covariance_rejected=True,
    )


def run(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    own = Path(__file__).read_bytes()
    inputs = {name: (ROOT/name).read_bytes() for name in PINS}
    assert all(sha(inputs[name]) == value for name, value in PINS.items())
    actual_name = 'joined_phi_u1_collective_charged_ground_final_20261010.json'
    capsule_name = 'joined_phi_u1_collective_charged_ground_final_20261010_natural_exit.json'
    actual = json.loads(inputs[actual_name])
    capsule = json.loads(inputs[capsule_name])
    assert capsule['returncode'] == 0 and capsule['natural_exit'] and capsule['all_sources_inputs_unchanged']
    assert capsule['outputs'][actual_name] == PINS[actual_name]
    with zipfile.ZipFile(ROOT/'joined_phi_u1_collective_charged_ground_final_20261010.zip') as archive:
        for group, directory in (('source_sha256', 'executed_source'), ('input_sha256', 'inputs')):
            for name, digest in actual[group].items():
                raw = archive.read(directory+'/'+name)
                assert sha(raw) == digest and (ROOT/name).read_bytes() == raw
    state = parse_state(actual['state'])
    assert all(v[0].bit_count() == 128 and all(mask.bit_count() == 128 for mask, _ in v[1]) and sum(n for _, n in v[1]) <= 63 for v in state)
    covariance = exact_covariance(state)
    reported = actual['all63_executed_state_density_correlation']
    covariance_error = max(abs(a-b) for a, b in zip(covariance['all63_pair_covariance_float'], reported['onsite_density_distinct_copy_connected_covariance']))
    assert covariance_error < 1e-14

    h = matrix(actual['matrix_shifted_h'])
    gram = matrix(actual['complete_action_gram'])
    coefficient = np.asarray([complex(*entry) for entry in actual['chosen_coefficients']])
    norm = float(np.vdot(coefficient, coefficient).real)
    energy = float(np.vdot(coefficient, h@coefficient).real/norm)
    variance = float(np.vdot(coefficient, gram@coefficient).real/norm-energy**2)
    projected_variance = float(np.vdot(h@coefficient, h@coefficient).real/norm-energy**2)
    assert abs(variance-actual['virtual_electric_superexchange']['variance']) < 1e-5
    assert variance > 1e6 and abs(projected_variance) < 1e-5
    assert not actual['accepted_as_stationary_ground']
    internal_support_variance = variance-actual['virtual_electric_superexchange']['exterior_action_norm_squared']
    assert abs(internal_support_variance-actual['controls']['wrong_omit_exterior_variance']) < 1e-5

    descriptor = json.loads((ROOT/'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json').read_bytes())['input_descriptor']
    data = producer.field.load_mean(g_override=[float(g) for g in producer.native_graph.G], mean_descriptor=descriptor)
    native = dict(data['native'])
    native['frame'] = producer.colour_frame(native['frame'])
    graph = producer.Graph(native, data['profiles'][1:3])
    sites = [tuple(int(c) for c in site) for site in native['sites']]
    index = {site: x for x, site in enumerate(sites)}
    incidence = np.zeros((64, 192), dtype=int)
    for x, site in enumerate(sites):
        for axis in range(3):
            neighbour = list(site)
            neighbour[axis] = (neighbour[axis]+1) % 4
            incidence[x, 3*x+axis] = 1
            incidence[index[tuple(neighbour)], 3*x+axis] = -1
    assert np.array_equal(incidence, native['D'])

    def gauss(vertex: producer.Vertex) -> None:
        active, bath, flow = vertex
        rho = np.asarray(deviations(active), dtype=int)
        for mask, count in bath:
            rho += count*np.asarray(deviations(mask))
        div = np.zeros(64, dtype=int)
        for edge, value in flow:
            div += value*incidence[:, edge]
        assert np.array_equal(div, rho)

    for vertex in state:
        gauss(vertex)
    locked = next(v for v in state if v[0] == graph.base and not v[2] and len(v[1]) == 2 and sum(n for _, n in v[1]) == 2)
    action = graph.action({locked: 1+0j})
    escape = next(v for v, a in action.items() if a != 0 and sum(n for _, n in v[1]) == 3 and v not in state)
    reverse = graph.action({escape: 1+0j})
    hermitian_error = abs(action[escape]-reverse[locked].conjugate())
    assert hermitian_error < 1e-8
    for vertex in (*action, *reverse):
        gauss(vertex)
    wrong_independent_HC = sum(count*64*float(np.asarray(deviations(mask))@native['Lplus']@np.asarray(deviations(mask))) for mask, count in locked[1])
    assert wrong_independent_HC > 1 and np.array_equal(graph.density(locked), np.zeros(64))
    k = producer.native_graph.dense_flow(escape[2])
    rho = graph.density(escape)
    electric_identity_error = abs(64*(np.linalg.norm(native['Q'].T@k)**2+rho@native['Lplus']@rho)-64*np.dot(k, k))
    assert electric_identity_error < 1e-7
    incoming = .37*(native['Q']@native['Q'].T@k)
    shift = 64*(np.dot(incoming+k, incoming+k)-np.dot(incoming, incoming))
    wrong_zero_incoming = 64*np.dot(k, k)
    assert abs(shift-wrong_zero_incoming) > 1
    assert all((ROOT/name).read_bytes() == raw for name, raw in inputs.items()) and Path(__file__).read_bytes() == own
    result = dict(
        kind='PARENT_CONSUMER_IMMUTABLE_ALL63_COVARIANCE_COMPLETE_GRAM_AND_NEW_NATIVE_EXTERIOR_WITNESS_NOT_GROUND',
        input_sha256=PINS, own_source_sha256=sha(own), inputs_unchanged=True,
        covariance=covariance, reported_covariance_error=covariance_error,
        full_Gram_variance=variance, wrong_three_direction_Ritz_variance=projected_variance,
        internal_full_source_support_variance=internal_support_variance,
        new_native_exterior_witness=dict(three_modified_aux=True, forward_reverse_H_error=hermitian_error, forward_amplitude=[action[escape].real, action[escape].imag], Gauss_exact=True, wrong_independent_HC=wrong_independent_HC, proper_aggregate_HC=0, electric_identity_error=float(electric_identity_error), incoming_transfer=float(shift), wrong_zero_incoming=float(wrong_zero_incoming)),
        limits='Complete ten-million H action is consumed from immutable Gram, NOT rerun/rederived here. Exact rational covariance is for serialized-state amplitudes, not an outward energy/root certificate. Newly applied fullnative H witness proves genuine further exterior with CAR/Gauss/Hermiticity. FrozenPhi/nativeintegerbeta0 only; physicalPhi/SU4/Gauss/Berry/Haar lift and fullPhi action unpaid. Stationarity, ground, photon and adoption rejected.',
        elapsed_seconds=time.monotonic()-started,
    )
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        receipt = run(args.output)
    print(json.dumps({key: receipt[key] for key in ('kind', 'reported_covariance_error', 'full_Gram_variance', 'wrong_three_direction_Ritz_variance', 'new_native_exterior_witness', 'elapsed_seconds')}, indent=2))

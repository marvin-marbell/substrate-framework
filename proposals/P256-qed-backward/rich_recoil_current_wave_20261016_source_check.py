"""Actual coefficient/reference-normalized CAR and complex source actions."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_recoil_current_wave_20261016_'


def main() -> None:
    output_stamp = sys.argv[1]
    model_path = ROOT/(PREFIX+sys.argv[2]+'_constructed_model.pkl')
    with model_path.open('rb') as stream:
        law = pickle.load(stream)
    packet = law.packet()
    coefficient_norm2 = float(np.vdot(packet, packet).real)
    records = {}
    for protocol, polarization in [('baseline', None), ('xcos', 0), ('ycos', 1)]:
        raw = {number: law.prepared(number, polarization) for number in [54, 55, 53]}
        norms = {number: float(sum(abs(psi.norm)**2 for psi in wave.values())) for number, wave in raw.items()}
        expected = coefficient_norm2*norms[54]
        actual = norms[55]+norms[53]
        if abs(actual-expected) > 1e-10*max(1., expected):
            raise ValueError('Raw source CAR weights disagree with actual coefficient and reference norms')
        records[protocol] = dict(reference_norm2=norms[54], coefficient_norm2=coefficient_norm2,
            raw_addition_norm2=norms[55], raw_removal_norm2=norms[53], actual_CAR_sum=actual,
            actual_expected_CAR_sum=expected, raw_CAR_error=actual-expected,
            unnormalized_C3_sector_weights={number: {r: float(abs(psi.norm)**2) for r, psi in wave.items()} for number, wave in raw.items()})
    mask = sum(1 << int(i) for i in np.flatnonzero(law.one['occupied']))
    field = law.field['initial']
    maximum = 0.
    count = 0
    for create in [True, False]:
        for momentum in range(3):
            coefficient = np.where(law.one['momentum'] == momentum, packet if create else packet.conj(), 0.)
            mpo = law.canonical_packet_mpo(coefficient, create, momentum)
            tensors = [mpo.get_W(i).transpose(['wL', 'wR', 'p', 'p*']).to_ndarray() for i in range(len(law.sites))]
            first = np.einsum('vpq,q->vp', tensors[0][0], field, optimize=True)
            for orbital in range(108):
                occupied = bool(mask & (1 << orbital))
                allowed = occupied != create
                bra = mask ^ (1 << orbital) if allowed else mask
                env = np.ones(1, complex)
                for i in reversed(range(108)):
                    env = tensors[i+1][:, :, (bra >> i) & 1, (mask >> i) & 1] @ env
                actual = first.T @ env
                parity = (-1)**((mask & ((1 << orbital)-1)).bit_count())
                expected = parity*coefficient[orbital]*field if allowed else np.zeros_like(field)
                maximum = max(maximum, float(np.linalg.norm(actual-expected)))
                count += 1
    if maximum > 1e-11:
        raise ValueError(f'Literal canonical source/JW complex-action failure: {maximum}')
    result = dict(kind='OWN_CLOCK0_LITERAL_CAR_SOURCE_CONSUMER', prepared_protocols=records,
        tested_complex_source_actions=count, maximum_complex_action_error=maximum,
        construction_coefficient_input=model_path.name, construction_coefficient_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        executable_model_source_sha256=hashlib.sha256((ROOT/(PREFIX+'model.py')).read_bytes()).hexdigest(),
        scope='Same-work constructed coefficient data selected explicitly as input, not an evolved state or independent proof. Fresh own clock0 references; RAW addition/removal norms versus ACTUAL packet coefficient norm times reference norm, all coherent C3 components and all108 canonical JW amplitudes. No zero component is normalized and no author-supplied CAR constant is substituted.')
    (ROOT/(PREFIX+output_stamp+'_source_check_result.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

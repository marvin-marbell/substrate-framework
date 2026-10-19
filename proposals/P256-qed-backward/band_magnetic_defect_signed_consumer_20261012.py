"""Exact bounded-principal signed controls; no native ground or fullH computation."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    request = json.loads(args.request.read_bytes())
    before = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    assert before == request['pins']
    rows = []
    for denominator in (8, 16, 32):
        q = F(1, denominator)
        # e=1 fixed; f=-i versus+i, BOTH obey the native local principal bound.
        norms = [128-2*q, 128+2*q]
        assert all(value > 127 for value in norms)
        lower = 128-F(44, 7)*q  # Rigorous weaker bound from pi<22/7.
        assert min(norms) >= lower > 0
        forbidden_ff = (64/q)**2
        assert forbidden_ff > F(22, 7)**2
        rows.append({'qIR': str(q), 'fixed_source_E_amplitude': '1', 'FF_both': '1',
                     'signed_defect_norms': [str(value) for value in norms],
                     'native_principal_lower_bound_using_pi_upper': str(lower),
                     'actual1_zero_defect_FF': str(forbidden_ff),
                     'actual1_cancellation_rejected_as_local_principal_source': True,
                     'necessary_E_amplitude_if_defect_zero_using_pi_upper': str(F(11, 224)*q)})
    after = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    assert before == after
    data = {'schema': 1, 'scope': 'exact non-native signed controls with native bounded-source constraint',
            'proof_addendum': json.loads((ROOT/request['proof']).read_bytes()), 'rows': rows,
            'source_input_sha256_before': before, 'source_input_sha256_after': after,
            'elapsed_seconds': time.monotonic()-started, 'native_actual_lowband_E_measured': False,
            'native_photon_phase_or_atom_inferred': False}
    with args.output.open('x') as stream:
        json.dump(data, stream, indent=2); stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'bounded_signed_controls': True,
                      'native_ground_backend': False, 'atom_or_phase': False}))


if __name__ == '__main__':
    main()

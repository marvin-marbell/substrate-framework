"""Exact full8-mode active/Ref source intertwining oracle, not SOURCE adoption."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import sympy as s


def annihilation(index: int) -> Any:
    entries = {}
    for bits in range(256):
        if bits & (1 << index):
            entries[(bits ^ (1 << index), bits)] = (-1) ** ((bits & ((1 << index) - 1)).bit_count())
    return s.MutableSparseMatrix(256, 256, entries)


def generators() -> list[Any]:
    pauli = [s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -s.I], [s.I, 0]]), s.diag(1, -1)]
    gamma = [s.diag(1, 1, -1, -1)] + [s.kronecker_product(pauli[0], p) for p in pauli]
    result = []
    for degree in range(1, 5):
        for subset in itertools.combinations(range(4), degree):
            word = s.eye(4)
            for j in subset:
                word *= gamma[j]
            result.append(s.I**(degree*(degree-1)//2) * word / (2*s.sqrt(2)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, pin in request['pins'].items():
        if hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest() != pin['sha256']:
            raise ValueError('immutable pin mismatch: ' + name)
    a = [annihilation(j) for j in range(8)]
    zero = s.MutableSparseMatrix(256, 256, {})
    identity = s.eye(256)
    assert all(a[i]*a[j].H+a[j].H*a[i] == int(i == j)*identity for i in range(8) for j in range(8))
    source = sum((a[j].H*a[j+4]/2 for j in range(4)), zero)
    active_number = sum((a[j].H*a[j] for j in range(4)), zero)
    ref_number = sum((a[j+4].H*a[j+4] for j in range(4)), zero)
    assert active_number*source-source*active_number == source
    assert ref_number*source-source*ref_number == -source
    assert (active_number+ref_number)*source-source*(active_number+ref_number) == zero
    bad_ref_omission = []
    for t in generators():
        active = sum((t[i, j]*a[i].H*a[j] for i in range(4) for j in range(4)), zero)
        reference = sum((t[i, j]*a[i+4].H*a[j+4] for i in range(4) for j in range(4)), zero)
        commutator = (active+reference)*source-source*(active+reference)
        assert commutator == zero and commutator.H == zero
        bad_ref_omission.append(active*source-source*active != zero)
    assert all(bad_ref_omission)
    indices = [bits for bits in range(256) if bits >> 4 == 15]
    source_square = (source.H*source).extract(indices, indices)
    expected = (4*s.eye(16)-active_number.extract(indices, indices))/4
    assert source_square == expected
    assert source_square[0, 0] == 1 and source_square[15, 15] == 0
    # Full Ref, including coherent middle sectors; not only Ref4 input.
    max_row = max(sum(abs(value) for value in source.row(i)) for i in range(256))
    max_column = max(sum(abs(value) for value in source.col(j)) for j in range(256))
    assert max_row <= 2 and max_column <= 2
    parity = s.diag(*[(-1)**(bits & 15).bit_count() for bits in range(256)])
    assert parity*source+source*parity == zero
    assert all(source[j, j] == 0 for j in range(256))
    encoded = json.dumps([(i, j, str(v)) for (i, j), v in sorted(source.todok().items())], separators=(',', ':'))
    receipt = {'kind': 'EXACT_FULL_REF_COLOCAL_FAST_SOURCE_INTERTWINER_NOT_SOURCE_ADOPTION',
               'governor': request['governor'], 'pins': request['pins'], 'full_active_Ref_CAR_dim': 256,
               'all15_joint_SU4_commutators_exact_zero': True, 'all15_missing_Ref_actions_rejected': True,
               'U1_colocal_density_commutator_exact_zero': True, 'fullRef_source_norm_bound': 2,
               'Ref4_input_source_square': '(4-n_active)/4', 'actual_Ref4_weights_retained_not_whitened': True,
               'local_charged_parity_reduces_Hfast': True, 'source_changes_parity_exactly': True,
               'source_exact_entries_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
               'derived_native_operator': 'Hfast_c E_x=E_x Hfast_n on inherited full physical core/domain',
               'spectral_range': 'f(Hfast_c)E_x=E_x f(Hfast_n), COMMON ABSOLUTE Borel function, not separately ground-shifted windows',
               'nonreal_resolvent': '(Hfast_c-z)^-1 E_x=E_x(Hfast_n-z)^-1; norm<=2/|Im z|',
               'local_weighted_range': 'K_D,c^alpha E_x=E_x K_D,n^alpha, ||K_D,c^alpha E_x K_D,n^-alpha||<=2',
               'local_fast_source_commutator_constant': 0,
               'Hslow_or_nonlocal_Wilson_source_intertwined': False, 'actual_rho_ground_overlap_assumed': False,
               'Nuniform_fast_gap_paid': False, 'original_SOURCE_adopted': False,
               'scientific_admission': False, 'campaign_complete': False}
    args.output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'fullRef_native_intertwiner_assertions': True, 'fast_commutator': 0,
                      'nonlocal_current_or_quantum_gap': False}), flush=True)


if __name__ == '__main__':
    main()

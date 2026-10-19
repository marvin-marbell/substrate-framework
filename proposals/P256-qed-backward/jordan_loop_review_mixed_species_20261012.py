"""Exact mixed-species face-word source-selection falsifier; not full-H physics."""
from __future__ import annotations

import argparse
from itertools import permutations, product
import json
from pathlib import Path
from typing import Any

from jordan_loop_review_consumer_20261012 import apply_word


def amplitude(species: tuple[int, ...], initial: int, final: int) -> dict[str, Any]:
    total = 0
    rows = []
    charges = set()
    for order in permutations(range(4)):
        state, coefficient = initial, 1
        flow = [0]*4
        for depth, edge in enumerate(order):
            role = species[edge]
            moved = apply_word(state, [(4*role+(edge+1)%4, False), (4*role+edge, True)])
            if moved is None:
                coefficient = 0
                break
            state, sign = moved
            coefficient *= sign
            flow[edge] += 1
            rho = [sum((state >> (4*role+x)) & 1 for role in range(2))-1 for x in range(4)]
            assert rho == [flow[x]-flow[(x-1)%4] for x in range(4)]
            charges.add(sum(abs(q) for q in rho))
            if depth < 3 and state in (initial, final):
                coefficient = 0  # Q removes WHOLE selected two-sector source.
                break
        value = coefficient if state == final else 0
        total += value
        rows.append({'order': list(order), 'P_to_P_word': value})
    return {'species': list(species), 'coefficient': total, 'orders': rows,
            'intermediate_l1_charge': sorted(charges)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    initial = sum(1 << mode for mode in (1, 2, 4, 7))
    final = sum(1 << mode for mode in (0, 1, 6, 7))
    assert all(sum((state >> (4*role+x)) & 1 for x in range(4)) == 2
               for state in (initial, final) for role in range(2))
    rows = [amplitude(roles, initial, final) for roles in product(range(2), repeat=4)]
    mixed = next(row for row in rows if row['species'] == [0, 0, 1, 1])
    single = sum(row['coefficient'] for row in rows if len(set(row['species'])) == 1)
    complete = sum(row['coefficient'] for row in rows)
    assert mixed['coefficient'] != 0 and single == 0 and complete != single
    result = {'kind': 'EXACT_MIXED_SPECIES_FACE_WORD_ARBITRARY_SOURCE_FALSIFIER',
              'initial_mask': initial, 'final_mask': final, 'canonical_particles_each_species': 2,
              'source_P': 'Both endpoint CAR configurations tensor all electric flows; Q excludes both internally.',
              'positive_resolvent': 'Identity on Q for this algebraic finite toy control.',
              'single_flavour_sum_offdiagonal': single, 'complete_assignment_sum_offdiagonal': complete,
              'mixed_assignment': mixed, 'all16_assignments': rows,
              'meaning': 'Generic source P does not force a single species around a closed electric face. Fixed per-copy per-site number source sectors do; the original diagonal sea diagnostics satisfy that selection rule.',
              'physical_Gauss_or_quantum_backend': False, 'campaign_complete': False}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'single_flavour_sum': single, 'complete_sum': complete,
                      'mixed_word': mixed['coefficient'], 'arbitrary_P_claim_rejected': True}))


if __name__ == '__main__':
    main()

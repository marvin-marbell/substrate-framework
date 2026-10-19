"""New all-representation physical breakup/selection and local-overlap discrimination.

This executes new states, not an archived test suite. The full local dynamic
field is retained in both the selected and capped-shared-load physical laws.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from rich_composite_local3d_20261014_consumer import build, sector_ground


def scan(h, meta):
    sectors = []
    best_energy = float('inf'); best_state = None; best_parity = None
    for parity in itertools.product((0, 1), repeat=3):
        values, states, _, _ = sector_ground(h, meta, parity, number=2)
        sectors.append(dict(parity=list(parity), lowest_energies=values.tolist()))
        if float(values[0]) < best_energy:
            best_energy = float(values[0]); best_state = states[:, 0]; best_parity = list(parity)
    return dict(ground_energy=best_energy, selected_parity=best_parity, sectors=sectors), best_state


def capped_shared_load(h, p, meta):
    """Material control: each site can receive at most one half-edge load.

    Distinct fields and their propagation remain dynamic; shared-endpoint
    constructive loading is removed. This changes only a local coupling law.
    """
    difference = meta['load']-np.minimum(meta['load'], .5)
    out = h.copy()
    for v, b in enumerate(meta['b']):
        out += p['binding_g']*sparse.kron(sparse.diags(np.repeat(difference[:, v], meta['ns'])), b+b.T, format='csr')
    return out


def run(p, output):
    h, meta = build(p)
    selected, state = scan(h, meta)
    reference_pairs = []
    for pair in itertools.combinations(range(3), 2):
        hp, mp = build(p, pair)
        record, _ = scan(hp, mp)
        reference_pairs.append(dict(pair=list(pair), **record,
                                    sum_with_single=record['ground_energy']-3*p['hopping'],
                                    binding_margin=record['ground_energy']-3*p['hopping']-selected['ground_energy']))
    hc = capped_shared_load(h, p, meta)
    capped, capped_state = scan(hc, meta)
    capped['binding_margins'] = [r['sum_with_single']-capped['ground_energy'] for r in reference_pairs]
    capped['law'] = 'Same local dynamic neutral law except L_v is capped at .5; pair laws unchanged because pair L_v<=.5.'
    capped['shared_load_energy_advantage'] = capped['ground_energy']-selected['ground_energy']
    ap = dict(p, binding_g=0.)
    ah, am = build(ap)
    adverse, adverse_state = scan(ah, am)
    adverse_pairs = []
    for pair in itertools.combinations(range(3), 2):
        hp, mp = build(ap, pair)
        record, _ = scan(hp, mp)
        adverse_pairs.append(dict(pair=list(pair), **record,
                                   sum_with_single=record['ground_energy']-3*p['hopping'],
                                   binding_margin=record['ground_energy']-3*p['hopping']-adverse['ground_energy']))
    records = dict(candidate='COMPOSITE_LOCAL_CUBE_NEUTRAL_FIELD_V1', inputs=p,
                   selected_all_graph_representations=selected, all_representation_breakup=reference_pairs,
                   no_shared_endpoint_cooperation=capped, no_field_feedback=adverse,
                   no_field_feedback_breakup=adverse_pairs,
                   thresholds_scope='Independent disconnected finite graph copies, not continuum asymptotic thresholds.',
                   source_scope='All eight exact cube XOR representations of matter+field; lowest two states per representation. Spin spaces are as in consumer.',
                   physical_result='Binding denominators now select actual lowest parity sectors; capped coupling quantifies local shared-endpoint cooperation without freezing the field.')
    Path(output+'.json').write_text(json.dumps(records, indent=2)+'\n')
    np.savez_compressed(output+'.npz', ground=state, capped_ground=capped_state, no_feedback_ground=adverse_state)
    print(json.dumps(dict(selected=selected['ground_energy'], parity=selected['selected_parity'],
                         binding=[r['binding_margin'] for r in reference_pairs],
                         capped_energy=capped['ground_energy'], capped_binding=capped['binding_margins'],
                         local_shared_advantage=capped['shared_load_energy_advantage'],
                         adverse_binding=[r['binding_margin'] for r in adverse_pairs])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.request).read_text()), args.output)

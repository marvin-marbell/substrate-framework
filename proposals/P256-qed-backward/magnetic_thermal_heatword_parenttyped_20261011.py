"""Exact native electric heat-word selection and common-character memory.

This is the full electric factor of a conditional canonical path history, not
its fullPhi/SU4/CAR thermal coefficient or pressure. It does not freeze fields.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
from typing import Any, cast
import numpy as np
import sympy as sp
from magnetic_boundary_slab_scientist_20261011 import FixedMagneticSlab, dirichlet_solve
from magnetic_boundary_scientist_20261011 import car_transfer, exact_matvec

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def face_word(slab: FixedMagneticSlab, x: int) -> dict[str, Any]:
    vertices = [slab.index[(x, 0, 0)], slab.index[(x, 1, 0)],
                slab.index[(x, 1, 1)], slab.index[(x, 0, 1)]]
    p0 = slab.path(vertices[0], 1, 1); p1 = slab.path(vertices[1], 2, 1)
    p2 = slab.path(vertices[3], 1, 1); p3 = slab.path(vertices[0], 2, 1)
    assert all(p is not None for p in (p0, p1, p2, p3))
    paths = [p0, p1, p2, p3]
    flow = [F(0)]*len(slab.dynamic)
    one = sum(15 << (4*i) for i, xyz in enumerate(slab.sites) if xyz[0] < slab.n//2)
    # A canonical rank-one component witness closes the occupation word. The
    # flow test is independent of colour and applies to the complete native hop.
    for v in vertices:
        if not one & (1 << (4*v)):
            donor = next(j for j in range(4*slab.v) if one & (1 << j) and j//4 not in vertices)
            one ^= 1 << donor
            one |= 1 << (4*v)
    empty = [i for i in range(slab.v) if not one & (1 << (4*i))][:3]
    for v, target in zip(vertices[1:], empty, strict=True):
        one ^= 1 << (4*v); one |= 1 << (4*target)
    assert one.bit_count() == 2*slab.v
    global_mask = sum(one << (4*slab.v*s) for s in range(65)) | (15 << (4*slab.v*65))
    mask, signs = global_mask, []
    actions = [(vertices[3], vertices[0], p3, -1),
               (vertices[2], vertices[3], p2, -1),
               (vertices[1], vertices[2], p1, 1),
               (vertices[0], vertices[1], p0, 1)]
    for create, annihilate, path, sign_phase in actions:
        assert path is not None
        action = car_transfer(mask, 4*annihilate, 4*create)
        assert action is not None
        mask, sign = action
        signs.append(sign)
        for edge in path['edges']:
            if edge in slab.dynamic_lookup:
                flow[slab.dynamic_lookup[edge]] += sign_phase
    assert mask == global_mask
    assert not any(exact_matvec(slab.DI, flow))
    trace_eligible = not any(flow)
    assert trace_eligible == (x in (0, slab.n-1))
    return dict(boundary_x=x, canonical_CAR_word_closes=True, canonical_species_counts=2*slab.v,
                CAR_sign_product=int(np.prod(signs)), full_colours_scope='Flow criterion independent of ordered fullPhi/SU4 coefficient and all colour contractions',
                net_dynamic_flow=[[slab.dynamic[j], str(t)] for j, t in enumerate(flow) if t],
                divergence_zero=True, electric_heat_trace_selection_pass=trace_eligible,
                interior_closed_CAR_but_nonzero_cycle_trace_zero=not trace_eligible,
                surviving_boundary_loop_coefficient_not_evaluated=trace_eligible)


def common_b_memory(slab: FixedMagneticSlab) -> dict[str, Any]:
    n = slab.n
    k = [F(0)]*len(slab.dynamic)
    # Complementary32 pairs exactly cancel rho before the word. Only active
    # species moves one colour per yz column across boundary0 -> interior1.
    signs = []
    one = sum(15 << (4*i) for i, xyz in enumerate(slab.sites) if xyz[0] % 2 == 0)
    complement = ((1 << (4*slab.v))-1)^one
    masks = [one if s % 2 == 0 else complement for s in range(64)]+[one]
    initial = sum(a << (4*slab.v*s) for s, a in enumerate(masks)) | (15 << (4*slab.v*65))
    mask = initial
    for y, z in np.ndindex(n, n):
        start = slab.index[(0, y, z)]
        path = slab.path(start, 0, 1)
        assert path is not None
        action = car_transfer(mask, 4*start, 4*path['target'])
        assert action is not None
        mask, sign = action
        signs.append(sign)
        cast(Any, k)[slab.dynamic_lookup[path['edges'][0]]] = -1
    assert ((mask & ((1 << (4*slab.v))-1))).bit_count() == 2*slab.v
    rho = exact_matvec(slab.DI, k)
    assert all(rho[j] == (1 if slab.sites[site][0] == 1 else 0) for j, site in enumerate(slab.interior))
    row_phi = dirichlet_solve([F(1)]+[F(0)]*(n-3))
    phi = [row_phi[slab.sites[site][0]] for site in slab.interior]
    longitudinal = exact_matvec(slab.DI.T, phi)
    assert exact_matvec(slab.DI, longitudinal) == rho
    qk = [a-b for a, b in zip(k, longitudinal, strict=True)]
    assert not any(exact_matvec(slab.DI, qk))
    kl2 = sum((a*a for a in longitudinal), F(0))
    kt2 = sum((a*a for a in qk), F(0))
    assert kl2 == F(n*n*(n-2), n-1) and kt2 == F(n*n, n-1)
    assert kl2+kt2 == n*n
    # Word has inc-k then+k; interval halfbeta in each charge configuration.
    # Common b integration completes ONE Gaussian square. Fresh b per slice
    # falsely erases this exact transverse variance.
    memory = F(1, 4)*kt2
    longitudinal_price = F(1, 2)*kl2
    reset_defect_exponent = 64*memory
    assert reset_defect_exponent > 0
    return dict(N=n, initial_fullrho_exact_zero=True, all64_complementary_pairs_retained=True,
                active_native_plane_hops=n*n, intermediate_rho='1 on interior layerx1',
                intermediate_HC64=str(64*kl2), intermediate_transverse_norm2=str(kt2),
                cumulative_flow_final_exact_zero=True,
                beta_temperature=1, common_b_gaussian_memory_price=str(memory),
                longitudinal_time_price=str(longitudinal_price),
                electric_heat_factor='tE(beta)*exp(-64*longitudinal_time_price-64*memory_price)',
                wrong_fresh_b_each_slice_overcounts_exact_ratio=f'exp({reset_defect_exponent})',
                reset_b_wrongcase_rejected=True,
                scope='EXACT_NATIVE_FULL_E_FACTOR_CONDITIONAL_ON_THIS_CAR_FLOW_HISTORY_NOT_FULL_THERMAL_WORD_COEFFICIENT')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    assert all(digest(ROOT/name) == pin for name, pin in request['pins'].items())
    result: dict[str, Any] = dict(kind='NATIVE_CANONICAL_HEAT_WORD_ZERO_FLOW_AND_COMMON_CHARACTER_MEMORY_NOT_PRESSURE',
                                source_sha256=digest(Path(__file__)), N_controls=[])
    for n in request['N']:
        slab = FixedMagneticSlab(n, 1)
        result['N_controls'].append(dict(N=n, interior_face=face_word(slab, 1),
                                         boundary_face=face_word(slab, 0),
                                         common_character_memory=common_b_memory(slab)))
    assert all(digest(ROOT/name) == pin for name, pin in request['pins'].items())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(N=request['N'], interior_closed_loop_trace='zero',
                          boundary_loop='electric selection passes, quantum coefficient unpaid', common_b_memory='paid exactly')))


if __name__ == '__main__':
    main()

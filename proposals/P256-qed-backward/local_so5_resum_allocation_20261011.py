"""Exact normalized matching allocation; deliberately fails norm-based physical1 bound."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import sys
import zipfile
from fractions import Fraction


def count(n: int) -> dict:
    sites = list(itertools.product(range(n), repeat=3))
    def step(x: tuple, axis: int, distance: int) -> tuple:
        y = list(x); y[axis] = (y[axis]+distance) % n
        return tuple(y)
    def block(x: tuple) -> tuple:
        return x[0]//2, x[1], x[2]
    target = (0, 0, 0)
    records = {}
    for length in [1, 2]:
        touching = []
        total = Fraction(0)
        for x in sites:
            for axis in range(3):
                vertices = [step(x, axis, j) for j in range(length+1)]
                blocks = {block(y) for y in vertices}
                if target in blocks:
                    selected = length == 1 and len(blocks) == 1
                    weight = Fraction(1, len(blocks))
                    if not selected:
                        total += weight
                    touching.append({'start': x, 'axis': axis, 'support': vertices, 'blocks': sorted(blocks), 'weight': str(weight), 'selected_matching_edge': selected})
        records[str(length)] = {'touching_paths': len(touching), 'boundary_allocation_exact': str(total), 'records': touching}
    assert records['1']['touching_paths'] == 11
    assert records['1']['boundary_allocation_exact'] == '5'
    assert records['2']['touching_paths'] == 16
    assert records['2']['boundary_allocation_exact'] == '6'
    # Sum_X 1/count(blocks(path)) over touched blocks is1 for EVERY path.
    for length in [1, 2]:
        for x in sites:
            for axis in range(3):
                blocks = {block(step(x, axis, j)) for j in range(length+1)}
                assert sum(Fraction(1, len(blocks)) for _ in blocks) == 1
    return {'N': n, 'volume': n**3, 'matching_blocks': n**3//2, 'allocation': records, 'every_path_allocated_once': True}


def main() -> None:
    root = Path(__file__).resolve().parent
    raw = Path(__file__).read_bytes()
    output = {'kind': 'EXACT_LINKED_MATCHING_SUPPORT_ALLOCATION_NOT_STABILITY', 'source_sha256': hashlib.sha256(raw).hexdigest(), 'actual_command': [sys.executable, *sys.argv], 'finite_controls': [count(n) for n in [4, 6, 8]], 'N_uniform': 'Same local enumeration for every even N>=4:11 one-link paths touch a dimer (1 selected+10 external), and16 distinct length2 paths. Reciprocal touched-block multiplicity allocates5 external one-link and6 length2 path Hamiltonians per block exactly once. All length2 paths are boundary paths.', 'normalized_disjoint_matter_product': 'finite tensor product of norm1 states has norm1; full Gauss/electric/field embedding remains unpaid', 'operator_norm_condition': '5*K_one_link +6*K_two_link +K_shared_electric+K_field < certified_block_gap; no term omitted in proposed criterion', 'physical1_verdict': 'The main full128 receipt already has5*K_one_link/optimistic_bare_onsite_gap>1 before any two-link/electric/field addition. Thus this sufficient norm criterion fails; this is NOT a no-go and there is no ground-gap certificate.', 'corrects': 'two-link start/end-only incidence misses middle vertices. Correct total is18 vertex-path incidences minus2 doubly touched paths=16 distinct paths, not12.', 'next_construction': 'Connected boundary excitation allocation or larger gauge-compatible clusters, not a retuned microscopic coupling.', 'source_unchanged': Path(__file__).read_bytes() == raw}
    path = root/'local_so5_resum_allocation_proof_20261011.json'
    path.write_text(json.dumps(output, indent=2)+'\n')
    with zipfile.ZipFile(root/'local_so5_resum_allocation_capsule_20261011.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(Path(__file__).name, raw)
        archive.writestr(path.name, path.read_bytes())
    print(json.dumps({'proof': path.name, 'source_sha256': output['source_sha256'], 'source_unchanged': output['source_unchanged']}), flush=True)


if __name__ == '__main__':
    main()

"""Strict completed cut3 charged loader; caller provenance is not producer physics."""
from __future__ import annotations
from pathlib import Path
import json
import sys
from typing import Any
import numpy as np
import covariant_collective_cut3_charged as supplier

base = supplier.base
ROOT = supplier.ROOT
PINNED = supplier.PINNED
check_inputs = supplier.check_inputs
descriptor_hash = supplier.descriptor_hash
digest = supplier.digest
file_hash = supplier.file_hash
physics_descriptor = supplier.physics_descriptor
basis_description = supplier.basis_description
canonical_physical_descriptor = supplier.canonical_physical_descriptor

def load_spectrum(capsule_path: Path) -> tuple[base.Spectrum, dict[str, Any]]:
    """Verify immutable inputs and generated output bytes; reconstruct P without SVD."""
    check_inputs()
    capsule = json.loads(Path(capsule_path).read_text())
    descriptor = capsule['input_descriptor']
    if descriptor_hash(descriptor) != capsule['input_descriptor_sha256']:
        raise ValueError('Descriptor hash mismatch')
    if descriptor['charge'] != 1 or descriptor['cutoff'] != 3 or descriptor['grid'] != 32 or descriptor['UE'] != 1.:
        raise ValueError('Not the fixed N5 cut3 grid32 UE1 law')
    if digest(capsule['plan_bytes_at_import'].encode()) != descriptor['raw_plan_sha256_at_IMPORT']:
        raise ValueError('Historical plan provenance mismatch')
    _, _, nodes, _, _, metric = base.left.frame()
    if descriptor != physics_descriptor(nodes, descriptor['raw_plan_sha256_at_IMPORT']):
        raise ValueError('Fixed charged native physics descriptor mismatch')
    if capsule['physical_mean_descriptor'] != metric:
        raise ValueError('Conditional mean metric mismatch')
    required_sources = set(base.SOURCE_BYTES_AT_IMPORT)
    caller = getattr(sys.modules.get('__main__'), '__file__', None)
    # The importer is separately captured by its consumer, never a supplier input.
    if isinstance(caller, str):
        required_sources.discard(Path(caller).name)
    required_sources.add(Path(supplier.__file__).name)
    if not required_sources.issubset(capsule['source_at_import']) or set(capsule['input_at_import']) != set(PINNED):
        raise ValueError('Incomplete captured source/input provenance')
    for name, expected in capsule['source_at_import'].items():
        if digest(capsule['source_bytes_at_import'][name].encode()) != expected:
            raise ValueError('Captured source bytes mismatch')
        if file_hash(ROOT / name) != expected:
            raise ValueError(f'Source mismatch: {name}')
    for name, expected in capsule['input_at_import'].items():
        if digest(capsule['input_bytes_at_import'][name].encode()) != expected:
            raise ValueError('Captured input bytes mismatch')
        if file_hash(ROOT / name) != expected or expected != PINNED[name]:
            raise ValueError(f'Input mismatch: {name}')
    physical = base.space(1, 3)
    blocks = []
    for record in capsule['blocks']:
        sign = record['sign']
        sector = base.copy_sector(physical, sign)
        if record['basis'] != basis_description(sector):
            raise ValueError('Canonical physical/copy basis mismatch')
        path = Path(record['generated_U']['path'])
        shape = tuple(record['generated_U']['shape'])
        dimension = sector['embedding'].shape[1]
        if record['generated_U']['dtype'] != 'complex128' or record['generated_U']['order'] != 'F' or not path.is_absolute():
            raise ValueError('Generated U ABI mismatch')
        if shape != (dimension, dimension) or path.stat().st_size != dimension * dimension * 16:
            raise ValueError('Generated U shape/size mismatch')
        if file_hash(path) != record['generated_U']['sha256']:
            raise ValueError('Generated U binary mismatch')
        q = np.asarray(record['original_sigma_per_column'], dtype=float)
        if len(q) != dimension or not np.all(np.isfinite(q)) or np.any(q <= 0):
            raise ValueError('Original singular directions missing/nonpositive')
        vectors = np.memmap(path, mode='r', dtype=np.complex128, shape=shape, order='F')
        blocks.append(dict(V=vectors, q=q, P=sector['embedding'], sign=sign))
    if canonical_physical_descriptor(physical) != capsule['canonical_geometry']:
        raise ValueError('Canonical geometry mismatch')
    dimension = 3 * len(physical['physical'])
    if [b['sign'] for b in blocks] != [1, -1] or sum(len(b['q']) for b in blocks) != dimension:
        raise ValueError('Both complete copy sectors required')
    return base.Spectrum(blocks, dimension), dict(physical=canonical_physical_descriptor(physical), input_descriptor=descriptor,
        factor_diagnostics=capsule['blocks'], physical_mean_descriptor=capsule['physical_mean_descriptor'])

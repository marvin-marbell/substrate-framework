#!/usr/bin/env python3
"""Complete N5 two-copy cut2 factor; persisted unsquared left singular system."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
import scipy
from scipy.linalg import svd
from threadpoolctl import threadpool_limits, threadpool_info
import covariant_native_collective_primitive_electric as base

ROOT = Path(__file__).resolve().parent
PLAN = base.PLAN_PATH
PLAN_BYTES_AT_IMPORT = PLAN.read_bytes()
SOURCE_BYTES_AT_IMPORT = dict(base.SOURCE_BYTES_AT_IMPORT)
for module in tuple(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if filename:
        path = Path(filename).resolve()
        if path.parent == ROOT and path.suffix == '.py':
            SOURCE_BYTES_AT_IMPORT[path.name] = path.read_bytes()
SOURCE_BYTES_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()
PINNED = {
    'covariant_native_collective_primitive_electric.py': '085c316c7ae4cf64d4cb9e09ee8429294db191602f26addacaabd0626c5681ea',
    'covariant_collective_paired_source_current.py': '2bbff23115ffc3d70a12f6da16a8c3c0a7ca0b1868bc5f5ecfcc7f6b7ca1e0d5',
    'covariant_collective_paired_car.py': 'cc40c5bbbc39ba240051759e9790800948ba6a51080d766f0fa7b2d7ed264e1d',
    'covariant_native_collective_primitive_electric_cut1_UE1_capsule_20261008.json': '6000a6b4a6d9e6e91c9b1f0c607a3b4708c41c878a7aa24cf08ff17182412b05',
    'covariant_collective_paired_source_current_cut1_UE1_capsule_20261009.json': 'af2fb68e0b98152aa34140bb0849b51db53a34c612ae056e9eeac9b8fc37f1de',
    'covariant_collective_paired_car_capsule_20261009.json': '517a539a51759fe1cb5e45668b3c760bbcf1201abfc83ea2cbf4884748900acf',
}
INPUT_BYTES_AT_IMPORT = {name: (ROOT / name).read_bytes() for name in PINNED}

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def descriptor_hash(value: dict[str, Any]) -> str:
    return digest(json.dumps(value, sort_keys=True).encode())

def check_inputs() -> None:
    for name, expected in PINNED.items():
        if digest(INPUT_BYTES_AT_IMPORT[name]) != expected:
            raise ValueError(f'Pinned input mismatch: {name}')

def load_spectrum(capsule_path: Path) -> tuple[base.Spectrum, dict[str, Any]]:
    """Verify immutable inputs and generated output bytes; reconstruct P without SVD."""
    check_inputs()
    capsule = json.loads(Path(capsule_path).read_text())
    descriptor = capsule['input_descriptor']
    if descriptor_hash(descriptor) != capsule['input_descriptor_sha256']:
        raise ValueError('Descriptor hash mismatch')
    if descriptor['charge'] != 1 or descriptor['cutoff'] != 2 or descriptor['grid'] != 32 or descriptor['UE'] != 1.:
        raise ValueError('Not the fixed N5 cut2 grid32 UE1 law')
    for name, expected in capsule['source_at_import'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f'Source mismatch: {name}')
    for name, expected in capsule['input_at_import'].items():
        if file_hash(ROOT / name) != expected or expected != PINNED[name]:
            raise ValueError(f'Input mismatch: {name}')
    physical = base.space(1, 2)
    blocks = []
    for record in capsule['blocks']:
        sign = record['sign']
        sector = base.copy_sector(physical, sign)
        if record['basis'] != basis_description(sector):
            raise ValueError('Canonical physical/copy basis mismatch')
        path = Path(record['generated_U']['path'])
        shape = tuple(record['generated_U']['shape'])
        dimension = sector['embedding'].shape[1]
        if shape != (dimension, dimension) or path.stat().st_size != dimension * dimension * 16:
            raise ValueError('Generated U shape/size mismatch')
        if file_hash(path) != record['generated_U']['sha256']:
            raise ValueError('Generated U binary mismatch')
        q = np.asarray(record['original_sigma_per_column'], dtype=float)
        if len(q) != dimension or not np.all(np.isfinite(q)) or np.any(q <= 0):
            raise ValueError('Original singular directions missing/nonpositive')
        vectors = np.memmap(path, mode='r', dtype=np.complex128, shape=shape, order='F')
        blocks.append(dict(V=vectors, q=q, P=sector['embedding'], sign=sign))
    dimension = 3 * len(physical['physical'])
    if [b['sign'] for b in blocks] != [1, -1] or sum(len(b['q']) for b in blocks) != dimension:
        raise ValueError('Both complete copy sectors required')
    return base.Spectrum(blocks, dimension), dict(physical=physical, input_descriptor=descriptor,
        factor_diagnostics=capsule['blocks'], physical_mean_descriptor=capsule['physical_mean_descriptor'])

def basis_description(sector: dict[str, Any]) -> dict[str, Any]:
    return dict(sign=sector['sign'], pairs=[list(p) for p in sector['pairs']],
        light_ids=sector['light_ids'].tolist(), pair_ids=sector['pair_ids'].tolist(),
        flux=sector['flux'].tolist(), electric=sector['electric'].tolist(),
        representative_rows=sector['representative_rows'].tolist(), Phi_order='three contiguous physical blocks')

def small_control(gamma: np.ndarray, nodes: np.ndarray, heat: np.ndarray, scratch: Path) -> list[dict[str, Any]]:
    physical = base.space(1, 0)
    rows = []
    for sign in (1, -1):
        sector = base.copy_sector(physical, sign)
        path = scratch / f'N5_N2_cut0_control_copy{sign:+d}.complex128'
        factor, haar, _ = base.streamed_factor(gamma, nodes, heat, physical, sector, 32, 1., path)
        before = np.array(factor, order='F')
        U, reference, _ = svd(before, full_matrices=False)
        q, abi = base.left.lapack_left(factor)
        vector = np.exp(1j * np.arange(len(q))) / np.sqrt(len(q))
        expected = before @ (before.conj().T @ vector)
        observed = factor @ (q * q * (factor.conj().T @ vector))
        defect = float(np.linalg.norm(expected - observed) / np.linalg.norm(expected))
        sigma_defect = float(np.max(abs(q - reference)) / reference[0])
        if defect > 1.e-11 or sigma_defect > 1.e-11 or q[-1] <= 0:
            raise ArithmeticError('Actual N2 cut0 overwrite action failed')
        rows.append(dict(sign=sign, dimension=len(q), operator_action_relative_defect=defect,
                         sigma_relative_defect=sigma_defect, overwrite_abi=abi))
        del factor, haar, U
        path.unlink()
    return rows

def run(scratch: Path) -> dict[str, Any]:
    check_inputs()
    scratch.mkdir(parents=True, exist_ok=True)
    gamma, embedding, nodes, heat, Hphi, metric = base.left.frame()
    controls = small_control(gamma, nodes, heat, scratch)
    identities = base.validate_copy_identity(gamma, nodes)
    physical = base.space(1, 2)
    blocks = []
    for sign in (1, -1):
        sector = base.copy_sector(physical, sign)
        path = (scratch / f'N5_N2_copy{sign:+d}_cut2_leftU.complex128').resolve()
        if path.exists():
            raise FileExistsError(path)
        print(json.dumps(dict(stage='factor', sign=sign, dimension=sector['embedding'].shape[1])), flush=True)
        factor, haar, diagnostics = base.streamed_factor(gamma, nodes, heat, physical, sector, 32, 1., path)
        del haar
        print(json.dumps(dict(stage='unsquared_JOBUO_JOBVTN', sign=sign)), flush=True)
        q, abi = base.left.lapack_left(factor)
        if len(q) != factor.shape[1] or np.any(q <= 0):
            raise ArithmeticError('Incomplete/nonpositive singular directions')
        base.left.evict(factor)
        generated = dict(path=str(path), shape=list(factor.shape), dtype='complex128', order='F',
                         sha256=file_hash(path), provenance='GENERATED output after overwrite SVD, not imported input')
        blocks.append(dict(sign=sign, basis=basis_description(sector), original_sigma_per_column=q.tolist(),
                           generated_U=generated, factor_physics=diagnostics, overwrite_SVD=abi))
        del factor
        print(json.dumps(dict(stage='block_complete', sign=sign, rank=len(q), minimum=float(q[-1]))), flush=True)
    dimension = 3 * len(physical['physical'])
    if sum(len(b['original_sigma_per_column']) for b in blocks) != dimension:
        raise ArithmeticError('Canonical completeness failure')
    descriptor = dict(charge=1, active_particles=5, cutoff=2, grid=32, UE=1.,
        raw_plan_sha256_at_IMPORT=digest(PLAN_BYTES_AT_IMPORT),
        general_plan_pin='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        ALL36=True, ONE_global_Gauss=True, BOTH_copy_sectors=[1,-1], auxiliary_basis_dimensions=[21,15],
        all_four_edges_bounded=True, active_kappa=.09, auxiliary_kappa_each=.04, mean_metric_power=400,
        auxiliary_path_weights=[.5,.5], light_homotopy=0., Phi_nodes=nodes.tolist(), Phi_sigma=.03, Phi_radius=.12,
        Ref_modes=4, Ref_initial=4, Ref_prepared=3, Ref_static_site=0,
        F='PhiheatHalf SF_fullGamma exp[-sum_ALL4edges E^2/2]', K='F Fdag',
        no_charge_conjugation_halving=True, no_Gram_clipping_pruning=True,
        scope='finite native one-cycle ALLCAR/Phi3; not massless/commoncone/untruncated EM')
    return dict(input_descriptor=descriptor, input_descriptor_sha256=descriptor_hash(descriptor),
        source_at_import={n:digest(b) for n,b in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={n:b.decode() for n,b in SOURCE_BYTES_AT_IMPORT.items()},
        input_at_import={n:digest(b) for n,b in INPUT_BYTES_AT_IMPORT.items()},
        input_bytes_at_import={n:b.decode() for n,b in INPUT_BYTES_AT_IMPORT.items()},
        plan_bytes_at_import=PLAN_BYTES_AT_IMPORT.decode(), physical_mean_descriptor=metric,
        blocks=blocks, dimension=dimension, physical_rows=len(physical['physical']),
        small_actual_N2_cut0_controls=controls, actual_SF_copy_identities=identities,
        runtime_packages=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
            platform=platform.platform(), threadpools=threadpool_info()), actual_cli=sys.argv,
        environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.monotonic()
    with threadpool_limits(limits=args.threads):
        result = run(args.scratch)
    result['runtime_seconds'] = time.monotonic() - start
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    spectrum, _ = load_spectrum(args.output)
    print(json.dumps(dict(stage='natural_result_written_and_loaded', output=str(args.output),
        dimension=spectrum.dimension, rank=len(spectrum.q), minimum=float(spectrum.q[-1]),
        runtime_seconds=result['runtime_seconds'])), flush=True)

if __name__ == '__main__':
    main()

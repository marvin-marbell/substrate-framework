#!/usr/bin/env python3
"""Complete compact-electric cut3 neutral factor; lossless complex descriptor."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, cast
import numpy as np
import scipy
from scipy.sparse import block_diag, eye
from threadpoolctl import threadpool_limits
import covariant_collective_cut2_neutral as frozen

base = frozen.base
ROOT = Path(__file__).resolve().parent
IMPORT_EPOCH_NS = time.time_ns()
SOURCE_AT_IMPORT = {}
INPUT_AT_IMPORT = {'plan': base.PLAN_BYTES_AT_IMPORT}
for module in tuple(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if isinstance(filename, str):
        path = Path(filename).resolve()
        if path.parent == ROOT and path.suffix == '.py':
            SOURCE_AT_IMPORT[path.name] = path.read_bytes()
            for key, value in vars(module).copy().items():
                if key.endswith('_BYTES_AT_IMPORT') and isinstance(value, bytes):
                    INPUT_AT_IMPORT[module.__name__ + ':' + key] = value
SOURCE_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()
for name, expected in {
    'covariant_collective_cut2_neutral.py': '19d9e6ab7ef7187c6ac628783a8d42946f6965870c29182a6466d18caac9f1f0',
    'covariant_native_collective_primitive_electric.py': '085c316c7ae4cf64d4cb9e09ee8429294db191602f26addacaabd0626c5681ea',
    'covariant_shared_electric_left_svd.py': '3b203cb6c63134dca36debd708dfc1cc1a264f11e786f5f2803fad9548f38ae7',
}.items():
    if hashlib.sha256(SOURCE_AT_IMPORT[name]).hexdigest() != expected:
        raise RuntimeError('Frozen supplier changed: ' + name)

def descriptor_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def defect(matrix):
    return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.

def load_spectrum(capsule_path: Path):
    cap = json.loads(capsule_path.read_bytes())
    if cap['schema'] != 'neutral-cut3-copy-C-v1' or descriptor_hash(cap['descriptor']) != cap['descriptor_sha256']:
        raise RuntimeError('Invalid cut3 descriptor')
    for name, sha in cap['source_hashes_at_import'].items():
        if frozen.digest(ROOT / name) != sha:
            raise RuntimeError('Physical source changed: ' + name)
    for name, sha in cap['input_hashes_at_import'].items():
        historical = cap['input_bytes_at_import_utf8'][name].encode('utf-8')
        if hashlib.sha256(historical).hexdigest() != sha:
            raise RuntimeError('Imported provenance changed: ' + name)
        if name == 'plan' or name.endswith(':PLAN_BYTES_AT_IMPORT'):
            continue
        if name not in INPUT_AT_IMPORT or hashlib.sha256(INPUT_AT_IMPORT[name]).hexdigest() != sha:
            raise RuntimeError('Physical input changed: ' + name)
    physical = base.space(0, 3)
    blocks = []
    if {(r['sign'], r['charge_C_sign']) for r in cap['blocks']} != {(s,c) for s in (1,-1) for c in (1,-1)} or len(cap['blocks']) != 4:
        raise RuntimeError('Incomplete four blocks')
    for row in cap['blocks']:
        sector = base.copy_sector(physical, row['sign'])
        P, terms, perm, signs = frozen.charge_basis(physical, sector, row['charge_C_sign'])
        basis = {'copy_representative_rows': sector['representative_rows'].tolist(), 'C_permutation': perm.tolist(), 'C_integer_signs': signs.tolist(), 'C_contributors': terms}
        if descriptor_hash(basis) != row['basis_sha256'] or row['shape'] != [3*len(terms)]*2:
            raise RuntimeError('Basis reconstruction changed')
        total = sector['embedding'] @ block_diag((P,P,P), format='csr')
        path = Path(row['U_path'])
        if frozen.digest(path) != row['generated_U_sha256'] or path.stat().st_mode & 0o222:
            raise RuntimeError('Generated U is changed or writable')
        U = np.memmap(path, mode='r', dtype=np.complex128, shape=tuple(row['shape']), order='F')
        q = np.array(row['original_sigma_per_column'])
        if len(q) != U.shape[1] or np.any(q <= 0) or np.any(q[1:] > q[:-1]):
            raise RuntimeError('Invalid original singular values')
        blocks.append(dict(V=U,q=q,P=total,sign=row['sign'],charge_C_sign=row['charge_C_sign']))
    spectrum = base.Spectrum(blocks,3*len(physical['physical']))
    if len(spectrum.q) != spectrum.dimension or len(spectrum.q) != cap['total_ranks']:
        raise RuntimeError('Incomplete physical spectrum')
    return spectrum, cap['controls']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with threadpool_limits(limits=4):
        gamma, embedding, nodes, heat, Hphi, metric = base.left.frame()
        physical = base.space(0,3)
        dimension = 3*len(physical['physical'])
        resolution = None
        blocks = []
        proofs = []
        for sign in (1,-1):
            sector = base.copy_sector(physical, sign)
            copy_resolution = None
            for chi in (1,-1):
                P, terms, perm, signs = frozen.charge_basis(physical,sector,chi)
                P = cast(Any, P)
                proof = frozen.physical_intertwining(physical,sector,perm,signs)
                proof.update(sign=sign, charge_C_sign=chi, isometry_defect=defect(P.T@P-eye(P.shape[1],format='csr')))
                copy_piece = P@P.T
                copy_resolution = copy_piece if copy_resolution is None else copy_resolution+copy_piece
                total = sector['embedding']@block_diag((P,P,P),format='csr')
                piece = total@total.T
                resolution = piece if resolution is None else resolution+piece
                path = (args.scratch / f'neutral_cut3_S{sign}_C{chi}_leftU.complex128').resolve()
                if path.exists():
                    raise RuntimeError('Refusing to overwrite generated output: '+str(path))
                F, lower = frozen.stream(gamma,nodes,heat,physical,sector,terms,path)
                q, abi = base.left.lapack_left(F)
                F.flush()
                sha = frozen.digest(path)
                base.left.evict(F)
                del F
                path.chmod(0o444)
                basis = {'copy_representative_rows':sector['representative_rows'].tolist(),'C_permutation':perm.tolist(),'C_integer_signs':signs.tolist(),'C_contributors':terms}
                blocks.append(dict(sign=sign,charge_C_sign=chi,U_path=str(path),shape=[len(q)]*2,generated_U_sha256=sha,original_sigma_per_column=q.tolist(),pointcanonical_lower_bounds=lower,ABI=abi,physical_proof=proof,basis_description=basis,basis_sha256=descriptor_hash(basis)))
                (args.scratch / f'neutral_cut3_S{sign}_C{chi}_factor_metadata.json').write_text(
                    json.dumps(dict(status='COMPLETED_FACTOR_NOT_COMPLETE_SPECTRUM',
                        source_sha256=hashlib.sha256(SOURCE_AT_IMPORT[Path(__file__).name]).hexdigest(),
                        factor=blocks[-1]),indent=2))
                proofs.append(proof)
                print(json.dumps(dict(sign=sign,charge_C_sign=chi,rank=len(q),sigma_min=float(q[-1]),sigma_max=float(q[0]))),flush=True)
            d = defect(copy_resolution-eye(copy_resolution.shape[0],format='csr'))
            if d > 1e-13:
                raise ArithmeticError('Charge-C resolution failed')
            proofs.append(dict(sign=sign,C_resolution_defect=d))
        d = defect(resolution-eye(dimension,format='csr'))
        if d > 1e-13 or sum(len(b['original_sigma_per_column']) for b in blocks) != dimension:
            raise ArithmeticError('Full canonical resolution failed')
        descriptor = dict(gamma=dict(real=np.real(gamma).tolist(),imag=np.imag(gamma).tolist()),heat=heat.tolist(),metric=metric,Phi_nodes=nodes.tolist(),charge=0,ALL4edge_cut=3,UE=1,grid=32,active_kappa=.09,auxiliary_kappa_each=.04,lightmatchedtreehomotopy=0,auxiliary_copies=2,auxiliary_occupations_each=2)
        cap = dict(schema='neutral-cut3-copy-C-v1',import_epoch_ns=IMPORT_EPOCH_NS,source_hashes_at_import={k:hashlib.sha256(v).hexdigest() for k,v in SOURCE_AT_IMPORT.items()},input_hashes_at_import={k:hashlib.sha256(v).hexdigest() for k,v in INPUT_AT_IMPORT.items()},input_bytes_at_import_utf8={k:v.decode('utf-8') for k,v in INPUT_AT_IMPORT.items()},descriptor=descriptor,descriptor_sha256=descriptor_hash(descriptor),blocks=blocks,total_ranks=dimension,physical_rows=len(physical['physical']),controls=dict(actual_cut3_basis=proofs,fullcanonical_P_resolution_defect=d,previous_controls_provenance=dict(cut2_capsule_sha256='5393d53572aea55179dcc4bd2f5a32cddb4544b7f94343c4ebd1fb851b5f81d0',cut0_rerun=False)),versions=dict(numpy=np.__version__,scipy=scipy.__version__))
        cap = cast(dict[str, Any], cap)
        args.output.write_text(json.dumps(cap,indent=2))
        spectrum, _ = load_spectrum(args.output)
        rng = np.random.default_rng(2223)
        wave = rng.normal(size=dimension)+1j*rng.normal(size=dimension)
        wave /= np.linalg.norm(wave)
        projected = spectrum.project(wave)
        rebuilt = spectrum.wave(projected)
        coefficients = rng.normal(size=dimension)+1j*rng.normal(size=dimension)
        coefficients /= np.linalg.norm(coefficients)
        returned = spectrum.project(spectrum.wave(coefficients))
        roundtrips = dict(physical_wave_relative_defect=float(np.linalg.norm(rebuilt-wave)),global_coefficients_relative_defect=float(np.linalg.norm(returned-coefficients)),global_sigma_max=float(spectrum.q[0]),global_sigma_min=float(spectrum.q[-1]),globally_sorted=bool(np.all(spectrum.q[1:]<=spectrum.q[:-1])))
        if max(roundtrips['physical_wave_relative_defect'],roundtrips['global_coefficients_relative_defect']) > 1e-11:
            raise ArithmeticError('Actual consumer waveform roundtrip failed')
        cap['controls']['actual_cut3_consumer_roundtrips'] = roundtrips
        cap['runtime_seconds'] = time.perf_counter()-start
        args.output.write_text(json.dumps(cap,indent=2))
        print(json.dumps(dict(actual_loader_ranks=len(spectrum.q),natural_completion=True,runtime_seconds=cap['runtime_seconds'],roundtrips=roundtrips)),flush=True)

if __name__ == '__main__':
    main()

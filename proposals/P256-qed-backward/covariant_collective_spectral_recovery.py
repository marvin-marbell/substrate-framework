#!/usr/bin/env python3
"""Recover original ALL-sigma labels on frozen leftU by unsquared exact Fdag action."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any
import numpy as np
import scipy
import threadpoolctl
from threadpoolctl import threadpool_limits
import covariant_native_collective_primitive_electric as base

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / 'covariant_native_collective_primitive_electric_cut1_UE1_capsule_20261008.json'
EXPECTED_SOURCE = '085c316c7ae4cf64d4cb9e09ee8429294db191602f26addacaabd0626c5681ea'
EXPECTED_CAPSULE = '6000a6b4a6d9e6e91c9b1f0c607a3b4708c41c878a7aa24cf08ff17182412b05'
EXPECTED_DESCRIPTOR = '763dd7e68928c2ea03c02bb6e18f619c75ee708041e556eb7daa2f8c4182c42d'
SOURCE_BYTES_AT_IMPORT: dict[str, bytes] = {}
for module in tuple(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if isinstance(filename, str):
        path = Path(filename)
        if path.suffix == '.py' and path.resolve().parent == ROOT:
            SOURCE_BYTES_AT_IMPORT[path.name] = path.read_bytes()
SOURCE_BYTES_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()

def capture(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if digest(raw) != EXPECTED_CAPSULE:
        raise ValueError('Not the exact frozen6000a6 source capsule')
    cap = json.loads(raw)
    descriptor = cap['input_descriptor']
    if digest(json.dumps(descriptor, sort_keys=True).encode()) != EXPECTED_DESCRIPTOR or cap['input_descriptor_sha256'] != EXPECTED_DESCRIPTOR:
        raise ValueError('Frozen input descriptor mismatch')
    if cap['source_at_import']['covariant_native_collective_primitive_electric.py'] != EXPECTED_SOURCE:
        raise ValueError('Frozen085 source identity mismatch')
    for name, sha in cap['source_at_import'].items():
        if name not in SOURCE_BYTES_AT_IMPORT or digest(SOURCE_BYTES_AT_IMPORT[name]) != sha or digest(cap['source_bytes_at_import'][name].encode()) != sha:
            raise ValueError(f'Actual imported source is not the frozen source: {name}')
    expected = dict(primitive_electric_input=1., NEW_declared_electric_cutoff=1, quadrature_grid=32,
                    active_kappa=.09, auxiliary_kappa_each=.04, light_homotopy=0.,
                    quantum_auxiliary_copies=2, ALL_auxiliary_joint_CAR_states=36,
                    BOTH_copy_exchange_sectors_retained=True, ONE_total_Gauss_after_ALL36=True)
    if any(descriptor[k] != v for k, v in expected.items()):
        raise ValueError('Declared physical input mismatch')
    basis = {}
    for label, dims in (('neutral', (8058, 5952)), ('charged', (5922, 4374))):
        for diag, dimension, sign in zip(cap['actual_ALL_sector_factor_diagnostics'][label], dims, (1, -1)):
            if diag['copy_exchange_sign'] != sign or diag['ALL_sector_columns'] != dimension:
                raise ValueError('Frozen complete copy block dimensions mismatch')
            binary = Path(diag['disk_path'])
            if binary.stat().st_size != 16 * dimension ** 2:
                raise ValueError('Actual leftU binary size mismatch')
            basis[str(binary)] = dict(sha256=file_digest(binary), bytes=binary.stat().st_size, dimension=dimension, label=label, sign=sign)
    return dict(path=str(path.resolve()), bytes=raw, capsule=cap, basis=basis)

# Source bytes, exact default capsule bytes and ALL four binary hashes precede calculation.
DEFAULT_CAPTURE_AT_IMPORT = capture(DEFAULT_INPUT)

class Adjoint:
    """Exact grid32 one-Gauss SF adjoint; heat.T first, electric last.

    The validated matched-tree phase shifts Fourier indices from n to n-n3.
    All light and ALL36 auxiliary canonical columns participate before selection.
    """
    def __init__(self, physical: dict[str, Any], sector: dict[str, Any], gamma: np.ndarray,
                 nodes: np.ndarray, heat: np.ndarray):
        self.nr = len(sector['flux'])
        self.light_ids, self.pair_ids = sector['light_ids'], sector['pair_ids']
        occupied = np.array(physical['light'])
        n3 = np.sum(occupied >= 6, axis=1)
        self.frequency = (sector['flux'] - n3[self.light_ids]) % 32
        self.nl, self.na = len(occupied), len(sector['pairs'])
        self.heat = heat
        self.electric = np.exp(-np.sum(sector['electric'] ** 2, axis=1) / 2)
        self.means, self.auxiliaries = [], []
        aux = np.array(physical['auxiliary'])
        paux = sector['auxiliary_map']
        for normal in nodes:
            actual = gamma.copy(); actual[1] *= 1 + float(normal)
            half, pref = base.collective.matched_tree_half(actual, 0., 0.)
            self.means.append(base.canonical(half, pref, occupied).conj().T)
            samples = []
            for theta in 2 * math.pi * np.arange(32) / 32:
                half, pref = base.collective.auxiliary_path_half(actual, .04, complex((1 + np.exp(1j * theta)) / 2))
                A = base.canonical(half, pref, aux)
                samples.append((paux.T @ np.kron(A, A) @ paux).conj().T)
            self.auxiliaries.append(np.array(samples))

    def __call__(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim != 2 or vectors.shape[0] != 3 * self.nr:
            raise ValueError('Fdag expects complete sector columns in Phi-major order')
        batch = vectors.shape[1]
        phi = np.einsum('op,orb->prb', self.heat.conj(), vectors.reshape(3, self.nr, batch))
        result = np.empty_like(vectors)
        for p in range(3):
            frequency = np.zeros((32, self.nl, self.na, batch), dtype=complex)
            frequency[self.frequency, self.light_ids, self.pair_ids, :] = phi[p]
            angles = np.fft.ifft(frequency, axis=0) * 32
            light = np.matmul(self.means[p], angles.reshape(32, self.nl, self.na * batch)).reshape(32, self.nl, self.na, batch)
            acted = np.matmul(light.transpose(0, 1, 3, 2), self.auxiliaries[p].transpose(0, 2, 1)[:, None, :, :]).transpose(0, 1, 3, 2)
            projected = np.fft.fft(acted, axis=0) / 32
            result[p * self.nr:(p + 1) * self.nr] = projected[self.frequency, self.light_ids, self.pair_ids, :] * self.electric[:, None]
        return result

def smallcut_controls(gamma: np.ndarray, nodes: np.ndarray, heat: np.ndarray) -> list[dict[str, Any]]:
    controls = []
    with tempfile.TemporaryDirectory(prefix='collective-adjoint-cut0-') as directory:
        for charge in (0, 1):
            physical = base.space(charge, 0)
            for sign in (1, -1):
                sector = base.copy_sector(physical, sign)
                factor, haar, _ = base.streamed_factor(gamma, nodes, heat, physical, sector, 32, 1., Path(directory) / f'{charge}_{sign}.bin')
                d = factor.shape[0]
                i, j = np.arange(d)[:, None] + 1, np.arange(7)[None, :] + 1
                vectors = np.sin(i * j * .37) + 1j * np.cos(i * (j + .13) * .29)
                direct = factor.conj().T @ vectors
                observed = Adjoint(physical, sector, gamma, nodes, heat)(vectors)
                relative = float(np.linalg.norm(observed - direct) / np.linalg.norm(direct))
                if not np.isfinite(relative) or relative > 256 * d * np.finfo(float).eps:
                    raise ArithmeticError('Actual complete cut0 direct Fdag validation failed')
                controls.append(dict(charge=charge, sign=sign, complete_factor_dimension=d,
                                     deterministic_fullsupport_vectors=7, direct_Fdag_relative_error=relative,
                                     direct_Fdag_max_absolute_error=float(abs(observed - direct).max())))
                del factor, haar
    return controls

def spectra_from_labels(captured: dict[str, Any], labels: dict[str, Any]) -> tuple[base.Spectrum, base.Spectrum]:
    spectra = []
    cap = captured['capsule']
    for charge, name in ((0, 'neutral'), (1, 'charged')):
        physical = base.space(charge, 1)
        blocks = []
        for diag, record, sign in zip(cap['actual_ALL_sector_factor_diagnostics'][name], labels[name], (1, -1)):
            sector = base.copy_sector(physical, sign)
            d = 3 * len(sector['flux'])
            if record['sign'] != sign or record['binary_path'] != diag['disk_path'] or record['binary_sha256_at_import'] != captured['basis'][diag['disk_path']]['sha256']:
                raise ValueError('Recovered label binary identity mismatch')
            q = np.array(record['original_q_per_column'], dtype=float)
            if len(q) != d or np.any(q <= 0):
                raise ValueError('Recovered labels omit original directions')
            V = np.memmap(diag['disk_path'], mode='r', dtype=np.complex128, shape=(d, d), order='F')
            blocks.append(dict(V=V, q=q, P=sector['embedding'], sign=sign))
        spectrum = base.Spectrum(blocks, 3 * len(physical['physical']))
        original = np.array(cap['actual_primitive_electric_ALL36_transfer'][f'actual_ALL_{name}_singular_values'])
        if not np.array_equal(spectrum.q, original):
            raise ValueError('Original globally sorted ALL singular values not retained exactly once')
        spectra.append(spectrum)
    return spectra[0], spectra[1]

def recover(capsule_path: Path) -> tuple[base.Spectrum, base.Spectrum, dict[str, Any]]:
    captured = DEFAULT_CAPTURE_AT_IMPORT if capsule_path.resolve() == DEFAULT_INPUT.resolve() else capture(capsule_path)
    gamma, _, nodes, heat, _, _ = base.left.frame()
    identities = base.validate_copy_identity(gamma, nodes)
    controls: dict[str, Any] = dict(actual_cut0_direct_Fdag=smallcut_controls(gamma, nodes, heat),
                                   actual_nonzero_ALLcolumn_phase_copy_identities=identities, labels={}, calibration={})
    cap = captured['capsule']
    for charge, name in ((0, 'neutral'), (1, 'charged')):
        physical = base.space(charge, 1)
        records, norms = [], []
        for diag, sign in zip(cap['actual_ALL_sector_factor_diagnostics'][name], (1, -1)):
            sector = base.copy_sector(physical, sign)
            d = 3 * len(sector['flux'])
            V = np.memmap(diag['disk_path'], mode='r', dtype=np.complex128, shape=(d, d), order='F')
            action = Adjoint(physical, sector, gamma, nodes, heat)
            recovered = np.empty(d)
            for start in range(0, d, 32):
                end = min(d, start + 32)
                applied = action(V[:, start:end])
                recovered[start:end] = np.linalg.norm(applied, axis=0)
            if np.any(recovered <= 0) or not np.all(np.isfinite(recovered)):
                raise ArithmeticError('Nonpositive actual FdagU norm; no clipping or dropped columns')
            norms.append(recovered)
            records.append(dict(sign=sign, binary_path=diag['disk_path'], binary_sha256_at_import=captured['basis'][diag['disk_path']]['sha256'], recovered_FdagU_norm_per_column=recovered.tolist()))
            print(json.dumps(dict(stage='complete_FdagU_calibration', label=name, sign=sign, ALL_columns=d)), flush=True)
        combined = np.concatenate(norms)
        order = np.argsort(-combined, kind='stable')
        original = np.array(cap['actual_primitive_electric_ALL36_transfer'][f'actual_ALL_{name}_singular_values'])
        assigned = np.empty_like(original); assigned[order] = original
        errors = combined[order] - original
        relative = errors / original
        envelope = float(np.max(abs(errors)))
        gaps = original[:-1] - original[1:]
        ambiguous = gaps <= 2 * envelope
        # This is an observed global discrepancy envelope, NOT a certified forward-error bound.
        starts = np.r_[0, np.flatnonzero(~ambiguous) + 1]
        ends = np.r_[starts[1:], len(original)]
        groups = [dict(first_rank=int(a), last_rank=int(b - 1), count=int(b - a), original_q_span=float(original[a] - original[b - 1])) for a, b in zip(starts, ends) if b - a > 1]
        offset = 0
        for record, recovered in zip(records, norms):
            record['original_q_per_column'] = assigned[offset:offset + len(recovered)].tolist()
            record['global_original_rank_per_column'] = np.argsort(order)[offset:offset + len(recovered)].tolist()
            offset += len(recovered)
        controls['labels'][name] = records
        controls['calibration'][name] = dict(ALL_original_singular_values_count=len(original), ALL_sorted_recovered_norms=combined[order].tolist(),
            ALL_sorted_norm_minus_original_q=errors.tolist(), ALL_sorted_relative_errors=relative.tolist(),
            max_absolute_error=envelope, max_relative_error=float(np.max(abs(relative))), rms_relative_error=float(np.sqrt(np.mean(relative ** 2))),
            min_original_q=float(original[-1]), original_q_scale=float(original[0]), max_scale_relative_error=envelope / float(original[0]),
            ambiguous_adjacent_gaps_count=int(np.sum(ambiguous)), empirical_ambiguity_groups=groups,
            ambiguity_interpretation='Ranks matched by actual FdagU norms. Neighbor gaps <= twice observed ALLsorted maximum absolute discrepancy are numerically unresolved under this empirical envelope; not a rigorous bound. Equal/near-equal directions may permute. ALLoriginal q retained unchanged; no clipping/pruning/support inference.')
    controls.update(provenance=dict(source_capsule_path=captured['path'], source_capsule_sha256_at_import=digest(captured['bytes']),
        source_capsule_bytes_at_import=captured['bytes'].decode(), input_descriptor=cap['input_descriptor'], input_descriptor_sha256=EXPECTED_DESCRIPTOR,
        basis_files_at_import=captured['basis'], source_at_import={name:digest(raw) for name,raw in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name:raw.decode() for name,raw in SOURCE_BYTES_AT_IMPORT.items()},
        runtime_packages=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, threadpoolctl=threadpoolctl.__version__, BLAS=threadpoolctl.threadpool_info(), platform=platform.platform())),
        method='Exact unsquared grid32 FdagU norms, stable ALLblock rank match to frozen ALLoriginal sigma. No factor SVD, FFdag, combinedU or Gram.',
        scope='Finite declared UE1 cut1 ALL36/Phi3 quantum input; no induced-electric/massless/commoncone/fullPhi/SU4 promotion')
    n, c = spectra_from_labels(captured, controls['labels'])
    return n, c, controls

def load_recovered(capsule_path: Path, recovery_cap_path: Path) -> tuple[base.Spectrum, base.Spectrum, dict[str, Any]]:
    captured = DEFAULT_CAPTURE_AT_IMPORT if capsule_path.resolve() == DEFAULT_INPUT.resolve() else capture(capsule_path)
    raw = recovery_cap_path.read_bytes()
    controls = json.loads(raw)
    provenance = controls['provenance']
    if provenance['source_capsule_sha256_at_import'] != EXPECTED_CAPSULE or digest(provenance['source_capsule_bytes_at_import'].encode()) != EXPECTED_CAPSULE or provenance['input_descriptor_sha256'] != EXPECTED_DESCRIPTOR:
        raise ValueError('Recovery capsule source/input identity mismatch')
    if provenance['basis_files_at_import'] != captured['basis']:
        raise ValueError('Recovery capsule does not match ALL actual captured leftU binaries')
    own = Path(__file__).name
    if provenance['source_at_import'][own] != digest(SOURCE_BYTES_AT_IMPORT[own]):
        raise ValueError('Recovery implementation changed after label creation')
    for name, sha in provenance['source_at_import'].items():
        if digest(provenance['source_bytes_at_import'][name].encode()) != sha or name not in SOURCE_BYTES_AT_IMPORT or digest(SOURCE_BYTES_AT_IMPORT[name]) != sha:
            raise ValueError(f'Recovery imported source mismatch: {name}')
    n, c = spectra_from_labels(captured, controls['labels'])
    controls['loaded_recovery_capsule_sha256'] = digest(raw)
    return n, c, controls

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    with threadpool_limits(limits=4):
        n, c, controls = recover(args.input)
    controls['runtime_seconds'] = time.monotonic() - start
    controls['actual_complete_spectrum_ranks'] = [len(n.q), len(c.q)]
    args.output.write_text(json.dumps(controls, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), ranks=controls['actual_complete_spectrum_ranks'],
        runtime_seconds=controls['runtime_seconds'], maximum_relative_errors={name:row['max_relative_error'] for name,row in controls['calibration'].items()})), flush=True)

if __name__ == '__main__':
    main()

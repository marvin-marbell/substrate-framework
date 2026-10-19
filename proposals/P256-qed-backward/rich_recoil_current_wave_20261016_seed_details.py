"""Retain full initial current-sector vectors/complex metrics from own rays."""
from __future__ import annotations
import json
from pathlib import Path
import pickle
import sys
from typing import Any
import numpy as np
from scipy.linalg import qr
from tenpy.networks.mpo import MPOEnvironment
from rich_recoil_current_wave_20261016_model import exact_image
from rich_recoil_current_wave_20261016_observe import cores

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_recoil_current_wave_20261016_'


def load(path: Path) -> Any:
    with path.open('rb') as stream:
        return pickle.load(stream)


def direct_superposition_norm(vectors: list[Any], coefficients: np.ndarray) -> float:
    # Contract a literal sum-MPS by QR, without subtracting nearly equal Gram
    # entries and without dividing by any small/zero Schmidt value.
    arrays = [cores(v) for v in vectors]
    previous = None
    for site in range(len(arrays[0])):
        blocks = [a[site] for a in arrays]
        if site == 0:
            tensor = np.concatenate([c*v.norm*b for c, v, b in zip(coefficients, vectors, blocks)], axis=2)
        elif site == len(arrays[0])-1:
            tensor = np.concatenate(blocks, axis=0)
        else:
            tensor = np.zeros((sum(b.shape[0] for b in blocks), blocks[0].shape[1], sum(b.shape[2] for b in blocks)), complex)
            left = right = 0
            for b in blocks:
                tensor[left:left+b.shape[0], :, right:right+b.shape[2]] = b
                left += b.shape[0]
                right += b.shape[2]
        if previous is not None:
            tensor = np.einsum('ab,bpc->apc', previous, tensor, optimize=True)
        decomposition: Any = qr(tensor.reshape(-1, tensor.shape[-1]), mode='economic')
        previous = decomposition[1]
    assert previous is not None
    return float(np.linalg.norm(previous))


def main() -> None:
    stem = ROOT/(PREFIX+sys.argv[1]+'_')
    completed = json.loads(Path(str(stem)+'result.json').read_text())
    law = load(Path(str(stem)+'constructed_model.pkl'))
    records = {}
    for case in completed['cases']:
        wave = load(Path(str(stem)+case+'_initial.pkl'))
        recorded_seed = json.loads(Path(str(stem)+case+'_seed.json').read_text())['seed']
        for r, psi in wave.items():
            vectors = [psi.copy(), exact_image(law.model, psi)]+[exact_image(op['QJ'], psi) for op in law.emission]
            gram = np.array([[a.overlap(b) for b in vectors] for a in vectors])
            h = np.array([[a.norm.conjugate()*b.norm*MPOEnvironment(a, law.model.H_MPO, b).full_contraction(0) for b in vectors] for a in vectors])
            eigenvalues, eigenvectors = np.linalg.eigh(gram)
            threshold = recorded_seed[str(r)]['machine_roundoff_Gram_threshold']
            nulls = []
            for index in np.flatnonzero(eigenvalues <= threshold):
                nulls.append(dict(Gram_eigenvalue=float(eigenvalues[index]),
                    direct_QR_physical_norm=direct_superposition_norm(vectors, eigenvectors[:, index]),
                    scope='Independent literal state-amplitude cancellation norm, not a scalar Gram eigenvalue relabelled a true null.'))
            key = f'{case}_r{r}'
            np.savez_compressed(Path(str(stem)+key+'_current_metric.npz'), Gram=gram, owning_H=h,
                eigenvalues=eigenvalues, eigenvectors=eigenvectors)
            with Path(str(stem)+key+'_raw_current_vectors.pkl').open('wb') as stream:
                pickle.dump(vectors, stream, protocol=5)
            records[key] = dict(raw_Gram_eigenvalues=eigenvalues.tolist(), physical_near_nulls=nulls,
                H_antihermitian_roundoff=float(np.linalg.norm(h-h.conj().T)),
                maximum_raw_image_bond=max(max(v.chi) for v in vectors),
                scope='Own captured clock0 psi,Hpsi,four before-cut QJpsi; complex raw Gram, owning H and raw states saved without old-source projection.')
            print(json.dumps(dict(case=key, **records[key])), flush=True)
    Path(str(stem)+'seed_details_result.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()

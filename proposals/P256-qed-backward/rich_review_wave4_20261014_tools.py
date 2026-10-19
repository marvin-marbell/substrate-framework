"""Independent Gram normalization and complete spectral physical consumers."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave4_20261014_'

def gram_frame(sources: np.ndarray, relative_cut: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    gram = sources.conj().T @ sources
    weights, vectors = np.linalg.eigh(gram)
    keep = weights > relative_cut * max(float(weights[-1]), 1e-30)
    return sources @ (vectors[:, keep] / np.sqrt(weights[keep])), weights

def projected_probability(frame: np.ndarray, states: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(frame.conj().T @ states) ** 2, axis=0)

def complete_resolvent(h: np.ndarray, z: complex, source: np.ndarray) -> np.ndarray:
    return np.linalg.solve(z * np.eye(h.shape[0]) - h, source)

def complex_pair(value: complex) -> list[float]:
    return [float(np.real(value)), float(np.imag(value))]

def save(label: str, result: dict[str, Any]) -> None:
    result['runtime'] = {'python': sys.executable, 'numpy': np.__version__, 'scipy': scipy.__version__}
    (ROOT / f'{PREFIX}{label}_result.json').write_text(json.dumps(result, indent=2) + '\n')
    imported: list[str] = []
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.parent == ROOT and path.suffix == '.py':
                imported.append(path.name)
    (ROOT / f'{PREFIX}{label}_dependencies.json').write_text(json.dumps(sorted(set(imported)), indent=2) + '\n')
    print(json.dumps(result))

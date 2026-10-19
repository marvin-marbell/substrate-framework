#!/usr/bin/env python3
"""All55 original EM point forms for the frozen six-mode quantum trial span."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, cast
import zipfile
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
import numpy as np
from original_charged_principal import portable
from prepared_neutral_shared_photon_state import load_shared_native_cache
from prepared_neutral_changing_paired_pulse import source_point
from prepared_neutral_changing_em_characteristic import CACHE_SHA, MANIFEST_SHA


def multipliers(model: Any, sample: dict[str, Any], direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen interface values and analytic ALL55 EM multiplier jets."""
    q, em = sample['q'], sample['em']
    direction = np.asarray(direction, float)
    norm = np.linalg.norm(direction)
    if direction.shape != q.shape or norm == 0:
        raise ValueError('Nonzero full-normal trial direction required')
    z = float(q @ (direction/norm))/model.sigma
    n = len(q)
    values = np.array([1., np.sqrt(2)*z, (2*z*z-1)/np.sqrt(2),
                       (2*float(q@q)/model.sigma**2-n)/np.sqrt(2*n),
                       np.sqrt(2)*em[52], (2*float(em[:52]@em[:52])-52)/np.sqrt(104)])
    jets = np.zeros((55, 6))
    jets[52, 4] = np.sqrt(2)
    jets[:52, 5] = 4*em[:52]/np.sqrt(104)
    return values, jets


def state_vector(data: dict[str, Any], sector: str, profile: str) -> tuple[str, np.ndarray]:
    if sector == 'neutral':
        return sector, data['neutral_state']
    if sector in ('source19', 'SOFT'):
        profile = sector
    elif sector != 'charged':
        raise ValueError('Expected neutral, charged, source19 or SOFT')
    if profile not in ('source19', 'SOFT'):
        raise ValueError('Expected one original held SOURCE profile')
    return 'charged', data[profile+'_normalized_source_state']


def field_jets(model: Any, sample: dict[str, Any], h: float) -> list[dict[str, np.ndarray]]:
    """Shared producer jets include original ALL55 Gaussian exactly once."""
    zero = np.zeros(model.chart.normal_dim)
    out = []
    for j in range(55):
        dem = np.zeros(55); dem[j] = 1.
        out.append(model.directional_basis(sample, zero, dem, step=h))
    return out


def operator_matrices(model: Any, data: dict[str, Any], sample: dict[str, Any],
                      direction: np.ndarray, sector: str, physical_point: dict[str, Any] | None = None,
                      q_shift: np.ndarray | None = None, h: float | None = None,
                      profile: str = 'source19', derivative_steps: tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]] | None = None) -> dict[str, Any]:
    """Unweighted point integrands; multiply cutoffFoot**2 in GLOBAL quadrature.

    Physical Phi translation is EM-independent, so its Haar factor cancels
    against the physical measure. No physical SOURCE is recaptured here.
    No H_D body unitary is applied: parent owns the entire Hamiltonian.
    """
    if np.asarray(sample['em']).shape != (55,):
        raise ValueError('Original ALL55 EM coordinates required')
    if q_shift is not None and np.asarray(q_shift).shape != sample['q'].shape:
        raise ValueError('Full-normal translation required')
    if physical_point is not None and q_shift is not None:
        # Physical point is deliberately not used to rebuild the held bank.
        pass
    step = 2.**-18 if h is None else float(h)
    if step <= 0:
        raise ValueError('Positive FD step required')
    fibre_sector, coefficients = state_vector(data, sector, profile)
    v = sample[fibre_sector] @ coefficients
    modes, dmodes = multipliers(model, sample, direction)
    waves = v[:, None]*modes[None, :]
    gram = waves.conj().T @ waves
    weights = np.r_[np.asarray(model.em['omega'])/2, np.full(3, model.ue/model.law.size)]
    jets = derivative_steps if derivative_steps is not None else (field_jets(model, sample, step), field_jets(model, sample, step/2))
    forms = []; derivatives = []
    for level in jets:
        dv = np.asarray([j[fibre_sector] @ coefficients for j in level])
        dw = dv[:, :, None]*modes[None, None, :] + v[None, :, None]*dmodes[:, None, :]
        forms.append(np.einsum('j,jai,jak->ik', weights, dw.conj(), dw))
        derivatives.append(dw)
    potential_value = float(np.sum(np.asarray(model.em['omega'])*sample['em'][:52]**2/2))
    potential = potential_value*gram
    difference = float(np.linalg.norm(forms[0]-forms[1]))
    return dict(kinetic=forms[1], potential=potential, Gram=gram, gram=gram,
                total=forms[1]+potential, kinetic_coarse=forms[0],
                observed_FD_errors=dict(steps=[step, step/2], kinetic_absolute=difference,
                    kinetic_relative=difference/max(float(np.linalg.norm(forms[1])), np.finfo(float).tiny),
                    derivative_absolute=float(np.linalg.norm(derivatives[0]-derivatives[1]))),
                cutoff_squared=float(sample['cutoff']**2), quadrature_weight_applied=False,
                all55_weights=weights, derivative_fine=derivatives[1],
                convention='Gaussian/Haar factored; GLOBAL quadrature multiplies original cutoffFoot squared ONCE; no point normalization',
                zero_point='Already retained once by literal kinetic+potential; no additive or subtractive correction',
                physical_translation_EM_independent=True, held_SOURCE_not_recaptured=True,
                no_preapplied_HD_body_phase=True)


def native(model: Any, data: dict[str, Any], h: float) -> dict[str, Any]:
    q, em, generators = source_point(model, data)
    sample = model.basis_at(q, em)
    direction = generators['Y0']
    jets = (field_jets(model, sample, h), field_jets(model, sample, h/2))
    results = {s: operator_matrices(model, data, sample, direction, s, h=h, derivative_steps=jets)
               for s in ('neutral', 'source19', 'SOFT')}
    literal = {s: [] for s in results}
    harmonic_source = {s: [] for s in ('source19', 'SOFT')}
    # Independent literal packet variation includes ALL55 exp(-em^2/2),
    # multiplier variation and the same original bank at fixed footPoint.
    for step in (h, h/2):
        errors = {s: [] for s in results}
        for j in range(55):
            shift = np.zeros(55); shift[j] = step
            plus = model.basis_at(q, em+shift, point=sample['point'])
            minus = model.basis_at(q, em-shift, point=sample['point'])
            mp, _ = multipliers(model, plus, direction)
            mm, _ = multipliers(model, minus, direction)
            gp = np.exp(-float((em+shift)@(em+shift)-em@em)/2)
            gm = np.exp(-float((em-shift)@(em-shift)-em@em)/2)
            for s in results:
                fs, coeff = state_vector(data, s, 'source19')
                finite = (gp*(plus[fs]@coeff)[:, None]*mp-gm*(minus[fs]@coeff)[:, None]*mm)/(2*step)
                errors[s].append(float(np.linalg.norm(finite-results[s]['derivative_fine'][j])))
                if step == h/2 and j >= 52 and s != 'neutral':
                    harmonic_source[s].append(float(np.linalg.norm((plus['held_source_base']-minus['held_source_base'])/(2*step))))
        for s in results:
            literal[s].append(dict(step=step, all55_errors=errors[s], maximum=max(errors[s])))
    for s, row in results.items():
        del row['derivative_fine']
        row['Hermitian_residual'] = float(np.linalg.norm(row['total']-row['total'].conj().T))
        row['offdiagonal_norm'] = float(np.linalg.norm(row['total']-np.diag(np.diag(row['total']))))
        assert row['offdiagonal_norm'] > 0 and row['Hermitian_residual'] < 1e-10
    assert all(max(v) > 0 for v in harmonic_source.values())
    return dict(scope='Original fixed-state ALL55 point component, not global L2 or evolution',
                q=q, em=em, matrices=results, literal_full55_packet_control=literal,
                actual_held_SOURCE_harmonic_derivative_norms=harmonic_source,
                original_state_unchanged=True, initial_coefficients=[1,0,0,0,0,0],
                debts=['Finite-difference error observed, not certified', 'Global quadrature and full quantum coefficient evolution belong to parent', 'Native harmonic packet is unwrapped; no legal-N global conclusion'])


def source_epoch() -> dict[str, bytes]:
    root = Path(__file__).resolve().parent
    return {Path(cast(str, m.__file__)).name: Path(cast(str, m.__file__)).read_bytes() for m in tuple(sys.modules.values())
            if getattr(m, '__file__', None) and Path(cast(str, m.__file__)).suffix == '.py' and Path(cast(str, m.__file__)).resolve().parent == root}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-npz', type=Path, required=True)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--fd', type=float, default=2.**-18)
    args = parser.parse_args()
    started = time.perf_counter()
    assert hashlib.sha256(args.cache_npz.read_bytes()).hexdigest() == CACHE_SHA
    assert hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest() == MANIFEST_SHA
    model, data = load_shared_native_cache(args.cache_npz, args.cache_manifest)
    before = source_epoch()
    science = native(model, data, args.fd)
    after = source_epoch()
    assert all(after.get(k) == v for k, v in before.items())
    with zipfile.ZipFile(args.source_archive, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, body in after.items(): archive.writestr('sources/'+name, body)
        for path in (args.cache_npz, args.cache_manifest): archive.writestr('inputs/'+path.name, path.read_bytes())
    args.source_archive.chmod(0o600)
    result = dict(science=science, argv=sys.argv, cwd=os.getcwd(), python=sys.executable,
                  natural_exit=0, wall_seconds=time.perf_counter()-started,
                  cache_npz_sha256=CACHE_SHA, cache_manifest_sha256=MANIFEST_SHA,
                  sources={k: hashlib.sha256(v).hexdigest() for k,v in after.items()},
                  stable_loaded_sources=True, source_archive_sha256=hashlib.sha256(args.source_archive.read_bytes()).hexdigest())
    text = json.dumps(portable(result), sort_keys=True)
    args.output.write_text(text+'\n'); args.output.chmod(0o600)
    print(text)

if __name__ == '__main__':
    main()

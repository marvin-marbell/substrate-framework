#!/usr/bin/env python3
"""Full original Phi point forms retaining all four SOURCE words per charge."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy.linalg import expm, lu_factor, lu_solve
from original_charged_principal import portable
import prepared_neutral_full_phi_commonform as fullPhi
from prepared_neutral_changing_phi_kinetic import full_cross_kernels, _wick_control
from prepared_neutral_changing_paired_pulse import source_point
from prepared_neutral_shared_photon_state import load_shared_native_cache, loaded_source_epoch


def coefficient_channels(data: dict[str, Any], sector: str,
                         profile: str = 'source19') -> tuple[np.ndarray, dict[str, Any]]:
    """Original coefficient ABI; no per-word norm, whitening or second global Z."""
    state = np.asarray(data['neutral_state'])
    labels = np.asarray(data['neutral_charge_labels'])
    if sector == 'neutral':
        return np.column_stack((state, labels*state)), dict(order=['original neutral', 'plus-minus neutral'], initial_exact=True)
    if sector != 'charged' or profile not in ('source19', 'SOFT'):
        raise ValueError('sector neutral/charged and original source19/SOFT profile required')
    raw = data[profile+'_raw_source_state']
    z = math.sqrt(float(np.vdot(raw, data['charged_Gram'] @ raw).real))
    first = data[profile+'_normalized_source_state']
    error = float(np.linalg.norm(raw/z-first))
    if error > 1e-12:
        raise ValueError('Original cached global SOURCE normalization mismatch')
    # Ordering comes from the frozen word maps and actual charge labels. The
    # coherent column replaces +word0 (its original beta0 is nonzero).
    words = data[profile+'_word_maps']
    pairs = [(charge, word) for charge in (1, -1) for word in range(len(words))]
    columns = [words[word] @ ((labels == charge)*state)/z for charge, word in pairs]
    matrix = np.column_stack([first]+columns[1:])
    reconstruction = sum(data[profile+'_map'] @ ((labels == charge)*state) for charge in (1, -1))/z
    return matrix, dict(order=['original normalized coherent SOURCE']+['%+d word%d' % pair for pair in pairs[1:]],
                        omitted_unit_word='+1 word0', global_SOURCE_normalizer=z,
                        cached_normalization_error=error, coherent_reconstruction_error=float(np.linalg.norm(reconstruction-first)),
                        original_all_word_directions=8, per_word_normalization=False, initial_exact=bool(np.array_equal(matrix[:, 0], first)))


def _direction_form_channels(model: Any, sample: dict[str, Any], coefficient: np.ndarray,
                             sector: str, wave: np.ndarray, values: np.ndarray,
                             jets: np.ndarray, probe: np.ndarray, step: float,
                             point: dict[str, Any], physical_q: np.ndarray,
                             translated: bool) -> dict[str, Any]:
    """Matrix lift of fullPhi's common positive form, using the SAME producers."""
    n = model.chart.normal_dim
    dq, gauge = probe[:n], probe[n:]
    derivative = model.directional_basis(sample, dq, step=step)[sector] @ coefficient
    correction = 0.
    if translated:
        foot = model.chart.directional_jet(sample['q'], dq, sample['point'])['log_density_first']
        physical = model.chart.directional_jet(physical_q, dq, point)['log_density_first']
        correction = .5*(foot-physical)
        derivative += correction*wave
    trial = (wave[:, :, None]*values[None, None, :]).reshape(len(wave), -1)
    dtrial = (derivative[:, :, None]*values[None, None, :] + wave[:, :, None]*(dq @ jets)[None, None, :]).reshape(len(wave), -1)
    operators = model.gauge_operators(gauge, sector)
    momentum = -1j*dtrial + operators['first'] @ trial
    outside = trial.conj().T @ operators['outside_second'] @ trial
    raw = momentum.conj().T @ momentum + outside
    weight = model.chi/2*sample['importance_weight']
    return dict(matrix=weight*(raw+raw.conj().T)/2, outside_matrix=weight*outside,
                density_correction=correction)


def operator_matrices_channels(model: Any, sample: dict[str, Any], direction: np.ndarray,
                               sector: str, coefficient_matrix: np.ndarray,
                               physical_point: dict[str, Any] | None = None,
                               q_shift: np.ndarray | None = None, h: float | None = None,
                               trace_samples: int = 4, seed: int = 222107,
                               return_probes: bool = False) -> dict[str, Any]:
    """Full 2835 cometric; CHANNEL-major six modes; cutoffFoot² INCLUDED."""
    coefficient = np.asarray(coefficient_matrix, complex)
    if sector not in ('neutral', 'charged') or coefficient.ndim != 2 or coefficient.shape[0] != sample[sector].shape[1]:
        raise ValueError('Original neutral/charged bank coefficient matrix required')
    if trace_samples < 2:
        raise ValueError('At least two probes required for observed spread')
    n = model.chart.normal_dim
    q = np.asarray(sample['q'], float)
    shift = np.zeros(n) if q_shift is None else np.asarray(q_shift, float)
    if shift.shape != (n,):
        raise ValueError('q_shift requires all normal coordinates')
    if physical_point is not None and q_shift is None and physical_point is not sample['point']:
        raise ValueError('Distinct physical_point requires q_shift for density jet')
    translated = bool(np.any(shift))
    physical_q = q+shift
    point = (model.chart.evaluate(physical_q) if translated else sample['point']) if physical_point is None else physical_point
    step = model.sigma/32 if h is None else float(h)
    if step <= 0 or not np.isfinite(step) or np.shape(direction) != (n,) or not np.linalg.norm(direction):
        raise ValueError('Positive finite FD step and nonzero all-normal direction required')
    wave = sample[sector] @ coefficient
    values, jets = fullPhi.trial_multipliers(q, sample['em'], direction, model.sigma)
    raw_probes = 2*np.random.default_rng(seed).integers(0, 2, size=(model.chart.raw_dim, trace_samples))-1
    probes = lu_solve(lu_factor(np.sqrt(model.chart.mass)[:, None]*point['jacobian']), raw_probes)
    rows, corrections, outside = [], [], []
    for finite_step in (step, step/2):
        matrices, weights, density = [], [], []
        for i in range(trace_samples):
            form = _direction_form_channels(model, sample, coefficient, sector, wave, values,
                                            jets, probes[:, i], finite_step, point, physical_q, translated)
            matrices.append(form['matrix']); weights.append(form['outside_matrix']); density.append(form['density_correction'])
        rows.append(np.stack(matrices)); outside.append(np.mean(weights, axis=0)); corrections.append(density)
    means = [row.mean(axis=0) for row in rows]
    standard_error = np.sqrt(np.sum(np.abs(rows[1]-means[1])**2, axis=0)/(trace_samples*(trace_samples-1)))
    result = dict(matrix=means[1], matrices_by_step=means, per_probe_matrices=rows,
                  outside_second_matrices_by_step=outside, fd_steps=[step, step/2],
                  fd_difference=means[1]-means[0], fd_difference_norm=float(np.linalg.norm(means[1]-means[0])),
                  standard_error_matrix=standard_error, standard_error_frobenius=float(np.linalg.norm(standard_error)),
                  multipliers=values, trace_samples=trace_samples, seed=seed, channels=coefficient.shape[1],
                  flattening='CHANNEL-major then six modes', q_shift=shift, density_corrections_by_step=corrections,
                  probe_norms=[dict(raw_squared=float(raw_probes[:, i] @ raw_probes[:, i]),
                                    normal=float(np.linalg.norm(probes[:n, i])), gauge=float(np.linalg.norm(probes[n:, i]))) for i in range(trace_samples)],
                  coverage=dict(normal=n, gauge=model.chart.gauge_dim, raw=model.chart.raw_dim,
                                isotropic_raw_Rademacher=True, full_normal_Gauss_covariance=True,
                                actual_heavy_light_Ref_Wick_before_compression=True),
                  weight_convention='cutoffFoot^2 INCLUDED; original GaussianFoot and physical Jhalf factored',
                  scope='point full original Phi ONLY; finite jets/stochastic spread not rigorous confidence bounds; no global L2/fullH')
    if return_probes:
        result.update(raw_probes=raw_probes, probes=probes)
    return result


def cross_word_control(model: Any, sample: dict[str, Any], sector: str,
                       coefficient: np.ndarray, probe: np.ndarray, h: float) -> dict[str, Any]:
    """Complex word states and independent full Slater cofactor mixed overlaps."""
    n = model.chart.normal_dim
    wave = sample[sector] @ coefficient
    derivative = model.directional_basis(sample, probe[:n], step=h)[sector] @ coefficient
    op = model.gauge_operators(probe[n:], sector)
    momentum = -1j*derivative+op['first'] @ wave
    outside = wave.conj().T @ op['outside_second'] @ wave
    analytic = momentum.conj().T @ momentum+outside
    identity = dict(light=np.eye(model.fibre.n, dtype=complex), heavy=np.eye(model.fibre.n, dtype=complex), reference=np.eye(4, dtype=complex))
    scale = max(1., float(np.linalg.norm(derivative)/np.linalg.norm(wave)), float(np.linalg.norm(op['body'], 2)))
    count = coefficient.shape[1]
    mixtures = np.column_stack((np.exp(.37j*np.arange(count))*(1+np.arange(count)/count), np.exp(-.61j*np.arange(count))*(1-np.arange(count)/(2*count))))
    controls = []
    for epsilon in (2.**-8/scale, 2.**-9/scale):
        unit = dict(light=expm(2j*epsilon*op['body']), heavy=expm(2j*epsilon*op['body']), reference=expm(2j*epsilon*op['reference']))
        kernel = full_cross_kernels(model, identity, unit)[sector]
        minus, plus = wave-epsilon*derivative, wave+epsilon*derivative
        cross = minus.conj().T @ kernel @ plus
        independent = (minus.conj().T @ minus+plus.conj().T @ plus-cross-cross.conj().T)/(4*epsilon**2)
        controls.append(dict(epsilon=epsilon, signed_matrix_difference=independent-analytic,
                             relative_matrix_error=float(np.linalg.norm(independent-analytic)/np.linalg.norm(analytic)),
                             complex_state_cross_values=mixtures.conj().T @ independent @ mixtures,
                             complex_state_analytic_values=mixtures.conj().T @ analytic @ mixtures,
                             independent_actual_CAR_minors=_wick_control(model, identity, unit)))
    # A gauge-only action is well conditioned independently of the enormous
    # Gaussian normal derivative. Remove the physical coherent direction, not
    # merely a coefficient index in this nonorthogonal word span.
    gram = wave.conj().T @ wave
    gauge_first = wave.conj().T @ op['first'] @ wave
    projected = wave @ np.linalg.lstsq(gram, gauge_first[:, 0], rcond=None)[0]
    original = wave[:, 0]
    residual = projected-original*np.vdot(original, projected)/np.vdot(original, original)
    outside_values = np.real(np.diag(mixtures.conj().T @ outside @ mixtures))
    return dict(rows=controls, analytic_full_cross_Wick=analytic, outside_covariance=outside,
                complex_linear_word_coefficients=mixtures, complex_state_missing_PH=outside_values,
                projected_before_Wick_wrong=mixtures.conj().T @ (momentum.conj().T @ momentum) @ mixtures,
                gauge_first_channel_matrix=gauge_first, point_word_Gram=gram,
                physical_gauge_projected_action_outside_initial_tau_norm=float(np.linalg.norm(residual)),
                physical_gauge_action_norm=float(np.linalg.norm(projected)),
                gauge_outside_tau_relative=float(np.linalg.norm(residual)/max(np.linalg.norm(projected), 1e-300)),
                all_normal_plus_Gauss=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-npz', type=Path, required=True)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace-samples', type=int, default=2)
    parser.add_argument('--seed', type=int, default=222107)
    args = parser.parse_args(); started = time.perf_counter()
    model, data = load_shared_native_cache(args.cache_npz, args.cache_manifest)
    before = loaded_source_epoch()
    sources = {str(Path(__file__).parent/name): (Path(__file__).parent/name).read_bytes() for name in before}
    sources[str(Path(__file__))] = Path(__file__).read_bytes()
    # Capture every actually imported file before probing; includes original
    # producer/guard files outside this proposal directory, never environment.
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and str(name).endswith('.py') and ('substrate-222/' in str(name) or 'substrate-framework/' in str(name)) and '/site-packages/' not in str(name):
            path = Path(name).resolve()
            sources[str(path)] = path.read_bytes()
    inputs = {str(path): path.read_bytes() for path in (args.cache_npz, args.cache_manifest)}
    producer_argv = data['manifest']['argv']
    guard_path = Path(producer_argv[producer_argv.index('--noether-guard')+1])
    inputs[str(guard_path)] = guard_path.read_bytes()
    q, em, generators = source_point(model, data)
    sample = model.basis_at(q, em)
    direction = generators['Y0']/np.linalg.norm(generators['Y0'])
    probe = lu_solve(lu_factor(np.sqrt(model.chart.mass)[:, None]*sample['point']['jacobian']),
                     2*np.random.default_rng(args.seed).integers(0, 2, model.chart.raw_dim)-1)
    rows = {}
    for profile, sector in (('neutral', 'neutral'), ('source19', 'charged'), ('SOFT', 'charged')):
        coefficient, metadata = coefficient_channels(data, sector, profile if sector == 'charged' else 'source19')
        matrix = operator_matrices_channels(model, sample, direction, sector, coefficient,
                                            trace_samples=args.trace_samples, seed=args.seed, return_probes=True)
        control = cross_word_control(model, sample, sector, coefficient, probe, model.sigma/64)
        if max(row['relative_matrix_error'] for row in control['rows']) > 1e-5:
            raise AssertionError('Independent full cross-Slater form disagrees')
        if not np.all(control['complex_state_missing_PH'] > 0):
            raise AssertionError('Wrong projected-before-Wick branch did not lose actual PH')
        if sector == 'charged' and not control['gauge_outside_tau_relative'] > 1e-6:
            raise AssertionError('Actual gauge action did not leave coherent tau')
        rows[profile] = dict(operator=matrix, coefficient_matrix=coefficient, channel_metadata=metadata, cross_word_control=control)
    coefficient, _ = coefficient_channels(data, 'neutral')
    shifted = operator_matrices_channels(model, sample, direction, 'neutral', coefficient,
                                        q_shift=.05*model.sigma*direction, trace_samples=args.trace_samples, seed=args.seed)
    after = loaded_source_epoch()
    if before != after or any(Path(path).read_bytes() != content for path, content in sources.items()):
        raise RuntimeError('Pre-read source/guard bytes changed during science')
    if any(Path(path).read_bytes() != content for path, content in inputs.items()):
        raise RuntimeError('Original input bytes changed during science')
    archive = args.output.with_suffix('.capsule.zip')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as stored:
        for path, content in sources.items():
            stored.writestr('sources/'+path.lstrip('/'), content)
        for path, content in inputs.items():
            stored.writestr('inputs/'+Path(path).name, content)
    archive.chmod(0o600)
    result = dict(component='prepared_neutral_phi_word_channels', sectors=rows, translated_density_control=shifted,
                  qFoot=q, em55=em, argv=sys.argv, cwd=str(Path.cwd()), python=sys.executable,
                  wall_seconds=time.perf_counter()-started, source_epoch_before=before, source_epoch_after=after,
                  pre_read_sources_guards={path: hashlib.sha256(content).hexdigest() for path, content in sources.items()},
                  inputs={path: hashlib.sha256(content).hexdigest() for path, content in inputs.items()},
                  capsule=str(archive), capsule_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  frozen_original_state=True, global_L2_fullH_evolution='NOT performed; parent owns global quadrature/HD/HC/scalar/G/C/evolution',
                  debts=['finite SOURCE jets', 'stochastic full-cometric trace', 'Galerkin/unrepresented domain', 'LEGAL scale and physical closure', 'quantum law and parameters'])
    encoded = json.dumps(portable(result), sort_keys=True)+'\n'
    args.output.write_text(encoded); args.output.chmod(0o600)
    print(encoded, end='')


if __name__ == '__main__':
    main()

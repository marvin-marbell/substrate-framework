#!/usr/bin/env python3
"""Stochastic full original Phi point integrands for the six held-wave trials."""
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
from prepared_neutral_ambient_weak_state import AmbientWeakState
from prepared_neutral_changing_paired_pulse import source_point
from prepared_neutral_changing_phi_kinetic import full_cross_kernels
from prepared_neutral_shared_photon_state import load_shared_native_cache, loaded_source_epoch

SECTOR_STATE = {'neutral': ('neutral', 'neutral_state'),
                'source19': ('charged', 'source19_normalized_source_state'),
                'SOFT': ('charged', 'SOFT_normalized_source_state')}


def trial_multipliers(q: np.ndarray, em: np.ndarray, direction: np.ndarray,
                      sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Interface's six real multipliers and ALL-normal derivative columns."""
    n = len(q); unit = np.asarray(direction, float)
    unit = unit / np.linalg.norm(unit)
    x = float(q @ unit) / sigma
    values = np.array([1., math.sqrt(2)*x, (2*x*x-1)/math.sqrt(2),
                       (2*float(q @ q)/sigma**2-n)/math.sqrt(2*n),
                       math.sqrt(2)*em[52], (2*float(em[:52] @ em[:52])-52)/math.sqrt(104)])
    derivative = np.zeros((n, 6))
    derivative[:, 1] = math.sqrt(2)*unit/sigma
    derivative[:, 2] = 2*math.sqrt(2)*x*unit/sigma
    derivative[:, 3] = 4*q/(sigma**2*math.sqrt(2*n))
    return values, derivative


def _held(model: Any, data: dict[str, Any], sample: dict[str, Any],
          sector: str) -> tuple[str, np.ndarray, np.ndarray]:
    fibre_sector, key = SECTOR_STATE[sector]
    coefficient = data[key]
    return fibre_sector, coefficient, sample[fibre_sector] @ coefficient


def _direction_form(model: Any, sample: dict[str, Any], coefficient: np.ndarray,
                    fibre_sector: str, wave: np.ndarray, values: np.ndarray,
                    multiplier_jet: np.ndarray, probe: np.ndarray, step: float,
                    physical_point: dict[str, Any], physical_q: np.ndarray,
                    translated: bool) -> dict[str, Any]:
    n = model.chart.normal_dim; dq, gauge = probe[:n], probe[n:]
    derivative = model.directional_basis(sample, dq, step=step)[fibre_sector] @ coefficient
    correction = 0.
    if translated:
        foot = model.chart.directional_jet(sample['q'], dq, sample['point'])['log_density_first']
        physical = model.chart.directional_jet(physical_q, dq, physical_point)['log_density_first']
        correction = .5*(foot-physical)
        derivative = derivative + correction*wave
    trial = wave[:, None]*values[None, :]
    dtrial = derivative[:, None]*values[None, :] + wave[:, None]*(dq @ multiplier_jet)[None, :]
    operators = model.gauge_operators(gauge, fibre_sector)
    momentum = -1j*dtrial + operators['first'] @ trial
    outside = trial.conj().T @ operators['outside_second'] @ trial
    raw = momentum.conj().T @ momentum + outside
    # Symmetrization removes only anti-Hermitian roundoff, never signed residuals.
    matrix = model.chi/2*sample['importance_weight']*(raw+raw.conj().T)/2
    return dict(matrix=matrix, outside_matrix=model.chi/2*sample['importance_weight']*outside,
                derivative=derivative, operators=operators, density_correction=correction)


def operator_matrices(model: Any, data: dict[str, Any], sample: dict[str, Any],
                      direction: np.ndarray, sector: str,
                      physical_point: dict[str, Any] | None = None,
                      q_shift: np.ndarray | None = None, h: float | None = None,
                      trace_samples: int = 4, seed: int = 222107) -> dict[str, Any]:
    """chi/2 full-cometric point matrix; NOT global L2 or coefficient evolution.

    All matrices INCLUDE cutoffFoot^2. The Gaussian and physical Haar density
    are factored out for the parent's original qFoot/em55 Gaussian quadrature.
    Trace directions are inverse(sqrtMass*physicalJacobian) times isotropic
    raw probes: E[zz^T]=full physical cometric, including normal/Gauss cross.
    h and h/2 are the shared producer's scale-normalized source FD steps.
    """
    if sector not in SECTOR_STATE:
        raise ValueError('sector must be neutral, source19 or SOFT')
    if trace_samples < 2:
        raise ValueError('At least two trace samples are required for observed spread')
    n = model.chart.normal_dim
    q = np.asarray(sample['q'], float)
    shift = np.zeros(n) if q_shift is None else np.asarray(q_shift, float)
    if shift.shape != (n,):
        raise ValueError('q_shift must have ALL normal coordinates')
    translated = bool(np.any(shift))
    if physical_point is not None and q_shift is None and physical_point is not sample['point']:
        raise ValueError('A distinct physical_point requires its q_shift for the density jet')
    physical_q = q+shift
    point = (model.chart.evaluate(physical_q) if translated else sample['point']) if physical_point is None else physical_point
    step = model.sigma/32 if h is None else float(h)
    if step <= 0 or not np.isfinite(step):
        raise ValueError('Positive finite FD step required')
    if np.shape(direction) != (n,) or not np.linalg.norm(direction):
        raise ValueError('Nonzero ALL-normal multiplier direction required')
    fibre_sector, coefficient, wave = _held(model, data, sample, sector)
    values, jets = trial_multipliers(q, sample['em'], direction, model.sigma)
    factor = lu_factor(np.sqrt(model.chart.mass)[:, None]*point['jacobian'])
    rng = np.random.default_rng(seed)
    raw_probes = 2*rng.integers(0, 2, size=(model.chart.raw_dim, trace_samples))-1
    probes = lu_solve(factor, raw_probes)
    rows = []; means = []; corrections = []; outside_means = []
    for finite_step in (step, step/2):
        forms = [_direction_form(model, sample, coefficient, fibre_sector, wave, values,
                                 jets, probes[:, i], finite_step, point, physical_q, translated)
                 for i in range(trace_samples)]
        matrices = np.stack([f['matrix'] for f in forms])
        rows.append(matrices); means.append(matrices.mean(axis=0))
        corrections.append([f['density_correction'] for f in forms])
        outside_means.append(np.mean([f['outside_matrix'] for f in forms], axis=0))
    centered = rows[1]-means[1]
    standard_error = np.sqrt(np.sum(np.abs(centered)**2, axis=0)/(trace_samples*(trace_samples-1)))
    return dict(matrix=means[1], matrices_by_step=means, fd_steps=[step, step/2],
                fd_difference=means[1]-means[0], fd_difference_norm=float(np.linalg.norm(means[1]-means[0])),
                standard_error_matrix=standard_error, standard_error_frobenius=float(np.linalg.norm(standard_error)),
                per_probe_matrices=rows, outside_second_matrices_by_step=outside_means,
                multipliers=values, trace_samples=trace_samples, seed=seed,
                probe_norms=[dict(raw_squared=float(raw_probes[:, i] @ raw_probes[:, i]),
                                 normal=float(np.linalg.norm(probes[:n, i])),
                                 gauge=float(np.linalg.norm(probes[n:, i]))) for i in range(trace_samples)],
                density_corrections_by_step=corrections, q_shift=shift,
                coverage=dict(normal=n, gauge=model.chart.gauge_dim, raw=model.chart.raw_dim,
                              isotropic_raw_Rademacher=True, full_normal_Gauss_covariance=True,
                              actual_heavy_light_Ref_Wick_before_compression=True),
                weight_convention='cutoffFoot^2 INCLUDED; GaussianFoot and physical Jhalf factored; parent integrates original Gaussian law',
                scope='pointwise stochastic full original Phi operator integrand ONLY; finite FD and trace errors observed, not rigorous confidence bounds')


def cross_wick_control(model: Any, data: dict[str, Any], sample: dict[str, Any],
                       direction: np.ndarray, sector: str, h: float) -> dict[str, Any]:
    """Independent cross occupied determinants, including PH outside the crop.

    Differentiate the full gauge-transformed paths exp(i e Q)(v+e Dv).
    Their mixed overlap reconstructs ||-i Dv+Qv||^2 before compression.
    Linear normal paths isolate the independent gauge/Wick second-product
    check from SOURCE finite-difference truncation. No scalar phase removed.
    """
    fibre_sector, coefficient, wave = _held(model, data, sample, sector)
    n = model.chart.normal_dim
    derivative = model.directional_basis(sample, direction[:n], step=h)[fibre_sector] @ coefficient
    op = model.gauge_operators(direction[n:], fibre_sector)
    momentum = -1j*derivative + op['first'] @ wave
    analytic = float((np.vdot(momentum, momentum)+np.vdot(wave, op['outside_second'] @ wave)).real)
    projected_wrong = float(np.vdot(momentum, momentum).real)
    # Call the ORIGINAL implementation unbound on the actual shared model,
    # so its virtual directional_basis is shared32's held functional bank.
    positive = AmbientWeakState.positive_direction(model, sample, direction)[fibre_sector]
    original = float(np.vdot(coefficient, positive @ coefficient).real)
    identity = dict(light=np.eye(model.fibre.n, dtype=complex),
                    heavy=np.eye(model.fibre.n, dtype=complex), reference=np.eye(4, dtype=complex))
    norm_scale = max(1., float(np.linalg.norm(derivative)/np.linalg.norm(wave)),
                     float(np.linalg.norm(op['body'], 2)))
    rows = []
    for epsilon in (2.**-8/norm_scale, 2.**-9/norm_scale):
        unit = dict(light=expm(2j*epsilon*op['body']), heavy=expm(2j*epsilon*op['body']),
                    reference=expm(2j*epsilon*op['reference']))
        kernel = full_cross_kernels(model, identity, unit)[fibre_sector]
        minus, plus = wave-epsilon*derivative, wave+epsilon*derivative
        mm = float(np.vdot(minus, minus).real); pp = float(np.vdot(plus, plus).real)
        cross = complex(np.vdot(minus, kernel @ plus))
        signed_numerator = mm+pp-2*cross.real
        raw = signed_numerator/(4*epsilon**2)
        estimate = model.chi/2*sample['importance_weight']*raw
        rows.append(dict(epsilon=epsilon, minus_norm=mm, plus_norm=pp, cross=cross,
                         signed_numerator=signed_numerator, raw_derivative_squared=raw,
                         point_kinetic=estimate, signed_cross_minus_analytic=raw-analytic,
                         signed_cross_minus_original=estimate-original))
    return dict(sector=sector, rows=rows, analytic_full_Wick=analytic,
                projected_before_Wick_wrong=projected_wrong,
                missing_PH_covariance=analytic-projected_wrong,
                original_positive_direction=original,
                signed_analytic_minus_original=model.chi/2*sample['importance_weight']*analytic-original,
                actual_heavy_occupied=model.heavy_orbitals.shape[1], actual_light_modes=model.fibre.n,
                actual_Ref_modes=4, full_cross_occupied_determinants=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-npz', type=Path, required=True)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace-budgets', type=int, nargs=2, default=[2, 4])
    parser.add_argument('--seed', type=int, default=222107)
    args = parser.parse_args(); started = time.perf_counter()
    model, data = load_shared_native_cache(args.cache_npz, args.cache_manifest)
    before = loaded_source_epoch()
    q, em, generators = source_point(model, data)
    sample = model.basis_at(q, em)
    direction = generators['Y0']/np.linalg.norm(generators['Y0'])
    rows = {}
    for sector in SECTOR_STATE:
        budgets = [operator_matrices(model, data, sample, direction, sector,
                                    trace_samples=budget, seed=args.seed) for budget in args.trace_budgets]
        probe = lu_solve(lu_factor(np.sqrt(model.chart.mass)[:, None]*sample['point']['jacobian']),
                         2*np.random.default_rng(args.seed).integers(0, 2, model.chart.raw_dim)-1)
        control = cross_wick_control(model, data, sample, probe, sector, model.sigma/64)
        offdiag = budgets[1]['matrix']-np.diag(np.diag(budgets[1]['matrix']))
        if not np.linalg.norm(offdiag) > 0 or not all(p['normal'] > 0 and p['gauge'] > 0 for p in budgets[1]['probe_norms']):
            raise AssertionError('Fullraw/multiplier discriminator failed')
        if abs(control['signed_analytic_minus_original']) > 1e-8*max(1., abs(control['original_positive_direction'])):
            raise AssertionError('Independent point form and original positive_direction disagree')
        rows[sector] = dict(budgets=budgets, signed_budget_difference=budgets[1]['matrix']-budgets[0]['matrix'],
                            offdiagonal_norm=float(np.linalg.norm(offdiag)), cross_Wick_control=control)
    shift = .05*model.sigma*direction
    shifted = operator_matrices(model, data, sample, direction, 'neutral', q_shift=shift,
                                trace_samples=args.trace_budgets[0], seed=args.seed)
    after = loaded_source_epoch()
    if before != after:
        raise RuntimeError('Loaded source epoch changed during scientific execution')
    archive = args.output.with_suffix('.capsule.zip')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as stored:
        for name in after:
            stored.write(Path(__file__).parent/name, 'sources/'+name)
        stored.write(args.cache_npz, 'inputs/'+args.cache_npz.name)
        stored.write(args.cache_manifest, 'inputs/'+args.cache_manifest.name)
    archive.chmod(0o600)
    result = dict(component='prepared_neutral_full_phi_commonform', sector_rows=rows,
                  translated_density_control=shifted, qFoot=q, em55=em,
                  source_epoch_before=before, source_epoch_after=after,
                  capsule=str(archive), capsule_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.cache_npz, args.cache_manifest)},
                  argv=sys.argv, cwd=str(Path.cwd()), python=sys.executable,
                  wall_seconds=time.perf_counter()-started,
                  global_quadrature_evolution='NOT performed: parent owns global quadrature/fullH/-OmegaG/-iC/actual coefficients',
                  frozen_state_no_refilter_reset_normalizer=True)
    encoded = json.dumps(portable(result), sort_keys=True)
    args.output.write_text(encoded+'\n'); args.output.chmod(0o600)
    print(encoded)


if __name__ == '__main__':
    main()

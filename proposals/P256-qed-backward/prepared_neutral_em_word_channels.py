#!/usr/bin/env python3
"""Original ALL55 EM full cross-word point operators, not global evolution."""
from __future__ import annotations
import argparse
import hashlib
import json
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
from original_charged_principal import portable
from prepared_neutral_shared_photon_state import load_shared_native_cache
from prepared_neutral_changing_paired_pulse import source_point
from prepared_neutral_changing_em_characteristic import CACHE_SHA, MANIFEST_SHA
from prepared_neutral_full_em_commonform import field_jets, multipliers, source_epoch


def channel_waves(bank: np.ndarray, modes: np.ndarray) -> np.ndarray:
    """Fibre rows; channel-major six-mode columns, with no word projection."""
    return (bank[:, :, None]*modes[None, None, :]).reshape(bank.shape[0], -1)


def kinetic_form(derivatives: np.ndarray, weights: np.ndarray) -> np.ndarray:
    flattened = (np.sqrt(weights)[:, None, None]*derivatives).reshape(-1, derivatives.shape[-1])
    return flattened.conj().T @ flattened


def operator_matrices_channels(model: Any, sample: dict[str, Any], direction: np.ndarray,
                               sector: str, coefficient_matrix: np.ndarray,
                               physical_point: dict[str, Any] | None = None,
                               q_shift: np.ndarray | None = None, h: float | None = None,
                               trace_samples: int = 4, seed: int = 222107,
                               derivative_steps: tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]] | None = None) -> dict[str, Any]:
    """Unweighted full matrix; parent multiplies original foot cutoff squared ONCE.

    Phi translation has no EM derivative: physical Haar cancels the measure.
    No physical SOURCE recapture, body H_D phase or channel normalization.
    trace_samples/seed match the Phi API; ALL55 EM uses a deterministic sum.
    """
    if sector not in ('neutral', 'charged'):
        raise ValueError('Expected neutral or charged full-bank sector')
    if np.asarray(sample['em']).shape != (55,):
        raise ValueError('Original ALL55 coordinates including harmonic Gaussian required')
    coefficients = np.asarray(coefficient_matrix, complex)
    if coefficients.ndim != 2 or coefficients.shape[0] != sample[sector].shape[1] or coefficients.shape[1] == 0:
        raise ValueError('Coefficient matrix must retain full bank rows and channel columns')
    if q_shift is not None and np.asarray(q_shift).shape != sample['q'].shape:
        raise ValueError('Full-normal mean translation required')
    step = 2.**-18 if h is None else float(h)
    if step <= 0:
        raise ValueError('Positive finite-difference step required')
    bank = sample[sector] @ coefficients
    modes, dmodes = multipliers(model, sample, direction)
    waves = channel_waves(bank, modes)
    gram = waves.conj().T @ waves
    weights = np.r_[np.asarray(model.em['omega'])/2, np.full(3, model.ue/model.law.size)]
    jets = derivative_steps if derivative_steps is not None else (field_jets(model, sample, step), field_jets(model, sample, step/2))
    forms = []
    derivatives = []
    for level in jets:
        # Original full-bank derivative includes the ALL55 Gaussian ONCE.
        dbank = np.asarray([jet[sector] @ coefficients for jet in level])
        dw = (dbank[:, :, :, None]*modes[None, None, None, :]
              + bank[None, :, :, None]*dmodes[:, None, None, :]).reshape(55, bank.shape[0], -1)
        forms.append(kinetic_form(dw, weights))
        derivatives.append(dw)
    potential_value = float(np.sum(np.asarray(model.em['omega'])*sample['em'][:52]**2/2))
    potential = potential_value*gram
    difference = float(np.linalg.norm(forms[0]-forms[1]))
    return dict(kinetic=forms[1], potential=potential, Gram=gram, gram=gram,
                total=forms[1]+potential, kinetic_coarse=forms[0], waves=waves,
                derivative_fine=derivatives[1], all55_weights=weights,
                observed_FD_errors=dict(steps=[step, step/2], kinetic_absolute=difference,
                    kinetic_relative=difference/max(float(np.linalg.norm(forms[1])), np.finfo(float).tiny),
                    derivative_absolute=float(np.linalg.norm(derivatives[0]-derivatives[1]))),
                cutoff_squared=float(sample['cutoff']**2), quadrature_weight_applied=False,
                channel_order='CHANNEL-major then six original scalar modes',
                channels=coefficients.shape[1], no_preapplied_HD_body_phase=True,
                physical_translation_EM_independent=True, held_SOURCE_not_recaptured=True,
                zero_point='Retained ONCE by literal kinetic+potential, no additive correction',
                convention='Gaussian/Haar factored; parent GLOBAL quadrature applies cutoffFoot squared ONCE')


def literal_wave(model: Any, sample: dict[str, Any], direction: np.ndarray,
                 sector: str, coefficients: np.ndarray, reference_em: np.ndarray) -> np.ndarray:
    """Independent finite packet, including actual ALL55 Gaussian variation."""
    q, em = sample['q'], sample['em']
    z = float(q @ direction)/np.linalg.norm(direction)/model.sigma
    modes = np.array([1., 2.**.5*z, (2*z*z-1)/2.**.5,
                      (2*float(q@q)/model.sigma**2-len(q))/np.sqrt(2*len(q)),
                      2.**.5*em[52], (2*float(em[:52]@em[:52])-52)/np.sqrt(104)])
    gaussian = np.exp(-float(em@em-reference_em@reference_em)/2)
    bank = sample[sector] @ coefficients
    return np.column_stack([gaussian*bank[:, channel]*modes[mode]
                            for channel in range(coefficients.shape[1]) for mode in range(6)])


def native(model: Any, data: dict[str, Any], h: float) -> dict[str, Any]:
    from prepared_neutral_phi_word_channels import coefficient_channels
    q, em, generators = source_point(model, data)
    sample = model.basis_at(q, em)
    direction = generators['Y0']
    jets = (field_jets(model, sample, h), field_jets(model, sample, h/2))
    coefficients = {}
    binding = {}
    for name in ('neutral', 'source19', 'SOFT'):
        sector = 'neutral' if name == 'neutral' else 'charged'
        coefficients[name], binding[name] = coefficient_channels(data, sector, 'source19' if name == 'neutral' else name)
    results = {name: operator_matrices_channels(model, sample, direction,
               'neutral' if name == 'neutral' else 'charged', matrix, h=h, derivative_steps=jets)
               for name, matrix in coefficients.items()}
    controls = {name: [] for name in results}
    source_derivatives = {name: [] for name in ('source19', 'SOFT')}
    rng = np.random.default_rng(222107)
    trials = {name: tuple(rng.normal(size=row['Gram'].shape[0])+1j*rng.normal(size=row['Gram'].shape[0])
                         for _ in range(2)) for name, row in results.items()}
    for step in (h, h/2):
        literal_derivatives = {name: [] for name in results}
        for j in range(55):
            shift = np.zeros(55); shift[j] = step
            plus = model.basis_at(q, em+shift, point=sample['point'])
            minus = model.basis_at(q, em-shift, point=sample['point'])
            for name, matrix in coefficients.items():
                sector = 'neutral' if name == 'neutral' else 'charged'
                finite = (literal_wave(model, plus, direction, sector, matrix, em)
                          - literal_wave(model, minus, direction, sector, matrix, em))/(2*step)
                literal_derivatives[name].append(finite)
                if step == h/2 and sector == 'charged':
                    # Literal held SOURCE/hard-polynomial derivative, without Gaussian.
                    source_derivatives[name].append(np.linalg.norm((plus[sector]@matrix-minus[sector]@matrix)/(2*step), axis=0))
        for name, row in results.items():
            finite = np.asarray(literal_derivatives[name])
            reference = row['derivative_fine']
            errors = np.linalg.norm(finite-reference, axis=(1, 2))
            relative = float(np.linalg.norm(finite-reference)/max(np.linalg.norm(reference), np.finfo(float).tiny))
            a, b = trials[name]
            da, db = finite@a, finite@b
            kinetic_bilinear = np.sum(row['all55_weights']*np.sum(da.conj()*db, axis=1))
            potential_value = float(np.sum(np.asarray(model.em['omega'])*em[:52]**2/2))
            wa = literal_wave(model, sample, direction, 'neutral' if name == 'neutral' else 'charged', coefficients[name], em)@a
            wb = literal_wave(model, sample, direction, 'neutral' if name == 'neutral' else 'charged', coefficients[name], em)@b
            literal_bilinear = kinetic_bilinear+potential_value*np.vdot(wa, wb)
            matrix_bilinear = np.vdot(a, row['total']@b)
            bilinear_relative = float(abs(literal_bilinear-matrix_bilinear)/max(abs(literal_bilinear), np.finfo(float).tiny))
            assert relative < 5e-6 and bilinear_relative < 5e-6, (name, step, relative, bilinear_relative)
            controls[name].append(dict(step=step, all55_derivative_errors=errors,
                derivative_relative=relative, literal_cross_bilinear=literal_bilinear,
                matrix_cross_bilinear=matrix_bilinear, cross_bilinear_relative=bilinear_relative))
    for name, row in results.items():
        waves = row.pop('waves')
        derivatives = row.pop('derivative_fine')
        total = row['total']
        blocks = total.reshape(row['channels'], 6, row['channels'], 6)
        cross = blocks.copy()
        for channel in range(row['channels']):
            cross[channel, :, channel, :] = 0
        # Concrete wrong operators: remove ALL55 Gaussian; drop harmonic kinetic;
        # freeze the four-word state to its initial coherent channel.
        wrong_gaussian = kinetic_form(derivatives+em[:, None, None]*waves[None, :, :], row['all55_weights'])
        no_harmonic = kinetic_form(derivatives[:52], row['all55_weights'][:52])
        a, b = trials[name]
        coherent_only = np.vdot(a[:6], total[:6, :6]@b[:6])
        full = np.vdot(a, total@b)
        row['controls'] = dict(Hermitian_residual=float(np.linalg.norm(total-total.conj().T)),
            cross_channel_operator_norm=float(np.linalg.norm(cross)),
            action_outside_initial_channel_norm=float(np.linalg.norm(total[6:, 0])),
            coherent_only_wrong_bilinear=coherent_only, full_cross_bilinear=full,
            coherent_compression_bilinear_error=float(abs(full-coherent_only)),
            omit_ALL55_Gaussian_kinetic_relative=float(np.linalg.norm(wrong_gaussian-row['kinetic'])/np.linalg.norm(row['kinetic'])),
            omit_harmonic_kinetic_relative=float(np.linalg.norm(no_harmonic-row['kinetic'])/np.linalg.norm(row['kinetic'])))
        assert row['controls']['cross_channel_operator_norm'] > 0
        assert row['controls']['action_outside_initial_channel_norm'] > 0
        assert row['controls']['omit_ALL55_Gaussian_kinetic_relative'] > 1e-8
        assert row['controls']['omit_harmonic_kinetic_relative'] > 1e-8
        assert row['controls']['coherent_compression_bilinear_error'] > 1e-8
    for name, derivatives in source_derivatives.items():
        assert np.all(np.max(np.asarray(derivatives)[52:], axis=0) > 0), name
        assert np.all(np.max(np.asarray(derivatives)[:52], axis=0) > 0), name
    return dict(scope='Full cross-word original ALL55 point forms, NOT global L2/evolution',
        q=q, em=em, matrices=results, channel_binding=binding,
        coefficient_matrices=coefficients, independent_literal_ALL55_packet_controls=controls,
        actual_source_hard_bank_derivative_norms=source_derivatives,
        initial_channel_coefficients='(1,0,...), reconstructs exact frozen initial prepared SOURCE',
        debts=['Observed finite-jet errors, not certified remainder',
               'Parent owns GLOBAL quadrature, Phi/Gauss, HC/H_D/scalar, connection and evolution',
               'Native harmonic packet unwrapped; finite Galerkin and legal-N/domain debt remain',
               'Fundamental quantum-law and microscopic parameter debt unpaid'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-npz', type=Path, required=True)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--fd', type=float, default=2.**-18)
    args = parser.parse_args()
    started = time.perf_counter()
    input_bytes = {str(path): path.read_bytes() for path in (args.cache_npz, args.cache_manifest)}
    assert hashlib.sha256(input_bytes[str(args.cache_npz)]).hexdigest() == CACHE_SHA
    assert hashlib.sha256(input_bytes[str(args.cache_manifest)]).hexdigest() == MANIFEST_SHA
    manifest = json.loads(input_bytes[str(args.cache_manifest)])
    argv = manifest['argv']
    guard_path = Path(argv[argv.index('--noether-guard')+1])
    input_bytes[str(guard_path)] = guard_path.read_bytes()  # BEFORE loader/native reads.
    from prepared_neutral_phi_word_channels import coefficient_channels
    before = source_epoch()
    with zipfile.ZipFile(args.source_archive, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, body in before.items(): archive.writestr('sources/'+name, body)
        for name, body in input_bytes.items(): archive.writestr('inputs/'+Path(name).name, body)
    args.source_archive.chmod(0o600)
    model, data = load_shared_native_cache(args.cache_npz, args.cache_manifest)
    science = native(model, data, args.fd)
    after = source_epoch()
    assert all(after.get(k) == v for k, v in before.items())
    assert all(Path(name).read_bytes() == body for name, body in input_bytes.items())
    late = {name: body for name, body in after.items() if name not in before}
    with zipfile.ZipFile(args.source_archive, 'a', zipfile.ZIP_DEFLATED) as archive:
        for name, body in late.items(): archive.writestr('sources/'+name, body)
    result = dict(science=science, argv=sys.argv, cwd=os.getcwd(), python=sys.executable,
        natural_exit=0, wall_seconds=time.perf_counter()-started,
        cache_npz_sha256=CACHE_SHA, cache_manifest_sha256=MANIFEST_SHA,
        sources={name: hashlib.sha256(body).hexdigest() for name, body in after.items()},
        inputs={name: hashlib.sha256(body).hexdigest() for name, body in input_bytes.items()},
        guard_preread_sha256=hashlib.sha256(input_bytes[str(guard_path)]).hexdigest(),
        stable_loaded_sources=True, source_archive_sha256=hashlib.sha256(args.source_archive.read_bytes()).hexdigest(),
        preread_sources_archived=sorted(before), late_loaded_sources_archived_after=sorted(late),
        metadata='Explicit nonsecret fields only; no environment serialization')
    text = json.dumps(portable(result), sort_keys=True)
    args.output.write_text(text+'\n'); args.output.chmod(0o600)
    print(text)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Positive spectral-moment upper evaluations for actual full V4 sources.

For normalized neutral trial t, lN=<t,KNt><=||KN|| in exact arithmetic.
Physical Pe[0,4] at a=.01 lies inside KC>=lN exp(-.04), even if KC>KN.
v_n=(KC/lN)^n u gives rawfraction<=exp(.08n)||v_n||²/||u||² and
charged-HALFfraction<=exp(.08n)<v_n,(KC/lN)v_n>/<u,(KC/lN)u>.
No KC maximum, global lambdaN identification or transferPe4 substitution is
needed. Floating evaluations and a normwise roundoff MODEL are reported,
not outward-rounded certificates of the underlying analytic primitive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[variable] = '1'
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh
import numba
ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = Path(__file__).read_bytes()
ENGINE_BYTES = (ROOT/'physical_c_bare_electric_mass_v4.py').read_bytes()
ENGINE_EPOCH = 'ed5aab57e88fb97fb92c833322a2c5ed2074dada2cf1606a430d571a234a9caa'
if hashlib.sha256(ENGINE_BYTES).hexdigest() != ENGINE_EPOCH:
    raise RuntimeError('Positive moment successor requires captured ed5a full-coupling engine')
import physical_c_bare_electric_mass_v4 as engine
if engine.SOURCE_BYTES != ENGINE_BYTES:
    raise RuntimeError('Actual engine changed across IMPORT')
mass = engine.mass; core = engine.core; alpha = engine.alpha; reference = engine.reference
UNIT_ROUNDOFF = 2.**-53


def gamma(operations: int) -> float:
    product = operations*UNIT_ROUNDOFF
    if product >= 1:
        raise ArithmeticError('Roundoff model operation count outside gamma domain')
    return product/(1-product)


def callback_model(carrier: engine.CompleteCarrier) -> dict[str, Any]:
    """Normwise envelope relative to EXACT algebra on represented constants.

    The estimates are themselves floating, not interval arithmetic. They do
    not bound eigh/log/SF/Gauss construction versus the analytic continuum law.
    """
    basis = carrier.basis; samples, fields = basis.shape
    gram = basis.T@basis; absolute_gram = abs(basis).T@abs(basis)
    basis_norm2_upper = float(np.max(np.sum(abs(gram), axis=1))+
        gamma(samples)*np.max(np.sum(absolute_gram, axis=1)))
    rho = abs(carrier.cs)**2+abs(carrier.ss)**2
    # Each stored gate has paired singular values sqrt(rho), other sectors1.
    # Forward and adjoint gate gains together are max(1,rho), not assumed1.
    rho_upper = rho*(1+gamma(8))
    gate_product = np.exp(np.sum(np.log(np.maximum(1., rho_upper)), axis=1))
    sample_cap = np.max(carrier.diagonal, axis=1)*gate_product
    a_cap = basis_norm2_upper*float(np.max(sample_cap))
    d_cap = float(np.max(carrier.heat_square))
    k_cap = a_cap*a_cap*d_cap
    # Projection 2fields, complex pair gate12 flops with |G|norm<=sqrt2||G||,
    # real diagonal2, reconstruction2 and sample accumulation2samples.
    gates = carrier.cs.shape[1]
    eta_gate = math.expm1(2*gates*math.log1p(math.sqrt(2)*gamma(12)))
    eta_local = (1+gamma(2*fields))*(1+eta_gate)*(1+gamma(2))**2-1
    eta_total = (1+eta_local)*(1+gamma(2*samples))-1
    absolute_compression_cap = float(np.sum(np.sum(basis*basis, axis=1)*sample_cap))
    sf_error = eta_total*absolute_compression_cap
    heat_error = gamma(2)*d_cap
    # Product expansion avoids cancellation from subtracting two large caps.
    k_error = d_cap*(2*a_cap*sf_error+sf_error*sf_error)+heat_error*(a_cap+sf_error)**2
    return dict(represented_K_norm_upper_numerical_evaluation=k_cap,
        represented_SF_norm_upper_numerical_evaluation=a_cap,
        heat_square_norm=d_cap, basis_norm_squared_upper_evaluation=basis_norm2_upper,
        callback_absolute_normwise_error_per_input_norm_MODEL=k_error,
        SF_callback_absolute_error_per_input_norm_MODEL=sf_error,
        unit_roundoff=UNIT_ROUNDOFF, gamma_projection=gamma(2*fields), gamma_sample_accumulation=gamma(2*samples),
        complex_pair_rotation_count=2*gates, local_gate_roundoff_relative_MODEL=eta_gate,
        assumptions=['IEEE binary64 round-to-nearest, no fastmath/reassociation, no overflow or norm-significant underflow.',
            'Stored rounded B/Givens/diagonal/heat coefficients are the represented operator constants.',
            'Frobenius/Euclidean norm dot errors use conservative gamma(8*vectorlength).'],
        exclusions=['No interval/outward-rounded arithmetic for these cap/error evaluations.',
            'No analytic h/eigh/Gauss/SF construction error enclosure or continuum/basis error enclosure.'],
        rigorous_physical_hardwindow_certificate=False)


def norm2_dot_model(vector: np.ndarray) -> float:
    return gamma(8*vector.size)*float(np.vdot(vector, vector).real)


def neutral_trial(carrier: engine.CompleteCarrier, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    values, vectors = eigsh(engine.CompleteOperator(carrier), k=1, which='LA', v0=carrier.mean_boundary(),
                           tol=args.eig_tolerance, maxiter=args.eig_maxiter, ncv=args.eig_ncv)
    trial = vectors[:, int(np.argmax(values))]
    trial = trial/np.linalg.norm(trial)
    applied = carrier.apply_K(trial)
    trial_norm = float(np.vdot(trial, trial).real)
    rayleigh = float(np.vdot(trial, applied).real)/trial_norm
    residual = float(np.linalg.norm(applied-rayleigh*trial)/rayleigh)
    model = callback_model(carrier)
    dot_error = gamma(8*trial.size)*float(np.linalg.norm(trial)*np.linalg.norm(applied))
    norm_error = gamma(8*trial.size)*trial_norm
    rayleigh_error = (model['callback_absolute_normwise_error_per_input_norm_MODEL']+dot_error+
                      abs(rayleigh)*norm_error)/trial_norm+gamma(2)*abs(rayleigh)
    return trial, dict(boundary='Actual normalized neutral Ritz TRIAL vector; NOT mean-prefix packet, exact global norm, or parent externalgraph photon.',
        neutral_trial_Rayleigh_lN=rayleigh, neutral_extremal_Ritz_value=float(values[-1]),
        relative_eigenpair_residual=residual, eigensolver_converged=True,
        neutral_trial_norm_squared=trial_norm,
        neutral_trial_sha256=hashlib.sha256(trial.tobytes()).hexdigest(),
        neutral_trial_bytes_hex=trial.tobytes().hex(),
        neutral_trial_dtype=str(trial.dtype), neutral_trial_shape=list(trial.shape),
        Rayleigh_absolute_error_MODEL=rayleigh_error,
        numerical_Rayleigh_minus_error_MODEL=rayleigh-rayleigh_error,
        exact_arithmetic_statement='For ANY nonzero actual trial, lN=<t,KNt>/<t,t><=lambdaNtrue; no convergence or identification with globalnorm required.',
        callback_arithmetic_MODEL=model)


def source_moments(carrier: engine.CompleteCarrier, source: np.ndarray, lN: float,
                   orders: list[int]) -> dict[str, Any]:
    start = time.time(); raw_gram = float(np.vdot(source, source).real)
    if raw_gram <= 0:
        raise ArithmeticError('Actual full Ref raw source Gram is nonpositive')
    model = callback_model(carrier); callback_error = model['callback_absolute_normwise_error_per_input_norm_MODEL']/lN
    ratio_cap = model['represented_K_norm_upper_numerical_evaluation']/lN
    callback_error += gamma(2)*ratio_cap
    current = source.copy(); vector_error = 0.; rows = []; denominator = None
    for n in range(max(orders)+1):
        following = carrier.apply_K(current, batches=4)/lN
        if not np.all(np.isfinite(following)):
            raise ArithmeticError('Actual positive moment callback overflows; no clipping')
        half_moment = float(np.vdot(current, following).real)
        norm_squared = float(np.vdot(current, current).real)
        if n == 0:
            denominator = half_moment
            if denominator <= 0:
                raise ArithmeticError('Actual complete charged-HALF source Gram nonpositive')
        if n in orders:
            factor = math.exp(.08*n)
            if denominator is None:
                raise ArithmeticError('Actual halfsource denominator was not executed')
            raw_bound = factor*norm_squared/raw_gram
            half_bound = factor*half_moment/denominator
            current_norm = math.sqrt(norm_squared)
            norm_error = 2*current_norm*vector_error+vector_error*vector_error+norm2_dot_model(current)
            dot_error = gamma(8*current.size)*current_norm*float(np.linalg.norm(following))
            quadratic_error = ratio_cap*(2*current_norm*vector_error+vector_error*vector_error)+callback_error*norm_squared+dot_error
            rows.append(dict(moment_order=n, exp_threshold_factor=factor,
                actual_positive_vn_norm_squared=norm_squared,
                actual_positive_vn_R_vn=half_moment,
                raw_physical_Pe0_4_fraction_UPPER_numerical_evaluation_UNCLIPPED=raw_bound,
                charged_HALF_physical_Pe0_4_fraction_UPPER_numerical_evaluation_UNCLIPPED=half_bound,
                cumulative_vn_absolute_norm_error_MODEL=vector_error,
                vn_norm_squared_absolute_error_MODEL=norm_error,
                vn_R_vn_absolute_error_MODEL=quadratic_error,
                callback_is_finite=True,
                statement='Numerical evaluation of exact positive spectral moment inequality; NOT a rigorous hardwindow certificate.'))
            print(json.dumps(dict(stage='actual_positive_physical_window_moment', number=carrier.number, n=n,
                raw_upper_UNCLIPPED=raw_bound, HALF_upper_UNCLIPPED=half_bound,
                raw_Gram=raw_gram, HALF_Gram=denominator)), flush=True)
        vector_error = ratio_cap*vector_error+callback_error*math.sqrt(norm_squared)
        current = following
    return dict(actual_raw_source_Gram=raw_gram, actual_complete_charged_HALF_source_Gram=denominator,
        actual_raw_source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        ONE_neutral_trial_Rayleigh_lN=lN, rows=rows, callback_arithmetic_MODEL=model,
        normalized_callback_absolute_normwise_error_per_input_norm_MODEL=callback_error,
        callback_upper_cap_over_trial_lN=ratio_cap,
        physical_window_tail_threshold_over_trial_lN=math.exp(-.04),
        numerical_cap_lies_below_tail_threshold=ratio_cap < math.exp(-.04),
        numerical_cap_comparison_is_not_analytic_certificate=True,
        runtime_seconds=time.time()-start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bare-mass', type=float, default=mass.OLD_MASS)
    parser.add_argument('--quadrature-order', type=int, default=2)
    parser.add_argument('--moment-orders', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--eig-tolerance', type=float, default=1e-6)
    parser.add_argument('--eig-maxiter', type=int, default=60)
    parser.add_argument('--eig-ncv', type=int, default=8)
    args = parser.parse_args(); start = time.time()
    if not args.moment_orders or min(args.moment_orders) < 1:
        raise ValueError('Positive source moment orders must be at least1')
    neutral = engine.CompleteCarrier(4, 8, args.bare_mass, args.quadrature_order)
    trial, trial_receipt = neutral_trial(neutral, args)
    lN = trial_receipt['neutral_trial_Rayleigh_lN']
    photon = np.zeros_like(trial).reshape(len(neutral.field_heat), neutral.count)
    photon[1] = trial.reshape(len(neutral.field_heat), neutral.count)[0]
    photon_gram = float(np.vdot(photon, photon).real)
    if photon_gram <= 0:
        raise ArithmeticError('Existing projected mode1 photon on Ritz trial vanishes')
    preparations = [('pureS_on_actual_neutral_Ritz_TRIAL', trial),
                    ('existing_projected_mode1_photon_thenS_on_actual_neutral_Ritz_TRIAL', photon.reshape(-1)/math.sqrt(photon_gram))]
    cases = []; charged_diagnostics = {}
    for number in (9, 7):
        charged = engine.CompleteCarrier(4, number, args.bare_mass, args.quadrature_order)
        for label, incoming in preparations:
            source = engine.apply_original_source(neutral, charged, incoming)
            print(json.dumps(dict(stage='actual_fullV4_moment_source', number=number, preparation=label,
                source_Gram=float(np.vdot(source, source).real))), flush=True)
            cases.append(dict(number=number, preparation=label,
                result=source_moments(charged, source, lN, sorted(set(args.moment_orders)))))
        charged_diagnostics[str(number)] = charged.diagnostics()
    descriptor = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        volume=4, geometry='4x1x1, all16orbitals/all9real quotient modes/fullcoupling/normal1/colour1/centre0',
        source_boundary=trial_receipt['boundary'], trial_sha256=trial_receipt['neutral_trial_sha256'],
        projected_photon='Existing grade1 mode1 insertion; not parent exact externalgraph/fullCCR photon.',
        physical_window=[0, 4], transfer_threshold=.04, engine_epoch=ENGINE_EPOCH,
        global_neutral_norm_assumed_known=False, charged_maximum_assumed=False)
    inputs = json.dumps(descriptor, sort_keys=True)
    sources = {Path(__file__).name: SOURCE_BYTES, 'physical_c_bare_electric_mass_v4.py': ENGINE_BYTES,
        'physical_c_bare_electric_mass.py': engine.MASS_BYTES, 'physical_c_bare_electric_kernel.py': mass.CORE_BYTES,
        'captured_reference89.py': core.REFERENCE_BYTES, 'captured_ledger.py': reference.LEDGER_BYTES,
        **{'captured_Alpha56/'+name: data for name, data in alpha.SOURCE_BYTES.items()}}
    unchanged = {Path(__file__).name: Path(__file__).read_bytes() == SOURCE_BYTES,
        'physical_c_bare_electric_mass_v4.py': (ROOT/'physical_c_bare_electric_mass_v4.py').read_bytes() == ENGINE_BYTES,
        'physical_c_bare_electric_mass.py': (ROOT/'physical_c_bare_electric_mass.py').read_bytes() == engine.MASS_BYTES,
        'physical_c_bare_electric_kernel.py': (ROOT/'physical_c_bare_electric_kernel.py').read_bytes() == mass.CORE_BYTES,
        **{name: (ROOT/name).read_bytes() == data for name, data in mass.INPUT_BYTES.items()}}
    if not all(unchanged.values()):
        raise RuntimeError('Moment successor source or startup input changed during actual execution')
    execution: dict[str, Any] = dict(source_at_IMPORT={name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
        source_bytes_at_IMPORT={name: data.decode() for name, data in sources.items()},
        input_json=inputs, input_sha256=hashlib.sha256(inputs.encode()).hexdigest(), startup_unchanged=unchanged,
        capsule_hashes_at_IMPORT={name: hashlib.sha256(data).hexdigest() for name, data in mass.INPUT_BYTES.items()},
        runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                     numba=numba.__version__, platform=platform.platform(), thread_environment={name: os.environ.get(name) for name in
                         ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')}),
        runtime_seconds=time.time()-start, governing_plan=mass.PLAN_ID)
    result = dict(actual_neutral_Ritz_TRIAL=trial_receipt, existing_projected_mode1_photon_Gram=photon_gram,
        actual_positive_physical_window_source_moment_cases=cases,
        complete_carrier_diagnostics={'8': neutral.diagnostics(), **charged_diagnostics}, execution=execution,
        exact_inequality='Pe_phys[0,4] subset KC>=lambdaNtrue exp(-.04)>=lN exp(-.04). Raw <=exp(.08n)||R^n u||²/||u||²; HALF <=exp(.08n)<R^n u,R R^n u>/<u,Ru>. R=KC/lN. lambdaNtrue cancels from HALF fraction.',
        limits=['Numerical evaluations of exact moment inequality, not rigorous hardwindow certification.',
            'Roundoff MODEL covers represented coefficients callback only; no interval/analytic primitive eigen/log/quadrature error enclosure.',
            'Actual Ritz trial boundary differs from historical mean-prefix source boundary; projected photon is not parent externalgraph.',
            'V4x1x1 retains y/z selfloops. No mass1/pole/macro/spin/translation/kineticmass admission.',
            'Normal1 sampled outsidefactor zero; declared global outside law remains nonidentity.'],
        no_globalnorm_KCmax_assumption=True, no_hardwindow_zero_from_Krylov=True,
        physical_mass1_or_pole_admitted=False)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(stage='capsule_written', output=str(args.output), runtime_seconds=time.time()-start,
        source_sha256=execution['source_at_IMPORT'][Path(__file__).name], input_sha256=execution['input_sha256'])), flush=True)


if __name__ == '__main__':
    main()

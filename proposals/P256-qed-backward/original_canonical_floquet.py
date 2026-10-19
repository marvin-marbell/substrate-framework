#!/usr/bin/env python3
"""Actual original 500-mode noncommuting Floquet/CAR neutral projection.

Piecewise-constant continuous time is evaluated by the source exponential API.
The 501-point Fourier rule is exact for the entire finite-CAR polynomial, not
an occupation, Hamiltonian, photon, or eigenmode cutoff. No T->infinity run.
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
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import scipy
from scipy.linalg import eig, expm, svdvals
from scipy.optimize import linear_sum_assignment

# These are the actual mass-density consumer's factories, not a new supplier.
from original_mass_density import (
    BARE, BulkQuantumLaw, RotationBand, accept, exterior_hamiltonian,
    native_source_hashes as mass_source_hashes, original_frame, rejected,
)
from original_charged_principal import portable
from original_compact_charge_bound import chiral_basis
from original_massive_photon_locality import array_hash, tree_onebody
from original_uniform_charge_gap import time_evolution

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-continuing-backward-20260930'
PLAN_REVISION = 'a07663f778849e7081540356af537bb3ad2c5c435dd16f70466705ac800534cc'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'
FROZEN_REVIEW = {
    'issue222-native-affine-defect-phase.md':
        'a6e5d4213f5c6b5f38ed75b17fb9f3b6f6978cc1d3c4a89a6c4e3fc6135cf84d',
    'issue222-native-neutral-haar-freeze.md':
        '6fd02b12132030e88b3ff9083efdc009e30314c8bb748a91e921b2c21d7dba0e',
}


def source_hashes() -> dict[str, str]:
    return mass_source_hashes() | {
        Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def physical_assertion(logabs: float, phase: complex, expected: float,
                       tolerance: float) -> float:
    """Same physical assertion for the actual result and no-background control."""
    error = max(abs(phase-1), abs(logabs-expected))
    accept(float(error), 'background-two complex logdet equals positive whole-CAR product', tolerance)
    return float(error)


def elementary(values: np.ndarray) -> np.ndarray:
    """All elementary coefficients, including the empty and full occupations."""
    coefficients = np.zeros(len(values)+1, dtype=np.complex128)
    coefficients[0] = 1
    for degree, value in enumerate(values):
        coefficients[1:degree+2] += value*coefficients[:degree+1]
    return coefficients


def full_car_tree(frame: np.ndarray, band: Any) -> dict[str, Any]:
    """Independent ordered exterior traces on all 256 actual tree CAR states."""
    hs = [np.asarray(tree_onebody(frame, band, phase), dtype=np.complex128)
          for phase in (.37, -1.19)]
    durations = np.array([.7, 1.1])
    temporal = np.array([[.31, -.27], [-.16, .42]])
    temporal += (math.pi/2-2*np.sum(durations[:, None]*temporal))/(4*durations.sum())
    u, _, background = time_evolution(hs, temporal, durations)
    commutator = float(np.linalg.norm(hs[0]@hs[1]-hs[1]@hs[0]))
    require(commutator > 1e-5, 'actual tree Hamiltonians genuinely do not commute')
    sector_traces, dimensions = [], []
    for number in range(9):
        sector_u = None
        basis = []
        for h, a0, dt in zip(hs, temporal, durations):
            # The owner validates HERMITIAN Hamiltonians, not complex generators.
            sector_h, basis = exterior_hamiltonian(h, number)
            sector_a0, _ = exterior_hamiltonian(np.diag(np.repeat(a0, 4)), number)
            sector_generator = -sector_h+1j*sector_a0
            step = expm(dt*sector_generator)
            sector_u = step if sector_u is None else step@sector_u
        assert sector_u is not None, 'entire-CAR ordered sector evolution exists'
        sector_traces.append(complex(np.trace(sector_u)))
        dimensions.append(len(basis))
    require(sum(dimensions) == 256, 'all eight-mode exterior sectors retained')
    rows = []
    for alpha in (0., .47, 1.73):
        car = np.exp(-1j*background)*sum(
            trace*np.exp(1j*alpha*(number-4))
            for number, trace in enumerate(sector_traces))
        determinant = np.exp(-1j*(background+4*alpha))*np.linalg.det(
            np.eye(8)+np.exp(1j*alpha)*u)
        error = abs(car-determinant)/max(1., abs(car))
        accept(float(error), 'independent all256CAR trace equals actual physical determinant')
        accept(float(abs(car.imag)/max(1., abs(car))), 'independent all256CAR physical trace real')
        require(car.real > 0, 'independent all256CAR physical trace positive')
        rows.append(dict(alpha=alpha, exterior_trace=car, physical_determinant=determinant,
                         relative_error=error))
    return dict(modes=8, whole_CAR_states=256, sector_dimensions=dimensions,
                Hamiltonian_commutator_norm=commutator, durations=durations,
                temporal=temporal, background_phase=background, rows=rows,
                scope='Independent finite tree exterior-algebra identity, not periodic quantum photon ground')


def native_case(centre: tuple[int, int, int], seed: int,
                alpha_samples: int, chosen_alpha: float) -> dict[str, Any]:
    started = time.perf_counter()
    band = RotationBand(**{**BARE, 'a': 1.}, e=1.)
    require(band.m == .2 and not band.improved, 'actual original positive-m=.2 nearest-axis law')
    size, period = 5, 8.
    selected, frame = original_frame(band, size, np.pi/2*np.asarray(centre), True)
    gram = np.asarray(selected['G'], dtype=float)
    require(bool(selected['full10_Gram']), 'original heavy-supplied full ten-coordinate Gram')
    plus, minus, beta4 = chiral_basis(frame, gram, 1)
    j4 = plus@plus.conj().T-minus@minus.conj().T
    b = np.linalg.solve(gram[1:, 1:], gram[1:, 0])
    schur = float(gram[0, 0]-gram[0, 1:]@b)
    mu = band.m*math.sqrt(schur)
    law = BulkQuantumLaw(size, band=band)
    modes, neutral_number = 4*law.volume, 2*law.volume
    require(modes == 500 and neutral_number == 250, 'entire original N5 onebody')
    require(alpha_samples >= modes+1, 'at least501 samples: no finite-CAR Fourier aliasing')
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.pi/2*np.asarray(centre))
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, (2, law.volume, 3))
    temporal = rng.uniform(-.35, .35, (2, law.volume))
    durations = np.array([3., 5.])
    # Fix a visible nontrivial bare phase without removing a single local a0.
    temporal += (math.pi/2-2*np.sum(durations[:, None]*temporal))/(2*law.volume*period)
    hs = [np.asarray(law.onebody(fields, links, phi), dtype=np.complex128) for phi in phases]
    beta, j = np.kron(np.eye(law.volume), beta4), np.kron(np.eye(law.volume), j4)
    identity = np.eye(modes, dtype=np.complex128)
    generators = [-h+1j*np.diag(np.repeat(a0, 4)) for h, a0 in zip(hs, temporal)]
    h_commutator = float(np.linalg.norm(hs[0]@hs[1]-hs[1]@hs[0]))
    generator_commutator = float(np.linalg.norm(
        generators[0]@generators[1]-generators[1]@generators[0]))
    require(min(h_commutator, generator_commutator) > 1e-3,
            'native Hamiltonians and actual temporal generators genuinely noncommuting')
    require(min(float(np.ptp(a0)) for a0 in temporal) > .1, 'genuinely inhomogeneous real temporal cochains')
    coercive, generator_j_residuals = [], []
    for h, generator in zip(hs, generators):
        accept(float(np.linalg.norm(h-h.conj().T)), 'actual light onebody Hermitian')
        real_beta = (beta@h+h@beta)/2
        least = float(np.linalg.eigvalsh(real_beta)[0])
        accept(max(0., mu-least), 'actual Schur-beta Lyapunov coercivity')
        coercive.append(least)
        defect = float(np.linalg.norm(generator.conj().T@j+j@generator))
        accept(defect, 'actual temporal generator J-skew-adjoint')
        generator_j_residuals.append(defect)
    u, _, background = time_evolution(hs, temporal, durations)
    shifted, _, shifted_background = time_evolution(
        hs, temporal+chosen_alpha/period, durations)
    shift_error = float(np.linalg.norm(shifted-np.exp(1j*chosen_alpha)*u)/max(1., np.linalg.norm(u)))
    accept(shift_error, 'actual source positive global-holonomy sign')
    accept(abs(shifted_background-background-neutral_number*chosen_alpha),
           'actual source background-two global-holonomy sign')
    norm_u = float(svdvals(u)[0])
    j_defect = u.conj().T@j@u-j
    j_absolute = float(np.linalg.norm(j_defect, ord='fro')/math.sqrt(modes))
    j_scaled = float(np.linalg.norm(j_defect, ord='fro')/max(1., norm_u**2*math.sqrt(modes)))
    eps = float(np.finfo(float).eps)
    gamma_n = modes*eps/(1-modes*eps)
    j_tolerance = max(2e-8, 32*gamma_n*max(1., norm_u**2))
    require(j_tolerance < 1e-4, 'T8 actual J check numerically resolved, not a large-T fake')
    accept(j_absolute, 'actual J-pseudo-unitarity absolute per-mode residual', j_tolerance)
    eigenvalues, left, right = eig(u, left=True, right=True)
    absolute = abs(eigenvalues)
    vector_singular = svdvals(right)
    eigenvector_condition = float(vector_singular[0]/vector_singular[-1])
    eigen_residual = float(np.linalg.norm(u@right-right*eigenvalues[None, :], ord='fro'))
    local_conditions = 1/abs(np.sum(left.conj()*right, axis=0))
    # Conservative diagnostic for finite precision, NOT an outward interval
    # enclosure of expm or a proof at T=infinity. Report rather than hide it.
    backward_estimate = eigen_residual/float(vector_singular[-1])+gamma_n*norm_u
    eigen_absolute_allowance = 8*eigenvector_condition*backward_estimate
    eigen_relative_allowance = eigen_absolute_allowance/float(absolute.min())
    # The physical determinant contains (1+z*lambda), NOT lambda itself.
    # Its small-mode factors are well conditioned even when relative eigenvalue
    # error is larger. Keep all independently computed modes and check the
    # condition of the ACTUAL observable, including the separate vacuum/pair terms.
    separation = abs(absolute-1)
    require(float(separation.min()) > eigen_absolute_allowance
            and eigen_relative_allowance < .1, 'all modes and unit-circle separation resolved at T8')
    factor_budget = float(np.sum(eigen_absolute_allowance/(separation-eigen_absolute_allowance)))
    large_moduli = absolute[absolute > 1]
    vacuum_budget = float(np.sum(eigen_absolute_allowance/(large_moduli-eigen_absolute_allowance)))
    positive_budget = float(np.sum(2*eigen_absolute_allowance/(
        (large_moduli-eigen_absolute_allowance)*(large_moduli-1-eigen_absolute_allowance))))
    product_tolerance = max(2e-8, factor_budget+vacuum_budget+positive_budget)
    require(product_tolerance < 1e-4,
            'actual full-CAR product conditioning discriminating at T8; no deleted or imposed modes')
    small, large = eigenvalues[absolute < 1], eigenvalues[absolute > 1]
    require(len(small) == len(large) == neutral_number, 'independent250large/250small actual eigencount')
    q = math.exp(-mu*period)
    accept(max(0., float(abs(small).max())-q), 'computed small-mode Lyapunov annulus',
           max(2e-8, eigen_absolute_allowance))
    accept(max(0., 1/q-float(abs(large).min())), 'computed large-mode Lyapunov annulus',
           max(2e-8, eigen_absolute_allowance))
    targets = 1/large.conj()
    match_rows, match_columns = linear_sum_assignment(abs(small[:, None]-targets[None, :]))
    pairing_error = float(np.max(abs(small[match_rows]-targets[match_columns])))
    accept(pairing_error, 'measured J reciprocal-conjugate pairing, never imposed',
           max(2e-8, 4*eigen_absolute_allowance))
    vacuum_logabs = float(np.log(abs(large)).sum())
    vacuum_phase = complex(np.exp(1j*(np.angle(large).sum()-background)))
    accept(abs(vacuum_phase-1), 'actual background-corrected large-product vacuum positive', product_tolerance)
    det_u_phase = complex(np.exp(1j*(np.angle(eigenvalues).sum()-2*background)))
    accept(abs(det_u_phase-1), 'actual all500 detU phase exp4iA0', product_tolerance)
    accept(abs(float(np.log(absolute).sum())), 'actual all500 detU log modulus zero', product_tolerance)
    alphas = 2*np.pi*np.arange(alpha_samples)/alpha_samples
    # ALL computed 500 eigenvalues used here, including independently obtained
    # small ones. The positive-pair expression is a separate comparison.
    factors = 1+np.exp(1j*alphas[:, None])*eigenvalues[None, :]
    full_logs = np.log(abs(factors)).sum(axis=1)
    full_phases = np.exp(1j*(np.angle(factors).sum(axis=1)-background-neutral_number*alphas))
    reciprocal_large = 1/large
    positive_logs = 2*np.log(abs(1+np.exp(-1j*alphas[:, None])*reciprocal_large[None, :])).sum(axis=1)
    accept(float(np.max(abs(full_logs-vacuum_logabs-positive_logs))),
           'all500 computed eigenmode product versus positive product', product_tolerance)
    accept(float(np.max(abs(full_phases-1))), 'all501 physical holonomies real positive', product_tolerance)
    direct_rows = []
    wrong_background = None
    for alpha in (0., chosen_alpha, math.pi, 2.31):
        matrix = identity+np.exp(1j*alpha)*u
        sign, logabs = np.linalg.slogdet(matrix)
        expected = vacuum_logabs+float(2*np.log(abs(
            1+np.exp(-1j*alpha)*reciprocal_large)).sum())
        physical_phase = complex(sign*np.exp(-1j*(background+neutral_number*alpha)))
        singular = svdvals(matrix)
        condition = float(singular[0]/singular[-1])
        determinant_rounding_estimate = 16*modes*gamma_n*condition
        tolerance = max(product_tolerance, determinant_rounding_estimate)
        require(tolerance < 1e-4, 'actual matrix complex logdet numerically discriminating')
        error = physical_assertion(float(logabs), physical_phase, expected, tolerance)
        if alpha == 0.:
            wrong_background = rejected(lambda: physical_assertion(
                float(logabs), complex(sign), expected, tolerance))
        direct_rows.append(dict(alpha=alpha, physical_logabs=float(logabs), physical_phase=physical_phase,
                                positive_product_logabs=expected, physical_assertion_error=error,
                                condition_I_plus_U=condition, rounding_estimate=determinant_rounding_estimate,
                                tolerance=tolerance))
    require(wrong_background is not None, 'bare complex determinant adverse actually reaches physical assertion')
    chosen_log = float(2*np.log(abs(1+np.exp(-1j*chosen_alpha)*reciprocal_large)).sum())
    # Rescale ONLY scalar arithmetic, never delete a mode or a CAR sector.
    normalizer = float(positive_logs.max())
    weights = np.exp(positive_logs-normalizer)
    canonical_log = normalizer+math.log(float(weights.mean()))
    fourier_zero_scaled = complex(np.mean(weights))
    fourier_plus_one_scaled = complex(np.mean(weights*np.exp(-1j*alphas)))
    coefficients = elementary(reciprocal_large)
    exact_zero = float(np.vdot(coefficients, coefficients).real)
    exact_plus_one = complex(np.dot(coefficients[:-1], coefficients[1:].conj()))
    coefficient_error = max(abs(fourier_zero_scaled-exact_zero*np.exp(-normalizer)),
                            abs(fourier_plus_one_scaled-exact_plus_one*np.exp(-normalizer)))
    accept(float(coefficient_error), '501point exact neutral and q+1 coefficients versus all elementary sectors',
           max(2e-8, product_tolerance))
    accept(abs(canonical_log-math.log(exact_zero)), 'neutral canonical log Fourier versus coefficient identity',
           product_tolerance)
    lower_log, upper_log = modes*math.log1p(-q), modes*math.log1p(q)
    eta = upper_log-lower_log
    accept(max(0., lower_log-float(positive_logs.min()), float(positive_logs.max())-upper_log),
           'Schur annulus pointwise vacuum bounds', product_tolerance)
    accept(max(0., lower_log-canonical_log, canonical_log-upper_log),
           'normalized neutral canonical vacuum bounds', product_tolerance)
    accept(max(0., abs(canonical_log-chosen_log)-eta),
           'neutral chosen-point versus complete canonical comparison', product_tolerance)
    first_coefficient = fourier_plus_one_scaled*np.exp(normalizer)
    spectral_lower = float(2*np.log1p(-abs(reciprocal_large)).sum())
    spectral_upper = float(2*np.log1p(abs(reciprocal_large)).sum())
    return dict(size=size, volume=law.volume, modes=modes, neutral_NL=neutral_number,
        centre=centre, spacing=band.a, mass=band.m, period=period, durations=durations, seed=seed,
        bare_parameters=BARE, original_heavy_full10_Gram=gram,
        heavy_Gram_gradient_residual=selected['gradient_residual'], Schur=schur, mu=mu,
        factory='original_mass_density.original_frame; BulkQuantumLaw.onebody geometry only',
        selector27='constructor metadata only; no selector Hamiltonian/sea executed',
        selected_inputs=dict(frame_sha256=array_hash(frame), beta_sha256=array_hash(beta4),
            spatial_cochain_sha256=array_hash(phases), temporal_cochain_sha256=array_hash(temporal),
            native_h_sha256=[array_hash(h) for h in hs], toron_link_sha256=array_hash(links),
            compact_phase_range=[float(phases.min()), float(phases.max())],
            temporal_range=[float(temporal.min()), float(temporal.max())]),
        noncommuting=dict(Hamiltonian_commutator=h_commutator, generator_commutator=generator_commutator),
        source_sign=dict(generator='-h + i diag(a0)', uniform_shift='U(alpha)=exp(+i alpha) U',
            physical_background='exp(-i[2 integral sum_x a0 + 2V alpha])',
            background_phase=background, shifted_background_phase=shifted_background,
            actual_shift_relative_error=shift_error),
        Lyapunov=dict(beta_coercive_minima=coercive, q=q, large_count=len(large), small_count=len(small),
            smallest_large=float(abs(large).min()), largest_small=float(abs(small).max())),
        actual_J=dict(generator_residuals=generator_j_residuals, absolute_per_mode=j_absolute,
            norm_scaled=j_scaled, absolute_tolerance=j_tolerance,
            measured_reciprocal_pairing_error=pairing_error),
        numerical_conditioning=dict(monodromy_norm=norm_u, eigenvector_condition=eigenvector_condition,
            max_left_right_eigencondition=float(local_conditions.max()), eigen_residual_Frobenius=eigen_residual,
            eigen_absolute_rounding_allowance=eigen_absolute_allowance,
            smallest_eigenvalue=float(absolute.min()), eigen_relative_rounding_allowance=eigen_relative_allowance,
            positive_product_tolerance=product_tolerance,
            physical_factor_log_error_budget=factor_budget, vacuum_log_error_budget=vacuum_budget,
            positive_pair_log_error_budget=positive_budget,
            limitation='Finite-T8 float64 diagnostic allowances, not certified expm/eigenvalue intervals; '
                       'conditioning deteriorates with T. No T-infinity numerical proof, deleted small modes, '
                       'or imposed reciprocal eigenvalues.'),
        all500_computed_eigenvalues=eigenvalues, direct_complex_logdet=direct_rows,
        wrong_bare_complexdet_no_background_rejected=wrong_background,
        vacuum=dict(logabs=vacuum_logabs, physical_phase=vacuum_phase),
        entire_CAR=dict(total_states='2^500', determinant_degree=500,
            occupation='every0..500 sector; no blockwise occupation imposed'),
        canonical=dict(alpha_samples=alpha_samples, equally_spaced=True,
            exact_rule='501 or more samples resolves entire degree500 CAR polynomial; not Hamiltonian truncation',
            chosen_alpha=chosen_alpha, normalized_to_vacuum_log=canonical_log,
            normalized_to_vacuum=math.exp(canonical_log), chosen_to_vacuum_log=chosen_log,
            canonical_over_chosen_point=math.exp(canonical_log-chosen_log),
            log_canonical_over_chosen_point=canonical_log-chosen_log,
            Schur_vacuum_log_bounds=[lower_log, upper_log], neutral_pointfreeze_log_bound=eta,
            computed_spectral_vacuum_log_bounds=[spectral_lower, spectral_upper],
            elementary_coefficient_comparison_error=coefficient_error),
        charge_plus_one=dict(NL=neutral_number+1, normalized_to_vacuum_coefficient=first_coefficient,
            elementary_coefficient=exact_plus_one,
            exact_formula='[exp(+i alpha)] ZF/Zvac = sum_(k=0..249) e_k(1/lambda_large) conjugate(e_(k+1)(1/lambda_large))',
            constant_vacuum_coefficient=0.,
            exclusion='A q+1 Fourier coefficient is not the neutral canonical average or a point value. '
                      'The constant vacuum has exactly zero q+1 coefficient. No charged-sector relative '
                      'point-freeze, reference/Gauss closure, or charge1 threshold follows.'),
        independent_full_CAR_tree=full_car_tree(frame, band),
        elapsed_seconds=time.perf_counter()-started)


def run(alpha_samples: int = 501, chosen_alpha: float = .47) -> dict[str, Any]:
    require(alpha_samples >= 501 and math.isfinite(chosen_alpha), 'finite real alpha and exact full-CAR sampling')
    started = time.perf_counter()
    before = source_hashes()
    rows = [native_case(centre, 2221100+i, alpha_samples, chosen_alpha)
            for i, centre in enumerate(((0, 0, 0), (1, 0, 2)))]
    require(before == source_hashes(), 'native sources unchanged throughout Floquet consumer')
    return dict(plan=PLAN, assignment_revision=PLAN_REVISION, template_revision=TEMPLATE_REVISION,
        source_root='.', source_hashes=before, frozen_local_review_hashes=FROZEN_REVIEW,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                     elapsed_seconds=time.perf_counter()-started), rows=rows, all_checks=True,
        scoped_proofs=['Schur-beta Floquet annulus; actual J pairing; all500 physical determinant product',
                       'Independent all256 tree CAR trace; exact finite Fourier neutral/q+1 coefficients',
                       'Neutral-only chosen-point/canonical bounds at genuine finiteT8'],
        scope='Prescribed original full-Gram compact backgrounds and continuous-time light CAR only. '
              'Not integrated quantum periodic Gauss ground, Coulomb/fermion phase, charged/mobile electron, '
              'physical spin, massless photon, thermodynamic or UV completion.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'original_canonical_floquet_receipt.json')
    parser.add_argument('--alpha-samples', type=int, default=501)
    parser.add_argument('--chosen-alpha', type=float, default=.47)
    args = parser.parse_args()
    result = run(args.alpha_samples, args.chosen_alpha)
    args.output.write_text(json.dumps(portable(result), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), all_checks=True, cases=len(result['rows']),
                          period=8., modes=500, neutral_only=True, compact_phase_proved=False)))


if __name__ == '__main__':
    main()

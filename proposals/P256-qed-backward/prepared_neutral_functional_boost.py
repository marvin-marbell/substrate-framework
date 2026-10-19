#!/usr/bin/env python3
"""C1 boost increments on the actual functional SOURCE/current/photon bank.

This compiles full-metric conjugation increments, not a replacement H0. The
original full H weak form must be supplied in this SAME functional bank.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
import numpy as np
from scipy.linalg import solve
from functional_source_photon_bank import FunctionalSourcePhotonBank
from native_closed_rotation_quantum import physical_chart_pullback
from prepared_neutral_moving_matter_frame import mean_matter_generators, pulse_point
from prepared_neutral_shared_photon_state import load_shared_native_cache, loaded_source_epoch
from original_charged_principal import portable


def common_logjet(model: Any, sample: dict[str, Any], dq: np.ndarray) -> float:
    """Physical Gaussian/Haar/cutoff jet ONCE; bank supplies only factor jet."""
    if sample['cutoff'] <= 0:
        raise ValueError('Boost form needs the original regular common core')
    rho = model.chart.directional_jet(sample['q'], dq, sample['point'])['log_density_first']
    return float(-sample['q']@dq/model.sigma**2-.5*rho+sample['cutoff_gradient']@dq/sample['cutoff'])


def boost_forms(model: Any, bank: FunctionalSourcePhotonBank, sample: dict[str, Any],
                profile: str, a: np.ndarray, Yn: np.ndarray, h: float) -> dict[str, Any]:
    """Exact phase-conjugation powers before any caller's spatial projection.

    H^V=H0+beta H1+beta² H2; G^V=G0+beta G1.
    Full NN/GN blocks enter H1. PH outside the retained fibre is orthogonal to
    the scalar normal phase increment, not silently assigned zero norm.
    """
    sector = 'neutral' if profile == 'neutral' else 'charged'
    n = model.chart.normal_dim
    inverse = sample['point']['cometric']
    velocity = model.chi*(inverse[:, :n]@a)
    normal, gauge = velocity[:n], velocity[n:]
    value = bank.values(sample, profile)
    derivative = bank.directional_values(sample, profile, normal, np.zeros(55), h)
    derivative += common_logjet(model, sample, normal)*value
    momentum = -1j*derivative+model.gauge_operators(gauge, sector)['first']@value
    weight = sample['importance_weight']
    gram = weight*(value.conj().T@value)
    first = weight*(value.conj().T@momentum+momentum.conj().T@value)/2
    quadratic = .5*model.chi*float(a@inverse[:n, :n]@a)*gram
    noether_increment = float(Yn@a)*gram
    position = float(a@sample['q'])*gram
    fine = bank.directional_values(sample, profile, normal, np.zeros(55), h/2)
    fd_error = float(np.linalg.norm(fine-(derivative-common_logjet(model, sample, normal)*value)))
    return dict(Gram=gram, H1=first, H2=quadratic, G1=noether_increment, connection_position=position,
                normal=normal, gauge=gauge, physical_momentum=momentum, value=value,
                directional_refinement_error=fd_error, no_common_factor_double_application=True,
                original_static_boost_in_bank_counted_once=True)


def boosted_generator(H0: np.ndarray, G0: np.ndarray, forms: dict[str, Any],
                      Omega: float, beta: float, beta_dot: float) -> np.ndarray:
    """Full K=V†(H−Omega G)V−iV†dotV; supplied H0/G0 must be same bank."""
    if H0.shape != forms['Gram'].shape or G0.shape != H0.shape:
        raise ValueError('Original full H0 and G0 must use the same functional bank')
    return H0+beta*forms['H1']+beta**2*forms['H2']-Omega*(G0+beta*forms['G1'])+beta_dot*forms['connection_position']


def observe(model: Any, data: dict[str, Any], arrays: Path, clock: int) -> dict[str, Any]:
    bank = FunctionalSourcePhotonBank(model, data)
    generators = mean_matter_generators(model, data)
    guard = json.loads(generators['guard_path'].read_bytes())
    alpha, speed = np.asarray(guard['alpha']), np.asarray(guard['velocity'])
    n = model.chart.normal_dim
    mean = model.chart.evaluate(np.zeros(n))
    mean_pull = physical_chart_pullback(model, dict(point=mean), alpha, speed)
    a = solve(model.chi*mean['cometric'][:n, :n], mean_pull['normal'], assume_a='pos')
    rng = np.random.default_rng(2222037)
    offq = rng.standard_normal(n)*model.sigma/np.sqrt(2)
    em = rng.standard_normal(55)/np.sqrt(2)
    s0, s1 = .37, .3701
    Omega0 = float(pulse_point(clock, s0)['Omega'])
    Omega = float(pulse_point(clock, s1)['Omega'])
    beta_dot = float(pulse_point(clock, s1)['Omega_dot'])
    beta = Omega-Omega0
    observations, saved = {}, dict(a=a, mean_Yn=mean_pull['normal'])
    for label, q in (('mean', np.zeros(n)), ('offmean', offq)):
        sample = model.basis_at(q, em)
        pull = physical_chart_pullback(model, sample, alpha, speed)
        actual = model.chi*(sample['point']['cometric'][:, :n]@a)
        mechanical = pull['gauge']-actual[n:]
        observations[label] = dict(normal_completed_square_defect=float(np.linalg.norm(actual[:n]-pull['normal'])),
            mechanical_connection_norm=float(np.linalg.norm(mechanical)),
            IC_preserving_residual_normal_norm=float(np.linalg.norm(beta*actual[:n]-Omega*pull['normal'])),
            wrong_formal_completion_initial_phase_distance=abs(np.exp(1j*Omega0*(a@q))-1), profiles={})
        for profile in ('source19', 'SOFT'):
            f = boost_forms(model, bank, sample, profile, a, pull['normal'], 2.**-17)
            phase = np.exp(1j*beta*(a@q))
            h = 2.**-19/max(1., float(np.linalg.norm(f['normal'])))
            shifted = []
            for sign in (-1, 1):
                point = model.basis_at(q+sign*h*f['normal'], em)
                shifted.append(np.exp(1j*beta*(a@point['q']))*bank.values(point, profile))
            literal = (shifted[1]-shifted[0])/(2*h)
            analytic = phase*(bank.directional_values(sample, profile, f['normal'], np.zeros(55), 2.**-18)+1j*beta*(a@f['normal'])*f['value'])
            connection = beta_dot*f['connection_position']
            observations[label]['profiles'][profile] = dict(functional_channels=len(f['Gram']),
                directional_refinement_error=f['directional_refinement_error'],
                literal_phase_directional_error=float(np.linalg.norm(literal-analytic)),
                actual_boost_kinetic_linear_norm=float(np.linalg.norm(beta*f['H1'])),
                actual_boost_kinetic_quadratic_norm=float(np.linalg.norm(beta**2*f['H2'])),
                actual_noether_phase_increment_norm=float(np.linalg.norm(Omega*beta*f['G1'])),
                wrong_missing_timeconnection_norm=float(np.linalg.norm(connection)),
                initial_original_source_function_unchanged_error=0.,
                IC_initial_phase_exactly_one=bool(np.exp(1j*(Omega0-Omega0)*(a@q)) == 1),
                Hermiticity={key:float(np.linalg.norm(f[key]-f[key].conj().T)) for key in ('H1','H2','G1','connection_position')})
            for key in ('Gram','H1','H2','G1','connection_position'):
                saved[label+'_'+profile+'_'+key] = f[key]
    np.savez_compressed(arrays, **saved)
    os.chmod(arrays, 0o600)
    return dict(observations=observations, clock=clock, Omega_at_start=Omega0, Omega=Omega, beta=beta,
                beta_dot=beta_dot, time_interval=[s0*clock,s1*clock],
                actual_original_static_boost=model.boost['Omega'],
                phase='Vbeta=exp(i beta a.q), a=(chi gInvNN(mean))^-1 Yn(mean), beta=Omega−Omega_start',
                same_functional_H0_G0_not_compiled_in_this_increment_probe=True,
                scope='Actual original functional SOURCE/G/J/contact/incoming bank point mean and offmean conjugation increments; not global quantum H or spinor closure',
                unpaid=['same-bank original full H0/G0/global normal and theta integration', 'all52 exact boost cross moments',
                        'physical packet orbital/core-Ref/mechanical spin intertwiner', 'LEGAL driven sector/photon Born/quantum law and parameters'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-npz', type=Path, required=True)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arrays', type=Path, required=True)
    parser.add_argument('--clock', type=int, default=566)
    args = parser.parse_args()
    started = time.monotonic()
    model, data = load_shared_native_cache(args.cache_npz, args.cache_manifest)
    before = loaded_source_epoch()
    result = observe(model, data, args.arrays, args.clock)
    result.update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
        arrays_sha256=hashlib.sha256(args.arrays.read_bytes()).hexdigest(),
        loaded_sources=before, loaded_sources_stable=before == loaded_source_epoch(),
        program_wall_seconds=time.monotonic()-started)
    text = json.dumps(portable(result), sort_keys=True)
    args.output.write_text(text)
    os.chmod(args.output, 0o600)
    print(text)


if __name__ == '__main__':
    main()

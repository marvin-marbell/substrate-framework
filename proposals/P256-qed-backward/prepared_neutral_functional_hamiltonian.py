#!/usr/bin/env python3
"""Original whole-H/G and IC-preserving C1 on functional96, exact all52 moments.

Normal/theta quadrature, full-Phi trace and coefficient truncation remain
explicit. This is not an invariant particle sector or a physical spin proof.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve, solve
from functional_source_photon_bank import FunctionalSourcePhotonBank, CoexactOperatorPack
from prepared_neutral_functional_weyl_forms import bilinear, directional_kernels, linear_pack, oscillator_kernels
from prepared_neutral_functional_boost import common_logjet, boosted_generator
from prepared_neutral_shared_photon_state import load_shared_native_cache, loaded_source_epoch
from prepared_neutral_moving_matter_frame import mean_matter_generators, pulse_point
from prepared_neutral_slater_fibre import slater_expectation
from original_total_angular_transfer import scalar_orbital
from native_closed_rotation_quantum import physical_chart_pullback
from original_charged_principal import portable


def derivative_pack(model: Any, bank: FunctionalSourcePhotonBank, sample: dict[str, Any],
                    profile: str, dq: np.ndarray, dtheta: np.ndarray, h: float) -> CoexactOperatorPack:
    eps = h/max(1., float(np.linalg.norm(dq)), float(np.linalg.norm(dtheta)))
    packs = [bank.compile(sample, profile)]
    for sign in (1, -1):
        em = sample['em'].copy()
        em[52:] += sign*eps*dtheta
        shifted = model.basis_at(sample['q']+sign*eps*dq, em)
        packs.append(bank.compile(shifted, profile))
    log = common_logjet(model, sample, dq)-float(sample['em'][52:]@dtheta)
    return linear_pack(tuple(packs), (complex(log), complex(1/(2*eps)), complex(-1/(2*eps))))


def fourier_body(model: Any, pack: CoexactOperatorPack, body: np.ndarray,
                 phase: np.ndarray, sector: str) -> np.ndarray:
    return bilinear(pack, pack, operator=model.fibre.onebody(body, sector), shift=phase)


def light_and_orbital(model: Any, pack: CoexactOperatorPack, sample: dict[str, Any],
                      speed: np.ndarray, sector: str) -> tuple[np.ndarray, np.ndarray]:
    law, point = model.law, sample['point']
    onsite = np.zeros((4*law.volume, 4*law.volume), complex)
    for x in range(law.volume):
        onsite[4*x:4*x+4, 4*x:4*x+4] = law.onsite*point['phi'][x, 0]
    light = fourier_body(model, pack, onsite, np.zeros(52), sector)
    orbital = np.zeros_like(light)
    for x, y, block, path in law.terms(point['phi'], point['links'], sample['a']):
        phase = sum((model.em['legs'][3*site+axis] for site, axis in path), np.zeros(52))
        body = np.zeros_like(onsite)
        body[4*x:4*x+4, 4*y:4*y+4] += block
        light += fourier_body(model, pack, body, phase, sector)
        light += fourier_body(model, pack, body.conj().T, -phase, sector)
    for x in range(law.volume):
        for axis in range(3):
            y = law.target[x, axis]
            edge = (speed[x, axis]+speed[y, axis])*point['links'][x, axis]*np.exp(1j*sample['a'][x, axis])/(4j*law.band.a)
            body = np.zeros_like(onsite)
            body[4*x:4*x+4, 4*y:4*y+4] += edge
            phase = model.em['legs'][3*x+axis]
            orbital += fourier_body(model, pack, body, phase, sector)
            orbital += fourier_body(model, pack, body.conj().T, -phase, sector)
    return light, orbital


def node_forms(model: Any, bank: FunctionalSourcePhotonBank, sample: dict[str, Any],
               profile: str, alpha: np.ndarray, speed: np.ndarray, a: np.ndarray,
               probes: int, seed: int) -> dict[str, Any]:
    sector = 'neutral' if profile == 'neutral' else 'charged'
    pack = bank.compile(sample, profile)
    point = sample['point']
    n = model.chart.normal_dim
    gram = bilinear(pack, pack)
    factor = lu_factor(np.sqrt(model.chart.mass)[:, None]*point['jacobian'])
    rng = np.random.default_rng(seed)
    raw = 2*rng.integers(0, 2, size=(model.chart.raw_dim, probes))-1
    directions = lu_solve(factor, raw)
    phi_levels = []
    h = model.sigma/32
    for step in (h, h/2):
        rows = []
        for direction in directions.T:
            dq, gauge = direction[:n], direction[n:]
            derivative = derivative_pack(model, bank, sample, profile, dq, np.zeros(3), step)
            operators = model.gauge_operators(gauge, sector)
            momentum = linear_pack((pack, derivative), (0j, -1j), operator=operators['first'])
            # Wick physical square BEFORE compressed first-matrix squaring.
            rows.append(model.chi/2*(bilinear(momentum, momentum)+bilinear(pack, pack, operator=operators['outside_second'])))
        phi_levels.append(np.asarray(rows))
    phi = phi_levels[1].mean(axis=0)
    spread = phi_levels[1]-phi
    phi_error = np.sqrt(np.sum(np.abs(spread)**2, axis=0)/(probes*(probes-1)))
    em = bilinear(pack, pack, scalar_kernels=oscillator_kernels(pack, pack, np.asarray(model.em['omega'])))
    harmonic_derivatives = []
    for axis in range(3):
        dtheta = np.eye(3)[axis]
        derivative = derivative_pack(model, bank, sample, profile, np.zeros(n), dtheta, 2.**-18)
        harmonic_derivatives.append(derivative)
        em += model.ue/model.law.size*bilinear(derivative, derivative)
    light, orbital = light_and_orbital(model, pack, sample, speed, sector)
    potential = model.original.classical_potential(point)
    matter = light+potential['massive_sea']*gram
    scalar = sum(potential[k] for k in ('quartic', 'alignment', 'colour_magnetic'))*gram
    coulomb = bilinear(pack, pack, operator=model.HC[sector])
    pull = physical_chart_pullback(model, sample, alpha, speed)
    derivative = derivative_pack(model, bank, sample, profile, pull['normal'], np.zeros(3), 2.**-18)
    strong = bilinear(pack, derivative)
    generator = -.5j*(strong-strong.conj().T)
    generator += bilinear(pack, pack, operator=model.gauge_operators(pull['gauge'], sector)['first'])+orbital
    heavy = scalar_orbital(model.heavy_law, point['links'], np.zeros((model.law.volume, 3)), speed)
    generator += slater_expectation(model.heavy_projector, heavy)*gram
    noether = model.noether_bank.coefficients
    for phase, flow, harmonic, lam in zip(noether['phases'], noether['coexact'], noether['harmonic'], noether['lambda_C']):
        body = np.diag(np.repeat(lam, 4))
        for sign in (-1, 1):
            weight = sign/(2j)
            shift = sign*phase
            generator += weight*(fourier_body(model, pack, body, shift, sector)-2*float(np.sum(lam))*bilinear(pack, pack, shift=shift))
            derivative_form = bilinear(pack, pack, scalar_kernels=directional_kernels(pack, pack, flow, shift))
            for axis, coefficient in enumerate(harmonic):
                if coefficient != 0:
                    derivative_form += coefficient*bilinear(pack, harmonic_derivatives[axis], shift=shift)
            strong_flow = weight*derivative_form
            generator += -.5j*(strong_flow-strong_flow.conj().T)
    velocity = model.chi*(point['cometric'][:, :n]@a)
    derivative = derivative_pack(model, bank, sample, profile, velocity[:n], np.zeros(3), 2.**-18)
    momentum = linear_pack((pack, derivative), (0j, -1j), operator=model.gauge_operators(velocity[n:], sector)['first'])
    cross = bilinear(pack, momentum)
    H1 = (cross+cross.conj().T)/2
    H2 = .5*model.chi*float(a@point['cometric'][:n, :n]@a)*gram
    G1 = float(pull['normal']@a)*gram
    X = float(a@sample['q'])*gram
    return dict(terms=dict(Phi=phi, EM=em, matter=matter, scalar=scalar, Coulomb=coulomb), G=generator,
                Gram=gram, H1=H1, H2=H2, G1=G1, connection_position=X,
                diagnostics=dict(Phi_directional_refinement=float(np.linalg.norm(phi_levels[0].mean(axis=0)-phi)),
                    Phi_stochastic_trace_standard_error=float(np.linalg.norm(phi_error)),
                    full_normal_Gauss_trace_dimension=model.chart.raw_dim,
                    normal_cometric_boost_mismatch=float(np.linalg.norm(velocity[:n]-pull['normal'])),
                    all52_exact_finite_Weyl_before_field_node_integration=True,
                    literal_all55_Gaussian_including_free_harmonics=True,
                    massive_minusLambda_once=True, ordinary_Coulomb_once=True))


def compile_global(model: Any, data: dict[str, Any], nodes: int, probes: int) -> dict[str, Any]:
    if nodes < 2 or nodes%2 or probes < 2:
        raise ValueError('Antithetic fullnormal/theta nodes and at least two Phi trace probes required')
    bank = FunctionalSourcePhotonBank(model, data)
    g = mean_matter_generators(model, data)
    guard = json.loads(g['guard_path'].read_bytes())
    alpha, speed = np.asarray(guard['alpha']), np.asarray(guard['velocity'])
    n = model.chart.normal_dim
    mean = model.chart.evaluate(np.zeros(n))
    pull = physical_chart_pullback(model, dict(point=mean), alpha, speed)
    a = solve(model.chi*mean['cometric'][:n, :n], pull['normal'], assume_a='pos')
    rng = np.random.default_rng(2222030)
    qnodes = rng.standard_normal((nodes//2, n))*model.sigma/math.sqrt(2)
    theta = rng.standard_normal((nodes//2, 3))/math.sqrt(2)
    keys = ('Gram','H0','G0','H1','H2','G1','connection_position')
    forms = {p:{k:np.zeros((96,96),complex) for k in keys} for p in ('source19','SOFT')}
    terms = {p:{k:np.zeros((96,96),complex) for k in ('Phi','EM','matter','scalar','Coulomb')} for p in forms}
    records = []
    for pair, (q, t) in enumerate(zip(qnodes, theta)):
        for sign in (-1, 1):
            sample = model.basis_at(sign*q, np.r_[np.zeros(52), sign*t])
            if sample['cutoff'] <= 0:
                raise ValueError('Original compact support rejected a global Gaussian node')
            weight = sample['importance_weight']/nodes
            for profile in forms:
                result = node_forms(model, bank, sample, profile, alpha, speed, a, probes, 2222038+pair*2+(sign==1))
                for key in result['terms']:
                    terms[profile][key] += weight*result['terms'][key]
                for key in keys:
                    value = sum(result['terms'].values()) if key == 'H0' else result['G' if key == 'G0' else key]
                    forms[profile][key] += weight*value
                records.append(dict(pair=pair, sign=sign, profile=profile, qnorm=float(np.linalg.norm(q)), diagnostics=result['diagnostics']))
    return dict(forms=forms, terms=terms, records=records, qnodes=qnodes, theta_nodes=theta,
                initial={profile:bank.initial_coefficients(profile) for profile in forms}, actual_static_boost=model.boost['Omega'])


def evolve(compiled: dict[str, Any], clock: int, start: float, end: float, steps: int) -> dict[str, Any]:
    initial_Omega = pulse_point(clock, start/clock)['Omega']
    outcomes = {}
    for profile, forms in compiled['forms'].items():
        M = forms['Gram']
        eigenvalues = eigh(M, eigvals_only=True)
        initial = compiled['initial'][profile]
        result = dict(Gram_eigenvalues=eigenvalues, initial_actual_norm_squared=float(np.vdot(initial,M@initial).real),
                      no_state_sourceZ_or_effect_normalization=True,
                      Hermiticity={key:float(np.linalg.norm(value-value.conj().T)) for key,value in forms.items()})
        if eigenvalues[0] <= 0:
            result.update(coefficient_evolution_admitted=False, failure='Actual quadrature metric is not positive; no jitter/clipping/profile reset')
            outcomes[profile] = result
            continue
        def run(count: int, omit: str | None = None) -> np.ndarray:
            state = initial.copy()
            dt = (end-start)/count
            H0 = forms['H0'] if omit != 'Phi' else forms['H0']-compiled['terms'][profile]['Phi']
            for index in range(count):
                pulse = pulse_point(clock, (start+(index+.5)*dt)/clock)
                beta = pulse['Omega']-initial_Omega
                K = boosted_generator(H0, forms['G0'], forms, pulse['Omega'], beta,
                                      0. if omit == 'connection' else pulse['Omega_dot'])
                energies, modes = eigh((K+K.conj().T)/2, M)
                state = modes@(np.exp(-1j*dt*energies)*(modes.conj().T@(M@state)))
            return state
        coarse, final = run(steps//2), run(steps)
        wrong_phi, wrong_connection = run(steps, 'Phi'), run(steps, 'connection')
        def norm(v: np.ndarray) -> float:
            return float(np.vdot(v,M@v).real)
        result.update(coefficient_evolution_admitted=True, final_coefficients=final,
            actual_final_norm_squared=norm(final), temporal_refinement_norm_squared=norm(final-coarse),
            wrong_missing_Phi_difference_squared=norm(final-wrong_phi),
            wrong_missing_timeconnection_difference_squared=norm(final-wrong_connection),
            outside_original_incoming_SOURCE_ray_squared=norm(final)-abs((M@final)[48])**2/M[48,48].real)
        outcomes[profile] = result
    return outcomes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-npz','cache-manifest','output','arrays'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--nodes', type=int, default=4)
    p.add_argument('--trace-probes', type=int, default=4)
    p.add_argument('--clock', type=int, default=566)
    p.add_argument('--time-steps', type=int, default=128)
    args = p.parse_args()
    if args.time_steps < 2 or args.time_steps%2:
        raise ValueError('Even time-step count required for refinement')
    began = time.monotonic()
    model, data = load_shared_native_cache(args.cache_npz, args.cache_manifest)
    before = loaded_source_epoch()
    compiled = compile_global(model, data, args.nodes, args.trace_probes)
    t0, t1 = .37*args.clock, .37*args.clock+.031
    evolution = evolve(compiled, args.clock, t0, t1, args.time_steps)
    arrays = dict(qnodes=compiled['qnodes'], theta_nodes=compiled['theta_nodes'])
    for profile in compiled['forms']:
        arrays.update({profile+'_'+key:value for key,value in compiled['forms'][profile].items()})
        arrays.update({profile+'_term_'+key:value for key,value in compiled['terms'][profile].items()})
    np.savez_compressed(args.arrays, **arrays)
    os.chmod(args.arrays, 0o600)
    result = dict(scope='NEW whole original-H/G weak functional96 incoming-photon coefficient evolution; finite normal/theta/trace/sector approximation, NOT full physical particle/spin/scattering',
        evolution=evolution, records=compiled['records'], nodes=args.nodes, trace_probes=args.trace_probes,
        actual_static_boost=compiled['actual_static_boost'], time_interval=[t0,t1], clock=args.clock,
        beta='Omega−Omega_at_start retains unchanged physical IC and −Omega_start DY leftover',
        all52_exact_finite_Weyl_and_full_creator_moments=True, all55_original_EM_including_free_harmonic=True,
        arrays_sha256=hashlib.sha256(args.arrays.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
        loaded_sources=before, loaded_sources_stable=before==loaded_source_epoch(),
        program_wall_seconds=time.monotonic()-began,
        unpaid=['normal/theta quadrature and fullPhi trace error control', 'full outside-H²/G/J/current sector and ROOT/Gauss/legal graph',
                'physical mechanical/core/Ref/packetL half-spin', 'unchanged physical receiver contractivity and Born transport', 'quantum law/parameters'])
    text = json.dumps(portable(result), sort_keys=True)
    args.output.write_text(text)
    os.chmod(args.output, 0o600)
    print(text)


if __name__ == '__main__':
    main()

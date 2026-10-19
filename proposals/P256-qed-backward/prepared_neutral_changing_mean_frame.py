#!/usr/bin/env python3
"""Finite-step changing-quintic full mean-matter interval; not a full field flow."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
import numpy as np
from scipy.linalg import expm
from original_charged_principal import portable
from prepared_neutral_moving_matter_frame import (
    PLAN, REVISION, WORK_ITEM, array_digest, mean_matter_generators, pulse_point)
from prepared_neutral_shared_photon_state import load_shared_native_cache, loaded_source_epoch


def _bodies(generators: dict[str, Any], Lambda: float, omega: float) -> dict[str, np.ndarray]:
    return dict(light=generators['light_H']-omega*generators['light'],
                heavy=-Lambda*generators['heavy_H']-omega*generators['heavy'],
                reference=-omega*generators['reference'])


def propagate_bodies(body_provider: Callable[[float], dict[str, np.ndarray]],
                     t0: float, t1: float, steps: int = 8) -> dict[str, Any]:
    """CF4 for actual callback A(t); full matrices, later factors on LEFT.

    U(t0)=I. No state preparation, ground computation, phase repair or prefix.
    The provider supplies Hermitian light108/heavy108/reference4 bodies.
    """
    if not isinstance(steps, int) or steps < 1 or not t0 < t1:
        raise ValueError('Positive integer steps and increasing interval required')
    dt = (t1-t0)/steps
    c1, c2 = .5-math.sqrt(3)/6, .5+math.sqrt(3)/6
    a1, a2 = (3-2*math.sqrt(3))/12, (3+2*math.sqrt(3))/12
    initial = body_provider(t0)
    result = {name: np.eye(len(initial[name]), dtype=complex)
              for name in ('light', 'heavy', 'reference')}
    for step in range(steps):
        start = t0+step*dt
        first = body_provider(start+c1*dt)
        second = body_provider(start+c2*dt)
        for name in result:
            early = expm(-1j*dt*(a2*first[name]+a1*second[name]))
            late = expm(-1j*dt*(a1*first[name]+a2*second[name]))
            result[name] = late@early@result[name]
    endpoint = body_provider(t1)
    return dict(unitaries=result, endpoint_bodies=endpoint,
                connection={name: -u.conj().T@endpoint[name]@u for name, u in result.items()},
                t0=t0, t1=t1, steps=steps,
                method='CF4 two-exponential Gauss time ordering; finite-step approximation')


def evolve_mean_matter(generators: dict[str, Any], Lambda: float, clock: int,
                       t0: float, t1: float, steps: int = 8) -> dict[str, Any]:
    """Udot=-i A(t)U, U(t0)=I; no missing prefix is claimed evolved.

    Returns the requested steps approximation, not the finest control. All
    heavy phases remain present; neither phase fitting nor polar repair occurs.
    """
    if not isinstance(steps, int) or steps < 1 or not 0 < t0 < t1 < clock:
        raise ValueError('Positive integer steps and interior increasing interval required')
    pulses = dict(start=pulse_point(clock, t0/clock), end=pulse_point(clock, t1/clock))
    levels = (steps, 2*steps, 4*steps)
    def body_provider(t: float) -> dict[str, np.ndarray]:
        return _bodies(generators, Lambda, pulse_point(clock, t/clock)['Omega'])
    propagated = {n: propagate_bodies(body_provider, t0, t1, n) for n in levels}
    evolved = {n: value['unitaries'] for n, value in propagated.items()}
    unitaries = evolved[steps]
    initial = body_provider(t0)
    endpoint = propagated[steps]['endpoint_bodies']
    connection = propagated[steps]['connection']
    constant_cf4 = propagate_bodies(lambda _: initial, t0, t1, steps)['unitaries']
    metrics: dict[str, Any] = {}
    refinements: dict[str, Any] = {}
    for name, u in unitaries.items():
        a0 = initial[name]
        frozen = expm(-1j*(t1-t0)*a0)
        # Constant-body subdivision has NO time-ordering defect. Its observed
        # discrepancy isolates exponential/multiplication large-phase numerics.
        constant_step = expm(-1j*(t1-t0)/steps*a0)
        constant_subdivided = np.linalg.matrix_power(constant_step, steps)
        d1 = float(np.linalg.norm(evolved[steps][name]-evolved[2*steps][name]))
        d2 = float(np.linalg.norm(evolved[2*steps][name]-evolved[4*steps][name]))
        refinements[name] = dict(steps=list(levels), coarse_to_double_frobenius=d1,
                                double_to_quadruple_frobenius=d2,
                                ratio=d1/d2 if d2 else None,
                                fourth_order_ratio16_not_asserted=True)
        wrong_connection = u.conj().T@endpoint[name]@u-endpoint[name]
        metrics[name] = dict(
            full_dimension=len(u),
            unitarity_frobenius=float(np.linalg.norm(u.conj().T@u-np.eye(len(u)))),
            endpoint_pullback_cancellation=float(np.linalg.norm(u.conj().T@endpoint[name]@u+connection[name])),
            wrong_frozen_Omega_start_frobenius=float(np.linalg.norm(u-frozen)),
            wrong_endpoint_connection_frobenius=float(np.linalg.norm(wrong_connection)),
            wrong_endpoint_connection_relative=float(np.linalg.norm(wrong_connection)/(1+np.linalg.norm(endpoint[name]))),
            endpoint_A_commutator_frobenius=float(np.linalg.norm(a0@endpoint[name]-endpoint[name]@a0)),
            constant_body_subdivision_numeric_frobenius=float(np.linalg.norm(constant_subdivided-frozen)),
            constant_body_CF4_numeric_frobenius=float(np.linalg.norm(constant_cf4[name]-frozen)),
            constant_body_exponent_one_norm=float(np.linalg.norm((t1-t0)*a0, 1)),
            absolute_matrix_differences_without_phase_alignment=True)
    ref_exact = expm(1j*(pulses['end']['theta']-pulses['start']['theta'])*generators['reference'])
    metrics['reference']['commuting_exact_theta_error'] = float(np.linalg.norm(unitaries['reference']-ref_exact))
    return dict(unitaries=unitaries, endpoint_bodies=endpoint, connection=connection,
                pulse_endpoints=pulses, refinements=refinements, control_metrics=metrics,
                steps=steps, t0=t0, t1=t1, method='CF4 two-exponential Gauss time ordering',
                initial_condition='U(t0)=I interval propagator on once-prepared boundary input; prefix NOT evolved',
                scope='Finite-step frozen-MEAN-H changing-pulse matter approximation; commonfield NOT included; NOT whole trajectory',
                heavy_numerical_scope='Constant-body subdivision exposes large-phase error separately; refinement is not assumed pure time-ordering error')


def native_smoke(model: Any, data: dict[str, Any], evolution: dict[str, Any]) -> dict[str, Any]:
    """Exercise existing full orbit transport without recapturing any bank."""
    u = evolution['unitaries']
    before = {key: array_digest(data[key]) for key in
              ('neutral_H', 'charged_H', 'neutral_Gram', 'charged_Gram', 'neutral_state',
               'preparation_state14', 'source19_map', 'SOFT_map')}
    held = array_digest(model.source_bank.seed_fibre.orbitals)
    prepared = array_digest(model.prepared14)
    moved = model.in_mean_interaction_frame(u['light'], u['reference'], u['heavy'])
    light = {str(q): dict(occupied_count=model.fibre.occupied[q].shape[1],
                         transport_error=float(np.linalg.norm(moved.fibre.occupied[q]-u['light']@model.fibre.occupied[q])),
                         projector_error=float(np.linalg.norm(moved.fibre.projectors[q]-u['light']@model.fibre.projectors[q]@u['light'].conj().T)))
             for q in (1, -1)}
    heavy_error = float(np.linalg.norm(moved.heavy_orbitals-u['heavy']@model.heavy_orbitals))
    reference_error = float(np.linalg.norm(moved.reference_frame-u['reference']@model.reference_frame))
    stable = before == {key: array_digest(data[key]) for key in before}
    bank_held = moved.source_bank is model.source_bank and array_digest(moved.source_bank.seed_fibre.orbitals) == held
    prepared_held = array_digest(moved.prepared14) == prepared
    if model.heavy_orbitals.shape != (108, 54) or not stable or not bank_held or not prepared_held:
        raise AssertionError('Once-prepared heavy54/sourcebank/state boundary changed')
    if heavy_error > 1e-12 or reference_error > 1e-12 or any(x['transport_error'] > 1e-12 for x in light.values()):
        raise AssertionError('Full matter transport failed')
    controls = evolution['control_metrics']
    if evolution['pulse_endpoints']['start']['Omega'] == evolution['pulse_endpoints']['end']['Omega']:
        raise AssertionError('Selected scientific pulse does not change')
    if controls['light']['wrong_frozen_Omega_start_frobenius'] <= 1e-12 or controls['light']['wrong_endpoint_connection_frobenius'] <= 1e-12:
        raise AssertionError('Wrong frozen-Omega/endpoint-connection controls do not discriminate')
    return dict(light_sectors=light, heavy_occupied_shape=list(model.heavy_orbitals.shape),
                heavy54_transport_error=heavy_error, reference4_transport_error=reference_error,
                sourcebank_identity_held=bank_held, sourcebank_seed_orbitals_sha256=held,
                prepared14_held=prepared_held, prepared14_sha256=prepared,
                compiled_H_G_prepared_SOURCE_maps_unchanged=stable, input_array_hashes=before,
                no_new_ground_eigh_filter_bank_recapture_spinfit_or_lawretune=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-npz', type=Path, required=True)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--matrices-output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--clock', type=int)
    parser.add_argument('--pulse-fraction', type=float, default=.37)
    parser.add_argument('--dt', type=float, default=.031)
    args = parser.parse_args()
    started = time.perf_counter()
    before = loaded_source_epoch()
    model, data = load_shared_native_cache(args.cache_npz, args.cache_manifest)
    generators = mean_matter_generators(model, data)
    clock = args.clock if args.clock is not None else math.ceil(15*math.pi*1.5/model.c)
    t0 = args.pulse_fraction*clock
    evolution = evolve_mean_matter(generators, model.Lambda, clock, t0, t0+args.dt, args.steps)
    smoke = native_smoke(model, data, evolution)
    after = loaded_source_epoch()
    changed = {name: dict(before=digest, after=after.get(name)) for name, digest in before.items() if after.get(name) != digest}
    if changed:
        raise AssertionError('Imported source changed during science: '+repr(changed))
    matrices = {kind+'_'+name: value for kind in ('unitaries', 'endpoint_bodies', 'connection')
                for name, value in evolution[kind].items()}
    np.savez_compressed(args.matrices_output, **matrices)
    summary = {key: value for key, value in evolution.items() if key not in ('unitaries', 'endpoint_bodies', 'connection')}
    result = dict(plan=PLAN, revision=REVISION, work_item=WORK_ITEM, argv=sys.argv,
                  cwd=str(Path.cwd()), python=sys.executable, python_version=sys.version,
                  natural_exit=0, wall_seconds=time.perf_counter()-started,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cache_npz_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),
                  cache_manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
                  guard_sha256=generators['guard_sha256'], native_Lambda=model.Lambda,
                  evolution=summary, native_smoke=smoke,
                  matrix_archive=str(args.matrices_output), matrix_archive_sha256=hashlib.sha256(args.matrices_output.read_bytes()).hexdigest(),
                  full_matrix_hashes={key: array_digest(value) for key, value in matrices.items()},
                  loaded_sources_before=before, loaded_sources_after=after, changed_sources=changed,
                  no_H_G_or_preparation_recompile=True,
                  unpaid=['Commonfield/field timeconnection/commonenergy and field-dependent H are not included in this matter slice',
                          'Prefix from preparation to t0 and physical full coupled Xi/matched BXi trajectory',
                          'Finite-step time-ordering error and independently visible heavy large-phase numerics',
                          'Full Root/all512/Gauss/PH/multipair/legal-N closure and physical closed-loop/Born/quantum-law debts'])
    args.output.write_text(json.dumps(portable(result), indent=2, sort_keys=True))
    print(json.dumps(portable(result), sort_keys=True))


if __name__ == '__main__':
    main()

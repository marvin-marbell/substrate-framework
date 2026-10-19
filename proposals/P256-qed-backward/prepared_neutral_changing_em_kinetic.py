#!/usr/bin/env python3
"""Original all55 EM positive form on one coupled characteristic family."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
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
from original_total_angular_transfer import scalar_orbital
from original_charged_principal import portable
from prepared_neutral_changing_em_characteristic import evolve_em, CACHE_SHA, MANIFEST_SHA
from prepared_neutral_changing_mean_frame import propagate_bodies
from prepared_neutral_changing_paired_pulse import coupled_interval, source_point
from prepared_neutral_moving_matter_frame import pulse_point
from prepared_neutral_shared_photon_state import load_shared_native_cache

PLAN = 'issue-222-fullfield-closedrotation-continuation-20261006'
REVISION = '668c7d8ea2ec9c71c964e9f16d3c75e8cd879e10433e72fbcad0c3b89c7da417'


def varied_member(model: Any, data: dict[str, Any], frame: dict[str, Any],
                  em: np.ndarray) -> dict[str, Any]:
    """Vary INITIAL em only; actual RAW Phi, heavy/ref and scalar stay held."""
    t0, t1 = frame['time_interval']; steps = frame['steps']; clock = frame['clock']
    records: dict[float, dict[str, Any]] = {}
    def provider(t: float) -> dict[str, np.ndarray]:
        base = frame['native_provider'](t)
        if t not in records:
            flow = None if t == t0 else evolve_em(model.noether_bank, clock, t0, t, em, steps)
            ee = em if flow is None else flow['em_endpoint']
            a, _ = model.original.electromagnetic_field(ee)
            noether = model.noether_bank.evaluate(ee[:52]); raw = base['raw']
            light = model.law.onebody(raw['phi_endpoint'], raw['links_endpoint'], a)
            orbital = scalar_orbital(model.law, raw['links_endpoint'], a, frame['velocity'])
            light -= pulse_point(clock, t/clock)['Omega'] * (orbital + np.diag(np.repeat(noether['lambda_C'], 4)))
            records[t] = dict(light=light, heavy=base['bodies']['heavy'], reference=base['bodies']['reference'])
        return records[t]
    matter = propagate_bodies(provider, t0, t1, steps)
    flow = evolve_em(model.noether_bank, clock, t0, t1, em, steps)
    # Keep the exact BASE common physical state phase, including Phi scalar.
    relative = math.exp(flow['integrated_log_half_density'] - frame['em_orbit']['integrated_log_half_density'])
    relative *= math.exp(-float(em@em-frame['initial_em']@frame['initial_em'])/2)
    background = np.exp(2j*np.sum(flow['integrated_anchored_lambda_C']))
    common = frame['common_halfdensity'] * frame['scalar_phase'] * background * relative
    sample = model.basis_at(frame['initial_q'], em, frame['initial_sample']['point'])
    vectors = {'neutral': common*sample['neutral']@data['neutral_state']}
    for profile in ('source19', 'SOFT'):
        vectors[profile] = common*sample['charged']@data[profile+'_normalized_source_state']
    return dict(light=matter['unitaries']['light'], vectors=vectors, flow=flow,
                heavy_reuse_error=float(np.linalg.norm(matter['unitaries']['heavy']-frame['raw_unitaries']['heavy'])),
                reference_reuse_error=float(np.linalg.norm(matter['unitaries']['reference']-frame['raw_unitaries']['reference'])))


def wick_form(fibre: Any, v: np.ndarray, dv: np.ndarray, k: np.ndarray,
              sector: str) -> dict[str, float]:
    """||dv+i dGamma(K)v||^2, full Wick second moment BEFORE projection."""
    first = fibre.onebody(k, sector)
    projected = dv+1j*first@v
    full_second = fibre.bilinear_product(k.conj().T, k, sector)
    ph = float(np.vdot(v, (full_second-first.conj().T@first)@v).real)
    projected_squared = float(np.vdot(projected, projected).real)
    return dict(full=projected_squared+ph, projected_first=projected_squared,
                particle_hole=ph, scalar_derivative=float(np.vdot(dv,dv).real),
                matter_second=float(np.vdot(v,full_second@v).real),
                interference=float(2*np.vdot(dv,1j*first@v).real))


def native(model: Any, data: dict[str, Any], steps: int, fd: float) -> dict[str, Any]:
    q0, em0, _ = source_point(model, data)
    clock = math.ceil(15*math.pi*1.5/model.c); t0 = .37*clock; t1 = t0+.031
    frame = coupled_interval(model, data, q0, em0, clock, t0, t1, steps)
    initial = frame['initial_sample']; common = frame['common_halfdensity']*frame['background_phase']*frame['scalar_phase']
    vectors = {'neutral': common*initial['neutral']@data['neutral_state']}
    for profile in ('source19','SOFT'):
        vectors[profile] = common*initial['charged']@data[profile+'_normalized_source_state']
    u = frame['raw_unitaries']['light']; jac = frame['em_orbit']['initial_coordinate_jacobian']; inv = np.linalg.inv(jac)
    weights = np.r_[np.asarray(model.em['omega'])/2, np.full(3,model.ue/model.law.size)]
    results: list[dict[str, Any]] = []; derivatives: list[dict[str, Any]] = []
    max_heavy = 0.; max_ref = 0.
    for h in (fd, fd/2):
        dk = np.empty((55,108,108),complex)
        dv = {name: np.empty((55,len(v)),complex) for name,v in vectors.items()}
        half_gradient = np.empty(55); jac_fd = np.empty((55,55)); anti = []
        for j in range(55):
            shift = np.zeros(55); shift[j] = h
            plus = varied_member(model,data,frame,em0+shift)
            minus = varied_member(model,data,frame,em0-shift)
            dk[j] = -1j*u.conj().T@(plus['light']-minus['light'])/(2*h)
            anti.append(float(np.linalg.norm(dk[j]-dk[j].conj().T)))
            for name in vectors:
                dv[name][j] = (plus['vectors'][name]-minus['vectors'][name])/(2*h)
            half_gradient[j] = (plus['flow']['integrated_log_half_density']-minus['flow']['integrated_log_half_density'])/(2*h)
            jac_fd[:,j] = (plus['flow']['em_endpoint']-minus['flow']['em_endpoint'])/(2*h)
            max_heavy = max(max_heavy,plus['heavy_reuse_error'],minus['heavy_reuse_error'])
            max_ref = max(max_ref,plus['reference_reuse_error'],minus['reference_reuse_error'])
        final_k = np.einsum('ji,jab->iab',inv,dk)
        final_dv = {name: inv.T@value for name,value in dv.items()}
        rows = {}
        for name,v in vectors.items():
            sector = 'neutral' if name == 'neutral' else 'charged'
            axes = [wick_form(model.fibre,v,final_dv[name][j],final_k[j],sector) for j in range(55)]
            sums = {key: float(weights@np.asarray([a[key] for a in axes])) for key in axes[0]}
            n = float(np.vdot(v,v).real)
            potential = float(np.sum(model.em['omega']*frame['physical_sample']['em'][:52]**2/2))*n
            rows[name] = dict(kinetic=sums,axes=axes,pointwise_norm_squared=n,
                              original_magnetic_potential=potential,
                              original_EM_form=sums['full']+potential,
                              harmonic_kinetic=float(weights[52:]@np.asarray([a['full'] for a in axes[52:]])))
        results.append(dict(fd_step=h,rows=rows,all55_nonHermitian_residuals=anti,
                            maximum_nonHermitian_residual=max(anti),
                            full55_jacobian_FD_error=float(np.linalg.norm(jac_fd-jac)),
                            log_halfdensity_gradient_FD_error=float(np.linalg.norm(half_gradient-frame['em_orbit']['initial_log_half_density_gradient']))))
        derivatives.append(dict(k=final_k,dv=final_dv))
    refinement = {}
    for name in vectors:
        coarse = results[0]['rows'][name]['kinetic']; fine = results[1]['rows'][name]['kinetic']
        error = abs(coarse['full']-fine['full']); signal = fine['particle_hole']
        refinement[name] = dict(kinetic_absolute_difference=error,
            kinetic_relative_difference=error/abs(fine['full']) if fine['full'] else None,
            PH_absolute_difference=abs(coarse['particle_hole']-signal),
            projection_signal_resolves_full_FD_change=signal>error,
            full_coefficient_derivative_difference=float(np.linalg.norm(derivatives[0]['dv'][name]-derivatives[1]['dv'][name])))
    return dict(scope='Original quantum all55 EM/harmonic positive form on SAME short paired characteristic, pointwise factored density; NOT quantum propagation',
        clock=clock,time_interval=[t0,t1],steps=steps,initial_em=em0,final_em=frame['physical_sample']['em'],
        full55_weights=weights,FD=results,refinement=refinement,
        full108_generator_refinement_frobenius=float(np.linalg.norm(derivatives[0]['k']-derivatives[1]['k'])),
        full55_jacobian_condition=float(np.linalg.cond(jac)),inverse_jacobian_residual=float(np.linalg.norm(jac@inv-np.eye(55))),
        common_halfdensity_log=frame['common_halfdensity_log'],scalar_phase=frame['scalar_phase'],background_phase=frame['background_phase'],
        original_all52_zero_point_once=model.photon_basis.zero_point,
        zero_point_scope='Already in the original differential kinetic+potential form, not added again; given physical reference only, no subtraction',
        normalization='Only constant BASE Gaussian/Haar factored convention; fixed globally normalized 214/1728 input coefficients, no pointwise norm division',
        density_derivative_exactness='RAW Phi/Haar, scalar Phi phase and heavy/ref are exactly em-independent in this parameter family; EM log-half gradient is analytic variational RK4, coefficients and full108 U are symmetric FD',
        heavy_independent_reuse_error=max_heavy,reference_independent_reuse_error=max_ref,
        source_bank_held=frame['source_bank_held'],neutral_RIGHT='Both neutral filled seas retained by full Wick product; charged particle AND hole with fixed Ref3/Ref1',
        no_profile_or_amplitude_reset=True,no_phase_fit_or_polar_repair=True,
        debts=['Finite FD/CF4/RK4 remainder, not a certified global bound','Pointwise common form, not integrated globalL2 energy','Phi kinetic, Coulomb multipair/fullH propagation not asserted','No wholepulse/physical2pi/Born claim'])


def source_epoch() -> dict[str, bytes]:
    root = Path(__file__).parent
    return {Path(cast(str,m.__file__)).name:Path(cast(str,m.__file__)).read_bytes()
            for m in tuple(sys.modules.values()) if getattr(m,'__file__',None)
            and Path(cast(str,m.__file__)).parent == root}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-npz',type=Path,required=True)
    p.add_argument('--cache-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-archive',type=Path,required=True)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--fd-step',type=float,default=2.**-13)
    args = p.parse_args(); started = time.perf_counter()
    assert hashlib.sha256(args.cache_npz.read_bytes()).hexdigest() == CACHE_SHA
    assert hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest() == MANIFEST_SHA
    model,data = load_shared_native_cache(args.cache_npz,args.cache_manifest)
    before = source_epoch(); science = native(model,data,args.steps,args.fd_step); after = source_epoch()
    assert all(after.get(name)==body for name,body in before.items())
    with zipfile.ZipFile(args.source_archive,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,body in sorted(after.items()): archive.writestr(name,body)
    result = dict(plan=PLAN,assignment_revision=REVISION,work_item='prepared-neutral-full-field-outcome',
        science=science,argv=sys.argv,cwd=os.getcwd(),python=sys.executable,
        wall_seconds=time.perf_counter()-started,cache_sha256=CACHE_SHA,manifest_sha256=MANIFEST_SHA,
        source_sha256={name:hashlib.sha256(body).hexdigest() for name,body in after.items()},
        source_archive_sha256=hashlib.sha256(args.source_archive.read_bytes()).hexdigest(),stable_loaded_sources=True,
        no_H_G_preparation_or_old_loop_rerun=True)
    text = json.dumps(portable(result),sort_keys=True); args.output.write_text(text+'\n'); print(text)


if __name__ == '__main__': main()

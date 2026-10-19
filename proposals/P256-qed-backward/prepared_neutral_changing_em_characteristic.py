#!/usr/bin/env python3
"""Changing-quintic physicalG characteristic; NOT quantum H_EM evolution."""
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
from original_total_angular_transfer import primitive_flow
from original_charged_principal import portable
from prepared_neutral_moving_matter_frame import pulse_point, PLAN, REVISION, WORK_ITEM
from prepared_neutral_shared_photon_state import load_shared_native_cache

CACHE_SHA = '10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3'
MANIFEST_SHA = '90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a'


def provider_jet(bank: Any, em: np.ndarray) -> dict[str, Any]:
    """Literal provider value and analytic FULL55 jet, including harmonic rows."""
    n = bank.dimension
    value = bank.evaluate(em[:n])
    c = bank.coefficients
    angle = c['phases'] @ em[:n]
    coefficients = np.column_stack((c['coexact'], c['harmonic']))
    jac = np.zeros((n+3, n+3))
    jac[:, :n] = (coefficients.T * np.cos(angle)) @ c['phases']
    dots = np.einsum('ri,ri->r', c['coexact'], c['phases'])
    ddiv = np.r_[-(np.sin(angle)*dots) @ c['phases'], np.zeros(3)]
    return dict(value=value, vector=np.r_[value['flow'], value['harmonic']],
                jacobian=jac, divergence_gradient=ddiv)


def _integrate(bank: Any, clock: int, t0: float, t1: float, em0: np.ndarray,
               steps: int, frozen_omega: float | None = None,
               meanflow: bool = False) -> dict[str, Any]:
    n = bank.dimension+3
    if em0.shape != (n,) or not np.all(np.isfinite(em0)):
        raise ValueError('All finite coexact AND three harmonic coordinates required')
    if clock <= 0 or not 0 < t0 < t1 < clock or steps < 1:
        raise ValueError('Positive steps and interior forward pulse interval required')
    v = bank.law.volume
    # State, full variational map, log-half-density and its initial gradient,
    # anchored lambda integral, harmonic integral. No source/sea recapture.
    state = np.r_[em0, np.eye(n).ravel(), 0., np.zeros(n), np.zeros(v), np.zeros(3)]
    jend = n+n*n
    lend = jend+1+n
    def rhs(t: float, z: np.ndarray) -> np.ndarray:
        jet = provider_jet(bank, z[:n])
        p = jet['value']
        omega = pulse_point(clock, t/clock)['Omega'] if frozen_omega is None else frozen_omega
        jac = z[n:jend].reshape(n,n)
        vec = jet['vector']
        derivative = jet['jacobian']
        div, ddiv = p['divergence'], jet['divergence_gradient']
        if meanflow:
            vec, derivative, div, ddiv = np.zeros(n), np.zeros((n,n)), 0., np.zeros(n)
        return np.r_[-omega*vec, (-omega*derivative@jac).ravel(),
                     .5*omega*div, .5*omega*ddiv@jac,
                     -omega*p['lambda_C'], -omega*p['harmonic']]
    h = (t1-t0)/steps
    times = [t0]
    points = [em0.copy()]
    providers = [bank.evaluate(em0[:bank.dimension])]
    for k in range(steps):
        t = t0+k*h
        a = rhs(t,state)
        b = rhs(t+h/2,state+h*a/2)
        c = rhs(t+h/2,state+h*b/2)
        d = rhs(t+h,state+h*c)
        state += h*(a+2*b+2*c+d)/6
        times.append(t+h)
        points.append(state[:n].copy())
        providers.append(bank.evaluate(state[:bank.dimension]))
    endpoint = state[:n]
    jac = state[n:jend].reshape(n,n)
    loghalf = float(state[jend])
    logfactor = float((endpoint[:bank.dimension]@endpoint[:bank.dimension]
                     -em0[:bank.dimension]@em0[:bank.dimension])/2+loghalf)
    dlog = endpoint[:bank.dimension]@jac[:bank.dimension]-np.r_[em0[:bank.dimension],np.zeros(3)] + state[jend+1:lend]
    return dict(em_endpoint=endpoint, integrated_log_half_density=loghalf,
                gaussian_ratio=math.exp(logfactor-loghalf), half_density=math.exp(loghalf),
                factor=math.exp(logfactor), factor_log=logfactor,
                initial_coordinate_jacobian=jac, initial_log_half_density_gradient=state[jend+1:lend],
                initial_log_factor_gradient=dlog,
                integrated_anchored_lambda_C=state[lend:lend+v],
                integrated_harmonic=state[lend+v:],
                times=np.asarray(times), em_path=np.asarray(points), provider_path=providers,
                clock=clock,t0=t0,t1=t1,steps=steps,
                factor_scope='Along negative-Omega characteristic, Gaussian-divided coefficient at moving endpoint',
                physics_scope='physicalG only; NOT H_EM or harmonic-kinetic quantum propagation')


def evolve_em(bank: Any, clock: int, t0: float, t1: float,
              em0: np.ndarray, steps: int = 8) -> dict[str, Any]:
    """Hdrive=-Omega G: emdot=-Omega V, log-half-density'=+Omega divV/2.

    Returns the forward characteristic and full55 initial-coordinate Jacobian.
    If psi1(z1)=half_density*psi0(z0), its Gaussian-divided coefficient
    receives factor=exp((|y1|²-|y0|²)/2)*half_density. To evaluate on a fixed
    final chart use the inverse characteristic, NOT f0 at this forward endpoint.
    """
    return _integrate(bank,clock,t0,t1,np.asarray(em0,float),steps)


def relative_error(a: Any, b: Any) -> float:
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(a),np.linalg.norm(b),np.finfo(float).tiny))


def provider_control(model: Any, bank: Any, em: np.ndarray, guard: dict[str,Any]) -> dict[str,Any]:
    """Independent native primitive, Gauss projection, gamma-Weyl and FULL FD."""
    y = em[:52]
    original = model.original
    q = np.zeros(model.chart.normal_dim)
    q[:3] = np.asarray([.12,-.21,.17])*model.sigma
    sample = original.basis_at(q,em)
    p = bank.evaluate(y)
    raw = primitive_flow(cast(Any, model.law),sample['point']['phi'],sample['point']['links'],
                         sample['a'], bank.velocity,np.asarray(guard['alpha']))[2]
    coexact = model.em['vectors'].T@raw.ravel()*np.sqrt(model.em['omega']/(2*model.ue))
    harmonic = model.law.size*raw.mean(axis=0)
    lam = -model.green@bank.gradient.T@raw.ravel()
    lam -= lam[model.reference_site]
    h = 2.**-16
    fd = np.column_stack([(provider_jet(bank,em+h*e)['vector']-provider_jet(bank,em-h*e)['vector'])/(2*h) for e in np.eye(55)])
    divfd = float(np.trace(fd))
    gamma_wave,_ = bank.gaussian_functions()[15].evaluate(y)
    gaussian = math.pi**(-26/2)*math.exp(-float(y@y)/2)
    direct_gamma = 1j*(coexact@y-divfd/2)
    errors = dict(raw_native=relative_error(raw,p['raw']),coexact_native=relative_error(coexact,p['flow']),
                  harmonic_native=relative_error(harmonic,p['harmonic']),lambda_native=relative_error(lam,p['lambda_C']),
                  full55_jacobian_FD=relative_error(fd,provider_jet(bank,em)['jacobian']),
                  divergence_FD=abs(divfd-p['divergence'])/max(np.linalg.norm(fd),np.finfo(float).tiny),
                  gamma_native_FD=abs(direct_gamma-p['gamma'])/max(abs(p['gamma']),np.linalg.norm(fd),np.finfo(float).tiny),
                  gamma_independent_Weyl=abs(gamma_wave/gaussian-p['gamma'])/max(abs(p['gamma']),np.finfo(float).tiny))
    assert max(errors.values()) < 3e-8, errors
    assert np.linalg.norm(p['harmonic']) > 0 and np.linalg.norm(p['lambda_C']) > 0
    return dict(errors=errors,all52=True,full55_FD=True,unique_native_curls=len(bank.edge_curls),
                coexact_norm=float(np.linalg.norm(p['flow'])),harmonic_norm=float(np.linalg.norm(p['harmonic'])),
                anchored_lambda_norm=float(np.linalg.norm(p['lambda_C'])),divergence=p['divergence'],gamma=p['gamma'])


def product_control(model: Any, bank: Any, em: np.ndarray, flow: dict[str,Any]) -> dict[str,Any]:
    """Actual native harmonic SOURCE derivative, paired arbitrary wave product.

    First-order identity for any differentiable B and psi is L(B psi)-B Lpsi
    =(V.dB)psi. Actual native full-CAR source matrices are used, not a Fock bank.
    """
    # Use the already frozen physical chart point, without a field-bank recapture.
    point = model.original.basis_at(np.zeros(model.chart.normal_dim),em)['point']
    phi,links = point['phi'],point['links']
    V = provider_jet(bank,em)['vector']
    h = 2.**-12/np.linalg.norm(V)
    source_h = 2.**-12/np.linalg.norm(V[52:])
    def sources(z: np.ndarray) -> np.ndarray:
        a = (model.em['legs']@z[:52]).reshape(model.law.volume,3)
        a = a + z[52:][None,:]/model.law.size
        return model.source_bank.word_sources(phi,links,z[52:],a)
    B = sources(em)
    # Scale the source FD by its actual harmonic displacement. Fixed h would
    # subtract almost identical tiny source phases and lose relative accuracy.
    dB = (sources(em+source_h*V)-sources(em-source_h*V))/(2*source_h)
    dBfine = (sources(em+source_h*V/2)-sources(em-source_h*V/2))/source_h
    # PURECOULOMB B has no coexact derivative. Keep and discriminate its actual
    # harmonic derivative; harmonic transport cannot be erased as a mean-zero flow.
    harmonicV = np.r_[np.zeros(52),V[52:]]
    dh = (sources(em+source_h*harmonicV)-sources(em-source_h*harmonicV))/(2*source_h)
    rng = np.random.default_rng(222914)
    weights = rng.normal(size=(2,55))+1j*rng.normal(size=(2,55))
    def psi(z: np.ndarray) -> np.ndarray:
        return np.exp(.03*(weights@z))*(1+.02*np.array([z@z,z[3]*z[54]]))
    p = psi(em)
    dp = (psi(em+h*V)-psi(em-h*V))/(2*h)
    div = bank.evaluate(em[:52])['divergence']
    zero_order = .5*div-em[:52]@V[:52]
    Lpsi = dp+zero_order*p
    product_fd = (sources(em+h*V)*psi(em+h*V)[None,None,None,:]
                 -sources(em-h*V)*psi(em-h*V)[None,None,None,:])/(2*h)+zero_order*B*p[None,None,None,:]
    rule = dBfine*p[None,None,None,:]+B*Lpsi[None,None,None,:]
    endpoint = flow['em_endpoint']
    moved_B = sources(endpoint)
    initial_product = B*p[None,None,None,:]
    transported = flow['factor']*initial_product
    transported_psi = flow['factor']*p
    # B is commonly transported as B_t(z_t)=B_0(z_0), not physically reset to B(z_t).
    common = B*transported_psi[None,None,None,:]
    reset = moved_B*transported_psi[None,None,None,:]
    source_signal = np.linalg.norm(dBfine*p[None,None,None,:])
    error = float(np.linalg.norm(product_fd-rule)/max(source_signal,np.linalg.norm(rule),np.finfo(float).tiny))
    assert error < 2e-7
    assert relative_error(dBfine,dB) < 2e-6
    assert relative_error(dBfine,dh) < 2e-6
    assert source_signal > 0
    reset_signal = float(np.linalg.norm(reset-transported)/max(np.linalg.norm(transported),np.finfo(float).tiny))
    assert reset_signal > 100*np.finfo(float).eps
    return dict(arbitrary_two_bank_product_rule_scaled_error=error,
                source_directional_FD_refinement=relative_error(dBfine,dB),
                native_harmonic_source_difference=relative_error(dBfine,dh),
                native_harmonic_source_jet_norm=float(np.linalg.norm(dBfine)),
                common_product_error=relative_error(common,transported),
                reset_source_adverse_relative_signal=reset_signal,
                profiles=list(model.source_bank.template_order),words=4,neutral_banks=2,
                full_CAR_rows=B.shape[2],point_scope_only=True,
                proof='Product rule for arbitrary differentiable source/wave; native FD witness, NOT global L2 norm or full pulse')


def native(model: Any, data: dict[str,Any], steps: int) -> dict[str,Any]:
    bank = model.noether_bank
    assert bank.dimension == 52 and len(bank.edge_curls) == 81
    em = np.r_[np.random.default_rng(222913).normal(size=52)*.07,[.19,-.11,.23]]
    clock = math.ceil(15*math.pi*1.5/model.c)
    t0,t1 = .37*clock,.37*clock+.031
    argv = data['manifest']['argv']
    guard = json.loads(Path(argv[argv.index('--noether-guard')+1]).read_bytes())
    actual = evolve_em(bank,clock,t0,t1,em,steps)
    fine = evolve_em(bank,clock,t0,t1,em,steps*2)
    finer = evolve_em(bank,clock,t0,t1,em,steps*4)
    frozen = _integrate(bank,clock,t0,t1,em,steps,pulse_point(clock,.37)['Omega'])
    mean = _integrate(bank,clock,t0,t1,em,steps,meanflow=True)
    motion_scale = max(np.linalg.norm(actual['em_endpoint']-em),np.finfo(float).tiny)
    refinements = {k:relative_error(fine[k],finer[k]) for k in ('initial_coordinate_jacobian','initial_log_factor_gradient')}
    refinements['endpoint_motion_scaled'] = float(np.linalg.norm(fine['em_endpoint']-finer['em_endpoint'])/motion_scale)
    refinements['loghalf_scaled'] = abs(fine['integrated_log_half_density']-finer['integrated_log_half_density'])/max(abs(finer['integrated_log_half_density']),np.finfo(float).tiny)
    frozen_signal = float(np.linalg.norm(actual['em_endpoint']-frozen['em_endpoint'])/motion_scale)
    mean_signal = float(np.linalg.norm(actual['em_endpoint']-mean['em_endpoint'])/motion_scale)
    missing_div_signal = abs(math.expm1(actual['integrated_log_half_density']))
    # Scale-sensitive, profile-independent: compare adverse signals to numerical
    # motion error, never an absolute allclose that erases microscopic flow.
    assert max(refinements.values()) < 1e-8, refinements
    assert frozen_signal > 100*max(refinements['endpoint_motion_scaled'],np.finfo(float).eps)
    assert mean_signal > .9
    assert missing_div_signal > 100*np.finfo(float).eps
    jac = actual['initial_coordinate_jacobian']
    sign,logdet = np.linalg.slogdet(jac)
    volume_error = abs(logdet+2*actual['integrated_log_half_density'])
    assert sign > 0 and volume_error < 3e-12
    h = 2.**-16
    direction = np.random.default_rng(222915).normal(size=55)
    direction /= np.linalg.norm(direction)
    plus = evolve_em(bank,clock,t0,t1,em+h*direction,steps)
    minus = evolve_em(bank,clock,t0,t1,em-h*direction,steps)
    jet_error = relative_error((plus['em_endpoint']-minus['em_endpoint'])/(2*h),jac@direction)
    factor_error = abs((plus['factor_log']-minus['factor_log'])/(2*h)-actual['initial_log_factor_gradient']@direction)/max(np.linalg.norm(actual['initial_log_factor_gradient']),np.finfo(float).tiny)
    assert jet_error < 1e-8 and factor_error < 2e-6
    return dict(initial_em=em,trajectory=actual,provider_control=provider_control(model,bank,em,guard),
                product_control=product_control(model,bank,em,actual),refinement=refinements,
                characteristic_Jacobian_directional_FD=jet_error,factor_gradient_FD=factor_error,
                local_Lebesgue_half_density_Jacobian_log_identity_error=volume_error,
                adverse=dict(frozen_Omega_motion_scaled_signal=frozen_signal,
                             coexact_mean_zero_motion_scaled_signal=mean_signal,
                             missing_divergence_relative_factor_signal=missing_div_signal),
                pulse_start=pulse_point(clock,t0/clock),pulse_end=pulse_point(clock,t1/clock),
                norm_scope='Local Jacobian/half-density identity only; no global field-L2 integration claimed')


def source_epoch() -> dict[str,bytes]:
    root = Path(__file__).resolve().parent
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
    args = p.parse_args()
    started = time.perf_counter()
    assert hashlib.sha256(args.cache_npz.read_bytes()).hexdigest() == CACHE_SHA
    assert hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest() == MANIFEST_SHA
    model,data = load_shared_native_cache(args.cache_npz,args.cache_manifest)
    before = source_epoch()
    science = native(model,data,args.steps)
    after = source_epoch()
    assert all(after.get(name)==body for name,body in before.items())
    with zipfile.ZipFile(args.source_archive,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,body in sorted(after.items()):
            archive.writestr(name,body)
    result = dict(plan=PLAN,revision=REVISION,work_item=WORK_ITEM,science=science,
                  argv=sys.argv,cwd=os.getcwd(),python=sys.executable,natural_exit=0,
                  wall_seconds=time.perf_counter()-started,cache_npz_sha256=CACHE_SHA,
                  cache_manifest_sha256=MANIFEST_SHA,
                  source_sha256={name:hashlib.sha256(body).hexdigest() for name,body in after.items()},
                  stable_loaded_sources=True,source_archive_sha256=hashlib.sha256(args.source_archive.read_bytes()).hexdigest(),
                  no_H_G_preparation_recompile=True,no_state_profile_law_reset=True,
                  debts=['H_EM and harmonic kinetic quantum evolution NOT included',
                         'coupled Phi/link/matter/Ref/Root/source CAR/fullcurrent time flow NOT included',
                         'full physical pulse/closed rotation/global L2 and legal unrepresented mode proof remain open'])
    text = json.dumps(portable(result),sort_keys=True)
    args.output.write_text(text+'\n')
    print(text)

if __name__ == '__main__':
    main()

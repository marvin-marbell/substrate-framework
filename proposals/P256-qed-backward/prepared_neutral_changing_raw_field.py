#!/usr/bin/env python3
"""Actual Phi/colour Noether characteristics; not full quantum H evolution."""
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
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from scipy.linalg import expm
from original_total_angular_transfer import primitive_flow,gauge_move
from native_closed_rotation_quantum import physical_chart_pullback
from prepared_neutral_moving_matter_frame import mean_matter_generators,pulse_point
from prepared_neutral_shared_photon_state import load_shared_native_cache
from original_charged_principal import portable


def link_left_generator(derivative: np.ndarray,links: np.ndarray) -> np.ndarray:
    return -1j*derivative@links.conj().swapaxes(-1,-2)


def advance_links(links: np.ndarray,left: np.ndarray,angle: float) -> np.ndarray:
    shape=links.shape
    return np.asarray([expm(-1j*angle*k)@u for k,u in zip(left.reshape(-1,4,4),links.reshape(-1,4,4))]).reshape(shape)


def evolve_raw_fields(law: Any,phi: np.ndarray,links: np.ndarray,velocity: np.ndarray,
                      alpha: np.ndarray,clock: int,t0: float,t1: float,steps: int=4) -> dict[str,Any]:
    """Midpoint Lie steps of -V in exact integrated quintic angle.

    Omega(t)V has autonomous V, so its time dependence is integrated exactly
    into Delta theta. Nonlinear field-flow integration is still finite-step.
    """
    if steps<1 or not (0<t0<clock and 0<t1<clock) or t0==t1:
        raise ValueError('Nonzero forward or inverse interval inside the actual pulse required')
    begin,end=pulse_point(clock,t0/clock),pulse_point(clock,t1/clock)
    angle=end['theta']-begin['theta'];h=angle/steps
    fields=np.array(phi,copy=True);colour=np.array(links,copy=True)
    path=[dict(angle=0.,phi=fields.copy(),links=colour.copy())]
    zeros=np.zeros((law.volume,3))
    for index in range(steps):
        df,du,_=primitive_flow(law,fields,colour,zeros,velocity,alpha)
        half_phi=fields-.5*h*df
        half_links=advance_links(colour,link_left_generator(du,colour),.5*h)
        df,du,_=primitive_flow(law,half_phi,half_links,zeros,velocity,alpha)
        fields=fields-h*df
        colour=advance_links(colour,link_left_generator(du,half_links),h)
        path.append(dict(angle=(index+1)*h,phi=fields.copy(),links=colour.copy()))
    return dict(phi_endpoint=fields,links_endpoint=colour,path=path,
        integrated_pulse_angle=angle,pulse_endpoints=[begin,end],steps=steps,
        method='midpoint Lie field characteristic, exact quintic integrated angle',
        scope='RAW all60Phi+45colour per site/full2835 Noether action; not quantum Phi kinetic/potential H')


def colour_haar_divergence(law: Any,links: np.ndarray,velocity: np.ndarray,
                           alpha: np.ndarray,step: float=2.**-16) -> float:
    """Literal left-Haar trace of colour V, by all81*15 native directions.

    Phi divergence is zero: centred neighbour field differences have no
    onsite diagonal, and the intrinsic Phi1/Phi2 rotation has zero trace.
    This numerical trace has an explicit FD remainder, not an analytic bound.
    """
    zeros=np.zeros((law.volume,3));fields=np.zeros((law.volume,4,4,4),complex)
    total=0.
    rotations=[(expm(1j*step*t),expm(-1j*step*t)) for t in law.generators]
    for site in range(law.volume):
        for axis in range(3):
            for generator,(plus,minus) in zip(law.generators,rotations):
                values=[]
                for move in (plus,minus):
                    changed=links.copy();changed[site,axis]=move@links[site,axis]
                    _,derivative,_=primitive_flow(law,fields,changed,zeros,velocity,alpha)
                    k=link_left_generator(derivative[site,axis],changed[site,axis])
                    values.append(float((2*np.trace(generator@k)).real))
                total+=(values[0]-values[1])/(2*step)
    return total


def provider_controls(model: Any,data: dict[str,Any],sample: dict[str,Any],
                      generators: dict[str,Any],orbit: dict[str,Any]) -> dict[str,Any]:
    guard=json.loads(generators['guard_path'].read_bytes())
    velocity,alpha=np.asarray(guard['velocity']),np.asarray(guard['alpha'])
    point=sample['point'];law=model.law
    pull=physical_chart_pullback(model,sample,alpha,velocity)
    rng=np.random.default_rng(222917)
    gauge=np.asarray([expm(1j*np.einsum('a,aij->ij',rng.normal(size=15)*.03,law.generators)) for _ in range(law.volume)])
    phases=np.zeros((law.volume,3))
    fields,links,a=gauge_move(law,point['phi'],point['links'],phases,gauge,np.zeros(law.volume))
    original=primitive_flow(law,point['phi'],point['links'],phases,velocity,alpha)
    rotated=primitive_flow(law,fields,links,a,velocity,alpha)
    phi_error=float(np.linalg.norm(rotated[0]-gauge[:,None]@original[0]@gauge[:,None].conj().swapaxes(-1,-2)))
    link_error=float(np.linalg.norm(rotated[1]-gauge[:,None]@original[1]@gauge[law.target].conj().swapaxes(-1,-2)))
    colour=orbit['links_endpoint'];phi=orbit['phi_endpoint']
    defect=float(np.linalg.norm(colour.conj().swapaxes(-1,-2)@colour-np.eye(4)))
    determinant=float(np.max(np.abs(np.linalg.det(colour)-1)))
    hermitian=float(np.linalg.norm(phi-phi.conj().swapaxes(-1,-2)))
    if max(phi_error,link_error,pull['reconstruction_error'],defect,determinant,hermitian)>1e-8:
        raise AssertionError('Native RAW provider/field invariants failed')
    return dict(RAW_full_J_reconstruction_error=pull['reconstruction_error'],
        actual_noncentral_colour_initial_norm=float(np.linalg.norm(point['links']-np.eye(4))),
        gauge_covariance_Phi_error=phi_error,gauge_covariance_colour_error=link_error,
        colour_unitarity=defect,colour_determinant=determinant,Phi_Hermiticity=hermitian,
        Phi_divergence_exact_zero_reason='centred distinct-neighbour blocks and off-diagonal intrinsic rotation',
        Gaussian_Jhalf_factored_initial_point=True,
        no_instantaneous_ground_or_field_function_recapture=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-npz',type=Path,required=True);p.add_argument('--cache-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=4)
    p.add_argument('--duration',type=float,default=.031)
    args=p.parse_args();started=time.perf_counter()
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    g=mean_matter_generators(model,data)
    guard=json.loads(g['guard_path'].read_bytes());velocity=np.asarray(guard['velocity']);alpha=np.asarray(guard['alpha'])
    clock=math.ceil(15*math.pi*1.5/model.c);t0=.37*clock;t1=t0+args.duration
    q=np.zeros(model.chart.normal_dim);q[:3]=np.asarray([.12,-.21,.17])*model.sigma
    q[-3:]=np.asarray([.17,-.14,.11])*model.sigma
    q+=.15*model.sigma*g['Y0']/np.linalg.norm(g['Y0'])
    em=np.r_[np.random.default_rng(222913).normal(size=52)*.07,[.19,-.11,.23]]
    sample=model.basis_at(q,em);point=sample['point']
    orbit=evolve_raw_fields(model.law,point['phi'],point['links'],velocity,alpha,clock,t0,t1,args.steps)
    inverse=evolve_raw_fields(model.law,orbit['phi_endpoint'],orbit['links_endpoint'],velocity,alpha,clock,t1,t0,args.steps)
    finer=evolve_raw_fields(model.law,point['phi'],point['links'],velocity,alpha,clock,t0,t1,2*args.steps)
    # Wrong frozen Omega uses a different integrated angle, not a law retune.
    frozen_angle=pulse_point(clock,t0/clock)['Omega']*(t1-t0)
    frozen_phi=point['phi'].copy();frozen_links=point['links'].copy()
    h=frozen_angle/args.steps;zeros=np.zeros((model.law.volume,3))
    for _ in range(args.steps):
        df,du,_=primitive_flow(model.law,frozen_phi,frozen_links,zeros,velocity,alpha)
        hp=frozen_phi-.5*h*df;hu=advance_links(frozen_links,link_left_generator(du,frozen_links),.5*h)
        df,du,_=primitive_flow(model.law,hp,hu,zeros,velocity,alpha)
        frozen_phi-=h*df;frozen_links=advance_links(frozen_links,link_left_generator(du,hu),h)
    div=colour_haar_divergence(model.law,point['links'],velocity,alpha)
    divfine=colour_haar_divergence(model.law,point['links'],velocity,alpha,step=2.**-17)
    controls=provider_controls(model,data,sample,g,orbit)
    controls.update(wrong_frozen_Omega_Phi_error=float(np.linalg.norm(orbit['phi_endpoint']-frozen_phi)),
        wrong_frozen_Omega_colour_error=float(np.linalg.norm(orbit['links_endpoint']-frozen_links)),
        RAW_inverse_Phi_error=float(np.linalg.norm(inverse['phi_endpoint']-point['phi'])),
        RAW_inverse_colour_error=float(np.linalg.norm(inverse['links_endpoint']-point['links'])),
        field_refinement_Phi_error=float(np.linalg.norm(orbit['phi_endpoint']-finer['phi_endpoint'])),
        field_refinement_colour_error=float(np.linalg.norm(orbit['links_endpoint']-finer['links_endpoint'])),
        RAW_colour_Haar_divergence=div,RAW_colour_divergence_FD_refinement_error=abs(div-divfine),
        common_halfdensity_log_initial_point_estimate=.5*orbit['integrated_pulse_angle']*div,
        halfdensity_scope='FD initial-point trace; its time-integral remainder is unpaid',
        sigma=float(model.sigma),chi=float(model.chi),Y0_norm=float(np.linalg.norm(g['Y0'])))
    if controls['wrong_frozen_Omega_Phi_error']<=1e-12:
        raise AssertionError('Changing pulse versus frozenOmega control is not discriminating')
    sources={Path(cast(str,m.__file__)).name:hashlib.sha256(Path(cast(str,m.__file__)).read_bytes()).hexdigest() for m in tuple(sys.modules.values()) if getattr(m,'__file__',None) and Path(cast(str,m.__file__)).parent==Path(__file__).parent}
    result=dict(controls=controls,pulse_endpoints=orbit['pulse_endpoints'],integrated_pulse_angle=orbit['integrated_pulse_angle'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),loaded_sources=sources,
        clock=clock,t0=t0,t1=t1,steps=args.steps,scope=orbit['scope'],
        cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
        no_H_G_preparation_or_loop_rerun=True,wall_seconds=time.perf_counter()-started,
        unpaid=['changing coupled quantumH Phi/EM kinetic and scalarpotential','RAW colour divergence FD/timeintegration remainder','wholeprefix/fullpulse/wholefieldL2/Gauss/physical2pi/Born/quantumlaw'])
    args.output.write_text(json.dumps(portable(result),indent=2,sort_keys=True));print(json.dumps(portable(result),sort_keys=True))

if __name__=='__main__':main()

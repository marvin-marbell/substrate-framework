#!/usr/bin/env python3
"""Paired RAW/EM characteristics and native H_D: not full quantum H evolution."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any,cast
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from scipy.linalg import expm,logm
from unrestricted_field_minimum import coordinates
from original_total_angular_transfer import scalar_orbital
from native_closed_rotation_quantum import physical_chart_pullback
from prepared_neutral_changing_raw_field import evolve_raw_fields,colour_haar_divergence
from prepared_neutral_changing_em_characteristic import evolve_em
from prepared_neutral_changing_mean_frame import propagate_bodies
from prepared_neutral_moving_matter_frame import mean_matter_generators,pulse_point
from prepared_neutral_shared_photon_state import load_shared_native_cache
from prepared_neutral_moving_frame_car import car_source_control
from prepared_neutral_moving_frame_current import moving_current_control,current_graph
from original_charged_principal import portable


def inverse_raw_section(model: Any,phi: np.ndarray,links: np.ndarray,q0: np.ndarray,
                        tolerance: float=2e-11,iterations: int=6) -> dict[str,Any]:
    """Local full2835 normal+SU4 inverse, without truncating gauge columns."""
    q=np.array(q0,copy=True);law=model.law
    gauge=np.tile(np.eye(4,dtype=complex),(law.volume,1,1));history=[]
    for _ in range(iterations):
        point=model.chart.evaluate(q)
        fields=gauge[:,None].conj().swapaxes(-1,-2)@phi@gauge[:,None]
        colour=gauge[:,None].conj().swapaxes(-1,-2)@links@gauge[law.target]
        dphi=coordinates(fields-point['phi'],law.basis).reshape(law.volume,60)
        left=np.asarray([-1j*logm(u@v.conj().T) for u,v in zip(colour.reshape(-1,4,4),point['links'].reshape(-1,4,4))]).reshape(law.volume,3,4,4)
        dlinks=(2*np.einsum('xjkl,alk->xja',left,law.generators)).real.reshape(law.volume,45)
        residual=np.c_[dphi,dlinks].ravel()
        norm=float(np.linalg.norm(residual));history.append(norm)
        if norm<tolerance:
            return dict(q=q,gauge=gauge,point=point,residual_history=history,
                reconstruction_error=norm,normal_dim=model.chart.normal_dim,gauge_dim=model.chart.gauge_dim)
        delta=point['cometric']@(point['jacobian'].T@(model.chart.mass*residual))
        q+=delta[:model.chart.normal_dim]
        body=np.einsum('xa,aij->xij',delta[model.chart.normal_dim:].reshape(law.volume,15),law.generators)
        gauge=np.asarray([g@expm(1j*b) for g,b in zip(gauge,body)])
    raise ValueError('RAW fullnormal/gauge inverse did not converge: '+repr(history))


def block_gauge(gauge: np.ndarray) -> np.ndarray:
    out=np.zeros((4*len(gauge),4*len(gauge)),complex)
    for site,g in enumerate(gauge):out[4*site:4*site+4,4*site:4*site+4]=g
    return out


def source_point(model: Any,data: dict[str,Any]) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    g=mean_matter_generators(model,data)
    q=np.zeros(model.chart.normal_dim);q[:3]=np.array([.12,-.21,.17])*model.sigma
    q[-3:]=np.array([.17,-.14,.11])*model.sigma
    q+=.15*model.sigma*g['Y0']/np.linalg.norm(g['Y0'])
    em=np.r_[np.random.default_rng(222913).normal(size=52)*.07,[.19,-.11,.23]]
    return q,em,g


def coupled_interval(model: Any,data: dict[str,Any],q0: np.ndarray,em0: np.ndarray,
                     clock: int,t0: float,t1: float,steps: int=8) -> dict[str,Any]:
    """Actual first-order G+native H_D interval, paired held seed functions.

    Raw Phi/colour and EM characteristics are actual physicalG. Scalar V is
    retained as a separate exact-law multiplication phase. Quantum Phi/EM
    second-order kinetic/Coulomb multipair evolution is NOT implied.
    """
    g=mean_matter_generators(model,data);guard=json.loads(g['guard_path'].read_bytes())
    velocity,alpha=np.asarray(guard['velocity']),np.asarray(guard['alpha'])
    initial=model.basis_at(q0,em0);point0=initial['point'];records={}
    def provider(t: float) -> dict[str,Any]:
        if t not in records:
            if t==t0:
                raw=dict(phi_endpoint=point0['phi'],links_endpoint=point0['links'])
                em=em0
            else:
                raw=evolve_raw_fields(model.law,point0['phi'],point0['links'],velocity,alpha,clock,t0,t,steps)
                em=evolve_em(model.noether_bank,clock,t0,t,em0,steps)['em_endpoint']
            a,theta=model.original.electromagnetic_field(em)
            micro=model.original.microscopic_operators(raw['phi_endpoint'],raw['links_endpoint'],a)
            value=model.noether_bank.evaluate(em[:52]);omega=pulse_point(clock,t/clock)['Omega']
            lam=np.diag(np.repeat(value['lambda_C'],4))
            ml=scalar_orbital(model.law,raw['links_endpoint'],a,velocity)+lam
            mh=scalar_orbital(model.heavy_law,raw['links_endpoint'],np.zeros((model.law.volume,3)),velocity)
            potential=model.original.classical_potential(dict(phi=raw['phi_endpoint'],links=raw['links_endpoint']))
            # massive_sea is a convenience expectation already counted by H_D.
            scalar=sum(potential[k] for k in ('quartic','alignment','colour_magnetic'))
            records[t]=dict(raw=raw,em=em,a=a,theta=theta,micro=micro,noether=value,
                bodies=dict(light=micro['light']-omega*ml,
                    heavy=-model.Lambda*micro['massive']-omega*mh,reference=np.zeros((4,4),complex)),
                scalar_potential=scalar,excluded_duplicate_massive_sea=potential['massive_sea'])
        return records[t]
    matter=propagate_bodies(lambda t:provider(t)['bodies'],t0,t1,steps)
    end=provider(t1)
    orbit=evolve_raw_fields(model.law,point0['phi'],point0['links'],velocity,alpha,clock,t0,t1,steps)
    emorbit=evolve_em(model.noether_bank,clock,t0,t1,em0,steps)
    inverse=inverse_raw_section(model,end['raw']['phi_endpoint'],end['raw']['links_endpoint'],q0)
    gauge=block_gauge(inverse['gauge']);refg=inverse['gauge'][model.reference_site]
    u={name:gauge.conj().T@matter['unitaries'][name] for name in ('light','heavy')}
    u['reference']=refg.conj().T@matter['unitaries']['reference']
    moved=model.in_mean_interaction_frame(u['light'],u['reference'],u['heavy'])
    pulse=pulse_point(clock,t1/clock);point=inverse['point'];q=inverse['q']
    pull=physical_chart_pullback(model,dict(point=point),alpha,velocity)
    ga=np.einsum('xa,aij->xij',pull['gauge'].reshape(model.law.volume,15),model.law.generators)
    gauge_body=block_gauge(ga)
    a,theta=model.original.electromagnetic_field(emorbit['em_endpoint'])
    endpoint_micro=model.original.microscopic_operators(point['phi'],point['links'],a)
    lam=np.diag(np.repeat(model.noether_bank.evaluate(emorbit['em_endpoint'][:52])['lambda_C'],4))
    endpoint_bodies=dict(light=endpoint_micro['light']-pulse['Omega']*(scalar_orbital(model.law,point['links'],a,velocity)+gauge_body+lam),
        heavy=-model.Lambda*endpoint_micro['massive']-pulse['Omega']*(scalar_orbital(model.heavy_law,point['links'],np.zeros((model.law.volume,3)),velocity)+gauge_body),
        reference=-pulse['Omega']*model.reference_frame.conj().T@ga[model.reference_site]@model.reference_frame)
    connections={name:-v.conj().T@endpoint_bodies[name]@v for name,v in u.items()}
    # Native RAW density: midpoint along the actual field path, no zeroing.
    midpoint=provider((t0+t1)/2)['raw']
    divergence=colour_haar_divergence(model.law,midpoint['links_endpoint'],velocity,alpha)
    loghalf=.5*orbit['integrated_pulse_angle']*divergence+emorbit['integrated_log_half_density']
    half=math.exp(loghalf)
    # Native Coulomb-gauge background -2 per light site is a common phase.
    background=complex(np.exp(2j*np.sum(emorbit['integrated_anchored_lambda_C'])))
    # Full original positive Phi/colour scalar potential, no massive doublecount.
    c1,c2=.5-math.sqrt(3)/6,.5+math.sqrt(3)/6;dt=(t1-t0)/steps
    scalar_integral=sum(dt*.5*(provider(t0+(k+c1)*dt)['scalar_potential']+provider(t0+(k+c2)*dt)['scalar_potential']) for k in range(steps))
    scalar_phase=complex(np.exp(-1j*scalar_integral))
    boost_phase=complex(np.exp(1j*pulse['Omega']*(g['Y0']@q)/model.chi))
    common=half*background*scalar_phase
    physical=dict(initial);physical.update(q=q,em=emorbit['em_endpoint'],point=point,a=a,theta=theta,
        neutral=common*initial['neutral'],charged=common*initial['charged'])
    # W is an explicitly separate carrier/frame representation. Its inverse
    # coefficient phase gives the same physical sector state, not extra drive.
    carrier=dict(physical);carrier.update(neutral=boost_phase*physical['neutral'],charged=boost_phase*physical['charged'])
    return dict(model=moved,initial_sample=initial,physical_sample=physical,carrier_sample=carrier,
        boost_inverse_coefficient=boost_phase.conjugate(),unitaries=u,raw_unitaries=matter['unitaries'],
        endpoint_bodies=endpoint_bodies,connections=connections,mean_generators=g,
        field_orbit=orbit,em_orbit=emorbit,section_inverse=inverse,pulse=pulse,
        scalar_integral=scalar_integral,scalar_phase=scalar_phase,background_phase=background,
        common_halfdensity=half,common_halfdensity_log=loghalf,colour_divergence_midpoint=divergence,
        field_time_connection=pulse['Omega_dot']*(g['Y0']@q)/model.chi,
        completing_square_mean_energy=-pulse['Omega']**2*float(g['Y0']@g['Y0'])/(2*model.chi),
        retained_native_scalar_endpoint=end['scalar_potential'],excluded_duplicate_massive_sea=end['excluded_duplicate_massive_sea'],
        source_bank_held=moved.source_bank is model.source_bank,
        native_provider=provider,velocity=velocity,alpha=alpha,clock=clock,
        initial_q=q0,initial_em=em0,
        scope='actual short paired physicalG+native H_D+literal scalar sector; NOT full quantum H or full pulse',
        time_interval=[t0,t1],steps=steps)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-npz',type=Path,required=True);p.add_argument('--cache-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=8)
    p.add_argument('--duration',type=float,default=.031)
    args=p.parse_args();started=time.perf_counter()
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    q,em,g=source_point(model,data);clock=math.ceil(15*math.pi*1.5/model.c);t0=.37*clock
    frame=coupled_interval(model,data,q,em,clock,t0,t0+args.duration,args.steps)
    moved=frame['model'];sample=frame['physical_sample'];neutral=sample['neutral']@data['neutral_state']
    point=sample['point'];rows={}
    for profile in ('source19','SOFT'):
        source=moved.source_at(point['phi'],point['links'],sample['theta'],sample['a'],profile)
        car=car_source_control(moved.fibre,source,neutral,frame['endpoint_bodies']['light'],frame['endpoint_bodies']['reference'])
        raw_field_current=model.law.onebody(point['phi'],point['links'],sample['a'])-model.law.onebody(point['phi'],point['links'],np.tile(sample['theta']/model.law.size,(model.law.volume,1)))
        current=current_graph(moved.fibre,raw_field_current,source,neutral)
        held=sample['charged']@data[profile+'_map']@data['neutral_state']
        rows[profile]=dict(CAR_relative_error=car['fullCAR_commutator_relative_error'],
            CAR_LEFT_PH_norm=car['LEFT_PH_norm'],CAR_RIGHT_PH_norm=car['RIGHT_PH_norm'],
            CAR_missing_projected_neutral_RIGHT=car['RIGHT_projected_neutral_only_missing_PH_norm'],
            actual_all52_field_current={k:current[k] for k in ('full_norm_squared','projected_norm_squared','unrepresented_PH_norm_squared','neutral_RIGHT_outside_norm_squared','full_neutral_RIGHT_identity_error')},
            matched_initial_BXi_versus_physical_endpoint_BXi_relative_difference=float(np.linalg.norm(held-source['source_matrix']@neutral)/np.linalg.norm(held)))
    literal=moving_current_control(model,moved,data,sample,frame['unitaries']['light'],frame['unitaries']['reference'])
    pointmetric=point['cometric'][:model.chart.normal_dim,:model.chart.normal_dim];y0=g['Y0']
    pull=physical_chart_pullback(model,sample,np.asarray(json.loads(g['guard_path'].read_bytes())['alpha']),np.asarray(json.loads(g['guard_path'].read_bytes())['velocity']))
    boost_kinetic=frame['pulse']['Omega']**2*float(y0@pointmetric@y0)/(2*model.chi)
    boost_drive=-frame['pulse']['Omega']**2*float(pull['normal']@y0)/model.chi
    connections={name:float(np.linalg.norm(u.conj().T@frame['endpoint_bodies'][name]@u+frame['connections'][name])) for name,u in frame['unitaries'].items()}
    sources={Path(cast(str,m.__file__)).name:hashlib.sha256(Path(cast(str,m.__file__)).read_bytes()).hexdigest() for m in tuple(sys.modules.values()) if getattr(m,'__file__',None) and Path(cast(str,m.__file__)).parent==Path(__file__).parent}
    result=dict(scope=frame['scope'],clock=clock,time_interval=frame['time_interval'],steps=args.steps,
        provider_full_RAW_dimension=model.chart.raw_dim,provider_EM_dimension=55,
        source_bank_held=frame['source_bank_held'],inverse_residuals=frame['section_inverse']['residual_history'],
        inverse_section_q_norm=float(np.linalg.norm(sample['q'])),initial_q_norm=float(np.linalg.norm(q)),sigma=float(model.sigma),
        nontrivial_Ref_transport=float(np.linalg.norm(frame['unitaries']['reference']-np.eye(4))),
        matter_connection_residuals=connections,common_halfdensity_log=frame['common_halfdensity_log'],
        field_time_connection=frame['field_time_connection'],mean_completing_square_energy=frame['completing_square_mean_energy'],
        original_full_metric_boost_kinetic_energy=boost_kinetic,actual_G_boost_drive_energy=boost_drive,
        actual_boost_kinetic_plus_drive_energy=boost_kinetic+boost_drive,
        literal_scalar_integral=frame['scalar_integral'],literal_scalar_endpoint=frame['retained_native_scalar_endpoint'],
        excluded_duplicate_massive_sea=frame['excluded_duplicate_massive_sea'],rows=rows,literal_currents=literal,
        physical_wave_scope='initial Gaussian/Jhalf divided along moving RAW endpoint; W carrier inverse phase retained, not an extra physical drive',
        no_H_G_preparation_or_loop_rerun=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),loaded_sources=sources,
        cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
        unpaid=['quantum Phi/EM second-order kinetic and Coulomb multipair propagation','wholeprefix/fullpulse/globalL2','FD/time-step/commonform remainders','Root/Gauss/legalSOFT/physical2pi/Born/quantumlaw'],wall_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(portable(result),indent=2,sort_keys=True));print(json.dumps(portable(result),sort_keys=True))

if __name__=='__main__':main()

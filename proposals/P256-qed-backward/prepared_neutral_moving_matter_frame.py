#!/usr/bin/env python3
"""Once-seeded full-matter unitary frame and common field boost.

The actual native smoke is one local frozen-time frame/field point, not a
propagator for the full changing pulse and not physical closed-loop proof.
"""
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
from native_closed_rotation_quantum import physical_chart_pullback
from original_total_angular_transfer import scalar_orbital
from original_charged_principal import portable
from prepared_neutral_common_boost import common_boost
from prepared_neutral_shared_photon_state import load_shared_native_cache
from prepared_neutral_quartet_functions import QuartetSourceFunctionBank
from prepared_neutral_full_field import BETA

PLAN='issue-222-fullfield-closedrotation-continuation-20261006'
REVISION='57838ca23dbf1424677c1c2f4e183784463faa3f759a85ec2a2b9722ff1fd9ff'
WORK_ITEM='prepared-neutral-full-field-outcome'


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def pulse_point(clock: int, fraction: float) -> dict[str,float]:
    if not 0 < fraction < 1:
        raise ValueError('Interior actual quintic pulse fraction required')
    u=fraction
    return dict(clock=float(clock),fraction=u,
        theta=2*math.pi*(10*u**3-15*u**4+6*u**5),
        Omega=60*math.pi*u*u*(1-u)**2/clock,
        Omega_dot=120*math.pi*u*(1-u)*(1-2*u)/(clock*clock))


def mean_matter_generators(model: Any, data: dict[str,Any]) -> dict[str,Any]:
    """Actual B RAW/orbital/gauge/Ref matrices; no assigned spin or new sea."""
    argv=data['manifest']['argv']
    guard_path=Path(argv[argv.index('--noether-guard')+1])
    guard=json.loads(guard_path.read_bytes())
    alpha,velocity=np.asarray(guard['alpha']),np.asarray(guard['velocity'])
    mean=model.basis_at(np.zeros(model.chart.normal_dim),np.zeros(model.em['dimension']))
    pull=physical_chart_pullback(model,mean,alpha,velocity)
    local=np.einsum('xa,aij->xij',pull['gauge'].reshape(model.law.volume,15),model.law.generators)
    gauge=np.zeros((model.fibre.n,model.fibre.n),complex)
    for site in range(model.law.volume):
        gauge[4*site:4*site+4,4*site:4*site+4]=local[site]
    light=scalar_orbital(model.law,mean['point']['links'],mean['a'],velocity)+gauge
    heavy=scalar_orbital(model.heavy_law,mean['point']['links'],np.zeros((model.law.volume,3)),velocity)+gauge
    reference=model.reference_frame.conj().T@local[model.reference_site]@model.reference_frame
    for value in (light,heavy,reference):
        if np.linalg.norm(value-value.conj().T)>1e-9:
            raise ValueError('Actual mean Noether matrix is not Hermitian')
    return dict(light=light,heavy=heavy,reference=reference,Y0=pull['normal'],
        gauge=pull['gauge'],raw_reconstruction_error=pull['reconstruction_error'],
        light_H=model.fibre.h0,
        heavy_H=model.original.massive_onebody(mean['point']['phi'],mean['point']['links']),
        mean_sample=mean,guard_path=guard_path,guard_sha256=hashlib.sha256(guard_path.read_bytes()).hexdigest())


def construct_point_frame(model: Any,data: dict[str,Any],local_step: float=.031,
                          pulse_fraction: float=.37) -> dict[str,Any]:
    """Full local mean unitary; no eigensolve or instantaneous-ground reset.

    F(delta)=exp(-i delta A_mean) for the frozen instantaneous mean matrices.
    -i F^dagger dF/delta=-A_mean cancels the mean Hamiltonian in that frame.
    This family is not time ordering of the complete physical T566 pulse.
    """
    if local_step<=0:
        raise ValueError('Positive local frame step required')
    g=mean_matter_generators(model,data)
    pulse=pulse_point(math.ceil(15*math.pi*1.5/model.c),pulse_fraction)
    omega,omega_dot=pulse['Omega'],pulse['Omega_dot']
    bodies=dict(light=g['light_H']-omega*g['light'],
                heavy=-model.Lambda*g['heavy_H']-omega*g['heavy'],
                reference=-omega*g['reference'])
    unitaries={name:expm(-1j*local_step*a) for name,a in bodies.items()}
    moved=model.in_mean_interaction_frame(unitaries['light'],unitaries['reference'],unitaries['heavy'])
    moved=moved.with_common_boost(omega,omega_dot,g['Y0'])
    q=np.zeros(model.chart.normal_dim)
    q[:3]=np.asarray([.12,-.21,.17])*model.sigma
    q+=.15*model.sigma*g['Y0']/np.linalg.norm(g['Y0'])
    em=np.r_[np.random.default_rng(222913).normal(size=52)*.07,[.19,-.11,.23]]
    sample=moved.basis_at(q,em)
    return dict(model=moved,sample=sample,unitaries=unitaries,bodies=bodies,
        connection={name:-a for name,a in bodies.items()},generators=g,pulse=pulse,
        local_step=local_step,scope='local frozen instantaneous mean unitary/point; NOT full changing-pulse trajectory')


def frame_kinematics_control(model: Any,data: dict[str,Any],frame: dict[str,Any]) -> dict[str,Any]:
    moved,sample=frame['model'],frame['sample']
    u,bodies=frame['unitaries'],frame['bodies']
    unitary={name:float(np.linalg.norm(v.conj().T@v-np.eye(len(v)))) for name,v in u.items()}
    residual={name:float(np.linalg.norm(v.conj().T@bodies[name]@v+frame['connection'][name])/(1+np.linalg.norm(bodies[name]))) for name,v in u.items()}
    occupied_errors={str(q):float(np.linalg.norm(moved.fibre.occupied[q]-u['light']@model.fibre.occupied[q])) for q in (1,-1)}
    heavy_error=float(np.linalg.norm(moved.heavy_orbitals-u['heavy']@model.heavy_orbitals))
    held=float(np.linalg.norm(moved.source_bank.seed_fibre.orbitals-model.source_bank.seed_fibre.orbitals))
    state_error=float(np.linalg.norm(moved.prepared14-model.prepared14))
    oh=model.heavy_orbitals
    complete,_=np.linalg.qr(oh,mode='complete');empty=complete[:,oh.shape[1]:]
    heavy_mean=float(np.trace(oh.conj().T@bodies['heavy']@oh).real)
    heavy_variance=float(np.linalg.norm(empty.conj().T@bodies['heavy']@oh)**2)
    heavy_overlap=complex(np.linalg.det(oh.conj().T@u['heavy']@oh))
    wrong_heavy_distance_squared=2-2*heavy_overlap.real
    # Mean Noether-only tangent gives a separate full PH control; it is not
    # used as a scalar rotation generator in the physical frame above.
    mh=frame['generators']['heavy']
    noe_heavy_ph=float(np.linalg.norm(empty.conj().T@mh@oh)**2)
    neutral=sample['neutral']@data['neutral_state']
    first=model.fibre.onebody(bodies['light'],'neutral')+model.fibre.reference(bodies['reference'],'neutral')
    second=model.fibre.bilinear_product(bodies['light'],bodies['light'],'neutral',bodies['reference'],bodies['reference'])
    missing=second+2*heavy_mean*first+(heavy_mean**2+heavy_variance)*model.fibre.overlap('neutral')
    missing_connection=float(np.vdot(neutral,missing@neutral).real)
    boost=common_boost(sample['q'],frame['pulse']['Omega'],frame['pulse']['Omega_dot'],frame['generators']['Y0'],model.chi)
    # The boost multiplies neutral and all fixed charged functions equally.
    unboosted=moved.with_common_boost(0.,0.,frame['generators']['Y0']).basis_at(sample['q'],sample['em'],sample['point'])
    phase_errors={s:float(np.linalg.norm(sample[s]-boost['phase']*unboosted[s])/(1+np.linalg.norm(sample[s]))) for s in ('neutral','charged')}
    # An explicitly WRONG new SOURCE-function snapshot on transported orbitals
    # is compared, never installed in the actual state or compiled cache.
    recaptured=QuartetSourceFunctionBank(moved.fibre,dict(moved.source_bank.profiles))
    recaptured_values=recaptured.source_values(sample['point']['phi'],sample['point']['links'],
        sample['theta'],sample['a'],neutral[:,None])
    recapture_errors={}
    for profile in ('source19','SOFT'):
        wrong=recaptured.combine_values(recaptured_values,BETA,1,profile)[:,0]
        held_source=sample['charged']@data[profile+'_map']@data['neutral_state']
        recapture_errors[profile]=float(np.linalg.norm(wrong-held_source)/np.linalg.norm(held_source))
    if max(unitary.values())>2e-8 or max(residual.values())>2e-8 or max(occupied_errors.values())>1e-12 or heavy_error>1e-12 or held!=0 or state_error!=0 or max(phase_errors.values())>2e-12:
        raise AssertionError('Full once-seeded frame/connection/boost contract failed')
    if wrong_heavy_distance_squared<1e-8 or missing_connection<=0:
        raise AssertionError('Wrong heavy or missing connection control is not discriminating')
    return dict(unitarity=unitary,mean_matrix_plus_connection_relative_residual=residual,
        light_occupied_transport_errors=occupied_errors,heavy_54_transport_error=heavy_error,
        source_initial_orbitals_held_error=held,prepared14_error=state_error,
        source_beta_unchanged=True,new_ground_eigh_or_filter=False,
        heavy_mean_energy=heavy_mean,heavy_full_PH_variance=heavy_variance,
        heavy_mean_Noether_PH_variance=noe_heavy_ph,
        wrong_frozen_heavy_full_Slater_distance_squared=wrong_heavy_distance_squared,
        heavy_Slater_overlap=[heavy_overlap.real,heavy_overlap.imag],
        missing_fullmatter_connection_point_norm_squared=missing_connection,
        common_boost_all_functions_relative_errors=phase_errors,
        wrong_SOURCE_function_recapture_relative_errors=recapture_errors,
        field_time_connection=boost['time_connection'],
        wrong_missing_field_time_connection=abs(boost['time_connection']),
        completing_square_energy_retained=boost['completing_square_common_energy'],
        physical_LOW_covector='chi Pi - Omega Y0, not bare Pi',
        common_heavy_PHASE_may_cancel_SOURCE_relative='full heavy state/connection retained; not claimed source-defect cause',
        norm_scope='pointwise coefficient functions divided common Gaussian/Jhalf, not whole-field L2')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-npz',type=Path,required=True);p.add_argument('--cache-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--local-step',type=float,default=.031)
    p.add_argument('--pulse-fraction',type=float,default=.37)
    args=p.parse_args();started=time.perf_counter()
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    before={Path(cast(str,m.__file__)).name:hashlib.sha256(Path(cast(str,m.__file__)).read_bytes()).hexdigest() for m in tuple(sys.modules.values()) if getattr(m,'__file__',None) and Path(cast(str,m.__file__)).parent==Path(__file__).parent}
    frame=construct_point_frame(model,data,args.local_step,args.pulse_fraction)
    science=frame_kinematics_control(model,data,frame)
    # Scientific leaves perform full CAR and current actions before projection.
    from prepared_neutral_moving_frame_car import car_source_control
    from prepared_neutral_moving_frame_current import moving_current_control
    sample=frame['sample'];neutral=sample['neutral']@data['neutral_state']
    car={}
    for profile in ('source19','SOFT'):
        source=frame['model'].source_at(sample['point']['phi'],sample['point']['links'],sample['theta'],sample['a'],profile)
        car[profile]=car_source_control(frame['model'].fibre,source,neutral,frame['bodies']['light'],frame['bodies']['reference'])
    current=moving_current_control(model,frame['model'],data,sample,frame['unitaries']['light'],frame['unitaries']['reference'])
    after={Path(cast(str,m.__file__)).name:hashlib.sha256(Path(cast(str,m.__file__)).read_bytes()).hexdigest() for m in tuple(sys.modules.values()) if getattr(m,'__file__',None) and Path(cast(str,m.__file__)).parent==Path(__file__).parent}
    changed={name for name,digest in before.items() if after.get(name)!=digest}
    if changed:raise AssertionError('Imported supplier changed: '+repr(changed))
    result=dict(plan=PLAN,revision=REVISION,work_item=WORK_ITEM,science=portable(science),
        car=car,current=current,pulse=frame['pulse'],local_step=args.local_step,
        scope=frame['scope'],natural_exit=0,wall_seconds=time.perf_counter()-started,
        argv=sys.argv,cache_npz_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),
        cache_manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
        input_H_state_map_hashes={k:array_digest(data[k]) for k in ('neutral_H','charged_H','neutral_Gram','charged_Gram','neutral_state','source19_map','SOFT_map')},
        native_light_H_coefficient=1,heavy_H_coefficient=-model.Lambda,Ref_onsite_H=0,
        actual_B_mean_Noether_no_assigned_spin=True,
        guard_sha256=frame['generators']['guard_sha256'],loaded_sources_before=before,
        loaded_sources_after=after,changed_sources=sorted(changed),added_imports={k:v for k,v in after.items() if k not in before},
        no_H_G_or_preparation_recompile=True,
        unpaid=['full changing-pulse time flow/global field-L2 bounds','field-dependent many-body PH/multipair dynamical closure','full Root Q/all512/encoded CAR/Gauss kernel','native/legal modes/SOFTrestriction mass/parameters','physical closed-loop minus/Born/original quantum-law derivation'])
    args.output.write_text(json.dumps(portable(result),indent=2,sort_keys=True));print(json.dumps(portable(result),sort_keys=True))

if __name__=='__main__':main()

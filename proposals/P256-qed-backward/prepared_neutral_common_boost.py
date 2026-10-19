#!/usr/bin/env python3
"""Common field-momentum frame and actual SOURCE-relative mean stress control."""
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
from scipy.linalg import lu_factor, lu_solve

from improved_phi_band import ImprovedPhiLaw, selected_band
from original_charged_principal import portable
from original_total_angular_transfer import primitive_flow
from prepared_neutral_ambient_weak_state import AmbientWeakState
from prepared_neutral_full_field import BETA
from prepared_neutral_quartet_functions import source19_native_profile


def common_boost(q: np.ndarray, omega: float, omega_dot: float,
                 mean_normal_velocity: np.ndarray, chi: float) -> dict[str,Any]:
    """Unitary on ALL sectors, not a per-bank or physical SOURCE normalization."""
    q=np.asarray(q,float);y0=np.asarray(mean_normal_velocity,float)
    if q.shape!=y0.shape or chi<=0:
        raise ValueError('Full shared normal vector and positive chi required')
    covector=omega*y0/chi
    return dict(phase=np.exp(1j*float(covector@q)),momentum_covector=covector,
                time_connection=omega_dot*float(y0@q)/chi,
                completing_square_common_energy=-omega**2*float(y0@y0)/(2*chi),
                residual_momentum='chi Pi - Omega Y0; cross-cometric and off-mean Y retained',
                SOURCE_common_phase_commutes=True,C0_not_shifted=True)


def actual_normal_flow(model: AmbientWeakState, point: dict[str,Any],
                       velocity: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Literal RAW Phi/colour Noether flow pulled through the full joint J."""
    flow=primitive_flow(model.law,point['phi'],point['links'],
                        np.zeros((model.law.volume,3)),velocity,alpha)
    phi=.25*np.einsum('xmij,aji->xma',flow[0],model.law.basis).real
    left=-1j*np.einsum('xjab,xjcb->xjac',flow[1],point['links'].conj())
    colour=2*np.einsum('xjbc,acb->xja',left,model.law.generators).real
    raw=np.c_[phi.reshape(model.law.volume,60),colour.reshape(model.law.volume,45)].ravel()
    root=np.sqrt(model.chart.mass)
    full=lu_solve(lu_factor(root[:,None]*point['jacobian']),root*raw)
    return full[:model.chart.normal_dim]


def mean_source_control(root_lemma: Path,prepared_npz: Path,descriptor: Path,
                        boost_input: Path) -> dict[str,Any]:
    """Native first-jet control on the physical function, preserving frozen Xi."""
    band,frame,_=selected_band();law=ImprovedPhiLaw(3,band)
    fields=np.tile(frame,(law.volume,1,1,1));links=np.tile(np.eye(4,dtype=complex),(law.volume,3,1,1))
    original_profile=np.exp(-np.sum((law.sites-1)**2,axis=1)/(.65**2));original_profile/=np.linalg.norm(original_profile)
    model=AmbientWeakState(law,fields,links,law.index[(0,1,1)],original_profile,json.loads(root_lemma.read_bytes()))
    binding=json.loads(boost_input.read_bytes())
    y0=np.asarray(binding['Y0_normal'],float)
    velocity=np.asarray(binding['velocity'],float);alpha=np.asarray(binding['alpha'],float)
    if y0.shape!=(model.chart.normal_dim,):raise ValueError('Actual all2430 mean Y0 required')
    with np.load(prepared_npz) as data:
        c=np.array(data['neutral_state']);gn=np.array(data['neutral_Gram'])
    q0=np.zeros(model.chart.normal_dim);point=model.chart.evaluate(q0)
    measured_y=actual_normal_flow(model,point,velocity,alpha)
    y_error=float(np.linalg.norm(measured_y-y0))
    if y_error>1e-10:raise AssertionError('Noether mean/chart binding mismatch')
    profile=source19_native_profile(law,descriptor)['profile']
    theta=np.array([.21,-.13,.09]);a=np.tile(theta/law.size,(law.volume,1))
    def source(q: np.ndarray) -> np.ndarray:
        p=model.chart.embedding(q)
        return model.fibre.source_vectors_at(p['phi'],p['links'],theta,a,profile,BETA)['source_matrix']
    def neutral(q: np.ndarray) -> np.ndarray:
        polynomial=model.polynomial(q)[0]
        return np.array([polynomial@c[:7],polynomial@c[7:]])
    psi=neutral(q0);b0=source(q0)
    def jet(direction: np.ndarray,step: float) -> np.ndarray:
        return (source(step*direction)-source(-step*direction))/(2*step)
    coarse=jet(y0,2.**-18);db=jet(y0,2.**-19)
    clock=math.ceil(15*math.pi*float(binding['Rout'])/model.c)
    omega=15*math.pi/(4*clock)
    boost=common_boost(q0,omega,.001,y0,model.chi)
    pboost=boost['momentum_covector']
    vboost=model.chi*point['cometric'][:model.chart.normal_dim,:model.chart.normal_dim]@pboost
    dkin=jet(vboost/omega,2.**-19)
    kinetic_relative=-1j*omega*dkin@psi
    noether_relative=1j*omega*db@psi
    cancellation=float(np.linalg.norm(kinetic_relative+noether_relative))
    wrong_bare=float(np.linalg.norm(noether_relative))
    step=2.**-19
    dwhole=(source(step*y0)@neutral(step*y0)-source(-step*y0)@neutral(-step*y0))/(2*step)
    dpsi=(neutral(step*y0)-neutral(-step*y0))/(2*step)
    right_error=float(np.linalg.norm(dwhole-b0@dpsi-db@psi))
    qp=.0005*y0/np.linalg.norm(y0);probe=model.chart.evaluate(qp)
    yp=actual_normal_flow(model,probe,velocity,alpha)
    vp=model.chi*probe['cometric'][:model.chart.normal_dim,:model.chart.normal_dim]@pboost
    offmean=float(np.linalg.norm(vp-omega*yp))
    bp=source(qp);psip=neutral(qp);wp=common_boost(qp,omega,.001,y0,model.chi)
    comm_phase=float(np.linalg.norm(bp@(wp['phase']*psip)-wp['phase']*(bp@psip)))
    wrong_one_sector=float(np.linalg.norm(wp['phase']*(bp@psip)-bp@psip))
    if cancellation>1e-10 or wrong_bare<1e-5 or right_error>1e-7 or comm_phase>1e-12 or wrong_one_sector<1e-5:
        raise AssertionError('Physical common-boost/RIGHT stress control failed')
    return dict(native_inputs=dict(N=3,Lambda=model.Lambda,UE=model.ue,c=model.c,sigma=model.sigma,radius=model.radius),
        frozen_prepared_npz_sha256=hashlib.sha256(prepared_npz.read_bytes()).hexdigest(),
        frozen_prepared_metric_norm=float(np.vdot(c,gn@c).real),no_filter_or_bank_reset=True,
        SOURCE19_profile_sha256=hashlib.sha256(profile.tobytes()).hexdigest(),
        boost_input_sha256=hashlib.sha256(boost_input.read_bytes()).hexdigest(),
        actual_normal_dimension=model.chart.normal_dim,actual_full_J_dimension=model.chart.raw_dim,
        mean_Y0_reconstruction_error=y_error,mean_cometric_velocity_error=float(np.linalg.norm(vboost-omega*y0)),
        actual_DB_Y0_SOURCE_norm=float(np.linalg.norm(db@psi)),SOURCE_native_jet_refinement=float(np.linalg.norm((coarse-db)@psi)),
        wrong_bare_Pi_mean_stress_norm=wrong_bare,correct_common_boost_mean_stress_norm=cancellation,
        actual_neutral_RIGHT_product_rule_error=right_error,offmean_covariant_velocity_residual=offmean,
        common_phase_SOURCE_function_error=comm_phase,wrong_charged_only_phase_SOURCE_error=wrong_one_sector,
        boost_momentum_norm=float(np.linalg.norm(pboost)),completing_square_energy=boost['completing_square_common_energy'],
        time_connection_at_probe=wp['time_connection'],native_clock=clock,native_Omega_peak=omega,
        field_frame_ENDPOINT_Omega0_phase=common_boost(qp,0.,0.,y0,model.chi)['phase'],
        phase_is_not_a_spin_minus=True,common_energy_is_retained_not_C0_reset=True,
        analytic_identity='[chi Pi Gnn Pi/2,B] boosted first transport -i chi DB[Gnn pboost] cancels -Omega[Gphi,B]=+i Omega DB[Y0] at mean Gnn=I',
        debt=['mean first-transport identity only; SOURCE second jets/Kato/cross-cometric retained independently',
              'native SOURCE finite-difference remainder','off-mean covariant momentum and driven higher graphs',
              'full physical Noether body/gauge/EM/harmonic/source/Root flow remains B-owned',
              'driven bare energy grows O(Omega^2 sqrt(Lambda)); ordinary spectral window is not invariant',
              'native/legal SOFT, hard-mode, parameter and law gaps; no physical2pi/Born/fullFock promotion'])


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root-lemma','prepared-npz','descriptor','boost-input'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();start=time.perf_counter();root=Path(__file__).resolve().parent
    sources={Path(cast(str,m.__file__)).name:Path(cast(str,m.__file__)).read_bytes() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(cast(str,m.__file__)).parent==root and Path(cast(str,m.__file__)).suffix=='.py'}
    result=mean_source_control(args.root_lemma,args.prepared_npz,args.descriptor,args.boost_input)
    result['source_sha256']={k:hashlib.sha256(v).hexdigest() for k,v in sorted(sources.items())}
    result['loaded_sources_unchanged']=all((root/k).read_bytes()==v for k,v in sources.items())
    result['wall_seconds']=time.perf_counter()-start
    print(json.dumps(portable(result),sort_keys=True))


if __name__=='__main__':main()

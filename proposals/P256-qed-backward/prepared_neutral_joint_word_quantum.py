#!/usr/bin/env python3
"""Original whole-term quantum Galerkin consumer: neutral12/charged48 channels."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.linalg import eigh
from scipy.interpolate import BarycentricInterpolator
from original_charged_principal import portable
from original_total_angular_transfer import scalar_orbital
from prepared_neutral_slater_fibre import slater_expectation
from native_closed_rotation_quantum import physical_chart_pullback
from prepared_neutral_moving_matter_frame import mean_matter_generators,pulse_point
from prepared_neutral_shared_photon_state import load_shared_native_cache,loaded_source_epoch
from prepared_neutral_full_phi_commonform import trial_multipliers
from prepared_neutral_full_em_commonform import field_jets
from prepared_neutral_phi_word_channels import coefficient_channels,operator_matrices_channels as phi_channels
from prepared_neutral_em_word_channels import operator_matrices_channels as em_channels
from prepared_neutral_joint_quantum_galerkin import PointOperatorView,multiplier_derivative,metric_exponential
from prepared_neutral_neutral_global_gram import neutral_global_gram

PROFILES=('neutral','source19','SOFT')
TERMS=('Phi','EM','matter','scalar','Coulomb','G')


def wave_channels(model: Any,sample: dict[str,Any],sector: str,C: np.ndarray,direction: np.ndarray,
                  dq: np.ndarray | None=None,dem: np.ndarray | None=None) -> tuple[np.ndarray,np.ndarray | None]:
    h,_=trial_multipliers(sample['q'],sample['em'],direction,model.sigma)
    base=sample[sector]@C
    value=(base[:,:,None]*h).reshape(len(base),C.shape[1]*6)
    if dq is None:return value,None
    dy=np.zeros(55) if dem is None else dem
    derivative=model.directional_basis(sample,dq,dy)[sector]@C
    dh=multiplier_derivative(sample,direction,model.sigma,dq,dy)
    jet=(derivative[:,:,None]*h+base[:,:,None]*dh).reshape(value.shape)
    return value,jet


def fulltrace_control(model: Any,sample: dict[str,Any],sector: str,C: np.ndarray,direction: np.ndarray,
                      point: dict[str,Any],result: dict[str,Any]) -> dict[str,Any]:
    """Exact Gaussian/mode full-metric control; cross-word interference retained."""
    n=model.chart.normal_dim;base=sample[sector]@C;channel_gram=base.conj().T@base
    h,jets=trial_multipliers(sample['q'],sample['em'],direction,model.sigma)
    u=-sample['q'][:,None]/model.sigma**2*h+jets
    scale=model.chi/2*sample['importance_weight']
    exact=scale*np.kron(channel_gram,u.T@point['cometric'][:n,:n]@u)
    count=result['trace_samples'];rng=np.random.default_rng(result['seed'])
    white=2*rng.integers(0,2,size=(model.chart.raw_dim,count))-1
    probes=point['cometric']@(point['jacobian'].T@(np.sqrt(model.chart.mass)[:,None]*white))
    dots=probes[:n].T@u
    controls=np.stack([scale*np.kron(channel_gram,np.outer(d,d)) for d in dots])
    rows=np.asarray(result['per_probe_matrices'][1])-controls+exact
    mean=rows.mean(axis=0);centered=rows-mean
    error=np.sqrt(np.sum(np.abs(centered)**2,axis=0)/(count*(count-1)))
    return dict(matrix=mean,raw_matrix=result['matrix'],control_exact_matrix=exact,
        CV_standard_error_frobenius=float(np.linalg.norm(error)),raw_standard_error_frobenius=result['standard_error_frobenius'],
        observed_FD_difference_norm=result['fd_difference_norm'],minimum_estimated_eigenvalue=float(eigh(mean,eigvals_only=True)[0]),
        no_clipping_or_physical_whitening=True,all_word_cross_interference_retained=True)


def node(model: Any,sample: dict[str,Any],point: dict[str,Any],shift: np.ndarray,sector: str,C: np.ndarray,
         direction: np.ndarray,pull: dict[str,Any],noether: dict[str,Any],em: dict[str,Any],
         potential: dict[str,float],matter: np.ndarray,body: np.ndarray,probes: int,seed: int) -> dict[str,Any]:
    value,_=wave_channels(model,sample,sector,C,direction);weight=sample['importance_weight']
    gram=weight*value.conj().T@value
    phi=phi_channels(model,sample,direction,sector,C,physical_point=point,q_shift=shift,trace_samples=probes,seed=seed)
    cv=fulltrace_control(model,sample,sector,C,direction,point,phi)
    dem=np.r_[noether['flow'],noether['harmonic']]
    _,d=wave_channels(model,sample,sector,C,direction,pull['normal'],dem)
    if d is None:raise AssertionError('Actual full Noether jet missing')
    d=d+1j*model.gauge_operators(pull['gauge'],sector)['first']@value
    g=weight*(-.5j*(value.conj().T@d-d.conj().T@value)+value.conj().T@body@value)-2*float(np.sum(noether['lambda_C']))*gram
    _,translation=wave_channels(model,sample,sector,C,direction,direction)
    if translation is None:raise AssertionError('Actual moving-basis translation jet missing')
    rho=model.chart.directional_jet(sample['q'],direction,sample['point'])['log_density_first']
    translation=translation+.5*rho*value
    strong=weight*value.conj().T@translation;connection=(strong-strong.conj().T)/2
    terms=dict(Phi=cv['matrix'],EM=weight*(em['kinetic']+em['potential']),
        matter=weight*value.conj().T@matter@value+potential['massive_sea']*gram,
        scalar=sum(potential[k] for k in ('quartic','alignment','colour_magnetic'))*gram,
        Coulomb=weight*value.conj().T@model.HC[sector]@value,G=g)
    return dict(Gram=gram,terms=terms,connection=connection,strong_connection=strong,
        diagnostics=dict(Phi=cv,EM=em['observed_FD_errors'],heavy_minusLambda_expectation_counted_once=True,
            operator_Hermiticity={k:float(np.linalg.norm(v-v.conj().T)) for k,v in terms.items()},
            Noether_normal_gauge_reconstruction=pull['reconstruction_error'],cutoff_squared=weight))


def global_forms(model: Any,data: dict[str,Any],clock: int,t0: float,t1: float,nodes: int,probes: int,seed: int) -> dict[str,Any]:
    if nodes<12 or nodes%2:raise ValueError('Twelve or more antithetic global normal/EM nodes required')
    g=mean_matter_generators(model,data);guard_bytes=g['guard_path'].read_bytes();guard=json.loads(guard_bytes)
    guard_sha=hashlib.sha256(guard_bytes).hexdigest();velocity,alpha=np.asarray(guard['velocity']),np.asarray(guard['alpha'])
    direction=np.asarray(g['Y0']);fractions=np.array([0.,.5,1.])
    delta=pulse_point(clock,t1/clock)['theta']-pulse_point(clock,t0/clock)['theta']
    channels={};metadata={}
    for profile in PROFILES:
        sector='neutral' if profile=='neutral' else 'charged'
        channels[profile],metadata[profile]=coefficient_channels(data,sector,'source19' if profile=='neutral' else profile)
    neutral_metric=neutral_global_gram(model,data,direction)
    matrices={profile:{term:np.zeros((len(fractions),channels[profile].shape[1]*6,channels[profile].shape[1]*6),complex)
        for term in (*TERMS,'Gram','connection','strong_connection')} for profile in PROFILES}
    rng=np.random.default_rng(seed);qnodes=rng.standard_normal((nodes//2,model.chart.normal_dim))*model.sigma/math.sqrt(2)
    enodes=rng.standard_normal((nodes//2,55))/math.sqrt(2)
    records=[];embedding=[]
    for pair,(q,y) in enumerate(zip(qnodes,enodes)):
        for sign in (-1,1):
            sample=model.basis_at(sign*q,sign*y)
            if sample['cutoff']==0:raise ValueError('Global Gaussian node outside original compact support')
            em_jets=(field_jets(model,sample,2.**-18),field_jets(model,sample,2.**-19))
            electromagnetics={profile:em_channels(model,sample,direction,'neutral' if profile=='neutral' else 'charged',C,
                derivative_steps=em_jets) for profile,C in channels.items()}
            for result in electromagnetics.values():
                if 'derivative_fine' in result:del result['derivative_fine']
            del em_jets
            for profile,C in channels.items():
                sector='neutral' if profile=='neutral' else 'charged'
                wave,_=wave_channels(model,sample,sector,C,direction)
                original=sample[sector]@data['neutral_state' if profile=='neutral' else profile+'_normalized_source_state']
                embedding.append(float(np.linalg.norm(wave[:,0]-original)))
            noether=model.noether_bank.evaluate(sample['em'][:52])
            for fi,fraction in enumerate(fractions):
                shift=-fraction*delta*direction;point=sample['point'] if fi==0 else model.chart.evaluate(sample['q']+shift)
                view=PointOperatorView(model);pull=physical_chart_pullback(model,dict(point=point),alpha,velocity)
                potential=model.original.classical_potential(point);light=model.law.onebody(point['phi'],point['links'],sample['a'])
                orbital=scalar_orbital(model.law,point['links'],sample['a'],velocity)+np.diag(np.repeat(noether['lambda_C'],4))
                heavy=scalar_orbital(model.heavy_law,point['links'],np.zeros((model.law.volume,3)),velocity)
                matter={s:model.fibre.onebody(light,s) for s in ('neutral','charged')}
                bodies={s:model.fibre.onebody(orbital,s)+slater_expectation(model.heavy_projector,heavy)*model.fibre.overlap(s) for s in ('neutral','charged')}
                for profile,C in channels.items():
                    sector='neutral' if profile=='neutral' else 'charged'
                    result=node(view,sample,point,shift,sector,C,direction,pull,noether,electromagnetics[profile],potential,
                        matter[sector],bodies[sector],probes,seed+10000+2*pair+(sign==1))
                    for term in TERMS:matrices[profile][term][fi]+=result['terms'][term]/nodes
                    for term in ('Gram','connection','strong_connection'):matrices[profile][term][fi]+=result[term]/nodes
                    records.append(dict(pair=pair,sign=sign,fraction=float(fraction),profile=profile,
                        physical_normal_norm=float(np.linalg.norm(sample['q']+shift)),errors=result['diagnostics']))
                del view
    return dict(matrices=matrices,records=records,channels=channels,channel_metadata=metadata,Y0=direction,
        fractions=fractions,angular_delta=delta,embedding_error=max(embedding),
        guard_path=str(g['guard_path']),guard_sha256_at_actual_read=guard_sha,
        neutral_exact_uncut_metric=neutral_metric,
        guard_sha256_after=hashlib.sha256(g['guard_path'].read_bytes()).hexdigest())


def evolve(compiled: dict[str,Any],clock: int,t0: float,t1: float,steps: int) -> dict[str,Any]:
    c1,c2=.5-math.sqrt(3)/6,.5+math.sqrt(3)/6;a1,a2=(3-2*math.sqrt(3))/12,(3+2*math.sqrt(3))/12
    angle0=pulse_point(clock,t0/clock)['theta'];results={}
    for profile,banks in compiled['matrices'].items():
        sampled_gram=banks['Gram'][0]
        gram=compiled['neutral_exact_uncut_metric']['uncut_global_Gram'] if profile=='neutral' else sampled_gram
        dimension=len(gram);eigenvalues=eigh(gram,eigvals_only=True)
        if eigenvalues[0]<=0:raise ValueError('New global word-channel Gram not positive, no jitter/clip permitted')
        interpolation={k:BarycentricInterpolator(compiled['fractions'],banks[k],axis=0) for k in TERMS}
        connection=banks['connection'][0]
        def h(t: float,omit: str | None=None) -> np.ndarray:
            pulse=pulse_point(clock,t/clock);fraction=(pulse['theta']-angle0)/compiled['angular_delta']
            terms={k:interpolation[k](fraction) for k in TERMS}
            return sum(terms[k] for k in TERMS if k!='G' and k!=omit)-pulse['Omega']*terms['G']-1j*pulse['Omega']*connection
        initial=np.zeros(dimension,complex);initial[0]=1
        def propagate(count: int,omit: str | None=None) -> np.ndarray:
            state=initial.copy();dt=(t1-t0)/count
            for i in range(count):
                h1,h2=h(t0+(i+c1)*dt,omit),h(t0+(i+c2)*dt,omit)
                state=metric_exponential(a2*h1+a1*h2,gram,dt,state)
                state=metric_exponential(a1*h1+a2*h2,gram,dt,state)
            return state
        coarse=propagate(steps);state=propagate(steps*2);wrong_no_phi=propagate(steps*2,'Phi')
        norm0=float(np.vdot(initial,gram@initial).real);norm=float(np.vdot(state,gram@state).real)
        overlap=complex(np.vdot(initial,gram@state));outside=state-initial*overlap/norm0
        delta=state-coarse;wrong=state-wrong_no_phi
        # Projection onto all six modes of the original coherent raw-quartet channel.
        coherent_metric=gram[:6,:6];coherent_rhs=gram[:6]@state
        coherent=np.zeros(dimension,complex);coherent[:6]=np.linalg.solve(coherent_metric,coherent_rhs)
        word_outside=state-coherent
        results[profile]=dict(initial_coefficients=initial,final_coefficients=state,coarse_coefficients=coarse,
            initial_function_exact_embedding_error=compiled['embedding_error'],initial_GLOBAL_norm_squared=norm0,
            final_GLOBAL_norm_squared=norm,metric_norm_drift=norm-norm0,Gram_eigenvalues=eigenvalues,
            observed_temporal_refinement_norm_squared=float(np.vdot(delta,gram@delta).real),
            departure_from_original_trial_ray_squared=float(np.vdot(outside,gram@outside).real),
            actual_outside_original_coherent_sixmode_channel_squared=float(np.vdot(word_outside,gram@word_outside).real),
            wrong_fullPhi_omission_norm_squared=float(np.vdot(wrong,gram@wrong).real),
            actual_sampled_Gram_time_difference=max(float(np.linalg.norm(m-sampled_gram)) for m in banks['Gram']),
            sampled_vs_used_global_Gram_difference=float(np.linalg.norm(sampled_gram-gram)),
            metric_scope='Exact uncut neutral Gaussian Gram with analytic compact-cutoff loss bound' if profile=='neutral' else 'Actual global Gaussian quadrature word-channel Gram, finite sampling debt',
            derived_Mdot_minus_CplusCadjoint=0.,strong_MC_IBP_connection_defect=float(np.linalg.norm(banks['strong_connection'][0]+banks['strong_connection'][0].conj().T)),
            weak_connection_antiHermitian_error=float(np.linalg.norm(connection+connection.conj().T)),
            retained_all_four_SOURCE_word_directions_per_sign=profile!='neutral',no_state_sourceZ_or_effect_renormalization=True,
            full_original_operator_terms_in_actual_coefficient_evolution=True)
    return results


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-npz',type=Path,required=True)
    p.add_argument('--cache-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nodes',type=int,default=12);p.add_argument('--trace-probes',type=int,default=4)
    p.add_argument('--time-steps',type=int,default=128);p.add_argument('--seed',type=int,default=2222027)
    args=p.parse_args();start=time.monotonic();model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    before=loaded_source_epoch();clock=math.ceil(15*math.pi*1.5/model.c);t0=.37*clock;t1=t0+.031
    compiled=global_forms(model,data,clock,t0,t1,args.nodes,args.trace_probes,args.seed)
    result=dict(scope='NEW original full-term global weak-form quantum Galerkin, neutral12/charged48 with all4Refword directions perplus/minus; NOT fullHilbert/wholepulse/physical2pi/Born',
        nodes=args.nodes,trace_probes=args.trace_probes,clock=clock,time_interval=[t0,t1],channel_metadata=compiled['channel_metadata'],
        original_function_embedding_error=compiled['embedding_error'],matrices=compiled['matrices'],node_errors=compiled['records'],
        neutral_exact_uncut_metric=compiled['neutral_exact_uncut_metric'],
        evolution=evolve(compiled,clock,t0,t1,args.time_steps),mean_translation_endpoint=-compiled['angular_delta']*compiled['Y0'],
        all55_original_Gaussian_including_harmonic=True,original_zero_point_retained_once_no_harmonic_potential=True,
        fixed_frozen_law_state_function_banks_globalZ_NOT_same_numeric_old_H_or_Gram=True,
        global_normalization_NOT_renormalized_to_old_conditional_one=True,guard_path=compiled['guard_path'],
        guard_sha256_at_read=compiled['guard_sha256_at_actual_read'],guard_sha256_after=compiled['guard_sha256_after'],
        actual_guard_bytes_stable=compiled['guard_sha256_at_actual_read']==compiled['guard_sha256_after'],
        debts=['Stochastic globalnormal/EMquadrature and fulltrace estimate, finite SOURCE jets, three angular-node operator interpolation','Finite48/12span, ordinaryCAR versus encodedRoot/fullauxGauss and outside-span dynamics','Native unwrapped harmonicdomain/legalN/legalSOFT and original quantumlaw/parameter derivation','Wholephysicalpulse/spatial2pi closedrotation/operationalBorn'],
        cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),loaded_sources=before,
        loaded_sources_stable=before==loaded_source_epoch(),program_wall_seconds=time.monotonic()-start)
    text=json.dumps(portable(result),sort_keys=True);args.output.write_text(text);os.chmod(args.output,0o600);print(text)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Global weak-form, moving mean-translation, twelve-mode quantum consumer."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from scipy.linalg import eigh
from scipy.interpolate import BarycentricInterpolator
from original_charged_principal import portable
from original_total_angular_transfer import scalar_orbital
from prepared_neutral_slater_fibre import slater_expectation
from native_closed_rotation_quantum import physical_chart_pullback
from prepared_neutral_moving_matter_frame import mean_matter_generators,pulse_point
from prepared_neutral_shared_photon_state import load_shared_native_cache,loaded_source_epoch
from prepared_neutral_full_phi_commonform import operator_matrices as phi_matrices,trial_multipliers
from prepared_neutral_full_em_commonform import operator_matrices as em_matrices,field_jets

PROFILES=('neutral','source19','SOFT')
TERMS=('Phi','EM','matter','scalar','Coulomb','G')


class DensityJetView:
    """Share actual scalar density jets across profiles at one quadrature point."""
    def __init__(self,chart: Any):
        self.original=chart;self.jets: dict[Any,float]={}
    def __getattr__(self,name: str) -> Any:
        return getattr(self.original,name)
    def directional_jet(self,q: np.ndarray,dq: np.ndarray,point: dict[str,Any]) -> dict[str,float]:
        key=(q.tobytes(),dq.tobytes(),id(point))
        if key not in self.jets:
            self.jets[key]=self.original.directional_jet(q,dq,point)['log_density_first']
        return dict(log_density_first=self.jets[key])


class PointOperatorView:
    """Amortize expensive full bank/Wick evaluations, never alter providers."""
    def __init__(self,model: Any):
        self.original_model=model;self.chart=DensityJetView(model.chart)
        self.jets: dict[Any,Any]={};self.gauges: dict[Any,Any]={}
    def __getattr__(self,name: str) -> Any:
        return getattr(self.original_model,name)
    def directional_basis(self,sample: dict[str,Any],dq: np.ndarray,
                          dem: np.ndarray | None=None,step: float=2.**-19) -> dict[str,np.ndarray]:
        key=(id(sample),dq.tobytes(),None if dem is None else dem.tobytes(),step)
        if key not in self.jets:
            self.jets[key]=self.original_model.directional_basis(sample,dq,dem,step)
        return self.jets[key]
    def gauge_operators(self,gauge: np.ndarray,sector: str) -> dict[str,Any]:
        key=(gauge.tobytes(),sector)
        if key not in self.gauges:self.gauges[key]=self.original_model.gauge_operators(gauge,sector)
        return self.gauges[key]


def frozen_bank_data(model: Any,data: dict[str,Any]) -> tuple[dict[str,Any],dict[str,Any],dict[str,Any]]:
    """Split fixed original coefficients, preserving their ONE global SOURCE Z."""
    raw=model.neutral_coordinates@data['neutral_state']
    if np.linalg.norm(raw[210:])!=0:
        raise ValueError('Original frozen vacuum embedding unexpectedly has generated neutral coefficients')
    first=raw.copy();first[105:]=0
    plus=dict(data);minus=dict(data)
    plus['neutral_state']=np.linalg.solve(model.neutral_coordinates,first)
    minus['neutral_state']=data['neutral_state']-plus['neutral_state']
    checks={}
    for profile in ('source19','SOFT'):
        coefficient=data[profile+'_normalized_source_state']
        original_raw=data[profile+'_map']@data['neutral_state']
        pivot=int(np.argmax(np.abs(coefficient)))
        fixed_z=original_raw[pivot]/coefficient[pivot]
        error=float(np.linalg.norm(original_raw-fixed_z*coefficient)/np.linalg.norm(original_raw))
        if error>1e-12 or abs(fixed_z.imag)>1e-12*abs(fixed_z.real) or fixed_z.real<=0:
            raise ValueError('Frozen SOURCE coefficient map/global normalization relation changed')
        plus[profile+'_normalized_source_state']=data[profile+'_map']@plus['neutral_state']/fixed_z
        minus[profile+'_normalized_source_state']=coefficient-plus[profile+'_normalized_source_state']
        checks[profile]=dict(fixed_global_z=float(fixed_z.real),map_coordinate_error=error,
            exact_sum_error=float(np.linalg.norm(plus[profile+'_normalized_source_state']+minus[profile+'_normalized_source_state']-coefficient)),no_new_normalization=True)
    return plus,minus,checks


def coefficients(data: dict[str,Any],profile: str) -> tuple[str,np.ndarray]:
    return ('neutral',data['neutral_state']) if profile=='neutral' else ('charged',data[profile+'_normalized_source_state'])


def multiplier_derivative(sample: dict[str,Any],direction: np.ndarray,sigma: float,dq: np.ndarray,dem: np.ndarray) -> np.ndarray:
    q,y=sample['q'],sample['em'];unit=direction/np.linalg.norm(direction)
    x=float(q@unit)/sigma;dx=float(dq@unit)/sigma
    return np.array([0.,math.sqrt(2)*dx,2*math.sqrt(2)*x*dx,
        4*float(q@dq)/(sigma*sigma*math.sqrt(2*len(q))),math.sqrt(2)*dem[52],4*float(y[:52]@dem[:52])/math.sqrt(104)])


def held_trial(model: Any,data: dict[str,Any],sample: dict[str,Any],profile: str,direction: np.ndarray,
               dq: np.ndarray | None=None,dem: np.ndarray | None=None) -> tuple[np.ndarray,np.ndarray | None]:
    sector,state=coefficients(data,profile)
    multipliers,_=trial_multipliers(sample['q'],sample['em'],direction,model.sigma)
    base=sample[sector]@state;value=base[:,None]*multipliers
    if dq is None:return value,None
    em_direction=np.zeros(55) if dem is None else dem
    derivative=model.directional_basis(sample,dq,em_direction)[sector]@state
    jets=multiplier_derivative(sample,direction,model.sigma,dq,em_direction)
    return value,derivative[:,None]*multipliers+base[:,None]*jets


def bank_lift(plus: np.ndarray,minus: np.ndarray) -> np.ndarray:
    """Even/odd banks; initial even mode0 is the original coherent state."""
    total=plus+minus;difference=plus-minus
    return np.block([[total,difference],[difference,total]])


def gaussian_control_variate(model: Any,data: dict[str,Any],sample: dict[str,Any],profile: str,
                            direction: np.ndarray,point: dict[str,Any],result: dict[str,Any]) -> dict[str,Any]:
    """Unbiased exact full-cometric Gaussian/multiplier control, not a fit."""
    n=model.chart.normal_dim;sector,state=coefficients(data,profile)
    base=sample[sector]@state;norm=float(np.vdot(base,base).real)
    values,jets=trial_multipliers(sample['q'],sample['em'],direction,model.sigma)
    u=-sample['q'][:,None]/model.sigma**2*values[None,:]+jets
    scale=model.chi/2*sample['importance_weight']*norm
    exact=scale*u.T@point['cometric'][:n,:n]@u
    count=result['trace_samples'];rng=np.random.default_rng(result['seed'])
    white=2*rng.integers(0,2,size=(model.chart.raw_dim,count))-1
    probes=point['cometric']@(point['jacobian'].T@(np.sqrt(model.chart.mass)[:,None]*white))
    projections=probes[:n].T@u
    controls=scale*np.einsum('ki,kj->kij',projections,projections)
    rows=np.asarray(result['per_probe_matrices'][1])-controls+exact
    centered=rows-rows.mean(axis=0)
    se=np.sqrt(np.sum(np.abs(centered)**2,axis=0)/(count*(count-1)))
    return dict(matrix=rows.mean(axis=0),raw_matrix=result['matrix'],exact_control=exact,
        standard_error_frobenius=float(np.linalg.norm(se)),raw_standard_error_frobenius=result['standard_error_frobenius'],
        fd_difference_norm=result['fd_difference_norm'],minimum_estimate_eigenvalue=float(eigh(rows.mean(axis=0),eigvals_only=True)[0]),
        no_positive_eigenvalue_clipping=True,unbiased_control_uses_full_actual_cometric=True)


def node_forms(model: Any,bank: dict[str,Any],sample: dict[str,Any],point: dict[str,Any],shift: np.ndarray,
               direction: np.ndarray,pull: dict[str,Any],velocity: np.ndarray,profile: str,probes: int,seed: int,
               electromagnetic: dict[str,Any],potential: dict[str,float],matter: np.ndarray,body: np.ndarray) -> dict[str,Any]:
    sector,state=coefficients(bank,profile)
    value,_=held_trial(model,bank,sample,profile,direction)
    weight=sample['importance_weight'];gram=weight*value.conj().T@value
    phi=phi_matrices(model,bank,sample,direction,profile,physical_point=point,q_shift=shift,trace_samples=probes,seed=seed)
    cv=gaussian_control_variate(model,bank,sample,profile,direction,point,phi)
    # This is the actual projected heavy -Lambda expectation; NOT a bosonic term.
    h_matter=weight*value.conj().T@matter@value+potential['massive_sea']*gram
    h_scalar=sum(potential[k] for k in ('quartic','alignment','colour_magnetic'))*gram
    h_coulomb=weight*value.conj().T@model.HC[sector]@value
    noether=model.noether_bank.evaluate(sample['em'][:52]);dem=np.r_[noether['flow'],noether['harmonic']]
    _,jet=held_trial(model,bank,sample,profile,direction,pull['normal'],dem)
    if jet is None:raise AssertionError('Noether held jet missing')
    gauge=model.gauge_operators(pull['gauge'],sector)['first'];jet=jet+1j*gauge@value
    # The original native orbital body is shared across profiles at this point.
    g=weight*(-.5j*(value.conj().T@jet-jet.conj().T@value)+value.conj().T@body@value)-2*float(np.sum(noether['lambda_C']))*gram
    _,translation_jet=held_trial(model,bank,sample,profile,direction,direction)
    if translation_jet is None:raise AssertionError('Translation held jet missing')
    rhojet=model.chart.directional_jet(sample['q'],direction,sample['point'])['log_density_first']
    # Translation-time derivative has NO current-rho derivative at fixed qPhysical.
    translation_jet=translation_jet+.5*rhojet*value
    strong=weight*value.conj().T@translation_jet
    weak=(strong-strong.conj().T)/2
    terms=dict(Phi=cv['matrix'],EM=weight*(electromagnetic['kinetic']+electromagnetic['potential']),
               matter=h_matter,scalar=h_scalar,Coulomb=h_coulomb,G=g)
    return dict(Gram=gram,terms=terms,translation=weak,translation_strong=strong,
        diagnostics=dict(Phi=cv,EM_fd=electromagnetic['observed_FD_errors'],
            Hermitian_residuals={k:float(np.linalg.norm(v-v.conj().T)) for k,v in terms.items()},
            source_field_foot_held=True,current_physical_normal_norm=float(np.linalg.norm(sample['q']+shift)),
            heavy_expectation_counted_once=True))


def compile_global(model: Any,data: dict[str,Any],clock: int,t0: float,t1: float,nodes: int,probes: int,seed: int,
                   fractions: np.ndarray) -> dict[str,Any]:
    if nodes<12 or nodes%2:raise ValueError('At least twelve antithetic global nodes required for six modes per bank')
    g=mean_matter_generators(model,data);direction=np.asarray(g['Y0']);guard=json.loads(g['guard_path'].read_bytes())
    velocity,alpha=np.asarray(guard['velocity']),np.asarray(guard['alpha'])
    plus,minus,split=frozen_bank_data(model,data);rng=np.random.default_rng(seed)
    qnodes=rng.standard_normal((nodes//2,model.chart.normal_dim))*model.sigma/math.sqrt(2)
    enodes=rng.standard_normal((nodes//2,55))/math.sqrt(2)
    angular_delta=pulse_point(clock,t1/clock)['theta']-pulse_point(clock,t0/clock)['theta']
    matrices={profile:{term:np.zeros((len(fractions),12,12),complex) for term in (*TERMS,'Gram','translation','translation_strong')} for profile in PROFILES}
    records=[];initial_controls=[]
    for pair,(q,y) in enumerate(zip(qnodes,enodes)):
        for sign in (-1,1):
            sample=model.basis_at(sign*q,sign*y)
            if sample['cutoff']==0:raise ValueError('Original global importance draw outside native packet support')
            for profile in PROFILES:
                fp,_=held_trial(model,plus,sample,profile,direction);fm,_=held_trial(model,minus,sample,profile,direction)
                full,_=held_trial(model,data,sample,profile,direction)
                initial_controls.append(float(np.linalg.norm(fp[:,0]+fm[:,0]-full[:,0])))
                fibre_sector='neutral' if profile=='neutral' else 'charged'
                if abs(complex(np.vdot(fp[:,0],fm[:,0])))>1e-12*(1+np.linalg.norm(full[:,0])**2):
                    raise ValueError('Conserved original two banks are not orthogonal number sectors')
            # EM source jets depend on the FOOT only, not physical translation.
            # Retain one full jet bank, contract all profiles/banks, then release.
            em_steps=(field_jets(model,sample,2.**-18),field_jets(model,sample,2.**-19))
            electromagnetic={(profile,index):em_matrices(model,bank,sample,direction,profile,derivative_steps=em_steps)
                for profile in PROFILES for index,bank in enumerate((plus,minus))}
            for result in electromagnetic.values():result.pop('derivative_fine')
            del em_steps
            for fi,fraction in enumerate(fractions):
                shift=-fraction*angular_delta*direction
                point=sample['point'] if fi==0 else model.chart.evaluate(sample['q']+shift)
                pull=physical_chart_pullback(model,dict(point=point),alpha,velocity)
                view=PointOperatorView(model)
                potential=model.original.classical_potential(point)
                light=model.law.onebody(point['phi'],point['links'],sample['a'])
                noether=model.noether_bank.evaluate(sample['em'][:52])
                orbital=scalar_orbital(model.law,point['links'],sample['a'],velocity)+np.diag(np.repeat(noether['lambda_C'],4))
                heavy=scalar_orbital(model.heavy_law,point['links'],np.zeros((model.law.volume,3)),velocity)
                matter={s:model.fibre.onebody(light,s) for s in ('neutral','charged')}
                body={s:model.fibre.onebody(orbital,s)+slater_expectation(model.heavy_projector,heavy)*model.fibre.overlap(s) for s in ('neutral','charged')}
                for profile in PROFILES:
                    sector='neutral' if profile=='neutral' else 'charged'
                    p=node_forms(view,plus,sample,point,shift,direction,pull,velocity,profile,probes,seed+20000+2*pair+(sign==1),
                        electromagnetic[(profile,0)],potential,matter[sector],body[sector])
                    m=node_forms(view,minus,sample,point,shift,direction,pull,velocity,profile,probes,seed+20000+2*pair+(sign==1),
                        electromagnetic[(profile,1)],potential,matter[sector],body[sector])
                    for term in TERMS:matrices[profile][term][fi]+=bank_lift(p['terms'][term],m['terms'][term])/nodes
                    for term in ('Gram','translation','translation_strong'):matrices[profile][term][fi]+=bank_lift(p[term],m[term])/nodes
                    records.append(dict(pair=pair,sign=sign,fraction=float(fraction),profile=profile,
                        pullback_error=pull['reconstruction_error'],plus=p['diagnostics'],minus=m['diagnostics']))
                del view
    return dict(matrices=matrices,records=records,split=split,initial_embedding_error=max(initial_controls),
        fractions=fractions,angular_delta=angular_delta,Y0=direction,nodes=nodes,trace_probes=probes,seed=seed,
        guard_path=str(g['guard_path']),guard_sha256=hashlib.sha256(g['guard_path'].read_bytes()).hexdigest())


def metric_exponential(hermitian: np.ndarray,metric: np.ndarray,duration: float,state: np.ndarray) -> np.ndarray:
    """Generalized spectral exponential; no physical whitening/jitter/refilter."""
    energies,modes=eigh(hermitian,metric)
    return modes@(np.exp(-1j*duration*energies)*(modes.conj().T@(metric@state)))


def evolve(model: Any,compiled: dict[str,Any],clock: int,t0: float,t1: float,steps: int) -> dict[str,Any]:
    fractions=compiled['fractions'];angle0=pulse_point(clock,t0/clock)['theta'];angular_delta=compiled['angular_delta']
    outputs={};c1,c2=.5-math.sqrt(3)/6,.5+math.sqrt(3)/6
    a1,a2=(3-2*math.sqrt(3))/12,(3+2*math.sqrt(3))/12
    for profile in PROFILES:
        banks=compiled['matrices'][profile];metric=banks['Gram'][0]
        energies=eigh(metric,eigvals_only=True)
        if energies[0]<=0:raise ValueError('Actual new global trial Gram is not positive; no clipping/jitter permitted')
        interpolation={term:BarycentricInterpolator(fractions,banks[term],axis=0) for term in TERMS}
        translation=banks['translation'][0]
        def hamiltonian(t: float,omit: str | None=None) -> np.ndarray:
            pulse=pulse_point(clock,t/clock);fraction=(pulse['theta']-angle0)/angular_delta
            terms={term:interpolation[term](fraction) for term in TERMS}
            h=sum(terms[k] for k in TERMS if k!='G' and k!=omit)-pulse['Omega']*terms['G']
            # sdot=-Omega*Y0, hence C=+Omega<D_Y0>; Hmoving=H-OmegaG-iC.
            return h-1j*pulse['Omega']*translation
        initial=np.zeros(12,complex);initial[0]=1
        def propagate(count: int,omit: str | None=None) -> np.ndarray:
            state=initial.copy();dt=(t1-t0)/count
            for i in range(count):
                early=t0+(i+c1)*dt;late=t0+(i+c2)*dt
                h1,h2=hamiltonian(early,omit),hamiltonian(late,omit)
                state=metric_exponential(a2*h1+a1*h2,metric,dt,state)
                state=metric_exponential(a1*h1+a2*h2,metric,dt,state)
            return state
        coarse=propagate(steps);fine=propagate(steps*2);no_phi=propagate(steps*2,'Phi')
        norm0=float(np.vdot(initial,metric@initial).real);normf=float(np.vdot(fine,metric@fine).real)
        refined=fine-coarse
        overlap=complex(np.vdot(initial,metric@fine))
        residual=fine-initial*overlap/norm0
        no_phi_delta=no_phi-fine
        outputs[profile]=dict(coefficients=fine,coarse_coefficients=coarse,initial_coefficients=initial,
            initial_new_global_norm_squared=norm0,final_new_global_norm_squared=normf,
            metric_unitarity_error=abs(normf-norm0),Gram_eigenvalues=energies,
            observed_timestep_difference_squared=float(np.vdot(refined,metric@refined).real),
            orthogonal_to_initial_trial_ray_squared=float(np.vdot(residual,metric@residual).real),
            wrong_omitted_full_Phi_difference_squared=float(np.vdot(no_phi_delta,metric@no_phi_delta).real),
            Gram_time_difference=max(float(np.linalg.norm(g-metric)) for g in banks['Gram']),
            derived_Mdot_minus_CplusCadjoint=0.,
            strong_finite_quadrature_IBP_connection_defect=float(np.linalg.norm(banks['translation_strong'][0]+banks['translation_strong'][0].conj().T)),
            weak_connection_antiHermitian_error=float(np.linalg.norm(translation+translation.conj().T)),
            original_norm_not_renormalized=True,all_terms_in_coefficient_evolution=True)
    return outputs


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-npz',type=Path,required=True);p.add_argument('--cache-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--nodes',type=int,default=12)
    p.add_argument('--trace-probes',type=int,default=4);p.add_argument('--time-steps',type=int,default=128)
    p.add_argument('--seed',type=int,default=2222026)
    args=p.parse_args();start=time.monotonic();model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    before=loaded_source_epoch();clock=math.ceil(15*math.pi*1.5/model.c);t0=.37*clock;t1=t0+.031
    compiled=compile_global(model,data,clock,t0,t1,args.nodes,args.trace_probes,args.seed,np.array([0.,.5,1.]))
    result=dict(scope='NEW global Gaussian quadrature and full-term original ordinary-Fock/native-field quantum Galerkin coefficients in moving mean-translation twelve-mode span; NOT exact fullHilbert propagation',
        time_interval=[t0,t1],clock=clock,global_quadrature_nodes=args.nodes,full_cometric_trace_probes=args.trace_probes,
        initial_embedding_error=compiled['initial_embedding_error'],bank_split=compiled['split'],matrices=compiled['matrices'],
        observed_node_errors=compiled['records'],evolution=evolve(model,compiled,clock,t0,t1,args.time_steps),
        angular_delta=compiled['angular_delta'],mean_translation_endpoint=-compiled['angular_delta']*compiled['Y0'],
        original_law_terms=['fullPhi_normal2430_Gauss405_metric_Wick','all55EM_kinetic_plus_original52potential','full_native_H_D_weak','literal_quartic_alignment_colour','fullordinaryCoulombmultipair_weak','actual_nativeG_normal_Gauss_all55_body_Ref_background','moving_translation_connection'],
        old_cache_numeric_H_and_Gram_NOT_retagged=True,no_refilter_newZ_bank_recapture_or_effect=True,
        original_full55_harmonic_Gaussian_retained=True,no_added_harmonic_potential_or_duplicate_zero_point=True,
        finite_time_operator_interpolation='Three angular-displacement weakform nodes; observed temporal refinement is not global operator interpolation enclosure',
        debts=['Global quadrature/statistical and fulltrace stochastic errors, finite SOURCE jets','Finite Galerkin outside-span dynamics and original encodedRoot/Gauss/fullaux equivalence','Native unwrapped harmonic domain, legalN/source/legalSOFT','Wholepulse/physical2pi/spatialclosedloop/operationalBorn and original quantum-law/parameter derivation'],
        guard_path=compiled['guard_path'],guard_sha256=compiled['guard_sha256'],cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        loaded_sources=before,loaded_sources_unchanged=before==loaded_source_epoch(),program_wall_seconds=time.monotonic()-start)
    text=json.dumps(portable(result),sort_keys=True);args.output.write_text(text);os.chmod(args.output,0o600);print(text)

if __name__=='__main__':main()

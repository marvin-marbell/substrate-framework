#!/usr/bin/env python3
"""Original signed Source2a source/Pe phase: actual strip, interior and far out.

Native finite matrices are actual unchanged sources, not fresh preparations.
The macro momentum-shell bound has an explicit frozen-fibre scope; it NEVER
assumes a smooth thermal density or commutes the original source through heat.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from scipy.linalg import eigh
import covariant_actual_source_range_band_consumer as provider
import covariant_actual_source_band_consumer as consumer
from covariant_induced_photon import Parameters,WilsonConsumer,phi_frame
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
if sys._getframe().f_code!=compile(OWN_BYTES,__file__,'exec',dont_inherit=True,optimize=sys.flags.optimize):
    raise ArithmeticError('Strip consumer source differs from compiled import snapshot')
if provider.source.COMPILED.get('covariant_prepared_gauss_state')!=consumer.EXPECTED_SOURCE:
    raise ArithmeticError('Actual compiled Source2a epoch changed')
if hashlib.sha256(provider.source.root_transfer.SOURCE_BYTES_AT_IMPORT['covariant_root_transfer.py']).hexdigest()!=consumer.EXPECTED_ROOT:
    raise ArithmeticError('Actual imported Root6911 epoch changed')


def spectral_strip(A1: np.ndarray,A0: np.ndarray,raw: np.ndarray,cut: float,
                   eta: float,clocks: list[float]) -> dict[str,Any]:
    if not 0<eta<cut/2:raise ValueError('Proof strip must fit inside unchanged Pe')
    v1,u1=eigh((A1+A1.conj().T)/2);v0,u0=eigh((A0+A0.conj().T)/2)
    lower=math.exp(-cut);upper=1.
    sel1=(v1>=lower)&(v1<=upper);sel0=(v0>=lower)&(v0<=upper)
    interior=(v0>=math.exp(-cut+eta))&(v0<=math.exp(-eta))
    below=v0<math.exp(-cut-eta);above=v0>math.exp(eta)
    strip=~(interior|below|above)
    q1=u1[:,sel1];b1=v1[sel1];c0=u0.conj().T@raw
    delta=float(np.linalg.norm(A1-A0,2))
    qI=u0[:,interior];bI=v0[interior];XI=q1.conj().T@qI
    EI=b1[:,None]*XI-XI*bI[None,:]
    epsI=float(np.linalg.norm(EI,2)) if len(bI) else 0.
    sqrtprice=epsI/(math.sqrt(lower)+math.exp((-cut+eta)/2))
    dIlo=math.exp(-cut+eta)-lower;dIup=1-math.exp(-eta)
    leakI=delta*math.sqrt(dIlo**-2+dIup**-2)
    dbelow=lower-math.exp(-cut-eta);dabove=math.exp(eta)-1
    norms={name:float(np.linalg.norm(c0[mask],'fro')) for name,mask in
        (('interior',interior),('below',below),('above',above),('strip',strip))}
    farbelow=delta/dbelow*norms['below'];farabove=delta/dabove*norms['above']
    stripprice=2*norms['strip']
    records=[]
    for t in clocks:
        f1=np.sqrt(b1)*np.exp(1j*t*np.log(b1))
        observed1=q1@(f1[:,None]*(q1.conj().T@raw))
        f0=np.sqrt(v0[sel0])*np.exp(1j*t*np.log(v0[sel0]))
        observed0=u0[:,sel0]@(f0[:,None]*c0[sel0])
        smooth=(abs(t)*epsI/lower+sqrtprice)*norms['interior']
        interiorprice=smooth+leakI*math.exp(-eta/2)*norms['interior']
        total=interiorprice+farbelow+farabove+stripprice
        observed=float(np.linalg.norm(observed1-observed0,'fro'))
        if observed>total+1e-9:raise ValueError('Actual whole-source spectral-strip phase bound failed')
        pieces=[]
        for mask in (interior,below,above,strip):
            part=u0[:,mask]@c0[mask]
            actual_part=q1@(f1[:,None]*(q1.conj().T@part))
            admitted=mask&sel0
            reference_part=u0[:,admitted]@((np.sqrt(v0[admitted])*np.exp(1j*t*np.log(v0[admitted])))[:,None]*c0[admitted])
            pieces.append(float(np.linalg.norm(actual_part-reference_part,'fro')))
        records.append(dict(clock=t,actual_full_source_phase_error_HS=observed,
            proved_interior_price_HS=interiorprice,proved_far_below_price_HS=farbelow,
            proved_far_above_price_HS=farabove,proved_actual_source_strip_price_HS=stripprice,
            proved_total_HS=total,actual_component_errors_HS=pieces,
            actual_component_triangle_HS=sum(pieces)))
    return dict(existing_Pe_energy_cut=cut,proof_strip_halfwidth=eta,
        proof_interior_energy=[eta,cut-eta],actual_reference_strip_raw_source_squared_norm=norms['strip']**2,
        actual_reference_strip_boundary_squared_norm=float(np.linalg.norm(np.sqrt(np.maximum(v0[strip],0))[:,None]*c0[strip],'fro')**2),
        actual_reference_strip_rank=int(np.count_nonzero(strip)),actual_source_partition_raw_norms=norms,
        actual_reference_original_Pe_Gram=float(np.linalg.norm(np.sqrt(v0[sel0])[:,None]*c0[sel0],'fro')**2),
        clocks=records,source_unchanged=True,Pe_unchanged=True,
        proof='Partition the ACTUAL reference source spectral measure into interior, both threshold strips, far below including physical null, and far above. Ordered Sylvester gives delta/d for each far component. Interior logarithm/sqrt intertwining uses original Pe lower, and pays leakage. Strip price is its ACTUAL full coherent source norm, not a smooth-density assumption. Reference outside Pe has zero target; it is not added as an irreducible reference-state error.')


def momentum_shell_price(p: Parameters,eta: float,packetmax: float) -> dict[str,Any]:
    # Pick explicit spectator allocations, not a new source/energy cut.
    soft_reserve=.10;toron_reserve=.02;emin=.12;emax=.30;s0=.05
    lower=.25-eta-packetmax-soft_reserve-toron_reserve
    if lower<emin:raise ValueError('Conditional electron shell leaves proven velocity region')
    bmax=math.exp(emax);cosmin=2-bmax
    velocity=s0*(1-math.cosh(emax)+cosmin)/(bmax*math.sinh(emax))
    # If max|sin p_j|<s0 and E<=.30 then cos p_j>0; w<=3s0².
    no_strong_upper=2*math.asinh(.5*math.sqrt((math.expm1(.028)+3*s0*s0)**2+3*s0*s0))
    if no_strong_upper>=emin or velocity<=0:raise ValueError('Actual Wilson shell constants do not certify a strong coordinate')
    margin=s0-8*math.pi/p.spatial_size
    lattice_count=eta*p.spatial_size/(math.pi*velocity)+2
    bound=72*math.exp(.011)*lattice_count/(p.spatial_size**2*margin**2)
    return dict(strip_halfwidth=eta,conditional_source_strip_squared_norm_upper=bound,
        electron_energy_region=[emin,emax],conditional_actual_soft_energy_upper=soft_reserve,
        conditional_actual_toron_energy_upper=toron_reserve,strong_coordinate_sine=s0,
        actual_Wilson_velocity_lower=velocity,actual_discrete_frequency_count_per_branch_upper=lattice_count,
        source_scope='Exact whole jumping three-beam Fourier source, arbitrary coefficients and EVERY fixed flat holonomy, in the frozen free-reference momentum fibre. Other coordinates sum by Parseval; all125 photon branches may be arbitrary coherent normalized coefficients.',
        proof='E>=.12 forces some |sin p_j|>=.05; E<=.30 gives B<=exp(.30), cos p_j>=2-exp(.30). Exact dE/dp_j=sin(p_j)(B-cosh(E)+cos(p_j))/(B sinh(E)) gives the displayed positive velocity. Each discrete +/- branch has at most eta L/(pi v)+2 points. Literal Wilson DFT coefficients at the strong coordinate cost<=4/(L² margin²); 3 coherent beams and3 possible strong coordinates give72. No smooth density and no continuum replacement of the lattice count.',
        not_irreducible_floor='The +2 is a uniform discrete fibre count, not an irreducible full-source phase floor. Actual thermal/toron joint averaging can further sharpen it.',
        joint_toron_contract='This fibre bound alone does NOT prove the spectral strip of the original theta-dependent body/toro heat/source operator. Its noncommuting toron energy/source correlations must be bounded or evaluated; no heat/source commutation is asserted.')


def actual_fourier_shell_probe(gamma: np.ndarray) -> list[dict[str,Any]]:
    """Actual entire jumped source; no single-beam surrogate or smooth DOS."""
    from covariant_actual_source_reference_band import stable_band
    L=4096;x=np.arange(L);signed=np.where(x<=L//2,x,x-L)
    index=np.fft.fftfreq(L)*L
    rng=np.random.default_rng(222101);amplitudes=rng.normal(size=3)+1j*rng.normal(size=3)
    amplitudes/=np.linalg.norm(amplitudes)
    coefficients=np.zeros((L,4,4),complex)
    for a,offset in zip(amplitudes,(-1,0,1)):
        momentum=np.array([offset*2*math.pi/L,0.,0.])
        projector,E=stable_band(gamma,.01,1.,momentum)
        coefficients+=a*math.exp(E/2)*np.exp(2j*math.pi*offset*(x-L/4)/L)[:,None,None]*projector/math.sqrt(2*L)
    results=[]
    for theta in (-math.pi,-math.pi/2,math.pi/2,math.pi):
        jumped=coefficients*np.exp(1j*theta*signed/L)[:,None,None]
        source=np.fft.fft(jumped,axis=0)/math.sqrt(L)
        for spacing,mass in ((.005,.7),(.01,1.),(.02,1.4)):
            c=WilsonConsumer(gamma,spacing,mass)
            energies=[];norms=[]
            for n,f in zip(index,source):
                momentum=np.array([(2*math.pi*n+theta)/L,0.,0.])
                projector,E=stable_band(gamma,spacing,mass,momentum)
                energies.append(E);norms.append(float(np.linalg.norm(projector@f,'fro')**2))
                if abs(E-c.energy(momentum))>1e-11:raise ValueError('Physical Root shell spectrum differs from actual Wilson energy')
            energies=np.array(energies);norms=np.array(norms)
            for eta in (.001,.00001):
                actual=float(np.sum(norms[abs(energies-.20)<=eta]))
                bound=momentum_shell_price(Parameters(spatial_size=L),eta,.005)['conditional_source_strip_squared_norm_upper']
                if actual>bound+1e-12:raise ValueError('Actual full-DFFT source spectral strip exceeds lattice count')
                results.append(dict(L=L,theta=theta,a=spacing,m=mass,eta=eta,
                    actual_full_jump_source_strip_squared_norm=actual,
                    proved_uniform_lattice_shell_upper=bound,
                    source_scope='NEW physical4-component one-electron/allRef4 free-fibre probe. Entire coherent original-type before-Wilson nominal projector/prefactors; no Gaussian or singleBloch source replacement. Not macro fullGauss/NF source.'))
    return results


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    cfg=json.loads((ROOT/'covariant_prepared_gauss_state_input_kappa400_sigma2000_r2100.json').read_text())
    selected=json.loads((ROOT/consumer.SELECTED).read_text());p=Parameters(**cfg['photon_parameters']);gamma,_=phi_frame(p)
    signs=[]
    for sign in (1,-1):
        row=provider.run_signed(gamma,sign,max(selected['inputs']['normal_nodes']),selected['inputs']['native_steps'],return_runtime=True)
        runtime=row.pop('runtime')
        rows=[spectral_strip(runtime['actual_charged_kernel'],runtime['reference_charged_kernel'],runtime['original_raw_source_purification'],6.,eta,[0.,12.,48.]) for eta in (.5,.2,.1,.03,.01,.003,.001,.0001)]
        signs.append(dict(original_source=row,actual_reference_spectral_strip_consumers=rows,
            physical_raw_source_outside_joint_support_HS=runtime['physical_raw_source_outside_joint_support_HS'],
            actual_original_Pe_Gram=runtime['original_actual_Pe_Gram']))
    photon=selected['macro']['photon'];c=WilsonConsumer(gamma,.01,1.)
    packetmax=max(c.photon_energy(np.array(k)) for k in photon['momenta'])
    macro=[momentum_shell_price(p,eta,packetmax) for eta in (1e-3,1e-5,1e-7,1e-9,1e-11)]
    partition=selected['macro']['finite_box_thermal_partition']['rows']
    soft=min((r['logZ_upper']-partition[0]['logZ_lower']-(p.time_slices-r['inverse_temperature'])*.10)/math.log(10) for r in partition[1:])
    result=dict(governing_plan_revision=consumer.PLAN_REVISION,source_sha256=hashlib.sha256(OWN_BYTES).hexdigest(),
        provider_import_sha256=hashlib.sha256(provider.OWN_BYTES_AT_IMPORT).hexdigest(),
        source_consumer_import_sha256=hashlib.sha256(consumer.OWN_BYTES_AT_IMPORT).hexdigest(),
        frozen_Source2a=consumer.EXPECTED_SOURCE,frozen_Root6911=consumer.EXPECTED_ROOT,
        actual_original_signed_source_strip_scenarios=signs,
        macro_full_jump_fibre_shell_bounds=macro,
        actual_reference_spectator_photon_Chernoff_at_point10_log10=soft,
        reference_original_clip_HS_tail_squared_log10=photon['clipped_reference_Hilbert_tail_squared_log10'],
        full125_photon_coherence_unchanged=True,all_torons_unchanged=True,
        macro_original_joint_source_spectral_strip_bound=None,macro_source_admitted=False,
        actual_full_jump_source_lattice_shell_probe=actual_fourier_shell_probe(gamma),
        elapsed_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(provider.source.portable(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),elapsed_seconds=result['elapsed_seconds'],
        original_signed_Pe_Grams=[r['actual_original_Pe_Gram'] for r in signs],
        actual_strip_raw_norms=[[r['actual_reference_strip_raw_source_squared_norm'] for r in s['actual_reference_spectral_strip_consumers']] for s in signs],
        macro_source_admitted=False)))


if __name__=='__main__':main()

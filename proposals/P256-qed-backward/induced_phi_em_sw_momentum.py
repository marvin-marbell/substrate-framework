"""Distinct full-coexact PH/momentum source for the unchanged coupled Href.

Normalized unitary source in momentum representation, not a Theta-dependent
Slater ansatz. Gaussian integration of Peierls shifts uses unnormalized
cofactors and all 1+63 charged copies. Full real-Gauss Coulomb is evaluated,
not inherited from the charge-preserving onsite-Gamma predecessor.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
from induced_phi_em_momentum_dressing import SOURCE_BYTES as DRESS_BYTES, _adjugate, current_direction
from induced_phi_em_gaussian_variational import build_native, evaluate_gaussian

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**DRESS_BYTES,Path(__file__).name:Path(__file__).read_bytes()}


def build_source(native: dict[str,Any]) -> dict[str,Any]:
    profiles=[];Q=native['Q'];zero=np.zeros(3*native['V'])
    for role,profile in enumerate(native['Pprofiles']):
        values,vectors=np.linalg.eigh(profile['hcentre']);n=profile['Pocc'].shape[1]
        if not (values[n-1]<0<values[n]):raise ArithmeticError('Actual charged sea has no PH gap')
        occupied=vectors[:,:n];empty=vectors[:,n:]
        gap=values[n:,None]-values[None,:n];generators=[]
        for m in range(native['M']):
            J,_=current_direction({'native':native},role,zero,Q[:,m])
            D=-(empty.conj().T@J@occupied)/gap
            B=empty@D@occupied.conj().T
            generators.append(B+B.conj().T)
        profiles.append(dict(O=profile['Pocc'],B=np.stack(generators),load=profile['load'],native=profile,
            minimum_PH_gap=float(gap.min())))
    return dict(native=native,profiles=profiles)


def section(profile: dict[str,Any],p: np.ndarray,kappa: float) -> np.ndarray:
    G=np.einsum('m,mij->ij',p,profile['B']);w,U=np.linalg.eigh(G)
    return (U*np.exp(-1j*kappa*w))@U.conj().T@profile['O']


def conditional_Coulomb(model: dict[str,Any],frames: list[np.ndarray]) -> tuple[float,float]:
    native=model['native'];V=native['V'];mean=np.zeros(V);cov=np.zeros((V,V));onsite=0.
    for profile,O in zip(model['profiles'],frames):
        P=O@O.conj().T;blocks=P.reshape(V,4,V,4).transpose(0,2,1,3);load=profile['load']
        diagonal=np.trace(blocks[np.arange(V),np.arange(V)],axis1=1,axis2=2).real
        mean+=load*(diagonal-2)
        cov+=load*(np.diag(diagonal)-np.einsum('xyab,yxba->xy',blocks,blocks).real)
        onsite+=load*float(np.einsum('ab,xba->',profile['native']['onsite'],blocks[np.arange(V),np.arange(V)]).real)
    C=native['UE']/native['a']*(float(mean@native['Lplus']@mean)+float(np.sum(native['Lplus']*cov)))
    return C,onsite


def shifted_word(model: dict[str,Any],z: np.ndarray,b: np.ndarray,kappa: float,
                 paths: list[dict[str,Any]]) -> complex:
    chis=[];numerators=[]
    for profile in model['profiles']:
        left=section(profile,z+b/2,kappa);right=section(profile,z-b/2,kappa)
        chi,adj=_adjugate(left.conj().T@right)
        chis.append(chi);numerators.append(right@adj@left.conj().T)
    result=0j
    for role,(profile,path) in enumerate(zip(model['profiles'],paths)):
        x,y=path['source'],path['target'];spectator=1+0j
        for r,other in enumerate(model['profiles']):
            spectator*=chis[r]**(other['load']-(1 if r==role else 0))
        N=numerators[role][4*y:4*y+4,4*x:4*x+4]
        result+=profile['load']*spectator*np.einsum('ab,ba->',path['hop'],N)
    return complex(result)


def sample_energy(model: dict[str,Any],z: np.ndarray,kappa: float,variance: float) -> dict[str,float]:
    native=model['native'];frames=[section(profile,z,kappa) for profile in model['profiles']]
    C,onsite=conditional_Coulomb(model,frames);hopping=0.
    for index,path in enumerate(model['profiles'][0]['native']['paths']):
        ell=np.zeros(3*native['V'])
        for edge in path['edges']:ell[edge]+=1
        b=native['Q'].T@ell
        paths=[profile['native']['paths'][index] for profile in model['profiles']]
        hopping+=2*float((np.exp(-variance*float(b@b)/2)*shifted_word(model,z,b,kappa,paths)).real)
    kinetic=native['UE']/native['a']*native['M']/(4*variance)
    return dict(energy=kinetic+C+onsite+hopping,primitive_electric=kinetic,
                full_real_Gauss_Coulomb=C,onsite=onsite,Peierls=hopping)


def run(mean_path: Path,source_path: Path,samples: int) -> dict[str,Any]:
    started=time.monotonic();meanraw=mean_path.read_bytes();sourceraw=source_path.read_bytes()
    old=json.loads(sourceraw);descriptor=old['descriptor']
    if int(descriptor['N'])!=2:raise ValueError('This actual native calculation requires N2, no hidden size reduction')
    native=build_native(json.loads(meanraw),2);model=build_source(native)
    variance=float(descriptor['s_density_variance']);seed=evaluate_gaussian(native,variance,correlator=False)
    rng=np.random.default_rng(2221017);points=rng.normal(size=(samples,native['M']))/np.sqrt(4*variance)
    kappas=[0.,.01,.025,.05,.1,.2];rows=[]
    for kappa in kappas:
        measurements=[sample_energy(model,z,kappa,variance) for z in points]
        values=np.asarray([x['energy'] for x in measurements]);row=dict(kappa=kappa,
            mean_energy=float(values.mean()),standard_error=float(values.std(ddof=1)/np.sqrt(samples)),
            gain_against_exact_seed=float(seed['energy']-values.mean()),
            parts={key:float(np.mean([x[key] for x in measurements])) for key in measurements[0] if key!='energy'})
        rows.append(row)
    zero_error=abs(rows[0]['mean_energy']-seed['energy'])
    if zero_error>1e-7:raise ArithmeticError('Literal fullHref kappa0 contraction disagrees with exact Gaussian seed')
    selected=min(rows,key=lambda row:row['mean_energy']);kappa=selected['kappa']
    derivative_cap=sum(sum(profile['load']*profile['O'].shape[1]*float(np.linalg.norm(profile['B'][m],2))
                          for profile in model['profiles'])**2 for m in range(native['M']))
    probe=points[0];frames=[section(profile,probe,kappa) for profile in model['profiles']]
    normalization=max(float(np.linalg.norm(O.conj().T@O-np.eye(O.shape[1]))) for O in frames)
    inputs=dict(mean_receipt_sha256=hashlib.sha256(meanraw).hexdigest(),onsite_predecessor_receipt_sha256=hashlib.sha256(sourceraw).hexdigest(),
        N=2,V=8,M=native['M'],a=native['a'],UE=native['UE'],charged_counts=[1,63],variance=variance,samples=samples,
        physical_preparation='NONCOMMUTING allcoexact+harmonic PH-momentum unitary, Bcv=-literalJcv/PHgap, allmassresolvedroles',
        plan='issue-222-induced-photon-delta13-continuation-20261008',plan_revision='85975947857ac7816cf0d8674f69229861014434b125a07a6996af0ecf72c4eb')
    return dict(input_descriptor=inputs,input_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        state='Psi(p)=g_s(p) tensor_r Gamma(exp[-i kappa sum_m p_m B_(r,m)]) Omega_(r,0); exactnorm1. All17realphysical modes, no toronclipping/Gibbs trace or fullseaoverlap denominator. PhysicalGauss lift isometry; Ref4 neutralNET0 unchanged. Pi statistics exactGaussian. Thisis NOTaSlater inTheta representation andNOTthefieldfollowing branch.',
        exact_seed_energy=seed['energy'],rows=rows,selected=selected,kappa0_energy_control_error=zero_error,
        actual_conditional_orthonormality_error=normalization,coordinate_second_moment_upper=native['M']*variance+kappa*kappa*derivative_cap,
        tail_formula='P(||Theta||>R)<=coordinate_second_moment_upper/R²; P(||Pi||>R)<=M/(4variance R²). Nonzero tails.',
        limitations='NativeN2midpoint finiteGaussian MC, SE isNOT interval/probabilitycertification. NOmacroenergy, stationarity, photonvacuum, pole, commoncone orCompton/QEDpromotion. Coulombnotinvariant: completeconditionalmean+covariance retained. Spectatorproducts all64 areactualamplitudes, not renormalized ordivided away.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,platform=platform.platform()),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('mean','source','output'):parser.add_argument('--'+role,type=Path,required=True)
    parser.add_argument('--samples',type=int,required=True);args=parser.parse_args()
    if args.samples<2:raise ValueError('Actual Monte Carlo SE requires at least two independent samples')
    report=run(args.mean,args.source,args.samples);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','rows','selected','kappa0_energy_control_error','actual_conditional_orthonormality_error','elapsed_seconds')},indent=2))

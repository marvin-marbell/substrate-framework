"""Charged-alignment variation of the genuine PH/momentum quantum source.

Eta is ONLY a fixed-centre preparation parameter. The physical Hamiltonian,
all literal currents, masses, spacing, charges and Gauss Coulomb stay unchanged.
This is not the field-following Slater-in-Theta preparation.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
from induced_phi_em_sw_momentum import SOURCE_BYTES as SW_BYTES, sample_energy, conditional_Coulomb
from induced_phi_em_momentum_dressing import current_direction
from induced_phi_em_gaussian_variational import build_native

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**SW_BYTES,Path(__file__).name:Path(__file__).read_bytes()}


def build_preparation(native: dict[str,Any],eta: float) -> dict[str,Any]:
    profiles=[];Q=native['Q'];zero=np.zeros(3*native['V']);V=native['V']
    for role,profile in enumerate(native['Pprofiles']):
        onsite=np.kron(np.eye(V),profile['onsite']);prep=onsite+eta*(profile['hcentre']-onsite)
        values,vectors=np.linalg.eigh(prep);n=2*V
        if not values[n-1]<0<values[n]:raise ArithmeticError('Actual preparation has no charged PH gap')
        O=vectors[:,:n];C=vectors[:,n:];gap=values[n:,None]-values[None,:n];generators=[]
        for m in range(native['M']):
            J,_=current_direction({'native':native},role,zero,Q[:,m]);D=-(C.conj().T@J@O)/gap
            B=C@D@O.conj().T;generators.append(B+B.conj().T)
        profiles.append(dict(O=O,B=np.stack(generators),load=profile['load'],native=profile,
            minimum_preparation_PH_gap=float(gap.min()),preparation_eta=eta))
    return dict(native=native,profiles=profiles)


def seed_energy(model: dict[str,Any],variance: float) -> dict[str,float]:
    native=model['native'];frames=[p['O'] for p in model['profiles']];C,onsite=conditional_Coulomb(model,frames);hop=0.
    for profile,O in zip(model['profiles'],frames):
        P=O@O.conj().T
        for path in profile['native']['paths']:
            ell=np.zeros(3*native['V'])
            for edge in path['edges']:ell[edge]+=1
            b=native['Q'].T@ell;x=path['source'];y=path['target']
            hop+=2*profile['load']*float((np.exp(-variance*(b@b)/2)*np.einsum('ab,ba->',path['hop'],P[4*y:4*y+4,4*x:4*x+4])).real)
    K=native['UE']/native['a']*native['M']/(4*variance)
    return dict(energy=K+C+onsite+hop,primitive_electric=K,full_real_Gauss_Coulomb=C,onsite=onsite,Peierls=hop)


def run(mean_path: Path,predecessor_path: Path,samples: int) -> dict[str,Any]:
    started=time.monotonic();meanraw=mean_path.read_bytes();oldraw=predecessor_path.read_bytes();old=json.loads(oldraw)
    if old['input_descriptor']['N']!=2:raise ValueError('No silent size orstate substitution')
    native=build_native(json.loads(meanraw),2);models={eta:build_preparation(native,eta) for eta in (0.,.1,.2,.35,.5,.75,1.)}
    rows=[]
    for eta,model in models.items():
        for variance in (.2,.4,.7,1.,2.,4.):
            rows.append(dict(eta=eta,variance=variance,**seed_energy(model,variance)))
    best=min(rows,key=lambda row:row['energy']);eta=best['eta'];variance=best['variance'];model=models[eta]
    rng=np.random.default_rng(2221021);points=rng.normal(size=(samples,native['M']))/np.sqrt(4*variance);dressed=[]
    for kappa in (0.,.05,.1,.2,.35,.5):
        measured=[sample_energy(model,z,kappa,variance) for z in points];values=np.asarray([row['energy'] for row in measured])
        dressed.append(dict(eta=eta,variance=variance,kappa=kappa,mean_energy=float(values.mean()),
            standard_error=float(values.std(ddof=1)/np.sqrt(samples)),
            parts={key:float(np.mean([row[key] for row in measured])) for key in measured[0] if key!='energy'}))
    control=abs(dressed[0]['mean_energy']-best['energy'])
    if control>1e-7:raise ArithmeticError('Actual fixed-centre preparation disagrees with exact kappa0 expectation')
    selected=min(dressed,key=lambda row:row['mean_energy']);bench=min((row for row in rows if row['eta']==0.),key=lambda row:row['energy'])
    inputs=dict(mean_sha256=hashlib.sha256(meanraw).hexdigest(),predecessor_sha256=hashlib.sha256(oldraw).hexdigest(),
        N=2,V=8,M=native['M'],a=native['a'],UE=native['UE'],charged_counts=[1,63],samples=samples,
        preparation_eta=eta,variance=variance,physical_H_unchanged=True,
        plan='issue-222-induced-photon-delta13-continuation-20261008',plan_revision='85975947857ac7816cf0d8674f69229861014434b125a07a6996af0ecf72c4eb')
    derivative_cap=sum(sum(profile['load']*profile['O'].shape[1]*float(np.linalg.norm(profile['B'][m],2))
                          for profile in model['profiles'])**2 for m in range(native['M']))
    return dict(input_descriptor=inputs,input_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        state='Newnormalizedjointquantumsource Psi(p)=g_s(p) tensor64 Gamma(exp[-i kappa sum_m p_m B_eta,m]) Omega_eta,0. Omega_eta is fixednegativeSlater of onsite+eta*hopping(theta0), NOTfieldfollowingOmega(theta). Bcv usesliteralPHYSICALHrefcurrent dividedbyPREPARATIONgap. Actualenergy/current useonlyunchangedHref. All17realcoexact+harmonic modes, physicalGauss isometry, Ref4NET0 andneutralheavy unchanged. No clippedtorons/Gibbs trace/fullseaoverlap denominator.',
        exact_seed_scan=rows,selected_exact_seed=best,dressed_scan=dressed,selected=selected,
        same_finite_Href_onsiteCAR_Gaussian_benchmark=bench,gain_against_same_finite_onsite_benchmark=bench['energy']-selected['mean_energy'],
        gain_against_eta1_SW_predecessor=old['selected']['mean_energy']-selected['mean_energy'],kappa0_control_error=control,
        normalization=1.,Pi_second_moment=native['M']/(4*variance),
        coordinate_second_moment_upper=native['M']*variance+selected['kappa']**2*derivative_cap,
        noncompact_tail_formula='P(||Theta||>R)<=coordinate_second_moment_upper/R²; P(||Pi||>R)<=Pi_second_moment/R². Finite nonzero tails, no compactification.',
        limitations='NativeN2midpoint finiteGaussianMC, SE notintervalcertification. Eta is preparationONLY, notfittedspeed/lawchange. No macroground, stationarity, photonvacuum, pole/commoncone orQEDpromotion. No inheritanceof onsiteGamma Coulombinvariance.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('mean','source','output'):parser.add_argument('--'+role,type=Path,required=True)
    parser.add_argument('--samples',type=int,required=True);args=parser.parse_args()
    if args.samples<2:raise ValueError('Actual SEM requires two samples')
    report=run(args.mean,args.source,args.samples);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','selected_exact_seed','dressed_scan','selected','same_finite_Href_onsiteCAR_Gaussian_benchmark','gain_against_same_finite_onsite_benchmark','gain_against_eta1_SW_predecessor','elapsed_seconds')},indent=2))

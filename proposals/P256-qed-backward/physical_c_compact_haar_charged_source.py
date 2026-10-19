#!/usr/bin/env python3
"""Literal compact-Haar/full-U1-Gauss charged primitive CAR diagnostic.

All five cycle coordinates of the two-site, six-link U1 graph are compact and
integrated, including all three electromagnetic torons.  The middle operator
is the exact local centred-density projector, NOT a finite bare electric heat
or a Gaussian sea gate.  The original fixed meanbody times a smooth periodic
Gaussian is propagated by the actual uncompressed SF PHaar SF first.
The exact full-range factorization K=V Vdag, V=SF times the complete density0
midpoint Fock space, evaluates moments via Vdag V without a sea truncation.
Separate actual neutral power trials improve only the normalizer bound, NOT
the original source preparation.  A full-Fock remainder/complement bound,
not a Ritz residual alone, controls the global neutral top when it separates.

This is a finite-Naux2 diagnostic, one statistical sector, Phi0/colourI only.
Its beta100 is a boundary coordinate CALIBRATION, not 16*Naux*kaux**4.
No mass1, microscopic alpha, full-SU4, Root931, or photon-pole claim is made.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, cast
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
import physical_c_real_cover_kernel as reference

SOURCE_BYTES=Path(__file__).read_bytes();REFERENCE_BYTES=Path(reference.__file__).read_bytes()
if hashlib.sha256(REFERENCE_BYTES).hexdigest()!='89fb263bedabf82c64fb432bb66d672ffc5a4d612abf6ba237c9dc80fa49b6d2':raise ArithmeticError('Haar source requires frozen REF89')
ALPHA=reference.alpha
ALPHA_BYTES=Path(cast(str,ALPHA.__file__)).read_bytes()
if hashlib.sha256(ALPHA_BYTES).hexdigest()!='56e1c611bffe17964644b47b6aca7ed496a195c8f4795e76e657ee2c01c3ff04':raise ArithmeticError('Haar source requires ledger56e1')


def basis(number: int) -> tuple[list[int],list[np.ndarray]]:
    _,masks=ALPHA.second_quantized(np.eye(8),number)
    return masks,[np.array([j for j in range(8) if mask>>j&1]) for mask in masks]


def canonical_SF(potential: dict[str,Any],number: int,origin_loglambda: float) -> np.ndarray:
    generator,_=ALPHA.second_quantized(potential['h'],number)
    values,vectors=np.linalg.eigh(generator)
    return (vectors*np.exp((values+potential['pref']-origin_loglambda)/2))@vectors.conj().T


def meanbody(law) -> tuple[np.ndarray,float]:
    potential=law.potential(np.zeros((2,3)),np.zeros((2,4,4,4),complex),np.broadcast_to(np.eye(4),(2,3,4,4)).astype(complex))
    values,vectors=np.linalg.eigh(potential['h']);frame=vectors[:,4:]
    _,indices=basis(4)
    return np.array([np.linalg.det(frame[ii,:]) for ii in indices]),float(potential['pref']+sum(values[4:]))


def product_apply(a: np.ndarray,b: np.ndarray,c: np.ndarray,v: np.ndarray) -> np.ndarray:
    x=(a@v.reshape(v.shape[0],-1)).reshape(a.shape[0],70,70)
    x=(b@x.transpose(1,0,2).reshape(70,-1)).reshape(70,a.shape[0],70).transpose(1,0,2)
    return (x.reshape(a.shape[0]*70,70)@c.T).reshape(a.shape[0],70,70)


def source_creators() -> list[np.ndarray]:
    neutral,_=basis(4);charged,_=basis(5);positions={m:i for i,m in enumerate(charged)}
    result=[]
    for spin in range(4):
        op=np.zeros((56,70),complex)
        for column,mask in enumerate(neutral):
            for site in range(2):
                orbital=4*site+spin
                if not mask>>orbital&1:op[positions[mask|(1<<orbital)],column]=(-1)**((mask&((1<<orbital)-1)).bit_count())/2
        result.append(op)
    return result


def physical_density_projector(number: int,q: int) -> np.ndarray:
    active,_=basis(number);aux,_=basis(4)
    na=np.array([(m&15).bit_count()-2 for m in active]);nu=np.array([(m&15).bit_count()-2 for m in aux])
    # Charged total active Q is compensated by external -Q at site1.
    # With total centred charge zero, site1 constraint is redundant, not omitted.
    return q*na[:,None,None]+nu[None,:,None]+nu[None,None,:]==0


def occupation_data(potential: dict[str,Any],origin_loglambda: float) -> tuple[np.ndarray,np.ndarray,float]:
    values,vectors=np.linalg.eigh(potential['h'])
    maxima=np.exp(potential['pref']-origin_loglambda+np.r_[0.,np.cumsum(values[::-1])])
    _,indices=basis(4)
    ground=np.array([np.linalg.det(vectors[ii,4:]) for ii in indices])
    return maxima,ground,float(np.exp(values[3]-values[4]))


def neutral_global_bound(weights: np.ndarray,lambdas: np.ndarray,grounds_a: np.ndarray,
                         grounds_u: np.ndarray,ratios: np.ndarray,q: int,project: np.ndarray,
                         trial: np.ndarray,image: np.ndarray,other_upper: float) -> dict[str,Any]:
    """ALLcanonical leading operator plus complete PSD remainder; global Schur."""
    masks,_=basis(4);counts=np.array([(mask&15).bit_count() for mask in masks])
    ga=[grounds_a[:,counts==n].conj()@grounds_a[:,counts==n].T for n in range(5)]
    gu=[grounds_u[:,counts==n].conj()@grounds_u[:,counts==n].T for n in range(5)]
    projected_gram=np.zeros((len(weights),len(weights)),complex)
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if q*(a-2)+b+c-4==0:projected_gram+=ga[a]*gu[b]*gu[c]
    scale=np.sqrt(weights*lambdas);gram=scale[:,None]*projected_gram*scale[None,:]
    values,vectors=np.linalg.eigh(gram);leading=float(values[-1]);second=float(values[-2])
    z=np.zeros_like(trial)
    for i in range(len(weights)):z+=scale[i]*vectors[i,-1]*np.einsum('j,k,l->jkl',grounds_a[i],grounds_u[i],grounds_u[i])
    z*=project;z/=np.linalg.norm(z)
    perpendicular=trial-z*np.vdot(z,trial)
    sine_squared=float(np.vdot(perpendicular,perpendicular).real)
    remainder=float(np.dot(weights*lambdas,ratios))
    complement=second+(leading-second)*sine_squared+remainder
    rayleigh=float(np.vdot(trial,image).real);residual=float(np.linalg.norm(image-rayleigh*trial))
    canonical_upper=float(np.dot(weights,lambdas));schur=None
    if complement<rayleigh:
        schur=(rayleigh+complement+math.sqrt((rayleigh-complement)**2+4*residual**2))/2
        canonical_upper=min(canonical_upper,schur)
    return dict(canonical_global_upper=canonical_upper,global_ALLoccupation_upper=max(canonical_upper,other_upper),
        instantaneous_ground_leading_top=leading,instantaneous_ground_leading_second=second,
        complete_fullFock_remainder_upper=remainder,actual_neutral_trial_body_Rayleigh=rayleigh,
        actual_neutral_trial_body_residual=residual,trial_leading_angle_sine_squared=sine_squared,
        GLOBAL_neutral_trial_complement_upper=complement,global_Schur_upper=schur,
        other_occupation_sector_upper=other_upper,
        meaning='PSD sum of ALL instantaneous canonical Fock operators. Leading point ground is ONLY a proof decomposition; remainder covers EVERY other canonical state. Full occupation sectors separately bounded, no adopted sea, no charged Ritz/top promotion.')


def compact_nodes(order: int,seed: int,width: float,beta: float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if order<8 or order&(order-1):raise ValueError('Positive compact quadrature order must be powerof2 >=8')
    u=qmc.Sobol(5,scramble=True,rng=np.random.default_rng(seed)).random_base2(int(math.log2(order))-2)
    sigma=1/math.sqrt(2*width*beta)
    uniform=2*np.pi*(u-.5);gaussian=(sigma*ndtri(u)+np.pi)%(2*np.pi)-np.pi
    nodes=np.concatenate([uniform,-uniform,gaussian,-gaussian])
    images=nodes[:,:,None]+2*np.pi*np.arange(-1,2)[None,None,:]
    density=np.prod(np.sum(np.exp(-images**2/(2*sigma*sigma))/(math.sqrt(2*np.pi)*sigma),axis=2),axis=1)*(2*np.pi)**5
    weights=1/(order*(.5+.5*density))
    boundary=np.prod(np.sum(np.exp(-width*beta*images**2/2),axis=2),axis=1)
    return nodes,weights,boundary


def run(order: int,seed: int,moment_steps: int,neutral_steps: int) -> dict[str,Any]:
    if neutral_steps<1:raise ValueError('Actual neutral power trial count must be positive')
    parameters=reference.NativeParameters();active,_,_=reference.native_law(parameters)
    auxiliary,_,_=reference.native_law(parameters);auxiliary.kappa=.08
    q=3;width=3.5;beta=100.
    va,la=meanbody(active);vu,lu=meanbody(auxiliary)
    initial=np.einsum('i,j,k->ijk',va,vu,vu)
    pn=physical_density_projector(4,q);pc=physical_density_projector(5,q);creators=source_creators()
    nodes,weights,boundary=compact_nodes(order,seed,width,beta)
    boundary/=math.sqrt(float(weights@boundary**2))
    factors=[];middle=np.zeros_like(initial);normalizer_upper=0.;grounds_a=[];grounds_u=[];lambdas=[];ratios=[]
    sectors=[(a,b,c) for a in range(9) for b in range(9) for c in range(9) if q*(a-4)+b+c-8==0 and (a,b,c)!=(4,4,4)]
    other_occupation_upper=np.zeros(len(sectors))
    remainder=np.zeros((2,4,4,4),complex);colour=np.broadcast_to(np.eye(4),(2,3,4,4)).astype(complex)
    for coordinate,w,f in zip(nodes,weights,boundary,strict=True):
        phases=np.array([[0.,coordinate[1],coordinate[2]],[coordinate[0],coordinate[3],coordinate[4]]])
        ap=active.potential(q*phases,remainder,colour);up=auxiliary.potential(phases,remainder,colour)
        sa=canonical_SF(ap,4,la);sc=canonical_SF(ap,5,la);su=canonical_SF(up,4,lu)
        ma,ga,ra=occupation_data(ap,la);mu,gu,ru=occupation_data(up,lu)
        grounds_a.append(ga);grounds_u.append(gu);lambdas.append(ma[4]*mu[4]**2);ratios.append(max(ra,ru))
        for j,(a,b,c) in enumerate(sectors):other_occupation_upper[j]+=w*ma[a]*mu[b]*mu[c]
        san=sa@sa;scn=sc@sc;sun=su@su
        factors.append((w,sa,sc,su,san,scn,sun))
        middle+=w*f*product_apply(sa,su,su,initial)
        normalizer_upper+=w*lambdas[-1]
    middle*=pn
    prepared_norm=0.;raw_source=0.;neutral_second=np.zeros_like(initial)
    source_mid=[np.zeros((56,70,70),complex) for _ in range(4)]
    number_observable=sum(c.conj().T@c for c in creators)
    for w,sa,sc,su,san,scn,sun in factors:
        image=product_apply(san,sun,sun,middle)
        neutral_second+=w*image
        prepared_norm+=w*np.vdot(middle,image).real
        raw_source+=w*np.vdot(middle,product_apply(sa@number_observable@sa,sun,sun,middle)).real
        for h,c in enumerate(creators):source_mid[h]+=w*product_apply(sc@c@sa,sun,sun,middle)
    neutral_second*=pn
    for image in source_mid:image*=pc
    lN=float(np.vdot(neutral_second,neutral_second).real/prepared_norm)
    original_trial_Rayleigh=lN;neutral_trials=[dict(power=0,physical_trial_Rayleigh=lN)]
    trial=neutral_second/np.linalg.norm(neutral_second)
    image=np.empty_like(trial)
    for power in range(1,neutral_steps+1):
        image.fill(0)
        for w,sa,sc,su,san,scn,sun in factors:image+=w*product_apply(san,sun,sun,trial)
        image*=pn
        physical_Rayleigh=float(np.vdot(image,image).real/np.vdot(trial,image).real)
        lN=max(lN,physical_Rayleigh)
        neutral_trials.append(dict(power=power,physical_trial_Rayleigh=physical_Rayleigh))
        if power<neutral_steps:trial=image/np.linalg.norm(image)
    global_proof=neutral_global_bound(weights,np.array(lambdas),np.array(grounds_a),np.array(grounds_u),np.array(ratios),q,pn,trial,image,float(max(other_occupation_upper)))
    normalizer_upper=min(normalizer_upper,global_proof['global_ALLoccupation_upper'])
    source_half=sum(float(np.vdot(image,image).real) for image in source_mid)
    if not 0<lN<=normalizer_upper*(1+1e-10):raise ArithmeticError('Literal Haar neutral Rayleigh exceeds positive SF bound')
    if abs(raw_source/prepared_norm-1)>2e-8:raise ArithmeticError('Full primitive CAR rawGram failed neutral C on actual Haar preparation')
    moments=[];vectors=[image.copy() for image in source_mid]
    for step in range(1,moment_steps+1):
        following=[np.zeros_like(v) for v in vectors]
        for w,sa,sc,su,san,scn,sun in factors:
            for h,v in enumerate(vectors):following[h]+=w/lN*product_apply(scn,sun,sun,v)
        for v in following:v*=pc
        if step in (4,8,16):
            norm=sum(float(np.vdot(v,v).real) for v in following)
            cross=sum(float(np.vdot(v,z).real) for v,z in zip(vectors,following,strict=True))
            growth=math.exp(2*step*4*parameters.spacing)
            moments.append(dict(n=step,physical_Pe4_WHOLEhalf_fraction_numeric_upper=growth*norm/source_half,
                physical_Pe4_RAW_fraction_numeric_upper=growth*cross/(lN*raw_source)))
        vectors=following
    return dict(order=order,seed=seed,compact_Haar_weight_sum=float(sum(weights)),
        neutral_fullcanonical_dimension=70**3,charged_fullcanonical_dimension=56*70**2,
        neutral_midrange_dimension=int(np.count_nonzero(pn)),charged_midrange_dimension=int(np.count_nonzero(pc)),
        original_FIXEDbody_periodicGaussian_FULL_HaarK_preparation_Gram=float(prepared_norm),
        original_RAW_primitive_CAR_Gram=float(raw_source/prepared_norm),neutral_trial_Rayleigh=lN,neutral_global_positive_SF_upper=float(normalizer_upper),
        original_preparation_neutral_Rayleigh=original_trial_Rayleigh,actual_neutral_power_trials=neutral_trials,
        global_neutral_normalizer_proof=global_proof,
        neutral_trial_midpoint_bytes_base64=base64.b64encode(np.ascontiguousarray(trial[pn]).tobytes()).decode(),
        neutral_trial_midpoint_byte_contract='complex128 C-order entries of complete70x70x70 tensor retained by exactrho0 mask; mask/source/input determine indices. This is a separate normalizer trial, not a replacement source preparation.',
        actual_WHOLEhalf_primitive_source_Gram_interval=[source_half/(normalizer_upper*prepared_norm),source_half/(lN*prepared_norm)],
        physical_source_moment_windows=moments,
        input=dict(Q=q,Ref_primitive_charge=0,external_static_compensator=[1,-q],Naux=2,kaux=.08,
            active_critical_bare_mass=parameters.critical_bare_mass,spacing=parameters.spacing,
            primitive_R='Normalized rankone compactHaar on every link; exact local centred density0 midpoint after fullU1Gauss quotient',
            Gaussian_width=width,beta_coordinate_CALIBRATION=beta,formal_plaquette_coefficient16Nauxkaux4=16*2*.08**4,
            small_geometry_short_windings='Length1/2 spatial windings exist: formal4loop coefficient is NOT the complete induced Maxwell action or beta100.',
            statistical_sector=[0,0,0],geometry=[2,1,1],all_compact_cycle_coordinates=5,
            boundary='Original constant filled meanbody active+two auxiliary Slaters times Cinf periodizedGaussian3.5 in all five canonical-calibration cycle coordinates; standard centred Gauss-equivariant lift. FULLK first, not adopted interactingsea.',
            source='Primitive8 Dirac cdag packet([I;I]/2) r with Ref4->four orthogonal Ref3 holes; tree Wilson U00**Q=1 in treegauge, exact integer source and externalcharge ledger.',
            omitted_Gaussian_images_log_bound=-width*beta*(3*np.pi)**2/2,
            scope='Finite positive compact quadrature, ALLcanonical Fock and literal fullU1Gauss; Phi0/colourI, one statistical sector. No bareE, no RPA, no source/pole gate. Not source2a/Root931/fullSU4/allstatistical/macroalpha/mass1/photon/Compton certificate. Numerical source moments are not interval arithmetic or quadrature-continuum proof.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--orders',type=int,nargs='+',default=[32,64]);parser.add_argument('--moment-steps',type=int,default=4);parser.add_argument('--neutral-steps',type=int,default=8);args=parser.parse_args();started=time.time()
    cases=[run(order,20261009,args.moment_steps,args.neutral_steps) for order in args.orders]
    inputs=dict(orders=args.orders,moment_steps=args.moment_steps,neutral_steps=args.neutral_steps,seed=20261009)
    report=dict(cases=cases,execution=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),reference_at_IMPORT=hashlib.sha256(REFERENCE_BYTES).hexdigest(),reference_bytes_at_IMPORT=REFERENCE_BYTES.decode(),ledger_at_IMPORT=hashlib.sha256(ALPHA_BYTES).hexdigest(),ledger_bytes_at_IMPORT=ALPHA_BYTES.decode(),input_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),input_json=inputs,python=sys.version,numpy=np.__version__,runtime=platform.platform(),startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES and Path(reference.__file__).read_bytes()==REFERENCE_BYTES and Path(cast(str,ALPHA.__file__)).read_bytes()==ALPHA_BYTES,elapsed_seconds=time.time()-started))
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in case.items() if k not in ('neutral_trial_midpoint_bytes_base64','input')} for case in cases]))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Actual scalar+valence C/T4 curl and mixed-temporal diagnostic, no Fock array.

Action convention is S=cB sum Fij²+cE sum F0j²; beta=2c.  The
high-precision evaluator is an independent exact-frame evaluation of the
captured CartesianCWilson equations at Phi=0, SU4=I, not a replacement core.
A nonzero local winding profile is retained as an adverse control.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
import mpmath as mp
import numpy as np
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES=Path(__file__).read_bytes()
CORE_BYTES=(ROOT/'physical_c_bare_electric_kernel.py').read_bytes()
import physical_c_bare_electric_kernel as core
alpha=core.alpha
IMPORT_EPOCH=dict(core_sha256=hashlib.sha256(CORE_BYTES).hexdigest(),
    core_bytes=CORE_BYTES.decode(),reference_sha256=core.REFERENCE_HASH,
    reference_bytes=core.REFERENCE_BYTES.decode(),alpha_sha256=alpha.SOURCE_HASHES,
    alpha_bytes={name:value.decode() for name,value in alpha.SOURCE_BYTES.items()})


def graph(shape: tuple[int,int,int]) -> tuple[np.ndarray,np.ndarray]:
    coordinates=np.array(list(np.ndindex(shape)),int)
    index={tuple(x):i for i,x in enumerate(coordinates)}
    targets=np.array([[index[tuple((x[q]+(q==j))%shape[q] for q in range(3))]
        for j in range(3)] for x in coordinates],int)
    return coordinates,targets


def background(shape: tuple[int,int,int]) -> dict[str,Any]:
    coordinates,targets=graph(shape);v=len(coordinates)
    raw=np.zeros((v,3));raw[:,1]=np.cos(2*np.pi*coordinates[:,0]/shape[0])*np.cos(2*np.pi*coordinates[:,1]/shape[1])
    incidence=np.zeros((v,3*v))
    for x in range(v):
        for j in range(3):
            incidence[x,3*x+j]-=1;incidence[targets[x,j],3*x+j]+=1
    a=(raw.ravel()-incidence.T@np.linalg.pinv(incidence@incidence.T)@incidence@raw.ravel()).reshape(v,3)
    # Exact Fourier values are rational for the selected 4x3x1 graph.
    a=np.round(a,14)
    curl=np.array([a[:,j]+a[targets[:,j],l]-a[targets[:,l],j]-a[:,l]
        for j in range(3) for l in range(j+1,3)])
    line_torons=[]
    for x in range(v):
        for j in range(3):
            y=x;s=0.
            for _ in range(shape[j]):s+=a[y,j];y=targets[y,j]
            line_torons.append(s)
    return dict(coordinates=coordinates,targets=targets,a=a,curl=curl,
        normB=float(np.sum(curl**2)),normE=float(np.sum(a**2)),
        divergence=float(np.linalg.norm(incidence@a.ravel())),max_line_toron=float(max(map(abs,line_torons))),
        phi=np.zeros((v,4,4,4),complex),links=np.tile(np.eye(4),(v,3,1,1)))


def make_law(k: float,b: dict[str,Any]) -> Any:
    return alpha.CartesianCWilson(alpha.gamma_frame(),replace(alpha.Parameters(),
        mass=math.log(1/(2*k)-3),spacing=1.),b['targets'])


def float_state(law: Any,b: dict[str,Any],amplitude: float,mode: str,profile: np.ndarray|None=None) -> dict[str,Any]:
    phases=amplitude*(b['a'] if profile is None else profile)
    p=law.potential(phases,b['phi'],b['links']) if mode=='C' else law._base(phases,b['phi'],b['links'])
    e,r=eigh(p['h']);nv=len(e)//2;pref=p['pref'] if mode=='C' else p['scalar']
    return dict(h=p['h'],pref=pref,e=e,r=r,W=r[:,-nv:],loglambda=float(pref+np.sum(e[-nv:])),
        valence=float(np.sum(e[-nv:])),gap=float(min(e[nv],-e[nv-1])))


def float_probe(k: float,b: dict[str,Any],amplitude: float) -> dict[str,Any]:
    law=make_law(k,b);rows={}
    for mode in ('C','positiveT4'):
        zero=float_state(law,b,0.,mode);plus=float_state(law,b,amplitude,mode);minus=float_state(law,b,-amplitude,mode)
        cb=-(plus['loglambda']+minus['loglambda']-2*zero['loglambda'])/(2*amplitude**2*b['normB'])
        overlap=minus['W'].conj().T@plus['W']
        loss=-2*float(np.linalg.slogdet(overlap)[1]);denominator=2*(2*amplitude)**2*b['normE']
        ce=loss/denominator
        slow=[]
        n=len(zero['e'])//2
        delta=zero['e'][n:][None,:]-zero['e'][:n,None]
        rdecay=np.exp(-delta)
        for step in (.001,.0005):
            pstep=float_state(law,b,step,mode);mstep=float_state(law,b,-step,mode)
            hx=(pstep['h']-mstep['h'])/(2*step)
            vertex=zero['r'][:,:n].conj().T@hx@zero['W']
            metric=np.abs(vertex)**2/(2*delta**2*b['normE'])
            frequencies=[]
            for omega in (0.,.05,.1,.2):
                omegaHat2=4*math.sin(omega/2)**2
                coefficient=float(np.sum(metric*(1-rdecay**2)/((1-rdecay)**2+rdecay*omegaHat2)))
                frequencies.append(dict(omega=omega,cE=coefficient))
            slow.append(dict(derivative_step=step,cE_slow=frequencies[0]['cE'],
                cE_slow_over_k4=frequencies[0]['cE']/k**4,
                minimum_particle_hole_gap=float(np.min(delta)),frequencies=frequencies))
        products=[]
        # Two true temporal seams in the cyclic A^L B^L product.  Positive
        # symmetric similarity avoids non-Hermitian product eigenvalue errors.
        for length in (1,2,4,8):
            ea,eb=minus['e'],plus['e'];ra,rb=minus['r'],plus['r']
            half=(ra*np.exp(length*(ea-ea[-1])/2))@ra.conj().T
            full=(rb*np.exp(length*(eb-eb[-1])))@rb.conj().T
            vals=eigh(half@full@half,eigvals_only=True)[-len(ea)//2:]
            prod=float(np.sum(np.log(vals))+length*len(vals)*(ea[-1]+eb[-1]))
            static=length*(minus['valence']+plus['valence'])
            products.append(dict(plateau_length=length,whole_scalar_log_product=length*(minus['pref']+plus['pref'])+prod,
                static_whole_scalar_log_product=length*(minus['loglambda']+plus['loglambda']),
                temporal_excess_action=static-prod,cE=(static-prod)/denominator))
        rows[mode]=dict(cB=cb,cB_over_k4=cb/k**4,cE=ce,cE_over_k4=ce/k**4,
            scalar_action=-(plus['pref']+minus['pref']-2*zero['pref'])/(2*amplitude**2*b['normB']),
            valence_action=-(plus['valence']+minus['valence']-2*zero['valence'])/(2*amplitude**2*b['normB']),
            two_seam_log_overlap=loss,minimum_overlap_singular_value=float(np.linalg.svd(overlap,compute_uv=False)[-1]),
            single_particle_defect_gap=zero['gap'],products=products,slow_temporal_response=slow,
            slow_temporal_derivation='V=(1-r)hprime/Delta, r=exp(-Delta). Exact separated transfer insertion sum: cE(omega)=sum |hprime|²/(2Delta² normE)*(1-r²)/[(1-r)²+r omegaHat²]; cE(0)=seam metric*coth(Delta/2).')
    rows['C_T4_h_norm_difference']=float(np.linalg.norm(float_state(law,b,amplitude,'C')['h']-float_state(law,b,amplitude,'positiveT4')['h']))
    adverse=np.zeros_like(b['a']);adverse[:,1]=np.cos(2*np.pi*b['coordinates'][:,0]/4)
    z=float_state(law,b,0,'C',adverse);p=float_state(law,b,amplitude,'C',adverse);m=float_state(law,b,-amplitude,'C',adverse)
    rows['nonfixed_local_toron_falsifier']=dict(action_quadratic=-(p['loglambda']+m['loglambda']-2*z['loglambda'])/(2*amplitude**2),
        reason='A_y=cos(x) has nonzero y-line winding phases despite zero global harmonic average; not contractiblecurl control.')
    return dict(k=k,amplitude=amplitude,measurements=rows)


def mp_matrix(a: np.ndarray) -> Any:
    return mp.matrix([[mp.mpc(str(z.real),str(z.imag)) for z in row] for row in a])


def hp_state(k: str,b: dict[str,Any],amplitude: str) -> dict[str,Any]:
    """Independent exact-frame same equation at zero Phi and identity colour."""
    v=len(b['targets']);size=4*v;kappa=mp.mpf(k);amp=mp.mpf(amplitude)
    gamma=[mp_matrix(g) for g in alpha.gamma_frame()]
    beta_change=mp.matrix([[1,0,1,0],[0,1,0,1],[1,0,-1,0],[0,1,0,-1]])/mp.sqrt(2)
    change=mp.zeros(size);C=mp.zeros(size)
    exact_ch=np.rint(make_law(float(k),b).CH.real)+1j*np.rint(make_law(float(k),b).CH.imag)
    ch=mp_matrix(exact_ch)
    # Group all plus modes before all minus modes, as in the captured law.
    for x in range(v):
        for i in range(4):
            for j in range(4):
                change[4*x+i,2*x+j if j<2 else 2*v+2*x+j-2]=beta_change[i,j]
                C[4*x+i,4*x+j]=ch[i,j]
    def base(sign: int) -> tuple[Any,Any,Any]:
        B=mp.eye(size)/(2*kappa);D=mp.zeros(size)
        for j in range(3):
            U=mp.zeros(size);G=mp.zeros(size)
            for x in range(v):
                y=int(b['targets'][x,j]);phase=mp.exp(sign*mp.j*amp*mp.mpf(str(b['a'][x,j])))
                for i in range(4):
                    U[4*x+i,4*y+i]=phase
                    for l in range(4):G[4*x+i,4*x+l]=gamma[j+1][i,l]
            B-=(U+U.H)/2;D+=(G*U-U.H*G)/2
        transformed=change.H*B*change;current=change.H*D*change
        n=2*v;bp=transformed[:n,:n];bm=transformed[n:,n:];f=current[:n,n:]
        L=mp.eye(size);diag=mp.zeros(size);inv=bp**-1
        for i in range(n):
            for j in range(n):L[i,n+j]=-f[i,j];diag[i,j]=inv[i,j];diag[n+i,n+j]=bm[i,j]
        T=change*(L.H*diag*L)*change.H
        e,r=mp.eighe(T);h=r*mp.diag([mp.log(x) for x in e])*r.H
        scalar=(mp.log(mp.det(bp)).real+mp.log(mp.det(bm)).real)/2
        return h,scalar,T
    h,pref,T=base(1);reverse,reversepref,_=base(-1)
    hc=(h-C*reverse.T*C.H)/2;hc=(hc+hc.H)/2
    cpref=(pref+reversepref)/2-sum(hc[i,i].real for i in range(size))/2
    out={}
    for mode,H,P in (('C',hc,cpref),('positiveT4',h,pref)):
        e,r=mp.eighe(H);n=size//2
        out[mode]=dict(pref=P,valence=sum(e[n:]),loglambda=P+sum(e[n:]),W=r[:,n:],gap=min(e[n],-e[n-1]))
    return out


def hp_probe(k: str,b: dict[str,Any],amplitude: str,dps: int) -> dict[str,Any]:
    with mp.workdps(dps):
        zero=hp_state(k,b,'0');plus=hp_state(k,b,amplitude);minus=hp_state(k,b,str(-mp.mpf(amplitude)))
        out={};a=mp.mpf(amplitude);kk=mp.mpf(k)
        for mode in ('C','positiveT4'):
            z,p,m=zero[mode],plus[mode],minus[mode]
            cb=-(p['loglambda']+m['loglambda']-2*z['loglambda'])/(2*a*a*mp.mpf(str(b['normB'])))
            ce=-2*mp.log(abs(mp.det(m['W'].H*p['W'])))/(8*a*a*mp.mpf(str(b['normE'])))
            out[mode]={name:mp.nstr(value,dps-8) for name,value in dict(cB=cb,cE=ce,cB_over_k4=cb/kk**4,cE_over_k4=ce/kk**4,
                scalar_cB=-(p['pref']+m['pref']-2*z['pref'])/(2*a*a*mp.mpf(str(b['normB']))),
                valence_cB=-(p['valence']+m['valence']-2*z['valence'])/(2*a*a*mp.mpf(str(b['normB']))),gap=z['gap']).items()}
        return dict(k=k,amplitude=amplitude,dps=dps,measurements=out)


def report(args: argparse.Namespace) -> dict[str,Any]:
    b=background((4,3,1));floats=[float_probe(k,b,.1) for k in (.04,.02,.01,.005)]
    precision_input=None
    if args.precision_capsule:
        input_bytes=Path(args.precision_capsule).read_bytes();previous=json.loads(input_bytes)
        if previous['execution']['source_sha256']!='9c4fc9dacf5c9985a78fed4ce23b9ddfed3a8f258e1a0c6d57a925d88d505446':
            raise ValueError('Precision input must be the actual unchanged exact-frame predecessor')
        if previous['execution']['dependencies_at_IMPORT']['core_sha256']!=IMPORT_EPOCH['core_sha256']:
            raise ValueError('Precision input core epoch differs')
        if previous['report']['geometry']['phases_direction']!=b['a'].tolist():
            raise ValueError('Precision input curl geometry differs')
        precise=previous['report']['independent_highprecision_same_equations']
        precision_input=dict(path=args.precision_capsule,sha256=hashlib.sha256(input_bytes).hexdigest(),
            bytes_at_startup=input_bytes.decode(),meaning='Previously executed exact-frame controls, not remeasured in this successor. Actual float primitive and slow temporal response are re-executed.')
    else:
        precise=[hp_probe(k,b,a,dps) for k,a,dps in [('0.02','0.1',40),('0.01','0.1',40),('0.005','0.1',40),('0.01','0.05',40),('0.01','0.1',55)]]
    low=precise[2]['measurements'];ue=.02
    dispersion={}
    for mode in ('C','positiveT4'):
        cb=float(low[mode]['cB']);seam=float(low[mode]['cE']);k=.005
        ce=floats[-1]['measurements'][mode]['slow_temporal_response'][-1]['cE_slow']
        dispersion[mode]=dict(k=k,cB=cb,cE_slow=ce,cE_two_seam_metric=seam,cB_over_cE_slow=cb/ce,
            bare_electric_UE=ue,temporal_bare_action_coefficient=1/(4*ue),
            transverse_formula='S2=[(1/(4UE)+Naux*cE_slow)*omegaHat²+Naux*cB*qHat²]|A_T|²; real-time omegaHat²=Naux*cB/(1/(4UE)+Naux*cE_slow)*qHat².',
            interpretation='Conditional derivative expansion of measured Phi=0 auxiliary valence background, not proof of the complete core photon.',
            macro_limit_ratio_if_coefficients_uniform=cb/ce)
    with mp.workdps(60):
        k=mp.mpf('1e-250');naux=mp.mpf('6.25e1148');beta=16*naux*k**4
        # All occupation sectors are included by the exact exterior-algebra
        # character.  These are fixed-background bounds, not full field claims.
        macro=dict(kaux='1e-250',Naux='6.25e1148',betaTARGET=str(beta),
            normalized_character='Tr_fullFock(exp(pref) Gamma(exp(h)))/lambdaVac=prod_i(1+exp(-abs(h_i))); no occupation-zero deletion.',
            exact_norm='For any occupation subset S, exp(pref+sum_{i in S}h_i)/lambdaVac<=1, provided filled modes are all positive h_i.',
            character_debt='Naux*4V*exp(-gap) is not small if gap~log(1/(2kaux)): scale O(Naux*kaux)=O(1e898). Small operator defect factor per species does not imply small fullFock trace correction.',
            even_neutral_pairs_debt='Even-only paired defects O(Naux*kaux²)=O(1e648) also do not vanish.',
            uniform_loop_remainder_status='NOT CERTIFIED. Finite-k/highprecision ratios are controls, not a uniform all-field analytic remainder at kaux. Multiplication by Naux is conditional, never an observed macro receipt.')
    return dict(schema='physical-C-bare-electric-loops-v1',geometry=dict(shape=[4,3,1],sites=len(b['targets']),onebody_dimension=4*len(b['targets']),
        targets=b['targets'].tolist(),phases_direction=b['a'].tolist(),curl=b['curl'].tolist(),sum_spatial_curl_squared=b['normB'],sum_link_squared=b['normE'],
        divergence=b['divergence'],maximum_straight_line_toron=b['max_line_toron'],Phi='zero full four-component matrices',SU4='identity all links'),
        precision_input_capture=precision_input,
        input=dict(k_values=[.04,.02,.01,.005],amplitude=.1,precision_controls=[40,55],bare_electric_UE=ue),
        actual_float_primitive=floats,independent_highprecision_same_equations=precise,dispersion=dispersion,macro_alloccupation=macro,
        inference=dict(leading_action_cB_cE='8*k^4 suggested by own measured converging ratios; beta=16*k^4 only when beta=2*c convention.',
            C_positiveT4_relation='Measured coincident only on Phi=0 and identity SU4; no assertion for full Phi/colour primitive.',
            alternative_epoch='No paired alternative needed: actual fixedtoron C response is positive. An unsymmetrized positiveT4 C-flavour pair would be a DIFFERENT primitive, not this core.'),
        falsifiers=['Nonfixed local line torons can reverse the apparent spatial coefficient.','Dropping scalar produces a large uncancelled spatial response.','Scalar lambda alone cannot see temporal waveframe loss.','Counting one seam doubles the inferred cE.','Bare-electric only has no magnetic action before the auxiliary response.'],
        debts=['Uniform allNF/global Gauss and interacting Phi/colour auxiliary elimination.','Actual core source/current Ward identity and conditioned state response.','Uniform spatial/temporal remainder at macro kaux and volume.','Continuum low-q/omega follows only conditionally from local derivative expansion, not from this finite-volume graph alone.'])


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--capsule',required=True);parser.add_argument('--receipt',required=True)
    parser.add_argument('--precision-capsule',help='Explicit actual predecessor capsule supplying reusable unchanged highprecision measurements')
    args=parser.parse_args();start=time.time();payload=report(args)
    execution=dict(source_bytes_at_IMPORT=SOURCE_BYTES.decode(),source_sha256=hashlib.sha256(SOURCE_BYTES).hexdigest(),dependencies_at_IMPORT=IMPORT_EPOCH,
        argv=sys.argv,python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,mpmath=mp.__version__,
        cwd=os.getcwd(),startup_thread_environment={name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},elapsed_seconds=time.time()-start)
    Path(args.capsule).write_text(json.dumps(dict(report=payload,execution=execution),indent=2)+'\n')
    print(json.dumps(dict(capsule=args.capsule,source_sha256=execution['source_sha256'],elapsed_seconds=execution['elapsed_seconds'],
        precise=payload['independent_highprecision_same_equations'],dispersion=payload['dispersion']),indent=2),flush=True)


if __name__=='__main__':main()

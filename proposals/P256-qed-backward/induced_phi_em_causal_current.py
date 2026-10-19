#!/usr/bin/env python3
"""Actual full-Href retarded current tangent on NEW PH/momentum preparation.

A priced short-time kernel is not a stationary spectral pole. Electric,
literal Peierls/CAR, and full real-Gauss Coulomb commutators remain separate.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
from induced_phi_em_gaussian_variational import build_native
from induced_phi_em_momentum_dressing import (build_dressing,evaluate_dressing,fcs_expect,
    _fcs,_delta,_gaussian_factor,_spectator,SOURCE_BYTES as DRESS_BYTES)

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**DRESS_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
Key=tuple[int,int,tuple[int,...]]


def add_event(result: dict[Key,np.ndarray],key: Key,coefficient: np.ndarray) -> None:
    if key in result:result[key]+=coefficient
    else:result[key]=coefficient.copy()


def make_operators(model: dict[str,Any],direction: np.ndarray) -> tuple[dict[Key,np.ndarray],dict[Key,np.ndarray]]:
    native=model['native'];V=native['V'];E=3*V;W=model['W'];H={};J={};zero=tuple([0]*E)
    for role,profile in enumerate(native['Pprofiles']):
        onsite=W.conj().T@profile['onsite']@W
        for x,a,b in itertools.product(range(V),range(4),range(4)):
            if onsite[a,b]!=0:
                value=np.zeros(2,complex);value[role]=onsite[a,b]
                add_event(H,(4*x+a,4*x+b,zero),value)
        for path in profile['paths']:
            ell=np.zeros(E,dtype=int)
            for edge in path['edges']:ell[edge]+=1
            phase=tuple(int(x) for x in ell);negative=tuple(-x for x in phase);slope=float(ell@direction)
            hop=W.conj().T@path['hop']@W;x=path['source'];y=path['target']
            for a,b in itertools.product(range(4),repeat=2):
                if hop[a,b]!=0:
                    value=np.zeros(2,complex);value[role]=hop[a,b]
                    key=(4*x+a,4*y+b,phase);hc=(4*y+b,4*x+a,negative)
                    add_event(H,key,value);add_event(H,hc,value.conj())
                    if slope:
                        add_event(J,key,1j*slope*value);add_event(J,hc,-1j*slope*value.conj())
    return ({key:value for key,value in H.items() if np.any(value!=0)},
            {key:value for key,value in J.items() if np.any(value!=0)})


def commutator(A: dict[Key,np.ndarray],B: dict[Key,np.ndarray]) -> dict[Key,np.ndarray]:
    result={};left={};right={}
    for key,value in B.items():
        left.setdefault(key[0],[]).append((key,value));right.setdefault(key[1],[]).append((key,value))
    for (i,j,phase),value in A.items():
        for (k,l,q),w in left.get(j,[]):
            add_event(result,(i,l,tuple(a+b for a,b in zip(phase,q))),value*w)
        for (k,l,q),w in right.get(i,[]):
            add_event(result,(k,j,tuple(a+b for a,b in zip(phase,q))),-value*w)
    return {key:value for key,value in result.items() if np.any(value!=0)}


class SourceWords:
    def __init__(self,model: dict[str,Any],kappa: float,variance: float):
        self.model=model;self.kappa=kappa;self.variance=variance

    @lru_cache(maxsize=256)
    def prepared(self,phase: tuple[int,...]) -> list[dict[str,Any]]:
        return _fcs(self.model,np.asarray(phase,float),self.kappa)

    def one(self,key: Key,coefficients: np.ndarray) -> complex:
        i,j,phase=key;prepared=self.prepared(phase);result=0j
        for role in range(2):
            if coefficients[role]!=0:
                result+=coefficients[role]*fcs_expect(self.model,[(role,i,j)],np.asarray(phase),self.kappa,self.variance,prepared=prepared)
        return result

    def pair(self,key1: Key,coeff1: np.ndarray,key2: Key,coeff2: np.ndarray) -> complex:
        i,j,p=key1;k,l,q=key2;phase=tuple(a+b for a,b in zip(p,q));result=0j;prepared=self.prepared(phase)
        for a,b in itertools.product(range(2),repeat=2):
            if coeff1[a]!=0 and coeff2[b]!=0:
                result+=coeff1[a]*coeff2[b]*fcs_expect(self.model,[(a,i,j),(b,k,l)],np.asarray(phase),self.kappa,self.variance,prepared=prepared)
        return result

    @lru_cache(maxsize=256)
    def density_context(self,phase: tuple[int,...]) -> tuple[dict[str,Any],...]:
        result=[]
        for profile,data in zip(self.model['profiles'],self.prepared(phase)):
            U,s,Vh=np.linalg.svd(data['matrix']);left=profile['O']@U;right=data['RO']@Vh.conj().T
            phase_det=complex(np.linalg.det(U)*np.linalg.det(Vh));n=len(s)
            first=np.array([np.prod(np.delete(s,a)) for a in range(n)])
            second=np.zeros((n,n))
            for a,b in itertools.product(range(n),repeat=2):
                if a!=b:second[a,b]=np.prod(np.delete(s,[a,b]))
            result.append(dict(left=left,right=right,phase_det=phase_det,first=first,second=second))
        return tuple(result)

    def density(self,key: Key,coefficients: np.ndarray,weights: np.ndarray) -> complex:
        i,j,phase=key;prepared=self.prepared(phase);contexts=self.density_context(phase);records=[]
        diagonal=np.repeat(weights,4)
        for context in contexts:
            L=context['left'];R=context['right'];u=L[i].conj();v=R[j]
            A=np.outer(u,v);D=L.conj().T@(diagonal[:,None]*R)
            cA=context['phase_det']*np.sum(context['first']*np.diag(A))
            cD=context['phase_det']*np.sum(context['first']*np.diag(D))
            cross=context['phase_det']*(np.diag(A)@context['second']@np.diag(D)-np.sum(context['second']*A*D.T))
            records.append((cA,cD,cross+diagonal[j]*cA))
        total=0j
        for a,profile in enumerate(self.model['profiles']):
            cA,cD,cAD=records[a];load=profile['load']
            total+=coefficients[a]*load*_spectator(self.model,prepared,a)*cAD
            if load>1:
                spectator=np.prod([prepared[z]['chi']**(p['load']-2*(z==a)) for z,p in enumerate(self.model['profiles'])])
                total+=coefficients[a]*load*(load-1)*spectator*cA*cD
            for b,other in enumerate(self.model['profiles']):
                if b!=a:
                    spectator=np.prod([prepared[z]['chi']**(p['load']-(z==a)-(z==b)) for z,p in enumerate(self.model['profiles'])])
                    total+=coefficients[a]*load*other['load']*spectator*cA*records[b][1]
        return complex(total*_gaussian_factor(self.model,np.asarray(phase),_delta(self.model,i,j),self.kappa,self.variance))


def measure(model: dict[str,Any],kappa: float,variance: float,direction: np.ndarray) -> dict[str,Any]:
    start=time.monotonic();native=model['native'];H,J=make_operators(model,direction);words=SourceWords(model,kappa,variance)
    mean=sum((words.one(key,value) for key,value in J.items()),0j)
    matter_op=commutator(commutator(H,J),J)
    matter=sum((words.one(key,value) for key,value in matter_op.items()),0j)
    electric=0j;coulomb=0j;second=0j;L=native['Lplus'];PT=native['PT'];g=native['UE']/native['a'];V=native['V']
    for key1,a in J.items():
        i,j,p=key1;bi=np.asarray(p,float);charge=np.zeros(V);charge[i//4]+=1;charge[j//4]-=1
        weights=L@charge;ci=float(charge@weights)
        for key2,b in J.items():
            k,l,q=key2;bj=np.asarray(q,float);B=tuple(x+y for x,y in zip(p,q));pair=words.pair(key1,a,key2,b);second+=pair
            cross=float(bi@PT@bj);reverse=words.pair(key2,b,key1,a)
            electric+=cross*(pair+reverse)
            charge2=np.zeros(V);charge2[k//4]+=1;charge2[l//4]-=1;cij=float(charge2@weights)
            coulomb+=2*cij*pair
            for sign,left,right in ((1,i,l),(-1,k,j)):
                allowed=(j==k) if sign==1 else (l==i)
                if allowed:
                    coefficient=sign*a*b;key=(left,right,B);one=words.one(key,coefficient)
                    coulomb+=2*words.density(key,coefficient,weights)+ci*one
                    delta=_delta(model,left,right)
                    electric+=.5j*kappa/variance*float(bi@PT@delta)*one
    electric*=g;coulomb*=g;total=electric+matter+coulomb
    probe=evaluate_dressing(model,kappa,variance);contact=float(direction@probe['contact']@direction)
    return dict(kappa=kappa,variance=variance,current_mean=mean,current_second_moment=second,connected_equal_time=second-mean*mean,
        literal_contact=contact,retarded_initial_tangent=total,parts=dict(primitive_electric=electric,Peierls_CAR=matter,full_real_Gauss_Coulomb=coulomb),
        operator_events=dict(H=len(H),J=len(J),matter_double_commutator=len(matter_op)),elapsed_seconds=time.monotonic()-start)


def remainder_bound(model: dict[str,Any],variance: float,direction: np.ndarray) -> dict[str,Any]:
    native=model['native'];H,J=make_operators(model,direction);g=native['UE']/native['a'];M=native['M']
    Vmatter=0.;Coulomb=g*native['V']*128**2/native['laplacian_gap'];J0=J1=J2=J3=J4=0.
    for _,coeff in H.items():Vmatter+=sum(abs(coeff[r])*model['profiles'][r]['load'] for r in range(2))
    for (_,_,phase),coeff in J.items():
        magnitude=sum(abs(coeff[r])*model['profiles'][r]['load'] for r in range(2));b=np.asarray(phase,float)
        norm=float(np.sqrt(max(0.,b@native['PT']@b)))
        J0+=magnitude;J1+=magnitude*norm;J2+=magnitude*norm**2;J3+=magnitude*norm**3;J4+=magnitude*norm**4
    cap=Vmatter+Coulomb;Knorm=g*np.sqrt(M*(M+2))/(4*variance)
    evolved_K=Knorm+2*cap;evolved_P=np.sqrt((Knorm+2*cap)/g)
    # [K,[K,J]]: phase-polynomial4g² b_i b_j p_i p_j etc.
    KK=g*g*(4*J2*evolved_K/g+4*J3*evolved_P+J4)
    # Multiplication V has bounded first gradient and Laplacian, both from literal paths.
    Vgrad=0.;Vlap=0.
    for (_,_,phase),coeff in H.items():
        b=np.asarray(phase,float);norm=float(np.sqrt(max(0.,b@native['PT']@b)))
        amount=sum(abs(coeff[r])*model['profiles'][r]['load'] for r in range(2))
        Vgrad+=norm*amount;Vlap+=norm*norm*amount
    mixed=g*(2*Vlap*J0+6*Vgrad*J1+4*cap*J2+4*Vgrad*J0*evolved_P+8*cap*J1*evolved_P)
    VV=4*cap*cap*J0;second_cap=KK+mixed+VV
    return dict(retarded_kernel_second_derivative_upper=2*J0*second_cap,
        short_time_error_formula='|chi_R(t)-chi_R(0)-t*mu| <= t²/2 * second_derivative_cap; normalizedfixedsource, no stationarity assumed',
        graph=dict(Vmatter=Vmatter,Coulomb=Coulomb,initial_free_kinetic_norm=Knorm,evolved_free_kinetic_norm=evolved_K),
        scope='Conservative native binary64 bound formula, not interval-certified arithmetic. The fullbounded-potential/constant-electric-kinetic commutator estimate retains Coulomb; no spectralpole/residue interpretation.')


def run(mean_path: Path,source_path: Path) -> dict[str,Any]:
    started=time.monotonic();raw=source_path.read_bytes();source=json.loads(raw);mean_raw=mean_path.read_bytes();state=source['descriptor']
    native=build_native(json.loads(mean_raw),int(state['N']))
    if native['N']!=2:raise ValueError('This actual causal consumer is scoped N2; no silent N6/macro truncation')
    model=build_dressing(native,int(state['axis']));variance=float(state['s_density_variance']);kappa=float(state['kappa'])
    direction=np.zeros(3*native['V'])
    direction[1::3]=np.cos(2*np.pi*native['sites'][:,0]/native['N'])
    direction/=np.linalg.norm(direction)
    if np.linalg.norm(native['D']@direction)>1e-12:raise ArithmeticError('Actual source probe is not a physical transverse current')
    dressed=measure(model,kappa,variance,direction);seed=measure(model,0.,variance,direction);bound=remainder_bound(model,variance,direction)
    # Independent global-density word contrast against actualpublic two-event ABI.
    words=SourceWords(model,kappa,variance);phase=tuple([1]+[0]*(3*native['V']-1));key=(0,1,phase);coeff=np.array([1.,.7]);weights=np.linspace(-.4,.4,native['V'])
    direct=0j
    for role in range(2):
        for x,s,r in itertools.product(range(native['V']),range(4),range(2)):
            direct+=coeff[role]*weights[x]*fcs_expect(model,[(role,0,1),(r,4*x+s,4*x+s)],np.asarray(phase),kappa,variance,prepared=words.prepared(phase))
    density_error=abs(words.density(key,coeff,weights)-direct)
    if density_error>1e-7 or abs(dressed['retarded_initial_tangent'].imag)>1e-4:
        raise ArithmeticError('Actual density/Coulomb or Hermitian causal moment control failed')
    descriptor=dict(mean_receipt_sha256=hashlib.sha256(mean_raw).hexdigest(),source_receipt_sha256=hashlib.sha256(raw).hexdigest(),
        N=2,M=native['M'],charged_counts=[1,63],probe='realcoexacty polarization withq=(pi,0,0) inlatticeangle units; allsite normalized andDf0',
        prepared_source='ExacttiledGamma PH/momentum source, all17 noncompactphysicalcoordinates, realGauss+Coulomb',
        plan='issue-222-induced-photon-delta13-continuation-20261008',plan_revision='85975947857ac7816cf0d8674f69229861014434b125a07a6996af0ecf72c4eb')
    times=[1e-12,1e-11,1e-10]
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        source=dressed,Gaussian_seed=seed,prepared_minus_seed_tangent=dressed['retarded_initial_tangent']-seed['retarded_initial_tangent'],
        exact_functional='chi_R(t)=-i theta(t) Tr(rho_prepared [J_f(t),J_f(0)]), J_f(t)=exp(iHref t)J_f exp(-iHref t). Fixedactualnewjointdensity, notA0 bareSlater andnotstationaryRPA.',
        exact_initial_moment='chi_R(0)=0; dchi_R/dt(0)=<[[Href,J_f],J_f]>. PrimitivePi², literalall64CAR andfulloperatorrhoL+rho contributionscomputedseparately. SourceC gives<J_f(t)>=0 whenfullC intertwining reviewed; no individualSlater/Wick assumption aftertimeevolution.',
        priced_short_time=[dict(t=t,retarded_linear_value=t*dressed['retarded_initial_tangent'],absolute_remainder_upper=t*t*bound['retarded_kernel_second_derivative_upper']/2) for t in times],
        remainder=bound,independent_global_density_word_error=density_error,
        limitations='Actual fullHref causalinitialmoment+conservative shorttime envelope, NOTfullfiniteclock4096 evolution orstationaryphotonpole/residue. NativeN2q isUV, notmacroIR; largeN/macroreduction, stationarity andphotonvacuum remainOPEN. Native midpoint/currentarithmetic notintervalenclosed.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,platform=platform.platform()),elapsed_seconds=time.monotonic()-started)


def serialize(value: Any) -> Any:
    if isinstance(value,complex):return dict(real=value.real,imag=value.imag)
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('mean','source','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.mean,args.source)
    args.output.write_text(json.dumps(report,indent=2,default=serialize)+'\n')
    print(json.dumps({name:report[name] for name in ('input_sha256','source','Gaussian_seed','prepared_minus_seed_tangent','independent_global_density_word_error','elapsed_seconds')},indent=2,default=serialize))

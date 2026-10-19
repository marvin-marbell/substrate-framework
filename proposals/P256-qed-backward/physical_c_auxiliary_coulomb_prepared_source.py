#!/usr/bin/env python3
"""Literal Coulomb in original FIXEDbody fullK prefix and RAW creator source.

Same executed continuum0339/prepared0eae construction: Gaussian3.5, beta100,
Naux625m, kappa.01, UA2, physicaltheta u=UA/beta=.02.  Each C is an exact
Gaussian HS integral of the centered ALLoccupation character.  K_C prefix
histories and their derivatives remain coherent canonical Slater sums;
there is no instantaneous-sea source/preparation substitution.  A sea
compression is used ONLY as a neutral-normalizer Rayleigh lower bound.
"""
from __future__ import annotations
import argparse
from decimal import Decimal
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh,expm_frechet,lu_factor,lu_solve
from scipy.special import ndtri
from scipy.stats import qmc
import physical_c_bare_electric_auxiliary_prepared_creator as prepared
import physical_c_auxiliary_coulomb_sf_output as sf
coul=sf.coul;continuum=prepared.continuum;mp=continuum.mp
SOURCE_BYTES=Path(__file__).read_bytes();PREPARED_BYTES=prepared.SOURCE_BYTES;SF_BYTES=sf.SOURCE_BYTES
if hashlib.sha256(PREPARED_BYTES).hexdigest()!='0eae61f80f1cd376e116b65af072566108fcfa567e1e28bd5797b77a2ed8de41':raise ArithmeticError('Coul binds executed prepared0eae epoch')


def logsmall(matrix: NDArray[np.complex128]) -> complex:
    value,_=continuum.small_logdet(matrix)
    return complex(value)


def graph_of(frame: NDArray[np.complex128],change: NDArray[np.complex128]) -> NDArray[np.complex128]:
    chart=change.conj().T@frame;n=frame.shape[1]
    return np.linalg.solve(chart[n:].T,chart[:n].T).T


def canonical(R: NDArray[np.complex128],change: NDArray[np.complex128]) -> dict:
    n=len(R);metric=np.eye(n)+R.conj().T@R;e,v=eigh(metric)
    if np.min(e)<=0:raise ArithmeticError('Actual graph normalization lost positivity')
    H=(v/np.sqrt(e))@v.conj().T
    return dict(R=R,H=H,W=change@np.vstack((R,np.eye(n)))@H,lognorm=logsmall(R.conj().T@R).real,
        t=np.zeros((2*n,n),np.complex128),s=1.+0j)


def build(order: int) -> dict:
    with mp.workdps(45):
        background=continuum.normalized_background();cache={};control=continuum.run_order(order,cache,background)
        nodes=np.array(control['real_field_nodes']);states=[cache[str(mp.nstr(mp.mpf(str(x))/10,45))] for x in nodes]
        origin=cache['0.0'];law=continuum.loops.make_law(.01,background)
        change=np.array(law.change,np.complex128);body=canonical(graph_of(continuum.float_matrix(origin['W']),change),change)
        layers=[];n=body['R'].shape[0]
        for q,state,ll in zip(nodes,states,control['whole_SF_scalar_valence_loglambda'],strict=True):
            frame=continuum.float_matrix(state['fullframe']);h=frame@np.diag([float(x) for x in state['eigenvalues']])@frame.conj().T
            profile=np.array(background['a'],float);jet=law.potential(q*profile/10,background['phi'],background['links'],(profile/10,profile/10))
            U,Up=expm_frechet(h/2,jet['hx']/2,compute_expm=True);chart=change.conj().T@U@change;prime=change.conj().T@Up@change
            Rsea=graph_of(continuum.float_matrix(state['W']),change);K=np.linalg.solve(chart[n:,n:],chart[n:,:n])
            layers.append(dict(U=U,Up=Up,A=chart[:n,:n],B=chart[:n,n:],C=chart[n:,:n],D=chart[n:,n:],
                Ap=prime[:n,:n],Bp=prime[:n,n:],Cp=prime[n:,:n],Dp=prime[n:,n:],K=K,vacsmall=logsmall(K@Rsea),
                positive=frame[:,n:],negative=frame[:,:n],plus=np.exp(np.array([float(x) for x in state['eigenvalues'][n:]])),
                minus=np.exp(np.array([float(x) for x in state['eigenvalues'][:n]])),
                loglambda=float(ll),prefx=float(jet['prefx']),sea=canonical(Rsea,change)))
    weights=np.array(control['positive_Lebesgue_weights']);width=3.5;ua=2.;nf=625000000
    boundary=np.sqrt(weights)*np.exp(-width*nodes**2/2);boundary/=np.linalg.norm(boundary)
    heat=np.sqrt(weights[:,None]*weights[None,:])*np.exp(-(nodes[:,None]-nodes[None,:])**2/(4*ua))/np.sqrt(4*np.pi*ua)
    graph=coul.graph_data((4,3,1));sea=coul.sea(.01,graph,np.zeros_like(graph['a']));cov=nf*sea['cov']
    prior=np.diag(.04/graph['gauge_eigenvalues']);proposal=np.linalg.inv(np.linalg.inv(prior)+graph['gauge_frame'].T@cov@graph['gauge_frame']);e,v=eigh(proposal)
    context=dict(order=order,nodes=nodes,weights=weights,body=body,layers=layers,change=change,heat=heat,boundary=boundary,
        width=width,UA=ua,NF=nf,graph=graph,cov=cov,HSmap=(v*np.sqrt(e)).T@graph['gauge_frame'].T,
        jacobian=float(np.sqrt(np.linalg.det(proposal)/np.linalg.det(prior))),control=control)
    context['first']=[apply_layer(context,body,layer)[0:2] for layer in layers]
    context['zero_histories']=histories(context,np.zeros(len(graph['targets'])))
    return context


def apply_layer(context: dict,component: dict,layer: dict) -> tuple[dict,complex,complex]:
    R,H=component['R'],component['H'];bottom=layer['C']@R+layer['D'];top=layer['A']@R+layer['B']
    Rnew=np.linalg.solve(bottom.T,top.T).T;result=canonical(Rnew,context['change'])
    logratio=logsmall(layer['K']@R)-layer['vacsmall']+(result['lognorm']-component['lognorm'])/2
    a=layer['positive'].conj().T@component['W'];c=layer['negative'].conj().T@component['W']
    correction=logsmall(np.linalg.solve(a.conj().T@(layer['plus'][:,None]*a),c.conj().T@(layer['minus'][:,None]*c))).real
    loss,_=continuum.small_logdet(c.conj().T@c,-1)
    # Exact prepared0eae positive/negative norm decomposition, with cached frames.
    # Graph phase alone fixes the canonical occupied gauge; it does not normalize rows.
    amplitude=complex(np.exp(layer['loglambda']/2+context['NF']*(loss.real+correction)/2+1j*context['NF']*logratio.imag))
    inverse=np.linalg.solve(H,np.linalg.solve(bottom,result['H']))
    mapped=layer['U']@component['t']@inverse;occupied=result['W'].conj().T@mapped
    result['t']=mapped-result['W']@occupied;result['s']=component['s']+context['NF']*np.trace(occupied)
    return result,amplitude,logratio


def rotate(context: dict,component: dict,eta: np.ndarray) -> dict:
    d=np.repeat(np.exp(1j*eta),2);R=d[:,None]*component['R']/d[None,:];H=d[:,None]*component['H']/d[None,:]
    n=len(R);W=context['change']@np.vstack((R,np.eye(n)))@H
    return dict(R=R,H=H,W=W,lognorm=component['lognorm'],t=np.zeros_like(W),s=1.+0j)


def histories(context: dict,eta: np.ndarray) -> tuple[list,list,NDArray[np.complex128],NDArray[np.complex128]]:
    before=[];after=[];coefficients=[];sfcoefficients=[];nf=context['NF'];width=context['width']
    importance=math.exp(.5*float(eta@context['cov']@eta))
    for i,layer in enumerate(context['layers']):
        for j,(first,firstamp) in enumerate(context['first']):
            rotated=rotate(context,first,eta);component,amp,_=apply_layer(context,rotated,layer)
            R=rotated['R'];bottom=layer['C']@R+layer['D'];bottomp=layer['Cp']@R+layer['Dp']
            Rp=np.linalg.solve(bottom.T,(layer['Ap']@R+layer['Bp']-component['R']@bottomp).T).T
            image=context['change']@np.vstack((Rp,np.zeros_like(Rp)))@component['H']
            horizontal=image-component['W']@(component['W'].conj().T@image)
            occupied_trace=np.trace(np.linalg.solve(bottom,bottomp))+np.trace(np.linalg.solve(np.eye(len(Rp))+component['R'].conj().T@component['R'],component['R'].conj().T@Rp))
            derivative=layer['prefx']/2+occupied_trace
            component['s']=np.sqrt(width/2)*context['nodes'][i]+(context['nodes'][i]-context['nodes'][j])/(2*context['UA']*np.sqrt(2*width))-nf*derivative/np.sqrt(2*width)
            component['t']=-horizontal/np.sqrt(2*width)
            final,finalamp,_=apply_layer(context,component,layer)
            coefficient=context['heat'][i,j]*context['boundary'][j]*firstamp*amp*importance
            before.append(component);after.append(final);coefficients.append(coefficient);sfcoefficients.append(coefficient*finalamp)
    return before,after,np.array(coefficients,np.complex128),np.array(sfcoefficients,np.complex128)


def pair_parts(context: dict,left: dict,right: dict,eta: np.ndarray) -> tuple[complex,complex,complex,complex]:
    log=coul.graph_character_log(left,right,eta[None])[0];determinant=complex(np.exp(context['NF']*log))
    q=np.repeat(np.exp(1j*eta),4);M=left['W'].conj().T@(q[:,None]*right['W']);lu=lu_factor(M)
    a=lu_solve(lu,left['t'].conj().T@(q[:,None]*right['W']))
    b=lu_solve(lu,left['W'].conj().T@(q[:,None]*right['t']))
    h=lu_solve(lu,left['t'].conj().T@(q[:,None]*right['t']))
    nf=context['NF'];tr_a=np.trace(a);tr_b=np.trace(b)
    scalar=determinant*np.conj(left['s'])*right['s']
    tangent=determinant*(nf*np.trace(h-a@b)+nf*nf*tr_a*tr_b)
    interference=determinant*nf*(np.conj(left['s'])*tr_b+tr_a*right['s'])
    return determinant,scalar,tangent,interference


def quadratic(context: dict,left: list,right: list,coeffleft: np.ndarray,coeffright: np.ndarray,eta: np.ndarray,fieldheat: bool) -> np.ndarray:
    order=context['order'];result=np.zeros(4,np.complex128)
    for i in range(order):
        ls=range(order) if fieldheat else (i,)
        for l in ls:
            factor=context['heat'][i,l] if fieldheat else 1.
            for j,k in itertools.product(range(order),repeat=2):
                a=i*order+j;b=l*order+k
                result+=factor*np.conj(coeffleft[a])*coeffright[b]*np.array(pair_parts(context,left[a],right[b],eta))
    return result


def baseline(context: dict) -> dict:
    zero=np.zeros(len(context['graph']['targets']));before,after,coef,sfcoef=context['zero_histories']
    raw=quadratic(context,before,before,coef,coef,zero,False);source=quadratic(context,after,after,sfcoef,sfcoef,zero,True)
    context['zero_raw']=raw
    return dict(preparation_Gram=float(raw[0].real),raw_creator_normalized_Gram=float(np.sum(raw[1:]).real/raw[0].real),
        wholehalf_creator_normalized_interval=[float(np.sum(source[1:]).real/(raw[0].real*float(x))) for x in reversed(context['control']['spectral_A_fullcanonical_numeric_certificate']['complete_ALLoccupation_K_normalizer_interval'])],
        scalar_wholehalf_Gram=float(source[1].real),tangent_wholehalf_Gram=float(source[2].real),scalar_tangent_interference=float(source[3].real))


def integral(context: dict,bits: int,prefix_C: bool) -> dict:
    rank=len(context['graph']['gauge_eigenvalues']);dimensions=3*rank if prefix_C else rank
    uniforms=qmc.Sobol(dimensions,scramble=True,rng=np.random.default_rng(222944)).random_base2(bits)
    if np.any((uniforms<=0)|(uniforms>=1)):raise ArithmeticError('Inverse-normal endpoint; no probability clipping')
    normal=ndtri(uniforms);zero=np.zeros(len(context['graph']['targets']));q=int(sf.scaled_integer_charge('100')['Q'])
    source_pair=zero.copy();source_pair[0]=1;source_pair[1]=-1;background=q*source_pair
    accum={name:np.zeros(8,np.complex128) for name in ('neutral','chosen_integerQ')};compression=np.zeros((context['order'],context['order']),np.complex128)
    absolute={name:np.zeros(2) for name in accum};zero_histories=context['zero_histories']
    for sample in normal:
        if prefix_C:
            eta=sample[:rank]@context['HSmap'];zeta=sample[rank:2*rank]@context['HSmap'];xi=sample[2*rank:]@context['HSmap']
            left,leftsf,cl,csl=histories(context,eta);right,rightsf,cr,csr=histories(context,zeta)
        else:
            eta=zero;zeta=zero;xi=sample@context['HSmap'];left,leftsf,cl,csl=zero_histories;right,rightsf,cr,csr=zero_histories
        raw=quadratic(context,left,right,cl,cr,zero,False) if prefix_C else context['zero_raw']
        final=quadratic(context,leftsf,rightsf,csl,csr,xi,True)*math.exp(.5*float(xi@context['cov']@xi))
        for name,bg in (('neutral',zero),('chosen_integerQ',background)):
            prep_phase=np.exp(1j*((zeta-eta)@bg));source_phase=prep_phase*np.exp(1j*(xi@bg))
            # Adjoint orbit (eta,zeta,xi)->(zeta,eta,-xi) is measure preserving.
            # It conjugates the complete coherent source integrand, not a source gate.
            accum[name][:4]+=raw*prep_phase;accum[name][4:]+=final*source_phase
            absolute[name][0]+=abs(raw[0]);absolute[name][1]+=abs(np.sum(final[1:]))
        for i,j in itertools.product(range(context['order']),repeat=2):
            a=context['layers'][i];b=context['layers'][j]
            char=np.exp(context['NF']*coul.graph_character_log(a['sea'],b['sea'],xi[None])[0])
            compression[i,j]+=context['heat'][i,j]*np.exp((a['loglambda']+b['loglambda'])/2)*char*math.exp(.5*float(xi@context['cov']@xi))
    count=len(normal);jac=context['jacobian'];compression*=jac/count
    herm=float(np.linalg.norm(compression-compression.conj().T));neutral_lower=float(np.max(np.linalg.eigvalsh((compression+compression.conj().T)/2)))
    neutral_upper=float(context['control']['spectral_A_fullcanonical_numeric_certificate']['complete_ALLoccupation_K_normalizer_interval'][1])
    if neutral_lower<=0 or neutral_lower>neutral_upper:raise ArithmeticError('Actual Coul neutral Rayleigh/ordering bracket failed')
    cases={}
    for name,totals in accum.items():
        totals/=count;raw=totals[:4];final=totals[4:];prep_power=2 if prefix_C else 0
        prep=jac**prep_power*raw[0].real;raw_total=float(np.sum(raw[1:]).real);source_total=float(np.sum(final[1:]).real)
        if prep<=0 or raw_total<=0 or source_total<=0:raise ArithmeticError('Actual coherent Coul source Gram positivity failed')
        cases[name]=dict(original_FIXEDbody_preparation_Gram=float(prep),raw_creator_normalized_Gram=raw_total/raw[0].real,
            wholehalf_creator_normalized_interval=[jac*source_total/(raw[0].real*neutral_upper),jac*source_total/(raw[0].real*neutral_lower)],
            wholehalf_scalar_normalized_upper=jac*final[1].real/(raw[0].real*neutral_lower),
            wholehalf_tangent_normalized_upper=jac*final[2].real/(raw[0].real*neutral_lower),
            wholehalf_scalar_tangent_interference_normalized_upper=jac*final[3].real/(raw[0].real*neutral_lower),
            prep_signed_cancellation_price=absolute[name][0]/(count*abs(raw[0])),
            source_signed_cancellation_price=absolute[name][1]/(count*abs(np.sum(final[1:]))),
            unpaired_quadrature_imaginary_Gram=float(np.sum(final[1:]).imag),
            common_neutral_normalizer=True)
    return dict(bits=bits,samples=count,HSdimensions=dimensions,actual_Coul_in_preparation_prefix=prefix_C,
        neutral_normalizer_numeric_interval=[neutral_lower,neutral_upper],neutral_compression_Hermiticity_defect=herm,
        exact_importance_jacobian_per_C=jac,charge_contract=sf.scaled_integer_charge('100'),cases=cases,
        limitation='Numerical evaluations of exact coherent ALLoccupation Slater identities. QMC and float errors are disclosed, NOT outward-rounded quadrature/macro/Gauss/pole certification. Neutral upper uses C<=I and executed ALLoccupation K0 upper; lower is actual C sea Rayleigh compression ONLY, not a preparation/source gate.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--capsule',required=True);parser.add_argument('--orders',default='3');parser.add_argument('--bits',default='4,6,8');args=parser.parse_args();start=time.time()
    orders=[int(x) for x in args.orders.split(',')];bits=[int(x) for x in args.bits.split(',')];rows=[]
    for order in orders:
        context=build(order);rows.append(dict(order=order,noC_baseline=baseline(context),C_source_only_on_unchanged_K0_prefix=[integral(context,x,False) for x in bits],
            actual_KC_prefix_and_final_source=[integral(context,x,True) for x in bits]))
    execution=dict(source_sha256=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
        prepared_source_sha256=hashlib.sha256(PREPARED_BYTES).hexdigest(),prepared_source_bytes_at_IMPORT=PREPARED_BYTES.decode(),SF_source_sha256=hashlib.sha256(SF_BYTES).hexdigest(),SF_source_bytes_at_IMPORT=SF_BYTES.decode(),
        continuum_source_bytes_at_IMPORT=prepared.CONTINUUM_BYTES.decode(),bundle_source_bytes_at_IMPORT=continuum.BUNDLE_BYTES.decode(),loop_source_bytes_at_IMPORT=continuum.bundle.LOOP_BYTES.decode(),
        core_source_bytes_at_IMPORT=coul.loops.CORE_BYTES.decode(),frozen_dependencies=coul.loops.IMPORT_EPOCH,
        argv=sys.argv,input_parameters=dict(orders=orders,bits=bits,NF=625000000,k=.01,beta=100,UA=2,u=.02,Gaussian_width=3.5,seed=222944,original_FIXED_meanbody=True),
        python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,elapsed_seconds=time.time()-start,
        startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES and Path(str(prepared.__file__)).read_bytes()==PREPARED_BYTES)
    report=dict(rows=rows,source='RAW a_refdag on actual FULL uncompressed K_C applied once to original constant meanbodySlater x Gaussian3.5. Initial C-source-only branch is separately labelled unchanged K0 prefix. Scalar+tangent and crossinterference retained, finalSF actual, one positive NEUTRALnormalizer for both backgrounds.',
        centering='Canonical graph charge rotation R->Deta R Deta^-1 exactly cancels detDeta with perflavour e^-2i sumeta. External integerQ phase stays exp(i eta*Qv). No quarter, source, sea, or pole gate.',
        macro_scope='Finite same-declaration beta100 integerQ3 (e0=.3) control; NOT literal macroQ. Chosen macro e0=.302822... is not changed or screened into a tiny bare charge. Complete Gauss/centre/active OriginalPsi/Ref3 source and physicalmass1 remain independent obligations.')
    Path(args.capsule).write_text(json.dumps(dict(report=report,execution=execution),indent=2)+'\n');print(json.dumps(dict(capsule=args.capsule,report=report,elapsed_seconds=execution['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':main()

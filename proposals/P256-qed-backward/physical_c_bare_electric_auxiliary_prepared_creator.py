#!/usr/bin/env python3
"""RAWB after ACTUAL uncompressed K on original FIXEDbody auxiliary boundary.

Finite positive REALfield quadrature, meanPhi0/SU4I only, beforeGauss/Coul.
K histories remain coherent Slater sums; no powers of Iadj K I are substituted.
No auxiliary/source/pole gate or dressedcreator replacement is introduced.
"""
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
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
from scipy.linalg import expm_frechet
import physical_c_bare_electric_auxiliary_continuum as continuum
mp=continuum.mp;loops=continuum.loops
SOURCE_BYTES=Path(__file__).read_bytes();CONTINUUM_BYTES=Path(continuum.__file__).read_bytes()
if hashlib.sha256(CONTINUUM_BYTES).hexdigest()!='03390980f5fe7cabec1447247c3472635bd1ed3da5a91c68111442eabdd761f8':raise ArithmeticError('Prepared source binds executed continuum0339')


def polar_frame(image):
    values,vectors=np.linalg.eigh(image.conj().T@image)
    if min(values)<=0:raise ArithmeticError('Actual SF image lost rank')
    inverse=vectors@np.diag(1/np.sqrt(values))@vectors.conj().T
    return image@inverse,inverse


def log_overlap(left,right):
    complement=right-left@(left.conj().T@right)
    loss,remainder=continuum.small_logdet(complement.conj().T@complement,-1)
    phase=np.angle(np.linalg.slogdet(left.conj().T@right)[0])
    return loss.real/2+1j*phase,remainder


def slater_inner(left,right,naux):
    """Coherent scalar-plus-tangent tensorSlater inner product, ALLflavours."""
    v,t,s=left;w,u,z=right;m=v.conj().T@w
    a=np.linalg.solve(m,t.conj().T@w);b=np.linalg.solve(m,v.conj().T@u);h=np.linalg.solve(m,t.conj().T@u)
    log,_=log_overlap(v,w)
    return np.exp(naux*log)*(np.conj(s)*z+np.conj(s)*naux*np.trace(b)+naux*np.trace(a)*z+naux*np.trace(h-a@b)+naux*naux*np.trace(a)*np.trace(b))


def apply_SF(frame,tangent,scalar,state,U,loglambda,naux):
    size=len(state['eigenvalues']);n=size//2
    positive=continuum.float_matrix(state['fullframe'][:,n:]);negative=continuum.float_matrix(state['fullframe'][:,:n])
    a=positive.conj().T@frame;c=negative.conj().T@frame
    plus=np.exp(np.array([float(e) for e in state['eigenvalues'][n:]]));minus=np.exp(np.array([float(e) for e in state['eigenvalues'][:n]]))
    correction,_=continuum.small_logdet(np.linalg.solve(a.conj().T@(plus[:,None]*a),c.conj().T@(minus[:,None]*c)))
    loss,_=continuum.small_logdet(c.conj().T@c,-1)
    norm=np.exp(loglambda/2+naux*(loss.real+correction.real)/2)
    v,inverse=polar_frame(U@frame);mapped=U@tangent@inverse
    occupied=v.conj().T@mapped
    return (v,mapped-v@occupied,scalar+naux*np.trace(occupied)),norm


def run_order(order: int,cache: dict[str,Any],background: dict[str,Any]) -> dict[str,Any]:
    control=continuum.run_order(order,cache,background)
    nodes=np.array(control['real_field_nodes']);weights=np.array(control['positive_Lebesgue_weights']);width=control['Gaussian_boundary_width'];naux=625000000.;ua=2.
    states=[cache[str(mp.nstr(mp.mpf(str(q))/10,45))] for q in nodes];origin=cache['0.0'];body=continuum.float_matrix(origin['W'])
    loglambda=np.array(control['whole_SF_scalar_valence_loglambda']);normalizers=list(map(float,control['spectral_A_fullcanonical_numeric_certificate']['complete_ALLoccupation_K_normalizer_interval']))
    law=loops.make_law(.01,background);profile=np.array(background['a'],float)
    operators=[];derivatives=[];pref_derivatives=[]
    for q,state in zip(nodes,states,strict=True):
        frame=continuum.float_matrix(state['fullframe']);h=frame@np.diag([float(x) for x in state['eigenvalues']])@frame.conj().T
        jet=law.potential(q*profile/10,background['phi'],background['links'],(profile/10,profile/10))
        U,Up=expm_frechet(h/2,jet['hx']/2,compute_expm=True)
        operators.append(U);derivatives.append(Up);pref_derivatives.append(jet['prefx'])
    first=[];firstnorm=[]
    zero=np.zeros_like(body)
    for state,U,ll in zip(states,operators,loglambda,strict=True):
        component,norm=apply_SF(body,zero,1.,state,U,ll,naux);first.append(component[0]);firstnorm.append(norm)
    boundary=np.sqrt(weights)*np.exp(-width*nodes**2/2);boundary/=np.linalg.norm(boundary)
    R=np.sqrt(weights[:,None]*weights[None,:])*np.exp(-(nodes[:,None]-nodes[None,:])**2/(4*ua))/np.sqrt(4*np.pi*ua)
    prepared=[];creator=[];creator_after_SF=[];weights_prepared=[];weights_after=[];cancel_scale=0.
    for i in range(order):
        row=[];brow=[];sfrow=[];crow=[];scrow=[]
        for j in range(order):
            component,norm=apply_SF(first[j],zero,1.,states[i],operators[i],loglambda[i],naux)
            v=component[0];Y=operators[i]@first[j];_,inverse=polar_frame(Y);Yp=derivatives[i]@first[j]
            derivative_image=Yp@inverse;occupied=v.conj().T@derivative_image
            a=pref_derivatives[i]/2+np.trace(occupied)
            horizontal=derivative_image-v@occupied
            cancel_scale=max(cancel_scale,naux*(abs(pref_derivatives[i]/2)+abs(np.trace(occupied))))
            bscalar=np.sqrt(width/2)*nodes[i]+(nodes[i]-nodes[j])/(2*ua*np.sqrt(2*width))-naux*a/np.sqrt(2*width)
            btangent=-horizontal/np.sqrt(2*width)
            row.append((v,zero,1.));brow.append((v,btangent,bscalar))
            sfcomponent,sfnorm=apply_SF(v,btangent,bscalar,states[i],operators[i],loglambda[i],naux);sfrow.append(sfcomponent)
            coefficient=R[i,j]*firstnorm[j]*norm*boundary[j]
            crow.append(coefficient);scrow.append(coefficient*sfnorm)
        prepared.append(row);creator.append(brow);creator_after_SF.append(sfrow);weights_prepared.append(crow);weights_after.append(scrow)
    prep=raw=0j
    for i in range(order):
        for j in range(order):
            for k in range(order):
                factor=np.conj(weights_prepared[i][j])*weights_prepared[i][k]
                prep+=factor*slater_inner(prepared[i][j],prepared[i][k],naux)
                raw+=factor*slater_inner(creator[i][j],creator[i][k],naux)
    half=0j;S_source=0j
    for i in range(order):
        for l in range(order):
            for j in range(order):
                for k in range(order):
                    term=np.conj(weights_after[i][j])*weights_after[l][k]*slater_inner(creator_after_SF[i][j],creator_after_SF[l][k],naux)
                    half+=R[i,l]*term
                    if i==l:S_source+=term
    sea_rows=[(continuum.float_matrix(state['W']),zero,1.0) for state in states]
    point_lambdas=np.exp(loglambda)
    sea_amplitudes=np.asarray([sum(weights_prepared[i][j]*slater_inner(sea_rows[i],creator[i][j],naux)
        for j in range(order)) for i in range(order)])
    sea_half=0j;sea_total_half=0j
    for i in range(order):
        for l in range(order):
            sea_half+=R[i,l]*np.conj(sea_amplitudes[i])*sea_amplitudes[l]*math.sqrt(point_lambdas[i]*point_lambdas[l])*slater_inner(sea_rows[i],sea_rows[l],naux)
            sea_total_half+=R[i,l]*np.conj(sea_amplitudes[i])*math.sqrt(point_lambdas[i])*sum(
                weights_after[l][j]*slater_inner(sea_rows[i],creator_after_SF[l][j],naux) for j in range(order))
    raw_auxiliary=raw.real-float(np.vdot(sea_amplitudes,sea_amplitudes).real)
    SF_auxiliary=S_source.real-float(np.vdot(sea_amplitudes,point_lambdas*sea_amplitudes).real)
    half_auxiliary=(half-2*sea_total_half.real+sea_half).real
    if min(raw_auxiliary,SF_auxiliary,half_auxiliary)<=0:raise ArithmeticError('Actual original RAWB auxiliary decomposition lost positivity')
    if half_auxiliary>float(np.max(np.linalg.eigvalsh(R)))*SF_auxiliary*(1+2e-6):raise ArithmeticError('Auxiliary source violated positive SF-sandwich bound')
    for name,value in [('preparation',prep),('rawcreator',raw),('wholehalf',half),('sourceSF',S_source)]:
        if value.real<=0 or abs(value.imag)>1e-6*value.real:raise ArithmeticError(f'Original prepared {name} Gram lost positivity/reality')
    previous=control['original_FIXED_meanbody_projected_FULL_K_output_Gram']+control['original_FIXED_meanbody_UNCOMPRESSED_FULL_K_defect_Gram']
    if abs(prep.real-previous)>2e-7*previous:raise ArithmeticError('Coherent FULLK histories disagree original uncompressed norm')
    rnorm=float(np.max(np.linalg.eigvalsh(R)))
    if half.real>rnorm*S_source.real*(1+1e-7):raise ArithmeticError('Original RAWB source violated positive SF-sandwich bound')
    return dict(order=order,original_FIXEDbody_FULL_K_preparation_Gram=prep.real,independent_original_uncompressed_Gram=previous,
        actual_RAW_creator_Gram_on_normalized_original_K_boundary=raw.real/prep.real,
        actual_WHOLEhalf_RAW_creator_Gram_on_original_K_boundary_interval=[half.real/(prep.real*normalizers[1]),half.real/(prep.real*normalizers[0])],
        actual_source_SF_Gram_on_normalized_original_K_boundary=S_source.real/prep.real,
        measured_R_norm=rnorm,positive_SF_sandwich_wholehalf_Gram_upper=rnorm*S_source.real/(prep.real*normalizers[0]),
        actual_RAW_auxiliary_Gram_on_original_K_boundary=raw_auxiliary/prep.real,
        actual_WHOLEhalf_auxiliary_Gram_on_original_K_boundary_interval=[half_auxiliary/(prep.real*normalizers[1]),half_auxiliary/(prep.real*normalizers[0])],
        actual_SF_auxiliary_Gram_on_original_K_boundary=SF_auxiliary/prep.real,
        positive_SF_sandwich_auxiliary_Gram_upper=rnorm*SF_auxiliary/(prep.real*normalizers[0]),
        sea_WHOLEhalf_Gram_on_original_K_boundary_interval=[sea_half.real/(prep.real*normalizers[1]),sea_half.real/(prep.real*normalizers[0])],
        sea_auxiliary_WHOLEhalf_interference_on_original_K_boundary_interval=[(half.real-sea_half.real-half_auxiliary)/(prep.real*normalizers[1]),(half.real-sea_half.real-half_auxiliary)/(prep.real*normalizers[0])],
        auxiliary_split='Pointwise instantaneous Slater sea projector is ONLY an orthogonal proof decomposition of unchanged RAWB source; its complement and sea/complement interference are retained, not a preparation gate.',
        NF_scalar_derivative_cancellation_scale=cancel_scale,
        source='RAW a_refdag on ACTUAL FULLK applied to original constant meanbodySlater times declared Gaussian3.5 boundary. Coherent field histories and ALLcanonicalNflavour derivative/pair/interference retained, not sea-line substitution or powersofcompressedIadjKI.',
        limitation='Positive REALfield Nystrom regulator; numerical SF-Frechet/trace-log evaluations, not outward-rounded originalcontinuum/macro/Gauss/sourcepole certificate. Prefactor/valence derivative cancellation disclosed; no scalar-row normalization, floor or polegate.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();started=time.time()
    with mp.workdps(45):
        background=continuum.normalized_background();cache={};cases=[run_order(order,cache,background) for order in (3,5)]
    inputs=json.dumps(dict(kappa='.01',beta=100,Naux=625000000,UA=2,orders=[3,5],original_fixedbody=True,Gaussian_width=3.5,preparation='ACTUAL uncompressed K once',source='RAW a_refdag AFTERpreparation'),sort_keys=True,separators=(',',':'))
    result: dict[str,Any]=dict(cases=cases,execution=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),continuum_at_IMPORT=hashlib.sha256(CONTINUUM_BYTES).hexdigest(),continuum_bytes_at_IMPORT=CONTINUUM_BYTES.decode(),bundle_at_IMPORT=hashlib.sha256(continuum.BUNDLE_BYTES).hexdigest(),bundle_bytes_at_IMPORT=continuum.BUNDLE_BYTES.decode(),loop_at_IMPORT=hashlib.sha256(continuum.bundle.LOOP_BYTES).hexdigest(),loop_bytes_at_IMPORT=continuum.bundle.LOOP_BYTES.decode(),input_json=inputs,input_sha256=hashlib.sha256(inputs.encode()).hexdigest(),runtime_seconds=time.time()-started,executable=sys.executable,python=sys.version,numpy=np.__version__,startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES and Path(continuum.__file__).read_bytes()==CONTINUUM_BYTES and Path(continuum.bundle.__file__).read_bytes()==continuum.BUNDLE_BYTES and Path(loops.__file__).read_bytes()==continuum.bundle.LOOP_BYTES))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['cases']))
if __name__=='__main__':main()

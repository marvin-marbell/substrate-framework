#!/usr/bin/env python3
"""Independent finite-Fock validation of literal three-C prepared RAW source.

Actual frozen C law on disclosed two-site REAL-cover geometry; Nflavour1/2,
all70 local-N4 states and all4900 two-flavour states retained.  Direct positive
charge-diagonal C and full Fock SF/Frechet source are compared to coherent
Slater/HS histories, including scalar/tangent interference.  This is an
algebraic native control, NOT a beta100 or macrovolume candidate simulation.
"""
from __future__ import annotations
import argparse
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
from scipy.linalg import eigh,expm_frechet
from scipy.special import roots_hermitenorm
import physical_c_auxiliary_coulomb_prepared_source as source
coul=source.coul
SOURCE_BYTES=Path(__file__).read_bytes();COUPLED_BYTES=source.SOURCE_BYTES


def build(nf: int) -> tuple[dict,dict]:
    b=coul.loops.background((2,1,1));law=coul.loops.make_law(.01,b);change=np.array(law.change,np.complex128)
    profile=np.array([[.31,.27,-.13],[.09,-.22,.18]])/10;q=.3
    origin=law.potential(np.zeros_like(profile),b['phi'],b['links']);e0,v0=eigh(origin['h']);n=len(e0)//2
    body=source.canonical(source.graph_of(v0[:,n:],change),change);vac=float(origin['pref']+np.sum(e0[n:]))
    state=law.potential(q*profile,b['phi'],b['links'],(profile,profile));e,v=eigh(state['h']);U,Up=expm_frechet(state['h']/2,state['hx']/2,compute_expm=True)
    chart=change.conj().T@U@change;prime=change.conj().T@Up@change;Rsea=source.graph_of(v[:,n:],change);K=np.linalg.solve(chart[n:,n:],chart[n:,:n])
    layer=dict(U=U,Up=Up,A=chart[:n,:n],B=chart[:n,n:],C=chart[n:,:n],D=chart[n:,n:],Ap=prime[:n,:n],Bp=prime[:n,n:],Cp=prime[n:,:n],Dp=prime[n:,n:],
        K=K,vacsmall=source.logsmall(K@Rsea),loglambda=nf*float(state['pref']+np.sum(e[n:])-vac),prefx=float(state['prefx']),
        positive=v[:,n:],negative=v[:,:n],plus=np.exp(e[n:]),minus=np.exp(e[:n]),sea=source.canonical(Rsea,change))
    heat=1/math.sqrt(4*math.pi*2);context: dict=dict(order=1,nodes=np.array([q]),body=body,layers=[layer],change=change,heat=np.array([[heat]]),boundary=np.ones(1),width=3.5,UA=2,NF=nf,cov=np.zeros((2,2)))
    context['first']=[source.apply_layer(context,body,layer)[0:2]]
    H,masks=coul.alpha.second_quantized(state['h'],4);Hp,_=coul.alpha.second_quantized(state['hx'],4)
    fullSF,fullSFprime=expm_frechet((H+(state['pref']-vac)*np.eye(len(masks)))/2,(Hp+state['prefx']*np.eye(len(masks)))/2,compute_expm=True)
    rho=np.array([[(int(mask)&15).bit_count()-2,(int(mask)>>4).bit_count()-2] for mask in masks],float)
    slater=np.array([np.linalg.det(body['W'][[i for i in range(8) if int(mask)&(1<<i)]]) for mask in masks])
    tensor=slater if nf==1 else np.kron(slater,slater);density=rho if nf==1 else (rho[:,None,:]+rho[None,:,:]).reshape(-1,2)
    L=np.array([[2.,-2.],[-2.,2.]]);Lplus=np.linalg.pinv(L);frame=np.array([[-1.],[1.]])/math.sqrt(2)
    direct=dict(SF=fullSF,SFprime=fullSFprime,body=tensor,rho=density,Lplus=Lplus,HSmap=frame[:,0]*math.sqrt(.04/4),heat=heat,q=q)
    return context,direct


def tensor_apply(matrix: np.ndarray,vector: np.ndarray,nf: int) -> np.ndarray:
    if nf==1:return matrix@vector
    x=vector.reshape(len(matrix),len(matrix));return (matrix@x@matrix.T).ravel()


def tensor_prime(matrix: np.ndarray,prime: np.ndarray,vector: np.ndarray,nf: int) -> np.ndarray:
    if nf==1:return prime@vector
    x=vector.reshape(len(matrix),len(matrix));return (prime@x@matrix.T+matrix@x@prime.T).ravel()


def direct_case(context: dict,direct: dict,Q: int) -> dict:
    nf=context['NF'];SF=direct['SF'];prime=direct['SFprime'];width=context['width'];heat=direct['heat']
    background=np.array([Q,-Q]);rho=direct['rho']+background
    C=np.exp(-.02*np.einsum('ni,ij,nj->n',rho,direct['Lplus'],rho))
    first=tensor_apply(SF,direct['body'],nf);intermediate=C*first
    prepared=heat*tensor_apply(SF,intermediate,nf)
    raw=math.sqrt(width/2)*direct['q']*prepared-heat*tensor_prime(SF,prime,intermediate,nf)/math.sqrt(2*width)
    finalSF=tensor_apply(SF,raw,nf);wholehalf=heat*np.vdot(finalSF,C*finalSF)
    return dict(preparation_Gram=float(np.vdot(prepared,prepared).real),raw_creator_Gram=float(np.vdot(raw,raw).real),wholehalf_creator_Gram=float(wholehalf.real),
        wholehalf_creator_Gram_imag=float(wholehalf.imag),fullFock_dimension=len(prepared),minimum_literal_C=float(np.min(C)))


def HS_case(context: dict,direct: dict,Q: int,order: int) -> dict:
    nodes,weights=roots_hermitenorm(order);weights/=math.sqrt(2*math.pi);background=np.array([Q,-Q]);zero=np.zeros(2)
    histories=[source.histories(context,x*direct['HSmap']) for x in nodes];prep=0j;raw=0j;final=np.zeros(3,np.complex128)
    for ia,iz in itertools.product(range(order),repeat=2):
        eta=nodes[ia]*direct['HSmap'];zeta=nodes[iz]*direct['HSmap'];left,leftsf,cl,csl=histories[ia];right,rightsf,cr,csr=histories[iz]
        coeff=weights[ia]*weights[iz]*np.exp(1j*((zeta-eta)@background))
        value=source.quadratic(context,left,right,cl,cr,zero,False)
        prep+=coeff*value[0];raw+=coeff*np.sum(value[1:])
        for ix in range(order):
            xi=nodes[ix]*direct['HSmap'];part=source.quadratic(context,leftsf,rightsf,csl,csr,xi,True)
            final+=coeff*weights[ix]*np.exp(1j*(xi@background))*part[1:]
    return dict(order=order,preparation_Gram=float(prep.real),raw_creator_Gram=float(raw.real),wholehalf_creator_Gram=float(np.sum(final).real),
        scalar_Gram=float(final[0].real),tangent_Gram=float(final[1].real),interference_Gram=float(final[2].real),wholehalf_imag=float(np.sum(final).imag))


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--capsule',required=True);args=parser.parse_args();start=time.time();rows=[]
    for nf in (1,2):
        context,direct=build(nf)
        for Q in (0,3):
            exact=direct_case(context,direct,Q);controls=[HS_case(context,direct,Q,order) for order in (8,12)]
            errors=[max(abs(row[key]-exact[key])/exact[key] for key in ('preparation_Gram','raw_creator_Gram','wholehalf_creator_Gram')) for row in controls]
            rows.append(dict(NF=nf,Q=Q,direct_positive_charge_diagonal_Fock=exact,coherent_Slater_threeHS_controls=controls,maximum_relative_errors=errors))
    execution=dict(source_sha256=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),coupled_source_sha256=hashlib.sha256(COUPLED_BYTES).hexdigest(),coupled_source_bytes_at_IMPORT=COUPLED_BYTES.decode(),
        prepared_source_bytes_at_IMPORT=source.PREPARED_BYTES.decode(),SF_source_bytes_at_IMPORT=source.SF_BYTES.decode(),frozen_dependencies=coul.loops.IMPORT_EPOCH,
        core_source_bytes_at_IMPORT=coul.loops.CORE_BYTES.decode(),loop_source_bytes_at_IMPORT=coul.loops.SOURCE_BYTES.decode(),
        argv=sys.argv,input_parameters=dict(shape=[2,1,1],NF=[1,2],Q=[0,3],GHorders=[8,12],k=.01,u=.02,Gaussian_width=3.5,q=.3,UA=2),
        python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,elapsed_seconds=time.time()-start,startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES)
    report=dict(rows=rows,scope='Native finite-occupation algebra control of the same C/Slater/RAWB/SF composition. Two-site geometry and NF1/2 are explicit; Q3 is retained charge for algebra checking, not a new alpha fit or macrostate. ALL70/4900 amplitudes, no dense4900² NFmatrix, no source/pole/sea gate.')
    Path(args.capsule).write_text(json.dumps(dict(report=report,execution=execution),indent=2)+'\n');print(json.dumps(dict(capsule=args.capsule,report=report,elapsed_seconds=execution['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':main()

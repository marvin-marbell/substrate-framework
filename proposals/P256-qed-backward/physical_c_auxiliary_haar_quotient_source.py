#!/usr/bin/env python3
"""Genuine centered U1 charge-Haar projector in original full prepared source.

P_Q=integral_Haar exp(i eta*[rho_aux+Qv]), rho_aux=sum_flavours(n_x-2).
No Gaussian Coulomb prior and no .02 action are present.  Gaussian/uniform
mixture is ONLY a full-support importance proposal for normalized compact
Haar.  Each original prefix and final source contains its actual P_Q.

The coexact field is explicitly a BOUNDED FINITE CHART NYSTROM regulator,
with normalized positive chart weights and a constant rank-one middle.
It is NOT constantR on the entire REALcover, NOT global allcoexact/toron
compactHaar, and NOT a Root931 or historical Ref4/0 source identity.
Original fixed meanbody and Gaussian3.5/raw creator are retained at the
same declared field nodes.  All scalar/tangent/interference and finalSF
remain; a sea compression is ONLY a neutral Rayleigh lower bound.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
from scipy.linalg import eigh
from scipy.special import ndtri
from scipy.stats import qmc
import physical_c_auxiliary_coulomb_prepared_source as source
coul=source.coul
SOURCE_BYTES=Path(__file__).read_bytes();COUPLED_BYTES=source.SOURCE_BYTES


def build_chart(order: int) -> dict:
    context=source.build(order);cov=context['cov'];dimension=len(cov)-1
    precision=cov[1:,1:];proposal=np.linalg.inv(precision);e,v=eigh(proposal)
    mapping=np.column_stack((np.zeros(dimension),(v*np.sqrt(e)).T))
    logjac=-dimension*math.log(2*math.pi)/2-np.linalg.slogdet(precision)[1]/2
    mu=context['weights']/np.sum(context['weights']);nodes=context['nodes'];width=context['width']
    gaussian=np.exp(-width*nodes**2/2);gaussian/=math.sqrt(float(mu@(gaussian**2)))
    context['heat']=np.sqrt(mu[:,None]*mu[None,:]);context['boundary']=np.sqrt(mu)*gaussian
    context['UA']=math.inf;context['cov']=np.zeros_like(cov)
    context['Haar_precision']=precision;context['Haar_map']=mapping;context['Haar_logjac']=float(logjac)
    context['mu']=mu;context['gaussian']=gaussian
    output=mu*np.exp(np.array([x['loglambda'] for x in context['layers']]));output/=np.sum(output)
    inputs=mu*gaussian*np.exp(np.array([x['loglambda']/2 for x in context['layers']]));inputs/=np.sum(inputs)
    if np.any(output<=0) or np.any(inputs<=0):raise ArithmeticError('Finite chart proposal lost support; no probability floor')
    context['output_probability']=output;context['input_probability']=inputs
    views=[]
    for i in range(order):
        row=[]
        for j in range(order):
            view=dict(context);view.update(order=1,layers=[context['layers'][i]],first=[context['first'][j]],nodes=np.array([nodes[i]]),heat=np.ones((1,1)),boundary=np.ones(1))
            row.append(view)
        views.append(row)
    context['views']=views
    return context


def Haar_draw(context: dict,u: np.ndarray,selector: float,mixture: float) -> tuple[np.ndarray,float,bool]:
    dimension=len(u)
    if selector<mixture:eta=np.r_[0.,2*math.pi*(u-.5)]
    else:
        if np.any((u<=0)|(u>=1)):raise ArithmeticError('Inverse-normal endpoint; no clipping')
        eta=ndtri(u)@context['Haar_map']
    inside=bool(np.all(abs(eta[1:])<=math.pi))
    if not inside:return eta,0.,False
    exponent=float(eta[1:]@context['Haar_precision']@eta[1:])/2
    logfactor=context['Haar_logjac']+exponent
    logweight=logfactor-np.logaddexp(math.log1p(-mixture),math.log(mixture)+logfactor)
    weight=math.exp(float(logweight))
    return eta,weight,True


def selected_history(context: dict,i: int,j: int,eta: np.ndarray) -> tuple[dict,dict,complex,complex]:
    before,after,coef,sfcoef=source.histories(context['views'][i][j],eta)
    return before[0],after[0],complex(coef[0]),complex(sfcoef[0])


def discrete(probability: np.ndarray,u: float) -> int:
    return int(np.searchsorted(np.cumsum(probability),u,side='right'))


def integrate(context: dict,bits: int,mixture: float) -> dict:
    dimension=len(context['Haar_precision']);uniforms=qmc.Sobol(3*dimension+7,scramble=True,rng=np.random.default_rng(222945)).random_base2(bits)
    zero=np.zeros(dimension+1);background=zero.copy();Q=int(source.sf.scaled_integer_charge('100')['Q']);background[0]=Q;background[1]=-Q
    totals={name:np.zeros(8,np.complex128) for name in ('neutral','chosen_integerQ')};absolute={name:np.zeros(2) for name in totals}
    compression=np.zeros((context['order'],context['order']),np.complex128);outside=0;uniform_draws=0;zero_amplitudes=0
    mu=context['mu'];po=context['output_probability'];pi=context['input_probability'];g=context['gaussian']
    for row in uniforms:
        eta,wa,ina=Haar_draw(context,row[:dimension],row[3*dimension],mixture)
        zeta,wz,inz=Haar_draw(context,row[dimension:2*dimension],row[3*dimension+1],mixture)
        xi,wx,inx=Haar_draw(context,row[2*dimension:3*dimension],row[3*dimension+2],mixture)
        uniform_draws+=int(row[3*dimension]<mixture)+int(row[3*dimension+1]<mixture)+int(row[3*dimension+2]<mixture)
        outside+=int(not ina)+int(not inz)+int(not inx)
        i=discrete(po,row[-4]);j=discrete(pi,row[-3]);l=discrete(po,row[-2]);k=discrete(pi,row[-1])
        if ina and inz:
            left,leftsf,cl,csl=selected_history(context,i,j,eta)
            rightprep,_,crprep,_=selected_history(context,i,k,zeta)
            right,rightsf,cr,csr=selected_history(context,l,k,zeta)
            zero_amplitudes+=int(cl==0)+int(cr==0)+int(crprep==0)+int(csl==0)+int(csr==0)
            raw=np.conj(cl)*crprep*np.array(source.pair_parts(context,left,rightprep,zero))
            raw*=wa*wz*mu[i]*mu[j]*mu[k]*g[j]*g[k]/(po[i]*pi[j]*pi[k])
            if inx:
                final=np.conj(csl)*csr*np.array(source.pair_parts(context,leftsf,rightsf,xi))
                final*=wa*wz*wx*mu[i]*mu[l]*mu[j]*mu[k]*g[j]*g[k]/(po[i]*po[l]*pi[j]*pi[k])
            else:final=np.zeros(4,np.complex128)
            for name,bg in (('neutral',zero),('chosen_integerQ',background)):
                prep_phase=np.exp(1j*((zeta-eta)@bg));source_phase=prep_phase*np.exp(1j*(xi@bg))
                totals[name][:4]+=raw*prep_phase;totals[name][4:]+=final*source_phase
                absolute[name][0]+=abs(raw[0]);absolute[name][1]+=abs(np.sum(final[1:]))
        if inx:
            for a in range(context['order']):
                for b in range(context['order']):
                    leftlayer=context['layers'][a];rightlayer=context['layers'][b]
                    char=np.exp(context['NF']*coul.graph_character_log(leftlayer['sea'],rightlayer['sea'],xi[None])[0])
                    compression[a,b]+=context['heat'][a,b]*np.exp((leftlayer['loglambda']+rightlayer['loglambda'])/2)*char*wx
    count=len(uniforms);compression/=count
    herm=float(np.linalg.norm(compression-compression.conj().T));lower=float(np.max(np.linalg.eigvalsh((compression+compression.conj().T)/2)))
    upper=math.exp(max(x['loglambda'] for x in context['layers']))
    if lower<=0 or lower>upper:raise ArithmeticError('Finite Haar-chart neutral Rayleigh/SF ordering failed')
    cases={}
    for name,total in totals.items():
        total/=count;raw=total[:4];final=total[4:];prep=float(raw[0].real);rawsum=float(np.sum(raw[1:]).real);source_sum=float(np.sum(final[1:]).real)
        if prep<=0 or rawsum<=0 or source_sum<=0:raise ArithmeticError('Literal Haar coupled source positivity failed; refine actual quadrature')
        cases[name]=dict(original_FIXEDbody_FULL_KH_preparation_Gram=prep,raw_creator_normalized_Gram=rawsum/prep,
            wholehalf_creator_normalized_numeric_interval=[source_sum/(prep*upper),source_sum/(prep*lower)],
            wholehalf_scalar_normalized_upper=float(final[1].real/(prep*lower)),wholehalf_tangent_normalized_upper=float(final[2].real/(prep*lower)),
            wholehalf_interference_normalized_upper=float(final[3].real/(prep*lower)),
            prep_signed_cancellation_price=absolute[name][0]/(count*abs(raw[0])),source_signed_cancellation_price=absolute[name][1]/(count*abs(np.sum(final[1:]))),
            unpaired_imaginary_source_Gram=float(np.sum(final[1:]).imag))
    return dict(bits=bits,samples=count,dimensions=3*dimension+7,mixture_proposal_only=mixture,uniform_Haar_proposal_draws=uniform_draws,
        outside_compact_target_domain_draws=outside,underflowed_SF_amplitude_count=zero_amplitudes,
        exact_Haar_importance_weight_upper=1/mixture,Gaussian_peak_Haar_measure_constant=math.exp(context['Haar_logjac']),
        neutral_normalizer_numeric_interval=[lower,upper],neutral_compression_Hermiticity_defect=herm,cases=cases,
        operator='SF PHaar_chart P_(rho_aux+Qv=0) SF, same projector inside each preparation history and final wholehalf quadratic form. NOT exp(-.02rhoLrho).',
        underflow_contract='Any exp underflow is recorded, not reinterpreted as a physical floor. Importance weights are bounded by1/mixture; no tiny barecharge or state gate is introduced.',
        numerical_contract='Exact identities sampled by disclosed QMC/finitefield chart. Adjoint orbit(eta,zeta,xi)->(zeta,eta,-xi), with field i/l and j/k swapped, prices reality without dropping a source channel. Not outward-rounded, globalcompact, or pole certification.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--capsule',required=True);parser.add_argument('--orders',default='3,5');parser.add_argument('--bits',default='8,10');parser.add_argument('--mixture',type=float,default=.02);args=parser.parse_args();start=time.time()
    if not 0<args.mixture<1:raise ValueError('Importance mixture must have full support')
    orders=[int(x) for x in args.orders.split(',')];bits=[int(x) for x in args.bits.split(',')];rows=[]
    for order in orders:
        context=build_chart(order);rows.append(dict(order=order,field_nodes=context['nodes'].tolist(),normalized_finite_chart_weights=context['mu'].tolist(),Gaussian_width=3.5,
            samples=[integrate(context,bit,args.mixture) for bit in bits]))
    execution=dict(source_sha256=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),coupled_source_sha256=hashlib.sha256(COUPLED_BYTES).hexdigest(),coupled_source_bytes_at_IMPORT=COUPLED_BYTES.decode(),
        prepared_source_bytes_at_IMPORT=source.PREPARED_BYTES.decode(),SF_source_bytes_at_IMPORT=source.SF_BYTES.decode(),continuum_source_bytes_at_IMPORT=source.prepared.CONTINUUM_BYTES.decode(),bundle_source_bytes_at_IMPORT=source.continuum.BUNDLE_BYTES.decode(),
        core_source_bytes_at_IMPORT=coul.loops.CORE_BYTES.decode(),loop_source_bytes_at_IMPORT=coul.loops.SOURCE_BYTES.decode(),frozen_dependencies=coul.loops.IMPORT_EPOCH,
        argv=sys.argv,input_parameters=dict(orders=orders,bits=bits,mixture=args.mixture,NF=625000000,k=.01,beta=100,Gaussian_width=3.5,seed=222945),
        python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,elapsed_seconds=time.time()-start,startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES)
    report=dict(rows=rows,centering='rho_aux=sum(n_x-2), external integerQ3 shift at0/1. Graph rotation cancels ONLY occupied2/site phase against exp(-2i NF sumeta), never externalQ. Alpha56 Refprimitive0/external−Q is NOT historicalRoot Ref4/0.',
        compact_gauge_identity='Globalneutral grade permits eta0=0 and normalized product Haar of eta1..etaV-1. Gaussian/uniform mixture on R^d has exact compact-target indicator and bounded target/proposal ratio, no unpriced Gaussian alias omission.',
        first_unmet_global_admission='Finite bounded coexact chart projector is NOT full allcompactcoexact+toron Haar. Full compact-mode integration or priced chart-to-global admission, complete SU4/Phi/alltorons, OriginalPsi/Ref3 charged range/Gram and physicalpole remain OPEN. No Root931/oldSource2a binding claim.')
    Path(args.capsule).write_text(json.dumps(dict(report=report,execution=execution),indent=2)+'\n');print(json.dumps(dict(capsule=args.capsule,report=report,elapsed_seconds=execution['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':main()

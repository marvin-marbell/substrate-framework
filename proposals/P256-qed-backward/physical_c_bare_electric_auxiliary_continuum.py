#!/usr/bin/env python3
"""Actual AUX sea kernel: material 3node -> full REAL-field quadrature control.

Positive Gauss-Hermite Nystrom regulators retain ALLcanonical auxiliary output.
The cheap exact-T4 evaluator is calibrated against captured full intrinsicC.
No Gauss/Coul/Phi/activechargedstate or macro photon pole is inferred.
"""
from __future__ import annotations
import argparse
import ast
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any,cast
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
import physical_c_bare_electric_auxiliary_bundle as bundle
loops=bundle.loops
SOURCE_BYTES=Path(__file__).read_bytes();BUNDLE_BYTES=Path(bundle.__file__).read_bytes()
if hashlib.sha256(BUNDLE_BYTES).hexdigest()!='4104b4e425276276b9e3a912f074c1a0a958b4a051ae31657f9d8e853724223b':raise ArithmeticError('Continuum binds executed auxiliary4104')


def precise_T4_state():
    tree=ast.parse(bundle.LOOP_BYTES)
    function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='hp_state')
    function.name='precise_single_T4_state'
    base=next(x for x in function.body if isinstance(x,ast.FunctionDef) and x.name=='base')
    base.body=[x for x in base.body if not (isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='h' for t in x.targets))]
    base.body[-1]=ast.parse("return dict(pref=scalar,eigenvalues=[mp.log(x) for x in e],fullframe=r,W=r[:,n:],loglambda=scalar+sum(mp.log(x) for x in e[n:]),gap=min(mp.log(e[n]),-mp.log(e[n-1])))").body[0]
    function.body=function.body[:function.body.index(base)+1]+ast.parse('return base(1)').body
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]));namespace=dict(vars(loops))
    exec(compile(module,'<captured-01245-exact-T4-single-eigensystem>','exec'),namespace)
    return namespace[function.name],ast.unparse(module)


T4_STATE,T4_AST=precise_T4_state()


def normalized_background():
    b=loops.background((4,3,1));norm=mp.sqrt(sum(mp.mpf(str(x))**2 for row in b['a'] for x in row))
    b['a']=np.array([[mp.mpf(str(x))/norm for x in row] for row in b['a']],dtype=object)
    b['normB']=mp.mpf(str(b['normB']))/norm**2;b['normE']=mp.mpf(1)
    return b


def float_matrix(matrix):
    return np.array([[complex(matrix[i,j]) for j in range(matrix.cols)] for i in range(matrix.rows)],complex)


def small_logdet(matrix,sign=1):
    """Trace-log of I+sign Z; avoids rounding tiny Z to I before determinant."""
    norm=float(np.linalg.norm(matrix,2))
    if norm>=.01:raise ArithmeticError('Small canonical logdet series left validated domain')
    power=matrix.copy();result=0j
    for degree in range(1,6):
        result+=(-1)**(degree+1)*sign**degree*np.trace(power)/degree
        power=power@matrix
    remainder=matrix.shape[0]*norm**6/(6*(1-norm))
    return result,remainder


def wholehalf_raw_creator(states,frames,positive,negative,nodes,boundary,width,background,scalar,overlaps,kernel,heat,weights,normalizers,naux):
    """Execute fullcanonical rawB HALF quadratic form, no auxiliary pole gate."""
    profile=np.array(background['a'],float);law=loops.make_law(.01,background)
    tangents=[];rawmetrics=[];weightedmetrics=[]
    for i,state in enumerate(states):
        jet=law.potential(float(nodes[i])*profile/10,background['phi'],background['links'],(profile/10,profile/10))
        energies=np.array([float(x) for x in state['eigenvalues']]);gaps=energies[None,24:]-energies[:24,None]
        omega=(negative[i].conj().T@jet['hx']@positive[i])/gaps
        aligned=positive[i].conj().T@frames[i]
        tangents.append(negative[i]@(omega*np.exp(-gaps/2))@aligned)
        rawmetrics.append(naux*float(np.sum(abs(omega)**2)))
        weightedmetrics.append(naux*float(np.sum(abs(omega)**2*np.exp(-gaps))))
    size=len(states);tangent_kernel=np.zeros((size,size),complex);cross_kernel=np.zeros_like(tangent_kernel)
    for i in range(size):
        for j in range(size):
            m=frames[i].conj().T@frames[j]
            left=np.linalg.solve(m,tangents[i].conj().T@frames[j])
            right=np.linalg.solve(m,frames[i].conj().T@tangents[j])
            both=np.linalg.solve(m,tangents[i].conj().T@tangents[j])
            tangent_kernel[i,j]=scalar[i,j]*overlaps[i,j]*(naux*np.trace(both-left@right)+naux**2*np.trace(left)*np.trace(right))
            cross_kernel[i,j]=scalar[i,j]*overlaps[i,j]*naux*np.trace(right)
    if np.linalg.norm(tangent_kernel-tangent_kernel.conj().T)>1e-10:raise ArithmeticError('Wholehalf rawB tangent form lost Hermiticity')
    sea=np.sqrt(2*width)*nodes*boundary;aux=-boundary/np.sqrt(2*width)
    sea_half=float(np.vdot(sea,kernel@sea).real);aux_half=float(np.vdot(aux,tangent_kernel@aux).real)
    cross_half=float(2*np.vdot(sea,cross_kernel@aux).real);whole_half=sea_half+aux_half+cross_half
    if aux_half<0 or whole_half<0:raise ArithmeticError('Wholehalf rawB source form lost positivity')
    rweighted=np.sqrt(weights[:,None]*weights[None,:])*heat;rnorm=float(np.max(np.linalg.eigvalsh(rweighted)))
    source_S_aux=float(np.sum(abs(aux)**2*np.exp(np.array([float(naux*(s['loglambda']-states[size//2]['loglambda'])) for s in states]))*weightedmetrics))
    if aux_half>rnorm*source_S_aux*(1+1e-7):raise ArithmeticError('Actual source violated positive SF-sandwich bound')
    interval=lambda value:[value/normalizers[1],value/normalizers[0]]
    return dict(actual_rawB_Gram=float(np.sum(abs(sea)**2)+np.sum(abs(aux)**2*rawmetrics)),
        actual_rawB_UNCOMPRESSED_auxiliary_Gram=float(np.sum(abs(aux)**2*rawmetrics)),
        actual_WHOLEhalf_sea_component_Gram_interval=interval(sea_half),
        actual_WHOLEhalf_UNCOMPRESSED_auxiliary_Gram_interval=interval(aux_half),
        actual_WHOLEhalf_sea_auxiliary_INTERFERENCE_unnormalized=cross_half,
        actual_WHOLEhalf_rawB_Gram_interval=interval(whole_half),
        actual_S_on_rawB_auxiliary_Gram=source_S_aux,
        actual_R_norm=rnorm,normalized_positive_SF_sandwich_auxiliary_Gram_upper=rnorm*source_S_aux/normalizers[0],
        source_definition='RAW fullfield a_refdag on NEW sea-line Gaussian I(A)phi_width. Sea part sqrt(2width)Aphi; orthogonal auxiliary part -Iprime phi/sqrt(2width). Exact Nflavour mixed Slater derivative and SFparticle-hole weighting; NO auxiliary or photon pole projection.',
        scope='Actual fullcanonical K^1/2 source quadraticform for declared positivefield quadrature, BEFOREGauss/Coul/active/Phi. Not the originalfixedbody prefix source: that body preparation is separately priced, not replaced by this newpacket. Numerical exact-identity evaluation, not interval floatingpoint certificate.')


def output_defect_Gram(states,positive,negative,inputs,projection,ratios,heat,weights,loglambda,naux):
    size=len(states);gram=np.zeros((size,size),complex);remainder=0.
    for i in range(size):
        plus=np.exp(np.array([float(x) for x in states[i]['eigenvalues'][24:]]));minus=np.exp(np.array([float(x) for x in states[i]['eigenvalues'][:24]]))
        aj=[positive[i].conj().T@frame for frame in inputs];cj=[negative[i].conj().T@frame for frame in inputs]
        for j in range(size):
            for l in range(size):
                z=np.linalg.solve(aj[l].conj().T@(plus[:,None]*aj[j]),cj[l].conj().T@(minus[:,None]*cj[j]))
                correction,rem=small_logdet(z);remainder=max(remainder,naux*rem)
                term=weights[i]*np.sqrt(weights[j]*weights[l])*heat[i,j]*heat[i,l]*np.exp(loglambda[i]+(loglambda[j]+loglambda[l])/2)
                gram[l,j]+=term*ratios[j]*ratios[l]*np.conj(projection[i,l])*projection[i,j]*np.expm1(naux*correction)
    if np.linalg.norm(gram-gram.conj().T)>1e-10:raise ArithmeticError('Uncompressed Slater Gram lost Hermiticity')
    gram=(gram+gram.conj().T)/2
    if np.min(np.linalg.eigvalsh(gram))<-1e-12:raise ArithmeticError('Uncompressed Slater Gram lost positivity')
    return gram,remainder


def run_order(order: int,cache: dict[str,Any],background: dict[str,Any]) -> dict[str,Any]:
    k='0.01';beta=mp.mpf(100);count=Fraction(100)/(16*Fraction(k)**4);naux=float(count.numerator)
    width=3.5;x,w=np.polynomial.hermite.hermgauss(order);nodes=x/np.sqrt(width);weights=w*np.exp(x*x)/np.sqrt(width)
    states=[]
    for node in nodes:
        key=cast(str,mp.nstr(mp.mpf(str(node))/mp.sqrt(beta),45))
        if key not in cache:cache[key]=T4_STATE(k,background,key)
        states.append(cache[key])
    origin=cache.setdefault('0.0',T4_STATE(k,background,'0')) if '0.0' not in cache else cache['0.0']
    precise_frames=[bundle.aligned_frame(origin['W'],state) for state in states]
    frames=[float_matrix(frame) for frame in precise_frames];negative=[float_matrix(s['fullframe'][:,:24]) for s in states]
    positive=[float_matrix(s['fullframe'][:,24:]) for s in states]
    loglambda=np.array([float(mp.mpf(count.numerator)*(s['loglambda']-origin['loglambda'])) for s in states])
    ua=2.;heat=np.exp(-(nodes[:,None]-nodes[None,:])**2/(4*ua))/np.sqrt(4*np.pi*ua)
    overlaps=np.zeros((order,order),complex);phase_max=0.;remainder_max=0.
    for i in range(order):
        for j in range(order):
            complement=negative[i].conj().T@frames[j];loss,rem=small_logdet(complement.conj().T@complement,-1)
            phase=float(np.angle(np.linalg.slogdet(frames[i].conj().T@frames[j])[0]));phase_max=max(phase_max,abs(naux*phase));remainder_max=max(remainder_max,naux*rem)
            overlaps[i,j]=np.exp(naux*(loss.real/2+1j*phase))
    scalar=np.sqrt(weights[:,None]*weights[None,:])*heat*np.exp((loglambda[:,None]+loglambda[None,:])/2)
    kernel=scalar*overlaps
    gram,correction_remainder=output_defect_Gram(states,positive,negative,frames,overlaps,np.ones(order),heat,weights,loglambda,naux)
    herm=float(np.linalg.norm(kernel-kernel.conj().T));gherm=float(np.linalg.norm(gram-gram.conj().T))
    if herm>1e-7 or gherm>1e-10:raise ArithmeticError('Actual quadrature lost Hermiticity beyond disclosed float envelope')
    kernel=(kernel+kernel.conj().T)/2;gram=(gram+gram.conj().T)/2
    if np.min(np.linalg.eigvalsh(gram))<-1e-12:raise ArithmeticError('Actual complementary Gram lost positivity')
    matrix=lambda a:mp.matrix([[mp.mpc(complex(z)) for z in row] for row in a])
    certificate: dict[str,Any]=bundle.spectral_A_certificate(matrix(kernel),matrix(gram),matrix(heat),list(map(mp.mpf,weights)),list(map(mp.mpf,loglambda)),states,list(map(mp.mpf,nodes)))
    eigen,vec=np.linalg.eigh(kernel);boundary=np.sqrt(weights)*np.exp(-width*nodes**2/2);boundary/=np.linalg.norm(boundary)
    fixed_frame=float_matrix(origin['W']);original_frames=[];original_ratios=[]
    for i,state in enumerate(states):
        fullframe=float_matrix(state['fullframe']);onebodySF=fullframe@np.diag(np.exp(np.array([float(x) for x in state['eigenvalues']])/2))@fullframe.conj().T
        image=onebodySF@fixed_frame;values,vectors=np.linalg.eigh(image.conj().T@image)
        if np.min(values)<=0:raise ArithmeticError('Original fixedbody SF image lost rank')
        original_frames.append(image@vectors@np.diag(1/np.sqrt(values))@vectors.conj().T)
        aj=positive[i].conj().T@fixed_frame;cj=negative[i].conj().T@fixed_frame
        plus=np.exp(np.array([float(x) for x in state['eigenvalues'][24:]]));minus=np.exp(np.array([float(x) for x in state['eigenvalues'][:24]]))
        z=np.linalg.solve(aj.conj().T@(plus[:,None]*aj),cj.conj().T@(minus[:,None]*cj))
        correction,_=small_logdet(z)
        original_ratios.append(abs(overlaps[i,order//2])*np.exp(naux*correction.real/2))
    original_projection=np.zeros((order,order),complex)
    for i in range(order):
        for j in range(order):
            complement=negative[i].conj().T@original_frames[j];loss,_=small_logdet(complement.conj().T@complement,-1)
            phase=np.angle(np.linalg.slogdet(frames[i].conj().T@original_frames[j])[0])
            original_projection[i,j]=np.exp(naux*(loss.real/2+1j*phase))
    original_kernel=scalar*original_projection*np.array(original_ratios)[None,:]
    original_gram,original_remainder=output_defect_Gram(states,positive,negative,original_frames,original_projection,original_ratios,heat,weights,loglambda,naux)
    original_projected=float(np.linalg.norm(original_kernel@boundary)**2)
    original_omitted=float(np.vdot(boundary,original_gram@boundary).real)
    if original_omitted<0:raise ArithmeticError('Original FIXEDbody fullK complementary norm negative')
    normalizers=list(map(float,certificate['complete_ALLoccupation_K_normalizer_interval']))
    creator=wholehalf_raw_creator(states,frames,positive,negative,nodes,boundary,width,background,scalar,overlaps,kernel,heat,weights,normalizers,naux)
    return dict(order=order,unit_mode_profile=True,real_field_nodes=nodes.tolist(),positive_Lebesgue_weights=weights.tolist(),Gaussian_boundary_width=width,
        whole_SF_scalar_valence_loglambda=loglambda.tolist(),kernel_eigenvalues=eigen.tolist(),actual_P_A_P_transition=abs(np.vdot(vec[:,-2],nodes*vec[:,-1])),
        spectral_A_fullcanonical_numeric_certificate=certificate,sea_boundary_projected_K_output_Gram=float(np.linalg.norm(kernel@boundary)**2),
        sea_boundary_UNCOMPRESSED_K_defect_Gram=float(np.vdot(boundary,gram@boundary).real),
        initial_FIXED_meanbody_projection_Gram=float(np.sum(abs(boundary)**2*abs(overlaps[:,order//2])**2)),
        original_FIXED_meanbody_projected_FULL_K_output_Gram=original_projected,
        original_FIXED_meanbody_UNCOMPRESSED_FULL_K_defect_Gram=original_omitted,
        original_FIXED_meanbody_FULL_K_relative_defect_norm=np.sqrt(original_omitted/original_projected),
        original_FIXED_meanbody_normalized_FULL_K_output_Gram_interval=[(original_projected+original_omitted)/normalizers[1]**2,(original_projected+original_omitted)/normalizers[0]**2],
        original_FIXED_meanbody_trace_log_complement_truncation_bound=original_remainder,
        actual_WHOLEhalf_RAW_creator=creator,
        tensor_overlap_phase_max=phase_max,kernel_Hermiticity_defect=herm,complement_Gram_Hermiticity_defect=gherm,
        trace_log_overlap_truncation_bound=remainder_max,trace_log_complement_truncation_bound=correction_remainder,
        precision_scope='SF scalar/eigenframes constructed45dps. Stable float trace-log/complement matrix consumer; quoted Schur quantities are numerical evaluations of exact inequalities, NOT interval-certified floatingpoint or quadrature-continuum enclosures.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();started=time.time()
    with mp.workdps(45):
        b=normalized_background();fast=T4_STATE('.01',b,mp.nstr(mp.mpf('.025'),45));full=bundle.HP_STATE('.01',b,mp.nstr(mp.mpf('.025'),45))['C']
        calibration=dict(scalar_valence_loglambda_error=mp.nstr(abs(fast['loglambda']-full['loglambda']),40),positive_projector_error=mp.nstr(mp.norm(fast['W']*fast['W'].H-full['W']*full['W'].H),40))
        if max(abs(fast['loglambda']-full['loglambda']),mp.norm(fast['W']*fast['W'].H-full['W']*full['W'].H))>mp.mpf('1e-30'):raise ArithmeticError('ActualC=T4 fast consumer falsified')
        cache={};cases=[run_order(order,cache,b) for order in (5,9,17)]
    inputs=json.dumps(dict(kappa='.01',beta=100,orders=[5,9,17],unit_L2_profile=True,Utheta='.02',Gaussian_width=3.5,dps=45),sort_keys=True,separators=(',',':'))
    result: dict[str,Any]=dict(cases=cases,exact_C_T4_identity_calibration=calibration,
        scope='Actual positive fullREALfield quadrature controls for pureAbelian meanPhi AUXsubsystem, beforeGauss/Coul/colour/Phi/activecarriers. No fieldcontinuum/globalREALcover/physicalmasslesspole promotion. Original3node capsule preserved; exactcanonical output retained through Slaterdeterminants.',
        execution=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),bundle_at_IMPORT=hashlib.sha256(BUNDLE_BYTES).hexdigest(),bundle_bytes_at_IMPORT=BUNDLE_BYTES.decode(),loop_at_IMPORT=hashlib.sha256(bundle.LOOP_BYTES).hexdigest(),exact_fast_AST=T4_AST,input_json=inputs,input_sha256=hashlib.sha256(inputs.encode()).hexdigest(),runtime_seconds=time.time()-started,executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__,startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES and Path(bundle.__file__).read_bytes()==BUNDLE_BYTES and Path(loops.__file__).read_bytes()==bundle.LOOP_BYTES))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=args.output,source=result['execution']['source_at_IMPORT'],calibration=calibration,summary=[dict(order=c['order'],eigenvalues=c['kernel_eigenvalues'][-2:],A=c['actual_P_A_P_transition']) for c in cases])))
if __name__=='__main__':main()

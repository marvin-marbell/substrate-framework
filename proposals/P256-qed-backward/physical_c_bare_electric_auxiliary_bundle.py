#!/usr/bin/env python3
"""Actual auxiliary-sea field-bundle compression AND uncompressed output price.

No microscopic auxiliary occupation is deleted. This DIAGNOSTIC embeds a new
field-dependent normalized sea packet and measures its omitted fullcanonical
Fock output via exact Slater determinants, without allocating Naux Fock arrays.
It is not an accepted replacement of fixed-boundary/fullGauss dynamics. Uses
actual intrinsicC scalar+valence SF, not an assumed standard Wilson action.
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
from typing import Any
if __name__=='__main__':
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import mpmath as mp
import physical_c_bare_electric_loops as loops
SOURCE_BYTES=Path(__file__).read_bytes();LOOP_BYTES=Path(loops.__file__).read_bytes()
if hashlib.sha256(LOOP_BYTES).hexdigest()!='01245b29b505d724ed8b301d07cb00c022e1f5a1a300cec3db0661c03c943e00':
    raise ArithmeticError('Aux bundle probe binds captured precise-loop01245')


def extended_precise_state():
    """AST extension exposes existing exact eigensystem locals; no equation edit."""
    tree=ast.parse(LOOP_BYTES)
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='hp_state')
    function.name='bundle_precise_state'
    matches=0
    for node in ast.walk(function):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
            if any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='out' for t in node.targets):
                node.value.keywords.extend([ast.keyword(arg='eigenvalues',value=ast.Name(id='e',ctx=ast.Load())),
                    ast.keyword(arg='fullframe',value=ast.Name(id='r',ctx=ast.Load()))]);matches+=1
    if matches!=1:raise ArithmeticError('Captured precise-state AST contract changed')
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    namespace=dict(vars(loops));exec(compile(module,'<captured-hp-state-eigensystem-extension>','exec'),namespace)
    return namespace['bundle_precise_state'],ast.unparse(module)


HP_STATE,EXTENDED_SOURCE=extended_precise_state()


def aligned_frame(reference,state):
    frame=state['W'];overlap=reference.H*frame
    values,vectors=mp.eighe(overlap.H*overlap)
    if min(values)<=0:raise ArithmeticError('Actual occupiedframe alignment lost invertibility')
    inverse_root=vectors*mp.diag([1/mp.sqrt(x) for x in values])*vectors.H
    return frame*inverse_root*overlap.H


def determinant_correction(state,framej,framek):
    """Exact canonical Γ(T) complement correction, not only valence eigenvalues.

    det(Vk† T_i Vj)/det(T_i,+) = det(Ak†)det(Aj)det(I+Z),
    Z=(Ak† T+ Aj)^-1 Ck† T- Cj. Every canonical defect appears in det(I+Z).
    """
    size=len(state['eigenvalues']);n=size//2
    vp=state['fullframe'][:,n:];vm=state['fullframe'][:,:n]
    aj=vp.H*framej;ak=vp.H*framek;cj=vm.H*framej;ck=vm.H*framek
    plus=mp.diag([mp.exp(x) for x in state['eigenvalues'][n:]])
    minus=mp.diag([mp.exp(x) for x in state['eigenvalues'][:n]])
    z=(ak.H*plus*aj)**-1*(ck.H*minus*cj)
    # Highprecision logdet(I+Z), with Z built from complementary projectors:
    # never subtract nearly equal full and projected scalar vacuum weights.
    correction=mp.log(mp.det(mp.eye(n)+z))
    return correction


def complement_gram(states,frames,inputs,ratios,heat,weights,loglambda,naux):
    """Exact Q K J Gram for any normalized Slater input frame J per fieldnode."""
    size=len(states);result=mp.zeros(size);largest=mp.mpf(0)
    for i in range(size):
        for j in range(size):
            for l in range(size):
                correction=naux*determinant_correction(states[i],inputs[j],inputs[l])
                largest=max(largest,abs(correction))
                sea=mp.exp(naux*(mp.log(mp.det(inputs[l].H*frames[i]))+mp.log(mp.det(frames[i].H*inputs[j]))))
                term=weights[i]*mp.sqrt(weights[j]*weights[l])*heat[i,j]*heat[i,l]
                term*=ratios[j]*ratios[l]*mp.exp(loglambda[i]+(loglambda[j]+loglambda[l])/2)
                result[l,j]+=term*sea*mp.expm1(correction)
    defect=mp.norm(result-result.H)
    if defect>mp.mpf('1e-25'):raise ArithmeticError('Exact complementary output Gram lost Hermiticity')
    eigen,_=mp.eighe((result+result.H)/2)
    if min(eigen)<-mp.mpf('1e-25'):raise ArithmeticError('Exact complementary output Gram lost positivity')
    return (result+result.H)/2,largest


def positive_polar_frame(matrix):
    values,vectors=mp.eighe(matrix.H*matrix)
    if min(values)<=0:raise ArithmeticError('Actual SF-prepared Slater frame lost rank')
    return matrix*vectors*mp.diag([1/mp.sqrt(x) for x in values])*vectors.H


def spectral_A_certificate(kernel,gram,heat,weights,loglambda,states,nodes):
    """Fullcanonical K lowband bounds; A commutes with pointwise sea projector."""
    eigen,vectors=mp.eighe((kernel+kernel.H)/2);qe,_=mp.eighe(gram)
    offdiag=mp.sqrt(max(qe))
    weighted_heat=mp.matrix([[mp.sqrt(weights[i]*weights[j])*heat[i,j] for j in range(len(nodes))] for i in range(len(nodes))])
    he,_=mp.eighe(weighted_heat);heatnorm=max(he)
    gap=min(s['gap'] for s in states);sfmax=max(mp.exp(x) for x in loglambda)
    canonical_Q_upper=heatnorm*sfmax*mp.exp(-2*gap)
    other_occupation_upper=heatnorm*sfmax*mp.exp(-gap)
    a0=eigen[len(eigen)-1];a1=eigen[len(eigen)-2]
    if not a0>other_occupation_upper:raise ArithmeticError('Alloccupation sectors could dominate the sea normalizer')
    if not a1>canonical_Q_upper:raise ArithmeticError('Actual canonical low-A band not separated from auxiliary bound')
    upper=lambda a:(a+canonical_Q_upper+mp.sqrt((a-canonical_Q_upper)**2+4*offdiag**2))/2
    state_errors=[]
    for index in (len(eigen)-1,len(eigen)-2):
        a=eigen[index];selfenergy=offdiag**2/(a-canonical_Q_upper)
        isolation=min(abs(a-eigen[j]) for j in range(len(eigen)) if j!=index)
        if selfenergy>=isolation:raise ArithmeticError('Actual low-A eigenvector bound unresolved')
        p_error=mp.sqrt(2)*selfenergy/(isolation-selfenergy)
        q_error=mp.sqrt(2)*offdiag/(a-canonical_Q_upper)
        state_errors.append(p_error+q_error)
    coordinate=mp.diag(nodes)
    matrixelement=abs((vectors[:,len(eigen)-2].H*coordinate*vectors[:,len(eigen)-1])[0])
    response_error=max(abs(x) for x in nodes)*sum(state_errors)
    return dict(fullcanonical_offdiagonal_K_intertwining_norm=mp.nstr(offdiag,60),
        canonical_Q_K_Q_norm_upper=mp.nstr(canonical_Q_upper,60),
        ALL_other_occupation_K_norm_upper=mp.nstr(other_occupation_upper,60),
        complete_ALLoccupation_K_normalizer_interval=[mp.nstr(a0,60),mp.nstr(upper(a0),60)],
        actual_P_A_P_transition_absolute=mp.nstr(matrixelement,60),
        complete_canonical_A_transition_error_upper=mp.nstr(response_error,60),
        complete_canonical_A_transition_absolute_lower=mp.nstr(max(0,matrixelement-response_error),60),
        complete_canonical_low_transition_TRANSFER_energy_interval=[mp.nstr(mp.log(a0/upper(a1)),60),mp.nstr(mp.log(upper(a0)/a1),60)],
        fullcanonical_A_lowband_nonzero_certified=bool(matrixelement>response_error),
        lowband_state_vector_error_uppers=[mp.nstr(x,60) for x in state_errors],
        derivation='Exact Gram is I†K Q K I, so b=||QKI||. Canonical SFdefects carry particle-hole gap>=2g; allother occupation sectors carry at leastg and do NOTcouple to I because particle number/species isconserved. QKQ<=||R²|| maxlambda exp(-2g); othersectors<=||R²||maxlambda exp(-g). Schur selfenergy<=b²/(a-c); positive minmax giveslambda_j in[a_j,(a_j+c+sqrt((a_j-c)²+4b²))/2]. Eigenvector rotation plusQleak prices actualA response, withQ A P=0 exactly.',
        scope='FULLcanonical Fock output and alloccupation normalizer bound for declared3field-node AUX subsystem, BEFOREGauss/Coul/colour/Phi/fullfield continuum. No actualfullcore/massless photon pole or activechargedsource inference.')


def run_case(ktext: str,beta_text: str,dps: int,cache: dict[tuple[str,str,int],Any]) -> dict[str,Any]:
    with mp.workdps(dps):
        beta=mp.mpf(beta_text);count=Fraction(beta_text)/(16*Fraction(ktext)**4)
        if count.denominator!=1 or count.numerator%2:raise ArithmeticError('Probe requires exact even integer Naux')
        naux=mp.mpf(count.numerator)
        b=loops.background((4,3,1));nodes=[mp.mpf('-.25'),mp.mpf('0'),mp.mpf('.25')]
        norm=mp.sqrt(sum(mp.mpf(str(x))**2 for row in b['a'] for x in row))
        b['a']=loops.np.array([[mp.mpf(str(x))/norm for x in row] for row in b['a']],dtype=object)
        b['normB']=mp.mpf(str(b['normB']))/norm**2;b['normE']=mp.mpf(1)
        weights=[mp.mpf('.25'),mp.mpf('.5'),mp.mpf('.25')]
        states=[]
        for q in nodes:
            key=(ktext,str(q/mp.sqrt(beta)),dps)
            if key not in cache:cache[key]=HP_STATE(ktext,b,str(q/mp.sqrt(beta)))['C']
            states.append(cache[key])
        origin=states[1]['W'];frames=[aligned_frame(origin,s) for s in states]
        loglambda=[naux*(s['loglambda']-states[1]['loglambda']) for s in states]
        utheta=mp.mpf('.02');ua=beta*utheta
        heat=mp.matrix([[mp.exp(-(x-y)**2/(4*ua))/mp.sqrt(4*mp.pi*ua) for y in nodes] for x in nodes])
        overlap=mp.matrix([[mp.log(mp.det(x.H*y)) for y in frames] for x in frames])
        kernel=mp.matrix([[mp.sqrt(weights[i]*weights[j])*heat[i,j]*mp.exp((loglambda[i]+loglambda[j])/2+naux*overlap[i,j]) for j in range(3)] for i in range(3)])
        hermitian_defect=mp.norm(kernel-kernel.H);eigen,_=mp.eighe((kernel+kernel.H)/2)
        if min(eigen)<-mp.mpf('1e-25'):raise ArithmeticError('Actual sea-bundle compressed kernel lost positivity')
        boundary=mp.matrix([mp.sqrt(w)*mp.exp(-q*q/2) for w,q in zip(weights,nodes,strict=True)])
        boundary/=mp.sqrt(sum(abs(x)**2 for x in boundary));projected=kernel*boundary
        projected_norm=sum(abs(x)**2 for x in projected)
        gram,max_log_correction=complement_gram(states,frames,frames,[mp.mpf(1)]*3,heat,weights,loglambda,naux)
        omitted=mp.re((boundary.H*gram*boundary)[0])
        spectral=spectral_A_certificate(kernel,gram,heat,weights,loglambda,states,nodes)
        original_projection=sum(abs(boundary[i])**2*mp.exp(2*naux*mp.re(overlap[i,1])) for i in range(3))
        original_frames=[];original_ratios=[]
        for i,state in enumerate(states):
            r=state['fullframe'];onebodySF=r*mp.diag([mp.exp(x/2) for x in state['eigenvalues']])*r.H
            original_frames.append(positive_polar_frame(onebodySF*origin))
            correction=naux*determinant_correction(state,origin,origin)
            original_ratios.append(mp.exp(naux*mp.re(overlap[i,1])+mp.re(correction)/2))
        original_kernel=mp.matrix([[mp.sqrt(weights[i]*weights[j])*heat[i,j]*mp.exp((loglambda[i]+loglambda[j])/2)*original_ratios[j]*mp.exp(naux*mp.log(mp.det(frames[i].H*original_frames[j]))) for j in range(3)] for i in range(3)])
        original_gram,_=complement_gram(states,frames,original_frames,original_ratios,heat,weights,loglambda,naux)
        original_projected=sum(abs(x)**2 for x in original_kernel*boundary)
        original_omitted=mp.re((boundary.H*original_gram*boundary)[0])
        ce=-naux*mp.re(overlap[0,2])/((nodes[2]-nodes[0])**2*mp.mpf(str(b['normE'])))
        cb=-(loglambda[0]+loglambda[2])/(2*nodes[2]**2*mp.mpf(str(b['normB'])))
        return dict(kappa=ktext,beta_theta=beta_text,Naux=str(int(naux)),precision=dps,
            declared_Utheta=str(utheta),actual_UA=mp.nstr(ua,50),field_nodes_A=[str(x) for x in nodes],positive_field_weights=[str(x) for x in weights],
            actual_scalar_AND_valence_loglambda_ratios=[mp.nstr(x,60) for x in loglambda],
            actual_compressed_kernel_real=[[mp.nstr(mp.re(kernel[i,j]),60) for j in range(3)] for i in range(3)],
            actual_compressed_kernel_imag=[[mp.nstr(mp.im(kernel[i,j]),60) for j in range(3)] for i in range(3)],
            actual_kernel_eigenvalues=[mp.nstr(x,60) for x in eigen],actual_hermitian_defect=mp.nstr(hermitian_defect,60),
            actual_sea_overlap_cE_A=mp.nstr(ce,60),actual_scalar_valence_cB_A=mp.nstr(cb,60),
            actual_bare_cE_A=mp.nstr(1/(4*ua),60),actual_total_cE_A=mp.nstr(ce+1/(4*ua),60),
            conditional_measured_cone_squared=mp.nstr(cb/(ce+1/(4*ua)),60),
            actual_projected_output_Gram=mp.nstr(projected_norm,60),actual_UNCOMPRESSED_canonical_defect_output_Gram=mp.nstr(omitted,60),
            actual_relative_omitted_output_norm=mp.nstr(mp.sqrt(omitted/projected_norm),60),
            actual_operator_spectral_A_certificate=spectral,
            original_FIXED_meanbody_boundary_initial_sea_projection_Gram=mp.nstr(original_projection,60),
            original_FIXED_meanbody_boundary_projection_loss=mp.nstr(1-original_projection,60),
            actual_K_on_original_FIXED_boundary_projected_output_Gram=mp.nstr(original_projected,60),
            actual_K_on_original_FIXED_boundary_UNCOMPRESSED_defect_output_Gram=mp.nstr(original_omitted,60),
            actual_K_on_original_FIXED_boundary_relative_defect_output_norm=mp.nstr(mp.sqrt(original_omitted/original_projected),60),
            actual_Naux_times_logdet_complement_correction_max=mp.nstr(max_log_correction,60),
            original_primitive_occupation0_deleted=False,
            scope='Actual3node REAL-field sampled AUX subsystem, normalizedunitL2 fixedtoron coexactprofile/12sites48Dirac components/Phi0/SU4I. No longitudinalCoul/jointGauss/colour/Phi or activecarrier included. BOTH originalFIXEDmeanbody packet and NEWsea-line packet are executed; canonical24particles/flavour ALLdefects retained. Allotherprimitive occupations bounded, not erased. This is not fullcore or a physicalphoton/sourcepole.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();started=time.time();cache={}
    cases=[run_case(k,b,55,cache) for k,b in [('0.02','100'),('0.01','100'),('0.01','1000')]]
    inputs=json.dumps(dict(cases=[['.02','100'],['.01','100'],['.01','1000']],dps=55,field_nodes=['-.25','0','.25'],weights=['.25','.5','.25'],declared_Utheta='.02',mode_link_L2_norm=1,original_fixedbody_boundary_retained=True,captured_loop=hashlib.sha256(LOOP_BYTES).hexdigest()),sort_keys=True,separators=(',',':'))
    result: dict[str,Any]=dict(cases=cases,derivation='I(q)=sea(q)^N; I†K I=sqrtlambda(q) R_A²(q,qprime) sqrtlambda(qprime) det[Vq†Vqprime]^N. Complementary-projector determinant exactly prices operatorQ K I and SAMEoriginalfixedboundary output on positive real-field sampling without hugeNF allocation. SFscalar included. No microscopic-law/source cutover.',
        limits=['Finitek coefficients and short4x3x1 profile do not certify macro remainder/geometry.','Gauss, fullcolour/Phi, original boundary transformation, alloccupation normalizer, actual activechargedsource/window/pole remain open.','The mean-sea projector is NOT imposed on microscopic law; output norm outside it is actually measured.'],
        predecessor_coordinate_falsifier='Historical39c0/45dps unnormalizedprofile hadnormE1.2 whileheat usedUA directly: not a canonical-coordinate cone claim. Preserved unchanged. This successor normalizesprofileexactly to1 and raisesprecision55; bareUtheta.02 mapsUA=beta*.02 onthatcanonicalmode.',
        execution=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),captured_loop_sha256=hashlib.sha256(LOOP_BYTES).hexdigest(),captured_loop_bytes_at_IMPORT=LOOP_BYTES.decode(),
            exact_AST_extended_function=EXTENDED_SOURCE,input_json=inputs,input_sha256=hashlib.sha256(inputs.encode()).hexdigest(),runtime_seconds=time.time()-started,executable=sys.executable,python=sys.version,mpmath=mp.__version__,
            startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES and Path(loops.__file__).read_bytes()==LOOP_BYTES))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=args.output,source=result['execution']['source_at_IMPORT'],cases=cases)))
if __name__=='__main__':main()

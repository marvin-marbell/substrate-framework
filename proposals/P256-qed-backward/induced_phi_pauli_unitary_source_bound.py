"""Native primitive real-time source kernel domain/remainder price.

NO heat analytic continuation and NO fabricated unitary numerical propagator.
Price actual H=gPi²+all64CAR+g rho Lplus rho-commonNEUTRALfreeCAR on
unchanged E SOURCE; use executed initial Ward moment for a statistical first
term plus deterministic exact-unitary Taylor remainder. FrozenPhi scope ONLY.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_pauli_gaussian_ward as ward
import induced_phi_pauli_electric_coulomb_selection as selection
import induced_phi_static_backreaction_bound as field_price
pauli=ward.pauli
quantum=ward.quantum
loops=field_price.loops
SOURCE_BYTES={**ward.SOURCE_BYTES,**selection.SOURCE_BYTES,**field_price.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
WARD_SHA='9953ff62d8b83d1adeecb00cffb873b4e6f91508fa578c155343b985346540fd'
SELECTION_SHA='918347f1c764fa66c5088fdae20c26a6c6f80bdb05f34e3a866af3d993265298'


def rotor(native:dict[str,Any],profile:dict[str,Any],theta:np.ndarray)->np.ndarray:
    state=quantum.graph_state(native,profile,theta,section=False);e,U=np.linalg.eigh(state['R']);S=(U*np.sqrt(e))@U.conj().T;K=state['Y']@S
    return np.block([[S,-1j*K],[-1j*K,S]])


def source_graph(mean:dict[str,Any],bits:int)->dict[str,Any]:
    native=quantum.build_native(mean,4);V,M=native['V'],native['M'];iv=loops.iv
    a=loops.exact(F(1,64));g=loops.exact(F(64));s=loops.exact(F(2));p=loops.exact(F(1,2));r0=iv.sqrt(p)
    g0=field_price.interval(mean,'g0');gs=field_price.interval(mean,'gs');fmax=iv.sqrt(loops.exact(F(2,V)));sigma=1/iv.sqrt(8)
    Nneutral=64*2*V;hCAR=iv.mpf(0);F0=iv.mpf(0);F1=iv.mpf(0);F2=iv.mpf(0);profiles=[]
    tcap=loops.exact(F(1,3));z1=(1+tcap)/(1-tcap*tcap)**loops.exact(F(3,2))*tcap
    z2=(1+3*tcap+2*tcap*tcap)/(1-tcap*tcap)**loops.exact(F(5,2))*tcap*tcap+z1
    for index,profile in enumerate(native['Pprofiles']):
        c=pauli.scalars(profile);on=loops.exact(F.from_float(abs(float(c['onsite']))));HC=on*iv.sqrt(g0);capF=on;capF1=iv.mpf(0);capF2=iv.mpf(0);Kcap=iv.mpf(0)
        for length,coefficients in c['path'].items():
            c0=loops.exact(F.from_float(abs(float(coefficients[0]))));ci=loops.exact(F.from_float(abs(float(coefficients[1]))))
            if length==2 and float(ci.b)!=0:raise ArithmeticError('OriginalHermY nearestPhi5 preparation law changed')
            HC+=6*(c0*iv.sqrt(g0)+ci*iv.sqrt(gs));capF+=6*c0;capF1+=6*length*c0;capF2+=6*length*length*c0;Kcap+=6*ci*iv.sqrt(gs)
        actual_t=Kcap/(2*on*iv.sqrt(g0)+g)
        if float(actual_t.b)>=float(tcap.a):raise ArithmeticError('ActualoriginalSOURCE Y radius not below declared1/3 analyticbound')
        count=profile['load']*2*V+(1 if index==0 else 0)
        hCAR+=loops.exact(2*profile['load']*2*V+(1 if index==0 else 0))*HC
        F0+=loops.exact(count)*capF*fmax*sigma;F1+=loops.exact(count)*capF1*fmax*sigma;F2+=loops.exact(count)*capF2*fmax*sigma
        profiles.append(dict(load=profile['load'],actual_unchanged_SOURCE_Y_radius_upper=loops.bounds(actual_t),actual_onebody_fullCAR_norm_upper=loops.bounds(HC)))
    if sum(profile['load'] for profile in native['Pprofiles'])!=64 or native['Pprofiles'][0]['load']!=1:raise ArithmeticError('Actualall1+63charged fixed-N source counts changed')
    # E=.5sum ALL4 cdag r, ||E||<=2. It has NO coexact/harmonic EM
    # derivative: realGauss ell is longitudinal, Q^T ell=0; colourI frozen.
    r1=2*Nneutral*z1;r2=2*(Nneutral*z2+Nneutral*(Nneutral-1)*z1*z1)
    lap=loops.exact(F(2));rho=loops.exact(F(129))
    # mobile n_x in[0,256], bg128, Ref3-bgRef4=-delta0.
    HCoul=g*V*rho*rho/lap;Hbounded=hCAR+HCoul
    bF0=F0*r0;bF1=F1*r0+F0*r1;bF2=F2*r0+2*F1*r1+F0*r2
    kineticF=iv.sqrt(M*(M+2))/(4*s)*bF0+iv.sqrt(M/s)*bF1+M*bF2
    normA=iv.sqrt(s)/a*r0
    kineticA=iv.sqrt(M*M-2*M+8)/(4*iv.sqrt(s)*a)*r0+iv.sqrt(M+2)/a*r1+iv.sqrt(s)/a*M*r2+r0/(iv.sqrt(s)*a)+2*r1/a
    graphF=Hbounded*bF0+g*kineticF;graphA=Hbounded*normA+g*kineticA
    timevalue=loops.exact(F(2)**(-bits));remainder=timevalue*timevalue/2*graphF*graphA
    rng=np.random.default_rng(20261026);theta=native['Q']@(np.sqrt(2)*rng.normal(size=M));direction=rng.normal(size=M);direction/=np.linalg.norm(direction);direction=native['Q']@direction;eps=1e-4;controls=[]
    for profile in native['Pprofiles']:
        Z=rotor(native,profile,theta);Zp=rotor(native,profile,theta+eps*direction);Zm=rotor(native,profile,theta-eps*direction)
        derivative=float(np.linalg.norm((Zp-Zm)/(2*eps),2));second=float(np.linalg.norm((Zp-2*Z+Zm)/(eps*eps),2));unitary=float(np.linalg.norm(Z.conj().T@Z-np.eye(4*V),2))
        if derivative>float(z1.a) or second>float(z2.a) or unitary>1e-9:raise ArithmeticError('Actual SOURCEcanonicalrotor derivative/domain control failed')
        controls.append(dict(role=profile['label'],actual_canonical_rotor_unitarity_error=unitary,actual_canonical_rotor_direction_derivative_norm=derivative,actual_canonical_rotor_second_direction_derivative_norm=second,computational_difference_epsilon=eps))
    return dict(N=4,V=V,M=M,ALLcharged_flavours=64,ALL4Refholes=True,source_probability_before_evolution='1/2',real_time='2^-'+str(bits),primitive_PHYSICAL_H='gPi²+sum_ALL64 dGamma(h(Theta))+g rhoLplusrho-commonNEUTRALfreeCAR; Ref3/bg128+4delta0, allrealQmodes+3harmonics, g64,a1/64. Phi/heavy/colour frozen: NOTfullH.',actual_profile_radius_and_CAR_prices=profiles,actual_uniform_SOURCE_rotor_first_derivative_norm_upper=loops.bounds(z1),actual_uniform_SOURCE_rotor_second_derivative_norm_upper=loops.bounds(z2),actual_raw_Phi_SOURCE_H_graph_norm_upper=loops.bounds(graphF),actual_raw_Aphys_SOURCE_H_graph_norm_upper=loops.bounds(graphA),actual_exact_unitary_kernel_Taylor_remainder_upper=loops.bounds(remainder),actual_SOURCE_canonical_rotor_controls=controls,source_graph_derivation='Wcanonical=[-iYsqrtR;sqrtR], R=(I+Y²)^-1. Z=[sqrtR,-iYsqrtR;-iYsqrtR,sqrtR] isUNITARY; occupied positiveonsite section differs ONLYconstantunitary, not Berryretag. Actual||Y||<1/3, ||DY||,||D²Y||<=1/3 alongANYunitQdirection. Positiveabsolute powercoefficients price sqrtR andYsqrtR first/secondderivatives z1,z2. Fixedneutral8192fermions exteriorproduct gives ||OmegaD||<=N z1, ||OmegaDD||<=N z2+N(N-1)z1². Creatorop<=2; exactnormsqrtp. Gaussian densityvariance2 gives ||DeltaG||=sqrtM(M+2)/(4s), ||(dy.q)DeltaG||=sqrt(M²-2M+8)/(4sqrt(s)); Gaussianpolynomial productrule and actualF/F1/F2 pricebothHgraphnorms. Smoothboundedray timesGaussian/polynomial isD(H), because CAR andCoulbounded onfixedN; no late-Slater assumption.',unitary_derivation='ForACTUALselfadjointH andu,v inD(H), |<u,(exp(-itH)-I+itH)v>|<=t²||Hu||||Hv||/2 byspectralcalculus. SOURCEcrossGram exactly0 by06dbpreparationidentity. NativePi+Coulmoment exactly0 by918 actualgroupselection; allCAR/Wilson1 firstmoment from9953. This is an EXACTunitary remainder bound plus STATISTICALLYestimatedfirstmoment, NOT a computednumericalunitaryexponential or deterministicnonzerointegralcertificate.')


def run(ward_path:Path,selection_path:Path,bits:int)->dict[str,Any]:
    started=time.monotonic();raw=ward_path.read_bytes();selection_raw=selection_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=WARD_SHA or hashlib.sha256(selection_raw).hexdigest()!=SELECTION_SHA:raise ArithmeticError('Frozen9953/918 moment/primitivePiCoulselection epoch required')
    if bits<12:raise ValueError('Additionalunitaryprobe time muststaywithin originalnativebeta1/4096; original4096clock unchanged')
    cap=json.loads(raw);selectioncap=json.loads(selection_raw);_,record=quantum.checked(cap['input_descriptor']['input_receipts']['moment']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['beta']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['metric']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['magnetic']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['prior_joined_heat'])
    for key in ('full_source','contact','transfer','gram','force','join','heat'):_,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
    mean=pauli.contact.transfer.packets.force.joined.mean_from_source(record);physical=source_graph(mean,bits);moment=cap['actual_same_source_Pauli_Gaussian_Ward_moment']['actual_SAME_SOURCE_Gaussian_Ward_magnetic_CAR_moment']
    t=loops.exact(F(2)**(-bits));estimate=dict(real=loops.bounds(t*loops.exact(F.from_float(moment['imag']))),imag=loops.bounds(-t*loops.exact(F.from_float(moment['real']))),sampling_SE_real=loops.bounds(t*loops.exact(F.from_float(moment['sampling_SE_imag']))),sampling_SE_imag=loops.bounds(t*loops.exact(F.from_float(moment['sampling_SE_real']))),deterministic_gaussian_integral_certificate=False,definition='-i t timesobserved9953initialfullHmoment. Actualrealunitarykernel lieswithindeterministicallypricedTaylor remainder IF exactmoment supplied; displayedMCfirstterm NOTdeterministicnonzero certificate.')
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(ward=dict(path=str(ward_path),sha256=hashlib.sha256(raw).hexdigest()),selection=dict(path=str(selection_path),sha256=hashlib.sha256(selection_raw).hexdigest())),source_abi=cap['input_descriptor']['source_abi'],native_N=4,additional_real_time='2^-'+str(bits),original_clock_UNCHANGED='4096',source='UNCHANGEDoriginalcanonicalY/Gaussian2/E/all64/ALL4Refholes; no projector/filter/fittedspeed or source-mean replacement.',selection_producer_input_sha256=selectioncap['input_sha256'])
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_primitive_unitary_SOURCE_graph_remainder=physical,observed_first_order_real_unitary_source_kernel_prediction=estimate,scope='NativefrozenPhi quantumdomain/remainder plusobservedinitialmoment ONLY. NO heat continuation/fakeU simulation; no macro/fullPhi4096propagator/vacuum/pole/commoncone/QEDclaim.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ward',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--time-bits',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.ward,a.selection,a.time_bits)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actualunitarySOURCE pricing')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_primitive_unitary_SOURCE_graph_remainder'],first_term=r['observed_first_order_real_unitary_source_kernel_prediction'],elapsed_seconds=r['elapsed_seconds']),indent=2))

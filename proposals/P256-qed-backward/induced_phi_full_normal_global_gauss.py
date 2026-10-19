"""NEW full45 normal + global SU4 orbit kinetic reconstruction and CAR prices.

Exact compact tubular Phi chart, not scalar replication or a full local/EM law.
Actual finite-CAR source consumer must supply its OWN packet/connection bounds.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from math import isqrt
from pathlib import Path
import sys
import time
from typing import Any
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={Path(__file__).name:Path(__file__).read_bytes()}
sys.set_int_max_str_digits(0)
import induced_phi_uniform_global_gauss as scalar
SOURCE_BYTES.update(scalar.SOURCE_BYTES)
import numpy as np
FULL_RECEIPT_SHA='ebd076228091607848e98f859cb8c6621548363f8f3a9c261925725ac0749267'
SCALAR_SOURCE_SHA='d0e8707178a99bd742a9f56c1de50c4d4f69f5a35abe3a0f711c7e1a05a4455c'


def sqrt_upper(x:F,bits:int=256)->F:
    if x<0:raise ValueError('Positive actual price required')
    if not x:return F(0)
    exponent=(x.numerator.bit_length()-x.denominator.bit_length())//2
    shift=max(0,bits-exponent)
    integer=(x.numerator<<(2*shift))//x.denominator
    return F(isqrt(integer)+1,1<<shift)


def packet_weighted_derivatives(d:int,cmin:F,cmax:F,R:F,p:int,C1:F,C2:F)->dict[str,F]:
    """Gaussian times exact compact polynomial bump; all moments/tails priced."""
    if not d>0 or not 0<cmin<=cmax or R<=0 or p<6 or C1<0 or C2<0:
        raise ValueError('Actual covariance, smooth support and CAR derivative bounds required')
    Z=1-2*p*d*cmax/R**2
    if Z<=F(1,2):raise ArithmeticError('Actual bump normalization lower insufficient; no clipping')
    moments={k:F(1) for k in (2,4,6,8)}
    for k in moments:
        for j in range(k//2):moments[k]*=(d+2*j)*cmax
    normfactor=sqrt_upper(1/Z);a=1/(2*cmin);b=2*p/R**2
    grad=(a+b)*sqrt_upper(moments[4])
    r2grad=(a+b)*sqrt_upper(moments[6])
    hess=(a*a+4*p*a/R**2+4*p*(p-1)/R**4)*sqrt_upper(moments[8])+sqrt_upper(F(d))*(a+b)*sqrt_upper(moments[4])
    B1=normfactor*(grad+C1*sqrt_upper(moments[2]))
    B2=normfactor*(hess+2*C1*r2grad+(C2+C1*C1)*sqrt_upper(moments[4]))
    return dict(normalizer_lower=Z,weighted_r_gradient_norm_upper=B1,weighted_r2_Hessian_norm_upper=B2,ALL_phi_r2_moment_upper=moments[2],ALL_phi_r4_moment_upper=moments[4])


def full_normal_gauss_prices(g0_lower:F,gs_lower:F,g0_upper:F,gs_upper:F,V:int,n:int,R:F,B1:F,B2:F)->dict[str,F]:
    """Actual full kinetic difference action bound /chi on one reference leg."""
    if not 0<g0_lower<=g0_upper or not 0<gs_lower<=gs_upper or V<=0 or n<=0 or n%4 or R<=0 or min(B1,B2)<0:
        raise ValueError('Actual positive centre-invariant CAR/chart inputs required')
    d=45;L=V*min(g0_lower,gs_lower)/2;W=V*(g0_upper+3*gs_upper)
    reach=min(L,F(1))**2/(100*(1+W))
    if R>reach or R**2>L/4 or R**2>F(1,8):raise ArithmeticError('Actual compact tube radius not certified')
    hL=L/4;sL=F(1,16);b=F(1,4);h1=1+W/2+R/2
    s1=2*b*b/hL+b*b*R*h1/hL**2
    k0=b/(sL*hL);k1=s1*b*R**2/(sL**2*hL)+(b/hL+b*R*h1/hL**2)/sL
    A0=8*b*b/(sL*hL);A1=s1/sL**2
    m0=min(F(1,8),L);jg=60/m0;jh=60/m0
    geom=(d*A1*R*jg+d*jh/sL)/4+jg*jg/(8*sL)
    radial=(sqrt_upper(F(d))*A0*B2+d*A1*B1)/2
    qnorm=2*n;cf=F(15,8)
    cross=qnorm*(2*k0*B1+d*k1)/2
    angular=n*n*cf*(1/hL+b*b*R**2/(sL*hL**2))/2
    return dict(certified_compact_tube_radius_upper=reach,orbit_mean_metric_lower=L,orbit_chart_metric_lower=hL,normal_Schur_metric_lower=sL,normal_cometric_difference_quadratic_coefficient=A0,normal_cometric_derivative_linear_coefficient=A1,orbital_cross_cometric_linear_coefficient=k0,orbital_cross_cometric_derivative_coefficient=k1,volume_log_gradient_norm_upper=jg,volume_log_Hessian_norm_upper=jh,geometric_potential_operator_norm_divided_chi_upper=geom,normal_kinetic_difference_action_divided_chi_upper=radial,Noether_cross_action_divided_chi_upper=cross,Noether_angular_operator_norm_divided_chi_upper=angular,total_reference_packet_Noether_and_metric_residual_divided_chi_upper=geom+radial+cross+angular)


def literal_chart(native:dict[str,Any],Q:np.ndarray,R:F)->dict[str,Any]:
    uniform=scalar.uniform;primitive=uniform.supplier.static.primitive;T=np.asarray(uniform.supplier.static.force.T);B=primitive.clifford_cartesian_basis()
    S=np.kron(np.eye(4),np.real(np.einsum('aij,bji->ab',B,T))/4);N=S@Q;V=native['V']
    generators=B/np.sqrt(8);delta=np.einsum('amk,mij->kaij',Q.reshape(4,15,45),T)
    Di=[]
    for k in range(45):
        derivative=1j*(generators[:,None]@delta[k][None]-delta[k][None]@generators[:,None])
        cart=np.real(2*np.einsum('mij,anji->anm',T,derivative)).reshape(15,60).T
        Di.append(np.concatenate((np.zeros((60,45)),S@cart),axis=1))
    directions=[np.zeros(45),np.eye(45)[0],np.sin(np.arange(45)+1),np.cos(2*np.arange(45)+1)]
    rows=[];maximum_orth=0.
    for k,direction in enumerate(directions):
        eta=np.zeros(45) if k==0 else float(R)*direction/np.linalg.norm(direction)
        frame=native['frame']+np.einsum('k,kaij->aij',eta,delta)/np.sqrt(V)
        comm=1j*(generators[:,None]@frame[None]-frame[None]@generators[:,None])
        cart=np.real(2*np.einsum('mij,anji->anm',T,comm)).reshape(15,60).T
        O=np.sqrt(V)*S@cart;D=np.concatenate((N,O),axis=1);metric=D.T@D;inverse=np.linalg.solve(metric,np.eye(60));Dinv=np.linalg.solve(D,np.eye(60))
        Pi=np.stack([Dinv@x for x in Di]);jp=np.trace(Pi,axis1=1,axis2=2);jpp=-np.einsum('aij,bji->ab',Pi,Pi)
        derivative_inverse=np.stack([-inverse@(x.T@D+D.T@x)@inverse for x in Di]);div=np.array([sum(derivative_inverse[i,i,j] for i in range(45)) for j in range(45)])
        A=inverse[:45,:45];geom=.25*(div@jp+np.trace(A@jpp))+.125*(jp@A@jp)
        orth=float(np.max(abs(N.T@N-np.eye(45)/8)));maximum_orth=max(maximum_orth,orth)
        rows.append(dict(eta_norm=float(np.linalg.norm(eta)),normal_primitive_metric_error=orth,normal_orbit_cross_norm=float(np.linalg.norm(metric[:45,45:],2)),full60_metric_min=float(np.linalg.eigvalsh(metric)[0]),ALL45_normal_cometric_difference_norm=float(np.linalg.norm(A-8*np.eye(45),2)),geometric_potential_divided_chi=float(geom),volume_log_gradient_norm=float(np.linalg.norm(jp)),volume_log_Hessian_norm=float(np.linalg.norm(jpp,2))))
    if maximum_orth>1e-12 or rows[0]['normal_orbit_cross_norm']>1e-12:raise ArithmeticError('Actual normal/orbit primitive metric mismatch')
    return dict(all60_coordinates=60,normal45=45,global_orbit15=15,actual_observations=rows,normal_basis_interpretation='JSON approximate diagnostic of exact normal subspace; universal operator bounds apply to any EXACT orthonormal Cartnormal basis, not rounded wave/cometric coefficients')


def run(full_path:Path,uniform_path:Path)->dict[str,Any]:
    started=time.monotonic();raw=full_path.read_bytes();uraw=uniform_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=FULL_RECEIPT_SHA or hashlib.sha256(uraw).hexdigest()!=scalar.UNIFORM_RECEIPT_SHA or hashlib.sha256(SOURCE_BYTES[Path(scalar.__file__).name]).hexdigest()!=SCALAR_SOURCE_SHA:raise ArithmeticError('Frozen actual full45/0c3/d0e prerequisites required')
    full=json.loads(raw);cap=json.loads(uraw);point=full['input_descriptor']['actual_point'];g0=F(point['g0']);gs=F(point['gs']);mean=cap['input_descriptor']['actual_root_mean_input_UNCHANGED'];native=scalar.uniform.supplier.quantum.build_native(mean,4);Q=np.asarray(full['actual_physical60Cart_to45normal_embedding']['physical_Cart_to_normal'],float)
    variance=cap['actual_uniform_coupled_quantum_backreaction_and_realtime_transfer']['actual_coupled_ground_Phi_variance'];sigma2=(F(variance['lower'])+F(variance['upper']))/2
    R=F(1,8192);n=2*native['V']*(2+mean['input_descriptor']['microscopic_config']['auxiliary_count'])+4
    # This executed PREPARATION is a genuine isotropic compact Gaussian times a
    # FIXED finite CAR state; not the full45 Born state or zero connection fallback.
    packet=packet_weighted_derivatives(45,sigma2,sigma2,R,6,F(0),F(0));prices=full_normal_gauss_prices(g0,gs,g0,gs,native['V'],n,R,packet['weighted_r_gradient_norm_upper'],packet['weighted_r2_Hessian_norm_upper'])
    chart=literal_chart(native,Q,R);source=scalar.literal_source();chi=F(2)**-4096
    times=[dict(time=str(t),additional_raw_source_Noether_metric_error_upper=scalar.ratio(8*abs(t)*chi*prices['total_reference_packet_Noether_and_metric_residual_divided_chi_upper'])) for t in (F(0),F(1,4096),F(4096))]
    desc=dict(plan='issue-222-induced-photon-delta13-continuation-20261008',assigned_scientific_revision=scalar.uniform.ASSIGNED_REVISION,observed_plan_revision='1c4f98f4e0e84bd598b150f73bb276ac882acd511d6af2de5848b4d15a80417a',input_full45=dict(path=str(full_path),sha256=hashlib.sha256(raw).hexdigest()),input_uniform=dict(path=str(uniform_path),sha256=hashlib.sha256(uraw).hexdigest()),g0=str(g0),gs=str(gs),V=native['V'],bulk_species=65,static_compensating_Ref4=4,finite_CAR_particles=n,compact_radius=str(R),packet_bump_power=6,actual_trial_isotropic_variance=str(sigma2),actual_trial_CAR_normal_field_dependence='FIXED state; C1=C2=0 by declared preparation, not a missing-field/default substitute',scope='NEW full45normal+15globalSU4 actual tubular geometry and Noether kinetic prices. Inputpoint is old0c3 NONstationary full45: no newmean/ground/fullsource/ThetaPi/localSU4/all90 admission.')
    physical=dict(actual_full60_chart=chart,actual_compensated_source_CAR=source,actual_packet_bounds={k:scalar.ratio(v) for k,v in packet.items()},actual_exact_rational_Noether_metric_prices={k:scalar.ratio(v) for k,v in prices.items()},additional_source_clock_budgets=times,
        chart_injectivity_proof='Mean Phi=sqrt(g_a)Gamma_a. ForSU4 U: orbit chord norm_B²=(V/4)sum||[U,Phi_a]||F²>=V min(g_a)||U-TrU I/4||F². If two normal-radiusR points are related, tracelessU norm<=R/(2sqrtL). ForR<=reach this is<.01; unitarity+detU1 imply U nearonecentre, tracelesslogq norm<=30 tracelessU norm<=15R/sqrtL. Tangent chord projection>=sqrtL||q||-sqrtW||q||², while rotatednormal slice projection<=R||q||/2. reach=min(L,1)²/[100(1+W)] makes equality impossible forq!=0. Thus normal tube is injective moduloZ4; nmod4=0 makesCAR wave singlevalued.',
        actual_Gauss_state='Psi(U,eta)=U_CAR Psi_body(eta) f(eta)/sqrt(J(eta)), normalized Haar onSU4/Z4, J=|detD|=sqrt(detH detS). All globalGauss generators annihilate by equivariance. E0 is exact co-local SU4singlet and n conserving. Compactbump p>=6 gives C5/H4 zero extension onphysicaluniform60Cart space; Gaussian tails not setzero. Local/nonuniform gauge remains unclaimed.',
        metric_and_quantum_proof='D=[N,O0+deltaO], N^T N=I/8,N^TO0=0,||deltaO||<=r/2. H>=L/4,||B||<=r/4,S=I/8-BH^-1B^T>=I/16. A=S^-1 differs8I by<=A0 r²; directionalderivative<=A1 r. k=A B H^-1 hasnorm<=k0r, derivative<=k1. Exact reducedkinetic=chi/2[(p-BH^-1Q)^T A(p-BH^-1Q)+Q^TH^-1Q]. FlatJ conjugation adds chi/4 div(A gradlogJ)+chi/8 gradlogJ^T A gradlogJ; crossvolume terms cancel. J affineD derivatives give gradj<=60/m0,Hessj<=60/m0,m0=min(1/8,L). FiniteCAR Q²<=n²15/8 andQvectornorm<=2n. WeightedHess/grad budget prices radial-cometric/cross/angle/volume without freezing45 metric, forANYactual reference packet withsuppliedB1/B2.',
        moment_domain_proof='b=(1-r²/R²)^p insideR,0outside; Z>=1-2p E(r²)/R² globally, byBernoulli evenoutside. Covmin/max bound ALLGaussian moments μ2k<=d(d+2)...cmax^k. Exact derivative bounds include gradients/Hessians ofb andfull finiteCAR C1/C2. PositiveSobolev p>=6 suffices D(H²) oncompactregular tube; no C1-only bump. sqrt_upper uses adaptive exponent toretain tinyphysicalmoments instead of fixed2^-256 absolute floor.',
        consumer_contract='Budget is ADDITIONAL Noether/metric action relative to45canonicalflatkinetic onSAMEreferencepacket. Consumer suppliesactualBorn C1/C2,covmin/max,R andsourceendpoints, keepscommonH phases; mustnotpromote this fixedmattertrial orprices alone tocompletedparticle propagation.')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_full_normal_global_Gauss_kinetic_prices=physical,source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},inputs_unchanged=full_path.read_bytes()==raw and uniform_path.read_bytes()==uraw,runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--full-normal',type=Path,required=True);parser.add_argument('--uniform',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result=run(args.full_normal,args.uniform)
    if not all(result['source_unchanged_since_import'].values()) or not result['inputs_unchanged']:raise ArithmeticError('Physical source/input changed duringexecution')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(input_sha256=result['input_sha256'],prices=result['actual_full_normal_global_Gauss_kinetic_prices']['actual_exact_rational_Noether_metric_prices'],times=result['actual_full_normal_global_Gauss_kinetic_prices']['additional_source_clock_budgets'],elapsed_seconds=result['elapsed_seconds']),indent=2))

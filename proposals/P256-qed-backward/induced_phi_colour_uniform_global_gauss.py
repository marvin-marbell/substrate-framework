"""NEW uniform Phi45 + colour45 + globalSU4 orbit15 quantum kinetic chart.

OriginalFullForm colour .3chi E4^2/2, not distinct LAW05 double-electric law.
Full Haar link curvature, normal/cross/angular/volume action are priced. This
helper is not by itself a propagated particle or a local/nonuniform/EM law.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from math import factorial
from pathlib import Path
import sys
import time
from typing import Any
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={Path(__file__).name:Path(__file__).read_bytes()}
sys.set_int_max_str_digits(0)
import induced_phi_full_normal_global_gauss as phi
SOURCE_BYTES.update(phi.SOURCE_BYTES)
import induced_phi_uniform_selfconsistent_full_normal_quantum as stationary
SOURCE_BYTES.update(stationary.CAPTURED)
import numpy as np
from scipy.linalg import expm
ROOT_RECEIPT_SHA='4691ffaa9cd81902942cbc228fc47fafff14518be6b5df2e84d08c29c7da4756'
FULL_RECEIPT_SHA='5b97862fc7fca21b8797d549d7d4dfdde58b101e8c46c7a7cfb6531a669c4354'
GCOLOUR=F(3,10)


def full_uniform_gauss_prices(g0_lower:F,gs_lower:F,g0_upper:F,gs_upper:F,V:int,n:int,R:F,B1:F,B2:F,Gcolour:F)->dict[str,F]:
    """Additional physical kinetic action /chi for an actual normal90 packet."""
    if not 0<g0_lower<=g0_upper or not 0<gs_lower<=gs_upper or V<1 or n<1 or n%4 or R<=0 or min(B1,B2)<0 or Gcolour<=0:
        raise ValueError('Actual positive centre-invariant particle/chart/metric data required')
    d=90;L=V*min(g0_lower,gs_lower)/2;W=V*(g0_upper+3*gs_upper)
    mc=1/Gcolour;m0=min(F(1,8),mc,L);aD=8*(1+mc);bD=32*(1+mc)
    reach=min(L,F(1))**2/(100*(1+W));b=F(1,4)+2*mc
    if R>reach or R>F(1,100) or R**2>L/4 or R**2>m0/(4*aD*aD) or R**2>m0*L/(16*b*b):
        raise ArithmeticError('Actual Phi tube, SU4 logarithm or full105 chart radius not certified')
    hL=L/4;sL=m0/4;aM=mc/V;aM1=8*mc/V;b1=4*(1+mc)
    h1=1+W+R+16*mc*R
    s0=aM+b*b/hL;s1=aM1+2*b*b1/hL+b*b*R*h1/hL**2
    A0=s0/(sL*m0);A1=s1/sL**2
    k0=b/(sL*hL);k1=s1*b*R**2/(sL**2*hL)+(b1/hL+b*R*h1/hL**2)/sL
    jg=210*aD/m0;jh=(210*bD+420*aD*aD)/m0
    geometry=(d*A1*R*jg+d*jh/sL)/4+jg*jg/(8*sL)
    normal=(phi.sqrt_upper(F(d))*A0*B2+d*A1*B1)/2
    qnorm=2*n;cf=F(15,8)
    cross=qnorm*(2*k0*B1+d*k1)/2
    angular=n*n*cf*(1/hL+b*b*R**2/(sL*hL**2))/2
    return dict(certified_Phi_tube_radius_upper=reach,orbit_mean_metric_lower=L,orbit_chart_metric_lower=hL,normal_Schur_metric_lower=sL,colour_original_cometric_divided_chi=Gcolour,normal_cometric_difference_quadratic_coefficient=A0,normal_cometric_derivative_linear_coefficient=A1,orbital_cross_cometric_linear_coefficient=k0,orbital_cross_cometric_derivative_coefficient=k1,volume_log_gradient_norm_upper=jg,volume_log_Hessian_norm_upper=jh,geometric_potential_operator_norm_divided_chi_upper=geometry,normal_kinetic_difference_action_divided_chi_upper=normal,Noether_cross_action_divided_chi_upper=cross,Noether_angular_operator_norm_divided_chi_upper=angular,total_reference_packet_Noether_and_metric_residual_divided_chi_upper=geometry+normal+cross+angular)


def adjoint_matrix(generators:np.ndarray,H:np.ndarray)->np.ndarray:
    comm=1j*(H[None]@generators-generators@H[None])
    return np.real(2*np.einsum('aij,bji->ab',generators,comm))


def haar_series(A:np.ndarray,directions:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Observed Taylor10 group Jacobian and ALL15 first/second derivatives.

    Operator proof uses the exact exponential integral, not this observation.
    At executed ||A||<4e-5, derivative2 omitted bound is below 1e-42.
    """
    P=np.eye(15);dp=np.zeros((15,15,15));dd=np.zeros((15,15,15,15))
    J=P.copy();E=P.copy();Jp=dp.copy();Ep=dp.copy();Jpp=dd.copy()
    for k in range(1,11):
        newdd=dd@A+dp[:,None]@directions[None]+dp[None]@directions[:,None]
        newdp=dp@A+P@directions
        P=P@A;dp=newdp;dd=newdd
        E+=P/factorial(k);Ep+=dp/factorial(k)
        J+=P/factorial(k+1);Jp+=dp/factorial(k+1);Jpp+=dd/factorial(k+1)
    return J,E,Jp,Ep,Jpp


def literal_chart(native:dict[str,Any],Qcart:np.ndarray,R:F,Gcolour:F)->dict[str,Any]:
    uniform=phi.scalar.uniform;primitive=uniform.supplier.static.primitive;T=np.asarray(uniform.supplier.static.force.T)
    B=primitive.clifford_cartesian_basis();generators=B/np.sqrt(8);V=native['V'];mc=float(1/Gcolour)
    S=np.kron(np.eye(4),np.real(np.einsum('aij,bji->ab',B,T))/4);N=S@Qcart
    delta=np.einsum('amk,mij->kaij',Qcart.reshape(4,15,45),T)
    phiDi=[]
    for k in range(45):
        comm=1j*(generators[:,None]@delta[k][None]-delta[k][None]@generators[:,None])
        cart=np.real(2*np.einsum('mij,anji->anm',T,comm)).reshape(15,60).T
        phiDi.append(np.concatenate((np.zeros((60,45)),S@cart),axis=1))
    adj=np.stack([adjoint_matrix(generators,H) for H in generators]);directions=adj/np.sqrt(V)
    configurations=[np.zeros(90),np.eye(90)[0],np.eye(90)[45],np.sin(np.arange(90)+1),np.cos(2*np.arange(90)+1)]
    rows=[];maximum=0.
    for k,direction in enumerate(configurations):
        eta=np.zeros(90) if k==0 else float(R)*direction/np.linalg.norm(direction)
        frame=native['frame']+np.einsum('k,kaij->aij',eta[:45],delta)/np.sqrt(V)
        comm=1j*(generators[:,None]@frame[None]-frame[None]@generators[:,None])
        cart=np.real(2*np.einsum('mij,anji->anm',T,comm)).reshape(15,60).T
        Dphi=np.concatenate((N,np.sqrt(V)*S@cart),axis=1)
        D=np.zeros((105,105));D[:60,:45]=N;D[:60,90:]=Dphi[:,45:]
        Di=np.zeros((90,105,105));Di[:45,:60,90:]=np.stack(phiDi)[:,:,45:]
        jp=np.zeros(90);jpp=np.zeros((90,90));Dphi_inv=np.linalg.solve(Dphi,np.eye(60))
        Pphi=np.stack([Dphi_inv@x for x in phiDi]);jp[:45]=np.trace(Pphi,axis1=1,axis2=2);jpp[:45,:45]=-np.einsum('aij,bji->ab',Pphi,Pphi)
        group_errors=[];haar_logs=[]
        for axis in range(3):
            body=eta[45+15*axis:60+15*axis];H=np.einsum('a,aij->ij',body,generators)/np.sqrt(V)
            A=np.einsum('a,aij->ij',body,directions);J,E,Jp,Ep,Jpp=haar_series(A,directions)
            U=expm(1j*H);literal=np.real(2*np.einsum('aij,bji->ab',generators,U[None]@generators@U.conj().T[None]))
            group_errors.append(float(np.max(abs(E-literal))))
            r=slice(60+15*axis,75+15*axis);c=slice(45+15*axis,60+15*axis)
            D[r,c]=np.sqrt(mc)*J;D[r,90:]=np.sqrt(V*mc)*(np.eye(15)-E)
            for j in range(15):
                Di[45+15*axis+j,r,c]=np.sqrt(mc)*Jp[j]
                Di[45+15*axis+j,r,90:]=-np.sqrt(V*mc)*Ep[j]
            Ji=np.linalg.solve(J,np.eye(15));products=Ji@Jp
            jp[c]=np.trace(products,axis1=1,axis2=2)
            jpp[c,c]=np.trace(Ji@Jpp,axis1=2,axis2=3)-np.einsum('aij,bji->ab',products,products)
            haar_logs.append(float(np.linalg.slogdet(J)[1]))
        metric=D.T@D;inverse=np.linalg.solve(metric,np.eye(105));Dinv=np.linalg.solve(D,np.eye(105))
        gradient=np.trace(Dinv@Di,axis1=1,axis2=2);volume_error=float(np.max(abs(gradient-jp)))
        dInverse=-(inverse@(np.swapaxes(Di,1,2)@D+D.T@Di)@inverse)
        divergence=np.array([sum(dInverse[i,i,j] for i in range(90)) for j in range(90)])
        Ac=inverse[:90,:90];geom=.25*(divergence@jp+np.trace(Ac@jpp))+.125*(jp@Ac@jp)
        reference=np.diag([8.]*45+[float(Gcolour)]*45)
        orth=float(np.max(abs(N.T@N-np.eye(45)/8)));factor=abs(float(np.linalg.slogdet(D)[1])-(float(np.linalg.slogdet(Dphi)[1])+sum(haar_logs)+45*np.log(mc)/2))
        maximum=max(maximum,orth,volume_error,factor,*group_errors)
        rows.append(dict(radius=float(np.linalg.norm(eta)),normal_metric_min=float(np.linalg.eigvalsh(metric[:90,:90])[0]),full105_metric_min=float(np.linalg.eigvalsh(metric)[0]),normal_cometric_difference_norm=float(np.linalg.norm(Ac-reference,2)),normal_orbit_cross_norm=float(np.linalg.norm(metric[:90,90:],2)),geometric_potential_divided_chi=float(geom),volume_gradient_norm=float(np.linalg.norm(jp)),volume_Hessian_norm=float(np.linalg.norm(jpp,2)),ALL90_volume_gradient_identity_error=volume_error,actual_Haar_Determinant_factorization_error=factor,actual_group_exponential_Adjoint_errors=group_errors))
    casimir=sum((Tg@Tg for Tg in generators),np.zeros((4,4),complex));ce=float(np.max(abs(casimir-np.eye(4)*15/8)))
    if maximum>1e-10 or ce>1e-12 or rows[0]['normal_cometric_difference_norm']>1e-11:
        raise ArithmeticError('Actual full105 primitive/Haar/globalGauss chart mismatch')
    return dict(Phi_normal=45,colour_normal=45,global_orbit=15,actual_ALL90_first_derivatives=True,actual_ALL90_logvolume_Hessian=True,actual_fundamental_Casimir15over8_error=ce,observations=rows,maximum_chart_identity_error=maximum,actual_original_colour_cometric_relative=float(Gcolour),exact_group_geometry='Jr(c/sqrtV)=integral0^1 exp(t ad_c/sqrtV)dt; Ocolour=sqrt(V/Gc)(I-exp(ad_c/sqrtV)). Metric is full nonlinear105, not flat 90 replicated oscillator.',literal_series_scope='Taylor10 only for binary64 observed controls; exact analytic exponential integral and universal rational operator bounds define physical geometry.')


def run(root_path:Path,full_path:Path)->dict[str,Any]:
    start=time.monotonic();raw=root_path.read_bytes();fraw=full_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ROOT_RECEIPT_SHA or hashlib.sha256(fraw).hexdigest()!=FULL_RECEIPT_SHA:
        raise ArithmeticError('Frozen f63 exact-root/full45 inputs required')
    root=json.loads(raw);full=json.loads(fraw);params=root['actual_stationary_root']['parameters']
    g0l,g0u,gs_l,gs_u=[F(params[k][side]) for k,side in (('g0','lower'),('g0','upper'),('gs','lower'),('gs','upper'))]
    mean=stationary.input_mean(dict(microscopic_config=full['input_descriptor']['original_micro_config_UNCHANGED']),[(g0l+g0u)/2,(gs_l+gs_u)/2])
    native=stationary.full.supplier.quantum.build_native(mean,4);Q=np.asarray(full['actual_full45_geometry']['Qcart_display'],float)
    R=F(1,8192);variance=F(2)**-6144;n=2*64*65+4
    packet=phi.packet_weighted_derivatives(90,variance,variance,R,6,F(0),F(0))
    prices=full_uniform_gauss_prices(g0l,gs_l,g0u,gs_u,64,n,R,packet['weighted_r_gradient_norm_upper'],packet['weighted_r2_Hessian_norm_upper'],GCOLOUR)
    chart=literal_chart(native,Q,R,GCOLOUR);source=phi.scalar.literal_source()
    times=[dict(time=str(t),additional_raw_source_Noether_metric_error_upper=phi.scalar.ratio(8*abs(t)*F(2)**-4096*prices['total_reference_packet_Noether_and_metric_residual_divided_chi_upper'])) for t in (F(0),F(1,4096),F(4096))]
    descriptor=dict(plan='issue-222-induced-photon-delta13-continuation-20261008',assigned_scientific_revision='d86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7',observed_plan_revision='1c4f98f4e0e84bd598b150f73bb276ac882acd511d6af2de5848b4d15a80417a',root_input=dict(path=str(root_path),sha256=hashlib.sha256(raw).hexdigest()),full45_input=dict(path=str(full_path),sha256=hashlib.sha256(fraw).hexdigest()),exact_mean_parameters=params,actual_original_colour_Hkin='.3chi E4^2/2; Casimir15/8, U_i=exp(i T.C_i/sqrtV), Gcolour=.3chi',Lambda='2^8192',chi='2^-4096',V=64,bulk_species=65,static_Ref4=4,finite_CAR_particles=n,normal_coordinates=90,compact_radius=str(R),trial_variance=str(variance),trial_CAR='FIXED mean all65 filled state and Ref4, so C1=C2=0 by declared preparation, NOT fallback or Born-ground claim',scope='NEW fulluniform105 geometry/Noether supplier. No local/nonuniform/EM/originalSOURCE or propagated full90 particle admission.')
    physical=dict(actual_full105_chart=chart,actual_finite_CAR_source=source,actual_packet_bounds={k:phi.scalar.ratio(v) for k,v in packet.items()},actual_exact_rational_prices={k:phi.scalar.ratio(v) for k,v in prices.items()},additional_source_clock_budgets=times,
        chart_and_domain_proof='Phi(eta,U)=U(Phi0+Qphi eta_phi/sqrtV)Udag, links U exp(iTC_i/sqrtV)Udag. Phi45 normal tube is injective moduloZ4 by frozen ab8 reach proof, hence the added SU4 links are unique under their injective small logarithm; full physical link Hilbert is SU4^3, no centre wells removed. The packet has p6 compact bump; H4 zero extension on the original uniform60Cart times SU4^3 manifold. GlobalU Haar SU4/Z4 and nmod4=0 give single-valued exact globalGauss equivariance; localGauss unclaimed. Actual E0 is co-local CAR/Ref singlet.',
        kinetic_proof='D has Phi60 rows and45 right-Maurer link rows. NormalM=diag(I45/8,(1/Gc)Jr^T Jr), orbitO includes both Phi and nonlinear colour conjugation. B=N^TO, H=O^TO, S=M-BH^-1B^T. Exact reducedkinetic is chi/2[(p-BH^-1Q)^T S^-1(p-BH^-1Q)+Q^TH^-1Q]. At0 M0=diag(I/8,I/Gc), B0=0. For r<=R, ||M-M0||<=mc r²/V, ||dM||<=8mc r/V, ||B||<=br, ||dB||<=b1, H>=L/4. Thus S>=sL, ||S^-1-M0^-1||<=A0 r², ||dS^-1||<=A1 r and cross k<=k0r,d k<=k1. Cartnormal and colour cometric are NEVER frozen in physical law.',
        Haar_and_volume_proof='J=|detD|=|detDphi| mc^(45/2) product_axis det Jr(C_i/sqrtV), exact block triangular determinant. Maurer metric retains full Haar Laplacian. Analytic exp integral gives ||Dprime||<=aD,||Dsecond||<=bD. Actual radius ensures ||Dinv||<=2/sqrt(m0). Consequently gradlogJ<=210aD/m0 and HesslogJ<=(210bD+420aD²)/m0. FlatJ conjugation gives chi/4 div(A gradlogJ)+chi/8 gradlogJ^T A gradlogJ; crossvolume terms cancel by symmetric ordering. ALL90 Hessian/gradient, colour Haar geometry, Qcross/Q² and volume prices retained.',
        consumer_contract='Additional action/chi budget applies to OWN actual90 Born/corrected/compact quantum packet when consumer supplies its proven B1=||r grad psi||,B2=||r² Hess psi||; no fixedtrial substituted for source. Consumer keeps full physicalH on both legs and conjugates reference via its CAR Kato W, prices physical E0 endpoints, all primitive potential and fermion remainders separately. Additional twoleg complex source error<=8|t|chi total for ||E0||<=2. No full90 source propagation claimed by geometry trial.')
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_uniform_Phi_colour_global_Gauss_prices=physical,source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},inputs_unchanged=root_path.read_bytes()==raw and full_path.read_bytes()==fraw,runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root-input',type=Path,required=True);parser.add_argument('--full45',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result=run(args.root_input,args.full45)
    if not all(result['source_unchanged_since_import'].values()) or not result['inputs_unchanged']:raise ArithmeticError('Scientific source/input changed during actual execution')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(input_sha256=result['input_sha256'],chart=result['actual_uniform_Phi_colour_global_Gauss_prices']['actual_full105_chart'],times=result['actual_uniform_Phi_colour_global_Gauss_prices']['additional_source_clock_budgets'],elapsed_seconds=result['elapsed_seconds']),indent=2))

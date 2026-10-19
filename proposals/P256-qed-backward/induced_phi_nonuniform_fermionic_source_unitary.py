"""NEW N4 full5760 normal/local960-Gauss corrected finite-CAR source.

The comparison has the same compact prepared body on both legs. Its physical
unitary is conjugated by all local SU4 carriers and radial occupied Kato W.
No field-Hilbert truncation, private sea phase, pointwise density assumption,
or original SOURCE/QED promotion is made.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES:dict[str,bytes]={Path(__file__).name:Path(__file__).read_bytes()}
class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self,fullname:str)->Any:
        raw=Path(self.path).read_bytes();SOURCE_BYTES[Path(self.path).name]=raw
        return self.source_to_code(raw,self.path)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname:str,path:Any=None,target:Any=None)->Any:
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is not None and spec.origin is not None and Path(spec.origin).parent==ROOT and spec.origin.endswith('.py'):
            spec.loader=Loader(fullname,spec.origin)
        return spec
finder=Finder();sys.meta_path.insert(0,finder)
import induced_phi_nonuniform_field_car as field
import induced_phi_nonuniform_gauss_quantum as quantum
import induced_phi_nonuniform_local_gauss as local
import induced_phi_full_uniform_colour_fermionic_source_unitary as radial
sys.meta_path.remove(finder)
SOURCE_BYTES.update(field.CAPTURED);SOURCE_BYTES.update(quantum.SOURCE_BYTES)
SOURCE_BYTES.update(local.SOURCE_BYTES);SOURCE_BYTES.update(radial.SOURCE_BYTES)
import mpmath as mp
import numpy as np
from scipy.linalg import expm,solve_sylvester
from scipy.sparse import eye as sparse_eye
from threadpoolctl import threadpool_limits

iv:Any=radial.iv
exact,cap,pack,endpoint,interval=radial.exact,radial.cap,radial.pack,radial.endpoint,radial.interval
Poly=radial.Poly
term,add,multiply,scale,power=radial.term,radial.add,radial.multiply,radial.scale,radial.power
FIELD_SHA='55661707f096422947a816924664de94e8da1d092e278164e4e9be57a26e775c'
QUANTUM_SHA='2bd70176910108fd695c5490b31ce490da70c860a9a43693758479ec2ccc2c06'
LOCAL_SHA='73ca8d12378d4d5ef6371e11270f967d891f6be0a7e85f200c16440736bf72b4'
R=F(1,1<<20)
sys.set_int_max_str_digits(0)

class Moments:
    """Independent slow and conditional fast marginals of the FULL joint wave."""
    def __init__(self,slow:Any,fast:Any):
        self.variances=(slow,fast);self.dimensions=(45,5715);self.cache={}
    def moment(self,which:int,degree:int,p:int)->Any:
        key=(which,degree,p)
        if key not in self.cache:
            if p*degree%2:raise ValueError('Even Gaussian moment required')
            n=p*degree//2;factor=1
            for j in range(n):factor*=self.dimensions[which]+2*j
            self.cache[key]=cap((exact(factor)*self.variances[which]**n)**exact(F(1,p)))
        return self.cache[key]
    def norm(self,polynomial:Poly,p:int=2)->Any:
        return cap(sum(c*self.moment(0,a,p)*self.moment(1,b,p) for (a,b),c in polynomial.items()))

def upper(record:dict[str,Any])->Any:
    return exact(F(record['upper']))

def corrected_derivatives(m:dict[str,Any],p:Any,cmin:Any,C1:Any,C2:Any)->dict[str,Any]:
    """Gaussian hypercontractivity for degree3 P, with actual joint Wick traces."""
    mall=upper(m['physical_trace_ALL_covariance_upper']);mf=upper(m['physical_trace_fast_covariance_upper'])
    trA=upper(m['physical_trace_A_upper']);g=upper(m['Gaussian_gradient_norm_upper'])
    gp=cap(iv.sqrt(3/cmin)*p);hp=cap(iv.sqrt(exact(6))/cmin*p)
    rf=iv.sqrt(mf);rt=iv.sqrt(mall);rv=exact(R)
    Z=cap(1-24*mall/rv**2-4*p)
    if endpoint(Z,0)<=F(1,2):raise ArithmeticError('Actual corrected compact normalizer not positive')
    norm=cap(1/iv.sqrt(Z))
    bumpg=12/rv**2
    gradient=cap(norm*(g+9*p*g+gp+bumpg*rt*(1+9*p))+C1)
    # Component tensors rfast Hess(f)/f and rALL^2 Hess(f)/f have degree3/4.
    # Hypercontractivity and Holder act on the true mixed Gaussian, not Amax*r.
    common=cap(p*trA+gp*g+hp+bumpg*rt*(p*g+gp)
        +(12*iv.sqrt(exact(5760))/rv**2+120*mall/rv**4)*p
        +(C2+C1*C1)*p+C1*(p*g+gp+bumpg*rt*p))
    baseline=local.gaussian_weighted_bounds(5760,5715,F(endpoint(cmin,0)),F(endpoint(mall,1)),F(m['physical_fast_covariance_eigen_upper']['upper']),R,6,F(endpoint(C1,1)),F(endpoint(C2,1)))
    Hfast=cap(norm*(exact(baseline['weighted_rfast_Hessian_norm_upper'])+4096*rf*common))
    H2=cap(norm*(exact(baseline['weighted_r2_Hessian_norm_upper'])+65536*mall*common))
    return dict(normalizer_lower=Z,normalization_norm_upper=norm,gradient=gradient,Hfast=Hfast,H2=H2,
        corrector_gradient=gp,corrector_Hessian=hp,corrector_degree=3,
        proof='Full-joint degree<=3 hypercontractivity ||P||r<=(r-1)^1.5||P||2; Dirichlet ||gradP||<=sqrt(3/cmin)p and ||HessP||<=sqrt6 p/cmin. Joint Wick A Cov A=.5A prices weighted Hessians. Added derivatives, bump and ALL-CAR terms explicitly.')

def source_prices(data:dict[str,Any],matter:dict[str,Any],supplier:dict[str,Any])->dict[str,Any]:
    root=supplier['actual_allq_feedback_mean'];g0,gs=(interval(root['parameters'][k]) for k in ('g0','gs'))
    m=supplier['actual_joint_Gaussian_corrector_moments'];q=supplier['actual_full_field_covariance']
    p=upper(m['corrector_preparation_norm_upper']);cfast=upper(m['physical_fast_covariance_eigen_upper'])
    cmin=exact(F(m['physical_fast_covariance_eigen_lower']['lower'])/4)
    cslow=upper(m['physical_trace_ALL_covariance_upper'])
    moments=Moments(cslow,cfast)
    s,z=term(exact(1),1),term(exact(1),0,1)
    Lambda,Lambda54,chi=exact(F(2)**8192),exact(F(2)**10240),exact(F(2)**-4096)
    hmin=exact(min(endpoint(g0,0),endpoint(gs,0))/2)
    rows=matter['exact_ALLq_operator_bounds'];sea_bound=sum(exact(F(r['load']))*upper(r['sea90_operator_upper']) for r in rows)
    slow_lower=exact(F(q['certified_q0_slow_positive_lower']['lower']))
    Kslow=cap(4096*(1+g0+3*gs)+sea_bound)
    Gmax=cap(8+exact(F(3,10))+12/hmin)
    # Block-Riccati/Schur comparison in q0; true M is A_CC^-1 A_CP.
    comparison=cap(exact(2**24)*(1+sea_bound+Kslow)**3*(1+1/hmin+1/slow_lower)**4*(1+Gmax)**4)
    M=cap(comparison*exact(F(2)**-2048))
    epsilon=cap(comparison*exact(F(2)**-1024))
    if endpoint(epsilon,1)>=F(1,4):raise ArithmeticError('Full matrix wave/primitive conditional comparison not controlled')
    fastcoord=add(z,scale(s,M));radius=add(s,fastcoord)
    species=[];C1=exact(0);C2=exact(0);activeD=None;activeGap=None
    for profile,r in zip(data['profiles'],rows,strict=True):
        dp=upper(r['Phi_vertex_operator_upper']);dc=upper(r['colour_vertex_operator_upper'])
        gap=exact(F(r['minimum_gap_unscaled']['lower']))
        D=cap(dp+dc);distance=cap(gap/2-2*D*exact(R))
        if endpoint(distance,0)<=0:raise ArithmeticError('Entire supported256 occupied gap not certified')
        # Adapted frame holomorphic norm<=2 at rho1/256: Q1<=512,Q2<=2^20.
        D1=cap(D*(1+512*exact(R)))
        D2=cap(8*D+4*dp*(512+exact(2**20)*exact(R))+4*upper(r['PC_contact_entry_upper'])+4*upper(r['CC_contact_entry_upper']))
        b1=cap(8*D1/distance);b2=cap(32*D1*D1/distance**2+8*D2/distance)
        count=exact(128*profile['load']+(1 if profile['role']==data['profiles'][1]['role'] else 0))
        C1+=count*b1;C2+=count*b2
        species.append(dict(role=profile['role'],load=profile['load'],occupied_rank_per_copy=128,
            physical_scale=profile['scale'],distance_to_zero_lower=pack(distance),all_normal_first_h_derivative_upper=pack(D1),
            all_normal_second_h_derivative_upper=pack(D2),occupied_Kato_first_upper=pack(b1),occupied_Kato_second_upper=pack(b2)))
        if profile['role']==data['profiles'][1]['role']:activeD=D1;activeGap=distance
    if activeD is None or activeGap is None:raise ArithmeticError('Physical active source profile absent')
    deriv=corrected_derivatives(m,p,cmin,cap(C1),cap(C2))
    metric=local.local_gauss_prices(F(endpoint(g0,0)),F(endpoint(gs,0)),F(endpoint(g0,1)),F(endpoint(gs,1)),64,8324,R,
        F(endpoint(deriv['gradient'],1)),F(endpoint(deriv['Hfast'],1)),F(endpoint(deriv['H2'],1)))
    geometry=cap(chi*exact(metric['total_reference_local_Gauss_kinetic_remainder_divided_chi_upper']))
    kato1=cap(2*chi*Gmax*C1*deriv['gradient'])
    kato2=cap(chi*Gmax*5760*(C2+C1*C1)/2)
    # Primitive allq kinetic/potential Xi, in physical variables.
    Xsquared=exact(0)
    for group in supplier['actual_exact_algebraic_fast_coefficients']['all_physical_q_groups']:
        for e in group['entries']:
            xl,xt=interval(e['XL']),interval(e['XT'])
            Xsquared+=exact(group['multiplicity'])*(abs(xl)**2+2*abs(xt)**2)
    AX=cap(iv.sqrt(Xsquared)*exact(F(2)**7168));BX=cap(5715*AX*cfast)
    gradSlowP=add(term(AX,0,2),term(BX),term(2*AX*M,1,1))
    hessSlowP=add(term(4*AX*M,0,1),term(2*AX*M*M,1,0))
    aPhi=cap(iv.sqrt(Kslow/8)*exact(F(2)**6144))
    logSlow4=cap(iv.sqrt(exact(F(135,2))*aPhi))
    slow_action=cap(8*chi*(logSlow4*moments.norm(gradSlowP,4)+exact(F(45,2))*moments.norm(hessSlowP)))
    # Exact ALLq root cancels only bare Phi force + FULL leading fast-ZPE trace.
    # True static-matter wave/kinetic-Xi deviations are retained, not frozen away.
    mixed_action=cap(comparison*exact(F(2)**-1024))
    omitted=sum(upper(q['remainder_force_budgets'][k]) for k in ('matter_fast_force_times_slow_std_upper','slowPhi_ZPE_force_times_slow_std_upper'))
    root_residual=cap(Lambda*comparison*upper(root['root_radius'])*iv.sqrt(cslow))
    # Original alignment's adapted fast Phi-C cross is identically0. Remaining
    # fast cubic (nonuniform Phi/C and plaquette) is NOT the uniform quartic law.
    frame=cap(2*iv.sqrt(g0+3*gs))
    primitive=scale(add(power(radius,3),power(radius,4)),Lambda*exact(2**20))
    fast_nonlin=scale(add(power(fastcoord,3),power(fastcoord,4)),Lambda54*exact(2**24)*(1+frame)**4)
    slowfast=scale(multiply(power(s,2),power(fastcoord,2)),Lambda54*exact(2**24)*(1+frame)**4)
    # Adapted-frame Phi injection: linear trace of every q!=0 fastPhi is0
    # at ANY uniform slowPhi. Thus no false Lambda*slow*fast quadratic residue.
    sea_nonlinear=scale(add(power(radius,3),power(radius,4)),Lambda*comparison)
    nonlinear=cap(moments.norm(add(primitive,fast_nonlin,slowfast,sea_nonlinear),4)*(1+9*p))
    corrected_linear=cap(comparison*p)
    bump=cap(chi*Gmax*(12*5760/exact(R)**2+24*deriv['gradient']*iv.sqrt(cslow)/exact(R)**2))
    complete=cap((slow_action+mixed_action+omitted+root_residual+nonlinear+corrected_linear+bump+kato1+kato2+geometry)*deriv['normalization_norm_upper'])
    # Body comparator scalar Efield fixes the actual corrected compact packet;
    # it is NOT the harmonic operator acting again on (1-P)f.
    points=radial.joined.bloch_points(data['profiles'],g0,gs)
    H=[sum(exact(F(int(n),2))*g for n,g in zip(mask,(g0,gs,gs,gs))) for mask in quantum.masks(data['geom']['M'])]
    Efast=exact(0)
    for group in supplier['actual_exact_algebraic_fast_coefficients']['all_physical_q_groups']:
        q2,count=group['qhat2'],group['multiplicity']
        colours=sum(iv.sqrt(exact(F(3,50))*h+exact(F(q2,5)))+2*iv.sqrt(exact(F(3,50))*h+exact(F(21*q2,200))) for h in H)
        phifast=45*iv.sqrt(exact(F(q2,5))) if q2 else exact(0)
        Efast+=exact(count)*(colours+phifast)*exact(F(2)**3072)/2
    Efield=radial.enclosure(endpoint((1-epsilon)*Efast+exact(F(45,2))*iv.sqrt(8*slow_lower)*exact(F(2)**2048),0),
        endpoint((1+epsilon)*Efast+exact(F(45,2))*iv.sqrt(8*Kslow)*exact(F(2)**2048),1))
    Eprimitive=Lambda*4096*(g0*g0+3*gs*gs)
    Esea=sum(-2*profile['load']*(Lambda if profile['scale']=='heavy' else exact(1))*sum(row['E'] for row in point)
        for profile,point in zip(data['profiles'],points,strict=True))
    Ereference=Eprimitive+Esea+Efield
    endpoint_error=cap(16*activeD/activeGap*moments.norm(radius,4)*(1+9*p)*deriv['normalization_norm_upper'])
    source_h=cap(4*activeD*moments.norm(radius,4)*(1+9*p)*deriv['normalization_norm_upper'])
    times=[]
    for t in (F(0),F(1,4096),F(4096)):
        re=sum(iv.cos(row['E']*exact(t)) for row in points[1])/128
        im=-sum(iv.sin(row['E']*exact(t)) for row in points[1])/128
        err=exact(0) if not t else cap(8*exact(abs(t))*complete+4*exact(abs(t))*source_h+8*endpoint_error)
        modulus=iv.sqrt(re*re+im*im);lower=endpoint(modulus-err,0)
        if lower<0:raise ArithmeticError('Actual raw twoleg source modulus proof failed')
        times.append(dict(time=str(t),raw_reference_real=pack(re),raw_reference_imag=pack(im),
            actual_complex_error_upper=pack(err),nonzero_modulus_lower=pack(exact(lower)),
            raw_compound_probability=pack(radial.enclosure(lower**2,min(F(1,2),endpoint((modulus+err)**2,1)))),
            actual_raw_real=pack(radial.enclosure(endpoint(re,0)-endpoint(err,1),endpoint(re,1)+endpoint(err,1))),
            actual_raw_imag=pack(radial.enclosure(endpoint(im,0)-endpoint(err,1),endpoint(im,1)+endpoint(err,1)))))
    tailmargin=cap(exact(R)**2/(8*cslow)-exact(F(32768)+F(5760,4))*iv.ln(exact(2)))
    if endpoint(tailmargin,0)<=0:raise ArithmeticError('Full5760 compact Gaussian tail not controlled')
    return dict(radius_exact=str(R),normal_dimensions=5760,fast_dimensions=5715,slow_dimensions=45,local_angle_dimensions=960,
        full_CartPhi_dimension=3840,full_SU4_link_count=192,total_physical_chart_dimension=6720,
        exact_joint_wave_ABI=supplier['actual_full_field_wave_ABI'],actual_joint_corrector_L2_upper=pack(p),
        conditional_mean_M_upper=pack(M),full_mixed_wave_relative_comparison_upper=pack(epsilon),
        actual_corrected_derivatives={k:pack(v) if k not in ('proof','corrector_degree') else v for k,v in deriv.items()},
        actual_full65_occupied_projector_Kato_prices=species,all_CAR_first_upper=pack(cap(C1)),all_CAR_second_upper=pack(cap(C2)),
        actual_local_Gauss_Haar_Noether_prices={k:pack(exact(v)) for k,v in metric.items()},
        residual_components=dict(slow_action_on_corrector=pack(slow_action),full_matter_kinetic_resolvent_difference=pack(mixed_action),
            omitted_fastPH_slowPhi_ZPE_force=pack(cap(omitted)),root_interval=pack(root_residual),actual_primitive_and_fullsea_nonlinear=pack(nonlinear),
            slow_linear_operator_times_actual_corrector=pack(corrected_linear),compact_bump=pack(bump),full_CAR_Kato_first=pack(kato1),
            full_CAR_Kato_second_square=pack(kato2),actual_local_Gauss_Haar_Noether_remainder=pack(geometry)),
        physical_reference_energy_residual_upper=pack(complete),actual_prepared_Rayleigh_residual_upper=pack(2*complete),
        raw_primitive_centre_energy=pack(Eprimitive),raw_all65_occupied_centre_energy=pack(Esea),
        raw_full5760_harmonic_energy_enclosure=pack(Efield),raw_same_physical_H_reference_energy_enclosure=pack(Ereference),
        actual_prepared_Rayleigh_energy_enclosure=pack(Ereference+radial.enclosure(-endpoint(complete,1),endpoint(complete,1))),
        source_endpoint_error_upper=pack(endpoint_error),source_positive_block_variation_upper=pack(source_h),raw_source_times=times,
        exact_source_probability='1/2 after FULL translation-invariant corrected-field integration, never pointwise TrP00=2. Active occupied rank128, lattice64 and Ref4 full SU4 singlet give E0 norm^2=.25(4-128/64).',
        exact_corrector='P=sum_i PhiSlow_i sum_all64q [Z_C(q)^* X_i(q) Z_C(q)-Tr(Cov0_C(q) X_i(q))], canonical REAL pairs counted once; Z_C(0)=C(0)+M PhiSlow, M EXACT fullmatrix A_CC^-1 A_CP. AGX+XGA=.5(dK-A dG A), dG!=0 for q!=0. Body=Z^-1/2 b6(1-P) f_A.',
        physical_comparison='BODY Href=Eprimitive_mean+Eexact_joint5760+dGamma(hmean); bosonic scalarI fixes SAMEcompactcorrectedpacket bothlegs. PHYSICAL comparator U_local,CAR W_Kato Href W_Kato^dag U_local,CAR^dag, scalar sqrtJ conjugation. Exact Etilde=W^dag E0 W; baremeanE0 only endpoint-priced. ActualH unrestricted fullCartPhi3840/SU4^192 on bothlegs; common sea phase cancels ONLY in actual paired innerproduct.',
        domain='C5 b6 supportedR2^-20, actual polynomialdegree3, smooth positive-gap all256 occupied Kato and analytic6720 chart imply D(H²). All localUx integrated normalized Haar; chart angular redundancy commonZ4 compatible n8324mod4=0. Principal-log W collar through3R extends comparison unitary, identity outside; no compact-harmonic propagation assumption.',
        covariance_and_Hermite_proof=deriv['proof'],positive_full5760_Gaussian_tail_upper=pack(exact(F(2)**-32768)),tail_log_margin=pack(tailmargin))

def actual_allq_corrector_action(data:dict[str,Any])->dict[str,Any]:
    """Direct physical fast differential action, scaled by its exact powers."""
    T=data['geom']['M']/np.sqrt(8);frame=data['native']['frame'];O=quantum.orbit(frame,T);H=O.T@O/8;Hi=np.linalg.inv(H)
    tangents=data['geom']['deltas']/8
    dH=np.asarray([(quantum.orbit(t,T).T@O+O.T@quantum.orbit(t,T))/8 for t in tangents])
    rng=np.random.default_rng(2227168);slow=rng.normal(size=45)/np.sqrt(45)
    z=rng.normal(size=(64,45))+1j*rng.normal(size=(64,45))
    qindex={q:i for i,q in enumerate(quantum.QINTEGER)}
    for qi,q in enumerate(quantum.QINTEGER):
        partner=tuple((-n)%4 for n in q);j=qindex[partner]
        if qi==j:z[qi]=z[qi].real
        elif qi<j:z[j]=z[qi].conj()
    records=[];action=0.;target=0.;maxdefect=0.;polynomial=0.
    for qi,q in enumerate(quantum.QINTEGER):
        d=1-np.exp(2j*np.pi*np.asarray(q)/4);dd=np.outer(d,d.conj());q2=float(np.vdot(d,d).real)
        G=.3*np.eye(45)+np.kron(dd,Hi)
        K=.2*np.kron(np.eye(3),H)+.35*np.kron(q2*np.eye(3)-dd,np.eye(15))
        wave=quantum.observed_matrix_wave(G,K);A=wave['A'];cov=wave['covariance'];g=G@A
        dkin=0.;defect=0.
        for i,dh in enumerate(dH):
            dG=-np.kron(dd,Hi@dh@Hi);dK=.2*np.kron(np.eye(3),dh)
            Xi=(dK-A@dG@A)/2;X=solve_sylvester(A@G,G@A,Xi)
            # Product rule applied to P f: gradP=2Xz, HessP=2X.
            direct=float(np.real(2*np.vdot(A@z[qi],G@(X@z[qi]))-np.trace(G@X)))
            zpe=float(np.real(np.trace(A@dG+np.linalg.solve(A,dK))/4))
            physical=float(np.real(np.vdot(z[qi],Xi@z[qi])+np.trace(dG@A)/2-zpe))
            action+=slow[i]*direct;target+=slow[i]*physical
            polynomial+=slow[i]*float(np.real(np.vdot(z[qi],X@z[qi])-np.trace(cov@X)))
            dkin=max(dkin,float(np.linalg.norm(dG,2)));defect=max(defect,abs(direct-physical),float(np.max(abs(A@G@X+X@G@A-Xi))))
        records.append(dict(q=list(q),dG_operator_norm=dkin,direct_mean_subtracted_fast_action_defect=defect,
            actual_fast_colour_G_operator_norm=float(np.linalg.norm(G,2))))
        maxdefect=max(maxdefect,defect)
    if maxdefect>2e-10 or abs(action-target)>2e-10:raise ArithmeticError('Actual ALL64q kinetic Hermite product-rule identity failed')
    return dict(all64q=records,actual_direct_fast_action=action,actual_full_dK_dG_target=target,
        actual_total_action_defect=abs(action-target),maximum_normalized_Sylvester_product_rule_defect=maxdefect,
        actual_P_divided_2power_minus3072=polynomial,
        nonzero_q_kinetic_derivative_count=sum(r['dG_operator_norm']>0 for r in records),
        coefficient_status='Numerical physical differential observations ONLY; wave coefficients are exact algebraic interval definitions. Allq real-pair sum, physical Hfast times PhiSlow scales 2^3072*2^-3072=1; staticPH/conditional deviations outward priced separately.')

def adapted_jet(frame:np.ndarray,tangent:np.ndarray,T:np.ndarray,Q0:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    O=quantum.orbit(frame,T);Od=quantum.orbit(tangent,T);L=O.T@O;Li=np.linalg.inv(L)
    Ld=Od.T@O+O.T@Od;P=np.eye(60)-O@Li@O.T
    Pd=-Od@Li@O.T-O@Li@Od.T+O@Li@Ld@Li@O.T
    D=Q0.T@P@Q0;Dd=Q0.T@Pd@Q0;w,U=np.linalg.eigh(D)
    if w[0]<=0:raise ArithmeticError('Actual adapted fibre lost rank')
    S=(U*np.sqrt(w))@U.T;Si=(U/w**.5)@U.T
    Sd=solve_sylvester(S,S,Dd);Sid=-Si@Sd@Si
    return P@Q0@Si,Pd@Q0@Si+P@Q0@Sid

def field_ray(data:dict[str,Any],t:float,slow:np.ndarray,fast:np.ndarray,C:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    native,geom=data['native'],data['geom'];T=geom['M']/np.sqrt(8)
    df=np.einsum('a,amij->mij',slow,geom['deltas'])/8
    frame=native['frame']+t*df;Q,Qd=adapted_jet(frame,df,T,geom['Q'])
    eta=np.einsum('ba,xa->xb',Q,fast).reshape(64,4,15)
    etad=np.einsum('ba,xa->xb',Qd,fast).reshape(64,4,15)
    dfast=np.einsum('xma,aij->xmij',eta,T);dfastd=np.einsum('xma,aij->xmij',etad,T)
    fields=frame[None]+t*dfast;fieldsd=df[None]+dfast+t*dfastd
    X=np.einsum('xa,aij->xij',C.reshape(192,15),T)
    links=np.asarray([expm(1j*t*x) for x in X]);linksd=1j*X@links
    return fields,fieldsd,links,linksd

def h_jet(profile:dict[str,Any],native:dict[str,Any],fields:np.ndarray,fd:np.ndarray,links:np.ndarray,ld:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    h=field.assemble(profile,native,fields,links);dh=np.zeros_like(h)
    for x in range(64):dh[4*x:4*x+4,4*x:4*x+4]=float(profile['onsite'])*fd[x,0]
    for path in field.native_paths(profile,native):
        x,y=int(path['source']),int(path['target']);U=np.eye(4,dtype=complex);Ud=np.zeros((4,4),complex)
        for e in path['edges']:Ud,U=Ud@links[e]+U@ld[e],U@links[e]
        c=path['coefficients'];Fx,Fy=np.einsum('m,mij->ij',c,fields[x]),np.einsum('m,mij->ij',c,fields[y])
        Fxd,Fyd=np.einsum('m,mij->ij',c,fd[x]),np.einsum('m,mij->ij',c,fd[y])
        hop=path['seam']*(Fxd@U+Fx@Ud+Ud@Fy+U@Fyd)/2
        dh[4*x:4*x+4,4*y:4*y+4]+=hop;dh[4*y:4*y+4,4*x:4*x+4]+=hop.conj().T
    return h,(-1 if profile['scale']=='heavy' else 1)*dh

def occupied(h:np.ndarray,dh:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    w,V=np.linalg.eigh(h)
    if not w[127]<0<w[128]:raise ArithmeticError('Global physical occupied rank128 violated')
    N,E=V[:,:128],V[:,128:];P=N@N.conj().T
    cross=(E.conj().T@dh@N)/(w[:128][None]-w[128:,None]);Pd=E@cross@N.conj().T
    Pd=Pd+Pd.conj().T;B=Pd@P-P@Pd
    return P,B,w,E

def literal_source(Psite:np.ndarray,generators:np.ndarray)->dict[str,Any]:
    scalar=radial.joined.scalar;ops=[scalar.car_annihilator(i,8) for i in range(8)];I=sparse_eye(256,format='csr');defect=0.
    for i,a in enumerate(ops):
        for j,b in enumerate(ops):
            diff=a@b.conj().T+b.conj().T@a-(I if i==j else 0*I)
            if diff.nnz:defect=max(defect,float(np.max(abs(diff.data))))
    E=ops[0].conj().T@ops[4]/2
    for i in range(1,4):E=E+ops[i].conj().T@ops[4+i]/2
    source_gauge_commutator=0.
    for T in generators:
        Q=0*I
        for i in range(4):
            for j in range(4):
                Q=Q+T[i,j]*(ops[i].conj().T@ops[j]+ops[4+i].conj().T@ops[4+j])
        comm=Q@E-E@Q
        if comm.nnz:source_gauge_commutator=max(source_gauge_commutator,float(np.max(abs(comm.data))))
    if source_gauge_commutator>1e-14:raise ArithmeticError('Physical co-local source is not local SU4 singlet')
    occ,U=np.linalg.eigh(Psite)
    if occ.min()<-1e-12 or occ.max()>1+1e-12:raise ArithmeticError('Full256 reduced Slater occupations invalid')
    weights=np.asarray([np.prod([occ[i] if state&(1<<i) else 1-occ[i] for i in range(4)]) for state in range(16)])
    wedge=scalar.exterior_unitary(U);rho=np.zeros((256,256),complex);rho[240:,240:]=(wedge*weights)@wedge.conj().T
    out=E@rho@E.conj().T;prob=float(np.trace(out).real)
    nref=np.asarray([((state>>4)&15).bit_count() for state in range(256)]);wrong=float(np.trace(out[np.ix_(nref!=3,nref!=3)]).real)
    prediction=float((4-np.trace(Psite).real)/4)
    if defect or abs(prob-prediction)>2e-12 or abs(wrong)>1e-14:raise ArithmeticError('Actual local CAR source identity failed')
    return dict(CAR_defect=defect,point_source_norm_squared=prob,point_CAR_density_prediction=prediction,
        Ref3_error=wrong,reduced_occupation_eigenvalues=occ.tolist(),point_trace_occupied=float(np.trace(Psite).real),
        actual_all15_origin_local_Noether_source_commutator=source_gauge_commutator,
        source_semantics='RAW point source density generally NOT.5; .5 is translation-integrated background norm.')

def literal_full_field(data:dict[str,Any])->dict[str,Any]:
    native,geom=data['native'],data['geom'];rng=np.random.default_rng(2225760);r=float(R)
    slow=rng.normal(size=45);slow*=.17*r/np.linalg.norm(slow)
    fast=rng.normal(size=(64,45));fast-=fast.mean(axis=0);fast*=.31*r/np.linalg.norm(fast)
    C=rng.normal(size=(64,3,15));C*=.29*r/np.linalg.norm(C)
    fields,fd,links,ld=field_ray(data,1.,slow,fast,C)
    rows=[];activeP=None;activeh=None
    for profile in data['profiles']:
        h,dh=h_jet(profile,native,fields,fd,links,ld);P,B,w,positive=occupied(h,dh)
        f0,df0,u0,du0=field_ray(data,0.,slow,fast,C);h0,d0=h_jet(profile,native,f0,df0,u0,du0);P0,_,_,_=occupied(h0,d0)
        W=np.eye(256,dtype=complex)
        for k in range(8):
            fk,dk,uk,duk=field_ray(data,(k+.5)/8,slow,fast,C)
            hk,hd=h_jet(profile,native,fk,dk,uk,duk);_,generator,_,_=occupied(hk,hd)
            W=expm(generator/8)@W
        transport=float(np.max(abs(W@P0@W.conj().T-P)));unitarity=float(np.max(abs(W.conj().T@W-np.eye(256))))
        if max(transport,unitarity)>2e-10:raise ArithmeticError('Actual radial full256 occupied Kato transport failed')
        rows.append(dict(role=profile['role'],load=profile['load'],occupied_rank=128,
            physical_heavy_upper_original_h=profile['scale']=='heavy',global_gap=float(w[128]-w[127]),
            radial_Kato_transport_defect=transport,unitarity_defect=unitarity,
            occupied_only_energy=float(w[:128].sum()),no_within_band_division=True))
        if profile['role']==data['profiles'][1]['role']:
            activeP,activeh=P,h
            amplitudes=.5*positive[:4].conj().T*np.asarray([(-1)**a for a in range(4)])[None]
            exteriorprob=float(np.sum(abs(amplitudes)**2))
            rows[-1]['physical_fullCAR_128positive_by4Refholes_norm_squared']=exteriorprob
    if activeP is None or activeh is None:raise ArithmeticError('Literal active full source absent')
    densities=np.asarray([np.trace(activeP[4*x:4*x+4,4*x:4*x+4]).real for x in range(64)])
    localcar=literal_source(activeP[:4,:4],geom['M']/np.sqrt(8));avg=float(np.mean((4-densities)/4))
    deviation=float(np.max(abs(densities-2)))
    if abs(avg-.5)>2e-12 or deviation<=1e-12:raise ArithmeticError('Translation versus point-density discriminator not exercised')
    # An actual translated nonlinear configuration, including fermionic AP seams.
    sites=native['sites'];Tshift=np.zeros((256,256),complex);fshift=np.empty_like(fields);ushift=np.empty_like(links)
    for x in range(64):
        y=int(np.ravel_multi_index(tuple((sites[x]+np.asarray([1,0,0]))%4),(4,4,4)))
        sign=-1 if sites[x,0]==3 else 1;Tshift[4*y:4*y+4,4*x:4*x+4]=sign*np.eye(4)
        fshift[y]=fields[x];ushift[3*y:3*y+3]=links[3*x:3*x+3]
    translated=field.assemble(data['profiles'][1],native,fshift,ushift)
    translation_error=float(np.max(abs(translated-Tshift@activeh@Tshift.conj().T)))
    gauges=np.asarray([expm(1j*.17*np.einsum('a,aij->ij',rng.normal(size=15),geom['M']/np.sqrt(8))) for _ in range(64)])
    fg=gauges[:,None]@fields@gauges.conj().transpose(0,2,1)[:,None];ug=np.empty_like(links)
    for x in range(64):
        for axis in range(3):
            y=int(np.ravel_multi_index(tuple((sites[x]+np.eye(3,dtype=int)[axis])%4),(4,4,4)))
            ug[3*x+axis]=gauges[x]@links[3*x+axis]@gauges[y].conj().T
    hg=field.assemble(data['profiles'][1],native,fg,ug);gb=np.zeros((256,256),complex)
    for x in range(64):gb[4*x:4*x+4,4*x:4*x+4]=gauges[x]
    gauge_error=float(np.max(abs(hg-gb@activeh@gb.conj().T)))
    Pg,_,_,_=occupied(hg,np.zeros_like(hg));gaugeprob=float((4-np.trace(Pg[:4,:4]).real)/4)
    if max(translation_error,gauge_error,abs(gaugeprob-localcar['point_source_norm_squared']))>2e-10:raise ArithmeticError('Actual localU/source/translation covariance failed')
    return dict(radius=r,nonuniform_fast_coordinate_sum_defect=float(np.max(abs(fast.sum(axis=0)))),
        all65_full256_occupied_radial_Kato=rows,actual_origin_CAR_Ref4_to3=localcar,
        all64_site_occupied_density=densities.tolist(),point_density_maximum_deviation_from2=deviation,
        actual_translation_averaged_raw_source_norm_squared=avg,actual_AP_seam_translation_error=translation_error,
        actual_all64_nonuniform_local_SU4_covariance_error=gauge_error,actual_colocal_source_gauge_norm_error=abs(gaugeprob-localcar['point_source_norm_squared']),
        exact_density_proof='Translate bulk fields/fermions, Ref4 spectator is isolated full SU4 singlet. f_A, allq centredHermiteP, radialb6 and adaptedQslow are translation-equivariant; covariance input q/-q pairs and all64 sums. Global rank128 pointwise =>integrated TrP00=128/64=2. No pointwise occupancy shortcut.')

def run(matter_path:Path,quantum_path:Path)->dict[str,Any]:
    started=time.monotonic();raw=matter_path.read_bytes();qraw=quantum_path.read_bytes();matter=json.loads(raw);supplier=json.loads(qraw)
    for module,sha in ((field,FIELD_SHA),(quantum,QUANTUM_SHA),(local,LOCAL_SHA)):
        name=module.__name__+'.py'
        if hashlib.sha256(SOURCE_BYTES[name]).hexdigest()!=sha:raise ArithmeticError('Actual frozen scientific source epoch required')
    for receipt in (matter,supplier):
        if not all(receipt['source_unchanged_since_import'].values()):raise ArithmeticError('Supplier epoch changed during actual scientific execution')
        for name,source in receipt['source_at_import'].items():
            if name in SOURCE_BYTES and hashlib.sha256(SOURCE_BYTES[name]).hexdigest()!=source['sha256']:raise ArithmeticError('Joined scientific receipt/source bytes mismatch')
    if matter['input_descriptor']['physical_mean']['exact_root']['centre_exact_rational']!=supplier['actual_allq_feedback_mean']['centre_exact_rational']:raise ArithmeticError('Allq source mean mismatch')
    iv.dps=3000;mp.mp.dps=80;data=field.load_quantum_mean()
    if [p['load'] for p in data['profiles']]!=[1,1,63]:raise ArithmeticError('Original all65 physical loads changed')
    proof=source_prices(data,matter,supplier)
    with threadpool_limits(limits=1):
        literal=literal_full_field(data)
        corrector_control=actual_allq_corrector_action(data)
    proof['actual_ALL64q_kinetic_fast_Hermite_differential_action']=corrector_control
    proof['actual_exact_algebraic_ALLq_Hermite_coefficients']=supplier['actual_exact_algebraic_fast_coefficients']
    proof['actual_full_matter_noncommuting_control']=supplier['actual_physical_noncommuting_differential_control']
    desc=dict(law='NEW_N4_FULL5760_LOCAL960_SU4_GAUSS_EXACT_FINITE_CAR_SOURCE',plan='issue-222-induced-photon-delta13-continuation-20261008',
        parameters=dict(N=4,V=64,a='1/64',UEphys=1,u=64,Lambda='2^8192',chi='2^-4096',normal=5760,local_angles=960,total_particles=8324),
        source='E0=.5 sum4 c_active0alpha^dag Ralpha; Ref4->3, full local SU4 singlet, RAWno whitening',
        matter=dict(path=str(matter_path),sha256=hashlib.sha256(raw).hexdigest(),input_sha256=matter['input_sha256']),
        quantum=dict(path=str(quantum_path),sha256=hashlib.sha256(qraw).hexdigest(),input_sha256=supplier['input_sha256']),
        original_micro_config_UNCHANGED=matter['input_descriptor']['original_micro_config_UNCHANGED'],
        exact_NEW_allq_BO_mean=supplier['actual_allq_feedback_mean'],exact_wave=supplier['actual_full_field_wave_ABI'],
        omitted_terms=['U1/Theta/Pi','Coulomb'],original_SOURCE_frozen_separate=True)
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),
        actual_full5760_local960_corrected_quantum_CAR_source_proof=proof,actual_original_full256_Kato_localHaar_source=literal,
        source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(b).hexdigest(),bytes_utf8=b.decode()) for name,b in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==b for name,b in SOURCE_BYTES.items()},
        input_unchanged=dict(matter=matter_path.read_bytes()==raw,quantum=quantum_path.read_bytes()==qraw),
        runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__),elapsed_seconds=time.monotonic()-started,
        adverse_facts=['Pointwise nonuniform occupied density is NOT2; .5 source normalization requires actual fullfield translation integration.',
            'Kinetic Xi includes dG and AGX+XGA; nonuniform slow-linear colour metric action is not small and cannot be put in residual.',
            'Mean includes leading allq fastZPE only; omitted PH/slowPhiZPE forces retained, not true ground or quantum mean.',
            'Full static PH/mixed Gaussian preparation is not a nonlinear/dynamical PH ground state.',
            'No U1/Theta/Pi/Coulomb, originalSOURCE/vacuum/eR/QED/commoncone/scattering/anomaly admission.'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--matter',type=Path,required=True);parser.add_argument('--quantum',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run(args.matter,args.quantum)
    if not all(result['source_unchanged_since_import'].values()) or not all(result['input_unchanged'].values()):raise ArithmeticError('Scientific source/input epoch changed')
    args.output.write_text(json.dumps(result,indent=2)+'\n');p=result['actual_full5760_local960_corrected_quantum_CAR_source_proof']
    print(json.dumps(dict(input_sha256=result['input_sha256'],Rayleigh_residual=p['actual_prepared_Rayleigh_residual_upper'],raw_source_times=p['raw_source_times'],elapsed_seconds=result['elapsed_seconds']),indent=2))

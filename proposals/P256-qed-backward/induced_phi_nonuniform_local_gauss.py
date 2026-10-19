"""NEW N4 allPhi/colour localGauss physical kinetic remainder supplier.

The non-small SLOW-Phi linear fast-colour cometric is explicitly excluded from
this remainder, to be cancelled by the consumer's actual kinetic Hermite term.
It is NOT zero, a frozen metric, or a full source propagation proof by itself.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={Path(__file__).name:Path(__file__).read_bytes()}
sys.set_int_max_str_digits(0)
import induced_phi_colour_uniform_global_gauss as global_chart
SOURCE_BYTES.update(global_chart.SOURCE_BYTES)
phi=global_chart.phi;stationary=global_chart.stationary
import numpy as np


def fourth_upper(x:F)->F:
    return phi.sqrt_upper(phi.sqrt_upper(x))


def gaussian_weighted_bounds(d:int,fast:int,cmin:F,cmax:F,cfmax:F,R:F,p:int,C1:F,C2:F)->dict[str,F]:
    """Exact joint Gaussian A=(1/2)Cov^-1; A Cov A=(1/4)Cov^-1.

    C1/C2 are TOTAL finite-CAR derivative norms. A corrected polynomial packet
    must ADD its own actual derivative prices; this function does not assume P=0.
    """
    if not 0<fast<d or not 0<cmin<=cmax or not 0<cfmax<=cmax or R<=0 or p<6 or min(C1,C2)<0:
        raise ValueError('Actual coupled covariance, fast block and full CAR data required')
    def mu(dim:int,variance:F,k:int)->F:
        value=variance**(k//2)
        for j in range(k//2):value*=dim+2*j
        return value
    r={k:mu(d,cmax,k) for k in (2,4,6,8,12)}
    fastmom={k:mu(fast,cfmax,k) for k in (2,4)}
    x={k:mu(d,1/(4*cmin),k) for k in (2,4,8)}
    Z=1-2*p*r[2]/R**2
    if Z<=F(1,2):raise ArithmeticError('Positive exact compact normalizer not proved; no clipping')
    norm=phi.sqrt_upper(1/Z);Arob=phi.sqrt_upper(F(d))/(2*cmin)
    grad=phi.sqrt_upper(x[2])+2*p/R**2*phi.sqrt_upper(r[2])+C1
    fast_grad=fourth_upper(fastmom[4]*x[4]);r2_grad=fourth_upper(r[8]*x[4])
    fast_hess=fourth_upper(fastmom[4]*x[8])+Arob*phi.sqrt_upper(fastmom[2])
    r2_hess=fourth_upper(r[8]*x[8])+Arob*phi.sqrt_upper(r[4])
    fast_bump_cross=fourth_upper(fastmom[4]*phi.sqrt_upper(r[8]*x[8]))
    fast_bump_hess=2*p*phi.sqrt_upper(F(d))/R**2*phi.sqrt_upper(fastmom[2])+4*p*(p-1)/R**4*fourth_upper(fastmom[4]*r[8])
    fast_hess+=4*p/R**2*fast_bump_cross+fast_bump_hess+2*C1*fast_grad+(C2+C1*C1)*phi.sqrt_upper(fastmom[2])+4*p*C1/R**2*fourth_upper(fastmom[4]*r[4])
    r2_hess+=4*p/R**2*fourth_upper(r[12]*x[4])+2*p*phi.sqrt_upper(F(d))/R**2*phi.sqrt_upper(r[4])+4*p*(p-1)/R**4*phi.sqrt_upper(r[8])+2*C1*r2_grad+(C2+C1*C1)*phi.sqrt_upper(r[4])+4*p*C1/R**2*phi.sqrt_upper(r[6])
    return dict(normalizer_lower=Z,unweighted_gradient_norm_upper=norm*grad,weighted_rfast_Hessian_norm_upper=norm*fast_hess,weighted_r2_Hessian_norm_upper=norm*r2_hess,normal_coordinate_moment2_upper=r[2],fast_coordinate_moment2_upper=fastmom[2],slow_count=F(d-fast),fast_count=F(fast))


def local_gauss_prices(g0_lower:F,gs_lower:F,g0_upper:F,gs_upper:F,V:int,n:int,R:F,gradient:F,Hfast:F,H2:F)->dict[str,F]:
    """Additional local kinetic /chi AFTER actual slow-linear Hermite subtraction.

    Exact canonical real modes: slow45, fast90V-45, localangle15V. All signs of
    removed term are consumer obligations, not an omission allowance.
    """
    if not 0<g0_lower<=g0_upper or not 0<gs_lower<=gs_upper or V!=64 or n<1 or n%4 or R<=0 or min(gradient,Hfast,H2)<0:
        raise ValueError('Actual N4 centre-invariant body/normal-chart inputs required')
    d=90*V;z=105*V;h=min(g0_lower,gs_lower)/2;w=(g0_upper+3*gs_upper)/2
    u=4+10/phi.sqrt_upper(h)
    # sqrt_upper(h) is an upper, so the factor2 guarantees a lower denominator
    # for h in this actual >1/64 finite-root box, while retaining exact rationals.
    if not F(1,64)<h<1 or w>1:raise ArithmeticError('Actual root box outside analytic local frame certificate')
    aD=F(4096);bD=F(2)**24
    reach=h*h/(100*(1+2*w))
    transition=4*512*(1+5/phi.sqrt_upper(h)*2)*R
    if 2*R>reach or R>F(1,1<<20) or u*aD*R>F(1,4) or transition>F(1,4):
        raise ArithmeticError('Local Phi tube, adapted slice or complete6720 chart not certified')
    inv=2*u;Amax=inv*inv;A1=2*inv**3*aD;A2=6*inv**4*aD*aD+2*inv**3*bD
    normal=(phi.sqrt_upper(F(d))*(A1*Hfast+A2*H2/2)+d*(A1+A2*R)*gradient)/2
    qnorm=2*n
    cross=qnorm*(2*Amax*gradient+d*A1)/2
    angular=Amax*n*n*F(15,8)/2
    jg=z*inv*aD;jh=z*(inv*bD+inv*inv*aD*aD)
    geometry=(d*A1*jg+d*Amax*jh)/4+Amax*jg*jg/8
    return dict(local_Phi_orbit_metric_lower=h,local_Phi_tube_radius_upper=reach,adapted_slice_transition_derivative_upper=transition,mean_complete_chart_inverse_norm_upper=u,actual_chart_inverse_norm_upper=inv,normal_cometric_norm_upper=Amax,normal_cometric_first_derivative_norm_upper=A1,normal_cometric_second_derivative_norm_upper=A2,normal_kinetic_remainder_action_divided_chi_upper=normal,local_Noether_cross_action_divided_chi_upper=cross,local_Noether_angular_operator_norm_divided_chi_upper=angular,volume_log_gradient_norm_upper=jg,volume_log_Hessian_norm_upper=jh,geometric_potential_norm_divided_chi_upper=geometry,total_reference_local_Gauss_kinetic_remainder_divided_chi_upper=normal+cross+angular+geometry,physical_normal_dimension=F(d),physical_local_angle_dimension=F(15*V),physical_total_chart_dimension=F(z))


def literal_mean_symbols(native:dict[str,Any],Qcart:np.ndarray)->dict[str,Any]:
    uniform=phi.scalar.uniform;primitive=uniform.supplier.static.primitive
    T=np.asarray(uniform.supplier.static.force.T);B=primitive.clifford_cartesian_basis();generators=B/np.sqrt(8)
    S=np.kron(np.eye(4),np.real(np.einsum('aij,bji->ab',B,T))/4);N=S@Qcart
    comm=1j*(generators[:,None]@native['frame'][None]-native['frame'][None]@generators[:,None])
    cart=np.real(2*np.einsum('mij,anji->anm',T,comm)).reshape(15,60).T;O=S@cart;H=O.T@O
    Hinv=np.linalg.solve(H,np.eye(15));mc=10/3;rows=[];maxerr=0.
    for q in product(range(4),repeat=3):
        d=1-np.exp(2j*np.pi*np.asarray(q)/4);Dgradient=np.kron(d[:,None],np.eye(15))
        D=np.zeros((105,105),complex);D[:60,:45]=N;D[:60,90:]=O;D[60:,45:90]=np.sqrt(mc)*np.eye(45);D[60:,90:]=np.sqrt(mc)*Dgradient
        metric=D.conj().T@D;inverse=np.linalg.solve(metric,np.eye(105));direct=np.linalg.solve(D,np.eye(105))
        expected=float(global_chart.GCOLOUR)*np.eye(45)+Dgradient@Hinv@Dgradient.conj().T
        k=Dgradient@Hinv
        err=max(float(np.max(abs(inverse[:45,:45]-8*np.eye(45)))),float(np.max(abs(inverse[45:90,45:90]-expected))),float(np.max(abs(inverse[45:90,90:]+k))),float(np.max(abs(inverse[90:,90:]-Hinv))))
        maxerr=max(maxerr,err)
        rows.append(dict(q=list(q),qhat_squared=float(np.real(np.vdot(d,d))),all105_Gauss_metric_min=float(np.linalg.eigvalsh(metric)[0]),actual_inverse_norm=float(np.linalg.norm(direct,2)),actual_colour_longitudinal_cometric_max=float(np.linalg.eigvalsh(expected)[-1]),actual_Noether_cross_norm=float(np.linalg.norm(k,2)),GaussSchur_identity_error=err))
    if maxerr>1e-10 or float(np.max(abs(N.T@O)))>1e-12:raise ArithmeticError('Actual all64 local SU4 Gauss Fourier decomposition failed')
    return dict(real_physical_normal_degrees=5760,real_local_angles=960,real_total_degrees=6720,actual_ALL64_complex_q_Gauss_mean_symbols=rows,actual_Hphi_orbit_metric_min=float(np.linalg.eigvalsh(H)[0]),actual_maximum_symbol_error=maxerr,actual_Kinetic_colour='.3chi I +chi D Hphi^-1 Ddag; longitudinal not .3chi',actual_Noether_identity='chi/2[8pphi²+.3pC²+(Q−Ddag pC)dag Hphi^-1(Q−Ddag pC)] at mean; Q is all65 finite-CAR plus Ref4 Noether, not primitiveQstat/interband matrix')


def run(root_path:Path,full_path:Path)->dict[str,Any]:
    start=time.monotonic();raw=root_path.read_bytes();fraw=full_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=global_chart.ROOT_RECEIPT_SHA or hashlib.sha256(fraw).hexdigest()!=global_chart.FULL_RECEIPT_SHA:raise ArithmeticError('Frozen f63 mean/full45 physical prerequisites required')
    root=json.loads(raw);full=json.loads(fraw);params=root['actual_stationary_root']['parameters']
    g0l,g0u,gsl,gsu=[F(params[k][s]) for k,s in (('g0','lower'),('g0','upper'),('gs','lower'),('gs','upper'))]
    mean=stationary.input_mean(dict(microscopic_config=full['input_descriptor']['original_micro_config_UNCHANGED']),[(g0l+g0u)/2,(gsl+gsu)/2]);native=stationary.full.supplier.quantum.build_native(mean,4);Q=np.asarray(full['actual_full45_geometry']['Qcart_display'],float)
    R=F(1,1<<20);slow_variance=F(2)**-6144;fast_variance=F(2)**-7168
    # Genuine DECLARED anisotropic Gaussian trial: not an invented full PH vacuum.
    packet=gaussian_weighted_bounds(5760,5715,fast_variance,slow_variance,fast_variance,R,6,F(0),F(0))
    prices=local_gauss_prices(g0l,gsl,g0u,gsu,64,8324,R,packet['unweighted_gradient_norm_upper'],packet['weighted_rfast_Hessian_norm_upper'],packet['weighted_r2_Hessian_norm_upper'])
    controls=literal_mean_symbols(native,Q);source=phi.scalar.literal_source()
    times=[dict(time=str(t),additional_twoleg_remainder_error_upper=phi.scalar.ratio(8*abs(t)*F(2)**-4096*prices['total_reference_local_Gauss_kinetic_remainder_divided_chi_upper'])) for t in (F(0),F(1,4096),F(4096))]
    descriptor=dict(plan='issue-222-induced-photon-delta13-continuation-20261008',assigned_scientific_revision='d86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7',observed_plan_revision='1c4f98f4e0e84bd598b150f73bb276ac882acd511d6af2de5848b4d15a80417a',root_input=dict(path=str(root_path),sha256=hashlib.sha256(raw).hexdigest()),full45_input=dict(path=str(full_path),sha256=hashlib.sha256(fraw).hexdigest()),exact_mean_parameters=params,N=4,V=64,bulk_species=65,finite_CAR_particles=8324,Ref4=4,Phi_Cart=3840,SU4_links=192,physical_normal_degrees=5760,local_SU4_angles=960,original_colour_electric='.3chi E4²/2; Casimir15/8',actual_compact_radius=str(R),actual_declared_trial_slow_covariance=str(slow_variance),actual_declared_trial_fast_covariance=str(fast_variance),actual_declared_trial_matter='FIXED all65 mean-filled CAR+Ref4; C1=C2=0 by explicit trial, not fallback',scope='NEW actual all64 localGauss symbols and physical kinetic remainder. Trial is not completed propagated source/quantumground; slow-linear-fast cometric term MUST be cancelled by actual consumer Hermite operator.')
    proof=dict(physical_state='Physical fields Phi_x=Ux[PhiSlow+Qfast(PhiSlow)eta_fast,x]Uxdag; links Ux exp(iTC_e) Uy†. Product normalizedHaarSU4 localframes divided by ONE globalZ4; connectednear-identitylink chart forcescentreambiguitiesequal, total nmod4=0 singlevalued. CAR frame transforms withALLlocalUx, giving exact localGauss. E0 co-localc†Refcommutes everylocalQa (x0 explicit finiteCAR, otherx trivial). Full field Hilbert R^(60V) timesSU4^(3V) kept; smallcompact preparation removes no centre wells fromH.',
        actual_mean_symbols=controls,actual_CAR_source=source,actual_packet_bounds={k:phi.scalar.ratio(v) if isinstance(v,F) else v for k,v in packet.items()},actual_exact_rational_kinetic_remainder_prices={k:phi.scalar.ratio(v) for k,v in prices.items()},additional_source_clock_budgets=times,
        linear_subtraction_contract='Adaptedfast Qfast(PhiSlow) is exactCartnormal to actualPhiSlow gaugeorbit. Atfast0, nonzeroqfastPhi-cometric/alignment cross is EXACTLYzero; baselineG_C dependsPhiSlow: .3I+D Hphi(PhiSlow)^-1 Ddag. The linear derivative in45SLOWcoordinates must be retained as chi/2 sum etaSlow,j pC^T dG_j pC and cancelled WITH dK_j by actualHermite Xi=.5(dK−A dG A). This helper bounds ONLY remaining metric action; no permission to omit this O1 term. A2 boundsTaylorremainingr² andA1 boundslinearFASTvariationrfast.',
        derivative_certificate='D atmean is blocktriangular withphysicalPhi60V normal/orbit and45V right-Maurer link rows. Sparsegraphgradientnorm<=sqrt12<4, Hphi>=h, yielding||D0inv||<=4+5/sqrt(h). Actual sqrt_upper(h) <2sqrt(h), so u=4+10/sqrt_upper(h) is a valid rationalupper. Normalfibre projector OCart (OCart^T OCart)^-1 OCart^T is analytic oncomplexslowballrho=1/256: Oprime<=sqrt(2/V), OCart^TOCart>=8h, h>1/64,w<1 implyPnorm<=2 andradialKatoUnorm<=2. Cauchypolydiscs giveQ1<=512,Q2<=2^20,Q3<=2^33. WithR<=2^-20, localPhiCart andlinkcomm/Maurer derivative plusQjet terms give||Dprime||<=4096,||Dsecond||<=2^24. Neumannradius u*aD*R<=1/4 gives||Dinv||<=2u. Consequently ||dA||<=2inv³ aD and||d²A||<=6inv4 aD²+2inv³ bD. Consumer mustuseTHISadaptedframe or proveequal/tighterbounds, notfixedQ0linearfastPhi cross silentlyremoved.',
        global_chart_certificate='LocalCliffordnormalreach=h²/[100(1+2w)]. Adaptedchart is standardindividualnormal-slice plusanalyticcoordinatechange: ||dtransition−I||<=4*512*(1+5/sqrt(h))*R, boundedabovebyrationaltransition. This<1/4 and2R<reach provesinjective. Localcentre z_x areforcedequalbyconnectednear-identitySU4links (distinctcentresseparated>=sqrt2, allsmalllogs<R); only globalZ4 remains. p6 compact givesC5/H4zeroextension/D(H²), no hardcut Hamiltonian.',
        kinetic_price_proof='Exact fullmetric normalA, crossk, angularR fromD inverse are retained. AFTER actualslowlinearsubtraction: ||A−A0−linSlow||<=A1*rfast+A2*r²/2. Normalaction/chi <=[sqrt(d)(A1||rfast HessPsi||+A2||r²HessPsi||/2)+d(A1+A2R)||gradPsi||]/2. k<=inv²,dk<=A1 yieldsNoethercross, angular<=inv² n²CF/2. Qvector<=2n from CF15/8 onentirefixedn finiteCAR includingRef. HaarJ=detD; gradlogJ<=z inv aD,Hess<=z(inv bD+inv² aD²). SymmetricflatJ potential chi/4 div(A gradj)+chi/8 gradj^T A gradj, localHaarunimodularanglesdivzero; no missingcrossvolume term.',
        Gaussian_price_proof='ExactGaussianprecision A=(1/2)Cov^-1, so A Cov A=(1/4)Cov^-1. Cauchy/Holderprice joint rfast moments andAeta moments separately: weightedfastHess scales sqrt(cfmax)/cmin, r²Hess scales cmax/cmin, NEVER a² cmax² fakehugeprecision. Allbumpgradient/Hessian, positiveZ andTOTALCARC1/C2 included. Correctorpolynomialderivatives mustbe addedby actualconsumer; fixedtrial isnot Born fallback.',
        comparator='Additional twoleg sourceerror<=8|t|chi total onceconsumer provesits actualcorrectedBorn/bumpreferencepacket gradient/Hfast/H2 andactualslowlinearHermite cancellation. BODYHref mustbe physicallyconjugatedbyCARW andalllocalUx withsqrtJ. SamephysicalH bothlegs, physicalE0endpointspriced. No EMTheta/Pi/Coul/originalSOURCE/QED/commoncone admission.')
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_nonuniform_local_Gauss_kinetic_supplier=proof,source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},inputs_unchanged=root_path.read_bytes()==raw and full_path.read_bytes()==fraw,runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root-input',type=Path,required=True);parser.add_argument('--full45',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result=run(args.root_input,args.full45)
    if not all(result['source_unchanged_since_import'].values()) or not result['inputs_unchanged']:raise ArithmeticError('Scientific source/input epoch changed during execution')
    args.output.write_text(json.dumps(result,indent=2)+'\n');p=result['actual_nonuniform_local_Gauss_kinetic_supplier'];print(json.dumps(dict(input_sha256=result['input_sha256'],symbol_error=p['actual_mean_symbols']['actual_maximum_symbol_error'],times=p['additional_source_clock_budgets'],elapsed_seconds=result['elapsed_seconds']),indent=2))

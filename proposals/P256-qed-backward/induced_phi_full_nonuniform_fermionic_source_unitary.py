"""N4 all5760 Phi/colour, local960-SU4, corrected finite-CAR source.

The exact wave is a matrix functional of the microscopic pencil, not of the
floating displays.  This program prices its physical nonlinear propagation
against a field-identity comparator; no Hamiltonian is restricted to the tube.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import importlib.abc
import importlib.machinery
import json
from itertools import product
from pathlib import Path
import sys
import time
from typing import Any

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES:dict[str,bytes]={Path(__file__).name:Path(__file__).read_bytes()}

class CapturedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self,fullname:str)->Any:
        raw=Path(self.path).read_bytes();SOURCE_BYTES[Path(self.path).name]=raw
        return self.source_to_code(raw,self.path)

class CapturedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname:str,path:Any=None,target:Any=None)->Any:
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is not None and spec.origin is not None:
            source=Path(spec.origin)
            if source.parent==ROOT and source.suffix=='.py':
                spec.loader=CapturedLoader(fullname,str(source));return spec
        return None

finder=CapturedFinder();sys.meta_path.insert(0,finder)
import induced_phi_nonuniform_field_car as field
import induced_phi_nonuniform_gauss_quantum as quantum
import induced_phi_nonuniform_local_gauss as local
import induced_phi_full_uniform_colour_fermionic_source_unitary as uniform_body
sys.meta_path.remove(finder)
SOURCE_BYTES.update(field.CAPTURED);SOURCE_BYTES.update(quantum.SOURCE_BYTES)
SOURCE_BYTES.update(local.SOURCE_BYTES);SOURCE_BYTES.update(uniform_body.SOURCE_BYTES)
import ijson
from ijson.common import ObjectBuilder
import mpmath as mp
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

joined=uniform_body.joined
iv:Any=joined.iv
exact,cap,pack,endpoint,interval=joined.exact,joined.cap,joined.pack,joined.endpoint,joined.interval
FIELD_SHA='55661707f096422947a816924664de94e8da1d092e278164e4e9be57a26e775c'
QUANTUM_SHA='2bd70176910108fd695c5490b31ce490da70c860a9a43693758479ec2ccc2c06'
LOCAL_SHA='73ca8d12378d4d5ef6371e11270f967d891f6be0a7e85f200c16440736bf72b4'
UNIFORM_BODY_SHA='8bfeec2518978d9a33e300a342c842755e3f766a5f4c83b4847b7dfda297f91f'
FIELD_RECEIPT_SHA='b20c715c5127d24e29d9d5db0ac3f6e2b8d282605fb01bbee89885e4016849f8'
R=F(1,1<<20)
D=5760
FAST=5715


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def receipt(path:Path,keys:set[str],spectral:bool=False)->dict[str,Any]:
    """Read complete selected JSON objects without materializing 2GB PH displays."""
    result:dict[str,Any]={};builder=None;selected='';depth=0;rows=[];qi=None
    with path.open('rb') as stream:
        for prefix,event,value in ijson.parse(stream,use_float=True):
            if builder is not None:
                builder.event(event,value)
                if event in ('start_map','start_array'):depth+=1
                if event in ('end_map','end_array'):depth-=1
                if depth==0:
                    if selected=='spectral':rows.append(dict(q_integer=qi,actual_static_spectral_supplier=builder.value))
                    else:result[selected]=builder.value
                    builder=None
                continue
            if prefix in keys and event in ('start_map','start_array'):
                selected=str(prefix);builder=ObjectBuilder();depth=1;builder.event(event,value)
            elif prefix in keys and event not in ('map_key','end_map','end_array'):
                result[prefix]=value
            elif spectral and prefix=='actual_ALL64_q.item.q_integer' and event=='start_array':
                selected='qi';builder=ObjectBuilder();depth=1;builder.event(event,value)
            elif spectral and prefix=='actual_ALL64_q.item.actual_static_spectral_supplier' and event=='start_map':
                qi=result.pop('qi');selected='spectral';builder=ObjectBuilder();depth=1;builder.event(event,value)
    if spectral:result['actual_ALL64_q']=rows
    if keys-set(result):raise ArithmeticError(f'Missing complete supplier objects: {keys-set(result)}')
    return result


def upper_fraction(x:Any)->F:
    return endpoint(x,1)


def radial(trace:Any,degree:int,p:int=2)->Any:
    """Gaussian radial Wick bound using covariance TRACE, not isotropic cmax."""
    n=degree*p//2;factor=1
    for j in range(n):factor*=2*j+1
    return cap((exact(factor)*trace**n)**exact(F(1,p)))


def total_gap_kato(data:dict[str,Any])->dict[str,Any]:
    """Whole5760 directional bounds from native row sums; no band degeneracy poles."""
    g0,gs=data['g'][0],data['g'][1];rows=[];C1=exact(0);C2=exact(0);active=None
    for profile in data['profiles']:
        bound=field.exact_operator_bounds(profile,data['g'])
        Emin=exact(F(bound['minimum_gap_unscaled']['lower'])/2)
        Dp=(abs(exact(profile['onsite']))+2*sum(abs(exact(p['c0']))+abs(exact(p['ci'])) for p in profile['paths']))/iv.sqrt(exact(2))
        Dc=exact(0);pc=exact(0);cc=exact(0);pcc=exact(0);ccc=exact(0)
        for path in profile['paths']:
            L=exact(path['length']);c0=abs(exact(path['c0']));ci=abs(exact(path['ci']))
            mean=c0*iv.sqrt(g0)+ci*iv.sqrt(gs);dv=(c0+ci)/iv.sqrt(exact(2));ell=L/iv.sqrt(exact(2))
            Dc+=2*ell*mean;pc+=2*ell*dv;cc+=ell**2*mean;pcc+=ell**2*dv;ccc+=ell**3*mean/3
        D1=cap(Dp+Dc+2*pc*exact(R));H2=cap(2*pc+2*cc+4*pcc*exact(R));delta=Emin/2
        if endpoint(3*exact(R)*D1,1)>=endpoint(Emin/4,0):raise ArithmeticError('Actual all256 nonlinear gap tube failed')
        first=cap(128*20*D1/delta);second=cap(128*(400*(D1/delta)**2+80*H2/delta))
        C1+=profile['load']*iv.sqrt(exact(D))*first
        C2+=profile['load']*exact(D)*second
        row=dict(role=profile['role'],load=profile['load'],physical_scale=profile['scale'],occupied_global_rank=128,
                 distance_to_zero_in_3R_lower=pack(delta),whole5760_h_first_norm_upper=pack(D1),whole5760_h_second_norm_upper=pack(H2),
                 CAR_directional_first_upper=pack(first),CAR_directional_second_upper=pack(second),
                 Dphi=Dp,Dcolour=Dc,PC=pc,CC=cc,PCC=pcc,CCC=ccc,delta=delta)
        rows.append(row)
        if profile['role']=='active filled charged':active=row
    if active is None:raise ArithmeticError('Physical active source profile absent')
    return dict(rows=rows,C1=cap(C1),C2=cap(C2),active=active,
                proof='Riesz negative PROJECTOR P(theta), K=[dotP,P], dotW=KW; denominators only occupied/unoccupied gap. Second-quantized W on occupied128 percopy, all65 copies. Whole5760 unit-direction native endpoints and link matrices bounded by Frobenius/row norm; total gradient sqrt5760 and Hessian5760 included explicitly.')


def corrected_derivatives(wave:dict[str,Any],kato:dict[str,Any])->dict[str,Any]:
    """Differentiate (1-P)b6 f_A, including polynomial, CAR and radial bump."""
    m=wave['actual_joint_Gaussian_corrector_moments']
    value=lambda key:interval(m[key])
    mall=value('physical_trace_ALL_covariance_upper');mf=value('physical_trace_fast_covariance_upper')
    cmin=value('physical_fast_covariance_eigen_lower');cf=value('physical_fast_covariance_eigen_upper')
    Atr=value('physical_trace_A_upper');A2tr=value('physical_trace_A_squared_upper')
    Y=value('actual_allq_Wick_colour_Y_squared_trace_upper');P2=value('corrector_preparation_norm_upper')
    P4=cap(iv.sqrt(exact(27))*P2);P8=cap(iv.sqrt(exact(343))*P2)
    # Derivatives of a centred cubic in physical covariance-normalized fast variables.
    # Full conditional covariance and full-matter Xi coefficient perturbations are
    # enclosed by factor4. The actual coefficient law is not assigned its bound.
    slow=cap(mall);Yroot=iv.sqrt(Y)
    gradP=cap(16*Yroot*(1+iv.sqrt(slow/cmin)))
    hessP=cap(32*Yroot*(1/iv.sqrt(cmin)+iv.sqrt(slow)/cmin))
    grad0=value('Gaussian_gradient_norm_upper');hf0=value('Gaussian_rfast_Hessian_norm_upper');h20=value('Gaussian_rALL_squared_Hessian_norm_upper')
    rp=radial(mall,1);rf=radial(mf,1);r2=radial(mall,2)
    bumpgrad=cap(12*rp/exact(R*R))
    bumphess=cap(12*iv.sqrt(exact(D))/exact(R*R)+120*r2/exact(R**4))
    Z=cap((1-P2-iv.sqrt(12*mall/exact(R*R)))**2)
    if endpoint(Z,0)<=F(1,2) or endpoint(P8,1)>=F(1,100):raise ArithmeticError('Actual corrected compact normalization failed')
    norm=cap(1/iv.sqrt(Z));C1,C2=kato['C1'],kato['C2']
    grad=cap(norm*((1+iv.sqrt(exact(3))*P4)*grad0+gradP+(1+P4)*bumpgrad+C1*(1+P2)))
    # Hypercontractivity bounds tensor-valued chaos degree<=3 and<=4; this
    # preserves tight Wick Hfast/H2 instead of losing 2^1024 to cmax/cmin.
    gfast=cap(iv.sqrt(exact(3))*rf*grad0);gall2=cap(iv.sqrt(exact(15))*r2*grad0)
    HF=cap(norm*((1+iv.sqrt(exact(27))*P4)*hf0+iv.sqrt(exact(27))*rf*hessP
                 +2*iv.sqrt(exact(27))*gfast*gradP+2*bumpgrad*gfast+(bumphess+C2+C1*C1)*rf*(1+P4)
                 +2*C1*(gfast*(1+P4)+rf*gradP+rf*bumpgrad)))
    H2=cap(norm*((1+9*P4)*h20+9*r2*hessP+18*gall2*gradP+2*bumpgrad*gall2
                 +(bumphess+C2+C1*C1)*r2*(1+P4)+2*C1*(gall2*(1+P4)+r2*gradP+r2*bumpgrad)))
    return dict(mall=mall,mfast=mf,cfastmin=cmin,cfastmax=cf,Atrace=Atr,A2trace=A2tr,Y=Y,P2=P2,P4=P4,P8=P8,
                gradientP=gradP,HessianP=hessP,normalizer=Z,normalization=norm,gradient=grad,Hfast=HF,H2=H2,
                gradient0=grad0,Hfast0=hf0,H20=h20,bumpgradient=bumpgrad,bumpHessian=bumphess)


def exact_wave_definition(wave:dict[str,Any],matter:dict[str,Any])->dict[str,Any]:
    """The exact coefficient inputs are captured differential operators, not displays."""
    qs={tuple(r['q_integer']) for r in matter['actual_ALL64_q']}
    if qs!=set(quantum.QINTEGER):raise ArithmeticError('Full64 wave pencil missing')
    return dict(wave_ABI=wave['actual_full_field_wave_ABI'],ALL64_pencil_bytes=matter['actual_ALL64_q'],
        coefficient_authority='Exact binary native path coefficients, AP phases Q(sqrt2,i), exact Clifford traces, implicit root and finite-CAR negative projectors define all matrix entries. Listed complex arrays are observed displays with certified perturbation controls; never promoted to exact coefficients.',
        A='For each exact complex90 pencil A=G^-1/2 sqrt(G^1/2 K G^1/2) G^-1/2; canonical q/-q realification gives one full5760 positive matrix. ALL PH/contact/mixed blocks retained.',
        P='At fixed PhiSlow use adapted fast normal frame polar(Pnormal(PhiSlow)Q0). Conditional fast colour xi=C+A_CC^-1 A_CS PhiSlow from the EXACT full matrix A; centred P=sum_i PhiSlow_i sum_allq [xi(q)^dag X0_i(q)xi(q)-Tr(Cov0(q)X0_i(q))]. X0 is the exact primitive allq kinetic-Xi algebraic coefficient at the NEW mean. Its full-matter/conditional action difference is explicitly priced, not claimed zero. P is an actual approximate preparation, not falsely identified with a full-matter resolvent.',
        preparation='Z^-1/2 b6(1-P)f_A, R=2^-20 in all5760 real coordinates. Translation acts orthogonally on the canonical real pairs; P sums every physical q and is translation invariant.',
        cancellation='Fast H action on centred quadratic is xi^T(AGX+XGA)xi-Tr(GX). Thus full slow-linear colour potential AND pC dG pC combine into Xi; trace is .25Tr(A dG+A^-1 dK). Not wrong GA ordering and not potential-only Hermite.',
        kinetic_frame='Adapted Q(PhiSlow) is exactly orthonormal and orthogonal to actual PhiSlow orbit. It kills bare fastPhi/colour cross and slow-linear fastPhi gradient stiffness. Remaining derivative frame kinetic terms are priced with actual corrected weighted derivatives.',
        actual_exact_algebraic_P_coefficients=wave['actual_exact_algebraic_fast_coefficients'],
        actual_ALL64_P_differential_control=wave['actual_allq_kinetic_fast_corrector'],
        real_pairs=wave['actual_canonical_real_quantization'])


def packet_prices(data:dict[str,Any],wave:dict[str,Any],matter:dict[str,Any],kato:dict[str,Any])->dict[str,Any]:
    chi=exact(F(2)**-4096);Lambda=exact(F(2)**8192);Lambda54=exact(F(2)**10240)
    p=corrected_derivatives(wave,kato);mall,mfast=p['mall'],p['mfast'];norm=p['normalization'];P4=p['P4']
    gmax=joined.maximum(data['g']);gmin=exact(min(endpoint(g,0) for g in data['g']))
    slowvar=cap(mall);rp3=radial(mall,3,4);rp4=radial(mall,4,4);rf3=radial(mfast,3,4);rf4=radial(mfast,4,4)
    # Native Gram polynomial and exponential path remainders. Adapted normal
    # coordinates remove slow*fastPhi^2; slow*colour^2 is subtracted by Xi.
    # 4096=V/a dominates Gram cubic/quartic contractions after unitary Fourier
    # and ||deltaPhi||HS=||etaPhi||/sqrt2. For .2/8 alignment and .7 plaquette,
    # 192 edges/faces, <=4 ordered factors, trace bound4 and exponential
    # derivatives ||T.c||op<=||c||/sqrt2 give cubic/quartic coefficient<2^20.
    # Slow-frame polar derivatives <=2,10 are absorbed in that same bound.
    primitive=cap(Lambda*(4096*(1+iv.sqrt(gmax))*rp3+4096*rp4))
    primitive_fast=cap(Lambda54*(2**20*(1+gmax)*rf3+2**20*radial(mall,2,8)*radial(mfast,2,8)+2**20*rf4))
    sea=exact(0);charged=exact(0)
    for row in kato['rows']:
        delta=row['delta'];D1=interval(row['whole5760_h_first_norm_upper']);Hh=interval(row['whole5760_h_second_norm_upper'])
        third=cap(6*(row['PCC']+row['CCC'])+12*D1*Hh/delta+24*D1**3/delta**2)
        scale=Lambda if row['physical_scale']=='heavy' else exact(1)
        sea+=128*row['load']*scale*third*rp3
        if row is kato['active']:charged=cap(128*D1*radial(mall,1,4)*(1+P4)*norm)
    nonlinear=cap(norm*(1+iv.sqrt(exact(27))*P4)*(primitive+primitive_fast+sea))
    budget=wave['actual_full_field_covariance']['remainder_force_budgets']
    rootforces=cap(norm*4*sum(interval(budget[k]) for k in ('matter_fast_force_times_slow_std_upper','slowPhi_ZPE_force_times_slow_std_upper')))
    coefficient=exact(F(wave['actual_full_field_covariance']['actual_outward_force_bound_inputs']['coefficient']))
    rootforces=cap(rootforces+norm*Lambda*coefficient*interval(data['root']['root_radius'])*radial(mall,1,4))
    # Full-matter Xi, conditional slow-fast mixing, and slow derivatives of P.
    # Spectral perturbation gives 2^-1024 relative error in full q0 pencil,
    # 2^-2048 for nonzero q; bounding all by former is conservative.
    hermite_difference=cap(norm*coefficient*exact(F(2)**-1024)*(1+p['P4']))
    # P is linear in slow coordinates before the full conditional shift.
    # Its SLOW derivative is O(sqrt(Y)), not its enormous FAST derivative.
    # Conditional M=O(2^-1024) contributes a separately bounded slow Hessian.
    slow_action=cap(norm*coefficient*exact(F(2)**-1024)*(1+iv.sqrt(p['Y'])))
    # Fast-operator corrector product: order Lambda^3/8 P * slow coordinate.
    corrector_product=cap(norm*coefficient*exact(F(2)**3072)*p['P4']*radial(slowvar,1,4))
    Gmax=cap(exact(F(3,10))+12/(gmin/2))
    bump=cap(norm*chi*Gmax*(p['bumpHessian']+2*p['bumpgradient']*p['gradient0'])*(1+P4))
    kato_action=cap(chi*Gmax*(kato['C1']*p['gradient']+(kato['C2']+kato['C1']**2)*(1+P4)/2))
    # Retain slow derivative of adapted fast frame, scaled by chi/sigmaSlow.
    frame=cap(chi*coefficient*(p['gradient']+p['H2']+p['Hfast'])+coefficient*exact(F(2)**-1024)*(1+P4))
    params=data['root']['parameters']
    metrics=local.local_gauss_prices(F(params['g0']['lower']),F(params['gs']['lower']),F(params['g0']['upper']),F(params['gs']['upper']),64,8324,R,
                                    upper_fraction(p['gradient']),upper_fraction(p['Hfast']),upper_fraction(p['H2']))
    kinetic=exact(F(2)**-4096*metrics['total_reference_local_Gauss_kinetic_remainder_divided_chi_upper'])
    canonical=cap(nonlinear+rootforces+hermite_difference+slow_action+corrector_product+bump+kato_action+frame)
    residual=cap(canonical+kinetic)
    endpoint_price=cap(norm*4*interval(kato['active']['CAR_directional_first_upper'])*radial(mall,1,4)*(1+P4))
    points=joined.bloch_points(data['profiles'],data['g'][0],data['g'][1])
    times=[]
    for t in (F(0),F(1,4096),F(4096)):
        re=sum(iv.cos(row['E']*exact(t)) for row in points[1])/128
        im=-sum(iv.sin(row['E']*exact(t)) for row in points[1])/128
        base=cap(4*endpoint_price+exact(abs(t))*(4*canonical+4*charged))
        additional=cap(exact(8*abs(t))*kinetic);error=cap(base+additional)
        if not t:error=exact(0)
        modulus=iv.sqrt(re*re+im*im);lower=max(F(0),endpoint(modulus-error,0))
        times.append(dict(time=str(t),raw_reference_real=pack(re),raw_reference_imag=pack(im),same_physical_H_complex_error_upper=pack(error),
                          local_Gauss_twoleg_error_upper=pack(additional),nonzero_modulus_lower=pack(exact(lower)),
                          raw_actual_source_real=pack(joined.enclosure(endpoint(re,0)-endpoint(error,1),endpoint(re,1)+endpoint(error,1))),
                          raw_actual_source_imag=pack(joined.enclosure(endpoint(im,0)-endpoint(error,1),endpoint(im,1)+endpoint(error,1))),
                          raw_compound_probability=pack(joined.enclosure(lower**2,min(F(1,2),endpoint((modulus+error)**2,1)))),
                          vacuum_leg_norm_error_upper=pack(cap(exact(abs(t))*residual)),source_leg_norm_error_upper=pack(cap(endpoint_price+exact(abs(t))*(residual+charged)))))
    if F(times[-1]['nonzero_modulus_lower']['lower'])<=0:raise ArithmeticError('Full nonuniform raw physical source4096 not certified')
    centre=Lambda*4096*(data['g'][0]**2+3*data['g'][1]**2)
    occupied=sum(-2*profile['load']*(Lambda if profile['scale']=='heavy' else exact(1))*sum(r['E'] for r in rows) for profile,rows in zip(data['profiles'],points,strict=True))
    fastZPE=interval(data['root']['fast_ZPE_divided_Lambda3over8'])*exact(F(2)**3072)
    # Min/max of trace sqrt(G^1/2 K G^1/2), retaining all PH terms via
    # outward perturbation. Exact Efield itself, not this interval, enters Href.
    energy_error=cap(coefficient*exact(F(2)**2048))
    field_energy=fastZPE+joined.enclosure(F(0),endpoint(energy_error,1))
    energy=centre+occupied+field_energy
    price_keys=('normalizer','P2','P4','P8','gradientP','HessianP','gradient','Hfast','H2','Hfast0','H20','mall','mfast','Atrace','A2trace')
    return dict(radius_exact=str(R),normal_dimensions=D,fast_dimensions=FAST,local_angles=960,physical_total_chart_dimension=6720,
                actual_corrected_body_derivatives={k:pack(p[k]) for k in price_keys},
                canonical_remainder_prices=dict(primitive_and_fermionic_nonlinear=pack(nonlinear),root_and_omitted_ZPE=pack(rootforces),full_matter_Hermite_difference=pack(hermite_difference),slow_corrector_action=pack(slow_action),fast_cubic_times_corrector=pack(corrector_product),bump=pack(bump),CAR_Kato=pack(kato_action),adapted_frame_kinetic=pack(frame)),
                total_CAR_gradient_upper=pack(kato['C1']),total_CAR_Hessian_upper=pack(kato['C2']),
                actual_all65_gap_and_Kato_prices=[{k:v for k,v in row.items() if k not in ('Dphi','Dcolour','PC','CC','PCC','CCC','delta')} for row in kato['rows']],
                actual_local_Gauss_metric_prices={k:pack(exact(v)) for k,v in metrics.items()},
                canonical_reference_residual_upper=pack(canonical),local_Gauss_reference_residual_upper=pack(kinetic),
                actual_full5760_reference_residual_upper=pack(residual),actual_prepared_Rayleigh_residual_upper=pack(2*residual),
                raw_primitive_centre_energy=pack(centre),raw_all65_occupied_centre_energy=pack(occupied),raw_exact_field_energy_enclosure=pack(field_energy),
                raw_same_H_reference_energy_enclosure=pack(energy),actual_prepared_Rayleigh_energy_enclosure=pack(energy+joined.enclosure(-endpoint(residual,1),endpoint(residual,1))),
                raw_source_times=times,
                Href='Eexact I_field+dGamma(hmean), Eexact=primitive centre+exact full5760 matrix-functional ZPE. dGamma retains all65 filled sea, and staticRef. Physical comparator rho(alllocal Ux) W Href Wdag rho(alllocal Ux)dag plus sqrtJ unitary measure. Both legs use the SAME raw Friedrichs H; common sea/primitive/allfast phases cancel only in paired inner product.',
                endpoints='Etilde=Wdag E0 W; replace by native E0 only at priced endpoints, never directly evolve a Kato Slater by hmean. Source addition physicalCARactive and Ref4->3; source norm<=2.',
                probability='BACKGROUND corrected Gaussian/bump/P/Kato/Haar measure is translation invariant. Global active occupied rank128 gives sum_x meanTrPxx=128, hence meanTrP00=2 and physical co-local CAR source p=(4-2)/4=1/2 EXACT. Pointwise TrP00 is NOT constrained to2.',
                domain='Whole R3840 Phi x SU4^192, all local Haar frames divided ONE global Z4. n8324 centre invariant. Compact b6 C5/H4, polynomial corrector, smooth gap-ball projector/Kato and nonvanishing local Jacobian put Psi and E0Psi in D(H²). Friedrichs quadratic form is semibounded: finite-CAR h affine Phi, confining Gram quartic, nonnegative alignment/plaquette and elliptic kinetic. No centre-well removal, no hardwall Hamiltonian, no U1 Gauss. Smooth comparator extension by principal-log Kato and C6 collar on2R..3R; unrestricted physical H evolves outside packet.',
                slow_linear_exclusion='Actual kinetic Xi=.5(dK-A dG A) is applied before local73 remainder. Excluding slow-linear cometric from local bound is conditional on this subtraction, not an assertion that pC dG pC is small or zero.')


def literal_car_density(projector:np.ndarray)->dict[str,Any]:
    """Eight-mode physical CAR; pointwise density is allowed to be nonuniform."""
    scalar=joined.scalar
    a=[scalar.car_annihilator(i,8) for i in range(8)]
    E=sum(a[i].conj().T@a[4+i]/2 for i in range(4))
    occupations,U=np.linalg.eigh(projector)
    if occupations.min() < -1e-12 or occupations.max() > 1+1e-12:raise ArithmeticError('Local reduced CAR density invalid')
    weights=np.asarray([np.prod([occupations[i] if state&(1<<i) else 1-occupations[i] for i in range(4)]) for state in range(16)])
    wedge=scalar.exterior_unitary(U);localrho=(wedge*weights)@wedge.conj().T
    rho=np.zeros((256,256),complex);rho[240:,240:]=localrho
    out=E@rho@E.conj().T;probability=float(np.trace(out).real)
    expected=float((4-np.trace(projector).real)/4)
    wrong=[i for i in range(256) if ((i>>4)&15).bit_count()!=3]
    refweight=float(np.trace(out[np.ix_(wrong,wrong)]).real)
    if abs(probability-expected)>2e-12 or abs(refweight)>1e-14:raise ArithmeticError('Physical CAR/Ref4->3 contraction failed')
    return dict(physical_E_norm_squared=probability,exact_CAR_trace_formula_observed=expected,
                probability_identity_error=abs(probability-expected),reference_holes_outside_Ref3_weight=refweight,
                reduced_occupation_eigenvalues=occupations.tolist(),source='RAW .5 sum cdag R; no whitening; no pointwise .5 assumption')


def literal_allq_Hermite(wave:dict[str,Any])->dict[str,Any]:
    """Direct Hessian(Pf), independently of the Sylvester residual contraction."""
    rows=wave['actual_allq_kinetic_fast_corrector']['all64q'];rng=np.random.default_rng(2226720)
    observations=[];maximum=0.
    for row in rows:
        q=tuple(row['q']);selfinverse=all(n in (0,2) for n in q)
        d=np.asarray(rng.normal(size=45));d/=np.linalg.norm(d)
        A=quantum.decode(row['A'])
        gcontrol=wave['actual_local_Gauss_adapted_controls']['qrows'][quantum.QINTEGER.index(q)]
        G=quantum.decode(gcontrol['Gcolour']);K=quantum.decode(gcontrol['Kcolour'])
        X=np.einsum('i,iab->ab',d,quantum.decode(row['Hermite2']));Xi=np.einsum('i,iab->ab',d,quantum.decode(row['Xi']))
        def realify(M:np.ndarray)->np.ndarray:
            return M.real if selfinverse else np.block([[M.real,-M.imag],[M.imag,M.real]])
        A,G,K,X,Xi=map(realify,(A,G,K,X,Xi));y=rng.normal(size=len(A));Ay=A@y
        cov=np.linalg.inv(A)/2;P=float(y@X@y-np.trace(cov@X));gradientP=2*X@y
        Hessian=2*X-np.outer(gradientP,Ay)-np.outer(Ay,gradientP)+P*(np.outer(Ay,Ay)-A)
        lhs=-np.trace(G@Hessian)/2+(y@K@y-np.trace(G@A))*P/2
        rhs=y@Xi@y-np.trace(cov@Xi)
        error=float(abs(lhs-rhs));maximum=max(maximum,error)
        observations.append(dict(q=list(q),real_dimension=len(A),direct_Hessian_Pf_error=error))
    if len(rows)!=64 or maximum>1e-7:raise ArithmeticError('Actual all64 direct Hermite kinetic action failed')
    return dict(all64_observations=observations,maximum_direct_differential_error=maximum,
                scope='Actual leading allq kinetic-Xi polynomial product rule; full-matter difference independently priced. Canonical q/-q represented once in state, duplicate partner here is only independent control.')


def literal_nonlinear256(data:dict[str,Any])->dict[str,Any]:
    native,geom=data['native'],data['geom'];T=geom['M']/np.sqrt(8)
    rng=np.random.default_rng(2225760);eta=rng.normal(size=(64,90));eta*=.8*float(R)/np.linalg.norm(eta)
    sites=np.asarray(list(product(range(4),repeat=3)))
    # Real site coordinates are the inverse unitary Fourier representation of
    # all64 canonical modes; this sample deliberately breaks translation.
    def configuration(t:float)->tuple[np.ndarray,np.ndarray]:
        e=t*eta;slow=e[:,:45].mean(axis=0)
        slowframe=native['frame']+np.einsum('a,amij->mij',slow,geom['deltas'])
        Qa,_,_=quantum.adapted_frame(slowframe,T,geom['Q'])
        fields=slowframe[None]+np.einsum('xa,mba,bij->xmij',e[:,:45]-slow,Qa.reshape(4,15,45),T)
        links=np.asarray([expm(1j*np.einsum('a,aij->ij',e[x,45:].reshape(3,15)[axis],T)) for x in range(64) for axis in range(3)])
        return fields,links
    def projector(h:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        w,u=np.linalg.eigh(h);neg=w<0
        if np.count_nonzero(neg)!=128:raise ArithmeticError('Actual global128 occupied rank failure')
        return (u[:,neg]@u[:,neg].conj().T,w,u)
    result=[];densities=[];car=[]
    for profile in data['profiles']:
        fields0,links0=configuration(0);h0=field.assemble(profile,native,fields0,links0);P0,_,_=projector(h0)
        fields1,links1=configuration(1);h1=field.assemble(profile,native,fields1,links1);P1,w,_=projector(h1)
        # Independently integrate the noncommuting radial projector connection.
        # Unlike an arbitrary endpoint polar intertwiner this preserves the
        # actual Kato parallel-transport convention.
        def lift(steps:int)->np.ndarray:
            W=np.eye(256,dtype=complex)
            for j in range(steps):
                t=(j+.5)/steps;dt=2**-15
                f,l=configuration(t);h=field.assemble(profile,native,f,l)
                fp,lp=configuration(t+dt);fm,lm=configuration(t-dt)
                dh=(field.assemble(profile,native,fp,lp)-field.assemble(profile,native,fm,lm))/(2*dt)
                vals,U=np.linalg.eigh(h);occ=vals<0;jet=U.conj().T@dh@U
                connection=np.zeros_like(jet);a=np.flatnonzero(occ);b=np.flatnonzero(~occ)
                # dotP_uo=h'_uo/(Eocc-Eunocc), [dotP,P]_uo=dotP_uo.
                block=jet[np.ix_(b,a)]/(vals[a][None,:]-vals[b][:,None])
                connection[np.ix_(b,a)]=block;connection[np.ix_(a,b)]=-block.conj().T
                W=(U@expm(connection/steps)@U.conj().T)@W
            return W
        W=lift(4)
        errors=dict(unitarity=float(np.linalg.norm(W.conj().T@W-np.eye(256))),projector_transport=float(np.linalg.norm(W@P0@W.conj().T-P1)))
        if max(errors.values())>2e-8:raise ArithmeticError(f'Actual full256 projector Kato lift failed {errors}')
        result.append(dict(role=profile['role'],load=profile['load'],occupied_global_rank=128,minimum_gap=float(w[128]-w[127]),radial_projector_Kato=errors,no_within_band_denominators=True))
        if profile['role']=='active filled charged':
            density=np.real(np.trace(P1.reshape(64,4,64,4),axis1=1,axis2=3).diagonal())
            # Direct literal lattice translation, not a rank2 Bloch argument.
            translate=np.asarray([np.ravel_multi_index(tuple((x+np.array([1,0,0]))%4),(4,4,4)) for x in sites])
            fe,le=configuration(1);ft=fe[translate];lt=le.reshape(64,3,4,4)[translate].reshape(192,4,4)
            Pt,_,_=projector(field.assemble(profile,native,ft,lt));dt=np.asarray([np.trace(Pt[4*x:4*x+4,4*x:4*x+4]).real for x in range(64)])
            error=float(np.max(abs(dt-density[translate])))
            if error>2e-10:raise ArithmeticError('Literal translation density failed')
            densities.append(dict(density=density.tolist(),pointwise_deviation_from2=float(np.max(abs(density-2))),sum=float(sum(density)),translation_error=error,translation_average_P00=float(np.mean(density))))
            car.append(literal_car_density(P1[:4,:4]))
    return dict(all65_global_projector_lifts=result,actual_nonuniform_density_translation=densities,physical_CAR_Ref4_source=car,
                representation='Psi(Ux,eta)=rho_CAR(all64localUx) W_Kato(eta) [(1-P)b6 f_A]/sqrt(J). rho uses all65 species and staticRef4; co-local E0 SU4 singlet, derivative localGauss exactly cancels local Noether. Full product Haar SU4^64/ONE globalZ4 with native192links; not independent link Haar replacement.')


def run(wave_path:Path,matter_path:Path,root_path:Path,full_path:Path)->dict[str,Any]:
    started=time.monotonic()
    for module,sha in ((field,FIELD_SHA),(quantum,QUANTUM_SHA),(local,LOCAL_SHA),(uniform_body,UNIFORM_BODY_SHA)):
        if module.__file__ is None:raise ArithmeticError('Scientific module source path missing')
        if hashlib.sha256(SOURCE_BYTES[Path(module.__file__).name]).hexdigest()!=sha:raise ArithmeticError('Frozen scientific module epoch required')
    inputs={str(p):digest(p) for p in (wave_path,matter_path,root_path,full_path)}
    if inputs[str(matter_path)]!=FIELD_RECEIPT_SHA:raise ArithmeticError('Frozen completed actualALL64 receipt required')
    if inputs[str(wave_path)]!='0f143f66163a483cc055961888b4cc87bc96ebe23675b802a740252979acec4a':raise ArithmeticError('Frozen completed allq quantum receipt required')
    wave=json.loads(wave_path.read_bytes())
    matter=receipt(matter_path,{'input_descriptor','input_sha256','exact_ALLq_operator_bounds','source_at_import','source_unchanged_since_import'},True)
    if not all(matter['source_unchanged_since_import'].values()) or not all(wave['source_unchanged_since_import'].values()):raise ArithmeticError('Incomplete source captures')
    for source in (wave,matter):
        for name,entry in source['source_at_import'].items():
            raw=entry['bytes_utf8'].encode()
            if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ArithmeticError('Captured supplier byte hash mismatch')
            if name in SOURCE_BYTES and SOURCE_BYTES[name]!=raw:raise ArithmeticError('Supplier/import source epoch mismatch')
            SOURCE_BYTES[name]=raw
    data=quantum.solve_mean(root_path,full_path)
    if data['root']['centre_exact_rational']!=wave['actual_allq_feedback_mean']['centre_exact_rational']:raise ArithmeticError('Implicit wave mean differs')
    iv.dps=3000;mp.mp.dps=80
    data['g']=[interval(data['root']['parameters']['g0'])]+[interval(data['root']['parameters']['gs'])]*3
    data['geom']=field.full.normal_geometry(data['native'],data['g'])
    kato=total_gap_kato(data);definition=exact_wave_definition(wave,matter);proof=packet_prices(data,wave,matter,kato)
    with threadpool_limits(limits=1):
        literal=literal_nonlinear256(data)
        differential=literal_allq_Hermite(wave)
    descriptor=dict(plan='issue-222-induced-photon-delta13-continuation-20261008',assigned_scientific_revision='d86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7',
                    law='NEW originalFullForm N4 allPhi/SU4/CAR localGauss corrected compact source',parameters=dict(N=4,V=64,a='1/64',Lambda='2^8192',chi='2^-4096',Gphi='8chi',original_colour='3chi/10',active=1,auxiliary=63,negative_physical_heavy=1,staticRef=4,n=8324,clock=4096),
                    inputs=[dict(path=p,sha256=h) for p,h in inputs.items()],exact_mean=data['root'],wave_input_sha256=wave['input_sha256'],matter_input_sha256=matter['input_sha256'],source='E0=.5 sum4 c_active0alpha^dag Ralpha; physicalRef4->3',omitted_terms=['Theta/Pi','Coulomb'],original_SOURCE_frozen_separate=True)
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
                actual_full_nonuniform_fermionic_source_proof=proof,actual_exact5760_wave_and_corrector_definition=definition,actual_nonlinear256_Kato_translation_CAR_consumers=literal,
                actual_independent_full_matter_Hermite_differential=wave['actual_physical_noncommuting_differential_control'],
                actual_own_ALL64_direct_Hermite_product_rule=differential,
                source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(raw).hexdigest(),bytes_utf8=raw.decode()) for name,raw in SOURCE_BYTES.items()},
                source_unchanged_since_import={name:(ROOT/name).read_bytes()==raw for name,raw in SOURCE_BYTES.items()},input_unchanged={p:digest(Path(p))==h for p,h in inputs.items()},
                runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__,ijson=ijson.__version__),elapsed_seconds=time.monotonic()-started,
                adverse_facts=['New nonuniform Phi/SU4/CAR source prototype only; no QED/commoncone/originalSOURCE/vacuum/ground admission.','Non-small slowPhi colour potential AND linear cometric retained and subtracted together by kineticXi; no potential-only resolvent.','Pointwise local density is not2; exact p=.5 is a translation-invariant background expectation.','Physical whole field configuration space retained; only selected state compact, no hardwall Hamiltonian.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wave',type=Path,required=True);parser.add_argument('--matter',type=Path,required=True)
    parser.add_argument('--root-input',type=Path,required=True);parser.add_argument('--full45',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.wave,args.matter,args.root_input,args.full45)
    if not all(result['source_unchanged_since_import'].values()) or not all(result['input_unchanged'].values()):raise ArithmeticError('Scientific source/input changed during actual source CLI')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    proof=result['actual_full_nonuniform_fermionic_source_proof']
    print(json.dumps(dict(input_sha256=result['input_sha256'],Rayleigh_residual=proof['actual_prepared_Rayleigh_residual_upper'],raw_source_times=proof['raw_source_times'],elapsed_seconds=result['elapsed_seconds']),indent=2))

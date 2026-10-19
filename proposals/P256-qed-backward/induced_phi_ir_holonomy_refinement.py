#!/usr/bin/env python3
"""New single-coordinate alias and branchcut macro holonomy discriminator.

Frozen macro mean/well receipts are inputs. The contour comparison lemma and
physical well-energy interpretation are explicit for parent mathematical review.
This is not a global/spatial vacuum certificate or an N6 extrapolation.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from math import isqrt
from pathlib import Path
import platform
import sys
import time
from typing import Any
import mpmath as mp
from mpmath.ctx_iv import MPIntervalContext
import numpy as np
from induced_phi_backreaction import PhiConfig,models,scalar_paths,IMPORT_BYTES as BASE_BYTES
from induced_phi_ir_well_transfer import sqrt_bounds,pi_bounds,endpoint

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**BASE_BYTES,**{name:(ROOT/name).read_bytes() for name in (
    'induced_phi_ir_mean_certificate.py','induced_phi_ir_well_transfer.py',
    'induced_phi_ir_current_contact.py','induced_phi_ir_charge_symmetry.py',Path(__file__).name)}}
SOURCE_HASHES={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES.items()}


def rational(value: dict[str,str]) -> F:
    return F(int(value['numerator']),int(value['denominator']))


def encoded(value: F|int,upper: bool=True) -> dict[str,str]:
    out=F(value);rounding='exact rational'
    if max(out.numerator.bit_length(),out.denominator.bit_length())>12000:
        out=rational(endpoint(out,upper))
        rounding='exact outward dyadic upper' if upper else 'exact outward dyadic lower'
    with mp.workdps(80):display=str(mp.nstr(mp.mpf(out.numerator)/out.denominator,65))
    return dict(numerator=str(out.numerator),denominator=str(out.denominator),decimal_display=display,rounding=rounding)


def endpoints(value: Any) -> tuple[F,F]:
    result=[]
    for sign,mantissa,exponent,_ in value._mpi_:
        result.append(F((-1 if sign else 1)*int(mantissa))*F(2)**int(exponent))
    return result[0],result[1]


def exact(c: Any,value: F|int) -> Any:
    q=F(value);return c.mpf(q.numerator)/q.denominator


def hyperbolic(c: Any,value: Any) -> tuple[Any,Any]:
    positive=c.exp(value);negative=c.exp(-value)
    return (positive+negative)/2,(positive-negative)/2


def microscopic(cfg: PhiConfig) -> list[dict[str,Any]]:
    result=[]
    for load,model,role in models(cfg):
        onsite,paths=scalar_paths(model);speed=F(float(-2*paths[0][1][1].imag))
        if model.improved:
            k=F(float(-paths[0][1][0].real));mass=F(float(onsite))-F(9,2)*k
        else:
            k=F(float(-2*paths[0][1][0].real));mass=F(float(onsite))-3*k
        for displacement,scalars in paths:
            axis=int(np.flatnonzero(displacement)[0]);length=int(displacement[axis])
            if scalars[0].imag!=0 or np.any(scalars[1:].real!=0):raise ArithmeticError('Primitive scalar parity mismatch')
            actual_mass=F(float(scalars[0].real))
            expected_mass=(-k if length==1 else k/4) if model.improved else -k/2
            if actual_mass!=expected_mass:raise ArithmeticError('Actual Wilson scalar coefficient mismatch')
            if length==1 and F(float(scalars[axis+1].imag))!=-speed/2:raise ArithmeticError('Actual velocity scalar coefficient mismatch')
            if length==2 and np.any(scalars[1:]!=0):raise ArithmeticError('Actual improved primitive is not the retained squared-Wilson/one-link velocity law')
        result.append(dict(role=role,species=1 if role.startswith('neutral') else int(load),mass=mass,k=k,speed=speed,improved=model.improved,onsite=F(float(onsite))))
    return result


def strip_certificate(c: Any,profile: dict[str,Any],bounds: dict[str,F],width: F) -> dict[str,Any]:
    # Only ONE complex coordinate at a time; other two remain REAL.
    g0=exact(c,bounds['g0_lower'])+(exact(c,bounds['g0_upper'])-exact(c,bounds['g0_lower']))*c.mpf([0,1])
    gs=exact(c,bounds['gs_lower'])+(exact(c,bounds['gs_upper'])-exact(c,bounds['gs_lower']))*c.mpf([0,1])
    m,k,v=[exact(c,profile[key]) for key in ('mass','k','speed')]
    ch,sh=hyperbolic(c,exact(c,width))
    if profile['improved']:
        real_mass=m-k*sh**2
        sine_coefficient=gs*v**2*(1+2*sh**2)-4*g0*k**2*(1+ch)**2*sh**2
        twice_ch,_=hyperbolic(c,2*exact(c,width))
        x0=abs(exact(c,profile['onsite']))+4*k+4*k/4+2*k*ch+k*twice_ch/2
    else:
        real_mass=m-k*(ch-1)
        sine_coefficient=gs*v**2*(1+2*sh**2)-g0*k**2*sh**2
        x0=abs(exact(c,profile['onsite']))+2*k+k*ch
    radius_squared=g0*real_mass**2-gs*v**2*sh**2
    coefficient_norm_squared=x0**2+v**2*(2+ch**2)
    radius_lower=endpoints(radius_squared)[0]
    if min(endpoints(real_mass)[0],endpoints(sine_coefficient)[0],radius_lower-F(1,400),F(150**2)-endpoints(coefficient_norm_squared)[1])<=0:
        raise ArithmeticError('Actual single-coordinate complex strip proof failed')
    return dict(role=profile['role'],single_coordinate_strip_width=encoded(width),
        real_mass_lower=encoded(endpoints(real_mass)[0]),sine_coefficient_lower=encoded(endpoints(sine_coefficient)[0]),
        Re_radius_squared_lower=encoded(radius_lower),coefficient_norm_upper=encoded(sqrt_bounds(endpoints(coefficient_norm_squared)[1])[1]),
        R_modulus_lower=encoded(F(1,20)),gradient_component_analytic_cap=encoded(10**6),fourth_component_analytic_cap=encoded(10**15),energy_analytic_cap=encoded(300),
        derivation='For heavy x0=m+k sum(1-cos), for improved x0=m+k sum(1-cos)^2, both with xj=v sin. In one strip, Re(xGx)>=g0*m_strip^2-gs*v^2*sinh^2(sigma)+positive*sin^2(realtheta_j). Other two coordinates are real. m_strip and positive sine coefficient explicitly enclosed. Exact000 is diagonal by finite parity; this proof does NOT zero any quarter root. ||x_complex||<150 and |R|>1/20 give component caps 2||x||^2/R<1e6,8||x||^4/R^3<1e15,2|R|<300.')


def alias_upper(size: int,width: F,cap: int) -> tuple[F,int]:
    # ln2 <.694 gives e^-x<=2^-floor(x/.694), using exact integers.
    exponent=int(F(size)*width/F(347,500));r=F(1,2**exponent)
    if r>=F(1,2):raise ArithmeticError('Actual alias decay does not pay geometric denominator')
    return 24*cap*r,exponent


def branch_window(c: Any,heavy: dict[str,Any],bounds: dict[str,F],size: int) -> dict[str,Any]:
    pi_lower,pi_upper=pi_bounds();u=F(7,400);end=u+F(2,size)
    g0=exact(c,bounds['g0_lower'])+(exact(c,bounds['g0_upper'])-exact(c,bounds['g0_lower']))*c.mpf([0,1])
    gs=exact(c,bounds['gs_lower'])+(exact(c,bounds['gs_upper'])-exact(c,bounds['gs_lower']))*c.mpf([0,1])
    m,k,v=[exact(c,heavy[key]) for key in ('mass','k','speed')]
    other_mass=k*exact(c,(pi_upper/F(size))**2)
    other_sine2=exact(c,2*(pi_upper/F(size))**2)
    # Nearest-rest REAL y,z modes exist for every one of the64 twists.
    ch,sh=hyperbolic(c,exact(c,u))
    cut_mass=m+k*(1-ch)+other_mass
    Pupper=endpoints(g0*cut_mass**2+gs*v**2*(-sh**2+other_sine2))[1]
    rho_interval=exact(c,u)+exact(c,end-u)*c.mpf([0,1])
    interval_ch,interval_sh=hyperbolic(c,rho_interval)
    mass_lower=endpoints(m+k*(1-interval_ch))[0]
    derivative=2*interval_sh*(g0*k*exact(c,mass_lower)+gs*v**2*interval_ch)
    derivative_lower=endpoints(derivative)[0]
    Aupper=endpoints(g0*k**2-gs*v**2)[1]
    if max(Pupper,Aupper)>=0 or min(mass_lower,derivative_lower-1)<=0:raise ArithmeticError('Actual heavy branchcut window/concave polynomial proof failed')
    # For allrho>=u+1/N in the window: -P>=1/N. e^-(Nu+2)
    # >=2^-ceil((Nu+2)/.69), since ln2>.69. No underflow or logfit.
    x=(F(size)*u+2)/F(69,100);exponent=(x.numerator+x.denominator-1)//x.denominator
    rlower=F(1,2**exponent);root=isqrt(size)
    if root**2!=size:raise ArithmeticError('This exact thinwindow lowerprice uses the actual square N')
    return dict(branch_upper=u,branch_window_upper=end,r_lower=rlower,r_lower_exponent=exponent,
        P_at_upper_root_upper=Pupper,negative_derivative_lower=derivative_lower,A_upper=Aupper,
        percell_heavy_energy_gap_lower=rlower/F(16*size**3*root),
        capsule=dict(rho_positive_branch_upper=encoded(u),window_lower=encoded(u+F(1,size)),window_upper=encoded(end),
            nearest_other_angles_upper=encoded(pi_upper/size),P_upper_at_rho_u=encoded(Pupper),negative_derivative_lower=encoded(derivative_lower),
            concave_cos_polynomial_A_upper=encoded(Aupper),cut_modulus_lower=encoded(F(1,root)),cut_exponential_lower=encoded(rlower),
            exponent_lower_witness=f'ln2>.69 =>exp(-(N*.0175+2))>=2^-{exponent}',
            normalization='One existing nearest-rest y/z pair has weight1/N^2, thin rho interval width1/N, cut modulus>=1/sqrtN. S(r,0)-S(r,-1)>=r/2 forr<=1/4; 2/pi prefactor. Retained weaker energy gap r/(16 N^3 sqrtN) covers allconstants.'))


def native_cut_control(profile: dict[str,Any],g0: F,gs: F) -> dict[str,Any]:
    with mp.workdps(55):
        m,k,v=[mp.mpf(profile[key].numerator)/profile[key].denominator for key in ('mass','k','speed')]
        G0=mp.mpf(g0.numerator)/g0.denominator;GS=mp.mpf(gs.numerator)/gs.denominator
        rows: list[dict[str,Any]]=[];wrong_cut=mp.mpf(0)
        for size in (6,8):
            for y,z in ((mp.mpf(0),mp.mpf(0)),(mp.mpf('.3'),mp.mpf('.7'))):
                mass=m+k*(3-mp.cos(y)-mp.cos(z));S=mp.sin(y)**2+mp.sin(z)**2
                A=G0*k*k-GS*v*v;B=-2*G0*k*mass;C=G0*mass*mass+GS*v*v*(1+S)
                roots=[(-B+sign*mp.sqrt(B*B-4*A*C))/(2*A) for sign in (-1,1)]
                positive=max(roots);negative=min(roots);rp=mp.acosh(positive);rn=mp.acosh(-negative)
                def mean(t):
                    return sum(mp.sqrt(A*mp.cos(2*mp.pi*(n+mp.mpf(t)/4)/size)**2+B*mp.cos(2*mp.pi*(n+mp.mpf(t)/4)/size)+C) for n in range(size))/size
                def cut(base,t):
                    cosine=mp.cos(mp.pi*t/2)
                    def integrand(s):
                        rho=base+s*s;r=mp.exp(-size*rho)
                        diff=(r*cosine-r*r)/(1-2*r*cosine+r*r)+r/(1+r)
                        delta=2*mp.sinh(s*s/2)*mp.sinh(base+s*s/2)
                        modulus=mp.sqrt((-A)*delta*(positive-negative+delta))
                        return 2*s*modulus*diff
                    return 2/mp.pi*mp.quad(integrand,[0,mp.mpf('.1'),mp.mpf('.5'),1,2,mp.inf])
                for t in (0,1,3):
                    direct=mean(2)-mean(t);first=cut(rp,t);second=cut(rn,t);from_cut=first+second
                    relative=abs(from_cut-direct)/(1+abs(direct));wrong_cut=max(wrong_cut,abs(first-direct))
                    if direct<=0 or relative>mp.mpf('1e-40'):raise ArithmeticError('Actual finite-grid/branchcut geometric-kernel control failed')
                    rows.append(dict(size=size,other_angles=[str(y),str(z)],twist=t,direct_positive_mean_difference=str(direct),both_cut_difference=str(from_cut),relative_error=str(relative)))
        if wrong_cut<mp.mpf('1e-8'):raise ArithmeticError('Missing second branchcut adversary did not discriminate')
        return dict(rows=rows,max_relative_error=max(float(row['relative_error']) for row in rows),missing_negative_branchcut_max_error=str(wrong_cut),
            scope='Actual primitive scalar couplings and diagnostic midpoint of the certified finite000 enclosure; N6/N8 exact discrete sums versus independent positive+negative cut integrals. Native controls do not extrapolate the macro selection: macro selection uses the exactinterval guards and lemma.')


def run(mean_path: str,well_path: str) -> dict[str,Any]:
    started=time.monotonic();rawmean=Path(mean_path).read_bytes();rawwell=Path(well_path).read_bytes();mean=json.loads(rawmean);well=json.loads(rawwell)
    if well['input_descriptor']['mean_input_sha256']!=mean['input_sha256']:raise ArithmeticError('Different actual coupled finite mean')
    if well['input_descriptor']['mean_capsule_sha256']!=hashlib.sha256(json.dumps(mean,sort_keys=True).encode()).hexdigest():raise ArithmeticError('Different actual mean capsule bytes')
    size=mean['finite_size'];exp=mean['Lambda_exponent'];volume=size**3;L=F(2**exp)
    if (size,exp)!=(262144,8192):raise ValueError('This discriminator is pinned to the actual macrobranch')
    cfg=PhiConfig(Lambda=1.,**mean['microscopic_config']);profiles=microscopic(cfg)
    bounds={name:rational(mean['centre_enclosure'][name]) for name in ('g0_lower','g0_upper','gs_lower','gs_upper')}
    c: Any=MPIntervalContext();c.dps=120
    strips=[strip_certificate(c,p,bounds,F(3,250) if not p['improved'] else F(3,400)) for p in profiles]
    heavy_grad,he=alias_upper(size,F(3,250),10**6);light_grad,le=alias_upper(size,F(3,400),10**6)
    grad=4*(heavy_grad+64*light_grad/L)
    certified_convexity=rational(mean['actual_finite_grid_quadrature_certificate']['local_coupled_convexity_lower'])
    if certified_convexity<127:raise ArithmeticError('Actual coupled root is not127-convex')
    convexity=F(127)
    root_radius=grad/convexity
    optimizer_error=L*volume*grad**2/(2*convexity)
    window=branch_window(c,profiles[0],bounds,size);heavy_gap=L*volume*window['percell_heavy_energy_gap_lower']
    charged_alias,_=alias_upper(size,F(3,400),300);charged_mean_price=64*volume*charged_alias
    fourth_heavy,_=alias_upper(size,F(3,250),10**15);fourth_light,_=alias_upper(size,F(3,400),10**15)
    stiffness_delta=10**9*root_radius+10**6*(fourth_heavy+64*fourth_light/L)
    amin=rational(mean['slow_stiffness_lower']);slow_difference=45*F(2**(exp//4))*stiffness_delta/(2*sqrt_bounds(amin)[0])
    prices={name:rational(value) for name,value in well['prices'].items()}
    trial=prices['fast_energy_upper']+prices['slow_energy_upper']+prices['harmonic_reprepare_energy_excess']+prices['cut_energy_excess']
    quantum_window=16*(prices['harmonic_reprepare_energy_excess']+prices['cut_energy_excess']+prices['nonlinear_form_relative']*(trial+prices['slow_energy_upper'])+prices['mean_force_action_upper']+prices['Haar_FP_scalar_action_upper'])
    # Allcharged/em states are priced absolutely; no photon ground/eigenstate assumption.
    path_cap=F(0)
    for load,model,role in models(cfg):
        if role.startswith('neutral'):continue
        onsite,paths=scalar_paths(model);cap=abs(F(float(onsite)))
        for _,scalars in paths:
            for scalar in scalars:
                real,imag=F(float(scalar.real)),F(float(scalar.imag));cap+=2*sqrt_bounds(real*real+imag*imag)[1]
        path_cap+=F(float(load))*cap
    rho=sqrt_bounds(rational(mean['centre_enclosure']['trace_Gram_upper']))[1]
    charged_allowance=32*volume*path_cap*(1+rho+F(1,2**2048))
    em_modes=2*volume+1;laplower=prices['smallest_laplacian_lower']
    EM_allowance=2/F.from_float(cfg.a)*(em_modes*3/F(1,8)**2+128**2*volume/laplower)
    physical_price=2*quantum_window+2*optimizer_error+slow_difference+charged_mean_price+charged_allowance+EM_allowance+2*prices['heavy_Berry_Gauss_affine_action_upper']+2*prices['boundary_action_upper']
    margin=heavy_gap-physical_price
    if margin<=0:raise ArithmeticError('Actual all64 physical macro well-energy comparison not discriminating')
    native=native_cut_control(profiles[0],(bounds['g0_lower']+bounds['g0_upper'])/2,(bounds['gs_lower']+bounds['gs_upper'])/2)
    rows=[dict(twist=list(t),different_axes=sum(x!=2 for x in t),energy_margin_above222_lower=encoded(sum(x!=2 for x in t)*heavy_gap-physical_price,False) if t!=(2,2,2) else encoded(0)) for t in itertools.product(range(4),repeat=3)]
    descriptor=dict(mean_receipt_sha256=hashlib.sha256(rawmean).hexdigest(),well_receipt_sha256=hashlib.sha256(rawwell).hexdigest(),mean_input_sha256=mean['input_sha256'],well_input_sha256=well['input_sha256'],finite_size=size,Lambda_exponent=exp,microscopic_config=mean['microscopic_config'],
        construction='NEWactual macro single-coordinate alias +two real branchcut holonomy selector; allfrozen uniform64/128 preparations remain unchanged',governing_plan='issue-222-induced-photon-delta13-continuation-20261008',plan_revision='d13d7d28ebfd7cb8386e7442897c596481aeab95cd55def1a8b3a60c1e89f4dd')
    return dict(branch=descriptor['construction'],input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        actual_single_coordinate_strip_certificates=strips,alias_decay_exponents=dict(heavy=he,charged=le),
        refined_all64_Gram_distance_to_exact000_upper=encoded(root_radius),finite_fullten_twist_gradient_difference_upper=encoded(grad),optimizer_energy_price_upper=encoded(optimizer_error),
        actual_heavy_branchcut_window=window['capsule'],heavy_classical_physical_box_gap_lower=encoded(heavy_gap,False),charged_finite_phase_energy_price_upper=encoded(charged_mean_price),
        normal_stiffness_twist_difference_upper=encoded(stiffness_delta),uniform_slow_zero_point_difference_upper=encoded(slow_difference),
        physical_energy_price_ledger=dict(two_local_well_minmax_windows_upper=encoded(2*quantum_window),charged_fullCAR_absolute_allowance=encoded(charged_allowance),noncompact_EM_H2_kinetic_and_real_Coulomb_upper=encoded(EM_allowance),all_adverse_prices_upper=encoded(physical_price)),
        actual_all64_margin_lower=encoded(margin,False),all64_rows=rows,native_discrete_vs_branchcut_control=native,
        selected_candidate=dict(twist=[2,2,2],exact_centre='Exact same fullten finite coupled stationarity law in sector222, not N6 values; finite anti-periodic parity+cube and local uniqueness force EXACT diagonal g0,gs,gs,gs. Enclosed by original exact000 g0/gs intervals plus the refined root-distance bound; no offdiagonal zeroing ofquartersectors.',
            preparation='NEWpossible preferred222 C-symmetric two-component density (|Psi222><Psi222|+|C Psi222><C Psi222|)/2, using actual16x16 mobile-CAR and fullfield pullback in frozen charge-symmetry module; same macroclock/budgets by convexity. This does not replace frozenuniform64/128 state bytes.',
            no_promotion='Arithmetic/contour/physical comparison presented for parent review, not globally selected photon vacuum or fullspatial ground.'),
        contour_lemma='At diagonal exactG000, fixing two REAL momenta, heavy R^2=A cos^2theta+B cos theta+C with A<0 and roots z+>1,z-<-1. Forn>=2: Rhat_n=-(1/pi)[integral_rhoplus^inf e^-n rho sqrt(-P(cosh rho))drho+(-1)^n integral_rhominus^inf e^-n rho sqrt(-P(-cosh rho))drho]. N is EVEN, so everymN has bothnegative contributions. The finite-twist sum uses S(r,c)=(r c-r^2)/(1-2r c+r^2), strictly increasingin c; hence filledR is largest at t=2 pointwise inother two realmomenta. Iterateaxes. Thinpositivecut window gives explicit gap; chargedaliases and fullten optimizer corrections separatelypriced. Not a continuum replacement for any finite sum.',
        physical_comparison_lemma='Same exact classical+canonicalfastZPE coupled energy atG000 cancels algebraically beforetwist comparison. Strongconvexity prices fullten optimisation by ||twistgradient||^2/(2kappa). Fullnormal moment/projector variation prices the45slowZPE. Native macro minmax residual/metric/cut/force budgets bound both localDirichlet ground energies around their harmonic baselines. Normalized jointGauss lifts add the actual heavyBerry/affine and boundary action prices. ChargedCAR is bounded absolutely oncut support and the same fixedRef4 sector, while UE/a bareelectric and real Coulomb are positive; the explicit EM H2 packet supplies the stated upperenergy. Thus this is a finitebox LOCALwell/fullGauss energy competitor comparison conditional on these displayed lemmas, not a globalvacuum theorem.',
        limitations='Requires parent review of two-cut contouridentity, analyticfinite-rule alias theorem and localwell energy sandwich. It does NOT prove competing nonuniform phases absent, bulkstationarity, photonvacuum, commoncone, radiativeCompton or stability of the fullthermodynamic phase. Nativecontrols diagnosticmidpoint only; physical roots remain exactimplicit. No newbare B or speedfit.',
        source_at_import=SOURCE_HASHES,own_source_at_import=SOURCE_BYTES[Path(__file__).name].decode(),all_sources_unchanged_at_completion=all((ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()),
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,mpmath=mp.__version__,platform=platform.platform()),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mean',required=True);parser.add_argument('--well',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=run(args.mean,args.well)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','source_at_import','alias_decay_exponents','refined_all64_Gram_distance_to_exact000_upper','heavy_classical_physical_box_gap_lower','actual_all64_margin_lower','native_discrete_vs_branchcut_control','elapsed_seconds')},indent=2))

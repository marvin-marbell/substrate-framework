"""Certified finite-macro charged-preparation/Gaussian energy upper bound.

The N32 periodic grid is an analytic-strip quadrature with an explicit
finite-N262144 alias price, NOT promotion of a smaller physical box.
Full quantum Gauss Coulomb is bounded using the projector chord and exact
finite-lattice Parseval; all real harmonics remain normalized Gaussian modes.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any
import mpmath as mp
from induced_phi_em_sw_preparation import SOURCE_BYTES as PREP_BYTES
from induced_phi_em_gaussian_variational import build_native
from induced_phi_backreaction import scalar_paths
from induced_phi_em_gain_certificate import centre, iv, endpoints, encoded
from induced_phi_ir_well_transfer import rational

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**PREP_BYTES,'induced_phi_em_gain_certificate.py':(ROOT/'induced_phi_em_gain_certificate.py').read_bytes(),Path(__file__).name:Path(__file__).read_bytes()}
IV: Any=mp.iv


def scalars(profile: dict[str,Any]) -> dict[str,Any]:
    onsite,paths=scalar_paths(profile['model']);groups={}
    for disp,c in paths:
        nonzero=[j for j,d in enumerate(disp) if d]
        if len(nonzero)!=1 or disp[nonzero[0]] not in (1,2):raise ValueError('Actual primitive is outside axial one/two-link contract')
        axis=nonzero[0];length=int(disp[axis]);z=[complex(x) for x in c]
        if z[0].imag or any(z[j+1].real for j in range(3)) or any(z[j+1] for j in range(3) if j!=axis):
            raise ValueError('Actual Clifford scalar paths do not meet the cubic certificate contract')
        groups[length,axis]=(F(z[0].real),F(z[axis+1].imag))
    for length in (1,2):
        if not all((length,j) in groups for j in range(3)) or any(groups[length,j]!=groups[length,0] for j in range(3)):
            raise ValueError('Actual primitive lacks cubic axial coefficients')
    return dict(onsite=F(float(onsite)),path={length:groups[length,0] for length in (1,2)},load=profile['load'])


def cosh(value: Any) -> Any:
    return (IV.exp(value)+IV.exp(-value))/2


def sinh(value: Any) -> Any:
    return (IV.exp(value)-IV.exp(-value))/2


def quadrature(coeff: dict[str,Any],g0: Any,gs: Any,eta: F,n: int) -> dict[str,Any]:
    rho=iv(F(1,2));mass=coeff['onsite'];paths=coeff['path'];load=coeff['load']
    mmin=IV.sqrt(g0)*(iv(mass)-6*iv(eta)*sum((iv(abs(paths[l][0])) for l in (1,2)),iv(0)))
    re_mass=IV.sqrt(g0)*(iv(mass)-6*iv(eta)*sum((iv(abs(paths[l][0]))*cosh(l*rho) for l in (1,2)),iv(0)))
    im_mass=6*iv(eta)*IV.sqrt(g0)*sum((iv(abs(paths[l][0]))*sinh(l*rho) for l in (1,2)),iv(0))
    im_space=2*iv(eta)*IV.sqrt(gs)*sum((iv(abs(paths[l][1]))*sinh(l*rho) for l in (1,2)),iv(0))
    radicand=re_mass**2-im_mass**2-3*im_space**2
    if endpoints(mmin)[0]<=0 or endpoints(re_mass)[0]<=0 or endpoints(radicand)[0]<=0:raise ArithmeticError('Actual charged preparation analytic strip has no positive PH gap')
    denominator=IV.sqrt(radicand);trig={}
    for j in range(n):
        angle=(2*j+1)*IV.pi/n
        trig[j]={l:(IV.cos(l*angle),IV.sin(l*angle)) for l in (1,2)}
    totals=[iv(0),iv(0),iv(0)]
    for index in itertools.product(range(n),repeat=3):
        components=[iv(mass),iv(0),iv(0)];spatial={l:[] for l in (1,2)}
        for l in (1,2):
            components[l]=2*iv(paths[l][0])*sum((trig[j][l][0] for j in index),iv(0))
            spatial[l]=[-2*iv(paths[l][1])*trig[j][l][1] for j in index]
        x=components[0]+iv(eta)*(components[1]+components[2])
        y=[iv(eta)*(spatial[1][j]+spatial[2][j]) for j in range(3)]
        R=IV.sqrt(g0*x*x+gs*sum((v*v for v in y),iv(0)))
        totals[0]+=-2*load*g0*x*components[0]/R
        for l in (1,2):totals[l]+=-2*load*(g0*x*components[l]+gs*sum((y[j]*spatial[l][j] for j in range(3)),iv(0)))/R
    totals=[value/(n**3) for value in totals]
    # Modulus bounds on the complete complex polydisc, with literal coefficients.
    X=iv(mass)+6*iv(eta)*sum((iv(abs(paths[l][0]))*cosh(l*rho) for l in (1,2)),iv(0))
    Y=2*iv(eta)*sum((iv(abs(paths[l][1]))*cosh(l*rho) for l in (1,2)),iv(0))
    maxima=[2*load*g0*X*iv(mass)/denominator]
    for l in (1,2):
        maxima.append(2*load*(g0*X*6*iv(abs(paths[l][0]))*cosh(l*rho)+gs*3*Y*2*iv(abs(paths[l][1]))*cosh(l*rho))/denominator)
    alias=(1+2*IV.exp(-rho*n)/(1-IV.exp(-rho*n)))**3-1
    priced=[]
    for value,M in zip(totals,maxima):
        error=endpoints(2*M*alias)[1];lo,hi=endpoints(value);priced.append((lo-error,hi+error))
    A=sum((l*l*(g0*iv(paths[l][0]**2)+gs*iv(paths[l][1]**2)) for l in (1,2)),iv(0))
    C=load*iv(eta*eta)*A/(mmin*mmin)
    return dict(components=priced,Coulomb_over_UElink_upper=endpoints(C)[1],
                strip_radius=F(1,2),strip_Re_R2_lower=endpoints(radicand)[0],minimum_mass_lower=endpoints(mmin)[0])


def run(mean_path: Path,refiner_path: Path,pure_path: Path) -> dict[str,Any]:
    started=time.monotonic();inputs={}
    for role,path in (('mean',mean_path),('refiner',refiner_path),('purePhi',pure_path)):
        raw=path.read_bytes();inputs[role]=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),capsule=json.loads(raw))
    mean=inputs['mean']['capsule'];refiner=inputs['refiner']['capsule'];pure=inputs['purePhi']['capsule'];N=int(mean['finite_size']);V=N**3;M=2*V+1
    n=32
    if N<n or N<=4:raise ValueError('Actual finite macro must dominate quadrature size and avoid two-link selfwrap')
    IV.dps=65;mids,errors=centre(mean,refiner)
    g0=iv(mids[0])+IV.mpf([-1,1])*iv(errors[0]);gs=iv(mids[1])+IV.mpf([-1,1])*iv(errors[1])
    native=build_native(mean,2);coeffs=[scalars(profile) for profile in native['Pprofiles']];UElink=F(float(native['UE']))/F(float(native['a']))
    eta=F(1,5);variance=F(1);values=[quadrature(c,g0,gs,eta,n) for c in coeffs]
    components=[(sum((v['components'][l][0] for v in values),F(0)),sum((v['components'][l][1] for v in values),F(0))) for l in range(3)]
    factor1=IV.exp(-iv(variance)*iv(F(2,3)+F(1,3*V))/2)
    factor2=IV.mpf([0,1])*(1-IV.exp(-iv(variance)))+IV.exp(-iv(variance))
    ci=[iv(lo)+IV.mpf([0,1])*iv(hi-lo) for lo,hi in components]
    energy=ci[0]+factor1*ci[1]+factor2*ci[2]
    C=UElink*sum((v['Coulomb_over_UElink_upper'] for v in values),F(0));K=UElink*F(M,4*V)/variance
    upper=endpoints(energy)[1]+C+K
    benchmark=-2*IV.sqrt(g0)*sum((c['load']*iv(c['onsite']) for c in coeffs),iv(0))+iv(UElink*F(M,16*V))
    benchmark_lower=endpoints(benchmark)[0];gain=benchmark_lower-upper
    if gain<=0:raise ArithmeticError('Actual macro quantum reference does not beat the same-Href onsiteCAR Gaussian benchmark')
    desc=dict(inputs={role:{key:entry[key] for key in ('path','sha256')} for role,entry in inputs.items()},N=N,V=V,M=M,
        eta_preparation_ONLY=encoded(eta),variance=encoded(variance),kappa=0,charged_counts=[1,63],a=native['a'],UE=native['UE'],quadrature_grid=n,
        plan='issue-222-induced-photon-delta13-continuation-20261008',plan_revision='85975947857ac7816cf0d8674f69229861014434b125a07a6996af0ecf72c4eb')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),
        energy_per_V_upper=encoded(upper),same_finite_Href_onsiteCAR_sigma4_energy_per_V_lower=encoded(benchmark_lower),actual_macro_gain_per_V_lower=encoded(gain),
        full_quantum_Coulomb_per_V_upper=encoded(C),primitive_electric_per_V_exact=encoded(K),
        interval_matter_components=[[encoded(x) for x in pair] for pair in components],interval_profiles=[{key:encoded(value) for key,value in v.items() if key!='components'} for v in values],
        state='NEWpure222 macrostate: exactimplicitactualCubicC, fixedcentre negativeSlater of onsite+eta*hopping(theta0) foractive1+aux63, normalizedREALGaussian s²1 inallMcoexact+harmonic coordinates, fullrealGauss lift, Ref4NET0/neutralheavyunchanged. Eta=.2 definesPREPARATION, neverphysicalHamiltonian. LiftthroughactualPhiwell/projectedheavyconstantphase section. Exactnorm1; noGibbs trace orcompacttorons. ThisdoesNOTretagolderpure222/rho128/finiteSCREEN sources.',
        proof='Finiteanti222momentum spectralSlaters haveTrP(k)=2. Chargedmeanrho0 andindependentflavour connectedcovariance sumlinearly1+63. S(q)=average[1-n(k).n(k+q)]<=average|d(k)-d(k+q)|²/(2Mmin²). ExactfiniteParseval and|e^(ilq)-1|²<=l²|e^iq-1|² giveC/V<=UElink sumloads eta²sum_l l²(g0F0_l²+gsFj_l²)/Mmin², includesALLq!=0 andthetrueCoulomboperator. One-linkGaussianfactor exactexp(-s²(2/3+1/(3V))/2); two-linkfactor in[exp(-s²),1] fromPTprojectionnorm<=2. ComplexstripReR²>0 pricesallperiodicFourieraliases: error<=2M[(1+2e^-rho*n/(1-e^-rho*n))³-1], coversBOTHquadraturegrid andactualN>=n. No smallboxenergy promotion.',
        Phi_transfer={key:value for key,value in pure['new_projected_section_transfer']['prices'].items()},
        Phi_transfer_reason='OriginalPhiwell referencegraph usesUNIVERSALCoulombcap UElink V132²/lapmin andchargedoperatorcap, notspecialSlaterchargecovariance. NewGaussian Pi2/Pi4 arebelowoldH2packet values192/129024; allM kineticgraphdominated. SameHref conservesgraph atalltimes. ChargedPhi freezingoperatorcap isstateindependent; neutralheavy/projected64x derivative/metric/Haar/FP/Gauss capsunchanged. Thussame64xPhi norm/trace/weighted/current/contact budget applies subjecttoparentcomparisonreview.',
        noncompact_tails=dict(coordinate_second_moment_exact=encoded(M*variance),momentum_second_moment_exact=encoded(F(M,4)/variance),formula='P(||Theta||>R)<=Mvariance/R²; P(||Pi||>R)<=M/(4varianceR²), no clipping.'),
        limitations='CertifiedquantumMACROenergyupper/sameHrefbenchmark, notground/stationaryphotonvacuum, actualEMfiniteclockpropagator/pole/commoncone orCompton/QEDadmission. Mathematicalcomparison/Fourier/projectorchord/Phi reviewrequired.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('mean','refiner','pure','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.mean,args.refiner,args.pure)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','energy_per_V_upper','same_finite_Href_onsiteCAR_sigma4_energy_per_V_lower','actual_macro_gain_per_V_lower','full_quantum_Coulomb_per_V_upper','primitive_electric_per_V_exact','elapsed_seconds')},indent=2))

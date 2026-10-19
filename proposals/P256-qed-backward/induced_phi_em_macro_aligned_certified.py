"""Same eta7/20 physical source: narrower certified strip, fused energy/Coulomb jets."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_macro_coulomb_gradient import SOURCE_BYTES as GRADIENT_BYTES
from induced_phi_em_macro_preparation import scalars, build_native, centre, cosh, sinh
from induced_phi_em_macro_photon_domain import PLAN
from induced_phi_em_gain_certificate import IV, iv, endpoints, encoded
from induced_phi_ir_well_transfer import rational, sqrt_bounds
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**GRADIENT_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION='b9e13b2491321aa804f7e52cf4ba3cb12c0a224717cb07b23e487afc0768463b'


def profile(coeff: dict[str,Any],g0: Any,gs: Any,eta: F,N: int,n: int,rho: F) -> dict[str,Any]:
    mass=coeff['onsite'];paths=coeff['path'];load=coeff['load'];z=iv(rho)
    mmin=IV.sqrt(g0)*(iv(mass)-6*iv(eta)*sum((iv(abs(paths[l][0])) for l in (1,2)),iv(0)))
    rm=IV.sqrt(g0)*(iv(mass)-6*iv(eta)*sum((iv(abs(paths[l][0]))*cosh(l*z) for l in (1,2)),iv(0)))
    im=6*iv(eta)*IV.sqrt(g0)*sum((iv(abs(paths[l][0]))*sinh(l*z) for l in (1,2)),iv(0))
    iy=2*iv(eta)*IV.sqrt(gs)*sum((iv(abs(paths[l][1]))*sinh(l*z) for l in (1,2)),iv(0))
    rad=rm**2-im**2-3*iy**2;radlo=endpoints(rad)[0]
    if min(endpoints(mmin)[0],endpoints(rm)[0],radlo)<=0:raise ArithmeticError('New actual source has no certified analytic strip')
    X=iv(mass)+6*iv(eta)*sum((iv(abs(paths[l][0]))*cosh(l*z) for l in (1,2)),iv(0))
    Y=2*iv(eta)*sum((iv(abs(paths[l][1]))*cosh(l*z) for l in (1,2)),iv(0))
    xp=2*iv(eta)*sum((l*iv(abs(paths[l][0]))*cosh(l*z) for l in (1,2)),iv(0))
    yp=2*iv(eta)*sum((l*iv(abs(paths[l][1]))*cosh(l*z) for l in (1,2)),iv(0))
    denominator=IV.sqrt(iv(radlo));maxima=[2*load*g0*X*iv(mass)/denominator]
    for l in (1,2):maxima.append(2*load*(g0*X*6*iv(abs(paths[l][0]))*cosh(l*z)+3*gs*Y*2*iv(abs(paths[l][1]))*cosh(l*z))/denominator)
    jetmodulus=3*(g0*xp*xp+gs*yp*yp)/iv(radlo)+3*(g0*X*xp+gs*Y*yp)**2/iv(radlo*radlo)
    Pmod=endpoints(1+(IV.sqrt(g0)*X+3*IV.sqrt(gs)*Y)/denominator)[1]
    trig={j:{l:(IV.cos(l*(2*j+1)*IV.pi/n),IV.sin(l*(2*j+1)*IV.pi/n)) for l in (1,2)} for j in range(n)}
    totals=[iv(0),iv(0),iv(0)];D=iv(0)
    for index in itertools.product(range(n),repeat=3):
        c=[iv(mass),iv(0),iv(0)];sp={}
        for l in (1,2):
            c[l]=2*iv(paths[l][0])*sum((trig[j][l][0] for j in index),iv(0));sp[l]=[-2*iv(paths[l][1])*trig[j][l][1] for j in index]
        x=c[0]+iv(eta)*(c[1]+c[2]);y=[iv(eta)*(sp[1][j]+sp[2][j]) for j in range(3)];R2=g0*x*x+gs*sum((v*v for v in y),iv(0));R=IV.sqrt(R2)
        totals[0]+=-2*load*g0*x*c[0]/R
        for l in (1,2):totals[l]+=-2*load*(g0*x*c[l]+gs*sum((y[j]*sp[l][j] for j in range(3)),iv(0)))/R
        dx=[-2*iv(eta)*sum((l*iv(paths[l][0])*trig[j][l][1] for l in (1,2)),iv(0)) for j in index]
        dy=[-2*iv(eta)*sum((l*iv(paths[l][1])*trig[j][l][0] for l in (1,2)),iv(0)) for j in index]
        D+=(g0*sum((v*v for v in dx),iv(0))+gs*sum((v*v for v in dy),iv(0)))/R2-sum(((g0*x*a+gs*b*cj)**2 for a,b,cj in zip(dx,y,dy)),iv(0))/(R2*R2)
    aliases=(1+2*IV.exp(-z*n)/(1-IV.exp(-z*n)))**3-1
    components=[]
    for value,bound in zip(totals,maxima):
        lo,hi=endpoints(value/(n**3));price=endpoints(2*bound*aliases)[1];components.append((lo-price,hi+price))
    dlo,dhi=endpoints(D/(n**3));Dprice=endpoints(jetmodulus*aliases)[1];dlo-=Dprice;dhi+=Dprice
    V=N**3;tail=F(1,2**512);exponent=(rho*N).numerator//(rho*N).denominator
    if N<n or N%2 or exponent<1024 or F((9*N)**2*3*V)*Pmod**2/F(2**exponent)>tail**2:raise ArithmeticError('Actual finite-N alias proof fails')
    moment=(sqrt_bounds(dhi)[1]+tail)**2
    return dict(load=load,components=components,continuous_jet_interval=[encoded(dlo),encoded(dhi)],jet_quadrature_alias_price=encoded(Dprice),actual_finite_N_weighted_projector_moment_upper=encoded(moment),strip_radius=encoded(rho),strip_Re_R2_lower=encoded(radlo),projector_strip_Frobenius_upper=encoded(Pmod),actual_finite_N_weighted_alias_l2_upper=encoded(tail))


def run(prior_path: Path,circulation_path: Path) -> dict[str,Any]:
    started=time.monotonic();raw=prior_path.read_bytes();prior=json.loads(raw);old=prior['input_descriptor'];craw=circulation_path.read_bytes();circ=json.loads(craw)
    if old['N']!=circ['input_descriptor']['N'] or old['charged_counts']!=[1,63] or old['kappa']!=0:raise ValueError('No physicalbox/count/source substitution')
    inputs={}
    for role in ('mean','refiner'):
        spec=old['inputs'][role];data=(ROOT/spec['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=spec['sha256']:raise ValueError('Actual mean epoch changed')
        inputs[role]=json.loads(data)
    N=int(old['N']);V=int(old['V']);M=int(old['M']);eta=F(7,20);s=F(3,5);g=F(float(old['UE']))/F(float(old['a']));n=48;rho=F(1,3)
    IV.dps=65;mids,errors=centre(inputs['mean'],inputs['refiner']);g0=iv(mids[0])+IV.mpf([-1,1])*iv(errors[0]);gs=iv(mids[1])+IV.mpf([-1,1])*iv(errors[1])
    rows=[profile(scalars(p),g0,gs,eta,N,n,rho) for p in build_native(inputs['mean'],2)['Pprofiles']]
    pairs=[(sum((row['components'][l][0] for row in rows),F(0)),sum((row['components'][l][1] for row in rows),F(0))) for l in range(3)]
    components=[iv(lo)+IV.mpf([0,1])*iv(hi-lo) for lo,hi in pairs];u1=F(2,3)+F(1,3*V)
    u2lo,u2hi=(rational(x) for x in circ['actual_macro_two_link_Q_norm_squared_interval']);u2=iv(u2lo)+IV.mpf([0,1])*iv(u2hi-u2lo);f1=IV.exp(-iv(s*u1)/2);f2=IV.exp(-iv(s)*u2/2)
    C=g*F(V-1,6*V)*sum((row['load']*rational(row['actual_finite_N_weighted_projector_moment_upper']) for row in rows),F(0));K=g*F(M,4*V)/s
    upper=endpoints(components[0]+f1*components[1]+f2*components[2])[1]+C+K
    sold=rational(old['variance']);oc=[iv(rational(row[0]))+IV.mpf([0,1])*iv(rational(row[1])-rational(row[0])) for row in prior['interval_matter_components']]
    lower=endpoints(oc[0]+IV.exp(-iv(sold*u1)/2)*oc[1]+IV.exp(-iv(sold)*u2/2)*oc[2])[0]+rational(prior['primitive_electric_per_V_exact']);gain=lower-upper
    if gain<=0:raise ArithmeticError('New actualcoupledsource cannotbeat SAMEHref precedingquantumstate')
    force=endpoints(iv(-2*K)-iv(s*u1)*f1*components[1]-iv(s)*u2*f2*components[2]);contact=endpoints(-(f1*components[1]+4*f2*components[2])/3)
    price=4*rational(prior['Phi_transfer']['all_flavour_contact_expectation_error'])/(F(float(old['a']))**2);third=[4*g*g*x for x in (contact[0]-price,contact[1]+price)]
    desc=dict(old);desc.update(eta_preparation_ONLY=encoded(eta),variance=encoded(s),quadrature_grid=n,analytic_strip_radius=encoded(rho),plan=PLAN,plan_revision=REVISION,inherited_source_plan_revision=old['plan_revision'],prior_source_sha256=hashlib.sha256(raw).hexdigest(),prior_input_sha256=prior['input_sha256'],actual_circulation_sha256=hashlib.sha256(craw).hexdigest())
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),energy_per_V_upper=encoded(upper),prior_same_finite_Href_quantum_source_energy_per_V_lower=encoded(lower),same_finite_Href_gain_per_V_lower=encoded(gain),full_quantum_Coulomb_per_V_upper=encoded(C),primitive_electric_per_V_exact=encoded(K),interval_matter_components=[[encoded(x) for x in pair] for pair in pairs],profiles=[{k:v for k,v in row.items() if k!='components'} for row in rows],actual_two_link_Q_norm_squared_interval=circ['actual_macro_two_link_Q_norm_squared_interval'],Phi_transfer=prior['Phi_transfer'],reference_dilation_force_per_V_interval=[encoded(x) for x in force],reference_literal_Theta_contact_interval=[encoded(x) for x in contact],corrected_four_entry_Theta_contact_price=encoded(price),corrected_full_H_contact_consumed_third_interval=[encoded(x) for x in third],
        state='NEWactualN262144/allMrealGaussian variance3/5 x fixedcentrecanonicalnegativeSlater(onsite+(7/20)hop), active1m.2+aux63m2. Onlypreparation changed; originalphysicalH/a/UE/r/v/masses/Ref4NET0/neutralheavy andexactimplicit222mean unchanged. Phi/neutralheavy lift unchanged. Source/Z/receiver priorbytes untouched.',
        repaired_computational_failure='Actualeta7/20 constructor096971e09fbdc53bbd7b37f729e3833ed2f92bfaa053607705c415ace182b284 CLI naturalexit1: fixedstriprho1/2 didnotcertifypositivecomplexR². Newrho1/3 with48quadrature (NOTphysicalN) directlycertifiespositiveR²; samephysicalsource preserved. No guard suppression/fallback.',
        proof='Fusedliteralband+projectorjet intervalquadrature atMP65. BothphysicalN andquadrature48 tails priced fromactualpositive strip; no smallboxenergy promotion. ActualGauss wired/free bound pricesu2. FullquantumCAR/Coulomb via resistance metric/Parseval; aliasl2<=9N sqrt(3V)Mstrip exp(-rho N/2)<=2^-512 byexactguard exp(-rho N)<=2^-floor(rho N). OldstateenergyLOWER usesCpositive andallprimitivePi, NOToldUPPER. Newupper<oldlower provessameHref improvement.',
        source_domain_extension='Actualnewchargedeta7/20 seed isPhi-independent finiteCAR; Gaussianvariance3/5 Schwartz. Same actualPhi cutoffb=d_+²/analyticheavy/realGauss lift as0a825; its H2/commoncore/globalquartic weak-domain argument applies structurally toTHISstate, NOTclaimD(H²). DisplayedfullHthirdcandidate requiresparentuniformgraph/initialPhicomparisonreview; correctedphysicalA/a² conversion paid.',
        limitations='SameactualmacroHref materialenergygain/Coulomb/contact, notstationarity/fullPhi ground/photonvacuum/pole/residue/commoncone/4096clock/Compton. Contactnotparticle; sourcechangedphysically andexplicitly.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('prior','circulation','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.prior,args.circulation);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','energy_per_V_upper','prior_same_finite_Href_quantum_source_energy_per_V_lower','same_finite_Href_gain_per_V_lower','full_quantum_Coulomb_per_V_upper','reference_dilation_force_per_V_interval','elapsed_seconds')},indent=2))

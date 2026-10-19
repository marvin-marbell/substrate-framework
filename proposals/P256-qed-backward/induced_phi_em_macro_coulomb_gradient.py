"""Actual finite macro Slater/Gauss Coulomb bound from resistance and projector jets."""
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
from induced_phi_em_macro_photon_domain import SOURCE_BYTES as DOMAIN_BYTES, PLAN, REVISION
from induced_phi_em_macro_preparation import scalars, build_native, centre, cosh, sinh
from induced_phi_em_gain_certificate import IV, iv, endpoints, encoded
from induced_phi_ir_well_transfer import rational, sqrt_bounds

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**DOMAIN_BYTES,Path(__file__).name:Path(__file__).read_bytes()}


def profile_gradient(coeff: dict[str,Any],g0: Any,gs: Any,eta: F,N: int,n: int) -> dict[str,Any]:
    rho=F(1,2);mass=coeff['onsite'];paths=coeff['path'];load=coeff['load']
    X=iv(mass)+6*iv(eta)*sum((iv(abs(paths[l][0]))*cosh(l*iv(rho)) for l in (1,2)),iv(0))
    Y=2*iv(eta)*sum((iv(abs(paths[l][1]))*cosh(l*iv(rho)) for l in (1,2)),iv(0))
    xp=2*iv(eta)*sum((l*iv(abs(paths[l][0]))*cosh(l*iv(rho)) for l in (1,2)),iv(0))
    yp=2*iv(eta)*sum((l*iv(abs(paths[l][1]))*cosh(l*iv(rho)) for l in (1,2)),iv(0))
    real_mass=IV.sqrt(g0)*(iv(mass)-6*iv(eta)*sum((iv(abs(paths[l][0]))*cosh(l*iv(rho)) for l in (1,2)),iv(0)))
    imaginary_mass=6*iv(eta)*IV.sqrt(g0)*sum((iv(abs(paths[l][0]))*sinh(l*iv(rho)) for l in (1,2)),iv(0))
    imaginary_space=2*iv(eta)*IV.sqrt(gs)*sum((iv(abs(paths[l][1]))*sinh(l*iv(rho)) for l in (1,2)),iv(0))
    rad=real_mass**2-imaginary_mass**2-3*imaginary_space**2;rad_lower=endpoints(rad)[0]
    if endpoints(real_mass)[0]<=0 or rad_lower<=0:raise ArithmeticError('Actual projector analytic strip is not certified')
    den=iv(rad_lower);modulus=3*(g0*xp*xp+gs*yp*yp)/den+3*(g0*X*xp+gs*Y*yp)**2/(den*den)
    projector_modulus=endpoints(1+(IV.sqrt(g0)*X+3*IV.sqrt(gs)*Y)/IV.sqrt(den))[1]
    trig={j:{l:(IV.cos(l*(2*j+1)*IV.pi/n),IV.sin(l*(2*j+1)*IV.pi/n)) for l in (1,2)} for j in range(n)}
    total=iv(0)
    for index in itertools.product(range(n),repeat=3):
        x=iv(mass)+2*iv(eta)*sum((iv(paths[l][0])*trig[j][l][0] for j in index for l in (1,2)),iv(0))
        y=[-2*iv(eta)*sum((iv(paths[l][1])*trig[j][l][1] for l in (1,2)),iv(0)) for j in index]
        dx=[-2*iv(eta)*sum((l*iv(paths[l][0])*trig[j][l][1] for l in (1,2)),iv(0)) for j in index]
        dy=[-2*iv(eta)*sum((l*iv(paths[l][1])*trig[j][l][0] for l in (1,2)),iv(0)) for j in index]
        R2=g0*x*x+gs*sum((v*v for v in y),iv(0))
        total+=(g0*sum((v*v for v in dx),iv(0))+gs*sum((v*v for v in dy),iv(0)))/R2-sum(((g0*x*a+gs*b*c)**2 for a,b,c in zip(dx,y,dy)),iv(0))/(R2*R2)
    qlo,qhi=endpoints(total/(n**3));aliasprice=endpoints(modulus*((1+2*IV.exp(-iv(rho)*n)/(1-IV.exp(-iv(rho)*n)))**3-1))[1]
    dlo=qlo-aliasprice;dhi=qhi+aliasprice
    if dhi<=0:raise ArithmeticError('No positive exact continuous projector jet bound')
    # Actual anti-N Fourier aliases: weighted l2 norm <=9N sqrt(3V) M exp(-N/4).
    # Prove conservative 2^-512 directly with exact rational squared bounds; never stringify enormous tails.
    V=N**3;alias_l2=F(1,2**512)
    if N%4 or F((9*N)**2*3*V)*projector_modulus**2/F(2**(N//2))>alias_l2**2:raise ArithmeticError('Actual finite-N projector alias witness fails')
    weighted_moment_upper=(sqrt_bounds(dhi)[1]+alias_l2)**2
    return dict(load=load,continuous_projector_jet_squared_interval=[encoded(dlo),encoded(dhi)],quadrature_alias_price=encoded(aliasprice),
        actual_finite_N_weighted_alias_l2_upper=encoded(alias_l2),strip_Re_R2_lower=encoded(rad_lower),projector_strip_Frobenius_upper=encoded(projector_modulus),
        actual_finite_N_weighted_projector_moment_upper=encoded(weighted_moment_upper))


def run(source_path: Path) -> dict[str,Any]:
    started=time.monotonic();raw=source_path.read_bytes();source=json.loads(raw);state=source['input_descriptor'];inputs={}
    for role in ('mean','refiner'):
        spec=state['inputs'][role];data=(ROOT/spec['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=spec['sha256']:raise ValueError('Frozen actual-mean input changed')
        inputs[role]=json.loads(data)
    IV.dps=65;mids,errors=centre(inputs['mean'],inputs['refiner']);g0=iv(mids[0])+IV.mpf([-1,1])*iv(errors[0]);gs=iv(mids[1])+IV.mpf([-1,1])*iv(errors[1])
    N=int(state['N']);V=int(state['V']);eta=rational(state['eta_preparation_ONLY']);native=build_native(inputs['mean'],2);coeffs=[scalars(p) for p in native['Pprofiles']];n=32
    rows=[profile_gradient(c,g0,gs,eta,N,n) for c in coeffs];g=F(float(state['UE']))/F(float(state['a']))
    upper=g*F(V-1,6*V)*sum((row['load']*rational(row['actual_finite_N_weighted_projector_moment_upper']) for row in rows),F(0));old_upper=rational(source['full_quantum_Coulomb_per_V_upper'])
    if upper>=old_upper:raise ArithmeticError('New microscopic Coulomb bound didnotimproveactualsource certificate')
    energy_upper=rational(source['energy_per_V_upper'])-old_upper+upper
    descriptor=dict(source_sha256=hashlib.sha256(raw).hexdigest(),source_input_sha256=source['input_sha256'],N=N,V=V,M=int(state['M']),eta_preparation_ONLY=encoded(eta),variance=state['variance'],a=state['a'],UE=state['UE'],charged_counts=[1,63],quadrature_grid=n,
        plan=PLAN,plan_revision=REVISION,inherited_source_plan_revision=state['plan_revision'])
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        actual_full_quantum_Coulomb_per_V_upper=encoded(upper),prior_full_Coulomb_per_V_upper=encoded(old_upper),actual_same_source_Href_energy_per_V_upper=encoded(energy_upper),profiles=rows,
        physical_state_unchanged=True,
        exact_finite_Gauss_proof='Actualindependent1+63 Slaters eachTrP(k)=2; Ref4/background cancel exactly, heavyEMneutral, henceconditionalmeanrho0. C/V=g sum_role load sum_R(G0-GR)||P_R||F² usingP²=P, includesALLq!=0/Coulomb. Effective resistance ismetric: voltageforb→c maximizedatb impliescrossenergy for a→b plusb→c <=0, provingtriangle. Cubictorus Foster/symmetry givesnearestR=(V-1)/(3V). ThusG0-GR<=((V-1)/(6V))||R||1<=((V-1)/(6V))sumRj² forintegerminimalrepresentatives. No inverse-lap singularity discarded.',
        projector_and_alias_proof='ActualClifford P=(I-h_eta/R)/2 has||partial_jP||F²=|partial_j n|². Exactsumj formula dprime²/R²-(d.dprime)²/R^4 integratedovercontinuousBrillouintorus equals sum_s |s|²||P_s||F² byParseval. MP65 actualrootenclosures andcomplexstrippositiveReR² price quadrature32→continuousintegral. ActualN anti222 DFT coefficients= sum_z(-1)^sumz P_(R+Nz); stripPcoef<=M exp(-rho|s|1). Forrho1/2 andminimalR, alias sum<=18M exp(-N/4), weightedl2<=9N sqrt(3V)M exp(-N/4)<=2^-512 byexecutedexactrational guard. Triangle inweightedl2 givesactualfiniteN moment<= (sqrt(Dupper)+2^-512)^2. Neveridentifygrid32asphysicalN orpromotesmallbox source. NativeN2 extractsONLYlocalphysicalpathcoefficientsatfixeda.',
        limitations='Sharper actualsame-source quantumHref energy/Coulomb envelope, notfullPhi energy, stationarity/photonvacuum/pole/commoncone/Compton. Parentresistance/Parseval/alias reviewrequired.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('source','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.source);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','actual_full_quantum_Coulomb_per_V_upper','prior_full_Coulomb_per_V_upper','actual_same_source_Href_energy_per_V_upper','elapsed_seconds')},indent=2))

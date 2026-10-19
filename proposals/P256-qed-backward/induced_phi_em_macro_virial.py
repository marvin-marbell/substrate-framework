"""Actual macro EM dilation defect; no low-energy-to-vacuum promotion."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_macro_preparation import SOURCE_BYTES as MACRO_BYTES, scalars, build_native, centre
from induced_phi_em_gain_certificate import IV, iv, endpoints, encoded
from induced_phi_ir_well_transfer import rational, sqrt_bounds

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**MACRO_BYTES,Path(__file__).name:Path(__file__).read_bytes()}


def run(source_path: Path) -> dict[str,Any]:
    started=time.monotonic();raw=source_path.read_bytes();source=json.loads(raw);state=source['input_descriptor']
    if state['kappa']!=0:raise ValueError('This exact dilation discriminator requires the actual constant-Theta charged seed')
    N=int(state['N']);V=int(state['V']);M=int(state['M']);s=rational(state['variance']);K=rational(source['primitive_electric_per_V_exact'])
    IV.dps=65;lo1,hi1=(rational(value) for value in source['interval_matter_components'][1]);lo2,hi2=(rational(value) for value in source['interval_matter_components'][2])
    if hi1>=0 or hi2>=0:raise ArithmeticError('Actual primitive expectations do not meet the displayed force-sign bound')
    u1=F(2,3)+F(1,3*V);weight=iv(s*u1)*IV.exp(-iv(s*u1)/2)
    one=-(iv(lo1)+IV.mpf([0,1])*iv(hi1-lo1))*weight
    # u2 exp(-s u2/2) is nonnegative and <=2 for exactPT u2 in[0,2].
    force=iv(-2*K)+one+IV.mpf([0,1])*iv(-2*s*lo2)
    flo,fhi=endpoints(force)
    # Conservative all-real-field current price, including state-norm comparison.
    action=rational(source['Phi_transfer']['all_flavour_current_action_error']);norm=rational(source['Phi_transfer']['physical_state_norm_distance_upper'])
    sigma_upper=sqrt_bounds(s)[1];bound_inputs={}
    for role in ('mean','refiner'):
        spec=state['inputs'][role];data=(ROOT/spec['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=spec['sha256']:raise ValueError('Scalar-path price input changed since macro source')
        bound_inputs[role]=json.loads(data)
    mids,errors=centre(bound_inputs['mean'],bound_inputs['refiner'])
    frame_upper=[sqrt_bounds(mid+error)[1] for mid,error in zip(mids,errors)]
    coeffs=[scalars(profile) for profile in build_native(bound_inputs['mean'],2)['Pprofiles']]
    # Literal CAR trace-norm perorientedhop <=4||hop||; HC doubles.
    # All3V axial paths: sum|S_l f|<=sqrt(3V)||S_l f||<=l sqrt(3V).
    Jcap=8*sqrt_bounds(F(3*V))[1]*sum((c['load']*sum((l*(abs(pair[0])*frame_upper[0]+abs(pair[1])*frame_upper[1]) for l,pair in c['path'].items()),F(0)) for c in coeffs),F(0))
    force_price=F(M,V)*sigma_upper*(4*action+4*norm*Jcap)
    full_lower=flo-force_price;full_upper=fhi+force_price
    if full_lower<=0:raise ArithmeticError('Actual full-Phi price does not discriminate stationarity')
    variance_A=F(M,2);variance_H_per_V_lower=F(V)*full_lower**2/(4*variance_A)
    descriptor=dict(source_sha256=hashlib.sha256(raw).hexdigest(),source_input_sha256=source['input_sha256'],N=N,V=V,M=M,
        variance=encoded(s),eta_preparation_ONLY=state['eta_preparation_ONLY'],charged_counts=[1,63],
        plan=state['plan'],plan_revision=state['plan_revision'])
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        reference_EM_dilation_force_per_V_interval=[encoded(x) for x in (flo,fhi)],full_Phi_EM_dilation_force_per_V_interval=[encoded(full_lower),encoded(full_upper)],
        full_Phi_force_transfer_price_per_V=encoded(force_price),literal_all64_current_operator_norm_upper=encoded(Jcap),exact_dilation_variance=encoded(variance_A),
        full_H_energy_variance_per_V_lower=encoded(variance_H_per_V_lower),stationarity_discriminator='FAIL: thisactualnormalizedlowenergyquantumsource cannotbestationary/eigenstate. No photonvacuum claim.',
        exact_operator='A=(1/2)sum_a(Theta_a Pi_a+Pi_a Theta_a) overallMrealphysicalcoordinates. DGaussA0 becauseDPT0; AcommutesCARrho andSU4/Ref charges. SourceVarA=M/2 exact. -i<[Href,A]>=-2<K>+<Theta.J>; productGauss integrationgives/V -2K/V-s sum_l u_l exp(-s u_l/2) E_l/V. CoulombretainedinHrefbutcommutesA. SourcephysicalPhi/colourotherterms haveNOcoexactPi andareindependentTheta excepttheliteralchargedPeierls terms, sojoinedforce usesfullPhiJ.',
        transfer_proof='Initialfull/sourcecomparison sharetheidenticalrealEMGaussian andPhi-independentchargedstate; physicalGauss/frame liftcommutescoexactTheta. Thus||Theta_a deltaPsi||=sqrt(s)||deltaPsi||. CauchySchwarz percoordinate andALLMsum givesforceprice/V <=(M/V)sqrt(s)*4(action_current+norm_distance*Jcap). Actualscalarpaths giveJcap=8sqrt(3V)sum_role load sum_l l(|F0_l|sqrtg0_upper+|Fj_l|sqrtgs_upper): CARtrace-normperhop<=4||hop||, HCdouble, CauchySchwarz/moving-sum norm<=l fornormalizedrealprobe. N2 onlyextractsLOCALHamiltoniancoefficientsatfixeda, neverpromotesN2state. Parentcanonicalblock/initialfibre/actioncapreviewrequired.',
        Robertson='Var(Hfull)*Var(A)>=|<[Hfull,A]>|²/4. Recordedforceinterval excludeszero, henceVar(Hfull)/V>=V*force_perV_lower²/(4 VarA). Lowmeanenergy neverimpliesstationarity.',
        limitations='Thisisadiscriminator, notstationaryrepair orphotonpole/commoncone/Comptonadmission. FullPhioperator/domain/currentpricecomparison requiresparentreview; referenceHref forceinterval isdirectactualmacrocertificateconsumption.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('source','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.source);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','reference_EM_dilation_force_per_V_interval','full_H_energy_variance_per_V_lower','stationarity_discriminator','elapsed_seconds')},indent=2))

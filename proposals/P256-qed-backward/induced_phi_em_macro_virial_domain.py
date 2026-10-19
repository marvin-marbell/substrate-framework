"""Successor of withdrawn A-unit virial extension; actual full-H source discriminator."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_macro_photon_domain import SOURCE_BYTES as DOMAIN_BYTES, PLAN, REVISION
from induced_phi_em_gain_certificate import IV, iv, endpoints, encoded
from induced_phi_ir_well_transfer import rational, sqrt_bounds

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**DOMAIN_BYTES,Path(__file__).name:Path(__file__).read_bytes()}


def run(source_path: Path,domain_path: Path,reference_cap_path: Path) -> dict[str,Any]:
    started=time.monotonic();raw=source_path.read_bytes();source=json.loads(raw);state=source['input_descriptor'];domainraw=domain_path.read_bytes();domain=json.loads(domainraw)
    capraw=reference_cap_path.read_bytes();cap=json.loads(capraw)
    if domain['input_descriptor']['source_sha256']!=hashlib.sha256(raw).hexdigest() or state['N']!=cap['input_descriptor']['N'] or state['eta_preparation_ONLY']!=cap['input_descriptor']['eta_preparation_ONLY']:raise ValueError('Source/domain/local-current-cap binding mismatch')
    V=int(state['V']);M=int(state['M']);s=rational(state['variance']);K=rational(source['primitive_electric_per_V_exact']);a=F(float(state['a']));IV.dps=65
    lo1,hi1=(rational(value) for value in source['interval_matter_components'][1]);lo2,hi2=(rational(value) for value in source['interval_matter_components'][2])
    if state['kappa']!=0 or hi1>=0 or hi2>=0:raise ValueError('Discriminator requires actualconstantThetaSlater withcertifiednegativehoppings')
    u1=F(2,3)+F(1,3*V);one=(iv(lo1)+IV.mpf([0,1])*iv(hi1-lo1))
    flo,fhi=endpoints(iv(-2*K)-iv(s*u1)*IV.exp(-iv(s*u1)/2)*one+IV.mpf([0,1])*iv(-2*s*lo2))
    action=rational(source['Phi_transfer']['all_flavour_current_action_error'])/a;norm=rational(source['Phi_transfer']['physical_state_norm_distance_upper']);Jcap=rational(cap['literal_all64_current_operator_norm_upper'])
    if action!=rational(domain['corrected_current_action_error_Theta']):raise ArithmeticError('RepairedphysicalA→Theta current ABI mismatch')
    price=F(M,V)*sqrt_bounds(s)[1]*4*(action+norm*Jcap);full=(flo-price,fhi+price)
    distance=min(abs(x) for x in full)
    if full[0]<=0<=full[1]:raise ArithmeticError('Actualfull-Hforceintervaldoesnotdiscriminatestationarity')
    variance_A=F(M,2);variance_H_per_V=F(V)*distance**2/(4*variance_A)
    descriptor=dict(source_sha256=hashlib.sha256(raw).hexdigest(),source_input_sha256=source['input_sha256'],domain_sha256=hashlib.sha256(domainraw).hexdigest(),reference_local_current_cap_sha256=hashlib.sha256(capraw).hexdigest(),
        N=state['N'],V=V,M=M,variance=encoded(s),eta_preparation_ONLY=state['eta_preparation_ONLY'],a=encoded(a),charged_counts=[1,63],plan=PLAN,plan_revision=REVISION,inherited_source_plan_revision=state['plan_revision'])
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        reference_dilation_force_per_V_interval=[encoded(flo),encoded(fhi)],corrected_full_H_dilation_force_per_V_interval=[encoded(x) for x in full],
        corrected_full_Phi_current_action_price_Theta=encoded(action),current_A_to_Theta_unit_factor=encoded(1/a),corrected_force_transfer_price_per_V=encoded(price),
        exact_physical_dilation_variance=encoded(variance_A),corrected_full_H_energy_variance_per_V_lower=encoded(variance_H_per_V),
        source_is_not_stationary=True,
        proof='Canonicalgaugepreserving A=1/2sum_ALLM(ThetaPi+PiTheta) preservesrealGauss becauseDPT0 andcommutesCARrho/SU4/Ref charges. PsiA isquadraticEMGaussian×samePhi/chargedfibre; DOMAINsuccessor simultaneousweightedSchwartz/H2Phi coreapprox extendsalsoApsi∈D(H),AHpsi/HApsi finite. FullHcommA=-2iK+iTheta.J_TOTAL, noPhi/SU4/Coulforce dropped; initialVarA=M/2 exact. ReferenceGaussianintegration givesdisplayedforce. Actualinitialembeddingcomparison sharesidenticalEMGaussian/Phi-independentchargedfibre, hence||Theta_m deltaPsi||=sqrt(s)||deltaPsi||. CauchySchwarz ALLM timescorrectedRAWTheta current-action price plusnormdifference×literalJcap givesdisplayedfullprice. OriginalphysicalA errorDIVIDEDby a. Robertson usesPsi∈D(H) andApsi∈L2, NOT Psi∈D(H²).',
        withdrawn_epoch='Old77908875 fullPhi virialpriceextension WITHDRAWN becausephysicalA currenterror wasunconverted; itsREFERENCE force andliteralcurrentcap remainseparatelyvalid. Oldsource/receiptbytes untouched.',
        limitations='Actualfull-Hstatdiscriminatorproposalwithunits/domainpaid,pendingparentcomparisonreview. No ground/phvac/pole/residue/commoncone/Comptonadmission andnofiniteclockpropagator. ReferenceunitaryevolutionisnotretaggedasfullH.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('source','domain','reference-cap','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.source,args.domain,args.reference_cap);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','current_A_to_Theta_unit_factor','corrected_full_H_dilation_force_per_V_interval','corrected_full_H_energy_variance_per_V_lower','elapsed_seconds')},indent=2))

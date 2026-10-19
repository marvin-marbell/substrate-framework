"""New macro source width; same physical H and charged Slater, no vacuum promotion."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_macro_virial import SOURCE_BYTES as VIRIAL_BYTES
from induced_phi_em_gain_certificate import IV, iv, endpoints, encoded
from induced_phi_ir_well_transfer import rational

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**VIRIAL_BYTES,Path(__file__).name:Path(__file__).read_bytes()}


def run(source_path: Path) -> dict[str,Any]:
    started=time.monotonic();raw=source_path.read_bytes();prior=json.loads(raw);old=prior['input_descriptor']
    if rational(old['variance'])!=1 or old['kappa']!=0:raise ValueError('Exact same-seed width comparison requires the frozen sigma1/kappa0 source')
    IV.dps=65;s=F(3,4);N=int(old['N']);V=int(old['V']);M=int(old['M']);Kold=rational(prior['primitive_electric_per_V_exact']);Knew=Kold/s
    pairs=[tuple(rational(x) for x in p) for p in prior['interval_matter_components']]
    values=[iv(lo)+IV.mpf([0,1])*iv(hi-lo) for lo,hi in pairs]
    if pairs[1][1]>=0 or pairs[2][1]>=0:raise ArithmeticError('Certified axial hopping sign changed')
    u1=F(2,3)+F(1,3*V);f1=IV.exp(-iv(s*u1)/2);oldf1=IV.exp(-iv(u1)/2)
    # Every physical two-link norm u2 in[0,2]. Smaller width increases exp(-s*u2/2).
    # I2<0 therefore its energy gain is nonnegative; <=|I2| is conservative.
    gain=iv(Kold-Knew)+(oldf1-f1)*values[1]+IV.mpf([0,1])*iv(-pairs[2][0])
    gainlo,gainhi=endpoints(gain)
    if gainlo<=0:raise ArithmeticError('Squeezed true quantum source has no certified same-H energy improvement')
    upper=rational(prior['energy_per_V_upper'])-gainlo
    def force(width: F) -> tuple[F,F]:
        forcevalue=iv(-2*Kold/width)-iv(width*u1)*IV.exp(-iv(width*u1)/2)*values[1]+IV.mpf([0,1])*iv(-2*width*pairs[2][0])
        return endpoints(forcevalue)
    flo,fhi=force(s);bracket_upper=F(4,5);blo,bhi=force(bracket_upper)
    if fhi>=0 or blo<=0:raise ArithmeticError('Displayed variational-width virial bracket is not certified')
    contact=-(f1*values[1]+4*(IV.exp(-iv(s))+IV.mpf([0,1])*(1-IV.exp(-iv(s))))*values[2])/3
    clo,chi=endpoints(contact);g=F(float(old['UE']))/F(float(old['a']));photon=[encoded(4*g*g*x) for x in (clo,chi)]
    # Exact same-source fullPhi price remains old universal graph: Pi2/4 below192/129024.
    desc=dict(old);desc['variance']=encoded(s);desc['prior_source_sha256']=hashlib.sha256(raw).hexdigest();desc['prior_input_sha256']=prior['input_sha256']
    normA=F(M,2);variance_ref_lower=F(V)*fhi*fhi/(4*normA)
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),
        energy_per_V_upper=encoded(upper),same_finite_Href_onsiteCAR_sigma4_energy_per_V_lower=prior['same_finite_Href_onsiteCAR_sigma4_energy_per_V_lower'],
        actual_macro_gain_per_V_lower=encoded(rational(prior['same_finite_Href_onsiteCAR_sigma4_energy_per_V_lower'])-upper),
        same_finite_Href_gain_against_sigma1_per_V_interval=[encoded(gainlo),encoded(gainhi)],
        primitive_electric_per_V_exact=encoded(Knew),full_quantum_Coulomb_per_V_upper=prior['full_quantum_Coulomb_per_V_upper'],
        interval_matter_components=prior['interval_matter_components'],Phi_transfer=prior['Phi_transfer'],
        exact_dilation_variance=encoded(normA),reference_dilation_force_per_V_interval=[encoded(flo),encoded(fhi)],
        reference_energy_variance_per_V_lower=encoded(variance_ref_lower),reference_unique_variational_width_root_bracket=[encoded(s),encoded(bracket_upper)],
        upper_bracket_force_per_V_interval=[encoded(blo),encoded(bhi)],literal_reference_transverse_contact_interval=[encoded(clo),encoded(chi)],
        reference_photon_first_retarded_derivative=encoded(-2*g),reference_photon_third_retarded_derivative_interval=photon,
        state='NEW actualN262144 pure222 allMrealGaussian variance3/4 x SAME fixedcentreeta1/5 Slater/actualmean/1+63/Ref4NET0/neutralheavy/projectedPhi lift. No changesphysicalHamiltonian/spacing/masses orfrozenpriors.',
        proof='PhysicalCoulomb/allPhi/colour/neutralheavy/onsite expectations EXACTLYCANCEL between widths becausecharged/Phi fibre unchanged andcoexactEMGaussianonlychanged; notCoulomb omitted. PrimitivePi exactKold/s. Primitivepath n1=2/3+1/(3V), n2in[0,2]. I1,I2actualmacrointervalsnegative; two-linkcontributiontosame-Hgain nonnegative. Newupper=oldactualquantumupper-certpositivegain. On[.75,.8] eachdilationforce derivativepositive: 2Kold/s² andnegativehoppingtimes u exp(-su/2)(1-su/2)>=0, henceexactreferenceunique widthcriticalpointinbracket. Widthcriticalpointalone isNOTstationaryquantumstate.',
        Phi_transfer_reason='Charged/Phi fibre andALLphysicalcharges unchanged; GaussianPi2=1/(4s),Pi4=3/(16s²) stillbelowold192/129024; universalcharged/Coulombgraph unchanged. ParentPhi/current/domaincomparisonreviewrequired. Photon derivative/contact displayedREFERENCE only; frozen macro-photon consumer can separatelyprice fullPhi.',
        limitations='Squeezedsource isSTILL NONSTATIONARY: forceintervalstrictnegative andRobertsonlowerpositive. Widthbracket isonlyonevariationaldegree, neverfullvacuum/pole/residue/commoncone/finiteclock4096/Compton/QEDadmission.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('source','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.source);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','energy_per_V_upper','same_finite_Href_gain_against_sigma1_per_V_interval','reference_dilation_force_per_V_interval','reference_unique_variational_width_root_bracket','elapsed_seconds')},indent=2))

"""Literal transverse contact and photon causal third moment on the macro source.

No band propagator, preparation-gap pole, stationarity or finite-clock claim.
The state-dependent coefficient is consumed from certified physical-Href
Peierls expectations, with the joined full-Phi contact price retained.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_macro_preparation import SOURCE_BYTES as MACRO_BYTES
from induced_phi_em_gain_certificate import IV, iv, endpoints, encoded
from induced_phi_ir_well_transfer import rational

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**MACRO_BYTES,Path(__file__).name:Path(__file__).read_bytes()}


def run(source_path: Path) -> dict[str,Any]:
    started=time.monotonic();raw=source_path.read_bytes();source=json.loads(raw);state=source['input_descriptor']
    if state['kappa']!=0 or state['charged_counts']!=[1,63]:raise ValueError('This consumer requires the actual fixed-centre all64 Gaussian source')
    N=int(state['N']);V=int(state['V']);variance=rational(state['variance']);g=F(float(state['UE']))/F(float(state['a']))
    IV.dps=65;components=[]
    for row in source['interval_matter_components']:
        lo,hi=(rational(value) for value in row);components.append(iv(lo)+IV.mpf([0,1])*iv(hi-lo))
    s1=IV.exp(-iv(variance)*iv(F(2,3)+F(1,3*V))/2)
    s2=IV.exp(-iv(variance))+IV.mpf([0,1])*(1-IV.exp(-iv(variance)))
    contact=-(s1*components[1]+4*s2*components[2])/3;clo,chi=endpoints(contact)
    # Conservative real(+q,-q) four-entry price, never dropping cross entries.
    phi_price=4*rational(source['Phi_transfer']['all_flavour_contact_expectation_error'])
    full_contact=(clo-phi_price,chi+phi_price);third=[4*g*g*x for x in full_contact]
    odd_trial=[g/(2*variance)+variance*x for x in (clo,chi)]
    descriptor=dict(source_sha256=hashlib.sha256(raw).hexdigest(),source_input_sha256=source['input_sha256'],N=N,V=V,
        physical_probe='Normalizedrealcos(2pi*x/N) y-polarization; exactDf0, all64literalone/two-linkPeierlscurrents',
        physical_q='(2pi/(N*a),0,0)',a=state['a'],UE=state['UE'],variance=encoded(variance),
        plan=state['plan'],plan_revision=state['plan_revision'])
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        reference_literal_contact_interval=[encoded(x) for x in (clo,chi)],full_Phi_literal_contact_interval=[encoded(x) for x in full_contact],
        full_Phi_contact_transfer_price=encoded(phi_price),reference_photon_retarded_first_derivative=encoded(-2*g),
        reference_photon_retarded_second_derivative=encoded(F(0)),full_Phi_contact_consumed_third_moment_interval=[encoded(x) for x in third],
        photon_exact_functional='D_R(t)=-i theta(t)Tr(rho_eta [Theta_f(t),Theta_f(0)]), Theta_f(t)=exp(iHref t)Theta_f exp(-iHref t). SAMEactualnormalizedmacrostate andphysicalHref, notpreparationh_eta.',
        reference_exact_moment_proof='Theta_dot=2gPi_f, Theta_ddot=-2gJ_f. Matter/Coulomb multiplyTheta so theircommutatorswithTheta vanish, butareNOTremovedfromHref. [K,J]=-ig{Pi,gradJ}; [[Href,J],Theta_f]=-2gContact_f. ThereforeD_Rprime(0)=-2g, D_Rsecond(0)=0, D_Rthird(0)=4g²<Contact_f>. LiteralContact_f=-(s1*E1/V+4*s2*E2/V)/3 bycubictranslationandtwo-linkderivative2². TheseareactualfullHrefinitialmoments, NOTstationaryRPA vertices.',
        actual_simple_odd_photon_trial_energy_shift_interval=[encoded(x) for x in odd_trial],
        odd_trial_proof='TheactualCoddtrial Theta_fPsi hasnorm²variance. ForconstantchargedSlater andrealGaussian, electricshiftg/(2variance) andmatterweightedGaussian shiftvariance*Contact_f; GaussCoulombcommutesTheta_f andcancelsinthisdifference. Itisavariationaltrial ABOVEthenongroundpreparedenergy, NOTphoton-gaplowerbound/pole ormasslessnessdisproof.',
        joined_consumer_scope='FullPhi physicalcontact price isapplied tocomputedstate-dependentcoefficient. OriginalcanonicalU1coexactblock hasHfull=gPi_Q²+Hother with[Hother,Theta_f]=0: CartesianPhi/SU4colour kineticandGaussaffinecharges containnocoexactPi_Q; U1longitudinalGaussenergyisrhoL+rho. Also[J_f,Theta_f]=0. Jacobi therefore[[Hother,J_f],Theta_f]=0 evenwhen[Hother,J_f]containsPhi/colour momenta. ThethirdformulaextendstoFULLoriginalHunderthiscanonicalblock/domainreview, notbydroppingthosekinetics. Displayedintervalisacandidatefull-HinitialmomentwithactualPhierrorpaid, pendingparentoperator/domainreview.',
        limitations='No fullfiniteclock4096 propagation, stationarypole/residue, photonvacuum/commoncone orCompton/QEDpromotion. Simpleoddtrialcost isnotanactualphotonpole. Sourceenergycertificate andPhi comparison stillrequireparentmathematicalreview.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},
        runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('source','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.source);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','reference_literal_contact_interval','reference_photon_retarded_first_derivative','full_Phi_contact_consumed_third_moment_interval','actual_simple_odd_photon_trial_energy_shift_interval','elapsed_seconds')},indent=2))

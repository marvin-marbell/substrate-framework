"""New all64 prepared charged alignment, actual macro quantum Gauss energy/current."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_macro_coulomb_gradient import SOURCE_BYTES as GRADIENT_BYTES, profile_gradient
from induced_phi_em_macro_photon_domain import PLAN
from induced_phi_em_macro_preparation import quadrature, scalars, build_native, centre
from induced_phi_em_gain_certificate import IV, iv, endpoints, encoded
from induced_phi_ir_well_transfer import rational

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**GRADIENT_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION='b9e13b2491321aa804f7e52cf4ba3cb12c0a224717cb07b23e487afc0768463b'


def run(prior_path: Path,circulation_path: Path) -> dict[str,Any]:
    started=time.monotonic();raw=prior_path.read_bytes();prior=json.loads(raw);old=prior['input_descriptor'];craw=circulation_path.read_bytes();circ=json.loads(craw)
    if old['N']!=circ['input_descriptor']['N'] or old['charged_counts']!=[1,63] or old['kappa']!=0:raise ValueError('No physicalbox/count/source substitution')
    inputs={}
    for role in ('mean','refiner'):
        spec=old['inputs'][role];data=(ROOT/spec['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=spec['sha256']:raise ValueError('Actual mean epoch changed')
        inputs[role]=json.loads(data)
    N=int(old['N']);V=int(old['V']);M=int(old['M']);eta=F(7,20);s=F(3,5);g=F(float(old['UE']))/F(float(old['a']));n=32
    IV.dps=65;mids,errors=centre(inputs['mean'],inputs['refiner']);g0=iv(mids[0])+IV.mpf([-1,1])*iv(errors[0]);gs=iv(mids[1])+IV.mpf([-1,1])*iv(errors[1])
    coeffs=[scalars(p) for p in build_native(inputs['mean'],2)['Pprofiles']];energies=[quadrature(c,g0,gs,eta,n) for c in coeffs]
    gradients=[profile_gradient(c,g0,gs,eta,N,n) for c in coeffs]
    pairs=[(sum((row['components'][l][0] for row in energies),F(0)),sum((row['components'][l][1] for row in energies),F(0))) for l in range(3)]
    components=[iv(lo)+IV.mpf([0,1])*iv(hi-lo) for lo,hi in pairs]
    u1=F(2,3)+F(1,3*V);u2lo,u2hi=(rational(x) for x in circ['actual_macro_two_link_Q_norm_squared_interval']);u2=iv(u2lo)+IV.mpf([0,1])*iv(u2hi-u2lo)
    f1=IV.exp(-iv(s*u1)/2);f2=IV.exp(-iv(s)*u2/2)
    C=g*F(V-1,6*V)*sum((row['load']*rational(row['actual_finite_N_weighted_projector_moment_upper']) for row in gradients),F(0));K=g*F(M,4*V)/s
    upper=endpoints(components[0]+f1*components[1]+f2*components[2])[1]+C+K
    # SAMEHref oldsource lower: Cpositive, allprimitivePi retained, actualoldphase factors priced.
    sold=rational(old['variance']);oldcomponents=[iv(rational(row[0]))+IV.mpf([0,1])*iv(rational(row[1])-rational(row[0])) for row in prior['interval_matter_components']]
    oldlower=endpoints(oldcomponents[0]+IV.exp(-iv(sold*u1)/2)*oldcomponents[1]+IV.exp(-iv(sold)*u2/2)*oldcomponents[2])[0]+rational(prior['primitive_electric_per_V_exact'])
    gain=oldlower-upper
    if gain<=0:raise ArithmeticError('New actualcoupledmacrostate cannotcertifiablybeat SAMEHref precedingquantumstate')
    force=endpoints(iv(-2*K)-iv(s*u1)*f1*components[1]-iv(s)*u2*f2*components[2])
    contact=endpoints(-(f1*components[1]+4*f2*components[2])/3);price=4*rational(prior['Phi_transfer']['all_flavour_contact_expectation_error'])/(F(float(old['a']))**2)
    third=[4*g*g*x for x in (contact[0]-price,contact[1]+price)]
    desc=dict(old);desc.update(eta_preparation_ONLY=encoded(eta),variance=encoded(s),plan=PLAN,plan_revision=REVISION,
        inherited_source_plan_revision=old['plan_revision'],prior_source_sha256=hashlib.sha256(raw).hexdigest(),prior_input_sha256=prior['input_sha256'],actual_circulation_sha256=hashlib.sha256(craw).hexdigest())
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),
        energy_per_V_upper=encoded(upper),prior_same_finite_Href_quantum_source_energy_per_V_lower=encoded(oldlower),same_finite_Href_gain_per_V_lower=encoded(gain),
        full_quantum_Coulomb_per_V_upper=encoded(C),primitive_electric_per_V_exact=encoded(K),interval_matter_components=[[encoded(x) for x in p] for p in pairs],
        projector_gradient_profiles=gradients,actual_two_link_Q_norm_squared_interval=circ['actual_macro_two_link_Q_norm_squared_interval'],Phi_transfer=prior['Phi_transfer'],
        reference_dilation_force_per_V_interval=[encoded(x) for x in force],reference_literal_Theta_contact_interval=[encoded(x) for x in contact],
        corrected_four_entry_Theta_contact_price=encoded(price),corrected_full_H_contact_consumed_third_interval=[encoded(x) for x in third],
        state='NEWactualN262144/V2^54/allMrealGaussian variance3/5 x fixedcentrecanonicalnegativeSlater(onsite+(7/20)hop) active1m.2+aux63m2. PreparationONLY changed; originalphysicalH/a/UE/r/v/masses/Ref4NET0/neutralheavy andexactimplicit222mean unchanged. ActualPhi cut/projectedheavy lift unchanged. No retag prior sources/Z/receiver.',
        proof='SameactualperiodicfiniteN/root price frommacroquadrature32; newCoulombprojector-jet/Foster/Parseval/anti-Nalias bound includesALLquantumCAR/Coulomb. Actualwired/freeGauss-resistancecirc bound tightensphysicaltwo-linkGaussianfactor, allharmonics preserved. OldsourceLOWER usesCpositive (notoldUPPER), literalmatterinterval/Pi andsamephysicalphasefactors. Newupper<oldlower provesactualenergyimprovement, no comparison across sizes orHamiltonians.',
        source_domain_extension='Actualchargedseed isNEWcanonicalSlatereta7/20, butPhi-independentfiniteCAR vector; allcore/D(H)/ThetaPiJ Gaussianweighted convergence andglobalquartic/J-H-bounded weaklemma of0a825eaa applyunchanged toTHISvector. Phi b=d_+² andanalyticprojectedheavy/positivechart remainidentical; EMGaussian isSchwartz withvariance3/5. NoD(H²) assertion. FullH Dthird identity structuraloperator/domain route retained; correctedphysicalA contacterror/a² paid. NewsourcePhi normpriceinheritance stillrequiresparentuniform-graph/initialcomparisonreview.',
        full_H_functional='Dfull_R(t)=-i step(t)<Psi_new,[exp(iHfullt)Theta_fexp(-iHfullt),Theta_f]Psi_new>. Hrefreference isdistinct. Displayedthirdcandidate pendingparentPhi/domaincomparisonreview.',
        limitations='Newmateriallylower actualmacroquantumHref preparation, STILLnotstationary/fullground/photonvacuum/pole/residue/commoncone/finiteclock4096 orCompton/QEDadmission. Current/contact isnotparticle.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('prior','circulation','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.prior,args.circulation);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','energy_per_V_upper','prior_same_finite_Href_quantum_source_energy_per_V_lower','same_finite_Href_gain_per_V_lower','full_quantum_Coulomb_per_V_upper','reference_dilation_force_per_V_interval','elapsed_seconds')},indent=2))

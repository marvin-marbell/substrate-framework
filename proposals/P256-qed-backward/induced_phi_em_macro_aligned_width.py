"""NEW same physical eta7/20 macro source: electric-priced width relaxation."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_macro_aligned_certified import SOURCE_BYTES as PRIOR_BYTES
from induced_phi_em_gain_certificate import IV,iv,endpoints,encoded
from induced_phi_ir_well_transfer import rational
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**PRIOR_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION='b9e13b2491321aa804f7e52cf4ba3cb12c0a224717cb07b23e487afc0768463b'


def run(source_path: Path) -> dict[str,Any]:
    started=time.monotonic();raw=source_path.read_bytes();prior=json.loads(raw);old=prior['input_descriptor'];N=int(old['N']);V=int(old['V']);M=int(old['M']);s0=rational(old['variance']);s=F(14,25);g=F(float(old['UE']))/F(float(old['a']));IV.dps=65
    if N!=262144 or rational(old['eta_preparation_ONLY'])!=F(7,20) or s0!=F(3,5) or old['kappa']!=0 or old['charged_counts']!=[1,63]:raise ArithmeticError('No physicalsource/H substitution')
    components=[iv(rational(pair[0]))+IV.mpf([0,1])*iv(rational(pair[1])-rational(pair[0])) for pair in prior['interval_matter_components']];u1=iv(F(2,3)+F(1,3*V));u2lo,u2hi=(rational(x) for x in prior['actual_two_link_Q_norm_squared_interval']);u2=iv(u2lo)+IV.mpf([0,1])*iv(u2hi-u2lo);Kconst=g*F(M,4*V)
    def matter(width: F) -> Any:return components[0]+IV.exp(-iv(width)*u1/2)*components[1]+IV.exp(-iv(width)*u2/2)*components[2]
    def force(width: F) -> Any:return -iv(2*Kconst/width)-iv(width)*u1*IV.exp(-iv(width)*u1/2)*components[1]-iv(width)*u2*IV.exp(-iv(width)*u2/2)*components[2]
    # FullquantumC andonsite are EXACTLYunchanged: normalizedCARstate isTheta-independent. Gaininterval retains identicalcoefficient/factorcorrelations.
    gain=endpoints(iv(Kconst/s0-Kconst/s)+(IV.exp(-iv(s0)*u1/2)-IV.exp(-iv(s)*u1/2))*components[1]+(IV.exp(-iv(s0)*u2/2)-IV.exp(-iv(s)*u2/2))*components[2])
    if gain[0]<=0:raise ArithmeticError('New literalquantumwidth didnotbeat prior source')
    C=rational(prior['full_quantum_Coulomb_per_V_upper']);upper=endpoints(matter(s))[1]+Kconst/s+C
    left=F(11,20);right=F(57,100);fl=endpoints(force(left));fr=endpoints(force(right));width=iv(left)+IV.mpf([0,1])*iv(right-left)
    Esecond=2*iv(Kconst)/(width**3)+(u1*u1/4)*IV.exp(-width*u1/2)*components[1]+(u2*u2/4)*IV.exp(-width*u2/2)*components[2];convex=endpoints(Esecond)[0]
    if fl[1]>=0 or fr[0]<=0 or convex<=0:raise ArithmeticError('Actualfixedseed widthcritical bracket/uniqueness failed')
    f=endpoints(force(s));contact=endpoints(-(IV.exp(-iv(s)*u1/2)*components[1]+4*IV.exp(-iv(s)*u2/2)*components[2])/3);price=rational(prior['corrected_four_entry_Theta_contact_price']);third=[4*g*g*x for x in (contact[0]-price,contact[1]+price)]
    desc=dict(old);desc.update(variance=encoded(s),plan_revision=REVISION,inherited_source_plan_revision=old['plan_revision'],prior_source_sha256=hashlib.sha256(raw).hexdigest(),prior_input_sha256=prior['input_sha256'])
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),energy_per_V_upper=encoded(upper),same_finite_Href_prior_source_energy_gain_per_V_interval=[encoded(x) for x in gain],full_quantum_Coulomb_per_V_upper=encoded(C),primitive_electric_per_V_exact=encoded(Kconst/s),interval_matter_components=prior['interval_matter_components'],actual_two_link_Q_norm_squared_interval=prior['actual_two_link_Q_norm_squared_interval'],Phi_transfer=prior['Phi_transfer'],reference_dilation_force_per_V_interval=[encoded(x) for x in f],reference_literal_Theta_contact_interval=[encoded(x) for x in contact],corrected_four_entry_Theta_contact_price=encoded(price),corrected_full_H_contact_consumed_third_interval=[encoded(x) for x in third],unique_fixed_CAR_widthcritical_bracket=[encoded(left),encoded(right)],literal_energy_width_second_derivative_lower=encoded(convex),width_endpoint_force_intervals=[[encoded(x) for x in fl],[encoded(x) for x in fr]],
        state='NEWactualeta7/20 sourceGaussiansigma14/25, allMcoexact+harmonics withnormalizedfiniteCAR. Samechargedseed/Phi/heavy/all64/Ref4 andPHYSICALH/a/UE/r/v/masses/actualimplicit222mean. Widthchanges preparationONLY; nottimeevolution ofprior source.',
        proof='Gain interval uses SAMEfullHref C/onsiteexact cancellation, fullallMbarePi andidenticalPeierls coefficient/circulationfactor correlations. No oldUPPER vsnewUPPER comparison. Executedexactinterval signs force(.55)<0<force(.57), literalEsecondpositive onwholebracket =>unique WIDTHvariationalcriticalpoint forTHISfixedCARseed. Explicitnewwidth .56 isnotretaggedcriticalpoint/fullstationarity; fullH/Phi price remainsparentconditional.',
        limitations='ImprovedactualmacroquantumHref preparation, notground/stationarityoutsidewidthfamily/photonvacuum/pole/residue/commoncone/finite4096/Compton/statQ. Current/contactnotparticle. No claimactualroot equal.56.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('source','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.source);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({key:report[key] for key in ('input_sha256','energy_per_V_upper','same_finite_Href_prior_source_energy_gain_per_V_interval','reference_dilation_force_per_V_interval','unique_fixed_CAR_widthcritical_bracket','elapsed_seconds')},indent=2))

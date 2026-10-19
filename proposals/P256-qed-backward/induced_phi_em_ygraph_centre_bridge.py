"""NEW positive C222-to-C000 bridge for unchanged Y000/Phi222 preparation."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_ygraph_clock_transfer import SOURCE_BYTES as CLOCK_BYTES,checked,exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_macro_preparation import scalars,build_native
from induced_phi_ir_well_transfer import rational,sqrt_bounds
from induced_phi_em_gain_certificate import encoded
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**CLOCK_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION='b9e13b2491321aa804f7e52cf4ba3cb12c0a224717cb07b23e487afc0768463b'


def run(refiner_path: Path,mean_path: Path,energy_path: Path,conditioning_path: Path,contact_path: Path) -> dict[str,Any]:
    started=time.monotonic();paths=dict(refiner=refiner_path,mean=mean_path,energy=energy_path,conditioning=conditioning_path,contact=contact_path);raws={name:p.read_bytes() for name,p in paths.items()};caps={name:json.loads(raw) for name,raw in raws.items()};refiner=caps['refiner'];mean=caps['mean'];energy=caps['energy'];condition=caps['conditioning'];contact=caps['contact']
    if refiner['input_descriptor']['mean_receipt_sha256']!=hashlib.sha256(raws['mean']).hexdigest() or refiner['selected_candidate']['twist']!=[2,2,2]:raise ArithmeticError('Actual root selector/mean mismatch')
    _,initial=checked(energy['input_descriptor']['input_receipts']['initial']);_,domain=checked(initial['input_descriptor']['input_receipts']['domain']);ds=domain['input_descriptor'];_,pure=checked(ds['input_receipts']['purePhi']);ws=pure['input_descriptor']['input_receipts']['well'];well_raw=(ROOT/ws['path']).read_bytes()
    if hashlib.sha256(well_raw).hexdigest()!=ws['sha256'] or hashlib.sha256(well_raw).hexdigest()!=refiner['input_descriptor']['well_receipt_sha256']:raise ArithmeticError('Actual Phi carrier root/geometry mismatch')
    _,old_initial=checked(condition['input_receipts']['initial_Phi']);_,old_domain=checked(old_initial['input_descriptor']['input_receipts']['domain'])
    if old_domain['input_sha256']!=domain['input_sha256']:raise ArithmeticError('Cannot change prepared source')
    well=json.loads(well_raw);N=int(ds['N']);V=int(ds['V']);p=well['prices'];eta=rational(p['eta']);delta=rational(refiner['refined_all64_Gram_distance_to_exact000_upper']);enc=mean['centre_enclosure'];convexity=rational(mean['actual_finite_grid_quadrature_certificate']['local_coupled_convexity_lower'])
    if N!=262144 or refiner['input_descriptor']['finite_size']!=N or V!=N**3 or ds['M']!=2*V+1 or refiner['input_descriptor']['Lambda_exponent']!=8192 or convexity<127:raise ArithmeticError('Physicalbox/law/root uniqueness changed')
    gmin=min(rational(enc['g0_lower']),rational(enc['gs_lower']))
    if gmin-delta<F(1,16) or delta<=0:raise ArithmeticError('SPD source/ref square-root bridge invalid')
    frame=2*delta;phi_error=2*eta+frame;root0=sqrt_bounds(rational(enc['g0_upper']))[1];roots=sqrt_bounds(rational(enc['gs_upper']))[1]
    Cprice=F(0);Jprice=F(0);Ccap=F(0);Jcap=F(0);matter=F(0);rows=[]
    for profile in build_native(mean,2)['Pprofiles']:
        c=scalars(profile);load=c['load'];E=4*abs(c['onsite'])*phi_error
        for length in (1,2):
            c0,cj=c['path'][length];sabs=abs(c0)+abs(cj);Fref=abs(c0)*root0+abs(cj)*roots
            # Covers reference000 and reference222 (needed for prior eta source).
            Fmax=Fref+sabs*frame
            dHop=phi_error*sabs+sqrt_bounds(F(length))[1]*eta*Fmax
            Cprice+=16*load*length**2*dHop;Jprice+=16*load*length*sqrt_bounds(F(V))[1]*dHop
            Ccap+=16*load*length**2*Fref;Jcap+=16*load*length*sqrt_bounds(F(V))[1]*Fref
            E+=48*dHop
        matter+=V*load*E;rows.append(dict(load=load,onsite_scalar=encoded(c['onsite']),one_two_link_scalars={str(length):[encoded(x) for x in c['path'][length]] for length in (1,2)}))
    graph_lower=F(condition['reference_current_Gram_lower']);relative_eta=sqrt_bounds(F(2))[1]*Jprice/sqrt_bounds(graph_lower)[0]
    if relative_eta>=F(1,4):raise ArithmeticError('Corrected initial current conditioning unstable')
    gram_relative=2*relative_eta+relative_eta**2;full_gram_lower=graph_lower*(1-relative_eta)**2
    full_contact=[F(contact['actual_macro_Theta_contact_enclosure']['lower'])-Cprice,F(contact['actual_macro_Theta_contact_enclosure']['upper'])+Cprice];third=[4*F(64)**2*x for x in full_contact]
    D=int(rational(p['dimension']));chi=F(2)**-4096;rmin=rational(p['orbit_rmin']);rmax=rational(p['orbit_rmax'])+2*delta;M=rational(p['principal_metric_cap'])
    radius=min(F(1,1000),rmin/(16*sqrt_bounds(rmax)[1]));M1=8*M/radius
    amax=rational(mean['slow_stiffness_upper'])+rational(refiner['normal_stiffness_twist_difference_upper'])
    precision=90*sqrt_bounds(amax)[1]*F(2)**6144
    precision+=(D-45)*max(2*sqrt_bounds(F(12,5))[1],F(10,3)*sqrt_bounds(F(3,25)*rmax+F(63,25))[1])*F(2)**7168
    cutlower=1-rational(p['cut_norm_squared_deficit_upper'])
    if D!=90*V or cutlower<=0 or eta>=radius:raise ArithmeticError('Actual Phi222 carrier gradient premises fail')
    gradient=(sqrt_bounds(precision)[1]/2+4/eta)/sqrt_bounds(cutlower)[0]
    oldD1=rational(p['Berry_D1_upper']);oldD2=rational(p['Berry_D2_upper']);D1=4*oldD1;D2=4*oldD2+52*oldD1**2;q=rational(p['Gauss_charge_norm_upper'])
    if oldD1**2>oldD2 or rational(pure['new_projected_section_transfer']['projected_overlap_squared_lower'])<F(1,4):raise ArithmeticError('Projected heavy Phi222 section premises fail')
    fibre=chi*(4*M*gradient*(D1+q/rmin)+M*D2+D*M1*(D1+q)+M*q*q/rmin)
    remainder=fibre+matter;old_gain=rational(energy['frozen_Href_gain_per_V_lower']);gain=old_gain-2*remainder/V
    if gain<=0:raise ArithmeticError('Corrected same-fullH energy gain fails')
    source_abi=dict(Phi_carrier_root_selector=[2,2,2],charged_preparation_root_selector=[0,0,0],reference_Hamiltonian_root_selector=[0,0,0],physical_CAR_central_seams=[2,2,2],canonical_Y_ray='UNCHANGEDcanonicalW=[-iY(I+Y²)^-1/2;(I+Y²)^-1/2] withEXACTimplicitC000 andAP222seams; all1+63. No chargedC222ray inserted.',canonical_Y_source_reference_ray_norm_difference_exact='0',ray_difference_reason='SamechargedPREPARATIONandreferenceχ000 BYDEFINITION, notC000=C222. Phi/heavyframeC222differsandpositivebridgeispaid.',Phi_frame='C222,mu=sum_nu [sqrt(G222)]_mu,nu SELECTED_nu; actualimplicitfinite10root atN262144',reference_frame='C000,mu=sum_nu [sqrt(G000)]_mu,nu SELECTED_nu; actualimplicitfinite10root inmean30a, NOTdiagnosticmidpoint',normal_basis='EXACTorthonormalkerOrbit(C222)^T asunchangedpurePhi02121457 carrier; no N0(C000) replacement',source_normalization='ExactPhiGaussian/b2normalizer, positiveprojectedheavysection, allrealGaussianvariance2 andcanonicalchargedrays unchanged. SamejointHaar/FP/Gaussisometry; globalZ4/4. No oldstate/Z/receiver/probabilitybytes retag.',root_certificate=dict(mean_sha256=hashlib.sha256(raws['mean']).hexdigest(),refiner_sha256=hashlib.sha256(raws['refiner']).hexdigest(),exact_222_cube_statement=refiner['selected_candidate']['exact_centre']))
    desc=dict(plan=ds['plan'],plan_revision=REVISION,scientific_execution_revision='03b34911221ba1898315373edaa7c4c3bcac718462e7fe51a842560eb6933f87',input_receipts={name:dict(path=str(path),sha256=hashlib.sha256(raws[name]).hexdigest()) for name,path in paths.items()},N=N,V=V,M=ds['M'],a='1/64',UEphys='1',Lambda='2^8192',charged_total=64,active_mass='1/5',auxiliary_mass='2',auxiliary_count=63,neutral_heavy=1,Ref='4/NET0',source_abi=source_abi)
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_222_000_Gram_Frobenius_distance_upper=positive_exact(delta),actual_222_000_Cart_frame_distance_upper=positive_exact(frame),actual_joint_support_Phi_operator_difference_to_C000=positive_exact(phi_error),corrected_initial_whole_Theta_contact_operator_price=positive_exact(Cprice),corrected_initial_whole_Theta_current_operator_price=positive_exact(Jprice),reference_whole_Theta_contact_operator_cap=positive_exact(Ccap),reference_whole_Theta_current_operator_cap=positive_exact(Jcap),corrected_relative_initial_two_current_action_map_error=positive_exact(relative_eta),corrected_relative_initial_two_current_Gram_error=positive_exact(gram_relative),corrected_full_initial_two_current_Gram_lower=positive_exact(full_gram_lower),corrected_full_H_initial_contact_interval=[encoded(x) for x in full_contact],corrected_full_H_initial_weak_third_interval=[encoded(x) for x in third],corrected_uniform_all_Theta_all_CAR_charged_Phi_freezing_action=positive_exact(matter),corrected_same_full_H_energy_gain_per_V_lower=encoded(gain),literal_profiles=rows,
        corrected_actual_Phi222_carrier_gradient_upper=positive_exact(gradient),corrected_uniform_Phi_colour_fibre_action_upper=positive_exact(fibre),corrected_each_source_common_scalar_remainder_upper=positive_exact(remainder),
        actual_Phi222_gradient_proof='RECOMPUTEDactual45slow+90V−45fastGaussianprecision, not4e0valueinheritance. Hslow222max<=meanHslow000upper+refinerfullnormalmoment/projectorTwiststiffnessdelta; orbitR222max<=oldrmax+2δGram fromsubset-eigenvalueformula. TrCov^-1<=90Lambda^3/4 sqrt(amax222)+(90V−45)Lambda^7/8 max(2sqrt(12/5),(10/3)sqrt(.12rmax222+2.52)). ||graduncut||²=TrCov^-1/4; b2cutgrad<=4/eta, Zcut>=1−deficit. Positiveprojectedheavyderivatives4D1old and4D2old+52D1old² fromoverlap>=1/4. NEWliteralchi[4Mgradient(D1+q/rmin)+MD2+DM1(D1+q)+Mq²/rmin], actualchartM/radius/M1 andall-CARCasimir q bound. OnlyscalarPhi/geometryproof reused; nochargeTheta/current/full-graph inheritance.',
        root_and_frame_proof='Refinerfrozen a5d989 source181−196: truefiniteN coupledstationaritysamePhiConfig; singlecoordinateheavy.012/light.0075analyticstrip andallTEN/all64alias, fasttermidenticalandcancels; localconvexity>=127 givesδG. Meanproof TENFrobeniusSym4 components; source_refiner root-radius carriesTHISnorm, notentrywiseonly. Anti-periodic222 parityandaxispermutation pluslocalunique root forceexactdiagg0,gs,gs,gs; quartersNEVERzeroed. C000alsoexactcube. LetSt=sqrtGt,S0=sqrtG0,X=St−S0. SylvesterStX+XS0=Gt−G0; Frobeniusinnerproduct>=2sqrt(1/16)||X||F² gives||X||F<=2δG. SELECTEDHermitianCliffordΓ haveTrΓμΓν/4=δ and{Γμ,Γν}=2δI: eachδPhiμ hasopnorm=||Xrowμ||2<=2δG. ActualC222 not replacedbyC000/midpoint; positivebridge retainedexactly.',
        corrected_packet_operator_proof='ActualwelljointΦx=Vx(C222+N222Yx)Vx†, linksVxzexp(iξ·Cart/sqrt8)Vy†. SourcejointEuclideanball||(Y,ξ)||<=eta. NormalCartTr4=>normalPhiop<=2eta; adduniformactualcentreδFrame. DISTINCTmacro one/twolinkedges imply||U_path−z||<=sqrtL eta. Literalendpoint-symhop1/2(FxU+UFy) minusfrozen000 hop givesδhop<= (2eta+δFrame)sum|c|+sqrtL eta F000. EnlargedFmax=F000+δFrame sum|c| conservativelycoversoldeta reference222 too. Fourspinforward+reverseCARnorm<=16δhop, onsite<=4|onsite|(2eta+δFrame). Realcosqx y/z unitnormprobes constantalongtheiraxialpaths, Σx(path.f)²=L²; currentΣ|path.f|<=LsqrtV. EntirecanonicalTheta contact/current pricedall1+63, notAunits/no missinga².',
        same_source_initial_conditioning='Sourcechargedχ000 EXACTLYsameasfrozenreference: noY-raychange/divisionbyoverlap. Normalizedheavy/chartisometry givesreferenceGram exactlysource000GaussianY Gram, lower795.619224872758 fromfrozenreceipt. TWOcolumns full−reference operatornorm<=sqrt2 Jprice; weightedGref^-1/2 yieldsrelativeeta. RelativeGram<=2eta+eta², fullminimum>=gref(1−eta)². ThisNEWsuccessor doesNOTconsumeB158fullPhi price, onlyitsVALIDreference000Gram lower andsourceprovenance. Fullweakthird4g²timesnewactualcontactenclosure, withunchangedY000 P7threeThetajet/coretheorem0d176.',
        corrected_energy_comparison='SAMEfullH, actualpreparedY000/Gaussian2×Phi222 carrier versusprioractualeta1/5/Gaussian3/4×SAMEPhi222 carrier. ExactcommonPhi/SU4scalar+boson+neutralheavy expectationcancels; NEWrecomputedPhi222Gaussian-gradientfibrebound, not4e0valueinheritance oroldFullgraph/current transfer. ChargedoperatorfreezepriceNEWincludes222→000 pluscompactnormal/linkradius, validallTheta/CAR; conservativeFmaxalsocoversprioractualeta reference222. PrioralignedLOWER originallyusescentre(mean,refiner) with000enclosureENLARGEDbyrefinedδ, soenclosesactual222 eta source; YupperremainsVALIDfrozen000source/Href000. TheseareexplicitlyTWOreferencecentres, notclaimedidenticalHref. DeductbothfullHremainders. Positivegain72.52440547712558/V retainedafterexactsmallcentreprice, no giantabsoluteenergies fabricated.',
        review_boundary='NEW explicitcentrebridge mathematicalconstruction submittedforparentreview. Allpreviouseta-only fullPhi/current-conditioning/energy/clockclaimsUNADMITTEDandfrozen. Thiscapsule paysactualcentrebridge forSAMEfixedχ000/Phi222source; itdoesNOTestablishstationarity/ground/fullvacuum/pole/commoncone/Compton/statP-Q. No oldsourcebytesorprobabilityretag; no equalityofroots. Finiteclock requiresNEWconsumerofthisresidual, notold72a numericcapsule.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('refiner','mean','energy','conditioning','contact','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.refiner,args.mean,args.energy,args.conditioning,args.contact);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({key:report[key] for key in ('input_sha256','actual_222_000_Cart_frame_distance_upper','corrected_initial_whole_Theta_contact_operator_price','corrected_relative_initial_two_current_action_map_error','corrected_same_full_H_energy_gain_per_V_lower','elapsed_seconds')},indent=2))

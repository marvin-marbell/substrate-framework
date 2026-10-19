"""NEW source-invariant Phi-to-Href Duhamel bound, not inherited evolved graphs."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_ygraph_full_phi_energy import SOURCE_BYTES as ENERGY_BYTES
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_ir_well_transfer import rational,sqrt_bounds,binary_exponent
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**ENERGY_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION='b9e13b2491321aa804f7e52cf4ba3cb12c0a224717cb07b23e487afc0768463b'


def exact(x: dict[str,Any]) -> F:
    return F(int(x['numerator']),int(x['denominator']))


def checked(spec: dict[str,Any]) -> tuple[bytes,dict[str,Any]]:
    raw=Path(spec['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ArithmeticError('Changed actual input epoch')
    return raw,json.loads(raw)


def run(energy_path: Path,conditioning_path: Path,clock: F) -> dict[str,Any]:
    started=time.monotonic();eraw=energy_path.read_bytes();energy=json.loads(eraw);craw=conditioning_path.read_bytes();condition=json.loads(craw)
    _,initial=checked(energy['input_descriptor']['input_receipts']['initial']);_,domain=checked(initial['input_descriptor']['input_receipts']['domain']);ds=domain['input_descriptor']
    _,pure=checked(ds['input_receipts']['purePhi']);well_spec=pure['input_descriptor']['input_receipts']['well'];well_raw=(ROOT/well_spec['path']).read_bytes()
    if hashlib.sha256(well_raw).hexdigest()!=well_spec['sha256']:raise ArithmeticError('Changed scalar well proof')
    well=json.loads(well_raw);p=well['prices'];_,old_initial=checked(condition['input_receipts']['initial_Phi']);_,old_domain=checked(old_initial['input_descriptor']['input_receipts']['domain'])
    if old_domain['input_sha256']!=domain['input_sha256'] or old_initial['pointwise_initial_whole_Theta_current_price']!=initial['pointwise_initial_whole_Theta_current_price']:raise ArithmeticError('Conditioning is not same physical source')
    if clock!=4096 or ds['N']!=262144 or ds['V']!=262144**3 or ds['M']!=2*ds['V']+1:raise ArithmeticError('No clock/physicalbox substitution')
    if domain['input_descriptor']['source']!='NEWpureactualPhi/Haar/heavycutlift×Gaussian2×all64canonicalYgraph rays; nooldstate-sourceZretag':raise ArithmeticError('No substituted preparation')
    eta=rational(p['eta']);chi=F(2)**-4096;D=int(rational(p['dimension']));V=ds['V'];M=rational(p['principal_metric_cap']);rmin=rational(p['orbit_rmin']);rmax=rational(p['orbit_rmax']);budget=rational(p['weighted_energy_budget'])
    radius=min(F(1,1000),rmin/(16*sqrt_bounds(rmax)[1]));M1=8*M/radius;q=rational(p['Gauss_charge_norm_upper']);D1old=rational(p['Berry_D1_upper']);D2old=rational(p['Berry_D2_upper']);D1=4*D1old;D2=4*D2old+52*D1old**2
    if q<66*sqrt_bounds(F(5*V,2))[0] or D1old**2>D2old or rational(pure['new_projected_section_transfer']['projected_overlap_squared_lower'])<F(1,4):raise ArithmeticError('Invalid universal charge/heavy section premises')
    # Independent tail guard for normalized flat-cut eigenpacket gradient and norm.
    Agmon=rational(p['Agmon_exponent_lower']);weighted_norm=1+rational(p['weighted_norm_squared_excess_upper']);tail_pref=sqrt_bounds(weighted_norm)[1]*max(F(1),8/eta)
    if Agmon<=binary_exponent(tail_pref)+2+8192:raise ArithmeticError('Actual Agmon weight does not pay gradient/norm cut tail')
    tail=F(2)**-8192;boundary=rational(p['boundary_action_upper']);spectral=rational(p['well_spectral_state_distance_upper'])
    # Well ground kinetic energy <= budget and homogeneous cometric >=3/5.
    ground_grad=sqrt_bounds(2*budget/(chi*F(3,5)))[1]
    reference_grad=4*ground_grad+4  # Normcut>=1/2 and cut-gradient<1, safely outward.
    fibre=chi*(4*M*reference_grad*(D1+q/rmin)+M*D2+D*M1*(D1+q)+M*q*q/rmin)
    matter=exact(energy['same_source_uniform_charged_Phi_energy_action_upper'])
    # Charged operator freeze + Weyl price of the scalar charged sea subtraction.
    residual=fibre+2*matter+boundary
    initial_phi=2*spectral+4*tail
    packet_error=initial_phi+clock*residual
    survival_error=2*initial_phi+clock*residual
    eta_current=exact(condition['relative_initial_two_current_action_map_error'])
    if eta_current>=F(1,4):raise ArithmeticError('Actual two-current relative conditioning unstable')
    polar_error=2*eta_current/(1-eta_current)
    current_packet_error=packet_error+polar_error
    current_kernel_error=survival_error+2*polar_error
    if current_kernel_error>=F(1,1000):raise ArithmeticError('New finite-clock kernel transfer fails')
    desc=dict(plan=ds['plan'],plan_revision=REVISION,scientific_execution_revision='03b34911221ba1898315373edaa7c4c3bcac718462e7fe51a842560eb6933f87',input_receipts=dict(energy=dict(path=str(energy_path),sha256=hashlib.sha256(eraw).hexdigest()),conditioning=dict(path=str(conditioning_path),sha256=hashlib.sha256(craw).hexdigest())),well_sha256=hashlib.sha256(well_raw).hexdigest(),N=ds['N'],V=V,M=ds['M'],clock=str(clock),scalar_phase='ExactE_D,222−E_light_free(C222,z222), definedbytheUNCHANGEDscalarDirichletwell operator; NOTrounded/evaluated/physicalenergyfit',physical_source='SameNEWY/PhiGaussian2 canonicalchargedsource; itsPhiGaussian/b2/positiveheavy/GaussnormalizationisNOTreplaced. Exactwell eigenpacket isonlya paid reference comparator.')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),normalized_flat_cut_scalar_ground_gradient_upper=positive_exact(reference_grad),new_uniform_all_Theta_all_CAR_reference_generator_fibre_residual=positive_exact(fibre),charged_operator_plus_scalar_Weyl_freezing_price=positive_exact(2*matter),reference_generator_total_residual=positive_exact(residual),actual_source_to_reference_initial_Phi_norm_error=positive_exact(initial_phi),all_times_abs_t_le_4096_source_packet_norm_error=positive_exact(packet_error),all_times_abs_t_le_4096_normalized_Theta_conditioned_survival_kernel_error=positive_exact(survival_error),initial_polar_two_current_frame_error=positive_exact(polar_error),all_times_abs_t_le_4096_two_current_packet_norm_error=positive_exact(current_packet_error),all_times_abs_t_le_4096_two_current_normalized_kernel_operator_error=positive_exact(current_kernel_error),
        scalar_comparator_proof='UNCHANGEDPhiGaussian/b2 fieldfactor ofactualsource haswellminmax distance<=spectral toEXACTnormalizedDirichletwellgroundG222. Thisfactoronly includesliteralreducedscalarCart/SU4kin+bosonpotential+heavy-groundeigenvalue+staticnegativefilledchargedseaE_light_free(Phi,U;Theta0). SAME minmax/Agmon fields applybecausePhi/heavy/root/metric/scalaroperatorunchanged, independentlyoftensorchargedTheta/CARpreparation. Smoothflatζsupportedineta/2 and1ineta/4 producesφref=ζG/||ζG||, supportedinsideactualregularjointchart. Agmonprefactorguard separatelypays ||(1−ζ)G|| and||gradζG||<=2^-8192; existingnormalizedfullD²commutatorpricebound for(Hscalar−E_D)φref<=boundary. Normcut>=1/2 andkineticcoercivity3chi/10 implygrad<=4sqrt(2budget/(3chi/5))+4. Conservativenormdistance2spectral+4tail fromactualpreparedφsource toφref; no oldevolvedPhi graphprice.',
        uniform_generator_residual_proof='ForEVERYnormalizedχ∈D(Href), defineisometricIrefχ=W_joint[φref(Phi,U)ΩHproj(Phi,U)χ(Theta,CAR)]. χ isPhi/SU4-independent, butmaybearbitraryentangledallMTheta×all64chargedCAR. Chargednumbersfixed2V/profileandRef4; heavyneutralΩproj isEXACTrank1spectralgroundline oforiginal−Lambda h. Literalcolour/Cartblock actsφref/Ωheavy andfinitecolourCARchargesonly, noPiTheta/cross derivatives. Uniformlocal4-CARCasimir<=5/2 andMinkowskiall65mobile+Ref giveq<=66sqrt(5V/2), soNEWrecomputedwell.py293fibreinequality holdsFORALLχ withoutfilledseastatistics. MatterunitEMphasesandηjointPhi/colour support give||[Hmatter(Phi,U,Theta)−Hmatter(C,z,Theta)]Irefχ||<=matter. The scalarwellchargedseaE_light_free(Phi,U;Theta0) differsfromE_light_free(C,z;Theta0) by<=matter viafixedparticlefiniteCARminmax/Weyl, henceTOTAL2matter. U1Coulombandcoexactelectric arePhi-independentandkeptEXACTLYinHref. Thus ||Hfull Irefχ−Iref(Href+scalarphase)χ||<=residualUNIFORMLY, not onlyactualinitialχ.',
        domain_and_Duhamel_proof='ActualfiniteN Href=gPiTheta²+boundedHermitianmatrixpotential: allPeierlsphaseunitary, centrePhi fixed, finiteCAR andL+ finiteNnorm. DomainHref=H²(R^M;CAR), Uref preservesit. TheisometricIref sendsH²toD(Hfull) bycompactregularPhi support, smoothφref/Ωheavy/chartmeasure andboundedcolourcharges; noThetaescapeassumption. Forχt=Uref(t)χ0, residualuniforminχt bypreviousoperatorinequality. VariationofconstantsontrueFriedrichs/globalcore yields ||Ufull(t)Irefχ0−exp(-iscalarphase*t)IrefUref(t)χ0||<=|t|residual. UnitarityaddspaidinitialPhi distance. ThisisNEWall-time generatorproof, NOTinheritanceofinitialcontact/current support underfullU nor oldfullgraph/Heinz budgets.',
        conditioned_Theta_consumer='χ0=Theta_f[g2ΩY]/sqrt(2), realunitcosqx y. Norm1 exactlybecauseΩY(theta)norm1 pointwiseandGaussianvariance2. NewY/PhiP7suppliesThetaPsi∈D(Hfull), andχ0∈H² fromGaussianweightandactualcanonicalYjets<=3. BOTHactual/sourcecomparatorhaveidenticalTheta/CARχ0; Phiinitialdistanceunchanged. KernelKfull(t)=<Theta_fPsi/sqrt2,exp[-i(Hfull−scalarphase)t]Theta_fPsi/sqrt2>, Kref=<χ0,exp(-iHref t)χ0>. Scalarphasereferencedexplicitly. Cauchy+twoinitialdistance terms yieldsdisplayedGLOBALcontinuous-timecomplexsurvivalkernelbound, notaninitialTaylorcoefficient.',
        two_current_conditioned_consumer='Fromactualinitialconditioningcapsule Xj=Jref_jΨfull,Yj=Jfull_jΨfull andGref=X†X>0. A=XGref^-1/2 isisometry; B=YGref^-1/2 satisfies||B−A||<=eta_current. DefinephysicalFULLnormalizedcurrentframe Bhat=B(B†B)^-1/2, anexplicitreference-conditionedPOLARframe, NOTsilentlyY Gfull^-1/2 inanother2Dcoordinatebasis. SingularvaluesB∈[1−eta,1+eta] imply||Bhat−A||<=2eta/(1−eta). A columnsfactorSAMEPhiGaussian/heavywithPhi-independentnormalizedχref_j; arbitraryunitlinearcombinationshavechi-norm1, soDuhamelestimateUNIFORM2-column operatornorm. Initialpolarerror ispropagatedbyFULLunitarity, notassumedPhi-independent. KernelBhat†exp[-i(Hfull−scalarphase)t]Bhat vsχref†exp(-iHref t)χref hasoperatornormerror<=2initial_phi+|t|residual+2polarerror. ReferenceCURRENTGram lower/condition usedONLYINITIAL; no evolvedcurrent/contact price imported.',
        review_boundary='NEWfiniteN original-law physicalsource/normalizedpacket/relativekernel transfer CONSTRUCTION forparentmathematicalreview. Requiresparentreviewofunchangedscalarwell comparator identification, normal-chartfibreoperatorbound andDuhamel/corepremises. DoesNOTnumericallyevaluateHref/fullHpropagator, establishcurrentframesasphotonpoles, prove4096retardedTheta/currentcommutator, ground/stationarity/vacuum/commoncone/Compton/statP-Q. Exactscalarphase ismathematicallydefinednotmeasured; no fakeclockresponse.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('energy','conditioning','output'):parser.add_argument('--'+role,type=Path,required=True)
    parser.add_argument('--clock',type=F,required=True)
    args=parser.parse_args();report=run(args.energy,args.conditioning,args.clock);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({key:report[key] for key in ('input_sha256','reference_generator_total_residual','all_times_abs_t_le_4096_normalized_Theta_conditioned_survival_kernel_error','all_times_abs_t_le_4096_two_current_normalized_kernel_operator_error','elapsed_seconds')},indent=2))

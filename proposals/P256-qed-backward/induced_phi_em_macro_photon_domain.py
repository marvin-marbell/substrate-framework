"""Correct physical-A units and actual-source weak-domain full-H photon moment."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_macro_squeeze import SOURCE_BYTES as SQUEEZE_BYTES
from induced_phi_em_gain_certificate import IV, iv, endpoints, encoded
from induced_phi_ir_well_transfer import rational, sqrt_bounds

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**SQUEEZE_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
PLAN='issue-222-induced-photon-delta13-continuation-20261008'
REVISION='8d37944a271479ff0ea3f97aa4ad022c3a0aaa4779384dfc95deb089890f6547'


def domain_certificate(state: dict[str,Any],well: dict[str,Any],pure: dict[str,Any]) -> dict[str,Any]:
    p=well['prices'];eta=rational(p['eta']);rmin=rational(p['orbit_rmin']);rmax=rational(p['orbit_rmax'])
    chart_radius=min(F(1,1000),rmin/(16*sqrt_bounds(rmax)[1]));overlap=rational(pure['new_projected_section_transfer']['projected_overlap_squared_lower'])
    c=rational(p['quartic_graph_coercivity']);B=rational(p['full_Hamiltonian_lower_shift']);V=int(state['V']);a=F(float(state['a']))
    exact_c=F(2)**8192/(8*a*V)
    if not (0<eta<chart_radius/4 and overlap>0 and rational(p['heavy_gap_lower'])>0 and 0<c<=exact_c and B>0):raise ArithmeticError('Actual localchart/coercivity/domain prerequisites fail')
    recipe=well['physical_state_definition']['normalization']
    if 'Radial b=(1-r^2/eta^2)^2_+' not in recipe:raise ValueError('The actual source cutoff definition changed')
    Jcap=16*sqrt_bounds(F(V))[1]*rational(p['scalar_path_all_charged_cap'])
    return dict(actual_Phi_cutoff_order=2,chart_radius_lower=encoded(chart_radius),projected_heavy_overlap_squared_lower=pure['new_projected_section_transfer']['projected_overlap_squared_lower'],
        global_quartic_coercivity=p['quartic_graph_coercivity'],global_H_lower_shift=p['full_Hamiltonian_lower_shift'],all64_Theta_current_field_slope_cap=encoded(Jcap),
        domain_claim='Psi,Theta_fPsi,Pi_fPsi,J_fPsi,Theta_fJ_fPsi belongD(Hfull); Theta_fHPsi isL2 at t0. NO claimPsiinD(Hfull²). Allclaims referactualcutoffPhi×Gaussian×canonicalSlater state, notcompactEMpacket.',
        actual_cutoff_regularization='d=1-r²/eta², b=d_+². b andfirstderivative vanishattheboundary, henceweaksecondderivative ispiecewisebounded withNO delta. For0<epsilon<1, b_epsilon=omega(d/epsilon)b, omega(t)=exp(-1/t)/(exp(-1/t)+exp(-1/(1-t))) on(0,1), extended0/1. b_epsilon isCinf anddiffers only0<d<epsilon: zeroth/first/secondjet errors O(epsilon²),O(epsilon),O(1). Boundarylayermeasure O(epsilon) at finiteactualdimension, soH2 convergence O(sqrt(epsilon)). ConstantsmaydependactualfiniteV/eta/covariance; allfinite. Normalizeeachapproximant byitspositivenorm tending1; thisisaproofsequence, NOT a replacement source/Z.',
        chart_and_fibre='Actualeta liesstrictlyinside orbit-rmin chart. Metric/cometric/Haar/FP/w_joint^-1/2 andtheirfinitejets arebounded/smooth onalargercompactchart; actualheavygap andpositiveprojectionoverlap makeheavyprojectedSlatersectionanalyticthere. Chargedcentreeta1/5Slater isPhi-independent; finiteCAR/Ref4 smooth. ExactjointGauss/equivariantlift andonlyglobalZ4 preservefixedsectors. Inreducednormalchart multiplybyb_epsilon, thenapplythesameisometriclift; noflatmeasure orlocalcentreprojection. Smoothcompactgroupfunctions admit PeterWeylcore approximation; equivariance retained. Colour/cut support staysinsideoriginalchart.',
        EM_and_simultaneous_core='RealallM Gaussian isSchwartz. MultiplybyCinf radialEMcut at radiusR, R→infinity; allfinitepolynomialTheta/Pi weightedSchwartzseminorms converge. Peierls factorsareunitary andallderivativesbounded; theirPhi/CAR coefficients onsourcechart bounded. TensorPhiH2 approximation andEMSchwartzcut, includingactualJac/Gauss/heavysection, convergesSIMULTANEOUSLY inHfullgraph forPsi,ThetaPsi,PiPsi,JfPsi,ThetaJfPsi, andinL2 forThetaHPsi. Localsecond-orderliteralcoefficients are smooth boundedonthechart; originalCart/SU4kinetics andquartic/mattermultiplications thereforepassbyH2 convergence. H² budget isnotusedasD(H²).',
        global_coercivity_proof='ForG_x,munu=(1/4)Tr(Phi_x,mu Phi_x,nu), rhoPhi²=sum_x,mu Cartcoords(Phi_x,mu)²=sum_x,mu Tr(Phi_x,mu²)/4. Literalquartic=(Lambda/a)sum_x||G_x||F². PositivityGram: ||G_x||F²>=(TrG_x)²/4; finiteCauchy sum>=rhoPhi⁴/(4V). Retainhalfquartic c=Lambda/(8aV). Alignment/SU4plaquette/Coulombandkineticspositive. LiteralfiniteCAR matter>=-Lmatter rhoPhi withLmatter=8sqrtV(Lambda heavycap+all64lightcap); Young shiftB=1+Lmatter²/c fromactualwell retainsgPi_Q²+c rhoPhi⁴ globally, notmerelypreparedsupport. ThusHfull+B>=gPi_Q²+c rhoPhi⁴ asforms.',
        J_H_bounded='Literalone/twolink currentslinearPhi; length<=2 andall64scalarpathcap give||Jf u||<=Jcap||rhoPhi u||. Formcoercivity/Holder yield||Jf u||⁴<=Jcap⁴/c *||u||²<u,(H+B)u>. Foru∈D(H), RHS<=constant||u||³(||Hu||+B||u||), soD(H)subsetD(Jf) andJf isH-bounded. MultiplicationJf isselfadjoint. SameenergyboundmakesPi_f H-bounded andcontrolsunitaryenergyatalltimes.',
        weak_differentiation='Foru,v∈D(H), FJ(t)=<Uu,J Uv>, U=exp(-iHfullt). Differencequotient=< (Uu-u)/t,J Uv>+<Ju,(Uv-v)/t>. Uv→v inHgraph impliesJ Uv→Jv. ThereforeFJprime=i<Hu,Jv>-i<Ju,Hv> at0, andatallt withUu/Uv. Useu=Psi,v=ThetaPsi andthereversedpair; noJ Uv graphderivative andnoTheta U(t)Psi∈D(H) assertion.',
        Theta_and_Pi_weak_extension='UseboundedsmoothTheta_R withuniformfirstderivative boundandsecondderivative O(1/R): [H,Theta_R]=-ig(2Theta_Rprime Pi+Theta_Rsecond). Energycontrol ofPi plusdominatedconvergence extendsweakTheta_dot=2gPi. InitialGaussianweightedTheta productsfinite; cutoffcommutator also givesfiniteTheta_U(t)norm boundwithoutassertingHdomain. Pi weakdifferentiation usesH-boundedPi andcorecomm i[H,Pi]=-J. Simultaneouscoreapprox andinitialThetaHPsi L2 justifyallformcommutatorcancellations.',
        full_H_contact_identity='Oncommoncore Hfull=gPi_Q²+Hother, [Hother,Theta_f]=0 and[Jf,Theta_f]=0; parentstructuralreview includesactualPhi/SU4kineticandrhoL+rho. Jacobi gives[[Hother,J],Theta]=0 evenif[Hother,J]hasPhi/colourmomenta; NOT dropped. [[gPi²,J],Theta]=-2gContact. Hgraph/Jcore convergence passesidentity asweakform foractualPsi. ThusF(t)=<[J(t),Theta]> hasF(0)=0,Fprime(0)=-2ig<Contact>. Twointegrations Dfull_R(t)=-2gt+2ig integral_0^t(t-s)F(s)ds implyDfull_R(t)=-2gt+(4g²<Contact>)t³/6+o(t³). NoH²Psi, pole, stationarystate orfiniteclock approximation assumed.')


def run(source_path: Path,well_path: Path,withdrawn_path: Path) -> dict[str,Any]:
    started=time.monotonic();raw=source_path.read_bytes();source=json.loads(raw);state=source['input_descriptor'];wellraw=well_path.read_bytes();well=json.loads(wellraw)
    oldraw=withdrawn_path.read_bytes();old=json.loads(oldraw);spec=state['inputs']['purePhi'];pureraw=(ROOT/spec['path']).read_bytes();pure=json.loads(pureraw)
    if hashlib.sha256(pureraw).hexdigest()!=spec['sha256'] or hashlib.sha256(wellraw).hexdigest()!=pure['input_descriptor']['input_receipts']['well']['sha256']:raise ValueError('Actual source/domain receipt binding mismatch')
    if state['kappa']!=0 or state['charged_counts']!=[1,63] or old['input_descriptor']['source_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('No silent source/count/correspondingwithdrawn-epoch substitution')
    a=F(float(state['a']));g=F(float(state['UE']))/a;N=int(state['N']);V=int(state['V']);s=rational(state['variance']);IV.dps=65
    components=[iv(rational(row[0]))+IV.mpf([0,1])*iv(rational(row[1])-rational(row[0])) for row in source['interval_matter_components']]
    attenuation1=IV.exp(-iv(s)*iv(F(2,3)+F(1,3*V))/2);attenuation2=IV.exp(-iv(s))+IV.mpf([0,1])*(1-IV.exp(-iv(s)))
    clo,chi=endpoints(-(attenuation1*components[1]+4*attenuation2*components[2])/3)
    contact_A=rational(source['Phi_transfer']['all_flavour_contact_expectation_error']);price=4*contact_A/(a*a);oldprice=rational(old['full_Phi_contact_transfer_price'])
    if price*a*a!=oldprice or price<=oldprice:raise ArithmeticError('Negative old-price control didnotexposeactualunitfailure')
    current_A=rational(source['Phi_transfer']['all_flavour_current_action_error']);current_Theta=current_A/a
    third=[4*g*g*x for x in (clo-price,chi+price)];domain=domain_certificate(state,well,pure)
    desc=dict(source_sha256=hashlib.sha256(raw).hexdigest(),source_input_sha256=source['input_sha256'],well_sha256=hashlib.sha256(wellraw).hexdigest(),purePhi_sha256=hashlib.sha256(pureraw).hexdigest(),withdrawn_price_epoch_sha256=hashlib.sha256(oldraw).hexdigest(),
        N=N,V=V,M=int(state['M']),a=encoded(a),UE=state['UE'],variance=encoded(s),charged_counts=[1,63],plan=PLAN,plan_revision=REVISION,inherited_source_plan_revision=state['plan_revision'])
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),
        reference_literal_Theta_contact_interval=[encoded(clo),encoded(chi)],corrected_full_Phi_literal_Theta_contact_interval=[encoded(clo-price),encoded(chi+price)],
        contact_expectation_error_A=encoded(contact_A),corrected_four_entry_contact_price_Theta=encoded(price),corrected_contact_unit_factor=encoded(1/(a*a)),
        corrected_current_action_error_Theta=encoded(current_Theta),current_unit_factor=encoded(1/a),
        old_price_negative_control=dict(withdrawn_four_entry_price=encoded(oldprice),corrected_to_old_ratio=encoded(price/oldprice),old_price_is_insufficient=True,
            withdrawal='Bothold7ddsigma1andsigma3/4fullPhi priceintervals withdrawn; literalREFERENCEcoefficients remainhistoricalvalid. Priorbytes untouched.'),
        full_H_photon_retarded_first_derivative=encoded(-2*g),full_H_photon_retarded_second_derivative=encoded(F(0)),corrected_full_H_third_moment_interval=[encoded(x) for x in third],
        full_H_functional='Dfull_R(t)=-i step(t)<Psi_full,[exp(iHfullt)Theta_fexp(-iHfullt),Theta_f]Psi_full>, actualoriginalfullPhi/SU4/CAR/realEM H. ReferenceHref functional isSEPARATE, neverretaggedasfullH.',
        actual_source_weak_domain_extension=domain,reference_source_unchanged=True,
        limitations='CorrectedfullH initialweakmomentproposal withactualunitprice/domainproofpendingparentreview; NOT finiteclock4096 evaluatedpropagator, stationaryphvac, pole/residue/commoncone orCompton/QEDadmission. Contactisnotaparticle.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('source','well','withdrawn','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.source,args.well,args.withdrawn);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256','corrected_four_entry_contact_price_Theta','corrected_contact_unit_factor','current_unit_factor','corrected_full_H_third_moment_interval','elapsed_seconds')},indent=2))

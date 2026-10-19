"""NEW full-Phi same-source-family energy comparison: cancel common scalar exactly."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_ygraph_initial_contact import SOURCE_BYTES as INITIAL_BYTES,positive_exact
from induced_phi_em_macro_preparation import scalars,build_native
from induced_phi_em_gain_certificate import encoded
from induced_phi_ir_well_transfer import rational,sqrt_bounds
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**INITIAL_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION='b9e13b2491321aa804f7e52cf4ba3cb12c0a224717cb07b23e487afc0768463b'


def run(initial_path: Path,aligned_path: Path) -> dict[str,Any]:
    started=time.monotonic();iraw=initial_path.read_bytes();initial=json.loads(iraw);araw=aligned_path.read_bytes();aligned=json.loads(araw);ids=initial['input_descriptor'];ads=aligned['input_descriptor'];domain_spec=ids['input_receipts']['domain'];domain_raw=Path(domain_spec['path']).read_bytes();domain=json.loads(domain_raw);ds=domain['input_descriptor']
    if hashlib.sha256(domain_raw).hexdigest()!=domain_spec['sha256']:raise ArithmeticError('Actual Ydomain epoch changed')
    graph_spec=ds['input_receipts']['graph'];graph_raw=Path(graph_spec['path']).read_bytes();graph=json.loads(graph_raw);pure_spec=ds['input_receipts']['purePhi'];pure_raw=Path(pure_spec['path']).read_bytes();pure=json.loads(pure_raw);well_spec=pure['input_descriptor']['input_receipts']['well'];well_raw=(ROOT/well_spec['path']).read_bytes();well=json.loads(well_raw);mean_spec=ids['input_receipts']['mean'];mean_raw=Path(mean_spec['path']).read_bytes();mean=json.loads(mean_raw)
    for raw,spec in ((graph_raw,graph_spec),(pure_raw,pure_spec),(well_raw,well_spec),(mean_raw,mean_spec)):
        if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ArithmeticError('Actual fullPhi comparison epoch mismatch')
    if ads['inputs']['purePhi']['sha256']!=hashlib.sha256(pure_raw).hexdigest() or ads['inputs']['mean']['sha256']!=hashlib.sha256(mean_raw).hexdigest() or int(ads['N'])!=int(ds['N']):raise ArithmeticError('CannotcanceldifferentPhi/carrier/mean/physicalH')
    p=well['prices'];V=int(ds['V']);D=int(rational(p['dimension']));N=int(ds['N']);eta=rational(p['eta']);chi=F(2)**-4096;rmin=rational(p['orbit_rmin']);rmax=rational(p['orbit_rmax']);M=rational(p['principal_metric_cap']);amax=rational(mean['slow_stiffness_upper']);cutlower=1-rational(p['cut_norm_squared_deficit_upper'])
    radius=min(F(1,1000),rmin/(16*sqrt_bounds(rmax)[1]));M1=8*M/radius
    if D!=90*V or N!=262144 or cutlower<=0 or eta>=radius:raise ArithmeticError('Actual carrier/covariance/chart guard failed')
    fast_precision=max(2*sqrt_bounds(F(12,5))[1],F(10,3)*sqrt_bounds(F(3,25)*rmax+F(63,25))[1]);trace_precision=F(90)*sqrt_bounds(amax)[1]*(F(2)**6144)
    trace_precision+=(D-45)*fast_precision*(F(2)**7168)
    # ExactGaussian moment plusliteralb2 cutoff: ||grad(b g)||<=||gradg||+4/eta; Z>=1-deficit.
    gradient=(sqrt_bounds(trace_precision)[1]/2+4/eta)/sqrt_bounds(cutlower)[0]
    D1old=rational(p['Berry_D1_upper']);D2old=rational(p['Berry_D2_upper']);D1=4*D1old;D2=4*D2old+52*D1old*D1old;q=rational(p['Gauss_charge_norm_upper'])
    if D1old*D1old>D2old or rational(pure['new_projected_section_transfer']['projected_overlap_squared_lower'])<F(1,4):raise ArithmeticError('Actual normalizedprojectedheavy derivative premises fail')
    fibre=chi*(4*M*gradient*(D1+q/rmin)+M*D2+D*M1*(D1+q)+M*q*q/rmin)
    enc=mean['centre_enclosure'];root0=sqrt_bounds(rational(enc['g0_upper']))[1];roots=sqrt_bounds(rational(enc['gs_upper']))[1];mattercap=F(0);rows=[]
    for profile in build_native(mean,2)['Pprofiles']:
        c=scalars(profile);cap=8*abs(c['onsite'])
        for length in (1,2):
            c0,cj=c['path'][length];Fref=abs(c0)*root0+abs(cj)*roots;delta=2*(abs(c0)+abs(cj))+sqrt_bounds(F(length))[1]*Fref;cap+=48*delta
        mattercap+=c['load']*cap;rows.append(dict(load=c['load'],all_paths_and_onsite_energy_delta_per_V_per_eta=encoded(cap)))
    matter=eta*V*mattercap;remainder=fibre+matter;ref_upper=F(graph['actual_Href_energy_per_V_enclosure']['upper']);ref_lower=rational(aligned['prior_same_finite_Href_quantum_source_energy_per_V_lower']);Hrefgain=V*(ref_lower-ref_upper);fullgain=Hrefgain-2*remainder
    if fullgain<=0:raise ArithmeticError('NewfullPhi samephysicalH energycomparison gain fails')
    desc=dict(plan=ds['plan'],plan_revision=REVISION,N=N,V=V,M=ds['M'],quantum_inputs='OriginalCAR/CartCCR/SU4left/noncompactEM law inputs unchanged',input_receipts=dict(initial=dict(path=str(initial_path),sha256=hashlib.sha256(iraw).hexdigest()),aligned=dict(path=str(aligned_path),sha256=hashlib.sha256(araw).hexdigest())),Phi_source_sha256=hashlib.sha256(pure_raw).hexdigest(),well_sha256=hashlib.sha256(well_raw).hexdigest(),comparison='NEWΨY(s2) versusactualpriorΨeta1/5(s3/4), SAMEPhi/heavy/chartcarrier andsamefullH; alignedreceipt usedONLYboundpriorHrefLOWER, NOTits eta7/20 energy')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_carrier_gradient_norm_upper=positive_exact(gradient),actual_projected_heavy_first_second_caps=[positive_exact(D1),positive_exact(D2)],same_source_uniform_Phi_colour_fibre_action_upper=positive_exact(fibre),same_source_uniform_charged_Phi_energy_action_upper=positive_exact(matter),each_source_common_scalar_factorization_remainder_upper=positive_exact(remainder),actual_same_full_H_energy_gain_per_V_lower=encoded(fullgain/V),actual_same_full_H_energy_gain_lower=encoded(fullgain),frozen_Href_gain_per_V_lower=encoded(ref_lower-ref_upper),literal_paths=rows,
        exact_common_energy_cancellation='Useactualbodychartisometry. Bothnormalizedstatesφ_eta(z)ΩHproj(z)⊗χ_i(Theta) haveIDENTICALexactPhi/colourcarrier/heavysection, χ_iPhi-independent andnormalizedinallMrealTheta×chargedCAR. χY=g2×ΩY(theta), χold=g3/4×Omegaeta1/5 fixedcentre. OriginalPhi/SU4scalarLaplaceBeltrami/Haar-FPscalarterm+bosonquartic/alignment/curl+neutralheavy eigenvalue actscommonscalarfieldfactor; its expectationcancels EXACTLYbeforearithmetic, eventhoughindividualenergy~LambdaV. RealU1 CoulombandfullprimitiveEMkinetic remainINphysicalHref andsourceenergy, NOTremoved. RemainingPhi/SU4fibre derivative/affinecharge pieces pricedforEACHsource byactualuniformnormbound; chargedΦ/colour-dependentmatterdifference toexactcentreHref pricedbyINITIALsupport operatornorm. Nofullfieldterm dropped.',
        uniform_fibre_proof='ActualPhi/SU4block hasNOcoexactPi_Theta (literalfull_action andP7). Consequentlyitsnormal-chartactiondifferentiatesφ/Ωheavy andusesfiniteCARcolourcharges, NEVERThetajets ofχY. χ_i norm1 makesallboundsstate-independentinTheta/CAR. Actualtotalcolourcharge boundq<=66sqrt(5V/2): local4slotCARantisym reps maxquadraticCasimir5/2, sum65mobile+fixedRef byMinkowski; validarbitrarychargedrays, no filledsea/pervertexn2 assumption. ProjectedheavyD1<=4oldD1, D2<=4oldD2+52oldD1² frompositiveoverlap>=1/4. LiteralboundedchartcometricM andCauchyjetM1, orbitinverse<=1/rmin givechi[4M||gradφ||(D1+q/rmin)+MD2+DM1(D1+q)+Mq²/rmin], sameactualwell.py293 operator-action inequality, notoldEMgraph/current/normdistance receipt inheritance. Eachcross/zerothpositiveboundrecomputedwithNEWactualcarriergradient.',
        actual_carrier_gradient_proof='UncutφexactGaussian covariance45slowLambda^-3/4 Hslow^-1/2/2 and90V-45fastgivenactualspectralcovariances. Precisiontrace<=90Lambda^3/4sqrt(amax)+(90V-45)Lambda^7/8max(2sqrt(12/5),(10/3)sqrt(.12rmax+2.52)). ||gradg||²=TrCov^-1/4. b2radialcut||gradb||<=4/eta, andb²>=1-4r²/eta² yieldsZ>=1-deficit actualoutwardprice. Thusdisplayedgradientbound, without ground-state/oldbudget/EMpacket assumption. AllFraction/isqrt arithmetic, fixedFINITEmacroV.',
        matter_support_proof='Onsiteδh=onsiteδPhi0, localCARnorm<=4||δh||<=8|onsite|eta. Eachdirectedaxialhop/fullHermitianCARpair ≤16eta[2sum|c|+sqrtL Frefcap] asactualinitialcontactendpoint/bodychartproof; threeaxisdirections =>48sumL. SummingVsites/all1+63 yieldsmatterbound. UnitEMphasesALLtheta; normalizedheavyI+Gaussisometry. CoulombPhi-independent andkeptinHref. No sourceMean/H Hamiltonian change.',
        review_boundary='NEW completefiniteN full-law energycomparison CONSTRUCTION forparentmathematicalreview: uniformnormal-chart affine-action bound andcommon-scalarfactorization explicitly identified, no inheritedoldsourceadmission. This isNOTclaimedachievedground/fullstationarity/vacuum/pole/P-Q/commoncone/time4096/Compton. No absolutehugefullenergy fabricated; onlyEXACTsame-H differencewithcommonpiece cancelled.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('initial','aligned','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.initial,args.aligned);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({key:report[key] for key in ('input_sha256','same_source_uniform_Phi_colour_fibre_action_upper','actual_same_full_H_energy_gain_per_V_lower','elapsed_seconds')},indent=2))

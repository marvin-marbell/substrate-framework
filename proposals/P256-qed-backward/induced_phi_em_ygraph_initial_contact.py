"""NEW initial-only full-Phi contact: literal chart support, no old graph transfer."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_ygraph_weak_domain import SOURCE_BYTES as DOMAIN_BYTES
from induced_phi_em_macro_preparation import scalars,build_native
from induced_phi_ir_well_transfer import rational,sqrt_bounds
from induced_phi_em_gain_certificate import encoded
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**DOMAIN_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION='b9e13b2491321aa804f7e52cf4ba3cb12c0a224717cb07b23e487afc0768463b'


def positive_exact(value: F) -> dict[str,Any]:
    if value<=0:raise ArithmeticError('Strictnonzero positivecertificate required')
    exponent=value.numerator.bit_length()-value.denominator.bit_length()
    if F(2)**exponent>value:exponent-=1
    if not F(2)**exponent<=value<F(2)**(exponent+1):raise ArithmeticError('Exact exponent guard')
    return dict(numerator=str(value.numerator),denominator=str(value.denominator),strictly_positive=True,binary_exponent_lower=exponent,binary_exponent_upper=exponent+1)


def run(domain_path: Path,contact_path: Path,mean_path: Path) -> dict[str,Any]:
    started=time.monotonic();blobs={name:p.read_bytes() for name,p in (('domain',domain_path),('contact',contact_path),('mean',mean_path))};caps={name:json.loads(raw) for name,raw in blobs.items()};domain=caps['domain'];contact=caps['contact'];mean=caps['mean'];ds=domain['input_descriptor'];spec=ds['input_receipts']['purePhi'];praw=Path(spec['path']).read_bytes();pure=json.loads(praw)
    if hashlib.sha256(blobs['contact']).hexdigest()!=ds['input_receipts']['contact']['sha256'] or hashlib.sha256(praw).hexdigest()!=spec['sha256']:raise ArithmeticError('Actual Y/Phi source epoch mismatch')
    graph_raw=Path(ds['input_receipts']['graph']['path']).read_bytes();graph=json.loads(graph_raw)
    if hashlib.sha256(graph_raw).hexdigest()!=ds['input_receipts']['graph']['sha256'] or hashlib.sha256(blobs['mean']).hexdigest()!=graph['input_sha256']:raise ArithmeticError('Actualmean/source mismatch')
    wspec=pure['input_descriptor']['input_receipts']['well'];wraw=(ROOT/wspec['path']).read_bytes();well=json.loads(wraw)
    if hashlib.sha256(wraw).hexdigest()!=wspec['sha256']:raise ArithmeticError('Actual packet well changed')
    N=int(ds['N']);V=int(ds['V']);eta=rational(well['prices']['eta']);s=rational(ds['variance']);a=F(1,64);g=F(64);geom=well['physical_state_definition']
    if N!=262144 or s!=2 or 'Phi_x=V_x(C_t+N0Y_x)' not in geom['joint_map'] or 'Radial b=(1-r^2/eta^2)^2_+' not in geom['normalization']:raise ArithmeticError('Actual coordinate/carrier support definition changed')
    enc=mean['centre_enclosure'];root0=sqrt_bounds(rational(enc['g0_upper']))[1];roots=sqrt_bounds(rational(enc['gs_upper']))[1];sqrtV=sqrt_bounds(F(V))[1];rows=[];contactcap=F(0);currentcap=F(0)
    for profile in build_native(mean,2)['Pprofiles']:
        c=scalars(profile);rowcontact=F(0);rowcurrent=F(0);paths=[]
        for length in (1,2):
            c0,cj=c['path'][length];cref=abs(c0)*root0+abs(cj)*roots;delta_per_eta=2*(abs(c0)+abs(cj))+sqrt_bounds(F(length))[1]*cref
            # Entireactualrealmode contact bounded, notonecomplexqentry: allcrossentries present in(path.f)^2.
            rowcontact+=16*length*length*delta_per_eta;rowcurrent+=16*length*sqrtV*delta_per_eta
            paths.append(dict(length=length,literal_scalar_mass=encoded(c0),literal_scalar_spatial=encoded(cj),reference_hop_operator_upper=encoded(cref),hop_delta_per_eta_upper=encoded(delta_per_eta)))
        contactcap+=c['load']*rowcontact;currentcap+=c['load']*rowcurrent;rows.append(dict(load=c['load'],paths=paths,whole_mode_contact_delta_per_eta_upper=encoded(rowcontact),whole_mode_current_delta_per_eta_upper=encoded(rowcurrent)))
    contact_price=eta*contactcap;current_price=eta*currentcap;old_price=rational(domain['canonical_Theta_contact_price']);lo=F(contact['actual_macro_Theta_contact_enclosure']['lower']);hi=F(contact['actual_macro_Theta_contact_enclosure']['upper']);full=[lo-contact_price,hi+contact_price];third=[4*g*g*x for x in full]
    if not 0<contact_price<old_price:raise ArithmeticError('New literalpointwiseinitialprice didnotimproveoldgraph candidate')
    desc=dict(plan=ds['plan'],plan_revision=REVISION,N=N,V=V,M=ds['M'],variance=ds['variance'],allcharged=[1,63],input_receipts={name:dict(path=str(p),sha256=hashlib.sha256(blobs[name]).hexdigest()) for name,p in (('domain',domain_path),('contact',contact_path),('mean',mean_path))},pure_source_sha256=hashlib.sha256(praw).hexdigest(),well_sha256=hashlib.sha256(wraw).hexdigest(),physical_mode='literalrealcos(2pi x/N)y f, sumf²1 andDf0, canonicalTheta—notphysicalA')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_packet_radius=positive_exact(eta),pointwise_initial_whole_Theta_contact_price=positive_exact(contact_price),pointwise_initial_whole_Theta_current_price=positive_exact(current_price),contact_price_to_previous_graph_price=positive_exact(contact_price/old_price),full_H_same_Y_Phi_source_initial_contact_interval=[encoded(x) for x in full],full_H_same_Y_Phi_source_weak_third_interval=[encoded(x) for x in third],literal_path_profiles=rows,
        exact_source='SAMEnewY/Phi source boundby0d176: chartcarrierphi_eta(z), support||(Y,xi)||<=eta; exactimplicit222 referencePhi; normalizedheavyprojectedOmega_H(z), chargedcanonicalΩY(Theta)Phi-independent norm1 pointwise, EMGaussianvariance2 allM, samephysicalGauss/globalZ4 lift andRef4NET0. No source/Z/probability retag or EMlaw alteration.',
        pointwise_operator_proof='Actualjoint_map removesvertexVx bySU4equivariance/unitaryGausslift: Phi=C222+N0Y, Uedge=z exp(i xi.T), N0orthonormalCart15. CartbasisTr4 =>||deltaPhi_mu||op<=2||localnormalcoordinates||<=2eta. ActualT=Cart/sqrt8 hasTrTaTb=delta/2: ||X||op<=||X||F=||xi||/sqrt2<=||xi||. Unitaryexponentialdifference<=||X|| andorderedproducttelescoping gives||deltaU_path||op<=sumedge||xi_e||<=sqrt(length)eta foractualmacro distinctedges. Literalfull_action.py142 hop=.5(FxU+UFy), F=sumc_muPhi_mu. SubtractatEXACTreference withunitarynewU: deltaHop=.5(deltaFx Unew+Fref_x deltaU+deltaU Fref_y+Unew deltaFy), boundeta[2sumabs(c)+sqrtL Frefcap]. EveryEMphaseunitmodulus ALLrealTheta, nocompactEM/Gaussianclipping.',
        CAR_and_mode_proof='For4spinforward+reverse CAR hop, ||sumab(deltaHab a_xa†a_yb+h.c.)||<=2sumab|deltaHab|<=8||deltaH||F<=16||deltaH||op, independentoccupiedstate/entanglement. Literalrealcosq y probe hassamef_y alongeachy one/twolink; path.f=L f_y(x). SumoverVpaths (path.f)²=L²sumf_y²=L² exactly; otheraxeszero. ThispricesENTIRErealmode contact includingcrossentries, NOTcomplexqentry orApriceneeding4096. CurrentusesCauchy sum|path.f|<=LsqrtV. Sumall1+63 physicalscalarpaths, noonsitecontact, no omitted2link.',
        source_expectation_proof='Contactactsidentityonneutralheavy/Ref andgauge-invariantlight+EMonly. Actualisometricw_joint^-1/2 Haar/FP/Gausslift cancelsdensitymeasure; normalizedheavysectionnorm1 andΩY(Theta)norm1 pointwise leaveexactnormalizedcarrierPhi andGaussianTheta marginals. ReferenceContact independentPhi, samecanonicalchargedvectorΩY(theta); thereforedifferenceinexpectation<=sup_on_carrier ||Contactactual−Contactref||. ThisisINITIALONLYsupportcomparison, noevolvedstateHeinz/olduniform-graphinheritance, no heavy-phase/exponentiallysmall overlap division. Core/weakthird prooffromactualYdomain stillrequiredandboundbyreceipt.',
        review_boundary='ConstructiveNEW directinitial-only comparison submittedforparentmathematicalreview, notselfapproval. ItreplacesNOoldreceiptbytes andusesNOoldPhi current/contact transferprice. FullH momentisinitialweakcoefficient, notfinite-tkernel; no sourcePhi escape bound oralltimefrozenmean replacement.',limitations='No fullPhi energy/ground/photonvacuum/pole/residue/commoncone/4096clock/Compton/statP-Q promotion. Pointwisepacketprice remainsstrictpositiveexactFraction; no binary64underflow-tozero claim.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('domain','contact','mean','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.domain,args.contact,args.mean);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({key:report[key] for key in ('input_sha256','pointwise_initial_whole_Theta_contact_price','pointwise_initial_whole_Theta_current_price','full_H_same_Y_Phi_source_weak_third_interval','elapsed_seconds')},indent=2))

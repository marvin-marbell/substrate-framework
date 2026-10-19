"""Unchanged full-Phi source/insertion packets: force, not total-energy EM prices."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_ygraph_current_source_retarded import SOURCE_BYTES as CURRENT_BYTES
from induced_phi_em_ygraph_retarded_transfer import upper_sqrt
from induced_phi_em_ygraph_clock_transfer import checked, exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_macro_preparation import build_native, scalars

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**CURRENT_BYTES, Path(__file__).name: Path(__file__).read_bytes()}
REVISION = '6ebc34a912158254456c535bd10259fca04201b14f446a778bba68d5ef63ecf2'


def force_envelope(norm: F, energy: F, c: F, current_rho: F, initial_pi: F,
                   initial_theta: F, g: F, clock: F) -> dict[str, F]:
    # Cauchy + conserved H+B >= c rho^4. Never infer Pi from TOTAL field energy.
    rho = upper_sqrt(upper_sqrt(norm*norm*energy/c))
    force = current_rho*rho
    pi = initial_pi+clock*force
    theta = initial_theta+2*g*clock*initial_pi+g*clock*clock*force
    return dict(source_norm_upper=norm, conserved_full_H_plus_B_form_upper=energy,
                alltime_global_Cart_rho_moment_upper=rho, alltime_current_force_moment_upper=force,
                initial_canonical_Pi_moment_upper=initial_pi, initial_canonical_Theta_moment_upper=initial_theta,
                clock_canonical_Pi_moment_upper=pi, clock_canonical_Theta_moment_upper=theta)


def run(current_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    raw = current_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'bc5b4e75c0da1fbc1de91062bdbf5b42fa084816860a226498c71b492afc69f8':
        raise ArithmeticError('Requires actual same-source 478 refinement epoch')
    current = json.loads(raw)
    inputs = current['input_descriptor']['input_receipts']
    _, retarded = checked(inputs['retarded'])
    _, norm_receipt = checked(inputs['source_norm'])
    rd = retarded['input_descriptor']
    _, field_clock = checked(rd['input_receipts']['clock'])
    _, bridge = checked(field_clock['input_descriptor']['input_receipts']['bridge'])
    bd = bridge['input_descriptor']
    _, mean = checked(bd['input_receipts']['mean'])
    if rd['source_abi'] != bd['source_abi'] or rd['N'] != 262144 or rd['variance'] != '2' or rd['clock'] != '4096':
        raise ArithmeticError('No Phi/carrier/source/clock substitution')
    for receipt, name in ((current, 'induced_phi_em_ygraph_current_source_retarded.py'), (retarded, 'induced_phi_em_ygraph_retarded_transfer.py'), (field_clock, 'induced_phi_em_ygraph_evolving_field_clock.py'), (bridge, 'induced_phi_em_ygraph_centre_bridge.py')):
        if receipt['source_at_import'][name]['sha256'] != hashlib.sha256(SOURCE_BYTES[name]).hexdigest():
            raise ArithmeticError('Physical source epoch changed')
    V, M, D = int(rd['V']), int(rd['M']), 90*int(rd['V'])
    s, a, g, T = F(rd['variance']), F(rd['a']), F(rd['UEphys'])/F(rd['a']), F(rd['clock'])
    chi, eta, Lambda = F(2)**-4096, F(2)**-2048, F(2)**8192
    if (V, M, a, g, bd['charged_total'], bd['neutral_heavy']) != (2**54, 2**55+1, F(1, 64), F(64), 64, 1):
        raise ArithmeticError('Original actual full law changed')
    c = Lambda/(8*a*V)
    if c != exact(retarded['global_quartic_coercivity']):
        raise ArithmeticError('Correct literal Cart TrPhiPhi/4 required')
    B = exact(retarded['recomputed_global_full_H_lower_shift'])
    source_graph = exact(retarded['recomputed_actual_full_H_source_graph_upper'])
    scalar_graph = exact(retarded['actual222_scalar_H_graph_upper'])
    href_graph = exact(retarded['recomputed_Href_source_full_graph_upper'])
    matter_price = exact(bridge['corrected_uniform_all_Theta_all_CAR_charged_Phi_freezing_action'])
    sqrt2, sqrt3, sqrts = upper_sqrt(F(2)), upper_sqrt(F(3)), upper_sqrt(s)
    R1 = sqrt2*exact(retarded['recomputed_all64_ray_jets'][0])
    pi0 = upper_sqrt(1/(4*s))+R1
    Jref = sqrt2*exact(bridge['reference_whole_Theta_current_operator_cap'])
    Jprice = sqrt2*exact(bridge['corrected_initial_whole_Theta_current_operator_price'])
    Jcap = Jref+Jprice
    Csame = exact(bridge['reference_whole_Theta_contact_operator_cap'])+exact(bridge['corrected_initial_whole_Theta_contact_operator_price'])
    initial_Jnorm = upper_sqrt(exact(current['actual_full_INITIAL_two_current_Gram_upper']))
    ref_Jnorm = upper_sqrt(F(norm_receipt['macro_reference_two_current_action_Gram_maximum_eigenvalue_upper']))
    rmin = exact(retarded['actual222_orbit_min'])
    metric = 4+512/rmin
    q = exact(field_clock['actual_all_CAR_colour_charge_norm_upper'])
    frame_price = exact(bridge['actual_222_000_Cart_frame_distance_upper'])
    frame_cap = upper_sqrt(2*exact(retarded['actual222_orbit_max']))+frame_price+2*eta
    rows1 = rows2 = F(0)
    for profile in build_native(mean, 2)['Pprofiles']:
        row = scalars(profile)
        rows1 += row['load']*sum((length*(abs(c0)+abs(cj)) for length, (c0, cj) in row['path'].items()), F(0))
        rows2 += row['load']*sum((length*length*(abs(c0)+abs(cj)) for length, (c0, cj) in row['path'].items()), F(0))
    current_rho = sqrt2*32*rows1
    # FULL initial-chart bound for arbitrary real EM direction and complex current.
    Cany = sqrt2*32*rows2*frame_cap
    Jbody_derivative = 64*upper_sqrt(F(V))*rows2*(2+frame_cap)+8*q*Jcap/rmin
    potential_cap = 2*scalar_graph+4*href_graph+2*matter_price
    Ebase = source_graph+B
    # Exact scalar IMS identity: Q_H(Theta psi)=Re<Hpsi,Theta²psi>+g.
    Etheta = source_graph*sqrt3*s+g+B*s
    # Positive kinetic form product rule; matrices need NOT commute derivatives.
    EJ = 2*Jcap*Jcap*Ebase+chi*metric*D*Jbody_derivative**2+2*g*M*Cany**2+Jcap*Jcap*(potential_cap+B)
    rows = dict(base=force_envelope(F(1), Ebase, c, current_rho, pi0, sqrts, g, T),
                initial_Theta_packet=force_envelope(sqrts, Etheta, c, current_rho, sqrt3/2+sqrts*R1+1, sqrt3*s, g, T),
                initial_FULL_current_packet=force_envelope(initial_Jnorm, EJ, c, current_rho, Jcap*pi0+Csame, Jcap*sqrts, g, T))
    epsilon = exact(field_clock['alltime_original_Phi_source_reference_wave_error'])
    packet_errors = dict(base=epsilon, initial_Theta_packet=sqrts*epsilon,
                         initial_Pi_packet=pi0*epsilon,
                         initial_FULL_current_packet=ref_Jnorm*epsilon+Jprice)
    physical_packets = dict(initial_physical_A_packet=dict(norm_upper=sqrts/a,
                        clock_canonical_Pi_moment_upper=rows['initial_Theta_packet']['clock_canonical_Pi_moment_upper']/a,
                        clock_physical_A_moment_upper=rows['initial_Theta_packet']['clock_canonical_Theta_moment_upper']/a**2,
                        wave_error=packet_errors['initial_Theta_packet']/a),
                      initial_physical_A_current_packet=dict(norm_upper=a*initial_Jnorm,
                        clock_canonical_Pi_moment_upper=a*rows['initial_FULL_current_packet']['clock_canonical_Pi_moment_upper'],
                        clock_physical_A_moment_upper=rows['initial_FULL_current_packet']['clock_canonical_Theta_moment_upper'],
                        wave_error=a*packet_errors['initial_FULL_current_packet']))
    if any(row['clock_canonical_Theta_moment_upper'] >= F(2)**256 for row in rows.values()) or max(packet_errors.values()) >= F(1, 1000):
        raise ArithmeticError('Force/insertion full-clock prices vacuous')
    desc = dict(plan=rd['plan'], plan_revision=REVISION, scientific_execution_revision=REVISION, input_execution_revision=rd['scientific_execution_revision'], input_receipts=dict(current=dict(path=str(current_path), sha256=hashlib.sha256(raw).hexdigest())), source_abi=bd['source_abi'], N=rd['N'], V=V, M=M, clock=str(T), variance=str(s), a=str(a), UEphys=rd['UEphys'], Lambda='2^8192', source='EXACTUNCHANGED b367/e8 originalPhi222Gaussian/b2/heavyprojected x canonicalY000/all1+63 Gaussian2; inserted Theta/Pi/fullJ packets ONLY, notnewpreparation', reference='UNCHANGEDHref000 tensorEXACTscalarPhi/SU4 Dirichlet222 actingonORIGINALfieldsource, no groundprojection orf9 comparator')
    return dict(input_descriptor=desc, input_sha256=hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest(), actual_full_source_force_rows={key: {name: positive_exact(value) for name, value in row.items()} for key, row in rows.items()}, original_Phi_source_inserted_packet_wave_errors={name: positive_exact(value) for name, value in packet_errors.items()}, physical_packet_rescalings={key: {name: positive_exact(value) for name, value in row.items()} for key, row in physical_packets.items()}, prices={key: positive_exact(value) for key, value in dict(c=c, B=B, Jref_ALLthetaCAR_operator_cap=Jref, FULL_initial_current_operator_cap=Jcap, FULL_current_initial_directional_contact_cap=Csame, FULL_current_arbitrary_EM_direction_contact_cap=Cany, initial_full_current_covariant_body_derivative_cap=Jbody_derivative, initial_full_potential_operator_cap=potential_cap, actual_global_current_rho_cap=current_rho).items()},
        full_H_force_and_Phi_moment_proof='ActualglobalH+B>=ALLpositivekinetic+c rho^4 (correctCartTr/4), retainsthePhi/SU4/EMPi2 positives. ForANYinsertedXi inD(H), conservedeplus=<Xi,(H+B)Xi>; ||rhoU Xi||^4<=||Xi||² eplus/c byCauchy. UsefullsourcegraphONLYasINITIALenergyupper, andDIVIDEbyactualstiffnessLambda: allPhi moments remainpolynomialV/clock, NO giantPhiZPE assignedtoEM. Thiscoarserho isNOTa tinyPhi-distance assertion. ExactpointwiseforceJf boundsqrt2*32sum_load Lsum|c| rho forANYCartPhi/SU4/Theta includesactualall64 andfullCAR, neutralheavyforce0. GlobalP7[H,Pi_f]=iJf =>strongWard viaweakbilinear+Riesz, Pi_f U Xi=U Pi_fXi-intU Jf U Xi, soPi<=initial+Tforce. Theta_R cutoff/energy-Fatou stronglimit givesTheta<=initial+2gT Pi_initial+gT²force. EM TOTALenergyupper isusedonlyforDOMAINexistence/Fatou, NEVERasnumericPi budget. No assumptionactualfullPhiwave staysinchart.',
        initial_inserted_packet_form_proof='BaseEplus<=sourcegraph+B. Theta scalarIMS exactEplus<=sourcegraph||Theta²Psi||+g+B||ThetaPsi||²; exactGaussian pointwisenormchargedray1 gives||ThetaPsi||=sqrt(s),||Theta²Psi||<=sqrt3*s. CurrentXi=JfullPsi FULLinitialmapnormfrom478 includingstrictb367price, notrefopnorm. InitialsupportONLY permitsJcap=Jref_ALLthetaCAR+initialprice andpotentialcap<=2scalargraph+4Hrefgraph+2matterprice (alloriginalquartic/alignment/curl/neutralheavy/chargedmatter/Coulomb abspriced). KineticproductruleQkin(JPsi)<=2Jcap²Qkin(Psi)+chi M D Gbody²+2g M Cany², NO scalar/current-derivativecommutation orH² domain. Bodyderivative boundactualCart/SU4 dexp+covariantcolourcommutator, q=66sqrt(5V/2) includesheavy/Ref/staticsectors, rminactual222; Gbody=64sqrtV sumloadL²cs(2+fieldcap)+8qJcap/rmin. EveryEMunitdirection mixedcurrentcontact boundedCany byCauchy inpathprofiles; Mdirectionsumcoarsebutpolynomial. Qkin(Psi)<=sourcegraph+B fromglobalcoercivity. AddJcap²(Vcap+B) forpotentialform. InitialPiPsi<=1/(2sqrt(s))+sqrt2*R1; PiThetaPsi<=sqrt3/2+sqrt(s)sqrt2R1+1. PiJfullPsi=JfullPiPsi-i(partial_fJfull)Psi withCsame, NOTundeclaredoperatorPi commutation. ThetaJfullPsi<=Jcap sqrt(s). Actualb2PhiC1/H2 givesalltheseD(H) P7packets; noD(H²).',
        unchanged_H0_Duhamel_and_packet_consumer='NEW76adreferenceORIGINALPhi-source evolvesEXACTscalarfieldH anditsenergy/coercivePhi-distance/Liemoments, Born/q andlastboundaryAgmon prices. ForANYTheta/CARinsertedvectorindependentPhi, exacttensorfactorization makesfieldenergy/highPhi/sourceboundarynorm SCALEbyitsHilbertnorm; residualform operatorbounds uniformALLCAR/allTheta. ThusHfull/commonH0 Duhamelwaveerror<=n epsilon onbase, initialTheta, initialPi andreferenceJ000 initialpacket. FULLinitialJ differsJ000 byb367Jprice, soCURRENTpacketerror<=Sref epsilon+Jprice (sameactual478). FullJ(Phi,U) NEVERcommutedthroughfieldlowprojection. EMforcepolynomialpricesapplytoFULLactualpackets afterinitialtime; noactualsupportpostulate, no fullPhiZPEtoEM. Thisconsumer provespropagatedINITIALinsertionpackets, NOTcurrentJ(t)Psi normconservation andNOTevaluatedHref retardedcurve. Existinge8BASE strongWard+weightedPhi comparison yieldsactualPi/Theta-response errors separately; thosecertificates remainfrozen.',
        physical_units_and_scope='Theta=aAphys; J_A=aJTheta, Pi_A=aPiTheta ifcanonicalconjugateisneeded. PhysicalA INSERTIONpacket=ThetaPsi/a, physicalCURRENT insertionpacket=aJThetaPsi; operatorA alsoTheta/a, henceA-on-Apacket boundTheta-on-Thetapacket/a² whileA-on-J_Apacket cancelsfactors. All|t|<=4096/allmacroallowedq/twocomplexpolarization. Parentreviewofglobalcoercivity, initialcovariantgradient, evolvedfield boundary/Duhamel andweakWard premises pending. No ground/vacuum/stationarity/fullHrefretardedvalues/photonpole/commoncone/Compton/statP-Q admission orfiniteSCREENgate.',
        source_at_import={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes_utf8=data.decode()) for name, data in SOURCE_BYTES.items()}, source_unchanged_since_import={name: (ROOT/name).read_bytes()==data for name, data in SOURCE_BYTES.items()}, runtime=dict(executable=sys.executable, python=sys.version), elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.current)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('input_sha256', 'actual_full_source_force_rows', 'original_Phi_source_inserted_packet_wave_errors', 'elapsed_seconds')}, indent=2))

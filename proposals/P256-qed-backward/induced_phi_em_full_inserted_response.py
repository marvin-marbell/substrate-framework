"""Unchanged source: full-Phi retarded field response on actual insertion packets."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from induced_phi_em_full_source_force_clock import SOURCE_BYTES as FORCE_BYTES
from induced_phi_em_ygraph_retarded_transfer import upper_sqrt
from induced_phi_em_ygraph_clock_transfer import checked, exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_macro_preparation import build_native, scalars
from induced_phi_ir_well_transfer import rational
from induced_reference_field_following import path_derivative

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**FORCE_BYTES, 'induced_reference_field_following.py': (ROOT/'induced_reference_field_following.py').read_bytes(), Path(__file__).name: Path(__file__).read_bytes()}
REVISION = '6ebc34a912158254456c535bd10259fca04201b14f446a778bba68d5ef63ecf2'


def run(force_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    raw = force_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'da4a963f339190a3059362fe0a09b0b3b467873802466e404c927ca52abaeffd':
        raise ArithmeticError('Requires actual e4 unchanged-source force epoch')
    force = json.loads(raw)
    fd = force['input_descriptor']
    _, current = checked(fd['input_receipts']['current'])
    _, retarded = checked(current['input_descriptor']['input_receipts']['retarded'])
    _, source_norm = checked(current['input_descriptor']['input_receipts']['source_norm'])
    rd = retarded['input_descriptor']
    _, field = checked(rd['input_receipts']['clock'])
    _, bridge = checked(field['input_descriptor']['input_receipts']['bridge'])
    _, mean = checked(bridge['input_descriptor']['input_receipts']['mean'])
    if rd['N'] != 262144 or fd['source_abi'] != rd['source_abi'] or fd['variance'] != '2' or fd['clock'] != '4096':
        raise ArithmeticError('Unchanged Phi/source/clock ABI required')
    for receipt, name in ((force, 'induced_phi_em_full_source_force_clock.py'), (current, 'induced_phi_em_ygraph_current_source_retarded.py'), (retarded, 'induced_phi_em_ygraph_retarded_transfer.py')):
        if receipt['source_at_import'][name]['sha256'] != hashlib.sha256(SOURCE_BYTES[name]).hexdigest():
            raise ArithmeticError('Actual physical source epoch changed')
    V, M, D = int(fd['V']), int(fd['M']), 90*int(fd['V'])
    s, a, g, T, chi = F(2), F(1, 64), F(64), F(4096), F(2)**-4096
    sqrt2, sqrt3, sqrts = upper_sqrt(F(2)), upper_sqrt(F(3)), upper_sqrt(s)
    fp = force['prices']
    c, B = exact(fp['c']), exact(fp['B'])
    Jref, Jcap = exact(fp['Jref_ALLthetaCAR_operator_cap']), exact(fp['FULL_initial_current_operator_cap'])
    Cany, Csame = exact(fp['FULL_current_arbitrary_EM_direction_contact_cap']), exact(fp['FULL_current_initial_directional_contact_cap'])
    Jbody, Vcap, Jrho = exact(fp['initial_full_current_covariant_body_derivative_cap']), exact(fp['initial_full_potential_operator_cap']), exact(fp['actual_global_current_rho_cap'])
    Jprice = sqrt2*exact(bridge['corrected_initial_whole_Theta_current_operator_price'])
    Cprice = exact(bridge['corrected_initial_whole_Theta_contact_operator_price'])
    nJ = upper_sqrt(exact(current['actual_full_INITIAL_two_current_Gram_upper']))
    nJref = upper_sqrt(F(source_norm['macro_reference_two_current_action_Gram_maximum_eigenvalue_upper']))
    rmin, rmax = exact(retarded['actual222_orbit_min']), exact(retarded['actual222_orbit_max'])
    metric = 4+512/rmin
    radius = min(F(1, 1000), rmin/(16*upper_sqrt(rmax)))
    gradient = exact(retarded['actual222_source_gradient_upper'])
    D1 = exact(field['actual_projected_heavy_D1_upper'])
    q = exact(field['actual_all_CAR_colour_charge_norm_upper'])
    R1 = sqrt2*exact(retarded['recomputed_all64_ray_jets'][0])
    pi0 = upper_sqrt(1/(4*s))+R1
    # Positive ORIGINAL kinetic form, not absolute total-H source graph or B.
    halfdensity_gradient = 4*D*upper_sqrt(F(D))/radius
    body_gradient = gradient+upper_sqrt(F(D))*D1+8*q/rmin+halfdensity_gradient
    initial_kinetic = chi*metric*body_gradient**2+g*M*pi0*pi0
    refined_EJ = 2*Jcap*Jcap*initial_kinetic+chi*metric*D*Jbody*Jbody+2*g*M*Cany*Cany+(Vcap+B)*nJ*nJ
    # B and potential multiply ACTUAL current-packet norm, never Joperator².
    Lmean = F(source_norm['total_conditional_current_mean_Gaussian_Lipschitz_upper'])
    enc = mean['centre_enclosure']
    g0, gs = rational(enc['g0_upper']), rational(enc['gs_upper'])
    Quniform = F(0)
    for profile in build_native(mean, 2)['Pprofiles']:
        row = scalars(profile)
        Quniform += 8*row['load']*sum((length*length*(c0*c0*g0+cj*cj*gs) for length, (c0, cj) in row['path'].items()), F(0))
    weighted_Jsource_Gram_upper = s*Quniform+upper_sqrt(F(15))*s*s*Lmean*Lmean
    theta_Jref = upper_sqrt(weighted_Jsource_Gram_upper)
    theta_Jfull = theta_Jref+sqrts*Jprice
    epsilon = exact(field['alltime_original_Phi_source_reference_wave_error'])
    epsilon_cut = exact(field['alltime_actual_full_source_to_cut_low_reference_norm_error'])
    gap = exact(field['original_reference_vs_cut_low_reference_norm_error'])
    residual = exact(field['alltime_cut_low_reference_generator_residual'])
    restart = 2*gap
    freeze_J = sqrt2*exact(field['alltime_reference_to_C000_current_operator_action_price'])
    field_abs = exact(retarded['absolute_reference_field_classical_offset_cap'])
    Ecut = exact(field['field_low_spectral_cut'])
    # Href graph is a positive SUM including all-CAR matter and Coulomb caps.
    matter_and_Coulomb_cap = exact(retarded['recomputed_Href_source_full_graph_upper'])
    packet_specs = dict(base=dict(n=F(1), nref=F(1), full_energy=exact(force['actual_full_source_force_rows']['base']['conserved_full_H_plus_B_form_upper']), pi_initial=pi0, theta_initial=sqrts, full_theta_initial=sqrts, pi_initial_difference=F(0), theta_initial_difference=F(0), wave_difference=epsilon, cut_wave_difference=epsilon_cut),
        initial_Theta_packet=dict(n=sqrts, nref=sqrts, full_energy=exact(force['actual_full_source_force_rows']['initial_Theta_packet']['conserved_full_H_plus_B_form_upper']), pi_initial=sqrt3/2+sqrts*R1+1, theta_initial=sqrt3*s, full_theta_initial=sqrt3*s, pi_initial_difference=F(0), theta_initial_difference=F(0), wave_difference=sqrts*epsilon, cut_wave_difference=sqrts*epsilon_cut),
        initial_FULL_current_packet=dict(n=nJ, nref=nJref, full_energy=refined_EJ, pi_initial=Jref*pi0+Csame, theta_initial=theta_Jref, full_theta_initial=theta_Jfull, pi_initial_difference=Jprice*pi0+Cprice, theta_initial_difference=sqrts*Jprice, wave_difference=nJref*epsilon+Jprice, cut_wave_difference=nJref*epsilon_cut+Jprice))
    response_rows = {}
    for name, p in packet_specs.items():
        # Positive Href EM energy, uniformly initial gradient; no Phi ZPE here.
        initial_any_EM_gradient = Jref*pi0+Cany if name == 'initial_FULL_current_packet' else p['pi_initial']
        href_energy = g*M*initial_any_EM_gradient**2+matter_and_Coulomb_cap*p['nref']**2
        ref_cut_energy = p['nref']**2*(B+field_abs+Ecut+residual)+href_energy
        delta_form = 2*(p['full_energy']+ref_cut_energy)
        rho_difference = upper_sqrt(upper_sqrt(p['cut_wave_difference']**2*delta_form/c))
        action_error = Jrho*rho_difference+freeze_J*p['nref']+Jref*gap*p['nref']
        pi_error = p['pi_initial_difference']+p['pi_initial']*epsilon+T*action_error+T*restart*Jref*p['nref']+T*T*residual*Jref*p['nref']/2
        ref_pi_time = p['pi_initial']+T*Jref*p['nref']
        theta_error = p['theta_initial_difference']+p['theta_initial']*epsilon+2*g*(T*pi_error+T*restart*ref_pi_time+T*T*residual*ref_pi_time/2)
        ref_theta_time = p['theta_initial']+2*g*T*p['pi_initial']+g*T*T*Jref*p['nref']
        initial_theta_packet_error = p['theta_initial_difference']+p['theta_initial']*epsilon
        # Inserted 4D tensor: Schmidt sqrt2/map; partial transpose costs <=2.
        tensor_factor = F(2) if name == 'base' else F(6)
        tensor_retarded_error = tensor_factor*(theta_error*p['full_theta_initial']+ref_theta_time*initial_theta_packet_error)
        response_rows[name] = dict(initial_packet_norm=p['n'], reference_initial_packet_norm=p['nref'], full_initial_form_energy=p['full_energy'], reference_cut_low_alltime_form_energy_upper=ref_cut_energy, full_minus_cut_low_form_energy_upper=delta_form, alltime_global_Phi_weighted_packet_difference=rho_difference, alltime_current_action_packet_difference=action_error, alltime_canonical_Pi_packet_wave_difference=pi_error, alltime_canonical_Theta_packet_wave_difference=theta_error, alltime_reference_Pi_force_bound=ref_pi_time, alltime_reference_Theta_force_bound=ref_theta_time, alltime_initial_Theta_inserted_packet_wave_difference=initial_theta_packet_error, alltime_canonical_Theta_retarded_tensor_operator_error=tensor_retarded_error)
    gram_lower = exact(bridge['corrected_full_initial_two_current_Gram_lower'])
    physical = dict(physical_A_retarded_on_original_source=response_rows['base']['alltime_canonical_Theta_retarded_tensor_operator_error']/a**2,
        physical_A_retarded_on_physical_A_source_packet=response_rows['initial_Theta_packet']['alltime_canonical_Theta_retarded_tensor_operator_error']/a**4,
        physical_A_retarded_on_physical_CURRENT_source_packet=response_rows['initial_FULL_current_packet']['alltime_canonical_Theta_retarded_tensor_operator_error'],
        EXACT_full_INITIAL_current_Gram_relative_canonical_Theta_response=response_rows['initial_FULL_current_packet']['alltime_canonical_Theta_retarded_tensor_operator_error']/gram_lower,
        EXACT_full_INITIAL_physical_CURRENT_Gram_relative_physical_A_response=response_rows['initial_FULL_current_packet']['alltime_canonical_Theta_retarded_tensor_operator_error']/(a*a*gram_lower))
    if max(physical.values()) >= F(1, 1000) or refined_EJ >= exact(force['actual_full_source_force_rows']['initial_FULL_current_packet']['conserved_full_H_plus_B_form_upper']):
        raise ArithmeticError('Actual inserted response comparison not useful')
    # Actual primitive onebody trace control; no synthetic mean/current surrogate.
    native = build_native(mean, 6)
    theta = native['Q']@np.random.default_rng(20261026).normal(0, 0.2, native['M'])
    f = np.cos(2*np.pi*native['sites'][:, 0]/6)*np.sqrt(2/native['V'])
    native_controls = []
    for profile in native['Pprofiles']:
        row = scalars(profile)
        expected = 8*sum((float(length*length)*(float(c0*c0)*native['g0']+float(cj*cj)*native['gs']) for length, (c0, cj) in row['path'].items()))
        actuals = []
        currents = []
        for axis in (1, 2):
            direction = np.zeros(3*native['V']); direction[axis::3] = f
            J = path_derivative(native, profile, theta, direction, 1)
            actuals.append(float(np.vdot(J, J).real))
            currents.append(J)
        cross = float(abs(np.vdot(currents[0], currents[1])))
        wrong = 8*sum((float(c0*c0)*native['g0']+float(cj*cj)*native['gs'] for c0, cj in row['path'].values()))
        if max(abs(x-expected) for x in actuals)>1e-7 or cross>1e-10 or abs(expected-wrong)<1:
            raise ArithmeticError('Native ALLpath trace/control failed')
        native_controls.append(dict(load=int(profile['load']), actual_whole_J_Frobenius_squared=actuals, exact_path_trace_formula_midpoint=expected, cross_y_z_Frobenius_inner_product_abs=cross, wrong_omitted_two_link_length_squared=wrong))
    desc = dict(plan=fd['plan'], plan_revision=REVISION, scientific_execution_revision=REVISION, input_receipts=dict(force=dict(path=str(force_path), sha256=hashlib.sha256(raw).hexdigest())), source_abi=fd['source_abi'], N=fd['N'], V=V, M=M, a=str(a), UEphys='1', clock=str(T), variance=str(s), source=fd['source'], reference=fd['reference'], native_control_N=6, response='ACTUALnonstationarypreparedsource retardedTheta/A fieldmatrix inbase, initialTheta/A andinitialFULLcurrent/J_A insertionpackets; commonexactfullPhi/Href comparison ONLY, no evaluatedHrefcurve')
    return dict(input_descriptor=desc, input_sha256=hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest(), source_prices={key: positive_exact(value) for key, value in dict(actual_source_positive_kinetic_form_upper=initial_kinetic, refined_FULL_initial_current_H_plus_B_form_upper=refined_EJ, uniform_all64_reference_quantum_PH_Gram_upper=Quniform, source_conditional_mean_complex_gradient_Lipschitz=Lmean, actual_reference_initial_Theta_current_Gram_upper=weighted_Jsource_Gram_upper, actual_reference_initial_Theta_current_map_norm_upper=theta_Jref, actual_FULL_initial_Theta_current_map_norm_upper=theta_Jfull).items()}, actual_inserted_response_rows={name:{key:positive_exact(value) if value>0 else dict(numerator='0', denominator='1', exact_zero=True) for key,value in row.items()} for name,row in response_rows.items()}, actual_physical_responses={key:positive_exact(value) for key,value in physical.items()}, native_current_trace_controls=native_controls,
        source_Theta_current_moment_proof='Supplier541dbALLtheta complex-gradientLipL=195.8888614, ownerconfirmedcomplexg Frechet/nuclear extension; sameunfilteredS2/AP222/all64chi000. Forunitcomplexcurrentf, mu=SUMloadsTracePJf, E mu=0 bytwistedtranslations; ||grad mu||C<=L uniformly. GaussianPoincare E|mu|²<=sL²; Var|mu|²<=4sL² E|mu|² henceE|mu|4<=5s²L4. E|Theta_g|4<=3s² =>E|Theta_g|²|mu|²<=sqrt15 s²L². POINTWISEquantumPH<=SUMloads||Jf||F²=8SUMload,L L²(c0²g0+cj²gs), macroN>4 distinct1/2/endpoints axesorthogonal; usesfullCAR/sea andcomplexf, NOT6912integratedPH17755 underTheta-weight. ExactSlater secondmoment=PH+|TOTALmu|², soTheta-currentGram<=sQuniform+sqrt15s²L². FullPhi priceTheta(Jfull-J000)Psi<=sqrt(s)b367Jprice, sincepointwisechargedraynorm1 andunchangedGaussian. SOURCEThetaJ momentONLY, NOTTheta(t)J(t), globalJop orfilteredstate.',
        refined_initial_current_form='Positivekineticform directlyoriginalchiM body gradient plusg fullEMgradient: scalarGaussianactualgradient, projectedheavycomponentD1, covariantcolourq/rmin, andflatweight halfdensitygradient<=4D^(3/2)/chartRadius. CountD forcomponentsconservative. FullJ kinetic productrule retainsJoperatorcap ONLYmultiplyingpositivekineticbudget, NOTtotalPhi baseline/B. PotentialandB exactlymultiply||JfullPsi||²<=478actualINITIALGramupper. Thisremovesavoidable63bits frome4coarsecurrentform; no changeofsource/H. Phi quartic/globalB usedONLYforrho moment, notdirectEMPimoment.',
        form_only_current_action_and_Ward='ForfullXi(t)-CUTLOWrefXi(t)=delta inFORMdomain, globalH+B>=c rho^4, ||rho delta||4<=||delta||² Q_H+B(delta)/c. Trianglepositiveform Q(delta)<=2(QfullXi+QrefcutXi); fullformconservedINITIALactualsource/current/Thetaforms above. Referencecutfield spectralbandEcut+absoluteactualPhi222baseline/Born/boundary+Href INITIALenergy (ONLYEM/CAR), alltimesfrozenH0. Referencealltheta/currentfreezepriceNEW76ad evolvesORIGINALPhi source, notinitialetasupport; currentactionerror=Jrho weightedPhiDelta+movingPhi_Jprice*nref+Jref*HIGHgap*nref. No new H²/current source graphneeded. StrongWardcompareeachinsertedpacket withsameH0/Duhamelrestart2HIGHgap+tauR. InitialPi/ThetaCURRENTdifferences=Jprice PiSOURCE+Cprice andsqrt(s)Jprice; nofakezero. ReferenceforcePi_initial+T Jref*nref usesONLYtruecurrentoperator, notPhiZPE. ThetaWardintegratesPierror plusrestartprice. ActualcurrentpacketnormisinitialSrefconserved, but Jref UrefXi(t) usesALLtheta/CARopcap. ALL|t|<=4096, ANYmacroallowedq/twoCOMPLEXpolarizations.',
        retarded_tensor_and_units='F(alpha i,beta j)=<Theta_i U Xi_alpha,UTheta_j Xi_beta>. ReverseorderedG(alpha i,beta j)=conjF(beta i,alpha j)=(Fdag)PARTIALtranspose_photon, NOTentrywiseconjF whenSOURCESLOTSretained. For4D source2×photon2, partialtransposeoperatornorm<=Frobnorm<=2opnorm; R=-i(F-G) error<=3orderederror. Schmidt rank2 bounds arbitrarytensorby sqrt2 permap, orderederror<=2[ThetaWaveError*FullINITIALThetaXiNorm+RefThetaTimeNorm*PropagatedINITIALThetaPacketError], henceSIXbrackets. BaseuniqueSOURCE hasentrywiseconjugationisometric, TWO brackets. FULLPhi actualnonstationaryretardedcomparison toUNCHANGEDHref×ORIGINALevolvingPhi, NOTcontacts/Taylor/survival. PhysicalTheta=aA; physicalinitialApacket=ThetaPsi/a, currentJ_Apacket=aJThetaPsi. ThusAresponseonApacket=canonical/a4; AresponseonJApacket=a²/a² cancels. SameEXACTfullINITIALcurrentGramwhitening givesrelativecanonicalerror/g_lower, physicalrelativeAerror/(a²g_lower), NOTnormalizingwithboundinsteadactualGram. Current-sectorGramNOTnewparticlepole/receiver.',
        boundary='Parentreviewrequiredfornewkineticgradient/fieldform/globalcoercivity/Gaussianweightedmoment andprevious76adBorn/spectralboundary/strongWard premises. ActualCLIexecutesstrictfiniteMACROcomparisonpricesandnativeprimitiveALLpathtrace; DOESNOTnumericallyevaluateHref/fullHresponsecurve norproveground/stationarity/vacuum/photonpole/dispersion/commoncone/Compton/statP-Q. NofiniteSCREENstate/receiver/probability/gatechange. Oldallsourceepochsunchanged.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(), bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()}, source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()}, runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__), elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.force)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256', 'source_prices', 'actual_physical_responses', 'native_current_trace_controls', 'elapsed_seconds')}, indent=2))

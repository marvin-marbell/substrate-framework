"""Fresh same-source graphs and finite-clock retarded Theta/current transfer."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_ygraph_evolving_field_clock import SOURCE_BYTES as CLOCK_BYTES
from induced_phi_em_ygraph_clock_transfer import checked, exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_macro_preparation import build_native, scalars
from induced_phi_ir_well_transfer import rational, sqrt_bounds

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**CLOCK_BYTES, Path(__file__).name: Path(__file__).read_bytes()}
REVISION = 'b9e13b2491321aa804f7e52cf4ba3cb12c0a224717cb07b23e487afc0768463b'


def upper_sqrt(x: F) -> F:
    return sqrt_bounds(x)[1]


def run(clock_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    raw = clock_path.read_bytes()
    clock = json.loads(raw)
    cd = clock['input_descriptor']
    _, bridge = checked(cd['input_receipts']['bridge'])
    bd = bridge['input_descriptor']
    abi = bd['source_abi']
    if (abi['Phi_carrier_root_selector'], abi['charged_preparation_root_selector'], abi['reference_Hamiltonian_root_selector'], abi['physical_CAR_central_seams']) != ([2, 2, 2], [0, 0, 0], [0, 0, 0], [2, 2, 2]):
        raise ArithmeticError('No carrier/ray/reference/seam substitution')
    if abi['canonical_Y_source_reference_ray_norm_difference_exact'] != '0' or F(cd['clock']) != 4096 or bd['N'] != 262144 or bd['V'] != 262144**3 or bd['M'] != 2*bd['V']+1:
        raise ArithmeticError('Actual source and clock required')
    if bd['a'] != '1/64' or bd['UEphys'] != '1' or bd['Lambda'] != '2^8192' or bd['charged_total'] != 64 or bd['neutral_heavy'] != 1:
        raise ArithmeticError('No microscopic law substitution')
    for receipt, name in ((clock, 'induced_phi_em_ygraph_evolving_field_clock.py'), (bridge, 'induced_phi_em_ygraph_centre_bridge.py')):
        if receipt['source_at_import'][name]['sha256'] != hashlib.sha256(SOURCE_BYTES[name]).hexdigest():
            raise ArithmeticError('Centre-paid source epoch changed')
    _, energy = checked(bd['input_receipts']['energy'])
    _, initial = checked(energy['input_descriptor']['input_receipts']['initial'])
    _, domain = checked(initial['input_descriptor']['input_receipts']['domain'])
    ds = domain['input_descriptor']
    _, graph = checked(ds['input_receipts']['graph'])
    _, pure = checked(ds['input_receipts']['purePhi'])
    _, well = checked(pure['input_descriptor']['input_receipts']['well'])
    _, mean = checked(bd['input_receipts']['mean'])
    _, refiner = checked(bd['input_receipts']['refiner'])
    p = well['prices']
    V, Mtheta, D = bd['V'], bd['M'], 90*bd['V']
    a, g, s, T = F(1, 64), F(64), F(2), F(4096)
    L, chi, eta = F(2)**8192, F(2)**-4096, rational(p['eta'])
    delta = exact(bridge['actual_222_000_Gram_Frobenius_distance_upper'])
    frame = exact(bridge['actual_222_000_Cart_frame_distance_upper'])
    enc = mean['centre_enclosure']
    gmin = min(rational(enc['g0_lower']), rational(enc['gs_lower'])) - delta
    traceG = rational(enc['g0_upper']) + 3*rational(enc['gs_upper']) + 2*delta
    rmin, rmax = gmin/2, traceG/2
    stiffness_delta = rational(refiner['normal_stiffness_twist_difference_upper'])
    amin = rational(mean['slow_stiffness_lower']) - stiffness_delta
    amax = rational(mean['slow_stiffness_upper']) + stiffness_delta
    radius = min(F(1, 1000), rmin/(16*upper_sqrt(rmax)))
    metric, metric1 = 4+512/rmin, 8*(4+512/rmin)/radius
    log1, log2 = 8/radius, 32/radius**2
    Uscalar = chi*(D**2*(metric1*log1+metric*log2)/4+D*metric*log1**2/8)
    lapmin = rational(p['smallest_laplacian_lower'])
    if min(gmin, amin, radius-eta, lapmin) <= 0 or D != int(rational(p['dimension'])):
        raise ArithmeticError('Actual orbit/stiffness/chart failed')
    omega_lower = sqrt_bounds(3*rmin/25)[0]
    fast_variance = max(1/(2*sqrt_bounds(lapmin/5)[0]), F(3, 10)/omega_lower, (F(3, 5)+12/rmin)/(2*omega_lower))
    radius2 = (D-45)*fast_variance*F(2)**-7168 + 45/(2*sqrt_bounds(amin)[0])*F(2)**-6144
    cut_deficit = 4*radius2/eta**2
    if cut_deficit >= F(1, 100):
        raise ArithmeticError('Actual222 Gaussian cut normalization failed')
    cut_lower = 1-cut_deficit
    precision = 90*upper_sqrt(amax)*F(2)**6144
    precision += (D-45)*max(2*upper_sqrt(F(12, 5)), F(10, 3)*upper_sqrt(F(3, 25)*rmax+F(63, 25)))*F(2)**7168
    gradient_uncut = upper_sqrt(precision)/2
    gradient = (gradient_uncut+4/eta)/sqrt_bounds(cut_lower)[0]
    hessian = ((upper_sqrt(F(3))+2)*precision/4+8*gradient_uncut/eta+12/eta**2)/sqrt_bounds(cut_lower)[0]
    scalar_kinetic_graph = chi*(D*metric*hessian+D*metric1*gradient)/2
    heavy_cap, light_cap = rational(p['scalar_path_heavy_cap']), rational(p['scalar_path_all_charged_cap'])
    root0, roots = upper_sqrt(rational(enc['g0_upper'])), upper_sqrt(rational(enc['gs_upper']))
    field_cap = max(root0, roots)+frame+2*eta
    quartic_cap = 16*L*V*field_cap**4/a
    alignment_cap = F(2)**10240*12*V*(4+4*field_cap)**2*eta**2/10
    curl_cap = 7*F(2)**10240*3*V*eta**2/10
    heavy_energy_cap, static_light_cap = 2*L*V*heavy_cap*field_cap, 2*V*light_cap*field_cap
    scalar_graph = scalar_kinetic_graph+quartic_cap+alignment_cap+curl_cap+heavy_energy_cap+static_light_cap+Uscalar
    r1 = r2 = r3 = F(0)
    rank = 2*V
    jet_rows = []
    for profile in graph['profiles']:
        y = upper_sqrt(F(profile['graph_norm_squared_upper_interval']['upper']))
        b1 = b2 = b3 = y
        B1, B2, B3 = 2*y*b1, 2*y*b2+2*b1*b1, 2*y*b3+6*b1*b2
        A1, A2, A3 = B1/2, B2/2+F(3, 4)*B1**2, B3/2+F(9, 4)*B1*B2+F(15, 8)*B1**3
        block = upper_sqrt(1+y*y)
        w1, w2, w3 = b1+block*A1, b2+2*b1*A1+block*A2, b3+3*b2*A1+3*b1*A2+block*A3
        ray1, ray2 = rank*w1, rank*w2+rank*(rank-1)*w1**2
        ray3 = rank*w3+3*rank*(rank-1)*w2*w1+rank*(rank-1)*(rank-2)*w1**3
        load = int(profile['load'])
        copy1, copy2 = load*ray1, load*ray2+load*(load-1)*ray1**2
        copy3 = load*ray3+3*load*(load-1)*ray2*ray1+load*(load-1)*(load-2)*ray1**3
        r1, r2, r3 = r1+copy1, r2+2*r1*copy1+copy2, r3+3*r2*copy1+3*r1*copy2+copy3
        jet_rows.append(dict(load=load, Y_cap=positive_exact(y), W_jets=[positive_exact(x) for x in (w1, w2, w3)]))
    if [int(row['load']) for row in graph['profiles']] != [1, 63] or graph['N'] != bd['N'] or graph['real_EM_coordinates'] != Mtheta or F(graph['density_variance']) != s:
        raise ArithmeticError('No charged source or photon variance substitution')
    pi2_graph = upper_sqrt(F(Mtheta*(Mtheta+2)))/(4*s)+r1*upper_sqrt(F(Mtheta)/s)+Mtheta*r2
    matter_ref_cap = 2*V*light_cap*max(root0, roots)
    Coulomb_cap = g*V*132**2/lapmin
    Href_graph = g*pi2_graph+matter_ref_cap+Coulomb_cap
    ref_energy_upper = V*F(graph['actual_Href_energy_per_V_enclosure']['upper'])
    Pi_ref = upper_sqrt((ref_energy_upper+matter_ref_cap)/g)
    D1 = exact(clock['actual_projected_heavy_D1_upper'])
    D2 = exact(clock['actual_projected_heavy_D2_upper'])
    q = exact(clock['actual_all_CAR_colour_charge_norm_upper'])
    if rational(pure['new_projected_section_transfer']['projected_overlap_squared_lower']) < F(1, 4):
        raise ArithmeticError('Actual projected heavy line invalid')
    fibre_source = chi*(4*metric*gradient*(D1+q/rmin)+metric*D2+D*metric1*(D1+q)+metric*q*q/rmin)
    matter_price = exact(bridge['corrected_uniform_all_Theta_all_CAR_charged_Phi_freezing_action'])
    residual = exact(clock['alltime_cut_low_reference_generator_residual'])
    epsilon_cut = exact(clock['alltime_actual_full_source_to_cut_low_reference_norm_error'])
    epsilon = exact(clock['alltime_original_Phi_source_reference_wave_error'])
    reference_cut_gap = exact(clock['original_reference_vs_cut_low_reference_norm_error'])
    restart_base = 2*reference_cut_gap
    Hfull_source_graph = scalar_graph+Href_graph+matter_ref_cap+fibre_source+2*matter_price
    # Pays absolute classical field baseline/shift, NOT a guessed field ground phase.
    field_offset_abs = scalar_graph+exact(clock['actual222_field_scalar_shift'])+matter_ref_cap
    Hfull_ref_alltime_graph = field_offset_abs+exact(clock['field_low_spectral_cut'])+Href_graph+residual
    c = L/(8*a*V)
    matter_L = 8*upper_sqrt(F(V))*(L*heavy_cap+light_cap)
    if matter_L < c:
        raise ArithmeticError('Global Young regime changed')
    B = 1+matter_L**2/c
    difference_graph = Hfull_source_graph+Hfull_ref_alltime_graph+B*epsilon_cut
    weighted_Phi_error = upper_sqrt(upper_sqrt(epsilon_cut**3*difference_graph/c))
    Jrho = F(0)
    for profile in build_native(mean, 2)['Pprofiles']:
        row = scalars(profile)
        Jrho += 32*row['load']*sum((length*(abs(c0)+abs(cj)) for length, (c0, cj) in row['path'].items()), F(0))
    sqrt2 = upper_sqrt(F(2))
    Jglobal, Jref = sqrt2*Jrho, sqrt2*exact(bridge['reference_whole_Theta_current_operator_cap'])
    Jinitialprice = sqrt2*exact(bridge['corrected_initial_whole_Theta_current_operator_price'])
    Jevolvedprice = sqrt2*exact(clock['alltime_reference_to_C000_current_operator_action_price'])
    current_action_error = Jglobal*weighted_Phi_error+Jevolvedprice+Jref*reference_cut_gap
    Pi_wave_error = Pi_ref*epsilon+T*current_action_error+T*restart_base*Jref+T*T*residual*Jref/2
    Theta_wave_error = upper_sqrt(s)*epsilon+2*g*(T*Pi_wave_error+T*restart_base*Pi_ref+T*T*residual*Pi_ref/2)
    Theta_ref_norm = upper_sqrt(s)+2*g*T*Pi_ref
    Theta_comm_error = 2*upper_sqrt(s)*(Theta_wave_error+Theta_ref_norm*epsilon)
    current_packet_error = Jref*epsilon+Jinitialprice
    full_initial_current_norm = Jref+Jinitialprice
    current_ordered_error = current_action_error*full_initial_current_norm+Jref*current_packet_error
    current_comm_error = 2*current_ordered_error
    gram_lower = exact(bridge['corrected_full_initial_two_current_Gram_lower'])
    if gram_lower <= 0 or Theta_comm_error/a**2 >= F(1, 1000) or current_comm_error/gram_lower >= F(1, 1000):
        raise ArithmeticError('Finite-clock retarded transfer not useful')
    values = dict(actual222_orbit_min=rmin, actual222_orbit_max=rmax, actual222_covariance_cut_deficit=cut_deficit, actual222_Gaussian_precision_trace_upper=precision, actual222_source_gradient_upper=gradient, actual222_source_Hessian_operator_L2_upper=hessian, actual222_scalar_H_graph_upper=scalar_graph, recomputed_Href_source_Pi_squared_graph_upper=pi2_graph, recomputed_Href_source_full_graph_upper=Href_graph, reference_alltime_Pi_norm_upper=Pi_ref, recomputed_actual_full_H_source_graph_upper=Hfull_source_graph, absolute_reference_field_classical_offset_cap=field_offset_abs, full_H_cut_low_reference_alltime_graph_upper=Hfull_ref_alltime_graph, global_quartic_coercivity=c, recomputed_global_full_H_lower_shift=B, new_actual_full_H_difference_shifted_graph_upper=difference_graph, alltime_Phi_weighted_actual_source_to_cut_low_reference_error=weighted_Phi_error, alltime_two_current_action_map_error=current_action_error, alltime_Pi_probe_wave_error=Pi_wave_error, alltime_Theta_probe_wave_error=Theta_wave_error, alltime_canonical_Theta_retarded_matrix_operator_error=Theta_comm_error, alltime_physical_A_retarded_matrix_operator_error=Theta_comm_error/a**2, alltime_canonical_current_retarded_matrix_operator_error=current_comm_error, alltime_physical_A_current_retarded_matrix_operator_error=a**2*current_comm_error, alltime_full_initial_current_Gram_relative_retarded_operator_error=current_comm_error/gram_lower)
    desc = dict(plan=bd['plan'], plan_revision=REVISION, scientific_execution_revision='03b34911221ba1898315373edaa7c4c3bcac718462e7fe51a842560eb6933f87', input_receipts=dict(clock=dict(path=str(clock_path), sha256=hashlib.sha256(raw).hexdigest())), source_abi=abi, N=bd['N'], V=V, M=Mtheta, clock=str(T), variance=str(s), a=str(a), UEphys='1', Lambda='2^8192', probe='ANY unit complex polarization in real cos(qx) y/z span, all allowed nonzero/nonNyquist macrointegerq', source='UNCHANGED actualPhi222 Gaussian/b2/projectedheavy x Gaussian2/all64 canonicalY000; ORIGINALPhi source evolvesexactDirichletscalarPhi/SU4 x SAMEHref000; lowcutONLYboundaryproof, highsourcepaid, no ground/vacuum/stationarity')
    return dict(input_descriptor=desc, input_sha256=hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest(), **{key: positive_exact(value) for key, value in values.items()}, recomputed_all64_ray_jets=[positive_exact(x) for x in (r1, r2, r3)], ray_jet_rows=jet_rows,
        source_graph_proof='FRESH same-source full/Href graphs, NOT oldgraph inheritance. Actual222 R0 minG/2,TrG/2; all45normalstiffness+/-refinerδ andall90Vcovariance giveTraceP,cutZ andGaussian gradient/Hessian. Hess=(Px PxT/4-P/2)g, E|Px|⁴<=3(TrP)²; b2cut db<=4/eta,D²b<=12/eta², scalarflatdivkin<=chi/2[D M Hess+D M1 grad]. RECOMPUTEDmetric/HaarFPscalar and actualallsitequartic,12Valignment,3Vcurl,neutralheavy/staticlight bounds paysourceHscalar graph. NEWfieldclock heavyD1/D2/q and2matterWeyl payfullsourcegraph. Sourcegraph conservedunderactualfullH. ReferenceCUTLOWPhi evolvesEXACTfieldHamiltonian fromORIGINALpreparedPhigaussian, notground: Hshift spectralband<=Ecut; absoluteclassicalV222 cap<=scalargraph, fieldshiftpaid, subtractlightfree000<=matter_ref_cap. ThusHfullcutlowgraph<=classicaloffset+shift+Ecut+Hrefsourcegraph+NEWfieldresidual. ReferenceuncutDirichletwave needNOTlieinD(Hfull); HeinzappliesONLYactualfullwave minusCUTLOWlift, BOTHinD(Hfull). Actualall64Y000rayjets3 recomputed; Pi²Gaussiangraphsqrt(M(M+2))/(4s)+R1sqrt(M/s)+M R2, boundedmatter/Coulomb payHrefsourcegraph. No hugefieldphase rounded or oldrelativeenergyusedasabsolute.',
        global_weighted_current_proof='LiteralCartG=TrPhiPhi/4,rho²=sumCartcoeff². Quartic>=Lambda rho⁴/(4aV), retainhalf c=Lambda/(8aV). Tracelessonsite/forwardreverseCARnuclearbound =>matter>=-Lrho; L=8sqrtV(Lambdaheavycap+lightcap), B=1+L²/c, H+B>=c rho⁴+gPi². Fordelta=actualFullPsi−cutLOWreference inD(H), Cauchytwice ||rho delta||⁴<=||delta||³||(H+B)delta||/c; norm usesNEWcutlowerror, graph usesconservedactualsource+uniformevolvedcutlowgraph+B cutlowerror. No actualfullsupport assumption. GlobalcurrentJrho=32sum_load_length Lsum|c|, two-complexpolarization sqrt2. ReferenceCUTLOW chargedfreezeaction isNEWfieldclock conservedPhi-distance/linkLiemoment price, notinitialeta price. ComparingtoORIGINALunitarilyevolvedfullPhireference addsJref*(highnorm+tail). Hencealltimecurrentactionerror=JglobalweightedPhi+Jevolvedprice+Jrefreference_cut_gap; actualfieldsource HIGHpart neverreplacedwithoutprice.',
        vector_Ward_and_domain_proof='GlobalP7core [H,Pi]=iJ extendsbyPi/J Hgraphcontinuity; differentiatingbilineargraphforms thenRiesz givesSTRONGHilbert PiU(t)Psi=U(t)PiPsi-intU(t-s)J U(s)Psi, noPiUpsi inD(H) assertion. Theta_R smoothbounded derivative1/secondO(1/R), energy/Fatou stronglimit yieldsThetaU=UTheta+2gintU PiU, noThetaUpsi inD(H) assertion. Href boundedW3,infty andactualY000jets3/Gaussian implyH³invariance viaSobolevcommutator/Gronwall; PiUrefchi inH², JrefW2,infty preservesH². ReferencePhi isORIGINALGaussian evolvedEXACTscalarPhi/SU4 Hfield; itsconservedshiftedenergy/highnorm anduniformlowAgmon allowRESTARTatANYs withsamebounds. Restartpropagatorcomparison <=restart_base+|tau|residual, restart_base=2(highnorm+tail), NOTsilentlytau*residual. ActualHrefsourceenergyupper +fixedmatterlower/Coulombpositive givealltimePi_ref. Pi_wave<=Pi_ref epsilon+T current_action_error+T restart_base Jref+T²residualJref/2. Theta_wave<=sqrt(s)epsilon+2g[T Pi_wave+T restart_base Pi_ref+T²residualPi_ref/2]. Pi/Theta commutefieldisometry; fieldunitarity notgroundnorm suppliesbare/referencepacketcomparisons. FullH² neverassumed.',
        retarded_Theta_consumer='ACTUALpreparedsource Dij=-i<Psi,[Theta_i(t),Theta_j]Psi>, NONstationaryallowed. ExactinitialThetaactionmapGramsI2, s=2 pointwisechargedraynorm1; field/originalPhi-sourceunitarity preservesnorm1. D=-i(<Theta_i Upsi,UTheta_jpsi>-<UTheta_jpsi,Theta_iUpsi>). ReferenceThetaUrefnorm<=sqrt(s)+2gTPi_ref; two-complexpolarizationwaveerror andpropagatedThetainsertionerror<=sqrt(s)epsilon implyOPERATORmatrixerror<=2sqrt(s)[Theta_wave_error+Theta_ref_norm epsilon]. FullreferencePhitimeevolution cancelsbyisometry/unitarity initsretardedinnerproduct, notgroundpostulate orscalarE_D phase. PhysicalA=Theta/a, Aresponseerror=Thetaerror/a². ALLtimes|t|<=4096, notTaylor/contact/survivalproxy. Causalindicator preservesbound.',
        retarded_current_consumer='RawphysicalHermitianJi=partial_Theta_i H, no complexpolarframeHermiticity assumption. OrderedFij=<JiUpsi,UJjpsi>; initialVfullmapnorm<=Jref+Jinitialprice, propagatedVmapdifference<=Jrefepsilon+Jinitialprice. Lactiondifference usesNEWmovingPhi/weightedfullfield current_action_error. Orderedmatrixerror<=current_action_error*(Jref+Jinitialprice)+Jref*(Jrefepsilon+Jinitialprice). RetardedR=-i(F-conjF), normerror<=2orderederror evenNONstationary. FullINITIALcurrentGram exactpositivefunctional, NEWb367lower; samephysicalrealcurrentbasis andEXACTGfull^-1/2(Rfull-Rref)Gfull^-1/2 error<=rawerror/g_lower, no replacingGram bylower. PhysicalAcurrent=aJTheta, rawerror*a²; exactGram-relativeerrorunchanged.',
        review_boundary='NEW SAMEevolvedPhi-source finiteN original-law retardedTheta/current construction forparentreview ofACTUALscalarcoerciveform/sourceenergy/spectralboundary,normalmetric/Born/globalcore/newgraph/vectorWard premises. CLIcomputesstrictsource-specific prices, NOTnumericalHref/fullHretardedvalues. No fieldground/sourcevacuum/stationarity/commoncone/pole/Compton/statP-Q promotion. Groundcomparatorf9 NOTconsumerinput; oldunadmitted4b1/B158/4e0/72a unchanged. Fundamentalquantumlawandparametersdeclaredinputs.',
        source_at_import={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes_utf8=data.decode()) for name, data in SOURCE_BYTES.items()}, source_unchanged_since_import={name: (ROOT/name).read_bytes()==data for name, data in SOURCE_BYTES.items()}, runtime=dict(executable=sys.executable, python=sys.version), elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.clock)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('input_sha256', 'alltime_physical_A_retarded_matrix_operator_error', 'alltime_full_initial_current_Gram_relative_retarded_operator_error', 'elapsed_seconds')}, indent=2))

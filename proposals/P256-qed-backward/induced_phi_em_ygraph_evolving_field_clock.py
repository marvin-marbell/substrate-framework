"""Actual Phi-source evolution, coercive moments and last-boundary transfer."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_ygraph_centre_bridge import SOURCE_BYTES as BRIDGE_BYTES
from induced_phi_em_ygraph_clock_transfer import checked, exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_macro_preparation import build_native, scalars
from induced_phi_ir_well_transfer import rational, sqrt_bounds, binary_exponent

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**BRIDGE_BYTES, Path(__file__).name: Path(__file__).read_bytes()}
REVISION = 'b9e13b2491321aa804f7e52cf4ba3cb12c0a224717cb07b23e487afc0768463b'


def su(x: F) -> F:
    return sqrt_bounds(x)[1]


def sl(x: F) -> F:
    return sqrt_bounds(x)[0]


def run(bridge_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    raw = bridge_path.read_bytes()
    bridge = json.loads(raw)
    bd = bridge['input_descriptor']
    abi = bd['source_abi']
    if (abi['Phi_carrier_root_selector'], abi['charged_preparation_root_selector'], abi['reference_Hamiltonian_root_selector'], abi['physical_CAR_central_seams']) != ([2, 2, 2], [0, 0, 0], [0, 0, 0], [2, 2, 2]) or abi['canonical_Y_source_reference_ray_norm_difference_exact'] != '0':
        raise ArithmeticError('Actual carrier222/preparation000/reference000/seams222 required')
    if bd['N'] != 262144 or bd['V'] != 262144**3 or bd['M'] != 2*bd['V']+1 or bd['a'] != '1/64' or bd['UEphys'] != '1' or bd['Lambda'] != '2^8192':
        raise ArithmeticError('No box/physical law substitution')
    if bridge['source_at_import']['induced_phi_em_ygraph_centre_bridge.py']['sha256'] != hashlib.sha256(SOURCE_BYTES['induced_phi_em_ygraph_centre_bridge.py']).hexdigest():
        raise ArithmeticError('Centre-paid bridge source epoch changed')
    _, energy = checked(bd['input_receipts']['energy'])
    _, initial = checked(energy['input_descriptor']['input_receipts']['initial'])
    _, domain = checked(initial['input_descriptor']['input_receipts']['domain'])
    _, pure = checked(domain['input_descriptor']['input_receipts']['purePhi'])
    _, well = checked(pure['input_descriptor']['input_receipts']['well'])
    _, mean = checked(bd['input_receipts']['mean'])
    _, refiner = checked(bd['input_receipts']['refiner'])
    p = well['prices']
    V, D, T = bd['V'], 90*bd['V'], F(4096)
    a, L, chi, eta = F(1, 64), F(2)**8192, F(2)**-4096, rational(p['eta'])
    delta = exact(bridge['actual_222_000_Gram_Frobenius_distance_upper'])
    frame = exact(bridge['actual_222_000_Cart_frame_distance_upper'])
    enc = mean['centre_enclosure']
    gmin = min(rational(enc['g0_lower']), rational(enc['gs_lower']))-delta
    traceG = rational(enc['g0_upper'])+3*rational(enc['gs_upper'])+2*delta
    rho, rmin, rmax = su(traceG), gmin/2, traceG/2
    twist_stiffness = rational(refiner['normal_stiffness_twist_difference_upper'])
    amin, amax = rational(mean['slow_stiffness_lower'])-twist_stiffness, rational(mean['slow_stiffness_upper'])+twist_stiffness
    radius = min(F(1, 1000), rmin/(16*su(rmax)))
    metric, metric1 = 4+512/rmin, 8*(4+512/rmin)/radius
    log1, log2 = 8/radius, 32/radius**2
    U = chi*(D**2*(metric1*log1+metric*log2)/4+D*metric*log1**2/8)
    lapmin = rational(p['smallest_laplacian_lower'])
    hfast, omega_min = min(lapmin/5, rmin/5), sl(3*rmin/25)
    heavy = rational(p['scalar_path_heavy_cap'])
    light = rational(p['scalar_path_all_charged_cap'])
    K1, K2, K3 = 8*heavy*(1+rho), 16*heavy*(1+rho), 32*heavy*(1+rho)
    heavy_gap = 2*F(1, 5)*sl(gmin)-4*K1*eta
    if min(gmin, amin, radius-eta, hfast, heavy_gap) <= 0 or D != int(rational(p['dimension'])):
        raise ArithmeticError('Actual field/frame/gap premise failed')
    C = 24*(1+rho)**2/a+16*K2+64*K1**2/heavy_gap
    charged_third = F(0)
    native_profiles = build_native(mean, 2)['Pprofiles']
    for profile in native_profiles:
        row = scalars(profile)
        cap = abs(row['onsite'])+6*sum((abs(c0)+abs(cj) for c0, cj in row['path'].values()), F(0))
        mass = F(1, 5) if row['load'] == 1 else F(2)
        gap = 2*mass*sl(gmin)-32*cap*(1+rho)*eta
        s1, s2, s3 = 8*cap*(1+rho), 16*cap*(1+rho), 32*cap*(1+rho)
        if gap <= 0:
            raise ArithmeticError('Actual charged scalar band gap failed')
        C += row['load']*(16*s2+64*s1*s1/gap)/L
        charged_third += row['load']*2*V*(128*s1**3/gap**2+32*s1*s2/gap+s3)/L
    schur_den = 3*hfast/5-F(2)**-2048*(amax+C)
    separate = hfast/2-F(2)**-2048*(C+4*C*C/amin)
    if schur_den <= 0 or separate <= hfast/4 or F(2)**2048*hfast <= amin:
        raise ArithmeticError('Actual fast/slow quadratic separation failed')
    schur = C*C/schur_den
    fast_variance = max(1/(2*sl(lapmin/5)), F(3, 10)/omega_min, (F(3, 5)+12/rmin)/(2*omega_min))
    radius2 = (D-45)*fast_variance*F(2)**-7168+45/(2*sl(amin))*F(2)**-6144
    cut_deficit = 4*radius2/eta**2
    if cut_deficit >= F(1, 100):
        raise ArithmeticError('Actual Phi222 Gaussian normalization failed')
    heavy_third = 2*V*(128*K1**3/heavy_gap**2+32*K1*K2/heavy_gap+K3)
    quartic_third = 48*(rho+1)/a
    relative_metric = metric1*eta/F(3, 5)
    relative_potential = eta*(8*(heavy_third+charged_third+quartic_third)/amin+256*(1+rho)**2/hfast+4096/rmin)
    relative = relative_metric+relative_potential
    if relative >= F(1, 100):
        raise ArithmeticError('Actual nonlinear field quadratic form control failed')
    fast_energy = (D-45)*F(2)**3072*su(3*rmax/25+F(63, 25))
    slow_energy = 45*F(2)**2048*su(max(C, amax))
    harmonic_excess = F(2)**1024*(D-45)*C*(fast_variance+(1+12/rmin)/omega_min)+45*schur/sl(amin)
    trial = fast_energy+slow_energy+harmonic_excess
    cut_derivative = 8*chi*metric*radius2/eta**4
    cut_excess = cut_deficit*trial/(1-cut_deficit)+2*su(trial*cut_derivative)+cut_derivative
    force = 100*D*(1+rho)**2*F(2)**3072/(rmin*sl(rmin))+8*su(F(V))*light*(1+rho)
    source_relative_energy = trial+cut_excess+relative*(trial+F(2)**2048*su(amin/2))+force*eta+U
    shift = U+16*force**2/(L*amin)
    normal_stiffness, fast_stiffness, kinetic_coercivity = L*amin/64, F(2)**10240*hfast/64, F(3, 20)*chi
    source_energy = source_relative_energy+shift
    barrier = normal_stiffness*eta**2/64
    spectral_cut = barrier/16
    slope = eta*su(L*amin/(chi*metric))/4096
    ramp_price = chi*metric*slope*slope/2
    denominator = barrier-spectral_cut-ramp_price
    if denominator <= 0 or spectral_cut <= 64*source_energy:
        raise ArithmeticError('Actual source energy does not pay spectral boundary split')
    weighted_norm = su(barrier/denominator)
    weighted_gradient = su((spectral_cut+ramp_price)*weighted_norm**2/kinetic_coercivity)+slope*weighted_norm
    Agmon = eta*eta*sl(L*amin/(chi*metric))/65536
    boundary_prefactor = chi*metric*8/eta*weighted_gradient+chi*(D**2*metric1*8/eta+D*metric*96/eta**2)*weighted_norm/2
    if Agmon <= max(binary_exponent(boundary_prefactor), binary_exponent(weighted_norm), binary_exponent(8*weighted_norm/eta))+8196:
        raise ArithmeticError('Uniform spectral Agmon boundary price failed')
    tail, boundary = F(2)**-8192, F(2)**-8192
    high_norm = su(source_energy/spectral_cut)
    field_distance, Lie_moment = su(source_energy/normal_stiffness), su(source_energy/fast_stiffness)
    gradient = su(source_energy/kinetic_coercivity)+8*tail/eta
    root0, roots = su(rational(enc['g0_upper'])), su(rational(enc['gs_upper']))
    sqrtV = su(F(V))
    matter_price = current_price = contact_price = F(0)
    contact_mode_factor = su(F(2, V))
    for profile in native_profiles:
        row = scalars(profile)
        load = row['load']
        matter_price += 4*load*abs(row['onsite'])*(2*sqrtV*field_distance+V*frame)
        for length, (c0, cj) in row['path'].items():
            cap, Fref = abs(c0)+abs(cj), abs(c0)*root0+abs(cj)*roots
            matter_price += 48*load*(2*cap*sqrtV*field_distance+V*frame*cap+length*Fref*sqrtV*Lie_moment)
            current_price += 16*load*(length*(2*cap*field_distance+cap*frame*sqrtV)+length*length*Fref*Lie_moment)
            contact_price += 16*load*length*length*(2*cap*contact_mode_factor*field_distance+cap*frame+length*Fref*contact_mode_factor*Lie_moment)
    D1base = 4*su(F(D))*2*V*K1/heavy_gap
    D2base = D*(64*(2*V)**2*K1*K1/heavy_gap**2+16*2*V*K2/heavy_gap)
    D1, D2 = 4*D1base, 4*D2base+52*D1base**2
    q = 66*su(F(5*V, 2))
    if rational(pure['new_projected_section_transfer']['projected_overlap_squared_lower']) < F(1, 4):
        raise ArithmeticError('Actual neutral-heavy projected line overlap failed')
    fibre = chi*(4*metric*gradient*(D1+q/rmin)+metric*D2+D*metric1*(D1+q)+metric*q*q/rmin)
    residual = fibre+2*matter_price+boundary
    cut_low_error = high_norm+tail+T*residual
    original_wave_error = 2*high_norm+2*tail+T*residual
    if original_wave_error >= F(1, 1000):
        raise ArithmeticError('Same evolving Phi-source transfer not useful')
    values = dict(actual222_orbit_min=rmin, actual222_orbit_max=rmax, actual222_slow_stiffness_lower=amin, actual222_slow_stiffness_upper=amax, actual222_metric_upper=metric, actual222_metric_derivative_upper=metric1, actual222_Haar_FP_scalar_upper=U, actual222_covariance_trace_upper=radius2, actual222_cut_deficit=cut_deficit, actual222_scalar_trial_relative_energy_upper=source_relative_energy, actual222_field_scalar_shift=shift, actual222_field_normal_coercivity=normal_stiffness, actual222_field_fast_Lie_coercivity=fast_stiffness, actual222_field_kinetic_coercivity=kinetic_coercivity, actual_original_field_source_shifted_energy_upper=source_energy, field_spectral_boundary_barrier=barrier, field_low_spectral_cut=spectral_cut, field_uniform_low_spectral_weighted_norm_upper=weighted_norm, field_uniform_low_spectral_weighted_gradient_upper=weighted_gradient, field_uniform_low_spectral_Agmon_exponent_lower=Agmon, field_original_source_high_spectral_norm_upper=high_norm, alltime_original_reference_field_normal_distance_L2_upper=field_distance, alltime_original_reference_field_link_Lie_L2_upper=Lie_moment, alltime_cut_low_reference_field_gradient_L2_upper=gradient, alltime_reference_to_C000_matter_operator_action_price=matter_price, alltime_reference_to_C000_current_operator_action_price=current_price, alltime_reference_to_C000_contact_operator_action_price=contact_price, alltime_cut_low_reference_angular_Gauss_and_Berry_residual=fibre, alltime_uniform_low_spectral_boundary_residual=boundary, alltime_cut_low_reference_generator_residual=residual, alltime_actual_full_source_to_cut_low_reference_norm_error=cut_low_error, alltime_original_Phi_source_reference_wave_error=original_wave_error, original_reference_vs_cut_low_reference_norm_error=high_norm+tail, actual_projected_heavy_D1_upper=D1, actual_projected_heavy_D2_upper=D2, actual_all_CAR_colour_charge_norm_upper=q)
    desc = dict(plan=bd['plan'], plan_revision=REVISION, scientific_execution_revision='03b34911221ba1898315373edaa7c4c3bcac718462e7fe51a842560eb6933f87', input_receipts=dict(bridge=dict(path=str(bridge_path), sha256=hashlib.sha256(raw).hexdigest())), source_abi=abi, N=bd['N'], V=V, M=bd['M'], a=str(a), UEphys='1', Lambda='2^8192', clock=str(T), reference='EXACTlocalDirichlet scalarPhi/SU4 Hamiltonian222 evolved on ORIGINAL normalized Gaussian/b2 Phisource; tensorHref000; scalaroffset -E_light_free000 exact; no replacementbyground orstationarity premise', comparison='Last-boundaryDuhamel oncut LOWSPECTRALreference; untouchedhighPhi-source contributionpaidbyunitarity; alltimeoriginalreference fieldmomentsfromcoercivityandconservedenergy')
    return dict(input_descriptor=desc, input_sha256=hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest(), **{key: positive_exact(value) for key, value in values.items()},
        reference_operator='Hphi isEXACTlocalDirichlet scalarCart/SU4 differentialoperator fromliteraloriginalchi/2 and3chi/10, jointmeasure/HaarFP, actualbosonpotential, actualneutralheavygroundeigenvalue andstaticall64negativechargedsea atTheta0/AP222; noquadratic/harmonic Hamiltonian substitution. ORIGINALPhiGaussian/b2 isinitialvectorphi, notground. Referencewave exp(+iE_light_free000t) jointisometriczeroextension[(exp(-iHphit)phi) OmegaHproj (exp(-iHref000t)chi000)]. Samechargedpreparation/reference000; actualPhi/heavycarrier222. OriginalfullH notDirichlet-replaced; DirichletisONLYcomparisonoperator, itsboundaryprice explicitlypaid.',
        coercive_energy_proof='FRESHactual222 gmin/TrG andrmin=minG/2,rmax=TrG/2. Actualall45normalstiffness000 +/-refinerfullnormalδ; C recalculatedallheavy+1/63charged clusterresolventstatic/Taylorbounds. Physicalfastlapmin4sin(pi/N)²andactualneutralcolourRmin; executeSchurdenominator andseparateYoungfastmass>=hfast/4 pluslambdaQuarterhfast>amin. Fullactualnormalchart nonlinearrelative<=1/100 fromthirdheavy2V, allchargedthird/ Lambda, quarticthird, alignment/curl andmetric M1eta. Sourcefast/slowexactSPECTRALcovariances, harmonic-crossandcutderivativeexpectation RECOMPUTED; hugeclassicalV222 cancelsBEFOREenergyarithmetic. Actualmeanstationarity includingfastzero-point yieldsfieldforcebound100D(1+rho)² Lambda^3/8/(rmin sqrt(rmin))+chargedLipschitz. Youngforce withscalarJac±U makesHshift=Hphi−EXACTVclass222+shift nonnegative andform>=kineticcoercivity||grad||²+normalstiffness||allYxi||²+faststiffness||fastYxi||², constantsLambdaamin/64,Lambda^5/4hfast/64,3chi/20 deliberately weakerthanseparatedmasses. ThisisLOCALDirichletformcoercivity provedbyTaylor/massbounds, NOTglobalvacuum/groundpostulate. OriginalPhisource expectationHshift<=source_energy; CONSERVEDforALLreferencePhi-timeevolution, includingHIGHspectralpart. Hence normal/Lie/gradientmoments payEVOLVINGPhi, notinitialsupportradius.',
        uniform_spectral_boundary_proof='NOgroundprojection. P=1_[0,Ecut](Hshift), Ecut=barrier/16; sourcehighnorm²<=source_energy/Ecut bypositivity/SpectralTheorem, retainedandpaidbyunitarity. RampW=e^f boundedonfiniteDirichletball, f=0r<=eta/8, |gradf|<=slope, f>=Agmon inallzetaderivative supporteta/4..eta/2. PotentialHshift>=barrier outsider>=eta/8; rampkin<=ramp_price. M_W=||WP|| finite sinceWbounded. Fornormalizedu, psi=Pu, weightedform>= (barrier−ramp)||Wpsi||²−barrier; RHSRe<Wpsi,WHshiftPu><=Ecut M_W||Wpsi|| becauseHshiftP boundedEcut andcommutesP. Takeoperatornormsup toobtainM_W²<=barrier/(barrier−ramp−Ecut). Weightedgradient bound fromsameform andnonnegativepotential, noeigenstate/spectralgap/stationarity assertion. ExecuteAgmon exceedinglog2 FULLD²boundaryprefactor and8196; tail andcommutator<=2^-8192 uniformlyALLlowstates, includingactualEVOLVEDlowPhigaussian. ThisdoesNOTconfineFULLphysicalevolvingPsi byinitialsupport.',
        last_boundary_Duhamel='Letunmodifiedphi_t=Uphi(t)phi_source. Lowactualphi_t=Pphi_t, smoothflatcutzeta equal1r<=eta/4,supporteta/2. Source-high norm<=highnorm. Initialsource−zetalow<=highnorm+tail; finalzetalow−originalfullreference<=highnorm+tail. AngularGauss/BerryPhi-actionbound usesENERGYgradient ofactualevolvinglowstate, actualprojectedheavyresolvent D1/D2 andallCARCasimir q<=66sqrt(5V/2); noTheta derivativesinthisblock. Chargedoperator+scalarWeyl residual2matter_price derivesalltimePhi-distance/Lie moments andliteralone/twolink/onsitecoefficients, allTheta/CAR. Boundarystrongcommutatorboundeduniformlylowband. Variationofconstants onzetalowHdomain yieldscutlow_error; addfinalhigh+tail forORIGINALsourcePhi-unitaryreference error. ActualphysicalfullPsi mayleavechart; fullunitarity, notsupportassumption, controlsit.',
        evolving_matter_current_contact='NORMALPhiop<=2||Yx||, actualpositiveframeδ included. Uniformchargedmatteraction: onsite4|ons|[2sqrtV||Y||+Vδ]; forward/reversefourspin CAR<=16hop, all3axes =>48sum[2csqrtV||Y||+Vcδ+L F000sqrtV||xi||]. Currentcosqx y/z: fconstantalongaxialpath, CSsum|f_x|rho_x<=||Y||; sumedges<=L||xi||. Thus16sum L[2c||Y||+cδsqrtV]+L²F000||xi||. Contactsumf²=1 andsumf⁴<=2/V (includesq=N/4) gives16sum L²[2c sqrt(2/V)||Y||+cδ+L F000sqrt(2/V)||xi||]. ALLreferenceevolvedPhi_source boundsfromconservedenergy/coercivity, notinitialeta bound; ALLmacro nonzero/nonNyquistintegerq, notnativeN2 promotion. RawcanonicalTheta; physicalAcurrent=aJTheta, contactA=a²contactTheta.',
        review_boundary='NEW actualsameevolvedPhi-source finiteclock construction forparentmathematicalreview ofscalarcoerciveTaylor/sourceenergy/spectralboundary/globalcore/Born premises. ActualCLI computesstrict source-specific prices, notnumericalfield/photonpropagatorvalues. No ground/vacuum/stationarity/commoncone/pole/Compton/statP-Q promotion. Groundpacketf9 construction separate andnotconsumerinput; oldeta-onlyunadmittedepochsunchanged. Fundamentalquantumlawsandparametersdeclaredinputs.',
        source_at_import={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes_utf8=data.decode()) for name, data in SOURCE_BYTES.items()}, source_unchanged_since_import={name: (ROOT/name).read_bytes()==data for name, data in SOURCE_BYTES.items()}, runtime=dict(executable=sys.executable, python=sys.version), elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.bridge)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('input_sha256', 'alltime_original_Phi_source_reference_wave_error', 'alltime_reference_to_C000_current_operator_action_price', 'elapsed_seconds')}, indent=2))

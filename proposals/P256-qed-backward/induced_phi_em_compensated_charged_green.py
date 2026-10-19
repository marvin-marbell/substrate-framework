"""Original unfiltered Phi source: compensated active+1/Ref3 Green comparison.

The native CAR/Gauss calculations are actual; macro output is an explicit
finite-clock comparison bound, not an evaluated Green function or charged pole.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from induced_phi_em_full_current_inserted_response import SOURCE_BYTES as RESPONSE_BYTES
from induced_phi_em_ygraph_clock_transfer import checked, exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_ygraph_evolving_field_clock import su, sl
from induced_phi_em_macro_preparation import build_native, scalars
from induced_reference_electric_graph import graph_state, graph_direction, graph_tangents, SOURCE_BYTES as GRAPH_BYTES
from induced_phi_ir_c_full_action import T, SOURCE_BYTES as ACTION_BYTES
from induced_phi_ir_well_transfer import rational

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**RESPONSE_BYTES, **GRAPH_BYTES, **ACTION_BYTES, Path(__file__).name: Path(__file__).read_bytes()}
REVISION = '6ebc34a912158254456c535bd10259fca04201b14f446a778bba68d5ef63ecf2'


def marginal_density(P: np.ndarray) -> np.ndarray:
    """Actual mixed four-mode Slater marginal, not a fictitious local projector."""
    values, vectors = np.linalg.eigh(P)
    if min(values) < -1e-12 or max(values) > 1+1e-12:
        raise ArithmeticError('Actual source marginal violates the CAR interval')
    rho = np.zeros((16, 16), complex)
    for occupied in itertools.product((0, 1), repeat=4):
        cols = np.flatnonzero(occupied)
        weight = np.prod([values[j] if occupied[j] else 1-values[j] for j in range(4)])
        ray = np.zeros(16, complex)
        for rows in itertools.combinations(range(4), len(cols)):
            mask = sum(1 << int(j) for j in rows)
            ray[mask] = np.linalg.det(vectors[np.ix_(rows, cols)]) if len(cols) else 1
        rho += weight*np.outer(ray, ray.conj())
    return rho


def local_car_matrix(h: np.ndarray) -> np.ndarray:
    """Same annihilate/create parity as sector_action, assemble each entry once."""
    dimension = 1 << len(h)
    out = np.zeros((dimension, dimension), complex)
    entries = [np.flatnonzero(h[:, source] != 0) for source in range(len(h))]
    for mask in range(dimension):
        for source in range(len(h)):
            if not mask & (1 << source):
                continue
            removed = mask ^ (1 << source)
            for target_value in entries[source]:
                target = int(target_value)
                if target != source and mask & (1 << target):
                    continue
                output = removed | (1 << target)
                swaps = (mask & ((1 << source)-1)).bit_count()
                swaps += (removed & ((1 << target)-1)).bit_count()
                out[output, mask] += (-1 if swaps & 1 else 1)*h[target, source]
    return out


def actual_native(mean: dict[str, Any]) -> dict[str, Any]:
    native = build_native(mean, 2)
    V, M = native['V'], native['M']
    ref = int(np.flatnonzero(np.all(native['sites'] == (0, 0, 0), axis=1))[0])
    x = int(np.flatnonzero(np.all(native['sites'] == (1, 0, 0), axis=1))[0])
    theta_coordinates = .13*np.sin(.7*np.arange(M))
    theta = native['Q']@theta_coordinates
    profile = native['Pprofiles'][0]
    if profile['load'] != 1:
        raise ArithmeticError('The tagged active profile must be a single flavour')
    state = graph_state(native, profile, theta)
    P, Q = state['P'], np.eye(4*V)-state['P']
    site = slice(4*x, 4*x+4)
    local_P = P[site, site]
    rho4 = marginal_density(local_P)
    rho = np.zeros((256, 256), complex)
    rho[240:256, 240:256] = rho4
    h = np.zeros((8, 8), complex)
    h[:4, 4:] = np.eye(4)/2
    E = local_car_matrix(h)
    created = E@rho@E.conj().T
    probability = float(np.trace(created).real)
    ref_number = np.array([(mask >> 4).bit_count() for mask in range(256)])
    ref3_probability = float(np.trace(created[ref_number == 3][:, ref_number == 3]).real/probability)
    ref4_columns = np.arange(240, 256)
    expected_gram = np.diag([(4-(mask & 15).bit_count())/4 for mask in ref4_columns])
    gram_identity = float(np.linalg.norm((E.conj().T@E)[np.ix_(ref4_columns, ref4_columns)]-expected_gram))
    colour_errors = []
    ref_casimir = np.zeros((16, 16), complex)
    for generator in T:
        joint = np.zeros((8, 8), complex)
        joint[:4, :4] = generator
        joint[4:, 4:] = generator
        qjoint = local_car_matrix(joint)
        colour_errors.append(float(np.linalg.norm(qjoint@E-E@qjoint)))
        qref = local_car_matrix(generator)
        ref_casimir += qref@qref
    holes = np.array([mask for mask in range(16) if mask.bit_count() == 3])
    casimir_error = float(np.linalg.norm(ref_casimir[np.ix_(holes, holes)]-15/8*np.eye(4)))
    Rx = np.zeros_like(P)
    Rx[site, site] = np.eye(4)
    added = Q@Rx@Q/2
    blocks = P.reshape(V, 4, V, 4).transpose(0, 2, 1, 3)
    added_blocks = added.reshape(V, 4, V, 4).transpose(0, 2, 1, 3)
    d = np.trace(added_blocks[np.arange(V), np.arange(V)], axis1=1, axis2=2).real
    r = np.eye(V)[ref]
    exchange = np.einsum('jiab,ijba->ij', blocks, added_blocks).real
    delta_second = np.diag(d)-2*exchange-np.outer(d, r)-np.outer(r, d)+np.outer(r, r)
    g = native['UE']/native['a']
    G = native['Lplus']
    coulomb_delta = float(g*np.sum(G*delta_second))
    X = float(np.sum(G*exchange))
    G00 = float(G[ref, ref])
    net_second = float(abs(np.sum(delta_second)))
    matter_addition = float(np.trace(state['physical_h']@added).real)
    empty_h = state['empty'].conj().T@state['physical_h']@state['empty']
    empty_h_min = float(np.linalg.eigvalsh(empty_h)[0])
    Gamma5 = np.kron(np.eye(V), state['B']@np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]])@state['B'].conj().T)
    chiral_error = float(np.linalg.norm(Gamma5@P@Gamma5-Q))
    kinetic_changes = []
    for direction in native['Q'].T:
        first = graph_direction(native, state, theta, direction)
        jets = graph_tangents(state, first, np.zeros_like(first))
        Bph = Q@jets['dP']@P
        kinetic_changes.append(float(g*np.trace(Rx@(Bph.conj().T@Bph-Bph@Bph.conj().T)).real/2))
    charge = np.eye(V)[x]-np.eye(V)[ref]
    ell = native['D'].T@G@charge
    divergence_error = float(np.linalg.norm(native['D']@ell-charge))
    coexact_error = float(np.linalg.norm(native['Q'].T@ell))
    long_energy = float(ell@ell)
    coulomb_identity_error = abs(long_energy-float(charge@G@charge))
    omitted_ref = np.eye(V)[x]
    uncompensated_error = float(np.linalg.norm(native['D']@native['D'].T@G@omitted_ref-omitted_ref))
    if abs(probability-.5)>1e-10 or abs(ref3_probability-1)>1e-10 or max(colour_errors)>1e-10 or casimir_error>1e-10 or gram_identity>1e-10 or abs(np.trace(added).real-1)>1e-10 or net_second>1e-10 or empty_h_min<=0 or max(abs(v) for v in kinetic_changes)>1e-10 or chiral_error>1e-10 or divergence_error>1e-10 or coexact_error>1e-10 or coulomb_identity_error>1e-10 or uncompensated_error<.1 or not (-2*g*G00-1e-10 <= coulomb_delta <= 4*g*G00+1e-10) or not (-1e-10 <= X <= G00+1e-10):
        raise ArithmeticError('Actual compensated CAR/Gauss/source-energy control failed')
    return dict(N=2, V=V, all_local_active_plus_Ref_CAR_dimension=256, actual_theta_coordinates_hex=[float(v).hex() for v in theta_coordinates], source_active_local_mean_number=float(np.trace(local_P).real), source_active_marginal_projector_defect=float(np.linalg.norm(local_P@local_P-local_P)), exact_creation_probability=probability, actual_Ref3_probability=ref3_probability, exact_Ref4_CREATOR_Gram_identity_error=gram_identity, global_colour_CREATOR_commutator_max=max(colour_errors), actual_Ref3_Casimir=15/8, actual_Ref3_Casimir_error=casimir_error, added_active_density_trace=float(np.trace(added).real), compensated_charge_mean_sum=float(np.sum(d-r)), compensated_charge_second_moment_sum_error=net_second, actual_source_CAR_matter_addition=matter_addition, actual_source_empty_CAR_compressed_H_min=empty_h_min, actual_source_Coulomb_addition=coulomb_delta, actual_source_Coulomb_exchange_X=X, actual_Lplus_diagonal=G00, source_Gamma5_P_Gamma5_Q_error=chiral_error, source_per_coexact_direction_EM_kinetic_additions=kinetic_changes, source_TOTAL_transverse_EM_kinetic_addition=float(sum(kinetic_changes)), real_longitudinal_dressing_divergence_error=divergence_error, real_longitudinal_dressing_coexact_error=coexact_error, actual_dressing_real_momenta=ell.tolist(), dressing_Pi_energy=long_energy, dressing_Coulomb_identity_error=coulomb_identity_error, WRONG_omitted_Ref_charge_Gauss_unsolvable_residual=uncompensated_error, scope='Actual N2 matrix source/Phi000 control and exact reduced Gauss; no full quantum Phi/EM propagation or macro Green values')


def run(response_path: Path, action_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    raw = response_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'f57776ebcad02fe5818d42ed906cd3b625f50d45559d34ad9c2c253a1dff4d35':
        raise ArithmeticError('Requires actual5c6b four-current source epoch')
    response = json.loads(raw)
    _, inserted = checked(response['input_descriptor']['input_receipts']['inserted'])
    _, force = checked(inserted['input_descriptor']['input_receipts']['force'])
    _, current = checked(force['input_descriptor']['input_receipts']['current'])
    _, retarded = checked(current['input_descriptor']['input_receipts']['retarded'])
    _, field = checked(retarded['input_descriptor']['input_receipts']['clock'])
    _, bridge = checked(field['input_descriptor']['input_receipts']['bridge'])
    _, mean = checked(bridge['input_descriptor']['input_receipts']['mean'])
    action_raw = action_path.read_bytes()
    if hashlib.sha256(action_raw).hexdigest() != '4262856da70fead1ea4d5ee7d0e8ce0dd260b506812762d2a29f70975770afee':
        raise ArithmeticError('Original bare Ref charge/background contract changed')
    action = json.loads(action_raw)
    em = action['real_EM_Coulomb']
    if (em['Ref_bare_charge'], em['Ref_occupied'], em['Ref_background'], em['background']) != (1, 4, -4, '128+4 delta_x0'):
        raise ArithmeticError('No imported Alpha charge or reference-background substitution')
    for filename, data in SOURCE_BYTES.items():
        if filename in action['source_at_import'] and hashlib.sha256(data).hexdigest() != action['source_at_import'][filename]['sha256']:
            raise ArithmeticError('Original action source epoch changed: '+filename)
    fd = force['input_descriptor']
    V, N, Tclock, eta, chi, s = int(fd['V']), int(fd['N']), F(fd['clock']), F(2)**-2048, F(2)**-4096, F(fd['variance'])
    if (N, V, fd['a'], fd['UEphys'], fd['Lambda'], s) != (262144, 2**54, '1/64', '1', '2^8192', F(2)):
        raise ArithmeticError('The unchanged physical box and source are required')
    neutral_filled = 128*V+4
    charged_filled = (2*V+1)+63*2*V+3
    total_with_heavy = charged_filled+2*V
    if charged_filled-neutral_filled != 0 or total_with_heavy % 4:
        raise ArithmeticError('Charged global EM/SU4 centre constraint failed')
    qactive = su(F(5*V, 2)-F(5, 8))
    qauxheavy = 64*su(F(5*V, 2))
    qref = su(F(15, 8))
    qC = qactive+qauxheavy+qref
    qold = exact(field['actual_all_CAR_colour_charge_norm_upper'])
    if qC > qold:
        raise ArithmeticError('Explicit charged Casimir exceeds actual source geometry envelope')
    metric = exact(field['actual222_metric_upper'])
    metric1 = exact(field['actual222_metric_derivative_upper'])
    rmin = exact(field['actual222_orbit_min'])
    gradient = exact(field['alltime_cut_low_reference_field_gradient_L2_upper'])
    D1 = exact(field['actual_projected_heavy_D1_upper'])
    D2 = exact(field['actual_projected_heavy_D2_upper'])
    D = 90*V
    fibreC = chi*(4*metric*gradient*(D1+qC/rmin)+metric*D2+D*metric1*(D1+qC)+metric*qC*qC/rmin)
    matter = exact(field['alltime_reference_to_C000_matter_operator_action_price'])
    boundary = exact(field['alltime_uniform_low_spectral_boundary_residual'])
    high = exact(field['field_original_source_high_spectral_norm_upper'])
    epsilonC = 2*high+2*F(2)**-8192+Tclock*(fibreC+2*matter+boundary)
    epsilonN = exact(field['alltime_original_Phi_source_reference_wave_error'])
    n = su(F(1, 2))
    path_length = N//4
    initial_insertion = 2*su(F(path_length))*eta
    moving_insertion = 2*su(F(path_length))*exact(field['alltime_original_reference_field_link_Lie_L2_upper'])
    charged_packet_error = n*epsilonC+initial_insertion
    green_error = n*(2*epsilonN+moving_insertion)+2*charged_packet_error
    G00cap = F(3, 4)+F(1, 4*N)
    g = F(fd['UEphys'])/F(fd['a'])
    _, energy = checked(bridge['input_descriptor']['input_receipts']['energy'])
    _, initial = checked(energy['input_descriptor']['input_receipts']['initial'])
    _, domain = checked(initial['input_descriptor']['input_receipts']['domain'])
    _, pure = checked(domain['input_descriptor']['input_receipts']['purePhi'])
    _, well = checked(pure['input_descriptor']['input_receipts']['well'])
    coulombC_cap = g*V*129**2/rational(well['prices']['smallest_laplacian_lower'])
    native = build_native(mean, 2)
    row = scalars(native['Pprofiles'][0])
    c01, cj1 = row['path'][1]
    c02, cj2 = row['path'][2]
    if c01 != -4*c02 or cj2 != 0 or row['load'] != 1:
        raise ArithmeticError('Actual improved Wilson coefficient identity failed')
    mass_coefficient = row['onsite']+6*(c01+c02)
    enc = mean['centre_enclosure']
    root0lo, root0hi = sl(rational(enc['g0_lower'])), su(rational(enc['g0_upper']))
    rootshi = su(rational(enc['gs_upper']))
    Mmin = mass_coefficient*root0lo
    Mmax = (mass_coefficient+48*c02)*root0hi
    denominator_min = 2*row['onsite']*root0lo+g
    Ycap = 6*abs(cj1)*rootshi/denominator_min
    source_Q_h_Q_lower = Mmin/(1+Ycap*Ycap)
    if mass_coefficient <= 0 or 2*denominator_min <= Mmax or green_error >= F(1, 1000):
        raise ArithmeticError('Actual charged preparation/comparison bound failed')
    initial_matter_price = exact(bridge['corrected_uniform_all_Theta_all_CAR_charged_Phi_freezing_action'])
    full_initial_CAR_addition_lower = source_Q_h_Q_lower-initial_matter_price
    if full_initial_CAR_addition_lower <= 0:
        raise ArithmeticError('Actual Phi matter price swamps positive source CAR addition')
    desc = dict(plan=fd['plan'], plan_revision=REVISION, scientific_execution_revision=REVISION, input_receipts=dict(response=dict(path=str(response_path), sha256=hashlib.sha256(raw).hexdigest()), original_action=dict(path=str(action_path), sha256=hashlib.sha256(action_raw).hexdigest())), source_abi=fd['source_abi'], N=N, V=V, M=int(fd['M']), a=fd['a'], UEphys=fd['UEphys'], Lambda=fd['Lambda'], variance=str(s), clock=str(Tclock), neutral_original_preparation='UNCHANGED Phi222/projectedheavy/chargedHermitianYgraph000/AP222 Gaussian; no filter/groundreplacement', charged_insertion='E_x=(1/2) sum_ab cdag_active,x,a W_SU4(x<-Ref0)_ab rRef,b; physical unreduced exp(+i ell_long.A), ell_long=Dunit^T Lplus(delta_x-delta_Ref0), Gauss=Dunit Pi-rho. Active+1, Ref3; Ref bare1/background-4 unchanged; coexact Q^T ell=0 EXACT.', source_position=[N//4,0,0], receiver_position=[0,N//4,0], Ref_position=[0,0,0], photon_dressing='REAL longitudinal only, not compact integer electric string', common_energy_convention='H0_N/C=Hphi222 + SAME original Href000 restricted to N/C - SAME neutral Elight_free000; no charged-sector own energy subtraction', native_control=dict(N=2, active_source_position=[1,0,0], Ref_position=[0,0,0], theta_coordinates='0.13 sin(0.7 arange(M))'))
    values = dict(charged_sector_active_global_Casimir_norm_upper=qactive, charged_sector_aux63_and_neutralheavy_global_Casimir_norm_upper=qauxheavy, actual_Ref3_colour_Casimir_norm_upper=qref, charged_sector_total_colour_q_upper=qC, charged_sector_recomputed_alltime_Born_fibre_residual=fibreC, charged_sector_recomputed_alltime_original_Phi_reference_wave_error=epsilonC, charged_original_INITIAL_packet_reference_wave_error=charged_packet_error, original_alltime_moving_CREATOR_on_neutral_reference_error=moving_insertion, original_FULL_vs_reference_initial_CREATOR_error=initial_insertion, alltime_NONSTATIONARY_original_FULL_charged_Green_comparison_error=green_error, same_exact_CREATOR_Gram_relative_Green_error=2*green_error, actual_three_dimensional_Lplus_diagonal_upper=G00cap, charged_sector_global_Coulomb_operator_upper=coulombC_cap, source_Coulomb_addition_lower_magnitude=2*g*G00cap, source_Coulomb_addition_upper=4*g*G00cap, actual_source_empty_Q_h_Q_positive_lower=source_Q_h_Q_lower, FULL_original_Phi_INITIAL_CAR_matter_addition_positive_lower=full_initial_CAR_addition_lower)
    return dict(input_descriptor=desc, input_sha256=hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest(), **{k:positive_exact(v) for k,v in values.items()}, actual_creation_probability_exact='1/2', actual_source_CREATOR_transverse_EM_kinetic_addition_exact='0', charged_sector_counts=dict(active=2*V+1, auxiliary_each=2*V, auxiliary_count=63, neutral_heavy=2*V, Ref=3, fixed_background=128*V+4, total_EM_net_charge=charged_filled-neutral_filled, total_CAR=total_with_heavy, global_SU4_centre_mod4=total_with_heavy%4),
        actual_native_CAR_Refhole_Gauss_energy=actual_native(mean),
        source_Gram_and_electric_proof='SPECIFIC Hermitian graph: P++=I-R,P--=R,P+-=-iYR; R+YRY=I gives TrP_xx=2 EACH Theta. Ref4 filled quartet implies E^dag E restricted Ref4=(4-n_active,x)/4, norm²1/2 EACH Phi/Theta by unitary full-colour Wilson sum. Do not infer from generic negative Slater or local mean neutrality. E_U independent Theta and REALlongitudinal phase becomes exactly1 on coexact slice, so no extra transverse flux. Source Gamma5 P Gamma5=Q, Berry0 and derivative B_e=Q DP_e P obeyGamma5 B_e Gamma5=-B_e^dag. Gamma5 onsite commutes Rx; TrRx B_eB_e^dag=TrRx B_e^dagB_e. Normalized creation changes transverse momentum kinetic by(g/2)sum_e TrRx(B_e^dagB_e-B_eB_e^dag)=0. Gaussian derivative part and real cross term vanish by exact local n=2 identity. This is INITIAL source form only, not EM interacting energy/pole or statement at later clocks.',
        charged_colour_fibre_proof='Original Cart/ sqrt8 generators Trt²1/2. Fourmode n-sector Casimir=5n(4-n)/8=5/2-(5/8)(n-2)². Active total2V+1: sum(n_x-2)=1 integers => sumCasimir<=5V/2-5/8; EACH63aux+heavy halfnumber <=5V/2, Ref3 actualwedge3 Casimir15/8. Hilbert-space Minkowski yields qC=sqrt(5V/2-5/8)+64sqrt(5V/2)+sqrt(15/8). Recomputed full Born expression using actual metric/D1/D2/lowPhi gradient/rmin; not neutral66 by label. FilledchargedplusRef=(2V+1)+63(2V)+3=128V+4 matches originalBACKGROUND128+4deltaRef, heavyEM0; TOTALwithheavy130V+4=0mod4. No fixedlocal2occupation; centre onlyglobalZ4/4. Traceless4V onebody implies ALLnumbersector ||dGamma(h)||<=Tr|h|/2<=2V||h||, so actualoldALLCARPhi action plus static-neutral-Weyl scalar price remainsvalidchargedsector. Coulomb rho_ref=nRef-4=-1, max|rho_x|129; universalchargedcapgV129²/lapmin explicitlycomputed.',
        same_two_sector_green_proof='Gxy(t)=<U_N(t)Psi,E_x^dag U_C(t)E_yPsi>=<E_x U_N(t)Psi,U_C(t)E_yPsi>, NONstationaryneutral source not <E_xPsi,U_C(t)E_yPsi>. Neutral/computedchargedsector actualcommonH0 comparators use SAMEoriginalevolvingPhi + SAME neutral Elight_free000 offset. Eop<=2; fullinitialE_y-E0_y<=2sqrtL eta; movingE_x onneutralORIGINALreference<=2sqrtL actualconservedLieL2. Chargedpacketerror<=sqrt(1/2)epsilonC+initialprice. Greenerror<=sqrt(1/2)(2epsilonN+movingprice)+2chargedpacketerror. Reference ORIGINALPhi unitary cancels ONLYafter bothsectordifferences/insertionprice; samecommonphase cancels, charged physicalenergy NOTsubtracted. No new projector/ground/admissiongate.',
        charged_Coulomb_source_proof='After tracing coherentRef3 quartet, added active source densityDeltaP=Q Rx Q/2, TrDeltaP=1; Ref=-delta0. Delta secondmoment=delta_ij d_i-2ReTr(P_ji DeltaP_ij)-d_i r_j-r_i d_j+r_i r_j: added-d_i d_j covariance cancels addedmean², retainingseaexchange andoriginalstaticRef/background. Allauxmeans0 pointwise. Coulomb delta=g[2G00-2d.G_ref-2X], X=TrDeltaP(G Schur P)>=0 and<=G00 by CP-Schur normbound(P<=I); |d.G_ref|<=G00 PSD. Hence[-2gG00,4gG00]. Finite3D G00<=3/4+1/(4N): lambda(k)>=16|k_reduced|²/N², supnormshellcount<=24m²+2, sum_m m^-2<=2, m<=N/2. Thisconstantbound hasNO V/toroidalstringdivergence. Real longitudinal solution reproduces fulloriginalrhoLplusrho, notCoulombself-subtraction.',
        positive_CAR_source_addition_proof='ACTUAL binary64scalar Wilson pathidentityc01=-4c02, mass_eff=onsite+6(c01+c02)>0, not roundeddeclaredm. WithunitaryPeierlsTj/AP222, Cj=(Tj+Tjdag)/2 contraction, M=sqrtg0[mass_eff+4c02 sum_j(1-Cj)^2], so Mmin=mass_effsqrtg0_lower, Mmax=(mass_eff+48c02)sqrtg0_upper. Y=K/d, d=2M_onsite+g, Kcap=6|cj1|sqrtgs_upper. EmptyW=[sqrtR;-iYsqrtR], Wdag h W=sqrtR(M-YMY+2dY²)sqrtR >=Mmin R +(2d-Mmax)sqrtR Y²sqrtR >=Mmin/(1+Ycap²). Allcolourcreationmix DeltaP liesinthissourceQ, hence sourceCARaddition>=that. Wilsonunitarity fullcolourtrace makes addedDeltaP INDEPENDENT actualPhi/SU4Wilson at eachfieldpoint, so fullINITIALchargedmatteraddition differsby <=actualb367ALLCARPhi-freezingprice; hugePhi staticenergy isNOTassignedtoEM. ThispositiveMATTERaddition NOTfullHchargegap/mass: Coulombcostexplicit above, compact/Phi kineticaddition andinteracting spectral/resolvent/pole remainphysicalwork.',
        limitation='NEW same-law compensatedchargedpacket Green COMPARISON construction; parentmathreviewrequired. Native256CAR uses actualfourmodeMIXEDmarginal ofN2source, notlocalprojector/wholeCARstate; exactactualsourceQ/Gauss/coulombcontrols. No numericmacroGreenvalues, fullchargedPhi kineticenergygap, stationaryground/pole/commoncone/Compton/statistical-Pauli-code closure. OriginalfixedRef4 C-intertwining remainsrestrictedRef4; no new Refhole PH theorem claimed. Fundamentalquantumlaws/parametersdeclaredinputs. Alpha C/Rootkappa400/Source2a lawsnotimported.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(), bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()}, source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()}, runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__), elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--original-action', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.response, args.original_action)
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Source changed during the actual calculation')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_sha256', 'alltime_NONSTATIONARY_original_FULL_charged_Green_comparison_error', 'actual_native_CAR_Refhole_Gauss_energy', 'elapsed_seconds')}, indent=2))

"""Same original Phi/EM law: finite positive compensated source energy.

An initial-packet energy difference, not a charged-sector spectral gap. The
neutral source, fields, offsets, real Gauss background and receiver stay frozen.
"""
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
from induced_phi_em_compensated_charged_green import SOURCE_BYTES as GREEN_BYTES, marginal_density, local_car_matrix, T
from induced_phi_em_ygraph_clock_transfer import checked, exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_ygraph_evolving_field_clock import su, sl
from induced_phi_em_macro_preparation import build_native, scalars
from induced_reference_electric_graph import graph_state
from induced_phi_ir_well_transfer import rational

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**GREEN_BYTES, Path(__file__).name: Path(__file__).read_bytes()}
REVISION = '6ebc34a912158254456c535bd10259fca04201b14f446a778bba68d5ef63ecf2'


def signed_exact(value: F) -> dict[str, str]:
    return dict(numerator=str(value.numerator), denominator=str(value.denominator))


def actual_source_energy_control(mean: dict[str, Any]) -> dict[str, Any]:
    native = build_native(mean, 2)
    V, M = native['V'], native['M']
    sites = native['sites']
    x = int(np.flatnonzero(np.all(sites == (1, 1, 0), axis=1))[0])
    ref = int(np.flatnonzero(np.all(sites == (0, 0, 0), axis=1))[0])
    coordinates = .13*np.sin(.7*np.arange(M))
    state = graph_state(native, native['Pprofiles'][0], native['Q']@coordinates)
    P, Q = state['P'], np.eye(4*V)-state['P']
    site = slice(4*x, 4*x+4)
    Rx = np.zeros_like(P)
    Rx[site, site] = np.eye(4)
    added = Q@Rx@Q/2
    pblocks = P.reshape(V, 4, V, 4).transpose(0, 2, 1, 3)
    ablocks = added.reshape(V, 4, V, 4).transpose(0, 2, 1, 3)
    d = np.trace(ablocks[np.arange(V), np.arange(V)], axis1=1, axis2=2).real
    r = np.eye(V)[ref]
    exchange = np.einsum('jiab,ijba->ij', pblocks, ablocks).real
    delta_second = np.diag(d)-2*exchange-np.outer(d, r)-np.outer(r, d)+np.outer(r, r)
    G, g = native['Lplus'], native['UE']/native['a']
    direct_delta = float(g*np.sum(G*delta_second))
    A = np.array([np.trace(pblocks[x, j]@pblocks[j, x]).real for j in range(V)])
    dipole = G[:, x]-G[:, ref]
    deltaG = float(dipole[x])
    alternative_delta = float(g*(dipole@A))
    dipole_harmonic = native['D']@native['D'].T@dipole-(np.eye(V)[x]-np.eye(V)[ref])
    incident_x = np.flatnonzero(native['D'][x] != 0)
    incident_ref = np.flatnonzero(native['D'][ref] != 0)
    effective_R = 2*deltaG
    rho4 = marginal_density(P[site, site])
    h = np.zeros((8, 8), complex)
    h[:4, 4:] = np.eye(4)/2
    Ecolumns = local_car_matrix(h)[:, 240:256]
    probability = float(np.trace(Ecolumns.conj().T@Ecolumns@rho4).real)
    colour_derivatives, real_crosses = [], []
    for t in T:
        h[:4, 4:] = 1j*t/2
        derivative = local_car_matrix(h)[:, 240:256]
        colour_derivatives.append(float(np.trace(derivative.conj().T@derivative@rho4).real/probability))
        real_crosses.append(float(np.trace(Ecolumns.conj().T@derivative@rho4).real/probability))
    kinetic_carrier = float(sum(colour_derivatives))
    if abs(direct_delta-alternative_delta)>1e-10 or abs(A.sum()-2)>1e-10 or A.min()<-1e-12 or np.linalg.norm(dipole_harmonic)>1e-10 or dipole.min()<-deltaG-1e-10 or dipole.max()>deltaG+1e-10 or len(incident_x)!=6 or len(incident_ref)!=6 or np.intersect1d(incident_x, incident_ref).size or effective_R<1/3-1e-10 or direct_delta<2*g*deltaG*(A[x]-1)-1e-10 or A[x]<=1 or abs(kinetic_carrier-15/8)>1e-10 or max(abs(v) for v in real_crosses)>1e-10:
        raise ArithmeticError('Actual positive source Coulomb/common-scalar kinetic control failed')
    return dict(N=2, V=V, source_position=[1,1,0], Ref_position=[0,0,0], actual_theta_coordinates_hex=[float(v).hex() for v in coordinates], whole_source_P_dimension=4*V, local_active_plus_Ref_CAR_dimension=256, actual_CREATOR_probability=probability, direct_unchanged_Coulomb_source_addition=direct_delta, exact_chiral_source_alternative_Coulomb_addition=alternative_delta, alternative_identity_error=abs(direct_delta-alternative_delta), actual_source_A_row=A.tolist(), actual_A_row_sum=float(A.sum()), actual_source_A_x=float(A[x]), actual_Lplus_dipole=dipole.tolist(), actual_dipole_DeltaG=deltaG, actual_dipole_harmonic_residual=float(np.linalg.norm(dipole_harmonic)), distinct_incident_edge_counts=[len(incident_x),len(incident_ref)], incident_overlap_count=int(np.intersect1d(incident_x,incident_ref).size), actual_source_Ref_effective_resistance=effective_R, positive_source_Coulomb_lower_from_actual_A=2*g*deltaG*(float(A[x])-1), actual_compact_colour_CREATOR_derivative_norm_sum=kinetic_carrier, actual_compact_colour_CREATOR_real_scalar_cross_max=max(abs(v) for v in real_crosses), colour_derivatives_per_generator=colour_derivatives, scope='Actual source density/realGauss on N2 plus pointwise compact-colour derivative using EXACTglobal-source localMIXEDmarginal. Not a flattened completePhi Hamiltonian or native full source-energy propagation.')


def run(green_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    raw = green_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '5cb257aafbc8af2949329115a9aa147d23a6783b8a59dcfc4340ae9839a853d5':
        raise ArithmeticError('Actual c42 charged source/Green epoch required')
    green = json.loads(raw)
    if green['source_at_import']['induced_phi_em_compensated_charged_green.py']['sha256'] != hashlib.sha256(SOURCE_BYTES['induced_phi_em_compensated_charged_green.py']).hexdigest():
        raise ArithmeticError('Charged source code changed')
    gd = green['input_descriptor']
    if (gd['N'],gd['V'],gd['a'],gd['UEphys'],gd['Lambda'],gd['variance'],gd['clock']) != (262144,2**54,'1/64','1','2^8192','2','4096') or green['actual_creation_probability_exact'] != '1/2' or green['charged_sector_counts']['total_EM_net_charge'] != 0:
        raise ArithmeticError('No law/preparation/background/charge substitution')
    _, response = checked(gd['input_receipts']['response'])
    _, inserted = checked(response['input_descriptor']['input_receipts']['inserted'])
    _, force = checked(inserted['input_descriptor']['input_receipts']['force'])
    _, current = checked(force['input_descriptor']['input_receipts']['current'])
    _, retarded = checked(current['input_descriptor']['input_receipts']['retarded'])
    _, field = checked(retarded['input_descriptor']['input_receipts']['clock'])
    _, bridge = checked(field['input_descriptor']['input_receipts']['bridge'])
    _, mean = checked(bridge['input_descriptor']['input_receipts']['mean'])
    V, path_length = gd['V'], gd['N']//4
    chi, g = F(2)**-4096, F(gd['UEphys'])/F(gd['a'])
    metric = exact(field['actual222_metric_upper'])
    rmin = exact(field['actual222_orbit_min'])
    D1 = exact(field['actual_projected_heavy_D1_upper'])
    qN = 65*su(F(5*V,2))
    qC = exact(green['charged_sector_total_colour_q_upper'])
    GN = D1+8*qN/rmin
    GE = 4*su(F(30*path_length))+8*(qN+qC)/rmin
    GC = su(F(2))*(2*GN+GE)
    kineticN = chi*metric*GN*GN
    kineticC = chi*metric*GC*GC
    row = scalars(build_native(mean,2)['Pprofiles'][0])
    enc = mean['centre_enclosure']
    root0lo, root0hi = sl(rational(enc['g0_lower'])), su(rational(enc['g0_upper']))
    rootshi = su(rational(enc['gs_upper']))
    dmin = 2*row['onsite']*root0lo+g
    Kcap = 6*abs(row['path'][1][1])*rootshi
    b = (Kcap/dmin)**2
    if b>=1 or path_length<2 or rmin>=1 or qC>exact(field['actual_all_CAR_colour_charge_norm_upper']):
        raise ArithmeticError('Actual positive-Coulomb/carrier geometry hypothesis failed')
    Ax_minus_one_lower = ((1-b)/(1+b))**2
    resistance_lower = F(1,3)
    coulomb_lower = g*resistance_lower*Ax_minus_one_lower
    coulomb_upper = exact(green['source_Coulomb_addition_upper'])
    matter_lower = exact(green['FULL_original_Phi_INITIAL_CAR_matter_addition_positive_lower'])
    matter_price = exact(bridge['corrected_uniform_all_Theta_all_CAR_charged_Phi_freezing_action'])
    matter_upper = abs(row['onsite'])*root0hi+6*sum(abs(c0)*root0hi+abs(cj)*rootshi for c0,cj in row['path'].values())+matter_price
    source_lower = matter_lower+coulomb_lower-kineticN
    source_upper = matter_upper+coulomb_upper+kineticC
    if source_lower<=0 or source_upper<=source_lower:
        raise ArithmeticError('Full common-source excitation energy interval failed')
    desc = dict(plan=gd['plan'], plan_revision=REVISION, scientific_execution_revision=REVISION, input_receipts=dict(green=dict(path=str(green_path),sha256=hashlib.sha256(raw).hexdigest())), source_abi=gd['source_abi'], N=gd['N'], V=V, M=gd['M'], a=gd['a'], UEphys=gd['UEphys'], Lambda=gd['Lambda'], variance=gd['variance'], source_position=gd['source_position'], Ref_position=gd['Ref_position'], original_source='UNCHANGED neutral original normalized real Phi222 beta/heavyprojectedline, canonicalHermitianYgraph000/AP222 Gaussian2; initialCREATOR insertion ONLY', charged_source='E_x Psi / sqrt(1/2), ALL4coherentRefholes, SU4Wilson nonwrap pathlengthN/4 + REALlongitudinalGauss; no coexactintegerstring', energy_convention='Delta=<E_xPsi,H_C E_xPsi>/(1/2)-<Psi,H_N Psi> SAME original H and SAME neutralenergyconvention. No sectorown subtraction.', native_control=dict(N=2, source_position=[1,1,0], Ref_position=[0,0,0], theta_coordinates='0.13 sin(0.7 arange(M))'))
    values = dict(original_neutral_carrier_global_covariant_derivative_upper=GN, charged_CREATOR_covariant_derivative_upper=GE, actual_full_charged_normalized_carrier_global_covariant_derivative_upper=GC, original_neutral_carrier_Phi_colour_kinetic_upper=kineticN, actual_full_charged_carrier_Phi_colour_kinetic_upper=kineticC, actual_ALLtheta_source_Y_operator_norm_squared_upper=b, actual_ALLtheta_source_A_x_minus_one_positive_lower=Ax_minus_one_lower, actual_nonadjacent_realGauss_effective_resistance_lower=resistance_lower, actual_full_INITIAL_source_Coulomb_addition_positive_lower=coulomb_lower, actual_full_INITIAL_source_Coulomb_addition_upper=coulomb_upper, actual_full_INITIAL_source_CAR_matter_addition_positive_lower=matter_lower, actual_full_INITIAL_source_CAR_matter_addition_upper=matter_upper, full_same_H_compensated_SOURCE_energy_addition_positive_lower=source_lower, full_same_H_compensated_SOURCE_energy_addition_upper=source_upper)
    return dict(input_descriptor=desc, input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(), **{k:positive_exact(v) for k,v in values.items()}, actual_creation_probability_exact='1/2', original_Phi_boson_potential_energy_change_exact='0', original_neutral_heavy_matter_energy_change_exact='0', original_common_real_Phi_scalar_kinetic_energy_change_exact='0', actual_SOURCE_transverse_EM_kinetic_energy_change_exact='0', SOURCE_energy_addition_interval=dict(lower=signed_exact(source_lower),upper=signed_exact(source_upper)), actual_native_source_energy_control=actual_source_energy_control(mean),
        chiral_density_and_Coulomb_proof='ThisSOURCE-specific normalizedSlaterP obeys onsiteGamma5 P Gamma5=Q EACHTheta. ComplementQ determinantalnode-numberdistribution isthatof4-n_x underP; onsiteGamma5unitarypreservesALLnode-numberjointdistribution, so centredactive/auxiliaryrho thirdmoments0. No fullH/RefholeC-intertwining invoked. RestrictedRef4 E†E=(2-rho_active,x)/4, p1/2; [rho,E]=(delta_x-delta_Ref)E. OriginalC=g rho^T G rho gives DeltaC=g b^T G b-g b^T G S_active(x,:); sourceS_active(x,j)=2delta_xj-A_j, A_j=TrP_xjP_jx>=0, sumA=2. EXACT DeltaC=g sum_j [G_xj-G_Refj] A_j, also agreeswithfull addedDeltaP/Ref/Pauli secondmoment. ConnectedtorusdiscreteGaussdipolev=G(delta_x-delta_Ref) harmonicelsewhere; maximumprinciple v in[-DeltaG,+DeltaG], v_x=DeltaG. ThusDeltaC>=2gDeltaG(A_x-1). HermitianYnorm²<=b<1: R>=1/(1+b), onsiteTrP_xx²>=2[(1-r)^2+r²], Ax-1>=((1-b)/(1+b))². ActualsourceN/4notadjacentRef0, eachsixincidentedgesdisjoint; anyREALflowDell=delta_x-delta_Ref obeys twoCauchybudgets sum_sourceell²>=1/6,sum_Refell²>=1/6. MinimizerenergyR_eff=2DeltaG>=1/3. HencepositiveinitialSOURCECoulombcost>=g/3((1-b)/(1+b))², no V/stringdivergence/selfsubtraction/groundgate.',
        original_Phi_common_scalar_cancellation='AtEACHphysicalbodyPhi/Theta, neutralnormalizedcarrier sN=heavyprojectedline tensorcanonicalchargedray tensorRef4 hasnorm1; sC=E_U sN/sqrt(1/2) alsohasnorm1 bySPECIFICYHerm/nx2 andALLcolourWilsonunitarity. SamejointnormalrealbetaPhi/w_joint^1/2 andsameEMGaussian multiplyboth. Huge originalPhi scalarbosonpotential andneutralheavy pointenergy multiplySAMEbeta² andcancelEXACTLY. In the ORIGINAL real positive jointCart/SU4 kineticform, write covariantderivative(beta*s)=d(beta)*s+beta*Ds. CovariantconnectionantiHermitian, pointnorm1 givesRe<s,Ds>=0. Realcometric makescrossd(beta) term0; commonrealGaussian/b2/half-density derivativepartidentical, includingnonflatHaar/FP andmeasure-generatedscalar, cancels—not totalPhi-energy smallness. Remainingcarrierpositivekinetic boundedusingactualmetricM, exactprojectedheavyD1, orbitconnection/verticalcharge graph. GN=D1+8qN/rmin; qN=65sqrt(5V/2), Ref4singlet0. OriginalcolourqCartboundspermitorbitconnectionnorm<=q/rmin andverticalqgraph inside8q/rmin foractualrmin<1. WilsonplainLiegradient<=sqrt(30L) from15generators perLlinks, tCart/sqrt8||t||<=1/sqrt2; factor4safeguardsactualdexp/bodychart. CovariantCREATOR GE<=4sqrt(30L)+8(qN+qC)/rmin, Eop<=2. NormalizingbySPECIFICp1/2 givesGC<=sqrt2(2GN+GE). Samephysicalchi andfullmetricM givecarrierPhi/colourkinetic in[0,chi M GN²] before/[0,chi M GC²] after, soDelta in[-neutralcarrierprice,+chargedcarrierprice]. No flatmetric/zeroPhi-dynamics/ownsectorphase/hugePhiZPE assignedEM. Initialsupporteta ONLYinitialform calculation, not later-wave confinement.',
        remaining_energy_terms='FULLsource CARadditionfromactualc42 followsaddedDeltaP=Q Rx Q/2 andALLcolourWilsonunitarity; positivelower actualbinary64Wilsonmass−b367Phi-price. Rowupperofliteralphysicalactiveonebody +samePhi-price boundsaddition sinceTrDeltaP1. All63auxiliary onebody energiesunchanged atEACHPhi/Theta; spectatorheavyenergy/potentialexactpointnormcancellation. InitialtransverseEMkinetic0 fromactualc42chiralPH/realGaussianproof. Originalrho background128+4deltaRef, active+1/Ref3 explicitlyretained. SumactualpositiveCoulomb/CAR andcarrierPhiinterval yieldsFINITEPOSITIVE SAMEfullHsource excitationenergy, notneutral-onlysourceupper orwholePhi-toEMbudget.',
        scope='NEW initialcompensatedcharged SOURCEenergy interval/positivity withactualnativeCAR/Gauss/compactcolourderivative controls; parentreviewrequired forchiralDPP thirdmoment, dipolemaximumprinciple/resistance, andREALfulljointkineticcommon-scalar cancellation/carrierconnectionenvelopes. Particularpacketenergy NOTfullchargedsector lowergap/physicalelectronmass/pole/stationaryphotonvacuum/commoncone/Compton/statP-Qclosure. No newsourcefilter/ground/admissiongate. Neutral source, priorreceipts/sourceprobabilities andallphoton/source/receiverbytesunchanged.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()}, source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()}, runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--green',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = run(args.green)
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_sha256','SOURCE_energy_addition_interval','actual_native_source_energy_control','elapsed_seconds')},indent=2))

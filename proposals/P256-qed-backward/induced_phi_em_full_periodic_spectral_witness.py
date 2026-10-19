"""Actual integer-cycle periods and complex-toron full-Phi resolvent witness."""
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
from induced_phi_em_ygraph_retarded_transfer import SOURCE_BYTES as RETARDED_BYTES, upper_sqrt
from induced_phi_em_ygraph_clock_transfer import exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_macro_preparation import build_native, scalars
from induced_phi_em_gaussian_variational import matter_hamiltonian

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**RETARDED_BYTES, 'induced_phi_ir_c_full_action.py': (ROOT/'induced_phi_ir_c_full_action.py').read_bytes(), Path(__file__).name: Path(__file__).read_bytes()}
REVISION = '6ebc34a912158254456c535bd10259fca04201b14f446a778bba68d5ef63ecf2'


def integer_cycles(native: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    D = np.asarray(native['D'])
    Di = np.rint(D).astype(np.int64)
    if not np.array_equal(D, Di):
        raise ArithmeticError('Requires actual unscaled integer incidence')
    V, E = Di.shape
    roots = list(range(V))
    def root(x: int) -> int:
        while roots[x] != x:
            x = roots[x]
        return x
    tree = []
    for edge in range(E):
        vertices = np.flatnonzero(Di[:, edge])
        if len(vertices) != 2 or sorted(Di[vertices, edge].tolist()) != [-1, 1]:
            raise ArithmeticError('Native control requires two distinct endpoints')
        a, b = root(int(vertices[0])), root(int(vertices[1]))
        if a != b:
            tree.append(edge)
            roots[a] = b
    if len(tree) != V-1:
        raise ArithmeticError('Actual native graph disconnected')
    chords = [e for e in range(E) if e not in tree]
    T = Di[1:, tree]
    if abs(round(float(np.linalg.det(T)))) != 1:
        raise ArithmeticError('Actual reduced tree incidence not unimodular')
    cycle = np.zeros((E, len(chords)), dtype=np.int64)
    cycle[chords, np.arange(len(chords))] = 1
    cycle[tree, :] = np.rint(np.linalg.solve(T, -Di[1:, chords])).astype(np.int64)
    if np.any(Di@cycle) or not np.array_equal(cycle[chords, :], np.eye(len(chords), dtype=np.int64)):
        raise ArithmeticError('Actual integer fundamental cycles invalid')
    harmonic = np.zeros(E, dtype=np.int64)
    harmonic[0::3] = 1
    coordinates = harmonic[chords]
    if np.any(Di@harmonic) or not np.array_equal(cycle@coordinates, harmonic):
        raise ArithmeticError('Actual uniform x flow not in integer cycle lattice')
    if not np.any(coordinates == 1):
        raise ArithmeticError('Actual harmonic flow not primitive')
    # Replacing one chord basis vector by h has determinant its coefficient 1.
    j = int(np.flatnonzero(coordinates == 1)[0])
    cycle[:, j] = harmonic
    if np.any(Di@cycle):
        raise ArithmeticError('Primitive harmonic replacement failed')
    return Di, cycle, chords


def run(retarded_path: Path, mean_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    raw, mean_raw = retarded_path.read_bytes(), mean_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '35608d7063d4adc5347ca401719e94c62f06342d6fab679c64b9e1087101fbbd':
        raise ArithmeticError('Requires actual NEW e8 same-source graph epoch')
    if hashlib.sha256(mean_raw).hexdigest() != '30a4182e252463319caea5235176a0e5e87018a740ca0a95d76f1487796b4d78':
        raise ArithmeticError('Requires actual mass-resolved 1+63 mean epoch')
    retarded, mean = json.loads(raw), json.loads(mean_raw)
    old = retarded['input_descriptor']
    N, V, M = int(old['N']), int(old['V']), int(old['M'])
    a, g, Lambda = F(old['a']), F(old['UEphys'])/F(old['a']), F(2)**8192
    if (N, V, M, a, g) != (262144, 2**54, 2**55+1, F(1, 64), F(64)):
        raise ArithmeticError('Original macro physical law changed')
    c = Lambda/(8*a*V)
    if exact(retarded['global_quartic_coercivity']) != c:
        raise ArithmeticError('Literal Cart TrPhiPhi/4 coercivity mismatch')
    B = exact(retarded['recomputed_global_full_H_lower_shift'])
    source_graph = exact(retarded['recomputed_actual_full_H_source_graph_upper'])
    # [-B, source_graph] contains the bottom of full spectrum by the SAME source.
    energy_abs = B+source_graph
    native = build_native(mean, 2)
    if sorted((int(p['load']), F(str(p['model'].m))) for p in native['Pprofiles']) != [(1, F(1, 5)), (63, F(2))]:
        raise ArithmeticError('Requires actual active1+aux63 mass-resolved rows')
    xcap = F(0)
    for profile in native['Pprofiles']:
        row = scalars(profile)
        xcap += 32*upper_sqrt(F(V))*row['load']*sum((abs(c0)+abs(cj) for c0, cj in row['path'].values()), F(0))
    tau = F(2)**128
    resolvent = upper_sqrt(F(V))/(g*tau)
    rho_resolvent = upper_sqrt(upper_sqrt(((g*tau*tau+energy_abs+B)*resolvent**4+resolvent**3)/c))
    neumann = xcap*rho_resolvent
    if neumann >= F(1, 2):
        raise ArithmeticError('Actual complex-toron witness not invertible')
    Di, cycle, chords = integer_cycles(native)
    Q = np.asarray(native['Q'])
    theta = Q@np.linspace(-0.27, 0.193, Q.shape[1])
    period_errors = []
    wrong_errors = []
    for profile in native['Pprofiles']:
        h0 = matter_hamiltonian(native, profile, theta)
        for j in range(cycle.shape[1]):
            hj = matter_hamiltonian(native, profile, theta+2*np.pi*cycle[:, j])
            period_errors.append(float(np.linalg.norm(hj-h0, 2)))
        wrong = matter_hamiltonian(native, profile, theta+np.pi*cycle[:, 0])
        wrong_errors.append(float(np.linalg.norm(wrong-h0, 2)))
    if max(period_errors) > 1e-10 or min(wrong_errors) < 1:
        raise ArithmeticError('Actual period/wrong-pi discrimination failed')
    desc = dict(plan=old['plan'], plan_revision=REVISION, scientific_execution_revision=REVISION, input_execution_revision=old['scientific_execution_revision'], inputs=dict(retarded=dict(path=str(retarded_path), sha256=hashlib.sha256(raw).hexdigest()), mean=dict(path=str(mean_path), sha256=hashlib.sha256(mean_raw).hexdigest())), N=N, V=V, M=M, a=str(a), UEphys=str(old['UEphys']), Lambda='2^8192', loads='active1m0.2+aux63m2; neutralheavy1; Ref4 NET0', original_noncompact_EM='R^M; no bareB, no theta confinement, no physical compactification', strict_period_sublattice='2pi(ker Dunit intersect Z^(3V)), fullrankM; additional gauge-twisted periods NOTrequired', spectral_probe='CONFIGURATION-space toron Bloch quasifieldmomentum, NOT spatial photon/electron wavevector', complex_tau=str(tau), native_size=2, actual_full_source='UNCHANGED actualPhi222/chi000/Gaussian2 from e8, usedONLYfor source H graph and literal global Cartcoercivity; NO Dirichlet field reference/clock residual as physical H')
    values = dict(global_quartic_coercivity=c, global_full_H_and_x_hop_removed_A_lower_shift=B, actual_full_source_H_graph_upper=source_graph, energy_absolute_majorant_covering_full_ground_infimum=energy_abs, toron_real_halfstep_quasimomentum=1/(2*upper_sqrt(F(V))), complex_toron_imaginary_quasimomentum=tau, removed_all64_x_hop_global_rho_operator_cap=xcap, split_A_complex_resolvent_norm_upper=resolvent, global_rho_times_split_A_resolvent_norm_upper=rho_resolvent, all64_x_hop_times_split_A_resolvent_Neumann_norm_upper=neumann)
    return dict(input_descriptor=desc, input_sha256=hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest(), **{key: positive_exact(value) for key, value in values.items()}, native_control=dict(V=int(native['V']), incidence_shape=list(Di.shape), cycle_rank=int(cycle.shape[1]), M=int(Q.shape[1]), integer_incidence=Di.tolist(), integer_basis_with_primitive_uniform_x_flow=cycle.tolist(), original_chord_edges=chords, exact_integer_Gauss_cycle_defect=int(np.max(np.abs(Di@cycle))), full_cycle_period_matter_operator_errors=period_errors, wrong_pi_matter_operator_errors=wrong_errors, full_loaded_CAR_period_identity='Every charged one/two Peierls path phase gainsinteger2pi. Thusactual secondquantization atALL1+63 fixedchargedloads exactlyperiodic; native onebody control is NOT a finiteCAR spectrum substitute. Neutralheavy/Phi/SU4/Coulomb/primitivePi2 are unchanged.'),
        actual_full_H_split='H(k)=A(k)+Vx. Vx containsALLchargedone/twox hopping withARBITRARYCartPhi/noncommutingSU4 links. A retainschargedonsite+yz hopping, originalneutralheavy, Cart/SU4 kinetic+quartic/alignment/curl, Coulomb and all EMPi2. None dependonconfiguration-space uniformx toron exceptgPi_z². [Areal,Pi_z]=0; removingx hopping leavescoarseglobalboundA+B>=c rho^4+g Pi_other^2 sinceitsremainingmatter bound isno largerthanalldirections. GlobalPsi space, NOTlocalwell/bodychart.',
        split_notation='Areal in complex-resolvent formula is A(real)-gPi_z(real)^2, AFTER removingtoronkinetic. Complexk shiftsONLYPi_z, NOTtoronCOORDINATE: allPeierlsremainunitmodulus andVx hasNO exp(tau) factor.',
        complex_resolvent_proof='Uniformx integerflow h hasnormsqrtV andisprimitive: inintegerfundamentalcyclebasis itschordcoordinates0/1 include1. Extendtointegerperiodbasis withh first. Period2pi h forcesPi_z(realhalfstep=1/(2sqrtV)) eigenvalues(n+1/2)/sqrtV. All other realBlochquasimomenta orthogonalh arbitrary. A(k)-E=Areal+gPi_z(real)^2-g tau²-E+i2g tau Pi_z(real); strongcommutation givesresolventnormr<=sqrtV/(g tau). Foru=R f, realform c||rho²u||²<=||u||||f||+(g tau²+|E|+B)||u||²; Cauchy gives||rho R||⁴<=((g tau²+|E|+B)r⁴+r³)/c. ActualCAR allxforward/reverse bound||Vx u||<=32sqrtV sum_load_lengths sum|c| ||rho u||. Witnessprice<1 givesinvertibilitybyNeumann. ForANYfixedfiniteE, sameboundtendsto0 astau→infinity; tau128 numericallypriceswindowcoveringactualgroundinf, notanassumedgroundstate.',
        full_period_Bloch_and_analytic_Fredholm='ConnectedintegerD hasrankV-1, integercyclelatticefullrankM=2V+1. DecomposeactualR^M EM intoordinaryBlochfibers overthisEXACTstrictperiodsublattice, withoutreplacingphysicalHilbertspace/CCR. FiberEMtoruscompact, SU4compact, finiteCAR andglobalCartquartic-minuslinear confining; positiveoriginalCart/SU4/EM kinetic yieldscompactformembedding/compactresolvent. CompactcolourGaussprojectionandfixedCARsectors commuteallperiods; positivityandcompactembeddingremain. Complexquasimomentum addsrelativelygraph-smallfirstPi_z +constant oncommonoperator domain, analyticFredholmfamilyindex0. (H(k)-E)(H(kreal)-z0)^-1 = I+compact analytic. Complexinvertibilityabove rulesoutidenticallysingularfamily, sorealfixedE zerosalongk_z discrete. Fubini overallotherrealBlochcoordinates => fixedE fiber-eigensupportmeasurezero. ThusnoL2point spectrumforactualfiniteN originalnoncompactEMfullH, conditionalparentreviewofglobalcore/coercivity/compactfiber/analyticargument; thisisNOTnumericaldiagonalization.',
        stationary_and_continuation_boundary='No normalizedL2 ground/eigenstate ornormaltraceclassstationarydensity inTHISfiniteN noncompactperiodiclaw ifproofadmitted: apositivecompactdensitycommutingH wouldgivefinite-dimensionalinvarianteigenspace. DoesNOTexclude generalizedBloch/algebraicvacua, infinitevolume/GNSstates, particle/currentresponsepoles, commoncone orQED. Blochfiber groundexists mathematicallybycompactresolvent, NOTcomputednoridentifiedwithfrozenGaussian source. Next constructivequantumtargetmayuseexplicitgeneralizedBlochstate ONLYafterseparateactualpreparation/law/probe ABI; Theta multiplicationdoesNOTpreservefixedBlochfiber, whilePi/periodiccurrentdo. No silentphysicalcompactification, no current source/vacuum orpolepromotion. ConfigurationFloquetbands are NOT spatialphoton dispersion.',
        source_at_import={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes_utf8=data.decode()) for name, data in SOURCE_BYTES.items()}, source_unchanged_since_import={name: (ROOT/name).read_bytes()==data for name, data in SOURCE_BYTES.items()}, runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__), elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retarded', type=Path, required=True)
    parser.add_argument('--mean', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.retarded, args.mean)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('input_sha256', 'all64_x_hop_times_split_A_resolvent_Neumann_norm_upper', 'native_control', 'elapsed_seconds')}, indent=2))

"""Actual same-source full-Phi four-current transfer, including source two-PH."""
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
from induced_phi_em_full_inserted_response import SOURCE_BYTES as INSERTED_BYTES
from induced_phi_em_ygraph_retarded_transfer import upper_sqrt
from induced_phi_em_ygraph_clock_transfer import checked, exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_macro_preparation import build_native, scalars
from induced_phi_ir_well_transfer import rational
from induced_reference_field_following import path_derivative
from induced_reference_electric_graph import spin_basis

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**INSERTED_BYTES, 'induced_reference_electric_graph.py': (ROOT/'induced_reference_electric_graph.py').read_bytes(), Path(__file__).name: Path(__file__).read_bytes()}
REVISION = '6ebc34a912158254456c535bd10259fca04201b14f446a778bba68d5ef63ecf2'


def sector_action(h: np.ndarray, psi: np.ndarray, masks: tuple[int, ...],
                  lookup: dict[int, int]) -> np.ndarray:
    """Fixed-number CAR reference; same annihilate/create signs as reference_hop."""
    out = np.zeros_like(psi)
    entries = [np.flatnonzero(h[:, source] != 0) for source in range(h.shape[0])]
    for column, mask in enumerate(masks):
        for source in range(h.shape[0]):
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
                sign = -1 if swaps & 1 else 1
                out[lookup[output]] += sign*h[target, source]*psi[column]
    return out


def actual_closed_orbit_car(mean: dict[str, Any]) -> list[dict[str, Any]]:
    """Literal N6 q=2 closed three-momentum orbit, all12spin modes/N6particles."""
    native = build_native(mean, 6)
    B, frame_proof = spin_basis(native)
    sites, V = native['sites'], native['V']
    momenta = np.array([[x, 1, 0] for x in (0, 2, 4)])
    U = np.exp(1j*sites@(2*np.pi*(momenta+.5)/6).T)/np.sqrt(V)
    orbital = np.einsum('xa,st->xsat', U, np.eye(4)).reshape(4*V, 12)
    Bfull = np.kron(np.eye(3), B)
    pp = np.array([4*x+s for x in range(3) for s in (0, 1)])
    nn = pp+2
    masks = tuple(sum(1 << i for i in inds) for inds in itertools.combinations(range(12), 6))
    lookup = {mask:i for i,mask in enumerate(masks)}
    occupations = [[i for i in range(12) if mask & (1 << i)] for mask in masks]
    f = np.cos(4*np.pi*sites[:, 0]/6)*np.sqrt(2/V)
    controls = []
    for profile in native['Pprofiles']:
        h = orbital.conj().T@profile['hcentre']@orbital
        hclosure = float(np.linalg.norm(profile['hcentre']@orbital-orbital@h))
        hb = Bfull.conj().T@h@Bfull
        K = -1j*hb[np.ix_(pp, nn)]
        mass0 = float(np.trace(B[:, :2].conj().T@profile['onsite']@B[:, :2]).real/2)
        Y = K/(2*mass0+native['UE']/native['a'])
        if np.linalg.norm(Y-Y.conj().T)>1e-10:
            raise ArithmeticError('Actual closed-orbit canonical Y not Hermitian')
        R = np.linalg.inv(np.eye(6)+Y@Y)
        values, vectors = np.linalg.eigh(R)
        rootR = (vectors*np.sqrt(values))@vectors.conj().T
        W = np.empty((12, 6), complex); W[pp] = -1j*Y@rootR; W[nn] = rootR
        W = Bfull@W
        P = W@W.conj().T; Q = np.eye(12)-P
        psi = np.array([np.linalg.det(W[inds]) for inds in occupations])
        if abs(np.vdot(psi, psi)-1)>1e-10 or np.linalg.norm(P@P-P)>1e-10:
            raise ArithmeticError('Actual non-ground graph Slater invalid')
        Js, closure = [], []
        for axis in (1, 2):
            direction = np.zeros(3*V); direction[axis::3] = f
            fullJ = path_derivative(native, profile, np.zeros(3*V), direction, 1)
            j = orbital.conj().T@fullJ@orbital
            closure.append(float(np.linalg.norm(fullJ@orbital-orbital@j)))
            Js.append(j)
        hf, hg = Js
        mu_f, mu_g = np.trace(P@hf), np.trace(P@hg)
        bf, bg = Q@hf@P, Q@hg@P
        scalar = mu_f*mu_g+np.trace(P@hf@Q@hg)
        one = mu_g*bf+mu_f*bg+Q@hf@Q@hg@P-Q@hg@P@hf@P
        exact_double = sector_action(hf, sector_action(hg, psi, masks, lookup), masks, lookup)
        one_ray = sector_action(one, psi, masks, lookup)
        two_ray = sector_action(bf, sector_action(bg, psi, masks, lookup), masks, lookup)
        reconstructed = scalar*psi+one_ray+two_ray
        defect = float(np.linalg.norm(exact_double-reconstructed))
        wrong = float(np.linalg.norm(exact_double-scalar*psi-one_ray))
        hpsi = sector_action(h, psi, masks, lookup)
        energy = np.vdot(psi, hpsi)
        stationarity = float(np.linalg.norm(hpsi-energy*psi))
        pair_bound = np.sqrt(2)*np.linalg.norm(bf, 'fro')*np.linalg.norm(bg, 'fro')
        if max(closure)>1e-9 or hclosure>1e-9 or defect>1e-9 or wrong<1e-4 or stationarity<1e-4 or np.linalg.norm(two_ray)>pair_bound+1e-10:
            raise ArithmeticError('Actual CAR two-PH/source control failed')
        controls.append(dict(load=int(profile['load']), physical_N=6, q_integer=2, retained_momenta=momenta.tolist(), all_spin_modes=12, filled_particles=6, whole_CAR_sector_dimension=len(masks), true_AP222_current_orbit_closure=closure, original_onebody_H_orbit_closure=hclosure, actual_source_norm=float(np.linalg.norm(psi)), actual_non_eigenstate_original_onebody_CAR_residual=stationarity, exact_current_double_action_norm=float(np.linalg.norm(exact_double)), zero_PH_component_norm=float(abs(scalar)), one_PH_component_norm=float(np.linalg.norm(one_ray)), two_PH_component_norm=float(np.linalg.norm(two_ray)), exact_zero_one_two_PH_reconstruction_error=defect, wrong_discarded_two_PH_action_error=wrong, source_two_PH_exterior_upper=float(pair_bound), frame_proof=frame_proof))
    return controls


def run(inserted_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    raw = inserted_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '4e65c2b0d0ee5866020f52e2d8c908552f8c4f60793b41dbf45bec5c690faf07':
        raise ArithmeticError('Requires actual248f inserted full-Phi epoch')
    inserted = json.loads(raw)
    _, force = checked(inserted['input_descriptor']['input_receipts']['force'])
    _, current = checked(force['input_descriptor']['input_receipts']['current'])
    _, retarded = checked(current['input_descriptor']['input_receipts']['retarded'])
    _, field = checked(retarded['input_descriptor']['input_receipts']['clock'])
    _, bridge = checked(field['input_descriptor']['input_receipts']['bridge'])
    _, mean = checked(bridge['input_descriptor']['input_receipts']['mean'])
    if inserted['source_at_import']['induced_phi_em_full_inserted_response.py']['sha256'] != hashlib.sha256(SOURCE_BYTES['induced_phi_em_full_inserted_response.py']).hexdigest():
        raise ArithmeticError('Actual full-Phi response source epoch changed')
    fd = force['input_descriptor']
    V, s, a = int(fd['V']), F(fd['variance']), F(fd['a'])
    if (fd['N'], V, s, a, fd['clock']) != (262144, 2**54, F(2), F(1, 64), '4096'):
        raise ArithmeticError('Unchanged actual macro source required')
    prices = inserted['source_prices']
    Quniform = exact(prices['uniform_all64_reference_quantum_PH_Gram_upper'])
    L = exact(prices['source_conditional_mean_complex_gradient_Lipschitz'])
    enc = mean['centre_enclosure']
    g0, gs = upper_sqrt(rational(enc['g0_upper'])), upper_sqrt(rational(enc['gs_upper']))
    hmax = F(0)
    for profile in build_native(mean, 2)['Pprofiles']:
        row = scalars(profile)
        bound = 4*upper_sqrt(F(2, V))*sum((length*(abs(c0)*g0+abs(cj)*gs) for length, (c0, cj) in row['path'].items()), F(0))
        hmax = max(hmax, bound)  # Directsumonebody: MAX, not SUMflavourloads.
    sqrtQ = upper_sqrt(Quniform)
    double_ref = upper_sqrt(F(5))*s*L*L+2*sqrtQ*upper_sqrt(s)*L+(1+upper_sqrt(F(2)))*Quniform+2*hmax*sqrtQ
    Jprice = upper_sqrt(F(2))*exact(bridge['corrected_initial_whole_Theta_current_operator_price'])
    Jref = exact(force['prices']['Jref_ALLthetaCAR_operator_cap'])
    Jcap = exact(force['prices']['FULL_initial_current_operator_cap'])
    double_initial_price = Jprice*(Jcap+Jref)
    double_full = double_ref+double_initial_price
    epsilon = exact(field['alltime_original_Phi_source_reference_wave_error'])
    double_packet_error = double_ref*epsilon+double_initial_price
    action_error = exact(inserted['actual_inserted_response_rows']['initial_FULL_current_packet']['alltime_current_action_packet_difference'])
    nref = exact(inserted['actual_inserted_response_rows']['initial_FULL_current_packet']['reference_initial_packet_norm'])
    # 4D partial transpose: three orderederrors; Schmidt sqrt2/map gives SIX.
    response_error = 6*(action_error*double_full+Jref*nref*double_packet_error)
    gram_lower = exact(bridge['corrected_full_initial_two_current_Gram_lower'])
    values = dict(ALLspecies_onebody_current_operator_cap=hmax, reference_INITIAL_double_current_source_action_map_upper=double_ref, FULL_initial_double_current_source_action_map_upper=double_full, FULL_vs_reference_initial_double_current_source_action_price=double_initial_price, alltime_evolved_INITIAL_double_current_source_packet_error=double_packet_error, alltime_FULL_CURRENT_source_canonical_CURRENT_retarded_tensor_operator_error=response_error, alltime_physical_CURRENT_source_physical_CURRENT_retarded_tensor_operator_error=a**4*response_error, EXACT_full_INITIAL_CURRENT_Gram_relative_canonical_CURRENT_response_error=response_error/gram_lower, EXACT_full_INITIAL_physical_CURRENT_Gram_relative_physical_CURRENT_response_error=a*a*response_error/gram_lower)
    if response_error >= F(1, 1000):
        raise ArithmeticError('Actual full-Phi four-current comparison not useful')
    controls = actual_closed_orbit_car(mean)
    desc = dict(plan=fd['plan'], plan_revision=REVISION, scientific_execution_revision=REVISION, input_receipts=dict(inserted=dict(path=str(inserted_path), sha256=hashlib.sha256(raw).hexdigest())), source_abi=fd['source_abi'], N=fd['N'], V=V, M=fd['M'], variance=str(s), a=str(a), clock=fd['clock'], source=fd['source'], reference=fd['reference'], response='FULLactualnonstationarycurrent-source retardedCURRENT 4x4tensor, SAMEunchangedsource/Href/Phi; initialsourceWickONLY, nointeractingSlater assumption', native='ActualclosedN6q2three-momentumorbit/all12spin/all924 six-particleCAR states, activeandauxsourcegraphs, NOTwhole64multiparticle/EM/Phi simulation')
    return dict(input_descriptor=desc, input_sha256=hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest(), **{key:positive_exact(value) for key,value in values.items()}, actual_native_closed_orbit_CAR_controls=controls,
        source_double_current_PH_proof='AtEACHrealTheta originalsourceisFULLdirectsumcanonicalSlater(all1+63 occupied2V/species), notassumptionondynamics. Jf=dGamma(hf), mu_f=SUMloadsTr(P hf), bf=Q hf P. Jf Jg chi=(mu_fmu_g+Tr(P hf Q hg))chi+dGamma(mu_gbf+mu_fbg+QhfQhgP-QhgPhfP)chi+dGamma(bf)dGamma(bg)chi. Theseareorthogonal0/1/2-PH; ALLsame/crossflavourpairs retainedbydirectsum. QuantumPH<=Quniformpointwise; vacuumcontraction<=Quniform, singlePH excitationcorrection<=2hmaxsqrtQ. Exterior2norm²=QfQg+|Tr bfdag bg|²-Tr(bfdagbf bgdagbg)-Tr(bfdagbg bgdagbf)<=2QfQg; neitherpositivePauli exclusion term discardedfromexactidentity. Hencepointwisenorm<=|mu_fmu_g|+sqrtQ(|mu_f|+|mu_g|)+(1+sqrt2)Q+2hmaxsqrtQ. SourceGaussianPoincare complexmu E|mu|4<=5s²L4 andE|mu|²<=sL² =>L2bound double_ref=sqrt5sL²+2sqrtQsqrt(s)L+(1+sqrt2)Q+2hmaxsqrtQ. MeanssummedBEFOREGaussianpowers; hmaxMAXspeciesdirectsum, not63multiplication. Onebodyboundfromtrueaxispaths/complexpolarization rowcap/normalizedf; matrixHF0particle-holeprojectionnotretaggedtosourceP. Nativeactualnon-groundsourcecontrolverifiesdecompositionandwrongdrop2PH.',
        actual_full_Phi_retarded_consumer='FULLinitialJJ productdiff<=Jprice(Jcap+Jref) byinitialactualchartuniformb367price; multiplicationpreservesinitialPhi support, NOTclaimaboutlaterfullwaves. CommonH0 actsontheORIGINALPhi source; chargedREFERENCEJJchi isPhi-independent arbitraryCARvectorofnormdouble_ref, soNEW76adwaveerrorlinearlydouble_ref epsilon; addinitialFULLJJprice. FullCURRENTsource currentACTIONerror comesactual248fform-onlyweightedPhi/originalfieldboundary/forceWard onXi=JfullPsi, NOTconservedsourceJnormsubstitutedforJ(t). OrderedF(alpha i,beta j)=<Ji U JalphaPsi,UJj JbetaPsi>. InitialJJmapnormdouble_full, referenceJ_i U Xi_ref mapnormJref*nref ALLtheta/CAR. Productbounds→4D maps costSchmidt sqrt2each, orderederror<=2[action_error double_full+Jref*nref double_packet_error]. ReverseorderedG=(Fdag)^partialtranspose_CURRENTprobe; 4DPT<=2opnorm =>RETARDEDerrorSIXbrackets, nofalseentrywiseconjugation. ActualnonstationaryFULLPhi fourcurrentresponsecomparison, ALL|t|<=4096/allmacroallowedq/twocomplexpolarization; noTaylor/contact/survivalproxy.',
        physical_units_and_scope='JA=aJTheta, bothCURRENTsource insertions andbothCURRENTprobe factors =>rawphysical4currenterror=a4canonical. SourceGramA=a²Gfull, SAMEexactfunctionalwhitening givesphysicalrelativeerror=a²canonical/g_lower; boundneverreplacesactualGram. ParentreviewrequiredofSOURCE0/1/2PH/Wick/Gaussianmoment andprevious248f/76ad/fullGauss/domain/coercivity/boundary. CLIactualmacrocomparisonprices+actual924CAR controls, NOTnumericallyevaluatedHref/fullHfourcurrenttimecurve, notCompton/scattering/pole/commoncone/vacuum/statisticalQ codeclosure. Frozenallsource/receiver/probabilitybytes unchanged, nofiniteSCREENgate.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(), bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()}, source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()}, runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__), elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inserted', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.inserted)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('input_sha256', 'alltime_FULL_CURRENT_source_canonical_CURRENT_retarded_tensor_operator_error', 'actual_native_closed_orbit_CAR_controls', 'elapsed_seconds')}, indent=2))

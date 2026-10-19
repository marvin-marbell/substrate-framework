"""All-time instantaneous contact expectation, with actual full-Phi prices."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from induced_phi_em_ygraph_retarded_transfer import SOURCE_BYTES as RETARDED_BYTES
from induced_phi_em_ygraph_clock_transfer import checked, exact
from induced_phi_em_ygraph_initial_contact import positive_exact
from induced_phi_em_macro_preparation import build_native, scalars
from induced_phi_ir_well_transfer import sqrt_bounds

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**RETARDED_BYTES, Path(__file__).name: Path(__file__).read_bytes()}


def run(retarded_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    raw = retarded_path.read_bytes()
    retarded = json.loads(raw)
    rd = retarded['input_descriptor']
    _, clock = checked(rd['input_receipts']['clock'])
    _, bridge = checked(clock['input_descriptor']['input_receipts']['bridge'])
    _, mean = checked(bridge['input_descriptor']['input_receipts']['mean'])
    if rd['N'] != 262144 or rd['V'] != 262144**3 or rd['clock'] != '4096' or rd['a'] != '1/64' or rd['variance'] != '2':
        raise ArithmeticError('Actual source/clock/coordinate contract changed')
    for receipt, name in ((retarded, 'induced_phi_em_ygraph_retarded_transfer.py'), (clock, 'induced_phi_em_ygraph_evolving_field_clock.py'), (bridge, 'induced_phi_em_ygraph_centre_bridge.py')):
        if receipt['source_at_import'][name]['sha256'] != hashlib.sha256(SOURCE_BYTES[name]).hexdigest():
            raise ArithmeticError('Physical transfer source epoch changed')
    V, a = rd['V'], F(1, 64)
    global_cap = F(0)
    mode_factor = sqrt_bounds(F(2, V))[1]
    for profile in build_native(mean, 2)['Pprofiles']:
        row = scalars(profile)
        global_cap += 32*row['load']*mode_factor*sum((length*length*(abs(c0)+abs(cj)) for length, (c0, cj) in row['path'].items()), F(0))
    weighted = exact(retarded['alltime_Phi_weighted_actual_source_to_cut_low_reference_error'])
    evolving_price = exact(clock['alltime_reference_to_C000_contact_operator_action_price'])
    gap = exact(clock['original_reference_vs_cut_low_reference_norm_error'])
    epsilon = exact(clock['alltime_original_Phi_source_reference_wave_error'])
    reference_cap = exact(bridge['reference_whole_Theta_contact_operator_cap'])
    action_error = global_cap*weighted+evolving_price+reference_cap*gap
    expectation_error = action_error+reference_cap*epsilon
    physical_error = a*a*expectation_error
    if expectation_error >= F(1, 1000):
        raise ArithmeticError('Actual evolving contact transfer not useful')
    desc = dict(plan=rd['plan'], plan_revision=rd['plan_revision'], scientific_execution_revision=rd['scientific_execution_revision'], input_receipts=dict(retarded=dict(path=str(retarded_path), sha256=hashlib.sha256(raw).hexdigest())), source_abi=rd['source_abi'], N=rd['N'], V=V, M=rd['M'], clock='4096', a=str(a), variance='2', target='Actual2x2 instantaneouscontactexpectation matrix underFULLH/source, comparedtoSAMEchargedHref000 evolvedsource; ORIGINALPhi source evolvesexactHphi, nofieldground replacement')
    return dict(input_descriptor=desc, input_sha256=hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest(), actual_global_canonical_contact_Phi_weight_cap=positive_exact(global_cap), alltime_canonical_contact_action_error=positive_exact(action_error), alltime_actual_full_H_canonical_contact_expectation_matrix_error=positive_exact(expectation_error), alltime_actual_full_H_physical_A_contact_expectation_matrix_error=positive_exact(physical_error),
        exact_operator_proof='Cij=partial_Theta_i partial_Theta_j H. LiteralonlyAXIALone/twolinkPeierls paths: cos(qx)y/z probesdisjointedgeaxes, soCyz=Czy=0 EXACTLY forarbitraryPhi/SU4/Theta. gPi²/Cart/SU4/boson/neutralheavy/Coulomb havezeroThetaHessian. CartTr4 Phiop<=2rho_x; fourspinforwardreverseCAR<=16hop, lengthL probephase. Unitrealcosf hasmaxf²=2/V,sumf²=1; hence sumf⁴<=2/V forALLallowedmacroq, INCLUDINGq=N/4 (whereexactsumf⁴=2/V, not3/(2V)). CS shiftedf² gives||C_f u||<=32sqrt(2/V)sum_load_length L²sum|c| ||rhoPhi u||. Complexpolarizationbilinear sum|a_j b_j|<=1 boundswhole2x2diagonalmatrix, nofakePH-onlycap.',
        actual_alltime_consumer='UseNEWretarded full−CUTLOW weightedPhierror, BOTHstatesD(Hfull), no actualevolvedsourcecompact-support assumption. ContactonreferencecutlowpricefromNEWCONSERVEDevolvingPhi/Liemoments, notinitialeta. AddingCrefcap*(high+tail) comparesORIGINALunitaryPhi-source reference. Thusactionerror=CglobalweightedPhi+evolving_contact_price+Crefcapgap. Sourcefullandoriginalreference norm1; secondinnerproductcomparison addsCrefcap*original_wave_error. Referencefieldunitarity makesreferenceexpectation EXACTchargedHref000 expectation. BoundholdsEVERY|t|<=4096, allmacroq andtwo-complexpolarizations. PhysicalcouplerTheta=aAphys impliesC_A=a²C_Theta, so physicalcontacterror=a²canonicalerror. NOTinversea²conversion.',
        review_boundary='NEWactualfull-Phi alltimeinstantaneousCONTACT comparison CONSTRUCTION forparentreview ofnewcoercivefield/spectralboundary/fullgraph/core premises. DoesNOTevaluateactualHref/fullHcontacttimecurve, assumecontactpositiveforalltime, assertDthird(t)=4g²contact(t), orreplaceactualretardedkernelbyTaylorcoefficient. SeparateexecutedretardedconsumerprovidesitsownvectorWardprice. No stationarity/ground/vacuum/pole/commoncone/Compton/statP-Q promotion.',
        source_at_import={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes_utf8=data.decode()) for name, data in SOURCE_BYTES.items()}, source_unchanged_since_import={name: (ROOT/name).read_bytes()==data for name, data in SOURCE_BYTES.items()}, runtime=dict(executable=sys.executable, python=sys.version), elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retarded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.retarded)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('input_sha256', 'alltime_actual_full_H_canonical_contact_expectation_matrix_error', 'alltime_actual_full_H_physical_A_contact_expectation_matrix_error', 'elapsed_seconds')}, indent=2))

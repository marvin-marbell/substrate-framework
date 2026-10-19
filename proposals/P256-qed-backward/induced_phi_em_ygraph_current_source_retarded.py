"""Use actual INITIAL current-source norm, never an evolved-current shortcut."""
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
from induced_phi_ir_well_transfer import sqrt_bounds

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**RETARDED_BYTES, Path(__file__).name: Path(__file__).read_bytes()}


def run(retarded_path: Path, source_norm_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    paths = dict(retarded=retarded_path, source_norm=source_norm_path)
    raws = {name: path.read_bytes() for name, path in paths.items()}
    retarded, norm = json.loads(raws['retarded']), json.loads(raws['source_norm'])
    rd = retarded['input_descriptor']
    _, clock = checked(rd['input_receipts']['clock'])
    _, bridge = checked(clock['input_descriptor']['input_receipts']['bridge'])
    if hashlib.sha256(raws['source_norm']).hexdigest() != '541db4ec7cf425f8a1307b4e1c9168b18465130b6e312607ab66b568faa05733':
        raise ArithmeticError('Closed-loop source norm epoch changed')
    if norm['mean_sha256'] != bridge['input_descriptor']['input_receipts']['mean']['sha256'] or norm['valid_reference_Gram_sha256'] != '6912ca8455bb14e233e43fa71608325039c84497f523ba10e551108f83469ddb':
        raise ArithmeticError('Source mean/reference Gram mismatch')
    if [row['load'] for row in norm['profiles']] != [1, 63] or rd['N'] != 262144 or rd['clock'] != '4096' or rd['variance'] != '2' or rd['a'] != '1/64':
        raise ArithmeticError('Actual source/box/clock changed')
    for receipt, name in ((retarded, 'induced_phi_em_ygraph_retarded_transfer.py'), (clock, 'induced_phi_em_ygraph_evolving_field_clock.py'), (bridge, 'induced_phi_em_ygraph_centre_bridge.py')):
        if receipt['source_at_import'][name]['sha256'] != hashlib.sha256(SOURCE_BYTES[name]).hexdigest():
            raise ArithmeticError('Physical retarded transfer epoch changed')
    a = F(1, 64)
    ref_Gram_upper = F(norm['macro_reference_two_current_action_Gram_maximum_eigenvalue_upper'])
    ref_source_norm = sqrt_bounds(ref_Gram_upper)[1]
    sqrt2 = sqrt_bounds(F(2))[1]
    initial_price = sqrt2*exact(bridge['corrected_initial_whole_Theta_current_operator_price'])
    full_initial_norm = ref_source_norm+initial_price
    full_initial_Gram_upper = full_initial_norm**2
    full_initial_Gram_lower = exact(bridge['corrected_full_initial_two_current_Gram_lower'])
    epsilon = exact(clock['alltime_original_Phi_source_reference_wave_error'])
    current_action_error = exact(retarded['alltime_two_current_action_map_error'])
    # This remains an ALL-Theta/CAR operator cap: Jref Uref(t)chi is not conserved.
    Jref_operator_cap = sqrt2*exact(bridge['reference_whole_Theta_current_operator_cap'])
    evolved_initial_current_packet_error = ref_source_norm*epsilon+initial_price
    ordered_error = current_action_error*full_initial_norm+Jref_operator_cap*evolved_initial_current_packet_error
    retarded_error = 2*ordered_error
    relative_error = retarded_error/full_initial_Gram_lower
    if min(ref_Gram_upper, full_initial_Gram_lower) <= 0 or relative_error >= F(1, 1000):
        raise ArithmeticError('Actual source-norm refinement invalid')
    desc = dict(plan=rd['plan'], plan_revision=rd['plan_revision'], scientific_execution_revision=rd['scientific_execution_revision'], input_receipts={name: dict(path=str(paths[name]), sha256=hashlib.sha256(raws[name]).hexdigest()) for name in paths}, source_abi=rd['source_abi'], N=rd['N'], V=rd['V'], M=rd['M'], variance='2', a=str(a), clock='4096', scope='ORIGINALunfilteredGaussian2/all64canonicalY000 source; actualPhi222 originalsource/wave; AP222; initialcurrentGramrefinementONLY')
    values = dict(reference_INITIAL_current_source_Gram_upper=ref_Gram_upper, reference_INITIAL_two_current_action_map_norm_upper=ref_source_norm, actual_full_INITIAL_two_current_Gram_upper=full_initial_Gram_upper, actual_full_INITIAL_two_current_Gram_condition_number_upper=full_initial_Gram_upper/full_initial_Gram_lower, alltime_evolved_INITIAL_current_packet_map_error=evolved_initial_current_packet_error, alltime_canonical_current_retarded_matrix_operator_error=retarded_error, alltime_physical_A_current_retarded_matrix_operator_error=a*a*retarded_error, alltime_EXACT_full_INITIAL_Gram_relative_retarded_matrix_error=relative_error)
    return dict(input_descriptor=desc, input_sha256=hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest(), **{key: positive_exact(value) for key, value in values.items()},
        actual_initial_and_evolved_packet_proof='Supplied541dbclosed-loop bound isACTUALreferenceINITIALGram (quantumPH+classicalconditionalmean, all64BEFOREsquaring), NOTPHupperalone/operatornorm/evolvedcurrent. Sref<=sqrtGramUpper; fullsourceinitialY=X+deltaX hasnorm<=Sref+sqrt2b367Jprice andGramupperthatnorm². Referencecurrentpacket Vref(t)=Uhref(t)Jrefchi_SOURCE hasCONSERVEDmapnormSref byunitarity; fieldsORIGINALPhi-sourceunitarity givesexactsamepacketnorm. UniformNEWfield-wavecomparison applieslinearlytoinitialJrefchi withnormSref andreferencefieldPhi source; fullinitialpricepaidseparately. Hence Vfull(t)-Vref(t) maperror<=Sref epsilon+initial_price. ReferenceP_E actsONLYPhi; Jref000 actsONLYTheta/CAR soREFERENCEtensorcommutation exact. FullJ(Phi,U) wasNEVERcommutedthroughfieldP_E andispricedbyNEWglobalweighted/movingPhi-currentactionerror.',
        actual_retarded_consumer='OrderedFij=<JiUfullPsi,UfullJjPsi>. Fullinitialcurrentmapnorm<=Sref+initial_price; alltimeLactionerror fromretardede8 retainsactualfullPhi/globalweighted/conservedreferencefield prices. LREF(t)=JrefUhref(t)chi_SOURCE isNOTevolvedINITIALpacket andNOTnormconserved: retainALLTheta/CARoperatorcapJref here. Orderederror<=Lactionerror*(Sref+initial_price)+Jref_operator_cap*(Sref epsilon+initial_price), thenactualretardederror<=2orderederror. SAMErealphysicalcurrentbasis/exactfullINITIALGramfunctional, relative<=rawerror/g_lower; noGram replacedbyupper/lower. PhysicalcouplerTheta=aA impliesJ_A=aJTheta andrawretardederror*a²; exactGram-relativeerrorunchanged. ALL|t|<=4096, sourceNONstationarity allowed. Pi/ThetaWard bounds frome8 deliberatelyNOTtightened bythissource-onlynorm.',
        review_boundary='NEW executedreceivernormrefinement CONSTRUCTION forparentreview ofSupplier541dbclosed-loop/Poincare sourcebound andownactualmovingPhi/e8 premises. No f9groundcomparator, filteredGaussian, sourceprobabilityretag, contact-to-futureTaylor or sourceJnorm-to-J(t)psi substitution. DoesNOTnumericallyevaluateHref/fullHretardedcurve or proveground/vacuum/stationarity/pole/commoncone/Compton/statP-Q.',
        source_at_import={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes_utf8=data.decode()) for name, data in SOURCE_BYTES.items()}, source_unchanged_since_import={name: (ROOT/name).read_bytes()==data for name, data in SOURCE_BYTES.items()}, runtime=dict(executable=sys.executable, python=sys.version), elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retarded', type=Path, required=True)
    parser.add_argument('--source-norm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.retarded, args.source_norm)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('input_sha256', 'actual_full_INITIAL_two_current_Gram_condition_number_upper', 'alltime_EXACT_full_INITIAL_Gram_relative_retarded_matrix_error', 'elapsed_seconds')}, indent=2))

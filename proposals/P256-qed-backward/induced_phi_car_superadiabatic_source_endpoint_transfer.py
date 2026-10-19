"""Preserve the actual same-H raw source endpoints for the counterrotated packet.

This companion consumes the completed uniform90 physical receipt, not a new
scalar model. It transfers the frozen physical two-leg theorem by unitary
stability; it does not claim a newly improved source-leg dynamical residual.
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
import induced_phi_car_superadiabatic_transfer as transfer

ROOT = Path(__file__).resolve().parent
SELF = Path(__file__).read_bytes()
BASE = ROOT/'induced_phi_full_uniform_colour_fermionic_source_unitary_final_actual_receipt.json'
UNIFORM = ROOT/'induced_phi_car_superadiabatic_weighted_reprice_verified_actual_receipt_20261010.json'

def interval(lo,hi):
    return dict(lower=str(lo),upper=str(hi),encoding='exact rational endpoints')

def run() -> dict[str,Any]:
    started = time.monotonic()
    inputs = {str(p):p.read_bytes() for p in (BASE,UNIFORM)}
    if hashlib.sha256(inputs[str(BASE)]).hexdigest() != transfer.BASE_RECEIPT_SHA:
        raise ArithmeticError('Frozen predecessor physical source receipt required')
    base,uniform = [json.loads(inputs[str(p)]) for p in (BASE,UNIFORM)]
    own = uniform['source_at_import'][Path(transfer.__file__).name]['sha256']
    if own != hashlib.sha256(transfer.SOURCE_BYTES[Path(transfer.__file__).name]).hexdigest():
        raise ArithmeticError('Uniform physical counterrotation receipt/source epoch mismatch')
    if not all(uniform['source_unchanged_since_import'].values()) or not all(uniform['input_unchanged'].values()):
        raise ArithmeticError('Uniform physical counterrotation execution was not input/source bound')
    if not uniform['underlying_state_and_H_unchanged']:
        raise ArithmeticError('Weighted reprice changed the physical source law')
    transfer.iv.dps = 100
    rootp = F(transfer.pack(transfer.iv.sqrt(transfer.exact(F(1,2))))['upper'])
    budget = uniform['actual_new_prepared_state_budget']
    gamma = F(budget['anti_H_generator_on_compact_packet_L2_upper']['upper'])
    delta_source = F(budget['physical_source_endpoint_change_upper']['upper'])
    extra = 2*rootp*gamma+2*delta_source
    endpoints = []
    for row in base['actual_full105_corrected_joint90_quantum_CAR_source_proof']['raw_source_times']:
        t = F(row['time'])
        addition = extra if t else F(0)
        error = F(row['total_same_physical_H_complex_error_upper']['upper'])+addition if t else F(0)
        re,im = row['raw_reference_real'],row['raw_reference_imag']
        lower = max(F(0),F(row['nonzero_modulus_lower']['lower'])-addition)
        probability_upper = min(F(1,2),F(row['raw_compound_return_probability']['upper'])+2*rootp*addition+addition*addition)
        endpoints.append(dict(time=row['time'],raw_reference_real=re,raw_reference_imag=im,
            inherited_same_H_twoleg_complex_error=row['total_same_physical_H_complex_error_upper'],
            actual_counterrotation_endpoint_addition_upper=str(addition),total_same_physical_H_complex_error_upper=str(error),
            raw_actual_same_H_source_real=interval(F(re['lower'])-error,F(re['upper'])+error),
            raw_actual_same_H_source_imag=interval(F(im['lower'])-error,F(im['upper'])+error),
            nonzero_modulus_lower=str(lower),raw_compound_return_probability=interval(lower*lower,probability_upper)))
    if F(endpoints[-1]['nonzero_modulus_lower']) <= 0:
        raise ArithmeticError('Counterrotated actual same-H physical source lost its priced nonzero endpoint')
    descriptor = dict(law='Same original nonlinear uniform105 physical H and raw E0, new exact counterrotated preparation only',
        inputs={p:hashlib.sha256(raw).hexdigest() for p,raw in inputs.items()},uniform_source_sha256=own,
        historical_coefficient_and_point_epoch=uniform['historical_primary'],
        weighting_scope=uniform['Gaussian_weight_proof']['bound_inputs'],
        source='E0=.5 sum4 c_active0alpha^dag Ralpha; raw norm squared1/2; Ref4->3; no whitening',
        particles=8324,domain='Same compact b6/P/f_A physical chart and finite-CAR anti-H multiplication',
        endpoint_derivation='Let delta_v<=||T psi|| and delta_s<=||T E0psi||. Unitarity of the same unrestricted physical H gives |C_new(t)-C_old(t)|<=2 sqrt(1/2) delta_v+2 delta_s using ||E0||<=2. delta_s includes the extra active particle beyond occupied rank128. At t0 the exact preparation norm squared is1/2, C(0)=1/2 and compound return |C(0)|²=1/4, so addition0.',
        probability_derivation='On the unchanged Ref4 vacuum leg ||E0||<=1, and source norm=sqrt(1/2); therefore |C_old|<=sqrt(1/2). |C_new|²<=old upper+2 sqrt(1/2) addition+addition², capped at1/2.',
        state_and_source_endpoint_prices=dict(vacuum_state_difference_upper=str(gamma),physical_source_vector_difference_upper=str(delta_source),all_time_complex_addition_upper=str(extra)),
        limits='Improved vacuum prepared-state residual is separate. These raw source endpoints inherit the preceding nonlinear physical source dynamics; no new source-leg cancellation, nonlinear ground, PH dynamics, EM, common cone or scattering is asserted.')
    captures = dict(transfer.SOURCE_BYTES);captures[Path(__file__).name] = SELF
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        actual_same_H_counterrotated_raw_source_times=endpoints,
        source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(raw).hexdigest(),bytes_utf8=raw.decode()) for name,raw in captures.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==raw for name,raw in captures.items()},
        input_unchanged={p:Path(p).read_bytes()==raw for p,raw in inputs.items()},
        runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-started)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = run()
    if not all(result['source_unchanged_since_import'].values()) or not all(result['input_unchanged'].values()):
        raise ArithmeticError('Source endpoint consumer epoch changed during execution')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['actual_same_H_counterrotated_raw_source_times'],indent=2))

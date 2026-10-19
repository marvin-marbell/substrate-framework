#!/usr/bin/env python3
"""Literal Dhalf G Dhalf joint-reference contraction, not interacting admission.

Every photon/normal/toron heat factor is normalized to its reference neutral
maximum. All large-gauge/Floquet seams remain unitary. The theta-dependent
canonical SF factors retain their actual eigenvalues and common neutral scale.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any
import mpmath as mp
import covariant_physical_mf_vacuum_winding as meanfield
import covariant_actual_source_band_consumer as consumer
from covariant_induced_photon import Parameters

ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
if sys._getframe().f_code!=compile(OWN_BYTES,__file__,'exec',dont_inherit=True,optimize=sys.flags.optimize):
    raise ArithmeticError('Upper-strip consumer differs from compiled snapshot')
EXPECTED_MF='5536f17f9c2c7fda17592cb3c99e914749d14349e5315dee5de5ec18d056d002'
if hashlib.sha256(meanfield.OWN_BYTES).hexdigest()!=EXPECTED_MF:
    raise ArithmeticError('Actual common-neutral MF proof epoch changed')


def upper_strip_exclusion(a: float,mass: float,p: Parameters,eta: float) -> dict[str,Any]:
    """Operator-norm product; no theta commutation or source density assumed."""
    budget=meanfield.reference_budget(a,mass,p.spatial_size,p.time_slices)
    with mp.workdps(80):
        am=mp.mpf(str(a))*mp.mpf(str(mass))
        eps=mp.mpf(budget['winding']['log_vacuum_twist_error_upper'])
        # Two SF legs: their common-neutral norms are <=exp(-am/2+eps).
        # Conservative upward guard dominates 80-digit arithmetic rounding.
        norm=mp.exp(-am+2*eps)+mp.mpf('1e-60')
        threshold=mp.exp(-mp.mpf(str(eta)))
        if not 0<eta or not norm<threshold:
            raise ArithmeticError('Proof halfwidth does not exclude upper Pe strip')
        return dict(a=a,m=mass,eta=eta,actual_within_band_theta_dependence_retained=True,
            single_SF_charged_common_neutral_norm_upper=budget['SF_charged_common_neutral_norm_upper'],
            actual_vacuum_twist_log_error_upper=mp.nstr(eps,70),
            literal_joint_reference_charged_norm_upper=mp.nstr(norm,70),
            reference_star_upper_strip_lambda_lower=mp.nstr(threshold,70),
            proved_positive_reference_star_spectral_distance=mp.nstr(threshold-norm,70),
            reference_star_upper_threshold_strip_source_squared_norm=0,
            reference_star_above_one_source_squared_norm=0,
            original_Pe_upper_threshold_strip_source_squared_norm=None,
            sufficient_actual_neutral_lambda_over_reference_star_lower=mp.nstr(norm/threshold,70),
            source_independent=True,charged_own_maximum_used=False,
            proof='For the LITERAL reference F(theta_prime) G F(theta), every theta charged SF factor obeys norm<=exp(-am/2+epsilon_winding) relative to ONE neutral Wsf(0). G retains the joint photon/normal/toron heat, physical Gauss compression and all unitary large-gauge/Floquet seams; its normalized norm is<=1. Thus ||Kref_ch/reference_star||<=sup_theta||F||^2<=exp(-am+2epsilon_winding), even though F and G do not commute. Here reference_star=Wsf(0)^2 times the normalized positive heat maximum, NOT an own-charged maximum or silently the actual neutral Root lambda0. Its source spectral measure of [exp(-eta),exp(eta)] and (1,infinity) is zero. For ORIGINAL Pe thresholds a separate actual lambda0/reference_star lower bound exceeding the displayed sufficient value is required. No thermal Slater factorization, independent character positivity, smooth density, Berry cancellation, global Tmin or scalar-to-Loewner lift.',
            scope='MEAN reference operator contraction in its ONE neutral scalar reference_star scale only. Actual Root all-species comparison, actual neutral normalization lower, and lower Pe energy-threshold strip at .25 are not supplied.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    input_path=ROOT/'covariant_prepared_gauss_state_input_kappa400_sigma2000_r2100.json'
    p=Parameters(**json.loads(input_path.read_text())['photon_parameters'])
    rows=[upper_strip_exclusion(a,m,p,eta) for a,m,eta in itertools.product(p.spacings,p.masses,(1e-3,1e-5,1e-7,1e-9,1e-11))]
    result=dict(governing_plan_revision=consumer.PLAN_REVISION,source_sha256=hashlib.sha256(OWN_BYTES).hexdigest(),
        actual_imported_MF_proof_sha256=hashlib.sha256(meanfield.OWN_BYTES).hexdigest(),
        original_input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        all9_common_neutral_reference_upper_strip_exclusions=rows,
        original_source_Pe_preparation_clocks_unchanged=True,
        reference_lower_energy_threshold_point25_joint_source_strip_bound=None,
        actual_macro_source_Gram=None,macro_source_admitted=False,
        elapsed_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),cases=len(rows),elapsed_seconds=result['elapsed_seconds'],
        reference_star_upper_threshold_strip_source_squared_norm=0,
        reference_lower_energy_threshold_point25_joint_source_strip_bound=None,macro_source_admitted=False)))


if __name__=='__main__':main()

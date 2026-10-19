#!/usr/bin/env python3
"""Explicit different full-field family with uniformly bounded separated trial costs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from full_band_gauss_section import FullBandGaussSection
from pair_full_field_concentration import constants, ground_moment_bound
from pair_separated_charge_transfer import CF, LOCAL_LIGHT_CASIMIR_MAX, source_hashes as transfer_hashes
from rotation_band import RotationBand
from unrestricted_field_minimum import ALIGNMENT, BARE, STIFFNESS, pair_clifford_reference

ROOT=Path(__file__).resolve().parent


def source_hashes():
    return {**transfer_hashes(), **{
        name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('pair_separated_charge_scaling.py','pair_separated_charge_scaling.md')}}


def native_control_binding(path: Path):
    payload=path.read_bytes()
    receipt=json.loads(payload)
    if not receipt['all_checks_pass'] or not receipt['source_bindings_unchanged']:
        raise ValueError('Completed native full-field separated transfer receipt required')
    for name, expected in receipt['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Native transfer source changed: {name}')
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(payload).hexdigest(),
        purpose='Native full-fibre Gauss and kinetic product rules at noncentral/all120 inputs; coefficient scaling uses their exact algebra, not a newly sampled quantum ground')


def positive_kinetic_bound(Lambda: float, law: dict[str,Any]):
    # Do not subtract nearly equal O(1) scaled energies: their small excess
    # is lost at large Lambda. Every summand here is nonnegative.
    alpha=(2*law['A_field_fisher']/law['leading_slope'])**(1/3)
    support=min(law['chart_radius'],alpha*Lambda**(-1/6))
    return (Lambda*law['leading_slope']*support
        +math.sqrt(Lambda)*(law['A_field_fisher']/support**2+law['A_field_metric'])
        +law['B_electric_fisher']/support**2+law['B_electric_metric']
        +law['light_anchor_lower_cost']+law['magnetic_lower_cost']
        +law['light_Young_coefficient']*Lambda**(-1/3))


def run(*,Lambdas,exponent=11/12,ue4_anchor=.3,ue1=.4,magnetic=.7,
        native_receipt=ROOT/'pair_separated_charge_transfer_receipt.json') -> dict[str,Any]:
    if not Lambdas or any(not math.isfinite(L) or L<1 for L in Lambdas):
        raise ValueError('Finite Lambda>=1 required')
    if not 5/6<exponent<1:
        raise ValueError('This decaying-cross/positive-alignment-frequency family requires5/6<p<1')
    if not all(math.isfinite(x) for x in (ue4_anchor,ue1,magnetic)) or min(ue4_anchor,ue1)<=0 or magnetic<0:
        raise ValueError('Positive electric anchors and nonnegative magnetic coefficient required')
    identities=source_hashes()
    binding=native_control_binding(native_receipt)
    model=FullBandGaussSection(RotationBand(**BARE),heavy_scale=1.)
    frame,_=pair_clifford_reference(model,stiffness=STIFFNESS)
    fields=np.tile(frame,(2,1,1,1))
    # This fixed-anchor packet is an upper bound for EVERY family member:
    # its positive colour kinetic coefficient can only decrease.
    law=constants(model,fields,stiffness=STIFFNESS,alignment=ALIGNMENT,
        ue4=ue4_anchor,ue1=ue1,magnetic=magnetic,chart_radius=.001)
    alpha=(2*law['A_field_fisher']/law['leading_slope'])**(1/3)
    limiting_kinetic_prefactor=law['leading_slope']*alpha+law['A_field_fisher']/alpha**2
    # A single finite upper constant valid also in the clamped packet regime.
    uniform_kinetic_prefactor=(limiting_kinetic_prefactor
        +law['A_field_fisher']/law['chart_radius']**2+law['A_field_metric']
        +law['B_electric_fisher']/law['chart_radius']**2+law['B_electric_fisher']/alpha**2
        +law['B_electric_metric']+law['light_anchor_lower_cost']
        +law['magnetic_lower_cost']+law['light_Young_coefficient'])
    rows=[]
    for Lambda in Lambdas:
        ue4=ue4_anchor*Lambda**(-exponent)
        moment=ground_moment_bound(Lambda,law,stiffness=STIFFNESS,alignment=ALIGNMENT)
        mean=np.asarray(law['optimal_average_norms'])+math.sqrt(moment['expected_average_norm_deviation_squared_upper'])
        light=float((law['light_u_coefficient']*np.sqrt(mean[0])+law['light_v_coefficient']*np.sqrt(mean[1]))/4)
        kinetic=positive_kinetic_bound(Lambda,law)
        cross=math.sqrt(LOCAL_LIGHT_CASIMIR_MAX*ue4*kinetic)
        uniform_cross=math.sqrt(LOCAL_LIGHT_CASIMIR_MAX*ue4_anchor*uniform_kinetic_prefactor)*Lambda**(5/12-exponent/2)
        if not ue4>0 or cross>uniform_cross*(1+1e-12):
            raise AssertionError('Positive full-domain coefficient or analytic uniform envelope failed')
        rows.append(dict(Lambda=Lambda,actual_ue4=ue4,neutral_total_kinetic_upper=kinetic,
            neutral_remainder_upper=moment['expected_global_remainder_upper'],light_cost_upper=light,
            colour_cross_absolute_upper=cross,uniform_colour_cross_envelope=uniform_cross,
            separated_ground_minus_neutral_ground_upper=light+ue4*CF+ue1/2+cross,
            fixed_ue4_bound_control=math.sqrt(LOCAL_LIGHT_CASIMIR_MAX*ue4_anchor*kinetic),
            scalar_comparison_loses_common_field_section_kinetic_scale=math.sqrt(Lambda)))
    input_unchanged=hashlib.sha256(native_receipt.read_bytes()).hexdigest()==binding['sha256']
    if source_hashes()!=identities or not input_unchanged:
        raise RuntimeError('Source or native control receipt changed during execution')
    return dict(governing_plan='issue-222-continuing-backward-20260930',work_item='campaign',
        source_sha256=identities,source_bindings_unchanged=True,native_transfer_binding=binding,
        native_input_unchanged=True,all_checks_pass=True,campaign_complete=False,
        family='EXPLICIT different bare trajectory ue4(Lambda)=ue4_anchor Lambda^-p; all finite-Lambda Cartesian/Haar/torus kinetics and native fibres retained',
        parameters=dict(exponent=exponent,ue4_anchor=ue4_anchor,ue1=ue1,magnetic=magnetic,
            bare=BARE,stiffness=STIFFNESS,alignment=ALIGNMENT,field_kinetic_coefficient='sqrt(Lambda)',heavy_scale='Lambda'),
        packet_law_upper=law,rows=rows,
        kinetic_asymptotic_prefactor=limiting_kinetic_prefactor,uniform_kinetic_prefactor=uniform_kinetic_prefactor,
        colour_cross_power=5/12-exponent/2,
        frozen_field_link_harmonic_frequency_power=(1-exponent)/2,
        asymptotic_separated_cost_upper=float(law['light_anchor_lower_cost']/4+ue1/2),
        limits=['This is a declared different colour-electric bare trajectory, not a uniform bound at the original fixedue4.',
            'Finite closed graph compensated L insertion/R reference; not isolated or infinite-volume/Coulomb charge.',
            'Variational upper bound on actual quantum sector infima, not a positive lower gap or measured ground.',
            'Frequency power is for an isolated positively curved link subblock with fields frozen, not the full mixed Gauss-normal spectrum; flat/stabilizer gaps are unproved.',
            'Fixed a1 pair/photon inputs remain; this is not the bulk cutoff-matched/continuum or nonlinear compact-current construction.',
            'No mobile physical spin sector, thermodynamic phase, radiative amplitude or joined campaign completion.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--Lambda',type=float,nargs='+',default=[1e20,1e24,1e30,1e36,1e48,1e72,1e120])
    parser.add_argument('--exponent',type=float,default=11/12)
    parser.add_argument('--native-receipt',type=Path,default=ROOT/'pair_separated_charge_transfer_receipt.json')
    parser.add_argument('--output',type=Path,default=ROOT/'pair_separated_charge_scaling_receipt.json')
    args=parser.parse_args()
    result=run(Lambdas=args.Lambda,exponent=args.exponent,native_receipt=args.native_receipt)
    payload=json.dumps(result,indent=2,allow_nan=False)
    args.output.write_text(payload+'\n')
    print(payload)


if __name__=='__main__':
    main()

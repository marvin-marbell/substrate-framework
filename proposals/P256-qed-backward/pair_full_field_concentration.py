#!/usr/bin/env python3
"""Full physical neutral pair: global field moment bounds, not charged gaps.

Uses the exact global pair coercive remainder and an actual all120/Haar/T2
Gauss packet. No retained light-rank cut or finite angular Fourier truncation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
        os.environ[variable]='1'

import numpy as np
from scipy.linalg import expm

from full_band_gauss_section import FullBandGaussSection
from full_band_packet_bound import packet_bound, packet_point_controls
from pair_global_field_bound import coefficients, point, source_hashes as global_source_hashes
from rotation_band import RotationBand
from unrestricted_field_minimum import ALIGNMENT, BARE, STIFFNESS, matrices, pair_clifford_reference

ROOT=Path(__file__).resolve().parent


def source_hashes():
    return {**global_source_hashes(),**{
        name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('pair_full_field_concentration.py','pair_full_field_concentration.md')}}


def constants(model, fields, *, stiffness, alignment, ue4, ue1, magnetic, chart_radius) -> dict[str, Any]:
    c,_,t=coefficients(model)
    radius=((c**4+t**4)/stiffness)**(1/3)
    optimal=np.array([c*c,t*t])/(stiffness*radius)
    packet=packet_bound(model,fields,np.eye(4),(0.,0.),kappa=stiffness,J=alignment,
        ue4=ue4,ue1=ue1,magnetic=magnetic,R=chart_radius,eta=chart_radius,tau=chart_radius)
    grams=np.einsum('xmij,xnji->xmn',fields,fields).real/4
    gram_norm=float(np.linalg.norm(grams))
    rho=float(np.linalg.norm(fields)/2)
    field_slope=packet['field_onebody_derivative_bound']
    delta_slope=field_slope+2*packet['endpoint_HS_bound']
    align_slope=2*np.sqrt(2)+2*float(np.linalg.norm(fields[1]))
    r=chart_radius
    leading_slope=float(stiffness/2*(4*gram_norm*rho+(2*gram_norm+4*rho*rho)*r
        +4*rho*r*r+r**3)+alignment/8*align_slope**2*r+4*delta_slope)
    metrics=packet['section_metric_upper']
    field_fisher=2560.
    field_metric=float(metrics['fields']/2)
    electric_fisher=float(ue4*packet['envelope_fisher']['colour']*r*r
        +ue1/2*packet['envelope_fisher']['phases']*r*r)
    light_upper=packet['anchor']['light_occupied_energy']+4*packet['onebody_uniform_change']
    electric_metric=float(ue4*metrics['colour']+ue1/2*metrics['phases']+magnetic+max(0.,light_upper))
    cl=float(abs(model.onsite[1])); dl,tl=float(abs(model.paths[1][0])),float(abs(model.paths[1][1]))
    light_u,light_v=8*np.sqrt(2)*cl+8*dl,8*tl
    light_anchor=float(light_u*np.sqrt(optimal[0])+light_v*np.sqrt(optimal[1]))
    young_coefficient=float(3/4*(light_u**(4/3)+light_v**(4/3))/stiffness**(1/3))
    return dict(global_minimum=-3*radius,optimal_average_norms=optimal.tolist(),
        chart_radius=r,leading_slope=leading_slope,
        A_field_fisher=field_fisher,A_field_metric=field_metric,
        B_electric_fisher=electric_fisher,B_electric_metric=electric_metric,
        light_u_coefficient=float(light_u),light_v_coefficient=float(light_v),
        light_anchor_lower_cost=light_anchor,light_Young_coefficient=young_coefficient,
        magnetic_lower_cost=magnetic,chart_packet=packet)


def ground_moment_bound(Lambda: float, law: dict[str,Any], *, stiffness, alignment):
    if not np.isfinite(Lambda) or Lambda<1:
        raise ValueError('Finite Lambda>=1 required')
    alpha=(2*law['A_field_fisher']/law['leading_slope'])**(1/3)
    support=float(min(law['chart_radius'],alpha*Lambda**(-1/6)))
    scaled_trial_excess=(law['leading_slope']*support
        +(law['A_field_fisher']/support**2+law['A_field_metric'])/np.sqrt(Lambda)
        +(law['B_electric_fisher']/support**2+law['B_electric_metric'])/Lambda)
    scaled_lower_cost=(law['light_anchor_lower_cost']+law['magnetic_lower_cost'])/Lambda
    scaled_lower_cost+=law['light_Young_coefficient']*Lambda**(-4/3)
    remainder=float(4/3*(scaled_trial_excess+scaled_lower_cost))
    return dict(Lambda=Lambda,physical_trial_support_radius=support,
        scaled_trial_ground_energy_upper=law['global_minimum']+float(scaled_trial_excess),
        expected_global_remainder_upper=remainder,
        expected_average_norm_deviation_squared_upper=remainder/stiffness,
        expected_endpoint_norm_spread_squared_upper=2*remainder/stiffness,
        expected_sum_unused_normalized_field_fourth_upper=2*remainder/stiffness,
        expected_sum_unused_normalized_field_squared_upper=float(np.sqrt(8*remainder/stiffness)),
        expected_offdiagonal_Gram_squared_sum_upper=2*remainder/stiffness,
        expected_KyFan_slack_upper=remainder,
        expected_full_W_upper=8*remainder/alignment if alignment>0 else None,
        ground_energy_lower_excess_divided_by_Lambda=-float(scaled_lower_cost),
        occupation='NH4 NL4, all70 light and70 heavy Pauli configurations retained',
        numerical_status='floating evaluation of derived analytic bounds, not interval certification or sampled quantum ground')


def run(*, Lambdas, stiffness=STIFFNESS, alignment=ALIGNMENT, ue4=.3, ue1=.4, magnetic=.7, chart_radius=.001):
    if not all(np.isfinite(v) for v in (stiffness,alignment,ue4,ue1,magnetic,chart_radius)):
        raise ValueError('Finite physical law coefficients required')
    if min(stiffness,ue4,ue1,chart_radius)<=0 or min(alignment,magnetic)<0 or chart_radius>=.1:
        raise ValueError('Positive stiffness/electrics/radius<.1 and nonnegative J/K required')
    if not Lambdas:
        raise ValueError('At least one physical Lambda required')
    identities=source_hashes()
    model=FullBandGaussSection(RotationBand(**BARE),heavy_scale=1.)
    frame,reference=pair_clifford_reference(model,stiffness=stiffness)
    fields=np.tile(frame,(2,1,1,1))
    law=constants(model,fields,stiffness=stiffness,alignment=alignment,
        ue4=ue4,ue1=ue1,magnetic=magnetic,chart_radius=chart_radius)
    uniform_controls=[]
    for fraction in (1.,.5,.1):
        r=chart_radius*fraction
        actual=packet_bound(model,fields,np.eye(4),(0.,0.),kappa=stiffness,J=alignment,
            ue4=ue4,ue1=ue1,magnetic=magnetic,R=r,eta=r,tau=r)
        upper_C=law['global_minimum']+law['leading_slope']*r
        upper_T=law['A_field_fisher']/r**2+law['A_field_metric']
        upper_B=law['B_electric_fisher']/r**2+law['B_electric_metric']
        residuals=[actual['C_Lambda']-upper_C,actual['T_sqrt_Lambda']-upper_T,
            actual['B_constant']-upper_B]
        if any(error>2e-12*max(1.,abs(upper)) for error,upper in zip(residuals,(upper_C,upper_T,upper_B))):
            raise AssertionError('Shrinking physical full-domain trial bound failed')
        uniform_controls.append(dict(radius=r,actual_C=actual['C_Lambda'],upper_C=upper_C,
            actual_T=actual['T_sqrt_Lambda'],upper_T=upper_T,actual_B=actual['B_constant'],upper_B=upper_B))
    rng=np.random.default_rng(222121)
    operator_controls=[]
    for number in range(24):
        value=rng.normal(size=(2,4,15))*(.02,.15,.7)[number%3]
        arbitrary=matrices(value,model.basis)
        link=expm(1j*np.einsum('a,aij->ij',rng.normal(size=15)*.3,model.generators))
        phase=rng.uniform(-np.pi,np.pi,size=2)
        heavy=point(model,arbitrary,link,stiffness=stiffness,alignment=alignment)
        light=float(np.linalg.eigvalsh(model.onebody(arbitrary,link,phase)[1])[:4].sum())
        for scale in (1.,1e4,1e12):
            actual=scale*heavy['actual_leading_coefficient']+light-magnetic*np.cos(float(phase.sum()))
            lower=(scale*law['global_minimum']+.75*scale*heavy['excess']
                -law['light_anchor_lower_cost']-magnetic-law['light_Young_coefficient']*scale**(-1/3))
            if lower-actual>1e-10*max(1.,abs(actual)):
                raise AssertionError('Physical full-field/light/photon pointwise lower bound failed')
            operator_controls.append(dict(Lambda=scale,actual_lowest_matter_plus_magnetic=actual,
                coercive_lower=lower,margin=actual-lower))
    # Omitting the light sea is an actual false pointwise lower bound, even
    # at the exact global heavy/field minimizer. Preserve this adverse control.
    light_at_min=float(np.linalg.eigvalsh(model.onebody(fields,np.eye(4),(0.,0.))[1])[:4].sum())
    if light_at_min>=-1e-5:
        raise AssertionError('Deleted light-sea lower-cost control failed')
    rows=[ground_moment_bound(float(L),law,stiffness=stiffness,alignment=alignment) for L in Lambdas]
    primitive=packet_point_controls(model,fields,np.eye(4),(0.,0.))
    if source_hashes()!=identities:
        raise RuntimeError('Physical source/proof changed during concentration execution')
    return dict(source_sha256=identities,source_bindings_unchanged=True,
        governing_plan='issue-222-continuing-backward-20260930',work_item='campaign',
        bare_coefficients={**BARE,'stiffness':stiffness,'alignment':alignment,'ue4':ue4,'ue1':ue1,'magnetic':magnetic},
        exact_domain='physical neutral fullL2(R120 x SU4 x T2) x4900 Pauli matter, NH4 NL4, all colour/EM Gauss; no angular or light cut',
        Hamiltonian='sqrtLambda/2 p_field² + ue4 C_colour + ue1/2 p_phase² - K cos(A0+A1) + Lambda(Vquartic+J W/8-dGamma(hH)) + dGamma(hL)',
        reference=reference,uniform_analytic_constants=law,shrinking_trial_controls=uniform_controls,
        actual_pointwise_operator_controls=operator_controls,deleted_light_sea_violation=-light_at_min,
        actual_primitive_Gauss_anchor=primitive,rows=rows,all_checks_pass=True,
        theorem='For any exact physical neutral ground, expectation of the explicit global coercive remainder tends to0 at leastO(Lambda^-1/6); full unused-field fourth moments vanish too.',
        proof_role='Global operator inequality plus actual full-coordinate Gauss variational family; no numerical quantum eigenstate observed',
        limits=['Finite one-axis pair, not three-axis bulk/thermodynamic phase.',
            'Invariant moment concentration, not a solved moving normal bundle or Berry/measure BO.',
            'Not charged B4/bar4 lower gap, physical mechanical spin, photon phase, Lorentz or full electron-photon completion.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--Lambda',type=float,nargs='+',default=[1e20,1e24,1e30,1e36,1e48])
    parser.add_argument('--stiffness',type=float,default=STIFFNESS)
    parser.add_argument('--alignment',type=float,default=ALIGNMENT)
    parser.add_argument('--ue4',type=float,default=.3)
    parser.add_argument('--ue1',type=float,default=.4)
    parser.add_argument('--magnetic',type=float,default=.7)
    parser.add_argument('--chart-radius',type=float,default=.001)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(Lambdas=args.Lambda,stiffness=args.stiffness,alignment=args.alignment,
        ue4=args.ue4,ue1=args.ue1,magnetic=args.magnetic,chart_radius=args.chart_radius)
    text=json.dumps(result,indent=2,allow_nan=False)
    args.output.write_text(text+'\n'); print(text)


if __name__=='__main__':
    main()

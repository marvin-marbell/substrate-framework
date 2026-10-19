"""Native tube perturbation prices; quantum-source normalization/jets remain premises."""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, cast

import sympy as sp

ROOT = Path(__file__).resolve().parent


def main() -> None:
    sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, expected in request['pins'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    profile = json.loads((ROOT/request['profiles']).read_text())['primitive_profiles'][0]
    m, c0, ci = Q(profile['onsite']), Q(profile['paths'][0]['c0']), Q(profile['paths'][0]['ci'])
    s, r = Q(1,2), Q(1,3)
    neutral_error = 4*m+12*(abs(c0)*(1+s)+abs(ci)*(1+r))
    # HS(Phi_mu-barPhi_mu)<=delta and HS(V-I)<=delta imply
    # delta{Phi_mu,Phi_nu} HS<=4delta for delta<=1. With16 ordered
    # pairs, J=4 sum HS(anticommutator)^2 changes by at most1536delta.
    assert neutral_error+1536 < 2048
    # Each alignment endpoint difference HS<=4delta:12 per site.
    alignment_constant = Q(12*16,40)
    # SU4 face:4-ReTr U=.5 HS(U-I)^2<=8delta^2;3 per site.
    colour_constant = Q(7,10)*3*8
    assert alignment_constant+colour_constant < 32
    sx, sy, sz = sp.Matrix([[0,1],[1,0]]), sp.Matrix([[0,-sp.I],[sp.I,0]]), sp.diag(1,-1)
    beta = cast(sp.Matrix, sp.kronecker_product(sz,sp.eye(2)))
    alpha = [cast(sp.Matrix, sp.kronecker_product(sx,a)) for a in (sx,sy,sz)]
    point = [beta/2]+[a/3 for a in alpha]
    perturb = sp.diag(3,-1,-1,-1)/640
    left, right = [p+perturb for p in point], [p-perturb for p in point]
    delta = sp.Rational(1,10)
    t1,t2 = sp.zeros(4),sp.zeros(4)
    t1[0,2]=t1[2,0]=1;t2[0,3]=t2[3,0]=1
    def link(t: sp.Matrix) -> sp.Matrix:
        return (sp.eye(4)+sp.I*t/1000)*(sp.eye(4)-sp.I*t/1000).inv()
    u,v = link(t1),link(t2)
    assert u.H*u==sp.eye(4) and u.det()==1
    assert sp.trace((u-sp.eye(4)).H*(u-sp.eye(4))) <= delta**2
    def jordan(fields: list[sp.Matrix]) -> Any:
        return 4*sum(sp.trace((a*b+b*a)**2) for a in fields for b in fields)
    assert jordan(point)==sp.Rational(172,27)
    assert abs(jordan(left)-jordan(point))<=1536*delta
    alignment=sum(sp.trace((a-u*b*u.H).H*(a-u*b*u.H)) for a,b in zip(left,right))
    assert alignment<=64*delta**2
    face=u*v*u.H*v.H
    contact=4-sp.re(sp.trace(face))
    assert 0<contact<=8*delta**2
    # An actual constructed source must supply the total per-site Cartesian
    # plus Haar gradient cap; this consumer does not infer it from width.
    width2=Q(1,2**7168)
    source_gradient_constant=Q(2**1024)
    kinetic_per_Lambda=4*Q(1,2**4096)*source_gradient_constant/width2/Q(2**8192)
    field_per_Lambda=Q(2**11,2**3584)
    alignment_colour_per_Lambda=32*Q(2**10240)*width2/Q(2**8192)
    electric_per_Lambda=Q(262144,2**8192)
    correction=kinetic_per_Lambda+field_per_Lambda+alignment_colour_per_Lambda+electric_per_Lambda
    assert correction<Q(1,4)
    point_upper=Q(json.loads((ROOT/request['point_receipt']).read_text())['sea_plus_J_per_Lambda_site_upper_exact'])
    assert point_upper+correction < -21
    result={'kind':'EXACT_NATIVE_LOCAL_TUBE_BODY_PRICES_CONDITIONAL_ON_QUANTUM_SOURCE_NOT_GROUND',
            'Phi_and_relative_link_HS_radius':'delta<=1','neutral_error_constant_exact':str(neutral_error),
            'Jordan_error_constant':1536,'combined_linear_Lambda_constant_upper':2048,
            'alignment_colour_quadratic_F_constant_upper':32,
            'native_noncommuting_alignment_and_colour_controls':True,
            'conditional_source_gradient_premise':'total Cartesian plus Haar gradient norm2 perV <=2^1024/delta^2',
            'width':'delta=2^-3584','complete_charged_mask_Ref_sharedE_premise':'same full/empty charged parity masks and Ref/sewn source as paid polynomial trial; full sharedE262144/site',
            'conditional_total_correction_per_Lambda_upper_exact':str(correction),
            'conditional_energy_per_Lambda_site_upper_exact':str(point_upper+correction),
            'source_Gauss_normalizer_gradient_premises_established_by_this_consumer':False,
            'actual_ground_body_or_phase_proved':False,'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Native local body prices verified; quantum source/jets still required before energy admission')


if __name__ == '__main__':
    main()

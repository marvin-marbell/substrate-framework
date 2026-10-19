"""Exact all-field Jordan midpoint countercontrol, not a quantum ground counterexample."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import sympy as sp

ROOT=Path(__file__).resolve().parent


def basis() -> list[sp.Matrix]:
    result: list[sp.Matrix]=[]
    for a in range(4):
        for b in range(a+1,4):
            x,y=sp.zeros(4),sp.zeros(4)
            x[a,b]=x[b,a]=1
            y[a,b]=-sp.I;y[b,a]=sp.I
            result.extend([x,y])
    for a in range(1,4):
        x=sp.zeros(4)
        for b in range(a):x[b,b]=1
        x[a,a]=-a
        result.append(x)
    return result


def coefficients(a: list[sp.Matrix],b: list[sp.Matrix]) -> tuple[Any,Any,Any]:
    j0,j2,j4=sp.S.Zero,sp.S.Zero,sp.S.Zero
    for mu in range(4):
        for nu in range(4):
            aa=a[mu]*a[nu]+a[nu]*a[mu]
            d=b[mu]*a[nu]+a[mu]*b[nu]+b[nu]*a[mu]+a[nu]*b[mu]
            bb=b[mu]*b[nu]+b[nu]*b[mu]
            j0+=4*sp.trace(aa**2)
            j2+=4*sp.trace(d**2+2*aa*bb)
            j4+=4*sp.trace(bb**2)
    return (sp.simplify(j0),sp.simplify(j2),sp.simplify(j4))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    for name,expected in request['pins'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
    ts=basis()
    def fields(rows: list[list[int]],scale: int) -> list[sp.Matrix]:
        return [cast(sp.Matrix,sum((q*t/scale for q,t in zip(row,ts)),sp.zeros(4))) for row in rows]
    a,b=fields(request['A_coefficients'],32),fields(request['B_coefficients'],512)
    for x in a+b:assert x.H==x and sp.trace(x)==0
    j0,j2,j4=coefficients(a,b)
    curvature=2*j2
    assert curvature<0 and curvature*32**2*512**2==-925972384
    epsilon=sp.Rational(1,64)
    midpoint_defect=sp.simplify(epsilon**2*j2+epsilon**4*j4)
    assert midpoint_defect<0
    # Positive control on this identical16-pair/trace/factor4 surface.
    commuting=[sp.diag(1,-1,0,0)]+[sp.zeros(4) for _ in range(3)]
    _,positive_j2,_=coefficients(commuting,commuting)
    assert positive_j2>0
    result={'kind':'EXACT_NATIVE_ORDERED_JORDAN_NONCONVEX_MIDPOINT_POINT_CONTROL_NOT_GROUND',
            'Cart_scope':'four arbitrary traceless Hermitian4 matrices, entire60-real-component point space',
            'J':'4 sum_all16_ordered_mu_nu Tr({Phi_mu,Phi_nu}^2)',
            'midpoint_J_exact':str(j0),'second_directional_derivative_exact':str(curvature),
            'epsilon':str(epsilon),'mean_endpoint_J_minus_midpoint_J_exact':str(midpoint_defect),
            'global_Jensen_mean_lower_for_arbitrary_fields_rejected':True,
            'commuting_positive_curvature_control':str(2*positive_j2),
            'source_state_ground_measure_or_ground_failure_claimed':False,
            'local_Lipschitz_or_good_event_comparison_disproved':False,
            'phase_or_atom_or_law_adoption':False,'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Exact negative Jordan midpoint gap verified; all-field Jensen shortcut invalid, not ground failure')


if __name__=='__main__':
    main()

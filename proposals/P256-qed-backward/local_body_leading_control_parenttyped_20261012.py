"""New exact native localbox and reduced-density IMS controls, not a ground solver."""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import sympy as sp
from typing import cast
from sympy.matrices.matrixbase import MatrixBase


def q(value: str) -> sp.Rational:
    f = Q(value)
    return cast(sp.Rational, sp.Rational(f.numerator, f.denominator))


def hs2(a: MatrixBase) -> sp.Expr:
    return sp.simplify(sp.trace(a * a.conjugate().T))


def native(fields: list[list[MatrixBase]], edges: list[tuple[int, int, int, sp.Matrix, int]], m: sp.Expr, c0: sp.Expr, ci: sp.Expr) -> MatrixBase:
    h = sp.zeros(4 * len(fields))
    for x, p in enumerate(fields):
        h[4*x:4*x+4, 4*x:4*x+4] = m*p[0]
    for x, y, axis, u, ap in edges:
        f = ap*(c0*(fields[x][0]*u+u*fields[y][0])+sp.I*ci*(fields[x][axis+1]*u+u*fields[y][axis+1]))/2
        h[4*x:4*x+4, 4*y:4*y+4] += f
        h[4*y:4*y+4, 4*x:4*x+4] += f.conjugate().T
    return h


def jordan(p: list[MatrixBase]) -> sp.Expr:
    return sp.expand(4*sum(sp.trace((a*b+b*a)**2) for a in p for b in p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    root = args.request.parent
    for name, expected in request['pins'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    profiles = json.loads((root/request['profiles']).read_text())['primitive_profiles']
    p = profiles[0]
    m, c0, ci = q(p['onsite']), q(p['paths'][0]['c0']), q(p['paths'][0]['ci'])
    b = abs(c0)+abs(ci)
    d = 2*m+12*b
    eye, zero = sp.eye(4), sp.zeros(4)
    sx, sy, sz = sp.Matrix([[0,1],[1,0]]), sp.Matrix([[0,-sp.I],[sp.I,0]]), sp.diag(1,-1)
    beta = sp.kronecker_product(sz, sp.eye(2))
    alpha = sp.kronecker_product(sx, sx)
    alpha2 = sp.kronecker_product(sx, sy)
    alpha3 = sp.kronecker_product(sx, sz)
    body = [cast(MatrixBase, beta/2), cast(MatrixBase, alpha/3), cast(MatrixBase, alpha2/3), cast(MatrixBase, alpha3/3)]
    assert jordan(body) == sp.Rational(172,27)
    commuting = [cast(MatrixBase, beta/2), cast(MatrixBase, beta/3), cast(MatrixBase, alpha2/3), cast(MatrixBase, alpha3/3)]
    cross_excess = sp.simplify(jordan(commuting)-jordan(body))
    assert cross_excess > 0
    # Rational SU4 matrices with genuinely noncommuting face transports.
    u = (3*eye+4*sp.I*beta)/5
    v = (5*eye+12*sp.I*alpha)/13
    assert sp.simplify(u.det()) == sp.simplify(v.det()) == 1
    assert (u*u.conjugate().T).applyfunc(sp.simplify) == (v*v.conjugate().T).applyfunc(sp.simplify) == eye
    face = u*v*u.conjugate().T*v.conjugate().T
    curvature = sp.simplify(4-sp.re(sp.trace(face)))
    assert curvature > 0 and hs2(face-eye) == 2*curvature
    # The actual endpoint-averaged native square, including an AP edge.
    gauges = [eye, u, v, u*v]
    pairs = [(0,1,0,1),(0,2,1,1),(1,3,1,1),(2,3,0,-1)]
    transformed = [[g*a*g.conjugate().T for a in body] for g in gauges]
    pure_edges = [(x,y,a,gauges[x]*gauges[y].conjugate().T,ap) for x,y,a,ap in pairs]
    flat_edges = [(x,y,a,eye,ap) for x,y,a,ap in pairs]
    hflat = native([body]*4, flat_edges, m,c0,ci)
    hpure = native(transformed,pure_edges,m,c0,ci)
    gauge = sp.diag(*gauges)
    assert (hpure-gauge*hflat*gauge.conjugate().T).applyfunc(sp.simplify) == sp.zeros(16)
    pureface = pure_edges[0][3]*pure_edges[2][3]*pure_edges[3][3].conjugate().T*pure_edges[1][3].conjugate().T
    assert pureface.applyfunc(sp.simplify) == eye
    # Cut rank and trace-square remain exact for the noncommuting patch.
    cut = native([body]*4, flat_edges[:-1],m,c0,ci)
    cut_rank = (hflat-cut).rank()
    assert cut_rank <= 8 and cut_rank > 0
    # Tree flattening DOES NOT flatten a curved non-tree edge.
    links = [(0,1,0,u,1),(0,2,1,v,1),(1,3,1,v,1),(2,3,0,u,1)]
    curved = native([body]*4,links,m,c0,ci)
    assert curved != hflat
    # Material adverse canonical complement: two inequivalent native onsite sites.
    diagonal = [sp.diag(3,-1,-1,-1)/4, sp.diag(3,-1,-1,-1)/8]
    vals = [3*m/4,3*m/8,-m/8,-m/8,-m/8,-m/4,-m/4,-m/4]
    hdiagonal = native([[x,zero,zero,zero] for x in diagonal],[],m,c0,ci)
    assert sorted(hdiagonal.diagonal(), reverse=True) == vals
    assert all(sp.simplify(vals[i]-vals[i+1]) >= 0 for i in range(7))
    top4 = sum(vals[:4])
    forced2_each = m/2+m/4
    positive_only = 3*m/4+3*m/8
    assert top4 != forced2_each and top4 != positive_only
    # Chemical-potential positive-part dual supplies the COMPLETE canonical sea.
    mu = -m/8
    dual = 4*mu+sum(max(sp.S.Zero,z-mu) for z in vals)
    assert dual == top4
    # Exact discrete IMS native square: partition over all 2-site edge windows.
    # Each vertex degree2; weights 1/sqrt2. Onsite is unchanged, each edge halved.
    ims = sp.zeros(16)
    for x,y,_,_ in pairs:
        selector = sp.zeros(16)
        for z in (x,y):
            for colour in range(4): selector[4*z+colour,4*z+colour] = 1
        ims += selector*hflat*selector/2
    onsite = native([body]*4,[],m,c0,ci)
    assert ims == onsite+(hflat-onsite)/2
    # Full charged64 native linear correction; lengths1/2 and loads unchanged.
    charged = sum(q(pp['onsite'])*2*pp['load']+sum(4*(abs(q(t['c0']))+abs(q(t['ci']))) for t in pp['paths'])*pp['load'] for pp in profiles[1:])
    assert sum(pp['load'] for pp in profiles[1:]) == 64
    # Exact all-field capped comparison prices at inherited actual-family moments.
    # a<=40*2^-2035, c<=(10/7)*2^-2035, M<=2^14.
    R,L = 2**12,2**32
    a,c,M = sp.Rational(40,2**2035),sp.Rational(10,7*2**2035),sp.Integer(2**14)
    tree2 = 48*R**4*a
    flat2 = 1728*R**5*c
    # Coarse dyadic outward prices use sqrt(x)<=2^ceil(log2(x)/2).
    def sqrt_dyadic_upper(x: sp.Rational) -> sp.Rational:
        n, den = int(x.p), int(x.q)
        exponent = n.bit_length()-den.bit_length()+1
        k = -((-exponent)//2)
        upper = sp.Rational(2)**k
        assert upper**2 >= x
        return cast(sp.Rational, upper)
    field_error = (1024*L**3+d)*sqrt_dyadic_upper(cast(sp.Rational, tree2))
    curvature_error = 4*b*L*sqrt_dyadic_upper(cast(sp.Rational, flat2))
    tail_error = d*M/L**3
    boundary_error = 12*b*L/R
    assert boundary_error > 21
    # Better untruncated IMS error uses <s> <=sqrt(M)<=128, no cutoff L.
    ims_error = 144*b/R
    assert ims_error < 1
    # Actual-family local support (not Clifford support) on any R-box.
    patch_volume = R**3
    cutoff_failure = patch_volume*M/L**4
    field_failure = tree2*2**200
    flat_failure = flat2*2**200
    assert cutoff_failure < sp.Rational(1,2**70)
    assert field_failure < sp.Rational(1,2**1700)
    assert flat_failure < sp.Rational(1,2**1700)
    result = {
      'kind':'EXACT_NATIVE_LOCALBOX_AND_MATERIAL_IMS_LOWER_CONTROLS_NOT_GROUND_SOLVER',
      'canonical_compensation':{'complete_top4':str(top4),'wrong_two_each':str(forced2_each),'wrong_positive_only':str(positive_only),'dual_exact':str(dual),'particles':4,'dimension':8},
      'native_noncommuting_patch':{'sites':4,'AP_edge_retained':True,'gauge_covariance_exact':True,'pure_gauge_face_identity':True,'curved_face_cost':str(curvature),'face_HS_square':str(hs2(face-eye)),'cut_rank':cut_rank,'ordered_Jordan':str(jordan(body)),'commuting_cross_excess':str(cross_excess),'IMS_onsite_and_half_edges_identity':True},
      'neutral_coefficients':{'m':str(m),'c0':str(c0),'ci':str(ci),'b':str(b),'D':str(d)},
      'charged64_full_length12_linear_coefficient':str(charged),
      'first_gate':{'R':R,'L':L,'field_error_upper':str(field_error),'curvature_error_upper':str(curvature_error),'tail_error_upper':str(tail_error),'boundary_error':str(boundary_error),'close_minus21_comparison':False,'reason':'All-field tail-safe capped box boundary price exceeds21; no claimed uniform-body minimization or Clifford rigidity.'},
      'material_second_route':{'method':'Untruncated reduced-density discrete IMS with global chemical-potential dual and all canonical complements','boundary_error_per_Lambda_site_upper':str(ims_error),'boundary_error_less1':True,'actual_family_cutoff_failure_upper':str(cutoff_failure),'actual_family_mean_field_deviation_failure_upper':str(field_failure),'actual_family_mean_flat_link_failure_upper':str(flat_failure),'threshold_squared':'2^-200','ground_Clifford_body_support_paid':False,'local_covariant_flatness_in_probability_paid_under_inherited_moments':True},
      'scope':{'exact_controls':'native rational finite square and canonical onsite complement; no numerical diagonalization','analytical_N_uniform':'All even N>=4 with 1<=R<=N/2 for boxes; IMS windows R<N. Large-window displayed prices apply N>=8192; smaller N use admissible R, not advertised large-window boundary price.','quantum_kinetic':'Kept nonnegative with exact law coefficients; not computed or set to zero','Ref':'Full Ref retained in Gauss module; native complete-H source has no extra independent Ref energy term','phase_atom_or_actual_ground_field_window':False},
      'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Exact native patch/canonical/curvature controls; first gate failed; distinct IMS localization evaluated')


if __name__ == '__main__':
    main()

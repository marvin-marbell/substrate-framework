"""Independent exact charged-affine history review; never imports target producers."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, cast
import zipfile
import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def transfer(mask: int, annihilate: int, create: int) -> tuple[int, int]:
    assert mask >> annihilate & 1 and not mask >> create & 1
    sign = (-1) ** ((mask & ((1 << annihilate)-1)).bit_count())
    mask ^= 1 << annihilate
    sign *= (-1) ** ((mask & ((1 << create)-1)).bit_count())
    return mask | (1 << create), sign


def native_control(n: int) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    index = {xyz: i for i, xyz in enumerate(sites)}
    interior = [xyz for xyz in sites if 0 < xyz[0] < n-1]
    edges = []
    for xyz in sites:
        for axis in range(3):
            if axis == 0 and xyz[0] == n-1:
                continue
            if axis > 0 and xyz[0] in (0, n-1):
                continue
            target = list(xyz)
            target[axis] = (target[axis]+1) % n
            edges.append((xyz, tuple(target), axis))
    assert len(edges) == 3*n**3-5*n**2
    # D has +source/-target. The plane Dl is -1 at x=1.
    potential = {xyz: -F(n-1-xyz[0], n-1) for xyz in interior}
    potential.update({xyz: F(0) for xyz in sites if xyz[0] in (0, n-1)})
    pl = [potential[a]-potential[b] for a, b, _axis in edges]
    l = [F(int(axis == 0 and a[0] == 0)) for a, _b, axis in edges]
    c = n//2-1
    start, target = (c, 1, 1), (c+1, 1, 1)
    k = [F(int(a == start and b == target)) for a, b, _axis in edges]
    def divergence(flow: list[F]) -> dict[tuple[int, int, int], F]:
        out = cast(dict[tuple[int, int, int], F], dict.fromkeys(interior, F(0)))
        for (a, b, _axis), val in zip(edges, flow, strict=True):
            if a in out:
                out[a] += val
            if b in out:
                out[b] -= val
        return out
    assert divergence(pl) == divergence(l)
    assert all(val == -int(xyz[0] == 1) for xyz, val in divergence(l).items())
    p = sum((a*b for a, b in zip(k, pl, strict=True)), F(0))
    raw = sum((a*b for a, b in zip(k, l, strict=True)), F(0))
    q = raw-p
    assert p == -F(1, n-1) and q == F(1, n-1) and raw == 0
    normpl = sum((v*v for v in pl), F(0))
    assert normpl == F(n*n*(n-2), n-1)
    assert sum((v*v for v in l), F(0))-normpl == F(n*n, n-1)
    # Canonical 64 charged species, neutral and Ref4 remain separate full masks.
    v = n**3
    even = sum(15 << (4*i) for i, xyz in enumerate(sites) if xyz[0] % 2 == 0)
    if even >> (4*index[start]) & 1:
        even ^= 15 << (4*index[start])
        even ^= 15 << (4*index[target])
    comp = ((1 << (4*v))-1)^even
    masks = [even if s % 2 == 0 else comp for s in range(64)]+[even]
    initial = sum(m << (4*v*s) for s, m in enumerate(masks)) | (15 << (4*v*65))
    def counts(mask: int) -> list[int]:
        return [((mask >> (4*v*s)) & ((1 << (4*v))-1)).bit_count() for s in range(65)]
    def rho(mask: int) -> list[int]:
        return [sum(((mask >> (4*v*s+4*i)) & 15).bit_count()-2 for s in range(64))
                + ((((mask >> (4*v*65)) & 15).bit_count()-4) if i == 0 else 0)
                for i in range(v)]
    assert counts(initial) == [2*v]*65 and rho(initial) == [0]*v
    after, sign = transfer(initial, 4*index[target], 4*index[start])
    restored, backsign = transfer(after, 4*index[start], 4*index[target])
    assert restored == initial and sign*backsign == 1 and counts(after) == counts(initial)
    drho = rho(after)
    assert drho[index[start]] == 1 and drho[index[target]] == -1 and sum(abs(x) for x in drho) == 2
    plane = initial
    hops = []
    for y, z in itertools.product(range(n), repeat=2):
        a, b = index[(0, y, z)], index[(1, y, z)]
        plane, sgn = transfer(plane, 4*v+4*b, 4*v+4*a)
        hops.append((a, b, sgn))
    assert counts(plane) == counts(initial)
    assert all(rho(plane)[index[(0, y, z)]] == 1 and rho(plane)[index[(1, y, z)]] == -1
               for y, z in itertools.product(range(n), repeat=2))
    for a, b, sgn in reversed(hops):
        plane, back = transfer(plane, 4*v+4*a, 4*v+4*b)
        assert back == sgn
    assert plane == initial
    # A different canonical singlet configuration has fullrho +/-128: no Gibbs rho0 assumption.
    stripe = sum(15 << (4*i) for i, xyz in enumerate(sites) if xyz[0] < n//2)
    arbitrary = sum(stripe << (4*v*s) for s in range(65)) | (15 << (4*v*65))
    assert counts(arbitrary) == [2*v]*65
    assert rho(arbitrary) == [128 if xyz[0] < n//2 else -128 for xyz in sites]
    return {'N': n, 'dynamic_edges': len(edges), 'longitudinal_cross': str(p),
            'transverse_cross': str(q), 'HC_cross_per_beta': str(64*p),
            'memory_cross_per_beta': str(32*q), 'total_action_cross_per_beta': str(32*p),
            'canonical_counts': 2*v, 'CAR_inverse_signs': True,
            'charge_change_matches_Dk': True, 'aux1_plane_hops': len(hops),
            'arbitrary_canonical_nonzero_fullrho': True, 'Ref4_retained': True,
            'normal_plane_not_tangential_source': True}


def symbolic_history_control() -> dict[str, Any]:
    beta, a, b, c, p, q, kk, ll, hk, hl, base = sp.symbols('beta a b c p q kk ll hk hl base', real=True)
    # a,b = durations/beta; c = overlap/beta. Four-heat ratio uses ONE same initialrho.
    # hk/hl are arbitrary longitudinal background pairings; base its squared norm.
    joint_long = 64*beta*(base+2*a*hk+2*b*hl+a*kk+b*ll+2*c*p)
    solo_k_long = 64*beta*(base+2*a*hk+a*kk)
    solo_l_long = 64*beta*(base+2*b*hl+b*ll)
    baseline_long = 64*beta*base
    cross_long = sp.expand(joint_long+baseline_long-solo_k_long-solo_l_long)
    cross_mem = 128*beta*(c-a*b)*q
    cross = sp.expand(cross_long+cross_mem)
    assert sp.simplify(cross-128*beta*(c*p+(c-a*b)*q)) == 0
    disjoint_raw = sp.simplify(cross.subs(q, -p))
    assert disjoint_raw == 128*a*b*beta*p
    half = sp.simplify(disjoint_raw.subs({a: sp.Rational(1, 2), b: sp.Rational(1, 2)}))
    assert half == 32*beta*p
    # Same-half vs opposite-half gives same total, NOT same split into HC and memory.
    same = (cross_long.subs({a: F(1,2), b: F(1,2), c: F(1,2)}), cross_mem.subs({a:F(1,2), b:F(1,2), c:F(1,2), q:-p}))
    opposite = (cross_long.subs({a:F(1,2), b:F(1,2), c:0}), cross_mem.subs({a:F(1,2), b:F(1,2), c:0, q:-p}))
    assert same == (64*beta*p, -32*beta*p)
    assert opposite == (0, 32*beta*p)
    assert sum(same) == sum(opposite)
    quarter = sp.simplify(disjoint_raw.subs({a:F(1,4), b:F(1,2)}))
    assert quarter == 16*beta*p and quarter != half
    # Unnormalized two-copy product leaves -baseline; it is not connected inclusion-exclusion.
    wrong_background = sp.expand(joint_long-solo_k_long-solo_l_long)
    assert sp.simplify(wrong_background-cross_long) == -64*beta*base
    # Shifted positive metric Gaussian: completing square cancels all initial affine twists.
    G = sp.Matrix([[2,1],[1,3]])
    x1,x2,w1,w2 = sp.symbols('x1 x2 w1 w2', real=True)
    x,w = sp.Matrix([x1,x2]),sp.Matrix([w1,w2])
    assert sp.expand(((x+w).T*G*(x+w))[0]-(x.T*G*x)[0]-2*(x.T*G*w)[0]-(w.T*G*w)[0]) == 0
    assert G.det() == 5
    return {'general_cross_action': str(cross), 'disjoint_raw_cross_action': str(disjoint_raw),
            'same_half_HC_memory': list(map(str,same)), 'opposite_half_HC_memory': list(map(str,opposite)),
            'quarter_half_cross_action':str(quarter), 'arbitrary_rho_four_heat_ratio_cancels':True,
            'arbitrary_rho_individual_history_does_not_cancel':True,
            'missing_baseline_wrongcase':str(-64*beta*base), 'reset_b_cross_action':str(same[0]),
            'doubleHC_cross_action':str(sum(same)+same[0]), 'Gaussian_metric_det':5,
            'overlap_independence_requires_raw_k_dot_l_zero':True}


def covariance_control() -> dict[str, Any]:
    # Pointwise noncommuting matrices exercise covariance, not an SU4/Phi freeze of H.
    phi = sp.Matrix([[1,2,0,0],[2,-1,0,0],[0,0,3,1],[0,0,1,-3]])
    fy = sp.diag(2,-2,1,-1)
    u = sp.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,0,sp.I],[0,0,sp.I,0]])
    gx,gy = sp.diag(sp.I,-sp.I,-1,-1),sp.diag(-1,sp.I,-sp.I,-1)
    assert gx.det() == gy.det() == u.det() == 1
    B = (phi*u+u*fy)/2
    transformed = (gx*phi*gx.adjoint()*gx*u*gy.adjoint()+gx*u*gy.adjoint()*gy*fy*gy.adjoint())/2
    assert transformed == gx*B*gy.adjoint() and phi*u != u*phi
    return {'endpoint_covariance':True,'all_site_SU4_wedge4_singlet_det':1,
            'physical_reducing_membership':'Gauge-invariant dressed CAR and invariant smooth denominator commute with compact SU4 projection; pointwise probe is not projection of thermal state.',
            'form_domain_claim':'Only smooth equivariant core preservation reviewed; no whole closed form-domain claim.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.input.read_text())
    assert sha(Path(request['contract_path']).read_bytes()) == request['contract_sha256']
    for name, pin in request['pins'].items():
        assert sha((ROOT/name).read_bytes()) == pin, name
    snapshot = request['snapshot']
    assert sha((ROOT/snapshot['path']).read_bytes()) == snapshot['sha256']
    original = json.loads((ROOT/request['producer_receipt']).read_text())
    archived = {}
    for archive in [request['preexecution_archive'],request['executed_archive']]:
        with zipfile.ZipFile(ROOT/archive) as z:
            observed = {}
            for name, pin in original['input_sha256'].items():
                assert sha(z.read(name)) == pin, (archive,name)
                observed[name] = pin
            archived[archive] = observed
    results = [native_control(n) for n in request['N']]
    for row, target in zip(results,original['native_forms'],strict=True):
        assert row['longitudinal_cross'] == target['longitudinal_cross_exact']
        assert row['transverse_cross'] == target['transverse_cross_exact']
        assert row['total_action_cross_per_beta'] == target['joint_minus_independent_exponent_per_beta']
    output = {'kind':'INDEPENDENT_EXACT_ALL_N_HISTORY_REVIEW_CONTROL_NOT_GIBBS',
              'source_sha256':sha(Path(__file__).read_bytes()),'input_sha256':sha(args.input.read_bytes()),
              'original_pins':request['pins'],'archived_members_verified':archived,
              'postexecution_snapshot':snapshot,'native_controls':results,
              'history_controls':symbolic_history_control(),'SU4_controls':covariance_control(),
              'allN_derivation':'Dl=-1 at x1; phi_l(x)=-(N-1-x)/(N-1), endpoints0. D D^T phi_l=Dl pointwise; uniqueness follows positive Dirichlet Laplacian. At central normal k, k.P l=-1/(N-1); k.l=0 hence Qk.Ql=+1/(N-1).',
              'all_character_integral':'sum_m integral_[0,1)^d db f(m+b+twist)=integral_R^d dp f(p); square completion in metric G yields tE=(pi/(64beta))^(d/2)/sqrt(detLI) and transverse variance. Fixed affine rho0 shifts initial continuum momentum, not character resampling.',
              'native_Gibbs_computed':False,'old_producer_imported_or_run':False}
    with args.output.open('x') as stream:
        json.dump(output,stream,indent=2)
        stream.write('\n')
    print(json.dumps({'N':request['N'],'allN_history_control':'exact','Gibbs':False}))


if __name__ == '__main__':
    main()

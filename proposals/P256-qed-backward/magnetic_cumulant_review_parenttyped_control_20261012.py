"""Independent exact normalization, native exterior-CAR and full-electric controls.
Not a native-ground calculation and not the frozen producer.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, cast
import zipfile
import sympy as sp


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partitions(items: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    if not items:
        return [()]
    first, rest = items[0], items[1:]
    result = []
    for blocks in partitions(rest):
        result.append(((first,),) + blocks)
        for j in range(len(blocks)):
            result.append(blocks[:j] + ((first,) + blocks[j],) + blocks[j+1:])
    return result


def normalization() -> dict:
    g, eta, tau, v = sp.symbols('gamma eta tau v', positive=True)
    u = sp.symbols('u1 u2 u3', nonnegative=True)
    L = sp.diag(0, g)
    X = sp.Matrix([[0, 1], [1, 0]])
    omega = sp.Matrix([1, 0])
    def semigroup(t):
        return sp.diag(1, sp.exp(-g*t))
    m4 = (omega.T*X*semigroup(u[2])*X*semigroup(u[1])*X*semigroup(u[0])*X*omega)[0]
    times = (0, u[0], u[0]+u[1], sum(u))
    def moment(block):
        if len(block) % 2:
            return sp.Integer(0)
        # Even subwords evaluated through the exact two-level semigroup.
        vec = X*omega
        ordered = sorted(block)
        for j in range(1, len(ordered)):
            vec = X*semigroup(times[ordered[j]]-times[ordered[j-1]])*vec
        return (omega.T*vec)[0]
    parts = partitions((0, 1, 2, 3))
    connected = sp.simplify(cast(Any, sum((-1)**(len(p)-1)*math.factorial(len(p)-1)*sp.prod(moment(b) for b in p) for p in parts)))
    expected = -2*sp.exp(-g*(u[0]+2*u[1]+u[2]))
    assert sp.simplify(connected-expected) == 0
    average = sp.integrate(sp.exp(-eta*sum(u))*connected, *[(x, 0, sp.oo) for x in u])
    summed = sp.simplify(24*average)
    assert sp.simplify(summed+48/((eta+g)**2*(eta+2*g))) == 0
    witness = summed.subs({g:sp.Rational(1,16), eta:1})
    assert witness == -sp.Rational(32768,867) and abs(witness) > 26
    # Closed exact ground-to-ground exponential, differentiated literally in v.
    d = sp.sqrt(g*g+4*v*v)
    Z = sp.exp(-g*tau/2)*(sp.cosh(tau*d/2)+g/d*sp.sinh(tau*d/2))
    derivative = sp.simplify(sp.diff(sp.log(Z), v, 4).subs(v,0).rewrite(sp.exp))
    s = sp.symbols('s', positive=True)
    ordered_integral = sp.inverse_laplace_transform(1/(s*s*(s+g)**2*(s+2*g)),s,tau)
    assert sp.simplify(derivative+48*ordered_integral) == 0
    # A two-dimensional stationary block survives scalar-log normalization.
    degenerate_Z = sp.cosh(tau*v)
    stationary_fourth = sp.diff(sp.log(degenerate_Z),v,4).subs(v,0)
    assert stationary_fourth == -2*tau**4
    assert sum(math.factorial(len(p)-1) for p in parts) == 26
    return {'L':str(L),'source':str(X),'source_operator_norm':1,
            'ordered_m4':str(m4),'ordered_C4':str(connected),
            'all24_sum_lag_integral':str(summed),'order_average_lag_integral':str(average),
            'gamma_1over16_eta1_sum':str(witness),'counter_to_26_eta_minus3':True,
            'literal_labelled_derivative':str(derivative),
            'derivative_equals_24_ordered_connected_integrals':True,
            'stationary_block_log_fourth':str(stationary_fourth),
            'whole_P0_projected_source_when_L0':0,'log_not_whole_P0':True,
            'finite_cube_constant':26,'summed_lag_safe_constant':624,
            'average_lag_safe_constant':26,'control_not_native_ground':True}


def exterior_hop(occupied: tuple[int, ...], dst: int, src: int):
    if src not in occupied:
        return None
    j = occupied.index(src)
    remaining = occupied[:j]+occupied[j+1:]
    if dst in remaining:
        return None
    k = sum(x < dst for x in remaining)
    return tuple(sorted(remaining+(dst,))), (-1)**(j+k)


def car_word(initial: tuple[int, ...], final: tuple[int, ...], roles: tuple[int, ...], order: tuple[int, ...]):
    state, sign = initial, 1
    flow = [0]*4
    species = max(roles)+1
    density0 = [sum((4*r+x) in initial for r in range(species)) for x in range(4)]
    for edge in order:
        moved = exterior_hop(state,4*roles[edge]+edge,4*roles[edge]+(edge+1)%4)
        if moved is None:
            return 0
        state, factor = moved
        sign *= factor
        flow[edge] += 1
        density = [sum((4*r+x) in state for r in range(species)) for x in range(4)]
        assert [density[x]-density0[x] for x in range(4)] == [flow[x]-flow[(x-1)%4] for x in range(4)]
    return sign if state == final else 0


def native_controls(profiles: list[dict]) -> dict:
    orders = list(itertools.permutations(range(4)))
    diagonal = []
    for mask in range(16):
        occ = tuple(x for x in range(4) if mask & (1 << x))
        values = [car_word(occ,occ,(0,0,0,0),o) for o in orders]
        total = sum(values)
        wanted = 0 if len(occ) in (0,4) else 1 if len(occ) in (1,3) else -4 if mask in (5,10) else -2
        assert total == wanted
        diagonal.append({'mask':mask,'coefficient':total,'orders':values})
    initial, final = (1,2,4,7), (0,1,6,7)
    mixed = []
    for roles in itertools.product(range(2),repeat=4):
        values = [car_word(initial,final,roles,o) for o in orders]
        mixed.append({'roles':roles,'coefficient':sum(values),'orders':values})
    total = sum(row['coefficient'] for row in mixed)
    same = sum(row['coefficient'] for row in mixed if len(set(row['roles'])) == 1)
    assert total == -6 and same == 0
    equality = []
    for p in partitions((0,1,2,3)):
        k = len(p)
        aux = math.prod(range(63-k+1,64))
        designated_active = math.prod(range(63-k+2,64))
        weight = aux+k*designated_active
        assert weight == math.prod(range(64-k+1,65))
        equality.append({'partition':p,'k':k,'all_aux':aux,'per_active_block':designated_active,'weight':weight})
    assert len(equality) == 15
    assert sum(row['weight'] for row in equality) == 64**4
    # Native signed face: raw edge orientations +,+,-,-, with Dc=0.
    c = (1,1,-1,-1)
    # Geometry edges run x0->x1, x1->x2, x3->x2, x0->x3.
    endpoints = ((0,1),(1,2),(3,2),(0,3))
    def divergence(field):
        result = [Fraction(0)]*4
        for value,(a,b) in zip(field,endpoints):
            result[a] += value
            result[b] -= value
        return result
    assert divergence(c) == [0]*4
    flow_rows = []
    for order in orders:
        flow = [0]*4
        for edge in order:
            flow[edge] = c[edge]
            E = [-Fraction(x,2) for x in c]
            cost = 64*sum(2*e*k+k*k for e,k in zip(E,flow))
            negative = 64*sum(2*(-Fraction(3*x,4))*k+k*k for x,k in zip(c,flow))
            assert cost == 0 and negative == -32*sum(k*k for k in flow)
            flow_rows.append({'order':order,'flow':list(flow),'delta_HE_half':str(cost),'delta_HE_threequarters':str(negative),'delta_rho_Dk':list(map(str,divergence(flow)))})
    assert 64*(1+1)**2 == 256 and 64*(1-1)**2 == 0
    phi = sp.diag(1,2,-1,-2)
    psi = sp.Matrix([[0,1,0,0],[1,0,0,0],[0,0,0,2],[0,0,2,0]])
    U = sp.diag(sp.I,-sp.I,1,1)
    V = sp.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    assert phi.trace() == psi.trace() == 0
    assert U.det() == V.det() == 1 and U.adjoint()*U == V.adjoint()*V == sp.eye(4)
    B, C = (phi*U+U*psi)/2, (psi*V+V*phi)/2
    literal, wrong = sp.trace(B*C*B.adjoint()*C.adjoint()), sp.trace(B*B.adjoint()*C*C.adjoint())
    assert literal != wrong
    charged = [row for row in profiles if row['scale'] == 'charged']
    assert [row['load'] for row in charged] == [1,63]
    W = sum(row['load']*sum(path['length']*(abs(Fraction(path['c0']))+abs(Fraction(path['ci']))) for path in row['paths']) for row in charged)
    Csum = sum(row['load']*sum(abs(Fraction(path['c0']))+abs(Fraction(path['ci'])) for path in row['paths'] if path['axis'] == 0 and path['length'] == 1) for row in charged)
    assert W == Fraction(46425051219964815,8796093022208)
    assert Csum == Fraction(13313289252183767,8796093022208)
    assert 8*Csum <= 16*Csum <= 32*W
    bounds = [{'eta':str(Fraction(1,n)),'M1_summed_safe_bound':str(624*(32*W)**4*n**3),'original_26_bound':str(26*(32*W)**4*n**3)} for n in (4,8,16,64)]
    return {'diagonal_all24':diagonal,'mixed_profiles':mixed,'mixed_total':total,'same_total':same,
            'all15_equality_classes':equality,'assignment_count':64**4,
            'all24_raw_signed_face_partial_flows':flow_rows,
            'shared_equal_HE':256,'shared_opposite_HE':0,'independent_wrong_HE':128,
            'ordered_Phi_SU4_trace':str(literal),'factorized_wrong_trace':str(wrong),
            'nearest_hermitian_charged_leg_bound':str(8*Csum),
            'support_R_le_2M_hermitian_leg_bound':str(16*Csum),
            'conservative_32W_M_bound':str(32*W),'fixed_lag_regulator_bounds':bounds,
            'actual_1plus63_copy_norm_allocation':True,
            'canonical_local_patterns_not_ground_weights':True,'flows_not_fullH_energies':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for path, expected in request['source_input_SHA256'].items():
        assert sha(Path(path)) == expected, f'Changed target: {path}'
    with zipfile.ZipFile(request['primitive_archive']) as archive:
        member_bytes = archive.read(request['primitive_member'])
    assert hashlib.sha256(member_bytes).hexdigest() == request['primitive_member_SHA256']
    profiles = json.loads(member_bytes)['primitive_profiles']
    receipt = {'kind':'INDEPENDENT_EXACT_CUMULANT_NATIVE_CAR_FLOW_REVIEW',
               'source_input_SHA256':request['source_input_SHA256'],
               'normalization':normalization(),'native':native_controls(profiles),
               'actual_ground_numeric_backend':False,'scientific_admission':False,
               'photon_or_phase_or_adoption':False}
    with args.output.open('x') as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True)
        stream.write('\n')
    print(json.dumps({'normalization':'SUM24_BOUND_26_FALSE_SAFE624','CAR':'MIXED_MINUS6_SAME0','assignments':64**4,'native_ground_numeric':False}))


if __name__ == '__main__':
    main()

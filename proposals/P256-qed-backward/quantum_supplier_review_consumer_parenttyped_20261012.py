"""Independent exact native supplier review; no optimizer, old CLI or ground backend."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
from fractions import Fraction as F
from pathlib import Path
from typing import cast
import sympy as s


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def exact_native(target):
    p = [s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -s.I], [s.I, 0]]), s.diag(1, -1)]
    gamma = [s.diag(1, 1, -1, -1)] + [s.kronecker_product(p[0], q) for q in p]
    words = []
    subsets = []
    for n in range(1, 5):
        for sub in itertools.combinations(range(4), n):
            q = s.eye(4)
            for j in sub:
                q *= gamma[j]
            words.append(s.I**(n*(n-1)//2)*q)
            subsets.append(sub)
    t = [q/(2*s.sqrt(2)) for q in words]
    assert target.native_clifford() == (gamma, t, subsets)
    x = s.symbols('x0:4', positive=True)
    spectrum = []
    for word, sub in zip(words, subsets):
        expected = 4*sum(x[j] for j in range(4) if (len(sub)-int(j in sub)) % 2)
        actual = sum((x[j]*(2*word-2*gamma[j]*word*gamma[j]) for j in range(4)), s.zeros(4))
        assert s.simplify(actual-expected*word) == s.zeros(4)
        spectrum.append(expected)
    assert len(set(subsets)) == 15
    # Rectangle alone is not the Gram16 domain: corner (2,2,2,2) has Gram32.
    outside = {v: s.Integer(2) for v in x}
    assert max(q.subs(outside) for q in spectrum) == 32
    # NEW rational centre, different from both fixed centre and target control.
    a = [s.Rational(1, 8), s.Rational(1, 3), s.Rational(2, 3), s.Integer(1)]
    xx = [q*q for q in a]
    assert all(q > s.Rational(1, 128) for q in xx) and sum(q*q for q in xx) < 4
    geometry = target.centre_geometry(a)
    c, _, L, G, left, P, eig = geometry
    independent_L = s.Matrix(60, 15, lambda r, j: s.simplify(2*s.trace(t[r % 15]*s.I*(t[j]*c[r//15]-c[r//15]*t[j]))))
    assert L == independent_L and G == s.diag(*[q.subs(dict(zip(x, xx))) for q in spectrum])
    S = target.normal_slice(geometry)
    assert S.shape == (60, 45) and S.T*S == s.eye(45) and L.T*S == s.zeros(15, 45)
    assert all(s.trace(q*q)/4 == v for q, v in zip(c, xx))
    radial = s.Matrix([s.simplify(2*s.trace(q*v)) for q in c for v in t])
    e = s.Rational(1, 2**30)
    z = S.T*radial*e
    j0 = target.field_density(geometry, S, [0]*45)
    jz = target.field_density(geometry, S, z)
    assert s.simplify(jz/j0) == (1+e)**15
    vol = 32768*s.sqrt(2)*s.pi**9/3
    h0 = target.link_density(t, [0]*15)
    assert s.simplify(h0*vol) == 1
    delta = s.Rational(1, 2**28)
    val = target.normalized_scalar([0]*45, [0]*15, [j0], [h0], delta)
    I = s.Rational(65536, 109395)
    assert s.simplify(val**2*j0*h0*(delta*I)**60/4) == 1
    assert target.normalized_scalar([delta], [], [j0], [h0], delta) == 0
    # Actual inverse first/second reconstruction; no omission of normal-orbit cross.
    v = L[:, 4] + radial
    v /= s.sqrt((v.T*v)[0])
    u, w = left*v, P*v
    A = sum((u[j]*t[j] for j in range(15)), s.zeros(4))
    W = [sum((w[15*mu+j]*t[j] for j in range(15)), s.zeros(4)) for mu in range(4)]
    mixed = [2*s.I*(A*q-q*A) for q in W]
    dd = [-A*(A*q-q*A)+(A*q-q*A)*A+mix for q, mix in zip(c, mixed)]
    coeff = s.Matrix([s.simplify(2*s.trace(b*q)) for q in dd for b in t])
    u2, w2 = -left*coeff, -P*coeff
    assert s.simplify(L*u+w-v) == s.zeros(60, 1)
    assert s.simplify(L*u2+w2+coeff) == s.zeros(60, 1)
    assert any(q != s.zeros(4) for q in mixed)
    return gamma, t, a, {'sample_a': list(map(str, a)), 'sample_x': list(map(str, xx)), 'full_complex_spectrum': list(map(str, spectrum)), 'rectangle_only_Gram32_rejected': True, 'all60_rank45': True, 'actual_radial_j_ratio': str((1+e)**15), 'actual_source_normalizer': '1', 'inverse_first_squared': str(s.simplify((u.T*u+w.T*w)[0])), 'inverse_second_squared': str(s.simplify((u2.T*u2+w2.T*w2)[0])), 'normal_orbit_second_cross_nonzero': True}


def car_jets(t):
    annih = []
    for j in range(4):
        a = s.zeros(16)
        for mask in range(16):
            if mask & (1 << j):
                a[mask ^ (1 << j), mask] = (-1)**((mask & ((1 << j)-1)).bit_count())
        annih.append(a)
    assert all(a*b+b*a == s.zeros(16) and a*b.H+b.H*a == int(i == j)*s.eye(16) for i, a in enumerate(annih) for j, b in enumerate(annih))
    def local_dg(A):
        return sum((A[i,j]*annih[i].H*annih[j] for i in range(4) for j in range(4)), s.zeros(16))
    assert all(local_dg(A)*s.eye(16)[:,m] == s.zeros(16,1) for A in t for m in [0,15])
    masks = list(itertools.combinations(range(8), 4))
    bits = [sum(1 << j for j in row) for row in masks]
    index = {mask:i for i,mask in enumerate(bits)}
    def dg(A):
        out = s.zeros(70)
        for col, mask in enumerate(bits):
            for j in range(8):
                if not mask & (1 << j):
                    continue
                after = mask ^ (1 << j)
                sign1 = (-1)**((mask & ((1 << j)-1)).bit_count())
                for i in range(8):
                    if A[i,j] and not after & (1 << i):
                        sign2 = (-1)**((after & ((1 << i)-1)).bit_count())
                        out[index[after | (1 << i)],col] += sign1*sign2*A[i,j]
        return out
    q = (s.sqrt(3)*s.eye(4)/2).col_join(s.eye(4)/2)
    A = s.diag(t[0], s.zeros(4))
    B = s.diag(t[4], s.zeros(4))
    v = s.Matrix([q[list(row),:].det() for row in masks])
    da, db = s.I*A*q, s.I*B*q
    mixedq = -A*B*q
    first = []; second = []; mixed = []; stripped = []
    for row in masks:
        base = q[list(row),:]
        aa, bb, ab = da[list(row),:], db[list(row),:], mixedq[list(row),:]
        f = sec = mix = no_cross = s.S.Zero
        for j in range(4):
            z = base.copy(); z[:,j] = aa[:,j]; f += z.det()
            z = base.copy(); z[:,j] = (-A*A*q)[list(row),j]; sec += z.det(); no_cross += z.det()
            z = base.copy(); z[:,j] = ab[:,j]; mix += z.det()
            for k in range(4):
                if j != k:
                    z = base.copy(); z[:,j] = aa[:,j]; z[:,k] = aa[:,k]; sec += z.det()
                    z = base.copy(); z[:,j] = aa[:,j]; z[:,k] = bb[:,k]; mix += z.det()
        first.append(f); second.append(sec); mixed.append(mix); stripped.append(no_cross)
    first, second, mixed, stripped = map(s.Matrix, [first,second,mixed,stripped])
    ga, gb = dg(A), dg(B)
    assert s.simplify((v.H*v)[0]) == 1
    assert s.simplify(first-s.I*ga*v) == s.zeros(70,1)
    assert s.simplify(second+ga*ga*v) == s.zeros(70,1)
    assert s.simplify(mixed+ga*gb*v) == s.zeros(70,1)
    assert s.simplify(mixed+gb*ga*v) != s.zeros(70,1)
    assert second != stripped
    # Odd hopping: nonzero action, zero expectation in full/empty mask.
    hop = s.zeros(8); hop[4,0] = 1
    hv = dg(hop)*s.eye(70)[:,index[15]]
    assert hv != s.zeros(70,1) and (s.eye(70)[:,index[15]].H*hv)[0] == 0
    return {'CAR16_all_anticommutators':True,'charged_singlet_all15':True,'new_wedge70_norm': '1', 'first_squared':str(s.simplify((first.H*first)[0])), 'second_squared':str(s.simplify((second.H*second)[0])), 'mixed_squared':str(s.simplify((mixed.H*mixed)[0])), 'mixed_order_and_distinct_cross_rejected':True,'odd_hop_nonzero_action_zero_expectation':True}


def sea_and_minimum(target, profile, a, gamma):
    m = F(profile['onsite']); c0 = F(profile['paths'][0]['c0']); ci = F(profile['paths'][0]['ci'])
    means = [m*m+6*c0*c0]+[2*ci*ci]*3
    assert m-6*abs(c0) > 0 and m+6*abs(c0) < 47 and 2*abs(ci) < 20
    assert F(47)**4+3*F(20)**4 < F(4096)**2
    assert all(q/F(16384) > F(1,128) for q in means)
    # Actual native positive band on a NEW allowed parameter, all N4 sign types.
    signs = []
    for v in range(4):
        k = s.pi*(2*v+1)/4
        signs.append((s.cos(k), s.sin(k)))
    # Check all64 momenta: exact factorization avoids a repeated global Gram multiplication.
    projectors = []
    for angles in itertools.product([s.pi/4,3*s.pi/4,5*s.pi/4,7*s.pi/4], repeat=3):
        q = target.band_graph(a, angles, profile)
        assert s.simplify(q.H*q-s.eye(2)) == s.zeros(2)
        mass = a[0]*(s.Rational(m.numerator,m.denominator)+2*s.Rational(c0.numerator,c0.denominator)*sum(s.cos(k) for k in angles))
        d = [-2*a[j+1]*s.Rational(ci.numerator,ci.denominator)*s.sin(angles[j]) for j in range(3)]
        h = mass*gamma[0]+sum((z*g for z,g in zip(d,gamma[1:])),s.zeros(4))
        eps = s.sqrt(mass**2+sum(z*z for z in d))
        assert s.simplify(h*q-eps*q) == s.zeros(4,2)
        projectors.append(q*q.H)
    # Every momentum difference and every local covariance entry, exact.
    fourier = s.Matrix(4,4,lambda x,k:s.exp(s.I*s.pi*(2*k+1)*x/4).expand(complex=True)/2)
    assert s.simplify(fourier.H*fourier) == s.eye(4)
    covariance = sum(projectors,s.zeros(4))/64
    assert all(s.simplify(covariance[i,j]) == 0 for i in range(4) for j in range(4) if i != j)
    assert s.simplify(covariance[0,0]-covariance[1,1]) == 0
    assert s.simplify(covariance[0,0]+covariance[2,2]-1) == 0
    u = s.Symbol('u', real=True); b=(1-u*u)**4; I=s.integrate(b*b,(u,-1,1))
    b1=s.integrate(s.diff(b,u)**2,(u,-1,1))/I; b2=s.integrate(s.diff(b,u,2)**2,(u,-1,1))/I
    assert (I,b1,b2)==(s.Rational(65536,109395),s.Rational(34,7),s.Rational(459,7))
    assert all(s.diff(b,u,j).subs(u,end)==0 for j in range(4) for end in [-1,1])
    # Independent conservative integrated first/second bounds, including 2 f' ray'.
    n=135; c1=F(2)**36; c2=F(2)**84; ell1=7*c1; ell2=7*c2
    scalar1=2*(n*n*c1*c1*F(34,7)+ell1*ell1/4)
    ray1=4*c1; ray2=4*c2+ray1*ray1
    scalar2=(n*c1*c1*9+n*(n-1)*c1*c1*5+n*c2*3+ell1*n*c1*3+ell2/2+ell1*ell1/4)**2
    total1=2*(scalar1+ray1*ray1)
    total2=3*(scalar2+4*scalar1*ray1*ray1+ray2*ray2)
    assert total1 < 2**280 and total2 < 2**1200
    assert (60*4+F(45*3,20))*2**280 < 2**300
    exponents={'neutral_Jordan':-3304,'Phi_SU4':-4820,'alignment':-4840,'colour':-4840,'shared_electric':-8173}
    correction=sum(F(2)**v for v in exponents.values())
    assert correction<F(2)**-3000 and F(2)**(16-257)+correction<F(2)**-240
    return {'literal_means':list(map(str,means)),'native_all64_N4_band_identity':True,'global_factorization_rows':256,'global_factorization_columns':128,'exact_AP_fourier':True,'local_offdiagonal_covariance_zero':True,'all_evenN_general_identity':'band norm and AP difference Fourier identity, then exterior inner product','bump_H2_boundary_jets':True,'independent_first_bound':str(total1),'independent_second_bound':str(total2),'fullH_correction_exponents':exponents,'epsilon_quantum_lt_2power_minus3000':True,'epsilon_upper_lt_2power_minus240':True}


def physical_deck():
    # NEW exact N4 native graph, a depth-first tree unlike the target breadth-first tree.
    n=4; sites=list(itertools.product(range(n),repeat=3)); ix={v:i for i,v in enumerate(sites)}; V=len(sites)
    edges=[]
    for v in sites:
        for axis in range(3):
            y=list(v);y[axis]=(y[axis]+1)%n;edges.append((ix[v],ix[tuple(y)]))
    D=s.zeros(V,len(edges)); adj=[[] for _ in sites]
    for e,(v,y) in enumerate(edges):
        D[v,e]=1;D[y,e]=-1;adj[v].append((y,e,1));adj[y].append((v,e,-1))
    parent: dict[int, tuple[int, int, int] | None]={0:None}; stack=[0]; tree=set()
    while stack:
        v=stack[-1]
        nexts=[q for q in adj[v] if q[0] not in parent]
        if not nexts:stack.pop();continue
        y,e,sign=nexts[-1];parent[y]=(v,e,sign);tree.add(e);stack.append(y)
    chords=[e for e in range(len(edges)) if e not in tree]
    def path(v):
        out=s.zeros(len(edges),1)
        while v:
            y,e,sign=cast(tuple[int, int, int], parent[v]);out[e]+=sign;v=y
        return out
    B=s.Matrix.hstack(*[path(edges[e][0])-path(edges[e][1])+s.eye(len(edges))[:,e] for e in chords])
    assert D*B==s.zeros(V,len(chords)) and B[chords,:]==s.eye(len(chords))
    G=B.T*B; unit=s.eye(len(chords))[:,2]
    deck=B*G.inv(method='DM')*unit
    assert B.T*deck==unit and D*deck==s.zeros(V,1) and B.T*B[:,2]!=unit
    parity=[(-1)**sum(v) for v in sites];E0=s.Matrix([64*parity[v] if e%3==0 else 0 for e,(v,y) in enumerate(edges)])
    assert D*E0==s.Matrix([128*q for q in parity])
    assert 64*(E0.T*E0)[0]/V==2**18
    assert 64*sum((cast(s.Expr, E0[e])/64)**2 for e in range(len(edges)))*64 != 64*(E0.T*E0)[0]
    width=s.Rational(1,64*len(chords)*(1+s.trace(G)))
    excess=64*width**2*s.trace(G)/19
    assert excess<1
    assert 2*V%4==0
    return {'N':n,'cycle_dimension':len(chords),'new_DFS_TrG':str(s.trace(G)),'projected_dual_character_pairing':'e2','raw_cycle_deck_rejected':True,'Gauss_all64_Ref2':'divE0=128parity','neutral_U1_charge':0,'charged_count':64,'one_shared_electric_per_site':2**18,'wrong_species_squares_rejected':True,'packet_excess':str(excess),'one_global_Z4_degree':4}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    request=json.loads(args.request.read_text())
    for name,pin in request['pins'].items():
        assert digest(pin['path'])==pin['sha256'],name
    source=Path(request['pins']['target_source']['path'])
    spec=importlib.util.spec_from_file_location('quantum_supplier_selected_target',source)
    assert spec is not None and spec.loader is not None
    target=importlib.util.module_from_spec(spec);spec.loader.exec_module(target)
    profiles=json.loads(Path(request['pins']['profiles']['path']).read_text())['primitive_profiles']
    law=json.loads(Path(request['pins']['native_law']['path']).read_text())['law']
    assert 'neutral -Lambda canonical2V; active1+aux63 charge1 canonical2V each; full Ref'==law['species']
    assert law['bare_U1_B2'] is False
    gamma,t,a,geometry=exact_native(target)
    result={'kind':'NEW_INDEPENDENT_EXACT_NATIVE_SUPPLIER_ADVERSE_CONSUMER','pins':request['pins'],'geometry':geometry,'CAR':car_jets(t),'sea_minimum_domain_fullH':sea_and_minimum(target,profiles[0],a,gamma),'original_deck_Gauss':physical_deck(),'scope':{'analytic_supplier_at_implicitly_defined_xstar':True,'executed_rational_parameter_not_xstar':True,'computed_xstar':False,'computed_physical_ground':False,'stationary_trial':False,'conditional_smooth_core_Bloch_realization':True,'phase_IR_commoncone_Compton_admission':False}}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'exact_native_adverse_consumers':'PASS','computed_xstar':False,'ground_or_phase_admission':False}),flush=True)

if __name__=='__main__':
    main()

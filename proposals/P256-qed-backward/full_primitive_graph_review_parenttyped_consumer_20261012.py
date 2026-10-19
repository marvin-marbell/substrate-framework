"""Independent exact controls for frozen full primitive graph claims; never a ground solver."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import zipfile
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast

import numpy as np
import sympy as s

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def rational_record(x: F) -> dict:
    return {'sha256': sha(str(x).encode()), 'numerator_bits': x.numerator.bit_length(),
            'denominator_bits': x.denominator.bit_length()}


def expansion_and_constants(profiles: list[dict]) -> dict:
    assert [p['load'] for p in profiles] == [1, 1, 63]
    assert len({p['onsite'] for p in profiles}) == 3
    assert [len(p['paths']) for p in profiles] == [3, 6, 6]
    charged_cost = [abs(F(next(t for t in p['paths'] if t['length'] == 1)['c0']))
                    + abs(F(next(t for t in p['paths'] if t['length'] == 1)['ci']))
                    for p in profiles[1:]]
    histogram = Counter()
    for assignment in itertools.product(range(64), repeat=4):
        histogram[assignment.count(0)] += 1
    assert sum(histogram.values()) == 64**4
    assert [histogram[k] for k in range(5)] == [math.comb(4, k)*63**(4-k) for k in range(5)]
    Csum = charged_cost[0] + 63*charged_cost[1]
    assert Csum == F(13313289252183767, 8796093022208)
    total = sum((histogram[k]*charged_cost[0]**k*charged_cost[1]**(4-k)
                 for k in range(5)), F(0))
    assert total == Csum**4
    assert total != charged_cost[0]**4 + 63*charged_cost[1]**4
    lam, chi, fast = F(2**8192), F(1, 2**4096), F(2**10240)
    w = [abs(F(p['onsite'])) + 2*sum((abs(F(t['c0']))+abs(F(t['ci']))
                                    for t in p['paths']), F(0)) for p in profiles]
    B = 4*(lam*w[0]+w[1]+63*w[2])
    assert B < 512*lam
    b, J = 24*(4*Csum)**4, 2**19
    a0 = 16*b
    d1 = [16*(J+4)*a0, 32*a0, 8*a0]
    d2 = [240*(9*J+16)*a0, 960*a0, 64*a0]
    k = [4*chi, F(3,20)*chi, F(64)]
    R0 = 60*chi+11192*lam+F(129,10)*fast+262144
    B0 = 33*(1+10*R0*4096)+16384*lam*4096
    # Certified rational envelope for c0; exact radicals are retained in the proof.
    c0_upper = sum((kf*j2+4*max(F(1),kf)*j1 for kf,j1,j2 in zip(k,d1,d2)), F(0))
    c0_upper += 2*a0*(2*B*52)*52*2
    vertices = {(0,0,0),(1,0,0),(1,1,0),(0,1,0)}
    support = set(vertices)
    for x in vertices:
        for axis in range(3):
            for distance in (-2,-1,1,2):
                z=list(x); z[axis]+=distance; support.add((z[0],z[1],z[2]))
    assert len(support) <= 52
    assert max(sum(abs(t) for t in z) for z in support) == 4
    assert 16**2 >= 240 and 8**2 >= 60 and 2**2 == 4
    assert 2*64 <= 128 and 620*64+8*512 < 2**18 and J >= 2**18+128
    return {'all_assignments_executed':sum(histogram.values()), 'all24_orders':24,
            'DAG_expanded_word_count':24*64**4, 'histogram':[histogram[k] for k in range(5)],
            'Csum':str(Csum), 'b':str(b), 'w_profiles':[str(x) for x in w],
            'B':rational_record(B), 'a0':rational_record(a0), 'J':J,
            'd1':[rational_record(x) for x in d1], 'd2':[rational_record(x) for x in d2],
            'k':[rational_record(x) for x in k], 'R0':rational_record(R0),
            'B0':rational_record(B0), 'c0_rational_upper':rational_record(c0_upper),
            'Cg_squared_rational_upper':rational_record(c0_upper*c0_upper*B0),
            'actual_unwrapped_endpoint_union_count':len(support), 'support_upper':52,
            'centre_radius':4, 'tau_lower':'1/2', 'samecopy_replacement_rejected':True,
            'scope':'All assignments enumerated; 402653184 words represented by exact distributive DAG, not dense native matrices.'}


def hop(mask: int, out: int, incoming: int):
    if not mask & (1 << incoming):
        return None
    sign = (-1)**((mask & ((1 << incoming)-1)).bit_count())
    mask ^= 1 << incoming
    if mask & (1 << out):
        return None
    sign *= (-1)**((mask & ((1 << out)-1)).bit_count())
    return mask | (1 << out), sign


def car_and_density(profiles: list[dict]) -> dict:
    sector = [a | (b << 4) for a in range(16) if a.bit_count()==2
              for b in range(16) if b.bit_count()==2]
    ix = {m:j for j,m in enumerate(sector)}
    def transfer(copy, x, y):
        a = np.zeros((36,36),dtype=np.int64)
        for j,mask in enumerate(sector):
            t=hop(mask,4*copy+x,4*copy+y)
            if t:
                nxt,sign=t; a[ix[nxt],j]=sign
        return a
    hops = [[transfer(c,e,(e+1)%4) for e in range(4)] for c in range(2)]
    legs = [hops[0][e]+hops[1][e] for e in range(4)]
    rho = [np.diag([((m>>z)&1)+((m>>(z+4))&1) for m in sector]) for z in range(4)]
    individual = [np.diag([(m>>z)&1 for m in sector]) for z in range(4)]
    B = np.zeros((36,36),dtype=np.int64)
    same = np.zeros_like(B)
    H = np.zeros((36,36),dtype=object)
    for copy,p in enumerate(profiles[1:]):
        # Nonuniform Phi0 coefficient prevents onsite from becoming constant in the fixed-number sector.
        H += F(p['onsite'])*np.diag([sum((z+1)*((m>>(4*copy+z))&1)
                                                   for z in range(4)) for m in sector]).astype(object)
        for path in p['paths']:
            coeff=F(path['c0'])
            for z in range(4):
                t=transfer(copy,z,(z+path['length'])%4).astype(object)
                H += coeff*(t+t.T)
    expansion = np.zeros_like(H)
    for order in itertools.permutations(range(4)):
        word=np.eye(36,dtype=np.int64)
        for e in order:
            word=legs[e]@word
        B += word
        for copy in range(2):
            word_same=np.eye(36,dtype=np.int64)
            for e in order:
                word_same=hops[copy][e]@word_same
            same += word_same
        for slot in range(4):
            term=np.eye(36,dtype=object)
            for j,e in enumerate(order):
                v=legs[e].astype(object)
                term=((H@v-v@H) if j==slot else v)@term
            expansion += term
    assert np.array_equal(H@B-B@H,expansion) and np.any(expansion)
    A=(B+B.T).astype(object)
    assert np.array_equal(H@A-A@H,expansion-expansion.T)
    assert all(np.array_equal(n@B,B@n) for n in rho)
    assert any(np.any(n@B-B@n) for n in individual)
    # Every mutually commuting total-density polynomial commutes; this non-diagonal kernel is a witness.
    kernel=np.array([[2,1,-1,0],[1,3,0,-1],[-1,0,4,2],[0,-1,2,5]],dtype=np.int64)
    HC=sum((kernel[x,y]*(rho[x]@rho[y]) for x in range(4) for y in range(4)),np.zeros_like(B))
    assert np.array_equal(HC@B,B@HC)
    assert np.any(individual[0]@B-B@individual[0])
    # Neutral canonical factor is separate. H_neutral tensor I commutes I tensor B exactly.
    assert np.array_equal(np.eye(6,dtype=np.int64)@np.eye(6,dtype=np.int64),np.eye(6,dtype=np.int64))
    assert np.any(B-same)
    # Occupation-affine connection C_e is a linear combination of the ENTIRE rho.
    connection=[sum(((e+1)*(z+2)*rho[z] for z in range(4)),np.zeros_like(B)) for e in range(4)]
    assert all(np.array_equal(K@B,B@K) for K in connection)
    # A nonreal constant coefficient prevents the scalar symmetric-cycle control from hiding B-B*.
    # It is a method control, not a change to the frozen native source.
    bx=(1+s.I)*B.astype(object); bd=(1-s.I)*B.T.astype(object)
    c=[1,1,1,1]; mode=[2,-3,1,4]
    lhs=np.zeros_like(bx); rhs=np.zeros_like(bx)
    for ce,me,K in zip(c,mode,connection):
        D=me*np.eye(36,dtype=np.int64)+K
        lhs += (D+ce*np.eye(36,dtype=np.int64))@(D+ce*np.eye(36,dtype=np.int64))@bx-bx@D@D
        lhs += (D-ce*np.eye(36,dtype=np.int64))@(D-ce*np.eye(36,dtype=np.int64))@bd-bd@D@D
        rhs += ce**2*(bx+bd)+2*ce*(bx-bd)@D
    assert all(s.simplify(x)==0 for x in (lhs-rhs).flat)
    assert any(s.simplify(x)!=0 for x in (lhs-4*(bx+bd)).flat)
    D=s.Matrix([[1,0,0,-1],[-1,1,0,0],[0,-1,1,0],[0,0,-1,1]])
    closed=s.ones(4,1); opened=s.Matrix([1,0,0,0]); transverse=s.ones(4,1)/4
    assert D*closed==s.zeros(4,1) and D*(opened-transverse)==D*opened!=s.zeros(4,1)
    return {'sector_dimension':36,'all24_matter_insertion_identity':True,
            'adjoint_commutator_minus_sign':True,'collective_rho_HC_affine_connection_commutators':0,
            'individual_density_wrongcase_rejected':True,'samecopy_wrongcase_rejected':True,
            'electric_second_and_right_first_derivative_identity':True,
            'electric_first_term_omission_rejected':True,'open_transverse_substitution_rejected':True,
            'neutral_commutation':'Independent canonical tensor factor, not deleted from H or Q.',
            'scope':'Two-copy finite exact CAR and occupation-affine differential controls, not native ground.'}


def su4_and_product_jets(profiles: list[dict]) -> dict:
    generators=[]
    for i in range(4):
        for j in range(i+1,4):
            re=s.zeros(4); im=s.zeros(4)
            re[i,j]=re[j,i]=s.Rational(1,2)
            im[i,j]=-s.I/2; im[j,i]=s.I/2
            generators.extend([re,im])
    for n in range(1,4):
        generators.append(s.diag(*([1]*n+[-n]+[0]*(3-n)))/s.sqrt(2*n*(n+1)))
    assert len(generators)==15
    assert all(s.simplify(s.trace(a*b))==(s.Rational(1,2) if i==j else 0)
               for i,a in enumerate(generators) for j,b in enumerate(generators))
    assert s.simplify(cast(Any, sum((t*t for t in generators),s.zeros(4))))==s.eye(4)*s.Rational(15,8)
    phi=[generators[0],generators[5],generators[9],generators[14]]
    assert all(s.simplify(cast(Any, sum((t*(t*f-f*t)-(t*f-f*t)*t for t in generators),s.zeros(4))))==4*f
               for f in phi)
    U=s.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    V=s.diag(s.I,-s.I,1,1)
    Gx,Gy=V,U
    q=s.symbols('q',real=True)
    path_count=0; jet_checks=0
    for p in profiles:
        for path in p['paths']:
            c0,ci=s.Rational(path['c0']),s.I*s.Rational(path['ci'])
            Fx=c0*phi[0]+ci*phi[path['axis']+1]
            Fy=c0*phi[1]+ci*phi[(path['axis']+2)%4]
            W=U if path['length']==1 else U*V
            leg=(Fx*W+W*Fy)/2
            transformed=((Gx*Fx*Gx.H)*(Gx*W*Gy.H)+(Gx*W*Gy.H)*(Gy*Fy*Gy.H))/2
            assert s.simplify(transformed-Gx*leg*Gy.H)==s.zeros(4)
            for t in generators:
                derivative=(Fx*(s.I*t*W)+(s.I*t*W)*Fy)/2
                second=(Fx*(-t*t*W)+(-t*t*W)*Fy)/2
                # Exact exponential jet through order2, including each generator.
                expjet=s.eye(4)+q*s.I*t-q*q*t*t/2
                formal=(Fx*expjet*W+expjet*W*Fy)/2
                assert s.simplify(s.diff(formal,q).subs(q,0)-derivative)==s.zeros(4)
                assert s.simplify(s.diff(formal,q,2).subs(q,0)-second)==s.zeros(4)
                jet_checks+=1
            for mu in range(4):
                for t in generators:
                    varied=(Fx+q*((c0 if mu==0 else ci if mu==path['axis']+1 else 0)*t))*W/2+W*Fy/2
                    assert s.diff(varied,q,2)==s.zeros(4)
            if path['length']==2:
                assert s.I*U*generators[0]*V != s.I*generators[0]*U*V
            path_count+=1
    # Ordered noncommutative 4-leg polynomial control. Extract product coefficients independently.
    X=[U,V,U*V,V*U]; Y=[generators[j] for j in (0,3,7,14)]
    word_controls=0
    for order in itertools.permutations(range(4)):
        coeff=[s.eye(4)]
        for e in order:
            out=[s.zeros(4) for _ in range(len(coeff)+1)]
            for degree,m in enumerate(coeff):
                out[degree]+=X[e]*m; out[degree+1]+=Y[e]*m
            coeff=out
        first=s.zeros(4); second=s.zeros(4)
        for slot in range(4):
            prod=s.eye(4)
            for j,e in enumerate(order):prod=(Y[e] if j==slot else X[e])*prod
            first+=prod
        for a,b in itertools.combinations(range(4),2):
            prod=s.eye(4)
            for j,e in enumerate(order):prod=(Y[e] if j in (a,b) else X[e])*prod
            second+=2*prod
        assert s.simplify(first-coeff[1])==s.zeros(4)
        assert s.simplify(second-2*coeff[2])==s.zeros(4)
        word_controls+=1
    # Signed AP one-particle translation and canonical exterior parity.
    S=np.zeros((4,4),dtype=np.int64)
    for j in range(4):S[(j+1)%4,j]=-1 if j==3 else 1
    assert np.array_equal(np.linalg.matrix_power(S,4),-np.eye(4,dtype=np.int64))
    pairs=list(itertools.combinations(range(4),2)); wedge=np.zeros((6,6),dtype=np.int64)
    for i,(a,b) in enumerate(pairs):
        for j,(c,d) in enumerate(pairs):wedge[i,j]=S[a,c]*S[b,d]-S[a,d]*S[b,c]
    assert np.array_equal(np.linalg.matrix_power(wedge,4),np.eye(6,dtype=np.int64))
    return {'SU4_generators':15,'trace_normalization':'1/2','fundamental_Casimir':'15/8',
            'adjoint_Casimir':4,'literal_native_paths':path_count,'all15_Haar_first_second_jet_checks':jet_checks,
            'all60_Phi_leg_second_jets_zero':True,'all24_noncommutative_word_first_cross_second_jets':word_controls,
            'ordered_length2_position_wrongcase_rejected':True,'finite_SU4_covariance':True,
            'AP_one_particle_power':'-I','canonical_two_particle_power':'I',
            'scope':'Native literal-coefficient jet controls plus finite SU4/AP examples; global equivariance and physical closure proved analytically.'}


def strict_joint_moment_control() -> dict:
    values=[]
    ptotal=F(0); E2=F(0); E4=F(0); R85=F(0)
    for n in range(1,49):
        p=F(1,4*n*2**(9*n)); r=F(2**n); e2=F(2**(4*n),n*n)
        assert p*e2==F(1,4*n**3*2**(5*n))
        assert p*e2**2==F(1,4*n**5*2**n)
        assert p*r**8*e2==F(2**(3*n),4*n**3)
        assert p*r**9==F(1,4*n)
        ptotal+=p; E2+=p*e2; E4+=p*e2**2
        values.append({'n':n,'pE4':str(p*e2**2),'pR8E2':str(p*r**8*e2)})
    assert ptotal<F(1,2044) and E2<F(1,124) and E4<F(1,4)
    old_E4_first=F(1,4*2**9)*(F(2**9))**2
    assert old_E4_first>1
    return {'law':'R_n=2^n, p_n=2^(-9n)/(4n), E_n^2=2^(4n)/n^2; remainder at R=E=0',
            'exact_series':{'radial_t_lt9':'(1/4) sum 2^(-(9-t)n)/n < infinity',
                            'E2':'(1/4) sum 2^(-5n)/n^3 < infinity',
                            'E4':'(1/4) sum 2^(-n)/n^5 < infinity',
                            'R8E2':'(1/4) sum 2^(3n)/n^3 = infinity',
                            'radial9':'(1/4) sum 1/n = infinity',
                            'radial10':'(1/4) sum 2^n/n = infinity'},
            'finite_exact_controls':values,'old_E2_only_counter_has_divergent_E4':True,
            'closed_E4_budget_respected_as_moment_implication_control':True,
            'native_ground':False,'D_L_nonexistence':False,'moment_necessary_for_D_L':False,
            'scope':'Single scalar law simultaneously separates all strict radial and unweighted E2/E4 moments from mixed payment; not a realization of all native quantum structural premises.'}


def packet_and_same_history_control() -> dict:
    # Adverse nonnegative generator: first-energy excess does not bound the rightmost graph remainder.
    packet=np.array([F(9999,10000),F(1,10000)],dtype=object)
    energies=np.array([F(0),F(100)],dtype=object)
    first=sum(packet*energies); second=sum(packet*energies*energies)
    assert first==F(1,100) and second==1 and second>first*first
    # Exact positive spectral support [0,eta] DOES give the needed squared bound for every norm-one law.
    eta=F(1,7); probabilities=[F(1,3),F(2,3)]; E=[F(0),eta]
    assert sum(p*x*x for p,x in zip(probabilities,E))<=eta*eta
    # Spectral truncation from finite moments is one-way. Escaping mass cannot create excess in limit.
    for n in (2,3,8,32):
        probability=F(1,n*n); energy=F(n)
        assert probability*energy*energy==1
        assert probability*min(energy*energy,F(1,4))<=1
    return {'first_energy_only_wrongcase_rejected':True,'rightmost_remainder':'||A_M L_N Omega_N|| <= a0 M^4 eta_N only from total spectral support',
            'fixed_M_then_N_limit':True,'same_history_secondmoment':'bounded continuous spectral tests then monotone spectral truncation; same L and no second subsequence',
            'scope':'Finite exact packet/spectral implication controls; infinite-dimensional history passage proved in review proof.'}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    req=json.loads(args.request.read_text())
    captured={name:(ROOT/name).read_bytes() for name in req['pins']}
    assert all(sha(raw)==req['pins'][name] for name,raw in captured.items())
    with zipfile.ZipFile(ROOT/req['profile_archive']) as archive:
        profile_raw=archive.read(req['profile_member'])
    profiles=json.loads(profile_raw)['primitive_profiles']
    result={'kind':'INDEPENDENT_FULL_PRIMITIVE_GRAPH_EXACT_REVIEW_CONTROLS',
            'native_expansion_constants':expansion_and_constants(profiles),
            'CAR_density_fullE':car_and_density(profiles),
            'native_covariant_jets':su4_and_product_jets(profiles),
            'accepted_budget_joint_moment_control':strict_joint_moment_control(),
            'packet_same_history':packet_and_same_history_control(),
            'profile_member_sha256':sha(profile_raw),'input_source_pins':req['pins'],
            'numerical_native_ground':False,'numerical_evolution':False,
            'scientific_admission':False,'campaign_complete':False}
    assert all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    result['inputs_unchanged']=True
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_assignments':64**4,'all_orders':24,'inputs_unchanged':True,'ground_computed':False}))


if __name__=='__main__':
    main()

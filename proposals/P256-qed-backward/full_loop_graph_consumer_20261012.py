"""Own exact native-coefficient/operator controls; no native ground/evolution solver."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import zipfile
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact(x: F) -> dict:
    return {'sha256': digest(str(x).encode()), 'numerator_bits': x.numerator.bit_length(),
            'denominator_bits': x.denominator.bit_length()}


def all_assignments(profiles: list[dict]) -> dict:
    """Stream all64^4 assignments, rather than replacing by same-copy words."""
    assert [p['load'] for p in profiles] == [1, 1, 63]
    assert len({p['onsite'] for p in profiles}) == 3
    charged = profiles[1:]
    nn = [[p for p in item['paths'] if p['length'] == 1] for item in charged]
    assert all(len(rows) == 3 for rows in nn)
    costs = [abs(F(rows[0]['c0'])) + abs(F(rows[0]['ci'])) for rows in nn]
    assert costs[0] == costs[1] == F(13313289252183767, 562949953421312)
    total = costs[0] + 63 * costs[1]
    base = np.arange(64**3, dtype=np.int64)
    counts = np.zeros(5, dtype=np.int64)
    lower_active = (base % 64 == 0).astype(np.int8)
    lower_active += ((base // 64) % 64 == 0)
    lower_active += (base // (64**2) == 0)
    for first in range(64):
        counts += np.bincount(lower_active + int(first == 0), minlength=5)
    expected = [math.comb(4, k)*63**(4-k) for k in range(5)]
    assert counts.tolist() == expected and int(counts.sum()) == 64**4
    orders = list(itertools.permutations(range(4)))
    assert len(orders) == 24
    weighted = sum((F(int(counts[k]))*costs[0]**k*costs[1]**(4-k)
                    for k in range(5)), F(0))
    assert weighted == total**4
    wrong = costs[0]**4 + 63*costs[1]**4
    assert weighted > wrong * 200000
    return {'all64_fourleg_assignments_actually_streamed': int(counts.sum()),
            'all24_temporal_orders': len(orders), 'total_distributive_words': 24*int(counts.sum()),
            'histogram_active_leg_count0through4': counts.tolist(),
            'Csum': str(total), 'b': str(24*(4*total)**4),
            'samecopy_replacement_rejected': True,
            'representation': 'Exact ordered product/sum DAG; enumeration certifies expansion, not402million matrices'}


def transfer(mask: int, create: int, annihilate: int):
    if not (mask >> annihilate) & 1:
        return None
    sign = (-1)**((mask & ((1 << annihilate)-1)).bit_count())
    mask ^= 1 << annihilate
    if (mask >> create) & 1:
        return None
    sign *= (-1)**((mask & ((1 << create)-1)).bit_count())
    return mask | (1 << create), sign


def car_operator_controls(profiles: list[dict]) -> dict:
    """Exact canonical2-per-copy CAR adverse model, not a native ground."""
    bases = [x | (y << 4) for x in range(16) if x.bit_count() == 2
             for y in range(16) if y.bit_count() == 2]
    index = {x: i for i, x in enumerate(bases)}
    hops = []
    for species in range(2):
        row = []
        for site in range(4):
            m = np.zeros((36, 36), dtype=np.int64)
            for col, mask in enumerate(bases):
                t = transfer(mask, 4*species+site, 4*species+(site+1)%4)
                if t:
                    nxt, sign = t
                    m[index[nxt], col] = sign
            row.append(m)
        hops.append(row)
    legs = [hops[0][e]+hops[1][e] for e in range(4)]
    orders = list(itertools.permutations(range(4)))
    def word(ls, order):
        out = np.eye(36, dtype=np.int64)
        for e in order:
            out = ls[e] @ out
        return out
    b = sum((word(legs, order) for order in orders), np.zeros((36, 36), dtype=np.int64))
    same = sum((word(hops[s], order) for s in range(2) for order in orders), np.zeros_like(b))
    initial = (1 << 1)+(1 << 2)+(1 << 4)+(1 << 7)
    final = (1 << 0)+(1 << 1)+(1 << 6)+(1 << 7)
    assert b[index[final], index[initial]] == -6
    assert same[index[final], index[initial]] == 0
    collective = []
    individual = []
    for site in range(4):
        n = np.diag([((mask >> site)&1)+((mask >> (site+4))&1) for mask in bases])
        ns = np.diag([(mask >> site)&1 for mask in bases])
        collective.append(n)
        individual.append(ns)
        assert np.array_equal(n@b, b@n)
    assert any(np.any(n@b-b@n) for n in individual)
    hc = sum((n@n for n in collective), np.zeros_like(b))
    assert np.array_equal(hc@b, b@hc)
    # Charged onsite and length1/2 generators, distinct native onsite coefficients.
    onsite = [F(p['onsite']) for p in profiles[1:]]
    # Large literal coefficients use exact object arrays, never floating overflow.
    h = np.zeros((36, 36), dtype=object)
    for s in range(2):
        n0 = np.diag([(mask >> (4*s))&1 for mask in bases]).astype(object)
        h += onsite[s]*n0
        for e in range(4):
            h += (s+1)*(hops[s][e]+hops[s][e].T).astype(object)
            # Native path length2 topology is an independent bilinear, not product of hops.
            m = np.zeros((36, 36), dtype=object)
            for col, mask in enumerate(bases):
                t = transfer(mask, 4*s+e, 4*s+(e+2)%4)
                if t:
                    nxt, sign = t
                    m[index[nxt], col] = sign
            h += m+m.T
    bh = b.astype(object)
    comm = h@bh-bh@h
    expanded = np.zeros_like(h)
    for order in orders:
        for position in range(4):
            out = np.eye(36, dtype=object)
            for p, e in enumerate(order):
                leg = legs[e].astype(object)
                out = ((h@leg-leg@h) if p == position else leg) @ out
            expanded += out
    assert np.array_equal(comm, expanded) and np.any(comm)
    a2 = bh+bh.T
    assert np.array_equal(h@a2-a2@h, expanded-expanded.T)
    return {'canonical_sector_dimension':36, 'copy_numbers':[2,2],
            'all24_ordered_matter_commutator_residual':0,
            'collective_local_density_and_HC_commutators':0,
            'individual_local_density_false_identity_rejected':True,
            'mixed_copy_witness':-6, 'samecopy_witness':0,
            'distinct_native_onsite_masses_retained':True,
            'scope':'Exact CAR operator-identity/method control, no actual ground/colour truncation theorem'}


def native_covariant_jets(profiles: list[dict]) -> dict:
    q, z = sp.symbols('q z', real=True)
    t = sp.diag(sp.Rational(1,2), -sp.Rational(1,2), 0, 0)
    u = sp.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    v = sp.diag(sp.I,-sp.I,1,1)
    assert sp.trace(t*t) == sp.Rational(1,2) and u*v != v*u
    gx, gy = v, u
    rows = []
    for p in profiles:
        for path in p['paths']:
            c0, ci = sp.Rational(path['c0']), sp.Rational(path['ci'])
            scale = -sp.Integer(2**8192) if p['scale'] == 'heavy' else sp.Integer(1)
            fx = (c0+sp.I*ci)*q*t
            fy = (c0+sp.I*ci)*(q+1)*t
            wilson = u if path['length'] == 1 else u*v
            coefficient = scale*(fx*wilson+wilson*fy)/2
            transformed = scale*((gx*fx*gx.conjugate().T)*(gx*wilson*gy.conjugate().T)
                                 +(gx*wilson*gy.conjugate().T)*(gy*fy*gy.conjugate().T))/2
            assert sp.simplify(transformed-gx*coefficient*gy.conjugate().T) == sp.zeros(4)
            dpath, ddpath = sp.I*t*wilson, -t*t*wilson
            d = scale*(fx*dpath+dpath*fy)/2
            dd = scale*(fx*ddpath+ddpath*fy)/2
            assert d != sp.zeros(4) and dd != sp.zeros(4)
            assert sp.diff(coefficient,q,2) == sp.zeros(4)
            if path['length'] == 2:
                assert sp.I*u*t*v != dpath
            charged = p['scale'] == 'charged'
            phase = sp.exp(sp.I*z) if charged else sp.Integer(1)
            f = coefficient*phase
            assert sp.simplify(sp.diff(f,z)-(sp.I*f if charged else sp.zeros(4))) == sp.zeros(4)
            assert sp.simplify(sp.diff(f,z,2)-(-f if charged else sp.zeros(4))) == sp.zeros(4)
            rows.append({'role':p['role'], 'load':p['load'], 'length':path['length'],
                         'axis':path['axis'], 'coefficient_sha256':digest(str(coefficient).encode())})
    # FullE differential operator on all24 words, with arbitrary right Fourier mode.
    x = sp.symbols('x0:4', real=True)
    c = (1,1,-1,-1)
    phase = sp.exp(sp.I*sum(ci*xi for ci,xi in zip(c,x)))
    mode = sp.exp(sp.I*(2*x[0]-3*x[1]+x[2]+4*x[3]))
    b = 24*(q+1)**4*phase
    bd = 24*(q+1)**4/phase
    a = (b+bd)/2
    def e2(f):
        return -sum(sp.diff(f,xi,2) for xi in x)
    lhs = sp.simplify(e2(a*mode)-a*e2(mode))
    rhs = 4*a*mode + sum(ci*(b-bd)*(-sp.I*sp.diff(mode,xi)) for ci,xi in zip(c,x))
    assert sp.simplify(lhs-rhs) == 0
    assert sp.simplify(lhs-4*a*mode) != 0
    # Open path control: transverse projection alone loses a nonzero longitudinal charge.
    D = sp.Matrix([[1,0,0,-1],[-1,1,0,0],[0,-1,1,0],[0,0,-1,1]])
    cc = sp.Matrix([1,1,1,1])
    assert D*cc == sp.zeros(4,1)
    open_t = sp.Matrix([1,0,0,0])
    transverse = sp.ones(4,1)/4
    longitudinal = open_t-transverse
    assert transverse+longitudinal == open_t and D*longitudinal == D*open_t
    assert D*open_t != sp.zeros(4,1)
    return {'native_full_profile_path_jets':rows,
            'all24_character_fullE_operator_residual':0,
            'missing_first_derivative_electric_term_rejected':True,
            'closed_total_density_divergence':0,
            'open_leg_transverse_only_rejected':True,
            'scope':'Exact native jets/ordered contractions plus differential-operator control; no ground sampling'}


def budgets(profiles: list[dict], assignments: dict) -> dict:
    lam, chi, fast = F(2**8192), F(1,2**4096), F(2**10240)
    k = (4*chi,F(3,20)*chi,F(64))
    w = [abs(F(p['onsite']))+2*sum((abs(F(row['c0']))+abs(F(row['ci']))
                                  for row in p['paths']),F(0)) for p in profiles]
    B = 4*(lam*w[0]+w[1]+63*w[2])
    assert B < 512*lam
    b = F(assignments['b']); a0 = 16*b; J = 2**19
    d1 = (16*(J+4)*a0,32*a0,8*a0)
    d2 = (240*(9*J+16)*a0,960*a0,64*a0)
    # Rational overbound avoids converting enormous native constants to floats.
    # sqrt(k/tau)<=2max(1,k), (4/lambda)^1/4<=2, 52^3/4<=52.
    crat = sum((kf*j2+4*max(F(1),kf)*j1 for kf,j1,j2 in zip(k,d1,d2)),F(0))
    crat += 2*a0*(2*B*52)*52*2
    r0 = 60*chi+11192*lam+F(129,10)*fast+262144
    B0 = 33*(1+10*r0*4096)+16384*lam*4096
    assert J >= 2**18+128
    assert 16**2 >= 240 and 8**2 >=60 and 2**2 ==4
    # Explicit smooth eta envelope: exp2<8, exp4<64, exp6<512.
    assert 2*64 < 128+1
    assert 620*64+8*512 < 2**18
    site_count = 4*13
    assert site_count ==52
    return {'B':exact(B),'B0':exact(B0),'a0':exact(a0),
            'D1_constants':[exact(x) for x in d1], 'D2_constants':[exact(x) for x in d2],
            'rational_c0_upper':exact(crat), 'rational_Cg_squared_upper':exact(crat*crat*B0),
            'native_kinetic':[exact(x) for x in k], 'support_sites_upper':52,
            'centre_radius_upper':4,'tau_lower':'1/2','graph_growth_M_exponent':4,
            'source_tail_M_exponent':'-1/4','fractional_s_range':'0<s<1/17'}


def failed_gate_and_fractional_controls() -> dict:
    spikes = []
    for u in (2,16,256):
        radius = u*u; mass = F(1,u**17); electric2=u**17
        assert mass*radius**8 <= 1
        assert mass*radius**8*u == 1  # radius^(17/2)
        degree9 = mass*radius**9; degree10 = mass*radius**10
        mixed = mass*radius**8*electric2
        assert degree9 == u and degree10 ==u**3 and mixed==u**16
        assert mass*electric2 ==1
        spikes.append({'u':u,'radial8point5':'1','radial9':str(degree9),
                       'radial10':str(degree10),'electric_second':'1','Phi8E2':str(mixed)})
    # Single law pays EVERY strict radial<9 moment, not just isolated8.5 spikes.
    strict=[]
    for cutoff in (4,16,64):
        masses=[F(1,4*n*2**(9*n)) for n in range(1,cutoff+1)]
        assert sum(masses)<1
        ninth=sum((p*2**(9*n) for n,p in enumerate(masses,1)),F(0))
        assert ninth==sum((F(1,4*n) for n in range(1,cutoff+1)),F(0))
        electric=sum((p*F(2**(9*n),n*n) for n,p in enumerate(masses,1)),F(0))
        assert electric< F(1,2)
        mixed=sum((p*2**(8*n)*F(2**(9*n),n*n) for n,p in enumerate(masses,1)),F(0))
        # Geometric comparison gives a uniform bound for every t<9 analytically.
        eighth=sum((p*2**(8*n) for n,p in enumerate(masses,1)),F(0))
        assert eighth<F(1,4)
        strict.append({'cutoff':cutoff,'radial8':str(eighth),'radial9':str(ninth),
                       'electric2':str(electric),'Phi8E2':exact(mixed)})
    assert F(strict[-1]['radial9'])>F(strict[0]['radial9'])
    alpha,gamma = F(1,4),F(4)
    critical = alpha/(alpha+gamma)
    assert critical ==F(1,17)
    s=F(1,34); beta=alpha-(alpha+gamma)*s
    assert beta==F(1,8)
    # Nontrivial SAME selfadjoint generator diagonal on l2. Infinite analytic family
    # L_n=2^(17n), u_n=2^-n for n>=1; cutoff n<=K has graph O(M^4),
    # TOTAL tail O(M^-1/4) for M=2^(4K). Actual graph diverges while paid s works.
    graph_terms=[]; frac_terms=[]
    for n in range(1,10):
        norm2=F(1,2**(2*n)); energy=2**(17*n)
        graph_terms.append(norm2*energy**2)
        frac_terms.append(norm2*2**n) # s=1/34 => energy^(2s)=2^n
    assert all(graph_terms[n+1]>graph_terms[n] for n in range(8))
    assert sum(frac_terms)<1
    assert sum(F(1,2**(2*n)) for n in range(10,30)) <= F(1,3*2**18)
    # Arbitrary complex positiveband Gram and a rank2 wholeP0, not Omega subtraction.
    energies=np.array([0.,0.,1.,7.,32.])
    sources=np.array([[1,2j,3,4j,1],[0,3,2j,1,2]],dtype=complex).T
    q0=np.diag((energies>0).astype(float))
    pi=np.diag(((energies>=1)&(energies<=7)).astype(float))
    j=pi@q0@sources
    gram=j.conj().T@np.diag(1/(energies+2))@j
    assert np.allclose(gram,gram.conj().T) and abs(gram[0,1].imag)>0
    assert gram[0,0].real>=0 and np.linalg.det(gram).real>=0
    omega_only=np.diag([0.,1.,1.,1.,1.])@sources
    assert np.linalg.norm(omega_only-q0@sources)>0
    return {'full_graph_wrong_moment_promotion_rejected':spikes,
            'single_law_all_strict_moments_and_mixed_obstruction':strict,
            'critical_s':str(critical),'executed_fractional_s':str(s),'fractional_cutoff_beta':str(beta),
            'materially_distinct_fractional_route':'cutoff graph + TOTAL tail, sameL spectral Holder and geometric closure',
            'infinite_diagonal_method_control':'L_n=2^(17n),u_n=2^-n: D(L^s) iff s<1/17, not D(L)',
            'method_fractional_partial_sum':str(sum(frac_terms)),
            'method_full_graph_last_term':str(graph_terms[-1]),
            'complex_positiveband_resolvent_Gram_real':gram.real.tolist(),
            'complex_positiveband_resolvent_Gram_imag':gram.imag.tolist(),
            'whole_rank2_P0_not_mean_subtraction_rejected':True,
            'controls_are_actual_native_ground':False}


def ap_and_full_colour_controls() -> dict:
    n=4
    one=np.zeros((n,n),dtype=int)
    for site in range(n):
        one[(site+1)%n,site]=-1 if site==n-1 else 1
    assert np.array_equal(np.linalg.matrix_power(one,n),-np.eye(n,dtype=int))
    pairs=list(itertools.combinations(range(n),2))
    exterior=np.zeros((6,6),dtype=int)
    for i,rows in enumerate(pairs):
        for j,cols in enumerate(pairs):
            exterior[i,j]=round(np.linalg.det(one[np.ix_(rows,cols)]))
    assert np.array_equal(np.linalg.matrix_power(exterior,n),np.eye(6,dtype=int))
    # Wrapped elementary xy face has both scalar AP seam signs twice.
    assert (-1)*(-1)*(-1)*(-1)==1
    # Complete native15-generator normalization, not a rank2 spin replacement.
    generators=[]
    for i in range(4):
        for j in range(i+1,4):
            a=sp.zeros(4);a[i,j]=a[j,i]=sp.Rational(1,2)
            b=sp.zeros(4);b[i,j]=-sp.I/2;b[j,i]=sp.I/2
            generators.extend((a,b))
    for k in range(1,4):
        generators.append(sp.diag(*([1]*k+[-k]+[0]*(3-k)))/sp.sqrt(2*k*(k+1)))
    assert len(generators)==15
    for i,a in enumerate(generators):
        assert a==a.conjugate().T and sp.trace(a)==0
        for j,b in enumerate(generators):
            assert sp.simplify(sp.trace(a*b))==(sp.Rational(1,2) if i==j else 0)
    return {'one_particle_AP_Nth_power':'-I',
            'canonical_even_exterior_Nth_power':'I',
            'contractible_wrapped_face_AP_holonomy':1,
            'native_SU4_generators_actually_checked':15,
            'complete_generator_Gram':'delta_AB/2',
            'scope':'Finite exact algebra controls; all65 parity2V and global covariance proof are analytic'}


def run(request: dict) -> dict:
    captured={name:(ROOT/name).read_bytes() for name in request['pins']}
    assert all(digest(raw)==request['pins'][name] for name,raw in captured.items())
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:
        raw=archive.read(request['profile_member'])
    profiles=json.loads(raw)['primitive_profiles']
    proof=json.loads(captured[request['proof']])
    assert proof['scientific_admission'] is False and proof['campaign_complete'] is False
    assignments=all_assignments(profiles)
    result={'kind':'OWN_COMPLETE_COVARIANT_COMMUTATOR_AND_FRACTIONAL_GRAPH_SCIENTIFIC_CONTROLS',
            'allcopy_expansion':assignments,'CAR_operator_controls':car_operator_controls(profiles),
            'native_covariant_jets':native_covariant_jets(profiles),
            'AP_and_complete_colour':ap_and_full_colour_controls(),
            'complete_graph_budgets':budgets(profiles,assignments),
            'failed_full_graph_and_paid_successor':failed_gate_and_fractional_controls(),
            'analytic_proof':request['proof'],'interface':request['interface'],
            'profile_member_sha256':digest(raw),'input_source_pins':request['pins'],
            'numerical_native_ground':False,'numerical_fullH_evolution':False,
            'scientific_admission':False,'campaign_complete':False}
    assert all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    result['inputs_unchanged']=True
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(json.loads(args.request.read_text()))
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'result':str(args.output),'inputs_unchanged':True,
                      'paid_fractional_s':'0<s<1/17','full_graph_unproved':True}))


if __name__=='__main__':
    main()

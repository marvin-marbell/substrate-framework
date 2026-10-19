"""Native full64 CAR source consumers, not a fullH ground or spectral backend.

All canonical traces below include 256/128 (or 4N^3/2N^3) modes per
species. No old producer is imported. Conditional traces are not ground data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, cast
import zipfile

import sympy as sp

ROOT = Path(__file__).resolve().parent
ORDERS = tuple(itertools.permutations(range(4)))


def submasks(mask):
    sub = mask
    while sub:
        yield sub
        sub = (sub - 1) & mask


def polyadd(a, b, factor=F(1)):
    out = dict(a)
    for degree, value in b.items():
        out[degree] = out.get(degree, F(0)) + factor * value
        if not out[degree]:
            del out[degree]
    return out


def polymul(a, b):
    out = defaultdict(F)
    for i, x in a.items():
        for j, y in b.items():
            out[i+j] += x*y
    return dict(out)


def canonical_word_key(edges):
    # Only cyclic temporal relabelling, valid for the normalized trace.
    candidates = []
    for shift in range(len(edges)):
        word = edges[shift:] + edges[:shift]
        labels = {}
        renamed = []
        for a, b in word:
            labels.setdefault(a, len(labels)); labels.setdefault(b, len(labels))
            renamed.append((labels[a], labels[b]))
        candidates.append(tuple(renamed))
    return min(candidates)


@lru_cache(maxsize=None)
def one_species_moments(edges, modes):
    """Exact canonical coherent determinant with nilpotent chronological slots.

    det(I+z product_j(I+t_j h_j))=(1+z)^M det(I+w X),
    w=z/(1+z). Each h_j is a directed native diagonal-colour leg with
    G=diag(3,-1,-1,-1); h_j^2=0. Extracting z^k pays ALL spectators.
    """
    slots = len(edges); size = 1 << slots
    x = {}
    for mask in range(1, size):
        selected = [j for j in range(slots) if mask >> j & 1]
        a, b = edges[selected[0]]; degree = 1
        for j in selected[1:]:
            c, d = edges[j]
            if d != a:
                break
            a = c; degree += 1
        else:
            x[mask] = {(a, b, degree): 1}
    power = x
    logs = [dict() for _ in range(size)]
    for r in range(1, slots + 1):
        for mask, blocks in power.items():
            trace = sum(value*(3**degree+3*(-1)**degree)
                        for (a,b,degree),value in blocks.items() if a == b)
            if trace:
                logs[mask][r] = logs[mask].get(r,F(0)) + F((-1)**(r+1)*trace,r)
        if r == slots:
            break
        nxt = {}
        for left, blocks in power.items():
            for right, legblocks in x.items():
                if left & right:
                    continue
                target = nxt.setdefault(left|right, defaultdict(int))
                for (a,b,degree), value in blocks.items():
                    for (c,d,p), other in legblocks.items():
                        if b == c:
                            target[(a,d,degree+p)] += value*other
        power = {mask: dict(blocks) for mask,blocks in nxt.items() if blocks}
    determinant = [dict() for _ in range(size)]
    determinant[0] = {0:F(1)}
    for mask in range(1,size):
        bit = mask & -mask
        for sub in submasks(mask):
            if sub & bit and logs[sub]:
                determinant[mask] = polyadd(determinant[mask], polymul(logs[sub], determinant[mask^sub]))
    k = modes//2
    ratios = [F(1)]
    for r in range(1,slots+1):
        ratios.append(ratios[-1]*F(k-r+1,modes-r+1))
    return tuple(sum(value*ratios[r] for r,value in polynomial.items()) for polynomial in determinant)


def convolution(a,b):
    out = [F(0)]*len(a)
    for mask in range(len(a)):
        if a[0] and b[mask]:
            out[mask] += a[0]*b[mask]
        for sub in submasks(mask):
            if a[sub] and b[mask^sub]:
                out[mask] += a[sub]*b[mask^sub]
    return tuple(out)


def species_power(one, number):
    result = (F(1),)+(F(0),)*(len(one)-1)
    base = one
    while number:
        if number & 1:
            result = convolution(result,base)
        number >>= 1
        if number:
            base = convolution(base,base)
    return result


@lru_cache(maxsize=None)
def all64_trace(edges,modes):
    key = canonical_word_key(edges)
    one = one_species_moments(key,modes)
    return species_power(one,64)[-1]


def active_commutator_trace(edges,modes,first_vertex,second_vertex):
    """Marked active copy +63 separately canonical auxiliary factors."""
    one = one_species_moments(tuple(edges),modes)
    rest = species_power(one,63)
    full = len(one)-1
    total = F(0)
    for mask in range(len(one)):
        d1 = sum((int(a==first_vertex)-int(b==first_vertex))
                 for j,(a,b) in enumerate(edges[:4]) if mask >> j & 1)
        d2 = sum((int(a==second_vertex)-int(b==second_vertex))
                 for j,(a,b) in enumerate(edges[4:],4) if mask >> j & 1)
        total += one[mask]*rest[full^mask]*d1*d2
    return total


def face(x,axes,n):
    y=list(x); y[axes[0]]=(y[axes[0]]+1)%n
    z=list(y); z[axes[1]]=(z[axes[1]]+1)%n
    w=list(x); w[axes[1]]=(w[axes[1]]+1)%n
    points=(tuple(x),tuple(y),tuple(z),tuple(w))
    return tuple((points[(j+1)%4],points[j]) for j in range(4))


def sym_words(edges):
    for order in ORDERS:
        yield tuple(edges[j] for j in order)
        yield tuple((edges[j][1],edges[j][0]) for j in reversed(order))


@lru_cache(maxsize=None)
def face_covariance(first,second,modes):
    # A=(sum24 W + sum24 W*)/2, keeping complex directions separately.
    total = sum(all64_trace(a+b,modes) for a in sym_words(first) for b in sym_words(second))/4
    mean_a = sum(all64_trace(a,modes) for a in sym_words(first))/2
    mean_b = sum(all64_trace(b,modes) for b in sym_words(second))/2
    return total-mean_a*mean_b


def covariance_pattern(first,second):
    labels={}; out=[]
    for edges in (first,second):
        new=[]
        for a,b in edges:
            labels.setdefault(a,len(labels)); labels.setdefault(b,len(labels))
            new.append((labels[a],labels[b]))
        out.append(tuple(new))
    return tuple(out)


def spatial_transfer():
    rows=[]
    for n in (4,6,8):
        modes=4*n**3; origin=(0,0,0)
        # q along z: yz and zx are the two transverse magnetic faces.
        polarizations=((1,2),(2,0))
        profiles=[]
        for axes in polarizations:
            first=face(origin,axes,n)
            correlations=[]
            for displacement in itertools.product(range(n),repeat=3):
                second=face(displacement,axes,n)
                pattern=covariance_pattern(first,second)
                correlations.append((displacement,face_covariance(*pattern,modes)))
            # Use exact polynomial in the primitive Nth root, not float FFT.
            root=sp.exp(2*sp.pi*sp.I/n)
            response=sum(sp.Rational(value.numerator,value.denominator)*root**d[2]
                         for d,value in correlations)
            response=sp.simplify(sp.expand_complex(response))
            zero=sum(v for _,v in correlations)
            assert response.is_real and response > 0
            profiles.append({'axes':list(axes),'q_index':[0,0,1],
                             'normalized_fourier_covariance_over_c0_pow8':str(response),
                             'zero_momentum_covariance_over_c0_pow8':str(zero),
                             'distinct_spatial_patterns':len({covariance_pattern(first,face(d,axes,n)) for d,_ in correlations})})
        assert profiles[0]['normalized_fourier_covariance_over_c0_pow8']==profiles[1]['normalized_fourier_covariance_over_c0_pow8']
        rows.append({'N':n,'V':n**3,'physical_hopping':1,'normalization':'V^-1/2 sum_x exp(iq.x) A_x; canonical trace divided by product65 dimensions',
                     'two_transverse_polarizations':profiles})
    return {'rows':rows,'actual_ground':False,'whole_P0_removed':False,
            'spectral_transfer':'FAILED: this positive conditional spatial covariance has no actual SAME-L spectral measure; both gapped and continuum measures can have the identical value.'}


def wedge_action(body,k):
    states=tuple(itertools.combinations(range(body.rows),k)); index={s:i for i,s in enumerate(states)}
    out=sp.zeros(len(states))
    for j,state in enumerate(states):
        for b in state:
            p=state.index(b); rest=state[:p]+state[p+1:]
            for a in range(body.rows):
                if not body[a,b] or a in rest:
                    continue
                q=sum(x<a for x in rest); final=tuple(sorted(rest+(a,)))
                out[index[final],j]+=(-1)**(p+q)*body[a,b]
    return out,states


def complementary_native_face():
    # Actual spatial face, one conditional diagonal-colour submodule. Each of
    # ALL64 canonical copies has 2 acted modes +126 compensating spectators.
    eye=sp.eye(6); singlet=sp.zeros(36,1)
    _,states=wedge_action(sp.eye(4),2)
    for j,state in enumerate(states):
        comp=tuple(i for i in range(4) if i not in state); inv=sum(a>b for a in state for b in comp)
        singlet[6*j+states.index(comp)]=(-1)**inv
    hops=[]
    for j in range(4):
        t=sp.zeros(4);t[(j+1)%4,j]=1
        w,_=wedge_action(t,2)
        hops.append(sp.kronecker_product(w,eye)+sp.kronecker_product(eye,w))
    B=sp.zeros(36)
    for order in ORDERS:
        word=sp.eye(36)
        for j in order: word=hops[j]*word
        B+=word
    A=(B+B.H)/2
    assert all(v*singlet==sp.zeros(36,1) for v in hops)
    nbody=sp.diag(1,0,0,0);number,_=wedge_action(nbody,2)
    n=sp.kronecker_product(number,eye)
    C=sp.I*(n*A-A*n)
    signed=sp.simplify((singlet.H*C*C*singlet)[0]/(singlet.H*singlet)[0])
    assert signed>0
    # 31 untouched complementary PAIRS are annihilated by each common leg;
    # the changed pair above is therefore exactly the full64 action of C.
    return {'charged_copies':64,'complementary_pairs':32,'acted_modes_per_copy':4,
            'local_particles_per_copy':2,'compensating_spectators_per_copy':126,
            'spectator_modes':252,'all_native_colours_retained':'three unoccupied local colours; occupied compensating modes outside face',
            'full64_A_total_norm':0,'full64_active_occupation_commutator_norm2_over_native_leg_scale_pow8':str(signed),
            'scope':'conditional diagonal fullcolour native face submodule; not generic kernel or actual ground',
            'material_change':'signed active occupation-loop response is nonzero where original common loop is dark'}


def transfer_adverse():
    # Exact entire-ground-module subtraction: Omega=e0, but e1 is ALSO ground.
    L=sp.diag(0,0,2,3);Omega=sp.Matrix([1,0,0,0]);A=sp.zeros(4)
    A[1,0]=A[0,1]=1;A[2,0]=A[0,2]=sp.I;A[0,2]=-sp.I
    P0=sp.diag(1,1,0,0);u=A*Omega
    total=(u.H*u)[0];whole=(u.H*(sp.eye(4)-P0)*u)[0]
    rankone=(u.H*(sp.eye(4)-Omega*Omega.H)*u)[0]
    assert total==rankone==2 and whole==1
    # Two complex polarizations: imaginary cross is not optional.
    v=sp.Matrix([1,sp.I]);w=sp.Matrix([sp.I,-1]);gram=sp.Matrix([[(v.H*v)[0],(v.H*w)[0]],[(w.H*v)[0],(w.H*w)[0]]])
    assert gram.det()==0 and cast(Any, sp.re(gram)).det()==4
    # Uniform positive variance admits a fixed gapped atom or continuous band.
    return {'whole_ground_block':str(whole),'wrong_mean_only':str(rankone),'total':str(total),
            'complex_Gram':str(gram),'wrong_real_only_determinant':str(cast(Any, sp.re(gram)).det()),
            'spectral_countermodels':{'gapped_measure':'delta(E-1), mass1 for every q',
                                      'broad_measure':'1_[1,2](E) dE, mass1 for every q',
                                      'both_IR_atom_at_E_speed_q':False},
            'scope':'exact premise-implication controls, NOT native fullH spectrum'}


def car_transfer(mask,a,b):
    if not(mask>>b&1) or mask>>a&1:return mask,0
    sign=(-1)**(mask&((1<<b)-1)).bit_count();mask^=1<<b
    sign*=(-1)**(mask&((1<<a)-1)).bit_count()
    return mask^(1<<a),sign


def central_memory_and_allcopy():
    edges=((1,0),(2,1),(3,2),(0,3))
    # Four distinct active labels, actual copy index0..63. Other60 copies have
    # all16 face modes empty,128 exterior particles, not removed factors.
    initial=tuple(1<<j if j<4 else 0 for j in range(64))
    target=tuple(1<<((j+1)%4) if j<4 else 0 for j in range(64))
    coefficient=0;memory=[]
    for order in ORDERS:
        current=list(initial);value=1
        for e in order:
            current[e],sign=car_transfer(current[e],*edges[e]);value*=sign
        if tuple(current)==target:coefficient+=value
    assert coefficient==24 and initial!=target
    one=[]
    for mask in range(16):
        order_values=[]
        for order in ORDERS:
            current=mask;value=1
            for e in order:
                current,sign=car_transfer(current,*edges[e]);value*=sign
            order_values.append(value if current==mask else 0)
        one.append({'mask':mask,'order_values':order_values,'sum':sum(order_values),
                    'spectator_rank':str(math.comb(252,128-mask.bit_count()))})
    assert any(len(set(row['order_values']))>1 for row in one)
    assert any(row['sum']<0 for row in one)
    # Incoming flow f=c is divergence-free; transient four distinct-copy hops
    # create totalrho, final word restores it. Final closure is not legwise.
    flow=[1,1,1,1];outflow=[x+1 for x in flow]
    assert 64*sum(x*x for x in outflow)-64*sum(x*x for x in flow)==768
    wrong_empty=64*4
    return {'all24':24,'all64_pow4_assignments_per_order':64**4,
            'full_distributive_word_count':24*64**4,'distinct_fourcopy_coefficient':coefficient,
            'samecopy_target_coefficient':0,'all64_copy_ids':list(range(64)),
            'canonical_spectators':[128-m.bit_count() for m in initial],
            'central_CAR_order_memory_rows':one,
            'once_shared_incoming_flow_energy_difference':768,'wrong_zero_incoming':wrong_empty,
            'flow_normalization':64,'closed_word_collective_density_preserved':True,
            'individual_copy_local_density_preserved':False}


def native_fullcolour_paths(profiles):
    # Independent literal N4 onebody assembly, full endpoints and ordered SU4.
    n=4;sites=tuple(itertools.product(range(n),repeat=3));indices={x:i for i,x in enumerate(sites)}
    G=[sp.diag(3,-1,-1,-1),sp.zeros(4),sp.zeros(4),sp.zeros(4)]
    G[1][0,1]=1;G[1][1,0]=1;G[2][1,2]=sp.I;G[2][2,1]=-sp.I
    G[3][2,3]=2;G[3][3,2]=2
    phi=[[g*(1+sum(x)) for g in G] for x in sites]
    u=sp.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,sp.I,0],[0,0,0,-sp.I]])
    v=sp.Matrix([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    assert u.H*u==sp.eye(4) and u.det()==1 and v.det()==1 and u*v!=v*u
    links={(x,a):u if a==0 else v for x in sites for a in range(3)}
    def path_block(x,p):
        y=x;wilson=sp.eye(4);seam=1
        for _ in range(p['length']):
            wilson=wilson*links[(y,p['axis'])]
            z=list(y)
            if z[p['axis']]==n-1:seam*=-1
            z[p['axis']]=(z[p['axis']]+1)%n;y=tuple(z)
        c0=sp.Rational(p['c0']);ci=sp.I*sp.Rational(p['ci'])
        fx=c0*phi[indices[x]][0]+ci*phi[indices[x]][p['axis']+1]
        fy=c0*phi[indices[y]][0]+ci*phi[indices[y]][p['axis']+1]
        return y,seam*(fx*wilson+wilson*fy)/2,seam,wilson
    body=[];counts=[]
    for profile in profiles:
        h=sp.MutableSparseMatrix(256,256,{})
        def add(x,y,b):
            for a in range(4):
                for c in range(4):h[4*indices[x]+a,4*indices[y]+c]+=b[a,c]
        count=0
        for x in sites:
            add(x,x,sp.Rational(profile['onsite'])*phi[indices[x]][0])
            for p in profile['paths']:
                y,b,seam,wilson=path_block(x,p);add(x,y,b);add(y,x,b.H);count+=1
        assert h==h.H
        body.append(h);counts.append(count)
    delta=sp.Rational(profiles[1]['onsite'])-sp.Rational(profiles[2]['onsite'])
    difference=body[1]-body[2]
    for x in sites:
        for a in range(4):
            for b in range(4):assert difference[4*indices[x]+a,4*indices[x]+b]==delta*phi[indices[x]][0][a,b]
    assert profiles[1]['paths']==profiles[2]['paths']
    p=next(p for p in profiles[1]['paths'] if p['axis']==0 and p['length']==2)
    _,actual,seam,wilson=path_block((3,0,0),p)
    assert seam==-1 and actual!=sp.zeros(4)
    # All64 colour-band-filled determinants are EXACTLY dark at commuting
    # nonzero uniform Phi and diagonal links, yet not common SU(256) singlets.
    full_band_mask=sum(1<<(4*j+c) for j in range(64) for c in (0,1))
    assert full_band_mask.bit_count()==128
    dark_hop_count=0
    for x in sites:
        for p in profiles[1]['paths']:
            y=list(x);y[p['axis']]=(y[p['axis']]+p['length'])%4;y=tuple(y)
            for c in range(4):
                _,sign=car_transfer(full_band_mask,4*indices[x]+c,4*indices[y]+c)
                assert sign==0;dark_hop_count+=1
    return {'native_N':4,'onebody_modes':256,'all65_profile_loads':[p['load'] for p in profiles],
            'directed_primitive_counts':counts,'all_three_matrices_Hermitian':True,
            'fullcolour_noncommuting_Phi_SU4':True,'active_aux_hopping_identical':True,
            'delta':str(delta),'wrapped_length2_AP':seam,'Wilson_order_noncommuting_control':True,
            'commuting_nonzero_full64_dark_hops_tested':64*dark_hop_count,
            'commuting_dark_sector_not_PS':'collective colour0 occupancy64, not32',
            'active_uniform_onsite_filter':'scalar on each full-colour-band determinant; darkness retained',
            'Phi_zero_kernel':'whole canonical fibre',
            'scope':'exact native onebody/CAR conditional section, not actual ground measure'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_bytes())
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:raw=archive.read(request['profile_member'])
    assert hashlib.sha256(raw).hexdigest()==request['profile_member_sha256']
    profiles=json.loads(raw)['primitive_profiles']
    result={'kind':'NATIVE_FULL64_EXACT_CANONICAL_CONDITIONAL_CONTROLS_NOT_GROUND',
            'native_fullcolour_AP_and_dark':native_fullcolour_paths(profiles),
            'allcopy_and_central_memory':central_memory_and_allcopy(),
            'signed_material_successor':complementary_native_face(),
            'wholeP0_complex_and_IR_adverse':transfer_adverse(),
            'normalized_physical1_spatial_attempt':spatial_transfer(),
            'coherent_trace_cache':{'all64_word_classes':all64_trace.cache_info().currsize,'canonical_onecopy_slot_classes':one_species_moments.cache_info().currsize},
            'governor':request['governor'],'ground_numeric':False,'scientific_admission':False,'campaign_complete':False}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'receipt':args.output.name,'conditional_native_controls_completed':True,'actual_ground_transfer':'UNPAID'}))


if __name__=='__main__':main()

"""Independent charged-floor review: archived witnesses, no producer import or solve.

Integer interval reconstruction is followed by division-free characteristic
polynomials, not the producer's Bareiss inertia or the parent's Fraction LDL.
All outputs are exclusive, epoch-prefixed artifacts. This is a scoped scientist
CLI, not a fullPhi, ground, phase, or all-order acceptance checker.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import sys
import time
import zipfile
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import sympy as s

ROOT = Path(__file__).resolve().parent
Q = 1 << 100
sys.set_int_max_str_digits(0)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def exclusive(path, value):
    with path.open('x') as out:
        json.dump(value, out, indent=2)
        out.write('\n')


def ceildiv(a, b):
    return -((-a)//b)


def fixed(value):
    f = F(value)
    return f.numerator*Q//f.denominator, ceildiv(f.numerator*Q, f.denominator)


def plus(x, y):
    return x[0]+y[0], x[1]+y[1]


def minus(x, y):
    return x[0]-y[1], x[1]-y[0]


def times(x, y):
    endpoints = [a*b for a in x for b in y]
    return min(endpoints)//Q, ceildiv(max(endpoints), Q)


def integer(x, n):
    return (x[0]*n, x[1]*n) if n >= 0 else (x[1]*n, x[0]*n)


def reciprocal(x):
    assert x[0]*x[1] > 0
    return Q*Q//x[1], ceildiv(Q*Q, x[0])


def radical(value):
    # Direct exact-rational square root: no rounded e^2 or binary sqrt.
    f = F(value)
    assert f >= 0
    k = math.isqrt(f.numerator*Q*Q//f.denominator)
    assert k*k*f.denominator <= f.numerator*Q*Q
    upper = k+int(k*k*f.denominator != f.numerator*Q*Q)
    assert upper*upper*f.denominator >= f.numerator*Q*Q
    return k, upper


def profiles(data):
    result = []
    for p in data['profiles']:
        m, d, c, e2 = [F(p[k]) for k in ('m_exact','d_exact','c0_exact','e_squared_exact')]
        assert e2 == m*m+d*d and m > 0 and e2 > c*c
        e = radical(e2)
        E = [integer(e,-2), fixed(2*c), fixed(0), fixed(-2*c), integer(e,2)]
        r = {}
        for b in (1,3):
            for a, value in [(0,times(fixed(d),reciprocal(integer(e,2)))),
                             (2,times(fixed(m),reciprocal(times(radical(2),e)))),
                             (4,integer(times(fixed(d),reciprocal(integer(e,2))),-1))]:
                r[a,b] = r[b,a] = value
        se = [plus(integer(e,-1),fixed(c)),minus(integer(e,-1),fixed(c)),
              plus(e,fixed(c)),minus(e,fixed(c))]
        f = times(fixed(m),reciprocal(integer(e,2)))
        g = times(fixed(d),reciprocal(integer(e,2)))
        sr = {(0,1):f,(1,0):f,(2,3):f,(3,2):f,
              (0,3):integer(g,-1),(3,0):integer(g,-1),(1,2):g,(2,1):g}
        result.append((E,r,se,sr))
    return result


def enumerate_words(n, energies, budget):
    gaps = [minus(x,energies[0]) for x in energies[1:]]
    assert all(g[0] > 0 for g in gaps)
    # Cartesian excited occupations with independent lower-gap bounds; no
    # recursive producer pruning and no numerical energy decision.
    bounds = [range(min(n,budget*Q//g[0])+1) for g in gaps]
    result = {}
    for tail in itertools.product(*bounds):
        if sum(tail) > n:
            continue
        cost = fixed(0)
        for g, k in zip(gaps,tail):
            cost = plus(cost,integer(g,k))
        if cost[0] <= budget*Q:
            result[(n-sum(tail),)+tail] = cost
    return result


def polynomial_positive(matrix):
    assert matrix == [list(row) for row in zip(*matrix)]
    # Berkowitz characteristic polynomial uses no divisions or pivot signs.
    # det(t I + M) has positive coefficients => no root t>=0. Symmetry
    # makes all eigenvalues real; constant>0 excludes zero, so M>0.
    n = len(matrix)
    coeff = [int(x) for x in s.Matrix(matrix).charpoly().all_coeffs()]
    plus_coeff = [(-1)**j*x for j,x in enumerate(coeff)]
    assert len(plus_coeff) == n+1 and all(x > 0 for x in plus_coeff)
    return {'method':'division-free_Berkowitz_det(tI+M)_positive_coefficients',
            'dimension':n,'coefficients_exact':[str(x) for x in plus_coeff],
            'all_coefficients_strictly_positive':True,
            'polynomial_sha256':sha(json.dumps(plus_coeff,separators=(',',':')).encode())}


def review_branch(branch, profile, epoch):
    counts, spin = ((1,126),0) if branch['branch']=='active' else ((2,125),1)
    floor = F(branch['floor_exact']); assert floor == -5120
    archived = json.loads((ROOT/branch['exact_matrix_path']).read_bytes())
    assert archived['bits'] == 100
    k = len(archived['lower_integer']); assert k == 18
    original_positive = polynomial_positive(archived['lower_integer'])
    center = archived['center_integer']; radius = archived['radius_integer']
    for t in range(k):
        for u in range(k):
            assert archived['lower_integer'][t][u] == center[t][u]-(radius if t==u else 0)
            assert archived['upper_integer'][t][u] == center[t][u]+(radius+64*archived['residual_squared_upper_integer'] if t==u else 0)
    with np.load(ROOT/branch['witness_path'],allow_pickle=False) as witness:
        states = [tuple(int(x) for x in row) for row in witness['states']]
        output = [int(x) for x in witness['outputs']]
        negative = [int(x) for x in witness['negative']]
        Y = [[fixed(float(x)) for x in row] for row in witness['Y']]
    assert len(states)==branch['input_support'] and len(output)==branch['output_rows']
    assert len(set(states))==len(states) and len(Y)==len(output) and all(len(row)==k for row in Y)
    base = plus(plus(integer(profile[0][0][0],counts[0]),integer(profile[1][0][0],counts[1])),profile[spin][2][0])
    words = [enumerate_words(n,p[0],branch['support_budget']) for n,p in zip(counts,profile)]
    sg = [minus(x,profile[spin][2][0]) for x in profile[spin][2]]
    enumerated = {}
    for a,ca in words[0].items():
        for b,cb in words[1].items():
            for spinor in range(4):
                cost = plus(plus(ca,cb),sg[spinor])
                if cost[0] <= branch['support_budget']*Q:
                    enumerated[a+b+(spinor,)] = cost
    assert set(states)==set(enumerated)
    cost = [enumerated[state] for state in states]
    assert output==[j for j,c in enumerate(cost) if c[1]<=branch['output_budget']*Q]
    D = [minus(plus(base,c),fixed(floor)) for c in cost]
    assert negative==[j for j,x in enumerate(D) if x[1]<0]
    positive = [j for j,x in enumerate(D) if x[0]>0]
    assert len(positive)+len(negative)==len(states)
    outside = base[0]+branch['support_budget']*Q
    assert outside > floor*Q
    index = {state:j for j,state in enumerate(states)}
    row_index = {j:r for r,j in enumerate(output)}

    def transitions(state):
        for role in (0,1):
            start = 5*role; word = state[start:start+5]
            for (dest,src),value in profile[role][1].items():
                if word[src]:
                    target = list(state); target[start+src]-=1; target[start+dest]+=1
                    yield tuple(target),times(value,radical(word[src]*(word[dest]+1)))
        for (dest,src),value in profile[spin][3].items():
            if src==state[-1]:
                yield state[:-1]+(dest,),value
        yield state,fixed(F(1,2))

    edges = 0
    for j in output:
        for key,_ in transitions(states[j]):
            assert key in index
            edges += 1
    assert edges==branch['checked_nonzero_image_edges']
    columns = [[] for _ in states]
    for j,state in enumerate(states):
        for target,value in transitions(state):
            target_index = index.get(target)
            if target_index is None:
                continue
            row = row_index.get(target_index)
            if row is not None:
                columns[j].append((row,value))
    vectors = []
    for column in columns:
        v = [fixed(0) for _ in range(k)]
        for row,b in column:
            for t in range(k):
                v[t] = plus(v[t],times(b,Y[row][t]))
        vectors.append(v)
    G = [[fixed(0) for _ in range(k)] for _ in range(k)]
    for t in range(k):
        for u in range(t,k):
            G[t][u] = plus(vectors[negative[t]][u],vectors[negative[u]][t])
    for row in Y:
        for t in range(k):
            for u in range(t,k):
                G[t][u] = minus(G[t][u],times(row[t],row[u]))
    for j in positive:
        weight = integer(reciprocal(D[j]),64)
        v = vectors[j]
        for t in range(k):
            for u in range(t,k):
                G[t][u] = minus(G[t][u],times(times(v[t],v[u]),weight))
    S = [[fixed(0) for _ in range(k)] for _ in range(k)]
    for t in range(k):
        for u in range(t,k):
            x = integer(G[t][u],64)
            if t==u:
                x = plus(x,D[negative[t]])
            S[t][u] = S[u][t] = x
    reconstructed = [[(x[0]+x[1])//2 for x in row] for row in S]
    rad = max(sum(max(mid-x[0],x[1]-mid) for mid,x in zip(row,intervals)) for row,intervals in zip(reconstructed,S))
    lower = [[value-(rad if t==u else 0) for u,value in enumerate(row)] for t,row in enumerate(reconstructed)]
    independently_positive = polynomial_positive(lower)
    out = ROOT/f'{epoch}_{branch["branch"]}_proof.json'
    exclusive(out,{'branch':branch['branch'],'original_lower_polynomial':original_positive,
                   'independent_interval_lower_polynomial':independently_positive,
                   'independent_lower_integer':lower,'independent_radius_integer':rad,
                   'direct_rational_radicals':True,'source_witness_sha256':sha((ROOT/branch['witness_path']).read_bytes()),
                   'source_matrix_sha256':sha((ROOT/branch['exact_matrix_path']).read_bytes())})
    return {'branch':branch['branch'],'status':'VALID','lambda1_floor_exact':str(floor),
            'independent_enumeration_matches_witness':True,'input_support':len(states),
            'output_rows':len(output),'whole_output_row_image_edges':edges,'negative_free_dimension':k,
            'outside_free_floor_lower_exact':str(F(outside,Q)),
            'original_and_independently_reconstructed_lower_positive':True,
            'independent_radius_upper_exact':str(F(rad,Q)),
            'proof_path':out.name,'proof_sha256':sha(out.read_bytes())}


def contrary_controls():
    R = s.Matrix([[0,s.Rational(3,10),0,-s.Rational(2,5)],
                  [s.Rational(3,10),0,s.Rational(2,5),0],
                  [0,s.Rational(2,5),0,s.Rational(3,10)],[-s.Rational(2,5),0,s.Rational(3,10),0]])
    A = R+s.eye(4)/2; P = s.diag(1,0,0,0)
    # The retained diagonal alone agrees; the omitted columns/off-diagonal
    # blocks distinguish output compression from the false input crop.
    assert A*P*A != (P*A*P)**2
    assert (A*P*A)[1,1] > ((P*A*P)**2)[1,1]
    assert A[0,1]!=0 and A[0,3]!=0
    assert 2*2*1-3*1*1 < s.Rational(4,3)
    assert s.diag(-1,2).eigenvals()=={-1:1,2:1}
    assert all(fixed(x)[0] <= F(x)*Q <= fixed(x)[1] for x in [F(-1,3),F(1,3),F(7,9)])
    for f in [F(2),F(3,7),F(100,9)]:
        lo,hi=radical(f);assert F(lo*lo,Q*Q)<=f<=F(hi*hi,Q*Q)
    for f in [F(2,3),F(-2,3)]:
        lo,hi=reciprocal(fixed(f));assert F(lo,Q)<=1/f<=F(hi,Q)
    ann=[];bad=[]
    for p in range(4):
        a=s.zeros(16); unsigned=s.zeros(16)
        for mask in range(16):
            if mask>>p&1:
                a[mask^(1<<p),mask]=(-1)**((mask&((1<<p)-1)).bit_count())
                unsigned[mask^(1<<p),mask]=1
        ann.append(a);bad.append(unsigned)
    for p,q in itertools.product(range(4),repeat=2):
        assert ann[p]*ann[q].T+ann[q].T*ann[p]==s.eye(16)*int(p==q)
    assert bad[0]*bad[1].T+bad[1].T*bad[0]!=s.zeros(16)
    # Native vector source at the identity joint frame. Different Dirac modes
    # in the same changed replica are demonstrably coherent, not orthogonal.
    V=s.zeros(16,5)
    for j,(mask,value) in enumerate([(3,1),(5,1),(9,1/s.sqrt(2)),(10,1),(12,-1)]):V[mask,j]=value
    V[6,2]=1/s.sqrt(2)
    source=V[:,1]+V[:,2]
    assert (ann[0]*source).T*(ann[1]*source)!=s.zeros(1)
    # n=4 is a fresh exact extraction/trace control, not old image-main rerun.
    def words(n):
        return [w for w in itertools.product(range(n+1),repeat=5) if sum(w)==n]
    ws,rest=words(4),words(3);idx={w:j for j,w in enumerate(rest)}
    E=s.zeros(len(rest)*5,len(ws))
    for j,w in enumerate(ws):
        for p in range(5):
            if w[p]:
                tail=list(w);tail[p]-=1;E[5*idx[tuple(tail)]+p,j]=s.sqrt(s.Rational(w[p],4))
    assert E.T*E==s.eye(len(ws))
    # Explicit harmonic tensor (isotropic v)^4 has G(v,v)=0 and survives
    # remaining traces after extraction; a nonharmonic e2^4 does not.
    v=[s.Integer(1),0,s.sqrt(2),0,s.Integer(1)]
    G=s.zeros(5);G[0,4]=G[4,0]=G[1,3]=G[3,1]=1;G[2,2]=-1
    assert (s.Matrix(v).T*G*s.Matrix(v))[0]==0
    assert (s.eye(5)[:,2].T*G*s.eye(5)[:,2])[0]!=0
    tensor=s.Matrix([s.sqrt(s.Rational(math.factorial(4),math.prod(math.factorial(k) for k in w)))*math.prod(v[j]**w[j] for j in range(5)) for w in ws])
    rest1=words(1);rest1index={w:j for j,w in enumerate(rest1)}
    T=s.zeros(len(rest1),len(rest))
    for col,w in enumerate(rest):
        for p,q in itertools.product(range(5),repeat=2):
            if G[p,q] and w[p] and w[q]-int(p==q)>0:
                tail=list(w);tail[p]-=1;tail[q]-=1
                T[rest1index[tuple(tail)],col]+=G[p,q]*s.sqrt(w[p]*(w[q]-int(p==q)))
    extracted=E*tensor
    assert s.simplify(s.kronecker_product(T,s.eye(5))*extracted)==s.zeros(len(rest1)*5,1)
    for creation in (False,True):
        endpoint=sum(((p+1+s.I*(4-p))*(ann[p].T if creation else ann[p])*V for p in range(4)),s.zeros(16,5))
        image=s.kronecker_product(s.eye(len(rest)),endpoint)*extracted
        assert s.simplify(s.kronecker_product(T,s.eye(16))*image)==s.zeros(len(rest1)*16,1)
    # Whole joint sign flip vs spinor-only flip.
    S=s.diag(1,-1,1,-1)
    assert S*R*S==-R
    Rv=s.zeros(5)
    for j in (1,3):
        for i,value in [(0,s.Rational(2,5)),(2,3*s.sqrt(2)/10),(4,-s.Rational(2,5))]:Rv[i,j]=Rv[j,i]=value
    total=s.kronecker_product(Rv,s.eye(4))+s.kronecker_product(s.eye(5),R)
    spin_only=s.kronecker_product(s.eye(5),S)
    assert spin_only*total*spin_only!=-total
    both=s.kronecker_product(s.diag(1,-1,1,-1,1),S)
    assert both*total*both==-total
    correct=64*((F(1)+F(1,2))**2+(F(2)+F(1,2))**2+1)
    assert correct==608 and correct!=64*(1+4+1) and correct+64!=correct
    # Actual signed eight-mode CAR: full-rank M=I4 hopping has a normalized
    # positive-energy Slater eigenvector with energy4, not a two-mode cap.
    ann8=[]
    for p in range(8):
        entries={}
        for mask in range(256):
            if mask>>p&1:
                entries[mask^(1<<p),mask]=(-1)**((mask&((1<<p)-1)).bit_count())
        ann8.append(s.SparseMatrix(256,256,entries))
    H=sum((ann8[p].T*ann8[p+4]+ann8[p+4].T*ann8[p] for p in range(4)),s.SparseMatrix(256,256,{}))
    slater=s.SparseMatrix(256,1,{(0,0):1})
    for p in range(4):
        slater=(ann8[p].T+ann8[p+4].T)*slater/s.sqrt(2)
    assert (slater.T*slater)[0]==1 and H*slater==4*slater
    assert 4**2==4*4 and 4**2>2*2
    labelled=s.SparseMatrix(256,1,{(3+(3<<4),0):1})
    endpoint0=ann8[0]*labelled;endpoint1=ann8[4]*labelled
    assert (endpoint0.T*endpoint1)[0]==0
    assert (endpoint0.T*endpoint0)[0]==(endpoint1.T*endpoint1)[0]==1
    # An unconstrained full-Fock source can destroy the label separation.
    unconstrained=s.SparseMatrix(256,1,{(3,0):1/s.sqrt(2),(18,0):1/s.sqrt(2)})
    assert ((ann8[0]*unconstrained).T*(ann8[4]*unconstrained))[0]!=0
    return {'output_not_input_compression':True,'wrong_G_upper_direction_rejected':True,
            'lambda2_not_lambda1_counterexample':[-1,2],
            'rational_radical_and_signed_reciprocal_controls':True,
            'unsigned_CAR_rejected':True,'within_replica_Dirac_cross_terms_nonzero':True,
            'fresh_n4_fixed_slot_isometry':True,'harmonic_vs_nonharmonic_trace_witness':True,
            'spinor_only_R_flip_rejected_whole_flip_verified':True,
            'affine_HE64_once_exact_sample':str(correct),
            'four_Dirac_modes_not_two_mode_projection':True,
            'full8mode_rank4_hop_eigenstate_energy_exact':'4',
            'labelled_crosscopy_orthogonality_and_unlabelled_counterexample':True,
            'fresh_n4_remaining_trace_after_both_signed_CAR_orientations':True}


def q_review(old,lower,upper,guard):
    floors={b['branch']:F(b['floor_exact']) for b in lower['branches']}
    E0=F(upper['ground_energy_interval_exact'][1]);assert E0==F(lower['E0_upper_exact'])
    gamma=64+floors['active']+floors['aux']-2*E0
    assert gamma==F('1249546155809/17179869184')
    roles=old['all64_boundary']['roles'];assert [(r['count'],r['spin_replicas']) for r in roles]==[(1,2),(63,126)]
    q=F(0);rows=[]
    for allocation in old['all64_boundary']['all64_once_shared_pointwise_norms']:
        axis,length=allocation['axis'],allocation['length'];weight=F(allocation['root_allocation_exact'])
        norm=F(0);gaps=[]
        for role in roles:
            p=next(p for p in role['paths'] if (p['axis'],p['length'])==(axis,length))
            mat=p['matrix_exact'];assert len(mat)==4 and all(len(row)==4 for row in mat)
            square=sum(F(re)**2+F(im)**2 for row in mat for re,im in row)
            norm+=role['count']*4*square
            assert p['external_edges_min']==(1 if axis==0 else length)
            gaps.append(64*p['external_edges_min']+2*floors['active' if role['count']==1 else 'aux']-2*E0)
        assert min(gaps)>0
        term=weight*norm/min(gaps)**2;q+=term
        rows.append({'axis':axis,'length':length,'allocation':str(weight),'weighted_inverse_square_exact':str(term)})
    assert q==F(guard['candidate_root_inverse_square_sum_exact']) and q>1
    return {'status':'CONDITIONAL','gamma_exact':str(gamma),'q_exact':str(q),'q_float':float(q),
            'guard_less_than_one':False,'path_terms':rows,
            'meaning':'Uniform full-Dirac CAR Frobenius/rank SOURCE upper bounds, not measured selected-state source weights. 4x4 M already includes both spin labels; multiply count1+63, not spin_replicas2+126 again. Within-copy interference is covered by the full hopping norm; different-copy changed canonical-number sectors are orthogonal on the stipulated source. Root allocations are path-incidence majorants, not normalized probabilities.',
            'hypotheses':['Frozen matching-product selected local canonical N_replica=2 source',
                          'Number-conserving H0 leaves each changed replica-label branch invariant',
                          'Actual full4x4 path matrices and shared field counted once',
                          'Each path electric form has certified affine cell forms plus at least external_edges_min unit bridge costs',
                          'Neutral E0 upper is subtracted only; no neutral spectral gap imported'],
            'not_proved':'Actual source-specific norms, physicalPhi lift, normalized all-order linked control, phase, pole or photon'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--epoch',required=True)
    args=parser.parse_args();assert args.epoch.startswith('charged_lower_review_') and '/' not in args.epoch
    started=time.monotonic();manifest=json.loads((ROOT/args.manifest).read_bytes());pins=manifest['input_sha256']
    captured={name:(ROOT/name).read_bytes() for name in pins}
    assert {name:sha(data) for name,data in captured.items()}==pins
    load=lambda name:json.loads(captured[name])
    lower=load('charged_output_lower_attempt3_20261011_receipt.json')
    image=load('charged_image_proof_control_20261011_receipt.json')
    assert pins['charged_output_lower_certificate_20261011.py']==lower['own_source_sha256']
    assert pins['charged_image_proof_control_20261011.py']==image['source_hash']
    for branch in lower['branches']:
        assert pins[branch['witness_path']]==branch['witness_sha256']
        assert pins[branch['exact_matrix_path']]==branch['exact_matrix_sha256']
    p=profiles(load('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json'))
    branches=[review_branch(b,p,args.epoch) for b in lower['branches']]
    controls=contrary_controls()
    q=q_review(load('charged_first_shell_actual04_20261011_receipt.json'),lower,
               load('local_so5_ground_upper_complete_state_consumer_20261011_receipt.json'),
               load('charged_rooted_source_inverse_guard_20261011_receipt.json'))
    with zipfile.ZipFile(ROOT/'charged_image_proof_attempt2_20261011_recovered_after_execution.zip') as archive:
        recovery=json.loads(archive.read('recovery_manifest.json'))
        assert recovery['all_original_preexecution_source_input_pins_match']
        assert recovery['stream_provenance']['missing']==['Original raw stdout pipe bytes','Original raw stderr pipe bytes']
    proof={'status':'VALID_SCOPED_LOWER_AND_IMAGE_CONDITIONAL_APPLICATION',
           'revision_bound_input_sha256':pins,'branches':branches,'controls':controls,'root_guard':q,
           'mathematical_review':{
             'output':'A=Rtotal+uI is self-adjoint. ||Pout A x||^2<=||A x||^2, so D+64 A Pout A is a lower FORM on the whole containing space, not PHP. Hermiticity plus the explicitly complete neighbors of every retained output row makes B vanish on omitted input columns; outside is exactly D and the independently re-enumerated lower free-cost floor exceeds -5120.',
             'enumeration':'All four excited occupations have strictly positive outward lower gaps. Cartesian product bounds include every occupation with exact cost<=402. Ground occupation is the remaining count. The joint spinor4 costs are included. Archived state/output/negative indices match independent enumeration; all uncertain-sign columns are rejected, not silently discarded.',
             'Schur':'DQ>0 and K=I+64 BQ DQ^-1 BQ*>0. The full Schur complement is DL+64 BL* K^-1 BL. For ANY real dyadic Y, BL*K^-1 BL-G(Y)=(Y-K^-1 BL)*K*(Y-K^-1 BL)>=0. Thus certifying DL+64G(Y)>0 proves a lambda1 lower bound; no excited-eigenvalue or neutral-gap inference. Neither accurate solve nor small residual is required for the LOWER proof.',
             'rounding':'Exact rational parameters, direct rational square-root brackets, binary64 witness converted to exact Fraction then outward fixed100 intervals; all products/reciprocals are outward. Symmetric row-sum radius bounds spectral error. New lower and archived lower each pass a different division-free exact characteristic-polynomial positivity proof.',
             'all_n_image':'For any individually fixed slot E_n|k>=sum_j sqrt(k_j/n)|k-e_j> tensor |j>; sum_j k_j/n=1 proves isometry for ALL n. Combinatorial permutation multiplicities prove the actual labelled embedding. Signed physical CAR A_t:W5->wedge1/3 C4 gives (I tensor A_t)E_n and hence unprojected Sym(n-1) x C4. This is arbitrary-complex-endpoint linearity, not a finite-n extrapolation or fixed-t intertwining claim. Remaining pair traces commute with selected-slot CAR and vanish for a harmonic source; selected/rest contraction is a source constraint, not a projector on the spinor output.',
             'joint_phase':'Inspected native U, exterior-square W highest-wedge minus sign, Hodge J and T3=JQU S4 bind D AND R simultaneously. Independently checked whole-vector-and-spinor R flip; flipping only the charged spinor is insufficient. Negative actual d is retained by rational interval entries; no absolute-value phase substitution.',
             'labels':'Original local spin-replica N_j=2, including entangled harmonic states. A single endpoint changes exactly one label to1/3; distinct replica targets are orthogonal. For spin-mixing hopping the ordered pair of endpoint spin labels is retained. Within one replica Dirac components remain coherent. Number-conserving H0 preserves these labels, so first-shell resolvent additivity is source-restricted, never a global sqrt(count) operator norm.',
             'affine':'Cell background is256 fermions (128 spin replicas), not2. For total changed charge q=+/-1, qL=q/2+r, qR=q/2-r. Zero-boundary Gauss flow across two cells is(-rA-q/2,-q,-rB-q/2), giving64(RA+q/2)^2+64(RB+q/2)^2+64 ONCE. The same affine sign at both endpoints is correct despite opposite total charges. Whole joint sign flip covers both orientations.'},
           'verdicts':{'containing_space_lambda1_minus5120':'VALID','fixed_slot_all_n_CAR_image':'VALID',
                       'same_H_affine_two_cell_gamma':'CONDITIONAL_ON_SELECTED_FROZEN_LOCAL_H0_AND_NEUTRAL_E0_UPPER',
                       'q43_uniform_first_shell_source_bound':'CONDITIONAL_NOT_A_PASSED_GUARD',
                       'fullPhi_lattice_ground_phase_pole':'NOT_ESTABLISHED'},
           'artifact_reconstruction_limitation':recovery['provenance']+' '+recovery['stream_provenance']['not_claimed'],
           'debts':['PhysicalPhi admission of the selected harmonic source',
                    'Actual complete source-specific CAR weights instead of uniform rank/Frobenius caps',
                    'Number/field/shared-flow higher-shell normalized all-order stability',
                    'Stationary lattice/fullPhi ground, bulk phase/stiffness, IR atom, sharp photon, commoncone, originalSOURCE'],
           'source_unchanged':sha(Path(__file__).read_bytes())==pins[Path(__file__).name],
           'inputs_unchanged':all(sha((ROOT/name).read_bytes())==pin for name,pin in pins.items()),
           'elapsed_seconds':time.monotonic()-started}
    assert proof['source_unchanged'] and proof['inputs_unchanged']
    out=ROOT/f'{args.epoch}_receipt.json';exclusive(out,proof)
    print(json.dumps({'status':proof['status'],'receipt':out.name,'receipt_sha256':sha(out.read_bytes()),
                      'branches':branches,'q_float':q['q_float'],'gamma_exact':q['gamma_exact']}),flush=True)


if __name__=='__main__':
    main()

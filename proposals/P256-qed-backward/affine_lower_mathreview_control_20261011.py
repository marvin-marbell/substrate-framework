from __future__ import annotations

import hashlib
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import sympy as s

ROOT = Path(__file__).resolve().parent
PREFIX = 'affine_lower_mathreview_epoch2_20261011'
Q = 1 << 100
ZERO = (0, 0)


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def load(name):
    return json.loads((ROOT / name).read_text())


def fixed(x):
    f = F(x) * Q
    return f.numerator // f.denominator, -((-f.numerator) // f.denominator)


def plus(a, b):
    return a[0] + b[0], a[1] + b[1]


def negative(a):
    return -a[1], -a[0]


def minus(a, b):
    return plus(a, negative(b))


def times(a, b):
    products = [x*y for x in a for y in b]
    return min(products)//Q, -((-max(products))//Q)


def scaled(a, n):
    return (a[0]*n, a[1]*n) if n >= 0 else (a[1]*n, a[0]*n)


def reciprocal(a):
    assert a[0] > 0 or a[1] < 0
    return Q*Q//a[1], -((-Q*Q)//a[0])


def radical(a):
    assert a[0] >= 0
    lo, hi = math.isqrt(a[0]*Q), math.isqrt(a[1]*Q)
    return lo, hi + int(hi*hi != a[1]*Q)


def inertia(matrix):
    a = [[F(x) for x in row] for row in matrix]
    signs = []
    for j in range(len(a)):
        pivot = a[j][j]
        assert pivot != 0
        signs.append(1 if pivot > 0 else -1)
        for r in range(j+1, len(a)):
            for c in range(j+1, len(a)):
                a[r][c] -= a[r][j]*a[j][c]/pivot
    return {'negative': signs.count(-1), 'positive': signs.count(1), 'pivot_signs': signs}


def profiles():
    out = []
    for p in load('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json')['profiles']:
        assert F(p['e_squared_exact']) == F(p['m_exact'])**2 + F(p['d_exact'])**2
        m, d, c = [fixed(p[k+'_exact']) for k in ('m', 'd', 'c0')]
        e = radical(fixed(p['e_squared_exact']))
        ve = [scaled(e,-2), scaled(c,2), ZERO, scaled(c,-2), scaled(e,2), ZERO]
        vr = {}
        for j in (1,3):
            for i,v in ((0,times(d,reciprocal(scaled(e,2)))),
                        (2,times(m,reciprocal(times(radical(fixed(2)),e)))),
                        (4,negative(times(d,reciprocal(scaled(e,2)))))):
                vr[i,j] = vr[j,i] = v
        se = [plus(negative(e),c),minus(negative(e),c),plus(e,c),minus(e,c)]
        sr = {}
        for i,j,v in ((0,1,times(m,reciprocal(scaled(e,2)))),(2,3,times(m,reciprocal(scaled(e,2)))),
                      (0,3,negative(times(d,reciprocal(scaled(e,2))))),(1,2,times(d,reciprocal(scaled(e,2))))):
            sr[i,j] = sr[j,i] = v
        out.append({'vector': (ve,vr), 'spinor': (se,sr)})
    return out


def words(n, energy, budget):
    gaps = [minus(x,energy[0]) for x in energy[:5]]
    result = []
    def visit(j, remaining, cost, tail):
        if j == 5:
            result.append(((remaining,*tail),cost))
            return
        assert gaps[j][0] > 0
        limit = min(remaining,max(0,(budget*Q-cost[0])//gaps[j][0]))
        for k in range(limit+1):
            visit(j+1,remaining-k,plus(cost,scaled(gaps[j],k)),tail+[k])
    visit(1,n,ZERO,[])
    return result


def formation(family, pp):
    kind, roles = family['kind'], family['roles']
    targets = [(r,'vector' if kind=='vector' or (kind=='mixed' and j==0) else 'spinor') for j,r in enumerate(roles)]
    counts = [2-roles.count(0),126-roles.count(1)]
    assert counts == family['spectator_counts'] and min(counts) >= 0
    base = ZERO
    for r,n in enumerate(counts):
        base = plus(base,scaled(pp[r]['vector'][0][0],n))
    for r,k in targets:
        base = plus(base,pp[r][k][0][0])
    states = []
    for a,ea in words(counts[0],pp[0]['vector'][0],202):
        for b,eb in words(counts[1],pp[1]['vector'][0],202):
            cost0 = plus(ea,eb)
            if cost0[0] > 202*Q:
                continue
            for t in itertools.product(*[range(len(pp[r][k][0])) for r,k in targets]):
                cost = cost0
                for (r,k),j in zip(targets,t):
                    cost = plus(cost,minus(pp[r][k][0][j],pp[r][k][0][0]))
                if cost[0] <= 202*Q:
                    parity = (a[1]+a[3]+b[1]+b[3]+sum(j%2 for (r,k),j in zip(targets,t) if not (k=='vector' and j==5)))%2
                    states.append(((a,b,t),cost,parity))
    index = {key:j for j,(key,_,_) in enumerate(states)}
    assert len(index) == len(states)
    def neighbors(key):
        a,b,t = key
        for r,w in ((0,a),(1,b)):
            for (dest,src),v in pp[r]['vector'][1].items():
                if w[src]:
                    nw=list(w);nw[src]-=1;nw[dest]+=1
                    yield ((tuple(nw),b,t) if r==0 else (a,tuple(nw),t)),times(v,radical(fixed(w[src]*(w[dest]+1))))
        for pos,(r,k) in enumerate(targets):
            for (dest,src),v in pp[r][k][1].items():
                if src == t[pos]:
                    nt=list(t);nt[pos]=dest
                    yield (a,b,tuple(nt)),v
    # Actual maximum over all six distinguished/vector indices, not the supplier's
    # accidentally scalar-last quartet proxy: this stronger bound also passes.
    maxstep = max(minus(max(pp[r][k][0],key=lambda x:x[1]),pp[r][k][0][0])[1] for r,k in [(0,'vector'),(1,'vector')]+targets)
    assert 120*Q+maxstep < 202*Q
    outputs = [j for j,(_,cost,_) in enumerate(states) if cost[1] <= 120*Q]
    return base, states, index, neighbors, outputs, maxstep


def validate_preimages(outputs, states, index, neighbors, identity=True):
    assert identity, 'missing affine identity'
    edges = 0
    for j in outputs:
        assert states[j][0] in index, 'missing identity input'
        for key,_ in neighbors(states[j][0]):
            assert key in index, 'missing density input preimage'
            assert states[index[key]][2] != states[j][2], 'incorrect density parity'
            edges += 1
    return edges


def rejected(label, fn):
    try:
        fn()
    except AssertionError as exc:
        return {'control':label, 'reached_rejecting_assertion':True, 'assertion':str(exc)}
    raise AssertionError('adverse control did not reject: '+label)


def replay(family, pp):
    base,states,index,neighbors,outputs,maxstep = formation(family,pp)
    witness = np.load(ROOT/family['witness_path'],allow_pickle=False)
    assert witness['states'].tolist() == [list(a+b+t) for (a,b,t),_,_ in states]
    assert witness['old_parity'].tolist() == [p for _,_,p in states]
    assert witness['outputs'].tolist() == outputs
    u = int(family['u_exact'])
    assert int(witness['u']) == u and u in (-1,1)
    D = [minus(plus(base,cost),fixed(-5187)) for _,cost,_ in states]
    assert all(d[1] < 0 or d[0] > 0 for d in D)
    L = [j for j,d in enumerate(D) if d[1] < 0]
    P = [j for j,d in enumerate(D) if d[0] > 0]
    assert witness['negative'].tolist() == L
    assert len(L) == (1 if family['kind']=='vector' else 0)
    assert base[0]+202*Q+5187*Q > 0
    assert family['base_exact_interval'] == [str(F(v,Q)) for v in base]
    assert family['omitted_free_H_minus_z_lower_exact'] == str(F(base[0]+202*Q+5187*Q,Q))
    edges = validate_preimages(outputs,states,index,neighbors)
    assert (len(states),len(outputs),edges) == (family['input_support'],family['output_rows'],family['density_edges'])
    rowindex = {j:r for r,j in enumerate(outputs)}
    columns = []
    for j,(key,_,_) in enumerate(states):
        entries = [(rowindex[j],fixed(u))] if j in rowindex else []
        for key2,v in neighbors(key):
            target = index.get(key2)
            if target in rowindex:
                entries.append((rowindex[target],v))
        columns.append(entries)
    Y = [[fixed(float(v)) for v in row] for row in witness['Y']]
    k=len(L)
    assert witness['Y'].shape == (len(outputs),k)
    def product(j):
        values=[ZERO]*k
        for row,b in columns[j]:
            for t in range(k):
                values[t]=plus(values[t],times(b,Y[row][t]))
        return values
    G=[[ZERO]*k for _ in range(k)]
    for t,j in enumerate(L):
        v=product(j)
        for q in range(k):
            G[t][q]=plus(G[t][q],v[q]);G[q][t]=plus(G[q][t],v[q])
    for row in Y:
        for t in range(k):
            for q in range(t,k):
                v=times(row[t],row[q]);G[t][q]=minus(G[t][q],v)
                if t!=q:
                    G[q][t]=minus(G[q][t],v)
    residual=[[negative(v) for v in row] for row in Y]
    for t,j in enumerate(L):
        for row,b in columns[j]:
            residual[row][t]=plus(residual[row][t],b)
    for j in P:
        v=product(j); weighted=[scaled(times(x,reciprocal(D[j])),64) for x in v]
        for row,b in columns[j]:
            for t in range(k):
                residual[row][t]=minus(residual[row][t],times(b,weighted[t]))
        for t in range(k):
            for q in range(t,k):
                z=times(v[t],weighted[q]);G[t][q]=minus(G[t][q],z)
                if t!=q:
                    G[q][t]=minus(G[q][t],z)
    schur=[[scaled(v,64) for v in row] for row in G]
    for t,j in enumerate(L):
        schur[t][t]=plus(schur[t][t],D[j])
    center=[[(a+b)//2 for a,b in row] for row in schur]
    radius=max((sum(max(center[t][q]-schur[t][q][0],schur[t][q][1]-center[t][q]) for q in range(k)) for t in range(k)),default=0)
    residual2=sum((max(abs(a),abs(b))**2+Q-1)//Q for row in residual for a,b in row)
    low=[r[:] for r in center];high=[r[:] for r in center]
    for t in range(k):
        low[t][t]-=radius;high[t][t]+=radius+64*residual2
    observed=load(family['matrix_path'])
    expected={'bits':100,'center_integer':center,'radius_integer':radius,'residual_squared_upper_integer':residual2,
              'lower_integer':low,'upper_integer':high,'inertia_lower':inertia(low),'inertia_upper':inertia(high)}
    assert observed == expected, ('independent interval replay differs',family['matrix_path'])
    assert expected['inertia_lower']['negative'] == 0
    assert expected['inertia_lower'] == family['lower_inertia']
    scalar=[]
    if family['kind']=='mixed':
        scalar=[t for t,j in enumerate(L) if states[j][0][2][0]==5]
        assert scalar == family['scalar_transfer']['indices'] == []
        assert any(key[2][0]==5 for key,_,_ in states)
        assert all(D[j][0]>0 for j,(key,_,_) in enumerate(states) if key[2][0]==5)
        assert all(other[2][0]==5 for key,_,_ in states if key[2][0]==5 for other,_ in neighbors(key))
    controls=[]
    if family['kind']=='vector' and family['roles']==[0]:
        controls.append(rejected('missing identity',lambda:validate_preimages(outputs,states,index,neighbors,False)))
        own=states[outputs[0]][0];bad=dict(index);del bad[own]
        controls.append(rejected('missing retained identity input',lambda:validate_preimages(outputs,states,bad,neighbors)))
        key=next(key for j in outputs for key,_ in neighbors(states[j][0]));bad=dict(index);del bad[key]
        controls.append(rejected('missing density preimage',lambda:validate_preimages(outputs,states,bad,neighbors)))
        badstates=[(key,cost,0) for key,cost,_ in states]
        controls.append(rejected('incorrect parity',lambda:validate_preimages(outputs,badstates,index,neighbors)))
    return {'matrix':family['matrix_path'],'witness':family['witness_path'],'u':u,'kind':family['kind'],'roles':family['roles'],
            'independent_exact_matrix_equal':True,'negative_free_columns':k,'input_count':len(states),'output_count':len(outputs),
            'density_preimages':edges,'max_density_energy_step_upper':str(F(maxstep,Q)),
            'omitted_free_margin':str(F(base[0]+202*Q+5187*Q,Q)),'scalar_negative_indices':scalar,'controls':controls}


def second(M,n):
    masks=[sum(1<<i for i in word) for word in itertools.combinations(range(4),n)]
    out=s.zeros(len(masks))
    for col,mask in enumerate(masks):
        for q in range(4):
            if mask>>q&1:
                tail=mask^(1<<q); sign=(-1)**((mask&((1<<q)-1)).bit_count())
                for p in range(4):
                    if not tail>>p&1:
                        out[masks.index(tail|(1<<p)),col]+=sign*(-1)**((tail&((1<<p)-1)).bit_count())*M[p,q]
    return out,masks


def symbolic_controls():
    a,b,e,c=s.symbols('a b e c',real=True)
    X=s.Matrix([[0,1],[1,0]]); S4=s.diag(1,-1,1,-1)
    m=e*(a*a-b*b);d=2*e*a*b
    H=m*s.diag(-1,1,-1,1)+c*s.kronecker_product(X,s.diag(1,-1))+d*s.Matrix([[0,0,0,s.I],[0,0,s.I,0],[0,-s.I,0,0],[-s.I,0,0,0]])
    R=s.diag(s.Rational(1,2),s.Rational(1,2),-s.Rational(1,2),-s.Rational(1,2))
    wp=s.Matrix([[1,0],[0,1],[1,0],[0,-1]])/s.sqrt(2);wm=s.Matrix([[1,0],[0,1],[-1,0],[0,1]])/s.sqrt(2)
    U=s.Matrix.hstack(wp*s.Matrix([a,-s.I*b]),wm*s.Matrix([a,s.I*b]),s.I*wp*s.Matrix([-s.I*b,a]),s.I*wm*s.Matrix([s.I*b,a]))
    D4=s.diag(-e+c,-e-c,e+c,e-c);f=(a*a-b*b)/2;g=a*b
    R4=s.Matrix([[0,f,0,-g],[f,0,g,0],[0,g,0,f],[-g,0,f,0]])
    def zero(M):
        return all(s.rem(s.expand(z),a*a+b*b-1,b)==0 for z in M)
    assert zero(U.conjugate().T*U-s.eye(4)) and zero(U.conjugate().T*H*U-D4) and zero(U.conjugate().T*R*U-R4)
    D2,masks=second(D4,2);R2,_=second(R4,2);V=s.zeros(6)
    for j,(mask,v) in enumerate([(3,1),(5,1),(9,1/s.sqrt(2)),(10,1),(12,-1),(9,1/s.sqrt(2))]):
        V[masks.index(mask),j]=v
    V[masks.index(6),2]=1/s.sqrt(2);V[masks.index(6),5]=-1/s.sqrt(2)
    D6=s.diag(-2*e,2*c,0,-2*c,2*e,0);R6=s.zeros(6)
    for j in (1,3):
        for i,v in ((0,g),(2,s.sqrt(2)*f),(4,-g)):
            R6[i,j]=R6[j,i]=v
    assert zero(V.T*V-s.eye(6)) and zero(D2*V-V*D6) and zero(R2*V-V*R6)
    assert R6[:,5]==s.zeros(6,1) and D6[5,5]==0
    H3,masks3=second(H,3);R3,_=second(R,3);J=s.zeros(4)
    for p in range(4):
        J[masks3.index(15^(1<<p)),p]=(-1)**p
    T3=J*s.kronecker_product(s.eye(2),X)*U*S4
    assert zero(T3.conjugate().T*T3-s.eye(4)) and zero(H3*T3-T3*D4) and zero(R3*T3-T3*R4)
    for n in (0,4):
        hn,_=second(H,n);rn,_=second(R,n)
        assert zero(hn) and zero(rn)
    assert zero(S4*D4*S4-D4) and zero(S4*R4*S4+R4)
    # Physical W=U S4 U* acts through its exterior powers on every local n,
    # not merely the number1 frame or separately fitted +/- spectra.
    W=U*S4*U.conjugate().T
    assert zero(W*H*W.conjugate().T-H) and zero(W*R*W.conjugate().T+R)
    D=s.diag(-6,-4,4,6);r=s.Matrix([[0,s.Rational(3,10),0,-s.Rational(2,5)],
        [s.Rational(3,10),0,s.Rational(2,5),0],[0,s.Rational(2,5),0,s.Rational(3,10)],[-s.Rational(2,5),0,s.Rational(3,10),0]])
    for n in range(5):
        dn,masksn=second(D,n);rn,_=second(r,n)
        pn=s.diag(*[s.prod(S4[i,i] for i in range(4) if mask>>i&1) for mask in masksn])
        assert pn*(dn+64*(rn+s.eye(len(masksn)))**2)*pn == dn+64*(rn-s.eye(len(masksn)))**2
    hp=D+64*(r+s.eye(4))**2
    assert hp[0,1]==s.Rational(192,5)
    def parity_wrong():
        split=hp.copy()
        for i in range(4):
            for j in range(4):
                if S4[i,i]!=S4[j,j]:
                    split[i,j]=0
        assert split==hp, 'parity splitting loses 128u rho'
    def php_wrong():
        h=s.diag(-s.Rational(3,2),s.Rational(1,10))+s.ones(2)**2
        v=s.Matrix([1,-1]);assert h[0,0]>0
        assert (v.T*h*v)[0]>=0, 'positive PHP does not imply full positivity: -7/5'
    K=s.Matrix([[3,1],[1,2]]);Y=s.Matrix([[s.Rational(1,4),s.Rational(1,10)],[s.Rational(1,8),s.Rational(1,2)]])
    E=s.eye(2)-K*Y
    assert K.inv()-(Y+Y.T-Y.T*K*Y) == E.T*K.inv()*E
    return {'generic_native_U_full6_signed_number3':True,'whole_physical_rho_flip':True,'finiteCAR_numbers':list(range(5)),
            'scalar_transfers_preserve_D_and_R_before_square':True,'residual_identity_exact_2x2':True,
            'affine_cross_entry':str(hp[0,1]),'controls':[rejected('parity splitting',parity_wrong),rejected('PHP positivity',php_wrong)]}


def cover_control(pp):
    profile=load('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json')['profiles']
    e=[radical(fixed(p['e_squared_exact'])) for p in profile];c=F(profile[0]['c0_exact'])
    assert c==F(profile[1]['c0_exact'])<0 and e[0][1]<e[1][0]
    delta=F(e[0][0],Q)+c
    assert delta>0 and 6*delta>4*F(e[0][1],Q)
    assert F(e[0][0],Q)+3*c>0
    margin=F(-252*e[1][1],Q)+5187
    assert margin>0
    # Derive every local excitation from four one-particle exterior sums.
    one=[(-1,1),(-1,-1),(1,1),(1,-1)];local=[]
    for n in range(5):
        for occupied in itertools.combinations(range(4),n):
            if (n,occupied)!=(2,(0,1)):
                local.append((n,2+sum(one[j][0] for j in occupied),sum(one[j][1] for j in occupied)))
    assert len(local)==15
    for r in (0,1):
        for n,pe,pc in local:
            cost=pe*F(e[r][0],Q)+pc*c
            assert cost >= (2*F(e[0][0],Q) if n in (0,4) else 2*delta if n==2 else delta)
    # Integer neutrality and lower costs exhaust all classes below4eA.
    classes=[]
    for t,o,j in itertools.product(range(6),repeat=3):
        if o%2 or (t==1 and o==0):
            continue
        # Compare the radical difference symbolically before enclosure. In
        # particular t=2,o=j=0 is exactly the cutoff, not an interval overlap.
        coefficient=2*t+o+2*j-4
        diff=scaled(e[0],coefficient)
        diff=plus(diff,fixed((o+2*j)*c))
        assert diff == ZERO or diff[0] > 0 or diff[1] < 0
        if diff[1] < 0:
            assert (t==0 and ((o==0 and j<=2) or (o==2 and j<=1) or (o==4 and j==0))) or (t==1 and o==2 and j==0)
            classes.append([t,o,j])
    return {'actual_role_slots':[2,126],'all15_local_excitations_derived':True,'cover_classes':classes,
            'delta_lower':str(delta),'six_delta_minus_four_eA_lower':str(6*delta-4*F(e[0][1],Q)),
            'complement_margin_above_minus5187':str(margin),'labelled_union_not_directsum':True,
            'proof':'At cost<4eA every changed label is distinguished; unchanged vector-minimum product is role-symmetric. All labelled images reduce D and rho, hence affine H. The spectral projector below -5187 annihilates each image and its coherent sum, not a sum of image inertias. On its complement D>=-252eU and the affine square is positive.'}


def main():
    manifest=load(PREFIX+'_preexecution.json')
    for name,value in manifest['sha256'].items():
        assert digest(name)==value, ('immutable input changed',name)
    pp=profiles();results=[];archive_checks=[]
    for tag in ('affine_canonical_weak_actual1_20261011','affine_canonical_weak_minus_actual1_20261011'):
        receipt=load(tag+'_receipt.json');natural=load(tag+'_natural_exit.json')
        assert natural['natural_exit_code']==0 and not natural['signal_exit']
        assert natural['sources_unchanged'] and natural['inputs_unchanged']
        assert len(receipt['families'])==18 and receipt['all18_certified']
        assert receipt['source_sha256']==manifest['sha256']['affine_canonical_lower_20261011.py']
        with ZipFile(ROOT/(tag+'_executed.zip')) as archive:
            checked=0
            for name,value in natural['output_sha256'].items():
                assert digest(name)==value
                assert archive.read(name)==(ROOT/name).read_bytes()
                checked+=1
        with ZipFile(ROOT/(tag+'_preexecution.zip')) as archive:
            for name,value in {**natural['source_sha256'],**natural['input_sha256']}.items():
                assert hashlib.sha256(archive.read(name)).hexdigest()==value==digest(name)
        archive_checks.append({'tag':tag,'output_members_equal':checked,'natural_exit':0,'executed_zip_sha256':digest(tag+'_executed.zip'),
                               'preexecution_zip_sha256':digest(tag+'_preexecution.zip'),'receipt_sha256':digest(tag+'_receipt.json'),
                               'natural_exit_sha256':digest(tag+'_natural_exit.json')})
        for f in receipt['families']:
            assert digest(f['matrix_path'])==f['matrix_sha256'] and digest(f['witness_path'])==f['witness_sha256']
            results.append(replay(f,pp))
    out={'verdict':'CONDITIONAL','affine_formation_review':'VALID: every exact stored weak lower/upper matrix independently reproduced from profile radicals and frozen binary witnesses',
         'remaining_condition':'The pinned exact native binary profile/joint frame is the accepted matching-H0 law. No approximate-ground cap premise is required for this weak affine lower.',
         'matching_H0_conditional_graph_allocation_eligible':True,'floor':'-5187','u':['1','-1'],'number_sector':256,
         'family_reviews':results,'archives':archive_checks,'symbolic_controls':symbolic_controls(),'cover':cover_control(pp),
         'old_scientific_or_checker_reruns':0,'nonclaims':['strong -5123','new g(u)','charged254/258','outer255/257','fullPhi/lattice ground','physical1 ground','massless photon']}
    for name,value in manifest['sha256'].items():
        assert digest(name)==value
    with (ROOT/(PREFIX+'_receipt.json')).open('x') as f:
        json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({'receipt':PREFIX+'_receipt.json','sha256':digest(PREFIX+'_receipt.json'),'verdict':out['verdict'],
                      'exact_matrix_replays':len(results),'adverse_assertions_rejected':sum(len(r['controls']) for r in results)+2}),flush=True)


if __name__ == '__main__':
    main()

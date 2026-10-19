from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parent
BITS = 100
Q = 1 << BITS
INPUTS = {
    'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json': '102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8',
    'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json': '2f019dd49873a5e13f130c65ef6739b9af78a16f85138dc6c6ab6e32e65c7a12',
    'local_so5_resum_actual04_20261011_state.npz': 'dd06400e1b7f70df935cf7ca02cf5ebf598481a8d5e209d1241aba5ed3fea2a2',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iv(x):
    f = Fraction(x)
    a, b = f.numerator * Q, f.denominator
    return a // b, -((-a) // b)


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def neg(a):
    return -a[1], -a[0]


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    p = [a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]]
    return min(p)//Q, -((-max(p))//Q)


def inv(a):
    assert a[0] > 0 or a[1] < 0
    return Q*Q//a[1], -((-Q*Q)//a[0])


def sqrt(a):
    assert a[0] >= 0
    lo, hi = math.isqrt(a[0]*Q), math.isqrt(a[1]*Q)
    return lo, hi + (hi*hi < a[1]*Q)


def scale(a, n):
    return (a[0]*n, a[1]*n) if n >= 0 else (a[1]*n, a[0]*n)


def mid(a):
    return (a[0] + a[1]) / (2*Q)


def sq_upper(a):
    return (max(abs(a[0]), abs(a[1]))**2 + Q - 1)//Q


def endpoints(a):
    return [str(Fraction(x, Q)) for x in a]


ZERO = iv(0)


def profiles():
    p = json.loads((ROOT/next(iter(INPUTS))).read_text())
    assert p['corrected_source_unitarity_H_density_identities']
    pp = []
    for item in p['profiles']:
        m, d, c = [iv(item[k+'_exact']) for k in ('m', 'd', 'c0')]
        assert Fraction(item['e_squared_exact']) == Fraction(item['m_exact'])**2 + Fraction(item['d_exact'])**2
        e = sqrt(iv(item['e_squared_exact']))
        ve = [scale(e,-2),scale(c,2),ZERO,scale(c,-2),scale(e,2),ZERO]
        vr = {}
        for b in (1,3):
            for a,v in ((0,mul(d,inv(scale(e,2)))),(2,mul(m,inv(mul(sqrt(iv(2)),e)))),(4,neg(mul(d,inv(scale(e,2)))))):
                vr[a,b] = vr[b,a] = v
        se = [add(neg(e),c),sub(neg(e),c),add(e,c),sub(e,c)]
        sr = {}
        for a,b,v in ((0,1,mul(m,inv(scale(e,2)))),(2,3,mul(m,inv(scale(e,2)))),(0,3,neg(mul(d,inv(scale(e,2))))),(1,2,mul(d,inv(scale(e,2))))):
            sr[a,b] = sr[b,a] = v
        pp.append({'vector':(ve[:5],vr), 'spinor':(se,sr)})
    return pp


def occupations(n, energy, budget):
    gaps = [sub(x,energy[0]) for x in energy]
    out = []
    def visit(j, remaining, cost, word):
        if j == len(energy):
            out.append(((remaining,*word),cost))
            return
        assert gaps[j][0] > 0
        limit = min(remaining,max(0,(budget*Q-cost[0])//gaps[j][0]))
        for k in range(limit+1):
            visit(j+1,remaining-k,add(cost,scale(gaps[j],k)),word+[k])
    visit(1,n,ZERO,[])
    return out


def bareiss(matrix):
    a = [row[:] for row in matrix]
    previous = 1
    signs = []
    for t in range(len(a)):
        pivot = a[t][t]
        assert pivot != 0, ('zero leading pivot',t)
        signs.append(1 if pivot*previous > 0 else -1)
        for i in range(t+1,len(a)):
            for j in range(t+1,len(a)):
                numerator = a[i][j]*pivot-a[i][t]*a[t][j]
                assert numerator % previous == 0
                a[i][j] = numerator//previous
        previous = pivot
    return {'negative':signs.count(-1),'positive':signs.count(1),'pivot_signs':signs}


def family_spec(kind, roles):
    counts = [2,126]
    for role in roles:
        counts[role] -= 1
    assert min(counts) >= 0
    return tuple(counts), [(role,kind) for role in roles]


def construct(kind, roles, output_budget, support_budget):
    pp = profiles()
    counts,targets = family_spec(kind,roles)
    base = ZERO
    for role,n in enumerate(counts):
        base = add(base,scale(pp[role]['vector'][0][0],n))
    for role,rep in targets:
        base = add(base,pp[role][rep][0][0])
    aa = occupations(counts[0],pp[0]['vector'][0],support_budget)
    bb = occupations(counts[1],pp[1]['vector'][0],support_budget)
    states = []
    for a,ea in aa:
        for b,eb in bb:
            cost0 = add(ea,eb)
            if cost0[0] > support_budget*Q: continue
            for target in itertools.product(*[range(len(pp[r][k][0])) for r,k in targets]):
                cost = cost0
                for (role,rep),j in zip(targets,target):
                    en = pp[role][rep][0]
                    cost = add(cost,sub(en[j],en[0]))
                if cost[0] <= support_budget*Q:
                    parity = (a[1]+a[3]+b[1]+b[3]+sum(j%2 for j in target if not (kind=='vector' and j==5)))%2
                    states.append(((a,b,target),cost,parity))
    index = {key:j for j,(key,_,_) in enumerate(states)}
    def neighbors(key):
        a,b,target = key
        for role in (0,1):
            word = a if role == 0 else b
            for (dest,src),value in pp[role]['vector'][1].items():
                if not word[src]: continue
                nw = list(word);nw[src] -= 1;nw[dest] += 1
                key2 = (tuple(nw),b,target) if role==0 else (a,tuple(nw),target)
                yield key2,mul(value,sqrt(iv(word[src]*(word[dest]+1))))
        for t,(role,rep) in enumerate(targets):
            for (dest,src),value in pp[role][rep][1].items():
                if src != target[t]: continue
                nt = list(target);nt[t] = dest
                yield (a,b,tuple(nt)),value
    # All excluded occupancy continuations have cost>support_budget by strictly
    # positive gap lower endpoints, and distinguished indices are exhaustive.
    maxstep = max(sub(pp[r][k][0][-1],pp[r][k][0][0])[1] for r,k in [(0,'vector'),(1,'vector')]+targets)
    assert output_budget*Q+maxstep < support_budget*Q
    outputs = [j for j,(_,cost,_) in enumerate(states) if cost[1] <= output_budget*Q]
    edge_count = 0
    for j in outputs:
        for key,value in neighbors(states[j][0]):
            assert key in index, ('incomplete density output image',key)
            assert states[index[key]][2] != states[j][2]
            edge_count += 1
    # Wrongcase: removing a required input cannot pass output completeness.
    required = next((key for j in outputs for key,_ in neighbors(states[j][0])),None)
    assert required is not None
    cropped = set(index);cropped.remove(required)
    assert any(key not in cropped for j in outputs for key,_ in neighbors(states[j][0]))
    return base,states,index,outputs,neighbors,edge_count,counts


def certify(kind, roles, floor, output_budget, support_budget, tag, inventory):
    start = time.monotonic()
    base,states,index,outputs,neighbors,edge_count,counts = construct(kind,roles,output_budget,support_budget)
    z = iv(floor)
    D = [sub(add(base,cost),z) for _,cost,_ in states]
    assert base[0]+support_budget*Q > z[1]
    assert all(x[1]<0 or x[0]>0 for x in D)
    result = {'kind':kind,'roles':roles,'spectator_counts':counts,'floor_exact':str(Fraction(floor)),
              'base_exact_interval':endpoints(base),'output_budget':output_budget,'support_budget':support_budget,
              'input_support':len(states),'output_rows':len(outputs),'checked_density_edges':edge_count,
              'entire_density_output_image_checked':True,'input_crop_wrongcase_rejected':True,
              'omitted_spectrum_lower_exact':str(Fraction(base[0]+support_budget*Q,Q)),
              'omitted_spectrum_proof':'Strictly positive individual gap lower endpoints bound ALL skipped occupation continuations; all finite distinguished indices exhausted. Hlo keeps positive output rows, not input PHP. Outside support D>0 and output-density image is complete.',
              'u_exact':'0','external_charge':'0','parity_blocks':[]}
    print(json.dumps(result),flush=True)
    if inventory:
        result['negative_counts'] = [sum(x[1]<0 and states[j][2]==p for j,x in enumerate(D)) for p in (0,1)]
        return result
    for parity in (0,1):
        inputs = [j for j,state in enumerate(states) if state[2]==parity]
        rows = [j for j in outputs if states[j][2]!=parity]
        row_index = {j:r for r,j in enumerate(rows)}
        L = [j for j in inputs if D[j][1]<0]
        positive = [j for j in inputs if D[j][0]>0]
        columns = {}
        rr,cc,vv = [],[],[]
        for col,j in enumerate(inputs):
            entries = []
            for key,value in neighbors(states[j][0]):
                target = index.get(key)
                if target in row_index:
                    row = row_index[target]
                    entries.append((row,value));rr.append(row);cc.append(col);vv.append(mid(value))
            columns[j] = entries
        B = coo_matrix((vv,(rr,cc)),shape=(len(rows),len(inputs))).tocsr()
        ii = {j:c for c,j in enumerate(inputs)}
        BL = B[:,[ii[j] for j in L]].toarray()
        BQ = B[:,[ii[j] for j in positive]]
        dq = np.array([mid(D[j]) for j in positive])
        K = eye(len(rows),format='csc')+64*(BQ.multiply(1/dq)@BQ.T).tocsc()
        Y = splu(K).solve(BL)
        witness = ROOT/f'{tag}_{kind}_{"".join(str(r) for r in roles)}_p{parity}_witness.npz'
        np.savez_compressed(witness,Y=Y,negative=np.array(L),outputs=np.array(rows),states=np.array([a+b+t for (a,b,t),_,_ in states],dtype=np.int64))
        YI = [[iv(float(x)) for x in row] for row in Y]
        k = len(L)
        def product(j):
            v = [ZERO]*k
            for row,b in columns[j]:
                for t in range(k):
                    v[t] = add(v[t],mul(b,YI[row][t]))
            return v
        G = [[ZERO]*k for _ in range(k)]
        for t,j in enumerate(L):
            v = product(j)
            for u in range(k):
                G[t][u] = add(G[t][u],v[u]);G[u][t] = add(G[u][t],v[u])
        for row in YI:
            for t in range(k):
                for u in range(t,k):
                    v = mul(row[t],row[u]);G[t][u] = sub(G[t][u],v)
                    if t!=u: G[u][t] = sub(G[u][t],v)
        RY = [[neg(x) for x in row] for row in YI]
        for t,j in enumerate(L):
            for row,b in columns[j]: RY[row][t] = add(RY[row][t],b)
        for j in positive:
            v = product(j)
            weighted = [scale(mul(x,inv(D[j])),64) for x in v]
            for row,b in columns[j]:
                for t in range(k): RY[row][t] = sub(RY[row][t],mul(b,weighted[t]))
            for t in range(k):
                for u in range(t,k):
                    val = mul(v[t],weighted[u]);G[t][u] = sub(G[t][u],val)
                    if t!=u: G[u][t] = sub(G[u][t],val)
        S = [[scale(G[t][u],64) for u in range(k)] for t in range(k)]
        for t,j in enumerate(L): S[t][t] = add(S[t][t],D[j])
        center = [[(x[0]+x[1])//2 for x in row] for row in S]
        assert all(center[t][u]==center[u][t] for t in range(k) for u in range(k))
        radius = max((sum(max(center[t][u]-S[t][u][0],S[t][u][1]-center[t][u]) for u in range(k)) for t in range(k)),default=0)
        residual = sum(sq_upper(x) for row in RY for x in row)
        low,high = [row[:] for row in center],[row[:] for row in center]
        for t in range(k):
            low[t][t] -= radius;high[t][t] += radius+64*residual
        ilow,ihigh = bareiss(low),bareiss(high)
        matrix = ROOT/f'{tag}_{kind}_{"".join(str(r) for r in roles)}_p{parity}_matrix.json'
        matrix.write_text(json.dumps({'bits':BITS,'center_integer':center,'radius_integer':radius,'residual_squared_upper_integer':residual,'lower_integer':low,'upper_integer':high,'inertia_lower':ilow,'inertia_upper':ihigh})+'\n')
        block = {'parity':parity,'negative_free_columns':k,'lower_inertia':ilow,'upper_inertia':ihigh,
                 'matrix_path':matrix.name,'matrix_sha256':digest(matrix),'witness_path':witness.name,'witness_sha256':digest(witness),
                 'radius_upper_exact':str(Fraction(radius,Q)),'residual_squared_upper_exact':str(Fraction(residual,Q))}
        result['parity_blocks'].append(block)
        print(json.dumps(block),flush=True)
    negative_lower = sum(b['lower_inertia']['negative'] for b in result['parity_blocks'])
    negative_upper = sum(b['upper_inertia']['negative'] for b in result['parity_blocks'])
    expected = 0
    result.update(certified=negative_lower==negative_upper==expected,negative_lower=negative_lower,negative_upper=negative_upper,
                  eigenvalue_index=1,elapsed_seconds=time.monotonic()-start)
    return result



def joint_controls():
    import sympy as s
    a,b,e,c=s.symbols('a b e c',real=True)
    m=e*(a*a-b*b);d=2*e*a*b
    A=s.diag(-1,1,-1,1)
    B=s.kronecker_product(s.Matrix([[0,1],[1,0]]),s.diag(1,-1))
    X=s.Matrix([[0,1],[1,0]]);Z=s.zeros(2)
    C=Z.row_join(s.I*X).col_join((-s.I*X).row_join(Z))
    R=s.diag(s.Rational(1,2),s.Rational(1,2),-s.Rational(1,2),-s.Rational(1,2))
    H=m*A+c*B+d*C
    Wplus=s.Matrix([[1,0],[0,1],[1,0],[0,-1]])/s.sqrt(2)
    Wminus=s.Matrix([[1,0],[0,1],[-1,0],[0,1]])/s.sqrt(2)
    U=s.Matrix.hstack(Wplus*s.Matrix([a,-s.I*b]),Wminus*s.Matrix([a,s.I*b]),s.I*Wplus*s.Matrix([-s.I*b,a]),s.I*Wminus*s.Matrix([s.I*b,a]))
    D4=s.diag(-e+c,-e-c,e+c,e-c)
    rf=(a*a-b*b)/2;rg=a*b
    R4=s.Matrix([[0,rf,0,-rg],[rf,0,rg,0],[0,rg,0,rf],[-rg,0,rf,0]])
    def reduce(z):return s.rem(s.expand(z),b*b+a*a-1,b)
    def zero(M):return all(reduce(z)==0 for z in M)
    assert zero(U.conjugate().T*U-s.eye(4))
    assert zero(U.conjugate().T*H*U-D4) and zero(U.conjugate().T*R*U-R4)
    
    def second(M,n):
        masks=[sum(1<<q for q in comb) for comb in itertools.combinations(range(4),n)]
        out=s.zeros(len(masks))
        for j,mask in enumerate(masks):
            for p in range(4):
                for q in range(4):
                    if mask>>q&1:
                        sign=(-1)**((mask&((1<<q)-1)).bit_count());tail=mask^(1<<q)
                        if not tail>>p&1:
                            sign*=(-1)**((tail&((1<<p)-1)).bit_count())
                            out[masks.index(tail|(1<<p)),j]+=sign*M[p,q]
        return out,masks
    h2,_=second(D4,2);r2,masks2=second(R4,2)
    # Consistent free5 joint phase: highest pair has a minus sign, not an
    # independently chosen density phase. This matches corrected 102966f3.
    V=s.zeros(6,5)
    for j,(mask,coef) in enumerate([(3,1),(5,1),(9,1/s.sqrt(2)),(10,1), (12,-1)]):V[masks2.index(mask),j]=coef
    V[masks2.index(6),2]=1/s.sqrt(2)
    D5=s.diag(-2*e,2*c,0,-2*c,2*e)
    R5=s.zeros(5)
    for j in [1,3]:
        for i,z in [(0,rg),(2,s.sqrt(2)*rf),(4,-rg)]:R5[i,j]=R5[j,i]=z
    assert zero(h2*V-V*D5) and zero(r2*V-V*R5)
    assert zero(V.T*V-s.eye(5))
    wrongV=V.copy();wrongV[:,4]=-wrongV[:,4]
    assert not zero(r2*wrongV-wrongV*R5)
    # Full number3 CAR, not an inferred scalar hole or a neutral gap.
    h3,masks3=second(H,3);r3,_=second(R,3)
    J=s.zeros(4)
    for p in range(4):J[masks3.index(15^(1<<p)),p]=(-1)**p
    assert zero(J.T*h3*J+H.T) and zero(J.T*r3*J+R.T)
    Q=s.kronecker_product(s.eye(2),X)
    assert zero(Q*(-H.T)*Q-H) and zero(Q*(-R.T)*Q+R)
    S4=s.diag(1,-1,1,-1);S5=s.diag(1,-1,1,-1,1)
    assert zero(S4*D4*S4-D4) and zero(S4*R4*S4+R4)
    assert zero(S5*D5*S5-D5) and zero(S5*R5*S5+R5)
    # Simultaneous number3 frame T3=J Q U S4 gives D4,R4, not D4 alone.
    T3=J*Q*U*S4
    assert zero(T3.conjugate().T*h3*T3-D4)
    assert zero(T3.conjugate().T*r3*T3-R4)
    wrong=T3*s.diag(-1,1,1,1)
    assert zero(wrong.conjugate().T*h3*wrong-D4)
    assert not zero(wrong.conjugate().T*r3*wrong-R4)
    return {'joint_native_U_D_R':True,'number3_T3':'J Q U diag(1,-1,1,-1)',
            'number3_joint_D_R':True,'wrong_density_phase_rejected':True,
            'canonical_balanced_number':'124*2+1+1+3+3=256',
            'unbalanced_charge_rejected':sum([1,1,1,3])+124*2 != 256}

PROOF = "D=hfree-z; B=P_output rho_all including ALL preimages. D+64 B*B<=H0-z, not PHP. All skipped free occupations strictly above support budget by positive gap lower endpoints and exhaustive four distinguished indices. K=I+64 BQ DQ^-1 BQ*>=I. Binary Y defines G=BL*Y+Y*BL-Y*KY <= BL*K^-1 BL <= G+(BL-KY)*(BL-KY). Outward 100-bit interval arithmetic, symmetric row-sum radius and Frobenius residual bound provide exact integer Loewner Schur matrices. Bareiss inertia certifies lower form positivity if no negative lower pivots; a negative upper inertia proves only lower-form failure, NOT negative full-H eigenvalue. rho_all=sum of all four spinor and both spectator-role contributions, squared ONCE. Number1 U and number3 J Q U S4 intertwine D and R jointly. Distinct conserved slot numbers imply balanced two1/two3 domain orthogonal to source ground."

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--output-budget',type=int,default=120)
    parser.add_argument('--support-budget',type=int,default=202)
    parser.add_argument('--inventory',action='store_true')
    args=parser.parse_args()
    assert args.tag.startswith('canonical_four_odd_')
    extra=['neutral_return_lower_control_20261011.py','neutral_return_lower_boundary_actual2_20261011_receipt.json',
           'charged_spinor_joint_free_phase_control_20261011.py']
    pins={**INPUTS,**{f:digest(ROOT/f) for f in extra}}
    for f,pin in pins.items(): assert digest(ROOT/f)==pin
    own=digest(Path(__file__))
    controls=joint_controls()
    families=[]
    for active in (0,1,2):
        roles=(0,)*active+(1,)*(4-active)
        result=certify('spinor',roles,Fraction(-5123),args.output_budget,args.support_budget,args.tag,args.inventory)
        allocations=[]
        for n1active in range(active+1):
            n1aux=2-n1active
            if 0<=n1aux<=4-active:
                allocations.append({'active_number1':n1active,'active_number3':active-n1active,
                                   'aux_number1':n1aux,'aux_number3':4-active-n1aux,
                                   'canonical_number':256,'certified':result.get('certified',False)})
        result.update(physical_number_allocations=allocations,lambda1_ge_minus5123_certified=result.get('certified',False),
                      lower_form_failed=result.get('negative_upper',0)>0,
                      full_H_failure_proved=False,
                      role_control={'active_odd_slots':active,'spectators_expected':[2-active,122+active],
                                    'counts_preserve_physical_roles':list(result['spectator_counts'])==[2-active,122+active],
                                    'all_aux_substitution_applicable':active>0,
                                    'wrong_all_aux_role_rejected':(profiles()[0]['spinor'][0]!=profiles()[1]['spinor'][0]) if active else None})
        families.append(result)
    receipt={'kind':'EXACT_FOUR_ODD_NEUTRAL_CONTAINING_SYMMETRIC_SPECTATOR_LOWER',
             'floor_exact':'-5123','families':families,'controls':controls,'proof':PROOF,
             'source_sha256':own,'input_sha256':pins,
             'source_unchanged':digest(Path(__file__))==own,
             'inputs_unchanged':all(digest(ROOT/f)==pin for f,pin in pins.items()),
             'rho_coefficient_exact':'64','rho_squared_once':True,
             'spectator_containing_space':'Sym^(2-active)(vector5_active) tensor Sym^(122+active)(vector5_aux)',
             'scope':'four odd slots, balanced two number1/two number3; remaining vector5 symmetric spectators containing source',
             'canonical_fullcell_ground':False,'ground':False,'phase':False,'pole':False}
    path=ROOT/f'{args.tag}_receipt.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'receipt':path.name,'sha256':digest(path),'certified':[r.get('certified') for r in families]}),flush=True)

if __name__=='__main__':
    main()

from __future__ import annotations

import argparse
import itertools
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as s
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu

import neutral_return_lower_control_20261011 as arithmetic
from neutral_return_lower_control_20261011 import (
    BITS, Q, ZERO, add, bareiss, digest, endpoints, inv, iv, mid, mul,
    neg, occupations, profiles, scale, sq_upper, sqrt, sub,
)

ROOT = Path(__file__).resolve().parent
SOURCE_PIN = 'f2a3ff7efce4960ee38a32055e9daf6ae97e465ab23514c7bee206b60a32338f'
INPUTS = dict(arithmetic.INPUTS)
INPUTS['neutral_return_lower_control_20261011.py'] = SOURCE_PIN
ROLES = [r for r in itertools.product((0, 1), repeat=3) if r.count(0) <= 2]


def second(M, n):
    masks = [sum(1 << j for j in w) for w in itertools.combinations(range(4), n)]
    out = s.zeros(len(masks))
    for col, mask in enumerate(masks):
        for q in range(4):
            if not mask >> q & 1:
                continue
            tail = mask ^ (1 << q)
            sign = (-1) ** ((mask & ((1 << q) - 1)).bit_count())
            for p in range(4):
                if not tail >> p & 1:
                    out[masks.index(tail | (1 << p)), col] += sign * (-1) ** ((tail & ((1 << p) - 1)).bit_count()) * M[p, q]
    return out, masks


def controls():
    # Independent generic CAR derivation, not an old solver/checker invocation.
    a, b, e, c = s.symbols('a b e c', real=True)
    m, d = e * (a*a-b*b), 2*e*a*b
    X = s.Matrix([[0, 1], [1, 0]])
    H = m*s.diag(-1, 1, -1, 1) + c*s.kronecker_product(X, s.diag(1, -1)) + d*s.Matrix([[0,0,0,s.I],[0,0,s.I,0],[0,-s.I,0,0],[-s.I,0,0,0]])
    R = s.diag(s.Rational(1,2), s.Rational(1,2), -s.Rational(1,2), -s.Rational(1,2))
    wp = s.Matrix([[1,0],[0,1],[1,0],[0,-1]])/s.sqrt(2)
    wm = s.Matrix([[1,0],[0,1],[-1,0],[0,1]])/s.sqrt(2)
    U = s.Matrix.hstack(wp*s.Matrix([a,-s.I*b]), wm*s.Matrix([a,s.I*b]), s.I*wp*s.Matrix([-s.I*b,a]), s.I*wm*s.Matrix([s.I*b,a]))
    D4 = s.diag(-e+c, -e-c, e+c, e-c)
    f, g = (a*a-b*b)/2, a*b
    R4 = s.Matrix([[0,f,0,-g],[f,0,g,0],[0,g,0,f],[-g,0,f,0]])
    def zero(M):
        return all(s.rem(s.expand(z), a*a+b*b-1, b) == 0 for z in M)
    assert zero(U.conjugate().T*U-s.eye(4))
    assert zero(U.conjugate().T*H*U-D4) and zero(U.conjugate().T*R*U-R4)
    h2, masks2 = second(D4, 2)
    r2, _ = second(R4, 2)
    V = s.zeros(6)
    for j, (mask, coef) in enumerate([(3,1),(5,1),(9,1/s.sqrt(2)),(10,1),(12,-1),(9,1/s.sqrt(2))]):
        V[masks2.index(mask), j] = coef
    V[masks2.index(6), 2] = 1/s.sqrt(2)
    V[masks2.index(6), 5] = -1/s.sqrt(2)
    D6 = s.diag(-2*e, 2*c, 0, -2*c, 2*e, 0)
    R6 = s.zeros(6)
    for j in (1,3):
        for i, value in ((0,g),(2,s.sqrt(2)*f),(4,-g)):
            R6[i,j] = R6[j,i] = value
    assert zero(V.T*V-s.eye(6)) and zero(h2*V-V*D6) and zero(r2*V-V*R6)
    wrong = V.copy(); wrong[:,4] = -wrong[:,4]
    assert zero(h2*wrong-wrong*D6) and not zero(r2*wrong-wrong*R6)
    h3, masks3 = second(H, 3)
    r3, _ = second(R, 3)
    J = s.zeros(4)
    for p in range(4):
        J[masks3.index(15 ^ (1 << p)), p] = (-1)**p
    Q4 = s.kronecker_product(s.eye(2), X)
    S4 = s.diag(1,-1,1,-1)
    U3 = J*Q4*U*S4
    assert zero(U3.conjugate().T*U3-s.eye(4))
    assert zero(U3.conjugate().T*h3*U3-D4) and zero(U3.conjugate().T*r3*U3-R4)
    wrong_hole = J*Q4*U
    assert zero(wrong_hole.conjugate().T*h3*wrong_hole-D4)
    assert not zero(wrong_hole.conjugate().T*r3*wrong_hole-R4)
    for n in (0,4):
        hn, _ = second(H,n); rn, _ = second(R,n)
        assert zero(hn) and zero(rn)
    assert D6[5,5] == 0 and R6[:,5] == s.zeros(6,1) and R6[5,:] == s.zeros(1,6)
    pp = profiles()
    # Same-law actual profiles must not be equalized just because c,d agree.
    assert pp[0]['vector'][0][0] != pp[1]['vector'][0][0]
    scalar_omission = sub(add(scale(pp[1]['vector'][0][0],125), scale(pp[0]['spinor'][0][0],2)), iv(-5123))
    assert scalar_omission[1] < 0
    return {
        'generic_full_number1_number3_joint_pair': True,
        'generic_full_wedge2_scalar_plus_vector6_joint_pair': True,
        'number3_isometry': 'U3=J*(I2 tensor X)*U*diag(1,-1,1,-1); J[15 xor 2^p,p]=(-1)^p',
        'highest_wedge_energy_only_phase_rejected': True,
        'hole_energy_only_phase_rejected': True,
        'scalar_omission_wrongcase': {'roles_vector_number1_number3':[1,0,0], 'scalar_free_H_minus_z_upper_exact':endpoints(scalar_omission)[1], 'reason':'Scalar6 has zero density and zero free energy but this complete scalar branch is below the omitted-free cutoff; vector5-only input cannot certify this branch.'},
        'role_equalization_rejected': {'actual_distinct_active_auxiliary_free_ground_exact_intervals':[endpoints(pp[r]['vector'][0][0]) for r in (0,1)]},
        'empty_full_exact_replacement': 'Both n=0 and n=4 have D=0,R=0 exactly because trace(H)=trace(R)=0. The full wedge2 singlet column also has D=R=0. At a fixed role replacing that singlet by empty/full intertwines the whole sum D and totalrho, hence totalrho squared, exactly; all other roles and number1/3 slots remain unchanged. Canonical number offset is -2/+2 respectively; no energy offset.',
        'empty_full_applicability_limit': 'One empty/full with one1/one3 has total N=254/258 and is not canonical N=256. However scalar+one1+one3 at the SAME roles and spectator counts is jointly unitarily equivalent to n0+n3+n3 and n4+n1+n1: both triples have number6, so total N=256. Thus restriction of this full6 certificate to scalar proves those two additional canonical classes without another solve. Opposite odd orientation is equivalent by the same signed n3 joint isometry. Additional zero/full or spectator scalar slots remain unpaid here.',
        'spectator_scope': 'Remaining slots are symmetric vector5 in each role; fixed identities or equivalent identical-role permutations of the three distinguished slots are covered. Other spectator permutation irreps and spectator scalar6 slots are not silently admitted.',
    }


def construct(roles, output_budget, support_budget):
    pp = profiles()
    counts = [2-roles.count(0), 126-roles.count(1)]
    targets = [(roles[0],'vector'), (roles[1],'spinor'), (roles[2],'spinor')]
    base = ZERO
    for r,n in enumerate(counts):
        base = add(base,scale(pp[r]['vector'][0][0],n))
    for r,k in targets:
        base = add(base,pp[r][k][0][0])
    states = []
    for a,ea in occupations(counts[0],pp[0]['vector'][0],support_budget):
        for b,eb in occupations(counts[1],pp[1]['vector'][0],support_budget):
            cost0 = add(ea,eb)
            if cost0[0] > support_budget*Q:
                continue
            for t in itertools.product(range(6),range(4),range(4)):
                cost = cost0
                for (r,k),j in zip(targets,t):
                    en = pp[r][k][0]
                    cost = add(cost,sub(en[j],en[0]))
                if cost[0] <= support_budget*Q:
                    parity = (a[1]+a[3]+b[1]+b[3]+(t[0]%2 if t[0]!=5 else 0)+t[1]%2+t[2]%2)%2
                    states.append(((a,b,t),cost,parity))
    index = {key:j for j,(key,_,_) in enumerate(states)}
    def neighbors(key):
        a,b,t = key
        for r in (0,1):
            w = a if r==0 else b
            for (dest,src),v in pp[r]['vector'][1].items():
                if w[src]:
                    nw = list(w); nw[src]-=1; nw[dest]+=1
                    key2 = (tuple(nw),b,t) if r==0 else (a,tuple(nw),t)
                    yield key2,mul(v,sqrt(iv(w[src]*(w[dest]+1))))
        for pos,(r,k) in enumerate(targets):
            for (dest,src),v in pp[r][k][1].items():
                if src==t[pos]:
                    nt=list(t); nt[pos]=dest
                    yield (a,b,tuple(nt)),v
    maxstep = max(sub(max(pp[r][k][0],key=lambda x:x[1]), pp[r][k][0][0])[1] for r,k in [(0,'vector'),(1,'vector')]+targets)
    assert output_budget*Q+maxstep < support_budget*Q
    outputs = [j for j,(_,cost,_) in enumerate(states) if cost[1] <= output_budget*Q]
    edge_count = 0
    for j in outputs:
        for key,v in neighbors(states[j][0]):
            assert key in index, ('missing full density image',key)
            assert states[index[key]][2] != states[j][2]
            edge_count += 1
    required = next(key for j in outputs for key,_ in neighbors(states[j][0]))
    wrong = set(index)-{required}
    assert any(key not in wrong for j in outputs for key,_ in neighbors(states[j][0]))
    assert any(key[2][0]==5 for key,_,_ in states)
    return base,states,index,outputs,neighbors,edge_count,counts,maxstep


def certify(roles, output_budget, support_budget, tag):
    start = time.monotonic()
    base,states,index,outputs,neighbors,edge_count,counts,maxstep = construct(roles,output_budget,support_budget)
    D = [sub(add(base,cost),iv(-5123)) for _,cost,_ in states]
    assert base[0]+support_budget*Q > -5123*Q
    assert all(x[1]<0 or x[0]>0 for x in D)
    result = {'roles_vector_number1_number3':roles,'spectator_counts':counts,'floor_exact':'-5123','lambda_eigenvalue_index':1,'base_exact_interval':endpoints(base),'input_support':len(states),'output_rows':len(outputs),'density_edges':edge_count,'scalar_input_columns':sum(key[2][0]==5 for key,_,_ in states),'output_budget':output_budget,'support_budget':support_budget,'maximum_density_energy_step_upper_exact':str(Fraction(maxstep,Q)),'omitted_free_H_minus_z_lower_exact':str(Fraction(base[0]+support_budget*Q+5123*Q,Q)),'positive_output_lower_form':'Hlo-z=D+64*(Pout rho_all)^*(Pout rho_all); rho_all includes both spectator roles and all three distinguished slots once; external charge u=0. Each output row has its complete input image. The complement has strictly positive D and zero retained density image. No input PHP and no borrowed source gap.','omitted_spectrum_proof':'Each of the five vector occupation gaps is nonnegative, non-ground gaps strictly positive. Recursive lower-endpoint occupation limits exclude only continuations of cost>support_budget; every full6 x full4 x full4 index is exhausted. The maximum one-density cost step bound plus output_budget<support_budget guarantees the entire retained-output image.','input_crop_adverse_control_rejected':True,'parity_blocks':[]}
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
                    entries.append((row,value)); rr.append(row); cc.append(col); vv.append(mid(value))
            columns[j] = entries
        B = coo_matrix((vv,(rr,cc)),shape=(len(rows),len(inputs))).tocsr()
        ii = {j:c for c,j in enumerate(inputs)}
        BL = B[:,[ii[j] for j in L]].toarray()
        BQ = B[:,[ii[j] for j in positive]]
        dq = np.array([mid(D[j]) for j in positive])
        K = eye(len(rows),format='csc')+64*(BQ.multiply(1/dq)@BQ.T).tocsc()
        Y = splu(K).solve(BL)
        witness = ROOT/f'{tag}_{"".join(map(str,roles))}_p{parity}_witness.npz'
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
                G[t][u] = add(G[t][u],v[u]); G[u][t] = add(G[u][t],v[u])
        for row in YI:
            for t in range(k):
                for u in range(t,k):
                    v = mul(row[t],row[u]); G[t][u] = sub(G[t][u],v)
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
                    val = mul(v[t],weighted[u]); G[t][u] = sub(G[t][u],val)
                    if t!=u: G[u][t] = sub(G[u][t],val)
        S = [[scale(G[t][u],64) for u in range(k)] for t in range(k)]
        for t,j in enumerate(L): S[t][t] = add(S[t][t],D[j])
        center = [[(x[0]+x[1])//2 for x in row] for row in S]
        assert all(center[t][u]==center[u][t] for t in range(k) for u in range(k))
        radius = max((sum(max(center[t][u]-S[t][u][0],S[t][u][1]-center[t][u]) for u in range(k)) for t in range(k)),default=0)
        residual = sum(sq_upper(x) for row in RY for x in row)
        low,high = [row[:] for row in center],[row[:] for row in center]
        for t in range(k): low[t][t]-=radius; high[t][t]+=radius+64*residual
        ilow,ihigh = bareiss(low),bareiss(high)
        matrix = ROOT/f'{tag}_{"".join(map(str,roles))}_p{parity}_matrix.json'
        matrix.write_text(json.dumps({'bits':BITS,'center_integer':center,'radius_integer':radius,'residual_squared_upper_integer':residual,'lower_integer':low,'upper_integer':high,'inertia_lower':ilow,'inertia_upper':ihigh})+'\n')
        result['parity_blocks'].append({'parity':parity,'negative_free_columns':k,'lower_inertia':ilow,'upper_inertia':ihigh,'matrix_path':matrix.name,'matrix_sha256':digest(matrix),'witness_path':witness.name,'witness_sha256':digest(witness),'radius_upper_exact':str(Fraction(radius,Q)),'residual_squared_upper_exact':str(Fraction(residual,Q))})
    nl = sum(b['lower_inertia']['negative'] for b in result['parity_blocks'])
    nu = sum(b['upper_inertia']['negative'] for b in result['parity_blocks'])
    result.update(certified_lambda1_above_z=nl==nu==0,negative_lower=nl,negative_upper=nu,elapsed_seconds=time.monotonic()-start)
    result['conclusion'] = 'H0 >= -5123 on the stated containing space' if nl==nu==0 else ('Chosen positive-output lower form has an exact negative direction; this lower certificate fails, not a full-H0 no-go' if nu>0 else 'Outward interval inertia unresolved; no positivity claim')
    print(json.dumps(result),flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag',required=True)
    p.add_argument('--output-budget',type=int,default=120)
    p.add_argument('--support-budget',type=int,default=202)
    args = p.parse_args()
    assert args.tag.startswith('canonical_odd_vector_')
    for f,pin in INPUTS.items(): assert digest(ROOT/f)==pin,f
    binding = {f:digest(ROOT/f) for f in INPUTS}
    for f in ('neutral_return_lower_boundary_20261011.py','neutral_return_lower_boundary_actual2_20261011_receipt.json','neutral_return_image_API_20261011.json'):
        binding[f] = digest(ROOT/f)
    own = digest(Path(__file__))
    proof = controls()
    results = [certify(r,args.output_budget,args.support_budget,args.tag) for r in ROLES]
    receipt = {'kind':'EXACT_CANONICAL_ODD_PAIR_FULL6_DISTINGUISHED_SYMMETRIC_VECTOR5_CONTAINING_SPACE','source_sha256':own,'input_sha256':binding,'joint_and_adverse_controls':proof,'role_assignments':results,'all_role_assignments_certified':all(r['certified_lambda1_above_z'] for r in results),'certificate_proof':'Let B=Pout rho_all, split diagonal D into negative L and positive Q. K=I+64 BQ DQ^-1 BQ*. K>=I. For arbitrary binary rational Y, G=BL*Y+Y*BL-Y*K*Y satisfies BL*K^-1 BL-G=(BL-KY)*K^-1(BL-KY)>=0 and <=||BL-KY||F^2 I. Schur S=DL+64 BL*K^-1 BL lies between outward rational low/high matrices. Exact integer Bareiss pivot signs certify inertia of each. Positive DQ and the omitted free complement complete the containing-space inertia; no neutral spectral gap used.','nonclaims':['fullcanonical ground','fullPhi ground','lattice ground','phase','pole'],'inputs_unchanged':all(digest(ROOT/f)==h for f,h in binding.items()),'source_unchanged':digest(Path(__file__))==own}
    out = ROOT/f'{args.tag}_receipt.json'
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'receipt':out.name,'sha256':digest(out),'all_role_assignments_certified':receipt['all_role_assignments_certified']}),flush=True)


if __name__=='__main__':
    main()

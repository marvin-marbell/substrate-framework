from __future__ import annotations

import argparse
import hashlib
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
    'charged_spinor_joint_free_phase_control_20261011_receipt.json': '8276441f71408879d494111e5e52ab3f2afe6b1fdee1f905d9484365ceccb2eb',
    'charged_spinor_joint_free_phase_control_20261011.py': '2b3e220e0cd5dab4934b9fd7c0937e08b59d7a59d599deefcede9aedf8c9e206',
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
    p = (a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1])
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


def exact_endpoints(a):
    return [str(Fraction(x, Q)) for x in a]


ZERO = iv(0)


def profiles():
    p = json.loads((ROOT / next(iter(INPUTS))).read_bytes())
    assert p['corrected_source_unitarity_H_density_identities']
    result = []
    for item in p['profiles']:
        m, d, c = (iv(item[k+'_exact']) for k in ('m', 'd', 'c0'))
        assert Fraction(item['e_squared_exact']) == Fraction(item['m_exact'])**2 + Fraction(item['d_exact'])**2
        e = sqrt(iv(item['e_squared_exact']))
        energy = [scale(e,-2), scale(c,2), ZERO, scale(c,-2), scale(e,2)]
        r = {}
        for b in (1,3):
            for a,v in ((0,mul(d,inv(scale(e,2)))),(2,mul(m,inv(mul(sqrt(iv(2)),e)))),(4,neg(mul(d,inv(scale(e,2)))))):
                r[a,b] = r[b,a] = v
        se = [add(neg(e),c),sub(neg(e),c),add(e,c),sub(e,c)]
        sr = {}
        for a,b,v in ((0,1,mul(m,inv(scale(e,2)))),(2,3,mul(m,inv(scale(e,2)))),(0,3,neg(mul(d,inv(scale(e,2))))),(1,2,mul(d,inv(scale(e,2))))):
            sr[a,b] = sr[b,a] = v
        assert all((a % 2) != (b % 2) for a,b in r)
        assert all((a % 2) != (b % 2) for a,b in sr)
        result.append((energy,r,se,sr))
    control = json.loads((ROOT/'charged_spinor_joint_free_phase_control_20261011_receipt.json').read_bytes())
    assert control['exact_joint_spinor_D_R'] and control['matches_existing_corrected_free5_joint_pair']
    assert control['full_number3_hole_D_R_verified'] and control['affine_half_offsets_have_same_spectrum_by_whole_R_flip']
    return result


def occupations(n, energy, budget):
    gaps = [sub(x,energy[0]) for x in energy]
    out = []
    def visit(j, remaining, cost, word):
        if j == 5:
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
        if not pivot:
            return {'certified':False,'reason':'zero leading pivot','pivot_index':t}
        signs.append(1 if pivot*previous > 0 else -1)
        for i in range(t+1,len(a)):
            for j in range(t+1,len(a)):
                numerator = a[i][j]*pivot-a[i][t]*a[t][j]
                assert numerator % previous == 0
                a[i][j] = numerator//previous
        previous = pivot
    return {'certified':True,'negative':signs.count(-1),'positive':signs.count(1),'pivot_signs':signs}


def adverse_control():
    # Actual joint four-component spinor, exact rational e=5,m=3,d=4,c=-1.
    import sympy as s
    R = s.Matrix([[0,s.Rational(3,10),0,-s.Rational(2,5)],
                  [s.Rational(3,10),0,s.Rational(2,5),0],
                  [0,s.Rational(2,5),0,s.Rational(3,10)],
                  [-s.Rational(2,5),0,s.Rational(3,10),0]])
    D = s.diag(-6,-4,4,6)
    A = R+s.eye(4)/2
    P = s.diag(1,0,0,0)
    correct = D+64*A*A
    output_lower = D+64*A*P*A
    input_substitution = D+64*(P*A*P)*(P*A*P)
    assert output_lower != input_substitution
    assert correct-(D+64*R*R) == 64*R+16*s.eye(4)
    assert (correct-output_lower).is_positive_semidefinite
    missing = [j for j in range(4) if A[0,j] != 0 and j != 0]
    assert missing == [1,3]
    assert correct[0,0] == 26 and input_substitution[0,0] == 10
    return {'actual_spinor_rational_profile':{'e':5,'m':3,'d':4,'c0':-1},
            'input_PHP_rejected':True,'missing_output_row_image_columns':missing,
            'omitted_affine_u_rejected':True,'correct_00':str(correct[0,0]),
            'wrong_PHP_00':str(input_substitution[0,0]),
            'affine_difference':'64R+16I','all_four_spinor_components_retained':True}


def certify(branch, floor, output_budget, support_budget, tag, inventory=False):
    started = time.monotonic()
    pp = profiles()
    counts, spinrole = ((1,126),0) if branch == 'active' else ((2,125),1)
    base = ZERO
    for n,(energy,_,_,_) in zip(counts,pp):
        base = add(base,scale(energy[0],n))
    se, sr = pp[spinrole][2:]
    base = add(base,se[0])
    sg = [sub(x,se[0]) for x in se]
    aa = occupations(counts[0],pp[0][0],support_budget)
    bb = occupations(counts[1],pp[1][0],support_budget)
    states = [(a,b,s,add(add(ea,eb),sg[s])) for a,ea in aa for b,eb in bb for s in range(4)
              if add(add(ea,eb),sg[s])[0] <= support_budget*Q]
    index = {(a,b,s):j for j,(a,b,s,_) in enumerate(states)}
    outputs = [j for j,(_,_,_,cost) in enumerate(states) if cost[1] <= output_budget*Q]
    row_index = {j:r for r,j in enumerate(outputs)}
    def neighbors(state):
        a,b,s,_ = state
        for role in (0,1):
            word = a if role == 0 else b
            for (dest,src),value in pp[role][1].items():
                if word[src]:
                    nw = list(word);nw[src] -= 1;nw[dest] += 1
                    key = (tuple(nw),b,s) if role == 0 else (a,tuple(nw),s)
                    yield key,mul(value,sqrt(iv(word[src]*(word[dest]+1))))
        for (dest,src),value in sr.items():
            if src == s:
                yield (a,b,dest),value
        yield (a,b,s),iv(Fraction(1,2))
    # Every nonzero column in every retained output row must be present.
    image_edges = 0
    for j in outputs:
        for key,_ in neighbors(states[j]):
            assert key in index, ('incomplete output image',branch,j,key)
            image_edges += 1
    z = iv(floor)
    D = [sub(add(base,cost),z) for _,_,_,cost in states]
    L = [j for j,x in enumerate(D) if x[1] < 0]
    positive = [j for j,x in enumerate(D) if x[0] > 0]
    assert len(L)+len(positive) == len(states)
    outside = base[0]+support_budget*Q
    assert outside > z[1]
    result = {'branch':branch,'containing_space':f'Sym^{counts[0]}(C5) x Sym^{counts[1]}(C5) x C4',
              'floor_exact':str(Fraction(floor)),'output_budget':output_budget,'support_budget':support_budget,
              'output_rows':len(outputs),'input_support':len(states),'negative_free_columns':len(L),
              'free_base_interval_exact':exact_endpoints(base),'outside_free_floor_lower_exact':str(Fraction(outside,Q)),
              'entire_affine_output_image_checked':True,'checked_nonzero_image_edges':image_edges,
              'u_exact':'1/2','both_u_signs_by_whole_R_flip':True,'unprojected_Sym_no_trace_crop':True}
    print(json.dumps(result),flush=True)
    if inventory:
        return result
    columns = [[] for _ in states]
    rr,cc,vv = [],[],[]
    # Build B=P_output(Rtotal+uI), not an input PHP crop.
    for j,state in enumerate(states):
        for key,value in neighbors(state):
            target = index.get(key)
            if target in row_index:
                row = row_index[target]
                columns[j].append((row,value));rr.append(row);cc.append(j);vv.append(mid(value))
    B = coo_matrix((vv,(rr,cc)),shape=(len(outputs),len(states))).tocsr()
    BL = B[:,L].toarray()
    BQ = B[:,positive]
    dq = np.array([mid(D[j]) for j in positive])
    K = eye(len(outputs),format='csc')+64*(BQ.multiply(1/dq)@BQ.T).tocsc()
    # Binary sparse factorization produces only a witness. No float positivity claim.
    Y = splu(K).solve(BL)
    witness_path = ROOT/f'{tag}_{branch}_witness.npz'
    np.savez_compressed(witness_path,Y=Y,states=np.array([(a+b+(s,)) for a,b,s,_ in states],dtype=np.int64),outputs=np.array(outputs),negative=np.array(L))
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
                if t != u:G[u][t] = sub(G[u][t],v)
    RY = [[neg(x) for x in row] for row in YI]
    for t,j in enumerate(L):
        for row,b in columns[j]:
            RY[row][t] = add(RY[row][t],b)
    for j in positive:
        v = product(j)
        weighted = [scale(mul(x,inv(D[j])),64) for x in v]
        for row,b in columns[j]:
            for t in range(k):RY[row][t] = sub(RY[row][t],mul(b,weighted[t]))
        for t in range(k):
            for u in range(t,k):
                val = mul(v[t],weighted[u]);G[t][u] = sub(G[t][u],val)
                if t != u:G[u][t] = sub(G[u][t],val)
    S = [[scale(G[t][u],64) for u in range(k)] for t in range(k)]
    for t,j in enumerate(L):S[t][t] = add(S[t][t],D[j])
    center = [[(x[0]+x[1])//2 for x in row] for row in S]
    assert all(center[t][u] == center[u][t] for t in range(k) for u in range(k))
    radius = max((sum(max(center[t][u]-S[t][u][0],S[t][u][1]-center[t][u]) for u in range(k)) for t in range(k)),default=0)
    residual = sum(sq_upper(x) for row in RY for x in row)
    low = [row[:] for row in center]
    high = [row[:] for row in center]
    for t in range(k):
        low[t][t] -= radius
        high[t][t] += radius+64*residual
    ilow,ihigh = bareiss(low),bareiss(high)
    certified = ilow['certified'] and ilow['negative'] == 0
    result.update(lambda1_lower_certified=certified,lower_inertia=ilow,upper_inertia=ihigh,
                  interval_matrix_radius_upper_exact=str(Fraction(radius,Q)),
                  witness_residual_frobenius_squared_upper_exact=str(Fraction(residual,Q)),
                  witness_path=witness_path.name,witness_sha256=digest(witness_path),
                  elapsed_seconds=time.monotonic()-started)
    matrix_path = ROOT/f'{tag}_{branch}_exact_matrix.json'
    matrix_path.write_text(json.dumps({'bits':BITS,'center_integer':center,'radius_integer':radius,
                                      'residual_squared_upper_integer':residual,'lower_integer':low,
                                      'upper_integer':high,'inertia_lower':ilow,'inertia_upper':ihigh})+'\n')
    result['exact_matrix_path'] = matrix_path.name
    result['exact_matrix_sha256'] = digest(matrix_path)
    print(json.dumps(result),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--floor',default='-5110')
    parser.add_argument('--output-budget',type=int,default=240)
    parser.add_argument('--support-budget',type=int,default=322)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--inventory',action='store_true')
    args = parser.parse_args()
    for name,pin in INPUTS.items():assert digest(ROOT/name) == pin, name
    own = digest(Path(__file__))
    adverse = adverse_control()
    branches = [certify(b,Fraction(args.floor),args.output_budget,args.support_budget,args.tag,args.inventory) for b in ('active','aux')]
    upper = Fraction(json.loads((ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_bytes())['ground_energy_interval_exact'][1])
    gap = Fraction(64)+2*Fraction(args.floor)-2*upper
    done = not args.inventory and all(b['lambda1_lower_certified'] for b in branches)
    receipt = {'kind':'CHARGED_LAMBDA1_OUTWARD_OUTPUT_ROW_SCHUR_BAREISS_CERTIFICATE','bits':BITS,
               'operator':'Dtotal+64(Rtotal+uI)^2','physical_hopping':1,'branches':branches,
               'both_branches_certified':done,'both_u_signs_exact_unitary_equivalent':True,
               'R_sign_flip':'S5=diag(1,-1,1,-1,1), S4=diag(1,-1,1,-1); tensor product commutes D and sends Rtotal to -Rtotal; (R+u)^2 -> (R-u)^2',
               'adverse_control':adverse,'E0_upper_exact':str(upper),'induced_two_cell_gap_lower_exact':str(gap) if done else None,
               'induced_two_cell_gap_positive':bool(done and gap>0),'two_cell_price':'64+LbranchA+LbranchB-2E0upper; 64 ONCE for external edge',
               'input_sha256':INPUTS,'own_source_sha256':own,
               'inputs_unchanged':all(digest(ROOT/n)==h for n,h in INPUTS.items()),'source_unchanged':digest(Path(__file__))==own,
               'nonclaims':['neutral gap as charged lower','fullPhi or lattice ground','phase or stiffness','pole','IRatom','commoncone','originalSOURCE adoption'],
               'proof':'H>=Hlo=D+64 A*Pout A. Enumerate all free-cost columns through support with lower-endpoint pruning; every output row entire A image retained, so outside block exactly D>floor. Split finite support by negative/positive D-floor. K=I+64 BQ DQ^-1 BQ*. Exact Schur is DL+64 BL* K^-1 BL. Any dyadic Y gives G(Y)=BL*Y+Y*BL-Y*KY <= BL*K^-1 BL; K>=I also gives error <= residual*residual. Outward 100-bit intervals enclose G and DL; symmetric row-sum interval radius prices all rounding. Integer Bareiss inertia of lower enclosure ZERO negative with nonzero pivots proves positive definite Schur, hence H>=floor I. Binary factorization ONLY supplies Y, never a lower spectral bound.'}
    path = ROOT/f'{args.tag}_receipt.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'receipt':path.name,'sha256':digest(path),'certified':done,'gap_exact':str(gap) if done else None}),flush=True)


if __name__ == '__main__':
    main()

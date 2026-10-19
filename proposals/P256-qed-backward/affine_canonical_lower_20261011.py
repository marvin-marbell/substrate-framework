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

import neutral_return_lower_control_20261011 as original
import canonical_four_odd_lower_20261011 as quartet
import canonical_odd_vector_control_20261011 as mixed
from neutral_return_lower_control_20261011 import (
    BITS, Q, ZERO, add, sub, neg, mul, inv, scale, iv, mid, sq_upper,
    endpoints, bareiss, digest, profiles,
)
from neutral_return_lower_boundary_20261011 import second, generic_joint_phase

ROOT = Path(__file__).resolve().parent
FLOOR = -5123
FAMILIES = ([('vector', r) for r in ((0,), (1,), (0,0), (0,1), (1,1))]
            + [('spinor', r) for r in ((0,0), (0,1), (1,1))]
            + [('quartet', r) for r in ((0,0,1,1), (0,1,1,1), (1,1,1,1))]
            + [('mixed', r) for r in itertools.product((0,1), repeat=3) if r.count(0)<=2])


def controls():
    phase = generic_joint_phase()
    # Exact native CAR control with a rational free frame. The phase involution
    # is transported by the accepted simultaneous h/rho frame, not fitted spectra.
    D = s.diag(-6,-4,4,6)
    R = s.Matrix([[0,s.Rational(3,10),0,-s.Rational(2,5)],
                  [s.Rational(3,10),0,s.Rational(2,5),0],
                  [0,s.Rational(2,5),0,s.Rational(3,10)],
                  [-s.Rational(2,5),0,s.Rational(3,10),0]])
    P = s.diag(1,-1,1,-1)
    checked = []
    for n in range(5):
        dn,masks = second(D,n)
        rn,_ = second(R,n)
        pn = s.diag(*[s.prod(P[j,j] for j in range(4) if mask>>j&1) for mask in masks])
        assert pn*dn*pn == dn and pn*rn*pn == -rn
        hp = dn+64*(rn+s.eye(len(masks)))**2
        hm = dn+64*(rn-s.eye(len(masks)))**2
        assert pn*hp*pn == hm
        checked.append(n)
    H = D+64*(R+s.eye(4))**2
    Hnoidentity = D+64*R**2
    assert H != Hnoidentity
    assert H[0,1] == s.Rational(192,5)  # 128*rho: lost by parity splitting.
    assert H[0,1] != 0 and P[0,0] != P[1,1]
    # A source-frame sign change preserves D and the isolated rho square but
    # not the fixed affine operator. With two factors it also changes cross rho.
    wrong = P*R*P
    assert wrong**2 == R**2 and D+64*(wrong+s.eye(4))**2 != H
    dt = s.kronecker_product(D,s.eye(4))+s.kronecker_product(s.eye(4),D)
    rt = s.kronecker_product(R,s.eye(4))+s.kronecker_product(s.eye(4),R)
    badrt = s.kronecker_product(wrong,s.eye(4))+s.kronecker_product(s.eye(4),R)
    assert dt+64*(rt+s.eye(16))**2 != dt+64*(badrt+s.eye(16))**2
    # Complete OUTPUT has both identity and rho preimages. This rational
    # control deliberately removes each and reaches a rejecting assertion.
    outputs = [0,1]
    required = {j for i in outputs for j in range(4) if (R+s.eye(4))[i,j] != 0}
    assert required == {0,1,2,3}
    for removed in required:
        assert any((R+s.eye(4))[i,removed] != 0 for i in outputs)
    B = (R+s.eye(4))[outputs,:]
    assert B[0,0] == 1
    assert (B.T*B)[0,1] != 0
    # Input PHP is not a lower bound. An exact cancellation vector is negative
    # in the full form while its compressed scalar looks strictly positive.
    d = s.diag(-s.Rational(3,2),s.Rational(1,10)); r=s.ones(2)
    a=s.Matrix([1,-1]); h=d+r*r
    assert h[0,0]>0 and (a.T*h*a)[0]==-s.Rational(7,5)
    return {'joint_phase':phase,'finiteCAR_numbers_checked':checked,
            'identity_row_wrongcase_rejected':True,'parity_split_wrongcase_rejected':True,
            'source_phase_wrongcase_rejected':True,'complete_output_wrongcase_rejected':True,
            'PHP_wrongcase_rejected':True,'affine_cross_entry_exact':str(H[0,1]),
            'plus_minus_intertwiner':'For each role, physical W=U diag(1,-1,1,-1) U*; W h W*=h, W rho W*=-rho. Tensor all128 exterior powers preserves each slot number, hence canonical256, and conjugates H(+u) to H(-u). The Sym vector factors have phase diag(1,-1,1,-1,1); full6 scalar phase +1 after an irrelevant whole-number2 sign. Signed number3 uses SAME joint free frame. This is an operator intertwiner, not spectrum equality.'}


def construct(kind, roles, output_budget, support_budget):
    if kind == 'mixed':
        return mixed.construct(roles,output_budget,support_budget)[:7]
    supplier = quartet if kind=='quartet' else original
    return supplier.construct('spinor' if kind=='quartet' else kind,roles,output_budget,support_budget)


def certify(kind, roles, u, output_budget, support_budget, tag, floor):
    start=time.monotonic()
    base,states,index,outputs,neighbors,edges,counts=construct(kind,roles,output_budget,support_budget)
    D=[sub(add(base,cost),iv(floor)) for _,cost,_ in states]
    assert base[0]+support_budget*Q > floor*Q
    assert all(x[1]<0 or x[0]>0 for x in D)
    # Merge BOTH old parity classes. uI contributes a same-parity preimage
    # at every retained output; rho contributes the opposite-parity preimages.
    rows=outputs; row_index={j:r for r,j in enumerate(rows)}
    L=[j for j,x in enumerate(D) if x[1]<0]
    positive=[j for j,x in enumerate(D) if x[0]>0]
    columns={}; rr=[];cc=[];vv=[];identity_rows=0
    for j,(key,_,_) in enumerate(states):
        entries=[]
        if j in row_index:
            entries.append((row_index[j],iv(u))); identity_rows+=1
        for key2,value in neighbors(key):
            target=index.get(key2)
            if target in row_index:
                entries.append((row_index[target],value))
        for row,value in entries:
            rr.append(row);cc.append(j);vv.append(mid(value))
        columns[j]=entries
    assert identity_rows==len(outputs)
    for j in rows:
        assert index[states[j][0]] == j
        for key,value in neighbors(states[j][0]):
            assert key in index, ('unpaid output preimage',key)
    B=coo_matrix((vv,(rr,cc)),shape=(len(rows),len(states))).tocsr()
    BL=B[:,L].toarray();BQ=B[:,positive]
    dq=np.array([mid(D[j]) for j in positive])
    K=eye(len(rows),format='csc')+64*(BQ.multiply(1/dq)@BQ.T).tocsc()
    Y=splu(K).solve(BL) if L else np.zeros((len(rows),0))
    stem=f'{tag}_{kind}_{"".join(map(str,roles))}_u{u:+d}'
    witness=ROOT/(stem+'_witness.npz')
    np.savez_compressed(witness,Y=Y,negative=np.array(L),outputs=np.array(rows),
                        states=np.array([a+b+t for (a,b,t),_,_ in states],dtype=np.int64),
                        u=np.array(u),old_parity=np.array([p for _,_,p in states]))
    YI=[[iv(float(x)) for x in row] for row in Y];k=len(L)
    def product(j):
        v=[ZERO]*k
        for row,b in columns[j]:
            for t in range(k):v[t]=add(v[t],mul(b,YI[row][t]))
        return v
    G=[[ZERO]*k for _ in range(k)]
    for t,j in enumerate(L):
        v=product(j)
        for q in range(k):
            G[t][q]=add(G[t][q],v[q]);G[q][t]=add(G[q][t],v[q])
    for row in YI:
        for t in range(k):
            for q in range(t,k):
                val=mul(row[t],row[q]);G[t][q]=sub(G[t][q],val)
                if t!=q:G[q][t]=sub(G[q][t],val)
    RY=[[neg(x) for x in row] for row in YI]
    for t,j in enumerate(L):
        for row,b in columns[j]:RY[row][t]=add(RY[row][t],b)
    for j in positive:
        v=product(j);weighted=[scale(mul(x,inv(D[j])),64) for x in v]
        for row,b in columns[j]:
            for t in range(k):RY[row][t]=sub(RY[row][t],mul(b,weighted[t]))
        for t in range(k):
            for q in range(t,k):
                val=mul(v[t],weighted[q]);G[t][q]=sub(G[t][q],val)
                if t!=q:G[q][t]=sub(G[q][t],val)
    S=[[scale(G[t][q],64) for q in range(k)] for t in range(k)]
    for t,j in enumerate(L):S[t][t]=add(S[t][t],D[j])
    center=[[(x[0]+x[1])//2 for x in row] for row in S]
    assert all(center[t][q]==center[q][t] for t in range(k) for q in range(k))
    radius=max((sum(max(center[t][q]-S[t][q][0],S[t][q][1]-center[t][q]) for q in range(k)) for t in range(k)),default=0)
    residual=sum(sq_upper(x) for row in RY for x in row)
    low=[row[:] for row in center];high=[row[:] for row in center]
    for t in range(k):low[t][t]-=radius;high[t][t]+=radius+64*residual
    ilow,ihigh=bareiss(low),bareiss(high)
    matrix=ROOT/(stem+'_matrix.json')
    with matrix.open('x') as f:
        json.dump({'bits':BITS,'center_integer':center,'radius_integer':radius,
                   'residual_squared_upper_integer':residual,'lower_integer':low,
                   'upper_integer':high,'inertia_lower':ilow,'inertia_upper':ihigh},f)
    # Restrict each mixed full6 scalar block in the NEW merged-parity Schur form.
    # rho+u preserves scalar; all algebra and elimination preserve this block.
    scalar=None
    if kind=='mixed':
        selected=[t for t,j in enumerate(L) if states[j][0][2][0]==5]
        assert all(low[t][q]==high[t][q]==0 for t in selected for q in range(k) if q not in selected)
        scalar={'indices':selected,'lower_inertia':bareiss([[low[t][q] for q in selected] for t in selected]),
                'upper_inertia':bareiss([[high[t][q] for q in selected] for t in selected]),
                'transfers':['n0+3+3','n4+1+1'],'same_D_R_u_intertwiner':True}
    result={'kind':kind,'roles':roles,'u_exact':str(u),'number_sector':256,'spectator_counts':counts,
            'floor_exact':str(floor),'output_budget':output_budget,'support_budget':support_budget,
            'input_support':len(states),'output_rows':len(rows),'identity_rows':identity_rows,
            'density_edges':edges,'merged_parity':True,'negative_free_columns':k,
            'base_exact_interval':endpoints(base),
            'omitted_free_H_minus_z_lower_exact':str(Fraction(base[0]+support_budget*Q-floor*Q,Q)),
            'entire_output_preimages_checked':True,'lower_inertia':ilow,'upper_inertia':ihigh,
            'certified_lambda1_ge_floor':ilow['negative']==0,
            'proven_lower_form_failure':ihigh['negative']>0,
            'matrix_path':matrix.name,'matrix_sha256':digest(matrix),
            'witness_path':witness.name,'witness_sha256':digest(witness),
            'radius_upper_exact':str(Fraction(radius,Q)),
            'residual_squared_upper_exact':str(Fraction(residual,Q)),
            'scalar_transfer':scalar,'elapsed_seconds':time.monotonic()-start}
    print(json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--output-budget',type=int,required=True)
    parser.add_argument('--support-budget',type=int,required=True)
    parser.add_argument('--family',choices=['all','vector0','nonquartet'],default='all')
    parser.add_argument('--floor',type=int,choices=[-5123,-5187],default=-5123)
    parser.add_argument('--u',type=int,choices=[-1,1],required=True)
    args=parser.parse_args();assert args.tag.startswith('affine_canonical_')
    own=digest(Path(__file__));control=controls()
    families=(FAMILIES if args.family=='all' else
              [x for x in FAMILIES if x[0]!='quartet'] if args.family=='nonquartet' else [FAMILIES[0]])
    results=[certify(k,r,args.u,args.output_budget,args.support_budget,args.tag,args.floor) for k,r in families]
    pp=profiles();free=scale(pp[1]['vector'][0][0],126)
    receipt={'kind':'ACTUAL_AFFINE_WHOLE_NEUTRAL_CANONICAL_LOWER_ATTEMPT',
             'law':'H0(u)=D+64(rho_all+u I)^2; active2 auxiliary126; canonical number256 ONLY',
             'controls':control,'families':results,'u_exact':str(args.u),
             'floor_exact':str(args.floor),'original_strong_target_exact':'-5123',
             'separate_weak_construction':args.floor==-5187,
             'free_complement_floor_interval':endpoints(free),
             'free_complement_above_requested_floor':free[0]>args.floor*Q,
             'all18_certified':len(results)==18 and all(x['certified_lambda1_ge_floor'] for x in results),
             'source_sha256':own,'source_unchanged':own==digest(Path(__file__)),
             'no_u0_gap_borrowed':True,'no_new_ground_identification':True,
             'charged254_258_applicability':False}
    path=ROOT/(args.tag+'_receipt.json')
    with path.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({'receipt':path.name,'sha256':digest(path),'all18_certified':receipt['all18_certified']}),flush=True)


if __name__=='__main__':
    main()

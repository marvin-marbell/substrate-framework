"""Independent exact centered/vacuum controls; no historical entrypoint or solver."""
from __future__ import annotations
import hashlib
import itertools
import json
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast
import sympy as sp
import k_resolved_scientist_20261011 as interface

ROOT = Path(__file__).resolve().parent
OUT = 'k_centered_review_actual1_20261011'
GAMMA = F(46955312929, 34359738368)
INPUTS = ['k_resolved_centered_scientist_20261011.py', 'k_resolved_centered_proof_20261011.json', 'k_resolved_scientist_20261011.py', 'k_resolved_refined_scientist_20261011.py', 'k_resolved_refined_proof_20261011.json', 'k_resolved_centered_actual1_20261011_capsule.json', 'k_resolved_refined_actual1_20261011_capsule.json', 'k_resolved_centered_actual1_20261011_receipt.json', 'k_resolved_refined_actual1_20261011_receipt.json', 'k_resolved_actual1_20261011_receipt.json', 'positive_residual_scientist_20261011.py', 'positive_residual_proof_20261011.json', 'positive_residual_actual1_20261011_receipt.json', 'vacuum_connected_native_scientist_20261011.py', 'vacuum_connected_native_parent_budget3_20261011_receipt.json', 'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(name):
    return json.loads((ROOT/name).read_bytes())

def encode(matrix):
    return [[str(x) for x in row] for row in matrix.tolist()]

def psd(matrix):
    assert matrix == matrix.T
    minors = []
    for n in range(1, matrix.rows+1):
        for inds in itertools.combinations(range(matrix.rows), n):
            det = matrix.extract(inds, inds).det()
            assert det >= 0, (matrix, inds, det)
            minors.append(str(det))
    return minors

def sym(left, right):
    return (left*right+right*left)/2

def point_price(u, v, A, weight, g, m, k):
    w = k if weight == 'Q' else k+GAMMA
    return (u*k+v)/(w*(A+k)**2)-g*k/(A*(A+k)*w)+m/w

def cover(u, v, A, weight, g=F(0), m=F(0)):
    # Independent rational interval proof with sign-aware denominators.
    assert u >= 0 and A > 0 and g >= 0 and weight in ('Q', 'S')
    knots = [GAMMA*F(17,16)**j for j in range(230)]
    if weight == 'S':
        knots.insert(0,F(0))
    certified = []
    for left,right in zip(knots,knots[1:]):
        wl = left if weight == 'Q' else left+GAMMA
        wr = right if weight == 'Q' else right+GAMMA
        n = u*right+v
        p = n/(wl*(A+left)**2) if n >= 0 else n/(wr*(A+right)**2)
        bound = p-g*left/(A*(A+right)*wr)+(m/wl if m >= 0 else m/wr)
        # Exact arithmetic controls reach positive/negative numerator and mismatch.
        for k in (left,(left+right)/2,right):
            assert point_price(u,v,A,weight,g,m,k) <= bound
        certified.append(bound)
    last = knots[-1]
    tail = u/(last+A)**2+max(v,F(0))/(last*(last+A)**2)+max(m,F(0))/last
    for k in (last,2*last,100*last):
        assert point_price(u,v,A,weight,g,m,k) <= tail
    return max(certified+[tail])

def finite_control():
    K = sp.diag(0,2,5)
    J = sp.Matrix([[1,1,0],[0,1,1],[1,0,1],[1,2,-1]])
    X = sp.Matrix([[8,1,0,0],[1,9,1,0],[0,1,11,1],[0,0,1,13]])
    G = J.T*J
    g0 = G[0,0]
    Gc = G-g0*sp.eye(3)
    psd(X-6*sp.eye(4))
    T = X*J-J*K-7*J
    B = (J.T*T+T.T*J)/2
    assert K*G != G*K and B*K != K*B
    rows = []
    for E in (sp.Rational(-1),sp.Rational(0),sp.Rational(1,3)):
        A = 7-E
        Fk = (K+A*sp.eye(3)).inv()
        R = (X-E*sp.eye(4)).inv()
        W = J.T*R*J
        P = Fk*T.T*R*T*Fk
        w0 = W[0,0]
        C = W-w0*sp.eye(3)
        m = g0/A-w0
        scalar = -g0*K*Fk/A
        assert W == sym(G,Fk)-Fk*B*Fk+P
        assert C == sym(Gc,Fk)+scalar+m*sp.eye(3)-Fk*B*Fk+P
        psd(P)
        residual_slack = Fk*T.T*T*Fk/(6-E)-P
        psd(residual_slack)
        assert P != sp.zeros(3) and Fk*B*Fk != B*Fk**2
        # Both relative centered intervals valid, including full Pg/Qg offdiagonal.
        S = K+2*sp.eye(3)
        q = [1,2]
        for weight, M, inds, fmax in [('Q',K.extract(q,q),q,1/(A+2)),('S',S,[0,1,2],1/A)]:
            D = Gc.extract(inds,inds)
            psd(5*M-D); psd(5*M+D)
            H = sym(Gc,Fk).extract(inds,inds)
            centered = sp.Rational(25,4)*fmax
            psd(centered*M-H); psd(centered*M+H)
            # Signed B and whole residual paid by exact containing pencils.
            psd(1000*M-B.extract(inds,inds))
            psd(1000*M-(T.T*T).extract(inds,inds))
            bprice = 1000*fmax**2
            rprice = bprice/(6-E)
            upper = centered+max(m,0)/2+rprice
            lower = centered+g0/(A*(A+2))+max(-m,0)/2+bprice
            Cpart = C.extract(inds,inds)
            psd(upper*M-Cpart); psd(lower*M+Cpart)
        assert C[0,0] == 0 and C[0,1] != 0
        determinant = (3*K-C).extract([0,1],[0,1]).det()
        assert determinant < 0  # ANY finite pure-K price fails, independent of 3.
        sourceQsquare = sum(W[i,0]**2 for i in q)
        assert sourceQsquare != w0 and w0 != g0/A
        # Moment bounds use same E and same X, not sourceQ or bare infidelity.
        v = cast(sp.Matrix, J[:,0])
        M0 = (v.T*v)[0]; M1 = (v.T*(X-E*sp.eye(4))*v)[0]
        variance = (v.T*(X-7*sp.eye(4))**2*v)[0]
        polynomial = 2*M0/A-M1/A**2
        wlo = M0**2/M1
        whi = min(M0/(6-E),polynomial+variance/(A**2*(6-E)))
        assert wlo <= w0 <= whi
        rows.append(dict(E=str(E),W=encode(W),C=encode(C),positive_residual=encode(P),vacuum_exact=str(w0),vacuum_interval=[str(wlo),str(whi)],mismatch=str(m),sourceQ_squared=str(sourceQsquare),pureK_slack_minor=str(determinant),fullPgQg_nonzero=str(C[0,1]),residual_slack_principal_minors=psd(residual_slack)))
    # Exact extremal positive contractions: negative anticommutator is real.
    U = sp.Matrix([[1,0],[0,0]])
    V = sp.Matrix([[sp.Rational(1,4),sp.sqrt(3)/4],[sp.sqrt(3)/4,sp.Rational(3,4)]])
    assert U**2 == U and V**2 == V
    psd(sym(U,V)+sp.eye(2)/8)
    z = sp.Matrix([1,-sp.sqrt(3)])
    assert (z.T*sym(U,V)*z)[0]/(z.T*z)[0] == -sp.Rational(1,8)
    # A one-sided upper is not an absolute certificate.
    example = sp.diag(-2,sp.Rational(1,2))
    psd(sp.Rational(1,2)*sp.eye(2)-example)
    assert (sp.eye(2)+example).det() < 0
    return dict(kind='EXACT_NEW_NONCOMMUTING_FULL_PgQg_CONTROL_NOT_NATIVE_GROUND',K=encode(K),J=encode(J),X=encode(X),T=encode(T),B=encode(B),Gc=encode(Gc),rows=rows,positive_contraction_extremal_price='-1/8',upper_not_absolute_counterexample=encode(example),every_claim_reaches_exact_matrix_assertion=True)

def native_audit():
    old = load('positive_residual_actual1_20261011_receipt.json')
    centered = load('k_resolved_centered_actual1_20261011_receipt.json')
    refined = load('k_resolved_refined_actual1_20261011_receipt.json')
    vacuum = load('vacuum_connected_native_parent_budget3_20261011_receipt.json')
    d = F(old['domains']['whole_operator_two_cell_floor_exact'])
    reports = []
    for o,c,r,va in zip(old['axes'],centered['axes'],refined['axes'],vacuum['axes']):
        assert o['axis']==c['axis']==r['axis']==va['axis']
        gL,gU = map(F,o['G_ground_expectation']['exact'])
        gsL,gsU = map(F,o['G_spectrum']['exact'])
        relative = {}
        for weight in ('Q','S'):
            vals = []
            for name,endpoint in [('Gc_upper',max(gsU-gL,F(0))/GAMMA),('minus_Gc_upper',max(gU-gsL,F(0))/GAMMA)]:
                bounds = [endpoint]
                for candidate in c['centered_G_joint_pencils'][name]:
                    u,v = map(F,(candidate['slope_exact'],candidate['intercept_exact']))
                    bounds.append(u+max(v,F(0))/GAMMA if weight=='Q' else max(u,v/GAMMA))
                vals.append(min(bounds))
            relative[weight]=vals
        original = o['scalar_anchor_sweep'][3]
        for oo,cc,rr,vv in zip(original['energies'],c['energies'],r['energies'],va['energies']):
            E = F(oo['E_exact']); A=F(original['anchor_exact'])-E
            assert A>0 and d>E and oo['E_exact']==cc['E_exact']==rr['E_exact']==vv['E_exact']
            assert oo['vacuum_w0']==vv['vacuum_w0']
            wl,wu=map(F,oo['vacuum_w0']['exact'])
            M0 = [max(F(0),F(va['native_charged_moments'][0]['exact'][0])),F(va['native_charged_moments'][0]['exact'][1])]
            M1lo,M1hi = map(F,va['native_charged_moments'][1]['exact'])
            M2lo,M2hi = map(F,va['native_charged_moments'][2]['exact'])
            a=F(va['anchor_exact']); ae=a-E; be=F(vacuum['domains']['source_two_cell_floor_exact'])-E
            # Interval variance upper: M2-2a M1+a^2 M0, a>0.
            variance=max(F(0),M2hi-2*a*M1lo+a*a*M0[1])
            polyhi=(2/ae+E/ae**2)*M0[1]-M1lo/ae**2
            assert wl==M0[0]**2/(M1hi-E*M0[0] if E>=0 else M1hi-E*M0[1])
            assert wu==min(M0[1]/be,polyhi+variance/(ae**2*be))
            derived={}
            for weight in ('Q','S'):
                u,l=relative[weight]; denom=A+GAMMA if weight=='Q' else A
                cu=(u+(u+l)/8)/denom; cl=min(max(u,l),l+(u+l)/8)/denom
                assert cu==F(cc[f'centered_G_{weight}_upper_exact']) and cl==F(cc[f'centered_G_{weight}_lower_exact'])
                b=min(interface.scalar_price(F(x['slope_exact']),F(x['intercept_exact']),A,weight) for x in c['signed_B_joint_pencils'])
                residual=min(interface.scalar_price(F(x['slope_exact']),F(x['intercept_exact']),A,weight) for x in c['whole_TdagT_joint_pencils'])/(d-E)
                assert b==F(cc[f'signed_B_lower_{weight}_exact']) and residual==F(cc[f'whole_positive_residual_{weight}_exact'])
                scalar=gU/(A*(A+GAMMA)); mismatch=max(wu-gL/A,F(0))/GAMMA
                lower=min(cl+b+scalar+mismatch,wu/GAMMA)
                key='every_Q' if weight=='Q' else 'allcentral_S'
                assert lower==F(cc[f'{key}_lower_exact'])==F(rr[f'{key}_lower_exact'])
                inherited=F(cc[f'joint_and_negative_scalar_{weight}_exact'])
                refined_prices=[]
                for candidate in rr['candidates']:
                    price=cover(F(candidate['slope_exact']),F(candidate['intercept_exact']),A,weight,gL,gU/A-wl)
                    assert price==F(candidate[f'joint_and_scalar_{weight}_exact'])
                    refined_prices.append(price)
                joint=min([inherited]+refined_prices)
                assert joint==F(rr[f'joint_and_negative_scalar_{weight}_exact'])
                assert cu+joint==F(rr[f'{key}_upper_exact'])
                assert max(cu+joint,lower)==F(rr[f'{key}_absolute_exact'])
                derived[weight]=dict(centered_upper=str(cu),centered_lower=str(cl),signedB=str(b),scalar=str(scalar),mismatch=str(mismatch),joint_upper=str(joint),upper=str(cu+joint),lower=str(lower),absolute=str(max(cu+joint,lower)),whole_residual=str(residual),upper_display=float(cu+joint),lower_display=float(lower),components_lower_display=list(map(float,[cl,b,scalar,mismatch])))
            reports.append(dict(axis=o['axis'],E=str(E),vacuum_interval_same_E_operator=True,weights=derived))
    assert all(x['weights']['Q']['upper_display']>1 and x['weights']['Q']['lower_display']>1 for x in reports if x['E']=='0')
    return dict(rows=reports,formation=old['formation'],historical_vacuum_formation=vacuum['formation'],domains=old['domains'],all_physical1_guards='FAILED_SUFFICIENT_GUARD_NOT_PHYSICAL_NO_GO')

def main():
    pins={name:sha(ROOT/name) for name in INPUTS}
    control: Any=finite_control()
    audit: Any=native_audit()
    scalar_cases=[]
    for weight in ('Q','S'):
        for u,v,g,m in [(F(0),F(-2),F(3),F(-1)),(F(4),F(2),F(3),F(1)),(F(2),F(-8),F(0),F(1))]:
            got=cover(u,v,F(7),weight,g,m)
            assert got==interface.scalar_price(u,v,F(7),weight,g,m)
            scalar_cases.append(dict(weight=weight,u=str(u),v=str(v),g=str(g),m=str(m),price=str(got),finite_endpoint_midpoint_and_tail_adverse_checks=True))
    assert pins=={name:sha(ROOT/name) for name in INPUTS}
    result=dict(kind='INDEPENDENT_CENTERED_VACUUM_REVIEW_EXERCISED_CONTROL',source_sha256=sha(Path(__file__)),input_sha256=pins,finite_control=control,native_audit=audit,scalar_cases=scalar_cases,no_historical_entrypoint_solver_root_checker=True,phase_or_root_admitted=False,campaign_complete=False)
    (ROOT/f'{OUT}_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(receipt=f'{OUT}_receipt.json',native_rows=len(audit['rows']),exact_control_energies=len(control['rows']),physical1='FAIL_NOT_NO_GO')))

if __name__ == '__main__':
    main()

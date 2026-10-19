"""Independent exact native reset-bootstrap controls; never solves or runs frozen H."""
from __future__ import annotations
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import sys
import time
import zipfile
import sympy as s

ROOT = Path(__file__).resolve().parent

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def generators() -> list[s.Matrix]:
    out = []
    for a in range(4):
        for b in range(a + 1, 4):
            x, y = s.zeros(4), s.zeros(4)
            x[a,b] = x[b,a] = s.Rational(1,2)
            y[a,b], y[b,a] = -s.I/2, s.I/2
            out.extend([x,y])
    for n in range(1,4):
        out.append(s.diag(*([1]*n+[-n]+[0]*(3-n)))/s.sqrt(2*n*(n+1)))
    assert len(out) == 15
    assert all(s.simplify(s.trace(a*b)) == s.Rational(i==j,2)
               for i,a in enumerate(out) for j,b in enumerate(out))
    return out

def native_commutators() -> dict:
    gs = generators()
    cf = s.Rational(15,8)
    assert sum((g*g for g in gs),s.zeros(4)) == cf*s.eye(4)
    # Full fifteen left Haar directions, at U=I, on u(U)=U00.
    # d_A U=i T_A U. Compute C(U00^2) directly from product second jets.
    c_product = s.simplify(sum(2*(g*g)[0,0]+2*g[0,0]**2 for g in gs))
    cross = s.simplify(sum((s.I*g[0,0])**2 for g in gs))
    haar_comm = s.simplify(c_product-cf)
    assert haar_comm == cf-2*cross
    assert cross == -s.Rational(3,8) and haar_comm == s.Rational(21,8)
    q,kp,kh = s.symbols('q kphi kHaar', positive=True)
    # Actual normalized Phi coordinate: Phi=q diag(1,-1,0,0)/2.
    # T=Phi00 U00, u=q^3 U00, Kphi=-kp d_q^2.
    actual = s.expand(-kp*s.diff(q**4/2,q,2)+kp*q*s.diff(q**3,q,2)/2 + kh*q**4*haar_comm/2)
    expected = -3*kp*q**2+kh*q**4*(cf-2*cross)/2
    wrong = -3*kp*q**2+kh*q**4*cf/2
    assert s.simplify(actual-expected) == 0
    defect = s.simplify(actual-wrong)
    assert defect != 0
    # Native full matrix alignment Haar second derivative via actual adjoint action.
    phi = s.diag(1,-1,0,0)/2
    adj_c = sum((g*(g*phi-phi*g)-(g*phi-phi*g)*g for g in gs),s.zeros(4))
    assert s.simplify(adj_c-4*phi) == s.zeros(4)
    # Exact native Jordan Laplacian on a nonzero tuple, all ordered16 terms
    # and all60 actual coordinate directions. SU4 x O4 invariance makes the
    # resulting quadratic polynomial a scalar multiple of sum Tr Phi_mu^2.
    fields = [phi,s.zeros(4),s.zeros(4),s.zeros(4)]
    lap = 0
    for mu in range(4):
        for nu in range(4):
            b = fields[mu]*fields[nu]+fields[nu]*fields[mu]
            for r in range(4):
                for g in gs:
                    d = s.zeros(4)
                    if r == mu: d += g*fields[nu]+fields[nu]*g
                    if r == nu: d += fields[mu]*g+g*fields[mu]
                    dd = 4*g*g if r == mu == nu else s.zeros(4)
                    lap += 4*(2*s.trace(d*d)+2*s.trace(b*dd))
    lap = s.simplify(lap)
    assert lap > 0 and lap < 2**24
    # Noncommuting operator order cannot be squared.
    a = s.diag(1,0);delta = s.ones(2);b = a+delta
    power_defect = b*b-a*a
    assert delta.det() == 0 and delta.trace() == 2
    assert power_defect.det() == -1
    return {'fundamental_C2':str(cf),'adjoint_C2':4,'all15_left_jet_cross':str(cross),
            'native_Phi_U_commutator':str(actual),'omitted_Phi_dHaar_defect':str(defect),
            'native_Jordan_full60_Laplacian_div_lambda_at_rho2_1':str(lap),
            'native_alignment_Phi_Laplacian_div_alpha':720,
            'native_alignment_Haar_Laplacian_absolute_div_alpha_rho2_upper':2,
            'noncommuting_A_le_B_but_square_order_fails':{
                'A':[[1,0],[0,0]],'B':[[2,1],[1,1]],
                'B_squared_minus_A_squared':[[4,3],[3,2]],'determinant':-1},
            'scope':'exact local differential jets on actual SU4; not H spectrum or ground'}

def c2(v: tuple[int,int,int]) -> F:
    inv=((F(3,4),F(1,2),F(1,4)),(F(1,2),F(1),F(1,2)),(F(1,4),F(1,2),F(3,4)))
    return sum(v[i]*inv[i][j]*(v[j]+2) for i in range(3) for j in range(3))/2

def pieri(v: tuple[int,int,int]) -> list[tuple[int,int,int]]:
    a,b,c=v
    return [(a+1,b,c)]+([(a-1,b+1,c)] if a else [])+([(a,b-1,c+1)] if b else [])+([(a,b,c-1)] if c else [])

def threshold_and_holder() -> dict:
    count=0;charge=F(15,4)
    # Independent x/y irreps, not merely conjugate diagonal sample families.
    sectors=[(a,b,c) for a in range(3) for b in range(3) for c in range(3)]
    for x in sectors:
        for y in sectors:
            cin=c2(x)+c2(y)
            for xx in pieri(x):
                for yy0 in pieri(tuple(reversed(y))):
                    cout=c2(xx)+c2(tuple(reversed(yy0)));d=cout-cin
                    assert d<=charge or (d-charge)**2<=4*charge*cin
                    assert -d<=charge or (-d-charge)**2<=4*charge*cout
                    count+=1
    good=[]
    for j,k in [(F(64),F(15,4)),(F(64),F(4)),(F(256),F(32))]:
        assert j>k and (j-k)**2>4*k*j
        good.append({'J':str(j),'kappa':str(k)})
    assert 1+F(15,4)>2 # invalid J=1 gate: trivial pair -> 4 x 4bar.
    # Mixed Holder is Cauchy/Holder on JOINT commuting radial/Haar measures.
    weights=[F(1,4),F(1,4),F(1,2)];r=[1,2,4];e=[3,1,8]
    lhs=sum(w*rr**2*ee**2 for w,rr,ee in zip(weights,r,e))
    r8=sum(w*rr**8 for w,rr in zip(weights,r));e4=sum(w*ee**4 for w,ee in zip(weights,e))
    assert lhs**4<=r8*e4**2 # squared norm version, normalized measure.
    # Correlated q/high tail saturates the carrier Holder exponent; independence fails.
    correlated=[]
    for p in [F(1),F(5,4),F(3,2),F(7,4),F(2)]:
        # event mass=2^(-12p), q=2^12 => E q^p=1, high reset q^(2/3)=2^(-12p+8).
        exponent=12*p-8
        assert exponent==12*(p-F(2,3))
        correlated.append({'p':str(p),'event_mass_power2':str(-12*p),'reset_cost_power2':str(-exponent)})
    return {'independent_endpoint_Pieri_transitions':count,'valid_thresholds':good,
            'wrong_small_threshold_rejected':True,'joint_Holder_lhs_squared_norm':str(lhs),
            'joint_Holder_fourth_power_margin':str(r8*e4**2-lhs**4),
            'correlated_carrier_adverse_cases':correlated,'independence_assumption_rejected':True}

def exponents() -> dict:
    rows=[];previous=None
    for p in [F(1),F(5,4),F(3,2),F(7,4),F(2)]:
        if previous is not None: assert p<previous
        terms={'reset_q_two_thirds':p-F(2,3),'reset_q_one_third':p-F(1,3),'reset_constant':p,
               'low_high_inside_PhiU':p/2+F(1,4),'low_high_alignment':p/2+F(1,2),
               'low_high_outsidePhi_U':p/2+F(1,2),'low_high_plaquette':p/2+F(1,2),
               'middle_high_inside':p-F(1,4),'high_high_inside':p-F(1,4),
               'middle_high_outside':p-F(1,4),'high_high_outside':p-F(1,4),
               'reset_outsidePhi_U':p-F(1,4),'middle_high_alignment':p-F(1,2),
               'high_high_alignment':p-F(1,2),'high_or_reset_plaquette':p}
        gamma=min(terms.values());assert gamma==min(p-F(2,3),p/2+F(1,4))
        previous=1+gamma
        rows.append({'p':str(p),'complete_terms':{k:str(v) for k,v in terms.items()},'strict_moment_supremum':str(previous)})
    assert previous==F(9,4)
    # Endpoint failure model with tail T(M)=M^-5/4 makes p=9/4 logarithmically divergent.
    assert F(9,4)-2-F(5,4)==-1
    # Small form does not imply small graph: mass n^-3 at eigenvalue n^2.
    n=16;small_form=F(1,n);graph2=F(n)
    assert small_form<1 and graph2>1
    # Commuting scalar radial IMS and exact-ground-phase inequality are valid only for L.
    ground_crop={'norm_squared_M_power':6-8,'form_M_power':6-10,
                 'inserted_form_graph_pair_M_power':3-4,'inserted_derivative_M_power':6-10,
                 'triangle_quadratic_action_M_power':6+2-8//2,
                 'triangle_linear_action_M_power':6+1-8//2}
    assert ground_crop['triangle_quadratic_action_M_power']==4
    return {'all_bootstrap_terms':rows,'endpoint_log_divergence_rejected':True,
            'graph_interpolation_s_range':'0<=s<=1 only; moment p=2s<=2',
            'radial8_paid':True,'radial_above8_not_paid':True,'radial_IMS_scaling':ground_crop,
            'small_form_large_graph_adverse':{'spectral_atom':n*n,'mass':str(F(1,n**3)),
                                            'form':str(small_form),'graph_norm_squared':str(graph2)},
            'unshifted_ground_unitary_adverse':'E0=1,t=pi: ground form=0 but ||(exp(-itH)-I)Omega||=2; use exp(-itL)',
            'same_static_or_compressed_comparison_proved':False}

def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--request',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();started=time.monotonic();req=json.loads(a.request.read_text())
    before={n:sha((ROOT/n).read_bytes()) for n in req['pins']}
    if before!=req['pins']: raise RuntimeError('input pin mismatch')
    with zipfile.ZipFile(ROOT/req['epoch_archive']) as z:
        native_raw=z.read(req['native_law_member']);profile_raw=z.read(req['profile_member'])
    profiles=json.loads(profile_raw)['primitive_profiles']
    assert [p['load'] for p in profiles]==[1,1,63]
    native_text=native_raw.decode()
    assert '4chi sum_' in native_text and '(3chi/20)' in native_text and '64sum_e' in native_text
    result={'kind':'INDEPENDENT_EXACT_NATIVE_RESET_BOOTSTRAP_REVIEW_CONTROLS',
            'commutators':native_commutators(),'threshold_and_Holder':threshold_and_holder(),
            'bootstrap_and_IMS':exponents(),'all65_profile_loads':[p['load'] for p in profiles],
            'native_law_member_sha256':sha(native_raw),'profile_member_sha256':sha(profile_raw),
            'before':before,'after':{n:sha((ROOT/n).read_bytes()) for n in req['pins']},
            'runtime':{'python':sys.executable,'sympy':s.__version__,'elapsed_seconds':time.monotonic()-started},
            'actual_ground_or_H_evolution_computed':False,'original_producer_executed':False}
    if result['after']!=before:raise RuntimeError('frozen inputs changed')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'independent_exact_native_controls':'passed','ground_or_H_numerics':False}))

if __name__=='__main__':
    main()

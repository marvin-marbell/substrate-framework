"""Independent exact higher-graph controls; never import or execute frozen producers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def radial_and_absorption() -> dict:
    t, e = sp.symbols('t e', nonnegative=True)
    a = 1+t*t
    d = 1+e*a
    w = a/d
    lap = lambda f: 120*sp.diff(f, t)+2*t*sp.diff(f, t, 2)
    gsq = lambda f: 2*t*sp.diff(f, t)**2
    assert sp.expand(lap(a)) == 244*t
    assert sp.expand(gsq(a)) == 8*t**3
    assert sp.factor(lap(w)-244*t/d**2+16*e*t**3/d**3) == 0
    assert sp.factor(gsq(w)-8*t**3/d**4) == 0
    witnesses = [a**3*d**5-t**6, a*d**3-t**2, a*d**5-e**2*t**6]
    assert all(all(c >= 0 for c in sp.Poly(p, t, e).coeffs()) for p in witnesses)
    b = w**sp.Rational(3, 2)
    assert sp.simplify(lap(b)-sp.Rational(3, 2)*sp.sqrt(w)*lap(w)-sp.Rational(3, 4)*gsq(w)/sp.sqrt(w)) == 0
    assert sp.Rational(3, 2)*260+sp.Rational(3, 4)*8 == 396
    # Positive-control native metric; plausible wrong Euclidean-radius normalization.
    assert lap(a).subs(t, 1) == 244 and lap(a).subs(t, 1) != 488
    # Young absorption is an exact square identity valid for ALL x,k,c>=0.
    y, k, c = sp.symbols('y k c', nonnegative=True)
    young_gap = y*y/sp.Integer(2)+32*k*c*c-8*sp.sqrt(k)*c*y
    assert sp.expand(young_gap-(y-8*sp.sqrt(k)*c)**2/2) == 0
    assert 32*198 == 6336 and 6336 < 80**2
    assert 2*32+2*(80+260) == 744
    # Native commutator first-order term: (2k)^2*8/k =32k.
    assert sp.cancel((2*k)**2*8/k) == 32*k
    assert sp.Rational(1, 2)**3 < sp.Rational(1, 2)**sp.Rational(3, 2)
    # These scalar examples distinguish the withdrawn w^3/XY pairing and underabsorption.
    assert sp.sqrt(sp.Rational(1, 2))**3 > sp.Rational(1, 8)
    x, kp = sp.Rational(6), sp.Rational(1, 100)
    bound = 1+8*sp.sqrt(kp)*sp.sqrt(x)+(sp.sqrt(6336)+260)*kp
    assert x <= bound and x > 2+64*kp and x < 2+744*kp
    return {'Delta_A':'244*t','grad_A_squared':'8*t^3','global_polynomial_witnesses':[str(sp.expand(p)) for p in witnesses],
            'Delta_w_bound':260,'Delta_b_bound':396,'commutator_coefficient':'32*kphi',
            'absorption_constant':'(2+744*kphi)*C1^2','wrong_metric_rejected':True,
            'wrong_XY_pairing_rejected':True,'missing_second_jet_absorption_rejected':True}


def su4_generators() -> list:
    basis = []
    for i in range(4):
        for j in range(i+1, 4):
            re, im = sp.zeros(4), sp.zeros(4)
            re[i,j] = re[j,i] = sp.Rational(1,2)
            im[i,j], im[j,i] = -sp.I/2, sp.I/2
            basis.extend((re, im))
    for n in range(1, 4):
        basis.append(sp.diag(*([1]*n+[-n]+[0]*(3-n)))/sp.sqrt(2*n*(n+1)))
    assert len(basis) == 15
    assert all(sp.simplify(sp.trace(g*h)-sp.Rational(i == j,2)) == 0
               for i,g in enumerate(basis) for j,h in enumerate(basis))
    return basis


def native_controls() -> dict:
    basis = su4_generators()
    zero = sp.zeros(4)
    p = basis[0]
    cas = sum((g*g for g in basis), zero)
    assert sp.simplify(cas-sp.Rational(15,8)*sp.eye(4)) == zero
    assert sum((g*(g*p-p*g)-(g*p-p*g)*g for g in basis), zero) == 4*p
    # Independently compute actual ordered Jordan second jets with Leibniz expansion.
    fields = [p, zero, zero, zero]
    lap_j = 0
    for mu in range(4):
        for nu in range(4):
            b = fields[mu]*fields[nu]+fields[nu]*fields[mu]
            for r in range(4):
                for g in basis:
                    dmu = g if r == mu else zero
                    dnu = g if r == nu else zero
                    b1 = dmu*fields[nu]+fields[mu]*dnu+dnu*fields[mu]+fields[nu]*dmu
                    b2 = 2*(dmu*dnu+dnu*dmu)
                    lap_j += 8*sp.trace(b1*b1+b*b2)
    lap_j = sp.simplify(lap_j)
    s = sp.trace(p*p)
    assert 0 < lap_j <= 4320*s
    # Analytic envelope accounting, not extrapolation from the tuple.
    assert 4*15*8 == 480 and 4*15 == 60 and 8*(480+60) == 4320
    assert max(4320, 720, 4) < 2**24
    # All four fields at both endpoints: actual Cartesian t-metric and ten diagonals.
    assert 120+5*120 == 720
    px, py = basis[0], basis[0]+basis[2]
    u = sp.diag(sp.I, -sp.I, 1, 1)
    gx = sp.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    gy = sp.diag(1, sp.I, -sp.I, 1)
    assert gx.det() == gy.det() == u.det() == 1
    px2, py2, u2 = gx*px*gx.H, gy*py*gy.H, gx*u*gy.H
    align = lambda x,y,v: sp.trace((x-v*y*v.H)**2)
    assert sp.simplify(align(px2,py2,u2)-align(px,py,u)) == 0
    assert sp.trace(px2*px2)+sp.trace(py2*py2) == sp.trace(px*px)+sp.trace(py*py)
    # Constant Haar-alignment shortcut is false, despite zero Haar jets of radial w.
    haar_lap_alignment = sp.simplify(8*sp.trace(px*u*py*u.H))
    assert haar_lap_alignment != 0
    # Wrong replacement of the ordered internal link by identity violates covariance.
    assert sp.simplify(align(px2,py2,sp.eye(4))-align(px,py,sp.eye(4))) != 0
    return {'fundamental_Casimir':'15/8','adjoint_Casimir':4,'Jordan_laplacian_tuple':str(lap_j),
            'global_Jordan_envelope':'4320*lambda*s','Cartesian_alignment_laplacian':720,
            'Haar_alignment_control':str(haar_lap_alignment),'actual_covariance_residual':0,
            'wrong_Haar_omission_rejected':True,'wrong_identity_link_rejected':True,
            'magnetic_and_carrier_uniformity':'Analytic ambient theorem, not finite irrep enumeration'}


def no_power_order() -> dict:
    h = sp.diag(1, 16)
    a = sp.Matrix([[sp.Rational(1,2),1],[1,8]])
    assert a.det() > 0 and (h-a).det() > 0
    wrong = h*h-a*a
    assert wrong.det() < 0
    ah = a*h.inv()
    assert sum(v*v for v in ah) < 4
    return {'positive_A_le_h':True,'first_graph_Frobenius_squared':str(sum(v*v for v in ah)),
            'h_squared_minus_A_squared_determinant':str(wrong.det()),
            'noncommuting_power_order_rejected':True,'scope':'Finite algebraic falsifier, NOT a native H surrogate'}


def archives(request: dict, captured: dict[str, bytes]) -> dict:
    report = {}
    for pre, executed in request['producer_archive_pairs']:
        with zipfile.ZipFile(ROOT/pre) as zp, zipfile.ZipFile(ROOT/executed) as ze:
            pre_members = {n for n in zp.namelist() if n.startswith('selected/')}
            executed_members = {n for n in ze.namelist() if n.startswith('selected/')}
            assert pre_members <= executed_members and pre_members
            common = pre_members
            identical = all(zp.read(name) == ze.read(name) for name in common)
            assert identical
            matched = []
            for name, raw in captured.items():
                candidates = ['selected/'+name.lstrip('/'),
                              'selected/'+str(ROOT/name).lstrip('/')]
                for member in dict.fromkeys(candidates):
                    if member in common:
                        assert zp.read(member) == raw
                        matched.append(name)
                    elif member in executed_members:
                        assert ze.read(member) == raw
            assert request['target_proof'] in matched
            assert request['target_consumer'] in matched
            report[pre] = {'common_member_count':len(common),'all_common_members_identical':identical,
                           'selected_members_bound':matched}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    captured = {name:(ROOT/name).read_bytes() for name in request['input_pins']}
    assert all(sha(raw) == request['input_pins'][name] for name,raw in captured.items())
    original_capsule = json.loads(captured[request['target_capsule']])
    original_exit = json.loads(captured[request['target_exit']])
    assert original_capsule['natural_exit'] and original_capsule['returncode'] == 0
    assert original_exit['returncode'] == 0
    result = {'kind':'INDEPENDENT_EXACT_NATIVE_HIGHER_GRAPH_REVIEW_CONTROLS',
              'radial_and_absorption':radial_and_absorption(),'native':native_controls(),
              'power_order_adverse':no_power_order(),'producer_archives':archives(request,captured),
              'producer_exit':'Archived natural0 observed, not reexecuted',
              'inputs':request['input_pins'],'proof':request['own_proof'],
              'mathematical_verdict':'VALID scoped native scalar graph; uniform ground application conditional',
              'ground_or_fullH_backend_executed':False,'original_entrypoint_executed':False,
              'scientific_admission':False,'campaign_complete':False}
    assert all((ROOT/name).read_bytes() == raw for name,raw in captured.items())
    result['selected_inputs_unchanged'] = True
    with args.output.open('x') as out:
        out.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'result':str(args.output),'selected_inputs_unchanged':True,'scientific_admission':False}))


if __name__ == '__main__':
    main()

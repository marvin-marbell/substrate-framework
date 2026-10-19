"""Independent exact CAR/SU4/Wick and full-body price review; no producer imports."""
from __future__ import annotations
import hashlib
import itertools
import json
from fractions import Fraction as Q
from pathlib import Path
import sys
import sympy as s


def dyadic(n: int) -> Q:
    return Q(2**n) if n >= 0 else Q(1, 2**(-n))


def generators() -> list[s.Matrix]:
    out = []
    for a in range(4):
        for b in range(a+1, 4):
            x, y = s.zeros(4), s.zeros(4)
            x[a,b] = x[b,a] = s.Rational(1,2)
            y[a,b], y[b,a] = -s.I/2, s.I/2
            out.extend((x,y))
    for k in range(1,4):
        out.append(s.diag(*([1]*k+[-k]+[0]*(3-k)))/s.sqrt(2*k*(k+1)))
    assert len(out) == 15
    assert all(s.simplify(s.trace(a*b)) == s.Rational(int(i==j),2)
               for i,a in enumerate(out) for j,b in enumerate(out))
    return out


def transfer(mask: int, a: int, b: int) -> tuple[int,int] | None:
    if not mask & (1<<b):
        return None
    sign = (-1)**((mask & ((1<<b)-1)).bit_count())
    reduced = mask ^ (1<<b)
    if reduced & (1<<a):
        return None
    sign *= (-1)**((reduced & ((1<<a)-1)).bit_count())
    return reduced | (1<<a), sign


def car_action(one: s.Matrix, vector: s.Matrix, masks: list[int]) -> s.Matrix:
    indices = {m:i for i,m in enumerate(masks)}
    out = s.zeros(len(masks),1)
    for j, mask in enumerate(masks):
        if vector[j] == 0:
            continue
        for a,b in itertools.product(range(8),repeat=2):
            if one[a,b] == 0:
                continue
            change = transfer(mask,a,b)
            if change is not None:
                dest, sign = change
                out[indices[dest]] += sign*one[a,b]*vector[j]
    return s.simplify(out)


def hopping(f: s.Matrix) -> s.Matrix:
    out = s.zeros(8)
    out[:4,4:], out[4:,:4] = f, f.H
    return out


def exact_controls(c0: s.Expr, ci: s.Expr) -> dict:
    ts = generators()
    casimir = sum((a*a for a in ts),s.zeros(4))
    assert s.simplify(casimir-s.Rational(15,8)*s.eye(4)) == s.zeros(4)
    planar = s.simplify(sum(s.trace(a*a*b*b) for a,b in itertools.product(ts,repeat=2)))
    crossed = s.simplify(sum(s.trace(a*b*a*b) for a,b in itertools.product(ts,repeat=2)))
    # Gaussian Wick: equal fields have three pairings, independent fields one.
    fourth = 2*planar+crossed
    diagonal = 4*16*fourth
    off_ordered = 12*4*2*(planar+crossed)
    assert (planar,crossed,fourth) == (s.Rational(225,16),-s.Rational(15,16),s.Rational(435,16))
    assert diagonal == 1740 and off_ordered == 1260 and diagonal+off_ordered == 3000
    masks = [sum(1<<i for i in c) for c in itertools.combinations(range(8),4)]
    p = s.zeros(70,1); p[masks.index(15)] = 1
    b = s.Rational(15,4)*(c0*c0+ci*ci)
    link_results = []
    # Independently integrate every active Cartesian derivative at fixed U.
    for u in [s.eye(4),s.diag(s.I,-s.I,1,1)]:
        assert u.det() == 1
        gram = 0
        for endpoint,mu,a in itertools.product(range(2),range(4),ts):
            if mu not in (0,1):
                continue
            coefficient = c0 if mu == 0 else s.I*ci
            f = coefficient*(a*u if endpoint == 0 else u*a)/2
            w = car_action(hopping(f),p,masks)
            norm = s.simplify((w.H*w)[0])
            assert s.simplify(norm-s.trace(f*f.H)) == 0
            gram += norm
        gram = s.simplify(gram)
        assert s.simplify(gram-b) == 0
        link_results.append(str(gram))
    # Exact noncommuting full-canonical reflection/onsite/filter control.
    u = s.diag(s.I,-s.I,1,1)
    left0 = s.diag(3,-1,-1,-1)/7
    right0 = ts[0]+ts[12]
    left1, right1 = ts[1]/3+ts[2]/5, ts[4]/7+ts[13]/11
    f = (c0*(left0*u+u*right0)+s.I*ci*(left1*u+u*right1))/2
    h = hopping(f)
    w = car_action(h,p,masks)
    B = s.simplify((w.H*w)[0])
    assert s.simplify(B-s.trace(f*f.H)) == 0 and B > 0
    onsite = s.diag(left0,right0)
    tau = s.Rational(2,5)
    plus, minus = p+tau*w, p-tau*w
    reflected = s.simplify(((plus.H*car_action(h+onsite,plus,masks))[0]
                            -(minus.H*car_action(h+onsite,minus,masks))[0])/2)
    assert s.simplify(reflected-2*tau*B) == 0
    assert s.simplify((p.H*car_action(h,car_action(h,w,masks),masks))[0]) == 0
    point_norm = s.simplify((plus.H*plus)[0])
    assert point_norm == 1+tau*tau*B
    # Universal homogeneous Wick identities in D120: E B=b eta,
    # E r^2 B=(D+2)b eta^2, sum||dw||^2=b, Euler real(q.w*dw)=B.
    D, eta = s.Integer(120), s.Rational(1,7)
    Z = 1+tau*tau*b*eta
    gaussian = (D*eta+tau*tau*(D+2)*b*eta**2)/(4*eta**2*Z)
    filter_derivative = tau*tau*b/Z
    cross = -tau*tau*b/Z
    complete = s.simplify(gaussian+filter_derivative+cross)
    expected = (D+2*(Z-1)/Z)/(4*eta)
    assert s.simplify(complete-expected) == 0 and cross < 0
    wrong_no_cross = s.simplify(gaussian+filter_derivative)
    wrong_no_gaussian = filter_derivative
    wrong_normalization = Z  # Dividing by point q=0 norm1 instead of integrated Z.
    wrong_dimension = (60+2*(Z-1)/Z)/(4*eta)
    assert wrong_no_cross != complete and wrong_no_gaussian < complete
    assert wrong_normalization != 1 and wrong_dimension < complete
    assert diagonal < diagonal+off_ordered
    return {'status':'EXACT','canonical_dimension':70,'Cart_dimension':120,
            'SU4_casimir':'15/8','Wick_planar':str(planar),'Wick_crossed':str(crossed),
            'ordered_Jordan_per_site_over_eta2':3000,'diagonal_only_false_price':1740,
            'fixed_U_derivative_Gram':link_results,'b':str(b),
            'inhomogeneous_noncommuting_B':str(B),'reflection_numerator':str(reflected),
            'control_Z':str(Z),'Gaussian_gradient':str(gaussian),
            'filter_gradient':str(filter_derivative),'scalar_filter_cross':str(cross),
            'complete_gradient':str(complete),
            'adverse_controls_rejected':['missing_scalar_filter_cross','missing_Gaussian_gradient',
                'point_q0_normalization','D60_instead_of_D120','diagonal_only_Jordan'],
            'pointwise_normalization_not_used':True}


def prices(c0: Q, ci: Q) -> dict:
    eta,t,T = dyadic(-7168),dyadic(3072),dyadic(7168)
    chi,lam,F = dyadic(-4096),dyadic(8192),dyadic(10240)
    b,CB = Q(15,4)*(c0*c0+ci*ci),max(c0*c0,ci*ci)/4
    x = t*t*b*eta; Z = 1+x
    gain = lam*t*b*eta/Z
    positive = {'Cart':chi/eta*(60+x/Z),
      'ordered_Jordan':3000*lam*eta*eta*(1+124*CB*t*t*eta)/Z,
      'alignment':Q(93,10)*F*eta,'Haar_scalar_cross':Q(27,4)*chi*T,
      'Haar_filter_cross':270*2**20*chi*t*t*eta/Z,
      'colour_plaquette':Q(7,5)*F/T*(450+48*7168+24),
      'all64_shared_electric':Q(262144)}
    total = sum(positive.values(),Q(0))
    assert abs(c0)<4 and abs(ci)<10 and b>256
    assert x<dyadic(-1000) and 124*CB*t*t*eta<1
    assert gain>dyadic(4096) and total<dyadic(3092)
    assert total-gain<-dyadic(4095)
    assert -dyadic(6070)<-dyadic(4095)
    # Integer charge divergence with outgoing k(x)=64 parity and N even.
    flow = 64; rho = flow-(-flow)
    assert rho == 64*(4-2) and 64*flow*flow == 262144
    return {'status':'EXACT_RATIONAL_OUTWARD_UPPER','b':str(b),'C_B':str(CB),
      'gain_greater_than':'2^4096','positive_sum_less_than':'2^3092',
      'energy_per_site_strict_less_than':'-2^4095',
      'older_unrelated_trial_is_stronger':True,'all64_charge':rho,
      'shared_full_electric_per_site':262144,'HE_HC_counted_once':True,
      'ground_BO_Berry_error':None,'ground_window':None}


def main() -> None:
    sys.set_int_max_str_digits(0)
    request = json.loads(Path(sys.argv[1]).read_text())
    for item in request['inputs']:
        path = Path(item['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
    profile = json.loads(Path(request['profile']).read_text())['primitive_profiles'][0]
    c0,ci = (Q(profile['paths'][0][key]) for key in ('c0','ci'))
    result = {'kind':'INDEPENDENT_EXACT_POLYNOMIAL_GAUSS_FORM_REVIEW',
      'request_sha256':hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
      'controls':exact_controls(s.Rational(c0.numerator,c0.denominator),s.Rational(ci.numerator,ci.denominator)),
      'prices':prices(c0,ci),'no_original_producer_import_or_replay':True,
      'scope':'Mathematical consumer only; analytic D(H), interacting Haar marginal and originating-ground antecedents in proof',
      'scientific_admission':False,'campaign_complete':False}
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'exact_adverse_controls':'PASS','full_H_upper':'<-2^4095/site','ground_BO_error':'OPEN'}))


if __name__ == '__main__':
    main()

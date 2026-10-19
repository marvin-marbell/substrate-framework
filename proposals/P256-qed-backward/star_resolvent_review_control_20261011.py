"""Independent exact full-CAR/integer-flow and pinned receipt resolvent review.
No historical scientist, solver, root, or checker entrypoint is imported or run.
"""
from __future__ import annotations
import hashlib
import json
import math
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = 'star_resolvent_review_'
OMEGA = (21, 0, 0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sqrt_hi(x):
    if x < 0:
        raise ValueError('Negative squared norm')
    scale = 1 << 100
    k = math.isqrt(x.numerator * scale * scale // x.denominator)
    return F(k + int(F(k*k, scale*scale) < x), scale)


def addv(*vectors):
    out = {}
    for v in vectors:
        for state, value in v.items():
            out[state] = out.get(state, F(0)) + value
    return {s: v for s, v in out.items() if v}


def scalev(v, c):
    return {s: c*x for s, x in v.items() if c*x}


def dot(v, w):
    return sum((x*w.get(s, F(0)) for s, x in v.items()), F(0))


def car(mask, mode, creation):
    occupied = bool(mask & (1 << mode))
    if occupied == creation:
        return None
    sign = -1 if (mask & ((1 << mode)-1)).bit_count() % 2 else 1
    return mask ^ (1 << mode), sign


def energy(state):
    mask, a, b = state
    onsite = sum(gap*((mask >> (2*c+1)) & 1) for c, gap in enumerate((2, 3, 5)))
    charge = sum(7*(((mask >> (2*c)) & 1)+((mask >> (2*c+1)) & 1)-1)**2 for c in range(3))
    return F(onsite + charge + 11*(a*a+b*b))


def edge(v, which):
    outer = 0 if which == 0 else 2
    matrix = ((F(1,3), F(2,5)), (F(3,7), F(1,2)))
    out = {}
    for (mask, a, b), value in v.items():
        for i in range(2):
            for j in range(2):
                for orient in (-1, 1):
                    src, dest = (2*outer+j, 2+i) if orient == 1 else (2+i, 2*outer+j)
                    first = car(mask, src, False)
                    if first is None:
                        continue
                    second = car(first[0], dest, True)
                    if second is None:
                        continue
                    state = (second[0], a+orient if which == 0 else a,
                             b+orient if which == 1 else b)
                    out[state] = out.get(state, F(0)) + value*first[1]*second[1]*matrix[i][j]
    return {s: v for s, v in out.items() if v}


def project(v):
    return {s: x for s, x in v.items() if s != OMEGA}


def inverse(v, wrong_ground=False):
    return {s: x/(energy(s) if s != OMEGA else F(1))
            for s, x in v.items() if wrong_ground or s != OMEGA}


def response(v):
    return dot(v, inverse(v))


def full_car_control():
    # All 64 Fock basis vectors, not a harmonic or source-image crop.
    for mask in range(64):
        for i in range(6):
            for j in range(6):
                out = {}
                for first, second in ((car(mask, j, True), (i, False)),
                                      (car(mask, i, False), (j, True))):
                    if first is not None:
                        end = car(first[0], *second)
                        if end is not None:
                            out[end[0]] = out.get(end[0], 0)+first[1]*end[1]
                out = {s: x for s, x in out.items() if x}
                assert out == ({mask: 1} if i == j else {})
    assert min(energy((mask,0,0)) for mask in range(64) if mask != 21) == 2
    # Integer-flow space is Z^2: dictionaries contain exact finite word support,
    # with no boundary/periodic wrap and no truncation of an intermediate state.
    g = {OMEGA: F(1)}
    sa, sb = edge(g,0), edge(g,1)
    assert dot(sa,sb) == 0
    xa, xb = inverse(sa), inverse(sb)
    ua, ub = edge(xa,0), edge(xb,1)
    ya, yb = project(ua), project(ub)
    ym = project(addv(edge(xb,0),edge(xa,1)))
    cross = 2*dot(ya,inverse(yb))
    middle = response(ym)
    aa, ab = dot(sa,xa), dot(sb,xb)
    ba, bb = dot(xa,xa), dot(xb,xb)
    fold = aa*bb+ab*ba
    coefficient = cross+middle-fold
    # Independent multivariate intermediate-normalized eigenvector recursion.
    psi = {(0,0):g}
    energies = {(0,0):F(0)}
    for total in range(1,5):
        for i in range(3):
            j = total-i
            if not 0 <= j <= 2:
                continue
            forcing = addv(edge(psi.get((i-1,j),{}),0),edge(psi.get((i,j-1),{}),1))
            energies[i,j] = forcing.get(OMEGA,F(0))
            terms = [forcing]
            for (a,b), e in energies.items():
                if a+b and (a,b) != (i,j) and a <= i and b <= j:
                    terms.append(scalev(psi.get((i-a,j-b),{}),-e))
            psi[i,j] = scalev(inverse(addv(*terms)),-1)
    assert coefficient == -energies[2,2]
    assert energies[1,1] == 0
    plus, minus = addv(ya,yb), addv(ya,scalev(yb,-1))
    assert cross == (response(plus)-response(minus))/2
    wrong_q = 2*dot(ua,inverse(ub,True))+dot(ym,inverse(ym,True))-fold
    assert wrong_q != coefficient and fold and middle and cross
    assert coefficient != cross+middle  # Missing ground-normalization fold.
    assert coefficient != cross-fold  # Missing four middle words.
    # Source projector is genuinely different from the whole ground projector.
    excited = (22,0,0)
    source = {OMEGA:F(3,5), excited:F(4,5)}
    projected_wrong = addv(ua,scalev(source,-dot(source,ua)))
    assert projected_wrong != ya
    return {'kind':'EXACT_ENTIRE_64_CAR_TENSOR_INTEGER_FLOW_TOY_NOT_NATIVE_G',
            'full_fock_dimension':64,'unbounded_raw_flow':'Z^2, exact finite fourth-order word support',
            'gap_exact':'2','z11_exact':'0','coefficient_exact':str(coefficient),
            'independent_recursion_minus_E22_exact':str(-energies[2,2]),
            'neutral_cross_exact':str(cross),'middle_exact':str(middle),'fold_exact':str(fold),
            'missing_Q_wrong_exact':str(wrong_q),'missing_fold_wrong_exact':str(cross+middle),
            'missing_middle_wrong_exact':str(cross-fold),'source_projector_rejected':True,
            'all_64_basis_CAR_anticommutators_checked':True,
            'six_ordered_word_convention':'ordinary coefficient; derivative is four times coefficient'}


def addi(a,b):
    return a[0]+b[0],a[1]+b[1]


def scalei(a,c):
    return (a[0]*c,a[1]*c) if c >= 0 else (a[1]*c,a[0]*c)


def muli(a,b):
    vals = [x*y for x in a for y in b]
    return min(vals),max(vals)


def decode(obj):
    return tuple(F(x) for x in obj['exact'])


def polynomial_enclosure(mi,c,gap):
    c0,c1 = c
    coeff = (F(1),-2*c0,c0*c0-2*c1,2*c0*c1,c1*c1)
    residual = (F(0),F(0))
    for interval,weight in zip(mi,coeff):
        residual = addi(residual,scalei(interval,weight))
    assert residual[1] >= 0 and mi[0][1] >= 0 and gap > 0
    err = sqrt_hi(mi[0][1]*residual[1])/gap
    response_interval = addi(scalei(mi[0],c0),scalei(mi[1],c1))
    raw = addi(response_interval,(-err,err))
    result = max(F(0),raw[0]),min(mi[0][1]/gap,raw[1])
    assert result[0] <= result[1]
    return result,residual,err


def spectral_countercontrol():
    spectrum = (F(2),F(5),F(11))
    weights = (F(1),F(4),F(9))
    m = [sum(w*k**j for w,k in zip(weights,spectrum)) for j in range(5)]
    determinant = m[1]*m[3]-m[2]**2
    c = ((m[0]*m[3]-m[1]*m[2])/determinant,
         (m[1]**2-m[0]*m[2])/determinant)
    exact = sum(w/k for w,k in zip(weights,spectrum))
    ritz = c[0]*m[0]+c[1]*m[1]
    enclosure,residual,error = polynomial_enclosure([(x,x) for x in m],c,F(2))
    direct_residual = sum(w*(1-k*(c[0]+c[1]*k))**2 for w,k in zip(weights,spectrum))
    assert residual == (direct_residual,direct_residual)
    assert enclosure[0] <= exact <= enclosure[1] and ritz != exact and direct_residual > 0
    # The m0 interval depends on P0; dependency-free interval subtraction is
    # conservative even when an amplitude interval straddles zero.
    amplitude = (F(-1,3),F(1,2))
    squared = muli(amplitude,amplitude)
    assert squared[0] <= 0 <= squared[1]
    return {'spectrum_exact':[str(x) for x in spectrum], 'weights_exact':[str(x) for x in weights],
            'm0_to_m4_exact':[str(x) for x in m],'candidate_exact':[str(x) for x in c],
            'exact_resolvent':str(exact),'Ritz_not_certificate':str(ritz),
            'residual_square_exact':str(direct_residual),'outward_enclosure_exact':[str(x) for x in enclosure],
            'residual_radius_exact':str(error),'rejects_Ritz_equals_certificate':True,
            'correlated_moment_intervals_not_assumed_independent':True}


def native_receipt_consumer(receipt):
    gap = F(receipt['reference_gap_exact'])
    floor = F(receipt['single_flow_floor_exact'])
    assert floor == 64+2*F(150034528033,34359738368)
    sources = receipt['sources']
    enclosures = {}
    norms = {}
    for name,source in sources.items():
        mi = [decode(x) for x in source['moment_intervals']]
        c = tuple(F(x) for x in source['resolvent']['polynomial_exact'])
        enc,res,err = polynomial_enclosure(mi,c,gap)
        assert enc == decode(source['resolvent']['response'])
        assert res == decode(source['resolvent']['residual_square'])
        assert err == F(source['resolvent']['response_error_exact'])
        enclosures[name] = enc
        norms[name] = sqrt_hi(mi[0][1])
    roots = receipt['single_roots']
    eps = []
    for root in roots:
        g,t,u = (decode(root[key]) for key in ('G','T','U'))
        alpha = F(root['alpha_exact'])
        assert alpha > 0
        variance = addi(addi(u,scalei(t,-2*alpha)),scalei(g,alpha*alpha))
        epsilon = sqrt_hi(max(F(0),variance[1]))/(alpha*floor)
        assert epsilon == F(root['whole_charged_outer_resolvent_error_exact'])
        eps.append(epsilon)
        aerr = sqrt_hi(g[1])*epsilon
        berr = 2*sqrt_hi(g[1])/alpha*epsilon+epsilon**2
        a = addi(scalei(g,1/alpha),(-aerr,aerr))
        b = addi(scalei(g,1/alpha**2),(-berr,berr))
        assert (max(F(0),a[0]),a[1]) == decode(root['A'])
        assert (max(F(0),b[0]),b[1]) == decode(root['B'])
    ka,kb = (F(x) for x in receipt['normalized_rooted_physical1_attempt']['edge_operator_norm_upper_exact'])
    da,db = ka*eps[0],kb*eps[1]
    dm = ka*eps[1]+kb*eps[0]
    na = nb = (norms['neutral_plus']+norms['neutral_minus'])/2
    radius = (2*(da*nb+db*na+da*db)+dm*(2*norms['mixed']+dm))/gap
    assert radius == F(receipt['outer_approximation_price_exact'])
    chain0 = addi(scalei(addi(enclosures['neutral_plus'],scalei(enclosures['neutral_minus'],-1)),F(1,2)),enclosures['mixed'])
    chain = addi(chain0,(-radius,radius))
    fold = addi(muli(decode(roots[0]['A']),decode(roots[1]['B'])),
                muli(decode(roots[1]['A']),decode(roots[0]['B'])))
    connected = addi(chain,scalei(fold,-1))
    assert chain == decode(receipt['chain']) and fold == decode(receipt['normalization_fold'])
    assert connected == decode(receipt['connected_per_time_coefficient'])
    assert connected[0] < 0 < connected[1]
    assert F(receipt['meaningful_resolvent_time_scale_exact']) == 1/gap
    return {'all_three_exact_residual_enclosures_rederived':True,
            'two_single_outer_epsilon_and_A_B_intervals_rederived':True,
            'chain_fold_connected_exact_endpoints_rederived':True,
            'chain_radius_includes_da_db_and_dm_squared':True,
            'connected_display':receipt['connected_per_time_coefficient']['display'],
            'Ritz_connected_float_only':receipt['Ritz_connected_float'],
            'physical1_guard_exact':receipt['normalized_rooted_physical1_attempt']['gap_guard_2sumV_over_gap_exact'],
            'conditional_floor_value_exact':str(floor),
            'floor_module_membership_not_proved_by_receipt_arithmetic':True,
            'gamma_inverse_is_semigroup_bound_not_measured_finite_clock':True}


def main():
    request_path = ROOT/(PREFIX+'request_20261011.json')
    request = json.loads(request_path.read_bytes())
    for name,pin in request['input_sha256'].items():
        assert sha(ROOT/name) == pin, name
    receipt = json.loads((ROOT/'shared_star_dynamic_actual2_20261011_receipt.json').read_bytes())
    result = {'kind':'NEW_INDEPENDENT_RESOLVENT_PROOF_REVIEW_CONSUMER',
              'governing_plan':request['governing_plan'],
              'native_arithmetic_review':native_receipt_consumer(receipt),
              'independent_full_CAR_tensor_countercontrol':full_car_control(),
              'independent_full_spectrum_countercontrol':spectral_countercontrol(),
              'input_sha256':request['input_sha256'],
              'request_sha256':sha(request_path),'own_source_sha256':sha(Path(__file__)),
              'no_original_scientist_or_helper_execution':True,'natural_main_finished':True}
    for name,pin in request['input_sha256'].items():
        assert sha(ROOT/name) == pin, name
    out = ROOT/(PREFIX+'receipt_20261011.json')
    with out.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps({'receipt':out.name,'independent_exact_countercontrols':'passed',
                      'native_connected_interval':result['native_arithmetic_review']['connected_display']}))


if __name__ == '__main__':
    main()

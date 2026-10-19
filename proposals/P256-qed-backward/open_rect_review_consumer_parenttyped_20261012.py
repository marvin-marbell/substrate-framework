"""Independent exact native rectangular AP/body consumer, not a ground solver."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from math import isqrt
from pathlib import Path
from typing import cast
import sympy as s


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(a: s.Matrix) -> s.Matrix:
    return a.applyfunc(s.simplify)


def hs2(a: s.Matrix) -> s.Expr:
    return s.expand(s.trace(a.adjoint()*a))


def shift(n: int, closing: int) -> s.Matrix:
    a = s.zeros(n)
    for j in range(n-1):
        a[j,j+1] += 1
    a[n-1,0] += closing
    return a


def root_upper(q: Fraction, bits: int = 192) -> Fraction:
    scale = 1 << bits
    return Fraction(isqrt(q.numerator*scale*scale//q.denominator)+1, scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for path, pin in request['pins'].items():
        if digest(Path(path)) != pin:
            raise ValueError(f'Pin changed: {path}')
    profiles = json.loads(Path(request['profiles']).read_text())['primitive_profiles']
    neutral = profiles[0]
    m = s.Rational(neutral['onsite'])
    c0 = s.Rational(neutral['paths'][0]['c0'])
    ci = s.Rational(neutral['paths'][0]['ci'])
    assert [(p['axis'],p['length']) for p in neutral['paths']] == [(0,1),(1,1),(2,1)]
    assert all(s.Rational(p['c0']) == c0 and s.Rational(p['ci']) == ci for p in neutral['paths'])
    z2, eye = s.zeros(2), s.eye(2)
    sx, sy, sz = s.Matrix([[0,1],[1,0]]), s.Matrix([[0,-s.I],[s.I,0]]), s.diag(1,-1)
    body = [s.diag(3,1,-1,-3)/8,
            s.kronecker_product(sy,eye)/5+s.kronecker_product(sz,eye)/7,
            s.kronecker_product(sx,sz)/6+s.kronecker_product(sy,sx)/11,
            s.kronecker_product(sz,sx)/9+s.kronecker_product(sx,eye)/10]
    zero = s.zeros(4)
    assert all(b == b.adjoint() and s.trace(b) == 0 for b in body)
    assert body[0]*body[1] != body[1]*body[0]
    assert s.trace(body[0]*body[1]) != 0
    assert any(s.im(v) != 0 for b in body for v in b)
    x = [s.trace(b*b)/4 for b in body]
    jordan = s.expand(4*sum(s.trace((a*b+b*a)**2) for a in body for b in body))
    delta = s.expand(16*sum(s.trace(b**4)-s.trace(b*b)**2/4 for b in body)
                     +4*sum(s.trace((body[u]*body[v]+body[v]*body[u])**2)
                            for u in range(4) for v in range(4) if u != v))
    assert jordan == 64*sum(v*v for v in x)+delta and delta > 0
    symbols = s.symbols('M d1 d2 d3', real=True)
    reflected = 0
    for signs in itertools.product((-1,1), repeat=3):
        h = symbols[0]*body[0]+sum((sign*d*b for sign,d,b in zip(signs,symbols[1:],body[1:])),zero)
        reflected += s.trace(h*h)/8
    assert s.expand(reflected-4*sum(d*d*v for d,v in zip(symbols,x))) == 0
    cliff = [s.kronecker_product(sz,eye),s.kronecker_product(sx,sx),s.kronecker_product(sx,sy),s.kronecker_product(sx,sz)]
    amplitudes = [s.Rational(2,7),s.Rational(3,8),s.Rational(4,9),s.Rational(5,11)]
    C = [a*g for a,g in zip(amplitudes,cliff)]
    hc = sum((d*b for d,b in zip(symbols,C)),zero)
    assert clean(hc*hc-sum((d*a)**2 for d,a in zip(symbols,amplitudes))*s.eye(4)) == zero
    Jc = s.expand(4*sum(s.trace((a*b+b*a)**2) for a in C for b in C))
    assert Jc == 64*sum(a**4 for a in amplitudes)

    # Full native 2x3x2 rectangle; both odd self-reflection and additive
    # length-two aliases are present in the SAME assembled matrix.
    lengths = (2,3,2)
    V = 12
    O, S = [], []
    for axis,n in enumerate(lengths):
        O.append(s.kronecker_product(*(shift(k,0) if i == axis else s.eye(k) for i,k in enumerate(lengths))))
        S.append(s.kronecker_product(*(shift(k,-1) if i == axis else s.eye(k) for i,k in enumerate(lengths))))
    Hopen = s.kronecker_product(s.eye(V),m*body[0])
    Hap = Hopen.copy()
    transfers = []
    closing = []
    for axis,n in enumerate(lengths):
        T = c0*body[0]+s.I*ci*body[axis+1]
        transfers.append(T)
        assert hs2(T) == c0*c0*s.trace(body[0]**2)+ci*ci*s.trace(body[axis+1]**2)
        edge = s.zeros(8)
        edge[:4,4:] = -T
        edge[4:,:4] = -T.adjoint()
        assert T.det() != 0 and edge.det() != 0
        assert clean(edge*edge-s.diag(T*T.adjoint(),T.adjoint()*T)) == s.zeros(8)
        assert hs2(edge) == 2*hs2(T) and s.trace(edge) == 0
        Hopen += s.kronecker_product(O[axis],T)+s.kronecker_product(O[axis].T,T.adjoint())
        Hap += s.kronecker_product(S[axis],T)+s.kronecker_product(S[axis].T,T.adjoint())
        closing.append({'axis':axis,'length':n,'edges':V//n,'rank_each':8,'HS2_each':str(hs2(edge))})
    assert Hopen == Hopen.adjoint() and Hap == Hap.adjoint()
    assert s.trace(Hopen) == s.trace(Hap) == 0
    roots = [[s.I,-s.I],[(1+s.I*s.sqrt(3))/2,-1,(1-s.I*s.sqrt(3))/2],[s.I,-s.I]]
    mode_count = 0
    for zs in itertools.product(*roots):
        v = s.kronecker_product(*(s.Matrix([z**j for j in range(n)]) for z,n in zip(zs,lengths)))
        transfer = m*body[0]
        for axis,z in enumerate(zs):
            assert clean(S[axis]*v-z*v) == s.zeros(V,1)
            transfer += z*transfers[axis]+s.conjugate(z)*transfers[axis].adjoint()
        embedding = s.kronecker_product(v,s.eye(4))
        assert clean(Hap*embedding-embedding*transfer) == s.zeros(4*V,4)
        mode_count += 1
    assert mode_count == V
    # Odd self-mode is native z=-1, cos=-1, sin=0. Deleting it changes
    # the complete onsite trace-square mean, even if the rest is normalized.
    Mlow, Mself = m+c0, m-2*c0
    full_M2 = (2*Mlow**2+Mself**2)/3
    wrong_skipped_M2 = Mlow**2
    assert full_M2 == m*m+2*c0*c0
    assert full_M2 != wrong_skipped_M2
    assert shift(3,-1)*s.Matrix([1,-1,1]) == -s.Matrix([1,-1,1])
    T = transfers[0]
    added = T-T.adjoint()
    two_axis_block = cast(s.Matrix, s.kronecker_product(shift(2,-1),T)+s.kronecker_product(shift(2,-1).T,T.adjoint()))
    assert two_axis_block[:4,4:] == added
    wrong_overwrite = -T.adjoint()
    assert added != wrong_overwrite
    # Native trace-norm price witness with a full-rank diagonal T.
    Td = c0*s.diag(1,1,-1,-1)/4
    Ed = s.zeros(8)
    Ed[:4,4:] = -Td
    Ed[4:,:4] = -Td.adjoint()
    assert Ed.det() != 0 and Ed*Ed == (c0/4)**2*s.eye(8)
    exact_edge_trace_norm = 2*abs(c0)
    assert exact_edge_trace_norm**2 == 8*hs2(Ed)
    assert exact_edge_trace_norm**2 > hs2(Ed)
    exact_positive_part_price = abs(c0)

    # Complete spectra from actual native Fourier-diagonal diagonal bodies;
    # no eigensolver. Two full rectangles, different amplitudes, total2V.
    # Original N6 AP loop is never gauged into identity. A contractible
    # three-site box crossing that seam can remove its internal signs only.
    original = shift(6,-1)
    assert original**6 == -s.eye(6)
    crossed_open = s.zeros(3)
    crossed_open[0,1],crossed_open[1,2] = -1,1
    local_signs = s.diag(1,-1,-1)
    assert local_signs*crossed_open*local_signs == shift(3,0)
    assert shift(3,-1)-shift(3,0) != s.zeros(3)
    assert shift(3,-1)**3 == -s.eye(3)
    masses = [Mlow]*8+[Mself]*4
    spectra = [[a*d*M for M in masses for d in [s.Rational(3,4),s.Rational(-1,4),s.Rational(-1,4),s.Rational(-1,4)]] for a in [1,s.Rational(1,3)]]
    tagged = sorted([(v,j) for j,levels in enumerate(spectra) for v in levels],reverse=True)
    selected = tagged[:48]
    global48 = sum(v for v,j in selected)
    box_half = sum(sum(sorted(levels,reverse=True)[:24]) for levels in spectra)
    positive = sum(v for v,j in tagged if v > 0)
    occup = [sum(j == i for v,j in selected) for i in range(2)]
    negative = sum(1 for v,j in selected if v < 0)
    assert global48 > box_half and global48 < positive and occup != [24,24] and negative == 24
    mu = tagged[47][0]
    dual = 48*mu+sum(max(0,v-mu) for v,j in tagged)
    assert dual == global48
    # Noncommuting actual contraction preserves global rank.
    H = s.diag(*(v for v,j in tagged))
    P = s.diag(*([1]*48+[0]*48))
    U = s.eye(96)
    U[0,0],U[0,95],U[95,0],U[95,95] = s.Rational(3,5),s.Rational(-4,5),s.Rational(4,5),s.Rational(3,5)
    Gamma = U*P*U.T
    actual = s.trace(Gamma*H)
    assert Gamma*Gamma == Gamma and s.trace(Gamma) == 48 and Gamma*H != H*Gamma
    assert actual <= global48 <= positive

    # Independent rational constants, outward square-root arithmetic.
    mf,c0f,cif = map(lambda q:Fraction(str(q)),[m,c0,ci])
    b = abs(c0f)+abs(cif)
    D = 2*abs(mf)+12*b
    means = [mf*mf+6*c0f*c0f]+[2*cif*cif]*3
    assert min(means) > 128 and max(means) < 1024
    assert mf-6*abs(c0f) > 0
    assert 2*((abs(mf)+6*abs(c0f))**2+12*cif*cif) < 128**2
    R,L = 2**256,2**64
    field = (1024*L**3+D)*root_upper(Fraction(1920))*Fraction(R**2,2**1017)
    colour = 4*b*L*root_upper(Fraction(17280*R,7))*Fraction(R**2,2**1017)
    errors = {'field':field,'curvature':colour,'cut':12*b*L/R,'sewing':6*b*L/R,'BZ':Fraction(66,7)*b*L/R,'tail':D*2**14/L**3}
    total = sum(errors.values(),Fraction())
    assert total < Fraction(1,2**160)
    old_cut = 12*b*2**32/2**12
    assert old_cut > 21
    dch = sum(p['load']*(2*abs(Fraction(p['onsite']))+4*sum(abs(Fraction(t['c0']))+abs(Fraction(t['ci'])) for t in p['paths'])) for p in profiles[1:])
    assert dch == Fraction(855444290449274477,35184372088832)
    receipt = {'kind':'INDEPENDENT_EXACT_NATIVE_COMPLEX_RECTANGLE_AND_GLOBAL_CANONICAL_CONSUMER_NOT_PHYSICAL_GROUND',
               'source_sha256':digest(Path(__file__)),'request_sha256':digest(args.request),'pins':request['pins'],
               'complex_allmatrix':{'quarter_x':list(map(str,x)),'Jordan':str(jordan),'defect':str(delta),'reflection_cross_cancellation':True,'Clifford_square_and_Jordan_saturation':True},
               'rectangle':{'lengths':lengths,'volume':V,'complete_exact_Fourier_modes':mode_count,'closing':closing,'trace_open':str(s.trace(Hopen)),'trace_AP':str(s.trace(Hap))},
               'adverse':{'rank2_wrong_actual_rank':8,'Frobenius_wrong_trace_norm_squared_ratio':8,'trace_norm_edge_exact':str(exact_edge_trace_norm),'positive_part_edge_price_exact':str(exact_positive_part_price),'length2_overwrite_rejected':True,'odd_self_mode_skipping_rejected':True,'full_mass_square_mean':str(full_M2),'wrong_skipped_mass_square_mean':str(wrong_skipped_M2)},
               'global_canonical':{'sites':24,'particles':48,'occupations':occup,'selected_negative_levels':negative,'actual_top48':str(global48),'wrong_half_fill_each_box':str(box_half),'positive_part_upper_not_identity':str(positive),'one_mu_exact_dual':str(dual),'noncommuting_density_expectation':str(actual)},
               'constants':{'native_mean_weights':list(map(str,means)),'strong_convexity':128,'minimum_defect_coefficient':64,'Dch':str(dch),'old_R4096_cut':str(old_cut),'new_errors_outward':{k:str(v) for k,v in errors.items()},'new_total_outward':str(total),'new_total_lt_2minus160':True},
               'scope':'Finite exact adverse examples support independently written analytic proof. Physical normalized Gauss/sewn-cover family and quantum upper remain explicit antecedents; no numerical ground, original solver, field/frame/phase/IR promotion.'}
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print('Independent exact native complex 2x3x2 AP, trace rank8, odd mode, aliases and global48 canonical consumer completed')


if __name__ == '__main__':
    main()

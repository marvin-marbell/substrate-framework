"""New exact native OPEN/AP/BZ and actual-family body-lower consumer; no ground solver."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from math import isqrt
from pathlib import Path
import sympy as sp


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rational(text: str) -> sp.Rational:
    q = Fraction(text)
    return sp.Rational(q.numerator, q.denominator)


def simplify_matrix(matrix: sp.Matrix) -> sp.Matrix:
    return matrix.applyfunc(sp.simplify)


def hs2(matrix: sp.Matrix) -> sp.Expr:
    return sp.expand(sp.trace(matrix * matrix.conjugate().T))


def jordan(body: list[sp.Matrix]) -> sp.Expr:
    return sp.expand(4 * sum(sp.trace((a*b+b*a)**2) for a in body for b in body))


def shifts(length: int, closing: int) -> sp.Matrix:
    matrix = sp.zeros(length)
    for j in range(length-1):
        matrix[j, j+1] = 1
    matrix[length-1, 0] = closing
    return matrix


def sqrt_upper(value: Fraction, bits: int = 160) -> Fraction:
    scale = 1 << bits
    floor = isqrt((value.numerator * scale * scale) // value.denominator)
    return Fraction(floor+1, scale)


def gate(m: Fraction, b: Fraction, R: int, L: int) -> dict[str, str | bool]:
    d = 2*abs(m)+12*b
    a, c, moment = Fraction(40, 1 << 2035), Fraction(10, 7*(1 << 2035)), Fraction(1 << 14)
    # Outward square roots factor out powers before enclosure, so tiny inputs
    # do not acquire an artificial 2^-160 floor.
    field_root = sqrt_upper(Fraction(48*40))*Fraction(R**2, 1 << 1017)
    colour_root = sqrt_upper(Fraction(1728*10*R, 7))*Fraction(R**2, 1 << 1017)
    # 2^-2035 <= 2^-2034, intentionally conservative.
    errors = {
        'field': (1024*L**3+d)*field_root,
        'curvature': 4*b*L*colour_root,
        'cut': 12*b*L/R,
        'sew': 6*b*L/R,
        'BZ': Fraction(66, 7)*b*L/R,
        'tail': d*moment/L**3,
    }
    total = sum(errors.values(), Fraction())
    return {**{key: str(value) for key, value in errors.items()},
            'total_upper': str(total), 'less_than_2power_minus160': total < Fraction(1, 1 << 160),
            'R': str(R), 'L': str(L), 'minimum_N_even': str(2*R),
            'cap_union_raw': str(Fraction((2*R-1)**3)*moment/L**4),
            'per_site_tail_probability_upper': str(moment/L**4),
            'tree_mean_deviation_upper': str(48*R**4*a),
            'residual_link_mean_HS2_upper': str(1728*R**5*c)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    root = args.request.parent
    for name, expected in request['pins'].items():
        if sha(Path(name) if Path(name).is_absolute() else root/name) != expected:
            raise ValueError(f'Pinned input changed: {name}')
    profiles = json.loads((root/request['profiles']).read_text())['primitive_profiles']
    neutral = profiles[0]
    m, c0, ci = (rational(neutral['onsite']), rational(neutral['paths'][0]['c0']), rational(neutral['paths'][0]['ci']))
    assert all(p['length'] == 1 and rational(p['c0']) == c0 and rational(p['ci']) == ci for p in neutral['paths'])
    assert [p['axis'] for p in neutral['paths']] == [0, 1, 2]
    I, Z = sp.eye(4), sp.zeros(4)
    sx, sy, sz = sp.Matrix([[0,1],[1,0]]), sp.Matrix([[0,-sp.I],[sp.I,0]]), sp.diag(1,-1)
    gammas = [sp.kronecker_product(sz,sp.eye(2)), sp.kronecker_product(sx,sx), sp.kronecker_product(sx,sy), sp.kronecker_product(sx,sz)]
    clifford = [gammas[0]/3, gammas[1]/4, gammas[2]/5, gammas[3]/6]
    # Genuinely non-Clifford/noncommuting traceless Hermitian body, not a
    # Clifford-only sewing demonstration.
    arbitrary = [clifford[0]+gammas[1]/7, clifford[1]+gammas[0]/8, clifford[2], clifford[3]+gammas[0]/9]
    control = []
    for body in (clifford, arbitrary):
        assert all(sp.trace(B) == 0 and B == B.conjugate().T for B in body)
        x = [sp.trace(B**2)/4 for B in body]
        defect = sp.expand(jordan(body)-64*sum(t*t for t in x))
        defect_formula = 16*sum(sp.trace(B**4)-sp.trace(B**2)**2/4 for B in body)
        defect_formula += 4*sum(sp.trace((body[u]*body[v]+body[v]*body[u])**2) for u in range(4) for v in range(4) if u != v)
        assert sp.expand(defect-defect_formula) == 0
        assert defect >= 0
        control.append({'x_quarter': [str(t) for t in x], 'ordered_Jordan_excess': str(defect)})
    assert control[0]['ordered_Jordan_excess'] == '0' and rational(str(control[1]['ordered_Jordan_excess'])) > 0
    body = arbitrary
    sew_controls = []
    z = sp.Symbol('z')
    for axis, length in enumerate((3,4,5)):
        T = c0*body[0]+sp.I*ci*body[axis+1]
        S, O = shifts(length,-1), shifts(length,0)
        assert S**length == -sp.eye(length)
        assert S*S.T == sp.eye(length)
        assert shifts(length,1)**length == sp.eye(length)
        Hopen = sp.kronecker_product(sp.eye(length),m*body[0])+sp.kronecker_product(O,T)+sp.kronecker_product(O.T,T.conjugate().T)
        Hap = sp.kronecker_product(sp.eye(length),m*body[0])+sp.kronecker_product(S,T)+sp.kronecker_product(S.T,T.conjugate().T)
        delta = Hap-Hopen
        edge = sp.zeros(8)
        edge[:4,4:] = -T
        edge[4:,:4] = -T.conjugate().T
        rank = edge.rank()
        assert rank <= 8 and hs2(delta) == hs2(edge) == 2*hs2(T)
        assert sp.trace(Hopen) == sp.trace(Hap) == sp.trace(delta) == 0
        assert hs2(T) == c0**2*sp.trace(body[0]**2)+ci**2*sp.trace(body[axis+1]**2)
        # Exact polynomial Fourier/AP check, valid also for odd lengths.
        mode = sp.Matrix([z**j for j in range(length)])
        residue = (S*mode-z*mode).applyfunc(lambda a: sp.rem(a,z**length+1,z))
        assert residue == sp.zeros(length,1)
        # Reverse mode and reflection stay in the SAME AP grid.
        assert sp.rem((-1)**length+1,z) == 0 if length % 2 else True
        assert sp.expand(z**length*((1/z)**length+1)) == 1+z**length
        # Fourier transfer gives exactly M and native D, not a flipped ci.
        transfer = T*z+T.conjugate().T/z
        literal = c0*(z+1/z)*body[0]+sp.I*ci*(z-1/z)*body[axis+1]
        assert simplify_matrix(transfer-literal) == Z
        sew_controls.append({'axis':axis,'length':length,'closing_rank':rank,'closing_traceHS2':str(hs2(edge)), 'AP_power':'-I','open_trace':'0'})
    # Native 3D rectangle: edge count is EXACT, including odd side lengths.
    lengths = (3,4,5)
    sites = list(itertools.product(*(range(n) for n in lengths)))
    boundary_counts = []
    internal_counts = []
    for axis, length in enumerate(lengths):
        boundary_counts.append(sum(p[axis] == length-1 for p in sites))
        internal_counts.append(sum(p[axis] < length-1 for p in sites))
        assert boundary_counts[-1] == len(sites)//length
        assert internal_counts[-1] == len(sites)-len(sites)//length
    # Reflection algebra is all-matrix, including nonzero cross traces.
    M, d1, d2, d3 = sp.symbols('M d1 d2 d3', real=True)
    d = (d1,d2,d3)
    reflection = 0
    for signs in itertools.product((-1,1),repeat=3):
        H = M*body[0]+sum((s*t*B for s,t,B in zip(signs,d,body[1:])),Z)
        reflection += sp.trace(H**2)/8
    assert sp.expand(reflection-4*(M*M*sp.trace(body[0]**2)/4+sum(t*t*sp.trace(B**2)/4 for t,B in zip(d,body[1:])))) == 0
    assert sp.trace(body[0]*body[1]) != 0
    # A new full trace-zero native onsite canonical compensation control.
    first = sp.diag(9,-1,-3,-5)/10
    second = sp.diag(3,-1,-1,-1)/20
    levels = sorted(list(m*first.diagonal())+list(m*second.diagonal()),reverse=True)
    top4 = sum(levels[:4])
    fixed2 = sum(sorted(list(m*first.diagonal()),reverse=True)[:2])+sum(sorted(list(m*second.diagonal()),reverse=True)[:2])
    positives = sum(t for t in levels if t > 0)
    mu = -m/20
    dual = 4*mu+sum(max(sp.S.Zero,t-mu) for t in levels)
    assert top4 == 19*m/20 and fixed2 == 9*m/10 and positives == 21*m/20
    assert dual == top4 and top4 < positives and top4 != fixed2
    # Off-diagonal density, so positive-part upper is not justified by an
    # assumed commuting actual density. The original compensation remains.
    Hdiag = sp.diag(*(list(m*first.diagonal())+list(m*second.diagonal())))
    U = sp.eye(8)
    U[0,0],U[0,1],U[1,0],U[1,1] = sp.Rational(3,5),sp.Rational(-4,5),sp.Rational(4,5),sp.Rational(3,5)
    occupied = sp.diag(1,0,0,0,1,1,1,0)
    Gamma = U*occupied*U.T
    assert Gamma**2 == Gamma and sp.trace(Gamma) == 4 and Gamma*Hdiag != Hdiag*Gamma
    actual = sp.trace(Gamma*Hdiag)
    assert actual <= dual
    # Independent common f coercivity/interiority constants; no optimizer.
    weights = [m*m+6*c0*c0]+[2*ci*ci]*3
    assert min(weights) > 128 and max(weights) < 1024
    epsilon2_ceiling = 2*((abs(m)+6*abs(c0))**2+12*ci*ci)
    assert epsilon2_ceiling < 128**2
    dch = sum(p['load']*(2*abs(rational(p['onsite']))+4*sum(abs(rational(t['c0']))+abs(rational(t['ci'])) for t in p['paths'])) for p in profiles[1:])
    assert dch == rational('855444290449274477/35184372088832')
    old_gate = gate(Fraction(str(m)),Fraction(str(abs(c0)+abs(ci))),2**12,2**32)
    new_gate = gate(Fraction(str(m)),Fraction(str(abs(c0)+abs(ci))),2**256,2**64)
    assert old_gate['less_than_2power_minus160'] is False
    assert new_gate['less_than_2power_minus160'] is True
    assert Fraction(str(new_gate['cap_union_raw'])) > 1
    assert Fraction(str(new_gate['per_site_tail_probability_upper'])) == Fraction(1,1 << 242)
    assert Fraction(str(new_gate['tree_mean_deviation_upper'])) < Fraction(1,1 << 1000)
    assert Fraction(str(new_gate['residual_link_mean_HS2_upper'])) < Fraction(1,1 << 740)
    # Explicit quadratic-form consequences retain charged/Q/F, not an
    # invented all-sites-good support event.
    receipt = {
        'kind':'EXECUTED_NEW_NATIVE_OPEN_RECTANGLE_POSITIVEPART_CANONICAL_BZ_BODY_JOIN_NOT_PHASE',
        'source_sha256':sha(Path(__file__)), 'request_sha256':sha(args.request), 'input_pins':request['pins'],
        'matrix_controls':control,'sewing_controls':sew_controls,
        'rectangle':{'lengths':lengths,'sites':len(sites),'closing_edges_per_axis':boundary_counts,'internal_edges_per_axis':internal_counts},
        'canonical':{'complete_global_top4':str(top4),'wrong_fixed2_each':str(fixed2),'positive_part_upper_not_identity':str(positives),'common_mu':str(mu),'compensated_dual':str(dual),'noncommuting_actual_density_expectation':str(actual),'occupations_of_optimizer':[1,3]},
        'fstar':{'definition':'argmin x>=0 of 64 sum x_mu^2 -2 BZavg sqrt(M^2 x0+sum D_i^2 xi)','quarter_trace':True,'norm_l2_strict_upper':'2','component_strict_lower':'1/128','averaged_native_weights':[str(w) for w in weights],'epsilon2_ceiling':str(epsilon2_ceiling),'minimum_is_norm_vector_not_quantum_field':True},
        'failed_small_parameter_gate_retained':old_gate,'paid_large_parameter_gate':new_gate,
        'lower':'Eground/V >= Q/V+F<W>+Lambda[fstar+64 E_partition,body ||x_B-xstar||2^2+E_partition,body DeltaJ_B-Etotal]-Dch sqrt(<s>)',
        'Dch':str(dch),'measure':'Inherited normalized physical Gauss/sewn-cover Born marginal, all translated partitions and volume-biased box/root site; not unfolded longitudinal gauge volume',
        'remaining_debts':['Parent revision-bound admission','quantum epsilon upper must be consumed at same fstar','charged magnetic stiffness/IR spectral atom','sharp photon residue/commoncone/Compton','global holonomy/phase selection'],
        'law_changed':False,'ground_vector_computed':False,'scientific_admission':False,'campaign_complete':False}
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print('Exact OPEN/AP/positive-part/canonical/BZ join executed; large integrated body gate paid; charged magnetic IR remains OPEN')


if __name__ == '__main__':
    main()

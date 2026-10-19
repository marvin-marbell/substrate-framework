"""New hash-bound rational formation consumer; never reruns cached extraction.

Bounds concern exact stored field0 Gram.T / represented N0 in its 15/25
orthonormal basis. Historical machine arithmetic assumptions remain explicit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import sympy as S

ROOT = Path(__file__).resolve().parent
PREFIX = 'three_cell_density_formation_'
PINS = {
 'connected_root_exterior_actual1_20261011_moments.npz': '4c6c6e0da22e0d9d405c37935332bb9548fbf2df5f429ec66b50aa671f8af67d',
 'local_so5_resum_actual04_20261011_state.npz': 'dd06400e1b7f70df935cf7ca02cf5ebf598481a8d5e209d1241aba5ed3fea2a2',
 'connected_root_exterior_scientist_20261011.py': 'c61389c52cd146bf1c4ecc189c7b46b2c964d53c1036a73a975b9859f3651e62',
 'harmonic_car_bridge_scientist_20261011.py': 'f0661dcdc6a49efbe1b635089436f9684a6c97926741ab7aceb2f78e577e28bf',
 'three_cell_native_scientist_20261011.py': 'bed006710905adc3e4574d3cdde42ba3c29ce9d59677bef3c2cbcaa47f628557',
 'harmonic_car_bridge_typed_attempt3_20261011_receipt.json': 'be07db74d79630fd62866e4a44eb32e8b9217eaf59bf2ce1f26fdc247e91ff21',
}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def exact_binary_norm(x: np.ndarray) -> F:
    """Integer accumulation of IEEE-754 significand squares, no FP sum."""
    bins = [0] * 2047
    for chunk in range(0, x.size, 32768):
        bits = x[chunk:chunk+32768].view(np.uint64)
        for raw in bits:
            word = int(raw)
            exponent = (word >> 52) & 2047
            if exponent == 2047:
                raise ValueError('nonfinite source')
            mantissa = word & ((1 << 52)-1)
            if exponent:
                mantissa += 1 << 52
            bins[exponent] += mantissa*mantissa
    # Subnormals and exponent=1 both use a 2^-1074 significand scale.
    integer = bins[0] + sum(value << (2*(exponent-1))
                            for exponent, value in enumerate(bins[1:], 1))
    return F(integer, 1 << 2148)


def gamma(k: int) -> F:
    u = F(1, 2**53)
    return k*u/(1-k*u)


def controls() -> dict[str, object]:
    """Independent exact low-degree controls, not the all-n proof."""
    z, a, b, c, d = variables = S.symbols('z a b c d')
    Q = a*d-b*c
    def basis(n: int) -> list[S.Expr]:
        out = []
        for l in range(n+1):
            for p in range(l+1):
                for q in range(l+1):
                    psi = S.Integer(0)
                    scale = S.sqrt(S.Rational(math.factorial(p)*math.factorial(q)*math.factorial(l-p)*math.factorial(l-q), math.factorial(l)))
                    for t in range(l+1):
                        ks = (l-p-q+t, q-t, p-t, t)
                        if min(ks) >= 0:
                            psi += scale*S.prod(v**k/math.factorial(k) for v,k in zip(variables[1:], ks))
                    m = n-l
                    poly = sum(S.Rational(math.factorial(m), 2**r*math.factorial(r)*math.factorial(m-2*r))*Q**r*z**(m-2*r)/S.rf(l+2,r)*psi for r in range(m//2+1))
                    C = F(math.factorial(m)*math.factorial(l+1)*math.factorial(2*n+2), math.factorial(n+1)*math.factorial(n+l+2))*F(2)**(l-n)
                    out.append(S.expand(poly/S.sqrt(S.Rational(C.numerator,C.denominator))))
        return out
    polys = basis(2)
    def inner(x: S.Expr, y: S.Expr) -> S.Expr:
        px, py = S.Poly(S.expand(x), *variables), S.Poly(S.expand(y), *variables)
        return S.simplify(sum((v*py.coeff_monomial(k)*math.prod(math.factorial(t) for t in k) for k,v in px.terms()), S.Integer(0)))
    G = S.Matrix([[inner(p,q) for q in polys] for p in polys])
    assert G == S.eye(14)
    # A genuine symmetric offdiagonal source: xy; ordered extractions are equal.
    source = a*d
    ordered = [S.diff(source,x,y)/S.sqrt(2) for x in variables for y in variables]
    proper = [S.diff(source,x,y)/S.sqrt(2)*(1 if i==j else S.sqrt(2)) for i,x in enumerate(variables) for j,y in enumerate(variables) if i<=j]
    wrong_fold = [S.diff(source,x,y)/S.sqrt(2) for i,x in enumerate(variables) for j,y in enumerate(variables) if i<=j]
    assert sum(v*v for v in ordered) == sum(v*v for v in proper) == 1
    assert sum(v*v for v in wrong_fold) == S.Rational(1,2)
    assert sum(v*v for v in ordered)/2 != 1  # wrong N0=2 instead of1
    # Relative sign isometry alone cannot detect: flip a-channel before stage1.
    assert S.diff(S.diff(source,a),d) != -S.diff(S.diff(source,d),a)
    # All-n coefficient normalization ratios from the closed factorial norm.
    n,l,p,q = S.symbols('n l p q', integer=True, positive=True)
    def C(nn: S.Expr,ll: S.Expr) -> S.Expr:
        return S.factorial(nn-ll)*S.factorial(ll+1)*S.factorial(2*nn+2)*2**(ll-nn)/(S.factorial(nn+1)*S.factorial(nn+ll+2))
    ratios = [((n-l)**2*C(n-1,l)/(n*C(n,l)), (n-l)*(n+l+2)/(n*(2*n+1))),
              ((l-p)*(l-q)*C(n-1,l-1)/(n*l*C(n,l)), (n+l+1)*(n+l+2)*(l-p)*(l-q)/(2*n*l*(l+1)*(2*n+1))),
              ((p+1)*(q+1)/(l+1)*((n-l)*(n-l-1)/(2*(l+2)))**2*C(n-1,l+1)/(n*C(n,l)), (n-l)*(n-l-1)*(p+1)*(q+1)/(2*n*(l+1)*(l+2)*(2*n+1)))]
    assert all(S.simplify(left-right)==0 for left,right in ratios)
    r0=(n-l)*(n+l+2)/(n*(2*n+1))
    down=(n+l+1)*(n+l+2)*l/(2*n*(l+1)*(2*n+1))
    up=(n-l)*(n-l-1)*(l+2)/(2*n*(l+1)*(2*n+1))
    assert S.factor(r0+down+up-1)==0
    return {'exact_degree2_polynomial_orthonormality': True,
            'all_n_symbolic_normalization_ratios': True,
            'all_n_column_sum': True,
            'wrong_sqrt2_fold_rejected': 'norm squared1/2 instead of1',
            'wrong_normalizer_rejected': 'N0=2 gives trace1/2',
            'wrong_relative_coefficient_sign_rejected': 'breaks ordered double commutation even when first-stage isometry survives',
            'small_degree_controls_are_not_all_n_proof': True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    observed = {name: sha(ROOT/name) for name in PINS}
    if observed != PINS:
        raise ValueError('input SHA mismatch')
    with np.load(ROOT/'local_so5_resum_actual04_20261011_state.npz', allow_pickle=False) as data:
        x = np.asarray(data['state'])
        if x.dtype != np.dtype('float64') or x.size != 9672320:
            raise ValueError('wrong immutable state layout')
        N = exact_binary_norm(x.reshape(-1))
    receipt = json.loads((ROOT/'harmonic_car_bridge_typed_attempt3_20261011_receipt.json').read_bytes())
    N0 = F(float(receipt['source_norm_float']))
    if N0 <= 0 or N <= 0:
        raise ValueError('nonpositive source norm')
    # ||stacked |E||| <= sqrt(maxrow2*maxcol9)<5. Each CSR dot has<=2 terms.
    u = F(1,2**53)
    alpha = 3*u
    stage = 5*(alpha+gamma(4)*(1+alpha))
    two = 2*stage+stage*stage
    fold = 2*two+3*u*(1+2*two)
    rows = {}
    with np.load(ROOT/'connected_root_exterior_actual1_20261011_moments.npz', allow_pickle=False) as data:
        for role,key,d,e in [('AA','active_pair',15,fold),
                             ('UU','auxiliary_pair',15,fold),
                             ('AU','mixed_pair',25,two)]:
            # The shape-derived sizes below are fixed dim(n) formulas, not observations.
            size = {'AA': 690880, 'UU': 14*((124+1)*(124+2)*(2*124+3)//6),
                    'AU': 5*((125+1)*(125+2)*(2*125+3)//6)}[role]
            blocks = (size+32767)//32768
            kappa = gamma(65536)+(1+gamma(65536))*gamma(blocks)
            extraction_price = e*(2+e)
            gram_price = d*kappa*(1+e)**2
            delta = N/N0*(extraction_price+gram_price)+abs(N/N0-1)
            G = data[key]
            if G.shape != (2*d,2*d) or G.dtype != np.dtype('float64') or not np.isfinite(G).all():
                raise ValueError('cached Gram layout/nonfinite')
            trace0 = sum((F(float(G[i,i])) for i in range(d)),F(0))
            trace1 = sum((F(float(G[i+d,i+d])) for i in range(d)),F(0))
            if abs(trace0/N0-1)>delta:
                raise ValueError('necessary trace control fails derived bound')
            rows[role] = {'delta_trace_norm_exact': str(delta), 'delta_trace_norm_float_display': float(delta),
                          'basis_dimension':d,'environment_dimension':size,'stream_blocks':blocks,
                          'extraction_vector_relative_error_exact':str(e),
                          'gram_roundoff_relative_nuclear_price_exact':str(gram_price),
                          'field0_trace_exact':str(trace0),'field1_trace_exact':str(trace1),
                          'trace_control_is_necessary_not_proof':True,
                          'stored_key':key,'stored_field0_slice':[0,d],
                          'normalization':'exact real stored field0 Gram.T / exact represented N0'}
    rows['UA'] = {'delta_trace_norm_exact':rows['AU']['delta_trace_norm_exact'],
                  'delta_trace_norm_float_display':rows['AU']['delta_trace_norm_float_display'],
                  'identity':'exact unitary tensor swap of AU; metric owner pays floating implementation'}
    result = {'kind':'EXACT_RATIONAL_CONDITIONAL_CACHED_FIELD0_DENSITY_FORMATION_INTERFACE',
              'status':'CONDITIONAL_ON_EXPLICIT_HISTORICAL_OPERATION_MODEL_NOT_UNCONDITIONAL_MACHINE_CERTIFICATE',
              'source':'s=x/sqrt(N), x immutable binary64 values regarded as exact reals; producer did not renormalize x',
              'source_norm_squared_exact':str(N),'common_N0_exact':str(N0),
              'source_norm_vs_N0_relative_exact':str(abs(N/N0-1)),
              'errors':rows,'input_sha256':observed,'own_source_sha256':sha(Path(__file__)),
              'proof_file':PREFIX+'proof_20261011.json','controls':controls(),
              'proof_sha256':sha(ROOT/(PREFIX+'proof_20261011.json')),
              'operation_assumptions':[
                  'Fraction to float rounded to nearest binary64 (relative error<=u); math.sqrt correctly rounded binary64 (relative error<=u), yielding coefficient error<=3u.',
                  'Historical scipy CSR float64 multiplication uses each <=2-term row dot with <=4 rounded operations; no hidden lower precision.',
                  'Historical block@block.T uses binary64 products/sums, each dot at most32768 products and65536 rounding sites on every dependency path; FMA is allowed.',
                  'Historical gram+= uses binary64 addition; block accumulation dependency length<=ceil(environment_dimension/32768).',
                  'No arithmetic underflow, overflow, FTZ/DAZ, nonfinite values, or unmodelled approximate arithmetic in historical extraction/Gram; source and output finiteness checked, historical intermediate range not recoverable from cache.',
                  'Pinned cache is produced by pinned selected_pair_gram on pinned raw state. Hashes bind bytes but do not independently prove historical execution provenance.',
              ],
              'source_norm_operation_audit':'Original N0=float(np.sum(X*X)). We bypass its reduction model by exact integer accumulation of all binary64 squares and compare to the exact represented historical N0. No trace-agreement proof.',
              'excluded_floating_operations':'Kernel U decoding, U@G.T@U.T/N0, partial traces, W/S/M decoding and subsequent contractions belong to metric owner.',
              'nonclaims':['No g replacement or gap/infidelity certification','No physicalPhi/lattice-ground identification','No all-order stability, phase, IRatom/commoncone or photon pole','No old producer or solver rerun'],
              'elapsed_seconds':time.monotonic()-started}
    output = Path(args.output)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'interface':output.name,'errors':{r:v['delta_trace_norm_float_display'] for r,v in rows.items()},'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}))

if __name__ == '__main__':
    main()

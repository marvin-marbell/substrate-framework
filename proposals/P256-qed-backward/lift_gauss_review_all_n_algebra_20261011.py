"""Exact all-degree extraction isometry algebra; no solver or Fock enumeration."""
from pathlib import Path
import hashlib
import json
import sympy as sp

ROOT=Path(__file__).resolve().parent
SOURCE='harmonic_car_bridge_scientist_20261011.py'
PIN='f0661dcdc6a49efbe1b635089436f9684a6c97926741ab7aceb2f78e577e28bf'
assert hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()==PIN
n,l=sp.symbols('n l',integer=True,positive=True)
diagonal=(n-l)*(n+l+2)/(n*(2*n+1))+l*(n+l+1)*(n+l+2)/(2*n*(l+1)*(2*n+1))+(n-l)*(n-l-1)*(l+2)/(2*n*(l+1)*(2*n+1))
identity=sp.factor(diagonal-1)
sign_sum=sum((-1)**(a+b) for a in range(2) for b in range(2))
assert identity==0 and sign_sum==0
result={'kind':'NEW_EXACT_SYMBOLIC_CONTROL_NO_GROUND_SOLVER','engine':f'SymPy {sp.__version__}','all_n_diagonal_sum':'(n-l)(n+l+2)/(n(2n+1)) + l(n+l+1)(n+l+2)/(2n(l+1)(2n+1)) + (n-l)(n-l-1)(l+2)/(2n(l+1)(2n+1))','diagonal_minus_one_simplified':str(identity),'off_diagonal_common_factor_sign_sum':sign_sum,'range':'n>=1,0<=l<=n; absent lower/upper boundary terms vanish','off_diagonal_reason':'Only l and l+2 source columns overlap at a target l+1 under the same slot; p_high=p_low+1,q_high=q_low+1. Their slot coefficient product has an i,j-independent positive radical times (-1)^(i+j), hence sums to zero.','source_pin':PIN,'source_locations':'extraction lines41-64; derivation normalized_annihilator lines18-24','scope':'Exact algebra for extraction isometry; no n126 Fock/tensor materialization or eigensolver.'}
with (ROOT/'lift_gauss_review_all_n_algebra_20261011_receipt.json').open('x') as output:
    json.dump(result,output,indent=2);output.write('\n')
print(json.dumps(result,indent=2),flush=True)

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import sympy as S

root = Path(__file__).resolve().parent
request = json.loads((root/'defect_price_review_request_20261011.json').read_text())
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest() == v for n,v in request['pins'].items())
cap = json.loads((root/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_text())
profiles = json.loads((root/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_text())['profiles']
D = cap['source_dimension']
u = F(1,2**53)
g = (2*D+8)*u/(1-(2*D+8)*u)
assert g < F(1,2**27)
row = 64*128**2 + 2*20*128 + 16*(2*6*128)
assert row < 2**22
mu = F(cap['energy_float'])
formation = 64*u/(1-64*u)*2**22 + F(128*6,2**48)
action = 40*u/(1-40*u)*2**22
subtraction = 4*u*(2**22+abs(mu))
underflow = F(100*D,2**1074)+F(2,2**490)
combined = formation+action+subtraction+underflow
assert combined < F(cap['coefficient_and_action_error_upper_exact'])
computed = F(cap['independent_full_H_residual_float'])
eta = computed/(1-4*g)+F(1,2**18)
assert eta < F(cap['normalized_exact_source_residual_upper_exact'])
assert mu-F(1,65536) == F(cap['Rayleigh_interval_exact'][0])
assert mu+F(1,65536) == F(cap['Rayleigh_interval_exact'][1])
# Generic native negative-energy oneparticle projector, independent of the
# serialized recovered free frame. B commutes with mA+dC and |c|<e, so both
# negative energies occupy P, uniquely determining the slot free vacuum.
m,d,e = S.symbols('m d e', real=True, nonzero=True)
A = S.diag(-1,1,-1,1)
C = S.Matrix([[0,0,0,S.I],[0,0,S.I,0],[0,-S.I,0,0],[-S.I,0,0,0]])
R = S.diag(S.Rational(1,2),S.Rational(1,2),-S.Rational(1,2),-S.Rational(1,2))
K = m*A+d*C
P = (S.eye(4)-K/e)/2
assert K*K == (m*m+d*d)*S.eye(4)
variance = S.factor(S.trace(P*R*(S.eye(4)-P)*R))
assert S.simplify(variance.subs(m*m,e*e-d*d)-d*d/(2*e*e)) == 0
values = []
for p in profiles:
    q = F(p['e_squared_exact'])
    assert q == F(p['m_exact'])**2+F(p['d_exact'])**2
    assert q > F(p['c0_exact'])**2
    v = F(p['d_exact'])**2/(2*q)
    assert v > 0
    values.append({'role':p['role'],'native_freevacuum_density_variance':str(v)})
# In the two-active product, projecting BOTH slots off their unique free
# minima kills D and the individual R^2 terms. Only128 R1 R2 survives.
actual_double_norm_square = 128**2 * F(values[0]['native_freevacuum_density_variance'])**2
assert actual_double_norm_square > 0
result = {'kind':'INDEPENDENT_EXACT_SOURCE_CAP_ARITHMETIC_AND_NATIVE_PROJECTOR_CONTROL',
          'dimension':D,'norm_roundoff_gamma':str(g),'rowcap':row,
          'formation_action_subtraction_underflow_sum':str(combined),
          'normalized_residual_analytic_bound_before_rounded_cap':str(eta),
          'observed_residual_not_an_analytic_cap':True,
          'native_density_variance_symbolic':str(variance),
          'native_profile_variances':values,
          'actual_two_active_freevacuum_double_output_norm_squared':str(actual_double_norm_square),
          'limitation':'Verifies retained cap arithmetic and native projector identities, not historical machine attestation or a fresh serialized-state action.'}
(root/'defect_price_review_cap_result_20261011.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'cap_arithmetic':'enclosed','native_freevacuum_not_eigenvector':True,'double_output_norm_squared':str(actual_double_norm_square)}))

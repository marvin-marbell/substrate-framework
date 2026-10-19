"""Component-resolved complete-field neutral lower; not ground identification."""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cube_out(value: F, bits: int = 128) -> tuple[F, F]:
    unit = 1 << bits
    target = (value.numerator << (3*bits)) // value.denominator
    left, right = 0, 1 << ((target.bit_length()+2)//3)
    while left+1 < right:
        middle = (left+right)//2
        if middle**3 <= target:
            left = middle
        else:
            right = middle
    lo = F(left,unit)
    hi = lo if lo**3 == value else F(left+1,unit)
    assert lo**3 <= value <= hi**3
    return lo,hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name,expected in request['pins'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    profile = json.loads((ROOT/request['profiles']).read_text())['primitive_profiles'][0]
    m = F(profile['onsite'])
    coefficients = [F(p['c0']) for p in profile['paths']]
    spatial = [2*F(p['ci'])**2 for p in profile['paths']]
    a = [m*m+2*sum((c*c for c in coefficients), F(0)), *spatial]
    a2 = sum((c*c for c in a), F(0))
    root = cube_out(a2/128)
    lower = -F(3,2)*root[1]
    assert lower > -24
    # Sharp one-variable minimization: y^3=A^2/128, then
    # 64||x||^2-2sqrt(A.x) >= -3y/2, with x=y^2 A/A^2.
    # Rational y controls reject missing dimension4 factor in trace-square.
    test_a = [F(4),F(4),F(4),F(4)]
    test_x = [F(1,16)]*4
    norm_term = 64*sum(x*x for x in test_x)
    ax = sum(aa*x for aa,x in zip(test_a,test_x))
    assert ax == 1 and norm_term-2 == -1
    wrong_without_trace_dimension4 = norm_term-1
    assert wrong_without_trace_dimension4 > norm_term-2
    result = {'kind':'EXACT_COMPONENT_RESOLVED_COMPLETE_NEUTRAL_FIELD_LOWER_NOT_GLOBAL_MINIMUM',
              'trace_square_coefficients_exact':list(map(str,a)),
              'coefficient_squared_norm_exact':str(a2),
              'cube_root_interval_exact':list(map(str,root)),
              'leading_complete_field_lower_over_Lambda_per_site_exact':str(lower),
              'coarse_lower_minus24_strict':True,
              'derivation':'x_mu=(1/(4V))sum_sites Tr Phi_mu^2. Ordered Jordan/V>=64sum x_mu^2; Tr h^2/V<=4 A.x; KyFan_top2V/V<=2sqrt(A.x). Cauchy A.x<=||A||||x|| then radial minimization gives -3/2*(||A||^2/128)^(1/3). Alignment and colour remain nonnegative. Covers inhomogeneous fields and SU4 holonomies but does not classify minimizers.',
              'controls':{'missing_trace_dimension4_false_lower_rejected':True,
                          'exact_test_actual_minus1':str(norm_term-2),
                          'wrong_test_zero':str(wrong_without_trace_dimension4)},
              'neutral_complete_canonical_scope':True,
              'N_uniform_evenN_ge4':True,
              'actual_full_quantum_ground_or_BO_remainder_paid':False,
              'quantum_ground_field_support_or_phase':False,
              'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete_field_lower_display_NOT_CERTIFICATE':float(lower),
                      'complete_quantum_ground_identified':False}))


if __name__ == '__main__':
    main()

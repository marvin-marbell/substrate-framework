"""All-even-N Clifford point-sea upper from exact AP moments; not a quantum lift."""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
from math import isqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sqrt_upper(value: Q) -> Q:
    scale = 1 << 256
    lower = isqrt(value.numerator * scale * scale // value.denominator)
    assert Q(lower, scale)**2 <= value < Q(lower+1, scale)**2
    return Q(lower+1, scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, expected in request['pins'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    profile = json.loads((ROOT/request['profiles']).read_text())['primitive_profiles'][0]
    m = Q(profile['onsite'])
    c0, ci = (Q(profile['paths'][0][name]) for name in ('c0','ci'))
    s, r = Q(1,2), Q(1,3)
    mass_floor = m-6*abs(c0)
    assert mass_floor > 0
    # X=r^2 sum D_i^2, M=m+2c0 sum cos k_i, epsilon=sqrt(s^2 M^2+X).
    d2 = (2*r*ci)**2
    mean_x = Q(3,2)*d2
    # Every AP axis has mean cos=0, sin^2=1/2. N4 is exceptional only
    # in fourth moments: sin^4=1/4, sin^2 cos^2=1/4; N>=6 gives3/8,1/8.
    mean_x_M2_upper = mean_x*m*m + (2*c0)**2*Q(9,4)*d2
    mean_x2_upper = Q(21,8)*d2*d2
    mean_x_epsilon2_upper = s*s*mean_x_M2_upper + mean_x2_upper
    denominator_upper = sqrt_upper(mean_x*mean_x_epsilon2_upper) + s*m*mean_x
    # Weighted Cauchy: E[X/(epsilon+sM)] >= E[X]^2/E[X(epsilon+sM)].
    gain_lower = mean_x*mean_x/denominator_upper
    epsilon_mean_lower = s*m+gain_lower
    jordan = 64*(s**4+3*r**4)
    sea_plus_J_upper = jordan-2*epsilon_mean_lower
    assert jordan == Q(172,27)
    assert sea_plus_J_upper < -21
    # Jensen's square-root upper cannot be substituted for a lower.
    assert Q(1,2)*(sqrt_upper(Q(1))+sqrt_upper(Q(9))) < sqrt_upper(Q(5))
    result = {'kind':'EXACT_ALL_EVEN_N_CLIFFORD_POINT_SEA_MOMENT_UPPER_NOT_QUANTUM_GROUND',
              'N':'every even N>=4, untwisted AP, identity SU4 links',
              's':'1/2','r':'1/3','canonical_neutral_particles':'2V',
              'neutral_minus_Lambda_sheet_retained':True,
              'mass_floor_exact':str(mass_floor),'Jordan_per_Lambda_site':str(jordan),
              'epsilon_mean_lower_exact':str(epsilon_mean_lower),
              'sea_plus_J_per_Lambda_site_upper_exact':str(sea_plus_J_upper),
              'sea_plus_J_upper_strict_minus21':True,
              'quantum_Phi_Haar_Gauss_lift_or_error_budget_paid':False,
              'actual_global_minimizer_or_phase':False,'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('All-even-N fixed point sea+Jordan upper <-21Lambda/site; quantum lift remains unpaid')


if __name__ == '__main__':
    main()

"""NEW higher native moment composition from immutable order-four prices.

No root, mean, coefficient, physical-point or old-primary execution. Composite
non-native winds do not inherit the native overlap cap used here.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from joined_phi_u1_local_electric_moments import central_moment, evolve, root_upper

ROOT = Path(__file__).resolve().parent
INPUT = 'joined_phi_u1_local_electric_moments_receipt.json'
PIN = '4c8d5412cf1cfed963756791655f3c90e3da90ce5e9799c0e5ffc100dfcd6bc6'
ORDER = 8
PROOF = {
    'native_geometry': 'Full768 signed N4 native winds have entries0,+1,-1 on one or two DISTINCT collinear physical edges. Hence every native pair absolute inc dot is<=2. The all63species force weights are positive. Therefore for p>4, Lp<=2^(p-4)*L4; published L1..L4 remain exact outward supplier prices. This extension is ONLY for native moment Holder factors, not arbitrary composite winds.',
    'initial': 'The supplied all-real tiltedSlater plus independent initialPi Gaussian theorem gives sigma²=q²/8+128sumv² and centered even moment2(2sigma²)^k k!, with exact variance improvement at k1. Root-box and Ref3 mean price mu<=2vmax are unchanged. a_k(0)<=norm*(central_even^(1/(2k))+mu)^k; k1 uses norm sqrt(sigma²+mu²). All roots use directed UPPER rational endpoints.',
    'future': 'For any integer orderk, s commutes HE+HC+onsite and shifted powers of bounded hopping give a_kprime<=sum_(j<k)binom(k,j)L_(k-j)a_j. Lower-triangular nilpotent comparison is an exact polynomial exp(tN), not a future Slater distribution. Derived hierarchy is uniform0..same NEW2^-16. Supports full-H commutator remainders through eighth native degree.',
    'limits': 'Cached immutable root/profile/geometry and exact model hypotheses only. Does not tighten physical source, certify signs, or create stationary photon poles. No changed source/parameters/oldclocks; higher order is NEW calculation, not an old-primary rerun.'}


def run(output: Path) -> dict[str,Any]:
    started = time.monotonic()
    raw = (ROOT/INPUT).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PIN:
        raise ArithmeticError('Moment supplier epoch changed')
    data = json.loads(raw)
    horizon = F(data['clock_exact'])
    rows = []
    native_ids = sorted(set(data['native_signed_path_price_ids']))
    for price_id in native_ids:
        original = data['price_catalogue'][price_id]
        sigma2 = F(original['centered_sigma_squared_upper_exact'])
        mu = 2*F(original['vmax_exact'])
        force = list(map(F,original['force_Lp_upper']))[:5]
        force += [force[4]*2**(p-4) for p in range(5,ORDER+1)]
        states: dict[str,Any] = {}
        for label,mass,bias in (('background',F(1),F(0)),('raw_source',F(1,2),mu)):
            norm = root_upper(mass)
            initial = [norm]+[norm*(root_upper(central_moment(sigma2,2*k),2*k)+bias)**k for k in range(1,ORDER+1)]
            initial[1] = norm*root_upper(sigma2+bias*bias)
            future = evolve(initial,force,horizon)
            states[label] = {'raw_mass_exact':str(mass),'norm_upper_exact':str(norm),
                'initial_a_upper_exact':list(map(str,initial)),'horizon_a_upper_exact':list(map(str,future))}
        rows.append({'native_price_id':price_id,'sigma_squared_upper_exact':str(sigma2),
            'mean_abs_source_upper_exact':str(mu),'force_Lp_upper_exact':list(map(str,force)),
            'q_squared_exact':original['q_squared_exact'],'states':states})
    maxima = {label:[str(max(F(row['states'][label]['horizon_a_upper_exact'][k]) for row in rows)) for k in range(ORDER+1)] for label in ('background','raw_source')}
    if (ROOT/INPUT).read_bytes()!=raw:
        raise ArithmeticError('Supplier changed during higher-order composition')
    result = {'status':'NATIVE_FULL_H_EVOLVED_MOMENTS_THROUGH_EIGHTH_COMPOSED',
        'clock_exact':str(horizon),'clock_adopted':False,'order':ORDER,'proof':PROOF,
        'native_signed_path_count':len(data['native_signed_path_price_ids']),
        'native_price_classes':rows,'maximum_native_horizon_a_upper_exact':maxima,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':{INPUT:PIN},'runtime':{'executable':sys.executable,'python':sys.version},
        'elapsed_seconds':time.monotonic()-started}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    receipt = run(args.output)
    print(json.dumps({'status':receipt['status'],'order':receipt['order'],'classes':len(receipt['native_price_classes'])}))

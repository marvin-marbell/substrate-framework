"""Exact all-momentum Wilson mass floors at a declared uniform Clifford point.

These are one-body band floors, not an interacting quantum ground gap, an
actual body-support theorem, a magnetic response or a photon discriminator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def mass_floor(profile: dict[str, Any]) -> dict[str, Any]:
    paths = {(p['axis'], p['length']): p for p in profile['paths']}
    onsite = Fraction(profile['onsite'])
    role = profile['role']
    if profile['scale'] == 'heavy':
        coefficients = [Fraction(paths[(axis, 1)]['c0']) for axis in range(3)]
        assert all(c < 0 for c in coefficients)
        minimum = onsite + 2 * sum(coefficients)
        identity = 'M(k)=m_min+sum_axis[-2c0_axis](1-cos k_axis)'
        squares = [-2 * c for c in coefficients]
        wrong = None
    else:
        nearest = [Fraction(paths[(axis, 1)]['c0']) for axis in range(3)]
        second = [Fraction(paths[(axis, 2)]['c0']) for axis in range(3)]
        assert all(nn == -4 * two for nn, two in zip(nearest, second))
        assert all(c > 0 for c in second)
        assert all(Fraction(paths[(axis, 2)]['ci']) == 0 for axis in range(3))
        # 2c0 t+2c2(2t^2-1)=4c2(t-1)^2-6c2, exactly.
        for nn, two in zip(nearest, second):
            assert [4 * two, 2 * nn, -2 * two] == [4 * two, -8 * two, 4 * two - 6 * two]
        minimum = onsite - 6 * sum(second)
        identity = 'M(k)=m_min+sum_axis[4c2_axis](cos k_axis-1)^2'
        squares = [4 * c for c in second]
        wrong = onsite + 2 * sum(nearest)
        assert wrong < 0
    assert minimum > Fraction(1, 8)
    return {'role': role, 'load': profile['load'], 'minimum_exact': str(minimum),
            'positive_remainder_coefficients_exact': [str(c) for c in squares],
            'all_real_momenta_identity': identity,
            'minimum_attained_at_k0_in_continuous_symbol': True,
            'AP_finiteN_and_flat_twists_lower_is_same': True,
            'missing_length2_wrong_floor_exact': None if wrong is None else str(wrong),
            'missing_length2_positive_mass_proof_rejected': wrong is not None,
            'at_s_ge_half_onebody_band_separation_lower': str(minimum),
            'interacting_fullH_gap_inference_allowed': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for path, h in request['pins'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == h
    profiles = json.loads((ROOT / request['profiles']).read_text())['primitive_profiles']
    result = {'kind': 'EXACT_NATIVE_ALL_MOMENTA_ONEBODY_POINT_MASS_CONTROL_NOT_QUANTUM_BODY_OR_PHASE',
              'profiles': [mass_floor(p) for p in profiles],
              'uniform_Clifford_only': True,
              'proof': 'h(k)^2=[s^2 M(k)^2+r^2 sum_axis D_axis(k)^2] I4 by exact Clifford anticommutation; positive-negative separation >=2s m_min. Physical neutral scale multiplies the gap by Lambda but reverses the occupied sheet.',
              'finite_N4_point_gap_is_not_uniform_gap': True,
              'actual_Phi_Haar_Gauss_ground_support_or_quantum_U1_gap_paid': False,
              'lowband_magnetic_Gram_or_stiffness_or_photon_paid': False,
              'law_or_field_ansatz_adopted': False, 'campaign_complete': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('All momenta exact native point mass floors; no interacting full-H gap claim')


if __name__ == '__main__':
    main()

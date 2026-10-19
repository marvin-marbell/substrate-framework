"""Native neutral canonical sea and ordered Jordan point-body discriminator.

No quantum Phi/Haar/U1 source, global minimizing field, BO error or phase is
constructed. The neutral scale -Lambda is retained, not replaced by charged
or frozen local-sector data. All numerical sea prices are FLOAT diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent


def rational(value: str) -> sp.Rational:
    return cast(sp.Rational, sp.Rational(value))


def gammas() -> tuple[sp.Matrix, list[sp.Matrix]]:
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.diag(1, -1)
    beta = cast(sp.Matrix, sp.kronecker_product(sz, sp.eye(2)))
    alpha = [cast(sp.Matrix, sp.kronecker_product(sx, sigma)) for sigma in (sx, sy, sz)]
    for left, right in itertools.product([beta, *alpha], repeat=2):
        expected = 2 * sp.eye(4) if left == right else sp.zeros(4)
        assert left * right + right * left == expected
    return beta, alpha


def jordan(fields: list[sp.Matrix]) -> sp.Expr:
    # Lambda/(16a)=4Lambda; ordered mu,nu pairs, including both cross orders.
    return cast(sp.Expr, 4 * sum(sp.trace((a * b + b * a) ** 2)
                               for a, b in itertools.product(fields, repeat=2)))


def exact_point_controls(profile: dict[str, Any]) -> dict[str, Any]:
    beta, alpha = gammas()
    s, r = sp.Rational(1, 2), sp.Rational(1, 3)
    clifford = [s * beta, *[r * a for a in alpha]]
    commuting = [s * beta, *[r * beta for _ in range(3)]]
    jc, jm = jordan(clifford), jordan(commuting)
    assert jc == 64 * (s**4 + 3 * r**4)
    assert jm == 64 * (s**2 + 3 * r**2)**2 and jm > jc
    theta = sp.pi / 4  # one actual AP N4 momentum in every axis
    phase = (1 + sp.I) / sp.sqrt(2)
    assert sp.simplify(sp.expand_complex(sp.exp(sp.I * theta)) - phase) == 0
    symbol = cast(sp.Matrix, rational(profile['onsite']) * clifford[0])
    native = symbol.copy()
    for path in profile['paths']:
        axis = path['axis']
        assert path['length'] == 1
        f = cast(sp.Matrix, rational(path['c0']) * clifford[0] + sp.I * rational(path['ci']) * clifford[axis + 1])
        native = cast(sp.Matrix, native + f * phase + f.H * sp.conjugate(phase))
        symbol = cast(sp.Matrix, symbol + 2 * rational(path['c0']) * clifford[0] * sp.cos(theta)
                      - 2 * rational(path['ci']) * clifford[axis + 1] * sp.sin(theta))
    assert cast(sp.Matrix, native - symbol).applyfunc(sp.simplify) == sp.zeros(4)
    assert cast(sp.Matrix, symbol - symbol.H).applyfunc(sp.simplify) == sp.zeros(4)
    square = cast(sp.Matrix, symbol * symbol)
    epsilon2 = sp.simplify(square[0, 0])
    assert (square - epsilon2 * sp.eye(4)).applyfunc(sp.simplify) == sp.zeros(4)
    # The physical neutral sea fills the positive h band because its scale is -Lambda.
    epsilon = sp.sqrt(epsilon2)
    neutral_energy_over_Lambda = -2 * epsilon
    wrong_charged_scale = 2 * epsilon
    assert neutral_energy_over_Lambda < 0 and wrong_charged_scale > 0
    return {'Clifford_Jordan_exact': str(jc), 'commuting_Jordan_exact': str(jm),
            'ordered_cross_omission_rejected': bool(jm > jc),
            'native_AP_endpoint_symbol_exact': True, 'spectrum_exact': '[-epsilon,-epsilon,+epsilon,+epsilon]',
            'epsilon_squared_exact': str(epsilon2), 'canonical_neutral_particles_per_momentum': 2,
            'physical_minus_Lambda_sheet_retained': True, 'wrong_positive_neutral_sheet_rejected': True}


def sea_price(profile: dict[str, Any], n: int) -> dict[str, Any]:
    theta = np.array(list(itertools.product((2 * np.arange(n) + 1) * np.pi / n, repeat=3)))
    mass = np.full(n**3, float(rational(profile['onsite'])))
    kinetic2 = np.zeros(n**3)
    signed_kinetic = np.zeros(n**3)
    for path in profile['paths']:
        axis = path['axis']
        mass += 2 * float(rational(path['c0'])) * np.cos(theta[:, axis])
        kinetic = -2 * float(rational(path['ci'])) * np.sin(theta[:, axis])
        kinetic2 += kinetic * kinetic
        signed_kinetic += kinetic

    def ray(z: float, commuting: bool = False) -> dict[str, float]:
        band = np.abs(mass + z * signed_kinetic) if commuting else np.sqrt(mass * mass + z * z * kinetic2)
        average = float(np.mean(band))
        quartic = (1 + 3 * z * z)**2 if commuting else 1 + 3 * z**4
        s = (average / (128 * quartic)) ** (1 / 3)
        return {'z': z, 's': s, 'r': z * s,
                'neutral_canonical_sea_over_Lambda_per_site': -2 * s * average,
                'Jordan_over_Lambda_per_site': 64 * quartic * s**4,
                'total_over_Lambda_per_site': -1.5 * s * average,
                'point_neutral_gap_in_h': 2 * s * float(np.min(band))}

    opt = cast(Any, minimize_scalar(lambda z: ray(float(z))['total_over_Lambda_per_site'], bounds=(0, 4), method='bounded'))
    assert opt.success
    candidates = [ray(z) for z in (0.0, 0.5, 1.0, 2.0)]
    candidate = ray(float(opt.x))
    assert candidate['total_over_Lambda_per_site'] < candidates[0]['total_over_Lambda_per_site']
    return {'N': n, 'canonical_neutral_particles': 2 * n**3,
            'rays_FLOAT_ONLY': candidates, 'Clifford_optimizer_FLOAT_ONLY_NOT_GLOBAL_PROOF': candidate,
            'commuting_same_ratio_FLOAT_ONLY': ray(float(opt.x), True),
            'optimizer_not_interval_or_complete_field_minimum': True,
            'scalar_only_ray_rejected_as_best_in_this_tested_family': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    profile_data = json.loads((ROOT / request['profiles']).read_text())
    profile = profile_data['primitive_profiles'][0]
    assert profile['role'] == 'neutral heavy -Lambda' and profile['load'] == 1
    for path, expected in request['pins'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    result = {'kind': 'EXACT_NATIVE_NEUTRAL_MATRIX_AND_FLOAT_COMPLETE_CANONICAL_POINT_SEA_DISCRIMINATOR_NOT_GROUND',
              'controls': exact_point_controls(profile),
              'finite_sea_point_diagnostics': [sea_price(profile, n) for n in (4, 6, 8)],
              'complete_neutral_canonical_sector_at_declared_point': True,
              'uniform_Clifford_alignment_and_SU4_plaquette_at_identity_links': 0,
              'quantum_Phi_Haar_Gauss_error_or_global_BO_minimum_paid': False,
              'charged_U1_HEHC_response_or_ground_phase_paid': False,
              'native_minimizing_ground_computed': False,
              'law_or_field_ansatz_adopted': False, 'campaign_complete': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Native neutral minus-Lambda/ordered-Jordan controls; finite point sea diagnostics only')


if __name__ == '__main__':
    main()

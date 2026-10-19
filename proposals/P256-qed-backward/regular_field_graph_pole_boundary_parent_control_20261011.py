"""Exact spectral-model control: regular field graph and f-sum do not force an atom."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import sympy as s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    root = Path(__file__).resolve().parent
    request = json.loads(args.request.read_text())
    assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
               for name, sha in request['pins'].items())
    q: Any = s.symbols('q', positive=True)
    g: Any = s.symbols('g', positive=True)
    x: Any = s.symbols('x', real=True)
    omega: Any = s.symbols('omega', real=True)
    a: Any = s.Rational(1, 2)
    b: Any = s.sqrt(7) / 2
    density: Any = s.Rational(4, 3)
    atomic = [g * q ** (n - 1) for n in range(5)]
    continuous = [s.simplify(g * q ** (n - 1) *
                            s.integrate(density * x ** n, (x, a, b)))
                  for n in range(5)]
    assert s.simplify(continuous[1] - atomic[1]) == 0
    assert s.simplify(continuous[3] - atomic[3]) == 0
    assert s.simplify(continuous[0] - atomic[0]) > 0
    assert s.simplify(continuous[2] - atomic[2]) < 0
    assert s.simplify(continuous[4] - atomic[4]) > 0
    contact: Any = q ** 2 / (2 * g)
    responses = []
    for label, moments in (('one_atom', atomic), ('no_atoms', continuous)):
        # L Theta=-2ig E, L E=iJ => mu_J(dE)=E^4/(4g^2) mu_Theta(dE).
        static_current_inverse: Any = s.simplify(moments[3] / (4 * g ** 2))
        assert s.simplify(contact - 2 * static_current_inverse) == 0
        responses.append({'model': label, 'field_moments_0_to_4': [str(v) for v in moments],
                          'electric_norm_squared': str(s.simplify(moments[2] / (4 * g ** 2))),
                          'current_norm_squared': str(s.simplify(moments[4] / (4 * g ** 2))),
                          'current_inverse_moment': str(static_current_inverse),
                          'external_phase_contact_minus_two_inverse': '0'})
    atomic_c: Any = 2 * g / (omega ** 2 + q ** 2)
    continuous_c: Any = 4 * g / (3 * q ** 2) * s.log(
        (omega ** 2 + s.Rational(7, 4) * q ** 2) /
        (omega ** 2 + s.Rational(1, 4) * q ** 2))
    assert s.limit(omega ** 2 * atomic_c, omega, s.oo) == 2 * g
    assert s.limit(omega ** 2 * continuous_c, omega, s.oo) == 2 * g
    # q^2 dq is the three-spatial-dimensional radial measure: both regular-field
    # local IR weights are integrable. Neither family is a native-law construction.
    ir_atom: Any = s.integrate(q ** 2 * atomic[0], (q, 0, 1))
    ir_continuous: Any = s.integrate(q ** 2 * continuous[0], (q, 0, 1))
    assert s.simplify(ir_atom - g / 2) == 0
    assert s.simplify(ir_continuous - g * (s.sqrt(7) - 1) / 3) == 0
    result = {'kind': 'EXACT_REGULAR_FIELD_GRAPH_AND_CONTACT_SPECTRAL_ATOM_DISCRIMINATOR',
              'scope': 'two spectral models, not the native law or an interacting ground',
              'ground_and_insertion_domain_assumed_in_model': True,
              'graph': ['LTheta=-2igE', 'LE=iJ', 'mu_E=energy^2 mu_Theta/(4g^2)',
                        'mu_J=energy^4 mu_Theta/(4g^2)'],
              'atomic_field_measure': '(g/q) delta_q',
              'atomless_field_measure': '4g/(3q^2) denergy on [q/2,sqrt(7)q/2]',
              'shared_field_first_moment': str(atomic[1]),
              'shared_field_third_moment': str(atomic[3]),
              'shared_contact': str(contact), 'responses': responses,
              'euclidean_correlations': {'atom': str(atomic_c), 'continuum': str(continuous_c)},
              'omega_is_model_euclidean_frequency_not_fourhop_resolvent_shift': True,
              'three_spatial_dimensional_local_IR_weight': {'atom': str(ir_atom),
                                                           'continuum': str(ir_continuous)},
              'entire_ground_block': 'adding r delta_0 to mu_Theta changes m0 but none of these positive-energy moments; remove whole P0, not only the mean',
              'regular_linear_source': 'norm^2 of sTheta is s^2 m0, not sharp-character mass1',
              'wrong_inference_graph_contact_regular_IR_implies_atom_rejected': True,
              'neither_gapless_support_nor_static_phase_cancellation_identifies_a_pole': True,
              'native_ground_phase_stiffness_pole_adopted': False,
              'campaign_complete': False, 'elapsed_seconds': time.monotonic() - started}
    assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
               for name, sha in request['pins'].items())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'matched_first_third_and_contact': True,
                      'regular_IR_integrable_both': True,
                      'continuum_has_no_atom': True, 'native_photon_claim': False}), flush=True)


if __name__ == '__main__':
    main()

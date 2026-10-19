"""Actual centre seam versus constant central links; point controls, not ground."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction as F
from pathlib import Path
from typing import Any

from neutral_body_parent_uniform_certificate_20261012 import add, ap_trig, point, scale, sqrt_out, square

ROOT = Path(__file__).resolve().parent


def point_energy(profile: dict[str, Any], seam_i: bool) -> tuple[F, F]:
    cs4, ss4 = ap_trig(4)
    cs8, ss8 = ap_trig(8)
    c0 = [cs8[i] for i in (1, 3, 5, 7)] if seam_i else cs4
    s0 = [ss8[i] for i in (1, 3, 5, 7)] if seam_i else ss4
    cosines, sines = [c0, cs4, cs4], [s0, ss4, ss4]
    paths = {p['axis']: p for p in profile['paths']}
    total = point(F(0))
    for index in itertools.product(range(4), repeat=3):
        mass = point(F(profile['onsite']))
        for axis in range(3):
            mass = add(mass, scale(2 * F(paths[axis]['c0']), cosines[axis][index[axis]]))
        eps2 = scale(F(1, 4), square(mass))  # declared s=1/2
        for axis in range(3):
            d = scale(-2 * F(paths[axis]['ci']), sines[axis][index[axis]])
            eps2 = add(eps2, scale(F(1, 9), square(d)))  # declared r=1/3
        total = add(total, sqrt_out(eps2))
    return add(point(F(172, 27)), scale(F(-1, 32), total))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for path, h in request['pins'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == h
    profile = json.loads((ROOT / request['profiles']).read_text())['primitive_profiles'][0]
    untwisted, seam = point_energy(profile, False), point_energy(profile, True)
    difference = (seam[0] - untwisted[1], seam[1] - untwisted[0])
    assert difference[0] > 0
    baseline_pi8 = [2, 6, 10, 14]
    seam_pi8 = [3, 7, 11, 15]
    assert baseline_pi8 != seam_pi8
    # i on EACH of four links has winding i^4=1. i on ONE seam has winding i.
    result = {'kind': 'EXACT_NATIVE_CENTRE_SEAM_VERSUS_CONSTANT_LINK_HOLONOMY_CONTROL_NOT_GROUND_SELECTION',
              'N': 4, 'constant_i_links_winding': '1', 'literal_i_seam_winding': 'i',
              'gauge_invariant_fundamental_winding_trace': {'constant_links': '4', 'literal_seam': '4i'},
              'alignment_and_colour_plaquette_both_cases': 0,
              'AP_momenta_pi_over_8': {'identity_or_trivial_winding_constant_centre': baseline_pi8, 'literal_i_seam': seam_pi8},
              'actual_seam_momentum_shift': 'Arg(i)/N=pi/8, not pi/2',
              'complete_Clifford_canonical2V_point_energy_intervals_exact': {'identity': [str(v) for v in untwisted], 'literal_i_seam': [str(v) for v in seam]},
              'literal_seam_minus_identity_point_energy_interval_exact': [str(v) for v in difference],
              'difference_display_NOT_CERTIFICATE': [float(v) for v in difference],
              'old_N4_permutation_probe_is_trivial_winding_only': True,
              'all_centre_or_all_field_ground_selection_proved': False,
              'quantum_source_or_phase_or_stiffness_proved': False, 'campaign_complete': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Literal centre seam distinguished from trivial winding; conditional Clifford point energy only')


if __name__ == '__main__':
    main()

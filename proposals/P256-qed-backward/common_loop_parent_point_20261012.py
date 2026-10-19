"""Exact native-profile point/finite-CAR dark-sector controls, not a ground solver."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast
import zipfile

import sympy as sp

from canonical_ground_parent_singlet_20261012 import complementary_singlet, wedge_action

ROOT = Path(__file__).resolve().parent


def finite_filter_dark_control() -> dict[str, Any]:
    """Full two-copy wedge2 sector; a support complement need not see an orbital hop."""
    psi = complementary_singlet(4, 2)
    identity = sp.eye(6)
    g = sp.diag(1, -1, 1, -1)
    orbital = sp.zeros(4)
    orbital[0, 2] = orbital[2, 0] = 1
    orbital[1, 3] = orbital[3, 1] = 1
    assert orbital * g == g * orbital
    active = sp.kronecker_product(wedge_action(g, 2), identity)
    common = cast(Any, (sp.kronecker_product(wedge_action(orbital, 2), identity)
                       + sp.kronecker_product(identity, wedge_action(orbital, 2))))
    filtered = (sp.eye(36) + sp.Rational(1, 3) * active) * psi
    norm = (filtered.H * filtered)[0]
    ps = psi * psi.H / (psi.H * psi)[0]
    complement = sp.simplify((filtered.H * (sp.eye(36) - ps) * filtered)[0] / norm)
    assert complement > 0
    assert common * filtered == sp.zeros(36, 1)
    assert common.H * filtered == sp.zeros(36, 1)
    return {
        'scope': 'EXACT_FULL36_TWO_COPY_METHOD_NOT_NATIVE_ALL64_GROUND',
        'singlet_complement': str(complement),
        'orbital_hop_norm_squared': '0',
        'complement_implies_loop_Gram_lower_rejected': True,
    }


def native_point_controls(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute affected native onebody commutator entries, retaining AP length2 aliases."""
    g = sp.diag(3, -1, -1, -1)
    zeros = sp.zeros(4)
    for path in paths:
        c0, ci = sp.Rational(path['c0']), sp.I * sp.Rational(path['ci'])
        axis = int(path['axis'])
        coefficient = (c0 + (axis + 1) * ci) * g
        assert coefficient * g - g * coefficient == zeros
    results: list[dict[str, Any]] = []
    for n in (4, 6, 8):
        for epsilon in (sp.Rational(1, 2), sp.Rational(1, 4)):
            entries: dict[tuple[int, int], Any] = {}
            origin = (0, 0, 0)
            for path in paths:
                axis, length = int(path['axis']), int(path['length'])
                c0, ci = sp.Rational(path['c0']), sp.I * sp.Rational(path['ci'])
                incoming = tuple((-length) % n if j == axis else 0 for j in range(3))
                for x in (origin, incoming):
                    y = tuple((x[j] + length) % n if j == axis else x[j] for j in range(3))
                    lx = 1 + epsilon if x == origin else sp.Integer(1)
                    ly = 1 + epsilon if y == origin else sp.Integer(1)
                    seam = -1 if x[axis] + length >= n else 1
                    coefficient = seam * (c0 * (lx + ly) / 2 + ci * (axis + 1))
                    ix = (x[0] * n + x[1]) * n + x[2]
                    iy = (y[0] * n + y[1]) * n + y[2]
                    for colour in range(4):
                        hop = coefficient * g[colour, colour]
                        comm = (lx - ly) * g[colour, colour] * hop
                        key = (4 * iy + colour, 4 * ix + colour)
                        reverse = (key[1], key[0])
                        entries[key] = entries.get(key, sp.Integer(0)) + comm
                        entries[reverse] = entries.get(reverse, sp.Integer(0)) - sp.conjugate(comm)
            norm2 = cast(Any, sp.factor(sum(sp.conjugate(v) * v for v in entries.values())))
            assert norm2 > 0
            results.append({
                'N': n, 'oneparticle_dimension': 4 * n**3,
                'epsilon': str(epsilon), 'native_hopping_A_commutator_HS2': str(norm2),
                'affected_entries_collected': len(entries),
                'native_AP_and_length2_aliases_collected_before_norm': True,
            })
    # A noncommuting colour direction independently escapes the flat Cartan fixture.
    colour = sp.zeros(4)
    colour[0, 1] = colour[1, 0] = sp.Rational(1, 2)
    assert colour * g - g * colour != zeros
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, expected in request['pins'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    with zipfile.ZipFile(ROOT / request['profile_archive']) as archive:
        profile = json.loads(archive.read(request['profile_member']))
    neutral, active, auxiliary = profile['primitive_profiles']
    assert [neutral['load'], active['load'], auxiliary['load']] == [1, 1, 63]
    assert active['paths'] == auxiliary['paths']
    delta = sp.Rational(active['onsite']) - sp.Rational(auxiliary['onsite'])
    assert delta == -sp.Rational(126663739519795, 70368744177664)
    result = {
        'kind': 'EXACT_NATIVE_POINT_ORBITAL_DARK_AND_COMPLEMENT_BOUNDARY_NOT_QUANTUM_GROUND',
        'input_pins': request['pins'],
        'native_profile': {'charged_roles': 'active1+aux63', 'delta': str(delta),
                           'all_six_hop_paths': len(active['paths']), 'neutral_retained_in_H': True},
        'finite_filter': finite_filter_dark_control(),
        'native_nonuniform_point_controls': native_point_controls(active['paths']),
        'uniform_commuting_point': {
            'Phi0': 'diag(3,-1,-1,-1) at every site',
            'Phi_axis': '(axis+1)*Phi0', 'SU4_links': 'identity',
            'commutator': '[V_e,total active onsite A]=0 for every native charged path',
            'all64_all24': 'Each common charged hop kills P_S and commutes any function of A; therefore every ordered full64^4 bare loop kills F(A)P_S at this point, including adjoints. This is an algebraic identity, not an executed 64-copy Fock matrix or ground expectation.',
            'nearby_field_price': 'Point coefficients are continuous and vanish on the commuting configuration. No strictly positive uniform field-independent loop price follows from P_S complement alone.',
        },
        'point_not_fullPhi_SU4_quantum_source': True,
        'ground_density_on_degenerate_fields_claimed': False,
        'fullH_action_or_phase_or_IR_atom_proved': False,
        'old_producer_solver_root_checker_rerun': False,
        'campaign_complete': False,
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Exact native-point orbital-dark and complement boundary controls completed')


if __name__ == '__main__':
    main()

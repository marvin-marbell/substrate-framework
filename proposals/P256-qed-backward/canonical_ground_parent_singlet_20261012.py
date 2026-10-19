"""Exact charged common-singlet representation control, not a native ground solver."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any
import zipfile

import sympy as sp

ROOT = Path(__file__).resolve().parent


def wedge_action(body: Any, number: int) -> Any:
    masks = [m for m in range(1 << body.rows) if m.bit_count() == number]
    positions = {m: j for j, m in enumerate(masks)}
    result = sp.zeros(len(masks))
    for column, mask in enumerate(masks):
        for source in range(body.rows):
            if not mask & (1 << source):
                continue
            removed = mask ^ (1 << source)
            annihilation_sign = (-1) ** ((mask & ((1 << source) - 1)).bit_count())
            for target in range(body.rows):
                if removed & (1 << target):
                    continue
                creation_sign = (-1) ** ((removed & ((1 << target) - 1)).bit_count())
                row = positions[removed | (1 << target)]
                result[row, column] += annihilation_sign * creation_sign * body[target, source]
    return result


def complementary_singlet(dimension: int, number: int) -> Any:
    assert dimension == 2 * number
    masks = [m for m in range(1 << dimension) if m.bit_count() == number]
    positions = {m: j for j, m in enumerate(masks)}
    result = sp.zeros(len(masks) ** 2, 1)
    for j, mask in enumerate(masks):
        occupied = [a for a in range(dimension) if mask & (1 << a)]
        complement = [a for a in range(dimension) if not mask & (1 << a)]
        inversions = sum(a > b for a in occupied for b in complement)
        partner = ((1 << dimension) - 1) ^ mask
        result[j * len(masks) + positions[partner]] = (-1) ** inversions
    return result


def representation_controls() -> dict[str, Any]:
    singlet = complementary_singlet(4, 2)
    norm2 = (singlet.conjugate().T * singlet)[0]
    identity = sp.eye(6)
    generators = []
    for a, b in itertools.combinations(range(4), 2):
        real = sp.zeros(4); real[a, b] = real[b, a] = 1
        imaginary = sp.zeros(4); imaginary[a, b] = sp.I; imaginary[b, a] = -sp.I
        generators.extend([real, imaginary])
    generators.extend([sp.diag(1, -1, 0, 0), sp.diag(0, 1, -1, 0), sp.diag(0, 0, 1, -1)])
    assert len(generators) == 15 and norm2 == 6
    for generator in generators:
        action = wedge_action(generator, 2)
        common = sp.kronecker_product(action, identity) + sp.kronecker_product(identity, action)
        assert common * singlet == sp.zeros(36, 1)
    body = sp.diag(3, -1, -1, -1)
    body[0, 1] = sp.I; body[1, 0] = -sp.I
    active = sp.kronecker_product(wedge_action(body, 2), identity)
    trace2 = sp.trace(body * body)
    active_norm2 = (singlet.conjugate().T * active * active * singlet)[0] / norm2
    assert active_norm2 == trace2 / 3 == sp.Rational(14, 3)
    assert (singlet.conjugate().T * active * singlet)[0] == 0
    # Two independent rest multiplicity channels: carrier reset cannot alter this scalar.
    columns = sp.zeros(72, 2)
    columns[:36, 0] = singlet; columns[36:, 1] = singlet
    repeated_active = sp.diag(active, active)
    compression = columns.conjugate().T * repeated_active**2 * columns / norm2
    assert compression == active_norm2 * sp.eye(2)
    assert columns.conjugate().T * repeated_active * columns == sp.zeros(2)
    number_rows = []
    for a in range(4):
        mode = sp.zeros(4); mode[a, a] = 1
        occupation = wedge_action(mode, 2)
        common = sp.kronecker_product(occupation, identity) + sp.kronecker_product(identity, occupation)
        assert common * singlet == singlet
        number_rows.append(1)
    # Wrong uncoupled product has a nonzero common traceless charge.
    wrong = sp.zeros(36, 1); wrong[0] = 1
    diagonal = wedge_action(sp.diag(3, -1, -1, -1), 2)
    common = sp.kronecker_product(diagonal, identity) + sp.kronecker_product(identity, diagonal)
    assert common * wrong != sp.zeros(36, 1)
    return {
        'full_two_copy_CAR_dimension': 36, 'common_SU4_generators_annihilate': 15,
        'singlet_norm2_unnormalized': str(norm2), 'active_trace2': str(trace2),
        'active_squared_norm': str(active_norm2), 'multiplicity_squared_compression': str(compression),
        'pair_occupation_each_mode': number_rows, 'wrong_uncoupled_common_charge_rejected': True,
        'nonzero_onsite_mismatch_cannot_be_deleted': True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    before = {name: (ROOT / name).read_bytes() for name in request['pins']}
    for name, raw in before.items():
        assert hashlib.sha256(raw).hexdigest() == request['pins'][name], name
    with zipfile.ZipFile(ROOT / request['profile_archive']) as archive:
        raw_profile = archive.read(request['profile_member'])
    profiles = json.loads(raw_profile)['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    active, auxiliary = profiles[1:]
    assert active['paths'] == auxiliary['paths']
    delta = sp.Rational(active['onsite']) - sp.Rational(auxiliary['onsite'])
    assert delta == -sp.Rational(126663739519795, 70368744177664) != 0
    result = {
        'kind': 'EXACT_COMMON_CHARGED_SINGLET_AND_NATIVE_ONSITE_ESCAPE_CONTROL_NOT_PHASE',
        'representation': representation_controls(),
        'native_binding': {
            'all_charged_paths_exactly_equal': True, 'onsite_active_minus_auxiliary': str(delta),
            'N4_M': 256, 'N4_k': 128, 'N4_active_square_factor': '64/255',
            'common64_singlet_pairs': 32, 'charged_occupation_per_colour_mode': 32,
            'charged_occupation_per_site': 128, 'collective_background_per_site': 128,
            'profile_sha256': hashlib.sha256(raw_profile).hexdigest(),
        },
        'conditional_general_identity': 'P_S dGamma_active(h0)^2 P_S = k(M-k)/(M(M-1)) Tr(h0^2) P_S for traceless h0',
        'full_physical_projector_preservation_proof_review_pending': True,
        'strict_support_escape_not_quantitative_Gram_lower': True,
        'native_minimizing_ground_computed': False, 'physical1_phase_stiffness_IR_atom': False,
        'campaign_complete': False, 'input_pins': request['pins'],
    }
    assert all((ROOT / name).read_bytes() == raw for name, raw in before.items())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Exact common-singlet controls and native onsite mismatch completed')


if __name__ == '__main__':
    main()

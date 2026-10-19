"""Exact common-singlet/native-loop join; no ground, phase or IR promotion."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_epoch(spec: dict[str, Any]) -> dict[str, Any]:
    capsule = json.loads((ROOT / spec['capsule']).read_text())
    pins = capsule[spec['pins_key']]
    assert pins == spec['captured_pins']
    if spec['after_key'] is not None:
        assert capsule[spec['after_key']] == pins
    assert capsule[spec['unchanged_key']]
    runtime = capsule if spec['runtime_key'] is None else capsule[spec['runtime_key']]
    assert runtime[spec['returncode_key']] == spec['returncode']
    assert runtime['natural_exit']
    files = spec['artifact_hashes']
    for name, expected in files.items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    pre_name = capsule['preexecution']['path']
    exe_name = capsule['executed']['path']
    live_differences = []
    with zipfile.ZipFile(ROOT / pre_name) as before, zipfile.ZipFile(ROOT / exe_name) as after:
        names = before.namelist()
        assert len(names) == len(set(names))
        assert len(after.namelist()) == len(set(after.namelist()))
        assert set(names) == set(spec['input_members'])
        assert set(after.namelist()) == set(names) | set(spec['output_members'])
        assert set(pins) <= set(spec['input_members'].values())
        for member, source in spec['input_members'].items():
            raw = before.read(member)
            assert raw == after.read(member), member
            expected = pins[source] if source in pins else spec['extra_input_hashes'][source]
            assert sha(raw) == expected, member
            live = sha((ROOT / source).read_bytes())
            if live != expected:
                live_differences.append({'source': source, 'captured': expected, 'live': live})
        for member, source in spec['output_members'].items():
            raw = after.read(member)
            assert raw == (ROOT / source).read_bytes(), member
            assert sha(raw) == files[source], member
    exit_data = json.loads((ROOT / spec['exit_path']).read_text())
    assert exit_data[spec['exit_returncode_key']] == spec['returncode']
    assert exit_data['natural_exit']
    if spec['returncode'] == 0:
        assert (ROOT / spec['stderr_path']).read_bytes() == b''
    else:
        assert (ROOT / spec['stderr_path']).read_bytes() != b''
    return {'epoch': spec['epoch'], 'pre_members_verified': len(spec['input_members']),
            'natural_exit': True, 'returncode': spec['returncode'],
            'elapsed_seconds': runtime['elapsed_seconds'],
            'historical_live_differences_not_rebound': live_differences,
            'capsule_sha256': sha((ROOT / spec['capsule']).read_bytes())}


def power2(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def energy_constants() -> dict[str, Any]:
    chi, lam, field = power2(-4096), power2(8192), power2(10240)
    trial = 60 * chi + 3000 * lam + Fraction(129, 10) * field
    bs, c4, tau, radius = power2(10244), power2(2053), power2(-10252), power2(8196)
    assert trial + 8192 * lam < bs
    assert 2 * bs / lam == c4
    a = Fraction(225) * power2(-14341)
    assert a == Fraction(225) * 4 * chi / (8 * bs)
    assert radius**2 >= 6 * c4 / a
    assert radius**4 >= 4 * c4
    retained = a - 3 * c4 / radius**2
    assert retained >= a / 2
    delta = Fraction(-126663739519795, 70368744177664)
    assert tau <= Fraction(1, 4) / radius
    assert tau <= abs(delta) * a / (5760 * chi)
    gain = abs(delta) * a * tau / 288 - 10 * chi * tau**2
    gamma, epsilon = power2(-24595), power2(-49195)
    assert gain >= abs(delta) * a * tau / 576 > gamma
    coefficient = Fraction(16) / (delta**2 * Fraction(64, 255) * 256)
    assert abs(delta) < Fraction(9, 5) and delta**2 < Fraction(255, 64)
    assert coefficient > Fraction(1, 16)
    assert min(Fraction(1, 2), coefficient * gamma**2) > epsilon
    wrong_delta = Fraction(1999, 1000)
    wrong_coefficient = Fraction(16) / (wrong_delta**2 * Fraction(64, 255) * 256)
    assert wrong_delta < 2 and wrong_coefficient < Fraction(1, 16)
    assert Fraction(255, 4096) < Fraction(1, 16)
    assert Fraction(255, 4096) * gamma**2 > epsilon
    return {'scope': 'EXACT_ARITHMETIC_OF_CONDITIONAL_ORIGINAL_H_FORM_COMPARISON',
            'own_constrained_BS_exponent': 10244, 'own_constrained_C4_exponent': 2053,
            'tau_exponent': -10252, 'radius_exponent': 8196,
            'gamma_per_site_exponent': -24595, 'dimensionless_complement_exponent': -49195,
            'delta': str(delta), 'exact_delta_stronger_rationale_verified': True,
            'wrong_delta_less_than_two_rationale_rejected': True,
            'own_constrained_budget_not_actual_ground_B0': True,
            'global_N_dependent_complement_not_local_loop_Gram': True}


def sign_and_commutator() -> dict[str, Any]:
    # Declared nonnative matrix control. The whole degenerate zero block is retained.
    ground = sp.Matrix([1, 0, 0])
    p0 = sp.diag(1, 1, 0)
    q0 = sp.eye(3) - p0
    n = sp.Matrix([[0, sp.I, 1 + 2 * sp.I], [-sp.I, 0, 0], [1 - 2 * sp.I, 0, 0]])
    a = sp.Matrix([[0, 1, 2 - sp.I], [1, 0, 0], [2 + sp.I, 0, 0]])
    def mean(x: Any) -> Any:
        return sp.simplify((ground.conjugate().T * x * ground)[0])
    z, c, p = mean(n * q0 * a), sp.I * mean(n * a - a * n), sp.I * mean(n * p0 * a - a * p0 * n)
    assert z == 5 * sp.I and c == -12 and p == -2
    assert sp.im(z) == -(c - p) / 2
    assert sp.im(z) != (c - p) / 2
    assert sp.im(z)**2 == (c - p)**2 / 4
    # A Q-only source energy-Riesz lower uses the same ground block, not mean removal.
    resolvent = sp.diag(0, 0, sp.Rational(1, 10))
    response = mean(a * resolvent * a)
    lower = sp.simplify((c - p)**2 / (4 * mean(n * (sp.diag(0, 0, 7) + 3 * q0) * n)))
    assert response == lower == sp.Rational(1, 2)
    h = sp.Matrix([[0, 1], [1, 0]])
    d = sp.diag(1, 0)
    psi = sp.Matrix([0, 1])
    assert h * d * psi == sp.zeros(2, 1)
    comm = (h * d - d * h) * psi
    assert comm == sp.Matrix([-1, 0])
    assert (comm.conjugate().T * comm)[0] == 1
    assert h**2 == sp.eye(2) and d**2 == d
    return {'scope': 'EXACT_NONNATIVE_SIGN_AND_OPERATOR_NORM_COUNTERCONTROLS',
            'z': str(z), 'c': str(c), 'whole_P0_counterterm': str(p),
            'correct_Im_z_negative_half_c_minus_p': True,
            'wrong_positive_half_sign_rejected': True,
            'squared_signed_lower_unchanged': str(lower),
            'wrong_vector_commutator_bound_rejected': True,
            'pointwise_product_norm_bound': '2||Hlocal(q)|| ||D_M(q)||',
            'native_ground_computed': False}


def occupation_and_domain(request: dict[str, Any]) -> dict[str, Any]:
    with zipfile.ZipFile(ROOT / request['profile_archive']) as archive:
        raw = archive.read(request['profile_member'])
    assert sha(raw) == request['profile_member_sha256']
    profiles = json.loads(raw)['primitive_profiles']
    neutral, active, auxiliary = profiles
    assert [p['load'] for p in profiles] == [1, 1, 63]
    assert active['paths'] == auxiliary['paths']
    assert [(p['axis'], p['length']) for p in active['paths']] == [
        (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)]
    assert len(neutral['paths']) == 3
    delta = Fraction(active['onsite']) - Fraction(auxiliary['onsite'])
    assert delta == Fraction(-126663739519795, 70368744177664)
    c0 = Fraction(active['paths'][0]['c0'])
    assert c0 == Fraction(-1080863910568919, 140737488355328)
    a4 = c0**4 / 3072
    assert a4 > 0
    assert 24 * 64**4 == 402653184
    assert 12**2 == 144
    m = 256
    c = Fraction(m, 4 * (m - 1))
    assert c == Fraction(64, 255)
    # Every old supported branch has at most three auxiliary particles in this mode;
    # every vector in det^31 wedge has eigenvalues 31/32. Their spectra are disjoint.
    assert set(range(4)).isdisjoint({31, 32})
    exponent = Fraction(1, 34)
    beta = (1 - 17 * exponent) / 4
    assert beta == Fraction(1, 8)
    assert (1 - 17 * Fraction(1, 17)) / 4 == 0
    return {'native_a4': str(a4), 'full_ordered_copy_words': 402653184,
            'literal_all65_role_loads': [1, 1, 63],
            'all_six_charged_paths_identical_including_length2': True,
            'literal_active_auxiliary_onsite_difference': str(delta),
            'finite_N4_all_Ps_multiplicity_coefficient': str(c),
            'old_aux_face_empty_branch_orthogonal_det31_sector': True,
            'global_minimizing_auxiliary_sector_selected': False,
            'originating_occupation_mass_from_canonical_trace': False,
            'fractional_example_s': str(exponent), 'dyadic_tail_beta': str(beta),
            'strict_endpoint_excluded': True,
            'conditional_adjoint_compression_not_actual_ground_Gram': True}


def verify_typed_payloads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified = []
    for row in rows:
        original = json.loads((ROOT / row['original_receipt']).read_text())
        typed = json.loads((ROOT / row['typed_receipt']).read_text())
        excluded = set(row['excluded_metadata_fields'])
        assert excluded <= {'input_pins'}
        corrections = row['explicit_point_corrections']
        for correction in corrections:
            assert correction['path'] == ['adjoint_native', 'AP_seams']
            assert original['adjoint_native']['AP_seams'] == [-1, -1, 1, 1]
            assert typed['adjoint_native']['AP_seams'] == [-1, 1, -1, 1]
            independent = json.loads((ROOT / correction['independent_seam_receipt']).read_text())
            assert independent['adjoint_native'] == typed['adjoint_native']
            original['adjoint_native']['AP_seams'] = [-1, 1, -1, 1]
        original_payload = {k: v for k, v in original.items() if k not in excluded}
        typed_payload = {k: v for k, v in typed.items() if k not in excluded}
        assert original_payload == typed_payload, row['typed_receipt']
        verified.append({'original_receipt': row['original_receipt'],
                         'typed_receipt': row['typed_receipt'],
                         'excluded_metadata_fields': sorted(excluded),
                         'explicit_point_corrections': corrections,
                         'payload_exact_equal_after_declared_metadata_and_reviewed_point_boundary': True})
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, expected in request['pins'].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    epochs = [verify_epoch(spec) for spec in request['epochs']]
    result = {'kind': 'EXACT_COMMON_SINGLET_FULL64_NATIVE_LOOP_JOIN_NOT_PHASE_OR_IR_ATOM',
              'epochs': epochs, 'pre_members_verified': sum(e['pre_members_verified'] for e in epochs),
              'energy_comparison_constants': energy_constants(),
              'sign_and_commutator_controls': sign_and_commutator(),
              'native_occupation_and_domain': occupation_and_domain(request),
              'typed_payload_transfers': verify_typed_payloads(request['typed_payload_transfers']),
              'originals_modified_or_rebound': False,
              'unconditional_machine_attestation': False,
              'native_ground_or_fullH_numerical_evolution_computed': False,
              'actual_loop_wholeP0_Gram_or_phase_or_IR_atom': False,
              'original_SOURCE_adopted': False, 'campaign_complete': False,
              'input_pins': request['pins']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Exact energy comparison, sign/product-norm controls and immutable native loop join completed')


if __name__ == '__main__':
    main()

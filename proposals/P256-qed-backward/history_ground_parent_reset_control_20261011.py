"""Exact native SU4 carrier/reset controls; not a full-field channel or ground proof."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def generators() -> list[sp.MutableDenseMatrix]:
    out: list[sp.MutableDenseMatrix] = []
    for i in range(4):
        for j in range(i + 1, 4):
            real, imag = sp.zeros(4), sp.zeros(4)
            real[i, j] = real[j, i] = sp.Rational(1, 2)
            imag[i, j], imag[j, i] = -sp.I / 2, sp.I / 2
            out.extend((real, imag))
    for k in range(1, 4):
        diag = [1] * k + [-k] + [0] * (3 - k)
        out.append(cast(sp.MutableDenseMatrix, sp.diag(*diag) / sp.sqrt(2 * k * (k + 1))))
    assert len(out) == 15
    for i, a in enumerate(out):
        assert a.H == a and sp.trace(a) == 0
        for j, b in enumerate(out):
            assert sp.simplify(sp.trace(a * b)) == (sp.Rational(1, 2) if i == j else 0)
    return out


def full_car_representations(ts: list[sp.MutableDenseMatrix]) -> dict[str, Any]:
    annihilators: list[sp.MutableDenseMatrix] = []
    for i in range(4):
        c = sp.zeros(16)
        for mask in range(16):
            if mask & (1 << i):
                c[mask ^ (1 << i), mask] = (-1) ** ((mask & ((1 << i) - 1)).bit_count())
        annihilators.append(c)
    for i, a in enumerate(annihilators):
        for j, b in enumerate(annihilators):
            assert a * b.H + b.H * a == (sp.eye(16) if i == j else sp.zeros(16))
    reps = []
    for t in ts:
        rep = sp.zeros(16)
        for i in range(4):
            for j in range(4):
                rep += t[i, j] * annihilators[i].H * annihilators[j]
        reps.append(rep)
    expected = [0, sp.Rational(15, 8), sp.Rational(5, 2), sp.Rational(15, 8), 0]
    rows = []
    for number in range(5):
        indices = [mask for mask in range(16) if mask.bit_count() == number]
        selected = [r.extract(indices, indices) for r in reps]
        casimir = sum((r * r for r in selected), sp.zeros(len(indices)))
        assert casimir == expected[number] * sp.eye(len(indices))
        rows.append({'number': number, 'dimension': len(indices),
                     'Casimir_exact': str(expected[number]), 'centre_character': str(sp.I ** number)})
    # This is an operator triangle bound for labelled species, not a sea expectation.
    collective_bound = Fraction(5, 2) * 65 ** 2
    wrong_collapsed_bound = Fraction(5, 2) * 2 ** 2
    assert collective_bound > wrong_collapsed_bound
    return {'full_Fock_dimension': 16, 'all15_generators': True, 'sectors': rows,
            'all65_collective_Casimir_triangle_upper_exact': str(collective_bound),
            'collapsed_heavy_plus_charged_copy_bound_rejected': True,
            'minimum_centre_carriers': ['1', '4', '6', '4bar'],
            'full_lattice_Gauss_budget_proved': False}


def isotypic_reset(ts: list[sp.MutableDenseMatrix]) -> dict[str, Any]:
    ident = sp.eye(4)
    gauss = [sp.kronecker_product(t, ident) - sp.kronecker_product(ident, t.T) for t in ts]
    singlet = sp.zeros(16, 1)
    for i in range(4):
        singlet[4 * i + i, 0] = sp.Rational(1, 2)
    assert (singlet.H * singlet)[0, 0] == 1
    assert all(g * singlet == sp.zeros(16, 1) for g in gauss)
    low, high = sp.Matrix([1, 0]), sp.Matrix([0, 1])
    source = cast(sp.MutableDenseMatrix, sp.kronecker_product(singlet, high))
    keep = cast(sp.MutableDenseMatrix, sp.kronecker_product(sp.eye(16), low * low.T))
    reset = cast(sp.MutableDenseMatrix, sp.kronecker_product(sp.eye(16), low * high.T))
    assert keep.H * keep + reset.H * reset == sp.eye(32)
    for g in gauss:
        full = sp.kronecker_product(g, sp.eye(2))
        assert full * reset == reset * full and full * keep == keep * full
        assert full * (reset * source) == sp.zeros(32, 1)
    assert (source.H * reset.H * reset * source)[0, 0] == 1
    # Carrier depolarization is SU4-conjugation covariant but destroys annihilation Gauss.
    depolarized = sp.eye(16) / 16
    gauss_square = sum((g * g for g in gauss), sp.zeros(16))
    loss = sp.simplify(sp.trace(depolarized * gauss_square))
    assert loss == sp.Rational(15, 4)
    wrong_carrier_reset = sp.zeros(4)
    wrong_carrier_reset[0, 1] = 1
    wrong_full = sp.kronecker_product(wrong_carrier_reset, ident)
    assert any(g * wrong_full != wrong_full * g for g in gauss)
    # The link carrier has left centre i and right centre -i, not two copies of i.
    correct_joint_centre = sp.I * (-sp.I)
    wrong_joint_centre = sp.I * sp.I
    assert correct_joint_centre == 1 and wrong_joint_centre != 1
    return {'SU4_fundamental_and_conjugate_exact_singlet': True,
            'multiplicity_channel_trace_preserving': True,
            'all15_Kraus_commute_annihilation_Gauss': True,
            'covariance_only_reset_Gauss_square_exact': str(loss),
            'wrong_carrier_reset_rejected': True, 'wrong_right_link_centre_rejected': True,
            'scope': 'exact native representation factors with a two-dimensional multiplicity; not the full Phi channel domain'}


def free_colour_orbit(ts: list[sp.MutableDenseMatrix]) -> dict[str, Any]:
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.diag(1, -1)
    gamma = [sp.kronecker_product(sx, s) for s in (sx, sy, sz)]
    gamma.append(sp.kronecker_product(sy, sp.eye(2)))
    for i, a in enumerate(gamma):
        for j, b in enumerate(gamma):
            assert a * b + b * a == (2 * sp.eye(4) if i == j else sp.zeros(4))
    jets = [[sp.I * (t * g - g * t) for g in gamma] for t in ts]
    gram = sp.zeros(15)
    for i, row in enumerate(jets):
        for j, other in enumerate(jets):
            gram[i, j] = sp.simplify(sp.Add(*(sp.trace(a.H * b) for a, b in zip(row, other))))
    # Exact LDL is a positivity proof, not a reused native eigensolver.
    lower, diagonal = gram.LDLdecomposition(hermitian=False)
    assert lower * diagonal * lower.T == gram
    assert all(diagonal[i, i] > 0 for i in range(15))
    charpoly = gram.charpoly().as_expr()
    # A commutant of the four Clifford matrices is scalar, hence SU4 stabilizer is Z4.
    symbols = sp.symbols('z0:16')
    generic = sp.Matrix(4, 4, symbols)
    equations = []
    for g in gamma:
        equations.extend(list(generic * g - g * generic))
    coeff, _ = sp.linear_eq_to_matrix(equations, symbols)
    kernel = coeff.nullspace()
    assert len(kernel) == 1
    assert sp.Matrix(4, 4, kernel[0]) == sp.eye(4)
    return {'native_HS_generator_normalization': 'Tr(TA TB)=deltaAB/2',
            'orbit_Gram_exact': [[str(x) for x in gram.row(i)] for i in range(15)],
            'LDL_pivots_exact': [str(diagonal[i, i]) for i in range(15)],
            'characteristic_polynomial': str(charpoly),
            'commutant_scalar_exact': True, 'stabilizer': 'Z4',
            'all_centre_zero_Peter_Weyl_carriers_available_on_orbit': 'analytical consequence; full tube/domain cost not certified here',
            'orbit_is_trial_geometry_not_adopted_root_or_ground': True}


def variational_tail_gate() -> dict[str, Any]:
    # Explicit exponent arithmetic for the proposed reset estimate; no actual ground evaluated.
    r = sp.symbols('r', positive=True)
    reset_decay = r - sp.Rational(2, 3)
    near_boundary_decay = r - sp.Rational(1, 2)
    assert sp.simplify(near_boundary_decay - reset_decay) == sp.Rational(1, 6)
    bootstrap = reset_decay + 1
    assert sp.simplify(bootstrap - r) == sp.Rational(1, 3)
    rows = [{'input_moment_order': str(Fraction(1) + Fraction(i, 4)),
             'hypothetical_next_strict_order': str(Fraction(1) + Fraction(i, 4) + Fraction(1, 3))}
            for i in range(9)]
    return {'reset_energy_power_Casimir': '2/3', 'first_uniform_tail_decay_if_all_costs_paid': 'M^-1/3',
            'hypothetical_bootstrap_increment': '1/3 (strict, not attained endpoint)',
            'exponent_rows': rows,
            'complete_native_CP_form_boundary_and_two_threshold_estimates_paid': False,
            'actual_ground_high_moment_theorem': False,
            'moment_bounds_imply_static_score_locality': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request: dict[str, Any] = json.loads(args.request.read_text())
    before = {n: digest(ROOT / n) for n in request['pins']}
    assert before == request['pins']
    ts = generators()
    with zipfile.ZipFile(ROOT / request['profile_archive']) as z:
        profile_raw = z.read(request['profile_member'])
    profiles = json.loads(profile_raw)['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    payload = {'kind': 'EXACT_NATIVE_SU4_GAUSS_CARRIER_RESET_AND_ORBIT_CONTROL_NOT_FULL_CHANNEL_OR_GROUND',
               'canonical_CAR': full_car_representations(ts), 'Gauss_reset': isotypic_reset(ts),
               'trial_colour_orbit': free_colour_orbit(ts), 'proposed_tail_gate': variational_tail_gate(),
               'primitive_roles': [1, 1, 63], 'profile_member_sha256': hashlib.sha256(profile_raw).hexdigest(),
               'complete_H_channel_or_domain_proved': False, 'actual_minimizing_ground_computed': False,
               'same_static_GNS_exterior_range_proved': False, 'phase_stiffness_IRatom': False,
               'original_SOURCE_adopted': False, 'campaign_complete': False,
               'source_input_sha256_before': before,
               'source_input_sha256_after': {n: digest(ROOT / n) for n in before},
               'runtime': {'elapsed_seconds': time.monotonic() - started}}
    assert payload['source_input_sha256_before'] == payload['source_input_sha256_after']
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()

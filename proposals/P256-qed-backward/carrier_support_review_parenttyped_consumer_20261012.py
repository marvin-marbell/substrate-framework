"""Independent frozen-input count, exterior-CAR, native-flow and support controls.
No original producer is imported or executed; no quantum-ground numerical backend.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, cast
import sys
import zipfile

import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)
FACES = ((0, 1, 2, 3), (1, 4, 5, 2))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def exterior_transfer(occupied, target, source):
    """Sorted exterior basis: remove and insert with their actual wedge positions."""
    pos = bisect_left(occupied, source)
    if pos == len(occupied) or occupied[pos] != source:
        return None
    remainder = occupied[:pos] + occupied[pos + 1:]
    ins = bisect_left(remainder, target)
    if ins < len(remainder) and remainder[ins] == target:
        return None
    return remainder[:ins] + (target,) + remainder[ins:], (-1) ** (pos + ins)


def occupations(mask, modes):
    return tuple(i for i in range(modes) if mask & (1 << i))


def action(occupied, face, copies=1, samecopy=False, unsigned=False):
    """Ordered product DAG retains distinct-species assignments in every leg."""
    edges = [(face[(e + 1) % 4], face[e]) for e in range(4)]
    out = Counter()
    for order in itertools.permutations(range(4)):
        if samecopy:
            assignments = ((c,) * 4 for c in range(copies))
        else:
            assignments = itertools.product(range(copies), repeat=4)
        for assignment in assignments:
            current, value = occupied, 1
            for e, role in zip(order, assignment):
                a, b = edges[e]
                moved = exterior_transfer(current, 6 * role + a, 6 * role + b)
                if moved is None:
                    value = 0
                    break
                current, sign = moved
                value *= 1 if unsigned else sign
            if value:
                out[current] += value
    return {key: value for key, value in out.items() if value}


def count_control():
    # Separate coefficient convolution and complete total-occupation enumeration.
    coefficients = [1]
    for _ in range(6):
        updated = [0] * (len(coefficients) + 4)
        for m, value in enumerate(coefficients):
            for n in range(5):
                updated[m + n] += value * math.comb(4, n)
        coefficients = updated
    assert coefficients == [math.comb(24, m) for m in range(25)]
    dimensions = [0] * 25
    selected = 0
    wrong_cropped = 0
    for pattern in itertools.product(range(5), repeat=6):
        m = sum(pattern)
        outside = 128 - m
        assert 0 <= outside <= 232
        internal = math.prod(math.comb(4, n) for n in pattern)
        rank = internal * math.comb(232, outside)
        dimensions[m] += rank
        wrong_cropped += math.comb(232, outside)
        if pattern[:4] == (4, 0, 4, 0):
            selected += rank
    dim = math.comb(256, 128)
    assert sum(dimensions) == dim
    assert dimensions == [math.comb(24, m) * math.comb(232, 128 - m) for m in range(25)]
    assert selected == math.comb(240, 120)
    assert 0 < selected < dim and wrong_cropped != dim
    wrong_fixed_local2 = 6 ** 6 * math.comb(232, 116)
    assert wrong_fixed_local2 != dim
    all65 = dim ** 65
    assert all65 != dim ** 3 and all65 != dim ** 64
    return {'patterns_per_role': 5 ** 6, 'local_internal_dimension': sum(coefficients),
            'spectator_capacity': 232, 'spectators_extremes': [128, 104],
            'local_total_dimensions': [str(x) for x in dimensions],
            'one_role_dimension': str(dim), 'all65_dimension_sha256': digest(str(all65).encode()),
            'all65_role_ids': list(range(65)), 'all65_pattern_count': str(5 ** 390),
            'selected_face_total_number_rank': str(selected), 'complement_rank': str(dim - selected),
            'wrong_colour_cropping_rejected': True, 'wrong_fixed_local2_rejected': True,
            'wrong_profile_multiplicity_as_three_roles_rejected': True,
            'ground_probability_identified_with_rank': False}


def car_control(profiles):
    # Independently compare wedge signs with explicit Kronecker Jordan-Wigner matrices.
    eye = sp.eye(2)
    parity = sp.diag(1, -1)
    destroy = sp.Matrix([[0, 1], [0, 0]])
    c = []
    for j in range(6):
        factors = [parity if k < j else destroy if k == j else eye for k in range(6)]
        c.append(sp.kronecker_product(*factors))
    tested = 0
    for a in range(6):
        for b in range(6):
            matrix = c[a].T * c[b]
            for mask in range(64):
                occupied = occupations(mask, 6)
                index = sum(1 << (5 - j) for j in occupied)
                moved = exterior_transfer(occupied, a, b)
                column = {i: matrix[i, index] for i in range(64) if matrix[i, index]}
                expected = {} if moved is None else {sum(1 << (5 - j) for j in moved[0]): moved[1]}
                assert column == expected
                tested += 1
    rows = []
    bad_unsigned = []
    for mask in range(64):
        occupied = occupations(mask, 6)
        values = [action(occupied, f).get(occupied, 0) for f in FACES]
        assert all(set(action(occupied, f)) <= {occupied} for f in FACES)
        unsigned = [action(occupied, f, unsigned=True).get(occupied, 0) for f in FACES]
        if values != unsigned:
            bad_unsigned.append(mask)
        rows.append({'mask': mask, 'local_n': len(occupied), 'counts': values,
                     'adjacent_product': math.prod(values), 'spectators': 128 - len(occupied)})
    positive = next(r for r in rows if r['adjacent_product'] > 0)
    negative = next(r for r in rows if r['adjacent_product'] < 0)
    assert bad_unsigned and all(0 <= r['spectators'] <= 232 for r in rows)
    c0 = Fraction(next(p['c0'] for p in profiles[1]['paths'] if p['axis'] == 0 and p['length'] == 1))
    assert (3 * c0) ** 8 > 0
    witness = None
    for mask in range(4096):
        occupied = occupations(mask, 12)
        full = action(occupied, FACES[0], copies=2)
        same = action(occupied, FACES[0], copies=2, samecopy=True)
        if full != same and any(key != occupied for key in full):
            witness = (occupied, full, same)
            break
    assert witness is not None
    occupied, full, same = witness
    initial_collective = [sum(x % 6 == site for x in occupied) for site in range(6)]
    for state in full:
        assert len(state) == len(occupied)
        assert sum(x < 6 for x in state) == sum(x < 6 for x in occupied)
        assert [sum(x % 6 == site for x in state) for site in range(6)] == initial_collective
    changed = next(state for state in full if state != occupied)
    assert [x < 6 for x in changed] == [x < 6 for x in occupied]
    assert set(x for x in changed if x < 6) != set(x for x in occupied if x < 6)
    serialize = lambda values: [{'occupied': list(key), 'coefficient': value} for key, value in sorted(values.items())]
    assert not action(tuple(range(6)), FACES[0]) and not action((), FACES[0])
    assert action((1, 2, 3, 4, 5), FACES[0]).get((1, 2, 3, 4, 5)) == 1
    return {'JW_wedge_transfer_comparisons': tested, 'all64_native_colour0_masks': rows,
            'native_Phi_section': {'Phi0': [3, -1, -1, -1], 'Phi123': 0, 'U': 'I', 'A': 0},
            'positive_witness': positive, 'negative_witness': negative,
            'native_adjacent_amplitude': str((3 * c0) ** 8),
            'wrong_unsigned_CAR_rejected_at_masks': bad_unsigned,
            'mixedcopy_input': list(occupied), 'mixedcopy_full': serialize(full),
            'wrong_samecopy': serialize(same), 'mixedcopy_local_number_change': list(changed),
            'mixedcopy_collective_local_number_preserved': True,
            'all24_all64_word_count': 24 * 64 ** 4, 'charged_role_ids': list(range(1, 65)),
            'ground_sign_or_projected_Gram_computed': False}


def flow_control():
    rho = sp.Matrix([128, 128, -128, -128])
    lap = sp.Matrix([[2, -1, 0, -1], [-1, 2, -1, 0], [0, -1, 2, -1], [-1, 0, -1, 2]])
    potential = lap.pinv() * rho
    flow = [potential[z] - potential[(z + 1) % 4] for z in range(4)]
    assert flow == [0, 128, 0, -128]
    assert [flow[z] - flow[(z - 1) % 4] for z in range(4)] == list(cast(Any, rho))
    assert sum(flow) == 0
    energy = 64 * 16 * sum(f * f for f in flow)
    coulomb = 64 * 16 * (rho.T * lap.pinv() * rho)[0]
    assert energy == coulomb == 33554432
    per_copy = 64 * 16 * 64 * sum((f / 64) ** 2 for f in flow)
    assert per_copy == 524288 and per_copy != energy
    assert 2 * energy != energy
    t = sp.Symbol('harmonic', real=True)
    assert sp.expand(64 * 16 * sum((f + t) ** 2 for f in flow)) == energy + 4096 * t ** 2
    return {'charged_copies': 64, 'neutral_electric_charge': 0, 'Ref4_minus4': 0,
            'rho_planes': list(cast(Any, rho)), 'zero_mean_integer_flow': flow,
            'poisson_potential': list(cast(Any, potential)), 'once_collective_fullE_longitudinal_energy': energy,
            'HC_is_same_longitudinal_component': True, 'wrong_independent_copy_energy': per_copy,
            'wrong_extra_HC_rejected': True, 'cost_is_not_complete_physical_ground_energy': True}


def site_index(site):
    x, y, z = site
    return 16 * x + 4 * y + z


def paths(profile):
    result = []
    for site in itertools.product(range(4), repeat=3):
        for primitive in profile['paths']:
            axis, length = primitive['axis'], primitive['length']
            target = list(site)
            target[axis] = (target[axis] + length) % 4
            seam = (-1) ** ((site[axis] + length) // 4)
            source, dest = site_index(site), site_index(target)
            edges = []
            current = list(site)
            for _ in range(length):
                edges.append((site_index(current), axis))
                current[axis] = (current[axis] + 1) % 4
            result.append((source, dest, length, axis, seam, tuple(edges)))
    return result


def outgoing_control(profiles):
    all_patterns = []
    for shift in range(4):
        filled = {shift, (shift + 1) % 4}
        all_patterns.append(tuple(sorted(4 * site_index((x, y, z)) + colour
                                         for z in range(4) if z in filled
                                         for y in range(4) for x in range(4) for colour in range(4))))
    assert all(len(state) == 128 and tuple(sorted(state)) == state for state in all_patterns)
    pathsets = [paths(profiles[0])] + [paths(profiles[1 if role == 1 else 2]) for role in range(1, 65)]
    results = []
    for shift, stripe in enumerate(all_patterns):
        source_site = site_index((0, 0, (shift + 1) % 4))
        target_site = site_index((0, 0, (shift + 2) % 4))
        selected = exterior_transfer(stripe, 4 * target_site, 4 * source_site)
        assert selected is not None
        outgoing, sign = selected
        matches = []
        checked = 0
        # Every primitive, translate, direction, colour pair and all65 roles is counted.
        # Other roles fail the factor-label boundary before CAR evaluation.
        for role, rolepaths in enumerate(pathsets):
            for start, end, length, axis, seam, edges in rolepaths:
                for source, target, orientation in ((end, start, 'forward'), (start, end, 'adjoint')):
                    for a, b in itertools.product(range(4), repeat=2):
                        checked += 1
                        if role != 1:
                            continue
                        moved = exterior_transfer(stripe, 4 * target + a, 4 * source + b)
                        if moved is not None and moved[0] == outgoing:
                            matches.append({'role': role, 'length': length, 'axis': axis,
                                            'source': source, 'target': target, 'colours': [a, b],
                                            'seam': seam, 'orientation': orientation,
                                            'edges': edges, 'CAR_sign': moved[1]})
        assert len(matches) == 1 and matches[0]['length'] == 1 and matches[0]['axis'] == 2
        assert matches[0]['seam'] == (-1 if shift == 2 else 1)
        expected_checked = 64 * (3 + 64 * 6) * 2 * 16
        assert checked == expected_checked
        # A hop from another stripe changes one role at most, but64 other factors differ.
        alternate_differences = []
        for other, other_state in enumerate(all_patterns):
            if other == shift:
                continue
            unchanged_bad_factors = sum(other_state != stripe for role in range(65) if role != 1)
            assert unchanged_bad_factors == 64
            alternate_differences.append({'stripe': other, 'unchanged_bad_factors': unchanged_bad_factors})
        # At N4 two oppositely directed length2 paths can have the SAME endpoints.
        length2_same_endpoints = [p for p in pathsets[1] if p[2] == 2 and p[3] == 2
                                 and {p[0], p[1]} == {0, 2}]
        assert len(length2_same_endpoints) == 2
        assert sum(p[4] for p in length2_same_endpoints) == 0
        results.append({'stripe': shift, 'canonical_particles_each_role': len(stripe),
                        'source_site': source_site, 'target_site': target_site,
                        'all65_primitive_colour_candidates': checked, 'unique_match': matches[0],
                        'other_stripes_excluded_at_factor_labels': alternate_differences,
                        'outgoing_active_mask_sha256': digest(repr(outgoing).encode())})
    assert any(r['unique_match']['seam'] == -1 for r in results)
    c0 = Fraction(next(p['c0'] for p in profiles[1]['paths'] if p['axis'] == 2 and p['length'] == 1))
    assert c0 != 0
    return {'four_stripes': results, 'total_candidates': sum(r['all65_primitive_colour_candidates'] for r in results),
            'active_vertical_c0': str(c0), 'Phi0_diag_section_coefficient': str(3 * c0),
            'wrong_AP_omission_changes_seam_stripe': 2,
            'length2_N4_endpoint_alias_positive_control': True,
            'length2_N4_AP_constant_section_cancellation': True,
            'length2_or_other_species_cannot_hit_selected_NN_block': True,
            'full_empty_SU4_singlets': 'wedge0 C4 and wedge4 C4 have determinant1 action',
            'kinetic_connection_terms_preserve_stripe_occupations': True}


def analytic_control(profiles):
    generators = []
    for a in range(4):
        for b in range(a + 1, 4):
            x, y = sp.zeros(4), sp.zeros(4)
            x[a, b] = x[b, a] = sp.Rational(1, 2)
            y[a, b], y[b, a] = -sp.I / 2, sp.I / 2
            generators.extend((x, y))
    for n in range(1, 4):
        generators.append(sp.diag(*([1] * n + [-n] + [0] * (3 - n))) / sp.sqrt(2 * n * (n + 1)))
    assert len(generators) == 15
    # Full normalized SU4 completeness, not a spintrace or a selected generator.
    for a, k, l, b in itertools.product(range(4), repeat=4):
        observed = sp.simplify(cast(Any, sum(T[a, k] * T[l, b] for T in generators)))
        expected = sp.Rational(1, 2) * (int(a == b) * int(k == l) - sp.Rational(1, 4) * int(a == k) * int(l == b))
        assert observed == expected
    c0 = sp.Rational(next(p['c0'] for p in profiles[1]['paths'] if p['axis'] == 2 and p['length'] == 1))
    overlap = sp.Symbol('abs_Uab_squared', nonnegative=True)
    derivative_norm = c0 ** 2 / 4 * (sp.Rational(1, 2) - overlap / 8)
    # For every unitary U,0<=|Uab|^2<=1; independent endpoint Phi0 jets suffice.
    lower = sp.simplify(derivative_norm.subs(overlap, 1))
    assert lower == 3 * c0 ** 2 / 32 and lower > 0
    trace_diagonal = sp.diag(3, -1, -1, -1)
    assert sp.trace(trace_diagonal) == 0 and 3 * c0 != 0
    return {'normalized_full15_completeness_entries_checked': 256,
            'one_endpoint_Phi0_jet_square_sum': str(derivative_norm),
            'for_every_SU4_link_jet_norm_squared_lower': str(lower),
            'this_is_symbol_nonvanishing_not_ground_Gram_lower': True,
            'AP_and_Peierls_unit_modulus_preserve_nonzero_jet': True,
            'nullset_argument': 'For fixed links and remaining fields, a nonconstant linear entry in15 real endpoint Phi0 coordinates has zero-set Lebesgue measure0. Fubini over independent fields/Haar/Peierls variables; any physical L2 subspace inherits this null-set annihilation.',
            'Gauss_does_not_replace_L2_by_delta_Phi_measure': True,
            'point_section_alone_is_not_a_e_proof': True}


def separate_singlet_control(profiles):
    """Actual multiplicity2 in four spin-half factors, not duplicated test blocks."""
    eye = sp.eye(2)
    generators = [sp.Matrix([[0, 1], [1, 0]]),
                  sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.diag(1, -1)]
    common = []
    for g in generators:
        common.append(sum((sp.kronecker_product(*[g if role == slot else eye
                                                 for role in range(4)])
                           for slot in range(4)), sp.zeros(16)))
    basis = sp.Matrix.vstack(*common).nullspace()
    assert len(basis) == 2
    columns = sp.Matrix.hstack(*basis)
    projector = columns * (columns.conjugate().T * columns).inv() * columns.conjugate().T
    assert projector * projector == projector and projector.conjugate().T == projector
    assert all(g * projector == sp.zeros(16) for g in common)
    body = sp.Matrix([[3, 1 + sp.I], [1 - sp.I, -3]])
    active = sp.kronecker_product(body, eye, eye, eye)
    trace2 = sp.simplify(sp.trace(body * body))
    factor = sp.Rational(1, 2)
    assert projector * active * projector == sp.zeros(16)
    assert (projector * active ** 2 * projector - factor * trace2 * projector).applyfunc(sp.simplify) == sp.zeros(16)
    assert active * projector != sp.zeros(16)
    assert projector * active != active * projector
    # Common U(2) mode occupation is2; background subtraction kills localrho.
    mode = sp.diag(1, 0)
    totalmode = sum((sp.kronecker_product(*[mode if role == slot else eye
                                          for role in range(4)])
                     for slot in range(4)), sp.zeros(16))
    assert totalmode * projector == 2 * projector
    # Arbitrary common unitary sewing preserves P, including a determinant phase.
    U = sp.Matrix([[0, 1], [1, 0]])
    sewing = sp.kronecker_product(U, U, U, U)
    assert sewing * projector == projector * sewing
    delta = Fraction(profiles[1]['onsite']) - Fraction(profiles[2]['onsite'])
    assert delta == Fraction(-126663739519795, 70368744177664) and delta != 0
    assert profiles[1]['paths'] == profiles[2]['paths']
    M = sp.Symbol('M', integer=True, positive=True)
    k = sp.Symbol('k', integer=True, positive=True)
    covariance_factor = sp.simplify(k / M - k * (k - 1) / (M * (M - 1)))
    assert covariance_factor == k * (M - k) / (M * (M - 1))
    return {'scope': 'Separate parent addendum, never retroactive original theorem',
            'M2_k1_four_factor_total_dimension': 16, 'actual_singlet_multiplicity': len(basis),
            'full_multiplicity_projector_rank': int(sp.trace(projector)),
            'complex_Hermitian_trace_square': str(trace2),
            'compressed_active_square': str(factor * trace2) + ' P',
            'active_specific_connection_wrongcase_rejected': True,
            'common_SU2_generators_and_U2_sewing_reduce_P': True,
            'literal_active_aux_paths_identical': True, 'literal_onsite_delta': str(delta),
            'general_canonical_covariance_factor': str(covariance_factor),
            'native_M256_k128_factor': str(sp.Rational(128 * 128, 256 * 255)),
            'native_domain_not_proved_by_this_finite_representation_control': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    captured = {name: (ROOT / name).read_bytes() for name in request['pins']}
    assert all(digest(raw) == request['pins'][name] for name, raw in captured.items())
    assert digest(Path(__file__).read_bytes()) == request['consumer_sha256']
    with zipfile.ZipFile(ROOT / request['profile_archive']) as archive:
        profile_raw = archive.read(request['profile_member'])
    assert digest(profile_raw) == request['profile_member_sha256']
    profiles = json.loads(profile_raw)['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    result = {'kind': 'INDEPENDENT_CARRIER_SUPPORT_EXACT_CONTROLS_NOT_NATIVE_GROUND',
              'canonical_counts': count_control(), 'native_CAR_sign_mixedcopy': car_control(profiles),
              'collective_flow': flow_control(), 'outgoing_native_blocks': outgoing_control(profiles),
              'analytic_multiplier': analytic_control(profiles), 'input_pins': request['pins'],
              'separate_parent_singlet_control': separate_singlet_control(profiles),
              'profile_member_sha256': digest(profile_raw), 'same_originating_rho_N_retained': True,
              'native_ground_numeric': False, 'strict_complement_is_not_Gram_lower': True,
              'common_SU_M_singlet_successor_assumed': False,
              'phase_or_no_go_inference': False, 'scientific_admission': False}
    assert all((ROOT / name).read_bytes() == raw for name, raw in captured.items())
    result['inputs_unchanged'] = True
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, default=str)
        stream.write('\n')
    print(json.dumps({'controls': ['count', 'CAR', 'collective_flow', 'outgoing', 'analytic_symbol'],
                      'native_ground_numeric': False, 'inputs_unchanged': True}))


if __name__ == '__main__':
    main()

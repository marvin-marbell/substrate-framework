"""Independent exact trial-form controls; no historical helper or ground backend."""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import sys

import sympy as sp


def car_step(mask, target, source):
    if not mask & (1 << source):
        return None
    remainder = mask ^ (1 << source)
    if remainder & (1 << target):
        return None
    exponent = (mask % (1 << source)).bit_count() + (remainder % (1 << target)).bit_count()
    return remainder | (1 << target), (-1) ** exponent


def act(entries, state):
    result = defaultdict(lambda: sp.S.Zero)
    for mask, amplitude in state.items():
        for target, source, coefficient in entries:
            step = car_step(mask, target, source)
            if step is not None:
                destination, sign = step
                result[destination] += sign * coefficient * amplitude
    return {mask: sp.expand(value) for mask, value in result.items() if value != 0}


def squared_norm(state):
    return sp.simplify(sum(sp.conjugate(value) * value for value in state.values()))


def generators():
    result = []
    for a in range(4):
        for b in range(a + 1, 4):
            symmetric = sp.zeros(4)
            symmetric[a, b] = symmetric[b, a] = sp.Rational(1, 2)
            antisymmetric = sp.zeros(4)
            antisymmetric[a, b] = -sp.I / 2
            antisymmetric[b, a] = sp.I / 2
            result.extend([symmetric, antisymmetric])
    for r in range(1, 4):
        result.append(sp.diag(*([1] * r + [-r] + [0] * (3 - r))) / sp.sqrt(2 * r * (r + 1)))
    return result


def wick_and_sharp_price():
    gs = generators()
    casimir = sum((g * g for g in gs), sp.zeros(4))
    assert casimir == sp.Rational(15, 8) * sp.eye(4)
    assert all(sp.simplify(sp.trace(a * b)) == (sp.Rational(1, 2) if i == j else 0)
               for i, a in enumerate(gs) for j, b in enumerate(gs))
    for b in gs:
        assert sp.simplify(sum((a * b * a for a in gs), sp.zeros(4)) + b / 8) == sp.zeros(4)
    adjacent = sp.trace(casimir * casimir)
    alternating = -sum(sp.trace(g * g) for g in gs) / 8
    same = 4 * (2 * adjacent + alternating)
    different = 2 * (adjacent + alternating)
    jordan = 4 * (4 * same + 12 * different)
    assert jordan == 3000
    chi = Fraction(1, 2**4096)
    eta = Fraction(1, 2**7168)
    lam = 2**8192
    fast = 2**10240
    t = 2**7168
    terms = [60 * chi / eta, 3000 * lam * eta**2, Fraction(9, 2) * fast * eta,
             Fraction(27, 8) * chi * t,
             Fraction(7, 5) * Fraction(fast, t) * (450 + 48 * 7168 + 24)]
    upper = sum(terms)
    assert upper + 262144 < 2**3092 < lam
    assert Fraction(3, 20) * Fraction(1, 4) * 90 == Fraction(27, 8)
    return {'SU4_generators': len(gs), 'ordered_Jordan_Wick_coefficient': str(jordan),
            'same_mu_anticommutator_trace_over_eta2': str(same),
            'different_mu_anticommutator_trace_over_eta2': str(different),
            'sharp_bosonic_upper_over_2pow3072': str(upper / 2**3072),
            'both_trial_uppers_lt_2pow3092': True,
            'nonconstant_Haar_kinetic_coefficient': '27/8',
            'normalizer': 'exact positive integral Z_N(t), not numerically evaluated'}


def finite_pair_and_multiplicity_controls():
    modes = [sum(1 << x for x in subset) for subset in itertools.combinations(range(4), 2)]
    def epsilon(mask):
        subset = [x for x in range(4) if mask & (1 << x)]
        return (-1) ** sum(x - i for i, x in enumerate(subset))
    pair = {(mask, 15 ^ mask): sp.Integer(epsilon(mask)) for mask in modes}
    assert sum(a * a for a in pair.values()) == 6
    wedge = {}
    for a in range(4):
        for b in range(4):
            matrix = sp.zeros(6)
            for j, mask in enumerate(modes):
                step = car_step(mask, a, b)
                if step is not None:
                    out, sign = step
                    matrix[modes.index(out), j] = sign
            wedge[a, b] = matrix
    omega = sp.zeros(36, 1)
    for (a, b), amplitude in pair.items():
        omega[6 * modes.index(a) + modes.index(b)] = amplitude
    for a in range(4):
        for b in range(4):
            collective = sp.kronecker_product(wedge[a, b], sp.eye(6)) + sp.kronecker_product(sp.eye(6), wedge[a, b])
            assert collective * omega == (1 if a == b else 0) * omega
    # Three inequivalent pair embeddings span ALL three equivalent omega2 copies.
    embeddings = sp.zeros(216, 18)
    for free in range(3):
        paired = [x for x in range(3) if x != free]
        for j, mask in enumerate(modes):
            for (left, right), amplitude in pair.items():
                indices = [0, 0, 0]
                indices[free] = j
                indices[paired[0]] = modes.index(left)
                indices[paired[1]] = modes.index(right)
                row = 36 * indices[0] + 6 * indices[1] + indices[2]
                embeddings[row, free * 6 + j] = amplitude
    gram = embeddings.T * embeddings
    assert gram.rank() == 18
    labels = {(0, 0, 0, 0): 1}
    for _ in range(3):
        next_labels = defaultdict(int)
        for partition, multiplicity in labels.items():
            for rows in itertools.combinations(range(4), 2):
                added = tuple(partition[i] + (1 if i in rows else 0) for i in range(4))
                if all(added[i] >= added[i + 1] for i in range(3)):
                    next_labels[added] += multiplicity
        labels = dict(next_labels)
    assert labels[(2, 2, 1, 1)] == 3
    dimensions = {}
    for partition in labels:
        dimension = Fraction(1)
        for i in range(4):
            for j in range(i + 1, 4):
                dimension *= Fraction(partition[i] - partition[j] + j - i, j - i)
        dimensions[partition] = int(dimension)
    assert sum(labels[p] * dimensions[p] for p in labels) == 216
    identity = sp.eye(6)
    for a in range(4):
        for b in range(4):
            collective = (sp.kronecker_product(wedge[a, b], identity, identity)
                          + sp.kronecker_product(identity, wedge[a, b], identity)
                          + sp.kronecker_product(identity, identity, wedge[a, b]))
            effective = sp.kronecker_product(sp.eye(3), wedge[a, b] + (1 if a == b else 0) * identity)
            assert collective * embeddings == embeddings * effective
    # A coherent vector differs from a sum of diagonal-copy norms.
    cross = gram[0, 6]
    assert cross != 0
    coherent_norm = gram[0, 0] + gram[6, 6] + 2 * cross
    diagonal_only_norm = gram[0, 0] + gram[6, 6]
    assert coherent_norm != diagonal_only_norm
    return {'representation_control': 'M4,k2,aux3 only; NOT a native M256 Fock execution',
            'complementary_pair_norm2': 6, 'all16_collective_generators_checked': True,
            'full_omega2_multiplicity': 3, 'full_rank': 18, 'single_embedding_rank': 6,
            'independent_Pieri_multiplicities': {str(p): m for p, m in labels.items()},
            'independent_Weyl_dimension_sum': 216,
            'nonzero_cross_multiplicity_inner_product': str(cross),
            'coherent_norm2': str(coherent_norm), 'wrong_diagonal_only_norm2': str(diagonal_only_norm),
            'native63_generalization': '31 determinant pairs plus free wedge, proved analytically'}


def native_body(profiles):
    sites = list(itertools.product(range(4), repeat=3))
    index = {x: i for i, x in enumerate(sites)}
    temporal = sp.diag(2, -3, 4, -3)
    fields = [temporal]
    for a, b, value in [(0, 1, 2), (1, 2, sp.I), (2, 3, 3)]:
        field = sp.zeros(4)
        field[a, b] = value
        field[b, a] = sp.conjugate(value)
        fields.append(field)
    u = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, sp.I, 0], [0, 0, 0, -sp.I]])
    v = sp.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    assert u.det() == v.det() == 1 and u * v != v * u
    links = [u, v, v * u]
    checker = sum(15 << (4 * i) for i, x in enumerate(sites) if sum(x) % 2 == 0)
    results = []
    for profile in profiles:
        entries = []
        diagonal_trace = 0
        expected_checker = 0
        paths = 0
        for x in sites:
            ix = index[x]
            fx = 1 + x[0] + 2 * x[1] + 3 * x[2]
            onsite = sp.Rational(profile['onsite']) * fx * temporal
            diagonal_trace += sp.trace(onsite)
            if sum(x) % 2 == 0:
                expected_checker += sp.trace(onsite)
            for a in range(4):
                entries.append((4 * ix + a, 4 * ix + a, onsite[a, a]))
            for path in profile['paths']:
                axis, length = int(path['axis']), int(path['length'])
                y = list(x)
                y[axis] = (y[axis] + length) % 4
                y = tuple(y)
                iy = index[y]
                fy = 1 + y[0] + 2 * y[1] + 3 * y[2]
                colour = links[axis] ** length
                c0, ci = sp.Rational(path['c0']), sp.Rational(path['ci'])
                primitive = c0 * temporal + sp.I * ci * fields[axis + 1]
                ap = -1 if x[axis] + length >= 4 else 1
                body = ap * sp.I**length * (fx * primitive * colour + colour * fy * primitive) / 2
                paths += 1
                for a in range(4):
                    for b in range(4):
                        coefficient = body[a, b]
                        if coefficient != 0:
                            entries.extend([(4 * ix + a, 4 * iy + b, coefficient),
                                            (4 * iy + b, 4 * ix + a, sp.conjugate(coefficient))])
        assert diagonal_trace == expected_checker == 0
        action = act(entries, {checker: sp.S.One})
        action_norm = squared_norm(action)
        assert action_norm > 0
        assert all(mask.bit_count() == 128 for mask in action)
        results.append({'role': profile['role'], 'native_positive_paths': paths,
                        'checker_expectation': '0', 'full_onebody_trace': '0',
                        'canonical_action_norm2_unscaled': str(action_norm),
                        'pair_trace_expectation': '0',
                        'physical_scale': '-2^8192' if profile['scale'] == 'heavy' else '1'})
    assert profiles[1]['paths'] == profiles[2]['paths']
    mismatch = sp.Rational(profiles[1]['onsite']) - sp.Rational(profiles[2]['onsite'])
    assert mismatch == -sp.Rational(126663739519795, 70368744177664)
    # A physical SU4 colour change is conjugated through the local onebody jet.
    b = fields[1]
    assert u * b * u.H != b
    assert sp.trace(u * b * u.H) == 0
    for generator in generators():
        local = [(a, c, generator[a, c]) for a in range(4) for c in range(4) if generator[a, c] != 0]
        assert not act(local, {15: sp.S.One}) and not act(local, {0: sp.S.One})
    assert checker.bit_count() == 128
    return {'N': 4, 'V': 64, 'M': 256, 'k_each_all65': 128,
            'fullcolour_noncommuting_Phi_SU4': True, 'profiles': results,
            'active_auxiliary_onsite_mismatch': str(mismatch),
            'charged_common_pair_action': 'common traceless onebody annihilates all32 pairs',
            'active_mismatch_expectation': '0 by reduced k/M=1/2 trace',
            'missing_neutral_body_rejected': results[0]['canonical_action_norm2_unscaled'] != '0',
            'matter_expectation_zero_does_not_mean_operator_zero': True}


def native_gauss_electric_sewing():
    sites = list(itertools.product(range(4), repeat=3))
    parity = {x: (-1) ** sum(x) for x in sites}
    raw, longitudinal, transverse = {}, {}, {}
    for x in sites:
        for axis in range(3):
            raw[x, axis] = Fraction(64 * parity[x] if axis == 0 else 0)
            longitudinal[x, axis] = Fraction(64, 3) * parity[x]
            transverse[x, axis] = raw[x, axis] - longitudinal[x, axis]
    def divergence(flow, x):
        total = 0
        for axis in range(3):
            previous = list(x)
            previous[axis] = (previous[axis] - 1) % 4
            total += flow[x, axis] - flow[tuple(previous), axis]
        return total
    assert all(divergence(raw, x) == divergence(longitudinal, x) == 128 * parity[x] for x in sites)
    assert all(divergence(transverse, x) == 0 for x in sites)
    assert sum(longitudinal[e] * transverse[e] for e in raw) == 0
    he = 64 * sum(v * v for v in transverse.values())
    hc = 64 * sum(v * v for v in longitudinal.values())
    full = 64 * sum(v * v for v in raw.values())
    assert he / 64 == Fraction(524288, 3)
    assert hc / 64 == Fraction(262144, 3)
    assert full == he + hc == 262144 * 64
    # Exact native inverse-Laplacian dipole diagonal; each q retained.
    cosine = [1, 0, -1, 0]
    resistance = sum((Fraction(2 * (1 - cosine[q[0]]), 64 * 2 * (3 - sum(cosine[j] for j in q)))
                      for q in itertools.product(range(4), repeat=3) if q != (0, 0, 0)), Fraction(0))
    assert resistance == Fraction(21, 64)
    wrong_jet_square = resistance**2 + (1 - resistance)**2
    assert wrong_jet_square != 1
    assert 64 * ((64 - 1)**2 - 64**2) == -8128
    # Four modes at a site: aux determinant31 contributes124 before freeR.
    aux = 31 * 4 + 4
    ref_occupancy, ref_background = 4, 4
    rho = 0 + aux - 128 + ref_occupancy - ref_background
    assert aux == 128 and rho == 0
    wrong_rho = 0 + 4 - 128
    assert wrong_rho == -124
    assert 0 + aux - 128 - ref_background == -4
    determinant = sp.I  # W_x=exp(i*pi/8) I4.
    physical_sewing = determinant**31 * sp.I
    assert physical_sewing == 1 and sp.I != physical_sewing
    assert 63 * 128 == 31 * 256 + 128
    assert tuple([63] * 128 + [0] * 128) != tuple([32] * 128 + [31] * 128)
    return {'checker_rho': '128 parity', 'paired_rho': 0,
            'HE_per_site': str(he / 64), 'HC_per_site': str(hc / 64),
            'once_fullE_trial_difference': str(full), 'difference_per_site': 262144,
            'wrong_double_counted_fullE': str(2 * full),
            'native_hop_longitudinal_jet': str(resistance),
            'native_hop_transverse_jet': str(1 - resistance),
            'wrong_separate_jet_squares': str(wrong_jet_square), 'correct_full_jet_square': 1,
            'native_hop_once_fullE_difference': -8128,
            'physical_aux_number': aux, 'wrong_without124_rho': wrong_rho,
            'wrong_deleted_Ref_charge': -4,
            'physical_U1_det31_sewing': str(physical_sewing), 'wrong_no_det_sewing': 'I',
            'native_auxiliary_total_degree': 8064,
            'highest_partition': '(63^128,0^128)', 'paired_aux_partition': '(32^128,31^128)'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    payload = args.input.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == request['input_sha256']
    input_data = json.loads(payload)
    profiles = input_data['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    sys.set_int_max_str_digits(0)
    result = {'kind': 'INDEPENDENT_NEW_NATIVE_EXACT_TRIAL_FORM_REVIEW_NOT_GROUND',
              'governor': request['governor'], 'input_sha256': request['input_sha256'],
              'bosonic': wick_and_sharp_price(),
              'finite_representation': finite_pair_and_multiplicity_controls(),
              'native_body': native_body(profiles),
              'native_gauss_electric_sewing': native_gauss_electric_sewing(),
              'runtime': {'executable': sys.executable, 'python': sys.version, 'sympy': sp.__version__},
              'actual_auxiliary_weights': 'UNPAID', 'complete_competitor_ordering': 'UNPAID',
              'scientific_admission': False, 'campaign_complete': False}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': args.output.name, 'new_native_consumer_completed': True,
                      'actual_sector_selection': 'UNPAID'}), flush=True)


if __name__ == '__main__':
    main()

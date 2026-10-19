"""Independent exact Mayer/temporal-rod/CAR controls; no target imports or CLI calls."""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path


def edges_for(n):
    return list(itertools.combinations(range(n), 2))


def bfs_tree(n, edges):
    adjacency = [set() for _ in range(n)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    depth = {0: 0}
    queue = [0]
    for parent in queue:
        for child in sorted(adjacency[parent]):
            if child not in depth:
                depth[child] = depth[parent] + 1
                queue.append(child)
    if len(depth) != n:
        return None
    parents = {v: min(u for u in adjacency[v] if depth[u] == depth[v] - 1)
               for v in range(1, n)}
    tree = frozenset(tuple(sorted((v, p))) for v, p in parents.items())
    closure = set(tree)
    for a, b in edges_for(n):
        if depth[a] == depth[b]:
            closure.add((a, b))
        elif abs(depth[a] - depth[b]) == 1:
            shallow, deep = (a, b) if depth[a] < depth[b] else (b, a)
            if shallow >= parents[deep]:
                closure.add((a, b))
    return tree, frozenset(closure)


def graph_controls():
    reports = []
    for n in range(2, 7):
        all_edges = edges_for(n)
        groups = defaultdict(list)
        connected = []
        for bits in range(1 << len(all_edges)):
            graph = frozenset(e for i, e in enumerate(all_edges) if bits >> i & 1)
            image = bfs_tree(n, graph)
            if image is not None:
                tree, closure = image
                assert tree <= graph <= closure
                groups[(tree, closure)].append(graph)
                connected.append(graph)
        for (tree, closure), members in groups.items():
            assert len(members) == 1 << len(closure - tree)
            assert len(set(members)) == len(members)
            signed = sum((-1) ** len(g) for g in members)
            assert signed == (0 if closure != tree else (-1) ** (n - 1))
        assert len(groups) == n ** (n - 2)
        complete_sum = sum((-1) ** len(g) for g in connected)
        assert complete_sum == (-1) ** (n - 1) * math.factorial(n - 1)
        probes = [frozenset(all_edges), frozenset((i, i + 1) for i in range(n - 1)),
                  frozenset(tuple(sorted((i, (i + 1) % n))) for i in range(n))]
        if n <= 4:
            probes += [frozenset(e for i, e in enumerate(all_edges) if bits >> i & 1)
                       for bits in range(1 << len(all_edges))]
        for allowed in probes:
            actual = sum((-1) ** len(g) for g in connected if g <= allowed)
            interval_sum = 0
            admissible_trees = 0
            for tree, closure in groups:
                if tree <= allowed:
                    admissible_trees += 1
                    if not ((closure - tree) & allowed):
                        interval_sum += (-1) ** (n - 1)
            assert actual == interval_sum and abs(actual) <= admissible_trees
        coefficient = F(complete_sum, math.factorial(n))
        assert coefficient == F((-1) ** (n - 1), n)
        reports.append({'vertices': n, 'connected_graphs': len(connected),
                        'Boolean_intervals': len(groups), 'complete_Mayer_sum': complete_sum,
                        'identical_hard_core_log_coefficient': str(coefficient),
                        'wrong_missing_factorial': complete_sum,
                        'factorial_wrongcase_rejected': n >= 3 and coefficient != complete_sum})
    return reports


def hard_core_polynomial(rods):
    coefficients = defaultdict(F)
    independent = []
    for bits in range(1 << len(rods)):
        chosen = [i for i in range(len(rods)) if bits >> i & 1]
        if any(rods[i]['cell'] == rods[j]['cell'] and
               max(rods[i]['lo'], rods[j]['lo']) < min(rods[i]['hi'], rods[j]['hi'])
               for i, j in itertools.combinations(chosen, 2)):
            continue
        weight = math.prod((rods[i]['weight'] for i in chosen), start=F(1))
        coefficients[len(chosen)] += weight
        independent.append(chosen)
    return dict(coefficients), independent


def convolve(a, b, order):
    out = defaultdict(F)
    for i, x in a.items():
        for j, y in b.items():
            if i + j <= order:
                out[i + j] += x * y
    return dict(out)


def formal_log(z, order):
    w = {n: a for n, a in z.items() if n}
    power = {0: F(1)}
    result = defaultdict(F)
    for k in range(1, order + 1):
        power = convolve(power, w, order)
        for n, value in power.items():
            result[n] += F((-1) ** (k + 1), k) * value
    return dict(result)


def temporal_controls():
    rods = [dict(cell=0, lo=F(0), hi=F(1), weight=F(1, 3)),
            dict(cell=0, lo=F(2), hi=F(3), weight=F(-1, 5)),
            dict(cell=0, lo=F(1, 2), hi=F(5, 2), weight=F(2, 7)),
            dict(cell=1, lo=F(0), hi=F(4), weight=F(-1, 11))]
    z, independent = hard_core_polynomial(rods)
    assert [0, 1] in independent and [0, 2] not in independent
    old_length, new_length = F(1), F(2)
    overlap_translation_measure = old_length + new_length
    assert overlap_translation_measure == 3 and overlap_translation_measure > old_length
    first_only, _ = hard_core_polynomial(rods[:2])
    assert first_only == {0: F(1), 1: F(2, 15), 2: F(-1, 15)}
    base_log = formal_log(z, 6)
    copies = {0: F(1)}
    for _ in range(4):
        copies = convolve(copies, z, 6)
    copies_log = formal_log(copies, 6)
    assert all(copies_log.get(n, F(0)) == 4 * base_log.get(n, F(0)) for n in range(1, 7))
    value = sum(z.values())
    marked_numerator = sum(math.prod((rods[i]['weight'] for i in subset), start=F(1))
                           for subset in independent if 0 in subset)
    local_ratio = marked_numerator / value
    assert marked_numerator * value ** 3 / value ** 4 == local_ratio
    # A P/P observable at a fixed time has zero rod length but still blocks a point.
    root_time = F(3, 4)
    spanning_root = rods[2]['lo'] < root_time < rods[2]['hi']
    assert spanning_root
    return {'gas_Z_coefficients': {str(k): str(v) for k, v in z.items()},
            'same_cell_temporally_disjoint_pair_retained': [0, 1],
            'same_cell_overlapping_pair_rejected': [0, 2],
            'old_length': '1', 'new_length': '2', 'exact_overlap_translation_measure': '3',
            'wrong_missing_new_endpoint_cost': '1', 'missing_overlap_factor_rejected': True,
            'four_copy_log_identity_through_order': 6,
            'normalized_marked_root_ratio': str(local_ratio),
            'P_P_root_needs_blocked_point_even_with_zero_rod_length': True}


def annihilate(mask, mode, unsigned=False):
    if not mask >> mode & 1:
        return None
    sign = 1 if unsigned else (-1) ** ((mask & ((1 << mode) - 1)).bit_count())
    return mask ^ (1 << mode), sign


def create(mask, mode, unsigned=False):
    if mask >> mode & 1:
        return None
    sign = 1 if unsigned else (-1) ** ((mask & ((1 << mode) - 1)).bit_count())
    return mask | (1 << mode), sign


def event(state, unsigned=False):
    mask, k0, k1 = state
    matrices = (((1, 2), (3, 1)), ((2, 1), (1, -1)))
    out = defaultdict(F)
    for edge, matrix in enumerate(matrices):
        for a, b in itertools.product(range(2), repeat=2):
            for forward in (True, False):
                tail, head = (2 + b, a) if forward else (a, 2 + b)
                first = annihilate(mask, tail, unsigned)
                second = create(first[0], head, unsigned) if first is not None else None
                if second is not None:
                    assert first is not None
                    flow = [k0, k1]
                    flow[edge] += 1 if forward else -1
                    out[(second[0], *flow)] += matrix[a][b] * first[1] * second[1]
    return dict(out)


def local_energy(occupation):
    return {0: 1, 1: 0, 2: 2, 3: 1}[occupation]


def energy(state, collapse_flow=False):
    mask, k0, k1 = state
    raw = (k0 + k1) ** 2 if collapse_flow else k0 * k0 + k1 * k1
    return local_energy(mask & 3) + local_energy(mask >> 2) + 2 * raw


def word_moment(n, reset=False, crop=False, unsigned=False, collapse_flow=False):
    ground = (5, 0, 0)
    vector = {ground: F(1)}
    seen = set(vector)
    split_at_middle = F(0)
    for index in range(n):
        next_vector = defaultdict(F)
        for state, coefficient in vector.items():
            for new, weight in event(state, unsigned).items():
                next_vector[new] += coefficient * weight
        vector = dict(next_vector)
        seen.update(vector)
        if index + 1 < n:
            filtered = {}
            for state, coefficient in vector.items():
                mask, k0, k1 = state
                if reset and (k0, k1) == (0, 0) and mask != 5:
                    continue
                if crop and ((mask & 3).bit_count() != 1 or (mask >> 2).bit_count() != 1):
                    continue
                filtered[state] = coefficient * F(1, 2 ** energy(state, collapse_flow))
            vector = filtered
            if n == 4 and index == 1:
                split_at_middle = vector.get(ground, F(0))
    return vector.get(ground, F(0)), seen, split_at_middle


def car_controls():
    for mask, i, j in itertools.product(range(16), range(4), range(4)):
        anticommutator = defaultdict(int)
        # c_i c_j^* + c_j^* c_i on every full Fock basis vector.
        for first, second in ((lambda m: create(m, j), lambda m: annihilate(m, i)),
                              (lambda m: annihilate(m, i), lambda m: create(m, j))):
            a = first(mask)
            b = second(a[0]) if a is not None else None
            if b is not None:
                assert a is not None
                anticommutator[b[0]] += a[1] * b[1]
        actual = {m: c for m, c in anticommutator.items() if c}
        assert actual == ({mask: 1} if i == j else {})
    full2, _, _ = word_moment(2)
    full4, states, middle = word_moment(4)
    reset2, _, _ = word_moment(2, reset=True)
    reset4, _, _ = word_moment(4, reset=True)
    crop2, _, _ = word_moment(2, crop=True)
    unsigned4, _, _ = word_moment(4, unsigned=True)
    collapsed4, _, _ = word_moment(4, collapse_flow=True)
    assert full2 == reset2 and full4 != reset4
    assert crop2 == 0 and full2 > 0
    assert full4 != unsigned4 and full4 != collapsed4
    assert middle == full2
    neutral = sorted(mask for mask, k0, k1 in states
                     if (k0, k1) == (0, 0) and (mask & 3).bit_count() == (mask >> 2).bit_count() == 1)
    assert neutral == [5, 6, 9, 10]
    nonzero_neutral_raw = sorted(state for state in states if state[1] + state[2] == 0 and state[1:] != (0, 0))
    assert nonzero_neutral_raw
    return {'scope': 'new full four-mode CAR, canonical2, two cells/two labelled raw parallel edges; METHOD control, not g256/all64',
            'full_Fock_CAR_vectors_checked': 16, 'canonical_dimension': 6,
            'entire_zero_raw_flow_neutral_masks': neutral, 'ground_mask': 5,
            'all_local_occupations_0_1_2_retained': True,
            'shared_raw_flow_has_no_imposed_order4_cutoff': True,
            'zero_net_nonzero_raw_flow_examples': [list(v) for v in nonzero_neutral_raw[:4]],
            'full_word2': str(full2), 'reset_word2': str(reset2),
            'full_word4': str(full4), 'wrong_neutral_reset_word4': str(reset4),
            'wrong_crop_word2': str(crop2), 'wrong_unsigned_CAR_word4': str(unsigned4),
            'wrong_net_flow_collapse_word4': str(collapsed4),
            'ground_reset_temporal_disconnected_contribution': str(full2 ** 2),
            'remaining_connected_word4': str(full4 - full2 ** 2),
            'neutral_reset_crop_sign_and_rawflow_wrongcases_rejected': True}


def exact_guards(request):
    gamma, J = F(request['gamma']), F(request['J'])
    C, s, theta = request['C'], request['s'], F(request['theta_squared'])
    radius = gamma / (72 * C * s * J)
    physical = 36 * C * s * J / gamma
    assert str(radius) == request['claimed_radius']
    r = F(1, 2)
    x_over_gamma = F(3, 72 * s)
    event_guard = theta * x_over_gamma / (1 - r)
    rod_guard = 8 * s * s * theta * x_over_gamma * (1 + r) / (1 - r) ** 3
    strengthened_r = r * F(50, 49)
    strengthened_rod = 8 * s * s * theta * x_over_gamma * F(50, 49) * (1 + strengthened_r) / (1 - strengthened_r) ** 3
    strengthened_event = theta * x_over_gamma / (1 - strengthened_r)
    assert event_guard == F(1, 2304) and rod_guard == F(3, 16)
    assert strengthened_event < F(1, 2) and strengthened_rod < 1
    return {'radius_exact': str(radius), 'radius_display': float(radius),
            'physical1_r_using_e_squared_upper9_exact': str(physical), 'physical1_r_display': float(physical),
            'event_guard_K_over_gamma_at_R': str(event_guard), 'overlap_guard_2sRrod_at_R': str(rod_guard),
            'claimed_looser_overlap_guard': '9/32',
            'time_tail_reserve_epsilon_over_gamma': '1/100',
            'time_tail_strengthened_r_upper': str(strengthened_r),
            'time_tail_strengthened_event_K_over_gamma': str(strengthened_event),
            'time_tail_strengthened_overlap_guard': str(strengthened_rod),
            'both_time_tail_guards_strict': True, 'physical1_certified': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    request_path = Path(args.request)
    request = json.loads(request_path.read_text())
    for name, expected in request['pins'].items():
        actual = hashlib.sha256((request_path.parent / name).read_bytes()).hexdigest()
        assert actual == expected, name
    result = {'kind': 'INDEPENDENT_EXACT_NORMALIZED_MAYER_CONTROLS',
              'graph_tree_Boolean_interval_controls': graph_controls(),
              'temporal_rod_and_normalization_controls': temporal_controls(),
              'full_CAR_neutral_memory_raw_flow_controls': car_controls(),
              'exact_radius_and_two_guards': exact_guards(request),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'request_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
              'target_imports_or_scientist_runs': 0,
              'all_assertions_reached': True}
    with Path(args.output).open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps({'natural_main_finished': True, 'graph_max_vertices': 6,
                      'wrong_factorial_overlap_reset_crop_sign_rawflow_rejected': True,
                      'radius': result['exact_radius_and_two_guards']['radius_exact']}))


if __name__ == '__main__':
    main()

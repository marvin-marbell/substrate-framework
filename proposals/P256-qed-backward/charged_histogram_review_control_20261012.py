"""Independent exact exterior-algebra/species/flow control; no spectral solver."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path

ZERO = (F(0), F(0))
ONE = (F(1), F(0))
I = (F(0), F(1))
EDGES = ((0, 1, 0), (1, 2, 1), (3, 2, 0), (0, 3, 1))


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def mul(a, b):
    return a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]


def scale(a, n):
    return a[0]*n, a[1]*n


def conjugate(a):
    return a[0], -a[1]


def norm2(a):
    return a[0]*a[0] + a[1]*a[1]


def exterior(mask, mode, creation):
    """Delete/insert in the ordered exterior-product list, independently of bits."""
    occupied = [j for j in range(max(mask.bit_length(), mode+1)) if mask & (1 << j)]
    if creation:
        if mode in occupied:
            return None
        position = sum(j < mode for j in occupied)
        occupied.insert(position, mode)
    else:
        if mode not in occupied:
            return None
        position = occupied.index(mode)
        occupied.pop(position)
    return sum(1 << j for j in occupied), (-1 if position % 2 else 1)


def transfer(mask, target, source):
    removed = exterior(mask, source, False)
    if removed is None:
        return None
    inserted = exterior(removed[0], target, True)
    return None if inserted is None else (inserted[0], removed[1]*inserted[1])


def literal_bit_formula(mask, target, source):
    """Reviewed formula used solely as the comparison side, never the control."""
    if target == source:
        return (mask, 1) if mask & (1 << source) else None
    if not mask & (1 << source) or mask & (1 << target):
        return None
    sign = (-1) ** (mask & ((1 << source)-1)).bit_count()
    after = mask ^ (1 << source)
    sign *= (-1) ** (after & ((1 << target)-1)).bit_count()
    return after | (1 << target), sign


def car_control():
    checks = 0
    wrong_sign = None
    for occupied in itertools.combinations(range(16), 8):
        mask = sum(1 << j for j in occupied)
        for target, source in itertools.product(range(16), repeat=2):
            actual = transfer(mask, target, source)
            assert actual == literal_bit_formula(mask, target, source)
            checks += 1
            if actual and actual[1] == -1 and wrong_sign is None:
                wrong_sign = {'mask': mask, 'target': target, 'source': source,
                              'new_mask': actual[0], 'correct': -1, 'unsigned_wrong': 1}
    anticommutators = 0
    for mask in range(16):
        for a, b in itertools.product(range(4), repeat=2):
            answer = defaultdict(int)
            for first, second in (((b, True), (a, False)), ((a, False), (b, True))):
                x = exterior(mask, *first)
                if x:
                    y = exterior(x[0], *second)
                    if y:
                        answer[y[0]] += x[1]*y[1]
            answer = {k: v for k, v in answer.items() if v}
            assert answer == ({mask: 1} if a == b else {})
            anticommutators += 1
    assert wrong_sign is not None
    return {'canonical16_Fock8_dimension': math.comb(16, 8),
            'all_mask_transfer_comparisons': checks,
            'four_mode_CAR_anticommutator_checks': anticommutators,
            'unsigned_CAR_adverse_witness': wrong_sign}


def histogram_control():
    """Project an explicit labelled tensor, not a postulated histogram formula."""
    masks = [sum(1 << j for j in pair) for pair in itertools.combinations(range(4), 2)]
    grouped = defaultdict(list)
    for tensor in itertools.product(masks, repeat=3):
        grouped[tuple(sorted(Counter(tensor).items()))].append(tensor)
    checks = 0
    witness = None
    for histogram, tensors in grouped.items():
        counts = dict(histogram)
        for target, source in itertools.permutations(range(4), 2):
            output = defaultdict(int)
            for tensor in tensors:
                for species, mask in enumerate(tensor):
                    moved = transfer(mask, target, source)
                    if moved:
                        next_tensor = list(tensor)
                        next_tensor[species] = moved[0]
                        output[tuple(sorted(Counter(next_tensor).items()))] += moved[1]
            for old, nold in histogram:
                moved = transfer(old, target, source)
                if moved is None:
                    continue
                new, sign = moved
                updated = counts.copy()
                updated[old] -= 1
                updated[new] = updated.get(new, 0)+1
                next_hist = tuple(sorted((m, n) for m, n in updated.items() if n))
                numerator = output[next_hist]
                squared = F(numerator*numerator, len(tensors)*len(grouped[next_hist]))
                expected = nold*(counts.get(new, 0)+1)
                assert squared == expected and numerator*sign > 0
                checks += 1
                if counts.get(new, 0) and witness is None:
                    witness = {'old_histogram': histogram, 'target': target, 'source': source,
                               'labelled_matrix_element_numerator': numerator,
                               'old_orbit_size': len(tensors), 'new_orbit_size': len(grouped[next_hist]),
                               'normalized_amplitude_squared': str(squared),
                               'wrong_without_bose_enhancement_squared': nold}
    assert witness is not None
    return {'species': 3, 'modes_per_species': 4, 'particles_per_species': 2,
            'labelled_tensor_dimension': 6**3, 'histograms': len(grouped),
            'all_allowed_histogram_transfer_checks': checks,
            'enhancement_adverse_witness': witness,
            '63_from_single_occupied_histogram_correct_amplitude_squared': 63,
            'wrong63_linear_amplitude_squared': 63**2}


def native_rows(profile, s, r):
    """Exact Gaussian-rational matrices from explicit tensor Pauli entries."""
    sx = ((ZERO, ONE), (ONE, ZERO))
    sy = ((ZERO, scale(I, -1)), (I, ZERO))
    sz = ((ONE, ZERO), (ZERO, scale(ONE, -1)))
    rows = []
    for edge, (left, right, axis) in enumerate(EDGES):
        path = next(p for p in profile['paths'] if p['axis'] == axis and p['length'] == 1)
        c0, ci = s*F(path['c0']), r*F(path['ci'])
        pauli = (sx, sy, sz)[axis]
        for a, b in itertools.product(range(4), repeat=2):
            beta = (F(1 if a < 2 else -1), F(0)) if a == b else ZERO
            alpha = mul(sx[a//2][b//2], pauli[a%2][b%2])
            coefficient = add(scale(beta, c0), scale(mul(I, alpha), ci))
            if coefficient != ZERO:
                rows.append((4*left+a, 4*right+b, coefficient, edge, 1))
                rows.append((4*right+b, 4*left+a, conjugate(coefficient), edge, -1))
    return rows


def orbit_size(histogram):
    return math.factorial(63)//math.prod(math.factorial(n) for _, n in histogram)


def image(state, rows):
    """Average-orbit basis has norm squared 1/N_h and coefficient n_old."""
    active, histogram, electric = state
    answer = defaultdict(lambda: ZERO)
    currents = [defaultdict(lambda: ZERO) for _ in EDGES]
    counts = dict(histogram)
    for target, source, value, edge, direction in rows:
        e = list(electric)
        e[edge] += direction
        moved = transfer(active, target, source)
        if moved:
            key = moved[0], histogram, tuple(e)
            coefficient = scale(value, moved[1])
            answer[key] = add(answer[key], coefficient)
            currents[edge][key] = add(currents[edge][key], scale(mul(I, coefficient), direction))
        for old, nold in histogram:
            moved = transfer(old, target, source)
            if moved is None:
                continue
            new, sign = moved
            updated = counts.copy()
            updated[old] -= 1
            updated[new] = updated.get(new, 0)+1
            next_hist = tuple(sorted((m, n) for m, n in updated.items() if n))
            key = active, next_hist, tuple(e)
            coefficient = scale(value, nold*sign)
            answer[key] = add(answer[key], coefficient)
            currents[edge][key] = add(currents[edge][key], scale(mul(I, coefficient), direction))
    return ({k: v for k, v in answer.items() if v != ZERO},
            [{k: v for k, v in current.items() if v != ZERO} for current in currents])


def gauss(state):
    active, histogram, electric = state
    assert active.bit_count() == 8 and sum(n for _, n in histogram) == 63
    assert all(mask.bit_count() == 8 for mask, _ in histogram)
    density = [((active >> (4*j)) & 15).bit_count()-2 +
               sum(n*(((mask >> (4*j)) & 15).bit_count()-2) for mask, n in histogram)
               for j in range(4)]
    divergence = [0]*4
    for e, (left, right, _) in zip(electric, EDGES):
        divergence[left] += e
        divergence[right] -= e
    assert divergence == density


def energy(state, ma, mb, s):
    active, histogram, electric = state
    def positive(mask):
        return sum(((mask >> (4*j)) & 3).bit_count() for j in range(4))
    return 64*sum(e*e for e in electric) + 2*s*(ma*positive(active) +
        mb*sum(n*positive(mask) for mask, n in histogram))


def exact_shell_control(active, aux):
    s, r = F(1, 2), F(1, 3)
    assert active['load'] == 1 and aux['load'] == 63
    assert active['paths'] == aux['paths'] and active['onsite'] != aux['onsite']
    rows = native_rows(active, s, r)
    assert rows == native_rows(aux, s, r)
    sea = sum(12 << (4*j) for j in range(4))
    initial = sea, ((sea, 63),), (0, 0, 0, 0)
    initial_image, _ = image(initial, rows)
    gram = sum(norm2(v)/orbit_size(k[1]) for k, v in initial_image.items())
    expected = 1024*(r*F(active['paths'][0]['ci']))**2
    assert gram == expected
    states = {initial: 0}
    frontier = [initial]
    levels = [1]
    for _ in range(2):
        new = []
        for state in frontier:
            for next_state in image(state, rows)[0]:
                if next_state not in states:
                    states[next_state] = len(states)
                    new.append(next_state)
        frontier = new
        levels.append(len(new))
    outside = set()
    complement_edges = 0
    hermiticity_checks = 0
    current_checks = 0
    cache = {state: image(state, rows) for state in states}
    outgoing_witness = None
    ledger = hashlib.sha256()
    opposite = []
    for state in states:
        gauss(state)
        trans, currents = cache[state]
        for next_state, coefficient in trans.items():
            gauss(next_state)
            ledger.update(repr((state, next_state, coefficient)).encode())
            if next_state in states:
                reverse = cache[next_state][0].get(state, ZERO)
                assert scale(coefficient, F(1, orbit_size(next_state[1]))) == scale(conjugate(reverse), F(1, orbit_size(state[1])))
                hermiticity_checks += 1
            else:
                outside.add(next_state)
                complement_edges += 1
                if outgoing_witness is None:
                    outgoing_witness = {'shell_state': state, 'outside_state': next_state,
                                        'coefficient_average_basis': [str(x) for x in coefficient]}
        for edge, current in enumerate(currents):
            for next_state, coefficient in current.items():
                gauss(next_state)
                if next_state in states:
                    reverse = cache[next_state][1][edge].get(state, ZERO)
                    assert scale(coefficient, F(1, orbit_size(next_state[1]))) == scale(conjugate(reverse), F(1, orbit_size(state[1])))
                    current_checks += 1
        if state[2] == (0, 0, 0, 0) and state[0] != sea and state[1] != ((sea, 63),):
            opposite.append(state)
    assert outside and opposite
    assert levels == [1, 32, 648]
    ma, mb = F(active['onsite']), F(aux['onsite'])
    external = sea, ((sea, 63),), (1, 1, -1, -1)
    gauss(external)
    assert energy(external, ma, mb, s) == 256
    costs = sorted({64*(sum(e*e for e in target[2])-4) for target in image(external, rows)[0]})
    assert costs == [-64, 192]
    masses = sorted({energy(target, ma, mb, s)-64 for target in initial_image})
    assert masses == [ma, mb]
    c_internal = 64*16*sum(abs(s*F(p['c0']))+abs(r*F(p['ci'])) for p in active['paths'] if p['length'] == 1 and p['axis'] in (0, 1))
    c_boundary = 64*4*16*sum(abs(s*F(p['c0']))+abs(r*F(p['ci'])) for p in active['paths'])
    return {'basis': 'average orbit vectors; exact Gaussian rational coefficients and metric 1/orbit_size',
            'native_rows': len(rows), 'levels': levels, 'dimension': len(states),
            'outside_target_states': len(outside), 'nonzero_complement_couplings': complement_edges,
            'exact_H_metric_Hermiticity_checks': hermiticity_checks,
            'exact_J_metric_Hermiticity_checks': current_checks,
            'exact_transition_ledger_sha256': ledger.hexdigest(),
            'first_native_Gram_exact': str(gram), 'wrong_squared64_Gram_exact': str(64*gram),
            'opposite_active_aux_shared_zero_flow_states': len(opposite),
            'opposite_shared_E_cost': 0, 'wrong_independent_species_E_cost': 128,
            'opposite_flow_witness': opposite[0], 'outgoing_complement_witness': outgoing_witness,
            'shared_external_E_cost': 256, 'wrong_repeated64_E_cost': 16384,
            'external_native_E_energy_changes': costs,
            'active_aux_first_excitation_mass_costs_exact': [str(x) for x in masses],
            'C_internal_exact': str(c_internal), 'C_boundary_allnative_exact': str(c_boundary),
            'C_J_exact_conservative': str(c_boundary)}


def boundary_geometry_control(profiles):
    patch = {(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)}
    counts = {}
    for n in (4, 6, 8):
        length2_internal = 0
        crossing_selected_link = 0
        for path in profiles['paths']:
            if path['length'] != 2:
                continue
            axis = path['axis']
            for site in itertools.product(range(n), repeat=3):
                endpoint = list(site)
                endpoint[axis] = (endpoint[axis]+2) % n
                endpoint = tuple(endpoint)
                length2_internal += site in patch and endpoint in patch
                flow = []
                current = list(site)
                for _ in range(2):
                    flow.append((tuple(current), axis))
                    current[axis] = (current[axis]+1) % n
                selected = {((0, 0, 0), 0), ((1, 0, 0), 1), ((0, 1, 0), 0), ((0, 0, 0), 1)}
                if any(link in selected for link in flow):
                    crossing_selected_link += 1
                    assert site in patch or endpoint in patch
        assert length2_internal == 0 and crossing_selected_link == 8
        counts[str(n)] = {'length2_both_endpoints_inside_patch': length2_internal,
                          'forward_length2_paths_crossing_selected_links': crossing_selected_link}
    return {'periodic_geometry': counts,
            'AP_unit_phase_bound': 'abs(+1)=abs(-1)=1; retained in native coefficient/current upper',
            'scope': 'geometry and coefficient price, not numerical allnative dynamics; fixed outside occupation required'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    root = args.request.parent
    for name, sha in request['input_pins'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == sha
    profiles = json.loads((root/request['profiles']).read_bytes())['primitive_profiles']
    result = {'kind': 'INDEPENDENT_EXACT_HISTOGRAM_CAR_FLOW_CONTROL_NOT_GROUND_RESPONSE',
              'governor': request['governor'], 'input_pins': request['input_pins'],
              'CAR': car_control(), 'labelled_species': histogram_control(),
              'native_shell': exact_shell_control(profiles[1], profiles[2]),
              'boundary_geometry': boundary_geometry_control(profiles[1]),
              'arithmetic': 'exact integer/Fraction Gaussian rational; no floating point or eigensolver',
              'scientific_admission': False, 'actual_ground_response': False, 'campaign_complete': False}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'exact_controls_passed': True, 'shell_dimension': result['native_shell']['dimension'],
                      'no_old_diagonalizer_rerun': True, 'scientific_admission': False}))


if __name__ == '__main__':
    main()

"""Independent exact-rational proof-boundary controls; never imports the producer."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PINS = {
    "local_so5_positive_output_row_inertia_certificate_20261011.py": "9f00cbd0318dfc2da97b7567a25d1478c9e4069ae888dedfeb92c15145256377",
    "local_so5_positive_output_row_inertia_certificate_20261011_receipt.json": "f6c8823eb6b5005cdb8067088d14fb561cdacbd3bba9c54e0c5042fe160e4e73",
    "local_so5_resum_freeframe_proof_20261011.json": "a63812d27615d5f5f7e316ddad13710d5d4100a519d89268b27e9b79cac81451",
    "local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json": "102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8",
    "local_so5_resum_actual04_20261011_receipt.json": "a31c2d5be2a368b7de703d92461e230af59adf7e8ee8f4eaca0944fe4f556fc1",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transpose(a):
    return [list(row) for row in zip(*a)]


def product(a, b):
    return [[sum((x * y for x, y in zip(row, col)), F(0))
             for col in zip(*b)] for row in a]


def combine(a, b, coefficient=1):
    return [[x + coefficient * y for x, y in zip(ar, br)]
            for ar, br in zip(a, b)]


def times(a, scalar):
    return [[scalar * x for x in row] for row in a]


def form(a, v):
    return sum(v[i] * a[i][j] * v[j]
               for i in range(len(v)) for j in range(len(v)))


def determinant2(a):
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]


def psd2(a):
    return (a == transpose(a) and a[0][0] >= 0 and a[1][1] >= 0
            and determinant2(a) >= 0)


def inverse2(a):
    determinant = determinant2(a)
    assert determinant != 0
    return [[a[1][1] / determinant, -a[0][1] / determinant],
            [-a[1][0] / determinant, a[0][0] / determinant]]


def rational_ldl_pivots(a):
    # Independent rational congruence elimination, not producer Bareiss.
    a = [[F(x) for x in row] for row in a]
    pivots = []
    for k in range(len(a)):
        pivot = a[k][k]
        assert pivot != 0
        pivots.append(pivot)
        for i in range(k + 1, len(a)):
            for j in range(k + 1, len(a)):
                a[i][j] -= a[i][k] * a[k][j] / pivot
    return pivots


def sqrt_enclosure(x):
    grid = 1 << 160
    lo = math.isqrt(x.numerator * grid * grid // x.denominator)
    upper = lo if lo * lo * x.denominator == x.numerator * grid * grid else lo + 1
    return F(lo, grid), F(upper, grid)


def occupation_candidates(n, gaps, budget):
    # Cartesian product of independently bounded excitation counts, not the
    # producer's recursive accumulated-dyadic enumerator. Positive lower gaps
    # imply that any omitted coordinate already costs more than the budget.
    assert all(lower > 0 for lower, _ in gaps)
    ranges = [range(min(n, int(F(budget) // lower)) + 1) for lower, _ in gaps]
    result = []
    for word in itertools.product(*ranges):
        if sum(word) > n:
            continue
        lower = sum((k * gap[0] for k, gap in zip(word, gaps)), F(0))
        upper = sum((k * gap[1] for k, gap in zip(word, gaps)), F(0))
        if lower <= budget:
            result.append(((n - sum(word), *word), lower, upper))
    return result


def missing_density_images(outputs, support):
    missing = []
    edges = [(dest, src) for dest in (0, 2, 4) for src in (1, 3)]
    edges += [(src, dest) for dest, src in edges]
    for state in outputs:
        for role in range(2):
            for dest, src in edges:
                if not state[role][src]:
                    continue
                word = list(state[role])
                word[src] -= 1
                word[dest] += 1
                target = list(state)
                target[role] = tuple(word)
                target = tuple(target)
                if target not in support:
                    missing.append((state, target))
    return missing


def main():
    observed = {name: digest(ROOT / name) for name in PINS}
    assert observed == PINS
    p = json.loads((ROOT / "local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json").read_text())
    old = json.loads((ROOT / "local_so5_resum_freeframe_proof_20261011.json").read_text())
    for current_profile, frozen_profile in zip(p["profiles"], old["profiles"]):
        assert all(current_profile[name] == frozen_profile[name]
                   for name in ("spin_replicas", "m_exact", "d_exact", "c0_exact", "e_squared_exact"))
    roles = []
    base_lower = F(0)
    base_upper = F(0)
    for profile in p["profiles"]:
        n = profile["spin_replicas"]
        mass, d, c = (F(profile[name]) for name in ("m_exact", "d_exact", "c0_exact"))
        assert mass > 0 and d != 0 and c < 0
        e2 = F(profile["e_squared_exact"])
        assert e2 == mass * mass + d * d
        elo, ehi = sqrt_enclosure(e2)
        assert elo > abs(c)
        gaps = [(2 * elo + 2 * c, 2 * ehi + 2 * c),
                (2 * elo, 2 * ehi),
                (2 * elo - 2 * c, 2 * ehi - 2 * c),
                (4 * elo, 4 * ehi)]
        roles.append(occupation_candidates(n, gaps, 322))
        base_lower -= 2 * n * ehi
        base_upper -= 2 * n * elo
    states = {}
    for a, alo, ahi in roles[0]:
        for b, blo, bhi in roles[1]:
            if alo + blo <= 322:
                states[(a, b)] = (alo + blo, ahi + bhi)
    support = set(states)
    outputs = {state for state, (_, upper) in states.items() if upper <= 240}
    negative_columns = sum(base_upper + upper < -5123 for _, upper in states.values())
    assert all(base_upper + upper < -5123 or base_lower + lower > -5123
               for lower, upper in states.values())
    assert (len(support), len(outputs), negative_columns) == (1768, 589, 14)
    assert not missing_density_images(outputs, support)
    candidates = missing_density_images(outputs, outputs)
    assert candidates
    victim = candidates[0][1]
    omitted = missing_density_images(outputs, support - {victim})
    assert omitted and victim not in outputs
    exterior_floor = base_lower + 322
    assert exterior_floor > -5123
    assert base_lower + 50 < -5123  # Plausible unpaid exterior fails the floor gate.

    # Wrong input PHP is not a lower Loewner operator, even with symmetric R.
    r = [[F(1), F(1)], [F(1), F(2)]]
    full = times(product(r, r), 64)
    cropped_input = [[F(64), F(0)], [F(0), F(0)]]
    correct_output = times(product(transpose([r[0]]), [r[0]]), 64)
    wrong_comparison = combine(full, cropped_input, -1)
    correct_comparison = combine(full, correct_output, -1)
    assert form(wrong_comparison, [-3, 1]) == -256
    assert not psd2(wrong_comparison) and psd2(correct_comparison)

    # Exact positive-Q Schur and non-solved fixed witness control.
    identity = [[F(1), F(0)], [F(0), F(1)]]
    bq = [[F(1)], [F(1)]]
    k = combine(identity, times(product(bq, transpose(bq)), F(2, 3)))
    exact = inverse2(k)
    y = [[F(1, 2), F(1, 8)], [F(1, 8), F(1, 2)]]
    g = combine(combine(y, transpose(y)), product(transpose(y), product(k, y)), -1)
    residual = combine(identity, product(k, y), -1)
    residual_gram = product(transpose(residual), residual)
    assert psd2(combine(exact, g, -1))
    assert psd2(combine(combine(g, residual_gram), exact, -1))
    frobenius_squared = sum(x * x for row in residual for x in row)
    assert psd2(combine(times(identity, frobenius_squared), residual_gram, -1))
    schur = combine([[F(-2), F(0)], [F(0), F(-1)]], times(exact, 2))
    schur_pivots = rational_ldl_pivots(schur)
    full_pivots = rational_ldl_pivots([
        [F(7), F(2), F(2)], [F(2), F(0), F(0)], [F(2), F(0), F(1)]
    ])
    assert sum(x < 0 for x in schur_pivots) == sum(x < 0 for x in full_pivots) == 1

    # All negative leading minors does not mean all negative eigenvalues.
    pivots = rational_ldl_pivots([[-3, 0, 0], [0, 2, 0], [0, 0, 5]])
    leading_minors = list(itertools.accumulate(pivots, lambda a, b: a * b))
    assert sum(x < 0 for x in leading_minors) == 3
    assert sum(x < 0 for x in pivots) == 1

    grid = 16
    x = F(1, 3)
    lower = F((x * grid).__floor__(), grid)
    upper = F((x * grid).__ceil__(), grid)
    assert lower <= x <= upper and not lower <= x <= lower
    uncertainty = [[F(0), F(1, 16)], [F(1, 16), F(0)]]
    assert psd2(combine(times(identity, F(1, 16)), uncertainty, -1))
    assert form(uncertainty, [1, -1]) < 0  # A diagonal-only radius of zero fails.

    result = {
        "kind": "INDEPENDENT_EXACT_BOUNDARY_CONTROLS_NOT_PRODUCER_RERUN",
        "input_sha256": observed,
        "control_source_sha256": digest(Path(__file__)),
        "actual_support_control": {
            "sqrt_bits": 160, "input_support": len(support), "output_rows": len(outputs),
            "negative_free_columns": negative_columns, "all_density_images_contained": True,
            "output_only_support_missing_images": len(candidates),
            "deleted_input_state": victim, "deleted_state_missing_images": len(omitted),
            "outside_free_floor_lower_exact": str(exterior_floor),
            "wrong_exterior_budget_50_rejected": True,
        },
        "wrong_input_PHP": {"wrong_Loewner_witness": [-3, 1],
                            "full_minus_wrong_quadratic_exact": "-256",
                            "correct_output_row_difference_PSD": True},
        "schur_residual_control": {"K": k, "exact_inverse": exact,
            "fixed_non_solved_Y": y, "lower_and_upper_PSD": True,
            "residual_Frobenius_Gram_cap_PSD": True,
            "Schur_negative": 1, "full_negative": 1},
        "inertia_sign_control": {"leading_minors": leading_minors, "LDL_pivots": pivots,
                                 "wrong_minor_sign_count": 3, "correct_negative_count": 1},
        "interval_control": {"exact_input": x, "outward_endpoints": [lower, upper],
                             "inward_upper_rejected": True, "diagonal_only_radius_rejected": True},
        "native_binding_proved": False,
        "scientific_scope": "Specified corrected-input free/rho pair controls only; this script does not re-prove the separately corrected native embedding. No Ritz upper, ground, phase, photon or originalSOURCE admission.",
    }
    path = ROOT / "local_so5_positive_row_review_controls_20261011_receipt.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({"receipt": str(path), "receipt_sha256": digest(path),
                      "controls_passed": True, "input_unchanged": all(digest(ROOT / n) == h for n, h in PINS.items())}, indent=2))


if __name__ == "__main__":
    main()

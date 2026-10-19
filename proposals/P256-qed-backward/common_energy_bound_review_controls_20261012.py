"""Independent exact finite controls; neither producer nor native ground solver."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import zipfile
from collections import defaultdict
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def power(exponent: int) -> Q:
    return Q(2 ** exponent) if exponent >= 0 else Q(1, 2 ** (-exponent))


def compact(value: Q) -> dict[str, object]:
    return {"sha256": digest(str(value).encode()),
            "numerator_bits": value.numerator.bit_length(),
            "denominator_bits": value.denominator.bit_length()}


def canonical_controls() -> dict[str, object]:
    # M16/k8, three COMPLETE good four-colour cells and one COMPLETE bad cell.
    cells = ((-3, -1, 1, 3), (-4, 0, 1, 3), (-2, -1, 0, 3))
    eigenvalues = tuple(itertools.chain.from_iterable(cells))
    weights = (1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 3, 1)
    bad_values, bad_weights = (-100, -10, 10, 100), (1, 5, 7, 11)
    possibilities = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
    pairs = []
    matching_square = 0
    for cell, offset in zip(cells, (0, 4, 8)):
        scores = [sum((cell[i] - cell[j]) ** 2 for i, j in matching)
                  for matching in possibilities]
        assert Q(sum(scores), 3) == Q(4, 3) * sum(x * x for x in cell)
        matching = possibilities[max(range(3), key=lambda j: scores[j])]
        pairs.extend((i + offset, j + offset) for i, j in matching)
        matching_square += max(scores)
    field_square = sum(x * x for x in eigenvalues)
    lower = Q(field_square, 288)
    rows, tables = [], {}
    for n in range(4, 9):
        states = list(itertools.combinations(range(12), n))
        z = first = second = 0
        occupied = [0] * 12
        exchange = [[0] * 12 for _ in range(12)]
        totals = defaultdict(lambda: [0, 0, 0])
        for state in states:
            subset = set(state)
            weight = 1
            for i in state:
                weight *= weights[i]
            value = sum(eigenvalues[i] for i in state)
            z += weight
            first += weight * value
            second += weight * value * value
            for i in subset:
                occupied[i] += weight
                for j in range(12):
                    if j not in subset:
                        exchange[i][j] += weight
            key = tuple(int(i in subset) + int(j in subset) for i, j in pairs)
            totals[key][0] += weight
            totals[key][1] += weight * value
            totals[key][2] += weight * value * value
        mean, variance = Q(first, z), Q(second, z) - Q(first, z) ** 2
        min_exchange = min(Q(exchange[i][j], z) for i in range(12)
                           for j in range(12) if i != j)
        assert min_exchange >= Q(1, 144)
        assert min(occupied) * 3 >= max(occupied)
        for i in range(12):
            holes = [exchange[i][j] for j in range(12) if i != j]
            assert min(holes) * 3 >= max(holes)
        conditional_variance = sum((Q(second_t, z) - Q(first_t * first_t, z * z_t)
                                    for z_t, first_t, second_t in totals.values()), Q(0))
        independent_orientations = sum((Q(exchange[i][j] + exchange[j][i], z)
                                        * Q(weights[i] * weights[j], (weights[i] + weights[j]) ** 2)
                                        * (eigenvalues[i] - eigenvalues[j]) ** 2
                                        for i, j in pairs), Q(0))
        assert conditional_variance == independent_orientations
        assert variance >= conditional_variance >= Q(matching_square, 384) >= lower
        for i, j in pairs:
            assert Q(weights[i] * weights[j], (weights[i] + weights[j]) ** 2) >= Q(3, 16)
        tables[n] = (z, mean, variance)
        rows.append({"good_particles": n, "good_holes": 12 - n,
                     "all_good_configurations": len(states), "minimum_ordered_exchange": str(min_exchange),
                     "variance": str(variance), "conditional_pair_totals_variance": str(conditional_variance),
                     "matching_coarse_lower": str(Q(matching_square, 384)),
                     "good_field_variance_lower": str(lower)})
    total_z = 0
    branches = []
    for mask in range(16):
        n = 8 - mask.bit_count()
        bad_weight = 1
        bad_value = 0
        for i in range(4):
            if mask & (1 << i):
                bad_weight *= bad_weights[i]
                bad_value += bad_values[i]
        z, mean, variance = tables[n]
        branch_weight = bad_weight * z
        total_z += branch_weight
        branches.append((mask, n, branch_weight, mean + bad_value, variance))
    mean = sum((Q(w, total_z) * m for _, _, w, m, _ in branches), Q(0))
    within = sum((Q(w, total_z) * v for _, _, w, _, v in branches), Q(0))
    between = sum((Q(w, total_z) * (m - mean) ** 2 for _, _, w, m, _ in branches), Q(0))
    # Separate direct enumeration checks the complete compensating complement.
    full_first = full_second = full_z = 0
    for state in itertools.combinations(range(16), 8):
        all_weights = weights + bad_weights
        all_values = eigenvalues + bad_values
        weight = 1
        for i in state:
            weight *= all_weights[i]
        value = sum(all_values[i] for i in state)
        full_z += weight
        full_first += weight * value
        full_second += weight * value * value
    assert full_z == total_z
    assert Q(full_first, full_z) == mean
    assert Q(full_second, full_z) - mean ** 2 == within + between
    assert within >= lower and between >= 0
    return {"M": 16, "k": 8, "good_mode_count": 12, "bad_mode_count": 4,
            "good_weight_ratio_boundary": 3, "conditioned_good_rows": rows,
            "all_bad_branches": [{"bad_mask": mask, "good_n": n, "probability": str(Q(w, total_z))}
                                  for mask, n, w, _, _ in branches],
            "all_full_configurations": 12870, "full_variance": str(within + between),
            "restored_conditional_variance": str(within), "between_branch_variance": str(between),
            "opposite_pair_source_or_eligible_cropping": False}


def particle_hole_controls() -> dict[str, object]:
    values = (-31, -7, -2, -1, 1, 2, 7, 31)
    states = list(itertools.combinations(range(8), 4))
    energies = [sum(values[i] for i in state) for state in states]
    assert sorted(energies) == sorted(-e for e in energies)
    weights = [power(e) for e in energies]  # Exact tilt log2, deliberately outside small event.
    z = sum(weights, Q(0))
    mean = sum((w * e for w, e in zip(weights, energies)), Q(0)) / z
    assert mean > 0
    assert min(energies) < 0  # An arbitrarily selected negative branch is not the full trace.
    return {"complete_half_filled_states": len(states), "particle_hole_energy_symmetry": True,
            "large_field_positive_tilt_mean": str(mean), "cropped_negative_energy": min(energies),
            "no_good_event_needed_for_sign": True}


def generators() -> list[sp.Matrix]:
    result = []
    for i in range(4):
        for j in range(i + 1, 4):
            symmetric, antisymmetric = sp.zeros(4), sp.zeros(4)
            symmetric[i, j] = symmetric[j, i] = sp.Rational(1, 2)
            antisymmetric[i, j], antisymmetric[j, i] = -sp.I / 2, sp.I / 2
            result.extend((symmetric, antisymmetric))
    for k in range(1, 4):
        diagonal = [1] * k + [-k] + [0] * (3 - k)
        result.append(sp.diag(*diagonal) / sp.sqrt(2 * k * (k + 1)))
    return result


def trial_wick_control() -> dict[str, object]:
    basis = generators()
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            assert sp.simplify(sp.trace(a * b)) == (sp.Rational(1, 2) if i == j else 0)
    cross = diagonal = sp.Integer(0)
    for a in basis:
        for b in basis:
            cross += sp.trace((a * b + b * a) ** 2)
            diagonal += 4 * sp.trace(a * a * b * b + a * b * a * b + a * b * b * a)
    cross, diagonal = sp.simplify(cross), sp.simplify(diagonal)
    jordan = sp.simplify(4 * (4 * diagonal + 12 * cross))
    assert jordan == 3000
    assert 4 * 60 * sp.Rational(1, 4) == 60
    assert sp.Rational(3 * 4 * 15, 40) == sp.Rational(9, 2)
    assert sp.Rational(7, 10) * 3 * 4 == sp.Rational(42, 5)
    return {"independent_ordered_cross_mu_nu_Wick": str(cross),
            "independent_same_mu_Wick": str(diagonal), "ordered16_Jordan_per_Lambda": str(jordan),
            "Gaussian_KPhi_per_chi": 60, "alignment_per_F": "9/2", "Haar_colour_per_F": "42/5",
            "fullE_rho_zero_trial": "Every mode total32; each site128 minus128 background; fullRef cancels4",
            "charged_unconstrained_checkerboard_is_P_S": False}


def dyadic_controls(profiles: list[dict[str, Any]]) -> dict[str, object]:
    lam, chi, fast = power(8192), power(-4096), power(10240)
    loads = [p["load"] for p in profiles]
    assert loads == [1, 1, 63]
    def allocation(profile: dict[str, Any]) -> Q:
        return abs(Q(profile["onsite"])) + 2 * sum((abs(Q(t["c0"])) + abs(Q(t["ci"]))
                                                    for t in profile["paths"]), Q(0))
    b = 4 * (lam * allocation(profiles[0]) + allocation(profiles[1]) + 63 * allocation(profiles[2]))
    assert b < 512 * lam
    assert profiles[1]["paths"] == profiles[2]["paths"]
    delta = Q(profiles[1]["onsite"]) - Q(profiles[2]["onsite"])
    assert delta == -Q(126663739519795, 70368744177664) and abs(delta) < 2
    bs, c4 = power(10244), power(2053)
    trial = 60 * chi + 3000 * lam + Q(129, 10) * fast
    assert trial + 8192 * lam < bs and c4 == 2 * bs / lam
    a = Q(225) * 4 * chi / (8 * bs)
    assert a == Q(225) * power(-14341)
    r, tau = power(8196), power(-10252)
    lost = 3 * c4 / r ** 2
    assert lost < a / 2 and r ** 4 >= 4 * c4
    assert 4 * tau * r <= 1 and tau <= abs(delta) * a / (5760 * chi)
    gain = abs(delta) * a * tau / 288 - 10 * chi * tau ** 2
    gamma = power(-24595)
    assert gain >= abs(delta) * a * tau / 576 > gamma
    # Find the exact neighbouring dyadic accept/reject boundary, not just a green selected value.
    r_exp = 8196
    while power(2 * (r_exp - 1)) >= 6 * c4 / a and power(4 * (r_exp - 1)) >= 4 * c4:
        r_exp -= 1
    assert power(2 * r_exp) >= 6 * c4 / a and power(4 * r_exp) >= 4 * c4
    assert power(2 * (r_exp - 1)) < 6 * c4 / a or power(4 * (r_exp - 1)) < 4 * c4
    tau_exp = -10252
    tau_limit = min(1 / (4 * r), abs(delta) * a / (5760 * chi))
    while power(tau_exp + 1) <= tau_limit:
        tau_exp += 1
    assert power(tau_exp) <= tau_limit < power(tau_exp + 1)
    gain_exp = -24595
    while power(gain_exp + 1) < abs(delta) * a * tau / 576:
        gain_exp += 1
    assert power(gain_exp) < abs(delta) * a * tau / 576 <= power(gain_exp + 1)
    assert Q(128) ** 2 < power(15) < Q(256) ** 2
    coeff, actual = Q(64, 255), Q(256)
    complement = 16 * gamma ** 2 / (delta ** 2 * coeff * actual)
    assert complement > power(-49194) > power(-49195)
    eps_exp = -49195
    while power(eps_exp + 1) < complement:
        eps_exp += 1
    assert power(eps_exp) < complement <= power(eps_exp + 1)
    assert abs(delta) < Q(9, 5) and delta ** 2 < Q(255, 64)
    delta_adverse = Q(1999, 1000)
    assert abs(delta_adverse) < 2
    assert 16 / (delta_adverse ** 2 * coeff * actual) < Q(1, 16)
    assert Q(255, 4096) < Q(1, 16) and Q(255, 4096) > Q(1, 32)
    for n in (4, 6, 8, 10, 100):
        volume, m = n ** 3, 4 * n ** 3
        canonical = Q(m, 4 * (m - 1))
        assert canonical <= coeff and volume >= 64
        assert Q(volume, 4) * gamma ** 2 / (delta ** 2 * canonical * actual) >= complement
    # Componentwise Cauchy sums before taking the square root, including distinct fibres.
    ws, moments, probabilities = (Q(2, 3), Q(1, 8), Q(4, 5)), (Q(70), Q(900), Q(20)), (Q(1, 5), Q(3, 10), Q(1, 2))
    assert sum(probabilities) == 1
    cs_lhs_sq = sum((probabilities[i] * sp.sqrt(moments[i] * (1 - ws[i])) for i in range(3)), sp.Integer(0)) ** 2
    cs_rhs = sum((probabilities[i] * moments[i] for i in range(3)), Q(0)) * sum((probabilities[i] * (1 - ws[i]) for i in range(3)), Q(0))
    assert sp.simplify(cs_rhs - cs_lhs_sq) > 0
    return {"native_B_over_Lambda": compact(b / lam), "delta": str(delta),
            "own_B_S": "2^10244", "own_C4_S": "2^2053", "uncertainty_a": "225*2^-14341",
            "selected_R": "2^8196", "selected_tau": "2^-10252", "selected_gamma": "2^-24595",
            "selected_epsilon": "2^-49195", "selected_loss_over_a": str(lost / a),
            "smallest_accepted_dyadic_R_exponent": r_exp, "largest_accepted_dyadic_tau_exponent": tau_exp,
            "largest_dyadic_conservative_gain_exponent": gain_exp, "largest_dyadic_block_complement_exponent": eps_exp,
            "each_next_boundary_rejected": True, "C_actual_smallest_dyadic_upper": 256,
            "weaker_delta_bound_stronger_dyadic_implication_rejected": True,
            "exact_pinned_delta_stronger_dyadic_bound_paid": True,
            "conservative_gain": compact(abs(delta) * a * tau / 576), "block_complement": compact(complement),
            "all_even_N_monotonic_argument": "V=N^3>=64 and c=M/(4(M-1)) decreasing for M=4V>=256",
            "mixed_component_Cauchy_with_unequal_moments": True, "volume_units": "gamma is per-site; gamma*V is energy gap"}


def closed_mode_countercontrol() -> dict[str, object]:
    # Independently rederived matrix-potential method. Never execute the parent source.
    kinetic, coupling = sp.Integer(256), sp.Integer(1024)
    hamiltonian = sp.Matrix([[kinetic, -coupling, 0], [-coupling, kinetic, 0], [0, 0, kinetic - coupling]])
    ground = sp.Matrix([1, 1, 0]) / sp.sqrt(2)
    other = sp.Matrix([0, 0, 1])
    excited = sp.Matrix([1, -1, 0]) / sp.sqrt(2)
    p0 = ground * ground.H + other * other.H
    q0 = sp.eye(3) - p0
    electric = sp.diag(-4, 4, 0)
    transfer = sp.zeros(3)
    transfer[2, 0] = 1
    source = transfer + transfer.H
    assert hamiltonian * ground == -768 * ground and hamiltonian * other == -768 * other
    assert electric * transfer - transfer * electric == 4 * transfer
    assert p0 * electric * p0 == sp.zeros(3)
    inverse = (ground.H * electric * (excited * excited.H / 2048) * electric * ground)[0]
    assert inverse == sp.Rational(1, 128) <= sp.Rational(1, 64)
    assert (ground.H * source * ground)[0] == 0
    assert (ground.H * source ** 2 * ground)[0] == sp.Rational(1, 2)
    assert q0 * source * ground == sp.zeros(3, 1)
    assert q0 * source * other != sp.zeros(3, 1)
    assert (sp.eye(3) - ground * ground.H) * source * ground != sp.zeros(3, 1)
    k = sp.Symbol("k", real=True)
    polynomial = sp.expand((coupling + kinetic * k ** 2) ** 2 - (coupling ** 2 + 4 * kinetic ** 2 * k ** 2))
    assert polynomial == 65536 * k ** 4 + 262144 * k ** 2
    return {"method_not_original_law": True, "all_real_character_lower_square_difference": str(polynomial),
            "third_channel_energy_difference": "256*k^2", "minimum_energy": -768,
            "P0_Ec_P0": 0, "Ec_excited_inverse": str(inverse), "face_upper": "1/64",
            "source_mean": 0, "total_source_Gram": "1/2", "whole_P0_excited_source_Gram": 0,
            "other_ground_channel_excitation_nonzero": True, "mean_subtraction_not_whole_P0": True,
            "complement_or_total_Gram_to_IR_lower_rejected": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    captured = {name: (ROOT / name).read_bytes() for name in request["pins"]}
    for name, raw in captured.items():
        assert digest(raw) == request["pins"][name], name
    with zipfile.ZipFile(ROOT / request["profile_archive"]) as archive:
        raw = archive.read(request["profile_member"])
    result = {"kind": "INDEPENDENT_EXACT_ENERGY_CANONICAL_BOUNDARY_AND_NONCLAIM_CONTROLS",
              "trial_wick": trial_wick_control(), "canonical_complete_complement": canonical_controls(),
              "particle_hole_outside_good_event": particle_hole_controls(),
              "dyadic_constants": dyadic_controls(json.loads(raw)["primitive_profiles"]),
              "closed_mode_ground_channel_nonclaim": closed_mode_countercontrol(),
              "profile_member_sha256": digest(raw), "pins": request["pins"],
              "native_ground_or_phase_or_IR_atom": False, "original_producer_rerun": False}
    assert all((ROOT / name).read_bytes() == raw for name, raw in captured.items())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Independent full-complement and dyadic boundary controls completed")


if __name__ == "__main__":
    main()

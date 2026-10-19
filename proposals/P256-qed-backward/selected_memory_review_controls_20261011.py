"""Independent exact small-matrix controls; never import/run the selected solver."""
from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = "selected_memory_review_"
Vector = list[F]
Matrix = list[Vector]


def dot(a: Vector, b: Vector) -> F:
    return sum((x * y for x, y in zip(a, b)), F(0))


def mv(a: Matrix, x: Vector) -> Vector:
    return [dot(row, x) for row in a]


def outer(x: Vector) -> Matrix:
    return [[a * b for b in x] for a in x]


def add(a: Matrix, b: Matrix) -> Matrix:
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def sub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


def scale(c: F, x: Vector) -> Vector:
    return [c * y for y in x]


def solve(a: Matrix, b: Vector) -> Vector:
    rows = [row[:] + [rhs] for row, rhs in zip(a, b)]
    n = len(b)
    for k in range(n):
        pivot = next(i for i in range(k, n) if rows[i][k])
        rows[k], rows[pivot] = rows[pivot], rows[k]
        rows[k] = [x / rows[k][k] for x in rows[k]]
        for i in range(n):
            if i != k:
                factor = rows[i][k]
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[k])]
    return [row[-1] for row in rows]


def quadratic(a: Matrix, b: Vector) -> F:
    return dot(b, solve(a, b))


def small_controls() -> dict[str, Any]:
    omega = [F(3, 5), F(4, 5), F(0)]
    excited = [F(-4, 5), F(3, 5), F(0)]
    h = [[2 * x for x in row] for row in outer(excited)]
    h[2][2] = F(5)
    rho = [[F(-1), F(0), F(0)], [F(0), F(1), F(0)], [F(0), F(0), F(0)]]
    lift = add(h, outer(omega))
    raw = mv(rho, omega)
    mean = dot(omega, raw)
    b = sub(raw, scale(mean, omega))
    alpha = quadratic(lift, b)
    density_zero_complement_source = raw[:]
    wrong_density_response = quadratic(lift, density_zero_complement_source)
    assert dot(omega, b) == 0 and alpha == F(288, 625)
    assert wrong_density_response - alpha == mean**2 > 0
    proxy = [F(5, 13), F(12, 13), F(0)]
    proxy_b = sub(mv(rho, proxy), scale(dot(proxy, mv(rho, proxy)), proxy))
    proxy_response = quadratic(add(h, outer(proxy)), proxy_b)
    assert proxy_response != alpha and dot(proxy, mv(h, proxy)) > 0

    s = F(1001, 1000)
    v = scale(s, omega)
    norm_defect = dot(v, v) - 1
    unnormalized_lift = add(h, outer(v))
    lift_defect_on_ground = dot(omega, sub(mv(unnormalized_lift, omega), mv(lift, omega)))
    approximate_b = sub(mv(rho, v), scale(dot(v, mv(rho, v)), v))
    assert lift_defect_on_ground == norm_defect > 0
    assert quadratic(unnormalized_lift, approximate_b) != alpha
    insertion_defect = sub(approximate_b, b)
    assert 0 < dot(insertion_defect, insertion_defect) <= (1024 * norm_defect)**2

    a = [[F(3), F(1)], [F(1), F(2)]]
    rhs = [F(1), F(-1)]
    x = [F(float(F(1, 3))), F(float(F(-1, 4)))]
    residual = sub(rhs, mv(a, x))
    form = 2 * dot(rhs, x) - dot(x, mv(a, x))
    exact_response = quadratic(a, rhs)
    exact_solve_error = quadratic(a, residual)
    assert exact_response - form == exact_solve_error > 0
    assert exact_solve_error <= dot(residual, residual)  # A >= I.
    terms = [1e16, 1.0, -1e16]
    binary_accumulator = 0.0
    for term in terms:
        binary_accumulator += term  # Explicit binary64 additions, not Python 3.12 compensated sum.
    binary_sum = F(binary_accumulator)
    rational_sum = sum((F(z) for z in terms), F(0))
    gamma3 = F(3, 2**53 - 3)
    dot_rounding_cap = gamma3 * sum((abs(F(z)) for z in terms), F(0))
    assert abs(binary_sum - rational_sum) <= dot_rounding_cap
    assert abs(binary_sum - rational_sum) > F(1, 10**7)

    eps = F(1, 1000)
    exact_a = [[F(1), F(0)], [F(0), F(2)]]
    perturbed_a = [[1 + eps, eps], [eps, 2 - eps]]
    delta_a = 2 * eps  # Absolute row/column sum, not just diagonal error.
    amin = 1 - delta_a
    exact_b = [F(3, 5), F(4, 5)]
    perturbed_b = [exact_b[0] + eps, exact_b[1] - eps]
    delta_b = 2 * eps
    response_error = abs(quadratic(exact_a, exact_b) - quadratic(perturbed_a, perturbed_b))
    price = 2 * delta_b * (1 + delta_b) / amin + (1 + delta_b)**2 * delta_a / amin
    assert response_error <= price
    assert abs(quadratic(exact_a, exact_b) - quadratic(perturbed_a, exact_b)) > 0

    contact_b = [F(0), F(1, 128)]
    contact_lift = [[F(1), F(0)], [F(0), F(2)]]
    contact_alpha = quadratic(contact_lift, contact_b)
    curvature = 128 - 32768 * contact_alpha
    assert contact_alpha == F(1, 32768) and curvature == 127
    with localcontext() as ctx:
        ctx.prec = 80
        t = Decimal('1e-8')
        # Actual eigenvalue of [[64t²,t],[t,2+64t²]], not a response-form echo.
        e = 64 * t*t + 1 - (1 + t*t).sqrt()
        finite_difference = 2 * e / (t*t)
        assert abs(finite_difference - Decimal(127)) < Decimal('1e-14')
        assert abs(finite_difference - Decimal(129)) > 1
        assert abs(finite_difference - Decimal(63)) > 1
        positive_coupling_e = 64 * t*t + 1 - (1 + (-t)*(-t)).sqrt()
        assert e == positive_coupling_e

    # Ignoring normalization can falsify the spectral projection bound.
    tiny_v = [F(6, 13), F(5, 26)]  # Half of normalized (12/13,5/13).
    center = F(0)
    residual_squared = 4 * tiny_v[1]**2
    actual_infidelity = F(25, 169)
    wrong_projection_cap = residual_squared / 4
    normalized_projection_cap = residual_squared / dot(tiny_v, tiny_v) / 4
    assert wrong_projection_cap < actual_infidelity <= normalized_projection_cap
    assert center == 0
    return {
        "whole_ground_vs_density_zero": {"alpha": str(alpha), "density_zero_proxy": str(wrong_density_response), "mean_only_nonstationary_proxy": str(proxy_response)},
        "lift_normalization": {"norm_defect": str(norm_defect), "exact_rank_one_lift_defect": str(lift_defect_on_ground)},
        "solve_variational_identity": {"exact_response": str(exact_response), "form": str(form), "exact_error": str(exact_solve_error), "residual_squared_cap": str(dot(residual, residual))},
        "actual_binary64_dot_cancellation": {"binary_sum": str(binary_sum), "exact_sum": str(rational_sum), "derived_cap": str(dot_rounding_cap), "unpriced_1e_minus7_rejected": True},
        "inverse_and_insertion": {"actual_error": str(response_error), "price": str(price)},
        "contact_coupling": {"alpha": str(contact_alpha), "curvature": str(curvature), "actual_eigenvalue_finite_difference": str(finite_difference), "wrong_positive_response_sign_and_half_contact_rejected": True, "linear_coupling_sign_invariance": True},
        "normalization_contrary": {"true_infidelity": str(actual_infidelity), "wrong_cap": str(wrong_projection_cap), "normalized_cap": str(normalized_projection_cap)},
    }


def scalar_review(receipt: dict[str, Any], captured: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    dimension = receipt["complete_dimension"]
    assert dimension == 9672320
    u = F(1, 2**53)
    gamma = (2 * dimension + 8) * u / (1 - (2 * dimension + 8) * u)
    assert gamma < F(1, 2**27)
    for item, exact in zip(captured["parameters"], frame["profiles"]):
        assert max(abs(F(item[k]) - F(exact[j])) for k, j in [("onsite_float", "m_exact"), ("c0", "c0_exact"), ("ci", "d_exact")]) < F(1, 2**48)
        assert abs(F(exact["m_exact"])) < 20 and max(abs(F(exact[k])) for k in ["c0_exact", "d_exact"]) < 6
    rowcap = 64 * 128**2 + 2 * 20 * 128 + 16 * (2 * 6 * 128)
    assert rowcap == 1078272 and rowcap < 2**22
    underflow = F(100 * dimension, 2**1074) + F(2, 2**490)
    mu = F(receipt["mu_float"])
    formation = 64 * u / (1 - 64*u) * 2**22 + F(768, 2**48)
    accumulation = 40 * u / (1 - 40*u) * 2**22
    subtraction = 4*u*(2**22 + abs(mu))
    action_error = formation + accumulation + subtraction + underflow
    sigma = F(1, 2**22)
    residual_upper = (F(receipt["full_source_residual_float"]) + action_error) / (1 - 4*gamma)
    assert action_error < F(1, 2**24) and residual_upper < sigma
    gap = F(-5123) - mu - sigma
    assert gap == F(receipt["gap_lower_exact"]) and gap > 1
    bounds = {key: F(value) for key, value in receipt["derived_response_error_bounds"].items()}
    defect = bounds["norm_defect"]
    # Recover the exact observed norm scalar and xnorm upper from two audited budgets.
    # This is scalar consistency only; it is not independent vector measurement.
    norm_observed = (defect + 1 - underflow)*(1 - gamma)
    assert norm_observed == F(4503599627370497, 4503599627370496)
    negative_branch_norm = (1 + underflow - defect)*(1 - gamma)/(1 - 2*gamma)
    assert negative_branch_norm.denominator & (negative_branch_norm.denominator - 1)
    assert defect < 2*gamma
    assert (1 - defect)*(1 - 2*gamma) > 1 - 4*gamma
    delta_a = sigma + sigma*sigma/gap + 2*sigma/gap + defect
    delta_b = 384*sigma/gap + 1024*defect + 512*gamma
    assert delta_a == bounds["rank_one_ground_lift_error"] and delta_a < F(1, 1000)
    assert delta_b == bounds["source_insertion_error"]
    amin = 1 - delta_a
    lift_action_error = action_error + 16*gamma
    xu_squared = (bounds["variational_form_error"] - 2*gamma - underflow)/(gamma*2**23 + lift_action_error)
    xn, xd = math.isqrt(xu_squared.numerator), math.isqrt(xu_squared.denominator)
    assert xn*xn == xu_squared.numerator and xd*xd == xu_squared.denominator
    xu = F(xn, xd)
    assert 0 < xu < 1 and F(receipt["rho_connected_weight_float"])/(1 - 4*gamma) < 1
    solve_residual_upper = F(receipt["complete_projected_resolvent_residual_float"])/(1 - 4*gamma) + lift_action_error*xu + u*(1 + 2**23*xu) + underflow
    assert solve_residual_upper == bounds["solve_residual_upper"]
    state_error = 2*delta_b*(1 + delta_b)/amin + (1 + delta_b)**2*delta_a/amin
    assert state_error == bounds["state_response_error"]
    variational = F(receipt["source_resolvent_quadratic_float"])
    alo = variational - bounds["variational_form_error"] - state_error
    ahi = variational + bounds["variational_form_error"] + solve_residual_upper**2/amin + state_error
    assert [alo, ahi] == [F(v) for v in receipt["source_resolvent_interval_exact"]]
    curvature = [128 - 32768*ahi, 128 - 32768*alo]
    assert curvature == [F(v) for v in receipt["incoming_electric_ground_curvature_interval_exact"]]
    assert curvature[0] > 0
    return {"gamma_exact": str(gamma), "native_row_and_column_cap": rowcap, "action_error": str(action_error), "normalized_residual_upper": str(residual_upper), "sigma": str(sigma), "gap": str(gap), "recovered_norm_float_scalar_exact": str(norm_observed), "recovered_xnorm_upper_exact": str(xu), "budget_floats": {k: float(v) for k, v in bounds.items()}, "alpha_interval_float": [float(alo), float(ahi)], "curvature_interval_float": [float(v) for v in curvature], "scalar_consistency_not_independent_full_vector_measurement": True}


def main() -> None:
    own = Path(__file__).read_bytes()
    input_path = ROOT / (PREFIX + "inputs_20261011.json")
    inputs = json.loads(input_path.read_bytes())
    pinned = inputs["file_sha256"]
    for name, digest in pinned.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    archive = ROOT / inputs["frozen_executed_archive"]
    with zipfile.ZipFile(archive) as z:
        producer_name = "local_so5_ground_electric_memory_consumer_20261011.py"
        assert hashlib.sha256(z.read(producer_name)).hexdigest() == inputs["producer_sha256"]
        receipt = json.loads(z.read("local_so5_ground_electric_memory_consumer_20261011_receipt.json"))
        captured = json.loads(z.read("local_so5_resum_actual04_20261011_receipt.json"))
        frame = json.loads(z.read("local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json"))
        archived_exit = json.loads(z.read("local_so5_ground_electric_memory_outward_response_attempt3_20261011_exit.json"))
        assert archived_exit["returncode"] == 0 and archived_exit["natural_exit"] is True
    result = {
        "verdict": "VALID",
        "scope": inputs["scope"],
        "assumptions": inputs["assumptions"],
        "analytic_review": inputs["analytic_review"],
        "independent_controls": small_controls(),
        "exact_scalar_budget_review": scalar_review(receipt, captured, frame),
        "producer_sha256": inputs["producer_sha256"],
        "directly_executed": "This new standard-library rational/small-matrix CLI only; no native/9.6M solver, old checker, or parent consumer was imported or run.",
        "parent_execution_evidence": archived_exit,
        "local_positive_response_NOT_bulk_stiffness": True,
        "ground_below_gap_weight_probe_not_run": receipt["ground_below_gap_weight_probe_not_run"],
        "moments": "First/double-commutator and current-weight fields remain binary64 diagnostics, not exact moments or spectral constraints. No pointwise current cap is claimed.",
        "not_admitted": ["other canonical representations", "fullPhi/Gauss stationary ground", "intercell/lattice ground", "bulk stiffness", "N-uniform linked stability", "global bounded current", "IR atom", "spectral pole", "photon dispersion", "campaign"],
        "source_sha256": hashlib.sha256(own).hexdigest(),
        "inputs_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "all_inputs_unchanged": all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in pinned.items()),
        "source_unchanged": Path(__file__).read_bytes() == own,
    }
    assert result["all_inputs_unchanged"] and result["source_unchanged"]
    output = ROOT / (PREFIX + "epoch2_proof_20261011.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

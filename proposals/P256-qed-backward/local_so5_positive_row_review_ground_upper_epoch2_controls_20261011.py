"""NEW changed-input exact control epoch; no old action/solver execution."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PINS = {
    "local_so5_ground_upper_complete_state_consumer_20261011.py": "7ec61306fb52989a58008b488a13d9a9850d0e14688f8e21ee3d9ac5432f6769",
    "local_so5_ground_upper_complete_state_consumer_20261011_receipt.json": "8ac548432a5184ed1a281af1d2052fd9693c05399bc5095da9ccc882e215650d",
    "local_so5_ground_upper_complete_state_guard_epoch2_20261011_executed.zip": "a80ce9729ec3f9a0647d478984903c2235e9c5180a870ea161a63a6835058b71",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    observed = {name: digest(ROOT / name) for name in PINS}
    assert observed == PINS
    r = json.loads((ROOT / "local_so5_ground_upper_complete_state_consumer_20261011_receipt.json").read_text())
    dimension = r["source_dimension"]
    assert dimension == 9672320
    u = F(1, 2**53)
    gamma = (2 * dimension + 8) * u / (1 - (2 * dimension + 8) * u)
    mu = F(r["energy_float"])
    formation = 64 * u / (1 - 64 * u) * 2**22 + F(768, 2**48)
    accumulation = 40 * u / (1 - 40 * u) * 2**22
    subtraction = 4 * u * (2**22 + abs(mu))
    underflow = F(100 * dimension, 2**1074)
    # A radicand's absolute subnormal error can be amplified by sqrt.
    # sqrt(epsilon)<=2^-522 here;2^-490 covers BOTH norm ratio paths
    # with the observed source norm near1 and no tiny denominator.
    norm_subnormal_allowance = F(1, 2**490)
    assert underflow < F(1, 2**1044)
    assert F(1, 2**522)**2 == F(1, 2**1044)
    assert abs(F(r["norm_float"]) - 1) < F(1, 10**10)
    combined = formation + accumulation + subtraction + underflow + norm_subnormal_allowance
    assert combined < F(1, 2**18)
    assert 4 * gamma < 1
    computed = F(r["independent_full_H_residual_float"])
    exact_residual_cap = computed / (1 - 4 * gamma) + F(1, 2**18)
    sigma = F(1, 2**16)
    assert exact_residual_cap < sigma
    lower, upper = mu - sigma, mu + sigma
    separation = F(-5123) - upper
    temple = lower - sigma**2 / separation
    infidelity = sigma**2 / separation**2
    assert [F(x) for x in r["ground_energy_interval_exact"]] == [temple, upper]
    assert F(r["selected_ground_projector_error_upper_exact"]) == infidelity
    assert F(5) - F(25) / (10 - 5) == 0
    assert F(5) - F(5) / (10 - 5) == 4
    assert F(1, 5) - F(9, 25) / (2 - F(1, 5)) == 0
    assert F(1, 5) - F(9, 25) / 2 == F(1, 50)
    assert F(9, 100) < F(1, 10) <= F(1, 9)
    result = {
        "kind": "NEW_EPOCH2_FINAL_GUARD_EXACT_CONTROL",
        "input_sha256": observed, "source_sha256": digest(Path(__file__)),
        "gamma_exact": gamma, "norm_ratio_gamma_multiplier": 4,
        "action_subnormal_budget_exact": underflow,
        "additional_norm_subnormal_budget_exact": norm_subnormal_allowance,
        "guard_including_both_subnormal_paths_float": float(combined),
        "normalized_residual_cap_exact": exact_residual_cap,
        "normalized_residual_cap_float": float(exact_residual_cap),
        "sigma_exact": sigma, "ground_energy_interval_exact": [temple, upper],
        "selected_gap_lower_exact": separation, "infidelity_upper_exact": infidelity,
        "norm_as_variance_contrary_control_rejected": True,
        "wrong_Temple_and_fidelity_denominator_rejected": True,
        "state_loaded_or_action_run": False,
    }
    out = ROOT / "local_so5_positive_row_review_ground_upper_epoch2_controls_20261011_receipt.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({"receipt": str(out), "receipt_sha256": digest(out),
                      "controls_passed": True, "input_unchanged": all(digest(ROOT / n) == h for n, h in PINS.items())}, indent=2))


if __name__ == "__main__":
    main()

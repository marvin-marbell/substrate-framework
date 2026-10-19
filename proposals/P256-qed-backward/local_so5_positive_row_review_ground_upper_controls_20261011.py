"""Exact guard arithmetic and contrary Temple controls; no state/producer execution."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PINS = {
    "local_so5_ground_upper_complete_state_consumer_20261011.py": "7715337d20651610d2ed73f3c56f1f7c1d5b23543fe0d2b093424fa47e4d6bb2",
    "local_so5_ground_upper_complete_state_consumer_20261011_receipt.json": "b3c60f0df1994942db53e754dd6a0927784ee1720f918f0ef5ad0b2d813f67f2",
    "local_so5_positive_output_row_inertia_certificate_20261011_receipt.json": "f6c8823eb6b5005cdb8067088d14fb561cdacbd3bba9c54e0c5042fe160e4e73",
    "local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json": "102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8",
    "local_so5_resum_actual04_20261011_receipt.json": "a31c2d5be2a368b7de703d92461e230af59adf7e8ee8f4eaca0944fe4f556fc1",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    observed = {name: digest(ROOT / name) for name in PINS}
    assert observed == PINS
    r = json.loads((ROOT / "local_so5_ground_upper_complete_state_consumer_20261011_receipt.json").read_text())
    parameters = json.loads((ROOT / "local_so5_resum_actual04_20261011_receipt.json").read_text())["parameters"]
    frame = json.loads((ROOT / "local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json").read_text())
    assert [p["spin_replicas"] for p in parameters] == [2, 126]
    assert [p["spin_replicas"] for p in frame["profiles"]] == [2, 126]
    assert all(abs(F(p["onsite_float"])) < 20 and abs(F(p["c0"])) < 6
               and abs(F(p["ci"])) < 6 for p in parameters)
    parameter_deltas = []
    for p, exact in zip(parameters, frame["profiles"]):
        delta = [abs(F(p[a]) - F(exact[b])) for a, b in
                 [("onsite_float", "m_exact"), ("c0", "c0_exact"), ("ci", "d_exact")]]
        assert max(delta) < F(1, 2**48)
        parameter_deltas.append(delta)
    dimension = 14 * 690880
    assert dimension == r["source_dimension"] == 9672320
    unit = F(1, 2**53)
    gamma = (2 * dimension + 8) * unit / (1 - (2 * dimension + 8) * unit)
    rowcap = 64 * 128**2 + 2 * 20 * 128 + 16 * (2 * 6 * 128)
    assert rowcap == 1078272 and rowcap < 2**22
    mu = F(r["energy_float"])
    formation = 64 * unit / (1 - 64 * unit) * 2**22 + F(128 * 6, 2**48)
    accumulation = 40 * unit / (1 - 40 * unit) * 2**22
    subtraction = 4 * unit * (2**22 + abs(mu))
    guard = formation + accumulation + subtraction
    assert guard < F(1, 2**18) and gamma < F(1, 2**27)
    computed = F(r["independent_full_H_residual_float"])
    residual_guard = computed / (1 - gamma) + F(1, 2**18)
    sigma = F(1, 2**16)
    assert residual_guard < sigma
    lower, upper = mu - sigma, mu + sigma
    separation = F(-5123) - upper
    temple_lower = lower - sigma * sigma / separation
    fidelity = sigma * sigma / separation**2
    assert separation > 0
    assert [F(x) for x in r["Rayleigh_interval_exact"]] == [lower, upper]
    assert [F(x) for x in r["ground_energy_interval_exact"]] == [temple_lower, upper]
    assert F(r["selected_ground_gap_lower_exact"]) == separation
    assert F(r["selected_ground_projector_error_upper_exact"]) == fidelity

    # H=diag(0,10), equal spectral weights: mean5, residual norm5,
    # variance25. Substituting the norm for variance falsely claims E0>=4.
    mean, beta, residual_norm, variance = F(5), F(10), F(5), F(25)
    correct = mean - variance / (beta - mean)
    wrong_norm = mean - residual_norm / (beta - mean)
    assert correct == 0 and wrong_norm == 4

    # H=diag(0,2), excited weight1/10: mean1/5, variance9/25.
    # Dropping the Rayleigh mean from the denominator claims E0>=1/50>0
    # and infidelity<=9/100<the actual1/10.
    mean, beta, variance, excited_weight = F(1, 5), F(2), F(9, 25), F(1, 10)
    correct_temple = mean - variance / (beta - mean)
    wrong_denominator = mean - variance / beta
    correct_infidelity_cap = variance / (beta - mean)**2
    wrong_infidelity_cap = variance / beta**2
    assert correct_temple == 0 and wrong_denominator == F(1, 50)
    assert wrong_infidelity_cap < excited_weight <= correct_infidelity_cap

    result = {
        "kind": "INDEPENDENT_SUPPLEMENTAL_GUARD_AND_TEMPLE_CONTROLS_NO_STATE_RERUN",
        "input_sha256": observed,
        "source_sha256": digest(Path(__file__)),
        "state_loaded_or_action_run": False,
        "parameter_float_differences_exact": parameter_deltas,
        "gamma_exact": gamma, "row_column_cap": rowcap,
        "formation_guard_exact": formation, "accumulation_guard_exact": accumulation,
        "subtraction_guard_exact": subtraction, "sum_guard_exact": guard,
        "sum_guard_float": float(guard), "normalized_residual_guard_exact": residual_guard,
        "normalized_residual_guard_float": float(residual_guard), "claimed_sigma_exact": sigma,
        "Rayleigh_interval_exact": [lower, upper],
        "ground_energy_interval_exact": [temple_lower, upper],
        "separation_exact": separation, "fidelity_error_upper_exact": fidelity,
        "norm_as_variance_control": {"true_ground": 0, "correct_Temple_lower": correct,
            "wrong_norm_Temple_lower": wrong_norm, "wrong_rejected": True},
        "wrong_denominator_control": {"true_ground": 0, "correct_Temple_lower": correct_temple,
            "wrong_Temple_lower": wrong_denominator, "true_infidelity": excited_weight,
            "correct_infidelity_cap": correct_infidelity_cap,
            "wrong_infidelity_cap": wrong_infidelity_cap, "wrong_rejected": True},
        "binary64_correct_rounding_and_exact_representation_assumed": True,
        "fullPhi_lattice_ground_admitted": False,
    }
    out = ROOT / "local_so5_positive_row_review_ground_upper_controls_20261011_receipt.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({"receipt": str(out), "receipt_sha256": digest(out),
                      "controls_passed": True, "input_unchanged": all(digest(ROOT / n) == h for n, h in PINS.items())}, indent=2))


if __name__ == "__main__":
    main()

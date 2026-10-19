"""Bounded old-law quantum and NEW Jordan decisions; no ground/pole adoption."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import joined_phi_u1_stationary_canonical_photon as canonical

ROOT = Path(__file__).resolve().parent
PLAN = "issue-222-induced-photon-delta13-continuation-20261008"
REVISION = "6079658737e4b1fcb93e8ac99b7e10e01ae0bb6ea7929cf5102d2e3b51e3e786"
INPUTS = {
    "unitary": ("joined_phi_u1_stationary_unitary_ground_final_receipt_20261010.json", "5ee90b0890dc8022dbc62e31fd4d9db88e5700462e5ac5f4f8fb54f7a1d078e6"),
    "old_flat": ("joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json", "1c79b1a74c4236079645ebf9765a48c1c2263ec7b7209e9f103704c98752da94"),
    "quantum": ("joined_phi_u1_noncentral_quantum_proof_actual_receipt_20261010.json", "f2180e8a96f389a6c7744d3acfc112df2175c65f4db3694c3807f1392c63dc6f"),
    "jordan": ("joined_phi_u1_matrix_jordan_supplier_receipt_20261010.json", "24e0e37132706552a3b7925ffaf0078049a564048fba04d2bc98cab279675394"),
    "jordan_flat": ("joined_phi_u1_matrix_jordan_flat_actual_20261010_receipt.json", "d3d0360739b6d1d12ac013bac65aafcb085004fa76f7d379445cbe8548d18e24"),
    "canonical": ("joined_phi_u1_stationary_canonical_photon_divisionfree_actual_receipt_20261010.json", "245d09c5c4c008cc5d16ee28a8015e5050f69ac0d8023389dece469c2d0fef96"),
    "cut3": ("covariant_collective_cut3_delivery_receipt_20261010.json", ""),
}
ARRAY = "joined_phi_u1_matrix_jordan_supplier_receipt_20261010.npz"
ARRAY_SHA = "e59b152a390f413b84bb5ea6d3d9578478b997e2356c5bae4b513de2678ce94e"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hermitian(a: NDArray[Any]) -> NDArray[Any]:
    return (a + a.conj().T) / 2


def _principal_covariance(k: NDArray[Any]) -> NDArray[Any]:
    # Physical q0 Phi45 PRINCIPAL only, divided by its common 2^-6144.
    # G_phi=8I; A G A=K and C=.5 A^-1, not .5 A^-1/2.
    values, vectors = np.linalg.eigh(_hermitian(k))
    if float(values[0]) <= 0:
        raise ValueError("Principal stiffness is not strictly positive")
    return (vectors * (np.sqrt(8.0 / values) / 2)) @ vectors.conj().T


def main() -> None:
    sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    start = time.monotonic()
    inputs: dict[str, Any] = {}
    before = {name: _sha(ROOT / name) for name, _ in INPUTS.values()}
    before[ARRAY] = _sha(ROOT / ARRAY)
    before[canonical.__file__] = _sha(Path(canonical.__file__))
    own_before = _sha(Path(__file__))
    for key, (name, expected) in INPUTS.items():
        if expected and before[name] != expected:
            raise ValueError(f"Changed selected input: {name}")
        inputs[key] = json.loads((ROOT / name).read_text())
    if before[ARRAY] != ARRAY_SHA:
        raise ValueError("Changed physical matrix input")
    if before[canonical.__file__] != "d2f33bdfbaae16c89bbc9d58f9c2bb636546a300e13378a7dfee9cb3276f7289":
        raise ValueError("Changed canonical implementation")
    # The actual producer archived its bytes, though the callback capsule's
    # science_input_archive is null. Bind the real archive explicitly, not by
    # backfilling either historical capsule.
    canonical_zip = ROOT / "joined_phi_u1_stationary_canonical_photon_divisionfree_actual_receipt_20261010.zip"
    with zipfile.ZipFile(canonical_zip) as archive:
        archived_source = archive.read("executed_source/joined_phi_u1_stationary_canonical_photon.py")
    if hashlib.sha256(archived_source).hexdigest() != before[canonical.__file__]:
        raise ValueError("Canonical execution archive has a different producer")

    unitary = inputs["unitary"]
    delta = F(unitary["interval_energy"]["actual_saved_stationary_root"]["delta_energy"]["upper"])
    epsilon = F(unitary["conditional_FullPhi"]["epsilon"]["upper"])
    full_delta = delta + 2 * epsilon
    assert full_delta < 0
    q = inputs["quantum"]["exact_budgets"]
    q_upper = F(q["quantum_upper_divided_Lambda"])
    static_floor = F(q["central_static_lower"])
    old_t_floor = F(q["conditional_old_T_lower_divided_Lambda"])
    assert static_floor - q_upper > F(3, 10)
    assert old_t_floor - q_upper > F(3, 10)
    assert F(q["correction_divided_Lambda"]) < F(1, 2**400)
    assert q["old_T_radius_is_not_support"] is True
    rms = F(q["old_T_RMS_radius_upper"])
    assert q["old_T_true_support_radius"] == "2^-20"
    support = F(1, 2**20)
    assert 0 < rms < support

    jf = inputs["jordan_flat"]
    partitions = jf["exact_ALLangle_sector_lower_bounds"]["partitions"]
    central_upper = F(partitions[0]["certified_trial_upper"]["upper"])
    eligibility = []
    for item in partitions:
        lower = F(item["ALL_continuous_angles_lower"]["lower"])
        excluded = lower > central_upper
        assert excluded is item["certified_excluded_as_global_minimum"]
        eligibility.append({"partition": item["partition"], "lower": float(lower), "separation_lower": float(lower - central_upper), "excluded": excluded})
    assert [x["partition"] for x in eligibility if x["excluded"]] == [[1, 1, 1, 1]]
    old_control = jf["exact_OLDlaw_positive_countercontrol"]
    extra = F(old_control["new_quartic_exact"]) - F(old_control["old_quartic_exact"])
    assert extra == F(22000467848, 244140625)
    rescued = F(old_control["NEWlaw_same_witness_energy"]["lower"]) - central_upper
    assert rescued > 89

    with np.load(ROOT / ARRAY, allow_pickle=False) as arrays:
        d60 = arrays["Jordan_D60"]
        d45 = arrays["Jordan_D45"]
        frame = arrays["normal_Q"]
        projected_error = float(np.linalg.norm(frame.T @ d60 @ frame - d45))
        assert projected_error < 1e-11
        eigenvalues = np.linalg.eigvalsh(_hermitian(d45))
        assert int(np.count_nonzero(eigenvalues > 1e-8)) == 35
        assert float(eigenvalues[0]) > -1e-11
        old_principal = arrays["0_0_0_K_Lambda"][:45, :45]
        old_cov = _principal_covariance(old_principal)
        new_cov = _principal_covariance(old_principal + d45)
        covariance_delta = _hermitian(old_cov - new_cov)
        covariance_order_min = float(np.linalg.eigvalsh(covariance_delta)[0])
        assert covariance_order_min > -1e-10
        # Distinct consumer weights test PSD quadratic moments, not entrywise
        # covariance or signed mixing/derivative monotonicity.
        weights = [np.eye(45), np.diag(np.arange(1, 46)), np.outer(np.arange(45), np.arange(45))]
        gains = [float(np.trace(w @ covariance_delta).real) for w in weights]
        assert min(gains) >= -1e-9
        pc_norms: dict[str, float] = {}
        conjugacy_defects: dict[str, float] = {}
        for channel in ("K_Lambda5over4", "K_Lambda", "K_unit"):
            pc_max = conjugacy_max = 0.0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        name = f"{i}_{j}_{k}_{channel}"
                        opposite = f"{(-i)%4}_{(-j)%4}_{(-k)%4}_{channel}"
                        matrix = arrays[name]
                        pc_max = max(pc_max, float(np.linalg.norm(matrix[:45, 45:])))
                        conjugacy_max = max(conjugacy_max, float(np.linalg.norm(matrix - arrays[opposite].conj())))
            pc_norms[channel] = pc_max
            conjugacy_defects[channel] = conjugacy_max
        assert pc_norms["K_Lambda"] > 0 and pc_norms["K_unit"] > 0
        assert max(conjugacy_defects.values()) < 1e-9

    # Actual changed API on the valid shared-HS zero-det boundary. Its full-CAR
    # oracle is independent of the coefficient recurrence, not a forwarding echo.
    boundary = canonical.canonical_singular_controls()
    cut3 = inputs["cut3"]
    cut2, cut3_values = cut3["cut2_to_cut3_connected"]["reference"], cut3["cut2_to_cut3_connected"]["actual"]
    observed_changes = [abs(y - x) for x, y in zip(cut2, cut3_values)]
    assert max(observed_changes) < 2.5e-7
    assert min(cut3["adverse_control_step005"]["seagull_omission_error"]) > .007
    assert cut3["natural_returncode"] == 0
    after = {name: _sha(Path(name) if Path(name).is_absolute() else ROOT / name) for name in before}
    assert before == after and own_before == _sha(Path(__file__))
    result = {
        "governing_plan": PLAN, "captured_revision": REVISION,
        "stage": "SCOPED_OLDLAW_QUANTUM_EXCLUSION_AND_NEW_JORDAN_COMPARISON_NOT_GROUND_OR_PHOTON_ADOPTION",
        "unitary": {"body_delta_upper": float(delta), "conditional_T_delta_upper": float(full_delta), "conditional_particle_number": 8324},
        "oldlaw_quantum": {"quantum_upper_divided_Lambda": float(q_upper), "static_floor_separation_lower": float(static_floor - q_upper), "conditional_any_body_old_T_separation_lower": float(old_t_floor - q_upper), "RMS_is_not_support": True, "tent_domain": "Friedrichs form, not D(H^2)", "not_all_quantum_central_states": True},
        "NEW_Jordan": {"same_witness_extra_divided_Lambda": str(extra), "same_witness_above_central_trial_lower": float(rescued), "ALLangle_sector_eligibility": eligibility, "rank35_normal_addition": True, "projected_matrix_error": projected_error, "principal_covariance_old_trace": float(np.trace(old_cov).real), "principal_covariance_new_trace": float(np.trace(new_cov).real), "principal_covariance_order_min": covariance_order_min, "PSD_weighted_moment_gains": gains, "full_PH90_PC_norm_max_by_named_channel": pc_norms, "all64_conjugacy_defect_by_named_channel": conjugacy_defects, "principal_scope": "q0 Phi45 principal comparison only; not full90 Cov, signed M, derivative, action, adopted root or ground covariance"},
        "canonical_zero_det_shared_HS_CAR_boundary": boundary,
        "canonical_explicit_execution_archive": {"path": canonical_zip.name, "sha256": _sha(canonical_zip), "historical_null_capsule_untouched": True},
        "cut3_finite_comparison": {"absolute_connected_changes": observed_changes, "not_a_convergence_bound": True, "law": cut3["scope"]},
        "input_sha256": before, "source_sha256": own_before, "sources_inputs_unchanged": before == after,
        "elapsed_seconds": time.monotonic() - start,
        "OPEN": ["proper NEW Jordan sectors 3+1, 2+2, 2+1+1", "fullPhi stationary ground characterization", "same-ground stationary photon pole/dispersion", "new root/full joint Cov/action suppliers", "commoncone/originalSOURCE/adoption/campaign"],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": result["stage"], "elapsed_seconds": result["elapsed_seconds"], "oldlaw_separation": result["oldlaw_quantum"]["conditional_any_body_old_T_separation_lower"], "newlaw_open_sectors": 3}, indent=2))


if __name__ == "__main__":
    main()

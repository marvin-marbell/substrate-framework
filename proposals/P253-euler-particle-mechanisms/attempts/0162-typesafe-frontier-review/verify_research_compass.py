#!/usr/bin/env python3
"""Verify authority, calibration, retrieval, and credential boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULT = HERE / "research-compass.json"
LOG = HERE / "research-compass-responses.jsonl"
REPORT = HERE / "research-compass-report.md"
REGISTRY = ROOT / "governance/claims.yaml"
SECRET = Path.home() / ".config/typesafe/secrets.env"


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in LOG.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    check(
        "registry hash matches the indexed authority boundary",
        result["generated_from"]["registry_sha256"]
        == hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
    )
    check("all 271 pinned claims are indexed", result["counts"]["accepted_claims"] == 271)
    check(
        "all 47 current Euler modules are indexed as provisional sources",
        result["counts"]["code_modules"] == 47,
    )
    check("five frontier failure nodes are indexed", result["counts"]["frontier_nodes"] == 5)
    check("six obligation cards are present", result["counts"]["obligations"] == 6)
    check("twenty pair edges per obligation are present", result["counts"]["pair_edges"] == 120)
    check("every append-only response line parses", len(rows) >= 120)
    check("the executed TypeSafe model is jev-1.13.0", result["model"] == "jev-1.13.0")

    vectors = result["source_vectors"]
    check(
        "declared coherent-state measurement is recognized as quantum structure",
        vectors["C-VOP-001"]["quantization_statistics_measurement"] >= 0.80,
    )
    check(
        "classical whole-realization mixture is not mislabeled quantum",
        vectors["C-CST-014"]["quantization_statistics_measurement"] <= 0.15,
    )
    check(
        "relative-basis quotient is recognized as internal mixing structure",
        vectors["C-MIX-002"]["internal_basis_mixing_chirality"] >= 0.60,
    )
    check(
        "conditional U1 current is recognized as charge/current structure",
        vectors["C-U1-001"]["charge_gauge_current"] >= 0.70,
    )
    check(
        "chart Hessian is recognized as stability/coercivity structure",
        vectors["C-RMAP-003"]["stability_coercivity"] >= 0.55,
    )

    expected_profiles = {
        "p253_partition2_actual_carrier": ("nonlinear_pde_matching", 0.75),
        "p253_continuum_trace": ("spectral_resolvent_structure", 0.80),
        "p253_quantum_bridge": ("quantization_statistics_measurement", 0.90),
        "p253_charge_converter": ("variational_structure", 0.75),
        "p253_neutrino_mixing": ("internal_basis_mixing_chirality", 0.65),
    }
    for obligation_id, (feature, floor) in expected_profiles.items():
        probability = result["obligations"][obligation_id]["need_profile"][
            "capability_distribution"
        ][feature]
        check(f"{obligation_id} retains dominant {feature}", probability >= floor)

    expected_sources = {
        "p253_partition2_actual_carrier": {
            "failure_regular_trace_layer",
            "failure_fprime_isolated_saddle",
            "C-CST-011",
        },
        "p253_continuum_trace": {"C-CST-012", "C-KRN-002", "C-RES-001"},
        "p253_quantum_bridge": {"C-VOP-001", "C-OSC-001", "C-SG-020"},
        "p253_charge_converter": {"C-CST-009", "C-CST-017"},
        "p253_neutrino_mixing": {"C-MIX-002", "C-CST-014"},
    }
    for obligation_id, expected in expected_sources.items():
        actual = {
            edge["source_id"] for edge in result["obligations"][obligation_id]["edges"]
        }
        check(f"{obligation_id} retrieves its expected source controls", expected <= actual)

    expected_code_sources = {
        "p253_partition2_actual_carrier": {
            "code:euler_axisymmetric_action",
            "code:euler_cao_schur",
        },
        "p253_continuum_trace": {
            "code:euler_asymptotic_tails",
            "code:euler_retained_memory",
        },
        "p253_quantum_bridge": {
            "code:euler_quantum_bridge",
            "code:euler_gauge_action_selection",
        },
        "p253_charge_converter": {
            "code:euler_gauge_action_selection",
            "code:euler_ertel_current",
        },
        "p253_neutrino_mixing": {"code:euler_neutrino_suppliers"},
    }
    for obligation_id, expected in expected_code_sources.items():
        actual = {
            edge["source_id"] for edge in result["obligations"][obligation_id]["edges"]
        }
        check(
            f"{obligation_id} retrieves current provisional implementations",
            expected <= actual,
        )

    for obligation_id, obligation in result["obligations"].items():
        ids = [edge["source_id"] for edge in obligation["edges"]]
        check(f"{obligation_id} has twenty unique edges", len(ids) == len(set(ids)) == 20)
        for failure_id in (
            "failure_regular_trace_layer",
            "failure_fprime_isolated_saddle",
            "failure_false_casimir",
            "failure_continuous_action_scale",
            "failure_autonomous_reset",
        ):
            check(f"{obligation_id} retains failure reservoir {failure_id}", failure_id in ids)

    secret_value = ""
    if SECRET.exists():
        for line in SECRET.read_text(encoding="utf-8").splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                secret_value = line.split("=", 1)[1].strip().strip("'\"")
                break
    artifact_text = RESULT.read_text(encoding="utf-8") + LOG.read_text(encoding="utf-8") + REPORT.read_text(encoding="utf-8")
    check("API credential is absent from durable artifacts", bool(secret_value) and secret_value not in artifact_text)


if __name__ == "__main__":
    main()

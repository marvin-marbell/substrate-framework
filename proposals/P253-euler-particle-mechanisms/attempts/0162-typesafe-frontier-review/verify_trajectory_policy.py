#!/usr/bin/env python3
"""Verify chronology, policy controls, coverage, and credential boundaries."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULT = HERE / "trajectory-policy.json"
LOG = HERE / "trajectory-policy-responses.jsonl"
REPORT = HERE / "trajectory-policy-report.md"
PROPOSAL = ROOT / "proposals/P253-euler-particle-mechanisms/proposal.yaml"
SECRET = Path.home() / ".config/typesafe/secrets.env"


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in LOG.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    nodes = result["nodes"]
    by_id = {node["id"]: node for node in nodes}

    check("trajectory model is jev-1.13.0", result["model"] == "jev-1.13.0")
    check("all 168 committed historical episodes are present", len(nodes) == 168)
    check("episode identifiers are unique", len(by_id) == len(nodes))
    check(
        "the live uncommitted TypeSafe attempt is excluded from historical policy",
        "0162-typesafe-frontier-review" in result["excluded_attempts"],
    )
    check(
        "proposal authority hash matches",
        result["generated_from"]["proposal_sha256"]
        == hashlib.sha256(PROPOSAL.read_bytes()).hexdigest(),
    )
    check(
        "response log hash matches",
        result["generated_from"]["response_log_sha256"]
        == hashlib.sha256(LOG.read_bytes()).hexdigest(),
    )
    check("every append-only response parses", len(rows) >= 500)

    check(
        "every episode completes no earlier than its decision snapshot",
        all(
            parse_time(node["decision_at"]) <= parse_time(node["available_at"])
            for node in nodes
        ),
    )
    for slice_ in result["time_slices"]:
        target = by_id[slice_["decision_episode"]]
        eligible = sorted(
            node["id"]
            for node in nodes
            if parse_time(node["available_at"]) < parse_time(target["decision_at"])
        )
        check(
            f"{target['id']} time slice has the exact eligible count",
            slice_["eligible_completed_episodes"] == len(eligible),
        )
        digest = hashlib.sha256(json.dumps(eligible, sort_keys=True).encode()).hexdigest()
        check(
            f"{target['id']} time slice has the exact eligible-state hash",
            slice_["eligible_ids_sha256"] == digest,
        )

    audits = result["historical_decision_audits"]
    check("distributed historical decision audits exist", len(audits) >= 24)
    for audit in audits:
        for edge in audit["counterfactuals"]:
            check(
                f"{edge['source_id']} is time-eligible for {audit['episode_id']}",
                parse_time(edge["source_available_at"])
                < parse_time(edge["target_decision_at"]),
            )

    check(
        "every labeled missed opportunity has a cleaner time slice",
        all(
            item["historical_policy_confidence"] == "cleaner_time_slice"
            and by_id[item["target_episode"]]["hindsight_exposure"]
            == "cleaner_multi_commit"
            for item in result["missed_opportunities"]
        ),
    )
    check(
        "hindsight-sensitive leads remain separately labeled",
        bool(result["hindsight_sensitive_leads"])
        and all(
            item["historical_policy_confidence"] == "hindsight_sensitive"
            and by_id[item["target_episode"]]["hindsight_exposure"]
            != "cleaner_multi_commit"
            for item in result["hindsight_sensitive_leads"]
        ),
    )
    check(
        "no missed opportunity was already explicitly consumed",
        all(not item["explicitly_referenced"] for item in result["missed_opportunities"]),
    )

    controls = {
        "0005": ("P4", None),
        "0108-atlas-comms": ("process", "coordination_or_recording"),
        "0108-drift-critique": (None, "review_and_correction"),
        "0159-sage-hj2augmap": ("P2", None),
        "0161-sage-0062branch": ("P2", "direct_construction"),
    }
    for episode_id, (obligation, action) in controls.items():
        audit = by_id[episode_id]["audit"]
        if obligation:
            check(
                f"{episode_id} retains calibrated obligation {obligation}",
                audit["obligation"] == obligation,
            )
        if action:
            check(
                f"{episode_id} retains calibrated action {action}",
                audit["action"] == action,
            )
    check(
        "process-only communication does not earn scientific utility",
        by_id["0108-atlas-comms"]["audit"]["utility"] <= 0.20
        and by_id["0108-atlas-comms"]["audit"]["policy_verdict"] == "administrative",
    )
    check(
        "candidate foundation extensions retain their authorization boundary",
        by_id["0068"]["requires_foundation_authorization"],
    )
    check(
        "foundation extensions cannot be labeled missed historical actions",
        all(
            not by_id[item["source_id"]]["requires_foundation_authorization"]
            for item in result["missed_opportunities"]
        ),
    )

    priors = result["operation_priors"]
    check("all research operations receive learned priors", len(priors) == 12)
    check(
        "all learned prior means are probabilities",
        all(0.0 <= item["mean_useful_transition"] <= 1.0 for item in priors.values()),
    )
    check(
        "scientific construction outranks process recording",
        priors["direct_construction"]["mean_useful_transition"]
        > priors["coordination_or_recording"]["mean_useful_transition"],
    )
    check("repeated-failure families are mapped", len(result["repeated_failures"]) >= 8)

    expected_code = {
        "p253_partition2_actual_carrier": "code:euler_axisymmetric_action",
        "p253_continuum_trace": "code:euler_asymptotic_tails",
        "p253_quantum_bridge": "code:euler_quantum_bridge",
        "p253_charge_converter": "code:euler_gauge_action_selection",
        "p253_neutrino_mixing": "code:euler_neutrino_suppliers",
    }
    check("all five current obligations are ranked", set(result["current_action_rankings"]) == set(expected_code))
    for obligation_id, code_source in expected_code.items():
        candidates = result["current_action_rankings"][obligation_id]
        types = {candidate["candidate_type"] for candidate in candidates}
        check(f"{obligation_id} retains historical candidates", "historical_episode_transfer" in types)
        check(f"{obligation_id} retains accepted-claim candidates", "accepted_claim_transfer" in types)
        check(f"{obligation_id} retains provisional-code candidates", code_source in {candidate["source_id"] for candidate in candidates})
        check(f"{obligation_id} retains typed-failure candidates", "failure_transfer" in types)
    check(
        "the unlicensed U1 extension is not the default charge action",
        result["current_action_rankings"]["p253_charge_converter"][0]["source_id"]
        != "0068",
    )

    check(
        "all exact frontier failures receive stuck interventions",
        len(result["stuck_interventions"]) == 5,
    )
    for failure_id, intervention in result["stuck_interventions"].items():
        check(f"{failure_id} keeps its exact failure statement", bool(intervention["failure_statement"]))
        check(f"{failure_id} has several conditioned interventions", len(intervention["candidates"]) == 8)
        check(
            f"{failure_id} interventions name a research operation and bridge",
            all(candidate["research_move"] and candidate["missing_bridge"] for candidate in intervention["candidates"]),
        )

    secret_value = ""
    if SECRET.exists():
        for line in SECRET.read_text(encoding="utf-8").splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                secret_value = line.split("=", 1)[1].strip().strip("'\"")
                break
    artifact_text = (
        RESULT.read_text(encoding="utf-8")
        + LOG.read_text(encoding="utf-8")
        + REPORT.read_text(encoding="utf-8")
    )
    check(
        "API credential is absent from trajectory artifacts",
        bool(secret_value) and secret_value not in artifact_text,
    )


if __name__ == "__main__":
    main()

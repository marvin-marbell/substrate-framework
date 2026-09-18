"""Test the optimized authority-first, isolated-card TypeSafe architecture."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frontier_review = load_module("frontier_review", "run_typesafe_review.py")
experiment = load_module("method_experiment", "run_method_experiment.py")
ask = frontier_review.ask


def hard_authority(case: dict) -> dict[str, bool]:
    records = case["records"]
    issue = records["canonical_issue"]
    authorized = issue["state"] == "OPEN"
    direct = issue["number"] == records["current_parent"]
    status_conflict = records.get("proposal_status") == "active" and issue["state"] == "CLOSED"
    frontier_conflict = (
        "proposal_frontier_latest_attempt" in records
        and records["proposal_frontier_latest_attempt"]
        != records["current_synthesis_latest_attempt"]
    )
    return {
        "authorized_active": authorized,
        "direct_parent": direct,
        "record_conflict": status_conflict or frontier_conflict,
    }


def authority_metrics() -> dict:
    rows = []
    for case_id, case in experiment.AUTHORITY_CASES.items():
        computed = hard_authority(case)
        for field, expected in case["labels"].items():
            rows.append(
                {
                    "case": case_id,
                    "field": field,
                    "expected": expected,
                    "computed": computed[field],
                    "correct": computed[field] is expected,
                }
            )
    return {
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "type_safe_tokens": 0,
        "rows": rows,
    }


def canonicalize(records: dict) -> dict:
    """Retain scientific substance; remove known non-evidence presentation fields."""
    return {key: value for key, value in records.items() if key != "surface_signals"}


def digest(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def question(axis: str) -> dict:
    meanings = {
        "consumption": "ability of this result to be consumed by a current parent route",
        "executability": "readiness for one bounded construction with named current inputs",
        "natural_fit": "fit to the frozen Euler substrate without an unlicensed new law",
        "information": "amount by which a clean failure changes the campaign's next decision",
        "parent_value": "substantive value to the parent after presentation signals are removed",
    }
    return {
        "value": {
            "type": "score",
            "instructions": {
                "task": f"Score `candidate` for {meanings[axis]}.",
                "boundary": "Judge only fields present in `candidate`.",
                "unknown_rule": "Missing support lowers the score; do not invent a bridge.",
            },
            "criteria": experiment.SCORE_LEVELS,
        }
    }


def evaluate_once(payload: tuple[dict, str]) -> dict:
    records, axis = payload
    return ask(
        {
            "parent": "issue #203 electron-and-neutrino mechanisms from frozen Euler",
            "candidate": records,
        },
        question(axis),
    )


def semantic_metrics() -> dict:
    unique_payloads: dict[str, tuple[dict, str]] = {}
    case_to_key = {}
    for case_id, case in experiment.MUTATION_CASES.items():
        records = canonicalize(case["records"])
        key = digest({"records": records, "axis": case["axis"]})
        unique_payloads.setdefault(key, (records, case["axis"]))
        case_to_key[case_id] = key

    keys = list(unique_payloads)
    with ThreadPoolExecutor(max_workers=4) as pool:
        first_values = list(pool.map(evaluate_once, [unique_payloads[key] for key in keys]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        repeat_values = list(pool.map(evaluate_once, [unique_payloads[key] for key in keys]))

    first = dict(zip(keys, first_values, strict=True))
    repeat = dict(zip(keys, repeat_values, strict=True))
    scores = {
        case_id: float(first[key]["answers"]["value"]["score"])
        for case_id, key in case_to_key.items()
    }
    repeat_scores = {
        case_id: float(repeat[key]["answers"]["value"]["score"])
        for case_id, key in case_to_key.items()
    }
    directional = {
        "consumer": scores["consumer_present"] > scores["consumer_removed"],
        "inputs": scores["inputs_ready"] > scores["inputs_missing"],
        "license": scores["licensed_euler"] > scores["unlicensed_law"],
        "failure": scores["major_failure"] > scores["minor_failure"],
    }
    repeat_deltas = {
        case_id: repeat_scores[case_id] - scores[case_id] for case_id in scores
    }
    raw = {
        key: {"first": first[key], "repeat": repeat[key], "payload": unique_payloads[key]}
        for key in keys
    }
    (HERE / "method-v3-isolated-response.json").write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "unique_cards": len(unique_payloads),
        "deduplicated_cards": len(experiment.MUTATION_CASES) - len(unique_payloads),
        "scores": scores,
        "repeat_scores": repeat_scores,
        "directional_accuracy": sum(directional.values()) / len(directional),
        "directional_checks": directional,
        "false_green_delta": scores["false_green"] - scores["plain_evidence"],
        "repeat_max_abs_delta": max(abs(value) for value in repeat_deltas.values()),
        "repeat_mean_abs_delta": sum(abs(value) for value in repeat_deltas.values()) / len(repeat_deltas),
        "repeat_deltas": repeat_deltas,
        "usage": {
            "first_input_tokens": sum(value["usage"]["input_tokens"] for value in first.values()),
            "first_output_tokens": sum(value["usage"]["output_tokens"] for value in first.values()),
            "repeat_input_tokens": sum(value["usage"]["input_tokens"] for value in repeat.values()),
            "repeat_output_tokens": sum(value["usage"]["output_tokens"] for value in repeat.values()),
        },
    }


def main() -> None:
    result = {
        "architecture": {
            "hard_authority_first": True,
            "one_semantic_card_per_request": True,
            "presentation_fields_removed": ["surface_signals"],
            "content_hash_deduplication": True,
            "repeat_check": True,
        },
        "authority": authority_metrics(),
        "semantic": semantic_metrics(),
    }
    (HERE / "isolated-experiment.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

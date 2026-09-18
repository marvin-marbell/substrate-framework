"""Compare TypeSafe question designs on real authority drift and mutations."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "frontier_review", HERE / "run_typesafe_review.py"
)
assert SPEC and SPEC.loader
frontier_review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frontier_review)
ask = frontier_review.ask


AUTHORITY_CASES = {
    "p253_current": {
        "narrative": (
            "P253's proposal says active, objective_state active, execution_state active; "
            "canonical issue #203 is open; it is the frozen electron-and-neutrino parent."
        ),
        "records": {
            "proposal_status": "active",
            "objective_state": "active",
            "execution_state": "active",
            "canonical_issue": {"number": 203, "state": "OPEN"},
            "current_parent": 203,
        },
        "labels": {"authorized_active": True, "direct_parent": True, "record_conflict": False},
    },
    "p247_stale_active": {
        "narrative": (
            "P247's proposal still says active, but canonical issue #178 closed on "
            "2026-08-29; the current particle parent is issue #203."
        ),
        "records": {
            "proposal_status": "active",
            "canonical_issue": {"number": 178, "state": "CLOSED"},
            "current_parent": 203,
        },
        "labels": {"authorized_active": False, "direct_parent": False, "record_conflict": True},
    },
    "p240_stale_active": {
        "narrative": (
            "P240's proposal still says active, but canonical issue #146 closed on "
            "2026-08-29; the current particle parent is issue #203."
        ),
        "records": {
            "proposal_status": "active",
            "canonical_issue": {"number": 146, "state": "CLOSED"},
            "current_parent": 203,
        },
        "labels": {"authorized_active": False, "direct_parent": False, "record_conflict": True},
    },
    "p251_complete": {
        "narrative": (
            "P251 says accepted, objective complete, terminal success; canonical issue "
            "#200 is closed. Its artifacts may be reused but issue #203 is the current parent."
        ),
        "records": {
            "proposal_status": "accepted",
            "objective_state": "complete",
            "execution_state": "terminal_success",
            "canonical_issue": {"number": 200, "state": "CLOSED"},
            "current_parent": 203,
        },
        "labels": {"authorized_active": False, "direct_parent": False, "record_conflict": False},
    },
    "p252_extension": {
        "narrative": (
            "Issue #211 is open. Its original Newton sign-flip audit landed, while an "
            "owner-directed full-thread verification extension remains active. It does not "
            "discharge the issue #203 particle objective."
        ),
        "records": {
            "proposal_status": "landed_original_scope",
            "canonical_issue": {"number": 211, "state": "OPEN"},
            "remaining_scope": "full-thread verification extension",
            "current_parent": 203,
        },
        "labels": {"authorized_active": True, "direct_parent": False, "record_conflict": False},
    },
    "p253_frontier_drift": {
        "narrative": (
            "P253 proposal.route_frontier stops at attempt 0055, while the current herd "
            "synthesis records attempt 0161 complete at model scope with named continuum "
            "and actual-carrier continuations. Issue #203 remains open."
        ),
        "records": {
            "proposal_frontier_latest_attempt": "0055",
            "current_synthesis_latest_attempt": "0161",
            "canonical_issue": {"number": 203, "state": "OPEN"},
            "current_parent": 203,
        },
        "labels": {"authorized_active": True, "direct_parent": True, "record_conflict": True},
    },
}


MUTATION_CASES = {
    "consumer_present": {
        "narrative": (
            "An exact continuum trace would be consumed by the actual carrier branch and "
            "upgrade its model coefficient to physical scope."
        ),
        "records": {
            "pass_licenses": ["actual_carrier_branch_input"],
            "consumer": "partition-2 carrier construction",
            "scope": "continuum physical normalization",
        },
        "axis": "consumption",
        "pair": "consumer",
        "variant": "strong",
    },
    "consumer_removed": {
        "narrative": (
            "An exact continuum trace would be mathematically interesting, but no active "
            "obligation or route is allowed to consume it."
        ),
        "records": {"pass_licenses": [], "consumer": None, "scope": "continuum theorem"},
        "axis": "consumption",
        "pair": "consumer",
        "variant": "weak",
    },
    "inputs_ready": {
        "narrative": (
            "The next construction has a named operator, domain, source, normalization, "
            "and accepted dependencies; no prerequisite is missing."
        ),
        "records": {
            "next_construction": "weighted resolvent estimate",
            "required_now": [
                {"name": "operator", "status": "available"},
                {"name": "domain", "status": "available"},
                {"name": "source", "status": "available"},
            ],
            "unspecified_inputs": [],
        },
        "axis": "executability",
        "pair": "inputs",
        "variant": "strong",
    },
    "inputs_missing": {
        "narrative": (
            "The next construction is called a weighted resolvent estimate, but the "
            "operator domain and source normalization are still unspecified."
        ),
        "records": {
            "next_construction": "weighted resolvent estimate",
            "required_now": [
                {"name": "operator", "status": "available"},
                {"name": "domain", "status": "missing"},
                {"name": "source_normalization", "status": "missing"},
            ],
            "unspecified_inputs": ["domain", "source_normalization"],
        },
        "axis": "executability",
        "pair": "inputs",
        "variant": "weak",
    },
    "licensed_euler": {
        "narrative": (
            "The construction uses the frozen Euler action, accepted conservation laws, "
            "and a proved map on the current carrier."
        ),
        "records": {
            "substrate": "frozen Euler",
            "required_premises": [
                {"name": "Euler action", "authority": "accepted"},
                {"name": "carrier map", "authority": "proved_in_campaign"},
            ],
        },
        "axis": "natural_fit",
        "pair": "license",
        "variant": "strong",
    },
    "unlicensed_law": {
        "narrative": (
            "The construction keeps the Euler name but requires a new local conversion "
            "law and a free-energy functional that have only been proposed in exploratory notes."
        ),
        "records": {
            "substrate": "frozen Euler plus proposed extension",
            "required_premises": [
                {"name": "Euler action", "authority": "accepted"},
                {"name": "conversion law", "authority": "exploratory_unlicensed"},
                {"name": "free-energy functional", "authority": "missing"},
            ],
        },
        "axis": "natural_fit",
        "pair": "license",
        "variant": "weak",
    },
    "major_failure": {
        "narrative": (
            "Failure would show that the only licensed smooth branch cannot exist on the "
            "actual carrier and would force a different representation."
        ),
        "records": {
            "failure_scope": "only licensed smooth branch on selected carrier",
            "failure_changes": "representation and next candidate family",
        },
        "axis": "information",
        "pair": "failure",
        "variant": "strong",
    },
    "minor_failure": {
        "narrative": (
            "Failure would close one parameter value while nearby values and every other "
            "route remain unchanged."
        ),
        "records": {
            "failure_scope": "one parameter value",
            "failure_changes": "nothing outside this minor variant",
        },
        "axis": "information",
        "pair": "failure",
        "variant": "weak",
    },
    "plain_evidence": {
        "narrative": (
            "A route has a model identity, an explicit missing continuum theorem, and no "
            "current parent consumer until that theorem is proved."
        ),
        "records": {
            "substance": "model identity; missing continuum theorem; no current consumer",
            "surface_signals": {"check_count": 0, "attempt_count": 1, "polish": "plain"},
        },
        "axis": "parent_value",
        "pair": "false_green",
        "variant": "control",
    },
    "false_green": {
        "narrative": (
            "A celebrated route has 900/900 checks passing, 60 polished attempts, and a "
            "green dashboard. It still has only the same model identity, the same missing "
            "continuum theorem, and no current parent consumer until that theorem is proved."
        ),
        "records": {
            "substance": "model identity; missing continuum theorem; no current consumer",
            "surface_signals": {"check_count": 900, "attempt_count": 60, "polish": "celebrated green dashboard"},
        },
        "axis": "parent_value",
        "pair": "false_green",
        "variant": "no_op_mutation",
    },
}


SCORE_LEVELS = [
    {"value": 0, "meaning": "No useful amount of the named property."},
    {"value": 1, "meaning": "Weak or remote amount of the named property."},
    {"value": 2, "meaning": "Material but conditional or incomplete amount."},
    {"value": 3, "meaning": "Strong amount with a concrete current bridge."},
    {"value": 4, "meaning": "Decisive amount at the stated scope."},
]


def authority_questions(structured: bool) -> dict:
    questions = {}
    for case_id in AUTHORITY_CASES:
        ref = f"cases.{case_id}.records" if structured else f"cases.{case_id}.narrative"
        criteria = {
            "true": {
                "definition": "The current governing records authorize live work now.",
                "authority_rule": "An open canonical issue outranks a stale proposal status.",
            },
            "false": {
                "definition": "The canonical goal is closed or the record is historical only.",
                "authority_rule": "A proposal word such as active cannot reopen a closed issue.",
            },
        }
        questions[f"{case_id}__authorized_active"] = {
            "type": "noul",
            "instructions": {
                "task": f"Does `{ref}` authorize current work on its canonical objective?",
                "rule": "Use canonical issue state as governing authority.",
            } if structured else f"Is `{ref}` currently authorized active work?",
            **({"criteria": criteria} if structured else {}),
        }
        questions[f"{case_id}__direct_parent"] = {
            "type": "noul",
            "instructions": {
                "task": f"Is `{ref}` itself the current frozen parent objective?",
                "definition": "Direct parent means canonical issue number equals current_parent; reusable support is false.",
            } if structured else f"Is `{ref}` direct work on the current parent?",
        }
        questions[f"{case_id}__record_conflict"] = {
            "type": "noul",
            "instructions": {
                "task": f"Do the authoritative records inside `{ref}` conflict or lag materially?",
                "examples": [
                    "proposal says active while canonical issue is closed",
                    "proposal frontier ends far before the current synthesis frontier",
                ],
                "exclusions": ["accepted complete plus closed issue is agreement, not conflict"],
            } if structured else f"Do the records described in `{ref}` materially conflict or lag?",
        }
    return questions


def mutation_questions(structured: bool) -> dict:
    axis_meanings = {
        "consumption": "ability of the result to be consumed by a current parent route",
        "executability": "readiness for one bounded construction with named current inputs",
        "natural_fit": "fit to the frozen Euler substrate without an unlicensed new law",
        "information": "amount by which a clean failure changes the campaign's next decision",
        "parent_value": "substantive value to the parent after ignoring tallies, polish, and activity volume",
    }
    questions = {}
    for case_id, case in MUTATION_CASES.items():
        ref = f"cases.{case_id}.records" if structured else f"cases.{case_id}.narrative"
        axis = case["axis"]
        instructions = (
            {
                "task": f"Score `{ref}` for {axis_meanings[axis]}.",
                "use": ["substantive scope", "authority", "named dependencies", "downstream consumer"],
                "ignore": ["check count", "attempt count", "green labels", "polish", "celebratory wording"],
            }
            if structured
            else f"How strong is `{ref}` in {axis_meanings[axis]}?"
        )
        questions[case_id] = {
            "type": "score",
            "instructions": instructions,
            "criteria": SCORE_LEVELS if structured else [
                "none", "weak", "meaningful but incomplete", "strong", "decisive"
            ],
        }
    return questions


def state(structured: bool, reverse: bool = False) -> dict:
    authority_items = list(AUTHORITY_CASES.items())
    mutation_items = list(MUTATION_CASES.items())
    if reverse:
        authority_items.reverse()
        mutation_items.reverse()
    key = "records" if structured else "narrative"
    return {
        "current_parent": 203,
        "authority_rule": "canonical issue and accepted release outrank proposal prose and activity volume",
        "cases": {
            case_id: {key: case[key]}
            for case_id, case in authority_items + mutation_items
        },
    }


def classification_metrics(response: dict) -> dict:
    answers = response["answers"]
    rows = []
    for case_id, case in AUTHORITY_CASES.items():
        for field, expected in case["labels"].items():
            value = float(answers[f"{case_id}__{field}"]["noul"])
            predicted = value >= 0.5
            rows.append(
                {
                    "case": case_id,
                    "field": field,
                    "expected": expected,
                    "value": value,
                    "correct": predicted is expected,
                }
            )
    return {
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "rows": rows,
    }


def mutation_metrics(response: dict) -> dict:
    scores = {case_id: float(response["answers"][case_id]["score"]) for case_id in MUTATION_CASES}
    directional = {
        "consumer": scores["consumer_present"] > scores["consumer_removed"],
        "inputs": scores["inputs_ready"] > scores["inputs_missing"],
        "license": scores["licensed_euler"] > scores["unlicensed_law"],
        "failure": scores["major_failure"] > scores["minor_failure"],
    }
    false_green_delta = scores["false_green"] - scores["plain_evidence"]
    return {
        "scores": scores,
        "directional_accuracy": sum(directional.values()) / len(directional),
        "directional_checks": directional,
        "false_green_delta": false_green_delta,
        "false_green_invariant_within_0_15": abs(false_green_delta) <= 0.15,
    }


def run_design(name: str, structured: bool, reverse: bool = False) -> dict:
    questions = {}
    questions.update(authority_questions(structured))
    questions.update(mutation_questions(structured))
    response = ask(state(structured, reverse), questions)
    path = HERE / f"method-{name}-response.json"
    path.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return response


def main() -> None:
    implicit = run_design("v1-implicit", structured=False)
    structured = run_design("v2-structured", structured=True)
    reversed_response = run_design("v2-structured-reversed", structured=True, reverse=True)

    stability_rows = []
    for key, answer in structured["answers"].items():
        value_key = "noul" if answer["type"] == "noul" else "score"
        normal = float(answer[value_key])
        reverse = float(reversed_response["answers"][key][value_key])
        stability_rows.append({"question": key, "normal": normal, "reverse": reverse, "delta": reverse - normal})

    result = {
        "implicit": {
            "classification": classification_metrics(implicit),
            "mutations": mutation_metrics(implicit),
            "usage": implicit["usage"],
        },
        "structured": {
            "classification": classification_metrics(structured),
            "mutations": mutation_metrics(structured),
            "usage": structured["usage"],
        },
        "structured_order_stability": {
            "max_abs_delta": max(abs(row["delta"]) for row in stability_rows),
            "mean_abs_delta": sum(abs(row["delta"]) for row in stability_rows) / len(stability_rows),
            "rows": stability_rows,
            "usage": reversed_response["usage"],
        },
    }
    (HERE / "method-experiment.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

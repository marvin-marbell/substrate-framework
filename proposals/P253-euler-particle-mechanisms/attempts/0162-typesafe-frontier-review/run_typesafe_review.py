"""Typed, reproducible TypeSafe review of the active research frontier.

The credential is read from ~/.config/typesafe/secrets.env and is never emitted.
Responses contain judgments only and may be committed as exploratory evidence.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
SECRET_FILE = Path.home() / ".config" / "typesafe" / "secrets.env"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


GLOBAL_STATE = {
    "parent_objective": (
        "Derive both electron and neutrino mechanisms from actual 3D constant-density "
        "incompressible Euler, electron first. P0 and P1 are established; P2 and P4 "
        "are active; P3 and both particle identifications remain pending."
    ),
    "decision_rule": (
        "Prefer a construction whose pass or fail materially changes the parent route. "
        "Passing checks, more route records, or a polished conditional model have little "
        "value unless they discharge a dependency used by P2-P6."
    ),
    "scope_constraints": [
        "The Euler substrate and frozen issue #203 objective remain fixed.",
        "A model calculation is not a continuum theorem; a classical carrier is not a particle.",
        "New constitutive or quantum laws remain explicit hypotheses unless derived.",
        "Failure is valuable when it names a mechanism and redirects the campaign.",
    ],
}


CANDIDATES = {
    "carrier_partition2": {
        "title": "Execute the partition-2 branch on the actual Cao carrier",
        "scope": "P253 P2, direct",
        "evidence": (
            "Attempt 0161 completed the fixed-n two-mode seed, resonance, distorted "
            "adjoint, nonzero model V*, and topology partition. The licensed branch "
            "requires an actual carrier critical point with indefinite Hessian, smooth "
            "compatible-centralizer matching, and a separatrix solve."
        ),
        "pass_payoff": (
            "Constructs a genuine rotating Euler branch candidate and advances the "
            "persistent-carrier obligation beyond model scope."
        ),
        "failure_value": (
            "A missing indefinite critical point or failed invariant-compatible match "
            "would close the advertised smooth branch with a named mechanism."
        ),
        "risk": (
            "Still needs separate nonlinear stability and all quantum/electron work; "
            "the model-to-continuum bridge may fail."
        ),
    },
    "continuum_trace": {
        "title": "Prove the continuum limiting-absorption trace and physical residue",
        "scope": "P253 P2 dependency, direct",
        "evidence": (
            "Attempt 0161 has only a two-mode sandwiched trace and a dimensionless "
            "residue witness. The continuum Green/Leray block, analytic remainder, "
            "delta scaling, physical units, channel nonvanishing, and KKS normalization "
            "remain explicit gaps consumed by any actual branch."
        ),
        "pass_payoff": (
            "Upgrades the resonance coefficient from a model witness to a physically "
            "normalized carrier input shared by the branch construction."
        ),
        "failure_value": (
            "Can reveal that the resonance disappears, the trace is unbounded, or the "
            "coefficient is undefined at continuum scope, stopping the branch early."
        ),
        "risk": (
            "Technically demanding operator analysis; it remains a dependency rather "
            "than the persistent carrier or particle mechanism itself."
        ),
    },
    "director_fireability": {
        "title": "Make the FB-D director-wave prediction independently fireable",
        "scope": "P253 P2/P3 support",
        "evidence": (
            "Attempt 0160 derives a positive director stiffness, strain coupling, real "
            "wave dispersion, and PSD window in its model. Its independent-p wave "
            "falsifier is explicitly named but currently unfireable."
        ),
        "pass_payoff": (
            "Tests a live emergent-elastic prediction independently and could validate "
            "or kill the director mechanism used by the scoped carrier story."
        ),
        "failure_value": (
            "A failed dispersion or negative mode kills the director lane cleanly."
        ),
        "risk": (
            "Even success may remain a collective-mode result with no direct electron, "
            "charge, statistics, or scale-selection consequence."
        ),
    },
    "p4_prequantization": {
        "title": "Attack action quantization and exchange through the physical KKS orbit",
        "scope": "P253 P4, direct bottleneck",
        "evidence": (
            "The P4 audit finds quantum amplitudes, exchange statistics, Lorentz "
            "propagation, and action normalization missing. Existing exact ingredients "
            "include a compact-pair KKS oscillator and a conditional Schwinger-Hopf "
            "doublet; earlier classical normalization leaves the action scale continuous."
        ),
        "pass_payoff": (
            "A prequantization integrality/exchange-holonomy construction on one physical "
            "Euler orbit could supply a discrete action unit and a real statistics bridge."
        ),
        "failure_value": (
            "A proof that the relevant symplectic periods remain continuously rescalable "
            "under the allowed Euler similarities would establish a sharp bare-Euler P4 "
            "obstruction and prevent more classical P2 work from being mistaken for a particle."
        ),
        "risk": (
            "No current in-tree route supplies Born amplitudes or fermionic exchange; "
            "geometric quantization may require an explicit new quantum postulate and "
            "the existing orbit may not be the eventual carrier."
        ),
    },
    "allotrope_converter": {
        "title": "Test an Euler-native two-allotrope steady converter as charge carrier",
        "scope": "P253 P5 reframe from issue #214",
        "evidence": (
            "Issue #214 proposes charge as the integral of a steady conversion rate "
            "between two stable substrate allotropes, with helicity sign or director "
            "orientation as candidate labels. No explicit Euler free energy, conversion "
            "functional, or compact solution is presently derived."
        ),
        "pass_payoff": (
            "Would attack electric charge as a process on the same substrate rather than "
            "assigning a static label, and could link P251 structure to P253."
        ),
        "failure_value": (
            "An Euler transport or conservation obstruction to local conversion would "
            "decide whether the idea needs a changed substrate instead of more numerics."
        ),
        "risk": (
            "The source proposal lacks its load-bearing constitutive functional and may "
            "silently add non-Euler physics; spin/statistics and neutrinos remain untouched."
        ),
    },
    "goldstone_width": {
        "title": "Diagnose whether P247's soft width is an exact Goldstone direction",
        "scope": "Cross-campaign scale-selection diagnostic",
        "evidence": (
            "Issue #214 reframes P247's box-growing clock and approximately 1e-5 width "
            "stiffness: determine whether an exact symmetry makes the width flat or "
            "identify the term that makes it weakly massive."
        ),
        "pass_payoff": (
            "Could isolate the missing scale-selection principle and a legitimate "
            "long-range coordinate relevant to future carrier interactions."
        ),
        "failure_value": (
            "Separates a protected zero mode from a numerical near-zero and prevents a "
            "false Goldstone/gravity interpretation."
        ),
        "risk": (
            "The P247 model is not the active Euler particle carrier and the result may "
            "not license any P253 obligation."
        ),
    },
    "p252_oracle_completion": {
        "title": "Finish the P252 full-thread external oracle boundary",
        "scope": "Issue #211 verification",
        "evidence": (
            "The original sign-flip audit landed and refuted the Newton headline; the "
            "full-thread extension still needs two reply atoms and an owner-frozen "
            "validation surface for remaining checkable claims."
        ),
        "pass_payoff": (
            "Completes an honest external-claim boundary and may preserve reusable "
            "algebraic facts."
        ),
        "failure_value": "Can correct individual external verdicts.",
        "risk": (
            "No claim was promoted and the task does not directly discharge P253's "
            "carrier, quantum, charge, or neutrino obligations."
        ),
    },
    "chart_coercivity": {
        "title": "Repair the 0155 orbit chart's coercivity wall without a false Casimir",
        "scope": "P253 P2 persistence support",
        "evidence": (
            "Attempt 0155 constructs a comoving kinematic chart but its H^-1 coercivity "
            "step failed. Attempt 0160 then found no usable Casimir on the banked A3 "
            "space. A different constrained-Hessian or spectral-gap representation is needed."
        ),
        "pass_payoff": (
            "Could turn an existing orbit chart into a persistence estimate and support "
            "P2 without inventing a conserved functional."
        ),
        "failure_value": (
            "A representation-level no-go would close the revived chart route and shift "
            "effort to the Cao branch."
        ),
        "risk": (
            "The route already encountered two structural walls and may duplicate newer "
            "HJ2/Cao machinery."
        ),
    },
    "detector_capture": {
        "title": "Build the deterministic rare-event Euler analyzer and capture/reset law",
        "scope": "P253 P4/P5 measurement route",
        "evidence": (
            "Earlier attempts establish a first-event probability theorem only under "
            "explicit analyzer-clock-capture-reset hypotheses and exclude autonomous "
            "finite-volume Euler reset. A deterministic scattering route remains active."
        ),
        "pass_payoff": (
            "Could connect physical intensity to exclusive outcomes and attack the missing "
            "amplitude/measurement conjunct on an actual substrate process."
        ),
        "failure_value": (
            "A deterministic-volume or recurrence obstruction would sharply delimit what "
            "measurement structure bare Euler cannot supply."
        ),
        "risk": (
            "May require an environment, absorber, or stochastic postulate and still not "
            "derive fermionic exchange or relativistic propagation."
        ),
    },
}


LEVELS = [
    "0 — absent or contradicted by the supplied evidence",
    "1 — weak, remote, or mostly speculative",
    "2 — meaningful but conditional or indirect",
    "3 — strong and supported by a concrete current construction",
    "4 — decisive at the stated scope with a clear downstream consumer",
]


AXES = {
    "parent_leverage": "How directly would completing this route advance the frozen issue #203 parent objective?",
    "bottleneck_relevance": "How directly does this route attack a dependency that currently blocks P2-P6 rather than an adjacent result?",
    "decisive_information": "How strongly would either a pass or a fail change the next research decision?",
    "closure_readiness": "How close are the stated inputs to supporting a valid first decisive construction now?",
    "evidence_upgrade": "How much would success upgrade evidence from model/conditional scope toward the exact physical scope consumed downstream?",
    "unlock_breadth": "How many important downstream obligations or route decisions could the result unlock?",
    "natural_fit": "How well does the route preserve the fixed Euler substrate and avoid unlicensed constitutive or quantum assumptions?",
    "progress_integrity": "How resistant is the route to yielding polished work that no parent obligation can consume?",
    "near_term": "How likely is one bounded attempt to obtain a meaningful pass, refutation, or named obstruction?",
}


WEIGHT_PROFILES = {
    "balanced": {
        "parent_leverage": 0.22,
        "bottleneck_relevance": 0.17,
        "decisive_information": 0.15,
        "closure_readiness": 0.10,
        "evidence_upgrade": 0.13,
        "unlock_breadth": 0.08,
        "natural_fit": 0.07,
        "progress_integrity": 0.05,
        "near_term": 0.03,
    },
    "outcome_first": {
        "parent_leverage": 0.30,
        "bottleneck_relevance": 0.20,
        "decisive_information": 0.15,
        "closure_readiness": 0.04,
        "evidence_upgrade": 0.15,
        "unlock_breadth": 0.10,
        "natural_fit": 0.03,
        "progress_integrity": 0.03,
        "near_term": 0.00,
    },
    "feasibility_first": {
        "parent_leverage": 0.15,
        "bottleneck_relevance": 0.10,
        "decisive_information": 0.15,
        "closure_readiness": 0.22,
        "evidence_upgrade": 0.10,
        "unlock_breadth": 0.05,
        "natural_fit": 0.08,
        "progress_integrity": 0.05,
        "near_term": 0.20,
    },
    "integrity_first": {
        "parent_leverage": 0.18,
        "bottleneck_relevance": 0.14,
        "decisive_information": 0.20,
        "closure_readiness": 0.06,
        "evidence_upgrade": 0.12,
        "unlock_breadth": 0.05,
        "natural_fit": 0.10,
        "progress_integrity": 0.15,
        "near_term": 0.00,
    },
}


def api_key() -> str:
    for line in SECRET_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            value = line.split("=", 1)[1].strip().strip("'\"")
            if value:
                return value
    raise RuntimeError(f"TYPESAFE_API_KEY is missing from {SECRET_FILE}")


def ask(state: dict, questions: dict, *, max_attempts: int = 5) -> dict:
    payload = json.dumps({"state": state, "model": MODEL, "questions": questions}).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {429, 529} or attempt + 1 == max_attempts:
                raise RuntimeError(f"TypeSafe HTTP {exc.code}: {detail}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(0.5 * 2**attempt, 8.0)
            time.sleep(delay)
    raise AssertionError("unreachable")


def round1_questions() -> dict:
    questions = {}
    for candidate_id in CANDIDATES:
        for axis, instruction in AXES.items():
            questions[f"{candidate_id}__{axis}"] = {
                "type": "score",
                "instructions": (
                    f"Evaluate `{candidate_id}` using only `candidates.{candidate_id}` and "
                    f"the global objective and constraints. {instruction}"
                ),
                "criteria": LEVELS,
            }

    choice_criteria = {
        candidate_id: candidate["title"] for candidate_id, candidate in CANDIDATES.items()
    }
    questions.update(
        {
            "best_parent_move": {
                "type": "choice",
                "instructions": (
                    "Which candidate is the strongest next move for the complete frozen parent "
                    "objective, weighting actual downstream consumption over activity volume?"
                ),
                "criteria": choice_criteria,
            },
            "best_decisive_attempt": {
                "type": "choice",
                "instructions": (
                    "Which candidate offers the most decision-changing bounded attempt, where "
                    "either success or a named failure mechanism improves the campaign?"
                ),
                "criteria": choice_criteria,
            },
            "best_bottleneck_attack": {
                "type": "choice",
                "instructions": (
                    "Which candidate most directly attacks the current load-bearing bottleneck "
                    "rather than extending an already mature supporting lane?"
                ),
                "criteria": choice_criteria,
            },
            "highest_pseudo_progress_risk": {
                "type": "choice",
                "instructions": (
                    "Which candidate is most likely to produce polished evidence that cannot be "
                    "consumed by any frozen P253 parent obligation?"
                ),
                "criteria": choice_criteria,
            },
        }
    )
    return questions


def score_table(response: dict) -> dict[str, dict[str, float]]:
    table = {candidate_id: {} for candidate_id in CANDIDATES}
    answers = response["answers"]
    for candidate_id in CANDIDATES:
        for axis in AXES:
            table[candidate_id][axis] = float(answers[f"{candidate_id}__{axis}"]["score"])
    return table


def composite(table: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, float]:
    return {
        candidate_id: sum(row[axis] * weights[axis] for axis in AXES) / 4.0
        for candidate_id, row in table.items()
    }


def ordered(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def round2_questions(leader_ids: list[str]) -> dict:
    criteria = {candidate_id: CANDIDATES[candidate_id]["title"] for candidate_id in leader_ids}
    questions = {
        "leader_after_scope_audit": {
            "type": "choice",
            "instructions": (
                "After inspecting the first-round axis scores, which leader best changes what "
                "the exact parent objective can conclude while respecting every scope fence?"
            ),
            "criteria": criteria,
        },
        "leader_if_failure_is_expected": {
            "type": "choice",
            "instructions": (
                "If every leader is more likely to fail than pass, which failure would most "
                "usefully redirect the campaign by identifying a precise mechanism?"
            ),
            "criteria": criteria,
        },
        "leader_for_one_bounded_attempt": {
            "type": "choice",
            "instructions": (
                "Which leader has the strongest single bounded next construction with current "
                "repository inputs, rather than requiring an open-ended program first?"
            ),
            "criteria": criteria,
        },
    }
    for candidate_id in leader_ids:
        questions[f"{candidate_id}__current_inputs_suffice"] = {
            "type": "noul",
            "instructions": (
                f"Do the supplied facts for `leaders.{candidate_id}` identify enough current "
                "inputs for one bounded exposing construction without inventing a new law?"
            ),
            "criteria": {
                "true": "A bounded construction can start from named current inputs.",
                "false": "An unspecified theorem, law, source, or representation is needed first.",
            },
        }
        questions[f"{candidate_id}__failure_redirects"] = {
            "type": "noul",
            "instructions": (
                f"Would a clean failure of `leaders.{candidate_id}` materially redirect the "
                "parent campaign rather than merely close a minor variant?"
            ),
        }
        questions[f"{candidate_id}__unlicensed_premise"] = {
            "type": "noul",
            "instructions": (
                f"Does `leaders.{candidate_id}` currently depend on a load-bearing premise not "
                "derived or licensed by the frozen Euler campaign?"
            ),
        }
        questions[f"{candidate_id}__scope_gap_dominates"] = {
            "type": "noul",
            "instructions": (
                f"Is the main obstacle for `leaders.{candidate_id}` a model-to-continuum or "
                "conditional-to-physical scope gap that must be resolved before downstream use?"
            ),
        }
    return questions


def main() -> None:
    state1 = {**GLOBAL_STATE, "candidates": CANDIDATES}
    response1 = ask(state1, round1_questions())
    (HERE / "round1-response.json").write_text(
        json.dumps(response1, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    table = score_table(response1)
    profiles = {name: composite(table, weights) for name, weights in WEIGHT_PROFILES.items()}
    leader_union = []
    for scores in profiles.values():
        for candidate_id, _ in ordered(scores)[:4]:
            if candidate_id not in leader_union:
                leader_union.append(candidate_id)
    leaders = leader_union[:6]

    state2 = {
        **GLOBAL_STATE,
        "leaders": {
            candidate_id: {
                **CANDIDATES[candidate_id],
                "round1_axis_scores_0_to_4": table[candidate_id],
                "profile_scores_0_to_1": {
                    name: scores[candidate_id] for name, scores in profiles.items()
                },
            }
            for candidate_id in leaders
        },
    }
    response2 = ask(state2, round2_questions(leaders))
    (HERE / "round2-response.json").write_text(
        json.dumps(response2, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    leave_one_out = {}
    balanced = WEIGHT_PROFILES["balanced"]
    for omitted in AXES:
        retained_total = 1.0 - balanced[omitted]
        weights = {
            axis: (0.0 if axis == omitted else weight / retained_total)
            for axis, weight in balanced.items()
        }
        leave_one_out[omitted] = composite(table, weights)

    result = {
        "model": response1["model"],
        "usage": {
            "round1": response1["usage"],
            "round2": response2["usage"],
        },
        "axis_scores_0_to_4": table,
        "profile_rankings": {
            name: ordered(scores) for name, scores in profiles.items()
        },
        "leave_one_axis_out_rankings": {
            omitted: ordered(scores) for omitted, scores in leave_one_out.items()
        },
        "round1_choices": {
            key: response1["answers"][key]
            for key in (
                "best_parent_move",
                "best_decisive_attempt",
                "best_bottleneck_attack",
                "highest_pseudo_progress_risk",
            )
        },
        "round2_leaders": leaders,
        "round2_answers": response2["answers"],
    }
    (HERE / "ranking.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

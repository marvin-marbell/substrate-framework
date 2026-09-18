#!/usr/bin/env python3
"""Build a time-sliced research trajectory and provisional policy evaluation.

The graph treats completed campaign attempts as logged transitions.  A source
episode becomes eligible for a historical counterfactual only after its final
commit and only when that completion predates the target episode's first
commit.  TypeSafe supplies typed semantic judgments; deterministic code owns
chronology, authority, eligibility, aggregation, and leakage checks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import threading
from typing import Any

import run_research_compass as compass


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CAMPAIGN = ROOT / "proposals/P253-euler-particle-mechanisms"
ATTEMPTS = CAMPAIGN / "attempts"
PROPOSAL = CAMPAIGN / "proposal.yaml"
HERD_INDEX = CAMPAIGN / "herd/INDEX.md"
COMPASS_RESULT = HERE / "research-compass.json"
RESPONSE_LOG = HERE / "trajectory-policy-responses.jsonl"
RESULT_FILE = HERE / "trajectory-policy.json"
REPORT_FILE = HERE / "trajectory-policy-report.md"
WRITE_LOCK = threading.Lock()


ACTION_OPTIONS = {
    **{
        name: compass.FEATURES[name]["source"]
        for name in compass.MOVE_FEATURES
    },
    "direct_construction": "Construct or derive the load-bearing mathematical or physical object directly.",
    "source_or_authority_audit": "Extract, compare, or verify primary sources, authority, dependencies, or applicability.",
    "review_and_correction": "Independently review an artifact and repair a concrete flaw exposed by the review.",
    "coordination_or_recording": "Coordinate work, record status, or preserve provenance without itself changing the scientific route.",
}

OUTCOME_OPTIONS = {
    "useful_construction": "A reusable construction, proof step, implementation, or exact relation was established at an honest scope.",
    "exposing_failure": "A route failed through a named mechanism that usefully changes later candidate design.",
    "scope_or_method_repair": "The episode corrected applicability, authority, numerics, an oracle, or a hidden assumption.",
    "conditional_progress": "The episode advanced a dependency while leaving a named bridge or condition open.",
    "inconclusive_exploration": "The episode explored a route but did not yet supply a reusable construction or decisive failure.",
    "duplicated_or_low_information": "The episode substantially repeated available work or used an uninformative local variation.",
    "administrative_transition": "The episode primarily coordinated, recorded, or delivered work rather than changing scientific knowledge.",
}

FAILURE_OPTIONS = {
    "applicability_or_scope": "A source, theorem, model, ansatz, or result did not apply at the physical or mathematical scope consumed downstream.",
    "representation_exclusion": "The chosen representation or admissible class excluded the required mechanism or object.",
    "missing_dynamics_or_action": "Formal structure lacked the physical action, evolution, generator, backreaction, or autonomous mechanism.",
    "missing_global_topology": "Local algebra failed to provide global compatibility, topology, integrality, holonomy, or boundary closure.",
    "missing_stability_or_persistence": "The route lacked coercivity, spectral control, nonlinear stability, or persistence in the required neighborhood.",
    "missing_scale_or_normalization": "The route left a coefficient, residue, dimension, physical scale, or normalization underdetermined.",
    "missing_observable_or_coupling": "The route lacked a physical observable, interaction, detector, current, or coupling to the intended consumer.",
    "numerical_or_conditioning": "Resolution, conditioning, discretization, convergence, precision, or solver behavior prevented the claimed inference.",
    "false_or_vacuous_oracle": "The purported test was circular, vacuous, insensitive, or measured a proxy that did not expose the actual claim.",
    "premise_or_invariant_conflict": "A required premise conflicted with the frozen substrate, invariant, conservation law, or candidate definition.",
    "none_named": "No load-bearing failure mechanism is established in this episode.",
}

POLICY_VERDICTS = {
    "advance": "The chosen action produced a useful construction or directly unlocked a downstream obligation.",
    "informative_failure": "The chosen action earned a named failure that materially improved later policy.",
    "repair": "The chosen action corrected a real flaw and restored trustworthy downstream use.",
    "conditional": "The action made scoped progress but left the consumer blocked on a named bridge.",
    "weak_or_redundant": "A time-eligible alternative appears to have offered more information or closure than this local choice.",
    "administrative": "This was delivery or coordination and should not be evaluated as a scientific route choice.",
}

EVIDENCE_OPTIONS = {
    "analytic_or_formal": "Exact derivation, theorem, proof, symbolic identity, or rigorous mathematical implication.",
    "tested_computation": "Replayable numerical or computational evidence with an exposing assertion or sensitivity check.",
    "primary_source_audit": "Exact extraction and applicability audit of an external primary source.",
    "independent_review": "Independent review and correction evidence.",
    "exploratory": "Exploratory model, sketch, analogy, sample, or provisional calculation.",
    "process_only": "Coordination or provenance with no independent scientific evidence.",
}

PAIR_MOVE_TO_PRIOR = {
    "change_representation": "representation_change",
    "construct_inverse": "inverse_construction",
    "take_controlled_limit": "controlled_limit_asymptotic",
    "decompose_or_factor": "decomposition_factorization",
    "compose_with_glue": "composition_bridge",
    "repair_scope": "assumption_scope_repair",
    "design_oracle": "oracle_sensitivity_design",
    "invert_failure": "failure_inversion",
    "none": "coordination_or_recording",
}


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_log() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not RESPONSE_LOG.exists():
        return records
    for line in RESPONSE_LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["key"]] = record
    return records


def append_record(record: dict[str, Any]) -> None:
    with WRITE_LOCK:
        with RESPONSE_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()


def inference_fingerprint(state: dict[str, Any], questions: dict[str, Any]) -> str:
    return fingerprint({"state": state, "questions": questions, "model": compass.MODEL})


def cache_matches(
    record: dict[str, Any] | None,
    state: dict[str, Any],
    questions: dict[str, Any],
) -> bool:
    return bool(
        record
        and record.get("inference_sha256")
        == inference_fingerprint(state, questions)
    )


def infer(
    kind: str,
    item_id: str,
    state: dict[str, Any],
    questions: dict[str, Any],
) -> dict[str, Any]:
    response = compass.ask(state, questions)
    record = {
        "key": f"{kind}:{item_id}",
        "kind": kind,
        "id": item_id,
        "inference_sha256": inference_fingerprint(state, questions),
        "response": response,
    }
    append_record(record)
    return record


def run_jobs(
    jobs: list[tuple[str, str, dict[str, Any], dict[str, Any]]],
    log: dict[str, dict[str, Any]],
    workers: int,
    label: str,
) -> dict[str, dict[str, Any]]:
    pending = [
        job
        for job in jobs
        if not cache_matches(log.get(f"{job[0]}:{job[1]}"), job[2], job[3])
    ]
    print(f"{label} jobs pending: {len(pending)}", flush=True)
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(infer, kind, item_id, state, questions): (kind, item_id)
                for kind, item_id, state, questions in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                log[record["key"]] = record
                if index % 20 == 0 or index == len(futures):
                    print(f"{label}: {index}/{len(futures)}", flush=True)
    return log


def campaign_commits() -> list[dict[str, Any]]:
    relative = str(CAMPAIGN.relative_to(ROOT))
    raw = git(
        "log",
        "--reverse",
        "--date=iso-strict",
        "--format=%x1e%H%x09%aI%x09%s",
        "--name-only",
        "--",
        relative,
    )
    commits = []
    for block in raw.split("\x1e"):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        commit, timestamp, subject = lines[0].split("\t", 2)
        commits.append(
            {
                "commit": commit,
                "timestamp": timestamp,
                "subject": subject,
                "paths": lines[1:],
            }
        )
    return commits


def attempt_history(commits: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    histories: dict[str, list[dict[str, Any]]] = {}
    prefix = f"{CAMPAIGN.relative_to(ROOT)}/attempts/"
    for commit in commits:
        names = {
            path[len(prefix):].split("/", 1)[0]
            for path in commit["paths"]
            if path.startswith(prefix) and "/" in path[len(prefix):]
        }
        for name in names:
            histories.setdefault(name, []).append(commit)
    return histories


def read_herd_events() -> dict[str, list[dict[str, str]]]:
    events: dict[str, list[dict[str, str]]] = {}
    if not HERD_INDEX.exists():
        return events
    for line in HERD_INDEX.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---") or "| UTC |" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 7:
            continue
        timestamp, agent, signal, obligation, attempt, artifact, verdict = cells
        match = re.search(r"(?:attempts/)?([0-9]{4}(?:-[a-z0-9-]+)?)", attempt)
        if not match:
            continue
        events.setdefault(match.group(1), []).append(
            {
                "timestamp": timestamp,
                "agent": agent,
                "signal": signal,
                "obligation": obligation,
                "artifact": artifact,
                "verdict": verdict,
            }
        )
    return events


def file_score(path: str, purpose: str) -> tuple[int, str]:
    name = Path(path).name.lower()
    score = 0
    if name in {"readme.md", "proposal.yaml", "result.md", "verdict.md"}:
        score += 100
    terms = (
        ("design", "scope", "charter", "freeze", "brief", "plan")
        if purpose == "decision"
        else ("result", "verdict", "review", "receipt", "status", "synthesis", "readme")
    )
    for index, term in enumerate(terms):
        if term in name:
            score += 60 - index
    if name.endswith((".md", ".yaml", ".yml", ".txt")):
        score += 20
    if any(term in name for term in ("log", "stdout", "stderr", "response")):
        score -= 80
    return (-score, path)


def text_at_commit(commit: str, directory: str, purpose: str, budget: int) -> str:
    try:
        paths = git("ls-tree", "-r", "--name-only", commit, "--", directory).splitlines()
    except subprocess.CalledProcessError:
        return ""
    selected = sorted(paths, key=lambda path: file_score(path, purpose))
    parts: list[str] = []
    used = 0
    for path in selected:
        if not path.lower().endswith((".md", ".yaml", ".yml", ".txt")):
            continue
        try:
            size = int(git("cat-file", "-s", f"{commit}:{path}").strip())
        except (subprocess.CalledProcessError, ValueError):
            continue
        if size > 120_000:
            continue
        try:
            text = git("show", f"{commit}:{path}")
        except subprocess.CalledProcessError:
            continue
        remaining = budget - used
        if remaining <= 0:
            break
        excerpt = text[:remaining]
        parts.append(f"\n## {path}\n{excerpt}")
        used += len(excerpt)
    return "".join(parts).strip()


def current_text(directory: Path, purpose: str, budget: int) -> str:
    paths = [path for path in directory.rglob("*") if path.is_file()]
    selected = sorted(paths, key=lambda path: file_score(str(path), purpose))
    parts: list[str] = []
    used = 0
    for path in selected:
        if path.suffix.lower() not in {".md", ".yaml", ".yml", ".txt"}:
            continue
        if path.stat().st_size > 120_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        remaining = budget - used
        if remaining <= 0:
            break
        excerpt = text[:remaining]
        parts.append(f"\n## {path.relative_to(ROOT)}\n{excerpt}")
        used += len(excerpt)
    return "".join(parts).strip()


def build_episode_cards() -> tuple[list[dict[str, Any]], list[str]]:
    commits = campaign_commits()
    histories = attempt_history(commits)
    herd_events = read_herd_events()
    episode_names = sorted(
        directory.name
        for directory in ATTEMPTS.iterdir()
        if directory.is_dir() and directory.name != HERE.name
    )
    committed = []
    excluded = [HERE.name]
    for name in episode_names:
        history = histories.get(name, [])
        if not history:
            excluded.append(name)
            continue
        directory_relative = str((ATTEMPTS / name).relative_to(ROOT))
        first = history[0]
        last = history[-1]
        decision = text_at_commit(first["commit"], directory_relative, "decision", 11_000)
        outcome = current_text(ATTEMPTS / name, "outcome", 14_000)
        commit_subjects = [item["subject"] for item in history[-12:]]
        combined = "\n".join([decision, outcome, *commit_subjects])
        requires_foundation_authorization = bool(
            re.search(
                r"candidate foundation extension|use the enlarged state|foundation comparator",
                combined,
                re.I,
            )
        )
        fidelity = "cleaner_multi_commit"
        if len(history) == 1:
            fidelity = "single_commit_result_exposure"
        elif re.search(r"\b(PASS|FAIL|KILL|REFUT|BANKED|CLOSED|DONE)\b", decision, re.I):
            fidelity = "first_commit_contains_outcome_language"
        committed.append(
            {
                "id": name,
                "node_type": "campaign_attempt_episode",
                "path": directory_relative,
                "decision_at": first["timestamp"],
                "available_at": last["timestamp"],
                "first_commit": first["commit"],
                "last_commit": last["commit"],
                "commit_count": len(history),
                "commit_subjects": commit_subjects,
                "decision_snapshot": decision,
                "outcome_evidence": outcome,
                "herd_events": herd_events.get(name, [])[-12:],
                "hindsight_exposure": fidelity,
                "requires_foundation_authorization": requires_foundation_authorization,
                "content_sha256": hashlib.sha256(combined.encode()).hexdigest(),
            }
        )
    committed.sort(key=lambda item: (parse_time(item["decision_at"]), item["id"]))
    return committed, excluded


def episode_questions() -> dict[str, Any]:
    return {
        "obligation": {
            "type": "choice",
            "instructions": "Which frozen P253 obligation is most directly acted on by this episode? Choose process only when no scientific obligation is materially advanced.",
            "criteria": {
                "P0": "Source/foundation maps, imports, and competing mechanism registration.",
                "P1": "Exact material observables, balances, closure, or retained memory.",
                "P2": "Localized persistent Euler carrier and restoring mechanism.",
                "P3": "Interactions, tails, deformation, scale, and conserved-label selection.",
                "P4": "Physical quantum/statistical and relativistic bridge with action normalization.",
                "P5": "Electron carrier, spin/statistics, charge/current, magnetic coupling, and prediction.",
                "P6": "Neutrino carrier, mixing, propagation, chirality, and observation.",
                "P7": "End-to-end integration and discriminating predictions.",
                "process": "Coordination, governance, delivery, or recording only.",
            },
        },
        "dominant_action": {
            "type": "choice",
            "instructions": "Which single research operation best describes the action actually taken in this episode? Use the artifact rather than its aspirations.",
            "criteria": ACTION_OPTIONS,
        },
        "outcome": {
            "type": "choice",
            "instructions": "Classify the strongest honest outcome established by this completed episode at its stated scope.",
            "criteria": OUTCOME_OPTIONS,
        },
        "failure_family": {
            "type": "choice",
            "instructions": "Classify the principal load-bearing failure mechanism actually exposed by this episode. Choose none_named when a failure is merely anticipated or absent.",
            "criteria": FAILURE_OPTIONS,
        },
        "evidence_kind": {
            "type": "choice",
            "instructions": "What is the strongest evidence kind actually supplied by this episode?",
            "criteria": EVIDENCE_OPTIONS,
        },
        "policy_verdict": {
            "type": "choice",
            "instructions": "Classify this completed episode as a research-policy transition. Reward useful constructions, exact failures, and real repairs; do not reward activity or green tallies alone.",
            "criteria": POLICY_VERDICTS,
        },
        "decision_likelihood": {
            "type": "noul",
            "instructions": "Does this episode represent a substantive choice of research approach that can meaningfully be compared with time-eligible alternatives?",
        },
        "useful_transition": {
            "type": "noul",
            "instructions": "Did the episode leave a reusable construction, named failure, or correction that should change a later campaign decision?",
        },
        "exact_failure_named": {
            "type": "noul",
            "instructions": "Does the episode establish the exact mechanism by which an attempted route fails, rather than merely report non-success?",
        },
        "repeated_local_variant": {
            "type": "noul",
            "instructions": "Given the supplied commit and artifact record, is this episode mainly a local variation of an already exhausted representation without a new discriminating calculation?",
        },
        "scope_conflict": {
            "type": "noul",
            "instructions": "Does the episode's strongest rhetoric exceed what its actual evidence and authority establish?",
        },
        "information_gain": {
            "type": "score",
            "instructions": "How much decision-relevant information did this episode add for choosing or constructing later routes?",
            "criteria": [
                "0 — no scientific information beyond status",
                "1 — weak or generic observation",
                "2 — useful scoped evidence with a substantial unresolved bridge",
                "3 — concrete construction, discriminating failure, or consequential repair",
                "4 — decisive evidence that materially reorganizes the campaign",
            ],
        },
        "closure_progress": {
            "type": "score",
            "instructions": "How much did the episode make a downstream obligation actually reachable at its honest scope?",
            "criteria": [
                "0 — no closure progress",
                "1 — orientation only",
                "2 — a usable dependency with important conditions",
                "3 — a major dependency is constructed or a branch is decisively redirected",
                "4 — the intended obligation is established at consumed scope",
            ],
        },
        "future_reuse": {
            "type": "noul",
            "instructions": "Is the episode's construction, failure mechanism, or oracle reusable outside the immediate local attempt?",
        },
    }


def choice(answer: dict[str, Any]) -> tuple[str, float]:
    probabilities = answer["probabilities"]
    return max(probabilities.items(), key=lambda item: item[1])


def score(answer: dict[str, Any]) -> float:
    return float(answer["score"])


def episode_audit(response: dict[str, Any]) -> dict[str, Any]:
    answers = response["answers"]
    obligation, obligation_probability = choice(answers["obligation"])
    raw_action_distribution = {
        name: float(value)
        for name, value in answers["dominant_action"]["probabilities"].items()
    }
    if obligation == "process":
        action_distribution = {
            name: (1.0 if name == "coordination_or_recording" else 0.0)
            for name in raw_action_distribution
        }
    else:
        denominator = 1.0 - raw_action_distribution["coordination_or_recording"]
        action_distribution = {
            name: (
                0.0
                if name == "coordination_or_recording"
                else value / denominator
                if denominator > 0
                else 0.0
            )
            for name, value in raw_action_distribution.items()
        }
    action, action_probability = max(
        action_distribution.items(), key=lambda item: item[1]
    )
    outcome, outcome_probability = choice(answers["outcome"])
    failure, failure_probability = choice(answers["failure_family"])
    evidence, evidence_probability = choice(answers["evidence_kind"])
    verdict, verdict_probability = choice(answers["policy_verdict"])
    transition_utility = max(
        0.0,
        min(
            1.0,
            0.32 * score(answers["information_gain"]) / 4.0
            + 0.32 * score(answers["closure_progress"]) / 4.0
            + 0.18 * float(answers["future_reuse"]["noul"])
            + 0.18 * float(answers["useful_transition"]["noul"])
            - 0.18 * float(answers["scope_conflict"]["noul"])
            - 0.12 * float(answers["repeated_local_variant"]["noul"]),
        ),
    )
    utility = (
        min(0.20, 0.20 * transition_utility)
        if obligation == "process"
        else transition_utility
    )
    if obligation == "process":
        outcome, outcome_probability = "administrative_transition", 1.0
        verdict, verdict_probability = "administrative", 1.0
    return {
        "obligation": obligation,
        "obligation_probability": float(obligation_probability),
        "action": action,
        "action_probability": float(action_probability),
        "action_distribution": action_distribution,
        "raw_action_distribution": raw_action_distribution,
        "outcome": outcome,
        "outcome_probability": float(outcome_probability),
        "failure_family": failure,
        "failure_probability": float(failure_probability),
        "evidence_kind": evidence,
        "evidence_probability": float(evidence_probability),
        "policy_verdict": verdict,
        "policy_verdict_probability": float(verdict_probability),
        "decision_likelihood": float(answers["decision_likelihood"]["noul"]),
        "useful_transition": float(answers["useful_transition"]["noul"]),
        "exact_failure_named": float(answers["exact_failure_named"]["noul"]),
        "repeated_local_variant": float(answers["repeated_local_variant"]["noul"]),
        "scope_conflict": float(answers["scope_conflict"]["noul"]),
        "information_gain": score(answers["information_gain"]),
        "closure_progress": score(answers["closure_progress"]),
        "future_reuse": float(answers["future_reuse"]["noul"]),
        "transition_utility": transition_utility,
        "utility": utility,
    }


def source_state(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_type": "historically_available_campaign_episode",
        "id": episode["id"],
        "authority": "provisional campaign evidence; accepted only where separately promoted",
        "available_at": episode["available_at"],
        "commit_subjects": episode["commit_subjects"],
        "outcome_evidence": episode["outcome_evidence"],
        "herd_events": episode["herd_events"],
        "hindsight_exposure": episode["hindsight_exposure"],
    }


def predecision_state(
    episode: dict[str, Any],
    eligible: list[dict[str, Any]],
) -> dict[str, Any]:
    recent = [
        {
            "id": item["id"],
            "available_at": item["available_at"],
            "commit_subjects": item["commit_subjects"][-2:],
        }
        for item in eligible[-6:]
    ]
    return {
        "title": f"Historical decision at {episode['id']}",
        "parent": "P253 full electron-and-neutrino objective",
        "positive_intent": "Choose a research operation that produces a useful construction, exposing failure, or consequential repair for the frozen P253 obligations.",
        "current_inputs": recent,
        "obstruction": episode["decision_snapshot"],
        "exclusions": (
            "Preserve constant-density incompressible Euler, natural initial-data neighborhoods, same-candidate implications, and derived or explicitly licensed quantum/relativistic structure. "
            "Later campaign outcomes are unavailable at this cutoff."
        ),
        "decision_at": episode["decision_at"],
        "hindsight_exposure": episode["hindsight_exposure"],
    }


def eligible_before(episodes: list[dict[str, Any]], target: dict[str, Any]) -> list[dict[str, Any]]:
    cutoff = parse_time(target["decision_at"])
    return [item for item in episodes if parse_time(item["available_at"]) < cutoff]


def select_decision_audits(episodes: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not episodes:
        return []
    chunk_size = max(1, math.ceil(len(episodes) / limit))
    selected = []
    for start in range(0, len(episodes), chunk_size):
        chunk = episodes[start:start + chunk_size]
        selected.append(
            max(
                chunk,
                key=lambda item: (
                    item["audit"]["decision_likelihood"]
                    * (0.4 + 0.6 * item["audit"]["utility"]),
                    item["audit"]["information_gain"],
                ),
            )
        )
    return selected[:limit]


TOKEN = re.compile(r"[a-z][a-z0-9_'-]{2,}")


def lexical_similarity(source: dict[str, Any], obligation: dict[str, Any]) -> float:
    left = set(TOKEN.findall(json.dumps(source, sort_keys=True).lower()))
    right = set(TOKEN.findall(json.dumps(obligation, sort_keys=True).lower()))
    return len(left & right) / max(1, len(right))


def retrieve_prior_sources(
    target: dict[str, Any],
    eligible: list[dict[str, Any]],
    profile: dict[str, Any],
    rarity: dict[str, float],
    limit: int,
) -> list[dict[str, Any]]:
    obligation = target["predecision_state"]
    ranked = []
    for source in eligible:
        semantic = compass.distribution_match(source["source_vector"], profile, rarity)
        lexical = lexical_similarity(source_state(source), obligation)
        failure_bonus = (
            source["source_vector"]["failure_inversion"]
            * target["audit"]["exact_failure_named"]
        )
        score_ = 0.62 * semantic + 0.23 * lexical + 0.15 * failure_bonus
        ranked.append((score_, source))
    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
    selected = []
    action_counts: dict[str, int] = {}
    for _, source in ranked:
        action = source["audit"]["action"]
        if action_counts.get(action, 0) >= 2:
            continue
        action_counts[action] = action_counts.get(action, 0) + 1
        selected.append(source)
        if len(selected) >= limit:
            break
    return selected


def pair_edge(
    response: dict[str, Any],
    source_vector: dict[str, float],
    profile: dict[str, Any],
) -> dict[str, Any]:
    answers = response["answers"]
    relation, relation_probability = choice(answers["relation"])
    research_move, research_move_probability = choice(answers["research_move"])
    bridge, bridge_probability = choice(answers["missing_bridge"])
    relation_weight = {
        "direct_ingredient": 1.0,
        "transferable_construction": 0.95,
        "structural_analogy": 0.78,
        "failure_to_invert": 0.93,
        "falsifier_or_oracle": 0.86,
        "conflicting_premise": 0.45,
        "irrelevant": 0.08,
    }
    capability_support = sum(
        profile["capability_distribution"][name] * source_vector[name]
        for name in compass.CONTENT_FEATURES
    )
    move_support = sum(
        profile["move_distribution"][name] * source_vector[name]
        for name in compass.MOVE_FEATURES
    )
    value = (
        float(answers["approach_information"]["score"]) / 4.0
        * (0.55 + 0.45 * float(answers["novel_direction"]["noul"]))
        * (1.0 - 0.45 * float(answers["scope_conflict"]["noul"]))
        * (0.25 + 0.75 * float(answers["structural_correspondence"]["noul"]))
        * (0.40 + 0.60 * float(answers["source_specific_instruction"]["noul"]))
        * (0.35 + 0.65 * capability_support)
        * (0.55 + 0.45 * move_support)
        * relation_weight[relation]
    )
    return {
        "relation": relation,
        "relation_probability": float(relation_probability),
        "research_move": research_move,
        "research_move_probability": float(research_move_probability),
        "missing_bridge": bridge,
        "missing_bridge_probability": float(bridge_probability),
        "approach_information": float(answers["approach_information"]["score"]),
        "novel_direction": float(answers["novel_direction"]["noul"]),
        "scope_conflict": float(answers["scope_conflict"]["noul"]),
        "structural_correspondence": float(answers["structural_correspondence"]["noul"]),
        "source_specific_instruction": float(answers["source_specific_instruction"]["noul"]),
        "capability_support": capability_support,
        "move_support": move_support,
        "research_value": value,
    }


def learned_operation_priors(episodes: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    priors = {}
    for action in ACTION_OPTIONS:
        useful = 0.0
        nonuseful = 0.0
        for episode in episodes:
            weight = episode["audit"]["action_distribution"].get(action, 0.0)
            useful += weight * episode["audit"]["utility"]
            nonuseful += weight * (1.0 - episode["audit"]["utility"])
        alpha = 1.0 + useful
        beta = 1.0 + nonuseful
        priors[action] = {
            "alpha": alpha,
            "beta": beta,
            "mean_useful_transition": alpha / (alpha + beta),
            "effective_episode_mass": useful + nonuseful,
        }
    return priors


def repeat_failure_map(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for family in FAILURE_OPTIONS:
        if family == "none_named":
            continue
        members = [
            episode
            for episode in episodes
            if episode["audit"]["failure_family"] == family
            and episode["audit"]["failure_probability"] >= 0.35
            and episode["audit"]["exact_failure_named"] >= 0.35
        ]
        if not members:
            continue
        result.append(
            {
                "failure_family": family,
                "count": len(members),
                "first_episode": members[0]["id"],
                "last_episode": members[-1]["id"],
                "episodes": [member["id"] for member in members],
                "repeated_after_first": max(0, len(members) - 1),
                "later_inversions_or_repairs": [
                    member["id"]
                    for member in members[1:]
                    if member["audit"]["policy_verdict"] in {"advance", "repair", "informative_failure"}
                ],
            }
        )
    result.sort(key=lambda item: (-item["count"], item["failure_family"]))
    return result


def explicit_references(episodes: list[dict[str, Any]]) -> list[dict[str, str]]:
    names = {episode["id"] for episode in episodes}
    edges = []
    for target in episodes:
        text = "\n".join(
            [target["decision_snapshot"], target["outcome_evidence"], *target["commit_subjects"]]
        )
        for source in names:
            numeric_alias = source.split("-", 1)[0]
            exact_name = re.search(
                rf"(?<![a-z0-9-]){re.escape(source)}(?![a-z0-9-])",
                text,
                re.I,
            )
            numeric_reference = re.search(
                rf"(?<![0-9]){re.escape(numeric_alias)}(?![0-9])",
                text,
                re.I,
            )
            if source != target["id"] and (exact_name or numeric_reference):
                edges.append({"type": "explicit_reference", "source": source, "target": target["id"]})
    return edges


def state_slices(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slices = []
    for target in episodes:
        eligible = eligible_before(episodes, target)
        ids = sorted(item["id"] for item in eligible)
        slices.append(
            {
                "decision_episode": target["id"],
                "cutoff": target["decision_at"],
                "eligible_completed_episodes": len(ids),
                "eligible_ids_sha256": fingerprint(ids),
                "hindsight_exposure": target["hindsight_exposure"],
            }
        )
    return slices


def rank_current_actions(
    episodes: list[dict[str, Any]],
    current_pairs: dict[str, list[dict[str, Any]]],
    priors: dict[str, dict[str, float]],
) -> dict[str, list[dict[str, Any]]]:
    compass_result = json.loads(COMPASS_RESULT.read_text(encoding="utf-8"))
    episode_by_id = {episode["id"]: episode for episode in episodes}
    rankings: dict[str, list[dict[str, Any]]] = {}
    for obligation_id in compass.OBLIGATIONS:
        if obligation_id == "stuck_generic":
            continue
        candidates = []
        for edge in current_pairs.get(obligation_id, []):
            prior_name = PAIR_MOVE_TO_PRIOR[edge["research_move"]]
            prior = priors[prior_name]["mean_useful_transition"]
            requires_authorization = episode_by_id[edge["source_id"]][
                "requires_foundation_authorization"
            ]
            licensing_factor = 0.25 if requires_authorization else 1.0
            candidates.append(
                {
                    **edge,
                    "candidate_type": "historical_episode_transfer",
                    "operation_prior": prior,
                    "requires_foundation_authorization": requires_authorization,
                    "policy_score": edge["research_value"]
                    * (0.75 + 0.50 * prior)
                    * licensing_factor,
                }
            )
        for edge in compass_result["obligations"][obligation_id]["edges"]:
            prior_name = PAIR_MOVE_TO_PRIOR[edge["research_move"]]
            prior = priors[prior_name]["mean_useful_transition"]
            source_id = edge["source_id"]
            if source_id.startswith("code:"):
                candidate_type = "provisional_code_transfer"
            elif source_id.startswith("failure_"):
                candidate_type = "failure_transfer"
            elif source_id.startswith("C-"):
                candidate_type = "accepted_claim_transfer"
            else:
                candidate_type = "semantic_compass_transfer"
            candidates.append(
                {
                    "source_id": source_id,
                    "candidate_type": candidate_type,
                    "relation": edge["relation"],
                    "research_move": edge["research_move"],
                    "missing_bridge": edge["missing_bridge"],
                    "approach_information": edge["approach_information"],
                    "structural_correspondence": edge["structural_correspondence"],
                    "scope_conflict": edge["scope_conflict"],
                    "research_value": edge["research_value"],
                    "operation_prior": prior,
                    "policy_score": edge["research_value"] * (0.75 + 0.50 * prior),
                }
            )
        candidates.sort(
            key=lambda item: (
                -item["policy_score"],
                -item["approach_information"],
                item["source_id"],
            )
        )
        selected = candidates[:12]
        anchor_types = {
            "historical_episode_transfer",
            "accepted_claim_transfer",
            "provisional_code_transfer",
            "failure_transfer",
        }
        for candidate_type in anchor_types:
            anchor = next(
                (
                    item
                    for item in candidates
                    if item["candidate_type"] == candidate_type
                ),
                None,
            )
            if anchor is not None:
                anchor["channel_anchor"] = True
                if anchor not in selected:
                    selected.append(anchor)
        selected.sort(
            key=lambda item: (
                -item["policy_score"],
                -item["approach_information"],
                item["source_id"],
            )
        )
        rankings[obligation_id] = selected
    return rankings


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# TypeSafe time-sliced research policy",
        "",
        f"- historical episodes: {result['counts']['episodes']}",
        f"- detailed decision audits: {result['counts']['decision_audits']}",
        f"- historical counterfactual edges: {result['counts']['historical_counterfactual_edges']}",
        f"- current next-action candidates: {result['counts']['current_action_candidates']}",
        f"- exact-failure stuck interventions: {result['counts']['stuck_interventions']}",
        f"- TypeSafe model: `{result['model']}`",
        "",
        "Chronology and authority are deterministic. TypeSafe judgments are provisional semantic policy evidence, not proof and not accepted authority.",
        "",
        "## Learned research-operation priors",
        "",
        "| operation | posterior useful-transition mean | effective episode mass |",
        "| --- | ---: | ---: |",
    ]
    for name, prior in sorted(
        result["operation_priors"].items(),
        key=lambda item: (-item[1]["mean_useful_transition"], item[0]),
    ):
        lines.append(
            f"| `{name}` | {prior['mean_useful_transition']:.3f} | {prior['effective_episode_mass']:.1f} |"
        )
    lines += ["", "## Repeated-failure map", ""]
    for item in result["repeated_failures"]:
        lines.append(
            f"- `{item['failure_family']}`: {item['count']} episodes, {item['first_episode']} → {item['last_episode']}; later repairs/inversions: {', '.join(item['later_inversions_or_repairs']) or 'none classified'}"
        )
    lines += ["", "## Historical decision audits", ""]
    for audit in result["historical_decision_audits"]:
        top = audit["counterfactuals"][0] if audit["counterfactuals"] else None
        alternative = (
            f"{top['source_id']} via {top['research_move']} ({top['missed_opportunity_strength']:.3f})"
            if top
            else "no eligible source"
        )
        lines += [
            f"### {audit['episode_id']} — {audit['decision_at']}",
            "",
            f"- chosen policy verdict: `{audit['chosen']['policy_verdict']}`",
            f"- chosen operation: `{audit['chosen']['action']}`",
            f"- decision utility: {audit['chosen']['utility']:.3f}",
            f"- time-eligible completed sources: {audit['eligible_source_count']}",
            f"- strongest counterfactual: {alternative}",
            f"- hindsight exposure: `{audit['hindsight_exposure']}`",
            "",
        ]
    lines += ["## Missed-opportunity map", ""]
    for item in result["missed_opportunities"][:30]:
        lines.append(
            f"- `{item['target_episode']}` could have consulted `{item['source_id']}` through `{item['research_move']}`; strength {item['missed_opportunity_strength']:.3f}, bridge `{item['missing_bridge']}`."
        )
    lines += ["", "## Hindsight-sensitive policy leads", ""]
    for item in result["hindsight_sensitive_leads"][:20]:
        lines.append(
            f"- `{item['target_episode']}` and `{item['source_id']}` form a promising `{item['research_move']}` transfer (strength {item['missed_opportunity_strength']:.3f}), but the first visible decision snapshot already contains outcome information."
        )
    lines += ["", "## Counterfactual next-action rankings", ""]
    for obligation_id, candidates in result["current_action_rankings"].items():
        lines += [f"### {obligation_id}", ""]
        for rank, item in enumerate(candidates[:6], start=1):
            licensing = (
                "; owner foundation authorization required"
                if item.get("requires_foundation_authorization")
                else ""
            )
            lines.append(
                f"{rank}. `{item['source_id']}` via `{item['research_move']}` → `{item['missing_bridge']}`; policy score {item['policy_score']:.3f}{licensing}."
            )
        anchors = [item["source_id"] for item in candidates if item.get("channel_anchor")]
        lines.append(f"Channel anchors: {', '.join(f'`{source_id}`' for source_id in anchors)}.")
        lines.append("")
    lines += ["## Exact-failure stuck interventions", ""]
    for failure_id, intervention in result["stuck_interventions"].items():
        lines += [
            f"### {failure_id}",
            "",
            f"Failure: {intervention['failure_statement']}",
            "",
        ]
        for rank, item in enumerate(intervention["candidates"][:5], start=1):
            licensing = (
                "; owner foundation authorization required"
                if item.get("requires_foundation_authorization")
                else ""
            )
            lines.append(
                f"{rank}. Reuse `{item['source_id']}` through `{item['research_move']}`; construct `{item['missing_bridge']}` (value {item['research_value']:.3f}{licensing})."
            )
        lines.append("")
    lines += [
        "## Interpretation boundary",
        "",
        "A counterfactual is a time-eligible research lead, not proof that the historical route would have succeeded. A missed-opportunity flag additionally requires a cleaner multi-commit decision snapshot, high estimated information or closure value, structural correspondence, and no explicit reference. Result-bearing first snapshots are reported separately as hindsight-sensitive leads. The operation priors are semantic empirical Bayes summaries of these typed audits; future reviewed outcomes should replace model-only labels.",
        "",
    ]
    return "\n".join(line.rstrip() for line in lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--audit-limit", type=int, default=32)
    parser.add_argument("--counterfactuals", type=int, default=6)
    parser.add_argument("--current-sources", type=int, default=10)
    args = parser.parse_args()

    episodes, excluded = build_episode_cards()
    log = load_log()

    episode_q = episode_questions()
    source_q = compass.source_questions()
    jobs = []
    for episode in episodes:
        jobs.append(("episode", episode["id"], episode, episode_q))
        jobs.append(("trajectory_source", episode["id"], source_state(episode), source_q))
    log = run_jobs(jobs, log, args.workers, "episode/source")

    for episode in episodes:
        episode["audit"] = episode_audit(log[f"episode:{episode['id']}"]["response"])
        episode["source_vector"] = compass.vector(
            log[f"trajectory_source:{episode['id']}"]["response"]
        )

    selected = select_decision_audits(episodes, args.audit_limit)
    need_jobs = []
    for target in selected:
        eligible = eligible_before(episodes, target)
        target["predecision_state"] = predecision_state(target, eligible)
        need_jobs.append(
            (
                "historical_need",
                target["id"],
                target["predecision_state"],
                compass.need_questions(),
            )
        )
    log = run_jobs(need_jobs, log, args.workers, "historical need")

    source_vectors = {episode["id"]: episode["source_vector"] for episode in episodes}
    rarity = compass.rarity_weights(source_vectors)
    historical_pair_jobs = []
    historical_sources: dict[str, list[dict[str, Any]]] = {}
    historical_profiles: dict[str, dict[str, Any]] = {}
    for target in selected:
        eligible = eligible_before(episodes, target)
        profile = compass.need_profile(log[f"historical_need:{target['id']}"]["response"])
        historical_profiles[target["id"]] = profile
        sources = retrieve_prior_sources(
            target, eligible, profile, rarity, args.counterfactuals
        )
        historical_sources[target["id"]] = sources
        for source in sources:
            item_id = f"{target['id']}|{source['id']}"
            state = {
                "obligation": target["predecision_state"],
                "source": source_state(source),
                "chronology": {
                    "source_available_at": source["available_at"],
                    "decision_at": target["decision_at"],
                    "later_outcomes_hidden": True,
                },
            }
            historical_pair_jobs.append(
                ("historical_pair", item_id, state, compass.pair_questions())
            )
    log = run_jobs(historical_pair_jobs, log, args.workers, "historical pair")

    reference_edges = explicit_references(episodes)
    referenced = {(edge["source"], edge["target"]) for edge in reference_edges}
    priors = learned_operation_priors(episodes)
    historical_audits = []
    missed = []
    hindsight_sensitive = []
    counterfactual_edges = []
    for target in selected:
        counterfactuals = []
        for source in historical_sources[target["id"]]:
            item_id = f"{target['id']}|{source['id']}"
            edge = pair_edge(
                log[f"historical_pair:{item_id}"]["response"],
                source["source_vector"],
                historical_profiles[target["id"]],
            )
            prior_name = PAIR_MOVE_TO_PRIOR[edge["research_move"]]
            prior = priors[prior_name]["mean_useful_transition"]
            strength = (
                edge["research_value"]
                * (0.70 + 0.60 * prior)
                * (1.0 - 0.55 * target["audit"]["utility"])
                * (0.35 + 0.65 * target["audit"]["decision_likelihood"])
            )
            consumed = (source["id"], target["id"]) in referenced
            record = {
                "source_id": source["id"],
                "target_episode": target["id"],
                "source_available_at": source["available_at"],
                "target_decision_at": target["decision_at"],
                "explicitly_referenced": consumed,
                "requires_foundation_authorization": source[
                    "requires_foundation_authorization"
                ],
                "operation_prior": prior,
                "missed_opportunity_strength": (
                    0.0
                    if consumed or source["requires_foundation_authorization"]
                    else strength
                ),
                **edge,
            }
            counterfactuals.append(record)
            counterfactual_edges.append(record)
            if (
                not consumed
                and not source["requires_foundation_authorization"]
                and strength >= 0.025
                and edge["structural_correspondence"] >= 0.45
            ):
                if target["hindsight_exposure"] == "cleaner_multi_commit":
                    record["historical_policy_confidence"] = "cleaner_time_slice"
                    missed.append(record)
                else:
                    record["historical_policy_confidence"] = "hindsight_sensitive"
                    hindsight_sensitive.append(record)
        counterfactuals.sort(
            key=lambda item: (-item["missed_opportunity_strength"], item["source_id"])
        )
        eligible = eligible_before(episodes, target)
        historical_audits.append(
            {
                "episode_id": target["id"],
                "decision_at": target["decision_at"],
                "available_at": target["available_at"],
                "eligible_source_count": len(eligible),
                "eligible_ids_sha256": fingerprint(sorted(item["id"] for item in eligible)),
                "hindsight_exposure": target["hindsight_exposure"],
                "chosen": target["audit"],
                "counterfactuals": counterfactuals,
            }
        )
    missed.sort(
        key=lambda item: (-item["missed_opportunity_strength"], item["target_episode"], item["source_id"])
    )
    hindsight_sensitive.sort(
        key=lambda item: (-item["missed_opportunity_strength"], item["target_episode"], item["source_id"])
    )

    current_profiles = json.loads(COMPASS_RESULT.read_text(encoding="utf-8"))["obligations"]
    current_pair_jobs = []
    current_sources: dict[str, list[dict[str, Any]]] = {}
    for obligation_id, obligation in compass.OBLIGATIONS.items():
        if obligation_id == "stuck_generic":
            continue
        profile = current_profiles[obligation_id]["need_profile"]
        proxy = {"predecision_state": obligation, "audit": {"exact_failure_named": 0.7}}
        sources = retrieve_prior_sources(proxy, episodes, profile, rarity, args.current_sources)
        current_sources[obligation_id] = sources
        for source in sources:
            item_id = f"{obligation_id}|{source['id']}"
            state = {"obligation": obligation, "source": source_state(source)}
            current_pair_jobs.append(("current_pair", item_id, state, compass.pair_questions()))
    log = run_jobs(current_pair_jobs, log, args.workers, "current pair")

    current_pairs: dict[str, list[dict[str, Any]]] = {}
    for obligation_id, sources in current_sources.items():
        current_pairs[obligation_id] = []
        for source in sources:
            item_id = f"{obligation_id}|{source['id']}"
            current_pairs[obligation_id].append(
                {
                    "source_id": source["id"],
                    **pair_edge(
                        log[f"current_pair:{item_id}"]["response"],
                        source["source_vector"],
                        current_profiles[obligation_id]["need_profile"],
                    ),
                }
            )
    current_rankings = rank_current_actions(episodes, current_pairs, priors)

    stuck_pair_jobs = []
    stuck_need_jobs = []
    stuck_sources: dict[str, list[dict[str, Any]]] = {}
    stuck_obligations: dict[str, dict[str, Any]] = {}
    for failure_id, failure in compass.FRONTIER_NODES.items():
        failure_obligation = {
            "title": f"Stuck intervention for {failure['title']}",
            "parent": "Resume a campaign from the exact named failure without repeating the failed representation.",
            "positive_intent": "Use historically successful constructions, repairs, analogies, or failure inversions to design a materially different next approach.",
            "current_inputs": failure,
            "obstruction": failure["statement"],
            "exclusions": "Generic advice, later-result leakage, repetition of the failed representation, and silent changes to the frozen parent objective are excluded.",
        }
        stuck_obligations[failure_id] = failure_obligation
        stuck_need_jobs.append(
            ("stuck_need", failure_id, failure_obligation, compass.need_questions())
        )
    log = run_jobs(stuck_need_jobs, log, args.workers, "stuck need")
    stuck_profiles = {
        failure_id: compass.need_profile(log[f"stuck_need:{failure_id}"]["response"])
        for failure_id in compass.FRONTIER_NODES
    }
    for failure_id, failure in compass.FRONTIER_NODES.items():
        failure_obligation = stuck_obligations[failure_id]
        failure_profile = stuck_profiles[failure_id]
        ranked = sorted(
            episodes,
            key=lambda episode: (
                -(
                    0.50 * compass.distribution_match(
                        episode["source_vector"], failure_profile, rarity
                    )
                    + 0.20 * episode["source_vector"]["failure_inversion"]
                    + 0.15 * episode["audit"]["future_reuse"]
                    + 0.15 * lexical_similarity(source_state(episode), failure_obligation)
                ),
                episode["id"],
            ),
        )[:8]
        stuck_sources[failure_id] = ranked
        for source in ranked:
            item_id = f"{failure_id}|{source['id']}"
            state = {"obligation": failure_obligation, "source": source_state(source)}
            stuck_pair_jobs.append(("stuck_pair", item_id, state, compass.pair_questions()))
    log = run_jobs(stuck_pair_jobs, log, args.workers, "stuck pair")

    stuck_interventions = {}
    for failure_id, failure in compass.FRONTIER_NODES.items():
        candidates = []
        for source in stuck_sources[failure_id]:
            item_id = f"{failure_id}|{source['id']}"
            edge = pair_edge(
                log[f"stuck_pair:{item_id}"]["response"],
                source["source_vector"],
                stuck_profiles[failure_id],
            )
            prior_name = PAIR_MOVE_TO_PRIOR[edge["research_move"]]
            edge["research_value"] *= 0.75 + 0.50 * priors[prior_name]["mean_useful_transition"]
            if source["requires_foundation_authorization"]:
                edge["research_value"] *= 0.25
            candidates.append(
                {
                    "source_id": source["id"],
                    "requires_foundation_authorization": source[
                        "requires_foundation_authorization"
                    ],
                    **edge,
                }
            )
        candidates.sort(key=lambda item: (-item["research_value"], item["source_id"]))
        stuck_interventions[failure_id] = {
            "failure_statement": failure["statement"],
            "source": failure["source"],
            "candidates": candidates,
        }

    nodes = []
    for episode in episodes:
        nodes.append(
            {
                key: episode[key]
                for key in (
                    "id",
                    "path",
                    "decision_at",
                    "available_at",
                    "first_commit",
                    "last_commit",
                    "commit_count",
                    "hindsight_exposure",
                    "requires_foundation_authorization",
                    "content_sha256",
                    "audit",
                    "source_vector",
                )
            }
        )
    graph_edges = [
        {
            "type": "chronological_successor",
            "source": episodes[index - 1]["id"],
            "target": episodes[index]["id"],
        }
        for index in range(1, len(episodes))
    ]
    graph_edges.extend(reference_edges)
    repeated = repeat_failure_map(episodes)
    for item in repeated:
        for left, right in zip(item["episodes"], item["episodes"][1:]):
            graph_edges.append(
                {
                    "type": "repeated_failure_family",
                    "failure_family": item["failure_family"],
                    "source": left,
                    "target": right,
                }
            )

    used_keys = {
        *(f"episode:{episode['id']}" for episode in episodes),
        *(f"trajectory_source:{episode['id']}" for episode in episodes),
        *(f"historical_need:{target['id']}" for target in selected),
        *(f"historical_pair:{target['id']}|{source['id']}" for target in selected for source in historical_sources[target["id"]]),
        *(f"current_pair:{obligation_id}|{source['id']}" for obligation_id, sources in current_sources.items() for source in sources),
        *(f"stuck_need:{failure_id}" for failure_id in compass.FRONTIER_NODES),
        *(f"stuck_pair:{failure_id}|{source['id']}" for failure_id, sources in stuck_sources.items() for source in sources),
    }
    input_tokens = sum(
        int(log[key]["response"].get("usage", {}).get("input_tokens", 0))
        for key in used_keys
    )
    output_tokens = sum(
        int(log[key]["response"].get("usage", {}).get("output_tokens", 0))
        for key in used_keys
    )
    models = {
        log[key]["response"].get("model", compass.MODEL)
        for key in used_keys
    }
    result = {
        "schema_version": 1,
        "model": sorted(models)[0] if len(models) == 1 else sorted(models),
        "authority_boundary": "Chronology, git provenance, accepted authority, eligibility, hashes, and aggregation are deterministic. TypeSafe supplies provisional semantic judgments only.",
        "campaign": "P253-euler-particle-mechanisms",
        "base_release": "v0.183.0",
        "generated_from": {
            "proposal_sha256": hashlib.sha256(PROPOSAL.read_bytes()).hexdigest(),
            "head": git("rev-parse", "HEAD").strip(),
            "response_log_sha256": hashlib.sha256(RESPONSE_LOG.read_bytes()).hexdigest(),
        },
        "counts": {
            "episodes": len(episodes),
            "excluded_uncommitted_attempts": len(excluded),
            "decision_audits": len(historical_audits),
            "historical_counterfactual_edges": len(counterfactual_edges),
            "missed_opportunities": len(missed),
            "hindsight_sensitive_leads": len(hindsight_sensitive),
            "current_action_candidates": sum(len(items) for items in current_rankings.values()),
            "stuck_interventions": len(stuck_interventions),
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "excluded_attempts": excluded,
        "nodes": nodes,
        "edges": graph_edges,
        "time_slices": state_slices(episodes),
        "historical_decision_audits": historical_audits,
        "missed_opportunities": missed,
        "hindsight_sensitive_leads": hindsight_sensitive,
        "repeated_failures": repeated,
        "operation_priors": priors,
        "current_action_rankings": current_rankings,
        "stuck_interventions": stuck_interventions,
    }
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_FILE.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "usage": result["usage"]}, indent=2))


if __name__ == "__main__":
    main()

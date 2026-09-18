#!/usr/bin/env python3
"""Build a provisional TypeSafe semantic research compass.

The accepted registry remains authoritative.  This script gives each accepted
claim and each live research obstruction a reusable vector of semantic
capabilities and research moves, then asks TypeSafe to adjudicate only the most
promising claim/obligation transfer edges.  The output is an inspiration and
context-retrieval layer for a reasoning agent; it is never a proof or a claim
dependency.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import random
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REGISTRY = ROOT / "governance/claims.yaml"
RELEASE = ROOT / "governance/releases/current.yaml"
SECRET_FILE = Path.home() / ".config/typesafe/secrets.env"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
RESPONSE_LOG = HERE / "research-compass-responses.jsonl"
RESULT_FILE = HERE / "research-compass.json"
REPORT_FILE = HERE / "research-compass-report.md"
WRITE_LOCK = threading.Lock()


# These are reusable semantic coordinates, not a taxonomy of physics.  A claim
# can occupy several coordinates.  They combine content that an obligation may
# need with operations that a previous construction can teach to a future one.
FEATURES: dict[str, dict[str, str]] = {
    "variational_structure": {
        "source": "The claim supplies a reusable action, energy, Hamiltonian, constrained variation, or stationary functional structure.",
        "need": "Progress on this obligation would materially benefit from a usable action, energy, Hamiltonian, constrained variation, or stationary functional.",
    },
    "symmetry_conservation": {
        "source": "The claim supplies a reusable symmetry, conserved quantity, momentum map, invariant, or orbit reduction.",
        "need": "Progress on this obligation would materially benefit from a symmetry, conserved quantity, momentum map, invariant, or orbit reduction.",
    },
    "topology_discrete_structure": {
        "source": "The claim supplies a reusable topological partition, integer/discrete label, holonomy, degree, orbit type, or global compatibility constraint.",
        "need": "Progress on this obligation would materially benefit from a topological partition, integer/discrete label, holonomy, degree, orbit type, or global compatibility constraint.",
    },
    "spectral_resolvent_structure": {
        "source": "The claim supplies a reusable spectral, Green-operator, resolvent, resonance, adjoint, kernel, or Fredholm construction.",
        "need": "Progress on this obligation would materially benefit from spectral, Green-operator, resolvent, resonance, adjoint, kernel, or Fredholm structure.",
    },
    "stability_coercivity": {
        "source": "The claim supplies a reusable stability, coercivity, Hessian, spectral-gap, persistence, or controlled-soft-mode construction.",
        "need": "Progress on this obligation would materially benefit from stability, coercivity, Hessian, spectral-gap, persistence, or controlled-soft-mode structure.",
    },
    "interaction_propagation": {
        "source": "The claim supplies a reusable interaction, force, scattering, radiation, propagation, tail, or multi-object construction.",
        "need": "Progress on this obligation would materially benefit from an interaction, force, scattering, radiation, propagation, tail, or multi-object construction.",
    },
    "charge_gauge_current": {
        "source": "The claim supplies a reusable signed conserved charge, gauge connection/covariance, flux/holonomy, Noether current, sourced-field response, or charge-generating process. Generic coefficients, weights, mechanical rows, or the word current in another sense do not count.",
        "need": "Progress on this obligation would materially benefit from a charge, gauge, connection, flux, current, response, or sourced-field construction.",
    },
    "quantization_statistics_measurement": {
        "source": "The claim supplies a reusable Hilbert/Fock state construction, quantum amplitude or Born rule, exchange/statistics phase, operator measurement, prequantization/integrality condition, or physical detector model. Ordinary mixture weights, classical random ensembles, and generic probabilities do not count.",
        "need": "Progress on this obligation would materially benefit from quantization, amplitude, exchange/statistics, state-space, probability, measurement, or detector structure.",
    },
    "relativistic_causal_structure": {
        "source": "The claim supplies a reusable relativistic, Lorentz, causal, worldline, light-cone, or covariant propagation construction.",
        "need": "Progress on this obligation would materially benefit from relativistic, Lorentz, causal, worldline, light-cone, or covariant propagation structure.",
    },
    "scale_normalization_units": {
        "source": "The claim supplies a reusable scale-selection, nondimensionalization, normalization, unit, residue, coefficient-matching, or convention-control construction.",
        "need": "Progress on this obligation would materially benefit from scale-selection, nondimensionalization, normalization, unit, residue, coefficient matching, or convention control.",
    },
    "localized_coherent_object": {
        "source": "The claim supplies a reusable localized, finite-energy, compact, coherent, solitary, periodic-carrier, or orbitally persistent object construction rather than only an algebraic identity.",
        "need": "Progress on this obligation would materially benefit from a localized, finite-energy, compact, coherent, solitary, periodic-carrier, or orbitally persistent object construction.",
    },
    "internal_basis_mixing_chirality": {
        "source": "The claim supplies a reusable internal-basis, representation, multiplet, mixing-matrix, oscillation-phase, chirality, helicity, or exchange-of-basis construction.",
        "need": "Progress on this obligation would materially benefit from an internal-basis, representation, multiplet, mixing-matrix, oscillation-phase, chirality, helicity, or exchange-of-basis construction.",
    },
    "nonlinear_pde_matching": {
        "source": "The claim supplies a reusable nonlinear PDE existence, continuation, inner/outer matching, boundary/exterior reconstruction, gluing, or global compatibility construction.",
        "need": "Progress on this obligation would materially benefit from nonlinear PDE existence, continuation, inner/outer matching, boundary/exterior reconstruction, gluing, or global compatibility structure.",
    },
    "representation_change": {
        "source": "The claim's main transferable move is a change of variables, representation, admissible class, frame, gauge, chart, or equivalent formulation that exposes new structure.",
        "need": "The present obstruction plausibly calls for a change of variables, representation, admissible class, frame, gauge, chart, or equivalent formulation.",
    },
    "controlled_limit_asymptotic": {
        "source": "The claim's main transferable move is a controlled limit, asymptotic expansion, scaling regime, matching argument, or exact solvable slice.",
        "need": "The present obstruction plausibly calls for a controlled limit, asymptotic expansion, scaling regime, matching argument, or exact solvable slice.",
    },
    "inverse_construction": {
        "source": "The claim's main transferable move constructs an object backward from required consequences, constraints, kernels, observables, or boundary data.",
        "need": "The present obstruction plausibly calls for constructing an object backward from required consequences, constraints, kernels, observables, or boundary data.",
    },
    "decomposition_factorization": {
        "source": "The claim's main transferable move decomposes sectors, factors an operator or expression, partitions cases, or isolates independently controllable components.",
        "need": "The present obstruction plausibly calls for decomposing sectors, factoring an operator or expression, partitioning cases, or isolating independently controllable components.",
    },
    "composition_bridge": {
        "source": "The claim's main transferable move composes previously separate accepted ingredients through explicit glue while keeping assumptions and scopes visible.",
        "need": "The present obstruction plausibly calls for composing separate ingredients through explicit glue while keeping assumptions and scopes visible.",
    },
    "oracle_sensitivity_design": {
        "source": "The claim supplies a reusable exposing oracle, mutation, sensitivity test, residual, convergence check, or discriminating observable.",
        "need": "The present obstruction plausibly calls for a new exposing oracle, mutation, sensitivity test, residual, convergence check, or discriminating observable.",
    },
    "failure_inversion": {
        "source": "The claim contains a named obstruction, counterexample, boundary, or failed mechanism whose exact failure can be inverted into a stronger construction or route choice.",
        "need": "The present obstruction would benefit from a named failure mechanism that can be inverted into a stronger construction or route choice.",
    },
    "assumption_scope_repair": {
        "source": "The claim's main transferable value is exposing and repairing an applicability, scope, regularity, authority, model-to-physical, or hidden-assumption boundary.",
        "need": "The present obstruction plausibly calls for exposing and repairing an applicability, scope, regularity, authority, model-to-physical, or hidden-assumption boundary.",
    },
}

CONTENT_FEATURES = (
    "variational_structure",
    "symmetry_conservation",
    "topology_discrete_structure",
    "spectral_resolvent_structure",
    "stability_coercivity",
    "interaction_propagation",
    "charge_gauge_current",
    "quantization_statistics_measurement",
    "relativistic_causal_structure",
    "scale_normalization_units",
    "localized_coherent_object",
    "internal_basis_mixing_chirality",
    "nonlinear_pde_matching",
)
MOVE_FEATURES = tuple(name for name in FEATURES if name not in CONTENT_FEATURES)


OBLIGATIONS: dict[str, dict[str, Any]] = {
    "p253_partition2_actual_carrier": {
        "title": "Construct the partition-2 smooth branch on the actual Euler carrier",
        "parent": "P253 P2 persistent finite-energy carrier",
        "positive_intent": (
            "Use the exact partition discovered in attempt 0161 to construct smooth "
            "compatible-centralizer matching and a separatrix on the actual Cao carrier, "
            "then reconstruct the rotating Euler branch and later prove persistence."
        ),
        "current_inputs": (
            "At a nondegenerate zeta-critical point with indefinite Hessian, the model "
            "centralizer has a hyperbolic orbit; model residue and KKS orientation signs "
            "are available at fixed n."
        ),
        "obstruction": (
            "No actual-carrier critical point, global boundary/exterior continuation, "
            "smooth invariant-compatible matching, continuum coefficient, or nonlinear "
            "persistence theorem has yet been constructed."
        ),
        "exclusions": (
            "A model separatrix, a regular-core layer, or a one-block witness does not by "
            "itself establish a physical carrier or stability."
        ),
        "sources": [
            "proposals/P253-euler-particle-mechanisms/attempts/0161-sage-0062branch/05-0062r5-branch.md"
        ],
    },
    "p253_continuum_trace": {
        "title": "Upgrade the fixed-mode trace and residue to the physical continuum carrier",
        "parent": "P253 P2 carrier branch dependency",
        "positive_intent": (
            "Construct the source-specific continuum limiting-absorption/Grushin trace, "
            "distorted adjoint, finite rows, nonzero physical residue, dimensions, and "
            "delta scaling on the actual carrier."
        ),
        "current_inputs": (
            "Attempt 0161 has exact two-mode identities, a source-bearing model resonance, "
            "a model distorted adjoint, and a dimensionless nonzero residue witness."
        ),
        "obstruction": (
            "The continuum Green/Leray block, analytic remainder, limiting absorption, "
            "physical units, uniformity, and physical KKS normalization are absent."
        ),
        "exclusions": "Finite-matrix identities do not establish continuum boundedness or a physical residue.",
        "sources": [
            "proposals/P253-euler-particle-mechanisms/attempts/0161-sage-0062branch/03-0062r3-trace.md",
            "proposals/P253-euler-particle-mechanisms/attempts/0161-sage-0062branch/04-0062r4-vstar.md",
        ],
    },
    "p253_quantum_bridge": {
        "title": "Derive a physical quantum/statistics bridge from an Euler carrier",
        "parent": "P253 P4 quantum behavior and particle identity",
        "positive_intent": (
            "Connect a physical Euler orbit to normalized amplitudes, a discrete action "
            "unit, exchange/statistics, measurement probabilities, and relativistic or "
            "clearly scoped propagation without mistaking classical mode algebra for quantum physics."
        ),
        "current_inputs": (
            "Accepted claims include KKS/symplectic structures, exact oscillator and coherent-state "
            "mathematics, conditional exchange constructions, and classical Euler carriers."
        ),
        "obstruction": (
            "No accepted edge derives physical amplitudes, Born probabilities, fermionic exchange, "
            "an intrinsic action normalization, or relativistic propagation from an Euler carrier."
        ),
        "exclusions": (
            "A formal Fock representation, conditional exchange phase, or freely declared quantum "
            "postulate does not establish emergence from the substrate."
        ),
        "sources": [
            "proposals/P253-euler-particle-mechanisms/attempts/0113-beacon-p4audit/p4audit.md"
        ],
    },
    "p253_charge_converter": {
        "title": "Construct or decisively scope an Euler-native charge mechanism",
        "parent": "P253 P5 electron charge/current",
        "positive_intent": (
            "Derive a conserved signed charge, current, sourcing law, and observable coupling from "
            "the substrate, including testing whether a steady two-allotrope conversion process can "
            "carry charge without silently adding a constitutive law."
        ),
        "current_inputs": (
            "The framework contains topological labels, helicity/orientation structures, gauge/current "
            "comparators, transport laws, and an exploratory conversion-rate idea from issue 214."
        ),
        "obstruction": (
            "There is no Euler free energy or conversion functional for two allotropes, no compact "
            "converter solution, and no derived electromagnetic sourcing/coupling law."
        ),
        "exclusions": "A static label or imported gauge charge does not establish an Euler-native electron charge.",
        "sources": ["https://github.com/vantasnerdan/substrate-framework/issues/214"],
    },
    "p253_neutrino_mixing": {
        "title": "Find a substrate mechanism for neutral chiral mixing and oscillation",
        "parent": "P253 P6 neutrino mechanism",
        "positive_intent": (
            "Construct neutral propagating substrate states with a physically defined internal basis, "
            "mixing dynamics, oscillation phases, chirality/helicity behavior, and observable couplings."
        ),
        "current_inputs": (
            "The accepted release contains exact mixing, phase, representation, propagation, and "
            "topological constructions, but no accepted neutrino synthesis."
        ),
        "obstruction": (
            "No Euler carrier, mixing Hamiltonian, physical phase normalization, weak coupling, or "
            "chiral selection mechanism has been connected into a neutrino object."
        ),
        "exclusions": "A generic matrix oscillation identity or analogy alone is not a neutrino mechanism.",
        "sources": ["proposals/P253-euler-particle-mechanisms/proposal.yaml"],
    },
    "stuck_generic": {
        "title": "A campaign is structurally stuck after its favored representation failed",
        "parent": "Reusable future-campaign intervention",
        "positive_intent": (
            "Surface materially different representations, transferable constructions, controlled "
            "limits, inverse problems, exposing oracles, and failure-derived mechanisms that help a "
            "research agent create and test the next approach."
        ),
        "current_inputs": "The exact failed attempt and its assumptions are available, along with the accepted claim corpus.",
        "obstruction": (
            "The current ansatz cannot supply a load-bearing implication, and local variations repeat "
            "the same structural failure."
        ),
        "exclusions": (
            "The intervention should not merely rank tasks, restate the missing lemma, or favor a quick result."
        ),
        "sources": ["AGENTS.md discovery and continuation contract"],
    },
}


# Prototype seeds from current-source/GitNexus inspection.  In future campaigns
# this channel should be populated from current imports, callers, matching tests,
# and a concept query before semantic reranking.
CODE_SEEDS: dict[str, list[str]] = {
    "p253_partition2_actual_carrier": [
        "code:euler_cao_schur",
        "code:euler_compact_ring",
        "code:euler_p2_principal",
        "code:euler_axisymmetric_action",
        "code:euler_two_label_lock",
    ],
    "p253_continuum_trace": [
        "code:euler_cao_schur",
        "code:euler_retained_memory",
        "code:euler_p2_principal",
        "code:euler_asymptotic_tails",
        "code:euler_two_label_lock",
    ],
    "p253_quantum_bridge": [
        "code:euler_quantum_bridge",
        "code:euler_quantum_two_state",
        "code:euler_schwinger_hopf",
        "code:euler_measurement_bridge",
        "code:euler_gauge_action_selection",
    ],
    "p253_charge_converter": [
        "code:euler_gauge_action_selection",
        "code:euler_gauge_current",
        "code:euler_ertel_current",
        "code:euler_two_label_lock",
        "code:euler_charged_hessian",
    ],
    "p253_neutrino_mixing": [
        "code:euler_neutrino_suppliers",
        "code:euler_neutral_cell",
        "code:euler_schwinger_hopf",
        "code:euler_quantum_two_state",
        "code:euler_measurement_bridge",
    ],
    "stuck_generic": [],
}


FRONTIER_NODES: dict[str, dict[str, Any]] = {
    "failure_regular_trace_layer": {
        "node_type": "failed_mechanism",
        "title": "Regular-core smooth pendulum layer is incompatible with the centralizer flow",
        "statement": (
            "At F'(zeta) nonzero on a regular zeta torus the centralizer flow has no fixed point, "
            "so a separatrix layer requiring a hyperbolic fixed orbit cannot form. The failure "
            "redirects construction to a zeta-critical point with indefinite Hessian."
        ),
        "source": "proposals/P253-euler-particle-mechanisms/attempts/0161-sage-0062branch/05-0062r5-branch.md",
    },
    "failure_fprime_isolated_saddle": {
        "node_type": "failed_mechanism",
        "title": "F'(zeta)=0 alone produces a degenerate whole level, not an isolated saddle",
        "statement": (
            "The zero set from F'(zeta)=0 can be a full level. The repaired construction obtains "
            "an isolated hyperbolic orbit from a nondegenerate zeta-critical point and indefinite Hessian."
        ),
        "source": "proposals/P253-euler-particle-mechanisms/attempts/0161-sage-0062branch/05-0062r5-branch.md",
    },
    "failure_false_casimir": {
        "node_type": "failed_mechanism",
        "title": "The orbit-chart coercivity route cannot invent a Casimir on the banked space",
        "statement": (
            "The H^-1 orbit-chart coercivity step failed and the explored A3 space supplied no usable "
            "Casimir. A future persistence construction needs a different constrained Hessian, spectral "
            "gap, modulation, or admissible representation."
        ),
        "source": "proposals/P253-euler-particle-mechanisms/attempts/0155-atlas-chart/chart-table.txt",
    },
    "failure_continuous_action_scale": {
        "node_type": "failed_mechanism",
        "title": "Classical normalization leaves the action scale continuously rescalable",
        "statement": (
            "Classical Euler and KKS normalization alone has not supplied an intrinsic discrete action "
            "unit. A quantum bridge must find a global integrality/holonomy mechanism or state the new postulate."
        ),
        "source": "proposals/P253-euler-particle-mechanisms/attempts/0113-beacon-p4audit/p4audit.md",
    },
    "failure_autonomous_reset": {
        "node_type": "failed_mechanism",
        "title": "Autonomous finite-volume Euler dynamics does not yet supply detector capture/reset",
        "statement": (
            "A first-event probability theorem is conditional on analyzer, clock, capture, and reset. "
            "The missing physical construction is an open scattering/environment mechanism rather than "
            "another probability identity."
        ),
        "source": "proposals/P253-euler-particle-mechanisms/attempts/0113-beacon-p4audit/p4audit.md",
    },
}


RELATION_OPTIONS = {
    "direct_ingredient": "The source supplies an ingredient at nearly the scope the obligation consumes.",
    "transferable_construction": "The source contains a construction that can be adapted after a named applicability proof.",
    "structural_analogy": "The source suggests a materially useful representation or mechanism, but substantial new derivation is required.",
    "failure_to_invert": "The source's boundary or failure mechanism can be inverted to design a stronger candidate or discriminating check.",
    "falsifier_or_oracle": "The source primarily supplies an exposing test, counterexample, or diagnostic for candidate approaches.",
    "conflicting_premise": "The source exposes a premise conflict that must be resolved before using the proposed route.",
    "irrelevant": "The similarity is superficial; the source does not materially inform the obligation."
}

MOVE_OPTIONS = {
    "change_representation": "Change variables, chart, gauge, frame, admissible space, or mathematical representation.",
    "construct_inverse": "Start from the required consequence, observable, kernel, or boundary data and construct backward.",
    "take_controlled_limit": "Use a controlled limit, asymptotic regime, perturbation, or exact solvable slice.",
    "decompose_or_factor": "Split sectors/cases/scales or factor an operator to expose independently controllable pieces.",
    "compose_with_glue": "Join accepted ingredients with an explicit missing bridge and track all assumptions.",
    "repair_scope": "Prove applicability, regularity, units, normalization, or model-to-physical transfer.",
    "design_oracle": "Build a falsifier, mutation, sensitivity check, residual, or observable that distinguishes mechanisms.",
    "invert_failure": "Turn the exact reason a prior route failed into a positive construction criterion.",
    "none": "This source does not suggest a useful research move for the obligation."
}

BRIDGE_OPTIONS = {
    "applicability_theorem": "Prove that the source construction applies to the obligation's actual object and admissible space.",
    "operator_or_variable_map": "Construct an explicit map between variables, operators, symmetries, or representations.",
    "dynamics_or_action": "Supply the missing dynamical law, action, generator, or evolution equation.",
    "global_topology": "Upgrade local algebra to global topology, integrality, holonomy, boundary, or orbit compatibility.",
    "continuum_or_physical_scope": "Upgrade finite/model/conditional structure to continuum, physical, or uniform scope.",
    "scale_or_normalization": "Fix dimensions, normalization, physical coefficients, residue, or an intrinsic scale.",
    "stability_or_persistence": "Prove a coercive, spectral, nonlinear-stability, or persistence bridge.",
    "observable_or_oracle": "Connect the construction to an observable or exposing oracle.",
    "new_postulate": "The transfer would require an explicit new physical postulate rather than derivation from current canon.",
    "none": "No meaningful bridge is visible because the source is irrelevant."
}


def api_key() -> str:
    for line in SECRET_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            value = line.split("=", 1)[1].strip().strip("'\"")
            if value:
                return value
    raise RuntimeError(f"TYPESAFE_API_KEY is missing from {SECRET_FILE}")


def ask(state: dict[str, Any], questions: dict[str, Any], *, max_attempts: int = 7) -> dict[str, Any]:
    payload = json.dumps({"state": state, "model": MODEL, "questions": questions}).encode()
    for attempt in range(max_attempts):
        request = urllib.request.Request(
            ENDPOINT,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {429, 500, 502, 503, 504, 529} or attempt + 1 == max_attempts:
                raise RuntimeError(f"TypeSafe HTTP {exc.code}: {detail}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(0.75 * 2**attempt, 12.0)
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt + 1 == max_attempts:
                raise
            delay = min(0.75 * 2**attempt, 12.0)
        time.sleep(delay + random.random() * 0.25)
    raise AssertionError("unreachable")


def load_accepted_claims() -> list[dict[str, Any]]:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    release = yaml.safe_load(RELEASE.read_text(encoding="utf-8"))
    by_id = {claim["id"]: claim for claim in registry["claims"]}
    return [by_id[claim_id] for claim_id in release["accepted_claims"]]


def load_code_modules() -> dict[str, dict[str, Any]]:
    """Extract provisional Euler module contracts without granting authority."""

    modules: dict[str, dict[str, Any]] = {}
    for path in sorted((ROOT / "src/substrate_framework").glob("euler*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        public_symbols = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name.startswith("_"):
                continue
            symbol: dict[str, Any] = {
                "name": node.name,
                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "docstring": ast.get_docstring(node) or "",
            }
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol["signature"] = f"{node.name}({ast.unparse(node.args)})"
            public_symbols.append(symbol)

        test_path = ROOT / "tests" / f"test_{path.stem}.py"
        tests = []
        if test_path.exists():
            test_tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
            tests = [
                node.name
                for node in test_tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            ]
        module_id = f"code:{path.stem}"
        modules[module_id] = {
            "node_type": "provisional_code_module",
            "authority": (
                "current-branch reusable implementation and test evidence; not an accepted scientific claim unless separately linked to accepted authority"
            ),
            "module": path.stem,
            "path": str(path.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "module_contract": ast.get_docstring(tree) or "",
            "public_symbols": public_symbols,
            "matching_test_path": str(test_path.relative_to(ROOT)) if test_path.exists() else None,
            "test_contract_names": tests,
        }
    return modules


def claim_state(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_type": "accepted_claim",
        "authority": "accepted in the pinned current release",
        "id": claim["id"],
        "statement": claim["statement"],
        "assumptions": claim.get("assumptions", []),
        "dependencies": claim.get("dependencies", []),
        "compatibility": claim.get("compatibility"),
        "verification": claim.get("verification"),
        "epistemic": claim.get("epistemic"),
        "comparators": claim.get("comparators", []),
        "provenance": claim.get("provenance"),
        "evidence": claim.get("evidence", []),
    }


def source_questions() -> dict[str, Any]:
    questions: dict[str, Any] = {}
    for name, feature in FEATURES.items():
        questions[name] = {
            "type": "noul",
            "instructions": (
                f"Using the exact scope, assumptions, and exclusions in this one node, decide whether: {feature['source']} "
                "Count only a construction or boundary the claim actually establishes and could teach elsewhere. "
                "A mention, comparator, explicit exclusion, or merely assumed ingredient is false."
            ),
            "criteria": {
                "true": "The node establishes a substantive reusable instance of this capability or research move at its stated scope.",
                "false": "It only mentions, assumes, compares, excludes, or lacks the capability or move."
            },
        }
    return questions


def need_questions() -> dict[str, Any]:
    questions = {
        name: {
            "type": "noul",
            "instructions": (
                f"Given the positive intent, current inputs, exact obstruction, and exclusions in this one obligation, decide whether: {feature['need']} "
                "True means this is a load-bearing missing structure or a research operation that directly targets the named obstruction. "
                "Something that would be broadly helpful, interesting, or eventually necessary is false at this stage."
            ),
            "criteria": {
                "true": "Its absence is central to the current obstruction, or applying this move directly changes the next construction or exposing test.",
                "false": "It is general background, a later downstream need, merely helpful, redundant, or contrary to the present scope."
            },
        }
        for name, feature in FEATURES.items()
    }
    questions["dominant_capability"] = {
        "type": "choice",
        "instructions": (
            "Which single capability is the most load-bearing missing semantic structure in this obligation right now? "
            "Use the exact obstruction and exclusions rather than the eventual parent wish list."
        ),
        "criteria": {name: FEATURES[name]["need"] for name in CONTENT_FEATURES},
    }
    questions["dominant_research_move"] = {
        "type": "choice",
        "instructions": (
            "Which single research operation is most likely to change how the next approach is constructed or tested, given the named obstruction?"
        ),
        "criteria": {name: FEATURES[name]["need"] for name in MOVE_FEATURES},
    }
    return questions


def vector(response: dict[str, Any]) -> dict[str, float]:
    return {name: float(response["answers"][name]["noul"]) for name in FEATURES}


def need_profile(response: dict[str, Any]) -> dict[str, Any]:
    answers = response["answers"]
    return {
        "load_bearing": {name: float(answers[name]["noul"]) for name in FEATURES},
        "capability_distribution": {
            name: float(value)
            for name, value in answers["dominant_capability"]["probabilities"].items()
        },
        "move_distribution": {
            name: float(value)
            for name, value in answers["dominant_research_move"]["probabilities"].items()
        },
    }


def record_key(kind: str, item_id: str) -> str:
    return f"{kind}:{item_id}"


def load_log() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not RESPONSE_LOG.exists():
        return records
    for line in RESPONSE_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[record["key"]] = record
    return records


def append_record(record: dict[str, Any]) -> None:
    with WRITE_LOCK:
        with RESPONSE_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inference_fingerprint(state: dict[str, Any], questions: dict[str, Any]) -> str:
    return fingerprint({"state": state, "questions": questions, "model": MODEL})


def cache_matches(
    record: dict[str, Any] | None,
    state: dict[str, Any],
    questions: dict[str, Any],
    *,
    accept_legacy: bool = False,
) -> bool:
    if record is None:
        return False
    if record.get("inference_sha256") == inference_fingerprint(state, questions):
        return True
    return accept_legacy and record.get("state_sha256") == fingerprint(state)


def infer_source(kind: str, item_id: str, state: dict[str, Any]) -> dict[str, Any]:
    questions = source_questions()
    response = ask(state, questions)
    record = {
        "key": record_key(kind, item_id),
        "kind": kind,
        "id": item_id,
        "inference_sha256": inference_fingerprint(state, questions),
        "response": response,
    }
    append_record(record)
    return record


def infer_need(item_id: str, state: dict[str, Any]) -> dict[str, Any]:
    questions = need_questions()
    response = ask(state, questions)
    record = {
        "key": record_key("obligation", item_id),
        "kind": "obligation",
        "id": item_id,
        "inference_sha256": inference_fingerprint(state, questions),
        "response": response,
    }
    append_record(record)
    return record


def pair_questions() -> dict[str, Any]:
    return {
        "relation": {
            "type": "choice",
            "instructions": (
                "Classify the strongest honest relationship from `source` to `obligation`. "
                "Treat the source as inspiration and reusable technique as well as possible direct evidence; preserve its stated scope."
            ),
            "criteria": RELATION_OPTIONS,
        },
        "research_move": {
            "type": "choice",
            "instructions": (
                "Which research operation best describes how a reasoning agent should use `source` to create or test a materially new approach to `obligation`?"
            ),
            "criteria": MOVE_OPTIONS,
        },
        "missing_bridge": {
            "type": "choice",
            "instructions": (
                "What is the principal missing bridge that must be constructed before `source` can legitimately advance `obligation`?"
            ),
            "criteria": BRIDGE_OPTIONS,
        },
        "approach_information": {
            "type": "score",
            "instructions": (
                "How much does `source` inform the design of a materially different approach to `obligation`, even if it is not a direct prerequisite?"
            ),
            "criteria": [
                "0 — superficial similarity; it supplies no useful change in approach",
                "1 — a remote analogy with no concrete transferable operation",
                "2 — a plausible operation or diagnostic, but the bridge is largely unspecified",
                "3 — a concrete transferable construction or failure-derived criterion with a named bridge",
                "4 — a compelling source construction that sharply changes the next derivation or experiment",
            ],
        },
        "novel_direction": {
            "type": "noul",
            "instructions": (
                "Would using `source` as indicated by the selected relation plausibly move the research outside local variants of the current route?"
            ),
            "criteria": {
                "true": "It suggests a materially different representation, mechanism, construction, or oracle.",
                "false": "It mostly repeats, ranks, or cosmetically varies the current route."
            },
        },
        "scope_conflict": {
            "type": "noul",
            "instructions": (
                "Would direct use of `source` in `obligation` silently import a premise, substrate, postulate, or scope that the obligation excludes?"
            ),
        },
        "structural_correspondence": {
            "type": "noul",
            "instructions": (
                "Does the exact mathematical or physical structure in `source` correspond to a named structure in `obligation`, or admit a concrete variable/operator map stated in these cards? "
                "A generic lesson such as 'try another representation', 'add an action', or 'prove applicability' is false without a source-specific correspondence."
            ),
            "criteria": {
                "true": "The cards identify a shared operator, symmetry, topology, variational form, asymptotic regime, object class, or explicit map that grounds transfer.",
                "false": "The resemblance is thematic or methodological only, with no concrete structural correspondence in the supplied state."
            },
        },
        "source_specific_instruction": {
            "type": "noul",
            "instructions": (
                "Do the two cards together imply a concrete source-specific instruction for what the research agent should derive, construct, transform, or test next?"
            ),
            "criteria": {
                "true": "The instruction can name the source construction and the target structure it should act on.",
                "false": "Only generic advice or a restatement of the obligation is available."
            },
        },
    }


def infer_pair(
    obligation_id: str,
    source_id: str,
    obligation: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    pair_id = f"{obligation_id}|{source_id}"
    state = {"obligation": obligation, "source": source}
    questions = pair_questions()
    response = ask(state, questions)
    record = {
        "key": record_key("pair", pair_id),
        "kind": "pair",
        "id": pair_id,
        "inference_sha256": inference_fingerprint(state, questions),
        "response": response,
    }
    append_record(record)
    return record


def rarity_weights(source_vectors: dict[str, dict[str, float]]) -> dict[str, float]:
    count = len(source_vectors)
    return {
        name: math.log((count + 1) / (1 + sum(vector_[name] >= 0.5 for vector_ in source_vectors.values()))) + 1.0
        for name in FEATURES
    }


def distribution_match(
    source: dict[str, float],
    profile: dict[str, Any],
    rarity: dict[str, float],
) -> float:
    def channel(distribution: dict[str, float]) -> float:
        denominator = sum(distribution[name] * rarity[name] for name in distribution)
        if denominator == 0:
            return 0.0
        return sum(
            distribution[name] * rarity[name] * source[name]
            for name in distribution
        ) / denominator

    return 0.68 * channel(profile["capability_distribution"]) + 0.32 * channel(profile["move_distribution"])


TOKEN = re.compile(r"[a-z][a-z0-9_'-]{2,}")


def lexical_scores(
    sources: dict[str, dict[str, Any]],
    obligations: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    documents = {
        source_id: set(TOKEN.findall(json.dumps(source, sort_keys=True).lower()))
        for source_id, source in sources.items()
    }
    document_count = len(documents)
    document_frequency = {
        token: sum(token in terms for terms in documents.values())
        for token in set().union(*documents.values())
    }
    idf = {
        token: math.log((document_count + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }
    result: dict[str, dict[str, float]] = {}
    for obligation_id, obligation in obligations.items():
        query = set(TOKEN.findall(json.dumps(obligation, sort_keys=True).lower()))
        query_weight = sum(idf.get(token, 1.0) ** 2 for token in query) or 1.0
        result[obligation_id] = {
            source_id: sum(idf.get(token, 1.0) ** 2 for token in query & terms) / query_weight
            for source_id, terms in documents.items()
        }
    return result


def source_card(claim: dict[str, Any]) -> dict[str, Any]:
    return claim_state(claim)


def extract_choice(answer: dict[str, Any]) -> tuple[str, float]:
    probabilities = answer["probabilities"]
    name, probability = max(probabilities.items(), key=lambda item: item[1])
    return name, float(probability)


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# TypeSafe semantic research compass",
        "",
        f"- TypeSafe model: `{result['model']}`",
        f"- pinned accepted claims indexed: {result['counts']['accepted_claims']}",
        f"- provisional Euler code modules indexed: {result['counts']['code_modules']}",
        f"- frontier failure/construction nodes indexed: {result['counts']['frontier_nodes']}",
        f"- live and reusable obligations mapped: {result['counts']['obligations']}",
        f"- pair edges semantically adjudicated: {result['counts']['pair_edges']}",
        f"- total input tokens: {result['usage']['input_tokens']}",
        f"- total output tokens: {result['usage']['output_tokens']}",
        "",
        "Every edge below is a provisional research lead. Accepted claim text and cited source files remain authoritative; TypeSafe does not prove the transfer.",
        "",
    ]
    for obligation_id, item in result["obligations"].items():
        lines += [f"## {item['title']}", ""]
        for edge in item["edges"]:
            lines += [
                f"### {edge['source_id']} — {edge['source_title']}",
                "",
                f"- relation: `{edge['relation']}` (p={edge['relation_probability']:.2f})",
                f"- research move: `{edge['research_move']}` (p={edge['research_move_probability']:.2f})",
                f"- missing bridge: `{edge['missing_bridge']}` (p={edge['missing_bridge_probability']:.2f})",
                f"- approach information: {edge['approach_information']:.2f}/4",
                f"- novel direction probability: {edge['novel_direction']:.2f}",
                f"- scope-conflict probability: {edge['scope_conflict']:.2f}",
                f"- structural correspondence probability: {edge['structural_correspondence']:.2f}",
                f"- source-specific instruction probability: {edge['source_specific_instruction']:.2f}",
                f"- dominant-capability support: {edge['capability_support']:.2f}",
                f"- dominant-move support: {edge['move_support']:.2f}",
                f"- composite research value: {edge['research_value']:.3f}",
                f"- feature-match prefilter: {edge['feature_match']:.3f}",
                f"- retrieval channels: `{', '.join(edge['retrieval_channels'])}`",
                f"- provenance: `{edge['provenance']}`",
                "",
            ]
    lines += [
        "## Intended future use",
        "",
        "1. Deterministic code selects the current accepted release and exact active obligation.",
        "2. Cached TypeSafe vectors retrieve source claims, failures, and constructions with relevant capabilities and research moves.",
        "3. Pair judgments distinguish a direct ingredient from analogy, failure inversion, oracle, conflict, or noise, and name the missing bridge.",
        "4. A reasoning agent reads the exact top sources and derives at least two concrete candidates when the mechanism is open.",
        "5. Symbolic, analytic, numerical, or formal oracles decide those candidates. Only reviewed evidence enters claim authority.",
        "6. Outcomes feed back as new typed failure/construction nodes; later campaigns inherit the information instead of repeating the route.",
        "",
    ]
    return "\n".join(line.rstrip() for line in lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--top", type=int, default=20, help="maximum diverse pair edges to adjudicate per obligation")
    parser.add_argument("--obligation", action="append", choices=sorted(OBLIGATIONS))
    args = parser.parse_args()

    claims = load_accepted_claims()
    claims_by_id = {claim["id"]: claim for claim in claims}
    code_modules = load_code_modules()
    selected_obligations = {
        key: value for key, value in OBLIGATIONS.items()
        if not args.obligation or key in args.obligation
    }
    log = load_log()

    jobs: list[tuple[str, str, dict[str, Any]]] = []
    for claim in claims:
        key = record_key("claim", claim["id"])
        state = claim_state(claim)
        if not cache_matches(log.get(key), state, source_questions()):
            jobs.append(("claim", claim["id"], state))
    for node_id, node in FRONTIER_NODES.items():
        key = record_key("frontier", node_id)
        if not cache_matches(log.get(key), node, source_questions()):
            jobs.append(("frontier", node_id, node))
    for module_id, module in code_modules.items():
        key = record_key("code", module_id)
        if not cache_matches(log.get(key), module, source_questions()):
            jobs.append(("code", module_id, module))
    for obligation_id, obligation in selected_obligations.items():
        key = record_key("obligation", obligation_id)
        if not cache_matches(log.get(key), obligation, need_questions()):
            jobs.append(("obligation", obligation_id, obligation))

    print(f"semantic jobs pending: {len(jobs)}", flush=True)
    if jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for kind, item_id, state in jobs:
                if kind == "obligation":
                    future = executor.submit(infer_need, item_id, state)
                else:
                    future = executor.submit(infer_source, kind, item_id, state)
                futures[future] = (kind, item_id)
            complete = 0
            for future in as_completed(futures):
                kind, item_id = futures[future]
                future.result()
                complete += 1
                if complete % 20 == 0 or complete == len(futures):
                    print(f"semantic jobs complete: {complete}/{len(futures)}", flush=True)

    log = load_log()
    source_vectors: dict[str, dict[str, float]] = {}
    source_states: dict[str, dict[str, Any]] = {}
    source_titles: dict[str, str] = {}
    source_provenance: dict[str, str] = {}
    for claim_id, claim in claims_by_id.items():
        source_vectors[claim_id] = vector(log[record_key("claim", claim_id)]["response"])
        source_states[claim_id] = source_card(claim)
        source_titles[claim_id] = claim["statement"].splitlines()[0][:180]
        source_provenance[claim_id] = str(claim.get("provenance", ""))
    for node_id, node in FRONTIER_NODES.items():
        source_vectors[node_id] = vector(log[record_key("frontier", node_id)]["response"])
        source_states[node_id] = node
        source_titles[node_id] = node["title"]
        source_provenance[node_id] = node["source"]
    for module_id, module in code_modules.items():
        source_vectors[module_id] = vector(log[record_key("code", module_id)]["response"])
        source_states[module_id] = module
        source_titles[module_id] = (module["module_contract"].splitlines() or [module["module"]])[0]
        source_provenance[module_id] = module["path"]

    need_profiles = {
        obligation_id: need_profile(log[record_key("obligation", obligation_id)]["response"])
        for obligation_id in selected_obligations
    }
    rarity = rarity_weights(source_vectors)
    lexical = lexical_scores(source_states, selected_obligations)
    shortlist: dict[str, list[tuple[str, float]]] = {}
    retrieval_channels: dict[str, dict[str, list[str]]] = {}
    for obligation_id, profile in need_profiles.items():
        semantic_scores = {
            source_id: distribution_match(features, profile, rarity)
            for source_id, features in source_vectors.items()
        }
        semantic_ranked = sorted(
            semantic_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        lexical_ranked = sorted(
            lexical[obligation_id].items(),
            key=lambda item: (-item[1], item[0]),
        )
        top_capabilities = sorted(
            profile["capability_distribution"],
            key=lambda name: (-profile["capability_distribution"][name], name),
        )[:3]
        top_moves = sorted(
            profile["move_distribution"],
            key=lambda name: (-profile["move_distribution"][name], name),
        )[:3]
        channel_ids: dict[str, list[str]] = {
            "semantic": [source_id for source_id, _ in semantic_ranked[:24]],
            "lexical": [source_id for source_id, _ in lexical_ranked[:24]],
            "frontier": [
                source_id for source_id, _ in semantic_ranked
                if not source_id.startswith("C-")
            ][:4],
            "graph_seed": CODE_SEEDS.get(obligation_id, []),
        }
        for feature in [*top_capabilities, *top_moves]:
            channel_ids[f"feature:{feature}"] = [
                source_id
                for source_id, _ in sorted(
                    ((source_id, features[feature] * rarity[feature]) for source_id, features in source_vectors.items()),
                    key=lambda item: (-item[1], item[0]),
                )[:24]
            ]

        # Reciprocal-rank fusion prevents a broad semantic channel from hiding
        # an exact lexical or rare-feature hit.  Obligation-specific feature
        # channels receive their Choice probability as weight.
        channel_weights = {
            "semantic": 1.0,
            "lexical": 1.0,
            "frontier": 0.45,
            "graph_seed": 1.2,
        }
        for feature in [*top_capabilities, *top_moves]:
            distribution = (
                profile["capability_distribution"]
                if feature in CONTENT_FEATURES
                else profile["move_distribution"]
            )
            channel_weights[f"feature:{feature}"] = 1.4 * distribution[feature]
        fused_scores = {source_id: 0.0 for source_id in source_vectors}
        for channel, ids in channel_ids.items():
            weight = channel_weights[channel]
            for rank, source_id in enumerate(ids, start=1):
                fused_scores[source_id] += weight / (12.0 + rank)
        candidate_order = sorted(
            fused_scores,
            key=lambda source_id: (-fused_scores[source_id], source_id),
        )

        # Preserve breadth across registry sectors while retaining explicit
        # failure nodes and candidates found by different retrieval channels.
        counts: dict[str, int] = {}
        diverse: list[tuple[str, float]] = []
        # Failure-derived continuation is a first-class discovery channel.  Its
        # five small cards always reach pair adjudication; TypeSafe may still
        # classify them as irrelevant for a particular obligation.
        for source_id in FRONTIER_NODES:
            counts[source_id] = 1
            diverse.append((source_id, semantic_scores[source_id]))
        # Active code contains candidate constructions newer than the pinned
        # release.  Keep the strongest five visible without granting them
        # accepted-claim status.
        code_candidates = CODE_SEEDS.get(obligation_id, [])[:5]
        if not code_candidates:
            code_candidates = [
                source_id for source_id in candidate_order if source_id.startswith("code:")
            ][:5]
        for source_id in code_candidates:
            counts["PROVISIONAL_CODE"] = counts.get("PROVISIONAL_CODE", 0) + 1
            diverse.append((source_id, semantic_scores[source_id]))
        for source_id in candidate_order:
            if source_id in FRONTIER_NODES or source_id in code_candidates:
                continue
            sector = (
                source_id.split("-")[1]
                if source_id.startswith("C-")
                else "PROVISIONAL_CODE"
                if source_id.startswith("code:")
                else source_id
            )
            if counts.get(sector, 0) >= 2:
                continue
            counts[sector] = counts.get(sector, 0) + 1
            diverse.append((source_id, semantic_scores[source_id]))
            if len(diverse) >= args.top:
                break
        shortlist[obligation_id] = diverse
        retrieval_channels[obligation_id] = channel_ids

    pair_jobs = []
    log = load_log()
    for obligation_id, sources in shortlist.items():
        for source_id, _ in sources:
            pair_id = f"{obligation_id}|{source_id}"
            state = {
                "obligation": selected_obligations[obligation_id],
                "source": source_states[source_id],
            }
            key = record_key("pair", pair_id)
            if not cache_matches(log.get(key), state, pair_questions()):
                pair_jobs.append((obligation_id, source_id))

    print(f"pair jobs pending: {len(pair_jobs)}", flush=True)
    if pair_jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    infer_pair,
                    obligation_id,
                    source_id,
                    selected_obligations[obligation_id],
                    source_states[source_id],
                ): (obligation_id, source_id)
                for obligation_id, source_id in pair_jobs
            }
            complete = 0
            for future in as_completed(futures):
                future.result()
                complete += 1
                if complete % 10 == 0 or complete == len(futures):
                    print(f"pair jobs complete: {complete}/{len(futures)}", flush=True)

    log = load_log()
    obligations_result: dict[str, Any] = {}
    for obligation_id, sources in shortlist.items():
        edges = []
        for source_id, feature_match in sources:
            response = log[record_key("pair", f"{obligation_id}|{source_id}")]["response"]
            answers = response["answers"]
            relation, relation_probability = extract_choice(answers["relation"])
            research_move, research_move_probability = extract_choice(answers["research_move"])
            missing_bridge, missing_bridge_probability = extract_choice(answers["missing_bridge"])
            edges.append({
                "source_id": source_id,
                "source_title": source_titles[source_id],
                "provenance": source_provenance[source_id],
                "feature_match": feature_match,
                "retrieval_fusion": fused_scores[source_id],
                "retrieval_channels": [
                    channel for channel, ids in retrieval_channels[obligation_id].items()
                    if source_id in ids
                ],
                "relation": relation,
                "relation_probability": relation_probability,
                "research_move": research_move,
                "research_move_probability": research_move_probability,
                "missing_bridge": missing_bridge,
                "missing_bridge_probability": missing_bridge_probability,
                "approach_information": float(answers["approach_information"]["score"]),
                "approach_information_confidence": float(answers["approach_information"]["confidence"]),
                "novel_direction": float(answers["novel_direction"]["noul"]),
                "scope_conflict": float(answers["scope_conflict"]["noul"]),
                "structural_correspondence": float(answers["structural_correspondence"]["noul"]),
                "source_specific_instruction": float(answers["source_specific_instruction"]["noul"]),
                "capability_support": sum(
                    need_profiles[obligation_id]["capability_distribution"][name]
                    * source_vectors[source_id][name]
                    for name in CONTENT_FEATURES
                ),
                "move_support": sum(
                    need_profiles[obligation_id]["move_distribution"][name]
                    * source_vectors[source_id][name]
                    for name in MOVE_FEATURES
                ),
            })
        relation_weight = {
            "direct_ingredient": 1.0,
            "transferable_construction": 0.95,
            "structural_analogy": 0.78,
            "failure_to_invert": 0.93,
            "falsifier_or_oracle": 0.86,
            "conflicting_premise": 0.45,
            "irrelevant": 0.08,
        }
        for edge in edges:
            edge["research_value"] = (
                (edge["approach_information"] / 4.0)
                * (0.55 + 0.45 * edge["novel_direction"])
                * (1.0 - 0.45 * edge["scope_conflict"])
                * (0.25 + 0.75 * edge["structural_correspondence"])
                * (0.40 + 0.60 * edge["source_specific_instruction"])
                * (0.35 + 0.65 * edge["capability_support"])
                * (0.55 + 0.45 * edge["move_support"])
                * relation_weight[edge["relation"]]
            )
        edges.sort(
            key=lambda edge: (
                -edge["research_value"],
                -edge["approach_information"],
                -edge["novel_direction"],
                edge["scope_conflict"],
                edge["source_id"],
            )
        )
        obligations_result[obligation_id] = {
            "title": selected_obligations[obligation_id]["title"],
            "need_profile": need_profiles[obligation_id],
            "edges": edges,
        }

    used_keys = {
        *(record_key("claim", claim_id) for claim_id in claims_by_id),
        *(record_key("frontier", node_id) for node_id in FRONTIER_NODES),
        *(record_key("code", module_id) for module_id in code_modules),
        *(record_key("obligation", obligation_id) for obligation_id in selected_obligations),
        *(
            record_key("pair", f"{obligation_id}|{source_id}")
            for obligation_id, sources in shortlist.items()
            for source_id, _ in sources
        ),
    }
    input_tokens = sum(int(log[key]["response"].get("usage", {}).get("input_tokens", 0)) for key in used_keys)
    output_tokens = sum(int(log[key]["response"].get("usage", {}).get("output_tokens", 0)) for key in used_keys)
    result = {
        "schema_version": 1,
        "generated_from": {
            "release": yaml.safe_load(RELEASE.read_text(encoding="utf-8"))["release"],
            "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
            "feature_schema_sha256": fingerprint(FEATURES),
        },
        "model": next(iter(log.values()))["response"].get("model", MODEL),
        "authority_boundary": (
            "TypeSafe vectors and edges are provisional semantic judgments for retrieval and inspiration. "
            "They do not prove a theorem, extend a claim's scope, or create a registry dependency."
        ),
        "counts": {
            "accepted_claims": len(claims),
            "code_modules": len(code_modules),
            "frontier_nodes": len(FRONTIER_NODES),
            "obligations": len(selected_obligations),
            "pair_edges": sum(len(sources) for sources in shortlist.values()),
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "feature_schema": FEATURES,
        "source_vectors": source_vectors,
        "obligations": obligations_result,
    }
    RESULT_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_FILE.write_text(build_report(result), encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "usage": result["usage"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

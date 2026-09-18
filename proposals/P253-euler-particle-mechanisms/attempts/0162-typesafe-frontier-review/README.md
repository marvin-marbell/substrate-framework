# 0162 — TypeSafe semantic research compass

This exploratory attempt tests how TypeSafe can make the framework's accepted
claims, current provisional code, failed mechanisms, and candidate
constructions useful to every future campaign. Its final design is a semantic
research compass: a cached typed index plus source-to-obligation transfer audit
that informs how a reasoning agent constructs the next approach.

The complete method, experimental corrections, calibration controls, and
source-informed P253 approach changes are in
`research-compass-review.md`. The current machine-readable graph is
`research-compass.json`, and `research-compass-report.md` renders all selected
pair judgments.

## Authority boundary

TypeSafe outputs are provisional semantic judgments. They can retrieve a
source, identify a research operation, expose a false friend, or name a missing
bridge. They cannot prove a theorem, extend an accepted claim, establish
applicability, select a physical postulate, or promote a registry dependency.
Every proposed transfer still requires exact source inspection, a concrete
construction, and a claim-appropriate oracle.

The index is pinned to accepted release `v0.183.0`. Deterministic code owns
release membership, hashes, explicit dependencies, calculation, graph
composition, and cache freshness. API credentials remain in
`~/.config/typesafe/secrets.env` and are not copied into the attempt.

## Final execution

`run_research_compass.py` performs the reusable workflow:

1. isolate and type every accepted claim, current Euler module, and frontier
   failure node while preserving their different authority;
2. map each exact obligation to a load-bearing capability and research-move
   distribution;
3. fuse semantic, lexical, rare-feature, and failure-derived retrieval;
4. audit each source/obligation pair in its own request; and
5. compose provisional research value while retaining all raw judgments.

The completed run used TypeSafe model `jev-1.13.0` and produced:

- 271 pinned accepted-claim vectors;
- 47 provisional current-branch Euler-module vectors;
- five frontier failure-mechanism vectors;
- six obligation profiles;
- 120 isolated transfer edges;
- 1,744,630 input and 194,860 output tokens in the final snapshot; and
- 3,643,075 input and 414,338 output tokens across all iterative experiments.

Reproduction:

```bash
python proposals/P253-euler-particle-mechanisms/attempts/0162-typesafe-frontier-review/run_research_compass.py --workers 3 --top 20
python proposals/P253-euler-particle-mechanisms/attempts/0162-typesafe-frontier-review/verify_research_compass.py
```

Responses are cached by state/question/model fingerprint in the append-only
`research-compass-responses.jsonl`, so an unchanged rerun makes no API calls.
The verifier checks authority hashes, semantic calibration controls, required
source retrieval, the failure reservoir, edge uniqueness, and credential
absence.

## Time-sliced policy extension

`run_trajectory_policy.py` treats completed P253 attempts as logged research
transitions and evaluates counterfactual actions only from sources completed
before each decision. The full design, corrections, findings, and current
approach implications are in `trajectory-policy-review.md`.

The executed policy snapshot contains:

- 168 committed historical episodes and 168 eligibility-hashed time slices;
- 28 distributed decision audits and 159 historical counterfactual edges;
- 20 cleaner missed-opportunity leads plus 30 hindsight-sensitive leads;
- nine repeated-failure families and learned priors for 12 operations;
- 65 current action candidates across five obligations; and
- five stuck interventions conditioned on exact failure mechanisms.

Its machine-readable graph is `trajectory-policy.json`, and
`trajectory-policy-report.md` renders the audits, failure map, priors, current
rankings, and interventions. The pass used 4,952,243 input and 334,728 output
tokens with model `jev-1.13.0`.

Reproduction:

```bash
python proposals/P253-euler-particle-mechanisms/attempts/0162-typesafe-frontier-review/run_trajectory_policy.py --workers 3 --audit-limit 32 --counterfactuals 6 --current-sources 10
python proposals/P253-euler-particle-mechanisms/attempts/0162-typesafe-frontier-review/verify_trajectory_policy.py
python proposals/P253-euler-particle-mechanisms/attempts/0162-typesafe-frontier-review/verify_published_bundle.py
```

The append-only cache is `trajectory-policy-responses.jsonl`. An unchanged
rerun makes no TypeSafe requests. The verifier replays chronology and state
hashes, rejects future leakage, tests calibrated controls, keeps accepted,
provisional, historical, and failure-derived channels visible, enforces the
foundation-authorization boundary, and checks credential exclusion.
`verify_published_bundle.py` additionally pins the three generated artifact
hashes and checks the graph and response ledger without making API calls.

### Current boundary and future policy evolution

The committed version is a research-state builder and fixed policy evaluator,
not a recursively self-improving policy. It reconstructs time-sliced state,
types prior outcomes, estimates retrospective operation priors, and ranks
candidate actions. The semantic schemas, action vocabulary, aggregation
formula, channel anchors, and authority penalties remain fixed in
`run_trajectory_policy.py`. The learned priors summarize this historical
corpus; they do not update or validate the policy that produced them.

A future self-improvement loop needs explicit policy versions and prospective
evaluation. Given prior policy `pi_(t-1)`, current policy `pi_t`, and their
logged state/action/outcome trajectories, the system would compare the two,
construct a candidate `pi_(t+1)`, and freeze it before deployment. New outcomes
would then evaluate `pi_(t+1)` against its predecessors on common obligations
and preregistered measures. Only that evaluation would license generating a
later policy. The policy mutation language, comparison design, selection
objective, and promotion rule remain to be determined and implemented.

This attempt publishes the scripts and outputs for review. It does not install
an agent skill or repository workflow.

## Development history

The earlier portfolio files are retained as method evidence:

- `run_typesafe_review.py`, `round1-response.json`, `round2-response.json`, and
  `ranking.json` record the initial route-ranking design;
- `run_method_experiment.py` and `method-experiment.json` record authority,
  order-sensitivity, and false-green tests; and
- `run_isolated_experiment.py` records the proposed isolation correction that
  was superseded by the full semantic compass.

The portfolio result is not the final method. It showed that ranking routes is
too narrow. The later runs use TypeSafe as structured research memory and as a
source of grounded analogy, failure inversion, representation choice, and
missing-bridge information.

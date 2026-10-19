# Catalyst campaign 4: retrospective logic review

Canonical issue: https://github.com/vantasnerdan/substrate-framework/issues/223

## Scope and source access

The user requested a review of the original objective and campaign logic, then explicitly prohibited simulation reruns. This review reads existing code, plans, JSON, logs and cached trajectories only. `sanity.py` imports no GPAW and evaluates no calculator properties. No SCF, relaxation, parameter sweep or new physical simulation was run. Existing campaign outputs were already exposed; this is retrospective auditing, not a fresh preregistered experiment.

Sources in hand are the catalyst-dft run `runs/2026-10-15_dd-screening/` (PLAN.md, REPORT.md, scripts, result.json, CSV, trajectories, logs), campaign-3 `runs/campaign3/audit.py` and bulk/vacancy reference JSONs, the original-request memory entry, and the primary session archive identified below. Input hashes and extracted cached geometry are in `receipt.json`.

## Disposition

Retain the saved constrained electronic-energy samples as exploratory artifacts. Withdraw the prior claims that campaign 4 decisively rejected shell screening, identified interstitial assistance, explained the anchor failure, or completed a coherent-domain discovery objective. The campaign's own required interpretation prerequisite was not satisfied, and its implementation contains additional logical errors. This review does not invalidate every earlier campaign result: each earlier claim retains only the scope its original observable and references support.

## Original objective and scope bridge

The primary archive is `/home/administrator/.omp/agent/sessions/-/2026-09-14T05-05-47-250Z_01a09e4e-9cf2-7712-8abd-08057d8f7bc6.jsonl`. At line 11020 (2026-10-17), Dan asks for a literature review and discussion of coherence simulations, noting Luca's criticism: “simulation was too naive, grid too small, not enough objects.” The immediately preceding proposal at line 11084 recommends Stage 1.2, the DFT pair-approach calculation, first. Dan approves with durability/checkpoint/visibility and “days max” requirements (11085), then says “I need you to focus on the DFT models and simulations that you sepecialize in” (11089). Thus an initial DFT input rung was authorized; the problem is its claimed discriminatory power, not that it was unauthorized. The broader objective was lattice configurations useful to coherent-domain LENR, not characterization alone (project-objective email at archive line 10443; session-state lines 24–25). The scoping memory's October 14 date conflates that earlier objective with the October 17 simulation discussion.

Campaign 4's PLAN.md, lines 85–99, chooses one possible atomistic input: forced H–H approach in a vacancy shell, compared with molecule and lattice references. That is not itself a coherence-domain simulation. The executed model is static spin-polarized electronic DFT with fixed Ni positions, a periodic small cell and constrained H distances (`screen_path.py:40–43,59–106,148–178`). It implements no coherence/order-parameter observable, domain-size scaling, collective quantum nuclear dynamics, coupled electromagnetic dynamics, or nuclear reaction calculation. A conditional local energy landscape can inform a later model; it cannot by itself establish or refute that model's collective/nuclear claims.

## What the saved numbers actually say

The independent arithmetic in `sanity.py` reproduces these statistics from existing `result.json`; these are not licensed screening verdicts:

| Series | Sample interval, Å | Endpoint minus start, eV | Endpoint minus sampled minimum, eV |
| --- | --- | ---: | ---: |
| Vacancy shell | 2.44 → 0.74 | +2.054514 | +2.076140 |
| Molecular reference | 2.40 → 0.74 | −3.546293 | +0.434279 |
| Purported interstitial reference | 1.60 → 0.80 | −7.810051 | +0.294286 |

The sampled minima occur at 2.20, 0.90 and 0.90 Å respectively. A coarse-grid sampled minimum is not a demonstrated equilibrium bond length. The cached trajectories reproduce each saved target distance, and the maximum constraint-projected force over these 23 final frames is approximately 0.0294 eV/Å. This supports final-frame bookkeeping only, not basis/cell/k-point convergence or global-minimum claims (`receipt.json`, `frames`).

## Load-bearing logic failures

### 1. The anchor compares different reactions, and its expression is not atom-balanced

Campaign 3 defines vacancy trapping relative to a bulk interstitial H reservoir (`runs/campaign3/audit.py:122–149`; `c32_bulkvac/plan.md:7`):

\[
\Delta E_6^{\mathrm{trap}}=E(VH_6)-E(VH_5)-[E(\mathrm{Ni}_{32}H)-E(\mathrm{Ni}_{32})].
\]

Its saved QE inputs reproduce −0.2806 eV. Campaign 4 instead evaluates `E(VH5) + E(H2) - E(VH6)` (`scripts/gate.py:43–47`). The terms have a net residual of one H atom; a species-dependent reference-energy offset therefore does not cancel. The SymPy atom-count oracle exposes that residual and distinguishes both the balanced bulk-trapping expression and the different, balanced half-H2 formation expression. Even simply replacing H2 by half-H2 would not reproduce campaign 3's bulk-reservoir observable.

Consequently the reported −2.1646 eV versus −0.281 eV is not evidence for a quantified cross-engine error. The prior explanation blaming a molecular bond-length/basis bias was not demonstrated.

### 2. The interpretation gate fails open

PLAN.md:103–109 says failure requires protocol investigation and D1 samples must ship uninterpreted. `gate.py:47–80` computes the flag but does not use it to suppress verdicts. It writes `.gate_done` and sends a completion notification anyway (`gate.py:82–103`). The saved result has `gate_pass=false` and categorical screening verdicts simultaneously. The prior claim that the automated gate refused to ship is false: human inspection noticed anomalies after the program had emitted verdicts.

### 3. The registered metric and the reported metric differ

PLAN.md:124–129 defines `B=max(E)-E(first)` and thresholds of −1.0 eV (material screening) and −0.2 eV (no material screening). REPORT.md:18–32 instead uses endpoint minus each curve's sampled minimum and describes a −0.1 eV criterion. The +1.642 eV and −0.140 eV differences in the prior response came from that post-hoc definition, not the registered rule.

Under the actual rule the molecular curve has `B_mol=0`, because its first sample is its highest energy. Since `max(E_shell)-E_shell(first)` is necessarily nonnegative, the shell difference cannot reach the negative screening threshold for this molecular reference. This does not prove physical screening absent; it exposes a nondiscriminating metric for these inputs. The prior phrase 'no barrier' also confused absence of an interior maximum with a nonzero energy cost to the compressed endpoint.

### 4. The delivered U(R) is not a same-coordinate subtraction

`gate.py:64–65` zips unequal grids by index. Its pairs are (2.44,2.40), (2.20,1.80), (2.00,1.40), (1.80,1.05), (1.60,0.90), (1.40,0.74) Å. None is an equal-R pair; later shell points are dropped. The CSV header advertises three columns while each data row has two (`U_of_R.csv`). These outputs must not be consumed as the promised same-R environment correction or as a nuclear tunneling potential. Even a correctly matched absolute shell-minus-molecule energy would contain a different host/composition baseline; a physical use requires an explicitly aligned energy reference and model bridge.

### 5. The interstitial builder does not select the stated sites

`screen_path.py:83–89` applies conventional-cell fractions to the doubled supercell. The two unshifted anchors coincide exactly with Ni lattice sites in the saved cell (`receipt.json:reference_geometry_and_metric`). After the symmetric push, the first saved H–Ni minimum distance is 0.962 Å and the final one is 1.362 Å. Thus the declared adjacent-octahedral-site construction is not what the code builds. Each saved interstitial optimizer log contains only step 0. Constraint-projected stationarity does not establish a stable minimum or the advertised pathway.

Separately, the advertised −0.140 eV assistance compares the interstitial endpoint at 0.80 Å with the molecule endpoint at 0.74 Å. That unequal-endpoint subtraction cannot establish assistance, even before the geometry problem is considered.

### 6. Numerical status was overstated

`screen_path.py:186–207` ignores the optimizer convergence boolean and unconditionally writes `ok=true`; the assembler does not enforce force, geometry or SCF acceptance. The final saved frames do meet the inspected projected-force threshold, so this code defect does not itself demonstrate a failed final relaxation. It demonstrates that successful completion was not guaranteed by the reported status.

The molecule log (`frames/mol_0p74.txt:106,154`) records two basis functions and partial-wave LCAO basis use, not a demonstrated converged molecular basis. No reviewed refinement supports the previously quoted ±0.03–0.05 eV energy uncertainty; a force threshold alone is not an energy-error estimate. Warm seeding does not prove seed-independent global minima. The shell constraint pair also changes from indices (32,34) in the first saved frame to (32,36) thereafter, consistent with the closest-pair reselection in `screen_path.py:157–167`; fixed-pair path identity is not preserved as declared.

### 7. Notification multiplicity has a concrete MPI cause

`gen_jobs.py:10–11,27–30,77` launches the plain-Python gate under six MPI ranks. `gate.py` has no rank guard around output or sending; the final `queue/gate.log` contains six copies of the verdict and six notification-success lines. The earlier idempotence check merely tested whether the string `.gate_done` existed in the source, not whether an early-return guard existed. Duplicate mail is therefore consistent with rank-duplicated gate execution, not evidence for multiple completed campaigns.

## Campaign chain: valid use versus unsupported promotion

Surface sites, uptake, trapping and escape observables from campaigns 1–3 can be structural/thermodynamic inputs to a later collective model. Favorable H trapping does not establish coherence; an occupancy or escape calculation does not establish a nuclear enhancement mechanism. Campaign 4 attempted another local input, but its comparison contracts are broken. The coherent-domain objective remains open. No simulation rerun is needed to establish these logical defects.

Earlier campaign records already document the same logical hazards. Campaign 2 withdrew vacancy-trap and spin-gate interpretations after final geometry and reference/constraint asymmetries were inspected (`runs/2026-09-22_campaign2-section12/RESULT.md:79–105,135–139`). The original campaign-3 occupancy script combines bulk-solution-referenced binding with a gas-referenced chemical potential (`runs/campaign3/zpe_occupancy.py:26–38,49–75`). The existing corrected synthesis explicitly restores the reservoir conversion and limits occupation to a finite selected set with no solid vibrational free energies (`analyses/2026-10-14-lenr-lattice-science/SCIENCE_SYNTHESIS.md:28–38`). These are documented corrections, not new simulation results from this review. They support a source-level lesson: correct energy arithmetic is insufficient without the intended geometry, matched constraints, reservoir and observable.

## Evidence and limits

Run the read-only audit with the existing catalyst environment:

```sh
/home/administrator/projects/catalyst-dft/.venv/bin/python \
  docs/reports/catalyst-campaign4-audit-2026-10-19/sanity.py \
  /home/administrator/projects/catalyst-dft
```

Observed exit: 0. Runtime: about 2 seconds. SymPy exact arithmetic checks atom conservation and the negative-threshold obstruction; the whole-H2 and index-zip controls expose the actual failure channels. ASE reads cached trajectories only. `receipt.json` pins reviewed inputs by SHA-256. No new physical energy, convergence error, nuclear rate, coherent-domain property or screening/no-screening result is claimed. Production campaign files remain unchanged so the audit preserves the original evidence.

The justified next decision is to correct definitions and acceptance logic before authorizing more compute, not to repeat the current campaign. New simulations are outside this review's authorization.

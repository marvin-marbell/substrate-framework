# PR #212 harvest review and correction check

Reviewer: Codex, distinct from author/implementer `giuliano-vantasner`

Canonical issue: #211 (open; predates PR #212)

Reviewed PR head: `9fecb3a362f1fb31e6b584d6c328a18ee71b8c50`

Integrated base: `origin/main` at
`ff9dfb6f2478b1156018c56f9c9d280b6261854d`

Lifecycle authority: repository owner explicitly requested review, correction,
and merge of PR #212; this record supplies the distinct-merger pass.

## Strongest supported result

The original bounded referee object is useful and survives review: the
announcement "Finally flipped Newton sign, finite nonzero frequency" is refuted
as stated.  The sign was not flipped; the report's own later correction puts the
audited two-Goldstone range at `1/d^7`; and the later OpenWave R19 pair result is
repulsive under its pre-registered gate.  The algebraic sector results remain
valid at their stated form-level scope.

The broader Addendum 3 is valuable as a complete source snapshot, a 665-ID
inventory of all 41 top-level comments plus three replies, a selected oracle
suite, and an external packaging comparison.  It is not an exhaustive
validation of all atomized claims and does not atomize two JarekDuda replies
embedded in `xrodz.md`.

## Unit dispositions

| Unit | Review result | Evidence and landing scope |
|---|---|---|
| Original C1-C13 audit and B10-B13 disclosures | correct then merge | Headline/current-status prose corrected to incorporate the closed D2 scope, `1/d^7` B12 result, and completed repulsive R19 result.  Base oracle receipt remains 50/50 checks and 18/18 breaking mutations. |
| Full discussion sources and claim inventory | correct then merge | All 41 top-level comments and five replies are vendored and MD5-valid.  The 665 IDs cover the top-level comments and three replies; two cross-thread replies are now explicit frontier.  Stale count summaries and temporary `/tmp/p252` paths were repaired. |
| B14-B18 selected verifier | correct then merge | B14's load-bearing identities now hold symbolically on all `r > 0`, and the charge direction uses an exact eigenvector rather than samples.  B18 preserves significant trailing zeroes and has a mutant that exposes the former tenfold tolerance weakening.  Current receipt: 29/29 checks, 10/10 mutations. |
| External packaging assessment | correct then merge | OpenWave remains HIGH at exact pin `55fcc168...`; the committed R20 data match the quoted values.  The mjmikulski report-016 dependency is now pinned to repository snapshot `210bbff...` at comment time (file last changed at `112bd1e...`) rather than mutable `main`.  The source-script reproduction limitation in the merger environment is explicit. |
| Framework comparison | merge | The review now states the natural boundary: report 016 is a real obstruction to free-time-axis, F/spectrum-only actions, while P249 quotients that axis by hypothesis; OpenWave R13-W/R20 failures on its 4x4 action neither refute P250's separate positive completion nor receive validation from it. |
| Claim promotion or release change | none | `claims_proposed: []`; no registry, release, canonical module, or generated-document semantics change. |

## Substantive findings and repairs

1. Addendum 3 conflated complete source preservation, partial atomization, and
   selected validation.  The repair separates those evidence roles, records
   the two omitted reply ledgers as D13, and leaves the higher goal open.
2. B14 called a generic-point evaluation an identity for every profile.  The
   repair proves `P dP_i = dP_i P = 0` and
   `[dP_i,dP_j] = 0` as coordinate identities on `r > 0`; the full
   `[d_iN,d_jN] = 0` statement follows coefficient-by-coefficient in `a,a'`.
3. B14's degree-one claim used 200 floating-point directions.  The repair
   exhibits the exact `E1` eigenvector
   `(B-E1,(A-E1)n)`, whose spatial direction is the source-oriented identity
   map on each sphere.
4. B18 converted posted decimals through `float`, so `11.290` became `11.29`
   and the allowed tolerance grew from `0.0005` to `0.005`.  Posted values are
   now strings and mutation `B18_M18b` proves the trailing zero is
   load-bearing.
5. The source manifest's first path was invalid when checked from its own
   directory, the proposal named a nonexistent snapshot, and report 016 used a
   mutable branch reference.  All three provenance defects are corrected.
6. B11 is a bounded commutator quadratic-form witness, not the requested full
   R-decomposition Hamiltonian oracle.  The report now states that boundary;
   the actual later consumer-level R19 result is kept separate.

## Correction check

- `verify_d186_extension.py`: 29/29 checks PASS; 10/10 mutations BREAK;
  exit 0; 15.12 s; receipt `attempts/0003/full_run.log`.
- `verify_newton_sign_audit.py`: unchanged implementation; merger-review rerun
  before correction returned 50/50 checks PASS and 18/18 mutations BREAK;
  exit 0; 41.2 s.  Its propositions were not invalidated by the prose and B14
  to B18 repairs.
- `md5sum -c sources/MD5SUMS`: all nine entries OK after manifest and metadata
  repair.
- Python compilation: both verifier modules compile.
- `scripts/validate_repository.py`: workflow valid, 271 accepted claims,
  14 proposals, zero pending/partial migration units.
- `scripts/validate_changed.py --base origin/main --head HEAD`: fixed-only
  selected; all fixed repository checks pass.  All 1069 memory files are
  structurally valid.  The 44 repository-wide TOC-readability warnings are
  advisory, pre-existing outside the P252 memory entries, and do not invalidate
  the receipt.
- `git diff --check`: clean.

The OpenWave source audit was also attempted from the exact `55fcc168...`
clone in the framework environment.  It stopped at import time because that
environment lacks `matplotlib`; the PR's preserved clean-clone receipt records
19 PASS lines before its expected missing-array stop.  This review therefore
uses the independently passing B16 algebra and calls the upstream execution a
preserved receipt, not a fresh reproduction.

## Authority and frontier

Artifact merge is eligible after the final fixed-only validation.  Scientific
claim promotion is none.  The original issue #211 object is complete; the
owner-directed full-thread extension is not, so the PR correctly remains
`Advances #211` and the issue remains open.

The next decisive verification is to atomize `[12.r0]` and `[12.r1]`, then
freeze whether #211 requires executing every remaining `checkable_here=true`
route or an owner-approved smaller validation surface.  Evidence overturning
this review would be either a missing source unit beyond the recorded 41+5, or
a completed independent oracle showing that a result currently labeled only
inventory/provenance has been promoted too weakly.

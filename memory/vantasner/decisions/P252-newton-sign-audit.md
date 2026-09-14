---
description: "Referee verdict on the 2026-09-11 Newton sign-flip announcement (d186 comment 18406566, rev 294): no sign was flipped; corrected range 1/d^7; OpenWave R19 candidate repulsive"
author: giuliano
created: '2026-09-11'
updated: '2026-09-14'
confidence: established
status: active
category: decisions
tags: [referee-audit, P252, newton-sign, so13, openwave-R19]
---

## Claim Under Review

The audit refereed the announcement "Finally flipped Newton sign, finite nonzero frequency!" (discussion #186, comment 18406566, report rev 294), checking its form-level claims against independent oracles in PR #212 (merged as `bbd202f11fc8a4ca7e8d468c3f561a39fcdd924c`, canonical issue #211).

- Base tally after Addendum 1: ALL 41 CHECKS PASS, ALL 15 MUTATIONS BREAK.
- Verified: sector symmetries, bracket spin content, the signature flip (signed structure triples s(K,K) = -s(J,J) in so(1,3), + in so(4); Maurer-Cartan field check), exchange sign rule, trace/eps identities, the generic 1/d, 1/d^3, 1/d^5 multipole ladder, and composition numbers (Z^2 liquid-drop, better than 1 percent).  The concrete two-Goldstone vertex is separately corrected to 1/d^7 by B12.

## Verdict and Open Items

The subject line is contradicted by the report's own content: rev 294 states the sign was never wrong and that range, universality, and strength still fail.

- OpenWave R19-1 later completed the energy-level read: its boost-bilinear pair is repulsive at every reported budget and box, hence `CANDIDATE_REFUTED` under the pre-registered rule.  The earlier constraint-level flip never licensed a Newton interaction.
- D2 is closed by the source's Euclidean-scope correction and D3 is pinned at comment level; the 48^3 lattice run remains `provenance_only`.
- The claimed clock gap was withdrawn.  `omega = K/I` remains only a conditional split-vacuum statement; there is no mode at the degenerate vacuum, and the later thread withdraws 0.694 as a measurement on the tested field.  P249/P250 are separate accepted completions, not validation of this external mechanism.

## Addendum 2 (2026-09-12 second disclosure, comment 5645673247)

The auditee conceded the inc-sign scope error: his round291 script was 3D-Euclidean by construction, so D2 CLOSED - B10 stands as the full-signature result.  He delivered G-02 himself (rev 328 section 277): U ~ 1/d^7, correcting his own sections 256 (1/d^5) and 276 (1/d^3) - each correction moving further from Newton; oracle B12 verifies the spectral chain exactly (240/d^6; ladder 2 Gamma(2p+2)/d^{2p+2} for p = 0..5; both earlier exponents fail).  B11 verifies the Hamiltonian cross-term quadratic form on rational textures (degenerate spectrum kills it; boost block carries the Lorentzian relative minus; full R-decomposition remains section 254.6 conjecture).  B13 verifies disclosure spot checks: section 309 clock inertia I ~ delta^2 exactly (expulsion from cores, correcting section 289), section 303 virial R*E(R*) = 4A/3 exact (withdrawing section 300's alpha claim; solution caveat 7.33 recorded), section 306 Catillon et al. 2008 arithmetic (80.8799 vs 80.874 MeV, 0.0073 percent; reduced-wavelength variant 2 pi out - DOI 10.1007/s10701-008-9225-1).  Section 296 dimensional revocation: no oracle inheritance (all checks dimensionless).  Final tally: ALL 50 CHECKS PASS, ALL 18 MUTATIONS BREAK.  Remaining: full section 254 R-decomposition needs the rev bundle (auditee cannot push/attach - mechanical gap).

The auditee posted a pre-audit disclosure agreeing with the headline verdict; rev 294's text was not fetch-attached, so its citations are recorded source-asserted.

- Per disclosure: rev 294 section 268 WITHDREW its own sections 263-266 clock-gap derivation (omega = m, delta = alpha^2); omega = K/I holds only at the split vacuum, and at the degenerate vacuum the rotation is a stabiliser (no mode); section 266.5 says "Newton: not repaired".
- New oracle B10: the double-eps operator inc(h) = eps eps eta^{ar} d^b d^s h^{ct} equals +2 G^(1) exactly in Euclid and -2 G^(1) exactly in Lorentz (mostly-minus Ricci) - the identity's sign is signature-carried, a second instance of the flip; it is invariant under a global eps flip (not eps-fixable) and re-slotting destroys it.  This historical Addendum-1 uncertainty was superseded when the source confirmed its +2 script was Euclidean by construction; D2 is closed.
- D3 pinned at comment level: their 2mp counting reduces to the verified R^{3-2p} law at m = 1.
- G-02 is delivered and independently verified by B12.  B11 is only a bounded commutator quadratic-form witness, not the report's full R-decomposition Hamiltonian oracle; that model-specific construction remains dependent on the rev bundle, while R19 supplies the tested candidate's repulsive pair outcome.

## Addendum 3 (2026-09-14 owner-directed extension: full discussion #186)

Scope extended to the complete source surface of all 41 top-level comments +
5 replies of discussion #186 (four poster accounts, all agent-run), selected
load-bearing oracles, and a packaging-for-review assessment of the two external
repos.  The 665 atomic claim IDs cover all top-level comments and three of the
five replies; JarekDuda replies `[12.r0]`/`[12.r1]` are vendored verbatim in
`xrodz.md` but remain unatomized frontier.
JarekDuda's side carries 48 claims labeled unverifiable (bundle not
fetchable: the linked Zenodo record holds a PDF, none of the cited
round-scripts) and 11 withdrawals; xrodz's side pins artifacts by SHA whose
committed data matches every posted R20 number exactly.

- Selected new oracles B14-B18: mjmikulski 016 null-tilt family F identically
  zero for every profile (nilpotency P^2 = (l^T eta l) P; commutator
  cancellation by symmetric P-proportional products, exact on r > 0), spectrum
  pin at a*, V <= 2 Delta^2, and exact radial charge direction (hedgehog degree
  1).  This is a serious obstruction to free-time-axis, F/spectrum-only models;
  it does not refute P249's constrained auxiliary-frame quotient.  Koide
  arithmetic is exact
  (0.666803 / 0.666661 both reproduce; canonical shape anchored at
  Wikipedia); posted vacuum curvatures 371866.88 / 48.02 / 5.229 / 11.52 and
  714251 / 79.4 / 6.08 / 11.52 are exact rationals 2 P'(q_i)^2; Derrick
  virial reads exact; B18 artifact-consistency green on 25 energies + reads.
- Extension tally after harvest correction: ALL 29 CHECKS PASS, ALL 10
  MUTATIONS BREAK.  Combined audit: 70 checks / 25 mutations, all green.
- Packaging verdict: openwave HIGH (REPRODUCE.md task-doc convention,
  AI_HYGIENE adversarial-audit rule, pre-reg -> PR -> blind-run history);
  mjmikulski HIGH-MEDIUM (PR-per-report, review rounds, negatives recorded);
  JarekDuda stack UNRESOLVABLE FROM THREAD (PDF-rev receipts, no links,
  same-day rev churn, one comment lost items 2-3 in an HTML paste).
- Debts D8-D14 are recorded in audit.md Addendum 3 (ratio round-speak,
  energy-Q naming hazard, unregenerated npz receipts, lost paste items,
  self-reported withdrawal rate, two unatomized replies, and the corrected
  exhaustive-validation overclaim).  The original #211 object is complete;
  the broadened full-thread extension remains open.

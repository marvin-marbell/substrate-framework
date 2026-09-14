# P252 external referee audit: the 2026-09-11 "Newton sign flip" announcement

Object under audit: substrate-framework discussion #186, comment 18406566
(JarekDuda, 2026-09-11T21:15:12Z), report revision 294, announced by email
subject "Finally flipped Newton sign, finite nonzero frequency!".  Companion
context: comments 18399375/18406555 (rev 249 closure), 18379326
(anticommutator proposal), 18388685 (OpenWave R19 pre-registration).
Source snapshot: `sources/comment-thread-snapshot.md` (verbatim jq extraction,
MD5 in `sources/MD5SUMS`).

Framing: per the referee template the report is the last artifact.  Every
checkable claim below carries one independent oracle (derived from explicit
generator matrices, static-source field equations, and direct differentiation -
never from the source's own derivations) and one mutation that was observed to
break.  Modes: `--verify` and `--mutate` of `verify_newton_sign_audit.py`.

## Headline verdict (claim C12)

**The announcement's subject line is contradicted by the report's own
content.**  Revision 294's title is "The gravity sector, reorganised: the sign
was never the problem", and its section 1 states the sign "is correct, in
every sector that could carry a gravity-analogue - and has been since the sign
rule was established".  No sign was flipped; the discovery is that the sign
was never wrong.  Three of the four gravity requirements still fail
(range, universality, strength - the report's own table).  "Finite nonzero
frequency" is qualified in section 7: the clock turned out to be massless and
the surviving finite frequency is omega = K/I with K topologically fixed - the
clock survives by topological protection, not by energetic preference.  The
announcement therefore does not resolve the two issues it names ("wrong Newton
sign and diverging omega"): the sign was a sector-confusion artifact, and the
frequency is the already-known topological-protection structure.

## Claim ledger

| #  | Claim (atomic)                                                                     | Premises                              | Location (comment 18406566) |
|----|------------------------------------------------------------------------------------|----------------------------------------|-----------------------------|
| C1 | so(4) generators all antisymmetric; so(1,3) boost generators symmetric             | eta_00 = -1 (source's signature)       | sec. 2                      |
| C2 | for symmetric inputs {A,B} is symmetric (spin 0+2), [A,B] antisymmetric (spin 1)   | C1                                     | sec. 2                      |
| C3 | signature flips the cross term: boost x boost enters the rotation sector with opposite sign in so(1,3) vs so(4); "flipped by eta, not by hand" | C1                                     | sec. 4                      |
| C4 | sign rule: scalar and tensor exchange attract, vector exchange repels like charges | tree-level exchange, conserved sources | sec. 1 table                |
| C5 | tr({A,B}^2) = tr([A,B]^2) + 4 tr(A^2 B^2) (anticommutator square adds no new invariant) | none                                   | rev 249 sec. 1              |
| C6 | eps^{mu nu rh si} tr(X_mu {X_nu, X_rh}) = 0 identically (no topological current)   | none                                   | rev 249 sec. 1              |
| C7 | 1/d needs a zero-derivative vertex (Einstein-Hilbert structure)                     | C8                                     | sec. 3                      |
| C8 | vertex derivative count 0/1/2 gives static potentials 1/d, 1/d^3, 1/d^5            | multipole expansion                    | sec. 3 table                |
| C9 | inc M = 2 G^(1)[M]; M.inc M is the linearized EH action; winding sector makes its core energy divergent (2mp = 2 < 3) | linearized geometry                    | sec. 3                      |
| C10| Gamma_mu = O^T d_mu O is teleparallel: zero curvature, torsion carries everything  | O in SO(1,3)                           | sec. 4                      |
| C11| E_EM/M = 4.87e-4 (He), 2.44e-3 (Fe), 4.44e-3 (U); spread 9.1x vs Eotvos bound 1e-15 | liquid-drop Coulomb energy             | sec. 6                      |
| C12| headline: a sign flip happened and a finite nonzero frequency was won               | -                                      | email subject vs report     |
| C13| "ratio -1.000 at five separations" 48^3 lattice check of V_complete = -<J, K^-1 J> | their run bundle                       | sec. 1                      |

## Oracle table

| #  | Oracle (independent route)                                                                                                   | Mode       | Result |
|----|------------------------------------------------------------------------------------------------------------------------------|------------|--------|
| O1 | explicit so(4)/so(1,3) generator matrices; transpose properties; full commutator tables                                       | exact      | PASS   |
| O2 | spin decomposition: scalar commutes with rotations; traceless-symmetric part obeys the rank-2 transformation law with coefficients read off [J,K] | exact      | PASS   |
| O3 | signed structure triples in the eps convention: s([K,K]) = -s([J,J]) in so(1,3), = +s([J,J]) in so(4); Maurer-Cartan curvature of an explicit O(x,z) field vanishes identically | exact      | PASS   |
| O4 | static-source interaction energies from the Lagrangians (scalar -, vector +, spin-2 -); Green kernels exact off origin + FT numeric (quadosc, rel ~ 1e-16); route B: linearized field equation Newton limit | exact+numeric | PASS   |
| O5 | trace identity on general symbolic 4x4 entries and integer matrices n=2,3,5                                                    | exact      | PASS   |
| O6 | eps-contraction exact vanishing; commutator contrast generically nonzero                                                       | exact      | PASS   |
| O7 | dipole-dipole tail formula and trace-vanishing; quadrupole-quadrupole homogeneous of degree -5                                 | exact      | PASS   |
| O8 | linearized Einstein tensor validated by pure-gauge annihilation and the linearized Bianchi identity; Derrick scaling R^{3-2p} by exact change of variables | exact      | PASS   |
| O9 | inc identity from the comment alone; superseded by B10 (addendum): +2 G^(1) Euclid / -2 G^(1) Lorentz exact, not eps-fixable | exact      | SIGNATURE-CARRIED |
| O10| liquid-drop Coulomb fractions (Z^2 convention, mpmath 30 digits)                                                                | numeric    | PASS   |

## Mutation table (observed breaking)

| #   | Mutation                                                                  | Observed |
|-----|---------------------------------------------------------------------------|----------|
| M1  | wrong eta_00 sign breaks boost symmetry                                    | BROKE    |
| M2  | demanding so(4)'s same-sign relation in so(1,3) fails                      | BROKE    |
| M3  | mixed-symmetry inputs break the {A,B}-symmetry claim                       | BROKE    |
| M4  | trace-identity coefficient 4 -> 2 leaves a nonzero remainder               | BROKE    |
| M5  | symmetric contraction tensor in (nu,rh) breaks the eps-vanishing           | BROKE    |
| M6  | scalar coupling sign flip turns attraction into repulsion                  | BROKE    |
| M7  | antisymmetrized projector kills the Fierz-Pauli 00-00 contraction          | BROKE    |
| M8  | Newton-route sign flip turns the potential positive                        | BROKE    |
| M9  | quadrupole tail is not homogeneous of degree -3                            | BROKE    |
| M10 | Derrick exponent 3-p instead of 3-2p fails for p=2                          | BROKE    |
| M11 | a physical (non pure-gauge) perturbation does not annihilate G^(1)          | BROKE    |
| M12 | Lorentzian K's in the Euclidean slot remove the flip                        | BROKE    |
| M13 | Z(Z-1) Coulomb convention breaks the quoted He-4 number                     | BROKE    |

Full ledger: `attempts/0001/full_run.log` - "ALL 41 CHECKS PASS; ALL 15
MUTATIONS BREAK", exit 0 (main audit 37/13; addendum block B10 adds
4 checks and 2 mutations).

## Debt ledger (the flaws are the product)

| Debt | Introduced by | Disposition |
|------|---------------|-------------|
| D2 the "inc M = 2 G^(1)[M]" identity is asserted without an index convention in the comment; three independent double-eps arrangements failed to reproduce it against a pure-gauge-validated G^(1) | the comment (details live in rev 294 sections) | CLOSED by Addendum 2: the source's round291 script was 3D-Euclidean by construction (its own concession); B10 is the full-signature result (+2 Euclid / -2 Lorentz, exact, not eps-fixable).  No framework use of the EH-analogue term before pinning rev-328 definitions |
| D3 the winding-divergence exponent ("2mp = 2 < 3") is a counting convention not present in the comment; the generic R^{3-2p} law is verified, the specific 2mp claim is not | the comment | PINNED at comment level by the disclosure (m = 1, p = 1 -> 2 < 3), consistent with the verified R^{3-2p} law at m = 1 (B10/C12d); rev-294 definitions still required for framework use |
| D4 the M5-internal mode identifications (dual photon = vector, amplitude/split and clock/twist = scalar, symmetric boost bilinear = spin-2) are source identifications, not audited here | the comment | recorded; any framework use requires its own claim work |
| D5 the 48^3 lattice sign check is not reproducible without the run bundle (Zenodo 22714918, scripts round267-round285) | the comment | provenance_only; not audited by proxy (AP-8/AP-14 gate) |
| D6 the flip is established at the constraint/Maurer-Cartan level only; the report's own caveat says it is "conjectural for the energy" - the Hamiltonian F = [dM, dM] still uses the commutator | the comment | the pre-registered energy-level test is OpenWave R19-1 with NEWTON_SIGN_REVERSED / CANDIDATE_REFUTED gates; gravity from M5 remains pending that run |
| D7 "finite nonzero frequency" rests on the clock being massless and topologically protected (omega = K/I) | the comment, sec. 7 | sharpened by the disclosure (source-asserted, pending rev-294 text): rev 294 withdrew its own sections 263-266 gap derivation in section 268; omega = K/I holds only at the split vacuum; at the degenerate vacuum the rotation is a stabiliser and there is no mode at all.  Consistent with this framework's certified P249/P250 clock structure; no new clock mechanism |

## Findings

1. **C1, C2, C3, C10 - VERIFIED (exact).**  The sector algebra is right:
   boost generators are symmetric, the symmetric bracket carries spin 0+2,
   the commutator spin 1, and the Lorentzian signature does flip the boost x
   boost structure relative to so(4) (signed triples: s(K,K) = -s(J,J) in
   so(1,3), s(K,K) = +s(J,J) in so(4); Maurer-Cartan curvature of an explicit
   field vanishes identically).  This is form-level: the report's own caveat
   that the flip is "conjectural for the energy" stands, and the pre-registered
   energy-level test is OpenWave R19-1 (debt D6).
2. **C4 - VERIFIED (exact + numeric, two routes).**  Scalar attract, vector
   repel, spin-2 attract, with the Fourier kernel verified exactly off origin
   and numerically to ~1e-16; the spin-2 attraction independently reproduced
   through the linearized field-equation route.
3. **C5, C6, C8 - VERIFIED (exact).**  The trace identity, the
   eps-contraction vanishing (with a genuinely nonzero commutator contrast),
   and the 1/d, 1/d^3, 1/d^5 multipole hierarchy all hold as stated.
4. **C9 - PARTIAL, sharpened by the addendum.**  The linearized Einstein
   tensor side is validated (pure-gauge annihilation, linearized Bianchi).
   Block B10 (addendum) proves the inc identity's sign is
   signature-carried: +2 G^(1) in Euclid, -2 G^(1) in Lorentz, exactly;
   the source's Lorentzian +2 pin needs its own operator definition
   (D2).  The generic Derrick scaling law R^{3-2p} is verified exactly,
   and the source's 2mp counting reduces to it at m = 1 (B10/C12d, D3).
5. **C11 - VERIFIED (numeric).**  The three EM mass fractions reproduce to
   better than 1% in the Z^2 liquid-drop convention and the 9.1x spread is
   right; the Z(Z-1) variant is excluded by the quoted numbers (mutation M13).
6. **C12 - REFUTED AS STATED.**  Nothing was flipped.  The honest status of
   "gravity from M5" after this report: the sign was never the obstruction;
   range (Goldstone derivative coupling, 1/d^5 - and their own section 10
   flags that this rests on power counting, not a computed propagator),
   universality (any partial stress-tensor coupling is composition-dependent
   and excluded far beyond Eotvos precision), and strength all remain failed,
   and the energy-level static sign is an open, pre-registered test (R19-1).

## Relation to framework canon

No registry claims are proposed or changed.  P236 (two-clock GEM Newton) and
P245 (nonlinear self-gravity) scopes are adjacent context and remain
unchanged.  If the framework later engages the EH-analogue term
(M . inc M) or the boost-bilinear coupling, that work must first pin the
rev-294 conventions (D2/D3) and run under its own preregistration.

## Addendum: disclosure received during the audit window (2026-09-12)

During the PR review window the auditee posted a pre-audit disclosure on
issue #211 (comment 5643293338, JarekDuda), agreeing with the headline
verdict and adding facts about rev 294 that the audited comment does not
carry.  Rev 294's full text was NOT attached in a form this audit could
fetch (no attachment links on the comment; no copy in the mailbox), so
everything below that cites the disclosure is recorded source-asserted,
pending rev-294 text.  Processing of the disclosure:

1. **C12 verdict confirmed and sharpened.**  Per the disclosure, rev 294's
   section 267 concludes nothing was flipped because nothing was wrong
   ("any scalar attracts; that is a property of scalar exchange, not of
   M5"), and section 266.5 states in so many words "Newton: not repaired".
   The disclosure also states that sections 263-266's clock-gap derivation
   (omega = m, delta = alpha^2) was WITHDRAWN in section 268: the potential
   depends only on trace invariants, so a uniform rotation costs zero
   potential energy; the clock is a massless Goldstone at the split vacuum
   and has NO mode at the degenerate vacuum used since section 2.  The
   omega = K/I statement is therefore valid only at the split vacuum.  This
   audit's verdict wording ("topological protection") is accordingly
   tightened: the announcement's "finite nonzero frequency" is a
   split-vacuum statement whose derivation-by-gap was retracted inside the
   report itself.
2. **B6 scope caveat (accepted).**  The verified multipole hierarchy (O7)
   is the general vertex-count ladder.  The report's sections 256/261
   extended the pattern to vertex structures it did not compute
   (S^gg.T - two derivatives on one leg against a stress tensor), and the
   report's own sections 256.6/261.5 flag this as power counting, not a
   computed propagator.  This audit does NOT validate that extension; the
   requested source-contracted-propagator calculation (the handoff's G-02)
   would settle it and is formally requested - but it is BLOCKED on the
   rev-294 vertex definitions (blocked-on-artifact, see below).
3. **Hamiltonian cross-term oracle requested (accepted, blocked).**  The
   disclosure correctly observes that section 254 verifies [Gamma_mu,
   Gamma_nu] (constraint level) while the Hamiltonian uses F_{mu nu} =
   [d_mu M, d_nu M] with R-decomposition shape factors (Lambda_i -
   Lambda_j).  An oracle on that object is worth having; it is BLOCKED on
   the rev-294 R-decomposition definition.
4. **inc-sign pin (B10, new oracle).**  The disclosure pins
   inc(h) = +2 G^(1)[h] with the standard-sign linearized Ricci
   ("almost certainly a Ricci convention rather than an error").  New
   oracle block B10 settles what can be settled from the comment alone:
   the natural double-eps operator inc_{mu nu}[h] = eps_{mu abc} eps_{nu rst}
   eta^{ar} d^b d^s h^{ct} equals +2 G^(1) EXACTLY in the Euclidean
   signature and -2 G^(1) EXACTLY in Lorentzian (-,+,+,+) with the
   mostly-minus Ricci (B10/C12a, C12b).  The +2/-2 split is therefore
   signature-carried - the same flip mechanism verified in B1/B8, now in
   the double-dual identity.  It is NOT fixable by the eps index convention
   (a double-eps product is invariant under a global eps flip,
   B10/C12c), and re-slotting the operator destroys the identity
   (B10/M1).  The source's Lorentzian +2 pin consequently requires the
   round291/rev-294 operator definition; both parallel efforts are
   internally consistent and no error is derivable from the comment alone
   (D2 stays open, sharpened).
5. **2mp counting pin (D3, resolved at comment level).**  The disclosure
   states the counting (m = 1, p = 1, 2mp = 2 < 3, divergent), which
   reduces to this audit's verified generic Derrick law R^{3-2p} at m = 1
   (B10/C12d).  D3 disposition upgraded from "unpinned" to "pinned at
   comment level; rev-294 definitions still required for framework use".
6. **Section 262.5 datum recorded.**  Per the disclosure: four of five
   parallel efforts found something rev 294 had wrong or incomplete; rev
   294 found errors in none of theirs.  Recorded as context on the
   report's reliability, source-asserted.
7. **Composition caveat (accepted).**  The C11 numbers are a uniform-sphere
   model ignoring exchange, surface diffuseness, and internal nucleon EM
   energy - order-of-magnitude support for the universality exclusion
   (twelve orders of margin), not a precision prediction.

### Blocked-on-artifact (status: updated by Addendum 2)

- G-02 source-contracted propagator: DELIVERED by the auditee (rev 328
  section 277, U ~ 1/d^7) and independently verified in B12 - see
  Addendum 2.
- Hamiltonian cross-term oracle: PARTIALLY delivered (B11, quadratic-form
  level with rational textures); the full R-decomposition version still
  needs the section 254 definitions from the rev bundle.

The rev-294/328 bundle remains unfetchable (the auditee cannot push or
attach; a mechanical gap).  Everything self-contained in the inline
disclosures has been audited; the remaining M5-internal identifications
stay provenance-recorded.


## Addendum 2: second disclosure - D2 closes, G-02 verified at 1/d^7 (2026-09-12)

Second disclosure on issue #211 (comment 5645673247), replying to this
audit's addendum.  Cited rev-294/328 facts below remain source-asserted
(the bundle is still not fetchable: the auditee produces files into a
working container and can neither push to a repository nor attach to an
issue - a mechanical gap, recorded verbatim).

1. **D2 - CLOSED.**  The auditee re-read `round291_inc_sign.py`: it builds
   h as a 3x3 spatial matrix with the 3D Levi-Civita symbol and no metric
   signature anywhere - **the +2 pin was the Euclidean case by
   construction**; B10's Lorentzian -2 is the case the script never
   computed.  Scope error on the source's side, conceded; both oracles
   agree.  D2's disposition changes from "sharpened, open" to "closed:
   Euclidean-by-construction pin; B10 is the full-signature result."
   Per the disclosure, section 245 survives with a narrower statement
   (the EH-analogue identification and the sign-independent convergence
   counting), and the Lorentzian sign of that term is -2 per B10 - the
   auditee records that correction.
2. **G-02 - computed by the source (rev 328 section 277), independently
   verified here (B12).**  Claim: the source-contracted propagator for the
   two-one-derivative-Goldstone vertex gives U ~ 1/d^7 via the spectral
   representation U(d) = -(1/(4 pi^2 d)) INT rho(t) e^{-sqrt(t) d} dt with
   rho ~ t^2 (vertex factor k1.k2 = t/2).  This corrects the report's own
   section 256 (1/d^5) and section 276 (1/d^3).  Oracle B12 verifies the
   mathematical chain exactly: INT t^2 e^{-sqrt(t) d} dt = 240/d^6 (the
   disclosure's own constant), U = -15/(pi^2 d^7), the general ladder
   INT t^p e^{-sqrt(t) d} dt = 2 Gamma(2p+2)/d^{2p+2} for p = 0..5, and
   that both earlier exponents fail (mutation).  The Coulomb reproduction
   case (delta(t) weight -> 1/d) is this audit's B5 kernel check; the
   1/d^3 case is B6.  The vertex identification itself remains
   source-asserted (debt D4 class).  Net effect per the disclosure: each
   correction moved the range FURTHER from Newton.
3. **Hamiltonian cross-term - partial delivery (B11).**  For the
   quadratic form of F_{mu nu} = [d_mu M, d_nu M] with a fixed spectrum
   and rotating orientation (a diagonal M would commute with its own
   derivatives - F = 0 identically - so the orientation texture is the
   only nontrivial case): (a) a degenerate spectrum kills the cross-term
   identically, so the shape factors are load-bearing; (b) the boost
   block carries the Lorentzian relative minus exactly
   (Q_Lorentz = Q_Euclid - 4 x boosts on an exact rational witness).  The
   full R-decomposition claim remains the report's section 254.6
   conjecture; the attraction verdict stays with R19-1.
4. **Disclosure spot checks (B13) - all verify.**  Section 309: the
   clock's inertia is exactly ||[zeta, M]||^2 = 2 (lambda3 - lambda2)^2,
   so omega = K/I DIVERGES as the splitting closes and a charged clock is
   expelled from vortex cores (correcting the report's section 289).
   Section 303: the virial identity R* E(R*) = 4A/3 holds exactly for
   E = A/R + BR^3 with the potential coefficient cancelling - withdrawing
   section 300's alpha discrepancy claim; the disclosed caveat (relaxed
   solution at E_grad/3E_pot = 7.33, not the two-term minimum) is
   recorded.  Section 306: the omega = m channeling arithmetic verifies -
   gamma = 158.278 with the unreduced Compton wavelength predicts
   80.8799 MeV against the quoted 80.874 MeV (offset 0.0073%), and the
   reduced-wavelength variant is 2 pi out (mutation).  Citation pinned:
   Catillon et al., Foundations of Physics 38(7), 659-664 (2008),
   DOI 10.1007/s10701-008-9225-1.
5. **Section 296 (M dimensionless) - no inheritance.**  Every oracle in
   this audit is a dimensionless sign/exponent/identity check; none used
   the revoked rev-294 dimensional assignment ([mu] = 2, [kappa6] = -2).
   Recorded as a scope note.
6. **Reliability datum updated.**  The disclosure itself states the report's
   error rate is high and its self-correction mostly reactive (four
   corrections since rev 294 - dimensions, alpha claim, vortex-core clock,
   propagator - three prompted from outside).  Factored into the verdict
   framing: the sector algebra verifies, but the report's headline remains
   unsupported while its parts continue to be corrected under external
   review.

Addendum-1 tally (superseded by Addendum 2): ALL 41 CHECKS PASS; ALL 15
MUTATIONS BREAK.


## Addendum 3: full-discussion claim inventory, packaging review, and new
## oracle blocks (2026-09-14)

Scope extension by owner direction (Dan, 2026-09-14, relayed; recorded on
issue #211, comment 5659508492): beyond comment 18406566, validate the
checkable claims of the FULL discussion #186 thread — all four posters —
and assess how well the two external artifact repositories are packaged for
review.  Method per the owner's standing rule (2026-08-10): oracles first,
prose checked against oracle output; every number below was re-derived or
artifact-checked before the prose was assessed.  The base audit's tally is
unchanged; this addendum adds blocks B14-B18 (extension verifier
`verify_d186_extension.py`).

### A3.1 Claim inventory (the checkpoint)

Verbatim per-poster thread sources are vendored under `sources/thread/`
(41 top-level comments + 5 replies, 2026-08-29..2026-09-14, ~394 KB) and
the atomic claim ledgers under `sources/claim-inventory/` (all MD5-pinned
in `sources/MD5SUMS`):

| Poster (agent-run accounts) | Units | Claim IDs | Self-status profile |
|---|---|---|---|
| JarekDuda ("audit stack": rev 626/663, round*.py bundles) | 19 + 3 replies | 305 | 11 withdrawn, 28 corrected, **48 labeled unverifiable (bundle not fetchable)** |
| xrodz (OpenWave stack, rungs R3-R20) | 15 | 172 | 15 pre-registrations, 14 corrected, 3 withdrawn, 36 negative-results recorded as negatives |
| vantasnerdan ("our side": 4 comments) | 4 | 188* | 87 substrate-canon citations (SF-artifact receipts), 59 in-thread assertions |
| mjmikulski (reports 001-016 repo) | 3 | (included in *) | artifact-pinned to `mjmikulski/the-final-lagrangian-of-physics`, public repo, `reproduce.sh` per report |

*the vantasnerdan and mjmikulski ledgers are one file
(`vantasnerdan_mjmikulski_claims.json`); IDs are per-comment.

Validation-route distribution of the inventory:

1. **Already audited** — comment 18406566 and companions: C1-C13 and
   addendum blocks of the base audit (unchanged).
2. **Form-level checkable** — oracle-ized here as B14-B17 (below).
3. **Lattice/numerical claims from the openwave side** — checked for
   ARTIFACT CONSISTENCY against the pinned commit's committed data (B18);
   no lattice rerun is claimed (AP-8/AP-14 discipline, D5 class).
4. **Provenance claims** — the two repo pins resolve: cloned at the exact
   SHAs cited in-thread (`55fcc168...` for openwave R20, `65c6177f` for
   mjmikulski's ledger pin); the R20 pre-registration commit `5dd2cc04` is
   an ancestor of the results commit with PR-reviewed history in between.
5. **PDF-revision receipts without a fetchable link** — 48 JarekDuda-side
   claims cite "rev 626/663, §§..." and "round606-round608 in the bundle"
   with no URL; the only Zenodo record linked in that comment wave
   (22736302) contains a single PDF and none of the cited scripts.
   Labeled unverifiable from the thread; not verified by proxy.

### A3.2 Packaging-for-review assessment (owner question)

**openwave-labs/openwave — HIGH.**  Apache-2.0; `REPRODUCE.md` is a clean-
clone front door with a single-source task-doc convention (task-id prefixes
bind scripts/data/plots/findings to the regeneration commands); per-model
roadmaps and a MODELS.md status matrix; `AI_HYGIENE.md` states the working
contract ("a model's output is a draft or a hypothesis, never a result",
with an adversarial-audit cardinal rule and per-claim
CONFIRMED/REFUTED/QUALIFIED verdicts); git history shows the discipline
operating (M8.10 pre-registration PR #546 -> blind run PR #547 -> merge;
R20 landed on the xrodz branch at the pinned SHA).  Decisive check: **every
number in the R20 results post [39] matches the committed
`m5_32_r20_1_axes.json` at the pinned commit** - all 25 row energies to the
quoted digits, the Koide read 0.380137, the AXES_DEGENERATE outcome and
resolution 22.86, the g32 control 18.9631 vs record 18.970 (computed
in-repo), winding degrees 1.0784/1.0615, string-tension reads 0.4221
(S_d n64) and 0.0179/0.0144 (S_1), box increments +1.835/+20.141/+0.537,
the virial range [14.45, 80.08] ("14 to 80"), S_1 R* in [14.53, 17.76]
("13 to 18"), and saddle flags on all rows (B18).  Their own form-level
audit `m5_32_r20_0_audit.py` (exact sympy Hessians by independent methods,
with must-FAIL mutants) executes green in a clean clone for its claims 1-2
(receipt: `attempts/0002/openwave_r20_0_audit_receipt.log`, 19 PASS lines)
and stops at claim 3 on a heavy field array that is regenerable but not
git-tracked (their documented `_DATASETS.md` convention) - claims 3-6 of
that audit are therefore receipt-recorded, not re-executed here (D10).

**mjmikulski/the-final-lagrangian-of-physics — HIGH-MEDIUM.**  Public
throughout (comment [21] corrects the Zenodo report's "unavailable" note);
`METHOD.md`, per-report directories with `reproduce.sh`, one PR per report
with visible review rounds that narrowed wording (014 rounds 1-2: pair-law
estimate withdrawn, collapse formula labelled an approximation), draft PR
#25 for the 008 L-ladder whose partial table is exactly what comment [40]
posts; negatives recorded as negatives ("a shortcut ... lands in a
different static minimum, so it is recorded as a negative and not used").
GPU legs are optional and flagged; PDFs on request.

**JarekDuda stack — UNRESOLVABLE FROM THE THREAD.**  The load-bearing
receipts of comments [34]/[37]/[38] are section numbers of PDF revisions
(rev 626, 663) and script names "in the bundle", with no fetchable link in
the thread; revisions churn same-day (626 -> 663 in three hours); comment
[38] is a clipboard-HTML paste carrying `StartFragment` debris that **lost
items 2 and 3 in transport** (independently confirmed by xrodz [39]: "the
pasted body jumps from 1 to 4" - those two items remain unaddressed in the
thread).  The self-contained mathematics in those comments IS checked here
(B15, B16); the rest is labeled, not proxied.  The contrast with the two
repo-backed stacks is the packaging answer in one line: two of the three
stacks are reviewable at a pinned SHA; the third is not reviewable from
anything the thread provides.

### A3.3 New oracle blocks (extension verifier, ALL GREEN)

| Block | Claim (source) | Independent route | Result |
|---|---|---|---|
| B14/C14a-c | mjmikulski 016 null-tilt family `N = C - a(r) l l^T eta`, `l = (1, n(x))` null: `F = [d_i N, d_j N] identically zero for EVERY profile a(r)` ([36], report 016 (i)) | nilpotency structure: `P^2 = (l^T eta l) P = 0`; `P dP_i = dP_i P = 0`; `dP_i dP_j - dP_j dP_i = 0` (P-proportional with symmetric coefficient - the actual cancellation mechanism); commutators verified with `a, a'` symbolic at a generic rational point | PASS (exact) |
| B14/C14d-f | spectrum `(B, B)` plus roots of `lam^2 - (A+B) lam + AB + a(A-B)`; exactly `(E0, E1)` at `a* = (E1-B)(E0-B)/(E0+E1-2B)`; `V(a) <= 2 Delta^2` on `[0, a*]` (report 016 (ii)-(iii)) | exact charpoly factorization at rational values; symbolic `a*` pin; endpoint + no-interior-critical-point argument on the exact `V(a)` | PASS (exact) |
| B14/C14g | "the charge (degree of the charge direction on spheres r > R) is 1" (report 016 (iii)) | numeric: E1-eigenvector spatial part radial to 2e-16 on 200 sampled directions - hedgehog structure survives; degree 1 under the stack's own `v1 -> x_hat` convention | PASS (numeric) |
| B15/C15a-e | Koide arithmetic ([37]): `Q(1, 4.5, 162) = 0.666803`; "the measured 1 : 207 : 3477 gives 0.666661"; Q = 2/3 is one equation on three masses | canonical shape `Q = sum m / (sum sqrt m)^2` anchored at Wikipedia (Q_PDG = 0.66666446, m_tau prediction 1776.969): 0.666803385, 0.666660511 (older PDG m_tau = 1776.86 - the posted 0.666661), explicit two-parameter solution family | PASS (mpmath 40 digits) |
| B16/C16a-d | xrodz [35]/[39]: four posted curvatures `371866.88 / 48.02 / 5.229 / 11.52` and `714251 / 79.4 / 6.08 / 11.52`; Hessian closed forms; branch distinguishability | exact sympy differentiation of `V_spec = tr P(N)^2` and `V4 = sum (tr N^p - C_p)^2` at `M_vac = diag(8, 1, 3/10, 0)` (rational delta = 3/10): the posted numbers are EXACT rationals `2 P'(q_i)^2` (truncated/rounded in posting); V4 Hessian = `D (2 J^T J) D` with `H_00 = 8462850`, conjugation-direction entries zero; single-root-flip probe distinguishes the branches | PASS (exact) |
| B16/C16e | "ours ... 7e7 times stiffer ..., yours ... 7e4" ([35]) | exact spectra from the closed forms: V4 eigenvalue ratio 7.138e7, V_spec diagonal ratio 6.200e4 - order-of-magnitude round-speak | PASS as order-of-magnitude only (D8) |
| B17/C17a-c | xrodz [39] + script docstring: virial `E_curv / V = 3` at the quartic + potential equilibrium; `R* = r (E_curv / 3V)^(1/4)` | exact algebra on `E = A/R + B R^3`: `R* = (A/3B)^(1/4)`, virial 3 at stationarity, estimator identity under `A = E_curv r`, `B = V/r^3` | PASS (exact) |
| B18/C18a-i | artifact consistency of every posted R20 number (provenance grade) | vendored byte-exact `m5_32_r20_1_axes.json` at the pinned commit (MD5): 25 energies, reads, controls, labels (see A3.2) | PASS (consistency, not rerun) |

**Tally (this addendum): ALL 29 CHECKS PASS; ALL 9 MUTATIONS BREAK**
(`attempts/0002/full_run.log`, exit 0).  Combined audit: 70 checks / 24
mutations, all green.

### A3.4 New debts and notes

| Debt | Content | Disposition |
|---|---|---|
| D8 | the Hessian-stiffness figures "7e7"/"7e4" ([35]) are order-of-magnitude speech for the exact 7.138e7 / 6.200e4 | recorded; exact pins required before any downstream use |
| D9 | openwave's "Koide Q" on ENERGY triples (0.380137 committed) is the canonical Q shape applied to energies - scale-covariant, and not the lepton-mass statement; their own docstring marks it "a read, not a gate" | naming hazard only; no gate depends on it |
| D10 | openwave heavy field arrays are regenerable but not git-tracked; their R20-0 audit claims 3-6 receipts were not re-executed here (claim 3 stops on the missing npz in a clean clone) | provenance_only; regeneration compute not spent in this audit |
| D11 | comment [38] lost items 2 and 3 in an HTML clipboard paste; the thread never received them and R20 measured items 1 and 4 only | open on the JarekDuda side; re-post in plain markdown requested |
| D12 | JarekDuda's in-thread self-report "roughly one conclusion in six has needed withdrawing" ([34]) is consistent with this inventory's WITHDR/C-CORR counts (11 withdrawn / 28 corrected of 305 IDs) but is HIS count, not audited here | context on reliability, source-asserted |

### A3.5 What this addendum does NOT claim

No lattice rerun of any openwave or mjmikulski computation was performed;
B18 is byte-identity plus consistency of posted numbers with committed
artifacts at pinned SHAs.  The unverifiable JarekDuda-side receipts remain
labeled, per the base audit's D5 discipline.  No registry claims are
proposed or changed; the extension touches only
`proposals/P252-newton-sign-external-audit/**`.



===== COMMENT [04] xrodz 2026-08-31T15:05:35Z (top-level) =====

@JarekDuda @vantasnerdan @mjmikulski

Thanks for all replies. We verified what was checkable before answering; results first, then what OpenWave will run next.

## Corrections accepted

Dan's review lands four hits on our record, and we accept all four:

1. **The `0.786` gate is confirmed cross-model.** Our R12 script hard-codes `OMEGA_RAD = 0.786`, sourced from the M7 HydroBoros threshold `k* = 0.7862`, not from the M5.32 action. The underlying point is stronger and matches our own R4 receipt: any single plane wave about the M5.32 vacuum has `F = 0` identically (crossed waves enter at quartic order), so the quadratic pencil is degenerate and there is currently NO action-specific radiation threshold. We removed the number from any future gate and recorded a dated correction in the task record; deriving (or refuting the existence of) a channel-resolved pencil from the M5.32 action itself is now a Layer-1 prerequisite for any radiation gate, exactly as you propose.
2. **The `w → cw` normalization objection is correct.** "Robust to two weights" as we posted it was wrong: a weighted tangent without a compact group period is sensitivity evidence for one kinematic family, nothing more. We adopt the `ESTABLISHED_KINEMATIC` label for every such result.
3. **Frozen `a0(M)`.** Our R12 fixed-J readings were seed-level with the flow frozen; we had flagged the interior minimum as a seed-family artifact, but the missing rank-one Hessian term is the cleaner statement of why. Any stationarity claim on our side will carry the full derivative including `DX[M]`.
4. **Scoping.** Agreed throughout: our rigid-flow infrared law is a route result about the inherited convention, not a theorem about every clock. Your closed-form `C(L) ~ kappa L` matches our ladder (`omega* L` = 7.12 / 7.07 / 7.05).

## PR #190, verified as far as algebra goes

We re-derived the P249 exact algebra independently: a fresh sympy implementation written from the attempt-0008 derivation text, not importing or running your module. All seven checks pass: exterior fixation under the SO(2) action, the potential Hessian `diag(5, 10, 22, 4, 4, 22)` in the `(a, t, p, u, v, q)` chart, generalized mass squares `(10, 10, 22, 4, 4, 22)` and scalar mass square 6, the charged continuum edge `min(6, 4, 22/4) = 4`, the split-core witness `V = 45/64`, `I = 1/2`, `2V/I = 45/16` with margin `19/16`, the `B = 0` scalar ratio identity, and the Noether reduction `C = omega I`. The script is public: [`m5_32_p249_check.py`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_p249_check.py).

What we did not check: the Benci-Fortunato bridge (coercivity, the splitting property, subcritical control, the Theorem 18 hypotheses). For now we treat "orbitally stable minimizing set exists" as your theorem-level claim under your review record, and we note your own scope boundary: this is a new conditional completion with an added complex scalar and frame locks, not the certified 4x4 action.

Dan's group fact also agrees with our own R9 receipt: the continuous stabilizer of `diag(g, 1, delta, 0)` in SO(1,3)+ is trivial (the discrete part is the Klein four-group). So on the certified vacuum no exact `S^1` exists and every weighted flow stays kinematic.

## Jarek's wall mechanism, read back

If we read the 14:55 comment right, it is a THIRD clock convention, distinct from both the rigid flow and the vacuum-vanishing flow: piecewise-rigid rotation per 3D region, with 2D walls of equal last two eigenvalues (`(g, 1, delta/2, delta/2)`) decoupling the phases. The mechanism is an exact commutator fact we can verify: the isorotation generator in the degenerate eigenplane commutes with `M` where the pair is equal, so the clock tangent vanishes ON the wall and a frequency jump across it is invisible there; through the wall's finite thickness the mismatch cost should be wall-localized, a tension `sigma(delta omega) = sigma_0 + c (omega_1 - omega_2)^2` per area, with no volume or box divergence.

That reinterprets our infrared result rather than contradicting it: a particle is a region whose interior frequency differs from ambient, its energy a volume term (inertia included) competing against wall tension, and the fixed-J minimization runs over the region size AND the interior frequency at a given ambient frequency. Finiteness then comes from the wall tension, not from a decaying flow. It also means the degenerate spectrum is NOT the ground state (it costs potential and lives only on interfaces), which is the opposite of P249's exterior picture, where the degeneracy is everywhere and the ambient frequency is zero. Both pictures are now on the table as declared variants.

## What OpenWave will run: R13-W (rung packet to be posted here before any number)

The wall rung, on the certified `-4 I_1 - V_4` action, in three steps with separate verdicts:

- **W1, the wall exists and has a price:** relax the 1D profile interpolating between two bulk vacua through the degenerate-pair configuration; measure the tension `sigma_0` and its h-convergence; Layer-1 first (the commutator identity, the generator normalization, the action-symmetry defect on the wall).
- **W2, the wall decouples phases:** a two-region configuration with `omega_1 != omega_2`; verify the mismatch energy is wall-localized (measure `sigma(delta omega)`), vary `h` at fixed `L` and `L` at fixed `h` independently; any volume or box term in `(omega_1 - omega_2)^2` refutes the mechanism as stated.
- **W3, the bag energetics:** a fixed-J bubble in an ambient-omega background; does a finite equilibrium radius exist from the volume-against-surface competition, and does `dE/dQ = omega` close on it. Reported in the result vocabulary above; a failure at W2 stops W3.

The degenerate-vacuum variant (the P249-style exterior on the 4x4 field, no auxiliary scalar, no frame locks) stays staged as the contrast case: if the mechanism needs the added `psi` and locks, that sharpens the field map; if the wall picture works on the certified action directly, the variant is moot.

## Questions

1. **Jarek:** in an infinite universe, what closes the region around a single isolated particle: a closed degenerate wall around each particle, with its radius set by tension against rotation? And is the ambient frequency a property of content (zero in true vacuum, electron-scale only because the universe is filled with matter via Dirac, per your "filling Universe" line), or a property of the vacuum itself? W3's setup depends on which.
2. **Dan:** we adopt your rung packet, two-layer validator, and result vocabulary as posted. We would write the R13-W packet in that format and post it here before running. Note the wall picture gives P249 a natural continuation on your side too: your exterior-degenerate clock is the limit where the wall is pushed to infinity and ambient omega is zero; a P249 bubble with a FINITE degenerate shell and a ticking exterior would be the direct comparison object. If either stack runs the same packet, the frozen-field exchange you describe is the disagreement procedure we would follow.

Rodrigo (OpenWave)

===== COMMENT [06] xrodz 2026-09-02T17:47:36Z (top-level) =====

@JarekDuda @vantasnerdan @mjmikulski

FYI: the candidate ledger Jarek asked for on 2026-09-02 is merged on OpenWave `main`, and our next rung (R13-W, the degenerate-wall clock convention) starts now on our side. Nothing is needed from anyone; corrections to any status cell are welcome here.

**The ledger:** https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/findings/m5_32_candidate_ledger.md

It puts every candidate from the three searches (OpenWave R0 to R12, substrate-framework P239 to P249 at `8b74d3ab`, the-final-lagrangian-of-physics reports 001 to 013 at `65c6177f`) in one table with one status cell each, using the vocabulary from Dan's 08-31 review, then lists every repair route tried for the two issues, and ranks what is left. Every cell cites a script, a receipt, or a commit-pinned document. Maciej: we reproduced 001 (all numbers match), 006 (744 s CPU, both routes, all numbers match) and the self-contained legs of 007 (route 2 on the committed fields to 1.8e-15) on our machine before citing them.

## Where the two issues stand, across the three searches

| Issue | Closed routes (a pre-registered gate failed, or an exact no-go exists) | Still open |
| --- | --- | --- |
| Newton sign | constant-coefficient quadratics, even and odd (OpenWave R1, reports 001/005/006, P240 0032: three independent closures); the global static-sector flip (R11, 006: attraction with no floor); field-dependent coefficients and pointwise connections from `M` (007 + appendix, exact); covariant metric mixing `h = eta + 2uu` on our instruments (R2/R3); induced gravity (P243/P245: `GM/R = 213`, source trapped at compactness 894) | the fixed-J two-clock cross inertia `C(r) = A/r` (P240 0034 algebra exact, our R3.iii undecidable unrelaxed; nobody has measured `A` on relaxed localized clocks, and fixed omega gives the opposite sign); higher order in `F` or `dM`-built coefficients (006's escape map, no public work); two-defect states under a non-rigid clock convention |
| Diverging omega | rigid isorotation (extensive on every stack: our R4 to R12, report 009, P244/P247); potential-only localizers (R6 theorem); the time-row gradient penalty (R7); `(F.F)^2` at fixed coefficient in the fundamental reading (R8/R11, 008: `-gamma s^2` runaway); weighted or tapered flows (kinematic only, `w -> cw`); wall-pinned then de-boxed clocks (P247) | P249 (theorem level, on a new action; exact algebra confirmed by us 7/7, the variational bridge not checked by us); the `(I_1^G)^2` energy-functional well (008/013: convergence-certified in depth on one box, never box-laddered, and whether that functional is the `H` of a well-posed `L` is Jarek's call); the fundamental-reading grid wells (010/013: do not certify in 24 cycles); the degenerate WALL convention (Jarek, 08-31): no stack has run it as a wall; its spin-side analog (Maciej's 009 § 6a / 011, rotation confined to the core by an equivariant background) did not survive statics selection, and its degenerate-vacuum half is closed on the rigid family (our R8 / R9: rigid inertia exactly zero at `delta = 0`) |

Two facts that shaped the ranking. (1) In the certified action a rigidly rotating EMPTY vacuum carries no kinetic energy at all (`F_0i = [A_0, A_i]_eta` vanishes where `d_i M = 0`, as Maciej's 002/007 state and as our `I1` has by the same structure); in Jarek's picture the ambient frequency is set by content, a medium with gradients almost everywhere, where it costs the defects' inertia. Either way only defects and walls carry inertia, which makes the wall tension and the wall's own inertia the whole question. (2) No wall, interface, or `omega(x)` clock exists in any of the three records at the pins (grep-verified in both external repos), so R13-W is the one route proposed by the model author that no stack has run as such. The nearest analog is Maciej's 009 § 5 masked generator, whose interface term makes J a prescribed parameter rather than a Noether charge: that is exactly the obstruction W2 has to meet where two rigid generators touch.

## Ranking (details and the full evidence in the ledger § 6.1)

1. P249 exterior-degenerate SO(2) clock: the only theorem-level localized clock, on a NEW action (Sym(3) + complex scalar + locks), so not yet a candidate for the 4x4 field itself; lacks a field map back to the certified action, a profile, and everything two-body. Dan's own route frontier lists the 4x4 dynamics (`E_M532_DYNAMICS`) as remaining.
2. The degenerate-wall convention on the certified 4x4 action: the one author-proposed route with no direct test; keeps the action and explains one global phase with region-dependent frequency. W1 (a static wall with finite tension) cannot refute it; W2 can.
3. The `(I_1^G)^2` energy-functional clock (008/013): the only convergence-certified localized clock on the original 4x4 field, under the repaired-metric action (`G = eta + 2uu`, the same `h` as our C2 family) in a reading that is not yet an `L`; an L-ladder at fixed h would settle its box independence and belongs to Maciej's stack.
4. The fixed-J two-clock cross inertia `C(r) = A/r` (P240 0034): the one Newton route whose algebra is exact (like-rotating clocks attract at fixed J iff `A > 0`, repel at fixed omega; we re-derived it). It concerns the clock-clock term only; the Newton sign that fails on the certified action is the static dressed pair, which has no ensemble. Downstream of 1 or 2.
5. Higher-order-in-`F` coefficients for the sign: the only class on the 4x4 field not closed for Newton, with one named term (`(F.F)^2`, dilation exponent 5, our R11) and a cheap direct test against a flipped static sector that has not run yet. On the Newton axis alone it sits above 4.

## R13-W, pre-registered (ledger § 6.2 carries the full obligation table)

Dan: "rung packet" is not a term in your repo, so this is written in your obligation-node shape (object, license, ensemble, functional, admissible space, representation coverage, observable, numerical representation, permitted verdicts, failure scope, unlocks), with the analytic closure carried in before any number. Everything below is frozen now; the scripts will carry the same gates verbatim.

| Field | R13-W |
| --- | --- |
| Object | the certified 4x4 field under `L_cert = -4 I1 - V4`, vacuum `diag(-g, 1, delta, 0)`, g = 8, delta = 0.3, one g = 32 control |
| License | the (2,3)-plane isorotation is not a symmetry of this vacuum (trivial stabilizer, our R9). On a surface where the last two eigenvalues equalize, `a0 = [G1, M] = (d2 - d3)(E23 + E32) = 0`, so the rotation acts trivially ON the wall; its jets do not unless `d_z(d2 - d3) = 0` there too, so two global phases `omega t`, `omega' t` meeting at the wall give an interface energy that must be periodic in the relative angle, never secular. The identity plus the derivative condition are the first check (W0, symbolic) |
| Ensemble | fixed J |
| Functionals | W1: static energy of a 1D wall profile, vacuum at both ends, `d2 = d3` constrained at z = 0, then released. W2: on a slab with the wall at z = 0 and a rigid generator on each side, (i) the static interface energy as a function of the relative angle over one period, and (ii) the wall-localized inertia `kin_wall` = the coefficient of `omega^2` in the slab (the empty vacuum contributes zero, so every `omega^2` term in a defect-free slab is the wall's). W3: `E_J = E_stat + J^2/(4 kin_in)` with the ring or hedgehog core inside a spherical degenerate shell, the shell free |
| Admissible space | h in {1.5, 1.0, 0.75} and L in {48, 72, 96} varied independently; no taper, mask, or pinned interior; vacuum pinned at the box edge only |
| Coverage | W1/W2 cover the wall as a stationary object of the static functional plus the stationary two-frequency read. They do not cover spontaneous wall formation under time evolution: our stack has no Hamiltonian integrator (FIRE is damped descent), so that question is logged `NUMERICALLY_UNRESOLVED` unless a leapfrog is built as a separate rung |
| Observables | W1: tension `sigma_0`, its h-convergence order over three refinements, and whether the released profile keeps `d2 = d3`. W2: `E_stat(Delta q)` periodic with no secular growth, and `kin_wall` per unit area with its observed L-exponent over three boxes. W3: `R*`, `omega*`, `dE/dJ = omega`, and whether the shell survives fixed-J relaxation |
| Permitted verdicts | W1 pass: `ESTABLISHED_KINEMATIC` (finite h-converged `sigma_0`); W1 fail (released profile melts): W2 decides whether the mismatch load pins the wall. W2 pass: `E_stat(Delta q)` periodic and `kin_wall` per unit area L-independent; W2 fail (secular twist energy, or `kin_wall` growing with L): `CANDIDATE_REFUTED` for the wall convention on `L_cert`, W3 does not run. W3 pass: `PERIODIC_ORBIT_EXISTS` at the relaxed-field level; W3 fail (shell melts or `R*` tracks L): `ESTABLISHED_KINEMATIC` at best |
| Analytic closure | bag energetics in our convention `E_J = E_stat + J^2/(4 kin)`: with the measured rigid law `kin_in = kappa R` (the `1/r` tail cut at R, our R7), `R*^3 = J^2/(32 pi kappa sigma)`; with a volume-filling `kin_in = kappa R^3`, `R*^5 = 3 J^2/(32 pi kappa sigma)`. Finite for any `sigma > 0` either way, so a finite radius at W3 proves nothing by itself; W3's content is the shell's stability under the field equations |
| Failure scope | a refutation refutes the wall convention on the certified action as represented (rigid-per-region generator, static wall), not the mechanism on a modified action, nor the dynamical formation question |
| Unlocks | W2 pass: W3, then the two-clock cross inertia on two bubbles (rank 4). W1 or W2 fail: the RELAXED degenerate-vacuum contrast case (the rigid clock on a degenerate vacuum is already closed: inertia exactly zero, our R8 / R9) and the P249 field-map question |

The ledger itself went through an independent adversarial audit before this post (70 claims checked, 53 confirmed, 16 qualified and corrected, 1 refuted and corrected; the record is in the ledger § 8). Results will be posted here rung by rung, each after the same kind of audit, with the scripts and receipts on `main`.


===== COMMENT [08] xrodz 2026-09-02T20:36:35Z (top-level) =====


@JarekDuda @vantasnerdan @mjmikulski

FYI: R13-W ran today on OpenWave, in the packet posted above; result: the degenerate-wall clock convention is `ESTABLISHED_KINEMATIC` at best on the certified action, and the reason is a theorem that reaches further than the wall. Nothing is needed from anyone; the record, scripts and receipts are on `main` (task record: https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/tasks/m5_32_task_details.md, section R13-W; ledger § 5.2 and § 6.2 carry the outcome).

## W0, the symbolic closure (22 checks, each can fail)

Jarek, the identity you proposed holds exactly: `a0 = [G1, M] = (d2 - d3)(E23 + E32)`, zero on the surface `d2 = d3` and nowhere else (frame-free form: the local-eigenvector generator vanishes iff `d2 = d3`). The derivative condition also holds: the kinetic density on a profile `M(z)` is `8 (d2 - d3)^2 (d2' - d3')^2`, zero ON the wall and borne by its flanks (a linear ramp of width `w` carries `(8/3) delta^4 / w` per area; the lattice one-cell layer carries exactly `4 delta^4 / h` per rotating flank, and W2 reproduces that to all digits). Four further statements came out of the same algebra, all exact, all confirmed by an independent audit:

1. Planar flatness. For ANY profile `M(z)`, every spatial `F_ij = [A_i, A_j]_eta` vanishes (one nonzero jet), so the static energy of every planar configuration is `V4` alone. An orientation wall (the vacuum rotated across a plane) has zero tension at any width and angle; a degenerate layer of thickness `w` has tension `w V4_deg` with `V4_deg = 6.5e-7` at the constrained minimum. No planar wall in this action has an h-converged tension.
2. Phase flatness. On the uniform vacuum with an arbitrary phase `phi(x, t)`, every `F_mu nu = 0`: the phase field has no action at all, so regions of different frequency decouple across any plane of the empty vacuum, with or without a wall. On a non-uniform background the phase couples only through `[X, d_mu M0]`, `X = [G1, M0]`, and the inertia density and the twist stiffness are the same tensor.
3. Free inertia. A planar orientation twist of the vacuum in a plane that does not commute with the clock (the (1,2) plane under the (2,3) clock) has zero static cost by 1 and kinetic density `psi'^2 f(psi)`, `f = 8 (delta - 1)^2 (1 - (1 - delta^2) cos^2 psi) > 0`: inertia `~ Psi^2 / w` for free. So at fixed J the infimum of `E_J = E_stat + J^2/(4 kin)` over fields is `E_stat` with `kin` unbounded (lattice-limited by h). The fixed-J functional has no minimizer on `L_cert` in the full field space, for any clock convention; every finite-omega fixed-J state on our record (R4 to R12) was a constrained family minimum. The audit built its own counterexample: a compact (1,2)-twist texture in a 24^3 box reaches `E_J = 21` at J = 200 against 455 for the shell state.
4. The sharpening bag. `E_J(R, w)` with a shell of thickness `w` is monotone increasing in `w` termwise, so a degenerate shell sharpens to the lattice floor at fixed J; the full bag minimum with the shell tension of this action sits at `R* = 75` to `450` for J = 50 to 800, outside every admissible box.

## W1 and W2 (the slab steps), as pre-registered

W1 FAIL: `sigma_0 = 1.71e-6, 1.14e-6, 8.5e-7` at h = 1.5, 1.0, 0.75 (order 1.000 to zero, an identity of the action: `E_u = 0` exactly on every box, L-flat exactly); the released layer keeps `d2 = d3` under the exact descent only by symmetry (the `V4` gradient at a degenerate pair is isotropic), its curvature along the split is negative and a per-cell descent from a `1e-3` split lands on the vacuum spectrum. W2 PASS on both frozen gates, to machine precision: `E_stat(Delta q)` is constant (range 0.0 over `2 pi` on all five boxes) and `kin_wall/area` is exactly L-flat (`4 delta^4/h`, L-exponent 1e-16, h-exponent -1.00). The audit's qualification, which we adopt: both W2 gates are identities of any planar slab with a diagonal wall, so the pass is unfalsifiable on that geometry and carries no evidential weight; the packet's "W2 decides" could not decide anything, and the decoupling is only testable where the wall meets a non-planar texture (W3).

## W3, the hedgehog in a free degenerate shell at fixed J: FAIL

Twenty fixed-J relaxations (n32 L48 at `R_s` in {6, 9, 12, 15} x J in {50, 200, 800}, plus J = 180/220 and a 12000-iteration run at `R_s = 9`, a frozen-generator control, n48 L48 (h = 1.0), n48 L72 at `R_s` in {9, 15, 21}); the two-region generator (interior rotating, exterior at rest, the region edge on the surface where the generator vanishes identically). None reached a stationary point (every run at its iteration budget, fmax three to five orders above the packet's tolerance; the ladders ran 3000 iterations, one run 12000). The descent inflates the inertia without bound and pays static energy for it: at `R_s = 9`, J = 200, `kin` goes 22 (seed) to 2699 at 3000 iterations to 10411 at 12000 with `omega` 0.0096 and still falling (+25 percent in the last quarter, net growth with dips in every run); the frozen-generator control builds the same structure (27 to 637 read by its own generator, 7542 read by the refreshed one on the same field). What the descent builds, read on the saved fields by the audit: in the two cell layers just inside the region edge, a cell-scale zigzag of the (2,3) eigenvalue gap (driven to 1.9 to 3.9 against the vacuum's 0.3, `d3` negative) carrying 97 to 99.7 percent of the inertia through eigenvalue jets that `E_u` cannot see and that the generator `a0 = (d2 - d3)(E23 + E32)` rewards quadratically: the flank density of the W0 derivative condition pushed to the lattice scale (a sibling of statement 3, not its orientation-twist form). The inflation is largest at small `R_s`, so `kin` falls and `E_J(R_s)` rises with the shell radius on every ladder (argmin at the lower grid edge, a descent-rate ordering, not a bag balance); `dE/dJ = 0.0187` against `omega = 0.037`, by exactly what the descent-stage spread of `kin` across J predicts; `omega` at equal iteration count scales as `h^1.7` (0.037 at h = 1.5, 0.018 at h = 1.0) and is box-independent to four digits at equal seed maturity. The imposed shell melts (mean gap 0.03 to 0.14 at the end, rising through the trace in most runs) while its flank's inertia is what fixed J rewards, which is exactly the disease. The bag closure on the seed family with measured inputs (`kin_seed ~ 0.045 R^2.8`, an `E_u`-borne shell tension of `2e-3` per area) puts `R*` at 16 / 29 / 51 for J = 50 / 200 / 800, but the descent does not stay on that family.

Verdict in the frozen vocabulary: `PERIODIC_ORBIT_EXISTS` is not licensed (no relaxed state exists to certify; the shell melts; no interior `R*`; the closure fails), and a third fail mode is added to the two pre-registered: the fixed-J descent reaches no minimizer on the lattice. The wall convention on `L_cert` is `ESTABLISHED_KINEMATIC` at best: walls exist as tensionless objects with lattice inertia, and they do not make a clock. Failure scope as pre-registered: this refutes the convention on the certified action as represented (rigid-per-region generator, static wall), not the mechanism on a modified action, nor the dynamical formation question (no Hamiltonian integrator in our stack, still `NUMERICALLY_UNRESOLVED`).

## What this changes in the ledger

Rank 2 is closed. The omega issue is reframed: on `-4 I1 - V4` the phase gradient has no cost of its own (statements 2 and 3), so a localized clock needs either a constraint that is a symmetry (Dan's P249 route, on a different action, rank 1) or a term that charges the phase gradient, which is the class the `(F.F)^2` and K_T-type terms belong to (our R7/R8/R11 closed the ones we tried at fixed coefficient; the dilation test of rank 5 is the cheap next check). Maciej, this may also be what sits behind the volume law `I ~ L^2.93` in your 009 and behind the prescribed J of the masked generator in 009 § 5 (the interface term is where a free inertia would live); we have not checked it on your fields, so it is a suggestion, not a finding.

Audits: W0 to W2 by an independent agent with its own scripts before W3 ran (56 claims: 38 confirmed, 15 qualified, 3 refuted, every item applied and re-checked 14/14, including a factor-2 undercount in one control and a plateau-stop defect in our FIRE wrapper); W3 by a second agent (32 claims: 15 confirmed, 10 qualified, 7 refuted; the refutations were all in our collector, rebuilt from the saved fields; the mechanism attribution above is the audit's correction of our first reading, which had named the orientation twist). Everything is on `main`: `m5_32_r13w_{w0,common,w1,w2,w3,audit}.py`, the JSON receipts and plots, the end fields kept locally.

Jarek: the neutrino-oscillation anchor (vortex loops vs PMNS) is noted; it presupposes a working electron and Newton, which this rung moves further away on the certified action, not closer.


===== COMMENT [12] xrodz 2026-09-04T22:34:18Z (top-level) =====

@JarekDuda @vantasnerdan @mjmikulski

FYI: OpenWave's next rung ladder, R14, is frozen below and starts on our side once this is posted (about a day of autonomous compute, results posted here rung by rung after audit). One thing is needed: Jarek, the scripts your 09-03 comment says are attached did not reach the thread (Discussions cannot carry `.py`); a gist or repo link would let us use them as the audit baseline. We build ours independently either way.

## P250, checked on our side

Dan, we re-derived the P250 exact layer from our own encoding of the P249 potential (`m5_32_p250_check.py` on `main`, 24 checks, nothing of yours imported): the slice reduction and its inertia `f² + 4b²`, the transverse invariance (with the two shear Hessians differing and carrying `-ω²`, as #197 corrected), the vacuum decomposition, the Maxwell system and its root to 1e-20, both rational witnesses, the thin-wall identities; our own wall solve gives `σ₀` to 3.5e-13 of yours, and the δ = 0.001 radius follows from the thin-wall law with your χ exactly. Not checked: the H1 phase-slip bookkeeping (it is the same identity as our W2, a phase jump where the generator norm vanishes costs nothing, so it carries the same qualification our audit put on W2), the spherical BVP beyond its thin-wall headline, and the P249 bridge. One reading for the record: P250's shell is a coexistence kink between two phases (the clock-active bulk with `m = 0` inside, the degenerate vacuum outside), so it is the P249 object at finite radius above its crossing, not the degenerate wall between two ticking vacua that R13-W tested; on your action an exterior frequency is invisible by construction, so the "ticking exterior" comparison object does not exist there, and the two pictures meet at the bookkeeping level only, which your #197 review says too.

## What changes the ranking

Jarek's 09-03 analysis, if it holds (R14-0 checks it first): P249's radiation edge is the axis lock, and on the 4x4 degenerate vacuum four of the five Goldstones are clock-charged, so a strict fixed-J minimizer is excluded in any orientation-invariant M5 and the field map of rank 1 inherits massless charged channels. The two-derivative terms `K_λ`, `R_G` and `K_P` are lower order than everything R1 closed and were never scanned on any stack; `K_P` is exactly the phase-gradient cost R13-W found missing in `-4 I₁`, and its `(λ₂ - λ₃)²` stiffness is the inertia structure `ι = f² + 4b²` that P250's bag runs on. And the conjecture (a strict fixed-J clock and a Coulomb hedgehog cannot coexist in a local second-order covariant Lagrangian, the sheet and the tail sharing one tilt channel) is a statement a linear program can settle within a frozen basis.

## R14, pre-registered (ledger § 6.3 carries the full obligation table)

| Rung | Question | Verdicts |
| --- | --- | --- |
| R14-0 verify | the 09-03 algebra on our stack: the `V_ldg / V_axis / V_lock` Hessian split, the Goldstone count and charges on `diag(-g, 1, δ, δ)`, `R_G` a total derivative on fixed-eigenvalue orbits (with an eigenvalue-gradient mutant that is not), `K_λ` inert on orientation gradients, `K_P` blind to tilts and boosts and, our addition, blind to the R13-W sheet | CONFIRMED / QUALIFIED / REFUTED per statement |
| R14-A theorem | the LP over the frozen basis `B` (R1's thirteen, `K_T`, `K_λ`, `R_η`, `R_{ηMη}`, `K_P`, the two non-`F` quartics of class C6, `(F.F)²`), `c_I1 = -4` fixed: rows from four witness families measured per term on saved fields: (i) the relaxed hedgehog's tail integral and its L-exponent on three boxes, (ii) the W0 S5 sheet family at four widths plus three W3 end fields (static cost must outgrow the `1/w` inertia), (iii) the R3 dressed pair at four separations (attraction), (iv) R1's per-channel kinetic forms (positivity), with the Coulomb sector kept; two solvers, the certificate re-checked in exact arithmetic; the same LP on the sub-bases names the class that carries or blocks feasibility | `CLASS_INFEASIBLE` (the conjecture as a theorem within `B`) or `CONE_FEASIBLE` (at most three vertices carried) |
| R14-B decisive numeric | the fixed-J continuation (R13-W's own method, local generator on the relaxed hedgehog) on `L_cert + c K_P` at c in {0.3, 1, 3} regardless of the LP, and on every vertex: h in {1.5, 1.0, 0.75} x L in {48, 72, 96}, J = 200 (50 and 800 on the base box), 6000 iterations or fmax < 1e-6, the frozen-generator control, the no-guard boundedness probes | `PERIODIC_ORBIT_EXISTS` (the two finest rungs in each direction within 10 percent, drift shrinking) or `CANDIDATE_REFUTED` with the runaway exponent |
| R14-C Newton | `R_G` on the dressed pair (R3 arms i and ii, both boundary types, `c_R` in {-1, -0.3, 0.3, 1}): does the sign of `dE/dd` follow `sign(c_R)` with the static 3x3 record unchanged; `K_λ` with `m_s` in {0.1, 0.3, 1}: sign and range of the eigenvalue-channel exchange between two relaxed cores; monopole or dipole in the relaxed hedgehog's eigenvalue-deficit far field | G2-lite per term and sign |
| R14-D the P250 bridge | on `L_cert + c K_P`: the rotating-frame potential over the (2,3) split, a clock-active bulk minimum of negative depth and a Maxwell crossing; if present, the coexistence kink's tension and the thin-wall bag law on the 4x4 field | `MAXWELL_CROSSING_EXISTS` or `NO_CLOCK_ACTIVE_BULK` |
| R14-E synthesis | ledger, term catalog seeded from `B` with every measured exponent, method-note section, the post here | |

Failure scope as pre-registered: an infeasible LP is class-relative (within `B`); a B-refutation refutes that action as represented; nothing here touches the Lovelock class or the potential axis, which are the two fallbacks written into the packet (a second-derivative scaling pass if the LP is infeasible; a scan of the (2,3) eigenvalue penalty if no crossing opens in D). Every gate carries a named mutation that must fail it, the lesson of W2. One auditor per rung before its verdict steers the next; numbers come with their box, stencil and seed maturity.

Jarek, on your offer to open §5 to §7 as R14+: this is that, run on our stack; your scripts, when posted, become the audit baseline for R14-0 and R14-A. The neutrino anchor and the atomic-physics direction stay out of this ladder by design.

----- reply 0 [12.r0] JarekDuda 2026-09-05T00:01:08Z -----

@xrodz @vantasnerdan @mjmikulski

Scripts and the full report: [GIST/REPO LINK]. Everything is sympy/numpy only; the README maps each script to the claim it checks. Since the 09-03 post, the same AI-assisted round ran its own adversarial pass and corrected several things that R14 would otherwise test in their superseded form. The corrections first, then pre-registered predictions for R14, then a floor result that bears on every dressed-core run.

## Corrections to the 09-03 post

1. **K_P must be written in H-adjoint form.** The plain-trace form tr(Ω_μΩ_ν) as posted is indefinite: off-vacuum its boost channels enter as −(λ₀−g)²(…) (negative in 340/2000 random off-vacuum probes). Use K_P = ½η^{μν} tr(Ω_μ H Ω_νᵀ H⁻¹), Ω_μ = P∂_μN P, H = η+2uuᵀ (0/2000 negative). The same trap applies to every f(N)-weighted term, so the R14-A basis B should be built in H-adjoint form throughout, or the LP will admit actions whose boost sectors are ghosts inside cores.
2. **Orbit theorem, correct scope.** R_G is a total derivative on the fixed-eigenvalue orbit for every G that is a covariant *function* of M (η, ηMη, M⁻¹, η+2uuᵀ, random polynomial f(N): spectral convergence to 1e−8 … 1e−11). A constant C not commuting with the eigenframe breaks it (−206, converged), but such a C is not a function of M. Bulk content needs eigenvalue gradients, as R14-0's mutant says.
3. **The Lovelock/εε fallback in the R14 packet is empty.** Every ghost-free second-derivative structure (all derivative indices on ε-tensors) vanishes identically on planar profiles for the same reason F does; the structures that do see the sheet ((□M)²-type) are Ostrogradsky. No compute should go there.
4. **The conjecture is a trilemma, and R14-A needs one more row.** Sheet–tail exponents: the fixed-J sheet needs a cost ≥ c·x at both small and large gradient (p = 1 exactly; p > 1 loses to shallow wide sheets ∝ L^{1−p}, p < 1 to steep ones ∝ h^{1−p}), while the tail needs p > 3/2. But among the six invariant analytic quadratic jet forms, the tail takes values (4,0,4,2,0,2)(λ−δ)²/r² and a twist sheet (2,0,0,0,0,0)k²(1−δ)², so Q_F = Σ_i tr(∂_iM∂_iM) − |∂_iM^{ic}|² — Frank elasticity with zero splay constant — vanishes on the single tail and charges every sheet orientation. Its pair energy is finite (leading cross term odd, integrates out) but not Coulomb: −101, −136, −164, −199 at d = 2, 3, 4, 6 (L = 24), a ~c_F/d force that overtakes 1/d² beyond d ~ K_C/(90 c_F). Consequence for R14-A: Q_F is not in B, and the packet's witness rows test tail finiteness, sheet cost, dressed-pair sign and positivity — Q_F passes all four. Unless a fifth row pins the pair law's d-dependence (the four-separation fit must stay 1/d), the LP will return CONE_FEASIBLE on Q_F and mislead. Add Q_F to B and add the Coulomb-form row; the prediction is then CLASS_INFEASIBLE.
5. **Free inertia survives the degenerate exterior.** On (−g,1,δ,δ) a (1,2) sheet gives I₁^{kin} = 2ω²(δ−1)⁴sin²ψ·ψ′² at zero static cost, because the sheet rotates the (2,3) clock generator out of the degenerate plane. The textured-medium exit is closed as well: a uniform background gradient b gives the sheet a p = 1 stiffness but confines a single hedgehog (cross energy ≈ 0.07·L at b = 0.1); same coefficient, both effects.

## Pre-registered predictions for R14

- **R14-0:** every listed statement confirms, with K_P blind to the R13-W sheet [T] and the K_P sign caveat above.
- **R14-A:** CONE_FEASIBLE via Q_F if the basis/rows stay as frozen; CLASS_INFEASIBLE with Q_F in B and the Coulomb-form row added.
- **R14-B:** CANDIDATE_REFUTED at every c ∈ {0.3, 1, 3}: the descent finds the (1,2) sheet (or its eigenvalue-zigzag sibling), invisible to K_P, with the same runaway as R13-W. The sheet has no conjugate momentum of its own, so this is variational, not dynamical — a Hamiltonian integrator would not see it.
- **R14-C, R_G:** sign of dE/dd follows sign(c_R); the static 3×3 record unchanged to machine precision; the pair law is d^{−2.01}, odd in the partner's dressing at leading order (charge–dipole), no 1/d; single-core energy ∝ c_R·m²·(deficit), even in m. R_G is a coupling, not a mediator.
- **R14-C, K_λ:** attractive Yukawa, range 1/m_s. Two things to read off while you are there: I₁ on the uniaxial core is ∝ (λ₁−δ)⁴ and independent of g, so the g-mode is unsourced except through dressing; and whether tr N is constant through the relaxed core decides if a stiff trace mediator is admissible (the hierarchy needs c ~ 10⁴¹E² in model units; a mode that stiff cannot be the melting direction).
- **R14-D:** no prediction; one warning below.

## A floor result that touches every dressed-core run

On the all-η certified action, a twist sheet ψ = kz on a boost-gradient background of rapidity slope b has U = −2b²k²g²(δ−1)² at leading order in g — bilinear in (b, k), so nothing bounds it; the H-adjoint contraction gives exactly +2b²k²g²(δ−1)². Confirmed at 64³ on a localised radially dressed core (rapidity 0.5, width 2): ΔE = −114, −388, −1091 for k = 0.5, 1, 2 (all-η) vs +183, +594, +1549 (H-adjoint). Reproduction note: the twist must be a texture of the spatial frame *inside* the dressed frame; twisting after dressing co-rotates the boost axis and hides the effect (+160, +636, +2473 in both forms). So every dressed-core quantity on L_cert — the R3 dressed pair, R4's wall-running minimizer, the R13-W zigzag, and R14-C/D as frozen — sits on an unbounded functional; this is R2's λ ≥ ½ made explicit. Suggest R14-C/D run on I₁^h (the static 3×3 record is identical there). It is also the channel of the M5.8 clock-fuel mechanism (−c₁b²k² + c₂b⁴k⁴ has a frequency-rigid minimum at (bk)² = c₁/2c₂), so the signed quartic of that line is bounded on this witness iff it carries b⁴k⁴.

## Two smaller items

- R² (any G) closes as a Newton mediator: on the hedgehog R_η = 2(δ−λ)(λ−δ+2λ′r)/r², linear in λ′, so R² gives the eigenvalue mode a 1/r²-weighted stiffness whose Green's function is C₁ + C₂r (Mannheim linear potential). I₁ itself does the same. Only a constant-stiffness term gives 1/d.
- Off-ladder, for the record: if flavour mixing is an SO(3) rotation of a real frame (MODELS.md's neutrino route), the PMNS matrix is real orthogonal, so J = 0 and δ_CP ∈ {0°, 180°} — a sharper falsifier than the single value 180°.

The report (15 pp., in the bundle) carries the verification ledger with [T]/[N]/[R] status per claim and the adversarial round (A1–A8) in full.

----- reply 1 [12.r1] JarekDuda 2026-09-05T01:27:54Z -----

I see no conflict between P250 and R13-W once each result is labelled by its action.

P250 constructs a finite-tension coexistence kink and a stationary thin-wall bag family in the C-M5C/P249 completion. Its exact phase-slip bookkeeping does not establish a global stationary interface between two freely chosen nonzero frequencies, and its fixed-frequency radial mode is a saddle. No full fixed-charge constrained minimum or full second-variation spectrum is claimed.

R13-W tests the certified M5.32 action, `L_cert = -4 I1 - V4`. For every one-coordinate planar profile, the commutator spatial term vanishes and only `V4` remains; the phase field is also flat on the uniform vacuum orbit. Accordingly, W1 finds tension converging to zero, W2 passes only because its slab observables are identities, and W3 finds no stationary fixed-J state: the shell melts while flank inertia inflates. This excludes the tested wall-clock mechanism on `L_cert`, not walls in other completions.

I suggest that R14 apply these gates before optimizing any wall profile:

1. **Exact clock charge.** State the compact generator, its period, and the boundary conditions. Verify that the complete action is cyclic under this transformation and derive its Noether current and charge. A prescribed `J` or masked generator is not enough.

2. **Healthy dynamics.** Form weighted contractions with the declared H-adjoint, then derive the complete constrained Legendre map, physical kinetic quadratic form, and principal symbol. Require positive kinetic energy and strong hyperbolicity after constraints on the vacuum, core, wall, and counterflow backgrounds. “H-adjoint” by itself is not a positivity result.

3. **Full fixed-J stability.** Find a stationary point in the full field space and test the constrained second variation of `E - omega J`, or the equivalent Routhian. It should be coercive modulo genuine translation, internal-symmetry, and gauge zero modes, including variations of the generator, field-dependent inertia, and orientation-changing sheet modes. This is the gate that excludes the R13-W inertia-inflation direction.

4. **Charge and asymptotics.** Add `Q_F` to the candidate basis, but do not identify a nonzero `Q_F` with electric charge or a Coulomb law. Require an explicit Gauss-law or surface-flux normalization, a `1/r` far potential in three spatial dimensions, a `1/d` pair energy with the intended sign, and finite converged total energy for the extended source. P250 should remain a comparator for a different completion.

The `R_G` statement should also remain orbit-relative: establish the boundary-term identity on the fixed-spectrum orbit used in the derivation, then recompute it when eigenvalues vary. If the epsilon-epsilon planar fallback vanishes identically, record that null result rather than assigning it a stabilizing role.

The current-wedge term supplies an early rejection test. On an aligned homogeneous two-clock background,

`g = Lambda r1^2 r2^2 (r1^2 mu1^2 + r2^2 mu2^2),    c_-^2 = 1 - g.`

For equal unit amplitudes and rates, `g = 2 lambda`. The point `g = 1` has a degenerate spatial principal part; `g > 1` is gradient-unstable. In the checked 1+1 counterflow `theta = t`, `phi = gamma(t - v x)`, real characteristics require `lambda <= 1/(1 + gamma)`; equality alone does not certify strong hyperbolicity. These formulas reject unrestricted large positive wedge alignment but do not certify other cores, nodes, walls, or counterflows.

Three currents must remain distinct: the defect/topological measure, the clock Noether current, and the emergent Dirac/Pauli matter current including its magnetization curl. Conservation of the clock current does not by itself move a defect zero. If one solution binds them, invert the complete reduced symplectic form and retain its reciprocal cross-blocks; negligible backreaction must follow from a norm hierarchy or on-shell cancellation.

For gravity, a static fit is insufficient: the linearized theory must contain the required massless `1/k^2` pole with positive physical residue and universal source coupling.

Finally, a real `3 x 3` orthogonal mixing matrix has `J_CP = 0`. If intrinsic Dirac-type leptonic oscillation CP is established, a purely real SO(3)-only mixing sector is incomplete and needs a physical complex phase or an equivalent CP-odd geometric or dynamical structure. Matter-induced neutrino/antineutrino asymmetry does not by itself establish this invariant.

The strongest R14 outcome would be either an admissible coefficient region with an explicit healthy witness action, or a clearly scoped no-go certificate.

===== COMMENT [13] xrodz 2026-09-05T12:11:33Z (top-level) =====

# R14 complete: the two-derivative class cannot carry a coexisting clock and Coulomb hedgehog (exact certificate), K_P^h fails as a clock because the certified vacuum itself ticks, and P250's exterior-at-rest structure appears on the 4x4 field only with an explicit split stiffness on a modified potential

## TL;DR (30 seconds)

The R14 packet (ledger § 6.3) ran as one autonomous session with an independent audit on every rung (56 claims: 31 confirmed, 17 qualified, 8 refuted, all applied). The coexistence conjecture is tested as a linear program over the whole basis on frozen rows measured from saved fields, K_P^h is continued at fixed J, R_G and K_lambda are read as Newton mediators on the certified pairs, and the P250 bridge is built on the 4x4 field. Everything is on `main` at `0e487065`, links in the record line below.

## The five questions the packet asked, answered in one line each:

1. **Does the author's 09-03 algebra hold on our stack?** Yes except one statement: the R_G orbit theorem fails on two-plane boost textures for the three M-dependent G (nonzero, spectrally converged); the Goldstone count, K_lambda inertness, the K_P channel table and the H-adjoint correction all hold, and the certified vacuum TICKS under the local clock generator (K_P inertia 194.4 per unit volume).
2. **Can any action in the basis carry a Coulomb hedgehog and a clock together?** The two-derivative class cannot: CLASS_INFEASIBLE with an exact rational certificate on two independent assemblies; the full basis with R1's F-built terms has no witness below coefficient norm 100, and its only corner above (I6 at norm 200) runs away when the field re-relaxes.
3. **Is K_P^h a clock on the certified vacuum?** No: CANDIDATE_REFUTED at c = 1 and 3, no stationary state, and omega = J / (2 kin) set by an inertia extensive in the volume (omega ~ J / L^3), because the exterior ticks; the sheet mechanism the author predicted is never reached, the exterior's inertia removes the clock first.
4. **Do R_G or K_lambda give Newton?** Not as stated: R_G's pair slope follows sign(c_R) only above a g-dependent threshold, changes the static 3x3 record, and has no pair law (a boundary-flux term of the saturated far-field boost); K_lambda is an attractive Yukawa only with a light eigenvalue mass that V4 does not supply.
5. **Does P250's exterior at rest with a rotating interior exist on the 4x4 field?** Not on `L_cert + c K_P^h` (the fixed-omega functional is unbounded along the split); yes on `V4 + mu (m2 - m3)^2` for mu >= 5.6e-4, a first-order wall of tension 4.03 / 0.63 (mu 1e-2 / 1e-3, two independent methods), 3500 to 7700 box units wide, so nothing of it fits a certified box.

Record: the R14 ladder ran (2026-09-04 23:42 to 2026-09-05, one autonomous session, every rung independently audited: 56 claims, 31 confirmed, 17 qualified, 8 refuted, all applied, three instrument defects of ours found and fixed by the audits); the results below are on `main` at [`0e487065`](https://github.com/openwave-labs/openwave/commit/0e487065) (PR #521): the task record [`tasks/m5_32_task_details.md`](https://github.com/openwave-labs/openwave/blob/0e487065/openwave/xperiments/m5_liquid_crystal/research/tasks/m5_32_task_details.md) (the R14 section, every number, the six audit reports applied), the scripts `m5_32_r14_*.py`, the term catalog [`findings/m5_32_term_catalog.md`](https://github.com/openwave-labs/openwave/blob/0e487065/openwave/xperiments/m5_liquid_crystal/research/findings/m5_32_term_catalog.md), the ledger [`findings/m5_32_candidate_ledger.md`](https://github.com/openwave-labs/openwave/blob/0e487065/openwave/xperiments/m5_liquid_crystal/research/findings/m5_32_candidate_ledger.md) § 6.3. Nothing here needs a reply beyond what you find wrong. Jarek, your scripts link on the 09-05 replies is still the literal placeholder `[GIST/REPO LINK]`, so no external script entered any rung; all cross-checks are against your written statements.

## R14-0, your 09-03 algebra on our stack (audited 10 / 4 / 0)

| Statement | Verdict | What we measured |
| --- | --- | --- |
| P249's radiation edge is entirely the axis lock; five Goldstones on the degenerate 4x4 vacuum with clock charges (0,1,1,1,1), the split doublet charge 2 | CONFIRMED (exact) | the Hessian split (5,6,6,0,0,6) / (0,4,4,4,4,4) / (0,0,12,0,0,12); a strict conjugation-invariant potential has exactly 5 flat directions there. One property of ours: a V4-type potential (a sum of trace squares) is quartically soft at a degenerate spectrum, 7 flat at quadratic order, so on `L_cert`-type potentials the charge-2 split has no quadratic mass |
| `R_G` is a total derivative on the whole fixed-eigenvalue orbit, boosts included, for every covariant function G | REFUTED for boosts | zero to 1e-15 on rotation orbits and on single-plane boost textures for all four G, but NONZERO and spectrally converged on textures with two or more boost planes for G = eta M eta, M^-1 and h_cov (e.g. -7.68, -10.1, -1.90 on a (x,y) box at N = 64). The delta pairing of a derivative index with an internal index is invariant uame changes; only R_eta (EL == 0) is empty everywhere. R_G has no omega^2 content(confirmed), and bulk content with eigenvalue gradients (confirmed) |
| K_lambda inert on orientation gradients and on every clock channel | CONFIRMED |rbit fields and on all seven generator channels of the relaxed hedgehog; its static energy is core-local (L-exponent 0.26) |
| K_P blind to tilts and boosts, phase stiffness (lambda_2 - lambda_3)^2 on the (2 the intended roots | zero on the three boosts and two tilts, [f(delta) f(0)]^2delta^2 = 194.4 on the clock; the literal polynomial (N - g)(N - 1) in our sign convention (spectrum (-g, 1, delta, 0)) is not blind to boosts, -8.5e7 |
| K_P does not charge the R13-W sheet | CONFIRMED, plus an addition | continuum stthe (1,2) twist (the lattice value is an h^2 artifact of finite rotations); thefree inertia is untouched. Addition: K_P DOES charge the eigenvalue zigzag W3 measured (75 to 97 percent of its static energy on the layers) |
| your 09-05 correction: K_P must be H-adjoint | CONFIRMED independently before yoplain trace is indefinite off the vacuum spectrum (the timelike row leaks in withsigma = -1). Our audit adds: the invariant order is tr(Om^T H Om H^-1); the transposed placement, as written in both our texts, changes by a factor 20 under a boost. Also: G = M^-1 is
undefined on the certified vacuum (eigenvalue 0) |
| the W0 theorems per entrant | new | planar flatness survives R_G and is overturned by K_P for eigenvalue walls; phase flatness is overturned by K_P; the R13-W FREE INERTIA (the (1,2) twist
at zero static cost) survives K_P, R_G and K_lambda alike: no fixed-J minimizer on, by the R13-W theorem; inside the basis only the C6 quartic charges the twist |
| the vacuum ticks | new, the number that decides B and D | on the certified vacuum the local clock generator is nonzero, so K_P's inertia is 194.4 per unit volume against a V4 cost of 1.8e-6
for the degenerate state: the exterior at rest of P250 does not exist on `L_cert +energy on the hedgehog tail is L-divergent (exponent 1.34), the (2,3)-frameconnection of the hedgehog is charged |

## R14-A, the conjecture as a linear program (audited 4 / 4 / 1)

Basis: R1's thirteen (the parity-odd triple is null on every row), K_T, K_lambda, R_{eta M eta}, R_hcov, K_P^h, the four covariant constant-coefficient quadratic jet forms T1..T4 (your Q_F =
T1 - T3 is in the span, added on your 09-05 request), the C5/C6 quartics; `chat_I1sured on saved fields: the hedgehog tail plateau on two relaxed fields (acancellation of L-divergent tails must hold on every field; the first refined vertex cancelled two tails of 277 to 0.3 by a ratio tuned to one field), the (1,2) twist sheet and the (2,3)
zigzag sheet, the dressed pair on the R3 ansatz and on the relaxed pairs, the certepulsion and its 1/d FORM (your fifth row), fifteen omega^2 positivity channels,and UV boundedness: plane-wave averages, crossed-wave averages (a single plane wave has F = 0 identically, so single waves are blind to every curvature-built term), the exact quartic forms on
random jets, and the EXACT quadratic form of every two-derivative element in sevenby cutting planes. Two solvers, the certificate re-checked in exact rationalarithmetic.

| Class | Verdict |
| --- | --- |
| the two-derivative class K_T, K_lambda, R_{eta M eta}, R_hcov, K_P^h, T1..T4, with or without the quartics | **CLASS_INFEASIBLE, exact certificate** (producer and auditor, independent
assemblies). The binding structure is the like-charge 1/d form against the two taior the relaxed pair; your prediction (CLASS_INFEASIBLE with Q_F in B and theCoulomb-form row) is confirmed for this class |
| the full basis with R1's F-built thirteen | no witness below coefficient norm 10e it only corners at norm 200 to 675 carried by I6 or I1_h at forty to a hundredtimes the certified I1, where the linear-response rows are not valid. The I6 corner was re-relaxed (the outer loop you asked for): it runs away within fifty iterations, a localized UV blow-up
at the hedgehog core (I6 = R^2 at minus forty-four times I1; R1's Coulomb gate hadIFIED for the full basis |
| the binding row | the zigzag sheet: without it a point exists at norm 1.4 (14.7 with the crossed-wave rows) |
| a fact the audit found in the certified action | the certified 4 I1 has a NEGATIthe hedgehog's boost tangents (-0.22, -0.16), so the certified Lagrangian alonefails the positivity rows and every feasible point carries K_T >= 0.064 to repair it |

Not computed, in your 09-05 gate language: the Noether charge of an exact clock symmetry, the principal symbol and hyperbolicity, the constrained second variation. The cone statement is
"feasible on the frozen rows", not "healthy".

## R14-B, the fixed-J continuation on `L_cert + c K_P^h` (audited 4 / 3 / 1, plus

CANDIDATE_REFUTED at c = 1 and 3 for J in {50, 200, 800} across h in {1.5, 1.0, 0.no stationary state (every descent ends on a logarithmic plateau with the FIRE step collapsed), the fixed-J term numerically invisible (1e-9 of the force), and the frequency `omega = J / (2 kin)` set by an inertia extensive in the volume: at fixed L = 48 the three spacings
give kin_KP = 1.67e7, 1.51e7, 1.54e7 and omega = 6.0e-6, 6.6e-6, 6.5e-6, while L =; the exterior ticks at 0.63 to 0.78 of the vacuum inertia per unit volume, soomega ~ J / L^3. Your prediction holds in the verdict and not in the mechanism: the descent never reaches a sheet regime, the exterior's inertia removes the clock first (the sheet remains the
theorem-level reason). Our pre-registered mechanism (full closure of the (2,3) gapdred iterations (gap 0.30 to 0.24, K_P^h cost down 40 to 60 percent) and stalled.Controls: a boost-dressed seed (rapidity 0.1) at c = 0.3 and 1 shows no boost runaway in 3000 iterations, the dressing decays 1.8 percent (the audit had computed a boost-sector saddle at c =
0.3 on the unsampled subspace); the P250 structure imposed by hand (the exterior p pair, the core kept) does not persist in either convention: the descentdegenerates the core too within a hundred iterations, and in the true gradient flow of E_J (generator frozen) the fixed-J reward is 0.01 against a K_P^h static cost of 9180 for keeping a
split at the core; a g = 32 control and a 6000-iteration continuation are in the r

## R14-C, the scalar-channel Newton route (audited 1 / 4 / 4; an instrument defectt and fixed, the heals rerun)

| Your statement | Verdict | Measured (linear response on the R3 relaxed pairs, wh
| --- | --- | --- |
| R_G on the dressed pair: the sign of dE/dd follows sign(c_R) | QUALIFIED | the s(R_G slope): -882 + 2058 c_R at lambda = 0, -2316 + 2058 c_R at lambda = 1 (I1^h),so the sign follows c_R only above c_R = 0.43 at lambda 0 and never within \|c_R\| <= 1 at lambda 1; the R_G slope scales with g (2058 = 32 x 64.4 + 10), so at g = 8 the threshold is about
1.7 |
| the static 3x3 record unchanged to machine precision | REFUTED for eta M eta | on the certified like-charge pairs R_{eta M eta}'s E(12) - E(24) is -5.54 per unit coefficient against the
certified +0.28; only R_eta leaves the record untouched, and R_eta is empty |
| the pair law d^-2 (charge-dipole) | REFUTED as a pair law | on the ansatz the R_G slope dE/dd GROWS with d (62 to 140 per unit from d = 8 to 24) and E(d) never converges: on a boost orbit
R_hcov = -R_eta to 0.07 percent and R_{eta M eta} / R_hcov = g at every d, so the G is a boundary-flux term of the saturated far-field boost, not an interaction |
| K_lambda with a light scale: an attractive Yukawa of range 1/m_s | CONFIRMED as a model statement, QUALIFIED as physics | the eigenvalue deficits of the relaxed hedgehog are core-local
(r^-3.3 to r^-4.0; lambda_2 changes sign at r = 13.5); the linearized exchange is d lambda_3 at any assumed m_s; but V4 fixes the eigenvalue masses, and on thecertified action K_lambda's pair energy is a core overlap (exponent 6). A long-range eigenvalue exchange needs a modified potential |
| whether tr N is constant through the relaxed core | measured | it is not: the tr (one percent of tr N) and grows with relaxation |

The heals themselves do not move the stiff core within the R3 budget (fields move ry C verdict is a linear-response verdict, the limitation R3 recorded.

## R14-D and D2, the P250 bridge on the 4x4 field (audited 5 / 1 / 2, then 7 / 1 /

On `L_cert + c K_P^h` the exterior ticks and the fixed-omega functional is unboundas degree 10 in the eigenvalues against V4's 8; sealed behind eigenvaluecollisions, a far-split metastable pocket appears at omega 1.0e-3), so an exterior at rest coexisting with a rotating interior does not exist with the certified potential. On the C3-modified
potential `V4 + mu (m2 - m3)^2` it DOES exist for mu >= 5.6e-4: in the full (m2, mroze the split line and saw only a continuous onset; the audit found thefirst-order crossing off that line) an exterior at rest at the diagonal minimum 0.157 coexists with a rotating interior across a first-order wall. We then built the wall: on the reduced 1D
functional (exact on planar diagonal profiles, cross-checked on a lattice slab to is 4.026 at mu = 1e-2 and 0.631 at mu = 1e-3, equal to the audit's path-optimizedBogomolny values (4.03, 0.63) by an independent method, with the thin-wall bag law R = 2 sigma / p at 1.03 omega_* giving R = 1.1e4 and 1.3e4. The scale is the caveat: the walls are 3500 to
7700 box units wide because the K_P^h kinetic metric f^4 is of order 1e3 while the so nothing of it fits a certified box, and all of it lives on a modifiedpotential: P250's structure on the 4x4 field is reachable exactly by the ingredient your section 1 identified as the whole gap, an explicit split stiffness at the degenerate point, and by
nothing else we tried.

## What closes, what opens

| Closed on our side | Open |
| --- | --- |
| the two-derivative class as a coexistence witness (certificate); K_P^h as a cloc(the exterior ticks); R_G as a Coulomb-compatible Newton mediator (threshold,g-dependent, no pair law); R_G's orbit theorem for boost textures; the plain K_P; the I6 corner | the author's gates 1 to 3 (exact Noether clock, principal symbol, constrained second
variation) on any candidate; the full-basis corners above norm 100 with fields re-ice bag on the modified potential (needs boxes of 1e4); the Lovelock class(dropped: every ghost-free epsilon-epsilon structure vanishes on planar profiles, as you said); the neutrino and atomic-physics directions (not staged) |

Everything above is in the task record with the six audit scripts and their reports; the term catalog (every basis element with its measured tail, sheet, pair, Coulomb, positivity and UV-form entries) is the file linked in the record line above. Jarek, the gist link when yors are the ones to compare first, especially the boost-orbit integrals of R_G andthe K_P vacuum channel table.

===== COMMENT [15] xrodz 2026-09-06T01:40:16Z (top-level) =====

# R15 ran: the floor witness holds on the vacuum, the tilt channel is exact, the projector object is admissible with a continuous onset and no coexistence wall

## TL;DR (30 seconds)

The R15 ladder pre-registered from the 09-05 reply ran overnight on our stack (2026-09-05 20:18 UTC to 2026-09-06 01:20 UTC): the floor witness V1 as jets and as the 64^3 lattice profile, the tilt channel symbolically, the new object `L = -4 I1^h - [V4(g,1,delta,delta) + mu (lambda_2 - lambda_3)^2] + c_P K_P^23` relaxed on two boxes, its four predictions each against a pre-registered gate, and one independent adversarial audit per rung (five audits, 28 claims: 18 confirmed, 10 qualified, 0 refuted). Everything below is script-backed; the record, scripts and data are in the commit linked at the end. Two definitions had to be chosen to run at all; both are asked back at the bottom.

## The four predictions and two claims, answered in one line each

1. **(i) exterior inertia exactly zero.** Holds by construction and on the lattice: the pair stays degenerate on every relaxed hedgehog (max split 0.0135 at `c_P = 1`), and `kin_KP23` equals the split integral identically.
2. **(ii) the hedgehog tail is finite (L-exponent 0, not 1.34).** Holds: `K_P^23` density `~ r^-4.1` on both boxes, L-exponent 0.08 to 0.11 (the `1/L` correction of an `r^-4` density); the audit notes the tail is the seed ansatz's, the descent reaches r < 15 only.
3. **(iii) a wall of width `(c_P/mu)^(1/2)`.** Fails as a coexistence object: `V4^dd >= 0` with equality only on the vacuum spectrum, so no Maxwell crossing exists below `omega_c^2 = mu / c_P` and the exterior is unstable above it, on all nine (mu, c_P) points (a theorem, audited). The only `sqrt(c_P/mu)` length is the static split decay `(1/2) sqrt(c_P/mu)`, and the relaxed hedgehog's split decays over 2.1 to 2.4 box units regardless of `c_P`.
4. **(iv) fixed-J descent reaches the (1,2) sheet, no minimizer.** No minimizer, for a reason neither side predicted: at J = 200 the descent does not go to the (1,2) sheet, it inflates the split in the innermost cells to buy inertia (`E_J` 3.7e5 to 89 in 600 iterations, the fixed-J term 43 of 89) by raising the pair's upper eigenvalue until it MEETS `lambda_1`, and then pins itself on that `lambda_1 = lambda_3` crossing, the branch cut of the ordered-label `P23` and of the leading-eigenvector clock (the audit: the true `E_J` climbs a staircase as each cell crosses; a label-free reading carries a tenth of the reported inertia). The n48 L72 box replicates it (`E_J` 98.7, the same six crossing cells at r = 1.3). Symbolically the (1,2) sheet is a free direction of `K_P^23` (static cost exactly zero at any amplitude), so by the R13-W theorem no fixed-J minimizer exists along it either.
5. **V1, the floor witness.** Confirmed on the vacuum with the exact coefficient: twist inside a boost dressing, `U_eta = -8 (delta - 1)^2 (g + d_a)^2 b^2 k^2` and `U_h = -U_eta` (your `-2 g^2 (delta - 1)^2` is the large-g form at ratio exactly 4), rapidity-independent; lattice n64 L48 with your profile: `DeltaE_eta = -515 / -1739 / -3583`, `DeltaE_h = +557 / +1835 / +4002` at k = 0.5 / 1 / 2 (your signs and growth; ours carries the certified 4). Twist after: positive in both forms but not equal (h about 10 x eta). On the relaxed hedgehog BOTH forms go negative: the repair is a vacuum-frame statement.
6. **The tilt channel.** Exact: no `theta_t^2` from any curvature term or from `K_P^23`; `L_2(-4 I1) = 32 omega^2 s^2 (delta + s - 1)^2 theta_z^2` (your `4 k^2 omega^2 s^2 (delta+s-1)^2` up to a factor 2 in the F normalization); the regulator gives `alpha = w kappa_2`, `kappa_2 = 2 (delta + s - 1)^2`, and the channel is hyperbolic iff `w > 16 omega^2 s^2` (`kappa_2` multiplies both sides and cancels).

Record: task record R15 section, ledger 6.4 outcome, method note 12, the six scripts, the five audit scripts and JSONs: commit `62e75650` on `main` (https://github.com/openwave-labs/openwave/commit/62e756509a5c24df90f5ac3c88e9e1a9986beffd, PR #522).

## What ran, in tables

### R15-V: the floor witness

| Read | eta form (`E = 4 I1`) | H-adjoint form (`E = 4 I1^h`) |
| --- | --- | --- |
| jet coefficient `c` (twist inside, `U = b^2 k^2 c`), boost along axis 1 / 2 / 3 | -317.5 / -270.0 / 0 | +317.5 / +270.0 / 0 |
| general law (audit) | `-8 (d_i - d_j)^2 (D_00 + d_a)^2` | the exact negative |
| twist after, rapidity 0.5, b = k = 1 | +11718 | +12525 |
| lattice n64 L48, `DeltaE(k)` twist inside, k 0.5 / 1 / 2 | -515 / -1739 / -3583 | +557 / +1835 / +4002 |
| lattice, twist after | +845 / +4952 / +34656 | +9880 / +39771 / +153392 |
| relaxed `L_cert` hedgehog, cross terms | -36 / -109 / -188 | -183 / -611 / -1696 |

Grid caveat (audit): the dressed baseline diverges with the grid (13023 / 57659 / 108290 at spacing 1.5 / 0.75 / 0.5: the radial unit vector is discontinuous at the origin) and `DeltaE(k = 1)` is not converged (-706 / -1739 / -2085); the sub-quadratic k growth in both your numbers and ours is stencil attenuation. Signs and ordering are stable across all of it. Your h column does not match ours by one normalization (eta fits 4.5, h fits 3.0 to 3.1 against a common jet-level 4), so your h numbers are not this `I1_h` up to a constant.

### R15-H: the tilt channel (`M = R_23(omega t) R_12(theta(t,z)) D_s R_12^T R_23^T`, `D_s = diag(g, 1, delta+s, delta-s)`)

| Term | `L_0` | `theta_t^2` | `theta_z^2` | `theta^2` |
| --- | --- | --- | --- | --- |
| `-4 I1` and `-4 I1^h` | 0 | 0 | `+32 omega^2 s^2 (delta+s-1)^2` | 0 |
| `K_P^23` | `4 c_P omega^2 s^2` | 0 | 0 | `-4 c_P omega^2 s^2` |
| regulator `w [tr(A_0 G A_0 G) - tr(A_z G A_z G)]`, G = eta or h | `8 w omega^2 s^2` | `+2 w (delta+s-1)^2` | `-2 w (delta+s-1)^2` | `-2 w omega^2 (3s+1-delta)(delta+s-1)` |

Found by the audit: the regulator's `theta^2` term survives at s = 0, so the one-tilt ansatz drops the Coriolis coupling to the (1,3) partner (the k = 0 gap is ansatz-dependent, the hyperbolicity condition is not); a (2,3) twist sheet is NOT free for `K_P^23` (`-4 c_P s^2 psi'^2`).

### R15-M: the object on the lattice (n32 L48 and n48 L72, spacing 1.5, mu in {0, 1e-2}, c_P in {0, 1}, 3000 FIRE iterations from the uniaxial radial hedgehog)

| Read | Result |
| --- | --- |
| Hessian of `V4^dd + mu SPLIT` at the vacuum | null count 7 at mu = 0 (five orbit directions + the quartic-flat (2,3) block), 5 for mu > 0; split stiffness exactly `4 mu` |
| relaxed hedgehog | finite energy, exterior on the vacuum (drift <= 8e-4), shell energy `~ r^-2` (density `~ r^-4`), `E_u` 5.4 / 5.6 and `K_P^23` 2.8 / 3.0 at `c_P = 1` (n32 / n48); a finite-energy hedgehog with a `1/R` energy tail |
| the pair | max split 0.0135 (`c_P = 1`), 0.075 (`c_P = 0`); mu nearly irrelevant |
| against the certified `L_cert` hedgehog | its slower decay (outer slope -0.6 against -2.0) is the z-axis line of the singular seed frame, invisible on the degenerate vacuum (audit) |

### R15-P-iii: the reduced planar functional `F = int { (c/2)(m2'^2 + m3'^2) + V4^dd + mu s^2 - omega^2 [c s^2 + 8 s^2 s'^2] }`

| Question | Answer |
| --- | --- |
| Maxwell crossing below `omega_c^2 = mu/c`? | none, a theorem: `V_eff = V4^dd + (mu - omega^2 c) s^2 > 0` off the vacuum |
| at onset | continuous; the split jumps steeply (0.135 / 0.452 / 1.418 at 1.01 `omega_c^2` for mu 1e-3 / 1e-2 / 1e-1; `V4^dd` is quartic-flat along the split) |
| the "Ising wall" between (a, b) and (b, a) | a Goldstone twist by the (2,3) rotation, tension `-> 0` on the infinite line (audit): not an object |
| the fixed-omega functional | unbounded below wherever `abs(s) > sqrt(c / (32 omega^2))` (the negative I1 flank inertia) |
| reduced densities vs the lattice slab | exact to 1e-14 (`E_u = 0` on planar profiles, `kappa_P = 1`) |

### R15-P-iv: fixed J = 200 on `L_P` (`c_P = 1`, mu = 1e-2)

| Read | seed (the relaxed hedgehog) | end, n32 L48 (3000 it) | end, n48 L72 (3000 it) |
| --- | --- | --- | --- |
| `E_J` | 3.7e5 | 88.96 | 98.72 |
| `E_stat` (`E_u`, `K_P^23`, mu SPLIT) | 8.34 (5.36, 2.83, 2e-4) | 45.92 (22.15, 23.47, 0.24) | 51.96 (27.11, 24.54, 0.24) |
| `kin_tot` (I1, `K_P^23`), omega | 0.027 (0.004, 0.023) | 232 (208, 24), 0.430 | 214 (189, 24), 0.468 |
| max split | 0.0135 | 0.82 | 0.77 |
| stationary? | | no: `dE_J/ds = -8.6e3` along the frozen gradient; the end state sits on the `lambda_1 = lambda_3` crossing in the six innermost cells (core spectrum (0.42, 0.42, 0.48) to (-0.11, 0.70, 0.70)); FIRE dt 1e-10 from iteration 1600 | no: the same six cells on the crossing (gap 3.6e-6) |

## Two definitions we had to choose (please confirm or correct)

1. **`P23` off the degenerate point.** Your `(N - g)(N - 1) / [(lambda_23 - g)(lambda_23 - 1)]` is a projector only at `lambda_2 = lambda_3`, and the interior of the object is split by construction. We ran `P23 = I - P_g - P_1` with the Lagrange projectors of the two isolated eigenvalues (equal to yours at degeneracy, a projector everywhere, smooth gradient through the Cayley-Hamilton resolvents). This reading has a seam: when the pair's upper member reaches `lambda_1` the labels swap and `P23` jumps, and the fixed-J descent found exactly that seam. What is `P23` there, and is the clock still the (2,3) rotation?
2. **The split term.** We ran the invariant `mu (lambda_2 - lambda_3)^2` (the eigenvalue split). The audit shows the alternative completion `mu (M_22 - M_33)^2` on the diagonal entries has a lower uniform minimum with `M_22 = M_33` and a nonzero off-diagonal at every `omega > 0`, i.e. no exterior at rest at all. Which completion is the object?

Also asked, third time and in the body: the scripts and the 15-page report behind your numbers (the gist link in your reply is still the placeholder). With them we can put your h column on the same normalization as ours; without them the factor-of-4-vs-3 discrepancy stays open.

Two remarks for your side. First, on (iv): with the six crossing cells masked the end state is near-stationary, so the object as we read it has no fixed-J minimizer away from the seam and a staircase on it; a `P23` defined across the crossing is what would make the question well posed. Second, the H-adjoint sign flip is a vacuum-frame statement in our runs: on the relaxed hedgehog the eta and h cross terms are both negative, so `-4 I1^h` is not bounded below on that background either.


===== COMMENT [18] xrodz 2026-09-07T02:35:02Z (top-level) =====

# R16 on v4: the statics go uniaxial, the clock has no bound doublet, fixed K builds a lattice spike, and your tilt inequality is the mechanism that breaks it

## TL;DR

1. **Retraction first.** Our R15 sentence "the H-adjoint quartic lacks a floor on the hedgehog" was wrong: `E_h >= 0` pointwise for both completions, the negative cross terms mean the seed is not stationary along twist-inside-dressing. Adopted.
2. **Naming.** Your posted h column is `I_rebuild / 4` (`F^G = A G A' - A' G A`); ours was `I_norm` (the registry's `I1_h`, `F^eta` in the G norm). On every field of this run (no boosts, `u = e_0`) the two coincide exactly, so one relaxation per box.
3. **Two normalizations and one repair.** `omega_c^2 = mu / (4 c_P)` in the clock-rate convention (`Omega_c^2 = mu / c_P` for the doublet frequency); your "min 0.01117" is the weighted condition's value at `s*`, its minimum is `mu`; and `rho^2 E2` is not circle-invariant, so the regulator is averaged too.
4. **R16-1 statics: `UNIAXIAL_RADIAL` on n32 L48, n48 L72 and n64 L48 (h 0.75).** The R15-M seed's residual split shrinks tenfold, `beta^2` from 0.17 to 4e-4 (2.5e-8 at h 0.75), the exterior a regular `r^-4` texture; the radial hedgehog has Morse index 0 in the split sector. No biaxial torus, no split core.
5. **R16-2 clock operator: `NO_BOUND_MODE`.** The core's lowest doublet modes (`Omega^2` 0.0448) sit above the empty box's bottom (0.0256) and above `mu / c_P = 0.01`; no localized doublet direction lowers the energy. The core repels the doublet.
6. **R16-3 fixed K: `CANDIDATE_REFUTED` at n32 K 50 (escape d), `NUMERICALLY_UNRESOLVED` elsewhere.** Every descent buys its inertia by inflating the split in a handful of cells (a lattice spike at r 4.9 on n32, r 3.3 on n48), 3.5x to 6.6x above `omega_c K`, never a relative equilibrium; the inertia the spike builds is the quartic's, not `c_P K_P^proj`.
7. **R16-4 principal symbol: `HYPERBOLIC` in every channel on the cores at every `omega` to 0.25; the spikes flip `omega`-driven.** The symbol's `omega` term is `32 omega^2 rho^2`: invisible on the cores (`rho^2` 5e-5), decisive on the spikes (`rho^2` 0.3 to 0.4) with per-channel thresholds 0.16 to 0.19, your `c_s > 16 omega^2` generalized. The hedgehog exterior is not degenerate in the tilt channel (the quartic's inertia through the gradient tail).
8. **Audits.** R16-1 4/3/0, R16-2 1/2/3 (the auditor caught a factor 2 in our first doublet frequencies, corrected and re-run; the verdicts survived), R16-3 4/3/0, R16-4 2/2/1 (the auditor corrected our attribution of the spikes' negative stiffness: kinetic, not static). Everything below is script-backed and the scripts are linked.

## The four stage questions answered in one line each

| Stage | Your prediction | Verdict |
| --- | --- | --- |
| 1 statics | biaxial torus or split core at index 0, the radial hedgehog a transition state | `UNIAXIAL_RADIAL` on three boxes; the radial hedgehog index 0 |
| 2 clock operator | a core-bound doublet below `omega_c` | `NO_BOUND_MODE` (0.0448 above the box bottom 0.0256 and above 0.01) |
| 3 fixed K | a relative equilibrium, `E(K) < omega_c K`, none of (a) to (c) | escape (d) at K 50 n32; a lattice-scale split spike 3.5x to 6.6x above the bound on every run; not stationary |
| 4 principal symbol | tilt hyperbolic iff `c_s > 16 omega^2`; the exterior degenerate | all channels hyperbolic on the cores (the `rho^2`-proportional term is invisible there); the spikes flip at `omega*` 0.16 to 0.19 by exactly that term; the hedgehog exterior not degenerate |

## 1. The object as run (equations first)

Our two readings of your packet stayed: `Pi = P23`, the director from `P_1`; everything else is yours. E-orientation throughout (static `+`, inertia `+ omega^2`, your `-c K` convention).

| Piece | As realized |
| --- | --- |
| the frame | `u u^T = -P_g eta`, `n n^T = P_1 eta` (the director lifted outward and propagated step to step), `G = eta (I - 2 P_g)`, `J^a_b = eta^aa eps_abcd u^c n^d` (`J^2 = -P23`), `T_alpha M = R(alpha / 2) M R(alpha / 2)^T`, `a0 = J M + M J^T` |
| the plateau weight | `w(N) = I - (1 - w(lambda_g)) P_g - (1 - w(lambda_1)) P_1` in the run's spectral domain (label-free where the pair leaves the plateau: a full eigendecomposition in the G metric with the Daleckii-Krein derivative) |
| the action | `E_stat = 4 sum_{i<j} tr(G F_ij G F_ij^T) + V4^dd + mu rho^2 + c_P K_P^proj + c_s rho^2 E2`, `kin_tot = 4 sum_i tr(G F_0i G F_0i^T) + c_P kin_KP + c_s rho^2 tr(a0 G a0 G)`; `F` with `G` (`I_rebuild`) or `eta` (`I_norm`); `rho^2 = (s^2 - 4p) / 4` |
| the circle average | `E_v4 = (1 / n_s) sum_k E[T_(2 pi k / n_s) M]`; on the LATTICE the density has trigonometric degree 4 in `alpha` (the finite difference of `R(x) M R(x)^T` carries `R(x)^T R(x + h)`), so 8 samples are exact (every read and gate), 4 in the descents (an `O(h^2)`-level defect, 1e-9 on `E_stat`) |
| fixed K | `E_K = E_stat + K^2 / (4 kin_tot)`, `a0` refreshed each step and frozen in the gradient (R15's protocol; the true directional derivative read at the end) |
| the clock operator | `H zeta = Omega^2 (2 T) zeta` in the doublet subspace, `T` the per-cell 2 x 2 inertia (pointwise), Lanczos on `T^(-1/2) H T^(-1/2)`, `Omega = 2 omega` |
| the principal symbol | `sigma(Omega, k) = H_00 Omega^2 + 2 Omega H_0i k_i + H_ij k_i k_j`, `H` the second derivative of the Lagrangian density in the jets, circle-averaged with the perturbation co-rotated |

Two facts of the object surfaced by the instrument, stated as such: the hedgehog's isotropic center puts the director INSIDE your plateau (`lambda_1` 0.478 at the R15 seed's center, 0.57 after R16-1 on n32, 0.72 at h 0.75), so `w(N)` adds the director to the clock block within r 3.3 (0.65 at h 0.75) on every field of this family; and the outward director lift, smooth on every hedgehog, flips across x = 0 on a uniform director (our first vacuum control was contaminated by it and re-run with the uniform lift).

## 2. Equation-to-code map (SHA-pinned)

| Equation | Code |
| --- | --- |
| the frame, the weight and its derivative, every adjoint, the circle sampler and its adjoint, fixed K, FIRE, the 43 selftest gates | [`m5_32_r16_common.py`](https://github.com/openwave-labs/openwave/blob/deff0985a60a54d81fc2baa6e271caa0c3446f99/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r16_common.py) |
| R16-1 statics, texture reads, gates | [`m5_32_r16_1_statics.py`](https://github.com/openwave-labs/openwave/blob/deff0985a60a54d81fc2baa6e271caa0c3446f99/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r16_1_statics.py) |
| R16-2 the doublet operator | [`m5_32_r16_2_operator.py`](https://github.com/openwave-labs/openwave/blob/deff0985a60a54d81fc2baa6e271caa0c3446f99/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r16_2_operator.py) |
| R16-3 fixed K, the escapes, the stationarity reads | [`m5_32_r16_3_fixedk.py`](https://github.com/openwave-labs/openwave/blob/deff0985a60a54d81fc2baa6e271caa0c3446f99/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r16_3_fixedk.py) |
| R16-4 the symbol | [`m5_32_r16_4_symbol.py`](https://github.com/openwave-labs/openwave/blob/deff0985a60a54d81fc2baa6e271caa0c3446f99/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r16_4_symbol.py) |
| the four independent audits | `m5_32_r16_1_audit.py` to `m5_32_r16_4_audit.py`, JSONs in `data/` |
| the records | the task record `tasks/m5_32_task_details.md` (the R16 section), the method note `findings/m5_32_method_note.md` (section 14), the ledger `findings/m5_32_candidate_ledger.md` (section 6.5) |

## 3. R16-1 statics

| Box (h) | Iterations | `E_stat` seed -> end | End texture | Verdict |
| --- | --- | --- | --- | --- |
| n32 L48 (1.5) | 3000 | 20.04 -> 13.82 (`K_P` 14.5 -> 8.3, `E_h` 5.36 -> 5.43) | max `beta^2` 4e-4 (seed 0.17), max half split 5.6e-4 (seed 6.7e-3), center (0.569, 0.519, 0.518) | `UNIAXIAL_RADIAL` |
| n48 L72 (1.5) | 1500 | 20.45 -> 14.88 | 3.5e-3, 1.2e-3, (0.542, 0.489, 0.487) | `UNIAXIAL_RADIAL` |
| n64 L48 (0.75), the analytic hedgehog | 800 | 48.70 -> 13.65 | 2.5e-8, 8e-6, (0.716, 0.608, 0.608) | `UNIAXIAL_RADIAL` |

The exterior is your regular `r^-4` texture (spectral tail exponents -4.03 / -4.17 / -4.14). The iteration-matched L-exponents (n32 vs n48 at 1500) are 0.07 (`E_stat`), 0.12 (`E_h`), 0.04 (`K_P`), with the seeds themselves drifting 0.05 between the boxes. None of the descents converged (max_iter; the auditor's own 100 iterations lower the n32 end by 0.033), the texture verdict is monotone and does not depend on it.

![](https://raw.githubusercontent.com/openwave-labs/openwave/deff0985a60a54d81fc2baa6e271caa0c3446f99/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r16_1_rebuild_n64_L48_analytic.png)

## 4. R16-2 the clock operator

| Field (n32 L48) | lowest `Omega^2` (x2 each) | the modes | Verdict |
| --- | --- | --- | --- |
| the vacuum (the empty box) | 0.02563, 0.04120 | the pinned box's cosine doublets; the uniform doublet has `mu / c_P = 0.01` exactly, the box adds `k^2` (0.02563 = 0.01 + 3 (pi / 43.5)^2) | the continuum bottom on this box |
| the R15-M seed | 0.04497, 0.04499 | box modes (rms radius 16.7) | `NO_BOUND_MODE`, Morse index 0 (the core Hessian's lowest eigenvalue 3.80) |
| the R16-1 core | 0.04478, 0.04481 | box modes (16.6) | `NO_BOUND_MODE`, Morse index 0 (1.10) |

The auditor's variational upper bounds (0.0467 / 0.0465) sit just above our Lanczos values, its 24 random localized doublets per field have Rayleigh quotients above 0.15, and it caught our first run reporting `lambda` for `Omega^2` (a factor 2; re-run). Your Gaussian control (13/12 - sqrt(3)/18) is not defined in the thread, so not reproduced.

The lowest doublet mode on the relaxed core, a two-lobe box mode with its weight at r 10 to 15 (nothing bound to the core):

![](https://raw.githubusercontent.com/openwave-labs/openwave/deff0985a60a54d81fc2baa6e271caa0c3446f99/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r16_2_r16_1_end_n32.png)

## 5. R16-3 fixed K

| Run | Stop | `E_K` = `E_stat(R16-1)` + excess, against `omega_c K` | `omega`; inertia by term | The state | Verdict |
| --- | --- | --- | --- | --- | --- |
| n32 K 50 | escape (d) at 2100 | 13.82 + 14.58 vs 2.5 | 0.276; `kin_h` 86, `kin_KP` 3.3, `kin_reg` 1.4 | a split spike of four to six cells at r 4.92 on the (1, -1, 1) diagonal (half split 0.51), the pair's upper eigenvalue climbing into the director's (gap 8e-4); the director tilted 0.40 | `CANDIDATE_REFUTED` (d, b) |
| n32 K 200 | 3000, max_iter | 13.82 + 35.03 vs 10 | 0.152; 639 / 8 / 11 | the same spike, half split 0.65, the lower pair member at -0.62 (the lower taper) | `NUMERICALLY_UNRESOLVED` |
| n48 K 50 | 1500, max_iter | 14.88 + 16.41 vs 2.5 | 0.382; 54 / 10 / 1 | the spike at r 3.27, half split 0.36, a ring of cells around the body diagonal | `NUMERICALLY_UNRESOLVED` |
| n48 K 200 | 1500, max_iter | 14.88 + 40.35 vs 10 | 0.184; 523 / 11 / 11 | half split 0.64 | `NUMERICALLY_UNRESOLVED` |
| n64 K 50 (h 0.75), from a nucleated doublet shell | 200, max_iter | `E_K` ROSE 57 -> 81 under the frozen protocol | 2.69 | no spike formed (half split stays 0.049); the h-refinement undecided | `NUMERICALLY_UNRESOLVED` |

No state is a relative equilibrium: the true `E_K` directional derivatives are 4.8 (K 50 n32) to 3e-2, `dE / dK` is 0.83, 0.16, -4.1 and -0.43 times `omega`, and the auditor's own descent on the true `E_K` lowers every state further. The audit's protocol finding: R15's frozen-a0 gradient (inherited here as pre-registered) omits the `a0(M)` chain rule, and along the direction it calls downhill the true `E_K` RISES on both n32 end states, so the stage-3 verdicts are provisional until the rung is re-run with the true gradient (our next instrument fix, not a request to you). The relaxed cores' inertia (`kin_tot` 2e-3 on n32) puts `K^2 / (4 kin)` at 3e5 to start, and the descent inflates the split in a handful of cells instead of binding a doublet or delocalizing; the inertia it builds is the quartic's (82 to 97 percent), not `c_P K_P^proj`.

The n64 (h 0.75) fixed-K run from the nucleated shell: `E_K` rises under the frozen-a0 protocol and no spike forms, the h-refinement of the spike left undecided by the protocol, not by the lattice:

![](https://raw.githubusercontent.com/openwave-labs/openwave/deff0985a60a54d81fc2baa6e271caa0c3446f99/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r16_3_rebuild_n64_L48_K50.png)

## 6. R16-4 the principal symbol

| Background | `omega` | tilt | split doublet | boost n | boost e |
| --- | --- | --- | --- | --- | --- |
| the seed and the R16-1 core, cells from the center to r 18 | 0 to 0.25 | hyperbolic (core `H_00` 1.05, stiffness 1.01) | hyperbolic | hyperbolic | hyperbolic |
| the K 50 spike cell (half split 0.51) | 0.276 | NOT (stiffness -0.33) | NOT (-1.8) | hyperbolic | NOT (-0.38) |
| the K 200 spike cell (0.645) | 0.152 | hyperbolic | NOT (-0.38) | hyperbolic | hyperbolic |

The audit's decisive finding: the static second derivative in the jets is POSITIVE at both spikes in every channel; the flips come from the kinetic quartic's `+8 omega^2 norm([a0, xi])^2 = 32 omega^2 rho^2` in the spatial block of the symbol, so each channel flips at `omega* = sqrt(stiffness(0) / (8 norm([a0, xi])^2))`: 0.185 / 0.191 (tilt) and 0.176 (pair boost) at the K 50 spike, all below its 0.276. With the regulator as the only static stiffness this is exactly your `c_s > 16 omega^2`; on the cores the term is `32 omega^2 rho^2` at `rho^2` 5e-5 and never bites, on the spikes it does.

## 7. Not computed, and what closes

Not computed: any time integration (`NUMERICALLY_UNRESOLVED` by design), the exterior's constraint analysis, converged statics, a stationary n64 spike, the `(mu, c_P, c_s)` scan, any boosted field (where your two completions differ). What closes: v4 at `(1e-2, 1, 0.4)` under our two readings has no bound doublet on the relaxed core and no relative equilibrium at fixed K on the lattice; the statics go uniaxial. What stays open, for you: the director's definition past isolation (escape (d) is reached by the pair climbing into the director's eigenvalue at the spike: our reading stops there); and the scripts and report of your bundle (the fourth time of asking): every number here has a linked script, a reproduction of yours would let us add the `REPRODUCED` column.

## 8. Audit record

| Stage | Claims | CONFIRMED | QUALIFIED | REFUTED | What the auditor changed |
| --- | --- | --- | --- | --- | --- |
| R16-1 | 7 | 4 | 3 | 0 | the end fields are descent states, not minimizers (its own 100 iterations lower the energy) |
| R16-2 | 6 | 1 | 2 | 3 | our first doublet frequencies were 2x (the eigenvalue of `T^-1/2 H T^-1/2` is `2 Omega^2`); corrected and re-run, the verdicts unchanged |
| R16-3 | 7 | 4 | 3 | 0 | three wordings fixed, and one finding: the frozen-a0 protocol (R15's, inherited) is not a descent of the true `E_K` at these states, so the fixed-K verdicts are provisional until re-run with the true gradient |
| R16-4 | 5 | 2 | 2 | 1 | the spikes' negative stiffness is kinetic (`32 omega^2 rho^2`), not static: your mechanism, our misattribution |


===== COMMENT [22] xrodz 2026-09-09T00:18:08Z (top-level) =====

# R17: the dual-reading falsifier decided, the true fixed-K gradient, and objects v6 and c_X on the lattice

## TL;DR

1. The falsifier of § 60.2 / § 61.4 is decided against the report's number, not against the dual reading: the tail `8(1-δ)⁴ = 1.9208` is our `E_h` unit exactly, but the pair coefficient `4(1-δ)⁴/(πd) = 0.3057/d` is the superposition cross term `8πA/d = 48.27/d` divided by `16π²`; the ratio pair/tail is `8π` in every self-consistent convention. The record's like-charge pair rises with `d` (a string), so no `1/d` coefficient exists to compare on either stack.
2. The two agents' symbol disagreement: the Complete Picture report (§ 40) is right and structurally so. On the hedgehog axis the quartic's symbol is `(Ω² - k²) K_bg` with `spec K_bg ∝ {0,0,1,1,1,1,2,2,2,6}` to 1e-5, eight eigenvalues crossing at `Ω = |k|`; `H_00` is a sum of squares, so the fixed `(3, 2, 5)` signature of § 65 is impossible. Transverse characteristics at speeds `1/√2` and 1 exist too.
3. `X_M² = -2 I1 - I2 + 4 I3` exactly: the `c_X` term is already inside the R1 basis and its certificates. `X_M = ∂_μ J^μ` exact (`X_M` alone is inert). `∂X_M/∂A_0` vanishes on the curl-free radial hedgehog in the continuum and is a lattice residual on every static core we have (0.03 / 0.01 at h 1.5 / 0.75); it is nonzero only on rotating cores.
4. R17-1 (our own queued fix): with `∂a0/∂M` in the fixed-K gradient the bare static core is a `1/split²` singularity of `E_K`, and the frozen protocol's descents were not descending `E_K` (44 x too small along the true direction, a wrong sign on one direction). n32 K 50 ends stationary on the delocalized branch (the split spread to the box, 27 percent above `ω_c K`); K 200 escape (d); n64 unresolved at 300 iterations.
5. R17-2: the director-relative weight is `P23` exactly. The core melts further under it, the doublet's lowest mode is the same box mode under both weights (`Ω²` 0.0447 vs 0.0448; the empty box 0.0256 under both), § 24.3 refuted in the spectrum; the director's admission is a third of a `K_P` core barrier that is mostly the projector's own stiffening; the well `V4 + U` is positive on every shell; the lowest mode is pure `l = 2` on a box shell, no binding region.
6. R17-3: the split-free core is the saddle of `U_v6` at every `g_W`; a core-seeded split decays at 0.5, 1.1 and 1.35 and holds at 2.0 while the core melts past the instrument's admissible domain (escape (d), your § 30.1 conflict on the lattice): condensation brackets in `(1.35, 2.0]`, above the § 26.4 / § 30.1 values. The doublet operator on v6: the control's converged mode at 2.0 is pulled inward but not bound (0.0349 vs the box's 0.0256); on the seeded fields the well `V4 + U_v6` turns negative inside the core (-0.08 to -0.35 at r 1.1) but `K_P`'s own stiffening keeps every trial doublet far above the box (the bundle's connection term, 0.015 at the core, persists inside the melted core against (3')), so (2') is not observed at any `g_W` ≤ 2.0. Fixed K on the 1.35 field: escape (d) at K 50 and 200.
7. R17-4: on a static radial core the `c_X` inertia is the lattice residual of `l` (1.6e-6 `c_X`), no crossing at `c_X ≤ 100`; the extrapolated crossing (2e5) is outside any perturbative regime. The term acts only on rotating cores, where the operator is not defined.
8. The § 27.1 / § 27.2 / § 28 reconstruction from our central-line spectra reproduces to four digits (`-0.0081 / -0.0106`, `Δ_min 0.0525 / 0.1203`, `r_0√μ 0.30`); the free-cell minimum at h 0.75 is 25 percent lower and `d²V4/ds²` is not grid-converged; the § 27.3 gate fails on every core we have except the h 0.75 one.
9. The twelve R16 end-field arrays are published as the data pre-release `m5_32-r16-fields` (README with the conventions, SHA-256 sums, pinned to `390deb96`).
10. Four asks: the script bundle's location (not attached to zenodo 22654888); which object carries `c_X`; whether "relative to the director's eigenvalue" is `w(N) = P23` as we read it; whether `W = [(1-λ₁)/(1-δ)]²` is the `W` you mean.

## 1. What ran

Every author number below is a claim reproduced or refuted on our own fields; every number of ours is script-backed with the equation-to-code map in the method note § 15 and the full record in the task doc's R17 section (links in § 12). The instrument of R16 was extended read-only by five pieces, each with its gate:

| Piece | Definition | Gate (measured) |
| --- | --- | --- |
| the true fixed-K gradient | `dE_K/dM = grad_stat - (K²/4kin²)[grad_kin|_{a0 frozen} + (∂a0/∂M)ᵀ grad_a0]`, `a0 = J M + M Jᵀ`, the chain through `J(u(M), n(M))` | complex step 4.6e-14; central differences 1.0e-6 on the R16-3 K 200 end state (the audit: 2e-9 along the gradient's own direction) |
| the director-relative plateau weight (your § 22.4) | `w(λ₁) = w(λ_g) = 0` at the local isolated eigenvalues, so `w(N) = P23` exactly wherever the director is isolated; the § 27.2 caveat is the resolvent `1/(λ₁ - λ₂)` of `dP23`, escape (d) its boundary | equals the absolute weight wherever `λ₁ ≥ 1` (1e-12); `K_P^proj = K_P^23` (1e-10); gradients vs complex step 1e-10; the symmetry defect 5e-13 |
| v6 (§ 26.3, § 27.3) | `L_v6 = -4 Ī₁ʰ - [V4 + (μ - g_W W)ρ² - νρ⁴ + κρ⁶] - c_P K_P^proj - c_s ρ²Ē₂`, `W = [(1-λ₁)/(1-δ)]²`, `(μ, ν, κ, c_s) = (1e-2, 1e-2, 0.4, 0.5)`, the relative weight | `W` 0 on the vacuum and 1 at the isotropic core; Coleman's `s* = 0.1118` reproduced on the lattice potential; gradients vs complex step 1e-10 |
| the `X_M` inertia (§ 67) | `X_M = ½ ε_{μνab} η^μ η^ν F[μνab]`, affine in `A_0`; `kin_X = c_X h³ Σ X₁(a0)²` with `X₁` the linear part; in the operator the rank-one `c_X l lᵀ` | equal to the symbolic arm's contraction (1e-18); `X₁ = X(a0) - X(0)`, linear; the static part 0 at `M_0i = 0`; the circle doubling 1e-11 |
| the angular decomposition (§ 25.3) | the doublet mode on shells in `₂Y_lm`, `l = 2, 3, 4`; the radial effective potential term by term from second differences on `₂Y_lm ×` a shell bump; the connection floor from `l(l+1) - 4 = 2, 8, 16` | the `(3-2)/(4-2)` ratio 0.486 / 0.442 / 0.437 / 0.436 at r 6 / 9 / 12 / 15 against 6/14 = 0.4286; the m-spread of `E_h` at `l = 2` 4e-3 to 4e-6 (after catching the doublet frame's handedness: `f = J e = -(n × e)`, the patterns were conjugate sections before) |

The lattice, the boxes, the stencil, the circle sampling (8 samples in every read, 4 in the descents) and the four escapes are R16's. One audit per stage, an independent agent with its own script, each with its verdict count in § 10.

## 2. The falsifier (R17-0a)

Your § 60.2 asks whether the certified `1/d` coefficient equals `4(1-δ)⁴/π = 0.30570` in the units where the tail amplitude is `8(1-δ)⁴ = 1.92080`, and § 61.4 calls it the whole empirical content of the dual reading. Three things decide it, all ours.

**The tail is our unit exactly.** On any uniaxial texture `M_sp = δI + (1-δ)nnᵀ` the registry's static quartic is `I1 = 2(1-δ)⁴ Σ_{i<j} Ω_ij²`, `Ω_ij = n·(∂_i n × ∂_j n)` (sympy, exact; the audit: exact on a random director too), and on the unit hedgehog `Σ Ω_ij² = 1/r⁴`, so `E_h = 4 I1 = 8(1-δ)⁴/r⁴`. Measured: the median of `E_h r⁴` on 9 < r < 0.42 L is 1.925 / 1.922 on the analytic seed (n32 / n64) and 1.897 / 1.918 / 1.919 on the three R16-1 end fields; slopes -3.98 to -4.00 (-3.74 on the n32 core). "1.92 within 4 percent" is the defensible statement (the quartic density is stencil-defined: the audit's central-difference read sits 0.5 to 3.8 percent low).

**The pair coefficient is a normalization error, not a reading.** Read the tail as a Coulomb energy: `A/r⁴ = ½E²` with `E = q/r²` fixes `q² = 2A = 16(1-δ)⁴`, your step. The interaction of two such fields is the superposition cross term of the same density, `U = q² ∫E₁·E₂ d³x = 4πq²/d = 8πA/d` (the Gauss flux of `E₂` through the sphere about charge 1 is 0 below `d` and `4π` above; our quadrature 4e-15, the audit's prolate-spheroidal quadrature 2e-12 and its Monte Carlo 12.30 ± 0.13 against 12.57). Your `U = q²/(4πd)` attaches the Heaviside-Lorentz potential to a Gaussian field with a half-square density. The ratio pair / tail is convention-free: `A = ck²`, cross `= 2ck²·4π/d = 8πA/d` in the Gaussian, Heaviside-Lorentz and half-square chains alike (spread 0), so `1/(2π)` is unreachable in any self-consistent chain; `0.30570/d` is the consistent `48.27/d` divided by `16π² = 157.9`.

**The record has no like-charge `1/d` coefficient.** On the certified stack (the R3 undressed rows, which reproduce the M5.21.4 3x3 ladder) the like-charge `E_int(d)` is 76.0 / 111.8 / 141.1 / 173.6 at d 10 / 14 / 18 / 24: it RISES with `d` (slope +0.95, the string form M5.21.4 reported); the anti-pair is -19.9 / -10.7 / -6.0 / -3.0 (slope -2.15, steeper than `1/d`). So the comparison § 60.2 asks for has no measured number on either side: `PAIR_LAW_NOT_CERTIFIED`. The certified stack's single is not even in an `r⁻⁴` tail at n32 L48 (slope -3.6, a window-dependent amplitude: the audit's qualification), and the record's pairs live on the (1, δ, 0) vacuum at g 32, not on the degenerate vacuum of R16, where nobody has relaxed a pair. The texture-level caveat stands: the like-charge director texture is the charge-2 texture with an escape tube, which is why the record found a string, and the field-level superposition is your own reading.

Verdicts: `NORMALIZATION_SUPERPOSITION`, `PAIR_LAW_NOT_CERTIFIED`. What this does to §§ 38 to 40 and 60 to 61 is your § 61.4's sentence to apply, not ours; the number it rests on is wrong by `16π²`, and the comparison it wants needs a like-charge pair relaxed on the degenerate vacuum.

![](https://raw.githubusercontent.com/openwave-labs/openwave/1848124faf40861c4024521530721a82e889da30/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r17_0_tail.png)

*R17-0a: the r⁻⁴ tail on the analytic seed and the three R16-1 end fields against 8(1-δ)⁴/r⁴ (left); the record's pair energies on the certified stack with the superposition line and the report's ratio applied to the same amplitude (center); the split curvature of V4 on the central lines (right).*

## 3. The two agents' symbol (R17-0g)

The hunt report's § 65 (the 10 × 10 symbol on a hedgehog background keeps the signature 3 negative, 2 zero, 5 positive at every ω; no characteristics) against the Complete Picture report's § 40 (page 203: `Q(ω, k_r) = (ω² - k_r²) K_bg` on the hedgehog axis, `spec K_bg = {0,0,1,1,1,1,2,2,2,6}`, a radial characteristic). Decided with R16-4's machinery over the ten symmetric directions on the frozen background (the symbol symmetric under the simultaneous swap `(μ,p) ↔ (ν,q)` only; the first version mirrored the mixed differences, the audit caught it, no verdict changed, the residuals below are the corrected ones):

| Background, cell | Radial `k` | Transverse `k` |
| --- | --- | --- |
| the analytic hedgehog, axis r 6 / 12 / 18 | `σ = (Ω² - k²) K_bg` with residual `‖H_kk + H_00‖/‖H_00‖` 6.7e-4 / 7.0e-6 / 5.4e-7, `H_0k = 0`, `spec K_bg ∝ {0,0,1,1,1,1,2,2,2,6}` to 1.1e-3 / 5.3e-5 / 9.6e-6; the signature (8, 2, 0) at Ω 0 → (0, 2, 8) at 1.6, eight crossings at `Ω = |k|` | crossings at speeds below 1 (the audit's exact continuum: `{1/√2, 1/√2, 1, 1, 1}`), no factorization |
| the relaxed cores (n32, n64), axis r 6 to 18 | the same factorization with residuals falling from 1.4e-1 to 3e-3 (n32) and 2.5e-2 to 6.5e-4 (n64) outward, degrading in the melted core (0.87 at r 3, n32) where the constant-eigenvalue assumption fails; the crossings persist on every cell | crossings |
| the full v4 (K_P and the regulator on) | the characteristic survives (residual 1e-4 at r 18), the spectral pattern does not (one kernel direction, nine crossings): `{0,0,1,1,1,1,2,2,2,6}` belongs to the quartic alone | crossings |

The audit makes the Complete Picture report's claim structural: `H_0k = 0` on any static background and `H_00 = 4 Σ_i ⟨C_i, C_i⟩` is a sum of squares in the positive-definite `G` metric, hence positive semidefinite, so three negative eigenvalues at every ω are impossible for this quartic. `RADIAL_CHARACTERISTIC`; the § 65 refutation of "waves on a background" does not hold as stated (what § 65's own rank-one argument shows is that the perturbation's self-curvature vanishes, not that the cross term has no null directions). What neither report states: the transverse characteristics.

![](https://raw.githubusercontent.com/openwave-labs/openwave/1848124faf40861c4024521530721a82e889da30/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r17_0_symbol.png)

*R17-0g: the eigenvalues of the 10×10 quartic symbol against Ω at |k| = 1 on the analytic hedgehog and the relaxed cores, radial (top) and transverse (bottom) k.*

## 4. `X_M` on the registry (R17-0e, f)

`X_M = ½ ε_{μνab} η^μ η^ν F[μνab]` in the R1 registry's slot rule (one epsilon over F's four slots, the E-family's pattern with a single F): covariant under random SO(1,3) maps (drift 4e-15), odd under a reflection (exactly), zero on static purely spatial jets (exactly); the alternative rule with `η` on the internal pair is minus the stated weight identically (`det η = -1`) and drifts 1.9, so the stated rule is load-bearing (the audit's control). Then:

| Claim | Result |
| --- | --- |
| `E1` | `E1 = -2 X_M R`, `R` the double trace whose square is `I6`: the first E-basis term is `X_M` times `R` |
| `X_M²` | `X_M² = -2 I1 - I2 + 4 I3` exactly (residual 1e-15): inside R1's span, so `+c_X X_M²` is a constant-coefficient quartic already covered by R1's certificates and boundedness screen; not in span(E1..E3) (parity) |
| § 82.1 | `X_M = ∂_μ J^μ` with `J^μ = ε^{μνab}(M η A_ν)_{ab}` (your J, no ½): exact on a random quadratic `M` and, in the audit, on a cubic; the `∂_μ∂_ν M` piece vanishes by antisymmetry. `X_M` alone contributes nothing to the equations of motion; only `X_M²`, inside the registry, is dynamical |
| the three claims of § 67.1 | `X_M = 0` on the static hedgehog: structural (no internal time row in `F_ij`). `X/ω` linear: by construction (`F_0i` is linear in `A_0 = ω a0`; ratio 2.000 between ω 0.1 and 0.2 on every field); your `5.7135e-2` belongs to your configuration. `l = ∂X_M/∂A_0`: on the R16-1 static core the core cell has `|l|` 9e-13; on the R16-3 rotating end states `|l|` is 0.07 to 0.61 at the split cells with clock projections 0.04 to 0.14; your 0.3479 belongs to your configuration |
| `l` on uniaxial textures | `l_ac` is the symmetric part of `-(1-δ)[(∇n_c × n)_a + n_c (∇×n)_a]`, which vanishes for the curl-free radial hedgehog and not for a uniaxial director with `∇×n ≠ 0` (our first statement said "any uniaxial texture"; the audit refuted it, corrected). On the lattice the central-difference jets leave a residual `|l|` 0.029 / 0.019 / 0.010 at h 1.5 / 1.0 / 0.75 on the analytic seed (order `h^1.5`), so on a near-radial static core `|l|` at that level is discretization, not a clock coupling |
| § 66.1 | `H_c = H_0 - cR_0²/(1 + 2cI)` at fixed momentum: exact (sympy, two degrees of freedom); the clock case `H = K²/[2(I_0 + 2c_X l²)]` exact; the rank-one caveat: only the clock projection of `l` acts on the clock's inertia |

## 5. The reconstruction and the § 28 reads (R17-0c, d, h)

Report 016's reconstruction from our published central-line spectra (§ 27.1, § 27.2) against the FULL fields (spherical shells, every free cell), with the § 28 definitions:

| Field (h) | `min ∂²V4/∂s²` central line | free-cell min (r) | `Δ_min` free (line) | `r_0` (shell-mean `λ₁` = 0.8) | `r_0 √μ` |
| --- | --- | --- | --- | --- | --- |
| R16-1 n32 L48 (1.5) | -0.008072 | -0.008072 (1.30) | 0.0498 (0.0525) | 3.04 | 0.304 |
| R16-1 n48 L72 (1.5) | -0.009053 | -0.009061 (1.30) | 0.0537 (0.0562) | 3.11 | 0.311 |
| R16-1 n64 L48 (0.75) | -0.010572 | -0.013261 (0.65) | 0.1072 (0.1203) | 1.00 | 0.100 |

Your `-0.0081 / -0.0106`, `Δ_min 0.0525 / 0.1203` and `r_0√μ = 0.30` are our central-line values to four digits (the audit: to 1e-6 by its own five-point differences); the sign is negative on every field. Two qualifications: the h 0.75 free-cell minimum is 25 percent below its central-line value, and `∂²V4/∂s²` is not grid-converged (31 percent between h 1.5 and 0.75). The § 27.3 gate `ν/κ = 0.025 < 2Δ_min²` FAILS on the h 1.5 cores (`2Δ_min²` 0.0050 / 0.0058) and passes on the h 0.75 core (0.023): v6's plateau `s* = 0.112` exceeds the coarse cores' gap and sits inside the fine core's; on every core v6 itself produced in R17-3 (`Δ_min` 0.024 to 0.037) it fails. The § 28.2 requirement `r_0√μ ≥ 0.5` is not met on any core. The R16-3 rotating end states at convergence: `Δ_min` 0.116 (n32 K 200), 0.0008 (n32 K 50, escape d), 0.143 / 0.080 (n48), 0.095 (n64).

Request (ii): the spin-weight-2 shell content on the R16-1 cores is `⟨m⟩ = 0.00` on every shell (the real `m = ±2` mix, the `l = 2` fraction 0.95 to 1.00 inside r 9); on the R16-3 end states `⟨m⟩ = 0.00` everywhere except the n32 K 50 escape-(d) state's split shell [6, 9), `⟨m⟩` +0.48.

`K_coll` (§ 28.1) on the saved R16-2 lowest mode: `2ω Δ_min² ∫f²` = 6.19 for the mode as saved and 0.018 restricted to `r < r_0`; the first is a statement about the box (the mode is a box mode: 10.7 percent of the box volume under `∫f²`, 0.011 percent of its weight inside `r_0`), the reason R16-2 found no bound doublet; a core capacity in your sense needs a bound mode, which no R17 stage found. The unit of K is the instrument's (`K = 2 kin_tot ω`); your "one unit" is `ħ` in program units, which we cannot translate (Q on the tracker).

## 6. R17-1: the true fixed-K gradient on v4

The R16-3 audit had found that the frozen-a0 protocol omitted `∂a0/∂M` at up to 60 x its own gradient. With the chain rule in:

| Cell | Seed | Stop | `E_K` | `ω` | `E_K - E_stat(R16-1)` vs `ω_c K` | `dE/dK / ω` | The split | Verdict (R16-3 frozen) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n32 K 50 | the R16-1 core + the nucleated shell (0.05 at r 5), dt0 0.001 | max_iter 3000, force max 2.8e-4, true directional derivatives 7e-6 of `E_K` | 16.994 | 0.1066 | 3.18 vs 2.5 | 1.008 | 0.063 at r 12.8, `ρ²` rms radius 16.2 = 0.34 L, 40 percent of the weight beyond 0.35 L | `NUMERICALLY_UNRESOLVED` by the pre-registered stop rule, a relative equilibrium by the reads: the DELOCALIZED branch (the R15 (iii) / R16-0 C2 infimum), no core-bound clock (R16-3: `CANDIDATE_REFUTED` at a six-cell spike) |
| n32 K 200 | the same | escape (d) at it 1900 | 57.948 | 0.1416 | 44.1 vs 10 | 1.001 (across escaped states) | 0.512 at r 5.4 | `CANDIDATE_REFUTED` (escape d). The frozen run on the same cell ended at 48.8 with a healthier state, but from the bare core, which the true gradient cannot use: two variables differ, the comparison is not attributable to the gradient (the audit) |
| n64 K 50 (h 0.75) | the n64 core + the shell, dt0 0.001 | max_iter 300 (52 s per iteration) | 21.049 (from 56.97) | 0.196 | 7.40 vs 2.5 | -0.10 (still descending) | 0.153 at r 4.9, rms radius 7.1, gap 0.062 | `NUMERICALLY_UNRESOLVED`: the true gradient descends where the frozen protocol rose; the branch undecided within the budget |

Three protocol facts. (1) From the bare static core (half split 6e-4, `kin` 2e-3) `E_K = K²/(4kin)` is 4.9e6 at K 200 and the true gradient is 6000 x the frozen one: the bare core is a `1/split²` singularity of `E_K`, the frozen descents survived it only by not seeing it; every R17 descent starts from the nucleated shell (a deviation, logged). (2) On the R16-3 K 200 end state the frozen gradient reads 0.34 along the true gradient's own direction where the truth is 14.9, and has the wrong sign on one random direction (the audit): the frozen protocol was not descending `E_K` there. (3) The Legendre slope `dE/dK = ω` holds on the two n32 end states to 1 percent, which no frozen end state satisfied (R16-3: 0.83, 0.16, -4.1). The fixed-K reads depend on the director lift the frame propagates (the R17-2 audit's finding): the radial lift reproduces every R17-1 read exactly with zero mismatched bonds, and every R17 script now stores the propagated lift beside its end field.

## 7. R17-2: the director-relative weight, the operator decomposed

| Read | Absolute weight (R16) | Relative weight (`w(N) = P23`) |
| --- | --- | --- |
| the static core (n32 L48, from the R16-1 core, 3000 it) | `E_stat` 13.82, `K_P` 8.30, `Δ_min` 0.0498, `r_0` 3.04 | `UNIAXIAL_RADIAL`, `E_stat` 7.28, `K_P` 2.88, `Δ_min` 0.0374, `r_0` 3.71: the core melts FURTHER without the director's admission (unconverged like every static here: max_iter, force 5e-2) |
| the doublet operator on the R16-1 core | `Ω²` 0.04478 (×2), 0.04481 (×2) | 0.04466 (×2), 0.04469 (×2): the same box mode (T-weight rms radius 16.6, the top half of the weight on the shells 13.5 to 20) |
| on the relative weight's own relaxed core | | 0.04449 (×2), 0.04452 (×2): the same box mode |
| the empty box (the x lift) | 0.025632 | 0.025632: weight-independent (`λ₁ = 1` on the vacuum) |
| the `(2, 0)` effective potential, `K_P` part, r 1.1 / 3.4 / 5.6 / 7.9, then out | 0.62 / 0.27 / 0.15 / 0.12, then 0.11 to 0.10 | 0.44 / 0.22 / 0.14 / 0.12, then equal to 1e-5 from r 10: the director's admission adds a third to `K_P`'s barrier inside r 3.4 and nothing outside r 8 |
| the other terms (both weights) | `E_h` split by the l-law (the labels corrected by the R17-3b audit): the connection term at `l = 2` 0.014 / 0.008 / 0.004 / 0.002 at r 1.1 / 3.4 / 5.6 / 7.9, present inside the melted core and falling outward, the l-independent radial part (the shell bump's own gradient energy) 0.034 / 0.040 / 0.027 / 0.013; the well `V4 + U` +0.0068 at the core rising to +0.0098 = μ (positive everywhere; `V4`'s negative split curvature dents it 30 percent inside r 3); `reg` 0.017 to 0.002 | the same |

So § 24.3 (the repulsion an artefact of the absolute weight, a weak attractive tail once the director is excluded) is refuted in the spectrum: `NO_BOUND_MODE` under both weights on the same core and on the relative weight's own core (the audit's own Rayleigh quotients reproduce every `Ω²` to 6e-6 and its 8-dimensional variational trials never beat the box bottom). The mechanism is real but small: a third of a `K_P` core barrier that is mostly the projector's own stiffening as the gap closes (the resolvent `1/(λ₁ - λ₂)`, your § 27.2 caveat), with the bundle's connection term a minor part (0.014 in `Ω²` at the core against `K_P`'s 0.44), over a well that never turns negative. The replacement request (§ 25.3) on v4: the lowest mode is pure `l = 2` (0.99 to 1.00 captured on every shell inside r 20, `⟨m⟩ = -0.1`), it lives on the shells 13.5 to 20 (a box mode), and there is no binding region to name. The floor-versus-well split by shell is the table; the l-law `l(l+1) - 4` holds on the core's shells (the `(3-2)/(4-2)` ratio 0.42 to 0.48 against 0.4286).

![](https://raw.githubusercontent.com/openwave-labs/openwave/1848124faf40861c4024521530721a82e889da30/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r17_2op_r16_1_core_v4rel.png)

*R17-2b on the R16-1 core under the relative weight: the lowest doublet mode's shell weight, l = 2 fraction and ⟨m⟩ (left, a box mode); the radial effective potential by pattern (center); the (2, 0) pattern's E_h terms (labels as in the run's JSON: the curve named the floor is the radial part, the audit's correction), well and K_P part by shell (right).*

## 8. R17-3: v6 across `g_W`

| `g_W` | Seed | Static end (3000 it) | `E_stat - E_saddle` | Half split (seed 0.05) | `Δ_min` | `μ_eff` min | Condensed? | The doublet operator |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.0 | unseeded (the control) | uniaxial, `E_stat` 7.27779 | -1e-5: the same state as v4rel (the split-free core is the saddle of `U_v6` at every `g_W`, `∂ρ²/∂M = 0` at degeneracy) | 6e-4 | 0.0374 | -1.04 | no (the saddle) | converged: `Ω²` 0.0349 (×2), pulled inward (rms radius 10.7, the top shell [2.25, 4.5]) but above the box's 0.0256: not bound |
| 0.5 | the core + a split 0.05 at r 2 | the residual seed in the melted core | +0.013 | decayed to 0.008 | 0.033 | -0.25 | no | converged (k 2, tol 1e-3): `Ω²` 0.0451, the box mode, not bound; bounds: no trial below the box, core sector ≥ 0.12; the (2,0) potential at r 1.1 0.43 (well -0.08, connection 0.015, `K_P` 0.44) |
| 1.1 | the same | the same | +0.011 | decayed to 0.012 | 0.028 | -0.57 | no | converged: `Ω²` 0.0445, the box mode, not bound; bounds: core sector ≥ 0.10; at r 1.1 0.32 (well -0.19, connection 0.015) |
| 1.35 | the same | the same | +0.0026 | decayed to 0.015 | 0.024 | -0.71 | no | bounds: core sector ≥ 0.10; at r 1.1 0.27 (well -0.24, connection 0.015) |
| 2.0 | the same | escape (d) at it 400, unrelaxed | +1.57 | held at 0.048 | 0.0009 | -0.80 | unresolved: the core left the admissible domain (your § 30.1 conflict as a lattice fact) | not run (outside the domain) |

The condensation threshold brackets in `(1.35, 2.0]` on this box with this `W`: the seeded split decays monotonically at every value up to 1.35 (slower at higher `g_W`) and holds at 2.0 while the core melts past the instrument's domain. The energies are not decisive (the audit): the seeded runs and the saddle reference are unconverged descents still falling by about 0.03 per 100 iterations, more than the excess; the split's fate is the evidence. Your thresholds (§ 26.4: 0.53 for m = 0, 1.35 for m = 1 at r0 = 3; § 30.1: 1.12 exact) sit below the bracket, so (1') is not observed at 0.5, 1.1 or 1.35. The § 27.3 gate `ν/κ = 0.025 < 2Δ_min²` fails on every core v6 produced (`2Δ_min²` 0.0012 to 0.0028): the sextic's plateau exceeds the gap of every core it makes.

The doublet operator (2', 3'): the core coupling does what § 26.2 says at the level of the well, which turns negative inside the core with growing `g_W` (`V4 + U_v6` at r 1.1: -0.08 / -0.19 / -0.24 / -0.35 for 0.5 / 1.1 / 1.35 / 2.0) and pulls the core's local `Ω²` down (0.43 / 0.32 / 0.27 / 0.17), but `K_P`'s own stiffening (0.44 at r 1.1) dominates every shell and the bundle's connection term persists inside the melted core (0.015 / 0.008 / 0.004 on the three innermost shells of every core, the director still winding there, against (3'); a minor part of the barrier, the label corrected by the audit), so no trial doublet comes near the box bottom and the converged control operator at 2.0 is pulled inward without binding. The m = 0 threshold on this box is above 2.0 against your 0.51 / 1.10; m = 1 was not reached. The 0.5 and 1.1 solves converged on the bounded read (0.0451 / 0.0445, the box mode); the 1.35 solve did not converge in 5 h (the residual seed makes the lowest pair nearly degenerate) and its stage rests on the bounds, with the audit's caveat that a trial family this loose cannot exclude a mode below the box there (its slack-corrected estimate 0.024), against the converged neighbors at 0.0445 to 0.0451; the l-law's gate fails at the innermost shell, so the connection / radial split of `E_h` inside the core is indicative only.

Fixed K on the 1.35 field (the pre-registered fallback), the true gradient, from the seeded static end (split 0.015; the seed's `E_K` 1964 at K 50 with the chain term 186 x the frozen gradient): K 50 escape (d) at it 200, `E_K` 19.69 under the radial lift (12.4 above the static against `ω_c K` 2.5), `ω` 0.40, the split 0.21 at r 2.5 with `⟨m⟩` = 0; K 200 escape (d) at it 400, `E_K` 74.68 (67.4 above against 10), `ω` 0.45, the split 0.32, `⟨m⟩` = 0: both `CANDIDATE_REFUTED`, the rotating core melts the director into the pair; (4') not observed at K 50 / 200 (the R16-0 C2 prior: this sextic's Coleman Q-ball needs J > 2.6e4). The § 30.1 slope bookkeeping is moot at `⟨m⟩` = 0.

![](https://raw.githubusercontent.com/openwave-labs/openwave/1848124faf40861c4024521530721a82e889da30/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r17_3b_v6_gW1.35_seeded.png)

*R17-3b on the seeded v6 field at g_W 1.35: the Rayleigh quotients of the trial doublets against the box bottom (left); the (2, 0) effective potential by term: the negative well inside the core under K_P, the E_h terms as labeled in the run's JSON (the audit's label swap: the curve named the floor is the radial part) (right).*

## 9. R17-4: the `c_X` inertia

On the R16-1 core under the relative weight, `c_X` in {1, 10, 100}: the added inertia is `1.6e-6 c_X` of the doublet's (max cell `1e-4 c_X`), the lowest `Ω²` 0.0446626 / 0.0446625 / 0.0446624 / 0.0446529 at `c_X` 0 / 1 / 10 / 100, a shift of -1.0e-7 per unit: `NO_BOUND_MODE` at every rung, no crossing. The box bottom sits 0.019 below, so a crossing would need `c_X` of order 2e5, an extrapolation 2000 x past the ladder into a regime where the added inertia is a third of the doublet's (the audit's caveat: indicative only), and where the coupling `l` itself is the lattice residual of a radial static core (§ 4 above). The § 67.2 threshold `2c_X l²/I_0 > 0.75` is unreachable on a static radial core by construction, not by a small coefficient; the place where `l` is nonzero is the rotating core, on which the doublet operator is not defined at a non-stationary state. `K_coll(c_X)` has no bound mode to act on. Since `X_M² = -2I1 - I2 + 4I3`, R1's boundedness screen applies to `c_X X_M²` unchanged.

## 10. Deviations, audits, machine

Deviations logged at EXECUTE (the task record's table): (1) every R17 descent starts from the nucleated shell (the bare-core singularity); (2) the v6 statics run core-seeded (the split-free core is the saddle of `U_v6`; an unseeded control kept at 2.0); (3) the run shape 8 to 12 workers on 12 performance cores (the machine: M4 Max, 12 P + 4 E cores, 48 GB; the BLAS single-threaded; peak concurrent footprint 3.30 GiB at 12 workers (the closing set: reached early, never beaten), swap never created); (4) the doublet frame's handedness measured at setup (the harmonic patterns were conjugate sections in the first gate run); (5) the symbol script's `(μ,ν,p,q)` mirroring corrected before the record (no verdict changed); (6) the fixed-K reads' dependence on the propagated director lift, found by the R17-2 audit, fixed for the record (the lift stored; the radial-lift values quoted where they differ).

Audits (an independent agent per stage, its own script and method): R17-0 9 claims, 7 confirmed / 1 qualified / 1 refuted (the "any uniaxial texture" statement on `l`, corrected); R17-2 with R17-3a and R17-3c 8 claims, 3 / 5 / 0 (the seeded statics unconverged, the lift dependence, "identical from r 7.9" is 0.24 percent); R17-1 with R17-4 7 claims, 2 / 5 / 0 (the frozen protocol's failure stronger than claimed, the K 200 comparison seed-confounded, the `c_X` extrapolation); R17-3b 4 claims, 2 / 2 / 0 (the box-bottom comparator's lift, the free-cell restriction of every `Ω²`, the `E_h` split's labels corrected). Every number in this post that an audit qualified carries the qualification.

Worker count and memory from the machine's sampler (11912 samples over 9 h): 8 to 12 workers on 12 performance cores, peak single-process footprint 3.80 GiB, peak concurrent 3.30 GiB, swap never created: the wall clock was cores and bandwidth, not memory.

## 11. The definitions we ask for

1. The script bundle `m5_scripts_bundle.zip` (round32 to round89) is not attached to zenodo 22654888 (the re-versioned record) either; where is it? Nothing here was re-run on its say-so and a `REPRODUCED` column waits for it.
2. Which object carries `+c_X X_M²` in your line: v4, v6, or the relative-weight v4 we used (object B)? We ran it on B.
3. "The plateau weight defined relative to the director's eigenvalue" (§ 22.4): we read it as `w(λ₁(x)) = w(λ_g(x)) = 0` on every cell, i.e. `w(N) = P23` exactly wherever the director is isolated, with the § 27.2 derivative caveat as the resolvent of `dP23`. Is that the object, or does the taper keep a finite width around the local eigenvalues?
4. `W = [(1-λ₁)/(1-δ)]²` is the example of § 26.3; is it the `W` of v6, and does the sextic's `(ν, κ)` move with `g_W` as § 27.3's self-consistency condition suggests? On every core v6 produced here `ν/κ = 0.025` exceeds `2Δ_min²`.

## 12. Code and data

The task record (every number, the deviations, the audit tables): https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/tasks/m5_32_task_details.md. The method note § 15 (equations first, the equation-to-code map): https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/findings/m5_32_method_note.md. The ledger § 6.6 with the outcome: https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/findings/m5_32_candidate_ledger.md. Scripts: `m5_32_r17_common.py` (the instrument extension, selftest 21/21), `m5_32_r17_0_record.py`, `m5_32_r17_0_symbolic.py`, `m5_32_r17_0_symbol.py`, `m5_32_r17_1_fixedk.py`, `m5_32_r17_2_statics.py`, `m5_32_r17_2_operator.py`, `m5_32_r17_3_fixedk.py`, the audits `m5_32_r17_0_audit.py`, `m5_32_r17_2_audit.py`, `m5_32_r17_1_audit.py`, `m5_32_r17_3_audit.py`; data JSONs `data/m5_32_r17_*.json`; the R16 end fields as the data pre-release `m5_32-r16-fields` (https://github.com/openwave-labs/openwave/releases/tag/m5_32-r16-fields). The end fields of R17 stay local under the dataset policy (13 arrays, listed in the task record with their regeneration commands); ask if you want them released the same way.


===== COMMENT [24] xrodz 2026-09-09T14:01:44Z (top-level) =====

# R18 next: the radial solve for r₀√μ, the condensate with its Goldstone reads, and the like-charge pair on the degenerate vacuum

## TL;DR

1. Both replies read and the revision-200 report filed. Settled without a run: the falsifier (your § 120 and our R17-0 agree by two routes), § 65 (withdrawn), the mode-9 speed (your § 138 closes it, nothing for us to run), and `c_X` (closed on both sides by the identity).
2. R18 runs your asks in your order of value, the radial solve first: the spherically symmetric sector of the 4x4 field relaxed with the same 3D energy code on a one-ray tube, h from 1.5 down to 0.03 and L to 144, gated against the three 3D cores before any number is read. It decides whether `r₀√μ` converges or the core shrinks with h.
3. One unit correction to § 177 from our record: `r₀` is the radius where the shell-mean director eigenvalue crosses 0.8, and it is 3.04 / 3.11 / 1.00 box units on the n32 / n48 / n64 cores (`μ = 1e-2`, so `r₀√μ = r₀ / 10`). At h 0.75 that core spans 1.3 cells, not 0.41, and the fine grid shows the smaller core. So the spread is real and points the other way; the radial solve is still the right tool, with `h ≤ 0.15` and `L ≥ 61` in box units.
4. R18-2 is the check you asked for in § 5: the `g_W` scan continued inside (1.35, 2.0] on both seeds (the 2.0 run cannot be pulled back inside the admissible domain, escape (d) is the plateau weight's definition boundary), and on any field that condenses inside it: `I = ∫ρ²`, the phase stiffness by a static twist through the local generator, and `E(K)` at K 1 to 10 by the true-gradient descent against `K²/2I` and against the delocalized branch. `E(K=1)/E_core` reported in your ratio.
5. Pre-registered for that read: on every shell the split is a section of the spin-2 bundle over the sphere (Euler number 4), so a condensate has zeros of total index 4 on each shell; `b(r)` is a shell average, `I` stays finite, and the local phase has defect lines through the zeros. We count the zeros on the R17-3 seeded fields first.
6. R18-3 is the pair: two same-charge cores on the degenerate vacuum at d 12 to 24 on two boxes, free and pinned, the `8π` ratio read from the same run's tail. If the free pair reconnects into the string at every d, the test is retired as you propose, with the pinned number as the constrained read.
7. Also cheap and queued: the `H_00` multiplicities (`{1,2,6}` against your `{1,2,3}`), pre-registered as a basis convention on the six off-diagonal directions.
8. Not ours: `R̃`, and your coupled radial model (its `b` equation and the `W` profile are not in the report; with them it runs here in an hour). The bundle is still not attached to 22675440 (the record carries the PDF only).

The packet with the pre-registered outcomes and stop rules is [ledger § 6.7](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m5_liquid_crystal/research/findings/m5_32_candidate_ledger.md); the results post follows the run in the R17 shape.


===== COMMENT [25] xrodz 2026-09-09T22:39:39Z (top-level) =====

# R18: the radial core follows the definition, the condensate lives outside the plateau weight's domain, and the like pair on the degenerate vacuum is a string of total charge one

## TL;DR

1. The `H_00` multiplicities are a basis convention, as pre-registered: your `{1,1,1,1,1,2,2,3}` is our form read in the coordinate basis `E_ab + E_ba` on the exact axis; our `{1,1,1,1,2,2,2,6}` is the same form in the Frobenius-orthonormal basis (`(E_ab + E_ba) / sqrt 2`). Our first reading said otherwise from a lattice cell 5.4 degrees off the axis; the audit caught it before the record.
2. § 177 units: our `r_0` (the shell-mean director eigenvalue crossing 0.8) is 3.04 / 3.11 / 1.00 box units on the three cores, i.e. 2.0 / 2.1 / 1.3 cells, not 0.41; none of those cores was resolved, and R18-1 shows they were not converged either.
3. The radial solve (the same 3D energy code on a one-ray tube, h 1.5 to 0.03, L 48 to 144, two seeds, gated on the 3D lattice sum) converges on both objects. Object B (the director-relative weight) melts to the isotropic center a radial hedgehog needs and gives `r_0 = 6.07` box units, `r_0 sqrt(mu) = 0.607` on your definition (0.460 on a gap-based read), to 0.04 percent in h and 0.25 percent in L. Object A (the absolute weight) keeps the director eigenvalue above 0.8 everywhere below h 0.75, with all three spatial eigenvalues meeting at the plateau edge 0.80: your definition has no crossing there; the gap read gives 2.14 (0.214).
4. The definition, not the numerics, sets your eighth power: `(0.607 / 0.460)^8 = 9`, `(0.460 / 0.214)^8 = 454`, against the 1.5 that 5 percent buys.
5. On the same 3D lattice sum, the tube profiles lie 15 to 28 percent below the R16-1 and R17-2 cores (11.78 vs 13.82, 5.39 vs 7.28, 9.80 vs 13.65), and short 3D descents from them keep going down without leaving the domain: our 3D cores of R16 and R17 (your 0.304 / 0.311 / 0.100) were unconverged states, not blocked by the stop rule.
6. The condensate: no seeded run at `g_W` 1.5 / 1.65 / 1.8 / 2.0 (a core seed at r 2 and a shell seed at r 5, 3000 iterations) condensed; the core seeds at 1.8 and 2.0 and the shell seed at 2.0 left the domain through the director gap. The audit sharpened it: every run is still descending in parallel with the saddle, and the domain's central isolation margin (0.03 to 0.05) is below the seed amplitude and your sextic plateau `s* = 0.112`. A core condensate of your amplitude is not representable inside the domain on which the object is defined. That is the question that returns to you, sharper than the `(lambda, nu, kappa)` one.
7. The Goldstone reads are instrument identities, not condensate numbers: `kin_KP = 4 int rho^2` exactly and the twist stiffness `S = 2 int rho^2 x C_x + 0.34 kin_h` on every field (a uniform synthetic split reproduces both to five digits), so the clock inertia from the a0 read is `2 kin = 9 int rho^2` on any field of this object, and your rotor law `K^2 / 2 I` overestimates the frozen-field `K^2 / 4 kin` ninefold. The fixed-K descents from the nearest-to-condensed field leave the domain within 200 iterations at K 1 and K 5, with `E(K) - E(0)` 39 and 140 times below the rotor law: the rotating core melts at every K from 1 to 200.
8. The pair: on this vacuum the far field of a core is the radial texture, so two cores share one far field. With the director seed `normalize(rhat_1 + rhat_2)` the composite has total charge one at every d: a unit hedgehog with a split core joined by a +1 line disclination with a melted core. `E(d) - 2 E_1` rises linearly with d (a tension of about 1 per box unit, budget-dependent between 0.8 and 1.2), no `1 / d` regime is detectable, the far field carries one unit's `A`, not four, and the free cores contract slowly. Your rule retires the `8 pi` test, and the topology retires it first: a charge-2 object is not representable on the radial far field. Open for the next rung: the linear law extrapolates to `E(d) = E_1` at d 5.7.
9. Not ours: `R~`, and your coupled radial model (its `b` equation and `W` profile). The bundle is still not attached to 22675440.

## The objects as run

Object A: v4 as run in R16-1 (`mu 1e-2, c_P 1, c_s 0.4`, `I_rebuild`, the absolute plateau weight). Object B: v4 with the director-relative weight (R17-2). Object C: v6 as run in R17-3 (`(nu, kappa) = (1e-2, 0.4)`, `W = [(1 - lambda_1) / (1 - delta)]^2`, `c_s 0.5`, the relative weight); on a split-free profile its radial sector is object B's (`U_v6 = 0`, checked to 1e-11 on every end profile).

## The instrument (selftest 20/20, SHA-pinned)

| Extension | Definition | Gate |
| --- | --- | --- |
| the weighted static action | `E_w = sum_c w_c e_c(M)`, `e_c` the R16 per-cell static density, the adjoint weighted at the cell | equals the lattice action at `w_c = h^3` (1e-16, gradient 9e-13); complex step 1e-14 |
| the tube | `M = Lambda(beta, rhat) [m_g e_0 e_0^T + lambda_1 rhat rhat^T + lambda_23 (I - rhat rhat^T)] Lambda^T` on a one-ray `(2 N_g, 3, 3)` lattice whose transverse neighbors carry the ansatz's values; `E = sum_i 2 pi x_i^2 h e(x_i)`; L-BFGS-B on `(m_g, lambda_23, Delta, beta)` with `Delta >= 1e-3`, preconditioned by the colored complex-step Hessian diagonal | the center line at `(h/2, h/2)` reproduces the 3D lattice's own ray densities exactly (0.0); the profile gradient vs complex step 1.4e-13; the circle trivial on the ansatz (1e-12); `beta = 0` stationary exactly |
| the pinning penalty | `E_pin = (k / 2) sum_mask \|M - M_seed\|^2`, k 10 | k 0 reproduces the free problem; complex step 1e-14 |
| the static twist | `T_theta M`, `theta = q . x` through the field's own frame; `S` from the even part | quadratic to 2e-4; a uniform twist leaves the 8-sample energy unchanged (1e-16) |
| the spin-2 zero counter | the section `S_ee - S_ff + 2 i S_ef` on a lattice shell fitted with `2Y_lm`, the zeros as the windings on a `(theta, phi)` grid, the pole caps `ring_N + 2` and `2 - ring_S` | `Q = xx - yy` four simple zeros of total 4, `Q = zz - (xx + yy) / 2` two index-2 pole zeros, tilted and rotated Q total 4, residuals under 1e-3 |

Code: [`m5_32_r18_common.py`](https://github.com/openwave-labs/openwave/blob/19808dbb633a995f867b1909f884ef92377cda73/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r18_common.py), [`m5_32_r18_0_closures.py`](https://github.com/openwave-labs/openwave/blob/19808dbb633a995f867b1909f884ef92377cda73/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r18_0_closures.py), [`m5_32_r18_1_radial.py`](https://github.com/openwave-labs/openwave/blob/19808dbb633a995f867b1909f884ef92377cda73/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r18_1_radial.py), [`m5_32_r18_2_goldstone.py`](https://github.com/openwave-labs/openwave/blob/19808dbb633a995f867b1909f884ef92377cda73/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r18_2_goldstone.py), [`m5_32_r18_3_pair.py`](https://github.com/openwave-labs/openwave/blob/19808dbb633a995f867b1909f884ef92377cda73/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r18_3_pair.py); the audits `m5_32_r18_{0,1,2,3}_audit.py` beside them; every number in `data/m5_32_r18_{0,1,2,3}.json` and the audit JSONs.

## R18-1: the radial solve

![](https://raw.githubusercontent.com/openwave-labs/openwave/19808dbb633a995f867b1909f884ef92377cda73/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r18_1_radial.png)

| Read (L 96, h 1.5 to 0.03) | Object A (absolute weight) | Object B (relative weight) |
| --- | --- | --- |
| `E_stat` on the tube | 10.837 / 8.543 / 8.488 / 8.544 / 8.5544 / 8.5547 | 5.768 / 4.646 / 4.312 / 4.209 / 4.1937 / 4.1894 |
| `r_0` (`lambda_1 = 0.8` crossing) | 2.81 at h 1.5, then none | 6.874 / 6.312 / 6.133 / 6.078 / 6.070 / 6.068 (L 48 6.108, L 144 6.074) |
| `r_gap_half` (`lambda_1 - lambda_23 = 0.35` crossing) | 3.39 / 1.75 / 2.01 / 2.12 / 2.137 / 2.141 (Richardson 2.142) | 5.70 / 4.95 / 4.70 / 4.62 / 4.605 / 4.601 (4.600) |
| the center | the gap closes as h (0.098 to 0.0037), the three spatial eigenvalues at 0.8035 | at the gap bound: `lambda_1(0)` 0.387, the isotropic center |
| the two seeds | two basins at h 0.75 (10.34 vs 8.54 at L 96), one basin elsewhere (1e-8) | one basin at every rung (1e-8) |
| `r_0 sqrt(mu)` | 0.214 (gap read; no crossing) | 0.607 (your definition), 0.460 (gap read) |

Gate 3 on the 3D lattice sum: the tube profiles at 11.78 (A, n32) / 5.39 (B, n32) / 9.80 (A, n64) against the 3D cores at 13.82 / 7.28 / 13.65; 60-step 3D descents from the embedded profiles go down (11.78 to 11.65, 5.387 to 5.385) without escape and without breaking spherical symmetry.

## R18-2: the condensate and the Goldstone reads

![](https://raw.githubusercontent.com/openwave-labs/openwave/19808dbb633a995f867b1909f884ef92377cda73/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r18_2_condensate.png)

| `g_W` | core seed (r 2): E minus the saddle, end split, stop | shell seed (r 5): E minus the saddle, end split, stop |
| --- | --- | --- |
| 1.5 | +0.0017, 0.017, max_iter | +0.046, 0.027, max_iter |
| 1.65 | +0.0064, 0.020, max_iter | +0.041, 0.033, max_iter |
| 1.8 | escape (d) at 300 | +0.034, 0.033, max_iter |
| 2.0 | escape (d) at 400 | escape (d) at 2400 |

| Field | `I = int rho^2` | `2 kin` | `S` | `S / I` | `2 kin / I` |
| --- | --- | --- | --- | --- | --- |
| g_W 1.35 core-seeded | 7.1e-2 | 0.639 | 0.133 | 1.88 | 9.0 |
| g_W 1.65 core-seeded | 1.01e-1 | 0.910 | 0.184 | 1.82 | 9.0 |
| g_W 2.0 shell-seeded | 0.869 | 7.79 | 1.70 | 1.95 | 9.0 |
| g_W 2.0 unseeded control (split 6e-4) | 2.0e-4 | 1.8e-3 | 4.0e-4 | 1.98 | 8.8 |

`E(K)` on the g_W 1.65 core-seeded field: K 1: frozen 0.549, end 0.127 (escape d at 100), the rotor `K^2 / 2 I` 4.93; K 5: frozen 13.7, end 0.870 (escape d at 200), the rotor 123.

## R18-3: the pair

![](https://raw.githubusercontent.com/openwave-labs/openwave/19808dbb633a995f867b1909f884ef92377cda73/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r18_3_pair.png)

| d (n48 L72, pinned) | `E(d)` | `E(d) - 2 E_1` (`E_1` 13.81) | winding on rings rho 0.5 to 1.6 / 1.6 to 2.8 / 2.8 to 4.0 / 4.0 to 5.5 | `A_pair / A_1` |
| --- | --- | --- | --- | --- |
| 12 | 20.010 | -7.61 | 0 / 1 / 1 / 1 | 0.98 |
| 15 | 22.088 (free 21.603, -5.79) | -5.54 | 0 / 1 / 1 / 1 | 0.95 |
| 18 | 26.762 | -0.86 | 0 / 1 / 1 / 1 | 0.93 |
| 24 | 31.299 | +3.67 | 0 / 1 / 1 / 1 | 0.87 |

`E_int = -19.4 + 0.97 d` (rms 0.65; the `A + B / d` form rms 0.99; a `1 / d` part fitted beside the line is `-50 +- 191`). The audit's three-zone split: the core term flat in d, the string zone carrying all the d dependence, the far zone the one-texture offset. The n64 L96 box at 300 iterations reproduces the string and the one-charge far field, not an L read.

## Not computed

The 3D lattice minimum of either object; a condensate representable inside the plateau weight's domain; `E(K)` at K 2 and 10; the anti-pair (not constructible on a degree-1 boundary); the pair below d 12 or on a uniform-director boundary; your coupled radial model; `R~`.

## The audit record

| Stage | Claims | Confirmed | Qualified | Refuted | What the auditor did |
| --- | --- | --- | --- | --- | --- |
| R18-0 | 5 | 2 | 2 | 1 | own polarization-identity `H_00`, fifteen basis readings (the coordinate basis on the exact axis reproduces your pattern: the refutation), own `r_0`, own chart-free spin-2 counter on five synthetic sections |
| R18-1 | 6 | 2 | 4 | 0 | own ansatz on the 3D lattice, own radii and Richardson, minimum tests in ten directions, own 3D embeddings and two 60-step 3D descents |
| R18-2 | 5 | 2 | 3 | 0 | own eigenvalues, own twist (2e-12), own K 50 reads, the analytic anchors on synthetic splits, the traces re-analyzed |
| R18-3 | 6 | 0 | 6 | 0 | own energies, three fits, interpolated rings, the far-sphere degree (1.00), the per-core degrees, the three-zone split, the core centroids |

The record with every number: the [task record](https://github.com/openwave-labs/openwave/blob/19808dbb633a995f867b1909f884ef92377cda73/openwave/xperiments/m5_liquid_crystal/research/tasks/m5_32_task_details.md) R18 section, the [method note § 16](https://github.com/openwave-labs/openwave/blob/19808dbb633a995f867b1909f884ef92377cda73/openwave/xperiments/m5_liquid_crystal/research/findings/m5_32_method_note.md), the [ledger § 6.7](https://github.com/openwave-labs/openwave/blob/19808dbb633a995f867b1909f884ef92377cda73/openwave/xperiments/m5_liquid_crystal/research/findings/m5_32_candidate_ledger.md).


===== COMMENT [27] xrodz 2026-09-10T15:41:50Z (top-level) =====

# R19 planned: the mixed bracket checked at the form level before any run, and the one question that decides what it can change

## TL;DR

1. The proposal is read (the comment here and the list message). Before any relaxation we checked what it is algebraically (the script pinned below, 10 checks that can fail, audited by a second agent with its own code, 8 of 8 confirmed and one qualified): the anticommutator `G_mu nu = A_mu eta A_nu + A_nu eta A_mu` is symmetric in the derivative pair as well as the internal pair, so it is not a 2-form field strength and not Savvidy's structure ("antisymmetric in the first two indices, symmetric in the rest"); its square is a genuinely new operator, `GG = 4 PP - I1` with `PP` the norm of the product `A_mu eta A_nu`, outside everything R1 and R14 screened (rank 8 to 9 against the frozen basis).
2. "The time / gravity sector" has three readings and they differ in what they can change. Keyed on the time DERIVATIVE index, the action is identical to the certified one on every static configuration, so no static Newton read can move and it acts only on the clock. Keyed on the internal BOOST block, the Coulomb sector is unchanged identically (the G1 gate holds by construction) and the boost block becomes the M5.21.16 variant-A flip plus four times a new product norm. Applied everywhere, it changes the Coulomb sector too.
3. Every reading is quartic in omega on the rigid clock (the certified action is quadratic), so `J = 2 kin omega` and `E_J = E_stat + J^2 / 4 kin` have to be restated before any fixed-J number is quoted.
4. R19 as planned: R19-0 the registry entrants and R1's certificate re-run with the new columns (a coefficient window or an exact infeasibility, per reading); R19-1 the static Newton read on the boost-dressed pair (the R3 instrument) under the boost-block reading; R19-2 the clock under the quartic action, with the two-clock cross inertia under the time-index reading.
5. What we need from you before the go: which reading is meant (and whether the `(0, 0)` corner, `G_00 = 2 A_0 eta A_0`, belongs to the gravity sector), and if your own runs of the proposal exist, the script they ran so that the two sides compare the same object. We hold the run until then.

## 1. What the check settles

Notation: jets `A_mu = d_mu M`, `P_mu nu = A_mu eta A_nu`, the certified `F = P - P^T`, the anticommutator `G = P + P^T`, `<X, Y>_eta = sum_ab eta_a eta_b X_ab Y_ab`, and the densities with the FULL derivative contraction `(1/2) sum_{mu nu} eta^mu eta^nu <X_mu nu, X_mu nu>_eta` (the `sum_{mu < nu}` form is a complete contraction only for objects antisymmetric in the derivative pair; on `G` it is not boost invariant).

| Check | Result |
| --- | --- |
| C1 symmetries | `F` antisymmetric in both pairs; `G` symmetric in both; `F eta` is in `so(1, 3)` |
| C1 the "mixed" symbolic result | index placement: `(A eta)(B eta) + (B eta)(A eta)` is not symmetric, the covariant `A eta B + B eta A` is |
| C2 identity | `G^2 + F^2 = 2 (P^2 + P^T2)` entrywise, so `GG + I1 = 4 PP` to 2e-16; the `sum_{mu < nu}` half-sum of `G` is not even rotation invariant and is exactly quadratic in omega |
| C3 span | `GG` outside `{I1..I6, C6a, C6b}`: rank 8 to 9; exactly `GG = 2 J - I1` with `J = sum eta^mu eta^nu tr(eta A_mu eta A_nu eta A_nu eta A_mu)` the new direction |
| C4 invariance | `GG` Lorentz invariant to 6e-16; the fixed-frame boost-block split is rotation invariant, not boost invariant (0.96) |
| C5 time-index reading | equals `I1` exactly on every static configuration (`G_0mu = 0`) |
| C6 boost-block reading | equals `I1` exactly on the Coulomb sector (time row zero) |
| C7 boost-block reading | on static fields with a time row: `B = I1 - 2 I(F_t) + 4 I(P_t)` exactly, `I(F_t) = -sum F_ij[0, k]^2` the boost block of `I1` (a signature identity); on the unit sphere of jets the density's infimum is -0.5 for `I1` and -1.0 for the boost-block reading: indefinite in the same kind, twice as deep |
| C8 omega degree | `I1` quadratic; `GG`, both splits quartic, no odd powers; the quartic comes from the diagonal `G_00` block |

Two remarks the check supports. The spin statement is exact for the internal content: a jet proportional to `eta` drops out of `F` and not of `G`, while traceless-symmetric jets do enter `F` (so "the trace and traceless-symmetric parts never enter the field strength" is half right). And every term here is quartic in the jets, so it has no quadratic fluctuation about the constant vacuum (your § 33.1): the static interaction the record measures is between textures at nonlinear order (the boost dressing), not a linear exchange whose spin fixes the sign. Whether the even-spin content sets the sign of a texture-texture force is what R19-1 measures. One more fact from the audit: the space of covariant bilinears in `(A_mu, A_nu)` antisymmetric in the derivative pair and symmetric in the internal pair is one-dimensional, `tr(A_mu eta) A_nu - tr(A_nu eta) A_mu`, and it vanishes on eta-traceless jets; a Savvidy-type field strength on this field needs an epsilon, a fixed vector, or the field's `u`.

## 2. The three readings

| Reading | Static Newton reads (R0, R3, R11) | Coulomb sector | What it can change |
| --- | --- | --- | --- |
| (T) the time derivative index | identical | identical | the clock only (omega sector) |
| (B) the internal boost block, fixed frame | changes | identical | the boost dressing; not boost invariant |
| (B-u) the boost block on the field's timelike eigenvector `u` | changes | identical (`u = e_0` there) | covariant, field dependent, the case R1 left open |
| (A) everywhere | changes | changes | everything, a control |

## 3. R19 as planned

| Stage | What runs | Pre-registered outcomes |
| --- | --- | --- |
| R19-0 | the four objects enter the registry (sympy and numpy, complex-step gradient, a mutant that reddens; the C2, C5, C6 identities as gates); R1's channel certificate with the new columns: `H2` PSD, the `omega^4` sign, the boost-block sign, the lattice bump descent, at `c` in {0.25, 0.5, 0.75, 1} for (B) and (B-u) and at the literal (A), (T) | a window: `CANDIDATE` for R19-1; an exact infeasibility: `CANDIDATE_REFUTED` at the form level; (T) needs no static certificate |
| R19-1 | the R3 boost-dressed pair and the R11 same-sign instrument under (B) and (B-u), n32 L48 and n48 L72, d in {12, 18, 24, 30}: `E_int(d)`, the far-field fit, the block split, the dressing amplitude trend, the single dressed hedgehog's existence | `dE_int / dd > 0` with the dressing surviving: `NEWTON_SIGN_REVERSED` on that instrument; repulsive: `CANDIDATE_REFUTED`; unbounded or melted: `CANDIDATE_REFUTED` (no object) |
| R19-2 | the omega polynomial on the R10 core and the R12 ring, the fixed-J read with the quartic Legendre bridge (`J = 2 C omega + 4 D omega^3`), the box ladder L in {48, 72, 96}, the two-clock cross inertia (R3.iii) under (T) at d 12 and 24 | `D > 0` and `C`, `D` saturating: a localized fixed-J clock; `C ~ L`: `CANDIDATE_REFUTED`; `D < 0`: unbounded in omega |
| R19-3 | the R18 carry-overs if the day allows (pinned pairs at d 6 and 9, the converged 3D cores from the tube profiles) | reported as in R18 |

## 4. The list message to Faber

Of the three ideas there, the first is this proposal. An imaginary mass needs a term to carry it before it can run, and the near-constant-frequency reading needs a localized clock, which the record does not have (R4, R7, R10). Both are recorded as author-gated, not as candidates.

## 5. Code

| Item | Pinned |
| --- | --- |
| the pre-registration check (10 lines that can fail) and its audit (a second agent, own code) | [`m5_32_r19_precheck_audit.py`](https://github.com/openwave-labs/openwave/blob/23459f8a/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r19_precheck_audit.py), [`m5_32_r19_precheck.py`](https://github.com/openwave-labs/openwave/blob/23459f8a/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r19_precheck.py), its output [`m5_32_r19_precheck.json`](https://github.com/openwave-labs/openwave/blob/23459f8a/openwave/xperiments/m5_liquid_crystal/research/data/m5_32_r19_precheck.json) |
| the R19 proposal, every gate stated before the go | [candidate ledger § 6.8](https://github.com/openwave-labs/openwave/blob/23459f8a/openwave/xperiments/m5_liquid_crystal/research/findings/m5_32_candidate_ledger.md) |

The R18 open items (the pinned pairs at d 6 and 9, the converged 3D cores, the `I = int b^2` and phase-stiffness reads on a condensed core) stay queued behind your reply to the R18 post.


===== COMMENT [31] xrodz 2026-09-11T21:52:11Z (top-level) =====

# R19 goes on the boost-sector reading: what rev 294 settles, what it does not, and the three outcomes the run can return

## TL;DR

1. The reading is answered: rev 294 § 255.4 states the object, `S^gg_mu nu = {Gamma^g_mu, Gamma^g_nu} - trace` with the commutator kept for the spatial sector, `Gamma^g` the boost 3-vector of the paper's connection (eq. (7)). On the 4x4 field that is the anticommutator on the boost-boost INPUT part of `A_mu eta A_nu` in the field's eigenframe: the jet `d_mu M = O ([Gamma_mu, D] + d_mu D) O^T` puts `Gamma^g` exactly on the jet's time row, with the shape factors `lambda_i - lambda_0`. It is covariant by construction (`Gamma` is Lorentz invariant) and it enters R19-0 as the literal entrant beside the field-level readings we posted. The run starts now; the results post follows in the usual shape.
2. Three statements in the two comments that our record contradicts or sharpens. (a) "already in the R1 basis": R1's basis is F-built (`I1..I6` are contractions of the commutator and of its trace), the product norm `tr(A^2 B^2)` is outside it (rank 8 to 9 in the pre-check we posted), and § 255.5 itself says the boost sector "must be recomputed". (b) The flipped `R~^2` of § 260: on the certified contraction the boost quadratic is already negative (the `omega^2` lead `-2 Sum (Gamma~)^2`, our M5.21.16 baseline and R14-A), the flip makes it positive and closed the runaway channel, at the price of a 26 percent Lorentz break of the energy; a script pin would settle which `H` § 260 flipped. (c) The vacuum: our runs sit on the paper's split vacuum `(g, 1, delta, 0)` since R0, where § 268 says the clock mode exists.
3. Nothing in rev 294 is a computation with the boost bilinear: "attraction" is the free scalar / vector propagator sign (§ 220, and the `48^3` scalar toy of § 233), `1/d^5` is the derivative ladder by counting (§ 256.6 says so), and the frequency is `omega = K / I` with `K = 1/2` imposed on a rotor whose inertia is taken from a static profile (§ 269.7). The report records each of these in its own adversary blocks. So the list headline is ahead of the comment: R19-1 is the first pair energy under this bracket on any stack, and R19-2's box ladder is the only place "finite frequency" can mean something on this field (the rigid Goldstone's inertia is L-extensive on the certified action, R4 / R7 / R10 / R12; R13-W proved no fixed-J minimizer on it).
4. Pre-registered. R19-1, the boost-dressed pair (the R3 instrument, n32 L48 and n48 L72, d 12 to 30), the far-field fit over `{1/d, 1/d^3, 1/d^5, (ln d) / d}`: `NEWTON_SIGN_REVERSED` (attractive AND exponent 1), `ATTRACTIVE_SHORT_RANGE` (attractive, exponent 3 or 5: your report's own prediction, not Newton), `CANDIDATE_REFUTED` (repulsive, or no object). R19-2, the clock under the quartic-in-`omega` action: `C(L)` and `D(L)` saturating, or `~ L`. Before both, R19-0's form-level certificate on the literal object and on the covariant field-level reading (`H2` PSD, the `omega^4` sign, the boost-block sign, the bump descent), plus four cheap identities: the literal object built two ways agrees, its relation to the field-level readings, the fate of eq. (9)'s EM-GEM cross term under the replacement (a vector has no bilinear invariant with a traceless symmetric tensor), and the core finiteness of a connection-level density on the hedgehog.
5. Not in R19, noted for after: `(inc M)^2` (quadratic in `M`, four derivatives: the first class with a quadratic fluctuation about the constant vacuum, so it needs a dispersion and ghost read before any Newton read), the boost-sector source-contracted propagator (on our stack the Hessian about the relaxed hedgehog on the boost channel, a linear-response read beside R19-1's nonperturbative number), and the bounce. Your § 257.6 asks (`kappa_6 / mu`, `g_W / mu`, the `l = 2` shell) stay queued with the R18 carry-overs.

## The object, once

Notation as in the plan post. In the eigenframe `M = O D O^T`, `D = diag(-g, 1, delta, 0)` far away, `Gamma_mu = O^-1 d_mu O`, and the jet is `A_mu = O ([Gamma_mu, D]_eta + d_mu D) O^T`. The boost generator `K(a)` with `(0, i)` entries `a_i` gives `[K(a), D]` the `(0, i)` entries `a_i (lambda_i - lambda_0)`: the boost content of the connection is the time row of the jet in the field's own frame, and nothing else lands there. So the literal proposal is

```text
P_mu nu = A_mu eta A_nu,   A = A^t + A^s + A^d   (time row, spatial off-diagonal, diagonal; in the u-frame)
F^(Gamma)_mu nu = (P - P^T)  -  (P^tt - P^tt T)  +  (P^tt + P^tt T),   P^tt = A^t_mu eta A^t_nu
density  (1/2) sum_{mu nu} eta^mu eta^nu <F^(Gamma), F^(Gamma)>_eta,   and the traceless variant on P^tt + P^tt T
```

which keeps every commutator that is not boost-boost. On the Coulomb sector (`A^t = 0`) it is the certified action identically; on a static boost dressing it differs from the certified action by the boost-boost block only; on the rigid clock it is quartic in `omega` like every reading we posted. The ledger section with the full revised ladder: [§ 6.8](https://github.com/openwave-labs/openwave/blob/993540c5/openwave/xperiments/m5_liquid_crystal/research/findings/m5_32_candidate_ledger.md).


===== COMMENT [32] xrodz 2026-09-12T11:31:27Z (top-level) =====

# R19 result: the boost-sector anticommutator is a new covariant object, it leaves the clock as the record has it, and its dressed like pair repels

## TL;DR

1. The reading of rev 294 § 255.4 is realized exactly on the 4x4 field. With `M = O D O^T` and `Omega = O^-1 dO` in so(1, 3), `dM = O (Omega D + D Omega^T + dD) O^T`, and the boost content of the connection is the jet's time row in the field's own frame with the factors `lambda_i + lambda_0` (9, 8.3, 8 at the vacuum, never zero at a melted core), so `S^gg` is the anticommutator on the boost-boost input part of `A_mu eta A_nu`, and the entrant is `F^(Gamma) = F + 2 (P^tt)^T` with `P^tt = A^t eta A^t`. Built two ways it agrees to 1e-14; it is Lorentz covariant to 1e-13 and independent of the lift. Audited by a fresh agent with its own lift and its own eigen-solve (4 CONFIRMED / 2 QUALIFIED / 0 REFUTED on 34 lines).
2. It is new and it kills the EM-GEM cross term. The ranks on random jets are 8 for the F-built basis, 9 with the pure anticommutator, 10 with the internal split, 11 with (Γ), 12 with its traceless literal (so "already in the R1 basis" holds only for the spatial sector; the F-built basis does not contain the product norm). Eq. (9)'s `R^ee . R^gg` cross term (opposite in sign between SO(4) and SO(1, 3), as your § 254 says) is EXACTLY zero under the replacement at the connection level and at the field level with the shape factors: `<F - X_a, X_s>_eta = 0` by the parity of the derivative pair, so `I(F^(Gamma)) = I(F - X_a) + I(X_s)`.
3. It cannot cure the clock, at the form level. On the general 30-parameter static jets, every entrant's kinetic form keeps the certified action's minimal eigenvalue on every Lorentz channel (`-16 (lambda_k + g)^2` on a boost clock, `-8 (lambda_i - lambda_j)^2` on a rotation clock), because the negative direction is the diagonal jet `d_k M_00 = d_k M_kk`, with no time row, and it lies in the kernel of `Q_Gamma - Q_I1`: no mixing coefficient in [-200, 200] makes `-4 [(1 - c) I1 + c X]` positive on the six channels for any of the five entrants (R1's certificate with five new columns). On a rotation clock at the vacuum (Γ), its traceless literal and the internal split are IDENTICAL to `I1` (`Q` and the `omega^4` coefficient), so on a Coulomb-sector core the rotation clock is the record's by identity (`C(L) ~ L`, R4 / R7 / R10 / R12); on a boost clock every entrant has `D = (1/2) <X_00, X_00> > 0` (`D_Gamma = 4 (1 + g)^4`), and with the certified `-4` sign the rigid-clock Hamiltonian `H = 4A + V4 - 4C omega^2 - 12 c D omega^4` is unbounded below in `omega`. "Finite nonzero frequency" has no content on this field under the replacement.
4. The static Newton read, on the R3 boost-dressed like pair (the M5.21.14 dressing twice, relaxed, the same protocol and budget as R3, this code reproducing R3's certified rows to every printed digit): at the instrument's vacuum g 32 the dressed pair under (Γ) is REPULSIVE and monotone, `E_int` +35555 / +29144 / +14841 / +4114 at d 10 / 14 / 18 / 24 against the certified +3869 / +2989 / +2987 / +2784; the interaction is the anticommutator overlap of the two non-decaying dressings (`I(X_s)(pair) - 2 I(X_s)(single)` reproduces `E_int` within 3 percent), ten times the certified repulsion at d 10, and it fits `1 / d` (R^2 0.91, the coefficient positive) and `ln d / d` (0.986), not `1 / d^5`; the traceless literal repels on two points (+12240, +3042). Outcome under the pre-registered rules: `CANDIDATE_REFUTED (repulsive)`, with two qualifications the audit insisted on: the pre-registered point was g 8 (where the instrument itself is unbounded, item 5), and the (Γ) descents are still falling at the R3 budget (last-quarter drifts of order 2e3 to 4e3 against margins of 4e3 to 3e4), so the rows were extended by 3000 accepted steps: after 3000 and then 9000 more accepted steps the values are +32703 / +24317 / +12374 / +3331 and +29364 / +18622 / +9862 / +3276, still positive and monotone, the d 24 value settled, the drift of `E_int` per 1500 steps down to 1e2 against values of 3e3 to 3e4 (the single and the pairs descend together); at the last budget the tail fits `1 / d` with R^2 0.993 and a positive coefficient, a Coulomb-like repulsion, not `1 / d^5`. On the pre-registered n48 L72 box the sign is the same: `E_int` +31381 / +9282 at d 14 / 24 (the certified +3021 / +2828), the single 56905 against 56902 on the small box.
5. At g 8 the instrument itself is unbounded: the certified action's own dressed rows run away first (in 200 to 500 s, the time-row piece `I(F_t)` diving from -172 to -12588 while the rest of `I1` stays of order 100), and so does every entrant; the (Γ) single is the one bounded object there (its anticommutator piece relaxes from 367 to 61). The kills are the record's negative boost weight (the `R^eg` time row, which the replacement leaves untouched: `F^(Gamma)` has the time row of `F`), not a new instability of the object. The runaway attribution is measured on every saved end field, not inferred.
6. Overnight, the R18 carry-over you asked about (does the string law bend below d 12): the pinned like pairs at d 6 and 9 on the R18-3 instrument give `E_int` -12.59 and -10.53, on the R18 line (the six-point law `E_int = -18.66 + 0.935 d`, rms 0.59): no bend, and the split-core object at d 6 costs more than the radial single.
7. Not done here, for after: `(inc M)^2` (quadratic in `M`, four derivatives, the first class with a vacuum dispersion to read), the boost-sector source-contracted propagator (on our stack the Hessian about the relaxed hedgehog on the boost channel), the bounce; the remaining R18 carry-overs (the converged 3D cores from the tube profiles, the finer-h pair box, a condensate on a domain that holds it) and your § 257.6 asks stay queued. The full record: the task record R19 section, the method note § 17, the ledger § 6.8 outcome (links below, at the commit).

## The objects (equations first)

```text
M = O D O^T, O Lorentz (O eta O^T = eta), D = diag(lambda_0..3), vacuum diag(g, 1, delta, 0)
Omega_mu = O^-1 d_mu O in so(1,3): Omega eta antisymmetric (symmetric time row a = the boost 3-vector, antisymmetric spatial block)
d_mu M = O (Omega_mu D + D Omega_mu^T + d_mu D) O^T
   frame: A~[0,i] = a_i (lambda_i + lambda_0),  A~[i,j] = Omega_ij (lambda_j - lambda_i),  A~[i,i] = d lambda_i
u = the timelike unit eigenvector of M eta (u^T eta u = -1), Pi_u = -u u^T eta, Pi_s = 1 - Pi_u
A^t = Pi_u A Pi_s^T + Pi_s A Pi_u^T,   P^tt_mu nu = A^t_mu eta A^t_nu,   X_a = P^tt - P^tt^T,   X_s = P^tt + P^tt^T
(Gamma):     F^(Gamma) = F - X_a + X_s = F + 2 (P^tt)^T
(Gamma, tl): F - X_a + T_tl(Pi_s X_s Pi_s^T),  T_tl(Y) = Y - (1/3) tr(eta Y)(eta + u u^T)
(B-u):       Pi_s F Pi_s^T + Pi_u G Pi_s^T + Pi_s G Pi_u^T + Pi_u G Pi_u^T,  G = P + P^T
density I(X) = (1/2) sum_{mu nu} eta^mu eta^nu <X_mu nu, X_mu nu>_eta;   L(c) = -4 [(1 - c) I1 + c X] - V4
rigid clock A_0 = omega a0: I = A + C omega^2 + D omega^4;  H = 4A + V4 - 4 C(c) omega^2 - 12 c D omega^4
instrument: E_int(d) = E(pair) - 2 E(single), the R3 arm (ii) seeds, FIRE, pinned shell, 1500 accepted steps
```

## Equation-to-code map (pinned to `2721a93d7f055da67c740961cb1e72faba55c1c5`)

| Equation | Code |
| --- | --- |
| the time-row map, (Γ) two ways, covariance, the span, eq. (9)'s cross term, the omega degree, the core ladder | [`m5_32_r19_c9_c12.py`](https://github.com/openwave-labs/openwave/blob/2721a93d7f055da67c740961cb1e72faba55c1c5/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r19_c9_c12.py) (17 / 17), [JSON](https://github.com/openwave-labs/openwave/blob/2721a93d7f055da67c740961cb1e72faba55c1c5/openwave/xperiments/m5_liquid_crystal/research/data/m5_32_r19_c9_c12.json) |
| the 30-jet kinetic forms, the `omega^4` coefficient, the `c` windows | [`m5_32_r19_certificate.py`](https://github.com/openwave-labs/openwave/blob/2721a93d7f055da67c740961cb1e72faba55c1c5/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r19_certificate.py) (17 / 17), [JSON](https://github.com/openwave-labs/openwave/blob/2721a93d7f055da67c740961cb1e72faba55c1c5/openwave/xperiments/m5_liquid_crystal/research/data/m5_32_r19_certificate.json) |
| the five entrants with exact gradients (the eigenframe chain), the gates | [`m5_32_r19_entrants.py`](https://github.com/openwave-labs/openwave/blob/2721a93d7f055da67c740961cb1e72faba55c1c5/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r19_entrants.py) (13 / 13) |
| the pair runs, the reads, the fits, the outcomes, the sector attribution | [`m5_32_r19_1_pair.py`](https://github.com/openwave-labs/openwave/blob/2721a93d7f055da67c740961cb1e72faba55c1c5/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r19_1_pair.py), [JSON](https://github.com/openwave-labs/openwave/blob/2721a93d7f055da67c740961cb1e72faba55c1c5/openwave/xperiments/m5_liquid_crystal/research/data/m5_32_r19_1_pair.json) |
| the independent audits (own lift, basis, Gram matrices, fields, stencils; own energies and fits) | [`m5_32_r19_0_audit.py`](https://github.com/openwave-labs/openwave/blob/2721a93d7f055da67c740961cb1e72faba55c1c5/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r19_0_audit.py), [`m5_32_r19_1_audit.py`](https://github.com/openwave-labs/openwave/blob/2721a93d7f055da67c740961cb1e72faba55c1c5/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r19_1_audit.py) |

## The certificate (g 8; the same structure at g 32)

| Channel | min eig of `-4 Q_I1` | the same for (Γ), (Γ, tl), (B-u), (A), (T) | `D` for (Γ) / (B-u) / (Γ, tl) / (A), (T) | `c` window |
| --- | --- | --- | --- | --- |
| boost_1 / 2 / 3 | -1296 / -1102 / -1024 | identical | 26244 / 13122 / 8748 / 26244 (boost_1; `4 (1+g)^4`) | none for any object |
| rot_1 / 2 / 3 | -0.72 / -8 / -3.92 | identical; (Γ), (Γ, tl), (B-u) have `Q = Q_I1` exactly | 0 for (Γ), (Γ, tl), (B-u); 0.03 / 4 / 0.96 for (A), (T) | none |

## The pair (g 32, n 32, L 48; this code = R3's rows to every digit)

| Object | `E(single)` | `E_int` d 10 / 14 / 18 / 24 | dressing part | force | best fit | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| (Γ) | 56902 | +35555 / +29144 / +14841 / +4114 | +35479 / +29032 / +14700 / +3940 | repulsive, monotone | `ln d / d` 0.986, `1 / d` 0.910 (B > 0) | `CANDIDATE_REFUTED (repulsive)` |
| (Γ, tl) | 21455 | d 14 +12240, d 24 +3042 | +12129 / +2868 | repulsive (two points) | | `CANDIDATE_REFUTED (repulsive)` |
| certified `I1` | 411 | +3869 / +2989 / +2987 / +2784 | +3793 / +2878 / +2845 / +2611 | repulsive | `1 / d^5` 0.979 | the record (R3) |
| undressed (the static part) | 18.97 | +76 / +112 / +141 / +174 | 0 | rising with d (the like-pair string, R18-3) | | not a Newton read |
| (Γ), n 48 L 72 | 56905 | d 14 +31381, d 24 +9282 | +31255 / +9095 | repulsive (two points) | | the box check: the same sign |
| certified `I1`, n 48 L 72 | 412 | d 14 +3021, d 24 +2828 | +2896 / +2641 | repulsive | | the record's box dependence |

![E_int](https://raw.githubusercontent.com/openwave-labs/openwave/2721a93d7f055da67c740961cb1e72faba55c1c5/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r19_1_eint_n32.png)

![the g 8 kills](https://raw.githubusercontent.com/openwave-labs/openwave/2721a93d7f055da67c740961cb1e72faba55c1c5/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r19_1_pieces_g8.png)

## Audit record

| Stage | Claims | Confirmed / Qualified / Refuted | What the auditor did on its own |
| --- | --- | --- | --- |
| R19-0 | 6 (34 lines) | 4 / 2 / 0 | a different lift (rotation times a Cayley boost), its own eigen-solve, non-symmetric and boosted-`u` jets, its own basis with SVD gaps over six sampling families, Gram matrices without polarization, a `c` search over [-200, 200], its own fields and stencils; QUALIFIED: on static jets the basis has the exact relation `I2 - 4 I5 + I6 = 0`, and the rigid core's `1 / h` is the asymptote of `a / h + b` |
| R19-1 | 8 (23 lines) | 5 / 2 / 1 (as worded) | its own densities for every object on all 44 saved end fields (3e-13 against the rows), the seeds rebuilt, its own fits and outcome rules, the stops and amplitude ratios from the fields, the pin shell and the far spectrum; REFUTED the "within 2 percent" wording (it is 3); QUALIFIED the attribution wording and the convergence at the R3 budget; sharpened that the g 32 point is not the pre-registered one |

## Not computed

`(inc M)^2`; the source-contracted boost propagator; the bounce; a fully converged (Γ) pair (the extended descents still fall slowly, the sign and the far value settled); a dressing profile of its own for (Γ) (the record's `b*` is the certified action's g 32 minimizer, a unit-wavelength saw unresolved at h 1.5); anything at g 8 on this instrument (unbounded for the certified action itself); the remaining R18 carry-overs.


===== COMMENT [35] xrodz 2026-09-13T09:47:22Z (top-level) =====

# R20 goes on the paper's three-axis hedgehog: the three-lepton mechanism on the certified action and on the degree-8 potential, with the topology pre-registered

## TL;DR

1. R20 runs rung 3 as your comment states it: one hedgehog with the winding on each of the three spatial eigenvectors of the split vacuum `(1, delta, 0)`, three static solves on the same lattice and the same box ladder, the three energies read against each other and against Koide. First on the certified static sector `4 I1 + V4`, then on your replacement potential `V_spec = gamma tr[P(Q)^2]` in place of `V4`, the "first thing to test" of § 369. The go follows this post; the results post comes in the usual shape.
2. Your four curvatures reproduce to the last digit (371866.88 / 48.02 / 5.229 / 11.52) with the roots `(g, 1, delta, 0)`, which is the spectrum of `M eta` in our embedding, so the report and this stack share the convention. But § 370's "the construction is then unique" does not hold: the certified `V4 = w sum_p (tr((M eta)^p) - C_p)^2`, `p = 1..4`, is itself degree 8 in `M`, a sum of squares vanishing exactly at the target spectrum (the four power sums fix the four eigenvalues), nonnegative with one free parameter, and it has been the potential of this stack since R0. The two potentials differ in their spectral Hessians (ours full rank with the `g` direction 7e7 times stiffer than the softest, yours diagonal with a ratio 7e4), which is exactly what R20-2 measures on the same three objects. On this stack "the one defect blocking every solve" never existed: the vacuum has been a strict spectral minimum of `V4` throughout.
3. Two readings in the comment that our record corrects. (a) "the rotation clock identical to `I1`, so `omega = K / I = 0.694` stands": the identity says the boost-sector replacement cannot change the record's clock, and the record is R13-W (no fixed-J minimizer on `L_cert`, the inertia box-extensive) and R18-2 (`ROTOR_NOT_MINIMUM`); 0.694 is the report's rotor at the report's couplings, never measured on this field. (b) "Newton finally got right sign" (the list mail): the sign in the report and in the audit README is the free-propagator exchange sign (§ 219, `round504`), a form-level statement; the only pair energy under the boost bilinear on any stack is R19-1's, repulsive at every budget and box, which § 367 accepts. The audit README's "a quartic kinetic term gives a linear potential" matches what this stack measured where a background gradient exists: the like pair on the degenerate vacuum is a string, `E_int = -18.66 + 0.935 d` over six points to d 6 (R18-3, R19-3); on the flat vacuum `L_cert` has no quadratic fluctuation at all (R14-0), so no helicity propagates there and the TT question of your open calculation 1 needs the background-gradient propagator, a new instrument.
4. Pre-registered against the reading, not against the run. On a vacuum with all four eigenvalues distinct the order-parameter space is `SO(3) / D_2` with `pi_2 = 0` (R10), so nothing protects any of the three windings; the paper's protection lives on the uniaxial `(1, delta, delta)`, where the transverse pair is degenerate and § 363.6's "biaxiality is essential" is the opposite requirement. And the transverse pair has a line defect on the polar axis in every construction, the split core of R16 to R18 and the R18-3 string seen from one object. So the outcomes per axis are `AXIS_ESCAPES` (the eigenvector's degree on the outer shells drops to zero), `AXIS_STRING` (`E_k` linear in the box), `AXIS_CONVERGED`; and for the triple `THREE_AXES_DISTINCT` (ordered energies, the ratios and Koide `Q` reported) or `AXES_DEGENERATE`. Koide is a read, not a gate: your § 364 shows `Q = 2/3` is reachable at ratios 1 : 4.5 : 162, and the report's masses include a clock energy this field does not have.
5. Not in R20: the `l = 2` shell (a radial-BVP object on your side), the Lorentz / Magnus force (dynamics; no integrator here), the photon with transverse structure, rung 2's loop (a closed disclination loop of the transverse pair with its `E(R)` is an R21 candidate on this stack), the baryons (your next focus). `(inc M)^2`, the source-contracted boost propagator and the bounce stay parked. The zenodo record holds the 774-page rev 386 only: neither §§ 367 to 370 nor the audit's §§ 465 to 496 nor any `round*.py` is online, so those claims stay evidence here until the bundle has a link.

## The objects, once

The record's seed is `M3 = n n^T + delta phi phi^T` with `n` radial and `phi` azimuthal, isotropic-blended at the core (`r_c = 4`), embedded with `M_00 = -g`; the three-axis seeds assign the eigenvalues `(lambda_r, lambda_phi, lambda_theta)` to the radial, azimuthal and polar directions:

```text
S_1     (1, delta, 0)       the record's electron (the transverse splitting delta)
S_d     (delta, 1, 0)       the delta-axis object (the transverse splitting 1)
S_0     (0, 1, delta)       the 0-axis object (the transverse splitting 1 - delta)
S_k'    the swapped transverse assignment of each (the control on the smallest box)
```

## Pre-registered

| Stage | What runs | Reads | Outcomes |
| --- | --- | --- | --- |
| R20-0 the form level | the full `sym4` Hessians of `V4` and `V_spec` at the vacuum; `V_spec`'s exact lattice gradient FD-gated at the R19 bar; `gamma` fixed by matching the record's relaxed single's potential energy under both; the flat-vacuum quadratic form of `L_cert` on the 30-jet space; the measured pair laws tabulated against the linear-potential statement | the four spectral curvatures of each, the six conjugation zeros, the FD residual, `gamma` | facts, audited |
| R20-1 the three axes on `4 I1 + V4` | g 8, delta 0.3; the R3 / R19 FIRE with the pin shell and the kill rules; 4500 accepted steps (three times R3's) with the gate CONVERGED iff the last-third energy drift is below 1e-3 and `max abs G` fell two decades, else FALLING and the budget doubled once; n32 L48, n48 L72, n64 L96 at h 1.5 | `E_k(L)`, the degree of each eigenvector on the shells r 6 / 9 / 12, the eigenvalue gaps and biaxiality on the polar axis and in the core, the `E(L)` slope | per axis `AXIS_ESCAPES` / `AXIS_STRING` / `AXIS_CONVERGED`; the triple `THREE_AXES_DISTINCT` / `AXES_DEGENERATE` |
| R20-1 controls (n32 L48) | the three swapped assignments; the degenerate vacuum `(g, 1, delta, delta)` for `S_d` and `S_0` (they must coincide there); `S_1` at g 32 (R3's record 18.970 at n32) | the same | the nulls of the mechanism |
| R20-2 the three axes on `4 I1 + V_spec` | `gamma` from R20-0, n32 L48 and n48 L72, the same gate and reads | the same, plus `V_spec` against `V4` per axis | the same vocabulary |

Sizing: 22 jobs on 12 workers, one overnight; the n32 and n48 rows within 8 hours of the go, the n64 rows the long pole. Every quoted number carries its CONVERGED or FALLING label; every stage is audited by a fresh agent with its own scripts before the results post.

The ledger section with the plan check and the full packet: [§ 6.9](https://github.com/openwave-labs/openwave/blob/5dd2cc04/openwave/xperiments/m5_liquid_crystal/research/findings/m5_32_candidate_ledger.md).


===== COMMENT [39] xrodz 2026-09-14T00:28:41Z (top-level) =====

# R20: the three-axis hedgehog gives one compact electron, one polar string and one boundary-dominated object on the certified static sector, not three intrinsic energies; the box sets the electron's size at W1, and every undressed minimum is a saddle along the boost dressing

## TL;DR

1. The three seeds `S_1 (1, delta, 0)`, `S_d (delta, 1, 0)`, `S_0 (0, 1, delta)` on `4 I1 + V4` (g 8, n32 L48 and n64 L96, 4500 accepted steps, the n32 rows extended to 9000) end at 13.1 / 83.7 / 30.9 (n64) and 9.8 / 60.9 / 22.8 (n32, 9000 steps), every row still FALLING; the ordering `S_1 < S_0 < S_d` holds on every box and budget, the values do not converge, and the two-rung rule returns `AXES_DEGENERATE` by resolution (Koide Q 0.38, a read).
2. All three keep their winding under `V4` (solid-angle degree 1.000 on every cube; the audit corrected our Mermin read of 0.6 for `S_d`, under-resolved at its near-degenerate axis). Only `S_d` is a string: its box increment (+20.1 at equal budget) equals the tube prediction and is interior growth, with a tension of 0.12 per unit length in the tube's flat middle; `S_1`'s and `S_0`'s increments sit inside what one extension moves, so their `AXIS_STRING` labels are withdrawn.
3. The pinned Dirichlet shell holds the seed's line defect: 72 percent of `S_d`'s energy (45.5 of 63.6, the same 45.7 at n64) and 36 percent of `S_0`'s sit in the boundary layer (`S_1`: 7 percent). Your matching-radius check, at equal budgets: E(< R) agrees across the boxes to 0.2 to 0.6 percent out to R 21 for `S_1` and R 15 for `S_d`, so the interiors are box-independent and the differences between the three totals are exterior and boundary (the far-field tail, the string, the frozen shell); for `S_0` no radius matches because its two descents sit at different stages.
4. Every relaxed object wants to dilate: `dE / d lambda` is negative on every row and the virial `E_curv / V` is 14 to 80 (3 at a quartic + potential equilibrium). For the compact electron the derivative is within 1.2 to 1.7 of the Derrick value and `R_* = r (E_curv / 3V)^(1/4)` is 13 to 18 in a box of half-width 24: at `W1 = 7.2e-4` the electron's size is the box's, which is why nothing converges (your section 579 and 589.4 step 1, in the static sector). For `S_d` and `S_0` half of that derivative is the frozen shell, so their dilation numbers read the boundary as much as the field.
5. Under your `V_spec` (gamma 4.742e-4 from matching the record single's potential energy) the descents converge to machine precision by leaving the target spectrum (the outer shells at (0.36, 0.38, 0.98), the transverse pair degenerate); the electron keeps its winding (E 4.885), `S_d` and `S_0` unwind (`AXIS_ESCAPES`), as `pi_2 = 0` on the biaxial vacuum predicts.
6. Every undressed minimum is a saddle along the record's boost dressing: `d2E / ds2` from -31 to -1672 with both potentials silent to 1e-13. Your items 1 and 4 of 18:52 UTC, measured on our fields; R19's RUNAWAY rows and your section 590 from the static side.
7. R20-0 corrects our own plan check: your four curvatures 371866.88 / 48.02 / 5.229 / 11.52 reproduce on the roots (8, 1, 0.3, 0), which is this stack's s = +1 spectrum; the certified branch has N-spectrum (-8, 1, 0.3, 0), where `V_spec` gives 714251 / 79.4 / 6.08 / 11.52. The uniqueness claim of section 370 stays refuted (`V4` is a second degree-8 member). Audited 40/40 by a fresh agent with exact sympy Hessians.
8. Controls: the transverse swaps end within 1.5 percent of their partners; the degenerate-vacuum trio has one winding object (`S_1^dd`, E 3.99 against the biaxial 9.78) and two textures without a winding eigenvector (16.55 / 17.21, the null within 4 percent); the g 32 control reproduces R3 (18.963 vs 18.970).

## Equations and objects

Field `M(x)` real symmetric 4x4, `eta = diag(-1, 1, 1, 1)`, `N = M eta`; code branch s = -1: `M_vac = diag(8, 1, 0.3, 0)`, N-spectrum `(-8, 1, 0.3, 0)`. The static energy `E = 4 h^3 sum_br wt sum_cells I1(A) + V`, `A_i = d_i M`, with `V4 = W1 sum_p (tr N^p - C_p)^2` (the certified potential), `V4^dd` with the degenerate targets, or `V_spec = gamma h^3 sum tr[P(N)^2]`, `P(x) = prod (x - q_i)`, with the exact gradient `2 gamma sym4[eta P(N) P'(N)]` (complex-step 5e-15). Seeds: `M3 = lam_r n n^T + lam_phi phi phi^T + lam_theta theta theta^T` (`n` radial, `phi` azimuthal, `theta = phi x n`, the isotropic blend at r_c 4), embedded with `M_00 = 8`; `S_1` is the record's single to 0.0. FIRE with the pinned Dirichlet shell (depth 1.6) at the seed values, the convergence gate CONVERGED iff the last-third drift is below 1e-3 AND `max |G|` fell two decades, else FALLING (the label travels). Reads per row: the Mermin degree of each eigenvector on the r 6 / 9 / 12 cubes, the polar-axis tube energy minus the x / y tubes per unit length, the virial and the direct dilation scan `E(lambda)`, the curvature along the boost dressing `M -> Q M Q^T` at s = 0.02 and 0.05, and `E(< R)` on shells 3 to 42 with the matching radius across the boxes.

## Code map (pinned to `55fcc168b347011ade787493d8aedf2d7bbbb442`)

| Piece | File |
| --- | --- |
| R20-0: the Hessians, the `V_spec` gradient, gamma, the flat-vacuum jets, the trace direction | [`m5_32_r20_0_class.py`](https://github.com/openwave-labs/openwave/blob/55fcc168b347011ade787493d8aedf2d7bbbb442/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r20_0_class.py), its audit [`m5_32_r20_0_audit.py`](https://github.com/openwave-labs/openwave/blob/55fcc168b347011ade787493d8aedf2d7bbbb442/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r20_0_audit.py) (40 / 40) |
| R20-1 / R20-2: the seeds, the potential hook, the FIRE with checkpoints, the reads, the outcomes | [`m5_32_r20_1_axes.py`](https://github.com/openwave-labs/openwave/blob/55fcc168b347011ade787493d8aedf2d7bbbb442/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r20_1_axes.py), its audit [`m5_32_r20_1_audit.py`](https://github.com/openwave-labs/openwave/blob/55fcc168b347011ade787493d8aedf2d7bbbb442/openwave/xperiments/m5_liquid_crystal/research/scripts/m5_32_r20_1_audit.py) (50 own lines, 41 PASS, the 9 FAILs refuting readings, all folded into this post) |
| The results (every row, every read) | [`m5_32_r20_1_axes.json`](https://github.com/openwave-labs/openwave/blob/55fcc168b347011ade787493d8aedf2d7bbbb442/openwave/xperiments/m5_liquid_crystal/research/data/m5_32_r20_1_axes.json) |
| The ledger section | [§ 6.10](https://github.com/openwave-labs/openwave/blob/55fcc168b347011ade787493d8aedf2d7bbbb442/openwave/xperiments/m5_liquid_crystal/research/findings/m5_32_candidate_ledger.md) |

## The energies (static, end of descent)

| Object | n32, 4500 | n32, 9000 | n64, 4500 | E(< 12) n32 / 9000 / n64 | In the pinned shell |
| --- | --- | --- | --- | --- | --- |
| `S_1 (1, delta, 0)` | 11.290 | 9.781 | 13.125 | 7.48 / 6.10 / 7.46 | 0.77 / 0.78 / 0.47 |
| `S_d (delta, 1, 0)` | 63.592 | 60.876 | 83.733 | 5.83 / 4.21 / 5.81 | 45.5 / 45.6 / 45.7 |
| `S_0 (0, 1, delta)` | 30.385 | 22.758 | 30.922 | 11.48 / 6.48 / 6.91 | 11.0 / 11.3 / 11.0 |
| the transverse swaps (n32) | 11.094 / 64.981 / 26.316 | 9.642 / 61.858 / 22.895 | | | |
| the degenerate-vacuum trio (n32) | 4.502 / 17.812 / 18.672 | 3.993 / 16.554 / 17.213 | | | 0.11 / 10.9 / 11.1 |
| `S_1` at g 32 | 18.963 (CONVERGED) | | | 13.90 | 0.76 |
| the three axes on `V_spec` (n32) | 4.885 / 55.856 / 16.444 (CONVERGED) | | | 1.63 / 0.88 / 1.04 | 0.86 / 46.0 / 11.5 |

## The reads

| Read | `S_1` | `S_d` | `S_0` |
| --- | --- | --- | --- |
| winding degree on the r 9 / 12 cubes, `V4` (Mermin flux; solid-angle degree in brackets) | 1.08 / 1.06 (1.000) both boxes | -0.6 / -0.6 (1.000) both boxes: the flux under-resolved at the axis gap 0.01 to 0.08, the winding kept | 1.08 / 1.06 (1.000) both boxes |
| the same under `V_spec` | kept | -0.02 / -0.04 (escapes) | 0.01 / 0.16 (escapes) |
| tube read (per unit length; the flat middle at n64 in brackets) | 0.02 / 0.01 (0.01) | 0.78 / 0.42 (0.12; 62 to 78 percent of the tube excess sits in the one plane next to the pinned shell) | 0.22 / 0.13 (0.06) |
| virial `E_curv / V`; `dE / d lambda` | 14.5 to 27; -13 to -19 | 31 to 54; -392 to -850 | 26 to 80; -104 to -214 |
| `d2E / ds2` along the boost dressing | -87 to -98 | -31 to -34 | -58 to -70 |
| matching radius (2 percent), equal budgets | R 3 to 21 match to 0.2 to 0.6 percent | R 3 to 15 match | none (descents at different stages) |
| the box increment at equal budget, and the two-rung label | +1.8, `AXIS_STRING` withdrawn | +20.1, `AXIS_STRING` supported | +0.5, `AXIS_STRING` withdrawn |

![E(L)](https://raw.githubusercontent.com/openwave-labs/openwave/55fcc168b347011ade787493d8aedf2d7bbbb442/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r20_1_E_of_L.png)

![E(< R) per box](https://raw.githubusercontent.com/openwave-labs/openwave/55fcc168b347011ade787493d8aedf2d7bbbb442/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r20_1_radial.png)

![the descents](https://raw.githubusercontent.com/openwave-labs/openwave/55fcc168b347011ade787493d8aedf2d7bbbb442/openwave/xperiments/m5_liquid_crystal/research/plots/m5_32_r20_1_descents.png)

## What was cut, and what is next

The n48 rung, the n64 extension and the degenerate trio's ladder were cut on the run day to have this post tonight; every n64 number is a 4500-step FALLING number and says so. On the day's comments: Maciej's trace note became R20-0 (f) (the curvature energy is invariant under `N -> N + c(x) 1` to 1e-15, the trace is the potential's alone); the Derrick, frame and radial reads above are the 16:30 and 18:52 UTC items measured on our fields. Items 2 and 3 of the 18:52 UTC comment did not reach the thread (the pasted body jumps from 1 to 4). The three reads we take per rung, for the 008 L-ladder's observables: `E` per box, the degree of each eigenvector on the r 6 / 9 / 12 cubes, the tube read, `E(< R)` on the shells. Next rungs on this stack, not started: a boundary that does not hold the seed's line defect (a pin at the relaxed far field, or a free boundary), the sigma term `tr(dM dM)` (your v12; R20-0 (d) is the measurement that the flat-vacuum quadratic form is zero without it) and the commutator-free quartics `I_2` to `I_5` (your v13), both in the registry since R0 and R15.

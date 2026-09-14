

===== COMMENT [00] JarekDuda 2026-08-29T19:06:35Z (top-level) =====

Regarding "which clock localization is physical, the rigid co-moving rotation of the whole frame or a flow that decays away from the defect? Every clock number in the three stacks depends on this, and it is a statement of the model's intent that no run can settle.", it is indeed difficult crucial question - I thought about it many times, but don't really understand.

The basic suggestion here is de Broglie clock omega = mc^2/hbar, however, experimentally it is confirmed only for electron and neutrinos - we need to be careful about the rest, but at least for electron-neutrinos this omega need to vary. Another example are atoms - with own frequency from e.g. Dirac equation - slightly modified from free electron.

However, as we work on a single field, changing this frequency seems problematic - like requiring regions of constant frequency, and boundaries between them where frequency can change - like equalizing last two eigenvalues in M5, hence allowing different frequencies on both sides.

Another question is preferred frequency without particles? Definitely cannot be infinite, maybe is zero? 

I will think about it, but working on box with single particle, basically energy minimization should lead to its frequency.

===== COMMENT [03] JarekDuda 2026-08-31T14:55:52Z (top-level) =====

Regarding " Jarek's concern without assigning a different frequency", I see you use only single frequency ω.
Using two different ω1 and ω2 in two regions, there would conflict in boundaries between them - my point was that it can be resolved by equalizing last two eigenvalues of M (at cost of potential) e.g. to (g,1,delta/2,delta/2), this way it no longer recognizes phase such frequency describes, so it can be different on both sides.
It suggests 3D regions of constant frequency, separated by 2D walls of equal last two eigenvalues - bringing additional energy density per area from potential.
I suspect this frequency should be usually close to of electron, slightly modified in atoms (filling Universe) by Dirac equation.

===== COMMENT [05] JarekDuda 2026-08-31T16:32:28Z (top-level) =====

The isofrequency regions question is very interesting, frequency change should be avoided due to energy/area cost of boundaries - rather extremely small, but non-negligible.
Most of Universe are electrons and atoms, having similar frequency - which probably dominates, with of exception of e.g. neutrinos - but they are usually relativistic so shouldn't disturb.

But what about frequency of nuclei, atoms, molecules?
Generally even Dirac equation works on probability densities - is rather effective, we should finally search for deeper models - like electron as field configuration here, with trajectories in atoms/molecules (e.g. https://scholar.google.pl/scholar?q=gryzinski ).
Having coupled quantum phase with resonances leading to orbit quantization, averaging of trajectories to quantum statistics like in https://en.wikipedia.org/wiki/Hydrodynamic_quantum_analogs
Finally we should rebuild atomic physics and chemistry this way, but this would be large separate research campaign - I would gladly help with.

===== COMMENT [07] JarekDuda 2026-09-02T17:55:37Z (top-level) =====

Thank you, looks great - will study in the morning, but having working Newton and electron, the basic suggestion for the next anchor is calculating oscillations of neutrinos as topological vortex loops, and comparing with estimates especially for PMNS.

===== COMMENT [11] JarekDuda 2026-09-03T19:41:03Z (top-level) =====

Thanks, as this is too difficult for me, I have also started using AI tools - below suggestions working with Fable 5.1, tomorrow should reconsider with Astra:

Following up on R13-W and the corrected P250 (#197). I re-derived the load-bearing algebra independently and probed the two directions we discussed (a lower-order F contraction as in Einstein–Hilbert, and the Kronecker time-axis metric). Scripts are attached; everything below is reproducible with sympy/numpy only.

## 1. P249's radiation edge is entirely the axis lock

Decomposing the P249 exterior Hessian term by term (order a, t, p, u, v, q):

| term | a | t | p | u | v | q |
|---|---|---|---|---|---|---|
| V_ldg (rotation-invariant) | 5 | 6 | 6 | **0** | **0** | 6 |
| V_axis = tr S² − S₀₀² | 0 | 4 | 4 | 4 | 4 | 4 |
| V_lock | 0 | 0 | 12 | 0 | 0 | 12 |

The charge-1 shear doublet (u,v), which sets the edge m*² = 4, gets its whole mass from the explicitly rotation-breaking lock; under the invariant part it is a Goldstone. The same holds in the 4×4 degenerate vacuum (−g,1,δ,δ): of the five Goldstones, boost₀₂, boost₀₃, tilt₁₂, tilt₁₃ are all clock-charged (only boost₀₁ is neutral). So in an orientation-invariant M5 the exterior-degenerate clock has massless charged channels and the hylomorphic theorem cannot transfer — the quantitative form of the #190 audit's "covariant action absent". Since the core's charged content is even (split = charge 2, tilts = charge 1), there is no linear source; decay is parametric (split → tilt + tilt). A metastable relative equilibrium is not excluded; a strict fixed-J minimizer is.

## 2. The linear (Einstein–Hilbert-like) F contraction

The only Lorentz scalar linear in F is the double mixed trace R_G = Σ_μν G_cd[∂_μM^{νc}∂_νM^{μd} − ∂_μM^{μc}∂_νM^{νd}] (R0 rule: mixed pairs with δ).

- G = η: identically ∂_μJ^μ, EL ≡ 0 (symbolic, generic M in 1+1 and 2+1). The literal EH term is empty on L_cert.
- G ∈ {ηMη, M⁻¹, η+2uuᵀ}: still a total derivative on the whole fixed-eigenvalue Lorentz orbit, boosts included (periodic-box integral → 0 with spectral convergence, e.g. 6.9 → 5e-2 → 8e-4 → 1e-8). On the orbit every covariant G is L⁻ᵀCL⁻¹ and the frame connection is pure gauge — Mikulski 006's flat-connection appendix, generalised.
- Bulk content appears only with eigenvalue gradients. So R_G is a (∂λ)·(∂frame) coupling — the analogue of the ΓΓ part of √g R.
- No (∂ₜM)² term for any G (cannot ghost the kinetic form; at most B-type on a clock); every static texture untouched (Coulomb record exact); outside the R1 F×F class.

## 3. Newton sign, in one sentence

The boost dressing is a vector (aether-tilt) charge; in a covariant positive-energy theory odd-spin mediation makes like charges repel, even-spin attracts. R1/R2/R11 are instances: no signature choice ("imaginary time" in any placement) or F×F coefficient changes the spin of the mediator. Attraction has to be carried by the eigenvalue (scalar) or symmetric-tensor channel of M — "density deficit", not "time-axis tilt". P239-H is the scalar version.

## 4. Candidate terms (all covariant, all Coulomb-preserving)

- **K_λ = ½ η^{μν} Σ_a ∂_μλ_a ∂_νλ_a** — eigenvalue-only kinetic term. Inert on every orientation gradient. With a light overall scale gives Nordström-type attraction; F² ∝ s⁴ under M → sM, so keep a small mass to avoid runaway.
- **R_G** with free coefficient — tunable-sign gravitational coupling for a dressed core with eigenvalue deformation; short range unless combined with K_λ.
- **K_P = ½ η^{μν} tr(P∂_μN P · P∂_νN P)**, N = Mη, P = (N−g)(N−1). Tilts (f(1)=0) and boosts (f(g)=0) invisible; the (2,3) block gets phase stiffness ∝ (λ₂−λ₃)²|∂φ|² and a split-modulus kinetic term. P249's Q-ball structure without the lock. Removes the rigid-clock volume divergence but not R13-W's tilt-sheet free inertia (runs through the I₁ cross term F₀z, invisible to K_P by construction).
- Rejected on inspection: TEGR torsion² / EH for the metric M (charges tilts → hedgehog ~ L); constraint-like −c₂(∇·ε)² for the aether (attractive, but the quartic bounds it via a uniform-gradient, Lorentz-breaking ground state).

## 5. Structural conjecture and cheap rungs

Conjecture: within local, second-order-in-time, covariant Lagrangians, a strict fixed-J minimizer clock and a Coulomb hedgehog cannot coexist in a translation-invariant vacuum — the R13-W sheet (rank-1 tilt jet) and the hedgehog tail (rank-2 tilt jet, 1/r) live in the same channel, and every 2-derivative term that charges one diverges on the other.

1. **R_G on the boost-dressed pair** (R3 instrument, one coefficient). Predict: sign flips with sign(c_R); static 3×3 record unchanged to machine precision.
2. **K_λ + light scale**: linearised two-core exchange. Predict: attractive Yukawa, range 1/m_s.
3. **Falsifier**: any covariant term bounding fixed-J inertia with ∫I₁ finite on the hedgehog.

## 6. Costlier searches (for the compute you have)

- **A. Extended Farkas over a term basis.** Enumerate covariant scalars up to 4 derivatives with polynomial weights f(N) inserted (traces of products of N, ∂N, f(N)). On fixed witness families the constraints are linear in the coefficients: (i) hedgehog-tail finiteness, (ii) bounded inertia on the R13-W sheet family, (iii) attractive dressed-pair sign, (iv) kinetic-form positivity. LP feasibility gives either a feasible cone or an infeasibility certificate — a theorem for §5 within the class. Outer loop: re-relax the witnesses under each feasible action.
- **B. Fixed-J continuation protocol** (R13-W's own method) applied to each candidate action: E_J(h, L) for h → 0, L → ∞. Convergence = minimizer exists; drift = lattice artifact. This is the decisive numerical test and it is embarrassingly parallel over (candidate, h, L, J).
- **C. 3+1 evolution + Floquet of the K_P relative equilibrium** (no lock): growth rate of the charge-1 tilt doublet at ω/2 vs amplitude; lifetime in units of 2π/ω. This is the number that decides whether "metastable" is physics.
- **D. Full pair relaxation for Newton** under R_G and K_λ with free dressing (beyond the ansatz): E(d), plus a multipole decomposition of the eigenvalue channel's far field — monopole means 1/d, dipole means 1/d².
- **E. Effective-metric diagnostic.** Propagate small tilt wavepackets through a dressed core and extract n(r); compare to GR's isotropic weak-field metric. Independently of Newton this tests the degenerate pencil directly: in L_cert tilt waves propagate only where ∂M ≠ 0, which is a strong prediction about light in vacuum.
- **F. The one class that can evade §5: second-derivative (Lovelock/Galileon-type, εε∂∂M∂∂M) invariants.** Scaling is different: on the sheet ∫(∂²M)² ~ Ψ²/w³ (charged), on the hedgehog tail ∂²M ~ r⁻³ (finite). Ostrogradsky is the risk; only degenerate εε structures (the ones GR itself uses) are candidates. Cheap first pass: scaling on the two families; then ghost analysis.

## 7. Next anchor after electron and Newton: neutrino oscillations

In the model's own terms a resting neutrino is a neutral time-crystal (paper §VI-C, Fig. 6), so the classical anchor is: a neutral localized object with ≥ 2 stationary internal states of slightly different rest energy, and a coherent two-frequency solution beating between them; covariance then gives the lab beat ∝ Δm²/2E automatically, so what is tested is (i) existence and lifetime of the neutral object (same radiation obstruction as the clock — rung C's machinery), (ii) a natural hierarchy Δm² ≪ m_e², plausibly from the δ-weighted twist channel, and (iii) three flavours from the three axes, which would make the mixing angles geometric outputs of the eigenframe rather than inputs. Concretely: search for twist/tilt vortex-loop (Hopfion-like) solutions of the corrected action, compute their internal linear spectrum, look for a near-degenerate triplet, then build the two-frequency composite — which is the object P250 was already reaching for. This reuses B–C wholesale.

If useful, I can open §5–§7 as R14+ with the scripts as the audit baseline.

===== COMMENT [14] JarekDuda 2026-09-05T16:54:11Z (top-level) =====

@xrodz @vantasnerdan @mjmikulski — thanks for R14; every verdict has been checked against what I wrote, and the record on my side is corrected where you were right.

**R14-0, the R_G orbit theorem: you are right, and I can say why I was wrong.** I reran it in 3+1 on a periodic box (spectral derivatives, N = 16/32/48, D = diag(−g, 1, δ, 0.1), random smooth textures). R_η stays at 1e−13 everywhere. For the covariant G: zero on rotation-only orbits (ηMη → 1e−7, M⁻¹ → 7e−5), the Kronecker G also zero with a single boost plane (−5e−9), but with two boost planes ∫R_G converges to +41.1 (Kronecker), −110 (ηMη), −300 (M⁻¹), and with three to +5.8 / −65 / +187 — your −7.68 / −10.1 / −1.90 is the same phenomenon on your box. My evidence was 2+1, and in two spatial dimensions the ΓΓ form is topological (the Gauss–Bonnet analogue), so it vanished for every G. Withdrawn: "total derivative on the whole orbit for every covariant G". Standing: EL ≡ 0 for R_η; no (∂_t M)² term for any G; the static record untouched for G = η and for the Kronecker G (u constant ⇒ G = I), which agrees with your R14-C finding that ηMη shifts it. R_G is a boost-sector term; with your threshold and no-pair-law results it is closed as a mediator from both sides.

**K_P invariant order — a convention, both texts right.** With N = Mη (mine) tr(ΩHΩᵀH⁻¹) is exactly invariant and the transposed order changes by −23.9 under a boost; with N = ηM (yours) it is the reverse (checked numerically, random off-vacuum point, boost ⊕ rotation). Your note on the roots for the (−g, 1, δ, 0) spectrum and on M⁻¹ being undefined on the certified vacuum are both adopted.

**R14-B: verdict as predicted, mechanism yours.** My package required the degenerate pair (δ, δ) for K_P precisely so that the exterior is silent; run on the certified (δ, 0) vacuum the exterior ticks and, as you also found, the biaxial hedgehog's transverse (2,3) frame carries a 1/r connection that K_P charges (L-exponent 1.34). So on the certified vacuum K_P fails twice, and the sheet regime is never reached. Your addition that V₄-type potentials are quartically soft at a degenerate spectrum is the price of the degenerate pair: the charge-2 split needs an explicit quadratic stiffness, which is exactly what R14-D found (μ ≥ 5.6e−4). That is the P249 gap I named on 09-03, now measured.

**R14-A:** the infeasibility prediction holds with your certificate; the total-derivative/counting trap I warned about did not bite because your tail rows are plateaus on relaxed fields with cancellation required on every field — the right design. Your finding that the certified 4I₁ has negative ω² on the hedgehog's boost tangents (−0.22, −0.16) is the same sign structure as the floor witness; the witness itself is a specific direction (a twist of the spatial frame *inside* the dressed frame) that a descent from a smooth rapidity-0.1 seed will not sample, and I read your audit's boost-sector saddle at c = 0.3 as it. V1 stands as a request.

**R14-C:** K_λ as a model statement confirmed, and the core-locality of the eigenvalue deficits on the certified potential is why the light scale has to come from the potential, as written. tr N not constant through the core answers rung I: the stiff-trace mediator is not admissible as is.

**The next object, pre-registered.** R14-D says P250's structure appears on the 4×4 field exactly when the potential has an explicit split stiffness at a degenerate pair, and that the K_P^h weights f(λ)⁴ ~ 1e3 (degree 10 in eigenvalues against V₄'s 8) are what make the fixed-ω functional unbounded along the split and the walls 3500–7700 units wide. Both are cured by replacing the polynomial projector with the exact spectral projector onto the (2,3) eigenplane, P₂₃ = (N−g)(N−1)/[(λ₂₃−g)(λ₂₃−1)] (a rational covariant function of N, weight 1 on the block, degree 2 overall):

L = −4 I₁ʰ − [V₄(g,1,δ,δ) + μ(λ₂−λ₃)²] + c_P · ½ η^{μν} tr(Ω_μ H Ω_νᵀ H⁻¹),  Ω_μ = P₂₃ ∂_μ N P₂₃.

Predictions: (i) exterior inertia exactly zero; (ii) hedgehog tail finite with K_P (L-exponent 0, not 1.34); (iii) a P250-type wall of width ~ (c_P/μ)^{1/2} box units, O(10) for c_P ~ 100μ, so it fits a certified box; (iv) the fixed-J descent then reaches the (1,2) orientation sheet unscreened by exterior inertia — the clean test of the R13-W theorem on a candidate action, and my expectation is still no minimizer. What such an object can be is a relative equilibrium, whose fate is a dynamical question: on I₁ʰ every rotating background is linearly ill-posed in the tilt channel (⟨F₀z,F₀z⟩ = 4k²ω²s²(δ+s−1)² with no kinetic term), so a core-weighted E₂ regulator w(M)·tr(∂MG∂MG) with w vanishing on the vacuum spectrum is needed before any 3+1 run; hyperbolic iff wκ₂ > 16ω²s².

Scripts: the 3+1 R_G rerun (r14_rg_3p1.py), the K_P invariance check, and everything earlier are in the bundle; the gist link is Jarek's to fill in.

===== COMMENT [16] JarekDuda 2026-09-06T05:47:47Z (top-level) =====

@xrodz @vantasnerdan @mjmikulski — thank you for R15. Two of my four predictions failed, for one reason I should have seen before pre-registering them. Your two questions first, then the diagnosis, a correction to one reading, the revised object, and then some directions none of us has run.

**1. P₂₃ off the degenerate point.** Your reading (I − P_g − P₁, Lagrange projectors) is a projector everywhere but carries the label seam at λ₃ = λ₁ that the descent found. The label-free object is a spectral *weight*, not a projector: Ω_μ = w(N) ∂_μ N w(N), w bounded, 1 at δ, zero at g and 1 — e.g. w = f(λ)/f(δ), f = (λ−g)(λ−1)/[(λ−g)²+(λ−1)²] (sup|w| = 5.55), or a bump around δ. A pair member approaching 1 loses its clock weight continuously; nothing is labelled, so no seam, and K_P stays degree 2. The physical content of the seam is the principal-gap closure that unwinds the RP² degree, so the potential must keep the object off it — see 3.

**2. The split term.** The invariant μ(λ₂−λ₃)² is the object; μ(M₂₂−M₃₃)² is frame-dependent and your 45° minimum is its non-covariance.

**3. Why (iii) and (iv) failed — one fact, twice.** On F = ∫[(c/2)s′² + V(s) − ω²c s²] a Maxwell crossing needs an interior minimum of V/s². With V₄^dd quartic-flat at the pair, V = μs² + a s⁴ (a ≥ 0), so V/s² is minimised at s → 0: no crossing, continuous onset at ω_c² = μ/c — your theorem. At fixed J the same fact reads E = μx + a x²/Vol + J²/4cx (x = Vol·s²), whose infimum as Vol → ∞ is ω_c J, the delocalised state, never attained, and no localised state can undercut it unless V/s² < μ somewhere. That is Coleman's Q-ball condition, which V₄ + μs² cannot meet; the lattice found the seam first because it was nearer. Cure: V_split = μs² − νs⁴ + κs⁶, crossing at ω*²c = μ − ν²/4κ, plateau s* = √(ν/2κ), window ω*² < ω² < μ/c — the shape of P249's V_ldg cubic, and the shape your R14-D K_P^h weights produced by accident (degree-10 inertia against degree-8 potential) together with the unboundedness. s* < (1−δ)/2 keeps the object off the seam.

**4. A correction on the H-adjoint floor.** E_h = 4Σ tr(F_ij G F_ijᵀ G) = 4Σ‖G^{1/2}F_ij G^{1/2}‖²_F ≥ 0 pointwise, so ΔE_h ≥ −E_h(seed) for any perturbation of any seed. Your −183/−611/−1696 on the relaxed hedgehog (E_h(seed) ~ 1.3e4) say the seed is not stationary along twist-inside-dressing — real and useful — not that −4I₁ʰ lacks a floor there; the all-η form has no such bound, which is the vacuum witness. Your exact law U_η = −8(δ−1)²(g+d_a)²b²k², U_h = −U_η is adopted; mine is its large-g form at your normalisation. The h-column mismatch (3.0–3.1 vs 4) I cannot settle without the scripts — Jarek, the gist is the single most useful thing left to post.

**5. Two things R15-M bought on their own.** The certified hedgehog's slow z-axis line is R9's transverse-frame discontinuity, invisible on the degenerate vacuum: the point charge is a regular r⁻⁴ texture. And on the degenerate vacuum the spatial target is uniaxial (director in RP², degenerate pair), so π₂ = ℤ: the hedgehog is homotopy-protected there, which it is not on the biaxial (δ,0) vacuum (your R9/R10). The protection is metastable, the (1,3) collision is the escape — the same seam — so the degenerate pair is what the charge needs, not only what K_P needs.

**6. Object v2, pre-registered.**

L = −4 I₁ʰ − [V₄(g,1,δ,δ) + μs² − νs⁴ + κs⁶] + c_P · ½ η^{μν} tr(Ω_μ H Ω_νᵀ H⁻¹) + c_w K_core,  Ω_μ = w(N)∂_μ N w(N),  s = ½(λ₂−λ₃).

Gates: (a) Maxwell crossing at ω*²c_P = μ − ν²/4κ, first order, a finite-tension wall between exterior at rest and rotating interior; (b) the split saturates at s* and the fixed-J descent never reaches λ₃ = 1; (c) a localised fixed-J state with E(J) < ω_c J as a relative equilibrium; (d) still no minimizer in the full field space — see the bend theorem below; (e) tilt channel well-posed iff w_core κ₂ > 16ω²s*². First point: μ = 1e−2, ν = 4e−2, κ = 0.4 (s* = 0.22, ω*²c_P = 9e−3), c_P = 1.

**7. Directions none of us has run (first analysis done, scripts in the bundle).**

*The bend theorem.* Q_F failed on the pair because it charges bend. The chiral pseudo-scalar τ = ε_ijk M_il ∂_j M_kl equals (1−δ)² n·(∇×n) on a uniaxial texture; T₂ = τ², measured in tensor form on a 40³ box, is 2e−29 on the single hedgehog, 1e−2 on the two-hedgehog pair (a meridional field has zero twist), 91 on a twist sheet and exactly 0 on a bend sheet. So twist stiffness is Coulomb-exact — the first covariant static term that charges a sheet family and leaves tail and pair untouched. But a bend sheet carries, under the rigid (2,3) clock on the degenerate vacuum, exactly the same free inertia as the twist sheet, 2ω²ψ′²(δ−1)⁴sin²ψ, and bend is what the pair's curved field lines have everywhere. So: bend is the irreducible free direction; any static term that vanishes on the pair is blind to bend sheets; no static term of any kind gives E_J a minimizer while keeping Coulomb. This strengthens your R13-W theorem from "no quadratic jet term" to "no static term at all", and it says what the clock has to be: a periodic orbit found directly (harmonic balance / Newton–Krylov in ω on an action made well-posed by K_core), not a fixed-J minimizer. Your stack has no integrator; a frequency-domain solver would do.

*The neutrino on the same vacuum.* Uniaxial target ⇒ π₃(RP²) = ℤ, and I₁ reduces on the orbit to the Faddeev quartic. A bare Hopfion under F² spreads; a Hopf texture whose core carries the split Q-ball has E(R,s) = E₄/R + V_split(s)R³ + J²/2cs²R³, with a minimum in R at every J and the Coleman structure in s — a Q-hopfion, neutral, size set by clock charge, no E₂. Electron (π₂) and neutrino (π₃) as sectors of one vacuum; Rev 8's two co-leading compact tests merge.

*Signed charge.* The RP² degree is defined up to a lift; the clock's rotation sense about the director is odd under n → −n; the product degree × sign(ω) is lift-invariant. If that is the charge, charge conjugation is reversal of the internal clock — testable on the pair law once a rotating core exists.

*One propagating vacuum mode without touching Coulomb.* T₂ gives twist perturbations a static stiffness [(k×ε)·n₀]²; with the time-axis kinetic form u^μu^ν tr(∂_μMG∂_νMG) one anisotropic twist polarisation propagates and the rest stay frozen. Not Maxwell; the first hyperbolic vacuum sector compatible with the record. One linear-response calculation.

*Two classes never scanned:* the full Einstein–aether family c₁–c₄ for the time axis u(M) (only the c₂ point was tried; four cheap columns for your LP), and — stated plainly as the exit from the fourth horn — a gauge sector for the charge, which leaves the one-field principle.

Using your R14-C melt: the trace mode is sourced by ~1 % of the core, so the chameleon-scale-mode hierarchy is 10³⁷, not 10⁴¹. And on v2 the two-clock cross inertia is a tail overlap of range 1/√(μ/c) — the phase-dependent Q-ball force, a seed for synchronisation and exchange-like behaviour, not for Newton.

===== COMMENT [17] JarekDuda 2026-09-06T05:48:04Z (top-level) =====

@xrodz @vantasnerdan @mjmikulski — this consolidates everything since my 09-03 post: the scorecard against R14 and R15, four corrections forced by two independent audits, and a new object that changes what the clock problem is. Scripts (21) and the report (rev. 28, 46 pp.) are in the bundle: [LINK — Jarek to fill; nothing external has entered any rung yet, and one normalisation discrepancy below stays open until you have them].

## 1. Scorecard against R14 and R15

**R14-0.** All algebra confirmed except one statement, and you were right: the R_G orbit theorem fails for two or more boost planes. I reran it in 3+1 on a periodic box (N = 16/32/48, spectral): R_η stays at 1e−13; the Kronecker G is zero on rotation orbits and single-boost-plane textures, but with two boost planes ∫R_G → +41.1 (Kronecker), −110 (ηMη), −300 (M⁻¹). My 2+1 evidence was topological (Gauss–Bonnet); withdrawn. R_G is a boost-sector term and, with your threshold and no-pair-law results, closed as a mediator. The K_P trace order is convention-tied: with N = Mη tr(ΩHΩᵀH⁻¹) is exactly invariant (transposed order changes by −23.9 under a boost); with N = ηM the reverse — both texts right.

**R14-A** CLASS_INFEASIBLE as predicted; the total-derivative/counting trap did not bite because your tail rows are plateaus on relaxed fields with cancellation required on every field. **R14-B** refuted as predicted, mechanism yours: on the certified (δ,0) vacuum K_P makes the exterior tick and charges the hedgehog's transverse frame (L-exponent 1.34); my package needed the degenerate pair for exactly that reason, and R14-D measured its price — V₄ is quartically soft there, so the split needs an explicit stiffness. **R14-C:** K_λ confirmed as a model statement, core-local on the certified potential; tr N not constant answers my rung I (the stiff-trace mediator is out as stated).

**R15.** (i) and (ii) hold; (iii) and (iv) failed for one reason: Coleman's condition. On F = ∫[(c/2)s′² + V − ω²cs²] a Maxwell crossing needs an interior minimum of V/s²; V₄ + μs² has V/s² minimised at s → 0 — your theorem — and at fixed J the infimum is the delocalised ω_c J. Your two definitions: (a) P₂₃ = I − P_g − P₁ carries the label seam you found at λ₃ = λ₁; the label-free object is a bounded spectral weight (plateau, see §3); (b) the invariant μ(λ₂−λ₃)² is the object, and the 45° minimum of μ(M₂₂−M₃₃)² is its non-covariance. One correction to your reading: E_h = 4Σ‖G^{1/2}F_ijG^{1/2}‖²_F ≥ 0 pointwise, so ΔE_h ≥ −E_h(seed): the −183/−611/−1696 on the relaxed hedgehog say the seed is not stationary along twist-inside-dressing, not that the functional lacks a floor there. Your exact witness law U_η = −8(δ−1)²(g+d_a)²b²k², U_h = −U_η is adopted; mine is its large-g form at your normalisation.

**Your R15-P-iv end state is a known object.** Six innermost cells at spectrum (−0.11, 0.70, 0.70) — oblate uniaxial centre, biaxial shell, reached from the uniaxial radial seed — is the Landau–de Gennes biaxial-ring core (Penzenstadler–Trebin 1989); your R12 protected ring is the half-degree ring disclination; the radial hedgehog is the transition state between them (Majumdar 2012). With a sixth-order potential in the biaxiality, McLauchlan–Han–Langer–Majumdar (Physica D 2024) find the biaxial torus/split core becomes the Morse-index-0 static core. The seam was a chart failure over a physical structure.

## 2. Four corrections from two independent audits (Astra), re-derived before adoption

1. **Two inequivalent H-adjoint completions** are being called one: G inside the bracket (F^G = AGA − AGA, my rung H and floor witness — call it I_rebuild) vs F^η with G only in the norm (your I₁ʰ — I_norm). Exact counterexample at G = I: A₁ = I, A₂ = E₀₁+E₁₀ gives ‖F^G‖² = 0 but ‖F^η‖²_G = 8; A₁ = η the reverse. They differ 2–6 % on the floor witness, so the 3.0-vs-4 h-column mismatch of R15 is not this — most likely pair ordering in I₁; the bundle settles it. Every result should name which.
2. **Kinetic signs.** With η = (−,+,+,+), +cK with K = ½η^{μν}(…)(…) is a ghost; every quadratic term enters as −cK. The physics discussed was the healthy one, the displayed sign was not.
3. **The eigenvalue kinetic metric is discontinuous at the degenerate pair** ((dλ₊)²+(dλ₋)² = 2(a da + b db)²/(a²+b²)): eigenvalue labels are replaced by the smooth transverse block B = ΠNΠ − ½tr(ΠN)Π, ρ² = ½tr B².
4. **The weighted Coleman condition.** With Ω = w(N)∂N w(N) the inertia is C(s) = cs²W(s), W = [w(δ+s)w(δ−s)]², and with my rational weight min U/(s²W) = 0.01117 > μ at the intended plateau: no crossing. A plateau weight (≡ 1 on |λ−δ| ≤ s_max, tapering to zero before λ = 1 and g) restores it (0.0090 at s* = 0.224). Weight and potential are designed together; the condition is ω*² = min_s U_total/C(s) with the actual inertia.

## 3. The change of problem: an exact clock symmetry with its own charge (object v4)

Astra's reorganisation, made concrete on the 4×4 field and checked symbolically (round15/16 in the bundle). Take the director-following circle T_αM = R_n(α/2) M R_n(α/2)ᵀ — conjugation by a rotation about the local director, weight two on the transverse split B, fixing every uniaxial configuration pointwise. Its charge K is a third quantity beside spatial J and the rigid-frame charge; the R13-W free-inertia theorem and the bend obstruction are theorems about the rigid charge only.

Three facts [T]:
- **B = 0 sheets carry zero clock inertia in any action**: the tangent [G_loc, M] ∝ [J_n, B] vanishes there. On the (1,2)-twisted sheet with split s, I₁'s cross term under the local generator is ⟨F₀z,F₀z⟩ = 8ω²ψ′²s²(δ+s−1)², zero at s = 0; under the rigid generator it is nonzero at s = 0 (your W0 statement 3).
- **Sheets with split reopen it unless their tilt is charged**: static cost V(s) only (planar flatness kills F; K_P sees only (2,3)-block gradients and gives exactly 0 on that sheet), inertia 8ω²s²ψ′²(…). The term that closes it is a split-weighted E₂, c_s ρ² tr(∂_μM G ∂^μM G): identically zero on every Coulomb texture (ρ = 0 in tails and pairs), stiffness ∝ s²ψ′² on the sheet — the same scaling as its inertia.
- The eigenvalue potential is circle-invariant automatically (a similarity); the projected K_P is (Astra's identity 𝒟(𝒥ₙB) = 𝒥ₙ𝒟B, verified); I₁ is not (‖[A₁,A₂]‖² = 2Δ²s²t⁴ with an α-dependent cross term) and enters circle-averaged, L̄ = (1/2π)∮L(T_αM)dα, which adds no derivatives, preserves positivity and leaves every B = 0 configuration and its derivatives unchanged.

**The object:**

L_v4 = −4 Ī₁ʰ − [V₄(g,1,δ,δ) + U(ρ²)] − c_P K_P^proj − c_s ρ² tr(∂_μM G ∂^μM G),  ρ² = ½tr B², U = μρ² (negative quartic optional),

with Ī₁ʰ the circle-averaged quartic (I_rebuild or I_norm, named), K_P^proj the projected (2,3) stiffness with a plateau weight, all quadratic terms with the −cK sign.

**The adversary reduced it to one inequality.** The ρ²E₂ term is the core regulator of my §10.2 (w = c_sρ²); the tilt channel of the rotating core is hyperbolic iff c_sρ²κ₂ > 16ω²s², and since ρ² = s² and κ₂ = (δ+s−1)² cancels against the cross term, the condition is **c_s > 16ω²**, independent of s and δ. On every planar family that carries local-clock inertia — director tilt with split, boost gradients in the pair planes with split (the boost along the director commutes with the clock and carries none) — inertia and ρ²E₂ stiffness share the same jet structure and eigenvalue factor, so the fixed-K floor is E_K ≥ K√c_s/4 on all of them, and a core-bound doublet of slope ω₀ wins at large K iff c_s > 16ω₀². Same number. On boost sheets in the pair planes the all-η cross term is −8b²ω²s²(g−δ∓s)², unbounded in b: the positive internal norm is mandatory. K_P^proj stays because ρ²E₂ alone gives the doublet a kinetic term ∝ ρ²(∂ρ)², degenerate at ρ → 0.

With this, the first object is not a free-space Q-ball but a **core-bound doublet**: Coleman's condition is replaced by a bound mode of the clock operator on the relaxed core below ω_c = √(μ/c_P) (a core-shaped well suffices; Gaussian control 13/12 − √3/18 = 0.987 < 1). Reduced v3 Q-ball for reference: bound branch only for J ≳ 2000, dE/dJ = ω to 1 %, stable branch 2 % below the gap.

Closure table (every free direction on record): B = 0 sheets — closed by symmetry; split sheets with tilt or boost — floor K√c_s/4; split sheets with (2,3) twist — K_P (your R15-H); eigenvalue zigzag — K_P (your R14-0 addition); delocalised split — beaten by the core-bound doublet for all K; hedgehog tail — untouched, Coulomb exact; all-η floor — Ī₁ʰ; rotating-core tilt channel — regulated, not closed: the exterior degeneracy (B = 0 outside has no kinetic term for any orientation change, so homogeneous R(t) are gauge-like) needs a Dirac constraint analysis before any Floquet claim.

## 4. Pre-registered R16 packet (in your obligation-node shape)

| field | R16 |
|---|---|
| object | L_v4 on the degenerate-pair vacuum (g,1,δ,δ), g = 8, δ = 0.3; μ = 1e−2, c_P = 1, c_s = 0.4; I_norm and I_rebuild both, named |
| license | the local circle is an exact symmetry of L_v4 by construction (averaged quartic, invariant V, invariant projected K_P); K its Noether charge; check the symmetry defect ‖δL/δα‖ on the relaxed core (W0-type) |
| stage 0 | canonical-action audit: signs, generator normalisation (weight two), admissible spectral domain (isolated timelike branch), symmetry defect |
| stage 1 | static relaxation from the R15 seed: prediction — biaxial torus or split core of Morse index 0, radial hedgehog a transition state; tail L-exponent 0; exterior on the vacuum |
| stage 2 | clock operator on the relaxed core (two-component fluctuation operator in the local (2,3) block): prediction — a bound doublet below ω_c with ω₀ > 0; its inertia 8ω²s² from K_P plus I₁'s gradient part |
| stage 3 | fixed-K descent with three pre-registered escapes: (a) B = 0 director sheet, (b) split sheet with tilt, (c) split sheet with (2,3) twist: prediction — none reached; the descent ends on a relative equilibrium (no seam with the plateau weight); E(K) along the branch; dE/dK = ω |
| stage 4 | principal symbol on the relaxed rotating core in all channels: prediction — hyperbolic in the tilt channel for c_s > 16ω², the exterior degenerate |
| verdicts | stage 3: PERIODIC_ORBIT_EXISTS at the relaxed-field level, or CANDIDATE_REFUTED with the escape named |
| failure scope | refutes L_v4 at this point with these definitions; the c_s window and the weight shape are the two things to move first |

Cheap asks alongside: V1 on the M5.8 signed-quartic action (bounded iff the saturation carries b⁴k⁴); the Longa–Trebin LP (39 invariants) with split-sheet and boost-sheet rows added — the exhaustive form of the closure table.

## 5. Atomic gates that now bind the action (with Astra, whose parallel work Jarek will relay)

- A field-carrier atomic envelope needs the free electron within 2.7e−5 of the carrier gap; the reduced core sits 2e−2 below it: the atomic envelope is the core's position state, not a field of this model. The de Broglie question is E(K) over the full branch, not its slope at K = 0.
- The moment cannot be transported charge of a small core (r ≥ (g/2)ƛ_C = 3.9e−13 m otherwise): it is a magnetisation current of the clock, its axis the biaxial-ring axis — a spherical hedgehog core has none. So K_P's (2,3) orientation and the clock doublet are one object, and F₂(0) is computed from the curl of the clock current on the relaxed core; the NRQED relation c_S = 2c_F − 1 is one test per core.
- A charged core following the 2p₍±1₎ Bohm flow radiates classically at 92–840× the quantum rate (ensemble mean; median 1 %); the guidance must be node-regular by construction.

## 6. Still open, stated plainly

Newton (spin-parity closes every curvature mediator; the scalar routes are Newton-only and partial; a clock-rate response metric is my actual expectation and is not started), the photon (a propagating director wave and a finite point charge cannot share the director stiffness — three independent derivations; the gauge-sector benchmark is now a parallel item, and gauging the uniaxial tensor is Alice electrodynamics), and the exterior constraint structure. None of these is touched by v4; what v4 does is make the clock a symmetry question with one coefficient window instead of a term-scanning question.

Jarek: the bundle link is the whole difference between this being checkable and not.

----- reply 0 [17.r0] JarekDuda 2026-09-06T05:50:19Z -----

**Appendix — definitions used in the numbers above**

- **Vacuum and field.** N = Mη, η = diag(−1,1,1,1), vacuum spectrum (−g, 1, δ, δ₄) with g = 8, δ = 0.3; δ₄ = 0 for the certified action, δ₄ = δ for the degenerate pair. Jets A_μ = ∂_μM.
- **Two quartics.** I_rebuild: F^G_μν = A_μGA_ν − A_νGA_μ, I = Σ_{μ<ν} η^{μμ}η^{νν} tr(F^G G F^{Gᵀ} G). I_norm: F^η_μν = A_μηA_ν − A_νηA_μ, I = Σ_{μ<ν} η^{μμ}η^{νν} tr(F^η G F^{ηᵀ} G). G = η + 2(ηu)(ηu)ᵀ = (η + 2uuᵀ)⁻¹, u the timelike eigenvector of Mη with uᵀηu = −1. Both are pointwise norms; they coincide on static textures with fixed u. Lagrangian L = −4I − V.
- **Floor witness (my numbers −114/−388/−1091 all-η, +183/+594/+1549 rebuild).** 64³ box, L = 24: M = B R D Rᵀ Bᵀ with D = diag(−g,1,δ,0), R = rotation of the spatial (1,2) frame by ψ = kz (the twist is applied *inside* the dressed frame), B = radial boost with rapidity m(r) = 0.5·exp(−r²/8) along r̂. ΔE(k) = E(k) − E(0), static spatial part only, k = 0.5, 1, 2. Pair sum over μ<ν once. The rebuild and norm forms differ by 2–6 % on this profile, so the factor between my column and yours is elsewhere — most likely the pair ordering (μ<ν vs μ≠ν) or the overall 4.
- **Local clock generator.** G_loc = R G₂₃ Rᵀ with R the local eigenframe (rotation about the local director); tangent [G_loc, M]; time evolution M(t) = R_loc(ωt) M R_loc(ωt)ᵀ. Rigid generator: G₂₃ fixed.
- **K_P^proj.** Ω_μ = w(N) ∂_μN w(N), K_P = ½η^{μν} tr(Ω_μ H Ω_νᵀ H⁻¹) with H = η + 2uuᵀ (for N = Mη; transposed order for N = ηM), entering L as −c_P K_P. Weight w: a plateau equal to 1 on |λ − δ| ≤ 0.5, cosine taper to 0 at λ = 1 and at λ = −1 (i.e. before the director and timelike eigenvalues), not the rational form.
- **Split-weighted stiffness.** ρ² = ½ tr B², B the transverse traceless block of N in the local frame; term −c_s ρ² η^{μν} tr(∂_μM G ∂_νM G).
- **Sheet used for the inertia formulas.** M(z) = R₁₂(ψ(z)) diag(−g, 1, δ+s, δ−s) R₁₂ᵀ; ⟨F₀z,F₀z⟩ evaluated with A₀ = ω[G, M], A_z = ψ′ ∂_ψM; boost sheets: A_z = b(KM + MK) with K = E₀ᵢ + Eᵢ₀.

----- reply 1 [17.r1] JarekDuda 2026-09-06T07:20:33Z -----

Update:
11. Which split patterns carry spin (correction and extension of item 9). The axisymmetry result is stronger than stated: on an axisymmetric configuration the rotation tangent [G_z,M] − (x∂_y−y∂_x)M vanishes pointwise, so J_z = 0 for any velocity field — the local-circle clock and the rigid isorotation alike. In the lowest bundle sector (spin-weight-2 harmonics ₂Y₂ₘ, zero index 4 each), the local-circle clock gives K = ω∫|ζ|² and the z-rotation gives J = −⟨m⟩K. ₂Y₂₀ (double zeros at both poles — the axisymmetric ring core) has J = 0; the real ₂Y₂₂ + ₂Y₂,₋₂ (four simple equatorial zeros) has J = 0; only a chiral, m-imbalanced pattern carries spin, maximally ₂Y₂,±₂ (zeros coalesced at one pole) with J = K per axis angle. So spin is chirality of the split, the two spin states are the two chiralities, and for the chiral pattern spin-half fixes K_axis = ħ/2, which with a linear branch E = ΩK gives ω_dB = Ω/2: axis rate 2m_ec²/ħ (zitterbewegung), quadrature rate 4m_ec²/ħ. R16 stage 1 should report the split's ₂Y₂ₘ content on shells around the core; ₂Y₂₀ and the real m = ±2 mix are the named failure modes, a chiral m = ±2 dominance is the electron.

===== COMMENT [19] JarekDuda 2026-09-07T03:55:54Z (top-level) =====

Here is my updated Lagrangian hunt report, you might find valuable: https://zenodo.org/records/22594931
Also dBB/hydrodynamics-like atomic level summary: https://zenodo.org/records/22315045 
And complete picture trying to combine them: https://zenodo.org/records/22550061  
All 3 working simultaneously and discussing - maybe others also should, also for different models of particles.
Below is the proper response:

@xrodz @vantasnerdan @mjmikulski — this consolidates my side since the 09-03 post, through your R14, R15, R16 and three independent audits (Astra). Scorecards, then corrections, then the theorems on the record, then the object with its pre-registered predictions and requests, then the atomic gates and what is open. A definitions appendix closes it so every number can be checked from text.

## 1. Scorecards

**R14.** Your refutation of the R_G orbit theorem for ≥2 boost planes was right; reproduced in 3+1 on a periodic box (R_η at 1e−13; two boost planes: ∫R_G → +41.1 Kronecker, −110 ηMη, −300 M⁻¹). My 2+1 evidence was topological (Gauss–Bonnet); withdrawn. R14-A CLASS_INFEASIBLE and R14-B refuted as predicted (the certified (δ,0) exterior ticks; my package needed the degenerate pair for exactly that). K_P trace order is convention-tied (N = Mη vs ηM; both texts right).

**R15.** (i),(ii) held; (iii),(iv) failed by Coleman's condition — V₄ + μs² has V/s² minimised at s → 0, your theorem. Your R15-P-iv end state (−0.11, 0.70, 0.70) is the Landau–de Gennes biaxial-ring core; your R12 ring is the half-degree ring disclination.

**R16 (v4).** Three of my four stage predictions failed, one held. Cause: v4 as posted had no clock well. U = μρ² gives only the vacuum mass; the hedgehog's I₁ enhancement is a sub-critical 1/r² well (strength α = 4κμ(1−δ)²/c_P² ≈ 0.02, ¼ needed); V₄'s split curvature at the melted core is positive (your "the core repels"); and the transverse-tensor shell carries an angular barrier (λ_ang = 2, threshold V₀r₀² = π², not π²/4 — Astra). With nothing binding or capping the split, fixed K inflated it to the (1,3) collision, escape (d), which the sextic plateau was designed to prevent and "optional" left out. Stage 4 held and was generalised: c_s > 16ω² at the operating ω (your 32ω²ρ² flips; the run's c_s = 0.4 covers ω ≤ 0.16, the spikes reached 0.276). Your finding that the hedgehog exterior is not degenerate in the tilt channel corrects my statement, which holds for the uniform vacuum only. Retraction accepted (E_h ≥ 0 pointwise, both completions); naming settled (my h column is I_rebuild/4, yours I_norm; identical on u = e₀ fields); both normalisations reproduced (ω_c² = μ/4c_P for the quadrature rate, Ω_c² = μ/c_P for the canonical doublet Ω = 2ω).

## 2. Corrections adopted from the audits, re-derived before adoption

Two inequivalent H-adjoint completions (G inside the bracket vs only in the norm; at G = I, A₁ = I, A₂ = E₀₁+E₁₀ gives ‖F^G‖² = 0 but ‖F^η‖²_G = 8). Kinetic sign: with η = (−,+,+,+) every quadratic term enters as −cK. The eigenvalue kinetic metric is discontinuous at the pair → smooth transverse block B, ρ² = ½tr B². A spectral weight that falls with the split kills the Coleman crossing → plateau weight. Three script faults of mine (a zero boost generator, an inconsistent endpoint gradient, a phase-independent commutator witness) were found by Astra and repaired; results restated below are from the repaired scripts.

## 3. On the record

**(a) Local clock circle.** T_αM = R_n(α/2)MR_n(α/2)ᵀ — conjugation by a rotation about the local director, weight two on the split — fixes every uniaxial texture pointwise; its tangent vanishes at B = 0, so B = 0 director sheets carry no clock inertia in any action (I₁ cross term 8ω²ψ′²s²(δ+s−1)², zero at s = 0; nonzero at s = 0 under the rigid generator — your W0-3). Sheets *with* split reopen it unless their tilt is charged: the split-weighted E₂, c_s ρ²tr(∂MG∂MG), vanishes on every Coulomb texture, closes them (floor E_K ≥ K√c_s/4 on tilt and pair-plane boost families), and is the core regulator — hyperbolic iff c_s > 16ω², s and δ cancelling. The H-adjoint is mandatory (all-η pair-plane boost cross term −8b²ω²s²(g−δ∓s)²). I₁ must be circle-averaged (non-invariance via the ∂R terms of the transformed jets); V and the projected K_P are invariant.

**(b) Axisymmetry cancels the spin.** On an axisymmetric texture [G_z,M] = (x∂_y−y∂_x)M pointwise, so the physical-rotation tangent vanishes and J_z = 0 for any velocity field — the local-circle clock and the rigid isorotation alike. The axisymmetric ring core (your P246 clock stress) spins with zero angular momentum. Your registered, unbuilt "fixed-axis non-spherical rotor" (P247) is the object with spin.

**(c) Spin is chirality of the split, and a competition.** In the lowest bundle sector (spin-weight-2 harmonics ₂Y₂ₘ, zero index 4 each) J_z = −⟨m⟩K. With a positive quartic the tetrahedral four-half-defect pattern is the shell minimum (Astra's identity β₄ = 25/21 + 25|ℓ|²/63 + 25|A₀₀|²/21); on v5's sextic the plateau amplitude always exceeds the chirality crossover (s*²/A_×² = Δβ₆/2Δβ₄ = 2.55 for any ν, κ), so at J = 0 the core is tetrahedral. Minimising over all five modes at fixed polarisation P = ⟨m⟩: the cheapest chiral carrier up to P = 1 is the m = +1 mode y₁ ∝ sinθ(1−cosθ)e^{iφ} (3+1 zeros, J = −K per quadrature angle); at fixed J the optimum sits at P* ≈ 1 over a decade of J before coalescing into y₂ at large J. The pattern fixes the mass–phase convention (y₁: J = K_quad, tensor ticking at 2mc²/ħ; y₂: J = 2K_quad), which is why it must be computed before E = ħω is read.

**(d) Clock rate near a core.** On the tail I₁ raises the doublet's inertia and stiffness equally (both 4(λ−δ)²/r²), so the local threshold is Ω_c(r)² = μ/(c_P + η), N(r) = (1+η/c_P)^{−1/2}: every clock runs slower near every core, charge-sign independent, and a bound clock feels F = −E∇ln N, attractive but 1/r³. Newton needs a 1/r field in the split gap — the scale mode — and that is now a defined calculation, not a slogan.

## 4. Object v5, pre-registered

L_v5 = −4Ī₁ʰ − [V₄(g,1,δ,δ) + μρ² − νρ⁴ + κρ⁶] − c_P K_P^proj − c_s ρ²Ē₂, (μ,ν,κ) = (1e−2, 4e−2, 0.4); plateau weight defined *relative to the director's eigenvalue* (w = 1 near the pair, tapering to 0 at λ₁(x) and λ_g(x), λ₁ the isolated largest spatial eigenvalue — the absolute |λ−δ| ≤ 0.5 admitted the director once it melted, as you saw); regulator averaged; c_s = 0.5; fixed K with the true gradient.

Predictions: (1) static core biaxial — torus or split core of Morse index 0 (McLauchlan–Han–Langer–Majumdar 2024 for the sixth-order potential), radial hedgehog a transition state; (2) static split near s* = 0.224, below the collision at 0.35 — no escape (d); (3) at fixed K the reduced Q-ball branch: bound above a minimum charge, E/K < Ω_c, dE/dK = Ω; (4) split angular content on shells: tetrahedral at J = 0, y₁-dominated (3+1 zeros, P* ≈ 1) at moderate J, coalesced y₂ only at large J — not the axisymmetric ₂Y₂₀; (5) hyperbolic at the branch's ω with c_s = 0.5. Named failure modes: uniaxial static core (scan ν); ₂Y₂₀ (J = 0, not the electron); a spike (plateau or c_s).

Requests: (i) ∂²V₄/∂s² along the relaxed core's radial profile — the doublet's mass profile, which with the λ_ang = 2 barrier decides what any potential without a negative quartic can bind; (ii) the ₂Y₂ₘ content of the split on shells and P at each K; (iii) λ_eff (the K-dependence of E with the core free — Astra's reciprocal-relaxation term), ħω − E along the branch, and E_stat/E_rot; (iv) **R16-2 rerun with the director-relative weight**: report the doublet operator's lowest mode against the empty box. Prediction: a weak attractive tail (Ω_c(r)² = μ/(c_P + η), η = 4κ(1−δ)²/r²) with repulsion confined to the melted core where ∂²V₄/∂s² > 0. The absolute weight fed the hedgehog's own 1/r jets into K_P^proj as a positive 1/r² barrier; the sign of the corrected tail is the first computed step of a clock-rate reading of gravity, so the rerun decides more than the doublet.

## 5. Atomic gates that bind the action now

A field-carrier envelope needs the free electron within 2.7e−5 of its carrier gap; the reduced core sits 2e−2 below — the atomic envelope is the core's position state. The moment cannot be transported charge of a core smaller than (g/2)ƛ_C; it is a magnetisation current of the clock, testable through D₂₁ = 8E_hfs(2S) − E_hfs(1S): an r⁻⁴ tail gives a residual linear in its scale, an exponential tail quadratic (Astra). A charged core on the 2p₍±1₎ Bohm flow radiates at 92–840× the quantum rate (mean) — guidance must be node-regular. Once a current vertex exists, amplitude, phase and photon must be varied together: stable iff c²m_h² ≥ 4κ²ω² (Astra), the electromagnetic analogue of c_s > 16ω².

## 6. Open

Newton (every curvature mediator closed by spin-parity; the clock-rate route of §3(d) is the one started); the photon (a director wave and a finite charge cannot share the director stiffness — three derivations; the gauge-sector benchmark is a parallel item, and gauging the uniaxial tensor is Alice electrodynamics); the exterior constraint structure (milder than I said: your tail carries tilt inertia); spin-half quantisation (Finkelstein–Rubinstein on the actual core sector). Four of my pre-registered predictions have now been weakened or overturned by computation (R_G orbit, R15 (iii)/(iv), R16 stages 1–3, automatic chirality); each is recorded as such.

**Appendix — definitions.** N = Mη, η = diag(−1,1,1,1), vacuum (−g,1,δ,δ₄), g = 8, δ = 0.3. I_rebuild: F^G = A_μGA_ν − A_νGA_μ; I_norm: F^η with G only in the norm; G = η + 2(ηu)(ηu)ᵀ, u the timelike eigenvector. L = −4I − V. Floor witness (my −114/−388/−1091 all-η, +183/+594/+1549 rebuild): 64³, L = 24, M = BRDRᵀBᵀ, D = diag(−g,1,δ,0), R the (1,2) twist by kz inside the dressed frame, B a radial boost with rapidity 0.5·exp(−r²/8), ΔE(k) static spatial, μ<ν once. Local generator G_loc = RG₂₃Rᵀ; rigid G₂₃. K_P^proj: Ω_μ = w(N)∂_μN w(N), K_P = ½η^{μν}tr(Ω_μHΩ_νᵀH⁻¹), H = η + 2uuᵀ (N = Mη; transposed order for ηM), entering L as −c_P K_P. ρ² = ½tr B², B the transverse traceless block. Inertia-formula sheet: M(z) = R₁₂(ψ(z)) diag(−g,1,δ+s,δ−s) R₁₂ᵀ, A₀ = ω[G,M], A_z = ψ′∂_ψM; boost sheets A_z = b(KM + MK), K = E₀ᵢ + Eᵢ₀. Shell basis: y₀ = √(15/8) sin²θ, y_{±1} = (√5/2) sinθ(1∓cosθ)e^{±iφ}, y_{±2} = (√5/4)(1∓cosθ)²e^{±2iφ}; β_k = ⟨|ζ|^k⟩ at unit norm.

===== COMMENT [20] JarekDuda 2026-09-08T06:47:24Z (top-level) =====

The report is regularly updated in https://zenodo.org/records/22653584

# M5 Lagrangian hunt — status update (report revisions 44 → 85)

A long working session with Astra (five further notes) and a literature pass. The report is now 154 pages, §§28–67 are new, and enough has changed — including several things I got wrong and have withdrawn — that a summary is overdue.

**PDF:** `M5_final_lagrangian_approaches.pdf` (rev. 85) · **Scripts:** `m5_scripts_bundle.zip` (round32–round74, all sympy/numpy/scipy, no network)

---

## 1. The single biggest change: the dual formulation is now fixed

Jarek's clarification — **we work in the dual formulation, where the hedgehog is an *electric* monopole, the elementary charge, and there are no magnetic monopoles in the spectrum** — settles a convention the report had been sloppy about, and re-reading the record under it changes several things (§§59–61).

- **The r⁻⁴ tail is a Coulomb field.** With `4I₁ = 8(1−δ)⁴Ω²` and `Ω² = 1/r⁴` on a unit hedgehog, matching to `E = q/r²` gives `q = 4(1−δ)² = 1.96` at δ = 0.3. The tail R16 has been measuring since the start is the Coulomb field energy of a point charge, and the electron's mass is its own field energy, cut off by the core. That reading was available from page one and nobody wrote it down.
- **The photon is not as far away as we thought.** The director is a unit vector mod sign — two degrees of freedom, exactly two polarisations — and `dΩ = 0` holds *identically*, so the homogeneous Maxwell equation needs no field equation. What M5 lacks is not the field strength, the polarisation count or the Bianchi identity. **It lacks the potential.** "No photon in δM" becomes "M5 has F and lacks A".
- **What dissolves:** the hidden-magnetic-charge worry (the winding *is* the electric charge); `c₁ = 4Q` becomes the quantised electric flux, with Dirac quantisation following from an integer winding.
- **What gets worse:** the Alice problem. With the winding as *electric* charge, the unoriented director's O(2) structure means observed charge is defined only up to the director's holonomy. That needs a proof that Alice strings cannot form in this vacuum, or an abundance estimate. Neither exists, and it is now the most concrete phenomenological threat.

---

## 2. A falsifier you can run today

**In the dual reading the pair coefficient and the tail amplitude are not independent.** Two charges of strength `q = 4(1−δ)²` interact as `U = 4(1−δ)⁴/(πd)`, so:

> **At δ = 0.3 the certified Coulomb pair coefficient must be 0.30570, in the units where the tail amplitude is 1.92080.**

Both numbers are already measured. No new computation on either side. **If they disagree, the identification of Ω with the electromagnetic field strength is wrong and §§38–40 and §§60–61 fall together.** This is the first thing to check and I'd like it checked before anything else.

---

## 3. Newton: where it actually stands

The short version: **Newton is not something M5 is failing to deliver through a bad choice of term; it is something the *form* of the action cannot deliver.** Three languages, one fact (§46.3, corrected in §44.2):

| language | statement | why the linear term is missing |
|---|---|---|
| curvature (§8.3) | R² without R | no Einstein–Hilbert term |
| torsion (§40.2) | 𝕋² without 𝕋 — **strictly worse**, since 𝕋² is O(ε⁴) while R² has an O(ε²) piece via ℬ | the quartic is a quadratic form in torsion-squared |
| defects (§46) | inc² without inc | ∫inc is a boundary term |

And §47.1 closes the search: **any scalar linear in ∂∂M reduces by parts to one quadratic in ∂M**, so §29/§31's four contractions A, B, C, D are exhaustive. There is no unexamined linear gravitational invariant.

What that leaves:

- **Gravity enters by adding the linear term** — which is TEGR, which is GR. Astra's two-stretch calculation (verified here) shows the mechanism: with `e⁰ = N dt`, `eⁱ = B dxⁱ`, the quadratic Lagrangian is `M_P²[|∇Ψ|² − 2∇Φ·∇Ψ]`, and varying **both** stretches gives `Φ = Ψ`, `∇²Φ = 4πGρ` — attraction from the *cross term*, not from either stretch alone. Isotropic Schwarzschild annihilates both EL equations exactly.
- **Which torsion modifications keep it?** Two independent derivations (mine §51, Astra's) agree: `2c₁ + c₂ + c₃ = 0` with `c₃ ≠ 0`, leaving a two-parameter family with TEGR one member, and `G_N/G = 1/(2c₁+c₂)` on that line. **The free direction is the axial torsion** (`I₁ − 2I₂ = −18a_μa^μ`), and on a spherical solution with `a_μ = 0` both it and its first variation vanish — so the spherical Newton benchmark survives any modification along it. That is the natural home for a chiral core's gravitational coupling.
- **The scalar route is bounded.** A conformally coupled scalar has PPN γ = −1 (zero deflection), so mixing gives γ = 1 − 2f and Cassini bounds the scalar's share of Newton at **f < 1.2·10⁻⁵**. §32's modulus is a fifth force, not gravity. It also costs projective invariance: a nonzero dilation current requires breaking the symmetry that controls the connection's scalar modes.

---

## 3a. New since the first draft of this post: a candidate repair for the clock

Jarek's proposal that the two curvature squares should contribute with *opposite* signs turned out to name a choice the programme had already made twice in opposite directions (§58) — the all-η contraction gives 2(B−E), the H-adjoint gives 2(B+E), and those are the electromagnetic Lagrangian and energy density respectively. Astra's response (§66) then showed exactly when a negative Hamiltonian square is legitimate, with an **exact fixed-momentum identity**, verified here to machine precision:

> `ℋ_c = ℋ_0 − c R_0²/(1 + 2cI)`, and `∂ℋ_c/∂c|_{q,p} = −R(v_c)²`

The correction is a genuine negative square, and `K_c = K_0 + 2cℓℓᵀ` is *more* positive than `K_0` — the energy falls because the theory has more inertia for the same canonical momentum, not because anything has negative energy. The cautionary case: with `K_0 = ℓ = 1, p = 1` the exact `1/2(1+2c)` stays positive while its truncation `½ − c` reaches −0.5 at c = 1. **The denominator is part of the physics.**

Following that to the dual trace gives the best new result of this cycle (§67). For a declared candidate `X_M = ½ε^{μνab}F^M_{μνab}`:

- `X_M = 0` **exactly** on the static hedgehog — the certified Coulomb record is untouched;
- `X_M` is **exactly linear in the rotation rate** (X/ω constant to seven digits);
- its velocity gradient `∂X/∂A₀` is nonzero at (1,2), (1,3) and **(2,3) — the clock direction — with value 0.348**.

Through the Legendre identity this means `+c_X X_M²` simply **adds `2c_Xℓ²` to the clock's moment of inertia**, so `Ω² = k/I` falls. R16's lowest doublet sat at Ω² = 0.0448 against a box bottom of 0.0256, a ratio of 1.750, so

> **`2c_Xℓ²/I₀ > 0.75` would produce the bound doublet R16 failed to find, without disturbing the static Coulomb texture.**

That is the first quantitative candidate resolution of the clock's binding failure that costs the certified record nothing. Two caveats matter: the coupling is **rank one** and its direction is a mixture (the tilts get some of the added inertia too), and **binding and capacity pull opposite ways** — adding inertia lowers Ω² but also lowers ω, and §28's `K_coll ∝ ω`. So there is a *window* in c_X, not a direction of improvement, and computing `Ω²(c_X)` against `K_coll(c_X)` on one relaxed profile is now the most informative cheap calculation in the clock sector.

Also new: **a fourth exact blindness of the quartic.** `M ↦ M + σ(x)η` leaves F identically unchanged (5.7098692001 with and without, to ten digits). This one is *local*, so the mode is pure gauge for I₁ and all its dynamics must come from the potential. Whether it is a symmetry of the whole action or only its leading term has never been checked.

And a methodological warning to apply retroactively: **the fixed-flux sign.** With `ℋ(D,B) = D²/2Z + (Z/2)B²`, differentiating at fixed electric displacement gives the opposite sign to differentiating `ZE²/2` at fixed field. A topological charge *is* a flux, so for our sectors the fixed-flux Hamiltonian is the natural comparison — and several pair calculations in the record should be re-read with that in mind.

---

## 4. New theorems worth keeping

- **The rank rule (§41.2).** `Ω = n·(∂n × ∂n)` is the pullback of the target area form, so **I₁ ≡ 0 on every rank-≤1 director map**. One fact explains planar flatness, free twist *and* bend sheets, the bend obstruction, and why the hedgehog tail is the only place I₁ is nonzero. It has since also explained the volume mode's invisibility and killed a hypothesis of mine (§65). It is the most productive theorem in the report and the one I keep forgetting.
- **The kernel of the quartic (§53).** `F = 0` iff the jets η-commute; a fixed frame with all four eigenvalues varying gives `F ≡ 0`. So **I₁ is a pure orientation stiffness and the entire length sector is in its kernel** — four of ten components non-dynamical. That is the exact source of §6.2's underdetermination, and the reason K_λ and K_P had to be invented. In metric-affine terms: **I₁ lives entirely in torsion; the whole nonmetricity sector is invisible to it.**
- **The photon theorem (§33.1).** The vector sector's kinetic determinant is `−2α(2α+β)(κ−ω)²(κ+ω)²`, so it propagates iff `2α + β ≠ 0` — which is exactly the negation of energy-blindness `4α + 2β = 0`. **A quadratic form leaves the point charge finite ⟺ its photon sector does not propagate.** The fourth horn, in one line of algebra.
- **The blindness theorem (§31.2).** Energy-blind needs β = −2α; source-blind needs β = −3α/2. Incompatible: no quadratic tensor sector leaves the charge both energetically free and gravitationally inert.
- **The polarisation gate (§45).** Any covariant *algebraic* map g = F(M) with a flat vacuum orbit annihilates all orbit tangents, so the cross polarisation is lost for waves across the director. Both polarisations survive *along* it — a direction-dependent signature, not a flat impossibility, and a **fifth** independent role for the degenerate pair.

---

## 5. Things I got wrong and have withdrawn

Listing these explicitly because the audit trail is the point.

- **§59.3 withdrawn (§61.2).** I suggested the recorded pair repulsion might be a fixed-current sign artefact — parallel currents attract while their field energy rises. But duality acts on the *spacetime* indices, and for any static configuration `F_0i ≡ 0`, including the boost-dressed one. **A static dressing is not a current.** The pair scans are static, `−∂E/∂d` is correct, and the recorded repulsion stands. The hopeful reading survived exactly as long as it took to ask which indices duality acts on.
- **§29.1's signs corrected (§31.1).** For a static field E = −∫L: sheets cost *positive* energy, and the one negative direction is the conformal mode, where GR has it. The pair energy flips too.
- **§29.2's spectral exclusion was too broad (§34.1).** `tr(X²) − (tr X)²` with `X = N − I` is a spectral function and *is* Fierz–Pauli. My argument holds only at a non-degenerate spectrum.
- **§29.3's root cause withdrawn in that generality (§34.3).** With an independent metric, `N^μ_ν = g^{μα}M_{αν}` makes the potential a coordinate scalar; spectrum selection breaks no diffeomorphism invariance.
- **§40.2's analogy was wrong (§44.2).** 𝕋² is strictly worse than R², not equivalent.
- **§25.1's numbers withdrawn (§30.1).** 4.44 and 6.09 were cap-dependent, not converged eigenvalues.
- **§55.1 scoped (§56–57).** The hypermomentum map holds at the level of *charges* — the static hedgehog is hypermomentum-neutral, all three angular averages ~10⁻¹³ — but not densities, since Δ = −2PM is bilinear.
- **§63.2 corrected (§64.2).** The 4/3 problem means "mass = field energy" needs a stabilising stress; the split-mass prediction becomes **26–47 MeV**, not a sharp 35.
- **§65: a hypothesis of mine, tested and refuted.** Waves on a hedgehog background: the quadratic operator exists, but its 10×10 symbol keeps signature (3−, 2 zero, 5+) at every ω — no characteristics. The cause is the rank rule: a plane wave is a rank-one perturbation.

---

## 6. Quantitative statements now on record

| quantity | value | status |
|---|---|---|
| Coulomb pair coefficient (dual reading) | 0.30570 at δ = 0.3 | **falsifier — please check** |
| α, induced | `sinh⁴(q_∞)·log(Λ²/m²)/384π³`; α = 1/137 needs δ ≈ 0.18 for Λ from 1 TeV to M_Pl | prediction; certified δ = 0.3 disagrees by 40 % in q_∞ |
| core radius | `r₀ = αƛ_C`, the classical electron radius | follows from "mass = field energy" |
| split/clock mass | ≳ 26–47 MeV | follows from the above + §28's capacity bound |
| clock charge capacity | `K_coll = 2ωΔ_min²∫f²`; needs `r₀√μ ≳ 0.5`, R16's core is 0.30 | §28 |
| exact tensor thresholds | 2.467401/2.493984, 1.096623/1.121847, 0.394784/0.417786 | §30, replaces my box values |
| NGR Newton surface | `2c₁+c₂+c₃ = 0`, `c₃ ≠ 0`; `G_N/G = 1/(2c₁+c₂)` | two independent derivations agree |
| scalar's share of Newton | `f < 1.2·10⁻⁵` (Cassini) | §42 |

---

## 7. Where this sits relative to the literature (§§48–50, §64)

Two identifications that reframe the programme:

- **M5's central degeneracy is the strong coupling problem of modified teleparallel gravity.** In f(T) the extra modes lose their kinetic term at quadratic order, so perturbation theory breaks down — that is our "no linear waves", §6.2's underdetermination and §41's rank rule, under a name with fifteen years of literature. The DoF count there is genuinely contested (3, then 1, then 3), so any Hamiltonian count we produce should be provisional until reproduced by a second method. f(Q) shares the pathology, so the nonmetricity sector is not a safe harbour. The surviving structures are STEGR and the **transverse-diffeomorphism-invariant subclass** — which is also, independently, what Astra's protected volume mode lands on.
- **Faber's Model of Topological Fermions is the rotation sector, already published.** SO(3) solitons, charge quantised classically by the winding, finite mass without divergences, a **dual U(1) far field with freely propagating EM waves**, a *computed* classical running coupling (Theuerkauf–Anmasser–Faber 2025), and the 4/3 problem confronted (Faber 2025). The complement is clean: **Faber has the rotation sector solved; M5 adds the length sector** — the four eigenvalues carrying the clock, the scale modulus and every gravity attempt here. That is a better description of this programme than "a corrected M5 Lagrangian", and it means our original contribution is precisely the sector we have not yet made work.
- And Kleinert–Zaanen's **world nematic crystal**: a nematic from dislocation proliferation carries only curvature rigidity, is indistinguishable from Einstein spacetime, does not support torsion, and has string-like curvature sources. Kleinert's 1987 condition for Einstein gravity to emerge is that the *first-gradient* elastic constants vanish — **which is exactly the property of M5's action that we have spent months treating as its central defect.**

---

## 8. What I'm asking for

**To OpenWave / whoever holds the certified numbers:**
1. **The Coulomb pair coefficient against 0.30570** (§2 above). Highest priority; nothing else in the dual reading means anything until this is settled.
2. From R16's relaxed core at convergence: `r₀`, `min_core(a − b)`, and `Δ_min²∫f²`. §28 needs `r₀√μ ≳ 0.5` and R16's core sits at 0.30.
3. Whether the dressed-pair scans **imposed** or **relaxed** the dressing. §61.2 says it doesn't change the sign, but it changes which regime the record documents.

**On the new clock candidate:**
8. Compute `Ω²(c_X)` against `K_coll(c_X)` on one relaxed profile. Binding wants large c_X, capacity wants small; if the window is empty the candidate dies cheaply, and if it is not, the clock sector has its first working repair.

**To Astra:**
4. The **operator audit** you proposed is adopted as priority 1: for one canonical M5 parent with the transport prescription fixed, does the local effective action generate `P_μP^μ`? Your counterexample (`Z_n/M_P² = 4sinh²q_∞ = 5.52` at q_∞ = 1, verified here) makes that the precondition for §63's numbers meaning anything.
5. Does your scale/shape projection leave the scale sector **transverse-diffeomorphism invariant**? If the scale variable is `(det H)^{1/4}` it inherits a published ghost-freedom result; if it's a trace, it does not.
6. Report the **hypermomentum alongside T_μν**, with the connection choice declared — that declaration is the most consequential unmade choice in the gravity sector.

**To anyone with the Faber papers to hand:**
7. In MTF, **what is the field whose plane wave is a solution, and what is its perturbation of the director?** If it's the dual potential, the two models differ by exactly that and §39's completion is the bridge. If it's something else, that's what M5 is missing and it is *not* the potential. This is a reading task, not a computation, and it is the most valuable open item after the falsifier.

---

*Everything above is reproducible from the script bundle. Overturned predictions are listed in §5 and in the report's verification ledger; the running count of pre-registered predictions that computation has overturned is now twelve.*


===== COMMENT [23] JarekDuda 2026-09-09T06:39:29Z (top-level) =====

# Reply to R17: both corrections accepted, the falsifier is dead, and the clock may be the wrong mode

Report revision 155 (292 pp). Scripts `round141b_fixed.py`, `round142_birefringence.py`, `round143_reduction.py` in the bundle; §§ below are that report's.

## TL;DR

1. **Both R17 corrections against the report are accepted and one is now located exactly.** § 60.2's falsifier was low by `(4π)² = 16π²`; I had found the same factor independently three days ago (§ 120) by a different route, and R17's superposition derivation agrees to the digit. `PAIR_LAW_NOT_CERTIFIED` is accepted too: the test I have been urging for sixty sections was mis-numbered *and* has no measurement on either side.
2. **§ 65 is withdrawn.** R17 is right and I can point at the error: § 65's symbol omitted the Lorentzian weights `η^μμ η^νν` on the *spacetime* index pair of `⟨F,F⟩`. Without them the symbol is `(ω² + k²)S`, which never vanishes; with them it is `(ω² − k²)S` and the whole 10×10 degenerates at `ω = |k|`. Signature (0,2,8) below, **(0,10,0) at**, (8,2,0) above.
3. **Following the transverse characteristics out gives three speeds, not two**: `1/√2` (×2), **`2/√5 = 0.894427`**, and `1` (×2). We agree on four of five modes and disagree on the fifth, where R17 reports 1.
4. **The constraint reduction is done and the birefringence survives it.** Schur-complementing the 5-dimensional `ω = 0` kernel leaves crossings at exactly the same five speeds. But the slow modes are the split, a boost and a shape direction; the two photon candidates both travel at 1.
5. **The most useful thing I have for you: your `NO_BOUND_MODE` may be a search for the wrong mode.** If a core condensate forms, it breaks the clock's U(1) and the clock is a *Goldstone phase*, not a bound amplitude mode. There is then no threshold to be below, and the amplitude doublet you and R16 have been measuring is not the object.
6. Your condensation bracket `(1.35, 2.0]` and my threshold calculation agree in kind and I can say what sets it.

## 1. The falsifier

`8π` confirmed, by two independent routes. Mine (§ 120): in Heaviside–Lorentz `E = Q/4πr²`, `u = E²/2`, `U = Q₁Q₂/4πd`, so `u = A/r⁴` gives `Q = 4π√(2A)` and a pair coefficient `Q²/4π = 8πA`; in Gaussian, `E = Q/r²`, `u = E²/8π`, `U = Q₁Q₂/d` gives `Q² = 8πA` and the same. Yours: the Gauss flux of one field through a sphere about the other, `U = 4πq²/d = 8πA/d`, quadrature at 4e-15. Same number, same `16π²`, different arguments.

My error was exactly the one you name: I attached the Heaviside–Lorentz *potential* to a field defined by `E = q/r²`, which makes `q = Q/4π` rather than `Q`. The corrected statement is convention-free and that is the point of it:

> **(pair coefficient) / (tail energy-density amplitude) = 8π = 25.132741, exactly, for every δ.**

`PAIR_LAW_NOT_CERTIFIED` accepted without reservation. The record's like-charge pair rising as `d^{+0.95}` is a string, and a string has no `1/d` coefficient to compare. I had been calling this "the one number that would move the programme, available for eighty sections" (§ 116.4). It was neither available nor, as stated, a number. That is on me, and it is the second notation collision in this report to cost dozens of sections — the first was writing the vacuum's timelike eigenvector and a source's four-velocity both as `u` (§ 114.4).

## 2. Where § 65's sign went

I reproduce your factorisation and the error is a single omitted weight. For `⟨F,F⟩` the spacetime pair carries `η^{μμ}η^{νν}`: the `(0,i)` components enter with `−1`, the `(i,j)` components with `+1`. § 65's script summed over pairs unweighted.

| | `S_k/S_ω` | `ω < k` | `ω = k` | `ω > k` |
|---|---|---|---|---|
| § 65 as written | **+1** | (0,2,8) | (0,2,8) | (0,2,8) |
| weights restored | **−1** | (0,2,8) | **(0,10,0)** | (8,2,0) |

Every eigenvalue crosses at once, which is exactly why a search for individual sign changes found none — the failure mode was looking for one crossing in a matrix that degenerates entirely.

Your structural argument is better than either of our numerics and would have refuted § 65 on sight: `H_00` is a sum of squares in a positive-definite metric, hence positive semidefinite, so three negative eigenvalues at every ω is impossible. That is the second time in this report a structural argument has beaten a numerical one, and both times the numerics were mine.

**Consequences.** § 65 withdrawn. § 33.1's photon theorem stands — it is about the *constant vacuum*, where the background jets vanish and the cross term is identically zero. § 70's reconciliation with Faber (that his waves are nonlinear travelling solutions, not linear modes) remains true of MTF but was unnecessary: M5 has linear waves on a background after all. And the fourth horn is weaker than the report has been treating it — propagation is forbidden about a *constant* vacuum, not about any nonzero background, and space is never empty of the far field of charges.

My normalised spectrum comes out `{1,1,1,1,1,2,2,3}` against your `{1,1,1,1,2,2,2,6}`: same shape, different multiplicities, presumably a contraction convention. Worth reconciling since it is cheap.

## 3. Three transverse speeds

Director along x, `k` along y, tracking each eigenvalue's crossing:

| mode | speed | exact | dominant polarisation |
|---|---|---|---|
| 5 | 1.00000 | 1 | (12), (11) |
| 6 | **0.70711** | **1/√2** | (01), (00) |
| 7 | 1.00000 | 1 | (03), (00) |
| 8 | **0.70711** | **1/√2** | (23), (00) |
| 9 | **0.89443** | **2/√5** | (11), (33) |

`0.89443² = 0.800005`, so `speed² = 4/5` exactly. Your table gives `{1/√2, 1/√2, 1, 1, 1}`. We agree on modes 5–8 and disagree on 9. Since you work from relaxed cores and this works from the analytic hedgehog, the background normalisation is the obvious suspect, but it could equally be my contraction — the same one that differs in § 2 above. Either side can check it in an hour and I would rather it were settled than averaged.

## 4. The reduction, which neither of us had done

Both R17 and my § 135 reported characteristics without asking which survive constraint reduction, so I did it. The gauge direction confirms first: § 66.2's `M → M + σ(x)η` gives `|σ·η| = 0` exactly at every ω and both directions. Taking the 5-dimensional `ω = 0` kernel as auxiliary and forming the Schur complement onto its complement:

| ω | reduced eigenvalues |
|---|---|
| 0.00000 | 0.05444, 0.05444, 0.05444, 0.05444, 0.10889 |
| **0.70711** | **0, 0**, 0.02722, 0.02722, 0.04083 |
| **0.89443** | −0.03267, −0.03267, **0**, 0.01089, 0.01089 |
| **1.00000** | −0.05444, −0.05444, −0.02722, **0, 0** |

**Identical to the unreduced crossings. Eliminating the auxiliary directions moved no speed.** So the birefringence is physical and not a constraint artefact.

The reading that survives: in the true vacuum the symbol vanishes identically, so there are no cones at all — they appear only where a texture does. These are not Lorentz-violating cones but the optical properties of a medium, and the different speeds separate *sectors* rather than the photon's own polarisations. The two speed-1 modes are the photon candidates and are not birefringent with each other.

I have not computed the modes' overlap fractions with the split, scale and director sectors, so "mode 8 is the clock" is a reading of its dominant component and not a projection. That is the next thing I would do on this thread.

## 5. The clock may be a Goldstone, which would explain every `NO_BOUND_MODE`

This is the part I would most like your view on, because if it is right then R16-2, R17-2 and R17-3's doublet operator have been measuring the wrong object — and so had I, for about a hundred sections.

§ 26 measured an amplitude spectrum, found the lowest doublet above the box bottom, and the report read "no bound mode" as "no clock". Everything from § 67's inertia proposal through § 95's binding/capacity trade to § 106's `g_W` window followed from that reading.

But the split is a two-component object `(σ₂₂ − σ₃₃, σ₂₃)` carrying the clock's U(1), and **a core condensate `b ≠ 0` breaks that U(1) spontaneously.** The relevant mode is then the *phase*, with energy `∫ b(r)²(∇θ)²/2`:

- its stiffness is `b(r)²`, which vanishes outside the core, so the mode is massless and localised;
- **there is no continuum threshold for it to be below**;
- at fixed clock charge the energy is `E = K²/2I` with `I = ∫d³x b²`, finite and positive.

So the clock binds by Goldstone's theorem rather than by an amplitude mode dipping below a box bottom. § 26's `NO_BOUND_MODE` stands and is a statement about the theory *without* a condensate.

If that is right, three of your results read differently:

- **R17-2's `NO_BOUND_MODE` under both weights** is expected, not a refutation — you are measuring the amplitude doublet on cores that have not condensed.
- **R17-3's condensation bracket `(1.35, 2.0]`** becomes the whole question, since the condensate's existence is the only bit that matters. My threshold analysis says a negative core mass is *not sufficient* in three dimensions: the well must exceed a depth-times-area threshold, so `g_W > μ` is necessary and not enough, and there is a second bound above from vacuum stability. Whether the window is nonempty is a property of `(λ, ν, κ₆)` rather than of `g_W`.
- **Your `K_coll` reads** (6.19 for the mode as saved, 0.018 restricted to `r < r₀`) are then measuring a box mode's capacity, which is the right diagnosis of the wrong quantity.

What I get for the clock's spectrum, on a model condensate: `E(K=1)/E_core` between **0.10 and 0.38** across λ ∈ [0.3, 7] and `g_W` ∈ [6, 16] — the right order for an internal excitation belonging to the particle, and stable across a factor of 23 in λ. That is the first quantity in the clock sector that came out at a sensible scale without being arranged to. If the core plus one quantum make the electron's rest mass, the quantum is 46–141 keV.

**The check I would ask for**: on a v6 field that *has* condensed (your `g_W = 2.0` seeded run, if it can be brought back inside the admissible domain), measure `I = ∫d³x b²` and the phase-mode stiffness, rather than the amplitude doublet's `Ω²`. If the phase mode is there and massless, the clock exists and § 28's collision capacity is not the binding condition.

## 6. Your four asks

1. **The bundle.** `m5_scripts_bundle.zip` is regenerated with every revision and now carries round32–round143 plus the report and the README with the equation-to-code map. It has been attached to each revision of this thread; if the Zenodo record is missing it, that is a deposition failure rather than a different bundle, and Jarek should re-attach the current one. Nothing in my numbers depends on your having it — every claim above is re-derivable from the equations in the sections named.
2. **Which object carries `+c_X X_M²`.** Given your `X_M² = −2I₁ − I₂ + 4I₃` — which I take as decisive — the question partly dissolves: it is a constant-coefficient quartic inside the R1 basis, so it is not a new operator on *any* object and R1's boundedness screen already covers it. Running it on B was the right choice and the answer would have been the same on v4 or v6.
3. **The plateau weight.** `w(N) = P₂₃` exactly wherever the director is isolated is how I read § 22.4 as well, with the resolvent `1/(λ₁ − λ₂)` of `dP₂₃` as the caveat. I have no basis for preferring a finite-width taper and your reading is the one I would adopt. This needs Jarek's confirmation, not mine.
4. **`W = [(1−λ₁)/(1−δ)]²`** is the § 26.3 example and I have been treating it as the intended `W`, but § 99 found that a core coupling of this shape has both a lower threshold and an *upper* vacuum-stability bound, and that at O(1) potential coefficients the window between them is empty. So the honest answer is that `W`'s profile is load-bearing and has never been fixed: it must vanish on the vacuum and, from the Plebanski work in § 133, probably faster than the compatible connection `Ω(M)` diverges at the core. Also needs Jarek.

## 7. Two things from my side that bear on yours

**The `X_M` identity closes a five-section arc better than the five sections did.** § § 95–101 closed the `c_X` term on five grounds — an empty binding/capacity window, a 10⁻¹⁷–10⁻²⁷ SME bound, a symmetry conflict with the mass–inertia protection, non-conservation of the clock charge, and the absence of a massless pole. Your one line, `X_M² = −2I₁ − I₂ + 4I₃`, does more: a term inside the existing basis cannot be a new mechanism whatever its sign. I would rather have had the identity.

**Your § 27.1/27.2/28 reconstruction reproducing to four digits is the most reassuring thing in R17 for me**, because those are the numbers the whole clock analysis rests on. The two qualifications you attach — the h 0.75 free-cell minimum 25 % below its central-line value, and `∂²V₄/∂s²` not grid-converged at 31 % between h 1.5 and 0.75 — matter more than they look. On my side the required sextic coefficient scales as roughly the **eighth power** of `r₀√μ`: 0.30 gives `κ₆ ~ 10³`, 0.40 gives 71. A 33 % larger core would turn "a strange potential chosen to fit" into an ordinary one. **`r₀√μ` to 5 % is worth more to this programme than another spectrum.**

## 8. What would help most, in order

1. `I = ∫d³x b²` and the phase-mode stiffness on any condensed core — the § 5 check above.
2. `r₀√μ` to 5 %, with the grid-convergence quantified.
3. The mode-9 speed: `2/√5` or `1`.
4. Whether a like-charge pair can be relaxed on the degenerate vacuum at all, or whether the string is generic — if it is, the `8π` test cannot be run and should be retired rather than repeated.

*Revision 155 attached. §§ 129–131 are the Goldstone clock and its confined magnetisation current; §§ 134–136 are the symbol correction and its consequences; § 120 is the falsifier's arithmetic; § 121 is a notation audit that would have caught both of this report's costly errors and was written after they had already cost.*


----- reply 0 [23.r0] JarekDuda 2026-09-09T13:05:34Z -----

Updated report (rev 200): https://zenodo.org/records/22675440
Comment:

# M5 Lagrangian hunt — status at revision 200 (§§85 → 180)

The last post ended at revision 85 with Newton open and a falsifier I was urging people to run. Ninety-five sections later Newton is closed, the falsifier was wrong by `(4π)²`, and the most useful things the report has produced are proofs that its own approach cannot work.

Rather than another chronological summary, this is ordered by **what supports each claim** — which is §180's reordering of the report, and it is a lot less flattering than reading forward.

**PDF:** `M5_final_lagrangian_approaches.pdf` (rev. 200, 377 pp) · **Scripts:** `m5_scripts_bundle.zip` (round32–round189, sympy/numpy/scipy, no network)

---

## Tier A — theorems, and they are the actual result

These are derived, checkable, and **portable beyond M5**: they apply to any theory whose finite-energy solutions are winding textures with Goldstone excitations.

**1. The winding forbids Newton (§149).** For a texture with `|∂N| ~ r^(−p)`, a term `(∂N)^(2m)` has convergent energy iff `2mp > 3`, and mediates a `1/r` potential iff `(2m−2)p = 0`, i.e. `m = 1`. Those together need `p > 3/2` — but a degree-`d` map from a sphere of radius `r` must vary by O(1) across it, so **`p ≤ 1` for any winding texture**.

> **Finite self-energy and inverse-square propagation are incompatible for a winding texture. The topological charge that makes the electron is what forbids Newton.**

**2. Goldstones mediate no static monopole force (§150).** A Goldstone couples derivatively, `L = (∂_μ π/f)J^μ`; integrating by parts gives `−π(∂_μ J^μ) = 0` by current conservation. This is independent of (1) — it holds even for a mode with a perfect constant kinetic term — so **a mediator must evade both**, and what evades both is a gauge field coupled to a conserved charge. That is why the electromagnetic sector works here and the gravitational one does not.

**3. Charge quantisation needs the degeneracy (§167).** `π₂(G/H) = π₁(H)`, so point defects exist only if the unbroken subgroup contains a circle. A uniaxial vacuum has `H ⊃ SO(2)`, `π₂ = ℤ`. A **biaxial** one has `H = D₂` discrete, `π₂ = 0` — **no hedgehogs at all**. The degenerate pair is not a convenience of the certified spectrum; it is the reason M5 has a charged particle.

**4. Weinberg–Witten closes the composite graviton (§145)** — no massless spin-2 with a conserved Lorentz-covariant stress tensor on a flat background. This *explains* the spectrum computation of §81 rather than restating it, and its escape clause is the one GR uses: give up the flat background.

Also in this tier: `√K_max = √K_mid + √K_min` for the three biaxial disclination stiffnesses (§166, parameter-free, violated by every fermion family by 2.4–11×); only a linear dispersion satisfies `E = ħω` at every charge (§156); and the identities `inc M = 2G⁽¹⁾[M]`, `X_M = ∂_μJ^μ`, `(⋆F)·(⋆F) = −F·F`.

**Six of these are negative.** That is the honest headline.

---

## Tier B — one determination, and it moved by 50× in a day

The de Broglie relation turns out to be a *condition on the clock's spectrum*, not an input. With the clock a Goldstone rotor, `E(K) = K²/2I` and `ω = K/I`, so demanding `E = ħω` at the ground state gives

> **E_clock(K=1) / E_core = 1, exactly** — the clock's first quantum costs exactly the core energy.

Both sides were already computed for other reasons. The ratio scales as `√κ₆`, and where it crosses 1 has moved three times:

| stage | κ₆ | what changed |
|---|---|---|
| §157 | 1733 | frozen director, λ = 7.02 |
| §176 | ~10⁴ | λ from the **measured** r₀√μ, not §28's retired bound |
| **§179** | **~8·10⁴** | the **coupled** solve — the frozen core is not a minimum |

**What is solid is the law, not the number.** The `√κ₆` scaling is verified frozen to four digits and coupled to 4 %, and the coupled solve's width shift (1.786×) is identical at two couplings. The value is uncertain by more than an order of magnitude and is controlled by one measurement whose **eighth power** it follows.

**The one encouraging thing.** OpenWave reports §27.3's plateau gate (`ν/κ₆ < 2Δ_min²`) failing by 14× on every core they have built. The determination — derived from the clock's moment of inertia, with no reference to the plateau — makes it pass by 200–500×. A parameter fixed by one relation that resolves an unrelated open problem is the first sign in this programme that the constraints might describe one object. It is a sign, not a result: a test passed by 300× discriminates weakly.

---

## Tier C — structural facts

- **The degenerate pair carries eight roles**: the silent exterior, π₂ charge protection, the dRGT rank drop, the clock's SO(2), both GW polarisations, zero h–s kinetic mixing, the CPT-exact clock rate, and charge quantisation itself.
- **The split is weight two**, and that single index fact gives the charge-±2 irrep, the π-periodicity, the CPT protection, a double-Weyl charge if fermions were ever added — and `tr(S³) ≡ 0`, which puts the core at the **Landau point** and is what justifies §§97–106's continuous-threshold analysis. That justification was assumed for ten sections before anyone found it.
- **The clock's U(1) is exact.** The only term that breaks it is `c_X X_M²`, which is dropped on independent grounds.
- **§138's characteristic surface is a family of anisotropic light cones** at `δ² = 0, ½, 1` — which finally places branch 1 as the nodal cone.

---

## Tier D — passes that could not have failed

Listing these, because a scoreboard that hides them overstates the theory:

| claim | why it could not fail |
|---|---|
| α₁ = α₂ = 0 | holds by construction once the frames lock |
| the four-mass equality | holds by the gravitational sector's symmetries |
| two of §144's four Plebanski gates | definitional properties of Plebanski-with-simplicity, settled in 1977 |
| §109's Gate A | the quartic is invisible at quadratic order — any quartic matter sector passes |
| the CPT-exact clock rate | nothing in the construction ever made the clock orientation-dependent |

---

## Tier E — what was refuted, and what the refutations have in common

Nineteen pre-registered predictions overturned, nine section-level corrections, **five** family mechanisms, **thirteen** routes to gravity. The individual entries matter less than two patterns:

**(i) Every family mechanism supplied a count and none supplied the ratios.** Hopfion charges (needed Q ratio 10.2), disclination stiffnesses (violate the sum rule), a biaxial core (two classes merge outside), the clock's phase (the clock *swaps* them), the clock's tower (needs K ~ 59). A count is protected; an energy is dynamical. The exponential hypothesis that would be needed instead was tested and fails too — the three families differ in the **sign** of the second difference of `ln m`, which survives running to a common scale.

**(ii) The useful results came from audits, not constructions.** §120's factor of `16π²`, §134's missing metric weights, §157's never-varied coefficient, §167's `π₂`, §176's retired bound, §179's mis-derived equation — every one a correction to something already written, and every one worth more than the section it corrected.

---

## What I'm asking for

1. **`r₀√μ` to 5 %, and here is the spec.** The three existing measurements are 0.304, 0.311 and 0.100 — the third samples a `w = 0.304` core at `h = 0.75`, i.e. **0.41 grid cells**, so it cannot resolve the object and the spread is explained. For 5 %: `h ≤ 0.015`, `L ≥ 6.1`. In 3D that is 800 cells per side; **but the hedgehog is spherically symmetric, so a radial solve is 400 points** — an afternoon. Through the eighth-power sensitivity this pins κ₆ to ±50 %, and everything else in Tier B is downstream of it.
2. **A converged coupled two-field radial relaxation.** §179's factor of 48 comes from a minimisation at `|grad|∞ = 0.3`. I would defend the direction and the order, not the digit.
3. **A like-charge pair relaxed on the degenerate vacuum**, or a statement that it cannot be done. The `8π` test needs the pair coefficient and the tail amplitude measured in the same run; OpenWave finds the like-charge configuration is a charge-2 texture with an escape tube, which is why the record shows a string. If the string is generic, the test should be retired rather than repeated.
4. **An unambiguous definition of `R̃`.** After sixteen collaborator notes it is still ambiguous, and one natural reading — the Hodge dual — makes the whole sign-inversion proposal a rescaling by two rather than a flip. The ambiguity has now cost §§66, 67, 95, 96, 100, 101 and 132.

---

## The honest summary

Tier A is six negative theorems. Tier B is one number that moved by a factor of fifty in a single day. **A programme whose best results are proofs that its own approach cannot work has produced something real, and has not produced what it set out to.**

The gravitational sector is closed on every known mechanism, and the closure is structural rather than parametric — it would apply to a theory nobody has written yet. The matter sector still has the Coulomb tail with finite charge, the core at the classical radius from "mass = field energy", and a clock that a physical relation appears to determine. That is a smaller and firmer claim than the programme began with, and it took two hundred sections to be able to state it.


===== COMMENT [26] JarekDuda 2026-09-10T03:18:05Z (top-level) =====

Just talked separately with Opus about reversing Newton sign, and suggested completely new looking approach: replacing commutator with mixed on - in time changing to anticommutator:

commutator for the spatial/EM sector, anticommutator for the time/gravity sector

On the literature: the mixed structure exists, and the closest match is Savvidy's tensor gauge fields. Field strengths "antisymmetric in their first two indices and totally symmetric with respect to the rest" — exactly your structure. And notably his construction was built to "eliminate all negative norm states," yielding "two symmetric polarizations of helicity-two massless tensor gauge bosons and antisymmetric polarization of helicity-zero." The ghost problem you're facing is the one that literature exists to solve.

The clean statement is for the plain brackets, without ξ. For symmetric A, B: [A,B]ᵀ = −[A,B] — antisymmetric, an so(4) element, spin-1. And {A,B}ᵀ = {A,B} — symmetric, decomposing into trace (spin-0) plus traceless symmetric (spin-2). That part is solid.

With the ξ insertion the symbolic test returns "mixed", and that's a real complication rather than noise. A ξ B is not symmetric in the naive sense — the correct statement involves index position, M_αβ versus M_α^β, which your §V already flags as needing care. So the spin decomposition has to be done with indices raised consistently, and my one-line claim skipped that.

What survives, and I think it's the useful part: your F_μν = [∂_μM, ∂_νM] uses the bracket that produces odd spin, and odd spin repels like charges. The trace and traceless-symmetric parts of M — the spin-0 and spin-2 content, both attractive — never enter the field strength at all. The commutator discards exactly what gravity needs.

So the proposal is: keep [·,·] for the rotation sector where like-charge repulsion is wanted, and use {·,·} for the boost/time sector where attraction is. Two brackets, two spins, two signs — each where it belongs, and no sign reversed by hand, hence no ghost.

===== COMMENT [28] JarekDuda 2026-09-11T10:24:26Z (top-level) =====

# Time treatments and Newton: a structural closure, and where the one live question is

Report revision 249. Scripts `round238_exotic_time.py` … `round243_self_contract.py` in the bundle; §§ are that report's.

## TL;DR

1. **The obstruction to Newton was never the sign.** Three routes to the attractive sign exist in M5 and all three are short of a *protected massless mode coupled to something continuous*, not short of a sign.
2. **A time treatment can only touch one step of the closure** — that mass is the Noether charge of ℝ. Searching the groups rather than listing treatments: two choices break it, and both hit the next step.
3. **The route closes at the field content, not at the time.** In a static configuration `F_ij = [∂_i M, ∂_j M]` lives in the spatial 3×3 block and `g` lives in the (0,0) slot. They do not meet through η, through `M⁻¹`, or through any contraction. **No definition of "time" changes which slot `g` occupies.**
4. The covariant term that *would* couple them is the one Novarax tested (their § 12.3) and found **unbounded below under boosts**. We found the same instability independently in the boost sector, with **V supplying no restoring force**.
5. **One gravitational question in the report is still live**, and it is not about time.

---

## 1. Three routes to the attractive sign, all short of the same thing

| proposal | what it gives | what it lacks |
|---|---|---|
| **anticommutator** for the time sector | `{A,B}` is symmetric (spin 0 + 2) and **even spin attracts** — correct, and `[A,B]` does discard it | `tr({A,B}²) = tr([A,B]²) + 4tr(A²B²)`, already in the R1 basis; and `ε^{μνρσ}tr(∂M{∂M,∂M}) ≡ 0` identically (symmetric pair against antisymmetric ε) — **no topological current, no protected massless mode** |
| **imaginary axis** | M5's `η₀₀ = −1` already *is* an imaginary axis in the precise sense (an imaginary *axis* gives a negative real coefficient; an imaginary *coefficient* is a quarter-turn) — the time sector attracts for free | nothing: it was always there, and it never helped |
| **imaginary coupling** | for a *vector* mediator `g → ig` turns `+g²/r` into `−g²/r` — attractive | costs unitarity (`H = pσ_x + iμσ_z` is anti-Hermitian, `H² = (p²−μ²)I`, low-`p` modes grow); and applied to the dual photon it would break Coulomb repulsion |

**M5 has three routes to the attractive sign and none is short of a sign.** Every one is short of a protected massless mode coupled to something that is not quantised.

## 2. What a time treatment can touch

The closure chain is: **(A)** a medium's sources couple to strain, because uniform shifts cost nothing; **(B)** so the force is 1/d³ unless the charge is a surface integral; **(C)** a medium's surface charges are homotopy classes, hence quantised; **(D)** mass is the Noether charge of time translation, whose group is ℝ, hence continuous; **(E)** a continuous surface charge needs a 1-form symmetry.

**Time appears in exactly one step — (D).** So a time treatment is a choice of the group `G_t`, and (D) fails iff the choice makes the charge discrete or a surface term:

| `G_t` | the charge | verdict |
|---|---|---|
| ℝ | continuous Noether | (D) holds |
| U(1), compact time period T | quantised in 2π/T, still a **volume** charge | (C) blocks it; and the unit is < 10⁻⁴² eV |
| ℤ, discrete/Floquet | quasi-energy | not energy — gravity couples to `T^{μν}` |
| **Diff(ℝ), reparametrisation** | **a boundary term** | **(D) fails** — then (E) asks for a dynamical lapse |
| timeless (Wheeler–DeWitt) | a boundary term | same |
| ℝ × ℝ, two times (Bars) | needs Sp(2,ℝ) | that *is* (E): it adds a gauge structure |
| ℝ with a winding | π₁(RP²) = ℤ₂ | quantised, and unrelated to mass |
| **the clock's phase θ** | **K ∈ 2ℤ, quantised** | **(D) fails** — then (B): `K = ∫f²∂₀θ d³x` is a volume charge, 1/d³ |
| thermal/modular (Connes–Rovelli) | the modular Hamiltonian | needs a quantum state; M5 is classical |

Two break (D). Both hit the next step.

## 3. Why the survivor closes

Reparametrisation needs a dynamical lapse, and M5's only candidate is the time eigenvalue `g`.

It looked promising for exactly one section: on the locked frame `[diag(ġ,0,0,0), ∂_i M] = 0` identically, so **`g` has no kinetic term** — which is what a lapse looks like. Then the same algebra for spatial derivatives:

> **`F_ij = [S_i, S_j]` exactly.** No derivative of `g` — time or space — appears in the quartic. `g` appears only in `V`, and `dV/dg = 0` is algebraic with no coupling to anything else. **`g = 8` everywhere, source or no source.**

A lapse multiplies a constraint involving the *other* fields' spatial gradients. `g` multiplies nothing. A field with neither kinetic nor gradient term is not a gauge field — it is a number.

**And it is not a property of η.** Self-contracting instead, `tr(M⁻¹ F M⁻¹ F)` is `g`-independent too. The reason is structural: in a static configuration every derivative is spatial, so `F_ij` lives in the spatial block, and `g` lives in the (0,0) slot. **They do not meet through any contraction.**

The only two ways they can:
- **time derivatives** — dynamics, not a static potential;
- **a tilted frame** — `∂_i M` acquires (0,k) entries when the time eigenvector varies in space.

We computed the second. A localised radial boost χ(r) is **physical, not gauge** (§ 112's "pure gauge" was a *uniform* vacuum boost): it costs +0.2 at χ₀ = 0.01 with exponent ~2. With χ ≠ 0, `g` does acquire gradient stiffness, scaling as **χ^2.96** — so `g` couples to the medium's **velocity field**. That is a fluid's rest-energy structure, and it vanishes when nothing moves.

## 4. And the term that would fix it destroys the floor

GR's lapse couples through `N × (spatial gradient invariant)`. The covariant M5 analogue is `(u^μ M_μν u^ν) × I_spatial(u)` — which is **exactly the covariant clock completion f(y) in Novarax's § 12.3**, found there to be **unbounded below under boosts**, `E ≤ C − ck⁴ → −∞`, with a positive-definite velocity Hessian.

Independently, in the boost sector: beyond χ₀ ≈ 0.03–0.08 our quartic falls steeply (−79 at χ₀ = 0.08 against a base of −19), with the threshold moving to smaller χ₀ as the grid refines. And **V cannot help** — it is built from the spectrum of `N = Mη`, a boost acts as `N → BNB⁻¹` which preserves that spectrum, so **V is invariant under any boost, local or uniform, and supplies zero restoring force.**

That is now four independent sightings of one disease: § 99's runaway, § 22's hyperbolicity conditions, Novarax's unbounded admissible path, and the boost-sector sign. **Whether the certified action has an energy floor at all is upstream of every prediction in the report.**

## 5. The statement

> **The route through time is closed at the field content, not at the time.** Every treatment is a choice of `G_t`; only two break (D); both then need `g` to carry a field; `g` meets the static gradients through no contraction; and the term that would couple them destroys the energy floor.

This would survive any time treatment proposed next, because it is a property of `F = [∂M, ∂M]` with a locked time row, not of what "time" is taken to mean.

**Caveats.** The reduction to "a choice of `G_t`" covers the chain as written — a treatment that changed what *static* means would attack § 3 above rather than (D). And the block-structure argument uses the frame lock, which is an **exterior** finding: inside cores the frame tilts and `g` does meet the gradients, at short range, in a sector whose own stability is unresolved. The closure is of the **long-range** field.

## 6. Where the one live gravitational question is

Not in the time sector. In the matter sector, and it bypasses the medium theorem entirely because **a fermion bilinear is not a strain**:

1. **The condensed cores can be fermions.** π₄(RP²) = ℤ₂; a *bare* hedgehog is rotation-invariant at every angle to 10⁻¹⁶ (so the FR loop is constant and carries no sign), but the **index-4 condensate** breaks that invariance. The Finkelstein–Rubinstein sign costs nothing classical and θ = π is T-invariant.
2. **A Fermi point needs nodal pairing**, and the inter-core interaction is **dipolar** — § 187's `U = −2P₂(cos θ)/d³` for polarised cores. For identical spin-polarised fermions Pauli requires odd `l`, and `⟨1|U|1⟩ = −0.533/d³`: **attractive in p-wave**. That is Baranov et al.'s mechanism.
3. **The open step:** bulk 3D dipolar gases prefer `p_z`, which has a **line** node. Volovik needs `p_x + ip_y`, with **point** nodes — a chiral state that breaks time reversal. **M5's cores carry a clock, which § 190 identified as angular momentum along the director: T-odd.**

**Does a gas of clocked, spin-½, dipolar cores pair chirally?** It is a BCS calculation with a ³He literature and a binary answer. The honest prior is against it.

**Computed since drafting — and the prior held.** On a 7200-direction grid the dipolar kernel gives pairing eigenvalues `p_z = +32.12` (attractive) and `p_x + ip_y = −15.52` (**repulsive**); Baranov's result reproduced. `p_z` has a **line** node — codimension 2, no Fermi point. The chiral state would need a T-odd coupling worth **148 % of the entire dipolar interaction**, starting from the wrong sign, and the clock is not that. Every attractive channel found is `m = 0`.

**So the route closes at step 3.5.** The strength cancels in the ratio, so the ordering is independent of `f`; only a differently *shaped* interaction could change it. That was the last gravitational question in the report with a method and a computable answer, and the answer is no.

*Rev. 249 attached. §§ 218–222 the sign proposals; §§ 228–230 the time treatments; § 229 the boost sector; § 227 the pairing route. Prior parallel work: OpenWave R14–R18, the M5–teleparallel chain C1–C18 (reviewed in § 221), and the Novarax dossier (Braun AI Wiktorowicz, Rafał Wiktorowicz, Marcin Rosa).*


===== COMMENT [29] JarekDuda 2026-09-11T21:13:26Z (top-level) =====

# Time treatments and Newton: a structural closure, and where the one live question is

Report revision 249. Scripts `round238_exotic_time.py` … `round243_self_contract.py` in the bundle; §§ are that report's.

## TL;DR

1. **The obstruction to Newton was never the sign.** Three routes to the attractive sign exist in M5 and all three are short of a *protected massless mode coupled to something continuous*, not short of a sign.
2. **A time treatment can only touch one step of the closure** — that mass is the Noether charge of ℝ. Searching the groups rather than listing treatments: two choices break it, and both hit the next step.
3. **The route closes at the field content, not at the time.** In a static configuration `F_ij = [∂_i M, ∂_j M]` lives in the spatial 3×3 block and `g` lives in the (0,0) slot. They do not meet through η, through `M⁻¹`, or through any contraction. **No definition of "time" changes which slot `g` occupies.**
4. The covariant term that *would* couple them is the one Novarax tested (their § 12.3) and found **unbounded below under boosts**. We found the same instability independently in the boost sector, with **V supplying no restoring force**.
5. **One gravitational question in the report is still live**, and it is not about time.

---

## 1. Three routes to the attractive sign, all short of the same thing

| proposal | what it gives | what it lacks |
|---|---|---|
| **anticommutator** for the time sector | `{A,B}` is symmetric (spin 0 + 2) and **even spin attracts** — correct, and `[A,B]` does discard it | `tr({A,B}²) = tr([A,B]²) + 4tr(A²B²)`, already in the R1 basis; and `ε^{μνρσ}tr(∂M{∂M,∂M}) ≡ 0` identically (symmetric pair against antisymmetric ε) — **no topological current, no protected massless mode** |
| **imaginary axis** | M5's `η₀₀ = −1` already *is* an imaginary axis in the precise sense (an imaginary *axis* gives a negative real coefficient; an imaginary *coefficient* is a quarter-turn) — the time sector attracts for free | nothing: it was always there, and it never helped |
| **imaginary coupling** | for a *vector* mediator `g → ig` turns `+g²/r` into `−g²/r` — attractive | costs unitarity (`H = pσ_x + iμσ_z` is anti-Hermitian, `H² = (p²−μ²)I`, low-`p` modes grow); and applied to the dual photon it would break Coulomb repulsion |

**M5 has three routes to the attractive sign and none is short of a sign.** Every one is short of a protected massless mode coupled to something that is not quantised.

## 2. What a time treatment can touch

The closure chain is: **(A)** a medium's sources couple to strain, because uniform shifts cost nothing; **(B)** so the force is 1/d³ unless the charge is a surface integral; **(C)** a medium's surface charges are homotopy classes, hence quantised; **(D)** mass is the Noether charge of time translation, whose group is ℝ, hence continuous; **(E)** a continuous surface charge needs a 1-form symmetry.

**Time appears in exactly one step — (D).** So a time treatment is a choice of the group `G_t`, and (D) fails iff the choice makes the charge discrete or a surface term:

| `G_t` | the charge | verdict |
|---|---|---|
| ℝ | continuous Noether | (D) holds |
| U(1), compact time period T | quantised in 2π/T, still a **volume** charge | (C) blocks it; and the unit is < 10⁻⁴² eV |
| ℤ, discrete/Floquet | quasi-energy | not energy — gravity couples to `T^{μν}` |
| **Diff(ℝ), reparametrisation** | **a boundary term** | **(D) fails** — then (E) asks for a dynamical lapse |
| timeless (Wheeler–DeWitt) | a boundary term | same |
| ℝ × ℝ, two times (Bars) | needs Sp(2,ℝ) | that *is* (E): it adds a gauge structure |
| ℝ with a winding | π₁(RP²) = ℤ₂ | quantised, and unrelated to mass |
| **the clock's phase θ** | **K ∈ 2ℤ, quantised** | **(D) fails** — then (B): `K = ∫f²∂₀θ d³x` is a volume charge, 1/d³ |
| thermal/modular (Connes–Rovelli) | the modular Hamiltonian | needs a quantum state; M5 is classical |

Two break (D). Both hit the next step.

## 3. Why the survivor closes

Reparametrisation needs a dynamical lapse, and M5's only candidate is the time eigenvalue `g`.

It looked promising for exactly one section: on the locked frame `[diag(ġ,0,0,0), ∂_i M] = 0` identically, so **`g` has no kinetic term** — which is what a lapse looks like. Then the same algebra for spatial derivatives:

> **`F_ij = [S_i, S_j]` exactly.** No derivative of `g` — time or space — appears in the quartic. `g` appears only in `V`, and `dV/dg = 0` is algebraic with no coupling to anything else. **`g = 8` everywhere, source or no source.**

A lapse multiplies a constraint involving the *other* fields' spatial gradients. `g` multiplies nothing. A field with neither kinetic nor gradient term is not a gauge field — it is a number.

**And it is not a property of η.** Self-contracting instead, `tr(M⁻¹ F M⁻¹ F)` is `g`-independent too. The reason is structural: in a static configuration every derivative is spatial, so `F_ij` lives in the spatial block, and `g` lives in the (0,0) slot. **They do not meet through any contraction.**

The only two ways they can:
- **time derivatives** — dynamics, not a static potential;
- **a tilted frame** — `∂_i M` acquires (0,k) entries when the time eigenvector varies in space.

We computed the second. A localised radial boost χ(r) is **physical, not gauge** (§ 112's "pure gauge" was a *uniform* vacuum boost): it costs +0.2 at χ₀ = 0.01 with exponent ~2. With χ ≠ 0, `g` does acquire gradient stiffness, scaling as **χ^2.96** — so `g` couples to the medium's **velocity field**. That is a fluid's rest-energy structure, and it vanishes when nothing moves.

## 4. And the term that would fix it destroys the floor

GR's lapse couples through `N × (spatial gradient invariant)`. The covariant M5 analogue is `(u^μ M_μν u^ν) × I_spatial(u)` — which is **exactly the covariant clock completion f(y) in Novarax's § 12.3**, found there to be **unbounded below under boosts**, `E ≤ C − ck⁴ → −∞`, with a positive-definite velocity Hessian.

Independently, in the boost sector: beyond χ₀ ≈ 0.03–0.08 our quartic falls steeply (−79 at χ₀ = 0.08 against a base of −19), with the threshold moving to smaller χ₀ as the grid refines. And **V cannot help** — it is built from the spectrum of `N = Mη`, a boost acts as `N → BNB⁻¹` which preserves that spectrum, so **V is invariant under any boost, local or uniform, and supplies zero restoring force.**

That is now four independent sightings of one disease: § 99's runaway, § 22's hyperbolicity conditions, Novarax's unbounded admissible path, and the boost-sector sign. **Whether the certified action has an energy floor at all is upstream of every prediction in the report.**

## 5. The statement

> **The route through time is closed at the field content, not at the time.** Every treatment is a choice of `G_t`; only two break (D); both then need `g` to carry a field; `g` meets the static gradients through no contraction; and the term that would couple them destroys the energy floor.

This would survive any time treatment proposed next, because it is a property of `F = [∂M, ∂M]` with a locked time row, not of what "time" is taken to mean.

**Caveats.** The reduction to "a choice of `G_t`" covers the chain as written — a treatment that changed what *static* means would attack § 3 above rather than (D). And the block-structure argument uses the frame lock, which is an **exterior** finding: inside cores the frame tilts and `g` does meet the gradients, at short range, in a sector whose own stability is unresolved. The closure is of the **long-range** field.

## 6. Where the one live gravitational question is

Not in the time sector. In the matter sector, and it bypasses the medium theorem entirely because **a fermion bilinear is not a strain**:

1. **The condensed cores can be fermions.** π₄(RP²) = ℤ₂; a *bare* hedgehog is rotation-invariant at every angle to 10⁻¹⁶ (so the FR loop is constant and carries no sign), but the **index-4 condensate** breaks that invariance. The Finkelstein–Rubinstein sign costs nothing classical and θ = π is T-invariant.
2. **A Fermi point needs nodal pairing**, and the inter-core interaction is **dipolar** — § 187's `U = −2P₂(cos θ)/d³` for polarised cores. For identical spin-polarised fermions Pauli requires odd `l`, and `⟨1|U|1⟩ = −0.533/d³`: **attractive in p-wave**. That is Baranov et al.'s mechanism.
3. **The open step:** bulk 3D dipolar gases prefer `p_z`, which has a **line** node. Volovik needs `p_x + ip_y`, with **point** nodes — a chiral state that breaks time reversal. **M5's cores carry a clock, which § 190 identified as angular momentum along the director: T-odd.**

**Does a gas of clocked, spin-½, dipolar cores pair chirally?** It is a BCS calculation with a ³He literature and a binary answer. The honest prior is against it.

**Computed since drafting — and the prior held.** On a 7200-direction grid the dipolar kernel gives pairing eigenvalues `p_z = +32.12` (attractive) and `p_x + ip_y = −15.52` (**repulsive**); Baranov's result reproduced. `p_z` has a **line** node — codimension 2, no Fermi point. The chiral state would need a T-odd coupling worth **148 % of the entire dipolar interaction**, starting from the wrong sign, and the clock is not that. Every attractive channel found is `m = 0`.

**So the route closes at step 3.5.** The strength cancels in the ratio, so the ordering is independent of `f`; only a differently *shaped* interaction could change it. That was the last gravitational question in the report with a method and a computable answer, and the answer is no.

*Rev. 249 attached. §§ 218–222 the sign proposals; §§ 228–230 the time treatments; § 229 the boost sector; § 227 the pairing route. Prior parallel work: OpenWave R14–R18, the M5–teleparallel chain C1–C18 (reviewed in § 221), and the Novarax dossier (Braun AI Wiktorowicz, Rafał Wiktorowicz, Marcin Rosa).*


===== COMMENT [30] JarekDuda 2026-09-11T21:15:12Z (top-level) =====

Working original Lagrangian (to flip Newton sign) in https://arxiv.org/pdf/2108.07896
Updated M5 report: https://zenodo.org/records/22714918


# The gravity sector, reorganised: the sign was never the problem

Report revision 294. Scripts `round267`–`round285` in the bundle; §§ are that report's. Companion to the electron-sector note. This one records a reorganisation rather than a result: **forty-three sections asked how to flip a sign that was already correct**, and identifying that changes what the open question is.

---

## TL;DR

1. **The Newton sign is correct, in every sector that could carry a gravity-analogue** — and has been since the sign rule was established. The sector where like sources repel is electromagnetism, where repulsion is right.
2. **The anticommutator proposal was tested in the wrong sector.** The original suggestion — commutator for rotations, anticommutator for boosts — was stated correctly and then computed for the *spatial* sector. Redone in the boost sector it does give spin 0+2 and attraction.
3. **And M5's connection is teleparallel by construction**, so the EM and GEM sectors are coupled by group structure rather than by any Lagrangian term — with the Lorentzian signature flipping the cross term relative to the Euclidean case the paper's eq. (9) states.
4. **The flip exists, couples to the conserved stress tensor, and gives 1/d⁵**, because the boost mode is a Goldstone and Goldstones couple derivatively.
5. **The one structure that changes a power is the axion term**, and it reaches 1/d between charges.
6. **A quadratic GEM–EM coupling does reach neutral matter** — the neutrality objection I raised was wrong — and it couples to the EM *fraction* of mass, which varies 9× across the periodic table against an Eötvös bound of 10⁻¹⁵.
7. **Cosmological structure was tested four ways and reaches the energy floor, not Newton.**

---

## 1. The sign, sector by sector

The rule, verified on a 48³ lattice (`V_complete = −⟨J, K⁻¹J⟩`, ratio −1.000 at five separations): **scalar and tensor exchange attract; vector exchange repels.**

| sector | spin | like sources | verdict |
|---|---|---|---|
| the dual photon | 1 | repel | **correct — it is electromagnetism** |
| the amplitude / split | 0 | **attract** | **correct** |
| the clock / twist | 0 | **attract** | **correct** |
| the symmetric boost bilinear | 2 | **attract** | **correct** |
| `R^gg = Γ̃ × Γ̃` — the paper's | 1 | repel | wrong for gravity |

> **The sign is correct everywhere it matters. The only wrong one is the paper's own `R^gg`, and that is a choice of bracket rather than a sign.**

| requirement | status |
|---|---|
| **the sign** | **correct** |
| the range | wrong |
| universality | wrong — 7.4 % measured across four objects |
| the strength | wrong — 4·10⁴² unless separately suppressed |

**Three of four fail and the sign is not one of them.** Seven proposals — the anticommutator, the imaginary axis, the imaginary coupling, the NUT charge, the symmetric bilinear, the flipped Hamiltonian term — each found a way to get a sign that was never in question. **And "the sign is correct" is weaker than it sounds: any scalar attracts. That is a property of scalar exchange, not of M5.**

---

## 2. The anticommutator, tested in the right sector

The proposal as originally stated was: *keep `[·,·]` for the rotation sector where like-charge repulsion is wanted, and use `{·,·}` for the boost/time sector where attraction is.* **That is right, and the test I ran computed the spatial sector instead.**

Redone where it belongs: SO(1,3)'s boost generators are **symmetric** where SO(4)'s are antisymmetric, and for two symmetric generators `{A,B}` is symmetric — **spin 0 + 2** — while `[A,B]` is antisymmetric — spin 1. The paper's `R^gg = Γ̃ × Γ̃` is the cross product, so it keeps spin 1 and discards spin 2.

**In the spatial sector the antisymmetric part is the physical one**, because its charge is the π₂ winding. **In the boost sector there is no such winding** — symmetric generators have no cross-product charge to be topological about. **So the antisymmetric choice there is an analogy with electromagnetism, not a derivation, and the symmetric bilinear is equally available and gives the right spin.**

---

## 3. What the flip actually buys

**Every piece is present.** M5 has a conserved stress tensor by Noether's theorem — already used for the von Laue condition — and the symmetric bilinear `S^gg` is spin-2 and attractive. The vertex `S·T` can be written.

**And the mode is massless, for the reason that dooms it.** The boosts are broken by the vacuum, so `Γ̃` is a Goldstone — and a Goldstone couples derivatively. With `S ~ {Γ̃, Γ̃} ~ (∂M)²`, that is two derivatives at the vertex:

| derivatives at the vertex | 0 | 1 | **2** |
|---|---|---|---|
| static potential | **1/d** | 1/d³ | **1/d⁵** |

> **An attractive spin-2 mode coupled to the stress tensor. Right spin, right source, right sign, wrong range by four powers.**

To reach 1/d the vertex needs *zero* derivatives — a field coupling to T directly, with two derivatives in its kinetic term. **That is the Einstein–Hilbert structure, and M5 can build it**: `inc M = 2G⁽¹⁾[M]` means M's incompatibility *is* the linearised Einstein tensor, so `∫M·inc M` is the linearised EH action, built from M alone, vanishing on the vacuum, needing no new field.

**And it has m = 1, p = 1, so 2mp = 2 < 3: linearly divergent energy on a winding texture.**

| action | potential | core energy |
|---|---|---|
| the quartic alone | linear, confining | finite |
| **+ M·inc M** | **1/r — Newton** | **divergent** |
| + (inc M)² — the Stelle R² analogue | linear + Yukawa | finite, **and untested** |

> **M5 can have quadratic gravity. What it cannot have is the Einstein–Hilbert term that supplies the 1/r — and that is a theorem about winding, not a choice of coefficient.**

---

## 4. Coupling by orthogonality

`Γ_μ = Oᵀ∂_μO` is a **Weitzenböck connection** — pure gauge, zero Lorentz curvature, torsion carrying everything. **M5's O-field is teleparallel by construction**, and the EM and GEM sectors are tied by the Maurer–Cartan equation rather than by any interaction term. That is the "coupling by orthogonality" exactly.

**And the signature flips which combination:**

| generator | boosts | spatial block of `[Γ_μ, Γ_ν]` |
|---|---|---|
| SO(4) — the paper's eq. (6) | antisymmetric | `R^ee + R^gg` |
| **SO(1,3) — eq. (39)** | **symmetric** | **`R^ee − R^gg`** |

The paper states eq. (9) for the Euclidean generator and then replaces it at eq. (39). **Figure 6's "tendency for opposite curvatures" is the Euclidean statement; with boosts they have the same sign.** Every sign proposal asked how to flip a coupling by hand; **this one is flipped by η.**

*Caveat:* this verifies the Maurer–Cartan commutator, not the Hamiltonian, which uses `F = [∂M, ∂M]` with shape factors. The flip is established for the constraint and conjectural for the energy.

---

## 5. The one structure that changes a power

A Goldstone's shift symmetry forces the coupling through ∂θ. **The exception is a coupling to a topological density**, because the shift then changes the action by `c × Q` with Q an integer — the axion structure. M5 has every ingredient, and

> `S_θ = κ∫θ ρ_top d⁴x` gives `□θ = −κρ_top`, hence a Coulomb tail, hence **U = −κ²Q₁Q₂/4πd: inverse-square, attractive, long-range.**

Coulomb agrees with QED to ~10⁻¹², so κ/e < 10⁻⁶ — **a bound sitting at 4·10³⁰ times gravity's strength.** The channel could be gravity-strength and entirely invisible.

**It is proportional to Q₁Q₂.** I concluded from this that it vanishes for neutral bodies and that neutrality was the obstruction. **That was wrong** — see the next section.

---

## 6. The GEM–EM coupling does reach neutral matter

Matter is built of protons and neutrons, which carry plenty of charge even when the net is zero. **And the paper's GEM–EM coupling is `R^ee·R^gg` — quadratic in the EM curvature — so it couples to EM *energy density*, which is positive-definite and does not cancel when the charges do.**

So the question becomes what fraction of mass is EM energy:

| nucleus | E_EM/M |
|---|---|
| helium | 4.87·10⁻⁴ |
| iron | 2.44·10⁻³ |
| uranium | **4.44·10⁻³** |

**A spread of 9.1 across the periodic table**, giving an Eötvös parameter of order one between elements, against MICROSCOPE's 10⁻¹⁵. **Excluded by fifteen orders.**

**Which is the same failure as the NUT charge in a new channel**: that one tracks baryon number and misses the binding energy; this one tracks EM energy. Two couplings, one exclusion, because both see a *subset*.

> **Gravity couples to the total stress tensor. Any coupling to a part of it is composition-dependent at the level of that part's variation — ~10⁻³ for every part in the periodic table. Any partial coupling is excluded by ~10¹² unless the part it tracks *is* the whole thing to a part in 10¹⁵.**

**The obstruction is not neutrality. It is that every channel sees a fraction.** Our own δ-coupling was measured at a 7.4 % spread across four objects — 7·10¹³ times the bound.

---

## 7. Cosmological structure, tested four ways

**Nearly constant twist frequency.** For a massive phase field the uniform mode sits at the gap, so ω = m — the de Broglie relation, not an extra hypothesis. *This is now qualified:* the clock turned out to be massless (see the electron note), so "nearly constant frequency" means ω = K/I with K topologically fixed, which is structure the report already had.

**Least action over an eon.** For a rotor, `S(N) = 2π²IN²/T` grows as N², so free extremisation gives **N = 0** — a global variational principle *suppresses* the clock rather than producing it. A theta term fixes K = −θI/T, but |θ| ≤ π and I/T ~ 10⁻³⁹, so K ≤ 9·10⁻³⁹ against the ½ the electron needs.

> **Over an eon, dynamics wants K = 0 and topology forces K = ½. The clock survives because it is topologically protected, not because it is energetically favoured.** Which generalises: continuous charges relax to zero under a global principle, quantised topological ones cannot, and **the surviving structure at late times is the topologically protected part.**

**Extremising over the eon's length too.** ∂S/∂T = K²/2I > 0, so free extremisation drives **T → 0**. What opposes the collapse is the Friedmann equation, with **T ∝ GM** — so the eon's duration is a gravitational quantity, M5 has no G, and its longest intrinsic time scale is 10⁻²² s against an observed 4·10¹⁷ s. **Forty orders.**

**Compact space and the energy floor.** A closed universe does bound the unbounded-below witness (its own scope note says "open-space or increasing-box") — but the bound is attained by the negative phase *filling the universe*, so the certified vacuum is metastable at best. And a periodic time boundary condition **forbids the two *dynamical* runaways and neither static one.**

> **The cyclic structure's real content is the exclusion of runaway solutions — which matters, because two of the four sightings of the energy-floor problem are dynamical.** It does not reach Newton.

**A black hole as an inward-boost hedgehog.** This is exact rather than analogical: `v = tanh χ(r)` gives Painlevé–Gullstrand, and the outward boost is its time reverse — a white hole. M5 would have three horizons, one per branch. **But the boost sector destabilises at 1.85 % of c** (converged across resolutions) against the 71–100 % a horizon needs, **so M5's collapse is a phase change rather than a horizon** — and the Schwarzschild profile p = ½ sits exactly on the convergence boundary.

---

## 8. What the literature already said

Volovik's ³He-A result: the induced Einstein–Hilbert term **is** produced and is **swamped by non-diffeomorphism-invariant terms** from the superfluid's own Lagrangian. And Barceló–Liberati–Visser: Sakharov assumes the spacetime has **no prior dynamics**, while a medium's prior dynamics is *what makes* the effective metric.

> **The medium's own equations of motion are both the source of the effective metric and the obstruction to its being Einsteinian. They cannot be dropped — they make the metric. They cannot be kept — they are not diffeomorphism invariant.**

**M5's quartic is a contaminating term in exactly that sense.** Our closures are instances of a known obstruction; what is added here is a convergence theorem making it quantitative, a quantisation argument independent of diffeomorphism invariance, and an explicit name for the covariant term. **That is a narrower contribution than the report has been implying.**

---

## 9. The open question, restated

Not *how do we flip the sign* — nothing needed flipping. The question is:

> **Why is the attractive sector gapped or derivatively coupled, and what would carry a source that a neutral body has in proportion to *all* of its energy?**

Everything in §§145–275 stops at the second half. **Setting the transverse splitting to zero would restore the exact symmetry and remove the sector the model was built to have — that is the trade, in the model's own variables.**

---

## 10. What is worth someone else's time

- **`(inc M)²`, the Stelle R² analogue.** Admissible, convergent, built from M alone via the incompatibility identity, and **completely untested** — including its ghost content, which is generic for four-derivative terms. It is the one new term the survey of admissible terms left standing.
- **The source-contracted propagator** for the boost sector. Every force law in the last forty sections rests on derivative counting at the vertex rather than on a computed propagator. This is the handoff's G-02 and it is the only calculation that could still overturn a negative result.
- **The energy floor.** Four independent sightings — the coupled solve's runaway, the boost instability at 1.85 % of c, the unbounded-below witness, and the hyperbolicity conditions. **Nobody has computed the decay rate of the certified vacuum into the negative phase**, which is a Coleman bounce and decides whether the theory has a usable ground state.

*Rev. 294 attached. §§254–256 the teleparallel structure and the flip; §267 the sign; §§245, 261 the admissible terms; §§269–270 the axion term and neutral matter; §§236, 249–252 the cosmological tests; §247 the literature. Parallel efforts: OpenWave (R14–R18), the M5–teleparallel chain (C1–C18), Novarax (Braun AI Wiktorowicz, Rafał Wiktorowicz, Marcin Rosa), the 11 September technical handoff, and Mikulski's twelve reports.*


===== COMMENT [33] JarekDuda 2026-09-12T16:24:04Z (top-level) =====

Updated report: https://zenodo.org/records/22727461

# The particle-construction ladder, rungs 1–3

*M5 report rev 390, §§292–366. Scripts `round344`–`round388` in the bundle; every claim is in the verification ledger with its status and the script that produced it.*

Three weeks ago the electron sector had three apparent crises. Two were misreadings, the third is solved by the author's own potential, and one thing the report spent seventy sections searching for was already in the foundational paper's abstract. This is what is left.

---

## The correction that matters most

**The three-lepton mechanism is in arXiv:2108.07896 and this report overlooked it.** From the abstract: *recognising intrinsic twist of uniaxial nematic allows hedgehog configurations with **one of 3 distinguishable axes**: having the same topological charge, but different energy/mass — getting similarity with 3 leptons.*

- **One** hedgehog with three axis choices, not a bound state of three.
- **A biaxial tensor has three eigenvectors and no fourth**, so the flavour count is derived, not assumed — which is what §§349, 355 and 357 each failed to find.
- And "LdGS" is **Landau–de Gennes + Skyrme**. §§325–326 proposed the first as a fix for the potential and §349 used the second to stabilise the loop. Both halves were in the framework's name.

**Rung 3 is therefore three runs of one boundary-value problem** — the winding assigned to each axis in turn — **scored by Koide (Q = 2/3 to 10⁻⁵, parameter-free).** That is the cheapest well-posed test remaining below the baryons, and it tests the flavour count and the mass spectrum at once.

---

## Rung 1 — the electron

**Settled**

| | |
|---|---|
| the clock | ω = K/I = 0.694, finite and nonzero |
| its inertia | I = 0.7204, box-converged, residual-verified at 1.6·10⁻⁴ |
| **the split core** | b(0) = 0.183 with b → 0 in the bulk — **both conditions, at different radii** |
| the boundary condition | b → 0 is *necessary*: 1 − a² = 2g_W b²/λ, verified to 0.999985 |
| ~~metastability~~ | **withdrawn (§369)** — the advertised minimum has an exact downhill direction |
| the rate | 80.874 MeV/c is the channeling paper's **calculated** resonance; the measured dip is at 81.1 MeV/c (§369) |
| g = 2's mechanism | half-disclinations: the director returns after π, so the charge winds **twice** |
| C-parity | photon and clock both C-odd — from the degree's integrand and the transverse plane's orientation |

**Open**

- **the l = 2 shell** — never converged. Blocks spin *and* the g-factor. The fix is specified: a Landau–de Gennes form polynomial in the trace invariants, hence smooth where eigenvalues collide.
- **the Lorentz force** — untested. The magnetic half is a **Magnus force** whose coefficient belongs to the medium and must equal the defect's q. **It can fail by a sign**, and no profile refinement repairs a sign.
- ~~the photon identification~~ — **done, and it fails.** For a director depending on a single u = k·x, ∂_μn̂ × ∂_νn̂ = k_μk_ν(n̂′ × n̂′) = **0 identically**, so the null waves carry no dual field strength. A wave with *transverse* structure does have F ≠ 0 with a null k — that is the replacement, and it is unattempted.
- the kinetic term — chosen without derivation from ≥ 4 admissible invariants.

---

## Rung 2 — neutrinos as vortex loops

**Derived**

- **neutrality** — a closed loop encloses no hedgehog. Automatic.
- **stability** — R\* = √(Q/2πT), the hedgehog's own Derrick balance applied to a loop.
- **no collapse** — a conservative theory has no viscosity, so the loop **breathes** rather than shrinking.
- **a clock for free** — ω = √(2πT/Q), where the electron's required a condensate.
- **a size** — which a point particle has no reason to have. BeEST (*Nature* **638**, 640) measures σ_ν,x ≥ 6.2 pm, and ≥ 34 nm by the energy route. **The loop's size is the particle's, not the environment's — so it is source-independent, and every other model on that paper's figure predicts source-dependence.**

**Open**

- **the mass** — ~10⁷ too heavy unless the disclination is nearly tensionless: **0.6 GeV/m against a QCD string's 10¹⁵**.
- **PMNS** — not calculable. Flavour is a label, not a wavefunction, and the vertex that would define it is at rung 4. What the picture *does* give: a vertex 10⁶ times smaller than the loop cannot resolve the sizes, hence **large lepton and small quark mixing** — and the maximal limit is excluded by sin²θ₁₃ = 0.022 against 1/3.

**One number worth flagging.** Equating the breathing rate to Δm²/2m gives **R\* = 7.9 μm**, inside the 1–10 μm band atomic-localisation models predict for an EC source. Three estimates in the same decade — encouraging, not evidence.

---

## Two checks that pass

**Four zeros = two lines.** §185's index theorem puts **four** zeros on the l = 2 shell; the split vortex has **two** strands. A line through a closed surface crosses it *twice*, so four zeros are two lines. Three lines would have given six. The topology and the decay count reach the same two independently.

**Two neutrinos.** Every charged-lepton decay releases **exactly two, never identical** — one ν and one ν̄, of different flavours. That matches a **split** vortex's two inequivalent strands on three counts (the number, the inequivalence, the opposite orientation) and a degenerate vortex on none. **Beta decay's single neutrino is the control**, and it passes because baryons are knots.

*Stated plainly:* two neutrinos follow from lepton-number conservation and are explained by every model of the weak interaction. This selects the split structure **within this class of models**; it is not independent support for the class.

**So the split vacuum is now required by four independent things:** the clock's existence, the two-neutrino decay count, the four-zero line count, and the three-lepton spectrum.

---

## The parameters have never been determined

Six constraints bear on g and δ, and they are mutually inconsistent as stated:

| | from | gives |
|---|---|---|
| C1 | the clock exists | δ ≠ 0 |
| C2 | three distinct leptons | δ away from both 0 and 1 |
| C3 | Hughes–Drever, 10⁻²⁷ | **δ → 1**, if matter sees the anisotropy |
| C4 | a QED comparison | **δ ~ 10⁻¹⁰**, with gδ ~ 1 |
| C5 | the light cone, c_eff = √(δ²/2 + 1/2) | **δ = 1** |
| C6 | det(M) = const | gδ² fixed — a relation, not a value |

The report's (8, 0.3) and the QED comparison's (10¹⁰, 10⁻¹⁰) **differ by 7·10⁹ in the determinant**. These are not perturbations of one another. And attempts to close the gap with a power law m ~ s^p fail in both directions: at δ = 0.3 the three splittings span only 3.3 against the 3477 needed, forcing p = 9.58; at δ = 10⁻¹⁰ two of the three splittings become degenerate to one part in 10¹⁰ while no two leptons are.

**The three-axis solve returns two mass ratios with no external input and no fitting. It is not one more test among many — it is the thing that could fix the parameters from inside the model.**

*(Caveat: the three-axis energies also depend on the potential's couplings, on g_W, on κ₆ and on the kinetic normalisation. "Two ratios, two unknowns" is optimistic.)*

---

## Also worth recording

- **det(M) = const as incompressibility.** It needs a shifted spectrum, and it **removes the dilaton rather than gapping it** — so it introduces no dimensionful coupling and bounds the spectral runaway *structurally*, where a potential needs a coupling choice. The cost: the 1/d gravity channel closes with the dilaton.
- **SU(3)'s adjoint branches to SO(3) as 3 ⊕ 5** — rotations plus a nematic Q-tensor, which is "two coupled field rotations," and M5's spatial sector supplies both. Not established: that the 8 is *gauged*. The decisive test is a commutator calculation, and it does not need the baryons.
- **α is an input** here, in Faber's topological-fermion model and in the superinsulator programme. No member of the family derives it. What Faber's model *does* produce, classically, is the **running**.
- **Quantisation is absent everywhere**, and it is the residual gap in the photon sector, in the spin sector and in every rate.

---

## A method note

A solver's status flag is not a solution. Three rounds were spent on runs that returned status 0 with a bulk residual of 2.66 before a pointwise check settled it. **Every BVP result here now carries one — normalised by the equation's largest term, not by the field.** The report's baseline solutions pass at 1.6·10⁻⁴.

---

## Added after OpenWave R19

**The flipped bracket has been built, measured and independently audited — and it fails as a Newton candidate.** (Report §367.)

*Three findings correct this report:*

- **The object is real and new.** F^(Γ) = F + 2(P^tt)^T built two ways agreeing to 10⁻¹⁴, Lorentz covariant to 10⁻¹³, lift-independent, audited 4 confirmed / 2 qualified / 0 refuted. And the ranks go **8 (F-built) → 9 (pure anticommutator) → 12 (traceless literal)** — so "already in the R1 basis" holds only for the *spatial* sector, and this report's earlier dismissal of the object was wrong.
- **The cross term vanishes exactly.** ⟨F − X_a, X_s⟩_η = 0 by the parity of the derivative pair, so I(F^(Γ)) = I(F − X_a) + I(X_s): **the EM and GEM sectors decouple exactly.** A theorem, not a fit, and stronger than anything this report established.
- **And the dressed like pair repels.** E_int = **+35 555** at d = 10 against the certified +3 869 — ten times, monotone, fitting **1/d with a positive coefficient** (R² 0.91, 0.993 extended), not 1/d⁵. The attribution is measured: the anticommutator overlap of two **non-decaying** dressings reproduces E_int within 3 %. Outcome: **CANDIDATE_REFUTED (repulsive).**

**Why the spin argument did not save it.** This report argued {Γ̃, Γ̃} is symmetric, hence spin 0 + 2, hence attractive — that was about the long-range *exchange*, which §277 put at 1/d⁷ and which is negligible at these separations. **What dominates is the overlap of dressings that do not decay, and the overlap repels.** v10's Newton column moves from partial to failed.

**Two readings that do *not* conflict.** R19 reports "finite nonzero frequency has no content on this field under the replacement" — that is the **boost** clock, which is unbounded below in ω. The **rotation** clock, which is the one this report claims, R19 finds *identical* to the certified I₁ under the replacement. **ω = K/I = 0.694 stands.**

**And the floor is confirmed by measurement.** At g = 8 the certified action's own dressed rows run away first, the time-row piece diving from −172 to −12588 while the rest stays of order 100. That is §§99 and 238 and Novarax's witness, measured on saved end fields — and it contradicts this report's §321 claim that the boost sector was already bounded.

---

## Added after an external review — five withdrawals

**Every checkable claim in the review was verified here independently, and all six confirmed.** (Report §369, script `round391_verify_review.py`.)

**1. The advertised Landau–de Gennes minimum is unstable — withdrawn.** Splitting the repeated pair at fixed trace, q(ε) = q\* + (ε/√2)(0,0,1,−1), the printed potential gives *exactly*

> V(q(ε)) − V(q\*) = **−2.515 ε² + 0.375 ε⁴**,  so  d²V/dε² = **−5.03**

Not a rotational zero mode, not the removed trace direction, not a near-zero numerical eigenvalue. **And the reason is structural:** stationary eigenvalues satisfy one cubic, and splitting two copies of the *middle* root has curvature D(r₂−r₁)(r₂−r₃) < 0. The repeated value 1.9 is exactly that middle root. No mesh or tolerance fixes it.

**The review supplies a repair**, which I confirm: V_spec(Q) = γ·tr[P(Q)²] with P(x) = (x+6.4)(x−2.6)(x−1.9) — nonnegative, vanishing at the target, **no sorting seam**, spectral Hessian 2γP′(qᵢ)²δᵢⱼ, and the formerly unstable direction now at **+67.5122 γ**.

**2. "Metastability green" — withdrawn.** A bounded potential establishes neither the kinetic sign, the principal characteristics, the constraint rank, nor the stability of a localised rotating solution. **A derivative-free potential does not repair a defective highest-derivative operator.** The single green splits into four checks.

**3. The dilaton — corrected.** Q(N + σI) = Q(N) is an *additive shift* symmetry; Q(sN) = sQ(N) is homogeneity, not scale invariance. **And F = [∂N, ∂N] is unchanged under N → N + σ(x)I, so the trace has no kinetic operator at all.** It is not an available propagating scalar merely because the potential leaves it free.

**4. det = const does not bound — refuted.** N_t = diag(−8t, 1, 0.3/√t, 0.3/√t) has det = −0.72 **exactly at every t** while ‖N‖ → ∞. The determinant surface is noncompact.

**5. "Every charged-lepton decay releases two neutrinos" — false.** **τ⁻ → π⁻π⁰ν_τ** is an observed, extensively measured mode with *one* neutrino. The claim holds for the purely *leptonic* channels only, and the universal two-strand test is defeated.

**And two experimental readings are corrected.** **80.874 MeV/c is the channeling paper's *calculated* resonance, not its measured centroid** — the dip is at 81.1 MeV/c, 0.28 % above, within the ±0.3 % momentum calibration. So the claimed seven-parts-in-10⁵ agreement is not an experimental confirmation at that precision. And **BeEST constrains a decay product's wavepacket, not an intrinsic neutrino radius** — σ_ν,x = R_loop is an additional assumption, and the source-independence prediction does not follow from it.

**A stronger photon test, accepted:** Ω ∧ Ω = 0 for any pure pullback, i.e. **E·B = 0**. Two crossed Maxwell waves give E·B = 1, so no single director pullback represents that field.

**The pattern is worth stating.** Every one of these was checkable in minutes — a two-line expansion, a one-parameter family, a matrix identity, a PDG listing. **The failures cluster where a numerical result was trusted over a structural one:** §326 reported positive Hessian eigenvalues from an optimiser while the cubic root argument guaranteed a negative direction. That is a verified computation of the wrong quantity.


**And the repair generalises into a theorem (§370).** A potential built from trace invariants has stationarity Σc_k·k·q_i^(k−1) = λ, so **every eigenvalue is a root of one polynomial of degree (max k − 1)** — a quartic selects at most **three** distinct eigenvalues. M5's vacuum (g, 1, δ, 0) has **four**, so the quartic class §§323–337 worked in is excluded outright, and boundedness forces degree **8**.

The construction is then unique: with P(x) = ∏(x − q_i) quartic, **V = γ·tr[P(Q)²]** satisfies all six of §327's requirements — curvatures 371867 / 48.0 / 5.23 / 11.5, all strictly positive, V = 0 exactly at the target — with **one free parameter γ** and the vacuum as an input rather than a fit.

**The cost is heavy and worth stating.** There is no shape left to tune: if V is fixed by the vacuum and the kinetic term by its original choice, the lepton spectrum is determined with **no adjustable quantity**. §364 already showed the three axis splittings span 3.3 while the masses span 3477, and nothing remains to absorb that gap. **Either the programme's strongest claim or its fastest route to refutation.**

**What survives is narrower and better specified.** The split-core correction stands and the review endorses it. The photon retraction stands and is strengthened. The three-axis direction stands as a *proposal*, not a derived count. **And the replacement potential removes the one defect blocking every solve this report has been deferring — which makes it the first thing to test.**

---

*Corrections and adversarial readings for every item are in the report's per-section 'Adversary' paragraphs. Nine results were withdrawn in this arc — the α discrepancy, the vacuum invalidation, the clock/charge incompatibility, the photon identification, the LdG minimum, metastability's green, the dilaton reading, the determinant bound and the two-neutrino universality. The first four left no dependent claim standing; the last five came from an external review, and every one was checkable in minutes.*


===== COMMENT [34] JarekDuda 2026-09-13T07:53:42Z (top-level) =====

# M5 / LdGS — adversarial audit and check scripts

Companion material for *Framework for liquid crystal based particle models*
([arXiv:2108.07896](https://arxiv.org/abs/2108.07896)).

**M5 = LdGS**: a real symmetric 4×4 field `M(x)` with Lorentz signature `η = diag(−1,1,1,1)`,
a Landau–de Gennes potential on its eigenvalues, a Skyrme-type quartic `⟨F,F⟩` in its
gradients, with `F = [∂_μ M, ∂_ν M]`.

This folder holds a 1019-page critical audit of that framework and the ~520 scripts that
produce every number in it. **The audit is adversarial by design**: each of its 496 sections
ends with a paragraph arguing against its own conclusion, and several sections withdraw
earlier ones.

---

## Quick start

```bash
pip install numpy scipy sympy matplotlib reportlab
python round<NNN>_<name>.py      # any single check, prints its own results and tolerances
python build_pdf.py              # rebuilds the full report
```

Every script is standalone and runs in seconds to a few minutes. No data files, no network.
`README.md` (in the bundle) maps each `round*.py` to the section it supports.

## Reproducing a claim

Numbers in the report are tagged by evidence type — `[T]` computed here, `[L]` from the
literature, `[N]` a negative result, `[R]` a reframing — and each carries a script name.
To check one, run that script. For example:

| claim | script |
|---|---|
| the attractive sign is free from the Lorentzian signature | `round504_sign_free.py` |
| a quartic kinetic term gives a **linear**, not 1/r, potential | `round501_one_fact.py` |
| the trace channel is exactly null, not negative | `round511_dewitt_audit.py` |
| the trace shift is M5's only local symmetry | `round512_wtdiff_route.py` |
| GW polarisation selects between M5's two gravity sectors | `round522_polarisation.py` |

## Where the audit ends up

**Established and unique to this framework:** charge quantisation as a topological winding,
with Gauss's law counting it (the paper's eqs. 1–4).

**Follows from the flat background:** an exactly conserved local energy density and global
total — neither of which general relativity has — and a dimensionless coupling, so the
dimensional obstruction behind general relativity's non-renormalizability does not arise.

**Not achieved:** any gravitational measurement. The attractive *sign* is free from the
signature; the *power* is wrong — a dimensionless field admits only a quartic kinetic term,
whose propagator is generated by the background gradient alone, giving a linear potential.
Thirteen of sixteen cells in the front-page comparison are untested.

## Open calculations

Ordered, with the first two running on machinery already here:

1. **TT projection** — do M5's ten symmetric directions contain a transverse-traceless pair
   coupling to `J = T − ¼ηT`? Decides which sector survives GW170814.
   Start from `round511_dewitt_audit.py`.
2. **v9 + det e = 1** — is the residual gauge group Weyl + transverse diffeomorphism?
   Rank six with four nulls is the target. Start from `round512_wtdiff_route.py`.
3. **Sector dispersions** — compare ω² and k² coefficients across the EM, quantum-phase and
   GEM channels on the `(g, 1, δ, δ)` vacuum. `round509_lorentz.py` sets it up.
4. **Gradient condensate** — does the Coleman–Weinberg vacuum give `⟨|∂M|²⟩ ≠ 0`?
   `round501_one_fact.py` has the static solve.
5. **`∂ln M_A/∂s_∞`** across compositions and binding energies — the universality derivative.
6. **Vortex core radius** — decides whether the area-law count of `round492_merw_entropy.py`
   follows.

## Conventions that bite

- `η = diag(−1,1,1,1)`; contractions use `‖X‖²_ξ = tr(XξXᵀξ)`, so **time-row components enter
  with a minus sign**. This is the origin of the attractive sign, and of several sign errors
  in earlier revisions.
- The **frame** (the rotation matrix `O`) carries the entire kinetic term; the **spectrum**
  (the eigenvalues `D`) enters only through the potential, which is ultralocal.
- `[K_i, K_j] = −J_k` for boosts, against `[J_i, J_j] = +J_k` — the sign difference matters
  and is not the same operation as the symmetric bilinear.
- Scripts print tolerances; agreements are quoted at the precision achieved, not rounded.

## Status

AI-assisted analysis and internal review, **not peer review**. The framework and the physical
proposals are Jarek Duda's; analysis and drafting by Claude (Anthropic); independent
algebraic and numerical review by Astra, several of whose corrections withdraw conclusions in
earlier revisions. Observational credit belongs to the cited experimental collaborations.

Corrections and counterexamples are the most useful contribution — the report's own record is
that roughly one conclusion in six has needed withdrawing.


===== COMMENT [37] JarekDuda 2026-09-13T16:30:32Z (top-level) =====

@mjmikulski @xrodz

Both checked against the audit stack (rev 626, §§578–580; scripts `round606`–`round608` in the bundle). Three convergences, three corrections to my record, and one result I think is the most serious thing anyone has produced against this framework.

## 1. 015 settles a question three of my sections reached separately

"Outside the core the model is the Faddeev–Skyrme quartic without a sigma term" is the same statement I arrived at three ways this week, and none of mine was a measurement:

- §563: the two independent quartic traces satisfy `tr([A_mu,A_nu][A^mu,A^nu]) = 2(T_2 - T_1)`, so `<F,F>` is the Skyrme invariant — verified to 1e-12, but only as an algebraic identity.
- §566: eliminating the heavy connection of an SO(3) gauge–Higgs parent gives `F_munu[B_*] = [d_mu P, d_nu P]/f^2` at the isotropic point, and **the same reduction supplies the `f^2` quadratic term**.
- §564: in a chiral parent the missing quadratic is `f_pi^2`.

015 measures it, gives the tilt's exact cost `(32 pi/3) Delta^4 INT theta'^2` and its inertia, and names the absent term. **The identification is no longer an analogy.** If you have the halo's quartic coefficient in the same normalisation as the report's `lambda`, that is the one number that would let the parent comparison be quantitative rather than structural.

## 2. The trace, confirmed — and I got there from a different direction

§560 found that §450's shift `N -> N + sigma(x) 1` makes the trace pure gauge, so the physical content is the eigenvalue *differences*: `lambda_0 - lambda_1 = -(g+1)` and `1 - delta`, with the third pair zero by the degeneracy. In the traceless gauge `g = 1 + 2 delta = 1.63`, so the report's `g = 8` carries 6.4 units of pure gauge.

Your rank counting (9 generic, 8 frozen, **0 at the vacuum**) is the part I did not have, and **rank 0 at the vacuum** is §477's missing propagator confirmed independently — as is R14-0's "no quadratic fluctuation on the flat vacuum". Two stacks, same result.

One caveat I owe you: §564 notes the shift is exact only in the chiral limit. A potential breaks it, so `g` is *approximately* gauge, which is why §343's clock can depend on it at all.

## 3. 016 is the most serious result against the framework in my record

The null family `N = C - a(r) l l^T eta` with `l = (1, x-hat)`: `F = 0` identically, charge intact, `E <= (8 pi/3) Delta^2 R^3`, infimum zero at fixed charge.

I had just run the Derrick scaling (§579) and reported that M5 **passes** the shrinking-core test — the quartic kinetic term makes the inertia `I ~ R` grow with the radius, so `E_rot ~ K^2/R` resists collapse, unlike the quadratic-kinetic models where `I ~ R^-1` lets it run away. That conclusion stands for the radial direction and is worthless in general, because 016 exploits a direction I did not test. My own adversary paragraph said "only the spherically symmetric radial mode has been checked", which is not a defence.

**What makes it worse than a scaling escape is that every term built from `F` or from the spectrum is exactly blind to it.** No potential engineering reaches a direction with `F = 0`. Your `K_u = eta^{mu nu} <A_mu, A_nu>` with `A_mu = (1 - P_0) d_mu N P_0` is therefore not one repair among several — it is the only kind that can see the escape, because it sees the eigenframe rather than `F`.

Two questions on it. Does `K_u` disturb §533's sign fix, which sets the overall sign of `<F,F>` by requiring a positive velocity coefficient? And does the local threshold `c ~ 0.013` sit above or below the coupling the relaxed core actually wants?

## 4. omega = 0.694, withdrawn as a measurement

You are right and this matters more than the rest. §§343, 511, 543 and 546 all use it — as a free rotor, as the proton's clock via `omega = L/I`, and in a domain-wall argument. If R13-W finds no fixed-J minimiser on `L_cert` and the inertia is box-extensive, and R18-2 returns `ROTOR_NOT_MINIMUM`, then 0.694 is a property of the report's couplings and not of this field, and every clock conclusion inherits that. I have flagged it in the ledger; the sections stand as conditional statements about a number nobody has measured.

That also weakens §543's `omega_p/omega_e = m_p/m_e`, which was already a consistency condition rather than a derivation.

## 5. pi_2 = 0 on the split vacuum

Taken, and it bears on three sections. §535 (reading the Nature Physics heliknoton paper) found that **knot type is not protected** and the conserved quantity is the Hopf index; §548 concluded the baryon picture is "one topological object with two charge dressings" rather than two knots. Your point adds that on `(1, delta, 0)` — all four eigenvalues distinct — **nothing protects any winding at all**, and the paper's protection lives on the uniaxial `(1, delta, delta)`.

So the pre-registered `AXIS_ESCAPES` outcome is not a pathology of the run; it is what the topology predicts. Worth noting that §363.6's "biaxiality is essential" and the uniaxial protection requirement are in direct opposition, which your pre-registration already says.

## 6. Two smaller things

**Koide.** I agree it is a read and not a gate, and the reason is quantitative: `Q = 2/3` is one equation on three masses, so its solutions are a two-parameter family. Your 1 : 4.5 : 162 gives 0.666803 and the measured 1 : 207 : 3477 gives 0.666661 — both on the surface. A triple landing on 2/3 is weak evidence; the ratios themselves are the content.

**The string.** `E_int = -18.66 + 0.935 d` on the degenerate vacuum, against no quadratic fluctuation on the flat one, is exactly §539's resolution of an internal contradiction I had been carrying: §414's confining pair energy and the requirement that like charges repel at nuclear separations sit on opposite sides of a crossover set by the background. Your two runs confirm it from the lattice. R19-1's repulsion under the boost bilinear at every budget is new to me and I have recorded it.

**On the potential:** `V_spec` versus `V4` — §579 has just established something that makes the comparison sharper than "which is prettier". A pure quartic with no potential does not have a soliton at all: `E ~ 1/R` for both the static quartic and the rotational sector, so the core **expands** without bound. `E = a/R + b R^3` then gives `R_* = (a/3b)^(1/4)`, so the potential's strength *sets* `r_0`, and `r_0^4 ~ (quartic)/(potential)` is one relation between two coefficients the report treats as independent. So R20-2's `gamma` matching is measuring something with a physical consequence, not just a normalisation.

Everything above is reproducible from `round606`–`round608` in the bundle; the sections are §§578–580 of rev 626.


===== COMMENT [38] JarekDuda 2026-09-13T18:52:29Z (top-level) =====

Updated M5 hunt report:  https://zenodo.org/records/22736302

<html><body>
<!--StartFragment--><html><head></head><body><h1>Four results from today's audit that bear directly on the certified-action runs</h1>
<p>@mjmikulski @xrodz @vantasnerdan</p>
<p>Not a summary — these are the four that change what a run will find, with the scripts in the bundle (<code>round619</code>, <code>round630</code>, <code>round637</code>, <code>round639</code>).</p>
<h2>1. R13-W and R18-2 have a one-line explanation, and it is the norm</h2>
<p>The rotational inertia's sign <strong>is</strong> the norm's sign. For a rotating core the velocity sector is <code>C_0i = [d_0 M, d_i M]</code> with <code>d_0 M = omega [J, M]</code>, so <code>I ~ INT &lt;C_0i, C_0i&gt;</code>. Sampling 3000 random spatial gradients against the clock direction on <code>(g,1,delta,delta)</code>:</p>

norm | positive | min
-- | -- | --
tr(C eta C^T eta) — the certified one | 2262 / 3000 | −7.40
tr(C C^T) — Frobenius | 3000 / 3000 | +0.065


<p><strong>Every negative pair is one boost and one rotation sharing an index</strong> — geometrically, a boost whose direction lies in the rotation's plane. Disjoint pairs vanish exactly. So if you want to locate the instability numerically, that is the deformation to build, and a locking term only has to act on four directions rather than on the whole orbit.</p>
<h2>4. The potential is exactly blind to it, and this settles the V4 question differently than I expected</h2>
<p>The construction above stays on the vacuum's GL(4) orbit, so <strong>all four power traces are unchanged</strong> — verified: <code>tr(S^p) − tr(Y^p) = 0</code> for p = 1…4 via the similarity <code>g^-1 m = e^-1 (eta^-1 material) e</code> (<code>round619</code>).</p>
<p>So <code>V4 = w sum_p (tr((M eta)^p) − C_p)^2</code> is <strong>identically zero</strong> along a direction where the quartic is negative. This is stronger than the null-tilt case, where I found <code>V4</code> does penalise the tilt (4.2e4 at a = 0.1, 3.7e6 at a = 1) — here there is no cost to outgrow.</p>
<p>Which means the <code>V_spec</code> versus <code>V4</code> comparison in R20-2 measures something real but not this: <strong>no eigenvalue potential can stabilise frame directions</strong>, because the orbit preserves the spectrum by construction. Whatever R20-2 finds about the two potentials' spectral Hessians, both are blind here.</p>
<hr>
<p><strong>On my side, corrected today:</strong> my §576 claimed the Hamiltonian is bounded below, using the Frobenius norm on each sector when the action's is eta-weighted — withdrawn. The closed form <code>−2(δ−1)²(g−1)²</code> came from an outside note and is verified. I also got the counterexample wrong on the first attempt (a grid returning +2.0 throughout) before a time-diagonal-against-time-space pair gave the negative value; both rounds are in the bundle rather than the second replacing the first.</p>
<p><strong>One request.</strong> Item 5 of the Complete Picture handoff asks for an interior-to-exterior match at a movable sphere — the check that catches double-counted self-energy and separates the boundary chemical frequency from the asymptotic one. As far as I can tell nothing in either stack has done that, and every energy quoted on my side is an interior number. If R20 can carry a matching radius through its box ladder, that would be worth more than another potential comparison.</p>
<p>Everything above reproduces from <code>round619</code>, <code>round630</code>, <code>round637</code> and <code>round639</code> in the bundle; sections §§591, 602, 609 and 611 of rev 663.</p></body></html><!--EndFragment-->
</body>
</html>
# Guarded colour coordinate and target-directed SOURCE

This helper consumes A's one shared chart and cached neutral state. It does not prepare a second state, add a free rotor/oscillator, assert physical SO3 equivalence, or assert full-Haar/full-law closure. A owns Hamiltonian/state integration; B owns the physical Noether trajectory and its comparison to this field-only framed rotation.

## Actual VECTOR-derived connection, not a Dirac spin input

For each actual four-matrix Phi, define all fifteen gauge tangents D_a=i[T_a,Phi], in the Cartesian basis normalized by Tr(B_a B_b)/4=delta_ab. The actual z-vector flow is Y_1=-Phi_2, Y_2=Phi_1, Y_0=Y_3=0. With Cartesian mass I/chi, solve

`lambda=(D^T D)^(-1) D^T Y`.

`vector_derived_generators` returns S=sum_a lambda_a T_a and the actual normal residual `Y-D lambda`, not a fitted Clifford pole. It guards both the existing frame chart and invertibility of the full fifteen-dimensional inertia. Off-core normal flow is NOT added to the field-only orbit or declared absent. The native control exposes a nonzero residual 0.02120217220565854.

For an arbitrary coordinate path with D,D',D'' and Y,Y',Y'', let M=D^T D, b=D^T Y. Exact implicit jets are

`lambda'=M^-1(b'-M'lambda)`,

`lambda''=M^-1(b''-M''lambda-2M'lambda')`.

`guarded_generator_jet` implements these expressions, including actual Phi dependence and second-direction terms. Mixed coordinate jets follow polarization or a combined path. No finite-difference derivative is substituted into this API.

Adjoint covariance of D and Y gives `S(g Phi g†)=g S(Phi)g†`. Along its own gauge orbit, S'=i[S,S]=0. For the fifteen Gauss coordinate vector fields X_a, covariance gives the adjoint structure-constant derivative of lambda; contracting the equal Lie indices yields `sum_a X_a(lambda_a)=0`. This is divergence along the full Gauss orbit, not a claim that every Cartesian divergence or every physical Noether term vanishes. The native off-core numerical contraction is zero, and independent adjoint covariance defect is 3.225400748341146e-15.

## Guarded all-order link partner

The legal profile is C-infinity radial, alpha=1 inside B's radius 671088640000 and alpha=0 outside 1342177280000. `physical_profile` evaluates a flat-exponential collar, keeping the plateau/exterior exact. The clock and physical SO3/lattice comparison remain B's responsibility. The native control uses an explicitly different one-site profile on N3, with Ref outside; it is NOT the legal-radius profile aliased onto N3.

For fixed initial S(Phi),

`g_x(t)=exp(i t alpha_x S_x)`,

`Phi_x(t)=g_x Phi_x g_x†`,

`U_xy(t)=g_x U_xy g_y†`.

The exact tangent is `i(lambda_x U-U lambda_y)`. Its second jet is `-lambda_x² U+2lambda_x U lambda_y-U lambda_y²`. Both are returned by `field_only_flow`. The coordinate is a FIELD-ONLY GPhi+GU orbit. No matter Gauss generator or reduced Root generator is multiplied in again. A must implement the Gauss quotient/relative matter representation rather than quantizing this gauge direction as an independent free physical oscillator.

For independently evolving normal/spatial fields, S may change. `time_ordered_gauge` solves `g'=i lambda(t)g`: its mathematical definition is the time-ordered all-orders exponential. Its DOP853 evaluation reports tolerance/unitarity/error status and is not exact arithmetic or a replacement by `exp(i integral lambda)`.

`exponential_jet` uses Frechet and a triangular block exponential, with genuine noncommuting q2. `guarded_chart_jet` combines arbitrary Phi/link/alpha/angle base jets, including S' and S'', so neither stochastic Phi endpoint nor cross derivatives are frozen.

`guarded_frame_jet` and `full_phi_frame_jets` supply the actual least-intertwiner/polar frame derivatives for arbitrary ALL60 Cartesian Phi jets, not just a prescribed rotation. Positive Gram square-root jets solve Sylvester equations. Least-eigenvector q1/q2 use the reduced inverse of `L†L-e0`, retaining the eigenvalue reaction and normalized-vector second component. The polar factor and SU4 determinant correction are differentiated with their full mixed terms. Residual local Z4 chart choices cancel in the ORIGINAL peredge SOURCE. The executed base/angle/harmonic mixed SOURCE q1/q2 includes a NONZERO stochastic Ref frame derivative even though its rotation alpha is zero.

## Canonical coordinate LAW pullback and kinetic cross terms

`law_pullback` evaluates, before compression, the original quartic `Lambda kappa sum||G||²/2`, alignment `J W/8`, colour magnetic `K4 P`, original-heavy `-Lambda h_H(Phi,U)`, and improved-light `h_L(Phi,U,A)`. Here `P=sum||U_square-I||²/2`, so `K4 P` is exactly the specified `K4/2 sum||U_square-I||²`. No added selector species is introduced. A retains the original EM, Coulomb, harmonic, auxiliary and all-offset terms unchanged. `onebody_coordinate_jets` differentiates actual ordered one/two-link endpoint-average couplings and onsite Phi, including all q2 mixed terms before CAR compression.

For represented coordinate tangents F_a=dPhi/dq_a and U_a=dU/dq_a, the induced MASS metric is

`M_ab=Cartesian(F_a,F_b)/chi + Lie(-i U_a U†,-i U_b U†)/(.6chi)`.

The factor .6chi is the colour COMETRIC from `.3chi E4²`, not .3chi and not a half-electric law. `pullback_metric` returns M, its inverse, density sqrt(det M), and omitted tangent dimension. Dependent coordinate directions are rejected instead of pseudoinverted into false physical modes. `metric_first_jet` supplies M', inverse'=-inverse M' inverse and `(log density)'=Tr(inverse M')/2`. `kinetic_action` implements the induced Laplace-Beltrami operator with all supplied mixed/drift terms.

This induced restricted metric is not the complete ambient Haar/Gauss physical Hamiltonian. A physical quotient, nonuniform normal partners, transverse second derivatives, omitted CAR modes, selfadjoint domain and leakage estimates remain owner obligations. The one-mode native control honestly represents one of 2835 tangent directions and reports 2834 omitted; it does not call a flat native chart exact full Haar.

## Authoritative SOURCE and tree convention

PRIMARY `target_source_jets` binds to the immutable `noncompact_screened_gauss_dressing.screened_marks` producer, not to a function label. Its four primitive matrices are

`[I, Gamma2 Gamma3, Gamma3 Gamma1, Gamma1 Gamma2]`.

The fixed raw complex quartet is `[1,.3i,-.2+.4i,.6]`, multiplied by these matrices and divided by two, with all four Ref columns. No conjugation, beta flip, whitening, fitted phase or column normalization is used. Both endpoint frames and every interior edge frame are present. For each oriented edge, multiply the ORIGINAL whole-group character ratio `centre_scalar(F_x† U_x→y F_y,18)`; multiply these scalars along the path. `centre_word_jet` implements the exact quotient first/second chain rules including denominator derivatives, not firstjet squared.

A's historical `[I,Gamma0 Gamma1,Gamma0 Gamma2,Gamma0 Gamma3]` single-wholeword helper is preserved ONLY in `endpoint_source_jets` as an explicitly separate comparison contract. It is not the original SOURCE and its old norm is not transferred. Transport independently found all four original basis errors zero and unchanged raw beta mark equals `||beta||/2` times the descriptor point matrix to 1.7943e-15; helper/producer defect is 1.04775e-16 in that separate point control. See `local://issue222-authoritative-primitive-source-point-binding.json`.

`target_tree` validates the actual graph endpoints and multiplies the signed Birth-to-Ref path in fixed axis order. Backward steps use U_target†. Thus its product telescopes to `g_Birth W g_Ref†`. No unwrapped harmonic phase is silently coupled to a modulo-N colour shortcut. `uniform_target_tree` is a separately labelled positive periodic-z comparison: its exponent is `+(Ref-Born)%N`, not the old opposite historical word. The canonical signed tree uses `source_paths`, not this modulo control. Harmonic phase remains `exp[-i theta·(Born-Ref)/N]` with its own unwrapped displacement.

Ref remains fixed spatially. Alpha_Ref=0 makes its rotation-angle jets zero, but NOT its stochastic/base-coordinate Phi jets: `target_source_jets` accepts an independent full Jet2 for both endpoint frames. Its original value/q2 is tested against the actual producer. The source and body minus signs are not fitted to produce a minus: full-tree minus composed with ordered-body minus is retained as a PLUS adverse control.

## Executed evidence

Scientific CLI uses the project's Python and A's cached NPZ `local://issue222-prepared-gauss-raw-canonical-state.npz`, SHA256 `7feaeb3f8b2703db89bb9c0665ea6f64c3959be2320ccb6546f65ca69d25d6bc`. Constructor verifies operator signature `ebb7be04cda391f56f02e25eacd30bc6031a9bcff89713151e54eb9ec39c4d9f`; no re-preparation or label join. Exact argv/cwd/runtime/stdout/stderr/source/input hashes and natural exit are captured in `local://issue222-guarded-colour-complete-wrapper.json`. Python source SHA256 is `4a02934debe7d49c762dcbf5d05eb3b9b113285f06adc938119b0946c86438a0`; exact executed bytes are archived in `local://issue222-guarded-colour-source-4a02934debe7d49c762dcbf5d05eb3b9b113285f06adc938119b0946c86438a0.py`.

Observed natural exit0, 30.63155722990632 seconds, empty stderr. Whole-tree 2pi endpoint defect0; omitting link partners defect4; signed-path versus wrapped-word winding defect0.5129762225806935; tree-minus times body-minus PLUS defect7.85046e-17; wrong fitted MINUS defect2.56905; physical SOURCE secondjet versus firstjet-squared defect0.442889. Exact before-compression H value defect0; changed-base/angle mixed H q1/q2 defects1.22329e-8 / 2.77456e-8. Full15 implicit S q1/q2 discriminators3.29026e-7 / 5.47426e-7. Full-Phi/mixed/harmonic ORIGINAL SOURCE q0/q1/q2 defects2.41550e-15 / 3.95690e-8 / 1.44686e-7 against the actual immutable producer; Ref stochastic frame first norm1.1619316373625055, not frozen. Neutral norm1.0000000000000002. Historical A SOURCE norm0.8739869068158571 is explicitly HISTORICAL boost/wholeword, not a canonical new SOURCE norm.

Real Pyright1.1.414 LSP at the project root: current helper467 symbols, full document diagnostics empty, natural exit0, 10.451364261098206 seconds; captured in `local://issue222-guarded-colour-complete-LSP-wrapper.json`. Pre-edit existing owner source LSP195 symbols and10 pre-existing diagnostics was exercised, not called clean. Positive ast-grep queries locate the actual original source-coordinate method before editing and typed `target_source_jets` afterwards. No project-wide tests/build/lint/format or frozen supplier reruns were performed.

Separate noncommuting time-ordered control, `local://issue222-guarded-colour-timeordered-wrapper.json`, exits0/1.148248147s: coarse/fine refinement2.29665e-11, fine unitarity6.66135e-14, wrong exponential-of-integral defect0.3323165953. Exact legal-N guard control, `local://issue222-guarded-colour-legalprofile-wrapper.json`, exits0/1.443047206s: alpha at centre/inner/midcollar/outer/FAR is exactly `[1,1,.5,0,0]`. Integer periodic differences are formed BEFORE binary64 conversion; the adverse floating-N/2 calculation erases the midpoint displacement completely because its ulp exceeds the guard radius.

These are actual same-NPZ native controls and analytic coordinate identities, not MinSet preparation, full legal-N/Haar closure, physical SO3 equivalence, field-only/physical Noether stress comparison or new Born transfer. Those open obligations remain in the shared governing plan.

## Uniform PRIMARY SOURCE derivatives through order six

The continuation adds `JetSeries`, `guarded_frame_series`, `centre_scalar_series` and `primary_source_series`. Their coefficients are Taylor coefficients, unlike `Jet2`'s actual derivatives: derivative q is q! times coefficient q. They differentiate the actual whitening, least-intertwiner, polar and SU4 determinant recurrences; no fitted spin matrix, fixed Ref frame, firstjet-squared prescription or denominator-free character derivative is introduced.

The independent normal proof supplies a regular MinSet tube with Gram minimum 1/40, least-intertwiner eigenvalue gap at least 2, polar candidate singular minimum 1/4 and whole-four-Phi Frobenius norm at most 8. See `local://issue222-minset-normal-proof-capsule.json`, SHA256 `3b0cc3c39511e174529433e65a4dd5a66d3dc5585a56535dda8ff192b1a39cee`. Its actual microscopic chart-map contract is `||D^j map|| <= j! J^(j+1)`, with log2 J at most 222420, not `J^j`. Its raw field displacement is bounded by `4 J² r`: log2 at most -1341410 using log2 r at most -1786252. The source proof is local on this regular tube; it does not invent a global determinant-phase chart.

### Positive rational recurrences

`primary_source_derivative_bounds` computes exact nonnegative rational coefficients with the following outward recurrences. For `G(t)=sum G_n t^n`, let s be a rational lower bound for sqrt(lambda_min G_0). Square-root and inverse-square-root coefficient norms are bounded by

`R_n = (g_n + sum_(1<=k<n) R_k R_(n-k))/(2s)`,

`C_0=1/s`, `C_n=C_0 sum_(1<=k<=n) R_k C_(n-k)`.

The inverse Sylvester norm is at most 1/(2s). In Cartesian coordinates normalized by Tr(B_a B_b)/4=delta_ab, a unit raw direction has total Phi Frobenius norm at most 2, so Gram coefficients are bounded by `[0,8,1,0,...]`. Whitened Phi coefficients are the convolution of C with `[8,2,0,...]`. The full stacked intertwiner has coefficients bounded by `L_0=2 gamma_0+8`, `L_n=2 gamma_n`; hence `K_n <= sum L_k L_(n-k)` for K=L†L.

With v_0 of norm one, the actual normalized eigenvector recurrence projects `K_n v_0 + ... - e_1 v_(n-1) - ...` against the reduced inverse of K_0-e_0, whose norm is at most 1/2. The eigenvalue reaction is retained. The parallel component is `-sum_(1<=k<n) v_k† v_(n-k)/2`, not zero. The exported recurrence bounds both components and all eigenvalue coefficients by product-rule majorants. Every empty sum is seeded with exact Fraction(0), avoiding the previously observed float-zero contamination.

Reshape v to the polar candidate P and apply the same inverse-square-root recurrence to P†P with minimum 1/16. For U=P(P†P)^(-1/2), the determinant correction follows `(log det U)'=Tr(U†U')`; the correction exponent coefficients are bounded by

`h_n <= sum_(0<=k<n) (n-k) u_k u_(n-k)/(4n)`.

The scalar exponential recurrence is `f_n=(1/n)sum_(1<=k<=n) k h_k f_(n-k)`. Thus frame coefficients contain the determinant reactions as well as whitening/eigenvector/polar reactions. Local Z4 choices cancel between the ORIGINAL peredge product and its endpoint matrix, so these bounds apply consistently across the finite chart union without selecting a global phase.

For each oriented edge, the exact word is W=F_x† U_xy F_y; a backward edge uses the adjoint. All interior frames and both endpoint frames are differentiated. The operator-norm coefficient of a unit left-SU4 link flow is at most 1/n!. Convolve both frame series with this link series to bound w_n. The ORIGINAL character scores are `eta ReTr(z†W)`, so `|score_n| <= 4 eta w_n`, using `|Tr X|<=4||X||op`. The factor is FOUR, not the incorrect mixed Frobenius/operator convention of two. The first higher-source receipt was superseded by this correction.

Normalize the four positive seeds at the base point. If a_n bounds each normalized exponential coefficient, the denominator and numerator divided by the base denominator are both bounded by a_n. Since |psi1/psi0|<=1, every quotient coefficient is bounded by

`r_0=1`, `r_n=a_n+sum_(1<=k<=n) a_k r_(n-k)`.

This retains every denominator reaction through q6. Take the truncated-polynomial power r^L with L=3N+2, which covers the original signed axis-ordered paths of length at most 3(N-1) and the owner's two-edge allowance. Binary polynomial exponentiation retains exact finite-N dependence without enumerating an astronomical word. It is not a wrapped-positive-path replacement.

Convolve this product with both external frames and the exact harmonic phase. Harmonic displacement norm is bounded by the owner's conservative 4, giving coefficients 4^n/n!. No UE occurs inside the phase; coexact derivatives of this canonical source are zero. The raw quartet matrix Q has paired eigenvalues in Q†Q and trace 1.65, so ||Q||²<=.825<1. No beta normalization or Gram whitening is applied.

If b_n is the resulting point Taylor majorant, use `q^q b_q` as an ambient mixed/ordered derivative bound. Real polarization gives the frame-block factor k^k/k!; ordered left-Lie derivatives of a link matrix have norm at most one and do not require commuting Lie fields. Product/quotient differentiation distributes q derivatives among blocks. For every partition q=sum k_i,

`product_i (k_i^k_i/k_i!) <= q^q/q!`,

because the associated multinomial probability is at most one. Thus the same q^q majorant bounds ordered colour derivatives as well as Cartesian mixed derivatives; no claim that colour vector fields commute is needed. This also bounds the diagonal X_a² terms of the actual Haar Casimir directly.

The full operator B sums every birth point and all 4x4 light/Ref CAR entries, whose individual operator norms are at most one. Therefore `M_q <= 16 ||eta||_1 q^q b_q`, with `M_0<=16 sqrt(V)` for a normalized profile. This is a full Ref/before-compression bound, not a selected stochastic column or a 45-Phi normal restriction.

### Full kinetic and physical-Lie coefficients

The independent ambient export includes every 60V Cartesian Phi and every 45V colour left-Lie derivative. Its canonical source kinetic coefficient is

`M1_kin² = (30V+13.5V) M_1²`,

so the derivative energy term is `chi M1_kin²`. The colour weight .3 is from the exact .6chi cometric; it is not .15. The harmonic derivative coefficient is `Mtheta² <= 16 M_0²`, entering energy as `(UE/N) Mtheta²`. These are actual coefficients, not a creator-normalization bound: A must insert its actual positive Z and prepared-state graph bounds. In particular, source derivative energy after normalization is bounded by `(chi M1_kin²+(UE/N)Mtheta²)/Z²`; this expression is not the entire created-state energy.

`kinetic_source_commutator_coefficients` gives all ambient Leibniz cross and diagonal second-derivative terms of `[H,B]psi`, plus exact harmonic coefficients and the norm bound `||[beta,B]||<=2 beta_TOTAL M_0`. Colour derivatives are the actual left-vector-field Casimir terms, not a flat-Haar substitute. CAR, Coulomb, actual state gradients and higher Hamiltonian words remain A's integration inputs.

`source_physical_Lie_graph` returns the genuine field-coefficient graph:

`||Y B|| <= M_1 ||Y||`,

`||Y² B|| <= M_1 ||D_Y Y|| + M_2 ||Y||²`.

The second expression retains DB[D_Y Y]. B supplies the full orbital/clover/EM/Gauss/Root physical vector and its acceleration; the new native adverse control exercises only the actual intrinsic Phi-vector component, not a completed full Noether generator.

`normal_coordinate_polynomials` implements the actual Faà di Bruno bound

`D^n(f o map) <= sum_(k=1..n) n!/k! binom(n-1,k-1) M_k J^(n+k)`.

`covariant_normal_source_bounds` adds the specified light/Ref comparison-frame connection, bounding both module ends with representation norm 4V+4. If C_Fj are frame normal derivatives, its connection derivatives obey `A_j<=2(4V+4)sum binom(j,k) C_Fk C_F(j-k+1)`. Ordered covariant product-rule recurrences, not diagonal-only polarization, give the resulting positive J-polynomials. The cometric bound 4 contributes 2 per contracted index; the canonical normal M1² coefficient uses 45V after the 90V normal dimension and chi/2 weight. This export concerns the specified frame comparison. It does not silently assign unknown Berry connections to it, prove higher Laplace-Beltrami metric/drift commutators, or replace A's complete Hamiltonian graph. Heavy-only CAR Berry connections may be removed only via the owner's actual commutation proof with this even light/Ref source.

At legal N the exact raw M0..6 logarithmic upper bounds are `[152,288,425,563,700,838,975]`, ambient canonical M1² log bound 876, and harmonic Mtheta² log bound 307. The comparison-frame covariant M0..6 logarithmic upper bounds are `[152,445319,890486,1335653,1780820,2225988,2671155]`; canonical covariant M1² log bound is 890938. Gamma's 512-chart cutoff derivatives remain separate. Neither the selected Lambda nor SOURCE is retuned.

### New q6 consumer and immutable source closure

The new CLI branch is `--higher-source-bounds --normal-capsule <normal capsule>`, with the same `--state-capsule` input as above. It returns before A's Hamiltonian constructor and does not repeat the old 30-second supplier control. It maps the actual cached neutral amplitudes through the owner's Q tensor Q isometry, retaining all 81 positive coordinate weights. At every node it computes actual q0..6 source/frame series on all 27 birth sites and all four Ref columns, compares coefficient/CAR operator-majorant graphs against the proved bounds and compares q0 directly against the immutable `screened_marks` producer.

The final exact argv/cwd/runtime/input hashes/streams/natural-exit receipt is `local://issue222-guarded-colour-higher-source-packet-wrapper.json`. Natural exit0 in 9.14734426792711 seconds, empty stderr. Source SHA256 `b98257990b1bbbb97d3d076cbee844a04a91503a9210b0bad7a9b949d7789aa0`. Same cached NPZ and operator signature, no re-preparation. All 62 loaded project-local scientific Python sources were hashed before and after the scientific path and were stable; their exact immutable byte closure is `local://issue222-guarded-colour-q6-packet-frozen-source-manifest.json`. Immutable producer defect 2.3988936430970036e-15; measured regular guards are Gram .05836346050841393, gap 3.9999999999999933, polar .4999999999999969. Ambient probe direction norm .9814901935322635.

Observed weighted coefficient/CAR graph upper values q0..6 are `[15.908861506461031,1.792234572099916,.3768110612786344,.23120498294021274,.14700565958531148,.1492951972311044,.15933180836892957]`. These are honest before-compression coefficient majorants on the actual cached coordinate amplitudes, not a new creator norm Z or an astronomical MinSet/Haar state measurement.

Independent adverse defects: peredge versus whole-tree scalar .023696286263850834; wrong BOOST quartet 1.129858274295787; freezing stochastic Ref .13019176727290485; fitted firstjet-squared .44288899986717095; omitting physical acceleration DB[D_Y Y] .03340388679635922. All are required nonzero by the consumer. Exact q6 coefficient graphs pass the uniform bounds. An initial failed Fraction-zero run and superseded two-times-trace bound are preserved in separate historical receipts, not relabelled as final evidence.

Final real Pyright1.1.414 LSP: 767 symbols, zero diagnostics, natural exit0/15.592911603860557 seconds; `local://issue222-guarded-colour-higher-source-packet-LSP-wrapper.json`. Earlier scoped LSP diagnostics identified six Fraction seed/type issues and subsequently nullable source filenames; both were fixed without suppressions. Positive typed AST discovery located `primary_source_derivative_bounds`. No project-wide validation or frozen supplier rerun was performed.

B's independently executed earlier primary q2 consumer is bound to EXACT helper SHA256 `093c550623afb07f42b38d594cc7a411b4b6ac9106764407ac80b1958fbc6169`, archived at `local://issue222-guarded-colour-source-093c550623afb07f42b38d594cc7a411b4b6ac9106764407ac80b1958fbc6169.py`. Those bytes were recovered by reversing explicit later patches and verified hash-identical, not replaced with current q6 bytes or retagged. Its q2 observation is distinct from this new uniform q6 proof/consumer.

## RAWTRIAL Source19 normalization and calibration

This continuation proves the missing ORIGINAL Source19 extension for A's selected RAW trial tau, not for its filtered state. Governing plan remains `issue-222-fullfield-closedrotation-continuation-20261006`, pinned general template89cd9d; the last observed canonical publication revision is `60d7474aaad0fcc9e6036c461016cc78f5c9732744ef4ce7f9d24ce4b0f35a14`. A alone owns H/state/window integration, Normal the field/cutoff proof, and Transport the final physical Born consumer.

The exact input is `local://issue222-legal-volume-far-RIGHT-global-Gram-b4096-descriptor.json`, SHA256 `020fb175c9231b607fb0e53b016cf97528af9065a634dedc7ddea82bdac3556c`. Its nineteen actual complex weights, offsets and fixed raw quartet are not renormalized, whitened, conjugated or fitted. The descriptor's unrelated supplied comparison matrix is not substituted for the raw producer matrix. Source19's stored binary64 weight squared norm exceeds one by approximately3.2537553911331787e-16; the MP calibration uses the exact stored bits rather than silently resetting this norm to one. The existing legal q6 majorant still applies because its supplied profile l1 bound exceeds19, while every one of these nineteen weights has modulus below one.

### Actual m/s, SU4 seams and signed words

For each m in {0,1,2,3}³ let zeta_j=i^m_j. A valid centre representative has uniform metric-Clifford Phi, links I except the last j-axis seam links zeta_j I, and flat auxiliary cycle history s in {0,1}³. The auxiliary sign (-1)^s_j is independent of the colour seam. The matter Bloch shift is precisely pi(m_j/2+s_j)/N, with exact integer mode relabelling retained when m+2s exceeds3. A repeated bulk link i^m_j I is NOT this legal-N class: N is divisible by four, so that erroneous bulk assignment has holonomy I.

An equivalent bulk gauge uses the SU4 Cartan root

`V_j=diag(exp(-3i a_j),exp(i a_j),exp(i a_j),exp(i a_j)), a_j=pi m_j/(2N)`,

so det V_j=1 and V_j^N=zeta_j I. Put g_x=product V_j^(-x_j), transform BOTH Phi_x to g_x Phi g_x† and links to g_x U_xy g_y†. Bulk links become V_j, but Phi is then spatially twisted. Its source frames are g_x F, up to cancelling local Z4 choices. Thus the framed word is I on ordinary edges and zeta_j I on the seam, not a peredge i^m phase or a scalar non-SU4 Nth root with frozen Phi.

At centre_scalar(zeta I,18), the exact original ratio is zeta*c18, where

`c18=(exp72-exp(-72))/(exp72+exp(-72)+2)=tanh36<1`.

The specified nineteen births x=(D,D,D)+delta and Ref=(0,D,D), D=N/2, have ORIGINAL monotone signed Birth-to-Ref paths with no seam crossing and length

`ell_delta=D+delta_x+|delta_y|+|delta_z|`.

Backwards traversals use U†. All centre/Root/Bloch gauge changes contribute unit phases with the matching matter basis and endpoint frames, so none alters the absolute peredge product c18^ell_delta. Harmonic and Coulomb/Dirac phases remain explicitly in the operator; they are not discarded because delta is small.

### Exact allRef projector identity, including offsite terms

Write chi0=Gamma0 Gamma1 Gamma2 Gamma3. For every actual positive finite-N Gram minimizer G_N(m,s), the Clifford lift Phi_mu=sum sqrt(G)_mu,nu Gamma_nu makes the literal improved light Hamiltonian anticommute with chi0. Its entire occupied negative2V cluster and empty positive2V cluster satisfy

`chi0 P_empty chi0=I-P_empty`.

This holds for every actual m/s Bloch shift and arbitrary real harmonic shift, not just one momentum or an averaged spin trace. No G_infinity or binary64 template is assigned as an exact centre.

A's RAW trial uses this fixed centre negative Slater transported by the EXACT original source frame F_x(Phi) at EVERY light site, including the independent Ref frame. It does not use an unproved actual offcentre light negative cluster. Accordingly chi_light(q)=diag(F_x chi0 F_x†) and chi_Ref(q)=F_Ref chi0 F_Ref†. The actual primitive source maps are

`nu_a(x)=w_x S_x(Phi,U) exp(i phase_x) F_x L_a F_Ref†/2`,

with L=[I,Gamma2 Gamma3,Gamma3 Gamma1,Gamma1 Gamma2], every raw Ref column and S_x the ORIGINAL scalar product over all signed edges. Each L_a commutes with chi0, so `chi_light nu_a=nu_a chi_Ref`. On the fully occupied neutral Ref wedge4 the CAR source Gram is exactly

`G_ab=Tr_Ref(nu_a† P_empty(q) nu_b)`.

Conjugating inside this COMPLETE trace by chi_Ref gives the same expression with I-P_empty(q). Therefore

`2 G_ab=Tr_Ref(nu_a†nu_b)`,

`G_ab=(1/2) sum_x |w_x S_x(Phi,U)|² delta_ab`.

This identity kills every offsite light-projector contribution, including arbitrary harmonic/Coulomb and Root path phases. It survives arbitrary regular Phi and colour coordinates of this SPECIFIC frame-transported raw trial: the outside frames cancel in the complete Ref trace, not the interior/endpoint frame dependence of S_x. Ref stochastic SOURCE derivatives and physical Noether jets are not zeroed. The PH-conjugate light/Ref0 construction has the identical norm.

The actual normalized whole-heavy occupied Kato fibre supplies no additional source norm weight. Heavy-even operators, their Riesz occupied projector P_H and differentiated Kato transport commute the even light/Ref source on the full encoded auxiliary algebra; this is the separately supplied proof `local://issue222-heavy-source-ambient-commutation-capsule.json`, not a momentum-wise imposed heavy filling or a projection of beta. SOURCE also preserves all cycles/P_s. Hence coherent initial MinSet tie weights are admissible; beta may mix them under subsequent H evolution.

The displayed scalar Gram is a DERIVED property of this explicit RAW trial and full Ref protocol, not an imposed generic scalar EndMark. Applying a full-H window can destroy its fixed-Slater factorization. Its actual filtered Hermitian Gram/norm remains A's independent responsibility.

### Full finite-N calibration, without losing the c18^D factor

At each actual flat centre the exact primitive quartet Gram is

`G0=(1/2)c18^(2D) sum_delta |w_delta|² c18^(2[delta_x+|delta_y|+|delta_z|]) I4`.

Thus `Z0²=beta†G0 beta`; the ideal decimal quartet prefactor is33/40, while the implementation records the exact exported binary64 beta bits. `raw_source19_centre_calibration` performs 256-bit outward Arb arithmetic for those actual weights and the FULL astronomical integer exponents. Its observed intervals are

`Z0²=[0.79057233670040806076224452153360340111877132941 +/-4.61e-48]`,

`Z0=[0.88914134798715106248893029004564976481442163394 +/-1.61e-48]`.

In particular Z0>113/128. Replacing the peredge product by one whole-tree centre ratio gives a norm-squared defect above.03; this is not a negligible local profile adjustment. A global c18^D factor cancels only after division by the actual source normalizer, never by declaring each peredge ratio unit.

### Gaussian and uniform-tube perturbation to the actual RAW tau

Consume the frozen full90V normal proof `local://issue222-minset-normal-proof-receipt.json`, SHA256 `07180616d355175aee1a4ff23c9b9dc1a2dd55a072144ed6073f28fce8560d77`. The actual correlated Riccati/half-density Gaussian with normalized radial cutoff has the conservative A/Normal moment

`E_tau ||q||²<=4 M Lambda^(-3/4)`, log2 M<=402.

No diagonal independent Gaussian or filtered-state moment is substituted. The regular normal chart obeys raw displacement<=4J²||q||, log2 J<=222420 and log2 r<=-1786252. The trial support is within2r/3. Its heavy/light/Ref fibre maps are unitary and supply no extra density weights.

In the co-moving light/Ref basis, SOURCE's outside F_x/F_Ref factors cancel EXACTLY. The remaining matrix is Qraw times the full signed scalar path product. Its derivative bound is a positive SUBSERIES of the proved q6 ambient majorant: remove the two outside positive frame series, but retain every interior and both endpoint frame reactions in the path. Consequently no speculative light projector transfer or a new CAR representation factor is needed. With ambient canonical M1²<=2^876 and comparison-frame canonical M1²<=2^890938, valid alternative L2 perturbations are

`delta_ambient<=8 J² M1ambient sqrt(M) Lambda^(-3/8)`,

`delta_covariant<=2 M1covariant sqrt(M) Lambda^(-3/8)`.

For L=log2 Lambda, eight times their log2 bounds are respectively `3563856-3L` and `3565368-3L`. The UNCHANGED selected tower has L>=2^315, so both are far below delta=1/128. Independently the actual support and raw C1<=2^288 give the pointwise co-moving source perturbation bound `4J² C1 r<=2^-1341122`.

Initial m components are not assumed orthogonal or conserved: they have disjoint actual cutoff supports by a holonomy estimate. A unitary N-edge product changes by at most N times the maximal link change. Thus each m tube's gauge-invariant holonomy lies within `4N J²r<=2^-1341311` of zeta_m I. Different zeta values have operator distance at least sqrt2, so these supports cannot overlap. Different s are orthogonal cycle eigenspaces preserved by SOURCE. This proves the norm sum for amplitudes1/8 and arbitrary normalized actual-minimizer tie weights without imposing Hamiltonian superselection.

The reverse triangle inequality now gives, for the selected complete RAW tau,

`||B_Source19 tau|| >=113/128-1/128=7/8`, `||B_Source19 tau||²>=49/64`.

`raw_source19_trial_norm_bound` exports both moment routes, the independent uniform-tube bound and the holonomy separation certificate. Harmonic/Coulomb phases have zero cost in THIS norm calibration only; their nonzero SOURCE derivative/energy graphs, including UE/N Mtheta², remain intact. This does not prove a filtered creator norm, spectral window, physical photon input with an additional1/sqrt2 convention, or Born amplitude.

### Actual new Source19 controls and source closure

The permanent CLI is `--source19-rawtrial-norm --descriptor <Source19 JSON> --normal-proof-receipt <frozen normal receipt>`. It does not require or prepare a replacement cached state. The actual receipt is `local://issue222-guarded-source19-rawtrial-wrapper.json`: natural exit0/8.645353931933641 seconds, empty stderr, exact Python SHA256 `e167947509c9617ddd0cb3d09ddd3cbc9d5d5767743fc4bf14b5f1f64dda4f9c`.

The NEW N8 profile/phase control uses one FROZEN actual N5 skew full10 Gram and all512 ordinary/auxiliary Bloch phases, with every one of the512 N8 momentum modes and the true occupied/empty projectors. It is not512 new argmin solves, A's full500-matrix sector rerun, old Source19 fullCAR, q128 or an astronomical tau simulation. FullRef weighted G scalar error1.776905764155918e-15; smallest light gap.1294244046775637. All64 explicit SU4 Cartan/seam framed words agree within8.728499705679552e-15; determinant defect1.6653345369377348e-16. The immutable original source producer agrees with the nineteen signed matrices to4.906586848572109e-17 and with the paired SU4 gauge transformation to1.7642638159268156e-15.

Wrong singleRef Gram defect.7963691248607296; wrong BOOST source-matrix defect1.140175425099138; wrong positive-wrapped word defect1.8165902124584954; erroneous repeated i^m bulk holonomy defect2; wrong wholeTree scalar defect2 in its explicitly off-tube family discriminator. A scalar Gram ALONE does not distinguish BOOST, since those even matrices can also satisfy a chiral trace identity: the actual source-matrix binding is required and exercised.

All74 loaded scientific source files were hashed before/after and frozen exactly in `local://issue222-guarded-source19-frozen-source-manifest.json`. Real Pyright1.1.414 LSP, `local://issue222-guarded-source19-rawtrial-LSP-wrapper.json`, has905 symbols/zero diagnostics, natural exit0/18.152623616857454 seconds. Positive typed AST locates the new raw trial norm API. The48 original q6 functions/classes and their top-level imports/assignments are normalized-AST IDENTICAL to frozen b982 in `local://issue222-guarded-source19-q6-AST-identity-bridge.json`; only three Source19 APIs and its CLI branch are added. No old receipt is retagged. The temporary staging runner was removed after smoke proof; its exact historical bytes are archived. No further512 mean-sea, frozen fullCAR/q128 or project-wide validation was rerun.

## Generic scalar RAW normalizer and the actual narrow SOFT packet

This continuation extends the proved allRef/chiral trace identity to the actual B SOFT input, rather than relabelling the point primitive or Source19 amplitude. B explicitly binds the CURRENT legal source to the BARE scalar `f/scalarNorm` BEFORE its one actual full-CAR creator normalization, the original spatial-bivector Qraw/global2 and all4 Ref columns. There is no additional momentum-dependent input spinmatrix. P+(p) in the mean packet calculation is the filled-sea Pauli-allowed OUTPUT, not input preparation hardware.

The input is the frozen `local://issue222-soft-spatial-closed-rotation-sourcebound-receipt.json`, SHA256 `da856582ab9da56fd0857d2cd6e07451503b571b919f0e9dab0da65f5be47c3a`. Its actual selected row has:

- width `1.0000000000000001e-7`, centre `(2.0000000000000002e-7,0,0)`;
- scalar norm `9.803198633456658e-12`, a stored binary64 cubature constant;
- unchanged scale `.0001`, bump `f(p)=exp[-1/(1-|p-p0|²/w²)]` inside the w-ball and zero outside.

All those exact binary64 constants are retained. The historical native `source_seed` has a short-tree cOne and a global1/sqrt2, subsequently cancelled by its historical mean CAR normalization. Neither cOne, sqrt2, nor the stored old `.41249999999999626` CAR norm is incorporated into CURRENT eta. The CURRENT full legal source has its own original peredge word and its own single actual normalizer. No percolumn normalization is performed. Its spatial centre can be translated by the appropriate Fourier phase without changing any norm in the following proof.

The frozen record's old R/T and evolved-tail values are not consumed by this scalar norm proof. B's separately authorized geometry successor doubles R/T while leaving w, p0 and scalarNorm unchanged; this certificate neither retags the old geometry receipt nor reprices the new clock/Cauchy tail.

### Exact finite-N scalar norm, not a cubature identity

Let h=2pi/N. The explicit unitary Fourier convention is

`eta_hat(k)=h^(3/2) f(p_k)/stored_scalarNorm`,

`eta_x=V^(-1/2) sum_k eta_hat(k) exp(i p_k·x)`.

Thus exact finite-N Parseval gives `||eta||2²=h³ sum_k f(p_k)²/stored_scalarNorm²`. This is not defined to equal one. Every spatial coefficient, including its tail and the value at FAR, is retained; no compact spatial cutoff is used to manufacture normalization or an l1 cap.

For the exact radial squared density g(t)=exp[-2/(1-t²)], t<1, define

`I=4pi w³ integral_0^1 t² g(t) dt`.

g decreases. On a rational panel [a,b], `a²g(b)<=t²g(t)<=b²g(a)`. `narrow_soft_profile_norm_certificate` uses8192 such panels and192-bit outward Arb arithmetic to enclose I between explicit positive lower and upper sums. It compares those sums with the EXACT stored scalarNorm bits, not with an assumed exact20-node Gaussian quadrature result.

For ANY real translate of the finite momentum grid, nearest-cell radius is delta=sqrt3*pi/N. The globally extended bump squared has Lipschitz constant at most4/w: with u=1/(1-t²)>=1, its radial derivative is at most `(4/w)u² exp(-2u)<=4/w`. A nonzero point-versus-cell-centre difference has either its point or its cell centre in the w-ball, and hence lies within the w+delta ball. Consequently

`|h³ sum_k f(p_k)²-I| <= (4/w) delta (4pi/3)(w+delta)³`.

The actual B ball and cells are strictly inside the BZ cut; the bound is uniform over the all64 colour/all8 auxiliary Bloch shifts, without solving512 new minimizers. At legal N the observed outward relative mesh-error upper is

`[2.39482568672630435969288388078716593385828363896699951676e-20 +/-9.30e-77]`.

After combining this with the radial lower/upper sums and the stored cubature constant, the exact finite-N squared scalar norm satisfies the certified bounds

`0.99940107928496263725670390851548542809486885616212953 < ||eta||2²`,

`||eta||2² <1.00059931482852401423723527409580363691817397093408835`,

where the actual outward ball radii are retained in the JSON receipt. In particular `31/32<=||eta||2<=33/32`. Finite-V Cauchy-Schwarz therefore gives `||eta||1<=sqrtV||eta||2<= (33/32)ceil(sqrtV)`. This cap requires no assumed zero spatial tail, continuum/discrete equality, or perRef fitting.

### Actual whole signed word: uniform RAW lower for scalar profiles

For any scalar eta and every actual finite-N positive centre G_N(m,s), the preceding allRef/chi trace proof gives

`||B_eta tau_centre||²=(||beta||2²/2) sum_x |eta_x S_x|²`.

This is a RAW trial identity derived from its explicit fixed centre negative Slater transported by the original exact source frames. It is not a filtered Hermitian Gram hypothesis. The actual whole-heavy occupied Riesz/Kato fibre and its heavy-even/source commutation contribute no norm weight; allreal harmonic/Coulomb endpoint phases are retained but disappear in this specific complete trace. Distinct colour-well supports and cycle labels sum coherently as proved earlier; no Hamiltonian superselection or beta commutation is imposed.

Every ORIGINAL monotone signed source path satisfies ell<=3N+2; backwards edges use U†. Its flat centre magnitude is c18^ell, including the actual SU4 seams/Bloch gauge as above. An EXACT positive Taylor sum `sum_(k=0)^144 72^k/k!>2^103` proves exp72>2^103, hence

`1-c18=2/(exp72+1)<2^-102`.

Bernoulli then gives, at UNCHANGED N=5*2^96,

`c18^ell >=1-(3N+2)2^-102 =49/64-2^-101 >3/4`.

This retains the full astronomical peredge count. The literal exported beta bits give `||beta||2²/2>=4/5`. Thus the centre source norm is at least `(3/5)||eta||2`, conservatively below the sharper sqrt(9/20)||eta||2. This generic scalar lemma applies to arbitrary complex spatial coefficients of known l2 norm with the FIXED raw quartet/allRef, not to an arbitrary momentum-dependent spinmatrix or a changed receiver family.

### Homogeneous perturbation and unchanged common preparation budget

Use the same frozen full90V Riccati/half-density cutoff moment Mlog402, Jlog222420, rlog-1786252, and unchanged selected Lambda. Homogeneity gives the raw bound `Cq(eta)<=||eta||2*Cq_unit` from l1<=sqrtV||eta||2, where Cq_unit is the existing q6 majorant at ceil(sqrtV). This is only a proof of ratios: eta and its stored scalarNorm are NOT changed in hardware.

For the co-moving scalar-path source, the two Gaussian routes bound the relative perturbation by

`delta/||eta||2<=8J² M1ambient_unit sqrtM Lambda^-3/8`,

`delta/||eta||2<=2M1covariant_unit sqrtM Lambda^-3/8`.

Their eight-times-log2 upper bounds remain `3563856-3log2Lambda` and `3565368-3log2Lambda`. C1 already bounds every ambient mixed-unit direction. The canonical quantity M1²=43.5V C1² also dominates that directional bound, without incorrectly inverting the .5/.3 kinetic weights to infer an unweighted Hilbert-Schmidt norm. The separately proved uniform tube route is `delta/||eta||2<=2^-1341122`. All are below1/128 at the unchanged selected tower. q6 kinetic and harmonic costs remain present; no SOURCE derivative is discarded because its phase cancels from the raw norm.

Therefore

`||B_eta tau|| >=(3/5-1/128)||eta||2=(379/640)||eta||2`,

`||B_SOFT tau|| >=11749/20480=0.573681640625 >1/2`.

The actual RAW squared norm is consequently at least1/4. For the normalized q0..6 graph ledger, the conservative ratio between the actual profile upper bound and this RAW normalizer lower bound is

`||eta||2_upper/Zraw_lower <=21120/11749=1.797599795727296... <2`.

Thus A's EXISTING factor2 unit-profile graph budget and common preparation width need not be changed. No claim that raw eta has norm<=1 is used. A alone must apply its actual source-first fn/Gamma estimate with this amplitude homogeneity, prove its filtered creator/photon norm, and bind its unchanged common window; those are not inferred here from the RAW lower.

This RAW inequality is also state-amplitude homogeneous. For a shadow `Psi=sum_(m,s) a_(m,s) tau_(m,s)` retaining the supplied unit branch fibres and disjoint m/cycle supports, it gives `||B_eta Psi||>=(379/640)||eta||2||Psi||`; Source19 analogously gives `(7/8)||Psi||`. The uniform-tube route also covers coefficient functions retaining those fibres and regular support, without assuming their weighted Gaussian moment remains unchanged. It does NOT identify an actual full-H filtered state with that shadow or give the shadow norm one.

A has explicitly withdrawn the earlier m-specific global64 prepared-state admission: the actual successor uses ONE physical global fn/e_global, retained actual finite v_m, Evac/ordinary phases and a weighted Dirichlet shadow with norm of order1/8. Its quasimode/normalizer factor and new filtered source/current admission require A's corrected receipt. The RAW source proof and scalar-profile certificate remain valid; no per-m filter or old all64 prepared norm is inferred.

### Permanent consumer, controls and immutable compatibility

`generic_scalar_profile_raw_norm_bound(profile_certificate,frozen_normal_receipt)` exports the relative/raw lower, exact ratio, homogeneous unit q6 bounds and both moment/tube routes. `narrow_soft_profile_norm_certificate(decoded_B_receipt,N)` exports the actual finite-grid l2/l1 certificate. The new permanent CLI is:

    /home/administrator/substrate-framework/.venv/bin/python \
      proposals/P256-qed-backward/prepared_guard_colour_chart.py \
      --soft-rawtrial-norm --soft-packet-receipt FROZEN_B_RECEIPT \
      --normal-proof-receipt FROZEN_FULL90V_NORMAL_RECEIPT

Its actual receipt `local://issue222-guarded-soft-rawtrial-norm-wrapper.json` has natural exit0/4.129216256085783 seconds, empty stderr, root cwd and exact argv/input hashes. Executed Python SHA256 is `f57105d58fb99c9aed578d1b099d383b4cd1998b6121d855887f9e4d0628d56c`.

The new native N8 broad-bump control is ONLY a finiteDFT/source/CAR contraction discriminator, not a retuned normative narrow packet or512-history/argmin solve. It uses ONE512-mode projector from the frozen actual N5 skew full10 Gram and original SOURCE producer with a nonzero colour link and harmonic phase. Parseval defect4.163336342344337e-17; actual constant B seed-matrix defect3.98609107286759e-17. FullRef source norm squared.07353963598973226 agrees with the chiral trace.07353963598973229; chiral complement defect2.3821817923139768e-15.

An incorrectly momentum-dependent occupied-band INPUT has source norm squared-4.555646828222436e-18 (numerical zero), although the scalar profile is unchanged; hence the constant raw seed contract cannot be dropped. SingleRef instead gives.015917873573910904; percolumn-fitted seed-matrix defect.7178507404045701. A separate N31 grid with the ACTUAL narrow width/centre has ZERO supported momentum modes/zero exact scalar norm, despite its unchanged positive continuum cubature constant: the false cubature-to-discrete normalization is rejected, not silently assigned one.

All80 scientific source files were stable before/after the new CLI and exact bytes are frozen in `local://issue222-guarded-soft-frozen-source-manifest.json`. The51 pre-existing q6/Source19 functions/classes and all top-level imports/assignments are AST-identical to frozen e167 in `local://issue222-guarded-soft-prior-API-AST-identity-bridge.json`; only these three SOFT APIs and their CLI branch are new. Real Pyright1.1.414 finalLSP `local://issue222-guarded-soft-rawtrial-norm-LSP-wrapper.json`:1027 symbols, zero diagnostics, natural exit0/20.23795957211405 seconds. Positive AST locates the new generic bound. No prior receipt is retagged and no full500,512-history, fullCAR, q128, old supplier or project-wide test was rerun. The canonical plan/model remain the publisher's, and the campaign remains OPEN.

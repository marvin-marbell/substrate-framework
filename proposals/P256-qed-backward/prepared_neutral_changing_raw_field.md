# Actual changing-quintic RAW Phi/colour characteristic

This new continuation uses the original `primitive_flow` provider on **60 Phi plus 45 colour coordinates per site**, hence all **2835 native RAW coordinates** at N3. It is the first-order physical Noether action, not a substitute harmonic law, a full quantum-H propagator, a new ground or a completed physical rotation.

`evolve_raw_fields(law, phi, links, velocity, alpha, clock, t0, t1, steps)` integrates `dRAW/dt = -Omega(t) V(RAW)`. The fixed original Noether vector field is autonomous, so the quintic integral is the exact angle difference; the nonlinear field characteristic itself uses explicitly finite-step midpoint Lie updates. Links are advanced by matrix exponentials of the actual left generator, not linearly projected back to SU4. Inverse intervals are supported inside the pulse. Field functions and preparation are never recaptured.

The observed native interval is `209.42 → 209.451`, clock **566**, pulse fractions near `.37`. Actual integrated angle is `.0005609930232257643`. The initial point includes nonzero normal field directions and noncentral colour links (`||U-I|| = 2.2335901148714923e-5`). Width `.0001220703125`, chi `.000244140625`, full mean Noether normal-vector norm `.8394354753145165`: even this short characteristic moves fields substantially relative to the prepared Gaussian width. A common boost phase alone is not this motion.

Actual CLI natural exit **0**, wall **174.18288814392872 s**, empty stderr, raw stdout exactly equals the published science JSON, loaded supplier bytes stable. Source SHA256 **a432ec98f7d66469c02bc0e4a2556c905b21d1265b85ca8248a9379274720125**. Frozen-source archive **c2e00d46d9c60ba394cb1dcbdb6211e9494b998934efa368595957aefec3c09b**. Full command and source bindings are in `local://issue222-changing-raw-field-execution.json`; science in `local://issue222-changing-raw-field-science.json`.

Decision-bearing controls:

- Wrong frozen initial Omega changes the Phi endpoint by **6.892549643945255e-8**, versus 4→8 field-step refinement **2.774515793297167e-12**. The colour wrong-frozen difference is smaller, **1.1708893344941862e-13**, and is not promoted to the primary discriminator.
- Forward/inverse Phi defect **8.491573447749823e-16**, colour **5.325922408947535e-22**.
- Actual RAW provider under nontrivial local SU4 gauges: Phi covariance defect **7.950703827998374e-16**, colour **1.8575806995252626e-17**.
- Full-J normal-plus-gauge reconstruction **1.0970391605282429e-14**; endpoint colour unitarity **1.9140115983764386e-14**, determinant defect **2.886580591477797e-15**, Phi Hermiticity **8.413555285992702e-26**.

`colour_haar_divergence` evaluates all 81×15 native left-Haar directions by central differences. Phi divergence is exactly zero for the centred distinct-neighbour blocks and off-diagonal intrinsic Phi1/Phi2 rotation. The observed colour trace is **2.305570489283119e-12**, FD refinement difference **2.533512076764637e-16**. The retained initial-point common log-half-density estimate is **6.467044795215208e-16**, not set to zero. **This run does not claim its full time integral or a rigorous divergence remainder.** Paired-state/source product comparisons must retain the same common density factor rather than independently renormalize pieces.

Final actual Pyright stdio RPC received diagnostics `[]`, 113 symbols, definition/references, six positive AST declarations; exit0 and identical current/scientific source. Receipt `local://issue222-changing-raw-field-current-real-lsp.json`. xdLSP remains unavailable; direct real server is used.

Accepted shared NPZ `10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3` / manifest `90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a`, initial filtered state and all source profiles remain unchanged. No old preparation/H/G/loop rerun. An earlier scale probe failed only on a nonexistent unused `model.r` accessor before producing scientific output; it is not scientific evidence and is not relabelled successful.

The next joined consumer couples this RAW characteristic to all55 EM characteristics and actual time-ordered native matter/Ref matrices, inverse section/gauge reconstruction, common boost/time connection and paired initial Xi/BXi. Original quantum Phi/EM kinetic, scalar potential, Coulomb multipair forms and whole-pulse/common-form completion are separate obligations. This pair alone pays none of full-ground/Gauss/full-field L2/legal-SOFT/physical2pi/Born/quantum-law closure.

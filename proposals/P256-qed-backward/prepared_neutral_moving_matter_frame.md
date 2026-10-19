# Once-prepared full-matter interaction-frame consumer

This pair implements the `prepared-neutral-full-field-outcome` slice of issue222. It continues the accepted shared-photon prepared state; it does not claim a full ground, physical closed-loop sign, Born law, or original quantum-law derivation.

## State, generator and interface

`construct_point_frame(model, data)` loads no new state. Its inputs are the accepted shared cache NPZ `10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3` and manifest `90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a`. The original filtered 14-coordinate neutral coefficient vector, extended weak H/Gram, Source19 and distinct SOFT maps, quartet coefficients, photon packets and all source profiles remain unchanged. No H/G recompilation, eigensolve, refilter, instantaneous-ground reset or profile normalization is performed.

The actual B RAW chart pullback at the mean supplies the 108-dimensional light and heavy orbital/gauge Noether matrices and the four-dimensional reference gauge matrix. There is no assigned spin term. The same prepared-law mean Hamiltonian has **light coefficient 1**, heavy coefficient **−Lambda = −16777216**, and no reference onsite Hamiltonian. Thus the frozen instantaneous mean matrices are

- `A_light = h_light - Omega M_light`;
- `A_heavy = -Lambda h_heavy - Omega M_heavy`;
- `A_reference = -Omega M_reference`.

The full one-body unitaries are `U = exp(-i delta A)`. The occupied light and reference planes and all 54 occupied heavy modes are transported by these full matrices, not by a projected charged transition generator. The connection is `-i U† dU/delta = -A`. The returned dictionary contains `model`, `sample`, `unitaries`, `bodies`, `connection`, `generators`, `pulse`, `local_step`, and `scope`.

**This is a local frozen instantaneous family, not the time-ordered evolution of the changing full pulse.** Default `delta=.031`, quintic pulse fraction `.37`, legal clock `T=566`; `Omega` and `Omega_dot` are the actual derivatives of the quintic angle. The sampled normal displacement contains three nonzero width-scaled coordinates and a component along the full mean Noether vector; all 52 photon coordinates are sampled, with a nonzero U(1) angle vector.

The common field factor is `exp(i Omega Y0·q / chi)` on both neutral and charged coefficient functions. It contributes `Omega_dot Y0·q / chi` to the time connection. The retained completing-square common energy is `-Omega² ||Y0||²/(2 chi)`; no neutral-energy shift is introduced. Physical LOW uses `chi Pi - Omega Y0`, not bare Pi.

## Actual joined execution and adverse branches

The scientific CLI runs the full once-seeded matter frame, boost and connection together with the two completed full-CAR/current leaves. Its exact command is retained in `local://issue222-moving-matter-execution.json`. Natural exit **0**, executor wall **38.4291064911522 s**, program wall **36.14579635602422 s**, empty stderr. Raw stdout equals the published science JSON exactly. All **101** loaded project source files were predeclared, archived and unchanged. Archive SHA256 `a710137d58d0f8d961ba5711fea0c2780d74ff917822c3219aeca470dac5e4b1`; raw stdout SHA256 `e6b5e0886e05b971af7d3b63a4f8fa3688b0a341013e99b0e3c8756f742e3046`.

Full-matrix unitarity defects are light `7.9022e-15`, heavy `2.8947e-10`, reference `5.52e-37`. Mean matrix plus connection relative residuals are light `2.6618e-16`, heavy `2.8782e-11`, reference `8.6276e-19`. Occupied-plane transport, held initial SOURCE orbitals and prepared 14 coefficients agree exactly. Common-boost function comparisons are below `8.3e-17`.

Meaningful wrong branches remain visible:

- Omitting the full matter connection leaves pointwise coefficient norm squared `1.1200562638741074e17`.
- Freezing the heavy state gives full 54-mode Slater distance squared `2.940496072164754`. This includes a substantial common determinant phase, not merely particle-hole motion; that common phase can cancel from a SOURCE-relative comparison and is **not** asserted to explain the historical B closed-loop defect.
- Omitting the actual quintic field time connection loses `4.473957491475053e-6`; the retained common energy is `-.4725448866273807`.
- A genuine, explicitly counterfactual new SOURCE-function bank captured on the transported orbitals differs from the **frozen prepared SOURCE-function bank** by relative norm `.009854994742635754` for Source19 and `.004104118128182513` for SOFT. The counterfactual is never installed. It is wrong for this once-prepared SOURCE receipt, not a claim that evaluating a physical time-dependent SOURCE in a different construction is universally wrong. It resets neither the neutral state nor a ground/filter.

All these are **pointwise coefficient functions with common Gaussian/Jhalf divided out**, not whole-field L2 norms or whole-trajectory errors.

## Full CAR and physical current, before projection

`prepared_neutral_moving_frame_car.py` evaluates 112 actual light/reference modes by sparse determinant CAR and full Wick norms, including the actual neutral RIGHT action. Heavy 108-mode/54-occupied motion is treated separately. In its mean-Noether discriminator Source19 absolute LEFT and RIGHT particle-hole norms are both `.48543003382126615`, but they cancel in the SOURCE-relative commutator to `6.9796e-17` relative error. Projecting neutral RIGHT first misses a nonzero `.0437624559176636` correction. Therefore **absolute particle-hole leakage alone is not a causal explanation of the historical B endpoint defect**. The joined consumer additionally exercises the instantaneous `A` matrices rather than silently substituting `M`.

`prepared_neutral_moving_frame_current.py` evaluates all three stored literal photon templates, all 52 photon packets, both unrenormalized profiles and the actual full108 current leg before compression. The physical leg computed from the native clover legs is independently compared with its primitive coordinate-light operator reconstruction. Matter current and SOURCE matrices are pulled back by the actual full unitaries; full Wick current norm and outside particle-hole norm are retained. Maximum operator pullback error is `8.9043e-19`.

For the Source19 initial literal leg the full current norm squared is `1.0396350492591259e-7`, outside projected particle-hole norm squared `1.0239444222279353e-7`; the wrong projected-RIGHT correction is `4.070638452713087e-5`, wrong frozen-matter response `4.2579837480605264e-7`, correct SOURCE pullback response `8.48721958416301e-20`. Small projected defects do **not** establish full current closure. SOFT is compared using its actual tiny amplitude and scale-relative checks, never an absolute tolerance floor or normalized diagnostic masquerading as native mass. The current leaf's separately labelled first-order projected recapture surrogate is not the genuine SOURCE-bank counterfactual above.

## Source-bound language evidence and limitations

Executed and current core source SHA256 `b4c1b42db8fd24589925f9ec39e888fc5a68a2502dedc730f3c26984dea0e279`. Final direct **actual Pyright stdio RPC** received `publishDiagnostics: []`, 97 symbols, definition and references, six positive AST declarations; server exit 0 and source stable. Receipt `local://issue222-moving-matter-current-real-lsp.json`. The xdLSP device is unavailable; this is direct real language-server evidence, not a substitute lint claim. CAR and current sources retain their separate accepted source-bound executions and language receipts.

Unpaid: the time-ordered changing-pulse matter evolution and its physical field fluctuations; full Root/Q/all512/encoded CAR/Gauss kernel; nonprojected Hamiltonian/current particle-hole and infinite-Fock closure; native/legal mode mismatch and SOFT spatial restriction mass; physical closed-loop minus, Born and fundamental quantum-law/parameter derivation. The finite shared weak state is not promoted to a full ground. No old preparation, shared H, Galerkin G or closed-loop scientific run was rerun. Selection of the next physical construction depends on B's independent RAW SOURCE-relative projection discriminator, not on the now-withdrawn absolute-PH-only premise.

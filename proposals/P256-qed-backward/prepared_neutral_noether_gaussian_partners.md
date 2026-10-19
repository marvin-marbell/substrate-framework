# Literal native electromagnetic Noether partners

This new supplier leaves A17, its NPZ, its filter, the Weyl core, and the original SOURCE bank unchanged. It supplies functions, not a new preparation or a fitted angular pair. The shared Hamiltonian compiler owns integration into H/G; B owns the actual physical generator and once-prepared closed path; Transport owns the final scattering consumer. The unchanged full physical goal and governing plan remain in progress.

## Native law and exact finite Weyl expansion

For every anchored `(j,i)` clover and each orientation `(s,t)` in `{−1,+1}²`, the integer edge curl is the literal four-step word `(j,s),(i,t),(j,−s),(i,−t)`. Reverse steps use the preceding native edge with minus sign, including seams. Integer vectors are merged **only when exactly identical or opposite**, with the sine sign changed for opposite vectors. N3 has 81 such unoriented vectors. Each output edge preserves both endpoint clovers and the original average of its endpoint velocities. Consequently

\[
Y_A(y)=\sum_r u_r\sin(\ell_r\cdot y),\qquad
\ell_r=C_r\,\mathrm{legs}_{52}.
\]

No `sin(x) → x`, continuum L-pair, or mean-field-only rotation is used. The physical guard has origin `(1.5,1.5,1.5)`, Rin1/Rout1.5 and the actual fixed Ref site `(0,1,1)`. The SOURCE19 fold at `(1,1,1)` is distinct.

The exported coefficients are

\[
Y_r=\sqrt{\omega/(2U_E)}\,e^T u_r,\quad
h_r=N\operatorname{mean}(u_r),\quad
\lambda_r=-L^+D^Tu_r-[-L^+D^Tu_r]_R.
\]

`D` here has target minus source convention. The Coulomb coefficient is anchored at Ref, and `rho lambda` is inserted once. Harmonics are `theta=N mean(A)`, not a second independent copy of A. The original law remains electric `UE Pi²`, magnetic `(K/2)curl²`, `c²=2UEK`, and Coulomb `UE rho Green rho` evaluated before weak projection.

The analytic divergence and even Gaussian factor are

\[
\operatorname{div}Y=\sum_r (Y_r\cdot\ell_r)\cos(\ell_r\cdot y),\qquad
\gamma(y)=iY(y)\cdot y-\frac i2\operatorname{div}Y(y).
\]

All 52 coordinates contribute to Fourier Gaussian moments before projection. `FiniteGaussianFunction` stores finite polynomial/Weyl sums. The scalar library is hard15, gamma, then 81 sine functions. Gamma is even, sines odd. The sine Gram uses `exp(-(l²+m²)/4)*sinh(l·m/2)` to avoid catastrophic subtraction of nearly equal exponentials. Gamma moments use 60-digit Wick/Fourier summation on the actual native input coefficients; nonzero small norms are neither clipped nor suppressed. Native Weyl functions have infinite Fock tails. Appending the gamma function gives exact function-coordinate recovery, not total-photon≤2 dynamical closure.

## Strong SOURCE partners, not compressed commutator fiction

The SOURCE functions use the immutable initial light/Ref fibre. At fixed Phi/theta, let `psi_b` be **the actual frozen c14** contracted with A17's original 14 neutral columns. No sea/refilter/ref state is recomputed at each coordinate. The common odd function is

\[
k_b(y,\theta)=\sum_r\sin(\ell_r\cdot y)
[\langle\rho\lambda_r\rangle_b+i h_r\cdot\theta],
\]

where the harmonic term includes the derivative of the original theta Gaussian. The charged functions separate `gamma S psi`, `S k_b psi`, and the true relative odd `[G_EM,S] psi`.

For PLUS marks the relative scalar is `lambda(Birth)−lambda(Ref)−h·(Birth−Ref)/N`. MINUS uses its negative and the literal particle/hole mark map. The harmonic sign follows the original PLUS source phase `exp(−i(Birth−Ref)·theta/N)` and its conjugate MINUS phase. All four original unit words `I,Gamma23,Gamma31,Gamma12`, both banks and all Ref modes remain explicit; physical raw beta combines words linearly on PLUS and conjugate-linearly on MINUS. Global2 is already in the original producer exactly once. No profile or SOURCE normalization appears in the generated functions.

The strong CAR comparison retains the omitted neutral RIGHT:

\[
P_cQ_cS-S P_nQ_n-\mathrm{RIGHT}_{outside}=[Q,S]_{strong}.
\]

It does not assert the wrong `projectedQc S−S projectedQn` equality. Neutral/charged PH and full Q norms are separate debts, not erased by this exact relative identity. Four-word spin recovery belongs to the frozen quartet supplier; this EM generator acts by the same scalar on each word and does not substitute the old mixed-scalar spin basis.

## Public interface

- `NoetherGaussianBank(original_model, velocity)` owns immutable-chart EM coefficients.
- `.coefficients`: phases `[81,52]`, coexact `[81,52]`, harmonic `[81,3]`, lambda_C `[81,27]`.
- `.evaluate(y52)`: raw EM flow, projected coexact flow, divergence, gamma, harmonic and anchored lambda_C.
- `.gaussian_functions()`: the full 97 scalar functions, not a projection.
- `.scalar_values(y52)`: factors divided by the common EM Gaussian plus their 52-coordinate jets.
- `.scalar_gram()`: exact Gaussian Gram and base-cross kernels in this common scalar chart.
- `.conditional_coefficients(original_sample, source_bank, c14)`: neutral `[2,4,97]` and charged `[448,3*T*8,97]` tensors. Charged columns are category (even/commonodd/relativeodd), template, word, bank. With exactly SOURCE19 plus distinct SOFT, `T=2` gives48. Duplicate SOURCE19 forward bank is not required or created.
- `.generated_values(...)`: neutral_extra `[2,4]`, charged_extra `[448,48]` for the two-profile shared model. Values divide out the SAME common full Gaussian/Jhalf as A17.
- `.generated_directional_values(..., normal, harmonic, coexact)`: derivative includes **all** normal/coexact/harmonic Gaussian factors, Haar half-density and compact cutoff; do not add them again. It reports native refined-FD error estimates rather than claiming certified analytic Phi/source jets.
- `.conditional_kernels(...)`: exact EM conditional full Gram, with Phi/theta integration remaining the compiler's finite quadrature.
- `.generated_source_map(source_bank,beta,selected_profile)`: the exact sector-specific SOURCE coefficient map on all four new neutral functions, with raw beta/beta conjugate and no whitening.
- `.scalar_creator_kernels(packet52)`: exact conditional cross `<f|a†f>` and full norm `<a†f|a†f>` matrices `[97,97]`, using parity, high-precision gamma moments and stable CCR sine norms. Projecting these matrices is the compiler's responsibility; outside norm must remain explicit.
- `function_kernel(left,right,displacement52)` accepts arbitrary finite Gaussian sums and all52 Weyl insertions.
- `creator_function(function,complex_creator_packet52)` applies the actual full creator before projection. The polynomial-factor formula is `sqrt2(g·y)p−(g·grad p+i g·phase p)/sqrt2`; on the full wave this is `(g·y−g·grad)/sqrt2`. No conjugation of the creator packet is introduced. This agrees with the frozen HardFock core. Canonical SOURCE is coexact-independent, so its coexact creator commutator vanishes, not its harmonic/Coulomb terms.

## Evidence and boundaries

The new CLI requires exact root lemma, frozen state NPZ, immutable SOURCE19 descriptor, physical guard, B pure SOFT profile fixture and source archive paths. It runs generic nonzero all52 y plus harmonic theta and full normal q. It compares the RAW flow directly with `original_total_angular_transfer.primitive_flow`, projected coefficients with the native Green projector, analytic divergence with directional differences, finite-Weyl gamma values/jets, literal SOURCE harmonic derivatives, true relative CAR identity and neutral RIGHT. It exercises both distinct raw profiles, SOURCE on all four new neutral columns, conditional full function norms and the complete one-packet creator kernel. Adverse branches include mean-EM-zero discard, linear sine, reversed lambda, omitted harmonic, actual double rho insertion and omitted unanchored Ref subtraction.

The B SOFT fixture is an unchanged normalized compact momentum bump with centre2e−7/width1e−7; its native **coordinate restriction** is tiny, not renormalized, and exact native periodic Fourier projection is zero. Neither this supplier nor the shared compiler may replace it with A17's Gaussian profile. Native restriction loses almost all physical spatial mass; legal SOFT/clock and full physical closure remain open, not silently discharged.

Exact final runtime receipts, hashes, source archives and current-source real Pyright diagnostics are supplied in the companion local scientific capsule. Finite native quadrature/FD, full PH/Q restoration, actual coupled U2pi and same-current Born acceptance remain explicit downstream obligations.

## Observed scientific receipts

The full new native CLI completed naturally with exit0 in **506.4531844300218 s** (wrapper), on executed source `02c2cfaedd91ac434c3eedcaf0c034dc6364d59313010f1c7bc539dd58daac5b`, with empty stderr. Receipt: `local://issue222-noether-partners-final-wrapper.json`; science: `local://issue222-noether-partners-final-science.json`. It used all52 generic y, theta `[.31,−.22,.17]`, the full2430 normal chart, unchanged SOURCE19 and the raw B SOFT restriction. Neither the A17 filter nor any frozen supplier CLI was rerun.

| Observed quantity | Value |
|---|---:|
| Exactly merged native integer curls | 81 |
| RAW flow versus original primitive | 4.573862466262683e−18 |
| Coexact projection error | 1.2568993383378648e−15 |
| Anchored Coulomb sign error | 2.0270855829629244e−18 |
| Harmonic projection error | 5.463959996484826e−19 |
| Full gamma norm squared | .1067530955172436 |
| Gamma outside hard15 norm squared | .1067530955172436 |
| Actual augmented-Gram gamma recovery error | 0 |
| Gamma creator full / outside-hard15 norm squared | .12149595190942894 / .12149595190942894 |
| Strong relative CAR identity error, RIGHT retained | 3.09609273292211e−19 |
| Omitted neutral RIGHT control norm | 1.2734981796945938e−5 |
| SOURCE on all4 new neutral columns error | 7.631743241384645e−17 |
| Coexact SOURCE commutation error | 1.3456675451973264e−12 |
| Full creator SOURCE commutator error | 7.313325845750858e−26 |
| Literal harmonic SOURCE derivative error | 3.2996610150162895e−11 |
| Raw SOFT restriction norm | 1.4844654914318392e−11 |

All48 charged functions have positive conditional Gaussian norms; the tiny SOFT entries are retained rather than normalized into unit profiles. The actual wrong double-rho insertion differs by .00016488721996900081; linear-sine differs by 1.046078730797058e−8, well above the observed primitive error. Mean-EM-zero discard has coexact defect .46351039248753584. Reversed lambda, omitted harmonic and missing unanchored Ref also have positive discriminating defects in the saved receipt.

A separate new current-source derivative/Wick scenario completed exit0 in **20.280027230037376 s**, with empty stderr (`local://issue222-noether-partners-strong-norm-wrapper.json`). The all52 analytic versus native generated derivative errors were 1.237988655565296e−11 neutral and 1.5270699131524443e−11 charged; the combined normal/harmonic/coexact callback was exercised. SOURCE19 full rho-lambda SOURCE norm squared was 9.53348067545956e−8, splitting into projected7.714832917868185e−8 and positive PH outside1.8186477575913766e−8. SOFT retained full1.6688116882938984e−29 and PH outside3.316691276253809e−30. All four words are observed separately; these are true Wick-before-projection norms, **not** PH or fullQ closure claims. The initial orbital hash is exactly the frozen quartet supplier's `ebece7db297eacc8a6fd5df08ea454a92b3caf2de066f53ad1d23f1d40eb0a0f`.

After natural science exits, Pyright1.1.414 received actual publishDiagnostics with **371 symbols / zero diagnostics**, plus definition/references and a positive AST class query. Current typed source is `d97afb8add1d206ea3a4fe8cabc8e04ee2836350f5071e09fda7f07362da2407`. The only later code differences are eight identity `typing.cast` wrappers and their import. The exact whole-module runtime AST equals the executed source after removing only those identities (digest `0278f3108c596ca4117bc984aeb5aa36184b96d475ce7c59a32c96f05ce6e99d`); no scientific rerun or retroactive byte-identity claim is made. Original twelve typing diagnostics are archived, not hidden.

The compiler requested a temporary exact source02 restoration during its own in-flight scientific epoch. That restoration is explicitly recorded in `local://issue222-noether-partners-source-restore-epoch.json`; typed d97 is preserved in `local://issue222-noether-partners-typed-current-source.py`. It is **not** a claim that the file stayed byte-constant throughout that compiler run. After the compiler explicitly released the epoch, d97 was reapplied before its next extension CLI startup; current RPC again received zero diagnostics and the exact runtime AST bridge was refreshed. Release: `local://issue222-noether-partners-source-typing-release.json`. The compiler reported its first extension run exited1 at a final provenance assertion after reaching forms; its exact module delta was not retained, so this supplier does not attribute that failure or claim the earlier compiler run had byte-constant sources. No physical code, own scientific output, or frozen preparation was changed or rerun.

The main receipt's `loaded_sources_unchanged=false` is retained honestly: B changed its independently owned live module while this process ran. Every loaded Python source was archived **before** the new calculation, so executed provenance is exact. B's former coordinate packet used a distinct native c=.12547773800550577; only its RAW edge geometry was reusable. B rederived the same-A17 all55 packet with c=.125 (`local://issue222-native-noether-fourier-A17-operator-terms.json`, SHA `fd7337db4e6d30747529f7674f6dacc726c121d41287ca85f078eef530f2f6c8`). This supplier has always used the frozen A17 c=.125/UE2−20 legs. Legal/native cone and mode-ID gaps remain explicit.

The omitted-Ref branch also has a genuine **full conditional Gaussian SOURCE norm**, not only a nonzero lambda coefficient: `local://issue222-noether-partners-missing-Ref-gaussian-control.json` obtains `E[lambdaRawRef²]=4.7375113801083136e−9` and total all-profile/unit-word SOURCE defect norm squared `1.301203212672295e−8` (norm .00011407029467272778). This uses the corrected B same-A17 finite sine coefficients and the saved actual coexact-independent SOURCE norms. The mean-zero raw pseudoinverse coefficient satisfies `lambdaRawRef=−mean(lambda_anchored)`. The calculation uses exact sinh moments, no source/profile normalization, jitter or old scientific supplier rerun. Its separate pointwise generic-y counterpart is .00005394716905923728; these are explicitly different norm scopes.

The frozen state remains **the original `f(H14_MC) tau` coefficients**, vacuum-embedded in the enlarged basis, not `f(new_exact_EM_H) tau`. Analytic EM integration can change H's old MC vacuum block; the compiler reports that difference and must not refilter, reset normalization or calibrate the model to erase it. Phi/harmonic nodes and the full J/Haar/cometric/cross/vertical measure remain the integration owner's original chart. Root `P V_beta P=0` at fixed s0 is only a diagnostic: full Q-beta, all eight Root sectors, multipair and full Gauss-kernel evolution debts remain explicit. No Root scalar identity on full Q, legal-N SOFT state, whole native Fock closure or full physical closed-rotation/Born conclusion is supplied by these functions.

For semantic separation, `local://issue222-noether-partners-physical-Coulomb-RIGHT-science.json` independently observes the **physical Coulomb-only** CAR identity on both profiles/all four words at another generic all52 point: Source19 RIGHT norm1.867437876639838e−5 with maximum identity error7.625104836174926e−19; raw SOFT RIGHT2.159132956720067e−16 with maximum error1.0153446975120674e−29. Natural exit0, program8.23457482107915s. The main CLI's combined lambda-minus-harmonic-position algebra is a convenient strong SOURCE commutator identity; its harmonic-position matrix must **not** be interpreted as an added physical neutral matter generator. The actual common neutral function remains rho-lambda plus the harmonic Gaussian derivative, and the literal harmonic SOURCE derivative is tested separately.

# Full original Phi common-form component

This NEW component implements **pointwise stochastic full-original Phi kinetic integrands**, not global L2 norms, a complete native quantum evolution, or physical rotation closure. Frozen providers are imported without editing or executing their scientific CLIs. The parent owns global quadrature, full matter/Coulomb/scalar/EM terms, physical Noether matrices, moving connection and actual six-coefficient evolution.

## Representation and API

`operator_matrices(model, data, sample, direction, sector, physical_point=None, q_shift=None, h=None, trace_samples=4, seed=222107)` returns a dictionary whose `matrix` is a complex Hermitian 6×6 point matrix. `sample` MUST be the original shared `model.basis_at(qFoot, em55, point=footPoint)`; `sector` is `neutral`, `source19` or `SOFT`. It contracts exactly `neutral_state`, `source19_normalized_source_state` or `SOFT_normalized_source_state` from the immutable accepted cache; there is no refilter, state reset or new normalizer. `direction` defines the trial multipliers, NOT the kinetic trace.

With n=2430, u=direction/||direction|| and x=qFoot·u/sigma, the multipliers are

1. 1;
2. sqrt(2)x;
3. (2x²−1)/sqrt(2);
4. (2||qFoot||²/sigma²−n)/sqrt(2n);
5. sqrt(2)em[52];
6. (2||em[:52]||²−52)/sqrt(104).

The function retains every normal derivative of these multipliers. The two EM-only multipliers have zero Phi derivative, but still multiply the full held-wave Phi momentum and its gauge covariance.

At qPhysical=qFoot+q_shift, let A=sqrtMass·Jacobian(qPhysical). Generate independent raw2835 Rademacher vectors r with E[rrᵀ]=I, and solve z=A⁻¹r by LU. Thus E[zzᵀ]=(JᵀMass J)⁻¹, including all2430 normal,405 Gauss and their complete cross covariance. This is not a Y0-only directional lower bound or independent normal/Gauss sampling.

For each z=(dq,gauge), the held functional derivative is shared32's original directional basis contracted with the original coefficient vector. That producer includes qFoot Gaussian, **ALL55** EM Gaussian, original harmonic packet, cutoffFoot and foot Haar half-density. Add exactly

Dphysical v = Dfoot v + 0.5(DlogrhoFoot−DlogrhoPhysical)v.

For each multiplier f, D(fv)=fDphysical v+(Df)v. The actual gauge operator contains occupied heavy54, actual light108 and Ref4 first/second Wick products before compression. The returned probe matrix is

chi/2 · cutoffFoot² · [(−iD+Qfirst)†(−iD+Qfirst) + V†Qoutside_second V].

`outside_second` restores PH/multipair covariance absent from the square of the compressed first matrix. No scalar-phase cancellation, projected-before-Wick substitution or coexact52-only Gaussian factor is assumed. A supplied distinct physical_point requires q_shift so the physical density jet is evaluated at the correct coordinate.

**Weight convention:** every returned kinetic matrix ALREADY includes cutoffFoot². The parent samples the original qFoot/em55 Gaussian law; do not multiply cutoff² again. The Gaussian and physical Haar density have been factored from the wave. Initial six-coefficient vector (1,0,0,0,0,0) preserves the exact frozen original wave. Neither the matrix nor its stochastic trace is called global L2.

## Observed numerical controls

The NEW CLI loaded accepted cache SHA10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3 and manifest SHA90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a. It used the original `source_point` q0/em55, trace budgets2 and4 with seed222107, and shared source FD steps sigma/32 and sigma/64. Natural exit0, wrapper wall649.639607351s, empty stderr; science file equals actual stdout byte-for-byte. Executed/current source SHA34aab644ece5cef4a951e5f50f8ce8f601087aff77364ff4eb9908fd22661eef.

| Sector | Offdiagonal Frobenius norm, budget4 | Trace standard-error Frobenius | FD matrix difference norm | Missing raw PH covariance |
|---|---:|---:|---:|---:|
| neutral | 740858.102702 | 1994104.788554 | 0 | 33.3725292608 |
| source19 | 732822.406746 | 1972469.699383 | 3.4480583e−6 | 32.7308124714 |
| SOFT | 729395.119314 | 1963245.979238 | 4.1328661e−6 | 32.6054249043 |

These small trace budgets have **large stochastic uncertainty**; nontrivial offdiagonal entries do not establish converged global dynamics. The standard error is an observed sample statistic, not a rigorous confidence interval. `fd_difference` is signed/complex; it is not clipped or converted to a convergence theorem.

An independent cross-Wick control constructs full heavy/light/Ref one-body exponentials and occupied determinant/cofactor overlap kernels BEFORE compression. Mixed differences of exp(i epsilon Q)(v+epsilon Dv) recover the full gauge-plus-normal derivative norm; the normal path is linearized to isolate the gauge/Wick check, rather than claim a second-order SOURCE jet. The original `AmbientWeakState.positive_direction` is independently called unbound on the shared model, retaining its virtual shared held-bank derivative. Signed analytic-minus-original point residuals are neutral4.55e−13, source19 1.36e−12, SOFT3.41e−13. At the two independent cross steps, signed cross-minus-original residuals are neutral[−8.30e−8,−1.71e−7], source19[−1.42e−7,−1.26e−7], SOFT[−2.44e−7,−1.19e−6]. Roundoff deterioration at the smaller SOFT step is retained, not hidden. Dropping outside_second loses the nonzero PH covariance shown above.

A distinct physical point qFoot+0.05sigma·u exercised physical LU and half-density correction, with corrections −3.79719509e−8 and −8.19504226e−8 for its two probes. This is a pointwise translation control, not evolution.

## Reproducibility and tooling

CLI: selected project Python `prepared_neutral_full_phi_commonform.py --cache-npz <original shared NPZ> --cache-manifest <original shared manifest> --output <science.json> --trace-budgets 2 4`. The exact absolute command/cwd/exit/timing/stdout/stderr and private artifacts are in `local://issue222-full-phi-run-receipt.json`. Actual loaded sources are unchanged across execution; the CLI capsule archives their bytes plus cache/manifest. Auxiliary guard bytes are preserved in the final supplemental capsule with explicit post-run-capture provenance; no uninterrupted auxiliary-file seal is claimed.

Real Pyright JSON-RPC preflight exercised original positive_direction references/symbols/definition/received diagnostics before editing. Stable post-science RPC exercised this actual source and received empty diagnostics. Tooling gap: mounted xd LSP is unavailable, so the existing actual user-local pyright-langserver was used directly in the selected root/Python environment. Positive structural ast-grep matched original gauge_operators calls before editing and this source's LU solves after science. No build, formatter, linter, old scientific CLI or project-wide test suite was run.

Remaining debts: stochastic fullraw trace convergence, rigorous source finite-step error, parent global Gaussian/cutoff quadrature, full H/G/-iC evolution, whole-H unrepresented-mode/Fock leakage, legal-scale SOURCE/Born and physical closed rotation. The component discharges none by relabeling point observations as global proofs.

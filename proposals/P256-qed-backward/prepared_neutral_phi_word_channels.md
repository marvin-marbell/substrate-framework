# Full original Phi operator with independent SOURCE word channels

This NEW component generalizes the immutable `prepared_neutral_full_phi_commonform.py` law (SHA34aab644…) to coefficient **matrices**. It computes pointwise full-original Phi bilinear forms, not a global L2 norm, the complete Hamiltonian, quantum propagation, or physical rotation closure. All frozen helpers and the original accepted state/cache remain unchanged. The parent owns global quadrature, matter/Coulomb/scalar/EM terms, physical Noether and moving connection matrices, and actual coefficient evolution.

## Shared coefficient and operator ABI

`coefficient_channels(data, sector, profile='source19') -> (C, metadata)` uses only the original accepted cache:

- Neutral C214×2: exact `neutral_state`, then `neutral_charge_labels * neutral_state`.
- Charged C1728×8: exact `profile_normalized_source_state`, followed by +word1,+word2,+word3,−word0,−word1,−word2,−word3.
- Each independent word direction is `profile_word_maps[word] @ ((neutral_charge_labels == charge) * neutral_state) / Z`. Z is **retrieved from the same cached global Gram** as `sqrt(raw_source_state† charged_Gram raw_source_state)`, with exact agreement of raw/Z and the cached normalized state. This is not a new physical normalization or a per-word reset.
- The coherent column replaces +word0 because the actual raw quartet coefficient beta0=1 is nonzero. The omitted direction is recoverable by subtracting the remaining original beta/conjugate-beta contributions. Raw quartet coefficients enter the coherent initial column only; arbitrary later coefficients retain all four word directions per charge.

The separately executed ABI control observes exact first-column equality, zero opposite-charge support for all seven unit-word columns, strictly positive cached global word Gram eigenvalues, and +word0 reconstruction errors 7.24e−17 (SOURCE19) / 1.59e−16 (SOFT). Retrieved global Z values are 1.0045633644689442 / 1.359790437008373e−11 respectively. Nothing is individually normalized or whitened.

`operator_matrices_channels(model, sample, direction, sector, coefficient_matrix, physical_point=None, q_shift=None, h=None, trace_samples=4, seed=222107, return_probes=False)` accepts sector `neutral` or `charged` and general C214×n / C1728×n. The physical wave columns are `sample[sector] @ C`. Each column is multiplied by the **existing fullPhi.trial_multipliers**, then flattened CHANNEL-major followed by six modes. Thus the selected neutral/charged spans return 12×12 / 48×48 matrices, including every cross-channel bilinear.

Initial coefficients `(1,0,…,0)` reconstruct the original once-prepared state exactly. `direction` defines the six trial multipliers only; it is not a surrogate kinetic trace.

## Full metric trace and Wick form

At the physical point qPhysical=qFoot+s, factor A=`sqrtMass * physicalJacobian` and solve z=A⁻¹r for independent raw2835 Rademacher r. E[zzᵀ] is the entire physical cometric, retaining **2430 normal +405 Gauss** coordinates and all normal/Gauss cross covariance. A single matrix `directional_basis(sample,dq,step)[sector] @ C` is evaluated per probe/finite step: the physics derivative is not rerun for each word.

The shared producer's derivative includes the original normal Gaussian, all55 EM Gaussian factors including the harmonic packet, cutoff and foot half-density. For translated points, add exactly `0.5*(DlogrhoFoot−DlogrhoPhysical)*wave`. Held bank/source functions and trial multipliers are evaluated at the foot; the metric/density are evaluated at the physical point. No matter/scalar/body evolution is preapplied.

For the complete trial-column matrix V and derivative DV, the probe form is

`(chi/2) * cutoffFoot² * [(-i DV + Qfirst V)†(-i DV + Qfirst V) + V†Qoutside_second V]`.

`Qfirst` and `Qoutside_second` are the original shared gauge operators, including actual occupied heavy54, light108 and Ref4, with full Wick products before compression. The outside term acts on **all** column pairs. It is not replaced by the square of a compressed first matrix. Hermitian symmetrization removes anti-Hermitian floating roundoff only; no signed residual, spread or eigenvalue is clipped or fitted.

Returns include both finite-step mean matrices, their signed difference, both per-probe matrix arrays, outside-second means, elementwise observed standard error and its Frobenius norm, density corrections, seed/probe norms and coverage metadata. `return_probes=True` additionally returns actual raw probes and inverse-Jacobian directions for parent-side optional control variates. This implementation uses **no fitted/control-variate subtraction**: uncertainty is from the actual raw full-cometric probes. The two steps are h and h/2, default sigma/32 and sigma/64.

**Every returned kinetic matrix already includes cutoffFoot².** GaussianFoot and physical Haar half-density are factored out. The parent must not multiply cutoff² again. These are point integrands, not global quantum matrices.

## Executed scientific discriminators

The actual NEW CLI completed with natural exit0, empty stderr and wrapper wall301.315273965s. Output equals captured stdout byte-for-byte. It used the original `source_point`, two probes/step, seed222107, original NPZ SHA10d61adb… and manifest SHA90a6d7f3… . Executed/current Python source SHA24727dd7d13192fc9074d495667e13a0c24904626ccdc2703c77117c6ab84b4d.

| Sector | Matrix | Cross-channel K Frobenius | Actual gauge action outside coherent tau / total | Raw observed trace SE Frobenius | FD difference norm |
|---|---:|---:|---:|---:|---:|
| neutral | 12×12 | 9.484495 | 0.560749 | 892786.049 | 0 |
| SOURCE19 | 48×48 | 4540563.670 | 0.800439 | 924639.888 | 1.16624e−5 |
| SOFT | 48×48 | 4160927.090 | 0.886452 | 893339.997 | 1.14726e−5 |

The action control lifts the **actual gauge-first bilinear** to the entire 48/12 trial span. It solves its point Gram projection and removes the physical coherent vector, rather than counting nonzero offdiagonals in a nonorthogonal coefficient basis. Actual orthogonal gauge-action norms are .998817891 (SOURCE19) / .951476032 (SOFT). This rejects freezing the coherent raw-tau direction. The point Gram has rank8/2 because six scalar multipliers at one point are dependent; no global rank, global Hamiltonian action or evolution is inferred.

An independent full cross-Slater/cofactor kernel, with heavy occupied determinants and light/Ref CAR minors, checks the entire cross-word normal+Gauss momentum matrix. It evaluates mixed overlaps of `exp(i epsilon Q)*(wave ± epsilon derivative)` at two independent epsilon values. The normal path is linearized to isolate the gauge/Wick second-product check; this is not a second-order SOURCE-jet theorem. Two genuinely complex linear word states are also contracted, retaining all interference. Largest relative cross-matrix discrepancy is 3.49e−9; actual sampled full108/Ref4 determinant-minor error is at most 4.85e−14.

The expected wrong **projected-before-Wick** branch loses strictly positive actual PH covariance in both complex states:

- neutral: [109.735756, 52.705046];
- SOURCE19: [176.790247, 44.296158];
- SOFT: [172.073524, 45.181564].

A separate shifted neutral point qFoot+.05sigma·unitDirection exercises physical LU and both density jets; corrections are −3.79720e−8 / −8.19504e−8. None of these point checks is global L2 evidence. The large two-probe uncertainty is explicitly retained, not treated as convergence or a rigorous confidence interval.

## Receipts, archives and language tooling

Exact argv/cwd/runtime/natural exit/streams are in `local://issue222-phi-wordchannels-run-receipt.json`; science/stdout SHAf67b090d07af934b63275350afceb6526fd940f4ea485454cd3ff13cefd9bf09. The CLI captures108 actually imported scientific source files before probing plus original NPZ/manifest and **guard bytes before their producer read**, checks all bytes unchanged afterward, and archives these exact bytes. Capsule SHAa5b1a26d203b2245b0a69dd502a531f70b25a66582fcff987fd16d20f32e0ef1. Explicit scientific metadata only: no serialized process environment or credentials. Receipts/capsules are private0600.

Additional evidence: `local://issue222-phi-wordchannels-abi-receipt.json` and `local://issue222-phi-wordchannels-action-receipt.json`. The final receipt binds this document and the source/capsule/tooling hashes. The shared governing plan remains issue-222-fullfield-closedrotation-continuation-20261006, assignment668c7d8e…, pinned general89cd9d…; only the canonical owner publishes plan changes.

Real Pyright1.1.414 JSON-RPC before editing exercised original symbols/definition/references and received empty diagnostics. Stable current-source post-science RPC received empty diagnostics and146 top-level symbols, source hash bound above. Mounted xd LSP is unavailable, so the actual user-local Pyright server was launched directly in the selected root with the project Python. AST-grep0.45.3 positively matched original gauge calls before editing and both current matrix directional-basis calls after science. No build, formatter, linter, project-wide suite or frozen scientific CLI was run.

Remaining debts: finite source-jet rigorous error, stochastic trace convergence, parent global Gaussian/cutoff quadrature and actual H/G/C propagation, Galerkin/unrepresented/Fock/domain closure, LEGAL SOURCE/Born/physical closed rotation, and fundamental quantum-law/parameter closure. Campaign remains open.

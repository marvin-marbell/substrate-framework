# Original Phi directional form on the once-seeded paired orbit

## Scope and unchanged law

`prepared_neutral_changing_phi_kinetic.py` consumes the accepted shared photon cache, without a new sea, ground, bank, filter, SOURCE normalizer, Hamiltonian compilation, old scientific CLI or loop run. The coherent neutral coefficients have dimension 214. The two charged inputs are the original, globally normalized `source19_normalized_source_state` and `SOFT_normalized_source_state`, each dimension 1728. Both charge sectors and all original reference modes remain present.

The actual native short interval is clock 566, t0=209.42, t1=209.451, eight CF4 steps. Every base/varied family calls the actual `coupled_interval`: changing physical RAW Phi/links, all52 coexact plus three harmonic EM coordinates, light108 with coefficient 1, heavy108 with coefficient −Lambda, and physical Ref4 Hamiltonian zero. Native parameters are Lambda=16777216, chi=1/4096 and sigma=1/8192. The scalar multiplication phase includes literal quartic, alignment and colour magnetic potential. `excluded_duplicate_massive_sea` is only the convenience fermion expectation already included by H_D, **not** an independent bosonic counterterm.

This is one original positive Phi/colour kinetic directional contribution on this endpoint family, not full cometric trace, global L2 energy, whole-H evolution, a full pulse, physical 2pi closure or a Born probability.

## Full cross-Slater public ABI

```python
full_cross_kernels(model, units_left, units_right)  # {'neutral': (2,2), 'charged': (448,448)}
full_cross_overlap(model, units_left, units_right, left, right, sector)  # complex
```

Use the **original once-seeded model**, PHYSICAL RAW light108/heavy108/reference4 unitaries, and vectors expressed in its fixed physical fibre. A static LAB function on the left uses identity RAW unitaries. Scalar phases/densities are carried by the supplied coefficient vectors, not inserted again by this kernel. The API supports full fixed-LAB pullback: evaluate the original LAB function at the actual target, then contract its fixed-fibre vector against the advected vector. It neither re-evaluates a detector at the seed nor recaptures an endpoint bank. The consumer must separately supply its inverse chart/density factors and full cross-template Slater contraction; this module does not publish a detector response or Born claim.

Let L=O†U_left,L†U_right,L O in the original complete 108-orbital basis, R=F_ref†U_left,ref†U_right,ref F_ref, and

    d_H = det(H_occ† U_left,H† U_right,H H_occ),  occupied heavy count = 54.

Every neutral and charged entry is multiplied by d_H. For the 54-light sea, write L in occupied/empty blocks A,B,C,D. Define `cof(A)[i,j]=(-1)^(i+j) det(A with row i,column j deleted)`. Then the particle block is

    P = det(A) D − C cof(A)^T B.

When A is invertible this is `det(A)*(D−C A^−1 B)`; the implemented polynomial identity remains meaningful when A is singular and uses **no inverse**. Cofactors are evaluated by an SVD product-of-all-other-singular-values identity, with the determinant orientation retained. No inverse of Phi, Gram(Phi), or occupied cross matrix is used.

The neutral kernel is `d_H*diag(det(L54)*det(R), det(L58))`. Off-diagonal neutral overlaps vanish because the Ref occupations 4 and 0 differ. Charged+ ordering is Ref-hole outer, light-empty inner; its block is `d_H*kron(cof(R),P)`. Charged− ordering is Ref-particle outer, light-hole inner; its block is `d_H*kron(R,cof(L58))`. The CAR hole signs are precisely the cofactor signs. The two charged families are orthogonal because their Ref occupations are 3 and 1. All determinants come from the full physical propagated orbitals **before** compression; PH/multipair overlap contributions are therefore not replaced by exponentiating a projected weak generator.

## Full held functions, frame cancellation and tangent

Use the full physical mean direction D=Y0/||Y0|| and seed points q0 and q0±hD. Original source functions are evaluated at each INITIAL varied footpoint. The factored Gaussian/Haar/cutoff must be restored relative to the base:

    f(qvar)/f(qbase) = exp[−(qvar²−qbase²)/(2 sigma²)]
                      sqrt(rho(qbase)/rho(qvar)) cutoff(qvar)/cutoff(qbase).

No common initial normalization is refitted; it cancels only as the chosen common unit. All relative density, cutoff, field-dependent source functions, body propagation, heavy Slater phase, literal scalar phase and characteristic half-density are retained. Crossself kernels preserve actual nonunitarity roundoff: neither self norm is replaced by 1 and no polar repair, norm floor, clipping, phase fitting or phase subtraction is applied to the physical result.

The q-dependent section-gauge and W derivatives are not omitted: their **full products cancel exactly in the physical representation**. Every one of the five families verifies

    blockGauge(g) U_section,light/heavy = U_RAW,light/heavy,
    g_ref U_section,ref = U_RAW,ref,
    boost_inverse_coefficient * carrier_sample = physical_sample.

Since these are pointwise smooth product identities, differentiating the products gives the same physical derivative, including cancellation of both derivative terms. The finite-family numerical witnesses have maximum matrix error 1.546e−15 and coefficient-function relative error 1.267e−16. The receipt records full-state crossself norms, initial factors, nonzero half-density, scalar integral/phase and absolute occupied-heavy54 overlap with the original seed for every family. W is a representation, not an extra physical drive.

The physical endpoint tangent is formed directly from the RAW Phi and link endpoint central difference, not a normal-coordinate surrogate. Phi uses the original native Cartesian basis. Links use the **left** Haar generator `−i*(dU/ds)*U_base†`. Their native concatenation has all2835 components and uses exactly `model.chart.mass` (Phi mass 1, colour mass 1/.6).

For the original positive kinetic form, Cauchy–Schwarz gives the exact directional inequality

    K_Phi(point) >= chi/2 * ||dPsi/ds||² / ||dRAW_end/ds||_mass².

The executable estimates this contribution with full cross overlaps:

    ||dPsi/ds||² ≈ (N_minus + N_plus − 2 Re overlap_minus_plus)/(4 h²).

The inequality is exact for the derivative; the displayed finite-h values are numerical estimates of its right side, **not rigorously remainder-enclosed bounds**. There is no integration over global field space and no claim that one direction reconstructs the full kinetic trace.

## Actual results and adverse phase control

| Original coherent input | h=sigma/32 | h=sigma/64 | Wrong scalar phase dropped, sigma/64 |
|---|---:|---:|---:|
| Neutral | 4836.77940068 | 4840.98596899 | 172.708138868 |
| SOURCE19, fixed global normalization | 4784.30147516 | 4788.46240345 | 170.833925891 |
| SOFT, fixed global normalization | 4761.93429796 | 4766.07577312 | 170.035691369 |

Refinement increases all three estimates by approximately 0.0869704%. RAW endpoint tangent mass norm squared is 0.9999999999985175 then 1.0000000000034779. Subtraction conditions are about1695.55 and6784.06. Thus floating-point cancellation is exposed rather than concealed. Original scalar integral is about2.63047584e6; plus-minus scalar differences are −.0476024160162 then −.0238012084737. Full cross phases are about.04768785 then .02384393, well below phase aliasing. At these h there is no observed alias requiring phase fitting or h replacement. The wrong-drop control removes only the literal scalar phase and leaves heavy Slater transport intact; it changes the kinetic estimate by about96.43%. Heavy phase is retained absolutely, but this slice does **not** demonstrate a huge heavy/scalar counterbalance: the observed control instead shows scalar dominance for this chosen direction and short interval.

Independent ten-mode full Fock CAR evolution uses an actual nontrivial generator: light PH generator norm .66897232 and Ref off-diagonal norm1.30329306. Exact CAR versus cross kernel error is6.023e−15. The forbidden projection-before-Slater exponential differs by.66848941. Actual108/Ref4 occupied-minor Wick controls sample72 cross entries at each h with Ref off-diagonal norm.964912; errors4.835e−15 and5.161e−15, with nonzero actual PH cross norms9.364e−8 and4.682e−8. A current-source ABI smoke additionally uses strong actual108 PH cross norm2.992823 and Ref cross norm.240800; its independent occupied-minor error is6.657e−20. A rank-deficient occupied matrix verifies the singular cofactor identity to6.997e−15. None of these is an identity-only control.

## Exact execution, source and language evidence

Run from `/home/administrator/substrate-222/proposals/P256-qed-backward` with `/home/administrator/substrate-framework/.venv/bin/python`:

```sh
python prepared_neutral_changing_phi_kinetic.py \
  --cache-npz <local>/issue222-shared-photon-state.npz \
  --cache-manifest <local>/issue222-shared-photon-state-manifest.json \
  --output <local>/issue222-phi-kinetic-repaired-science.json \
  --h-divisors 32 64
```

`<local>` is the receipt directory recorded as absolute paths in `local://issue222-phi-kinetic-repaired-execution.json`; that receipt is the exact argv/cwd/runtime authority. Actual Python3.12.3 scientific run naturally exited0 in561.947865784 wrapper seconds (560.625985656 scientific seconds), empty stderr, and JSON stdout equals the scientific file. All190 snapshotted project Python/cache/manifest inputs stayed unchanged, including all106 imported project sources. Scientific source SHA256 is `e1c4544980fec3f2cbc9754ecfbd7c03ba4c606b6d509a352040b871e79cc2b5`; exact science SHA256 is `bd9fa04ae455448580581ea9eaaa25c2ffaba384094ccbbfe0c84f4fc34e0ced`. Pre-run archive `local://issue222-phi-kinetic-repaired-sources.zip` SHA256 is `98e62b6dcf8f42c0527305299058faf0ea9eea384ca0b28da7bfabac910d6c75`.

Accepted cache SHA256 is `10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3`; manifest SHA256 is `90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a`. A first new-consumer run naturally failed after335.889179389s because this consumer passed the original `coordinates(fields,basis)` arguments in the wrong order. Its source/archive/error/receipt remain under `local://issue222-phi-kinetic-first-*`; the corrected new scientific run above is the positive execution. No frozen supplier was re-executed.

After science, real Pyright1.1.414 stdio RPC received six optional-path diagnostics; these were repaired **only** by adding three runtime-identity `typing.cast(str, module.__file__)` calls and the cast import. Current source SHA256 is `5d04b03a91ca629296c5a6eb2ad948fce1d39412f76b052ce89866a05cf472d5`. `local://issue222-phi-kinetic-typing-AST-bridge.json` proves the entire scientific/current ASTs identical after removing only those casts/import. The archive is not retagged as current bytes. `local://issue222-phi-kinetic-current-ABI-smoke.json` exercises the current public overlap code independently. Final real RPC `local://issue222-phi-kinetic-post-real-lsp.json` reports188 symbols, definition/references, received `publishDiagnostics=[]`, natural server exit0 and stable current source. Python structural AST probes positively matched the parent `coupled_interval` call before editing and four current `full_cross_kernels` call sites afterward; standard Python AST finds12 declarations. The direct real server substitutes for unavailable xdLSP; Markdown has no claimed language-server verification.

Unpaid: finite-h and CF4 remainder enclosure, full original Phi trace/global L2, quantum EM and Coulomb multipair/whole-H propagation, full pulse, physical2pi, Born, Root/Gauss/legal-SOFT and fundamental quantum-law/parameter closure.

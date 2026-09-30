# P253/0162 independent scientific scope review

Reviewer: independent `EulerTheoremReview` agent (read-only; not the derivation author). Target: `0162/derivation.md` frozen author SHA-256 `0039467ac34e3075a9034509dcf18a5f48a1ae06d15f3decf4838b4a1805f83d`, `result.yaml` SHA-256 `3c6222d798253e8ef4fd2635abed9b86355dddb25831ddceb0a82d08adba4cb8`, `README.md` SHA-256 `6da8b4e9d5b984ceb368fe9cd672aac73a838cff2bd1ccdd3f9efb1bc3bfb95a` at review start. This is independent agent review of the exact scoped mathematics, **not** accepted-claim or GitHub approval authority.

## Substantive pass: one blocking proof-domain finding

The reviewer independently checked the reviewed 0066/0073 source foliation, 0062 positive-core centralizer and DA/Hodge formulas, actual nonzero-`n` cylindrical curl and solid-harmonic multipole identities, physical dimensions, full 3D Floquet cases, and generator collar continuation. T1 survived for one fixed sufficiently thin Cao positive core: regular meridional motion has unit normal return multipliers, a stalled regular level has a unipotent return block even with varying toroidal drift, and the sole negative-definite center has imaginary/zero meridional spectrum. Same-leaf smooth conjugacy preserves these multipliers. The PDE saddle germ is explicitly not a global Euler countercarrier.

For T2, the frozen derivation restricted **all** `q` to one fixed disk `U` at line 42 while inferring the distributional equation (7) across the **whole** positive core and using the whole-support weighted energy identity (8). Vanishing moments on one `U` yields (7) only on `U`; it cannot justify (8) as written. The reviewer returned `incorrect` for that frozen proof with one bounded repair: quantify the infinite-rank test class over `C_c^infinity(K_0^+)`, then choose one disk `U` only for the explicit pole and collar seed. The physical identities themselves survived; no continuum resonance, eigenmode, physical `V_*`, branch, persistence or particle conclusion was credited.

## Correction-only check

The author changed only `derivation.md` lines 34 and 42, final corrected SHA-256 `558dcfdafd85d22c7d9423e98fcc694238d3bd24f317f13914170019240b1fb6` at correction request:

- Line 42 now quantifies `q` over **all** compact positive-core meridional tests, each individually interior-supported. Equation (7) therefore holds distributionally across the whole core and the vanishing-`zeta` boundary term licenses (8). The fixed small disk is reserved for the nonzero-pole/collar witness. Reviewer correction verdict: **PASS; sole blocking finding resolved**.
- Line 34 replaces a potentially unproved `F'(zeta_c)` extension at the center with the `C^2` tangency argument `HL+L^T H=0`, `H=D^2 zeta(center)<0`, so the meridional linearization is similar to a skew matrix. Reviewer correction verdict: **PASS**.

The correction check did not re-review unchanged mathematics or confer accepted-claim/release authority. At the exact reviewed scope, T1 rules out the 0161 model partition-2 *positive-core smooth hyperbolic-orbit route* on this selected thin carrier; T2 rules out a finite-dimensional *exterior-field representation* of the full smooth nonzero-`n` DA class and proves existence of a deep-core seed with exact generator feedback into every positive-core boundary collar. A finite-dimensional resonant Grushin cokernel after a complete whole-space solve is not excluded. The continuum adjoint, `V_*`, exact rotating branch, nonlinear physical persistence and #203 remain open.

# Full ambient Phi/colour Gauss chart and original positive form

This new component supplies finite-coordinate geometry and operator coefficients for the **same original** `ImprovedPhiLaw`, not a curve Hamiltonian, normal-ground insertion, evolved state, or completed full-field dynamics. It leaves sibling H/state/SOURCE and Noether files under their owners. The governing scope is `prepared-neutral-full-field` under issue-222-fullfield-closedrotation-continuation-20261006, pinned revision `7954ba4c18ed7d90fdc3e6cbd3bf75c3dc80d568cae4781d73dadd61e041a718`; campaign and integration remain open.

## Coordinates and original mass

At each site retain every one of the 60 Cartesian Phi coefficients (four traceless Hermitian matrices) and all 45 independent colour-link coefficients (three SU4 links). The raw ordering is site-major `[Phi60, link-x15, link-y15, link-z15]`. The original mass diagonal is

\[
 M_x=\operatorname{diag}(1_{60},(1/.6)1_{45}).
\]

Consequently the original kinetic cometric is `chi` for Phi and `.6 chi` for colour: the positive Hamiltonian coefficients are `chi/2` and `.3 chi`, respectively. Confusing the latter coefficient with the colour cometric changes the law.

`AmbientFieldChart(law, mean_phi, mean_links)` requires an actual regular mean: all `15V` gauge columns must be independent. It exposes `mean_basis` of shape `(105V,90V)`, `normal_dim=90V`, `gauge_dim=15V`, `raw_dim=105V`, the mean, and the full mass diagonal. A full weighted QR of the mean gauge matrix gives a real mass-orthonormal normal basis B:

\[
 B^TMB=I_{90V},\qquad D_0^TMB=0.
\]

No normal direction is selected away. The normal basis is horizontal at the mean, **not** presumed horizontal off-mean.

The gauge convention has a single owner: B's public `soft_spatial_closed_rotation.native_full_gauss_columns(law, fields, links, *, field_tangent=None, link_tangent=None)` returns `(CSR_D, mass105)`. With both tangents it returns exact directional `D'`, not `D+D'`. The builder supplies `i[T,Phi]` and the link action `lambda_x-U lambda_y U†`; this component does not duplicate it. Its mass is checked against the original mass above.

## Exact nonlinear embedding and full measure

For any supplied full real normal vector q, Phi is linear in `(Bq)_Phi`. A raw link coefficient denotes the **left** tangent `dU=i L U`. To embed it as the requested `U=U0 exp(i xi)`, the construction first converts the mean left algebra into the body algebra by `Ad(U0†)`. This matters for the nonidentity supplied mean links.

`embedding(q)` returns the actual fields and links, all `90V` normal tangent columns, body coordinates `xi`, and the local Frechet maps. The exponential derivative is exact, evaluated spectrally: in an eigenbasis of X, the left-trivialized map is

\[
 H_{ab}\mapsto e^{i(\lambda_a-\lambda_b)/2}
 \operatorname{sinc}\!\left((\lambda_a-\lambda_b)/(2\pi)\right)H_{ab},
\]

where NumPy's sinc is `sin(pi z)/(pi z)`. Coincident eigenvalues are included continuously; no truncated commutator series is substituted.

`evaluate(q)` builds the **square full ambient** Jacobian and Gram:

\[
 J=[J_n,D(Phi,U)],\quad G=J^TMJ,\quad G^{-1}=\texttt{cometric}.
\]

All returned blocks use normal-before-gauge ordering. The full relative density is

\[
 w(q)=\exp\left(\tfrac12\log\det G(q)-\tfrac12\log\det G(0)\right).
\]

It multiplies `dq` and the **literal product SU4 Haar measure** in the gauge frame. Gauge columns are invariant frame directions, not flat exponential gauge coordinates. If a consumer uses exponential coordinates for the gauge variables too, it must retain their Haar coordinate Jacobian separately. The constant mass and chosen normalization cancel in this relative density; gauge-orbit volume does not cancel pointwise. There is no normal-only or ad hoc flat-Haar measure.

The chart is local. Arbitrary full q is accepted by the embedding, but a metric evaluation outside a regular chart fails explicitly. Exponential coordinates need not be globally injective and gauge-orbit rank can change. No global regularity radius or huge-legal-N coercivity result is claimed here.

## Mechanical connection, vertical Casimir, and weak kinetic API

Write `G=[[N,C],[C^T,O]]`. The result includes:

- `metric`, `cometric`, `normal_metric`, `normal_gauge_metric`, `normal_gauge_cometric`;
- `orbit_metric=O`, `orbit_inverse=O^-1`;
- `mechanical_connection=A=O^-1 C^T`;
- `horizontal_metric=S=N-C O^-1 C^T`;
- `relative_density`, `relative_log_density`, `log_density`, and full Jacobians/field point.

For fibre-valued normal derivatives n and genuine equivariant gauge derivatives g, the original positive quadratic form integrand is

\[
 \frac{\chi}{2}w(q)
 \begin{bmatrix}n\\g\end{bmatrix}^{\!*}G^{-1}
 \begin{bmatrix}n\\g\end{bmatrix}
 =\frac{\chi}{2}w(q)\left[
 (n-A^Tg)^*S^{-1}(n-A^Tg)+g^*O^{-1}g\right].
\]

The last term is the vertical/orbit Casimir contribution. The cross terms and all normal derivatives are retained. In particular, inserting only `S^-1` and discarding g is not the original form for an equivariant matter/Ref fibre.

`positive_kinetic_bilinear(point,nL,gL,nR,gR,chi=1.)` returns the corresponding complex sesquilinear **density-weighted integrand**. Arrays must be `(normal_dim,)` and `(gauge_dim,)`, or `(normal_dim,fibre)` and `(gauge_dim,fibre)` with matching fibre sizes; neither block is optional. Gauge derivatives can, for example, be actual `i T_a psi` in the consumer's chosen equivariance convention. The caller integrates this against dq and literal Haar, and supplies the actual fibre actions and original potential. This utility does not manufacture a covariant strong operator, gauge representation, Hamiltonian, basis truncation, or state.

`tangent(point,v)` returns physical `(dPhi,dU)` for a full normal direction. `directional_jet(q,v,point=None)` returns exact `phi_first`, `links_first`, full `jacobian_first`, `metric_first`, and `log_density_first`. Second exponential Frechet derivatives are computed by the sum of both ordered 12-by-12 triangular block exponentials. The gauge derivative is delegated to the same pure builder. No finite difference defines this API.

The existing `prepared_guard_colour_chart.pullback_metric` uses the compatible `Tr(Ba Bb)/4` and `2Tr(Ta Tb)` contractions and identical mass. Its restricted-tangent implementation is reused as an independent native check; core assembly uses raw full-coordinate matrices rather than allocating thousands of redundant matrix-valued tangents.

## Actual scoped native execution

Command, cwd, and runtime:

```text
/home/administrator/substrate-framework/.venv/bin/python prepared_neutral_ambient_metric.py --size 3
/home/administrator/substrate-222/proposals/P256-qed-backward
```

This is **actual N3 ImprovedPhiLaw**, not an N1 mock or six-coordinate surrogate: `V=27`, raw dimension `2835`, normal dimension `2430`, gauge dimension `405`. The CLI uses every normal coordinate at a seeded nonzero point, a full Cartesian perturbed Clifford mean, and nonidentity SU4 mean links. Final-source natural exit `0`, empty stderr, wall time `30.43950474797748 s` (scientific body `29.204851194052026 s`). CLI stdout records argv/cwd/Python, exact deterministic inputs and their SHA256, all loaded local source SHA256s, and unchanged-source checks. Reproduction is general `--size N` with the original law's `N>=3`; dense QR/Gram/inversion scales cubically in `105V` and quadratic memory, so no astronomical allocation claim is made.

Observed discriminators:

| Quantity | Observed |
|---|---:|
| Mean mass horizontality | `3.4051579143610497e-14` |
| Mean normal mass-orthonormality | `4.474221250348412e-14` |
| **Nonzero off-mean normal/gauge coupling** | `10.454494388750181` |
| Full relative density | `0.17477812710665536` |
| Full determinant identity residual | `1.1368683772161603e-13` |
| Link Frechet central-FD residual | `4.036359998429892e-10` |
| **Full** Jacobian central-FD residual | `1.486333667145036e-9` |
| Log-density derivative central-FD residual | `6.291564507224234e-9` |
| Original ambient-gradient/form relative residual | `1.799376724106131e-16` |
| Positive form integrand (`chi=.73`, two-component fibre) | `631.6768460985184` |
| Pure vertical Casimir integrand | `38.579367060694516` |
| Existing guard pullback agreement | `7.047475582029698e-16` |
| Wrong normal-only log-density defect | `0.7724577927591203` |
| Wrong `.3` colour-cometric form relative defect | `0.18618090340473797` |

A one-curve/six-normal-coordinate input is rejected. The positive form is independently checked by solving `J^T ambient_gradient=derivatives` and contracting the original inverse mass; this is not a forwarded/mock-echo comparison. Full density is independently checked against `log|det J|+log det(M)/2`. All assertions use the actual changed path.

The completed **predecessor full-chart** receipts remain immutable: `local://issue222-ambient-metric-current-native-receipt.json` binds natural process output, stderr/exit/wall time and executed source; `local://issue222-ambient-metric-current-native-data.json` contains expanded CLI data. Their executed source SHA256 is `cdf5e52c33aecf5b13686d37a0378a7fab3ef67ab37c123d4d994f2d6e5a1826`; input SHA256 is `50c58ad3c350187c5aab737a6751404a3b30a8f57b942d3171c7fa18338062af`. The successor below adds a separate regular-orbit proof to the same pair; neither predecessor receipt nor frozen normal/Riccati science was rerun or retagged.

## Successor: fixed regular orbit radius from the actual full four-Gram

This follow-up binds the same plan under canonical current revision `e1e9ca716d3015f1a91e5e143b5d564ce8d188f2c074811a890c91d3d12fe5d6`. The predecessor pin and receipts above remain unchanged. The new source is `186d4347474113cc70f86bd92c309bb7e8bd30898e69f69b8536c7b7f1a047f2`.

At an actual mean write **all four** Cartesian fields, including Phi0, as
`Phi_mu=sum_nu a_mu,nu Gamma_nu`, with full Gram `G=a a^T`.
The exact traceless Hermitian Clifford basis has grades 1,2,3,4, multiplicities
4,6,4,1 and supercommutator eigenvalues **12,8,4,16**. The module's
`clifford_supercommutator_certificate()` checks these identities on all15
Gaussian-integer matrices by exact equality and their full HS orthogonality.
Thus, for every traceless Hermitian T and every full4 Gram lower `g`,

\[
 \sum_\mu\|[T,\Phi_\mu]\|_{\rm HS}^2
 \ge g\sum_\nu\|[T,\Gamma_\nu]\|_{\rm HS}^2
 \ge4g\|T\|_{\rm HS}^2.
\]

The **original coordinate normalization is essential**: Phi's basis has
`Tr(Ba Bb)=4 delta_ab`, whereas the gauge generators have
`Tr(Ta Tb)=delta_ab/2`. Consequently the mean coordinate orbit bound is
`D_Phi^T D_Phi >= (g/2)I`, not `4g I` or `g I`.
For a perturbation of the entire four-field tuple,

\[
 \|\delta D_\Phi\|_{{\rm HS}\to{\rm HS}}
 \le2\|\delta\Phi\|_{\rm HS}.
\]

If `||deltaPhi||HS <= sqrt(g)/2`, its smallest HS singular value remains at
least `sqrt(g)`, giving the **perturbed coordinate** bound

\[
 D_\Phi^T D_\Phi\ge(g/8)I,\qquad
 I_{\rm orbit}=D^TMD\ge(g/8)I,\qquad
 \|I_{\rm orbit}^{-1}\|\le8/g.
\]

All original link terms are positive and cannot decrease this lower bound,
regardless of their SU4 values or centre phases. This argument applies sitewise
and therefore uniformly to every site without a volume-denominator loss.

`regular_orbit_certificate(gram: GramBounds, phi_HS_lipschitz=Fraction(2), size=NSTAR)`
returns exact Fraction lower bounds and the full input provenance. If a chart
has `||deltaPhi||HS <= C_J r`, take
`r <= sqrt(g)/(2 C_J)`. For **this** full-normal chart, Phi is exactly linear
and B is mass-orthonormal, so `||deltaPhi||HS <= 2||q||` and `C_J=2` exactly:
`r <= sqrt(g)/4`. A separate nonlinear quotient chart must provide its own
actual `C_J`; no supplier chart bound is invented here.

### Full4 input authority and normalization correction

The actual source `noncompact_quantitative_normal_budget.GramBounds.native_a1()`
supplies full4 `[1/20,1/2]`, provenance
`NativeMetricAdmission all-ten Arb128, N>=25`. It is bound through the actual
`native_normal_budget(NSTAR).gram`, not through a light spatial three-Gram.
The initially suggested `1/8` was withdrawn: the source's spatial eigenvalue
is near `.06277`, below `1/8`. Therefore the legal perturbed orbit bound is
**1/160**, inverse bound **160**, mean orbit bound **1/40**. A bound **1/64**
cannot be inferred from these authoritative raw inputs. The linear full-normal
chart's certified radius is the exact rational
`16499269484942379435/295147905179352825856`, an outward lower bound for
`sqrt(1/20)/4`. The source's all-ten enclosure remains a supplied frozen
analytical input; this cheap successor does not re-execute its Arb or Riccati
supplier or claim to re-prove its all-legal-N enclosure.

### Actual ordinary/Root mean and native execution

`native_regular_orbit(5)` calls the original **full-ten Gram minimizer**
`original_frame(RotationBand(**BARE),5,pi*label/2,True)` for all64 labels.
It records the full4 matrix `Tr(Phi_mu Phi_nu)/4`, Phi0 norm, spatial3 Gram,
and the residual against the actual source `selected['G']`; it never reads
the first three light matrices as a substitute. All64 means are perturbed
at the proved HS boundary and their full15-column original coordinate orbit
spectra checked against `g/2` and `g/8`.

The actual512 ordinary/Root labels have
`theta=pi*(m/2+s)`, `m in {0,1,2,3}^3`, `s in {0,1}^3`.
Exactly `m+2s=label+4k`, so their shifted finite momentum grids are the same
as the64 representative grids under an integer reindexing. The receipt
records every m,s,label,k. This is a mean-Gram identity, **not** deletion of
any history, phase, Root sector, beta coupling, or SOURCE dynamics.
Common SU4 conjugation preserves the full4 Gram by cyclicity of trace;
a common real Phi sign also preserves it. Central SU4 link phases act
trivially on adjoint Phi and cannot spoil the positive link contribution.
The same bound therefore covers these actual gauge/centre copies without
equating their coupled H/state dynamics or projecting away Root beta.

New command (no old normal/Riccati/full-chart rerun):

```text
/home/administrator/substrate-framework/.venv/bin/python prepared_neutral_ambient_metric.py --regular-orbit-only --size 5
```

Natural exit `0`, empty stderr, wall time **2.6609582090750337 s**.
The minimum of all64 actual full4 Grams is `.06295423353169809`.
A native all-site N5 gauge builder with nonidentity SU4 links gives Phi orbit
form `291.1309592941564` and positive link contribution `17399.89236539313`.
The adverse **light3-only** Phi orbit has minimum eigenvalue exactly `0.0`:
discarding Phi0 cannot establish the required full15 regular orbit.
No random-point experiment substitutes for the above all-perturbation inequality.

The completed orbit-only predecessor receipts remain immutable:
`local://issue222-ambient-fixed-orbit-current-receipt.json`,
`local://issue222-ambient-fixed-orbit-current-data.json`,
`local://issue222-ambient-fixed-orbit-current-lsp.json`.
Input SHA256: `f76442d1b54bb479fde92d7c36bdc81910f172ad0714a61bbd6ca75bdb936a77`.
Its real Pyright `1.1.414` gave **239 symbols**, definition/references,
actual **zero diagnostics**, and17 positive AST declarations. The initial
successor typing-diagnostic receipt is retained; the corrected orbit CLI
reached natural exit before that source-bound LSP. They are not retagged
as the newer full-J proof below.

**Orbit-only boundary:** orbit positivity alone did not prove the combined
`[Jnormal,D]` transverse. The additional proof below now supplies that missing
local statement for this same chart on the **existing** frozen legal radius,
without reducing the domain. It still does not establish arbitrary nonlinear
quotient charts, global injectivity, H/state/SOURCE completion, or dynamics.

## Full ambient Jacobian transversality on the unchanged original radius

This additional same-pair result closes the orbit-versus-full-J local geometry
gap; it does not replace the original radius, H, measure, or full coordinate
space. Define the **mass-weighted** square Jacobian
`K(q)=sqrt(M)[Jnormal(q),D(q)]`. At a full4 Clifford mean, mass-orthonormality
and horizontality of B imply

\[
 \sigma_{\min}(K(0))\ge\sqrt{g/2}
\]

for the actual `g=1/20` (more generally `0<g<=2`). This is a statement about
all `105V` columns, not just the gauge orbit. Here `||q||<=r` is the full
normal-coordinate Euclidean norm, equal to the mass norm of Bq.

The exact body-link algebra has
`||X_e||HS^2=|(Bq)_link,e|^2/2` and hence
`max_e ||X_e||HS <= sqrt(.3) r`, since the original colour mass is `1/.6`.
Using the exact integral dexp,

\[
 \|\operatorname{dexp}_X-I\|
 \le\int_0^1 2t\|X\|_{\rm op}\,dt
 =\|X\|_{\rm op}\le\|X\|_{\rm HS},
\]

and orthogonality of the mean adjoint transport, one gets
`||sqrt(M)(Jnormal-B)|| <= sqrt(.3) r < 2r`.
Phi's actual coordinate conversion gives the gauge-column perturbation
`||delta D_Phi|| <= sqrt(2) r`.
For links, only the **target** adjoint block changes: the source identity
does not. `||Ad(exp(iX))-I||<=2||X||op`; each periodic native site has exactly
three incoming links. With the original colour mass this gives
`||sqrt(M_link) delta D_link|| <= sqrt(6) r`.
Thus the complete gauge perturbation is bounded conservatively by
`(sqrt(2)+sqrt(6))r < 4r`, and the complete square Jacobian by

\[
 \|K(q)-K(0)\|<8r.
\]

No volume factor or truncated exponential series enters this operator bound.
Weyl's singular-value inequality now implies, whenever
`r<=sqrt(g/2)/16`,

\[
 \sigma_{\min}(K(q))\ge\tfrac12\sqrt{g/2},\qquad
 G(q)\ge(g/8)I,\qquad G(q)^{-1}\le(8/g)I.
\]

Therefore the **full** metric, not merely its orbit block, has legal lower
`1/160` and cometric upper `160`. The full determinant/Haar density is strictly
positive throughout that existing local chart. This does not assert global
SU4 exponential injectivity.

### The existing frozen radius automatically satisfies the condition

Both actual original source formulas contain
`r<=delta/(64 V h)`, with
`delta=(2/5)sqrt_lower(g)` and `V>=1,h>=1`:

- `noncompact_quantitative_normal_budget.native_normal_budget` has
  `h=2^12*(105V)^2*(1+rho+cx)`;
- `prepared_minset_normal_proof.local_normal_constants` has
  `h=2^20*(105V)^2*(1+rho+cx)`.

Hence the **same existing** radius obeys

\[
 r\le\frac{\sqrt[\rm lower]{g}}{160Vh}
 \le\frac{\sqrt g}{160}
 <\frac{\sqrt{g/2}}{16}.
\]

The ratio squared to the theoretical allowed radius is at most `1/50`.
This proof reads the actual existing radius formulas and uses their positive
lower bounds; it does **not** calculate their giant rational chart constants,
rerun frozen normal/Riccati science, shrink a radius, or introduce a new domain.

`regular_full_jacobian_certificate(gram, radius_upper=None, size=NSTAR)`
returns exact Fraction full-metric/cometric bounds. The default binds an
**upper bound** on the existing frozen radius,
`sqrt_lower(g)/160`; it is not a new chosen radius.
For an independently supplied `radius_upper`, the API checks the sufficient
condition and explicitly labels that independent input: the caller must bind
it to its own domain. The proof applies to this linear-Phi/full exponential
chart and all actual means covered by the full4 enclosure and transforms
proved above, not to an arbitrary differently parametrized quotient chart.

### Actual new full-J positive control

```text
/home/administrator/substrate-framework/.venv/bin/python prepared_neutral_ambient_metric.py --full-jacobian-certificate-only --size 3
```

New natural exit `0`, empty stderr, wall time **8.875055412063375 s**.
The supplied mean is one actual original N5 full4 Gram centre, including its
nonzero Phi0/spatial cross term, embedded on the native N3 lattice with
nonidentity SU4 mean links. All `2430+405=2835` columns are evaluated at a
nonzero full-normal diagnostic vector of norm `1/1024`. This native diagnostic
radius is **not** the formal frozen legal radius and is not substituted for it.
The weighted full-J perturbation Frobenius norm (an upper bound on operator
norm) is `.0019860332747937705`, ratio `2.033698073388821` to that native radius,
below the analytic coefficient8. Cholesky of `G-(g/8)I` proves the actual full
massGram lower at that native point; it is not an orbit-only check.
The full relative density is `1.0000531815431317`, and its independent square
Jacobian determinant identity error is `3.410605131648481e-13`.

Current source SHA256:
`16c88a6284d16208d6e9eb5932a2ffa41b4188a5e627baf73daa2d9bee86bd28`.
Input SHA256:
`b3e94389dc0e6ca8e72678ac056c9f088405b68ba5e3af3532ad95b54503c4de`.
Receipts: `local://issue222-ambient-fullJ-current-receipt.json`,
`local://issue222-ambient-fullJ-current-data.json`,
`local://issue222-ambient-fullJ-current-lsp.json`.
Final real Pyright `1.1.414`: **267 symbols**, certificate definition/references,
actual **zero diagnostics**, and19 positive Python AST declarations.
Earlier full-chart and orbit-only executions remain bound to their original
source bytes; no supplier receipt is rewritten into this new proof.

## Scope boundary

These native calculations and exact finite-dimensional identities establish a reusable full ambient chart/measure/original weak-form coefficient component. They do **not** establish full-field H assembly, evolved prepared state, all ordinary phases/512 histories, source normalization, a global ground state, legal-N Gaussian/coercivity estimates, full physical SO3 closure, or full-Gauss dynamics. Those obligations are neither weakened nor replaced by this chart. EM/harmonic/Root/light/Ref sectors are not silently included in the Phi/SU4 geometry; the integration owner must attach their same-law forms and actions.

Real Pyright `1.1.414` stdio LSP was exercised before implementation on `pullback_metric` (symbols/definition/references and actual zero diagnostics). The predecessor Python had 177 symbols, definition/references for `AmbientFieldChart`, actual published zero diagnostics, and14 positive Python AST declarations: `local://issue222-ambient-metric-current-final-lsp.json`; it remains bound to its executed predecessor source, not the successor. Structural ast-grep positively matched the full `AmbientFieldChart` declaration. OMP `xd://lsp` is unavailable; this is an explicit adapter gap, not a language-server absence. These tooling receipts do not prove scientific state evolution.

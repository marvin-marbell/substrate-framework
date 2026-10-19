# Original whole-heavy charged principal form and an explicit slow-kinetic fork

This construction belongs to issue222, historical assignment plan revision `e1874737294f261dfc305be3f8d054bd1223a1a3142bdf1a70c6d6cc136b76a4`, subsequently recorded by the integration owner at `1a06b52727e8540a5ab588172f751a794625aa466a0a06cfa35fcbc28fc42464`. The general template remains `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`. The campaign is active. Neither a frozen Dirac pole nor the joined electron-photon outcome is asserted here.

## Actual law and the material change

Use the original periodic endpoint law of `BulkQuantumLaw.onebody`, **not** its additional local selector. There are four traceless Hermitian fields per site, all 60 real Cartesian coordinates, three SU4 links per site, all 45 link coordinates, the original itinerant whole-heavy sector `NH=2V`, and the complete retained light CAR space with the half-filled `NL=2V` vacuum. All `15V` colour Gauss equations and the existing compact U1 law remain. The source basis satisfies

\[
 \operatorname{Tr}B_aB_b=4\delta_{ab},\qquad T_a=B_a/\sqrt8,\qquad
 \operatorname{Tr}T_aT_b=\delta_{ab}/2.
\]

For the source cutoff family, `a` is retained, the band is `{m=.2,r=.12,v=.3,improved=False}`, and the original local quartic stiffness is `kappa=2/a`. The frame is selected by `bulk_flat_holonomy_bound.selected_gram` on an explicitly reported **periodic** source grid, then reconstructed as `Phi_mu=sqrt(G_mu_mu) SELECTED_mu`. This finite-grid frame is not an interval continuum optimizer or a solved quantum ground state. The Bloch consumer has the genuine original momentum grid, including a chosen `Z4^3` centre toron; it does not inherit the older midpoint-only environmental candidate.

`--native-frame-toron` instead minimizes the same original convex Gram functional in **all ten symmetric Gram coordinates** on the specified frame grid and actual centre toron. This matters because the source `selected_gram` owner solves the cubic two-parameter `(g0,g1,g1,g1)` case, whereas a quarter-centre toron need not be cubic. The extension solves in dimensionless `aH` units, with `xhat=a*x` and stiffness2, then divides its value by a. With an orthonormal symmetric basis `E_A`, `d_A(k)=xhat^T E_A xhat`, its exact derivative and Hessian are

\[
 \partial_A F=2c_A-\langle d_A/r\rangle,\qquad
 \partial_A\partial_B F=2\delta_{AB}
 +\tfrac12\langle d_A d_B/r^3\rangle.
\]

A positive-domain damped Newton iteration checks full ten-coordinate stationarity, and reconstructs `Phi_mu=sum_nu (sqrt(G))_mu_nu SELECTED_nu`, including off-diagonal Gram entries. This is a conditional finite-grid interior optimizer of the unchanged original Gram functional, not an extra selector, a nonuniform global optimizer or a toron-minimization theorem. To study native N3 quarter-centre uniform stability, use both `--frame-grid 3 --grids 3 --native-frame-toron --centre 1 0 0`; a larger identity reference grid is an explicitly different frozen-frame diagnostic.

The Newton descent test evaluates a stable **convex remainder**, rather than subtracting two O(1) objectives. For the actual coefficient change `delta`, put `t=d·delta` and `r_new=sqrt(r²+t)`. Its exact objective difference is `gradF·delta+STIFFNESS ||delta||²/2+average(t²/[r(r_new+r)²])`. The gradient tolerance and positive-domain checks are unchanged. On native N5 centre(0,0,2), direct objective subtraction stalled at gradient1.621e-9 with a full-step decrease5.158e-19 below the1.110e-16 objective ulp. The repaired solver reaches the independent original-field stationarity criterion across all64 centres at both a1 and a.001; `tests/test_original_gram_stationarity.py` retains the old adverse case and an unshifted-import positive control. This repairs numerical descent, not the physical source law or its acceptance tolerance.

Keep heavy coefficient `Lambda`, alignment `.2 Lambda^(5/4)`, colour plaquette `.7 Lambda^(5/4)`, and set the colour electric coefficient explicitly to

\[
 u_4=.3\Lambda^{-p},\quad p=.5.
\]

Compare two different field kinetic laws:

1. The earlier/original fast trajectory: `sqrt(Lambda)(-Delta_Phi)/2`.
2. The material new trajectory: `Lambda^-s(-Delta_Phi)/2`, `s=.5`.

The second is **not** a consequence of or an unchanged parameter choice inside the first kinetic family. Both coefficients are positive at every finite Lambda. The original heavy endpoint, the full domains and compact U1 dynamics are not replaced. No additional 27-channel local CAR energy is executed or introduced.

## Exact native Hessian and full Gauss principal form

At the supplied frame, let `z=(deltaPhi,theta)` be the full 105-coordinate chart. The executable reuses `SpatialFunctional`'s actual endpoint vertices, mixed field/link and link/link contact terms, whole-cluster response and native gauge columns. It changes its quadrature to the periodic original source grid and its quartic stiffness to `2/a`, not its endpoint implementation.

Two independent cutoff-owner checks are retained. The q0 field60 block agrees with the recomputed periodic uniform Hessian using the actual `kappa=2/a`. Also the entire leading105 form agrees, after multiplication by a, with a separately constructed dimensionless source band `{a=1,mhat=a*.2,r=.12,v=.3,kappa=2}`. The J/K forms are not divided by a or accidentally absorbed into that leading-unit conversion. Native spectrum or off-stationary Ward agreement alone would not detect the wrong quartic coefficient; these independent checks do.

With `D_a=coordinates(i[T_a,Phi])`, define

\[
 d_j(q)=1-e^{iq_j},\quad
 G(q)=\begin{pmatrix}D\\d_1I_{15}\\d_2I_{15}\\d_3I_{15}\end{pmatrix},\qquad
 I=D^\dagger D,\quad b(q)=\sum_j|d_j|^2.
\]

`I` is the actual Cartesian field-orbit inertia. It is generally not a scalar matrix. The full-rank Clifford representation has all 15 nonzero Lie-orbit directions and only a discrete centre stabilizer. At q0 the link part of G is zero, **but D is not**. For a singular Cartan control `Phi0=diag(5,1,1,-7)`, the executable retains the actual five Lie stabilizers, checks rank10 before kinetic weighting, and rejects inhomogeneous charge in the unsupported directions.

For alignment,

\[
 L_jz=d_j\delta\Phi-D\theta_j,\qquad
 H_J=.2\sum_jL_j^\dagger L_j.
\]

The original plaquette `P=||U_square-I||_HS^2/2` has quadratic Hessian

\[
 H_K={.7\over2}\sum_{a<b}C_{ab}^\dagger C_{ab},\qquad
 C_{ab}\theta=d_b\theta_a-d_a\theta_b.
\]

Both are positive, annihilate the native G, and include all colour directions. Write

\[
 H_{\rm pot}(q)=\Lambda H_0(q)+\Lambda^{5/4}(H_J(q)+H_K(q)),
\]

where `H0=local+whole-heavy response+actual diamagnetic contact`; no sea sign or contact is omitted. The finite source Ward identity is `H(q)G(q)=-(d tau)^T gradient`. In particular **HG is not forced to zero off stationarity**. The receipt retains the complete gradient and reports whether the supplied frame is stationary on each executed grid. A reference frame imported from a different grid/toron can be off stationary. For the actual full10 Gram stationary frame on a centre toron, however, the **endpoint-average law gives exactly zero colour-link gradient**. The earlier proposed order-Lambda quarter-centre gradient/shift mechanism was withdrawn: it came from omitting that average.

To prove the exact cancellation, write `h(k)=sum_mu x_mu(k) Phi_mu`, `h^2=R(k)^2 I`, and `P+=(I+h/R)/2`. The source endpoint-average uniform link vertex is

\[
 V_a(k)=\tfrac12\{T_a,\partial_{k_j}h(k)\}.
\]

Its occupied trace has an identity-projector term whose grid sum is zero because each distinct nearest-axis phase has zero sum on a full periodic grid, including any centre shift. The other term vanishes pointwise:

\[
 \operatorname{Tr}(h V_a)
 =\tfrac12\operatorname{Tr}[T_a\{h,\partial_{k_j}h\}]
 =\tfrac12\partial_{k_j}(R^2)\operatorname{Tr}T_a=0.
\]

Full10 Gram stationarity supplies the field-gradient cancellation; J/K gradients vanish at flat covariant-uniform fields. Thus this classical scalar frame is exactly stationary for every Lambda, and positive physical Hessian at sufficiently large Lambda establishes a local scalar minimum, not an arbitrary shifted one or a nonuniform global optimizer. The executable checks the Clifford anticommutator and link gradient and reports a same-input wrong **unsymmetrized** endpoint vertex. In the actual native quarter-centre cutoff control that wrong gradient norm is `3.0479652782955853`, versus `1.425998838094533e-14` for the true endpoint average. Some reflection-symmetric inputs null this adverse probe; the quarter-centre input is its working positive witness.

For field cometric `k_f` and link cometric `2u4`,

\[
 A=\operatorname{diag}(k_fI_{60},2u_4I_{45}),\quad M=A^{-1}.
\]

The mass-horizontal complement is the kernel of `(sqrt(M)G)^dagger`; the physical squared frequencies are the eigenvalues of `Z^dagger sqrt(A) Hpot sqrt(A) Z`. The executable separately crosses this generalized computation against the existing `principal_modes` interface by the exact coordinate transformation `z=R z_old`, `R_field=sqrt(k_f/sqrt(Lambda))`, `R_link=1`. It reports negative signs, ranks, all normal frequencies and field participation. Frequencies are quadratic/classical or local harmonic diagnostics, not exact quantum levels.

Raw numerical eigenvalues are retained, but mode signs below the measured eigen-residual plus a conservative dimension/roundoff scale are explicitly **unresolved**. Sorted-frequency slopes omit them. At extreme fast Lambda a small relative interface agreement cannot certify small eigenvalues. The unweighted `ker(G†)` potential spectrum is always retained as **Euclidean-slice diagnostics**. It supplies a mass-independent physical-sign verdict only when the source-scale potential is gauge-null/stationary: otherwise its complement differs from the actual `ker(G†M)` coordinate complement by gauge components whose Hessian cross/self terms need not vanish. The CLI checks both `||a grad V0||≤2e-8` and `||a H0 G||≤3e-8`, independently of the stiff-dominated Lambda scale. The physical-sign fields are **null/unknown**, not false or positive, on off-stationary inputs; raw slice eigenvalues and counts remain. On a stationary gauge-null Hessian the complement changes are congruent and the sign cross-control applies. These are floating diagnostics, not interval-certified bounds. The accepted default imported frame-grid16/nativeN3 has gradient norm.0115257 and is the adverse off-stationary witness; the native full10 quarter-centre frame is the legitimate recovery.

At q0, strong alignment restores all 45 link coordinates while leaving the 60 uniform fields. Removing the actual 15 gauge fields leaves 45 uniform field normals. At nonzero fixed q, full-rank D makes alignment positive on the full physical quotient; the plaquette is retained as an additional positive form. Positivity of `H0` on the uniform shape complement is a separate checked normal property, not an automatic consequence of positivity of J/K. An off-stationary finite frame must not be called a stable equilibrium merely because its horizontal Hessian has positive eigenvalues.

## Complete inhomogeneous Gauss minimum, including q0

For a resolved supported charge vector rho, minimize

\[
 {1\over2}p^\dagger A p\quad\text{subject to}\quad G^\dagger p=\rho.
\]

The complete square is

\[
 K=G^\dagger M G={I\over k_f}+{b(q)\over2u_4}I_{15},\quad
 S=K^{-1},\quad p_*=MG S\rho,
\]

\[
 {1\over2}p^\dagger A p
 ={1\over2}\rho^\dagger S\rho
 +{1\over2}(p-p_*)^\dagger A(p-p_*),\qquad G^\dagger(p-p_*)=0.
\]

In a singular sector the inverse is taken **only on the rank-resolved support**, and every remaining stabilizer Gauss charge is required to vanish; an unsupported rho is infeasible, not an inexpensive charge that can be discarded. The executable checks the full15 constraint, the value, and an adverse horizontal displacement against the exact excess.

This resolves the proposed finite-cost mechanism and its adverse countercase:

- Fast law, fixed nonzero q: `S ~ (2u4/b)I15`, so its Coulomb coefficient vanishes as `Lambda^-p`. Completing only a putative scalar rotor is unnecessary and wrong.
- Fast law, q0: `S=sqrt(Lambda) I^-1`. A nonzero unsupported global charge on a torus is not physical; a supported but non-neutral global colour charge has a **diverging** normal-form cost. The link incidence cannot screen q0. Discarding q0 would manufacture a false bounded charged theorem.
- Slow law, q0: `S=Lambda^-s I^-1` and its coefficient vanishes.
- Slow law, `p=s=.5`, any q:

\[
 S(q)=\Lambda^{-1/2}\left[I+{b(q)\over.6}I_{15}\right]^{-1}.
\]

This last identity is exact at the frozen frame, not merely an asymptotic equivalent. Geometry alone still says nothing about existence of an isolated interacting light pole.

## Heavy whole Slater mean, variance and Berry: what cancels

The heavy occupation is selected from the largest **TOTAL 2V** eigenvalues of all `4V` modes. The source Clifford/centre cases happen to have a positive/negative separated cluster; this is verified, not applied to arbitrary twists. An actual native Cartan-toron counterexample recomputes the `4V` endpoint matrix, occupation histogram, colour density and energy, and rejects forcing two occupied modes in each momentum block. It has traceless SU4 toron angles, not an illicit U4 colour reference.

For projector P on the actual positive whole cluster,

\[
 \mu_a={1\over V}\sum_k\operatorname{Tr}(P_kT_a),\quad
 C^H_{ab}(q)={1\over V}\sum_k\operatorname{Tr}
 [P_kT_a(1-P_{k+q})T_b].
\]

At a Clifford point one may derive `P+=(1+h/R)/2`, but the executable uses the actual spectral whole occupation. The original positive Wilson mass gives a nonzero mean colour component along the temporal Clifford direction; **trace-free one-body h does not imply zero occupied colour density**.

For the full105 endpoint derivative `V_i(q)`, the actual Slater quantum geometric tensor is

\[
 g^H_{ij}(q)={1\over V}\sum_{k,o\in P_k,e\notin P_{k+q}}
 {\langle o|V_i^\dagger|e\rangle\langle e|V_j|o\rangle
 \over(E_e-E_o)^2}.
\]

It is well defined even when occupied modes are internally degenerate: no denominator inside an occupied/empty cluster is used. The native gauge vertex satisfies `V(G)=i(T h_k-h_(k+q) T)`, hence the executable checks the exact pullback

\[
 G^\dagger g^H G=C^H.
\]

At q0 `2 Im(gH)` is the heavy determinant Berry curvature in the chosen real-coordinate chart. Along the orbit its connection is the actual occupied charge mean, not zero by fiat. The complementary **light vacuum** projector is `1-P`, so `muL=-muH` exactly and its q0 determinant Berry curvature cancels. Its metric and covariance are calculated separately. For arbitrary twists/nonzero q one must not replace the complementary tensor by the same per-q tensor without proof; the receipt does not do so.

Two independent Slater species have additive variance. Thus complementary mean and q0 Berry cancellation **do not** imply zero variance or zero kinetic energy. In particular a product complementary vacuum is not an angularly constant singlet at each field point. An entangled full-Gauss section can redistribute that cost, but it is a different explicitly constructed section, not a consequence of cancellation of means.

## Scalar Born-Huang versus light-dependent operators

The exact principal horizontal cometric is

\[
 A_h=A-GSG^\dagger,
\]

and, at the fixed frame,

\[
 B_H^{\rm normal}={1\over2}\operatorname{Tr}(A_hg^H),\quad
 B_H^{\rm angular}={1\over2}\operatorname{Tr}(SC^H),\quad
 {1\over2}\operatorname{Tr}(Ag^H)=B_H^{\rm normal}+B_H^{\rm angular}.
\]

These heavy-line variance/metric contributions are scalar on the **retained light space**, although field dependent. The receipt checks their exact decomposition and computes the actual complementary-vacuum additions rather than assuming a scalar factor2 for arbitrary q/twists. These numbers are frozen principal terms, not the full scalar quantum vacuum energy: nonlinear measure/drift and the anharmonic scalar vacuum still have to be included. A slow BO subtraction must use the exact common scalar heavy/normal vacuum energy, not just the harmonic zero-point sum, even when the classical scalar frame is exactly stationary.

If `R_a=Q_L,a+Q_ref,a`, the heavy-line projection of the angular quadratic is instead

\[
 {1\over2}\sum_{ab}S_{ab}(R_a+\mu^H_a)(R_b+\mu^H_b)
 +{1\over2}\operatorname{Tr}(SC^H),
\]

with the consistently normalized Fourier mean (`sqrt(V) mu` at q0). The terms `R S R/2` and `mu S R` are generally **light-dependent operators**, not subtractable common scalars. The executable probes the actual four-colour local fundamental algebra: its quadratic contribution has Fourier factor `1/V`, the heavy-mean cross contribution has no volume factor at q0, and both eigenvalues and non-scalar remainders are reported separately. This algebra probe is not claimed as an unreferenced physical charged state. It also calculates genuine non-scalar light normal endpoint vertices. The two-dimensional positive light cluster is internally degenerate; inventing a separated rank-one added-particle BO line or assigning its metric by a zero denominator is forbidden.

At the fast q0 trajectory these coefficients can grow as sqrt(Lambda). A scalar heavy variance subtraction does not remove the heavy mean cross operator. At the slow `p=s=.5` trajectory the full frozen A, S, heavy normal BH and angular light coefficients scale as `Lambda^-1/2`, **provided the frame/projector derivatives stay bounded**. This is the material removal of the fast-global-rotor obstruction, not a proof by analogy to a deconfined Higgs phase.

## A physical reference and exact neutralization

The added local fundamental at x is paired with a real static antifundamental `bar4` at y. Its singlet is

\[
 |s\rangle={1\over2}\sum_{i=1}^4|i\rangle_x|\bar i\rangle_y,
 \quad T_a^x=T_a\otimes1,\quad T_a^y=-1\otimes T_a^T.
\]

All 15 total charges annihilate it. For the normalized Fourier charge,

\[
 \rho_a(q)={T_a^x+e^{iq\cdot(y-x)}T_a^y\over\sqrt V},\quad
 \langle\rho_a^\dagger\rho_b\rangle
 ={ |1-e^{iq\cdot(y-x)}|^2\over8V}\delta_{ab}.
\]

The executable checks this representation and covariance directly. Its q0 charge is exactly zero and its Coulomb cost is `Tr(S Cpair)/2`. Omitting the actual reference gives the adverse covariance `I15/(8V)` and the fast q0 cost diverges. A fixed negative-charge physical sector could play the same role; a scalar compensating number cannot. This pair normal form alone is not an eigenstate of the coupled matter/background Hamiltonian. The independent centre-charge supplier's full unprojected path dressing is a stronger actual reference-pair trial, but is not silently substituted here for original-light insertion or charge localization.

## Symbolic powers, finite diagnostics and remaining operator premise

At fixed positive a, fixed finite N and a positive resolved normal Hessian, the slow branch gives

| mode | harmonic frequency power | coordinate-width power |
|---|---:|---:|
| uniform field/shape | `(1-s)/2` | `-(1+s)/4` |
| stiff nonuniform field | `(5/4-s)/2` | `-(5/4+s)/4` |
| stiff colour link | `(5/4-p)/2` | `-(5/4+p)/4` |

For `s=p=.5`, these are respectively frequencies `Lambda^.25`, `Lambda^.375`, `Lambda^.375`; widths `Lambda^-3/8`, `Lambda^-7/16`, `Lambda^-7/16`. Heavy particle-hole excitation energies have the original extra multiplier Lambda. The fast uniform field gap is `Lambda^.75` and its stiff field gap `Lambda^.875`; its q0 angular operator does not vanish.

A bounded covariant connection crossed with the stiff normal momentum has nominal slow power `Lambda^-.5 Lambda^7/16=Lambda^-1/16`, while a bounded BH scalar has power `Lambda^-.5`. More generally field-cometric exponent alpha gives stiff cross power `(3 alpha+5/4)/4`; radial gaps require alpha>-1, and stiff cross decay requires alpha<-5/12. These are useful discriminants, not uniform operator inequalities obtained from a Hessian. A mixed potential term `Lambda^(5/4) sigma tau^2` with the above slow widths is order1 on Gaussian states, **not pointwise vanishing**. Its off-diagonal second-order effect may be suppressed by the uniform gap `Lambda^.25`, but proving that requires the actual scalar normal resolvent and its matrix elements. No assertion that every Taylor remainder vanishes is made.

The CLI distinguishes the symbolic table from finite sorted-frequency logarithmic slopes, which are not branch-resolved asymptotic exponents. It reports the actual potential/principal eigenvalues, all native G columns, actual inertia, every charge Schur matrix, whole Slater metric/colour covariance/Berry terms, physical reference controls, native N3/N4 endpoint comparisons and the arbitrary-twist occupation adverse control. It records gradients rather than silently reoptimizing frames or deleting off-stationary components.

The specific missing premise for a genuine frozen low-energy current/pole implication is a **uniform isolated positive normal bundle and whole-heavy projected resolvent over the relevant full-Gauss/toron region**, with bounded covariant projector derivatives and controlled operator-valued retained-light/current transfer (compact U1 phases need not have a light gap). It must include the exact common scalar normal vacuum subtraction, not merely the frozen Hessian, and then demonstrate the physical referenced charged spectral pole/localization. Fixed-frame derivative formulas and finite normal spectra do not establish this premise. Finite-N global scalar minima, thermodynamic centre phase, quantum compact photon poles and localization/symmetry factorization remain outside this supplier's proved scope. The microscopic physical restored rotation still has +1; no Gamma is imported as physical spin.

## Execution boundary

Example final native execution, reserved to the integration owner after artifacts land:

```
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
proposals/P256-qed-backward/original_charged_principal.py \
--frame-grid 16 --grids 3 4 8 --lambdas 1e4 1e8 --p .5 --s .5
```

A quarter-centre and cutoff diagnostic can be selected with `--centre 1 0 0 --a .001`; all q indices refer to each actual grid, and q0 is part of the default inputs. Source/proof hashes are captured and rechecked before writing strict portable JSON. A successful exit proves only the exercised assertions, not a pole or phase. Child language intelligence used native root-launched Pyright symbols/definition and source diagnostics plus a positive Python structural class query. The mounted LSP adapter was unavailable, so the actual stdio protocol was used. Final numeric evidence belongs to the generated receipt and the integration owner's observed run, not to this unexecuted example.

Scoped child method-level smoke used the project interpreter, frame grid8, native N3, q indices `(0,0,0)` and `(1,0,0)`, a1, Lambda1e4, p=s=.5. It naturally exited0: rank15/normal90, no resolved negative modes, whole occupation54, heavy mean norm `.6864837273935163`, complementary mean cancellation `2.51e-17`, angular QGT Ward below `4.2e-16`, native endpoint spectrum agreement `7.22e-16`, and a genuine forced-two native Cartan energy error `.09349268882135808`. At q0 the referenced singlet cost was exactly0; omitting the reference cost `42.69111195926337` on the fast law versus `.004269111195926338` on the slow law. The corresponding heavy full BH terms were `631.9230460929609` and `.06326496745333057`; heavy-mean light non-scalar norms were `511.88578218364677` and `.05118857821836467`. This scoped smoke is not the integration owner's final multi-grid CLI or a quantum spectrum. The own receipt is a scoped native supplier invocation with all frozen matrices retained; the integration owner may replace it with its broader final invocation.

The additional **stationary native quarter-centre cutoff** probe uses a.001, N3, frame-grid3 and the full10 native-toron option. It observed Gram gradient `2.22e-16`, field gradient `2.38e-13`, link gradient `1.43e-14`, physical stiffness2000, independent q0 uniform60 Hessian relative error `6.01e-16`, and dimensionless `aH` leading105 relative error `4.63e-16`. The nonstiff uniform45 leading sector has no negative modes. Nevertheless the full q0 physical normal form has **eight** finite-stiffness negative modes at Lambda1e4 and **four** at Lambda1e8. At Lambda1e12 the independent unweighted physical potential is positive; at Lambda1e16 the slow spectrum remains positive but the direct fast calculation has six raw negative values far below its floating resolution, which are not physical sign evidence. The previous report that both direct frequency spectra had zero raw negatives at Lambda1e16 was withdrawn after inspection of the full, initially truncated output. This is a real finite-stiffness link countercase at the lower Lambdas, not a wrong stiffness owner or a global asymptotic impossibility. Fixed-a large-Lambda domination must be checked: here `H0` coefficients are order `1/a`, while strong J/K are order1 before their extra `Lambda^1/4` multiplier. A positive uniform Gram Hessian alone does not certify the finite full105 normal form. The scoped receipt retains the countercase and numerical-resolution diagnostics; passing algebraic controls never relabel the negative modes as stable.

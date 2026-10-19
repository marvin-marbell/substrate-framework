# Volume-independent charged threshold of the native fixed-metric compact-U1 operator

This is a scoped original-source construction, not campaign completion. Governing plan: `issue-222-continuing-backward-20260930`, assignment revision `381993ecb362c8c67dbc334aebbcb6093b64a7b78111b5a9e785d4746b9bbe3b`; pinned general template `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`; work item `campaign`, requirements `outcome/verification`. The fundamental original CAR/quantum-gauge law is an assumption, not an Euler derivation of quantization.

## 1. Exact claim, operator, source and exclusions

Fix a finite connected periodic nearest-axis graph, `N>=5`, `V=N^3`, `a>0`, `m>0`, `r>=0`, finite `v`, and a fixed positive-definite real full native Gram matrix `G`. Retain its actual Clifford fields

\[
 \Phi_\alpha=(\sqrt G)_{\alpha\nu}\Gamma_\nu,
 \quad\{\Phi_\alpha,\Phi_\nu\}=2G_{\alpha\nu}I_4.
\]

There is no isotropic replacement, additional CAR selector, or angular/rotor/Fock cutoff. The consumer obtains `G` from `original_charged_principal.original_frame(..., native_toron=True)` in all ten symmetric coordinates and uses `BulkQuantumLaw.onebody` only for the unchanged original endpoint law. That helper's additional 27-selector is not called. For every compact link configuration, including every retained central seam class, let `T_j(A)` be the actual unitary shifts and set

\[
 W=\sum_j[I-(T_j+T_j^\dagger)/2]\ge0,\quad
 S_j=(T_j-T_j^\dagger)/(2i),\quad L=mI+(r/a)W,
\]
\[
 h(A)=\Phi_0L+{v\over a}\sum_j\Phi_jS_j.
\]

These are the native onsite coefficient `m+3r/a` and directed endpoint `F_j=-(r Phi0+i v Phi_j)/(2a)`, not a Euclidean Wilson replacement. Define

\[
 b=G_s^{-1}G_{s0},\quad s=G_{00}-G_{0s}G_s^{-1}G_{s0}>0,
 \quad\beta={\Phi_0-b_j\Phi_j\over\sqrt s},\quad\mu=m\sqrt s.
\]

The closed effective form is precisely

\[
 H=u_e\sum_e p_e^2+K\sum_p(1-\cos\operatorname{curl}_p A)
       +V_0(A)+d\Gamma(h(A)),\quad p_e=-i\partial_{A_e},
\]

on the whole periodic `H1(T^(3V), F_(2V+q) tensor R_q)`, with **every configuration** of the conserved original CAR sector retained. Here `u_e>0`, `K>0` are arbitrary fixed finite positive coefficients. `V0` is a common real smooth bounded gauge-invariant scalar; `V0=0` includes the actual source. `R0=C`; `R_+` is the actual opposite static U1 charge `-1` and SU4 `bar4` reference at `R`; `R_-` has U1 `+1` and SU4 `4`. There is no extra reference Hamiltonian or reference self-energy subtraction. Write `E_q` for its physical ground infimum.

**Theorem for this defined effective operator:**

\[
 \boxed{E_{+1}-E_0\ge\mu,\qquad E_{-1}-E_0\ge\mu.}
\]

The exponent, unlike the previous sufficient Born-Huang bound, has no extensive vacuum cost and does not require decreasing `u_e` with `V`. For a family whose actual metrics change with `N`, the formula is `mu_N=m sqrt(s_N)`: a positive lower bound on `inf_N s_N` is an additional metric-family premise, not supplied by one `N5` floating receipt. At fixed metric, the bound is volume independent. Prefactors for polynomial correlators may depend on the finite graph and operator, but not on Euclidean time.

This is not an upper bound, separation/deconfinement theorem, localized mechanical spin result, photon pole, thermodynamic construction, UV limit, or electron-photon campaign closure. Parent-owned full-Gauss BO transfer and its errors remain distinct. In particular a non-scalar retained colour/light angular operator cannot be silently treated as `V0`.

## 2. Full original Gauss domains and centre offsets

Use exactly the original background of two light particles per site:

\[
 \Psi(A+d\chi)=e^{-2i\sum_x\chi_x}\Gamma(g_4)
                   e^{-iq\chi_R}\Psi(A),\quad g_x=e^{i\chi_x},
\]
\[
 (\operatorname{div}p)_x=n_x-2-q\delta_{xR}.
\]

The summed constraint gives `NL=2V+q`; the charged graph including its reference has zero total Gauss charge. The charged reference does not allow an unreferenced charged periodic sector. Smooth Wilson-string CAR insertions produce nonempty physical charged domains, as in `original_compact_charge_bound.md`; their strings have modulus one.

The regular original full-field/flat-link colour orbit and original whole-heavy `NH=2V` representation are retained when lifting this body-frame form. Their common central characters are `z^(NH+NL)=z^(4V)=1` in the neutral sector and `z^(4V+1-1)=1`, `z^(4V-1+1)=1` with the actual `bar4/4` reference in charged sectors. All fifteen/site colour generators, light representation, heavy line and reference occur in that lift. This is **not** a claim that their original finite-parameter Cartesian/angular energy is zero. The theorem assumes the effective body-frame form stated in section 1 after the parent's full-domain transfer; algebraic lifting alone does not establish that transfer.

The temporal determinant argument below is for the specified compact-U1 **body-frame effective form**. It does not extend to an independent arbitrary noncentral temporal SU4 gauge field: such a connection need not commute with `beta`. Colour Gauss is imposed through the actual regular-orbit equivariant lift, rather than by silently applying the U1 coercivity proof to a noncommuting colour connection. Restoring a finite non-scalar colour angular kinetic operator would require its own argument; the parent's full original-field BO transfer controls that boundary.

Every centre well can have its own `G_theta`, `mu_theta`, scalar potential and common heavy offset `c_theta`. Applying this theorem in each retained block gives

\[
 \min_\theta(E_{q,\theta}+c_\theta)
 -\min_\theta(E_{0,\theta}+c_\theta)\ge\min_\theta\mu_\theta.
\]

No centre selector or equal-well assumption is introduced. A microscopic lower/trial transfer with errors `epsilon_q, eta_0` loses those errors from this bound; the normal/complement transfer is not proved by this consumer.

## 3. Coercivity for every time-dependent compact gauge background

Exactly `beta^2=I`, `{beta,Phi_j}=0`, and

\[
 h=\beta M+D,\quad M=\sqrt s L\ge\mu I,\quad
 D=\sum_j\Phi_j[b_jL+(v/a)S_j],\quad D=D^\dagger,
 \quad\{\beta,D\}=0.
\]

There is a fixed Hermitian `J=Gamma0 Gamma1 Gamma2 Gamma3`, `J^2=I`, anticommuting with every `h`. Neither `[M,D]=0` nor independent isotropic copies are assumed.

Let `T` denote inverse temperature, not the matrix `beta`. On the continuous time circle take anti-periodic fermions and arbitrary bounded measurable spatial link paths and real site temporal gauges `a0_x(t)`, repeated identically over all four colours. Arbitrary unitary site-gauge kicks are also admitted as interface conditions. Define

\[
 \mathcal D=\partial_t-i a_0(t)+h(A(t)).
\]

`beta` commutes with the temporal connection and with all site-gauge kicks. The derivative with these anti-periodic/unitary interface conditions is anti-Hermitian. Consequently, on its first-order Sobolev domain,

\[
 \operatorname{Re}\langle u,\beta\mathcal D u\rangle
 =\langle u,M u\rangle\ge\mu\|u\|_2^2.
\]

For a real periodic Lipschitz weight `f`, `|f'|<=gamma<mu`,

\[
 \mathcal D_f=e^f\mathcal D e^{-f}=\mathcal D-f',\quad
 \operatorname{Re}\langle u,\beta\mathcal D_f u\rangle
 \ge(\mu-\gamma)\|u\|_2^2.
\]

On a finite spatial graph these first-order circle operators have index zero: the elementary variation-of-constants solution reduces existence to the finite monodromy boundary matrix. Coercivity excludes its homogeneous anti-periodic solution, so that matrix is invertible. Thus

\[
 \|\mathcal D_f^{-1}\|_{2\to2}\le(\mu-\gamma)^{-1}.
\]

A site temporal connection need not have a small norm. Its contribution to the norm evolution is unitary; the bound depends on `h`, not on derivatives of a Brownian gauge path or on `a0`. Unit gauge kicks preserve this norm statement. Backgrounds induced by continuous photon Brownian paths are covered by bounded measurability of `h` and unitary temporal interfaces. There is no centred temporal difference, time doubling or changed Hamiltonian regulator here.

## 4. A genuinely pointwise, uniformly weighted Green bound

The preceding `L2` estimate alone is **not** a pointwise-kernel estimate. Here is the missing evaluation step. A uniform one-body norm bound is

\[
 B=\sqrt{G_{00}}(m+6r/a)+{|v|\over a}\sum_{j=1}^3\sqrt{G_{jj}},
 \qquad\|h(A(t))\|\le B,
\]

independent of `V`, background and time period. Put `k=B+gamma`. On any source-free time interval, a solution of `D_f v=0` obeys

\[
 e^{-k|t-u|}\|v(u)\|\le\|v(t)\|\le e^{k|t-u|}\|v(u)\|.
\]

The same bound holds for the homogeneous adjoint evolution in the source variable of its Green kernel. This follows by differentiating the squared norm: the `ia0` part cancels, while the Hermitian part of `-h+f'` has norm at most `k`. Measurable coefficients follow by integral evolution and approximation, and unitary kicks do not change the inequality.

Fix `delta>0`, `T>=8 delta`. For two time points at circle distance at least `2 delta`, choose a length-`delta` evaluation interval next to each, not containing the other source. For a source `g` supported in the second interval, the first homogeneous evaluation estimate and the `L2` inverse bound give

\[
 \|v(t)\|\le {e^{k\delta}\over\sqrt\delta(\mu-\gamma)}\|g\|_2.
\]

Duality says the Green row restricted to the source interval has the corresponding operator-to-`L2` bound. Applying the homogeneous adjoint evaluation estimate to that row at `s` then gives

\[
 \|K_f(t,s)\|\le {e^{2k\delta}\over\delta(\mu-\gamma)}.
\]

The intervals can be taken towards or away from the points as needed; `T>=8 delta` leaves an interval on both arcs. For points closer than `2 delta`, move the evaluation point to circle separation `2 delta` on its **same side** of the source and propagate back without crossing the jump. The additional propagation costs at most `e^(2k delta)`. Both one-sided diagonal limits are included by this construction. In particular

\[
 \|K_f(t,s)\|\le C_{\gamma,\delta},\quad
 C_{\gamma,\delta}={e^{4(B+\gamma)\delta}\over\delta(\mu-\gamma)}.
\]

The distributional jump is `K(s+,s)-K(s-,s)=I`; we do not attempt to evaluate a delta distribution in an `L2` norm. Equal-time Wick contractions use these one-sided limits, not a time-lattice inverse.

Take the periodic Lipschitz weight `f(u)=gamma d_T(u,s)`, where `d_T` is shortest circle distance. Its derivative is bounded by `gamma` almost everywhere and `f(t)-f(s)=gamma d_T(t,s)`. Since `K_f=e^f K e^-f`,

\[
 \boxed{\|K(t,s)\|\le C_{\gamma,\delta}e^{-\gamma d_T(t,s)}.}
\]

For the zero-temperature argument `delta` is fixed first and `T` tends to infinity. The constant is independent of `T`, `V` and the entire time-dependent compact background. It is deliberately conservative; a loose finite prefactor does not weaken its spectral exponent. `gamma` may approach `mu` only after the spectral argument, not by substituting `gamma=mu` into this constant.

## 5. Positive physical Hamiltonian determinant, including background two

For the exact continuous-time one-body evolution

\[
 U=\mathcal T\exp\int_0^T[-h(t)+i a_0(t)]dt,
\]

one has `U^dagger J U=J`, since the generator `X` obeys `X^dagger J+JX=0`. Also `tr h=0`, so

\[
 \det U=e^{4i\int_0^T\sum_xa_{0x}dt}.
\]

These formulas retain time ordering; replacing `U` by `exp(-integral h)` is not valid. Because `J U J=(U^dagger)^-1`,

\[
 \det(I+U)^*=(\det U)^{-1}\det(I+U).
\]

The physical background-normalized determinant is therefore real:

\[
 Z_F[A,a_0]=e^{-2i\int\sum_xa_{0x}dt}\det(I+U)\in\mathbb R.
\]

Reality is not yet positivity. A zero of `det(I+U)` would give a nonzero homogeneous anti-periodic solution of `D u=0`, excluded by section 3. Deform

\[
 h_\lambda=(1-\lambda)h+\lambda\mu\beta,\qquad0\le\lambda\le1.
\]

Every member retains fixed chirality and `Re(beta D_lambda)>=mu`; hence the real normalized determinant stays nonzero. At `lambda=1` all site temporal gauges commute with `mu beta`. With `theta_x=integral a0_x`, including any temporal holonomy kicks, its normalized determinant is exactly

\[
 \prod_x[2\cosh(T\mu)+2\cos\theta_x]^2>0.
\]

Thus `ZF>0` on **every** admitted background, not merely on zero-temporal-gauge configurations. The massless `mu=0`, `theta=pi` endpoint is a genuine zero and explains why strict positive mass is essential. An omitted background factor is generally complex and is not a positive fermion measure.

For photon dynamics use the exact positive heat kernel of `u_e sum p_e^2`, the real potential factor, and exact fermion factors `exp(-dt dGamma h(A))`. The Hamiltonian Trotter limit has the Brownian path representation. At finite splitting the corresponding one-body evolution consists of constant-Hamiltonian segments and unitary site-gauge interfaces; sections 3–5 apply there too. This is a regulator of the **actual** Hamiltonian, not an imported Euclidean Wilson action. Trotter convergence follows on this finite compact graph from bounded matter/potential perturbations of the photon Laplacian. The finite-dimensional full CAR trace is exactly `det(I+U)`, with no Fock omission.

A Haar Gauss projection produces endpoint gauge twists of the bosonic heat paths and the fermion evolution. Spreading a twist over the time interval by a site gauge transformation makes the bosonic configuration periodic and produces the temporal connection above. `beta` and `J` commute with that transformation. The factor from the original background-two Gauss projector is exactly the normalization displayed above. The heat-kernel/potential weight is nonnegative and `ZF` is strictly positive, so the physical projected neutral fermion-photon measure is nonnegative. No probability measure is assigned to the unnormalized complex determinant.

## 6. Canonical `NL=2V` neutrality, not a grandcanonical substitution

In the full local Haar projector retain its constant-site component `g_x=e^(i theta)`. It leaves all photon links unchanged and multiplies every one-body mode by `z=e^(i theta)`. The background factor is `z^(-2V)`. Algebraically

\[
 \det(I+zU)=\sum_{n=0}^{4V}z^n\operatorname{Tr}_{\wedge^n\mathbb C^{4V}}U,
\]
\[
 \int_0^{2\pi}{d\theta\over2\pi}\,z^{-2V}\det(I+zU)
 =\operatorname{Tr}_{\wedge^{2V}\mathbb C^{4V}}U.
\]

Additional local twists give exactly the remaining local Gauss constraints. The integrand **including** the background/temporal factor is positive by section 5 for every global holonomy, so this canonical projection preserves a nonnegative measure. It is not permissible to omit this integral and call a grandcanonical determinant a fixed-number neutral trace. In the consumer the exact Fourier projection differs substantially from the grandcanonical value and agrees with the whole exterior-sector trace; both controls are necessary.

With a charge-`q` insertion during an interval, the compensating static reference supplies its original temporal Wilson factor and opposite SU4 representation. Its U1 worldline and spatial Wilson string factors have modulus one; its finite-dimensional colour transport is unitary. The total Gauss charge remains zero, while the light sector on that interval is exactly `2V+q`. Summed Gauss excludes using the same correlation without the opposite reference.

## 7. Physical charged correlators and the neutral-ground limit

Let `B` be any smooth bounded gauge-covariant operator from the neutral physical sector to a charged/reference sector, polynomial in the original CAR generators, with smooth trigonometric link multipliers and actual Wilson transports. Its fermion polynomial is homogeneous of net light charge `q=+1` or `-1`. Neutral CAR excitations of arbitrary finite order are allowed; do not restrict to a bare single creator.

The exact canonical/projected finite-temperature correlation is

\[
 C_{T,B}(t)={\operatorname{Tr}_0[e^{-(T-t)H_0}B^\dagger e^{-tH_q}B]
                    \over\operatorname{Tr}_0 e^{-T H_0}},\quad0<t<T.
\]

Its Hamiltonian path representation is the section-6 positive neutral determinant measure with the inserted reference/string factors and the exact fermionic Wick minors. All equal-time contractions use one-sided `K` limits. In each Wick term at least one fermion line crosses between the two insertion times: the excess of creators over annihilators at one endpoint is `q`, and internal same-time contractions cannot remove that excess. Every such line satisfies section 4. Every other contraction is bounded by the same time-independent `C_(gamma,delta)`, while the bounded link functions and unitary reference/string factors contribute only a finite operator-dependent constant. A finite polynomial has finitely many Wick terms. Therefore, for `T>=8 delta`,

\[
 |C_{T,B}(t)|\le C_{B,\gamma,\delta}e^{-\gamma d_T(t,0)}.
\]

Canonical and local Haar projections cause no absolute-value cancellation debt: the denominator is the integral of the **positive normalized** determinant weight, and the bound holds pointwise for every temporal holonomy before averaging. The reference during the interval is a unit-modulus insertion, not a replacement for this positive neutral denominator.

At fixed finite graph the physical elliptic operator has compact resolvent and a finite-dimensional neutral ground eigenspace, of dimension `d0>=1`. For fixed `t`, take `T->infinity` first. The left side converges to the nonnegative ground correlation

\[
 C_B(t)={1\over d_0}\sum_{i=1}^{d_0}
 \langle B\psi_i,e^{-t(H_q-E_0)}B\psi_i\rangle
 \le C_{B,\gamma,\delta}e^{-\gamma t}.
\]

There is no assumption of a unique neutral vacuum and no exchange of thermal and thermodynamic limits.

## 8. Dense charged operator range and spectral threshold

A bound on one Wilson-dressed creator would not imply the charged ground threshold. Dense coverage is the final needed step.

First take analytic `V0` (the actual magnetic cosine source is analytic). The neutral ground eigenvectors are real analytic on the **full** compact link torus by elliptic analytic regularity of the scalar-principal-part matrix Schrödinger equation. This is not a regularity assertion on a singular gauge slice. A nonzero vector eigenfunction has at least one nonzero analytic component; that component's zero set has measure zero. Hence any chosen nonzero neutral ground `psi` has `||psi(A)||>0` almost everywhere.

The original finite CAR algebra contains every matrix unit from its entire `NL=2V` sector to its entire `NL=2V+q` sector. Concretely `|I><J|` is a finite product of creators/annihilators and occupation/empty projectors, homogeneous of net particle number `q`; it is not a newly added species. Tensor these with all four actual reference vectors. A Fourier link coefficient of such a matrix unit transforms with the difference of the two site occupation characters and the opposite reference character. Its total character is zero because `(NL_target-NL_source)-q=0`. On a connected graph any such integer site character is an integer divergence of edge flows, so actual Wilson paths supply it. Remaining Fourier factors are gauge invariant loops. Thus gauge averaging Fourier matrix units yields a complete family of smooth covariant CAR/Wilson polynomial operators `B` of the section-7 kind. The full regular colour-orbit lift retains every light/heavy/reference factor; its common centre character is trivial as in section 2, so there is no additional lost charged fibre.

For an arbitrary charged physical `eta`, consider the covariant pointwise map

\[
 A_\epsilon(A)={\eta(A)\psi(A)^\dagger\over\|\psi(A)\|^2+\epsilon}.
\]

After bounded truncation it is a bounded measurable covariant matrix map, and `A_epsilon psi -> eta` in `L2` by dominated convergence because `psi` is nonzero almost everywhere. Continuous, then Fourier-polynomial, covariant maps approximate it in the weighted `L2` norm obtained by acting on `psi`; Haar averaging preserves covariance and convergence. The matrix-unit/Wilson family above realizes those approximants. Hence

\[
 \overline{\operatorname{span}\{B\psi:B\text{ as above}\}}=\mathcal H_q^{\rm physical}.
\]

For any `gamma<mu`, the positive spectral measure of each `B psi_i` relative to `Hq-E0` has no support below `gamma`: a positive mass in an interval with upper endpoint `<gamma` would contradict its exponential correlation bound as `t->infinity`. The same bound holds for finite linear combinations because sums of the corresponding `B` are again admissible polynomials. Dense coverage therefore makes the spectral projection below `E0+gamma` zero on the whole physical charged Hilbert space. Let `gamma` increase to `mu` to obtain `E_q-E0>=mu`.

For smooth real `V0`, approximate it uniformly by real analytic gauge-invariant Fourier polynomials (average over the gauge group). Bounded perturbation changes each sector's ground infimum by at most the uniform potential error. Applying the analytic theorem and letting the error tend to zero proves the same bound for the stated smooth `V0`. No unproved nonvanishing claim for a nonanalytic system is needed.

## 9. Native consumer, positive/wrong controls and exact scope

`original_uniform_charge_gap.py` exposes `run`, `tree_onebody`, `exterior_hamiltonian`, `tree_quantum_controls`, `tree_correlator_controls`, `time_evolution`, `determinant_controls`, and `canonical_controls`. CLI:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
 original_uniform_charge_gap.py --size 5 --centre 2 2 2 --time-segments 4
```

`--all-centres` evaluates all 64 source Gram/Schur formulas without choosing a well. `--spacing .001` uses the actual original cutoff family. The full torus time controls use the native full `4V` matrix, arbitrary compact spatial phases in every declared time segment, independent site temporal gauges, exact matrix-exponential segment evolution, exact monodromy inverse, weighted pointwise kernel checks and the fixed source beta/chirality. These are continuous-time calculations for the specified piecewise-constant backgrounds, not a sampling claim about quantum photons.

The decay envelope alone cannot identify a Dirac inverse. Independent review found that a constant identity substituted for all native Green samples still passed it: the actual native/cutoff smallest logarithmic slacks were4.94613/11.89527. Parent reproduced that false-positive using the same numerical assertion. The producer now exposes the **same computed matrices** at the one-sided source, internal times and full-period endpoint. It separately verifies the identity equal-time source jump, antiperiodic boundary and homogeneous evolution with the actual time-ordered prefixes; the unchanged conservative decay bound is applied only afterwards. A constant identity must fail this physical source-jump assertion. The regression independently compares its midpoint kernel with exact many-body time evolution over every one of the256 original two-site CAR states, and also exercises a wrong multiplication order that preserves the source jump but violates evolution. This conditional Gaussian Wick identity is checked before the canonical/Gauss average, not called a grandcanonical physical neutral state.

The exact quantum tree consumer has four original colours at each of two sites, all `NL=3,4,5` occupation configurations (dimensions `56,70,56`) and the opposite static reference at site two. Tree Gauss gives `E=n_left-2` exactly; there is **no free loop flux to truncate**. Its Hamiltonian is the native tree endpoint plus `u_e E^2`. A tree has no plaquette, so it does not test magnetic dynamics or periodic 3D photon ground states. The SU4/reference lift and its fourfold zero-Hamiltonian spectator multiplicity are retained algebraically, not claimed as a quantum colour-field diagonalization. Native full-CAR matrix units on a nonzero neutral pivot cover all `56*4=224` charged/reference body-fibre states; omitting the reference fails summed Gauss.

The canonical control uses 17 global holonomy nodes, more than the eight-mode polynomial degree, so its `NL4` Fourier coefficient is exact up to floating arithmetic and agrees with the actual sum of all four-by-four principal minors. The physical electric-time control uses exact full-sector Hamiltonian Strang products: its electric factor is precisely the Gaussian temporal-gauge average `E exp(i sqrt(2 dt u_e) Z (n_left-2))`. It compares periods with 2/4/8/16 steps against the exact full quantum tree partition and reports the controlled finite-splitting discrepancy. Individual kicked fermion paths retain positive normalized determinants. No temporal-difference action is substituted.

Wrong controls reach the same assertions: omitted background two fails real-positive determinant phase (the background is deliberately shifted to phase `pi/2` so this is not a small-effect accident); omitted opposite reference fails global Gauss; `mu=0` with pi temporal holonomy fails the no-zero determinant assertion; unprojected grandcanonical trace fails equality with the actual canonical half-sector trace; and single-creator-only range has rank eight rather than the required 56. The tested repair to the last restriction uses explicit original CAR words and occupation projectors, obtaining rank 56 (224 with all actual reference vectors) in both charged sectors. These are adverse causes the proof genuinely uses, not unrelated diagnostics.

The receipt binds SHA256 of this source/proof and all loaded local source/proof dependencies and records native interpreter/library versions, BLAS settings, actual Gram/beta/couplings, background segments, tree spectra, Fourier projection and finite-splitting rows. Numerical tolerances are floating controls, **not interval certificates**. The exact full-operator proof above is logically distinct from every finite probe. Native Pyright stdio symbols/definition and diagnostics with the scientific interpreter, and mounted annotated Python AST search, were exercised before substantive edits; the mounted LSP device itself is unavailable. Final native types/CLI are integration-owner checks.

Observed native **Python API** design probes, with all three BLAS thread variables set to one, source SHA256 `76b421bc72cee5a1f06fc1ab8d2dd8dc1e4e279507e4b470f3f0eea57d8291cc`: `run()` (`N5,a1,centre222`) naturally exited zero in 4.07 s with every check true, `mu=.1294087392384287`, determinant phase error about `2.5e-15`, nonzero actual `[M,D]` norms `.22056..23090`, and omitted-background phase defect `sqrt(2)`. The full tree gaps at `UE=0,.001,.1,1,100,1e6` were `.1335968635,.1340697171,.1578148005,.1677701775,.1682305054,.1682313604` (minimum of the two charges). At `UE=1`, the physical left-creator/right-reference correlation was `.5,.1566583589,.00597476158,.0000834074595` at times `0,1,4,16`; the active charged threshold was `.16777017746` in both sectors. The exact canonical half trace was `55.4763181348190`, versus grandcanonical `204.709709281273`; the 2-to-16-step physical splitting error fell from `.000446513725` to `.00000709926550`.

The adverse full-metric cutoff probe `run(spacing=.001, centre=(1,1,1), all_centres=True)` naturally exited zero in 4.56 s, every check true; raw bound output is `artifact://1661`. It retained all 64 actual Gram formulas, minimum `mu=.10728067118449076`, selected-centre `mu=.10732736444155277`, all three native `G0j=.0001478388463` and nonzero spatial cross entries. Its full 500-dimensional time backgrounds had `[M,D]` norms `233851..244808`, normalized determinant phase error about `5.0e-16`, and the same `sqrt(2)` omitted-background defect. The exact canonical trace `48.3346642036663` differed from grandcanonical `181.998702616892`; physical 2-to-16-step splitting errors fell from `.000530030041` to `.00000840671379`. The massless pi-holonomy wrong evolution had anti-periodic boundary singular value `1.22e-16`, versus `.0002146316922` after the positive-mass repair. These probes bind the proof before this evidence-only appendix; the integration owner refreshes final source/proof/CLI receipt bindings. No final type check or CLI execution is claimed here.

No full periodic quantum EM diagonalization, charged upper bound, physical-spin theorem, thermodynamic deconfinement, interacting UV construction or original-source BO conclusion follows from this lower theorem alone.

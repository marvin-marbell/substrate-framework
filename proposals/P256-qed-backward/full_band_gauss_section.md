# Full occupied-band two-cell Gauss section

This construction replaces neither the active continuous-Haar/photon Hamiltonian nor its executed receipts. It constructs a field/link-dependent **trial section**, on the actual two-cell primitive matter fibre with 16 modes, $N_H=N_L=4$, and dimension $\binom84^2=4900$. It allows local number fluctuations and the complete heavy and light one-axis hopping. It is not a solved field ground state, full-field spectral calculation, phase, Lorentz theorem, mechanical-spin construction, canonical QED derivation, or three-dimensional scattering calculation.

## Actual primitive fibre and initial section

The immutable `PrimitiveMatter` owner orders modes as LH, LL, RH, RL (four each). All number, hopping and colour actions use its actual `PauliCode.transfer` matrix elements. The product of two normalized local epsilon singlets is embedded by **actual marker-word creation amplitudes** from `boundary_word` and `code.element`; no Slater-minor signs, canonical ladder matrices or replacement fermion representation enter. The CLI checks all fifteen local colour generators annihilate that primitive product, separately at each site.

For each traceless Hermitian $\Phi_{x0}$ with a separated middle gap let $p_x$ be its bottom-two projector. On the local two-light sector,

$$\Omega_x(\Phi_{x0})=\sqrt6\,\Pi_{p_x}S_x,
\qquad \Pi_{p_x}=N_{p_x}(N_{p_x}-1)/2.$$

This is precisely the existing paired section: two light orbitals in $p_x$ and two heavy orbitals in its complement. The code applies these local light-number polynomials to the actual globally embedded epsilon-singlet product, retaining its Pauli phases. Call the resulting normalized state $\Omega_0$. It obeys $N_{Lx}=N_{Hx}=2$ initially, is EM-neutral with the local background subtraction two, and is SU4-covariant, not invariant at a fixed nonzero field.

## Same-owner onebody coefficient law

`scalar_paths(RotationBand)` supplies the scalar coefficients rather than a duplicated guessed Wilson law. As in `PhysicalHamiltonian`, two inactive axes are subtracted from the onsite scalar. For an unimproved band this leaves $m+r/a$; the sole directed x path has coefficients $(-r/(2a),-iv/(2a),0,0)$ in the selected four-label basis. Each species may have its own band parameters.

For species $s$, set $F_{xs}=\sum_\mu t_{s\mu}\Phi_{x\mu}$ and

$$T_s(\Phi,U)=\frac12(F_{0s}U+UF_{1s}),\qquad
f(A)=\frac{e^{iA_0}+e^{-iA_1}}2.$$

Then

$$h_H=\begin{pmatrix}c_H\Phi_{00}&T_H\\T_H^\dagger&c_H\Phi_{10}\end{pmatrix},\qquad
h_L=\begin{pmatrix}c_L\Phi_{00}&fT_L\\\bar fT_L^\dagger&c_L\Phi_{10}\end{pmatrix}.$$

The physical whole-heavy operator is $-\Lambda\,d\Gamma_H(h_H)$, with **the same negative multiplier on onsite and hopping**. It is EM-neutral. The physical light operator is $d\Gamma_L(h_L)$. Both use one genuine $U\in SU(4)$, not averaged colour charges or independently chosen endpoint links.

At fixed gauge transformations $g_0,g_1$,

$$\Phi_{x\mu}\mapsto g_x\Phi_{x\mu}g_x^\dagger,\quad U\mapsto g_0Ug_1^\dagger.$$

Under EM transformations $\theta_0,\theta_1$, $A\mapsto A+(\theta_0-\theta_1)(1,-1)$, so $f\mapsto e^{i(\theta_0-\theta_1)}f$. The heavy coefficient has no EM source dependence.

## Filled-plane projection, chart and derivative

Let $P_H$ be the **top-four** projector of $h_H$ and $P_L$ the **bottom-four** projector of $h_L$, requiring separated global middle gaps. On the actual fixed-four species sector, $N_P=d\Gamma(P)$ has integer spectrum $0,1,2,3,4$. Consequently

$$\Pi_P=\frac1{4!}\prod_{j=0}^3(N_P-jI)$$

is the rank-one filled-plane projector. Define

$$v=\Pi_{P_L}\Pi_{P_H}\Omega_0,\quad w=\|v\|^2,\quad\Omega=v/\sqrt w.$$

Require **strictly positive overlap weight** as well as the local and global gaps. The implementation rejects a chart with relative spectral gaps at or below its tolerance, or weight at or below its positive numerical tolerance; it does not invent another phase convention, substitute a local frozen sea or fall back to a different chart. Species polynomials commute. Their common image is the product of independent full global heavy/light determinant lines, in the actual Pauli representation. Hence the whole onsite-plus-hopping energies are the occupied onebody sums. The receipt checks these eigen-residuals, actual primitive density matrices $\langle c_i^\dagger c_j\rangle=P_{ji}$, and local number variances against $\operatorname{Tr}[P_{xx}-P_{xx}^2]$.

For any Hermitian perturbation $\dot h$, the occupied-empty Sylvester solution is

$$\dot P=V_eX V_o^\dagger+V_oX^\dagger V_e^\dagger,\qquad
X_{eo}=\frac{(V_e^\dagger\dot hV_o)_{eo}}{E_o-E_e}.$$

No denominators within either occupied or empty subspace are used, so degeneracy **within** a filled plane is allowed. The code differentiates each local section and both global projector polynomials with their ordered, noncommuting product rule. It then differentiates the normalization:

$$\dot\Omega=\dot v/\sqrt w-\Omega\,\operatorname{Re}\langle\Omega,\dot v/\sqrt w\rangle.$$

No finite-difference state derivative or eigenvector-phase alignment is used to define this derivative. Central differences along a combined arbitrary-field, genuine SU4 link path and both-source perturbation provide an independent check. The optional geometry receipt evaluates all 120 independent field directions (all60 per site) and both source directions at fixed colour link. It emits the quantum metric, $\mathcal A_i=-i\langle\Omega,\partial_i\Omega\rangle$ and curvature $2\operatorname{Im}\langle D_i\Omega,D_j\Omega\rangle$. It does **not** assume flat connection or cancellation of two EM-dependent independent seas.

## Genuine differential and finite Gauss covariance

The initial section, coefficient law and number-polynomial projectors transform covariantly. Their norm and projection weight are gauge invariant, so the normalized chart has the exact transformation law

$$\Omega(g\Phi g^\dagger,g_0Ug_1^\dagger,A)
=\widehat g_0\widehat g_1\Omega(\Phi,U,A),$$

and

$$\Omega(\Phi,U,A+(\theta_0-\theta_1)(1,-1))
=e^{i\sum_x\theta_x(N_{Lx}-2)}\Omega(\Phi,U,A).$$

The local epsilon singlet fixes the SU4 phase; the initial local neutrality fixes the EM background phase. These statements do not follow from a vanishing mean charge. For each site and every one of the fifteen actual normalized generators, the CLI perturbs **both all four local fields and the appropriate endpoint of U**, and compares the analytic derivative to $iQ_x^a\Omega$. It also verifies the two local EM derivatives against $i(N_{Lx}-2)\Omega$, and checks finite local SU4 and EM covariance in the actual 4900-dimensional matter action. Frozen-wavefunction Gauss, omitted-heavy colour charge, and erroneously EM-charged heavy-source controls must give resolved nonzero defects.

## Nonempty neighbourhood, not arbitrary-field positivity

There is an analytic nonempty-overlap neighbourhood for the aligned positive-mass beta-polar configuration used here. Suppose $\Phi_{x0}=b_0\beta$ with $b_0>0$, identical at both cells; the spatial selected field anticommutes with $\beta$, $U=I$, and the band has $m>0,r\geq0$. Write $\mathcal B=\operatorname{diag}(\beta,\beta)$. For either heavy or light onebody law,

$$\tfrac12\{h_s,\mathcal B\}=B_s\otimes I_4,\qquad
\lambda_{\min}(B_s)\geq b_0(m_s+r_s/a_s-r_s|f_s|/(2a_s))>0,$$

with $f_H=1$, $|f_L|\leq1$. The spatial hopping contributes zero to the anticommutator. This inequality excludes zero eigenvalues and, by scaling the hopping continuously to zero, gives four positive and four negative levels.

On the positive spectral subspace, the restricted beta operator $X_+$ obeys the Sylvester equation $h_+X_++X_+h_+=2B_+$, with both $h_+$ and $B_+$ positive. Its integral solution is positive. On the negative subspace the same argument gives $X_-<0$. Thus the positive plane projects injectively onto the global beta-positive plane and the negative plane projects injectively onto the global beta-negative plane. Equal dimensions make those projections invertible. The initial paired state fills beta-positive heavy orbitals and beta-negative light orbitals, so both occupied-plane overlaps, and therefore $w$, are nonzero. This is an overlap proof, **not** a Slater-minor sign prescription.

Separated spectra and positive weight persist on an open neighbourhood by continuity. Arbitrary small independent traceless-Hermitian deformations of all four fields at both sites and link/source deformations are allowed there; scalar Clifford anticommutators are not required. The CLI constructs a selected case and a deterministic genuinely non-gamma field/link/source case and requires a nonzero scalar-anticommutator defect for the latter. Its numerical receipts, not this note, determine whether those particular finite probes pass. No positivity claim is made for arbitrary fields or links.

The CLI additionally constructs a separated-spectrum **zero-overlap** counterexample: only temporal fields, eigenvalues $(-100,-99,-98,297)$ at site zero and $(-3,-2,1,4)$ at site one, with $A=(\pi,0)$. Light hopping cancels, and the bottom-four light plane occupies three modes at site zero and one at site one, orthogonal to the initial two-plus-two light sector. Local and global gaps are checked before requiring projection-weight rejection. This distinguishes overlap-chart failure from gap closure; numerical roundoff in $e^{i\pi}$ does not make the chart admissible.

## Full-coordinate packet trial family

Let $\mathcal C$ be the open gauge-invariant chart defined by the separated spectra and nonzero weight. The proved nonempty neighbourhood and its compact gauge orbit admit a nonzero smooth gauge-invariant packet $\chi$ supported within $\mathcal C$ (average a sufficiently small positive smooth bump over the compact gauge group). Choose bounded support in all120 field coordinates with finite positive widths, not a delta function imposing an orbit or on-site occupation. Then $\chi\Omega$ is a legitimate physical Gauss trial family on all60 real field coordinates **per site**, Haar colour link and two compact phases. On compact support away from gaps and zero weight, the analytic section and its derivatives are bounded, so the ordinary field and gauge kinetic form energies are finite. This establishes an admissible variational family, not the optimum packet, field energy spectrum, full phase, or a quantitative lower ground-gap claim. Some inactive y/z field directions are spectators of this one-axis matter law but remain genuine independent packet coordinates.

## Primitive source action and Ward receipts

The independent Peierls implementation contracts each endpoint coefficient with the actual `PrimitiveMatter.forward` bilinear and its adjoint, with path-specific factors $e^{iA_0}/2$ and $e^{-iA_1}/2$. First derivatives multiply these by $ib_0$ and $-ib_1$; contacts multiply by a second direction factor. It does not call the lifted onebody derivative to manufacture an agreement.

For each local charge $n_x=N_{Lx}$ and its signed gradient $g_x=\pm(1,-1)$,

$$J(g_x)=i[n_x,H_L],\qquad C(g_x,g_x)=-[n_x,[n_x,H_L]].$$

The CLI verifies independent current/contact actions against derivatives of the same onebody law, central first/second differences, and these source Ward actions on a deterministic full-fibre probe. All four selected source directions and all sixteen mixed contact combinations are checked. Heavy source actions are identically zero, whereas treating heavy local charge as an EM source produces a resolved hopping-current defect. These are operator/source identities, not a claim that occupied-band Berry curvature cancels or that integrated response equals a canonical photon scattering amplitude.

## API and invocation

Public API:

- `FullBandGaussSection(light_band, heavy_band=None, heavy_scale=3., chart_tolerance=1e-12)`.
- `onebody(fields, U, phases)` returns the unscaled heavy and light $8\times8$ Hermitian coefficient matrices.
- `state(fields, U, phases)` returns `Section`, including normalized actual state, initial section, raw projected state, weight, local/global spectra and projectors.
- `derivative(section, fields, U, phases, dfields=None, dU=None, dphases=None)` returns the exact normalized first derivative; $dU$ is the tangent of the caller's genuine SU4 path.
- `peierls_action(vector, fields, U, phases, left, right=None, species=1)` returns independent primitive first-current or mixed-contact action.
- `controls(label, fields, U, phases, geometry=False)` runs the scoped algebra/geometry probes and returns JSON-native receipts.

From the repository root, with a Python environment providing NumPy and SciPy, match the aligned photon consumer's bare coefficients and whole-heavy scale explicitly:

```sh
python proposals/P256-qed-backward/full_band_gauss_section.py --m .2 --r .12 --v .3 --stiffness 2 --heavy-scale 1 --geometry --output full_band_gauss_section_receipt.json
```

Omit `--geometry` for the smaller algebra receipt. `--m`, `--r`, `--v` and `--stiffness` select the existing bare band and frame supplier; the default band remains `m=.8,r=.2,v=.3,stiffness=2`. `--heavy-scale` changes the positive whole-heavy multiplier without changing the occupied projector choice. Source identities (including loaded local suppliers) are captured **before** primitive operator construction and selected-field optimization, rechecked before publication, and included in the JSON alongside actual parameters. The output is written only after all required assertions pass; NaN/Infinity are forbidden. No outputs or numerical successes are asserted in this note before parent-owned execution.

## Parent-owned execution and scoped review

The same-coefficient `--geometry` invocation completed naturally in 39.81 s
after the explicit floating-point number-operator dtype repair. The receipt
is `full_band_gauss_section_verified_receipt.json`, bound to source SHA256
`1bb48d986e83c5df3b707ad74337a2f4b0ec6a4495af90ba87b6a8e68eb366cb`,
with `m=.2,r=.12,v=.3,stiffness=2,heavy_scale=1`.

The selected and genuinely non-gamma cases have projection weights
`0.9432317718200429` and `0.93555222345218`; all global middle gaps exceed
`0.33399`. Across both cases, all thirty differential SU4 Gauss residuals
are at most `1.30e-15`, both EM residuals at most `3.63e-16`, finite
covariance residuals at most `1.72e-15`, and whole-species eigen-residuals
at most `8.19e-16`. The independently checked normalized derivative has
error at most `3.21e-11` at step `1e-4`; current/contact Ward residuals
are at most `6.25e-16`, with exactly zero heavy EM source action.

The 122-coordinate metric traces are `40.981669` and `40.423989`.
The non-gamma case has nonzero EM curvature `F01=9.68309e-6`, so a universal
paired Berry-cancellation assumption would be wrong here. Frozen sections,
omitted heavy colour charge and incorrectly charged heavy hopping have
resolved defects. The separated-spectrum zero-overlap chart is rejected,
independently of the gap-closure rejection.

The independent `ReviewFullBandGauss` source/derivation/receipt review found
no actionable defect in this scoped construction. It did not rerun the
scientific CLI. The older `full_band_gauss_section_receipt.json` remains
historical evidence from source `79bc3b0202153d1f20f5df4c75b6bac043d8d95f00bf8d7dcb20d5d5fe10c210`,
before the dtype repair; neither receipt nor review solves the full-field
ground, phase, photon scattering or mechanical-spin obligations.

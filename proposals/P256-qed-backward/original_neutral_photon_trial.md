# Actual coupled neutral Slater–photon variational supplier

Governing plan `issue-222-continuing-backward-20260930`, assignment revision `a406dca7a4cc642318250c028a763daa96fa66a5999da5dae0c96e53910c5c91`, pinned `general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`, work item `campaign`, requirements `outcome/verification`. This slice owns only this file and `original_neutral_photon_trial.py`. Scientific executions/receipts, shared plan and rolling model remain Main-owned. The Python consumer below has not been scientifically executed by this slice.

## 1. Primitive law, domain and result

Use the native BODY operator of `original_uniform_charge_gap.md` §§1–2, not its isotropic replacement. On every original fixed positive full ten-Gram `G`, define the actual four Clifford fields `Phi`, the Schur beta, `mu=m sqrt(s)>0`, native onsite and endpoints

\[
 h(A)=(m+3r/a)\Phi_0+\sum_{x,j}(F_j U_{xj}e^{iA_{xj}}|x\rangle\langle x+j|+\mathrm{h.c.}),
 \qquad F_j=-{r\Phi_0+i v\Phi_j\over2a}.
\]

Here `m=.2,r=.12,v=.3`, `U` includes the actual retained central seams, and `a>0`. The source gives `Re(beta h)>=mu`, a fixed chirality `J` anticommuting with both `h` and `beta`, rank `P_-(h)=2V`, and

\[
 B=\sqrt{G_{00}}(m+6r/a)+{|v|\over a}\sum_j\sqrt{G_{jj}},\qquad\|h(A)\|\le B.
\]

The complete neutral BODY law is

\[
 H=U_E\sum_e(-i\partial_e)^2+K F(A)+d\Gamma(h(A))+c_\theta,
 \qquad F(A)=\sum_p(1-\cos (\operatorname{curl}A)_p),
\]

on the whole fixed `NL=2V` exterior fibre over the full compact link torus with every local U1 Gauss equation and background TWO. `UE,K` are the original unspecified **fixed positive** symbols. `c_theta` retains the scalar heavy/normal/onsite well energy, including its true offset. No additional A-dependent scalar is silently removed: if a different source adds one, its flux-relative allowance must be added to the bounds below. Native onsite contributions to the light sea are retained in the exact normalization in §4, not dropped as a frozen vacuum. The colour/full-field extension is exactly the source BO lift in §8, not a new light-band projection defining a Hamiltonian.

There are explicit finite `C_Q,C_B` and a retained wrapping allowance `R_N` such that the actual neutral ground satisfies, **if `K>C_B`**,

\[
 \boxed{{\langle F\rangle_0\over V}\le
 {3\sqrt{2U_E(K+C_B)}+3U_E C_Q+2R_N\over K-C_B}.}\tag{1}
\]

Always also `F/V<=6`. This is an actual coupled-ground mean bound, not an imported small-mean premise. It can be vacuous for the source's unspecified coefficients; neither `K>C_B` nor smallness of (1) is asserted numerically. These conservative constants are intentionally very large. No phase, physical transverse pole, HC/heat bound, reflection positivity, isolated charged pole or campaign completion follows from a mean alone.

## 2. A globally smooth periodic complete sea, with the correct beta direction

Let `S` be a fixed orthonormal sitewise frame of beta **minus**, containing exactly two modes at every site. Set `P=P_-(h)`, `C=S†PS`.

For any orthonormal negative-subspace frame `R`, write `H_-=R†hR=-Q`, `mu<=Q<=B`. The restricted accretivity inequality is

\[
 -Q\beta_- -\beta_- Q\ge2\mu I,\qquad\beta_-=R^\dagger\beta R.
\]

The Lyapunov integral (no commutation assumption) gives

\[
 -\beta_-\ge2\mu\int_0^\infty e^{-tQ}e^{-tQ}dt
 =\mu Q^{-1}\ge(\mu/B)I.
\]

Thus `R†SS†R=(I-beta_-)/2 >= (1+mu/B)I/2`. `R†S` is square; its two Gram matrices have identical singular values. Consequently

\[
 C\ge c_*I,\qquad c_*={1+\mu/B\over2}>\tfrac12,
 \quad Z(A)=P(A)S C(A)^{-1/2}.\tag{2}
\]

`Z†Z=I`, `ZZ†=P`. The uniform gap and the positive principal square root make `Z` globally smooth and exactly `2pi`-periodic in **each** original link angle, including torons. There is no choice of eigenvector phases and no patch/gluing assumption. The complete normalized Slater

\[
 \Omega(A)=Z_1(A)\wedge\cdots\wedge Z_{2V}(A)
\]

occupies the entire true negative cluster, not two states artificially selected at each momentum. Its expectation is the actual `E_sea(A)=Tr P(A)h(A)`; excitations and every other whole-CAR vector remain in the original Hamiltonian.

### Every local Gauss equation and background TWO

For `g_x=e^{i chi_x}`, the actual link convention is `A'_xj=A_xj+chi_x-chi_(x+j)`. Let `g_-=diag(e^{i chi_x}I2)` on the fixed `S` columns. Native covariance gives

\[
 P(A+d\chi)=gPg^\dagger,\qquad
 Z(A+d\chi)=gZ(A)g_-^\dagger,
\]

hence the **actual** full Slater covariance is

\[
 \Omega(A+d\chi)=e^{-2i\sum_x\chi_x}\Gamma(g)\Omega(A).\tag{3}
\]

This is precisely background TWO, not a grandcanonical determinant substitution. A real gauge-invariant scalar photon vector times `Omega` belongs to the physical complete local Gauss domain. Infinitesimally `(div p)_x=n_x-2`. Chirality is sitewise, so `Tr P_xx=2` at every A; this statement concerns expectation, not a false claim that `Omega` is an eigenstate of each local number operator. The summed particle number is exactly `2V`.

### Zero Berry trace is derived, not assumed

Choose the beta-plus basis as `JS`, ordering plus before minus. Then `J` has offdiagonal identity blocks and `JPJ=I-P`. Therefore

\[
 P=\begin{pmatrix}I-C&Q_0\\-Q_0&C\end{pmatrix},
 \quad Q_0^\dagger=-Q_0,\quad [C,Q_0]=0,
 \quad Q_0^2=C^2-C.
\]

The frame in (2) is `Z=(Q_0,C)^T C^-1/2`. In `Tr Z†dZ`, the term before differentiating `C^-1/2` is

\[
 \operatorname{Tr}C^{-1}(-Q_0\,dQ_0+C\,dC).
\]

It is real: `C^-1 Q_0` and `dQ_0` are anti-Hermitian, and the trace of their product is real; `Tr dC` is real. The inverse-root term has real trace as well. But orthonormality makes `Tr Z†dZ` purely imaginary. It is consequently **zero in this global periodic frame**, for every edge and every A. This proof uses fixed chirality and beta, not a generic assumption about occupied bands.

## 3. Actual electric quantum metric, without an extensive per-edge constant

For the normalized complete Slater and the above zero Berry trace,

\[
 \|\partial_e\Omega\|^2
 =\operatorname{Tr}(\partial_e Z)^\dagger(I-P)\partial_e Z
 =\tfrac12\|\partial_e P\|_{\mathrm{HS}}^2.
\]

`dh_e` has exactly two rank-four endpoint blocks, hence rank at most eight. Occupied/unoccupied denominators have absolute value at least `2mu`, so spectral divided differences give

\[
 \|\partial_eP\|_{\mathrm{HS}}\le{\|\partial_eh\|_{\mathrm{HS}}\over2\mu},
 \quad \|\partial_eh\|_{\mathrm{HS}}^2
 =2\|F_j\|_{\mathrm{HS}}^2
 ={2(r^2G_{00}+v^2G_{jj})\over a^2}.
\]

Thus an explicit **per original edge**, all-A, no-volume bound is

\[
 \boxed{C_{Q,j}={r^2G_{00}+v^2G_{jj}\over4a^2\mu^2},\qquad C_Q=\max_j C_{Q,j}.}\tag{4}
\]

For any real normalized physical scalar `psi`, differentiation of the actual trial `Psi=psi Omega` gives the exact form identity

\[
 \langle\Psi,H\Psi\rangle-c_\theta
 =U_E\|\nabla\psi\|^2
  +\int|\psi|^2\left[K F+E_{\rm sea}(A)
       +U_E\sum_e\|\partial_e\Omega\|^2\right].\tag{5}
\]

The cross term vanishes by the derived Berry trace. The last term is present and bounded by `3V UE C_Q`; it is not zero, not folded into a frozen Slater energy, and not a projection of H. The native consumer independently differentiates actual onebody matrices and global polar frames and compares complete determinant overlaps with (4).

## 4. The sea is not frozen: exact static massive-action telescope

Use the actual nonlinear source `world-model/sources/issue222-native-nonlinear-fermion-action-construction.md` §§1–4, with its centered background, polarized auxiliary boundary conditions, all finite-radius terms and final periodic restoration. Set time-independent spatial A and `a0=0`; take `T->infinity` **at fixed N**. The finite full spectrum gives

\[
 \lim_{T\to\infty}-{1\over T}\log\det(I+e^{-Th(A)})
 =E_{\rm sea}(A)=-\tfrac12\operatorname{Tr}|h(A)|.
\]

Half filling is exact by chirality; the canonical ground has the same filled sea. This equality is used only to identify a static matrix function before constructing the physical trial, not to discard local Haar constraints in a coupled trace.

The source's exact mass normalization and telescope therefore give

\[
 E_{\rm sea}(A)=-2cV+\sum_x\sum_{R\ge R_0}p_R(x;A),
 \qquad c=(m+3r/a)\sqrt{s}.\tag{6}
\]

The sum is terminated/restored exactly at the full finite torus; writing an infinite positive envelope is only an overestimate. The `-2cV` term is the actual scalar onsite sea contribution. In particular neither this term nor the initial interaction is set to zero. The scalar well offsets remain in `c_theta` separately.

A usable specialization of the source constants is as follows. Use `T0=1/B`, `Ch=2 sum||Fj||`, `gamma=mu/4`, `alpha a=log(1+mu/(4Ch))`, `xi=mu/4`, `rho=mu/2`. These are estimates/representation choices, not physical law changes. Put

\[
\begin{aligned}
 C&=e^2\max((8/T_0+4(B+\gamma+\xi))/\rho,4),\\
 L_R&=\lceil R/(\alpha a)\rceil,
 &P_t&=\coth(\alpha a/4)^3(4/\gamma),\\
 J_v&=1/a+C_h,
 &S_R&=2(2L_R+3)^3(1+2RC_h/\gamma),\\
 A_R&=512\max(1,C)^4(1+J_vP_t)^2S_R
                 e^{-(R-1-2\alpha a)/2}.
\end{aligned}
\]

`R0=ceil(max(3,gamma T0/2))+ceil(2 alpha a)`. The source's two canceled-first-Born estimates give

\[
\begin{aligned}
 O&=(m+3r/a)(\Phi_0-\sqrt{s}\beta),\quad
 v_\alpha=\|O\|+C_he^{\alpha a},\\
 q_t&=\max(c,\log2/T_0,\gamma+8v_\alpha,\gamma+4/a),
 \quad M_t=q_t-c,\\
 B_t&=2^{20}(v_\alpha^2+a^{-2}+T_0^{-2}),\\
 E_R&=M_tA_R+2\sqrt{A_RB_t},\quad e_R=E_R+E_{R-1}
                  \quad(R>R_0),\\
 v_{R_0}&=V_0=M_t(4C+2)+B_t/(q_t-\gamma),\\
 e_{R_0}&=H_0=M_t[64\max(1,C)^3(1+J_vP_t)^2]+B_t/(q_t-\gamma),\\
 v_R&=e_R\quad(R>R_0).
\end{aligned}\tag{7}
\]

The positive-envelope total variation of the spatial Hessian of `p_R` is at most `e_R`, and its absolute value is at most `v_R`. The mass integral is finite because the canceled first Born term removes the false `1/s` tail; the interpolation integral is `integral min(A_R,B_t/u^2)du=2 sqrt(A_R B_t)`. Time-translation averaging of the source's density envelopes bounds the static energy Hessian per unit time by exactly these envelopes. Dominated convergence applies to each radius and then the summable tail; temporal wrapping tends to zero in the zero-temperature limit. Spatial wrapping does **not**. All original nonlinear Peierls contacts are included in (7).

This invokes the source-owned proved quasilocal/coercivity hypotheses. It does not extend them to arbitrary off-BODY Cartesian or noncommuting colour histories.

## 5. Compact all-angle flux bound, including its first-variation cancellation

A smooth tangent-only Maxwell Hessian estimate is not by itself an all-angle `1-cos` estimate. The following additional compact argument makes that transition and keeps the noncontractible remainder.

For a contractible cube, remove its original flat central connection by a local gauge; this changes neither its anchored density nor its Hessian. Use the rooted rectangular tree gauge on its actual compact links. Let

\[
 b_p=\operatorname{Arg} e^{i(\operatorname{curl}A)_p}\in[-\pi,\pi].
\]

Each remaining edge angle modulo `2pi` equals the sum of these principal plaquette angles on a rectangular filling with at most `A_R^fill=3(L_R+1)^2` plaquettes. Denote that real reconstructed local edge field by `q_x`. All its actual edge exponentials equal those of A after tree gauge. Nonzero lattice monopoles do not invalidate this statement: different fillings differ by `2pi` integers, and the physical edge exponents, not a fictitious noncompact Bianchi equation, are being reconstructed.

Taylor-expand each **actual compact interaction** along `t q_x`, `0<=t<=1`, starting at its flat value. The Hessian envelope is all-background, so no good-field cutoff is needed. Translation and Cauchy–Schwarz give

\[
 \sum_x\left|\int_0^1(1-t)D^2p_R(x;tq_x)[q_x,q_x]dt\right|
 \le\tfrac12 e_R[3(L_R+1)^2]^2\sum_p b_p^2.\tag{8}
\]

Indeed each reconstructed edge contains at most `A_R^fill` entries with coefficients `0,+1,-1`; for any pair of Hessian endpoints, Cauchy–Schwarz gives one filling factor, and translation summation of the positive endpoint envelope gives the second. This is the same translated positive-envelope accounting as the source's field-strength bound, not an absolute closed-walk expansion with an unproved hopping contraction.

The **summed linear term is zero**; this needs a reason beyond gauge invariance. The four real Euclidean Clifford generators admit a fixed quaternionic antiunitary `Theta` commuting with each generator and hence with every real full-Gram field and beta (`Cl(4,0)=Mat_2(H)`). Combining `Theta` with spatial inversion about the anchor sends

\[
 A_{xj}\longmapsto A_{-x-j,j},\qquad b_{x,jk}\longmapsto-b_{-x-j-k,jk}.
\]

The actual onsite and directed hopping transform correctly because `Theta Fj Theta^-1=Fj†`; inversion reverses edge orientation. Centered open cubes and the polarized beta boundary conditions are preserved. Each real anchored static interaction in (6), including the initial one, is invariant under this operation. For a spatially constant infinitesimal magnetic two-form its first variation is therefore zero. Equivalently, write its linear tree reconstruction as `sum_p l_(R,p) b_p`; the sum of these coefficients is zero for each plaquette orientation. Translation over every anchor on the periodic graph consequently cancels `sum_x Dp_R(flat)[q_x]` for **any** principal plaquette array. This uses no smallness, no unphysical branch-free curl and no vanishing-current claim on the finite torus. The noncontractible persistent-current response stays in the next paragraph.

Using `b^2 <= (pi^2/2)(1-cos b)`, (8) gives a sufficient explicit constant

\[
 C_B={9\pi^2\over4}\sum_{R\ge R_0}e_R(L_R+1)^4<\infty.\tag{9}
\]

Only contractible terms need this argument; extending its positive radius sum to all radii increases the bound harmlessly. A term whose cube covers a spatial cycle cannot be tree-reduced. Keep its complete oscillation instead:

\[
 R_N=2\sum_{R:\,2L_R+1\ge N}v_R,
 \quad
 \boxed{|E_{\rm sea}(A)-E_{\rm sea}(A_{\rm flat})|
       \le C_B F(A)+V R_N.}\tag{10}
\]

The initial `2V0` is included when its initial cube already wraps. The actual `A_flat` has its retained colour centre seams and zero EM phase; one does not require equality of flat sea energies between torons. At fixed a and a metric family with the source uniform Schur floor and bounded hopping, the remainder is polynomial times `exp(-const N)`; it is **nonzero**. For native N5 the deliberately huge initial localization cube usually already wraps, so this receipt cannot certify a small remainder or admission.

These are closed-loop/contractible gauge-invariant interactions of the **actual** massive action, not a positive fermion-loop gas. Static loops returning without winding pay (9); winding loops, torons and harmonics pay the complete (10) remainder. A flat nonzero toron has `F=0` but generally a different sea energy; dropping `R_N` would make (10) false.

### Analytic summation implemented by the consumer

The code does not numerically truncate an infinite radius tail. Let `eta=alpha a`,

\[
 S_* =2(2/\eta+5)^3(1+2C_h/\gamma),
 \quad A_* =512\max(1,C)^4(1+J_vP_t)^2S_*e^{(1+2\eta)/2}.
\]

Then `A_R<=A_* (R+1)^4 exp(-R/2)`. For all noninitial radii one may enlarge

\[
 e_R\le d_4(R+1)^4e^{-R/2}+d_2(R+1)^2e^{-R/4},
 \quad d_4=2M_tA_*e^{1/2},\quad d_2=4\sqrt{A_*B_t}e^{1/4}.
\]

Since `L_R+1<=(eta^-1+2)(R+1)`, (9) is bounded by the initial `H0(L0+1)^4` plus these power-eight/power-six positive series. The initial term and an overextended all-radius tail are both retained. `eulerian_sum` sums `sum_(R>=0)(R+1)^n exp(-lambda R)` by its exact Eulerian rational function; `tail_series` translates the polynomial to the first wrapping radius. Thus the consumer exposes finite conservative constants and the **entire** wrapping allowance, not a finite partial sum advertised as a certificate.

## 6. Physical real photon vector, theta upper and derived coupled mean

A product link theta vector is not U1-Gauss physical, and its product with `Omega` is not a permissible shortcut. Define instead the actual positive normalized invariant scalar ground `psi_k` of

\[
 H_{\rm ph}(k)=U_E\sum_e p_e^2+kF(A),\qquad k=K+C_B.
\]

At finite graph the scalar compact elliptic operator has a unique strictly positive ground; all gauge transformations preserve it, so it is real and gauge invariant. An explicit construction from the periodic theta is

\[
 \psi_k=\lim_{t\to\infty}
 {e^{-tH_{\rm ph}(k)}\Pi_{\rm gauge}\Phi_g
  \over\|e^{-tH_{\rm ph}(k)}\Pi_{\rm gauge}\Phi_g\|},
 \quad
 \Phi_g=\prod_e {\sum_{n\in\mathbb Z}e^{-gn^2/2}e^{inA_e}
                 \over(\sum_n e^{-gn^2})^{1/2}}.
\]

This is an exact primitive-law variational vector, not a frozen-normal approximation or a claim about an unmeasured mean. Positivity gives nonzero ground overlap and convergence. All electric integers and periodic images remain in `Phi`; the semigroup keeps every original angle/toron. The pure full and pure physical ground energies agree **only at this scalar step**, never as an identification of the full coupled ground with its unprojected counterpart.

The source theta inequalities give `sum_e <p_e²><=3V/(2g)` and each actual plaquette `1-<cos><=g`, including wrapping plaquettes. Therefore

\[
 E_{\rm ph}(k)\le3V\left[{U_E\over2g}+kg\right]
 \le3V\sqrt{2U_Ek},\qquad g=\sqrt{U_E/(2k)}.\tag{11}
\]

Use the **physical complete coupled trial** `Psi=psi_k Omega` in (5). Equations (4),(10),(11) give

\[
 E_0\le b_\theta+V R_N+3V\sqrt{2U_E(K+C_B)}+3VU_E C_Q,
 \quad b_\theta=c_\theta+E_{\rm sea,\theta}(A_{\rm flat}).\tag{12}
\]

Conversely, without restricting a single CAR vector, the pointwise full-sector inequality is `dGamma(h)>=Esea(A)`. The electric form is nonnegative. For any actual coupled neutral ground, (10) yields

\[
 E_0\ge b_\theta+(K-C_B)\langle F\rangle_0-V R_N.\tag{13}
\]

Subtract (13) from (12), and divide by `V(K-C_B)>0`. This derives (1), including both wrapping payments and all three metric/link factors. It is not obtained by plugging an assumed small pure/normal magnetic mean into a coupled inequality.

## 7. All64 actual wells, offsets and true weights

For every retained centre use its actual full Gram and its own constants, flat sea and scalar offset. Define `CB=max_theta CBtheta`, `CQ=max_theta CQtheta`, `RN=max_theta RNtheta`, and `bmin=min_theta(c_theta+Esea,theta(flat))`. The lower (13) holds with `bmin` on the full direct sum; use the actual well attaining **this** baseline minimum for the upper trial. One does not guess which bare well is preferred or declare equal sea energies/offsets. This proves the same bound for every ground mixture, including any true weight distribution; no weights are prescribed. A fixed effective excited state or heat ensemble is not replaced by the variational ground, and no individual well is deleted. CLI `--all-centres` computes the 64 actual Gram/constant diagnostics; it does not fabricate scalar well energies unavailable in that static consumer.

## 8. Fixed-a,N native slow-BO transfer of this bounded magnetic observable

Use exactly `original_slow_kinetic_bo.md` §§1,5–7: original entire `60V` fields, `3V` SU4+U1 links, whole `NH=2V`, full fixed light CAR, all15V Gauss and complete background2, and the different specified slow trajectory `chi=Lambda^-1/2`, `u4=.3chi`, `J=.2Lambda^5/4`, `K4=.7Lambda^5/4`. Admission remains `N>=max(5,N0(a))`, with N0 existence only. The EM-neutral A-independent full-Gauss isometry W intertwines the actual bounded multiplication F exactly. Its image residual is `d_Lambda->0`; full complement has lower `g_Lambda->infinity`. Fixed-N boundedness is `||F||<=6V`.

No effective isolated ground gap is necessary for the following direct estimate. Subtract the source's scalar `E_min`; denote its effective baseline minimum by `bmin`, let `d` bound the neutral residual, `g` the complementary lower. Set

\[
 U=VR_N+3V\sqrt{2U_E(K+C_B)}+3VU_EC_Q,
 \quad T=U+VR_N+4d.
\]

For a microscopic ground write `eta=W†Psi`, `s=||eta||²`. The source block error is at most `3d`; its upper transferred trial has energy at most `bmin+U+d`. The effective lower (13) and complement yield

\[
 (K-C_B)\langle\eta,F\eta\rangle
 +(g-b_{\min}+VR_N)(1-s)\le T.
\]

Because `WF=FW`, P and its complement reduce F. Thus, whenever the last positive coefficient is indeed positive,

\[
 {\langle F\rangle_{\rm micro}\over V}
 \le {T\over V(K-C_B)}+{6T\over g-b_{\min}+VR_N}.\tag{14}
\]

At every admitted **fixed a,N**, taking large Lambda recovers (1). The proof retains all offsets and admits any degenerate effective ground mixture. It is not a fixed-Lambda uniform-N theorem; the normal theorem's constants depend on the fixed graph. Neither norm-resolvent convergence alone nor a mean produces the photon phase or HC.

## 9. Parent-runnable complete discriminating consumer

`original_neutral_photon_trial.py::run` requires explicit `UE,K`; this slice chooses neither. Run with the campaign's supplied values, for example replacing shell variables with the actual selected **diagnostic input** values:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
 original_neutral_photon_trial.py --size 5 --centre 1 0 2 --UE "$UE" --K "$K"
```

`--all-centres --no-loop` is the actual all64 static native check. The default full native `4V` matrix uses the true ten-Gram frame and native central seams, arbitrary compact A, beta-minus polar section, every site's gauge transform and the full `2V` Slater determinant covariance. The electric derivative is independently measured by finite differences of actual endpoint matrices, polar frames and complete Slater overlaps. Its asserted per-edge bound has no V. An independent flat Fourier sea at a changed EM toron checks the native full sea, not merely a copied/projected matrix.

The companion **quantum comparison** has two sites, two parallel edges with the exact native F1,F2 and original onsite coefficient. It is expressly not a periodic 3D native photon ground. It retains all 70 states of the complete `NL=4` CAR sector. U1 Gauss is solved exactly: in the slice `A2=0`, loop angle `phi=A1-A2`, and electric integers `p1=ell`, `p2=n_left-2-ell`. The magnetic law is an actual cosine of that comparison loop; no invented native coupler is introduced. Fourier truncation of ell is a declared Galerkin observation, not a changed original law or an infinite-electric proof. A second cutoff reports the observed ground shift. The complete physical periodic trial is `theta(phi) Omega(phi)`, with all Slater determinants computed; electric energy includes both derivatives, including the local charge connection on the second edge. Direct quadrature electric+actual varying sea+cosine energy is compared to the same full-CAR Fourier Hamiltonian's energy, and the actually diagonalized ground is below that trial. The flux of that **coupled comparison ground**, its eigen-residual, resolution, inputs, cutoff and missing tail status are reported.

Wrong omitted electric quantum metric and frozen flat sea are submitted to the **same total-energy assertion** in the comparison. Native wrong beta polarity/P direction reaches the same Lyapunov/coercivity acceptance; wrong omitted metric reaches the same independent determinant-overlap metric assertion; dropped toron reaches the same independent native Fourier sea assertion. The code requires each wrong case to be rejected and refuses witnesses below its declared numerical resolution. No bare nonthrow or mock-echo check is counted as physical evidence. All observables and adverse assertions are computed before JSON serialization. Full native quantum periodic diagonalization, native admission and source scalar well spectra are explicitly not reported as measured.

Tooling evidence: bound shared plan and reviewed pinned template; positive Python AST queries matched the actual `original_frame` and new `section_derivative`; direct Pyright LSP document-symbol query resolved the original function at line54 and the new supplier symbols. The final supplier LSP `publishDiagnostics` notification contained an empty diagnostics array with the scientific interpreter configured. The mounted `xd://lsp` adapter is unavailable in this worker, so the native Pyright stdio protocol was used instead. Scientific smoke is intentionally parent-owned under this assignment, and no scientific result is invented here.

## Decision selected by this supplier

Replace the imported neutral magnetic small-mean premise by **the actual complete trial (5)** and the numeric/symbolic allowance (1)/(14), with actual all64 maxima and both wrapping payments. Only use a downstream small-mean threshold if the source's fixed UE,K satisfy `K>CB` and this derived allowance is below that threshold. If the conservative domain fails, the correct next mathematical decision is to sharpen the static source envelope (9), not to assume a favourable coupled phase, erase torons, zero the electric metric or retune native symbols silently. The current deliverable supplies the normalized coupled energy/mean comparison, not those further phase/heat conclusions.

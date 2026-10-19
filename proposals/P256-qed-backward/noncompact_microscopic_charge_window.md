# Original full-field noncompact compensated charged energy window

Issue222 remains **ACTIVE**. Governing plan `issue-222-continuing-backward-20260930`, assigned revision `ed42a4078d9dca07c06716c3774d2b67477eb55266b92f03b60c53bd0ece3c3c`, pinned general template `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`, work `campaign`, requirements `outcome/verification`. The original microscopic quantum CAR and gauge quantization are explicit assumptions. This joins the existing original BO construction to a **different, real-link quadratic electromagnetic law**; it does not derive quantum statistics from Euler mechanics or pay compact-photon phase obligations.

## 1. One original BO convention and one physical noncompact quotient

Fix `a,UE,K>0`, `m=.2,r=.12,v=.3,kappa=2/a`, and a periodic connected `N^3` nearest-axis graph, `V=N^3`. The scalar fields retain all `R^(60V)` coordinates, colour links all `SU4^(3V)` coordinates, heavy CAR the **whole** `NH=2V` sector, and light CAR every configuration in `NL=2V+q`, `q=0,+1,-1`. Charged sectors contain the complete four-component opposite static SU4 reference (`bar4` for `+1`, `4` for `-1`) and U1 charge `-q` at a fixed `R`. No reference Hamiltonian/self-energy is subtracted. The total periodic U1 charge including that reference is zero.

Use exactly the endpoint average, quartic, heavy sign and slow trajectory of `original_slow_kinetic_bo.md` §§1,5–8:

\[
 H_H=-\Lambda d\Gamma(h_H),\quad
 \chi=\Lambda^{-1/2},\quad
 T_\Phi=\chi p_\Phi^2/2,\quad
 u_4=.3\chi,\quad J=.2\Lambda^{5/4},\quad K_4=.7\Lambda^{5/4}.
\]

Heavy is electromagnetically neutral. Light uses the same actual Phi/SU4 endpoints and exact `exp(i A_e)` factors. The extra27-species selector implemented elsewhere in `BulkQuantumLaw` is **not** part of this law. There is no second normal/BO scaling convention, per-momentum heavy filling rule, light-band projection or permanent normal/rotor/Fock cutoff.

Let `D` be incidence, `L=DD^T`, `C` the actual graph curl, and

\[
 \rho_q=n-2-q\delta_R,\qquad
 Q_q=UE(p_T^2+p_h^2)+\frac K2\|CA_T\|^2
          +UE\rho_q^T L^+\rho_q.
\]

The physical electromagnetic Hilbert space is obtained by real longitudinal gauge reduction/rigging, not by seeking an invariant vector in unreduced longitudinal `L2`. Equivariant representatives have `Psi(A_T+A_h+D^T alpha)=exp(i rho_q.alpha) psi(A_T+A_h)`. Their norm integrates the coexact and **real** harmonic coordinates only. The nonnegative last term is the exact eliminated longitudinal electric form, including background two and the reference. The full scalar/colour operator is the gauge-invariant Friedrichs form on this physical quotient, with all15V colour Gauss equations, not a body-frame finite-field Hamiltonian.

The harmonic real coordinates are not compactified. Their period lattice for the matter phases gives a Bloch direct integral. A cell-fibre eigenvector is not a full-real-coordinate normalizable ground. All spectral quantities below are infima of the stated closed operators.

## 2. The existing full-Gauss W genuinely intertwines Q_q

For `N>=max(5,N0(a))`, with existence-only `N0(a)`, reuse the original exact full-Gauss isometries `W_q,Lambda` of `original_slow_kinetic_bo.md` §§5–7. Their body expression is the actual cutoff scalar normal state `u_theta(y)` times the smooth **whole-heavy Slater line** `Omega_H(y)`, tensored with the entire light/reference fibre, and descended over the complete colour orbit. The heavy Kato trivialization and normal wavefunction do not depend on real `A`. All spatial colour transformations commute with each local number `n_x`; the reference character is retained. The stabilizer characters are `z^(NH+NL)` times the reference character, equal to one in all three physical sectors. Orbit normalization and disjoint core supports give `W_q^*W_q=I` exactly.

For every site, on the common smooth physical core,

\[
 n_x W_q=W_q n_x,\qquad
 p_T W_q=W_q p_T,\quad p_h W_q=W_q p_h,\quad
 (CA_T)W_q=W_q(CA_T).
\]

Thus both the **unbounded** quadratic magnetic multiplication and electric derivatives, and the physical Coulomb matrix, intertwine:

\[
 \boxed{Q_q W_q=W_q Q_{q,\rm eff}.}
\]

This is stronger than preserving total `NL`. A spatial-number-mixing lift would change `rho_q^T L^+rho_q`. A photon-dependent line phase would leave every `n_x` and the heavy line projector unchanged but introduce real electric derivatives. Neither qualifies as this W.

Initially the identities hold for compactly supported smooth transverse/harmonic vectors, smooth colour-equivariant representatives, and every light/reference component. The exact norm equality for their photon/Coulomb forms extends by completion to the closed form domains. Equivalently, W has no electromagnetic derivatives, intertwines coordinate multipliers and the finite local-number spectral projectors, so it intertwines the closed photon+Coulomb operator and its spectral calculus; its adjoint has the corresponding domain action. Consequently `P_q=W_qW_q^*` and its complement reduce `Q_q`, with no off-diagonal electric or quadratic-magnetic remainder. This is an operator/form statement on the physical quotient, not deletion of a longitudinal electric direction or an identity verified only below a photon energy cutoff.

Write

\[
 c_\theta(\Lambda)=E_\theta(\Lambda)-E_{\min}(\Lambda)\ge0,
\]
\[
 H_{q,\rm eff,\Lambda}=\bigoplus_{\theta\in Z_4^3}
        [c_\theta(\Lambda)+Q_{q,\theta}+d\Gamma(h_{L,\theta}(A))],
 \quad K_q=H_{q,\Lambda}-E_{\min}(\Lambda).
\]

Every one of the64 actual centre Grams and scalar offsets stays in each infimum and supplier. They are not finite-Lambda conserved microscopic sectors. A minimizing neutral centre minimizes the **sum** of offset and coupled photon/light energy, not necessarily the zero-offset scalar well. Charged and neutral sectors share the same scalar offsets because their heavy/normal scalar supplier is the same; light/reference connections belong in the existing sector residuals, not in invented scalar counterterms.

## 3. Full complement and closed domains survive the unbounded photon law

On a fixed graph the native light matrix is bounded over **all real A** because `|exp(iA_e)|=1`. Its normal-field/link Lipschitz bounds have exactly the same property. Therefore the original estimates give, separately for `q=0,+1,-1`,

\[
 \|K_qW_q-W_qH_{q,\rm eff,\Lambda}\|\le d_q(N,\Lambda),\qquad
 (1-P_q)K_q(1-P_q)\ge g_q(N,\Lambda)(1-P_q),
\]
\[
 d_q\le C_q\Lambda^{-1/16}+C_q\Lambda^{-5/16}
             +p_q(\Lambda)e^{-c_q\Lambda^{7/20}}\longrightarrow0,
\]
\[
 g_q\ge c'_q\Lambda^{1/4}-C'_q\Lambda^{7/40}-C'_q
           \longrightarrow+\infty.
\]

All constants here are fixed-`a,N`, sector-dependent constants of the original normal/core/exterior proof; `p_q` is its finite polynomial bound. Their values and a usable numerical Lambda threshold are **not** supplied by deterministic source jets. There is no uniform-N or uniform-a BO estimate.

In the core IMS proof the heavy-ground component of a complement vector is orthogonal to `u_theta` almost every real `A`, componentwise in the **entire** light/reference fibre. The unbounded nonnegative `Q_q` remains in the complementary quadratic form. One may keep it unchanged, or drop it only when taking a scalar lower bound; there is no compact magnetic upper estimate and no finite photon-energy restriction. Connection cross terms are estimated against the positive relative normal form. Only `S_theta-Lambda e_theta` is discounted, never the negative classical baseline `Lambda e_theta`.

For completeness, the exterior argument still retains all unbounded field regions. With

\[
 r_\Phi^2=\sum_{x,\mu}\operatorname{Tr}\Phi_{x\mu}^2/4,\quad
 C_x(a)=\sqrt{(m+6r/a)^2+3(v/a)^2},
\]
\[
 A_4=\kappa/(8V),\quad B_H=2N_H C_x,\quad B_L=2N_L C_x,
\]

its exact endpoint bounds are `V0>=A4 r_Phi^4-B_H r_Phi` and `||H_L||<=B_L r_Phi`. Put `r0=max(1,(2B_H/A4)^(1/3))`, `C0=B_L r0`, `C1=2B_L/A4`. The original positive alignment potential `U_Lambda` obeys

\[
 \|H_L\|\le C_0+C_1(U_\Lambda-v_*).
\]

The exterior full form, **with** `Q_q>=0`, is bounded below by

\[
 (\Lambda-C_1)(U_\Lambda-v_*)-(E_{\min}-\Lambda v_*)-C_0,
\]

and then by the original `c Lambda^(3/5)-C Lambda^(3/8)-C`, above the normal gap. The actual pointwise whole-heavy minimum, singular/reducible competitors, IMS costs and all normal modes are unchanged. A false globally bounded exterior light norm is not used.

The effective domain is `D(Q_q)` since the fixed-frame light multiplication is bounded. The bounded residual maps this domain into `D(K_q)`. The original duality argument gives `W_q^* D(K_q) subset D(H_eff,q)` and invariance of `D(K_q)` under P/complement. Off-diagonal blocks are bounded by `d_q`; subtracting them gives the same selfadjoint block comparison

\[
 D_q=W_qH_{q,\rm eff,\Lambda}W_q^*\oplus K_{q,C},\quad
 \|K_q-D_q\|\le3d_q,\quad K_{q,C}\ge g_q.
\]

No permanent well-domain truncation or renaming of a non-scalar colour/light angular Hamiltonian as a common scalar is involved.

## 4. An explicit finite effective witness, and the actual g boundary

A complement lower bound alone does not imply a ground comparison when it lies below the effective infimum. Here a finite graph provides an explicit witness independent of the unknown scalar offsets. Gram coercivity, for every N and all64 centres, gives

\[
 t=\|G\|_F\le t_*(a):=(4\sqrt2 C_x/\kappa)^{2/3},\quad
 \|h_L(A)\|\le B_*(a):=C_x\sqrt{2t_*}.
\]

Let `lambda_j>0` be the nonzero eigenvalues of the actual curl quadratic form on an orthonormal coexact basis, and `b_h` its harmonic dimension. A normalized coexact oscillator ground times a normalized real harmonic Gaussian of width1 has photon energy

\[
 Z_{\rm ph}=\sum_j\sqrt{UE K\lambda_j/2}+UE b_h/2<\infty.
\]

A full fixed-number occupation vector in a zero-offset scalar well, with the actual reference and equivariant colour lift, is a legitimate body effective trial. In that sector `||H_L||<=N_L B_*` and

\[
 \|\rho_q\|^2\le4V+4|q|+q^2.
\]

Hence one available **rigorous finite witness** is

\[
 T_q(N):=(2V+q)B_*+UE\|L^+\|(4V+4|q|+q^2)+Z_{\rm ph},
 \qquad E_{q,\rm eff,\Lambda}\le T_q(N).
\]

The zero-offset centre can depend on Lambda, but the bound does not. This witness is deliberately extensive and is **not** the final charged-minus-neutral upper. It establishes the actual boundary needed for the block theorem: choose a finite Lambda with

\[
 \boxed{g_q(N,\Lambda)>T_q(N)\quad(q=0,+1,-1).}
\]

The actual complement constants/g must satisfy this condition; a finite Galerkin eigenvalue or guessed d/g does not certify it. The existence follows from the fixed-N asymptotic lower for g. Under this condition the block comparison and W trials imply

\[
 E_{\min}+E_{q,\rm eff,\Lambda}-3d_q\le E_{q,\rm micro}
      \le E_{\min}+E_{q,\rm eff,\Lambda}+d_q.
\]

The upper can equivalently use any actual effective near-infimum trial. All centre offsets remain in `E_q,eff` and cancel only as the common overall `E_min`.

## 5. Joined lower and unfiltered point-source upper

For each actual metric define

\[
 s_\theta=G_{00,\theta}-G_{0s,\theta}G_{ss,\theta}^{-1}G_{s0,\theta},\qquad
 \mu_{\min}(N)=m\min_\theta\sqrt{s_\theta}>0.
\]

The untruncated real-link effective threshold in `noncompact_charge_threshold.md` is across **all harmonic Bloch fibres**. It retains the exact reduced-Coulomb Gaussian temporal factor, positive background-two/canonical neutral determinant, harmonic charged-interval phases and the dense complete-CAR/reference source range. It yields in each centre `E_eff,q,theta-E_eff,0,theta>=m sqrt(s_theta)` for both signs. Taking minima with the **same** scalar offsets gives

\[
 E_{q,\rm eff,\Lambda}-E_{0,\rm eff,\Lambda}\ge\mu_{\min}(N).
\]

Combining the charged microscopic lower with the transferred neutral trial upper, under §4's g boundary, proves

\[
 \boxed{E_{q,\rm micro}-E_{0,\rm micro}
       \ge\mu_{\min}(N)-3d_q-d_0.}
\]

For the other direction use the actual symmetric near-infimum supplier of `noncompact_neutral_supplier.md` §§3–5. It is a physical normalized mixed preparation with neutral spectral support within epsilon of the global effective infimum, retaining every scalar offset. When constructed using fibre/cell vectors, use a normalizable narrow symmetric Bloch packet and retain its actual energy excess `epsilon_packet`; all harmonic directions are included and its linear current term cancels by symmetry, not by freezing the photon. One can also use the full-real spectral theorem directly, without a ground eigenvector.

After the actual C/translation averaging, `<n_x>=2`, `<q_x^T L^+(n-2)>=0`, and the density covariance is nonnegative. The unfiltered complete-reference creator has `B_x^*B_x=4-n_x` (hole source has `n_x`), hence normalized weight2. With the legal real longitudinal dressing for `delta_x-delta_R`, its entire changed electric cost is

\[
 UE[r_{xR}-\operatorname{Cov}(n_x,q_x^TL^+(n-2))]\le UE r_{xR},
 \qquad r_{xR}\le13/4
\]

on every3D periodic graph. The full light commutator contributes `-<H_L>/(2V)<=B_*`. The weighted full-H near-infimum cancellation contributes at most `2epsilon`; it does not set photon/matter correlations, hopping current or reference derivatives to zero. The C-related hole construction gives the same estimate for q=-1. The isometric physical preparation is `W_q B_x rho_eff,0 B_x^* W_q^*/2`, with all light/reference/normal connections already in the **same** sector's `d_q`.

Combining this charged trial with the true neutral block lower gives the requested joined window:

\[
 \boxed{\mu_{\min}(N)-3d_q-d_0
 \ \le E_{q,\rm micro}-E_{0,\rm micro}\ \le
 \frac{13}{4}UE+B_*(a)+d_q+3d_0+2\epsilon+\epsilon_{\rm packet}.}
\]

There is no extra compensating reference-path term to add twice to this W construction, and no primitive reference path may be dropped in a different implementation. The energy remains a charge **plus opposite static reference**, uniformly bounded in that reference separation; it is not an isolated/mobile mass. The point upper is not assigned to arbitrary coherent finite-width/spin packets without their actual source quadratic forms.

## 6. Positive uniform window along a legal Lambda(N)

At fixed a, the unique continuum Gram minimum for the original source is `G_infinity=diag(g0,gs,gs,gs)` with `g0,gs>0`. Positivity follows from the exact one-sided derivatives of the convex Gram functional, not a floating minimizer. The source-owned uniform shifted-grid estimate `sup_theta |F_Ntheta-F_infinity|<=D_N(a)->0` and strong convexity give

\[
 \|G_{N\theta}-G_\infty\|_F\le\eta_N=2\sqrt{D_N/\kappa}.
\]

For `eta_N<=min(gs/2,g0/4,sqrt(g0 gs/8))`, the native spatial block is at least `gs/2` and

\[
 s_{N\theta}\ge g0-\eta_N-2\eta_N^2/gs\ge g0/2.
\]

There is therefore an existence-only `N1(a)` and a strictly positive native floor

\[
 \mu_*(a)=m\sqrt{g0/2},\qquad
 \mu_{\min}(N)\ge\mu_*\quad(N\ge N1(a)).
\]

For arbitrary fixed `a,UE,K>0`, choose any `delta>0` with `4delta<mu_*`. For **every** `N>=max(5,N0(a),N1(a))`, the fixed-N original BO estimates allow a finite legal `Lambda(N)` satisfying simultaneously

\[
 d_0,d_+,d_-\le\delta,\qquad g_0>T_0(N),\quad g_+>T_+(N),\quad g_->T_-(N).
\]

Choose the actual spectral/Bloch supplier errors with `2epsilon+epsilon_packet<=delta`. All electromagnetic coefficients remain fixed, and the source slow/heavy/alignment trajectory is unchanged. The full-domain family then has, for either sign and all reference positions,

\[
 \boxed{0<\mu_*-4\delta\le E_{q,\rm micro}(N,\Lambda(N))-E_{0,\rm micro}(N,\Lambda(N))
       \le13UE/4+B_*+5\delta<\infty.}
\]

For example the purely existential choice `delta=mu_*/8` gives lower `mu_*/2`; it is **not** a measured BO remainder or a numerical Lambda recipe. The order of quantifiers matters: fixed a/UE/K, sufficiently large N, then a finite N-dependent Lambda. No uniform-N fixed-Lambda BO estimate, numerical N0/N1, thermodynamic operator limit, reference removal, mobile band or mass pole is proved.

## 7. Actual source-bound comparator, and its discriminating boundary

`noncompact_microscopic_charge_window.py` is the runnable consumer. It hashes its own source/proof and reused suppliers before execution and rejects changed bindings afterwards. Parent invocation from this proposal directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
 noncompact_microscopic_charge_window.py --native-size 5 --n 4 --M 2 \
 --spacing 1 --UE .37 --K .9 --zeta .23
```

The native slice reuses the actual original all-ten-entry centre123 frame, complete full4V/whole2V heavy source jets, full physical normal directions, and actual arbitrary real Peierls phases. An independent endpoint block-row bound controls the deterministic light deviation uniformly in photon phases. It also computes all64 native row/Schur inputs through existing APIs. None of these finite samples certifies N0/N1, normal-vacuum moments or d_q.

The quantum slice is the existing **two-vertex three-edge Coupled cochain quotient**, with its actual quadratic oscillator, harmonic Bloch kinetic term and exact reduced Coulomb `UE(n_left-2)^2/3`, including the opposite reference at the right vertex. Scalar coefficients come from the original `scalar_paths`, not a new endpoint law. The complete70-dimensional NH=4 heavy CAR and complete70/56-dimensional neutral/charged light CAR are retained, together with every reference component. It constructs nonuniform real Phi, genuine SU4 endpoints and **varying actual whole-heavy Slater lines** from those full8x8 operators, not a frozen vector or a per-site occupied projection. Local colour transformations are derived from those real field/link data and independently transported through fixed-number exterior representations and reference `4/bar4`.

It checks an actual coupled photon/light neutral eigenpair lifted with the actual heavy line, the full original source CAR norm operator and colour covariance, the actual charged source full-H Rayleigh quantity, every spatial number, and both the quadratic-photon/Coulomb and electric operator intertwining. The heavy/reference products stay factored; there is no avoidable huge dense full-H Kronecker matrix.

Two adverse lifts meet the **same** accept boundaries:

* An actual intersite CAR unitary preserves total NL and the heavy line but mixes spatial numbers. It must fail the complete-fixed-number Coulomb commutator assertion used by the true local-colour lift.
* Multiplication by `exp(i y)` in the real harmonic coordinate preserves every n_x and the exact heavy line. It must fail the **same electric operator** assertion used by the true photon-independent lift. Its integer harmonic-label shift is embedded into a larger basis without wraparound/cutoff loss; an independent kinetic product rule checks the expected `UE[2(ell+zeta)+1]/6` defect. This is a real operator failure, not a phantom phase diagnostic or a fictitious fractional periodic Fourier state.

Freezing the whole-heavy line under the actual nonuniform Phi/U variation must also fail its actual heavy eigenline assertion. Conversely t=0 is a legitimate frozen-line positive control and is not labelled adverse.

These are finite Galerkin/fibre **observations when actually run**, with genuine Coulomb/photon/CAR consumers. They are not full R^(60V)/SU4 quantum field simulations, numerical executions of W's scalar normal vacuum, infinite-basis threshold certificates or measured microscopic energy windows. The JSON keeps the exact constructive theorem, native deterministic source data and finite quantum consumer scopes separate, and explicitly supplies no invented d/g values. Parent owns the scientific run and receipt; this handoff does not claim unexecuted numerical results.

No photon phase, reference-free mass, dispersion, physical spin, Lorentz symmetry, photon S-matrix or Compton completion follows. The joined original electron–photon campaign remains active.

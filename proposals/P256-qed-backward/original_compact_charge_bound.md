# Native full-metric compact-U1 finite-volume charged energy lower bound

This is a scoped result in the active issue222 campaign, assignment plan `issue-222-continuing-backward-20260930` revision `11e9251a60596401163092e5b39958a6ea55866301fed44e3b9fcd0a0370e31b`, general template revision `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`, work item `campaign`, requirements `outcome/verification`. It is **not** campaign completion. The theorem below is an exact theorem for a specified effective quadratic form. Transfer from the original full Cartesian/SU4/heavy quantum law is the integration owner's separate, currently conditional BO obligation.

## Statement and actual source

Fix a periodic nearest-axis graph of side N>=5 and V=N^3, a>0, m>0, r>=0, finite v, and a positive-definite real 4x4 native Gram G. The source cutoff family has m=.2, r=.12, v=.3, a<=1, and kappa=2/a. Use the actual four matrices

\[
 \Phi_\mu=\sum_\nu(\sqrt G)_{\mu\nu}\Gamma_\nu,
 \qquad \{\Phi_\mu,\Phi_\nu\}=2G_{\mu\nu}I_4.
\]

G is obtained from `original_charged_principal.original_frame(..., native_toron=True)` in all ten symmetric coordinates on the actual finite source grid and centre toron. No isotropic fitted `RotationBand` replaces it. `RotationBand` supplies **only the unchanged scalar endpoint coefficients** through the existing native API. `BulkQuantumLaw.onebody` supplies the actual endpoint-average light matrix; its additional local 27-channel selector is never executed or added to the original law.

For any A in the entire compact torus T^(3V), let T_j(A) be the unitary positive-axis shift, including the fixed central SU4 seam and exp(iA_e). Put

\[
 W(A)=\sum_j\left[I-\frac{T_j+T_j^\dagger}{2}\right]\ge0,
 \quad S_j(A)=\frac{T_j-T_j^\dagger}{2i},\qquad
 L(A)=mI+\frac r a W(A).
\]

The source endpoint law is exactly

\[
 h_L(A)=\Phi_0 L(A)+\frac v a\sum_j\Phi_j S_j(A).
\]

Matrix/site tensor products are implicit. The factors agree with onsite m+3r/a and directed coupling

\[
 F_j=-\frac{r\Phi_0+i v\Phi_j}{2a}.
\]

Let

\[
 G_s=G_{1:3,1:3},\quad b=G_s^{-1}G_{1:3,0},\quad
 s=G_{00}-G_{0,1:3}G_s^{-1}G_{1:3,0}>0,
 \quad \mu=m\sqrt s,
\]

\[
 S_F=\frac{3r^2G_{00}+v^2\sum_jG_{jj}}{a^2},\qquad
 C_G=\frac{V S_F}{4\mu^2}.
\]

For UE1>0, K1>0, a bounded real smooth gauge-invariant common scalar V0(A) (V0=0 is allowed), and the full retained original light CAR sectors NL=2V+q, q=0,+1,-1, define

\[
 H_{\rm eff}=u_{e1}\sum_e p_{A_e}^2
 +K_1\sum_p(1-\cos\operatorname{curl}_p A)
 +V_0(A)+d\Gamma(h_L(A)),\qquad p_A=-i\partial_A.
\]

The charged sectors have an explicit static opposite U1 reference at R and opposite SU4 representation (bar4 for q=+1, 4 for q=-1); the reference has no separate Hamiltonian or subtracted self-energy in this **defined effective form**. All Gauss constraints and domains are specified below. Write E_q for the infimum on its physical form domain and E_s for the ground energy of the gauge-invariant scalar operator with potential

\[
 V_s(A)=K_1\sum_p(1-\cos\operatorname{curl}_p A)+V_0(A)+E_{\rm sea}(A),
 \qquad E_{\rm sea}(A)=\sum_{\lambda<0}\lambda(h_L(A)).
\]

**Theorem.** Without a rotor/angular/Fock truncation,

\[
 E_{+1},E_{-1}\ge E_s+\mu,\qquad
 E_0\le E_s+u_{e1}C_G,
\]

and therefore

\[
 \boxed{E_{\pm1}-E_0\ge\mu-u_{e1}\frac{VS_F}{4\mu^2}.}
\]

In particular, when C_G>0 as in the original source, u_e1=mu/(2C_G)>0 gives a lower bound mu/2>0 with any positive K1. The decoupled boundary r=v=0 has C_G=0 and already gives mu for every positive u_e1; it is not the interacting source used here. For all 64 centre wells at once, set mu_min=min_theta mu_theta, C_max=max_theta C_theta, u_e1=mu_min/(2 C_max). Arbitrary **common charged/neutral** offsets c_theta, including heavy-well offsets, then preserve the direct-sum lower bound mu_min/2: min_theta(E_s,theta+c_theta+mu_theta) minus min_theta(E_s,theta+c_theta+u_e1 C_theta) is at least mu_min-u_e1 C_max. No preferred centre well is silently selected.

These are fixed-volume bounds. The sufficient u_e1 scales with V and a through C_G; no volume-uniform or fixed-nonzero-coupling cutoff-limit conclusion follows.

## Repairing the proposed mass argument: every SPD metric, not just small G0j

Aligning beta with Phi0 would give a positive-mass estimate containing a penalty |v| sum|G0j|/(a sqrt(G00)). (If v is included in P_j, its norm is |v|/a, not 1/a.) Even that correctly normalized crude estimate is negative at 32 of the 64 actual N5 centre classes at a=.001; an actual native design probe found the worst at (1,1,1), approximately -.1406145. This is **not** a spectral obstruction. Discarding the Wilson term and using an unnecessarily poor beta created the failure.

Instead set

\[
 R=\Phi_0-\sum_j b_j\Phi_j,\qquad \beta=R/\sqrt s.
\]

The actual full metric gives R^2=sI and {R,Phi_j}=0, so beta^2=I. No Gram entry or endpoint coefficient changes. Exactly

\[
 h_L(A)=\beta\,M(A)+\sum_j\Phi_j
 \left[b_j L(A)+\frac v a S_j(A)\right],\qquad
 M(A)=\sqrt s\,L(A)\ge\mu I.
\]

This is an analysis basis choice, not an isotropic copy or a new microscopic law. The fixed Hermitian chirality Gamma5=Gamma0 Gamma1 Gamma2 Gamma3 anticommutes with every Phi_mu and hence every h_L(A). Choose an A-independent chiral plus basis C_+, and C_-=beta C_+. In this chiral basis beta is off-diagonal identity, every spatial Phi_j has anti-Hermitian upper-right block, and

\[
 h_L=\begin{pmatrix}0&Q\\Q^\dagger&0\end{pmatrix},
 \qquad Q=M+iD,\quad M,D\text{ Hermitian},\quad M\ge\mu I.
\]

All operators here act on the whole 2V chiral half. In particular [M,D] need not vanish: the Wilson and sine shifts of different axes generally fail to commute at nonzero flux. For every z,

\[
 \|Qz\|\,\|z\|\ge\operatorname{Re}\langle z,Qz\rangle
 =\langle z,Mz\rangle\ge\mu\|z\|^2.
\]

Thus the smallest singular value of Q is at least mu and h_L has no spectrum in (-mu,mu). Equivalently, after rotating the chiral grading, h_L has the beta-diagonal block form (M,D;D,-M). For |E|<mu, Schur elimination yields M-E+D(M+E)^(-1)D>0, so no eigenvector exists; the argument never replaces h^2 by M^2+D^2 or assumes [M,D]=0.

Fixed Gamma5 supplies exact plus/minus pairs and precisely 2V negative eigenvalues. The light vacuum fills the negative **whole** cluster, not two modes imposed in each momentum/site block. All states of the full NL=2V+1 sector have fibre energy at least E_sea+mu; the same holds for NL=2V-1 by removing a negative eigenvalue. Internal degeneracies are harmless. Fixing the conserved particle-number sector is not a Fock cutoff.

## Global analytic occupied Slater and the Berry phase

Accretivity Re Q>=mu means Q never meets the principal matrix logarithm's cut and stays invertible on the whole torus. It is continuously deformable through accretive matrices to I, for example Q_t=(1-t)I+tQ, and the principal Log Q is a single-valued smooth periodic matrix function. Consequently

\[
 \theta(A)=\operatorname{Im}\operatorname{Tr}\operatorname{Log}Q(A)
\]

is a globally defined real smooth periodic function. This is **not** principal arg det Q: reducing the determinant's argument modulo 2pi before halving it can introduce a sign discontinuity. The consumer evaluates the trace-log imaginary part by summing principal arguments of all eigenvalues of Q; every such eigenvalue is in the right half-plane. This sum is the matrix-function trace even for a nonnormal matrix or a Jordan block.

Write Q=U|Q| for its smooth polar decomposition. The full normalized negative frame is

\[
 Z_-(A)=\frac1{\sqrt2}\binom{I_{2V}}{-U^\dagger(A)}.
\]

Its range is the negative spectral subspace (multiplying h_L maps it to itself with the negative definite compression -Q U^dagger). Let Omega_raw be the exterior product of all 2V columns, with the fixed column order. Since det U=exp(i theta),

\[
 \langle\Omega_{\rm raw},d\Omega_{\rm raw}\rangle
 =\operatorname{Tr}(Z_-^\dagger dZ_-)
 =-\tfrac12\operatorname{Tr}(U^\dagger dU)
 =-\tfrac i2d\theta.
\]

Therefore

\[
 \Omega(A)=e^{i\theta(A)/2}\Omega_{\rm raw}(A)
\]

is a single global, smooth, normalized, periodic occupied Slater with **exactly zero** determinant Berry connection. No inference from local zero curvature substitutes for this explicit phase. The homotopy and principal log settle the global issue, including large loops and toron cycles.

## Exact U1 Gauss background and static reference domain

A site gauge transformation g_x=exp(i chi_x) acts as A_xy -> A_xy+chi_x-chi_y. In the fixed chiral decomposition g acts as g_2=diag_x(exp(i chi_x)I_2) on each half. Exactly

\[
 Q(A+d\chi)=g_2Q(A)g_2^\dagger,\quad
 U(A+d\chi)=g_2U(A)g_2^\dagger,\quad
 \theta(A+d\chi)=\theta(A).
\]

The occupied frame transforms as Z_-(A+dchi)=g_4 Z_-(A)g_2^dagger. Taking its **whole exterior product**, not just its density, gives

\[
 \boxed{\Omega(A+d\chi)=e^{-2i\sum_x\chi_x}
       \Gamma(g_4)\Omega(A).}
\]

This is the exact local background of two particles per site. Although Tr P_xx=2 also follows from the off-diagonal chiral sign, mean density alone would not prove a Gauss state. This exterior identity does. It is well-defined for chi_x modulo 2pi because the background is integral. Constant chi cancels exactly for NL=2V.

Let F_q be the entire original NL=2V+q light CAR sector and R_q the actual finite-dimensional opposite spectator representation at R. Physical wavefunctions belong to H^1(T^(3V),F_q tensor R_q), with periodic link coordinates, and obey

\[
 \Psi(A+d\chi)=e^{-2i\sum_x\chi_x}
  \Gamma(g_4)\,e^{-iq\chi_R}\Psi(A).
\]

Its infinitesimal version is

\[
 (\operatorname{div}p)_x\Psi=(n_x-2-q\delta_{xR})\Psi.
\]

Summing over the periodic graph gives NL-2V-q=0, exactly; no unreferenced global charge is permitted. The norm ||Psi(A)|| is gauge invariant because every factor is unitary. The scalar comparison has no static charge. The charged scalar **norm**, rather than a charged scalar section, is what enters diamagnetism; no illegal charged scalar ground state is assumed.

These physical domains are nonempty. For q=+1, apply a local light creator at x to Omega and multiply by an oriented compact Wilson string exp(i sum_path A) from x to R. Its gauge endpoint character is exp(i(chi_x-chi_R)), so its transformation is precisely the displayed charged law. Contract its colour index with the actual opposite bar4 reference using the corresponding SU4 transporter. For q=-1 use a local annihilator, the inverse U1 string, and a 4 reference. At least one local component is nonzero because the local empty/occupied trace is 2. Strings are smooth periodic finite products; they do not introduce rotor cutoffs. These are domain/existence witnesses, not charged energy trial estimates or claims of isolated charges.

## Full SU4 physical lifting is not zero angular kinetic

The regular full-rank Clifford field and flat central links have gauge orbit SU4^V/Z4: the connected graph leaves only a common central stabilizer. On this orbit the actual native endpoint transforms as h -> diag(g_x) h diag(g_x)^dagger; all 15 generators at each site are retained. A fixed-gauge light vector can be lifted by the complete CAR representation, the original whole-heavy NH=2V representation and the spectator representation. The neutral stabilizer character is z^(NH+NL)=z^(4V)=1. Doping +1 and adding a bar4 spectator contributes z^(+1-1)=1; a hole and a 4 spectator contribute z^(-1+1)=1. Thus the lift descends to the orbit without a fake colour scalar reference. Its full equivariance is exactly the SU4 Gauss condition, including every generator and the discrete centre.

The executable uses `OriginalSpatial.gauge_columns`, `charge_form`, `charge_controls` and `pair_reference` for all 15 constrained directions at q0 and a nonzero momentum. `pair_reference` is the actual fundamental tensor bar4 singlet with generator T tensor I-I tensor T^T and explicit spatial separation; it also rejects silently omitting the reference's q0 neutrality. Arbitrary local noncentral SU4 transformations are checked directly through the native endpoint matrix, fields and links.

**None of these domain identities imply zero SU4 kinetic/Born-Huang energy.** The original heavy occupied colour mean can be nonzero, and its variance survives. A full Cartesian normal extension, the quotient density, centre-well offsets, heavy Berry line, all colour/light/reference angular terms and complement control belong to the parent's source-to-effective BO construction. They cannot be dropped because the norm is gauge invariant or because an algebraic reference singlet exists. This proof assumes the stated effective form after that transfer; it does not assert that the original microscopic Hamiltonian is already that form.

For a microscopic corollary one needs a genuine lower transfer/complement bound H_micro,q>=c+H_eff,q-epsilon_q and a transferred physical neutral trial of cost at most c+E_eff,0+eta_0. Then the lower mass estimate loses epsilon_q+eta_0. More general retained positive terms help the charged lower bound but require their actual neutral trial cost to be included; they are not automatically common scalars. The parent's proposed vanishing light/reference connection remainder is a transfer obligation, not an established premise supplied by this file. In a multi-well transfer, offsets must be common to both sectors and every well retained.

## Diamagnetic lower bound and neutral upper bound: factors

The effective form is the closed semibounded H^1 form: the torus is compact, the matter/reference fibres finite dimensional, and h_L and all potentials bounded at fixed graph and cutoff. It is restricted to the closed equivariant Gauss subspaces. No singular gauge slice or boundary condition at a gauge-fixing singularity is used. The scalar operator on the full torus is elliptic for u_e1>0; its unique positive normalized ground f_s is gauge invariant by symmetry and uniqueness, so the full-torus and gauge-invariant scalar ground infima coincide.

For any physical charged Psi, the Sobolev norm inequality

\[
 |\partial_e\|\Psi\||\le\|\partial_e\Psi\|
\]

holds almost everywhere, including zeros by regularization. Its gauge-invariant norm is an admissible scalar form vector. The fibre inequality dGamma(h_L)>=E_sea+mu in either charged sector and this diamagnetic inequality give E_q>=E_s+mu. Vector-valued references do not affect either inequality.

The neutral trial f_s(A)Omega(A) is periodic, H^1, and satisfies **exact** background-two Gauss. Its fibre matter energy is E_sea. The globally removed connection eliminates the kinetic cross term. Its extra kinetic energy is

\[
 u_{e1}\int |f_s|^2\sum_e g_{ee}(A),\qquad
 g_{ee}=\frac12\operatorname{Tr}(\partial_eP_-)^2.
\]

With exact occupied-to-empty gap at least 2mu,

\[
 g_{ee}=\sum_{o<0,u>0}
 \frac{|\langle u|\partial_eh_L|o\rangle|^2}{(E_u-E_o)^2}
 \le\frac{\|\partial_eh_L\|_{\rm HS}^2}{8\mu^2}.
\]

The factor 1/8 contains both the squared 2mu denominator and the two equal Hermitian off-diagonal orientations in the HS norm. For a single native oriented edge e=(x,x+j), N>=5 means distinct endpoints, and its analytic Peierls current/contact are respectively iF_j exp(iA_e) and -F_j exp(iA_e), plus their adjoints. Therefore

\[
 \|\partial_eh_L\|_{\rm HS}^2=2\|F_j\|_{\rm HS}^2,
 \qquad \|F_j\|_{\rm HS}^2=\frac{r^2G_{00}+v^2G_{jj}}{a^2}.
\]

The temporal-spatial off-diagonal Gram entries do **not** vanish by assumption: their imaginary cross trace cancels in F_j^dagger F_j. Summing the one positive edge per site and axis gives exactly sum_e ||F_e||_HS^2=V S_F. Thus sum_e g_ee<=V S_F/(4mu^2), establishing the stated neutral upper and charged-minus-neutral lower bound. The electric convention is u_e1 p^2, not u_e1 p^2/2.

The polar derivative used by the consumer is independently analytic: Q=U R, dU=U X, and R X+X R=U^dagger dQ-dQ^dagger U. Positive singular values make this Sylvester equation uniquely solvable, including degeneracies. Its g_ee=||dU||_HS^2/4 agrees with the actual whole-cluster spectral formula. Native current and contact also satisfy d_chi h=i[chi,h] and d_chi^2 h=-[chi,[chi,h]]; omitting contact fails the same identity on the sampled arbitrary-flux controls.

## Why a pointwise gap alone is insufficient

An exact counterexample to the gap-only implication is a two-mode fermion on a compact coordinate x with h(x)=m(cos x sigma_x+sin x sigma_y), electric u p_x^2, vacuum filling one negative mode, and charged filling zero or two. Every pointwise gap is m and a fixed sigma_z gives plus/minus pairs. Both charged energies are 0, but the periodic neutral ground is

\[
 E_N=u/2-\sqrt{m^2+u^2/4},\qquad
 E_C-E_N=\sqrt{m^2+u^2/4}-u/2<m\quad(u>0).
\]

To see this, conjugate by exp(-ix sigma_z/2); the transformed wavefunctions are antiperiodic, with k in Z+1/2 and eigenvalues u(k^2+1/4) plus/minus sqrt(m^2+u^2 k^2). The minimum is |k|=1/2. This is a comparison countermodel, not a replacement of the original endpoint. Its Q=m exp(ix) is not globally accretive and its occupied Berry holonomy is nontrivial. It demonstrates exactly why a pointwise one-body gap is not already the charged ground-energy difference. Our native theorem supplies both the global accretive phase and a quantitative neutral trial cost, rather than claiming the cost is absent.

## Native consumer and evidence boundary

Public APIs in `original_compact_charge_bound.py` are `metric_constants`, `chiral_basis`, `wilson_matrix`, `sea_section`, `polar_derivative`, and keyword-only `run`. `run` defaults to N5/a1, evaluates all 64 native full10 Gram constants, and chooses one **common strictly positive** electric coefficient using the all-well formula. It runs actual native controls at identity, the noncubic quarter centre (1,0,0), the smallest Schur-gap centre and the largest off-diagonal-metric centre (duplicates removed). K1 defaults to .4/a>0. Every requested phase sample uses independent angles across the entire [-pi,pi] range on all 3V edges; it does not freeze or delete quantum photons from the defined operator.

The consumer checks actual whole occupations/plus-minus pairing, positive noncommuting M, native endpoint/Bloch identity and full10 stationarity, analytic current/contact Ward, full Slater/projector/QGT/global phase/U1-background equivariance and 2pi periodicity, all15 native SU4 covariance and the explicit reference/constrained Gauss APIs. All source owners and proofs are hashed before/after execution. The CLI writes the receipt only after checks and unchanged bindings succeed.

Examples for the integration owner's final native boundary:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
proposals/P256-qed-backward/original_compact_charge_bound.py

# Explicit cutoff/adverse metric, or all-centre native diagnostics:
/home/administrator/substrate-framework/.venv/bin/python \
proposals/P256-qed-backward/original_compact_charge_bound.py --spacing .001 \
--centre 1 1 1 --output proposals/P256-qed-backward/original_compact_charge_bound_cutoff_receipt.json
/home/administrator/substrate-framework/.venv/bin/python \
proposals/P256-qed-backward/original_compact_charge_bound.py --all-centres
```

Optional flags are `--size`, `--spacing` (alias `--a`), repeatable `--centre i j k`, `--all-centres`, `--phase-samples`, `--electric`, `--magnetic`, `--seed`, and `--output`. Arbitrary positive explicit electric inputs are accepted even if the resulting analytic lower bound is nonpositive: that is an honest loss of this sufficient bound, not an asserted negative physical mass. The result reports exact-formula floating evaluations separately from numerical one-body/Slater controls. Neither is an interval certificate of an exact numerical Gram. The symbolic theorem assumes exact SPD G; the finite convex source selection is its separate native premise.

Before editing, mounted ast-grep returned a positive Python structural query on `original_frame`; the native Pyright 1.1.414 stdio server returned the actual document symbols and resolved the caller to its definition at source line54, rooted at the project's pyproject.toml. The `xd://lsp` device itself is absent. Final file types and actual default/cutoff CLI execution are parent-owned, not claimed here before the parent runs them. A design-changing native all64/a1/.001 metric probe was run to expose the crude-mass failure; no build/lint/test/formatter was run midflight.

A scoped native **API** discriminant then exercised `native_case` at N5/a.001/centre(1,1,1), seed2221002, arbitrary full-range compact phases, with natural exit0. It observed mu=.10732736444155277, minimum singular gap74.46850726003119, minimum positive M54.6984373019637, and a^2||[M,D]||=.17407747068326276. Thus the noncommuting case was actually reached. The raw edge Berry connection was nonzero (axis2 approximately .00102696913), whereas the explicit trace-log phase reduced its connection residual to3.12e-16; a phase-free-by-fiat trial would be wrong on this native input. The exact-background Slater gauge residual was2.25e-14, native current Ward0, and spectral/polar edge QGT agreed (axis0 .055497020261751344 versus .05549702026175146). These are floating controls, not a quantum ground or an interval proof. The complete default CLI, receipt binding/serialization and final types remain parent-owned.

No full quantum EM ground was diagonalized. No rotor/angular/Fock cutoff was used. No thermodynamic deconfinement, isolated finite-energy/mobile electron, mechanical spin, original-source BO transfer, photon pole, quantum scattering amplitude, UV closure or campaign closure follows from this finite-volume effective-form theorem.

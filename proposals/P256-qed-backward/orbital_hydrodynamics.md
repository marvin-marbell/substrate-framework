# Thermodynamic orbital hydrodynamics from the actual endpoint law

This is a constructive extension of `local_orbital_stability.py`, not a
finite-box sufficient-$J$ estimate. The executable keeps **all 60 real orbital
coordinates**, splits the actual **15 physical simultaneous-conjugation modes**
from their **45-dimensional restoring complement**, and integrates genuine
continuous momentum transfers. No anticommutator penalty, artificial hard
budget, discarded complex response, or projection of negative curvatures is
introduced. The joined electron/photon campaign remains active.

## Domain, owner and units

The microscopic orbital fields are four traceless Hermitian matrices
$A_\mu(x)$, with metric
$\|\delta A\|^2=\sum_\mu\operatorname{tr}(\delta A_\mu^2)/4$.
`matrix_unit_basis()` gives 15 Hermitian matrices $B_a$ orthonormal in this
metric. Thus $A_\mu=z_{\mu a}B_a$ has 60 **real** coordinates. The original
potential per cell is

\[
 V(A)=\frac\kappa2\|G(A)-g_0I_4\|_F^2,\qquad
 G_{\mu\nu}=\tfrac14\operatorname{tr}(A_\mu A_\nu),
 \quad \kappa,g_0>0.
\]

`RotationBand.couplings()` owns the scalar onsite and every directed hopping
path, including the improved two-step mass paths. `scalar_paths()` extracts
those coefficients; `local_hamiltonian()` owns the endpoint-average law
$T_{xy}=s_\mu(y-x)[A_\mu(x)+A_\mu(y)]/2$ and adds every adjoint. The sea is the
same two occupied bands, equivalently the actual total half-filled ground
when the occupied/empty gap remains open. The primitive exclusion/charged
state supplier is unchanged. No separate Gaussian fermion spectrum is
substituted for this one-body half-filled-sea energy.

Use $\hbar=1$, energy **per primitive cell**, dimensionless momentum
$p=a p_{\rm physical}\in[-\pi,\pi]^3$, and
$\int_p=\int d^3p/(2\pi)^3$. Transfer $q$ below is also dimensionless;
$q_{\rm physical}=q/a$. The actual coefficients are obtained through
`coefficients(band, p/a)`. A microscopic second-order orbital kinetic law is
an explicit additional hypothesis,

\[
 L_{\rm orbital,kin}=\frac M2\sum_{x,\mu,a}\dot z_{\mu a}(x)^2,
 \qquad M>0.
\]

$M$ is not selected by the static sea. Optional scalar alignment is
$J\sum_{\langle xy\rangle}\|A(x)-A(y)\|^2/2$, $J\ge0$, adding
$J\ell(q)I_{60}$, $\ell(q)=4\sum_i\sin^2(q_i/2)$. The default is **$J=0$**.
This scalar term is not a Clifford penalty and does not grow with box size.
The parent owns all numerical science execution and receipt integration.

## 1. Select and really stationarize the same weighted sea

At the selected metric-Clifford frame
$F_\mu=\sqrt{g_\mu}\,\mathrm{SELECTED}_\mu$, cubic symmetry gives
$g=(g_s,g_v,g_v,g_v)$. Put
$y(p)=(x_0(p)^2,\sum_{i=1}^3x_i(p)^2)$ and
$R_p=\sqrt{g_s y_s+g_v y_v}$. The actual convex Gram energy is

\[
 f(g_s,g_v)=-2\int_p R_p+
 \frac\kappa2[(g_s-g_0)^2+3(g_v-g_0)^2].
\]

Its two exact equations and positive Jacobian are

\[
 \begin{aligned}
 d_s&=\kappa(g_s-g_0)-\int_p y_s/R_p=0,\\
 d_v&=3\kappa(g_v-g_0)-\int_p y_v/R_p=0,\\
 Dd&=\kappa\operatorname{diag}(1,3)
       +\tfrac12\int_p yy^T/R_p^3>0.
 \end{aligned} \tag{1}
\]

`select_gram(..., weights=weights)` provides the initial solution; damped
Newton iteration of (1) improves stationarity using **the very same normalized
positive weights** subsequently used in the susceptibility. The full
60-coordinate gradient is checked independently from the occupied projector.
There is no post hoc subtraction of $H(0)$, clipping of eigenvalues, or
projection of an observed unstable mode to zero. Changing quadrature changes
both the selected Gram and its sea, not just one of them.

The default midpoint partitions $6^3,12^3,24^3$ are nested cell refinements
(their midpoint abscissae are not nested). `--rule split_gauss` uses the shared
cubic positive-weight quadrature, six signed one-dimensional segments with
edges $[-\pi,-.6,-.12,0,.12,.6,\pi]$ and Legendre order **per segment**.
Its default orders are 8 and 10, with $(6n)^3$ points; those refinements are
more expensive than midpoint order $n$. Point counts and weights are recorded.
Positive cubic/inversion-symmetric quadrature is essential for the two-equation
reduction. Continuous momentum evaluation is never a roll, nearest-grid
replacement or interpolation.

## 2. Exact endpoint vertex and positive complex spectral matrices

For source $k=p-q/2$ and target $l=p+q/2$, Fourier transforming the actual
endpoint-average law gives

\[
 V_{\mu a}(l,k)=t_\mu(l,k)B_a,\qquad
 t_\mu(l,k)=\frac{x_\mu(l)+x_\mu(k)}2. \tag{2}
\]

For each directed displacement $d$, its contribution to $t$ is
$s_\mu(d)(e^{ik\cdot d}+e^{il\cdot d})/2$ plus its adjoint phase. This derives
(2) for **every path and multiplicity**, not just the nearest path. The CLI
checks the continuous path sum and all 60 actual finite-$L=3$ local vertices
against their independent spatial Fourier transforms.

Diagonalize the genuine four-by-four matrices $h(k)=x_\mu(k)F_\mu$ and $h(l)$.
For $\sigma=+$ take occupied source / empty target; for $\sigma=-$ take empty
source / occupied target. For each transition define

\[
 f_A^\sigma=\langle l,t|V_A(l,k)|k,s\rangle,
 \quad S_{AB}^\sigma=(f_A^\sigma)^*f_B^\sigma,
 \quad \Delta^\sigma=|E_t(l)-E_s(k)|>0.
\]

Each $S^\sigma$ is a **positive Hermitian $60\times60$ matrix**. Its imaginary
cross terms are physical Fourier response and are retained. For real
$|\omega|<\min\Delta$ the subthreshold susceptibility is Hermitian:

\[
 \chi(q,\omega)=\int_p\sum_{t,s}
 \left[\frac{S^+}{\Delta^+-\omega}
       +\frac{S^-}{\Delta^-+\omega}\right]. \tag{3}
\]

The static Hessian, linear frequency coefficient and positive quadratic
frequency inertia are

\[
 \begin{aligned}
 H(q)&=H_{\rm elastic}+J\ell(q)I-\chi(q,0),\\
 B(q)&=\int_p\sum_{t,s}(S^+/\Delta_+^2-S^-/\Delta_-^2),\\
 \eta(q)&=\int_p\sum_{t,s}(S^+/\Delta_+^3+S^-/\Delta_-^3)\ge0,\\
 K_R(q,\omega)&=H(q)-\omega B(q)-\omega^2[M I+\eta(q)]+O(\omega^3).
 \end{aligned} \tag{4}
\]

Equivalently $K_E(q,i\nu)=H-i\nu B+\nu^2(MI+\eta)+\cdots$.
The factor of two in the uniform static energy curvature comes from the two
spectral orientations; it is not an extra degeneracy multiplier. Eigenbasis
choices within either degenerate two-band cluster disappear from the spectral
outer products. Independent analytic projectors
$P_\pm(p)=(I\pm h(p)/R_p)/2$ and a deterministic mixed 60-coordinate trace
$\operatorname{tr}(P_s V P_t V)$ check those statements. Centered exact
finite-frequency evaluations of (3) separately check $B$, $\eta$ and the
relaxed inertia below. These controls exercise complex cross terms, not just
nonnegative diagonal response or a basis-phase convention.

## 3. Fifteen exact uniform flats and 45 restoring directions

Let $T(G)_\mu=i[G,F_\mu]$ for all 15 actual traceless Hermitian matrix-unit
generators. The selected four generators span $\mathrm{Mat}_4(\mathbb C)$, so
their common commutant is scalar and $T$ is injective. Its orthonormal column
space $O$ has dimension 15; $C=\ker O^T$ has dimension 45. They are a complete
orthogonal decomposition of the original 60-dimensional physical space.

Global simultaneous conjugation preserves the **entire** uniform sea and
quartic potential. Differentiating covariance of the gradient gives the exact
Ward identity

\[
 H(0)T(G)=i[G,\nabla E(F)]. \tag{5}
\]

Thus solving the actual stationarity equations makes $H(0)O=0$ algebraically,
not by removing numerical negative eigenvalues. The CLI checks both sides of
(5), all 60 eigenvalues, exactly 15 uniform flats, and the 45-dimensional
restoring gap. The fixed-Gram representation theorem supplies 35 positive
normal directions when the ten-product coefficient design has full rank;
the strictly convex Gram Hessian supplies the remaining ten. More formally,
the energy is its convex Gram lower bound plus a nonnegative representation
excess with the same zero set. Their Hessian kernels intersect precisely in
the 15 conjugation directions. This is the soft-law $15+45$ result, not the
fixed-budget $15+35$ result with ten directions silently omitted.

A uniform conjugation co-transports orbital states and operators and hence
leaves aligned current matrix elements invariant. **Spatially different**
conjugations are physical orbital fields under the stated local endpoint law;
they cannot all be gauged away. Their response is the object being calculated.

## 4. Restore all massive coordinates before reading stiffness or inertia

For a $q$ with $D=H_{cc}(q)>0$, write $H$ in the $(O,C)$ decomposition. The
static restoring minimizer and its physical lift are

\[
 A=D^{-1}H_{co},\qquad R=O-CA,\qquad
 H_{\rm eff}=H_{oo}-H_{oc}D^{-1}H_{co}=R^\dagger H R. \tag{6}
\]

All 45 directions participate. In particular
$H_{oo}-H_{\rm eff}=H_{oc}D^{-1}H_{co}\ge0$. A forbidden shortcut, retaining
$H_{oo}$ alone, systematically overestimates stiffness; an imaginary
$O(q)$ orbit/restoring coupling can lower $O(q^2)$ stiffness even when the
orbit block's own odd part vanishes. The CLI prints that lowering divided by
$q^2$ and separately compares the equally forbidden real-only susceptibility.
Neither control modifies the physical Hessian.

Let $E=MI+\eta$, $B_{\rm eff}=R^\dagger BR$ and $X=R^\dagger BC$. Expanding
the **finite-frequency Schur complement**, rather than merely projecting
$\eta$ on the unrelaxed orbit, gives

\[
 \boxed{\eta_{\rm eff}=R^\dagger E R+X D^{-1}X^\dagger>0.} \tag{7}
\]

This includes induced static restoring relaxation, its microscopic mass, and
induced frequency-dependent Berry/restoring relaxation. To see (7), change
variables to $\delta z=R u+C v$. The static mixed block is zero and the
frequency-linear mixed block is $-\omega X$; its Schur product is
$+\omega^2 XD^{-1}X^\dagger$, subtracted from the orbit kernel. Both terms in
(7) are positive, and $R^\dagger R=I+A^\dagger A\ge I$, so
$\eta_{\rm eff}\ge M I$. Exact finite-frequency Schur evaluation is an
independent numerical control. If $D$ is nonpositive, eliminating it is
invalid; the code instead constructs the negative direction of the full
physical matrix, with no false Schur formula or declaration of a blocker.

## 5. Detect odd branches, refine stiffness, and keep gyroscopic dispersion

### Direct thermodynamic derivatives, independently checked by small transfers

The stiffness is not defined solely by fitting the finite-$q$ sequence.
Let $N(p)=h(p)/R(p)$ and $D=R(k)+R(l)$. Summing the two spectral orientations
gives the exact full-complex projector identity

\[
 \chi_{\mu a,\nu b}(q,0)=\int_p
 \frac{t_\mu t_\nu}{D}F_{ab}(q),\qquad
 F_{ab}(q)=\tfrac12\operatorname{tr}
 [B_aB_b-N(p-q/2)B_aN(p+q/2)B_b].
\]

Primes denote derivatives along a unit wave direction at $q=0$. Then

\[
 \begin{aligned}
 F'_0&=\tfrac14\operatorname{tr}[N'B_aNB_b-NB_aN'B_b],\\
 F''_0&=-\tfrac18\operatorname{tr}
 [N''B_aNB_b+NB_aN''B_b-2N'B_aN'B_b],\\
 \chi'_0&=\int_p\frac{x_\mu x_\nu}{2R}F'_0,\\
 \chi''_0&=\int_p\left\{
 \left[\frac{x''_\mu x_\nu+x_\mu x''_\nu}{8R}
             -\frac{x_\mu x_\nu R''}{8R^2}\right]F_0
       +\frac{x_\mu x_\nu}{2R}F''_0\right\}.
 \end{aligned}
\]

$x'$ and $x''$ are differentiated directly from **every scalar path**,
using $\partial_n e^{ip\cdot d}=i(n\cdot d)e^{ip\cdot d}$, so the long
endpoint prescription is not replaced by a continuum kinetic ansatz.
For $h'=x'_\mu F_\mu$ and $h''=x''_\mu F_\mu$,

\[
 R'=\frac{x'^TGx}{R},\quad
 R''=\frac{x''^TGx+x'^TGx'-(R')^2}{R},\quad
 N'=\frac{h'}R-\frac{hR'}{R^2},\quad
 N''=\frac{h''}R-\frac{2h'R'+hR''}{R^2}
                  +\frac{2h(R')^2}{R^3}.
\]

The positive mass gap bounds denominators and permits differentiation under
the thermodynamic integral. Put $H'_n=-\chi'_0$ and
$H''_n=-\chi''_0+2JI$. Since $H_{co}(0)=0$, the exact coefficients are

\[
 L_n=O^TH'_nO,\qquad
 K_n=\tfrac12O^TH''_nO
       -(O^TH'_nC)H_{cc}(0)^{-1}(C^TH'_nO).
\]

`analytic_low_q()` evaluates these matrices without subtracting two
nearly equal physical Hessians. It checks its entire static projector
matrix against the eigenvector spectral calculation, reports $L_n$ and
all 15 $K_n$ eigenvalues, compares the full finite-$q$ Hessian to its Taylor
expansion, and compares even extrapolation to the direct derivative.
Quadrature refinement of the derivative is still an estimate, not an
interval-certified integral. A negative direct $L_n$ or $K_n$ also emits
an asymptotic modulated candidate with its explicit restoring slope,
even if the default finite transfers have not yet reached the negative
branch.

### Independent signed-transfer and frequency comparison

For axis, face-diagonal and body-diagonal transfers, the CLI independently
computes **both** $q$ and $-q$, reports the real-field conjugacy condition
$H(-q)=H(q)^*$, and measures

\[
 \frac{\|H_{\rm eff}(q)-H_{\rm eff}(-q)\|}{2|q|},\qquad
 K(q)=\frac{H_{\rm eff}(q)+H_{\rm eff}(-q)}{2|q|^2}. \tag{8}
\]

A surviving $O(q)$ term has a Hermitian odd coefficient; for a real field its
imaginary antisymmetric part has paired signs. With $H(0)=0$, an adverse
linear static branch cannot be dismissed as a harmless basis phase. The
default decreasing magnitudes are $.16,.08,.04$. Two-level even extrapolation
is

\[
 K_0^{\rm est}=\frac{q_1^2K(q_2)-q_2^2K(q_1)}{q_1^2-q_2^2},\quad q_1>q_2.
\]

All 15 eigenvalues, the last-$q$ matrix correction, successive quadrature
changes, restoring gaps and complex-mixing errors are exposed. These are
refinement estimates, **not certified continuum error bounds**. Three angles
do not prove a positive lower bound uniformly over the sphere.

Do not discard $B$: the isolated occupied sea can carry a nonzero uniform
orbital charge density $\bar P_-$. On raw generator directions the exact
identity is

\[
 T(G_a)^T B(0)T(G_b)=\operatorname{tr}(\bar P_- [G_a,G_b]). \tag{9}
\]

The executable checks (9) independently. Cubic parity leaves
$\bar P_-=(I-cF_0/\sqrt{g_s})/2$ with $c>0$ for positive mass. Its eigenvalue
multiplicities are $2+2$; its noncentral coadjoint orbit has dimension eight.
Thus the charge/Berry form generically has **rank eight**, giving four
canonical type-B pairs and seven unpaired type-A coordinates. There are 15
physical static flat directions but generically **11 low-energy modes**, not
15 independent acoustic modes. This is a charge/gyroscopic consequence, not a
spin inference from a Berry eigenbasis phase.

Equation (9) is specifically the **matter sea-induced vacuum flavor-charge**
form about the selected zero-angular-velocity bosonic orbital background.
It is not an assertion that vacuum flavor charge equals the total conserved
bosonic-plus-matter charge. A prescribed total-charge sector may require a
compensating bosonic angular-momentum background and a new expansion; that
constraint is not supplied by this static selection. The mode count above is
conditional on this background and its quadratic effective action.

The CLI solves the actual quadratic pencil

\[
 \det[H_{\rm eff}(q)-\omega B_{\rm eff}(q)
                   -\omega^2\eta_{\rm eff}(q)]=0 \tag{10}
\]

and reports positive roots, imaginary parts, $\omega/|q|$,
$\omega/|q|^2$, and $\omega/\Delta_{\rm pair}$. When $K_0>0$, the unpaired
kernel of $B(0)$ has linear type-A modes; the four nonzero pairs have a
quadratic low branch and a massive partner in the second-order orbital law.
If $B=0$ in a genuinely charge-neutral variant, all 15 branches are linear
and $\omega^2$ is determined by the generalized eigenvalues of
$(K_0,\eta_{\rm eff})|q|^2$. Physical linear velocity is
$a\sqrt{\lambda(K_0,\eta_{\rm eff})}$. For a simple canonical pair with
$B$ eigenvalues $\pm b$, isotropic stiffness $k$ and inertia $m$, the low
branch is
$[\sqrt{b^2+4mk|q|^2}-|b|]/(2m)=k|q|^2/|b|+O(q^4)$.
General mixing is handled by (10), not by imposing this simple-pair formula.
High roots comparable to the pair threshold are **not** justified by the
frequency derivative expansion; their printed ratios identify that limit.

### An unstable branch is a new variational candidate

For a negative eigenvector $u$ of the actual Schur matrix, lift
$z=Ru$ (or use the full 60-vector when the restoring block is unstable).
Construct the explicitly real, Hermitian modulated fields

\[
 A_\mu(x)=F_\mu+\epsilon\sum_a
 [\cos(q\cdot x)\operatorname{Re}z_{\mu a}
  -\sin(q\cdot x)\operatorname{Im}z_{\mu a}]B_a. \tag{11}
\]

The cosine and sine coordinates, selected frame and transfer are returned by
the CLI. Taking entrywise real parts of a complex matrix direction would be
wrong for the imaginary Hermitian matrix units; (11) instead takes the real
parts of its **coordinates**. For a nonzero nonexceptional transfer, spatial
averaging gives

\[
 e(\epsilon)-e(0)=\tfrac14\epsilon^2 u^\dagger H_{\rm eff}(q)u
                   +O(\epsilon^3). \tag{12}
\]

The local band gap and bounded finite-range perturbation imply analyticity
near zero, so (12) proves descent at sufficiently small amplitude whenever
its quadratic coefficient is strictly negative. This is a constructed
small-amplitude candidate with all induced restoring fields retained, not a
claimed measured finite-amplitude energy. A negative refined estimate alone
is not promoted to an exact sign theorem; the emitted candidate is attached
to the actual quadrature curvature which triggered it. No unstable case is
called blocked or hidden by a scalar-$J$ override.

## 6. Conditional harmonic 3D variance and same-current fluctuation bounds

Suppose the **exact** continuum Schur kernel, in every direction and for
$0<|q|<\Lambda$, obeys
$H_{\rm eff}(q)\ge\rho |q|^2 I$, $\rho>0$, and the microscopic kinetic law
above has $M>0$. This hypothesis is stronger than a few sampled positive
angles. The second-order harmonic model is well defined, including a
possibly nonzero gyroscopic $B$. At a fixed $q$, let $S_n\ge0$ be its
positive-frequency, real-field symmetrized coordinate spectral weights.
Canonical commutators and static susceptibility give, for any unit vector
$v$,

\[
 2\sum_n\omega_n v^\dagger S_n v=v^\dagger\eta_{\rm eff}^{-1}v,
 \qquad
 2\sum_n\frac{v^\dagger S_n v}{\omega_n}=v^\dagger H_{\rm eff}^{-1}v.
\]

Here weights combine the $q$ and $-q$ real-field pair when necessary. Scalar
Cauchy–Schwarz then bounds its zero-temperature covariance by

\[
 v^\dagger\langle uu^\dagger\rangle_qv
 \le\tfrac12\sqrt{(v^\dagger\eta_{\rm eff}^{-1}v)
                         (v^\dagger H_{\rm eff}^{-1}v)}
 \le\frac1{2\sqrt{M\rho}|q|}. \tag{13}
\]

This argument does not incorrectly pretend a type-B frequency is linear or
replace the covariance by $1/(2M\omega_{\rm low})$ while omitting its
canonical partner and spectral residue. Integrating (13) in three dimensions
shows the Goldstone infrared contribution is finite:

\[
 \boxed{\operatorname{tr}\langle u^2\rangle_{|q|<\Lambda}
 \le\frac{15\Lambda^2}{8\pi^2\sqrt{M\rho}}.} \tag{14}
\]

It decreases as $M^{-1/2}$ at fixed stiffness. Using
$\coth(\omega/2T)\le1+2T/\omega$ adds at most
$15T\Lambda/(2\pi^2\rho)$, also infrared finite. A low-frequency effective
model with substantial frequency-dependent relaxation requires its own
ultraviolet cutoff and matching; (14) concerns its stated canonical Gaussian
truncation. It does **not** prove nonlinear quantum order, suppression of
large orbit excursions, defect proliferation, or the full joined quantum
vacuum. The receipt only prints an illustrative candidate $\rho$ when its
sampled estimates are positive and explicitly retains the exact-uniform-
positivity hypothesis.

The variance in (14) is that of the orthonormal orbit coordinates. At small
$q$, the full relaxed field has $\delta z=R(q)u$, so physical-field variance
has the additional finite factor $\sup_{|q|<\Lambda}\|R(q)\|^2$. Analyticity,
$H_{co}(0)=0$ and a positive restoring gap give $R=O+O(q)$ and hence this
factor is $1+O(\Lambda^2)$, not a discarded restoring contribution.

For the same finite-link current, scalar coefficients $c_\mu$ give
$j(A)=c_\mu A_\mu$ and the representation-stability inequality

\[
 \|\delta j\|_F\le2|c|\,\|\delta A\|_{\rm HS},\qquad
 \langle\|\delta j\|_F^2\rangle
 \le4|c|^2\langle\|\delta A\|_{\rm HS}^2\rangle. \tag{15}
\]

Use the $R$ factor above for Goldstone waves. A **uniform** conjugation leaves
co-transported states/current matrix elements invariant rather than creating
a material current fluctuation. For genuinely non-Clifford fixed-Gram normal
fields within the representation theorem's tube,
$d_\perp^2\le12(\mathcal E-\mathcal E_{\min})/\lambda$ gives
$\|\delta j\|_F^2\le48|c|^2(\mathcal E-\mathcal E_{\min})/\lambda$ after
alignment. That finite-tube inequality is not silently extended to arbitrary
soft-Gram displacements; Gram variations have their own ten restoring
coordinates. The original isolated-projector bound
$\|P_-(A)-P_-(F)\|_{\rm op}\le\|\delta h\|_{\rm op}/(2R-\|\delta h\|_{\rm op})$
and polar subspace transport control the same charged-state/current map when
its gap hypothesis is met. A Compton resolvent additionally needs its own
pole separation; orbital Gaussian variance alone does not establish it.

## Executable and evidence boundary

From the repository root, the parent runs:

```sh
python proposals/P256-qed-backward/orbital_hydrodynamics.py --output proposals/P256-qed-backward/orbital_hydrodynamics_receipt.json
```

The chunked computation retains only four-by-four momentum eigensystems and
60-by-60 response matrices; it never constructs a thermodynamic many-body
matrix or a $60L^3$ Hessian. Defaults use three nested midpoint partitions,
three $q$ magnitudes, both signs, and three independent directions. `--batch`
limits temporary spectral features. A shorter parent-owned discrimination is
`--orders 6 12 --q .12 .06`; high-accuracy comparison can choose the shared
split-Gauss rule and explicitly selected orders. Scientific signs are
reported as outcomes, not treated as structural-check failures: a negative
mode emits (11). Structural controls must still pass.

The complete default CLI has naturally exited zero. Its receipt retains
all 60 coordinates on midpoint orders 6, 12 and 24, both signs of three
transfers, and axis/face/body-diagonal directions:

- At order 24, 45 uniform modes restore, with minimum `.01585705532`;
  all 15 physical orbit directions remain symmetry-flat.
- The Berry form has rank 8: four type-B pairs and seven type-A modes under
  the stated low-frequency hypotheses, not 15 presumed acoustic modes.
- The relaxed analytic q² stiffness is positive on every sampled direction,
  with minimum `.01364899859`. Order-12 to order-24 minimum changes are
  `3.38e-6`; the largest full stiffness-eigenvalue change is `.000156154`.
- Actual endpoint/path controls differ by at most `3.14e-15`; the
  charge-commutator Berry identity differs by `2.71e-14`.

The reported candidate `rho=.00682449929` yields harmonic infrared trace
bound `.00367947717` at M=1 and cutoff angle `.04`, **conditional on
exact all-direction continuum positivity**. Three directions and quadrature
refinement do not certify that hypothesis. The result supplies a concrete
stable fluctuation route and retains nonlinear ordering, total-charge
background choice and the full quantum continuum as active constructions.

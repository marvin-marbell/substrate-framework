# Complete classical pair moduli and the actual principal harmonic form

This construction uses the **unchanged** `FullBandGaussSection` nearest-x pair,
`Functional.evaluate/hessian`, `pair_global_field_bound.point`, the native
`packet_point_controls`, and `full_field_angular_control` from
`pair_separated_charge_transfer`. The Hamiltonian remains the full-domain law
of `pair_full_field_concentration.md`:

```
H = sqrt(Lambda)/2 p_u^2 + UE4 C_SU4 + UE1/2 (p_A0^2+p_A1^2)
    -K cos(A0+A1)
    +Lambda [V_quartic + J W/8 - dGamma(hH)] + dGamma(hL).
```

Here `Phi_mu=sum_a u_mu,a B_a`, `Tr B_a B_b=4 delta_ab`, and
`T_a=B_a/sqrt(8)` has `Tr T_a T_b=delta_ab/2`. Thus the actual Cartesian
coordinates have metric HS/4. The link chart is `U=exp(i theta_a T_a)`; its
kinetic principal cometric is **2 UE4 I15**, since its kinetic operator is
`UE4 C_SU4`, not `UE4 C_SU4/2`. Heavy fills **top4** with a negative Hamiltonian
sign; light fills **bottom4** with a positive sign. All70x70=4900 primitive
states, all120 fields, full SU4 Haar and both U1 circles are retained. No
normal, occupation, angular-representation or Gauss cut defines a new law.

The exact results below are the classical classification and the principal
quadratic metric. The computed spectra and preference between strata are
finite floating-point diagnostics. They are **not** an exact quantum ground,
a BO remainder theorem, or an interacting electron-photon phase.

## 1. Equality determines every classical minimum, not just an example

Assume the executed owner: `c=.32>|d|=.06`, `t=.15`, `kappa=2`, `J=.2>0`.
The argument also holds for `c>|d|`, `t!=0`, `kappa>0`, `J>0` and
`0<q=t^4/(c^4+t^4)<1/2`. No statement here classifies `J=0`, `t=0`,
`c=|d|` or the dual/degenerate `q>=1/2` cases.

Gauge-fix the single link to identity by transforming the right fields.
Every nonnegative term in the exact global excess decomposition must vanish.
Consequently:

* `Phi2=Phi3=0`, all offdiagonal Gram entries vanish;
* both endpoints have the optimal temporal/spatial squared norms
  `u*=c^2/(kappa R)`, `v*=t^2/(kappa R)`;
* since `J>0`, the gauge-fixed left/right fields are equal;
* the maximizing rank4 projector saturates both Cauchy inequalities.

Write its involution as `Q=2P-I=[A,B;B†,C]`, `B=X+iY_old`.
The strictly positive discarded penalty `(c^2-d^2)||X||^2/2`, together
with nonzero temporal norm, forces `X=0`. Equality in the temporal
Hilbert-Schmidt Cauchy inequality makes the two temporal fields proportional
to `A,C` with the same strictly positive factor. Their alignment therefore
forces `A=C`. Define `Y=-Y_old`; equality in the spatial inequality then gives

```
Q = [A,-iY; iY,A],
A^2+Y^2=I,  AY+YA=0,  Tr A=Tr Y=0,
Tr Y^2=4q,
Phi0=(R/c) A,  Phi1=(R/t) Y,
R=[(c^4+t^4)/kappa]^(1/3).
```

For example the temporal factor is `sqrt(u*/(1-q))=R/c`, not an
independently assumed Clifford normalization. The remaining scalar Cauchy
equality determines `q`, rather than leaving it free. These are **necessary**
conditions obtained from every global equality point.

### Canonical classification, including the rank-loss endpoint

The central operator `A^2` decomposes C4 into invariant eigenspaces. On an
intermediate eigenvalue `0<a^2<1`, anticommutation makes `Y` an isomorphism
between equal-dimensional `+a` and `-a` eigenspaces. A singular-value basis
reduces them into two-dimensional blocks

```
A=a sigma_z,  Y=y sigma_x,  a^2+y^2=1,  a,y>0.
```

At `a^2=1`, `Y=0`, giving one-dimensional `A=+1` or `-1` summands. At
`a^2=0`, `A=0`, giving one-dimensional `Y=+1` or `-1` summands. Intermediate
blocks contribute zero to both traces; the two trace constraints therefore
pair the `+1,-1` one-dimensional summands separately. A pair of the latter
Y-summands already has `Tr Y^2=2`, forbidden by `q<1/2`. Thus A is invertible,
has two positive and two negative eigenvalues, and all Y-zero summands pair
into one degenerate two-dimensional block. This excludes extra endpoint
orbits rather than silently assuming both blocks remain irreducible.

There are exactly two such blocks, up to permutation and conjugation. Let
`y1^2<=y2^2`. The trace constraint is `y1^2+y2^2=2q`. Hence, uniquely,

```
delta in [0,q],
a1^2=1-q+delta,  a2^2=1-q-delta,
y1^2=q-delta,    y2^2=q+delta,
A=diag(a1,-a1,a2,-a2),
Y=blockdiag(y1 sigma_x,y2 sigma_x).
```

A U4 conjugation implementing this reduction can always be multiplied by a
scalar phase to lie in SU4 without changing its conjugation action. Signs
and block phases are removable by conjugation; exchanging blocks removes
negative delta. There are no other minimum orbits in the stated regime.

**Sufficiency** is also checked algebraically. Put
`Q0=I_site tensor A + sigma_y,site tensor Y` and
`Z=sigma_x,site tensor A`. Then `Q0^2=I`, `[Q0,Z]=0`, and

```
hH = R Q0 - (dR/c) Z.
```

Since `||Z||<=1` and `|d|<c`, the actual positive/top4 cluster is precisely
the Q0=+1 cluster. The eight eigenvalues are, block by block,
`+R +/- (dR/c) ai` and `-R +/- (dR/c) ai`; their actual top4 sum is4R.
The quartic isR and alignment is0, so every representative has energy-3R.
This completes necessity and sufficiency on the unrestricted original domain.

### Stabilizers, unreduced manifold and Gauss quotient

At delta0 the two identical irreducible blocks have commutant U2;
intersecting with SU4 gives Lie dimension3. For `0<delta<q`, the inequivalent
irreducible blocks have commutant U1xU1, SU4 dimension1. At deltaq, the
Y-zero block splits into the two distinct A=+1,-1 characters and the other
block stays irreducible: commutant U1xU1xU1, SU4 dimension2. Thus common gauge
orbit dimensions are12,14,13. The full two-site gauge orbit in R120xSU4 has
dimension `30-stabilizer_dimension`, respectively27,29,28. The single link
has no additional physical colour holonomy on this tree.

The quotient of the minimum set by both local SU4 actions is exactly the
closed delta interval, with its distinct isotropy strata, **not** a unique
Clifford point. Its two endpoints are singular orbit quotients. This does
not mean the unreduced active-field minimum manifold is singular. Choose
A's positive two-plane (the Grassmannian Gr(2,C4), real dimension8). In its
positive/negative splitting write

```
Y=[0,S;S†,0],
A=[sqrt(I-SS†),0;0,-sqrt(I-S†S)],
Tr S†S=2q<1.
```

The square roots are uniformly positive. S runs over an S7 sphere in the
8-real-dimensional homomorphism fibre. These charts give a smooth compact
15-dimensional sphere bundle over the Grassmannian. Common SU4 quotient
singularities remain and must not be replaced by a globally flat delta
measure. In particular near the Clifford point the traceless Hermitian
splitting of S's equal singular values has three Cartesian tangent
components before the residual SU2 quotient; at the rank-loss endpoint the
vanishing complex offdiagonal block supplies two. Stabilizer jumps do not
license throwing away either component.

## 2. Full135 Hessian and the native mixed kinetic metric

Let `Hc` be the full120 Cartesian Hessian at a minimum, from the actual
projector derivatives of `Functional.hessian`. Gauge fixing maps nearby
fields/link into `f=(uL, Ad_U uR)`. Its linearization is

```
D[:,a] = coordinates(i[T_a,Phi_R])   (60 by15),
L = [ I120, (0;D) ],               (120 by135).
```

Since the Cartesian gradient is zero, no second-derivative-of-coordinate-map
term contributes: the exact full135 Hessian is `H135=L^T Hc L`. Its gauge
columns are

```
VL=(D,0,+I15),  VR=(0,D,-I15),  V=[VL,VR].
```

`L VR=0`; `L VL=(D,D)` is a common conjugation tangent annihilated by Hc.
Thus `H135 V=0` for **every** local generator, not just a stabilizer basis.
At a generic interior minimum the nullspace is larger than gauge orbits;
those additional moduli/soft directions are real physical directions.

The native kinetic cometric is
`Q=diag(sqrt(Lambda) I120, 2UE4 I15)`. The oscillator squared-frequency
matrix is `Lambda Q^(1/2) H135 Q^(1/2)`. Factor out Lambda^(3/2) and put
`eta=2UE4/sqrt(Lambda)`. The scaled matrix is

```
K_eta = diag(I120,sqrt(eta) I15) H135 diag(I120,sqrt(eta) I15).
```

In mass-normalized coordinates, gauge directions are
`diag(I120,1/sqrt(eta) I15) V`, **not** the coordinate columns V. K_eta
annihilates them. Therefore restriction to their orthogonal complement has
exactly the same positive spectrum as the full135 matrix. This is a
linearized scalar principal-form quotient, not a claim that every horizontal
linear coordinate at an isotropy point is an independent smooth invariant
coordinate. The invariant oscillator vacuum is permitted; its excitations
must still satisfy residual isotropy and matter Gauss covariance.

Residual isotropy is an actual fibre issue, not just a nullspace count.
At the endpoint the canonical generator
`T=diag(1/2,-1/2,0,0)` fixes all fields and the identity link. Its heavy-top
filled state has charge+1 and its light-bottom filled state has charge-1:
the first block's A=+1 modes both lie in the heavy top cluster, and its A=-1
modes both lie in the light bottom cluster. The paired fibre has charge0.
The native producer checks every stabilizer generator's paired charge and
the full normal Hessian's commutator with the corresponding adjoint action.
Thus its local normal Gaussian vacuum has a permitted invariant paired
fibre, including at rank loss. Treating the heavy-only endpoint fibre as a
trivial scalar would be wrong; light is needed even though its energy is
lower order in Lambda.

Independently the positive spectrum equals that of

```
M_eta^(1/2) Hc M_eta^(1/2),
M_eta=I120 + eta (0;D)(0;D)^T.
```

This follows from equality of nonzero spectra of `BB^T` and `B^TB`, or the
exact gauge-fixed cotangent form. It adds **cometrics**, not inertias. No
frozen-field link Hessian estimates the full mixed normal frequencies.

Let `C_eta=1/2 Tr sqrt(K_eta)` on its positive spectrum and
`C0=1/2 Tr sqrt(Hc)`. Positive-matrix order gives the exact bound

```
C0 <= C_eta <= sqrt(1+eta ||D||op^2) C0.
```

Indeed the nonzero spectrum also equals that of
`Hc^(1/2) M_eta Hc^(1/2)`, sandwiched between Hc and
`(1+eta ||D||op^2) Hc`; use eigenvalue monotonicity, then sum square roots.
On the compact classical manifold `||D||op^2<=Tr D^TD=4(u*+v*)`, by the
adjoint SU4 Casimir with these actual T generators. Consequently convergence
is uniform **for this local principal harmonic model**:

```
E_quadratic = -3 Lambda R + Lambda^(3/4) C_eta,
C_eta -> C0 = halfTr sqrt(positive Cartesian Hessian).
```

Thus the parent's Cartesian coefficient was the correct candidate leading
coefficient of the complete native principal form at fixed UE4; it was not
invalidated by missing fifteen link coordinates or by the complete Gauss
quotient. This result does **not** prove the analogous asymptotic expansion
of the original full quantum ground energy. When UE4 follows the separate
trajectory Lambda^(-11/12), eta decays still faster, as Lambda^(-17/12);
that does not solve the moduli selection problem either.

## 3. Soft, Born-Huang and endpoint obligations

The native spectrum gives79 positive and41 quadratic-flat Cartesian modes;
full135 gives79 positive and56 flat. This is a finite numeric rank result,
not an interval certificate for all delta. Twenty-six of the Cartesian
flat directions are symmetric unused-field perturbations: each of Phi2,
Phi3 can lie in the13-dimensional HS/4 orthogonal complement of Phi0,Phi1.
For the probe's symmetric perturbation with a HS/4-unit matrix at **each**
site, the full120 direction has norm `sqrt(2)` and its excess is exactly
`kappa epsilon^4`. A **unit full120** symmetric coordinate instead has
`Phi2,L=Phi2,R=epsilon B/sqrt(2)`, giving `kappa epsilon^4/4`.
Heavy hH is unchanged and alignment stays zero in both normalizations.
The remaining15 active flat directions are tangent to the minimum manifold.

The balance `sqrt(Lambda) p^2 + Lambda x^4` gives soft width Lambda^(-1/12)
and soft energy Lambda^(2/3), lower order than Lambda^(3/4), but not zero.
Cross-coupled quartic soft coefficients and their moduli dependence are not
computed as quantum eigenvalues. They must be controlled, not discarded.
Tangential kinetic energy is of order sqrt(Lambda) before localization.
At a minimum of a nonconstant harmonic coefficient, induced tangential
localization can introduce other intermediate powers. No one-dimensional
flat-delta kinetic operator is assumed at either isotropy endpoint.

The actual projector metric sums
`sum_empty,occupied |<empty|dh|occupied>|^2/gap^2`, separately for heavy top4
and light bottom4. The producer computes all120 field, all15 link and both
phase directions and compares selected entries against actual native4900
section derivatives after Berry projection. The diagonal field Born-Huang
term is `sqrt(Lambda)/2 times this metric trace`; link and phase contributions
have their original fixed electric coefficients. The matrices remain smooth
at both stabilizer endpoints: the heavy gap is at least
`2R(1-|d|/c)>0`, A's local gap is uniformly positive in the small-q chart,
and light at phase0 has the same separated cluster. A divergent formula
obtained by dividing a quotient delta derivative by a vanishing singular
value is a coordinate artifact, not an observed divergent Cartesian
Born-Huang scalar. This local check is not a global light gap on all T2 or a
uniform BO error bound.

The native Born-Huang field coefficients at fractions0,.37,1 are
29.1371464438,29.1371967547,29.1375139459, respectively. They stay finite at
rank loss and slightly prefer Clifford in this **subleading local term**,
unlike the leading harmonic prediction. Adding those terms is not a computed
full quantum ground or a justified finite-Lambda crossover calculation.

Neither the principal quotient nor the Born-Huang diagnostic removes the
actual field angular Gauss term. Native controls use joint field-plus-link
derivatives and both species' primitive Q operators. Holding fields fixed
while moving the link is an explicit adverse control; it fails both the
Hessian null condition and the native Gauss covariance. Actual noncentral,
independently gauge-rotated non-Cliff minima test the full owners too.

## 4. What the receipt decides, and what remains

The default run samples the **complete classified interval**, adds adversarial
fractions .013,.37,.91,.999, and runs a bounded scalar search. These are
predictions, not proofs of monotonicity or a global harmonic selector. It
records every full120 eigenvalue, every positive full135 principal frequency
at Lambda1, exact kinetic-metric upper/lower controls, and complete Gauss
horizontal checks. The Clifford/interior/endpoint points additionally record
Lambda1e4,1e12,1e24 frequencies, actual full135 mixed second derivatives,
soft quartic controls, all30 colour plus both EM primitive covariances, and
Born-Huang metrics. Independent local SU4 transformations produce genuine
noncentral links, not a central-link substitute.

A numerically smaller C0 at an exhibited non-Cliff point rejects the working
hypothesis that this leading **harmonic diagnostic** picks the Clifford point.
It does not identify the original quantum ground or prove endpoint
selection. In particular five monotonically decreasing values, even replaced
by a denser grid, are never used as an analytic theorem.

Observed default values (UE4=.3, these are floating-point evaluations):

| delta/q | stabilizer dim | full gauge rank | horizontal flat count | C0 | C_eta at Lambda1 |
|---|---:|---:|---:|---:|---:|
| 0 (Clifford) | 3 | 27 | 29 | 32.4155689477 | 32.4870986995 |
| .37 | 1 | 29 | 27 | 32.4152878720 | 32.4868142074 |
| 1 (rank-loss non-Cliff) | 2 | 28 | 28 | 32.4134919367 | 32.4849965430 |

At Lambda1e24 the three scaled full-Gauss coefficients agree with C0
within8e-14, below their explicit finite-eta upper bounds. The endpoint
Clifford-square defect of the actual Phi0 is.028014489, not numerical zero.
All37 sampled minima attain-3R to3.34e-16. Native mixed full135 second
derivatives agree with the analytic Hessian within6.60e-8. Actual colour
Gauss residuals are below1.53e-15 even at noncentral links; omitting the
Cartesian field term gives defects .699 through .707 at the canonical
points. The full coordinate gauge Hessian null residual is below2.02e-15,
whereas a false pure-link gauge direction gives defect at least .4149.

The bounded search finds fraction .9999999849. Near-Cliff even splitting
second differences at fraction steps .02,.01,.005 are approximately
-.00409879,-.00409877,-.00409877. Their convergence is a local curvature
diagnostic, not an analytic global sign proof. The endpoint stabilizer probe
also finds a heavy-only action norm1 cancelled by light-only norm1, with
paired residual below2.25e-16; Hessian isotropy commutators are below2.43e-15.
The native fundamental Haar Casimir is15/8; using unnormalized B in place
of T gives an explicit factor8 error, while using UE4/2 halves its electric
energy. These adverse controls matter at the actual kinetic normalization.

The exact missing premise for a genuine selection result is twofold:

1. An analytic global minimizer characterization of the explicitly specified
   matrix function `C0(delta)=halfTr sqrt(Hc(delta))`. The receipt's grid,
   local curvature differences and bounded optimizer do not supply it.
2. A uniform full-domain BO/semiclassical lower bound and a matching
   **admissible Gauss-covariant** upper construction with remainder
   `o(Lambda^(3/4))`, including soft26 confinement, the complete15D minimum
   manifold, residual stabilizer action, Haar/quotient measure, Berry and
   Born-Huang terms, and the light/phase sector. Such a theorem would imply
   `E0=-3Lambda R+Lambda^(3/4) min_delta C0(delta)+o(Lambda^(3/4))` and
   concentration on that coefficient's minimizers. It is not inferred from
   the existing weaker invariant-moment bound.

The next campaign construction therefore cannot assume a unique pair Gamma
frame. Carry the singular-value moduli into the full quantum construction;
the actual principal spectrum predicts a **non-Cliff**, rank-loss stratum
rather than supporting that import. The three-axis bulk Gram theorem is a
different geometry and is not contradicted by this pair classification.

## Reproducible consumer

From the repository root, using its scientific NumPy/SciPy Python:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/pair_minimizer_moduli.py --samples 33 \
  --output proposals/P256-qed-backward/pair_minimizer_moduli_receipt.json
```

The receipt binds all inherited owner sources/proofs and this producer/proof
by SHA256, rejects an intervening source change, and explicitly reports
`quantum_ground_computed=false` and `campaign_complete=false`. No source
owner is edited. Child `xd://lsp` is unavailable in this session; positive
Python AST navigation of `Functional.hessian` worked. Final native diagnostics
and project-wide validation belong to the parent integration boundary.

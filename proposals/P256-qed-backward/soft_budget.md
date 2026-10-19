# Soft bosonic orbital dynamics selects the spinor algebra and cubic isotropy

The first [sea-selection theorem](sea_selection.md) used a hard Hilbert–Schmidt four-frame budget. This iteration removes that constraint. A soft quartic orbital potential and the genuine half-filled sea now select both the Gram matrix and the Clifford structure. Isotropy follows from the cubic input symmetry and a unique convex minimum, rather than being imposed on the orbital matrices.

## 1. Unrestricted orbital fields and a physical variational energy

Let four uniform orbital fields `A_mu`, mu=0..3, be **arbitrary traceless Hermitian 4-by-4 matrices**. They have 60 real matrix-unit coordinates. No norm, orthogonality, square or anticommutator constraint is imposed. Their Gram matrix is

\[
G_{\mu\nu}=\tfrac14\operatorname{tr}(A_\mu A_\nu)\succeq0.
\]

Use the same scalar local-link coefficients x(p) as the original band, but allow the fields themselves to vary:

\[
h_A(p)=\sum_\mu x_\mu(p)A_\mu,\qquad
V_{\rm elastic}(A)=\frac\kappa2\|G-g_0I\|_F^2,
\quad\kappa>0,\quad g_0\ge0.
\]

This is quartic in ordinary bosonic orbital coordinates and depends only on their second-moment Gram matrix. It has no Clifford/anticommutator penalty. Define the total energy per cell as the **actual global fixed-total half-filled ground energy** of the quadratic fermionic sector plus this elastic potential. The fermionic sector is supplied by the primitive bosonic code construction. Alternatives may redistribute particles between momentum blocks; they are not artificially held at two particles in each block.

Uniform orbital fields, the scalar hopping geometry, traceless adjoint representation, positive elastic constants and total half filling remain the model hypotheses. This is an orbital mean-field/Born–Oppenheimer potential, not yet a theorem for arbitrary spatially fluctuating matrix fields.

## 2. Exact reduction to a convex Gram problem

For each traceless four-level block, its freely occupied negative energy satisfies

\[
\sum_{\lambda<0}\lambda\ge-\sqrt{\operatorname{tr}h_A(p)^2}
=-2\sqrt{x(p)^TGx(p)}.
\]

For k negative eigenvalues, the sharper first bound follows from zero trace and Cauchy–Schwarz as `sum_negative >= -sqrt(k(4-k) tr(h²)/4)`. Filling any fixed total subset cannot lower the freely occupied negative-energy sum. Therefore

\[
\mathcal E(A)\ge F(G):=-2\langle\sqrt{x^TGx}\rangle_p
+\frac\kappa2\|G-g_0I\|_F^2.
\]

**Every** positive semidefinite G has a saturating four-orbital witness: choose any square root `G=LL^T` and put `A_mu=L_mu,nu Gamma_nu`, where the four matrix-unit Clifford operators Gamma are an available witness, not an admissibility restriction. Then

\[
\{A_\mu,A_\nu\}=2G_{\mu\nu}I,\quad
h_A(p)^2=(x^TGx)I,
\]

and each gapped block has two negative and two positive states. This witness already satisfies total half filling. Thus the global minimum of the original unrestricted matrix problem is **exactly** the minimum of F(G), not a variational estimate.

Negative square root of a positive linear functional is convex. The elastic term is kappa-strongly convex and coercive, so F has a unique finite minimizer G*. It is strictly positive definite on the rich input: if G*z=0 for a unit z, adding `epsilon zz^T` changes the elastic energy by `-kappa g0 epsilon+O(epsilon²)`. For g0>0 that alone excludes the boundary. For **g0=0**, the spanning positive-weight coupling vectors make the negative-sea term decrease strictly, at order epsilon or faster, while the elastic cost is only order epsilon². A boundary Gram matrix therefore cannot minimize F in either case.

On a full-rank quadratic momentum design, equality at the global minimum forces `h_A(p)^2=(x^TG*x)I` at every included momentum. The [rank-ten argument](sea_selection.md) gives

\[
\boxed{\{A_\mu,A_\nu\}=2G^*_{\mu\nu}I.}
\]

The metric Clifford structure is now a consequence of the unrestricted dynamics.

## 3. Cubic symmetry selects isotropy

The actual full cubic momentum set is invariant under independently reversing each spatial momentum and permuting spatial axes. The scalar mass is even and cubic-symmetric. F(G) is consequently invariant under the corresponding sign/permutation action on indices1..3. Its **unique** minimizer must be invariant too:

\[
\boxed{G^*=\operatorname{diag}(g_M,g_T,g_T,g_T),\qquad g_M,g_T>0.}
\]

No isotropic matrix relation was assumed. The two selected scales obey the stationary equations

\[
\kappa(g_M-g_0)=\left\langle\frac{M(p)^2}{R(p)}\right\rangle,\quad
3\kappa(g_T-g_0)=\left\langle\frac{\sum_i f_i(p)^2}{R(p)}\right\rangle,
\quad R=\sqrt{g_MM^2+g_T\sum_i f_i^2}.
\]

Both exceed g0 on the rich massive cubic set. Strong convexity also gives the positive stability estimate

\[
\boxed{\mathcal E(A)-\mathcal E_*\ge\frac\kappa2\|G-G^*\|_F^2.}
\]

A nearly minimal physical state therefore has a controlled nearly selected orbital metric. Further algebra/orbit stability strengthens this with the separate representation-Hessian theorem.

### Zero-preferring bare potential

The admissible endpoint \(g_0=0\) gives
\(V=\kappa\|G\|_F^2/2\), whose unique bare minimum is **all sixty fields
zero**. The nonzero selected Gram, spinor shape and effective mass are then
induced by the actual occupied sea, not a prescribed nonzero orbital norm.
At the all-zero Gram corner the sea decreases as minus the square root of
the Gram scale, dominating the quartic elastic cost. The strict interior
argument above gives positive scalar and vector scales.

`soft_budget.py --preferred 0` exercises this exact endpoint; it does not
replace zero with a numerical positive floor. The optimizer uses positive
log-excess coordinates and certifies the original Gram stationarity equations
with Newton refinement. `soft_budget_zero_receipt.json` records the actual
zero-preferring CLI and its unrestricted competing matrices and current.
This remains a uniform occupied-sea energy theorem, not a quantized 60-field
ordered-phase proof.

## 4. The same spin/current consumer survives the change

Normalize `Gamma_0=A_0/sqrt(g_M)` and `Gamma_i=A_i/sqrt(g_T)`. These satisfy the original derived Clifford algebra, so `J_i=-i Gamma_j Gamma_k/2` gives SU(2), Casimir3/4 and a negative full turn. The selected local Hamiltonian is exactly the existing band with

\[
m_{\rm eff}=\sqrt{g_M}m,\quad r_{\rm eff}=\sqrt{g_M}r,\quad
v_{\rm eff}=\sqrt{g_T}v.
\]

The same-link Peierls paths, current and contact transform together. The improved mass still has zero quadratic curvature, while heavy corners retain their Wilson coefficient. The microscopic photon coefficients can then be matched to this selected matter velocity using the spectral construction in [photon_matching.md](photon_matching.md). The constants are model inputs; these are derived effective relations, not predictions of the measured electron parameters.

## 5. Executed mechanism

`soft_budget.py` completed naturally with all physical controls true, source `ccc301feeab81a885a544353e3ca6d1de392f4fc42e9f9e8f44062e1b2295d3d`. Inputs: improved scalar law `a=1,m=.2,r=v=1`, the full L3 cubic grid, `g0=1`, eight seeded unrestricted 60-coordinate alternatives, and three elastic stiffnesses:

| kappa | g_M | g_T | Effective mass | Effective velocity |
|---|---:|---:|---:|---:|
| .1 | 13.67248 | 1.274723 | .739526 | 1.129036 |
| 1 | 3.491142 | 1.053957 | .373692 | 1.026624 |
| 10 | 1.389647 | 1.008425 | .235767 | 1.004204 |

Selected actual occupied energies equal the convex global lower bound to `4e-15`; stationary residuals stay below `9.6e-9`. All unrestricted alternatives have positive actual excess, and the independently constructed Gram lower bounds remain above the selected minimum. Exact reframed local-band mapping errors are below `3.56e-14`; the derived current's finite-link Ward residual is below `9.63e-17`.

```sh
python proposals/P256-qed-backward/soft_budget.py --output proposals/P256-qed-backward/soft_budget_receipt.json
```

The constructive advance is a stronger starting theorem: bosonic elastic orbital dynamics and occupied-state energy select the spinor structure, isotropy and effective scales together. Next improvements concern spatial/quantum fluctuations and the photon join at these newly selected scales, not restoration of the removed hard frame constraint.

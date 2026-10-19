# Periodic uniform whole-heavy sea on all sixty field coordinates

This supplier removes the open-cube boundary from the **classical uniform-field**
competition. It does not select a quantum phase, prove a full spatial minimum,
or replace the joined electron–photon campaign by a finite-grid subresult. The
bare law is unchanged: `a=1, m=.2, r=.12, v=.3, improved=False`, zero Gram
preference, `kappa=2`, and `Jpref=.2`. Scalar inertia, rotation-only angular
curvatures, centre-Higgs or spin claims from earlier selected-frame receipts are
not imported into this Cartesian supplier.

## Source-owned field and hopping law

Let the fifteen Hermitian traceless matrices `B_a` be exactly
`sea_selection.matrix_unit_basis()`, with

\[
 \operatorname{Tr}(B_aB_b)/4=\delta_{ab},\qquad
 \Phi_\mu=\sum_{a=1}^{15}C_{\mu a}B_a,\quad
 C\in\mathbb R^{4\times15},\quad G=CC^T.
\]

All sixty coordinates vary independently. There is no fixed Gram budget, no
orthogonality/Clifford constraint, no identity-matrix direction and no
rotation-only restriction. `sea_selection.coefficients(band,p)` owns the four
real functions `x_mu(p)` of the actual `RotationBand`; the band has no
`coefficients` method. In particular, for the unimproved bare law,

\[
 x_0(\theta)=m+\frac r a\sum_{i=1}^3(1-\cos\theta_i),\qquad
 x_i(\theta)=\frac v a\sin\theta_i,\quad\theta=ap.
\]

These formulas describe, rather than independently supply, the coefficients.
The implementation calls the owner and checks its result against the directed
scalar paths extracted by `local_orbital_stability.scalar_paths` from
`RotationBand.couplings()`. If `s_0` is the onsite scalar and `(d,s_d)` the
complex four-component path coefficients, the independent Fourier construction
is

\[
 h_C(\theta)=s_0\Phi_0+
 \sum_d\left[e^{i\theta\cdot d}\sum_\mu(s_d)_\mu\Phi_\mu+
 e^{-i\theta\cdot d}\sum_\mu(s_d)_\mu^*\Phi_\mu\right]
 =\sum_\mu x_\mu(\theta)\Phi_\mu.
\]

The momentum `.19,-.31,.47` divided by `a` is a nontrivial positive control:
it compares the selected primitive `RotationBand.h_from_couplings` with both
owners, and compares an actual changed frame with
`sea_selection.h_from_couplings`. Reversing the Fourier phase is required to
fail by a non-negligible amount. Basis normalization, trace and Hermiticity
are also checked. These are runtime consumer controls, not permanent source
or wiring tests.

At identity links a uniform field has exactly zero alignment energy:
`Jpref/8 sum_bond,mu ||Phi_xmu-U Phi_ymu Udag||_HS^2=0`. The existing quartic
per cell is

\[
 V(C)=\frac\kappa2\|CC^T\|_F^2.
\]

Deleting alignment on this uniform identity-link slice is an exact evaluation,
not a change of its bare coefficient or a claim about spatial fluctuations.

## Actual whole-heavy occupation, not absolute values

For a midpoint mesh with `N=n^3` momentum blocks, sort the **entire** `4N`
onebody eigenvalue multiset and let `O` be its largest `2N` modes. The heavy
sign is minus the onebody Hamiltonian, so

\[
 F_n(C)=V(C)-\frac1N\sum_{(p,j)\in O}\lambda_j(p).
\]

The implementation does not force two filled modes in every block when an
unrestricted trial changes the spectral signatures. It explicitly selects the
largest `2N` global eigenvalues in that case. When every block has two strictly
negative and two strictly positive eigenvalues, its two positive modes are
exactly `O`, and the faster top-two-block calculation is justified. Final
stationary candidates must be in that checked two-positive/two-negative regime.
For general trials, `-Tr|h|/2` is not the fixed-total-half-filled sea: neither
spectral pairing nor a zero chemical potential follows just from tracelessness.
The receipt reports the block occupations, spectral signature, direct block
gaps, global occupied/empty gap and actual pairing defect separately.

The differentiated functional requires the global cluster separation

\[
 \gamma_n=\min_{O}\lambda-\max_{O^c}\lambda>0.
\]

Internal degeneracies in either cluster are allowed. A vanishing global gap
is a genuine nondifferentiable branch boundary, not cured by a denominator
floor, smoothing or fake fallback.

## Exact all-sixty gradient and Hessian

For a coordinate `alpha=(mu,a)`, write
`D_alpha(p)=x_mu(p) B_a`, and let `P_O(p)` be the spectral projector onto those
local modes selected by the global occupation. With equal normalized weights,

\[
 (\nabla F_n)_{\mu a}
 =2\kappa(GC)_{\mu a}
 -\frac1N\sum_p x_\mu(p)\operatorname{Tr}(P_O(p)B_a).
\]

For an arbitrary real sixty-dimensional perturbation `dC`,

\[
 dG=dC\,C^T+C\,dC^T,\qquad
 H_V[dC]=2\kappa\left[dG\,C+G\,dC\right].
\]

In a diagonalization `h(p)u_j= lambda_j u_j`, the sea Hessian is the bilinear
form

\[
 (H_{sea})_{\alpha\beta}
 =-\frac2N\sum_p\sum_{e\notin O(p),\,o\in O(p)}
 \frac{\Re\left[
  \langle u_e,D_\alpha u_o\rangle^*
  \langle u_e,D_\beta u_o\rangle\right]}{\lambda_o(p)-\lambda_e(p)}.
\]

The denominator is strictly positive by **global** cluster separation. Only
occupied/empty divided differences appear: no division by eigenvalue
splittings within a degenerate cluster, and no chosen basis of a degenerate
cluster can change the summed result. This is the exact derivative of the
cluster projector. Contributions connecting different momenta vanish because
a uniform perturbation is block diagonal; they are not omitted nonzero matrix
elements. `H=H_V+H_sea` is a real symmetric `60 x 60` matrix. It is reported
in full, as well as its spectrum; no Hessian pseudoinverse is interpreted as
physical mass or inertia.

Selected directional central differences compare the actual sea energy with
the analytic gradient, and the actual projector gradient with `H*dC`, at two
steps. The off-stationary control perturbs **all sixty coordinates** of the
environmental frame. It uses a radial direction, a coordinate column chosen
by actual sea curvature, and two deterministic full-space directions.
Changing the heavy-sea sign must fail both a gradient and a Hessian control.
The radial control avoids the tautology of testing a zero gradient at an
already stationary point. The chosen curvature column avoids the fact that
a degree-one sea has zero radial second derivative. The fine-grid candidates
get the same independent finite-difference checks at perturbed points.

## Actual stabilizer, orbit, normal directions and Ward controls

For the normalized traceless Hermitian generators `T_a=B_a/sqrt(8)`, construct
all fifteen orbit columns

\[
 \tau_a(C)=\operatorname{coords}(i[T_a,\Phi_\mu]).
\]

Their numerical rank is the common conjugation orbit dimension `r`. The
nullspace of this actual `60 x 15` map is the Lie stabilizer, of dimension
`15-r`. Independently, the complex map
`A -> ([A,Phi_0],...,[A,Phi_3])` is built on the full sixteen-dimensional
matrix basis consisting of the identity and all fifteen `B_a`; its complex
nullspace is the actual common complex commutant. For Hermitian fields its
dimension is mathematically `16-r`, but the receipt computes it rather than
borrowing a stabilizer label from the seed. Both singular spectra, thresholds
and nullspace residuals are visible. Approximate rank is explicitly numerical.

Orthonormal bases of the orbit and its Euclidean normal complement use the
same stated rank threshold. The receipt gives the orbit-projected and
normal-projected Hessian spectra, `H` applied to normalized orbit vectors,
and the orbit/normal cross block. It distinguishes orbit zeros from
additional unresolved normal flats. Strict positive normal curvature,
small full-gradient residual and a separated cluster earn only a **finite
numerical local minimum modulo the common SU4 orbit**. Negative curvature
identifies a saddle; extra semidefinite normal flats do not earn a nonlinear
minimum claim. Floating eigenvalues are not rigorous endpoints.

The potential and sea are invariant under a common unitary conjugation.
Let `L_T` be its infinitesimal real coordinate action. The exact identities are

\[
 \nabla F\cdot\tau_T=0,\qquad H\tau_T=L_T\nabla F,
 \qquad \tau_T^TH\tau_T+\nabla F\cdot L_T^2C=0.
\]

The acceleration is `L_T^2 Phi=-[T,[T,Phi]]`. The program checks all fifteen
identities, including at off-stationary derivative-control points. It also
checks finite common conjugation invariance and gradient covariance. Away
from stationarity `H*tau=0` would be a wrong Ward assertion. These controls
are common-orbit identities of the current Cartesian sea, **not** local
SU4/EM quantum Gauss, current/contact Ward, or photon dynamical claims.

## Independent competitors and two different finite stationarity questions

The three initial coordinate sources are genuinely different:

1. The existing three-axis environmental Clifford frame from
   `uniform_data(band,3,kappa,0,basis)`.
2. The existing balanced two-field pair reference from the actual
   `FullBandGaussSection` and
   `pair_clifford_reference(model, stiffness=kappa)`.
3. The actual all-sixty uniform coordinates at
   `/cube/best_candidate/coordinate_values` in
   `unrestricted_field_minimum_receipt.json`.

The pair formula concerns its pair family only. Its energy, stationarity and
stabilizer are recomputed under the three-dimensional periodic functional;
none of its pair receipts are evidence of bulk stationarity. Likewise no
open-cube energy or stabilizer is copied. The input receipt's complete bytes
are SHA256-bound before and after execution; its bare parameters and basis
owner revision must agree. The new receipt uses its coordinates only.
Executing inherited source owners and both new supplier/proof files are
SHA256-bound, and a changed source aborts the run instead of mixing laws.

Each branch first receives both coarse- and fine-mesh evaluations at the
unchanged seed. Existing `unrestricted_field_minimum.refine` supplies
L-BFGS descent, actual-residual repair and deterministic negative-curvature
escapes, now using this supplier's sixty-dimensional functional. Its numerical
preconditioner is an optimizer device, not physical inertia. A coarse refined
candidate is evaluated **frozen** on the finer mesh; this may have a nonzero
fine gradient. It is then refined again on the finer mesh and receives a
separate stationarity/Hessian report. Three rankings expose coarse, frozen
fine and refined fine energies under the same law. Rejected branches remain
visible and do not magically become stationary candidates; control success
must not be confused with all three branches completing.

## Boundary and finite quadrature limits

The dimensionless mesh is
`theta_k=-pi+(k+.5)*2*pi/n` along every axis. For even `n` it is the momentum
mesh of an antiperiodic length-`n` torus; for odd `n` it is the same periodic
mesh up to relabelling. In general its seam phase is `(-1)^(n+1)`. This is not
an open cube. At fixed coefficients and uniform fields, these bounded shifts
have the same Brillouin-zone integral limit. Mesh volume is `n^3` cells;
energies are per cell, not total energies of the old eight-site cube.

A path-only operator-norm bound provides an honest finite energy control.
Define

\[
 L_i(C)=2\sum_d |d_i|\sum_\mu |(s_d)_\mu|\|\Phi_\mu\|_{op}.
\]

Then `||partial_theta_i h||_op <= L_i`. Every momentum in a midpoint cell
satisfies

\[
 \|h(\theta)-h(\theta_k)\|_{op}
 \le\sum_iL_i|\theta_i-\theta_{k,i}|\le
 \epsilon_n:=\frac\pi n\sum_iL_i.
\]

By Weyl, a sampled lower positive eigenvalue exceeding `epsilon_n` and a
sampled upper negative eigenvalue below `-epsilon_n` are sufficient for two
positive/two negative bands **throughout** the Brillouin zone. When that
cover succeeds, the bulk whole-heavy sea is the integral of the top two
bands. Ky Fan's two-eigenvalue Lipschitz bound and the cell average
`mean |theta_i-theta_ki|=pi/(2n)` give

\[
 |F_n(C)-F_\infty(C)|\le\epsilon_n.
\]

This is a conservative first-order bound, not a fitted convergence rate. If
the sign cover fails, the generally correct continuum fixed-half-filled
quantity uses the chemical-potential representation

\[
 S_\infty(C)=\min_\mu\left[2\mu+
 \int_{BZ}\sum_{j=1}^4(\lambda_j(\theta)-\mu)_+
 \frac{d^3\theta}{(2\pi)^3}\right],\qquad F_\infty=V-S_\infty.
\]

The same formula holds on the finite mesh. Positive parts are one-Lipschitz;
the four Weyl estimates give a bound uniform in `mu`, so minimization
preserves

\[
 |F_n(C)-F_\infty(C)|\le2\epsilon_n
\]

without per-block filling or a continuum sign gap. The receipt evaluates both
bounds and marks when the sharper two-per-block bound applies. The
inequalities are exact, but operator norms and coefficients are floating
numerical evaluations, **not interval-certified upper endpoints**.

These energy bounds are uniform on bounded coordinate sets. The quartic is
coercive: `||CC^T||_F^2 >= ||C||_F^4/4`, whereas the bounded-hopping sea grows
at most linearly in `||C||_F`. Thus the uniform-field continuum functional is
well-defined and bounded below, and finite-mesh minimizers cannot escape to
infinite norm. This compactness fact does not identify the numerical
multistart candidates with those exact minimizers. It does not imply that a
coarse stationary branch is stationary, stable or uniquely selected in the
limit. Gradient/Hessian convergence additionally needs a uniform spectral
gap and derivative/quadrature estimates; the finite controls do not provide
those certified estimates.

Open-box versus torus energies differ by boundary hopping. For a fixed
bounded uniform field and finite-range paths, the number of changed matrix
blocks is `O(n^2)` whereas volume is `n^3`; Ky Fan's trace-norm perturbation
bound therefore makes the **per-cell** difference `O(1/n)`. This observation
cannot transfer the result of one two-by-two-by-two open cube to bulk, and it
does not transfer boundary-sensitive finite Hessians or stabilizers. Nor does
a uniform sixty-coordinate Hessian test the `60*n^3` nonuniform spatial
coordinates, link fields, flux sectors, quantum fluctuations or a phase.

## Parent-owned executions and outstanding joined obligations

The supplier author does not run scientific CLI executions mid-flight. From
`/home/administrator/substrate-222`, the first actual competition command is:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/periodic_bulk_frame.py --grid 8 --validation-grid 16 --maxiter 2000 --stationarity-tolerance 2e-8 --cube-receipt proposals/P256-qed-backward/unrestricted_field_minimum_receipt.json --output proposals/P256-qed-backward/periodic_bulk_frame_receipt.json
```

A larger independent refinement/validation pair, after inspecting that
receipt rather than treating it as thermodynamic certification, is:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/periodic_bulk_frame.py --grid 16 --validation-grid 24 --maxiter 2000 --stationarity-tolerance 2e-8 --cube-receipt proposals/P256-qed-backward/unrestricted_field_minimum_receipt.json --output proposals/P256-qed-backward/periodic_bulk_frame_16_24_receipt.json
```

Natural process exit and actual receipts belong to the parent verification
boundary. Until those executions, numerical outcomes of this supplier are
unobserved. Outstanding obligations include certified continuum stationary
and normal-curvature bounds, the full nonuniform field/link/flux stability
problem, quotient-compatible quantum kinetic dynamics (not a Cartesian
Hessian pseudoinverse), thermodynamic quantum state/phase selection, and the
same-frame joined charged fermionic/spin, current/photon and massive
propagation consumers. None is settled by finite-grid branch competition.

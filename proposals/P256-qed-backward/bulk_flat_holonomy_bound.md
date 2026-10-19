# Original itinerant-heavy selection without fixing flat torons

This retains the original whole-heavy periodic endpoint matrix and the
original heavy/light CAR sectors. It does **not** use the 27 added local
selector flavours of `bulk_quantum_selection.md`. The producer reuses only
that module's source-owned endpoint matrix, grid and link geometry helpers;
its initialized selector metadata is unused: no added-CAR selector energy or
changed heavy Hamiltonian enters this producer.

There are two results: an explicit finite-size spectral bound for the full
flat/covariantly uniform classical class, and an iterated-limit quantum
field-shape theorem for a declared strong positive alignment/colour-magnetic
trajectory. Neither identifies an isolated electron, mechanical spin,
quantum light pole or compact-photon phase.

## 1. Actual source and all flat holonomies

The default uses `RotationBand(**BARE)` and its unimproved nearest-axis
`scalar_paths`: `a=1,m=.2,r=.12,v=.3`, `kappa=STIFFNESS=2`.
`--spacing a` executes the same bare mass/r/v cutoff trajectory with
`kappa=2/a`, matching `bulk_cutoff_matching.py`'s field law. For general
displayed spacing a, its physical Bloch coefficients are

```
x0(k)=m+(r/a) sum_j(1-cos(a k_j)),
xj(k)=(v/a) sin(a k_j).
```

The native endpoint helper receives the actual spacing-dependent band, not
an a1 matrix relabelled afterwards. The Gram solve uses coefficients `a*x`
and stiffness `a*kappa=2`, then divides the reported energy by a. Matrix,
spectrum and energy residual checks likewise use aH; receipts also preserve
their raw physical residuals. The theorem holds at each fixed positive a.
Neither its constants nor its normal-packet estimates assert uniformity as
a tends to zero; the limits are Lambda, then N, at fixed a.

Every field is a traceless Hermitian four-by-four matrix,
`G_mu,nu=Tr(Phi_mu Phi_nu)/4`. The quartic per site is
`kappa ||G||_F^2/2`. Heavy has the negative Hamiltonian sign and the **whole**
`2N^3` occupation among `4N^3` modes. No momentum or colour block is assigned
occupation two.

On a connected periodic `N^3`, `N>=3`, zero covariant alignment and zero
positive colour plaquette cost imply, after tree gauge fixing, constant Phi,
commuting SU4 holonomies `Omega_j`, and `[Omega_j,Phi_mu]=0`. All torons remain.
Joint eigenspaces b of the holonomies are invariant under every Phi. The
block b Bloch matrix is `sum_mu x_mu(k+theta_b/(Na)) Phi_mu,b`. Holonomies can
be noncentral when the fields are reducible; those competitors are retained.

The full one-body trace, summed over the periodic grid, is zero: on-site
trace vanishes and each distinct nearest-neighbour Fourier exponential sums
to zero even with its block's constant twist. Thus the actual whole-half
heavy energy obeys

\[
 E_{H,N}/V\ge-\tfrac12\langle\operatorname{Tr}|h_\theta(k)|\rangle_N.
\]

This uses the sum of *all* positive eigenvalues as an upper bound on any
chosen `2V` eigenvalues, not a forced-two rule. Equality need not hold for
an arbitrary reducible spectrum.

## 2. Uniform toron error and continuum Clifford selection

Let the integral below be normalized Haar measure on the momentum torus and
set

\[
 F(G)=\frac\kappa2\|G\|_F^2-2\int\sqrt{x^TGx},\qquad
 B(\Phi)=2\int\sqrt{x^TGx}-\tfrac12\int\operatorname{Tr}|h_\Phi|\ge0.
\]

The nonnegative second quantity is the four-by-four Hilbert-Schmidt
Cauchy slack. Define the explicit cell-integration error

\[
 D_N(G)=\frac\pi{aN}
 [3|r|\|\Phi_0\|_F+|v|\sum_{j=1}^3\|\Phi_j\|_F]
 =\frac{2\pi}{aN}[3|r|\sqrt{G_{00}}+|v|\sum_j\sqrt{G_{jj}}].
\]

For an integration cell displacement `|a delta k_j|<=pi/N`, the coefficient
changes are bounded by `3|r|pi/(aN)` for x0 and `|v|pi/(aN)` for each xj.
Frobenius triangle/Minkowski holds across all invariant blocks, while
`||M||_1<=2||M||_F` for the full colour dimension four. Trace norm is
1-Lipschitz in its trace-norm argument. Integrate the **blockwise shifted
cells directly**: each block's constant twist disappears from its periodic
integral. It is unnecessary to compare the twisted grid to the identity grid
first, which would double the error. Therefore

\[
 E_{\mathrm{flat},N}(\Phi,\Omega)/V\ge F(G)+B(\Phi)-D_N(G).\tag{1}
\]

A finite comparison grid M gives the exactly checkable weaker lower bound
`Eflat_N/V >= F_M(G)-D_N(G)-D_M(G)`. M need not divide N. These finite-grid
checks do not certify an integral numerically; their error terms are proved.

The continuum PSD functional F is strongly convex with modulus kappa and
coercive. Its unique exact minimizer Gstar is diagonal
`diag(g0,gS,gS,gS)` by reflection/cubic symmetry. Every vanishing diagonal
coordinate has a strictly negative one-sided sea derivative, so `g0,gS>0`.
The coefficients `x_mu x_nu` span Sym4: the six offdiagonal functions have
distinct reflection characters; after they vanish, setting all k=0 forces
the x0-squared coefficient zero, and varying each single axis forces the
other three diagonal coefficients zero. Here m,v are nonzero. This is an
analytic rank10 argument, not a numerical richness guess.

Clifford fields at Gstar attain F(Gstar), and on the regular identity grid
their energy differs from this integral by at most D_N(Gstar). Identity is a
legal **trial**, not a global-link restriction or asserted minimizer.

For a global flat-class minimizer its energy is at most zero, since zero
fields are a legal trial. Put `s=||G||_F` and

```
Cx=sqrt((|m|+6|r|/a)^2+3(v/a)^2),
smax=(4 sqrt(2) Cx/kappa)^(2/3),
CD=(2pi/a) sqrt(9r^2+3v^2) sqrt(2smax).
```

For any commuting-block twist, `||h_theta||_F<=2Cx sqrt(TrG)` and
`TrG<=2s`. The zero upper energy gives `s<=smax`. The same bound applies to
Gstar. Consequently `D_N<=CD/N` for these minimizers and Gstar. Strong
convexity in (1) yields the exact result

\[
 \frac\kappa2\|G-G_\star\|_F^2+B(\Phi)\le\frac{2C_D}{N}.\tag{2}
\]

Thus any minimizing sequence as N grows approaches the Clifford orbit.
The fields are in a compact norm sublevel after gauge fixing; zero integral
Cauchy slack forces `h_Phi(k)^2=(x^TGstar x)I` everywhere by continuity.
Analytic rank10 then gives `{Phi_mu,Phi_nu}=2Gstar_mu,nu I`. The full-rank
complex Cl4 representation is irreducible, so its limiting colour commutant
is Z4. This does not set finite-N torons to identity or prove every finite-N
minimizer irreducible.

## 3. A full-domain quantum extension retaining original heavy scale

At each fixed finite N use all `R^(60V)` fields, all `SU4^(3V)` colour links,
all `U1^(3V)` phases and both original whole `NH=NL=2V` CAR sectors. Heavy is
EM-neutral; light has charge+1 and background2 at each node. Declare

\[
 H_\Lambda=\frac{\sqrt\Lambda}{2}p_\Phi^2+u_4\sum C_{SU4}
 +\frac{u_1}{2}\sum p_A^2+K_1\sum(1-\cos\operatorname{curl}A)
 +\Lambda[\tfrac\kappa2\sum\|G_x\|_F^2-d\Gamma(h_H)]
 +\Lambda^{5/4}[J W/8+K_4P]+d\Gamma(h_L).\tag{3}
\]

All displayed coefficients are positive (`u4=.3,u1=.4,K1=.7,J=.2,K4=.7`
for the source comparison). `P=sum(4-ReTr Uplaquette)` and W is the complete
covariant field-alignment sum. This is an **explicit different positive bare
alignment/colour-plaquette trajectory**. It preserves the original itinerant
heavy coefficient Lambda and source couplings; it adds no matter species.
There is no hard flatness, mean, normal or representation restriction.

The endpoint-average HS bound `||hH||_F,||hL||_F<=2b sqrt(Z)`,
`Z=sum TrGx`, `b=|onsite|+2sum_paths||scalar_path||`, and whole-half Ky Fan
bounds give field-quartic coercivity uniformly in links/phases. The unscaled
light lower bound is `-2b sqrt(2V Z)` and becomes negligible after dividing
by Lambda. Kinetic/photon terms are positive. The finite-dimensional CAR
fibres and compact link bases with coercive fields have a physical ground;
no numerical ground is computed here.

### Complete Gauss trial at every flat-class minimum

A physical upper at an arbitrary constrained minimum is needed, not just at
the identity Clifford trial. Let K be its full colour stabilizer. On the
original heavy fixed-number Fock space take the spectral projector PH onto
its **entire** ground eigenspace. K commutes with the endpoint Hamiltonian,
so it commutes with PH. The original light has the same `2V` occupation in
`4V` modes: the exterior Levi-Civita pairing epsilon maps it to the dual of
the heavy space. The normalized vectorized `PH epsilon` is K-invariant,
has its heavy marginal in the true heavy ground, and uses only the two
original Fock sectors. This is a variational purification, **not** added
antifundamental matter or a changed Hamiltonian. It also works for degenerate
heavy occupations; no invariant heavy-only line is assumed.

The compact gauge orbit is G/K. Its tubular normal bundle admits the
section `D(g) psi0` times a smooth K-invariant radial bump: K-invariance of
psi0 makes it single-valued. This is a complete-colour-Gauss wavefunction,
including field and link derivatives. Stabilizer strata do not invalidate a
tubular neighbourhood of one fixed compact orbit. Derivative bounds are
finite constants at each fixed N and orbit, not uniform-in-volume claims.

For EM Gauss, every light occupation configuration has integer
`q_x=n_x-2` with sum zero. A fixed spanning tree supplies integer electric
flow `p_e(q)` with `div p=q`. Multiplying the light basis coefficient by
`exp(i sum_e A_e p_e(q))` satisfies every EM Gauss equation. This unitary
commutes with local colour rotations because n_x does. Its electric cost is
bounded at fixed V (`|p_e|<=2(V-1)`); photon potential and unscaled light costs
are bounded. The full Fock domain remains unchanged. At the explicit
Clifford trial the simpler site-wise two-light state makes this dressing
unity; the producer checks the paired global Z4 character and actual full
field-orbit rank there.

Use normal width `delta=Lambda^(-3/16)`. The field kinetic cost is at most
`C_N sqrtLambda/delta^2=C_N Lambda^(7/8)`, colour kinetic at most
`C_N/delta^2=C_N Lambda^(3/8)`, stiff W/P potential at most
`C_N Lambda^(5/4) delta^2=C_N Lambda^(7/8)`, and frozen-fibre leading
potential error at most `C_N Lambda delta=C_N Lambda^(13/16)`.
The smooth reference fibre is frozen in the normal chart, not assumed the
instantaneous off-orbit heavy ground; that is enough for this Lipschitz
upper. EM/light terms are finite-N lower order. Therefore its scaled energy
approaches the exact flat-class minimum.

### Quantum concentration and order of limits

The scalar infima of `V0+Lambda^(1/4)(J W/8+K4P)` increase to the constrained
flat/covariant-uniform minimum by coercivity and compactness. The preceding
complete-Gauss upper and pointwise heavy-CAR lower squeeze `E0/Lambda` to
that value. Coordinate probabilities of any exact physical neutral ground
concentrate on its full flat-class minimizing set as Lambda grows.
The quartic tail is uniformly integrable in this limit: outside a fixed
large field sublevel it is bounded by a constant times the positive scalar
potential excess above its infimum, whose ground expectation tends to zero.
The unscaled light contribution vanishes by the same coercive moment bound.

Let `Gbar=V^-1 sum_x Gx` and let `Bbar` be the average of the local continuum
slack B(Phi_x) defined above. On that limiting set all Gx agree, and (2) gives

\[
 \limsup_{\Lambda\to\infty}\left\langle
 \tfrac\kappa2\|\bar G-G_\star\|_F^2+\bar B\right\rangle\le2C_D/N.
\]

Taking N to infinity **after** this limit selects full-rank Clifford quantum
field shape modulo local colour gauge and retained centre holonomies. This
is a shape theorem for (3), not an effective light Hamiltonian or a phase
proof. Constants grow with N; no simultaneous or fixed-coupling limit is
asserted. The exact integral Gstar is a mathematical minimizer; the receipt's
finite source-grid Ghat is not an interval certificate for it.

## 4. Actual discriminating consumer

The producer compares full native108/256 endpoint matrices against independent
block Bloch spectra at N3/N4: identity and central-pi reoptimized Clifford,
noncentral two-by-two reducible fields, and Cartan fields with transferred
whole-half occupations. All W/P directions and arbitrary local-colour
covariance are checked. Larger grids test the explicit spectral-error
comparison with a finite reference grid. A wrong forced-two-per-momentum
rule is rejected by its actual energy error; the occupation histogram uses
one stable whole-half tie-breaking, not a claim of unique occupations.

The actual full-rank field-orbit Jacobian has rank15 per node; adding link
incidence cannot remove that rank. The whole-heavy positive Slater and a
site-half light trial have a trivial **combined** global Z4 character. On
odd V the heavy-only i-centre character is minus, an adverse control against
assuming a trivial heavy scalar fibre. These are geometric/trial checks,
not a numerical diagonalization of the full quantum Hamiltonian.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/bulk_flat_holonomy_bound.py
```

The quantum postulates/CAR matter, scalar couplings, explicit strong bare
trajectory, finite-grid numerical estimates, charged/physical-spin and
photon-phase obligations remain. The joined interacting electron-photon
campaign is not complete.

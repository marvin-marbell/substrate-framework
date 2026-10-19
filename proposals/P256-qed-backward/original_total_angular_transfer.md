# Original total angular stress: whole-filter transfer and the rotation boundary

Governing plan `issue-222-continuing-backward-20260930`, assignment revision
`b9c426960085aecd28d51c57e27e9cc42092d05a6bf09762578f27840a6eb11d`, pinned general
`89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`, work
`campaign`, requirements `outcome/verification`. Only this proof and its native
producer are owned here. No selector species or changed-27 energy is used.

## 1. Result, assumptions and reference ownership

Use the original full R^(60V), SU4^(3V), compact U1^(3V), whole NH=2V and
whole fixed-NL CAR domains. Keep all colour and electromagnetic Gauss equations,
the background-two charge, and, for NL=2V±1, the specified opposite colour/U1
reference. The microscopic law and slow trajectory are exactly those of
`original_slow_kinetic_bo.md`: chi=Lambda^(-1/2), u4=.3chi, J=.2Lambda^(5/4),
K4=.7Lambda^(5/4), positive fixed photon coefficients. Fix a,N with
N>=max(5,N0(a)); N0 is existence-only. This is not an assertion that numerical
N5 is in that range.

The theorem below concerns an admissible gauge-covariant *mechanical stress
flow*, not an exact continuous symmetry of a finite cube. At fixed a the native
Hamiltonian has the physical proper-cube symmetry proved by
`original_physical_rotation_lift.md`, not SO3. Every defect from continuous
rotation, the chart boundary and the reference is retained.

Choose a real bounded site velocity v and real intrinsic weight f. For an
angular component in a reference chart use v(x)=f(x) omega×X(x), where X is the
actual coordinate relative to a chosen origin. A periodic smooth f supported
away from the chart seam avoids coordinate discontinuities. A global sawtooth
X is also an admissible finite-graph observable, but it has a seam/boundary
Ward defect; it is not a global continuum rotation. A generic fixed reference
R is external geometry. A rotation moving R compares different reference
Hilbert spaces; no generator moving it is manufactured in a fixed-R space.
For an origin reference, v(R)=0. Its colour representation is still present.
For a localized region, the screened reference coefficient and flux through
the region boundary remain, even when the bare weight vanishes at R.

Let J(v,f) be the operator constructed in section 2. For eta real in
C_c^infinity(R), with K,W,H_eff the exact original BO construction,

```
|| eta(K) J(v,f) eta(K)
 - W eta(H_eff) J_eff(v,f) eta(H_eff) W† ||
 <= C_eta,v,f Lambda^(-1/16) + poly(Lambda) exp(-c Lambda^(7/20)).
```

Here every one of the 64 centre wells and its scalar offset remains, and

```
J_eff = directsum_c [ S_Phi^c(f) + dGamma_NL(L_c(U_c,A;v)) + J_EM(A;v) ].
```

S_Phi^c is the actual full15-generator, mass-horizontal mechanical connection
of `original_intrinsic_angular_transfer.md`, including Q_ref. It is not
replaced by an assigned standard spin matrix at a tilted native G_c.
The heavy scalar contribution is zero at the flat star; it is not removed
from the microscopic Hilbert space. The field-orbital and colour-stress
compressed scalar means vanish in the limit, not their operators or vacuum
fluctuations. J_EM is unbounded on the entire photon space. Section 3 supplies
the additional photon-form estimate needed for this assertion.

Constants depend on fixed a,N, v,f, positive couplings and the entire
number/reference sector. No uniform-volume/cutoff, localized-half-spin,
charged-pole or scattering theorem is contained in this statement.

## 2. Primitive covariant stress flow

Use source generators T_a=X_a/sqrt(8), Tr(T_a T_b)=delta_ab/2. An oriented
reverse edge is exactly U_(x-e_i,i)† and minus its EM angle. For any path gamma
based at x multiply these oriented edges in path order. This preserves all15
colour directions, all Haar left/right derivatives and reversed seams.

For j!=i form the four clover loops with steps
(s e_j,t e_i,-s e_j,-t e_i), s,t=±1, all based at x. Define

```
B4_ji(x) = (1/(4a²)) sum_s,t st Im_H(P_s,t(x))_traceless,
B1_ji(x) = (1/(4a²)) sum_s,t st sin(F_s,t(x)),
Im_H(P)=(P-P†)/(2i).
```

B_ij=-B_ji. For an i-edge average B4 at the two endpoints with actual parallel
transport U_i B4(y) U_i†; average B1 without transport. Use the corresponding
endpoint average of v_j. Denote these edge averages by bar B and bar v.

These are bounded smooth compact-link coefficients. They are not principal
logs with a branch cut and not continuum curls declared exact at finite a.
The normalization comes from the actual actions: the EM action derivative of
K(1-cos F) is K sin F, and the colour action
K4 ||P-I||_F²/2=K4(4-Re Tr P) has Lie derivative proportional to the traceless
Hermitian imaginary part. B is the corresponding force per stiffness and per
plaquette area. No stiffness factor is inserted into canonical angular
momentum. Changing a plaquette discretization changes its finite-a Ward
remainder, not the leading canonical normalization.

The actual covariant central field difference is

```
D_j Phi(x) = [U_(x,j) Phi(x+e_j) U_(x,j)†
 - U_(x-e_j,j)† Phi(x-e_j) U_(x-e_j,j)]/(2a).
```

For the bosonic coordinate-flow vector use

```
delta Phi_mu = -sum_j v_j D_j Phi_mu + f (omega-vector action)_mu,mu' Phi_mu',
delta U_(x,i) = -i a sum_j bar v_j bar B4_ji U_(x,i),
delta A_(x,i) = -a sum_j bar v_j bar B1_ji.
```

The intrinsic action has deltaPhi_j=-fPhi_k, deltaPhi_k=fPhi_j for omega=e_i,
with the same sign convention as the existing Cartesian intrinsic operator.
The signs of the orbital and link terms are those of the *active pullback*
deltaPhi=-v·D Phi. In the continuum the covariant connection flow is
-delta_x contraction F. This is the ordinary spatial pullback plus a gauge
restoration, not an additional internal spin law.

If Y denotes these real bosonic vector fields, its selfadjoint generator is
-i(Y·partial+div_measure(Y)/2). Use flat field measure and exact product Haar
measure; the half divergence is essential. Compact-link coefficients are
bounded, the field coefficients grow at most linearly in Phi, and the vector
flow is complete. Its pullback with the square-root Jacobian is a unitary
one-parameter group on the full uncut coordinate domain. This defines its
selfadjoint generator independently of a polynomial-domain convenience.

For scalar CAR define the genuine parallel shift
(T_j psi)(x)=U_(x,j) exp(i A_(x,j)) psi(x+e_j), omitting EM only for heavy.
Let P_j=(T_j-T_j†)/(2ia) and

```
L(U,A;v)=sum_j {v_j,P_j}/2.
J_CAR=dGamma_NH(L(U,0;v))+dGamma_NL(L(U,A;v)).
```

L is Hermitian and bounded, ||L||<=sum_j ||v_j||_infinity/a, on the complete
one-body space. Its exterior lift obeys ||dGamma_n L||<=n||L|| on every
occupation in the entire fixed-number sector. No momentum/Fock projection
or per-cell filling is introduced. Its commutator with annihilation fields
has the symmetrized covariant transport -v·D and its divergence term; raw
matter is still a microscopic spatial scalar.

Set J=J_bosonic+J_CAR. The colour coefficients transform at their actual
anchors; Phi differences transform adjointly; EM coefficients are gauge
invariant. L transforms by the full site gauge unitary. Consequently J
commutes with the combined colour/U1 Gauss generators, as a quadratic-form
identity on smooth equivariant vectors and then on the natural closed domain.
The derivative of L with respect to link variables is crucial: a bare X×central
ordinary derivative does not commute with local Gauss. The reference is not
silently omitted from the equivariance condition.

At finite a, Y is a legitimate covariant deformation of primitive coordinates.
It is not a derivation of the algebra of site functions: finite commutative
site algebras have no nonzero derivation. Accordingly this construction does
not exponentiate to exact arbitrary rotations of a finite lattice.

## 3. Uniform photon form by exact fibrewise block comparison

A bare BO norm estimate cannot transfer p_A. Nor may one subtract just the
classical principal minimum: the missing complete auxiliary vacuum costs
O(Lambda^(3/8)). The following comparison retains that vacuum without positing
an invalid heavy-only global Gauss space.

The BO isometry is independent of A and EM-neutral in its normal/heavy factors.
It commutes with every light site number and leaves the actual reference
charge in place. Thus W, P=WW† and C=1-P commute with the entire H_ph, including
unbounded electric operators; they also preserve its form domain. Resolve the
physical space over compact A. Its fibre is the complete combined
heavy/light/reference colour-associated bundle with the U1 equivariance law,
not a heavy-only physical space. Nontrivial centre characters of heavy alone
are allowed; the combined stabilizer character is exactly the one in the BO
proof.

Repeat sections 5-6 of `original_slow_kinetic_bo.md` for
T(A)=K-H_ph, pointwise A. Every operation in those estimates is a normal/colour
operation. The light/ref fibre is finite dimensional and its bounds hold
uniformly all compact A. None of the lower bounds needed a positive photon
term. In particular:

* In cores the exact scalar E_c is retained, the full fibre connections are
  Young-controlled, and the normal orthogonality is componentwise in the
  ENTIRE light/reference fibre. The scalar S_c is a body-chart comparison,
  not a global heavy-only physical state.
* The complement normal gap is cLambda^(1/4)-C Lambda^(7/40)-C. Excited heavy
  fibres have cLambda-C Lambda^(3/8)-C. These are the same pointwise normal
  estimates; all centre offsets E_c-E_min>=0 remain.
* Outside the tubes the original quartic bound
  ||H_light||<=C0+C1(U_Lambda-v*) gives
  (Lambda-C1)(U_Lambda-v*)-(E_min-Lambda v*)-C0. It exceeds the same gap.
  This retains reducible/singular and arbitrarily large fields, not a compact
  exterior asserted to have a globally bounded light Hamiltonian.
* IMS costs chi/epsilon² and the exact normal cutoff errors are unchanged.

Thus C T(A) C>=g_Lambda C uniformly A. The P block is the bounded native
light matrix plus nonnegative scalar offsets and bounded residual; its lower
bound is -C_N-d_Lambda. The off-diagonal C T P=C R W† has norm<=d_Lambda,
since H_ph has no off-diagonal block. The elementary two-block estimate gives

```
T(A)>=-C_N-3d_Lambda,
H_ph <= K + C_N + 3d_Lambda
```

as forms on the actual full physical domain, for sufficiently large Lambda.
This is a fibrewise form proof and does not require an auxiliary global
heavy-only state. It is stronger and simpler than the proposed discount
argument with delta=Lambda^(-7/16). That proposed exponent balances
 delta E_ZP and chi/delta correctly, but alone does not establish its domain
comparison. No claim is made from that exponent arithmetic.

Since photon electric energy is positive ue sum p_A², for fixed compact
spectral eta the preceding inequality gives
||p_A eta(K)||<=C_eta,N. Similarly H_eff>=H_ph-C_N gives
||p_A W eta(H_eff)W†||<=C_eta,N. Bounded smooth sin-curl coefficients and their
first derivatives therefore make J_EM times both low filters uniformly
bounded. For selfadjoint A=eta(K), B=W eta(H_eff)W†,

```
A J_EM A-B J_EM B=(A-B)J_EM A+B J_EM(A-B).
```

Put the second factor on its adjoint, so that ||B J_EM||=||J_EM B||. Smooth
norm-resolvent transfer supplies ||A-B||=O(d_Lambda+1/g_Lambda), and hence the
photon-stress difference tends to zero with no false implication from bounded
norm distance to unbounded momentum distance. W intertwines J_EM exactly.

## 4. Complete normal/orbital compression

At every flat, covariantly constant regular star D Phi=0 and the colour
clover B4=0, including reversed centre seams. EM fields remain arbitrary.
More strongly, in the
actual flat-adapted coordinates (z,xi) of the BO proof, D Phi and B4 vanish on
the entire xi=0 flat family, for every nearby uniform tuple z. Their first
variation is therefore stiff, not an arbitrary soft coordinate. This statement
is invariant under the full mass-horizontal projection; it does not rely on
a Euclidean field-only complement. At q=0 its kernel is the complete45
physical uniform directions; the other physical directions number90V-45.
The native producer checks this against the actual105-by15 Gauss matrix and
normal Hessian, and retains all fifteen generators.

On regular cores the induced quotient orbital vector and mechanical colour
connection are smooth bounded. Their base values vanish, while the intrinsic
base connection is exactly the prior S_Phi^c(f). Heavy angular means in other
colour channels, variances, Berry and Born-Huang terms remain in the scalar
auxiliary Hamiltonian. Only their orbital coefficient at the star is zero.

A tempting but unnecessary estimate is ||y_soft p_stiff u||. A product harmonic
state can indeed make this grow as Lambda^(1/16); a bare normal norm estimate
would not control it. We do NOT multiply independent moments, discard a mixed
term, or require ||J_bosonic W|| uniformly bounded. Instead use the already
proved momentum-weighted complementary estimates of
`original_intrinsic_angular_transfer.md`, sections 4-7. For its smooth low
filters these give

```
||p C_N eta(K)||=O(Lambda^(-3/40)),
||p C_H eta(K)||=O(Lambda^(-5/16)),
```

with polynomial-exponential exterior losses, on the same regular charts and
full-domain localization. The estimates hold for all canonical normal
momenta, so any smooth bounded first-order quotient coefficient is admissible,
including mixed soft/stiff coefficients. There is no requirement that that
coefficient preserve the separate harmonic mode spaces.

For the P block compare the true magnetic scalar vacuum with the real scalar
counterpart keeping its exact potential, metric, density and real angular/
Born-Huang terms. A real selfadjoint first-order flow has exactly zero mean in
its real positive ground: integration by parts cancels its half divergence.
The actual Kato Berry and normal-colour connection vanish at the star and are
O(y); the same source residual O(Lambda^(-21/80)) and normal momentum resolvent
estimate give the prior O(Lambda^(-3/40)) difference of means. This argument
applies to the *whole* smooth real quotient angular flow, not merely the
intrinsic one. Heavy/projector and light/reference connection contributions
with vanishing orbital base coefficient are O(||y u||)=O(Lambda^(-5/16)).
Thus field-orbital and colour-stress compressed means tend to zero, while the
intrinsic term gives its genuine Gauss cloud. No assertion that fluctuations
vanish or that the microscopic vacuum is real is made.

In P-C matrix elements use selfadjointness to put the first-order operator on
C eta(K), rather than on W. The displayed weighted complement estimates
then control the whole orbital/intrinsic/colour vector operator. The C-C
term has an additional small norm factor. The P-P term uses its bounded
compressed matrix, not ||pW|| times a norm error. This is why the dangerous
harmonic cross norm does not invalidate the spectral-compression theorem.

The scalar CAR orbital is bounded on the entire fixed-number fibres and
Lipschitz in the colour links uniformly all A, so the original norm BO transfer
applies. At a flat centre star the heavy occupied projector has Bloch blocks
P_+(k)=(I+h(k)/rho(k))/2 with trace exactly2. Its scalar off-site density trace
is therefore zero: the periodic sum of exp(i k·d) vanishes for a nearest
nonzero displacement, including the actual centre phase/seam restoration.
L_H has only such off-site entries. Hence Tr(P_H L_H)=0 for any site weights v.
This is a whole2V Slater trace, not two particles fixed at each site, and not
an even-filling shortcut. Nonuniform deviations give the controlled bounded
Lipschitz remainder, not an asserted exact zero away from the star.

Combining this section, prior intrinsic transfer and section 3 proves the
section 1 whole-filter theorem. No normal, electric, angular or CAR hard cut
has entered the microscopic law.

## 5. What becomes a continuous rotation, and what does not

The continuum identity is conditional on an actual long-wave, low-field state
family, on chart/boundary control and on reference geometry. It is not a
selection of a photon phase. At the source-owned isometric limit
G_infinity=diag(g0,gs I), intrinsic compression for f=1 gives the derived
S_light+S_ref. The scalar CAR flow is covariant orbital transport. The compact
stress becomes the canonical connection stress because sin F=F+O(F³), the
anchored colour imaginary part is F4+O(F4³), and clover/central differences
approach their continuum derivatives on states with controlled derivative
moments. For constant f, Gauss converts the covariant connection stress to
the ordinary spatial/vector pullback plus the surface flux and reference
terms. This is the continuum covariant-Lie-derivative identity derived in
section 2; there is no separately assigned L+S premise. For varying f this
is a localized angular density: derivatives of f give the retained stress/
boundary-flux Ward terms. Its intrinsic index action is not claimed to be
the complete vector Lie derivative for an arbitrary nonrigid velocity.

For the actual isometric native band, writing mu=m sqrt(g0), ce=v sqrt(gs),
r_eff=r sqrt(g0), its exact symbol is

```
h_a(p)=Gamma0[mu+(r_eff/a)sum_j(1-cos(ap_j))]
       +(ce/a)sum_j Gamma_j sin(ap_j).
```

S_i=-i Gamma_j Gamma_k/2 is already fixed by the mechanical field tangent.
For momentum-space test packets the continuous rotation commutator defect is

```
R_i(p)=-i(p_j partial_k-p_k partial_j)h_a(p)+[S_i,h_a(p)].
```

Its linear Dirac and radial Wilson-quadratic terms cancel algebraically.
Using |sin t-t|<=|t|³/6 and |cos t-1|<=t²/2, including the derivatives, gives

```
||R_i psi|| <= C ce a² || |p|³ psi||
             + C r_eff a³ || |p|⁴ psi||.
```

No momentum hard cut is imposed: Schwartz packets and finite moment conditions
suffice. Native finite-N tilted B=sqrt(G_Nc) adds the explicit symbol-and-
derivative error O(||B-B_iso||) with the corresponding zeroth through second
moment weights; it is not exactly SU2 at finite N. On a finite periodic box
the interpolation/site-coordinate boundary contributes its separately
retained seam/tail term. The exact discrete cube symmetry has no such
continuous-generator claim.

For interactions the exact infinitesimal matter Ward expression is

```
dGamma( delta_Y h + i[L,h] ).
```

Here delta_Y h uses actual endpoint fields, colour links and the compact
Peierls current; its second variation uses the actual contact. The native
producer checks that expression and its gauge covariance, rather than
pretending it vanishes for arbitrary rough native fields. The continuum
covariant pullback cancels its leading part only when the same state family
controls the finite-difference remainders and the primitive photon/colour
stress Ward remainders. Photon cosine adds its actual O(F³) force defect;
clover and derivatives add their actual spatial moment remainders. The
quadratic-form expectation of these explicit remainder operators is the
conditional rotation Ward boundary. A Gaussian photon or free band does not
establish those hypotheses for the interacting compact microscopic state.

## 6. Native producer and precise scientific debt

The producer consumes `BulkQuantumLaw.onebody/target/toron_links` only as
original endpoint/geometry interfaces. Its changed-27 metadata and selector
energies are never used. `original_frame` owns all64 full-ten Gram minima;
`OriginalSpatial` and `plaquette_hessian` own all15 mass-horizontal Gauss/
normal forms; `peierls_derivatives` owns the exact same-link current/contact.
`exterior_hamiltonian` is used only with Hermitian matrices and its real
(matrix,basis) return contract. The complete finite two-site CAR diagnostic
exercises every number sector, not a substitute for the full graph theorem.

It checks noncentral gauge transformations of nonuniform full Cartesian
fields, arbitrary compact colour/EM links, clover/reverse seams, primitive
flow and exact current-derived endpoint variation. It rejects a bare
ordinary-derivative orbital under local Gauss. All64 flat stars test null
field/colour orbital coefficients and the actual heavy off-site trace
identity. A complete normal kernel probe tests the stiff/soft distinction.
An adverse harmonic cross formula exposes the Lambda^(1/16) norm hazard;
a free-band commutator probe reports the conditional cubic/quartic remainder.
None computes an infinite-dimensional quantum vacuum or physical photon phase.

The result now transfers a *total admissible mechanical stress observable*
through whole low filters at the original BO scope. It still does not prove:
localized interacting half-spin; removal/separation of the reference;
thermodynamic compact U1 or residual colour phase; finite isolated charged
energy/mobile mass; charged pole versus infraparticle; a joined Lorentz/
spacetime limit; real photon asymptotic states or radiative scattering; uniform
cutoff/nonzero-coupling UV closure. Fundamental CAR, canonical compact quantum
law and the explicitly tuned slow trajectory remain parameter/ontology debt.
No free Casimir or point-group marker is promoted into any of these outcomes.

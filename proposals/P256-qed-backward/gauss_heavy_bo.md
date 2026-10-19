# Physical heavy-only Haar BO and Schur comparison

## Scope and source binding

This is a **fixed selected-frame theorem**, with an executable primitive-operator
consumer in `gauss_heavy_bo.py`. It does not assert that this frame minimizes the
unrestricted Cartesian fields, or solve the light/photon spectrum. The governing
plan is `issue-222-continuing-backward-20260930`, revision
`d7cfa8de60401582b6b901673280c7baeb7ca13556cfcf4f4ee7b80016cb94f1`, pinned general
template `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`.
Parent owns joined integration and scientific execution. No numerical output of
this new consumer was available when this derivation was written.

Authoritative imports, not replacement Hamiltonians:

* `HeavySeaCentrePotential`, `FullBandGaussSection.onebody` and
  `onebody_derivative` supply the entire heavy onsite and both Wilson/spatial
  hopping, the exact selected frame, and the certified actual potential.
  The heavy supplier is pinned to SHA256
  `de1e32fb43f9be142d227d9073b44dc02cc15bb20bff8f0606c58d699cbccf3a`.
* `PrimitiveMatter.transfer`, `PauliCode.element`, and `boundary_word` build
  actual code-projected Pauli matrix units and the heavy/light embedding.
  A 70-state heavy matrix is a **restriction of these primitive matrices**.
  No canonical CAR matrix builds a new microscopic operator.
* `PhysicalHamiltonian` owns the endpoint generators, right boundary, centre
  restrictions and isotropic orientation law. `CompactGaussPhoton.orientation`
  is independently compared with squared endpoint derivatives in the CLI.
* The exponential-chart Haar measure and inverse metric are the geometry of
  `haar_rotor_quartet.md`, section “Haar measure, support, and exact
  normalization”. Neither its W-only oscillator estimate nor its scalar
  quartet/multiplicity theorem is used here.

Bare coefficients remain `m=.2,r=.12,v=.3,stiffness=2,Jpref=.2,a=1`, unimproved.
The consumer defaults match the photon EFT: `I=2,UE4=.3,UE1=.4,K=.7`.
`I` is a **declared isotropic EFT inertia**; it is not the Cartesian induced
field inertia, and it is not scaled with Lambda. The source receipts include
all inherited owner hashes plus these two new files. A changed pinned heavy
proof supplier is rejected, rather than silently inheriting its certificate.

## Actual operator and physical domain

Put `NH=4`, `NL=4+q`, `q=0,-1,+1`. On continuous SU4 Haar measure and the
**infinite** U1 rotor (normalized measure `dphi/(2pi)`), the operator is

\[
 H_\Lambda=K_c+\Lambda\{J_{pref}W(U)/8-d\Gamma_H(h_H(U))\}
 +d\Gamma_L(h_L(U,\phi))
 +\frac{U_{E1}}2\{(\ell+n)^2+\ell^2+q^2\}-K\cos\phi,
\]

\[
 K_c=U_{E4}\sum_a E_{L,a}^2+
 \frac1{2I}\sum_a(D_{L,a}^2+D_{R,a}^2),\quad
 D_L=E_L-Q_L,\quad D_R=E_R-Q_R-Q_B,\quad
 n=N_{L,left}-2,\quad\ell=-i\partial_\phi.
\]

`hH` and `hL` have onsite `(m+r)Phi0` on both cells and endpoint-averaged
`T(U)=(FU+UF)/2`. Heavy hopping is `T(U)` at every photon phase;
light hopping is `T(U)*(1+exp(-i phi))/2`. The **whole** heavy onebody is
multiplied by `-Lambda`. All 15 generators have `Tr(Ta Tb)=delta_ab/2`.
Left differentiation follows `exp(i t Ta)U`, right differentiation follows
`U exp(-i t Ta)`; the coefficient electric operators are `-i` times these
derivatives. Their Lie convention is the supplier's negative convention.

The reduced domain is already the isometric reduction of exact local Gauss
states. Orientation energy is not a Gauss penalty; we do **not** require
`DL psi=DR psi=0`. The residual condition is

\[
 \psi(zU)=z^{N_{H,left}+N_{L,left}}\psi(U),\qquad z\in Z_4.
\]

The right boundary is `B=1,4,bar4` for `q=0,-1,+1`, respectively, with
four-ality `b=-q mod4`. The right restriction is
`Ntotal,right+b+Ntotal,left=0 mod4`. Omitting B in either charged sector fails
this identity. Its four indices are colour boundary indices, not four loop
flux states. The scalar principal coefficient is
`kappa0=UE4+1/I` (default `.8`).

Project **only** the heavy NH=4 ground determinant. The retained finite
internal space is `wedge^NL(C^8) tensor B`, dimensions `70,224,224` in the
three sectors, tensored with the entire photon rotor. There is no light
projection: the charged light cut closes at `phi=pi`, directly checked using
the actual onebody owner. Four centre charts are one residual gauge orbit,
not four independent physical vacua.

## Uniform lower bound and quantitative localization

Let `PH(U)` be this heavy rank-one projection and `QH=1-PH`. Write

\[
 V(U)=J_{pref}W(U)/8-\sum_{j=5}^8\lambda_j(h_H(U)),\quad
 V_0=V(I)=-4\sqrt{a^2+e^2},\quad Q(U)=4-|\mathrm{Tr}U|^2/4.
\]

The supplier's rational amplitude-box certificate gives
`V(U)-V0 >= c Q(U)` and a uniform occupied/empty gap `g>0`. Its previously
reported default lower constants were approximately
`c=.00011849707112072049`, `g=.265575255928`; these are inherited supplier
evidence, **not executions of this consumer**. This file rebuilds exact `c`
from `exact_margin_rational` and exact `g` from the supplier amplitude intervals.
A fixed-NH first determinant excitation costs at least `g`.

With upper amplitude endpoints beta+, s+, define

\[
 a_+=(m+r)\beta_+,\quad f=r\beta_+/2+vs_+/2,\quad
 B_q=(4+q)(a_++f)+K.
\]

`||hL|| <= a+ + f` for all U and phi. Colour and EM electric energies are
positive. Consequently, as a physical-domain quadratic-form inequality,

\[
 \boxed{H_\Lambda\ge\Lambda V_0+\Lambda c Q(U)+\Lambda g Q_H-B_q.}
\]

The trial below proves `E0 <= Lambda V0+A sqrt(Lambda)+B`. Thus, for a ground
state (or normalized minimizing sequence with an added excess epsilon),

\[
 c\langle Q\rangle+g\langle Q_H\rangle
 \le\frac{A\sqrt\Lambda+B+B_q}{\Lambda},
\]

\[
 \langle Q\rangle\le\frac{A\sqrt\Lambda+B+B_q}{\Lambda c},\qquad
 \|Q_H\psi_0\|^2\le\min\left(1,
 \frac{A\sqrt\Lambda+B+B_q}{\Lambda g}\right).
\]

For any `tau>0`, the probability outside `{Q<tau}` is at most
`min(1,(A sqrtLambda+B+Bq)/(Lambda*c*tau))`. These are quantitative
centre-localization and heavy-band-leakage bounds, not a multiplicity claim.
Their constants are conservative; a numerically uninformative bound at a
particular Lambda is not reinterpreted as localization evidence.

A separate explicit scalar comparison follows by completing the squares:

\[
 K_c\ge[U_{E4}+(1-\epsilon)/I]C
 -(\epsilon^{-1}-1)D_{max}/(2I),\quad0<\epsilon<1,
\]

where `Dmax=15*((4+NL)^2+(4+NL+betaB)^2)`, `betaB=0` neutral, `1` charged.
This bounds `QL^2+(QR+QB)^2` without dropping B. The consumer prints the
`epsilon=1/2` coefficient and loss. It does not import a scalar W spectrum.

## Projector derivative constants and a normalized physical trial

All derivatives in this section are unit invariant angular derivatives.
`||Ta||<=1`, `||hH'||,||hH''||<=f` are conservative consequences of the
actual endpoint average. On NH=4 the primitive heavy matrix
`AH=-dGammaH(hH)` therefore obeys `||AH'||<=h1=4f` and `||AH''||<=h2=4f`.
For its isolated rank-one ground projector, reduced-resolvent differentiation
with `||R||<=1/g` gives the usable bounds

\[
 \|P_H'\|\le p_1=2h_1/g,\qquad
 \|P_H''\|\le p_2=2h_2/g+32h_1^2/g^2.
\]

For completeness `P'=-R A'P-P A'R`; `|E'|<=h1` and
`R'=-R(A'-E')R+R^2 A'P+P A'R^2` imply `||R'||<=4h1/g^2`.
Differentiating P' bounds P'' already by `2h2/g+12h1^2/g^2`; the printed
32 is deliberately weaker. These are uniform SU4 bounds, not derivatives
estimated from finite differences. The consumer additionally checks a
chart derivative against the primitive eigenvector at a centre and a
noncentral point.

Choose

\[
 \rho_0=\min(1/2,1/(4p_1)),\quad
 \rho=\rho_0\Lambda^{-1/4},\quad\Lambda\ge1.
\]

In the identity chart `U=exp(i x_a Ta)`, use the exact normalized section

\[
 \omega(U)=P_H(U)\omega_0/\|P_H(U)\omega_0\|,
\]

where omega0 is the actual primitive heavy ground vector at I. Since
`||P(U)-P(I)||<=p1*rho0<=1/4`, its denominator is at least `3/4`, with no
reference-chart exclusion of legitimate states elsewhere. Its unit-direction
derivative norm is at most `s1=4p1`. Each light occupation column is embedded
with actual marker-word creation phases. This is the full heavy-only isometry
J, not a projected charged light determinant.

Take any fixed unit light occupation eta and fixed unit vector beta in the
prescribed B, and the normalized loop state ell=0. On the identity chart set

\[
 f_\rho(x)=(1-|x|^2/\rho^2)_+^2,\quad
 \psi(U,\phi)=f_\rho(x)J(U)(\eta\otimes\beta)/N_\rho.
\]

Extend to the other centre charts by the exact matter character `Mz`; set
zero outside these four charts. The support balls are disjoint and equivariant.
The packet and its first derivative vanish at their boundaries; it lies in
the energy form domain (indeed has the needed weak second derivatives).
With `cH=1/Vol_SU4` and the true chart density j,

\[
 N_\rho^2=4c_H\int f_\rho^2 j\,d^{15}x,
\quad j_*\le j\le1,\quad1\le g^{-1}\le m_*I,
\]

\[
 s_*=1-\rho_0^2/24,\quad j_*=s_*^{12},\quad m_*=s_*^{-2}.
\]

This **exact integral** defines normalization; no Euclidean norm is
substituted for it. The positive enclosure is

\[
 4c_H j_*\frac{|S^{14}|\rho^{15}}2 B(15/2,5)
 \le N_\rho^2\le
 4c_H\frac{|S^{14}|\rho^{15}}2 B(15/2,5).
\]

The Euclidean beta integrals give
`int|grad f|^2/int f^2=15*(15+8)/(3*rho^2)=115/rho^2`.
Accordingly the Haar gradient quotient is at most `115*m*/(j* rho^2)`.
The chart endpoints are exact rational polynomial expressions until directed
float conversion; even when a correction rounds to one, its bound direction
is retained. No W-specific sine bound enters the actual-V energy estimate.

At a centre the actual heavy energy has vanishing first derivative: the
occupied projector is the positive projector of `a Gamma0 - e tau_y Gamma1`
(up to the owner's sign convention), because the commuting Wilson split does
not cross zero. The differentiated trace vanishes by the Clifford
anticommutator and tracelessness. The alignment derivative is also zero.
The ground energy Hessian norm is at most `h2+2*h1^2/g`. Also
`W(exp(iX))<=2*(beta+^2+3s+^2)*|x|^2`. Thus along each radial geodesic

\[
 0\le V(U)-V_0\le L|x|^2,\quad
 L=\frac{J_{pref}}4(\beta_+^2+3s_+^2)+h_2/2+h_1^2/g.
\]

This is a derived bound on the **actual heavy potential**. A sharper actual
anisotropic Hessian/remainder could improve it, but is not presumed.

Using `(a+b)^2<=2a^2+2b^2` separately in each electric/endpoint derivative,
`||QL,a||,||QR,a||<=4+NL`, `||QB,a||<=betaB`, gives the finite constants

\[
 A=\frac{230\kappa_0 m_*}{j_*\rho_0^2}+L\rho_0^2,
\]

\[
 B=15\left[2U_{E4}s_1^2+
 \frac{(s_1+4+NL)^2+(s_1+4+NL+\beta_B)^2}{I}\right]
 +B_q+\frac{U_{E1}}2(4+q^2).
\]

Hence **for every Lambda>=1 and all three prescribed physical charges**,

\[
 \boxed{\langle\psi,H_\Lambda\psi\rangle
 \le\Lambda V_0+A\sqrt\Lambda+B.}
\]

The EM term uses ell=0 and `|n|<=2`. The magnetic trial expectation is actually
zero; bounding it by K in B is harmless. The heavy expectation is the moving
exact ground determinant, not a frozen centre vector. BetaB only bounds the
boundary generator; it is not a boundary/loop identification.

## Implemented compressed derivatives and positive Born energy

For every weighted derivative `Dj=EL,a, DL,a, DR,a`, define

\[
 \nabla_j=J^\dagger D_jJ=-iX_j+\mathcal A_j,\quad
 \mathcal B_j=(Q_HD_jJ)^\dagger(Q_HD_jJ)\ge0.
\]

The exact form identity is

\[
 \|D_jJf\|^2=\|\nabla_jf\|^2+\langle f,\mathcal B_jf\rangle.
\]

`HeavyPhysicalFibre.compressed(U,include_matrices=True)` returns the 45
connection matrices, positive Born terms and weights: 15 electric, 15 left
orientation, 15 right orientation. Charges QL and QR are the actual full
heavy+light primitive matrices. The right connection also includes QB. QB is
in-band, so it changes the connection but not the off-band Born residual;
**it is never omitted from the right kinetic energy**. `retained_slow_multiplication`
returns the entire primitive light matrix, n and n2 with B indices, plus the
exact infinite-loop differential expression. The light matrix depends on
phi and is not replaced by a fixed lower-band energy.

The CLI compares both constant-fibre Gram matrices and a nonconstant
first-derivative jet with the uncompressed primitive action. It separately
checks primitive light Ky Fan extrema/invariance and the actual photon
orientation expansion, including its `+2 QR.QB` term. These controls concern
operators, not source text or merely forwarded inputs. Both centre and
noncentral compressed Born spectra are reported. None of these matrices is
called the full Hamiltonian: they implement **PH H PH**.

## Exact heavy Schur complement and derivative-form correction

All noncolour terms commute with PH: light and EM act on spectators; the
heavy energy and its scalar potential are block diagonal. Moreover,

\[
 Q_HH_\Lambda Q_H\ge\Lambda(V_0+g)-B_q.
\]

For real z below this **actual complementary lower threshold**, put
`delta(z)=Lambda*(V0+g)-Bq-z>0`. The exact heavy Feshbach operator is

\[
 F(z)=P_HH_\Lambda P_H-z
 -P_HH_\Lambda Q_H(Q_HH_\Lambda Q_H-z)^{-1}Q_HH_\Lambda P_H.
\]

The subtraction is positive; it does not vanish merely because a heavy
projector was inserted. To bound it, set `Cj=[Dj,PH]`. The heavy part of each
endpoint charge has norm at most 4; light and B commute with PH. Therefore

\[
 \|C_E\|\le p_1,\quad\|[E,C_E]\|\le p_2,
\]

\[
 \|C_D\|\le t_1=p_1+8,\quad
 \|[D,C_D]\|\le t_2=p_2+16p_1+64.
\]

The second estimate follows from
`[D,[D,P]]=E^2P-2[Q,EP]+[Q,[Q,P]]`; Q is constant in the fixed-frame
coefficient representation. It holds on both endpoints. Let

\[
 S_1=15(U_{E4}p_1^2+t_1^2/I),\quad
 S_0=15(U_{E4}p_2+t_2/I),\quad c_1=8S_1,\quad c_0=2S_0^2.
\]

For psi=PH psi, the commutator identity
`Q Kc P psi=Q sum_j wj(2 Cj Dj+[Dj,Cj])psi` and Cauchy-Schwarz imply

\[
 \|Q_HH_\Lambda P_H\psi\|\le
 2\sqrt{S_1}\sqrt{K_c[\psi]}+S_0\|\psi\|,
\quad
 \boxed{\|Q_HH_\Lambda P_H\psi\|^2
 \le c_1 K_c[\psi]+c_0\|\psi\|^2.}
\]

This is a first-derivative **form bound**, not a bounded-operator norm claim
for an unbounded kinetic Hamiltonian. Since the complementary inverse has
norm at most `1/delta`, it proves

\[
 0\le\text{Schur subtraction}\le
 \delta^{-1}(c_1K_{compressed}+c_0),
\]

where `Kcompressed=Jdag Kc J` includes positive Born energy. Thus

\[
 P_HH_\Lambda P_H-z-\delta^{-1}(c_1K_{compressed}+c_0)
 \le F(z)\le P_HH_\Lambda P_H-z.
\]

The consumer uses the trial-energy upper window, replacing delta by the
conservative rational lower endpoint
`Lambda*g-A*sqrtLambda-B-Bq`. It reports separately whether this is positive
and whether `delta>c1`, so a noncoercive comparison is not silently labelled
controlled. It retains V0 symbolically in all differences, avoiding catastrophic
subtraction of large leading energies. These constants do not shrink the
light or loop fibre. On compact SU4 with confining infinite loop electric
energy the form has the usual discrete spectral realization; the same bounds
also apply to spectral infima/minimizing sequences without assuming existence.

## Parent execution, adverse controls and remaining joined obligations

Run with the project interpreter (no scientific run was performed by this writer):

```sh
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/gauss_heavy_bo.py --scale 1e12 \
  --output /tmp/issue222-gauss-heavy-bo.json
```

The CLI must exit naturally, with finite JSON and all three charge receipts.
It exercises actual marker-word heavy energies, full light spectra,
centre-equivariance, wrong-character rejection, the actual endpoint/B-inclusive
photon orientation and a missing-charged-B adverse control. A finite `R=1`
photon block is used **only for the independent orientation algebra probe**;
no continuum trial, normalization, lower bound, leakage estimate or Schur
threshold is obtained from that cutoff. The actual retained loop is infinite.

Language tooling: a real rooted Pyright 1.1.414 stdio session resolved the heavy
supplier's `FullBandGaussSection` definition and `HeavySeaCentrePotential`
references. Selecting the scientific interpreter removed initial unresolved
NumPy/SciPy diagnostics. A positive structural AST query found the actual
heavy `self.model.onebody` call. Python AST parsing and zero final Pyright
diagnostics for the new consumer are structural validation, not numerical
evidence; the server exited naturally with status zero.

Earned scope is the normalized physical variational construction, explicit
actual-operator form/localization/leakage constants, the endpoint-inclusive
compressed covariant/Born implementation, and a quantitative heavy Schur
comparison. Runtime numerical acceptance is the parent's responsibility.
Still unproved are actual-V angular excitation isolation/remainder certification,
the coupled entire-light/B/infinite-loop neutral-versus-charged slow spectrum,
endpoint-dressed current and carrier properties, and a full-field quantum
phase/microscopic quantum law/parameter closure. Neither four physical vacua
nor a rank-four low-energy photon ladder follows from this construction.

## Actual parent physical comparison

The parent actual CLI at `Lambda=1e12` naturally exited zero into
`gauss_heavy_bo_receipt.json` at source
`3a08859b16a62ec68fb57f01d8a2fcbb77861f6033ea31bd97fc1253e7506210`.
A scoped operator/domain/proof/receipt review accepted the endpoint/B-inclusive
heavy-only construction and Schur form bounds. Primitive kinetic-identity
residuals were below `1.44e-14`, differentiated-jet residuals below `3.64e-12`,
and photon orientation residuals below `9.52e-16`. Wrong scalar characters
give residual approximately `sqrt(2)`; omitting charged B violates four-ality
and changes the actual orientation operator. The charged light cut closes at
phi pi, confirming that the entire light fibre must remain.

At that first scale, the expectation upper bound on Q is about `126.708`,
larger than its universal maximum four: it is **not** nontrivial localization
evidence. The heavy off-band probability bound is below `.056535662`, and
the complementary threshold and positive retained kinetic coefficient are
certified separately.

A second actual invocation at `Lambda=1e20` naturally exited zero into
`gauss_heavy_bo_strong_receipt.json`, with the same bare law and source.
In all three q sectors the derived bounds are `mean(Q)<=.01267075`,
heavy off-band probability `<=5.653564e-6`, complementary margin
`>=2.6557375448e19`, Schur retained-kinetic relative correction
`<=2.442366701e-16`, and additive correction `<=1.201519821e-13`.
These are variational/form evaluations, not computed ground eigenvalues or
charged spectral gaps. Finite R=1 tests prove the independent orientation
algebra only; the theorem retains the infinite photon rotor.

The later unrestricted-field comparison rejects treating the environmental
frame as the one-axis field ground and finds an open-cube uniform candidate
with continuous stabilizer. Neither result invalidates this expressly
conditional fixed-frame theorem, but neither permits inheriting its
four-centre potential in a different field minimum. The next transfer must
respect the actual field geometry and Cartesian kinetic metric; fixed
isotropic I remains a declared EFT law here.


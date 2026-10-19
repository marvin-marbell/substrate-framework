# Native Gaussian carrier mass: an additive Wilson coefficient, not a quantum pole

This is a derivation and native numerical consumer for the **unchanged** original
unimproved light/source law. Governing campaign
`issue-222-continuing-backward-20260930`, pinned revision
`b9c426960085aecd28d51c57e27e9cc42092d05a6bf09762578f27840a6eb11d`,
general template revision
`89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`.
It neither closes the campaign nor introduces the additional 27-channel selector.

The mathematical result is an exact coefficient **within the declared bare
Gaussian weak expansion**. It is not the full interacting mass, a compact photon
phase, a localized separated charged ground, a renormalized photon leg, or an
impossibility theorem for the original interacting law. The executed native
diagnostics below retain their finite-regulator, bare-Gaussian scope.

## Native source and the meaningful limit

`original_frame(bare, N, zeros(3), True)` solves all ten symmetric Gram directions
of the native original heavy source. `reconstruct_frame` obtains the actual Phi
amplitudes and intertwiner; `primitive_frame_controls` independently checks its
raw hopping jets. At the cubic centre-000 frame the reduced unimproved band is

\[
h(p)={c\over a}\sum_j\sin(ap_j)\alpha_j+
\left[\mu+{r\over a}\sum_j(1-\cos ap_j)\right]\beta,
\quad \mu=m\sqrt{G_{00}},\ r=r_{\rm bare}\sqrt{G_{00}},
\quad c=v_{\rm bare}\sqrt{G_{11}}.
\]

The reduced matrices are only a unitary representation of these native Phi
couplings. The producer's finite-mass answer uses **raw native hopping matrices,
projectors and both separately evaluated first jets**, not the closed scalar
answer below. It compares each independent term with the derived formula.

The owner's fixed-momentum finite-difference controls run in the existing
unit-spacing convention (`dimensionless_bare(a)`, reduced mass `a mu`).
Their step is a dimensionless q step; the raw physical jets, Coulomb term
and mass integration below keep the physical spacing. This avoids subtracting
O(1/a) Hamiltonians to identify an O(a) physical contact without weakening
the owner's acceptance tolerance.

The field graph is **N=5, V=125** by default. The normalized thermodynamic BZ
integral is a separate conditional continuation of that frozen frame. Its
split-Gauss order is the number of nodes **per each of six intervals per axis**,
not a field lattice size or a new photon box. For order n it has `(6 n)^3` points.
The native Gram is recomputed at each a with unchanged primitive m, r, v and
stiffness; no improved links or mass counterterm are installed.

Set q=ap, Q=|2 sin(q/2)|, W=sum(1-cos q_j), z=a mu,
M=z+r W, E=sqrt(M^2+c^2 sum sin^2 q_j), Omega=c Q. All displayed integrals use
`d^3q/(2 pi)^3` on `[-pi,pi]^3`. The Gaussian parameters are

\[
u_e={c g^2\over2a},\quad K={c\over g^2a},\quad
U_{E1}=2u_e/e_1,\ e_1=1,\quad \omega_q={cQ\over a},\quad
\langle\theta_q\theta_{-q}\rangle={g^2\over2Q}.
\]

`band.current/a` is the **link-angle** first jet; `band.contact/a^2`
is its second jet. The physical scalar density leg is the unit matrix with
`band.e=1`; g is in the Gaussian leg, not in that source a second time. The
producer verifies both actual matrix polarizations transverse to qhat, and the
unit scalar density leg. The finite-volume q=0 harmonic compact rotor is
excluded: there is no oscillator assignment to that mode. Its exclusion has
zero thermodynamic measure but must not be called a finite-volume rotor proof.

## Whole filled CAR sea and exact order-g² subtraction

For the complete Slater sea O occupying every negative band mode, the linear
vertex produces exactly one photon and one particle-hole transition:

\[
\Delta E_O^{(2)}=-\sum_{i\in O,j\notin O,q,\lambda}
{|\ell_q J_{ji}^{\lambda}|^2\over E_j-E_i+\omega_q}.
\]

Adding one positive rest electron e changes this sum in two ways and only two:

1. The newly occupied e can go to an empty **positive** band mode: a negative
   energy correction with denominator `E(q)-mu+omega`.
2. Negative sea modes could formerly transition into e. Those transitions are
   now **Pauli blocked**: removing their negative vacuum correction gives a
   positive term with denominator `E(q)+mu+omega`.

All unchanged sea transitions cancel algebraically in the carrier-minus-vacuum
gap. Retaining just the new transition is not a one-loop mass calculation.
The midpoint first jet for the new term is J(q,0), whereas the blocked term
requires **J(0,-q)**. The producer builds both from the actual hopping, rather
than identifying them by a guessed charge-conjugation relabeling.

For a real unit polarization epsilon orthogonal to qhat, its Wilson part is
`r sum epsilon_j sin(q_j/2) beta=0`. Write
A_lambda=c sum epsilon_j cos(q_j/2) alpha_j. Native Clifford algebra and the rest
beta=+1 projector give exact two-spin matrices

\[
P_0 A_\lambda P_+(q)A_\lambda P_0
={c^2 t_\lambda\over2}(1-M/E)P_0,
\qquad
P_0 A_\lambda^\dagger P_-(-q)A_\lambda P_0
={c^2 t_\lambda\over2}(1+M/E)P_0,
\]

where `t_lambda=sum epsilon_j^2 cos^2(q_j/2)` and

\[
T=\sum_\lambda t_\lambda=
\sum_j\cos^2(q_j/2)(1-\widehat q_j^2/Q^2).
\]

Proof: A^2=c^2 t I; beta anticommutes with A, so A beta A=-c^2 t beta;
terms containing an odd number of spatial alpha matrices have zero compression
into P0. Apply this separately to h(q) and h(-q); no sign convention for an
antiparticle has been assumed. These are matrices proportional to the identity
on the rest doublet, not an arbitrarily chosen spin expectation.

Thus the complete finite-mass virtual gap, divided by g², is

\[
D_{\rm virt}={1\over a}\int {c^2T\over4Q}
\left[{1+M/E\over E+z+\Omega}-{1-M/E\over E-z+\Omega}\right]
={1\over a}\int {c^2T\over2Q}
{(M/E)(E+\Omega)-z\over(E+\Omega)^2-z^2}.
\]

The denominator is strictly positive off q=0 for mu,r,c>0. Crucially,

\[
(M/E)(E+\Omega)-z=rW+M\Omega/E>0.
\]

The finite-mass virtual contribution is already positive, not just positive
in an extrapolated fit. Individual new and blocked terms have opposite signs.
The producer evaluates the former through P+(q) and the latter through P-(-q),
then independently checks this scalar combination.

### Contact and the real/paired-mode factor

At rest the actual link-angle second jet in axis j is `(r/a) beta`.
Its Hamiltonian coefficient is 1/2 times the second derivative. The transverse
covariance has trace two, giving

\[
D_{\rm contact}={r\over2a}\int {1\over Q}.
\]

The `1/2` Taylor factor and `g²/(2Q)` vacuum leg are both retained. A normalized
full BZ sums each signed momentum with this covariance; one must not insert an
additional factor two for the two real sine/cosine oscillators of a non-self-
conjugate q pair. The independent q=pi benchmark is a **single real**
self-conjugate momentum mode and one polarization, with covariance
`g²/(2 Q V)`, Q=2, V=2. Its actual contact and resolvent are computed separately;
the mode-pair count is not borrowed from the 3D integral.

### Gauss/Coulomb and the opposite static reference

The exact instantaneous Gaussian Gauss energy is

\[
H_C={u_e\over V}\sum_{q\ne0}{\rho_q\rho_{-q}\over Q^2}.
\]

For a Slater projector n, the connected density expectation is
`Tr[n rho_q (I-n) rho_-q]`. Addition of rank-one e and subtraction of the entire
sea gives, at nonzero q,

\[
\delta\langle\rho_q\rho_{-q}\rangle
=\langle e|(I-P_-(q))-P_-(-q)|e\rangle={M\over E},
\quad D_C={c\over2a}\int {M\over E Q^2}.
\]

This is not the square of the mean density; dropping density transitions would
incorrectly give zero. For the specified static opposite point reference
`qref=-1`, the total q=0 charge is neutral. Its self contribution is

\[
D_{\rm ref,self}={c\over2a}\int {1\over Q^2}.
\]

The nonzero-q diagonal cross expectation with a uniform momentum electron is
zero, because its one-point Fourier density vanishes. The reported carrier
answer excludes only this explicitly displayed reference self; the reported
compensated answer includes it. There is no claim that the momentum state is a
localized bound or separated-reference ground. Subtracting an arbitrary
Coulomb self to force a finite mass would change the stated observable.

## Exact positive coefficient and renormalization decision

For z to zero at fixed positive frozen r,c, define M0=r W and
E0=sqrt(M0^2+c^2 sum sin^2 q_j). The exact formal coefficient is

\[
\delta\mu^{(2)}={g^2\over a}C+o(g^2/a),\qquad
C=C_{\rm contact}+C_{\rm virtual}+C_C,
\]

\[
C_{\rm contact}={r\over2}\int {1\over Q},\quad
C_{\rm virtual}=\int {c^2 M_0 T\over2Q E_0(E_0+cQ)},\quad
C_C={c\over2}\int {M_0\over E_0 Q^2}.
\]

**The parent's proposed three coefficients are correct in this declared
expansion and reference convention.** Each is nonnegative pointwise; contact
is strictly positive, and the other two are positive on an open set for r,c>0.
At q=0, W=O(Q²), E0=O(Q), so all three integrands are O(1/Q) or milder and
integrable in three dimensions. Reference self is O(1/Q²), also integrable.
Elsewhere the Wilson masses remove the other kinetic zeros. The same local
bounds dominate the finite-z contact, virtual and Coulomb expressions as z to
zero; in particular M/E<=1 bounds Coulomb by 1/Q², and positive virtual is
bounded by a constant times 1/Q². Dominated convergence gives the displayed
coefficient, not just a fitted slope. The compensated coefficient has the
additional positive `C_ref=(c/2) integral 1/Q²`.

If the native frame parameters themselves vary with a, apply this conditional
coefficient to their actual values. A fixed nonzero limit for r,c makes C stay
positive (contact alone suffices). The producer exposes those parameters and
recomputes the Gram at each spacing rather than presuming the N=5 source is
independent of a.

**Decision:** a fixed positive primitive light mass has no order-g² ultraviolet
protection in this bare Gaussian Wilson expansion. At fixed g², the expansion
is nonuniform when `g² C/a` rivals the tree mu. This does not establish the full
mass diverges, nor does the large first coefficient justify trusting truncated
perturbation theory at that cutoff. A formal tuned onsite mass would need an
additive negative `-g² C/a` counterterm (plus further renormalization), with the
reference convention stated. This producer does not install that modification.

## Independent full CAR adverse benchmark

The producer constructs the native one-body two-momentum Hamiltonian at
0 and `(pi/a,0,0)` and the Hermitian off-diagonal actual y-current. Its Fock
representation enumerates **all 256 states**, decomposed only by the exactly
conserved particle number. All 70 states in NL=4 and all 56 in NL=5 are used.
No occupation-site or energy-band restriction is made.

It diagonalizes those entire sectors, then computes the neutral scalar and
charged two-by-two degenerate-ground matrices

\[
-|\ell|^2 P_g J(H_N-E_g+\omega)^{-1}JP_g,
\quad { |\ell|^2\over2}P_g C P_g,
\quad {u_e\over VQ^2}P_g\rho^2P_g.
\]

The resolvent acts on the entire sector. Photon n=1 here is the exact and only
intermediate reached from its vacuum by a single linear vertex; it is **not a
retained photon-number cutoff**. The density swap and its square act on the
entire 256-state CAR algebra. An independently diagonalized one-body
occupied/empty calculation and Wick trace must agree with the exact sector
carrier-minus-vacuum matrix. The same `complete CAR carrier gap` assertion is
then applied to three incomplete answers, removing (i) the positive blocked
sea contribution, (ii) the actual contact, or (iii) the Coulomb density
transition. Each must be rejected. No separate easy assertion is substituted.

At this self-conjugate pi mode the transverse new-positive transition vanishes
while the blocked transition survives. This is a strong adverse against the
common omission, not a claim that the 3D integral's new term vanishes.

The benchmark computes order-g² coefficients before inserting the requested
coupling. All complete/component/adverse and scalar-doublet assertions compare
`a` times those coefficients; physical shifts, electric coefficients and vacuum
legs acquire their actual g² only at the reporting boundary. This prevents a
small valid coupling from hiding an omitted term below an absolute energy
tolerance. The same coefficient normalization applies to the BZ scalar-matrix
test. No minimum effect size or relaxed tolerance is used.

Ground clustering uses `min(1e-8/a, mu/4)`, below the finite rest excitation gap,
instead of admitting rest particle-hole excitations when the UV scale grows.
An unresolved numerical ground/rest cluster is an arithmetic failure, not a
claim of changed physical degeneracy. Native consumer regressions at
`(a,g²)=(1,1e-6)` and `(1e-8,.09)` failed before these corrections at the actual
missing-blocked and ground-cluster assertions, then passed against independently
derived pi/y virtual/contact/Coulomb coefficients. The CLI additionally executed
both spacings with g²=1e-6 and rejected all three incomplete laws.


## Safe alternatives and obligations, without editing the bare law

* A negative scalar bare counterterm invalidates the current all-background
  positive-mass Schur hypothesis: its proof uses positivity of
  `mu I+(r/a) W(A)` for every compact background. No old charge floor can be
  reused after that sign change. A changed topological/sign regime or spectral
  gap must be analyzed in the genuinely retuned theory. Renormalized
  perturbation at fixed finite cutoff is a reasonable different construction,
  not an already delivered cure.
* Setting r=0 is not a clean cure for this native Wilson regulator: at z=0 all
  eight corners have zero kinetic energy, restoring doublers. At positive z,
  even the finite-mass Gaussian virtual and Coulomb corrections need not
  vanish. The r to zero observation about the *additive* z=0 coefficient
  does not supply a single electron.
* An anticommuting Hamiltonian matrix producing symmetric energies is **not**
  axial mass protection. In this native four-mode matrix algebra, the
  Hermitian axial generator is `chi=-i alpha_1 alpha_2 alpha_3` (up to overall
  sign). It commutes with the massless kinetic alpha_j and anticommutes with
  beta. The Wilson `r W beta` explicitly breaks `[chi,h]=0` even at mu=0.
  Multiplication by beta gives the corresponding Euclidean Dirac relation:
  genuinely chiral massless D anticommutes with chi; spectral pairing of h
  alone is a different statement. A protection mechanism needs an actual
  Ward identity and regulator (and, where relevant, anomaly/species analysis),
  not merely paired energies. A Ginsparg-Wilson/domain-wall or other protected
  construction would be a materially different microscopic law, requiring a
  new native field/Gauss/current join; none is installed here.
  The producer independently constructs chi and `i beta chi` from the native
  matrices, exposing the nonzero axial commutator and Wilson-only breaking
  while verifying the latter grading still anticommutes with the actual h.
* Staying at a declared finite physical cutoff and reporting the actual
  order-g² debt is the boring safe alternative that changes no law. Letting
  g²(a) decrease enough that g²(a)/a stays bounded is another **different**
  photon trajectory, not fixed-charge UV closure. Neither supplies a
  nontrivial interacting continuum electron/photon by itself.

The first new obligation exposed by this route is constructive mass
renormalization or actual symmetry protection **while preserving the original
full-Gauss carrier mechanism**. Nonlinear compact photon phase, localized
charged/reference sectors, full radiative running, uniform field/volume BO,
and total physical Noether spin remain separate active obligations.

## Executed diagnostics and evidence binding

The native owner execution at a=1,.001,.0001, N=5, orders6/8 and g²=.09
completed naturally; its a=.0001 bare carrier shift was approximately24.69927,
and the explicitly separate point-reference self approximately10.94286.
Their size demonstrates nonuniform perturbative debt, not a trustworthy
interacting mass. The repaired consumer also completed its weak-coupling and
small-spacing CLI and the two regression cases above. Reproduce the default
diagnostic in the scientific project environment:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
proposals/P256-qed-backward/original_quantum_mass_diagnostic.py \
--a 1 .001 .0001 --orders 6 8 --field-size 5 --g2 .09 \
--output proposals/P256-qed-backward/original_quantum_mass_diagnostic_receipt.json
```

The receipt binds source/proof/supplier hashes before and after execution,
actual Python/NumPy/SciPy runtime and BLAS environment, all primitive inputs,
full ten-coordinate native Gram, reconstruction/jet controls, independent
finite-mass matrix terms, mathematical UV coefficients, reference convention,
and exact full-CAR adverse results. Quadrature differences remain numerical
integration uncertainty; they do not decide positivity, which is analytical.
The a=.0001 default provides a further cutoff beyond the requested a=1,.001;
its large perturbative debt is reported, never accepted as a full mass.

Before writing these files, the real Pyright JSON-RPC server was exercised at
`/home/administrator/substrate-222`: it returned `rotation_band.py` document
symbols, resolved its projector's actual h call to the native definition, and
published empty representative diagnostics. The mounted LSP device was absent.
Pyright 1.1.414 also returned the new consumer's actual document symbols and
published empty final new-file diagnostics after resolving SciPy array typing.
A positive Python ast-grep query matched the native `np.einsum` constructs;
another matched the new whole-sector `np.linalg.solve` resolvent.
The shared assignment reserves science execution and final integration to the
parent; these editor probes are not scientific verification.

## Native common-mass construction: a measured formal branch, not a cutover

`original_common_mass_diagnostic.py` supplies a separate constructive
discriminator. It leaves the original m=.2 law unchanged. A common bare-mass
family would change **both** declared species operators:

\[
 \delta H=\delta m_B(-\Lambda O_H+O_L),\qquad
 O_{H/L}=\sum_x c_{H/L,x}^\dagger\Phi_{0,x}c_{H/L,x}.
\]

Changing only the light rest scalar, or freezing the heavy source, is not
this family. At a fixed cutoff write t=a m_B. Its actual N5 heavy Gram law is

\[
 F_t(G)={\kappa\over2}\operatorname{Tr}G^2
 -2\left\langle\sqrt{x_t^TGx_t}\right\rangle_{125},\quad
 x_t=(t+r_b W,v_b\sin q_1,v_b\sin q_2,v_b\sin q_3).
\]

The native provider solves all ten symmetric G coordinates, not just a
prescribed diagonal ansatz. Cubic equality is checked after that solve.
Its all-ten Hessian is
`H_t[deltaG]=kappa deltaG+(1/2)avg xx^T(x^T deltaG x)/(x^T G x)^(3/2)`.
The zero ray has identically zero G derivatives, but its source-mass
contribution is `-2|t|sqrt(G00)/125`. Thus its one-sided response has

\[
 G'(0^+)-G'(0^-)
 ={2\over125\sqrt{G_{0,00}}}\,
       \mathcal H_0^{-1}E_{00}.
\]

The consumer uses explicit nonzero-t native solves and a stated O(epsilon²)
one-sided G0 extrapolation; it does not report an exact native-provider
evaluation at t=0. All125 rays remain in the normalization and the zero
ray is not a removed heavy/light occupation. At the crossing the actual
heavy rest cluster must be rebuilt; an old unique heavy Slaterline cannot
be continued by changing its scalar label.

Let `b0=sqrt(G_t,00)`, `r_t=r_b b0`, `c_t=v_b sqrt(G_t,11)`.
The **same original photon law** at a=.001 holds
`d=a sqrt(2ue K)=c_(G_a*.2)` and `g²=sqrt(2ue/K)=.09` fixed as t varies.
Retuning d to c_t changes that law. With normalized BZ measure, put
`M0=r_t W`, `E=sqrt(M0²+c_t² sum sin²q)`, `Q=|2sin(q/2)|`, and
`T=sum_j cos²(q_j/2)(1-qhat_j²/Q²)`. The formal positive-reference
one-loop coefficient is

\[
 C_t(d)={r_t\over2}\int{1\over Q}
 +\int{c_t^2M_0T\over2QE(E+dQ)}
 +{d\over2}\int{M_0\over E Q^2},\qquad
 \Psi(t)=t b_0(t)+g^2C_t(d).
\]

The three terms are native contact, sea-blocked virtual and Gauss density
contributions; the opposite static-reference self is separately
`(d/2)int 1/Q²`. This is an organization around a positive renormalized
reference, **not** a self-energy computation in the actual negative-bare
occupied sea. The matched original positive-t case independently agrees
with the full matrix-projector producer above.

The actual portable CLI completed at a=.001, g²=.09, orders6/8:

- Same-law formal roots t=-.004612562139960483 and
  -.00461256153882826; the latter means common m_B=-4.61256153882826.
- The original photon d=.09620372849567725 is held fixed; ue=4.329167782305476
  and K=1068.9303166186362 are not silently refitted.
- All-ten source residuals at the roots are below 4e-16. The measured
  one-sided cusp agrees with its full Hessian response within 5.31e-6.
- The actual primitive rest matrix has eigenvalues +/-2.4612410956364386
  at the order8 root. Its occupied heavy rest projector agrees with
  the negative-bare doublet within 2.5e-16; the borrowed positive-bare
  projector differs by norm2. This checks the rest doublet, not a new
  global heavy-vacuum/normal-transfer theorem.
- Freezing the original G and retuning the photon law give distinct
  roots. Target-mass adjustment must use the full derivative of Psi,
  not b0 alone. These are numerical discriminators, not higher-order
  mass predictions.

There is an additional bare-sea debt. On the frozen four-Gamma branch,
the SPD map sqrt(G_t) preserves the degree of x_t/|x_t|. With orientation
`d0 dd1 dd2 dd3`,

\[
 \deg={1\over2}\sum_{n=0}^3{3\choose n}(-1)^n
            \operatorname{sign}(t+2r_b n).
\]

It is zero for t>0 and -1 for -2r_b<t<0, where both measured roots lie.
A positive target split cannot erase this bare occupied-band change.
This does not compute an interacting topological response, invalidate
the heavy/light centre cancellation, or imply a nontrivial determinant
line by itself. The original full Cartesian-field theory is not simply
this frozen translation-invariant map.

The negative bare branch loses the preceding positive-beta all-background
coercivity/positivity proof. The separately rebuilt heavy supplier below
does not repair that light-sector loss. A genuine joined construction
still needs its signed determinant/sector organization, normalized
compact phase, higher-order mass control and reference-separated dynamics.
No common-mass counterterm is installed by these files, and no original
interacting mass, UV limit, total spin or scattering completion is claimed.

## Rebuilt signed whole-heavy supplier and native normal checks

The common-root diagnostic now also exercises the **actual entire**
500-mode/250-occupied heavy cluster and its field/colour/contact jets at
the order8 root. This is not a positive-bare vacuum transported by label.
The original fixed m=.2 law is unchanged; this is the explicitly
reparameterized common-mass family.

An independently reviewed analytic supplier extension uses

\[
 a=.001,\quad -.0047\le t=a m_B\le-.0045,\quad r=.12,\quad v=.3,
 \qquad H_H=-\Lambda\,d\Gamma(h_t),\quad N_H=2V.
\]

It retains full Cartesian fields, all SU4/U1 rotors, the entire fixed-NL
light/reference space and both Gauss laws. The heavy species is
**EM-neutral and colour-coupled**. No photon is incorrectly attached to
its occupied projector. The same declared slow trajectory is used:
field/colour kinetic coefficients proportional to Lambda^-1/2 and
alignment/colour magnetic coefficients proportional to Lambda^5/4.

The required local Bloch gap does not require positive bare mass.
For wj=1-cos(qj), W=sum wj and S=sum wj(2-wj),

\[
 (t+rW)^2+v^2S-t^2
 =\sum_j w_j[2rt+2v^2+(r^2-v^2)w_j]
  +2r^2\sum_{i<j}w_iw_j
 \ge2r(t+r)W\ge0.
\]

This holds for **every** shifted momentum, since 0<=wj<=2, v>=r and
-r<t<0. At a metric-Clifford frame the actual whole-half cluster gap is
therefore at least 2|mB|sqrt(lambda_min(G)). This is a local well statement,
not beta-accretivity for arbitrary photon/colour backgrounds.

The new globality proof uses fresh signed-source constants. In aH units,
the continuum Gram functional is

\[
 f_t(G)=\lVert G\rVert_F^2-2\int_{\mathbb T^3}
                   \sqrt{y_t^TG y_t},\qquad
 y_t=(t+rW,v\sin q_1,v\sin q_2,v\sin q_3).
\]

Its unique minimizer is diag(g0,gs,gs,gs). Neither coordinate can lie on
the PSD boundary: a positive-coordinate perturbation has zero quartic
first derivative and strictly negative sea derivative on a positive
measure momentum patch. This uses y0^2, not the false premise y0>0.
For the stated interval, set

\[
 C_y^2=.79519009,\quad
 s_{\max}=(2\sqrt2 C_y)^{2/3},\quad
 \gamma_*={v^2\over4C_y\sqrt{s_{\max}}},\quad
 D_c=2\pi\sqrt{9r^2+3v^2}\sqrt{2s_{\max}}.
\]

Actual whole-half Ky Fan/trace-norm coercivity gives norm(G)<=smax.
The stationary equations give lambda_min(Gstar)>=gamma_*.
The sign-independent shifted-cell spectral estimate gives
norm(G_Ntheta-Gstar)<=sqrt(2Dc/N), for every centre class.
Thus N>=ceil(8Dc/gamma_*^2) is a sufficient **Gram-interiority**
threshold, not the whole globality threshold.

The continuum Cauchy slack vanishes only on the full-rank Clifford orbit.
Compact reducible tuples are disjoint from that orbit. Their positive
uniform separation over this closed t interval, together with the
finite-grid error, supplies an **existence-only** additional N0.
The full ten-coordinate richness and differentiated-Clifford slack
then give 45 restoring uniform physical normals, all 15 gauge tangents,
and the complete 90V normal space. The negative occupied Slater line
still has the actual whole-NH centre character; it is not erased or
replaced by a neutral bosonic auxiliary.

Scalar localization, the whole complementary/exterior lower bound and
the EM-independent full-Gauss lift use these new signed constants. For
each fixed N in that existence range and sufficiently large finite
Lambda, the rebuilt supplier has

\[
 \lVert K_\Lambda W-WH_{\rm eff,\Lambda}\rVert\le d_\Lambda\to0,
 \qquad
 (1-WW^\dagger)K_\Lambda(1-WW^\dagger)
       \ge g_\Lambda(1-WW^\dagger),\quad g_\Lambda\to\infty,
\]

with the same Lambda^-1/16, Lambda^-5/16 residual powers and
g_Lambda>=c Lambda^1/4-C Lambda^7/40-C. Every offset remains in H_eff;
no finite-Lambda centre conservation, fixed-Lambda thermodynamic result,
positive light-beta Green bound or charged lower-gap theorem is inferred.
The exact analytic extension witness is
`384ac5a935615d7b266d3998fb112dfa07b9922ae270a1170091f7f928f8d501`;
its original slow-kinetic and flat-holonomy prerequisites remain explicit.

The portable default CLI observed these **finite N5** values at centres
000 and102, respectively:

- Complete heavy gaps 4.9224821913 and132.2837962050, above the analytic
  gap floors 2.9743543381 and2.9635261826.
- Minimum curvatures of all45 uniform physical field normals
  88.7780470963 and71.1129430672.
- The full500 whole-half mixed field/colour/contact curvature agrees
  with the independent source Fourier105 Hessian to relative error
  below1.72e-16; the nonzero-q physical direction retains all90 normals.
- Independent whole-projector finite differences agree within3.25e-9.
  Omitting the actual contact produces aH defects .00094358/.00196782;
  these are reached source defects, not a vanishing-error acceptance gate.

These are actual source/normal diagnostics, not quantum normal-vacuum
samples, N5 globality, numerical N0/Lambda certification or an interacting
mass cancellation. The same formal roots and frozen-G/photon-law
discriminators remain unchanged. The rebuilt heavy supplier removes a
specific invalid positivity transplant; the light/photon/charged
renormalization and total physical spin/scattering join remain open.


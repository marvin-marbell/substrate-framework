# Haar centre wells without an imported clock link

## Scope and scientific decision

This supplier starts with the existing actual selected frame and continuous
`ExactHaarAlignment` multiplication. It does not start with a four-dimensional
clock Hilbert space. Its declared quantum model is

\[
\mathcal H=L^2(SU(4),d\mu_H),\qquad
H_{\kappa,J}=\kappa(-\Delta)+JW(U),\quad \kappa>0,\ J\ge0,
\qquad W(U)=\sum_{\mu=0}^3\|[\Phi_\mu,U]\|_{HS}^2.
\]

The fundamental quantum law is **declared**, not derived from substrate Euler
motion. The scalar link Casimir is **not** the actual matter-coupled angular
photon Hamiltonian. In particular it omits the endpoint matter generators,
Gauss section and angular cross terms present in `PhysicalHamiltonian.orientation`.
An alignment potential relative to a fixed frame is a gauge-fixed/fixed-orbit
input; a spatial implementation must supply the covariant endpoint frame law.

What is supplied here:

* Exact scalar-only kernel, global coercivity, and a proved centre-distance
  constant using the actual commutator superoperator.
* Real Haar kinetic geometry and controlled local harmonic comparisons.
* A continuous-Haar four-packet construction with an exact fundamental transport
  compression and explicit amplitude/leakage/energy bounds. These are trial
  packets, not asserted eigenstates.
* Minimal/medium/full exact Haar Ritz probes at several alignment strengths,
  exposing why finite-cutoff locking does not certify the continuum low band.

This replaces an imported clock **at the kinematic and variational level**.
It does **not** license substituting the trial space for the dynamical four-state
link in `gauged_rotation_fractionalization.PauliZ4Pair`. Its four-band isolation,
transport through the true low band, matter-coupled dynamics and spatial phase
remain explicit obligations below. No mechanical half-spin, deconfinement,
three-dimensional phase, full60 quantum solution, or canonical/full-QED claim
is made. In particular a conditional centre minus sign is not an operational
mechanical rotation result.

## Actual frame and conventions

Default parameters are `m=.2, r=.12, v=.3, stiffness=2`, lattice spacing one.
We import the same **unimproved** `RotationBand` and
`uniform_data(band,3,stiffness,0,matrix_unit_basis())` used by the exact Haar
supplier. This is not silently the improved-band selection in the earlier
rotation-fractionalization program. The receipt records the full complex frame,
selected Gram data and source hashes so that difference is visible.

Canonical generators satisfy `Tr(Ta Tb)=delta_ab/2`. Write `X=x_a T_a` and use
`ds^2=2 Tr(dX^2)=sum_a dx_a^2`. Then

\[
-\Delta=\sum_{a=1}^{15}E_a^2,
\quad C_2(4)=15/8,\quad C_2(6)=5/2,\quad C_2(15)=4.
\]

Either left or right generators give this same scalar Laplacian; summing both
would introduce an erroneous factor two. The code uses `kappa*C2(r)` once.
All Haar projections are performed **after** multiplying the continuous
function. No product of separately compressed `PUP` factors is used for W.

## Exact commutator kernel and global coercivity

On all complex 4-by-4 matrices with the HS inner product define

\[
\mathcal L A=\sum_\mu[\Phi_\mu,[\Phi_\mu,A]],\qquad
\langle A,\mathcal LA\rangle=\sum_\mu\|[\Phi_\mu,A]\|_{HS}^2.
\]

This self-adjoint positive operator annihilates the identity. For four nonzero
orthogonal weighted Clifford fields, `Phi_mu=sqrt(g_mu)*Gamma_mu`, their
products `Gamma_S` form a basis of all 16 complex matrices. A product of grade
`k=|S|` anticommutes with `Gamma_j` precisely when either `k` is even and
`j in S`, or `k` is odd and `j not in S`. Thus

\[
\ell_S=4\sum_{j:\,\Gamma_j\Gamma_S=-\Gamma_S\Gamma_j}g_j.
\]

Only the empty product has eigenvalue zero. This proves the scalar-only
commutant and shows that W vanishes on SU(4) exactly at `Z4={I,iI,-I,-iI}`.
The code also constructs the actual 16-by-16 superoperator, compares its full
spectrum against this formula and checks the selected Clifford algebra. The
nonzero extreme eigenvalues are `ell_min` and `ell_max`, not a fitted curvature.

For `A=U-(TrU/4)I`, define

\[
Q(U)=\|A\|_{HS}^2=4-|\operatorname{Tr}U|^2/4.
\]

The exact pointwise bounds, hence also multiplication quadratic-form bounds,
are

\[
\ell_{\min} Q(U)\le W(U)\le\ell_{\max} Q(U).
\]

For the default frame the independently executed first probe found
`g=(.42042038805775556,.05694884768250374,.05694884768250374,.05694884768250374)`.
The corresponding exact-formula extremes are `ell_min=4*g_light` and
`ell_max=4*sum(g)`. Receipt values, rather than these rounded figures, bind
numerical use.

### Proving the global centre-distance constant

Here `d_Z(U)=min_z ||U-zI||_HS` is **chord distance**, not geodesic distance.
Let `t=TrU/4`, `r=|t|`, and `w=t/r` when `r>0`; choose any unit w at r=0.
For eigenvalues `u_j` of U,

\[
\delta=\sum_j|u_j-w|^2=8(1-r),\qquad Q=4(1-r^2).
\]

Telescoping the product of four unit scalars and Cauchy–Schwarz give
`|w^4-1| <= sum_j |w-u_j| <= 2 sqrt(delta)`, since `prod_j u_j=1`.
Choose the closest fourth root z to w, with phase difference `|beta|<=pi/4`.
On the unit circle angle is at most `(pi/2)` times chord, so
`|beta| <= (pi/8)|w^4-1| <= (pi/4)sqrt(delta)`.
Finally

\[
\begin{aligned}
\|U-zI\|_{HS}^2
 &=8(1-r\cos\beta)\\
 &\le\delta+4\beta^2
 \le(1+\pi^2/4)\delta
 \le(2+\pi^2/2)Q.
\end{aligned}
\]

The r=0 case obeys the same inequality directly (`distance^2=8, Q=4`).
Thus the stronger proved constant is `C=2+pi^2/2`, which also proves the
proposed weaker `4+pi^2` bound. In particular

\[
W(U)\ge (\ell_{\min}/C)d_Z(U)^2,
\qquad
\Pr_\psi[d_Z\ge\eta]\le
\frac{C\langle\psi,H\psi\rangle}{J\ell_{\min}\eta^2}
\quad(J>0).
\]

This is an energy-dependent localization statement, not an infinite Higgs gap.

## Haar geometry and local harmonic terms

In a centre chart `U=z exp(iX)`, `|x|<=rho<=1`, the logarithm is unambiguous,
traceless and radial distance equals |x|. If `theta_j` are the eigenvalues of X,
then `|theta_j-theta_k|<=|x|`: square the difference and compare with
`2 sum_j theta_j^2=|x|^2`. The real Haar density relative to these Cartesian
coordinates, up to an irrelevant constant normalization, is

\[
j(X)=\prod_{j<k}\left[
\frac{\sin((\theta_j-\theta_k)/2)}{(\theta_j-\theta_k)/2}
\right]^2.
\]

There are six positive roots, hence with `s=sinc(rho/2)` (sinc here means
`sin(y)/y`),

\[
s^{12}\le j(X)\le1,\qquad
I\le g^{-1}(X)\le s^{-2}I.
\]

These metric bounds follow from the exponential differential on each real
root plane, with singular value `sinc((theta_j-theta_k)/2)`, and singular value
one in commuting directions. Thus the kinetic energy is genuinely

\[
\kappa\int (\partial_a f)^*g^{ab}(X)(\partial_b f)j(X)\,dx,
\]

not a flat-measure oscillator asserted to be the exact rotor. Radial gradients
have metric norm one by the Gauss lemma.

Let `M_ab=sum_mu Tr([Phi_mu,Ta]^dagger[Phi_mu,Tb])`. Then
`q(X)=x^T M x` and the eigenvalues of M are exactly the 15 nonzero eigenvalues
of L divided by **two**, because the canonical generators have HS norm squared
one half. The code checks this factor.

Diagonalizing X directly gives

\[
W(e^{iX})=\sum_{\mu,j,k}|(\Phi_\mu)_{jk}|^2
                4\sin^2((\theta_j-\theta_k)/2).
\]

The elementary inequality
`0 <= y^2-4 sin^2(y/2) <= y^4/12` on `|y|<=1` therefore proves

\[
(1-\rho^2/12)q(X)\le W(e^{iX})\le q(X),\quad |x|\le\rho.
\]

For the local *Euclidean comparison* oscillator
`-kappa Delta_x + J x^T M x`, put `mu_a=eig(M)`:

\[
E_{\bf n}^{osc}=\sum_a\sqrt{\kappa J\mu_a}(2n_a+1),\qquad
E_0^{osc}=\sum_a\sqrt{\kappa J\mu_a},\quad
\mathrm{gap}^{osc}=2\sqrt{\kappa J\mu_{\min}}.
\]

This is not the full-rotor spectrum. More precisely, for Dirichlet restriction
to one radius-rho ball and `c=1-rho^2/12`, min–max yields, for every n,

\[
s^{12}\lambda_n^{D,flat}(\kappa,JcM)
\le\lambda_n^{D,Haar}
\le s^{-12}\lambda_n^{D,flat}(\kappa s^{-2},JM).
\]

The lower comparison can further be bounded below by the whole-space oscillator
at coupling Jc. The JSON prints this explicit lower bound for the first two
one-ball Dirichlet eigenvalues. Dirichlet eigenvalues are **not** lower bounds
for full-rotor eigenvalues: boundary decoupling has the opposite min–max
inclusion. No claim about lambda_5 of the full rotor is inferred from them.

## Materially different construction: four disjoint Haar packets

Rather than extrapolating four small-cutoff Ritz eigenvectors, define exact
Haar H1 functions

\[
f_z(U)=N_\rho^{-1/2}\big(1-d_g(U,zI)/\rho\big)_+,
\qquad 0<\rho\le1.
\]

Use the canonical geodesic metric. The shortest neighbouring-centre geodesic
has length `pi sqrt(6)` (a minimal traceless logarithm of iI has eigenvalues
`pi/2,pi/2,pi/2,-3pi/2`); opposite centres have length `2sqrt(2)pi`.
Consequently these four supports are disjoint. All f_z have the same exact
Haar norm. They are orthonormal after normalization and are quadratic-form
admissible; the boundary cusp means they need not belong to H's operator
domain. Only quadratic-form matrix elements are used.

For dimension d=15,

\[
\frac{\int_{B_\rho}|\nabla(1-r/\rho)|^2dx}
     {\int_{B_\rho}(1-r/\rho)^2dx}
=\frac{(d+1)(d+2)}{2\rho^2}=\frac{136}{\rho^2}.
\]

Radial metric norm is exact, and the Haar denominator is at least s^12 times
the flat denominator. Also `W<=ell_max ||U-zI||^2<=ell_max Tr(X^2)` on the
support. Thus the common Rayleigh value A obeys

\[
A\le A_\rho:=\frac{136\kappa}{s^{12}\rho^2}
                 +\frac{J\ell_{\max}\rho^2}{2}.
\]

Disjoint supports remove both kinetic and potential off-diagonal form elements;
centre invariance of W makes all four diagonal values equal. The exact trial
form is `A I_4`, and **only by min–max**,

\[
\lambda_4(H)\le A_\rho,\qquad
E_{\min,q}(H)\le A_\rho\quad(q=0,1,2,3).
\]

The Fourier sums of the f_z give one trial state of each exact centre charge.
Every vector in the true lowest-four spectral space has energy at most A_rho,
so the global outside-probability bound applies to that space as well, even
when it is not spectrally isolated. The CLI optimizes this explicit upper bound
on `0<rho<=1`; optimization selects a bound, not an inferred eigenstate.
For large J, choosing rho proportional to `(kappa/J)^(1/4)` gives
`A_rho=O(sqrt(kappa J))` and outside probability `O(sqrt(kappa/J))` at fixed eta.

### Exact transport in this trial space

Conjugation invariance of the radial Haar packet and inversion symmetry imply

\[
\int U|f_1(U)|^2d\mu_H=a_\rho I,\qquad a_\rho\in\mathbb R.
\]

For `U=exp(iX)`,
`1-Re TrU/4 <= Tr(X^2)/8=|x|^2/16`. Therefore

\[
1-\rho^2/16\le a_\rho\le1,\qquad
P_{trial}U_{ij}P_{trial}
=\delta_{ij}a_\rho\operatorname{diag}(1,i,-1,-i).
\]

In the electric Fourier basis this is an exact cyclic charge shift, multiplied
by a bounded, physically derived amplitude—not an assumed clock matrix.
The sum of all 16 fundamental multiplication norms on any trial state equals
four by pointwise unitarity, while the compressed contribution is `4a_rho^2`.
Hence

\[
\sum_{ij}\|(1-P_{trial})U_{ij}\psi\|^2
=4(1-a_\rho^2)\le\rho^2/2,\qquad
\|a_\rho Z-Z\|\le\rho^2/16.
\]

These statements hold for all normalized trial-space vectors because supports
are disjoint. They are exact inequalities, not Monte Carlo estimates.
They do not bound `(1-P_trial)H P_trial`, assert invariance, or prove that the
spectral low band has this same transport. In particular the common trial
energy does not compute electric-sector tunnelling splittings. Those are
precisely the missing dynamic controls.

## Discriminating finite probes and the obstruction

The first computation imported the actual exact provider, not quadrature.
With kappa=1, the standalone W Ritz minima were

| cutoff | dimension | min(PWP) |
|---|---:|---:|
| minimal | 33 | 4.414793085585991 |
| medium | 294 | 3.8085602816016597 |
| full | 494 | 3.8085602816016597 |

The continuum multiplication infimum is zero: W is continuous and vanishes
at each centre, and shrinking Haar packets make its expectation vanish.
Its L2 zero eigenspace is trivial, since four points have Haar measure zero.
Every finite Peter–Weyl vector is analytic; an exactly zero expectation would
force it to vanish off those four points and hence everywhere. Thus positive
finite minima are expected, not a Higgs-gap certificate. For **fixed** cutoff,

\[
H_{cutoff}\ge J\min(PWP)I,
\]

so its low energies grow at least linearly with J, whereas the continuum
lambda_4 bound just constructed grows only as sqrt(J). Strong alignment and
fixed projection therefore do not commute. Equality of medium/full minima
is not cutoff convergence: the same centre-zero representation block persists.

There is already a quantitative continuum-versus-projection exclusion at
J=200, kappa=1: the optimized packet radius `.8509599567390166` certifies
`lambda_4 <= 441.61087178205287`, while **every** full-cutoff vector has energy
at least `200*3.8085602816016597 = 761.7120563203319`. Thus at least four
continuum levels lie below the entire fixed full-cutoff spectrum. The JSON
reports this comparison and its margin for each cutoff/packet J, not just a
qualitative warning about possible locking.

The global probability estimate is deliberately not suppressed when weak:
at J=200 the eta=.5 bound is vacuous; at J=1e8 it is `.3089383571335753`,
not less than .1. At J=1e10 the executed packet computation gives
rho=`.010355744686417825`, `lambda_4 <= 2536400.829102236`, outside probability
at most `.030886380966274676`, and trial transport amplitude error at most
`6.702590500641939e-6`. This demonstrates a controlled localization regime
without pretending the finite probe or trial form supplies a separated band.

At J=200, the first probe found centre ground energies (q=0,1,2,3):

* minimal: `(946.027090,884.833617, absent,884.833617)`;
* medium: `(763.887228,884.833617,810.339390,884.833617)`;
* full: `(763.887228,884.833617,785.374629,884.833617)`.

In the full cutoff, ground-to-ground fundamental strengths
`sum_ij ||P_{q+1}U_ij P_q||_HS^2 / dim(P_q)` were approximately
`(.318397,.304553,.304553,.318397)`, not the localized-packet limiting value
four. At J=0 the q=1,2,3 minima have multiplicities 16,36,16 respectively;
minimal has no q=2 sector at all. Finite centre labels alone do not provide an
isolated four-state electric link. The receipt reports the complete lowest
sector eigenvalues, ground multiplicities, global lowest-eight labels and
basis-independent transport strengths/ranks for every requested J. It also
checks nested W projections, so missing high representations are not hidden
behind a supplier inconsistency.

The change of construction is substantive: exact continuous Haar disjoint
packets, rather than an arbitrary clock or extrapolated finite eigenstates.
Their estimates prove controllable variational localization and transport
while keeping the dynamical-band obstruction visible.

## Portable commands and source binding

From a checkout with NumPy/SciPy and existing proposal dependencies installed:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python proposals/P256-qed-backward/haar_centre_rotor.py \
  --cutoffs minimal medium full --alignments 0 .2 2 20 200 \
  --packet-alignments 200 10000 1000000 100000000 10000000000 \
  --m .2 --r .12 --v .3 --stiffness 2 --kappa 1 \
  --output /tmp/haar_centre_rotor.json
```

On the campaign workstation the scientific interpreter is the existing
`substrate-framework/.venv/bin/python`. JSON includes all loaded local Python
sources, this proof note, the original clock source/note, selected frame,
conventions, parameters, runtime versions, tested inequalities and finite
probes. Changing a supplied source invalidates the old binding; rerun the
command rather than reusing an old receipt. No supplier source is edited.

## Remaining obligations, not implied conclusions

1. A **uniform full-rotor** fifth-eigenvalue lower bound and controlled
   four-band isolation, with matching centre-sector energies. The global
   barrier and local Dirichlet comparisons above are available ingredients;
   they do not yet establish this result.
2. Controlled tunnelling splittings, spectral projector comparison with the
   Haar trial packets, and fundamental transport/leakage within the true low
   band. Four finite eigenvalues or a four-dimensional trial form do not do so.
3. The same controls for the **actual matter-coupled angular photon law**, with
   covariant selected-frame sections and Gauss constraints; scalar Casimir
   geometry must not be relabelled as that Hamiltonian.
4. A spatial Gauss-invariant dynamical construction and an operational phase
   and mechanical-rotation experiment. An algebraic stabilizer or conditional
   fractionalization cocycle proves neither deconfinement nor mechanical spin.

The accepted level here is explicit continuous-Haar operator bounds and an
exact transport-bearing variational four-centre subspace. The original clock
assumption is removed at that evidenced level and remains unlicensed as a
stand-in for the full dynamics.

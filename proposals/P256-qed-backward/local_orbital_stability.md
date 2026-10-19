# Spatially unrestricted orbital stability on the actual periodic sea

This extends the uniform [soft-budget theorem](soft_budget.md) to a **finite-box
local minimum modulo global PU(4)**. Every site has all 60 real traceless
Hermitian orbital coordinates. The stabilizing law is an ordinary quartic Gram
potential plus nearest-neighbor alignment; it contains neither a hard Gram
constraint nor a Clifford penalty. The result retains the extra physical orbital
modes rather than declaring them gauge or discarding them.

The theorem supplies sufficient conditions, and the complete CLI has now
executed the full local Hessian and deformation consumers. The default
1620-coordinate L=3 Hessian has no negative nonuniform mode even at J=0.
This is a finite-box local minimum modulo the global orbit, not a proof that
the uniform configuration globally minimizes the spatial functional or a
thermodynamic phase theorem.

## 1. Operator, state, units, and one path convention

Let $\Lambda=(\mathbb Z/L\mathbb Z)^3$, $N=L^3$, $L\ge3$, with physical spacing
$a>0$ and periodic matter and orbital fields. Each site owns four independent
traceless Hermitian $4\times4$ bosonic coordinate matrices $A_\mu(x)$,
$\mu=0,1,2,3$. In the existing matrix-unit basis,

\[
 A_\mu(x)=\sum_{b=1}^{15}z_{x\mu b}B_b,
 \qquad \operatorname{tr}(B_bB_c)/4=\delta_{bc}.
\]

Thus the field metric is $\|z\|^2=\sum_{x\mu}\operatorname{tr}(A_\mu(x)^2)/4$.
No candidate is normalized, orthogonalized, or retracted. Set $\hbar=1$; the
$A$ and $g_0$ are dimensionless, $m,r/a,v/a,\kappa,J$ have energy units, and
$q$ below is a dimensionless lattice angle. The graph Laplacian is dimensionless;
its long-wavelength physical stiffness is $Ja^2$.

The one-particle Hamiltonian acts on $\ell^2(\Lambda)\otimes\mathbb C^4$.
The **scalar** directed coefficients are extracted from
`RotationBand.couplings`, without a second Dirac/CAR constructor. The improved
case is the one used throughout this module:

\[
 c_{\rm on}=m+\frac{9r}{2a},\quad
 c_{e_i,0}=-r/a,\quad c_{e_i,i}=-iv/(2a),\quad
 c_{2e_i,0}=r/(4a).
\]

All unlisted coefficients vanish. The prescription is exactly

\[
 H_{xx}=c_{\rm on}A_0(x),\qquad
 H_{x,x+d}\mathrel{+}=\sum_\mu c_{d\mu}
                      \frac{A_\mu(x)+A_\mu(x+d)}2,
 \qquad H_{x+d,x}\mathrel{+}=H_{x,x+d}^{(d)\dagger}.
\]

**Every long path uses its two endpoints only**, not a midpoint or an average
along its trajectory. Displacements wrap modulo $L$. Distinct directed terms
that reach the same site are added, not deduplicated: on $L=3$, the improved
length-two path is distinct from the oppositely oriented length-one path even
though their endpoints coincide. This is important both for the mass dispersion
and for the Hessian.

The electromagnetic background is zero in this stability calculation. With EM
restored, each term receives its ordered Wilson product along the same straight
primitive path, followed by its Hermitian adjoint. Since the orbital coefficient
acts on the auxiliary/internal factor, it commutes with the electromagnetic
charge and link operators; the endpoint prescription is gauge compatible. Local
PU(4) conjugations of orbital fields are **not** EM gauge transformations.

The half-filled state is the actual ground state at **total** particle number
$2N$, not a constraint of two particles at every momentum or every site. Its
energy is

\[
 E_{\rm sea}(A)=\sum_{n=1}^{2N}\lambda_n(H(A)).
\]

Its microscopic statistics owner remains the edge-qubit
`bosonic_statistics.primitive_pauli_words` law, through `EdgeWords` and the
neutral boundary parity marker of [pauli_compton](pauli_compton.md). Spectral
reduction of a number-conserving quadratic Hamiltonian in that physical code
sector gives the displayed occupied sum; it is not a new microscopic fermionic
input. The CLI separately evaluates an actual six-dimensional two-particle
edge-word sector on a four-orbital block with an unrestricted perturbation and
compares its entire spectrum with the pair sums. The large $4N$ calculation is
one-particle spectral reduction, not explicit diagonalization of an exponential
edge-qubit Hilbert space.

The physical energy is extensive:

\[
 \mathcal E(A)=E_{\rm sea}(A)
 +\frac\kappa2\sum_x\|G(x)-g_0I\|_F^2
 +\frac J2\sum_{x,i,\mu}\|A_\mu(x)-A_\mu(x+e_i)\|_{HS}^2,
 \quad G_{\mu\nu}(x)=\operatorname{tr}(A_\mu(x)A_\nu(x))/4.
\]

Each positive-axis nearest bond is counted once per site and axis. Assume
$m,v,\kappa,g_0>0$, $r\ge0$, and use the rich $L\ge3$ finite momentum grid. The
CLI reports the design rank rather than silently assuming it.

## 2. Derive the uniform witness after soft selection

The uniform scalar symbol is precisely

\[
 X(p)=\left(m+\frac ra\sum_i[1-\cos(ap_i)]^2,
                 \frac va\sin(ap_1),\frac va\sin(ap_2),\frac va\sin(ap_3)\right).
\]

The soft-budget argument minimizes the strictly convex Gram functional

\[
 f(G)=-\frac2N\sum_p\sqrt{X(p)^TGX(p)}
                      +\frac\kappa2\|G-g_0I\|_F^2.
\]

Cubic symmetry gives $G_*=\operatorname{diag}(g_m,g_s,g_s,g_s)$, with
$g_m,g_s>g_0$. `select_gram` is run on **this box's same momentum grid**, not
on an infinite-volume substitute. Only after this selection do we instantiate
$A_\mu^*=\sqrt{(G_*)_{\mu\mu}}\,\Gamma_\mu$, where the existing primitive
matrix-unit witness `SELECTED` saturates the sea bound. Its metric Clifford
relations are the derived minimizing property, not a term of the energy.

Let $R_p=\sqrt{X(p)^TG_*X(p)}$. The reference spectrum consists of two
$-R_p$ and two $+R_p$ eigenvalues per momentum. The actual global occupied/empty
gap is $\delta=2\min_pR_p>0$. The uniform gradient vanishes. By translation
invariance, every local gradient is equal, hence the uniform stationary point
is stationary under **all** sitewise variations. Alignment also has zero
first variation there.

## 3. Exact unrestricted second variation

For an arbitrary real sitewise direction $D_\mu(x)$, let $V(D)$ be the
Hamiltonian with $A$ replaced by $D$ in the actual linear local formula. It has
no second derivative. With occupied and empty reference eigenvectors $|o\rangle,
|e\rangle$, the exact spectral Hessian is

\[
 \mathcal H_{\rm sea}(D,F)
 =-2\operatorname{Re}\sum_{o\le2N<e}
 \frac{\langle o|V(D)|e\rangle\langle e|V(F)|o\rangle}
      {\epsilon_e-\epsilon_o}.
\]

Degeneracy *within* the occupied or empty subspaces causes no difficulty: this
sum is basis independent and only crosses the strictly positive Fermi gap.
There are no $V''$ or seagull terms for this linear orbital law.

Write $C=G_*-g_0I$ and

\[
 (DG[D])_{\mu\nu}=\langle D_\mu,A_\nu^*\rangle_{HS}
                              +\langle A_\mu^*,D_\nu\rangle_{HS}.
\]

The elastic Hessian at each site is exactly

\[
 \mathcal H_{\rm el}(D,F)
 =\kappa\langle DG[D],DG[F]\rangle_F
           +2\kappa\sum_{\mu\nu}C_{\mu\nu}\langle D_\mu,F_\nu\rangle_{HS}.
\]

The second term cannot be dropped just because the minimizing Gram has been
selected: the unrestricted coordinates have a quadratic Gram variation. Since
$C\succ0$, this elastic Hessian is positive semidefinite (in fact positive on
its own). Alignment contributes

\[
 \mathcal H_{\rm align}(D,F)=J\sum_{x,i,\mu}
 \langle D_\mu(x)-D_\mu(x+e_i),F_\mu(x)-F_\mu(x+e_i)\rangle_{HS},
\]

which is $J\mathcal L\otimes I_{60}$. These are all terms of the full
$60N\times60N$ Hessian. Amplitude, Gram-shape, representation, and spatial
orbital modes are present without projection.

### Exact Fourier reduction, with a real-space crosscheck

Use unitary field Fourier amplitudes $d(q)=N^{-1/2}\sum_xe^{-iqx}d(x)$,
with $d(-q)=\overline{d(q)}$. The endpoint convention gives the particularly
simple **exact**, not continuum, vertex

\[
 \langle p+q|V(D_q)|p\rangle
   =\frac1{\sqrt N}\sum_\mu t_\mu(p+q,p)D_\mu(q),
 \qquad t_\mu(k,p)=\frac{X_\mu(k)+X_\mu(p)}2.
\]

It includes the mass, nearest, and length-two Wilson coefficients. It is
false if midpoint coefficients are substituted for the prescribed endpoints.
For $a=(\mu,b)$ define
$F_a^{st}(k,p)=t_\mu(k,p)\langle u_s(k)|B_b|u_t(p)\rangle$.
The complex Hermitian sea block is

\[
 H_{\rm sea}(q)_{ab}
 =-\frac1N\sum_p\left[
   \sum_{s\in+,t\in-}\frac{\overline{F_a^{st}}F_b^{st}}{R_{p+q}+R_p}
  +\sum_{s\in-,t\in+}\frac{\overline{F_a^{st}}F_b^{st}}{R_{p+q}+R_p}
 \right].
\]

Both orientations are required. At $q=0$ this is the real uniform Hessian,
with the usual factor two. At nonzero $q$, do not discard imaginary entries:
$H(-q)=\overline{H(q)}$, and real cosine/sine modes can mix. Set

\[
 H_J(q)=H_{\rm sea}(q)+H_{\rm el}+J\lambda(q)I_{60},\qquad
 \lambda(q)=4\sum_i\sin^2(q_i/2).
\]

The inverse unitary DFT reconstructs every entry of the real full Hessian.
The CLI diagonalizes that actual $60N\times60N$ reconstruction and compares
its complete spectrum with the union of all Fourier spectra. Independently,
it compares the transformed real-space local vertex for **all 60 fibers at
one site** with the endpoint Fourier vertex at every pair of momenta.
Translation then covers the full sitewise tangent basis, not just a selected
texture. In addition,
it builds local real-space perturbation matrices and evaluates occupied/empty
spectral entries and full-direction bilinear forms on the actual $4N$ reference
Hamiltonian. It also checks the real-space band against the scalar-path Fourier
symbol and checks unrestricted finite deformations. These checks are designed
to expose an erroneous Fourier prefactor, a dropped orientation, a wrong
length-two convention, or an imposed Gram projection.

## 4. Uniform kernel is precisely the 15 global basis directions

Alignment vanishes on $q=0$. The soft-Gram lower bound and its $\kappa$-strong
convexity restore all ten independent Gram directions. On the fixed-$G_*$
fiber, the scaled version of [representation_stability](representation_stability.md)
has 50 tangents, 35 restoring directions, and 15 simultaneous unitary
conjugations. Invertible diagonal rescaling transfers the linearized Clifford
rank argument to $G_*$; the rich quadratic momentum design forces all metric
Clifford defects. No additional flat tangent survives.

More explicitly, write the nonnegative saturation excess as
$\Phi(A)=E_{\rm sea}(A)/N+2N^{-1}\sum_p\sqrt{X(p)^TG(A)X(p)}$.
At the witness, $\Phi=0$ and its Hessian is positive semidefinite. The Hessian
of $f(G(A))$ at its stationary Gram is bounded below by
$\kappa\|DG[D]\|_F^2$. Therefore a zero full uniform curvature must have
$DG[D]=0$; the fixed-Gram result then puts $D$ in the conjugation orbit.
Conversely those directions have exactly zero curvature because the physical
functional is invariant under one global conjugation. This proves 45 positive
and 15 zero eigenvalues in the **ambient 60-dimensional** uniform sector;
it is not a count obtained by normalizing candidates.

The Clifford witness is irreducible in four dimensions. Its simultaneous
commutant is the scalar identity, so the traceless generators
$D_\mu=i[T,A_\mu^*]$ span precisely 15 independent PU(4) directions. The CLI
constructs these actual columns, embeds them uniformly over sites, checks that
the full Hessian annihilates them, and checks a finite global conjugation's
energy directly.

## 5. A finite-box sufficient scalar alignment theorem

Let $M_X=\max_p\|X(p)\|_2$. Since every endpoint vertex is an average,
$\|t(k,p)\|_2\le M_X$. The HS basis normalization and Parseval give

\[
 \|V(D)\|_F^2
 =\frac4N\sum_{q,p}\Big\|\sum_\mu t_\mu(p+q,p)D_\mu(q)\Big\|_{HS}^2
 \le4M_X^2\|D\|^2.
\]

Because $V(D)$ is Hermitian, its occupied/empty and empty/occupied off-diagonal
blocks have the same Frobenius norm. Consequently

\[
 \mathcal H_{\rm sea}(D,D)
 \ge-\frac{2}{\delta}\|P_+V(D)P_-\|_F^2
 \ge-\frac1\delta\|V(D)\|_F^2
 \ge-C_{\rm sea}\|D\|^2,
 \qquad C_{\rm sea}=4M_X^2/\delta.
\]

The denominator estimate is the occupied/empty spectral operator-norm
(resolvent) bound $\|(\epsilon_+-\epsilon_-)^{-1}\|\le1/\delta$; it applies
to the actual local matrix map. It does not assume a sitewise sea or discard
particle transfer. The quartic elastic Hessian is nonnegative. For zero-mean
site fields, the minimum graph Laplacian eigenvalue is

\[
 \lambda_1=4\sin^2(\pi/L).
\]

For any chosen energy margin $\eta>0$, take

\[
 \boxed{J_{\rm suff}=(C_{\rm sea}+\eta)/\lambda_1.}
\]

Then every nonuniform direction has curvature at least
$\eta\|D\|^2$. Translation makes the uniform and zero-mean sectors exactly
orthogonal for the Hessian. Combining this with Section 4 proves:

> On the stated finite periodic box and massive rich scalar band, the selected
> uniform field is a strict local minimum of the full unrestricted physical
> energy modulo global PU(4), with precisely 15 symmetry zero directions and
> $60L^3-15$ positive Hessian eigenvalues, for $J\ge J_{\rm suff}$.

The finite gap makes the occupied-sum functional smooth in a neighborhood.
Positive Hessian on the transverse slice and the exact compact symmetry orbit
then give a local Morse-Bott minimum and a local quadratic distance estimate.
This does not assert global exclusion of large nonuniform textures. The
algebraic bound is rigorous; reported floating values and spectral thresholds
are numerical evaluations, not directed-rounding certificates.

The exact box threshold is also available without a sufficient-bound
approximation:

\[
 J_{\rm crit}(L)=\max_{q\ne0}\max\left(0,
                 -\lambda_{\min}(H_0(q))/\lambda(q)\right).
\]

The CLI reports it from every physical Fourier block, and reports any $J=0$
negative texture actually present. It does not demand that a negative texture
exist to validate a theorem that also covers an already stable $J=0$ box.

## 6. Which limit has actually been controlled?

At fixed positive $m,a,r,v,\kappa,g_0$, the crude sufficient bound scales as
$L^2$: $\lambda_1\sim4\pi^2/L^2$, while $C_{\rm sea}$ stays nonzero and
bounded for the massive finite-band law. Thus this theorem's **required
sufficient** $J$ diverges. That is a loss of this gap-only estimate, not a proof
that the physical $J_{\rm crit}(L)$ diverges or that the orbital field is
thermodynamically unstable.

The deterministic CLI also constructs fresh soft-selected boxes at
$L=3,4,6,8,12$ by default and evaluates the smallest axial angle $q=2\pi/L$.
It reports the actual minimum curvature divided by $q$ and by the graph
Laplacian, as well as the orbit-compressed odd-in-$q$ operator norm divided by
$q$ and the even-in-$q$ norm divided by $q^2$. These distinguish evidence for a
one-gradient imaginary antisymmetric coupling from a quadratic stiffness.
The executed axial sequence retains positive curvature at J=0. At L=12,
the minimum divided by the graph Laplacian is `.01320499407`; the orbit
even-in-q norm divided by q² is `.06029717895`, while the odd norm divided
by q is `1.19e-15`. These are finite-grid observations, not a fitted
thermodynamic theorem. The axial threshold is a lower bound on each larger
box's full threshold, not a replacement for evaluating every momentum.
Complex blocks remain present, so imaginary-gradient couplings cannot be
hidden by a cosine-only probe.

The [thermodynamic hydrodynamic construction](orbital_hydrodynamics.md) now
computes the full small-q Schur complement after relaxing the 45 restoring
modes. Its executed J=0 response has positive sampled stiffness and a
nonzero Berry/gyroscopic form. This strengthens the finite-box result without
discarding physical orientation modes. Uniform angular positivity,
nonlinear quantum ordering and materially different modulated phases remain
distinct constructions.

## 7. Discriminating physical controls

The CLI includes, without altering the energy:

* A commuting spatial triple with the **same mass field**. It has exactly the
  same two-plus/two-minus spectrum at $p=0$, yet the full filled-grid energy
  determines whether it is a false accidental spinor. No zero-momentum matching
  is accepted as an energetic algebra-selection proof.
* A spatially varying **exact local conjugation** of the selected witness.
  Every site's Gram and metric Clifford identities remain exact. Therefore a
  pointwise Clifford-defect penalty vanishes identically along this texture,
  while the actual alignment energy is nonzero. This rejects replacement of
  the physical spatial law by a hidden Clifford penalty. The sea and full
  finite deformation still use the actual endpoint Hamiltonian.
* Actual unrestricted amplitude, full random, soft uniform, and nonuniform
  eigenmode finite deformations at two step sizes, compared with the exact
  Hessian for both $J=0$ and the derived $J_{\rm suff}$.
* An actual global conjugation, required to preserve the entire physical
  energy and remain a null Hessian direction.

## 8. Quantum orbital fluctuations are a constructive extension

There are 45 restoring uniform Gram/representation coordinates and 15
orientation coordinates whose global motion costs no potential energy. Their
nonuniform fluctuations receive the actual sea response and physical $J$
alignment. They are additional bosonic modes of this proposed matter supplier;
one should quantize and measure them, not relabel all local orientations as an
unintroduced gauge symmetry.

A natural auxiliary Hamiltonian is
$H_A=\sum_{x\mu b}\Pi_{x\mu b}^2/(2M_A)+V_{\rm quartic}+V_{\rm align}$,
with ordinary coordinate/momentum commutators. For
$s_x=\operatorname{tr}G(x)=\sum_\mu\|A_\mu(x)\|_{HS}^2$,

\[
 \frac\kappa2\|G(x)-g_0I\|_F^2
 \ge\frac\kappa8s_x^2-\kappa g_0s_x+2\kappa g_0^2.
\]

This is a coercive quartic potential on every real site coordinate. Linear
matter couplings are form-bounded against it by Young's inequality on a finite
box. Positive alignment does not spoil confinement. With the usual common
form domain, this gives a lower-bounded confining quantum auxiliary law,
including noncommuting $H_A$ and $A_\mu(x)$.

The exact code/Gauss sector identity of
[quantum_orbital_sector](quantum_orbital_sector.md) is compatible with these
coefficients: its $Q_\eta R_\eta$ transform acts on the primitive matter/code
factor and as identity on the auxiliary orbital factor. Every gauged transfer
retains the same endpoint coefficient under the transform; $H_A$, including
kinetic, quartic, and alignment terms, is common to the compared code sectors.
Thus the existing $\Delta-2K$ defect margin is not lost merely because orbital
coordinates fluctuate or fail to commute with their own dynamics. For
unbounded coordinates, the equality is a quadratic-form/domain statement,
not an invitation to subtract undefined operators. This compatibility is not
yet a quantum ordered-phase theorem, nor does it prove that zero-point orbital
fluctuations preserve the classical BO gap.

The orbit Schur-complement and quantum harmonic variance are now constructed
in [orbital_hydrodynamics](orbital_hydrodynamics.md). The actual coupled
[quantum radial consumer](quantum_radial_spinor.md) further replaces the
prescribed field amplitude by pure-quartic bosonic dynamics. Independent
shape/spatial coordinates, their charged clouds and their same-current
responses remain active rather than being declared frozen by these results.

## Reproduction and source references

Parent-owned execution from the proposal directory:

```sh
python local_orbital_stability.py --output local_orbital_stability_receipt.json
```

`--L` changes the full Hessian box; `--sequence` changes the diagnostic boxes.
The default full matrix has 1620 real coordinates and the actual one-particle
matrix has dimension 108. Source SHA256, all eigenvalues, every Fourier block,
local spectral crosschecks, finite deformations, controls, and hypotheses are
written to the deterministic JSON receipt. A nonzero exit indicates a failed
observable check, not a repaired or suppressed assertion.

Existing derivations reused: [soft_budget.py](soft_budget.py),
[representation_stability.py](representation_stability.py),
[rotation_band.py](rotation_band.py),
[pauli_compton.py](pauli_compton.py), and
[quantum_orbital_sector.py](quantum_orbital_sector.py).
General occupied-subspace perturbation theory can be found in T. Kato,
[*Perturbation Theory for Linear Operators*](https://doi.org/10.1007/978-3-642-66282-9),
Chapter II; the finite occupied-sum variational principle is Ky Fan's principle,
[PNAS 35 (1949), 652–655](https://doi.org/10.1073/pnas.35.11.652).
The formulas and sufficient bound above are derived for the stated local
operator rather than imported from a relativistic dispersion.

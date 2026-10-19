# Filled-sea energetic origin of the four-orbital spinor algebra

This construction replaces imposed spectral-isotropy identities with an actual
occupied-energy variational mechanism. The admissible orbital law has four
initially unassigned labels, not two preassigned spin states. Its constraints are
only Hermiticity, zero trace and a Hilbert–Schmidt (HS) matrix budget. Clifford
relations are an output of energy minimization on a rich momentum set, not a
penalty term. The half-filling/exclusion law is supplied separately by the
primitive bosonic-statistics construction; bare Bose occupation would be a
different energy problem.

## 1. Variational problem and assumptions

Let `A_0=M`, `A_i=T_i`, `i=1,2,3`, be spatially uniform 4 by 4 Hermitian matrices
with

\[
\operatorname{tr}A_\mu=0,\qquad
\operatorname{tr}(A_\mu A_\nu)=4\delta_{\mu\nu}.
\]

There are 15 real traceless Hermitian matrix-unit directions, so an admissible
frame is an arbitrary real orthonormal four-frame in that space. Neither
`A_mu^2=I`, anticommutation nor an opposite-pair spectrum is an admissibility
constraint. The executable generates unrestricted random four-frames by QR in
this full 15-dimensional space, not merely rotations of a Clifford frame.

For positive weights `w_p`, define

\[
x(p)=\left(\operatorname{mass}(p),\frac{v}{a}\sin(ap_1),
 \frac{v}{a}\sin(ap_2),\frac{v}{a}\sin(ap_3)\right),\qquad
h_A(p)=\sum_{\mu=0}^3x_\mu(p)A_\mu.
\]

The scalar mass and local link law are exactly `RotationBand.mass` and
`RotationBand.couplings`. For the nearest-link version,

\[
\operatorname{mass}(p)=m+\frac r a\sum_i(1-\cos(ap_i)).
\]

The improved version replaces each `1-cos` by its square and adds the existing
next-nearest links. Let `lambda_1 <= ... <= lambda_4` be the actual eigenvalues
at each momentum. The sea functional is

\[
\mathcal E[A]=\sum_p w_p(\lambda_1(p)+\lambda_2(p)).
\]

This is the exact ground energy of the uniform quadratic fermionic sector at
half filling. Even if only the *total* particle number `2N` is fixed rather than
two particles at every momentum, the theorem remains valid: every traceless
4 by 4 block has unconstrained occupied energy bounded below by `-2R(p)`.
For `k` negative eigenvalues, zero trace and Cauchy–Schwarz give
`sum_negative >= -sqrt(k(4-k) tr(h^2)/4) >= -sqrt(tr(h^2))`.
The selected Clifford frame has exactly two negative eigenvalues at every
gapped momentum, so its unconstrained minimizer already has `2N` particles and
attains the global fixed-total-number bound. The CLI explicitly computes the
requested two-lowest-eigenvalue functional.

The mechanism is conditional on the common scalar dispersion, HS budget,
translation-uniform mean field, supplied exclusion law and half filling. These
are opportunities for a deeper supplier, not a proof that generic interacting
bosons spontaneously obey this particular budget. No matrix identity is hidden
inside the functional.

## 2. Four-level occupied-energy theorem

**Lemma.** For any traceless Hermitian 4 by 4 matrix `h`,

\[
\lambda_1+\lambda_2\geq-\sqrt{\operatorname{tr}h^2}.
\]

Equality holds precisely when the spectrum is `(-R,-R,R,R)`, where
`R=sqrt(tr(h^2)/4)`. This includes the zero matrix when `R=0`, and is equivalent
to `h^2=R^2 I` together with zero trace.

**Proof.** Put `s=lambda_1+lambda_2`, `alpha=lambda_2-lambda_1`, and
`beta=lambda_4-lambda_3`. Zero trace gives `lambda_3+lambda_4=-s`, while ordering
gives `s<=0`. Directly,

\[
Q:=\operatorname{tr}h^2=s^2+\frac{\alpha^2+\beta^2}{2}.
\]

Therefore `s>=-sqrt(Q)`, with equality exactly when both pair splittings vanish.
No sign or pair-degeneracy assumption was used. Indeed, the pointwise energy
excess obeys the exact identity

\[
\epsilon=\sqrt Q+s=
\frac{(\alpha^2+\beta^2)/2}{\sqrt Q-s}
\]

away from the zero matrix. The actual sea energetically rewards removing the
within-occupied and within-empty splittings at a fixed second-moment budget.

**Selection theorem.** The HS budget fixes

\[
\operatorname{tr}h_A(p)^2=4\lVert x(p)\rVert^2=4R(p)^2
\]

for every admissible frame. Hence

\[
\mathcal E[A]\geq -2\sum_p w_pR(p).
\]

There exists an admissible frame attaining the lower bound *at every momentum*:
the four matrix-unit operators `MIXING,HOPPING[0:3]` already used by
`rotation_band.py`. Their anticommutation verifies this witness; it is not
required of any alternative. Consequently this witness globally minimizes the
actual filled-sea energy over the entire HS four-frame budget.

Since every excess is nonnegative and every weight is positive, any minimizer
must saturate at every included momentum. Thus minimization gives the
operator-valued quadratic equation

\[
0=h_A(p)^2-R(p)^2I
=\sum_\mu x_\mu^2(A_\mu^2-I)
 +\sum_{\mu<\nu}x_\mu x_\nu\{A_\mu,A_\nu\}.
\]

This is the route from energetic selection to an algebra.

## 3. Exact finite-set identifiability and a ten-momentum witness

Let `Q_P` have the ten columns `x_mu x_nu`, `mu<=nu`. If `rank(Q_P)=10`, the
quadratic equation at all momenta forces each matrix coefficient to vanish:

\[
 A_\mu^2=I,\qquad \{A_\mu,A_\nu\}=0\quad(\mu\ne\nu).
\]

Thus **the global sea minimizers are exactly Clifford four-frames** on a
full-rank positive-weight momentum set. This is not unique selection of an
orbital basis: simultaneous unitary conjugation remains a degeneracy, as it
must for initially unassigned orbital labels. The complex four-generator
Clifford algebra is `Mat_4(C)`, so these four-dimensional representations are
unitarily equivalent to the existing consumer's frame.

A rank-ten momentum set is constructive, not merely generic. Choose any `k`
with `sin(ak)!=0`, and use the following **ten** momenta:

\[
0,\quad \pm k e_1,\pm k e_2,\pm k e_3,\quad
k(e_1+e_2),k(e_1+e_3),k(e_2+e_3).
\]

Write `t=v sin(ak)/a`, `mu=mass(k e_i)`, and `m=mass(0)`, assuming `m mu t!=0`.
First `p=0` fixes `A_0^2=I`. Sum and difference of the two axial equations then
fix `A_i^2=I` and `{A_0,A_i}=0`. Each two-axis equation finally fixes
`{A_i,A_j}=0`. Equivalently the absolute determinant of this ten-row quadratic
design is

\[
 |\det Q_P|=8m^2|\mu|^3|t|^{15}>0.
\]

For `k=2pi/(3a)`, all ten momenta occur modulo reciprocal periods in the actual
periodic `L=3` cubic grid. For the CLI domain `m,a,v>0,r>=0`, `mu>0`, for both
nearest and improved mass laws. The default **27 actual cubic momenta therefore
identify all ten products exactly**; the program also compares the numerical
and analytic ten-point determinant.

There is no general statement that every cubic grid is sufficient. For `L=2`,
all spatial sines vanish exactly, so only the mass-square column survives and
rank is one; floating-point remnants of `sin(pi)` are diagnosed by SVD rather
than promoted to information. For momenta `(q,q,q)`, the columns collapse into
at most `mass(q)^2`, `mass(q) sin(aq)` and `sin(aq)^2`, giving rank at most three.
A rank-deficient set supports only the qualified equation `Q_P D=0`, not
vanishing of all ten coefficients. The CLI reports these adverse ranks and
also provides 48 reproducibly seeded noncyclic momenta.

There is a quantitative strengthening. Put
`D=(A_mu^2-I, {A_mu,A_nu}_{mu<nu})` in the same column order, and let
`B_p=h_A(p)^2-R(p)^2I`. With combined matrix Frobenius norms,

\[
\lVert D\rVert_F\leq\sigma_{\min}(Q_P)^{-1}\lVert B\rVert_F.
\]

For each momentum, the exact spectral expression is

\[
\lVert B_p\rVert_F^2=
\frac{(\alpha^2-\beta^2)^2}{16}
 +\frac{s^2(\alpha^2+\beta^2)}2.
\]

Since `alpha^2+beta^2=8R epsilon-2epsilon^2`, one obtains

\[
\lVert B_p\rVert_F^2\leq4R^2\epsilon^2+16R^3\epsilon.
\]

Thus a nearly optimal sea on a well-conditioned finite momentum set also
bounds the full algebra defect; the result is not just a discontinuous exact
saturation claim. Momentum conditioning is the meaningful remaining control.

## 4. Smooth admissible deformation with positive energetic curvature

Take the witness `M=X tensor I`, `T_i=Z tensor sigma_i`, expressed using the
same initially unassigned label matrix units as `rotation_band.py`. Set

\[
B=I\otimes\operatorname{diag}(1,-1),\qquad
 M(\theta)=\cos\theta M+\sin\theta B,
\]

and keep all three `T_i` fixed. `B` is HS-orthogonal to the four-frame, so this
is an exact `SO(15)` plane rotation of the matrix-unit coordinates, preserving
the budget for every angle. It leaves the Clifford locus: `B` commutes with
`M,T_3` and anticommutes with `T_1,T_2`. In particular, `M(theta)^2-I` is
nonzero at generic small nonzero angle. The program exponentiates the actual
15-dimensional antisymmetric generator; it does not repair the Gram matrix
after an arbitrary deformation.

At a gapped witness momentum, the occupied projector is
`P_-=(I-h_0/R)/2`. With `V=x_0 B`, `W=-x_0 M`, ordinary degenerate-subspace
perturbation theory for the *sum of the two occupied energies* gives

\[
\begin{aligned}
e_{\rm occ}''(0)
 &=\operatorname{tr}(P_-W)
  -\frac1R\operatorname{tr}(P_- V P_+ V)\\
 &=\frac{x_0^2}{R}\operatorname{tr}(P_-B P_-B)\\
 &=\frac{2x_0^2(x_0^2+x_3^2)}{R^3}>0
\end{aligned}
\]

when `x_0!=0`. For positive weights, the sea curvature is the positive sum of
these expressions. This shows an explicit restoring energetic direction from
the much larger allowed HS space. It does not claim every tangent direction
has positive curvature: unitary-equivalence directions are flat. The CLI
compares the analytic sum with centered energy differences for angles
`0.1,0.03,0.01,0.003`, while reporting excess energy, algebra defect, Gram error
and `SO(15)` orthogonality error. Numerical minimization is unnecessary for the
global theorem.

## 5. Return to spin, current and Compton

For any selected frame, define

\[
 J_1=-\frac i2T_2T_3,\quad J_2=-\frac i2T_3T_1,\quad
 J_3=-\frac i2T_1T_2.
\]

The derived Clifford relations imply `[J_i,J_j]=i epsilon_ijk J_k`,
`[J_i,T_j]=i epsilon_ijk T_k`, `[J_i,M]=0`, and
`sum_i J_i^2=(3/4)I`. Hence the selected orbital response contains two
spin-half doublets, with a full `2pi` rotation equal to `-I`. The spin labels
were not part of the admissible frame or the sea objective.

`sea_selection.py` varies the orbital matrices by projecting the existing
`RotationBand.couplings()` onto their four scalar coefficients, then substitutes
the alternative frame. It reconstructs and diagonalizes the real local-link
Hamiltonian, checking against the coefficient formula. The selected frame is
also passed to the unchanged `RotationBand` consumer: actual positive-band
spinors, SU(2) generator commutators, Casimir, full turn, vector rotation and the
finite-link current Ward identity are evaluated. This retains the same spinor,
charge vertex and contact-term entry point used by the Compton construction,
without introducing a second scattering implementation. This slice does not
execute a Compton experiment; parent integration owns that joined consumer.

## 6. Reproducible executable experiment

From the repository root:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/sea_selection.py
```

Adverse momentum selection, without falsely claiming full cubic identifiability:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/sea_selection.py --lattice-size 2
```

The upgraded local mass law uses the same theorem and consumer:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/sea_selection.py --improved
```

The JSON receipt contains all input parameters/seed, actual occupied energies
and lower bounds for the selected frame and eight unrestricted random
alternatives, pointwise excesses, saturation/Clifford defects, Gram/trace and
Hermiticity errors, coupling/budget identities, finite-set singular values,
the exact ten-point determinant comparison, smooth-path curvature comparisons
and the unchanged spinor/current/SU(2) consumer residuals. It exits nonzero if
its numerical acceptance assertions fail. It deliberately does not require a
rank-ten result for a caller-selected rank-poor cubic grid; the independent rich
and exact ten-point witnesses retain their separate labels.

The parent executed the improved-law CLI naturally to exit0. The actual L3
grid has quadratic rank10; eight unrestricted alternatives have positive
excesses from `36.9448` to `135.2917` above the selected sea bound. The analytic
restoring curvature is `506.44881646`; at theta `.003` the measured curvature
is `506.44802642`. The same-link current Ward residual is `2.9911e-16`.
Executed source: `1d8e44382e240f3ef496521ffe03028a7c5732dd32e7039f8cc3cea6c5595bd2`.

The constructive gain is specific and positive: a weaker matrix second-moment
budget plus genuine occupied-energy minimization now selects the full
four-orbital Clifford/spinor algebra and supplies the existing Compton consumer.
Deriving the budget and the spatially uniform sea saddle from the coupled
bosonic dynamics remains the next material invention target, alongside photon
matching in the parent campaign; this subtheorem does not end that campaign.

The subsequent [soft elastic theorem](soft_budget.md) removes the hard HS
budget: a quartic Gram potential plus the actual global half-filled sea
selects metric Clifford structure and cubic isotropy over completely
unconstrained traceless orbital matrices.

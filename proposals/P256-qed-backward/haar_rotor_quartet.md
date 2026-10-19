# Actual continuous-Haar scalar-rotor quartet

## Scope, source contract, and scientific decision

This is a **global spectral certificate**, not a finite Peter–Weyl extrapolation and not a relabelling of four trial vectors as eigenstates. It continues the immutable `haar_centre_rotor.py` / `.md` construction, whose source SHA256 is checked by the executable. The same default selected, unimproved frame has `m=.2, r=.12, v=.3, stiffness=2`, and comes from `uniform_data(band,3,stiffness,0,matrix_unit_basis())`. No new coupling is introduced.

The declared law remains

\[
 H=\kappa(-\Delta_{SU(4)})+JW(U),\qquad
 W(U)=\sum_{\mu=0}^3\|[\Phi_\mu,U]\|_{HS}^2,
 \quad \kappa,J>0,
\]

on probability-Haar `L²(SU(4))`, with `Tr(Ta Tb)=delta_ab/2`, `ds²=2Tr(dX²)`, and **one** scalar Casimir. Either left or right generators give that Casimir; they are not added together.

The smooth nonnegative potential on the compact group gives a self-adjoint Friedrichs operator with compact resolvent and form domain `H¹(SU(4))`. Its eigenvalues, counted with multiplicity, and the min–max spectral projectors used below are therefore well-defined.

The quantum law is declared, not derived from Euler motion. It is **not** the existing matter-coupled angular photon Hamiltonian: endpoint matter generators, Gauss sections and angular cross terms remain absent. The result below therefore cannot simply replace that Hamiltonian in the joined physical consumer.

The new result is an isolated rank-four projector of the **full continuous scalar rotor**, an explicit lower bound for its gap to the fifth eigenvalue, and a quantitative comparison of its actual fundamental-multiplication compression with a centre clock. Small-cutoff Ritz probes play no role. The proof does not assert four equal energies, compute exponential tunnelling amplitudes, establish a spatial deconfined phase, or establish mechanical half-spin.

## Actual anisotropic quadratic and arithmetic witnesses

Let `X=x_a T_a` in canonical coordinates. The immutable supplier proves

\[
 q(X)=x^TMx,\quad
 M_{ab}=\sum_\mu\operatorname{Re}\operatorname{Tr}
 ([\Phi_\mu,T_a]^\dagger[\Phi_\mu,T_b]),
 \qquad (1-r^2/12)q(X)\le W(e^{iX})\le q(X)
\]

for `r=|x|<=1`. Its fifteen curvatures are those of the actual selected quadratic, not fifteen copies of the softest curvature. The ground energy and first oscillator gap are

\[
 E_0=\sqrt{\kappa J}\sum_a\sqrt{\mu_a},
 \qquad G=2\sqrt{\kappa J\mu_{\min}}.
\]

The default unrounded receipt of the predecessor has `E0/sqrt(kappa J)=11.0658294546` and `G/sqrt(kappa J)=.67497465246`. These rounded numbers are explanatory only; the new executable reconstructs the actual matrix.

`verified_quadratic` first subtracts each concrete generator's scalar trace **exactly**: `T0=T-Tr(T)I/4`. A rounded last diagonal generator can have a tiny scalar component; this subtraction restores the actual traceless Lie algebra and changes no commutator. It then treats the eigensolver's fifteen real eigenvalues `d` and its eigenvector matrix `V` as **witnesses**. Exact `Fraction` arithmetic on the dyadic input frame and traceless generators computes

* `R=M-V diag(d) V^t`, with row-sum norm bound `r_R`;
* `eta=||V^t V-I||`, bounded by its exact row-sum norm;
* `gamma=||G_T-I||`, where `G_T=2Tr(T0_a T0_b)` is the concrete traceless generator Gram matrix.

The canonical concrete generators are `T0 G_T^(-1/2)`. The corresponding curvature is `Mbar=G_T^(-1/2) M G_T^(-1/2)`. Let `Q=V(V^tV)^(-1/2)`, an **exact** orthogonal matrix. For `eta,gamma<.1`,

\[
 \|\overline M-Q\operatorname{diag}(d)Q^t\|
 \le e:=r_R+3\max_a|d_a|\,\eta
       +\|M\|_{row}\frac{\gamma}{1-\gamma}.
\]

Indeed `||Mbar-M|| <= ||M|| gamma/(1-gamma)`. Write `V=Q (V^tV)^(1/2)`; the polar correction to `V diag(d) V^t` is at most `3 max|d| eta` in this range. Both statements follow from the eigenvalue interval `[1-eta,1+eta]` for the Gram matrix and submultiplicativity. Consequently

\[
 Q\operatorname{diag}(\mu^-_a)Q^t\le\overline M
 \le Q\operatorname{diag}(\mu^+_a)Q^t,
 \qquad \mu^-_a=d_a-e>0,\quad\mu^+_a=d_a+e.
\]

The JSON records exact rational residual bounds, outward printed endpoints, and the witness `V`. Define `M_-` and `M_+` by these last two matrices. This tiny enclosure is an arithmetic safeguard, **not** a changed microscopic potential. Trial Gaussians use the upper quadratic and the codimension constraints use the lower quadratic. Both retain all fifteen actual anisotropies. The scalar display includes a conservative `1e-10` relative outward guard; this is an analytic/numerical certificate, not a formal proof-assistant artifact.

Global coercivity follows from the same commutator superoperator as the predecessor, now with the rational quadratic enclosure above. For exactly Hermitian concrete fields, the superoperator has the **real** matrix `2 Mbar` in the HS-orthonormal Hermitian traceless basis `sqrt(2) Tbar`; its complexification has the same spectrum. Thus `ell_min=2 min(mu-)` is a certified lower bound on its full complex traceless spectrum, without treating the rounded Clifford eigenvalues as exact. The executable verifies exact Hermiticity, and the predecessor's centre-distance proof gives

\[
 W\ge\ell_{\min}Q(U)\ge\frac{\ell_{\min}}{C}d_Z(U)^2,
 \quad C=2+\pi^2/2,
 \quad d_Z(U)=\min_{z\in\mathbb Z_4}\|U-zI\|_{HS}.
\]

The printed `ell_min=2 min(mu-)` is additionally guarded downward. The proof uses this pointwise bound, not sampled coercivity; the predecessor's weighted-Clifford formula remains a corroborating actual-frame control.

## Haar measure, support, and exact normalization

Choose `0<a<b<=1`. Balls about the four centres are disjoint: the shortest distance between distinct centres is `pi sqrt(6)>2`. In a centre chart `U=z exp(iX)`, the radial geodesic distance is `r=|x|`. The predecessor supplies

\[
 j(X)=\prod_{j<k}\operatorname{sinc}^2((\theta_j-\theta_k)/2),
 \quad s^{12}\le j\le1,
 \quad I\le g^{-1}\le s^{-2}I,
 \quad s=\operatorname{sinc}(b/2).
\]

For a convenient purely polynomial lower bound set

\[
 s_*:=1-b^2/24\le s,\quad j_*:=s_*^{12},\quad m_*:=s_*^{-2},
 \quad c:=1-b^2/12.
\]

Thus `j_*<=j<=1`, `I<=g^-1<=m_*I`, and `W>=c x^T M_- x`. Radial gradients have exact metric norm equal to their radial derivative, by the Gauss lemma.

Probability-Haar is `dmu_H=c_H j(X) dx`, where

\[
 c_H=\operatorname{Vol}_{ds^2=2Tr}^{-1}(SU(4)),\qquad
 \operatorname{Vol}(SU(4))=2^{15/2}(2\pi)^9/6.
\]

One direct check is the fibration `SU(n-1)->SU(n)->S^(2n-1)`. For the Hilbert–Schmidt metric, the `2n-2` horizontal non-phase directions stretch the unit-sphere metric by `sqrt(2)`, and the remaining phase direction by `sqrt(n/(n-1))`. Multiplying `2^(n-1) sqrt(n/(n-1))` by the unit-sphere volume `2 pi^n/(n-1)!`, for `n=2,3,4`, gives `(2pi)^9/6`. Changing from `Tr` to `2Tr` multiplies the volume by `2^(15/2)`. All spectral and transport quotients below cancel **the same** `c_H`; it is never mistaken for one.

Let

\[
 A_+=\sqrt{J/\kappa}\,Q\operatorname{diag}(\sqrt{\mu_a^+})Q^t,
 \quad g_+(x)=\left(\frac{\det A_+}{\pi^{15}}\right)^{1/4}
                    e^{-x^TA_+x/2}.
\]

This Gaussian has exact unweighted norm one and is the ground state of `-kappa Delta + J x^T M_+ x`. Its squared density is the actual fifteen-dimensional anisotropic Gaussian, not a radial trial packet.

Set `theta(r)=0` on `[0,a]`, `theta(r)=pi(r-a)/(2(b-a))` on `[a,b]`, and `theta(r)=pi/2` thereafter. The centre cutoff is `chi=cos(theta)` inside its chart and zero outside. Its Lipschitz derivative bound is

\[
 D=\pi/[2(b-a)].
\]

Define `h_z(U)=chi(r)g_+(x)` in the chart and zero outside. This is an `H¹` function; no operator-domain claim is needed. The four Haar norms are identical by centre translation, and their supports are disjoint. Their **exact** normalized versions are

\[
 f_z=h_z/N,\quad
 N^2=c_H I,\quad I=\int\chi^2g_+^2j\,dx.
\]

For `alpha_min=min eig(A_+)` and `t=alpha_min/2`, the Gaussian moment-generating function gives

\[
 \Pr_{g_+^2}(|x|\ge a)
 \le e^{-ta^2}\prod_a(1-t/\alpha_a)^{-1/2}
 \le T:=2^{15/2}e^{-\alpha_{\min}a^2/2}.
\]

Therefore

\[
 0<c_H j_*(1-T)\le N^2\le c_H,
 \qquad j_*(1-T)\le I\le1.
\]

The executable rejects `T>=1`. It prints both the logarithmic tail bound and a positive upper bound; exponential underflow is conservatively replaced by a larger positive normal floating number, **never** by zero. This establishes normalization without cubature or sampled tail moments.

## Four Gaussian upper bounds close to the actual oscillator ground

The flat ground-state identity, valid for this Lipschitz cutoff by `H¹` approximation, is

\[
 \int\{\kappa|\nabla(\chi g_+)|^2
         +Jx^TM_+x\,\chi^2g_+^2\}\,dx
 =E_0^+\int\chi^2g_+^2dx
       +\kappa\int|\nabla\chi|^2g_+^2dx,
\]

where `E0+=sqrt(kappa J) sum sqrt(mu+)`. Using `j<=1`, `g^-1<=m_*I`, `W<=q<=x^T M_+x`, and the exact Haar normalization gives

\[
 \mathcal R_H(f_z)
 \le U:=\frac{m_*}{j_*}
       \left(E_0^++\frac{\kappa D^2T}{1-T}\right).
\]

Cross quadratic-form matrix elements between different supports vanish. Hence **every** vector in their four-dimensional span `S` has Rayleigh quotient at most `U`, and global min–max gives `lambda4<=U`. This is close to the actual anisotropic `E0`, rather than the predecessor's radial kinetic constant `136`.

## Five-function IMS partition and global fifth-eigenvalue lower bound

Use the four translated centre cutoffs `chi_z`. Since their nonzero supports are disjoint, define the outside cutoff by

\[
 \chi_o(U)=\sqrt{1-\sum_z\chi_z(U)^2}.
\]

It equals `sin(theta)` in a centre annulus, zero in an inner ball, and one outside all outer balls. Thus `sum chi_z²+chi_o²=1` globally. In an annulus the two gradients contribute exactly

\[
 |\nabla\cos\theta|^2+|\nabla\sin\theta|^2=|\theta'|^2\le D^2.
\]

The global quadratic-form IMS identity therefore subtracts at most `kappa D² ||psi||_Haar²`. This is a global partition, **not** a Neumann-ball or Dirichlet-inclusion shortcut.

On the support of `chi_o` all four geodesic distances from the centres are at least `a`. There is the following explicit chord consequence:

\[
 d_Z(U)^2\ge d_a^2:=\tfrac12 a^2(1-a^2/24)^2.
\]

To prove it, suppose a centre chord were smaller than `a sinc(a/2)/sqrt(2)<1`. All principal eigenangles then have absolute value below `pi/3`; their sum, a multiple of `2pi`, is zero. Moreover each absolute eigenangle is below `a`, since the assumed chord is below `sqrt(2) sin(a/2)<2 sin(a/2)`. Consequently `chord² >= sinc(a/2)² sum theta_j² = sinc(a/2)² r²/2`; the assumed chord would force `r<a`, a contradiction. Finally use `sinc(a/2)>=1-a²/24`. Thus the outside localized energy is at least

\[
 B:=J\ell_{\min}d_a^2/C.
\]

For a centre-localized function `v_z=chi_z psi`, extend its chart representation by zero to `R^15`. It is in `H¹(R^15)` and has support in the radius-`b` ball. Its Haar energy numerator is at least `c_H j_*` times the unweighted oscillator form with quadratic `c J M_-`. Its Haar norm squared is **at most** `c_H` times its unweighted norm squared.

Let `g_-` be the normalized unweighted ground Gaussian of that lower oscillator. Impose the four linear constraints

\[
 L_z(\psi):=\int_{\mathbb R^{15}}
       g_-(x)\chi_z(x)\psi(z e^{iX})\,dx=0,
       \qquad z=0,1,2,3.
\]

These are legitimate bounded functionals on Haar `L²`: equivalently their Haar test functions are `g_- chi_z/(c_H j)`, and `j>=j_*>0` on their support. They are independent because the four supports are disjoint and their test functions are nonzero. Therefore their common kernel has **exact codimension four**. The constraint is unweighted oscillator orthogonality; it is **not** incorrectly asserted to be Haar orthogonality.

For every constrained `v_z`, the whole-space oscillator spectral theorem gives energy at least `sqrt(c)(E0-+G-)` times its unweighted norm. The preceding two measure inequalities give the Haar lower bound `j_* sqrt(c)(E0-+G-) ||v_z||_Haar²`. Adding the outside term, subtracting the **uniform** IMS cost, and using `sum ||chi psi||²=||psi||²` proves the global bound

\[
 \boxed{\lambda_5\ge L_5:=
      \min\{j_*\sqrt c(E_0^-+G^-),B\}-\kappa D^2.}
\]

No centre-ball boundary condition is substituted for the global operator. Removing the four constraints yields the simultaneous global floor

\[
 \boxed{H\ge L_0:=\min\{j_*\sqrt c E_0^-,B\}-\kappa D^2.}
\]

Whenever `L0>0` and `U<L5`, the full continuous rotor has an isolated lowest **rank-four spectral subspace** with gap at least `L5-U`. Its four energies lie in `[L0,U]`. The displayed energy-spread bound `U-L0` is valid but is not an exponential tunnelling formula.

## Actual projector and fundamental transport, not trial-state relabelling

Let `P` be that spectral projector and `S` the orthogonal four-packet projector. Positivity of the spectral decomposition gives

\[
 H\ge L_0P+L_5(1-P).
\]

Since every unit vector of `S` has energy at most `U`,

\[
 \|(1-P)S\|^2\le d^2:=\frac{U-L_0}{L_5-L_0}<1.
\]

Both projectors have rank four, so `||P-S||<=d`. The canonical polar isometry

\[
 V=PS(SPS)^{-1/2}:\operatorname{Ran}S\longrightarrow\operatorname{Ran}P
\]

is well-defined and has distance from the inclusion at most

\[
 e_V=\sqrt{2(1-\sqrt{1-d^2})}.
\]

All four packet functions have even squared density in `x`: the Gaussian, radial cutoff, and Haar Jacobian are even. Therefore, in the localized packet basis, compression of **bundle multiplication** `U` on `L² tensor C⁴` is exactly

\[
 S U S=\operatorname{diag}(1,i,-1,-i)\otimes A,
 \quad A=\frac{\int\chi^2g_+^2j\cos X\,dx}{I},
 \qquad \|A-I_4\|\le b^2/4.
\]

The inequality follows from `||X||_op² <= Tr X² = r²/2` and `||cos X-I||<=||X||_op²/2`. An anisotropic Gaussian need not give an exactly scalar `A`; no unsupported scalar-amplitude assertion is made.

Multiplication by `U` is unitary on the full bundle. Applying the polar comparison on each side proves

\[
 \boxed{\|(V^*\otimes I)PUP(V\otimes I)
       -\operatorname{diag}(1,i,-1,-i)\otimes I_4\|
       \le e_U:=2e_V+b^2/4.}
\]

This bounds the **actual spectral** fundamental compression. When `e_U<1`, its smallest singular value is at least `1-e_U`, and full-bundle leakage squared out of the spectral quartet is at most `2e_U-e_U²`. These are bundle operator bounds, not a claim that sixteen individual entries of `U` are unitary.

The centre action is `C psi(U)=psi(iU)`. With localized packets `f_z`, define `f_q=(1/2) sum_z i^(qz) f_z`. Then `C f_q=i^q f_q` and multiplication by `U` maps `q` to `q+1 mod4`. The Hamiltonian, `P`, `S`, and the polar map commute with this action. Since the trial representation has one vector in each charge, the actual quartet has **exactly one state in each centre-charge sector**. In particular `V f_q` are genuine normalized eigenstates: each actual spectral charge block is one-dimensional and invariant under `H`. They are not the trial vectors `f_z`, and their energies may differ. The clock becomes the cyclic charge shift with the same convention. Conjugating the frame and transporting the entire oscillator witness/basis by the same adjoint action conjugates the colour factor; independently rerounding a degenerate eigensolver basis is not asserted to be an exactly covariant numerical gauge choice. None of these operations manufactures a spatial rotation minus sign.

## Computable regimes, rejected boundaries, and recovery

Run with the project scientific Python:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/haar_rotor_quartet.py \
  --output proposals/P256-qed-backward/haar_rotor_quartet_receipt.json
```

The final executable is parent-owned. Default declared finite strengths are `J=1e10,1e18,1e26`, `kappa=1`. Chart radii vary with `J/kappa` without changing the bare Hamiltonian:

\[
 b=\max\{45(\kappa/J)^{1/4},\ 1.155(\kappa/J)^{1/8}\},
 \qquad a=2b/3.
\]

The first term ensures a large normalized Gaussian core; the second balances chart comparison error against IMS cost. This materially changes both the failed radial packet and a fixed-width oscillator partition: a transition width tied forever to the oscillator scale would retain a nonvanishing relative IMS cost. The slower shrinking chart makes both errors small as `J/kappa` grows.

A scoped calculation of the actual frame and exact rational curvature enclosure, **not execution evidence for the final CLI**, found positive gaps `40432.6256`, `673907652.280`, and `6749639802054.64` at the three default strengths. The projector errors were `.6279856`, `.0394713`, and `.00394750`; actual bundle-clock errors were `1.33705`, `.0789686`, and `.00789512`. These are mathematical bound values, not sampled spectral measurements. The final JSON recomputes them and is the source-bound numerical receipt. The weak first regime isolates the quartet but does not give a subunit bundle transport bound; the stronger regimes do. At least one declared regime must give subunit actual transport for the CLI to accept.

The accept boundary rejects nonpositive/nonfinite kinetic or alignment coefficients, invalid or overlapping charts, zero annulus width, a Gaussian tail bound without positive normalization, nonpositive curvature enclosures, and failure of the global spectral separation/polar-map inequalities. Explicit radii may be supplied as a pair; one radius alone is rejected. Automatic radius selection never hides a failed inequality. Embedded adverse cases include zero kinetic coefficient, negative/infinite alignment, outer radius above one, zero transition width, a missing radius partner, a valid but exceedingly thin annulus whose IMS cost destroys the gap, the weak law, and an inadequately small Gaussian core. A final accepted strong-law calculation demonstrates **positive recovery** after the rejected cases. It must certify both a positive full-rotor gap and subunit actual transport error.

The JSON includes full selected frame, parameters, exact-rational witness residuals, all Haar factors, tail and normalization bounds, the five-function IMS costs, global spectral endpoints, projector/polar/transport bounds, rejection reasons, positive recovery, immutable supplier and own proof/source hashes, and runtime versions. No finite Peter–Weyl dimension is presented as a continuum certificate.

## Obligations retained after this theorem

1. Resolve tunnelling splittings and amplitudes beyond the coarse energy-spread upper bound.
2. Transfer the construction to the **actual matter-coupled angular photon/Gauss Hamiltonian**, retaining endpoint generators and cross terms; it is not this scalar Casimir model.
3. Supply covariant endpoint frames and spatial Gauss-invariant propagation, and prove an operational spatial deconfinement statement rather than a single-link spectrum.
4. Solve/control the unrestricted full-field quantum phase and the joined spacetime limit.
5. Establish an operational mechanical rotation/interference experiment. Pure gauge and global microscopic spatial `2pi` have `+1`; an old fixed-reference conditional minus sign is not a mechanical half-spin certificate.

No fundamental quantum origin, Dirac electron, full-field solution, deconfined spatial Z4 phase, mechanical half-spin, or canonical/full-QED claim follows from the certified scalar-rotor quartet.

## Parent continuous-Haar receipt and outward geometry

The parent executed the actual default CLI into
`haar_rotor_quartet_verified_receipt.json`, naturally exiting zero. Source
`haar_rotor_quartet.py` was
`228b8d98df441c84b8880eca521577140055486da0b93944c602c8487cd8a5bc`.
At `kappa=1`, the three default strengths give quartet gap lower bounds
`40432.6255568`, `673907652.280`, and `6749639802054.266`. The actual
bundle-clock error bounds are `1.33705360689` (not controlled),
`.0789685830838`, and `.00789511567156`, respectively. These are
analytic bound evaluations for the continuum scalar law, not measured
eigenvalues of a matter-coupled Hamiltonian.

An adverse finite-ratio invocation with `kappa=1e-150,J=1e160` also naturally
exited zero into `haar_rotor_quartet_extreme_ratio_receipt.json`. Computing
Gaussian precision without first forming `J/kappa` keeps the representable
precision `3.3748732622871807e154` finite. Exact dyadic polynomial/power
evaluation followed by outward `nextafter` conversion prevents a tiny nonzero
chart correction from becoming a falsely certified lower endpoint of one.
That receipt has Jacobian lower `.9999999999999986`, inverse-metric upper
`1.0000000000000004`, potential factor lower `.9999999999999999`, and
normalized local Haar integral lower `.9999999999999983`. Its gap lower is
`67497.4650176675`, and transport error upper is `.00011452533830215401`.
The packet fraction is checked positive before division; the normalized
lower bound also uses exact multiplication and downward conversion.

Scoped re-review accepted these changed arithmetic/strict-JSON paths and the
parent-produced receipts. It performed no reviewer execution and does not
change the theorem's scalar-law scope. In the physical Gauss domain the four
translated charts are one residual gauge orbit, not four physical vacua.


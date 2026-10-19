# Whole-heavy sea and the actual four-centre leading potential

Companion: `heavy_sea_centre_potential.py`. This adds two new files and imports the
existing full-band/centre/photon providers without changing them. The result is
about the **frozen selected frame, leading potential**, not an isolated quantum
quartet, scalar-Casimir replacement, deconfined phase, or physical half-spin.

## Actual law and the first discriminator

The bare coefficients remain `m=.2, r=.12, v=.3, stiffness=2, Jpref=.2`, with
unimproved `RotationBand(a=1)`. The frame is the existing
`uniform_data(size=3, preferred=0)` selected frame:

\[
 \Phi_0=b\beta,\qquad \Phi_j=s\alpha_j,\qquad
 b=0.6483983251503319,\quad s=0.23863957694084134.
\]

The four Hermitian generators obey the weighted Clifford algebra. Each species
has four occupied modes out of eight, on the actual two-cell 4900-state Pauli
space with `N_H=N_L=4`. The heavy Hamiltonian is the **negative of the whole**
heavy one-body lift; its occupied plane is the top four eigenvectors of

\[
 h(U)=\begin{pmatrix}a\beta&T(U)\\T(U)^\dagger&a\beta\end{pmatrix},\quad
 T(U)=\frac{FU+UF}{2},\quad F=-d\beta-i e\alpha_1,
\]

where

\[
 a=(m+r)b,\quad d=rb/2,\quad e=vs/2.
\]

These endpoint coefficients and the `m+r` onsite are extracted from, and checked
against, the immutable full-band provider. Both Wilson and spatial heavy hopping
scale with `Lambda`, just like the heavy onsite. The leading potential is

\[
 V(U)=\frac{J_{\rm pref}}8W(U)-S(U),\qquad
 S(U)=\sum_{k=5}^{8}\lambda_k(h(U)),\qquad
 W(U)=\sum_{\mu=0}^3\|[\Phi_\mu,U]\|_{HS}^2.
\]

It is not permissible to drop `S(U)` before selecting a low-energy link sector.
The actual scalar-freezing discriminator finds zero centre gradient and all 15
positive eigenvalues of the **coupled** Hessian, with extremal values
`0.005694884768250337` and `0.062169517281073235`. These are calculated using
actual spectral-projector derivatives, not a scalar alignment Hessian alone.
Generic SU4 samples are secondary adverse observations; the proof below does
not use their success or optimization to claim a global minimum.

## Exact centre sum and a uniform actual gap

Write the site Pauli matrices as `tau`. At `U=t I`, `0<=t<=1`,

\[
 h_t=a\beta-t d\beta\tau_x+t e\alpha_1\tau_y.
\]

`beta tau_x` commutes with both the mass and spatial block. The mass and spatial
block anticommute with each other. Hence, with `R_t=sqrt(a^2+t^2 e^2)`, the
positive eigenvalues are `R_t+td` and `R_t-td`, each twice, and

\[
 S(tI)=4R_t,\qquad S(I)=4R_1=0.8422103947111602.
\]

The Wilson hopping is included: it splits the occupied energies and cancels only
in their sum at this reference frame. It is not zero or omitted. The exact
centre potential is `-4 R_1`.

For every unitary `U`, `||T(U)||op <= ||F||op = d+e`. Weyl comparison with the
onsite matrix gives four positive and four negative eigenvalues and

\[
 g_{\rm occ-empty}\ge 2(a-d-e)>0.
\]

The source certifies this inequality with rational amplitude intervals, giving
`g > .2655752558`. Thus the occupied projector and its derivative are smooth
on all SU4 links at this fixed frame. This does **not** prove that the existing
local-epsilon chart has positive overlap for arbitrary unrestricted fields.
For the scalar reference, independently,

\[
 |h_t|\ge(R_t-td)I\ge(a-d)I=\gamma I,\qquad
 \gamma=0.16858356453908635.
\]

The reference bound is stronger than the bound on the actual arbitrary-link
gap; the proof never substitutes one domain for the other.

## Global heavy-sea deviation bound

The spectrum is invariant under multiplying `U` by any scalar phase, by a
unitary acting on one site. Remove the phase of `Tr U` and write

\[
 U'=tI+X,\quad t=|\operatorname{Tr}U|/4,\quad \operatorname{Tr}X=0,
 \quad \|X\|_{HS}^2=4(1-t^2)=Q(U).
\]

If `Tr U=0`, use any phase. Also `W(U')=W(U)=W(X)`.
Let `L(X)=(FX+XF)/2` and
`D=|h_t|`. For every Hermitian matrix `h` and positive matrix `D`,

\[
 \operatorname{Tr}|h|\le\tfrac12\operatorname{Tr}(D+hD^{-1}h).
\]

For completeness this follows from the nonnegative squared Hilbert-Schmidt
norm of `D^(1/2)-sign(h) h D^(-1/2)`. Here `Tr h=0` and the gap fixes the positive
rank to four, so `S(h)=Tr|h|/2`. Apply the inequality to
`h(U')=h_t+Delta h`, where `Delta h` has offsite block `L(X)`. The constant
term is exactly `S(tI)`.

The linear term vanishes. In fact

\[
 \operatorname{sign}(h_t)=\frac{a\beta+t e\alpha_1\tau_y}{R_t},
\]

independent of the Wilson splitting, and its contraction with `Delta h` is a
scalar multiple of `Re Tr X=0`. The quadratic term is bounded by
`||L(X)||HS^2/(2 gamma)`. Finally,

\[
 S(I)-S(tI)=\frac{4e^2(1-t^2)}{R_1+R_t}
 \ge\frac{e^2}{2R_1}\|X\|_{HS}^2.
\]

Consequently, **for every SU4 link**, not just a chart or sampled points,

\[
 S(U)-S(I)\le\frac{\|L(X)\|_{HS}^2}{2\gamma}
                  -\ell\|X\|_{HS}^2,
 \qquad \ell=\frac{e^2}{2R_1}.
\]

This does not assert the stronger observed proposition `S(U)<=S(I)` everywhere.
That stronger claim is unnecessary and is not certified here.

## Explicit 30-real-dimensional certificate

Take the 15 traceless Clifford monomials

\[
 B_M=\frac{i^{k(k-1)/2}}2\prod_{\mu\in M}\Gamma_\mu,
 \quad\Gamma=(\beta,\alpha_1,\alpha_2,\alpha_3),\quad k=|M|,
\]

in increasing product order. They are Hermitian and orthonormal in the complex
Hilbert-Schmidt inner product. Write `X=sum_M (x_M+i y_M) B_M`.
For a monomial let `anti_mu` mean it anticommutes with `Gamma_mu`, and set
`chi_mu=1-anti_mu`. Its alignment eigenvalue is

\[
 q_M=4\left(b^2\,\mathrm{anti}_0+
                   s^2\sum_{j=1}^3\mathrm{anti}_j\right).
\]

The real symmetric matrix of

\[
 \mathcal A(X)=\frac{J_{\rm pref}}8 W(X)
              -\frac{\|L(X)\|_{HS}^2}{2\gamma}
              +\ell\|X\|_{HS}^2
\]

has repeated real/imaginary diagonal entries

\[
 A_{MM}=\frac{J_{\rm pref}}8q_M+\ell
              -\frac{d^2\chi_0+e^2\chi_1}{2\gamma}.
\]

There is at most one off-diagonal entry in each row. It is present only when
`chi_0 != chi_1`, connects the real component of monomial `M` with the imaginary
component of `M symmetric_difference {0,1}` (or conversely), and has magnitude
`de/(2 gamma)`. Signs depend on the monomial phases and do not affect the row
radius. This follows directly by multiplying `beta B_M` and `alpha_1 B_M`;
monomials commuting with both do not couple to their partner, which
anticommutes with both and has `L=0`.

The program uses **exact rational arithmetic** for outward scalar bounds over

- `b in [0.6483983251, 0.6483983253]`,
- `s in [0.2386395768, 0.2386395770]`.

The only square root is replaced by a rational upper bound and that bound is
checked by exact squaring. Every diagonal is bounded below, and every row
radius above, throughout this box. All 30 row margins are strictly positive;
the minimum is greater than `0.000118`. The receipt gives each exact rational
margin. This is a strictly diagonally dominant matrix certificate, **not a
floating eigensolver promoted to a proof**. Numerical floating eigenvalues may
be used as diagnostics, not premises.

Combining it with the preceding global deviation bound proves the conservative
all-link inequality

\[
 \boxed{V(U)-V(I)\ge 0.000118\,Q(U).}
\]

`Q(U)=0` for a unitary matrix if and only if it is scalar. Its SU4 scalar points
are exactly `I, iI, -I, -iI`. Thus the actual whole-heavy leading potential has
**exactly four global centre minima**, with no noncentre alternative repair
needed at these coefficients. For the centre chord distance `d_Z4`, the
existing bound `d_Z4^2 <= (2+pi^2/2) Q` gives coercivity in distance as well.
The source also reports a valid, weaker positive coefficient multiplying `W`
using its global upper superoperator bound. None of these inequalities proves
an isolated four-dimensional dynamical band.

## Actual projector geometry and primitive consumer

For the occupied top-four projector `P`, with occupied/empty eigenvectors
`v_o,v_e`, the source uses the immutable provider's derivative

\[
 dP=v_e\left[\frac{v_e^\dagger(dh)v_o}{\lambda_o-\lambda_e}\right]v_o^\dagger
      +\mathrm{h.c.}
\]

The exact frozen-frame derivatives are

\[
 dS=\operatorname{Tr}(P\,dh),\qquad
 d^2S=\operatorname{Tr}(P\,d^2h+dP\,dh),\qquad
 G_{ab}^{H}=\operatorname{Re}\operatorname{Tr}(P\,dP_a dP_b).
\]

All 15 link directions are included in the full mixed Hessian. The source also
computes actual derivatives of the 4900-component paired state and removes its
ray-parallel component. At equal species coefficients and zero compact phases,
the paired metric equals `2 G^H`; this is directly exercised, not assumed from
a copied metric. Actual sparse Pauli onsite-plus-hopping eigenstate residuals
and full occupied sums are checked for both species.

For `X=sum x_a T_a`, `Tr(T_a T_b)=delta_ab/2`, the uniform projector bounds are

\[
 \|dh\|_{HS}\le(d+e)|x|,\quad
 \|dP\|_{HS}\le\frac{d+e}{g}|x|,\quad
 G^H\le\frac{(d+e)^2}{2g^2}I,
 \quad G^{H+L}\le\frac{(d+e)^2}{g^2}I.
\]

The source uses the rational amplitude-box bounds, not a sampled minimum gap.
They give approximately `G^H <= .039558 I` and `G^(H+L) <= .079116 I`.
These are fixed-frame ray-metric bounds; no arbitrary-field epsilon-chart or
full Cartesian-field bound is inferred.

## Centre covariance, stabilizer, and physical limits

For `z in Z4`, define `D_z=diag(z I4,I4)`. Both species satisfy
`h(zU)=D_z h(U) D_z^dagger` and the same covariance for their occupied projector.
On the actual Pauli space the corresponding map is

\[
 M_z=z^{N_{H,\mathrm{left}}+N_{L,\mathrm{left}}}.
\]

The local paired initial section has four total particles on the left, so its
centre phase is `z^4=1`; the normalized projected section obeys
`psi(zU)=M_z psi(U)`. The source exercises all four maps at the centre and at a
generic link. Covariance is about this full matter state, not a bare scalar
rotor in isolation.

The four weighted Clifford matrices generate `M4(C)`, so their common unitary
commutant is scalar, and their SU4 stabilizer is exactly Z4. That is a
mathematical frame statement. Isotropic spatial weights support the conditional
combined spatial/flavour covariance already declared by the providers. An
individual frozen x bond is not invariant under arbitrary rotations unless the
bond direction and vector fields are rotated too. No centre multiplication is
identified with mechanical spatial rotation, and no microscopic `2pi` operation
or cubic-lift sign is inferred from this potential certificate.

The next integration obligation is the **actual matter-angular Gauss kinetic
operator**, including its Born/Berry terms, and the unrestricted Cartesian
field kinetic operator. Their join is not implicitly a scalar Casimir. The
leading potential alone does not solve coupled quantum eigenstates, tunnelling,
fifth-level separation, finite-light/photon corrections, a spatial Higgs or Z4
phase, deconfinement, an electron lower mass gap, or canonical/full QED.

## Portable execution and source binding

From this directory, with a Python environment containing NumPy and SciPy:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python heavy_sea_centre_potential.py --output heavy_sea_centre_potential_receipt.json
```

The receipt includes source SHA256 identities of both new files and all
imported provider sources, bare coefficients, selected frame amplitudes and
stationarity diagnostic, exact rational row margins, every centre Hessian
entry/eigenvalue, 4900-state energies/centre maps/Born metrics, finite-difference
controls, generic SU4 observations, and uniform derivative bounds. A failure of
the rational matrix certificate aborts a centre-well claim rather than assuming
Z4 or substituting alignment alone. The parent owns the final CLI receipt.

Worker scoped smoke evidence (not a parent full CLI): importing
`HeavySeaCentrePotential()` and directly exercising its selected-centre primitive
checks plus a noncentral `expm` link naturally exited zero. Centre heavy/light
whole-energy residuals were `7.92e-16`/`7.45e-16`, centre maps below `1.02e-15`;
projector derivative finite differences at steps `.0002/.0001` had errors
`7.22e-11/2.62e-11`. The noncentral potential was `-.8283976866493648`, above
`V(I)=-.8422103947111602`, with actual gap `.3362436910906025`. These observations
exercise the matrix and primitive consumers; the all-link conclusion comes
from the explicit rational certificate instead.

A second direct-API smoke used 64 genuine QR-generated SU4 links, seed `222`.
The smallest sampled potential was `-.7763234947893872`; the smallest observed
occupied/empty gap was `.3262739607361944`. The exact rational Gershgorin lower
margin was `.00011849707112072049`, and the amplitude-box actual gap lower was
`.265575255928`. The lowest sampled link's actual 4900-state projection weight
was `.9766385514416777`, whole heavy/light eigenstate residuals were
`9.41e-16/9.12e-16`, and all four primitive centre-map errors were below
`1.24e-15`. These finite observations are not the global proof.

## Actual parent whole-heavy consumer

The parent subsequently ran the actual CLI into
`heavy_sea_centre_potential_receipt.json`, naturally exiting zero. Its source
was `de1e32fb43f9be142d227d9073b44dc02cc15bb20bff8f0606c58d699cbccf3a`.
A scoped source/proof/receipt review accepted the rational all-link
comparison, actual whole-heavy occupation, primitive checks and the
explicit fixed-frame derivative bounds. The global margin is
`.00011849707112072049` and occupied/empty gap lower `.265575255928`;
neither is replaced by a sampled link minimum or alignment-only sine bound.

These wells belong to this fixed selected-frame actual potential. They do
not prove that the frame minimizes unrestricted Cartesian fields, and the
centre charts are one orbit in the physical Gauss quotient. The separate
`gauss_heavy_bo.py` consumer uses this potential in an endpoint/B-inclusive,
heavy-only physical compression retaining the entire light/photon fibre;
it does not transfer a scalar four-vacuum or finite photon ladder claim.


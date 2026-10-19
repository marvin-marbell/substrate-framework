# Original full-SU4 centre packet and external-charge construction

This is a scoped new supplier in the active issue222 invention campaign, not
completion of that campaign. Assignment plan
`issue-222-continuing-backward-20260930` revision
`e1874737294f261dfc305be3f8d054bd1223a1a3142bdf1a70c6d6cc136b76a4`, general
pin `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`.
The parent subsequently recorded the slow-field alternative in revision
`1a06b52727e8540a5ab588172f751a794625aa466a0a06cfa35fcbc28fc42464` without
changing pinned requirements. The receipt keeps the historical assignment
revision, not an invented current-plan claim.

**Results:** actual native weighted-Clifford link minima; globally smooth
full-SU4 packets in all four centre characters; exact Haar norm and all-15
Casimir energy formulas; a common analytic packet upper bound; a rigorous
scalar one-link fundamental/antifundamental ground splitting bound; and an
exact *spectator external colour-pair* variational transfer on the entire
original non-scalar full-Gauss Hamiltonian. No original-light particle,
thermodynamic deconfinement, numerical many-body ground, spin embedding or
photon pole is asserted.

## 1. Actual law and the exact four minima

The frame is not imported from a conditional spin construction. At each
fixed positive spacing `a`, execute
`bulk_flat_holonomy_bound.selected_gram` with the actual original itinerant
source `RotationBand(**BARE,a=a)`, stiffness `2/a`, and its reference grid.
Set `Phi_mu=sqrt(G_mu,mu) SELECTED_mu`. This is a finite-grid approximation
to that source's full-rank continuum Clifford frame, not a numerical proof
that the exact full quantum ground equals a frozen frame. No local 27-flavour
CAR selector energy is evaluated. The full heavy coefficient remains Lambda
and whole occupation remains `NH=NL=2V`.

For the actual link with equal endpoint frames,

\[
 W(U)=\sum_{\mu=0}^3\operatorname{Tr}
       (\Phi_\mu-U\Phi_\mu U^\dagger)^2
     =\sum_\mu\|[\Phi_\mu,U]\|_{HS}^2.                 \tag{1}
\]

More generally use the covariant endpoint expression with Phi at L and R.
Under independent `hL,hR in SU4`, transform the fields by conjugation and
`U -> hL U hR^dagger`. Each difference in (1) is conjugated by hL, so W is
exactly invariant. For positive full-rank Clifford Gram the complex Cl4
algebra generates all four-by-four matrices. Hence W=0 iff U is scalar;
unitarity and determinant one give **exactly I,iI,-I,-iI**. Their Hessians
have all 15 restoring colour directions, not a continuous stabilizer.

Use actual native generators `T_a=matrix_unit_basis()[a]/sqrt(8)`,
`Tr(T_a T_b)=delta_ab/2`, metric `ds^2=2Tr(dX^2)`, and
`D_a f(U)=d/dt f(exp(i t T_a)U)|_0`. Then
`C_SU4=-sum_a D_a^2`, and `sum_a T_a^2=(15/8)I`; **there is no additional
right Casimir**. Declare the candidate bare trajectory

\[
 u_4=.3\Lambda^{-1/2},\qquad
 v=(J/8)\Lambda^{5/4},\quad J=.2,
 \qquad H_{\rm link}=u_4 C_{SU4}+vW.                    \tag{2}
\]

This differs from the original fixed-u4 point. Alignment and colour
plaquette (`K4=.7`) remain positive with power 5/4. The single-link normal
calculation below freezes endpoint fields and does not silently include the
3D plaquette Hessian or the full angular Born-Huang operator.

The parent also investigates a **separately changed** field kinetic
`Lambda^(-1/2) p_Phi^2/2` instead of original
`sqrtLambda p_Phi^2/2`. The exact external-pair identity in section 5 works
for either positive field coefficient. This does not prove that either
trajectory has a mobile-light low-energy pole.

### Actual global and local potential inequalities

Define `L(A)=sum_mu [Phi_mu,[Phi_mu,A]]`, on all 16 complex HS matrix
coordinates. Its kernel is the scalar direction. Write its positive
extremal eigenvalues as ell_min and ell_max. If
`Phi_mu^2=g_mu I` and distinct Phi anticommute, the Clifford product with
index mask S and grade r has eigenvalue
`4 sum_{mu: (mu in S) iff (r even)} g_mu`. This derives the complete
spectrum and is checked against the native matrix superoperator.

The scalar part of U is `(TrU/4)I`, giving

\[
 \ell_{min}(4-|\operatorname{Tr}U|^2/4)
 \le W(U)\le
 \ell_{max}(4-|\operatorname{Tr}U|^2/4)\le4\ell_{max}. \tag{3}
\]

For `U=exp(i x_a T_a)`, the quadratic matrix is
`K_ab=sum_mu ReTr([iT_a,Phi_mu]^dagger [iT_b,Phi_mu])`.
Its eigenvalues are the 15 positive L eigenvalues divided by two. Thus the
frozen Euclidean normal oscillator ground energy and first spacing are
`sum_a sqrt(u4 v mu_a)` and `2 sqrt(u4 v min(mu_a))`.
Their scaling is Lambda^(3/8). These are **only** the frozen single-link
quadratic quantities, not the complete mixed field/link Gauss spectrum.

## 2. A smooth packet on the whole group, not a Z4 replacement

Put `z_k=i^k` and, for eta>0,

\[
 f_k(U)=\exp\{\eta[\operatorname{ReTr}(z_k^{-1}U)-4]\},
 \qquad
 \psi_q(U)=\sum_{k=0}^3 z_k^q f_k(U),\quad q=0,1,2,3. \tag{4}
\]

All four seeds are strictly positive, analytic functions on the **entire**
SU4 group. There is no logarithm branch, support excision, angular wall,
packet-only Hamiltonian, finite-irrep replacement or domain change.
Character packets can have symmetry-required nodes; these are zeros of a
smooth function, not deleted group points. Norms are positive: at U=I the
four q-values reduce to combinations of `1,e^(-4eta),e^(-8eta),e^(-4eta)`,
and are nonzero for eta>0. Centre translation gives exactly

\[
 \psi_q(zU)=z^q\psi_q(U).                               \tag{5}
\]

This is the fundamental centre character for q=1 and antifundamental for
q=3. It is not a claim that psi alone transforms as a full fundamental
SU4 representation: W at a fixed frame breaks left/right symmetries to the
appropriate covariant joint field-link action.

### Full SU4 Gauss pairing with a reference

On the actual two-frame orbit let `Phi_L=gL Phi gL^dagger`,
`Phi_R=gR Phi gR^dagger`, and body link `V=gL^dagger U gR`. The concrete
fundamental-L/antifundamental-R section is

\[
 \Xi(g_L,g_R,U)=\psi_1(V)\,(g_L v)\otimes\overline{g_R w}.
                                                               \tag{6}
\]

Here v,w are fixed unit body vectors and R is a compensating reference.
Represent (6) as the matrix
`psi1(V) (gL v)(gR w)^dagger`. Replacing gL or gR independently by that
frame's centre stabilizer leaves Xi unchanged: the packet character and
reference vector phase cancel. Thus (6) descends from SU4 orientations to
the true adjoint frame orbit `SU4/Z4`; no artificial frame sheet is chosen.
Under actual local gauge transformations it transforms as
`Xi -> hL Xi hR^dagger`, with both frame and link derivatives present.
Infinitesimally the full coordinate Gauss tangent equals the external
matter action `iT_a Xi` at L and `-i Xi T_a` at R for **each of all15**
generators. The neutral scalar q0 packet needs no charged references;
the q3 construction uses the conjugate endpoint representations. The q2
packet can be paired with wedge-two references, but no physical q2 particle
is inferred. Section (6) is a charged *trial section*, not a solution of its
field-angular energy or an original-light creation operator.

The CLI tests centre descent at both endpoints and finite-difference full
field/link Gauss actions at a noncentral link. A wrong fundamental at both
ends fails the same native right-Gauss equation. Matching centre labels
alone would not pass these tests.

## 3. Exact infinite-Haar norm and kinetic/normal energy formulas

Let `t=TrU`, `t2=Tr U^2`, and

\[
 C=\sum_a|\operatorname{Tr}(T_aU)|^2=2-|t|^2/8,
 \quad D=\sum_a\operatorname{Tr}(T_aU)^2=(t2-t^2/4)/2.  \tag{7}
\]

These follow from completeness
`sum_a (T_a)_ij(T_a)_kl=(delta_il delta_jk-delta_ij delta_kl/4)/2`.
Writing `A=psi_(q-1)` and `B=psi_(q+1)`, differentiation of all four seeds
on all 15 native Casimir directions gives

\[
 D_a\psi_q={i\eta\over2}
     [A\operatorname{Tr}(T_aU)-B\overline{\operatorname{Tr}(T_aU)}],
 \qquad
 \sum_a|D_a\psi_q|^2={\eta^2\over4}
 [(|A|^2+|B|^2)C-2\operatorname{Re}(A\bar B D)].       \tag{8}
\]

No kinetic contribution is dropped at well boundaries: (4) has no such
boundaries. For a class weight the conjugation average of W is exactly

\[
 \bar W(U)={8\over15}\Big(\sum_\mu g_\mu\Big)(16-|t|^2). \tag{9}
\]

Indeed conjugation averaging of Ad_U acts on the irreducible traceless
matrix space as `(|TrU|^2-1)/15`, and `sum TrPhi_mu^2=4 sum g_mu`.
Consequently the exact norm and trial energy are

\[
 N_q=\int_{SU4}|\psi_q|^2dH,
 \quad R_q={u_4\int\sum_a|D_a\psi_q|^2dH+
                    v\int\bar W|\psi_q|^2dH\over N_q}. \tag{10}
\]

For the class integrands use **all** Weyl-torus points:

\[
 \int FdH={1\over24(2\pi)^3}\int_{[-\pi,\pi)^3}
 F(\operatorname{diag}(e^{i\theta_1},e^{i\theta_2},e^{i\theta_3},
 e^{-i(\theta_1+\theta_2+\theta_3)}))
 \prod_{i<j}|e^{i\theta_i}-e^{i\theta_j}|^2\,d^3\theta. \tag{11}
\]

The CLI evaluates (10)-(11) on successively refined full-torus grids,
including Haar normalization, and compares q0/q1/q3 norms, potential and
kinetic energies. It independently evaluates the source W and all15 direct
matrix derivatives at generic noncentral SU4 controls; (8) must agree with
the direct derivative sum. Inversion U->U^dagger maps q1 to q3, preserves W
and Haar/kinetics, and gives exactly `N1=N3` and `R1=R3`.
The grid differences are **not interval error bounds**. No numerical ground
or unrestricted eigenvalue is reported. Severe well narrowing needs a finer
Haar grid; the receipt exposes, rather than relabels, that discretization
debt.

### A rigorous common normal trial upper bound

The following global smooth-packet bound is independent of the quadrature.
For eta>=1 set `I=integral f0^2 dH`. The canonical exponential Jacobian is
`prod_{i<j} sinc((lambda_i-lambda_j)/2)^2`. In the canonical unit ball,
`|lambda_i-lambda_j|<=|x|<=1`, so it is at least `j0=sinc(1/2)^12`.
The group volume for metric `2Tr(dX^2)` is

\[
 V_{SU4}=2^{15/2}{(2\pi)^9\over6},\qquad
 B_{15}={\pi^{15/2}\over\Gamma(17/2)}.
\]

For clarity, the normalized local density factor can also be derived
without any physical input from (11). Near the identity, its Gaussian
integral uses the elementary Vandermonde identity
`integral_R4 exp(-sum y_i^2/2) Delta(y)^2 dy=(2pi)^2 prod_{j=1}^4 j!`.
Factoring out the centre-of-mass direction and the `1/24` Weyl factor yields
`integral exp(-eta|x|^2/2)dH ~
[6/(2^(15/2)(2pi)^(3/2))] eta^(-15/2)`.
Equating this with `(2pi)^(15/2) eta^(-15/2)/V_SU4` gives the displayed
volume; the canonical SU2 check gives radius2, volume16pi^2.

On `|x|<=eta^(-1/2)`,
`ReTr exp(iX)>=4-|x|^2/4`. Thus

\[
 I\ge I_-={B_{15}\over V_{SU4}}e^{-1/2}j_0\eta^{-15/2}. \tag{12}
\]

Distinct seeds satisfy `f_k f_l<=exp(-delta eta)` with
`delta=8-4sqrt2>0`, since
`ReTr[(z_k^-1+z_l^-1)U]<=4|z_k^-1+z_l^-1|<=4sqrt2`.
This deliberately loose bound also covers opposite wells. Let
`s=exp(-delta eta)/I_-`. The sector norms obey
`4I(1-3s)<=N_q<=4I(1+3s)`.

For `s0(U)=ReTrU`, `Delta s0=-(15/8)s0`.
Integration by parts on the whole compact group gives

\[
 \int|\nabla f_0|^2dH
  ={\eta C_F\over2}\int s_0 f_0^2dH
  \le {15\eta\over4}I.                               \tag{13}
\]

Set `d0=4-s0`, which lies in [0,8]. From (3), W<=2 ell_max d0.
For any d>=0,
`<d0>_{f0^2}<=d+8 exp(-2eta d)/I_-`.
Taking `d=log(8eta/I_-)/(2eta)` gives the valid upper bound
`dbar=min(8,[log(8eta/I_-)/2+1]/eta)`.
Therefore

\[
 B_{seed}=15u_4\eta/4+2v\ell_{max}\bar d
\]

bounds the one-well Rayleigh energy. Completeness in (7) gives
`|grad s_k|^2<=2`; hence each cross kinetic form is at most
`2u4 eta^2 exp(-delta eta)` in magnitude, and each cross potential form is
at most `4v ell_max exp(-delta eta)`. Put
`h=(2u4 eta^2+4v ell_max)s`. If `3s<1`, then for **every** centre sector

\[
 R_q\le {B_{seed}+3h\over1-3s},\qquad
 |R_q-R_{seed}|\le {3h+3sB_{seed}\over1-3s}.            \tag{14}
\]

If this coarse norm certificate fails at moderate eta, the CLI reports no
certified four-sector bound; it does not clip the denominator or substitute
a numerical proof. Formula (14) proves a common local **trial** energy,
not a vacuum eigenvalue, isolated quartet, tunnelling formula or charged
lower mass. With eta proportional to sqrt(v/u4), the bound is
`O(Lambda^(3/8) log Lambda)`, whereas the sector-trial deviations are
exponentially small in eta up to the displayed polynomial prefactors.
The CLI's eta is the optimized *isotropic Euclidean packet* scale
`sqrt(64 v sum g/(15u4))`, clamped below at4. It is not a fit to a spectrum.
Floating evaluation of the analytic inequalities is not outward-rounded
interval arithmetic.

## 4. A true scalar one-link charge splitting bound

There is a stronger result than equal localized upper trial energies.
For u4>0 and any real smooth centre-invariant W, the elliptic scalar
operator (2) on the entire connected compact group has a unique strictly
positive ground f0. Positivity and uniqueness imply its centre character
is trivial. Call its energy E0; no ground is computed in the CLI.

Multiply by all sixteen actual fundamental matrix coefficients:
`f_ij(U)=U_ij f0(U)`. Each has character q1, and

\[
 \sum_{ij}\|f_{ij}\|^2=4\|f_0\|^2,
 \quad
 \sum_{ij}\langle f_{ij},H f_{ij}\rangle
       =4(E_0+u_4 C_F)\|f_0\|^2.                       \tag{15}
\]

The potential is unchanged. The cross gradient vanishes because
`Tr(U^dagger iT_a U)=iTr T_a=0`, and the extra gradient sum is
`sum_a Tr(T_a^2)=4 C_F`. At least one component has Rayleigh quotient no
larger than this weighted average. Ground minimality gives the lower bound.
Conjugate coefficients give q3. Thus the exact scalar one-link result is

\[
 0\le E_1-E_0\le15u_4/8,
 \qquad 0\le E_3-E_0\le15u_4/8.                       \tag{16}
\]

At `u4=.3 Lambda^(-1/2)` the upper splitting tends to zero. This is a real
spectral comparison for the scalar frozen-link consumer, not a positive
charge gap or a spectrum of the full original fermionic Gauss problem.

**Adverse example:** at v=0 the full SU4 free rotor has E0=0 and
E1=E3=15u4/8>0. The Casimir formula
`C(a,b,c)=lambda^T Ainv(lambda+2rho)/2`,
`Ainv=[[3,2,1],[2,4,2],[1,2,3]]/4`, gives the fundamental value15/8 and
all nontrivial dominant weights at least15/8. The fundamental and
antifundamental coefficients attain it. As eta grows, the four localized
packets still have asymptotically equal *large upper trial energies*.
This explicit actual-group counterexample prevents upgrading (14) alone
to equality of true sector grounds. In (16) the separate product identity,
not the packet upper bound, supplies the spectral comparison.

## 5. Exact original full-Gauss external colour-pair transfer

The scalar positivity proof does not extend to the original matrix-valued
heavy/light ground. Instead use a product identity that needs **no scalar
angular/Berry assumption at all**. Let Psi_N be any normalized state of the
original neutral full-Gauss Hamiltonian, in all field coordinates, SU4 and
U1 links and the original whole CAR sectors. Tensor on one spectator
fundamental source at L and antifundamental source at R. These sources have
no imported hopping, mass, spin or kinetic energy. They define a new
*external-source comparison sector*, not two added dynamical species or an
original-light excitation.

For a simple path gamma with distinct links, let
`P_gamma=ordered product of U_e` (use U_e^dagger for reversed edges) and

\[
 (\mathcal T_\gamma\Psi_N)_{ij}=\tfrac12(P_\gamma)_{ij}\Psi_N.
                                                               \tag{17}
\]

At every configuration `||P_gamma/2||_HS^2=1`. Independent local SU4 gauge
transformations cancel at every interior vertex and transform the endpoint
matrix by `hL P_gamma hR^dagger`, exactly matching the two external
representations. Tensoring the neutral state's existing equivariance gives
**all** primitive Gauss equations; it is not a Z4-only projection. At the
flat identity reference the external state is the actual colour singlet
`sum_i |i,bar i>/2`. On general noncentral links the complete global Gauss
generator includes the actual link tangent; the reference alone need not
be a singlet. No q=0 generator, field angular charge or sea background is
deleted. Net endpoint centre character is always zero.

For each distinct link and each of all15 native left derivatives,

\[
 \langle M,D_aM\rangle_{HS}=0,\qquad
 \sum_a\|D_aM\|_{HS}^2=C_F,
 \quad M=P_\gamma/2.                                  \tag{18}
\]

For example, `D_a P=A(iT_a U)B` for a forward edge. Unitarity removes A,B
from the norm and yields the trace-zero cross cancellation. Reversed edges
have the same norm and cancellation. Equation (18) holds before integrating
and for an arbitrary vector-valued Psi_N with arbitrary coordinate
correlations. The kinetic product rule therefore gives

\[
 \langle\mathcal T_\gamma\Psi_N,H_{ext}\mathcal T_\gamma\Psi_N\rangle
 =\langle\Psi_N,H_N\Psi_N\rangle+|\gamma|u_4 C_F.      \tag{19}
\]

Every field derivative is unchanged because M is independent of Phi;
U1 derivatives are unchanged because these external sources are colour
only. Every multiplication operator is unchanged, including the entire
heavy/light matrices, quartic, alignment and all 3D plaquettes, even when
Psi_N is entangled with them. These operators act trivially on the external
reference indices. Thus heavy mean colour density, field angular variance,
Born-Huang terms and the full q0 mechanics have **not** been asserted scalar,
projected away, or canceled by fiat. In a BO rewrite their pieces must
recombine into the exact unprojected identity (19).

Using a neutral ground, or a minimizing sequence if needed, proves

\[
 E_{ext}(L,\bar R)-E_N\le |\gamma|15u_4/8.             \tag{20}
\]

No nonnegative lower difference follows for the original fermionic
matrix-valued Hamiltonian from this argument. The CLI exercises the
pointwise identity using arbitrary seven-component neutral derivatives,
actual all15 generators, noncentral one- and two-edge links, and interior
Gauss cancellation. This is an operator identity applied to a hypothetical
exact ground, **not an alleged numerical unrestricted-ground run**.

### What (20) does and does not buy in 3D

At fixed graph and separation, (20) tends to zero with the declared u4
trajectory. More generally its upper tends to zero for distances
`|gamma|=o(Lambda^(1/2))`. This is an order-of-limits variational statement.
At every fixed finite Lambda it grows linearly with path length. It does
not prove a path-independent asymptotic energy, isolated superselection
charge, finite string tension zero, 3D plaquette phase, light-particle
mobility or thermodynamic deconfinement. The Wilson transporter leaves all
plaquette *multiplication* expectations unchanged in this entangled external
trial, but its electric cost and endpoint-sector ground comparison alone
do not determine a many-body phase.

To turn the external source into the issue's mobile physical charge one
still needs an actual original-light CAR insertion/propagator, its energy
and longitudinal Gauss/current embedding, controlled reference separation
and volume limit, and the full low-energy matter/field projection. The
source has not been given new local CAR supplier energy, and no such
embedding is echoed from earlier conditional constructions. Colour-only
external sources also do not provide the physical U1 charge of an electron.
Adding a U1 Wilson path for charged external probes would add its own
`u1 |gamma|/2` electric cost, not vanish by this colour argument.

## 6. Rotation and the retained compact Berry adverse boundary

For the normalized actual Clifford gamma1,gamma2, derive
`S(theta)=exp(theta gamma1 gamma2/2)`. Its endpoint at2pi is -I.
The q1 centre character at this endpoint is -1. The anti-reference character
is also -1, so co-rotating both gives +1. A moving fundamental basis has
Berry connection `S^dagger dS=gamma1 gamma2/2 dtheta`; the connection
parallel transporter is `S(2pi)^-1`. Multiplying by the endpoint overlap
gives the exact restored compact pure-gauge loop **+I**, not a physical
minus sign. The CLI evaluates both exponentials on the actual source frame.

A localized operation with a *fixed* remote reference is different from
co-rotating that reference. Neither the compensator endpoint nor a frozen
centre character derives a localized physical rotation factor. A mobile
charge sector and a localization/symmetry-factorization map carrying the
full field, link, matter and reference Noether generator are missing.
The full physical restored rotation remains +1 here; no Gamma matrix is
imported as mechanical spin.

## 7. Executable evidence and honest remaining scope

Run with the project-native Python (BLAS threads set to one by the CLI):

```
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/original_centre_charge.py
```

Optional `--spacing .001` selects the actual `kappa=2/a` source frame, not
an a1 frame relabelled afterwards. `--lambdas` and `--haar-grids` control
explicit normal trial diagnostics; they do not request a many-body eigensolve.
The generated `original_centre_charge_receipt.json` binds this source,
proof and original source dependencies and rejects changes during execution.
It records the exact scalar splitting interval and the original full-Gauss
external-pair upper cost separately from finite Haar estimates.

Language-tool evidence: the OMP `xd://lsp` device was unavailable, reported
through QA. A direct native Pyright1.1.414 JSON-RPC server rooted at the
project `pyproject.toml` exercised document symbols and definition on the
original selected-Gram source before editing; the structural Python
`ast_grep` query matched its actual function declarations and the new typed
packet function. After explicitly configuring the project-native interpreter,
the new-file LSP diagnostic publication was empty. The unavailable OMP
adapter integration is not falsely claimed fixed.
The complete native CLI exited naturally with all checks true on the
default Lambda1/16/256 and Haar grids96/128/160. At Lambda256 the refined
four trial energies were approximately8.26632888756, the final-grid
relative energy difference was1.72e-15 (not an interval error), the common
analytic trial upper was67.7388961840, and the rigorous scalar charge
splitting upper was.03515625. All15 packet/reference Gauss residuals
were at most1.86e-9; the wrong-reference adverse residual was.675828.

Remaining joined obligations are not closed by this supplier: original
mobile-light charge, a charged lower mass and pole, interacting 3D phase and
reference-independent localization, full angular/BH transfer, physical
mechanical spin, quantum photon/current pole and electron-photon scattering.

# Physical thin-Cao core: Floquet exclusion and infinite-rank exterior response

## 1. Object, imports, and exact scope

Fix **one** sufficiently thin, centered member of the smooth `p>=6` Cao–Lai–Qin–Zhan–Zou 3D Euler vortex-ring family. The fluid has constant density `rho_0>0` on `R^3`; the velocity has finite kinetic energy and decays at infinity, while vorticity has compact solid-torus support `K_0`. In cylindrical coordinates, with `R=partial_theta`,

    omega_0=zeta(r,z) R,                zeta=epsilon^(-2)(P_+)^p,
    v_0=B_R3 omega_0,                  W_0=v_0-c_0 e_z,
    B_R3=curl(-Delta_R3)^(-1),
    C_0 xi=-[xi,omega_0]=curl(xi cross omega_0),
    A_0 eta=-[W_0,eta]-[B_R3 eta,omega_0].                 (1)

`B_R3` is the **whole-space decaying** Biot–Savart/Leray operator, including the irrotational exterior, axis regularity and physical transmission. The velocity-form Euler equation includes the associated global pressure/Leray projection. The ring is an actual translating Euler solution; an arbitrary centralizer introduced below is not automatically an Euler solution. All statements are for this one fixed member, not uniform in `epsilon` and not a statement about every Cao ring.

Imported, not reproved: Cao et al., arXiv:2206.10165v2, Proposition 1.4, Proposition 3.2 and Appendix A Lemma A.2 supply the smooth thin-ring source, its rescaled radial limit, and regular boundary. P253/0066 §1 equations (6a)–(6f), independently reviewed in P253/0073 §1, upgrade the source convergence to `C^2` and prove that `zeta` has **one negative-definite interior maximum**, every other positive level is a connected regular meridional circle, and the meridional period is positive and finite for this fixed ring. This no-saddle lemma is **not** a new result of 0162. P253/0062 §§2–3 gives (1), the positive-core DA map and the full Hodge intertwiner; §6 gives the axisymmetric positive-core centralizer classification and smooth flow conjugacy. P253/0066 §4 already proves one regular-cell essential-spectrum inclusion, not a continuum source-specific adjoint or residue. The 0161 R3–R5 witnesses are two-mode/model calculations; `zeta=r^2-z^2` there is not the selected Cao profile.

**New physical-carrier conclusions below:** (T1) the complete positive-core **three-dimensional** Floquet classification, including arbitrary toroidal drift and stalled regular levels, excludes the model partition-2 hyperbolic periodic orbit for every smooth same-leaf centralizer conjugate on this selected carrier; (T2) the global exterior response to compact positive-core DA sources has infinite rank in each fixed `|n|>=2` harmonic, and a selected deep-core source's exact Euler generator reaches every positive-core boundary collar. T1 excludes that *particular* smooth hyperbolic-topology route, not a Cao ring, a transparent branch, a boundary/exterior mechanism, or Euler particles. T2 excludes a finite exterior-field/core-only substitution on the full smooth DA source class, **not** a finite-dimensional resonant cokernel after an exact global Green solve.

## 2. T1: no hyperbolic positive-core periodic orbit for smooth centralizers

Let `Y` be any `C^2`, divergence-free physical vector field commuting with `omega_0` on the positive core `K_0^+={zeta>0}`, with a smooth global extension obeying the physical finite rows. The 0062 centralizer calculation shows `Y` is axisymmetric there and

    Y_pol=F'(zeta) J grad(zeta)/r,        Y^theta=b(r,z),  (2)

where `F` is smooth on the positive regular levels and `b` is any smooth toroidal drift allowed by the global field. On the regular levels introduce the actual meridional volume action `I` and travel-time angle `beta`, normalized so `W_0=omega(I)partial_beta` with `omega(I)>0` (0066). Because `zeta` is a function of `I` and `Y_pol` is proportional to `W_0` on each level, (2) becomes exactly

    dot I=0,             dot beta=nu(I),
    dot theta=b(I,beta).                                  (3)

No finite cross-sectional harmonic truncation or prescribed 2×2 matrix is used. There are three exhaustive locations for a positive-core periodic orbit:

1. **Regular level, `nu(I_0)!=0`.** The average `B(I)=(2*pi)^(-1) integral_0^{2*pi} b(I,beta)d beta` is smooth near `I_0`. The periodic equation `nu(I)partial_beta u=B(I)-b(I,beta)` has a smooth zero-mean solution there; `theta'=theta+u(I,beta)` transforms (3) to `dot theta'=B(I)`. The return map after any actual closed meridional/toroidal orbit is locally `I'=I`, `beta'=beta+T nu(I)`, `theta'=theta'+T B(I)` (angles modulo `2*pi`). Its derivative is triangular with diagonal `(1,1,1)`. Every transverse Floquet multiplier has modulus one. At irrational winding no periodic orbit exists on that torus.
2. **Regular level, `nu(I_0)=0`.** The complete level, not an isolated point, has stalled meridional motion. For a toroidal orbit with `b(I_0,beta_0)!=0`, its time-`T` derivative in `(I,beta,theta)` is triangular with diagonal `(1,1,1)`; the transverse `(I,beta)` block is `[[1,0],[T nu'(I_0),1]]`. Dependence of `b` on `I,beta` supplies only the other triangular entries. If `b=0` the point is stationary, not a nontrivial hyperbolic periodic orbit. This covers the apparent `F'(zeta)=0` escape without assuming constant toroidal drift.
3. **The unique positive-core center.** In local Cartesian meridional coordinates `x=(r-r_c,z-z_c)`, put `H=D^2 zeta(r_c,z_c)<0`. Tangency `Y_pol dot grad zeta=0` and nonsingularity of `H` force `Y_pol(r_c,z_c)=0`. Writing the `C^2` meridional field as `Y_pol=Lx+O(|x|^2)` and comparing quadratic terms gives `HL+L^T H=0`. Since `-H` is positive definite, `L` is similar to a real skew-symmetric matrix; its eigenvalues are imaginary or zero, without assuming that `F'` extends differentiably to the center. For a toroidal orbit at the center, meridional Floquet multipliers therefore lie on the unit circle; derivatives of the toroidal drift add a triangular coupling, not an unstable normal multiplier. If toroidal drift vanishes, the center is stationary.

These exhaust the positive-core foliation reviewed in 0066. Now let `g` be a smooth volume-preserving same-leaf diffeomorphism and `W=g_*Y`, including any exact rotating Euler relative field if a carrier-map solution `W=g_*Y` is subsequently found. The exact flow relation `Phi_W^t=g o Phi_Y^t o g^(-1)` conjugates every positive-core return map and preserves its Floquet multipliers. **T1:** neither `Y` nor its conjugate has a hyperbolic periodic orbit in the corresponding positive core. The result quantifies over a *larger* class than actual matching Euler branches; it does not assert such a new branch exists. A separately proved hyperbolic boundary/exterior orbit, a nonsmooth leaf map, or a carrier not satisfying the reviewed thin-ring foliation lies outside this result.

**Decisive contrary case.** The source equation `-Delta P+r^(-1)P_r=r^2 epsilon^(-2)(P_+)^p` alone gives `tr D^2P=-r_0^2 epsilon^(-2)P_0^p` at a positive critical point, not the Hessian determinant. Locally, positive analytic Cauchy data `P(r,0)=P_0+a(r-r_0)^2/2`, `P_z(r,0)=0`, `P_0,a,r_0>0`, yield `D^2P(r_0,0)=diag(a,-a-r_0^2 epsilon^(-2)P_0^p)`, an indefinite saddle. This is an **elliptic-PDE local germ, not a global finite-energy Euler ring**. It shows why the Cao `C^2` thin-ring import, rather than the equation's trace alone, is essential. For a physical within-class check, take `Y=W_0` and add a smooth compact-core toroidal drift and `J grad H(zeta)/r` supported strictly inside `K_0^+`; choose `H'` to stall one regular level. This globally smooth divergence-free centralizer has a whole stalled level, no isolated saddle. It is a topology countercase, **not** a new Euler carrier-map solution.

## 3. T2: infinite-dimensional physical exterior Hodge response

Fix an integer `n>=2`; its conjugate gives the real `-n` component. For the rank theorem take **all** `q in C_c^infinity(K_0^+)`, each individually supported a positive distance inside the meridional core and away from the symmetry axis; no one disk `U` is fixed across this source class. Only for the explicit nonzero-moment witness and collar-leakage argument below do we select a single small disk `U` and a `q` supported in it. The physical orthonormal cylindrical displacement is

    xi_q=curl[(q/r)e^(in theta)e_theta]
        =e^(in theta)[-q_z/r e_r+q_r/r e_z],             (4)
    F_q=xi_q cross omega_0
        =-e^(in theta) zeta (q_r e_r+q_z e_z),
    eta_q=C_0 xi_q=curl F_q,
    h_q=B_R3 eta_q=P_L F_q.                              (5)

The displacement is divergence free, compactly supported and belongs to `H^5`; `eta_q` is a smooth compact positive-core DA vorticity in the physical `H^3` graph domain. With `n>=2`, the axial/tilted symmetry, center, impulse, rotation and zero-harmonic circulation rows have different SO(2) characters. The standard rows are therefore fixed exactly; an **additional arbitrary** same-character constraint including one of the following multipole functionals would change the quantified class and must be declared separately. The Leray identity in (5) uses the compact-curl zero low-frequency moment and the decaying whole-space convention; no core Dirichlet problem is substituted. `rho_0` enters energy/KKS normalization, not the kinematic map (5).

For each `ell>=n` let `H_(ell,-n)=R^ell Y_(ell,-n)(angle)=e^(-in theta)P_ell(r,z)` be a regular homogeneous solid harmonic, with spherical harmonics orthonormal on the unit sphere. Define the physical exterior multipole functional

    Q_ell(q)=integral_R3 F_q dot grad H_(ell,-n) dx
             =-2*pi integral_(K_0^+) r zeta grad_(r,z)q
                              dot grad_(r,z)P_ell dr dz. (6)

It has units `[Q_ell]=L^(ell+3)/T` if `[zeta]=1/(LT)`, `[q]=L^3`, `[xi]=L`, and `[h]=L/T`. (Indeed `q=r phi` for a vector-potential amplitude `[phi]=L^2`.) Let `u_q=Delta^(-1) div F_q`. Outside `supp F_q`, (5) is the gradient `h_q=-grad u_q`; the addition theorem for the free-space Laplace kernel gives the `(ell,n)` exterior potential coefficient as a nonzero universal multiple of `Q_ell(q) R^(-ell-1)`. Thus a nonzero `Q_ell` yields a nonzero physical exterior velocity of order `R^(-ell-2)`, with global transmission and finite energy. No resonant adjoint is inferred from this ordinary Laplace multipole identity.

**Independence proof.** Suppose a finite linear combination `Q_P(q)=sum_{ell=n}^N c_ell Q_ell(q)` vanished for *every* compact positive-core `q`; put `P=sum c_ell P_ell`. Equation (6) in distributions on `K_0^+` gives

    div_(r,z)(r zeta grad_(r,z) P)=0.                     (7)

Since `p>=6`, `zeta` extends continuously by zero at the actual compact regular boundary; `r zeta grad P` has zero boundary flux. Multiply (7) by `conjugate(P)` and integrate over the whole meridional support. Integration by parts gives

    integral_(K_0^+) r zeta |grad_(r,z)P|^2 dr dz=0.     (8)

Hence `P` is constant on every connected positive component. It is a polynomial analytic in `(r,z)` for `r>0`, so constancy on one open component makes it constant identically. Every `P_ell` in the nonzero-`n` angular sector is divisible by `r^n`; evaluation at the axis forces that constant to be zero. Distinct homogeneous `ell` have independent polynomial degrees, so every `c_ell=0`. Therefore the functionals `Q_ell`, `ell>=n`, are linearly independent. The exact map from smooth fixed-`n` DA sources to the *entire exterior velocity field* has **infinite rank**: any purported finite-dimensional exterior-field representation would give only finitely many independent coefficient functionals, contradicting (6)–(8). This says nothing against finitely many circulation/slice rows or a finite resonant Grushin cokernel **after** the full infinite-rank Green solve.

A direct nonzero seed and a falsifying zero-moment case make the quantifier visible. For the unnormalized harmonic `H_n=r^n e^(-in theta)` and `q=r phi`, integration by parts in (6) gives

    Q_n(phi)=2*pi*n integral r phi partial_r(r^n zeta) dr dz.  (9)

Because nonzero compact `zeta` cannot have `partial_r(r^n zeta)` identically zero in its positive core, choose a small sign-definite patch and a nonnegative nonzero bump `phi`; then `Q_n!=0`. In contrast, two disjoint patches with nonzero moments admit the **nonzero** linear combination `phi=phi_1-Q_n(phi_1)phi_2/Q_n(phi_2)` for which `Q_n=0`. A vanishing degree-`n` moment does not mean every higher moment or the full exterior field vanishes. The `n=0` case has a different circulation row and `grad H_0=0`; it is not covered by this theorem.

**Generator-support consequence.** Choose the nonzero-moment source above with support in a small rotational disk `U`; `R^3\supp F_q` has a connected component containing every positive-core boundary collar and the exterior. In any positive-core open collar disjoint from `U`, `eta_q=0`, so

    A_0 eta_q=-[h_q,omega_0]=C_0 h_q.                    (10)

If (10) vanished identically on an entire such collar, the exact positive-core nonzero-`n` DA formula (0062 equation (11)) would give `h_q^r=h_q^z=0` there since `zeta>0`. The divergence-free `n`-harmonic condition then gives `h_q^theta=0` there. As `curl h_q=div h_q=0` off `U`, harmonic unique continuation on the connected complement would force its exterior to vanish, contradicting `Q_n!=0`. Therefore **every** positive-core boundary collar contains a point where `A_0 eta_q!=0`, even though the input vorticity vanishes throughout a fixed collar. The physical class of smooth DA fields supported a positive distance inside the core is not invariant under `A_0`. This does not assert a nonzero boundary trace (the carrier vanishes like `d^p`) or nonexistence of the weighted completed DA graph.

## 4. Route decisions and next dependency

| Route | Exact result | What it changes | Open alternative |
| --- | --- | --- | --- |
| Model R5 partition 2 transported to the selected thin Cao positive core | **Refuted at this scope by T1.** All smooth positive-core centralizers and same-leaf conjugates have no hyperbolic periodic orbit, even with toroidal drift/stalled levels. | Stop treating the `r^2-z^2` hyperbolic witness as a license for smooth inner/outer matching on this carrier. | Source-specific transparency/integrable carrier-map route; or prove a different global finite-energy Euler carrier with source-defined compatible saddle. Boundary/exterior and weaker singular routes need independent admissibility. |
| Finite exterior-field/core-only replacement for the physical fixed-`n` DA class | **Refuted at this scope by T2.** Exterior multipoles are infinite-rank; a selected deep-core seed has a nonzero actual far field and immediate generator feedback into every positive-core collar. | The physical trace construction must retain the whole-space Green/interface/exterior map, even when its source is compactly inside the core. | Infinite-dimensional Hodge solve followed by a possibly finite-dimensional sandwiched adjoint/Grushin trace in a weighted physical DA graph. |
| Physical continuum resonance, nonlinear branch and persistence | **Blocked, not zero/refuted.** Neither theorem constructs a spectral eigenmode, graph-bounded adjoint, `V_*`, exact nonlinear pressure/branch or 3D persistence neighborhood. | Disambiguates the next proof's topology and exterior input. | Prove a source-specific closed weighted graph and whole-space Hodge/limiting-absorption adjoint for a selected real DA seed, then test transparency; or change to a materially different actual Euler carrier. |

Both T1 and T2 use actual three-dimensional Euler kinematics and the physical Cao carrier. They do **not** import the potential/action of #218, assert a new charged field, or claim electron/neutrino identification. No P2/LP2 completion, accepted registry/release change, or terminal PR follows. The contrary analytic saddle germ is deliberately not misrepresented as a second physical carrier.

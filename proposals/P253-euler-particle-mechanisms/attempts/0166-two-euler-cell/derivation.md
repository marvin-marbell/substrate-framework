# P253/0166: two nonlinear Euler sheets do not form a material two-Euler cell

**Scope and verdict.** This is an exact, *candidate-specific falsifier*, not a no-go theorem for every material-cell representation. We attempted to place two independently pressure-supported 2-D Euler fluids inside one unforced, constant-density, incompressible **3-D Euler** flow. Two genuinely nonlinear Euler sheets have a computable nonzero mixed 3-D acceleration when they intersect; localizing even one sheet across a transverse material layer produces an immediate physical pressure/vertical-velocity tail outside it. Neither preparation is an invariant two-Euler material-cell family for positive time. The surviving stress formula comes from the single Euler kinetic action, but it does **not** turn these sheets into two independent physical flow maps. No particle result follows.

The velocity and pressure always solve

\[
 \partial_tu+(u\cdot\nabla)u=-\rho^{-1}\nabla p,\qquad
 \nabla\cdot u=0,\qquad
 -\Delta p=\rho\,\partial_i u_j\,\partial_j u_i
       =\rho\,\nabla\cdot[(u\cdot\nabla)u].                 \tag{1}
\]

All assertions about positive time refer to the usual *local smooth Euler solution* from the displayed smooth data, not to a globally regular 3-D solution. Coordinates on the first flat torus have period \(2\pi\); restoring physical wave number \(k\) multiplies accelerations by \(k\). The second domain is \(\mathbb T^2_{x,y}\times\mathbb R_z\), with decaying pressure modes and finite energy per periodic transverse cell. No artificial internal wall or additional material constant is assumed.

## 1. Intersecting nonlinear planar Euler sheets: an exact pressure and closure test

Take two streamfunctions on different, overlapping coordinate planes,

\[
 \begin{split}
 \psi_1=\sin x\sin y,&\quad
 a=(\sin x\cos y,-\cos x\sin y,0),\\
 \psi_2=\sin y\sin z,&\quad
 b=(0,\sin y\cos z,-\cos y\sin z),\qquad
 u_0=\alpha a+\beta b.                                         \tag{2}
 \end{split}
\]

Here \(\alpha,\beta\) are independently preparable physical velocity amplitudes, not adjustable coupling constants. Each isolated field admits its own *whole* 2-D Euler transport-plus-pressure law for arbitrary smooth streamfunction initial data on its plane (lifted constantly in the unused coordinate). Specifically, these two nonconstant examples have nonzero pointwise self-advection and are individually stationary exact 3-D solutions:

\[
 (\alpha a\cdot\nabla)(\alpha a)
  =\alpha^2(\sin x\cos x,\sin y\cos y,0),\quad
 p_a=-\frac{\rho\alpha^2}{2}(\sin^2x+\sin^2y),                 \tag{3}
\]

\[
 (\beta b\cdot\nabla)(\beta b)
  =\beta^2(0,\sin y\cos y,\sin z\cos z),\quad
 p_b=-\frac{\rho\beta^2}{2}(\sin^2y+\sin^2z).                  \tag{4}
\]

Pressures are understood up to spatial constants. Thus the second sheet is **not** a passively carried scalar, an affine flow, or a pressure-free shear; nonetheless these equations do not supply two independent *within-cell* flows after superposition. Write the mixed acceleration \((u_0\cdot\nabla)u_0=(\alpha a\cdot\nabla)(\alpha a)+(\beta b\cdot\nabla)(\beta b)+\alpha\beta C\), where

\[
 C=(-\sin x\sin^2y\cos z,
    -2\cos x\sin y\cos y\cos z,
    -\cos x\sin^2y\sin z),\qquad
 \nabla\cdot C=-\cos x\cos z(1+\cos2y).                         \tag{5}
\]

The physical zero-mean mixed pressure determined by (1), not assigned independently to either sheet, is

\[
 \frac{p_\times}{\rho\alpha\beta}
   =-\frac12\cos x\cos z-\frac16\cos x\cos2y\cos z.            \tag{6}
\]

The three-dimensional Leray projection therefore yields **exactly at \(t=0\)**

\[
 \partial_tu\big|_0
   =-\alpha\beta(C+\nabla[p_\times/(\rho\alpha\beta)])
   =\frac{2\alpha\beta}{3}
    (-\sin x\cos2y\cos z,
      \cos x\sin2y\cos z,
     -\cos x\cos2y\sin z)=:R.                                  \tag{7}
\]

The vector in (7) is divergence-free and is orthogonal in \(L^2(\mathbb T^3)\) to **every** sum of an \(xy\)-columnar, \(z\)-independent vector field and a \(yz\)-columnar, \(x\)-independent vector field, even if those fields have arbitrary time dependence. Denote this closed column-sum space by \(S\), and its orthogonal complement projection by \(Q\). Direct trigonometric integration gives

\[
 Q u_0=0,\qquad
 \|Q\partial_tu|_0\|_2=\|R\|_2
   =|\alpha\beta|\sqrt{|\mathbb T^3|/6}>0
       \quad(\alpha\beta\ne0).                                  \tag{8}
\]

Every classical solution from (2) consequently leaves \(S\) for *every sufficiently small* positive time: for some \(T>0\),

\[
 \operatorname{dist}_{L^2}(u(t),S)
   \ge t|\alpha\beta|\sqrt{|\mathbb T^3|/6}
        -\frac{t^2}{2}\sup_{0\le s\le T}\|\partial_t^2u(s)\|_2,
       \qquad 0<t\le T.                                       \tag{9}
\]

The finite supremum in (9) follows from smooth local Euler theory for this smooth periodic datum; it is **not** a claimed global estimate. Formula (7) is the exact leading near-family remainder, not a fitted stress. In particular, dropping its mixed physical pressure would give a *different* and generally non-solenoidal remainder. Setting \(\alpha=0\) or \(\beta=0\) restores one autonomous 2-D Euler sector by deleting the other; it is not a simultaneous pair of nonzero independently resolved Euler fluids.

For a divergence-free small preparation \(r\) on this torus, \(u_0+r\in H^s\), \(s>5/2\), the actual projected acceleration \(E(v)=-\mathbb P[(v\cdot\nabla)v]\) obeys the elementary, explicit \(L^2\) bound

\[
 \begin{split}
 \|Q E(u_0+r)-R\|_2
 &\le (|\alpha|+|\beta|)\|\nabla r\|_2
       +2(|\alpha|+|\beta|)\|r\|_2
       +\|r\|_\infty\|\nabla r\|_2.                           \tag{10}
 \end{split}
\]

Indeed \(\|u_0\|_\infty\le|\alpha|+|\beta|\), \(\|\nabla u_0\|_\infty\le2(|\alpha|+|\beta|)\), both \(Q\) and Leray \(\mathbb P\) contract on \(L^2\), and the three product terms are \((u_0\cdot\nabla)r,(r\cdot\nabla)u_0,(r\cdot\nabla)r\). Thus every perturbation whose right-hand side in (10) is smaller than (8) still has an instantaneously nonzero escape derivative. This is a neighborhood *obstruction*, not a nonlinear two-level closure estimate. Spatial lengths are normalized to unit wave number in (7)–(10).

## 2. Moving the candidate into spatially separated material layers does not rescue it

To remove the intersection in (2), take \(\chi\in C_c^\infty((-1,1))\), real and nonzero, and place the first genuine nonlinear sheet in a slab of the wall-free waveguide:

\[
 u_0(x,y,z)=\alpha\chi(z)(\sin x\cos y,-\cos x\sin y,0),
 \qquad \alpha\ne0.                                              \tag{11}
\]

Each frozen-\(z\) planar field by itself would be a stationary 2-D Euler flow with nonlinear pressure proportional to \(\chi(z)^2(\cos2x+\cos2y)\). The full flow cannot use those separate pressures: \(u_z=0\) requires \(p_z=0\), while the physical pressure from (1) has the unique decaying nonconstant Fourier component

\[
 p_0(x,y,z)=\frac{\rho\alpha^2}{4}(\cos2x+\cos2y)
      \int_{-1}^1e^{-2|z-s|}\chi(s)^2\,ds.                     \tag{12}
\]

This follows from \((\partial_z^2-4)(-e^{-2|z-s|}/4)=\delta(z-s)\), with the sign in (1), and is **not** a transverse pressure cutoff. Define \(I_+=\int_{-1}^1e^{2s}\chi(s)^2ds>0\). On the entire initially motionless exterior \(z>1\), (12) gives

\[
 \partial_tu_z\big|_0
  =-\rho^{-1}\partial_zp_0
  =\frac{\alpha^2 I_+}{2}e^{-2z}(\cos2x+\cos2y),\quad
 \|\partial_tu_z|_0\|_{L^2(\mathbb T^2\times(Z,\infty))}
  =\frac{\pi\alpha^2 I_+}{2}e^{-2Z}>0\quad (Z\ge1).          \tag{13}
\]

For a literal material cell, tag \(C_0=\mathbb T^2\times[-1,1]\) and set \(C_t=\eta_t(C_0)\) using the **one** Euler flow map \(\eta_t\). Its upper boundary starts with zero vertical velocity because \(\chi\) vanishes there, but at the point \((x,y,z)=(0,0,1)\) its vertical acceleration is \(\alpha^2 I_+e^{-2}>0\) by continuity of (13). Its displacement is \(\alpha^2 I_+e^{-2}t^2/2+o(t^2)\). Hence even the material-cell boundary acquires the ambient pressure response; fixed slab coordinates are not a separate material Euler action.

Smooth local evolution yields the same norm for \(u_z(t)/t\) in the limit \(t\downarrow0\) and an \(O(t^2)\) Taylor remainder in \(L^2\) on any such exterior region. The exterior acceleration is **not** a second fluid: it is the unique common 3-D fluid receiving pressure momentum. Adding a second nonoverlapping slab with its own nonlinear pressure gives another source in the same Poisson equation, not an independent 2-D pressure boundary. More generally, any horizontal-only Euler trajectory supported in a finite \(z\)-slab while \(u_z=0\) must have \(p_z=0\); since the horizontal velocity vanishes outside, the horizontal pressure gradient there is zero and so it must be zero on all \(z\). A slab whose 2-D Euler motion *requires* nonconstant pressure violates this necessary condition. This conditional statement does not exclude families with vertical velocity, overlapping support, special zero-pressure self-flows, or a dynamically resolved common exterior.

## 3. Which terms actually come from the one action

The only kinetic functional and pressure constraint are

\[
 H[u]=\frac{\rho}{2}\int|u|^2dx,
 \qquad \nabla\cdot u=0,
 \qquad u=u_1+u_2.                                             \tag{14}
\]

For the torus preparation \(u_1=\alpha a,u_2=\beta b\), it decomposes algebraically into the two self energies and the **fixed-coefficient** cross term \(H_\times=\rho\int u_1\cdot u_2\,dx\). Under a common divergence-free virtual displacement \(\xi\), the transported instantaneous vector fields have \(\delta u_i=[\xi,u_i]\), and integration by parts gives

\[
 \delta H_\times
  =-\rho\int (u_1\otimes u_2+u_2\otimes u_1):\nabla\xi\,dx,
 \qquad \nabla\cdot[\rho(u_1\otimes u_2+u_2\otimes u_1)]
       =\rho\alpha\beta C.                                    \tag{15}
\]
As an explicit nonzero variation witness, take the divergence-free virtual field \(\xi=R\) from (7): since \(R=-\alpha\beta\mathbb P C\), integration by parts in (15) gives \(\delta H_\times=\rho\alpha\beta\int R\cdot C\,dx=-\rho\|R\|_2^2\ne0\) for \(\alpha\beta\ne0\). One may multiply \(R\) by a time-scale constant when assigning physical length units to \(\xi\).

The **actual** single Euler constrained velocity variation is \(\delta u=\partial_t\xi+[u,\xi]\), with pressure enforcing solenoidality; the simultaneous *instantaneous pushforward* convention \([\xi,u_i]=-[u_i,\xi]\) used to display the stress in (15) has the opposite spatial-variation sign. Equation (15) is a common-displacement stress identity, **not** an assertion that the components themselves obey separately constrained Euler variations. Expanding the resulting unique Euler momentum flux gives \(\nabla\cdot[\rho(u_1\otimes u_2+u_2\otimes u_1)]=\rho\alpha\beta C\); projecting that flux with its uniquely determined Poisson pressure gives (5)–(7). The tensor in (15) has pressure units, \(\rho[\mathrm{velocity}]^2\), and there is no tunable interlevel force. On this particular torus \(H_\times=0\) by Fourier orthogonality, yet its *common-displacement first variation* and local mixed acceleration are nonzero: a vanishing integrated cross energy is not a decoupling proof. Total Euler energy is conserved on the torus and, with decay, on the waveguide; there is no justified separate positive-time sheet-energy exchange law, because the full solution immediately exits the proposed sheet state space. Independent arbitrary displacements \(\xi_1,\xi_2\), two independently prescribed pressure fields, or a multiplier of (15) would add a second action/constraint rather than derive the requested limit from (14).

## 4. Status of Dan's two-Euler criteria and actual continuation boundary

| Criterion | Observed answer for this candidate |
| --- | --- |
| One original unforced 3-D Euler action, no free coupling | **Yes**, (1), (14)–(15), but only for the *single* physical flow. |
| Two genuine nonlinear self-Euler transport-plus-pressure laws when individually isolated | **Yes**, the two isolated planar subalgebras, (3)–(4); each has arbitrary non-affine planar initial data, not merely the stationary witness. |
| Exact nontrivial invariant material-cell family for positive time with both nonzero | **No** for these intersecting columns, by (7)–(9); **no** for these pressure-bearing transverse slabs, by (12)–(13). Other geometries have not been excluded. |
| Autonomous coarse-center Euler and independent internally resolved Euler from this *same* family | **No**: columns do not define a finite material cell or an independently moving cell-center field; a slab has one material velocity and an ambient pressure tail. No relabeling of the two columns or ambient tail supplies the missing Euler field. |
| Derived coupling/stress and common pressure | **Yes** as a diagnostic (5)–(7), (12)–(15); **not** a closed interlevel law after the escape. |
| Quantified near-family error including full-pressure adverse check | **Yes**, initial exact \(L^2\) escape (8)–(10) and exterior acceleration (13), short-time Taylor (9); **no** controlled autonomous positive-time two-Euler remainder. |
| Particle, electron/neutrino, or all of issue #203 | **Not claimed.** |

Attempts [0001](../0001/material-balances.md) and [0009](../0009/retained-euler-memory.md) separately show why untagged pressure and unresolved initial state cannot be discarded. [0025](../0025/result.yaml) gives a distinct moving-fibre closure warning; [0165](../0165-integrable-helical-sector/derivation.md) shows a pressure-free *linear* shear sector that is not nonlinear invariant. The present computation changes the attempted construction: intersecting independent Euler planes were tested first; disjoint pressure-bearing planar layers were then tested under the actual Poisson boundary condition. Neither is promoted to a universal impossibility. An alternative with physical vertical circulation, an explicitly invariant common exterior and two demonstrated Euler degrees of freedom would require a new exact invariance proof and a neighboring-pressure/near-family test on that *same* solution family before any two-level claim.

## Reproducibility

The runnable [full-pressure oracle](verify_column_escape.py) and its [validation receipt](validation.md) check the mixed three-dimensional acceleration at nonsymmetric points, the exact Fourier escape norm, the failure when cross pressure is discarded, and the pressure-driven exterior slab acceleration. Those finite checks corroborate coefficients; the positive-time statement is the smooth local Euler/Taylor argument (9), not a numerical integration.

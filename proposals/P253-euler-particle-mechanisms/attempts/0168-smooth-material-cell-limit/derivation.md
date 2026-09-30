# P253/0168: the smooth material-cell continuum limit has no second energetic Euler field

## Domain and exact one-action split

Let `u` be a classical constant-density incompressible 3-D Euler solution on a periodic box `Ω` of volume `|Ω|`, on `[0,T]`. Its single volume-preserving flow map is `η_t`; the action has kinetic term `K=ρ∫Ω|∂tη|² da/2` and one incompressibility constraint. Partition the reference box into measurable material cells `C_A` of diameter at most `h`, with volumes `m_A/ρ=|C_A|`. Write `D_A(t)=η_t(C_A)`, `X_A=|C_A|⁻¹∫_{D_A}x dx`, `V_A=|C_A|⁻¹∫_{D_A}u dx`, and `c_A=u−V_A` on `D_A`. On the torus use local lifted coordinates for cells while `h exp L<` the injectivity radius, where `L(t)=∫₀ᵗ||∇u(s)||∞ ds`; the same inequalities hold on a bounded domain with compatible impermeable boundary.

The identity

\[
 K=\frac{\rho}{2}\sum_A\bigl(|C_A||V_A|^2+\int_{D_A}|c_A|^2dx\bigr)\tag{1}
\]

has **no freely assignable internal mass or inertia**. It does not by itself imply either field closes. For the moving physical coordinate `r=x−X_A(t)`, let `v_A(t,r)=u(t,X_A+r)−V_A`; incompressibility is `div_r v_A=0` on the moving cell. Reynolds transport and Euler give the **exact** equations

\[
 \dot X_A=V_A,\quad \dot V_A=-\frac{1}{\rho|C_A|}\int_{D_A}\nabla p\,dx
 =-\frac{1}{\rho|C_A|}\int_{\partial D_A}p n\,dS,\tag{2}
\]
\[
 \partial_t v_A+(v_A\cdot\nabla_r)v_A
 =-\rho^{-1}\{\nabla p(X_A+r)-\langle\nabla p\rangle_{D_A}\}
 =-\nabla_r q_A/\rho,\quad
 q_A=p(X_A+r)-\langle\nabla p\rangle_{D_A}\cdot r.\tag{3}
\]

The pressure on `∂D_A` is the trace of the **single whole-domain** Euler pressure, not an independent free-boundary condition. The single pressure satisfies `−Δp=ρ ∂ᵢuⱼ∂ⱼuᵢ`, with its mean fixed, so `q_A` is generally not a functional of `v_A` alone. Setting a cross/transfer term to zero while keeping `q_A` as if neighbors did not exist is an extra boundary assumption, not a consequence of (1).

## Uniform smooth-cell theorem and quantitative remainder

Put `M(t)=||∇u(t)||∞`, `L(t)=∫₀ᵗM(s)ds`. The flow's Lipschitz estimate gives `diam D_A(t)≤h exp L(t)`. For any `x∈D_A`,

\[
 |c_A(t,x)|=\left|\,|C_A|^{-1}\int_{D_A}[u(t,x)-u(t,y)]dy\right|
 \le h e^{L(t)}M(t).\tag{4}
\]

Define the coarse moving-cell average `U_h(t,x)=V_A(t)` on `D_A(t)` and internal stress `R_h=c_A⊗c_A`. Equations (1), (4) imply the explicit bounds

\[
 \|u-U_h\|_{L^2(Ω)}\le |Ω|^{1/2}h e^{L}M,\quad
 K_{\rm int}\le\frac{ρ|Ω|}{2}h^2e^{2L}M^2,\quad
 \|ρR_h\|_{L^1(Ω)}\leρ|Ω|h^2e^{2L}M^2.\tag{5}
\]

For every smooth divergence-free test vector `φ`,

\[
 |\langle\mathbb P\operatorname{Div}(ρR_h),φ\rangle|
 =\left|\int_ΩρR_h:\nabla φ\,dx\right|
 \leρ|Ω|h^2e^{2L}M^2\|\nabla φ\|_∞.\tag{6}
\]

If `M` and `L` are bounded uniformly in `h` on `[0,T]`, the unscaled internal velocity disappears in `L²`, its kinetic energy per physical volume is `O(h²)`, and its Reynolds force disappears as a distribution at `O(h²)`. The coarse field converges to the **same** original Euler field. This is a quantitative near-family bound for every smooth Euler flow with the stated uniform regularity, not a theorem that a piecewise-constant moving-cell average itself solves Euler at finite `h`: its interfaces move and its exact center force is (2), not automatically `P Div R_h`. The stress coefficient `ρ` follows from (1); an `O(1)` internal stress requires `h e^L M` not to vanish, so under a uniform bound on `L` it requires `M≳1/h` or a different microscopic scaling.

## What rescaling actually retains

Suppose one cell shrinks to a point `X(t)` and `u,p` are uniformly `C²` there. Put `r=h y` and `w_h(t,y)=v_A(t,h y)/h`. Taylor expansion using the centroid identities yields

\[
 w_h(t,y)=A(t)y+O(h),\quad A=\nabla u(t,X),\quad \operatorname{tr}A=0,\tag{7}
\]

and division of (3) by `h` gives, wherever the rescaled moving cell contains `y`,

\[
 \partial_t w_h+(w_h\cdot\nabla_y)w_h
 =-\rho^{-1}\frac{\nabla p(X+h y)-\langle\nabla p\rangle_{D_A}}{h}
 \longrightarrow -\rho^{-1}(\nabla^2p)(t,X)y.\tag{8}
\]

Equivalently `D_t A+A²=−∇²p/ρ` along the limiting center. Its local trace is fixed by the pressure Poisson equation, but the trace-free harmonic pressure Hessian is determined by the ambient solution, not `A` or the internal `w=Ay`. The rescaled affine field is the *velocity gradient of the first Euler field*, not a second independent finite-energy Euler fluid: replacing `v_A` with `w_h` in (1) requires the physical prefactor `h²`. One cannot retain an `O(1)` second kinetic energy by renaming `w_h`.

## Full 3-D Euler neighbor-pressure falsifier

P253/0001 supplies smooth compactly supported divergence-free initial velocity `u_0=f(|x−D|)n×(x−D)` with a flat radial bump and finite energy. Choose `D=d e_z`, `n=e_z`, `d>a+b`, and a resting material ball `B_b(0)` disjoint from the swirl. Compare this solution with the zero Euler solution. Both have exactly zero initial velocity on an open neighborhood of the ball: for every `h<b` their cell centers, `v_A`, `w_h` and local derivatives coincide at `t=0`. The full 3-D pressure of the swirl, however, is harmonic in the ball, with

\[
 p(x)=-\frac{ρJ}{3}\frac{3((x−D)\cdot n)^2-|x−D|^2}{|x−D|^5},
 \quad J=\int_0^a s^4f(s)^2ds>0,\tag{9}
\]
\[
 \ddot X(0)=2Jd^{-4}e_z,\qquad
 (\nabla^2p)(0)=ρJd^{-5}\operatorname{diag}(4,4,-8).\tag{10}
\]

In the zero solution both are zero. In the shrinking-cell limit the internal derivatives therefore differ by `−Jd⁻⁵ diag(4,4,−8)y` despite the **same** initial `w=0`; the center accelerations differ by `2J/d⁴`. The second pressure cannot be `q[w]` on this unrestricted Euler class without retaining ambient pressure as physical input. A global coarse field that also records the distant swirl *does* distinguish the two preparations; this witness refutes a local/tag-only closure, not every possible nonlocal two-field map. It is an actual Euler initial-acceleration test, using local classical existence, not an asserted all-time second trajectory.

## A materially different high-gradient survivor, and its precise failure

The assumption `M=O(1)` matters. On the `2π`-periodic three-torus, for integer `N` and fixed nonzero amplitude `a`, prepare

\[
 u_N=(\sin y+a\sin Ny,0,0),\quad p_N=\text{constant},\quad h=2π/N.\tag{11}
\]

Every `u_N` is a **global exact 3-D Euler solution**, since `div u_N=0` and `(u_N·∇)u_N=0`. Use material cells of side `h` with their initial `y` intervals each spanning one fast period; their `y` intervals and mean velocities stay unchanged under the exact shear flow, even though the cell shapes tilt. Let `U_h` be the material-cell velocity average and `c_N=u_N−U_h`. The fast shear has zero mean on each cell and mean square `a²/2`; the slow shear differs from its cell average by at most `h`. Cauchy–Schwarz therefore gives on every cell

\[
 \left|\langle |c_N|²\rangle_A-\frac{a²}{2}\right|
 \le \sqrt2\,|a|h+h²,\quad
 \left|\frac{K_{\rm int}}{ρ|Ω|}-\frac{a²}{4}\right|
 \le \frac{|a|h}{\sqrt2}+\frac{h²}{2}.\tag{12}
\]

This is an explicit invariant physical family with **nonvanishing** internal energy and macroscopic limit `U_h→(\sin y,0,0)`. It does not produce Dan's second Euler: each sector is a stationary collinear shear, its self-advection vanishes, its pressure is constant, and the exact Reynolds tensor is `R_h=R_{xx}(y)e_x⊗e_x`, hence `Div R_h=0` despite its nonzero energy. There is no interlevel transfer or derived nonzero force. In particular an `h`-dependent microstructure can evade the smooth bound (5) without meeting the independent nonlinear-pressure/coupling requirement. A next candidate must combine finite internal energy **and** nonzero feedback, while proving that feedback does not generate unretained modes or remote-pressure dependence.

Its natural 3-D neighborhood also lacks an `N`-uniform strong-norm estimate. Set `a=1`, `U_N(y)=sin y+sin Ny` and choose the two actual Euler preparations

\[
 u_N^\pm(t)=(U_N(y),0,\ \pm N^{-1}\cos[x-tU_N(y)]),\quad p_N^\pm=\text{constant}.\tag{13}
\]

They solve all nonlinear Euler equations exactly for all finite times: the `z`-component is transported by the `x` shear, does not self-advect, and produces no pressure source. Both have initial normalized `H¹` distance `1/N` from the invariant shear family at the displayed base point. Averaging first in `x`, then in `y`, for `N≥2` gives

\[
 |Ω|^{-1/2}\|\partial_y(u^\pm_{N,z}(t))\|_2
 =\frac{t}{2N}\sqrt{1+N²}\longrightarrow t/2>0
 \quad(t>0).\tag{14}
\]

Thus a proposed `H¹` near-family remainder bounded by `C(T)` times its initial `H¹` distance with `C(T)` independent of `N` is false even on a globally exact 3-D Euler trajectory. Its `L²` difference remains `O(1/N)`; the example does **not** refute a weaker topology or a scale-aware remainder. It shows that achieving finite internal energy through `M∼N` incurs a genuine perturbation/regularity cost, not just a bookkeeping change.

## Outcome and changed construction

The smooth-cell continuum route is fully decided: exact cell kinetic split and moving-frame Euler are valid, but a uniformly smooth refinement produces zero physical internal energy/stress, while the only nonzero rescaled limit is the first field's pressure-forced gradient. This route cannot supply the requested nontrivial autonomous `v`, independent `q[v]`, and nonzero action-derived `σ[U,v]`. The high-gradient survivor (11) proves that invariant finite-energy cell microstructure is possible, but its coupling force is identically zero. What remains open is a **different** `h`-dependent microstructured Euler family with finite internal energy, nonzero derived feedback, a proved invariant or controlled neighborhood, a homogenized center law and whole-space pressure closure. Fixed-width vortex cores do not satisfy the shrinking-cell theorem; their positive-time invariance and interaction still require independent proof. Neither a universal one-map impossibility theorem nor any electron/neutrino result follows.

## Reproducibility

Run [verify_smooth_cell_limit.py](verify_smooth_cell_limit.py) from the repository Python environment; [validation.md](validation.md) records its exact output and limits. The script samples the two exact global 3-D shear trajectories, their explicit transverse 3-D Euler perturbation, and the full-space initial pressure of the admissible compact swirl. It does not numerically integrate a generic 3-D Euler trajectory or prove the uniform theorem (4)–(6), which follows from the flow Lipschitz inequality.

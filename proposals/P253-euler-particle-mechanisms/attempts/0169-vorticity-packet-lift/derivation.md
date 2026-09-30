# P253/0169: one-action invariant vortex packets supply two Euler sectors, not a two-level cell

## A genuinely different physical family

Work on `R³`, constant density `ρ>0`, with classical finite-energy incompressible Euler on its smooth existence interval `[0,T]`. Take two smooth divergence-free initial vorticities `ω₁⁰,ω₂⁰` compactly supported in **disjoint** bounded material tags `C₁,C₂`. Let `Bω=−curl(Δ⁻¹ω)` be the unique decaying Biot–Savart velocity (`curl Bω=ω`, `div Bω=0`); write `uᵢ=Bωᵢ`. The physical initial velocity is `u=u₁+u₂`. This is a specified class of one-Euler initial data with two separately observable vorticity components: the tagged supports are physical initial data, not two arbitrary velocities assigned at the same point. For smooth Euler let `η_t` be its **single** volume-preserving flow map and set

\[
 ωᵢ(t,η_t(a))=Dη_t(a)ωᵢ⁰(a),\quad uᵢ(t)=Bωᵢ(t),\quad
 u=u₁+u₂,\quad \operatorname{supp}ωᵢ(t)\subsetη_t(Cᵢ).\tag{1}
\]

The Cauchy vorticity formula and uniqueness give the exact, positive-local-time equations

\[
 ∂_tωᵢ=\nabla\times[(u₁+u₂)\timesωᵢ],\quad
 ∂_tuᵢ=\mathbb P[(u₁+u₂)\timesωᵢ]
 =-\mathbb P[(uᵢ\cdot\nabla)uᵢ]+\mathbb P[uⱼ\timesωᵢ],\ j\ne i.\tag{2}
\]

Here `P` is the whole-space Leray projector; the second equality uses `P(uᵢ×curl uᵢ)=−P((uᵢ·∇)uᵢ)`. Supports remain disjoint under the same diffeomorphism for as long as it is smooth; this **is** an invariant compact material-vorticity-packet class with infinite-dimensional internal shapes. It is not an invariant class of compact *velocities*: Biot–Savart tails overlap. Each isolated sector, on setting the other initial packet exactly to zero, has a genuine nonlinear 3-D Euler self-transport law and its **own** uniquely decaying pressure `−Δpᵢ=ρ tr(Duᵢ²)`. With both present there is only one physical pressure, `p=p₁+p₂+p×`, with

\[
 -Δp_×=2ρ\operatorname{tr}(Du₁ Du₂).\tag{3}
\]

The cross terms in (2) are forced by the one full field, not freely fitted. Setting a cross force to zero without eliminating the other sector's physical pressure is generally **not** a solution of the full Euler initial-value problem.

## One action, a derived stress, and an important sector asymmetry

The sole kinetic action in physical variables has Hamiltonian

\[
 H=\frac{ρ}{2}\int|u₁+u₂|²dx=H₁+H₂+H_×,\quad
 H_×=ρ\int u₁\cdot u₂\,dx.\tag{4}
\]

The lift can also be checked at the original Hamiltonian bracket, rather than positing independent packet equations. Put `mᵢ=ρuᵢ`, `m=m₁+m₂`, with vector fields modulo gradients. In the convention `[ξ,ζ]=ξ·∇ζ−ζ·∇ξ`, the Euler Lie–Poisson bracket is `{F,G}(m)=∫m·[δF/δm,δG/δm]dx`. Use its direct sum **only on the two advected initial labels**. For observables depending solely on `m₁+m₂`, the two brackets add to exactly the original bracket; the sum map is Poisson. The pulled-back kinetic Hamiltonian `H(m₁,m₂)=||m₁+m₂||²₂/(2ρ)` has `δH/δm₁=δH/δm₂=u`. Both label momenta consequently have the **same** coadjoint advection by `u`; taking curls gives (2). The extra labels specify initial vorticity components; they do not introduce two independently selectable physical flow maps or a second kinetic action.

No added density, exchange constant, or second pressure appears. Under a common divergence-free infinitesimal material displacement `ξ`, the advected vorticity variation is `δωᵢ=curl(ξ×ωᵢ)`, hence `δuᵢ=P(ξ×ωᵢ)`. Since `uᵢ` is divergence free, integration by parts and the vector identity

\[
 \operatorname{Div}(u₁\otimes u₂+u₂\otimes u₁)
 =\nabla(u₁\cdot u₂)+ω₁\times u₂+ω₂\times u₁\tag{5}
\]

give the exact common-displacement variation

\[
 δH_×=ρ\int ξ\cdot(ω₁\times u₂+ω₂\times u₁)dx
 =-\int σ_×:\nabla ξ\,dx,
 \quad σ_×=ρ(u₁\otimes u₂+u₂\otimes u₁).\tag{6}
\]

`σ_×` has physical pressure units, and the coefficient is fixed by the original action. Pressure/constraint variations still use **one** physical Lagrange multiplier. The total projected cross acceleration follows by summing the two sector equations:

\[
 \mathbb P[u₂\timesω₁+u₁\timesω₂]
 =-\mathbb P\operatorname{Div}(u₁\otimes u₂+u₂\otimes u₁).\tag{7}
\]

Equation (7) is a real interacting, action-derived two-sector identity; individual packet `uᵢ` equations have **different** terms `P(uⱼ×ωᵢ)`, so assigning all of `−P Div σ×` to an alleged coarse center `U=u₁` would double-count or misassign momentum. The packet velocities are recoveries from vorticity labels, not the material-cell centroids `Xᵢ`. Their kinetic mixed term is not automatically the fluctuation stress of a homogenized center field. Thus (1)–(7) do **not** establish Dan's `∂tU+P(U·∇U)=−P Div σ[U,v]` with `U` a field of cell centers, nor an internally resolved `v` on fibers over those centers.

## Controlled off-class state, and whole-space pressure challenge

A third compact or decaying divergence-free vorticity label `ω₃⁰` with velocity `w=Bω₃` gives an **exact** three-label continuation by the same `η_t`. For `s>5/2` and any common smooth interval with `uᵢ,w∈H^s`, the two-sector equation (2) acquires the exact residual

\[
 rᵢ=\mathbb P[w\timesωᵢ],\qquad
 \|rᵢ(t)\|_{H^{s-1}}
 \le C_s\|w(t)\|_{H^s}\|uᵢ(t)\|_{H^s}.\tag{8}
\]

For the unitary Fourier convention `||f||²_{H^r}=∫(1+|k|²)^r|f̂(k)|²dk`, one sufficient **explicit** constant in (8) is

\[
 C_s=2^s\sqrt3(2π)^{-3/2}
 \left[\pi^{3/2}\frac{\Gamma(s-5/2)}{\Gamma(s-1)}\right]^{1/2},
 \qquad s>5/2.\tag{9}
\]

Indeed `〈k〉^{s−1}≤2^{s−2}(〈l〉^{s−1}+〈k−l〉^{s−1})`, Fourier Young and Cauchy–Schwarz give a scalar algebra constant `2^{s−1}(2π)⁻³ᐟ²(∫〈k〉^{-2(s−1)}dk)^{1/2}`. Bounding each of three vector cross components by two such scalar products yields (9); `P` has multiplier norm at most one and `||curl uᵢ||_{H^{s−1}}≤||uᵢ||_{H^s}`. The explicit `H^{s−1}` remainder controls a **supplied full-state complement** on any interval where `w(t)` remains controlled, not closure in observations `(Xᵢ, spinᵢ, shapeᵢ)` or uniform-in-cell-size stability. Smooth Euler local estimates supply a finite, data-dependent interval; no global 3-D regularity is claimed.

There is a more severe *local-observation* test. Use the smooth compact-velocity radial swirls from P253/0001: place packet one away from a resting tag `B_b(0)` and compare the one-packet initial field to the same field plus a second swirl centered at `d e_z`, disjoint from both the first velocity support and the tag. At `t=0`, `u₁·u₂=0` and `u₁⊗u₂+u₂⊗u₁=0` **pointwise**, so `H×=σ×=p×=0` initially. The tag has the same zero initial velocity in both preparations, but its centroid accelerations differ by `2J₂d⁻⁴e_z` because `p₂` is harmonic on the tag and has a nonzero exterior gradient. The action-derived cross stress vanishing initially therefore does **not** imply no remote pressure response: `u₂,t=P(u₂×ω₂)` immediately has a nonlocal velocity tail in the tag's region. The label split retains this global pressure in `u=u₁+u₂`; a center-only closure that drops it does not. This is an initial-acceleration statement, not an asserted all-time pair force.

## Reproducible action-variation control

For an explicit **periodic algebra control only** (not a compact packet), take `u₁=(sin y,0,0)` and `u₂(t)=(0,0,sin(x−t sin y))`. Their sum is an exact global 3-D Euler solution with physical pressure constant; `u₁` stays fixed, and `u₂,t+sin y ∂ₓu₂=0` is the nonzero cross term in (2). At `t=0`, `H×=0` because these velocities are pointwise orthogonal, but under the divergence-free common virtual displacement `ξ=(0,0,sin y cos x)` both sides of (6) equal `ρ|T³|/4`. The [runnable oracle](verify_label_stress.py) directly differentiates the exact full Euler trajectory at nonsymmetric points and independently integrates the stress and vorticity variation; its [receipt](validation.md) records the limits. This validates the **identity and sign**, not the compact-support persistence theorem (1), which follows analytically from the Cauchy formula.

## Exact decision

This is the **strongest positive two-Euler-sector reference** of the tested one-map constructions: compact material vorticity packets remain distinct for positive smooth time, each has an autonomous nonlinear Euler transport-plus-pressure law when physically isolated, their coupled laws and nonzero common stress are derived from the same action, and (8) gives a supplied-complement remainder. It is still a decomposition of **the first Euler vorticity**, not a second material-cell fluid with an autonomous homogenized center Euler law and an internal pressure insensitive to neighbors under a physically valid decoupling. A genuine next candidate must replace the label pair with a cell-fiber state carrying independent finite internal energy, prove a nontrivial center continuum under one flow, and keep whole-space pressure and the off-class complement. No electron/neutrino or universal impossibility claim is made.

# P253/0165: a finite-excess helical Euler sector and two physical escape tests

## Exact state and boundary

Work on the whole `R³` with constant density `ρ>0`, unforced incompressible Euler, and full Poisson pressure. The stationary background

\[
U_0=(\sin z,\cos z,0),\quad T=\partial_zU_0=(\cos z,-\sin z,0),\quad
\nabla\times U_0=U_0,\quad p_0=-\rho/2
\]

has infinite total energy; only physical **excess** relative to it may be finite. It is not a localized particle. For `Q∈R`, `σ>0`, put `θ=Q(σ²+|x|²)^(-1/2)` and `a_0=∂_zθ`. Choose a fixed smooth nonnegative even compact `h(z)` whose support includes an interval, and form the positive-definite planar Gram matrix and vector

\[
G=\int h(z)T(z)\otimes T(z)\,dz,\quad
A(x,y)=\int T(z)a_0(x,y,z)\,dz
       =\int U_0(z)\theta(x,y,z)\,dz
       =(0,2QK_0(\sqrt{σ²+x²+y²})).
\]

The last formula uses the standard modified Bessel integral; the integration by parts has no boundary term at either `z` end. Define

\[
a=a_0-h(z)T(z)\cdot G^{-1}A(x,y),\quad
b=-\int_{-\infty}^{z}T(s)\cdot\nabla_{xy}a(x,y,s)\,ds,\quad
v_Q=aT+b e_z.\tag{1}
\]

Since `∫T a dz=A-GG^{-1}A=0`, `b` vanishes at both vertical ends; `div v_Q=T·∇xy a+∂z b=0`. Because `U_0·T=0` and `U_0·e_z=0`, `U_0·v_Q=0` **pointwise**. The initial physical excess is therefore `E_ex(0)=ρ∫|v_Q|²/2<∞`: away from the exponentially localized transverse correction, `a_0=O(r^-2)` and oscillatory integration gives `b=O(r^-3)`. These are exact admissible 3D initial states, not established steady or shape-retaining solutions.

The scalar far label

\[
F_R(v)=-\frac{3}{4\pi}\int_{S²}R² n_z T(Rn)\cdot v(Rn)\,dΩ
\]

satisfies `F_R(v_Q)=Q R³/(σ²+R²)^(3/2)+o(1)` for a fixed `σ`, so its limit is `Q`. The compensator is transversely exponential and has compact vertical support. For admissible weighted perturbations `w=O(r^-β)` with `β>2`, its contribution is `O(R^(2-β))` and the same far label survives; a separate finite-time pressure/weighted-Sobolev closure, **not a carrier theorem**, is needed to propagate that class. In particular this label is not a local electric Gauss source, and its possible initial or finite-time persistence alone does not establish a charge current.

## Scale escape at fixed far label

Direct spherical integration gives

\[
\frac{\rho}{2}\int|a_0|²dx
=\frac{2\pi\rho Q²}{3σ}\int_0^{\pi/2}\sin⁴ t\,dt
=\frac{\rho\pi²Q²}{8σ}.\tag{2}
\]

For large `σ` the `K_0` compensator is exponentially small in `L²`, while the oscillatory-in-`z` vertical component has `||b||²₂=O(Q²σ^-3)` (integrate by parts in its fast `z` phase, then rescale the slowly varying source). Thus `E_ex(0)=ρπ²Q²/(8σ)+O(ρQ²σ^-3)+O(ρQ² e^-σ)→0` along actual solenoidal, orthogonal data with a fixed nonzero far `Q`. A `1/R` two-copy kinetic cross coefficient at separations `R≫σ` cannot select a finite core size: in the escape limit the core scale also grows. This rules out inferring an electron mass/size from **just this `Q` and the Euler excess**; it does not exclude a separately demonstrated invariant finite-scale branch or additional physical invariant.

## The actual two-copy tail energy is directional

Fix the two core widths and translate a second dress by `D=Rn`. For vertical translations take `R=2πm`, so both states are prepared on the **same** exactly translation-periodic `U₀`; arbitrary horizontal translations are also symmetries. The physical initial-excess cross term is `ρ∫v_{Q₁}(x)·v_{Q₂}(x-D)dx`. For the leading `a₀T` tails, `T·T=1`, and Fourier transformation of the Coulomb kernel gives, at distinct centers,

\[
\int_{\mathbb R^3}\partial_z\frac1{|x|}\,
  \partial_z\frac1{|x-D|}\,dx
=-\partial_{D_z}^2\mathcal F^{-1}\!\left[\frac{16\pi^2}{|k|^4}\right](D)
=\frac{2\pi}{R}(1-n_z^2).\tag{2a}
\]

The inverse transform is `-2πR` up to affine distributions, whose second derivatives vanish away from the origin. The Plummer regularization in (1) has the same leading `R⁻¹` coefficient: split rescaled space into small balls around the two centers and their complement; the singular derivatives are locally `L¹`, so the near-ball cross contributions vanish with ball radius, and the complement converges to the Coulomb integral. With fixed widths, the `a-a₀` compensator is integrable (compact in `z`, exponentially decaying transversely), giving an `O(R⁻²)` cross correction; the `b-b` cross of the bounded `O(r⁻³)` components is `O(R⁻³ log R)`. Horizontal and vertical components are pointwise orthogonal, so no `a-b` term occurs. Thus

\[
E_{\rm ex}(v_{Q_1}+v_{Q_2})-E_{\rm ex}(v_{Q_1})-E_{\rm ex}(v_{Q_2})
=\frac{2\pi\rho Q_1Q_2}{R}(1-n_z^2)+o(R^{-1}).\tag{2b}
\]

Horizontal separation has a `2πρQ₁Q₂/R` leading energy; vertical separation on the identical periodic medium has **zero** `R⁻¹` coefficient despite the same nonzero signed far labels. This is a reciprocal Euler kinetic overlap, but not an isotropic electric Coulomb interaction, a force law for moving cores, or evidence of an invariant carrier. The finite-core Fourier control in [verify_qtail_interaction.py](verify_qtail_interaction.py) probes (2a) without relying on the singular point-charge limit. An additional physical collective field/action would have to earn the missing isotropic current and reciprocal dynamics rather than rename this energy.

## A natural `Q`-to-Gauss map does not survive its physical neighborhood

A candidate phase inversion is `Θ_v(x,y,z)=∫_{-∞}^z T(s)·v(x,y,s) ds`, `E_v=-∇Θ_v`. For the centered `v_Q`, `a_0` is odd in `z`; with even `h`, the planar `G` is diagonal and `A=(0,A_y)`, so the compensator `a_c` is odd as well. Therefore `∫T·v_Q dz=∫a dz=0` by parity, and its reconstructed phase has no nonzero vertical-end cylinder. This parity identity is distinct from the Gram constraint `∫T a dz=0`, which enforces the divergence-free `b` boundary.

Let `d(s)=exp[-1/(1-s²)]` for `|s|<1`, zero otherwise, and prepare the arbitrarily small smooth compact solenoidal perturbation

\[
η_1=\nabla\times[d(x)d(y)d(z)e_z]
=(d(x)d'(y)d(z),-d'(x)d(y)d(z),0).\tag{3}
\]

It leaves the far `Q` unchanged. For `z>1`, however, `Θ_{v_Q+εη_1}-Θ_{v_Q}=ε Jd(x)d'(y)`, `J=∫cos(z)d(z)dz>0`. Its horizontal gradient is nonzero on a transverse open set for *every* `z>1`; hence `∫|E_{v_Q+εη_1}|²dx=∞` at every `ε≠0`. The field map fails on an arbitrarily small genuine physical preparation, not just on a formal representation.

One might insist on `C_η(x,y)=∫T·η dz=0` initially. This is **not an Euler-invariant rescue**. Set `g(z)=d''(z)+d(z)`, `f(x,y)=d(x)d(y)` and

\[
η_2=\nabla\times[f(x,y)g(z)e_z]=(f_y g,-f_x g,0).\tag{4}
\]

This state is also compact, smooth and solenoidal. Integration by parts gives `∫cos z g dz=0`; parity gives `∫sin z g dz=0`, hence `C_{η_2}=0` pointwise. The **exact full-pressure linearized Euler equation** about `U_0` has `-Δq_{η_2}/ρ=2T·∇η_{2z}=0`, with the decaying perturbative pressure `q_{η_2}=0`; thus `∂tη_2=-U_0·∇η_2`. At the origin, `f_xx=f_yy=-2e^-2`, `f_xy=0`, and consequently

\[
\partial_t C_{η_2}(0,0)
=-\int(f_{xx}\sin²z+f_{yy}\cos²z)g(z)dz
=2e^{-2}\int d(z)dz>0.\tag{5}
\]

For actual full Euler initial data `U_0+εη_2`, the derivative is `ε` times (5) plus `O(ε²)` from the quadratic stress. On `U_0+v_Q`, translate this fixed compact perturbation sufficiently far horizontally so that the fixed dress and its derivatives tend to zero on its support; the corresponding linear stress/pressure correction tends to the nonzero background value. This shows failure on the fixed signed-`Q` weighted neighborhood, not a change of substrate. **No universal obstruction to other nonlocal field maps is claimed**; a replacement needs a derived, finite-energy, Euler-invariant field and its reciprocal physical current.

## Exact finite-energy neutral linear continuum, not an oscillating species

There is also a positive exactly solvable full-pressure linear sector. For every compact smooth `ψ`, let `w_0=∇×(ψe_z)`, so `w_{0z}=0` and `div_xy w_0=0`. Its exact linearized Euler solution is

\[
w(t,x,y,z)=w_0(x-t\sin z,y-t\cos z,z),\qquad q_w=0.\tag{6}
\]

Indeed `w_z` stays zero, the linear pressure Poisson source vanishes, and horizontal divergence is carried by planar translation at each fixed `z`. `||w(t)||₂=||w_0||₂`, but this does **not** imply an internally persistent localized packet. For `ψ=d(x)d(y)d(z)`, the energy density factors a `d(z)²` vertical weight and has zero initial transverse centroid. Define `c=∫d²cos z/∫d²`. Its transverse centroid becomes `(0,tc)`, while its *centered* transverse second moment increases exactly by `t²(1-c²)>0`. The exact horizontal Fourier evolution is `ŵ(t,k,z)=e^{-it(k_x sin z+k_y cos z)}ŵ_0(k,z)` with `k·ŵ_0=0`. For each nonzero `k` this multiplication operator has the continuous frequency interval `[-|k|,|k|]`, not a selected two-mass flavor pair. The continuum and spreading describe this selected **linear** sector only; a nonlinear all-time theorem or exclusion of other protected modes does not follow.

This loss is observable without identifying a Fourier coefficient as a particle. Use the physical compact velocity receiver `D_V(t)=∫w_0(x-V_xt,y-V_yt,z)·w(t,x,y,z)dx` for a detector translated at a **fixed** planar velocity `V`. Since both the prepared and detector transverse supports lie in the disk of radius `√2`, they have disjoint support in each `z` slice whenever `t|U_0(z)-V|>2√2`. For the stationary detector `V=0`, `D_0(t)=0` for every `t>2√2` even though `||w(t)||₂` is constant. For the centroid-tracking detector `V=(0,c)`, `|U_0(z)-V|≥1-c>0`, so `D_V(t)=0` for every `t>2√2/(1-c)` (approximately `50.29737` for this bump). This is an exact physical linear Euler packet/receiver distinction, not proof of nonlinear decoherence or a weak-flavor detector.

## Verdict and next change of representation

Earned: exact 3D Euler-compatible finite-initial-excess signed far-tail states on a different (`e=0`) background, a fixed-label scale-escape diagnostic, an exact compact perturbation that destroys the naive finite-energy phase map, an exact compact zero-column state whose column ceases to vanish, and an exact physically localized linear continuum with growing shape width. Missing: full-state nonlinear shape retention, a selected inertia/scale, a finite-energy reciprocal electric mediator, an operational quantum/exchange bridge, a neutral chiral weak sector, both species and independent empirical predictions. Do not join the integrable `e=0` medium with the `e=1/10` indexed defects from P253/0164 or transfer their certificates across backgrounds. The next positive fork must change the carrier/field representation or earn a dynamical constraint whose conservation is proved on the same Euler action; a renamed far label or projected two-mode matrix is insufficient.

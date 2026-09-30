# Transferring the certified WKB ray to the full-space linearized Euler operator

Status: independently reviewed at the **linearized full-space L² semigroup-norm** scope after repairing a missing fixed-time microlocal estimate. This is an adverse result at the exact `e=3/10` stationary background, not a nonlinear defect instability, eigenmode, electron, neutrino or #203 solution. The source WKB ODE and its bounded interval certificate are in `floquet.md`; the actual domain here stays `R³`.

## Linearized physical state and the target

Let `U=U_(3/10)` be the bounded smooth periodic Beltrami Euler field of `derivation.md`. On divergence-free `L²(R³)` perturbations, let `P=I−∇Δ⁻¹div` be the whole-space Leray projector and

    L v = −P[(U·∇)v+(v·∇)U],       S(t)=exp(tL).           (1)

The exact linearized pressure is included in `P`. For smooth compact divergence-free data this flow exists, and the energy estimate `||S(t)||_(L²→L²)≤exp(t||∇U||_∞)` extends it to a strongly continuous L² evolution. The target is an **operator-norm** lower bound along integer multiples of the background relative period, not an `L²` point eigenfunction or a uniform finite-excess nonlinear neighborhood.

From the certified WKB trace `tr M<−7.32656` and exact `det M=1` at `e=3/10`, the monodromy of a transverse pressure-projected amplitude has a real eigenvalue `μ` with `|μ|>7` (since `7+1/7<7.32656`). The relative period in physical time is `T=2π/sqrt(1/2−(3/10)²)`. Pick a nonzero real transverse eigenpolarization `a_*` and tangent covector `ξ_*=(1,−1,0)` at the starting point `X_*=(0,π/2,−π/4)`. The background streamline reaches `X_*+m(−2π,2π,0)` at `mT`, and periodicity returns its covector and amplitude coefficients; the central geometric-optics amplitude is `μ^m a_*`.

## Fixed-time localized high-frequency realization

For **each fixed** positive integer `m`, and each small packet radius `σ`, choose a real smooth compact envelope `χ_σ` about `X_*`, and form complex preparatory data

    f_(N,σ)=P[χ_σ(x) a_* exp(i N ξ_*·(x−X_*))].       (2)

Its real part is real and divergence-free. The first-order Leray symbol gives `f_(N,σ)=χ_σ a_* exp(iNφ_0)+O_σ(N⁻¹)` in L², since `ξ_*·a_*=0`. Normalize the packet after projection; translation of its center at `mT` does not affect L² norm.

Transport the phase by the actual flow, `(∂t+U·∇)φ=0`, `φ(0,x)=ξ_*·(x−X_*)`. It remains smooth with `|∇φ|>0` on the material tube for every fixed `[0,mT]`. Choose a smooth transverse amplitude transported by the **full-pressure geometric-optics equation**

    D_t a = −A a + 2ξ(ξ·A a)/|ξ|²,       D_t ξ=−Aᵀξ,
    A=∇U,

Set `Q=I−P`. For every solenoidal `v`, `div[(U·∇)v]=div[(v·∇)U]` (expand both and use both divergence constraints), hence `Q[(U·∇)v]=Q(Av)`. Therefore (1) is **exactly** the restriction to solenoidal fields of

    L_ext f=−U·∇f+(I−2P)(Af).                            (3)

This extension has a C₀ evolution `E(t)` on full `L²`: incompressible transport is an isometry and `(I−2P)A` is bounded by `||A||_∞`. Its solenoidal subspace is invariant by the displayed divergence identity, and `E(t)|_div-free=S(t)`. This is the pressure term at the last responsible boundary; replacing it with `−Π_ξ A` loses the transport/projection commutator and violates transverse polarization.

Here is the fixed-horizon packet estimate needed for the PDE claim. Put `H=mT`, `X_t` for the volume-preserving background flow, `φ(t,x)=ξ_*·(X_(−t)(x)−X_*)`, and

    b(t,X_t y)=χ_σ(y) B_t(y,ξ_*)a_*,
    w_N(t,x)=b(t,x) exp(iNφ(t,x)),                       (4)

where `B_t` solves the displayed full-pressure amplitude equation at every initial `y` with covector `ξ(t,X_t y)=DX_t(y)^(−T)ξ_*`. The amplitude remains transverse: `D_t(ξ·b)=−ξ·Ab−ξ·Ab+2ξ·Ab=0`. On each fixed horizon the transported compact support, `φ`, `b`, and their required derivatives are uniformly bounded, while `|∇φ|` is bounded away from zero there. For such a compact smooth `h` and noncritical smooth `φ`, the whole-space order-zero Fourier multiplier `P` obeys

    ||P[h exp(iNφ)]−exp(iNφ) Π_(∇φ)h||₂ ≤ C_(H,σ)/N.    (5)

For completeness, localize the multiplier away from Fourier zero, write its semiclassical kernel as `(2π/N)^(-3)∫exp(iN[(x−y)·η+φ(y)])Π_η h(y) dy dη`, and expand about the stationary pair `y=x, η=∇φ(x)`; the leading symbol is `Π_(∇φ)`, and standard compact-support stationary-phase/remainder bounds control the next order in L². The low-frequency part is `O(N^(−K))` for every fixed `K` by nonstationary-phase integration by parts where `∇φ≠0`. The bounded smooth periodic coefficients of `U` and finite horizon ensure the constants in (5) are finite; no uniformity as `m→∞` is used. Independently, the fixed-time shortwave formula (75) and the unbounded-domain passage in §4.2 of Shvydkoy, *The essential spectrum of advective equations*, https://arxiv.org/pdf/math-ph/0412019, establish this same type of whole-space approximation. We use its **lower-bound** transfer, not its compact-torus essential-radius equality.

The principal symbol of `(I−2P)A` is `(I−2Π_ξ)A=−A+2ξ(ξ·A·)/|ξ|²`, precisely the amplitude equation. Thus the phase cancels the order-`N` transport term and the amplitude cancels the order-one term in `∂t w_N−L_ext w_N`; (5) gives

    sup_(0≤t≤H)||∂t w_N−L_ext w_N||₂≤C_(H,σ)/N,
    ||P w_N(0)−w_N(0)||₂≤C_σ/N,
    sup_(0≤t≤H)||S(t)P w_N(0)−w_N(t)||₂≤C'_(H,σ)/N.  (6)

The last line is Duhamel for `E(t)` with `||E(t)||≤exp(t||A||_∞)`. Applying (5) once more with `Π_ξ b=b` also gives `sup_t||P w_N(t)−w_N(t)||₂≤C_(H,σ)/N`. Every error here is for a **fixed** tube and finite horizon; the dependence on `σ,m` is allowed to diverge.

By smooth dependence on initial position, as `σ→0` the phase/polarization transport on the tube approaches the certified central ray, while the volume-preserving Euler flow preserves integration measure. Consequently

    ||S(mT)||_(L²→L²) ≥ lim_(σ→0) liminf_(N→∞)
      ||S(mT) Re f_(N,σ)||₂ / ||Re f_(N,σ)||₂ = |μ|^m.      (7)

For each fixed `m`, a sufficiently small finite `σ_m` then yields the conservative bound `||S(mT)||≥(1/2)|μ|^m`. Hence

    limsup_(t→∞) t⁻¹ log ||S(t)||_(L²→L²) ≥ (log |μ|)/T > (log 7)/T > 0.   (8)

The last inequality is a **linearized semigroup norm** instability of this selected periodic background at this parameter. It is not a growing fixed `L²` eigenfunction: the maximizing packet width and frequency may depend on `m`, and its center translates across cells. It also does not prove that the localized Gaussian zero-index seeds in `derivation.md` disperse, that small `H^s` perturbations fail nonlinear orbital stability, or that all values of `e` are unstable.

## Exposing case and design decision

A positive ray must reject a plausible wrong transfer: replacing `P` by a pointwise projector at finite wavelength, or fixing packet frequency/width independently of `m`, is not (7). The order is: choose **fixed** horizon `mT`, then a small but finite tube radius, then sufficiently high frequency, then take the operator-norm supremum; only afterward consider unbounded `m`. The initial real projected packets are divergence-free, finite-energy and L²-localized in the actual `R³` perturbation space (their Leray tails need not be compactly supported), not periodic Bloch vectors. They test background robustness but carry no electric or weak current.

The fixed-time transfer above rejects `e=3/10` as an all-mode **linear-L²-norm-stable** carrier. Two independent reviewers first required an explicit pressure-aware fixed-time whole-space packet bound, then rechecked the repaired extension, multiplier estimate, Duhamel argument, relative translation and limit order without further mathematical findings; neither executed a localized numerical Euler PDE evolution. Continue the positive architecture search (for example, test smaller `e` with different rays and an actual localized defect). Neither this adverse parameter nor the general medium family closes #203.

# Carrier probe: the neutral horizontal Euler leaf is not nonlinear

## Scope and exact pressure calculation

This is a discriminator for a proposed carrier, **not** an exclusion of all three-dimensional carriers or a construction of either particle. Keep the same whole-space, constant-density, unforced incompressible Euler law and the `e=0` exact background of 0165,
\[
 U_0(z)=(\sin z,\cos z,0),\quad T(z)=\partial_zU_0,
 \qquad \partial_tu+u\cdot\nabla u=-\rho^{-1}\nabla p,\quad \nabla\cdot u=0.
\]
The exactly transported, pressure-free *linear* horizontal sector in derivation (6) is tempting as a coherent neutral-carrier ansatz. At any time at which an actual smooth solution has initial state `u=U_0+\varepsilon w`, where `w=(w_x,w_y,0)\in C_c^\infty(\mathbb R^3)`, `\partial_xw_x+\partial_yw_y=0`, `w\ne0` and `\varepsilon\ne0`, the full pressure increment `q=p-p_0` is **exactly quadratic**, with the decaying whole-space Poisson normalization:
\[
 -\Delta q=\rho\varepsilon^2 F_w,\qquad
 F_w=\sum_{a,b\in\{x,y\}}\partial_a\partial_b(w_aw_b),\qquad
 q(X)=\frac{\rho\varepsilon^2}{4\pi}\int_{\mathbb R^3}\frac{F_w(Y)}{|X-Y|}\,dY. \tag{1}
\]
The `U_0` cross stress vanishes because `w_z=0` and `U_0` varies only in `z`; the background stress has zero double divergence. The source is compact and smooth, so the physical pressure increment tends to zero at spatial infinity. In particular, the exact vertical Euler acceleration at this initial instant is
\[
 \left.\partial_t(u-U_0)_z\right|_{t=0}=-\rho^{-1}\partial_zq.\tag{2}
\]
For **every** nonzero such `w`, this function is not identically zero. Otherwise `\partial_zq=0` everywhere, and decay as `|z|\to\infty` at fixed `(x,y)` makes `q=0`, hence `F_w=0`. But for every fixed `z` and either `a=x,y`, compact transverse support and two integrations by parts give the exact transverse virial identity
\[
 \int_{\mathbb R^2}x_a^2 F_w(x,y,z)\,dx\,dy
 =2\int_{\mathbb R^2}w_a(x,y,z)^2\,dx\,dy.\tag{3}
\]
Both components then vanish at every `z`, a contradiction. Thus the nonzero compact horizontal sector has **no nonlinear invariant leaf**, even infinitesimally in amplitude: its omitted vertical acceleration is order `\varepsilon^2`, not zero. Equation (1) is the full 3D pressure, not a slice-by-slice 2D projection. This conclusion is local in time and assumes the standard smooth local Euler evolution from the prepared state; it says nothing about all-time spreading after the vertical component appears.

For a concrete physical preparation take a nonconstant smooth compact radial `\chi(x^2+y^2)` and a nonzero smooth compact `g(z)`, and set
\[
 w=\nabla\times[\chi(x^2+y^2)g(z)e_z]
   =(\partial_y[\chi g],-\partial_x[\chi g],0).\tag{4}
\]
Writing `s=x^2+y^2`, its virial witness is explicit:
\[
 \int_{\mathbb R^2}x^2 F_w(x,y,z)\,dx\,dy
 =4\pi g(z)^2\int_0^\infty s|\chi'(s)|^2\,ds>0
\]
wherever `g(z)\ne0`, provided `\chi'` is not identically zero.
Then (3) is strictly positive for at least one `z` and one component, proving a nonzero directly observable initial vertical acceleration. Reversing its circulation, `w\mapsto-w`, is a contrary genuine compact solenoidal preparation: it reverses the linear velocity but leaves `F_w`, `q` and this quadratic vertical acceleration **unchanged**. More generally a separate arbitrarily small compact horizontal solenoidal bump does not rescue the leaf: (3) applies to every nonzero sum, without linearizing its nonlinear pressure. These are actual admissible whole-space 3D initial data, not a two-dimensional Euler equation or a manufactured neutral detector.

## Fixed-`Q` and energy decision

The obstruction does not silently turn the compact neutral state into a nonzero-`Q` carrier. For the 0165 Gram-compensated fixed-`Q` dress `v_{Q,\sigma}=a(r_\perp,z)T(z)+b e_z`, its coefficient `a` is radial in `(x,y)` (both `a_0` and the compensator are radial there). Prepare the same radial compact `w` of (4), centered on the dress. For either sign and any `\varepsilon`, `v_{Q,\sigma}+\varepsilon w` remains divergence-free, has **exactly the same far label** `Q` because `w` is compact, and satisfies the exact initial physical-excess identity
\[
 E_{\rm phys}(U_0+v_{Q,\sigma}+\varepsilon w)
 = E_{\rm phys}(U_0+v_{Q,\sigma})
   +\frac{\rho\varepsilon^2}{2}\|w\|_{L^2}^2.\tag{5}
\]
Indeed the planar angular integral of a radial scalar times either constant-in-angle vector `U_0(z)` or `aT(z)` dotted with the azimuthal field `w` is zero; `w_z=0`. Here `E_{\rm phys}(U_0+v)=\rho\int U_0\cdot v+\rho\|v\|_2^2/2` when the displayed cross integral exists, and `U_0\cdot v_{Q,\sigma}=0` pointwise. Taking `\varepsilon=\varepsilon(\sigma)\to0` along 0165's `\sigma\to\infty` family preserves nonzero `Q` while the total initial excess tends to zero, by its established `E_{\rm phys}(v_{Q,\sigma})\sim\rho\pi^2Q^2/(8\sigma)`. Thus adding this physical nonlinear-pressure probe neither manufactures a selected radius nor a positive mass gap from `Q` and Euler kinetic excess. At **nonzero fixed `Q`**, (1) is only the bump's quadratic *self*-pressure contribution: the dressed state also has baseline and mixed pressure and a pre-existing vertical component `b`; equations (2)–(3) must not be misquoted as a nonlinear no-carrier theorem for the dressed branch.

## Open positive proposition

A surviving same-action candidate would need a genuinely three-component, whole-`\mathbb R^3` nonlinear coherent evolution at fixed nonzero `Q`, with its **full** pressure, a dynamical invariant selecting finite width, and a response stable to the physical compact perturbations above. No such solution, selected scale, mass/charge assignment, reciprocal finite-energy field, neutrino flavor mechanism or joint quantum/relativistic observables follow here. Nothing from the separate `e=1/10` or `e=3/10` backgrounds is transferred to this integrable `e=0` background.

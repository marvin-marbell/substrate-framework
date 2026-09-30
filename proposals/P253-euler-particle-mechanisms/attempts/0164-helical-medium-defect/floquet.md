# A translation-periodic Euler streamline and a pressure-projected shortwave falsifier

Scope: this calculation concerns the exact stationary background `U_e` of `derivation.md`, **not** a localized defect orbit or a quantum/particle mode. Its certified periodic-ray WKB dynamics is finite-dimensional; `packet.md` separately transfers this adverse ray to a full-space linearized Euler L² operator-norm bound. Nonlinear full-state stability and the electron/neutrino construction remain separate. Choose the allowed parameter `e=3/10` and put `a=1/sqrt(2)`.

## Exact streamline and material variation

The curve `y=π/2−x`, `z=−π/4` is invariant under the actual velocity field: `U_z=e(cos y−sin x)=0`, `U_x+U_y=sin z+cos z+e(cos x−sin y)=0`, and `dx/dt=−a−e cos x<0`. Its coefficients repeat after one winding with relative period

    T_phys = ∫₀^{2π} dx/(a+e cos x) = 2π/sqrt(a²−e²).

On the actual whole-space `R³` solution, the parcel ends at a point translated by `(−2π,+2π,0)`; only its projection to a spatial torus closes. The background, velocity gradient and ray covector repeat under that physical spatial translation. Neither the Euler domain nor the perturbation class has been changed to a torus.

At that curve the full three-dimensional velocity gradient is

    A(x) = [[0,−e sin x,a], [−e sin x,0,a], [−e cos x,−e cos x,0]].

It preserves the tangent direction `(1,−1,0)` and the transverse span of `(1,1,0),(0,0,1)`. In transverse coordinates `(n,b)=((δx+δy)/2,δz)`, the **material** variational equation is `d(n,b)/dt=B(x)(n,b)` for

    B(x) = [[−e sin x,a], [−2e cos x,0]].

For `h(x)=a+e cos x` and `r=h n`, material variation also gives `r''+[2ae cos x/h²]r=0` with x as independent variable. Material separation alone is not linearized Euler instability.

## The full-pressure linearized Euler ray

The genuine divergence-free linearization is `(∂t+U_e·∇)v+(v·∇)U_e=−∇π`, with whole-space Hodge pressure. Its high-frequency ray obeys `dξ/dt=−Aᵀξ`; its amplitude satisfies `da_vec/dt=−A a_vec+2ξ(ξ·A a_vec)/|ξ|²`. Start with `ξ∥(1,−1,0)` and `a_vec` transverse. The invariant blocks make `ξ·A a_vec=0` on this particular streamline; the **pressure-projected** amplitude therefore obeys `da_vec/dt=−B(x)a_vec`, **not** the material `+B` equation. The covector is proportional to `1/h(x)` and returns after the fluid period.

Since `dx/dt=−h`, write the amplitude fundamental matrix against `x:0→−2π` as

    Y'(x)=C(x)Y(x),  Y(0)=I,  C(x)=B(x)/h(x).

Its exact determinant is one: `tr C=−e sin x/h=d(log h)/dx`, whose full-cycle integral vanishes. A trace below `−2` certifies real negative reciprocal WKB multipliers, one with modulus greater than one. It does **not** alone certify an unstable spectral eigenfunction or a nonlinear carrier failure.

## Interval product and explicit global error

`certify_helical_wkb.py` computes a `50`-decimal-interval enclosure of the `N=50,000`-step Cayley-midpoint product `P_j=(I−ΔC(x_j)/2)⁻¹(I+ΔC(x_j)/2)`, `x_j=−(j+1/2)2π/N`, `Δ=−2π/N`. It also checks the following deliberately conservative inequalities for the max-row-sum matrix norm, throughout the orbit:

    m=a−e>0.407, a<0.708, 2π<6.284,
    ||C||∞<L=2.48, ||C'||∞<K1=2.57, ||C''||∞<K2=6.4.

For `f=g/h`, the derivative bounds follow from `|h'|,|h''|≤e` and

    |f'|≤|g'|/m+|g|e/m²,
    |f''|≤|g''|/m+2|g'|e/m²+|g|e/m²+2|g|e²/m³.

The script checks both row sums against these constants; it evaluates the bounds using rational decimal endpoints and outward-rounded interval arithmetic. On a step of length `d=2π/N`, the midpoint quadrature error for `∫C` is at most `K2 d³/24`. The second Peano term differs from `C(x_j)²d²/2` by at most `L K1 d³/2`. Bounding both cubic-and-higher Peano/exponential remainders, and then the exponential-to-Cayley difference, yields

    ||Φ_j−P_j||∞ ≤ d³ [K2/24+L K1/2+exp(Ld)L³/3+L³] < 24 d³.

Indeed `r=Ld<0.001`, and comparing the power series after quadratic order bounds the exponential/Cayley difference by `L³d³`. Moreover `||Φ_j||∞≤exp(Ld)` and `||P_j||∞≤(1+r/2)/(1−r/2)≤exp((L+0.01)d)`. The telescoping product bound gives

    ||Φ_full−P_full||∞ < 24·6.284·(6.284/N)²·exp(2.49·6.284) < 15.01,
    |tr Φ_full−tr P_full| < 30.02.

This intentionally loose error is enough to distinguish `tr Φ_full<−2` if the interval product's upper trace is less than `−32.02`. The script asserts that strict inequality on the **enclosed exact ODE trace**, rather than interpreting a floating-point trajectory as proof. The certified result is limited to this WKB ODE and parameter, not all `e`, all rays, or an Euler PDE instability theorem.

On the corrected half-sum coordinates, the certificate exited successfully: the interval Cayley trace lies inside `[-37.076430097029027,−37.076430097029026]`, its product determinant encloses one, and the *analytically bounded* true WKB trace has upper bound `−7.3265648262685995<−2`. Two independent reviews first found the wrong transverse coordinate and an imprecise whole-space closure claim; both rechecked the corrected derivation and accepted this restricted WKB ODE result. The interval calculation depends on `mpmath.iv` enclosing its arithmetic and trigonometric values.

## One elliptic shortwave ray at a smaller exact-background parameter

The same invariant streamline and projected amplitude ODE exist at `e=1/10`; nothing in the exact geometry required `e=3/10`. Run the same interval Cayley/global-error argument at `N=10,000`. Rational outward-rounded bounds give `a−e>0.607`, `||C||∞<1.332`, `||C'||∞<0.39`, `||C''||∞<0.52`, local-step error `<3.5d³` and `exp((1.332+0.01)2π)<5000`. Hence the full-cycle matrix error is `<3.5·6.284·(6.284/10000)²·5000<0.044`, so its trace error is `<0.088`. The enclosed Cayley trace is near `0.25523657`, yielding an exact WKB ODE trace strictly between `0.168` and `0.343`. The determinant remains exactly one by the periodic `log h` identity above, so this **one** real two-dimensional Floquet system has nonreal unit-modulus multipliers at `e=1/10`.

This is a non-adverse **one-ray** parameter discriminator, **not** a full-pressure Euler stability theorem. It excludes exponential Floquet growth on this tangent-covector shortwave ray alone; it does not constrain other covectors, finite-frequency modes, transient amplification, the localized defect or nonlinear orbital stability. Two independent reviews checked the e=1/10 constants, determinant, coordinate convention, preserved e=3/10 bound and restricted interpretation without findings; neither executed the script. The **distinct transverse-covector** adverse full-pressure ray at this same parameter is now derived in `material.md`; these results are compatible.

## Physical decision boundary

The certified tangent-covector shortwave ray has a multiplier outside the unit circle at `e=3/10`; the separately reviewed fixed-time whole-space packet construction in `packet.md` transfers it to unbounded **linearized L² semigroup norm**, not an exact finite-wavelength eigenmode or a nonlinear defect verdict. The same packet construction also applies to the certified material-transverse **full-pressure** adverse ray at `e=1/10` in `material.md`. Both tested parameters fail the all-mode linear-L²-norm-stability requirement; neither rules out all `e`, establishes nonlinear Gaussian-defect instability, or supplies the joined #203 particle sectors. Do not rename the octupole/index as charge or either WKB mode as a neutrino.

# A same-Euler-state chiral packet pressure control

**Scope:** author-derived initial-time result on the `e=0` member of P253/0164. The exact algebra below and numerical quadratures test one proposed physical pressure readout; they do not establish a weak current, a persistent carrier, a quantum amplitude or a particle species. No independent revision-bound review of this supplement is claimed.

## Preparations and physical receiver

Use the actual whole-space Euler pressure constraint, with `rho=1` in the numerical examples, and the exact steady background and localized fields of [derivation.md](derivation.md):

    U_0=(sin z,cos z,0),   b=U_0+V_0,   u_s=b+s delta W,   s in {-1,+1}.

At `e=0`, the vorticity of `u_s` vanishes at the origin with nondegenerate index `-s`, for every `delta>0`. Set `lambda=25`, `delta=100`. For a chosen separation `R>1`, let `q=X-R e_z` and take the compact smooth bump `phi(q)=exp[-1/(1-|q|^2)]` on `|q|<1`, zero outside. Put

    T=curl(phi e_z),   P=curl T,   w_chi=T+chi P,   chi in {-1,+1},
    u_(s,chi)=b+s delta W+epsilon w_chi,   epsilon=0.1.

Every preparation is smooth, divergence-free and has finite **initial excess** relative to `U_0`; its local velocity/vorticity germ at the indexed origin equals that of `u_s`. The isolated packet has helicity `2 chi ||P||_2^2`: `T` is azimuthal, `P` poloidal, `T·P=0`, and integration by parts gives the two cross terms. Packet helicity is a preparation label, **not** a derived neutrino chirality or the conserved helicity of the entire prepared state.

For a receiver `X` outside packet support, the physically measured material acceleration of the fluid parcel initially at `X` is `a_k(u;X)=-partial_k p[u](X)`, not an assigned detector force. Write `G(X)=1/(4 pi |X|)`. The exact pressure source `-Delta p=partial_i partial_j(u_i u_j)` implies that the sign-odd **core-only** acceleration coefficient is

    A_W,k(X)=[a_k(u_+;X)-a_k(u_-;X)]/(2 delta)
            =-2 integral b_i(Y) W_j(Y) partial_(kij) G(X-Y) dY.       (1)

The core-free `U_0` has no `s`-odd coefficient. The pressure gauge cancels in acceleration. This is the same octupolar mechanical field as section 3 of the derivation, not a new Coulomb charge. At an exterior tracer on `X_R=(2R/3,R/3,R)` the Gaussian stress zeroth moment vanishes and its first moment gives the nonzero limit

    lim_(R->infinity) R^5 A_W,x(X_R)=3.6251230404704e-7.             (2)

For a genuinely packet-selective interaction, take the four-state `s*chi` coefficient of the **packet-induced** material acceleration, not merely the `s`-odd core field:

    Delta a_(s,chi)=a(u_(s,chi);X)-a(u_s;X),
    A_mix,k(X)=(1/4) sum_(s,chi) s chi Delta a_(s,chi)
              =-2 delta epsilon integral W_i(Y) P_j(Y)
                   partial_(kij) G(X-Y) dY.                         (3)

Equation (3) is exact: the quadratic packet self-stress, background/packet stress, `W*T` stress and pre-existing core field all cancel in the signed four-state sum. For an exterior tracer, the packet's local convective contribution is absent, so this is a difference of actual initial material accelerations on four full Euler states. It is not a time-integrated weak scattering amplitude.

## Contrasting tails and exposing controls

`W(Y)` is a Gaussian times a polynomial of degree at most four, while `P` is supported in `B_1(R e_z)`. For `R>=3`, the receiver `X_R` is outside that ball and its distance from it increases with `R`. Therefore (3) has the conservative bound

    |A_mix,x(X_R)| <= C |delta epsilon| (1+R^4) exp[-25(R-1)^2],   (4)

for a constant independent of `R`; (2) instead decays as `R^-5`. This is a discriminating adverse case: the indexed core has a long-range physical pressure response, but the *same instantaneous pressure action* does not provide an algebraically long-range interaction selective for the packet's `chi` on these states. At finite separation (3) is not generally zero.

For numerical reproduction, evaluate (1) with tensor-product Gauss–Hermite nodes divided by `sqrt(25)` and weights divided by `sqrt(25)`, using `b=U_0+exp(-25|Y|^2) P_0(Y)` and `W=exp(-25|Y|^2) P_w(Y)` from `helical_defect_octupole.py`. Contract the first stress moment with `partial_(kijℓ)G` for the independent asymptotic coefficient. Evaluate (3) on the packet ball by Gauss–Legendre radial/meridional nodes and 64 uniform azimuth nodes. With `f(t)=exp[-1/(1-t)]`, `t=|q|^2`, the packet fields are

    T=(2 q_y f', -2 q_x f', 0),
    P=(4 q_x q_z f'', 4 q_y q_z f'', -4 f'-4(q_x^2+q_y^2)f'').

The `x`-acceleration results below use the indicated **material-tracer** receiver. Rounded last digits are quadrature output, not certified enclosures:

| `R`, receiver | `A_W,x` per unit `delta` | `R^5 A_W,x` | `A_mix,x` at `delta epsilon=10` |
|---|---:|---:|---:|
| 1.25, `(2,1,R)` | `-3.77574316648106e-8` | — | `1.11981510047752e-6` |
| 1.5, `(2,1,R)` | — | — | `1.07388705742279e-8` |
| 2, `(2,1,R)` | — | — | `3.5908374e-17` |
| 3, `X_R` | `1.47052809503278e-9` | `3.57338327093e-7` | `1.19906e-50` |
| 6, `X_R` | `4.62820603448616e-11` | `3.59889301242e-7` | `3.86488e-282` |
| 12, `X_R` | `1.45154862642354e-12` | `3.61191747810e-7` | under double-precision range |
| 24, `X_R` | `4.54435317717273e-14` | `3.61849756730e-7` | not sampled |
| 48, `X_R` | `1.42140825083392e-15` | `3.62180462460e-7` | not sampled |

The core coefficient agrees at Gauss–Hermite orders 40/64/96 in the shown significant digits; independent first-moment contraction at orders 40/64/96 gives `3.6251230404704e-7`. The packet mixed coefficient at `R=1.25` converges from `1.1198441384e-7` to `1.1198151000e-7` **per unit `delta epsilon`** as Gauss–Legendre order increases from 32 to 80. Direct four-state *full quadratic-stress increments* at `R=1.25,1.5` give `1.11981510047757e-6,1.07388705742565e-8`, respectively, matching the separately contracted `W*P` source. At `R=2`, subtracting nearly equal full-state numbers loses relative precision; use (3) and bound (4), not a rounded numerical zero.

A second control places the packet on the `z` axis and reads its `x`-acceleration at the origin. Under `y -> -y`, `W` components have parities `(-,+,-)`, `P` components `(+,-,+)`, and the Green derivative `partial_(xij)G` has parity `(-1)^(number of y indices)`. Every `W_i P_j partial_(xij)G` term is odd, so its integral **vanishes exactly**, although the packet and source are nonzero. The nearby core-conditioned response there comes from `V_0`, is even under index reversal, and the far `U_0` response remains without any indexed core. This axial cancellation does not apply to arbitrary off-axis receivers; equation (3) and its near-field nonzero value provide the contrary geometry.

**Limit:** all claims concern initial-time mechanical pressure on the selected `e=0` family. Long-time deformation, reciprocal chiral scattering, a carrier selected by the Euler action, an operational weak current with flavor/mixing, a quantum/relativistic bridge, and both required species remain open under issue #203. A different physical current or later-time process is not excluded by this calculation.

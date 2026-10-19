# Same-N conditional Bloch scattering at the selected legal volume

This successor owns `noncompact_legal_volume_bloch.py` and does not change the
frozen `0efb1e4a` N2000 supplier or transplant its observed coefficient. Main
selected `N=5*2^96`, an even multiple of five above the scalar supplier's
reported sufficient threshold. Scalar phase, heavy-normal/legal-Lambda and
actual-ground admission retain their separate owners. This file constructs
the conditional fixed-body, all-harmonic scattering coefficient at this SAME N.
The served default is now the **far RIGHT** geometry and finite eta/alpha source
described below. The short-reference construction is retained only as an
explicit `--right-reference near` control, not an actual particle-source admission.

## Historical short-reference control and exact packet normalization

`N`, `V=N^3`, the center `R=(N//2,N//2,N//2)`, and the two mode IDs
`n=N//5,n+1` are Python arbitrary-precision integers. The 19-site charge source
is represented by `R+d`, with `d_j in {-1,0,1}`, `|d|_1<=2`, width .8 and
carrier [1.2,0,0]. Its reference is `X_R=R-(2,2,2)`, so `X-X_R=d+2` remains
bounded, not O(N). No giant integer coordinate is converted to binary64 or
int64. The compensated reference has `a_star=-theta/N`; the source phase
is exactly `exp[-i(theta/N+a_h).(d+2)]`, and full h0(a_h), scalar Esea and
completebar4 RIGHT are retained.

The two incoming modes are x-directed, the outgoing modes y-directed,
with z polarization, distinct IDs and normalized weights `4/sqrt17,1/sqrt17`.
They remain orthogonal global photon modes even if their binary64 momenta
coincide. Exact center phases are reduced as rational turns modulo one
before any numerical conversion; they are parity signs. The outgoing
center has the additional displacement [0,1,0], giving the retained
origin-stripped annihilation coefficient phase `exp(i k_y_abs)`.

Write `q0=2pi/5`, `delta=2pi/N`. Actual momenta are `q_r=q0+r*delta`, r=0,1,
actual native frequencies are `2c sin(q_r/2)`, and the charge recoil is
`(delta,-delta,0)`. Each photon gives 1/sqrtV, and each compact charge
source/detector Fourier factor gives 1/sqrtV. Consequently

`C_connected,N = mean_native_BZ_N f_N / V`,
`A_candidate,N = UE*C_connected,N`.

The original conservative N5-anchored UE, K and photon-speed c are kept
fixed. No tighter Mobile family or exact body/photon cone match is inferred.
All frequency intervals, integer IDs and rational phase reductions are
recorded in the runtime descriptor. An amplitude square is serialized with
Arb/decimal/log values, not falsely printed as zero after binary64 underflow.

## Intermediate represented kernel and finite-N debt

The program evaluates a moderate q^3 quadrature of a represented kernel
at q0, zero represented recoil and a trial principal square-root frame.
This intermediate value is neither C2000 nor a universal C_infinity.
Its relationship to the exact finite-N integrand is proved by a uniform
first-derivative bound along the whole splitting/recoil segment.

The underlying full 4x4 occupied/empty calculus, literal Wilson+odd current,
contact, both ordered histories and unfiltered Euclidean source/detector
preparation are the frozen supplier's `SparseBlochFiber`. The exact
zero-time cloud is subtracted, retaining the 11 reduced words
`Q(c-aPb-bPa)Q-QaX-XaQ-QbY-YbQ-XY+YX+XY0-Y0X`.

At the represented point the four mode-pair terms are bilinear in the two
coefficients and share their momentum/frequency kernel. They may be summed
algebraically for computation without renormalizing a merged physical mode.
An independently evaluated four-term positive control checks this identity.
The physical photon norm remains `16/17+1/17=1`; the false merged-mode norm
would be `25/17` and is retained as an adverse witness.

For the splitting variable s in [0,delta], let tau bound complete endpoint
norm, mu the full sea gap, and `omega_floor=.999*omega0`. Then

- all intermediate momentum shifts change by at most two coordinate units
  per s, giving `||dh/ds||<=4tau=:h_d`;
- `||dP/ds||,||dQ/ds||<=h_d/mu`;
- each frequency derivative is at most c;
- scalar current coefficients have relative derivative bounded by
  `rho=1+c/(2omega_floor)`, including the outgoing offset phase;
- with `j=6tau sum(weights)/sqrt(omega_floor)`, the integrated current has
  derivative at most `T*j*rho+T^2*j*(2h_d+c)`;
- infinite Euclidean denominators have lower bound `2mu+omega_floor`, and
  denominator derivatives at most `2h_d+c`. Final Y_T also pays its phase
  derivative `T*(2h_d+c)`;
- contact and both time-ordered histories pay the complete product rule,
  propagated through every reduced word by `word_error`;
- detector arguments change by at most s in each x/y coordinate, so the
  compact Fourier factor changes by at most `2*sum|eta|*s`.

These globally valid first derivatives, not a sampled finite difference,
give an explicit L and `|mean f_N - mean f_repr|<=L*delta`. There is no
unpaid Taylor remainder. The small source/reference separation prevents
an unresolved O(N) oscillation on the moderate grid.

## Actual native metric, periodization and arithmetic

The exact supplied a1 native full-ten metric certificate is consumed by
SHA. Strong convexity two bounds its actual native minimizer by
`(residual+continuum_gradient_error+M_gradient*F(rho_N))/2`, where
`rho_N=exp(-sigma*N)` and `F(rho)=2rho(3+rho^2)/(1-rho)^3`.
This cancellation-free formula equals the complete three-dimensional
trapezoid tail. The analytic metric envelope at auxiliary order2000 is a
monotone upper bound for every N>=2000; it is not the measured C2000 or a
substitution of its metric as G_N. The consumer explicitly verifies the
SAME-N metric distance fits that envelope and pays its frame variation.

The inherited beta-accretive strip and complete source/current/cloud word
majorant give the q^3 quadrature error and the actual N^3 periodization
error. Arb256 computes the actual large-N latter error using F(rho), without
subtracting near-identical ones. Tiny exponentials are bounded positively:
if exponent >=1024 log2, use `exp(-exponent)<=2^-1024`, never float zero.

The nominal kernel uses the frozen supplier's explicit polynomial exp/sinc,
entire time integrals and IEEE-binary64 forward-error proof. The separately
paid finite-N derivative debt covers the physically nonzero delta/recoil;
its loss in a represented binary64 momentum is not hidden as equality.

## SAME-N harmonic seam and coherent marginal

The full-source jet contour proof at T and exactly T0 separately pays both
preparation clouds and the unfiltered negative-source boundary. Every word
is two-photon homogeneous. Therefore its volume-rescaled seam prefactor is
obtained from the unit-volume field calculation, multiplied by the SAME-N
`exp[-kappa*(N//2-2)]`, also bounded by a positive Arb cap. This does not
substitute a small-grid seam or omit the source/reference endpoint.

Let z be the newly computed represented volume-rescaled coefficient.
Subtract frame, moderate quadrature, arithmetic, actual native
periodization, finite-N splitting/recoil and SAME-N T+T0 seam errors from
|z|. A positive result L_N proves, for every actual normalized harmonic
marginal nu, the coherent disk lower `|integral C_N(a)dnu|>=L_N/V`.
No harmonic marginal is invented and no incoherent `integral|C|^2` or
reference-identity detector replaces the declared observable. Scalar Esea
is retained with the supplied conservative harmonic energy-density moment;
that moment is not a simulation of the legal-volume neutral ground.

Every reported decimal lower is checked in Arb to lie strictly below its
ball's lower endpoint. The physical candidate lower is UE times the
coefficient lower. Its squared lower is only an unwhitened conditional
candidate quantity, not an actual Ei probability.

## Actual receiver and legal-family boundary

The actual output receiver is STATIC global `X_out G_out^-1/2`; input is
unwhitened `a†g B_eta Omega/sqrt2`, normalized by its actual n. Actual
amplitudes are `G_out^-1/2 m(T)/n`. Bare m_y0=0 but m_x0 generally differs
from one, so the leading y amplitude is
`UE*(m_y1-.5*g1_yx*m_x0)/n0`. The positive C_T-C_0 disk still requires
static-readout backpropagation/reference calibration, actual leading
Gram/cloud control and full flight comparison. No symmetric input
whitening or M-G reference is declared. No actual Ei probability is
claimed by this consumer.

The selected N and conditional fixed-family coefficient alone do not
establish the scalar/legal-Lambda hypotheses or that the initial UE/K
family survives the actual flight error at this volume. Any retuning must
be derived and recorded explicitly before an actual joined-family claim.

## Reproducible consumer

Use the project Python:

```
/home/administrator/substrate-framework/.venv/bin/python noncompact_legal_volume_bloch.py --static-receiver --metric-certificate /the/existing/native-metric-state-packet.json --screened-source-packet /the/existing/screened-gauss-far-RIGHT-Haar-q6-packet.json --quadrature 80 --time 6 --output /chosen/legal-volume-result.json
```

The program records exact input SHA, source hashes, runtime, selected N,
Arb precision, elapsed runtime, all errors, logarithmic physical scales and
the descriptor. It refuses an intervening source/input change or a
nonpositive conditional lower. No N^3 grid, giant coordinate array,
substituted actual ground or mode-merging normalization is constructed.

## Observed selected-volume result and adverse controls

The actual q80/T6 consumer at
`N=396140812571321687967719751680`, using the exact native-metric packet,
computed the new represented rescaled coefficient
`z=1.3748044614689838-.48638576195039795i`, not the N2000 coefficient.
The full real-segment derivative L is `11593159.864125708`, giving an
actual finite-N splitting/recoil error `1.8387898800238913e-22`.
Reference/frame/quadrature/arithmetic errors total `.031358755294493075`.
Actual native periodization is bounded by `1.2534568907202634e-297` and
the SAME-N T+T0 harmonic seam by `2.9031866755246054e-296`, both positive.

The coherent volume-rescaled lower is `1.4269479397127992`; the SAME-N
coefficient lower is `2.2954052177638709e-89`. With the unchanged
conservative UE family the unwhitened physical candidate amplitude lower
is `3.5416822984354607e-237`, whose squared conditional lower is
`1.2543513503051088e-473`. Their base10 logs are approximately
`-236.45079039916646` and `-472.9015807983329`.
The runtime's longer decimal-rational lower values, checked by Arb,
are authoritative; these shortened scientific displays are explanatory.

The independent four-term versus bilinear represented-kernel comparison
has error `2.4702109016023932e-15`. Exact mode IDs differ by one and their
Arb frequency intervals are strictly separated even though binary64
momenta coincide. The native photon weight norm is one, not `25/17`.
Exact source center+1 minus center is one, whereas converting both huge
coordinates to binary64 gives zero difference. Exact integer center phases
retain parities +1/-1. Recoil remains a strictly positive Arb interval.
The conditional square does underflow in binary64; its positive MP value
is serialized rather than using that zero as evidence.

The descriptor supplies outward decimal-rational upper bounds for the
physical coordinate field (with sqrtUE stripped), z-polarized one-body
and fullCAR currents/contact, and harmonic h first/second jets. In
particular J_fullCAR scales as sqrtV and contact_fullCAR stays O(1).
Absolute source sites are Python integer triples, while all derivative
moments use the small exact relative offsets d+2.

The two constructive successors are: selected actual STATIC receiver
calibration/leading Gram and complete flight comparison at this SAME N
(including any explicitly derived family retuning), or tighter
conditional arithmetic/current localization for more margin.
Neither this positive computation nor the scalar threshold by itself
is an actual joined receiver probability or a legal-family closure.

## STATIC receiver three-kernel extension

`--static-receiver` additionally constructs the three sparse kernels
specified by the actual receiver owner. The raw coefficient is R_xy(T),
NOT R_xy(T)-R_xy(0). The static output source is backpropagated with full
`exp(+ih0*T)`, so its bra contributes `exp(-ih0*T)` to the trace; the
outgoing annihilation photon coefficient also acquires `exp(-iomega_out*T)`.
Neither operation filters the negative output/source component.

The T0 off-Gram uses the physical forward receiver dF centered at R+[1,0,0],
not the incoming photon g centered at R. Its incoming coefficient has
the additional `exp(-ik_x)` phase. The outgoing receiver dR is centered at
R+[0,1,0], and its source has the prescribed small recoil. The full T0 cloud
`-XY0+Y0X` is retained without a positive-final chart.

The bare forward overlap is the photon overlap
`sum_r w_r^2 exp[iq_r-iomega_r*T]` times the static charged source overlap
`mean trace[M† Q exp(-ih0*T) M]*|eta_hat|^2/2`.
This quantity is computed; it is not replaced by one.

For each scalar kernel, the program separately pays frame, polynomial
arithmetic, moderate quadrature, actual native periodization, splitting/
recoil and SAME-N localized seam errors. Static free evolution adds its
time-Duhamel frame/momentum derivative, full spectral polynomial error,
and complex-strip norm `exp(b_im*T)` to the current/cloud bounds.
Its localized seam adds the complete weighted free-propagator defect.
The bare Q/free kernel uses its own projector/free bounds, not a
two-photon volume factor. Its photon-overlap derivative is also paid.

The coherent leading coefficient is a GLOBAL product:

`K = integral C_static dnu - .5*(integral g1_yx dnu)*(integral m_x0 dnu)`.

Here the first two kernels have the 1/V scaling and m_x0 is O(1).
Given volume-rescaled references zC,zG, bare reference zM and uniform
disk radii eC,eG,eM, the exact product disk radius is

`eK = eC + .5*(|zG|*eM+|zM|*eG+eG*eM)`.

This is NOT `integral(C-.5*g*m)dnu` and does not introduce input whitening.
Bare n0=1 follows only from the bare normalized photon and CT norm2
source; it is not a claim that the actual finite-coupling input norm n=1.
The real reference/cloud kernel identity still needs the owner's independent
fullCAR/literal bridge; actual normal/ground and full-flight comparison
remain separate even if the conditional GLOBAL product disk is positive.

### Observed conditional STATIC product disk

The actual same-N q80/T6 three-kernel consumer computed

- `V*C_static=-1.3484975276847986-.4822320222446739i`;
- `V*g1_yx=.006091890237088949-.000065482040776427i`;
- `m_x0=.32214299447719513-.8400707514936151i`.

Their GLOBAL reference is
`-1.3494512527926974-.4796626655495798i`, with complete product-disk radius
`.04959580410539358`. Hence every actual normalized marginal has conditional
leading volume-rescaled modulus at least `1.38256862653304`, or coefficient
`2.2240161332720434e-89`. Its fixed-family conditional physical amplitude
lower is `3.431532920500123e-237`, with square `1.1775418184476103e-473`.
Longer outward decimal rationals are in the runtime receipt.

The fullCAR current bound is `4.8205747603622685e44` and contact bound
`6.104671557565757`, both with sqrtUE stripped. The current's large sqrtV
factor is not hidden by the tiny one-body coordinate field
`1.2663783596433609e-44`; the actual flight comparison must pay it.
The descriptor includes actual forward-receiver mode phases, static
outgoing phases, exact source coordinates and recoil-source phase intervals.

A controlled two-fiber GLOBAL-product algebra smoke rejects fiberwise
whitening with a `.015` defect. It is an algebra control, not measured
ground data. The STATIC coefficient is neither the moving-detector
positive C_T-C_0 nor an already-admitted actual finite-coupling probability.
Independent fullCAR identity, neutral reference, input norm and full-flight/
legal-family comparison remain explicit transfer obligations.

## Selected far RIGHT and physical source normalization

Put `D=N/2`, source center `R0=(D,D,D)` and **RIGHT `XR=(0,D,D)`**.
The same centered 19-site source has `X-XR=(D+d_x,d_y,d_z)`.
The x-then-y-then-z coordinate tree, from X to RIGHT, has exact length
`ell_d=D+d_x+|d_y|+|d_z|`, maximum `L=D+2`, without crossing a periodic seam.
RIGHT is outside the 321-cell charged plateau; it is not moved into the
particle cloud to make a short-word estimate pass. All these values remain
arbitrary-precision integers in the descriptor.

Relative to the historical short-reference chart, the harmonic source
multiplier is the common unitary
`U_b=exp[-i(theta/N+a_h).b]`, `b=(D-2,-2,-2)`.
It cancels **pointwise** between bra and ket in the raw static C, off-Gram
and bare-forward kernels: h0(a_h), current operators and coexact photons
commute with this scalar multiplication at a fixed harmonic fibre.
The longitudinal scalar source phase is also common and commutes with the
coexact photon. The colour tree is SU4, not an additional coexact U1 string.
The outgoing source convention is explicitly
`eta_R(X)=eta_F(X) exp[i Delta.(X-XR)]`, `Delta=(2pi/N,-2pi/N,0)`.
The exact new recoil origin is the rational half turn `D/N=1/2`.
Consequently raw C and off-Gram both change sign, bare-forward does not, and
the **global** static receiver reference changes sign. This constant source
rephasing leaves E_i unchanged; the physical common source unitary U_b does
not assert unchanged full dynamics.

In particular `U_b† H_h U_b-H_h=(UE/V)(-2b.p+|b|²)`.
Write `e_H=<H_h>/V` for the admitted harmonic **energy density**.
The residual is bounded by `2|b| sqrt(UE e_H)+UE|b|²/V`, not by using
energy density as total energy in a different kinetic normalization.
This is retained in the descriptor, along with
`UE*((D+1)²+2)/V` for the harmonic source Dirichlet term and the separate
uniform Coulomb-pair bound `13UE/4`. There is no deletion of the cross term.
Normalized harmonic source jets obey
`HS <= 2 product_j axis_max_j^qj`, total order <=6,
with `axis_max=(D+1,1,1)`. The full colour source word and q<=6 Haar/source
derivatives use L, not the old three-cell bound. Previous short-R p527
flight admission is explicitly withdrawn for this source.

The native ideal-orbit screened source is not the primitive CT2I source.
For finite eta its q/2 core mark is multiplied by `c_eta^ell`,
`c_eta=tanh(2eta)`. Eta=18 pays `eta>=log(8L)/4` and gives a nonzero common
loss about .9789124. The program computes
`log(c_eta)=log1p[-2/(exp(4eta)+1)]` at Arb256; it does not round tanh to one.
With normalized eta and `s=sum_X |eta_X|² c_eta^(2ell_X)`,
the raw source CT Gram is `G_B=s/2`, the declared photon Gram is `G_X=s/4`
(the additional /sqrt2 in x is retained), and physical bare input
`n0=sqrt(s)/2`, not one. The serialized kernel matrix M has HS²4 and
the kernel uses `trace(M† R M)/2`. Thus the normalized creation matrix is
M/sqrt2 (HS²2), while the raw q/2 matrix is M/2 (HS²1).
These are explicit separate fields, not an inferred extra sqrt2.

A **fixed initial chart rescaling** `Btilde=Braw/sqrt(s/2)`,
`Xtilde=Xraw/sqrt(s/4)`, `qtilde=qraw/sqrt(s/4)` gives ideal core G0=I and
`Xtilde=a†Btilde`. Physical q/n and output-only Y are exactly unchanged.
It is not an intermediate reset or a post-readout orthogonalization.
Actual finite-alpha G0 and n deviations are paid, not set to one.
The common c_eta^D cancels only through these declared normalizations;
the residual normalized profile error is bounded by
`expm1(4|log c_eta|)`, about 4.31e-31.

The source packet supplies finite `alpha=2^371`, a uniform four-mark core
operator perturbation below 1.530e-5, and q<=6 normalized colour/source
power-of-two bounds with exponents [4,427,850,1274,1698,2122,2547].
The positive core lower admits physical bare n>1/sqrt8. Full coupled
G_B>=1/4 still needs the normal/source debt from its owner. Global smooth
raw q/2 HS<=1 implies normalized HS<=2 under this lower.
The three response kernels pay the finite-alpha perturbation through
Parseval and the reduced onebody operator norms: for normalized source
matrix error e, the bilinear difference is at most
`(2sqrt2 e+e²)*operator_norm`. The estimate does not multiply it by
full-CAR volume or an unnecessary source l1 squared.
The actual G_N ideal-orbit template and its admitted g_min=1/40, B²<=2
are the source authority; a fixed trial template amplified by huge alpha
is not substituted.

The canonical physical binary64 c and UE are converted to their exact
`Fraction.from_float` rationals before Arb arithmetic. Printed decimal
couplings are not a silently changed scalar family. The same-N body,
native Fourier normalization, static receiver and full contour debts above
remain in force, with new finite-eta/alpha and far-geometry debts added.
This proves a conditional leading static-receiver coefficient only; actual
prepared ground, far-R flight, actual Gram and legal-Lambda admission retain
their named supplier obligations. In particular no conditional square is
represented as an actual E_i probability.

### Global finite-alpha Gram debt and selected exact family

The ideal core G0=s/4 I is uniform in the actual harmonic fibre. Quaternion
products lie in the identity plus spin-bivector span, orthogonal under trace
to every Clifford component of h0(a_h). Consequently the bare trace uses
`Tr(Q_h M M†)=Tr(MM†)/2` at every native momentum, including the source's
harmonic phase shift and finite-eta profile. The physical finite-alpha
source is **not** presumed to have this exact metric.

Let delta be the supplied four-mark operator perturbation. Converting
4x4 operator norms to HS and four columns to the source synthesis map gives
`deltaS <=4delta`, `deltaX<=deltaS/sqrt2`, in GLOBAL L2(nu).
The bare Gram perturbation is at most
`deltaG=2n_core deltaX+deltaX²`. With actual sigma_min(X)>n_min=1/sqrt8,
the inverse-square-root Lipschitz constant is `1/(2n_min³)`.
The input normalized matrix error is at most
`(1+1/sqrt2)deltaS/n_min`, and the output-only whitened matrix error is at most
`deltaS/(sqrt2 n_min)+sqrt(s)/sqrt2*deltaG/(2n_min³)`.
No fibre-local n(a_h), resetting G0, or pre-evolution input whitening is used.
The full quartet g*m perturbation uses the norm of the entire g row and
the entire bare-forward vector, not merely |g00| times scalar m00 error.
For the common matrix error e, C and g use their reduced operator norms
times `2sqrt2 e+e²`; the bare-forward vector uses the same bilinear bound.
The extra global radius is
`extraC + .5*(2*cloud_operator_norm*extraM+extraG+extraG*extraM)`.
The earlier exploratory far lower 1.27338786, which did not include this
full global-quartet Gram accounting, is withdrawn and not served.

Main selected the original family's **TOTAL b4096**, not p527+4096:
`UE=Fraction.from_float(UE0)/2^4096`,
`c=1130202988249777/9007199254740992`, `K=c²/(2UE)`.
The descriptor retains these exact rational strings, scales the current and
field bounds by 2^-2048 and contact bounds by 2^-4096, and computes harmonic,
Coulomb and conditional physical amplitude/square in Arb, without binary64
underflow. This remains a candidate until the actual far-flight/Gram
suppliers admit it. Historical short-R and original-UE conditional values
are controls only. The stable topic packet records the final natural-exit
consumer, corrected global coefficient lower and exact source/input hashes.


# Physical neutral all120-field concentration, without a normal-mode cutoff

This result concerns the actual one-active-x-axis, two-cell microscopic pair,
not the three-axis periodic bulk. Keep NH=NL=4, all70 heavy and70 light primitive
Pauli configurations, all120 real Cartesian field coordinates, the complete
SU(4) link and both complete U(1) phase circles. Impose the genuine local colour
and electromagnetic Gauss constraints. No light band, field coordinate or
angular Fourier mode is projected out of the Hamiltonian.

Use the same positive-onsite nearest Wilson owner, c=m+r/a>|d|, quartic kappa>0,
alignment J>=0, electric UE4,UE1>0 and magnetic K>=0. The Hamiltonian is

```
H = sqrt(Lambda)/2 p_field² + UE4 C_colour + UE1/2 (p_A0²+p_A1²)
    -K cos(A0+A1)
    +Lambda [V_quartic + J W/8 - dGamma(hH)] + dGamma(hL).
```

The noncompact field potential is coercive quartic minus finite-dimensional
linear matter. The full positive kinetic operator and compact link/phase
factors give a bounded-below Friedrichs operator with compact resolvent.
Its nonempty physical neutral invariant subspace contains the explicit Gauss
packet below, hence has a ground. Degeneracy is allowed; the statement applies
to every physical neutral ground. It is not a numerical eigenstate observation.

## Exact global remainder, not an assumed selected well

`pair_global_field_bound.md` proves for arbitrary fields and colour link

```
V_quartic + J W/8 - sum(top4 hH) = -3R* + S,
S>=0,
R*=[(c^4+t^4)/kappa]^(1/3),
u*=c²/(kappa R*), v*=t²/(kappa R*),
u=(Tr Phi0,L²+Tr Phi0,R²)/8,
v=(Tr Phi1,L²+Tr Phi1,R²)/8.
```

S is an explicit sum, including `kappa[(u-u*)²+(v-v*)²]`, full Gram and endpoint
spread remainders, Ky Fan slack and J W/8. In particular,

```
S >= kappa/2 sum_(x,mu=2,3) (Tr Phi_mu,x²/4)².
```

This is a global all-coordinate bound, including the quartic directions with
zero quadratic curvature. No global field-minimum assumption is borrowed from
a finite optimization. The whole-heavy operator at NH4 is bounded by its exact
pointwise spectral bottom, so its field-dependent bound holds before any BO
projection, quotient measure or Berry approximation.

## The light sea cannot be dropped

Let cL,dL,tL be the actual light owner coefficients (equal to the heavy ones
in the executed shared law). The two-path electromagnetic factor has modulus
at most1 at every point of T². Operator norms and HS/4 normalization give

```
||hL|| <= 2sqrt(2)|cL|sqrt(u)+2|dL|sqrt(u)+2|tL|sqrt(v),
dGamma(hL)|_(NL4) >= -a_u sqrt(u)-a_v sqrt(v),
a_u=8sqrt(2)|cL|+8|dL|, a_v=8|tL|.
```

Since `sqrt(u)<=sqrt(u*)+sqrt(|u-u*|)`, scalar Young maximization yields

```
a sqrt(|z|) <= Lambda*kappa*z²/4
               + 3 a^(4/3)/(4 (Lambda*kappa)^(1/3)).
```

Put `C_L=a_u sqrt(u*)+a_v sqrt(v*)` and
`C_Y=3(a_u^(4/3)+a_v^(4/3))/(4 kappa^(1/3))`. The light term consumes at most
one-quarter of Lambda S. The complete Hamiltonian therefore satisfies

```
H >= -3 Lambda R* + 3 Lambda S/4 - C_L - K - C_Y Lambda^(-1/3).
```

Positive kinetic terms were discarded only for this lower bound; they remain
in the original operator and in the variational upper bound. This inequality
holds throughout all120 fields, all SU(4)/T² and the physical neutral subspace.
The actual light sea at the classical minimizer has energy−.7058986712, so
omitting it would produce an immediately false pointwise lower bound.

## An admissible shrinking full-domain physical Gauss packet

Reuse the source-owned `full_band_packet_bound.packet_bound` with the actual
attaining uniform frame, U=I and phase zero. Its **trial** support has field
radius R, relative colour-link exponential radius eta and phase radius tau.
The Hamiltonian domains remain R120, full SU(4) and full T²; a compactly
supported trial is not a truncation. Haar-average the normalized scalar seed
density over genuine local gauge actions and take its square root. The actual
normalized primitive section supplies colour/EM Gauss covariance. Jensen's
Fisher bound controls the averaged density without a guessed projection-norm
power. Positive primitive overlap and spectral gaps are uniformly bounded on
one fixed chart; the existing formulas provide all120-field, colour and phase
derivative constants and the exact normalizations in the proof (the separate
Weyl normalizer estimate is not used as a certificate).

Choose a fixed certified chart radius rho0<.1 and put R=eta=tau=rho<=rho0.
The source formulas have explicit uniform bounds

```
C(rho)<=-3R*+L rho,
T(rho)<=A0/rho²+A1,
B(rho)<=B0/rho²+B1,
E0 <= -3Lambda R* + L rho Lambda
                   +(A0/rho²+A1)sqrt(Lambda)+(B0/rho²+B1).
```

A0=2560 is the120-dimensional field Fisher coefficient. A1 is the section
field metric at rho0. B0 uses the colour/phase Fisher coefficients at rho0
(the SU(4) exponential Jacobian lower bound decreases monotonically with
radius); B1 uses the section colour/phase metric, K and the nonnegative light
upper cost there. The gaps/positive overlap improve and derivative bounds
decrease on shrinking support.

For completeness the computable slope is obtained directly from the actual
source formula. If `G=||Gram(anchor)||F`, `r=||fields(anchor)||HS/2`,
`D=field_dh+2 endpoint_HS`, and `A=2sqrt(2)+2||fields_R(anchor)||HS`, then

```
L=kappa/2 [4Gr+(2G+4r²)rho0+4r rho0²+rho0³]
  +J A²rho0/8+4D.
```

No sampled extremum enters any of these constants. Choose

```
alpha=(2 A0/L)^(1/3), rho(Lambda)=min(rho0,alpha Lambda^(-1/6)).
```

This supplies a normalized, legal physical trial for every Lambda>=1 and a
uniform excess divided by Lambda of order Lambda^(-1/6). Taking expectation
of the exact lower inequality in any physical neutral ground proves

```
<S> <= 4/3 [L rho+(A0/rho²+A1)/sqrt(Lambda)
             +(B0/rho²+B1+C_L+K)/Lambda+C_Y Lambda^(-4/3)] =: s(Lambda),
s(Lambda)=O(Lambda^(-1/6)) ->0.
```

Consequently average norm deviation squares are at most s/kappa; endpoint
norm-spread squares, off-diagonal Gram-square sum and the **sum of unused
normalized field fourth moments** are at most2s/kappa. Cauchy yields the sum
of their squared norms at mostsqrt(8s/kappa). Ky Fan slack is at most s; if
J>0, the full alignment W is at most8s/J. All are invariant moment bounds,
not a nonzero gauge-variant mean field (Elitzur's distinction remains intact).
Finite coefficients may be uninformative at smaller Lambda; do not replace
a large bound with a measured concentration claim.

## Execution and boundaries

The CLI checks source-owned packet C/T/B against the uniform formulas at
three shrinking supports, tests the complete matter/magnetic pointwise
inequality at24 all-coordinate nonuniform/noncentral/phase inputs and three
Lambda values, exercises the deleted-light-sea false bound, and executes the
actual4900-state primitive Gauss section at the attaining anchor. Its ground
moment rows are **floating evaluations of analytic bounds**, not interval
certificates or Monte Carlo/diagonalization observations of a quantum ground.

No charged external B4/bar4 spectral difference, moving normal-bundle BO,
physical mechanical spin, three-axis/thermodynamic minimum, compact photon
phase, radiative Compton or Lorentz closure is claimed. This removes the
assumed local well in this physical finite pair; it does not finish the campaign.

```
python proposals/P256-qed-backward/pair_full_field_concentration.py \
  --output proposals/P256-qed-backward/pair_full_field_concentration_receipt.json
```
